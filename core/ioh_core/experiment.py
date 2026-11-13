"""
Experiment runner for HpDesignBench optimization studies.

Applies an optimization algorithm to one or more IOH problem bundles,
records per-evaluation data with an IOH Analyzer logger, patches the
generated IOHprofiler metadata files to report the custom suite name
``HpDesignBench``, and optionally zips the result folders.

Adapted from: Max Dombrowski

Modifications by Elliot Melcer:

* one shared logger for all problems in a call
* docstrings
* configured sequential construction via :func:`run_configured_experiment`
"""
from __future__ import annotations

import os
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from ioh.iohcpp.logger import trigger as ltr
import ioh
import json

from core.ioh_core.algorithm_factory import create_algorithm
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import (
    SeedPolicy,
    get_algorithm_spec,
    normalize_algorithm_parameters,
)
from core.ioh_core.workflows.optimization_runs.planner import derive_random_seed


SUITE_NAME = "HpDesignBench"

AlgorithmProvider = Callable[[str, int], IOHAlgorithm]


def _zip_folders(folders: list[Path], zip_path: Path, base_dir: Path) -> None:
    """
    Zip a list of folders into a single archive.

    Parameters
    ----------
    folders : list[Path]
        Folders to include in the archive.
    zip_path : Path
        Destination path for the ``.zip`` file.
    base_dir : Path
        Root directory used to compute relative archive paths.
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in folders:
            for file in folder.rglob("*"):
                if file.is_file():
                    zf.write(file, arcname=file.relative_to(base_dir))


def _run_experiment_with_provider(
    problem_bundle: dict[str, dict],
    *,
    algorithm_provider: AlgorithmProvider,
    n_runs: int,
    name: str = "",
    zip_results: bool = True,
    delete_after_zip: bool = False,
    return_summary: bool = False,
    output_root: str | Path | None = None,
    algorithm_for_logger: Any | None = None,
):
    """
    Shared IOH logging loop driven by an algorithm provider.

    ``algorithm_for_logger``, when provided, supplies logger metadata immediately
    (legacy path). When omitted, the first real provider instance is used so no
    throwaway algorithm is constructed solely for metadata.
    """
    started_at = time.perf_counter()
    run_summaries: list[dict] = []

    if output_root is None:
        out_root = Path(os.getcwd()) / "logger_results"
    else:
        out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # snapshot existing JSONs so we only patch the ones written this call
    pre_existing = set(out_root.rglob("IOHprofiler_f*.json"))
    pre_existing = {p.resolve() for p in pre_existing}
    # snapshot existing folders so we only zip the ones written this call
    pre_existing_folders = {p.name for p in out_root.iterdir() if p.is_dir()}

    folder_name = name if name else "my_benchmarking_study"
    print("Logging root:", out_root.resolve(), "\n")

    from core.ioh_core.workflows.optimization_runs.io_logging import (
        patch_ioh_suite_metadata,
        staged_ioh_logger_directory,
    )

    # One logger for the whole call (all problems share it) # Modification by: Elliot Melcer
    logger = None
    first_algorithm = None
    with staged_ioh_logger_directory(out_root) as logger_root:
        if algorithm_for_logger is not None:
            # The list must outlive the run. IOH 0.3.22 does not retain
            # trigger objects, so an inline list is use-after-free.
            trigger_alw = [ltr.Each(1)]
            logger = ioh.logger.Analyzer(
                root=str(logger_root),
                folder_name=folder_name,
                algorithm_name=algorithm_for_logger.name,
                algorithm_info=getattr(
                    algorithm_for_logger,
                    "info",
                    "No additional information provided",
                ),
                store_positions=True,
                triggers=trigger_alw,
            )

        try:
            for problem_id, bundle in problem_bundle.items():
                problem = bundle["problem"]
                eval_context = bundle["ctx"]
                var_names = bundle["var_names"]
                constraint_names = bundle["active_constraint_names"]
                decode = bundle["decode"]

                eval_context.ensure_constraints(constraint_names)
                eval_context.ensure_params(var_names)

                if logger is None:
                    if n_runs < 1:
                        continue
                    first_algorithm = algorithm_provider(str(problem_id), 0)
                    # The list must outlive the run. IOH 0.3.22 does not retain
                    # trigger objects, so an inline list is use-after-free.
                    trigger_alw = [ltr.Each(1)]
                    logger = ioh.logger.Analyzer(
                        root=str(logger_root),
                        folder_name=folder_name,
                        algorithm_name=first_algorithm.name,
                        algorithm_info=getattr(
                            first_algorithm,
                            "info",
                            "No additional information provided",
                        ),
                        store_positions=True,
                        triggers=trigger_alw,
                    )

                # Each problem has its own EvalContext, so the watches must
                # be (re)bound before this problem is attached. See note below.
                logger.watch(eval_context, "t_eval_s")
                logger.watch(eval_context, "y")
                logger.watch(eval_context, "y_p")
                logger.watch(eval_context, "misses")
                logger.watch(eval_context, "hits")
                for constr in constraint_names:
                    logger.watch(eval_context, f"c__{constr}")
                for var in var_names:
                    logger.watch(eval_context, f"var__{var}")

                problem.attach_logger(logger)

                for run in range(n_runs):
                    if run == 0 and first_algorithm is not None:
                        algorithm = first_algorithm
                        first_algorithm = None
                    else:
                        algorithm = algorithm_provider(str(problem_id), int(run))
                    problem.reset()
                    eval_context.reset()
                    algorithm(problem)

                    best_y = problem.state.current_best.y
                    best_x = list(problem.state.current_best.x)
                    best_vars = {n: decode(best_x)[n] for n in var_names}
                    print(f"[{problem.meta_data.name}] run {run + 1}/{n_runs}  "
                          f"best y = {best_y:.3f}")
                    print(f"    x_idx = {best_x}")
                    print(f"    vars  = {best_vars}")
                    print("")
                    print("")

                    if return_summary:
                        run_summaries.append(
                            {
                                "problem_id": problem_id,
                                "problem_name": problem.meta_data.name,
                                "run": run + 1,
                                "hits": eval_context.hits,
                                "misses": eval_context.misses,
                                "optimizer_evaluations": getattr(
                                    algorithm, "n_optimizer_evaluations_", None
                                ),
                                "best_y": best_y,
                                "best_x": best_x,
                            }
                        )

                problem.detach_logger()
        finally:
            if logger is not None:
                logger.close()

    patch_ioh_suite_metadata(
        out_root, suite_name=SUITE_NAME, preexisting=pre_existing
    )

    # ---- zip the result folders created during this call ----
    if zip_results:
        new_folders = [
            p for p in out_root.iterdir()
            if p.is_dir() and p.name not in pre_existing_folders
        ]
        if new_folders:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = folder_name if name else "experiment"
            zip_path = out_root / f"{base}_{timestamp}.zip"
            _zip_folders(new_folders, zip_path, out_root)
            print(f"\nZipped {len(new_folders)} folder(s) to: {zip_path}")

            if delete_after_zip:
                for folder in new_folders:
                    shutil.rmtree(folder, ignore_errors=True)
                print("Removed source folders after zipping.")
        else:
            print("\nNo new folders to zip.")

    if return_summary:
        return {
            "wall_time_s": time.perf_counter() - started_at,
            "output_root": str(out_root.resolve()),
            "runs": run_summaries,
        }


def run_experiment(
    problem_bundle: dict[str, dict],
    algorithm,
    n_runs: int,
    name: str = "",
    zip_results: bool = True,
    delete_after_zip: bool = False,
    return_summary: bool = False,
    *,
    output_root: str | Path | None = None,
):
    """
    Run an optimization algorithm on a set of IOH problems and log results.

    For each problem in ``problem_bundle`` the algorithm is called
    ``n_runs`` times. All runs share a single IOH Analyzer logger.
    After all problems are processed, the IOHprofiler JSON metadata files
    are patched to report ``HpDesignBench`` as the suite name, and the
    new result folders are optionally zipped.

    Parameters
    ----------
    problem_bundle : dict[str, dict]
        Mapping of problem IDs to problem bundle dicts as returned by
        :func:`build_problems_for_slab`. Each bundle must contain:

        - ``"problem"`` — IOH problem instance.
        - ``"ctx"`` — :class:`EvalContext` instance.
        - ``"var_names"`` — list of optimization variable names.
        - ``"active_constraint_names"`` — list of active constraint names.
        - ``"decode"`` — callable ``decode(x_idx) -> dict``.

    algorithm : callable
        Optimization algorithm object with a ``name`` attribute (and
        optionally an ``info`` attribute). Called as ``algorithm(problem)``
        for each run. The same object is reused for every problem/run.
    n_runs : int
        Number of independent optimization runs per problem.
    name : str, optional
        Label for the logger folder and zip file. Defaults to
        ``"my_benchmarking_study"`` if empty.
    zip_results : bool, optional
        If ``True``, zip all newly created result folders after the
        experiment. Default is ``True``.
    delete_after_zip : bool, optional
        If ``True``, remove the source folders after zipping. Only
        effective when ``zip_results=True``. Default is ``False``.
    return_summary : bool, optional
        If ``True``, return a dict with wall-clock timing, resolved
        ``output_root``, and per-run evaluation counters from each
        problem's :class:`EvalContext`. Default is ``False``.
    output_root : str or Path or None, optional
        Directory for IOH logger output. When omitted, uses
        ``<cwd>/logger_results`` (invocation-CWD relative). When given,
        the path is resolved once here; callers/workers should reuse that
        absolute path rather than re-deriving from their own CWD.
    """

    def provider(_problem_id: str, _repetition: int) -> IOHAlgorithm:
        return algorithm

    return _run_experiment_with_provider(
        problem_bundle,
        algorithm_provider=provider,
        n_runs=n_runs,
        name=name,
        zip_results=zip_results,
        delete_after_zip=delete_after_zip,
        return_summary=return_summary,
        output_root=output_root,
        algorithm_for_logger=algorithm,
    )


def run_configured_experiment(
    problem_bundle: dict[str, dict],
    *,
    algorithm_name: str,
    algorithm_parameters: Mapping[str, Any],
    repetitions: int,
    seed_base: int,
    slab_type: str,
    name: str = "",
    zip_results: bool = True,
    delete_after_zip: bool = False,
    return_summary: bool = False,
    output_root: str | Path | None = None,
):
    """
    Run a registry algorithm with planner-compatible seed derivation.

    Constructs a fresh algorithm instance for every problem/repetition pair via
    :func:`create_algorithm`. Stochastic algorithms receive the same
    SHA-256-derived seeds as the parallel planner; deterministic algorithms
    receive ``None``.

    Parameters
    ----------
    problem_bundle : dict[str, dict]
        Mapping of problem IDs to problem bundles.
    algorithm_name : str
        Registry algorithm token.
    algorithm_parameters : mapping
        Declared hyperparameters including ``max_evaluations``.
    repetitions : int
        Number of independent runs per problem (>= 1).
    seed_base : int
        Non-negative experiment seed base.
    slab_type : str
        Slab package name used in seed derivation.
    name : str, optional
        Logger folder / zip label.
    zip_results : bool, optional
        Zip newly created result folders.
    delete_after_zip : bool, optional
        Remove folders after zipping.
    return_summary : bool, optional
        Return wall-clock and per-run counters.
    output_root : str or Path or None, optional
        Logger output directory.

    Returns
    -------
    dict or None
        Summary mapping when ``return_summary`` is ``True``.

    Raises
    ------
    ValueError
        If ``repetitions``, ``seed_base``, or ``slab_type`` are invalid.
    RegistryError
        If algorithm parameters fail registry validation.
    """
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions < 1
    ):
        raise ValueError(
            f"repetitions must be an int >= 1, got {repetitions!r}"
        )
    if (
        not isinstance(seed_base, int)
        or isinstance(seed_base, bool)
        or seed_base < 0
    ):
        raise ValueError(
            f"seed_base must be a non-negative int, got {seed_base!r}"
        )
    if not isinstance(slab_type, str) or not slab_type.strip():
        raise ValueError(
            f"slab_type must be a non-empty string, got {slab_type!r}"
        )

    normalized = dict(
        normalize_algorithm_parameters(algorithm_name, algorithm_parameters)
    )
    spec = get_algorithm_spec(algorithm_name)

    def provider(problem_id: str, repetition: int) -> IOHAlgorithm:
        if spec.seed_policy is SeedPolicy.REQUIRED:
            random_seed = derive_random_seed(
                seed_base=seed_base,
                slab_type=slab_type,
                problem_id=str(problem_id),
                algorithm_name=algorithm_name,
                repetition=int(repetition),
            )
        else:
            random_seed = None
        return create_algorithm(
            algorithm_name,
            normalized,
            random_seed=random_seed,
        )

    return _run_experiment_with_provider(
        problem_bundle,
        algorithm_provider=provider,
        n_runs=repetitions,
        name=name,
        zip_results=zip_results,
        delete_after_zip=delete_after_zip,
        return_summary=return_summary,
        output_root=output_root,
        algorithm_for_logger=None,
    )
