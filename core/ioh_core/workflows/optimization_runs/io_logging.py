"""
Isolated single-run IOH logging for optimisation workers.

Provides :func:`run_logged_optimization`, which mirrors the watched fields
and suite-metadata patching of the legacy :func:`run_experiment` runner
while executing exactly one problem and one algorithm call into a private
output directory.

IOHexperimenter's C++ Analyzer can fail silently on Windows when the logger
root path contains non-ASCII characters (e.g. ``Universität``). This module
stages such runs through an ASCII temporary directory and copies results into
the requested destination.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import ioh
from ioh.iohcpp.logger import trigger as ltr

from core.ioh_core.algorithms.common import (
    RunController,
    attach_eval_context,
    detach_eval_context,
)
from core.ioh_core.experiment import SUITE_NAME

_TERMINATION_MODES = frozenset({"fixed_budget", "fixed_target"})
_TARGET_REFERENCES = frozenset({"all"})
_STOP_REASONS = frozenset(
    {"target_hit", "budget_exhausted", "algorithm_terminated"}
)
_THRESHOLD_RTOL = 1e-12


def _validated_algorithm_diagnostics(value: Any) -> dict[str, Any]:
    """Return a finite JSON-safe diagnostics mapping or raise ``ValueError``."""
    if not isinstance(value, Mapping):
        raise ValueError("algorithm summary_diagnostics() must return a mapping")

    def validate(item: Any, path: str) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(
                    f"algorithm diagnostic {path} must be finite, got {item!r}"
                )
            return item
        if isinstance(item, list):
            return [
                validate(child, f"{path}[{index}]")
                for index, child in enumerate(item)
            ]
        if isinstance(item, Mapping):
            output: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(
                        f"algorithm diagnostic {path} keys must be "
                        "non-empty strings"
                    )
                output[key] = validate(child, f"{path}.{key}")
            return output
        raise ValueError(
            f"algorithm diagnostic {path} is not JSON-compatible: "
            f"{type(item).__name__}"
        )

    return validate(value, "algorithm_diagnostics")


def path_has_non_ascii(path: Path | str) -> bool:
    """
    Return True if the path string contains non-ASCII characters.

    Parameters
    ----------
    path : Path or str
        Filesystem path (resolved by the caller when needed).

    Returns
    -------
    bool
        ``True`` when any character has ``ord(ch) > 127``.
    """
    return any(ord(ch) > 127 for ch in str(path))


def _copy_tree_contents(source: Path, destination: Path) -> None:
    """
    Copy all entries from ``source`` into ``destination``.

    Parameters
    ----------
    source : Path
        Directory whose children are copied.
    destination : Path
        Existing destination directory.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


@contextmanager
def staged_ioh_logger_directory(destination: Path | str) -> Iterator[Path]:
    """
    Yield an IOH Analyzer root that is safe on Windows non-ASCII paths.

    When ``destination`` (resolved) contains non-ASCII characters, yields a
    temporary ASCII directory and copies its contents into ``destination`` when
    the context exits. Otherwise yields ``destination`` unchanged.

    Parameters
    ----------
    destination : Path or str
        Intended durable logger output directory.

    Yields
    ------
    Path
        Directory to pass as ``ioh.logger.Analyzer(..., root=...)``.
    """
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not path_has_non_ascii(destination):
        yield destination
        return

    staging = Path(tempfile.mkdtemp(prefix="hdb_ioh_"))
    try:
        yield staging
    finally:
        try:
            if staging.exists() and any(staging.iterdir()):
                _copy_tree_contents(staging, destination)
        except Exception as exc:
            print(
                f"[warn] could not copy staged IOH logs to {destination}: {exc}"
            )
        shutil.rmtree(staging, ignore_errors=True)


def patch_ioh_suite_metadata(
    root: Path,
    *,
    suite_name: str = SUITE_NAME,
    preexisting: set[Path] | None = None,
) -> list[Path]:
    """
    Patch ``suite`` in newly written ``IOHprofiler_f*.json`` files.

    Parameters
    ----------
    root : Path
        Directory tree to scan for IOHprofiler JSON metadata.
    suite_name : str, optional
        Suite name to write, default :data:`SUITE_NAME`.
    preexisting : set of Path or None, optional
        Absolute paths that already existed before the run and must not be
        patched. ``None`` means patch every matching file under ``root``.

    Returns
    -------
    list of Path
        JSON files that were successfully patched.
    """
    root = Path(root)
    preexisting = preexisting or set()
    patched: list[Path] = []
    for jf in root.rglob("IOHprofiler_f*.json"):
        resolved = jf.resolve()
        if resolved in preexisting:
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            data["suite"] = suite_name
            jf.write_text(json.dumps(data, indent=2), encoding="utf-8")
            patched.append(jf)
        except Exception as exc:
            print(f"[warn] could not patch suite name in {jf.name}: {exc}")
    return patched


def resolve_algorithm_max_evaluations(
    algorithm: object,
    max_evaluations: int | None = None,
) -> int:
    """
    Resolve the hard evaluation budget for one algorithm call.

    Parameters
    ----------
    algorithm : object
        Constructed algorithm exposing ``max_evaluations`` or ``evaluations``.
    max_evaluations : int or None, optional
        Explicit override from the sealed task payload.

    Returns
    -------
    int
        Positive evaluation budget.

    Raises
    ------
    ValueError
        If no positive integer budget can be resolved.
    """
    if max_evaluations is not None:
        if (
            not isinstance(max_evaluations, int)
            or isinstance(max_evaluations, bool)
            or max_evaluations < 1
        ):
            raise ValueError(
                f"max_evaluations must be an int >= 1, got {max_evaluations!r}"
            )
        return int(max_evaluations)
    for attr in ("max_evaluations", "evaluations"):
        value = getattr(algorithm, attr, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return int(value)
    raise ValueError(
        "algorithm {!r} does not expose max_evaluations/evaluations and no "
        "explicit max_evaluations was provided".format(
            getattr(algorithm, "name", type(algorithm).__name__)
        )
    )


def build_run_controller(
    problem: Any,
    *,
    max_evaluations: int,
    termination_mode: str = "fixed_budget",
    global_optimum: float | None = None,
    allowed_deviation_percent: float | None = None,
    target_reference: str | None = None,
    target_threshold: float | None = None,
) -> RunController:
    """
    Construct a validated :class:`RunController` for one logged optimisation.

    Parameters
    ----------
    problem : Any
        IOH-like problem instance (already reset for this attempt).
    max_evaluations : int
        Hard evaluation budget.
    termination_mode : str, optional
        ``fixed_budget`` or ``fixed_target``.
    global_optimum : float or None, optional
        Known optimum for fixed-target mode.
    allowed_deviation_percent : float or None, optional
        Allowed percentage deviation for fixed-target mode.
    target_reference : str or None, optional
        ``all`` for fixed-target mode.
    target_threshold : float or None, optional
        Derived minimisation threshold for fixed-target mode.

    Returns
    -------
    RunController
        Controller bound to ``problem``.
    """
    del global_optimum, allowed_deviation_percent  # sealed inputs echoed later
    if termination_mode not in _TERMINATION_MODES:
        raise ValueError(
            f"termination_mode must be one of {sorted(_TERMINATION_MODES)}, "
            f"got {termination_mode!r}"
        )
    if termination_mode == "fixed_budget":
        if target_reference is not None or target_threshold is not None:
            raise ValueError(
                "target_reference and target_threshold must be None when "
                "termination_mode='fixed_budget'"
            )
        return RunController(
            problem,
            max_evaluations,
            mode="fixed_budget",
            target_threshold=None,
            target_reference=None,
        )
    if target_reference not in _TARGET_REFERENCES:
        raise ValueError(
            "target_reference must be one of "
            f"{sorted(_TARGET_REFERENCES)}, got {target_reference!r}"
        )
    if target_threshold is None:
        raise ValueError(
            "target_threshold is required when termination_mode='fixed_target'"
        )
    return RunController(
        problem,
        max_evaluations,
        mode="fixed_target",
        target_threshold=float(target_threshold),
        target_reference=str(target_reference),
    )


def summarize_termination(
    controller: RunController,
    *,
    global_optimum: float | None,
    allowed_deviation_percent: float | None,
) -> dict[str, Any]:
    """
    Build the additive termination fields for one attempt summary.

    Parameters
    ----------
    controller : RunController
        Controller used for the completed algorithm call.
    global_optimum : float or None
        Sealed optimum (null for fixed-budget).
    allowed_deviation_percent : float or None
        Sealed percentage deviation (null for fixed-budget).

    Returns
    -------
    dict
        Termination summary fields including ``stop_reason``.
    """
    stop_reason = controller.stop_reason()
    if stop_reason is None:
        stop_reason = "algorithm_terminated"
    if stop_reason not in _STOP_REASONS:
        raise ValueError(f"unexpected stop_reason {stop_reason!r}")
    return {
        "termination_mode": controller.mode,
        "global_optimum": global_optimum,
        "allowed_deviation_percent": allowed_deviation_percent,
        "target_threshold": controller.target_threshold,
        "target_hit": bool(controller.target_hit),
        "evaluations_to_target": controller.evaluations_to_target,
        "budget_exhausted": bool(controller.budget_exhausted),
        "stop_reason": stop_reason,
    }


def validate_termination_summary(
    summary: Mapping[str, Any],
    *,
    sealed_termination_mode: str,
    sealed_global_optimum: float | None,
    sealed_allowed_deviation_percent: float | None,
    sealed_target_threshold: float | None,
    max_evaluations: int,
) -> None:
    """
    Reject summaries that disagree with sealed termination inputs / outcomes.

    Parameters
    ----------
    summary : Mapping[str, Any]
        Attempt summary about to be published.
    sealed_termination_mode : str
        Task-sealed termination mode.
    sealed_global_optimum : float or None
        Task-sealed optimum.
    sealed_allowed_deviation_percent : float or None
        Task-sealed percentage deviation.
    sealed_target_threshold : float or None
        Task-sealed threshold.
    max_evaluations : int
        Task-sealed hard budget.

    Raises
    ------
    ValueError
        If any consistency check fails.
    """
    required = (
        "termination_mode",
        "global_optimum",
        "allowed_deviation_percent",
        "target_threshold",
        "target_hit",
        "evaluations_to_target",
        "budget_exhausted",
        "stop_reason",
        "evaluations",
        "best_objective",
    )
    missing = [key for key in required if key not in summary]
    if missing:
        raise ValueError(f"summary missing termination fields: {missing}")

    if summary["termination_mode"] != sealed_termination_mode:
        raise ValueError(
            "summary termination_mode "
            f"{summary['termination_mode']!r} != sealed {sealed_termination_mode!r}"
        )
    if summary["global_optimum"] != sealed_global_optimum:
        raise ValueError(
            "summary global_optimum "
            f"{summary['global_optimum']!r} != sealed {sealed_global_optimum!r}"
        )
    if summary["allowed_deviation_percent"] != sealed_allowed_deviation_percent:
        raise ValueError(
            "summary allowed_deviation_percent "
            f"{summary['allowed_deviation_percent']!r} != sealed "
            f"{sealed_allowed_deviation_percent!r}"
        )
    if summary["target_threshold"] != sealed_target_threshold:
        raise ValueError(
            "summary target_threshold "
            f"{summary['target_threshold']!r} != sealed {sealed_target_threshold!r}"
        )

    stop_reason = summary["stop_reason"]
    if stop_reason not in _STOP_REASONS:
        raise ValueError(f"invalid stop_reason {stop_reason!r}")

    evaluations = int(summary["evaluations"])
    if evaluations < 0 or evaluations > int(max_evaluations):
        raise ValueError(
            f"evaluations {evaluations} outside [0, {max_evaluations}]"
        )

    target_hit = bool(summary["target_hit"])
    budget_exhausted = bool(summary["budget_exhausted"])
    evaluations_to_target = summary["evaluations_to_target"]

    if stop_reason == "target_hit" and not target_hit:
        raise ValueError("stop_reason='target_hit' requires target_hit=True")
    if stop_reason == "budget_exhausted" and not budget_exhausted:
        raise ValueError(
            "stop_reason='budget_exhausted' requires budget_exhausted=True"
        )
    if stop_reason == "algorithm_terminated" and (
        target_hit or budget_exhausted
    ):
        raise ValueError(
            "stop_reason='algorithm_terminated' requires no target hit and "
            "no budget exhaustion"
        )

    if target_hit:
        if evaluations_to_target is None:
            raise ValueError("target_hit requires evaluations_to_target")
        hit_at = int(evaluations_to_target)
        if hit_at < 1 or hit_at > evaluations:
            raise ValueError(
                f"evaluations_to_target {hit_at} outside [1, {evaluations}]"
            )
        if sealed_termination_mode == "fixed_target":
            if sealed_target_threshold is None:
                raise ValueError(
                    "fixed_target hit requires a sealed target_threshold"
                )
            best = float(summary["best_objective"])
            if not math.isfinite(best) or best > float(sealed_target_threshold):
                raise ValueError(
                    "target_hit requires finite best_objective <= "
                    f"target_threshold; got best={best!r}, "
                    f"threshold={sealed_target_threshold!r}"
                )
    elif evaluations_to_target is not None:
        raise ValueError(
            "evaluations_to_target must be null when target_hit is false"
        )

    if (
        sealed_termination_mode == "fixed_target"
        and sealed_global_optimum is not None
        and sealed_allowed_deviation_percent is not None
        and sealed_target_threshold is not None
    ):
        expected = float(sealed_global_optimum) * (
            1.0 + float(sealed_allowed_deviation_percent) / 100.0
        )
        if not math.isclose(
            float(sealed_target_threshold),
            expected,
            rel_tol=_THRESHOLD_RTOL,
            abs_tol=0.0,
        ):
            raise ValueError(
                "sealed target_threshold does not match "
                "global_optimum * (1 + allowed_deviation_percent / 100)"
            )


def run_logged_optimization(
    *,
    bundle: dict,
    algorithm: object,
    output_directory: Path,
    max_evaluations: int | None = None,
    termination_mode: str = "fixed_budget",
    global_optimum: float | None = None,
    allowed_deviation_percent: float | None = None,
    target_reference: str | None = None,
    target_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Run one algorithm once on one problem bundle with a private IOH logger.

    Parameters
    ----------
    bundle : dict
        Problem bundle from :func:`build_problems_for_slab` /
        :func:`build_problem_from_snapshot`.
    algorithm : object
        Algorithm instance with a ``name`` attribute and optional ``info``.
        Called as ``algorithm(problem, run_controller=controller)``.
    output_directory : Path
        Fresh directory that receives the IOH Analyzer output. Created if
        missing. Must not be shared with other concurrent runs.
    max_evaluations : int or None, optional
        Explicit hard budget; defaults to the algorithm attribute.
    termination_mode : str, optional
        ``fixed_budget`` (default) or ``fixed_target``.
    global_optimum : float or None, optional
        Known optimum for fixed-target mode.
    allowed_deviation_percent : float or None, optional
        Allowed percentage deviation for fixed-target mode.
    target_reference : str or None, optional
        ``all`` for fixed-target mode.
    target_threshold : float or None, optional
        Derived minimisation threshold for fixed-target mode.

    Returns
    -------
    dict
        Summary with best objective/vector/params, cache hits/misses,
        IOH evaluation count, wall time, ``finite_best`` flag, and
        termination outcome fields.

    Notes
    -----
    Best objective, best index vector, and evaluation count are read from
    the IOH problem state after the run. Algorithm-specific attributes are
    not used. The logger is always detached and closed in ``finally``.
    Output is never zipped or deleted. Non-ASCII destination paths are
    staged through an ASCII temporary directory when required.
    """
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    problem = bundle["problem"]
    eval_context = bundle["ctx"]
    var_names = list(bundle["var_names"])
    constraint_names = list(bundle["active_constraint_names"])
    decode = bundle["decode"]

    budget = resolve_algorithm_max_evaluations(algorithm, max_evaluations)

    eval_context.ensure_constraints(constraint_names)
    eval_context.ensure_params(var_names)

    preexisting = {p.resolve() for p in output_directory.rglob("IOHprofiler_f*.json")}
    started = time.perf_counter()
    summary: dict[str, Any]

    with staged_ioh_logger_directory(output_directory) as logger_root:
        # The list must outlive the run. IOH 0.3.22 does not retain
        # trigger objects, so an inline list is use-after-free.
        trigger_alw = [ltr.Each(1)]
        logger = ioh.logger.Analyzer(
            root=str(logger_root),
            folder_name="run",
            algorithm_name=getattr(algorithm, "name", type(algorithm).__name__),
            algorithm_info=getattr(
                algorithm, "info", "No additional information provided"
            ),
            store_positions=True,
            triggers=trigger_alw,
        )
        try:
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
            problem.reset()
            eval_context.reset()
            attach_eval_context(problem, eval_context)
            controller = build_run_controller(
                problem,
                max_evaluations=budget,
                termination_mode=termination_mode,
                global_optimum=global_optimum,
                allowed_deviation_percent=allowed_deviation_percent,
                target_reference=target_reference,
                target_threshold=target_threshold,
            )
            algorithm(problem, run_controller=controller)

            best_y = float(problem.state.current_best.y)
            best_x = [int(v) for v in list(problem.state.current_best.x)]
            best_params = {name: decode(best_x)[name] for name in var_names}
            evaluations = int(problem.state.evaluations)
            finite_best = bool(math.isfinite(best_y))
            summary = {
                "best_objective": best_y,
                "best_x": best_x,
                "best_params": best_params,
                "hits": int(eval_context.hits),
                "misses": int(eval_context.misses),
                "evaluations": evaluations,
                "wall_time_s": float(time.perf_counter() - started),
                "finite_best": finite_best,
                "algorithm_name": getattr(
                    algorithm, "name", type(algorithm).__name__
                ),
                "output_directory": str(output_directory),
            }
            summary.update(
                summarize_termination(
                    controller,
                    global_optimum=global_optimum,
                    allowed_deviation_percent=allowed_deviation_percent,
                )
            )
            diagnostics_hook = getattr(algorithm, "summary_diagnostics", None)
            if callable(diagnostics_hook):
                summary["algorithm_diagnostics"] = (
                    _validated_algorithm_diagnostics(diagnostics_hook())
                )
            if not finite_best:
                summary["warning"] = "best objective is non-finite"
        finally:
            try:
                detach_eval_context(problem)
            except Exception:
                pass
            try:
                problem.detach_logger()
            except Exception:
                pass
            try:
                logger.close()
            except Exception:
                pass

    patch_ioh_suite_metadata(
        output_directory, suite_name=SUITE_NAME, preexisting=preexisting
    )
    return summary
