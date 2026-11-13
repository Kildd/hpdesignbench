"""
Single-task optimisation worker for prepared experiments.

Invoked as:

``python -m core.ioh_core.workflows.optimization_runs.worker
    --experiment-dir PATH --task-index N --attempt N``

The worker never calls legacy :func:`run_experiment`. Scientific
``analysis_error`` results remain successful task completions; only
infrastructure exceptions write ``failure.json`` and exit nonzero.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from core.execution_core.artifacts import (
    read_task_status,
    touch_attempt_heartbeat,
    transition_task_status,
    write_attempt_owner,
)
from core.execution_core.atomic_io import write_json_atomic
from core.execution_core.manifest import load_manifest
from core.execution_core.models import ExperimentPaths, TaskState, format_task_id
from core.ioh_core.algorithm_factory import create_algorithm
from core.ioh_core.problem_builder import build_problem_from_snapshot
from core.ioh_core.structural_artifacts.artifact import (
    get_structural_build_count,
    reset_structural_build_count,
)
from core.ioh_core.structural_artifacts.eval_backend import (
    configure_bundle_structural_backend,
)
from core.ioh_core.structural_artifacts.sealing import (
    combined_snapshot_hash,
    open_structural_provider_session,
)
from core.ioh_core.workflows.optimization_runs.io_logging import (
    run_logged_optimization,
    validate_termination_summary,
)
from core.ioh_core.workflows.optimization_runs.models import OptimizationTaskSpec
from core.ioh_core.workflows.optimization_runs.planner import OPTIMIZATION_TASK_KIND
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
    validate_snapshot_hashes,
)


class WorkerError(RuntimeError):
    """
    Raised for invalid worker inputs or ownership checks.

    Typical causes: unknown task index, wrong workflow/kind, missing attempt
    directory, or status not in the expected pre-running state.
    """


def experiment_paths(experiment_dir: Path) -> ExperimentPaths:
    """
    Build the standard path bundle for an experiment root.

    Parameters
    ----------
    experiment_dir : Path
        Absolute or relative experiment directory.

    Returns
    -------
    ExperimentPaths
        Resolved layout paths under the experiment root.
    """
    root = Path(experiment_dir).expanduser().resolve()
    return ExperimentPaths(
        root=root,
        inputs=root / "inputs",
        submissions=root / "submissions",
        tasks=root / "tasks",
        aggregate=root / "aggregate",
    )


def task_directory(paths: ExperimentPaths, task_index: int) -> Path:
    """
    Return the on-disk task directory for ``task_index``.

    Parameters
    ----------
    paths : ExperimentPaths
        Experiment path bundle.
    task_index : int
        Zero-based task index.

    Returns
    -------
    Path
        ``tasks/task_{index:06d}``.
    """
    return paths.tasks / format_task_id(task_index)


def attempt_directory(task_dir: Path, attempt: int) -> Path:
    """
    Return the reserved attempt directory path.

    Parameters
    ----------
    task_dir : Path
        Parent task directory.
    attempt : int
        One-based attempt index.

    Returns
    -------
    Path
        ``attempts/{attempt:03d}``.
    """
    if attempt < 1:
        raise WorkerError(f"attempt must be >= 1, got {attempt}")
    return Path(task_dir) / "attempts" / f"{attempt:03d}"


def validate_attempt_ownership(task_dir: Path, attempt: int) -> Path:
    """
    Require that the reserved attempt directory exists for this worker.

    Parameters
    ----------
    task_dir : Path
        Parent task directory.
    attempt : int
        One-based attempt index claimed by the caller.

    Returns
    -------
    Path
        Existing attempt directory.

    Raises
    ------
    WorkerError
        If the attempt directory is missing.
    """
    attempt_dir = attempt_directory(task_dir, attempt)
    if not attempt_dir.is_dir():
        raise WorkerError(
            f"attempt directory missing for ownership check: {attempt_dir}"
        )
    return attempt_dir


def _json_safe(value: Any) -> Any:
    """
    Convert summary values to JSON-serialisable forms.

    Parameters
    ----------
    value : Any
        Nested summary object.

    Returns
    -------
    Any
        JSON-safe value (lists/dicts/scalars/strings).
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value  # json allows non-finite via default=str later
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def run_worker(
    *,
    experiment_dir: Path,
    task_index: int,
    attempt: int,
) -> dict[str, Any]:
    """
    Execute one prepared optimisation task attempt.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root containing ``manifest.json`` and ``inputs/``.
    task_index : int
        Zero-based task index from the manifest.
    attempt : int
        One-based reserved attempt index.

    Returns
    -------
    dict
        Written summary document.

    Raises
    ------
    WorkerError
        If ownership/workflow validation fails before execution.
    Exception
        Infrastructure failures after ``running`` are recorded as
        ``failure.json`` by :func:`main` and re-raised.
    """
    paths = experiment_paths(experiment_dir)
    manifest = load_manifest(paths.manifest)
    if manifest.workflow_type != "optimization_runs":
        raise WorkerError(
            f"worker requires workflow_type='optimization_runs', "
            f"got {manifest.workflow_type!r}"
        )
    if task_index < 0 or task_index >= len(manifest.tasks):
        raise WorkerError(
            f"task_index {task_index} out of range for "
            f"{len(manifest.tasks)} manifest tasks"
        )
    task = manifest.tasks[task_index]
    if task.kind != OPTIMIZATION_TASK_KIND:
        raise WorkerError(
            f"task {task_index} has kind {task.kind!r}, "
            f"expected {OPTIMIZATION_TASK_KIND!r}"
        )

    validate_snapshot_hashes(
        inputs_dir=paths.inputs,
        expected_hashes=manifest.input_files,
    )

    task_dir = task_directory(paths, task_index)
    attempt_dir = validate_attempt_ownership(task_dir, attempt)

    status = read_task_status(task_dir)
    current = TaskState(status["state"])
    if current is TaskState.PENDING:
        # Allow a direct pending→dispatched hand-off for local tests that
        # reserved an attempt without a backend.
        transition_task_status(
            task_dir,
            expected={TaskState.PENDING},
            new_state=TaskState.DISPATCHED,
            attempt=attempt,
        )
        current = TaskState.DISPATCHED
    if current is not TaskState.DISPATCHED:
        raise WorkerError(
            f"task {task_index} must be dispatched before running, "
            f"got {current.value!r}"
        )

    transition_task_status(
        task_dir,
        expected={TaskState.DISPATCHED},
        new_state=TaskState.RUNNING,
        attempt=attempt,
    )
    write_attempt_owner(attempt_dir, backend="local", pid=os.getpid())
    touch_attempt_heartbeat(attempt_dir)

    spec = OptimizationTaskSpec.from_payload(task.payload)
    snapshot_dir = resolve_snapshot_slab_dir(paths.inputs, spec.slab_type)
    snapshot_hash = combined_snapshot_hash(manifest.input_files)

    with open_structural_provider_session(paths.root) as (seal, provider):
        bundle = build_problem_from_snapshot(
            slab_type=spec.slab_type,
            problem_id=spec.problem_id,
            snapshot_dir=snapshot_dir,
        )
        configure_bundle_structural_backend(
            bundle,
            provider=provider,
            snapshot_hash=snapshot_hash,
        )
        reset_structural_build_count()

        algorithm = create_algorithm(
            spec.algorithm_name,
            spec.algorithm_parameters,
            random_seed=spec.random_seed,
        )

        ioh_dir = attempt_dir / "ioh"
        max_evaluations = int(spec.algorithm_parameters["max_evaluations"])
        summary = run_logged_optimization(
            bundle=bundle,
            algorithm=algorithm,
            output_directory=ioh_dir,
            max_evaluations=max_evaluations,
            termination_mode=spec.termination_mode,
            global_optimum=spec.global_optimum,
            allowed_deviation_percent=spec.allowed_deviation_percent,
            target_reference=spec.target_reference,
            target_threshold=spec.target_threshold,
        )
        structural_builds = int(get_structural_build_count())
        if seal.mode == "library" and structural_builds != 0:
            raise WorkerError(
                "library mode performed unexpected structural builds: "
                f"{structural_builds}"
            )
        summary.update(
            {
                "structural_artifacts_mode": seal.mode,
                "structural_library_hits": int(provider.metrics.library_hits),
                "structural_library_build_failure_fanouts": int(
                    provider.metrics.library_build_failure_fanouts
                ),
                "structural_builds": structural_builds,
            }
        )
        bundle["ctx"].clear_structural_backend()

    summary.update(
        {
            "task_index": task_index,
            "task_id": task.task_id,
            "attempt": attempt,
            "problem_id": spec.problem_id,
            "slab_type": spec.slab_type,
            "repetition": spec.repetition,
            "random_seed": spec.random_seed,
            "algorithm_parameters": dict(spec.algorithm_parameters),
            # Echo sealed termination inputs (authoritative task payload).
            "termination_mode": spec.termination_mode,
            "global_optimum": spec.global_optimum,
            "allowed_deviation_percent": spec.allowed_deviation_percent,
            "target_threshold": spec.target_threshold,
            "target_reference": spec.target_reference,
        }
    )
    try:
        validate_termination_summary(
            summary,
            sealed_termination_mode=spec.termination_mode,
            sealed_global_optimum=spec.global_optimum,
            sealed_allowed_deviation_percent=spec.allowed_deviation_percent,
            sealed_target_threshold=spec.target_threshold,
            max_evaluations=max_evaluations,
        )
    except ValueError as exc:
        raise WorkerError(
            f"termination summary inconsistent for task {task_index}: {exc}"
        ) from exc
    summary_path = attempt_dir / "summary.json"
    touch_attempt_heartbeat(attempt_dir)
    write_json_atomic(summary_path, _json_safe(summary))

    transition_task_status(
        task_dir,
        expected={TaskState.RUNNING},
        new_state=TaskState.COMPLETED,
        attempt=attempt,
        metadata={"summary": str(summary_path)},
    )
    return summary


def _write_failure(attempt_dir: Path, exc: BaseException) -> Path:
    """
    Atomically write ``failure.json`` for an infrastructure exception.

    Parameters
    ----------
    attempt_dir : Path
        Reserved attempt directory.
    exc : BaseException
        Exception to record.

    Returns
    -------
    Path
        Path of the written failure document.
    """
    path = Path(attempt_dir) / "failure.json"
    write_json_atomic(
        path,
        {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        },
    )
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the worker CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``--experiment-dir``, ``--task-index``, and ``--attempt``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m core.ioh_core.workflows.optimization_runs.worker",
        description="Execute one prepared optimisation task attempt.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Prepared experiment root containing manifest.json",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        required=True,
        help="Zero-based task index from the manifest",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        required=True,
        help="One-based reserved attempt index",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint for the optimisation worker.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (``0`` success, ``1`` infrastructure failure).
    """
    args = build_arg_parser().parse_args(argv)
    experiment_dir = Path(args.experiment_dir)
    task_index = int(args.task_index)
    attempt = int(args.attempt)

    attempt_dir: Path | None = None
    task_dir: Path | None = None
    try:
        paths = experiment_paths(experiment_dir)
        task_dir = task_directory(paths, task_index)
        attempt_dir = validate_attempt_ownership(task_dir, attempt)
        run_worker(
            experiment_dir=experiment_dir,
            task_index=task_index,
            attempt=attempt,
        )
        return 0
    except Exception as exc:
        # Do not catch BaseException (KeyboardInterrupt/SystemExit).
        if attempt_dir is not None:
            failure_path = _write_failure(attempt_dir, exc)
            if task_dir is not None:
                try:
                    status = read_task_status(task_dir)
                    state = TaskState(status["state"])
                    if state is TaskState.RUNNING:
                        transition_task_status(
                            task_dir,
                            expected={TaskState.RUNNING},
                            new_state=TaskState.FAILED,
                            attempt=attempt,
                            metadata={
                                "error_type": type(exc).__name__,
                                "failure": str(failure_path),
                            },
                        )
                except Exception:
                    pass
        print(f"[worker] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
