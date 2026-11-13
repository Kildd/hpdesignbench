"""
Experiment lifecycle operations: prepare, dispatch, status, retry, reconcile.

Workflow planners and aggregators are looked up via the workflow registry.
This module stays free of IOH/slab imports; it only schedules generic tasks
and administers on-disk status/ownership artifacts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.execution_core.artifacts import (
    create_task_layout,
    pid_is_alive,
    read_attempt_heartbeat,
    read_attempt_owner,
    read_task_status,
    touch_attempt_heartbeat,
    utc_now,
    write_attempt_owner,
)
from core.execution_core.atomic_io import read_json, write_json_atomic
from core.execution_core.backends.factory import create_execution_backend
from core.execution_core.config import (
    ExperimentConfig,
    experiment_root_for,
    load_experiment_config,
)
from core.execution_core.manifest import (
    load_environment,
    load_manifest,
    prepare_experiment_artifacts,
)
from core.execution_core.models import (
    ExperimentPaths,
    SubmissionRecord,
    TaskState,
    TERMINAL_TASK_STATES,
    format_task_id,
)

DEFAULT_STALE_GRACE_SECONDS = 60.0
RETRY_ELIGIBLE_STATES: frozenset[TaskState] = frozenset(
    {
        TaskState.FAILED,
        TaskState.TIMED_OUT,
        TaskState.CANCELLED,
    }
)
ACTIVE_STATES: frozenset[TaskState] = frozenset(
    {
        TaskState.DISPATCHED,
        TaskState.RUNNING,
    }
)


class LifecycleError(RuntimeError):
    """
    Raised when a lifecycle operation cannot proceed safely.

    Typical causes: missing experiment artifacts, illegal retry of a completed
    task, or an unsupported workflow type.
    """


# Re-export ownership helpers for callers that import from lifecycle.
__all__ = [
    "LifecycleError",
    "DEFAULT_STALE_GRACE_SECONDS",
    "prepare_experiment",
    "dispatch_experiment",
    "list_task_statuses",
    "wait_experiment",
    "retry_failed_tasks",
    "cancel_tasks",
    "reconcile_stale_tasks",
    "reconcile_experiment",
    "collect_experiment",
    "run_experiment_lifecycle",
    "format_experiment_startup_message",
    "print_experiment_startup",
    "write_attempt_owner",
    "touch_attempt_heartbeat",
    "read_attempt_owner",
    "read_attempt_heartbeat",
    "pid_is_alive",
    "valid_summary_artifact",
    "attempt_directory",
    "experiment_paths",
    "pending_task_indices",
    "load_prepared_config",
]


def _iso_utc(moment: datetime) -> str:
    """
    Format a timezone-aware datetime as a UTC ISO-8601 ``Z`` string.

    Parameters
    ----------
    moment : datetime
        Timezone-aware timestamp.

    Returns
    -------
    str
        UTC timestamp ending in ``Z``.
    """
    if moment.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_utc(value: str) -> datetime:
    """
    Parse a UTC ISO-8601 timestamp into a timezone-aware datetime.

    Parameters
    ----------
    value : str
        Timestamp string; trailing ``Z`` is accepted.

    Returns
    -------
    datetime
        Timezone-aware UTC datetime.
    """
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise LifecycleError(f"timestamp must be timezone-aware, got {value!r}")
    return moment.astimezone(timezone.utc)


def experiment_paths(experiment_dir: Path) -> ExperimentPaths:
    """
    Build the standard path bundle for an experiment root.

    Parameters
    ----------
    experiment_dir : Path
        Experiment directory containing ``manifest.json``.

    Returns
    -------
    ExperimentPaths
        Resolved layout paths.
    """
    root = Path(experiment_dir).expanduser().resolve()
    return ExperimentPaths(
        root=root,
        inputs=root / "inputs",
        submissions=root / "submissions",
        tasks=root / "tasks",
        aggregate=root / "aggregate",
    )


def load_prepared_config(experiment_dir: Path) -> ExperimentConfig:
    """
    Load the sealed ``source-config.toml`` from a prepared experiment.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.

    Returns
    -------
    ExperimentConfig
        Validated configuration.

    Raises
    ------
    LifecycleError
        If the source config is missing.
    """
    paths = experiment_paths(experiment_dir)
    if not paths.source_config.is_file():
        raise LifecycleError(f"missing source-config.toml under {paths.root}")
    return load_experiment_config(paths.source_config)


def format_experiment_startup_message(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    n_tasks: int | None = None,
) -> str:
    """
    Build a short human-readable startup banner for an experiment.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment configuration.
    experiment_dir : Path
        Prepared experiment root that will hold tasks and aggregates.
    n_tasks : int or None, optional
        Planned task count. For optimization, the fallback respects each
        algorithm's repetition override.

    Returns
    -------
    str
        Multi-line banner ending with a newline.
    """
    exp = config.experiment
    exe = config.execution
    n_problems = len(exp.problem_ids)
    n_algorithms = len(exp.algorithms)
    n_reps = 0 if exp.repetitions is None else exp.repetitions.count
    effective_repetitions = tuple(
        algo.repetitions if algo.repetitions is not None else n_reps
        for algo in exp.algorithms
    )
    if n_tasks is None:
        if exp.workflow in {
            "brute_force",
            "structural_library",
            "sensitivity_analysis",
        }:
            n_tasks = 0
        elif exp.workflow == "optimization_runs":
            n_tasks = n_problems * sum(effective_repetitions)
        else:
            n_tasks = n_problems * n_algorithms * n_reps

    algorithm_lines: list[str] = []
    for algo, repetitions in zip(
        exp.algorithms,
        effective_repetitions,
        strict=True,
    ):
        extras = [f"max_evaluations={algo.max_evaluations}"]
        if algo.repetitions is not None:
            extras.append(f"repetitions={repetitions}")
        for key in sorted(algo.parameters):
            extras.append(f"{key}={algo.parameters[key]!r}")
        algorithm_lines.append(f"{algo.name} ({', '.join(extras)})")

    problems = ", ".join(exp.problem_ids)
    timeout = (
        f"{exe.timeout_seconds:g}s"
        if exe.timeout_seconds is not None
        else "none"
    )
    lines = [
        f"[hdb-experiment] Starting {exp.id}",
        f"  directory   : {Path(experiment_dir).expanduser().resolve()}",
        f"  workflow    : {exp.workflow}",
        f"  slab        : {exp.slab_type}",
        f"  problems    : {problems} ({n_problems})",
        f"  structural  : mode={config.structural_artifacts.mode}",
    ]
    if exp.workflow == "brute_force" and config.brute_force is not None:
        bf = config.brute_force
        lines.extend(
            [
                f"  brute_force : rows_per_shard={bf.rows_per_shard}, "
                f"compression={bf.compression}, "
                f"reuse_structural_artifacts={bf.reuse_structural_artifacts}, "
                f"start_index={bf.start_index}, "
                f"max_rows={bf.max_rows}, "
                f"acknowledge_large_run={bf.acknowledge_large_run}",
                f"  tasks       : {n_tasks} shard(s)",
            ]
        )
    elif (
        exp.workflow == "structural_library"
        and config.structural_library is not None
    ):
        sl = config.structural_library
        max_ids = "full" if sl.max_ids is None else str(sl.max_ids)
        from core.ioh_core.workflows.structural_library.validation import (
            library_directory,
        )

        lines.extend(
            [
                f"  library     : rows_per_shard={sl.rows_per_shard}, "
                f"start_id={sl.start_id}, max_ids={max_ids}, "
                f"acknowledge_large_run={sl.acknowledge_large_run}",
                f"  tasks       : {n_tasks} shard(s)",
                f"  publish     : {library_directory(Path(experiment_dir))}",
            ]
        )
        try:
            from core.ioh_core.workflows.structural_library.preflight import (
                run_preflight,
            )

            report = run_preflight(config, raise_if_blocked=False)
            parallel = max(1, int(exe.max_parallel_tasks))
            est_wall_h = report.estimated_serial_time_s / 3600.0 / parallel
            lines.append(
                f"  estimate    : {report.planned_ids} IDs, "
                f"~{report.estimated_total_bytes / 1e9:.1f} GB, "
                f"~{est_wall_h:.1f} h wall @ {parallel} workers (preflight)"
            )
        except Exception:  # noqa: BLE001
            pass
    elif (
        exp.workflow == "sensitivity_analysis"
        and config.sensitivity_analysis is not None
    ):
        sa = config.sensitivity_analysis
        factors = ", ".join(sa.factor_names)
        lines.extend(
            [
                f"  sensitivity : N={sa.sample_size}, factors=[{factors}], "
                f"response={sa.response_key!r}, "
                f"blocks_per_shard={sa.blocks_per_shard}",
                f"  tasks       : {n_tasks} shard(s)",
            ]
        )
    else:
        lines.append("  algorithms  :")
        if algorithm_lines:
            for line in algorithm_lines:
                lines.append(f"    - {line}")
        else:
            lines.append("    - (none)")
        seed = (
            "n/a"
            if exp.repetitions is None
            else str(exp.repetitions.seed_base)
        )
        if exp.workflow == "optimization_runs":
            task_breakdown = (
                f"{n_problems} problem(s) x "
                f"{sum(effective_repetitions)} effective "
                "algorithm repetition(s) per problem"
            )
        else:
            task_breakdown = (
                f"{n_problems} problem(s) x {n_algorithms} algorithm(s) "
                f"x {n_reps} repetition(s)"
            )
        lines.extend(
            [
                f"  repetitions : {n_reps} (seed_base={seed})",
                f"  tasks       : {n_tasks} ({task_breakdown})",
            ]
        )
    lines.append(
        f"  execution   : backend={exe.backend}, "
        f"max_parallel_tasks={exe.max_parallel_tasks}, "
        f"cpus_per_task={exe.cpus_per_task}, "
        f"timeout={timeout}"
    )
    return "\n".join(lines) + "\n"


def print_experiment_startup(
    experiment_dir: Path,
    *,
    config: ExperimentConfig | None = None,
) -> None:
    """
    Print a startup banner for a prepared experiment directory.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.
    config : ExperimentConfig or None, optional
        Config to summarise. When omitted, loads the sealed source config.
    """
    paths = experiment_paths(experiment_dir)
    if config is None:
        config = load_prepared_config(paths.root)
    manifest = load_manifest(paths.manifest)
    message = format_experiment_startup_message(
        config,
        experiment_dir=paths.root,
        n_tasks=len(manifest.tasks),
    )
    print(message, flush=True)


def _admin_write_status(
    task_dir: Path,
    *,
    state: TaskState,
    attempt: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Privileged status publisher used for retry/cancel/stale recovery.

    Bypasses the normal directed transition table because administrative
    recovery must return terminal tasks to ``pending`` and may cancel
    non-running work.

    Parameters
    ----------
    task_dir : Path
        Task directory containing ``status.json``.
    state : TaskState
        New administrative state.
    attempt : int
        Attempt index recorded on the status document.
    metadata : dict or None, optional
        Metadata object replacement/merge source.
    """
    current = read_task_status(task_dir)
    if TaskState(current["state"]) is TaskState.COMPLETED and state is not TaskState.COMPLETED:
        raise LifecycleError(f"completed status is immutable for {task_dir}")
    merged: dict[str, Any] = {}
    existing = current.get("metadata")
    if isinstance(existing, dict):
        merged.update(existing)
    if metadata:
        merged.update(metadata)
    write_json_atomic(
        Path(task_dir) / "status.json",
        {
            "task_id": current.get("task_id", Path(task_dir).name),
            "task_index": current.get("task_index"),
            "state": state.value,
            "attempt": int(attempt),
            "updated_at": _iso_utc(utc_now()),
            "metadata": merged,
        },
    )


def valid_summary_artifact(attempt_dir: Path) -> Path | None:
    """
    Return the path of a valid ``summary.json`` under ``attempt_dir``.

    Parameters
    ----------
    attempt_dir : Path
        Attempt directory.

    Returns
    -------
    Path or None
        Summary path when a mapping with ``best_objective`` exists.
    """
    path = Path(attempt_dir) / "summary.json"
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if isinstance(payload, dict) and "best_objective" in payload:
        return path
    return None


def attempt_directory(task_dir: Path, attempt: int) -> Path:
    """
    Return ``attempts/{attempt:03d}`` under ``task_dir``.

    Parameters
    ----------
    task_dir : Path
        Task directory.
    attempt : int
        One-based attempt index.

    Returns
    -------
    Path
        Attempt directory path.
    """
    if attempt < 1:
        raise LifecycleError(f"attempt must be >= 1, got {attempt}")
    return Path(task_dir) / "attempts" / f"{attempt:03d}"


@dataclass(frozen=True)
class TaskStatusSnapshot:
    """
    Read-only view of one task's on-disk status.

    Attributes
    ----------
    task_index : int
        Zero-based task index.
    task_id : str
        Stable task id.
    state : TaskState
        Current lifecycle state.
    attempt : int
        Latest attempt index (``0`` before first reservation).
    updated_at : str
        Status timestamp string.
    metadata : Mapping[str, Any]
        Status metadata object.
    """

    task_index: int
    task_id: str
    state: TaskState
    attempt: int
    updated_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentStatus:
    """
    Aggregate counts and per-task snapshots for one experiment.

    Attributes
    ----------
    experiment_dir : Path
        Experiment root.
    counts : Mapping[str, int]
        Counts keyed by ``TaskState.value``.
    tasks : tuple of TaskStatusSnapshot
        Ordered task snapshots matching the manifest.
    complete : bool
        ``True`` when every task is in a terminal state.
    """

    experiment_dir: Path
    counts: Mapping[str, int]
    tasks: tuple[TaskStatusSnapshot, ...]
    complete: bool


def _import_registry():
    """
    Import the workflow registry lazily to keep execution_core free of IOH.

    Returns
    -------
    module
        ``core.ioh_core.workflows.registry``.
    """
    from core.ioh_core.workflows import registry as workflow_registry

    return workflow_registry


def prepare_experiment(
    *,
    config_path: Path,
    output: Path | None = None,
    overwrite: bool = False,
) -> ExperimentPaths:
    """
    Validate, snapshot, plan, and seal a new experiment directory.

    Parameters
    ----------
    config_path : Path
        Experiment TOML path.
    output : Path or None, optional
        Explicit experiment root. When omitted, uses
        :func:`experiment_root_for` relative to the config directory.
    overwrite : bool, optional
        Allow replacing an existing sealed manifest.

    Returns
    -------
    ExperimentPaths
        Prepared experiment layout.

    Raises
    ------
    LifecycleError
        If the workflow is unknown or preparation fails.
    """
    config = load_experiment_config(config_path)
    registry = _import_registry()
    entry = registry.get_workflow(config.experiment.workflow)
    if output is not None:
        experiment_dir = Path(output).expanduser().resolve()
    else:
        base = config.source_path.parent if config.source_path is not None else Path.cwd()
        experiment_dir = experiment_root_for(config, cwd=base)

    from core.execution_core.output_layout import (
        layout_root_from_output_root,
        write_layout_sidecar,
    )

    config_base = (
        config.source_path.parent
        if config.source_path is not None
        else Path.cwd()
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_layout_sidecar(
        experiment_dir,
        layout_root_from_output_root(
            config.experiment.output_root,
            cwd=config_base,
        ),
    )

    hashes = entry.snapshot(config, experiment_dir)
    tasks = entry.planner(config)
    if entry.validator is not None:
        entry.validator(config, tasks)

    paths, _manifest, _env = prepare_experiment_artifacts(
        config,
        root=experiment_dir,
        tasks=tasks,
        input_files=hashes,
        overwrite=overwrite,
    )
    for task in tasks:
        create_task_layout(paths, task.task_index)
    return paths


def list_task_statuses(experiment_dir: Path) -> ExperimentStatus:
    """
    Read every task status without mutating disk state.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.

    Returns
    -------
    ExperimentStatus
        Counts and ordered snapshots.
    """
    paths = experiment_paths(experiment_dir)
    manifest = load_manifest(paths.manifest)
    counts = {state.value: 0 for state in TaskState}
    snapshots: list[TaskStatusSnapshot] = []
    for task in manifest.tasks:
        task_dir = paths.tasks / task.task_id
        status = read_task_status(task_dir)
        state = TaskState(status["state"])
        counts[state.value] = counts.get(state.value, 0) + 1
        meta = status.get("metadata") if isinstance(status.get("metadata"), dict) else {}
        snapshots.append(
            TaskStatusSnapshot(
                task_index=task.task_index,
                task_id=task.task_id,
                state=state,
                attempt=int(status.get("attempt", 0)),
                updated_at=str(status.get("updated_at", "")),
                metadata=dict(meta),
            )
        )
    complete = all(snap.state in TERMINAL_TASK_STATES for snap in snapshots)
    if not snapshots:
        complete = True
    return ExperimentStatus(
        experiment_dir=paths.root,
        counts=counts,
        tasks=tuple(snapshots),
        complete=complete,
    )


def pending_task_indices(experiment_dir: Path) -> list[int]:
    """
    Return task indices whose status is ``pending``.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.

    Returns
    -------
    list of int
        Pending indices in manifest order.
    """
    status = list_task_statuses(experiment_dir)
    return [snap.task_index for snap in status.tasks if snap.state is TaskState.PENDING]


def dispatch_experiment(
    experiment_dir: Path,
    *,
    task_indices: list[int] | None = None,
    reconcile_first: bool = True,
) -> list[SubmissionRecord]:
    """
    Dispatch pending (or selected) tasks through the configured backend.

    Local backends block until the batch finishes. Slurm backends (later)
    submit and return.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.
    task_indices : list of int or None, optional
        Explicit indices to dispatch. Defaults to all pending tasks.
    reconcile_first : bool, optional
        When ``True``, run stale-state reconciliation before dispatch.

    Returns
    -------
    list of SubmissionRecord
        Backend submission records (empty when nothing is pending).

    Raises
    ------
    LifecycleError
        If no pending work remains when indices were explicitly requested
        and none are pending, or artifacts are missing.
    """
    paths = experiment_paths(experiment_dir)
    if not paths.manifest.is_file():
        raise LifecycleError(f"missing manifest.json under {paths.root}")
    if reconcile_first:
        reconcile_stale_tasks(paths.root)

    config = load_prepared_config(paths.root)
    manifest = load_manifest(paths.manifest)
    load_environment(paths.environment)

    if task_indices is None:
        indices = pending_task_indices(paths.root)
    else:
        indices = list(task_indices)
        for index in indices:
            task_dir = paths.tasks / format_task_id(index)
            state = TaskState(read_task_status(task_dir)["state"])
            if state is not TaskState.PENDING:
                raise LifecycleError(
                    f"task {index} is {state.value!r}, expected pending"
                )

    if not indices:
        return []

    backend = create_execution_backend(config)
    return backend.dispatch(
        experiment_dir=paths.root,
        task_indices=indices,
        worker_module=manifest.worker_module,
    )


def wait_experiment(
    experiment_dir: Path,
    *,
    poll_interval_seconds: float = 0.25,
    timeout_seconds: float | None = None,
    reconcile: bool = True,
) -> ExperimentStatus:
    """
    Block until every task is terminal (or timeout).

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.
    poll_interval_seconds : float, optional
        Sleep between status polls.
    timeout_seconds : float or None, optional
        Optional wait deadline; ``None`` waits indefinitely.
    reconcile : bool, optional
        When ``True``, reconcile stale active tasks each poll.

    Returns
    -------
    ExperimentStatus
        Final status snapshot.

    Raises
    ------
    TimeoutError
        If ``timeout_seconds`` elapses before completion.
    """
    started = time.monotonic()
    while True:
        if reconcile:
            reconcile_experiment(experiment_dir)
        status = list_task_statuses(experiment_dir)
        if status.complete:
            return status
        if timeout_seconds is not None and (time.monotonic() - started) >= timeout_seconds:
            raise TimeoutError(
                f"experiment {experiment_dir} still incomplete after "
                f"{timeout_seconds}s"
            )
        time.sleep(poll_interval_seconds)


def retry_failed_tasks(experiment_dir: Path) -> list[int]:
    """
    Reset failed, timed-out, or cancelled tasks to ``pending``.

    Completed tasks are never touched. Previous attempt directories are
    preserved; the next dispatch reserves a new attempt index.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.

    Returns
    -------
    list of int
        Task indices reset for retry.
    """
    paths = experiment_paths(experiment_dir)
    reset: list[int] = []
    for snap in list_task_statuses(paths.root).tasks:
        if snap.state not in RETRY_ELIGIBLE_STATES:
            continue
        task_dir = paths.tasks / snap.task_id
        _admin_write_status(
            task_dir,
            state=TaskState.PENDING,
            attempt=0,
            metadata={
                "retry_of_attempt": snap.attempt,
                "previous_state": snap.state.value,
            },
        )
        reset.append(snap.task_index)
    return reset


def cancel_tasks(
    experiment_dir: Path,
    *,
    task_indices: list[int] | None = None,
    force_stale_seconds: float | None = None,
    slurm_commands=None,
) -> list[int]:
    """
    Cancel pending tasks and Slurm-dispatched work when configured.

    For ``backend='slurm'``, active submitted tasks are cancelled via
    ``scancel`` and then reconciled. Purely pending tasks are marked
    cancelled locally. ``scancel`` failures leave task state unchanged.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.
    task_indices : list of int or None, optional
        Optional subset; defaults to all non-terminal tasks.
    force_stale_seconds : float or None, optional
        Forwarded to local stale reconciliation.
    slurm_commands : object or None, optional
        Optional injectable Slurm CLI runner for tests.

    Returns
    -------
    list of int
        Task indices marked cancelled (or requested for Slurm cancel).
    """
    paths = experiment_paths(experiment_dir)
    selected = None if task_indices is None else set(task_indices)
    cancelled: list[int] = []

    reconcile_stale_tasks(
        paths.root,
        force_stale_seconds=force_stale_seconds,
    )

    config = None
    try:
        config = load_prepared_config(paths.root)
    except LifecycleError:
        config = None

    slurm_backend = None
    if config is not None and config.execution.backend == "slurm":
        from core.execution_core.backends.factory import create_execution_backend
        from core.execution_core.backends.slurm import SlurmArrayBackend
        from core.execution_core.backends.slurm_commands import SlurmCommandError
        from core.execution_core.backends.slurm_status import (
            SlurmStatusError,
            list_submission_dirs,
        )

        backend = create_execution_backend(config)
        if isinstance(backend, SlurmArrayBackend):
            if slurm_commands is not None:
                backend.commands = slurm_commands
            slurm_backend = backend

            if selected is None:
                # Cancel whole submissions first, then reconcile.
                for submission_dir in list_submission_dirs(paths.root):
                    try:
                        backend.cancel_submission(paths.root, submission_dir)
                    except (SlurmCommandError, SlurmStatusError):
                        # Leave state unchanged on command failure.
                        continue
            else:
                for task_index in sorted(selected):
                    snap_state = TaskState(
                        read_task_status(paths.tasks / format_task_id(task_index))["state"]
                    )
                    if snap_state in TERMINAL_TASK_STATES:
                        continue
                    if snap_state is TaskState.PENDING:
                        continue
                    try:
                        backend.cancel_task(paths.root, task_index)
                        cancelled.append(task_index)
                    except (SlurmCommandError, SlurmStatusError):
                        continue

    for snap in list_task_statuses(paths.root).tasks:
        if selected is not None and snap.task_index not in selected:
            continue
        if snap.state in TERMINAL_TASK_STATES:
            continue
        if snap.state is TaskState.RUNNING and slurm_backend is None:
            # Live local running work is left alone.
            continue
        if snap.state is TaskState.RUNNING and slurm_backend is not None:
            # Slurm running tasks should already have been scancelled above
            # when selected/all; if still running, leave for reconcile.
            continue
        if snap.state is TaskState.DISPATCHED and slurm_backend is not None:
            # Already handled via scancel+reconcile when possible.
            if selected is None or snap.task_index in cancelled:
                continue
            # Pending-only path: if still dispatched after failed scancel, skip.
            continue
        task_dir = paths.tasks / snap.task_id
        _admin_write_status(
            task_dir,
            state=TaskState.CANCELLED,
            attempt=int(snap.attempt),
            metadata={"failure_kind": "cancelled"},
        )
        cancelled.append(snap.task_index)
    return sorted(set(cancelled))


def _heartbeat_age_seconds(heartbeat: Mapping[str, Any] | None) -> float | None:
    """
    Return heartbeat age in seconds, or ``None`` when unavailable.

    Parameters
    ----------
    heartbeat : Mapping or None
        Heartbeat document.

    Returns
    -------
    float or None
        Age in seconds.
    """
    if not heartbeat or "updated_at" not in heartbeat:
        return None
    try:
        moment = _parse_iso_utc(str(heartbeat["updated_at"]))
    except Exception:
        return None
    return max(0.0, (utc_now() - moment).total_seconds())


def _owner_is_local_dead(owner: Mapping[str, Any] | None) -> bool | None:
    """
    Classify local owner liveness.

    Parameters
    ----------
    owner : Mapping or None
        Owner document.

    Returns
    -------
    bool or None
        ``True`` when a local pid is confirmed dead, ``False`` when alive,
        ``None`` when ownership is remote/unknown and cannot be decided.
    """
    if not owner:
        return None
    backend = str(owner.get("backend") or "")
    if backend == "slurm" or owner.get("slurm_job_id"):
        return None
    pid = owner.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool):
        return None
    return not pid_is_alive(pid)


def reconcile_stale_tasks(
    experiment_dir: Path,
    *,
    timeout_seconds: float | None = None,
    grace_seconds: float = DEFAULT_STALE_GRACE_SECONDS,
    force_stale_seconds: float | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """
    Adopt completed artifacts or mark confirmed-stale active attempts.

    Never infers failure from ``status.json`` age alone. A task may be marked
    failed for staleness only when the heartbeat exceeds
    ``timeout + grace`` and the local owner is confirmed dead, or when the
    caller supplies an explicit ``force_stale_seconds`` threshold for
    unknown/remote ownership.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.
    timeout_seconds : float or None, optional
        Per-task wall timeout. Defaults to the sealed config value.
    grace_seconds : float, optional
        Extra heartbeat grace beyond the timeout.
    force_stale_seconds : float or None, optional
        Explicit age after which unknown/remote owners may be forced stale.
    now : datetime or None, optional
        Optional clock override for tests.

    Returns
    -------
    list of dict
        Actions taken (``adopted`` / ``stale`` records).
    """
    del now  # reserved for future clock injection; heartbeat ages use utc_now
    paths = experiment_paths(experiment_dir)
    config = load_prepared_config(paths.root)
    if timeout_seconds is None:
        timeout_seconds = config.execution.timeout_seconds
    threshold = float(timeout_seconds or 0.0) + float(grace_seconds)
    actions: list[dict[str, Any]] = []

    for snap in list_task_statuses(paths.root).tasks:
        if snap.state not in ACTIVE_STATES:
            continue
        if snap.attempt < 1:
            continue
        task_dir = paths.tasks / snap.task_id
        attempt_dir = attempt_directory(task_dir, snap.attempt)

        summary_path = valid_summary_artifact(attempt_dir)
        if summary_path is not None:
            _admin_write_status(
                task_dir,
                state=TaskState.COMPLETED,
                attempt=snap.attempt,
                metadata={
                    "summary": str(summary_path),
                    "reconciled": "adopted_summary",
                },
            )
            actions.append(
                {
                    "task_index": snap.task_index,
                    "action": "adopted",
                    "summary": str(summary_path),
                }
            )
            continue

        heartbeat = read_attempt_heartbeat(attempt_dir)
        age = _heartbeat_age_seconds(heartbeat)
        owner = read_attempt_owner(attempt_dir)
        local_dead = _owner_is_local_dead(owner)

        stale = False
        reason = None
        if age is not None and age > threshold and local_dead is True:
            stale = True
            reason = "stale_heartbeat_dead_owner"
        elif (
            force_stale_seconds is not None
            and age is not None
            and age > float(force_stale_seconds)
            and local_dead is not False
        ):
            # Unknown/remote (local_dead is None) or already dead.
            stale = True
            reason = "forced_stale"

        if not stale:
            continue

        _admin_write_status(
            task_dir,
            state=TaskState.FAILED,
            attempt=snap.attempt,
            metadata={
                "failure_kind": reason,
                "heartbeat_age_s": age,
                "reconciled": reason,
            },
        )
        actions.append(
            {
                "task_index": snap.task_index,
                "action": "stale",
                "reason": reason,
                "heartbeat_age_s": age,
            }
        )
    return actions


def reconcile_experiment(
    experiment_dir: Path,
    *,
    timeout_seconds: float | None = None,
    grace_seconds: float = DEFAULT_STALE_GRACE_SECONDS,
    force_stale_seconds: float | None = None,
    include_slurm: bool = True,
    slurm_commands=None,
) -> list[dict[str, Any]]:
    """
    Run local stale reconciliation and optional Slurm scheduler reconciliation.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.
    timeout_seconds : float or None, optional
        Forwarded to :func:`reconcile_stale_tasks`.
    grace_seconds : float, optional
        Forwarded to :func:`reconcile_stale_tasks`.
    force_stale_seconds : float or None, optional
        Forwarded to :func:`reconcile_stale_tasks`.
    include_slurm : bool, optional
        When ``True`` and the sealed backend is ``slurm``, also reconcile
        against ``squeue``/``sacct``.
    slurm_commands : object or None, optional
        Optional injectable Slurm CLI runner for tests.

    Returns
    -------
    list of dict
        Combined reconciliation actions.
    """
    actions = reconcile_stale_tasks(
        experiment_dir,
        timeout_seconds=timeout_seconds,
        grace_seconds=grace_seconds,
        force_stale_seconds=force_stale_seconds,
    )
    if not include_slurm:
        return actions
    try:
        config = load_prepared_config(experiment_dir)
    except LifecycleError:
        return actions
    if config.execution.backend != "slurm":
        return actions
    from core.execution_core.backends.factory import create_execution_backend
    from core.execution_core.backends.slurm import SlurmArrayBackend
    from core.execution_core.backends.slurm_commands import SlurmCommandError

    backend = create_execution_backend(config)
    if not isinstance(backend, SlurmArrayBackend):
        return actions
    if slurm_commands is not None:
        backend.commands = slurm_commands
    try:
        actions.extend(backend.reconcile(experiment_dir))
    except SlurmCommandError as exc:
        # Query failure must not invent terminal states.
        actions.append(
            {
                "action": "slurm_query_failed",
                "message": str(exc),
            }
        )
    return actions


def collect_experiment(
    experiment_dir: Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """
    Run the workflow aggregator and optionally require full completion.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.
    require_complete : bool, optional
        When ``True``, raise if any task is non-terminal.

    Returns
    -------
    dict
        Aggregator summary document.

    Raises
    ------
    LifecycleError
        If ``require_complete`` is set and work remains unfinished.
    """
    reconcile_experiment(experiment_dir)
    status = list_task_statuses(experiment_dir)
    if require_complete and not status.complete:
        raise LifecycleError(
            f"experiment incomplete under {experiment_dir}: "
            f"counts={dict(status.counts)}"
        )
    paths = experiment_paths(experiment_dir)
    manifest = load_manifest(paths.manifest)
    registry = _import_registry()
    entry = registry.get_workflow(manifest.workflow_type)
    return entry.aggregator(paths.root)


def run_experiment_lifecycle(
    *,
    config_path: Path,
    output: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Local convenience: prepare, dispatch, then collect.

    Parameters
    ----------
    config_path : Path
        Experiment TOML path.
    output : Path or None, optional
        Explicit experiment root override.
    overwrite : bool, optional
        Allow replacing an existing sealed experiment.

    Returns
    -------
    dict
        Aggregation summary.

    Raises
    ------
    LifecycleError
        If the configured backend is not local (Slurm ``run`` only prepares
        and submits in a later phase) or collection fails.
    """
    paths = prepare_experiment(
        config_path=config_path,
        output=output,
        overwrite=overwrite,
    )
    config = load_prepared_config(paths.root)
    print_experiment_startup(paths.root, config=config)
    if config.execution.backend != "local":
        raise LifecycleError(
            "run requires backend='local'; for Slurm use prepare + dispatch "
            "then status/collect"
        )
    dispatch_experiment(paths.root)
    return collect_experiment(paths.root, require_complete=True)
