"""
Typed records for the generic experiment execution runtime.

These models are workflow-agnostic: optimisation and brute-force share the
same task identity, paths, and lifecycle vocabulary. Workflow-specific
detail belongs in ``TaskRecord.payload`` and in ``ioh_core.workflows``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class TaskState(str, Enum):
    """
    Lifecycle states for one generic task directory.

    Attributes
    ----------
    PENDING :
        Planner has created the task; nothing has claimed it yet.
    DISPATCHED :
        A backend reserved an attempt and handed the task to a worker process.
    RUNNING :
        The worker has started and owns the attempt directory.
    COMPLETED :
        Terminal success. Immutable once written.
    FAILED :
        Terminal failure reported by the worker.
    TIMED_OUT :
        Terminal timeout written by the local backend after the process is dead.
    CANCELLED :
        Terminal cancellation.
    """

    PENDING = "pending"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


# Legal directed edges for ``transition_task_status``.
ALLOWED_TASK_TRANSITIONS: Mapping[TaskState, frozenset[TaskState]] = {
    TaskState.PENDING: frozenset({TaskState.DISPATCHED}),
    TaskState.DISPATCHED: frozenset({TaskState.RUNNING}),
    TaskState.RUNNING: frozenset(
        {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.TIMED_OUT,
            TaskState.CANCELLED,
        }
    ),
    TaskState.COMPLETED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.TIMED_OUT: frozenset(),
    TaskState.CANCELLED: frozenset(),
}


TERMINAL_TASK_STATES: frozenset[TaskState] = frozenset(
    {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.TIMED_OUT,
        TaskState.CANCELLED,
    }
)


def format_task_id(task_index: int) -> str:
    """
    Return the stable generic task id ``task_{index:06d}``.

    Parameters
    ----------
    task_index : int
        Zero-based contiguous task index from the manifest.

    Returns
    -------
    str
        Folder / id string such as ``\"task_000007\"``.

    Raises
    ------
    ValueError
        If ``task_index`` is negative.
    """
    if task_index < 0:
        raise ValueError(f"task_index must be >= 0, got {task_index}")
    return f"task_{task_index:06d}"


@dataclass(frozen=True)
class TaskRecord:
    """
    One generic task entry as stored in the experiment manifest.

    The execution runtime schedules by ``task_index`` / ``task_id`` only.
    Workflow workers interpret ``kind`` and ``payload``.

    Attributes
    ----------
    task_index : int
        Zero-based contiguous index in the manifest task list.
    task_id : str
        Stable generic id ``task_{index:06d}``; must match ``task_index``.
    kind : str
        Coarse task type string understood by the workflow worker, e.g.
        ``\"optimization_run\"`` or ``\"brute_force_shard\"``. Distinguishes
        payload shapes when a manifest could theoretically mix kinds; the
        backend does not branch on it.
    payload : Mapping[str, Any]
        JSON-compatible workflow-specific specification. Examples:

        - optimisation: ``problem_id``, ``algorithm_name``, ``repetition``,
          ``random_seed``, …
        - brute-force shard: ``start_index``, ``stop_index``, ``shard_index``, …

        Descriptive identities belong here, not in ``task_id``.
    """

    task_index: int
    task_id: str
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = format_task_id(self.task_index)
        if self.task_id != expected:
            raise ValueError(
                f"task_id must be {expected!r} for task_index={self.task_index}, "
                f"got {self.task_id!r}"
            )
        # Copy so the frozen record does not alias a caller-owned mapping.
        object.__setattr__(self, "payload", dict(self.payload))


@dataclass(frozen=True)
class ExperimentManifest:
    """
    Immutable experiment description written at preparation time.

    Attributes
    ----------
    schema_version : int
        Manifest schema version; unknown versions must be rejected by loaders.
    experiment_id : str
        Directory name / human id under ``experiments/``.
    workflow_type : str
        Workflow selector such as ``\"optimization_runs\"`` or
        ``\"brute_force\"``.
    workflow_version : int
        Workflow contract version for forward compatibility.
    worker_module : str
        Import path of the worker module the backend invokes, e.g.
        ``\"core.ioh_core.workflows.optimization_runs.worker\"``.
    created_at : datetime
        Timezone-aware UTC timestamp of preparation.
    input_files : Mapping[str, str]
        Relative input paths mapped to SHA-256 hex digests.
    tasks : tuple of TaskRecord
        Contiguous task list; ``task_index`` values must be ``0..n-1``.
    """

    schema_version: int
    experiment_id: str
    workflow_type: str
    workflow_version: int
    worker_module: str
    created_at: datetime
    input_files: Mapping[str, str]
    tasks: tuple[TaskRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_files", dict(self.input_files))
        object.__setattr__(self, "tasks", tuple(self.tasks))
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware UTC")


@dataclass(frozen=True)
class ExperimentPaths:
    """
    Resolved directories under one experiment root.

    Attributes
    ----------
    root : Path
        Experiment root (``{output_root}/runs/<bucket>/{experiment_id}/``).
    inputs : Path
        Snapshotted CSV / input directory.
    submissions : Path
        Backend submission records directory.
    tasks : Path
        Parent of per-task directories ``task_XXXXXX/``.
    aggregate : Path
        Aggregation / publication outputs.
    """

    root: Path
    inputs: Path
    submissions: Path
    tasks: Path
    aggregate: Path

    @property
    def manifest(self) -> Path:
        """Path to immutable ``manifest.json``."""
        return self.root / "manifest.json"

    @property
    def environment(self) -> Path:
        """Path to ``environment.json`` (runtime expectations)."""
        return self.root / "environment.json"

    @property
    def source_config(self) -> Path:
        """Path to the copied source TOML config."""
        return self.root / "source-config.toml"


@dataclass(frozen=True)
class AttemptRef:
    """
    Pointer to one reserved attempt directory for a task.

    Attributes
    ----------
    task_dir : Path
        Parent task directory containing ``status.json`` and ``attempts/``.
    attempt : int
        One-based attempt index (folder name ``001``, ``002``, …).
    attempt_dir : Path
        Reserved attempt directory for logs and summaries.
    """

    task_dir: Path
    attempt: int
    attempt_dir: Path

    def __post_init__(self) -> None:
        if self.attempt < 1:
            raise ValueError(f"attempt must be >= 1, got {self.attempt}")


@dataclass(frozen=True)
class SubmissionRecord:
    """
    Backend submission metadata for a batch of dispatched tasks.

    Attributes
    ----------
    backend : str
        Backend name such as ``\"local\"`` or ``\"slurm\"``.
    submitted_at : datetime
        Timezone-aware UTC submission timestamp.
    task_indices : tuple of int
        Task indices included in this submission.
    metadata : Mapping[str, Any]
        Backend-specific fields (local PIDs, Slurm job id, script path, …).
    """

    backend: str
    submitted_at: datetime
    task_indices: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.submitted_at.tzinfo is None:
            raise ValueError("submitted_at must be timezone-aware UTC")
        object.__setattr__(self, "task_indices", tuple(self.task_indices))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ExecutionResources:
    """
    Resource and thread-limit knobs shared by local and Spool/Slurm backends.

    Attributes
    ----------
    max_parallel_tasks : int
        Maximum concurrently running task processes / array slots locally.
    timeout_seconds : float or None
        Optional wall-time limit per task in seconds; ``None`` means no limit.
    cpus_per_task : int
        CPUs requested or expected per task.
    thread_environment : Mapping[str, str]
        Environment variables that cap BLAS/OpenMP threads inside workers.
    """

    max_parallel_tasks: int = 1
    timeout_seconds: float | None = None
    cpus_per_task: int = 1
    thread_environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_parallel_tasks < 1:
            raise ValueError(
                f"max_parallel_tasks must be >= 1, got {self.max_parallel_tasks}"
            )
        if self.cpus_per_task < 1:
            raise ValueError(f"cpus_per_task must be >= 1, got {self.cpus_per_task}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be > 0 when set, got {self.timeout_seconds}"
            )
        object.__setattr__(self, "thread_environment", dict(self.thread_environment))
