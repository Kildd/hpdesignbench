"""
Experiment directory layout and task status ownership.

Creates the on-disk tree under an experiment root and owns atomic transitions
of each task's ``status.json``. Attempt directories are claimed with
``mkdir(exist_ok=False)`` so concurrent workers cannot reuse an attempt.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import os
import socket

from core.execution_core.atomic_io import read_json, write_json_atomic
from core.execution_core.models import (
    ALLOWED_TASK_TRANSITIONS,
    AttemptRef,
    ExperimentPaths,
    TaskState,
    format_task_id,
)


class TaskStatusError(RuntimeError):
    """
    Raised when a status read or transition is illegal.

    Typical causes: missing ``status.json``, immutable ``completed`` state,
    current state not in the caller's ``expected`` set, or an edge that is
    not in :data:`~core.execution_core.models.ALLOWED_TASK_TRANSITIONS`.
    """


def utc_now() -> datetime:
    """
    Return the current timezone-aware UTC timestamp.

    Returns
    -------
    datetime
        ``datetime.now(timezone.utc)``.
    """
    return datetime.now(timezone.utc)


def _iso_utc(moment: datetime) -> str:
    """
    Format a timezone-aware datetime as a UTC ISO-8601 string ending in ``Z``.

    Parameters
    ----------
    moment : datetime
        Timezone-aware timestamp.

    Returns
    -------
    str
        UTC timestamp such as ``\"2026-07-28T09:15:00.123456Z\"``.

    Raises
    ------
    ValueError
        If ``moment`` is naive (no ``tzinfo``).
    """
    if moment.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def create_experiment_layout(root: Path) -> ExperimentPaths:
    """
    Create the standard experiment directory tree under ``root``.

    Parameters
    ----------
    root : Path
        Experiment root directory (``experiments/{experiment_id}/``).

    Returns
    -------
    ExperimentPaths
        Resolved subdirectory paths (``inputs``, ``tasks``, …).
    """
    root = Path(root)
    paths = ExperimentPaths(
        root=root,
        inputs=root / "inputs",
        submissions=root / "submissions",
        tasks=root / "tasks",
        aggregate=root / "aggregate",
    )
    for directory in (
        paths.root,
        paths.inputs,
        paths.submissions,
        paths.tasks,
        paths.aggregate,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def create_task_layout(paths: ExperimentPaths, task_index: int) -> Path:
    """
    Create ``tasks/task_{index:06d}/`` with an initial ``pending`` status.

    Parameters
    ----------
    paths : ExperimentPaths
        Experiment path bundle from :func:`create_experiment_layout`.
    task_index : int
        Zero-based contiguous task index.

    Returns
    -------
    Path
        Task directory containing ``status.json`` and ``attempts/``.
    """
    task_id = format_task_id(task_index)
    task_dir = paths.tasks / task_id
    attempts_dir = task_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    status_path = task_dir / "status.json"
    if not status_path.exists():
        write_json_atomic(
            status_path,
            {
                "task_id": task_id,
                "task_index": task_index,
                "state": TaskState.PENDING.value,
                "attempt": 0,
                "updated_at": _iso_utc(utc_now()),
                "metadata": {},
            },
        )
    return task_dir


def reserve_attempt(task_dir: Path) -> AttemptRef:
    """
    Reserve the next attempt directory under ``task_dir/attempts/``.

    Uses ``mkdir(exist_ok=False)`` so two concurrent claims cannot share the
    same attempt index. Failed claims advance to the next free index.

    Parameters
    ----------
    task_dir : Path
        Task directory created by :func:`create_task_layout`.

    Returns
    -------
    AttemptRef
        Reserved one-based attempt index and directory.

    Raises
    ------
    TaskStatusError
        If attempt indices ``001``–``999`` are exhausted.
    """
    task_dir = Path(task_dir)
    attempts_dir = task_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    attempt = 1
    while attempt <= 999:
        attempt_dir = attempts_dir / f"{attempt:03d}"
        try:
            attempt_dir.mkdir(exist_ok=False)
        except FileExistsError:
            attempt += 1
            continue
        return AttemptRef(
            task_dir=task_dir,
            attempt=attempt,
            attempt_dir=attempt_dir,
        )
    raise TaskStatusError(
        f"exhausted attempt indices under {attempts_dir} (max 999)"
    )


def read_task_status(task_dir: Path) -> dict[str, Any]:
    """
    Read ``status.json`` for ``task_dir``.

    Parameters
    ----------
    task_dir : Path
        Task directory containing ``status.json``.

    Returns
    -------
    dict
        Status document with at least ``state``, ``attempt``, and
        ``updated_at``.

    Raises
    ------
    TaskStatusError
        If the file is missing or is not a JSON object.
    """
    status_path = Path(task_dir) / "status.json"
    if not status_path.exists():
        raise TaskStatusError(f"missing status.json in {task_dir}")
    payload = read_json(status_path)
    if not isinstance(payload, dict):
        raise TaskStatusError(f"status.json in {task_dir} is not an object")
    return payload


def transition_task_status(
    task_dir: Path,
    *,
    expected: set[TaskState],
    new_state: TaskState,
    attempt: int,
    metadata: dict | None = None,
) -> None:
    """
    Atomically transition a task status when the current state is allowed.

    Parameters
    ----------
    task_dir : Path
        Task directory containing ``status.json``.
    expected : set of TaskState
        Allowed current states for this writer (compare-and-set guard).
    new_state : TaskState
        Target state; must be a legal successor of the current state.
    attempt : int
        Attempt index associated with this transition (``0`` only while
        still pending before the first reservation).
    metadata : dict or None, optional
        Optional key/value pairs merged into the published status
        ``metadata`` object (e.g. exit code, error class).

    Raises
    ------
    TaskStatusError
        If the status is missing, ``completed`` (immutable), not in
        ``expected``, or the edge is not allowed.
    ValueError
        If ``expected`` is empty or ``attempt`` is negative.
    """
    if not expected:
        raise ValueError("expected must be a non-empty set of TaskState")
    if attempt < 0:
        raise ValueError(f"attempt must be >= 0, got {attempt}")

    current = read_task_status(task_dir)
    try:
        current_state = TaskState(current["state"])
    except (KeyError, ValueError) as exc:
        raise TaskStatusError(
            f"invalid state in status.json for {task_dir}: {current.get('state')!r}"
        ) from exc

    if current_state is TaskState.COMPLETED:
        raise TaskStatusError(
            f"completed status is immutable for {task_dir}"
        )
    if current_state not in expected:
        raise TaskStatusError(
            f"status for {task_dir} is {current_state.value!r}, "
            f"expected one of {{{', '.join(sorted(s.value for s in expected))}}}"
        )
    allowed = ALLOWED_TASK_TRANSITIONS[current_state]
    if new_state not in allowed:
        raise TaskStatusError(
            f"illegal transition {current_state.value!r} -> {new_state.value!r} "
            f"for {task_dir}"
        )

    merged_metadata: dict[str, Any] = {}
    existing_meta = current.get("metadata")
    if isinstance(existing_meta, dict):
        merged_metadata.update(existing_meta)
    if metadata:
        merged_metadata.update(metadata)

    write_json_atomic(
        Path(task_dir) / "status.json",
        {
            "task_id": current.get("task_id", Path(task_dir).name),
            "task_index": current.get("task_index"),
            "state": new_state.value,
            "attempt": attempt,
            "updated_at": _iso_utc(utc_now()),
            "metadata": merged_metadata,
        },
    )


def pid_is_alive(pid: int) -> bool:
    """
    Return whether a local process id appears to exist.

    Parameters
    ----------
    pid : int
        Candidate process id.

    Returns
    -------
    bool
        ``True`` when the process exists (or existence cannot be ruled out
        due to permissions); ``False`` when it is confirmed absent.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_attempt_owner(
    attempt_dir: Path,
    *,
    backend: str = "local",
    pid: int | None = None,
    hostname: str | None = None,
    slurm_job_id: str | None = None,
    slurm_task_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Atomically publish ``owner.json`` for a running attempt.

    Parameters
    ----------
    attempt_dir : Path
        Reserved attempt directory.
    backend : str, optional
        Owning backend name (``\"local\"`` or ``\"slurm\"``).
    pid : int or None, optional
        Local process id when known.
    hostname : str or None, optional
        Host that owns the attempt; defaults to ``socket.gethostname()``.
    slurm_job_id : str or None, optional
        Slurm job id when applicable.
    slurm_task_id : str or None, optional
        Slurm array task id when applicable.
    extra : dict or None, optional
        Additional JSON-compatible fields merged into the document.

    Returns
    -------
    dict
        Published owner document.
    """
    attempt_dir = Path(attempt_dir)
    document: dict[str, Any] = {
        "backend": str(backend),
        "pid": int(pid) if pid is not None else None,
        "hostname": hostname if hostname is not None else socket.gethostname(),
        "slurm_job_id": slurm_job_id,
        "slurm_task_id": slurm_task_id,
        "started_at": _iso_utc(utc_now()),
    }
    if extra:
        document.update(dict(extra))
    write_json_atomic(attempt_dir / "owner.json", document)
    return document


def touch_attempt_heartbeat(
    attempt_dir: Path,
    *,
    pid: int | None = None,
    updated_at: datetime | None = None,
) -> dict[str, Any]:
    """
    Atomically publish / refresh ``heartbeat.json`` for an attempt.

    Parameters
    ----------
    attempt_dir : Path
        Reserved attempt directory.
    pid : int or None, optional
        Process id recorded on the heartbeat; defaults to ``os.getpid()``.
    updated_at : datetime or None, optional
        Explicit heartbeat timestamp (for tests); defaults to now.

    Returns
    -------
    dict
        Published heartbeat document.
    """
    attempt_dir = Path(attempt_dir)
    moment = updated_at if updated_at is not None else utc_now()
    document = {
        "pid": int(pid) if pid is not None else int(os.getpid()),
        "updated_at": _iso_utc(moment),
    }
    write_json_atomic(attempt_dir / "heartbeat.json", document)
    return document


def read_attempt_owner(attempt_dir: Path) -> dict[str, Any] | None:
    """
    Read ``owner.json`` when present.

    Parameters
    ----------
    attempt_dir : Path
        Attempt directory.

    Returns
    -------
    dict or None
        Owner document, or ``None`` when missing/invalid.
    """
    path = Path(attempt_dir) / "owner.json"
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def read_attempt_heartbeat(attempt_dir: Path) -> dict[str, Any] | None:
    """
    Read ``heartbeat.json`` when present.

    Parameters
    ----------
    attempt_dir : Path
        Attempt directory.

    Returns
    -------
    dict or None
        Heartbeat document, or ``None`` when missing/invalid.
    """
    path = Path(attempt_dir) / "heartbeat.json"
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None
