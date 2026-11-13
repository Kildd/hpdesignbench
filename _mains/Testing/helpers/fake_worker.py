"""
Deterministic fake worker for local-backend concurrency and failure tests.

Modes:

- ``success`` — write a minimal valid ``summary.json`` and complete
- ``failure`` — write ``failure.json`` and exit nonzero
- ``sleep`` — sleep, then succeed (used for timeouts/concurrency)
- ``env`` — require injected thread-limit env vars, then succeed
- ``concurrency`` — hold an exclusive lock file while sleeping
- ``empty_success`` — exit 0 without writing ``summary.json``
- ``exit_137`` — exit with code 137 (OOM-like)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from core.execution_core.artifacts import (
    read_task_status,
    touch_attempt_heartbeat,
    transition_task_status,
    write_attempt_owner,
)
from core.execution_core.atomic_io import write_json_atomic
from core.execution_core.models import TaskState, format_task_id


def _task_dir(experiment_dir: Path, task_index: int) -> Path:
    """
    Return the task directory for ``task_index``.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root.
    task_index : int
        Zero-based task index.

    Returns
    -------
    Path
        Task directory path.
    """
    return Path(experiment_dir) / "tasks" / format_task_id(task_index)


def _attempt_dir(task_dir: Path, attempt: int) -> Path:
    """
    Return the attempt directory.

    Parameters
    ----------
    task_dir : Path
        Parent task directory.
    attempt : int
        One-based attempt index.

    Returns
    -------
    Path
        Attempt directory.
    """
    return Path(task_dir) / "attempts" / f"{attempt:03d}"


def _ensure_running(task_dir: Path, attempt: int) -> None:
    """
    Advance pending/dispatched status to ``running``.

    Parameters
    ----------
    task_dir : Path
        Task directory.
    attempt : int
        Attempt index.
    """
    status = read_task_status(task_dir)
    state = TaskState(status["state"])
    if state is TaskState.PENDING:
        transition_task_status(
            task_dir,
            expected={TaskState.PENDING},
            new_state=TaskState.DISPATCHED,
            attempt=attempt,
        )
        state = TaskState.DISPATCHED
    if state is TaskState.DISPATCHED:
        transition_task_status(
            task_dir,
            expected={TaskState.DISPATCHED},
            new_state=TaskState.RUNNING,
            attempt=attempt,
        )


def _claim_ownership(attempt_dir: Path) -> None:
    """
    Write local owner identity and an initial heartbeat.

    Parameters
    ----------
    attempt_dir : Path
        Reserved attempt directory.
    """
    write_attempt_owner(attempt_dir, backend="local", pid=os.getpid())
    touch_attempt_heartbeat(attempt_dir, pid=os.getpid())


def _write_summary(attempt_dir: Path, task_index: int) -> None:
    """
    Write a minimal valid worker summary document.

    Parameters
    ----------
    attempt_dir : Path
        Attempt directory.
    task_index : int
        Task index recorded in the summary.
    """
    write_json_atomic(
        attempt_dir / "summary.json",
        {
            "best_objective": float(task_index),
            "best_x": [0],
            "best_params": {"x": 0},
            "hits": 0,
            "misses": 1,
            "evaluations": 1,
            "wall_time_s": 0.0,
            "finite_best": True,
            "algorithm_name": "FAKE",
            "task_index": task_index,
        },
    )


def run_mode(
    *,
    experiment_dir: Path,
    task_index: int,
    attempt: int,
    mode: str,
    sleep_seconds: float,
    hold_file: Path | None,
) -> int:
    """
    Execute one fake-worker mode.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.
    task_index : int
        Task index.
    attempt : int
        Attempt index.
    mode : str
        Worker mode name.
    sleep_seconds : float
        Sleep duration for ``sleep`` / ``concurrency``.
    hold_file : Path or None
        Exclusive lock path for ``concurrency`` mode.

    Returns
    -------
    int
        Process exit code.
    """
    task_dir = _task_dir(experiment_dir, task_index)
    attempt_dir = _attempt_dir(task_dir, attempt)
    if not attempt_dir.is_dir():
        print(f"missing attempt dir {attempt_dir}", file=sys.stderr)
        return 2

    _ensure_running(task_dir, attempt)
    _claim_ownership(attempt_dir)

    if mode == "env":
        required = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        missing = [name for name in required if name not in os.environ]
        if missing:
            write_json_atomic(
                attempt_dir / "failure.json",
                {
                    "error_type": "EnvironmentError",
                    "message": f"missing {missing}",
                    "traceback": "",
                },
            )
            transition_task_status(
                task_dir,
                expected={TaskState.RUNNING},
                new_state=TaskState.FAILED,
                attempt=attempt,
            )
            return 1
        _write_summary(attempt_dir, task_index)
        transition_task_status(
            task_dir,
            expected={TaskState.RUNNING},
            new_state=TaskState.COMPLETED,
            attempt=attempt,
        )
        return 0

    if mode == "failure":
        write_json_atomic(
            attempt_dir / "failure.json",
            {
                "error_type": "RuntimeError",
                "message": "forced failure",
                "traceback": "forced failure",
            },
        )
        transition_task_status(
            task_dir,
            expected={TaskState.RUNNING},
            new_state=TaskState.FAILED,
            attempt=attempt,
        )
        return 1

    if mode == "concurrency":
        if hold_file is None:
            print("concurrency mode requires --hold-file", file=sys.stderr)
            return 2
        hold_file = Path(hold_file)
        try:
            fd = os.open(str(hold_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            write_json_atomic(
                attempt_dir / "failure.json",
                {
                    "error_type": "RuntimeError",
                    "message": "concurrency lock already held",
                    "traceback": "",
                },
            )
            transition_task_status(
                task_dir,
                expected={TaskState.RUNNING},
                new_state=TaskState.FAILED,
                attempt=attempt,
            )
            return 1
        try:
            os.write(fd, f"task-{task_index}\n".encode("utf-8"))
            time.sleep(sleep_seconds)
        finally:
            os.close(fd)
            try:
                hold_file.unlink()
            except OSError:
                pass
        _write_summary(attempt_dir, task_index)
        transition_task_status(
            task_dir,
            expected={TaskState.RUNNING},
            new_state=TaskState.COMPLETED,
            attempt=attempt,
        )
        return 0

    if mode == "empty_success":
        # Intentionally no summary.json — backend must treat this as failure.
        return 0

    if mode == "exit_137":
        write_json_atomic(
            attempt_dir / "failure.json",
            {
                "error_type": "SystemExit",
                "message": "simulated oom/kill",
                "traceback": "",
            },
        )
        return 137

    if mode in {"sleep", "success"}:
        if mode == "sleep":
            time.sleep(sleep_seconds)
        _write_summary(attempt_dir, task_index)
        transition_task_status(
            task_dir,
            expected={TaskState.RUNNING},
            new_state=TaskState.COMPLETED,
            attempt=attempt,
        )
        return 0

    print(f"unknown mode {mode!r}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry for the fake worker.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector excluding program name.

    Returns
    -------
    int
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Fake local-backend worker")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument(
        "--mode",
        default=os.environ.get("HDB_FAKE_WORKER_MODE", "success"),
        choices=(
            "success",
            "failure",
            "sleep",
            "env",
            "concurrency",
            "empty_success",
            "exit_137",
        ),
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=float(os.environ.get("HDB_FAKE_WORKER_SLEEP", "0.3")),
    )
    parser.add_argument("--hold-file", type=Path, default=None)
    args = parser.parse_args(argv)
    hold = args.hold_file
    if hold is None and os.environ.get("HDB_FAKE_WORKER_HOLD_FILE"):
        hold = Path(os.environ["HDB_FAKE_WORKER_HOLD_FILE"])
    return run_mode(
        experiment_dir=args.experiment_dir,
        task_index=args.task_index,
        attempt=args.attempt,
        mode=args.mode,
        sleep_seconds=args.sleep_seconds,
        hold_file=hold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
