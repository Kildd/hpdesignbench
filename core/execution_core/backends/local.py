"""
Bounded local subprocess backend for generic experiment tasks.

Runs worker modules with ``subprocess.Popen`` under a hard concurrency cap.
Uses a single-threaded poll loop (no threads, pools, or ``shell=True``).
"""
from __future__ import annotations

import importlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TextIO

from core.execution_core.artifacts import (
    read_task_status,
    reserve_attempt,
    transition_task_status,
    utc_now,
)
from core.execution_core.atomic_io import read_json, write_json_atomic
from core.execution_core.models import (
    AttemptRef,
    SubmissionRecord,
    TaskState,
    format_task_id,
)

_IS_WINDOWS = os.name == "nt"


class LocalBackendError(RuntimeError):
    """
    Raised for illegal local dispatch requests.

    Typical causes: duplicate dispatch of a non-pending task, missing task
    directories, or an empty task-index list.
    """


def _worker_import_roots(worker_module: str) -> list[str]:
    """
    Return filesystem roots that make ``worker_module`` importable.

    Workers run with ``cwd`` under the experiment tree, so the repo root is not
    automatically on ``sys.path``. Prepending the parent of the worker's
    top-level package keeps both installed ``core.*`` workers and test-only
    ``_mains.*`` workers resolvable.

    Parameters
    ----------
    worker_module : str
        Dotted module path passed to ``python -m``.

    Returns
    -------
    list of str
        Absolute directory paths to prepend to ``PYTHONPATH``.
    """
    top_name = worker_module.split(".", 1)[0]
    if not top_name.isidentifier():
        return []
    try:
        module = importlib.import_module(top_name)
    except ImportError:
        return []

    roots: list[str] = []
    path_entries = getattr(module, "__path__", None)
    if path_entries is not None:
        for entry in path_entries:
            roots.append(str(Path(entry).resolve().parent))
    elif getattr(module, "__file__", None):
        roots.append(str(Path(module.__file__).resolve().parent.parent))

    unique: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique.append(root)
    return unique


def _env_with_worker_pythonpath(
    env: Mapping[str, str],
    worker_module: str,
) -> dict[str, str]:
    """
    Return a copy of ``env`` with worker import roots prepended to PYTHONPATH.

    Parameters
    ----------
    env : mapping of str to str
        Base process environment.
    worker_module : str
        Dotted worker module path.

    Returns
    -------
    dict of str to str
        Environment suitable for ``subprocess.Popen``.
    """
    merged = dict(env)
    roots = _worker_import_roots(worker_module)
    if not roots:
        return merged
    existing = merged.get("PYTHONPATH", "")
    parts = list(roots)
    if existing:
        parts.append(existing)
    merged["PYTHONPATH"] = os.pathsep.join(parts)
    return merged


@dataclass
class _ActiveTask:
    """
    Runtime state for one locally running worker process.

    Attributes
    ----------
    task_index : int
        Zero-based task index.
    attempt : AttemptRef
        Reserved attempt ownership.
    process : subprocess.Popen
        Active worker process handle.
    started_at : float
        ``time.monotonic()`` start timestamp.
    stdout_handle : TextIO
        Open stdout log file handle.
    stderr_handle : TextIO
        Open stderr log file handle.
    timed_out : bool
        Set when the wall-time limit triggered termination.
    """

    task_index: int
    attempt: AttemptRef
    process: subprocess.Popen
    started_at: float
    stdout_handle: TextIO
    stderr_handle: TextIO
    timed_out: bool = False


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
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class LocalSubprocessBackend:
    """
    Local FIFO subprocess scheduler with bounded parallelism.

    Parameters
    ----------
    max_parallel_tasks : int
        Maximum concurrently running worker processes.
    timeout_seconds : float or None, optional
        Per-task wall-time limit; ``None`` disables timeouts.
    python_executable : str, optional
        Interpreter used to launch ``python -m <worker_module>``.
    thread_environment : Mapping[str, str], optional
        Environment variables overwritten in each worker.
    poll_interval_seconds : float, optional
        Sleep between poll iterations.
    termination_grace_seconds : float, optional
        Wait after polite terminate before forced kill.
    """

    def __init__(
        self,
        *,
        max_parallel_tasks: int = 1,
        timeout_seconds: float | None = None,
        python_executable: str | None = None,
        thread_environment: Mapping[str, str] | None = None,
        poll_interval_seconds: float = 0.05,
        termination_grace_seconds: float = 1.0,
    ) -> None:
        if max_parallel_tasks < 1:
            raise ValueError("max_parallel_tasks must be >= 1")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0 when set")
        self.max_parallel_tasks = int(max_parallel_tasks)
        self.timeout_seconds = timeout_seconds
        self.python_executable = str(python_executable or sys.executable)
        self.thread_environment = dict(thread_environment or {})
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.termination_grace_seconds = float(termination_grace_seconds)

    def _idle(self, seconds: float) -> None:
        """
        Sleep helper used by the poll loop (patchable in tests).

        Parameters
        ----------
        seconds : float
            Idle duration.
        """
        time.sleep(seconds)

    def dispatch(
        self,
        *,
        experiment_dir: Path,
        task_indices: list[int],
        worker_module: str,
    ) -> list[SubmissionRecord]:
        """
        Run ``task_indices`` locally under the concurrency cap.

        Parameters
        ----------
        experiment_dir : Path
            Prepared experiment root.
        task_indices : list of int
            Tasks to run in FIFO order.
        worker_module : str
            Worker launched as ``python -m <worker_module>``.

        Returns
        -------
        list of SubmissionRecord
            One submission covering the dispatched batch.

        Raises
        ------
        LocalBackendError
            If the request is empty or a task is not pending.
        KeyboardInterrupt
            After active workers have been terminated.
        """
        if not task_indices:
            raise LocalBackendError("task_indices must be non-empty")
        experiment_dir = Path(experiment_dir).expanduser().resolve()
        pending = list(task_indices)
        active: dict[int, _ActiveTask] = {}
        outcomes: dict[int, dict[str, Any]] = {}
        submitted_at = utc_now()
        pids: dict[str, int] = {}

        try:
            while pending or active:
                while pending and len(active) < self.max_parallel_tasks:
                    task_index = pending.pop(0)
                    active_task = self._start_task(
                        experiment_dir=experiment_dir,
                        task_index=task_index,
                        worker_module=worker_module,
                    )
                    active[task_index] = active_task
                    pids[str(task_index)] = int(active_task.process.pid)

                finished = []
                now = time.monotonic()
                for task_index, state in list(active.items()):
                    if (
                        self.timeout_seconds is not None
                        and not state.timed_out
                        and (now - state.started_at) >= self.timeout_seconds
                    ):
                        self._terminate_process_tree(state.process)
                        state.timed_out = True
                        # Keep polling until death is confirmed.
                    code = state.process.poll()
                    if code is None:
                        continue
                    finished.append(task_index)
                    outcomes[task_index] = self._finalize_task(
                        experiment_dir=experiment_dir,
                        state=state,
                        returncode=int(code),
                    )

                for task_index in finished:
                    active.pop(task_index, None)

                if active or pending:
                    self._idle(self.poll_interval_seconds)
        except KeyboardInterrupt:
            for state in list(active.values()):
                self._terminate_process_tree(state.process)
                try:
                    state.process.wait(timeout=self.termination_grace_seconds + 1.0)
                except Exception:
                    pass
                self._close_handles(state)
                self._mark_cancelled(state.attempt.task_dir, state.attempt.attempt)
            raise

        record = SubmissionRecord(
            backend="local",
            submitted_at=submitted_at,
            task_indices=tuple(task_indices),
            metadata={
                "python_executable": self.python_executable,
                "worker_module": worker_module,
                "max_parallel_tasks": self.max_parallel_tasks,
                "timeout_seconds": self.timeout_seconds,
                "pids": pids,
                "outcomes": outcomes,
                "finished_at": _iso_utc(utc_now()),
            },
        )
        submissions = experiment_dir / "submissions"
        submissions.mkdir(parents=True, exist_ok=True)
        stamp = submitted_at.strftime("%Y%m%dT%H%M%S%fZ")
        write_json_atomic(
            submissions / f"local_{stamp}.json",
            {
                "backend": record.backend,
                "submitted_at": _iso_utc(record.submitted_at),
                "task_indices": list(record.task_indices),
                "metadata": dict(record.metadata),
            },
        )
        return [record]

    def _start_task(
        self,
        *,
        experiment_dir: Path,
        task_index: int,
        worker_module: str,
    ) -> _ActiveTask:
        """
        Reserve an attempt, mark dispatched, and launch one worker process.

        Parameters
        ----------
        experiment_dir : Path
            Experiment root.
        task_index : int
            Task to launch.
        worker_module : str
            Worker module path.

        Returns
        -------
        _ActiveTask
            Active process bookkeeping record.
        """
        task_dir = experiment_dir / "tasks" / format_task_id(task_index)
        if not task_dir.is_dir():
            raise LocalBackendError(f"missing task directory: {task_dir}")
        status = read_task_status(task_dir)
        state = TaskState(status["state"])
        if state is not TaskState.PENDING:
            raise LocalBackendError(
                f"duplicate/illegal dispatch for task {task_index}: "
                f"state is {state.value!r}, expected pending"
            )

        attempt = reserve_attempt(task_dir)
        transition_task_status(
            task_dir,
            expected={TaskState.PENDING},
            new_state=TaskState.DISPATCHED,
            attempt=attempt.attempt,
        )

        stdout_path = attempt.attempt_dir / "stdout.log"
        stderr_path = attempt.attempt_dir / "stderr.log"
        stdout_handle = open(stdout_path, "w", encoding="utf-8")
        stderr_handle = open(stderr_path, "w", encoding="utf-8")

        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in self.thread_environment.items()})
        env = _env_with_worker_pythonpath(env, worker_module)

        cwd = experiment_dir / ".worker_cwd"
        cwd.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_executable,
            "-m",
            worker_module,
            "--experiment-dir",
            str(experiment_dir),
            "--task-index",
            str(task_index),
            "--attempt",
            str(attempt.attempt),
        ]
        popen_kwargs: dict[str, Any] = {
            "args": cmd,
            "cwd": str(cwd),
            "env": env,
            "stdout": stdout_handle,
            "stderr": stderr_handle,
            "shell": False,
        }
        if _IS_WINDOWS:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(**popen_kwargs)
        return _ActiveTask(
            task_index=task_index,
            attempt=attempt,
            process=process,
            started_at=time.monotonic(),
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
        )

    def _finalize_task(
        self,
        *,
        experiment_dir: Path,
        state: _ActiveTask,
        returncode: int,
    ) -> dict[str, Any]:
        """
        Close logs and map process death to an infrastructure outcome.

        Parameters
        ----------
        experiment_dir : Path
            Experiment root.
        state : _ActiveTask
            Finished active task.
        returncode : int
            Process exit code.

        Returns
        -------
        dict
            Outcome metadata stored on the submission record.
        """
        self._close_handles(state)
        summary_path = state.attempt.attempt_dir / "summary.json"
        failure_path = state.attempt.attempt_dir / "failure.json"
        status = read_task_status(state.attempt.task_dir)
        current = TaskState(status["state"])

        if state.timed_out:
            self._ensure_terminal(
                state.attempt.task_dir,
                current=current,
                attempt=state.attempt.attempt,
                terminal=TaskState.TIMED_OUT,
                metadata={"failure_kind": "timeout", "returncode": returncode},
            )
            return {
                "failure_kind": "timeout",
                "returncode": returncode,
                "state": TaskState.TIMED_OUT.value,
            }

        if returncode == 0:
            if not summary_path.is_file():
                kind = "missing_summary"
                self._ensure_failed(
                    state.attempt.task_dir,
                    current=current,
                    attempt=state.attempt.attempt,
                    metadata={
                        "failure_kind": kind,
                        "returncode": returncode,
                    },
                )
                return {"failure_kind": kind, "returncode": returncode}
            try:
                summary = read_json(summary_path)
                if not isinstance(summary, dict) or "best_objective" not in summary:
                    raise ValueError("summary missing best_objective")
            except Exception as exc:
                kind = "invalid_summary"
                self._ensure_failed(
                    state.attempt.task_dir,
                    current=current,
                    attempt=state.attempt.attempt,
                    metadata={
                        "failure_kind": kind,
                        "returncode": returncode,
                        "message": str(exc),
                    },
                )
                return {"failure_kind": kind, "returncode": returncode}
            return {
                "failure_kind": None,
                "returncode": 0,
                "state": TaskState.COMPLETED.value,
                "summary": str(summary_path),
            }

        kind = self._classify_nonzero_exit(returncode)
        self._ensure_failed(
            state.attempt.task_dir,
            current=current,
            attempt=state.attempt.attempt,
            metadata={
                "failure_kind": kind,
                "returncode": returncode,
                "failure": str(failure_path) if failure_path.exists() else None,
            },
        )
        return {"failure_kind": kind, "returncode": returncode}

    @staticmethod
    def _classify_nonzero_exit(returncode: int) -> str:
        """
        Map a nonzero exit code to an infrastructure failure kind.

        Parameters
        ----------
        returncode : int
            Process exit code.

        Returns
        -------
        str
            Failure kind token.
        """
        if returncode < 0:
            return "native_crash"
        if returncode == 137:
            return "oom_or_kill"
        return "nonzero_exit"

    def _ensure_failed(
        self,
        task_dir: Path,
        *,
        current: TaskState,
        attempt: int,
        metadata: dict[str, Any],
    ) -> None:
        """
        Ensure a task ends in ``failed`` if the worker did not already finish.

        Parameters
        ----------
        task_dir : Path
            Task directory.
        current : TaskState
            Last observed status.
        attempt : int
            Attempt index.
        metadata : dict
            Failure metadata merged into status.
        """
        if current in {TaskState.COMPLETED, TaskState.FAILED, TaskState.TIMED_OUT, TaskState.CANCELLED}:
            return
        self._ensure_terminal(
            task_dir,
            current=current,
            attempt=attempt,
            terminal=TaskState.FAILED,
            metadata=metadata,
        )

    def _ensure_terminal(
        self,
        task_dir: Path,
        *,
        current: TaskState,
        attempt: int,
        terminal: TaskState,
        metadata: dict[str, Any],
    ) -> None:
        """
        Advance a non-terminal task to ``terminal`` via legal edges.

        Parameters
        ----------
        task_dir : Path
            Task directory.
        current : TaskState
            Current state.
        attempt : int
            Attempt index.
        terminal : TaskState
            Desired terminal state.
        metadata : dict
            Status metadata.
        """
        if current is terminal:
            return
        if current is TaskState.PENDING:
            transition_task_status(
                task_dir,
                expected={TaskState.PENDING},
                new_state=TaskState.DISPATCHED,
                attempt=attempt,
            )
            current = TaskState.DISPATCHED
        if current is TaskState.DISPATCHED:
            transition_task_status(
                task_dir,
                expected={TaskState.DISPATCHED},
                new_state=TaskState.RUNNING,
                attempt=attempt,
            )
            current = TaskState.RUNNING
        if current is TaskState.RUNNING:
            transition_task_status(
                task_dir,
                expected={TaskState.RUNNING},
                new_state=terminal,
                attempt=attempt,
                metadata=metadata,
            )

    def _mark_cancelled(self, task_dir: Path, attempt: int) -> None:
        """
        Best-effort cancellation mark after Ctrl+C cleanup.

        Parameters
        ----------
        task_dir : Path
            Task directory.
        attempt : int
            Attempt index.
        """
        try:
            status = read_task_status(task_dir)
            current = TaskState(status["state"])
            if current in {
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.TIMED_OUT,
                TaskState.CANCELLED,
            }:
                return
            self._ensure_terminal(
                task_dir,
                current=current,
                attempt=attempt,
                terminal=TaskState.CANCELLED,
                metadata={"failure_kind": "interrupted"},
            )
        except Exception:
            pass

    def _terminate_process_tree(self, process: subprocess.Popen) -> None:
        """
        Politely stop a worker process group, then escalate to forced kill.

        Parameters
        ----------
        process : subprocess.Popen
            Worker process started by this backend.
        """
        if process.poll() is not None:
            return
        pid = process.pid
        try:
            if _IS_WINDOWS:
                process.terminate()
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass

        deadline = time.monotonic() + self.termination_grace_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.05)

        try:
            if _IS_WINDOWS:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                )
                if process.poll() is None:
                    process.kill()
            else:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    @staticmethod
    def _close_handles(state: _ActiveTask) -> None:
        """
        Close redirected stdout/stderr handles for a finished task.

        Parameters
        ----------
        state : _ActiveTask
            Active task whose log handles should be closed.
        """
        for handle in (state.stdout_handle, state.stderr_handle):
            try:
                handle.close()
            except Exception:
                pass
