"""
Direct Slurm array backend for generic experiment tasks.

Generates transparent Bash array scripts and submits them with ``sbatch``.
Does not wait for completion; Phase A10 reconciles scheduler state.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.execution_core.artifacts import (
    read_task_status,
    reserve_attempt,
    transition_task_status,
    utc_now,
)
from core.execution_core.atomic_io import write_json_atomic, write_text_atomic
from core.execution_core.backends.slurm_commands import (
    SlurmCommandError,
    SlurmCommands,
    SubprocessSlurmCommands,
)
from core.execution_core.models import AttemptRef, SubmissionRecord, TaskState, format_task_id


class SyncSlurmUnavailable(RuntimeError):
    """
    Raised when Slurm is unavailable before any task state mutation.

    Callers can treat this as a safe pre-flight failure: no attempts were
    reserved and no task left ``pending``.
    """


class SlurmBackendError(RuntimeError):
    """
    Raised for illegal Slurm dispatch requests or generation failures.

    Typical causes: empty task lists, non-pending tasks, or incomplete
    Slurm configuration.
    """


def _bash_quote_path(path: str | Path) -> str:
    """
    Shell-quote a filesystem path for generated Linux Bash scripts.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path. Backslashes are normalised to ``/`` so
        Windows generation still emits POSIX-style Slurm scripts.

    Returns
    -------
    str
        ``shlex.quote``d path using forward slashes.
    """
    text = str(path).replace("\\", "/")
    return shlex.quote(text)


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


def chunk_indices(indices: Sequence[int], maximum_array_size: int) -> list[list[int]]:
    """
    Split task indices into contiguous chunks of at most ``maximum_array_size``.

    Parameters
    ----------
    indices : sequence of int
        Global task indices in dispatch order.
    maximum_array_size : int
        Maximum length of one Slurm array.

    Returns
    -------
    list of list of int
        Chunks; never groups multiple tasks into one array element.

    Raises
    ------
    ValueError
        If ``maximum_array_size`` is less than 1.
    """
    if maximum_array_size < 1:
        raise ValueError(f"maximum_array_size must be >= 1, got {maximum_array_size}")
    values = list(indices)
    if not values:
        return []
    chunks: list[list[int]] = []
    for start in range(0, len(values), maximum_array_size):
        chunks.append(values[start : start + maximum_array_size])
    return chunks


def next_submission_directory(submissions_root: Path) -> Path:
    """
    Return the next free ``submission_NNN`` directory path.

    Parameters
    ----------
    submissions_root : Path
        Experiment ``submissions/`` directory.

    Returns
    -------
    Path
        Path such as ``submissions/submission_001`` (not yet created).
    """
    submissions_root = Path(submissions_root)
    submissions_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while index <= 999_999:
        candidate = submissions_root / f"submission_{index:03d}"
        if not candidate.exists():
            return candidate
        index += 1
    raise SlurmBackendError(f"exhausted submission indices under {submissions_root}")


def render_task_map_tsv(rows: Sequence[tuple[int, int, int]]) -> str:
    """
    Render a ``task-map.tsv`` body.

    Parameters
    ----------
    rows : sequence of tuple
        ``(local_array_index, task_index, attempt)`` rows.

    Returns
    -------
    str
        TSV text including a header line.
    """
    lines = ["local_array_index\ttask_index\tattempt"]
    for local_index, task_index, attempt in rows:
        lines.append(f"{local_index}\t{task_index}\t{attempt}")
    return "\n".join(lines) + "\n"


def render_job_script(
    *,
    experiment_dir: Path,
    worker_module: str,
    python_executable: Path,
    submission_dir: Path,
    task_map_path: Path,
    partition: str,
    account: str,
    memory: str,
    time_limit: str,
    cpus_per_task: int,
    array_concurrency: int,
    array_count: int,
    thread_environment: Mapping[str, str],
) -> str:
    """
    Render a Bash Slurm array script for one submission chunk.

    Parameters
    ----------
    experiment_dir : Path
        Shared experiment root visible to compute nodes.
    worker_module : str
        Worker launched as ``python -m <worker_module>``.
    python_executable : Path
        Absolute cluster Python interpreter.
    submission_dir : Path
        Directory that will hold wrapper logs.
    task_map_path : Path
        Path to ``task-map.tsv``.
    partition : str
        Slurm partition.
    account : str
        Slurm account (omitted when blank).
    memory : str
        Memory request.
    time_limit : str
        Wall-time limit.
    cpus_per_task : int
        CPUs per array element.
    array_concurrency : int
        ``%N`` concurrency throttle.
    array_count : int
        Number of array elements (``0 .. array_count-1``).
    thread_environment : Mapping[str, str]
        Thread-limit environment exports.

    Returns
    -------
    str
        Bash script text.

    Raises
    ------
    ValueError
        If ``array_count`` is less than 1.
    """
    if array_count < 1:
        raise ValueError(f"array_count must be >= 1, got {array_count}")

    exp_q = _bash_quote_path(experiment_dir)
    py_q = _bash_quote_path(python_executable)
    map_q = _bash_quote_path(task_map_path)
    worker_q = shlex.quote(str(worker_module))
    out_q = _bash_quote_path(Path(submission_dir) / "wrapper_%a.out")
    err_q = _bash_quote_path(Path(submission_dir) / "wrapper_%a.err")

    last_index = array_count - 1
    array_spec = f"0-{last_index}%{int(array_concurrency)}"

    directives = [
        "#!/bin/bash",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --cpus-per-task={int(cpus_per_task)}",
        f"#SBATCH --array={array_spec}",
        f"#SBATCH --output={out_q}",
        f"#SBATCH --error={err_q}",
    ]
    if account.strip():
        directives.insert(2, f"#SBATCH --account={account.strip()}")

    export_lines = ['export PYTHONUNBUFFERED="1"']
    for key, value in thread_environment.items():
        export_lines.append(f"export {key}={shlex.quote(str(value))}")

    body = f"""\
set -euo pipefail

EXPERIMENT_DIR={exp_q}
PYTHON={py_q}
WORKER_MODULE={worker_q}
TASK_MAP={map_q}
LOCAL_INDEX="${{SLURM_ARRAY_TASK_ID}}"

# Resolve global task index and attempt from the sealed task map.
mapfile -t MAP_ROW < <(awk -F '\\t' -v idx="$LOCAL_INDEX" 'NR>1 && $1==idx {{print $2; print $3; exit}}' "$TASK_MAP")
if [[ ${{#MAP_ROW[@]}} -ne 2 ]]; then
  echo "task-map lookup failed for local array index $LOCAL_INDEX" >&2
  exit 2
fi
TASK_INDEX="${{MAP_ROW[0]}}"
ATTEMPT="${{MAP_ROW[1]}}"

TASK_ID=$(printf 'task_%06d' "$TASK_INDEX")
ATTEMPT_DIR="$EXPERIMENT_DIR/tasks/$TASK_ID/attempts/$(printf '%03d' "$ATTEMPT")"
mkdir -p "$ATTEMPT_DIR"

# Never install packages on compute nodes. Use the configured interpreter only.
"$PYTHON" -m "$WORKER_MODULE" \\
  --experiment-dir "$EXPERIMENT_DIR" \\
  --task-index "$TASK_INDEX" \\
  --attempt "$ATTEMPT" \\
  >"$ATTEMPT_DIR/stdout.log" 2>"$ATTEMPT_DIR/stderr.log"
"""
    return "\n".join(directives + [""] + export_lines + ["", body])


@dataclass(frozen=True)
class _PreparedMapping:
    """
    One reserved task→attempt mapping ready for a Slurm array element.

    Attributes
    ----------
    task_index : int
        Global task index.
    attempt : AttemptRef
        Reserved attempt ownership.
    """

    task_index: int
    attempt: AttemptRef


class SlurmArrayBackend:
    """
    Generate and submit direct Slurm arrays for pending experiment tasks.

    Parameters
    ----------
    python_executable : str
        Absolute cluster interpreter path written into job scripts.
    partition : str
        Slurm partition name.
    account : str
        Slurm account (may be empty).
    memory : str
        Memory request such as ``\"8G\"``.
    time_limit : str
        Wall-time limit such as ``\"00:45:00\"``.
    cpus_per_task : int
        CPUs requested per array element.
    array_concurrency : int
        Maximum concurrently running array tasks.
    maximum_array_size : int
        Upper bound on one generated array length.
    thread_environment : Mapping[str, str], optional
        Thread-limit environment exports inside job scripts.
    commands : SlurmCommands or None, optional
        Injectable Slurm CLI runner; defaults to subprocess-backed commands.
    """

    def __init__(
        self,
        *,
        python_executable: str,
        partition: str,
        account: str = "",
        memory: str = "8G",
        time_limit: str = "00:45:00",
        cpus_per_task: int = 1,
        array_concurrency: int = 64,
        maximum_array_size: int = 1000,
        thread_environment: Mapping[str, str] | None = None,
        commands: SlurmCommands | None = None,
    ) -> None:
        if not str(python_executable).strip():
            raise SlurmBackendError("python_executable must be a non-empty absolute path")
        if not str(partition).strip():
            raise SlurmBackendError("partition must be a non-empty string")
        if cpus_per_task < 1:
            raise SlurmBackendError(f"cpus_per_task must be >= 1, got {cpus_per_task}")
        if array_concurrency < 1:
            raise SlurmBackendError(
                f"array_concurrency must be >= 1, got {array_concurrency}"
            )
        if maximum_array_size < 1:
            raise SlurmBackendError(
                f"maximum_array_size must be >= 1, got {maximum_array_size}"
            )
        self.python_executable = str(Path(python_executable).expanduser())
        self.partition = str(partition).strip()
        self.account = str(account or "")
        self.memory = str(memory).strip()
        self.time_limit = str(time_limit).strip()
        self.cpus_per_task = int(cpus_per_task)
        self.array_concurrency = int(array_concurrency)
        self.maximum_array_size = int(maximum_array_size)
        self.thread_environment = dict(thread_environment or {})
        self.commands: SlurmCommands = commands or SubprocessSlurmCommands()

    def dispatch(
        self,
        *,
        experiment_dir: Path,
        task_indices: list[int],
        worker_module: str,
    ) -> list[SubmissionRecord]:
        """
        Reserve attempts, generate array scripts, and submit with ``sbatch``.

        Returns immediately after submission; does not wait for workers.

        Parameters
        ----------
        experiment_dir : Path
            Prepared experiment root.
        task_indices : list of int
            Pending task indices (may be noncontiguous for retries).
        worker_module : str
            Worker module import path.

        Returns
        -------
        list of SubmissionRecord
            One record per generated/submitted array chunk.

        Raises
        ------
        SlurmBackendError
            If the request is empty, tasks are not pending, or Slurm is
            unavailable before any task state mutation.
        SlurmCommandError
            If ``sbatch`` fails after scripts were generated.
        """
        if not task_indices:
            raise SlurmBackendError("task_indices must be non-empty")
        experiment_dir = Path(experiment_dir).expanduser().resolve()
        if not worker_module.strip():
            raise SlurmBackendError("worker_module must be a non-empty string")

        # Availability check must happen before any task state mutation.
        if not self.commands.available():
            raise SyncSlurmUnavailable(
                "sbatch is not available; refusing to mutate task state"
            )

        for task_index in task_indices:
            task_dir = experiment_dir / "tasks" / format_task_id(task_index)
            if not task_dir.is_dir():
                raise SlurmBackendError(f"missing task directory: {task_dir}")
            state = TaskState(read_task_status(task_dir)["state"])
            if state is not TaskState.PENDING:
                raise SlurmBackendError(
                    f"duplicate/illegal dispatch for task {task_index}: "
                    f"state is {state.value!r}, expected pending"
                )

        chunks = chunk_indices(task_indices, self.maximum_array_size)
        records: list[SubmissionRecord] = []
        for chunk in chunks:
            records.append(
                self._dispatch_chunk(
                    experiment_dir=experiment_dir,
                    task_indices=chunk,
                    worker_module=worker_module,
                )
            )
        return records

    def _dispatch_chunk(
        self,
        *,
        experiment_dir: Path,
        task_indices: list[int],
        worker_module: str,
    ) -> SubmissionRecord:
        """
        Reserve, generate, and submit one array chunk.

        Parameters
        ----------
        experiment_dir : Path
            Experiment root.
        task_indices : list of int
            Task indices for this chunk.
        worker_module : str
            Worker module path.

        Returns
        -------
        SubmissionRecord
            Submission metadata including Slurm job id.
        """
        mappings: list[_PreparedMapping] = []
        for task_index in task_indices:
            task_dir = experiment_dir / "tasks" / format_task_id(task_index)
            attempt = reserve_attempt(task_dir)
            transition_task_status(
                task_dir,
                expected={TaskState.PENDING},
                new_state=TaskState.DISPATCHED,
                attempt=attempt.attempt,
            )
            mappings.append(_PreparedMapping(task_index=task_index, attempt=attempt))

        submissions_root = experiment_dir / "submissions"
        submission_dir = next_submission_directory(submissions_root)
        submission_dir.mkdir(parents=True, exist_ok=False)

        rows = [
            (local_index, mapping.task_index, mapping.attempt.attempt)
            for local_index, mapping in enumerate(mappings)
        ]
        task_map_path = submission_dir / "task-map.tsv"
        write_text_atomic(task_map_path, render_task_map_tsv(rows))

        job_script_path = submission_dir / "job.sh"
        script = render_job_script(
            experiment_dir=experiment_dir,
            worker_module=worker_module,
            python_executable=Path(self.python_executable),
            submission_dir=submission_dir,
            task_map_path=task_map_path,
            partition=self.partition,
            account=self.account,
            memory=self.memory,
            time_limit=self.time_limit,
            cpus_per_task=self.cpus_per_task,
            array_concurrency=self.array_concurrency,
            array_count=len(mappings),
            thread_environment=self.thread_environment,
        )
        write_text_atomic(job_script_path, script)

        submitted_at = utc_now()
        meta_path = submission_dir / "submission.json"
        draft_meta: dict[str, Any] = {
            "backend": "slurm",
            "submitted_at": _iso_utc(submitted_at),
            "task_indices": [m.task_index for m in mappings],
            "attempts": {
                str(m.task_index): m.attempt.attempt for m in mappings
            },
            "submission_dir": str(submission_dir),
            "job_script": str(job_script_path),
            "task_map": str(task_map_path),
            "array_count": len(mappings),
            "array_concurrency": self.array_concurrency,
            "maximum_array_size": self.maximum_array_size,
            "partition": self.partition,
            "account": self.account,
            "memory": self.memory,
            "time_limit": self.time_limit,
            "cpus_per_task": self.cpus_per_task,
            "python_executable": self.python_executable,
            "worker_module": worker_module,
            "job_id": None,
        }
        write_json_atomic(meta_path, draft_meta)

        try:
            job_id = self.commands.sbatch(job_script_path)
        except SlurmCommandError as exc:
            draft_meta["job_id"] = None
            draft_meta["sbatch_error"] = str(exc)
            write_json_atomic(meta_path, draft_meta)
            raise

        draft_meta["job_id"] = job_id
        write_json_atomic(meta_path, draft_meta)

        record = SubmissionRecord(
            backend="slurm",
            submitted_at=submitted_at,
            task_indices=tuple(m.task_index for m in mappings),
            metadata={
                "job_id": job_id,
                "submission_dir": str(submission_dir),
                "job_script": str(job_script_path),
                "task_map": str(task_map_path),
                "array_count": len(mappings),
                "array_concurrency": self.array_concurrency,
                "attempts": {str(m.task_index): m.attempt.attempt for m in mappings},
                "partition": self.partition,
                "account": self.account,
                "memory": self.memory,
                "time_limit": self.time_limit,
            },
        )
        return record

    def reconcile(self, experiment_dir: Path) -> list[dict[str, Any]]:
        """
        Reconcile all Slurm submissions under ``experiment_dir``.

        Parameters
        ----------
        experiment_dir : Path
            Prepared experiment root.

        Returns
        -------
        list of dict
            Actions taken by the reconciler.
        """
        from core.execution_core.backends.slurm_status import reconcile_slurm_experiment

        return reconcile_slurm_experiment(
            Path(experiment_dir),
            commands=self.commands,
        )

    def cancel_submission(
        self,
        experiment_dir: Path,
        submission_dir: Path,
        *,
        reconcile_after: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Cancel one submission job id with ``scancel``.

        Parameters
        ----------
        experiment_dir : Path
            Experiment root.
        submission_dir : Path
            Submission directory.
        reconcile_after : bool, optional
            Reconcile after a successful scancel.

        Returns
        -------
        list of dict
            Reconciliation actions.
        """
        from core.execution_core.backends.slurm_status import cancel_slurm_submission

        return cancel_slurm_submission(
            experiment_dir=Path(experiment_dir),
            submission_dir=Path(submission_dir),
            commands=self.commands,
            reconcile_after=reconcile_after,
        )

    def cancel_task(
        self,
        experiment_dir: Path,
        task_index: int,
        *,
        reconcile_after: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Cancel one global task via its array mapping.

        Parameters
        ----------
        experiment_dir : Path
            Experiment root.
        task_index : int
            Global task index.
        reconcile_after : bool, optional
            Reconcile after a successful scancel.

        Returns
        -------
        list of dict
            Reconciliation actions.
        """
        from core.execution_core.backends.slurm_status import cancel_slurm_task

        return cancel_slurm_task(
            experiment_dir=Path(experiment_dir),
            task_index=int(task_index),
            commands=self.commands,
            reconcile_after=reconcile_after,
        )
