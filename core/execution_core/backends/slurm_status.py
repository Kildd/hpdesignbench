"""
Slurm array status parsing, reconciliation, and cancellation helpers.

Uses parseable ``squeue`` / ``sacct`` output and ignores ``.batch`` /
``.extern`` step records. Worker ``summary.json`` artifacts override
ambiguous scheduler reports when valid.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.execution_core.artifacts import (
    read_task_status,
    transition_task_status,
    utc_now,
)
from core.execution_core.atomic_io import read_json, write_json_atomic
from core.execution_core.backends.slurm_commands import SlurmCommands
from core.execution_core.models import TERMINAL_TASK_STATES, TaskState, format_task_id

_ACTIVE_SLURM_STATES = frozenset(
    {
        "PENDING",
        "CONFIGURING",
        "RUNNING",
        "COMPLETING",
        "SUSPENDED",
        "REQUEUED",
        "RESIZING",
        "STAGE_OUT",
        "SIGNALING",
    }
)
_FAILED_SLURM_STATES = frozenset(
    {
        "FAILED",
        "BOOT_FAIL",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
    }
)
_TIMEOUT_SLURM_STATES = frozenset({"TIMEOUT", "DEADLINE"})
_CANCELLED_SLURM_STATES = frozenset({"CANCELLED"})
_COMPLETED_SLURM_STATES = frozenset({"COMPLETED"})


class SlurmStatusError(RuntimeError):
    """
    Raised when Slurm status parsing or reconciliation cannot proceed safely.

    Typical causes: unreadable submission metadata, malformed task maps, or
    Slurm CLI failures when mutation would otherwise be unsafe.
    """


@dataclass(frozen=True)
class SlurmTaskRecord:
    """
    One array-element scheduler observation.

    Attributes
    ----------
    job_id : str
        Base Slurm job id (without array suffix).
    array_index : int or None
        Local array index, or ``None`` for non-array job records.
    state : str
        Uppercase Slurm state token.
    raw_job_id : str
        Original JobID field from squeue/sacct.
    reason : str
        Optional reason / exit hint from sacct.
    """

    job_id: str
    array_index: int | None
    state: str
    raw_job_id: str
    reason: str = ""


@dataclass(frozen=True)
class TaskMapEntry:
    """
    One ``task-map.tsv`` row.

    Attributes
    ----------
    local_array_index : int
        Slurm array element index.
    task_index : int
        Global experiment task index.
    attempt : int
        One-based attempt reserved for that element.
    """

    local_array_index: int
    task_index: int
    attempt: int


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


def is_ignored_job_step(job_id_field: str) -> bool:
    """
    Return whether a sacct/squeue JobID is a ``.batch`` or ``.extern`` step.

    Parameters
    ----------
    job_id_field : str
        Raw JobID column value.

    Returns
    -------
    bool
        ``True`` when the record must be ignored.
    """
    text = str(job_id_field).strip().lower()
    return text.endswith(".batch") or text.endswith(".extern")


def parse_job_id_field(job_id_field: str) -> tuple[str, int | None]:
    """
    Split a Slurm JobID into base id and optional array index.

    Parameters
    ----------
    job_id_field : str
        Values such as ``\"12345\"``, ``\"12345_7\"``, or ``\"12345_7.batch\"``.

    Returns
    -------
    tuple
        ``(base_job_id, array_index_or_none)``.
    """
    text = str(job_id_field).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    if "_" in text:
        base, _, suffix = text.partition("_")
        try:
            return base, int(suffix)
        except ValueError:
            return base, None
    return text, None


def classify_slurm_state(state: str) -> str:
    """
    Map a Slurm state token to a coarse outcome class.

    Parameters
    ----------
    state : str
        Slurm state string.

    Returns
    -------
    str
        One of ``active``, ``completed``, ``failed``, ``timed_out``,
        ``cancelled``, or ``unknown``.
    """
    token = str(state).strip().upper()
    token = token.split("+", 1)[0].split(" ", 1)[0]
    if token in _ACTIVE_SLURM_STATES:
        return "active"
    if token in _COMPLETED_SLURM_STATES:
        return "completed"
    if token in _TIMEOUT_SLURM_STATES:
        return "timed_out"
    if token in _CANCELLED_SLURM_STATES:
        return "cancelled"
    if token in _FAILED_SLURM_STATES:
        return "failed"
    return "unknown"


def _prefer_record(new: SlurmTaskRecord, old: SlurmTaskRecord) -> bool:
    """
    Prefer a terminal sacct observation over an active duplicate.

    Parameters
    ----------
    new : SlurmTaskRecord
        Candidate replacement.
    old : SlurmTaskRecord
        Previously stored record.

    Returns
    -------
    bool
        ``True`` when ``new`` should replace ``old``.
    """
    new_class = classify_slurm_state(new.state)
    old_class = classify_slurm_state(old.state)
    if old_class == "active" and new_class != "active":
        return True
    if new_class == "active" and old_class != "active":
        return False
    return True


def parse_squeue_parsable(text: str) -> list[SlurmTaskRecord]:
    """
    Parse ``squeue --parsable2`` rows into task records.

    Expected columns: ``JobID|State`` (header optional).

    Parameters
    ----------
    text : str
        Command stdout.

    Returns
    -------
    list of SlurmTaskRecord
        Queue observations (``.batch`` / ``.extern`` skipped).
    """
    records: list[SlurmTaskRecord] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("jobid"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_field, state = parts[0].strip(), parts[1].strip()
        if is_ignored_job_step(job_field):
            continue
        base, array_index = parse_job_id_field(job_field)
        if not base:
            continue
        records.append(
            SlurmTaskRecord(
                job_id=base,
                array_index=array_index,
                state=state.upper(),
                raw_job_id=job_field,
            )
        )
    return records


def parse_sacct_parsable(text: str) -> list[SlurmTaskRecord]:
    """
    Parse ``sacct --parsable2`` rows into task records.

    Expected columns: ``JobID|State|ExitCode|Reason`` (header optional).
    Duplicate rows for the same array element keep a terminal-preferring
    observation (last write among equals).

    Parameters
    ----------
    text : str
        Command stdout.

    Returns
    -------
    list of SlurmTaskRecord
        Accounting observations with ``.batch`` / ``.extern`` removed.
    """
    by_key: dict[tuple[str, int | None], SlurmTaskRecord] = {}
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("jobid"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_field = parts[0].strip()
        state = parts[1].strip()
        reason = parts[3].strip() if len(parts) > 3 else ""
        if not reason and len(parts) > 2 and parts[2].strip():
            reason = f"ExitCode={parts[2].strip()}"
        if is_ignored_job_step(job_field):
            continue
        base, array_index = parse_job_id_field(job_field)
        if not base:
            continue
        candidate = SlurmTaskRecord(
            job_id=base,
            array_index=array_index,
            state=state.upper(),
            raw_job_id=job_field,
            reason=reason,
        )
        key = (base, array_index)
        existing = by_key.get(key)
        if existing is None or _prefer_record(candidate, existing):
            by_key[key] = candidate
    return list(by_key.values())


def load_task_map(path: Path) -> list[TaskMapEntry]:
    """
    Load a ``task-map.tsv`` file.

    Parameters
    ----------
    path : Path
        Task map path.

    Returns
    -------
    list of TaskMapEntry
        Parsed rows.

    Raises
    ------
    SlurmStatusError
        If the file is missing or malformed.
    """
    path = Path(path)
    if not path.is_file():
        raise SlurmStatusError(f"missing task-map.tsv: {path}")
    entries: list[TaskMapEntry] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("local_array_index"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise SlurmStatusError(f"malformed task-map line {line_no}: {raw!r}")
        try:
            entries.append(
                TaskMapEntry(
                    local_array_index=int(parts[0]),
                    task_index=int(parts[1]),
                    attempt=int(parts[2]),
                )
            )
        except ValueError as exc:
            raise SlurmStatusError(
                f"malformed task-map line {line_no}: {raw!r}"
            ) from exc
    return entries


def list_submission_dirs(experiment_dir: Path) -> list[Path]:
    """
    Return sorted Slurm submission directories under an experiment.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root.

    Returns
    -------
    list of Path
        ``submissions/submission_*`` directories in name order.
    """
    root = Path(experiment_dir) / "submissions"
    if not root.is_dir():
        return []
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name.startswith("submission_")
    ]


def read_submission_meta(submission_dir: Path) -> dict[str, Any]:
    """
    Read ``submission.json`` for one submission directory.

    Parameters
    ----------
    submission_dir : Path
        Submission directory.

    Returns
    -------
    dict
        Submission metadata.

    Raises
    ------
    SlurmStatusError
        If the file is missing or not an object.
    """
    path = Path(submission_dir) / "submission.json"
    if not path.is_file():
        raise SlurmStatusError(f"missing submission.json in {submission_dir}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise SlurmStatusError(f"submission.json is not an object: {path}")
    return payload


def valid_summary_path(attempt_dir: Path) -> Path | None:
    """
    Return a valid ``summary.json`` path when present.

    Parameters
    ----------
    attempt_dir : Path
        Attempt directory.

    Returns
    -------
    Path or None
        Summary path, or ``None``.
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


def _admin_write_status(
    task_dir: Path,
    *,
    state: TaskState,
    attempt: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Privileged status publisher used by Slurm reconciliation.

    Parameters
    ----------
    task_dir : Path
        Task directory.
    state : TaskState
        New state.
    attempt : int
        Attempt index.
    metadata : dict or None, optional
        Metadata merge.
    """
    current = read_task_status(task_dir)
    if TaskState(current["state"]) is TaskState.COMPLETED and state is not TaskState.COMPLETED:
        return
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


def fetch_job_records(
    commands: SlurmCommands,
    job_id: str,
) -> dict[int | None, SlurmTaskRecord]:
    """
    Query squeue then sacct for one job and merge by array index.

    Parameters
    ----------
    commands : SlurmCommands
        Slurm CLI runner.
    job_id : str
        Base job id.

    Returns
    -------
    dict
        Mapping array index → merged observation.

    Raises
    ------
    SlurmCommandError
        If a CLI call fails (caller must not mutate state).
    """
    squeue_text = commands.squeue(
        [
            "-j",
            str(job_id),
            "-h",
            "--parsable2",
            "-o",
            "JobID|State",
        ]
    )
    sacct_text = commands.sacct(
        [
            "-j",
            str(job_id),
            "-n",
            "-P",
            "-X",
            "-o",
            "JobID,State,ExitCode,Reason",
        ]
    )
    out: dict[int | None, SlurmTaskRecord] = {}
    for record in parse_squeue_parsable(squeue_text):
        if record.job_id != str(job_id):
            continue
        out[record.array_index] = record
    for record in parse_sacct_parsable(sacct_text):
        if record.job_id != str(job_id):
            continue
        existing = out.get(record.array_index)
        if existing is None or _prefer_record(record, existing):
            out[record.array_index] = record
    return out


def apply_scheduler_outcome(
    *,
    experiment_dir: Path,
    task_index: int,
    attempt: int,
    slurm_record: SlurmTaskRecord,
) -> dict[str, Any] | None:
    """
    Apply one scheduler observation to a task directory.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root.
    task_index : int
        Global task index.
    attempt : int
        Attempt index from the task map.
    slurm_record : SlurmTaskRecord
        Scheduler observation.

    Returns
    -------
    dict or None
        Action record when a mutation occurred, else ``None``.
    """
    task_dir = Path(experiment_dir) / "tasks" / format_task_id(task_index)
    status = read_task_status(task_dir)
    current = TaskState(status["state"])
    if current is TaskState.COMPLETED:
        return None

    attempt_dir = task_dir / "attempts" / f"{attempt:03d}"
    summary = valid_summary_path(attempt_dir)
    if summary is not None:
        _admin_write_status(
            task_dir,
            state=TaskState.COMPLETED,
            attempt=attempt,
            metadata={
                "summary": str(summary),
                "reconciled": "adopted_summary",
                "slurm_state": slurm_record.state,
            },
        )
        return {
            "task_index": task_index,
            "action": "adopted",
            "summary": str(summary),
            "slurm_state": slurm_record.state,
        }

    outcome = classify_slurm_state(slurm_record.state)
    if outcome == "active":
        if current is TaskState.DISPATCHED and slurm_record.state.upper() == "RUNNING":
            try:
                transition_task_status(
                    task_dir,
                    expected={TaskState.DISPATCHED},
                    new_state=TaskState.RUNNING,
                    attempt=attempt,
                    metadata={"slurm_state": slurm_record.state},
                )
                return {
                    "task_index": task_index,
                    "action": "running",
                    "slurm_state": slurm_record.state,
                }
            except Exception:
                return None
        return None

    if current in TERMINAL_TASK_STATES:
        return None

    if outcome == "completed":
        _admin_write_status(
            task_dir,
            state=TaskState.FAILED,
            attempt=attempt,
            metadata={
                "failure_kind": "missing_summary",
                "reconciled": "slurm_completed_missing_summary",
                "slurm_state": slurm_record.state,
                "slurm_reason": slurm_record.reason,
            },
        )
        return {
            "task_index": task_index,
            "action": "missing_summary",
            "slurm_state": slurm_record.state,
        }

    if outcome == "timed_out":
        terminal = TaskState.TIMED_OUT
        kind = "timeout"
    elif outcome == "cancelled":
        terminal = TaskState.CANCELLED
        kind = "cancelled"
    elif outcome == "failed":
        terminal = TaskState.FAILED
        kind = str(slurm_record.state).lower()
    else:
        return None

    _admin_write_status(
        task_dir,
        state=terminal,
        attempt=attempt,
        metadata={
            "failure_kind": kind,
            "reconciled": "slurm_scheduler",
            "slurm_state": slurm_record.state,
            "slurm_reason": slurm_record.reason,
        },
    )
    return {
        "task_index": task_index,
        "action": kind,
        "slurm_state": slurm_record.state,
        "state": terminal.value,
    }


def reconcile_submission(
    *,
    experiment_dir: Path,
    submission_dir: Path,
    commands: SlurmCommands,
) -> list[dict[str, Any]]:
    """
    Reconcile one submission directory against squeue/sacct.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root.
    submission_dir : Path
        ``submissions/submission_NNN`` directory.
    commands : SlurmCommands
        Slurm CLI runner.

    Returns
    -------
    list of dict
        Actions taken.

    Raises
    ------
    SlurmCommandError
        If querying Slurm fails before any mutation for this submission.
    SlurmStatusError
        If submission metadata / task map is invalid.
    """
    meta = read_submission_meta(submission_dir)
    job_id = meta.get("job_id")
    if not job_id:
        return []
    task_map_path = Path(meta.get("task_map") or (Path(submission_dir) / "task-map.tsv"))
    entries = load_task_map(task_map_path)
    records = fetch_job_records(commands, str(job_id))
    actions: list[dict[str, Any]] = []
    for entry in entries:
        record = records.get(entry.local_array_index)
        if record is None:
            continue
        action = apply_scheduler_outcome(
            experiment_dir=experiment_dir,
            task_index=entry.task_index,
            attempt=entry.attempt,
            slurm_record=record,
        )
        if action is not None:
            action["submission_dir"] = str(submission_dir)
            action["job_id"] = str(job_id)
            actions.append(action)
    return actions


def reconcile_slurm_experiment(
    experiment_dir: Path,
    *,
    commands: SlurmCommands,
) -> list[dict[str, Any]]:
    """
    Reconcile every Slurm submission under an experiment.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.
    commands : SlurmCommands
        Slurm CLI runner.

    Returns
    -------
    list of dict
        Combined actions across submissions.

    Raises
    ------
    SlurmCommandError
        If a Slurm query fails.
    """
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    actions: list[dict[str, Any]] = []
    for submission_dir in list_submission_dirs(experiment_dir):
        actions.extend(
            reconcile_submission(
                experiment_dir=experiment_dir,
                submission_dir=submission_dir,
                commands=commands,
            )
        )
    return actions


def find_task_in_submissions(
    experiment_dir: Path,
    task_index: int,
) -> tuple[Path, dict[str, Any], TaskMapEntry] | None:
    """
    Locate the submission and task-map row for a global task index.

    Prefers the latest submission directory that contains the task.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root.
    task_index : int
        Global task index.

    Returns
    -------
    tuple or None
        ``(submission_dir, meta, entry)`` when found.
    """
    found = None
    for submission_dir in list_submission_dirs(experiment_dir):
        try:
            meta = read_submission_meta(submission_dir)
        except SlurmStatusError:
            continue
        task_map_path = Path(meta.get("task_map") or (submission_dir / "task-map.tsv"))
        try:
            entries = load_task_map(task_map_path)
        except SlurmStatusError:
            continue
        for entry in entries:
            if entry.task_index == task_index:
                found = (submission_dir, meta, entry)
    return found


def cancel_slurm_submission(
    *,
    experiment_dir: Path,
    submission_dir: Path,
    commands: SlurmCommands,
    reconcile_after: bool = True,
) -> list[dict[str, Any]]:
    """
    Cancel an entire Slurm submission with ``scancel JOBID``.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root.
    submission_dir : Path
        Submission directory.
    commands : SlurmCommands
        Slurm CLI runner.
    reconcile_after : bool, optional
        When ``True``, reconcile only after ``scancel`` succeeds.

    Returns
    -------
    list of dict
        Reconciliation actions (empty when ``reconcile_after`` is false).

    Raises
    ------
    SlurmCommandError
        If ``scancel`` fails; task state is left unchanged.
    SlurmStatusError
        If submission metadata is invalid.
    """
    meta = read_submission_meta(submission_dir)
    job_id = meta.get("job_id")
    if not job_id:
        raise SlurmStatusError(f"submission {submission_dir} has no job_id")
    commands.scancel([str(job_id)])
    if not reconcile_after:
        return []
    return reconcile_submission(
        experiment_dir=experiment_dir,
        submission_dir=submission_dir,
        commands=commands,
    )


def cancel_slurm_task(
    *,
    experiment_dir: Path,
    task_index: int,
    commands: SlurmCommands,
    reconcile_after: bool = True,
) -> list[dict[str, Any]]:
    """
    Cancel one task by mapping it to ``scancel JOBID_ARRAYINDEX``.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root.
    task_index : int
        Global task index.
    commands : SlurmCommands
        Slurm CLI runner.
    reconcile_after : bool, optional
        When ``True``, reconcile the owning submission after scancel.

    Returns
    -------
    list of dict
        Reconciliation actions.

    Raises
    ------
    SlurmStatusError
        If the task is not found in any submission.
    SlurmCommandError
        If ``scancel`` fails; task state is left unchanged.
    """
    located = find_task_in_submissions(experiment_dir, task_index)
    if located is None:
        raise SlurmStatusError(
            f"task {task_index} not found in any Slurm submission under "
            f"{experiment_dir}"
        )
    submission_dir, meta, entry = located
    job_id = meta.get("job_id")
    if not job_id:
        raise SlurmStatusError(f"submission {submission_dir} has no job_id")
    commands.scancel([f"{job_id}_{entry.local_array_index}"])
    if not reconcile_after:
        return []
    return reconcile_submission(
        experiment_dir=experiment_dir,
        submission_dir=submission_dir,
        commands=commands,
    )
