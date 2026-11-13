"""
Aggregate optimisation-run task outcomes into CSV and JSON summaries.

Uses only the standard library. Ordering follows the sealed manifest task
index order so finish order cannot affect published rows.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

from core.execution_core.artifacts import read_task_status
from core.execution_core.atomic_io import read_json, write_json_atomic, write_text_atomic
from core.execution_core.manifest import load_manifest
from core.execution_core.models import (
    ExperimentPaths,
    TaskState,
    TERMINAL_TASK_STATES,
    format_task_id,
)
from core.ioh_core.workflows.optimization_runs.fixed_target_feasibility_audit import (
    feasibility_guarantee_for_rows,
)
from core.ioh_core.workflows.optimization_runs.io_logging import (
    validate_termination_summary,
)

_TERMINATION_CSV_FIELDS: tuple[str, ...] = (
    "termination_mode",
    "global_optimum",
    "allowed_deviation_percent",
    "target_threshold",
    "target_hit",
    "evaluations_to_target",
    "budget_exhausted",
    "stop_reason",
)

CSV_FIELDS: tuple[str, ...] = (
    "task_index",
    "task_id",
    "problem_id",
    "algorithm_name",
    "repetition",
    "random_seed",
    "attempt",
    "state",
    "best_objective",
    "best_x_json",
    "hits",
    "misses",
    "evaluations",
    "wall_time_s",
    "artifact_path",
    "failure_kind",
    "failure_message",
) + _TERMINATION_CSV_FIELDS


class AggregationError(ValueError):
    """Raised when a task summary fails termination aggregation checks."""


def _experiment_paths(experiment_dir: Path) -> ExperimentPaths:
    """
    Build the standard path bundle for an experiment root.

    Parameters
    ----------
    experiment_dir : Path
        Experiment directory.

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


def _attempt_directory(task_dir: Path, attempt: int) -> Path:
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
        Attempt directory.
    """
    return Path(task_dir) / "attempts" / f"{attempt:03d}"


def _valid_summary_artifact(attempt_dir: Path) -> Path | None:
    """
    Return a valid ``summary.json`` path when present.

    Validity still keys only on ``best_objective`` so additive C4 fields do
    not break reconciliation of older attempts.

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


def _json_cell(value: Any) -> str:
    """
    Encode a value as a compact JSON string for CSV cells.

    Parameters
    ----------
    value : Any
        Cell value.

    Returns
    -------
    str
        JSON text, or empty string for ``None``.
    """
    if value is None:
        return ""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _csv_cell(value: Any) -> Any:
    """
    Convert optional values to CSV-friendly cells.

    Parameters
    ----------
    value : Any
        Row value.

    Returns
    -------
    Any
        ``""`` for ``None``; otherwise ``value``.
    """
    return "" if value is None else value


def _truthy(value: Any) -> bool:
    """
    Interpret CSV/JSON boolean-ish values.

    Parameters
    ----------
    value : Any
        Cell or summary value.

    Returns
    -------
    bool
        Parsed boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _payload_max_evaluations(payload: Mapping[str, Any]) -> int | None:
    """
    Read sealed ``max_evaluations`` from a task payload when present.

    Parameters
    ----------
    payload : Mapping
        Manifest task payload.

    Returns
    -------
    int or None
        Positive budget, or ``None``.
    """
    parameters = payload.get("algorithm_parameters")
    if not isinstance(parameters, Mapping):
        return None
    value = parameters.get("max_evaluations")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return int(value)
    return None


def normalize_summary_termination(
    summary: Mapping[str, Any] | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Normalize termination fields for aggregation, including legacy summaries.

    Parameters
    ----------
    summary : Mapping or None
        Parsed ``summary.json`` when valid.
    payload : Mapping
        Manifest task payload (used for legacy budget derivation).

    Returns
    -------
    dict
        Termination fields ready for CSV/JSON rows.

    Raises
    ------
    AggregationError
        If termination fields are partial or fail fixed-target checks.
    """
    empty = {field: None for field in _TERMINATION_CSV_FIELDS}
    if summary is None:
        return empty

    present = [key for key in _TERMINATION_CSV_FIELDS if key in summary]
    if not present:
        max_evaluations = _payload_max_evaluations(payload)
        evaluations = summary.get("evaluations")
        budget_exhausted = (
            max_evaluations is not None
            and evaluations is not None
            and int(evaluations) >= int(max_evaluations)
        )
        return {
            "termination_mode": "fixed_budget",
            "global_optimum": None,
            "allowed_deviation_percent": None,
            "target_threshold": None,
            "target_hit": False,
            "evaluations_to_target": None,
            "budget_exhausted": bool(budget_exhausted),
            "stop_reason": (
                "budget_exhausted" if budget_exhausted else "algorithm_terminated"
            ),
        }

    if len(present) != len(_TERMINATION_CSV_FIELDS):
        raise AggregationError(
            f"summary has partial termination fields: {present}; "
            f"expected all of {list(_TERMINATION_CSV_FIELDS)}"
        )

    fields = {key: summary[key] for key in _TERMINATION_CSV_FIELDS}
    mode = str(fields["termination_mode"])
    if mode == "fixed_target":
        max_evaluations = _payload_max_evaluations(payload)
        if max_evaluations is None:
            raise AggregationError(
                "fixed-target summary requires sealed max_evaluations"
            )
        try:
            validate_termination_summary(
                {
                    **dict(summary),
                    **fields,
                },
                sealed_termination_mode=mode,
                sealed_global_optimum=fields["global_optimum"],
                sealed_allowed_deviation_percent=fields[
                    "allowed_deviation_percent"
                ],
                sealed_target_threshold=fields["target_threshold"],
                max_evaluations=max_evaluations,
            )
        except ValueError as exc:
            raise AggregationError(str(exc)) from exc
    return fields


def _failure_fields(status: Mapping[str, Any], attempt_dir: Path | None) -> tuple[str, str]:
    """
    Extract failure kind and message from status / failure.json.

    Parameters
    ----------
    status : Mapping
        Task status document.
    attempt_dir : Path or None
        Latest attempt directory when known.

    Returns
    -------
    tuple of str
        ``(failure_kind, failure_message)``.
    """
    meta = status.get("metadata") if isinstance(status.get("metadata"), dict) else {}
    kind = meta.get("failure_kind") or meta.get("error_type") or ""
    message = meta.get("message") or ""
    if attempt_dir is not None:
        failure_path = Path(attempt_dir) / "failure.json"
        if failure_path.is_file():
            try:
                payload = read_json(failure_path)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                kind = kind or str(payload.get("error_type") or "")
                message = message or str(payload.get("message") or "")
    return str(kind), str(message)


def build_task_row(
    *,
    task_index: int,
    task_id: str,
    payload: Mapping[str, Any],
    status: Mapping[str, Any],
    attempt_dir: Path | None,
    summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Build one aggregate CSV/JSON row for a task.

    Parameters
    ----------
    task_index : int
        Zero-based task index.
    task_id : str
        Stable task id.
    payload : Mapping
        Manifest task payload.
    status : Mapping
        On-disk status document.
    attempt_dir : Path or None
        Latest attempt directory.
    summary : Mapping or None
        Parsed ``summary.json`` when valid.

    Returns
    -------
    dict
        Row keyed by :data:`CSV_FIELDS`.
    """
    state = str(status.get("state", ""))
    attempt = int(status.get("attempt", 0))
    failure_kind, failure_message = _failure_fields(status, attempt_dir)
    artifact = ""
    if attempt_dir is not None:
        summary_path = _valid_summary_artifact(attempt_dir)
        if summary_path is not None:
            artifact = str(summary_path)
        elif (Path(attempt_dir) / "failure.json").is_file():
            artifact = str(Path(attempt_dir) / "failure.json")

    best_objective = ""
    best_x_json = ""
    hits = ""
    misses = ""
    evaluations = ""
    wall_time_s = ""
    if summary is not None:
        if "best_objective" in summary:
            best_objective = summary["best_objective"]
        if "best_x" in summary:
            best_x_json = _json_cell(summary["best_x"])
        if "hits" in summary:
            hits = summary["hits"]
        if "misses" in summary:
            misses = summary["misses"]
        if "evaluations" in summary:
            evaluations = summary["evaluations"]
        if "wall_time_s" in summary:
            wall_time_s = summary["wall_time_s"]

    termination = normalize_summary_termination(summary, payload)

    return {
        "task_index": task_index,
        "task_id": task_id,
        "problem_id": payload.get("problem_id", ""),
        "algorithm_name": payload.get("algorithm_name", ""),
        "repetition": payload.get("repetition", ""),
        "random_seed": payload.get("random_seed", ""),
        "attempt": attempt,
        "state": state,
        "best_objective": best_objective,
        "best_x_json": best_x_json,
        "hits": hits,
        "misses": misses,
        "evaluations": evaluations,
        "wall_time_s": wall_time_s,
        "artifact_path": artifact,
        "failure_kind": failure_kind,
        "failure_message": failure_message,
        **{key: _csv_cell(termination[key]) for key in _TERMINATION_CSV_FIELDS},
    }


def collect_optimization_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    """
    Collect deterministic aggregate rows for an optimisation experiment.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.

    Returns
    -------
    list of dict
        Rows in ascending ``task_index`` order.
    """
    paths = _experiment_paths(experiment_dir)
    manifest = load_manifest(paths.manifest)
    rows: list[dict[str, Any]] = []
    for task in manifest.tasks:
        task_dir = paths.tasks / format_task_id(task.task_index)
        status = read_task_status(task_dir)
        attempt = int(status.get("attempt", 0))
        attempt_dir = _attempt_directory(task_dir, attempt) if attempt >= 1 else None
        summary = None
        if attempt_dir is not None:
            summary_path = _valid_summary_artifact(attempt_dir)
            if summary_path is not None:
                loaded = read_json(summary_path)
                if isinstance(loaded, dict):
                    summary = loaded
        rows.append(
            build_task_row(
                task_index=task.task_index,
                task_id=task.task_id,
                payload=task.payload,
                status=status,
                attempt_dir=attempt_dir,
                summary=summary,
            )
        )
    return rows


def write_tasks_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Atomically publish ``tasks.csv``.

    Parameters
    ----------
    path : Path
        Destination CSV path.
    rows : list of dict
        Aggregate rows.
    """
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    write_text_atomic(path, buffer.getvalue())


def _termination_rollups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build experiment-level termination statistics.

    Parameters
    ----------
    rows : list of dict
        Aggregate task rows.

    Returns
    -------
    dict
        Rollup fields for ``aggregate/summary.json``.
    """
    completed = [row for row in rows if str(row.get("state")) == "completed"]
    fixed_target = [
        row
        for row in completed
        if str(row.get("termination_mode")) == "fixed_target"
    ]
    hit_rows = [row for row in fixed_target if _truthy(row.get("target_hit"))]
    miss_rows = [row for row in fixed_target if not _truthy(row.get("target_hit"))]
    n_fixed = len(fixed_target)
    hit_count = len(hit_rows)
    miss_count = len(miss_rows)

    hitting_times: list[int] = []
    for row in hit_rows:
        value = row.get("evaluations_to_target")
        if value in ("", None):
            continue
        hitting_times.append(int(value))

    if hitting_times:
        e2t_min = min(hitting_times)
        e2t_max = max(hitting_times)
        e2t_mean = float(statistics.fmean(hitting_times))
        e2t_median = float(statistics.median(hitting_times))
    else:
        e2t_min = None
        e2t_max = None
        e2t_mean = None
        e2t_median = None

    return {
        "fixed_target_task_count": n_fixed,
        "target_hit_count": hit_count,
        "target_hit_share": (hit_count / n_fixed) if n_fixed else None,
        "target_miss_count": miss_count,
        "target_miss_share": (miss_count / n_fixed) if n_fixed else None,
        "evaluations_to_target_min": e2t_min,
        "evaluations_to_target_median": e2t_median,
        "evaluations_to_target_mean": e2t_mean,
        "evaluations_to_target_max": e2t_max,
        "budget_exhausted_count": sum(
            1 for row in completed if _truthy(row.get("budget_exhausted"))
        ),
        "algorithm_terminated_count": sum(
            1
            for row in completed
            if str(row.get("stop_reason")) == "algorithm_terminated"
        ),
    }


def aggregate_optimization_experiment(experiment_dir: Path) -> dict[str, Any]:
    """
    Write ``aggregate/tasks.csv`` and ``aggregate/summary.json``.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.

    Returns
    -------
    dict
        Summary document also written to disk.
    """
    paths = _experiment_paths(experiment_dir)
    paths.aggregate.mkdir(parents=True, exist_ok=True)
    rows = collect_optimization_rows(paths.root)

    write_tasks_csv(paths.aggregate / "tasks.csv", rows)

    counts: dict[str, int] = {state.value: 0 for state in TaskState}
    for row in rows:
        state = str(row.get("state") or "")
        counts[state] = counts.get(state, 0) + 1

    summary = {
        "experiment_dir": str(paths.root),
        "task_count": len(rows),
        "counts": counts,
        "complete": all(
            TaskState(str(row["state"])) in TERMINAL_TASK_STATES
            for row in rows
            if row.get("state")
        ),
        "tasks_csv": str(paths.aggregate / "tasks.csv"),
        **_termination_rollups(rows),
    }
    guarantee, pairs = _feasibility_guarantee(rows, paths.root)
    summary["feasibility_guarantee"] = guarantee
    summary["feasibility_guarantee_pairs"] = pairs
    write_json_atomic(paths.aggregate / "summary.json", summary)
    return summary


def _feasibility_guarantee(
    rows: list[dict[str, Any]],
    experiment_dir: Path,
) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Attach the catalogue audit without changing official target metrics.

    Parameters
    ----------
    rows : list of dict
        Aggregate task rows.
    experiment_dir : Path
        Experiment root.

    Returns
    -------
    tuple
        Rollup state and per fixed-target pair. Fixed-budget experiments
        return ``(None, [])``.
    """
    return feasibility_guarantee_for_rows(rows, experiment_dir)
