"""
Brute-force dataset aggregation and ``_SUCCESS`` publication (Phase B8).

Validates the planned Parquet parts first. On failure, raises and never
writes ``dataset/_SUCCESS``. On success, writes ``aggregate/summary.json``
then atomically publishes ``_SUCCESS``.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

from core.execution_core.artifacts import read_task_status
from core.execution_core.atomic_io import read_json, write_json_atomic, write_text_atomic
from core.execution_core.manifest import load_manifest
from core.execution_core.models import (
    ExperimentPaths,
    TaskState,
    TERMINAL_TASK_STATES,
    format_task_id,
)
from core.ioh_core.workflows.brute_force.deps import import_pyarrow
from core.ioh_core.workflows.brute_force.models import BruteForceShardSpec
from core.ioh_core.workflows.brute_force.parquet_writer import shard_part_path
from core.ioh_core.workflows.brute_force.planner import BRUTE_FORCE_TASK_KIND
from core.ioh_core.workflows.brute_force.validation import (
    BruteForceValidationError,
    validate_dataset_or_raise,
)


class BruteForceAggregationError(RuntimeError):
    """
    Raised when a brute-force summary cannot be written after validation.
    """


def _experiment_paths(experiment_dir: Path) -> ExperimentPaths:
    root = Path(experiment_dir).expanduser().resolve()
    return ExperimentPaths(
        root=root,
        inputs=root / "inputs",
        submissions=root / "submissions",
        tasks=root / "tasks",
        aggregate=root / "aggregate",
    )


def _attempt_directory(task_dir: Path, attempt: int) -> Path:
    return Path(task_dir) / "attempts" / f"{int(attempt):03d}"


def _load_pq():
    import_pyarrow()
    return importlib.import_module("pyarrow.parquet")


def _violation_columns(schema_field_names: list[str]) -> list[str]:
    return [name for name in schema_field_names if name.startswith("violation__")]


def _param_columns(schema_field_names: list[str]) -> list[str]:
    return [name for name in schema_field_names if name.startswith("param__")]


def _scan_part_stats(
    part_path: Path,
    *,
    violation_cols: list[str],
) -> dict[str, Any]:
    """
    Stream scientific summary columns from one part (no full-table materialization
    beyond the projected columns).
    """
    pq = _load_pq()
    columns = [
        "global_index",
        "y_p",
        "feasible",
        "analysis_ok",
        "evaluation_time_s",
        *violation_cols,
    ]
    table = pq.read_table(str(part_path), columns=columns)
    indices = table.column("global_index").to_pylist()
    y_p_values = table.column("y_p").to_pylist()
    feasible = table.column("feasible").to_pylist()
    analysis_ok = table.column("analysis_ok").to_pylist()
    timings = table.column("evaluation_time_s").to_pylist()
    violation_maps = {
        name: table.column(name).to_pylist() for name in violation_cols
    }
    return {
        "rows": len(indices),
        "global_index": indices,
        "y_p": y_p_values,
        "feasible": feasible,
        "analysis_ok": analysis_ok,
        "evaluation_time_s": timings,
        "violations": violation_maps,
        "bytes": Path(part_path).stat().st_size,
    }


def _read_best_parameters(
    part_path: Path,
    *,
    global_index: int,
    param_cols: list[str],
) -> dict[str, Any]:
    """
    Read ``param__*`` values for one winning global index from one part.
    """
    pq = _load_pq()
    columns = ["global_index", *param_cols]
    table = pq.read_table(str(part_path), columns=columns)
    indices = table.column("global_index").to_pylist()
    try:
        row_pos = indices.index(int(global_index))
    except ValueError as exc:
        raise BruteForceAggregationError(
            f"best global_index {global_index} not found in {part_path}"
        ) from exc
    params: dict[str, Any] = {}
    for name in param_cols:
        key = name.removeprefix("param__")
        params[key] = table.column(name)[row_pos].as_py()
    return params


def write_success_marker(dataset_dir: Path) -> Path:
    """
    Atomically write ``dataset/_SUCCESS`` as the final publication action.
    """
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / "_SUCCESS"
    write_text_atomic(path, "ok\n")
    return path


def aggregate_brute_force_experiment(experiment_dir: Path) -> dict[str, Any]:
    """
    Validate the planned dataset, write aggregate summary, then ``_SUCCESS``.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.

    Returns
    -------
    dict
        Published aggregate summary.

    Raises
    ------
    BruteForceValidationError
        If dataset validation fails (no ``_SUCCESS`` is written).
    BruteForceAggregationError
        If aggregation cannot complete after a valid dataset.
    """
    paths = _experiment_paths(experiment_dir)
    manifest = load_manifest(paths.manifest)
    if manifest.workflow_type != "brute_force":
        raise BruteForceAggregationError(
            f"expected workflow_type='brute_force', got {manifest.workflow_type!r}"
        )

    dataset_dir = paths.root / "dataset"
    success_path = dataset_dir / "_SUCCESS"
    if success_path.exists():
        try:
            success_path.unlink()
        except OSError:
            pass

    # Fail closed: validate first; never publish _SUCCESS on invalid data.
    validation = validate_dataset_or_raise(experiment_dir)

    schema_path = dataset_dir / "schema.json"
    schema_doc = read_json(schema_path)
    field_names = [str(field["name"]) for field in schema_doc.get("fields", [])]
    violation_cols = _violation_columns(field_names)
    param_cols = _param_columns(field_names)

    task_rows: list[dict[str, Any]] = []
    total_rows = 0
    analysis_ok_count = 0
    analysis_error_count = 0
    feasible_count = 0
    finite_objective_count = 0
    total_bytes = 0
    best_y_p: float | None = None
    best_global_index: int | None = None
    best_part_path: Path | None = None
    violation_counts: dict[str, int] = {name: 0 for name in violation_cols}
    per_shard_timing: list[dict[str, Any]] = []

    for task in manifest.tasks:
        if task.kind != BRUTE_FORCE_TASK_KIND:
            raise BruteForceAggregationError(
                f"task {task.task_index} has unexpected kind {task.kind!r}"
            )
        spec = BruteForceShardSpec.from_payload(task.payload)
        task_dir = paths.tasks / format_task_id(task.task_index)
        status = read_task_status(task_dir)
        state = TaskState(status["state"])
        attempt = int(status.get("attempt") or 0)
        part = shard_part_path(dataset_dir, spec.shard_index)
        row: dict[str, Any] = {
            "task_index": task.task_index,
            "task_id": task.task_id,
            "state": state.value,
            "attempt": attempt,
            "shard_index": spec.shard_index,
            "start_index": spec.start_index,
            "stop_index": spec.stop_index,
            "expected_rows": spec.stop_index - spec.start_index,
            "part_path": str(part),
            "part_present": True,
        }
        if attempt >= 1:
            summary_path = _attempt_directory(task_dir, attempt) / "summary.json"
            if summary_path.is_file():
                row["attempt_summary"] = read_json(summary_path)

        stats = _scan_part_stats(part, violation_cols=violation_cols)
        total_rows += int(stats["rows"])
        total_bytes += int(stats["bytes"])
        shard_time = 0.0
        for index, y_p, feasible, analysis_ok, t_eval in zip(
            stats["global_index"],
            stats["y_p"],
            stats["feasible"],
            stats["analysis_ok"],
            stats["evaluation_time_s"],
        ):
            if analysis_ok:
                analysis_ok_count += 1
            else:
                analysis_error_count += 1
            if feasible:
                feasible_count += 1
            try:
                value = float(y_p)
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value):
                finite_objective_count += 1
                if best_y_p is None or value < best_y_p:
                    best_y_p = value
                    best_global_index = int(index)
                    best_part_path = part
            try:
                shard_time += float(t_eval)
            except (TypeError, ValueError):
                pass

        for name, values in stats["violations"].items():
            for raw in values:
                try:
                    violation = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(violation) and violation > 0.0:
                    violation_counts[name] += 1

        per_shard_timing.append(
            {
                "shard_index": spec.shard_index,
                "rows": stats["rows"],
                "evaluation_time_sum_s": shard_time,
                "bytes": stats["bytes"],
            }
        )
        task_rows.append(row)

    best_params: dict[str, Any] | None = None
    if best_global_index is not None and best_part_path is not None:
        best_params = _read_best_parameters(
            best_part_path,
            global_index=best_global_index,
            param_cols=param_cols,
        )

    terminal = sum(
        1 for row in task_rows if TaskState(row["state"]) in TERMINAL_TASK_STATES
    )
    completed = sum(
        1 for row in task_rows if row["state"] == TaskState.COMPLETED.value
    )
    failed = sum(1 for row in task_rows if row["state"] == TaskState.FAILED.value)

    # Strip param__/violation__ prefixes in published counts keys.
    published_violation_counts = {
        name.removeprefix("violation__"): count
        for name, count in violation_counts.items()
    }

    summary: dict[str, Any] = {
        "workflow_type": "brute_force",
        "preview_only": False,
        "validation_ok": True,
        "success_marker_written": False,
        "dataset_dir": str(dataset_dir),
        "schema_path": str(schema_path),
        "schema_hash": validation.schema_hash,
        "planned_rows": validation.planned_rows,
        "total_evaluations": total_rows,
        "analysis_ok_count": analysis_ok_count,
        "analysis_error_count": analysis_error_count,
        "finite_objective_count": finite_objective_count,
        "feasible_count": feasible_count,
        "best_finite_y_p": best_y_p,
        "best_global_index": best_global_index,
        "best_params": best_params,
        "per_constraint_violation_counts": published_violation_counts,
        "per_shard_timing": per_shard_timing,
        "total_bytes": total_bytes,
        "task_count": len(task_rows),
        "terminal_task_count": terminal,
        "completed_task_count": completed,
        "failed_task_count": failed,
        "ignored_part_files": list(validation.ignored_files),
        "tasks": task_rows,
    }

    paths.aggregate.mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths.aggregate / "summary.json", summary)
    write_success_marker(dataset_dir)
    summary["success_marker_written"] = True
    write_json_atomic(paths.aggregate / "summary.json", summary)
    return summary


# Re-export for callers that catch validation failures from collect.
__all__ = [
    "BruteForceAggregationError",
    "BruteForceValidationError",
    "aggregate_brute_force_experiment",
    "write_success_marker",
]
