"""
Load finished ``optimization_runs`` experiment directories for benchmarking plots.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.visualization_core.optimization.models import ExperimentBundle, RunTrajectory
from core.visualization_core.optimization.trajectories import (
    best_so_far,
    feasible_mask_from_constraints,
)


class ExperimentLoadError(ValueError):
    """Raised when an experiment directory cannot be loaded fail-closed."""


ProgressCallback = Callable[[int, int, str], None]


def _as_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ExperimentLoadError(f"expected numeric float, got bool {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentLoadError(f"expected float, got {value!r}") from exc
    if not np.isfinite(out):
        raise ExperimentLoadError(f"expected finite float, got {value!r}")
    return out


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ExperimentLoadError(f"expected int, got bool {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentLoadError(f"expected int, got {value!r}") from exc


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ExperimentLoadError(f"expected bool, got {value!r}")


def parse_ioh_dat(path: Path) -> dict[str, np.ndarray]:
    """
    Parse one IOHProfiler ``.dat`` file into named numeric columns.

    Parameters
    ----------
    path : Path
        Path to ``IOHprofiler_*_DIM*.dat``.

    Returns
    -------
    dict[str, ndarray]
        Column name → 1-D float array. Requires ``evaluations`` and ``y_p``.
    """
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ExperimentLoadError(f"empty IOH .dat file: {path}")
    header = lines[0].split()
    if "evaluations" not in header or "y_p" not in header:
        raise ExperimentLoadError(
            f"IOH .dat at {path} must contain 'evaluations' and 'y_p' columns; "
            f"got {header!r}"
        )
    rows: list[list[float]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != len(header):
            raise ExperimentLoadError(
                f"IOH .dat row width mismatch in {path}: "
                f"expected {len(header)} fields, got {len(parts)}"
            )
        try:
            rows.append([float(p) for p in parts])
        except ValueError as exc:
            raise ExperimentLoadError(
                f"non-numeric IOH .dat value in {path}: {line!r}"
            ) from exc
    if not rows:
        raise ExperimentLoadError(f"IOH .dat has header but no data rows: {path}")
    matrix = np.asarray(rows, dtype=float)
    return {name: matrix[:, idx].copy() for idx, name in enumerate(header)}


def find_ioh_dat(attempt_dir: Path) -> Path:
    """
    Locate the primary IOH ``.dat`` under an attempt directory.

    Parameters
    ----------
    attempt_dir : Path
        Attempt root containing ``ioh/``.

    Returns
    -------
    Path
        Chosen ``.dat`` path (lexicographically first if several match).
    """
    ioh_root = attempt_dir / "ioh"
    if not ioh_root.is_dir():
        raise ExperimentLoadError(f"missing ioh/ directory under {attempt_dir}")
    dats = sorted(ioh_root.rglob("IOHprofiler_*_DIM*.dat"))
    if not dats:
        dats = sorted(p for p in ioh_root.rglob("*.dat") if p.is_file())
    if not dats:
        raise ExperimentLoadError(f"no IOH .dat files under {ioh_root}")
    return dats[0]


def _attempt_dir_from_row(experiment_dir: Path, row: Mapping[str, str]) -> Path:
    artifact = (row.get("artifact_path") or "").strip()
    if artifact:
        artifact_path = Path(artifact)
        if not artifact_path.is_absolute():
            artifact_path = experiment_dir / artifact_path
        if artifact_path.name == "summary.json" and artifact_path.is_file():
            return artifact_path.parent
        if artifact_path.is_dir():
            return artifact_path
    task_index = _optional_int(row.get("task_index"))
    attempt = _optional_int(row.get("attempt"))
    if task_index is None or attempt is None:
        raise ExperimentLoadError(
            f"cannot resolve attempt directory for row {dict(row)!r}"
        )
    attempt_dir = (
        experiment_dir
        / "tasks"
        / f"task_{int(task_index):06d}"
        / "attempts"
        / f"{int(attempt):03d}"
    )
    if not attempt_dir.is_dir():
        raise ExperimentLoadError(f"missing attempt directory: {attempt_dir}")
    return attempt_dir


def _read_tasks_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ExperimentLoadError(f"empty tasks.csv: {path}")
        required = {"task_index", "algorithm_name", "state"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ExperimentLoadError(
                f"tasks.csv at {path} missing columns {sorted(missing)}"
            )
        return [dict(row) for row in reader]


def _max_evaluations_from_summary(summary: Mapping[str, Any], n_logged: int) -> int:
    params = summary.get("algorithm_parameters")
    if isinstance(params, Mapping) and "max_evaluations" in params:
        value = _optional_int(params.get("max_evaluations"))
        if value is None or value < 1:
            raise ExperimentLoadError(
                f"algorithm_parameters.max_evaluations must be >= 1, got {value!r}"
            )
        return int(value)
    # Fall back to logged length (fixed-target early stop still needs sealed B).
    if n_logged < 1:
        raise ExperimentLoadError("cannot infer max_evaluations from empty trajectory")
    return int(n_logged)


def _constraint_columns(columns: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(values, dtype=float)
        for name, values in columns.items()
        if name.startswith("c__")
    }


def _overlay_label_map(
    algorithm_by_dir: Sequence[tuple[Path, str]],
) -> dict[tuple[Path, str], str]:
    """
    Build overlay labels, disambiguating duplicate algorithms across dirs.
    """
    stems_by_algo: dict[str, set[str]] = defaultdict(set)
    for experiment_dir, algorithm_name in algorithm_by_dir:
        stems_by_algo[algorithm_name].add(experiment_dir.name)
    labels: dict[tuple[Path, str], str] = {}
    for experiment_dir, algorithm_name in algorithm_by_dir:
        if len(stems_by_algo[algorithm_name]) > 1:
            labels[(experiment_dir, algorithm_name)] = (
                f"{algorithm_name} [{experiment_dir.name}]"
            )
        else:
            labels[(experiment_dir, algorithm_name)] = algorithm_name
    return labels


def load_experiment(
    experiment_dir: Path | str,
    *,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """
    Load completed-run metadata and trajectories from one experiment directory.

    Parameters
    ----------
    experiment_dir : Path or str
        Prepared ``optimization_runs`` experiment root.
    progress : callable or None
        Optional ``(done, total, message)`` reporter. ``done`` is the number
        of completed runs whose trajectories have been read.

    Returns
    -------
    list of dict
        Intermediate records with keys ``experiment_dir``, ``algorithm_name``,
        ``task_index``, ``repetition``, ``columns``, ``summary``, ``dat_path``,
        plus sealed termination fields. An empty list means ``tasks.csv``
        has rows and none of them are completed. Plot commands then name
        those unfinished tasks. A missing or empty task file still raises.
    """
    root = _as_path(experiment_dir)
    tasks_csv = root / "aggregate" / "tasks.csv"
    if not tasks_csv.is_file():
        raise ExperimentLoadError(f"missing aggregate/tasks.csv under {root}")
    rows = _read_tasks_csv(tasks_csv)
    completed = [
        row
        for row in rows
        if (row.get("state") or "").strip().lower() == "completed"
    ]
    total = len(completed)
    if progress is not None and total:
        progress(0, total, f"loading {root.name}")
    loaded: list[dict[str, Any]] = []
    for done, row in enumerate(completed, start=1):
        attempt_dir = _attempt_dir_from_row(root, row)
        summary_path = attempt_dir / "summary.json"
        if not summary_path.is_file():
            raise ExperimentLoadError(f"missing summary.json at {summary_path}")
        summary = _read_json(summary_path)
        if not isinstance(summary, Mapping):
            raise ExperimentLoadError(f"summary.json must be an object: {summary_path}")
        dat_path = find_ioh_dat(attempt_dir)
        columns = parse_ioh_dat(dat_path)
        algorithm_name = str(
            row.get("algorithm_name") or summary.get("algorithm_name") or ""
        ).strip()
        if not algorithm_name:
            raise ExperimentLoadError(
                f"missing algorithm_name for task_index={row.get('task_index')}"
            )
        problem_id = str(
            row.get("problem_id") or summary.get("problem_id") or ""
        ).strip()
        if not problem_id:
            raise ExperimentLoadError(
                f"missing problem_id for task_index={row.get('task_index')}"
            )
        task_index = _optional_int(row.get("task_index"))
        if task_index is None:
            raise ExperimentLoadError("tasks.csv row missing task_index")
        repetition = _optional_int(row.get("repetition"))
        if repetition is None:
            repetition = _optional_int(summary.get("repetition"))
        if repetition is None:
            raise ExperimentLoadError(
                f"missing repetition for task_index={task_index}"
            )
        n_logged = int(columns["evaluations"].shape[0])
        max_evaluations = _max_evaluations_from_summary(summary, n_logged)
        loaded.append(
            {
                "experiment_dir": root,
                "algorithm_name": algorithm_name,
                "problem_id": problem_id,
                "task_index": int(task_index),
                "repetition": int(repetition),
                "columns": columns,
                "summary": dict(summary),
                "dat_path": dat_path,
                "max_evaluations": max_evaluations,
                "global_optimum": _optional_float(
                    summary.get("global_optimum", row.get("global_optimum"))
                ),
                "target_reference": summary.get("target_reference"),
                "target_hit": _optional_bool(
                    summary.get("target_hit", row.get("target_hit"))
                ),
                "evaluations_to_target": _optional_int(
                    summary.get(
                        "evaluations_to_target",
                        row.get("evaluations_to_target"),
                    )
                ),
                "wall_time_s": _optional_float(
                    summary.get("wall_time_s", row.get("wall_time_s"))
                ),
            }
        )
        if progress is not None:
            progress(
                done,
                total,
                f"loading {root.name} {problem_id} {algorithm_name}",
            )
    if not loaded:
        if rows:
            return []
        raise ExperimentLoadError(
            f"no completed tasks with IOH trajectories under {root}"
        )
    loaded.sort(key=lambda item: (item["task_index"], item["repetition"]))
    return loaded


def load_experiments(
    paths: Sequence[Path | str],
    *,
    progress: ProgressCallback | None = None,
) -> ExperimentBundle:
    """
    Load one or more experiment directories into a multi-algorithm overlay bundle.

    Parameters
    ----------
    paths : sequence of Path or str
        Experiment roots (order preserved for legend stability).
    progress : callable or None
        Optional ``(done, total, message)`` reporter, forwarded to each
        directory load. Counts restart for each directory.

    Returns
    -------
    ExperimentBundle
        Combined runs with overlay labels.
    """
    if not paths:
        raise ExperimentLoadError("load_experiments requires at least one path")
    raw_records: list[dict[str, Any]] = []
    experiment_dirs: list[Path] = []
    for path in paths:
        root = _as_path(path)
        if root in experiment_dirs:
            raise ExperimentLoadError(f"duplicate experiment directory: {root}")
        experiment_dirs.append(root)
        raw_records.extend(load_experiment(root, progress=progress))

    algo_keys = [(rec["experiment_dir"], rec["algorithm_name"]) for rec in raw_records]
    labels = _overlay_label_map(algo_keys)

    runs: list[RunTrajectory] = []
    for rec in raw_records:
        columns: dict[str, np.ndarray] = rec["columns"]
        evaluations = np.asarray(columns["evaluations"], dtype=float)
        if not np.all(np.isfinite(evaluations)):
            raise ExperimentLoadError(
                f"non-finite evaluations in {rec['dat_path']}"
            )
        eval_ints = np.asarray(np.rint(evaluations), dtype=int)
        if not np.allclose(evaluations, eval_ints):
            raise ExperimentLoadError(
                f"evaluations must be integral in {rec['dat_path']}"
            )
        y_p = np.asarray(columns["y_p"], dtype=float)
        constraints = _constraint_columns(columns)
        mask = (
            feasible_mask_from_constraints(constraints) if constraints else None
        )
        target_reference = rec["target_reference"]
        if target_reference is not None:
            target_reference = str(target_reference)
            # Older summaries sealed this token. Hitting times use every finite y_p.
            if target_reference == "feasible":
                target_reference = "all"
            if target_reference != "all":
                raise ExperimentLoadError(
                    f"unknown target_reference {target_reference!r} in "
                    f"{rec['dat_path'].parent}"
                )
        experiment_dir = rec["experiment_dir"]
        algorithm_name = rec["algorithm_name"]
        runs.append(
            RunTrajectory(
                experiment_dir=experiment_dir,
                overlay_label=labels[(experiment_dir, algorithm_name)],
                algorithm_name=algorithm_name,
                problem_id=str(rec["problem_id"]),
                task_index=int(rec["task_index"]),
                repetition=int(rec["repetition"]),
                evaluations=eval_ints,
                y_p=y_p,
                y_p_best_so_far_all=best_so_far(y_p),
                constraint_utilizations=constraints,
                feasible_mask=mask,
                max_evaluations=int(rec["max_evaluations"]),
                global_optimum=rec["global_optimum"],
                target_reference=target_reference,
                target_hit=rec["target_hit"],
                evaluations_to_target=rec["evaluations_to_target"],
                wall_time_s=rec["wall_time_s"],
                dat_path=rec["dat_path"],
            )
        )
    return ExperimentBundle(runs=tuple(runs), experiment_dirs=tuple(experiment_dirs))
