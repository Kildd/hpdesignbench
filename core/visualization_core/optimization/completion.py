"""
Refuse optimisation plots while a remaining algorithm is unfinished.

The loader keeps completed trajectories only. These checks read the same
``tasks.csv`` and treat every other task state as unfinished.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Sequence

from core.visualization_core.optimization.io import _read_tasks_csv
from core.visualization_core.optimization.models import ExperimentBundle, RunTrajectory


def _scheduled_rows(bundle: ExperimentBundle) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for experiment_dir in bundle.experiment_dirs:
        path = Path(experiment_dir) / "aggregate" / "tasks.csv"
        rows.extend(_read_tasks_csv(path))
    return rows


def _present_algorithms(
    bundle: ExperimentBundle,
    rows: Sequence[dict[str, str]],
) -> list[str]:
    present: list[str] = []
    for row in rows:
        name = (row.get("algorithm_name") or "").strip()
        if name and name not in present:
            present.append(name)
    for run in bundle.runs:
        if run.algorithm_name not in present:
            present.append(run.algorithm_name)
    return present


def _exclude_names(excluded: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(name.strip() for name in excluded if name and name.strip())
    )


def scheduled_algorithm_names(bundle: ExperimentBundle) -> list[str]:
    """
    Return sealed algorithm names in task-file order, then completed runs.

    Parameters
    ----------
    bundle : ExperimentBundle
        Loaded experiments whose ``tasks.csv`` files are readable.

    Returns
    -------
    list of str
        Names that exclusion may name, including algorithms with no
        completed run.
    """
    return _present_algorithms(bundle, _scheduled_rows(bundle))


def prepare_plot_bundle(
    bundle: ExperimentBundle,
    exclude_algorithms: Sequence[str] = (),
    *,
    known_algorithms: Sequence[str] | None = None,
    problem_id: str | None = None,
) -> ExperimentBundle:
    """
    Drop excluded algorithms and refuse unfinished or unusable repetitions.

    Parameters
    ----------
    bundle : ExperimentBundle
        Loaded experiments. ``experiment_dirs`` is scanned even when ``runs``
        has already been limited to one problem.
    exclude_algorithms : sequence of str
        Sealed algorithm names to omit. Blank entries are ignored.
    known_algorithms : sequence of str or None
        Names that count as present. ``None`` uses this bundle's task rows
        and completed runs. A compare command passes the union of its panels
        so a name that exists on only one panel is still known.
    problem_id : str or None
        When set, keep only this problem after unfinished tasks and an empty
        exclusion have been refused, and before the sealed-optimum check.
        A problem that has no completed run still reports unfinished tasks
        instead of ``unknown problem_id``.

    Returns
    -------
    ExperimentBundle
        ``bundle`` without the excluded runs. Directories are unchanged.

    Raises
    ------
    ValueError
        Unknown exclusion name, a remaining unfinished task, a completed run
        with no positive sealed optimum, or an exclusion that removes every
        completed run.
    """
    rows = _scheduled_rows(bundle)
    present = (
        list(dict.fromkeys(known_algorithms))
        if known_algorithms is not None
        else _present_algorithms(bundle, rows)
    )
    names = _exclude_names(exclude_algorithms)
    missing = [name for name in names if name not in present]
    if missing:
        raise ValueError(
            f"unknown algorithm(s) {missing}; available={present}"
        )
    blocked = set(names)
    unfinished: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        algorithm_name = (row.get("algorithm_name") or "").strip()
        if algorithm_name in blocked:
            continue
        state = (row.get("state") or "").strip().lower()
        if state == "completed":
            continue
        row_problem = (row.get("problem_id") or "").strip() or "(missing problem)"
        unfinished[
            (
                algorithm_name or "(missing algorithm)",
                row_problem,
                state or "(blank)",
            )
        ] += 1
    if unfinished:
        parts = [
            f"{algorithm} on {problem} has {count} {state} "
            f"{'task' if count == 1 else 'tasks'}"
            for (algorithm, problem, state), count in sorted(unfinished.items())
        ]
        raise ValueError(
            "refusing to plot unfinished repetitions: " + "; ".join(parts)
        )
    if problem_id is None:
        matched = bundle.runs
    else:
        matched = tuple(
            run for run in bundle.runs if run.problem_id == problem_id
        )
    kept = tuple(run for run in matched if run.algorithm_name not in blocked)
    if not kept:
        if problem_id is not None and not matched and bundle.runs:
            available = bundle.problem_ids()
            raise ValueError(
                f"unknown problem_id {problem_id!r}; available={list(available)!r}"
            )
        raise ValueError("exclusion removed every completed run")
    _require_positive_optima(kept)
    return ExperimentBundle(runs=kept, experiment_dirs=bundle.experiment_dirs)


def _require_positive_optima(runs: Sequence[RunTrajectory]) -> None:
    bad: list[str] = []
    for run in runs:
        optimum = run.global_optimum
        usable = (
            optimum is not None
            and math.isfinite(float(optimum))
            and float(optimum) > 0.0
        )
        if usable:
            continue
        bad.append(
            f"{run.algorithm_name} on {run.problem_id} "
            f"task_index={run.task_index} has no positive sealed optimum"
        )
    if bad:
        raise ValueError("; ".join(bad))
