"""
Frozen data models for algorithm-benchmarking optimisation plots.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

DisplayMode = Literal["median", "mean", "best", "all"]
SuccessFilter = Literal["successes", "all_runs"]
TargetReference = Literal["all"]


@dataclass(frozen=True)
class PlotConfig:
    """
    User-facing plot configuration (aggregation semantics locked in Phase V1).

    Parameters
    ----------
    display : DisplayMode
        Anytime / convergence aggregation mode. Default ``median``.
    success_filter : SuccessFilter
        Runtime-plot filter for box/histogram/ECDF. Default ``successes``.
        ERT always uses the Hansen all-runs formula regardless of this flag.
    target_percents : tuple of float
        Relative target deviations in percent. Default densifies near the
        optimum (1% steps through 5%), then 5% steps through 20%:
        ``(1, 2, 3, 4, 5, 10, 15, 20)``.
    global_optimum : float or None
        Override / supply known optimum when absent from sealed summaries.
    target_reference : TargetReference or None
        Hitting-time reference. ``\"all\"`` counts every finite ``y_p``.
        ``None`` uses the same rule.
    extend_short_runs : bool
        When true (default), carry forward last best-so-far values past each
        run's last logged evaluation up to ``max_i B_i``, so fixed-target
        early stops still appear on the full evaluation axis.
    show_iqr : bool
        When true and ``display="median"``, draw a 25–75% IQR ribbon.
        Default true.
    y_max : float or None
        Optional upper limit for the convergence y-axis. When a positive
        global optimum is known, the axis is relative (``100%`` = optimum)
        and this value is a percent (default top ``300`` if unset). Without
        a usable optimum the axis is absolute ``y_p`` and this value is the
        absolute upper limit.
    """

    display: DisplayMode = "median"
    success_filter: SuccessFilter = "successes"
    target_percents: tuple[float, ...] = (
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        10.0,
        15.0,
        20.0,
    )
    global_optimum: float | None = None
    target_reference: TargetReference | None = None
    extend_short_runs: bool = True
    show_iqr: bool = True
    y_max: float | None = None


@dataclass(frozen=True)
class RunTrajectory:
    """
    One completed optimisation attempt with its IOH quality trajectory.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root that produced this run.
    overlay_label : str
        Legend / grouping key (algorithm name, disambiguated by directory
        stem when the same algorithm appears in multiple directories).
    algorithm_name : str
        Sealed algorithm name from ``tasks.csv`` / attempt summary.
    problem_id : str
        Sealed problem identifier from ``tasks.csv`` / attempt summary.
    task_index : int
        Manifest task index.
    repetition : int
        Zero-based repetition index.
    evaluations : ndarray
        1-based IOH evaluation indices from the ``.dat`` file.
    y_p : ndarray
        Penalised objective values (the only quality channel used in V1).
    y_p_best_so_far_all : ndarray
        Cumulative minimum of ``y_p`` over all evaluations. Hitting times
        use this curve.
    constraint_utilizations : Mapping[str, ndarray]
        Logged ``c__*`` utilisation columns (may be empty).
    feasible_mask : ndarray or None
        Per-evaluation feasibility when ``c__*`` columns exist; otherwise
        ``None``. Used by feasibility summaries.
    max_evaluations : int
        Allocated evaluation ceiling. Hansen ERT charges ``n_evaluations``
        for an unsuccessful run, not this ceiling.
    global_optimum : float or None
        Sealed known optimum, if any.
    target_reference : str or None
        Sealed target reference (``all``), if any.
    target_hit : bool or None
        Sealed study target-hit flag for the experiment's own percentage.
    evaluations_to_target : int or None
        Sealed hitting time for the experiment's own percentage.
    wall_time_s : float or None
        Wall-clock seconds for the completed attempt (from summary /
        ``tasks.csv``), if available.
    dat_path : Path
        Source IOH ``.dat`` path.
    """

    experiment_dir: Path
    overlay_label: str
    algorithm_name: str
    task_index: int
    repetition: int
    evaluations: np.ndarray
    y_p: np.ndarray
    y_p_best_so_far_all: np.ndarray
    constraint_utilizations: Mapping[str, np.ndarray]
    feasible_mask: np.ndarray | None
    max_evaluations: int
    global_optimum: float | None
    target_reference: str | None
    target_hit: bool | None
    evaluations_to_target: int | None
    wall_time_s: float | None
    dat_path: Path
    problem_id: str = ""

    @property
    def budget(self) -> int:
        """Allocated evaluation ceiling. Not the ERT cost of a shorter run."""
        return int(self.max_evaluations)

    @property
    def n_evaluations(self) -> int:
        """Number of logged evaluations in the ``.dat`` file."""
        return int(self.evaluations.shape[0])


@dataclass(frozen=True)
class ExperimentBundle:
    """
    One or more loaded experiment directories ready for overlay plotting.

    Parameters
    ----------
    runs : tuple of RunTrajectory
        All completed runs in stable order (by experiment path, then
        ``task_index``).
    experiment_dirs : tuple of Path
        Distinct experiment roots in load order.
    """

    runs: tuple[RunTrajectory, ...]
    experiment_dirs: tuple[Path, ...]

    def problem_ids(self) -> tuple[str, ...]:
        """Return unique problem IDs in first-seen run order."""
        seen: list[str] = []
        for run in self.runs:
            if run.problem_id not in seen:
                seen.append(run.problem_id)
        return tuple(seen)

    def for_problem(self, problem_id: str) -> ExperimentBundle:
        """Return a new bundle containing only runs for ``problem_id``."""
        available = self.problem_ids()
        if problem_id not in available:
            raise ValueError(
                f"unknown problem_id {problem_id!r}; available={list(available)!r}"
            )
        return ExperimentBundle(
            runs=tuple(run for run in self.runs if run.problem_id == problem_id),
            experiment_dirs=self.experiment_dirs,
        )

    def overlay_labels(self) -> tuple[str, ...]:
        """
        Return unique overlay labels in first-seen order.

        Returns
        -------
        tuple of str
            Legend keys for multi-algorithm overlays.
        """
        seen: list[str] = []
        for run in self.runs:
            if run.overlay_label not in seen:
                seen.append(run.overlay_label)
        return tuple(seen)

    def runs_for(self, overlay_label: str) -> tuple[RunTrajectory, ...]:
        """
        Return runs belonging to one overlay series.

        Parameters
        ----------
        overlay_label : str
            Legend / grouping key.

        Returns
        -------
        tuple of RunTrajectory
            Matching runs in bundle order.
        """
        return tuple(r for r in self.runs if r.overlay_label == overlay_label)

    def by_overlay(self) -> Mapping[str, tuple[RunTrajectory, ...]]:
        """
        Group runs by overlay label.

        Returns
        -------
        Mapping[str, tuple[RunTrajectory, ...]]
            Overlay label → runs.
        """
        return {label: self.runs_for(label) for label in self.overlay_labels()}
