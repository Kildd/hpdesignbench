"""
Convergence aggregation and Hansen expected running time (ERT).

An unsuccessful run contributes the evaluations it performed. The allocated
maximum budget is a separate ceiling and is not substituted for a shorter log.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.visualization_core.optimization.models import (
    DisplayMode,
    ExperimentBundle,
    RunTrajectory,
    SuccessFilter,
    TargetReference,
)
from core.visualization_core.optimization.trajectories import (
    feasible_share,
    first_feasible_evaluation,
    hit_time_at_percent,
)


class AggregationError(ValueError):
    """Raised when aggregation inputs violate Phase V1 locks."""


@dataclass(frozen=True)
class ConvergenceSeries:
    """
    One anytime quality series for plotting.

    Parameters
    ----------
    evaluations : ndarray
        Evaluation indices on the x-axis.
    y_p_best_so_far : ndarray
        Best-so-far quality values.
    label : str
        Legend label (overlay or per-run label).
    kind : str
        ``central``, ``best_run``, or ``individual``.
    """

    evaluations: np.ndarray
    y_p_best_so_far: np.ndarray
    label: str
    kind: str


@dataclass(frozen=True)
class ErtResult:
    """
    Hansen ERT at one target percentage.

    Parameters
    ----------
    percent : float
        Relative target deviation percent.
    ert : float or None
        ``sum(T_i) / n_s`` when ``n_s > 0``; ``None`` when undefined.
    n_success : int
        Number of successful runs ``n_s``.
    n_runs : int
        Total runs ``n``.
    sum_t : float
        Numerator. A success contributes its first hitting evaluation. An
        unsuccessful run contributes the evaluations it performed.
    """

    percent: float
    ert: float | None
    n_success: int
    n_runs: int
    sum_t: float

    @property
    def defined(self) -> bool:
        """True when ERT is finite and defined."""
        return self.ert is not None and math.isfinite(float(self.ert))


def _final_best(run: RunTrajectory) -> float:
    curve = run.y_p_best_so_far_all
    if curve.size == 0:
        return math.inf
    return float(curve[-1])


def select_best_run(runs: Sequence[RunTrajectory]) -> RunTrajectory:
    """
    Select the single run with the lowest final best-so-far ``y_p``.

    Ties break toward the lower ``task_index``, then lower ``repetition``.

    Parameters
    ----------
    runs : sequence of RunTrajectory
        Candidate runs (one overlay group).

    Returns
    -------
    RunTrajectory
        Winning run.
    """
    if not runs:
        raise AggregationError("select_best_run requires at least one run")
    return min(
        runs,
        key=lambda r: (_final_best(r), int(r.task_index), int(r.repetition)),
    )


def _aligned_matrix(
    runs: Sequence[RunTrajectory],
    *,
    extend_short_runs: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack best-so-far curves on a common evaluation grid.

    Default lock: grid is ``1 .. B_common`` with ``B_common = min length``.
    When ``extend_short_runs`` is true, grid extends to ``max length`` and
    shorter runs carry forward their last best-so-far value.
    """
    if not runs:
        raise AggregationError("alignment requires at least one run")
    lengths = [int(r.n_evaluations) for r in runs]
    if any(length < 1 for length in lengths):
        raise AggregationError("every run must have at least one evaluation")
    if extend_short_runs:
        target = max(lengths)
    else:
        target = min(lengths)
    evaluations = np.arange(1, target + 1, dtype=int)
    matrix = np.empty((len(runs), target), dtype=float)
    for i, run in enumerate(runs):
        curve = np.asarray(run.y_p_best_so_far_all, dtype=float)
        n = int(curve.shape[0])
        if n >= target:
            matrix[i, :] = curve[:target]
        else:
            if not extend_short_runs:
                raise AggregationError(
                    "internal alignment error: short run under B_common mode"
                )
            matrix[i, :n] = curve
            matrix[i, n:] = curve[-1]
    return evaluations, matrix


@dataclass(frozen=True)
class ConvergenceBand:
    """
    Median central curve with IQR ribbon on a common evaluation grid.

    Parameters
    ----------
    evaluations : ndarray
        Evaluation indices.
    median : ndarray
        Pointwise median best-so-far.
    q25 : ndarray
        Pointwise 25th percentile.
    q75 : ndarray
        Pointwise 75th percentile.
    label : str
        Overlay label.
    """

    evaluations: np.ndarray
    median: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    label: str


def median_iqr_band(
    runs: Sequence[RunTrajectory],
    *,
    extend_short_runs: bool = True,
    overlay_label: str | None = None,
) -> ConvergenceBand:
    """
    Compute median and IQR of best-so-far curves on the common grid.

    Requires at least two runs. Used for the median-mode convergence ribbon.

    Parameters
    ----------
    runs : sequence of RunTrajectory
        Overlay group.
    extend_short_runs : bool
        Alignment policy matching ``aggregate_convergence``.
    overlay_label : str or None
        Legend label override.

    Returns
    -------
    ConvergenceBand
        Median and quartile curves.
    """
    if len(runs) < 2:
        raise AggregationError("median_iqr_band requires at least two runs")
    label = overlay_label or runs[0].overlay_label
    evaluations, matrix = _aligned_matrix(
        runs, extend_short_runs=extend_short_runs
    )
    return ConvergenceBand(
        evaluations=evaluations,
        median=np.median(matrix, axis=0),
        q25=np.percentile(matrix, 25, axis=0),
        q75=np.percentile(matrix, 75, axis=0),
        label=label,
    )


def aggregate_convergence(
    runs: Sequence[RunTrajectory],
    *,
    display: DisplayMode = "median",
    extend_short_runs: bool = True,
    overlay_label: str | None = None,
) -> tuple[ConvergenceSeries, ...]:
    """
    Build convergence series for one overlay group.

    Parameters
    ----------
    runs : sequence of RunTrajectory
        Runs sharing one overlay label.
    display : DisplayMode
        ``median``, ``mean``, ``best``, or ``all``.
    extend_short_runs : bool
        Carry-forward past ``B_common`` when true.
    overlay_label : str or None
        Legend label for central/best curves; defaults to the runs' overlay.

    Returns
    -------
    tuple of ConvergenceSeries
        One central/best series, or many individual series for ``all``.
    """
    if not runs:
        raise AggregationError("aggregate_convergence requires at least one run")
    label = overlay_label or runs[0].overlay_label

    if display == "all":
        series: list[ConvergenceSeries] = []
        for run in runs:
            series.append(
                ConvergenceSeries(
                    evaluations=np.asarray(run.evaluations, dtype=int).copy(),
                    y_p_best_so_far=np.asarray(
                        run.y_p_best_so_far_all, dtype=float
                    ).copy(),
                    label=f"{label}#{run.task_index}",
                    kind="individual",
                )
            )
        return tuple(series)

    if display == "best":
        winner = select_best_run(runs)
        return (
            ConvergenceSeries(
                evaluations=np.asarray(winner.evaluations, dtype=int).copy(),
                y_p_best_so_far=np.asarray(
                    winner.y_p_best_so_far_all, dtype=float
                ).copy(),
                label=label,
                kind="best_run",
            ),
        )

    if display not in {"mean", "median"}:
        raise AggregationError(f"unknown display mode {display!r}")

    evaluations, matrix = _aligned_matrix(
        runs, extend_short_runs=extend_short_runs
    )
    if display == "mean":
        central = np.mean(matrix, axis=0)
    else:
        central = np.median(matrix, axis=0)
    return (
        ConvergenceSeries(
            evaluations=evaluations,
            y_p_best_so_far=np.asarray(central, dtype=float),
            label=label,
            kind="central",
        ),
    )


def runtime_sample(
    runs: Sequence[RunTrajectory],
    percent: float,
    *,
    success_filter: SuccessFilter = "successes",
    global_optimum: float | None = None,
    target_reference: TargetReference | None = None,
) -> tuple[np.ndarray, int, int]:
    """
    Collect fixed-target runtimes for box / histogram / ECDF plots.

    Parameters
    ----------
    runs : sequence of RunTrajectory
        Overlay group.
    percent : float
        Target deviation percent.
    success_filter : SuccessFilter
        ``successes`` keeps hits only. ``all_runs`` records the evaluations
        performed by each miss.
    global_optimum : float or None
        Optimum override.
    target_reference : TargetReference or None
        Reference override.

    Returns
    -------
    times : ndarray
        Runtime sample (may be empty under ``successes`` with zero hits).
    n_success : int
        Success count.
    n_runs : int
        Total runs.
    """
    if not runs:
        raise AggregationError("runtime_sample requires at least one run")
    times: list[float] = []
    n_success = 0
    for run in runs:
        hit = hit_time_at_percent(
            run,
            percent,
            global_optimum=global_optimum,
            target_reference=target_reference,
        )
        if hit is not None:
            n_success += 1
            times.append(float(hit))
        elif success_filter == "all_runs":
            times.append(float(run.n_evaluations))
        elif success_filter != "successes":
            raise AggregationError(f"unknown success_filter {success_filter!r}")
    return np.asarray(times, dtype=float), n_success, len(runs)


@dataclass(frozen=True)
class SuccessProbabilityCurve:
    """
    Success-probability ECDF for one algorithm overlay.

    ``F(t) = (# runs that hit by t) / n_runs``, so the asymptote equals the
    empirical success rate (e.g. 10/20 → 0.5). Misses never contribute a
    finite hitting time.

    Parameters
    ----------
    hit_times : ndarray
        Sorted successful hitting times (evaluations), length ``n_success``.
    n_success : int
        Number of runs that hit the target.
    n_runs : int
        Total runs in the overlay (denominator).
    x_max : float
        Right edge for drawing the plateau (typically ``max`` hard budget).
    """

    hit_times: np.ndarray
    n_success: int
    n_runs: int
    x_max: float

    @property
    def success_rate(self) -> float:
        """Empirical success rate ``n_success / n_runs``."""
        if self.n_runs <= 0:
            return 0.0
        return float(self.n_success) / float(self.n_runs)


def success_probability_curve(
    runs: Sequence[RunTrajectory],
    percent: float,
    *,
    global_optimum: float | None = None,
    target_reference: TargetReference | None = None,
) -> SuccessProbabilityCurve:
    """
    Build the all-runs success-probability ECDF sample for one overlay.

    Parameters
    ----------
    runs : sequence of RunTrajectory
        Overlay group (all repetitions).
    percent : float
        Target deviation percent.
    global_optimum : float or None
        Optimum override.
    target_reference : TargetReference or None
        Reference override.

    Returns
    -------
    SuccessProbabilityCurve
        Sorted hit times and denominator metadata.

    Raises
    ------
    AggregationError
        If ``runs`` is empty.
    """
    if not runs:
        raise AggregationError("success_probability_curve requires at least one run")
    hits: list[float] = []
    budgets: list[float] = []
    for run in runs:
        budgets.append(float(run.budget))
        hit = hit_time_at_percent(
            run,
            percent,
            global_optimum=global_optimum,
            target_reference=target_reference,
        )
        if hit is not None:
            hits.append(float(hit))
    hit_times = np.asarray(sorted(hits), dtype=float)
    x_max = float(max(budgets)) if budgets else 1.0
    if hit_times.size:
        x_max = max(x_max, float(hit_times[-1]))
    return SuccessProbabilityCurve(
        hit_times=hit_times,
        n_success=int(hit_times.size),
        n_runs=len(runs),
        x_max=x_max,
    )


def hansen_ert(
    runs: Sequence[RunTrajectory],
    percent: float,
    *,
    global_optimum: float | None = None,
    target_reference: TargetReference | None = None,
) -> ErtResult:
    """
    Hansen expected running time at one target percentage.

    A success contributes the first evaluation at which the target is
    reached. An unsuccessful run contributes the evaluations stored in its
    trajectory. The allocated ``max_evaluations`` ceiling is not used.
    Independent of plot ``success_filter``.

    Parameters
    ----------
    runs : sequence of RunTrajectory
        Overlay group.
    percent : float
        Target deviation percent.
    global_optimum : float or None
        Optimum override.
    target_reference : TargetReference or None
        Reference override.

    Returns
    -------
    ErtResult
        ERT summary; ``ert`` is ``None`` when ``n_s = 0``.

    Raises
    ------
    AggregationError
        If ``runs`` is empty, a trajectory is empty, or a hitting time
        lies beyond the logged evaluations.
    """
    if not runs:
        raise AggregationError("hansen_ert requires at least one run")
    hits: list[int | None] = []
    performed: list[int] = []
    for run in runs:
        hits.append(
            hit_time_at_percent(
                run,
                percent,
                global_optimum=global_optimum,
                target_reference=target_reference,
            )
        )
        performed.append(int(run.n_evaluations))
    result = hansen_ert_from_outcomes(
        hits,
        performed,
        percent=float(percent),
        budget=max(performed),
    )
    return ErtResult(
        percent=result.percent,
        ert=result.ert,
        n_success=result.n_success,
        n_runs=result.n_runs,
        sum_t=result.sum_t,
    )


@dataclass(frozen=True)
class BudgetErtResult:
    """
    Hansen ERT when every run is observed only up to a selected budget.

    A hit counts only when it falls at or before
    ``min(evaluations performed, budget)``. Each unsuccessful run contributes
    that same censored length. ``ert`` is ``None`` when no run succeeds.
    ``failure_cap`` is that shared length when every unsuccessful run uses
    it, and ``None`` when the lengths differ or no run fails.

    Parameters
    ----------
    percent : float
        Relative target deviation percent.
    ert : float or None
        ``sum_t / n_success`` when ``n_success > 0``.
    n_success : int
        Runs that hit the target within the cap.
    n_runs : int
        Runs in the denominator.
    sum_t : float
        ``sum_hits + failure_cost``.
    sum_hits : float
        Sum of the first-hit evaluations of the successful runs.
    failure_cost : float
        Sum of the censored evaluation counts charged to unsuccessful runs.
    failure_cap : int or None
        Shared unsuccessful-run count, when every miss uses the same one.
    """

    percent: float
    ert: float | None
    n_success: int
    n_runs: int
    sum_t: float
    sum_hits: float
    failure_cost: float
    failure_cap: int | None

    @property
    def defined(self) -> bool:
        """True when ERT is finite and defined."""
        return self.ert is not None and math.isfinite(float(self.ert))


def hansen_ert_from_outcomes(
    hits: Sequence[int | None],
    evaluations_performed: Sequence[int],
    *,
    percent: float,
    budget: int,
) -> BudgetErtResult:
    """
    Hansen ERT from first-hit times and the evaluations each run performed.

    Parameters
    ----------
    hits : sequence of int or None
        First hitting evaluation of each run, or ``None`` when the logged
        run never reaches the target. A hit later than the censored length
        counts as a miss.
    evaluations_performed : sequence of int
        Evaluations stored for each run, aligned with ``hits``. This is the
        length of the trial, not the allocated maximum budget.
    percent : float
        Target deviation percent recorded on the result.
    budget : int
        Selected evaluation budget. A contribution is censored at this value.
        Must be positive.

    Returns
    -------
    BudgetErtResult
        ERT at ``percent`` within ``budget``. ``ert`` is ``None`` when no
        run succeeds.

    Raises
    ------
    AggregationError
        If ``hits`` is empty, the sequences differ in length, a count is
        not positive, or a hitting time lies beyond the logged evaluations.
    """
    if len(hits) != len(evaluations_performed):
        raise AggregationError(
            "hits and evaluations performed must have the same length"
        )
    if not hits:
        raise AggregationError("hansen_ert requires at least one run")
    selected = int(budget)
    if selected <= 0:
        raise AggregationError(
            f"evaluation budget must be positive, got {budget}"
        )
    contributions: list[float] = []
    n_success = 0
    sum_hits = 0.0
    failure_cost = 0.0
    failure_caps: set[int] = set()
    for hit, performed in zip(hits, evaluations_performed, strict=True):
        logged = int(performed)
        if logged <= 0:
            raise AggregationError(
                f"evaluations performed must be positive, got {performed}"
            )
        if hit is not None and int(hit) > logged:
            raise AggregationError(
                f"hitting evaluation {hit} exceeds the {logged} evaluations performed"
            )
        cap = min(logged, selected)
        if hit is not None and int(hit) <= cap:
            n_success += 1
            contributions.append(float(int(hit)))
            sum_hits += float(int(hit))
        else:
            contributions.append(float(cap))
            failure_cost += float(cap)
            failure_caps.add(cap)
    sum_t = float(sum(contributions))
    n_runs = len(hits)
    shared_cap = next(iter(failure_caps)) if len(failure_caps) == 1 else None
    ert = None if n_success == 0 else sum_t / float(n_success)
    return BudgetErtResult(
        percent=float(percent),
        ert=ert,
        n_success=n_success,
        n_runs=n_runs,
        sum_t=sum_t,
        sum_hits=sum_hits,
        failure_cost=failure_cost,
        failure_cap=shared_cap,
    )


def hansen_ert_value(
    hits: Sequence[int | None],
    evaluations_performed: Sequence[int],
    *,
    budget: int,
    percent: float = 1.0,
) -> float | None:
    """
    Return Hansen ERT, or ``None`` when no run succeeds.

    Parameters
    ----------
    hits : sequence of int or None
        First hitting evaluation, or ``None`` on a miss.
    evaluations_performed : sequence of int
        Evaluations performed by each run. Not the allocated budget.
    budget : int
        Selected evaluation budget used to censor each contribution.
    percent : float, optional
        Target percent recorded on the result. Default 1.

    Returns
    -------
    float or None
        ``sum(N_r) / n_success``, or ``None`` when ``n_success`` is zero.
    """
    return hansen_ert_from_outcomes(
        hits,
        evaluations_performed,
        percent=float(percent),
        budget=int(budget),
    ).ert


def warm_instance_deployment_ert(
    solver_ert: float | None,
    *,
    predesign_evaluations: int = 2700,
) -> float | None:
    """
    Add one instance's predesign grid to a solver-only ERT.

    The grid is built once for that instance. It is not multiplied by the
    number of unsuccessful runs, and this function is not applied to a
    pooled ERT.

    Parameters
    ----------
    solver_ert : float or None
        Solver-only ERT. ``None`` stays undefined.
    predesign_evaluations : int, optional
        Evaluations in the predesign grid. Default 2,700.

    Returns
    -------
    float or None
        ``solver_ert + predesign_evaluations``, or ``None``.
    """
    if solver_ert is None:
        return None
    grid = int(predesign_evaluations)
    if grid < 0:
        raise AggregationError(
            f"predesign evaluations must be >= 0, got {predesign_evaluations}"
        )
    return float(solver_ert) + float(grid)


def hansen_ert_within_budget(
    runs: Sequence[RunTrajectory],
    percent: float,
    budget: int,
    *,
    global_optimum: float | None = None,
    target_reference: TargetReference | None = None,
) -> BudgetErtResult:
    """
    Hansen ERT at one target, counting only evaluations up to ``budget``.

    A miss contributes ``min(evaluations performed, budget)``. When
    ``budget`` is at least every logged length and every hit falls inside
    that length, the result matches :func:`hansen_ert`.

    Parameters
    ----------
    runs : sequence of RunTrajectory
        Overlay group.
    percent : float
        Target deviation percent.
    budget : int
        Selected evaluation budget.
    global_optimum : float or None
        Optimum override.
    target_reference : TargetReference or None
        Reference override.

    Returns
    -------
    BudgetErtResult
        Budget-limited ERT summary.
    """
    if not runs:
        raise AggregationError("hansen_ert requires at least one run")
    hits: list[int | None] = []
    budgets: list[int] = []
    for run in runs:
        hits.append(
            hit_time_at_percent(
                run,
                percent,
                global_optimum=global_optimum,
                target_reference=target_reference,
            )
        )
        budgets.append(int(run.n_evaluations))
    return hansen_ert_from_outcomes(
        hits,
        budgets,
        percent=float(percent),
        budget=int(budget),
    )


@dataclass(frozen=True)
class AlgorithmFeasibilitySummary:
    """
    Per-algorithm feasibility rollup (independent of fixed-target percent).

    Parameters
    ----------
    algorithm : str
        Overlay label.
    n_runs : int
        Completed runs in the overlay.
    n_runs_with_feasible : int or None
        Runs that recorded at least one feasible evaluation; ``None`` when
        masks are unavailable.
    mean_first_feasible_fe : float or None
        Mean evaluation index of the first feasible sample over runs that
        found one; ``None`` if no such run (or masks unavailable).
    mean_share_feasible : float or None
        Mean over runs of the share of feasible evaluations; ``None`` when
        masks are unavailable.
    """

    algorithm: str
    n_runs: int
    n_runs_with_feasible: int | None
    mean_first_feasible_fe: float | None
    mean_share_feasible: float | None


@dataclass(frozen=True)
class AlgorithmTargetSummary:
    """
    Per-algorithm fixed-target summary row for one target percent.

    Parameters
    ----------
    algorithm : str
        Overlay label.
    percent : float
        Target deviation percent.
    n_runs : int
        Completed runs in the overlay.
    n_success : int
        Runs that hit the target.
    success_rate : float
        ``n_success / n_runs``.
    e2t_min : float or None
        Fastest successful hitting time (FE), or ``None`` if no hits.
    e2t_max : float or None
        Slowest successful hitting time (FE).
    e2t_mean : float or None
        Mean successful hitting time (FE).
    e2t_median : float or None
        Median successful hitting time (FE).
    e2t_std : float or None
        Sample std of successful hitting times (``ddof=1``), or ``None``
        when fewer than two successes.
    ert : float or None
        Hansen ERT; ``None`` when ``n_success = 0``.
    budget : int
        Hard evaluation budget ``B`` used for ERT (max over runs).
    n_runs_with_feasible : int or None
        Runs with ≥1 feasible evaluation; ``None`` when masks unavailable.
    mean_first_feasible_fe : float or None
        Mean first-feasible evaluation index (runs with ≥1 feasible only).
    mean_share_feasible : float or None
        Mean share of feasible evaluations across all runs.
    mean_wall_time_s : float or None
        Mean wall-clock seconds over runs with a finite ``wall_time_s``.
    """

    algorithm: str
    percent: float
    n_runs: int
    n_success: int
    success_rate: float
    e2t_min: float | None
    e2t_max: float | None
    e2t_mean: float | None
    e2t_median: float | None
    e2t_std: float | None
    ert: float | None
    budget: int
    n_runs_with_feasible: int | None
    mean_first_feasible_fe: float | None
    mean_share_feasible: float | None
    mean_wall_time_s: float | None


def mean_wall_time_seconds(runs: Sequence[RunTrajectory]) -> float | None:
    """
    Mean wall-clock seconds over runs with a finite ``wall_time_s``.

    Returns ``None`` when no run reports a usable wall time.
    """
    values = [
        float(run.wall_time_s)
        for run in runs
        if run.wall_time_s is not None and math.isfinite(float(run.wall_time_s))
    ]
    if not values:
        return None
    return float(np.mean(values))


@dataclass(frozen=True)
class AlgorithmWallTimeSummary:
    """Per-algorithm mean wall-clock seconds."""

    algorithm: str
    n_runs: int
    n_runs_with_wall_time: int
    mean_wall_time_s: float | None


def algorithm_wall_time_summaries(
    bundle: ExperimentBundle,
) -> tuple[AlgorithmWallTimeSummary, ...]:
    """Build per-algorithm mean wall-time rows."""
    rows: list[AlgorithmWallTimeSummary] = []
    for label in bundle.overlay_labels():
        runs = bundle.runs_for(label)
        if not runs:
            continue
        mean_s = mean_wall_time_seconds(runs)
        n_timed = sum(
            1
            for run in runs
            if run.wall_time_s is not None
            and math.isfinite(float(run.wall_time_s))
        )
        rows.append(
            AlgorithmWallTimeSummary(
                algorithm=label,
                n_runs=len(runs),
                n_runs_with_wall_time=n_timed,
                mean_wall_time_s=mean_s,
            )
        )
    return tuple(rows)


def algorithm_feasibility_summaries(
    bundle: ExperimentBundle,
) -> tuple[AlgorithmFeasibilitySummary, ...]:
    """
    Build per-algorithm feasibility means from reconstructed ``feasible_mask``.

    Fail closed: if any run in an overlay lacks a mask, both means are
    ``None`` for that overlay.
    """
    rows: list[AlgorithmFeasibilitySummary] = []
    for label in bundle.overlay_labels():
        runs = bundle.runs_for(label)
        if not runs:
            continue
        rows.append(_feasibility_summary_for_runs(label, runs))
    return tuple(rows)


def _feasibility_summary_for_runs(
    algorithm: str,
    runs: Sequence[RunTrajectory],
) -> AlgorithmFeasibilitySummary:
    n_runs = len(runs)
    if any(run.feasible_mask is None for run in runs):
        return AlgorithmFeasibilitySummary(
            algorithm=algorithm,
            n_runs=n_runs,
            n_runs_with_feasible=None,
            mean_first_feasible_fe=None,
            mean_share_feasible=None,
        )
    firsts: list[float] = []
    shares: list[float] = []
    for run in runs:
        first = first_feasible_evaluation(run)
        if first is not None:
            firsts.append(float(first))
        share = feasible_share(run)
        if share is None:
            return AlgorithmFeasibilitySummary(
                algorithm=algorithm,
                n_runs=n_runs,
                n_runs_with_feasible=None,
                mean_first_feasible_fe=None,
                mean_share_feasible=None,
            )
        shares.append(share)
    return AlgorithmFeasibilitySummary(
        algorithm=algorithm,
        n_runs=n_runs,
        n_runs_with_feasible=len(firsts),
        mean_first_feasible_fe=(
            float(np.mean(firsts)) if firsts else None
        ),
        mean_share_feasible=float(np.mean(shares)) if shares else None,
    )


def algorithm_target_summaries(
    bundle: ExperimentBundle,
    percents: Sequence[float],
    *,
    global_optimum: float | None = None,
    target_reference: TargetReference | None = None,
) -> tuple[AlgorithmTargetSummary, ...]:
    """
    Build per-algorithm summaries for each target percent in ``percents``.

    Parameters
    ----------
    bundle : ExperimentBundle
        Loaded experiment overlay.
    percents : sequence of float
        Target deviation percents (e.g. ``(1, 2, 5, 10)``).
    global_optimum : float or None
        Optimum override.
    target_reference : TargetReference or None
        Reference override.

    Returns
    -------
    tuple of AlgorithmTargetSummary
        Rows ordered by percent, then overlay label order.
    """
    feasibility_by_label = {
        row.algorithm: row for row in algorithm_feasibility_summaries(bundle)
    }
    rows: list[AlgorithmTargetSummary] = []
    for percent in percents:
        p = float(percent)
        for label in bundle.overlay_labels():
            runs = bundle.runs_for(label)
            if not runs:
                continue
            feas = feasibility_by_label.get(label)
            n_with_feas = feas.n_runs_with_feasible if feas else None
            mean_first = feas.mean_first_feasible_fe if feas else None
            mean_share = feas.mean_share_feasible if feas else None
            mean_wall = mean_wall_time_seconds(runs)
            times, n_success, n_runs = runtime_sample(
                runs,
                p,
                success_filter="successes",
                global_optimum=global_optimum,
                target_reference=target_reference,
            )
            ert_result = hansen_ert(
                runs,
                p,
                global_optimum=global_optimum,
                target_reference=target_reference,
            )
            budget = max(int(run.budget) for run in runs)
            if n_success == 0:
                rows.append(
                    AlgorithmTargetSummary(
                        algorithm=label,
                        percent=p,
                        n_runs=n_runs,
                        n_success=0,
                        success_rate=0.0,
                        e2t_min=None,
                        e2t_max=None,
                        e2t_mean=None,
                        e2t_median=None,
                        e2t_std=None,
                        ert=None,
                        budget=budget,
                        n_runs_with_feasible=n_with_feas,
                        mean_first_feasible_fe=mean_first,
                        mean_share_feasible=mean_share,
                        mean_wall_time_s=mean_wall,
                    )
                )
                continue
            arr = np.asarray(times, dtype=float)
            e2t_std: float | None
            if arr.size >= 2:
                e2t_std = float(np.std(arr, ddof=1))
            else:
                e2t_std = None
            rows.append(
                AlgorithmTargetSummary(
                    algorithm=label,
                    percent=p,
                    n_runs=n_runs,
                    n_success=n_success,
                    success_rate=float(n_success) / float(n_runs),
                    e2t_min=float(np.min(arr)),
                    e2t_max=float(np.max(arr)),
                    e2t_mean=float(np.mean(arr)),
                    e2t_median=float(np.median(arr)),
                    e2t_std=e2t_std,
                    ert=ert_result.ert,
                    budget=budget,
                    n_runs_with_feasible=n_with_feas,
                    mean_first_feasible_fe=mean_first,
                    mean_share_feasible=mean_share,
                    mean_wall_time_s=mean_wall,
                )
            )
    return tuple(rows)


def hansen_ert_grid(
    runs: Sequence[RunTrajectory],
    percents: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0),
    *,
    global_optimum: float | None = None,
    target_reference: TargetReference | None = None,
) -> tuple[ErtResult, ...]:
    """
    Compute Hansen ERT for several target percentages.

    Parameters
    ----------
    runs : sequence of RunTrajectory
        Overlay group.
    percents : sequence of float
        Target deviation percents.
    global_optimum : float or None
        Optimum override.
    target_reference : TargetReference or None
        Reference override.

    Returns
    -------
    tuple of ErtResult
        One result per percent, in input order.
    """
    return tuple(
        hansen_ert(
            runs,
            float(percent),
            global_optimum=global_optimum,
            target_reference=target_reference,
        )
        for percent in percents
    )
