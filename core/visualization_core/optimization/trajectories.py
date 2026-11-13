"""
Best-so-far curves, feasibility masks, and post-hoc hitting times.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from core.visualization_core.optimization.models import RunTrajectory, TargetReference


class TrajectoryError(ValueError):
    """Raised when trajectory reconstruction fails closed."""


def best_so_far(y_p: np.ndarray) -> np.ndarray:
    """
    Cumulative minimum of penalised objectives (minimisation).

    Parameters
    ----------
    y_p : ndarray
        Per-evaluation ``y_p`` values.

    Returns
    -------
    ndarray
        Best-so-far ``y_p*(e)`` aligned with ``y_p``.
    """
    values = np.asarray(y_p, dtype=float)
    if values.ndim != 1:
        raise TrajectoryError(f"y_p must be 1-D, got shape {values.shape}")
    if values.size == 0:
        return values.copy()
    return np.minimum.accumulate(values)


def feasible_mask_from_constraints(
    constraint_utilizations: Mapping[str, np.ndarray],
) -> np.ndarray:
    """
    Reconstruct per-evaluation feasibility from logged ``c__*`` utilisations.

    An evaluation is feasible iff every ``c__*`` value is finite and ``<= 1.0``.

    Parameters
    ----------
    constraint_utilizations : Mapping[str, ndarray]
        Non-empty map of ``c__*`` columns.

    Returns
    -------
    ndarray
        Boolean mask, one entry per evaluation.

    Raises
    ------
    TrajectoryError
        If no constraint columns are provided.
    """
    if not constraint_utilizations:
        raise TrajectoryError(
            "feasible reconstruction requires at least one c__* column"
        )
    columns = [np.asarray(v, dtype=float) for v in constraint_utilizations.values()]
    length = columns[0].shape[0]
    if any(col.shape != (length,) for col in columns):
        raise TrajectoryError("c__* columns must share the same 1-D length")
    stacked = np.column_stack(columns)
    finite = np.isfinite(stacked).all(axis=1)
    within = (stacked <= 1.0).all(axis=1)
    return finite & within


def first_feasible_evaluation(run: RunTrajectory) -> int | None:
    """
    Evaluation index of the first feasible sample, or ``None`` if none / unknown.

    Parameters
    ----------
    run : RunTrajectory
        Loaded run with optional ``feasible_mask``.

    Returns
    -------
    int or None
        1-based evaluation index from ``run.evaluations``, or ``None`` when
        the mask is missing or no evaluation is feasible.
    """
    if run.feasible_mask is None:
        return None
    mask = np.asarray(run.feasible_mask, dtype=bool)
    if mask.shape != run.evaluations.shape:
        raise TrajectoryError(
            "feasible_mask and evaluations must have the same shape"
        )
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return None
    return int(run.evaluations[int(idxs[0])])


def feasible_share(run: RunTrajectory) -> float | None:
    """
    Share of logged evaluations that are feasible, or ``None`` if unknown.

    Parameters
    ----------
    run : RunTrajectory
        Loaded run with optional ``feasible_mask``.

    Returns
    -------
    float or None
        ``mean(feasible_mask)`` in ``[0, 1]``, or ``None`` when the mask is
        missing or the trajectory is empty.
    """
    if run.feasible_mask is None:
        return None
    mask = np.asarray(run.feasible_mask, dtype=bool)
    if mask.size == 0:
        return None
    return float(mask.mean())


def target_threshold(global_optimum: float, percent: float) -> float:
    """
    Derive a minimisation threshold from a known optimum and percent deviation.

    Parameters
    ----------
    global_optimum : float
        Known optimum ``y_star`` (must be finite and non-negative).
    percent : float
        Allowed deviation in percent (e.g. ``5.0`` means five percent).

    Returns
    -------
    float
        ``y_star * (1 + percent / 100)``.
    """
    y_star = float(global_optimum)
    p = float(percent)
    if not np.isfinite(y_star) or y_star < 0.0:
        raise TrajectoryError(
            f"global_optimum must be finite and non-negative, got {global_optimum!r}"
        )
    if not np.isfinite(p) or p < 0.0:
        raise TrajectoryError(
            f"percent must be finite and non-negative, got {percent!r}"
        )
    threshold = y_star * (1.0 + p / 100.0)
    if not np.isfinite(threshold):
        raise TrajectoryError(f"derived target threshold is not finite: {threshold!r}")
    return float(threshold)


def first_hitting_time(
    best_so_far_curve: np.ndarray,
    evaluations: np.ndarray,
    threshold: float,
) -> int | None:
    """
    Smallest evaluation index whose best-so-far is at or below ``threshold``.

    Parameters
    ----------
    best_so_far_curve : ndarray
        Best-so-far quality values aligned with ``evaluations``.
    evaluations : ndarray
        1-based IOH evaluation indices.
    threshold : float
        Target threshold ``tau``.

    Returns
    -------
    int or None
        Hitting evaluation index, or ``None`` if the target was never hit.
    """
    curve = np.asarray(best_so_far_curve, dtype=float)
    evals = np.asarray(evaluations)
    if curve.shape != evals.shape:
        raise TrajectoryError("best-so-far and evaluations must align")
    if curve.size == 0:
        return None
    tau = float(threshold)
    if not np.isfinite(tau):
        raise TrajectoryError(f"threshold must be finite, got {threshold!r}")
    hits = np.flatnonzero(np.isfinite(curve) & (curve <= tau))
    if hits.size == 0:
        return None
    return int(evals[int(hits[0])])


def resolve_target_reference(
    run: RunTrajectory,
    override: TargetReference | None,
) -> TargetReference:
    """
    Resolve the target reference for hitting-time reconstruction.

    Parameters
    ----------
    run : RunTrajectory
        Loaded run. The sealed value is metadata; the curve always uses
        every finite ``y_p``.
    override : TargetReference or None
        Explicit plot-config value. Only ``\"all\"`` is accepted.

    Returns
    -------
    TargetReference
        ``\"all\"``.
    """
    del run
    if override is not None and override != "all":
        raise TrajectoryError(
            f"target_reference must be 'all', got {override!r}"
        )
    return "all"


def hitting_curve_for_reference(
    run: RunTrajectory,
    target_reference: TargetReference,
) -> np.ndarray:
    """
    Best-so-far curve used for hitting times.

    Parameters
    ----------
    run : RunTrajectory
        Loaded run.
    target_reference : TargetReference
        ``\"all\"``. Every finite ``y_p`` enters the cumulative minimum.

    Returns
    -------
    ndarray
        Best-so-far series for hitting-time scans.
    """
    if target_reference != "all":
        raise TrajectoryError(
            f"target_reference must be 'all', got {target_reference!r}"
        )
    return run.y_p_best_so_far_all


def hit_time_at_percent(
    run: RunTrajectory,
    percent: float,
    *,
    global_optimum: float | None = None,
    target_reference: TargetReference | None = None,
) -> int | None:
    """
    Post-hoc first hitting time at a relative target percentage.

    Parameters
    ----------
    run : RunTrajectory
        Loaded run.
    percent : float
        Relative deviation percent (1, 2, 5, 10, ...).
    global_optimum : float or None
        Override sealed optimum; required if sealed value is missing.
    target_reference : TargetReference or None
        ``\"all\"`` or omitted. Every finite ``y_p`` enters the curve.

    Returns
    -------
    int or None
        Hitting evaluation, or ``None`` on miss.
    """
    y_star = global_optimum if global_optimum is not None else run.global_optimum
    if y_star is None:
        raise TrajectoryError(
            "global_optimum is required for fixed-target hitting times "
            "(sealed value missing and no override provided)"
        )
    reference = resolve_target_reference(run, target_reference)
    curve = hitting_curve_for_reference(run, reference)
    tau = target_threshold(float(y_star), percent)
    return first_hitting_time(curve, run.evaluations, tau)


def hit_times_at_percents(
    run: RunTrajectory,
    percents: Sequence[float],
    *,
    global_optimum: float | None = None,
    target_reference: TargetReference | None = None,
) -> dict[float, int | None]:
    """
    Compute hitting times for several percentages.

    Parameters
    ----------
    run : RunTrajectory
        Loaded run.
    percents : sequence of float
        Relative deviation percents.
    global_optimum : float or None
        Optimum override.
    target_reference : TargetReference or None
        ``\"all\"`` or omitted. Every finite ``y_p`` enters the curve.

    Returns
    -------
    dict[float, int | None]
        Percent → hitting evaluation (or ``None``).
    """
    return {
        float(p): hit_time_at_percent(
            run,
            float(p),
            global_optimum=global_optimum,
            target_reference=target_reference,
        )
        for p in percents
    }
