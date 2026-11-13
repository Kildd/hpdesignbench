"""
Deterministic landscape preprocessing for ELA.

Derives ``X_0`` / ``X_norm`` from sealed catalogue bounds, analysis-local
``V`` / ``phi`` / ``F`` from selected utilizations, and per-repetition
min–max ``z̃`` for sampled pflacco families. Exhaustive metrics stay on
raw ``y`` / ``phi``. Do not write extra Parquet columns. Do not use
sample min/max for X. Do not use stored ``feasible`` as ``F``.

This module must not import DuckDB, pflacco, or SALib. DuckDB SQL is
returned as text for exhaustive scans; the caller imports DuckDB.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.ioh_core.workflows.la.models import (
    ElaError,
    ElaNotImplementedError,
    STATUS_CONSTANT_LANDSCAPE,
    STATUS_OK,
)

CONSTANT_RESPONSE_WARNING = (
    "response landscape is constant (z_max == z_min); "
    "z-tilde is unset; raw values are preserved. "
    "Sampled ELA features that need variation are undefined on this landscape."
)


def preprocess_landscape(*args, **kwargs):
    """Build X_norm / z-tilde views. Orchestration lands in a later tick."""
    raise ElaNotImplementedError("ELA preprocessing is not implemented yet")


@dataclass(frozen=True)
class MinMaxResponse:
    """
    Per-repetition min–max view of one response landscape.

    Attributes
    ----------
    z_raw : ndarray
        Copy of the input response (never rewritten).
    z_tilde : ndarray or None
        Unit-interval map, or ``None`` when the landscape is constant.
    z_min, z_max : float
        Min and max of this repetition’s raw ``z``.
    response_view : str
        ``minmax`` when scaled; ``unset`` when ``z_max == z_min``.
    """

    z_raw: NDArray[np.floating]
    z_tilde: NDArray[np.floating] | None
    z_min: float
    z_max: float
    response_view: str


def minmax_response(z: ArrayLike) -> MinMaxResponse:
    """
    Min–max one response vector from one repetition.

    ``z̃ = (z − min z) / (max z − min z)``. Call separately for
    ``y_raw``, ``y_p``, and ``phi``. Do not pool repetitions. Do not
    apply this map to exhaustive engineering metrics.

    If ``max == min``, emit :data:`CONSTANT_RESPONSE_WARNING`, leave
    ``z_tilde`` unset, and preserve raw values. Does not reject the source.
    """
    raw = np.asarray(z, dtype=float).reshape(-1).copy()
    if raw.size < 1:
        raise ElaError("response vector must be non-empty")
    if not np.isfinite(raw).all():
        raise ElaError("response vector must be finite to form z-tilde")
    z_min = float(np.min(raw))
    z_max = float(np.max(raw))
    if z_max == z_min:
        warnings.warn(CONSTANT_RESPONSE_WARNING, RuntimeWarning, stacklevel=2)
        return MinMaxResponse(
            z_raw=raw,
            z_tilde=None,
            z_min=z_min,
            z_max=z_max,
            response_view="unset",
        )
    scaled = (raw - z_min) / (z_max - z_min)
    return MinMaxResponse(
        z_raw=raw,
        z_tilde=scaled,
        z_min=z_min,
        z_max=z_max,
        response_view="minmax",
    )


def landscape_status(view: MinMaxResponse) -> str:
    """
    Feature-level status for one response landscape.

    ``constant_landscape`` when ``z_max == z_min`` (``z_tilde`` unset).
    ``ok`` when the landscape varies. Status is a property of the
    landscape: there is no ``needs_variation`` flag. Does not reject
    the source and does not fabricate ``z̃``. Family wrappers in later
    ticks must consult this instead of inventing a scaled vector.
    """
    if view.z_tilde is None or view.response_view == "unset":
        return STATUS_CONSTANT_LANDSCAPE
    return STATUS_OK


def normalize_decision_coordinates(
    X_index: ArrayLike,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> tuple[NDArray[np.integer], NDArray[np.floating]]:
    """
    Map catalogue indices to zero-based and unit-interval coordinates.

    Parameters
    ----------
    X_index : array-like
        ``(n, d)`` catalogue indices in sealed ``var_names`` order.
        A length-``n`` vector is accepted when ``d == 1``.
    lower_bounds, upper_bounds : sequence of int
        Inclusive sealed catalogue bounds. Must match ``d``.
        Sample min/max must not be passed here.

    Returns
    -------
    X_0 : ndarray of int
        ``X_index - lb``.
    X_norm : ndarray of float
        ``(X_index - lb) / (ub - lb)``.

    Raises
    ------
    ElaError
        If shapes mismatch or any ``ub == lb`` (cannot form ``X_norm``).
    """
    lowers, uppers = _require_bounds(lower_bounds, upper_bounds)
    values = np.asarray(X_index)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ElaError(f"X_index must be 1-d or 2-d, got shape {values.shape}")
    dimension = values.shape[1]
    if dimension != len(lowers):
        raise ElaError(
            f"X_index has d={dimension} but bounds have length {len(lowers)}"
        )
    lower = np.asarray(lowers, dtype=np.int64)
    span = np.asarray(uppers, dtype=np.int64) - lower
    x0 = np.asarray(values, dtype=np.int64) - lower
    x_norm = x0.astype(float) / span.astype(float)
    return x0, x_norm


def constraint_violations(
    U: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_]]:
    """
    Recompute analysis-local ``V``, ``phi``, and ``F`` from selected utils.

    ``v_i = max(u_i - 1, 0)``. ``phi = sum_i v_i``. ``F`` is true iff every
    selected ``v_i`` is exactly ``0`` (no epsilon; same as
    ``_constraint_feasible``). Does not read stored ``feasible``. Constant
    constraint columns are kept.

    Parameters
    ----------
    U : array-like
        ``(n, k)`` utilizations for the selected constraint set.
        A length-``n`` vector is accepted when ``k == 1``.

    Returns
    -------
    V : ndarray of float
        ``(n, k)`` violations.
    phi : ndarray of float
        ``(n,)`` sums over selected constraints.
    F : ndarray of bool
        ``(n,)`` analysis-local feasibility.
    """
    values = np.asarray(U, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ElaError(f"util array must be 2-d, got shape {values.shape}")
    if values.shape[1] < 1:
        raise ElaError("util array must have at least one selected constraint")
    violation = np.maximum(values - 1.0, 0.0)
    phi = np.sum(violation, axis=1)
    feasible = np.all(violation == 0.0, axis=1)
    return violation, phi, feasible


def constraint_violation_sql(constraint_names: Sequence[str]) -> str:
    """
    Return a DuckDB SELECT list for ``v__*``, ``phi``, and ``F``.

    Uses selected ``util__{{name}}`` columns. Does not import DuckDB.
    Does not reference stored ``feasible`` / ``constraint_feasible``.

    Parameters
    ----------
    constraint_names : sequence of str
        Selected constraint names (sealed relative order).

    Returns
    -------
    str
        Comma-separated SELECT items.
    """
    names = _require_var_names(constraint_names)
    v_exprs: list[str] = []
    pieces: list[str] = []
    for name in names:
        util = _quote_ident(f"util__{name}")
        v_expr = f"GREATEST(CAST({util} AS DOUBLE) - 1.0, 0.0)"
        v_exprs.append(v_expr)
        pieces.append(f"{v_expr} AS {_quote_ident(f'v__{name}')}")
    pieces.append(analysis_local_phi_f_sql(constraint_names))
    return ", ".join(pieces)


def analysis_local_phi_f_sql(constraint_names: Sequence[str]) -> str:
    """Return ``phi`` / ``F`` SELECT items from selected ``util__*`` columns."""
    names = _require_var_names(constraint_names)
    v_exprs = [
        f"GREATEST(CAST({_quote_ident(f'util__{name}')} AS DOUBLE) - 1.0, 0.0)"
        for name in names
    ]
    phi_expr = " + ".join(f"({expr})" for expr in v_exprs)
    f_expr = " AND ".join(f"(({expr}) = 0.0)" for expr in v_exprs)
    return (
        f"({phi_expr}) AS {_quote_ident('phi')}, "
        f"({f_expr}) AS {_quote_ident('F')}"
    )


def decision_coordinate_sql(
    var_names: Sequence[str],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> str:
    """
    Return a DuckDB SELECT list for ``x0__*`` and ``xnorm__*``.

    Uses sealed ``idx__{{name}}`` columns and catalogue bounds. Does not
    import DuckDB. Does not reference ``param__*``.

    Parameters
    ----------
    var_names : sequence of str
        Sealed variable names (same order as bounds).
    lower_bounds, upper_bounds : sequence of int
        Inclusive sealed catalogue bounds.

    Returns
    -------
    str
        Comma-separated SELECT items, e.g.
        ``("idx__t" - 0) AS "x0__t", ...``.
    """
    names = _require_var_names(var_names)
    lowers, uppers = _require_bounds(lower_bounds, upper_bounds)
    if len(names) != len(lowers):
        raise ElaError(
            f"var_names length {len(names)} does not match bounds {len(lowers)}"
        )
    pieces: list[str] = []
    for name, lo, hi in zip(names, lowers, uppers):
        idx = _quote_ident(f"idx__{name}")
        span = int(hi) - int(lo)
        pieces.append(f"({idx} - {int(lo)}) AS {_quote_ident(f'x0__{name}')}")
        pieces.append(
            f"((CAST({idx} AS DOUBLE) - {int(lo)}) / {float(span)}) "
            f"AS {_quote_ident(f'xnorm__{name}')}"
        )
    return ", ".join(pieces)


def _require_var_names(var_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(var_names, (str, bytes)):
        raise ElaError("var_names must be a sequence of strings")
    names = tuple(str(name) for name in var_names)
    if not names or not all(name.strip() and name == name.strip() for name in names):
        raise ElaError("var_names must be a non-empty sequence of non-blank strings")
    return names


def _require_bounds(
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(lower_bounds, (str, bytes)) or isinstance(upper_bounds, (str, bytes)):
        raise ElaError("bounds must be sequences of integers")
    try:
        lowers = tuple(int(value) for value in lower_bounds)
        uppers = tuple(int(value) for value in upper_bounds)
    except (TypeError, ValueError) as exc:
        raise ElaError("bounds must be sequences of integers") from exc
    if not lowers or len(lowers) != len(uppers):
        raise ElaError(
            "lower_bounds and upper_bounds must be non-empty and the same length"
        )
    for index, (lo, hi) in enumerate(zip(lowers, uppers)):
        if hi < lo:
            raise ElaError(
                f"upper_bounds[{index}]={hi} is below lower_bounds[{index}]={lo}"
            )
        if hi == lo:
            raise ElaError(
                "cannot form X_norm: upper bound equals lower bound at "
                f"index {index} (ub={hi}, lb={lo})"
            )
    return lowers, uppers


def _quote_ident(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ElaError(f"identifier must be a non-empty string, got {name!r}")
    return '"' + name.replace('"', '""') + '"'
