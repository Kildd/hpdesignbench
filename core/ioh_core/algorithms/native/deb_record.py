"""
Deb auxiliary-record validation shared by the optimizers and predesign.

Successful analyses require the active constraint channels. Failed analyses
keep the optimizer's current acceptance rules: the five record keys and a
numeric ``y``, with violation contents ignored and ``phi = +inf``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DebAuxiliaryChannels:
    """Validated objective, aggregate violation, and feasibility flags."""

    y: float
    phi: float
    feasible: bool
    analysis_ok: bool
    sentinel_constraints: tuple[str, ...]


def deb_quality_key(*, feasible: bool, y: float, phi: float) -> tuple[int, float]:
    """Return the Deb key ``(0, y)`` when feasible and ``(1, phi)`` otherwise."""
    if feasible:
        return (0, float(y))
    return (1, float(phi))


def _validated_objective(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"last evaluation record 'y' must be numeric, got {raw!r}"
        ) from exc
    if math.isnan(value) or (math.isinf(value) and value < 0.0):
        raise ValueError("last evaluation record 'y' must not be NaN or -inf")
    return value


def _finite_number(raw: Any, *, field: str) -> float:
    try:
        if isinstance(raw, bool):
            raise TypeError
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite, got {raw!r}")
    return value


def _finite_nonnegative(raw: Any, *, field: str) -> float:
    value = _finite_number(raw, field=field)
    if value < 0.0:
        raise ValueError(f"{field} must be non-negative, got {raw!r}")
    return value


def validate_deb_auxiliary_record(
    record: Mapping[str, Any],
    *,
    active_constraint_names: Sequence[str],
    invalid_utilization: float,
) -> DebAuxiliaryChannels:
    """
    Validate one auxiliary evaluation record and return its Deb channels.

    ``phi`` on a successful analysis is ``math.fsum`` of the violations in
    ``active_constraint_names`` order. A failed analysis yields ``phi = inf``.
    """
    required = {
        "y",
        "violations_",
        "constraint_values",
        "analysis_ok",
        "feasible",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(
            "last evaluation record is missing required key(s): "
            + ", ".join(missing)
        )
    analysis_ok = record["analysis_ok"]
    feasible = record["feasible"]
    if not isinstance(analysis_ok, bool):
        raise ValueError("last evaluation record 'analysis_ok' must be bool")
    if not isinstance(feasible, bool):
        raise ValueError("last evaluation record 'feasible' must be bool")
    y = _validated_objective(record["y"])
    sentinel_constraints: tuple[str, ...] = ()
    if analysis_ok:
        if not math.isfinite(y):
            raise ValueError(
                "analysis-valid last evaluation record 'y' must be finite"
            )
        violations = record["violations_"]
        if not isinstance(violations, Mapping):
            raise ValueError(
                "last evaluation record 'violations_' must be a mapping"
            )
        expected = set(active_constraint_names)
        actual = set(violations)
        if actual != expected:
            raise ValueError(
                "last evaluation record 'violations_' must contain exactly "
                f"the active constraints; expected {sorted(expected)!r}, "
                f"got {sorted(actual)!r}"
            )
        ordered_violations = [
            _finite_nonnegative(
                violations[name],
                field=f"violations_[{name!r}]",
            )
            for name in active_constraint_names
        ]
        phi = math.fsum(ordered_violations)
        if not math.isfinite(phi) or phi < 0.0:
            raise ValueError("aggregate constraint violation must be finite")
        if feasible != (phi == 0.0):
            raise ValueError(
                "last evaluation record 'feasible' must equal (phi == 0)"
            )
        constraint_values = record["constraint_values"]
        if not isinstance(constraint_values, Mapping):
            raise ValueError(
                "last evaluation record 'constraint_values' must be a mapping"
            )
        missing_values = [
            name
            for name in active_constraint_names
            if name not in constraint_values
        ]
        if missing_values:
            raise ValueError(
                "last evaluation record 'constraint_values' is missing "
                f"active constraints {missing_values!r}"
            )
        values = {
            name: _finite_number(
                constraint_values[name],
                field=f"constraint_values[{name!r}]",
            )
            for name in active_constraint_names
        }
        sentinel_constraints = tuple(
            name
            for name in active_constraint_names
            if values[name] == invalid_utilization
        )
    else:
        if feasible:
            raise ValueError(
                "failed analysis must have last evaluation record "
                "'feasible' equal to false"
            )
        phi = math.inf
    return DebAuxiliaryChannels(
        y=y,
        phi=phi,
        feasible=feasible,
        analysis_ok=analysis_ok,
        sentinel_constraints=sentinel_constraints,
    )
