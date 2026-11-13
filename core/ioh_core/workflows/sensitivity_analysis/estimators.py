"""
SALib Sobol index estimation for packed Saltelli responses.

Index estimation and optional bootstrap confidence half-widths are owned by
``SALib.analyze.sobol``. This module only packs ``Y`` into SALib order and
applies HpDesignBench fail-closed rules for degenerate responses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.ioh_core.workflows.sensitivity_analysis.deps import import_salib
from core.ioh_core.workflows.sensitivity_analysis.design import make_unit_problem


@dataclass(frozen=True)
class SobolIndexEstimate:
    """First-order and total-effect estimates from SALib."""

    first_order: NDArray[np.floating]
    total_effect: NDArray[np.floating]
    first_order_conf_half_width: NDArray[np.floating] | None
    total_effect_conf_half_width: NDArray[np.floating] | None
    salib_version: str


def pack_saltelli_responses(
    y_a: ArrayLike,
    y_b: ArrayLike,
    y_ab: ArrayLike,
) -> NDArray[np.floating]:
    """
    Pack base and cross responses into SALib Saltelli order.

    Within each radial block the order is ``A``, ``AB_0`` … ``AB_{k-1}``,
    ``B``.

    Parameters
    ----------
    y_a:
        Responses for matrix ``A``, shape ``(N,)``.
    y_b:
        Responses for matrix ``B``, shape ``(N,)``.
    y_ab:
        Responses for cross matrices, shape ``(N, k)``.

    Returns
    -------
    ndarray
        Length ``N * (k + 2)`` response vector.
    """
    a_values = np.asarray(y_a, dtype=float).reshape(-1)
    b_values = np.asarray(y_b, dtype=float).reshape(-1)
    ab_values = np.asarray(y_ab, dtype=float)
    if ab_values.ndim != 2:
        raise ValueError(
            f"y_ab must be 2-D with shape (N, k); got {ab_values.shape}."
        )
    sample_size, factor_count = ab_values.shape
    if a_values.shape != (sample_size,) or b_values.shape != (sample_size,):
        raise ValueError(
            "y_a and y_b must each have shape (N,) matching y_ab rows; "
            f"got y_a={a_values.shape}, y_b={b_values.shape}, "
            f"y_ab={ab_values.shape}."
        )
    if factor_count < 1:
        raise ValueError("y_ab must have at least one factor column.")

    packed = np.empty((sample_size, factor_count + 2), dtype=float)
    packed[:, 0] = a_values
    packed[:, 1 : factor_count + 1] = ab_values
    packed[:, factor_count + 1] = b_values
    return packed.reshape(-1)


class ConstantResponseError(ValueError):
    """Raised when a packed Saltelli response has zero range."""


def _validate_finite_nonconstant(y_packed: NDArray[np.floating]) -> None:
    if y_packed.size < 1:
        raise ValueError("Packed response vector is empty.")
    if np.any(~np.isfinite(y_packed)):
        raise ValueError(
            "Selected responses must be finite for every Saltelli row."
        )
    if float(np.ptp(y_packed)) == 0.0:
        raise ConstantResponseError(
            "Selected response has zero range; Sobol indices cannot be "
            "published for a constant output."
        )


def estimate_sobol_indices(
    y_a: ArrayLike,
    y_b: ArrayLike,
    y_ab: ArrayLike,
    *,
    compute_bootstrap_ci: bool,
    num_resamples: int,
    conf_level: float,
    seed: int,
) -> SobolIndexEstimate:
    """
    Pack ``Y`` and call ``SALib.analyze.sobol`` (``calc_second_order=False``).

    Always uses SALib for point estimates. When ``compute_bootstrap_ci`` is
    false, confidence half-width fields are set to ``None`` even though SALib
    may still compute internal bootstrap draws.

    Parameters
    ----------
    y_a, y_b, y_ab:
        Response arrays with shapes ``(N,)``, ``(N,)``, and ``(N, k)``.
    compute_bootstrap_ci:
        When true, publish SALib ``S1_conf`` / ``ST_conf`` half-widths.
    num_resamples:
        SALib ``num_resamples`` (must be ``>= 2`` when CI is enabled).
    conf_level:
        SALib confidence level in ``(0, 1)``.
    seed:
        SALib bootstrap seed.

    Returns
    -------
    SobolIndexEstimate
        First-order and total-effect arrays, optional half-widths, and the
        SALib version string.
    """
    if not isinstance(compute_bootstrap_ci, bool):
        raise ValueError("compute_bootstrap_ci must be a bool.")

    ab_values = np.asarray(y_ab, dtype=float)
    if ab_values.ndim != 2:
        raise ValueError(
            f"y_ab must be 2-D with shape (N, k); got {ab_values.shape}."
        )
    factor_count = int(ab_values.shape[1])

    y_packed = pack_saltelli_responses(y_a, y_b, y_ab)
    _validate_finite_nonconstant(y_packed)

    if not isinstance(num_resamples, (int, np.integer)) or isinstance(
        num_resamples, bool
    ):
        raise ValueError("num_resamples must be an integer.")
    resamples = int(num_resamples)
    if compute_bootstrap_ci and resamples < 2:
        raise ValueError(
            "num_resamples must be at least 2 when bootstrap CIs are enabled."
        )
    if resamples < 2:
        # SALib still bootstraps internally; keep a legal floor when CI is off.
        resamples = 2

    if not isinstance(conf_level, (int, float, np.floating)) or isinstance(
        conf_level, bool
    ):
        raise ValueError("conf_level must be a float.")
    confidence = float(conf_level)
    if not np.isfinite(confidence) or confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("conf_level must be strictly between 0 and 1.")

    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    seed_value = int(seed)

    salib = import_salib()
    from SALib.analyze import sobol as salib_sobol_analyze

    result = salib_sobol_analyze.analyze(
        make_unit_problem(factor_count),
        y_packed,
        calc_second_order=False,
        num_resamples=resamples,
        conf_level=confidence,
        print_to_console=False,
        parallel=False,
        seed=seed_value,
    )

    first_order = np.asarray(result["S1"], dtype=float).reshape(-1)
    total_effect = np.asarray(result["ST"], dtype=float).reshape(-1)
    if first_order.shape != (factor_count,) or total_effect.shape != (
        factor_count,
    ):
        raise ValueError(
            "SALib returned unexpected S1/ST shapes "
            f"{first_order.shape}, {total_effect.shape}; "
            f"expected ({factor_count},)."
        )
    if np.any(~np.isfinite(first_order)) or np.any(~np.isfinite(total_effect)):
        raise ValueError(
            "SALib returned non-finite Sobol index estimates."
        )

    first_conf: NDArray[np.floating] | None = None
    total_conf: NDArray[np.floating] | None = None
    if compute_bootstrap_ci:
        first_conf = np.asarray(result["S1_conf"], dtype=float).reshape(-1)
        total_conf = np.asarray(result["ST_conf"], dtype=float).reshape(-1)
        if first_conf.shape != (factor_count,) or total_conf.shape != (
            factor_count,
        ):
            raise ValueError(
                "SALib returned unexpected S1_conf/ST_conf shapes "
                f"{first_conf.shape}, {total_conf.shape}."
            )
        if np.any(~np.isfinite(first_conf)) or np.any(~np.isfinite(total_conf)):
            raise ValueError(
                "SALib returned non-finite bootstrap confidence half-widths."
            )

    return SobolIndexEstimate(
        first_order=first_order,
        total_effect=total_effect,
        first_order_conf_half_width=first_conf,
        total_effect_conf_half_width=total_conf,
        salib_version=str(getattr(salib, "__version__", "unknown")),
    )
