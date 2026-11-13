"""
Frozen per-run / per-problem analysis configuration.

Settings that are fixed for a problem or worker process (design code,
invalid-utilization sentinel, slab type) live here and are applied once at
initialization. They must not be mutated on every candidate evaluation.

``invalid_utilization`` is not hardcoded here: it comes from the problem
parameter ``opt_invalid_utilization`` (``parameter_defaults.csv`` /
``problem_list.csv``) when building an ``AnalysisRunConfig``.
"""
from __future__ import annotations

from dataclasses import dataclass

from structuralcodes import set_design_code


def require_invalid_utilization(invalid_utilization: float | None) -> float:
    """
    Return a finite invalid-utilization sentinel.

    Parameters
    ----------
    invalid_utilization : float or None
        Explicit sentinel from ``AnalysisRunConfig`` /
        ``opt_invalid_utilization``. ``None`` is rejected so callers cannot
        silently fall back to a hardcoded value.

    Raises
    ------
    ValueError
        If ``invalid_utilization`` is ``None``.
    """
    if invalid_utilization is None:
        raise ValueError(
            "invalid_utilization is required; pass AnalysisRunConfig.invalid_utilization "
            "from opt_invalid_utilization (parameter_defaults.csv / problem_list.csv)."
        )
    return float(invalid_utilization)


@dataclass(frozen=True)
class AnalysisRunConfig:
    """Immutable analysis settings for one problem or worker run."""

    invalid_utilization: float
    design_code: str = "ec2_2004"
    slab_type: str = "hp_slab"


# Last design code applied in this process (avoid redundant set_design_code calls).
_applied_design_code: str | None = None


def analysis_run_config_from_params(
    params: dict,
    *,
    slab_type: str = "hp_slab",
    design_code: str = "ec2_2004",
) -> AnalysisRunConfig:
    """
    Build a run-config from fixed/decoded parameters.

    Requires ``opt_invalid_utilization`` in ``params`` (CSV-defined).
    """
    if "opt_invalid_utilization" not in params:
        raise KeyError(
            "opt_invalid_utilization missing from params; define it in "
            "parameter_defaults.csv / problem_list.csv and include it among "
            "fixed or decoded parameters."
        )
    return AnalysisRunConfig(
        design_code=str(design_code),
        invalid_utilization=float(params["opt_invalid_utilization"]),
        slab_type=str(slab_type),
    )


def apply_analysis_run_config(config: AnalysisRunConfig) -> None:
    """
    Apply process-level effects of ``config`` at most once per distinct value.

    Currently sets the structuralcodes design code when it differs from the
    last applied value. Does not invent or mutate an invalid-utilization default.
    """
    global _applied_design_code
    if _applied_design_code != config.design_code:
        set_design_code(config.design_code)
        _applied_design_code = config.design_code


def reset_applied_analysis_run_config_for_tests() -> None:
    """Test helper: clear the process-local applied-design-code cache."""
    global _applied_design_code
    _applied_design_code = None
