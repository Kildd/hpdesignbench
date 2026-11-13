"""
Integer catalogue-index design spaces for SMT EGO adapters.
"""
from __future__ import annotations

from typing import Any, Sequence


def build_integer_design_space(
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    *,
    seed: int,
) -> Any:
    """
    Build an SMT ``DesignSpace`` for inclusive catalogue indices.

    Parameters
    ----------
    lower_bounds : sequence of int
        Inclusive lower catalogue index per dimension.
    upper_bounds : sequence of int
        Inclusive upper catalogue index per dimension.
    seed : int
        Deterministic seed stored on the design space.

    Returns
    -------
    smt.design_space.DesignSpace
        Integer design space with one ``IntegerVariable`` per dimension.

    Raises
    ------
    ValueError
        If bound lengths disagree, any lower exceeds its upper, or any
        dimension is degenerate (``lb == ub``). SMT requires
        ``upper > lower`` for integer variables and does not accept
        single-level ordinal/categorical variables.
    """
    from smt.design_space import DesignSpace, IntegerVariable

    dimension = len(lower_bounds)
    if len(upper_bounds) != dimension:
        raise ValueError(
            f"bound lengths disagree: lb={dimension} ub={len(upper_bounds)}"
        )
    variables: list[Any] = []
    for index, (lo, hi) in enumerate(zip(lower_bounds, upper_bounds)):
        lo_i = int(lo)
        hi_i = int(hi)
        if lo_i > hi_i:
            raise ValueError(
                f"bound[{index}] lower {lo_i} exceeds upper {hi_i}"
            )
        if lo_i == hi_i:
            raise ValueError(
                f"bound[{index}] is degenerate ({lo_i}=={hi_i}); "
                "SMT_EGO requires upper > lower on every dimension"
            )
        variables.append(IntegerVariable(lo_i, hi_i))
    return DesignSpace(variables, seed=int(seed))


def default_n_doe(max_evaluations: int) -> int:
    """
    Default initial DOE size for a given hard IOH budget.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.

    Returns
    -------
    int
        ``min(10, max(1, max_evaluations // 2))``.
    """
    return min(10, max(1, int(max_evaluations) // 2))
