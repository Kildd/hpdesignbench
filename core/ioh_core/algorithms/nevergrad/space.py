"""
Discrete catalogue-index parametrizations for Nevergrad adapters.
"""
from __future__ import annotations

from typing import Any, Sequence


def build_transition_choice_tuple(
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> Any:
    """
    Build a Nevergrad ``Tuple`` of ``TransitionChoice`` catalogue spaces.

    Parameters
    ----------
    lower_bounds : sequence of int
        Inclusive lower catalogue index per dimension.
    upper_bounds : sequence of int
        Inclusive upper catalogue index per dimension.

    Returns
    -------
    nevergrad.p.Tuple
        Parametrization whose ``value`` is a tuple of integer indices.

    Raises
    ------
    ValueError
        If bound lengths disagree or any lower exceeds its upper.
    """
    import nevergrad as ng

    dimension = len(lower_bounds)
    if len(upper_bounds) != dimension:
        raise ValueError(
            f"bound lengths disagree: lb={dimension} ub={len(upper_bounds)}"
        )
    choices: list[Any] = []
    for index, (lo, hi) in enumerate(zip(lower_bounds, upper_bounds)):
        lo_i = int(lo)
        hi_i = int(hi)
        if lo_i > hi_i:
            raise ValueError(
                f"bound[{index}] lower {lo_i} exceeds upper {hi_i}"
            )
        choices.append(
            ng.p.TransitionChoice(list(range(lo_i, hi_i + 1)))
        )
    return ng.p.Tuple(*choices)
