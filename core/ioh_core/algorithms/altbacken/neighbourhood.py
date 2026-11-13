"""
Integer catalogue-index neighbourhoods for Altbacken simulated annealing.
"""
from __future__ import annotations

from random import Random
from typing import Sequence


class CatalogueResetNeighbourhood:
    """
    Single-coordinate random reset within inclusive catalogue bounds.

    Parameters
    ----------
    lower_bounds : sequence of int
        Inclusive lower catalogue index per dimension.
    upper_bounds : sequence of int
        Inclusive upper catalogue index per dimension.
    rng : Random
        Seeded RNG shared with the annealer acceptance draws.
    """

    def __init__(
        self,
        lower_bounds: Sequence[int],
        upper_bounds: Sequence[int],
        rng: Random,
    ) -> None:
        if len(lower_bounds) != len(upper_bounds):
            raise ValueError(
                f"bound lengths disagree: lb={len(lower_bounds)} "
                f"ub={len(upper_bounds)}"
            )
        if len(lower_bounds) < 1:
            raise ValueError("dimension must be >= 1")
        self._lower_bounds = [int(v) for v in lower_bounds]
        self._upper_bounds = [int(v) for v in upper_bounds]
        for index, (lo, hi) in enumerate(
            zip(self._lower_bounds, self._upper_bounds)
        ):
            if lo > hi:
                raise ValueError(
                    f"bound[{index}] lower {lo} exceeds upper {hi}"
                )
        self._rng = rng

    def __call__(self, state: object) -> list[int]:
        """
        Return a Hamming-1 neighbour by resampling one coordinate.

        Parameters
        ----------
        state : AnnealingState-like
            Object exposing ``current_solution`` as an integer sequence.

        Returns
        -------
        list of int
            New candidate inside inclusive catalogue bounds.
        """
        current = getattr(state, "current_solution")
        candidate = [int(v) for v in current]
        if len(candidate) != len(self._lower_bounds):
            raise ValueError(
                f"solution dimension {len(candidate)} disagrees with "
                f"bounds dimension {len(self._lower_bounds)}"
            )
        index = self._rng.randrange(len(candidate))
        lo = self._lower_bounds[index]
        hi = self._upper_bounds[index]
        choices = [value for value in range(lo, hi + 1) if value != candidate[index]]
        if choices:
            candidate[index] = self._rng.choice(choices)
        return candidate


def sample_integer_vector(
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    rng: Random,
) -> list[int]:
    """
    Draw a seeded random integer vector inside inclusive bounds.

    Parameters
    ----------
    lower_bounds : sequence of int
        Inclusive lower catalogue index per dimension.
    upper_bounds : sequence of int
        Inclusive upper catalogue index per dimension.
    rng : Random
        Seeded RNG.

    Returns
    -------
    list of int
        Random feasible catalogue-index vector.
    """
    if len(lower_bounds) != len(upper_bounds):
        raise ValueError(
            f"bound lengths disagree: lb={len(lower_bounds)} "
            f"ub={len(upper_bounds)}"
        )
    return [
        rng.randint(int(lo), int(hi))
        for lo, hi in zip(lower_bounds, upper_bounds)
    ]
