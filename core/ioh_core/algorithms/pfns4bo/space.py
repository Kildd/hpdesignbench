"""
Integer catalogue helpers for PFNs4BO unit-cube adapters.
"""
from __future__ import annotations

from random import Random
from typing import Sequence


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


def encode_integers_to_unit_cube(
    candidates: Sequence[Sequence[int]],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> list[list[float]]:
    """
    Map inclusive integer catalogue vectors onto ``[0, 1]^d``.

    Parameters
    ----------
    candidates : sequence of int sequences
        Integer points to encode.
    lower_bounds : sequence of int
        Inclusive lower bounds.
    upper_bounds : sequence of int
        Inclusive upper bounds.

    Returns
    -------
    list of list of float
        Unit-cube rows aligned with ``candidates``.
    """
    encoded: list[list[float]] = []
    for candidate in candidates:
        row: list[float] = []
        for value, lo, hi in zip(candidate, lower_bounds, upper_bounds):
            lo_i = int(lo)
            hi_i = int(hi)
            if hi_i == lo_i:
                row.append(0.0)
            else:
                row.append((int(value) - lo_i) / float(hi_i - lo_i))
        encoded.append(row)
    return encoded


def sample_integer_candidates(
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    *,
    n_samples: int,
    rng: Random,
) -> list[list[int]]:
    """
    Draw ``n_samples`` inclusive integer vectors from ``rng``.

    Parameters
    ----------
    lower_bounds : sequence of int
        Inclusive lower bounds.
    upper_bounds : sequence of int
        Inclusive upper bounds.
    n_samples : int
        Number of candidates.
    rng : Random
        Seeded RNG.

    Returns
    -------
    list of list of int
        Sampled integer candidates (with replacement).
    """
    return [
        [rng.randint(int(lo), int(hi)) for lo, hi in zip(lower_bounds, upper_bounds)]
        for _ in range(int(n_samples))
    ]
