"""
L1=1 catalogue neighbourhood on raw index space.

In-bounds axis steps only: no wrap, no diagonal, no jump greater than 1.
Does not consult a feasible mask. Shared by GBLT and catalogue
local-search basins.

Importing this module must not load DuckDB, pflacco, NetworkX, or SALib.
"""
from __future__ import annotations

from typing import Sequence

from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.la.models import ElaError


def catalogue_l1_neighbours(
    x_index: Sequence[int], space: IndexSpace
) -> tuple[tuple[int, ...], ...]:
    """
    Return in-bounds L1=1 neighbours of ``x_index``.

    For each sealed variable, try ``−1`` then ``+1``. Other coordinates
    stay unchanged.
    """
    try:
        point = tuple(int(value) for value in x_index)
    except (TypeError, ValueError) as exc:
        raise ElaError(f"catalogue indices must be ints, got {x_index!r}") from exc
    n_dim = len(space.var_names)
    if len(point) != n_dim:
        raise ElaError(f"expected {n_dim} indices, got {len(point)}")
    neighbours: list[tuple[int, ...]] = []
    coords = list(point)
    for dim, (lower, upper) in enumerate(
        zip(space.lower_bounds, space.upper_bounds, strict=True)
    ):
        current = coords[dim]
        for step in (-1, 1):
            candidate = current + step
            if candidate < lower or candidate > upper:
                continue
            coords[dim] = candidate
            neighbours.append(tuple(coords))
        coords[dim] = current
    return tuple(neighbours)
