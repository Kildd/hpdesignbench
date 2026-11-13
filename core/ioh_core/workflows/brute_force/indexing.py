"""
Mixed-radix linear indexing over :class:`IndexSpace`.

Uses ``itertools.product`` order: first variable outermost, last fastest.
"""
from __future__ import annotations

from typing import Iterator, Sequence

from core.ioh_core.workflows.brute_force.models import IndexSpace, IndexSpaceError


def linear_to_indices(linear: int, space: IndexSpace) -> tuple[int, ...]:
    """
    Map a linear address in ``[0, cardinality)`` to catalogue indices.

    Parameters
    ----------
    linear : int
        Zero-based linear candidate address.
    space : IndexSpace
        Target index space.

    Returns
    -------
    tuple of int
        Catalogue indices, one per variable, within inclusive bounds.

    Raises
    ------
    IndexSpaceError
        If ``linear`` is outside ``[0, space.cardinality)``.
    """
    if not isinstance(linear, int) or isinstance(linear, bool):
        raise IndexSpaceError(f"linear must be an int, got {linear!r}")
    if linear < 0 or linear >= space.cardinality:
        raise IndexSpaceError(
            f"linear {linear} out of range for cardinality {space.cardinality}"
        )

    remaining = linear
    indices: list[int] = []
    for lower, stride in zip(space.lower_bounds, space.strides):
        offset = remaining // stride
        remaining = remaining % stride
        indices.append(lower + offset)
    return tuple(indices)


def indices_to_linear(indices: Sequence[int], space: IndexSpace) -> int:
    """
    Map catalogue indices to a linear address in ``[0, cardinality)``.

    Parameters
    ----------
    indices : sequence of int
        Catalogue indices aligned with ``space.var_names``.
    space : IndexSpace
        Target index space.

    Returns
    -------
    int
        Zero-based linear candidate address.

    Raises
    ------
    IndexSpaceError
        If the length mismatches or any index is outside its inclusive bounds.
    """
    if len(indices) != len(space.var_names):
        raise IndexSpaceError(
            f"expected {len(space.var_names)} indices, got {len(indices)}"
        )

    linear = 0
    for index, (value, lower, upper, stride) in enumerate(
        zip(indices, space.lower_bounds, space.upper_bounds, space.strides)
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise IndexSpaceError(
                f"indices[{index}] must be an int, got {value!r}"
            )
        if value < lower or value > upper:
            raise IndexSpaceError(
                f"indices[{index}]={value} out of bounds [{lower}, {upper}] "
                f"for {space.var_names[index]!r}"
            )
        linear += (value - lower) * stride
    return linear


def iter_range(
    start: int,
    stop: int,
    space: IndexSpace,
) -> Iterator[tuple[int, ...]]:
    """
    Yield catalogue index tuples for linear addresses in ``[start, stop)``.

    Parameters
    ----------
    start : int
        Inclusive linear start (``0 <= start <= stop``).
    stop : int
        Exclusive linear stop (``stop <= space.cardinality``).
    space : IndexSpace
        Target index space.

    Yields
    ------
    tuple of int
        Catalogue index vector for each linear address in the half-open range.

    Raises
    ------
    IndexSpaceError
        If ``start``/``stop`` are not a valid half-open subrange of the space.
    """
    if not isinstance(start, int) or isinstance(start, bool):
        raise IndexSpaceError(f"start must be an int, got {start!r}")
    if not isinstance(stop, int) or isinstance(stop, bool):
        raise IndexSpaceError(f"stop must be an int, got {stop!r}")
    if start < 0 or stop < start or stop > space.cardinality:
        raise IndexSpaceError(
            f"invalid range [{start}, {stop}) for cardinality {space.cardinality}"
        )
    for linear in range(start, stop):
        yield linear_to_indices(linear, space)
