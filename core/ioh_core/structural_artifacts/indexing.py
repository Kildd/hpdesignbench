"""
Structural mixed-radix indexing for precomputed artifact libraries.

Finish catalogue variables are the last two dimensions of an
:class:`~core.ioh_core.workflows.brute_force.models.IndexSpace`. All earlier
dimensions define the structural subspace. When that finish-last ordering
holds, consecutive finish combinations share one ``structural_id`` and:

``structural_id == global_index // finish_group_size(space)``.

Cardinality and group size are always derived from the space; Problem 1
values (24 / 2,223,936) are never hard-coded here.
"""
from __future__ import annotations

from typing import Sequence

from core.ioh_core.workflows.brute_force.indexing import (
    indices_to_linear,
    linear_to_indices,
)
from core.ioh_core.workflows.brute_force.models import IndexSpace, IndexSpaceError


class StructuralIndexError(ValueError):
    """
    Raised when structural indexing cannot proceed safely.
    """


def _require_index_space(space: IndexSpace) -> IndexSpace:
    if not isinstance(space, IndexSpace):
        raise StructuralIndexError(
            f"space must be an IndexSpace, got {type(space)!r}"
        )
    return space


def _require_finish_pair(space: IndexSpace) -> None:
    if len(space.var_names) < 2:
        raise StructuralIndexError(
            "structural indexing requires at least two catalogue variables "
            f"(finish pair last), got {len(space.var_names)}"
        )


def finish_variable_names(space: IndexSpace) -> tuple[str, ...]:
    """
    Return the last two catalogue variable names (floor-finish group).

    Parameters
    ----------
    space : IndexSpace
        Full candidate index space.

    Returns
    -------
    tuple of str
        Finish variable names in product order.

    Raises
    ------
    StructuralIndexError
        If fewer than two variables are present.
    """
    space = _require_index_space(space)
    _require_finish_pair(space)
    return (str(space.var_names[-2]), str(space.var_names[-1]))


def structural_variable_names(space: IndexSpace) -> tuple[str, ...]:
    """
    Return catalogue variable names excluding the finish pair.

    Parameters
    ----------
    space : IndexSpace
        Full candidate index space.

    Returns
    -------
    tuple of str
        Structural variable names in product order.

    Raises
    ------
    StructuralIndexError
        If fewer than two variables are present, or no structural variables
        remain (exactly two dimensions).
    """
    space = _require_index_space(space)
    _require_finish_pair(space)
    names = tuple(str(name) for name in space.var_names[:-2])
    if not names:
        raise StructuralIndexError(
            "structural subspace is empty: catalogue has only finish variables"
        )
    return names


def finish_group_size(space: IndexSpace) -> int:
    """
    Return the product of the last two catalogue sizes.

    Parameters
    ----------
    space : IndexSpace
        Full candidate index space.

    Returns
    -------
    int
        Finish-group cardinality (``24`` for Problem 1).

    Raises
    ------
    StructuralIndexError
        If fewer than two variables are present.
    """
    space = _require_index_space(space)
    _require_finish_pair(space)
    return int(space.sizes[-2] * space.sizes[-1])


def structural_space(space: IndexSpace) -> IndexSpace:
    """
    Return an :class:`IndexSpace` over structural dimensions only.

    Parameters
    ----------
    space : IndexSpace
        Full candidate index space.

    Returns
    -------
    IndexSpace
        Mixed-radix space for the structural subspace.

    Raises
    ------
    StructuralIndexError
        If the structural subspace cannot be formed.
    IndexSpaceError
        If derived bounds are inconsistent (propagated from ``IndexSpace``).
    """
    space = _require_index_space(space)
    names = structural_variable_names(space)
    try:
        return IndexSpace.from_bounds(
            names,
            space.lower_bounds[: len(names)],
            space.upper_bounds[: len(names)],
        )
    except IndexSpaceError as exc:
        raise StructuralIndexError(
            f"cannot build structural IndexSpace: {exc}"
        ) from exc


def structural_cardinality(space: IndexSpace) -> int:
    """
    Return the number of distinct structural states.

    Parameters
    ----------
    space : IndexSpace
        Full candidate index space.

    Returns
    -------
    int
        ``product(sizes[:-2])`` (``2_223_936`` for Problem 1).
    """
    return int(structural_space(space).cardinality)


def candidate_indices_to_structural_id(
    indices: Sequence[int],
    space: IndexSpace,
) -> int:
    """
    Map a full catalogue index vector to its structural ID.

    Parameters
    ----------
    indices : sequence of int
        Catalogue indices aligned with ``space.var_names``.
    space : IndexSpace
        Full candidate index space.

    Returns
    -------
    int
        Zero-based structural ID in ``[0, structural_cardinality)``.

    Raises
    ------
    StructuralIndexError
        If the vector length or bounds are invalid.
    """
    space = _require_index_space(space)
    struct = structural_space(space)
    if len(indices) != len(space.var_names):
        raise StructuralIndexError(
            f"expected {len(space.var_names)} indices, got {len(indices)}"
        )
    # Validate the full vector (including finish bounds) via the full space.
    try:
        indices_to_linear(indices, space)
    except IndexSpaceError as exc:
        raise StructuralIndexError(str(exc)) from exc
    structural_indices = tuple(int(value) for value in indices[: len(struct.var_names)])
    try:
        return indices_to_linear(structural_indices, struct)
    except IndexSpaceError as exc:
        raise StructuralIndexError(str(exc)) from exc


def structural_id_to_structural_indices(
    structural_id: int,
    space: IndexSpace,
) -> tuple[int, ...]:
    """
    Map a structural ID to structural catalogue indices.

    Parameters
    ----------
    structural_id : int
        Zero-based structural ID.
    space : IndexSpace
        Full candidate index space.

    Returns
    -------
    tuple of int
        Catalogue indices for structural variables only.

    Raises
    ------
    StructuralIndexError
        If ``structural_id`` is outside the structural subspace.
    """
    space = _require_index_space(space)
    struct = structural_space(space)
    if not isinstance(structural_id, int) or isinstance(structural_id, bool):
        raise StructuralIndexError(
            f"structural_id must be an int, got {structural_id!r}"
        )
    try:
        return linear_to_indices(structural_id, struct)
    except IndexSpaceError as exc:
        raise StructuralIndexError(str(exc)) from exc


def global_index_to_structural_id(global_index: int, space: IndexSpace) -> int:
    """
    Map a full-catalogue linear address to its structural ID.

    Under finish-last mixed-radix ordering this is
    ``global_index // finish_group_size(space)``.

    Parameters
    ----------
    global_index : int
        Linear address in ``[0, space.cardinality)``.
    space : IndexSpace
        Full candidate index space.

    Returns
    -------
    int
        Structural ID.

    Raises
    ------
    StructuralIndexError
        If ``global_index`` is outside the full catalogue.
    """
    space = _require_index_space(space)
    group = finish_group_size(space)
    if not isinstance(global_index, int) or isinstance(global_index, bool):
        raise StructuralIndexError(
            f"global_index must be an int, got {global_index!r}"
        )
    if global_index < 0 or global_index >= space.cardinality:
        raise StructuralIndexError(
            f"global_index {global_index} out of range for cardinality "
            f"{space.cardinality}"
        )
    if space.cardinality % group != 0:
        raise StructuralIndexError(
            "full cardinality is not divisible by finish_group_size "
            f"({space.cardinality} % {group} != 0); finish-last ordering "
            "invariant broken"
        )
    return global_index // group
