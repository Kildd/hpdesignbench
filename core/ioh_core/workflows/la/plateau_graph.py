"""Shared exhaustive plateau DAG for catalogue local-search analyses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.la.models import ElaError


@dataclass(frozen=True)
class PlateauGraph:
    """Equal-value plateaus and their unique-best improving destinations."""

    plateau_of_linear: NDArray[np.int32]
    plateau_size: NDArray[np.int64]
    plateau_min_linear: NDArray[np.int64]
    plateau_value: NDArray[np.float64]
    best_exit_value: NDArray[np.float64]
    destination_offsets: NDArray[np.int64]
    destination_ids: NDArray[np.int32]
    node_plateaus: NDArray[np.int32]
    plateau_to_node: NDArray[np.int32]

    @property
    def n_plateaus(self) -> int:
        return int(self.plateau_size.size)

    @property
    def n_nodes(self) -> int:
        return int(self.node_plateaus.size)

    @property
    def plateau_y_p(self) -> NDArray[np.float64]:
        """Backward-compatible alias for y_p-only basin consumers."""
        return self.plateau_value

    @property
    def best_exit_y_p(self) -> NDArray[np.float64]:
        """Backward-compatible alias for y_p-only basin consumers."""
        return self.best_exit_value


def dense_preorder_ranks(keys: ArrayLike) -> NDArray[np.int64]:
    """
    Map each design to a dense rank in a declared total preorder.

    Smaller keys are better. A 1-d array uses IEEE equality. A 2-d array
    uses lexicographic row order (Fieldsend ``h`` fits here). Ranks are
    not a packed ``y_p`` column.
    """
    arr = np.asarray(keys)
    if arr.ndim == 1:
        _unique, inverse = np.unique(arr, return_inverse=True)
        return inverse.astype(np.int64, copy=False)
    if arr.ndim != 2:
        raise ElaError(f"preorder keys must be 1-d or 2-d, got shape {arr.shape}")
    if arr.shape[0] < 1:
        return np.zeros(0, dtype=np.int64)
    _unique, inverse = np.unique(arr, axis=0, return_inverse=True)
    return inverse.astype(np.int64, copy=False)


def nested_best_exit_support(
    graph: PlateauGraph,
) -> tuple[NDArray[np.int32], dict[int, tuple[int, ...]]]:
    """Dynamic program over strictly descending best-exit plateau edges."""
    n_plateaus = graph.n_plateaus
    single = np.full(n_plateaus, -1, dtype=np.int32)
    multiple: dict[int, tuple[int, ...]] = {}
    order = np.argsort(graph.plateau_value, kind="stable")
    for raw_pid in order:
        pid = int(raw_pid)
        node_id = int(graph.plateau_to_node[pid])
        if node_id >= 0:
            single[pid] = node_id
            continue
        start = int(graph.destination_offsets[pid])
        stop = int(graph.destination_offsets[pid + 1])
        if start == stop:
            raise ElaError(f"non-node plateau {pid} has no best exit")
        targets: set[int] = set()
        for destination in graph.destination_ids[start:stop]:
            dest = int(destination)
            multi = multiple.get(dest)
            if multi is not None:
                targets.update(multi)
            else:
                target = int(single[dest])
                if target < 0:
                    raise ElaError(
                        "plateau best-exit DAG is not strictly descending"
                    )
                targets.add(target)
        ordered = tuple(sorted(targets))
        if not ordered:
            raise ElaError(f"plateau {pid} has empty terminal support")
        if len(ordered) == 1:
            single[pid] = ordered[0]
        else:
            multiple[pid] = ordered
    return single, multiple


def build_plateau_graph(
    values: ArrayLike,
    space: IndexSpace,
) -> PlateauGraph:
    """Build the common equal-height plateau DAG used by basins and LON."""
    raw = np.asarray(values).reshape(-1)
    if np.issubdtype(raw.dtype, np.integer):
        values = raw.astype(np.float64, copy=False)
    else:
        values = np.asarray(raw, dtype=np.float64)
    if int(values.size) != int(space.cardinality):
        raise ElaError(
            f"landscape length ({values.size}) does not match cardinality "
            f"({space.cardinality})"
        )
    if not bool(np.all(np.isfinite(values))):
        raise ElaError("non-finite values in landscape source; reject whole source")

    plateau = equal_value_l1_components(values, space)
    n = int(values.size)
    n_plateaus = int(plateau.max()) + 1 if n else 0
    linear_ids = np.arange(n, dtype=np.int64)
    plateau_size = np.bincount(plateau, minlength=n_plateaus).astype(np.int64)
    plateau_min_linear = np.full(n_plateaus, n, dtype=np.int64)
    np.minimum.at(plateau_min_linear, plateau, linear_ids)
    plateau_value = values[plateau_min_linear]
    best_y, offsets, destinations = best_exits(
        values, plateau, plateau_value, space
    )
    node_plateaus = np.flatnonzero(~np.isfinite(best_y)).astype(np.int32)
    if not node_plateaus.size:
        raise ElaError("best-improvement plateau graph found no LON nodes")
    plateau_to_node = np.full(n_plateaus, -1, dtype=np.int32)
    plateau_to_node[node_plateaus] = np.arange(
        node_plateaus.size, dtype=np.int32
    )

    for array in (
        plateau,
        plateau_size,
        plateau_min_linear,
        plateau_value,
        best_y,
        offsets,
        destinations,
        node_plateaus,
        plateau_to_node,
    ):
        array.setflags(write=False)
    return PlateauGraph(
        plateau_of_linear=plateau,
        plateau_size=plateau_size,
        plateau_min_linear=plateau_min_linear,
        plateau_value=plateau_value,
        best_exit_value=best_y,
        destination_offsets=offsets,
        destination_ids=destinations,
        node_plateaus=node_plateaus,
        plateau_to_node=plateau_to_node,
    )


def equal_value_l1_components(
    values: NDArray[np.float64],
    space: IndexSpace,
) -> NDArray[np.int32]:
    """Return stable plateau ids ordered by minimum catalogue linear id."""
    n = int(values.size)
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int8)

    def find(index: int) -> int:
        root = index
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[index]) != index:
            nxt = int(parent[index])
            parent[index] = root
            index = nxt
        return root

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
        elif rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_right] = root_left
            rank[root_left] += 1

    linear = np.arange(n, dtype=np.int64)
    for stride, size in zip(space.strides, space.sizes, strict=True):
        stride_i = int(stride)
        size_i = int(size)
        can = ((linear // stride_i) % size_i) + 1 < size_i
        lefts = linear[can]
        rights = lefts + stride_i
        match = values[lefts] == values[rights]
        for left, right in zip(
            lefts[match].tolist(), rights[match].tolist(), strict=True
        ):
            union(int(left), int(right))

    min_linear_by_root: dict[int, int] = {}
    roots = np.empty(n, dtype=np.int32)
    for index in range(n):
        root = find(index)
        roots[index] = root
        previous = min_linear_by_root.get(root)
        if previous is None or index < previous:
            min_linear_by_root[root] = index
    ordered = sorted(min_linear_by_root, key=min_linear_by_root.__getitem__)
    remap = np.full(n, -1, dtype=np.int32)
    remap[np.fromiter(ordered, dtype=np.int32, count=len(ordered))] = np.arange(
        len(ordered), dtype=np.int32
    )
    return remap[roots]


def equal_yp_l1_components(
    y_p: NDArray[np.float64],
    space: IndexSpace,
) -> NDArray[np.int32]:
    """Backward-compatible name for equal-value L1 components."""
    return equal_value_l1_components(y_p, space)


def best_exits(
    values: NDArray[np.float64],
    plateau: NDArray[np.int32],
    plateau_value: NDArray[np.float64],
    space: IndexSpace,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int32]]:
    """Return all plateau destinations at each plateau's best lower height."""
    n_plateaus = int(plateau_value.size)
    best_y = np.full(n_plateaus, np.inf, dtype=np.float64)
    linear = np.arange(int(values.size), dtype=np.int64)
    for src, _dst, height in improving_l1_edges(values, linear, space):
        np.minimum.at(best_y, plateau[src], height)

    pair_chunks: list[NDArray[np.int32]] = []
    for src, dst, height in improving_l1_edges(values, linear, space):
        keep = height == best_y[plateau[src]]
        if not bool(np.any(keep)):
            continue
        chunk = np.unique(
            np.column_stack((plateau[src[keep]], plateau[dst[keep]])),
            axis=0,
        )
        if chunk.size:
            pair_chunks.append(chunk.astype(np.int32, copy=False))
    pairs = (
        np.unique(np.vstack(pair_chunks), axis=0)
        if pair_chunks
        else np.empty((0, 2), dtype=np.int32)
    )
    counts = np.bincount(pairs[:, 0], minlength=n_plateaus)
    offsets = np.empty(n_plateaus + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return best_y, offsets, pairs[:, 1]


def improving_l1_edges(
    values: NDArray[np.float64],
    linear: NDArray[np.int64],
    space: IndexSpace,
) -> Iterator[
    tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]
]:
    """Yield directed improving design edges one catalogue dimension at a time."""
    for stride, size in zip(space.strides, space.sizes, strict=True):
        stride_i = int(stride)
        size_i = int(size)
        can = ((linear // stride_i) % size_i) + 1 < size_i
        lefts = linear[can]
        rights = lefts + stride_i
        y_left = values[lefts]
        y_right = values[rights]
        to_right = y_right < y_left
        to_left = y_left < y_right
        if bool(np.any(to_right)):
            yield lefts[to_right], rights[to_right], y_right[to_right]
        if bool(np.any(to_left)):
            yield rights[to_left], lefts[to_left], y_left[to_left]
