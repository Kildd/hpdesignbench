"""
Graph-based landscape topology (GBLT) on raw catalogue indices.

Paper name: graph-based landscape topology. Preferred CLI:
``hdb-la-analyze topology`` (``family=topology``).

Two sub-objects:

- **Feasible-set topology** — |F|, L1=1 components, sizes, isolation,
  ∂F, and optima counts on analysis-local ``F``. ``y*`` is a reference
  threshold (reported, not a primary feature).
- **Near-optimal-set topology** — ``G_τ = {x in F : y <= (1+tau) y*}``,
  prevalence, F-component coverage, boundary share, and L1=1 connected
  components of the subgraph induced by ``G_τ``.

Complete brute-force catalogues only. Analysis-local ``F`` from
:func:`constraint_violation_sql`; stored ``feasible`` is not read.
Neighbourhood is L1=1 on ``X_index`` (no wrap, no diagonal, no jump>1).
Does not build the graph of the complete design space.

Linear addresses use :class:`IndexSpace` / :func:`indices_to_linear`
(same mixed-radix order as brute-force ``global_index``). Components are
local union-find on feasible linear ids and L1=1 feasible neighbours.
Global optima are exact stored ``y == y*`` on analysis-local ``F``
(IEEE equality; not BF ``SCIENTIFIC_RTOL``). Boundary is an in-bounds
infeasible L1 neighbour, not a variable bound.

See ``CONTEXT.md`` for domain terms. Importing this module must not load
DuckDB, pflacco, NetworkX, or SALib.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from core.ioh_core.workflows.brute_force.indexing import (
    indices_to_linear,
    linear_to_indices,
)
from core.ioh_core.workflows.brute_force.models import IndexSpace, IndexSpaceError
from core.ioh_core.workflows.la.catalogue_l1 import catalogue_l1_neighbours
from core.ioh_core.workflows.la.deps import import_duckdb
from core.ioh_core.workflows.la.models import (
    ElaError,
    FeatureResult,
    LANDSCAPE_FEASIBILITY,
    LANDSCAPE_Y_RAW,
    RESPONSE_VIEW_NA,
    ROLE_DIAGNOSTIC,
    ROLE_PRIMARY,
    SOURCE_BRUTE_FORCE,
    STATUS_OK,
    STATUS_UNDEFINED,
    X_VIEW_INDEX,
    feature_result,
)
from core.ioh_core.workflows.la.preprocess import constraint_violation_sql
from core.ioh_core.workflows.la.signatures import analysis_signature

FAMILY = "topology"
LIBRARY = "hpdesignbench"
FEATURE_N_FEASIBLE = "topology.n_feasible"
FEATURE_N_COMPONENTS = "topology.n_components"
FEATURE_N_ISOLATED = "topology.n_isolated"
FEATURE_SIZE_MIN = "topology.size_min"
FEATURE_SIZE_MEDIAN = "topology.size_median"
FEATURE_SIZE_MAX = "topology.size_max"
FEATURE_LARGEST_SHARE = "topology.largest_share"
FEATURE_ISOLATED_FRACTION_OF_F = "topology.isolated_fraction_of_f"
FEATURE_ISOLATED_FRACTION_OF_COMPONENTS = (
    "topology.isolated_fraction_of_components"
)
FEATURE_Y_STAR = "topology.y_star"
FEATURE_N_OPTIMA = "topology.n_optima"
FEATURE_N_OPTIMA_COMPONENTS = "topology.n_optima_components"
FEATURE_N_BOUNDARY = "topology.n_boundary"
FEATURE_BOUNDARY_FRACTION_OF_F = "topology.boundary_fraction_of_f"
_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class FeasibleRegion:
    """
    Feasible designs as a cardinality-length mask on mixed-radix linear ids.

    Attributes
    ----------
    space :
        Sealed catalogue index space.
    mask :
        ``mask[linear]`` is true iff that design is analysis-local feasible.
        Length equals ``space.cardinality``. Read-only.
    n_feasible :
        ``int(mask.sum())``.
    n_input, n_unique :
        Observed catalogue counts after the completeness gate.
    constraint_names :
        Keep-list used to form ``F``.
    """

    space: IndexSpace
    mask: NDArray[np.bool_]
    n_feasible: int
    n_input: int
    n_unique: int
    constraint_names: tuple[str, ...]

    def linear_of(self, x_index: Sequence[int]) -> int:
        """Return the mixed-radix linear address of ``x_index``."""
        try:
            return indices_to_linear(_as_int_tuple(x_index), self.space)
        except IndexSpaceError as exc:
            raise ElaError(str(exc)) from exc

    def is_feasible(self, x_index: Sequence[int]) -> bool:
        """Return whether ``x_index`` is in the analysis-local feasible set."""
        return bool(self.mask[self.linear_of(x_index)])

    def catalogue_neighbours(
        self, x_index: Sequence[int]
    ) -> tuple[tuple[int, ...], ...]:
        """
        Return in-bounds L1=1 neighbours of ``x_index``.

        For each sealed variable, try ``−1`` then ``+1``. Other coordinates
        stay unchanged. Does not wrap. Does not consult ``mask``.
        """
        return catalogue_l1_neighbours(x_index, self.space)

    def feasible_neighbours(
        self, x_index: Sequence[int]
    ) -> tuple[tuple[int, ...], ...]:
        """Return L1=1 neighbours that are analysis-local feasible."""
        return tuple(
            neighbour
            for neighbour in self.catalogue_neighbours(x_index)
            if self.is_feasible(neighbour)
        )


@dataclass(frozen=True)
class FeasibleComponents:
    """
    Connected components of analysis-local ``F`` under L1=1.

    Attributes
    ----------
    region :
        Feasible mask this partition was computed from.
    labels :
        ``labels[linear]`` is the compact component id, or ``-1`` if
        infeasible. Length equals catalogue cardinality. Read-only.
        Ids are ``0 .. n_components-1`` ordered by min linear id.
    sizes :
        Component sizes in that id order. Read-only. This is every size.
    n_components, n_isolated :
        Number of components, and the number of size-1 components
        (equal to the number of isolated feasible points).
    size_min, size_median, size_max :
        Aggregates of ``sizes``. Median is NumPy ``median``. ``None``
        when ``F`` is empty.
    largest_share :
        ``max(sizes) / |F|``, or ``None`` when ``F`` is empty.
    isolated_fraction_of_f :
        ``n_isolated / |F|``, or ``None`` when ``F`` is empty.
    isolated_fraction_of_components :
        ``n_isolated / n_components``, or ``None`` when ``F`` is empty.
    """

    region: FeasibleRegion
    labels: NDArray[np.int32]
    sizes: NDArray[np.int64]
    n_components: int
    n_isolated: int
    size_min: int | None
    size_median: float | None
    size_max: int | None
    largest_share: float | None
    isolated_fraction_of_f: float | None
    isolated_fraction_of_components: float | None

    def component_of(self, x_index: Sequence[int]) -> int:
        """Return the compact component id of ``x_index``, or ``-1``."""
        return int(self.labels[self.region.linear_of(x_index)])


def ela_feasible_components(region: FeasibleRegion) -> FeasibleComponents:
    """
    Partition analysis-local ``F`` by L1=1 adjacency.

    Union-find runs only over feasible linear ids. Does not build the
    graph of the complete design space. Empty ``F`` yields
    ``n_components=0`` and ``None`` ratio fields.
    """
    space = region.space
    mask = region.mask
    cardinality = space.cardinality
    if mask.shape != (cardinality,):
        raise ElaError(
            "topology feasible mask length "
            f"({int(mask.size)}) does not match cardinality ({cardinality})"
        )
    n_feasible = int(np.count_nonzero(mask))
    if n_feasible != region.n_feasible:
        raise ElaError(
            "topology n_feasible "
            f"({region.n_feasible}) does not match mask sum ({n_feasible})"
        )
    try:
        labels = np.full(cardinality, -1, dtype=np.int32)
    except MemoryError as exc:
        raise ElaError(
            "topology union-find could not allocate a cardinality-length "
            "label array"
        ) from exc
    if n_feasible == 0:
        empty_sizes = np.zeros(0, dtype=np.int64)
        labels.setflags(write=False)
        empty_sizes.setflags(write=False)
        return FeasibleComponents(
            region=region,
            labels=labels,
            sizes=empty_sizes,
            n_components=0,
            n_isolated=0,
            size_min=None,
            size_median=None,
            size_max=None,
            largest_share=None,
            isolated_fraction_of_f=None,
            isolated_fraction_of_components=None,
        )
    feasible_linears = np.flatnonzero(mask)
    labels[feasible_linears] = np.arange(n_feasible, dtype=np.int32)
    parent = np.arange(n_feasible, dtype=np.int32)
    rank = np.zeros(n_feasible, dtype=np.int8)

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

    for dense_i in range(n_feasible):
        linear = int(feasible_linears[dense_i])
        point = linear_to_indices(linear, space)
        for neighbour in region.catalogue_neighbours(point):
            dense_j = int(labels[indices_to_linear(neighbour, space)])
            if dense_j >= 0:
                union(dense_i, dense_j)

    min_linear_by_root: dict[int, int] = {}
    member_root = np.empty(n_feasible, dtype=np.int32)
    for dense_i in range(n_feasible):
        root = find(dense_i)
        member_root[dense_i] = root
        linear = int(feasible_linears[dense_i])
        previous = min_linear_by_root.get(root)
        if previous is None or linear < previous:
            min_linear_by_root[root] = linear
    ordered_roots = sorted(min_linear_by_root, key=min_linear_by_root.__getitem__)
    root_to_id = {root: cid for cid, root in enumerate(ordered_roots)}
    n_components = len(ordered_roots)
    sizes = np.zeros(n_components, dtype=np.int64)
    for dense_i in range(n_feasible):
        cid = root_to_id[int(member_root[dense_i])]
        labels[int(feasible_linears[dense_i])] = cid
        sizes[cid] += 1
    n_isolated = int(np.count_nonzero(sizes == 1))
    size_min = int(sizes.min())
    size_max = int(sizes.max())
    labels.setflags(write=False)
    sizes.setflags(write=False)
    return FeasibleComponents(
        region=region,
        labels=labels,
        sizes=sizes,
        n_components=n_components,
        n_isolated=n_isolated,
        size_min=size_min,
        size_median=float(np.median(sizes)),
        size_max=size_max,
        largest_share=float(size_max) / float(n_feasible),
        isolated_fraction_of_f=float(n_isolated) / float(n_feasible),
        isolated_fraction_of_components=float(n_isolated) / float(n_components),
    )


@dataclass(frozen=True)
class InducedSubsetComponents:
    """
    L1=1 connected components on a feasible subset ``S ⊆ F``.

    Union-find runs only over ``S``; an edge requires both endpoints in
    ``S``. Empty ``S`` yields ``n_components=0`` and ``None`` aggregates.
    """

    n_points: int
    n_components: int
    n_isolated: int
    size_min: int | None
    size_median: float | None
    size_max: int | None
    largest_share: float | None


def _induced_l1_components(
    region: FeasibleRegion,
    subset: NDArray[np.bool_],
) -> InducedSubsetComponents:
    """
    Partition ``S`` by L1=1 adjacency on the catalogue index graph.

    ``subset`` must be a boolean mask of catalogue length with ``True``
    only on analysis-local ``F``.
    """
    space = region.space
    feasible = region.mask
    cardinality = space.cardinality
    if subset.shape != (cardinality,):
        raise ElaError(
            "topology subset mask length "
            f"({int(subset.size)}) does not match cardinality ({cardinality})"
        )
    if np.any(subset & ~feasible):
        raise ElaError("topology subset must lie within analysis-local F")
    n_points = int(np.count_nonzero(subset))
    if n_points == 0:
        return InducedSubsetComponents(
            n_points=0,
            n_components=0,
            n_isolated=0,
            size_min=None,
            size_median=None,
            size_max=None,
            largest_share=None,
        )
    subset_linears = np.flatnonzero(subset)
    labels = np.full(cardinality, -1, dtype=np.int32)
    labels[subset_linears] = np.arange(n_points, dtype=np.int32)
    parent = np.arange(n_points, dtype=np.int32)
    rank = np.zeros(n_points, dtype=np.int8)

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

    for dense_i in range(n_points):
        linear = int(subset_linears[dense_i])
        point = linear_to_indices(linear, space)
        for neighbour in region.catalogue_neighbours(point):
            neighbour_linear = indices_to_linear(neighbour, space)
            if subset[neighbour_linear]:
                dense_j = int(labels[neighbour_linear])
                union(dense_i, dense_j)

    min_linear_by_root: dict[int, int] = {}
    member_root = np.empty(n_points, dtype=np.int32)
    for dense_i in range(n_points):
        root = find(dense_i)
        member_root[dense_i] = root
        linear = int(subset_linears[dense_i])
        previous = min_linear_by_root.get(root)
        if previous is None or linear < previous:
            min_linear_by_root[root] = linear
    ordered_roots = sorted(min_linear_by_root, key=min_linear_by_root.__getitem__)
    root_to_id = {root: cid for cid, root in enumerate(ordered_roots)}
    n_components = len(ordered_roots)
    sizes = np.zeros(n_components, dtype=np.int64)
    for dense_i in range(n_points):
        cid = root_to_id[int(member_root[dense_i])]
        sizes[cid] += 1
    n_isolated = int(np.count_nonzero(sizes == 1))
    size_min = int(sizes.min())
    size_max = int(sizes.max())
    return InducedSubsetComponents(
        n_points=n_points,
        n_components=n_components,
        n_isolated=n_isolated,
        size_min=size_min,
        size_median=float(np.median(sizes)),
        size_max=size_max,
        largest_share=float(size_max) / float(n_points),
    )


@dataclass(frozen=True)
class FeasibleBoundary:
    """
    Constraint boundary ``∂F`` under L1=1.

    A feasible design is on ``∂F`` iff at least one in-bounds L1 neighbour
    is infeasible. Catalogue bounds alone do not place a point on ``∂F``.
    """

    region: FeasibleRegion
    mask: NDArray[np.bool_]
    n_boundary: int
    boundary_fraction_of_f: float | None

    def is_boundary(self, x_index: Sequence[int]) -> bool:
        """Return whether ``x_index`` is on analysis-local ``∂F``."""
        return bool(self.mask[self.region.linear_of(x_index)])


def ela_feasible_boundary(region: FeasibleRegion) -> FeasibleBoundary:
    """
    Mark feasible designs that touch an in-bounds infeasible neighbour.

    Walks feasible linear ids only. Does not build the graph of the
    complete design space. Empty ``F`` yields ``n_boundary=0`` and a
    ``None`` fraction.
    """
    space = region.space
    feasible = region.mask
    if feasible.shape != (space.cardinality,):
        raise ElaError(
            "topology feasible mask length "
            f"({int(feasible.size)}) does not match cardinality "
            f"({space.cardinality})"
        )
    n_feasible = int(np.count_nonzero(feasible))
    if n_feasible != region.n_feasible:
        raise ElaError(
            "topology n_feasible "
            f"({region.n_feasible}) does not match mask sum ({n_feasible})"
        )
    try:
        mask = np.zeros(space.cardinality, dtype=bool)
    except MemoryError as exc:
        raise ElaError(
            "topology boundary could not allocate a cardinality-length mask"
        ) from exc
    for linear in np.flatnonzero(feasible):
        linear_i = int(linear)
        point = linear_to_indices(linear_i, space)
        for neighbour in region.catalogue_neighbours(point):
            if not bool(feasible[indices_to_linear(neighbour, space)]):
                mask[linear_i] = True
                break
    n_boundary = int(np.count_nonzero(mask))
    mask.setflags(write=False)
    fraction = (
        float(n_boundary) / float(n_feasible) if n_feasible else None
    )
    return FeasibleBoundary(
        region=region,
        mask=mask,
        n_boundary=n_boundary,
        boundary_fraction_of_f=fraction,
    )


@dataclass(frozen=True)
class GlobalOptima:
    """
    Every analysis-local feasible design with stored ``y == y*``.

    Attributes
    ----------
    components :
        Partition used to look up membership.
    y_star :
        ``min y`` on analysis-local ``F``, or ``None`` when ``F`` is empty.
    linear_ids :
        Mixed-radix addresses of all optima, increasing. Read-only.
    component_ids :
        Component id of each optimum, same order. Read-only.
    n_optima :
        ``len(linear_ids)``.
    n_optima_components :
        Number of distinct components that contain at least one optimum.
    """

    components: FeasibleComponents
    y_star: float | None
    linear_ids: NDArray[np.int64]
    component_ids: NDArray[np.int32]
    n_optima: int
    n_optima_components: int


def ela_global_optima(
    *,
    connection: Any,
    table: str,
    components: FeasibleComponents,
) -> GlobalOptima:
    """
    Collect every feasible design whose stored ``y`` equals ``y*``.

    ``y*`` is ``MIN(y)`` on analysis-local ``F``. Ties use exact DOUBLE
    equality against that min on the same subquery. Sampling is not
    accepted here: the region was already gated as complete BF.
    Empty ``F`` returns ``y_star=None`` and empty id arrays.
    """
    region = components.region
    if region.n_feasible == 0:
        return _empty_global_optima(components)
    space = region.space
    names = region.constraint_names
    table_sql = _quote_table(table)
    import_duckdb()
    idx_sql, y_sql, f_sql = _f_idx_y_sql(
        table_sql, space.var_names, names
    )
    summary = connection.execute(
        f"SELECT MIN({y_sql}) AS y_star, COUNT(*) AS n_f, "
        f"BOOL_OR(NOT isfinite({y_sql})) AS any_nonfinite "
        f"FROM ({f_sql}) AS feasible"
    ).fetchone()
    if summary is None:
        raise ElaError("topology y* summary query returned no row")
    n_f = int(summary[1])
    if n_f != region.n_feasible:
        raise ElaError(
            "topology y* feasible count "
            f"({n_f}) does not match region n_feasible ({region.n_feasible})"
        )
    if bool(summary[2]):
        raise ElaError("topology y* requires finite stored y on analysis-local F")
    if summary[0] is None:
        raise ElaError("topology y* MIN(y) returned NULL on non-empty F")
    y_star = float(summary[0])
    if not math.isfinite(y_star):
        raise ElaError("topology y* is non-finite")
    rows = connection.execute(
        f"SELECT {idx_sql} FROM ({f_sql}) AS feasible "
        f"WHERE {y_sql} = (SELECT MIN({y_sql}) FROM ({f_sql}) AS feasible_min)"
    ).fetchall()
    pairs: list[tuple[int, int]] = []
    for row in rows:
        point = tuple(int(value) for value in row)
        try:
            linear = indices_to_linear(point, space)
        except IndexSpaceError as exc:
            raise ElaError(str(exc)) from exc
        if not bool(region.mask[linear]):
            raise ElaError(
                "topology y* row is analysis-local F in SQL but not in the mask"
            )
        cid = int(components.labels[linear])
        if cid < 0:
            raise ElaError("topology y* row has no component id")
        pairs.append((linear, cid))
    pairs.sort(key=lambda item: item[0])
    n_optima = len(pairs)
    try:
        linear_ids = np.array([item[0] for item in pairs], dtype=np.int64)
        component_ids = np.array([item[1] for item in pairs], dtype=np.int32)
    except MemoryError as exc:
        raise ElaError("topology y* could not allocate the optimum id arrays") from exc
    linear_ids.setflags(write=False)
    component_ids.setflags(write=False)
    n_optima_components = int(np.unique(component_ids).size) if n_optima else 0
    return GlobalOptima(
        components=components,
        y_star=y_star,
        linear_ids=linear_ids,
        component_ids=component_ids,
        n_optima=n_optima,
        n_optima_components=n_optima_components,
    )


def _empty_global_optima(components: FeasibleComponents) -> GlobalOptima:
    linear_ids = np.zeros(0, dtype=np.int64)
    component_ids = np.zeros(0, dtype=np.int32)
    linear_ids.setflags(write=False)
    component_ids.setflags(write=False)
    return GlobalOptima(
        components=components,
        y_star=None,
        linear_ids=linear_ids,
        component_ids=component_ids,
        n_optima=0,
        n_optima_components=0,
    )


@dataclass(frozen=True)
class NearOptimalBand:
    """One caller-supplied ``τ`` evaluated on analysis-local ``F``."""

    tau: float
    n_g: int | None
    g_fraction_of_f: float | None
    n_components_represented: int | None
    b_tau: float | None
    n_g_components: int | None
    g_largest_share: float | None
    n_g_isolated: int | None
    g_size_min: int | None
    g_size_median: float | None
    g_size_max: int | None


@dataclass(frozen=True)
class NearOptimalSets:
    """``G_τ`` / ``B_τ`` for every requested ``τ``."""

    optima: GlobalOptima
    boundary: FeasibleBoundary
    bands: tuple[NearOptimalBand, ...]


def ela_near_optimal(
    *,
    connection: Any,
    table: str,
    optima: GlobalOptima,
    boundary: FeasibleBoundary,
    taus: Sequence[float],
) -> NearOptimalSets:
    """
    Compute ``G_τ`` and ``B_τ`` for a caller-supplied fractional ``τ`` list.

    ``G_τ = {x in F : y <= (1+τ) y*}`` on raw ``y``. ``τ = 0`` uses
    ``y <= y*`` with no multiply. ``B_τ`` is undefined when ``G`` is
    empty. Negative ``y*`` or empty ``F`` leaves every band field
    ``None`` except ``tau``. This is not a BF ``y_p`` percent band.
    """
    if boundary.region is not optima.components.region:
        raise ElaError(
            "topology near-optimal boundary region does not match optima"
        )
    tau_values = _require_taus(taus)
    y_star = optima.y_star
    if y_star is None or y_star < 0.0:
        bands = tuple(
            NearOptimalBand(
                tau=tau,
                n_g=None,
                g_fraction_of_f=None,
                n_components_represented=None,
                b_tau=None,
                n_g_components=None,
                g_largest_share=None,
                n_g_isolated=None,
                g_size_min=None,
                g_size_median=None,
                g_size_max=None,
            )
            for tau in tau_values
        )
        return NearOptimalSets(optima=optima, boundary=boundary, bands=bands)
    region = optima.components.region
    n_feasible = region.n_feasible
    linears, y = _project_f_idx_y(connection, table, region)
    labels = optima.components.labels
    feasible_mask = region.mask
    bands_list: list[NearOptimalBand] = []
    for tau in tau_values:
        if tau == 0.0:
            in_g = y <= y_star
        else:
            threshold = (1.0 + tau) * y_star
            if not math.isfinite(threshold):
                raise ElaError(
                    f"topology near-optimal threshold is non-finite for tau={tau!r}"
                )
            in_g = y <= threshold
        n_g = int(np.count_nonzero(in_g))
        if n_g == 0:
            bands_list.append(
                NearOptimalBand(
                    tau=tau,
                    n_g=0,
                    g_fraction_of_f=(0.0 if n_feasible else None),
                    n_components_represented=0,
                    b_tau=None,
                    n_g_components=0,
                    g_largest_share=None,
                    n_g_isolated=0,
                    g_size_min=None,
                    g_size_median=None,
                    g_size_max=None,
                )
            )
            continue
        g_linears = linears[in_g]
        subset = np.zeros(feasible_mask.shape, dtype=bool)
        subset[g_linears] = True
        induced = _induced_l1_components(region, subset)
        n_components_represented = int(np.unique(labels[g_linears]).size)
        n_g_boundary = int(np.count_nonzero(boundary.mask[g_linears]))
        bands_list.append(
            NearOptimalBand(
                tau=tau,
                n_g=n_g,
                g_fraction_of_f=float(n_g) / float(n_feasible),
                n_components_represented=n_components_represented,
                b_tau=float(n_g_boundary) / float(n_g),
                n_g_components=induced.n_components,
                g_largest_share=induced.largest_share,
                n_g_isolated=induced.n_isolated,
                g_size_min=induced.size_min,
                g_size_median=induced.size_median,
                g_size_max=induced.size_max,
            )
        )
    return NearOptimalSets(
        optima=optima,
        boundary=boundary,
        bands=tuple(bands_list),
    )


def ela_feasible_region(
    *,
    connection: Any,
    table: str,
    constraint_names: Sequence[str],
    var_names: Sequence[str],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    cardinality: int,
    problem_id: str,
    source_dir: Path,
    source_type: str = SOURCE_BRUTE_FORCE,
) -> FeasibleRegion:
    """
    Build the feasible mask for a complete BF table.

    DuckDB projects ``idx__*`` only for analysis-local ``F``. Sampling is
    a hard error. ``problem_id`` / ``source_dir`` are accepted for later
    tidy rows and are not stored on the region in this tick.
    """
    del problem_id, source_dir
    names = tuple(constraint_names)
    variables = tuple(var_names)
    if not names:
        raise ElaError("topology requires at least one selected constraint")
    if not variables:
        raise ElaError("topology requires sealed variable names")
    if source_type != SOURCE_BRUTE_FORCE:
        raise ElaError(
            "topology requires a complete brute-force catalogue, "
            f"got source_type={source_type!r}"
        )
    if not isinstance(cardinality, int) or isinstance(cardinality, bool) or cardinality < 1:
        raise ElaError(f"cardinality must be a positive int, got {cardinality!r}")
    try:
        space = IndexSpace.from_bounds(variables, lower_bounds, upper_bounds)
    except IndexSpaceError as exc:
        raise ElaError(str(exc)) from exc
    if space.cardinality != cardinality:
        raise ElaError(
            "topology IndexSpace cardinality "
            f"({space.cardinality}) does not match sealed cardinality "
            f"({cardinality})"
        )
    table_sql = _quote_table(table)
    import_duckdb()
    idx_sql = ", ".join(_quote_ident(f"idx__{name}") for name in variables)
    unique_sql = (
        f"COUNT(DISTINCT ({idx_sql})) AS n_unique"
        if len(variables) > 1
        else f"COUNT(DISTINCT {idx_sql}) AS n_unique"
    )
    count_row = connection.execute(
        f"SELECT COUNT(*) AS n_input, {unique_sql} FROM {table_sql}"
    ).fetchone()
    if count_row is None:
        raise ElaError("topology count query returned no row")
    n_input = int(count_row[0])
    n_unique = int(count_row[1])
    if n_input != cardinality:
        raise ElaError(
            "topology requires observed rows to equal catalogue "
            f"cardinality ({cardinality}), got {n_input}"
        )
    if n_unique != cardinality:
        raise ElaError(
            "topology requires unique complete designs equal to "
            f"cardinality ({cardinality}), got n_unique={n_unique}"
        )
    violation_sql = constraint_violation_sql(names)
    query = (
        f"SELECT {idx_sql} FROM ("
        f"SELECT {idx_sql}, {violation_sql} FROM {table_sql}"
        f') WHERE {_quote_ident("F")}'
    )
    mask = np.zeros(space.cardinality, dtype=bool)
    result = connection.execute(query)
    rows = result.fetchall()
    for row in rows:
        point = tuple(int(value) for value in row)
        try:
            linear = indices_to_linear(point, space)
        except IndexSpaceError as exc:
            raise ElaError(str(exc)) from exc
        mask[linear] = True
    mask.setflags(write=False)
    return FeasibleRegion(
        space=space,
        mask=mask,
        n_feasible=int(mask.sum()),
        n_input=n_input,
        n_unique=n_unique,
        constraint_names=names,
    )


def parse_ela_taus(raw: Sequence[float]) -> tuple[float, ...]:
    """
    Validate caller-supplied ``τ`` values.

    Empty is allowed. Duplicates fail closed. There is no library default.
    """
    values = _require_taus(raw)
    if len(set(values)) != len(values):
        raise ElaError(f"taus must not contain duplicates, got {values!r}")
    return values


def tau_feature_stem(tau: float) -> str:
    """Return the tidy key stem for one ``τ`` (canonical JSON number)."""
    return json.dumps(float(tau), ensure_ascii=True)


def n_g_feature_name(tau: float) -> str:
    """Return the tidy key for ``|G_τ|``."""
    return f"topology.n_G.{tau_feature_stem(tau)}"


def g_fraction_feature_name(tau: float) -> str:
    """Return the tidy key for ``|G_τ| / |F|``."""
    return f"topology.G_fraction_of_f.{tau_feature_stem(tau)}"


def n_components_represented_feature_name(tau: float) -> str:
    """Return the tidy key for components represented in ``G_τ``."""
    return f"topology.n_components_represented.{tau_feature_stem(tau)}"


def b_tau_feature_name(tau: float) -> str:
    """Return the tidy key for ``B_τ``."""
    return f"topology.B_tau.{tau_feature_stem(tau)}"


def n_g_components_feature_name(tau: float) -> str:
    """Return the tidy key for connected components of the induced ``G_τ`` graph."""
    return f"topology.n_G_components.{tau_feature_stem(tau)}"


def g_largest_share_feature_name(tau: float) -> str:
    """Return the tidy key for the largest ``G_τ`` component share."""
    return f"topology.G_largest_share.{tau_feature_stem(tau)}"


def n_g_isolated_feature_name(tau: float) -> str:
    """Return the tidy key for isolated points in the induced ``G_τ`` graph."""
    return f"topology.n_G_isolated.{tau_feature_stem(tau)}"


def g_size_min_feature_name(tau: float) -> str:
    return f"topology.G_size_min.{tau_feature_stem(tau)}"


def g_size_median_feature_name(tau: float) -> str:
    return f"topology.G_size_median.{tau_feature_stem(tau)}"


def g_size_max_feature_name(tau: float) -> str:
    return f"topology.G_size_max.{tau_feature_stem(tau)}"


def ela_topology_features(
    *,
    connection: Any,
    table: str,
    constraint_names: Sequence[str],
    var_names: Sequence[str],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    cardinality: int,
    problem_id: str,
    source_dir: Path,
    taus: Sequence[float],
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
    seed: int | None = None,
) -> list[FeatureResult]:
    """
    Compute tidy topology aggregates for a complete BF table.

    Does not emit per-component sizes, optimum linear ids, or
    cardinality-length masks. ``taus`` is caller-supplied; empty means
    no ``G_τ`` / ``B_τ`` rows.
    """
    tau_values = parse_ela_taus(taus)
    config_payload: dict[str, Any] = {"taus": list(tau_values)}
    started = time.monotonic()
    region = ela_feasible_region(
        connection=connection,
        table=table,
        constraint_names=constraint_names,
        var_names=var_names,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        cardinality=cardinality,
        problem_id=problem_id,
        source_dir=source_dir,
        source_type=source_type,
    )
    components = ela_feasible_components(region)
    boundary = ela_feasible_boundary(region)
    optima = ela_global_optima(
        connection=connection, table=table, components=components
    )
    near = ela_near_optimal(
        connection=connection,
        table=table,
        optima=optima,
        boundary=boundary,
        taus=tau_values,
    )
    runtime_s = max(0.0, time.monotonic() - started)
    names = tuple(constraint_names)
    sig_feasibility = analysis_signature(
        family=FAMILY,
        landscape=LANDSCAPE_FEASIBILITY,
        constraint_names=names,
        x_view=X_VIEW_INDEX,
        response_view=RESPONSE_VIEW_NA,
        config=config_payload,
    )
    sig_y_raw = analysis_signature(
        family=FAMILY,
        landscape=LANDSCAPE_Y_RAW,
        constraint_names=names,
        x_view=X_VIEW_INDEX,
        response_view=RESPONSE_VIEW_NA,
        config=config_payload,
    )
    identity = {
        "problem_id": problem_id,
        "source_type": source_type,
        "source_dir": Path(source_dir),
        "n_input": region.n_input,
        "n_unique": region.n_unique,
        "runtime_s": runtime_s,
        "config_payload": config_payload,
        "sampling_design_signature": sampling_design_signature,
        "seed": seed,
    }
    rows: list[FeatureResult] = [
        _topology_row(
            identity,
            landscape=LANDSCAPE_FEASIBILITY,
            feature_name=FEATURE_N_FEASIBLE,
            value=float(region.n_feasible),
            analysis_signature=sig_feasibility,
        ),
        _topology_row(
            identity,
            landscape=LANDSCAPE_FEASIBILITY,
            feature_name=FEATURE_N_COMPONENTS,
            value=float(components.n_components),
            analysis_signature=sig_feasibility,
        ),
        _topology_row(
            identity,
            landscape=LANDSCAPE_FEASIBILITY,
            feature_name=FEATURE_N_ISOLATED,
            value=float(components.n_isolated),
            analysis_signature=sig_feasibility,
        ),
        _topology_row(
            identity,
            landscape=LANDSCAPE_FEASIBILITY,
            feature_name=FEATURE_SIZE_MIN,
            value=_optional_float(components.size_min),
            analysis_signature=sig_feasibility,
        ),
        _topology_row(
            identity,
            landscape=LANDSCAPE_FEASIBILITY,
            feature_name=FEATURE_SIZE_MEDIAN,
            value=_optional_float(components.size_median),
            analysis_signature=sig_feasibility,
        ),
        _topology_row(
            identity,
            landscape=LANDSCAPE_FEASIBILITY,
            feature_name=FEATURE_SIZE_MAX,
            value=_optional_float(components.size_max),
            analysis_signature=sig_feasibility,
        ),
        _topology_row(
            identity,
            landscape=LANDSCAPE_FEASIBILITY,
            feature_name=FEATURE_LARGEST_SHARE,
            value=_optional_float(components.largest_share),
            analysis_signature=sig_feasibility,
        ),
        _topology_row(
            identity,
            landscape=LANDSCAPE_FEASIBILITY,
            feature_name=FEATURE_ISOLATED_FRACTION_OF_F,
            value=_optional_float(components.isolated_fraction_of_f),
            analysis_signature=sig_feasibility,
        ),
        _topology_row(
            identity,
            landscape=LANDSCAPE_FEASIBILITY,
            feature_name=FEATURE_ISOLATED_FRACTION_OF_COMPONENTS,
            value=_optional_float(components.isolated_fraction_of_components),
            analysis_signature=sig_feasibility,
        ),
        _topology_row(
            identity,
            landscape=LANDSCAPE_Y_RAW,
            feature_name=FEATURE_Y_STAR,
            value=_optional_float(optima.y_star),
            analysis_signature=sig_y_raw,
            role=ROLE_DIAGNOSTIC,
        ),
        _topology_row(
            identity,
            landscape=LANDSCAPE_Y_RAW,
            feature_name=FEATURE_N_OPTIMA,
            value=float(optima.n_optima),
            analysis_signature=sig_y_raw,
        ),
        _topology_row(
            identity,
            landscape=LANDSCAPE_Y_RAW,
            feature_name=FEATURE_N_OPTIMA_COMPONENTS,
            value=float(optima.n_optima_components),
            analysis_signature=sig_y_raw,
        ),
        _topology_row(
            identity,
            landscape=LANDSCAPE_FEASIBILITY,
            feature_name=FEATURE_N_BOUNDARY,
            value=float(boundary.n_boundary),
            analysis_signature=sig_feasibility,
        ),
        _topology_row(
            identity,
            landscape=LANDSCAPE_FEASIBILITY,
            feature_name=FEATURE_BOUNDARY_FRACTION_OF_F,
            value=_optional_float(boundary.boundary_fraction_of_f),
            analysis_signature=sig_feasibility,
        ),
    ]
    for band in near.bands:
        rows.extend(
            [
                _topology_row(
                    identity,
                    landscape=LANDSCAPE_Y_RAW,
                    feature_name=n_g_feature_name(band.tau),
                    value=_optional_float(band.n_g),
                    analysis_signature=sig_y_raw,
                ),
                _topology_row(
                    identity,
                    landscape=LANDSCAPE_Y_RAW,
                    feature_name=g_fraction_feature_name(band.tau),
                    value=_optional_float(band.g_fraction_of_f),
                    analysis_signature=sig_y_raw,
                ),
                _topology_row(
                    identity,
                    landscape=LANDSCAPE_Y_RAW,
                    feature_name=n_components_represented_feature_name(band.tau),
                    value=_optional_float(band.n_components_represented),
                    analysis_signature=sig_y_raw,
                ),
                _topology_row(
                    identity,
                    landscape=LANDSCAPE_Y_RAW,
                    feature_name=b_tau_feature_name(band.tau),
                    value=_optional_float(band.b_tau),
                    analysis_signature=sig_y_raw,
                ),
                _topology_row(
                    identity,
                    landscape=LANDSCAPE_Y_RAW,
                    feature_name=n_g_components_feature_name(band.tau),
                    value=_optional_float(band.n_g_components),
                    analysis_signature=sig_y_raw,
                ),
                _topology_row(
                    identity,
                    landscape=LANDSCAPE_Y_RAW,
                    feature_name=g_largest_share_feature_name(band.tau),
                    value=_optional_float(band.g_largest_share),
                    analysis_signature=sig_y_raw,
                ),
                _topology_row(
                    identity,
                    landscape=LANDSCAPE_Y_RAW,
                    feature_name=n_g_isolated_feature_name(band.tau),
                    value=_optional_float(band.n_g_isolated),
                    analysis_signature=sig_y_raw,
                ),
                _topology_row(
                    identity,
                    landscape=LANDSCAPE_Y_RAW,
                    feature_name=g_size_min_feature_name(band.tau),
                    value=_optional_float(band.g_size_min),
                    analysis_signature=sig_y_raw,
                ),
                _topology_row(
                    identity,
                    landscape=LANDSCAPE_Y_RAW,
                    feature_name=g_size_median_feature_name(band.tau),
                    value=_optional_float(band.g_size_median),
                    analysis_signature=sig_y_raw,
                ),
                _topology_row(
                    identity,
                    landscape=LANDSCAPE_Y_RAW,
                    feature_name=g_size_max_feature_name(band.tau),
                    value=_optional_float(band.g_size_max),
                    analysis_signature=sig_y_raw,
                ),
            ]
        )
    return rows


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _topology_row(
    identity: Mapping[str, Any],
    *,
    landscape: str,
    feature_name: str,
    value: float | None,
    analysis_signature: str,
    role: str = ROLE_PRIMARY,
) -> FeatureResult:
    return feature_result(
        problem_id=identity["problem_id"],
        source_type=identity["source_type"],
        source_dir=identity["source_dir"],
        landscape=landscape,
        response_view=RESPONSE_VIEW_NA,
        x_view=X_VIEW_INDEX,
        family=FAMILY,
        feature_name=feature_name,
        status=STATUS_OK if value is not None else STATUS_UNDEFINED,
        role=role,
        n_input=identity["n_input"],
        n_unique=identity["n_unique"],
        runtime_s=identity["runtime_s"],
        library=LIBRARY,
        library_version=_library_version(),
        config_payload=identity["config_payload"],
        value=value,
        sampling_design_signature=identity["sampling_design_signature"],
        analysis_signature=analysis_signature,
        seed=identity["seed"],
    )


def _library_version() -> str:
    try:
        return importlib.metadata.version("hpdesignbench")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _as_int_tuple(x_index: Sequence[int]) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in x_index)
    except (TypeError, ValueError) as exc:
        raise ElaError(f"catalogue indices must be ints, got {x_index!r}") from exc


def _idx_sql(var_names: Sequence[str]) -> str:
    return ", ".join(_quote_ident(f"idx__{name}") for name in var_names)


def _f_idx_y_sql(
    table_sql: str,
    var_names: Sequence[str],
    constraint_names: Sequence[str],
) -> tuple[str, str, str]:
    idx_sql = _idx_sql(var_names)
    y_sql = _quote_ident("y")
    violation_sql = constraint_violation_sql(constraint_names)
    f_sql = (
        f"SELECT {idx_sql}, {y_sql} FROM ("
        f"SELECT {idx_sql}, CAST({y_sql} AS DOUBLE) AS {y_sql}, "
        f"{violation_sql} FROM {table_sql}"
        f') WHERE {_quote_ident("F")}'
    )
    return idx_sql, y_sql, f_sql


def _project_f_idx_y(
    connection: Any,
    table: str,
    region: FeasibleRegion,
) -> tuple[NDArray[np.int64], NDArray[np.floating]]:
    table_sql = _quote_table(table)
    import_duckdb()
    _idx_sql_text, _y_sql, f_sql = _f_idx_y_sql(
        table_sql, region.space.var_names, region.constraint_names
    )
    del _idx_sql_text, _y_sql
    rows = connection.execute(f_sql).fetchall()
    if len(rows) != region.n_feasible:
        raise ElaError(
            "topology near-optimal feasible count "
            f"({len(rows)}) does not match region n_feasible "
            f"({region.n_feasible})"
        )
    n_vars = len(region.space.var_names)
    try:
        linears = np.empty(len(rows), dtype=np.int64)
        y = np.empty(len(rows), dtype=float)
    except MemoryError as exc:
        raise ElaError(
            "topology near-optimal could not allocate F objective arrays"
        ) from exc
    for index, row in enumerate(rows):
        point = tuple(int(value) for value in row[:n_vars])
        try:
            linear = indices_to_linear(point, region.space)
        except IndexSpaceError as exc:
            raise ElaError(str(exc)) from exc
        if not bool(region.mask[linear]):
            raise ElaError(
                "topology near-optimal row is analysis-local F in SQL "
                "but not in the mask"
            )
        linears[index] = linear
        y[index] = float(row[n_vars])
    if not np.isfinite(y).all():
        raise ElaError(
            "topology near-optimal requires finite stored y on analysis-local F"
        )
    return linears, y


def _require_taus(taus: Sequence[float]) -> tuple[float, ...]:
    if isinstance(taus, (str, bytes)):
        raise ElaError("taus must be a sequence of finite floats >= 0")
    try:
        values = tuple(taus)
    except TypeError as exc:
        raise ElaError("taus must be a sequence of finite floats >= 0") from exc
    out: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ElaError(f"tau must be a finite float >= 0, got {item!r}")
        number = float(item)
        if not math.isfinite(number) or number < 0.0:
            raise ElaError(f"tau must be a finite float >= 0, got {item!r}")
        out.append(number)
    return tuple(out)


def _quote_table(table: str) -> str:
    if not isinstance(table, str) or not _TABLE_NAME.match(table):
        raise ElaError(f"table must be a simple identifier, got {table!r}")
    return _quote_ident(table)


def _quote_ident(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ElaError(f"identifier must be a non-empty string, got {name!r}")
    return '"' + name.replace('"', '""') + '"'
