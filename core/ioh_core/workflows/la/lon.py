"""
Exhaustive set-valued local optima network on a declared scalar landscape.

An exhaustive, set-valued adaptation of Vérel’s stochastic best-improvement
LON that records all reachable local-optimum neutral networks under equal-best
choices.

Nodes are IEEE-equal L1-connected local-minimum plateaus. Basins are overlapping
terminal-support sets, not the disjoint co-reach components emitted by the
legacy ``basins`` family. Escape edges enumerate exact L1 shells around nodes;
funnels are sink supports in SCC-condensed cumulative non-worsening graphs.
"""
from __future__ import annotations

import importlib.metadata
import math
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.la.catalogue_table import load_catalogue_landscape
from core.ioh_core.workflows.la.lon_landscape import get_lon_landscape
from core.ioh_core.workflows.la.models import (
    ElaError,
    FeatureResult,
    LANDSCAPE_Y_P,
    RESPONSE_VIEW_RAW,
    ROLE_DIAGNOSTIC,
    ROLE_PRIMARY,
    ROLE_SECONDARY,
    SOURCE_BRUTE_FORCE,
    STATUS_OK,
    STATUS_UNDEFINED,
    X_VIEW_INDEX,
    feature_result,
)
from core.ioh_core.workflows.la.plateau_graph import (
    PlateauGraph,
    build_plateau_graph,
    nested_best_exit_support,
)
from core.ioh_core.workflows.la.signatures import analysis_signature

FAMILY = "lon"
LIBRARY = "hpdesignbench"
PRODUCTION_RADII = (2, 3, 4)

FEATURE_N_NODES = "lon.n_nodes"
FEATURE_N_GLOBAL_NODES = "lon.n_global_nodes"
FEATURE_NODE_SIZE_MIN = "lon.node_size_min"
FEATURE_NODE_SIZE_MEDIAN = "lon.node_size_median"
FEATURE_NODE_SIZE_MAX = "lon.node_size_max"
FEATURE_LARGEST_NODE_SHARE = "lon.largest_node_share"
FEATURE_N_SINGLETON_NODES = "lon.n_singleton_nodes"
FEATURE_SUPPORT_SIZE_MIN = "lon.support_size_min"
FEATURE_SUPPORT_SIZE_MEDIAN = "lon.support_size_median"
FEATURE_SUPPORT_SIZE_MAX = "lon.support_size_max"
FEATURE_LARGEST_SUPPORT_SHARE = "lon.largest_support_share"
FEATURE_GLOBAL_SUPPORT_SHARE = "lon.global_support_share"
FEATURE_AMBIGUOUS_DESIGN_COUNT = "lon.ambiguous_design_count"
FEATURE_AMBIGUOUS_DESIGN_SHARE = "lon.ambiguous_design_share"
FEATURE_TERMINAL_MULTIPLICITY_MEAN = "lon.terminal_multiplicity_mean"
FEATURE_TERMINAL_MULTIPLICITY_MEDIAN = "lon.terminal_multiplicity_median"
FEATURE_TERMINAL_MULTIPLICITY_MAX = "lon.terminal_multiplicity_max"
FEATURE_ALL_FEASIBLE_NODE_SHARE = "lon.all_feasible_node_share"
FEATURE_MIXED_FEASIBILITY_NODE_SHARE = "lon.mixed_feasibility_node_share"
FEATURE_ALL_INFEASIBLE_NODE_SHARE = "lon.all_infeasible_node_share"
FEATURE_BASIN_F_SHARE_MIN = "lon.basin_f_share_min"
FEATURE_BASIN_F_SHARE_MEDIAN = "lon.basin_f_share_median"
FEATURE_BASIN_F_SHARE_MAX = "lon.basin_f_share_max"
FEATURE_Y_P_STAR = "lon.y_p_star"
FEATURE_Y_BOX_STAR = "lon.y_box_star"
FEATURE_RUNTIME = "lon.costs_runtime"


@dataclass(frozen=True)
class FunnelResult:
    """Funnel summaries for one cumulative escape radius."""

    radius: int
    n_funnels: int
    n_global_sinks: int
    global_node_share: float
    global_design_share: float
    ambiguous_node_share: float
    largest_nonglobal_node_share: float | None
    largest_nonglobal_design_share: float | None
    nonglobal_hops_to_global_min: float | None
    nonglobal_hops_to_global_median: float | None
    nonglobal_hops_to_global_max: float | None
    global_sink_ids: tuple[int, ...]
    node_sink_support: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class LonResult:
    """In-memory exact LON result and compact terminal-support representation."""

    space: IndexSpace
    landscape: str
    graph: PlateauGraph
    support_single: NDArray[np.int32]
    support_multiple: Mapping[int, tuple[int, ...]]
    node_size: NDArray[np.int64]
    node_value: NDArray[np.float64]
    node_min_linear: NDArray[np.int64]
    node_n_feasible: NDArray[np.int64]
    node_any_feasible: NDArray[np.bool_]
    node_all_feasible: NDArray[np.bool_]
    has_feasible: bool
    support_size: NDArray[np.int64]
    basin_n_feasible: NDArray[np.int64]
    global_support_count: int
    ambiguous_design_count: int
    terminal_multiplicity_counts: Mapping[int, int]
    overlaps: Mapping[tuple[int, int], int]
    edge_support_count: Mapping[tuple[int, int, int], int]
    shell_candidate_count: Mapping[tuple[int, int], int]
    shell_improving_candidate_count: Mapping[tuple[int, int], int]
    funnels: Mapping[int, FunnelResult]
    radii: tuple[int, ...]
    global_value: float
    stage_runtime_s: Mapping[str, float]
    runtime_s: float

    @property
    def n_nodes(self) -> int:
        return int(self.node_size.size)

    @property
    def basin_f_share(self) -> NDArray[np.float64]:
        """Stored-feasibility share of each overlapping terminal-support basin."""
        sizes = self.support_size.astype(np.float64, copy=False)
        return self.basin_n_feasible.astype(np.float64, copy=False) / sizes

    @property
    def node_y_p(self) -> NDArray[np.float64]:
        """Backward-compatible y_p node values; invalid for a raw-objective LON."""
        if self.landscape != LANDSCAPE_Y_P:
            raise ElaError("node_y_p is only defined for landscape='y_p'")
        return self.node_value

    @property
    def y_p_star(self) -> float:
        """Backward-compatible y_p global value; invalid for a raw-objective LON."""
        if self.landscape != LANDSCAPE_Y_P:
            raise ElaError("y_p_star is only defined for landscape='y_p'")
        return self.global_value

    def support_for_plateau(self, plateau_id: int) -> tuple[int, ...]:
        multiple = self.support_multiple.get(int(plateau_id))
        if multiple is not None:
            return multiple
        node_id = int(self.support_single[int(plateau_id)])
        if node_id < 0:
            raise ElaError(f"plateau {plateau_id} has no terminal support")
        return (node_id,)

    def support_for_linear(self, linear_id: int) -> tuple[int, ...]:
        plateau_id = int(self.graph.plateau_of_linear[int(linear_id)])
        return self.support_for_plateau(plateau_id)


def set_valued_lon(
    *,
    space: IndexSpace,
    values: ArrayLike | None = None,
    landscape: str = LANDSCAPE_Y_P,
    y_p: ArrayLike | None = None,
    feasible: ArrayLike | None = None,
    radii: Sequence[int] = PRODUCTION_RADII,
    source_type: str = SOURCE_BRUTE_FORCE,
) -> LonResult:
    """Compute nodes, overlapping basin support, escape edges, and funnels."""
    started = time.monotonic()
    selected_landscape = get_lon_landscape(landscape).token
    if str(source_type) != SOURCE_BRUTE_FORCE:
        raise ElaError(
            "catalogue LON requires source_type='brute_force', "
            f"got {source_type!r}"
        )
    raw_values = _resolve_landscape_values(
        values=values,
        y_p=y_p,
        landscape=selected_landscape,
    )
    n = int(space.cardinality)
    if int(raw_values.size) != n:
        raise ElaError(
            f"{selected_landscape} length ({raw_values.size}) "
            f"does not match cardinality ({n})"
        )
    if not bool(np.all(np.isfinite(raw_values))):
        raise ElaError(
            f"non-finite {selected_landscape} in landscape source; "
            "reject whole source"
        )
    resolved_radii = _parse_radii(radii)

    flags: NDArray[np.bool_] | None
    if feasible is None:
        flags = None
    else:
        flags = np.asarray(feasible, dtype=np.bool_).reshape(-1)
        if int(flags.size) != n:
            raise ElaError(
                f"feasible length ({flags.size}) does not match cardinality ({n})"
            )

    stage_runtime_s: dict[str, float] = {}
    stage_started = time.monotonic()
    graph = build_plateau_graph(raw_values, space)
    stage_runtime_s["plateau_graph"] = time.monotonic() - stage_started
    stage_started = time.monotonic()
    support_single, support_multiple = nested_best_exit_support(graph)
    stage_runtime_s["terminal_support"] = time.monotonic() - stage_started
    stage_started = time.monotonic()
    (
        support_size,
        ambiguous_count,
        multiplicity_counts,
        overlaps,
        global_support_count,
    ) = _support_aggregates(graph, support_single, support_multiple, raw_values)
    (
        node_size,
        node_value,
        node_min_linear,
        node_n_feasible,
        node_any_feasible,
        node_all_feasible,
    ) = _node_arrays(graph, flags)
    basin_n_feasible = _basin_feasible_counts(
        graph, support_single, support_multiple, flags
    )
    stage_runtime_s["support_aggregates"] = time.monotonic() - stage_started

    edge_counts: dict[tuple[int, int, int], int] = {}
    shell_counts: dict[tuple[int, int], int] = {}
    improving_counts: dict[tuple[int, int], int] = {}
    stage_started = time.monotonic()
    if resolved_radii:
        edge_counts, shell_counts, improving_counts, _feas, _infeas = _escape_edges(
            space=space,
            graph=graph,
            support_single=support_single,
            support_multiple=support_multiple,
            node_value=node_value,
            max_radius=max(resolved_radii),
        )
    stage_runtime_s["escape_edges"] = time.monotonic() - stage_started
    funnels: dict[int, FunnelResult] = {}
    for radius in resolved_radii:
        stage_started = time.monotonic()
        funnels[radius] = _funnel_result(
            radius=radius,
            n_designs=n,
            graph=graph,
            support_single=support_single,
            support_multiple=support_multiple,
            node_value=node_value,
            edge_counts=edge_counts,
        )
        stage_runtime_s[f"funnels_d{radius}"] = time.monotonic() - stage_started

    for array in (
        support_single,
        node_size,
        node_value,
        node_min_linear,
        node_n_feasible,
        node_any_feasible,
        node_all_feasible,
        support_size,
        basin_n_feasible,
    ):
        array.setflags(write=False)
    result = LonResult(
        space=space,
        landscape=selected_landscape,
        graph=graph,
        support_single=support_single,
        support_multiple=support_multiple,
        node_size=node_size,
        node_value=node_value,
        node_min_linear=node_min_linear,
        node_n_feasible=node_n_feasible,
        node_any_feasible=node_any_feasible,
        node_all_feasible=node_all_feasible,
        has_feasible=flags is not None,
        support_size=support_size,
        basin_n_feasible=basin_n_feasible,
        global_support_count=global_support_count,
        ambiguous_design_count=ambiguous_count,
        terminal_multiplicity_counts=multiplicity_counts,
        overlaps=overlaps,
        edge_support_count=edge_counts,
        shell_candidate_count=shell_counts,
        shell_improving_candidate_count=improving_counts,
        funnels=funnels,
        radii=resolved_radii,
        global_value=float(np.min(raw_values)),
        stage_runtime_s=stage_runtime_s,
        runtime_s=max(0.0, time.monotonic() - started),
    )
    if global_support_count <= 0:
        raise ElaError(
            "no catalogue design supports a node at the landscape global value"
        )
    _validate_lon_result(result)
    return result


def ela_lon(
    *,
    space: IndexSpace,
    values: ArrayLike | None = None,
    landscape: str = LANDSCAPE_Y_P,
    y_p: ArrayLike | None = None,
    feasible: ArrayLike | None = None,
    radii: Sequence[int] = PRODUCTION_RADII,
    problem_id: str,
    source_dir: Path,
    constraint_names: Sequence[str],
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
    seed: int | None = None,
) -> tuple[list[FeatureResult], LonResult]:
    """Compute the exact LON and its tidy scalar feature rows."""
    if feasible is None:
        raise ElaError("LON publication requires a stored feasible column")
    result = set_valued_lon(
        space=space,
        values=values,
        landscape=landscape,
        y_p=y_p,
        feasible=feasible,
        radii=radii,
        source_type=source_type,
    )
    rows = _feature_rows(
        result,
        problem_id=problem_id,
        source_dir=source_dir,
        constraint_names=constraint_names,
        source_type=source_type,
        sampling_design_signature=sampling_design_signature,
        seed=seed,
    )
    return rows, result


def ela_lon_from_table(
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
    radii: Sequence[int],
    landscape: str = LANDSCAPE_Y_P,
    source_type: str = SOURCE_BRUTE_FORCE,
) -> tuple[list[FeatureResult], LonResult]:
    """Load a complete catalogue table and compute the exhaustive LON."""
    space = IndexSpace.from_bounds(var_names, lower_bounds, upper_bounds)
    if int(space.cardinality) != int(cardinality):
        raise ElaError(
            f"IndexSpace cardinality ({space.cardinality}) does not match "
            f"schema ({cardinality})"
        )
    selected_landscape = get_lon_landscape(landscape).token
    values, feasible = load_catalogue_landscape(
        connection,
        table,
        space,
        expected_n=int(cardinality),
        landscape=selected_landscape,
    )
    return ela_lon(
        space=space,
        values=values,
        landscape=selected_landscape,
        feasible=feasible,
        radii=radii,
        problem_id=problem_id,
        source_dir=source_dir,
        constraint_names=constraint_names,
        source_type=source_type,
    )


def lon_node_records(result: LonResult) -> list[dict[str, Any]]:
    """Return one publication record per LONN/node."""
    records: list[dict[str, Any]] = []
    spec = get_lon_landscape(result.landscape)
    for node_id in range(result.n_nodes):
        size = int(result.node_size[node_id])
        n_feasible = (
            int(result.node_n_feasible[node_id]) if result.has_feasible else None
        )
        records.append(
            {
                "node_id": node_id,
                "plateau_id": int(result.graph.node_plateaus[node_id]),
                "min_linear": int(result.node_min_linear[node_id]),
                spec.token: float(result.node_value[node_id]),
                "node_size": size,
                spec.global_marker_column: bool(
                    float(result.node_value[node_id]) == result.global_value
                ),
                "n_feasible": n_feasible,
                "feasible_share": (
                    float(n_feasible) / float(size)
                    if n_feasible is not None and size
                    else None
                ),
                "all_feasible": (
                    bool(result.node_all_feasible[node_id])
                    if result.has_feasible
                    else None
                ),
                "any_feasible": (
                    bool(result.node_any_feasible[node_id])
                    if result.has_feasible
                    else None
                ),
                "support_size": int(result.support_size[node_id]),
                "basin_n_feasible": (
                    int(result.basin_n_feasible[node_id])
                    if result.has_feasible
                    else None
                ),
                "basin_f_share": (
                    float(result.basin_f_share[node_id])
                    if result.has_feasible
                    else None
                ),
            }
        )
    return records


def overlap_records(result: LonResult) -> list[dict[str, int]]:
    """Return exact pairwise basin-overlap counts."""
    return [
        {"node_a": left, "node_b": right, "design_count": int(count)}
        for (left, right), count in sorted(result.overlaps.items())
    ]


def escape_edge_records(result: LonResult) -> list[dict[str, Any]]:
    """Return one exact-shell record per supported escape edge."""
    records: list[dict[str, Any]] = []
    spec = get_lon_landscape(result.landscape)
    for (source, target, shell), count in sorted(result.edge_support_count.items()):
        source_y = float(result.node_value[source])
        target_y = float(result.node_value[target])
        relation = (
            "self"
            if source == target
            else "better"
            if target_y < source_y
            else "equal"
            if target_y == source_y
            else "worse"
        )
        records.append(
            {
                "source_node_id": source,
                "target_node_id": target,
                "shell": shell,
                "support_count": int(count),
                "source_shell_candidate_count": int(
                    result.shell_candidate_count[(source, shell)]
                ),
                spec.source_value_column: source_y,
                spec.target_value_column: target_y,
                "relation": relation,
            }
        )
    return records


def _resolve_landscape_values(
    *,
    values: ArrayLike | None,
    y_p: ArrayLike | None,
    landscape: str,
) -> NDArray[np.float64]:
    if values is not None and y_p is not None:
        raise ElaError("provide values or legacy y_p, not both")
    if values is None:
        if y_p is None:
            raise ElaError(f"missing values for landscape {landscape!r}")
        if landscape != LANDSCAPE_Y_P:
            raise ElaError("legacy y_p values require landscape='y_p'")
        values = y_p
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _parse_radii(raw: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw)
    if any(isinstance(value, bool) for value in raw):
        raise ElaError("LON radii must be positive integers")
    if any(value <= 0 for value in values):
        raise ElaError("LON radii must be positive integers")
    if len(set(values)) != len(values):
        raise ElaError("LON radii must not contain duplicates")
    if values != tuple(sorted(values)):
        raise ElaError("LON radii must be strictly increasing")
    return values


def _validate_lon_result(result: LonResult) -> None:
    """Fail closed on support, escape, and funnel consistency invariants."""
    n = int(result.space.cardinality)
    n_nodes = result.n_nodes
    if int(sum(result.terminal_multiplicity_counts.values())) != n:
        raise ElaError("terminal-support multiplicity counts do not cover catalogue")
    node_support = result.support_single[result.graph.node_plateaus]
    expected = np.arange(n_nodes, dtype=np.int32)
    if not bool(np.array_equal(node_support, expected)):
        raise ElaError("at least one LON node does not support itself")
    if bool(np.any(result.support_size < result.node_size)):
        raise ElaError("a node support size is smaller than its node size")
    if result.has_feasible and (
        bool(np.any(result.node_n_feasible < 0))
        or bool(np.any(result.node_n_feasible > result.node_size))
    ):
        raise ElaError("node feasibility counts are inconsistent with node sizes")
    if result.has_feasible and (
        bool(np.any(result.basin_n_feasible < 0))
        or bool(np.any(result.basin_n_feasible > result.support_size))
    ):
        raise ElaError(
            "basin feasible counts are inconsistent with terminal-support sizes"
        )
    if any(
        left < 0
        or right >= n_nodes
        or left >= right
        or count <= 0
        or count > n
        for (left, right), count in result.overlaps.items()
    ):
        raise ElaError("invalid pairwise basin-overlap relation")

    previous_pairs: set[tuple[int, int]] = set()
    for radius in result.radii:
        cumulative_pairs = {
            (source, target)
            for (source, target, shell), count
            in result.edge_support_count.items()
            if shell <= radius and count > 0
        }
        if not previous_pairs.issubset(cumulative_pairs):
            raise ElaError("cumulative escape graphs are not nested")
        previous_pairs = cumulative_pairs
    for (source, target, shell), count in result.edge_support_count.items():
        if not (0 <= source < n_nodes and 0 <= target < n_nodes):
            raise ElaError("escape edge references an unknown node")
        denominator = result.shell_candidate_count.get((source, shell))
        if denominator is None or count <= 0 or count > denominator:
            raise ElaError("escape support count exceeds its shell denominator")
    if result.radii:
        max_radius = max(result.radii)
        expected_shells = {
            (source, shell)
            for source in range(n_nodes)
            for shell in range(1, max_radius + 1)
        }
        if set(result.shell_candidate_count) != expected_shells:
            raise ElaError("escape shell denominators are incomplete")
        if any(count < 0 for count in result.shell_candidate_count.values()):
            raise ElaError("escape shell denominator is negative")
        if set(result.shell_improving_candidate_count) != expected_shells:
            raise ElaError("improving candidate counts are incomplete")
        for key, improving in result.shell_improving_candidate_count.items():
            denominator = result.shell_candidate_count.get(key)
            if denominator is None or improving < 0 or improving > denominator:
                raise ElaError("improving candidate count exceeds shell denominator")

    global_nodes = np.flatnonzero(result.node_value == result.global_value)
    for radius in result.radii:
        funnel = result.funnels.get(radius)
        if funnel is None or funnel.n_funnels < 1:
            raise ElaError(f"radius {radius} has no funnel result")
        if len(funnel.node_sink_support) != n_nodes or any(
            not support for support in funnel.node_sink_support
        ):
            raise ElaError(f"radius {radius} has empty node funnel support")
        global_sinks = set(funnel.global_sink_ids)
        if not global_sinks or funnel.n_global_sinks != len(global_sinks):
            raise ElaError(f"radius {radius} has no valid global sink")
        if any(
            not global_sinks.intersection(funnel.node_sink_support[int(node)])
            for node in global_nodes
        ):
            raise ElaError(
                f"a global node cannot reach a global sink at radius {radius}"
            )


def _support_aggregates(
    graph: PlateauGraph,
    support_single: NDArray[np.int32],
    support_multiple: Mapping[int, tuple[int, ...]],
    values: NDArray[np.float64],
) -> tuple[
    NDArray[np.int64],
    int,
    dict[int, int],
    dict[tuple[int, int], int],
    int,
]:
    n_nodes = graph.n_nodes
    singleton_mask = support_single >= 0
    support_size = np.bincount(
        support_single[singleton_mask],
        weights=graph.plateau_size[singleton_mask],
        minlength=n_nodes,
    ).astype(np.int64)
    ambiguity = 0
    multiplicity: dict[int, int] = {
        1: int(np.sum(graph.plateau_size[singleton_mask]))
    }
    overlaps: dict[tuple[int, int], int] = {}
    for pid, nodes in support_multiple.items():
        size = int(graph.plateau_size[pid])
        ambiguity += size
        multiplicity[len(nodes)] = multiplicity.get(len(nodes), 0) + size
        for node_id in nodes:
            support_size[node_id] += size
        for pair in combinations(nodes, 2):
            overlaps[pair] = overlaps.get(pair, 0) + size

    node_y = graph.plateau_value[graph.node_plateaus]
    global_value = float(np.min(values))
    global_node = node_y == global_value
    global_count = int(
        np.sum(
            graph.plateau_size[
                singleton_mask & global_node[support_single.clip(min=0)]
            ]
        )
    )
    for pid, nodes in support_multiple.items():
        if any(bool(global_node[node_id]) for node_id in nodes):
            global_count += int(graph.plateau_size[pid])
    return support_size, ambiguity, multiplicity, overlaps, global_count


def _basin_feasible_counts(
    graph: PlateauGraph,
    support_single: NDArray[np.int32],
    support_multiple: Mapping[int, tuple[int, ...]],
    feasible: NDArray[np.bool_] | None,
) -> NDArray[np.int64]:
    n_nodes = graph.n_nodes
    if feasible is None:
        return np.zeros(n_nodes, dtype=np.int64)
    n_plateaus = graph.plateau_size.size
    plateau_feasible = np.bincount(
        graph.plateau_of_linear,
        weights=feasible.astype(np.int64, copy=False),
        minlength=n_plateaus,
    ).astype(np.int64)
    singleton_mask = support_single >= 0
    basin = np.bincount(
        support_single[singleton_mask],
        weights=plateau_feasible[singleton_mask],
        minlength=n_nodes,
    ).astype(np.int64)
    for pid, nodes in support_multiple.items():
        count = int(plateau_feasible[pid])
        for node_id in nodes:
            basin[node_id] += count
    return basin


def _node_arrays(
    graph: PlateauGraph,
    feasible: NDArray[np.bool_] | None,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.bool_],
    NDArray[np.bool_],
]:
    node_size = graph.plateau_size[graph.node_plateaus].astype(np.int64, copy=True)
    node_y = graph.plateau_value[graph.node_plateaus].astype(np.float64, copy=True)
    node_min = graph.plateau_min_linear[graph.node_plateaus].astype(
        np.int64, copy=True
    )
    if feasible is None:
        counts = np.zeros(graph.n_nodes, dtype=np.int64)
    else:
        node_of_linear = graph.plateau_to_node[graph.plateau_of_linear]
        in_node = node_of_linear >= 0
        counts = np.bincount(
            node_of_linear[in_node],
            weights=feasible[in_node].astype(np.int64),
            minlength=graph.n_nodes,
        ).astype(np.int64)
    any_feasible = counts > 0
    all_feasible = counts == node_size if feasible is not None else np.zeros(
        graph.n_nodes, dtype=np.bool_
    )
    return node_size, node_y, node_min, counts, any_feasible, all_feasible


@lru_cache(maxsize=None)
def _l1_offsets(n_dim: int, radius: int) -> tuple[tuple[int, ...], ...]:
    offsets: list[tuple[int, ...]] = []

    def visit(dim: int, remaining: int, prefix: list[int]) -> None:
        if dim == n_dim:
            if remaining == 0:
                offsets.append(tuple(prefix))
            return
        for value in range(-remaining, remaining + 1):
            prefix.append(value)
            visit(dim + 1, remaining - abs(value), prefix)
            prefix.pop()

    visit(0, int(radius), [])
    return tuple(offset for offset in offsets if any(offset))


def _escape_edges(
    *,
    space: IndexSpace,
    graph: PlateauGraph,
    support_single: NDArray[np.int32],
    support_multiple: Mapping[int, tuple[int, ...]],
    node_value: NDArray[np.float64],
    max_radius: int,
    feasible: NDArray[np.bool_] | None = None,
) -> tuple[
    dict[tuple[int, int, int], int],
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
]:
    node_of_linear = graph.plateau_to_node[graph.plateau_of_linear]
    members: list[list[int]] = [[] for _ in range(graph.n_nodes)]
    for linear in np.flatnonzero(node_of_linear >= 0):
        members[int(node_of_linear[linear])].append(int(linear))

    edges: dict[tuple[int, int, int], int] = {}
    shell_counts: dict[tuple[int, int], int] = {}
    improving_counts: dict[tuple[int, int], int] = {}
    feasible_counts: dict[tuple[int, int], int] = {}
    infeasible_counts: dict[tuple[int, int], int] = {}
    flags = None if feasible is None else np.asarray(feasible, dtype=np.bool_).reshape(-1)
    lowers = tuple(int(value) for value in space.lower_bounds)
    uppers = tuple(int(value) for value in space.upper_bounds)
    strides = tuple(int(value) for value in space.strides)
    sizes = tuple(int(value) for value in space.sizes)

    for source, source_members in enumerate(members):
        seen = set(source_members)
        coordinates = [
            tuple(
                lower + (linear // stride) % size
                for lower, stride, size in zip(
                    lowers, strides, sizes, strict=True
                )
            )
            for linear in source_members
        ]
        for shell in range(1, max_radius + 1):
            candidates: set[int] = set()
            for coordinate in coordinates:
                for offset in _l1_offsets(len(strides), shell):
                    moved = tuple(
                        value + delta
                        for value, delta in zip(coordinate, offset, strict=True)
                    )
                    if any(
                        value < lower or value > upper
                        for value, lower, upper in zip(
                            moved, lowers, uppers, strict=True
                        )
                    ):
                        continue
                    linear = sum(
                        (value - lower) * stride
                        for value, lower, stride in zip(
                            moved, lowers, strides, strict=True
                        )
                    )
                    if linear not in seen:
                        candidates.add(int(linear))
            seen.update(candidates)
            shell_counts[(source, shell)] = len(candidates)
            improving = 0
            n_feasible = 0
            n_infeasible = 0
            for linear in candidates:
                if flags is not None:
                    if bool(flags[linear]):
                        n_feasible += 1
                    else:
                        n_infeasible += 1
                pid = int(graph.plateau_of_linear[linear])
                targets = support_multiple.get(pid)
                if targets is None:
                    target = int(support_single[pid])
                    if target < 0:
                        raise ElaError(f"escape candidate {linear} has no support")
                    targets = (target,)
                if any(
                    float(node_value[target]) < float(node_value[source])
                    for target in targets
                ):
                    improving += 1
                for target in targets:
                    key = (source, int(target), shell)
                    edges[key] = edges.get(key, 0) + 1
            improving_counts[(source, shell)] = improving
            feasible_counts[(source, shell)] = n_feasible
            infeasible_counts[(source, shell)] = n_infeasible
    return edges, shell_counts, improving_counts, feasible_counts, infeasible_counts


def _funnel_result(
    *,
    radius: int,
    n_designs: int,
    graph: PlateauGraph,
    support_single: NDArray[np.int32],
    support_multiple: Mapping[int, tuple[int, ...]],
    node_value: NDArray[np.float64],
    edge_counts: Mapping[tuple[int, int, int], int],
) -> FunnelResult:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n_nodes = int(node_value.size)
    pairs = sorted(
        {
            (source, target)
            for (source, target, shell), count in edge_counts.items()
            if count > 0
            and shell <= radius
            and source != target
            and float(node_value[target]) <= float(node_value[source])
        }
    )
    if pairs:
        rows = np.fromiter((pair[0] for pair in pairs), dtype=np.int32)
        cols = np.fromiter((pair[1] for pair in pairs), dtype=np.int32)
        matrix = csr_matrix(
            (np.ones(len(pairs), dtype=np.int8), (rows, cols)),
            shape=(n_nodes, n_nodes),
        )
    else:
        matrix = csr_matrix((n_nodes, n_nodes), dtype=np.int8)
    n_components, labels = connected_components(
        matrix, directed=True, connection="strong", return_labels=True
    )
    outgoing: list[set[int]] = [set() for _ in range(n_components)]
    indegree = np.zeros(n_components, dtype=np.int32)
    for source, target in pairs:
        left = int(labels[source])
        right = int(labels[target])
        if left != right and right not in outgoing[left]:
            outgoing[left].add(right)
            indegree[right] += 1
    queue = deque(int(i) for i in np.flatnonzero(indegree == 0))
    topological: list[int] = []
    while queue:
        component = queue.popleft()
        topological.append(component)
        for target in outgoing[component]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(topological) != n_components:
        raise ElaError("SCC condensation is unexpectedly cyclic")

    sinks = tuple(i for i, targets in enumerate(outgoing) if not targets)
    sink_support: list[tuple[int, ...]] = [tuple() for _ in range(n_components)]
    for component in reversed(topological):
        if not outgoing[component]:
            sink_support[component] = (component,)
        else:
            sink_support[component] = tuple(
                sorted(
                    {
                        sink
                        for target in outgoing[component]
                        for sink in sink_support[target]
                    }
                )
            )
    node_support = tuple(sink_support[int(labels[node])] for node in range(n_nodes))
    global_value = float(np.min(node_value))
    global_sinks = {
        sink
        for sink in sinks
        if bool(np.any(node_value[labels == sink] == global_value))
    }
    node_reaches_global = np.fromiter(
        (bool(global_sinks.intersection(support)) for support in node_support),
        dtype=np.bool_,
        count=n_nodes,
    )
    global_node_share = (
        float(np.count_nonzero(node_reaches_global)) / float(n_nodes)
        if n_nodes
        else 0.0
    )
    ambiguous_node_share = (
        float(sum(len(support) > 1 for support in node_support)) / float(n_nodes)
        if n_nodes
        else 0.0
    )

    node_counts = np.zeros(n_components, dtype=np.int64)
    for support in node_support:
        for sink in support:
            node_counts[sink] += 1
    design_counts = np.zeros(n_components, dtype=np.int64)
    node_single_sink = np.full(n_nodes, -1, dtype=np.int32)
    for node, support in enumerate(node_support):
        if len(support) == 1:
            node_single_sink[node] = support[0]
    singleton_mask = support_single >= 0
    singleton_pids = np.flatnonzero(singleton_mask)
    singleton_nodes = support_single[singleton_mask]
    singleton_sizes = graph.plateau_size[singleton_mask]
    resolved_sinks = node_single_sink[singleton_nodes]
    simple = resolved_sinks >= 0
    if bool(np.any(simple)):
        design_counts += np.bincount(
            resolved_sinks[simple],
            weights=singleton_sizes[simple],
            minlength=n_components,
        ).astype(np.int64)
    global_designs = int(
        np.sum(singleton_sizes[node_reaches_global[singleton_nodes]])
    )
    for raw_pid in singleton_pids[~simple]:
        pid = int(raw_pid)
        node = int(support_single[pid])
        size = int(graph.plateau_size[pid])
        for sink in node_support[node]:
            design_counts[sink] += size
    for pid, nodes in support_multiple.items():
        design_sinks = {
            sink for node in nodes for sink in node_support[int(node)]
        }
        size = int(graph.plateau_size[pid])
        for sink in design_sinks:
            design_counts[sink] += size
        if global_sinks.intersection(design_sinks):
            global_designs += size
    global_design_share = (
        float(global_designs) / float(n_designs) if n_designs else 0.0
    )

    nonglobal_sinks = tuple(sink for sink in sinks if sink not in global_sinks)
    if nonglobal_sinks:
        largest_node = float(max(node_counts[sink] for sink in nonglobal_sinks))
        largest_design = float(max(design_counts[sink] for sink in nonglobal_sinks))
        largest_node_share = largest_node / float(n_nodes) if n_nodes else 0.0
        largest_design_share = (
            largest_design / float(n_designs) if n_designs else 0.0
        )
    else:
        largest_node_share = None
        largest_design_share = None

    distance = np.full(n_components, np.inf, dtype=np.float64)
    for sink in global_sinks:
        distance[sink] = 0.0
    for component in reversed(topological):
        if outgoing[component]:
            best = min(float(distance[target]) for target in outgoing[component])
            if math.isfinite(best):
                distance[component] = min(float(distance[component]), best + 1.0)
    nonglobal_hops = [
        float(distance[int(labels[node])])
        for node in range(n_nodes)
        if float(node_value[node]) != global_value
        and math.isfinite(float(distance[int(labels[node])]))
    ]
    if nonglobal_hops:
        hop_min = float(np.min(nonglobal_hops))
        hop_median = float(np.median(nonglobal_hops))
        hop_max = float(np.max(nonglobal_hops))
    else:
        hop_min = hop_median = hop_max = None
    return FunnelResult(
        radius=radius,
        n_funnels=len(sinks),
        n_global_sinks=len(global_sinks),
        global_node_share=global_node_share,
        global_design_share=global_design_share,
        ambiguous_node_share=ambiguous_node_share,
        largest_nonglobal_node_share=largest_node_share,
        largest_nonglobal_design_share=largest_design_share,
        nonglobal_hops_to_global_min=hop_min,
        nonglobal_hops_to_global_median=hop_median,
        nonglobal_hops_to_global_max=hop_max,
        global_sink_ids=tuple(sorted(global_sinks)),
        node_sink_support=node_support,
    )


def _feature_rows(
    result: LonResult,
    *,
    problem_id: str,
    source_dir: Path,
    constraint_names: Sequence[str],
    source_type: str,
    sampling_design_signature: str | None,
    seed: int | None,
) -> list[FeatureResult]:
    n = int(result.space.cardinality)
    n_nodes = result.n_nodes
    multiplicity = result.terminal_multiplicity_counts
    multiplicity_mean = (
        sum(level * count for level, count in multiplicity.items()) / float(n)
        if n
        else None
    )
    multiplicity_median = _weighted_median(multiplicity)
    multiplicity_max = max(multiplicity, default=None)
    global_nodes = result.node_value == result.global_value
    config = {"radii": list(result.radii)}
    sig = analysis_signature(
        family=FAMILY,
        landscape=result.landscape,
        constraint_names=tuple(constraint_names),
        x_view=X_VIEW_INDEX,
        response_view=RESPONSE_VIEW_RAW,
        config=config,
    )
    identity = {
        "problem_id": problem_id,
        "landscape": result.landscape,
        "source_type": source_type,
        "source_dir": Path(source_dir),
        "n_input": n,
        "n_unique": n,
        "runtime_s": result.runtime_s,
        "config_payload": config,
        "sampling_design_signature": sampling_design_signature,
        "seed": seed,
        "analysis_signature": sig,
    }
    rows = [
        _row(identity, FEATURE_N_NODES, float(n_nodes)),
        _row(identity, FEATURE_N_GLOBAL_NODES, float(np.count_nonzero(global_nodes)), role=ROLE_SECONDARY),
        _row(identity, FEATURE_NODE_SIZE_MIN, float(np.min(result.node_size))),
        _row(identity, FEATURE_NODE_SIZE_MEDIAN, float(np.median(result.node_size))),
        _row(identity, FEATURE_NODE_SIZE_MAX, float(np.max(result.node_size))),
        _row(identity, FEATURE_LARGEST_NODE_SHARE, float(np.max(result.node_size)) / float(n)),
        _row(identity, FEATURE_N_SINGLETON_NODES, float(np.count_nonzero(result.node_size == 1)), role=ROLE_SECONDARY),
        _row(identity, FEATURE_SUPPORT_SIZE_MIN, float(np.min(result.support_size))),
        _row(identity, FEATURE_SUPPORT_SIZE_MEDIAN, float(np.median(result.support_size))),
        _row(identity, FEATURE_SUPPORT_SIZE_MAX, float(np.max(result.support_size))),
        _row(identity, FEATURE_LARGEST_SUPPORT_SHARE, float(np.max(result.support_size)) / float(n)),
        _row(identity, FEATURE_GLOBAL_SUPPORT_SHARE, float(result.global_support_count) / float(n)),
        _row(identity, FEATURE_AMBIGUOUS_DESIGN_COUNT, float(result.ambiguous_design_count), role=ROLE_SECONDARY),
        _row(identity, FEATURE_AMBIGUOUS_DESIGN_SHARE, float(result.ambiguous_design_count) / float(n)),
        _row(identity, FEATURE_TERMINAL_MULTIPLICITY_MEAN, multiplicity_mean, role=ROLE_SECONDARY),
        _row(identity, FEATURE_TERMINAL_MULTIPLICITY_MEDIAN, multiplicity_median, role=ROLE_SECONDARY),
        _row(identity, FEATURE_TERMINAL_MULTIPLICITY_MAX, _optional_float(multiplicity_max), role=ROLE_SECONDARY),
    ]
    if result.has_feasible:
        mixed = result.node_any_feasible & ~result.node_all_feasible
        rows.extend(
            [
                _row(identity, FEATURE_ALL_FEASIBLE_NODE_SHARE, float(np.count_nonzero(result.node_all_feasible)) / float(n_nodes), role=ROLE_SECONDARY),
                _row(identity, FEATURE_MIXED_FEASIBILITY_NODE_SHARE, float(np.count_nonzero(mixed)) / float(n_nodes), role=ROLE_SECONDARY),
                _row(identity, FEATURE_ALL_INFEASIBLE_NODE_SHARE, float(np.count_nonzero(~result.node_any_feasible)) / float(n_nodes), role=ROLE_SECONDARY),
            ]
        )
    else:
        rows.extend(
            [
                _row(identity, FEATURE_ALL_FEASIBLE_NODE_SHARE, None, role=ROLE_SECONDARY),
                _row(identity, FEATURE_MIXED_FEASIBILITY_NODE_SHARE, None, role=ROLE_SECONDARY),
                _row(identity, FEATURE_ALL_INFEASIBLE_NODE_SHARE, None, role=ROLE_SECONDARY),
            ]
        )
    if result.has_feasible:
        shares = result.basin_f_share
        rows.extend(
            [
                _row(identity, FEATURE_BASIN_F_SHARE_MIN, float(np.min(shares))),
                _row(identity, FEATURE_BASIN_F_SHARE_MEDIAN, float(np.median(shares))),
                _row(identity, FEATURE_BASIN_F_SHARE_MAX, float(np.max(shares))),
            ]
        )
    else:
        rows.extend(
            [
                _row(identity, FEATURE_BASIN_F_SHARE_MIN, None),
                _row(identity, FEATURE_BASIN_F_SHARE_MEDIAN, None),
                _row(identity, FEATURE_BASIN_F_SHARE_MAX, None),
            ]
        )

    for radius in result.radii:
        prefix_escape = f"lon.escape.d{radius}"
        prefix_funnel = f"lon.funnel.d{radius}"
        cumulative_pairs = {
            (source, target)
            for (source, target, shell), count in result.edge_support_count.items()
            if shell <= radius and source != target and count > 0
        }
        escapable = {source for source, _target in cumulative_pairs}
        improving = {
            source
            for source, target in cumulative_pairs
            if float(result.node_value[target]) < float(result.node_value[source])
        }
        denominator = sum(
            count
            for (source, shell), count in result.shell_candidate_count.items()
            if shell <= radius
        )
        retained = sum(
            count
            for (source, target, shell), count in result.edge_support_count.items()
            if shell <= radius and source == target
        )
        improving_candidates = sum(
            count
            for (source, shell), count in result.shell_improving_candidate_count.items()
            if shell <= radius
        )
        edge_density = (
            float(len(cumulative_pairs)) / float(n_nodes * (n_nodes - 1))
            if n_nodes > 1
            else 0.0
        )
        rows.extend(
            [
                _row(
                    identity,
                    f"{prefix_escape}.candidate_count",
                    float(denominator),
                    role=ROLE_DIAGNOSTIC,
                ),
                _row(
                    identity,
                    f"{prefix_escape}.support_relation_count",
                    float(
                        sum(
                            count
                            for (_source, _target, shell), count
                            in result.edge_support_count.items()
                            if shell <= radius
                        )
                    ),
                    role=ROLE_DIAGNOSTIC,
                ),
                _row(identity, f"{prefix_escape}.n_edges", float(len(cumulative_pairs)), role=ROLE_SECONDARY),
                _row(identity, f"{prefix_escape}.edge_density", edge_density, role=ROLE_SECONDARY),
                _row(identity, f"{prefix_escape}.escapable_node_share", float(len(escapable)) / float(n_nodes), role=ROLE_SECONDARY),
                _row(identity, f"{prefix_escape}.improving_escapable_node_share", float(len(improving)) / float(n_nodes)),
                _row(identity, f"{prefix_escape}.source_retained_candidate_share", float(retained) / float(denominator) if denominator else None, role=ROLE_SECONDARY),
                _row(identity, f"{prefix_escape}.improving_candidate_share", float(improving_candidates) / float(denominator) if denominator else None, role=ROLE_SECONDARY),
            ]
        )
        funnel = result.funnels[radius]
        rows.extend(
            [
                _row(identity, f"{prefix_funnel}.n_funnels", float(funnel.n_funnels)),
                _row(identity, f"{prefix_funnel}.n_global_sinks", float(funnel.n_global_sinks), role=ROLE_SECONDARY),
                _row(identity, f"{prefix_funnel}.global_node_share", funnel.global_node_share),
                _row(identity, f"{prefix_funnel}.global_design_share", funnel.global_design_share),
                _row(identity, f"{prefix_funnel}.ambiguous_node_share", funnel.ambiguous_node_share, role=ROLE_SECONDARY),
                _row(identity, f"{prefix_funnel}.largest_nonglobal_node_share", funnel.largest_nonglobal_node_share),
                _row(identity, f"{prefix_funnel}.largest_nonglobal_design_share", funnel.largest_nonglobal_design_share),
                _row(identity, f"{prefix_funnel}.nonglobal_hops_to_global_min", funnel.nonglobal_hops_to_global_min, role=ROLE_SECONDARY),
                _row(identity, f"{prefix_funnel}.nonglobal_hops_to_global_median", funnel.nonglobal_hops_to_global_median, role=ROLE_SECONDARY),
                _row(identity, f"{prefix_funnel}.nonglobal_hops_to_global_max", funnel.nonglobal_hops_to_global_max, role=ROLE_SECONDARY),
            ]
        )
    rows.extend(
        [
            _row(
                identity,
                "lon.support_relation_rows",
                float(
                    int(np.count_nonzero(result.support_single >= 0))
                    + sum(len(nodes) for nodes in result.support_multiple.values())
                ),
                role=ROLE_DIAGNOSTIC,
            ),
            _row(
                identity,
                "lon.support_multiple_plateaus",
                float(len(result.support_multiple)),
                role=ROLE_DIAGNOSTIC,
            ),
            _row(
                identity,
                "lon.overlap_pairs",
                float(len(result.overlaps)),
                role=ROLE_DIAGNOSTIC,
            ),
            _row(
                identity,
                get_lon_landscape(result.landscape).global_feature,
                result.global_value,
                role=ROLE_DIAGNOSTIC,
            ),
            _row(identity, FEATURE_RUNTIME, result.runtime_s, role=ROLE_DIAGNOSTIC),
        ]
    )
    for stage, runtime_s in result.stage_runtime_s.items():
        rows.append(
            _row(
                identity,
                f"lon.costs_runtime.{stage}",
                float(runtime_s),
                role=ROLE_DIAGNOSTIC,
            )
        )
    return rows


def _weighted_median(counts: Mapping[int, int]) -> float | None:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return None
    left_rank = (total - 1) // 2
    right_rank = total // 2
    cumulative = 0
    left_value: int | None = None
    for value, count in sorted(counts.items()):
        cumulative += int(count)
        if left_value is None and cumulative > left_rank:
            left_value = int(value)
        if cumulative > right_rank:
            return (float(left_value) + float(value)) / 2.0
    raise ElaError("terminal multiplicity median could not be computed")


def _row(
    identity: Mapping[str, Any],
    feature_name: str,
    value: float | None,
    *,
    role: str = ROLE_PRIMARY,
) -> FeatureResult:
    return feature_result(
        problem_id=identity["problem_id"],
        source_type=identity["source_type"],
        source_dir=identity["source_dir"],
        landscape=identity["landscape"],
        response_view=RESPONSE_VIEW_RAW,
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
        analysis_signature=identity["analysis_signature"],
        seed=identity["seed"],
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _library_version() -> str:
    try:
        return importlib.metadata.version("hpdesignbench")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
