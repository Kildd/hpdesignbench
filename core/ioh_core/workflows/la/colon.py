"""
Exhaustive CoLON on Fieldsend Eq. (6) / Deb comparator h.

Nodes are L1-connected local-minimum plateaus of the Deb preorder, not a
packed ``y_p`` column. Analysis-local ``phi`` / **F** come from selected
utils via :func:`constraint_violations`. Stored Parquet ``feasible`` is
not read. Escape balls use exact L1 shells at production radii 2/3/4;
funnels are SCCs on the cumulative non-worsening Deb-rank graph.
"""
from __future__ import annotations

import importlib.metadata
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.la.catalogue_table import load_catalogue_y_phi_f
from core.ioh_core.workflows.la.lon import (
    FunnelResult,
    PRODUCTION_RADII,
    _basin_feasible_counts,
    _escape_edges,
    _funnel_result,
    _node_arrays,
    _parse_radii,
    _support_aggregates,
)
from core.ioh_core.workflows.la.models import (
    ElaError,
    FeatureResult,
    LANDSCAPE_DEB,
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
    dense_preorder_ranks,
    nested_best_exit_support,
)
from core.ioh_core.workflows.la.signatures import analysis_signature

FAMILY = "colon"
LIBRARY = "hpdesignbench"
STAGE_BASIN_CENSUS = "basin_census"
STAGE_ESCAPES = "escapes"
STAGE_FUNNELS = "funnels"

FEATURE_N_NODES = "colon.n_nodes"
FEATURE_N_GLOBAL_NODES = "colon.n_global_nodes"
FEATURE_NODE_SIZE_MIN = "colon.node_size_min"
FEATURE_NODE_SIZE_MEDIAN = "colon.node_size_median"
FEATURE_NODE_SIZE_MAX = "colon.node_size_max"
FEATURE_LARGEST_NODE_SHARE = "colon.largest_node_share"
FEATURE_N_SINGLETON_NODES = "colon.n_singleton_nodes"
FEATURE_SUPPORT_SIZE_MIN = "colon.support_size_min"
FEATURE_SUPPORT_SIZE_MEDIAN = "colon.support_size_median"
FEATURE_SUPPORT_SIZE_MAX = "colon.support_size_max"
FEATURE_LARGEST_SUPPORT_SHARE = "colon.largest_support_share"
FEATURE_GLOBAL_SUPPORT_SHARE = "colon.global_support_share"
FEATURE_AMBIGUOUS_DESIGN_COUNT = "colon.ambiguous_design_count"
FEATURE_AMBIGUOUS_DESIGN_SHARE = "colon.ambiguous_design_share"
FEATURE_TERMINAL_MULTIPLICITY_MEAN = "colon.terminal_multiplicity_mean"
FEATURE_TERMINAL_MULTIPLICITY_MEDIAN = "colon.terminal_multiplicity_median"
FEATURE_TERMINAL_MULTIPLICITY_MAX = "colon.terminal_multiplicity_max"
FEATURE_BASIN_F_SHARE_MIN = "colon.basin_f_share_min"
FEATURE_BASIN_F_SHARE_MEDIAN = "colon.basin_f_share_median"
FEATURE_BASIN_F_SHARE_MAX = "colon.basin_f_share_max"
FEATURE_N_TRAP_NODES = "colon.n_trap_nodes"
FEATURE_N_MIXED_BASINS = "colon.n_mixed_basins"
FEATURE_N_PURE_FEASIBLE_BASINS = "colon.n_pure_feasible_basins"
FEATURE_Y_STAR = "colon.y_star"
FEATURE_RUNTIME = "colon.costs_runtime"
FEATURE_STAGE_BASIN_CENSUS = "colon.stage.basin_census"
FEATURE_STAGE_ESCAPES = "colon.stage.escapes"
FEATURE_STAGE_FUNNELS = "colon.stage.funnels"

GLOBAL_FEATURE_NAMES = frozenset(
    {
        FEATURE_N_GLOBAL_NODES,
        FEATURE_GLOBAL_SUPPORT_SHARE,
        FEATURE_Y_STAR,
    }
)


@dataclass(frozen=True)
class ColonResult:
    """In-memory CoLON: basin census, exact-shell escapes, and funnels."""

    space: IndexSpace
    graph: PlateauGraph
    support_single: NDArray[np.int32]
    support_multiple: Mapping[int, tuple[int, ...]]
    node_size: NDArray[np.int64]
    node_rank: NDArray[np.float64]
    node_y_raw: NDArray[np.float64]
    node_min_linear: NDArray[np.int64]
    terminal_feasible: NDArray[np.bool_]
    node_n_feasible: NDArray[np.int64]
    support_size: NDArray[np.int64]
    basin_n_feasible: NDArray[np.int64]
    basin_n_total: NDArray[np.int64]
    global_node: NDArray[np.bool_]
    y_star: float | None
    n_global_nodes: int | None
    global_support_count: int | None
    ambiguous_design_count: int
    terminal_multiplicity_counts: Mapping[int, int]
    overlaps: Mapping[tuple[int, int], int]
    edge_support_count: Mapping[tuple[int, int, int], int]
    shell_candidate_count: Mapping[tuple[int, int], int]
    shell_improving_candidate_count: Mapping[tuple[int, int], int]
    shell_feasible_candidate_count: Mapping[tuple[int, int], int]
    shell_infeasible_candidate_count: Mapping[tuple[int, int], int]
    funnels: Mapping[int, FunnelResult]
    radii: tuple[int, ...]
    stage_runtime_s: Mapping[str, float]
    runtime_s: float

    @property
    def n_nodes(self) -> int:
        return int(self.node_size.size)

    @property
    def basin_f_share(self) -> NDArray[np.float64]:
        sizes = self.basin_n_total.astype(np.float64, copy=False)
        return self.basin_n_feasible.astype(np.float64, copy=False) / sizes

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


def set_valued_colon(
    *,
    space: IndexSpace,
    y_raw: ArrayLike,
    feasible: ArrayLike,
    phi: ArrayLike,
    radii: Sequence[int] = PRODUCTION_RADII,
    source_type: str = SOURCE_BRUTE_FORCE,
) -> ColonResult:
    """Compute CoLON nodes, overlapping Deb support, escape edges, and funnels."""
    started = time.monotonic()
    if str(source_type) != SOURCE_BRUTE_FORCE:
        raise ElaError(
            "catalogue CoLON requires source_type='brute_force', "
            f"got {source_type!r}"
        )
    n = int(space.cardinality)
    y = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    flags = np.asarray(feasible, dtype=np.bool_).reshape(-1)
    phi_values = np.asarray(phi, dtype=np.float64).reshape(-1)
    if int(y.size) != n:
        raise ElaError(f"y_raw length ({y.size}) does not match cardinality ({n})")
    if int(flags.size) != n:
        raise ElaError(
            f"feasible length ({flags.size}) does not match cardinality ({n})"
        )
    if int(phi_values.size) != n:
        raise ElaError(
            f"phi length ({phi_values.size}) does not match cardinality ({n})"
        )
    if not bool(np.all(np.isfinite(y))):
        raise ElaError("non-finite y_raw in CoLON source; reject whole source")
    if not bool(np.all(np.isfinite(phi_values))):
        raise ElaError("non-finite phi in CoLON source; reject whole source")
    resolved_radii = _parse_radii(radii)

    keys = np.empty((n, 2), dtype=np.float64)
    keys[:, 0] = np.where(flags, 0.0, 1.0)
    keys[:, 1] = np.where(flags, y, phi_values)
    ranks = dense_preorder_ranks(keys)

    stage_runtime_s: dict[str, float] = {}
    stage_started = time.monotonic()
    graph = build_plateau_graph(ranks, space)
    _reject_mixed_feasibility_plateaus(graph, flags)
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
        _rank_global_count,
    ) = _support_aggregates(graph, support_single, support_multiple, ranks)
    (
        node_size,
        node_rank,
        node_min_linear,
        node_n_feasible,
        _node_any,
        _node_all,
    ) = _node_arrays(graph, flags)
    basin_n_feasible = _basin_feasible_counts(
        graph, support_single, support_multiple, flags
    )
    node_y_raw = y[node_min_linear].astype(np.float64, copy=True)
    terminal_feasible = flags[node_min_linear].astype(np.bool_, copy=True)
    has_f = bool(np.any(flags))
    y_star: float | None
    global_node: NDArray[np.bool_]
    n_global_nodes: int | None
    global_support_count: int | None
    if has_f:
        y_star = float(np.min(y[flags]))
        global_node = terminal_feasible & (node_y_raw == y_star)
        n_global_nodes = int(np.count_nonzero(global_node))
        global_support_count = _support_count_for_nodes(
            graph, support_single, support_multiple, global_node
        )
        if n_global_nodes < 1 or global_support_count is None or global_support_count <= 0:
            raise ElaError(
                "no catalogue design supports a CoLON node at y* on analysis-local F"
            )
    else:
        y_star = None
        global_node = np.zeros(graph.n_nodes, dtype=np.bool_)
        n_global_nodes = None
        global_support_count = None
    stage_runtime_s["support_aggregates"] = time.monotonic() - stage_started

    edge_counts: dict[tuple[int, int, int], int] = {}
    shell_counts: dict[tuple[int, int], int] = {}
    improving_counts: dict[tuple[int, int], int] = {}
    feasible_counts: dict[tuple[int, int], int] = {}
    infeasible_counts: dict[tuple[int, int], int] = {}
    stage_started = time.monotonic()
    if resolved_radii:
        (
            edge_counts,
            shell_counts,
            improving_counts,
            feasible_counts,
            infeasible_counts,
        ) = _escape_edges(
            space=space,
            graph=graph,
            support_single=support_single,
            support_multiple=support_multiple,
            node_value=node_rank,
            max_radius=max(resolved_radii),
            feasible=flags,
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
            node_value=node_rank,
            edge_counts=edge_counts,
        )
        stage_runtime_s[f"funnels_d{radius}"] = time.monotonic() - stage_started

    for array in (
        support_single,
        node_size,
        node_rank,
        node_y_raw,
        node_min_linear,
        terminal_feasible,
        node_n_feasible,
        support_size,
        basin_n_feasible,
        global_node,
    ):
        array.setflags(write=False)
    result = ColonResult(
        space=space,
        graph=graph,
        support_single=support_single,
        support_multiple=support_multiple,
        node_size=node_size,
        node_rank=node_rank,
        node_y_raw=node_y_raw,
        node_min_linear=node_min_linear,
        terminal_feasible=terminal_feasible,
        node_n_feasible=node_n_feasible,
        support_size=support_size,
        basin_n_feasible=basin_n_feasible,
        basin_n_total=support_size,
        global_node=global_node,
        y_star=y_star,
        n_global_nodes=n_global_nodes,
        global_support_count=global_support_count,
        ambiguous_design_count=ambiguous_count,
        terminal_multiplicity_counts=multiplicity_counts,
        overlaps=overlaps,
        edge_support_count=edge_counts,
        shell_candidate_count=shell_counts,
        shell_improving_candidate_count=improving_counts,
        shell_feasible_candidate_count=feasible_counts,
        shell_infeasible_candidate_count=infeasible_counts,
        funnels=funnels,
        radii=resolved_radii,
        stage_runtime_s=stage_runtime_s,
        runtime_s=max(0.0, time.monotonic() - started),
    )
    _validate_colon_result(result)
    return result


def ela_colon(
    *,
    space: IndexSpace,
    y_raw: ArrayLike,
    feasible: ArrayLike,
    phi: ArrayLike,
    problem_id: str,
    source_dir: Path,
    constraint_names: Sequence[str],
    radii: Sequence[int] = PRODUCTION_RADII,
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
    seed: int | None = None,
) -> tuple[list[FeatureResult], ColonResult]:
    """Compute the CoLON census and tidy scalar feature rows."""
    result = set_valued_colon(
        space=space,
        y_raw=y_raw,
        feasible=feasible,
        phi=phi,
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


def ela_colon_from_table(
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
    radii: Sequence[int] = PRODUCTION_RADII,
    source_type: str = SOURCE_BRUTE_FORCE,
) -> tuple[list[FeatureResult], ColonResult]:
    """Load catalogue y_raw and analysis-local phi / F; compute CoLON census."""
    space = IndexSpace.from_bounds(var_names, lower_bounds, upper_bounds)
    if int(space.cardinality) != int(cardinality):
        raise ElaError(
            f"IndexSpace cardinality ({space.cardinality}) does not match "
            f"schema ({cardinality})"
        )
    y_raw, phi, feasible = load_catalogue_y_phi_f(
        connection,
        table,
        space,
        constraint_names,
        expected_n=int(cardinality),
    )
    return ela_colon(
        space=space,
        y_raw=y_raw,
        feasible=feasible,
        phi=phi,
        radii=radii,
        problem_id=problem_id,
        source_dir=source_dir,
        constraint_names=constraint_names,
        source_type=source_type,
    )


def colon_node_records(result: ColonResult) -> list[dict[str, Any]]:
    """Return one publication record per CoLON node."""
    records: list[dict[str, Any]] = []
    shares = result.basin_f_share
    for node_id in range(result.n_nodes):
        size = int(result.node_size[node_id])
        basin_total = int(result.basin_n_total[node_id])
        basin_feasible = int(result.basin_n_feasible[node_id])
        records.append(
            {
                "node_id": node_id,
                "plateau_id": int(result.graph.node_plateaus[node_id]),
                "min_linear": int(result.node_min_linear[node_id]),
                "y_raw": float(result.node_y_raw[node_id]),
                "node_size": size,
                "terminal_feasible": bool(result.terminal_feasible[node_id]),
                "is_global": (
                    bool(result.global_node[node_id])
                    if result.y_star is not None
                    else None
                ),
                "n_feasible": int(result.node_n_feasible[node_id]),
                "basin_n_feasible": basin_feasible,
                "basin_n_total": basin_total,
                "basin_f_share": float(shares[node_id]),
            }
        )
    return records


def colon_escape_edge_records(result: ColonResult) -> list[dict[str, Any]]:
    """Return one exact-shell record per supported CoLON escape edge."""
    records: list[dict[str, Any]] = []
    for (source, target, shell), count in sorted(result.edge_support_count.items()):
        source_rank = float(result.node_rank[source])
        target_rank = float(result.node_rank[target])
        relation = (
            "self"
            if source == target
            else "better"
            if target_rank < source_rank
            else "equal"
            if target_rank == source_rank
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
                "source_shell_candidate_feasible_count": int(
                    result.shell_feasible_candidate_count[(source, shell)]
                ),
                "source_shell_candidate_infeasible_count": int(
                    result.shell_infeasible_candidate_count[(source, shell)]
                ),
                "source_node_rank": source_rank,
                "target_node_rank": target_rank,
                "relation": relation,
            }
        )
    return records


def overlap_records(result: ColonResult) -> list[dict[str, int]]:
    """Return exact pairwise basin-overlap counts."""
    return [
        {"node_a": left, "node_b": right, "design_count": int(count)}
        for (left, right), count in sorted(result.overlaps.items())
    ]


def _reject_mixed_feasibility_plateaus(
    graph: PlateauGraph,
    feasible: NDArray[np.bool_],
) -> None:
    n_plateaus = graph.n_plateaus
    feasible_count = np.bincount(
        graph.plateau_of_linear,
        weights=feasible.astype(np.int64, copy=False),
        minlength=n_plateaus,
    )
    mixed = (feasible_count > 0) & (feasible_count < graph.plateau_size)
    if bool(np.any(mixed)):
        raise ElaError("CoLON plateau mixes analysis-local F and not-F")


def _support_count_for_nodes(
    graph: PlateauGraph,
    support_single: NDArray[np.int32],
    support_multiple: Mapping[int, tuple[int, ...]],
    selected: NDArray[np.bool_],
) -> int:
    singleton_mask = support_single >= 0
    count = int(
        np.sum(
            graph.plateau_size[
                singleton_mask & selected[support_single.clip(min=0)]
            ]
        )
    )
    for pid, nodes in support_multiple.items():
        if any(bool(selected[node_id]) for node_id in nodes):
            count += int(graph.plateau_size[pid])
    return count


def _validate_colon_result(result: ColonResult) -> None:
    """Fail closed on support, escape-shell, F-split, and funnel invariants."""
    n = int(result.space.cardinality)
    n_nodes = result.n_nodes
    if int(sum(result.terminal_multiplicity_counts.values())) != n:
        raise ElaError("terminal-support multiplicity counts do not cover catalogue")
    node_support = result.support_single[result.graph.node_plateaus]
    expected = np.arange(n_nodes, dtype=np.int32)
    if not bool(np.array_equal(node_support, expected)):
        raise ElaError("at least one CoLON node does not support itself")
    if bool(np.any(result.support_size < result.node_size)):
        raise ElaError("a node support size is smaller than its node size")
    if (
        bool(np.any(result.node_n_feasible < 0))
        or bool(np.any(result.node_n_feasible > result.node_size))
    ):
        raise ElaError("node feasibility counts are inconsistent with node sizes")
    if (
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
        if set(result.shell_feasible_candidate_count) != expected_shells:
            raise ElaError("feasible candidate counts are incomplete")
        if set(result.shell_infeasible_candidate_count) != expected_shells:
            raise ElaError("infeasible candidate counts are incomplete")
        for key, improving in result.shell_improving_candidate_count.items():
            denominator = result.shell_candidate_count.get(key)
            if denominator is None or improving < 0 or improving > denominator:
                raise ElaError("improving candidate count exceeds shell denominator")
        for key, n_feasible in result.shell_feasible_candidate_count.items():
            n_infeasible = result.shell_infeasible_candidate_count.get(key)
            denominator = result.shell_candidate_count.get(key)
            if (
                n_infeasible is None
                or denominator is None
                or n_feasible < 0
                or n_infeasible < 0
                or n_feasible + n_infeasible != denominator
            ):
                raise ElaError(
                    "perturbed-design F split does not partition the shell"
                )

    global_nodes = np.flatnonzero(result.node_rank == float(np.min(result.node_rank)))
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


def _feature_rows(
    result: ColonResult,
    *,
    problem_id: str,
    source_dir: Path,
    constraint_names: Sequence[str],
    source_type: str,
    sampling_design_signature: str | None,
    seed: int | None,
) -> list[FeatureResult]:
    n_designs = int(result.space.cardinality)
    n_nodes = result.n_nodes
    node_sizes = result.node_size.astype(np.float64)
    support_sizes = result.support_size.astype(np.float64)
    shares = result.basin_f_share
    has_f = result.y_star is not None
    trap = ~result.terminal_feasible
    mixed = result.terminal_feasible & (shares < 1.0)
    pure = result.terminal_feasible & (shares == 1.0)
    signature = analysis_signature(
        family=FAMILY,
        landscape=LANDSCAPE_DEB,
        response_view=RESPONSE_VIEW_RAW,
        x_view=X_VIEW_INDEX,
        constraint_names=tuple(constraint_names),
        config={
            "engine": "nested_best_exit_deb",
            "stages": [STAGE_BASIN_CENSUS, STAGE_ESCAPES, STAGE_FUNNELS],
            "radii": list(result.radii),
        },
    )
    identity: dict[str, Any] = {
        "problem_id": problem_id,
        "source_type": source_type,
        "source_dir": source_dir,
        "n_input": n_designs,
        "n_unique": n_designs,
        "runtime_s": result.runtime_s,
        "config_payload": {
            "engine": "nested_best_exit_deb",
            "stages": [STAGE_BASIN_CENSUS, STAGE_ESCAPES, STAGE_FUNNELS],
            "radii": list(result.radii),
        },
        "sampling_design_signature": sampling_design_signature,
        "analysis_signature": signature,
        "seed": seed,
    }
    global_share = (
        float(result.global_support_count) / float(n_designs)
        if has_f and result.global_support_count is not None and n_designs
        else None
    )
    multiplicity = result.terminal_multiplicity_counts
    mean_mult = (
        sum(int(k) * int(v) for k, v in multiplicity.items()) / float(n_designs)
        if n_designs
        else None
    )
    rows = [
        _row(identity, FEATURE_N_NODES, float(n_nodes)),
        _row(
            identity,
            FEATURE_N_GLOBAL_NODES,
            float(result.n_global_nodes) if has_f else None,
        ),
        _row(identity, FEATURE_NODE_SIZE_MIN, float(np.min(node_sizes))),
        _row(identity, FEATURE_NODE_SIZE_MEDIAN, float(np.median(node_sizes))),
        _row(identity, FEATURE_NODE_SIZE_MAX, float(np.max(node_sizes))),
        _row(
            identity,
            FEATURE_LARGEST_NODE_SHARE,
            float(np.max(node_sizes)) / float(n_designs) if n_designs else None,
        ),
        _row(
            identity,
            FEATURE_N_SINGLETON_NODES,
            float(np.count_nonzero(result.node_size == 1)),
        ),
        _row(identity, FEATURE_SUPPORT_SIZE_MIN, float(np.min(support_sizes))),
        _row(identity, FEATURE_SUPPORT_SIZE_MEDIAN, float(np.median(support_sizes))),
        _row(identity, FEATURE_SUPPORT_SIZE_MAX, float(np.max(support_sizes))),
        _row(
            identity,
            FEATURE_LARGEST_SUPPORT_SHARE,
            float(np.max(support_sizes)) / float(n_designs) if n_designs else None,
        ),
        _row(identity, FEATURE_GLOBAL_SUPPORT_SHARE, global_share),
        _row(
            identity,
            FEATURE_AMBIGUOUS_DESIGN_COUNT,
            float(result.ambiguous_design_count),
        ),
        _row(
            identity,
            FEATURE_AMBIGUOUS_DESIGN_SHARE,
            float(result.ambiguous_design_count) / float(n_designs)
            if n_designs
            else None,
        ),
        _row(identity, FEATURE_TERMINAL_MULTIPLICITY_MEAN, mean_mult),
        _row(
            identity,
            FEATURE_TERMINAL_MULTIPLICITY_MEDIAN,
            _weighted_median(multiplicity),
        ),
        _row(
            identity,
            FEATURE_TERMINAL_MULTIPLICITY_MAX,
            float(max(multiplicity)) if multiplicity else None,
        ),
        _row(identity, FEATURE_BASIN_F_SHARE_MIN, float(np.min(shares))),
        _row(identity, FEATURE_BASIN_F_SHARE_MEDIAN, float(np.median(shares))),
        _row(identity, FEATURE_BASIN_F_SHARE_MAX, float(np.max(shares))),
        _row(
            identity,
            FEATURE_N_TRAP_NODES,
            float(np.count_nonzero(trap)),
            role=ROLE_SECONDARY,
        ),
        _row(
            identity,
            FEATURE_N_MIXED_BASINS,
            float(np.count_nonzero(mixed)),
            role=ROLE_SECONDARY,
        ),
        _row(
            identity,
            FEATURE_N_PURE_FEASIBLE_BASINS,
            float(np.count_nonzero(pure)),
            role=ROLE_SECONDARY,
        ),
        _row(identity, FEATURE_Y_STAR, result.y_star),
    ]
    for radius in result.radii:
        prefix_escape = f"colon.escape.d{radius}"
        prefix_funnel = f"colon.funnel.d{radius}"
        cumulative_pairs = {
            (source, target)
            for (source, target, shell), count in result.edge_support_count.items()
            if shell <= radius and source != target and count > 0
        }
        escapable = {source for source, _target in cumulative_pairs}
        improving = {
            source
            for source, target in cumulative_pairs
            if float(result.node_rank[target]) < float(result.node_rank[source])
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
        feasible_candidates = sum(
            count
            for (source, shell), count in result.shell_feasible_candidate_count.items()
            if shell <= radius
        )
        infeasible_candidates = sum(
            count
            for (source, shell), count in result.shell_infeasible_candidate_count.items()
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
                _row(
                    identity,
                    f"{prefix_escape}.n_edges",
                    float(len(cumulative_pairs)),
                    role=ROLE_SECONDARY,
                ),
                _row(
                    identity,
                    f"{prefix_escape}.edge_density",
                    edge_density,
                    role=ROLE_SECONDARY,
                ),
                _row(
                    identity,
                    f"{prefix_escape}.escapable_node_share",
                    float(len(escapable)) / float(n_nodes) if n_nodes else None,
                    role=ROLE_SECONDARY,
                ),
                _row(
                    identity,
                    f"{prefix_escape}.improving_escapable_node_share",
                    float(len(improving)) / float(n_nodes) if n_nodes else None,
                ),
                _row(
                    identity,
                    f"{prefix_escape}.source_retained_candidate_share",
                    float(retained) / float(denominator) if denominator else None,
                    role=ROLE_SECONDARY,
                ),
                _row(
                    identity,
                    f"{prefix_escape}.improving_candidate_share",
                    float(improving_candidates) / float(denominator)
                    if denominator
                    else None,
                    role=ROLE_SECONDARY,
                ),
                _row(
                    identity,
                    f"{prefix_escape}.escape_candidate_feasible",
                    float(feasible_candidates),
                    role=ROLE_DIAGNOSTIC,
                ),
                _row(
                    identity,
                    f"{prefix_escape}.escape_candidate_infeasible",
                    float(infeasible_candidates),
                    role=ROLE_DIAGNOSTIC,
                ),
            ]
        )
        funnel = result.funnels[radius]
        rows.extend(
            [
                _row(identity, f"{prefix_funnel}.n_funnels", float(funnel.n_funnels)),
                _row(
                    identity,
                    f"{prefix_funnel}.n_global_sinks",
                    float(funnel.n_global_sinks),
                    role=ROLE_SECONDARY,
                ),
                _row(identity, f"{prefix_funnel}.global_node_share", funnel.global_node_share),
                _row(
                    identity,
                    f"{prefix_funnel}.global_design_share",
                    funnel.global_design_share,
                ),
                _row(
                    identity,
                    f"{prefix_funnel}.ambiguous_node_share",
                    funnel.ambiguous_node_share,
                    role=ROLE_SECONDARY,
                ),
                _row(
                    identity,
                    f"{prefix_funnel}.largest_nonglobal_node_share",
                    funnel.largest_nonglobal_node_share,
                ),
                _row(
                    identity,
                    f"{prefix_funnel}.largest_nonglobal_design_share",
                    funnel.largest_nonglobal_design_share,
                ),
                _row(
                    identity,
                    f"{prefix_funnel}.nonglobal_hops_to_global_min",
                    funnel.nonglobal_hops_to_global_min,
                    role=ROLE_SECONDARY,
                ),
                _row(
                    identity,
                    f"{prefix_funnel}.nonglobal_hops_to_global_median",
                    funnel.nonglobal_hops_to_global_median,
                    role=ROLE_SECONDARY,
                ),
                _row(
                    identity,
                    f"{prefix_funnel}.nonglobal_hops_to_global_max",
                    funnel.nonglobal_hops_to_global_max,
                    role=ROLE_SECONDARY,
                ),
            ]
        )
    stage_rows = [
        _row(
            identity,
            FEATURE_STAGE_BASIN_CENSUS,
            1.0,
            role=ROLE_DIAGNOSTIC,
        ),
    ]
    if result.radii:
        stage_rows.extend(
            [
                _row(
                    identity,
                    FEATURE_STAGE_ESCAPES,
                    1.0,
                    role=ROLE_DIAGNOSTIC,
                ),
                _row(
                    identity,
                    FEATURE_STAGE_FUNNELS,
                    1.0,
                    role=ROLE_DIAGNOSTIC,
                ),
            ]
        )
    rows.extend(
        [
            *stage_rows,
            _row(
                identity,
                "colon.support_relation_rows",
                float(
                    int(np.count_nonzero(result.support_single >= 0))
                    + sum(len(nodes) for nodes in result.support_multiple.values())
                ),
                role=ROLE_DIAGNOSTIC,
            ),
            _row(
                identity,
                "colon.support_multiple_plateaus",
                float(len(result.support_multiple)),
                role=ROLE_DIAGNOSTIC,
            ),
            _row(
                identity,
                "colon.overlap_pairs",
                float(len(result.overlaps)),
                role=ROLE_DIAGNOSTIC,
            ),
            _row(identity, FEATURE_RUNTIME, result.runtime_s, role=ROLE_DIAGNOSTIC),
        ]
    )
    for stage, runtime_s in result.stage_runtime_s.items():
        rows.append(
            _row(
                identity,
                f"colon.costs_runtime.{stage}",
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
    if feature_name in GLOBAL_FEATURE_NAMES and value is None:
        status = STATUS_UNDEFINED
    elif value is None:
        status = STATUS_UNDEFINED
    else:
        status = STATUS_OK
    return feature_result(
        problem_id=identity["problem_id"],
        source_type=identity["source_type"],
        source_dir=identity["source_dir"],
        landscape=LANDSCAPE_DEB,
        response_view=RESPONSE_VIEW_RAW,
        x_view=X_VIEW_INDEX,
        family=FAMILY,
        feature_name=feature_name,
        status=status,
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


def _library_version() -> str:
    try:
        return importlib.metadata.version("hpdesignbench")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
