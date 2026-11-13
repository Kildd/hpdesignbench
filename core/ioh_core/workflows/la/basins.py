"""
Catalogue local-search basins on stored ``y_p``.

Paper name: catalogue local-search basin. Preferred CLI:
``hdb-la-analyze basins`` (``family=basins``). Output ``la/basins/``.
Keys ``basins.*``. Not a ``topology.*`` family.

Complete brute-force catalogues only. Sampling is refused. Neighbourhood
is L1=1 on ``X_index`` (no wrap, no diagonal, no jump>1), same helper
as GBLT. Ranking is stored evaluation-penalty ``y_p`` on the whole box
(feasible and infeasible). Search does not stop at analysis-local
``F``: a feasible point may move to an infeasible neighbour if that
neighbour has strictly smaller ``y_p``. Not lexicographic (φ then y).
Not a GBLT F-component, not a nearest-component halo, not pflacco
``number_of_peaks``.

**Plateau.** An L1-connected set of designs whose stored ``y_p`` values
are IEEE-equal (``==``). No ``isclose``. Non-finite ``y_p`` fails closed.

**Core** (local-min plateau). A plateau with no L1 neighbour of
strictly smaller ``y_p``.

**Nested best exit.** From a plateau, take every improving L1 neighbour
at the unique best (smaller) ``y_p``; expand those equal-``y_p``
plateaus; repeat. One-step greedy: a worse first height is never taken,
even if that path would reach a better terminal later. Never pick one
tied plateau by linear id. From one start the height sequence is unique.

**Co-reach.** Several cores are co-reached from a start when they all
sit at the last best height of that start's nested-best-exit sequence
and all touch the current frontier. Same terminal ``y_p`` in different
regions is not co-reach.

**Basin identity (co-reach transitivity).** A basin is a connected
component of the co-reach graph on cores: union-find over cores that
ever appear together as co-reached termini of at least one start. If
start P co-reaches {A, B} and start Q co-reaches {B, C}, then A, B, and
C are one basin even though no start co-reaches {A, C}. Two cores that
never co-occur, including two cores at the same ``y_p`` or at
``y_p*``, are two basins. Do not merge cores only because they share a
height. Do not use exact attractor-set frozensets as basin ids (that
would count {A,B} and {B,C} as two basins).

**y_p*.** Global minimum of stored ``y_p`` on the whole catalogue
(feasible and infeasible). Feature/code name ``y_p_star``. Not GBLT
``y*`` / ``topology.y_star``. Several basins may have ``is_y_p_star``
if their cores sit at ``y_p*`` and never co-reach.

Canonical SUCCESS is ``features.parquet`` only. Optional
``--write-attractors`` writes one row per core (size, ``y_p``,
feasible, ``is_y_p_star``, basin key). Off by default.

**Related work (basin scalars, not the full LON graph).** The
comparable exhaustive method is local optima networks (LON): nodes are
local optima (or local-optimum *neutral networks* when fitness ties
exist); a basin is the set of configurations a hill-climber maps to that
node (Ochoa, Tomassini, Vérel & Darabos, GECCO 2008,
doi:10.1145/1389095.1389204; Vérel, Ochoa & Tomassini, IEEE TEVC 2010,
doi:10.1109/tevc.2010.2046175; Ochoa, Vérel, Daolio & Tomassini, 2013
chapter, doi:10.1007/978-3-642-41888-4_9 / arXiv:1402.2959). Those
papers also add *edges* (basin adjacency, transition probabilities,
escape edges). This family emits the **partition** under nested
best-exit (the basin-side of a LON), not those edges. A full LON is
fitness-landscape analysis, not a pflacco ELA family; it was not
rejected, only the graph was left out of v1 SUCCESS. Ochoa, Vérel &
Tomassini (PPSN 2010, doi:10.1007/978-3-642-15844-5_11) compare
first-improvement vs best-improvement LONs; nested best-exit is a
best-improvement map. Vérel et al. (2010) resolve plateau ties with
*stochastic* local search; this map uses nested best-exit plus co-reach
instead. A core here is their local-optimum neutral network.
``n_basins`` (co-reach components) is *not* the LON node count: LON
keeps one node per local opt / LONN. ``n_cores`` is the LON analogue of
node count.

Reported basin-side statistics in that literature, mapped to this
family's SUCCESS keys:

- number of local optima / LON nodes → ``basins.n_cores``
- basin-size distribution (min / typical / max, largest share) →
  ``basins.size_min`` / ``size_median`` / ``size_max`` /
  ``largest_share``
- relative size of the global-optimum basin (Ochoa et al. 2008 §4.4.1)
  → ``basins.share_at_y_p_star``
- how many attractors sit at the global height →
  ``basins.n_basins_at_y_p_star``
- diagnostic global height → ``basins.y_p_star`` (not a LON feature)

Not in v1 SUCCESS unless separately locked: clustering / path length /
degree (need LON edges), funnel count (needs those edges), first- vs
best-improvement comparison (second operator).
``n_singleton_cores`` is core cardinality 1, not LON isolated vertices
(degree 0). ``fitness_basin_size_spearman`` is Spearman of each basin's
terminal ``y_p`` vs its design-count; ``undefined`` if fewer than two
basins. A later optional LON graph (basin-transition edges, funnel
count) is not v1. This family is not a sampled ILS-LON and not a
drop-in of a LON library.

See ``CONTEXT.md`` for domain terms. Importing this module must not
load DuckDB, pflacco, NetworkX, or SALib.
"""
from __future__ import annotations

import importlib.metadata
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.ioh_core.workflows.brute_force.models import IndexSpace
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
from core.ioh_core.workflows.la.catalogue_table import load_catalogue_y_p
from core.ioh_core.workflows.la.plateau_graph import build_plateau_graph
from core.ioh_core.workflows.la.signatures import analysis_signature

FAMILY = "basins"
LIBRARY = "hpdesignbench"
FEATURE_N_BASINS = "basins.n_basins"
FEATURE_N_CORES = "basins.n_cores"
FEATURE_SIZE_MIN = "basins.size_min"
FEATURE_SIZE_MEDIAN = "basins.size_median"
FEATURE_SIZE_MAX = "basins.size_max"
FEATURE_LARGEST_SHARE = "basins.largest_share"
FEATURE_N_SINGLETON_CORES = "basins.n_singleton_cores"
FEATURE_N_BASINS_AT_Y_P_STAR = "basins.n_basins_at_y_p_star"
FEATURE_SHARE_AT_Y_P_STAR = "basins.share_at_y_p_star"
FEATURE_Y_P_STAR = "basins.y_p_star"
FEATURE_FITNESS_BASIN_SIZE_SPEARMAN = "basins.fitness_basin_size_spearman"
FEATURE_RUNTIME = "basins.costs_runtime"


@dataclass(frozen=True)
class BasinPartition:
    """Catalogue partition under nested best-exit / co-reach."""

    basin_of_linear: NDArray[np.int32]
    plateau_of_linear: NDArray[np.int32]
    core_of_linear: NDArray[np.int32]
    basin_sizes: NDArray[np.int64]
    basin_y_p: NDArray[np.float64]
    core_size: NDArray[np.int64]
    core_y_p: NDArray[np.float64]
    core_basin_id: NDArray[np.int32]
    core_min_linear: NDArray[np.int64]
    core_feasible: NDArray[np.bool_]
    n_basins: int
    n_cores: int
    y_p_star: float


def ela_basins(
    *,
    space: IndexSpace,
    y_p: ArrayLike,
    feasible: ArrayLike | None = None,
    problem_id: str,
    source_dir: Path,
    constraint_names: Sequence[str],
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
    seed: int | None = None,
) -> list[FeatureResult]:
    """
    Compute tidy basin aggregates on an in-memory catalogue.

    Parameters
    ----------
    space :
        Complete catalogue index space.
    y_p :
        Stored evaluation-penalty, length ``space.cardinality``, linear order.
    feasible :
        Optional stored feasible flags, same length. Unused for SUCCESS
        scalars; used when building the attractor table.
    constraint_names :
        Keep-list identity for ``analysis_signature``. Ranking still uses
        stored ``y_p``; the keep-list is provenance, not a filter.
    """
    started = time.monotonic()
    partition = ela_basin_partition(space=space, y_p=y_p, feasible=feasible)
    runtime_s = max(0.0, time.monotonic() - started)
    return _feature_rows(
        partition,
        problem_id=problem_id,
        source_dir=source_dir,
        source_type=source_type,
        runtime_s=runtime_s,
        n_input=int(space.cardinality),
        n_unique=int(space.cardinality),
        sampling_design_signature=sampling_design_signature,
        seed=seed,
        constraint_names=constraint_names,
    )


def ela_basin_partition(
    *,
    space: IndexSpace,
    y_p: ArrayLike,
    feasible: ArrayLike | None = None,
    source_type: str = SOURCE_BRUTE_FORCE,
) -> BasinPartition:
    """Partition the catalogue into co-reach basins on stored ``y_p``."""
    if str(source_type) != SOURCE_BRUTE_FORCE:
        raise ElaError(
            "catalogue local-search basins require source_type='brute_force', "
            f"got {source_type!r}"
        )
    values = np.asarray(y_p, dtype=np.float64).reshape(-1)
    n = int(space.cardinality)
    if values.size != n:
        raise ElaError(
            "y_p length "
            f"({int(values.size)}) does not match cardinality ({n})"
        )
    if not np.all(np.isfinite(values)):
        raise ElaError("non-finite y_p in landscape source; reject whole source")
    flags: NDArray[np.bool_] | None
    if feasible is None:
        flags = None
    else:
        flags = np.asarray(feasible, dtype=np.bool_).reshape(-1)
        if flags.size != n:
            raise ElaError(
                "feasible length "
                f"({int(flags.size)}) does not match cardinality ({n})"
            )

    graph = build_plateau_graph(values, space)
    plateau = graph.plateau_of_linear
    n_plateaus = graph.n_plateaus
    plateau_size = graph.plateau_size
    plateau_min_linear = graph.plateau_min_linear
    yp_plateau = graph.plateau_y_p
    best_h = graph.best_exit_y_p
    dest_offsets = graph.destination_offsets
    dest_ids = graph.destination_ids
    core_plateaus = graph.node_plateaus
    plateau_to_core = graph.plateau_to_node
    n_cores = graph.n_nodes

    core_parent = np.arange(n_cores, dtype=np.int32)
    core_rank = np.zeros(n_cores, dtype=np.int8)

    def core_find(index: int) -> int:
        root = index
        while int(core_parent[root]) != root:
            root = int(core_parent[root])
        while int(core_parent[index]) != index:
            nxt = int(core_parent[index])
            core_parent[index] = root
            index = nxt
        return root

    def core_union(left: int, right: int) -> None:
        root_left = core_find(left)
        root_right = core_find(right)
        if root_left == root_right:
            return
        if core_rank[root_left] < core_rank[root_right]:
            core_parent[root_left] = root_right
        elif core_rank[root_left] > core_rank[root_right]:
            core_parent[root_right] = root_left
        else:
            core_parent[root_right] = root_left
            core_rank[root_left] += 1

    # Joint nested best-exit on the plateau DAG (not independent
    # terminals of first-hop dests). Destinations have strictly smaller
    # y_p, so the walk cannot cycle.
    rep_core = np.full(n_plateaus, -1, dtype=np.int32)
    for pid_i in range(n_plateaus):
        current = {pid_i}
        while True:
            best: float | None = None
            for plateau_id in current:
                height = float(best_h[plateau_id])
                if not math.isfinite(height):
                    continue
                if best is None or height < best:
                    best = height
            if best is None:
                break
            nxt: set[int] = set()
            for plateau_id in current:
                if float(best_h[plateau_id]) != best:
                    continue
                start = int(dest_offsets[plateau_id])
                stop = int(dest_offsets[plateau_id + 1])
                nxt.update(int(value) for value in dest_ids[start:stop])
            current = nxt
        core_ids = [int(plateau_to_core[plateau_id]) for plateau_id in current]
        if not core_ids or any(cid < 0 for cid in core_ids):
            raise ElaError("nested best-exit terminated on a non-core plateau")
        first = core_ids[0]
        for cid in core_ids[1:]:
            core_union(first, cid)
        rep_core[pid_i] = first

    min_linear_by_root: dict[int, int] = {}
    for cid in range(n_cores):
        root = core_find(cid)
        linear = int(plateau_min_linear[int(core_plateaus[cid])])
        previous = min_linear_by_root.get(root)
        if previous is None or linear < previous:
            min_linear_by_root[root] = linear
    ordered_roots = sorted(min_linear_by_root, key=min_linear_by_root.__getitem__)
    root_to_basin = {root: bid for bid, root in enumerate(ordered_roots)}
    n_basins = len(ordered_roots)
    core_basin_id = np.empty(n_cores, dtype=np.int32)
    for cid in range(n_cores):
        core_basin_id[cid] = root_to_basin[core_find(cid)]

    plateau_basin = np.empty(n_plateaus, dtype=np.int32)
    for pid in range(n_plateaus):
        plateau_basin[pid] = int(core_basin_id[core_find(int(rep_core[pid]))])

    basin_of_linear = plateau_basin[plateau]
    core_of_linear = plateau_to_core[plateau]
    basin_sizes = np.bincount(basin_of_linear, minlength=n_basins).astype(np.int64)
    core_size = plateau_size[core_plateaus]
    core_y_p = yp_plateau[core_plateaus]
    basin_y_p = np.full(n_basins, np.inf, dtype=np.float64)
    np.minimum.at(basin_y_p, core_basin_id, core_y_p)
    y_p_star = float(np.min(values))
    if n and not bool(np.any(basin_y_p == y_p_star)):
        raise ElaError("no catalogue basin terminates at y_p_star")
    core_min_linear = plateau_min_linear[core_plateaus]
    if flags is None:
        core_feasible = np.zeros(n_cores, dtype=np.bool_)
    else:
        in_core = core_of_linear >= 0
        any_infeasible = np.zeros(n_cores, dtype=np.bool_)
        np.logical_or.at(
            any_infeasible,
            core_of_linear[in_core],
            np.logical_not(flags[in_core]),
        )
        core_feasible = np.logical_not(any_infeasible)
    basin_of_linear.setflags(write=False)
    plateau.setflags(write=False)
    core_of_linear.setflags(write=False)
    basin_sizes.setflags(write=False)
    basin_y_p.setflags(write=False)
    core_size.setflags(write=False)
    core_y_p.setflags(write=False)
    core_basin_id.setflags(write=False)
    core_min_linear.setflags(write=False)
    core_feasible.setflags(write=False)
    return BasinPartition(
        basin_of_linear=basin_of_linear,
        plateau_of_linear=plateau,
        core_of_linear=core_of_linear,
        basin_sizes=basin_sizes,
        basin_y_p=basin_y_p,
        core_size=core_size,
        core_y_p=core_y_p,
        core_basin_id=core_basin_id,
        core_min_linear=core_min_linear,
        core_feasible=core_feasible,
        n_basins=n_basins,
        n_cores=n_cores,
        y_p_star=y_p_star,
    )


def attractor_records(partition: BasinPartition) -> list[dict[str, Any]]:
    """One row per core for the optional attractor artefact."""
    rows: list[dict[str, Any]] = []
    for cid in range(partition.n_cores):
        y_p = float(partition.core_y_p[cid])
        rows.append(
            {
                "core_id": cid,
                "basin_id": int(partition.core_basin_id[cid]),
                "size": int(partition.core_size[cid]),
                "y_p": y_p,
                "feasible": bool(partition.core_feasible[cid]),
                "is_y_p_star": bool(y_p == partition.y_p_star),
                "min_linear": int(partition.core_min_linear[cid]),
            }
        )
    rows.sort(key=lambda item: (item["basin_id"], item["min_linear"]))
    return rows


def ela_basins_from_table(
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
    sampling_design_signature: str | None = None,
    seed: int | None = None,
) -> tuple[list[FeatureResult], BasinPartition]:
    """Load ``y_p`` from a complete DuckDB catalogue view and emit features."""
    started = time.monotonic()
    space = IndexSpace.from_bounds(var_names, lower_bounds, upper_bounds)
    if int(space.cardinality) != int(cardinality):
        raise ElaError(
            "IndexSpace cardinality "
            f"({space.cardinality}) does not match schema ({cardinality})"
        )
    y_p, feasible = load_catalogue_y_p(
        connection, table, space, expected_n=int(cardinality)
    )
    partition = ela_basin_partition(
        space=space, y_p=y_p, feasible=feasible, source_type=source_type
    )
    runtime_s = max(0.0, time.monotonic() - started)
    rows = _feature_rows(
        partition,
        problem_id=problem_id,
        source_dir=source_dir,
        source_type=source_type,
        runtime_s=runtime_s,
        n_input=int(cardinality),
        n_unique=int(cardinality),
        sampling_design_signature=sampling_design_signature,
        seed=seed,
        constraint_names=constraint_names,
    )
    return rows, partition


def _feature_rows(
    partition: BasinPartition,
    *,
    problem_id: str,
    source_dir: Path,
    source_type: str,
    runtime_s: float,
    n_input: int,
    n_unique: int,
    sampling_design_signature: str | None,
    seed: int | None,
    constraint_names: Sequence[str],
) -> list[FeatureResult]:
    n_basins = partition.n_basins
    sizes = partition.basin_sizes
    y_p_star = partition.y_p_star
    at_star = partition.basin_y_p == y_p_star
    n_at_star = int(np.count_nonzero(at_star))
    share_star = (
        float(np.sum(sizes[at_star])) / float(n_input) if n_input else None
    )
    n_singleton = int(np.count_nonzero(partition.core_size == 1))
    spearman_value, spearman_ok = _spearman(partition.basin_y_p, sizes)
    names = tuple(constraint_names)
    sig = analysis_signature(
        family=FAMILY,
        landscape=LANDSCAPE_Y_P,
        constraint_names=names,
        x_view=X_VIEW_INDEX,
        response_view=RESPONSE_VIEW_RAW,
        config={},
    )
    identity = {
        "problem_id": problem_id,
        "source_type": source_type,
        "source_dir": Path(source_dir),
        "n_input": n_input,
        "n_unique": n_unique,
        "runtime_s": runtime_s,
        "config_payload": {},
        "sampling_design_signature": sampling_design_signature,
        "seed": seed,
        "analysis_signature": sig,
    }
    size_min = int(sizes.min()) if n_basins else None
    size_max = int(sizes.max()) if n_basins else None
    size_median = float(np.median(sizes)) if n_basins else None
    largest_share = (
        float(size_max) / float(n_input) if n_basins and n_input else None
    )
    return [
        _row(identity, FEATURE_N_BASINS, float(n_basins)),
        _row(identity, FEATURE_N_CORES, float(partition.n_cores)),
        _row(identity, FEATURE_SIZE_MIN, _optional_float(size_min)),
        _row(identity, FEATURE_SIZE_MEDIAN, _optional_float(size_median)),
        _row(identity, FEATURE_SIZE_MAX, _optional_float(size_max)),
        _row(identity, FEATURE_LARGEST_SHARE, _optional_float(largest_share)),
        _row(identity, FEATURE_N_SINGLETON_CORES, float(n_singleton)),
        _row(identity, FEATURE_N_BASINS_AT_Y_P_STAR, float(n_at_star)),
        _row(identity, FEATURE_SHARE_AT_Y_P_STAR, _optional_float(share_star)),
        _row(
            identity,
            FEATURE_Y_P_STAR,
            y_p_star,
            role=ROLE_DIAGNOSTIC,
        ),
        _row(
            identity,
            FEATURE_FITNESS_BASIN_SIZE_SPEARMAN,
            spearman_value if spearman_ok else None,
            role=ROLE_SECONDARY,
        ),
        _row(
            identity,
            FEATURE_RUNTIME,
            runtime_s,
            role=ROLE_DIAGNOSTIC,
        ),
    ]


def _spearman(
    y_p: NDArray[np.float64], sizes: NDArray[np.int64]
) -> tuple[float | None, bool]:
    from scipy.stats import spearmanr

    if y_p.size < 2:
        return None, False
    result = spearmanr(y_p, sizes)
    value = float(result.statistic)
    if not math.isfinite(value):
        return None, False
    return value, True


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


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
        landscape=LANDSCAPE_Y_P,
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


def _library_version() -> str:
    try:
        return importlib.metadata.version("hpdesignbench")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
