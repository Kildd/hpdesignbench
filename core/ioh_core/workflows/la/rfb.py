"""
RFB× discrete random walks on raw catalogue indices.

Walks belong inside ELA, not in ``sampling/``. Starts are uniform over
the complete catalogue. Transitions are in-bounds L1=1 on ``X_index``
(no wrap, no diagonal, no jump>1, not feasibility-directed). Feasibility
is the Phase 15 mask; Parquet is not scanned per step.

``RFB_× = n_crossings / n_transitions``. Empty neighbourhoods raise.
``FAMILY`` matches the CLI token ``rfbx``. Tidy rows are walk-level
scalars only; paths stay on :class:`RfbWalk` for library tests.

Importing this module must not load DuckDB, pflacco, NetworkX, or SALib.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from core.ioh_core.workflows.brute_force.indexing import linear_to_indices
from core.ioh_core.workflows.la.models import (
    ElaError,
    FeatureResult,
    LANDSCAPE_FEASIBILITY,
    RESPONSE_VIEW_NA,
    ROLE_DIAGNOSTIC,
    ROLE_PRIMARY,
    SOURCE_BRUTE_FORCE,
    STATUS_OK,
    X_VIEW_INDEX,
    feature_result,
)
from core.ioh_core.workflows.la.signatures import analysis_signature
from core.ioh_core.workflows.la.topology import FeasibleRegion, ela_feasible_region

FAMILY = "rfbx"
LIBRARY = "hpdesignbench"
FEATURE_RFB_X = "rfbx.rfb_x"
FEATURE_N_CROSSINGS = "rfbx.n_crossings"
START_POLICY_UNIFORM_CATALOGUE = "uniform_catalogue"
START_POLICY_VALUES = frozenset({START_POLICY_UNIFORM_CATALOGUE})
_SEED_TOKEN = "rfbx"
_UINT32_MAX = 0xFFFFFFFF


@dataclass(frozen=True)
class RfbWalkConfig:
    """
    Caller-supplied RFB× walk knobs.

    Attributes
    ----------
    n_walks :
        Number of independent walks. Positive int.
    n_transitions :
        Steps per walk and the RFB× denominator. Positive int.
    seed_base :
        Non-negative int mixed into each walk seed.
    start_policy :
        Must be ``uniform_catalogue``.
    """

    n_walks: int
    n_transitions: int
    seed_base: int
    start_policy: str

    def __post_init__(self) -> None:
        _require_positive_int(self.n_walks, field="n_walks")
        _require_positive_int(self.n_transitions, field="n_transitions")
        _require_seed_base(self.seed_base)
        if self.start_policy not in START_POLICY_VALUES:
            raise ElaError(
                "rfbx start_policy must be "
                f"{START_POLICY_UNIFORM_CATALOGUE!r}, got {self.start_policy!r}"
            )

    def seed_for(self, *, problem_id: str, walk_id: int) -> int:
        """Return the uint32 mix for one ``walk_id`` in ``0 .. n_walks-1``."""
        if (
            not isinstance(walk_id, int)
            or isinstance(walk_id, bool)
            or walk_id < 0
            or walk_id >= self.n_walks
        ):
            raise ElaError(
                f"walk_id must be an int in [0, {self.n_walks}), got {walk_id!r}"
            )
        return derive_rfb_seed(
            seed_base=self.seed_base,
            problem_id=problem_id,
            walk_id=walk_id,
        )


@dataclass(frozen=True)
class RfbWalk:
    """
    One RFB× walk on the complete catalogue.

    Attributes
    ----------
    walk_id :
        Index in ``0 .. n_walks-1``.
    seed :
        Uint32 mix actually passed to ``numpy.random.Generator``.
    start_linear, start_index :
        Uniform catalogue start (may be infeasible).
    n_crossings :
        Times feasibility flipped across the ``n_transitions`` steps.
    rfb_x :
        ``n_crossings / n_transitions``.
    linear_path :
        Mixed-radix addresses, length ``n_transitions + 1``. Read-only.
    """

    walk_id: int
    seed: int
    start_linear: int
    start_index: tuple[int, ...]
    n_crossings: int
    rfb_x: float
    linear_path: NDArray[np.int64]


def ela_rfb_walks(
    *,
    region: FeasibleRegion,
    config: RfbWalkConfig,
    problem_id: str,
) -> tuple[RfbWalk, ...]:
    """
    Run ``n_walks`` independent L1 walks on ``region``.

    Each walk uses its own ``Generator`` from :meth:`RfbWalkConfig.seed_for`.
    Starts are ``U{0..C-1}``. A point with no in-bounds L1 neighbour is a
    hard error. Lookup is ``region.mask``; stored ``feasible`` is not read.
    """
    if config.start_policy != START_POLICY_UNIFORM_CATALOGUE:
        raise ElaError(
            "rfbx start_policy must be "
            f"{START_POLICY_UNIFORM_CATALOGUE!r}, got {config.start_policy!r}"
        )
    space = region.space
    cardinality = space.cardinality
    mask = region.mask
    if mask.shape != (cardinality,):
        raise ElaError(
            "rfbx feasible mask length "
            f"({int(mask.size)}) does not match cardinality ({cardinality})"
        )
    walks: list[RfbWalk] = []
    for walk_id in range(config.n_walks):
        seed = config.seed_for(problem_id=problem_id, walk_id=walk_id)
        rng = np.random.default_rng(seed)
        start_linear = int(rng.integers(0, cardinality))
        start_index = linear_to_indices(start_linear, space)
        point = start_index
        feasible = bool(mask[start_linear])
        path = np.empty(config.n_transitions + 1, dtype=np.int64)
        path[0] = start_linear
        n_crossings = 0
        for step in range(config.n_transitions):
            neighbours = region.catalogue_neighbours(point)
            if not neighbours:
                raise ElaError(
                    "rfbx cannot step: no in-bounds L1 neighbour at "
                    f"{point!r}"
                )
            chosen = neighbours[int(rng.integers(0, len(neighbours)))]
            next_linear = region.linear_of(chosen)
            next_feasible = bool(mask[next_linear])
            if next_feasible != feasible:
                n_crossings += 1
            feasible = next_feasible
            point = chosen
            path[step + 1] = next_linear
        path.setflags(write=False)
        walks.append(
            RfbWalk(
                walk_id=walk_id,
                seed=seed,
                start_linear=start_linear,
                start_index=start_index,
                n_crossings=n_crossings,
                rfb_x=float(n_crossings) / float(config.n_transitions),
                linear_path=path,
            )
        )
    return tuple(walks)


def rfb_config_payload(config: RfbWalkConfig) -> dict[str, Any]:
    """Return the four caller knobs for ``config`` / analysis signature."""
    return {
        "n_transitions": config.n_transitions,
        "n_walks": config.n_walks,
        "seed_base": config.seed_base,
        "start_policy": config.start_policy,
    }


def ela_rfb_features(
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
    config: RfbWalkConfig,
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
) -> list[FeatureResult]:
    """
    Run RFB× walks on a complete BF table and emit walk-level tidy rows.

    Does not dump ``linear_path`` or start coordinates. Min / median / max
    of ``rfb_x`` stay with ``hdb-la-analyze aggregate``.
    """
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
    walks = ela_rfb_walks(region=region, config=config, problem_id=problem_id)
    runtime_s = max(0.0, time.monotonic() - started)
    return ela_rfb_rows(
        walks=walks,
        config=config,
        problem_id=problem_id,
        source_dir=source_dir,
        constraint_names=constraint_names,
        n_input=region.n_input,
        n_unique=region.n_unique,
        runtime_s=runtime_s,
        source_type=source_type,
        sampling_design_signature=sampling_design_signature,
    )


def ela_rfb_rows(
    *,
    walks: Sequence[RfbWalk],
    config: RfbWalkConfig,
    problem_id: str,
    source_dir: Path,
    constraint_names: Sequence[str],
    n_input: int,
    n_unique: int,
    runtime_s: float,
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
) -> list[FeatureResult]:
    """Map each walk to ``rfb_x`` (primary) and ``n_crossings`` (diagnostic)."""
    payload = rfb_config_payload(config)
    names = tuple(constraint_names)
    signature = analysis_signature(
        family=FAMILY,
        landscape=LANDSCAPE_FEASIBILITY,
        constraint_names=names,
        x_view=X_VIEW_INDEX,
        response_view=RESPONSE_VIEW_NA,
        config=payload,
    )
    rows: list[FeatureResult] = []
    for walk in walks:
        identity = {
            "problem_id": problem_id,
            "source_type": source_type,
            "source_dir": Path(source_dir),
            "n_input": n_input,
            "n_unique": n_unique,
            "runtime_s": runtime_s,
            "config_payload": payload,
            "sampling_design_signature": sampling_design_signature,
            "walk_id": walk.walk_id,
            "seed": walk.seed,
        }
        rows.append(
            _rfb_row(
                identity,
                feature_name=FEATURE_RFB_X,
                value=float(walk.rfb_x),
                role=ROLE_PRIMARY,
                analysis_signature=signature,
            )
        )
        rows.append(
            _rfb_row(
                identity,
                feature_name=FEATURE_N_CROSSINGS,
                value=float(walk.n_crossings),
                role=ROLE_DIAGNOSTIC,
                analysis_signature=signature,
            )
        )
    return rows


def _rfb_row(
    identity: Mapping[str, Any],
    *,
    feature_name: str,
    value: float,
    role: str,
    analysis_signature: str,
) -> FeatureResult:
    return feature_result(
        problem_id=identity["problem_id"],
        source_type=identity["source_type"],
        source_dir=identity["source_dir"],
        walk_id=identity["walk_id"],
        landscape=LANDSCAPE_FEASIBILITY,
        response_view=RESPONSE_VIEW_NA,
        x_view=X_VIEW_INDEX,
        family=FAMILY,
        feature_name=feature_name,
        status=STATUS_OK,
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


def derive_rfb_seed(
    *,
    seed_base: int,
    problem_id: str,
    walk_id: int,
) -> int:
    """
    Derive a stable unsigned 32-bit seed via SHA-256.

    The preimage is:

    ``seed_base|ela|rfbx|{problem_id}|{walk_id}``

    Parallel to sampling's SHA-256 uint32 mix, implemented locally so ELA
    does not import ``sampling/``.
    """
    _require_seed_base(seed_base)
    if not isinstance(problem_id, str) or not problem_id:
        raise ElaError(f"problem_id must be a non-empty string, got {problem_id!r}")
    if not isinstance(walk_id, int) or isinstance(walk_id, bool) or walk_id < 0:
        raise ElaError(f"walk_id must be a non-negative int, got {walk_id!r}")
    material = f"{seed_base}|ela|{_SEED_TOKEN}|{problem_id}|{walk_id}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _require_positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ElaError(f"{field} must be a positive int, got {value!r}")
    return value


def _require_seed_base(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ElaError(f"seed_base must be a non-negative int, got {value!r}")
    if value > _UINT32_MAX:
        raise ElaError(f"seed_base must fit in uint32, got {value!r}")
    return value
