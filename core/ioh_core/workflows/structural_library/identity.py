"""
Build sealed LibraryIdentity values for structural-library experiments.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from core.analysis_core.run_config import AnalysisRunConfig
from core.ioh_core.structural_artifacts.indexing import (
    finish_group_size,
    finish_variable_names,
    structural_cardinality,
    structural_space,
    structural_variable_names,
)
from core.ioh_core.structural_artifacts.models import (
    LibraryIdentity,
    run_config_fingerprint,
)
from core.ioh_core.workflows.brute_force.models import IndexSpace


def build_library_identity(
    *,
    slab_type: str,
    problem_id: str,
    space: IndexSpace,
    snapshot_hash: str,
    run_config: AnalysisRunConfig,
    library_structural_cardinality: int,
    input_hashes: Mapping[str, str] | None = None,
    compatible_problem_ids: Sequence[str] | None = None,
) -> LibraryIdentity:
    """
    Construct the sealed identity for one library build.
    """
    struct = structural_space(space)
    finish_names = finish_variable_names(space)
    primary = str(problem_id)
    if compatible_problem_ids is None:
        compatible: tuple[str, ...] = (primary,)
    else:
        compatible = tuple(str(item) for item in compatible_problem_ids)
    return LibraryIdentity(
        slab_type=str(slab_type),
        problem_id=primary,
        snapshot_hash=str(snapshot_hash),
        run_config_fingerprint=run_config_fingerprint(run_config),
        structural_cardinality=int(library_structural_cardinality),
        full_cardinality=int(space.cardinality),
        finish_names=finish_names,
        finish_sizes=(int(space.sizes[-2]), int(space.sizes[-1])),
        structural_names=structural_variable_names(space),
        structural_sizes=tuple(int(v) for v in struct.sizes),
        input_hashes=tuple(
            (str(k), str(input_hashes[k]))
            for k in sorted((input_hashes or {}).keys(), key=str)
        ),
        compatible_problem_ids=compatible,
    )


def require_hp_slab(slab_type: str) -> None:
    """
    Reject unsupported slab types for structural libraries.
    """
    if str(slab_type) != "hp_slab":
        raise ValueError(
            "structural_library currently supports slab_type='hp_slab' only, "
            f"got {slab_type!r}"
        )


def assert_finish_group_consistent(space: IndexSpace) -> int:
    """
    Return finish-group size after validating finish-last ordering.
    """
    group = finish_group_size(space)
    if space.cardinality % group != 0:
        raise ValueError(
            "candidate cardinality is not divisible by finish_group_size "
            f"({space.cardinality} % {group} != 0)"
        )
    full_struct = structural_cardinality(space)
    if full_struct * group != space.cardinality:
        raise ValueError(
            "structural_cardinality * finish_group_size != candidate cardinality"
        )
    return group
