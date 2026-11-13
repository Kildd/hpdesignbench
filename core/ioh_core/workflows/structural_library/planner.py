"""
Plan contiguous structural-ID shards for library builds.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from core.analysis_core.run_config import analysis_run_config_from_params
from core.execution_core.config import ExperimentConfig
from core.execution_core.models import TaskRecord, format_task_id
from core.ioh_core.import_specs import make_decode
from core.ioh_core.structural_artifacts import structural_cardinality
from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.brute_force.planner import (
    load_index_space,
    load_problem_space_and_schema_from_csv_dir,
)
from core.ioh_core.workflows.structural_library.identity import (
    assert_finish_group_consistent,
    require_hp_slab,
)
from core.ioh_core.workflows.structural_library.models import (
    StructuralLibraryShardSpec,
)

STRUCTURAL_LIBRARY_TASK_KIND = "structural_library_shard"


class StructuralLibraryPlanError(ValueError):
    """
    Raised when a structural-library experiment cannot be planned.
    """


def combined_snapshot_hash(input_files: dict[str, str] | None = None) -> str:
    """
    Placeholder combined hash; workers recompute from the sealed manifest.
    """
    payload = "structural_library".encode("utf-8")
    if input_files:
        for key in sorted(input_files):
            payload += f"{key}:{input_files[key]};".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_structural_bounds(
    *,
    full_structural_cardinality: int,
    start_id: int,
    max_ids: int | None,
) -> tuple[int, int]:
    """
    Return ``(start_id, stop_id)`` for the planned structural-ID range.
    """
    if start_id != 0:
        raise StructuralLibraryPlanError(
            "structural_library.start_id must be 0 in L4 "
            f"(non-zero starts are unsupported), got {start_id}"
        )
    if start_id < 0:
        raise StructuralLibraryPlanError(f"start_id must be >= 0, got {start_id}")
    if start_id >= full_structural_cardinality:
        raise StructuralLibraryPlanError(
            f"start_id {start_id} out of range for structural cardinality "
            f"{full_structural_cardinality}"
        )
    stop = full_structural_cardinality
    if max_ids is not None:
        if max_ids < 1:
            raise StructuralLibraryPlanError(
                f"max_ids must be >= 1 when set, got {max_ids}"
            )
        stop = min(full_structural_cardinality, max_ids)
    if stop <= start_id:
        raise StructuralLibraryPlanError(
            f"empty structural range [{start_id}, {stop})"
        )
    return start_id, stop


def plan_structural_id_shards(
    *,
    id_start: int,
    id_stop: int,
    rows_per_shard: int,
    problem_id: str,
    slab_type: str,
    snapshot_hash: str,
    run_config_fingerprint: str,
    library_structural_cardinality: int,
    full_structural_cardinality: int,
    full_candidate_cardinality: int,
) -> tuple[StructuralLibraryShardSpec, ...]:
    """
    Partition ``[id_start, id_stop)`` into contiguous shard specs.
    """
    if rows_per_shard < 1:
        raise StructuralLibraryPlanError(
            f"rows_per_shard must be >= 1, got {rows_per_shard}"
        )
    if id_start != 0:
        raise StructuralLibraryPlanError("planned id_start must be 0")
    if library_structural_cardinality != id_stop - id_start:
        raise StructuralLibraryPlanError(
            "library_structural_cardinality must equal id_stop - id_start"
        )
    specs: list[StructuralLibraryShardSpec] = []
    shard_index = 0
    cursor = id_start
    while cursor < id_stop:
        stop = min(id_stop, cursor + rows_per_shard)
        specs.append(
            StructuralLibraryShardSpec(
                problem_id=problem_id,
                slab_type=slab_type,
                shard_index=shard_index,
                id_start=cursor,
                id_stop=stop,
                rows_per_shard=rows_per_shard,
                snapshot_hash=snapshot_hash,
                run_config_fingerprint=run_config_fingerprint,
                library_structural_cardinality=library_structural_cardinality,
                full_structural_cardinality=full_structural_cardinality,
                full_candidate_cardinality=full_candidate_cardinality,
            )
        )
        shard_index += 1
        cursor = stop
    return tuple(specs)


def specs_to_task_records(
    specs: tuple[StructuralLibraryShardSpec, ...],
) -> tuple[TaskRecord, ...]:
    """
    Convert shard specs into contiguous TaskRecords.
    """
    tasks: list[TaskRecord] = []
    for index, spec in enumerate(specs):
        if index != spec.shard_index:
            raise StructuralLibraryPlanError(
                f"shard_index gap at {index}: got {spec.shard_index}"
            )
        tasks.append(
            TaskRecord(
                task_index=index,
                task_id=format_task_id(index),
                kind=STRUCTURAL_LIBRARY_TASK_KIND,
                payload=spec.to_payload(),
            )
        )
    return tuple(tasks)


def plan_structural_library_tasks(
    config: ExperimentConfig,
    *,
    snapshot_hash: str = "",
    input_hashes: dict[str, str] | None = None,
) -> tuple[TaskRecord, ...]:
    """
    Plan structural-library shard tasks from a validated config.
    """
    section = config.structural_library
    if section is None:
        raise StructuralLibraryPlanError("missing [structural_library] section")
    require_hp_slab(config.experiment.slab_type)
    problem_id = config.experiment.problem_ids[0]
    try:
        space = load_index_space(
            slab_type=config.experiment.slab_type,
            problem_id=problem_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise StructuralLibraryPlanError(str(exc)) from exc
    try:
        assert_finish_group_consistent(space)
    except ValueError as exc:
        raise StructuralLibraryPlanError(str(exc)) from exc

    full_struct = structural_cardinality(space)
    start_id, stop_id = resolve_structural_bounds(
        full_structural_cardinality=full_struct,
        start_id=section.start_id,
        max_ids=section.max_ids,
    )
    # Build a representative run-config fingerprint from catalogue defaults.
    from core.ioh_core.workflows.optimization_runs.snapshot import (
        SNAPSHOT_CSV_NAMES,
        read_packaged_slab_csv_bytes,
    )
    import tempfile

    with tempfile.TemporaryDirectory(prefix="hdb_sl_plan_") as tmp:
        root = Path(tmp)
        for name in SNAPSHOT_CSV_NAMES:
            (root / name).write_bytes(
                read_packaged_slab_csv_bytes(config.experiment.slab_type, name)
            )
        _space, _schema, info = load_problem_space_and_schema_from_csv_dir(
            slab_dir=root,
            problem_id=problem_id,
        )
        decode = make_decode(info["model"], space.var_names)
        params0 = decode(list([0] * len(space.var_names)))
        run_config = analysis_run_config_from_params(
            params0, slab_type=config.experiment.slab_type
        )

    from core.ioh_core.structural_artifacts import run_config_fingerprint

    snap = snapshot_hash or combined_snapshot_hash(input_hashes)
    specs = plan_structural_id_shards(
        id_start=start_id,
        id_stop=stop_id,
        rows_per_shard=section.rows_per_shard,
        problem_id=problem_id,
        slab_type=config.experiment.slab_type,
        snapshot_hash=snap,
        run_config_fingerprint=run_config_fingerprint(run_config),
        library_structural_cardinality=stop_id - start_id,
        full_structural_cardinality=full_struct,
        full_candidate_cardinality=space.cardinality,
    )
    return specs_to_task_records(specs)
