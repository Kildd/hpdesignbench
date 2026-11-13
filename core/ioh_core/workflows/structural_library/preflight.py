"""
Preflight estimates for structural-library builds.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.execution_core.config import ExperimentConfig
from core.ioh_core.structural_artifacts import DEFAULT_ROWS_PER_SHARD, structural_cardinality
from core.ioh_core.workflows.brute_force.planner import load_index_space
from core.ioh_core.workflows.structural_library.identity import require_hp_slab
from core.ioh_core.workflows.structural_library.planner import (
    StructuralLibraryPlanError,
    resolve_structural_bounds,
)


class StructuralLibraryPreflightError(RuntimeError):
    """
    Raised when preflight blocks preparation.
    """


@dataclass(frozen=True)
class StructuralLibraryPreflightReport:
    """
    Summary of a structural-library preflight check.
    """

    full_structural_cardinality: int
    planned_ids: int
    shard_count: int
    rows_per_shard: int
    estimated_serial_time_s: float
    estimated_total_bytes: float
    blocked: bool
    block_reasons: tuple[str, ...]
    warnings: tuple[str, ...]


# Rough constants from L0/L1 timing probes.
_APPROX_SECONDS_PER_ARTIFACT = 0.146
_APPROX_BYTES_PER_ARTIFACT = 5043.0


def run_preflight(
    config: ExperimentConfig,
    *,
    raise_if_blocked: bool = True,
) -> StructuralLibraryPreflightReport:
    """
    Estimate CPU-hours / bytes and optionally block oversized builds.
    """
    section = config.structural_library
    if section is None:
        raise StructuralLibraryPlanError("missing [structural_library] section")
    require_hp_slab(config.experiment.slab_type)
    space = load_index_space(
        slab_type=config.experiment.slab_type,
        problem_id=config.experiment.problem_ids[0],
    )
    full_struct = structural_cardinality(space)
    _start, stop = resolve_structural_bounds(
        full_structural_cardinality=full_struct,
        start_id=section.start_id,
        max_ids=section.max_ids,
    )
    planned = stop
    rows_per_shard = int(section.rows_per_shard)
    shard_count = (planned + rows_per_shard - 1) // rows_per_shard
    est_time = planned * _APPROX_SECONDS_PER_ARTIFACT
    est_bytes = planned * _APPROX_BYTES_PER_ARTIFACT

    warnings: list[str] = []
    blocks: list[str] = []
    if planned > section.maximum_expected_ids:
        msg = (
            f"planned structural IDs {planned} exceed "
            f"maximum_expected_ids {section.maximum_expected_ids}"
        )
        if section.acknowledge_large_run:
            warnings.append(msg + " (acknowledged)")
        else:
            blocks.append(msg + "; set acknowledge_large_run = true")
    if est_bytes > section.maximum_expected_bytes:
        msg = (
            f"estimated library bytes {est_bytes:.0f} exceed "
            f"maximum_expected_bytes {section.maximum_expected_bytes}"
        )
        if section.acknowledge_large_run:
            warnings.append(msg + " (acknowledged)")
        else:
            blocks.append(msg + "; set acknowledge_large_run = true")
    if rows_per_shard != DEFAULT_ROWS_PER_SHARD:
        warnings.append(
            f"rows_per_shard={rows_per_shard} differs from production default "
            f"{DEFAULT_ROWS_PER_SHARD}"
        )

    report = StructuralLibraryPreflightReport(
        full_structural_cardinality=full_struct,
        planned_ids=planned,
        shard_count=shard_count,
        rows_per_shard=rows_per_shard,
        estimated_serial_time_s=est_time,
        estimated_total_bytes=est_bytes,
        blocked=bool(blocks),
        block_reasons=tuple(blocks),
        warnings=tuple(warnings),
    )
    if report.blocked and raise_if_blocked:
        raise StructuralLibraryPreflightError(
            "; ".join(report.block_reasons) or "structural_library preflight blocked"
        )
    return report
