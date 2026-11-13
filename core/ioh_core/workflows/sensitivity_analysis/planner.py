"""
Block-aligned shard planning and preflight for sensitivity experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.execution_core.config import ExperimentConfig, SensitivityAnalysisSection
from core.execution_core.models import TaskRecord, format_task_id
from core.ioh_core.workflows.sensitivity_analysis.models import (
    SensitivityDesignError,
    SensitivityShardSpec,
    expected_evaluation_count,
)
from core.ioh_core.workflows.sensitivity_analysis.schema import rows_per_block

SENSITIVITY_TASK_KIND = "sensitivity_shard"


class SensitivityPlanError(ValueError):
    """Raised when sensitivity shards cannot be planned."""


class SensitivityPreflightError(RuntimeError):
    """Raised when preflight blocks preparation."""


@dataclass(frozen=True)
class SensitivityPreflightReport:
    """Summary of a sensitivity preflight check."""

    sample_size: int
    factor_count: int
    expected_evaluations: int
    shard_count: int
    blocks_per_shard: int
    rows_per_block: int
    approx_bytes_per_row: float
    estimated_total_bytes: float
    warnings: tuple[str, ...]
    blocked: bool
    block_reasons: tuple[str, ...]


def plan_sensitivity_shard_specs(
    *,
    sample_size: int,
    factor_count: int,
    blocks_per_shard: int,
    problem_id: str,
    slab_type: str,
    schema_hash: str,
) -> tuple[SensitivityShardSpec, ...]:
    """
    Cover ``[0, N)`` radial blocks with contiguous shard ranges.

    The last shard may contain fewer than ``blocks_per_shard`` blocks.
    """
    if sample_size < 1:
        raise SensitivityPlanError(f"sample_size must be >= 1, got {sample_size}")
    if factor_count < 1:
        raise SensitivityPlanError(f"factor_count must be >= 1, got {factor_count}")
    if blocks_per_shard < 1:
        raise SensitivityPlanError(
            f"blocks_per_shard must be >= 1, got {blocks_per_shard}"
        )
    if not schema_hash.strip():
        raise SensitivityPlanError("schema_hash must be a non-empty string")

    step = rows_per_block(factor_count)
    specs: list[SensitivityShardSpec] = []
    cursor = 0
    shard_index = 0
    while cursor < sample_size:
        stop = min(cursor + int(blocks_per_shard), int(sample_size))
        specs.append(
            SensitivityShardSpec(
                problem_id=problem_id,
                slab_type=slab_type,
                shard_index=shard_index,
                block_start=cursor,
                block_stop=stop,
                factor_count=int(factor_count),
                rows_per_block=step,
                schema_hash=str(schema_hash),
            )
        )
        cursor = stop
        shard_index += 1
    if not specs:
        raise SensitivityPlanError("planning produced no shards")
    return tuple(specs)


def specs_to_task_records(
    specs: Iterable[SensitivityShardSpec],
) -> tuple[TaskRecord, ...]:
    """Convert shard specs into contiguous generic task records."""
    records: list[TaskRecord] = []
    for index, spec in enumerate(specs):
        if spec.shard_index != index:
            raise SensitivityPlanError(
                f"shard_index mismatch: expected {index}, got {spec.shard_index}"
            )
        records.append(
            TaskRecord(
                task_index=index,
                task_id=format_task_id(index),
                kind=SENSITIVITY_TASK_KIND,
                payload=spec.to_payload(),
            )
        )
    return tuple(records)


def approx_bytes_per_row(*, var_count: int, factor_count: int) -> float:
    """Conservative stored-bytes heuristic before real Parquet sizing."""
    # Identity columns + idx/param overhead + scientific floats.
    return float(320 + 20 * int(var_count) + 8 * int(factor_count))


def run_sensitivity_preflight(
    *,
    section: SensitivityAnalysisSection,
    factor_count: int,
    var_count: int,
) -> SensitivityPreflightReport:
    """
    Check planned evaluation count and storage against configured maxima.
    """
    try:
        expected = expected_evaluation_count(section.sample_size, factor_count)
    except SensitivityDesignError as exc:
        raise SensitivityPlanError(str(exc)) from exc

    specs = plan_sensitivity_shard_specs(
        sample_size=section.sample_size,
        factor_count=factor_count,
        blocks_per_shard=section.blocks_per_shard,
        problem_id="preflight",
        slab_type="preflight",
        schema_hash="preflight",
    )
    bytes_per_row = approx_bytes_per_row(
        var_count=var_count, factor_count=factor_count
    )
    estimated_bytes = float(expected) * bytes_per_row

    warnings: list[str] = []
    block_reasons: list[str] = []
    if expected > section.maximum_expected_evaluations:
        message = (
            f"expected evaluations {expected} exceed "
            f"maximum_expected_evaluations={section.maximum_expected_evaluations}"
        )
        if section.acknowledge_large_run:
            warnings.append(message + " (acknowledged)")
        else:
            block_reasons.append(message)
    if estimated_bytes > float(section.maximum_expected_bytes):
        message = (
            f"estimated bytes {estimated_bytes:.0f} exceed "
            f"maximum_expected_bytes={section.maximum_expected_bytes}"
        )
        if section.acknowledge_large_run:
            warnings.append(message + " (acknowledged)")
        else:
            block_reasons.append(message)

    blocked = bool(block_reasons)
    report = SensitivityPreflightReport(
        sample_size=section.sample_size,
        factor_count=factor_count,
        expected_evaluations=expected,
        shard_count=len(specs),
        blocks_per_shard=section.blocks_per_shard,
        rows_per_block=rows_per_block(factor_count),
        approx_bytes_per_row=bytes_per_row,
        estimated_total_bytes=estimated_bytes,
        warnings=tuple(warnings),
        blocked=blocked,
        block_reasons=tuple(block_reasons),
    )
    if blocked:
        raise SensitivityPreflightError(
            "sensitivity preflight blocked preparation: "
            + "; ".join(report.block_reasons)
        )
    return report


def plan_sensitivity_tasks(
    config: ExperimentConfig,
) -> tuple[TaskRecord, ...]:
    """
    Plan contiguous sensitivity ``TaskRecord`` entries for a config.

    Rebuilds the Parquet schema from packaged slab CSVs (same contract as
    brute-force planning) so the planner does not need ``experiment_dir``.
    """
    if config.experiment.workflow != "sensitivity_analysis":
        raise SensitivityPlanError(
            "plan_sensitivity_tasks requires workflow='sensitivity_analysis', "
            f"got {config.experiment.workflow!r}"
        )
    if config.sensitivity_analysis is None:
        raise SensitivityPlanError("missing sensitivity_analysis section")
    section = config.sensitivity_analysis
    if len(config.experiment.problem_ids) != 1:
        raise SensitivityPlanError("sensitivity_analysis requires one problem_id")

    factor_count = len(section.factor_names)
    schema_info = load_sensitivity_schema_from_packaged(
        slab_type=config.experiment.slab_type,
        problem_id=config.experiment.problem_ids[0],
        factor_names=section.factor_names,
    )
    specs = plan_sensitivity_shard_specs(
        sample_size=section.sample_size,
        factor_count=factor_count,
        blocks_per_shard=section.blocks_per_shard,
        problem_id=config.experiment.problem_ids[0],
        slab_type=config.experiment.slab_type,
        schema_hash=schema_info.schema_hash,
    )
    return specs_to_task_records(specs)


def load_sensitivity_schema_from_packaged(
    *,
    slab_type: str,
    problem_id: str,
    factor_names: Iterable[str],
):
    """
    Build the sensitivity Parquet schema from packaged slab CSVs.

    Uses a temporary CSV snapshot so planning matches prepare-time sealing
    without reading ``experiment_dir``.
    """
    import tempfile
    from pathlib import Path

    from core.ioh_core.problem_builder import build_problem_from_snapshot
    from core.ioh_core.workflows.optimization_runs.snapshot import (
        SNAPSHOT_CSV_NAMES,
        read_packaged_slab_csv_bytes,
    )
    from core.ioh_core.workflows.sensitivity_analysis.schema import (
        build_sensitivity_schema,
    )

    names = tuple(str(name) for name in factor_names)
    try:
        with tempfile.TemporaryDirectory(prefix="hdb_sa_schema_") as tmp:
            root = Path(tmp)
            for filename in SNAPSHOT_CSV_NAMES:
                (root / filename).write_bytes(
                    read_packaged_slab_csv_bytes(slab_type, filename)
                )
            bundle = build_problem_from_snapshot(
                slab_type=slab_type,
                problem_id=problem_id,
                snapshot_dir=root,
            )
            return build_sensitivity_schema(
                var_names=bundle["var_names"],
                factor_names=names,
                model=bundle["model"],
                constraint_names=tuple(bundle["constraints"].keys()),
            )
    except SensitivityPlanError:
        raise
    except Exception as exc:
        raise SensitivityPlanError(
            f"could not build sensitivity schema for slab_type={slab_type!r} "
            f"problem_id={problem_id!r}: {exc}"
        ) from exc
