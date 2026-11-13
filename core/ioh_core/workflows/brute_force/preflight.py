"""
Preflight estimates and threshold checks for brute-force experiments.

Derives cardinality from the packaged :class:`IndexSpace`, never from a
hard-coded total. Does not write Parquet outputs.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from core.execution_core.config import ExperimentConfig
from core.ioh_core.workflows.brute_force.indexing import linear_to_indices
from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.brute_force.planner import (
    BruteForcePlanError,
    aligned_rows_per_shard,
    finish_group_size,
    load_index_space,
    plan_shard_specs,
    resolve_enumeration_bounds,
)


class BruteForcePreflightError(RuntimeError):
    """
    Raised when preflight blocks preparation (limits exceeded without ack).
    """


@dataclass(frozen=True)
class PreflightReport:
    """
    Summary of a brute-force preflight check.

    Attributes
    ----------
    cardinality : int
        Exact catalogue size from :class:`IndexSpace`.
    shard_count : int
        Number of planned shards.
    rows_per_shard : int
        Effective finish-group-aligned shard length.
    finish_group_size : int
        Product of the last two variable sizes.
    probe_indices : tuple of int
        Deterministic linear probe addresses.
    mean_evaluation_time_s : float
        Mean wall seconds over sampled probes.
    estimated_serial_time_s : float
        ``cardinality * mean_evaluation_time_s``.
    estimated_parallel_time_s : float
        Serial estimate divided by ``max_parallel_tasks``.
    approx_bytes_per_row : float
        Approximate stored bytes per evaluation row.
    estimated_total_bytes : float
        ``cardinality * approx_bytes_per_row``.
    filesystem_capacity_bytes : int or None
        Detected total capacity for ``probe_path``, when available.
    filesystem_free_bytes : int or None
        Detected free bytes for ``probe_path``, when available.
    warnings : tuple of str
        Non-fatal quota / uncertainty messages.
    blocked : bool
        ``True`` when preparation must not proceed.
    block_reasons : tuple of str
        Human-readable block reasons when ``blocked``.
    """

    cardinality: int
    shard_count: int
    rows_per_shard: int
    finish_group_size: int
    probe_indices: tuple[int, ...]
    mean_evaluation_time_s: float
    estimated_serial_time_s: float
    estimated_parallel_time_s: float
    approx_bytes_per_row: float
    estimated_total_bytes: float
    filesystem_capacity_bytes: int | None
    filesystem_free_bytes: int | None
    warnings: tuple[str, ...]
    blocked: bool
    block_reasons: tuple[str, ...]


def deterministic_probe_indices(cardinality: int, *, count: int = 5) -> tuple[int, ...]:
    """
    Return deterministic linear probes spanning a catalogue.

    Always includes ``0`` and ``cardinality - 1`` when ``cardinality >= 1``.
    Additional interior points use fixed fractions of the domain.

    Parameters
    ----------
    cardinality : int
        Catalogue size (>= 0).
    count : int, optional
        Maximum number of probes to return.

    Returns
    -------
    tuple of int
        Unique sorted linear indices.
    """
    if cardinality <= 0:
        return ()
    if cardinality == 1:
        return (0,)
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    chosen: list[int] = []
    for fraction in fractions:
        index = int(round(fraction * (cardinality - 1)))
        if index not in chosen:
            chosen.append(index)
        if len(chosen) >= count:
            break
    return tuple(sorted(chosen))


def _default_approx_bytes_per_row(space: IndexSpace) -> float:
    """
    Heuristic bytes/row until Parquet schema sizing exists.

    Parameters
    ----------
    space : IndexSpace
        Candidate space (variable count scales the estimate).

    Returns
    -------
    float
        Approximate bytes per row.
    """
    # Fixed overhead + int32 index/param columns + float constraint block.
    return float(256 + 16 * len(space.var_names) + 64)


def _sample_evaluation_times(
    *,
    space: IndexSpace,
    probe_indices: Sequence[int],
    evaluate: Callable[[tuple[int, ...]], object] | None,
) -> tuple[float, tuple[str, ...]]:
    """
    Time probe evaluations or fall back to a conservative synthetic estimate.

    Parameters
    ----------
    space : IndexSpace
        Candidate space.
    probe_indices : sequence of int
        Linear probes.
    evaluate : callable or None
        Optional ``evaluate(catalogue_indices) -> result``. When ``None``,
        uses a synthetic 0.05 s/eval estimate and a warning.

    Returns
    -------
    tuple
        ``(mean_seconds, warnings)``.
    """
    warnings: list[str] = []
    if not probe_indices:
        return 0.0, ("no probe indices available for timing",)

    if evaluate is None:
        warnings.append(
            "preflight used a synthetic 0.05 s/eval estimate "
            "(no evaluator provided)"
        )
        return 0.05, tuple(warnings)

    times: list[float] = []
    for linear in probe_indices:
        vector = linear_to_indices(int(linear), space)
        started = time.perf_counter()
        evaluate(vector)
        times.append(time.perf_counter() - started)
    mean = float(sum(times) / len(times))
    return mean, tuple(warnings)


def run_preflight(
    config: ExperimentConfig,
    *,
    probe_path: Path | None = None,
    evaluate: Callable[[tuple[int, ...]], object] | None = None,
    raise_if_blocked: bool = True,
) -> PreflightReport:
    """
    Estimate cost/storage for a brute-force config and optionally block.

    Parameters
    ----------
    config : ExperimentConfig
        Validated configuration with ``workflow='brute_force'``.
    probe_path : Path or None, optional
        Filesystem path used for capacity detection (defaults to CWD).
    evaluate : callable or None, optional
        Optional catalogue-index evaluator for timing samples.
    raise_if_blocked : bool, optional
        When ``True`` (default), raise :class:`BruteForcePreflightError` if
        limits are exceeded without acknowledgement.

    Returns
    -------
    PreflightReport
        Structured preflight summary.

    Raises
    ------
    BruteForcePlanError
        If planning inputs are invalid.
    BruteForcePreflightError
        If ``raise_if_blocked`` and the run is blocked.
    """
    if config.experiment.workflow != "brute_force":
        raise BruteForcePlanError(
            "run_preflight requires workflow='brute_force', "
            f"got {config.experiment.workflow!r}"
        )
    if config.brute_force is None:
        raise BruteForcePlanError("missing [brute_force] configuration")

    settings = config.brute_force
    problem_id = config.experiment.problem_ids[0]
    space = load_index_space(
        slab_type=config.experiment.slab_type,
        problem_id=problem_id,
    )
    group = finish_group_size(space)
    rows = aligned_rows_per_shard(settings.rows_per_shard, group)
    start_index, stop_index = resolve_enumeration_bounds(
        start_index=settings.start_index,
        max_rows=settings.max_rows,
        group_size=group,
        cardinality=space.cardinality,
    )
    planned_cardinality = stop_index - start_index
    shards = plan_shard_specs(
        space=space,
        problem_id=problem_id,
        slab_type=config.experiment.slab_type,
        rows_per_shard=settings.rows_per_shard,
        start_index=start_index,
        stop_index=stop_index,
    )
    probes = tuple(
        start_index + offset
        for offset in deterministic_probe_indices(planned_cardinality)
    )
    mean_s, timing_warnings = _sample_evaluation_times(
        space=space,
        probe_indices=probes,
        evaluate=evaluate,
    )
    serial_s = float(planned_cardinality) * mean_s
    parallel = max(1, int(config.execution.max_parallel_tasks))
    parallel_s = serial_s / float(parallel)
    bytes_per_row = _default_approx_bytes_per_row(space)
    total_bytes = float(planned_cardinality) * bytes_per_row

    capacity: int | None = None
    free: int | None = None
    warnings = list(timing_warnings)
    path = Path(probe_path) if probe_path is not None else Path.cwd()
    try:
        usage = shutil.disk_usage(path)
        capacity = int(usage.total)
        free = int(usage.free)
        if free is not None and total_bytes > 0.5 * free:
            warnings.append(
                "estimated dataset size exceeds 50% of detectable free disk "
                f"at {path} (estimate is approximate; quota may differ)"
            )
        warnings.append(
            "filesystem free-space figures may not reflect user quotas"
        )
    except OSError:
        warnings.append(
            f"could not detect filesystem capacity for {path}; "
            "storage estimate is unchecked against free disk"
        )

    block_reasons: list[str] = []
    if planned_cardinality > settings.maximum_expected_rows:
        block_reasons.append(
            f"cardinality {planned_cardinality} exceeds "
            f"maximum_expected_rows={settings.maximum_expected_rows}"
        )
    if total_bytes > float(settings.maximum_expected_bytes):
        block_reasons.append(
            f"estimated_total_bytes {total_bytes:.0f} exceeds "
            f"maximum_expected_bytes={settings.maximum_expected_bytes}"
        )
    blocked = bool(block_reasons) and not settings.acknowledge_large_run
    if block_reasons and settings.acknowledge_large_run:
        warnings.append(
            "large-run thresholds exceeded but acknowledge_large_run=true"
        )

    report = PreflightReport(
        cardinality=planned_cardinality,
        shard_count=len(shards),
        rows_per_shard=rows,
        finish_group_size=group,
        probe_indices=probes,
        mean_evaluation_time_s=mean_s,
        estimated_serial_time_s=serial_s,
        estimated_parallel_time_s=parallel_s,
        approx_bytes_per_row=bytes_per_row,
        estimated_total_bytes=total_bytes,
        filesystem_capacity_bytes=capacity,
        filesystem_free_bytes=free,
        warnings=tuple(warnings),
        blocked=blocked,
        block_reasons=tuple(block_reasons),
    )
    if blocked and raise_if_blocked:
        joined = "; ".join(block_reasons)
        raise BruteForcePreflightError(
            "brute-force preflight blocked preparation: "
            f"{joined}. Set acknowledge_large_run=true to proceed."
        )
    return report
