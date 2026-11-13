"""
Immutable data models for the brute-force dataset analysis toolbox.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.analysis_core.run_config import AnalysisRunConfig
from core.execution_core.config import ExperimentConfig
from core.execution_core.models import ExperimentManifest, ExperimentPaths
from core.ioh_core.workflows.brute_force.models import (
    BruteForceShardSpec,
    IndexSpace,
)
from core.ioh_core.workflows.brute_force.schema import BruteForceSchema


@dataclass(frozen=True)
class DatasetReportOptions:
    """
    Tunable options for brute-force dataset analysis reports.

    Attributes
    ----------
    max_violation_examples : int
        Maximum example ``global_index`` values retained per integrity rule.
    include_pairwise_level_coverage : bool
        When true, emit optional pairwise variable-level coverage tables.
    include_coviolation_matrix : bool
        When true, emit the pairwise constraint co-violation matrix.
    max_violation_combinations : int
        Maximum complete violated-constraint combinations to retain.
    near_boundary_low : float
        Inclusive lower bound of the feasible near-boundary band.
    near_boundary_high : float
        Inclusive upper bound of the violated near-boundary band.
    duckdb_threads : int
        DuckDB thread count (default ``1`` for deterministic aggregates).
    duckdb_memory_limit : str | None
        Optional DuckDB memory limit string such as ``\"4GB\"``.
    duckdb_temp_directory : Path | None
        Optional DuckDB spill directory.
    """

    max_violation_examples: int = 20
    include_pairwise_level_coverage: bool = False
    include_coviolation_matrix: bool = True
    max_violation_combinations: int = 50
    near_boundary_low: float = 0.95
    near_boundary_high: float = 1.05
    duckdb_threads: int = 1
    duckdb_memory_limit: str | None = None
    duckdb_temp_directory: Path | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_violation_examples, int)
            or isinstance(self.max_violation_examples, bool)
            or self.max_violation_examples < 1
        ):
            raise ValueError(
                "max_violation_examples must be an int >= 1, "
                f"got {self.max_violation_examples!r}"
            )
        if (
            not isinstance(self.max_violation_combinations, int)
            or isinstance(self.max_violation_combinations, bool)
            or self.max_violation_combinations < 1
        ):
            raise ValueError(
                "max_violation_combinations must be an int >= 1, "
                f"got {self.max_violation_combinations!r}"
            )
        if (
            not isinstance(self.duckdb_threads, int)
            or isinstance(self.duckdb_threads, bool)
            or self.duckdb_threads < 1
        ):
            raise ValueError(
                f"duckdb_threads must be an int >= 1, got {self.duckdb_threads!r}"
            )
        if not (0.0 <= float(self.near_boundary_low) <= 1.0):
            raise ValueError(
                "near_boundary_low must be in [0, 1], "
                f"got {self.near_boundary_low!r}"
            )
        if float(self.near_boundary_high) <= 1.0:
            raise ValueError(
                "near_boundary_high must be > 1, "
                f"got {self.near_boundary_high!r}"
            )


@dataclass(frozen=True)
class BruteForceDatasetContext:
    """
    Sealed read-only view of a published brute-force experiment.

    Attributes
    ----------
    experiment_dir : Path
        Prepared experiment root.
    paths : ExperimentPaths
        Standard experiment path bundle.
    config : ExperimentConfig
        Sealed source configuration.
    manifest : ExperimentManifest
        Sealed experiment manifest.
    dataset_dir : Path
        ``dataset/`` directory containing schema, parts, and ``_SUCCESS``.
    parts_glob : str
        Canonical Parquet glob for DuckDB / filesystem scans.
    schema_doc : Mapping[str, Any]
        Parsed ``dataset/schema.json`` document.
    schema_info : BruteForceSchema
        Schema rebuilt from the sealed snapshotted CSVs.
    index_space : IndexSpace
        Catalogue index space from the sealed snapshot.
    problem_info : Mapping[str, Any]
        Per-problem model/constraint dictionary from the sealed CSVs.
    run_config : AnalysisRunConfig
        Invalid-utilization / design-code settings derived from fixed params.
    fixed_params : Mapping[str, Any]
        Decoded fixed parameter values for the sealed problem.
    opt_penalty_alpha : float
        Penalty alpha from sealed fixed parameters.
    opt_penalty_beta : float
        Penalty beta from sealed fixed parameters.
    problem_id : str
        Sole sealed problem ID.
    slab_type : str
        Sealed slab type.
    var_names : tuple of str
        Optimisation variable names in catalogue order.
    param_names : tuple of str
        Full parameter names in schema order.
    constraint_names : tuple of str
        Constraint names in schema order (sealed, including modeling/Z).
    selected_constraint_names : tuple of str or None
        Optional analysis keep list. ``None`` means use the sealed list.
    shard_specs : tuple of BruteForceShardSpec
        Planned shard specifications from the sealed manifest.
    full_catalogue_rows : int
        Full catalogue cardinality from the sealed ``IndexSpace``.
    planned_start_index : int
        Inclusive planned linear start.
    planned_stop_index : int
        Exclusive planned linear stop.
    planned_rows : int
        Planned evaluation count (``stop - start``).
    is_full_enumeration : bool
        True when the sealed plan covers ``[0, full_catalogue_rows)``.
    coverage_fraction : float
        ``planned_rows / full_catalogue_rows``.
    collect_summary : Mapping[str, Any] | None
        Optional B8 ``aggregate/summary.json`` payload when present.
    """

    experiment_dir: Path
    paths: ExperimentPaths
    config: ExperimentConfig
    manifest: ExperimentManifest
    dataset_dir: Path
    parts_glob: str
    schema_doc: Mapping[str, Any]
    schema_info: BruteForceSchema
    index_space: IndexSpace
    problem_info: Mapping[str, Any]
    run_config: AnalysisRunConfig
    fixed_params: Mapping[str, Any]
    opt_penalty_alpha: float
    opt_penalty_beta: float
    problem_id: str
    slab_type: str
    var_names: tuple[str, ...]
    param_names: tuple[str, ...]
    constraint_names: tuple[str, ...]
    shard_specs: tuple[BruteForceShardSpec, ...]
    full_catalogue_rows: int
    planned_start_index: int
    planned_stop_index: int
    planned_rows: int
    is_full_enumeration: bool
    coverage_fraction: float
    collect_summary: Mapping[str, Any] | None = None
    selected_constraint_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_doc", dict(self.schema_doc))
        object.__setattr__(self, "problem_info", dict(self.problem_info))
        object.__setattr__(self, "fixed_params", dict(self.fixed_params))
        if self.collect_summary is not None:
            object.__setattr__(self, "collect_summary", dict(self.collect_summary))
