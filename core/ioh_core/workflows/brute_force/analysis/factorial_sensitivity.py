"""
Exact finite-grid first- and total-order Sobol indices for full factorials.

This module owns the mathematical contracts, eligibility gate, and a small
in-memory tensor oracle used only for correctness tests. Production analysis
must compute the same quantities out-of-core over sealed brute-force Parquet
and must not materialize the exhaustive response tensor.

Reference: Owen (2013), DOI ``10.1137/120876782``, §2 functional ANOVA /
Sobol–Hoeffding variance components on product domains.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.execution_core.atomic_io import write_text_atomic
from core.ioh_core.workflows.brute_force.analysis.context import (
    open_brute_force_dataset,
)
from core.ioh_core.workflows.brute_force.analysis.coverage import (
    run_coverage_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.engine import (
    BruteForceQueryEngine,
    open_query_engine,
    quote_ident,
)
from core.ioh_core.workflows.brute_force.analysis.integrity import (
    run_integrity_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.keep_list import (
    analysis_constraint_names,
    attach_selected_constraint_names,
    merge_constraint_provenance,
)
from core.ioh_core.workflows.brute_force.analysis.models import (
    BruteForceDatasetContext,
    DatasetReportOptions,
)
from core.ioh_core.workflows.brute_force.analysis.serialization import (
    dumps_report_json,
    normalize_report_value,
)

METHOD_ID = "sobol_hoeffding_functional_anova"
METHOD_TITLE = "Exact finite-grid Sobol–Hoeffding / functional ANOVA"
OWEN_CITATION = (
    "Owen, A. B. (2013). Variance components and generalized Sobol' indices. "
    "SIAM/ASA Journal on Uncertainty Quantification, 1(1), 19–41."
)
OWEN_DOI = "10.1137/120876782"
VARIANCE_DDOF = 0
MEASURE_DESCRIPTION = (
    "finite discrete-uniform product measure over a complete Cartesian "
    "catalogue; every level combination has equal weight"
)

RESPONSE_FAMILY_OBJECTIVE = "objective"
RESPONSE_FAMILY_UTILIZATION = "utilization"
RESPONSE_FAMILY_FEASIBILITY = "feasibility"

EXPRESSION_COLUMN = "column"
EXPRESSION_BINARY_FEASIBLE = "binary_feasible"

STATUS_OK = "ok"
STATUS_CONSTANT = "constant"


@dataclass(frozen=True)
class FactorialSensitivityMethodMeta:
    """Stable methodology metadata recorded with published results."""

    method_id: str = METHOD_ID
    method_title: str = METHOD_TITLE
    variance_ddof: int = VARIANCE_DDOF
    measure: str = MEASURE_DESCRIPTION
    reference: str = OWEN_CITATION
    doi: str = OWEN_DOI
    publishes_sampling_ci: bool = False
    publishes_second_order: bool = False
    notes: str = (
        "S1_i = Var(E[Y|Xi]) / Var(Y); "
        "ST_i = E[Var(Y|X~i)] / Var(Y); "
        "1 - Var(E[Y|X~i]) / Var(Y) is retained only as an identity diagnostic."
    )


@dataclass(frozen=True)
class FactorialResponseSpec:
    """
    One analyzable response derived from a sealed brute-force row.

    Attributes
    ----------
    family:
        ``objective``, ``utilization``, or ``feasibility``.
    name:
        Stable response label (``y_p``, ``util__<c>``, or ``feasible__<c>``).
    source_column:
        Dataset column providing the numeric source (``y_p`` or ``util__*``).
    expression:
        ``column`` copies the source; ``binary_feasible`` maps
        ``1[source <= 1]`` (finite invalid-utilization sentinels above 1 are
        infeasible).
    """

    family: str
    name: str
    source_column: str
    expression: str

    def __post_init__(self) -> None:
        if self.family not in {
            RESPONSE_FAMILY_OBJECTIVE,
            RESPONSE_FAMILY_UTILIZATION,
            RESPONSE_FAMILY_FEASIBILITY,
        }:
            raise ValueError(f"unknown response family: {self.family!r}")
        if self.expression not in {
            EXPRESSION_COLUMN,
            EXPRESSION_BINARY_FEASIBLE,
        }:
            raise ValueError(f"unknown response expression: {self.expression!r}")
        if not self.name:
            raise ValueError("response name must be non-empty")
        if not self.source_column:
            raise ValueError("source_column must be non-empty")


@dataclass(frozen=True)
class FactorSobolIndices:
    """Exact first- and total-order indices for one factor and one response."""

    variable: str
    axis: int
    first_order: float
    total_effect: float
    total_effect_via_complement_mean: float
    interaction_gap: float
    identity_residual: float


@dataclass(frozen=True)
class TensorSobolResult:
    """
    Exact indices for every axis of a small full-factorial response tensor.

    This is a correctness oracle result, not a production artifact.
    """

    first_order: NDArray[np.floating]
    total_effect: NDArray[np.floating]
    total_effect_via_complement_mean: NDArray[np.floating]
    identity_residual: NDArray[np.floating]
    mean: float
    total_variance: float
    factor_count: int


def default_method_meta() -> FactorialSensitivityMethodMeta:
    """Return the locked methodology metadata for exact factorial indices."""
    return FactorialSensitivityMethodMeta()


def build_factorial_response_specs(
    utilization_names: Sequence[str],
    *,
    objective_column: str = "y_p",
) -> tuple[FactorialResponseSpec, ...]:
    """
    Build the sealed response list: ``y_p``, each ``util__*``, each binary.

    Parameters
    ----------
    utilization_names:
        Constraint names in sealed catalogue order (without the ``util__``
        prefix).
    objective_column:
        Penalized objective column name (default ``y_p``).

    Returns
    -------
    tuple of FactorialResponseSpec
        Deterministic response contract used by later eligibility/engine steps.
    """
    if not isinstance(objective_column, str) or not objective_column:
        raise ValueError("objective_column must be a non-empty str")

    specs: list[FactorialResponseSpec] = [
        FactorialResponseSpec(
            family=RESPONSE_FAMILY_OBJECTIVE,
            name=objective_column,
            source_column=objective_column,
            expression=EXPRESSION_COLUMN,
        )
    ]
    seen: set[str] = set()
    for raw_name in utilization_names:
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(
                f"utilization names must be non-empty strings; got {raw_name!r}"
            )
        if raw_name in seen:
            raise ValueError(f"duplicate utilization name: {raw_name!r}")
        seen.add(raw_name)
        util_column = f"util__{raw_name}"
        specs.append(
            FactorialResponseSpec(
                family=RESPONSE_FAMILY_UTILIZATION,
                name=util_column,
                source_column=util_column,
                expression=EXPRESSION_COLUMN,
            )
        )
        specs.append(
            FactorialResponseSpec(
                family=RESPONSE_FAMILY_FEASIBILITY,
                name=f"feasible__{raw_name}",
                source_column=util_column,
                expression=EXPRESSION_BINARY_FEASIBLE,
            )
        )
    return tuple(specs)


def materialize_response_values(
    source_values: ArrayLike,
    *,
    expression: str,
) -> NDArray[np.floating]:
    """
    Materialize one response vector/tensor from a numeric source array.

    Used by unit tests and small synthetic grids. Production SQL builds the
    same expressions without calling this helper on full datasets.
    """
    values = np.asarray(source_values, dtype=float)
    if expression == EXPRESSION_COLUMN:
        return values
    if expression == EXPRESSION_BINARY_FEASIBLE:
        return np.where(values <= 1.0, 1.0, 0.0).astype(float, copy=False)
    raise ValueError(f"unknown response expression: {expression!r}")


def exact_sobol_from_tensor(Y: ArrayLike) -> TensorSobolResult:
    """
    Exact first- and total-order indices for a full-factorial response tensor.

    Each axis is one discrete input. Every cell has equal weight. Variances use
    population ``ddof=0``. ``total_effect`` is ``E[Var(Y|X~i)] / Var(Y)``;
    ``total_effect_via_complement_mean`` is the algebraically equivalent
    ``1 - Var(E[Y|X~i]) / Var(Y)`` retained as an identity diagnostic.

    This function is the small-grid correctness oracle only. Do not call it on
    production brute-force enumerations.

    Parameters
    ----------
    Y:
        Dense response tensor with one axis per variable.

    Returns
    -------
    TensorSobolResult
        Exact indices and variance diagnostics.

    Raises
    ------
    ValueError
        If ``Y`` is empty, non-finite, zero-dimensional, or constant.
    """
    tensor = np.asarray(Y, dtype=float)
    if tensor.ndim < 1:
        raise ValueError(
            f"response tensor must have at least one axis; got shape {tensor.shape}"
        )
    if tensor.size < 1:
        raise ValueError("response tensor must be non-empty")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("response tensor must contain only finite values")

    mean = float(np.mean(tensor))
    total_variance = float(np.mean((tensor - mean) ** 2))
    if total_variance == 0.0:
        raise ValueError("The objective has zero variance.")

    factor_count = int(tensor.ndim)
    first_order = np.empty(factor_count, dtype=float)
    total_effect = np.empty(factor_count, dtype=float)
    total_effect_via_complement_mean = np.empty(factor_count, dtype=float)

    for axis in range(factor_count):
        other_axes = tuple(i for i in range(factor_count) if i != axis)

        # E[Y | Xi]: one mean per level of axis.
        mean_given_xi = np.mean(tensor, axis=other_axes)
        first_order[axis] = (
            float(np.mean((mean_given_xi - mean) ** 2)) / total_variance
        )

        # E[Var(Y | X~i)] / V  (primary total-effect definition)
        moved = np.moveaxis(tensor, axis, 0)
        conditional_mean = np.mean(moved, axis=0, keepdims=True)
        conditional_variance = np.mean((moved - conditional_mean) ** 2, axis=0)
        total_effect[axis] = float(np.mean(conditional_variance)) / total_variance

        # Identity form: 1 - Var(E[Y | X~i]) / V
        mean_given_others = np.mean(tensor, axis=axis)
        complement_mean_variance = float(
            np.mean((mean_given_others - mean) ** 2)
        )
        total_effect_via_complement_mean[axis] = (
            1.0 - complement_mean_variance / total_variance
        )

    identity_residual = total_effect - total_effect_via_complement_mean
    return TensorSobolResult(
        first_order=first_order,
        total_effect=total_effect,
        total_effect_via_complement_mean=total_effect_via_complement_mean,
        identity_residual=identity_residual,
        mean=mean,
        total_variance=total_variance,
        factor_count=factor_count,
    )


def factor_results_from_tensor(
    Y: ArrayLike,
    *,
    variable_names: Sequence[str] | None = None,
) -> tuple[FactorSobolIndices, ...]:
    """
    Wrap :func:`exact_sobol_from_tensor` as per-factor frozen records.

    Parameters
    ----------
    Y:
        Dense response tensor.
    variable_names:
        Optional axis names. Defaults to ``x0`` … ``x{d-1}``.
    """
    result = exact_sobol_from_tensor(Y)
    if variable_names is None:
        names = tuple(f"x{axis}" for axis in range(result.factor_count))
    else:
        names = tuple(variable_names)
        if len(names) != result.factor_count:
            raise ValueError(
                "variable_names length must match tensor ndim; "
                f"got {len(names)} names for {result.factor_count} axes"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"variable_names must be unique; got {names!r}")

    records: list[FactorSobolIndices] = []
    for axis, name in enumerate(names):
        s1 = float(result.first_order[axis])
        st = float(result.total_effect[axis])
        st_identity = float(result.total_effect_via_complement_mean[axis])
        records.append(
            FactorSobolIndices(
                variable=name,
                axis=axis,
                first_order=s1,
                total_effect=st,
                total_effect_via_complement_mean=st_identity,
                interaction_gap=st - s1,
                identity_residual=float(result.identity_residual[axis]),
            )
        )
    return tuple(records)


@dataclass(frozen=True)
class FactorialSensitivityEligibility:
    """
    Fail-closed eligibility result for exact factorial sensitivity analysis.

    Attributes
    ----------
    ok:
        True only when every required invariant passes.
    reasons:
        Stable machine-readable failure codes (empty when ``ok``).
    diagnostics:
        Bounded JSON-compatible details for operator diagnostics.
    response_specs:
        Response contract derived from the sealed constraint list.
    variable_names:
        Optimisation variables in sealed catalogue order.
    level_counts:
        Declared level counts per variable (``IndexSpace.sizes``).
    cardinality:
        Declared full-catalogue cardinality.
    invalid_utilization:
        Sealed finite invalid-utilization sentinel.
    """

    ok: bool
    reasons: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    response_specs: tuple[FactorialResponseSpec, ...]
    variable_names: tuple[str, ...]
    level_counts: tuple[int, ...]
    cardinality: int
    invalid_utilization: float


def _existing_columns(engine: BruteForceQueryEngine) -> set[str]:
    """
    Discover columns present on the bound DuckDB view (not sealed schema alone).
    """
    rows = engine.fetch_rows(
        f"DESCRIBE SELECT * FROM {quote_ident(engine.view_name)}",
        max_rows=10_000,
    )
    return {str(row[0]) for row in rows}


def _distinct_index_tuple_count(
    engine: BruteForceQueryEngine,
    *,
    var_names: Sequence[str],
) -> int:
    view = quote_ident(engine.view_name)
    idx_cols = [quote_ident(f"idx__{name}") for name in var_names]
    if not idx_cols:
        raise ValueError("factorial eligibility requires at least one variable")
    sql = (
        f"SELECT COUNT(*) FROM ("
        f"SELECT DISTINCT {', '.join(idx_cols)} FROM {view}"
        f")"
    )
    row = engine.fetch_one(sql)
    if row is None:
        raise RuntimeError("distinct index-tuple count query returned no row")
    return int(row[0])


def _response_quality_counts(
    engine: BruteForceQueryEngine,
    *,
    objective_column: str,
    util_columns: Sequence[str],
    max_examples: int,
) -> dict[str, Any]:
    view = quote_ident(engine.view_name)
    aggregates = [
        "COUNT(*) AS row_count",
        (
            f"SUM(CASE WHEN NOT {quote_ident('analysis_ok')} THEN 1 ELSE 0 END) "
            "AS analysis_not_ok"
        ),
        (
            "SUM(CASE WHEN "
            f"{quote_ident(objective_column)} IS NULL OR "
            f"NOT isfinite({quote_ident(objective_column)}) "
            "THEN 1 ELSE 0 END) AS nonfinite_objective"
        ),
    ]
    for column in util_columns:
        safe = column.replace('"', "")
        aggregates.append(
            "SUM(CASE WHEN "
            f"{quote_ident(column)} IS NULL OR "
            f"NOT isfinite({quote_ident(column)}) "
            f'THEN 1 ELSE 0 END) AS "nonfinite__{safe}"'
        )
    agg_rows = engine.fetch_dicts(
        f"SELECT {', '.join(aggregates)} FROM {view}",
        max_rows=1,
    )
    if not agg_rows:
        raise RuntimeError("response quality aggregate returned no row")
    agg = agg_rows[0]

    def _examples(where_sql: str) -> list[int]:
        rows = engine.fetch_rows(
            f"SELECT {quote_ident('global_index')} FROM {view} "
            f"WHERE {where_sql} "
            f"ORDER BY {quote_ident('global_index')} "
            f"LIMIT {int(max_examples)}",
            max_rows=max_examples,
        )
        return [int(row[0]) for row in rows]

    analysis_not_ok = int(agg.get("analysis_not_ok") or 0)
    nonfinite_objective = int(agg.get("nonfinite_objective") or 0)
    util_bad: dict[str, dict[str, Any]] = {}
    for column in util_columns:
        count = int(agg.get(f"nonfinite__{column}") or 0)
        util_bad[column] = {
            "count": count,
            "examples": (
                _examples(
                    f"{quote_ident(column)} IS NULL OR "
                    f"NOT isfinite({quote_ident(column)})"
                )
                if count
                else []
            ),
        }

    return {
        "row_count": int(agg.get("row_count") or 0),
        "analysis_not_ok": {
            "count": analysis_not_ok,
            "examples": (
                _examples(f"NOT {quote_ident('analysis_ok')}")
                if analysis_not_ok
                else []
            ),
        },
        "nonfinite_objective": {
            "count": nonfinite_objective,
            "examples": (
                _examples(
                    f"{quote_ident(objective_column)} IS NULL OR "
                    f"NOT isfinite({quote_ident(objective_column)})"
                )
                if nonfinite_objective
                else []
            ),
        },
        "nonfinite_utilization": util_bad,
    }


def assess_factorial_sensitivity_eligibility(
    context: BruteForceDatasetContext,
    engine: BruteForceQueryEngine,
    *,
    options: DatasetReportOptions | None = None,
    include_integrity: bool = True,
) -> FactorialSensitivityEligibility:
    """
    Fail-closed eligibility gate for exact full-factorial sensitivity.

    Requires a complete sealed enumeration (``is_full_enumeration``), matching
    cardinality and unique catalogue combinations, exact coverage marginals,
    all ``analysis_ok``, finite ``y_p`` / ``util__*``, and (by default) passing
    B8/scientific integrity uniqueness rules.

    Parameters
    ----------
    context:
        Sealed brute-force dataset context.
    engine:
        Open DuckDB engine over the same dataset.
    options:
        Shared report options (example bounds, DuckDB settings unused here).
    include_integrity:
        When true (production default), run ``run_integrity_analysis`` and
        require ``ok`` plus uniqueness. Synthetic-grid unit tests may set this
        false when the fixture omits full BF scientific columns.

    Returns
    -------
    FactorialSensitivityEligibility
        Deterministic pass/fail payload. Never writes artifacts.
    """
    options = options or DatasetReportOptions()
    max_examples = int(options.max_violation_examples)

    variable_names = tuple(context.var_names)
    level_counts = tuple(int(size) for size in context.index_space.sizes)
    cardinality = int(context.index_space.cardinality)
    invalid_utilization = float(context.run_config.invalid_utilization)
    response_specs = build_factorial_response_specs(
        analysis_constraint_names(context)
    )
    objective_column = "y_p"
    util_columns = tuple(
        spec.source_column
        for spec in response_specs
        if spec.family == RESPONSE_FAMILY_UTILIZATION
    )

    reasons: list[str] = []
    diagnostics: dict[str, Any] = {
        "is_full_enumeration": bool(context.is_full_enumeration),
        "planned_rows": int(context.planned_rows),
        "planned_start_index": int(context.planned_start_index),
        "planned_stop_index": int(context.planned_stop_index),
        "full_catalogue_rows": int(context.full_catalogue_rows),
        "index_space_cardinality": cardinality,
        "include_integrity": bool(include_integrity),
    }

    if not context.is_full_enumeration:
        reasons.append("not_full_enumeration")

    if int(context.full_catalogue_rows) != cardinality:
        reasons.append("cardinality_context_mismatch")
        diagnostics["cardinality_context_mismatch"] = {
            "full_catalogue_rows": int(context.full_catalogue_rows),
            "index_space_cardinality": cardinality,
        }

    if (
        int(context.planned_start_index) != 0
        or int(context.planned_stop_index) != cardinality
        or int(context.planned_rows) != cardinality
    ):
        # Redundant with is_full_enumeration for sealed plans, but keeps the
        # diagnostic explicit when a synthetic context is inconsistent.
        if "not_full_enumeration" not in reasons:
            reasons.append("planned_window_not_full_catalogue")
        diagnostics["planned_window"] = {
            "start": int(context.planned_start_index),
            "stop": int(context.planned_stop_index),
            "rows": int(context.planned_rows),
            "cardinality": cardinality,
        }

    present = _existing_columns(engine)
    required_columns = (
        "global_index",
        "analysis_ok",
        objective_column,
        *[f"idx__{name}" for name in variable_names],
        *util_columns,
    )
    missing_columns = tuple(
        name for name in required_columns if name not in present
    )
    diagnostics["missing_columns"] = list(missing_columns)
    if missing_columns:
        reasons.append("missing_response_columns")

    observed_rows = engine.count_rows()
    diagnostics["observed_rows"] = int(observed_rows)
    if observed_rows != cardinality:
        reasons.append("row_count_mismatch")

    integrity_report: dict[str, Any] | None = None
    if include_integrity and "missing_response_columns" not in reasons:
        integrity_report = run_integrity_analysis(
            context,
            engine,
            options=options,
        )
        diagnostics["integrity_ok"] = bool(integrity_report.get("ok"))
        diagnostics["integrity_overall_status"] = integrity_report.get(
            "overall_status"
        )
        uniqueness = integrity_report.get("uniqueness") or {}
        diagnostics["integrity_uniqueness_ok"] = bool(uniqueness.get("ok"))
        if not integrity_report.get("ok") or not uniqueness.get("ok"):
            reasons.append("integrity_failed")
    elif include_integrity:
        diagnostics["integrity_ok"] = False
        diagnostics["integrity_skipped"] = "missing_response_columns"
        reasons.append("integrity_failed")
    else:
        diagnostics["integrity_ok"] = None
        diagnostics["integrity_skipped"] = "include_integrity=false"

    coverage_report: dict[str, Any] | None = None
    if "missing_response_columns" not in reasons:
        coverage_report = run_coverage_analysis(
            context,
            engine,
            options=options,
        )
        diagnostics["coverage_ok"] = bool(coverage_report.get("ok"))
        if not coverage_report.get("ok"):
            reasons.append("coverage_failed")
        failed_vars = [
            {
                "name": item.get("name"),
                "failures": list(item.get("failures") or []),
            }
            for item in (coverage_report.get("variables") or [])
            if not item.get("ok")
        ]
        if failed_vars:
            diagnostics["coverage_variable_failures"] = failed_vars
    else:
        diagnostics["coverage_ok"] = False
        diagnostics["coverage_skipped"] = "missing_response_columns"
        reasons.append("coverage_failed")

    if "missing_response_columns" not in reasons:
        distinct_tuples = _distinct_index_tuple_count(
            engine,
            var_names=variable_names,
        )
        diagnostics["distinct_index_tuples"] = int(distinct_tuples)
        if distinct_tuples != cardinality:
            reasons.append("duplicate_or_incomplete_combinations")

        gi_row = engine.fetch_one(
            f"SELECT COUNT(DISTINCT {quote_ident('global_index')}), "
            f"MIN({quote_ident('global_index')}), "
            f"MAX({quote_ident('global_index')}) "
            f"FROM {quote_ident(engine.view_name)}"
        )
        distinct_gi = int(gi_row[0]) if gi_row else 0
        min_gi = gi_row[1] if gi_row else None
        max_gi = gi_row[2] if gi_row else None
        diagnostics["global_index"] = {
            "distinct": distinct_gi,
            "min": min_gi,
            "max": max_gi,
        }
        if not (
            distinct_gi == cardinality
            and min_gi == 0
            and max_gi == cardinality - 1
        ):
            reasons.append("global_index_incomplete")

        quality = _response_quality_counts(
            engine,
            objective_column=objective_column,
            util_columns=util_columns,
            max_examples=max_examples,
        )
        diagnostics["response_quality"] = quality
        if int(quality["analysis_not_ok"]["count"]) > 0:
            reasons.append("analysis_not_ok_rows")
        if int(quality["nonfinite_objective"]["count"]) > 0:
            reasons.append("nonfinite_objective")
        if any(
            int(item["count"]) > 0
            for item in quality["nonfinite_utilization"].values()
        ):
            reasons.append("nonfinite_utilization")

    # De-duplicate while preserving order.
    unique_reasons: list[str] = []
    for reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)

    ok = len(unique_reasons) == 0
    diagnostics["ok"] = ok
    if integrity_report is not None:
        diagnostics["integrity"] = {
            "ok": integrity_report.get("ok"),
            "overall_status": integrity_report.get("overall_status"),
            "uniqueness": integrity_report.get("uniqueness"),
        }
    if coverage_report is not None:
        diagnostics["coverage"] = {
            "ok": coverage_report.get("ok"),
            "observed_rows": coverage_report.get("observed_rows"),
            "is_full_enumeration": coverage_report.get("is_full_enumeration"),
        }

    return FactorialSensitivityEligibility(
        ok=ok,
        reasons=tuple(unique_reasons),
        diagnostics=diagnostics,
        response_specs=response_specs,
        variable_names=variable_names,
        level_counts=level_counts,
        cardinality=cardinality,
        invalid_utilization=invalid_utilization,
    )


# Absolute tolerance for the ST identity residual diagnostic.
IDENTITY_ABS_TOL = 1e-12


@dataclass(frozen=True)
class FactorialResponseResult:
    """Exact indices (or constant status) for one response."""

    spec: FactorialResponseSpec
    status: str
    mean: float
    total_variance: float
    factors: tuple[FactorSobolIndices, ...]


@dataclass(frozen=True)
class FactorialSensitivityComputeResult:
    """
    Out-of-core DuckDB exact factorial sensitivity result.

    ``query_count`` should be ``1 + 2 * len(variable_names)`` for the fused
    aggregation plan (one overall scan plus one S1 and one ST query per factor).
    """

    method: FactorialSensitivityMethodMeta
    responses: tuple[FactorialResponseResult, ...]
    variable_names: tuple[str, ...]
    query_count: int
    largest_complement_group_count: int
    identity_abs_tol: float = IDENTITY_ABS_TOL


def _sql_response_expression(spec: FactorialResponseSpec) -> str:
    source = quote_ident(spec.source_column)
    if spec.expression == EXPRESSION_COLUMN:
        return source
    if spec.expression == EXPRESSION_BINARY_FEASIBLE:
        return f"(CASE WHEN {source} <= 1.0 THEN 1.0 ELSE 0.0 END)"
    raise ValueError(f"unknown response expression: {spec.expression!r}")


def compute_factorial_sobol_indices(
    context: BruteForceDatasetContext,
    engine: BruteForceQueryEngine,
    *,
    response_specs: Sequence[FactorialResponseSpec] | None = None,
    eligibility: FactorialSensitivityEligibility | None = None,
) -> FactorialSensitivityComputeResult:
    """
    Compute exact S1/ST indices with fused out-of-core DuckDB aggregates.

    Does not write artifacts and does not re-check eligibility unless an
    eligibility payload is provided. Callers that publish results must run
    :func:`assess_factorial_sensitivity_eligibility` first.

    Parameters
    ----------
    context:
        Sealed dataset context (variable order / names).
    engine:
        Open DuckDB engine over the dataset (or a synthetic fixture view).
    response_specs:
        Responses to analyze. Defaults to the sealed
        ``y_p`` / ``util__*`` / ``feasible__*`` contract.
    eligibility:
        Optional prior eligibility result; must be ``ok`` when provided.

    Returns
    -------
    FactorialSensitivityComputeResult
        Per-response indices with query-count diagnostics.
    """
    if eligibility is not None and not eligibility.ok:
        raise ValueError(
            "cannot compute factorial Sobol indices for an ineligible dataset: "
            f"{list(eligibility.reasons)}"
        )

    specs = tuple(
        response_specs
        if response_specs is not None
        else (
            eligibility.response_specs
            if eligibility is not None
            else build_factorial_response_specs(
                analysis_constraint_names(context)
            )
        )
    )
    if not specs:
        raise ValueError("at least one response specification is required")

    variable_names = tuple(context.var_names)
    if not variable_names:
        raise ValueError("factorial sensitivity requires at least one variable")

    view = quote_ident(engine.view_name)
    query_count = 0
    largest_complement_group_count = 0

    response_exprs = [_sql_response_expression(spec) for spec in specs]
    aliases = [f"r{index}" for index in range(len(specs))]

    # --- 1) Overall mean / population variance for every response ----------
    overall_parts = ["COUNT(*) AS n"]
    for alias, expr in zip(aliases, response_exprs):
        overall_parts.append(f"AVG({expr}) AS \"mean__{alias}\"")
        overall_parts.append(f"VAR_POP({expr}) AS \"var__{alias}\"")
    overall_rows = engine.fetch_dicts(
        f"SELECT {', '.join(overall_parts)} FROM {view}",
        max_rows=1,
    )
    query_count += 1
    if not overall_rows:
        raise RuntimeError("overall factorial sensitivity query returned no row")
    overall = overall_rows[0]
    n_rows = int(overall["n"] or 0)
    if n_rows < 1:
        raise ValueError("dataset view is empty")

    means = [float(overall[f"mean__{alias}"]) for alias in aliases]
    variances = [float(overall[f"var__{alias}"] or 0.0) for alias in aliases]

    # Accumulators: per response, per variable.
    s1 = {alias: [0.0] * len(variable_names) for alias in aliases}
    st = {alias: [0.0] * len(variable_names) for alias in aliases}
    st_identity = {alias: [0.0] * len(variable_names) for alias in aliases}

    for axis, var_name in enumerate(variable_names):
        idx_col = quote_ident(f"idx__{var_name}")
        other_names = [name for name in variable_names if name != var_name]
        other_cols = [quote_ident(f"idx__{name}") for name in other_names]

        # --- 2) S1: group by Xi, count-weighted Var(E[Y|Xi]) ---------------
        mean_selects = [
            f"AVG({expr}) AS \"mean__{alias}\""
            for alias, expr in zip(aliases, response_exprs)
        ]
        overall_means = ", ".join(
            f'SUM(n_g * "mean__{alias}") / SUM(n_g) AS "mu__{alias}"'
            for alias in aliases
        )
        var_e_selects = ", ".join(
            (
                f'SUM(level.n_g * POWER(level."mean__{alias}" - overall."mu__{alias}", 2)) '
                f'/ SUM(level.n_g) AS "var_e__{alias}"'
            )
            for alias in aliases
        )
        level_sql = (
            "WITH level AS ("
            f"SELECT {idx_col} AS level_index, COUNT(*) AS n_g, "
            f"{', '.join(mean_selects)} "
            f"FROM {view} GROUP BY 1"
            "), overall AS ("
            f"SELECT {overall_means} FROM level"
            ") "
            f"SELECT {var_e_selects} FROM level CROSS JOIN overall"
        )
        s1_rows = engine.fetch_dicts(level_sql, max_rows=1)
        query_count += 1
        if not s1_rows:
            raise RuntimeError(f"S1 query failed for variable {var_name!r}")
        for alias in aliases:
            s1[alias][axis] = float(s1_rows[0][f"var_e__{alias}"] or 0.0)

        # --- 3) ST: group by X~i; E[Var(Y|X~i)] and Var(E[Y|X~i]) ----------
        if other_cols:
            group_clause = ", ".join(other_cols)
            group_select = f"{group_clause}, "
            group_by = f" GROUP BY {group_clause}"
        else:
            group_select = ""
            group_by = ""

        agg_selects: list[str] = []
        for alias, expr in zip(aliases, response_exprs):
            agg_selects.append(f"AVG({expr}) AS \"mean__{alias}\"")
            agg_selects.append(f"VAR_POP({expr}) AS \"var__{alias}\"")

        overall_means = ", ".join(
            f'SUM(n_g * "mean__{alias}") / SUM(n_g) AS "mu__{alias}"'
            for alias in aliases
        )
        st_selects = ", ".join(
            (
                f'(SUM(complement.n_g * complement."var__{alias}") / SUM(complement.n_g)) '
                f'AS "e_var__{alias}", '
                f'SUM(complement.n_g * POWER(complement."mean__{alias}" - overall."mu__{alias}", 2)) '
                f'/ SUM(complement.n_g) AS "var_e_comp__{alias}"'
            )
            for alias in aliases
        )
        complement_sql = (
            "WITH complement AS ("
            f"SELECT {group_select}COUNT(*) AS n_g, "
            f"{', '.join(agg_selects)} "
            f"FROM {view}{group_by}"
            "), overall AS ("
            f"SELECT {overall_means} FROM complement"
            ") "
            "SELECT MAX(complement.n_g) AS max_group_n, "
            f"{st_selects} "
            "FROM complement CROSS JOIN overall"
        )
        st_rows = engine.fetch_dicts(complement_sql, max_rows=1)
        query_count += 1
        if not st_rows:
            raise RuntimeError(f"ST query failed for variable {var_name!r}")
        st_row = st_rows[0]
        largest_complement_group_count = max(
            largest_complement_group_count,
            int(st_row.get("max_group_n") or 0),
        )
        for alias in aliases:
            st[alias][axis] = float(st_row[f"e_var__{alias}"] or 0.0)
            st_identity[alias][axis] = float(
                st_row[f"var_e_comp__{alias}"] or 0.0
            )

    responses: list[FactorialResponseResult] = []
    for alias, spec, mean, variance in zip(
        aliases, specs, means, variances, strict=True
    ):
        if variance == 0.0:
            responses.append(
                FactorialResponseResult(
                    spec=spec,
                    status=STATUS_CONSTANT,
                    mean=mean,
                    total_variance=0.0,
                    factors=(),
                )
            )
            continue

        factors: list[FactorSobolIndices] = []
        for axis, var_name in enumerate(variable_names):
            first = s1[alias][axis] / variance
            total = st[alias][axis] / variance
            via_complement = 1.0 - (st_identity[alias][axis] / variance)
            factors.append(
                FactorSobolIndices(
                    variable=var_name,
                    axis=axis,
                    first_order=float(first),
                    total_effect=float(total),
                    total_effect_via_complement_mean=float(via_complement),
                    interaction_gap=float(total - first),
                    identity_residual=float(total - via_complement),
                )
            )
        responses.append(
            FactorialResponseResult(
                spec=spec,
                status=STATUS_OK,
                mean=mean,
                total_variance=variance,
                factors=tuple(factors),
            )
        )

    expected_queries = 1 + 2 * len(variable_names)
    if query_count != expected_queries:
        raise RuntimeError(
            "unexpected factorial sensitivity query count: "
            f"got {query_count}, expected {expected_queries}"
        )

    analyzable = sum(1 for item in responses if item.status == STATUS_OK)
    if analyzable == 0:
        raise ValueError(
            "no non-constant responses available for factorial sensitivity"
        )

    return FactorialSensitivityComputeResult(
        method=default_method_meta(),
        responses=tuple(responses),
        variable_names=variable_names,
        query_count=query_count,
        largest_complement_group_count=largest_complement_group_count,
        identity_abs_tol=IDENTITY_ABS_TOL,
    )


# ---------------------------------------------------------------------------
# Publication (C7.4)
# ---------------------------------------------------------------------------

FACTORIAL_SENSITIVITY_DIRNAME = "factorial_sensitivity"
FACTORIAL_RESULTS_JSON = "results.json"
FACTORIAL_INDICES_CSV = "indices.csv"
FACTORIAL_SUMMARY_MD = "summary.md"
FACTORIAL_DIAGNOSTIC_JSON = "diagnostic.json"
FACTORIAL_SUCCESS_MARKER = "_SUCCESS"
FACTORIAL_SCHEMA_VERSION = 1

_CSV_FIELDNAMES: tuple[str, ...] = (
    "response_family",
    "response_name",
    "status",
    "variable",
    "axis",
    "first_order",
    "total_effect",
    "interaction_gap",
    "identity_residual",
    "mean",
    "total_variance",
)


def factorial_sensitivity_output_dir(experiment_dir: Path) -> Path:
    """Return ``results/sensitivity-analysis/factorial/{id}``."""
    from core.execution_core.output_layout import (
        factorial_publication_dir,
        require_results_publication,
        resolve_results_root,
    )

    return require_results_publication(
        factorial_publication_dir(
            resolve_results_root(experiment_dir),
            Path(experiment_dir).name,
        )
    )


def _resolve_factorial_output_dir(
    experiment_dir: Path,
    output_dir: Path | None,
) -> Path:
    experiment_dir = Path(experiment_dir)
    if output_dir is None:
        return factorial_sensitivity_output_dir(experiment_dir)
    out_dir = Path(output_dir)
    from core.execution_core.output_layout import require_results_publication

    out_dir = require_results_publication(out_dir, error_type=ValueError)
    dataset_root = (experiment_dir / "dataset").resolve()
    resolved = out_dir.resolve()
    if resolved == dataset_root or dataset_root in resolved.parents:
        raise ValueError(f"output_dir must not be inside dataset/: {out_dir}")
    return out_dir


def _method_meta_dict(meta: FactorialSensitivityMethodMeta) -> dict[str, Any]:
    return {
        "method_id": meta.method_id,
        "method_title": meta.method_title,
        "variance_ddof": int(meta.variance_ddof),
        "measure": meta.measure,
        "reference": meta.reference,
        "doi": meta.doi,
        "publishes_sampling_ci": bool(meta.publishes_sampling_ci),
        "publishes_second_order": bool(meta.publishes_second_order),
        "notes": meta.notes,
    }


def build_factorial_sensitivity_results(
    context: BruteForceDatasetContext,
    eligibility: FactorialSensitivityEligibility,
    compute: FactorialSensitivityComputeResult,
    *,
    options: DatasetReportOptions | None = None,
) -> dict[str, Any]:
    """
    Build the authoritative ``results.json`` document.
    """
    options = options or DatasetReportOptions()
    responses_payload: list[dict[str, Any]] = []
    for item in compute.responses:
        factors_payload: list[dict[str, Any]] = []
        for factor in item.factors:
            factors_payload.append(
                {
                    "variable": factor.variable,
                    "axis": int(factor.axis),
                    "first_order": float(factor.first_order),
                    "total_effect": float(factor.total_effect),
                    "total_effect_via_complement_mean": float(
                        factor.total_effect_via_complement_mean
                    ),
                    "interaction_gap": float(factor.interaction_gap),
                    "identity_residual": float(factor.identity_residual),
                }
            )
        responses_payload.append(
            {
                "family": item.spec.family,
                "name": item.spec.name,
                "source_column": item.spec.source_column,
                "expression": item.spec.expression,
                "status": item.status,
                "mean": float(item.mean),
                "total_variance": float(item.total_variance),
                "factors": factors_payload,
            }
        )

    return normalize_report_value(
        merge_constraint_provenance(
            {
            "schema_version": FACTORIAL_SCHEMA_VERSION,
            "kind": "factorial_sensitivity",
            "ok": True,
            "overall_status": "success",
            "method": _method_meta_dict(compute.method),
            "dataset": {
                "experiment_id": context.manifest.experiment_id,
                "workflow_type": context.manifest.workflow_type,
                "problem_id": context.problem_id,
                "slab_type": context.slab_type,
                "is_full_enumeration": bool(context.is_full_enumeration),
                "full_catalogue_rows": int(context.full_catalogue_rows),
                "planned_rows": int(context.planned_rows),
                "var_names": list(context.var_names),
                "constraint_names": list(context.constraint_names),
                "opt_invalid_utilization": float(
                    context.run_config.invalid_utilization
                ),
            },
            "eligibility": {
                "ok": bool(eligibility.ok),
                "reasons": list(eligibility.reasons),
                "cardinality": int(eligibility.cardinality),
                "level_counts": list(eligibility.level_counts),
                "variable_names": list(eligibility.variable_names),
            },
            "computation": {
                "query_count": int(compute.query_count),
                "largest_complement_group_count": int(
                    compute.largest_complement_group_count
                ),
                "identity_abs_tol": float(compute.identity_abs_tol),
                "variable_names": list(compute.variable_names),
            },
            "responses": responses_payload,
            "tool_metadata": {
                "duckdb_threads": int(options.duckdb_threads),
                "duckdb_memory_limit": options.duckdb_memory_limit,
                "duckdb_temp_directory": (
                    None
                    if options.duckdb_temp_directory is None
                    else str(options.duckdb_temp_directory)
                ),
                "max_violation_examples": int(options.max_violation_examples),
            },
            },
            sealed=context.constraint_names,
            selected=analysis_constraint_names(context),
        )
    )


def build_factorial_sensitivity_diagnostic(
    *,
    context: BruteForceDatasetContext | None,
    eligibility: FactorialSensitivityEligibility | None,
    error: str | None = None,
    overall_status: str = "eligibility_failure",
) -> dict[str, Any]:
    """
    Build a bounded failure diagnostic (no success marker).
    """
    payload: dict[str, Any] = {
        "schema_version": FACTORIAL_SCHEMA_VERSION,
        "kind": "factorial_sensitivity_diagnostic",
        "ok": False,
        "overall_status": overall_status,
        "error": error,
    }
    if context is not None:
        payload["dataset"] = {
            "experiment_id": context.manifest.experiment_id,
            "problem_id": context.problem_id,
            "is_full_enumeration": bool(context.is_full_enumeration),
            "full_catalogue_rows": int(context.full_catalogue_rows),
            "planned_rows": int(context.planned_rows),
        }
        payload.update(
            merge_constraint_provenance(
                {},
                sealed=context.constraint_names,
                selected=analysis_constraint_names(context),
            )
        )
    if eligibility is not None:
        payload["eligibility"] = {
            "ok": bool(eligibility.ok),
            "reasons": list(eligibility.reasons),
            "diagnostics": dict(eligibility.diagnostics),
        }
    return normalize_report_value(payload)


def render_factorial_indices_csv(results: Mapping[str, Any]) -> str:
    """
    Render long-form ``indices.csv`` from a results document.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=list(_CSV_FIELDNAMES),
        lineterminator="\n",
    )
    writer.writeheader()
    for response in results.get("responses") or []:
        status = str(response.get("status"))
        base = {
            "response_family": response.get("family"),
            "response_name": response.get("name"),
            "status": status,
            "mean": response.get("mean"),
            "total_variance": response.get("total_variance"),
        }
        factors = list(response.get("factors") or [])
        if status == STATUS_CONSTANT or not factors:
            row = {
                **base,
                "variable": "",
                "axis": "",
                "first_order": "",
                "total_effect": "",
                "interaction_gap": "",
                "identity_residual": "",
            }
            writer.writerow(row)
            continue
        for factor in factors:
            writer.writerow(
                {
                    **base,
                    "variable": factor.get("variable"),
                    "axis": factor.get("axis"),
                    "first_order": factor.get("first_order"),
                    "total_effect": factor.get("total_effect"),
                    "interaction_gap": factor.get("interaction_gap"),
                    "identity_residual": factor.get("identity_residual"),
                }
            )
    text = buf.getvalue()
    if not text.endswith("\n"):
        text += "\n"
    return text


def render_factorial_summary_markdown(results: Mapping[str, Any]) -> str:
    """
    Render ``summary.md`` from a results or diagnostic document.
    """
    lines = [
        "# Exact full-factorial sensitivity",
        "",
        f"- overall_status: {results.get('overall_status')}",
        f"- ok: {results.get('ok')}",
        f"- schema_version: {results.get('schema_version')}",
        f"- kind: {results.get('kind')}",
    ]
    method = results.get("method") or {}
    if method:
        lines.extend(
            [
                f"- method_id: {method.get('method_id')}",
                f"- doi: {method.get('doi')}",
                f"- variance_ddof: {method.get('variance_ddof')}",
            ]
        )
    dataset = results.get("dataset") or {}
    if dataset:
        lines.extend(
            [
                f"- experiment_id: {dataset.get('experiment_id')}",
                f"- problem_id: {dataset.get('problem_id')}",
                f"- is_full_enumeration: {dataset.get('is_full_enumeration')}",
                f"- full_catalogue_rows: {dataset.get('full_catalogue_rows')}",
            ]
        )
    eligibility = results.get("eligibility") or {}
    if eligibility and not results.get("ok"):
        lines.extend(
            [
                "",
                "## Eligibility failure",
                f"- reasons: {eligibility.get('reasons')}",
            ]
        )
    computation = results.get("computation") or {}
    if computation:
        lines.extend(
            [
                "",
                "## Computation",
                f"- query_count: {computation.get('query_count')}",
                f"- largest_complement_group_count: "
                f"{computation.get('largest_complement_group_count')}",
            ]
        )
    responses = results.get("responses") or []
    if responses:
        lines.extend(["", "## Responses"])
        for response in responses:
            lines.append(
                f"- `{response.get('name')}` "
                f"({response.get('family')}): status={response.get('status')}, "
                f"variance={response.get('total_variance')}"
            )
            for factor in response.get("factors") or []:
                lines.append(
                    f"  - {factor.get('variable')}: "
                    f"S1={factor.get('first_order')}, "
                    f"ST={factor.get('total_effect')}"
                )
    if results.get("error"):
        lines.extend(["", f"- error: {results.get('error')}"])
    lines.extend(
        [
            "",
            "See `results.json` for the full structured document.",
            "",
        ]
    )
    return "\n".join(lines)


def _remove_success_marker(out_dir: Path) -> None:
    marker = out_dir / FACTORIAL_SUCCESS_MARKER
    if marker.is_file():
        marker.unlink()


def write_factorial_sensitivity_artifacts(
    experiment_dir: Path,
    results: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
    formats: Sequence[str] = ("json", "csv", "md"),
    success: bool = True,
) -> dict[str, Path]:
    """
    Atomically publish factorial sensitivity artifacts.

    When ``success`` is true, write ``results.json`` / ``indices.csv`` /
    ``summary.md`` then ``_SUCCESS``. When false, withhold ``_SUCCESS``,
    remove a stale marker if present, and publish ``diagnostic.json`` (plus
    optional markdown summary) instead.
    """
    experiment_dir = Path(experiment_dir)
    selected = {str(item).lower() for item in formats}
    unknown = selected - {"json", "csv", "md"}
    if unknown:
        raise ValueError(f"unsupported factorial formats: {sorted(unknown)}")
    if not selected:
        raise ValueError("formats must include at least one of: json, csv, md")

    out_dir = _resolve_factorial_output_dir(experiment_dir, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    finalized = normalize_report_value(results)
    payloads: dict[str, str] = {}
    if success:
        if "json" in selected:
            payloads["json"] = dumps_report_json(finalized)
        if "csv" in selected:
            payloads["csv"] = render_factorial_indices_csv(finalized)
        if "md" in selected:
            payloads["md"] = render_factorial_summary_markdown(finalized)
    else:
        _remove_success_marker(out_dir)
        if "json" in selected:
            payloads["diagnostic"] = dumps_report_json(finalized)
        if "md" in selected:
            payloads["md"] = render_factorial_summary_markdown(finalized)

    # Serialize fully before any write; then publish.
    written: dict[str, Path] = {}
    if "json" in payloads and success:
        path = out_dir / FACTORIAL_RESULTS_JSON
        write_text_atomic(path, payloads["json"])
        written["json"] = path
    if "csv" in payloads:
        path = out_dir / FACTORIAL_INDICES_CSV
        write_text_atomic(path, payloads["csv"])
        written["csv"] = path
    if "md" in payloads:
        path = out_dir / FACTORIAL_SUMMARY_MD
        write_text_atomic(path, payloads["md"])
        written["md"] = path
    if "diagnostic" in payloads:
        path = out_dir / FACTORIAL_DIAGNOSTIC_JSON
        write_text_atomic(path, payloads["diagnostic"])
        written["diagnostic"] = path

    if success:
        marker = out_dir / FACTORIAL_SUCCESS_MARKER
        write_text_atomic(marker, "ok\n")
        written["success"] = marker
    return written


def run_factorial_sensitivity_analysis(
    context: BruteForceDatasetContext,
    engine: BruteForceQueryEngine,
    *,
    options: DatasetReportOptions | None = None,
    include_integrity: bool = True,
) -> tuple[dict[str, Any], bool]:
    """
    Run eligibility + computation and return ``(document, success)``.

    On eligibility or compute failure, ``document`` is a diagnostic payload and
    ``success`` is false. Does not write artifacts.
    """
    options = options or DatasetReportOptions()
    eligibility = assess_factorial_sensitivity_eligibility(
        context,
        engine,
        options=options,
        include_integrity=include_integrity,
    )
    if not eligibility.ok:
        return (
            build_factorial_sensitivity_diagnostic(
                context=context,
                eligibility=eligibility,
                overall_status="eligibility_failure",
            ),
            False,
        )
    try:
        compute = compute_factorial_sobol_indices(
            context,
            engine,
            eligibility=eligibility,
        )
    except ValueError as exc:
        return (
            build_factorial_sensitivity_diagnostic(
                context=context,
                eligibility=eligibility,
                error=str(exc),
                overall_status="scientific_failure",
            ),
            False,
        )
    return (
        build_factorial_sensitivity_results(
            context,
            eligibility,
            compute,
            options=options,
        ),
        True,
    )


def run_and_write_factorial_sensitivity(
    experiment_dir: Path,
    *,
    options: DatasetReportOptions | None = None,
    output_dir: Path | None = None,
    formats: Sequence[str] = ("json", "csv", "md"),
    selected_constraint_names: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Path], bool]:
    """
    Open a sealed experiment, analyze, and publish factorial sensitivity artifacts.
    """
    options = options or DatasetReportOptions()
    experiment_dir = Path(experiment_dir)
    context = open_brute_force_dataset(experiment_dir)
    if selected_constraint_names is not None:
        context = attach_selected_constraint_names(
            context, selected_constraint_names
        )
    with open_query_engine(context, options=options) as engine:
        document, success = run_factorial_sensitivity_analysis(
            context,
            engine,
            options=options,
            include_integrity=True,
        )
    written = write_factorial_sensitivity_artifacts(
        experiment_dir,
        document,
        output_dir=output_dir,
        formats=formats,
        success=success,
    )
    return document, written, success
