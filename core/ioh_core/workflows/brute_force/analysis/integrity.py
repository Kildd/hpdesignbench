"""
Structural and scientific integrity analysis for sealed brute-force datasets.

Calls B8 ``validate_dataset()`` first. Scientific DuckDB scans run only when
structural validation succeeds. Report writing is deferred to C3.8.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.ioh_core.workflows.brute_force.analysis.engine import (
    BruteForceQueryEngine,
    quote_ident,
    quote_string,
)
from core.ioh_core.workflows.brute_force.analysis.models import (
    BruteForceDatasetContext,
    DatasetReportOptions,
)
from core.ioh_core.workflows.brute_force.analysis.serialization import (
    normalize_report_value,
)
from core.ioh_core.workflows.brute_force.validation import (
    DatasetValidationResult,
    validate_dataset,
)

_ZERO_ATOL = 0.0


def _rule(
    *,
    ok: bool,
    count: int,
    examples: Sequence[int] | None = None,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": bool(ok),
        "count": int(count),
        "examples": [int(x) for x in (examples or ())],
    }
    if detail:
        payload["detail"] = dict(detail)
    return payload


def _sql_finite_close(left: str, right: str, *, rtol: float) -> str:
    """
    SQL predicate approximating ``math.isclose(..., rel_tol=rtol, abs_tol=0)``.
    """
    return (
        f"(isfinite({left}) AND isfinite({right}) AND "
        f"abs(({left}) - ({right})) <= ({rtol}) * "
        f"greatest(abs({left}), abs({right})))"
    )


def _sql_is_sentinel(util_expr: str, *, sentinel: float, atol: float) -> str:
    return (
        f"(isfinite({util_expr}) AND "
        f"abs(({util_expr}) - ({sentinel})) <= ({atol}))"
    )


def _structural_section(result: DatasetValidationResult) -> dict[str, Any]:
    return {
        "ok": bool(result.ok),
        "errors": list(result.errors),
        "schema_hash": result.schema_hash,
        "planned_rows": int(result.planned_rows),
        "observed_rows": int(result.observed_rows),
        "part_count": len(result.parts),
        "parts_ok": sum(1 for part in result.parts if part.ok),
        "ignored_files": list(result.ignored_files),
    }


def _dataset_identity(
    context: BruteForceDatasetContext,
    *,
    structural: DatasetValidationResult,
    duckdb_rows: int | None,
) -> dict[str, Any]:
    observed = int(structural.observed_rows)
    full = int(context.full_catalogue_rows)
    coverage = (float(observed) / float(full)) if full > 0 else 0.0
    return {
        "full_catalogue_rows": full,
        "planned_rows": int(context.planned_rows),
        "planned_start_index": int(context.planned_start_index),
        "planned_stop_index": int(context.planned_stop_index),
        "b8_observed_rows": observed,
        "duckdb_row_count": duckdb_rows,
        "duckdb_matches_b8": (
            None if duckdb_rows is None else int(duckdb_rows) == observed
        ),
        "is_full_enumeration": bool(context.is_full_enumeration),
        "coverage_fraction": float(coverage),
    }


def _numeric_field_names(schema_doc: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for field in schema_doc.get("fields", []):
        if not isinstance(field, Mapping):
            continue
        name = field.get("name")
        ftype = str(field.get("type", "")).lower()
        if not isinstance(name, str) or not name:
            continue
        if "double" in ftype or "float" in ftype:
            names.append(name)
    return tuple(names)


def _register_catalogue_tables(
    engine: BruteForceQueryEngine,
    context: BruteForceDatasetContext,
) -> None:
    """
    Register small sealed catalogue / lookup maps as temporary DuckDB tables.
    """
    model = context.problem_info["model"]
    conn = engine.connection

    for var_name in context.var_names:
        meta = model[var_name]
        values = list(meta["values"])
        table = f"cat_idx__{var_name}"
        conn.execute(f"DROP TABLE IF EXISTS {quote_ident(table)}")
        conn.execute(
            f"CREATE TEMP TABLE {quote_ident(table)} "
            "(idx INTEGER, val VARCHAR)"
        )
        lower = int(meta["lb"])
        upper = int(meta["ub"])
        rows = [
            (index, None if values[index] is None else str(values[index]))
            for index in range(lower, upper + 1)
        ]
        if rows:
            conn.executemany(
                f"INSERT INTO {quote_ident(table)} VALUES (?, ?)",
                rows,
            )

    for name, meta in model.items():
        if meta.get("role") != "lookup":
            continue
        key = str(meta["lookup_key"])
        key_vals = list(model[key]["values"])
        lookup_vals = list(meta["values"])
        table = f"cat_lookup__{name}"
        conn.execute(f"DROP TABLE IF EXISTS {quote_ident(table)}")
        conn.execute(
            f"CREATE TEMP TABLE {quote_ident(table)} "
            "(key_val VARCHAR, val VARCHAR)"
        )
        rows = [
            (
                None if key_vals[index] is None else str(key_vals[index]),
                None if lookup_vals[index] is None else str(lookup_vals[index]),
            )
            for index in range(len(key_vals))
        ]
        if rows:
            conn.executemany(
                f"INSERT INTO {quote_ident(table)} VALUES (?, ?)",
                rows,
            )


def _fetch_examples(
    engine: BruteForceQueryEngine,
    *,
    where_sql: str,
    limit: int,
) -> list[int]:
    view = quote_ident(engine.view_name)
    rows = engine.fetch_rows(
        f"SELECT {quote_ident('global_index')} FROM {view} "
        f"WHERE {where_sql} "
        f"ORDER BY {quote_ident('global_index')} "
        f"LIMIT {int(limit)}",
        max_rows=limit,
    )
    return [int(row[0]) for row in rows]


def _finalize_rule(
    engine: BruteForceQueryEngine,
    *,
    count: int,
    where_sql: str,
    max_examples: int,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    examples: list[int] = []
    if count > 0:
        examples = _fetch_examples(
            engine, where_sql=where_sql, limit=max_examples
        )
    return _rule(ok=count == 0, count=count, examples=examples, detail=detail)


def run_integrity_analysis(
    context: BruteForceDatasetContext,
    engine: BruteForceQueryEngine | None = None,
    *,
    options: DatasetReportOptions | None = None,
    structural: DatasetValidationResult | None = None,
) -> dict[str, Any]:
    """
    Run B8 structural validation plus scientific integrity scans.

    Parameters
    ----------
    context : BruteForceDatasetContext
        Sealed dataset context.
    engine : BruteForceQueryEngine or None, optional
        Open DuckDB engine over the same dataset. Required only when
        structural validation succeeds.
    options : DatasetReportOptions or None, optional
        Example-bound options.
    structural : DatasetValidationResult or None, optional
        Precomputed B8 result; otherwise ``validate_dataset`` is called.

    Returns
    -------
    dict
        JSON-compatible integrity section payload (normalized).
    """
    from core.ioh_core.workflows.brute_force.analysis.engine import (
        SCIENTIFIC_RTOL,
        SENTINEL_ATOL,
    )

    options = options or DatasetReportOptions()
    max_examples = int(options.max_violation_examples)
    rtol = float(engine.scientific_rtol) if engine is not None else SCIENTIFIC_RTOL
    sentinel_atol = (
        float(engine.sentinel_atol) if engine is not None else SENTINEL_ATOL
    )
    sentinel = float(context.run_config.invalid_utilization)
    alpha = float(context.opt_penalty_alpha)
    beta = float(context.opt_penalty_beta)
    expected_sentinel_violation = max(0.0, sentinel - 1.0)
    expected_sentinel_penalty = alpha * (expected_sentinel_violation**beta)

    if structural is None:
        structural = validate_dataset(context.experiment_dir)

    report: dict[str, Any] = {
        "structural": _structural_section(structural),
        "tolerances": {
            "scientific_rtol": rtol,
            "sentinel_atol": sentinel_atol,
            "opt_invalid_utilization": sentinel,
            "opt_penalty_alpha": alpha,
            "opt_penalty_beta": beta,
        },
        "dataset_identity": None,
        "rules": {},
        "numeric_quality": {},
        "uniqueness": None,
        "scientific_ran": False,
        "scientific_ok": False,
        "ok": False,
        "overall_status": "structural_failure",
    }

    if not structural.ok:
        report["dataset_identity"] = _dataset_identity(
            context, structural=structural, duckdb_rows=None
        )
        report["scientific"] = {
            "skipped": True,
            "reason": "structural_validation_failed",
        }
        return normalize_report_value(report)

    if engine is None:
        raise ValueError(
            "engine is required when structural validation succeeds"
        )

    view = quote_ident(engine.view_name)
    duckdb_rows = engine.count_rows()
    identity = _dataset_identity(
        context, structural=structural, duckdb_rows=duckdb_rows
    )
    report["dataset_identity"] = identity
    report["scientific_ran"] = True
    report["scientific"] = {"skipped": False}

    rules: dict[str, Any] = {}

    aggregates: list[str] = [
        "COUNT(*) AS row_count",
        f"COUNT(DISTINCT {quote_ident('global_index')}) AS distinct_global_index",
        f"MIN({quote_ident('global_index')}) AS min_global_index",
        f"MAX({quote_ident('global_index')}) AS max_global_index",
        (
            "SUM(CASE WHEN "
            f"{quote_ident('feasible')} IS DISTINCT FROM "
            f"({quote_ident('analysis_ok')} AND {quote_ident('constraint_feasible')}) "
            "THEN 1 ELSE 0 END) AS feasible_composite_bad"
        ),
        (
            "SUM(CASE WHEN "
            f"{quote_ident('analysis_ok')} AND ("
            f"{quote_ident('analysis_error_stage')} IS NOT NULL OR "
            f"{quote_ident('analysis_error_message')} IS NOT NULL OR "
            f"{quote_ident('analysis_error_exception_type')} IS NOT NULL OR "
            f"{quote_ident('analysis_error_json')} IS NOT NULL"
            ") THEN 1 ELSE 0 END) AS success_has_error"
        ),
        (
            "SUM(CASE WHEN "
            f"NOT {quote_ident('analysis_ok')} AND "
            f"{quote_ident('analysis_error_stage')} IS NULL "
            "THEN 1 ELSE 0 END) AS error_missing_stage"
        ),
        (
            "SUM(CASE WHEN "
            f"len({quote_ident('remapped_utilizations')}) > 0 AND "
            f"{quote_ident('analysis_ok')} "
            "THEN 1 ELSE 0 END) AS remapped_with_ok"
        ),
        (
            "SUM(CASE WHEN "
            f"len({quote_ident('remapped_utilizations')}) > 0 AND "
            f"({quote_ident('analysis_error_stage')} IS DISTINCT FROM "
            f"{quote_string('nonfinite_utilization')}) "
            "THEN 1 ELSE 0 END) AS remapped_wrong_stage"
        ),
        (
            "SUM(CASE WHEN "
            f"{quote_ident('cost_components_missing')} IS NULL "
            "THEN 1 ELSE 0 END) AS missing_cost_null"
        ),
        (
            "SUM(CASE WHEN NOT ("
            + _sql_finite_close(
                quote_ident("penalty_factor"),
                f"(1.0 + {quote_ident('penalty_sum')})",
                rtol=rtol,
            )
            + ") THEN 1 ELSE 0 END) AS penalty_factor_bad"
        ),
        (
            "SUM(CASE WHEN isfinite("
            f"{quote_ident('y')}) AND isfinite({quote_ident('penalty_factor')}) "
            "AND isfinite("
            f"{quote_ident('y_p')}) AND NOT ("
            + _sql_finite_close(
                quote_ident("y_p"),
                f"({quote_ident('y')} * {quote_ident('penalty_factor')})",
                rtol=rtol,
            )
            + ") THEN 1 ELSE 0 END) AS y_p_identity_bad"
        ),
    ]

    space = context.index_space
    decoder_bad_parts: list[str] = []
    bounds_bad_parts: list[str] = []
    for var_name, lower, upper, size, stride in zip(
        space.var_names,
        space.lower_bounds,
        space.upper_bounds,
        space.sizes,
        space.strides,
    ):
        col = quote_ident(f"idx__{var_name}")
        expected = (
            f"(({int(lower)}) + "
            f"CAST((CAST({quote_ident('global_index')} AS BIGINT) // {int(stride)}) "
            f"% {int(size)} AS INTEGER))"
        )
        decoder_bad_parts.append(f"{col} IS DISTINCT FROM {expected}")
        bounds_bad_parts.append(
            f"({col} IS NULL OR {col} < {int(lower)} OR {col} > {int(upper)})"
        )

    if decoder_bad_parts:
        aggregates.append(
            "SUM(CASE WHEN ("
            + " OR ".join(decoder_bad_parts)
            + ") THEN 1 ELSE 0 END) AS decoder_bad"
        )
        aggregates.append(
            "SUM(CASE WHEN ("
            + " OR ".join(bounds_bad_parts)
            + ") THEN 1 ELSE 0 END) AS index_bounds_bad"
        )
    else:
        aggregates.append("0 AS decoder_bad")
        aggregates.append("0 AS index_bounds_bad")

    required_null_checks = [
        "global_index",
        "shard_index",
        "problem_id",
        "evaluation_time_s",
        "analysis_ok",
        "constraint_feasible",
        "feasible",
        "y",
        "y_p",
        "penalty_sum",
        "penalty_factor",
        "remapped_utilizations",
        "cost_components_missing",
    ]
    for name in context.var_names:
        required_null_checks.append(f"idx__{name}")
    required_expr = " OR ".join(
        f"{quote_ident(name)} IS NULL" for name in required_null_checks
    )
    aggregates.append(
        f"SUM(CASE WHEN ({required_expr}) THEN 1 ELSE 0 END) AS required_null_bad"
    )

    constraint_feasible_terms: list[str] = []
    util_identity_parts: list[str] = []
    penalty_identity_parts: list[str] = []
    triplet_parts: list[str] = []
    for cname in context.constraint_names:
        util = quote_ident(f"util__{cname}")
        viol = quote_ident(f"violation__{cname}")
        pen = quote_ident(f"penalty__{cname}")
        triplet_parts.append(
            f"(CAST({util} IS NULL AS INTEGER) + "
            f"CAST({viol} IS NULL AS INTEGER) + "
            f"CAST({pen} IS NULL AS INTEGER)) IN (1, 2)"
        )
        is_sentinel = _sql_is_sentinel(
            util, sentinel=sentinel, atol=sentinel_atol
        )
        expected_viol = f"greatest(0.0, ({util}) - 1.0)"
        expected_pen = (
            f"(({alpha}) * power(greatest(0.0, ({util}) - 1.0), {beta}))"
        )
        util_identity_parts.append(
            f"({util} IS NOT NULL AND {viol} IS NOT NULL AND NOT ("
            f"({is_sentinel} AND "
            + _sql_finite_close(
                viol, str(expected_sentinel_violation), rtol=rtol
            )
            + f") OR ((NOT {is_sentinel}) AND "
            + _sql_finite_close(viol, expected_viol, rtol=rtol)
            + ")))"
        )
        penalty_identity_parts.append(
            f"({util} IS NOT NULL AND {pen} IS NOT NULL AND NOT ("
            f"({is_sentinel} AND "
            + _sql_finite_close(
                pen, str(expected_sentinel_penalty), rtol=rtol
            )
            + f") OR ((NOT {is_sentinel}) AND "
            + _sql_finite_close(pen, expected_pen, rtol=rtol)
            + ")))"
        )
        constraint_feasible_terms.append(
            f"({viol} IS NULL OR (isfinite({viol}) AND {viol} <= {_ZERO_ATOL}))"
        )

    if triplet_parts:
        aggregates.append(
            "SUM(CASE WHEN ("
            + " OR ".join(triplet_parts)
            + ") THEN 1 ELSE 0 END) AS constraint_triplet_bad"
        )
        aggregates.append(
            "SUM(CASE WHEN ("
            + " OR ".join(util_identity_parts)
            + ") THEN 1 ELSE 0 END) AS util_violation_identity_bad"
        )
        aggregates.append(
            "SUM(CASE WHEN ("
            + " OR ".join(penalty_identity_parts)
            + ") THEN 1 ELSE 0 END) AS util_penalty_identity_bad"
        )
        expected_cf = " AND ".join(constraint_feasible_terms)
        aggregates.append(
            "SUM(CASE WHEN "
            f"{quote_ident('constraint_feasible')} IS DISTINCT FROM ({expected_cf}) "
            "THEN 1 ELSE 0 END) AS constraint_feasible_bad"
        )
    else:
        aggregates.extend(
            [
                "0 AS constraint_triplet_bad",
                "0 AS util_violation_identity_bad",
                "0 AS util_penalty_identity_bad",
                "0 AS constraint_feasible_bad",
            ]
        )

    numeric_cols = _numeric_field_names(context.schema_doc)
    for col_name in numeric_cols:
        col = quote_ident(col_name)
        safe = col_name.replace("`", "")
        aggregates.append(
            f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS "
            f'"nq_null__{safe}"'
        )
        aggregates.append(
            f"SUM(CASE WHEN isnan({col}) THEN 1 ELSE 0 END) AS "
            f'"nq_nan__{safe}"'
        )
        aggregates.append(
            f"SUM(CASE WHEN isinf({col}) AND {col} > 0 THEN 1 ELSE 0 END) AS "
            f'"nq_posinf__{safe}"'
        )
        aggregates.append(
            f"SUM(CASE WHEN isinf({col}) AND {col} < 0 THEN 1 ELSE 0 END) AS "
            f'"nq_neginf__{safe}"'
        )

    agg_sql = f"SELECT {', '.join(aggregates)} FROM {view}"
    agg_rows = engine.fetch_dicts(agg_sql, max_rows=1)
    if not agg_rows:
        raise RuntimeError("integrity aggregate query returned no row")
    agg = agg_rows[0]

    def _count(key: str) -> int:
        return int(agg.get(key) or 0)

    row_count = _count("row_count")
    distinct_gi = _count("distinct_global_index")
    min_gi = agg.get("min_global_index")
    max_gi = agg.get("max_global_index")
    planned_start = int(context.planned_start_index)
    planned_stop = int(context.planned_stop_index)
    planned_rows = int(context.planned_rows)

    duplicate_gi = max(0, row_count - distinct_gi)
    if duplicate_gi:
        rules["duplicate_global_index"] = _finalize_rule(
            engine,
            count=duplicate_gi,
            where_sql=(
                f"{quote_ident('global_index')} IN ("
                f"SELECT {quote_ident('global_index')} FROM {view} "
                f"GROUP BY 1 HAVING COUNT(*) > 1)"
            ),
            max_examples=max_examples,
        )
    else:
        rules["duplicate_global_index"] = _rule(ok=True, count=0)

    coverage_complete = (
        row_count == planned_rows
        and distinct_gi == planned_rows
        and min_gi == planned_start
        and max_gi == planned_stop - 1
        and identity["duckdb_matches_b8"] is True
    )
    rules["global_index_coverage"] = _rule(
        ok=coverage_complete,
        count=0 if coverage_complete else 1,
        examples=[],
        detail={
            "row_count": row_count,
            "distinct_global_index": distinct_gi,
            "min_global_index": min_gi,
            "max_global_index": max_gi,
            "planned_start_index": planned_start,
            "planned_stop_index": planned_stop,
            "planned_rows": planned_rows,
        },
    )

    decoder_count = _count("decoder_bad")
    rules["candidate_decoder"] = _finalize_rule(
        engine,
        count=decoder_count,
        where_sql=(
            "(" + " OR ".join(decoder_bad_parts) + ")"
            if decoder_bad_parts
            else "FALSE"
        ),
        max_examples=max_examples,
    )

    uniqueness_ok = coverage_complete and decoder_count == 0 and duplicate_gi == 0
    report["uniqueness"] = {
        "ok": uniqueness_ok,
        "direct_tuple_distinct_scan": "skipped" if uniqueness_ok else "not_run",
        "reason": (
            "complete_global_index_coverage_and_decoder_invariant"
            if uniqueness_ok
            else "uniqueness_proof_failed"
        ),
    }

    rule_specs: list[tuple[str, str, str]] = [
        (
            "index_bounds",
            "index_bounds_bad",
            (
                "(" + " OR ".join(bounds_bad_parts) + ")"
                if bounds_bad_parts
                else "FALSE"
            ),
        ),
        ("required_nulls", "required_null_bad", f"({required_expr})"),
        (
            "feasible_composite",
            "feasible_composite_bad",
            (
                f"{quote_ident('feasible')} IS DISTINCT FROM "
                f"({quote_ident('analysis_ok')} AND "
                f"{quote_ident('constraint_feasible')})"
            ),
        ),
        (
            "success_has_error",
            "success_has_error",
            (
                f"{quote_ident('analysis_ok')} AND ("
                f"{quote_ident('analysis_error_stage')} IS NOT NULL OR "
                f"{quote_ident('analysis_error_message')} IS NOT NULL OR "
                f"{quote_ident('analysis_error_exception_type')} IS NOT NULL OR "
                f"{quote_ident('analysis_error_json')} IS NOT NULL)"
            ),
        ),
        (
            "error_missing_stage",
            "error_missing_stage",
            (
                f"NOT {quote_ident('analysis_ok')} AND "
                f"{quote_ident('analysis_error_stage')} IS NULL"
            ),
        ),
        (
            "remapped_with_analysis_ok",
            "remapped_with_ok",
            (
                f"len({quote_ident('remapped_utilizations')}) > 0 AND "
                f"{quote_ident('analysis_ok')}"
            ),
        ),
        (
            "remapped_wrong_error_stage",
            "remapped_wrong_stage",
            (
                f"len({quote_ident('remapped_utilizations')}) > 0 AND "
                f"({quote_ident('analysis_error_stage')} IS DISTINCT FROM "
                f"{quote_string('nonfinite_utilization')})"
            ),
        ),
        (
            "cost_components_missing_null",
            "missing_cost_null",
            f"{quote_ident('cost_components_missing')} IS NULL",
        ),
        (
            "penalty_factor_identity",
            "penalty_factor_bad",
            "NOT ("
            + _sql_finite_close(
                quote_ident("penalty_factor"),
                f"(1.0 + {quote_ident('penalty_sum')})",
                rtol=rtol,
            )
            + ")",
        ),
        (
            "y_p_identity",
            "y_p_identity_bad",
            (
                f"isfinite({quote_ident('y')}) AND "
                f"isfinite({quote_ident('penalty_factor')}) AND "
                f"isfinite({quote_ident('y_p')}) AND NOT ("
                + _sql_finite_close(
                    quote_ident("y_p"),
                    f"({quote_ident('y')} * {quote_ident('penalty_factor')})",
                    rtol=rtol,
                )
                + ")"
            ),
        ),
        (
            "constraint_triplet_completeness",
            "constraint_triplet_bad",
            "(" + " OR ".join(triplet_parts) + ")" if triplet_parts else "FALSE",
        ),
        (
            "util_violation_identity",
            "util_violation_identity_bad",
            (
                "(" + " OR ".join(util_identity_parts) + ")"
                if util_identity_parts
                else "FALSE"
            ),
        ),
        (
            "util_penalty_identity",
            "util_penalty_identity_bad",
            (
                "(" + " OR ".join(penalty_identity_parts) + ")"
                if penalty_identity_parts
                else "FALSE"
            ),
        ),
        (
            "constraint_feasible_identity",
            "constraint_feasible_bad",
            (
                f"{quote_ident('constraint_feasible')} IS DISTINCT FROM "
                f"({' AND '.join(constraint_feasible_terms)})"
                if constraint_feasible_terms
                else "FALSE"
            ),
        ),
    ]

    for rule_name, agg_key, where_sql in rule_specs:
        rules[rule_name] = _finalize_rule(
            engine,
            count=_count(agg_key),
            where_sql=where_sql,
            max_examples=max_examples,
        )

    _register_catalogue_tables(engine, context)
    model = context.problem_info["model"]
    param_bad_total = 0
    param_where_parts: list[str] = []

    for var_name in context.var_names:
        idx_col = quote_ident(f"idx__{var_name}")
        param_col = quote_ident(f"param__{var_name}")
        cat = quote_ident(f"cat_idx__{var_name}")
        where = (
            f"CAST({param_col} AS VARCHAR) IS DISTINCT FROM ("
            f"SELECT c.val FROM {cat} c WHERE c.idx = {idx_col})"
        )
        count_row = engine.fetch_one(
            f"SELECT COUNT(*) FROM {view} WHERE {where}"
        )
        count = int(count_row[0]) if count_row else 0
        param_bad_total += count
        if count:
            param_where_parts.append(f"({where})")

    for name, meta in model.items():
        if meta.get("role") != "fixed":
            continue
        expected = meta["values"][int(meta["fixed_idx"])]
        param_col = quote_ident(f"param__{name}")
        if isinstance(expected, float):
            where = (
                f"NOT ({_sql_finite_close(param_col, str(float(expected)), rtol=rtol)})"
            )
        else:
            where = (
                f"CAST({param_col} AS VARCHAR) IS DISTINCT FROM "
                f"{quote_string(str(expected))}"
            )
        count_row = engine.fetch_one(
            f"SELECT COUNT(*) FROM {view} WHERE {where}"
        )
        count = int(count_row[0]) if count_row else 0
        param_bad_total += count
        if count:
            param_where_parts.append(f"({where})")

    for name, meta in model.items():
        if meta.get("role") != "lookup":
            continue
        key = str(meta["lookup_key"])
        param_col = quote_ident(f"param__{name}")
        key_col = quote_ident(f"param__{key}")
        cat = quote_ident(f"cat_lookup__{name}")
        where = (
            f"CAST({param_col} AS VARCHAR) IS DISTINCT FROM ("
            f"SELECT c.val FROM {cat} c WHERE c.key_val IS NOT DISTINCT FROM "
            f"CAST({key_col} AS VARCHAR))"
        )
        count_row = engine.fetch_one(
            f"SELECT COUNT(*) FROM {view} WHERE {where}"
        )
        count = int(count_row[0]) if count_row else 0
        param_bad_total += count
        if count:
            param_where_parts.append(f"({where})")

    rules["decoded_parameters"] = _finalize_rule(
        engine,
        count=param_bad_total,
        where_sql=(
            " OR ".join(param_where_parts) if param_where_parts else "FALSE"
        ),
        max_examples=max_examples,
    )

    numeric_quality: dict[str, Any] = {}
    for col_name in numeric_cols:
        numeric_quality[col_name] = {
            "null": _count(f"nq_null__{col_name}"),
            "nan": _count(f"nq_nan__{col_name}"),
            "pos_inf": _count(f"nq_posinf__{col_name}"),
            "neg_inf": _count(f"nq_neginf__{col_name}"),
        }
    report["numeric_quality"] = numeric_quality
    report["rules"] = rules

    scientific_ok = all(bool(rule.get("ok")) for rule in rules.values()) and bool(
        report["uniqueness"]["ok"]
    )
    report["scientific_ok"] = scientific_ok
    report["ok"] = bool(structural.ok) and scientific_ok
    report["overall_status"] = (
        "success" if report["ok"] else "scientific_integrity_failure"
    )
    return normalize_report_value(report)
