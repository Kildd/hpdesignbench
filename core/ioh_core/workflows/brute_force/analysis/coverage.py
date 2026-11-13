"""
Enumeration coverage and marginal distributions for sealed brute-force datasets.

This module fills the ``enumeration_coverage`` section. It does not
duplicate integrity uniqueness / numeric-quality scans (C3.3) and does not
write report files (C3.8).
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping

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


def _share(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def _register_var_catalogue(
    engine: BruteForceQueryEngine,
    *,
    var_name: str,
    values: Mapping[int, Any],
) -> str:
    """
    Register ``(idx, val)`` for one variable; return the temp table name.
    """
    table = f"cov_cat__{var_name}"
    conn = engine.connection
    conn.execute(f"DROP TABLE IF EXISTS {quote_ident(table)}")
    conn.execute(
        f"CREATE TEMP TABLE {quote_ident(table)} (idx INTEGER, val VARCHAR)"
    )
    rows = [
        (int(index), None if value is None else str(value))
        for index, value in sorted(values.items())
    ]
    if rows:
        conn.executemany(
            f"INSERT INTO {quote_ident(table)} VALUES (?, ?)",
            rows,
        )
    return table


def _variable_catalogue_map(
    context: BruteForceDatasetContext,
    var_name: str,
) -> dict[int, Any]:
    meta = context.problem_info["model"][var_name]
    values = list(meta["values"])
    lower = int(meta["lb"])
    upper = int(meta["ub"])
    return {index: values[index] for index in range(lower, upper + 1)}


def _analyze_variable(
    context: BruteForceDatasetContext,
    engine: BruteForceQueryEngine,
    *,
    var_name: str,
    observed_rows: int,
) -> dict[str, Any]:
    space = context.index_space
    position = space.var_names.index(var_name)
    declared_lower = int(space.lower_bounds[position])
    declared_upper = int(space.upper_bounds[position])
    declared_level_count = int(space.sizes[position])
    full_rows = int(context.full_catalogue_rows)
    expected_count = (
        full_rows // declared_level_count if declared_level_count else 0
    )
    expected_share = _share(expected_count, full_rows)

    catalogue = _variable_catalogue_map(context, var_name)
    cat_table = _register_var_catalogue(engine, var_name=var_name, values=catalogue)
    view = quote_ident(engine.view_name)
    idx_col = quote_ident(f"idx__{var_name}")

    summary = engine.fetch_one(
        f"SELECT MIN({idx_col}), MAX({idx_col}), "
        f"COUNT(DISTINCT {idx_col}) FROM {view}"
    )
    observed_min = None if summary is None else summary[0]
    observed_max = None if summary is None else summary[1]
    observed_distinct = 0 if summary is None else int(summary[2] or 0)

    level_rows = engine.fetch_dicts(
        f"SELECT c.idx AS catalogue_index, c.val AS decoded_value, "
        f"COALESCE(o.row_count, 0) AS row_count "
        f"FROM {quote_ident(cat_table)} c "
        f"LEFT JOIN ("
        f"  SELECT {idx_col} AS idx, COUNT(*) AS row_count "
        f"  FROM {view} GROUP BY 1"
        f") o ON o.idx = c.idx "
        f"ORDER BY c.idx",
        max_rows=max(declared_level_count, 1),
    )

    levels: list[dict[str, Any]] = []
    for row in level_rows:
        count = int(row["row_count"])
        share = _share(count, observed_rows)
        levels.append(
            {
                "index": int(row["catalogue_index"]),
                "value": (
                    None
                    if row["decoded_value"] is None
                    else (
                        # Prefer native sealed catalogue value over VARCHAR render.
                        catalogue[int(row["catalogue_index"])]
                    )
                ),
                "row_count": count,
                "share_of_observed": share,
                "expected_count_full_enumeration": expected_count,
                "expected_share_full_enumeration": expected_share,
                "count_deviation": count - expected_count,
                "share_deviation": share - expected_share,
            }
        )

    failures: list[str] = []
    if context.is_full_enumeration:
        if observed_min != declared_lower or observed_max != declared_upper:
            failures.append("observed_bounds_mismatch")
        if observed_distinct != declared_level_count:
            failures.append("observed_distinct_mismatch")
        if any(level["row_count"] == 0 for level in levels):
            failures.append("missing_declared_level")
        if any(level["row_count"] != expected_count for level in levels):
            failures.append("exact_marginal_mismatch")

    return {
        "name": var_name,
        "declared_lower": declared_lower,
        "declared_upper": declared_upper,
        "declared_level_count": declared_level_count,
        "observed_min": observed_min,
        "observed_max": observed_max,
        "observed_distinct_count": observed_distinct,
        "levels": levels,
        "failures": failures,
        "ok": len(failures) == 0,
        "marginals_are_pass_fail": bool(context.is_full_enumeration),
    }


def _analyze_fixed_parameters(
    context: BruteForceDatasetContext,
    engine: BruteForceQueryEngine,
) -> list[dict[str, Any]]:
    view = quote_ident(engine.view_name)
    model = context.problem_info["model"]
    results: list[dict[str, Any]] = []
    for name, meta in model.items():
        if meta.get("role") != "fixed":
            continue
        expected = meta["values"][int(meta["fixed_idx"])]
        param_col = quote_ident(f"param__{name}")
        distinct_row = engine.fetch_one(
            f"SELECT COUNT(DISTINCT CAST({param_col} AS VARCHAR)) FROM {view}"
        )
        distinct_observed = int(distinct_row[0]) if distinct_row else 0
        if isinstance(expected, float):
            mismatch_sql = (
                f"NOT (isfinite({param_col}) AND "
                f"abs({param_col} - ({float(expected)})) <= "
                f"(1e-12) * greatest(abs({param_col}), abs({float(expected)})))"
            )
        else:
            mismatch_sql = (
                f"CAST({param_col} AS VARCHAR) IS DISTINCT FROM "
                f"{quote_string(str(expected))}"
            )
        mismatch_row = engine.fetch_one(
            f"SELECT COUNT(*) FROM {view} WHERE {mismatch_sql}"
        )
        mismatch_count = int(mismatch_row[0]) if mismatch_row else 0
        ok = distinct_observed == 1 and mismatch_count == 0
        results.append(
            {
                "name": name,
                "expected": expected,
                "distinct_observed": distinct_observed,
                "mismatch_count": mismatch_count,
                "ok": ok,
            }
        )
    return results


def _analyze_pairwise(
    context: BruteForceDatasetContext,
    engine: BruteForceQueryEngine,
    *,
    observed_rows: int,
) -> list[dict[str, Any]]:
    view = quote_ident(engine.view_name)
    pairs: list[dict[str, Any]] = []
    for left, right in combinations(context.var_names, 2):
        left_col = quote_ident(f"idx__{left}")
        right_col = quote_ident(f"idx__{right}")
        rows = engine.fetch_dicts(
            f"SELECT {left_col} AS left_index, {right_col} AS right_index, "
            f"COUNT(*) AS row_count "
            f"FROM {view} "
            f"GROUP BY 1, 2 "
            f"ORDER BY 1, 2",
            max_rows=10_000,
        )
        combinations_out = [
            {
                "left_index": int(row["left_index"]),
                "right_index": int(row["right_index"]),
                "row_count": int(row["row_count"]),
                "share_of_observed": _share(int(row["row_count"]), observed_rows),
            }
            for row in rows
        ]
        pairs.append(
            {
                "left": left,
                "right": right,
                "combination_count": len(combinations_out),
                "combinations": combinations_out,
            }
        )
    return pairs


def run_coverage_analysis(
    context: BruteForceDatasetContext,
    engine: BruteForceQueryEngine,
    *,
    options: DatasetReportOptions | None = None,
) -> dict[str, Any]:
    """
    Compute enumeration-coverage marginals for a sealed brute-force dataset.

    Parameters
    ----------
    context : BruteForceDatasetContext
        Sealed dataset context.
    engine : BruteForceQueryEngine
        Open DuckDB engine over the same dataset.
    options : DatasetReportOptions or None, optional
        Controls optional pairwise tables.

    Returns
    -------
    dict
        JSON-compatible ``enumeration_coverage`` payload.
    """
    options = options or DatasetReportOptions()
    observed_rows = engine.count_rows()

    variables = [
        _analyze_variable(
            context,
            engine,
            var_name=name,
            observed_rows=observed_rows,
        )
        for name in context.var_names
    ]
    fixed_parameters = _analyze_fixed_parameters(context, engine)
    pairwise = None
    if options.include_pairwise_level_coverage:
        pairwise = _analyze_pairwise(
            context, engine, observed_rows=observed_rows
        )

    variables_ok = all(bool(item["ok"]) for item in variables)
    fixed_ok = all(bool(item["ok"]) for item in fixed_parameters)
    report = {
        "section": "enumeration_coverage",
        "is_full_enumeration": bool(context.is_full_enumeration),
        "observed_rows": int(observed_rows),
        "planned_rows": int(context.planned_rows),
        "full_catalogue_rows": int(context.full_catalogue_rows),
        "coverage_fraction": float(context.coverage_fraction),
        "variables": variables,
        "fixed_parameters": fixed_parameters,
        "pairwise_level_coverage": pairwise,
        "ok": bool(variables_ok and fixed_ok),
    }
    return normalize_report_value(report)
