"""
Feasibility counts and variable-level feasibility summaries.

Fills the ``feasibility`` section of the dataset report. Report writing is deferred to C3.8.
"""
from __future__ import annotations

from typing import Any, Mapping

from core.ioh_core.workflows.brute_force.analysis.engine import (
    BruteForceQueryEngine,
    quote_ident,
)
from core.ioh_core.workflows.brute_force.analysis.models import (
    BruteForceDatasetContext,
    DatasetReportOptions,
)
from core.ioh_core.workflows.brute_force.analysis.serialization import (
    normalize_report_value,
)


def _share_or_zero(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def _variable_catalogue_map(
    context: BruteForceDatasetContext,
    var_name: str,
) -> dict[int, Any]:
    meta = context.problem_info["model"][var_name]
    values = list(meta["values"])
    lower = int(meta["lb"])
    upper = int(meta["ub"])
    return {index: values[index] for index in range(lower, upper + 1)}


def _register_var_catalogue(
    engine: BruteForceQueryEngine,
    *,
    var_name: str,
    values: Mapping[int, Any],
) -> str:
    table = f"feas_cat__{var_name}"
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


def _dataset_counts(engine: BruteForceQueryEngine) -> dict[str, int]:
    view = quote_ident(engine.view_name)
    row = engine.fetch_dicts(
        f"SELECT "
        f"COUNT(*) AS observed_rows, "
        f"SUM(CASE WHEN {quote_ident('analysis_ok')} THEN 1 ELSE 0 END) "
        f"AS analysis_ok, "
        f"SUM(CASE WHEN NOT {quote_ident('analysis_ok')} THEN 1 ELSE 0 END) "
        f"AS analysis_error, "
        f"SUM(CASE WHEN {quote_ident('constraint_feasible')} THEN 1 ELSE 0 END) "
        f"AS constraint_feasible, "
        f"SUM(CASE WHEN {quote_ident('feasible')} THEN 1 ELSE 0 END) "
        f"AS feasible, "
        f"SUM(CASE WHEN isfinite({quote_ident('y')}) THEN 1 ELSE 0 END) "
        f"AS finite_y, "
        f"SUM(CASE WHEN isfinite({quote_ident('y_p')}) THEN 1 ELSE 0 END) "
        f"AS finite_y_p "
        f"FROM {view}",
        max_rows=1,
    )
    if not row:
        raise RuntimeError("feasibility aggregate query returned no row")
    return {key: int(value or 0) for key, value in row[0].items()}


def _metric_block(
    counts: Mapping[str, int],
    *,
    observed_rows: int,
    full_catalogue_rows: int,
) -> dict[str, Any]:
    metrics = (
        "analysis_ok",
        "analysis_error",
        "constraint_feasible",
        "feasible",
        "finite_y",
        "finite_y_p",
    )
    dataset_shares = {
        name: {
            "count": int(counts[name]),
            "share_of_observed": _share_or_zero(int(counts[name]), observed_rows),
        }
        for name in metrics
    }
    catalogue_coverage = {
        name: {
            "count": int(counts[name]),
            "share_of_catalogue": _share_or_zero(
                int(counts[name]), full_catalogue_rows
            ),
        }
        for name in metrics
    }
    return {
        "observed_rows": observed_rows,
        "full_catalogue_rows": full_catalogue_rows,
        "dataset_shares": dataset_shares,
        "catalogue_coverage": catalogue_coverage,
    }


def _level_status(
    *,
    observed_count: int,
    feasible_total: int,
) -> str:
    if observed_count == 0:
        return "unobserved_level"
    if feasible_total == 0:
        return "no_feasible_rows"
    return "ok"


def _analyze_variable_levels(
    context: BruteForceDatasetContext,
    engine: BruteForceQueryEngine,
    *,
    var_name: str,
    observed_rows: int,
    feasible_total: int,
) -> dict[str, Any]:
    catalogue = _variable_catalogue_map(context, var_name)
    cat_table = _register_var_catalogue(
        engine, var_name=var_name, values=catalogue
    )
    view = quote_ident(engine.view_name)
    idx_col = quote_ident(f"idx__{var_name}")
    feasible_col = quote_ident("feasible")

    level_rows = engine.fetch_dicts(
        f"SELECT c.idx AS catalogue_index, c.val AS decoded_value, "
        f"COALESCE(o.observed_count, 0) AS observed_count, "
        f"COALESCE(o.feasible_count, 0) AS feasible_count "
        f"FROM {quote_ident(cat_table)} c "
        f"LEFT JOIN ("
        f"  SELECT {idx_col} AS idx, "
        f"  COUNT(*) AS observed_count, "
        f"  SUM(CASE WHEN {feasible_col} THEN 1 ELSE 0 END) AS feasible_count "
        f"  FROM {view} GROUP BY 1"
        f") o ON o.idx = c.idx "
        f"ORDER BY c.idx",
        max_rows=max(len(catalogue), 1),
    )

    levels: list[dict[str, Any]] = []
    for row in level_rows:
        observed_count = int(row["observed_count"])
        feasible_count = int(row["feasible_count"])
        observed_share = _share_or_zero(observed_count, observed_rows)
        status = _level_status(
            observed_count=observed_count, feasible_total=feasible_total
        )

        if status == "unobserved_level":
            share_within_feasible = 0.0 if feasible_total > 0 else None
            conditional_rate = None
            enrichment = None
        elif status == "no_feasible_rows":
            share_within_feasible = None
            conditional_rate = (
                None
                if observed_count == 0
                else _share_or_zero(feasible_count, observed_count)
            )
            enrichment = None
        else:
            share_within_feasible = _share_or_zero(feasible_count, feasible_total)
            conditional_rate = _share_or_zero(feasible_count, observed_count)
            enrichment = (
                None
                if observed_share == 0.0
                else float(share_within_feasible) / float(observed_share)
            )

        levels.append(
            {
                "index": int(row["catalogue_index"]),
                "value": catalogue[int(row["catalogue_index"])],
                "observed_count": observed_count,
                "observed_share": observed_share,
                "feasible_count_at_level": feasible_count,
                "share_within_all_feasible_designs": share_within_feasible,
                "conditional_feasibility_rate": conditional_rate,
                "feasible_level_enrichment": enrichment,
                "status": status,
            }
        )

    return {
        "name": var_name,
        "declared_level_count": len(catalogue),
        "levels": levels,
    }


def _b8_cross_check(
    context: BruteForceDatasetContext,
    counts: Mapping[str, int],
) -> dict[str, Any]:
    summary = context.collect_summary
    if summary is None:
        return {"status": "unavailable", "reason": "summary_missing", "ok": None}

    checks = {
        "analysis_ok_count": ("analysis_ok", summary.get("analysis_ok_count")),
        "analysis_error_count": (
            "analysis_error",
            summary.get("analysis_error_count"),
        ),
        "feasible_count": ("feasible", summary.get("feasible_count")),
    }
    disagreements: list[str] = []
    for label, (count_key, summary_value) in checks.items():
        if summary_value is None or int(summary_value) != int(counts[count_key]):
            disagreements.append(label)
    if disagreements:
        return {
            "status": "disagree",
            "ok": False,
            "disagreements": disagreements,
        }
    return {"status": "ok", "ok": True, "disagreements": []}


def run_feasibility_analysis(
    context: BruteForceDatasetContext,
    engine: BruteForceQueryEngine,
    *,
    options: DatasetReportOptions | None = None,
) -> dict[str, Any]:
    """
    Compute dataset feasibility totals and per-level feasible enrichment.

    Parameters
    ----------
    context : BruteForceDatasetContext
        Sealed dataset context.
    engine : BruteForceQueryEngine
        Open DuckDB engine over the same dataset.
    options : DatasetReportOptions or None, optional
        Reserved for future knobs; currently unused.

    Returns
    -------
    dict
        JSON-compatible ``feasibility`` payload.
    """
    del options
    counts = _dataset_counts(engine)
    observed_rows = int(counts["observed_rows"])
    feasible_total = int(counts["feasible"])
    full_catalogue_rows = int(context.full_catalogue_rows)

    totals = _metric_block(
        counts,
        observed_rows=observed_rows,
        full_catalogue_rows=full_catalogue_rows,
    )
    variables = [
        _analyze_variable_levels(
            context,
            engine,
            var_name=name,
            observed_rows=observed_rows,
            feasible_total=feasible_total,
        )
        for name in context.var_names
    ]
    b8 = _b8_cross_check(context, counts)

    overall_status = "ok"
    if feasible_total == 0:
        overall_status = "no_feasible_rows"

    report = {
        "section": "feasibility",
        "is_full_enumeration": bool(context.is_full_enumeration),
        "overall_status": overall_status,
        "totals": totals,
        "variables": variables,
        "b8_cross_check": b8,
        "ok": b8.get("status") in {"ok", "unavailable"},
        "feasible_counts_use_stored_full_set_flags": True,
        "feasible_column": "feasible",
        "constraint_feasible_column": "constraint_feasible",
    }
    if b8.get("status") == "disagree":
        report["ok"] = False
    return normalize_report_value(report)
