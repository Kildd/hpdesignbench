"""
Best/worst-design extraction and near-optimum bands for sealed brute-force datasets.

Fills the ``best_designs`` section of the dataset report. Report writing is deferred to C3.8.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from core.ioh_core.workflows.brute_force.analysis.engine import (
    BruteForceQueryEngine,
    quote_ident,
)
from core.ioh_core.workflows.brute_force.analysis.keep_list import (
    analysis_constraint_names,
)
from core.ioh_core.workflows.brute_force.analysis.models import (
    BruteForceDatasetContext,
    DatasetReportOptions,
)
from core.ioh_core.workflows.brute_force.analysis.report import (
    NEAR_OPTIMUM_TOLERANCES_PERCENT,
)
from core.ioh_core.workflows.brute_force.analysis.serialization import (
    normalize_report_value,
)
from core.ioh_core.workflows.brute_force.structural import scientific_values_equal


def _unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    payload = {"status": "unavailable", "reason": reason}
    payload.update(extra)
    return payload


def _finite_close(left: float, right: float, *, rtol: float) -> bool:
    return bool(
        math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(left, right, rel_tol=rtol, abs_tol=0.0)
    )


def _row_to_design_record(
    row: Mapping[str, Any],
    context: BruteForceDatasetContext,
    *,
    rtol: float,
    check_unpenalized: bool,
) -> dict[str, Any]:
    indices = {
        name: row.get(f"idx__{name}") for name in context.var_names
    }
    params = {
        name: row.get(f"param__{name}") for name in context.param_names
    }
    constraints = {
        name: {
            "util": row.get(f"util__{name}"),
            "violation": row.get(f"violation__{name}"),
            "penalty": row.get(f"penalty__{name}"),
        }
        for name in analysis_constraint_names(context)
    }
    record: dict[str, Any] = {
        "status": "found",
        "global_index": int(row["global_index"]),
        "shard_index": int(row["shard_index"]),
        "problem_id": row.get("problem_id"),
        "evaluation_time_s": row.get("evaluation_time_s"),
        "y": row.get("y"),
        "y_p": row.get("y_p"),
        "penalty_sum": row.get("penalty_sum"),
        "penalty_factor": row.get("penalty_factor"),
        "analysis_ok": row.get("analysis_ok"),
        "constraint_feasible": row.get("constraint_feasible"),
        "feasible": row.get("feasible"),
        "indices": indices,
        "params": params,
        "constraints": constraints,
        "diagnostics": {
            "analysis_error_stage": row.get("analysis_error_stage"),
            "analysis_error_message": row.get("analysis_error_message"),
            "analysis_error_exception_type": row.get(
                "analysis_error_exception_type"
            ),
            "analysis_error_material_id": row.get("analysis_error_material_id"),
            "analysis_error_json": row.get("analysis_error_json"),
            "remapped_utilizations": row.get("remapped_utilizations"),
            "cost_components_missing": row.get("cost_components_missing"),
        },
    }
    if check_unpenalized:
        penalty_factor = row.get("penalty_factor")
        y_val = row.get("y")
        y_p_val = row.get("y_p")
        record["penalty_factor_is_one"] = (
            penalty_factor is not None
            and _finite_close(float(penalty_factor), 1.0, rtol=rtol)
        )
        record["y_equals_y_p"] = (
            y_val is not None
            and y_p_val is not None
            and _finite_close(float(y_val), float(y_p_val), rtol=rtol)
        )
    return record


def _select_winner(
    engine: BruteForceQueryEngine,
    context: BruteForceDatasetContext,
    *,
    where_sql: str,
    unavailable_reason: str,
    rtol: float,
    check_unpenalized: bool,
    y_p_order: str = "ASC",
) -> dict[str, Any]:
    order = str(y_p_order).upper()
    if order not in {"ASC", "DESC"}:
        raise ValueError(f"y_p_order must be 'ASC' or 'DESC', got {y_p_order!r}")
    view = quote_ident(engine.view_name)
    # SELECT * is bounded by LIMIT 1; never materialize the base relation.
    # Equal-y_p ties always break on global_index ASC (deterministic, matches best).
    rows = engine.fetch_dicts(
        f"SELECT * FROM {view} "
        f"WHERE {where_sql} "
        f"ORDER BY {quote_ident('y_p')} {order}, "
        f"{quote_ident('global_index')} ASC "
        f"LIMIT 1",
        max_rows=1,
    )
    if not rows:
        return _unavailable(unavailable_reason)
    return _row_to_design_record(
        rows[0],
        context,
        rtol=rtol,
        check_unpenalized=check_unpenalized,
    )


def _b8_cross_check(
    context: BruteForceDatasetContext,
    *,
    best_finite: Mapping[str, Any],
    finite_objective_count: int,
    rtol: float,
) -> dict[str, Any]:
    summary = context.collect_summary
    if summary is None:
        return {"status": "unavailable", "reason": "summary_missing"}

    disagreements: list[str] = []
    if best_finite.get("status") != "found":
        if summary.get("best_finite_y_p") is not None:
            disagreements.append("best_finite_missing_but_summary_has_winner")
        if summary.get("best_global_index") is not None:
            disagreements.append("best_index_missing_but_summary_has_winner")
    else:
        summary_y = summary.get("best_finite_y_p")
        summary_index = summary.get("best_global_index")
        summary_params = summary.get("best_params")
        if summary_y is None or not _finite_close(
            float(best_finite["y_p"]), float(summary_y), rtol=rtol
        ):
            disagreements.append("best_finite_y_p")
        if summary_index is None or int(best_finite["global_index"]) != int(
            summary_index
        ):
            disagreements.append("best_global_index")
        if isinstance(summary_params, Mapping):
            if not scientific_values_equal(
                dict(best_finite["params"]),
                dict(summary_params),
                rtol=rtol,
            ):
                disagreements.append("best_params")
        elif summary_params is not None:
            disagreements.append("best_params")

    summary_finite = summary.get("finite_objective_count")
    if summary_finite is None or int(summary_finite) != int(finite_objective_count):
        disagreements.append("finite_objective_count")

    if disagreements:
        return {
            "status": "disagree",
            "ok": False,
            "disagreements": disagreements,
        }
    return {"status": "ok", "ok": True, "disagreements": []}


def _near_optimum_bands(
    engine: BruteForceQueryEngine,
    *,
    y_p_star: float | None,
    tolerances_percent: tuple[float, ...] = NEAR_OPTIMUM_TOLERANCES_PERCENT,
) -> dict[str, Any]:
    if y_p_star is None or not math.isfinite(y_p_star):
        return {
            "status": "unavailable",
            "reason": "optimum_missing_or_nonfinite",
            "tolerances_percent": list(tolerances_percent),
            "y_p_star": y_p_star,
            "bands": [],
            "monotone_ok": None,
            "winner_in_all_bands": None,
        }
    if y_p_star < 0.0:
        return {
            "status": "unavailable",
            "reason": "negative_optimum",
            "tolerances_percent": list(tolerances_percent),
            "y_p_star": float(y_p_star),
            "bands": [],
            "monotone_ok": None,
            "winner_in_all_bands": None,
        }

    view = quote_ident(engine.view_name)
    y_col = quote_ident("y_p")
    feasible_col = quote_ident("feasible")
    select_parts = [
        "COUNT(*) AS observed_rows",
        f"SUM(CASE WHEN {feasible_col} THEN 1 ELSE 0 END) AS feasible_count",
        f"SUM(CASE WHEN isfinite({y_col}) THEN 1 ELSE 0 END) AS finite_y_p_count",
    ]
    thresholds = [
        float(y_p_star) * (1.0 + float(tol) / 100.0) for tol in tolerances_percent
    ]
    for index, threshold in enumerate(thresholds):
        select_parts.append(
            f"SUM(CASE WHEN isfinite({y_col}) AND {y_col} <= ({threshold}) "
            f"THEN 1 ELSE 0 END) AS within_{index}"
        )
        select_parts.append(
            f"SUM(CASE WHEN isfinite({y_col}) AND {y_col} <= ({threshold}) "
            f"AND {feasible_col} THEN 1 ELSE 0 END) AS feasible_within_{index}"
        )

    agg = engine.fetch_dicts(
        f"SELECT {', '.join(select_parts)} FROM {view}",
        max_rows=1,
    )
    if not agg:
        raise RuntimeError("near-optimum aggregate query returned no row")
    row = agg[0]
    observed_rows = int(row["observed_rows"] or 0)
    feasible_count = int(row["feasible_count"] or 0)

    bands: list[dict[str, Any]] = []
    counts: list[int] = []
    for index, (tol, threshold) in enumerate(zip(tolerances_percent, thresholds)):
        count_within = int(row[f"within_{index}"] or 0)
        count_feasible_within = int(row[f"feasible_within_{index}"] or 0)
        counts.append(count_within)
        bands.append(
            {
                "tolerance_percent": float(tol),
                "threshold": float(threshold),
                "count_within": count_within,
                "share_of_all_solutions": (
                    float(count_within) / float(observed_rows)
                    if observed_rows > 0
                    else 0.0
                ),
                "count_feasible_within": count_feasible_within,
                "share_of_feasible_solutions": (
                    float(count_feasible_within) / float(feasible_count)
                    if feasible_count > 0
                    else None
                ),
            }
        )

    monotone_ok = all(
        counts[index] <= counts[index + 1] for index in range(len(counts) - 1)
    )
    winner_in_all = all(count >= 1 for count in counts)

    return {
        "status": "computed",
        "reason": None,
        "tolerances_percent": list(tolerances_percent),
        "y_p_star": float(y_p_star),
        "observed_rows": observed_rows,
        "feasible_count": feasible_count,
        "finite_y_p_count": int(row["finite_y_p_count"] or 0),
        "bands": bands,
        "monotone_ok": monotone_ok,
        "winner_in_all_bands": winner_in_all,
    }


def run_best_designs_analysis(
    context: BruteForceDatasetContext,
    engine: BruteForceQueryEngine,
    *,
    options: DatasetReportOptions | None = None,
) -> dict[str, Any]:
    """
    Extract best finite / best and worst feasible designs and near-optimum bands.

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
        JSON-compatible ``best_designs`` payload.
    """
    del options  # reserved
    rtol = float(engine.scientific_rtol)
    view = quote_ident(engine.view_name)
    y_col = quote_ident("y_p")

    finite_where = f"isfinite({y_col})"
    feasible_where = (
        f"isfinite({y_col}) AND "
        f"{quote_ident('analysis_ok')} AND "
        f"{quote_ident('constraint_feasible')} AND "
        f"{quote_ident('feasible')}"
    )

    best_finite = _select_winner(
        engine,
        context,
        where_sql=finite_where,
        unavailable_reason="no_finite_y_p",
        rtol=rtol,
        check_unpenalized=False,
    )
    best_feasible = _select_winner(
        engine,
        context,
        where_sql=feasible_where,
        unavailable_reason="no_feasible_finite_y_p",
        rtol=rtol,
        check_unpenalized=True,
        y_p_order="ASC",
    )
    worst_feasible = _select_winner(
        engine,
        context,
        where_sql=feasible_where,
        unavailable_reason="no_feasible_finite_y_p",
        rtol=rtol,
        check_unpenalized=True,
        y_p_order="DESC",
    )

    finite_count_row = engine.fetch_one(
        f"SELECT COUNT(*) FROM {view} WHERE {finite_where}"
    )
    finite_objective_count = int(finite_count_row[0]) if finite_count_row else 0

    b8 = _b8_cross_check(
        context,
        best_finite=best_finite,
        finite_objective_count=finite_objective_count,
        rtol=rtol,
    )

    y_p_star = (
        float(best_finite["y_p"])
        if best_finite.get("status") == "found"
        else None
    )
    near = _near_optimum_bands(engine, y_p_star=y_p_star)

    report = {
        "best_finite": best_finite,
        "best_feasible": best_feasible,
        "worst_feasible": worst_feasible,
        "finite_objective_count": finite_objective_count,
        "b8_cross_check": b8,
        "near_optimum_bands": near,
        "scientific_rtol": rtol,
        "ok": (
            b8.get("status") in {"ok", "unavailable"}
            and near.get("status") in {"computed", "unavailable"}
            and (near.get("monotone_ok") in {True, None})
            and (near.get("winner_in_all_bands") in {True, None})
        ),
    }
    # Near-optimum internal consistency failures make the section not ok.
    if near.get("status") == "computed":
        if not near.get("monotone_ok") or not near.get("winner_in_all_bands"):
            report["ok"] = False
    if b8.get("status") == "disagree":
        report["ok"] = False
    return normalize_report_value(report)
