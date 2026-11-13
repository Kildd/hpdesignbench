"""
Constraint activity, sentinel, and penalty diagnostics for dataset reports.

Per-design activity means finite ``util > 1`` (violated), not the problem-level
``active`` flag. ``utilization_shares`` records the three Layer-plot shares
(active / maximum violation / invalid utilization) persisted in ``report.json``.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

from core.ioh_core.workflows.brute_force.analysis.engine import (
    BruteForceQueryEngine,
    quote_ident,
    quote_string,
)
from core.ioh_core.workflows.brute_force.analysis.keep_list import (
    analysis_constraint_names,
)
from core.ioh_core.workflows.brute_force.analysis.models import (
    BruteForceDatasetContext,
    DatasetReportOptions,
)
from core.ioh_core.workflows.brute_force.analysis.serialization import (
    normalize_report_value,
)

QUANTILE_METHOD = "quantile_cont"
_QUANTILES = (0.5, 0.9, 0.95, 0.99)


def _share(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return float(count) / float(total)


def _sql_active(util: str) -> str:
    return f"(isfinite({util}) AND {util} > 1.0)"


def _sql_sentinel(util: str, *, sentinel: float, atol: float) -> str:
    return (
        f"(isfinite({util}) AND abs(({util}) - ({sentinel})) <= ({atol}))"
    )


def _sql_close(left: str, right: str, *, rtol: float) -> str:
    return (
        f"(isfinite({left}) AND isfinite({right}) AND "
        f"abs(({left}) - ({right})) <= ({rtol}) * "
        f"greatest(abs({left}), abs({right})))"
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _constraint_refs(context: BruteForceDatasetContext) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for name in analysis_constraint_names(context):
        refs.append(
            {
                "name": name,
                "util": quote_ident(f"util__{name}"),
                "violation": quote_ident(f"violation__{name}"),
                "penalty": quote_ident(f"penalty__{name}"),
            }
        )
    return refs


def _sql_max_finite_util(refs: list[dict[str, str]]) -> str:
    """Greatest finite utilization across constraints (NULLs ignored)."""
    if not refs:
        return "CAST(NULL AS DOUBLE)"
    pieces = [
        f"CASE WHEN isfinite({ref['util']}) THEN {ref['util']} END"
        for ref in refs
    ]
    return f"greatest({', '.join(pieces)})"


def _utilization_shares(
    engine: BruteForceQueryEngine,
    refs: list[dict[str, str]],
    *,
    sentinel: float,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """
    Per-constraint shares used by the brute-force utilisation plot.

    Definitions (denominator = all observed rows):

    1. ``active`` — finite utilization ``> 1`` (includes sentinel when present).
    2. ``maximum_violation`` — active and tied for the greatest finite
       utilization among all constraints (ties count for every tied constraint).
    3. ``invalid_utilization`` — utilization equals ``opt_invalid_utilization``.
    """
    view = quote_ident(engine.view_name)
    observed = engine.count_rows()
    if not refs:
        return {
            "observed_rows": observed,
            "definitions": {
                "active": "finite(util) and util > 1",
                "maximum_violation": (
                    "active and util tied for max finite util across constraints"
                ),
                "invalid_utilization": (
                    "finite(util) and abs(util - opt_invalid_utilization) "
                    "<= sentinel_atol"
                ),
                "denominator": "all_observed_rows",
            },
            "per_constraint": [],
        }

    max_util = _sql_max_finite_util(refs)
    select_parts = ["COUNT(*) AS observed_rows"]
    for ref in refs:
        name = ref["name"]
        util = ref["util"]
        active = _sql_active(util)
        is_sent = _sql_sentinel(util, sentinel=sentinel, atol=atol)
        is_max = (
            f"({active} AND "
            + _sql_close(util, max_util, rtol=rtol)
            + ")"
        )
        select_parts.extend(
            [
                f"SUM(CASE WHEN {active} THEN 1 ELSE 0 END) AS active__{name}",
                f"SUM(CASE WHEN {is_max} THEN 1 ELSE 0 END) AS maxviol__{name}",
                f"SUM(CASE WHEN {is_sent} THEN 1 ELSE 0 END) AS invalid__{name}",
            ]
        )

    row = engine.fetch_dicts(
        f"SELECT {', '.join(select_parts)} FROM {view}",
        max_rows=1,
    )[0]
    observed = int(row["observed_rows"] or 0)
    per_constraint: list[dict[str, Any]] = []
    for ref in refs:
        name = ref["name"]
        active_count = int(row[f"active__{name}"] or 0)
        max_count = int(row[f"maxviol__{name}"] or 0)
        invalid_count = int(row[f"invalid__{name}"] or 0)
        per_constraint.append(
            {
                "name": name,
                "active_count": active_count,
                "active_share_of_all_observed_rows": _share(
                    active_count, observed
                ),
                "maximum_violation_count": max_count,
                "maximum_violation_share_of_all_observed_rows": _share(
                    max_count, observed
                ),
                "invalid_utilization_count": invalid_count,
                "invalid_utilization_share_of_all_observed_rows": _share(
                    invalid_count, observed
                ),
            }
        )
    return {
        "observed_rows": observed,
        "definitions": {
            "active": "finite(util) and util > 1",
            "maximum_violation": (
                "active and util tied for max finite util across constraints"
            ),
            "invalid_utilization": (
                "finite(util) and abs(util - opt_invalid_utilization) "
                "<= sentinel_atol"
            ),
            "denominator": "all_observed_rows",
            "scientific_rtol": rtol,
            "sentinel_atol": atol,
            "opt_invalid_utilization": sentinel,
        },
        "per_constraint": per_constraint,
    }


def _activity_and_sentinel(
    engine: BruteForceQueryEngine,
    refs: list[dict[str, str]],
    *,
    sentinel: float,
    atol: float,
    expected_violation: float,
    expected_penalty: float,
    rtol: float,
) -> dict[str, Any]:
    view = quote_ident(engine.view_name)
    remapped = quote_ident("remapped_utilizations")
    analysis_ok = quote_ident("analysis_ok")

    select_parts = [
        "COUNT(*) AS observed_rows",
        f"SUM(CASE WHEN {analysis_ok} THEN 1 ELSE 0 END) AS analysis_ok_rows",
    ]
    for ref in refs:
        name = ref["name"]
        util = ref["util"]
        viol = ref["violation"]
        pen = ref["penalty"]
        active = _sql_active(util)
        is_sent = _sql_sentinel(util, sentinel=sentinel, atol=atol)
        in_remap = f"list_contains({remapped}, {quote_string(name)})"
        select_parts.extend(
            [
                f"SUM(CASE WHEN {active} THEN 1 ELSE 0 END) AS active__{name}",
                f"SUM(CASE WHEN {analysis_ok} AND {active} THEN 1 ELSE 0 END) "
                f"AS active_ok__{name}",
                (
                    f"SUM(CASE WHEN {active} AND NOT ("
                    f"(isfinite({viol}) AND {viol} > 0) AND "
                    f"(isfinite({pen}) AND {pen} > 0)"
                    f") THEN 1 ELSE 0 END) AS activity_disagree__{name}"
                ),
                f"SUM(CASE WHEN {is_sent} THEN 1 ELSE 0 END) AS sentinel__{name}",
                (
                    f"SUM(CASE WHEN {analysis_ok} AND {is_sent} THEN 1 ELSE 0 END) "
                    f"AS sentinel_ok__{name}"
                ),
                (
                    f"SUM(CASE WHEN {is_sent} AND NOT {in_remap} THEN 1 ELSE 0 END) "
                    f"AS sentinel_direct__{name}"
                ),
                (
                    f"SUM(CASE WHEN {is_sent} AND {in_remap} THEN 1 ELSE 0 END) "
                    f"AS sentinel_remap__{name}"
                ),
                (
                    f"SUM(CASE WHEN {is_sent} AND NOT ("
                    + _sql_close(viol, str(expected_violation), rtol=rtol)
                    + f" AND "
                    + _sql_close(pen, str(expected_penalty), rtol=rtol)
                    + f") THEN 1 ELSE 0 END) AS sentinel_formula_bad__{name}"
                ),
            ]
        )

    row = engine.fetch_dicts(
        f"SELECT {', '.join(select_parts)} FROM {view}",
        max_rows=1,
    )[0]
    observed = int(row["observed_rows"] or 0)
    analysis_ok_rows = int(row["analysis_ok_rows"] or 0)

    per_constraint: list[dict[str, Any]] = []
    for ref in refs:
        name = ref["name"]
        active_count = int(row[f"active__{name}"] or 0)
        active_ok = int(row[f"active_ok__{name}"] or 0)
        inactive = observed - active_count
        sentinel_count = int(row[f"sentinel__{name}"] or 0)
        per_constraint.append(
            {
                "name": name,
                "active_count": active_count,
                "inactive_count": inactive,
                "active_share_of_all_observed_rows": _share(active_count, observed),
                "active_share_of_analysis_ok_rows": _share(
                    active_ok, analysis_ok_rows
                ),
                "activity_signal_disagreements": int(
                    row[f"activity_disagree__{name}"] or 0
                ),
                "invalid_sentinel_count": sentinel_count,
                "invalid_sentinel_share_of_all_observed_rows": _share(
                    sentinel_count, observed
                ),
                "invalid_sentinel_share_of_analysis_ok_rows": _share(
                    int(row[f"sentinel_ok__{name}"] or 0), analysis_ok_rows
                ),
                "direct_sentinel_count": int(row[f"sentinel_direct__{name}"] or 0),
                "nonfinite_remapped_to_sentinel_count": int(
                    row[f"sentinel_remap__{name}"] or 0
                ),
                "sentinel_formula_mismatches": int(
                    row[f"sentinel_formula_bad__{name}"] or 0
                ),
            }
        )
    return {
        "observed_rows": observed,
        "analysis_ok_rows": analysis_ok_rows,
        "per_constraint": per_constraint,
    }


def _max_penalty_violation(
    engine: BruteForceQueryEngine,
    refs: list[dict[str, str]],
    *,
    rtol: float,
) -> dict[str, Any]:
    if not refs:
        return {
            "rows_with_any_positive_penalty": 0,
            "rows_with_a_unique_maximum": 0,
            "rows_with_tied_maxima": 0,
            "max_penalty_vs_max_violation_disagreements": 0,
            "per_constraint": [],
        }

    view = quote_ident(engine.view_name)
    pens = [ref["penalty"] for ref in refs]
    viols = [ref["violation"] for ref in refs]
    max_pen = f"greatest({', '.join(f'COALESCE({p}, 0.0)' for p in pens)})"
    max_viol = f"greatest({', '.join(f'COALESCE({v}, 0.0)' for v in viols)})"
    penalty_sum = quote_ident("penalty_sum")
    penalized = f"(isfinite({penalty_sum}) AND {penalty_sum} > 0.0 AND {max_pen} > 0.0)"

    select_parts = [
        f"SUM(CASE WHEN {penalized} THEN 1 ELSE 0 END) AS penalized_rows",
    ]
    is_max_pen = []
    is_max_viol = []
    for ref in refs:
        name = ref["name"]
        pen = ref["penalty"]
        viol = ref["violation"]
        pen_max = (
            f"({penalized} AND isfinite({pen}) AND {pen} > 0.0 AND "
            + _sql_close(pen, max_pen, rtol=rtol)
            + ")"
        )
        viol_max = (
            f"({penalized} AND isfinite({viol}) AND {viol} > 0.0 AND "
            + _sql_close(viol, max_viol, rtol=rtol)
            + ")"
        )
        is_max_pen.append(pen_max)
        is_max_viol.append(viol_max)
        select_parts.append(
            f"SUM(CASE WHEN {pen_max} THEN 1 ELSE 0 END) AS maxpen__{name}"
        )
        select_parts.append(
            f"SUM(CASE WHEN {viol_max} THEN 1 ELSE 0 END) AS maxviol__{name}"
        )

    # Unique vs tied maxima among penalized rows.
    max_pen_count_expr = " + ".join(
        f"CASE WHEN {expr} THEN 1 ELSE 0 END" for expr in is_max_pen
    )
    select_parts.append(
        f"SUM(CASE WHEN {penalized} AND ({max_pen_count_expr}) = 1 "
        f"THEN 1 ELSE 0 END) AS unique_max_rows"
    )
    select_parts.append(
        f"SUM(CASE WHEN {penalized} AND ({max_pen_count_expr}) > 1 "
        f"THEN 1 ELSE 0 END) AS tied_max_rows"
    )

    # Set disagreement: XOR of max-penalty vs max-violation membership.
    disagree_parts = [
        f"(({is_max_pen[i]}) AND NOT ({is_max_viol[i]})) OR "
        f"((NOT ({is_max_pen[i]})) AND ({is_max_viol[i]}))"
        for i in range(len(refs))
    ]
    select_parts.append(
        "SUM(CASE WHEN "
        + " OR ".join(f"({part})" for part in disagree_parts)
        + " THEN 1 ELSE 0 END) AS max_set_disagreements"
    )

    # Sentinel governs maximum.
    # Handled later with sentinel util checks in a separate lightweight query if needed.

    row = engine.fetch_dicts(
        f"SELECT {', '.join(select_parts)} FROM {view}",
        max_rows=1,
    )[0]
    observed = engine.count_rows()
    penalized_rows = int(row["penalized_rows"] or 0)
    per_constraint: list[dict[str, Any]] = []
    for ref in refs:
        name = ref["name"]
        max_count = int(row[f"maxpen__{name}"] or 0)
        unique = None  # filled below via separate unique/tied split if needed
        per_constraint.append(
            {
                "name": name,
                "max_penalty_count": max_count,
                "max_violation_count": int(row[f"maxviol__{name}"] or 0),
                "max_penalty_share_of_all_observed_rows": _share(
                    max_count, observed
                ),
                "max_penalty_share_of_penalized_rows": _share(
                    max_count, penalized_rows
                ),
            }
        )

    return {
        "rows_with_any_positive_penalty": penalized_rows,
        "rows_with_a_unique_maximum": int(row["unique_max_rows"] or 0),
        "rows_with_tied_maxima": int(row["tied_max_rows"] or 0),
        "max_penalty_vs_max_violation_disagreements": int(
            row["max_set_disagreements"] or 0
        ),
        "per_constraint": per_constraint,
        "_unique_placeholder": unique,
    }


def _distributions(
    engine: BruteForceQueryEngine,
    refs: list[dict[str, str]],
    *,
    sentinel: float,
    atol: float,
) -> list[dict[str, Any]]:
    view = quote_ident(engine.view_name)
    out: list[dict[str, Any]] = []

    def _stats(expr: str, *, where: str) -> dict[str, Any]:
        q_parts = [
            f"quantile_cont({expr}, {q}) AS q_{str(q).replace('.', '_')}"
            for q in _QUANTILES
        ]
        sql = (
            f"SELECT COUNT({expr}) AS n, "
            f"MIN({expr}) AS min_v, MAX({expr}) AS max_v, "
            f"AVG({expr}) AS mean_v, "
            + ", ".join(q_parts)
            + f" FROM {view} WHERE {where}"
        )
        row = engine.fetch_dicts(sql, max_rows=1)[0]
        return {
            "count": int(row["n"] or 0),
            "min": _as_float(row["min_v"]),
            "max": _as_float(row["max_v"]),
            "mean": _as_float(row["mean_v"]),
            "median": _as_float(row["q_0_5"]),
            "p90": _as_float(row["q_0_9"]),
            "p95": _as_float(row["q_0_95"]),
            "p99": _as_float(row["q_0_99"]),
        }

    for ref in refs:
        util = ref["util"]
        viol = ref["violation"]
        pen = ref["penalty"]
        is_sent = _sql_sentinel(util, sentinel=sentinel, atol=atol)
        finite_util = f"isfinite({util})"
        pos_viol = f"isfinite({viol}) AND {viol} > 0.0"
        pos_pen = f"isfinite({pen}) AND {pen} > 0.0"
        out.append(
            {
                "name": ref["name"],
                "utilization_including_sentinel": _stats(
                    util, where=finite_util
                ),
                "utilization_excluding_sentinel": _stats(
                    util, where=f"{finite_util} AND NOT {is_sent}"
                ),
                "positive_violation_including_sentinel": _stats(
                    viol, where=pos_viol
                ),
                "positive_violation_excluding_sentinel": _stats(
                    viol, where=f"{pos_viol} AND NOT {is_sent}"
                ),
                "penalty_including_sentinel": _stats(pen, where=pos_pen),
                "penalty_excluding_sentinel": _stats(
                    pen, where=f"{pos_pen} AND NOT {is_sent}"
                ),
            }
        )
    return out


def _near_boundary(
    engine: BruteForceQueryEngine,
    refs: list[dict[str, str]],
    *,
    low: float,
    high: float,
) -> list[dict[str, Any]]:
    view = quote_ident(engine.view_name)
    observed = engine.count_rows()
    out: list[dict[str, Any]] = []
    for ref in refs:
        util = ref["util"]
        feasible_band = (
            f"isfinite({util}) AND {util} >= ({low}) AND {util} <= 1.0"
        )
        violated_band = (
            f"isfinite({util}) AND {util} > 1.0 AND {util} <= ({high})"
        )
        row = engine.fetch_one(
            f"SELECT "
            f"SUM(CASE WHEN {feasible_band} THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN {violated_band} THEN 1 ELSE 0 END) "
            f"FROM {view}"
        )
        feas_count = int(row[0] or 0) if row else 0
        viol_count = int(row[1] or 0) if row else 0
        out.append(
            {
                "name": ref["name"],
                "feasible_near_boundary": {
                    "band": [low, 1.0],
                    "inclusive": [True, True],
                    "count": feas_count,
                    "share_of_observed": _share(feas_count, observed),
                },
                "violated_near_boundary": {
                    "band": [1.0, high],
                    "inclusive": [False, True],
                    "count": viol_count,
                    "share_of_observed": _share(viol_count, observed),
                },
            }
        )
    return out


def _simultaneous_histogram(
    engine: BruteForceQueryEngine,
    refs: list[dict[str, str]],
) -> dict[str, Any]:
    view = quote_ident(engine.view_name)
    if not refs:
        return {"counts_by_active_constraint_count": {"0": engine.count_rows()}}
    active_sum = " + ".join(
        f"CASE WHEN {_sql_active(ref['util'])} THEN 1 ELSE 0 END" for ref in refs
    )
    rows = engine.fetch_dicts(
        f"SELECT ({active_sum}) AS n_active, COUNT(*) AS row_count "
        f"FROM {view} GROUP BY 1 ORDER BY 1",
        max_rows=max(len(refs) + 1, 1),
    )
    counts = {str(int(row["n_active"])): int(row["row_count"]) for row in rows}
    for index in range(len(refs) + 1):
        counts.setdefault(str(index), 0)
    return {"counts_by_active_constraint_count": counts}


def _pairwise_coviolation(
    engine: BruteForceQueryEngine,
    refs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    view = quote_ident(engine.view_name)
    observed = engine.count_rows()
    pairs: list[dict[str, Any]] = []
    for left, right in combinations(refs, 2):
        both = (
            f"{_sql_active(left['util'])} AND {_sql_active(right['util'])}"
        )
        row = engine.fetch_one(
            f"SELECT SUM(CASE WHEN {both} THEN 1 ELSE 0 END) FROM {view}"
        )
        count = int(row[0] or 0) if row else 0
        pairs.append(
            {
                "left": left["name"],
                "right": right["name"],
                "co_violation_count": count,
                "co_violation_share_of_observed": _share(count, observed),
            }
        )
    return pairs


def _top_combinations(
    engine: BruteForceQueryEngine,
    refs: list[dict[str, str]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    view = quote_ident(engine.view_name)
    if not refs:
        return []
    pieces = [
        f"CASE WHEN {_sql_active(ref['util'])} THEN {quote_string(ref['name'])} END"
        for ref in refs
    ]
    combo_expr = f"CONCAT_WS(',', {', '.join(pieces)})"
    rows = engine.fetch_dicts(
        f"SELECT {combo_expr} AS combination, COUNT(*) AS row_count "
        f"FROM {view} "
        f"GROUP BY 1 "
        f"ORDER BY row_count DESC, combination ASC "
        f"LIMIT {int(limit)}",
        max_rows=limit,
    )
    observed = engine.count_rows()
    out: list[dict[str, Any]] = []
    for row in rows:
        raw = row["combination"]
        names = [] if raw in (None, "") else str(raw).split(",")
        out.append(
            {
                "constraints": names,
                "count": int(row["row_count"]),
                "share_of_observed": _share(int(row["row_count"]), observed),
            }
        )
    return out


def _penalty_burden(
    engine: BruteForceQueryEngine,
    refs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    view = quote_ident(engine.view_name)
    if not refs:
        return []
    select_parts: list[str] = []
    for ref in refs:
        pen = ref["penalty"]
        select_parts.append(
            f"SUM(CASE WHEN isfinite({pen}) AND {pen} > 0 THEN {pen} ELSE 0 END) "
            f"AS sum__{ref['name']}"
        )
        select_parts.append(
            f"AVG(CASE WHEN isfinite({pen}) AND {pen} > 0 THEN {pen} END) "
            f"AS mean__{ref['name']}"
        )
    row = engine.fetch_dicts(
        f"SELECT {', '.join(select_parts)} FROM {view}",
        max_rows=1,
    )[0]
    total_sum = sum(float(row[f"sum__{ref['name']}"] or 0.0) for ref in refs)
    out: list[dict[str, Any]] = []
    for ref in refs:
        name = ref["name"]
        s = float(row[f"sum__{name}"] or 0.0)
        out.append(
            {
                "name": name,
                "sum_positive_penalty": s,
                "mean_positive_penalty": _as_float(row[f"mean__{name}"]),
                "share_of_total_positive_penalty": (
                    None if total_sum <= 0 else s / total_sum
                ),
            }
        )
    return out


def run_constraints_analysis(
    context: BruteForceDatasetContext,
    engine: BruteForceQueryEngine,
    *,
    options: DatasetReportOptions | None = None,
) -> dict[str, Any]:
    """
    Compute constraint activity, sentinel, and penalty diagnostics.

    Parameters
    ----------
    context : BruteForceDatasetContext
        Sealed dataset context.
    engine : BruteForceQueryEngine
        Open DuckDB engine over the same dataset.
    options : DatasetReportOptions or None, optional
        Near-boundary bands, coviolation toggle, combination bound.

    Returns
    -------
    dict
        JSON-compatible ``constraints`` payload.
    """
    options = options or DatasetReportOptions()
    rtol = float(engine.scientific_rtol)
    atol = float(engine.sentinel_atol)
    sentinel = float(context.run_config.invalid_utilization)
    alpha = float(context.opt_penalty_alpha)
    beta = float(context.opt_penalty_beta)
    expected_violation = max(0.0, sentinel - 1.0)
    expected_penalty = alpha * (expected_violation**beta)
    refs = _constraint_refs(context)

    activity = _activity_and_sentinel(
        engine,
        refs,
        sentinel=sentinel,
        atol=atol,
        expected_violation=expected_violation,
        expected_penalty=expected_penalty,
        rtol=rtol,
    )
    utilization_shares = _utilization_shares(
        engine,
        refs,
        sentinel=sentinel,
        atol=atol,
        rtol=rtol,
    )
    maxima = _max_penalty_violation(engine, refs, rtol=rtol)
    maxima.pop("_unique_placeholder", None)

    # Merge max-penalty frequency into burden later; keep maxima block clean.
    distributions = _distributions(
        engine, refs, sentinel=sentinel, atol=atol
    )
    near_boundary = _near_boundary(
        engine,
        refs,
        low=float(options.near_boundary_low),
        high=float(options.near_boundary_high),
    )
    histogram = _simultaneous_histogram(engine, refs)
    pairwise = None
    if options.include_coviolation_matrix:
        pairwise = _pairwise_coviolation(engine, refs)
    combinations_out = _top_combinations(
        engine,
        refs,
        limit=int(options.max_violation_combinations),
    )
    burden = _penalty_burden(engine, refs)
    max_by_name = {
        item["name"]: item["max_penalty_count"]
        for item in maxima["per_constraint"]
    }
    for item in burden:
        item["max_penalty_count"] = int(max_by_name.get(item["name"], 0))

    activity_disagreements = sum(
        int(item["activity_signal_disagreements"])
        for item in activity["per_constraint"]
    )
    sentinel_formula_bad = sum(
        int(item["sentinel_formula_mismatches"])
        for item in activity["per_constraint"]
    )

    report = {
        "section": "constraints",
        "metadata": {
            "quantile_method": QUANTILE_METHOD,
            "scientific_rtol": rtol,
            "sentinel_atol": atol,
            "opt_invalid_utilization": sentinel,
            "expected_invalid_violation": expected_violation,
            "expected_invalid_penalty": expected_penalty,
            "opt_penalty_alpha": alpha,
            "opt_penalty_beta": beta,
            "near_boundary_low": float(options.near_boundary_low),
            "near_boundary_high": float(options.near_boundary_high),
            "activity_definition": "finite(util) and util > 1",
        },
        "activity": activity,
        "utilization_shares": utilization_shares,
        "maxima": maxima,
        "distributions": distributions,
        "near_boundary": near_boundary,
        "simultaneous_violation_histogram": histogram,
        "pairwise_coviolation": pairwise,
        "complete_violation_combinations": combinations_out,
        "penalty_burden": burden,
        "ok": (
            activity_disagreements == 0
            and sentinel_formula_bad == 0
            and int(maxima["max_penalty_vs_max_violation_disagreements"]) == 0
        ),
    }
    return normalize_report_value(report)
