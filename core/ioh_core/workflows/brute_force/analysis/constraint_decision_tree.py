"""
Infeasible repair decision trees from sealed brute-force evaluations.

v1 scope (feasible volume-seeking deferred):

- State: violation pattern over active A–D utils (Z excluded) plus leading
  constraint ``argmax η``.
- Actions: one-factor ±1 catalogue-index moves.
- Success: neighbour reduces ``max η`` among active constraints.
- Models: pattern→move association table and a shallow CART (depth 2–4)
  predicting the best successful action from ``over_*`` flags + leading.

Designed for full factorial BF today and non-factorial samples later
(neighbour joins only when the neighbour row exists).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.ioh_core.constraint_selection import warn_if_constraint_subset
from core.execution_core.atomic_io import write_text_atomic
from core.ioh_core.workflows.brute_force.analysis.context import (
    open_brute_force_dataset,
)
from core.ioh_core.workflows.brute_force.analysis.engine import (
    BruteForceQueryEngine,
    open_query_engine,
    quote_ident,
)
from core.ioh_core.workflows.brute_force.analysis.keep_list import (
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

METHOD_ID = "constraint_decision_tree_repair_v1"
METHOD_TITLE = "Infeasible repair decision tree (violation patterns → moves)"
SCHEMA_VERSION = 1

DECISION_TREE_DIRNAME = "constraint_decision_tree"
RESULTS_JSON = "results.json"
RULES_MD = "rules.md"
RULES_CSV = "rules.csv"
SUCCESS_MARKER = "_SUCCESS"

DEFAULT_MAX_DEPTH = 3
DEFAULT_MIN_SUPPORT = 50
DEFAULT_MIN_LIFT = 1.2
DEFAULT_SAMPLE_ROWS = 100_000
DEFAULT_MIN_SAMPLES_LEAF = 25

_SEEDS_TABLE = "bf_repair_seeds"
_ACTION_TABLE = "bf_repair_actions"
_LABELED_TABLE = "bf_repair_labeled"


def _active_constraint_names(context: BruteForceDatasetContext) -> tuple[str, ...]:
    """Active util stems excluding Z-family constraints unless a keep list is set."""
    selected = getattr(context, "selected_constraint_names", None)
    if selected is not None:
        return tuple(selected)
    return tuple(
        name
        for name in context.constraint_names
        if not str(name).startswith("Z")
    )


def _util_col(name: str) -> str:
    return f"util__{name}"


def _idx_col(name: str) -> str:
    return f"idx__{name}"


def _var_bounds(context: BruteForceDatasetContext) -> dict[str, tuple[int, int]]:
    bounds: dict[str, tuple[int, int]] = {}
    model = context.problem_info["model"]
    for var_name in context.var_names:
        meta = model[var_name]
        bounds[var_name] = (int(meta["lb"]), int(meta["ub"]))
    return bounds


def _sql_max_util(active: Sequence[str]) -> str:
    cols = ", ".join(quote_ident(_util_col(n)) for n in active)
    return f"GREATEST({cols})"


def _sql_leading(active: Sequence[str]) -> str:
    """Deterministic argmax η; earlier catalogue names win exact ties."""
    if not active:
        return "CAST(NULL AS VARCHAR)"
    if len(active) == 1:
        return f"'{active[0]}'"
    parts: list[str] = []
    for i, name in enumerate(active):
        col = quote_ident(_util_col(name))
        others = [n for n in active if n != name]
        ge_others = " AND ".join(
            f"{col} >= {quote_ident(_util_col(o))}" for o in others
        )
        earlier = active[:i]
        if earlier:
            gt_earlier = " AND ".join(
                f"{col} > {quote_ident(_util_col(e))}" for e in earlier
            )
            parts.append(f"WHEN ({ge_others}) AND ({gt_earlier}) THEN '{name}'")
        else:
            parts.append(f"WHEN ({ge_others}) THEN '{name}'")
    return "CASE " + " ".join(parts) + f" ELSE '{active[0]}' END"


def _sql_pattern(active: Sequence[str]) -> str:
    """Sorted '|' joined names with η > 1; empty → ''."""
    pieces = [
        f"CASE WHEN {quote_ident(_util_col(n))} > 1.0 THEN '{n}|' ELSE '' END"
        for n in active
    ]
    return "regexp_replace(" + " || ".join(pieces) + ", '\\|$', '')"


def _sql_over_flags(active: Sequence[str]) -> str:
    return ", ".join(
        f"CAST({quote_ident(_util_col(n))} > 1.0 AS INTEGER) AS "
        f"{quote_ident(f'over__{n}')}"
        for n in active
    )


@dataclass
class _TreeNode:
    n_samples: int
    prediction: str
    class_counts: dict[str, int]
    feature: str | None = None
    # For binary over_* : threshold 0.5, left is feature==0
    # For leading__{name}: left is leading==name
    left_value: Any = None
    left: _TreeNode | None = None
    right: _TreeNode | None = None

    def is_leaf(self) -> bool:
        return self.feature is None


def _gini(counts: Counter[str], n: int) -> float:
    if n <= 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def _majority(counts: Counter[str]) -> str:
    if not counts:
        return "none"
    # Deterministic: highest count, then name
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _fit_cart(
    rows: list[dict[str, Any]],
    *,
    feature_names: Sequence[str],
    leading_names: Sequence[str],
    max_depth: int,
    min_samples_leaf: int,
) -> _TreeNode:
    """Shallow CART on binary over_* features and leading one-vs-rest splits."""

    def _counts(data: list[dict[str, Any]]) -> Counter[str]:
        return Counter(str(r["best_action"]) for r in data)

    def _grow(data: list[dict[str, Any]], depth: int) -> _TreeNode:
        counts = _counts(data)
        n = len(data)
        pred = _majority(counts)
        node = _TreeNode(
            n_samples=n,
            prediction=pred,
            class_counts=dict(sorted(counts.items())),
        )
        if (
            depth >= max_depth
            or n < 2 * int(min_samples_leaf)
            or len(counts) <= 1
        ):
            return node

        parent_gini = _gini(counts, n)
        best_gain = 0.0
        best: tuple[str, Any, list[dict], list[dict]] | None = None

        # Binary over_* features
        for feat in feature_names:
            left = [r for r in data if int(r[feat]) == 0]
            right = [r for r in data if int(r[feat]) == 1]
            if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                continue
            g = (
                len(left) / n * _gini(_counts(left), len(left))
                + len(right) / n * _gini(_counts(right), len(right))
            )
            gain = parent_gini - g
            if gain > best_gain + 1e-15 or (
                abs(gain - best_gain) <= 1e-15
                and best is not None
                and feat < best[0]
            ):
                best_gain = gain
                best = (feat, 0, left, right)

        # Leading one-vs-rest
        for lead in leading_names:
            feat = f"leading__{lead}"
            left = [r for r in data if str(r["leading"]) == lead]
            right = [r for r in data if str(r["leading"]) != lead]
            if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                continue
            g = (
                len(left) / n * _gini(_counts(left), len(left))
                + len(right) / n * _gini(_counts(right), len(right))
            )
            gain = parent_gini - g
            if gain > best_gain + 1e-15 or (
                abs(gain - best_gain) <= 1e-15
                and best is not None
                and feat < best[0]
            ):
                best_gain = gain
                best = (feat, lead, left, right)

        if best is None or best_gain <= 1e-12:
            return node

        feat, left_value, left_rows, right_rows = best
        node.feature = feat
        node.left_value = left_value
        node.left = _grow(left_rows, depth + 1)
        node.right = _grow(right_rows, depth + 1)
        return node

    return _grow(rows, 0)


def _extract_paths(
    node: _TreeNode,
    *,
    prior: Mapping[str, float],
    min_support: int,
    min_lift: float,
    path: list[str] | None = None,
) -> list[dict[str, Any]]:
    path = list(path or [])
    if node.is_leaf():
        n = int(node.n_samples)
        if n < int(min_support):
            return []
        pred = str(node.prediction)
        conf = float(node.class_counts.get(pred, 0)) / float(n)
        base = float(prior.get(pred, 0.0))
        lift = (conf / base) if base > 0.0 else float("inf")
        if lift < float(min_lift):
            return []
        return [
            {
                "path": list(path),
                "path_str": " AND ".join(path) if path else "(root)",
                "action": pred,
                "support": n,
                "confidence": conf,
                "lift": None if not math.isfinite(lift) else float(lift),
                "class_counts": dict(node.class_counts),
            }
        ]

    assert node.left is not None and node.right is not None
    feat = str(node.feature)
    if feat.startswith("over__"):
        left_cond = f"{feat}=0"
        right_cond = f"{feat}=1"
    elif feat.startswith("leading__"):
        name = feat[len("leading__") :]
        left_cond = f"leading={name}"
        right_cond = f"leading!={name}"
    else:
        left_cond = f"{feat}={node.left_value!r}"
        right_cond = f"{feat}!={node.left_value!r}"

    out: list[dict[str, Any]] = []
    out.extend(
        _extract_paths(
            node.left,
            prior=prior,
            min_support=min_support,
            min_lift=min_lift,
            path=path + [left_cond],
        )
    )
    out.extend(
        _extract_paths(
            node.right,
            prior=prior,
            min_support=min_support,
            min_lift=min_lift,
            path=path + [right_cond],
        )
    )
    return out


def _serialize_tree(node: _TreeNode) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "n_samples": int(node.n_samples),
        "prediction": str(node.prediction),
        "class_counts": dict(node.class_counts),
    }
    if node.is_leaf():
        payload["leaf"] = True
        return payload
    payload["leaf"] = False
    payload["feature"] = node.feature
    payload["left_value"] = node.left_value
    payload["left"] = _serialize_tree(node.left)  # type: ignore[arg-type]
    payload["right"] = _serialize_tree(node.right)  # type: ignore[arg-type]
    return payload


@dataclass(frozen=True)
class DecisionTreeOptions:
    """Tunables for repair decision-tree analysis."""

    max_depth: int = DEFAULT_MAX_DEPTH
    min_support: int = DEFAULT_MIN_SUPPORT
    min_lift: float = DEFAULT_MIN_LIFT
    sample_rows: int = DEFAULT_SAMPLE_ROWS
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF
    duckdb_threads: int = 1
    duckdb_memory_limit: str | None = None
    duckdb_temp_directory: Path | None = None

    def __post_init__(self) -> None:
        if self.max_depth < 1 or self.max_depth > 8:
            raise ValueError(f"max_depth must be in 1..8, got {self.max_depth}")
        if self.min_support < 1:
            raise ValueError("min_support must be >= 1")
        if self.min_lift < 0.0:
            raise ValueError("min_lift must be >= 0")
        if self.sample_rows < 1:
            raise ValueError("sample_rows must be >= 1")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if self.duckdb_threads < 1:
            raise ValueError("duckdb_threads must be >= 1")


def _seed_select_sql(
    context: BruteForceDatasetContext,
    active: Sequence[str],
    *,
    from_relation: str,
    where_extra: str = "",
) -> str:
    idx_select = ", ".join(
        quote_ident(_idx_col(v)) for v in context.var_names
    )
    util_select = ", ".join(quote_ident(_util_col(c)) for c in active)
    max_expr = _sql_max_util(active)
    lead_expr = _sql_leading(active)
    pat_expr = _sql_pattern(active)
    over_expr = _sql_over_flags(active)
    finite_utils = " AND ".join(
        f"isfinite({quote_ident(_util_col(c))})" for c in active
    )
    where = (
        f"{quote_ident('analysis_ok')} AND {finite_utils}"
        + (f" AND ({where_extra})" if where_extra else "")
    )
    return (
        f"SELECT "
        f"  {quote_ident('global_index')} AS global_index, "
        f"  {idx_select}, "
        f"  {util_select}, "
        f"  ({max_expr}) AS max_util, "
        f"  ({lead_expr}) AS {quote_ident('leading')}, "
        f"  ({pat_expr}) AS {quote_ident('pattern')}, "
        f"  {over_expr} "
        f"FROM {from_relation} "
        f"WHERE {where}"
    )


def _sample_infeasible_seeds(
    engine: BruteForceQueryEngine,
    context: BruteForceDatasetContext,
    active: Sequence[str],
    *,
    sample_rows: int,
) -> tuple[int, int]:
    """
    Sample infeasible seeds.

    Returns
    -------
    (n_analysis_ok_finite, n_seeds)
    """
    view = quote_ident(engine.view_name)
    seeds = quote_ident(_SEEDS_TABLE)
    conn = engine.connection
    base_sql = _seed_select_sql(context, active, from_relation=view)
    # Count eligible analysis_ok rows with finite utils (not only infeasible).
    count_row = engine.fetch_one(
        f"SELECT COUNT(*) FROM ({base_sql}) AS _base"
    )
    n_base = int(count_row[0] or 0) if count_row else 0

    conn.execute(f"DROP TABLE IF EXISTS {seeds}")
    conn.execute(
        f"CREATE TEMP TABLE {seeds} AS "
        f"SELECT * FROM ({base_sql}) AS _base "
        f"WHERE max_util > 1.0 "
        f"ORDER BY hash(global_index), global_index "
        f"LIMIT {int(sample_rows)}"
    )
    seed_row = engine.fetch_one(f"SELECT COUNT(*) FROM {seeds}")
    n_seeds = int(seed_row[0] or 0) if seed_row else 0
    return n_base, n_seeds


def _score_actions(
    engine: BruteForceQueryEngine,
    context: BruteForceDatasetContext,
    active: Sequence[str],
) -> int:
    """Create action rows for ±1 neighbours that exist in the sealed view."""
    view = quote_ident(engine.view_name)
    seeds = quote_ident(_SEEDS_TABLE)
    actions = quote_ident(_ACTION_TABLE)
    bounds = _var_bounds(context)
    max_expr_n = _sql_max_util(active)
    # Qualify util columns for neighbour alias ``n``.
    max_expr_n_qual = "GREATEST(" + ", ".join(
        f"n.{quote_ident(_util_col(c))}" for c in active
    ) + ")"
    conn = engine.connection
    conn.execute(f"DROP TABLE IF EXISTS {actions}")
    conn.execute(
        f"CREATE TEMP TABLE {actions} ("
        f"  seed_global_index BIGINT, "
        f'  "pattern" VARCHAR, '
        f'  "leading" VARCHAR, '
        f"  max_util_before DOUBLE, "
        f'  "action" VARCHAR, '
        f"  max_util_after DOUBLE, "
        f"  success BOOLEAN"
        f")"
    )

    finite_n = " AND ".join(
        f"isfinite(n.{quote_ident(_util_col(c))})" for c in active
    )

    for var_name in context.var_names:
        lb, ub = bounds[var_name]
        idx_v = quote_ident(_idx_col(var_name))
        other_eq = " AND ".join(
            f"n.{quote_ident(_idx_col(v))} = s.{quote_ident(_idx_col(v))}"
            for v in context.var_names
            if v != var_name
        )
        if not other_eq:
            other_eq = "TRUE"

        for direction, delta, bound_pred in (
            ("inc", "+ 1", f"s.{idx_v} < {ub}"),
            ("dec", "- 1", f"s.{idx_v} > {lb}"),
        ):
            conn.execute(
                f"INSERT INTO {actions} "
                f"SELECT "
                f"  s.global_index, s.{quote_ident('pattern')}, "
                f"  s.{quote_ident('leading')}, s.max_util, "
                f"  '{direction}__{var_name}', ({max_expr_n_qual}), "
                f"  (({max_expr_n_qual}) < s.max_util) "
                f"FROM {seeds} s "
                f"INNER JOIN {view} n "
                f"  ON n.{quote_ident('analysis_ok')} "
                f" AND {finite_n} "
                f" AND n.{idx_v} = s.{idx_v} {delta} "
                f" AND {other_eq} "
                f"WHERE {bound_pred}"
            )

    labeled = quote_ident(_LABELED_TABLE)
    over_select = ", ".join(
        f"s.{quote_ident(f'over__{c}')}" for c in active
    )
    conn.execute(f"DROP TABLE IF EXISTS {labeled}")
    conn.execute(
        f"CREATE TEMP TABLE {labeled} AS "
        f"WITH ranked AS ("
        f"  SELECT a.*, "
        f"    ROW_NUMBER() OVER ("
        f"      PARTITION BY a.seed_global_index "
        f"      ORDER BY a.success DESC, "
        f"        (a.max_util_before - a.max_util_after) DESC, "
        f"        a.{quote_ident('action')} ASC"
        f"    ) AS rn "
        f"  FROM {actions} a "
        f"  WHERE a.success"
        f") "
        f"SELECT "
        f"  r.seed_global_index AS global_index, "
        f"  r.{quote_ident('pattern')} AS pattern, "
        f"  r.{quote_ident('leading')} AS leading, "
        f"  r.{quote_ident('action')} AS best_action, "
        f"  r.max_util_before, r.max_util_after, "
        f"  {over_select} "
        f"FROM ranked r "
        f"INNER JOIN {seeds} s ON s.global_index = r.seed_global_index "
        f"WHERE r.rn = 1"
    )
    _ = max_expr_n  # kept for readability / future neighbour-side filters
    row = engine.fetch_one(f"SELECT COUNT(*) FROM {labeled}")
    return int(row[0] or 0) if row else 0


def _pattern_move_table(
    engine: BruteForceQueryEngine,
    *,
    min_support: int,
    min_lift: float,
) -> list[dict[str, Any]]:
    actions = quote_ident(_ACTION_TABLE)
    sql = f"""
    WITH base AS (
      SELECT "pattern" AS pattern, "action" AS action,
             COUNT(*) AS n_tried,
             SUM(CASE WHEN success THEN 1 ELSE 0 END) AS n_success
      FROM {actions}
      GROUP BY 1, 2
    ),
    prior AS (
      SELECT action,
             SUM(n_success)::DOUBLE / NULLIF(SUM(n_tried), 0) AS p_success
      FROM base
      GROUP BY 1
    ),
    scored AS (
      SELECT b.pattern, b.action, b.n_tried, b.n_success,
             b.n_success::DOUBLE / NULLIF(b.n_tried, 0) AS confidence,
             CASE
               WHEN p.p_success IS NULL OR p.p_success = 0 THEN NULL
               ELSE (b.n_success::DOUBLE / NULLIF(b.n_tried, 0)) / p.p_success
             END AS lift
      FROM base b
      LEFT JOIN prior p ON p.action = b.action
    )
    SELECT pattern, action, n_tried, n_success, confidence, lift
    FROM scored
    WHERE n_tried >= {int(min_support)}
      AND confidence IS NOT NULL
      AND (lift IS NULL OR lift >= {float(min_lift)})
    ORDER BY pattern ASC, lift DESC NULLS LAST, confidence DESC, action ASC
    """
    rows = engine.fetch_rows(sql, max_rows=50_000)
    out: list[dict[str, Any]] = []
    for pattern, action, n_tried, n_success, confidence, lift in rows:
        out.append(
            {
                "pattern": str(pattern or ""),
                "action": str(action),
                "n_tried": int(n_tried),
                "n_success": int(n_success),
                "confidence": float(confidence),
                "lift": None if lift is None else float(lift),
            }
        )
    return out


def _load_labeled_rows(
    engine: BruteForceQueryEngine,
    active: Sequence[str],
) -> list[dict[str, Any]]:
    labeled = quote_ident(_LABELED_TABLE)
    over_cols = [f"over__{c}" for c in active]
    select_parts = [
        "global_index",
        quote_ident("pattern"),
        quote_ident("leading"),
        "best_action",
        *[quote_ident(c) for c in over_cols],
    ]
    select = ", ".join(select_parts)
    raw = engine.fetch_rows(
        f"SELECT {select} FROM {labeled}", max_rows=200_000
    )
    rows: list[dict[str, Any]] = []
    for tup in raw:
        item = {
            "global_index": int(tup[0]),
            "pattern": str(tup[1] or ""),
            "leading": str(tup[2]),
            "best_action": str(tup[3]),
        }
        for i, name in enumerate(over_cols):
            item[name] = int(tup[4 + i])
        rows.append(item)
    return rows


def _rules_markdown(
    *,
    tree_rules: Sequence[Mapping[str, Any]],
    pattern_moves: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> str:
    lines = [
        f"# {METHOD_TITLE}",
        "",
        f"- method_id: `{METHOD_ID}`",
        f"- schema_version: {SCHEMA_VERSION}",
        f"- max_depth: {meta.get('max_depth')}",
        f"- min_support: {meta.get('min_support')}",
        f"- min_lift: {meta.get('min_lift')}",
        f"- sample_rows_requested: {meta.get('sample_rows')}",
        f"- seeds: {meta.get('n_seeds')}",
        f"- labeled_seeds: {meta.get('n_labeled')}",
        "",
        "## Tree rules (filtered)",
        "",
    ]
    if not tree_rules:
        lines.append("_No tree paths passed support/lift thresholds._")
    else:
        for i, rule in enumerate(tree_rules, start=1):
            lines.append(
                f"{i}. `{rule['path_str']}` → **{rule['action']}** "
                f"(support={rule['support']}, "
                f"conf={rule['confidence']:.3f}, "
                f"lift={rule['lift']})"
            )
    lines.extend(["", "## Pattern → move table (filtered)", ""])
    if not pattern_moves:
        lines.append("_No pattern/move pairs passed thresholds._")
    else:
        lines.append("| pattern | action | n_tried | n_success | conf | lift |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in pattern_moves[:200]:
            pat = row["pattern"] or "(none)"
            lift = row["lift"]
            lift_s = "" if lift is None else f"{lift:.3f}"
            lines.append(
                f"| `{pat}` | `{row['action']}` | {row['n_tried']} | "
                f"{row['n_success']} | {row['confidence']:.3f} | {lift_s} |"
            )
        if len(pattern_moves) > 200:
            lines.append("")
            lines.append(f"_… {len(pattern_moves) - 200} more rows in results.json_")
    lines.append("")
    return "\n".join(lines)


def _rules_csv(tree_rules: Sequence[Mapping[str, Any]]) -> str:
    lines = ["path,action,support,confidence,lift"]
    for rule in tree_rules:
        lift = rule.get("lift")
        lift_s = "" if lift is None else f"{float(lift):.6g}"
        path = str(rule.get("path_str", "")).replace('"', "'")
        lines.append(
            f"\"{path}\",{rule.get('action')},{rule.get('support')},"
            f"{float(rule.get('confidence', 0.0)):.6g},{lift_s}"
        )
    return "\n".join(lines) + "\n"


def run_constraint_decision_tree_analysis(
    experiment_dir: Path,
    *,
    options: DecisionTreeOptions | None = None,
    engine: BruteForceQueryEngine | None = None,
    selected_constraint_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Mine infeasible repair trees/rules from a sealed BF experiment.
    """
    options = options or DecisionTreeOptions()
    context = open_brute_force_dataset(Path(experiment_dir))
    if selected_constraint_names is not None:
        context = attach_selected_constraint_names(
            context, selected_constraint_names
        )
        active = tuple(selected_constraint_names)
    else:
        active = _active_constraint_names(context)
        warn_if_constraint_subset(context.constraint_names, active)
    if len(active) < 1:
        raise ValueError("no active non-Z constraints available for repair trees")

    own_engine = engine is None
    if engine is None:
        report_opts = DatasetReportOptions(
            duckdb_threads=options.duckdb_threads,
            duckdb_memory_limit=options.duckdb_memory_limit,
            duckdb_temp_directory=options.duckdb_temp_directory,
        )
        engine = open_query_engine(context, options=report_opts)

    try:
        n_base, n_seeds = _sample_infeasible_seeds(
            engine,
            context,
            active,
            sample_rows=options.sample_rows,
        )
        n_labeled = _score_actions(engine, context, active) if n_seeds else 0
        pattern_moves = (
            _pattern_move_table(
                engine,
                min_support=options.min_support,
                min_lift=options.min_lift,
            )
            if n_seeds
            else []
        )
        labeled_rows = _load_labeled_rows(engine, active) if n_labeled else []
    finally:
        if own_engine:
            engine.close()

    over_features = [f"over__{c}" for c in active]
    prior_counts = Counter(r["best_action"] for r in labeled_rows)
    n_lab = max(len(labeled_rows), 1)
    prior = {k: v / n_lab for k, v in prior_counts.items()}

    tree_rules: list[dict[str, Any]] = []
    tree_json: dict[str, Any] | None = None
    if labeled_rows:
        tree = _fit_cart(
            labeled_rows,
            feature_names=over_features,
            leading_names=active,
            max_depth=options.max_depth,
            min_samples_leaf=options.min_samples_leaf,
        )
        tree_json = _serialize_tree(tree)
        tree_rules = _extract_paths(
            tree,
            prior=prior,
            min_support=options.min_support,
            min_lift=options.min_lift,
        )

    meta = {
        "method_id": METHOD_ID,
        "method_title": METHOD_TITLE,
        "schema_version": SCHEMA_VERSION,
        "regime": "repair_infeasible",
        "active_constraints": list(active),
        "variables": list(context.var_names),
        "max_depth": options.max_depth,
        "min_support": options.min_support,
        "min_lift": options.min_lift,
        "min_samples_leaf": options.min_samples_leaf,
        "sample_rows": options.sample_rows,
        "n_analysis_ok_finite": n_base,
        "n_seeds": n_seeds,
        "n_labeled": n_labeled,
        "n_tree_rules": len(tree_rules),
        "n_pattern_move_rows": len(pattern_moves),
        "action_prior": prior,
    }
    document = {
        "ok": True,
        "overall_status": "success",
        "metadata": meta,
        "tree": tree_json,
        "tree_rules": tree_rules,
        "pattern_moves": pattern_moves,
    }
    return normalize_report_value(
        merge_constraint_provenance(
            document,
            sealed=context.constraint_names,
            selected=active,
        )
    )


def _resolve_output_dir(
    experiment_dir: Path, output_dir: Path | None
) -> Path:
    experiment_dir = Path(experiment_dir)
    from core.execution_core.output_layout import (
        catalogue_analysis_publication_dir,
        require_results_publication,
        resolve_results_root,
    )

    if output_dir is not None:
        out_dir = require_results_publication(
            Path(output_dir), error_type=ValueError
        )
    else:
        out_dir = require_results_publication(
            catalogue_analysis_publication_dir(
                resolve_results_root(experiment_dir),
                experiment_dir.name,
            )
            / DECISION_TREE_DIRNAME
        )
    dataset_dir = (experiment_dir / "dataset").resolve()
    try:
        out_dir.resolve().relative_to(dataset_dir)
    except ValueError:
        return out_dir
    raise ValueError(f"output_dir must not be inside dataset/: {out_dir}")


def write_constraint_decision_tree_artifacts(
    experiment_dir: Path,
    document: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Write JSON/MD/CSV artifacts under ``constraint_decision_tree/``."""
    out = _resolve_output_dir(Path(experiment_dir), output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    json_path = out / RESULTS_JSON
    write_text_atomic(json_path, dumps_report_json(document))
    written["results_json"] = json_path

    meta = document.get("metadata") or {}
    md_path = out / RULES_MD
    write_text_atomic(
        md_path,
        _rules_markdown(
            tree_rules=list(document.get("tree_rules") or []),
            pattern_moves=list(document.get("pattern_moves") or []),
            meta=meta if isinstance(meta, Mapping) else {},
        ),
    )
    written["rules_md"] = md_path

    csv_path = out / RULES_CSV
    write_text_atomic(
        csv_path, _rules_csv(list(document.get("tree_rules") or []))
    )
    written["rules_csv"] = csv_path

    marker = out / SUCCESS_MARKER
    write_text_atomic(marker, "ok\n")
    written["success"] = marker
    return written


def run_and_write_constraint_decision_tree(
    experiment_dir: Path,
    *,
    options: DecisionTreeOptions | None = None,
    output_dir: Path | None = None,
    selected_constraint_names: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Run analysis and publish artifacts."""
    document = run_constraint_decision_tree_analysis(
        experiment_dir,
        options=options,
        selected_constraint_names=selected_constraint_names,
    )
    written = write_constraint_decision_tree_artifacts(
        experiment_dir, document, output_dir=output_dir
    )
    return document, written
