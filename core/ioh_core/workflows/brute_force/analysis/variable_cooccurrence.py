"""
Variable-level co-occurrence among feasible / top-ranked feasible designs.

For each selected design, every unordered variable pair contributes +1 to the
matrix cell of the two concrete levels present in that design. Subsets are:

- ``all_feasible`` — every feasible row with finite ``y_p``
- ``top_{p}_percent`` — best ``ceil(p/100 * N)`` feasible rows by ascending
  ``y_p`` (ties broken by ``global_index``)

This is distinct from Layer-1 near-optimum bands (``y_p <= y_p* · (1+tol%)``).
"""
from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    analysis_constraint_names,
)
from core.ioh_core.workflows.brute_force.analysis.models import (
    BruteForceDatasetContext,
    DatasetReportOptions,
)
from core.ioh_core.workflows.brute_force.analysis.serialization import (
    dumps_report_json,
    normalize_report_value,
)

COOCCURRENCE_DIRNAME = "variable_cooccurrence"
COOCCURRENCE_RESULTS_JSON = "results.json"
COOCCURRENCE_SUMMARY_MD = "summary.md"
COOCCURRENCE_SUCCESS_MARKER = "_SUCCESS"

DEFAULT_TOP_PERCENTS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 50.0)
_SELECTED_TABLE = "bf_feasible_ranked"


def _variable_levels(
    context: BruteForceDatasetContext,
) -> list[dict[str, Any]]:
    """Ordered axis entries: one per declared catalogue level."""
    labels: list[dict[str, Any]] = []
    for var_name in context.var_names:
        meta = context.problem_info["model"][var_name]
        values = list(meta["values"])
        lower = int(meta["lb"])
        upper = int(meta["ub"])
        for index in range(lower, upper + 1):
            value = values[index]
            value_str = "null" if value is None else str(value)
            labels.append(
                {
                    "variable": var_name,
                    "index": int(index),
                    "value": None if value is None else value,
                    "label": f"{var_name}={value_str}",
                }
            )
    return labels


def _label_key(variable: str, index: int) -> tuple[str, int]:
    return (variable, int(index))


def _materialize_feasible_ranked(
    engine: BruteForceQueryEngine,
    context: BruteForceDatasetContext,
) -> int:
    """
    Build a temp table of feasible rows with dense rank by ascending ``y_p``.

    Returns
    -------
    int
        Number of selected feasible rows (finite ``y_p``).
    """
    view = quote_ident(engine.view_name)
    table = quote_ident(_SELECTED_TABLE)
    idx_cols = [quote_ident(f"idx__{name}") for name in context.var_names]
    select_idx = ", ".join(idx_cols)
    conn = engine.connection
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    # Rank only among feasible + finite y_p; deterministic tie-break.
    conn.execute(
        f"CREATE TEMP TABLE {table} AS "
        f"SELECT "
        f"  {quote_ident('global_index')} AS global_index, "
        f"  {quote_ident('y_p')} AS y_p, "
        f"  ROW_NUMBER() OVER ("
        f"    ORDER BY {quote_ident('y_p')} ASC, "
        f"    {quote_ident('global_index')} ASC"
        f"  ) AS rank_asc, "
        f"  {select_idx} "
        f"FROM {view} "
        f"WHERE {quote_ident('feasible')} "
        f"  AND isfinite({quote_ident('y_p')})"
    )
    row = engine.fetch_one(f"SELECT COUNT(*) FROM {table}")
    return int(row[0] or 0) if row else 0


def _top_count(feasible_count: int, percent: float) -> int:
    if feasible_count <= 0:
        return 0
    return int(math.ceil(float(percent) / 100.0 * float(feasible_count)))


def _empty_matrix(n: int) -> list[list[int]]:
    return [[0 for _ in range(n)] for _ in range(n)]


def _empty_float_matrix(n: int) -> list[list[float | None]]:
    return [[None for _ in range(n)] for _ in range(n)]


def _fill_pair_stats(
    engine: BruteForceQueryEngine,
    *,
    left_var: str,
    right_var: str,
    rank_limit: int | None,
    key_to_pos: Mapping[tuple[str, int], int],
    count_matrix: list[list[int]],
    sum_y_matrix: list[list[float]],
    min_y_matrix: list[list[float | None]],
) -> None:
    """Accumulate pair counts, Σ y_p, and min y_p into symmetric matrices."""
    table = quote_ident(_SELECTED_TABLE)
    left = quote_ident(f"idx__{left_var}")
    right = quote_ident(f"idx__{right_var}")
    where = ""
    if rank_limit is not None:
        where = f" WHERE rank_asc <= {int(rank_limit)}"
    rows = engine.fetch_dicts(
        f"SELECT {left} AS left_index, {right} AS right_index, "
        f"COUNT(*) AS row_count, "
        f"SUM({quote_ident('y_p')}) AS sum_y_p, "
        f"MIN({quote_ident('y_p')}) AS min_y_p "
        f"FROM {table}{where} "
        f"GROUP BY 1, 2",
        max_rows=10_000,
    )
    for row in rows:
        left_idx = int(row["left_index"])
        right_idx = int(row["right_index"])
        count = int(row["row_count"])
        sum_y = float(row["sum_y_p"])
        min_y = float(row["min_y_p"])
        i = key_to_pos[_label_key(left_var, left_idx)]
        j = key_to_pos[_label_key(right_var, right_idx)]
        count_matrix[i][j] += count
        sum_y_matrix[i][j] += sum_y
        prev_ij = min_y_matrix[i][j]
        min_y_matrix[i][j] = (
            min_y if prev_ij is None else min(float(prev_ij), min_y)
        )
        if i != j:
            count_matrix[j][i] += count
            sum_y_matrix[j][i] += sum_y
            prev_ji = min_y_matrix[j][i]
            min_y_matrix[j][i] = (
                min_y if prev_ji is None else min(float(prev_ji), min_y)
            )


def _mean_from_sum(
    count_matrix: Sequence[Sequence[int]],
    sum_y_matrix: Sequence[Sequence[float]],
) -> list[list[float | None]]:
    n = len(count_matrix)
    means = _empty_float_matrix(n)
    for i in range(n):
        for j in range(n):
            count = int(count_matrix[i][j])
            if count > 0:
                means[i][j] = float(sum_y_matrix[i][j]) / float(count)
    return means


def _fetch_top_designs(
    engine: BruteForceQueryEngine,
    context: BruteForceDatasetContext,
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Return the best ``top_k`` feasible designs (ascending ``y_p``).

    Each record includes ``rank``, ``global_index``, ``y_p``, and ``indices``.
    """
    if top_k <= 0:
        return []
    table = quote_ident(_SELECTED_TABLE)
    idx_select = ", ".join(
        f"{quote_ident(f'idx__{name}')} AS {quote_ident(f'idx__{name}')}"
        for name in context.var_names
    )
    rows = engine.fetch_dicts(
        f"SELECT global_index, y_p, rank_asc"
        f"{(', ' + idx_select) if idx_select else ''} "
        f"FROM {table} "
        f"WHERE rank_asc <= {int(top_k)} "
        f"ORDER BY rank_asc ASC",
        max_rows=max(int(top_k), 1),
    )
    designs: list[dict[str, Any]] = []
    for row in rows:
        indices = {
            name: None if row.get(f"idx__{name}") is None else int(row[f"idx__{name}"])
            for name in context.var_names
        }
        designs.append(
            {
                "rank": int(row["rank_asc"]),
                "global_index": int(row["global_index"]),
                "y_p": float(row["y_p"]),
                "indices": indices,
            }
        )
    return designs


def _subset_payload(
    engine: BruteForceQueryEngine,
    context: BruteForceDatasetContext,
    *,
    subset_id: str,
    selected_rows: int,
    rank_limit: int | None,
    labels: Sequence[Mapping[str, Any]],
    key_to_pos: Mapping[tuple[str, int], int],
    top_percent: float | None = None,
) -> dict[str, Any]:
    n = len(labels)
    count_matrix = _empty_matrix(n)
    sum_y_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    min_y_matrix = _empty_float_matrix(n)
    if selected_rows > 0 and context.var_names:
        for left_var, right_var in combinations(context.var_names, 2):
            _fill_pair_stats(
                engine,
                left_var=left_var,
                right_var=right_var,
                rank_limit=rank_limit,
                key_to_pos=key_to_pos,
                count_matrix=count_matrix,
                sum_y_matrix=sum_y_matrix,
                min_y_matrix=min_y_matrix,
            )
    shares = [
        [
            (None if selected_rows <= 0 else float(count_matrix[i][j]) / float(selected_rows))
            for j in range(n)
        ]
        for i in range(n)
    ]
    mean_y_p = _mean_from_sum(count_matrix, sum_y_matrix)
    payload: dict[str, Any] = {
        "subset_id": subset_id,
        "selected_rows": int(selected_rows),
        "rank_limit": None if rank_limit is None else int(rank_limit),
        "count_matrix": count_matrix,
        "share_matrix": shares,
        "mean_y_p_matrix": mean_y_p,
        "min_y_p_matrix": min_y_matrix,
    }
    if top_percent is not None:
        payload["top_percent"] = float(top_percent)
    return payload


def run_variable_cooccurrence_analysis(
    context: BruteForceDatasetContext,
    engine: BruteForceQueryEngine,
    *,
    top_percents: Sequence[float] = DEFAULT_TOP_PERCENTS,
    top_designs_k: int = 10,
) -> dict[str, Any]:
    """
    Compute level×level co-occurrence matrices for feasible subsets.
    """
    labels = _variable_levels(context)
    key_to_pos = {
        _label_key(str(item["variable"]), int(item["index"])): pos
        for pos, item in enumerate(labels)
    }
    feasible_count = _materialize_feasible_ranked(engine, context)
    top_designs = _fetch_top_designs(
        engine, context, top_k=int(top_designs_k)
    )

    subsets: dict[str, Any] = {}
    subsets["all_feasible"] = _subset_payload(
        engine,
        context,
        subset_id="all_feasible",
        selected_rows=feasible_count,
        rank_limit=None,
        labels=labels,
        key_to_pos=key_to_pos,
    )
    for percent in top_percents:
        limit = _top_count(feasible_count, float(percent))
        # Normalize ids: 1.0 -> top_1_percent, 10.0 -> top_10_percent
        if float(percent) == int(percent):
            subset_id = f"top_{int(percent)}_percent"
        else:
            subset_id = f"top_{str(percent).replace('.', '_')}_percent"
        subsets[subset_id] = _subset_payload(
            engine,
            context,
            subset_id=subset_id,
            selected_rows=limit,
            rank_limit=limit,
            labels=labels,
            key_to_pos=key_to_pos,
            top_percent=float(percent),
        )

    report = {
        "section": "variable_cooccurrence",
        "schema_version": 3,
        "overall_status": "success" if feasible_count >= 0 else "failed",
        "ok": True,
        "metadata": {
            "definition": (
                "For each selected design, every unordered variable pair "
                "increments the level×level co-occurrence cell once "
                "(matrix stored symmetrically). mean_y_p_matrix stores the "
                "mean feasible y_p among designs contributing to each cell; "
                "min_y_p_matrix stores the best (minimum) feasible y_p."
            ),
            "ranking": (
                "Top-percent subsets use ascending y_p among feasible rows "
                "with finite y_p; ties broken by global_index."
            ),
            "top_percent_rule": "selected_rows = ceil(percent/100 * feasible_count)",
            "feasible_count": int(feasible_count),
            "variable_count": len(context.var_names),
            "level_count": len(labels),
            "experiment_id": context.manifest.experiment_id,
            "problem_id": context.problem_id,
            "top_designs_k": int(top_designs_k),
        },
        "labels": labels,
        "label_names": [str(item["label"]) for item in labels],
        "top_designs": top_designs,
        "subsets": subsets,
    }
    return normalize_report_value(
        merge_constraint_provenance(
            report,
            sealed=context.constraint_names,
            selected=analysis_constraint_names(context),
        )
    )


def render_cooccurrence_summary_markdown(report: Mapping[str, Any]) -> str:
    """Short markdown summary derived only from the results object."""
    meta = report.get("metadata") or {}
    lines = [
        "# Variable co-occurrence",
        "",
        f"- experiment_id: {meta.get('experiment_id')}",
        f"- problem_id: {meta.get('problem_id')}",
        f"- feasible_count: {meta.get('feasible_count')}",
        f"- level_count: {meta.get('level_count')}",
        "",
        "## Subsets",
    ]
    subsets = report.get("subsets") or {}
    for key, payload in subsets.items():
        lines.append(
            f"- `{key}`: selected_rows={payload.get('selected_rows')}"
        )
    lines.append("")
    top_designs = report.get("top_designs") or []
    if isinstance(top_designs, list) and top_designs:
        lines.append("## Top designs (ascending y_p)")
        for item in top_designs:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                f"- rank={item.get('rank')}  y_p={item.get('y_p')}  "
                f"global_index={item.get('global_index')}"
            )
        lines.append("")
    lines.append(
        "See `results.json` for full count/share/mean_y_p/min_y_p matrices."
    )
    lines.append("")
    return "\n".join(lines)


def _resolve_output_dir(
    experiment_dir: Path,
    output_dir: Path | None,
) -> Path:
    experiment_dir = Path(experiment_dir)
    if output_dir is None:
        from core.execution_core.output_layout import (
            catalogue_analysis_publication_dir,
            require_results_publication,
            resolve_results_root,
        )

        return require_results_publication(
            catalogue_analysis_publication_dir(
                resolve_results_root(experiment_dir),
                experiment_dir.name,
            )
            / COOCCURRENCE_DIRNAME
        )
    from core.execution_core.output_layout import require_results_publication

    out_dir = require_results_publication(Path(output_dir), error_type=ValueError)
    dataset_root = (experiment_dir / "dataset").resolve()
    resolved = out_dir.resolve()
    if resolved == dataset_root or dataset_root in resolved.parents:
        raise ValueError(f"output_dir must not be inside dataset/: {out_dir}")
    return out_dir


def write_variable_cooccurrence_artifacts(
    experiment_dir: Path,
    report: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Atomically publish co-occurrence JSON/Markdown under analysis/."""
    out_dir = _resolve_output_dir(experiment_dir, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    finalized = normalize_report_value(report)
    json_text = dumps_report_json(finalized)
    md_text = render_cooccurrence_summary_markdown(finalized)
    written: dict[str, Path] = {}
    json_path = out_dir / COOCCURRENCE_RESULTS_JSON
    write_text_atomic(json_path, json_text)
    written["json"] = json_path
    md_path = out_dir / COOCCURRENCE_SUMMARY_MD
    write_text_atomic(md_path, md_text)
    written["md"] = md_path
    marker = out_dir / COOCCURRENCE_SUCCESS_MARKER
    write_text_atomic(marker, "ok\n")
    written["success"] = marker
    return written


def run_and_write_variable_cooccurrence(
    experiment_dir: Path,
    *,
    options: DatasetReportOptions | None = None,
    output_dir: Path | None = None,
    top_percents: Sequence[float] = DEFAULT_TOP_PERCENTS,
    selected_constraint_names: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """
    Run co-occurrence analysis and publish artifacts.
    """
    experiment_dir = Path(experiment_dir)
    options = options or DatasetReportOptions()
    context = open_brute_force_dataset(experiment_dir)
    if selected_constraint_names is not None:
        context = attach_selected_constraint_names(
            context, selected_constraint_names
        )
    with open_query_engine(context, options=options) as engine:
        report = run_variable_cooccurrence_analysis(
            context,
            engine,
            top_percents=top_percents,
        )
    written = write_variable_cooccurrence_artifacts(
        experiment_dir,
        report,
        output_dir=output_dir,
    )
    return report, written
