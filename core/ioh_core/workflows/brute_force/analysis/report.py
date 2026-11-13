"""
Dataset report schema, Markdown summary, and atomic publish helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from core.execution_core.atomic_io import write_text_atomic
from core.ioh_core.workflows.brute_force.analysis.engine import (
    SCIENTIFIC_RTOL,
    SENTINEL_ATOL,
)
from core.ioh_core.workflows.brute_force.analysis.models import (
    BruteForceDatasetContext,
    DatasetReportOptions,
)
from core.ioh_core.workflows.brute_force.analysis.serialization import (
    dumps_report_json,
    normalize_report_value,
)

REPORT_SCHEMA_VERSION = 1
REPORT_SECTION_KEYS: tuple[str, ...] = (
    "dataset",
    "integrity",
    "enumeration_coverage",
    "best_designs",
    "feasibility",
    "constraints",
    "tool_metadata",
)
NEAR_OPTIMUM_TOLERANCES_PERCENT: tuple[float, ...] = (
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    *tuple(float(t) for t in range(10, 101, 5)),
)

ANALYSIS_DIRNAME = "analysis"
REPORT_JSON_NAME = "report.json"
REPORT_MARKDOWN_NAME = "summary.md"
INTEGRITY_JSON_NAME = "integrity_report.json"
INTEGRITY_MARKDOWN_NAME = "integrity_summary.md"


def empty_near_optimum_bands() -> dict[str, Any]:
    """
    Reserved near-optimum band block (computed in C3.5).
    """
    return {
        "status": "not_computed",
        "tolerances_percent": list(NEAR_OPTIMUM_TOLERANCES_PERCENT),
        "bands": [],
    }


def empty_best_designs_section() -> dict[str, Any]:
    """
    Reserved best-designs section including near-optimum scaffolding.
    """
    return {
        "best_finite": None,
        "best_feasible": None,
        "worst_feasible": None,
        "near_optimum_bands": empty_near_optimum_bands(),
    }


def empty_tool_metadata(
    *,
    options: DatasetReportOptions | None = None,
    scientific_rtol: float = SCIENTIFIC_RTOL,
    sentinel_atol: float = SENTINEL_ATOL,
) -> dict[str, Any]:
    """
    Build the stable tool-metadata shell for dataset analysis reports.
    """
    options = options or DatasetReportOptions()
    return {
        "duckdb_threads": int(options.duckdb_threads),
        "duckdb_memory_limit": options.duckdb_memory_limit,
        "duckdb_temp_directory": (
            None
            if options.duckdb_temp_directory is None
            else str(options.duckdb_temp_directory)
        ),
        "max_violation_examples": int(options.max_violation_examples),
        "include_pairwise_level_coverage": bool(
            options.include_pairwise_level_coverage
        ),
        "include_coviolation_matrix": bool(options.include_coviolation_matrix),
        "max_violation_combinations": int(options.max_violation_combinations),
        "near_boundary_low": float(options.near_boundary_low),
        "near_boundary_high": float(options.near_boundary_high),
        "scientific_rtol": float(scientific_rtol),
        "sentinel_atol": float(sentinel_atol),
    }


def empty_dataset_report(
    *,
    context: BruteForceDatasetContext | None = None,
    options: DatasetReportOptions | None = None,
    scientific_rtol: float = SCIENTIFIC_RTOL,
    sentinel_atol: float = SENTINEL_ATOL,
) -> dict[str, Any]:
    """
    Build a schema-version-1 dataset analysis report with empty sections.

    When ``context`` is provided, fill a minimal deterministic ``dataset``
    identity block from sealed experiment metadata (no absolute paths).
    """
    options = options or DatasetReportOptions()
    dataset: dict[str, Any] = {}
    if context is not None:
        dataset = {
            "experiment_id": context.manifest.experiment_id,
            "workflow_type": context.manifest.workflow_type,
            "problem_id": context.problem_id,
            "slab_type": context.slab_type,
            "full_catalogue_rows": int(context.full_catalogue_rows),
            "planned_start_index": int(context.planned_start_index),
            "planned_stop_index": int(context.planned_stop_index),
            "planned_rows": int(context.planned_rows),
            "is_full_enumeration": bool(context.is_full_enumeration),
            "coverage_fraction": float(context.coverage_fraction),
            "var_names": list(context.var_names),
            "param_names": list(context.param_names),
            "constraint_names": list(context.constraint_names),
            "constraint_names_sealed": list(context.constraint_names),
            "constraint_names_selected": list(
                context.selected_constraint_names
                if context.selected_constraint_names is not None
                else context.constraint_names
            ),
            "constraint_set_is_subset": (
                tuple(
                    context.selected_constraint_names
                    if context.selected_constraint_names is not None
                    else context.constraint_names
                )
                != tuple(context.constraint_names)
            ),
            "opt_penalty_alpha": float(context.opt_penalty_alpha),
            "opt_penalty_beta": float(context.opt_penalty_beta),
            "opt_invalid_utilization": float(
                context.run_config.invalid_utilization
            ),
        }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "overall_status": None,
        "ok": None,
        "dataset": dataset,
        "integrity": {},
        "enumeration_coverage": {},
        "best_designs": empty_best_designs_section(),
        "feasibility": {},
        "constraints": {},
        "tool_metadata": empty_tool_metadata(
            options=options,
            scientific_rtol=scientific_rtol,
            sentinel_atol=sentinel_atol,
        ),
    }
    return normalize_report_value(report)


def analysis_output_dir(experiment_dir: Path) -> Path:
    """
    Return ``aggregate/analysis`` under the experiment root.
    """
    return Path(experiment_dir) / "aggregate" / ANALYSIS_DIRNAME


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    return str(value)


def render_dataset_report_markdown(report: Mapping[str, Any]) -> str:
    """
    Render Markdown only from a finalized JSON-compatible report object.
    """
    report = normalize_report_value(report)
    dataset = report.get("dataset") or {}
    integrity = report.get("integrity") or {}
    coverage = report.get("enumeration_coverage") or {}
    best = report.get("best_designs") or {}
    near = best.get("near_optimum_bands") or {}
    feasibility = report.get("feasibility") or {}
    constraints = report.get("constraints") or {}
    tool = report.get("tool_metadata") or {}

    lines: list[str] = [
        "# Brute-force dataset analysis",
        "",
        "## Status",
        f"- overall_status: {_fmt(report.get('overall_status'))}",
        f"- ok: {_fmt(report.get('ok'))}",
        f"- schema_version: {_fmt(report.get('schema_version'))}",
        f"- experiment_id: {_fmt(dataset.get('experiment_id'))}",
        f"- problem_id: {_fmt(dataset.get('problem_id'))}",
        f"- planned_rows: {_fmt(dataset.get('planned_rows'))}",
        f"- full_catalogue_rows: {_fmt(dataset.get('full_catalogue_rows'))}",
        f"- is_full_enumeration: {_fmt(dataset.get('is_full_enumeration'))}",
        (
            f"- tolerances: scientific_rtol={_fmt(tool.get('scientific_rtol'))}, "
            f"sentinel_atol={_fmt(tool.get('sentinel_atol'))}"
        ),
        "",
        "## Integrity",
        f"- structural.ok: {_fmt((integrity.get('structural') or {}).get('ok'))}",
        f"- scientific_ok: {_fmt(integrity.get('scientific_ok'))}",
        f"- scientific_ran: {_fmt(integrity.get('scientific_ran'))}",
        "",
        "## Enumeration coverage",
    ]

    if coverage.get("status") == "skipped":
        lines.append(f"- skipped: {_fmt(coverage.get('reason'))}")
    else:
        lines.append(f"- ok: {_fmt(coverage.get('ok'))}")
        lines.append(f"- observed_rows: {_fmt(coverage.get('observed_rows'))}")
        for variable in coverage.get("variables") or []:
            lines.append(
                f"- variable `{variable.get('name')}`: "
                f"distinct={_fmt(variable.get('observed_distinct_count'))}/"
                f"{_fmt(variable.get('declared_level_count'))}, "
                f"ok={_fmt(variable.get('ok'))}"
            )

    lines.extend(["", "## Best designs"])
    if best.get("status") == "skipped":
        lines.append(f"- skipped: {_fmt(best.get('reason'))}")
    else:
        finite = best.get("best_finite") or {}
        feasible = best.get("best_feasible") or {}
        worst_feasible = best.get("worst_feasible") or {}
        lines.append(
            f"- best_finite: status={_fmt(finite.get('status'))}, "
            f"global_index={_fmt(finite.get('global_index'))}, "
            f"y_p={_fmt(finite.get('y_p'))}"
        )
        lines.append(
            f"- best_feasible: status={_fmt(feasible.get('status'))}, "
            f"global_index={_fmt(feasible.get('global_index'))}, "
            f"y_p={_fmt(feasible.get('y_p'))}"
        )
        lines.append(
            f"- worst_feasible: status={_fmt(worst_feasible.get('status'))}, "
            f"global_index={_fmt(worst_feasible.get('global_index'))}, "
            f"y_p={_fmt(worst_feasible.get('y_p'))}"
        )
        lines.append(f"- near_optimum.status: {_fmt(near.get('status'))}")
        for band in near.get("bands") or []:
            lines.append(
                f"- near {band.get('tolerance_percent')}%: "
                f"threshold={_fmt(band.get('threshold'))}, "
                f"count_within={_fmt(band.get('count_within'))}, "
                f"share_all={_fmt(band.get('share_of_all_solutions'))}, "
                f"count_feasible_within={_fmt(band.get('count_feasible_within'))}, "
                f"share_feasible={_fmt(band.get('share_of_feasible_solutions'))}"
            )

    lines.extend(["", "## Feasibility"])
    if feasibility.get("status") == "skipped":
        lines.append(f"- skipped: {_fmt(feasibility.get('reason'))}")
    else:
        totals = (feasibility.get("totals") or {}).get("dataset_shares") or {}
        lines.append(f"- ok: {_fmt(feasibility.get('ok'))}")
        lines.append(
            f"- feasible count/share: "
            f"{_fmt((totals.get('feasible') or {}).get('count'))} / "
            f"{_fmt((totals.get('feasible') or {}).get('share_of_observed'))}"
        )
        for variable in feasibility.get("variables") or []:
            lines.append(
                f"- variable `{variable.get('name')}`: "
                f"{len(variable.get('levels') or [])} levels"
            )

    lines.extend(["", "## Constraints"])
    if constraints.get("status") == "skipped":
        lines.append(f"- skipped: {_fmt(constraints.get('reason'))}")
    else:
        lines.append(f"- ok: {_fmt(constraints.get('ok'))}")
        activity = (constraints.get("activity") or {}).get("per_constraint") or []
        for item in activity:
            lines.append(
                f"- `{item.get('name')}` active_count="
                f"{_fmt(item.get('active_count'))}, "
                f"sentinel={_fmt(item.get('invalid_sentinel_count'))}"
            )
        shares = (constraints.get("utilization_shares") or {}).get(
            "per_constraint"
        ) or []
        if shares:
            lines.append("- utilization shares (of all observed rows):")
            for item in shares:
                lines.append(
                    f"  - `{item.get('name')}`: "
                    f"active={_fmt(item.get('active_share_of_all_observed_rows'))}, "
                    f"max_violation="
                    f"{_fmt(item.get('maximum_violation_share_of_all_observed_rows'))}, "
                    f"invalid="
                    f"{_fmt(item.get('invalid_utilization_share_of_all_observed_rows'))}"
                )

    lines.extend(
        [
            "",
            "See `report.json` for the full structured document.",
            "",
        ]
    )
    return "\n".join(lines)


def write_dataset_reports(
    experiment_dir: Path,
    report: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
    formats: Sequence[str] = ("json", "md"),
) -> dict[str, Path]:
    """
    Fully serialize then atomically publish selected report artifacts.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root (used when ``output_dir`` is omitted).
    report : Mapping
        Finalized dataset report object.
    output_dir : Path or None, optional
        Alternate output directory. Must not lie under ``dataset/``.
    formats : sequence of str, optional
        Subset of ``{\"json\", \"md\"}``.

    Returns
    -------
    dict
        Mapping of written format name to path.
    """
    experiment_dir = Path(experiment_dir)
    selected = {str(item).lower() for item in formats}
    unknown = selected - {"json", "md"}
    if unknown:
        raise ValueError(f"unsupported report formats: {sorted(unknown)}")
    if not selected:
        raise ValueError("formats must include at least one of: json, md")

    if output_dir is None:
        out_dir = analysis_output_dir(experiment_dir)
    else:
        out_dir = Path(output_dir)
        dataset_root = (experiment_dir / "dataset").resolve()
        resolved = out_dir.resolve()
        if resolved == dataset_root or dataset_root in resolved.parents:
            raise ValueError(
                f"output_dir must not be inside dataset/: {out_dir}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    finalized = normalize_report_value(report)
    # Fully serialize before any publish so a render failure cannot
    # partially replace a previous good report.
    payloads: dict[str, str] = {}
    if "json" in selected:
        payloads["json"] = dumps_report_json(finalized)
    if "md" in selected:
        payloads["md"] = render_dataset_report_markdown(finalized)

    written: dict[str, Path] = {}
    if "json" in payloads:
        json_path = out_dir / REPORT_JSON_NAME
        write_text_atomic(json_path, payloads["json"])
        written["json"] = json_path
    if "md" in payloads:
        markdown_path = out_dir / REPORT_MARKDOWN_NAME
        write_text_atomic(markdown_path, payloads["md"])
        written["md"] = markdown_path
    return written


def write_integrity_reports(
    experiment_dir: Path,
    integrity: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
    formats: Sequence[str] = ("json", "md"),
) -> dict[str, Path]:
    """
    Publish integrity-only artifacts without touching ``report.json``.
    """
    experiment_dir = Path(experiment_dir)
    selected = {str(item).lower() for item in formats}
    unknown = selected - {"json", "md"}
    if unknown:
        raise ValueError(f"unsupported report formats: {sorted(unknown)}")
    if not selected:
        raise ValueError("formats must include at least one of: json, md")

    if output_dir is None:
        out_dir = analysis_output_dir(experiment_dir)
    else:
        out_dir = Path(output_dir)
        dataset_root = (experiment_dir / "dataset").resolve()
        resolved = out_dir.resolve()
        if resolved == dataset_root or dataset_root in resolved.parents:
            raise ValueError(
                f"output_dir must not be inside dataset/: {out_dir}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = normalize_report_value(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": "integrity",
            "integrity": integrity,
            "ok": bool(integrity.get("ok")),
            "overall_status": (
                "structural_failure"
                if not (integrity.get("structural") or {}).get("ok", False)
                else (
                    "success"
                    if integrity.get("ok")
                    else "scientific_integrity_failure"
                )
            ),
        }
    )
    payloads: dict[str, str] = {}
    if "json" in selected:
        payloads["json"] = dumps_report_json(payload)
    if "md" in selected:
        structural = integrity.get("structural") or {}
        lines = [
            "# Brute-force integrity diagnostic",
            "",
            f"- overall_status: {payload.get('overall_status')}",
            f"- ok: {payload.get('ok')}",
            f"- structural.ok: {structural.get('ok')}",
            f"- scientific_ok: {integrity.get('scientific_ok')}",
            f"- scientific_ran: {integrity.get('scientific_ran')}",
            "",
            "See `integrity_report.json` for the full structured document.",
            "",
        ]
        payloads["md"] = "\n".join(lines)

    written: dict[str, Path] = {}
    if "json" in payloads:
        path = out_dir / INTEGRITY_JSON_NAME
        write_text_atomic(path, payloads["json"])
        written["json"] = path
    if "md" in payloads:
        path = out_dir / INTEGRITY_MARKDOWN_NAME
        write_text_atomic(path, payloads["md"])
        written["md"] = path
    return written
