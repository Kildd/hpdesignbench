"""
Assemble the full brute-force dataset analysis report.

Compute and write are separate: call :func:`run_dataset_report` then
:func:`write_dataset_reports`, or use :func:`run_and_write_dataset_report`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.ioh_core.workflows.brute_force.analysis.best_designs import (
    _near_optimum_bands,
    run_best_designs_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.constraints import (
    run_constraints_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.context import (
    BruteForceAnalysisError,
    open_brute_force_dataset,
)
from core.ioh_core.workflows.brute_force.analysis.coverage import (
    run_coverage_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.engine import (
    SCIENTIFIC_RTOL,
    SENTINEL_ATOL,
    open_query_engine,
)
from core.ioh_core.workflows.brute_force.analysis.feasibility import (
    run_feasibility_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.integrity import (
    run_integrity_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.keep_list import (
    analysis_constraint_names,
    attach_selected_constraint_names,
    constraint_provenance,
    merge_constraint_provenance,
)
from core.ioh_core.workflows.brute_force.analysis.models import DatasetReportOptions
from core.ioh_core.workflows.brute_force.analysis.report import (
    REPORT_JSON_NAME,
    analysis_output_dir,
    empty_best_designs_section,
    empty_dataset_report,
    empty_tool_metadata,
    write_dataset_reports,
)
from core.ioh_core.workflows.brute_force.analysis.serialization import (
    normalize_report_value,
)
from core.ioh_core.workflows.brute_force.validation import validate_dataset


def _skipped(reason: str) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason}


def _dataset_block(context) -> dict[str, Any]:
    selected = analysis_constraint_names(context)
    payload = {
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
        "opt_penalty_alpha": float(context.opt_penalty_alpha),
        "opt_penalty_beta": float(context.opt_penalty_beta),
        "opt_invalid_utilization": float(
            context.run_config.invalid_utilization
        ),
    }
    payload.update(constraint_provenance(context.constraint_names, selected))
    return payload


def run_dataset_report(
    experiment_dir: Path,
    *,
    options: DatasetReportOptions | None = None,
    selected_constraint_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Run the full dataset-report pipeline without writing artifacts.

    Parameters
    ----------
    experiment_dir : Path
        Prepared sealed brute-force experiment root.
    options : DatasetReportOptions or None, optional
        Tunable analysis options.

    Returns
    -------
    dict
        Finalized JSON-compatible dataset report.
    """
    options = options or DatasetReportOptions()
    experiment_dir = Path(experiment_dir)
    context = open_brute_force_dataset(experiment_dir)
    if selected_constraint_names is not None:
        context = attach_selected_constraint_names(
            context, selected_constraint_names
        )
    structural = validate_dataset(experiment_dir)

    report = empty_dataset_report(
        context=context,
        options=options,
        scientific_rtol=SCIENTIFIC_RTOL,
        sentinel_atol=SENTINEL_ATOL,
    )
    report["dataset"] = _dataset_block(context)
    report["tool_metadata"] = empty_tool_metadata(
        options=options,
        scientific_rtol=SCIENTIFIC_RTOL,
        sentinel_atol=SENTINEL_ATOL,
    )

    def _finalize(payload: dict[str, Any]) -> dict[str, Any]:
        return normalize_report_value(
            merge_constraint_provenance(
                payload,
                sealed=context.constraint_names,
                selected=analysis_constraint_names(context),
            )
        )

    if not structural.ok:
        integrity = run_integrity_analysis(
            context, engine=None, options=options, structural=structural
        )
        report["integrity"] = integrity
        report["enumeration_coverage"] = _skipped(
            "structural_validation_failed"
        )
        report["best_designs"] = {
            **empty_best_designs_section(),
            "status": "skipped",
            "reason": "structural_validation_failed",
        }
        report["feasibility"] = _skipped("structural_validation_failed")
        report["constraints"] = _skipped("structural_validation_failed")
        report["overall_status"] = "structural_failure"
        report["ok"] = False
        return _finalize(report)

    with open_query_engine(context, options=options) as engine:
        integrity = run_integrity_analysis(
            context, engine, options=options, structural=structural
        )
        report["integrity"] = integrity
        coverage = run_coverage_analysis(context, engine, options=options)
        best = run_best_designs_analysis(context, engine, options=options)
        feasibility = run_feasibility_analysis(
            context, engine, options=options
        )
        constraints = run_constraints_analysis(
            context, engine, options=options
        )

    report["enumeration_coverage"] = coverage
    report["best_designs"] = best
    report["feasibility"] = feasibility
    report["constraints"] = constraints

    if not integrity.get("scientific_ok", False):
        report["overall_status"] = "scientific_integrity_failure"
        report["ok"] = False
        return _finalize(report)

    section_ok = all(
        bool(section.get("ok", True))
        for section in (coverage, best, feasibility, constraints)
    )
    if section_ok:
        report["overall_status"] = "success"
        report["ok"] = True
    else:
        report["overall_status"] = "scientific_integrity_failure"
        report["ok"] = False
    return _finalize(report)


def run_and_write_dataset_report(
    experiment_dir: Path,
    *,
    options: DatasetReportOptions | None = None,
    write: bool = True,
    output_dir: Path | None = None,
    formats: Sequence[str] = ("json", "md"),
    selected_constraint_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Run the dataset report and optionally publish JSON/Markdown artifacts.

    On structural failure, still writes the bounded integrity-only full report
    document when ``write`` is true (other sections marked skipped).
    """
    report = run_dataset_report(
        experiment_dir,
        options=options,
        selected_constraint_names=selected_constraint_names,
    )
    if write:
        write_dataset_reports(
            experiment_dir,
            report,
            output_dir=output_dir,
            formats=formats,
        )
    return report


def _load_existing_report(
    experiment_dir: Path,
    *,
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Load ``report.json`` from the analysis output directory."""
    experiment_dir = Path(experiment_dir)
    out_dir = (
        analysis_output_dir(experiment_dir)
        if output_dir is None
        else Path(output_dir)
    )
    path = out_dir / REPORT_JSON_NAME
    if not path.is_file():
        raise BruteForceAnalysisError(
            f"report.json not found for refresh: {path}. "
            "Run a full `hdb-bf-analyze report` first."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise BruteForceAnalysisError(
            f"report.json must be a JSON object: {path}"
        )
    return payload, path


def _y_p_star_from_report(report: Mapping[str, Any]) -> float | None:
    best = report.get("best_designs")
    if not isinstance(best, Mapping):
        return None
    finite = best.get("best_finite")
    if not isinstance(finite, Mapping):
        return None
    if finite.get("status") != "found":
        return None
    y_p = finite.get("y_p")
    if y_p is None:
        return None
    try:
        value = float(y_p)
    except (TypeError, ValueError):
        return None
    return value if value == value else None  # NaN check


def _patch_best_designs_ok(best: dict[str, Any], near: Mapping[str, Any]) -> None:
    """Recompute ``best_designs.ok`` after replacing near-optimum bands."""
    b8 = best.get("b8_cross_check") or {}
    ok = (
        b8.get("status") in {"ok", "unavailable"}
        and near.get("status") in {"computed", "unavailable"}
        and (near.get("monotone_ok") in {True, None})
        and (near.get("winner_in_all_bands") in {True, None})
    )
    if near.get("status") == "computed":
        if not near.get("monotone_ok") or not near.get("winner_in_all_bands"):
            ok = False
    if b8.get("status") == "disagree":
        ok = False
    best["ok"] = bool(ok)


def refresh_near_optimum_bands_report(
    experiment_dir: Path,
    *,
    options: DatasetReportOptions | None = None,
    output_dir: Path | None = None,
    formats: Sequence[str] = ("json", "md"),
) -> tuple[dict[str, Any], dict[str, Path]]:
    """
    Recompute only ``best_designs.near_optimum_bands`` on an existing report.

    Keeps integrity / coverage / feasibility / constraints sections intact.
    Requires a previously written ``report.json`` (full report first).
    """
    options = options or DatasetReportOptions()
    experiment_dir = Path(experiment_dir)
    report, _path = _load_existing_report(
        experiment_dir, output_dir=output_dir
    )
    best = report.get("best_designs")
    if not isinstance(best, dict):
        raise BruteForceAnalysisError(
            "existing report.json missing best_designs; run a full report first"
        )

    y_p_star = _y_p_star_from_report(report)
    context = open_brute_force_dataset(experiment_dir)
    with open_query_engine(context, options=options) as engine:
        near = _near_optimum_bands(engine, y_p_star=y_p_star)

    best["near_optimum_bands"] = near
    _patch_best_designs_ok(best, near)
    report["best_designs"] = best
    report = normalize_report_value(report)
    written = write_dataset_reports(
        experiment_dir,
        report,
        output_dir=output_dir,
        formats=formats,
    )
    return report, written
