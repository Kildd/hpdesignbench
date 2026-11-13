"""
CLI for sealed brute-force dataset analysis.

Console script: ``hdb-bf-analyze``
Module entry: ``python -m core.ioh_core.workflows.brute_force.analysis``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.ioh_core.workflows.brute_force.analysis.context import (
    BruteForceAnalysisError,
    open_brute_force_dataset,
)
from core.ioh_core.workflows.brute_force.analysis.dataset_report import (
    refresh_near_optimum_bands_report,
    run_dataset_report,
)
from core.ioh_core.workflows.brute_force.analysis.engine import (
    BruteForceEngineError,
    open_query_engine,
)
from core.ioh_core.workflows.brute_force.analysis.factorial_sensitivity import (
    factorial_sensitivity_output_dir,
    run_and_write_factorial_sensitivity,
)
from core.ioh_core.workflows.brute_force.analysis.integrity import (
    run_integrity_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.models import DatasetReportOptions
from core.ioh_core.workflows.brute_force.analysis.report import (
    analysis_output_dir,
    write_dataset_reports,
    write_integrity_reports,
)
from core.ioh_core.workflows.brute_force.analysis.variable_cooccurrence import (
    run_and_write_variable_cooccurrence,
    _resolve_output_dir as _resolve_cooccurrence_output_dir,
)
from core.ioh_core.workflows.brute_force.analysis.constraint_decision_tree import (
    DecisionTreeOptions,
    run_and_write_constraint_decision_tree,
    _resolve_output_dir as _resolve_decision_tree_output_dir,
)
from core.ioh_core.workflows.brute_force.deps import BruteForceDependencyError
from core.ioh_core.workflows.brute_force.validation import validate_dataset
from core.ioh_core.workflows.brute_force.analysis.keep_list import (
    assert_subset_analysis_output_dir,
    load_sealed_constraint_names,
    resolve_analysis_constraint_names,
)


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that exits with code 1 on usage errors."""

    def error(self, message: str) -> None:  # noqa: D102
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    """
    Build the ``hdb-bf-analyze`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Root parser with ``report``, ``integrity``, and ``sensitivity``
        subcommands.
    """
    parser = _ArgumentParser(
        prog="hdb-bf-analyze",
        description=(
            "Read-only analysis of sealed brute-force datasets. "
            "Does not expose arbitrary SQL."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser(
        "report",
        help="Run the full dataset analysis report and publish artifacts",
    )
    _add_common_flags(report)
    report.add_argument(
        "--refresh-near-optimum-bands",
        action="store_true",
        help=(
            "Reuse an existing report.json and recompute only "
            "best_designs.near_optimum_bands (keeps constraints / "
            "feasibility / coverage). Requires a prior full report."
        ),
    )

    integrity = sub.add_parser(
        "integrity",
        help=(
            "Run the fast integrity diagnostic without overwriting "
            "report.json / summary.md"
        ),
    )
    _add_common_flags(integrity)

    sensitivity = sub.add_parser(
        "sensitivity",
        help=(
            "Exact full-factorial Sobol–Hoeffding sensitivity for a complete "
            "enumeration (y_p, util__*, binary feasibility)"
        ),
    )
    _add_sensitivity_flags(sensitivity)

    cooccurrence = sub.add_parser(
        "cooccurrence",
        help=(
            "Variable-level co-occurrence among all feasible designs and "
            "top-ranked feasible subsets (1/2/5/10%)"
        ),
    )
    cooccurrence.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Prepared sealed brute-force experiment root",
    )
    cooccurrence.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Alternate output directory (default: "
            "<experiment-dir>/aggregate/analysis/variable_cooccurrence). "
            "Must not be under dataset/"
        ),
    )
    cooccurrence.add_argument(
        "--duckdb-threads",
        type=int,
        default=1,
        help="DuckDB threads (default: 1)",
    )
    cooccurrence.add_argument(
        "--duckdb-memory-limit",
        default=None,
        help="Optional DuckDB memory limit, e.g. 4GB",
    )
    cooccurrence.add_argument(
        "--duckdb-temp-directory",
        type=Path,
        default=None,
        help="Optional DuckDB spill directory",
    )
    _add_constraint_names_flag(cooccurrence)

    decision_tree = sub.add_parser(
        "decision-tree",
        help=(
            "Infeasible repair decision tree / pattern→move rules "
            "(Z excluded; feasible volume-seeking deferred)"
        ),
    )
    decision_tree.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Prepared sealed brute-force experiment root",
    )
    decision_tree.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Alternate output directory (default: "
            "<experiment-dir>/aggregate/analysis/constraint_decision_tree). "
            "Must not be under dataset/"
        ),
    )
    decision_tree.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="CART max depth (default: 3; useful range 2–4)",
    )
    decision_tree.add_argument(
        "--min-support",
        type=int,
        default=50,
        help="Minimum support for retained rules / pattern-move rows",
    )
    decision_tree.add_argument(
        "--min-lift",
        type=float,
        default=1.2,
        help="Minimum lift vs action prior for retained rules",
    )
    decision_tree.add_argument(
        "--sample-rows",
        type=int,
        default=100000,
        help="Max infeasible seeds for neighbour scoring (default: 100000)",
    )
    decision_tree.add_argument(
        "--min-samples-leaf",
        type=int,
        default=25,
        help="CART minimum leaf size (default: 25)",
    )
    decision_tree.add_argument(
        "--duckdb-threads",
        type=int,
        default=1,
        help="DuckDB threads (default: 1)",
    )
    decision_tree.add_argument(
        "--duckdb-memory-limit",
        default=None,
        help="Optional DuckDB memory limit, e.g. 4GB",
    )
    decision_tree.add_argument(
        "--duckdb-temp-directory",
        type=Path,
        default=None,
        help="Optional DuckDB spill directory",
    )
    _add_constraint_names_flag(decision_tree)
    return parser


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Prepared sealed brute-force experiment root",
    )
    parser.add_argument(
        "--format",
        choices=("json", "md", "both"),
        default="both",
        dest="format_name",
        help="Artifact format(s) to publish (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Alternate output directory (default: "
            "<experiment-dir>/aggregate/analysis). Must not be under dataset/"
        ),
    )
    parser.add_argument(
        "--max-violation-examples",
        type=int,
        default=20,
        help="Maximum example global_index values per integrity rule",
    )
    parser.add_argument(
        "--include-pairwise-level-coverage",
        action="store_true",
        help="Emit optional pairwise variable-level coverage tables",
    )
    parser.add_argument(
        "--max-violation-combinations",
        type=int,
        default=50,
        help="Maximum violated-constraint combinations to retain",
    )
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=1,
        help="DuckDB thread count (default 1 for deterministic aggregates)",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        type=str,
        default=None,
        help='Optional DuckDB memory limit string such as "4GB"',
    )
    parser.add_argument(
        "--duckdb-temp-directory",
        type=Path,
        default=None,
        help="Optional DuckDB spill directory",
    )
    _add_constraint_names_flag(parser)


def _add_constraint_names_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--constraint-names",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Keep list of sealed constraint names (same relative order). "
            "Omitting names ignores them. A proper subset warns and "
            "requires a non-canonical --output-dir. Integrity rejects this flag."
        ),
    )


def _add_sensitivity_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Prepared sealed brute-force experiment root",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv", "md", "all"),
        default="all",
        dest="format_name",
        help="Artifact format(s) to publish (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Alternate output directory (default: "
            "<experiment-dir>/aggregate/analysis/factorial_sensitivity). "
            "Must not be under dataset/"
        ),
    )
    parser.add_argument(
        "--max-violation-examples",
        type=int,
        default=20,
        help="Maximum example global_index values retained in diagnostics",
    )
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=1,
        help="DuckDB thread count (default 1 for deterministic aggregates)",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        type=str,
        default=None,
        help='Optional DuckDB memory limit string such as "4GB"',
    )
    parser.add_argument(
        "--duckdb-temp-directory",
        type=Path,
        default=None,
        help="Optional DuckDB spill directory",
    )
    _add_constraint_names_flag(parser)


def _formats_from_arg(format_name: str) -> tuple[str, ...]:
    name = str(format_name).lower()
    if name == "both":
        return ("json", "md")
    if name in {"json", "md"}:
        return (name,)
    raise ValueError(f"unsupported format {format_name!r}")


def _sensitivity_formats_from_arg(format_name: str) -> tuple[str, ...]:
    name = str(format_name).lower()
    if name == "all":
        return ("json", "csv", "md")
    if name in {"json", "csv", "md"}:
        return (name,)
    raise ValueError(f"unsupported sensitivity format {format_name!r}")


def _options_from_args(args: argparse.Namespace) -> DatasetReportOptions:
    return DatasetReportOptions(
        max_violation_examples=int(args.max_violation_examples),
        include_pairwise_level_coverage=bool(
            getattr(args, "include_pairwise_level_coverage", False)
        ),
        max_violation_combinations=int(
            getattr(args, "max_violation_combinations", 50)
        ),
        duckdb_threads=int(args.duckdb_threads),
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_temp_directory=args.duckdb_temp_directory,
    )


def _paths_payload(written: Mapping[str, Path]) -> dict[str, str]:
    return {key: str(path) for key, path in sorted(written.items())}


def _emit_json(payload: Mapping[str, Any], *, stream=None) -> None:
    target = sys.stdout if stream is None else stream
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), file=target)


def _exit_for_status(overall_status: str, *, ok: bool) -> int:
    if ok and overall_status == "success":
        return 0
    if overall_status in {
        "structural_failure",
        "scientific_integrity_failure",
        "eligibility_failure",
        "scientific_failure",
    }:
        return 2
    if not ok:
        return 2
    return 0


def _resolve_output_dir(
    experiment_dir: Path, output_dir: Path | None
) -> Path:
    if output_dir is None:
        return analysis_output_dir(experiment_dir)
    return Path(output_dir)


def _prepare_keep_list(
    args: argparse.Namespace,
    *,
    experiment_dir: Path,
    output_dir: Path,
    implicit_z_drop: bool = False,
) -> tuple[str, ...] | None:
    requested = getattr(args, "constraint_names", None)
    sealed = load_sealed_constraint_names(experiment_dir)
    if requested is not None and sealed is None:
        raise BruteForceAnalysisError(
            "constraint keep-list requires dataset/schema.json "
            f"under {experiment_dir}"
        )
    if sealed is None:
        return None
    selected = resolve_analysis_constraint_names(
        sealed,
        requested,
        implicit_z_drop=implicit_z_drop,
    )
    explicit_subset = requested is not None and tuple(selected) != tuple(sealed)
    if explicit_subset:
        assert_subset_analysis_output_dir(
            experiment_dir,
            output_dir,
            output_dir_provided=getattr(args, "output_dir", None) is not None,
        )
    return selected


def _cmd_report(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.experiment_dir)
    options = _options_from_args(args)
    formats = _formats_from_arg(args.format_name)
    output_dir = _resolve_output_dir(experiment_dir, args.output_dir)
    selected = _prepare_keep_list(
        args, experiment_dir=experiment_dir, output_dir=output_dir
    )
    refresh_bands = bool(getattr(args, "refresh_near_optimum_bands", False))
    mode = (
        "refreshing near-optimum bands in existing report"
        if refresh_bands
        else "starting Layer 1 dataset analysis"
    )
    print(
        f"hdb-bf-analyze report: {mode} for "
        f"{experiment_dir} "
        f"(output: {output_dir})",
        file=sys.stderr,
        flush=True,
    )

    if refresh_bands:
        report, written = refresh_near_optimum_bands_report(
            experiment_dir,
            options=options,
            output_dir=output_dir,
            formats=formats,
        )
    else:
        report = run_dataset_report(
            experiment_dir,
            options=options,
            selected_constraint_names=selected,
        )
        written = write_dataset_reports(
            experiment_dir,
            report,
            output_dir=output_dir,
            formats=formats,
        )
    overall = str(report.get("overall_status", "unknown"))
    ok = bool(report.get("ok"))
    near = ((report.get("best_designs") or {}).get("near_optimum_bands") or {})
    _emit_json(
        {
            "command": "report",
            "experiment_dir": str(experiment_dir),
            "ok": ok,
            "output_dir": str(output_dir),
            "overall_status": overall,
            "refresh_near_optimum_bands": refresh_bands,
            "near_optimum_status": near.get("status"),
            "near_optimum_band_count": len(near.get("bands") or []),
            "paths": _paths_payload(written),
        }
    )
    return _exit_for_status(overall, ok=ok)


def _run_integrity_only(
    experiment_dir: Path,
    *,
    options: DatasetReportOptions,
) -> dict[str, Any]:
    context = open_brute_force_dataset(experiment_dir)
    structural = validate_dataset(experiment_dir)
    if not structural.ok:
        return run_integrity_analysis(
            context, engine=None, options=options, structural=structural
        )
    with open_query_engine(context, options=options) as engine:
        return run_integrity_analysis(
            context, engine, options=options, structural=structural
        )


def _cmd_integrity(args: argparse.Namespace) -> int:
    if getattr(args, "constraint_names", None) is not None:
        raise BruteForceAnalysisError(
            "integrity always uses the full sealed constraint schema; "
            "do not pass --constraint-names"
        )
    experiment_dir = Path(args.experiment_dir)
    options = _options_from_args(args)
    formats = _formats_from_arg(args.format_name)
    output_dir = _resolve_output_dir(experiment_dir, args.output_dir)
    print(
        "hdb-bf-analyze integrity: starting integrity diagnostic for "
        f"{experiment_dir} "
        f"(output: {output_dir})",
        file=sys.stderr,
        flush=True,
    )

    integrity = _run_integrity_only(experiment_dir, options=options)
    written = write_integrity_reports(
        experiment_dir,
        integrity,
        output_dir=output_dir,
        formats=formats,
    )
    structural_ok = bool((integrity.get("structural") or {}).get("ok", False))
    if not structural_ok:
        overall = "structural_failure"
    elif bool(integrity.get("ok")):
        overall = "success"
    else:
        overall = "scientific_integrity_failure"
    ok = bool(integrity.get("ok"))
    _emit_json(
        {
            "command": "integrity",
            "experiment_dir": str(experiment_dir),
            "ok": ok,
            "output_dir": str(output_dir),
            "overall_status": overall,
            "paths": _paths_payload(written),
        }
    )
    return _exit_for_status(overall, ok=ok)


def _cmd_sensitivity(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.experiment_dir)
    options = _options_from_args(args)
    formats = _sensitivity_formats_from_arg(args.format_name)
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    resolved_out = (
        output_dir
        if output_dir is not None
        else factorial_sensitivity_output_dir(experiment_dir)
    )
    selected = _prepare_keep_list(
        args, experiment_dir=experiment_dir, output_dir=resolved_out
    )
    print(
        "hdb-bf-analyze sensitivity: starting exact factorial analysis for "
        f"{experiment_dir} "
        f"(output: {resolved_out})",
        file=sys.stderr,
        flush=True,
    )

    document, written, success = run_and_write_factorial_sensitivity(
        experiment_dir,
        options=options,
        output_dir=output_dir,
        formats=formats,
        selected_constraint_names=selected,
    )
    overall = str(document.get("overall_status", "unknown"))
    _emit_json(
        {
            "command": "sensitivity",
            "experiment_dir": str(experiment_dir),
            "ok": bool(success),
            "output_dir": str(resolved_out),
            "overall_status": overall,
            "paths": _paths_payload(written),
        }
    )
    return _exit_for_status(overall, ok=success)


def _cmd_cooccurrence(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.experiment_dir)
    options = DatasetReportOptions(
        duckdb_threads=int(args.duckdb_threads),
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_temp_directory=(
            None
            if args.duckdb_temp_directory is None
            else Path(args.duckdb_temp_directory)
        ),
    )
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    resolved_out = _resolve_cooccurrence_output_dir(experiment_dir, output_dir)
    selected = _prepare_keep_list(
        args, experiment_dir=experiment_dir, output_dir=resolved_out
    )
    print(
        "hdb-bf-analyze cooccurrence: starting variable co-occurrence for "
        f"{experiment_dir} "
        f"(output: {resolved_out})",
        file=sys.stderr,
        flush=True,
    )
    document, written = run_and_write_variable_cooccurrence(
        experiment_dir,
        options=options,
        output_dir=output_dir,
        selected_constraint_names=selected,
    )
    overall = str(document.get("overall_status", "success"))
    ok = bool(document.get("ok", True))
    _emit_json(
        {
            "command": "cooccurrence",
            "experiment_dir": str(experiment_dir),
            "ok": ok,
            "output_dir": str(resolved_out),
            "overall_status": overall,
            "feasible_count": (document.get("metadata") or {}).get(
                "feasible_count"
            ),
            "paths": _paths_payload(written),
        }
    )
    return _exit_for_status(overall, ok=ok)


def _cmd_decision_tree(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.experiment_dir)
    options = DecisionTreeOptions(
        max_depth=int(args.max_depth),
        min_support=int(args.min_support),
        min_lift=float(args.min_lift),
        sample_rows=int(args.sample_rows),
        min_samples_leaf=int(args.min_samples_leaf),
        duckdb_threads=int(args.duckdb_threads),
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_temp_directory=(
            None
            if args.duckdb_temp_directory is None
            else Path(args.duckdb_temp_directory)
        ),
    )
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    resolved_out = _resolve_decision_tree_output_dir(experiment_dir, output_dir)
    selected = _prepare_keep_list(
        args,
        experiment_dir=experiment_dir,
        output_dir=resolved_out,
        implicit_z_drop=getattr(args, "constraint_names", None) is None,
    )
    print(
        "hdb-bf-analyze decision-tree: starting infeasible repair mining for "
        f"{experiment_dir} "
        f"(output: {resolved_out})",
        file=sys.stderr,
        flush=True,
    )
    document, written = run_and_write_constraint_decision_tree(
        experiment_dir,
        options=options,
        output_dir=output_dir,
        selected_constraint_names=selected,
    )
    overall = str(document.get("overall_status", "success"))
    ok = bool(document.get("ok", True))
    meta = document.get("metadata") or {}
    _emit_json(
        {
            "command": "decision-tree",
            "experiment_dir": str(experiment_dir),
            "ok": ok,
            "output_dir": str(resolved_out),
            "overall_status": overall,
            "n_seeds": meta.get("n_seeds"),
            "n_labeled": meta.get("n_labeled"),
            "n_tree_rules": meta.get("n_tree_rules"),
            "n_pattern_move_rows": meta.get("n_pattern_move_rows"),
            "paths": _paths_payload(written),
        }
    )
    return _exit_for_status(overall, ok=ok)


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entrypoint.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Argument vector excluding the program name.

    Returns
    -------
    int
        ``0`` on successful analysis, ``1`` on usage/context/dependency
        failure, ``2`` on structural, scientific, or eligibility failure.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1

    command = str(args.command)
    try:
        if command == "report":
            return _cmd_report(args)
        if command == "integrity":
            return _cmd_integrity(args)
        if command == "sensitivity":
            return _cmd_sensitivity(args)
        if command == "cooccurrence":
            return _cmd_cooccurrence(args)
        if command == "decision-tree":
            return _cmd_decision_tree(args)
        _emit_json(
            {
                "command": command,
                "error": f"unknown command {command!r}",
                "ok": False,
            },
            stream=sys.stderr,
        )
        return 1
    except (
        BruteForceDependencyError,
        BruteForceAnalysisError,
        BruteForceEngineError,
        ValueError,
        FileNotFoundError,
        OSError,
    ) as exc:
        _emit_json(
            {
                "command": command,
                "error": f"{type(exc).__name__}: {exc}",
                "ok": False,
            },
            stream=sys.stderr,
        )
        return 1
    except Exception as exc:  # pragma: no cover - unexpected
        _emit_json(
            {
                "command": command,
                "error": f"{type(exc).__name__}: {exc}",
                "ok": False,
            },
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
