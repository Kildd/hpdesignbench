"""
Read-only fixed-target feasibility audit of sealed brute-force catalogues.

The audit does not run structural analysis, does not modify catalogue
files, and does not change official ``y_p`` target attainment. It reports
whether a finite ``y_p <= T`` row is guaranteed to be production-feasible.

Aggregation reads ``fixed_target_feasibility_audit.json`` from the
experiment root. Real 21-problem artefacts stay outside the source tree.
Run them with the JSON request accepted by :func:`main` only when the
sealed catalogues are available locally.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.execution_core.atomic_io import write_json_atomic, write_text_atomic
from core.ioh_core.workflows.optimization_runs.models import _THRESHOLD_RTOL

HASH_VERSION = "sha256-v1"
AUDIT_FILENAME = "fixed_target_feasibility_audit.json"
AUDIT_CSV_FILENAME = "fixed_target_feasibility_audit.csv"
FEASIBILITY_INVARIANT_SQL = "feasible = (analysis_ok AND constraint_feasible)"
_TABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_HEX_HASH = re.compile(r"[0-9a-f]{64}")
_GUARANTEE_STATES = frozenset({"pass", "fail", "unverified"})
_HP_SPANS = (6750, 7425, 8100, 8775, 9450, 10125, 10800)
_HP_CATEGORIES = ("A1", "B1", "B2")
HP_SLAB_PROBLEM_IDS: tuple[str, ...] = tuple(
    f"L{span}_{category}"
    for span in _HP_SPANS
    for category in _HP_CATEGORIES
)
_CSV_FIELDS = (
    "problem_id",
    "target_threshold",
    "state",
    "reason",
    "y_f_star",
    "infeasible_or_failed_minimum",
    "offending_row_count",
    "admissible_posinf_failed_count",
    "catalogue_hash",
    "penalty_alpha",
    "penalty_beta",
    "row_count",
    "distinct_global_indices",
    "min_global_index",
    "max_global_index",
)


def feasibility_invariant_holds(
    analysis_ok: bool,
    constraint_feasible: bool,
    feasible: bool,
) -> bool:
    """
    Return whether the stored flags match the parenthesized schema equality.

    Parameters
    ----------
    analysis_ok, constraint_feasible, feasible : bool
        Catalogue flags. The parentheses are required because ``==`` binds
        more tightly than ``and``.

    Returns
    -------
    bool
        True for the four combinations admitted by
        ``feasible == (analysis_ok and constraint_feasible)``.
    """
    return feasible == (analysis_ok and constraint_feasible)


def feasible_design_target_claim_allowed(
    feasibility_guarantee: str | None,
) -> bool:
    """
    Return whether a plot may say the official target was a feasible design.

    Parameters
    ----------
    feasibility_guarantee : str or None
        Aggregate reporting state.

    Returns
    -------
    bool
        True only for ``pass``.
    """
    return feasibility_guarantee == "pass"


def catalogue_sha256_v1(rows: Sequence[Mapping[str, Any]]) -> str:
    """
    Hash catalogue rows with the ``sha256-v1`` line format.

    Rows are ordered by ``global_index``. Each line is the index, the three
    booleans, and ``float.hex(y_p)``, separated by spaces. Lines are joined
    by ``\\n`` with no trailing newline. Python's built-in ``hash`` is not
    used.

    Parameters
    ----------
    rows : sequence of mapping
        Rows that already have an integer index, three bools, and a float
        ``y_p``.

    Returns
    -------
    str
        Lowercase SHA-256 hex digest.
    """
    ordered = sorted(rows, key=lambda row: int(row["global_index"]))
    lines = [_catalogue_line(row) for row in ordered]
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_catalogue_table(connection: Any, table_name: str) -> list[dict[str, Any]]:
    """
    Read one brute-force result table through an open DuckDB connection.

    Parameters
    ----------
    connection : duckdb connection
        Open connection. This function does not close it.
    table_name : str
        Simple SQL identifier of the result table.

    Returns
    -------
    list of dict
        Python rows with ``global_index``, the three flags, and ``y_p``.

    Raises
    ------
    ValueError
        If ``table_name`` is not a simple identifier.
    """
    if not isinstance(table_name, str) or _TABLE_NAME.fullmatch(table_name) is None:
        raise ValueError(
            f"catalogue table name must be a simple identifier, got {table_name!r}"
        )
    query = (
        "SELECT global_index, analysis_ok, constraint_feasible, feasible, y_p, "
        f"({FEASIBILITY_INVARIANT_SQL}) AS invariant_ok "
        f"FROM {table_name}"
    )
    cursor = connection.execute(query)
    description = cursor.description
    names = [str(column[0]) for column in description]
    loaded: list[dict[str, Any]] = []
    for raw in cursor.fetchall():
        record = dict(zip(names, raw))
        loaded.append(
            {
                "global_index": _native(record.get("global_index")),
                "analysis_ok": _native(record.get("analysis_ok")),
                "constraint_feasible": _native(record.get("constraint_feasible")),
                "feasible": _native(record.get("feasible")),
                "y_p": _native(record.get("y_p")),
            }
        )
    return loaded


def audit_catalogue_thresholds(
    catalogues: Sequence[Mapping[str, Any]],
    *,
    campaign_problem_ids: Sequence[str] | None = None,
    campaign_constraint_names: Sequence[str],
    campaign_penalty_alpha: float,
    campaign_penalty_beta: float,
    campaign_objective_identity: str,
    code_revision: str = "unknown",
    audited_at: str | None = None,
) -> dict[str, Any]:
    """
    Audit exhaustive catalogues at exact manifest thresholds.

    Parameters
    ----------
    catalogues : sequence of mapping
        One entry per available catalogue. A missing campaign problem is
        reported as ``unverified`` / ``missing_catalogue``. Rows may be
        in-memory or loaded from ``duckdb_path`` and ``table_name``.
    campaign_problem_ids : sequence of str or None
        Problems that must be inventoried. ``None`` uses the 21 HP-slab ids.
    campaign_constraint_names, campaign_penalty_alpha, campaign_penalty_beta
        Campaign definition the catalogue must match.
    campaign_objective_identity : str
        Objective identity string. The audit compares it; it does not
        recompute objectives.
    code_revision : str, optional
        Revision recorded in the summary.
    audited_at : str or None, optional
        UTC timestamp. ``None`` uses the current time.

    Returns
    -------
    dict
        JSON summary plus an ``offending_rows`` list. Publish with
        :func:`write_audit_artifacts`, which removes that list from the
        JSON document and writes complete offending rows to Parquet.
    """
    required = (
        tuple(campaign_problem_ids)
        if campaign_problem_ids is not None
        else HP_SLAB_PROBLEM_IDS
    )
    if len(required) != len(set(required)):
        raise ValueError("campaign_problem_ids contains a duplicate")
    by_id: dict[str, Mapping[str, Any]] = {}
    extras: list[Mapping[str, Any]] = []
    for catalogue in catalogues:
        problem_id = str(catalogue.get("problem_id", ""))
        if problem_id in by_id:
            raise ValueError(f"duplicate catalogue for problem_id {problem_id!r}")
        by_id[problem_id] = catalogue
        if problem_id not in required:
            extras.append(catalogue)

    problems: list[dict[str, Any]] = []
    offending_rows: list[dict[str, Any]] = []
    for problem_id in required:
        if problem_id not in by_id:
            problems.append(_missing_problem(problem_id))
            continue
        problem, offending = _audit_one(
            by_id[problem_id],
            constraint_names=tuple(campaign_constraint_names),
            penalty_alpha=float(campaign_penalty_alpha),
            penalty_beta=float(campaign_penalty_beta),
            objective_identity=str(campaign_objective_identity),
        )
        problems.append(problem)
        offending_rows.extend(offending)
    for catalogue in extras:
        problem, offending = _audit_one(
            catalogue,
            constraint_names=tuple(campaign_constraint_names),
            penalty_alpha=float(campaign_penalty_alpha),
            penalty_beta=float(campaign_penalty_beta),
            objective_identity=str(campaign_objective_identity),
        )
        problems.append(problem)
        offending_rows.extend(offending)

    hashes = {
        str(problem["problem_id"]): problem["catalogue_hash"]
        for problem in problems
        if isinstance(problem.get("catalogue_hash"), str)
    }
    return {
        "hash_version": HASH_VERSION,
        "audited_at": audited_at or _utc_now(),
        "code_revision": str(code_revision),
        "campaign": {
            "problem_ids": list(required),
            "constraint_names": [str(name) for name in campaign_constraint_names],
            "penalty_alpha": float(campaign_penalty_alpha),
            "penalty_beta": float(campaign_penalty_beta),
            "objective_identity": str(campaign_objective_identity),
            "catalogue_hashes": hashes,
        },
        "problems": problems,
        "offending_rows": offending_rows,
    }


def write_audit_artifacts(audit: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """
    Write the JSON summary, CSV summary, and offending-row Parquet files.

    Parameters
    ----------
    audit : mapping
        Result of :func:`audit_catalogue_thresholds`.
    output_dir : path
        Directory created if needed. This function does not modify catalogues.

    Returns
    -------
    dict
        Public JSON document, without the complete offending-row lists.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    document = _public_document(audit)
    _attach_parquet(document, audit.get("offending_rows") or [], destination)
    _reject_nonfinite_json(document)
    write_json_atomic(destination / AUDIT_FILENAME, document)
    _write_csv(destination / AUDIT_CSV_FILENAME, document)
    return document


def feasibility_guarantee_for_rows(
    rows: Sequence[Mapping[str, Any]],
    experiment_dir: Path,
) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Match fixed-target rows to a published audit document.

    Parameters
    ----------
    rows : sequence of mapping
        Aggregate task rows.
    experiment_dir : path
        Experiment root that may contain :data:`AUDIT_FILENAME`.

    Returns
    -------
    tuple
        ``(None, [])`` when no row is fixed-target. Otherwise the worst
        state among the pairs, with ``fail`` ahead of ``unverified`` and
        ``unverified`` ahead of ``pass``, and one pair dict per fixed-target
        row. Official ``target_hit`` is not read or changed.
    """
    selected = [
        row
        for row in rows
        if str(row.get("termination_mode") or "") == "fixed_target"
    ]
    if not selected:
        return None, []
    document = _read_audit_document(Path(experiment_dir) / AUDIT_FILENAME)
    pairs = [_match_row(row, document) for row in selected]
    return _rollup([str(pair["state"]) for pair in pairs]), pairs


def main(argv: Sequence[str] | None = None) -> int:
    """
    Audit catalogues described by a JSON request file.

    The request supplies campaign identity and catalogue locations. Missing
    HP-slab ids stay ``unverified``. Output is written under ``--output-dir``
    and is not a source-tree artefact.

    Parameters
    ----------
    argv : sequence of str or None
        CLI arguments. ``None`` reads the process arguments.

    Returns
    -------
    int
        Process status code.
    """
    parser = argparse.ArgumentParser(
        description="Read-only fixed-target feasibility audit."
    )
    parser.add_argument(
        "request",
        type=Path,
        help="JSON file of campaign identity, catalogues, and thresholds",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-revision", default="unknown")
    args = parser.parse_args(list(argv) if argv is not None else None)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    audit = audit_catalogue_thresholds(
        list(request.get("catalogues") or []),
        campaign_problem_ids=request.get("problem_ids"),
        campaign_constraint_names=tuple(request["constraint_names"]),
        campaign_penalty_alpha=float(request["penalty_alpha"]),
        campaign_penalty_beta=float(request["penalty_beta"]),
        campaign_objective_identity=str(request["objective_identity"]),
        code_revision=str(args.code_revision),
    )
    write_audit_artifacts(audit, args.output_dir)
    return 0


def _catalogue_line(row: Mapping[str, Any]) -> str:
    return " ".join(
        (
            str(int(row["global_index"])),
            _bool_token(row["analysis_ok"]),
            _bool_token(row["constraint_feasible"]),
            _bool_token(row["feasible"]),
            float(row["y_p"]).hex(),
        )
    )


def _bool_token(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    raise ValueError("catalogue hash requires real bool flags")


def _native(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        native = item()
        if isinstance(native, (bool, int, float)):
            return native
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _missing_problem(problem_id: str) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "source_path": None,
        "table_name": None,
        "catalogue_hash": None,
        "hash_version": None,
        "penalty_alpha": None,
        "penalty_beta": None,
        "objective_identity": None,
        "constraint_names": [],
        "cardinality": None,
        "row_count": None,
        "distinct_global_indices": None,
        "min_global_index": None,
        "max_global_index": None,
        "y_f_star": None,
        "admissible_posinf_failed_count": None,
        "preflight_reason": "missing_catalogue",
        "thresholds": [],
    }


def _audit_one(
    catalogue: Mapping[str, Any],
    *,
    constraint_names: Sequence[str],
    penalty_alpha: float,
    penalty_beta: float,
    objective_identity: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    problem_id = str(catalogue.get("problem_id", ""))
    rows, load_reason = _load_rows(catalogue)
    definition_reason = _definition_reason(
        catalogue,
        constraint_names=constraint_names,
        penalty_alpha=penalty_alpha,
        penalty_beta=penalty_beta,
        objective_identity=objective_identity,
    )
    cardinality = catalogue.get("cardinality")
    preflight_reason = load_reason or definition_reason
    coverage = _empty_coverage()
    catalogue_hash = None
    y_f_star = None
    infeasible_minimum = None
    posinf_count = None
    if rows is not None and isinstance(cardinality, int) and not isinstance(cardinality, bool):
        (
            structural_reason,
            coverage,
            catalogue_hash,
            y_f_star,
            infeasible_minimum,
            posinf_count,
        ) = _preflight_rows(rows, int(cardinality))
        if preflight_reason is None:
            preflight_reason = structural_reason
    elif preflight_reason is None:
        preflight_reason = "definition_mismatch"

    thresholds = list(catalogue.get("thresholds") or [])
    offending: list[dict[str, Any]] = []
    published_thresholds: list[dict[str, Any]] = []
    if preflight_reason is not None:
        for spec in thresholds:
            published_thresholds.append(
                _unverified_threshold(
                    problem_id,
                    spec,
                    reason=preflight_reason,
                    catalogue_hash=catalogue_hash,
                    penalty_alpha=_optional_float(catalogue.get("penalty_alpha")),
                    penalty_beta=_optional_float(catalogue.get("penalty_beta")),
                )
            )
    else:
        assert rows is not None and y_f_star is not None
        for spec in thresholds:
            entry, full_rows = _classify_threshold(
                problem_id,
                rows,
                y_f_star=y_f_star,
                infeasible_minimum=infeasible_minimum,
                spec=spec,
                catalogue_hash=catalogue_hash,
                penalty_alpha=float(catalogue["penalty_alpha"]),
                penalty_beta=float(catalogue["penalty_beta"]),
            )
            published_thresholds.append(entry)
            if full_rows:
                offending.append(
                    {
                        "problem_id": problem_id,
                        "threshold_index": len(published_thresholds) - 1,
                        "target_threshold": entry["target_threshold"],
                        "rows": full_rows,
                    }
                )
    problem = {
        "problem_id": problem_id,
        "source_path": catalogue.get("source_path"),
        "table_name": catalogue.get("table_name"),
        "catalogue_hash": catalogue_hash,
        "hash_version": HASH_VERSION if catalogue_hash else None,
        "penalty_alpha": _optional_float(catalogue.get("penalty_alpha")),
        "penalty_beta": _optional_float(catalogue.get("penalty_beta")),
        "objective_identity": catalogue.get("objective_identity"),
        "constraint_names": list(catalogue.get("constraint_names") or []),
        "cardinality": cardinality if isinstance(cardinality, int) else None,
        "row_count": coverage["row_count"],
        "distinct_global_indices": coverage["distinct_global_indices"],
        "min_global_index": coverage["min_global_index"],
        "max_global_index": coverage["max_global_index"],
        "y_f_star": y_f_star if preflight_reason is None else None,
        "admissible_posinf_failed_count": (
            posinf_count
            if preflight_reason in (None, "no_finite_feasible_optimum")
            else None
        ),
        "preflight_reason": preflight_reason,
        "thresholds": published_thresholds,
    }
    return problem, offending


def _load_rows(
    catalogue: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]] | None, str | None]:
    if catalogue.get("rows") is not None:
        rows = catalogue.get("rows")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            return list(rows), None
        return None, "catalogue_unreadable"
    path = catalogue.get("duckdb_path")
    table_name = catalogue.get("table_name")
    if not path or not table_name:
        return None, "missing_catalogue"
    try:
        duckdb = importlib.import_module("duckdb")
    except ImportError:
        return None, "catalogue_unreadable"
    try:
        connection = duckdb.connect(str(path), read_only=True)
        try:
            return read_catalogue_table(connection, str(table_name)), None
        finally:
            connection.close()
    except (OSError, ValueError, duckdb.Error):
        return None, "catalogue_unreadable"


def _definition_reason(
    catalogue: Mapping[str, Any],
    *,
    constraint_names: Sequence[str],
    penalty_alpha: float,
    penalty_beta: float,
    objective_identity: str,
) -> str | None:
    names = list(catalogue.get("constraint_names") or [])
    if len(names) != len(set(names)):
        return "definition_mismatch"
    if set(map(str, names)) != set(map(str, constraint_names)):
        return "definition_mismatch"
    if str(catalogue.get("objective_identity")) != objective_identity:
        return "definition_mismatch"
    alpha = catalogue.get("penalty_alpha")
    beta = catalogue.get("penalty_beta")
    if not _same_float(alpha, penalty_alpha) or not _same_float(beta, penalty_beta):
        return "definition_mismatch"
    cardinality = catalogue.get("cardinality")
    if (
        isinstance(cardinality, bool)
        or not isinstance(cardinality, int)
        or cardinality < 1
    ):
        return "definition_mismatch"
    return None


def _empty_coverage() -> dict[str, int | None]:
    return {
        "row_count": None,
        "distinct_global_indices": None,
        "min_global_index": None,
        "max_global_index": None,
    }


def _preflight_rows(
    rows: Sequence[Mapping[str, Any]],
    cardinality: int,
) -> tuple[str | None, dict[str, int | None], str | None, float | None, float | None, int | None]:
    coverage = _empty_coverage()
    coverage["row_count"] = len(rows)
    index_reason, indices = _index_values(rows)
    if index_reason is not None:
        return index_reason, coverage, None, None, None, None
    assert indices is not None
    distinct = set(indices)
    coverage["distinct_global_indices"] = len(distinct)
    coverage["min_global_index"] = min(indices) if indices else None
    coverage["max_global_index"] = max(indices) if indices else None
    catalogue_hash = _hash_if_possible(rows)
    complete = (
        len(rows) == cardinality
        and len(distinct) == cardinality
        and coverage["min_global_index"] == 0
        and coverage["max_global_index"] == cardinality - 1
    )
    if not complete:
        return "incomplete_index_coverage", coverage, catalogue_hash, None, None, None
    invariant_reason = _boolean_reason(rows)
    if invariant_reason is not None:
        return invariant_reason, coverage, catalogue_hash, None, None, None
    if _y_p_reason(rows) is not None:
        return "invalid_y_p", coverage, catalogue_hash, None, None, None
    y_f_star, infeasible_minimum, posinf_count = _minima(rows)
    structural = None if y_f_star is not None else "no_finite_feasible_optimum"
    return (
        structural,
        coverage,
        catalogue_hash,
        y_f_star,
        infeasible_minimum,
        posinf_count,
    )


def _index_values(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str | None, list[int] | None]:
    indices: list[int] = []
    saw_non_integer = False
    for row in rows:
        if not isinstance(row, Mapping) or "global_index" not in row:
            return "null_global_index", None
        value = row.get("global_index")
        if value is None:
            return "null_global_index", None
        if isinstance(value, bool) or not isinstance(value, int):
            saw_non_integer = True
            continue
        indices.append(int(value))
    if saw_non_integer:
        return "non_integer_global_index", None
    return None, indices


def _hash_if_possible(rows: Sequence[Mapping[str, Any]]) -> str | None:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        index = row.get("global_index")
        if isinstance(index, bool) or not isinstance(index, int):
            return None
        flags = [
            row.get("analysis_ok"),
            row.get("constraint_feasible"),
            row.get("feasible"),
        ]
        if any(flag is not True and flag is not False for flag in flags):
            return None
        y_p = row.get("y_p")
        if isinstance(y_p, bool) or not isinstance(y_p, (int, float)):
            return None
        prepared.append(
            {
                "global_index": int(index),
                "analysis_ok": flags[0],
                "constraint_feasible": flags[1],
                "feasible": flags[2],
                "y_p": float(y_p),
            }
        )
    return catalogue_sha256_v1(prepared)


def _boolean_reason(rows: Sequence[Mapping[str, Any]]) -> str | None:
    for row in rows:
        flags = (
            row.get("analysis_ok"),
            row.get("constraint_feasible"),
            row.get("feasible"),
        )
        if any(flag is not True and flag is not False for flag in flags):
            return "feasibility_invariant"
        if not feasibility_invariant_holds(flags[0], flags[1], flags[2]):
            return "feasibility_invariant"
    return None


def _y_p_reason(rows: Sequence[Mapping[str, Any]]) -> str | None:
    for row in rows:
        if _y_p_status(row) == "invalid":
            return "invalid_y_p"
    return None


def _y_p_status(row: Mapping[str, Any]) -> str:
    value = row.get("y_p")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "invalid"
    number = float(value)
    if math.isnan(number) or (math.isinf(number) and number < 0.0):
        return "invalid"
    if math.isinf(number):
        if row.get("analysis_ok") is False and row.get("feasible") is False:
            return "posinf"
        return "invalid"
    return "finite"


def _minima(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float | None, float | None, int]:
    feasible_values: list[float] = []
    infeasible_values: list[float] = []
    posinf_count = 0
    for row in rows:
        status = _y_p_status(row)
        if status == "posinf":
            posinf_count += 1
            continue
        if status != "finite":
            continue
        number = float(row["y_p"])
        if row.get("feasible") is True and row.get("analysis_ok") is True:
            feasible_values.append(number)
        elif row.get("feasible") is not True:
            infeasible_values.append(number)
    y_f_star = min(feasible_values) if feasible_values else None
    infeasible_minimum = min(infeasible_values) if infeasible_values else None
    return y_f_star, infeasible_minimum, posinf_count


def _classify_threshold(
    problem_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    y_f_star: float,
    infeasible_minimum: float | None,
    spec: Mapping[str, Any],
    catalogue_hash: str | None,
    penalty_alpha: float,
    penalty_beta: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    threshold = spec.get("target_threshold")
    percent = spec.get("allowed_deviation_percent")
    reconstructed = None
    if _same_float(percent, percent):
        reconstructed = y_f_star * (1.0 + float(percent) / 100.0)
    matches = (
        reconstructed is not None
        and math.isfinite(reconstructed)
        and _same_float(threshold, reconstructed)
    )
    base = {
        "problem_id": problem_id,
        "target_threshold": None if not _same_float(threshold, threshold) else float(threshold),
        "allowed_deviation_percent": (
            None if not _same_float(percent, percent) else float(percent)
        ),
        "catalogue_hash": catalogue_hash,
        "penalty_alpha": penalty_alpha,
        "penalty_beta": penalty_beta,
        "infeasible_or_failed_minimum": infeasible_minimum,
        "offending_row_count": 0,
        "examples": [],
        "parquet_path": None,
    }
    if not matches:
        return {
            **base,
            "state": "unverified",
            "reason": "threshold_mismatch",
        }, []
    assert threshold is not None
    offending = _offending_rows(rows, float(threshold))
    state = "fail" if offending else "pass"
    reason = "infeasible_can_meet_target" if offending else None
    return {
        **base,
        "target_threshold": float(threshold),
        "state": state,
        "reason": reason,
        "offending_row_count": len(offending),
        "examples": offending[:20],
    }, offending


def _offending_rows(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    offending: list[dict[str, Any]] = []
    for row in rows:
        if _y_p_status(row) != "finite":
            continue
        y_p = float(row["y_p"])
        if y_p > threshold:
            continue
        if row.get("analysis_ok") is True and row.get("feasible") is True:
            continue
        offending.append(
            {
                "global_index": int(row["global_index"]),
                "y_p": y_p,
                "analysis_ok": bool(row["analysis_ok"]),
                "constraint_feasible": bool(row["constraint_feasible"]),
                "feasible": bool(row["feasible"]),
            }
        )
    offending.sort(key=lambda item: int(item["global_index"]))
    return offending


def _unverified_threshold(
    problem_id: str,
    spec: Mapping[str, Any],
    *,
    reason: str,
    catalogue_hash: str | None,
    penalty_alpha: float | None,
    penalty_beta: float | None,
) -> dict[str, Any]:
    threshold = spec.get("target_threshold") if isinstance(spec, Mapping) else None
    percent = spec.get("allowed_deviation_percent") if isinstance(spec, Mapping) else None
    return {
        "problem_id": problem_id,
        "target_threshold": float(threshold) if _same_float(threshold, threshold) else None,
        "allowed_deviation_percent": (
            float(percent) if _same_float(percent, percent) else None
        ),
        "state": "unverified",
        "reason": reason,
        "catalogue_hash": catalogue_hash,
        "penalty_alpha": penalty_alpha,
        "penalty_beta": penalty_beta,
        "infeasible_or_failed_minimum": None,
        "offending_row_count": 0,
        "examples": [],
        "parquet_path": None,
    }


def _public_document(audit: Mapping[str, Any]) -> dict[str, Any]:
    document = {
        key: value
        for key, value in dict(audit).items()
        if key != "offending_rows"
    }
    return json.loads(json.dumps(document, allow_nan=False))


def _attach_parquet(
    document: dict[str, Any],
    offending_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    grouped = {
        (str(item["problem_id"]), int(item["threshold_index"])): list(item["rows"])
        for item in offending_rows
    }
    for problem in document.get("problems") or []:
        for index, threshold in enumerate(problem.get("thresholds") or []):
            if int(threshold.get("offending_row_count") or 0) <= 0:
                continue
            rows = grouped.get((str(problem["problem_id"]), index)) or []
            if not rows:
                continue
            name = f"offending_{problem['problem_id']}_{index}.parquet"
            _write_parquet(output_dir / name, rows)
            threshold["parquet_path"] = name


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    pyarrow = importlib.import_module("pyarrow")
    parquet = importlib.import_module("pyarrow.parquet")
    table = pyarrow.table(
        {
            "global_index": [int(row["global_index"]) for row in rows],
            "y_p": [float(row["y_p"]) for row in rows],
            "analysis_ok": [bool(row["analysis_ok"]) for row in rows],
            "constraint_feasible": [bool(row["constraint_feasible"]) for row in rows],
            "feasible": [bool(row["feasible"]) for row in rows],
        }
    )
    parquet.write_table(table, path, compression="snappy")


def _write_csv(path: Path, document: Mapping[str, Any]) -> None:
    lines = [",".join(_CSV_FIELDS)]
    for problem in document.get("problems") or []:
        thresholds = list(problem.get("thresholds") or [])
        if not thresholds:
            lines.append(_csv_line(problem, None))
            continue
        for threshold in thresholds:
            lines.append(_csv_line(problem, threshold))
    write_text_atomic(path, "\n".join(lines) + "\n")


def _csv_line(problem: Mapping[str, Any], threshold: Mapping[str, Any] | None) -> str:
    payload = {field: problem.get(field) for field in _CSV_FIELDS}
    if threshold is not None:
        for field in (
            "target_threshold",
            "state",
            "reason",
            "infeasible_or_failed_minimum",
            "offending_row_count",
            "catalogue_hash",
            "penalty_alpha",
            "penalty_beta",
        ):
            payload[field] = threshold.get(field)
    else:
        payload["state"] = "unverified"
        payload["reason"] = problem.get("preflight_reason")
    buffer = csv.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_CSV_FIELDS, lineterminator="")
    writer.writerow({key: "" if payload.get(key) is None else payload.get(key) for key in _CSV_FIELDS})
    return buffer.getvalue()


def _reject_nonfinite_json(document: Mapping[str, Any]) -> None:
    json.dumps(document, allow_nan=False)


def _read_audit_document(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("hash_version") != HASH_VERSION:
        return None
    return payload


def _match_row(row: Mapping[str, Any], document: Mapping[str, Any] | None) -> dict[str, Any]:
    problem_id = str(row.get("problem_id") or "")
    threshold = row.get("target_threshold")
    base = {
        "problem_id": problem_id,
        "target_threshold": float(threshold) if _same_float(threshold, threshold) else None,
        "state": "unverified",
        "reason": "missing_audit",
        "catalogue_hash": None,
    }
    if document is None:
        return base
    matches = [
        item
        for item in _threshold_records(document)
        if item["problem_id"] == problem_id
        and _identical_float(item.get("target_threshold"), threshold)
    ]
    if len(matches) != 1:
        base["reason"] = "threshold_mismatch" if matches else "missing_audit"
        if not matches and any(
            str(problem.get("problem_id")) == problem_id
            for problem in document.get("problems") or []
        ):
            base["reason"] = "threshold_mismatch"
        return base
    pair = matches[0]
    campaign = document.get("campaign") if isinstance(document.get("campaign"), Mapping) else {}
    expected_hash = dict(campaign.get("catalogue_hashes") or {}).get(problem_id)
    pair_hash = pair.get("catalogue_hash")
    if (
        not isinstance(expected_hash, str)
        or not isinstance(pair_hash, str)
        or expected_hash != pair_hash
        or _HEX_HASH.fullmatch(pair_hash) is None
    ):
        base["reason"] = "hash_mismatch"
        base["catalogue_hash"] = pair_hash if isinstance(pair_hash, str) else None
        return base
    if not _same_float(pair.get("penalty_alpha"), campaign.get("penalty_alpha")):
        base["reason"] = "penalty_mismatch"
        base["catalogue_hash"] = pair_hash
        return base
    if not _same_float(pair.get("penalty_beta"), campaign.get("penalty_beta")):
        base["reason"] = "penalty_mismatch"
        base["catalogue_hash"] = pair_hash
        return base
    state = str(pair.get("state") or "")
    if state not in _GUARANTEE_STATES:
        base["reason"] = "missing_audit"
        base["catalogue_hash"] = pair_hash
        return base
    return {
        "problem_id": problem_id,
        "target_threshold": float(threshold),
        "state": state,
        "reason": pair.get("reason"),
        "catalogue_hash": pair_hash,
    }


def _threshold_records(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for problem in document.get("problems") or []:
        if not isinstance(problem, Mapping):
            continue
        for threshold in problem.get("thresholds") or []:
            if isinstance(threshold, Mapping):
                records.append(dict(threshold))
    return records


def _rollup(states: Sequence[str]) -> str:
    if any(state == "fail" for state in states):
        return "fail"
    if any(state != "pass" for state in states):
        return "unverified"
    return "pass"


def _identical_float(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        first = float(left)
        second = float(right)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(first) or not math.isfinite(second):
        return False
    return first == second


def _same_float(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        first = float(left)
        second = float(right)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(first) or not math.isfinite(second):
        return False
    return math.isclose(first, second, rel_tol=_THRESHOLD_RTOL, abs_tol=0.0)


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


if __name__ == "__main__":
    raise SystemExit(main())
