"""
HP-shell predesign grid, ranked by the production penalized objective.

Each candidate is scored by the same ``analysis`` the optimisation run
uses. ``predesign_y_p`` is that ``y_p``. An optional structural library
reuses sealed section artifacts; a miss or key mismatch writes nothing.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from core.ioh_core.algorithms.native.deb_record import (
    deb_quality_key,
    validate_deb_auxiliary_record,
)

PREDESIGN_H_GES_MM = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0)
PREDESIGN_T_MM = (20.0, 40.0, 60.0, 80.0, 100.0)
PREDESIGN_REINF_PAIRS: tuple[tuple[float, float], ...] = (
    (5.0, 1.81),
    (15.0, 1.81),
    (30.0, 1.81),
    (30.0, 3.62),
    (30.0, 7.24),
    (30.0, 10.86),
)
PREDESIGN_KAP_T_PERCENT = (0.0, 15.0, 30.0, 45.0, 60.0)
PREDESIGN_CONCRETE_GRADES = (
    "C25/30",
    "C50/60",
    "C100/115",
)
PREDESIGN_FIXED_PARAMS = {
    "geom_dy_mm": 50.0,
    "geom_t_infill_mm": 0.0,
    "geom_t_screed_mm": 50.0,
}
PREDESIGN_CANDIDATE_COUNT = (
    len(PREDESIGN_H_GES_MM)
    * len(PREDESIGN_T_MM)
    * len(PREDESIGN_REINF_PAIRS)
    * len(PREDESIGN_KAP_T_PERCENT)
    * len(PREDESIGN_CONCRETE_GRADES)
)





def _catalog_index(values: Sequence[Any], value: Any) -> int:
    """Return the catalog index of ``value``, matching floats by equality."""
    catalog = list(values)
    try:
        return catalog.index(value)
    except ValueError:
        pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        target = float(value)
        for index, item in enumerate(catalog):
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                if float(item) == target:
                    return index
    raise ValueError(f"value {value!r} is not in catalog {catalog}")


def encode_catalogue_indices(
    model: Mapping[str, dict],
    var_names: Sequence[str],
    params: Mapping[str, Any],
) -> tuple[int, ...]:
    """
    Map decoded parameter values to inclusive catalogue-index coordinates.

    Parameters
    ----------
    model : mapping
        Parameter model with ``values``, ``lb``, and ``ub`` per variable.
    var_names : sequence of str
        Ordered optimisation variable names.
    params : mapping
        Decoded parameter dict (must contain every ``var_names`` entry).

    Returns
    -------
    tuple of int
        Catalogue-index vector in ``var_names`` order.

    Raises
    ------
    ValueError
        If a value is missing from the catalog or the index is outside
        the variable's ``lb``/``ub``.
    """
    indices: list[int] = []
    for name in var_names:
        meta = model[name]
        idx = _catalog_index(meta["values"], params[name])
        lo = int(meta["lb"])
        hi = int(meta["ub"])
        if idx < lo or idx > hi:
            raise ValueError(
                f"{name} catalog index {idx} is outside inclusive bounds "
                f"[{lo}, {hi}]"
            )
        indices.append(idx)
    return tuple(indices)


def predesign_base_params(decode, lb_idx: Iterable[int]) -> dict:
    """Decoded lower-bound params with locked predesign fixed values.

    Span and other problem-fixed coordinates come from ``decode(lb_idx)``.
    """
    params = dict(decode(list(lb_idx)))
    params.update(PREDESIGN_FIXED_PARAMS)
    return params


def _apply_reinf_lookup(params: dict, model: Mapping[str, dict]) -> None:
    key_vals = list(model["reinf_a_tex_mm2"]["values"])
    idx = key_vals.index(params["reinf_a_tex_mm2"])
    params["mat_reinf_id"] = model["mat_reinf_id"]["values"][idx]


def iter_predesign_candidates(
    base_params: Mapping[str, Any],
    model: Mapping[str, dict],
) -> Iterator[tuple[int, dict]]:
    """Yield ``(candidate_index, params)`` in height/t/reinf/prestress/grade order."""
    index = 0
    for h_ges in PREDESIGN_H_GES_MM:
        for t_mm in PREDESIGN_T_MM:
            for nt, a_tex in PREDESIGN_REINF_PAIRS:
                for kap in PREDESIGN_KAP_T_PERCENT:
                    for conc_id in PREDESIGN_CONCRETE_GRADES:
                        params = dict(base_params)
                        params.update(PREDESIGN_FIXED_PARAMS)
                        params["geom_h_ges_mm"] = h_ges
                        params["geom_t_mm"] = t_mm
                        params["geom_nt"] = nt
                        params["reinf_a_tex_mm2"] = a_tex
                        params["reinf_kap_t_percent"] = kap
                        params["mat_conc_id"] = conc_id
                        _apply_reinf_lookup(params, model)
                        yield index, params
                        index += 1


def rank_predesign_rows(rows: list[dict]) -> list[dict]:
    """Stable rank: predesign_y_p, then y, then candidate_index. Keeps every row."""

    def _sort_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
        yp = row.get("predesign_y_p", row.get("y_p"))
        y = row.get("y")
        idx = int(row["candidate_index"])
        yp_f = float(yp) if yp is not None and math.isfinite(float(yp)) else math.inf
        y_f = float(y) if y is not None and math.isfinite(float(y)) else math.inf
        return (yp_f, y_f, idx)

    ranked = [dict(row) for row in rows]
    ranked.sort(key=_sort_key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _deb_sort_key(row: Mapping[str, Any]) -> tuple[int, float, int]:
    quality = deb_quality_key(
        feasible=bool(row["feasible"]),
        y=float(row["y"]),
        phi=float(row["predesign_phi"]),
    )
    return (quality[0], quality[1], int(row["candidate_index"]))


def assign_deb_ranks(rows: list[dict]) -> list[dict]:
    """
    Attach a unique one-based ``deb_rank`` and return that Deb order.

    ``deb_rank`` is the position in the sorted sequence, so equal-phi
    infeasible designs still receive distinct ranks. ``candidate_index``
    breaks those ties. The input list keeps its order.
    """
    ordered = [dict(row) for row in rows]
    ordered.sort(key=_deb_sort_key)
    rank_by_index: dict[int, int] = {}
    for rank, row in enumerate(ordered, start=1):
        row["deb_rank"] = rank
        rank_by_index[int(row["candidate_index"])] = rank
    for row in rows:
        row["deb_rank"] = rank_by_index[int(row["candidate_index"])]
    return ordered


def predesign_prefix_diagnostics(
    legacy_rows: Sequence[Mapping[str, Any]],
    deb_rows: Sequence[Mapping[str, Any]],
    prefix_lengths: Sequence[int],
) -> list[dict[str, int]]:
    """
    Compare feasible counts and candidate-index overlap for each prefix.

    Both rankings must contain the same candidate indices. Overlap is the
    size of the index-set intersection, so duplicate vectors still count
    as different candidates.
    """
    legacy_ids = [int(row["candidate_index"]) for row in legacy_rows]
    deb_ids = [int(row["candidate_index"]) for row in deb_rows]
    if sorted(legacy_ids) != sorted(deb_ids):
        raise ValueError(
            "predesign rankings do not contain the same candidate indices"
        )
    reports: list[dict[str, int]] = []
    for raw_k in prefix_lengths:
        k = int(raw_k)
        if k < 1 or k > len(legacy_rows):
            raise ValueError(
                f"prefix length {k} is outside 1..{len(legacy_rows)}"
            )
        legacy_prefix = list(legacy_rows[:k])
        deb_prefix = list(deb_rows[:k])
        overlap = {
            int(row["candidate_index"]) for row in legacy_prefix
        } & {int(row["candidate_index"]) for row in deb_prefix}
        reports.append(
            {
                "k": k,
                "legacy_feasible": sum(
                    1 for row in legacy_prefix if bool(row["feasible"])
                ),
                "deb_feasible": sum(
                    1 for row in deb_prefix if bool(row["feasible"])
                ),
                "candidate_index_overlap": len(overlap),
            }
        )
    return reports


def predesign_prefix_report(
    results_csv: str | Path,
    prefix_lengths: Sequence[int],
) -> list[dict[str, int]]:
    """
    Read one ``results.csv`` and compare its two rankings.

    ``feasible`` is the boolean stored by the writer. Overlap uses
    ``candidate_index``.
    """
    with Path(results_csv).open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    parsed: list[dict] = []
    for row in table:
        flag = row["feasible"]
        if flag == "True":
            feasible = True
        elif flag == "False":
            feasible = False
        else:
            raise ValueError(
                f"results.csv feasible must be True or False, got {flag!r}"
            )
        parsed.append(
            {
                "candidate_index": int(row["candidate_index"]),
                "feasible": feasible,
                "rank": int(row["rank"]),
                "deb_rank": int(row["deb_rank"]),
            }
        )
    legacy = sorted(parsed, key=lambda row: int(row["rank"]))
    deb = sorted(parsed, key=lambda row: int(row["deb_rank"]))
    return predesign_prefix_diagnostics(legacy, deb, prefix_lengths)


def _flatten_result_row(candidate_index: int, params: dict, result: dict) -> dict:
    cv = result.get("constraint_values") or {}
    err = result.get("analysis_error")
    err_text = ""
    if isinstance(err, dict):
        err_text = str(err.get("message") or err)
    elif err is not None:
        err_text = str(err)
    row = {
        "candidate_index": candidate_index,
        "geom_h_ges_mm": params.get("geom_h_ges_mm"),
        "geom_t_mm": params.get("geom_t_mm"),
        "geom_nt": params.get("geom_nt"),
        "reinf_a_tex_mm2": params.get("reinf_a_tex_mm2"),
        "mat_conc_id": params.get("mat_conc_id"),
        "mat_reinf_id": params.get("mat_reinf_id"),
        "geom_span_mm": params.get("geom_span_mm"),
        "geom_dy_mm": params.get("geom_dy_mm"),
        "reinf_kap_t_percent": params.get("reinf_kap_t_percent"),
        "geom_t_infill_mm": params.get("geom_t_infill_mm"),
        "geom_t_screed_mm": params.get("geom_t_screed_mm"),
        "y": result.get("y"),
        "predesign_y_p": result.get("y_p"),
        "analysis_ok": result.get("analysis_ok"),
        "feasible": result.get("feasible"),
        "analysis_error": err_text,
    }
    for name, value in cv.items():
        row[str(name)] = value
    return row


def _active_constraint_names(constraints: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for name, conf in constraints.items():
        if isinstance(conf, Mapping) and not conf.get("active", True):
            continue
        names.append(str(name))
    return tuple(names)


def _invalid_utilization(run_config: Any, base_params: Mapping[str, Any]) -> float:
    if run_config is not None and hasattr(run_config, "invalid_utilization"):
        return float(run_config.invalid_utilization)
    return float(base_params.get("opt_invalid_utilization", 99.0))


def _record_from_analysis(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "y",
            "violations_",
            "constraint_values",
            "analysis_ok",
            "feasible",
        )
        if key in result
    }


def _attach_phi(
    row: dict,
    result: Mapping[str, Any],
    *,
    active: Sequence[str],
    invalid: float,
) -> dict:
    channels = validate_deb_auxiliary_record(
        _record_from_analysis(result),
        active_constraint_names=active,
        invalid_utilization=invalid,
    )
    _require_predesign_y_p(result.get("y_p"), analysis_ok=channels.analysis_ok)
    row["_predesign_phi"] = channels.phi
    return row


def rank_predesign_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    active_constraint_names: Sequence[str],
    invalid_utilization: float,
) -> list[dict]:
    """Validate auxiliary records and return production-penalty order."""
    prepared: list[dict] = []
    for source in rows:
        row = dict(source)
        channels = validate_deb_auxiliary_record(
            _record_from_analysis(row),
            active_constraint_names=active_constraint_names,
            invalid_utilization=invalid_utilization,
        )
        y_p = row["predesign_y_p"] if "predesign_y_p" in row else row.get("y_p")
        _require_predesign_y_p(y_p, analysis_ok=channels.analysis_ok)
        row.pop("violations_", None)
        row.pop("constraint_values", None)
        row["_predesign_phi"] = channels.phi
        prepared.append(row)
    return finalize_predesign_ranks(prepared)


def _require_unique_candidate_indexes(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[int] = set()
    for row in rows:
        raw = row.get("candidate_index")
        if isinstance(raw, bool):
            raise ValueError(
                f"candidate_index must be an integer, got {raw!r}"
            )
        try:
            index = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"candidate_index must be an integer, got {raw!r}"
            ) from exc
        if index in seen:
            raise ValueError(f"duplicate candidate_index {index}")
        seen.add(index)


def _require_predesign_y_p(raw: Any, *, analysis_ok: bool) -> None:
    """Reject a production ``y_p`` the Deb optimizer would also reject."""
    try:
        if isinstance(raw, bool):
            raise TypeError
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"predesign y_p must be numeric, got {raw!r}"
        ) from exc
    if math.isnan(value) or (math.isinf(value) and value < 0.0):
        raise ValueError("predesign y_p must not be NaN or -inf")
    if analysis_ok and math.isinf(value):
        raise ValueError(
            "y_p=+inf is permitted only for an infeasible failed analysis"
        )


def finalize_predesign_ranks(rows: list[dict]) -> list[dict]:
    """Production-penalty order with ``predesign_phi`` and ``deb_rank`` appended."""
    _require_unique_candidate_indexes(rows)
    ranked = rank_predesign_rows(rows)
    for row in ranked:
        row["predesign_phi"] = row.pop("_predesign_phi")
    assign_deb_ranks(ranked)
    return ranked


def _csv_text(rows: list[dict], fieldnames: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _initial_design_text(
    problem_id: str,
    vectors: Sequence[Sequence[int]],
) -> str:
    payload = {
        "schema_version": 1,
        "problem_id": str(problem_id),
        "n": len(vectors),
        "vectors": [list(vector) for vector in vectors],
    }
    return json.dumps(payload, indent=2) + "\n"


def _publish_predesign_texts(files: Sequence[tuple[Path, str]]) -> None:
    """
    Replace the shelf files together.

    Handled errors roll the replaced files back, including files that did
    not exist. A crash during the replacement window is not recovered.
    Concurrent readers of that window are unsupported.
    """
    staged: list[Path] = []
    try:
        for dest, text in files:
            dest.parent.mkdir(parents=True, exist_ok=True)
            stage = dest.with_name("." + dest.name + ".staged")
            stage.write_text(text, encoding="utf-8")
            staged.append(stage)
    except Exception:
        for stage in staged:
            if stage.is_file():
                stage.unlink()
        raise

    backups: list[tuple[Path, Path | None, bool]] = []
    committed: list[int] = []
    try:
        for index, (dest, _text) in enumerate(files):
            existed = dest.is_file()
            backup: Path | None = None
            if existed:
                backup = dest.with_name("." + dest.name + ".bak")
                backup.write_bytes(dest.read_bytes())
            backups.append((dest, backup, existed))
            os.replace(staged[index], dest)
            committed.append(index)
    except Exception:
        for index in reversed(committed):
            dest, backup, existed = backups[index]
            if existed and backup is not None and backup.is_file():
                os.replace(backup, dest)
            elif not existed and dest.is_file():
                dest.unlink()
        for stage in staged:
            if stage.is_file():
                stage.unlink()
        for _dest, backup, existed in backups:
            if backup is not None and backup.is_file() and existed:
                backup.unlink()
        raise
    else:
        for _dest, backup, _existed in backups:
            if backup is not None and backup.is_file():
                backup.unlink()



def _row_kap_t_percent(row: Mapping[str, Any]) -> float:
    """Prestress percent stored on the predesign row."""
    if "reinf_kap_t_percent" not in row:
        raise ValueError("predesign row is missing reinf_kap_t_percent")
    raw = row["reinf_kap_t_percent"]
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        raise ValueError("predesign row is missing reinf_kap_t_percent")
    return float(raw)


def index_vectors_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: Mapping[str, dict],
    var_names: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    """Encode ranked predesign rows to catalogue-index vectors."""
    vectors: list[tuple[int, ...]] = []
    for row in rows:
        params = {
            "geom_h_ges_mm": float(row["geom_h_ges_mm"]),
            "geom_t_mm": float(row["geom_t_mm"]),
            "geom_nt": float(row["geom_nt"]),
            "reinf_a_tex_mm2": float(row["reinf_a_tex_mm2"]),
            "mat_conc_id": row["mat_conc_id"],
            "mat_reinf_id": row["mat_reinf_id"],
            "geom_span_mm": float(row["geom_span_mm"]),
            **PREDESIGN_FIXED_PARAMS,
            "reinf_kap_t_percent": _row_kap_t_percent(row),
        }
        vectors.append(encode_catalogue_indices(model, var_names, params))
    return tuple(vectors)


def _resolve_hp_slab_bundle(problem_id: str) -> tuple[dict, dict]:
    """Load one HP-slab problem with the optimisation problem builder."""
    from importlib import import_module
    from importlib.resources import files

    from core.ioh_core.import_specs import load_materials_registry
    from core.ioh_core.problem_builder import build_problems_for_slab

    slab_module = import_module("slab_construction.slabs.hp_slab.analysis")
    slab_dir = Path(str(files("slab_construction.slabs").joinpath("hp_slab")))
    problems = build_problems_for_slab(
        slab_dir,
        slab_module,
        problem_ids=(str(problem_id),),
    )
    materials = load_materials_registry(slab_dir / "materials.csv")
    return problems[str(problem_id)], materials


def write_predesign_run(
    problem_id: str,
    output_root: str | Path,
    *,
    evaluated_rows: Sequence[Mapping[str, Any]] | None = None,
    library_path: str | Path | None = None,
) -> None:
    """
    Write one problem's ranked predesign table and both InitialDesign files.

    ``evaluated_rows`` injects already evaluated rows for tests. Those rows
    carry the same auxiliary record the analysis returns. The production
    path scores every candidate once. ``library_path`` reuses a sealed
    structural library. A missing artifact, a structural-key mismatch, or a
    malformed auxiliary record raises before the shelf is replaced.

    The three shelf files are staged and then replaced. A handled error
    restores the previous files, including restoring absence. Rebuilding
    always rescores the supplied rows; it does not read a surviving JSON
    file as the source.
    """
    from core.ioh_core.workflows.predesign_paths import (
        predesign_initial_design_deb_path,
        predesign_initial_design_path,
        predesign_results_path,
    )

    bundle, materials = _resolve_hp_slab_bundle(problem_id)
    active = tuple(str(name) for name in bundle["active_constraint_names"])
    invalid = _invalid_utilization(bundle["run_config"], {})
    if evaluated_rows is None:
        base = predesign_base_params(bundle["decode"], bundle["lb_idx"])
        rows = evaluate_predesign_grid(
            materials=materials,
            base_params=base,
            model=bundle["model"],
            constraints=bundle["constraints"],
            analysis_fn=bundle["analysis_fn"],
            library_path=library_path,
            run_config=bundle["run_config"],
            var_names=bundle["var_names"],
            lb_idx=bundle["lb_idx"],
            ub_idx=bundle["ub_idx"],
        )
    else:
        rows = rank_predesign_records(
            evaluated_rows,
            active_constraint_names=active,
            invalid_utilization=invalid,
        )

    results_path = predesign_results_path(output_root, problem_id)
    design_path = predesign_initial_design_path(output_root, problem_id)
    deb_path = predesign_initial_design_deb_path(output_root, problem_id)
    replaced = [
        path.name
        for path in (results_path, design_path, deb_path)
        if path.is_file()
    ]
    if replaced:
        print(
            "replacing InitialDesign for problem "
            f"{problem_id}: {', '.join(replaced)}",
            file=sys.stderr,
        )
    if rows:
        fieldnames = list(rows[0].keys())
        for row in rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    else:
        fieldnames = [
            "candidate_index",
            "rank",
            "predesign_y_p",
            "y",
            "predesign_phi",
            "deb_rank",
        ]
    deb_rows = sorted(rows, key=lambda row: int(row["deb_rank"]))
    legacy_vectors = index_vectors_from_rows(
        rows,
        model=bundle["model"],
        var_names=bundle["var_names"],
    )
    deb_vectors = index_vectors_from_rows(
        deb_rows,
        model=bundle["model"],
        var_names=bundle["var_names"],
    )
    _publish_predesign_texts(
        (
            (results_path, _csv_text(rows, fieldnames)),
            (design_path, _initial_design_text(problem_id, legacy_vectors)),
            (deb_path, _initial_design_text(problem_id, deb_vectors)),
        )
    )


def _score_with_library(
    params: Mapping[str, Any],
    *,
    materials: dict,
    model: Mapping[str, dict],
    constraints: Mapping[str, Any],
    analysis_fn: Any,
    run_config: Any,
    var_names: Sequence[str],
    space: Any,
    provider: Any,
    snapshot_hash: str,
) -> dict:
    from core.ioh_core.structural_artifacts.artifact import finish_variable_names
    from core.ioh_core.structural_artifacts.indexing import (
        global_index_to_structural_id,
    )
    from core.ioh_core.workflows.brute_force.indexing import indices_to_linear
    from core.ioh_core.workflows.brute_force.structural import evaluate_finish_group

    vector = encode_catalogue_indices(model, var_names, params)
    structural_id = global_index_to_structural_id(
        indices_to_linear(vector, space),
        space,
    )
    results = evaluate_finish_group(
        [dict(params)],
        constraints=constraints,
        materials=materials,
        analysis_fn=analysis_fn,
        finish_names=finish_variable_names(var_names),
        snapshot_hash=snapshot_hash,
        run_config=run_config,
        provider=provider,
        structural_id=structural_id,
    )
    return results[0]


def evaluate_predesign_grid(
    *,
    materials: dict,
    base_params: dict,
    model: Mapping[str, dict],
    constraints: Mapping[str, Any],
    limit: int | None = None,
    analysis_fn: Any | None = None,
    library_path: str | Path | None = None,
    run_config: Any | None = None,
    var_names: Sequence[str] | None = None,
    lb_idx: Sequence[int] | None = None,
    ub_idx: Sequence[int] | None = None,
    snapshot_hash: str = "",
) -> list[dict]:
    """Score the predesign grid with production analysis and rank the rows."""
    from slab_construction.slabs.hp_slab.analysis import analysis

    if library_path is None:
        active = _active_constraint_names(constraints)
        invalid = _invalid_utilization(run_config, base_params)
        rows: list[dict] = []
        for index, params in iter_predesign_candidates(base_params, model):
            if limit is not None and index >= int(limit):
                break
            if analysis_fn is None:
                result = analysis(params, constraints, materials)
            else:
                result = analysis_fn(params)
            row = _flatten_result_row(index, params, result)
            rows.append(
                _attach_phi(
                    row,
                    result,
                    active=active,
                    invalid=invalid,
                )
            )
        return finalize_predesign_ranks(rows)

    if analysis_fn is None or var_names is None or lb_idx is None or ub_idx is None:
        raise ValueError(
            "library predesign requires analysis_fn, var_names, lb_idx, and ub_idx"
        )
    from core.ioh_core.structural_artifacts.library import open_structural_library
    from core.ioh_core.structural_artifacts.provider import build_structural_provider
    from core.ioh_core.workflows.brute_force.models import IndexSpace

    names = tuple(str(name) for name in var_names)
    space = IndexSpace.from_bounds(
        names,
        tuple(int(v) for v in lb_idx),
        tuple(int(v) for v in ub_idx),
    )
    active = _active_constraint_names(constraints)
    invalid = _invalid_utilization(run_config, base_params)
    rows = []
    with open_structural_library(
        Path(library_path),
        allow_snapshot_hash_mismatch=True,
    ) as library:
        provider = build_structural_provider(
            "library",
            lookup=library,
            allow_snapshot_hash_mismatch=True,
        )
        for index, params in iter_predesign_candidates(base_params, model):
            if limit is not None and index >= int(limit):
                break
            result = _score_with_library(
                params,
                materials=materials,
                model=model,
                constraints=constraints,
                analysis_fn=analysis_fn,
                run_config=run_config,
                var_names=names,
                space=space,
                provider=provider,
                snapshot_hash=snapshot_hash,
            )
            row = _flatten_result_row(index, params, result)
            rows.append(
                _attach_phi(
                    row,
                    result,
                    active=active,
                    invalid=invalid,
                )
            )
    return finalize_predesign_ranks(rows)





