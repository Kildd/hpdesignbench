"""
Load one sealed landscape sample for ELA.

Consumes brute-force or sampling ``dataset/parts/`` via DuckDB. One
repetition is one initial design: ``WHERE repetition = :r`` and
``ORDER BY sample_index ASC``. Reject the whole source on non-finite
``y`` / ``y_p``. Never filter ``analysis_ok``. Never concatenate
repetitions. Never silently deduplicate. Never require ``requested_index``.

This module must not import pflacco or SALib. Call
:func:`core.ioh_core.workflows.la.deps.import_duckdb` only from the
load path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from core.ioh_core.constraint_selection import (
    ConstraintSelectionError,
    select_constraint_names,
    warn_if_constraint_subset,
)
from core.ioh_core.workflows.la.deps import import_duckdb
from core.ioh_core.workflows.la.models import ElaError, LandscapeSample
from core.ioh_core.workflows.la.preprocess import constraint_violations

_SOURCE_SAMPLING = "sampling"
_SOURCE_BRUTE_FORCE = "brute_force"


def phi_and_feasible_from_util(util: NDArray[np.floating]) -> tuple[
    NDArray[np.floating], NDArray[np.bool_]
]:
    """
    Recompute analysis-local ``phi`` and ``F`` from selected utilizations.

    Delegates to :func:`constraint_violations` in ``preprocess.py``.
    """
    _violation, phi, feasible = constraint_violations(util)
    return phi, feasible


def load_landscape_sample(
    experiment_dir: Path,
    *,
    repetition: int | None = None,
    requested_constraint_names: Sequence[str] | None = None,
    source_type: str | None = None,
    lower_bounds: Sequence[int] | None = None,
    upper_bounds: Sequence[int] | None = None,
) -> LandscapeSample:
    """
    Load one repetition (or one brute-force scan) as a landscape sample.

    Parameters
    ----------
    experiment_dir : Path
        Sealed experiment root containing ``dataset/schema.json`` and
        ``dataset/parts/``.
    repetition : int or None, optional
        Required for sampling sources. Ignored when the Parquet has no
        ``repetition`` column (brute-force).
    requested_constraint_names : sequence of str or None, optional
        Keep list in sealed relative order. ``None`` keeps every sealed
        name, including modeling/Z constraints.
    source_type : str or None, optional
        ``sampling`` or ``brute_force``. Inferred from the manifest or
        columns when omitted.
    lower_bounds, upper_bounds : sequence of int or None, optional
        Inclusive catalogue bounds. When omitted, sampling reads
        ``inputs/sampling/rep_XXX/design.json``.

    Returns
    -------
    LandscapeSample
        Ordered rows for one ELA initial design.

    Raises
    ------
    ElaError
        If the source is missing columns, mixes repetitions, contains
        non-finite ``y`` / ``y_p``, or has indices outside bounds.
    """
    experiment_dir = Path(experiment_dir)
    schema = _load_schema(experiment_dir)
    var_names = tuple(schema["var_names"])
    sealed = tuple(schema["constraint_names"])
    try:
        selected = select_constraint_names(sealed, requested_constraint_names)
    except ConstraintSelectionError as exc:
        raise ElaError(str(exc)) from exc
    warn_if_constraint_subset(sealed, selected)

    duckdb = import_duckdb()
    parts_glob = str((experiment_dir / "dataset" / "parts" / "*.parquet").as_posix())
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 1")
        columns = _parquet_columns(connection, parts_glob)
        resolved_source = _resolve_source_type(
            experiment_dir,
            columns,
            source_type=source_type,
        )
        rows, column_names = _fetch_rows(
            connection,
            parts_glob,
            columns=columns,
            source_type=resolved_source,
            repetition=repetition,
            var_names=var_names,
            selected=selected,
        )
    finally:
        connection.close()

    if not rows:
        raise ElaError(
            f"no ELA rows for source_type={resolved_source!r} "
            f"repetition={repetition!r} under {experiment_dir}"
        )

    table = _columns_to_arrays(rows, column_names)
    n_input = int(len(rows))
    X_index = np.column_stack(
        [np.asarray(table[f"idx__{name}"], dtype=np.int64) for name in var_names]
    )
    y_raw = np.asarray(table["y"], dtype=float)
    y_p = np.asarray(table["y_p"], dtype=float)
    if not np.isfinite(y_raw).all() or not np.isfinite(y_p).all():
        raise ElaError(
            "non-finite y or y_p in landscape source; reject whole source"
        )

    U = np.column_stack(
        [
            np.asarray(table[f"util__{name}"], dtype=float)
            for name in selected
        ]
    )
    _violation, phi, feasible = constraint_violations(U)

    if "analysis_ok" in table:
        analysis_ok = np.asarray(table["analysis_ok"], dtype=bool)
    else:
        analysis_ok = np.ones(n_input, dtype=bool)

    sample_index = None
    if "sample_index" in table:
        sample_index = np.asarray(table["sample_index"], dtype=np.int64)

    resolved_repetition = repetition
    if resolved_source == _SOURCE_SAMPLING:
        if repetition is None:
            raise ElaError("sampling landscape load requires repetition")
        resolved_repetition = int(repetition)
        if "repetition" in table:
            unique_reps = {int(value) for value in table["repetition"]}
            if unique_reps != {resolved_repetition}:
                raise ElaError(
                    "adapter concatenated repetitions or failed the "
                    f"repetition filter: {sorted(unique_reps)!r}"
                )

    bounds_lo, bounds_hi, delete_duplicates, design_meta = _resolve_bounds(
        experiment_dir,
        var_names=var_names,
        repetition=resolved_repetition,
        source_type=resolved_source,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    _assert_indices_in_bounds(X_index, bounds_lo, bounds_hi, var_names)

    unique_count = int(np.unique(X_index, axis=0).shape[0])
    n_duplicates = n_input - unique_count
    duplicate_fraction = (
        float(n_duplicates) / float(n_input) if n_input else 0.0
    )

    metadata: dict[str, Any] = {
        "parts_glob": parts_glob,
        "n_input": n_input,
        "n_unique": unique_count,
        "n_duplicates": n_duplicates,
        "duplicate_fraction": duplicate_fraction,
        "uses_stored_feasible_as_F": False,
        "order_by": (
            "sample_index"
            if resolved_source == _SOURCE_SAMPLING
            else "global_index"
        ),
    }
    metadata.update(design_meta)

    return LandscapeSample(
        source_type=resolved_source,
        experiment_dir=experiment_dir,
        repetition=resolved_repetition,
        var_names=var_names,
        constraint_names_sealed=sealed,
        constraint_names_selected=selected,
        constraint_set_is_subset=tuple(selected) != tuple(sealed),
        lower_bounds=bounds_lo,
        upper_bounds=bounds_hi,
        X_index=X_index,
        y_raw=y_raw,
        y_p=y_p,
        U=U,
        phi=phi,
        F=feasible,
        analysis_ok=analysis_ok,
        sample_index=sample_index,
        n_input=n_input,
        n_unique=unique_count,
        n_duplicates=n_duplicates,
        duplicate_fraction=duplicate_fraction,
        delete_duplicates=delete_duplicates,
        metadata=metadata,
    )


def _quote_ident(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ElaError(f"identifier must be a non-empty string, got {name!r}")
    return '"' + name.replace('"', '""') + '"'


def _load_schema(experiment_dir: Path) -> dict[str, Any]:
    path = Path(experiment_dir) / "dataset" / "schema.json"
    if not path.is_file():
        raise ElaError(f"missing dataset/schema.json under {experiment_dir}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ElaError(f"invalid dataset/schema.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ElaError("dataset/schema.json must be a JSON object")
    var_names = payload.get("var_names")
    constraint_names = payload.get("constraint_names")
    if not isinstance(var_names, list) or not var_names:
        raise ElaError("schema.json var_names must be a non-empty list")
    if not isinstance(constraint_names, list) or not constraint_names:
        raise ElaError("schema.json constraint_names must be a non-empty list")
    if not all(isinstance(name, str) and name.strip() for name in var_names):
        raise ElaError("schema.json var_names must be non-empty strings")
    if not all(
        isinstance(name, str) and name.strip() for name in constraint_names
    ):
        raise ElaError("schema.json constraint_names must be non-empty strings")
    document: dict[str, Any] = {
        "var_names": [str(name) for name in var_names],
        "constraint_names": [str(name) for name in constraint_names],
    }
    if "lower_bounds" in payload or "upper_bounds" in payload:
        lower = payload.get("lower_bounds")
        upper = payload.get("upper_bounds")
        if not _is_int_sequence(lower) or not _is_int_sequence(upper):
            raise ElaError(
                "schema.json lower_bounds/upper_bounds must be lists of ints"
            )
        if len(lower) != len(var_names) or len(upper) != len(var_names):
            raise ElaError(
                "schema.json lower_bounds/upper_bounds must match var_names length"
            )
        document["lower_bounds"] = [int(value) for value in lower]
        document["upper_bounds"] = [int(value) for value in upper]
    if "problem_id" in payload:
        problem_id = payload.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise ElaError("schema.json problem_id must be a non-empty string")
        document["problem_id"] = problem_id.strip()
    return document


def load_ela_schema(experiment_dir: Path) -> dict[str, Any]:
    """Return sealed ``dataset/schema.json`` fields needed by ELA."""
    return _load_schema(experiment_dir)


def _is_int_sequence(raw: Any) -> bool:
    if not isinstance(raw, list) or not raw:
        return False
    return all(isinstance(value, int) and not isinstance(value, bool) for value in raw)


def _parquet_columns(connection: Any, parts_glob: str) -> set[str]:
    quoted = parts_glob.replace("'", "''")
    try:
        rows = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{quoted}')"
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        raise ElaError(f"could not read Parquet parts at {parts_glob}: {exc}") from exc
    return {str(row[0]) for row in rows}


def _resolve_source_type(
    experiment_dir: Path,
    columns: set[str],
    *,
    source_type: str | None,
) -> str:
    if source_type is not None:
        name = str(source_type)
        if name not in {_SOURCE_SAMPLING, _SOURCE_BRUTE_FORCE}:
            raise ElaError(f"unknown source_type {source_type!r}")
        return name
    manifest_path = Path(experiment_dir) / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ElaError(f"invalid manifest.json: {exc}") from exc
        workflow = None
        if isinstance(manifest, dict):
            workflow = manifest.get("workflow_type") or manifest.get("workflow")
        if workflow in {_SOURCE_SAMPLING, _SOURCE_BRUTE_FORCE}:
            return str(workflow)
    if "sample_index" in columns and "repetition" in columns:
        return _SOURCE_SAMPLING
    if "global_index" in columns:
        return _SOURCE_BRUTE_FORCE
    raise ElaError(
        "cannot infer ELA source_type; pass source_type=sampling or brute_force"
    )


def _fetch_rows(
    connection: Any,
    parts_glob: str,
    *,
    columns: set[str],
    source_type: str,
    repetition: int | None,
    var_names: Sequence[str],
    selected: Sequence[str],
) -> tuple[list[tuple[Any, ...]], list[str]]:
    if "y" not in columns:
        raise ElaError("landscape source is missing required column y")
    if "y_p" not in columns:
        raise ElaError("landscape source is missing required column y_p")
    missing_idx = [f"idx__{name}" for name in var_names if f"idx__{name}" not in columns]
    if missing_idx:
        raise ElaError(f"landscape source is missing index columns: {missing_idx}")
    missing_util = [
        f"util__{name}" for name in selected if f"util__{name}" not in columns
    ]
    if missing_util:
        raise ElaError(
            "landscape source is missing util columns for selected "
            f"constraints: {missing_util}"
        )
    if "requested_index" in columns:
        # Presence is allowed; it must never be the order key.
        pass

    select_names = [
        *[f"idx__{name}" for name in var_names],
        "y",
        "y_p",
        *[f"util__{name}" for name in selected],
    ]
    if "analysis_ok" in columns:
        select_names.append("analysis_ok")
    if source_type == _SOURCE_SAMPLING:
        if "sample_index" not in columns:
            raise ElaError("sampling landscape source requires sample_index")
        if "repetition" not in columns:
            raise ElaError("sampling landscape source requires repetition")
        select_names.extend(["sample_index", "repetition"])
        if repetition is None:
            raise ElaError("sampling landscape load requires repetition")
        order_sql = f"ORDER BY {_quote_ident('sample_index')} ASC"
        where_sql = f"WHERE {_quote_ident('repetition')} = ?"
        params: list[Any] = [int(repetition)]
    else:
        if "global_index" not in columns:
            raise ElaError("brute_force landscape source requires global_index")
        select_names.append("global_index")
        order_sql = f"ORDER BY {_quote_ident('global_index')} ASC"
        where_sql = ""
        params = []
        if repetition is not None and "repetition" in columns:
            where_sql = f"WHERE {_quote_ident('repetition')} = ?"
            params = [int(repetition)]

    quoted_glob = parts_glob.replace("'", "''")
    sql = (
        "SELECT "
        + ", ".join(_quote_ident(name) for name in select_names)
        + f" FROM read_parquet('{quoted_glob}') "
        + where_sql
        + " "
        + order_sql
    )
    result = connection.execute(sql, params)
    fetched = result.fetchall()
    header = [str(item[0]) for item in result.description]
    return fetched, header


def _columns_to_arrays(
    rows: Sequence[Sequence[Any]],
    column_names: Sequence[str],
) -> dict[str, list[Any]]:
    table: dict[str, list[Any]] = {name: [] for name in column_names}
    for row in rows:
        for name, value in zip(column_names, row):
            table[name].append(value)
    return table


def _resolve_bounds(
    experiment_dir: Path,
    *,
    var_names: Sequence[str],
    repetition: int | None,
    source_type: str,
    lower_bounds: Sequence[int] | None,
    upper_bounds: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...], bool | None, dict[str, Any]]:
    if lower_bounds is not None and upper_bounds is not None:
        lo = tuple(int(value) for value in lower_bounds)
        hi = tuple(int(value) for value in upper_bounds)
        if len(lo) != len(var_names) or len(hi) != len(var_names):
            raise ElaError("lower_bounds/upper_bounds must match var_names length")
        return lo, hi, None, {}

    if source_type == _SOURCE_SAMPLING and repetition is not None:
        design_path = (
            Path(experiment_dir)
            / "inputs"
            / "sampling"
            / f"rep_{int(repetition):03d}"
            / "design.json"
        )
        if design_path.is_file():
            try:
                payload = json.loads(design_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ElaError(f"invalid design.json: {exc}") from exc
            if not isinstance(payload, dict):
                raise ElaError("design.json must be a JSON object")
            lo = tuple(int(value) for value in payload["lower_bounds"])
            hi = tuple(int(value) for value in payload["upper_bounds"])
            delete_duplicates = payload.get("delete_duplicates")
            meta = {
                "n_requested": payload.get("n_requested"),
                "n_unique_sealed": payload.get("n_unique"),
                "n_duplicates_sealed": payload.get("n_duplicates"),
                "duplicate_fraction_sealed": payload.get("duplicate_fraction"),
                "delete_duplicates": delete_duplicates,
            }
            return lo, hi, (
                bool(delete_duplicates) if delete_duplicates is not None else None
            ), meta

    raise ElaError(
        "catalogue bounds are required (pass lower_bounds/upper_bounds or "
        "provide inputs/sampling/rep_XXX/design.json)"
    )


def _assert_indices_in_bounds(
    X_index: NDArray[np.integer],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    var_names: Sequence[str],
) -> None:
    for axis, name in enumerate(var_names):
        lo = int(lower_bounds[axis])
        hi = int(upper_bounds[axis])
        column = X_index[:, axis]
        if np.any(column < lo) or np.any(column > hi):
            raise ElaError(
                f"catalogue index for {name!r} is outside [{lo}, {hi}]"
            )
