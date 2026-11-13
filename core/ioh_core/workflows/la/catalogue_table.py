"""Load complete indexed catalogue landscapes from DuckDB result tables."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.la.lon_landscape import get_lon_landscape
from core.ioh_core.workflows.la.models import (
    ElaError,
    LANDSCAPE_Y_P,
    LANDSCAPE_Y_RAW,
)
from core.ioh_core.workflows.la.preprocess import analysis_local_phi_f_sql


def load_catalogue_y_p(
    connection: Any,
    table: str,
    space: IndexSpace,
    *,
    expected_n: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_] | None]:
    """Scatter stored ``y_p`` and optional feasibility into catalogue order."""
    return load_catalogue_landscape(
        connection,
        table,
        space,
        expected_n=expected_n,
        landscape=LANDSCAPE_Y_P,
    )


def load_catalogue_landscape(
    connection: Any,
    table: str,
    space: IndexSpace,
    *,
    expected_n: int,
    landscape: str,
) -> tuple[NDArray[np.float64], NDArray[np.bool_] | None]:
    """Scatter one supported stored landscape and feasibility into catalogue order."""
    spec = get_lon_landscape(landscape)
    selected = spec.token
    n_dim = len(space.var_names)
    idx_selects = ", ".join(
        f"CAST({_quote_ident(f'idx__{name}')} AS BIGINT) AS "
        f"{_quote_ident(f'i{dim}')}"
        for dim, name in enumerate(space.var_names)
    )
    y_col = _quote_ident(spec.source_column)
    has_feasible = _has_column(connection, table, "feasible")
    feasible_sql = (
        f"CAST({_quote_ident('feasible')} AS BOOLEAN) AS {_quote_ident('feasible')}"
        if has_feasible
        else "NULL AS feasible"
    )
    query = (
        f"SELECT {idx_selects}, CAST({y_col} AS DOUBLE) AS {_quote_ident('value')}, "
        f"{feasible_sql} FROM {table}"
    )
    try:
        result = connection.execute(query)
    except Exception as exc:
        raise ElaError(f"could not read catalogue {selected}: {exc}") from exc
    fetch_numpy = getattr(result, "fetchnumpy", None)
    if callable(fetch_numpy):
        arrays = fetch_numpy()
        idx_cols = [_as_int64_vector(arrays[f"i{dim}"]) for dim in range(n_dim)]
        y_values = _as_float64_vector(arrays["value"])
        feasible_values = (
            _as_bool_vector(arrays["feasible"]) if has_feasible else None
        )
    else:
        rows = result.fetchall()
        if len(rows) != expected_n:
            raise ElaError(
                "catalogue row count "
                f"({len(rows)}) does not match cardinality ({expected_n})"
            )
        payload = np.asarray(rows, dtype=object)
        idx_cols = [
            np.asarray(payload[:, dim], dtype=np.int64) for dim in range(n_dim)
        ]
        y_values = np.asarray(payload[:, n_dim], dtype=np.float64)
        if has_feasible:
            raw_feasible = payload[:, n_dim + 1]
            if any(value is None for value in raw_feasible):
                raise ElaError("catalogue feasible column contains NULL")
            feasible_values = np.asarray(raw_feasible, dtype=np.bool_)
        else:
            feasible_values = None
    return _scatter_catalogue(
        idx_cols,
        y_values,
        feasible_values,
        space=space,
        expected_n=expected_n,
    )


def load_catalogue_y_phi_f(
    connection: Any,
    table: str,
    space: IndexSpace,
    constraint_names: Sequence[str],
    *,
    expected_n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Scatter stored ``y`` plus analysis-local ``phi`` / ``F`` into catalogue order."""
    names = tuple(constraint_names)
    if not names:
        raise ElaError("selected phi/F load requires at least one constraint")
    n_dim = len(space.var_names)
    idx_selects = ", ".join(
        f"CAST({_quote_ident(f'idx__{name}')} AS BIGINT) AS "
        f"{_quote_ident(f'i{dim}')}"
        for dim, name in enumerate(space.var_names)
    )
    y_col = _quote_ident(get_lon_landscape(LANDSCAPE_Y_RAW).source_column)
    query = (
        f"SELECT {idx_selects}, CAST({y_col} AS DOUBLE) AS {_quote_ident('value')}, "
        f"{analysis_local_phi_f_sql(names)} FROM {table}"
    )
    try:
        result = connection.execute(query)
    except Exception as exc:
        raise ElaError(f"could not read catalogue y/phi/F: {exc}") from exc
    fetch_numpy = getattr(result, "fetchnumpy", None)
    if callable(fetch_numpy):
        arrays = fetch_numpy()
        idx_cols = [_as_int64_vector(arrays[f"i{dim}"]) for dim in range(n_dim)]
        y_values = _as_float64_vector(arrays["value"])
        phi_values = _as_float64_vector(arrays["phi"])
        feasible_values = _as_bool_vector(arrays["F"])
    else:
        rows = result.fetchall()
        if len(rows) != expected_n:
            raise ElaError(
                "catalogue row count "
                f"({len(rows)}) does not match cardinality ({expected_n})"
            )
        payload = np.asarray(rows, dtype=object)
        idx_cols = [
            np.asarray(payload[:, dim], dtype=np.int64) for dim in range(n_dim)
        ]
        y_values = np.asarray(payload[:, n_dim], dtype=np.float64)
        phi_values = np.asarray(payload[:, n_dim + 1], dtype=np.float64)
        raw_feasible = payload[:, n_dim + 2]
        if any(value is None for value in raw_feasible):
            raise ElaError("catalogue analysis-local F contains NULL")
        feasible_values = np.asarray(raw_feasible, dtype=np.bool_)
    linears = _catalogue_linears(
        idx_cols, y_values, space=space, expected_n=expected_n
    )
    y_raw = np.empty(expected_n, dtype=np.float64)
    y_raw[linears] = y_values
    phi = np.empty(expected_n, dtype=np.float64)
    phi[linears] = phi_values
    flags = np.empty(expected_n, dtype=np.bool_)
    flags[linears] = feasible_values
    return y_raw, phi, flags


def _catalogue_linears(
    idx_cols: Sequence[NDArray[np.int64]],
    y_values: NDArray[np.float64],
    *,
    space: IndexSpace,
    expected_n: int,
) -> NDArray[np.int64]:
    n_rows = int(y_values.size)
    if n_rows != expected_n:
        raise ElaError(
            "catalogue row count "
            f"({n_rows}) does not match cardinality ({expected_n})"
        )
    linears = np.zeros(n_rows, dtype=np.int64)
    for col, lower, upper, stride in zip(
        idx_cols,
        space.lower_bounds,
        space.upper_bounds,
        space.strides,
        strict=True,
    ):
        if int(col.size) != n_rows:
            raise ElaError("catalogue index column length does not match landscape")
        lower_i = int(lower)
        upper_i = int(upper)
        if bool(np.any(col < lower_i)) or bool(np.any(col > upper_i)):
            raise ElaError("catalogue index vector is outside IndexSpace bounds")
        linears += (col - lower_i) * int(stride)
    if n_rows and (int(linears.min()) < 0 or int(linears.max()) >= expected_n):
        raise ElaError("catalogue linear address is outside cardinality")
    counts = np.bincount(linears, minlength=expected_n)
    if int(counts.size) != expected_n or not bool(np.all(counts == 1)):
        if bool(np.any(counts > 1)):
            raise ElaError("duplicate catalogue index vector")
        raise ElaError("catalogue is missing at least one index vector")
    return linears


def _scatter_catalogue(
    idx_cols: Sequence[NDArray[np.int64]],
    y_values: NDArray[np.float64],
    feasible_values: NDArray[np.bool_] | None,
    *,
    space: IndexSpace,
    expected_n: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_] | None]:
    linears = _catalogue_linears(
        idx_cols, y_values, space=space, expected_n=expected_n
    )
    values = np.empty(expected_n, dtype=np.float64)
    values[linears] = y_values
    if feasible_values is None:
        flags = None
    else:
        if int(feasible_values.size) != int(y_values.size):
            raise ElaError("feasible length does not match landscape")
        flags = np.empty(expected_n, dtype=np.bool_)
        flags[linears] = feasible_values
    return values, flags


def load_catalogue_utils(
    connection: Any,
    table: str,
    space: IndexSpace,
    constraint_names: Sequence[str],
    *,
    expected_n: int,
) -> NDArray[np.float64]:
    """Scatter selected ``util__*`` columns into catalogue order."""
    names = tuple(constraint_names)
    if not names:
        raise ElaError("selected util load requires at least one constraint")
    n_dim = len(space.var_names)
    idx_selects = ", ".join(
        f"CAST({_quote_ident(f'idx__{name}')} AS BIGINT) AS "
        f"{_quote_ident(f'i{dim}')}"
        for dim, name in enumerate(space.var_names)
    )
    util_selects = ", ".join(
        f"CAST({_quote_ident(f'util__{name}')} AS DOUBLE) AS "
        f"{_quote_ident(f'u{col}')}"
        for col, name in enumerate(names)
    )
    query = f"SELECT {idx_selects}, {util_selects} FROM {table}"
    try:
        result = connection.execute(query)
    except Exception as exc:
        raise ElaError(f"could not read catalogue utils: {exc}") from exc
    fetch_numpy = getattr(result, "fetchnumpy", None)
    if callable(fetch_numpy):
        arrays = fetch_numpy()
        idx_cols = [_as_int64_vector(arrays[f"i{dim}"]) for dim in range(n_dim)]
        util_values = np.column_stack(
            [_as_float64_vector(arrays[f"u{col}"]) for col in range(len(names))]
        )
    else:
        rows = result.fetchall()
        if len(rows) != expected_n:
            raise ElaError(
                "catalogue row count "
                f"({len(rows)}) does not match cardinality ({expected_n})"
            )
        payload = np.asarray(rows, dtype=object)
        idx_cols = [
            np.asarray(payload[:, dim], dtype=np.int64) for dim in range(n_dim)
        ]
        util_values = np.asarray(payload[:, n_dim:], dtype=np.float64)
    dummy = np.zeros(int(util_values.shape[0]), dtype=np.float64)
    linears = _catalogue_linears(
        idx_cols, dummy, space=space, expected_n=expected_n
    )
    ordered = np.empty((expected_n, len(names)), dtype=np.float64)
    ordered[linears] = util_values
    return ordered


def _as_int64_vector(raw: object) -> NDArray[np.int64]:
    if np.ma.isMaskedArray(raw):
        filled = np.ma.filled(raw, 0)
        return np.asarray(filled, dtype=np.int64).reshape(-1)
    return np.asarray(raw, dtype=np.int64).reshape(-1)


def _as_float64_vector(raw: object) -> NDArray[np.float64]:
    if np.ma.isMaskedArray(raw):
        filled = np.ma.filled(raw, np.nan)
        return np.asarray(filled, dtype=np.float64).reshape(-1)
    return np.asarray(raw, dtype=np.float64).reshape(-1)


def _as_bool_vector(raw: object) -> NDArray[np.bool_]:
    if np.ma.isMaskedArray(raw):
        if bool(np.any(np.ma.getmaskarray(raw))):
            raise ElaError("catalogue feasible column contains NULL")
        return np.asarray(np.ma.getdata(raw), dtype=np.bool_).reshape(-1)
    return np.asarray(raw, dtype=np.bool_).reshape(-1)


def _has_column(connection: Any, table: str, name: str) -> bool:
    try:
        connection.execute(f"SELECT {_quote_ident(name)} FROM {table} LIMIT 0")
    except Exception:
        return False
    return True


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
