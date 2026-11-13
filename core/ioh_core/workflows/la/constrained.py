"""
Constrained-landscape ratios, Spearman, IZ, and structure metrics.

Same formulae on a complete brute-force catalogue or one sampling
repetition; ``source_type`` disambiguates. Not pflacco and not ``z̃``.
DuckDB reuses :func:`constraint_violation_sql`; stored ``feasible`` is
not read. CLI wiring lives in ``orchestrate.py`` /
``hdb-la-analyze constraints``.

``FsR = #{x : phi(x)=0} / cardinality``. ``FsR_i = P(v_i = 0)`` for every
selected constraint, including constants (``FsR_i = 1``).
``FVC = Spearman(y_raw, phi)``. ``rho_i = Spearman(y_raw, v_i)``. Spearman
is SciPy ``spearmanr`` (average ranks), not DuckDB ``corr`` / ``RANK``.
Constant either side → ``status=undefined``, ``value=null``.
``25_IZ`` / ``4_IZ`` (ideal-zone overlap; Malan, Oberholzer &
Engelbrecht 2015 CEC) are ``|intersection|/n`` with strict ``<`` vs
linear quantiles 0.5 and 0.2. Per-``v_i`` distributions, pairwise Spearman,
``M_ij = P(v_i>0 AND v_j>0)``, and diagnostic ``C_i`` are tidy rows only
(no sidecar files in this tick). BF: observed ``COUNT(*)`` must equal
sealed catalogue cardinality. Sampling: ``n_input == n_unique ==`` design
``N`` (pass ``cardinality`` = observed design size).

Importing this module must not load DuckDB, SciPy, pflacco, or SALib.
"""
from __future__ import annotations

import importlib.metadata
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from core.ioh_core.workflows.la.deps import import_duckdb
from core.ioh_core.workflows.la.models import (
    CONSTRAINT_LANDSCAPE_PREFIX,
    ElaError,
    FeatureResult,
    LANDSCAPE_FEASIBILITY,
    LANDSCAPE_PHI,
    RESPONSE_VIEW_NA,
    ROLE_DIAGNOSTIC,
    ROLE_PRIMARY,
    SOURCE_BRUTE_FORCE,
    SOURCE_SAMPLING,
    SOURCE_TYPE_VALUES,
    STATUS_OK,
    STATUS_UNDEFINED,
    X_VIEW_NA,
    feature_result,
)
from core.ioh_core.workflows.la.preprocess import constraint_violation_sql

FAMILY = "constrained"
LIBRARY = "duckdb"
LIBRARY_SCIPY = "scipy"
FEATURE_FSR = "constrained.FsR"
FEATURE_FVC = "constrained.FVC"
FEATURE_25_IZ = "constrained.25_IZ"
FEATURE_4_IZ = "constrained.4_IZ"
SPEARMAN_METHOD = "scipy.stats.spearmanr"
RANK_METHOD = "average"
QUANTILE_METHOD = "linear"
IZ_Y_QUANTILES = (0.5, 0.2)
V_DISTRIBUTION_QUANTILES = (0.9, 0.95, 0.99)
_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_constrained_source_type(source_type: str, purpose: str) -> str:
    name = str(source_type)
    if name not in SOURCE_TYPE_VALUES:
        raise ElaError(
            f"constrained {purpose} requires source_type in "
            f"{sorted(SOURCE_TYPE_VALUES)}, got {source_type!r}"
        )
    return name


def _assert_constrained_counts(
    source_type: str,
    cardinality: int,
    n_input: int,
    n_unique: int,
    var_names: Sequence[str] | None,
    purpose: str,
) -> None:
    if source_type == SOURCE_BRUTE_FORCE:
        if n_input != cardinality:
            raise ElaError(
                f"constrained {purpose} requires observed rows to equal catalogue "
                f"cardinality ({cardinality}), got {n_input}"
            )
        if var_names is not None and n_unique != cardinality:
            raise ElaError(
                f"constrained {purpose} requires unique complete designs equal to "
                f"cardinality ({cardinality}), got n_unique={n_unique}"
            )
        return
    if source_type == SOURCE_SAMPLING:
        if n_input != cardinality:
            raise ElaError(
                f"constrained {purpose} requires observed rows to equal design N "
                f"({cardinality}), got {n_input}"
            )
        if n_unique != cardinality:
            raise ElaError(
                f"constrained {purpose} requires unique designs equal to design N "
                f"({cardinality}), got n_unique={n_unique}"
            )
        return
    raise ElaError(
        f"constrained {purpose} requires source_type in "
        f"{sorted(SOURCE_TYPE_VALUES)}, got {source_type!r}"
    )


def _require_constraint_name(constraint_name: str) -> str:
    if not isinstance(constraint_name, str) or not constraint_name.strip():
        raise ElaError(f"constraint name must be non-empty, got {constraint_name!r}")
    return constraint_name


def fsr_i_feature_name(constraint_name: str) -> str:
    """Return the tidy key for one constraint’s feasible-share."""
    return f"constrained.FsR_i.{_require_constraint_name(constraint_name)}"


def rho_i_feature_name(constraint_name: str) -> str:
    """Return the tidy key for one constraint’s Spearman with ``y_raw``."""
    return f"constrained.rho_i.{_require_constraint_name(constraint_name)}"


def v_stat_feature_name(stat: str, constraint_name: str) -> str:
    """Return the tidy key for one per-constraint distribution statistic."""
    if not isinstance(stat, str) or not stat.strip():
        raise ElaError(f"distribution stat must be non-empty, got {stat!r}")
    return f"constrained.v.{stat}.{_require_constraint_name(constraint_name)}"


def v_quantile_feature_name(quantile: float, constraint_name: str) -> str:
    """Return the tidy key for one configured ``v_i`` quantile."""
    if not isinstance(quantile, float) or isinstance(quantile, bool):
        raise ElaError(f"quantile must be a float, got {quantile!r}")
    percent = int(round(quantile * 100.0))
    return f"constrained.v.q_{percent}.{_require_constraint_name(constraint_name)}"


def rho_ij_feature_name(left: str, right: str) -> str:
    """Return the tidy key for Spearman(``v_left``, ``v_right``)."""
    return (
        f"constrained.rho_ij.{_require_constraint_name(left)}."
        f"{_require_constraint_name(right)}"
    )


def m_ij_feature_name(left: str, right: str) -> str:
    """Return the tidy key for co-occurrence ``M_ij``."""
    return (
        f"constrained.M_ij.{_require_constraint_name(left)}."
        f"{_require_constraint_name(right)}"
    )


def c_i_feature_name(constraint_name: str) -> str:
    """Return the tidy key for diagnostic contribution ``C_i``."""
    return f"constrained.C_i.{_require_constraint_name(constraint_name)}"


def ela_constrained(
    *,
    connection: Any,
    table: str,
    constraint_names: Sequence[str],
    cardinality: int,
    problem_id: str,
    source_dir: Path,
    var_names: Sequence[str] | None = None,
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
    analysis_signature: str | None = None,
    seed: int | None = None,
    repetition: int | None = None,
) -> list[FeatureResult]:
    """
    Compute FsR and per-constraint FsR_i from a complete BF or sampling table.

    Parameters
    ----------
    connection :
        Open DuckDB connection. This function calls :func:`import_duckdb`
        only to record the library version; it does not create the
        connection.
    table :
        Bare table name in ``connection`` (letters, digits, underscore).
    constraint_names :
        Selected keep-list in sealed relative order. Used to build
        analysis-local ``v_i`` / ``phi``.
    cardinality :
        BF: sealed catalogue size. Sampling: observed design ``N``.
        Must equal ``COUNT(*)``.
    var_names :
        Optional sealed variable names. When given, ``n_unique`` is
        ``COUNT(DISTINCT idx__*)`` and must also equal ``cardinality``
        (always required for sampling).
    problem_id, source_dir :
        Identity fields forwarded to :func:`feature_result`.
    source_type :
        ``brute_force`` or ``sampling``.
    repetition :
        Sampling repetition id; ``None`` for brute-force.

    Returns
    -------
    list of FeatureResult
        ``constrained.FsR`` then one ``constrained.FsR_i.<stem>`` per
        selected constraint.
    """
    names = tuple(constraint_names)
    if not names:
        raise ElaError("constrained FsR requires at least one selected constraint")
    source_type = _require_constrained_source_type(source_type, "FsR")
    if not isinstance(cardinality, int) or isinstance(cardinality, bool) or cardinality < 1:
        raise ElaError(f"cardinality must be a positive int, got {cardinality!r}")
    table_sql = _quote_table(table)
    violation_sql = constraint_violation_sql(names)
    idx_sql = _idx_select(var_names)
    inner = violation_sql if not idx_sql else f"{idx_sql}, {violation_sql}"
    fsr_i_selects = ", ".join(
        "AVG(CASE WHEN "
        f"{_quote_ident(f'v__{name}')} = 0.0 THEN 1.0 ELSE 0.0 END) "
        f"AS {_quote_ident(f'fsr_i__{name}')}"
        for name in names
    )
    unique_sql = (
        f"COUNT(DISTINCT ({_idx_tuple(var_names)})) AS n_unique"
        if var_names
        else "COUNT(*) AS n_unique"
    )
    query = (
        f"SELECT COUNT(*) AS n_input, {unique_sql}, "
        f'SUM(CASE WHEN "F" THEN 1 ELSE 0 END) AS n_feasible, '
        f"{fsr_i_selects} "
        f"FROM (SELECT {inner} FROM {table_sql})"
    )
    import_duckdb()
    started = time.monotonic()
    row = connection.execute(query).fetchone()
    runtime_s = max(0.0, time.monotonic() - started)
    if row is None:
        raise ElaError("constrained FsR query returned no row")
    n_input = int(row[0])
    n_unique = int(row[1])
    n_feasible = int(row[2] or 0)
    _assert_constrained_counts(
        source_type,
        cardinality,
        n_input,
        n_unique,
        var_names,
        "FsR",
    )
    fsr = float(n_feasible) / float(cardinality)
    fsr_i_values = [float(row[3 + index]) for index in range(len(names))]
    config_payload: dict[str, Any] = {"constraint_names": list(names)}
    identity = {
        "problem_id": problem_id,
        "source_type": source_type,
        "source_dir": Path(source_dir),
        "n_input": n_input,
        "n_unique": n_unique,
        "runtime_s": runtime_s,
        "config_payload": config_payload,
        "sampling_design_signature": sampling_design_signature,
        "analysis_signature": analysis_signature,
        "seed": seed,
        "repetition": repetition,
    }
    rows = [
        _row(
            identity,
            landscape=LANDSCAPE_FEASIBILITY,
            feature_name=FEATURE_FSR,
            value=fsr,
        )
    ]
    for name, value in zip(names, fsr_i_values, strict=True):
        rows.append(
            _row(
                identity,
                landscape=f"{CONSTRAINT_LANDSCAPE_PREFIX}{name}",
                feature_name=fsr_i_feature_name(name),
                value=value,
            )
        )
    return rows


def ela_constrained_spearman(
    *,
    connection: Any,
    table: str,
    constraint_names: Sequence[str],
    cardinality: int,
    problem_id: str,
    source_dir: Path,
    var_names: Sequence[str] | None = None,
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
    analysis_signature: str | None = None,
    seed: int | None = None,
    repetition: int | None = None,
) -> list[FeatureResult]:
    """
    Compute FVC and per-constraint ρ_i from a complete BF or sampling table.

    DuckDB projects ``y``, analysis-local ``phi``, and ``v_i``. Spearman is
    ``scipy.stats.spearmanr`` (average ranks). Constant ``y``, ``phi``, or
    ``v_i`` yields ``status=undefined`` and a null value.

    Parameters
    ----------
    connection :
        Open DuckDB connection. :func:`import_duckdb` is called only to
        ensure the extra is present; SciPy is imported on this path only.
    table :
        Bare table name in ``connection`` (letters, digits, underscore).
    constraint_names :
        Selected keep-list in sealed relative order.
    cardinality :
        BF: sealed catalogue size. Sampling: observed design ``N``.
        Must equal ``COUNT(*)``.
    var_names :
        Optional sealed variable names. When given, ``n_unique`` is
        ``COUNT(DISTINCT idx__*)`` and must also equal ``cardinality``
        (always required for sampling).
    problem_id, source_dir :
        Identity fields forwarded to :func:`feature_result`.
    source_type :
        ``brute_force`` or ``sampling``.
    repetition :
        Sampling repetition id; ``None`` for brute-force.

    Returns
    -------
    list of FeatureResult
        ``constrained.FVC`` then one ``constrained.rho_i.<stem>`` per
        selected constraint.
    """
    names = tuple(constraint_names)
    if not names:
        raise ElaError("constrained FVC requires at least one selected constraint")
    source_type = _require_constrained_source_type(source_type, "FVC")
    if not isinstance(cardinality, int) or isinstance(cardinality, bool) or cardinality < 1:
        raise ElaError(f"cardinality must be a positive int, got {cardinality!r}")
    table_sql = _quote_table(table)
    import_duckdb()
    started = time.monotonic()
    n_input, n_unique = _catalogue_counts(connection, table_sql, var_names)
    _assert_constrained_counts(
        source_type,
        cardinality,
        n_input,
        n_unique,
        var_names,
        "FVC",
    )
    y, phi, violations = _project_spearman_columns(connection, table_sql, names)
    fvc = _spearman_or_undefined(y, phi)
    rho_values = [
        _spearman_or_undefined(y, violations[:, index]) for index in range(len(names))
    ]
    runtime_s = max(0.0, time.monotonic() - started)
    config_payload: dict[str, Any] = {
        "constraint_names": list(names),
        "spearman": SPEARMAN_METHOD,
        "rank_method": RANK_METHOD,
    }
    identity = {
        "problem_id": problem_id,
        "source_type": source_type,
        "source_dir": Path(source_dir),
        "n_input": n_input,
        "n_unique": n_unique,
        "runtime_s": runtime_s,
        "config_payload": config_payload,
        "sampling_design_signature": sampling_design_signature,
        "analysis_signature": analysis_signature,
        "seed": seed,
        "repetition": repetition,
        "library": LIBRARY_SCIPY,
        "library_version": _scipy_version(),
    }
    rows = [
        _row(
            identity,
            landscape=LANDSCAPE_PHI,
            feature_name=FEATURE_FVC,
            value=fvc,
            status=STATUS_OK if fvc is not None else STATUS_UNDEFINED,
        )
    ]
    for name, value in zip(names, rho_values, strict=True):
        rows.append(
            _row(
                identity,
                landscape=f"{CONSTRAINT_LANDSCAPE_PREFIX}{name}",
                feature_name=rho_i_feature_name(name),
                value=value,
                status=STATUS_OK if value is not None else STATUS_UNDEFINED,
            )
        )
    return rows


def ela_constrained_iz(
    *,
    connection: Any,
    table: str,
    constraint_names: Sequence[str],
    cardinality: int,
    problem_id: str,
    source_dir: Path,
    var_names: Sequence[str] | None = None,
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
    analysis_signature: str | None = None,
    seed: int | None = None,
    repetition: int | None = None,
) -> list[FeatureResult]:
    """
    Compute 25_IZ and 4_IZ from a complete BF or sampling table.

    DuckDB ``quantile_cont`` (linear, matching ``np.quantile`` default)
    then ``AVG`` of the strict-``<`` intersection. Scalar is
    ``|intersection| / n``, not divided by 0.25 or 0.04.

    Parameters
    ----------
    connection :
        Open DuckDB connection. :func:`import_duckdb` is called only to
        ensure the extra is present.
    table :
        Bare table name in ``connection`` (letters, digits, underscore).
    constraint_names :
        Selected keep-list in sealed relative order.
    cardinality :
        BF: sealed catalogue size. Sampling: observed design ``N``.
        Must equal ``COUNT(*)``.
    var_names :
        Optional sealed variable names. When given, ``n_unique`` is
        ``COUNT(DISTINCT idx__*)`` and must also equal ``cardinality``
        (always required for sampling).
    problem_id, source_dir :
        Identity fields forwarded to :func:`feature_result`.
    source_type :
        ``brute_force`` or ``sampling``.
    repetition :
        Sampling repetition id; ``None`` for brute-force.

    Returns
    -------
    list of FeatureResult
        ``constrained.25_IZ`` then ``constrained.4_IZ``.
    """
    names = tuple(constraint_names)
    if not names:
        raise ElaError("constrained IZ requires at least one selected constraint")
    source_type = _require_constrained_source_type(source_type, "IZ")
    if not isinstance(cardinality, int) or isinstance(cardinality, bool) or cardinality < 1:
        raise ElaError(f"cardinality must be a positive int, got {cardinality!r}")
    table_sql = _quote_table(table)
    import_duckdb()
    started = time.monotonic()
    n_input, n_unique = _catalogue_counts(connection, table_sql, var_names)
    _assert_constrained_counts(
        source_type,
        cardinality,
        n_input,
        n_unique,
        var_names,
        "IZ",
    )
    landscape_sql = _landscape_sql(table_sql, names)
    q50, q20 = IZ_Y_QUANTILES
    quantiles = connection.execute(
        f"SELECT quantile_cont(y, {q50}), quantile_cont(phi, {q50}), "
        f"quantile_cont(y, {q20}), quantile_cont(phi, {q20}) "
        f"FROM ({landscape_sql})"
    ).fetchone()
    if quantiles is None or any(value is None for value in quantiles):
        raise ElaError("constrained IZ quantile query returned no values")
    q50_y, q50_phi, q20_y, q20_phi = (float(value) for value in quantiles)
    iz_row = connection.execute(
        f"SELECT "
        f"AVG(CASE WHEN y < ? AND phi < ? THEN 1.0 ELSE 0.0 END), "
        f"AVG(CASE WHEN y < ? AND phi < ? THEN 1.0 ELSE 0.0 END) "
        f"FROM ({landscape_sql})",
        [q50_y, q50_phi, q20_y, q20_phi],
    ).fetchone()
    runtime_s = max(0.0, time.monotonic() - started)
    if iz_row is None or iz_row[0] is None or iz_row[1] is None:
        raise ElaError("constrained IZ intersection query returned no values")
    iz_25 = float(iz_row[0])
    iz_4 = float(iz_row[1])
    config_payload: dict[str, Any] = {
        "constraint_names": list(names),
        "y_quantile": [q50, q20],
        "quantile_method": QUANTILE_METHOD,
    }
    identity = {
        "problem_id": problem_id,
        "source_type": source_type,
        "source_dir": Path(source_dir),
        "n_input": n_input,
        "n_unique": n_unique,
        "runtime_s": runtime_s,
        "config_payload": config_payload,
        "sampling_design_signature": sampling_design_signature,
        "analysis_signature": analysis_signature,
        "seed": seed,
        "repetition": repetition,
    }
    return [
        _row(
            identity,
            landscape=LANDSCAPE_PHI,
            feature_name=FEATURE_25_IZ,
            value=iz_25,
        ),
        _row(
            identity,
            landscape=LANDSCAPE_PHI,
            feature_name=FEATURE_4_IZ,
            value=iz_4,
        ),
    ]


def ela_constrained_distributions(
    *,
    connection: Any,
    table: str,
    constraint_names: Sequence[str],
    cardinality: int,
    problem_id: str,
    source_dir: Path,
    var_names: Sequence[str] | None = None,
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
    analysis_signature: str | None = None,
    seed: int | None = None,
    repetition: int | None = None,
) -> list[FeatureResult]:
    """
    Compute per-constraint ``v_i`` distribution summaries.

    DuckDB aggregates: min, max, mean, median (``quantile_cont`` 0.5),
    configured tail quantiles, and fraction zero. No plots.

    One constraint per query so ``quantile_cont`` never materialises all
    ``k`` ``v_i`` columns at once. Values match the former wide query.

    Returns
    -------
    list of FeatureResult
        For each selected constraint, in sealed order: min, max, mean,
        median, ``q_90`` / ``q_95`` / ``q_99``, then ``frac_zero``.
    """
    names, table_sql, n_input, n_unique = _prepare_complete_catalogue(
        connection=connection,
        table=table,
        constraint_names=constraint_names,
        cardinality=cardinality,
        source_type=source_type,
        var_names=var_names,
        purpose="distributions",
    )
    started = time.monotonic()
    keys: list[tuple[str, str]] = []
    values: list[float] = []
    for name in names:
        inner = f"SELECT {constraint_violation_sql((name,))} FROM {table_sql}"
        v_sql = _quote_ident(f"v__{name}")
        landscape = f"{CONSTRAINT_LANDSCAPE_PREFIX}{name}"
        selects = [
            f"MIN({v_sql})",
            f"MAX({v_sql})",
            f"AVG({v_sql})",
            f"quantile_cont({v_sql}, 0.5)",
        ]
        keys.extend(
            [
                (v_stat_feature_name("min", name), landscape),
                (v_stat_feature_name("max", name), landscape),
                (v_stat_feature_name("mean", name), landscape),
                (v_stat_feature_name("median", name), landscape),
            ]
        )
        for quantile in V_DISTRIBUTION_QUANTILES:
            selects.append(f"quantile_cont({v_sql}, {quantile})")
            keys.append((v_quantile_feature_name(quantile, name), landscape))
        selects.append(
            f"AVG(CASE WHEN {v_sql} = 0.0 THEN 1.0 ELSE 0.0 END)"
        )
        keys.append((v_stat_feature_name("frac_zero", name), landscape))
        row = connection.execute(
            f"SELECT {', '.join(selects)} FROM ({inner})"
        ).fetchone()
        if row is None or any(value is None for value in row):
            raise ElaError(
                "constrained distributions query returned no values "
                f"for {name}"
            )
        values.extend(float(value) for value in row)
    runtime_s = max(0.0, time.monotonic() - started)
    config_payload: dict[str, Any] = {
        "constraint_names": list(names),
        "v_quantile": list(V_DISTRIBUTION_QUANTILES),
        "quantile_method": QUANTILE_METHOD,
    }
    identity = _identity(
        problem_id=problem_id,
        source_type=source_type,
        source_dir=source_dir,
        n_input=n_input,
        n_unique=n_unique,
        runtime_s=runtime_s,
        config_payload=config_payload,
        sampling_design_signature=sampling_design_signature,
        analysis_signature=analysis_signature,
        seed=seed,
        repetition=repetition,
    )
    return [
        _row(
            identity,
            landscape=landscape,
            feature_name=feature_name,
            value=value,
        )
        for (feature_name, landscape), value in zip(keys, values, strict=True)
    ]


def ela_constrained_pairwise(
    *,
    connection: Any,
    table: str,
    constraint_names: Sequence[str],
    cardinality: int,
    problem_id: str,
    source_dir: Path,
    var_names: Sequence[str] | None = None,
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
    analysis_signature: str | None = None,
    seed: int | None = None,
    repetition: int | None = None,
) -> list[FeatureResult]:
    """
    Compute pairwise Spearman among selected ``v_i``.

    Full ``k×k`` including the diagonal. Average-rank SciPy ``spearmanr``,
    **parallel to** :func:`ela_constrained_spearman`. Constant either side
    → ``status=undefined``, ``value=null``.
    """
    names, table_sql, n_input, n_unique = _prepare_complete_catalogue(
        connection=connection,
        table=table,
        constraint_names=constraint_names,
        cardinality=cardinality,
        source_type=source_type,
        var_names=var_names,
        purpose="pairwise Spearman",
    )
    started = time.monotonic()
    _y, _phi, violations = _project_spearman_columns(connection, table_sql, names)
    runtime_s = max(0.0, time.monotonic() - started)
    config_payload: dict[str, Any] = {
        "constraint_names": list(names),
        "spearman": SPEARMAN_METHOD,
        "rank_method": RANK_METHOD,
    }
    identity = _identity(
        problem_id=problem_id,
        source_type=source_type,
        source_dir=source_dir,
        n_input=n_input,
        n_unique=n_unique,
        runtime_s=runtime_s,
        config_payload=config_payload,
        sampling_design_signature=sampling_design_signature,
        analysis_signature=analysis_signature,
        seed=seed,
        repetition=repetition,
        library=LIBRARY_SCIPY,
        library_version=_scipy_version(),
    )
    rows: list[FeatureResult] = []
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            value = _spearman_or_undefined(violations[:, i], violations[:, j])
            rows.append(
                _row(
                    identity,
                    landscape=f"{CONSTRAINT_LANDSCAPE_PREFIX}{left}",
                    feature_name=rho_ij_feature_name(left, right),
                    value=value,
                    status=STATUS_OK if value is not None else STATUS_UNDEFINED,
                )
            )
    return rows


def ela_constrained_cooccurrence(
    *,
    connection: Any,
    table: str,
    constraint_names: Sequence[str],
    cardinality: int,
    problem_id: str,
    source_dir: Path,
    var_names: Sequence[str] | None = None,
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
    analysis_signature: str | None = None,
    seed: int | None = None,
    repetition: int | None = None,
) -> list[FeatureResult]:
    """
    Compute ``M_ij = P(v_i > 0 AND v_j > 0)``.

    Full ``k×k``, symmetric, diagonal ``P(v_i > 0)``. Not BF variable-level
    co-occurrence and not conditional probabilities.
    """
    names, table_sql, n_input, n_unique = _prepare_complete_catalogue(
        connection=connection,
        table=table,
        constraint_names=constraint_names,
        cardinality=cardinality,
        source_type=source_type,
        var_names=var_names,
        purpose="co-occurrence",
    )
    inner = f"SELECT {constraint_violation_sql(names)} FROM {table_sql}"
    selects: list[str] = []
    keys: list[tuple[str, str]] = []
    for left in names:
        left_sql = _quote_ident(f"v__{left}")
        landscape = f"{CONSTRAINT_LANDSCAPE_PREFIX}{left}"
        for right in names:
            right_sql = _quote_ident(f"v__{right}")
            selects.append(
                "AVG(CASE WHEN "
                f"{left_sql} > 0.0 AND {right_sql} > 0.0 THEN 1.0 ELSE 0.0 END)"
            )
            keys.append((m_ij_feature_name(left, right), landscape))
    started = time.monotonic()
    row = connection.execute(
        f"SELECT {', '.join(selects)} FROM ({inner})"
    ).fetchone()
    runtime_s = max(0.0, time.monotonic() - started)
    if row is None or any(value is None for value in row):
        raise ElaError("constrained co-occurrence query returned no values")
    identity = _identity(
        problem_id=problem_id,
        source_type=source_type,
        source_dir=source_dir,
        n_input=n_input,
        n_unique=n_unique,
        runtime_s=runtime_s,
        config_payload={"constraint_names": list(names)},
        sampling_design_signature=sampling_design_signature,
        analysis_signature=analysis_signature,
        seed=seed,
        repetition=repetition,
    )
    return [
        _row(
            identity,
            landscape=landscape,
            feature_name=feature_name,
            value=float(value),
        )
        for (feature_name, landscape), value in zip(keys, row, strict=True)
    ]


def ela_constrained_contribution(
    *,
    connection: Any,
    table: str,
    constraint_names: Sequence[str],
    cardinality: int,
    problem_id: str,
    source_dir: Path,
    var_names: Sequence[str] | None = None,
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
    analysis_signature: str | None = None,
    seed: int | None = None,
    repetition: int | None = None,
) -> list[FeatureResult]:
    """
    Compute diagnostic ``C_i = sum_x v_i(x) / sum_x phi(x)``.

    Denominator 0 (all selected ``v_i`` zero) → ``status=undefined``.
    """
    names, table_sql, n_input, n_unique = _prepare_complete_catalogue(
        connection=connection,
        table=table,
        constraint_names=constraint_names,
        cardinality=cardinality,
        source_type=source_type,
        var_names=var_names,
        purpose="contribution",
    )
    inner = f"SELECT {constraint_violation_sql(names)} FROM {table_sql}"
    sum_v = ", ".join(
        f"SUM({_quote_ident(f'v__{name}')})" for name in names
    )
    started = time.monotonic()
    row = connection.execute(
        f'SELECT {sum_v}, SUM({_quote_ident("phi")}) FROM ({inner})'
    ).fetchone()
    runtime_s = max(0.0, time.monotonic() - started)
    if row is None or row[-1] is None:
        raise ElaError("constrained contribution query returned no values")
    denom = float(row[-1])
    identity = _identity(
        problem_id=problem_id,
        source_type=source_type,
        source_dir=source_dir,
        n_input=n_input,
        n_unique=n_unique,
        runtime_s=runtime_s,
        config_payload={"constraint_names": list(names)},
        sampling_design_signature=sampling_design_signature,
        analysis_signature=analysis_signature,
        seed=seed,
        repetition=repetition,
    )
    rows: list[FeatureResult] = []
    for index, name in enumerate(names):
        raw = row[index]
        if denom == 0.0 or raw is None:
            value = None
            status = STATUS_UNDEFINED
        else:
            value = float(raw) / denom
            status = STATUS_OK
        rows.append(
            _row(
                identity,
                landscape=f"{CONSTRAINT_LANDSCAPE_PREFIX}{name}",
                feature_name=c_i_feature_name(name),
                value=value,
                status=status,
                role=ROLE_DIAGNOSTIC,
            )
        )
    return rows


def _quote_table(table: str) -> str:
    if not isinstance(table, str) or not _TABLE_NAME.match(table):
        raise ElaError(f"table must be a simple identifier, got {table!r}")
    return _quote_ident(table)


def _quote_ident(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ElaError(f"identifier must be a non-empty string, got {name!r}")
    return '"' + name.replace('"', '""') + '"'


def _idx_select(var_names: Sequence[str] | None) -> str:
    if not var_names:
        return ""
    return ", ".join(_quote_ident(f"idx__{name}") for name in var_names)


def _idx_tuple(var_names: Sequence[str]) -> str:
    cols = [_quote_ident(f"idx__{name}") for name in var_names]
    if len(cols) == 1:
        return cols[0]
    return ", ".join(cols)


def _library_version() -> str:
    try:
        return importlib.metadata.version("duckdb")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _scipy_version() -> str:
    try:
        return importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _catalogue_counts(
    connection: Any,
    table_sql: str,
    var_names: Sequence[str] | None,
) -> tuple[int, int]:
    unique_sql = (
        f"COUNT(DISTINCT ({_idx_tuple(var_names)})) AS n_unique"
        if var_names
        else "COUNT(*) AS n_unique"
    )
    row = connection.execute(
        f"SELECT COUNT(*) AS n_input, {unique_sql} FROM {table_sql}"
    ).fetchone()
    if row is None:
        raise ElaError("constrained catalogue count query returned no row")
    return int(row[0]), int(row[1])


def _project_spearman_columns(
    connection: Any,
    table_sql: str,
    names: Sequence[str],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    y_sql = _quote_ident("y")
    v_selects = ", ".join(_quote_ident(f"v__{name}") for name in names)
    violation_sql = constraint_violation_sql(names)
    query = (
        f"SELECT CAST({y_sql} AS DOUBLE) AS y, "
        f'{_quote_ident("phi")} AS phi, {v_selects} '
        f"FROM (SELECT {y_sql}, {violation_sql} FROM {table_sql})"
    )
    result = connection.execute(query)
    fetch_numpy = getattr(result, "fetchnumpy", None)
    if callable(fetch_numpy):
        arrays = fetch_numpy()
        y = _as_float_vector(arrays["y"])
        phi = _as_float_vector(arrays["phi"])
        columns = [_as_float_vector(arrays[f"v__{name}"]) for name in names]
    else:
        rows = result.fetchall()
        if not rows:
            y = np.empty(0, dtype=float)
            phi = np.empty(0, dtype=float)
            columns = [np.empty(0, dtype=float) for _ in names]
        else:
            payload = np.asarray(rows, dtype=float)
            y = payload[:, 0]
            phi = payload[:, 1]
            columns = [payload[:, 2 + index] for index in range(len(names))]
    if y.shape[0] != phi.shape[0]:
        raise ElaError("constrained FVC projection row counts do not match")
    violations = np.column_stack(columns) if columns else np.empty((y.shape[0], 0))
    if violations.shape[0] != y.shape[0] or violations.shape[1] != len(names):
        raise ElaError("constrained FVC violation projection shape is inconsistent")
    return y, phi, violations


def _landscape_sql(table_sql: str, names: Sequence[str]) -> str:
    y_sql = _quote_ident("y")
    violation_sql = constraint_violation_sql(names)
    return (
        f"SELECT CAST({y_sql} AS DOUBLE) AS y, "
        f'{_quote_ident("phi")} AS phi '
        f"FROM (SELECT {y_sql}, {violation_sql} FROM {table_sql})"
    )


def _prepare_complete_catalogue(
    *,
    connection: Any,
    table: str,
    constraint_names: Sequence[str],
    cardinality: int,
    source_type: str,
    var_names: Sequence[str] | None,
    purpose: str,
) -> tuple[tuple[str, ...], str, int, int]:
    names = tuple(constraint_names)
    if not names:
        raise ElaError(
            f"constrained {purpose} requires at least one selected constraint"
        )
    source_type = _require_constrained_source_type(source_type, purpose)
    if not isinstance(cardinality, int) or isinstance(cardinality, bool) or cardinality < 1:
        raise ElaError(f"cardinality must be a positive int, got {cardinality!r}")
    table_sql = _quote_table(table)
    import_duckdb()
    n_input, n_unique = _catalogue_counts(connection, table_sql, var_names)
    _assert_constrained_counts(
        source_type,
        cardinality,
        n_input,
        n_unique,
        var_names,
        purpose,
    )
    return names, table_sql, n_input, n_unique


def _identity(
    *,
    problem_id: str,
    source_type: str,
    source_dir: Path,
    n_input: int,
    n_unique: int,
    runtime_s: float,
    config_payload: Mapping[str, Any],
    sampling_design_signature: str | None,
    analysis_signature: str | None,
    seed: int | None,
    repetition: int | None = None,
    library: str = LIBRARY,
    library_version: str | None = None,
) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "source_type": source_type,
        "source_dir": Path(source_dir),
        "n_input": n_input,
        "n_unique": n_unique,
        "runtime_s": runtime_s,
        "config_payload": dict(config_payload),
        "sampling_design_signature": sampling_design_signature,
        "analysis_signature": analysis_signature,
        "seed": seed,
        "repetition": repetition,
        "library": library,
        "library_version": library_version if library_version is not None else _library_version(),
    }


def _as_float_vector(raw: object) -> NDArray[np.floating]:
    if np.ma.isMaskedArray(raw):
        return np.asarray(np.ma.filled(raw, np.nan), dtype=float).reshape(-1)
    return np.asarray(raw, dtype=float).reshape(-1)


def _is_constant(values: NDArray[np.floating]) -> bool:
    if values.size <= 1:
        return True
    return bool(np.all(values == values[0]))


def _spearman_or_undefined(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
) -> float | None:
    if left.size < 2 or _is_constant(left) or _is_constant(right):
        return None
    spearmanr = _import_spearmanr()
    statistic = spearmanr(left, right).statistic
    try:
        value = float(statistic)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _import_spearmanr() -> Any:
    from scipy.stats import spearmanr

    return spearmanr


def _row(
    identity: Mapping[str, Any],
    *,
    landscape: str,
    feature_name: str,
    value: float | None,
    status: str = STATUS_OK,
    role: str = ROLE_PRIMARY,
) -> FeatureResult:
    return feature_result(
        problem_id=identity["problem_id"],
        source_type=identity["source_type"],
        source_dir=identity["source_dir"],
        landscape=landscape,
        response_view=RESPONSE_VIEW_NA,
        x_view=X_VIEW_NA,
        family=FAMILY,
        feature_name=feature_name,
        status=status,
        role=role,
        n_input=identity["n_input"],
        n_unique=identity["n_unique"],
        runtime_s=identity["runtime_s"],
        library=str(identity.get("library", LIBRARY)),
        library_version=str(identity.get("library_version", _library_version())),
        config_payload=identity["config_payload"],
        value=value,
        sampling_design_signature=identity["sampling_design_signature"],
        analysis_signature=identity["analysis_signature"],
        seed=identity["seed"],
        repetition=identity.get("repetition"),
    )
