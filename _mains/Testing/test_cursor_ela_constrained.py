"""
ELA constrained FsR / FsR_i / FVC / ρ_i: complete BF DuckDB + SciPy Spearman.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from core.ioh_core.workflows.la.constrained import (
    FAMILY,
    FEATURE_25_IZ,
    FEATURE_4_IZ,
    FEATURE_FSR,
    FEATURE_FVC,
    IZ_Y_QUANTILES,
    QUANTILE_METHOD,
    RANK_METHOD,
    SPEARMAN_METHOD,
    V_DISTRIBUTION_QUANTILES,
    c_i_feature_name,
    ela_constrained,
    ela_constrained_contribution,
    ela_constrained_cooccurrence,
    ela_constrained_distributions,
    ela_constrained_iz,
    ela_constrained_pairwise,
    ela_constrained_spearman,
    fsr_i_feature_name,
    m_ij_feature_name,
    rho_i_feature_name,
    rho_ij_feature_name,
    v_quantile_feature_name,
    v_stat_feature_name,
)
from core.ioh_core.workflows.la.deps import import_duckdb
from core.ioh_core.workflows.la.models import (
    ElaError,
    STATUS_OK,
    STATUS_UNDEFINED,
    canonical_config_json,
)
from core.ioh_core.workflows.la.preprocess import (
    constraint_violation_sql,
    constraint_violations,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = Path("experiments/bf_p1")
_CONSTRAINTS = ("A_bending_capacity", "Z1_nt_dt_combination_capacity")
_VARS = ("geom_t_mm", "mat_conc_id")
_SPEARMAN_CONFIG = {
    "constraint_names": list(_CONSTRAINTS),
    "spearman": SPEARMAN_METHOD,
    "rank_method": RANK_METHOD,
}
_IZ_CONFIG = {
    "constraint_names": list(_CONSTRAINTS),
    "y_quantile": list(IZ_Y_QUANTILES),
    "quantile_method": QUANTILE_METHOD,
}
_DIST_CONFIG = {
    "constraint_names": list(_CONSTRAINTS),
    "v_quantile": list(V_DISTRIBUTION_QUANTILES),
    "quantile_method": QUANTILE_METHOD,
}


def _complete_grid() -> tuple[np.ndarray, np.ndarray]:
    """2×2 catalogue: two feasible, one A-only violation, one Z-only."""
    X_index = np.array([[0, 3], [0, 4], [1, 3], [1, 4]], dtype=np.int64)
    U = np.array(
        [
            [0.5, 0.5],
            [0.5, 1.2],
            [1.5, 0.5],
            [0.8, 0.8],
        ],
        dtype=float,
    )
    return X_index, U


def _tied_y() -> np.ndarray:
    """Objective with repeated values so average-rank ties matter."""
    return np.array([10.0, 20.0, 20.0, 10.0], dtype=float)


def _load_table(
    X_index: np.ndarray,
    U: np.ndarray,
    *,
    y: np.ndarray | None = None,
    extra_feasible: np.ndarray | None = None,
) -> object:
    duckdb = import_duckdb()
    connection = duckdb.connect()
    columns = (
        '"idx__geom_t_mm" INTEGER, "idx__mat_conc_id" INTEGER, '
        '"util__A_bending_capacity" DOUBLE, '
        '"util__Z1_nt_dt_combination_capacity" DOUBLE, '
        '"y" DOUBLE, '
        '"feasible" BOOLEAN'
    )
    connection.execute(f"CREATE TABLE src ({columns})")
    n_rows = X_index.shape[0]
    stored = (
        extra_feasible
        if extra_feasible is not None
        else np.zeros(n_rows, dtype=bool)
    )
    objective = np.arange(n_rows, dtype=float) if y is None else np.asarray(y, dtype=float)
    connection.executemany(
        "INSERT INTO src VALUES (?, ?, ?, ?, ?, ?)",
        [
            (int(x[0]), int(x[1]), float(u[0]), float(u[1]), float(yi), bool(flag))
            for x, u, yi, flag in zip(X_index, U, objective, stored, strict=True)
        ],
    )
    return connection


def _run(connection: object, cardinality: int, **overrides: object) -> list:
    payload: dict[str, object] = {
        "connection": connection,
        "table": "src",
        "constraint_names": _CONSTRAINTS,
        "cardinality": cardinality,
        "problem_id": "L6750_B2",
        "source_dir": _SOURCE_DIR,
        "var_names": _VARS,
    }
    payload.update(overrides)
    return ela_constrained(**payload)  # type: ignore[arg-type]


def _run_spearman(connection: object, cardinality: int, **overrides: object) -> list:
    payload: dict[str, object] = {
        "connection": connection,
        "table": "src",
        "constraint_names": _CONSTRAINTS,
        "cardinality": cardinality,
        "problem_id": "L6750_B2",
        "source_dir": _SOURCE_DIR,
        "var_names": _VARS,
    }
    payload.update(overrides)
    return ela_constrained_spearman(**payload)  # type: ignore[arg-type]


def _run_iz(connection: object, cardinality: int, **overrides: object) -> list:
    return _call(ela_constrained_iz, connection, cardinality, **overrides)


def _run_distributions(connection: object, cardinality: int, **overrides: object) -> list:
    return _call(ela_constrained_distributions, connection, cardinality, **overrides)


def _run_pairwise(connection: object, cardinality: int, **overrides: object) -> list:
    return _call(ela_constrained_pairwise, connection, cardinality, **overrides)


def _run_cooccurrence(connection: object, cardinality: int, **overrides: object) -> list:
    return _call(ela_constrained_cooccurrence, connection, cardinality, **overrides)


def _run_contribution(connection: object, cardinality: int, **overrides: object) -> list:
    return _call(ela_constrained_contribution, connection, cardinality, **overrides)


def _call(fn: object, connection: object, cardinality: int, **overrides: object) -> list:
    payload: dict[str, object] = {
        "connection": connection,
        "table": "src",
        "constraint_names": _CONSTRAINTS,
        "cardinality": cardinality,
        "problem_id": "L6750_B2",
        "source_dir": _SOURCE_DIR,
        "var_names": _VARS,
    }
    payload.update(overrides)
    return fn(**payload)  # type: ignore[operator]


def _iz_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n=5 catalogue: many phi==0 ties so the below-median phi set is < 50%."""
    X_index = np.array(
        [[0, 3], [0, 4], [1, 3], [1, 4], [2, 3]],
        dtype=np.int64,
    )
    U = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [2.0, 0.5],
            [3.0, 0.5],
            [4.0, 0.5],
        ],
        dtype=float,
    )
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
    return X_index, U, y


class TestElaConstrainedFsr(unittest.TestCase):
    """DuckDB FsR / FsR_i on a tiny complete catalogue."""

    def test_complete_grid_matches_numpy_fixture(self) -> None:
        X_index, U = _complete_grid()
        _V, phi, feasible = constraint_violations(U)
        connection = _load_table(X_index, U)
        try:
            rows = _run(connection, cardinality=4)
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        self.assertEqual(by_name[FEATURE_FSR].value, float(feasible.mean()))
        self.assertAlmostEqual(by_name[FEATURE_FSR].value, 0.5)
        np.testing.assert_allclose(phi == 0.0, feasible)
        self.assertAlmostEqual(
            by_name[fsr_i_feature_name(_CONSTRAINTS[0])].value,
            float((_V[:, 0] == 0.0).mean()),
        )
        self.assertAlmostEqual(
            by_name[fsr_i_feature_name(_CONSTRAINTS[1])].value,
            float((_V[:, 1] == 0.0).mean()),
        )
        sample = rows[0]
        self.assertEqual(sample.family, FAMILY)
        self.assertEqual(sample.landscape, "feasibility")
        self.assertEqual(sample.x_view, "n/a")
        self.assertEqual(sample.response_view, "n/a")
        self.assertEqual(sample.status, STATUS_OK)
        self.assertEqual(sample.role, "primary")
        self.assertEqual(sample.n_input, 4)
        self.assertEqual(sample.n_unique, 4)
        self.assertEqual(sample.source_type, "brute_force")
        self.assertEqual(sample.library, "duckdb")
        self.assertEqual(
            sample.config_payload,
            {"constraint_names": list(_CONSTRAINTS)},
        )
        self.assertEqual(
            sample.config_json,
            canonical_config_json({"constraint_names": list(_CONSTRAINTS)}),
        )
        self.assertNotIn("n_used", sample.__dataclass_fields__)
        self.assertEqual(
            by_name[fsr_i_feature_name(_CONSTRAINTS[0])].landscape,
            "constraint_A_bending_capacity",
        )
        self.assertNotIn(FEATURE_FVC, by_name)
        self.assertNotIn(FEATURE_25_IZ, by_name)

    def test_constant_constraint_has_fsr_i_one(self) -> None:
        X_index = np.array([[0, 3], [1, 4]], dtype=np.int64)
        U = np.array([[0.4, 0.7], [1.2, 0.7]], dtype=float)
        connection = _load_table(X_index, U)
        try:
            rows = _run(connection, cardinality=2)
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        self.assertEqual(by_name[fsr_i_feature_name(_CONSTRAINTS[1])].value, 1.0)
        self.assertEqual(by_name[FEATURE_FSR].value, 0.5)

    def test_ignores_stored_feasible_flag(self) -> None:
        X_index, U = _complete_grid()
        connection = _load_table(
            X_index,
            U,
            extra_feasible=np.zeros(X_index.shape[0], dtype=bool),
        )
        try:
            rows = _run(connection, cardinality=4)
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        self.assertEqual(by_name[FEATURE_FSR].value, 0.5)

    def test_incomplete_catalogue_fails(self) -> None:
        X_index, U = _complete_grid()
        connection = _load_table(X_index[:3], U[:3])
        try:
            with self.assertRaises(ElaError) as ctx:
                _run(connection, cardinality=4)
        finally:
            connection.close()
        self.assertIn("cardinality", str(ctx.exception))

    def test_sampling_source_succeeds(self) -> None:
        X_index, U = _complete_grid()
        connection = _load_table(X_index, U)
        try:
            rows = _run(connection, cardinality=4, source_type="sampling")
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        self.assertEqual(by_name[FEATURE_FSR].source_type, "sampling")
        self.assertEqual(by_name[FEATURE_FSR].n_input, 4)
        self.assertEqual(by_name[FEATURE_FSR].n_unique, 4)
        self.assertEqual(by_name[FEATURE_FSR].value, 0.5)

    def test_importing_module_does_not_load_duckdb_or_pflacco(self) -> None:
        script = """
import sys
import core.ioh_core.workflows.la
import core.ioh_core.workflows.la.constrained
loaded = [name for name in ("duckdb", "pflacco", "scipy.stats") if name in sys.modules]
if loaded:
    raise SystemExit("loaded forbidden modules: " + ", ".join(loaded))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stderr or completed.stdout,
        )


class TestElaConstrainedSpearman(unittest.TestCase):
    """SciPy Spearman FVC / ρ_i on a tiny complete catalogue."""

    def test_tied_vectors_match_spearmanr(self) -> None:
        X_index, U = _complete_grid()
        y = _tied_y()
        V, phi, _feasible = constraint_violations(U)
        connection = _load_table(X_index, U, y=y)
        try:
            rows = _run_spearman(connection, cardinality=4)
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        expected_fvc = float(spearmanr(y, phi).statistic)
        self.assertAlmostEqual(by_name[FEATURE_FVC].value, expected_fvc)
        self.assertAlmostEqual(
            by_name[rho_i_feature_name(_CONSTRAINTS[0])].value,
            float(spearmanr(y, V[:, 0]).statistic),
        )
        self.assertAlmostEqual(
            by_name[rho_i_feature_name(_CONSTRAINTS[1])].value,
            float(spearmanr(y, V[:, 1]).statistic),
        )
        sample = by_name[FEATURE_FVC]
        self.assertEqual(sample.family, FAMILY)
        self.assertEqual(sample.landscape, "phi")
        self.assertEqual(sample.x_view, "n/a")
        self.assertEqual(sample.response_view, "n/a")
        self.assertEqual(sample.status, STATUS_OK)
        self.assertEqual(sample.role, "primary")
        self.assertEqual(sample.library, "scipy")
        self.assertEqual(sample.n_input, 4)
        self.assertEqual(sample.n_unique, 4)
        self.assertEqual(sample.config_payload, _SPEARMAN_CONFIG)
        self.assertEqual(sample.config_json, canonical_config_json(_SPEARMAN_CONFIG))
        self.assertNotIn("n_used", sample.__dataclass_fields__)
        self.assertEqual(
            by_name[rho_i_feature_name(_CONSTRAINTS[0])].landscape,
            "constraint_A_bending_capacity",
        )
        pearson = float(np.corrcoef(y, phi)[0, 1])
        self.assertNotAlmostEqual(expected_fvc, pearson)

    def test_constant_y_is_undefined(self) -> None:
        X_index, U = _complete_grid()
        y = np.full(X_index.shape[0], 5.0)
        connection = _load_table(X_index, U, y=y)
        try:
            rows = _run_spearman(connection, cardinality=4)
        finally:
            connection.close()
        for row in rows:
            self.assertIsNone(row.value)
            self.assertEqual(row.status, STATUS_UNDEFINED)

    def test_constant_constraint_rho_i_is_undefined(self) -> None:
        X_index = np.array([[0, 3], [1, 4]], dtype=np.int64)
        U = np.array([[0.4, 0.7], [1.2, 0.7]], dtype=float)
        y = np.array([1.0, 2.0], dtype=float)
        V, phi, _feasible = constraint_violations(U)
        connection = _load_table(X_index, U, y=y)
        try:
            rows = _run_spearman(connection, cardinality=2)
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        constant = by_name[rho_i_feature_name(_CONSTRAINTS[1])]
        self.assertTrue(np.all(V[:, 1] == V[0, 1]))
        self.assertIsNone(constant.value)
        self.assertEqual(constant.status, STATUS_UNDEFINED)
        self.assertAlmostEqual(
            by_name[FEATURE_FVC].value,
            float(spearmanr(y, phi).statistic),
        )
        self.assertEqual(by_name[FEATURE_FVC].status, STATUS_OK)
        self.assertAlmostEqual(
            by_name[rho_i_feature_name(_CONSTRAINTS[0])].value,
            float(spearmanr(y, V[:, 0]).statistic),
        )

    def test_ignores_stored_feasible_flag(self) -> None:
        X_index, U = _complete_grid()
        y = _tied_y()
        V, phi, _feasible = constraint_violations(U)
        connection = _load_table(
            X_index,
            U,
            y=y,
            extra_feasible=np.zeros(X_index.shape[0], dtype=bool),
        )
        try:
            rows = _run_spearman(connection, cardinality=4)
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        self.assertAlmostEqual(
            by_name[FEATURE_FVC].value,
            float(spearmanr(y, phi).statistic),
        )
        self.assertAlmostEqual(
            by_name[rho_i_feature_name(_CONSTRAINTS[0])].value,
            float(spearmanr(y, V[:, 0]).statistic),
        )

    def test_incomplete_catalogue_fails(self) -> None:
        X_index, U = _complete_grid()
        connection = _load_table(X_index[:3], U[:3], y=_tied_y()[:3])
        try:
            with self.assertRaises(ElaError) as ctx:
                _run_spearman(connection, cardinality=4)
        finally:
            connection.close()
        self.assertIn("cardinality", str(ctx.exception))

    def test_sampling_source_succeeds(self) -> None:
        X_index, U = _complete_grid()
        connection = _load_table(X_index, U, y=_tied_y())
        try:
            rows = _run_spearman(connection, cardinality=4, source_type="sampling")
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        self.assertEqual(by_name[FEATURE_FVC].source_type, "sampling")
        self.assertEqual(by_name[FEATURE_FVC].n_input, 4)
        self.assertEqual(by_name[FEATURE_FVC].n_unique, 4)
        self.assertIsNotNone(by_name[FEATURE_FVC].value)


class TestElaConstrainedIz(unittest.TestCase):
    """DuckDB 25_IZ / 4_IZ on a tiny complete catalogue."""

    def test_matches_numpy_quantile_and_strict_intersection(self) -> None:
        X_index, U, y = _iz_grid()
        _V, phi, _feasible = constraint_violations(U)
        q50_y = float(np.quantile(y, 0.5))
        q50_phi = float(np.quantile(phi, 0.5))
        q20_y = float(np.quantile(y, 0.2))
        q20_phi = float(np.quantile(phi, 0.2))
        expected_25 = float(np.mean((y < q50_y) & (phi < q50_phi)))
        expected_4 = float(np.mean((y < q20_y) & (phi < q20_phi)))
        self.assertLess(float(np.mean(phi < q50_phi)), 0.5)
        connection = _load_table(X_index, U, y=y)
        try:
            viol_sql = constraint_violation_sql(_CONSTRAINTS)
            duck_q = connection.execute(
                "SELECT quantile_cont(y, 0.5), quantile_cont(phi, 0.5), "
                "quantile_cont(y, 0.2), quantile_cont(phi, 0.2) "
                f'FROM (SELECT CAST("y" AS DOUBLE) AS y, {viol_sql} FROM src)'
            ).fetchone()
            rows = _run_iz(connection, cardinality=5)
        finally:
            connection.close()
        self.assertIsNotNone(duck_q)
        np.testing.assert_allclose(
            [q50_y, q50_phi, q20_y, q20_phi],
            [float(duck_q[0]), float(duck_q[1]), float(duck_q[2]), float(duck_q[3])],
            atol=0.0,
            rtol=0.0,
        )
        by_name = {row.feature_name: row for row in rows}
        self.assertAlmostEqual(by_name[FEATURE_25_IZ].value, expected_25)
        self.assertAlmostEqual(by_name[FEATURE_4_IZ].value, expected_4)
        self.assertNotAlmostEqual(by_name[FEATURE_25_IZ].value, expected_25 / 0.25)
        sample = by_name[FEATURE_25_IZ]
        self.assertEqual(sample.family, FAMILY)
        self.assertEqual(sample.landscape, "phi")
        self.assertEqual(sample.x_view, "n/a")
        self.assertEqual(sample.response_view, "n/a")
        self.assertEqual(sample.status, STATUS_OK)
        self.assertEqual(sample.role, "primary")
        self.assertEqual(sample.library, "duckdb")
        self.assertEqual(sample.n_input, 5)
        self.assertEqual(sample.n_unique, 5)
        self.assertEqual(sample.config_payload, _IZ_CONFIG)
        self.assertEqual(sample.config_json, canonical_config_json(_IZ_CONFIG))
        self.assertNotIn("n_used", sample.__dataclass_fields__)

    def test_constant_y_is_zero(self) -> None:
        X_index, U = _complete_grid()
        y = np.full(X_index.shape[0], 5.0)
        connection = _load_table(X_index, U, y=y)
        try:
            rows = _run_iz(connection, cardinality=4)
        finally:
            connection.close()
        for row in rows:
            self.assertEqual(row.value, 0.0)
            self.assertEqual(row.status, STATUS_OK)

    def test_ignores_stored_feasible_flag(self) -> None:
        X_index, U, y = _iz_grid()
        _V, phi, _feasible = constraint_violations(U)
        expected_25 = float(
            np.mean((y < np.quantile(y, 0.5)) & (phi < np.quantile(phi, 0.5)))
        )
        connection = _load_table(
            X_index,
            U,
            y=y,
            extra_feasible=np.zeros(X_index.shape[0], dtype=bool),
        )
        try:
            rows = _run_iz(connection, cardinality=5)
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        self.assertAlmostEqual(by_name[FEATURE_25_IZ].value, expected_25)

    def test_incomplete_catalogue_fails(self) -> None:
        X_index, U, y = _iz_grid()
        connection = _load_table(X_index[:4], U[:4], y=y[:4])
        try:
            with self.assertRaises(ElaError) as ctx:
                _run_iz(connection, cardinality=5)
        finally:
            connection.close()
        self.assertIn("cardinality", str(ctx.exception))

    def test_sampling_source_succeeds(self) -> None:
        X_index, U, y = _iz_grid()
        connection = _load_table(X_index, U, y=y)
        try:
            rows = _run_iz(connection, cardinality=5, source_type="sampling")
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        self.assertEqual(by_name[FEATURE_25_IZ].source_type, "sampling")
        self.assertEqual(by_name[FEATURE_25_IZ].n_input, 5)
        self.assertEqual(by_name[FEATURE_25_IZ].n_unique, 5)
        self.assertIsNotNone(by_name[FEATURE_25_IZ].value)


class TestElaConstrainedDistributions(unittest.TestCase):
    """DuckDB per-constraint v_i summaries."""

    def test_matches_numpy_and_frac_zero_equals_fsr_i(self) -> None:
        X_index, U = _complete_grid()
        V, _phi, _feasible = constraint_violations(U)
        connection = _load_table(X_index, U)
        try:
            dist_rows = _run_distributions(connection, cardinality=4)
            fsr_rows = _run(connection, cardinality=4)
            viol_sql = constraint_violation_sql(_CONSTRAINTS)
            duck_q = connection.execute(
                "SELECT quantile_cont("
                f'{_quote("v__" + _CONSTRAINTS[0])}, 0.9) '
                f"FROM (SELECT {viol_sql} FROM src)"
            ).fetchone()
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in dist_rows}
        fsr_by_name = {row.feature_name: row for row in fsr_rows}
        for index, name in enumerate(_CONSTRAINTS):
            column = V[:, index]
            self.assertAlmostEqual(
                by_name[v_stat_feature_name("min", name)].value,
                float(column.min()),
            )
            self.assertAlmostEqual(
                by_name[v_stat_feature_name("max", name)].value,
                float(column.max()),
            )
            self.assertAlmostEqual(
                by_name[v_stat_feature_name("mean", name)].value,
                float(column.mean()),
            )
            self.assertAlmostEqual(
                by_name[v_stat_feature_name("median", name)].value,
                float(np.quantile(column, 0.5)),
            )
            for quantile in V_DISTRIBUTION_QUANTILES:
                self.assertAlmostEqual(
                    by_name[v_quantile_feature_name(quantile, name)].value,
                    float(np.quantile(column, quantile)),
                )
            frac_zero = by_name[v_stat_feature_name("frac_zero", name)].value
            self.assertAlmostEqual(frac_zero, float(np.mean(column == 0.0)))
            self.assertAlmostEqual(
                frac_zero,
                fsr_by_name[fsr_i_feature_name(name)].value,
            )
        self.assertIsNotNone(duck_q)
        self.assertEqual(
            float(duck_q[0]),
            float(np.quantile(V[:, 0], 0.9)),
        )
        sample = dist_rows[0]
        self.assertEqual(sample.family, FAMILY)
        self.assertEqual(sample.library, "duckdb")
        self.assertEqual(sample.status, STATUS_OK)
        self.assertEqual(sample.role, "primary")
        self.assertEqual(sample.config_payload, _DIST_CONFIG)
        self.assertEqual(sample.landscape, "constraint_A_bending_capacity")

    def test_incomplete_fails_sampling_succeeds(self) -> None:
        _assert_incomplete_fails_and_sampling_succeeds(self, _run_distributions)


class TestElaConstrainedPairwise(unittest.TestCase):
    """Pairwise Spearman among v_i."""

    def test_full_matrix_matches_spearmanr(self) -> None:
        X_index, U = _complete_grid()
        V, _phi, _feasible = constraint_violations(U)
        connection = _load_table(X_index, U)
        try:
            rows = _run_pairwise(connection, cardinality=4)
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        self.assertEqual(len(rows), 4)
        for i, left in enumerate(_CONSTRAINTS):
            for j, right in enumerate(_CONSTRAINTS):
                expected = float(spearmanr(V[:, i], V[:, j]).statistic)
                got = by_name[rho_ij_feature_name(left, right)]
                self.assertAlmostEqual(got.value, expected)
                self.assertEqual(got.status, STATUS_OK)
                self.assertEqual(got.library, "scipy")
                self.assertEqual(
                    got.landscape,
                    f"constraint_{left}",
                )
        self.assertAlmostEqual(
            by_name[rho_ij_feature_name(_CONSTRAINTS[0], _CONSTRAINTS[0])].value,
            1.0,
        )
        self.assertAlmostEqual(
            by_name[rho_ij_feature_name(_CONSTRAINTS[0], _CONSTRAINTS[1])].value,
            by_name[rho_ij_feature_name(_CONSTRAINTS[1], _CONSTRAINTS[0])].value,
        )

    def test_constant_constraint_row_is_undefined(self) -> None:
        X_index = np.array([[0, 3], [1, 4]], dtype=np.int64)
        U = np.array([[0.4, 0.7], [1.2, 0.7]], dtype=float)
        connection = _load_table(X_index, U)
        try:
            rows = _run_pairwise(connection, cardinality=2)
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        constant = _CONSTRAINTS[1]
        varying = _CONSTRAINTS[0]
        for other in _CONSTRAINTS:
            row = by_name[rho_ij_feature_name(constant, other)]
            self.assertIsNone(row.value)
            self.assertEqual(row.status, STATUS_UNDEFINED)
            row = by_name[rho_ij_feature_name(other, constant)]
            self.assertIsNone(row.value)
            self.assertEqual(row.status, STATUS_UNDEFINED)
        self.assertAlmostEqual(
            by_name[rho_ij_feature_name(varying, varying)].value,
            1.0,
        )

    def test_incomplete_fails_sampling_succeeds(self) -> None:
        _assert_incomplete_fails_and_sampling_succeeds(self, _run_pairwise)


class TestElaConstrainedCooccurrence(unittest.TestCase):
    """M_ij = P(v_i>0 AND v_j>0)."""

    def test_symmetric_matrix_matches_numpy(self) -> None:
        X_index, U = _complete_grid()
        V, _phi, _feasible = constraint_violations(U)
        connection = _load_table(
            X_index,
            U,
            extra_feasible=np.zeros(X_index.shape[0], dtype=bool),
        )
        try:
            rows = _run_cooccurrence(connection, cardinality=4)
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        positive = V > 0.0
        for i, left in enumerate(_CONSTRAINTS):
            for j, right in enumerate(_CONSTRAINTS):
                expected = float(np.mean(positive[:, i] & positive[:, j]))
                got = by_name[m_ij_feature_name(left, right)]
                self.assertAlmostEqual(got.value, expected)
                self.assertEqual(got.status, STATUS_OK)
        self.assertAlmostEqual(
            by_name[m_ij_feature_name(_CONSTRAINTS[0], _CONSTRAINTS[0])].value,
            float(np.mean(positive[:, 0])),
        )
        self.assertAlmostEqual(
            by_name[m_ij_feature_name(_CONSTRAINTS[0], _CONSTRAINTS[1])].value,
            by_name[m_ij_feature_name(_CONSTRAINTS[1], _CONSTRAINTS[0])].value,
        )
        self.assertAlmostEqual(
            by_name[m_ij_feature_name(_CONSTRAINTS[0], _CONSTRAINTS[1])].value,
            0.0,
        )

    def test_incomplete_fails_sampling_succeeds(self) -> None:
        _assert_incomplete_fails_and_sampling_succeeds(self, _run_cooccurrence)


class TestElaConstrainedContribution(unittest.TestCase):
    """Diagnostic C_i = sum v_i / sum phi."""

    def test_matches_numpy_and_is_diagnostic(self) -> None:
        X_index, U = _complete_grid()
        V, phi, _feasible = constraint_violations(U)
        denom = float(phi.sum())
        connection = _load_table(X_index, U)
        try:
            rows = _run_contribution(connection, cardinality=4)
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        for index, name in enumerate(_CONSTRAINTS):
            row = by_name[c_i_feature_name(name)]
            self.assertAlmostEqual(row.value, float(V[:, index].sum()) / denom)
            self.assertEqual(row.role, "diagnostic")
            self.assertEqual(row.status, STATUS_OK)
            self.assertEqual(row.landscape, f"constraint_{name}")

    def test_all_feasible_is_undefined(self) -> None:
        X_index = np.array([[0, 3], [1, 4]], dtype=np.int64)
        U = np.array([[0.4, 0.5], [0.8, 0.9]], dtype=float)
        connection = _load_table(X_index, U)
        try:
            rows = _run_contribution(connection, cardinality=2)
        finally:
            connection.close()
        for row in rows:
            self.assertIsNone(row.value)
            self.assertEqual(row.status, STATUS_UNDEFINED)
            self.assertEqual(row.role, "diagnostic")

    def test_incomplete_fails_sampling_succeeds(self) -> None:
        _assert_incomplete_fails_and_sampling_succeeds(self, _run_contribution)


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _assert_incomplete_fails_and_sampling_succeeds(
    test: unittest.TestCase, runner: object
) -> None:
    X_index, U = _complete_grid()
    connection = _load_table(X_index[:3], U[:3])
    try:
        with test.assertRaises(ElaError) as ctx:
            runner(connection, cardinality=4)  # type: ignore[operator]
        test.assertIn("cardinality", str(ctx.exception))
    finally:
        connection.close()
    connection = _load_table(X_index, U)
    try:
        rows = runner(connection, cardinality=4, source_type="sampling")  # type: ignore[operator]
    finally:
        connection.close()
    test.assertGreater(len(rows), 0)
    test.assertEqual(rows[0].source_type, "sampling")
    test.assertEqual(rows[0].n_input, 4)
    test.assertEqual(rows[0].n_unique, 4)


if __name__ == "__main__":
    unittest.main()
