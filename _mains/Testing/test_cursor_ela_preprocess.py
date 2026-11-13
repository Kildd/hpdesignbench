"""
ELA preprocessing: X_0 / X_norm, V / phi / F, z-tilde, Cartesian-grid checks.
"""
from __future__ import annotations

import unittest
import warnings

import numpy as np

from core.ioh_core.workflows.la.deps import import_duckdb
from core.ioh_core.workflows.la.models import ElaError
from core.ioh_core.workflows.la.preprocess import (
    CONSTANT_RESPONSE_WARNING,
    STATUS_CONSTANT_LANDSCAPE,
    STATUS_OK,
    constraint_violation_sql,
    constraint_violations,
    decision_coordinate_sql,
    landscape_status,
    minmax_response,
    normalize_decision_coordinates,
    preprocess_landscape,
)


class TestDecisionCoordinates(unittest.TestCase):
    """Sealed-bound X_0 / X_norm; never sample min/max."""

    def test_tiny_grid_with_nonzero_lower_bound(self) -> None:
        # geom_t_mm in [0, 2], mat_conc_id in [3, 5]
        X_index = np.array(
            [
                [0, 3],
                [1, 4],
                [2, 5],
            ],
            dtype=np.int64,
        )
        x0, x_norm = normalize_decision_coordinates(
            X_index, lower_bounds=(0, 3), upper_bounds=(2, 5)
        )
        np.testing.assert_array_equal(
            x0,
            np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64),
        )
        np.testing.assert_allclose(
            x_norm,
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
        )
        self.assertTrue(np.all((x_norm >= 0.0) & (x_norm <= 1.0)))

    def test_does_not_use_sample_minmax(self) -> None:
        X_index = np.array([[1, 4], [2, 5]], dtype=np.int64)
        # Sample min/max would be lb=(1,4), ub=(2,5) → first row at 0.
        x0, x_norm = normalize_decision_coordinates(
            X_index, lower_bounds=(0, 3), upper_bounds=(2, 5)
        )
        np.testing.assert_array_equal(x0[0], np.array([1, 1], dtype=np.int64))
        np.testing.assert_allclose(x_norm[0], [0.5, 0.5])

    def test_equal_bounds_fail_closed(self) -> None:
        with self.assertRaises(ElaError) as ctx:
            normalize_decision_coordinates(
                np.array([[0, 3]], dtype=np.int64),
                lower_bounds=(0, 3),
                upper_bounds=(2, 3),
            )
        self.assertIn("upper bound equals lower bound", str(ctx.exception))

    def test_sql_matches_numpy(self) -> None:
        var_names = ("geom_t_mm", "mat_conc_id")
        lower = (0, 3)
        upper = (2, 5)
        X_index = np.array([[0, 3], [1, 4], [2, 5]], dtype=np.int64)
        x0, x_norm = normalize_decision_coordinates(
            X_index, lower_bounds=lower, upper_bounds=upper
        )
        sql = decision_coordinate_sql(var_names, lower, upper)
        duckdb = import_duckdb()
        connection = duckdb.connect()
        try:
            connection.execute(
                'CREATE TABLE t ("idx__geom_t_mm" INTEGER, "idx__mat_conc_id" INTEGER)'
            )
            connection.executemany(
                'INSERT INTO t VALUES (?, ?)',
                [(int(row[0]), int(row[1])) for row in X_index],
            )
            rows = connection.execute(f"SELECT {sql} FROM t").fetchall()
        finally:
            connection.close()
        sql_x0 = np.array([[row[0], row[2]] for row in rows], dtype=np.int64)
        sql_xnorm = np.array([[row[1], row[3]] for row in rows], dtype=float)
        np.testing.assert_array_equal(sql_x0, x0)
        np.testing.assert_allclose(sql_xnorm, x_norm)

    def test_sql_does_not_reference_param_columns(self) -> None:
        sql = decision_coordinate_sql(("geom_t_mm",), (0,), (2,))
        self.assertNotIn("param__", sql)
        self.assertIn("idx__geom_t_mm", sql)

    def test_preprocess_landscape_still_deferred(self) -> None:
        with self.assertRaises(Exception) as ctx:
            preprocess_landscape()
        self.assertIn("not implemented", str(ctx.exception).lower())


class TestConstraintViolations(unittest.TestCase):
    """Selected-util V / phi / F; not stored feasible."""

    def test_util_below_and_above_one(self) -> None:
        U = np.array(
            [
                [0.5, 0.9],
                [0.8, 1.4],
            ],
            dtype=float,
        )
        V, phi, feasible = constraint_violations(U)
        np.testing.assert_allclose(V, [[0.0, 0.0], [0.0, 0.4]])
        np.testing.assert_allclose(phi, [0.0, 0.4])
        np.testing.assert_array_equal(feasible, [True, False])

    def test_constant_constraint_column_is_kept(self) -> None:
        U = np.array(
            [
                [0.4, 0.7],
                [1.2, 0.7],
            ],
            dtype=float,
        )
        V, phi, feasible = constraint_violations(U)
        self.assertEqual(V.shape, (2, 2))
        np.testing.assert_allclose(V[:, 1], [0.0, 0.0])
        np.testing.assert_allclose(phi, [0.0, 0.2])
        np.testing.assert_array_equal(feasible, [True, False])

    def test_f_ignores_a_stored_feasible_flag(self) -> None:
        U = np.array([[0.5, 0.9]], dtype=float)
        stored_feasible = np.array([False])
        _V, _phi, feasible = constraint_violations(U)
        np.testing.assert_array_equal(feasible, [True])
        self.assertFalse(bool(stored_feasible[0]))

    def test_sql_matches_numpy_and_omits_stored_flags(self) -> None:
        names = ("A_bending_capacity", "Z1_nt_dt_combination_capacity")
        U = np.array([[0.5, 0.9], [0.8, 1.4]], dtype=float)
        V, phi, feasible = constraint_violations(U)
        sql = constraint_violation_sql(names)
        self.assertNotIn("feasible", sql)
        self.assertNotIn("constraint_feasible", sql)
        duckdb = import_duckdb()
        connection = duckdb.connect()
        try:
            connection.execute(
                'CREATE TABLE t ("util__A_bending_capacity" DOUBLE, '
                '"util__Z1_nt_dt_combination_capacity" DOUBLE)'
            )
            connection.executemany(
                "INSERT INTO t VALUES (?, ?)",
                [(float(row[0]), float(row[1])) for row in U],
            )
            rows = connection.execute(f"SELECT {sql} FROM t").fetchall()
        finally:
            connection.close()
        sql_v = np.array([[row[0], row[1]] for row in rows], dtype=float)
        sql_phi = np.array([row[2] for row in rows], dtype=float)
        sql_f = np.array([bool(row[3]) for row in rows])
        np.testing.assert_allclose(sql_v, V)
        np.testing.assert_allclose(sql_phi, phi)
        np.testing.assert_array_equal(sql_f, feasible)


class TestMinMaxResponse(unittest.TestCase):
    """Per-repetition z-tilde; independent landscapes; constant unset."""

    def test_independent_landscapes_and_unit_interval(self) -> None:
        y_raw = np.array([10.0, 20.0, 30.0])
        y_p = np.array([100.0, 100.0, 200.0])
        phi = np.array([0.0, 0.5, 1.0])
        raw_view = minmax_response(y_raw)
        penalty_view = minmax_response(y_p)
        phi_view = minmax_response(phi)
        np.testing.assert_allclose(raw_view.z_tilde, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(penalty_view.z_tilde, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(phi_view.z_tilde, [0.0, 0.5, 1.0])
        self.assertEqual(raw_view.response_view, "minmax")
        np.testing.assert_allclose(raw_view.z_raw, y_raw)
        np.testing.assert_allclose(penalty_view.z_raw, y_p)

    def test_does_not_pool_repetitions(self) -> None:
        rep0 = np.array([0.0, 1.0])
        rep1 = np.array([10.0, 11.0])
        view0 = minmax_response(rep0)
        view1 = minmax_response(rep1)
        np.testing.assert_allclose(view0.z_tilde, [0.0, 1.0])
        np.testing.assert_allclose(view1.z_tilde, [0.0, 1.0])
        self.assertEqual(view0.z_max, 1.0)
        self.assertEqual(view1.z_min, 10.0)

    def test_does_not_mutate_input(self) -> None:
        z = np.array([2.0, 4.0])
        view = minmax_response(z)
        z[0] = 99.0
        np.testing.assert_allclose(view.z_raw, [2.0, 4.0])

    def test_constant_warns_and_leaves_z_tilde_unset(self) -> None:
        z = np.array([3.0, 3.0, 3.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            view = minmax_response(z)
        runtime = [
            item for item in caught if issubclass(item.category, RuntimeWarning)
        ]
        self.assertEqual(len(runtime), 1)
        self.assertEqual(str(runtime[0].message), CONSTANT_RESPONSE_WARNING)
        self.assertIsNone(view.z_tilde)
        np.testing.assert_allclose(view.z_raw, z)
        self.assertEqual(view.z_min, 3.0)
        self.assertEqual(view.z_max, 3.0)
        self.assertEqual(view.response_view, "unset")


class TestLandscapeStatus(unittest.TestCase):
    """Feature-level constant_landscape mark; not a source reject."""

    def test_constant_and_varying_landscapes_are_independent(self) -> None:
        y_raw = np.array([4.0, 4.0, 4.0])
        y_p = np.array([1.0, 2.0, 3.0])
        phi = np.array([0.0, 0.1, 0.2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw_view = minmax_response(y_raw)
        penalty_view = minmax_response(y_p)
        phi_view = minmax_response(phi)
        self.assertEqual(landscape_status(raw_view), STATUS_CONSTANT_LANDSCAPE)
        self.assertEqual(landscape_status(penalty_view), STATUS_OK)
        self.assertEqual(landscape_status(phi_view), STATUS_OK)
        self.assertIsNone(raw_view.z_tilde)
        np.testing.assert_allclose(raw_view.z_raw, y_raw)

    def test_varying_landscape_is_ok(self) -> None:
        view = minmax_response(np.array([0.0, 1.0]))
        self.assertEqual(landscape_status(view), STATUS_OK)
        self.assertIsNotNone(view.z_tilde)


class TestPreprocessCartesianGrid(unittest.TestCase):
    """§5.7: one 3×3 grid with non-zero lb, constant constraint, constant y."""

    VAR_NAMES = ("geom_t_mm", "mat_conc_id")
    LOWER = (0, 3)
    UPPER = (2, 5)
    CONSTRAINT_NAMES = (
        "A_bending_capacity",
        "Z1_nt_dt_combination_capacity",
    )

    def test_grid_covers_preprocess_checklist(self) -> None:
        X_index = np.array(
            [
                [0, 3],
                [0, 4],
                [0, 5],
                [1, 3],
                [1, 4],
                [1, 5],
                [2, 3],
                [2, 4],
                [2, 5],
            ],
            dtype=np.int64,
        )
        expected_x0 = np.array(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 1],
                [1, 2],
                [2, 0],
                [2, 1],
                [2, 2],
            ],
            dtype=np.int64,
        )
        expected_x_norm = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.5],
                [0.0, 1.0],
                [0.5, 0.0],
                [0.5, 0.5],
                [0.5, 1.0],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],
            ]
        )
        U = np.array(
            [
                [0.5, 0.7],
                [0.8, 0.7],
                [1.0, 0.7],
                [1.2, 0.7],
                [0.9, 0.7],
                [1.5, 0.7],
                [0.4, 0.7],
                [1.1, 0.7],
                [0.6, 0.7],
            ]
        )
        expected_v = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.2, 0.0],
                [0.0, 0.0],
                [0.5, 0.0],
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.0],
            ]
        )
        expected_phi = np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.5, 0.0, 0.1, 0.0])
        expected_f = np.array(
            [True, True, True, False, True, False, True, False, True]
        )
        y_raw = np.full(9, 5.0)
        y_p = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])

        x0, x_norm = normalize_decision_coordinates(
            X_index, lower_bounds=self.LOWER, upper_bounds=self.UPPER
        )
        np.testing.assert_array_equal(x0, expected_x0)
        np.testing.assert_allclose(x_norm, expected_x_norm)

        V, phi, feasible = constraint_violations(U)
        np.testing.assert_allclose(V, expected_v)
        np.testing.assert_allclose(phi, expected_phi)
        np.testing.assert_array_equal(feasible, expected_f)
        self.assertEqual(V.shape, (9, 2))
        np.testing.assert_allclose(V[:, 1], np.zeros(9))

        stored_feasible = np.zeros(9, dtype=bool)
        self.assertFalse(np.array_equal(feasible, stored_feasible))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw_view = minmax_response(y_raw)
        runtime = [
            item for item in caught if issubclass(item.category, RuntimeWarning)
        ]
        self.assertEqual(len(runtime), 1)
        self.assertEqual(str(runtime[0].message), CONSTANT_RESPONSE_WARNING)
        self.assertIsNone(raw_view.z_tilde)
        np.testing.assert_allclose(raw_view.z_raw, y_raw)
        self.assertEqual(raw_view.response_view, "unset")
        self.assertEqual(landscape_status(raw_view), STATUS_CONSTANT_LANDSCAPE)

        penalty_view = minmax_response(y_p)
        np.testing.assert_allclose(
            penalty_view.z_tilde,
            [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        )
        self.assertEqual(landscape_status(penalty_view), STATUS_OK)

        phi_view = minmax_response(phi)
        np.testing.assert_allclose(
            phi_view.z_tilde, [0.0, 0.0, 0.0, 0.4, 0.0, 1.0, 0.0, 0.2, 0.0]
        )
        self.assertEqual(landscape_status(phi_view), STATUS_OK)

        coord_sql = decision_coordinate_sql(self.VAR_NAMES, self.LOWER, self.UPPER)
        viol_sql = constraint_violation_sql(self.CONSTRAINT_NAMES)
        self.assertNotIn("param__", coord_sql)
        self.assertNotIn("feasible", viol_sql)
        duckdb = import_duckdb()
        connection = duckdb.connect()
        try:
            connection.execute(
                'CREATE TABLE t ('
                '"idx__geom_t_mm" INTEGER, "idx__mat_conc_id" INTEGER, '
                '"util__A_bending_capacity" DOUBLE, '
                '"util__Z1_nt_dt_combination_capacity" DOUBLE)'
            )
            connection.executemany(
                "INSERT INTO t VALUES (?, ?, ?, ?)",
                [
                    (int(row[0]), int(row[1]), float(u[0]), float(u[1]))
                    for row, u in zip(X_index, U)
                ],
            )
            rows = connection.execute(
                f"SELECT {coord_sql}, {viol_sql} FROM t "
                'ORDER BY "idx__geom_t_mm", "idx__mat_conc_id"'
            ).fetchall()
        finally:
            connection.close()
        sql_x0 = np.array([[row[0], row[2]] for row in rows], dtype=np.int64)
        sql_x_norm = np.array([[row[1], row[3]] for row in rows], dtype=float)
        sql_v = np.array([[row[4], row[5]] for row in rows], dtype=float)
        sql_phi = np.array([row[6] for row in rows], dtype=float)
        sql_f = np.array([bool(row[7]) for row in rows])
        np.testing.assert_array_equal(sql_x0, expected_x0)
        np.testing.assert_allclose(sql_x_norm, expected_x_norm)
        np.testing.assert_allclose(sql_v, expected_v)
        np.testing.assert_allclose(sql_phi, expected_phi)
        np.testing.assert_array_equal(sql_f, expected_f)


if __name__ == "__main__":
    unittest.main()
