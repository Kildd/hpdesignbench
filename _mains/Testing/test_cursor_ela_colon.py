"""CoLON basin census on Deb nested best-exit (#57)."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.la.colon import (
    FEATURE_GLOBAL_SUPPORT_SHARE,
    FEATURE_N_GLOBAL_NODES,
    FEATURE_STAGE_ESCAPES,
    FEATURE_Y_STAR,
    GLOBAL_FEATURE_NAMES,
    colon_escape_edge_records,
    ela_colon,
    set_valued_colon,
)
from core.ioh_core.workflows.la.models import STATUS_UNDEFINED
from core.ioh_core.workflows.la.preprocess import constraint_violations


class TestColonBasinCensus(unittest.TestCase):
    def test_violation_plateau_and_mixed_basin(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (3,))
        y = np.array([1.0, 99.0, 0.0, 1.0])
        phi = np.array([0.0, 2.0, 2.0, 0.0])
        feasible = np.array([True, False, False, True])
        result = set_valued_colon(
            space=space, y_raw=y, feasible=feasible, phi=phi
        )
        self.assertEqual(
            int(result.graph.plateau_of_linear[1]),
            int(result.graph.plateau_of_linear[2]),
        )
        self.assertEqual(result.support_for_linear(1), (0, 1))
        self.assertEqual(result.n_nodes, 2)
        shares = result.basin_f_share
        self.assertTrue(np.all(shares < 1.0))
        self.assertTrue(np.all(shares > 0.0))
        self.assertEqual(result.y_star, 1.0)
        self.assertEqual(result.n_global_nodes, 2)
        self.assertTrue(bool(np.all(result.terminal_feasible)))

    def test_empty_f_publishes_traps_and_undefines_global_keys(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (2,))
        y = np.array([0.0, 8.0, 1.0])
        phi = np.array([3.0, 1.0, 2.0])
        feasible = np.array([False, False, False])
        rows, result = ela_colon(
            space=space,
            y_raw=y,
            feasible=feasible,
            phi=phi,
            problem_id="L6750_B2",
            source_dir=Path("."),
            constraint_names=("A_bending_capacity",),
        )
        self.assertGreater(result.n_nodes, 0)
        self.assertIsNone(result.y_star)
        self.assertFalse(bool(np.any(result.terminal_feasible)))
        by_name = {row.feature_name: row for row in rows}
        for name in GLOBAL_FEATURE_NAMES:
            self.assertEqual(by_name[name].status, STATUS_UNDEFINED)
            self.assertIsNone(by_name[name].value)
        self.assertEqual(
            GLOBAL_FEATURE_NAMES,
            {
                FEATURE_N_GLOBAL_NODES,
                FEATURE_GLOBAL_SUPPORT_SHARE,
                FEATURE_Y_STAR,
            },
        )
        self.assertEqual(by_name[FEATURE_STAGE_ESCAPES].status, "ok")
        self.assertEqual(by_name[FEATURE_STAGE_ESCAPES].value, 1.0)
        self.assertIn("colon.escape.d2.escape_candidate_feasible", by_name)
        self.assertIn("colon.escape.d2.escape_candidate_infeasible", by_name)
        self.assertNotIn("colon.escape.d2.crossing", by_name)

    def test_escape_shells_split_perturbed_f_and_nest_balls(self) -> None:
        space = IndexSpace.from_bounds(("r", "c"), (0, 0), (1, 1))
        result = set_valued_colon(
            space=space,
            y_raw=np.array([0.0, 5.0, 5.0, 9.0]),
            feasible=np.array([True, True, True, False]),
            phi=np.array([0.0, 0.0, 0.0, 1.0]),
            radii=(1, 2),
        )
        self.assertEqual(result.n_nodes, 1)
        self.assertEqual(result.shell_candidate_count[(0, 1)], 2)
        self.assertEqual(result.shell_candidate_count[(0, 2)], 1)
        self.assertEqual(result.shell_feasible_candidate_count[(0, 1)], 2)
        self.assertEqual(result.shell_feasible_candidate_count[(0, 2)], 0)
        self.assertEqual(result.shell_infeasible_candidate_count[(0, 1)], 0)
        self.assertEqual(result.shell_infeasible_candidate_count[(0, 2)], 1)
        d1_pairs = {
            (source, target)
            for source, target, shell in result.edge_support_count
            if shell <= 1
        }
        d2_pairs = {
            (source, target)
            for source, target, shell in result.edge_support_count
            if shell <= 2
        }
        self.assertTrue(d1_pairs.issubset(d2_pairs))
        records = colon_escape_edge_records(result)
        self.assertTrue(records)
        self.assertIn("source_shell_candidate_feasible_count", records[0])
        self.assertIn("source_shell_candidate_infeasible_count", records[0])
        self.assertNotIn("crossing", records[0])
        rows, _published = ela_colon(
            space=space,
            y_raw=np.array([0.0, 5.0, 5.0, 9.0]),
            feasible=np.array([True, True, True, False]),
            phi=np.array([0.0, 0.0, 0.0, 1.0]),
            radii=(1, 2),
            problem_id="L6750_B2",
            source_dir=Path("."),
            constraint_names=("A_bending_capacity",),
        )
        by_name = {row.feature_name: row for row in rows}
        self.assertEqual(by_name["colon.escape.d1.escape_candidate_feasible"].value, 2.0)
        self.assertEqual(by_name["colon.escape.d1.escape_candidate_infeasible"].value, 0.0)
        self.assertEqual(by_name["colon.escape.d2.escape_candidate_feasible"].value, 2.0)
        self.assertEqual(by_name["colon.escape.d2.escape_candidate_infeasible"].value, 1.0)
        self.assertGreaterEqual(
            by_name["colon.escape.d2.n_edges"].value,
            by_name["colon.escape.d1.n_edges"].value,
        )

    def test_funnels_use_nonworsening_h_not_y_raw(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (2,))
        result = set_valued_colon(
            space=space,
            y_raw=np.array([5.0, 0.0, 5.0]),
            feasible=np.array([False, False, False]),
            phi=np.array([0.0, 2.0, 0.0]),
            radii=(1,),
        )
        self.assertEqual(result.n_nodes, 2)
        self.assertIsNone(result.y_star)
        self.assertEqual(result.funnels[1].n_funnels, 1)
        self.assertEqual(result.funnels[1].n_global_sinks, 1)

    def test_analysis_local_f_from_utils_not_a_packed_y_column(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (2,))
        y = np.array([5.0, 0.0, 1.0])
        utils = np.array([[0.5], [1.5], [0.2]])
        _v, phi, feasible = constraint_violations(utils)
        result = set_valued_colon(
            space=space, y_raw=y, feasible=feasible, phi=phi
        )
        self.assertTrue(bool(feasible[0]))
        self.assertFalse(bool(feasible[1]))
        self.assertEqual(result.y_star, 1.0)
        self.assertNotEqual(result.y_star, float(np.min(y)))

    def test_catalogue_y_phi_f_loader_matches_numpy_and_uses_fetchnumpy(self) -> None:
        from core.ioh_core.workflows.la.catalogue_table import load_catalogue_y_phi_f
        from core.ioh_core.workflows.la.deps import import_duckdb

        space = IndexSpace.from_bounds(("x",), (0,), (2,))
        y = np.array([5.0, 0.0, 1.0])
        utils = np.array([0.5, 1.5, 0.2])
        _v, phi_expected, f_expected = constraint_violations(utils)
        duckdb = import_duckdb()
        connection = duckdb.connect()
        try:
            connection.execute(
                "CREATE TABLE catalogue AS SELECT * FROM (VALUES "
                "(0, 5.0, 0.5), (1, 0.0, 1.5), (2, 1.0, 0.2)"
                ") t(idx__x, y, util__A_bending_capacity)"
            )
            y_raw, phi, feasible = load_catalogue_y_phi_f(
                connection,
                "catalogue",
                space,
                ("A_bending_capacity",),
                expected_n=3,
            )
        finally:
            connection.close()
        np.testing.assert_array_equal(y_raw, y)
        np.testing.assert_allclose(phi, phi_expected)
        np.testing.assert_array_equal(feasible, f_expected)


if __name__ == "__main__":
    unittest.main()
