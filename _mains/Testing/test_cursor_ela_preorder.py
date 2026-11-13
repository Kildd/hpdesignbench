"""Nested best-exit basin engine on a declared total preorder (#56)."""
from __future__ import annotations

import unittest

import numpy as np

from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.la.deb import deb_key
from core.ioh_core.workflows.la.lon import set_valued_lon
from core.ioh_core.workflows.la.plateau_graph import (
    build_plateau_graph,
    dense_preorder_ranks,
    nested_best_exit_support,
)


class TestPreorderBasinEngine(unittest.TestCase):
    def test_float_ranks_match_ieee_y_p_plateaus_and_support(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (4,))
        values = np.array([0.0, 4.0, 6.0, 4.0, 2.0])
        from_values = build_plateau_graph(values, space)
        from_ranks = build_plateau_graph(dense_preorder_ranks(values), space)
        lon = set_valued_lon(space=space, y_p=values, radii=())

        np.testing.assert_array_equal(
            from_values.plateau_of_linear,
            from_ranks.plateau_of_linear,
        )
        np.testing.assert_array_equal(
            from_values.destination_ids,
            from_ranks.destination_ids,
        )
        support_r = nested_best_exit_support(from_ranks)
        support_v = nested_best_exit_support(from_values)
        np.testing.assert_array_equal(support_r[0], support_v[0])
        self.assertEqual(support_r[1], support_v[1])
        self.assertEqual(lon.support_for_linear(2), (0, 1))

    def test_deb_keys_do_not_use_a_packed_y_p_column(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (4,))
        feasible = np.array([True, False, False, False, True])
        phi = np.array([0.0, 1.0, 1.0, 2.0, 0.0])
        y = np.array([5.0, 0.0, 99.0, 0.0, 1.0])
        keys = np.array(
            [
                deb_key(feasible=bool(f), y=float(yi), phi=float(p))
                for f, yi, p in zip(feasible, y, phi, strict=True)
            ],
            dtype=np.float64,
        )
        ranks = dense_preorder_ranks(keys)
        self.assertEqual(int(ranks[1]), int(ranks[2]))
        packed = y
        packed_ranks = dense_preorder_ranks(packed)
        self.assertNotEqual(int(packed_ranks[1]), int(packed_ranks[2]))

        graph = build_plateau_graph(ranks, space)
        self.assertEqual(
            int(graph.plateau_of_linear[1]),
            int(graph.plateau_of_linear[2]),
        )
        single, multiple = nested_best_exit_support(graph)
        self.assertFalse(multiple)
        self.assertGreaterEqual(int(graph.n_nodes), 1)

    def test_equal_best_deb_exits_keep_all_branches_without_linear_id(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (4,))
        feasible = np.array([False, True, False, True, False])
        phi = np.array([3.0, 0.0, 2.0, 0.0, 3.0])
        y = np.array([9.0, 1.0, 8.0, 1.0, 9.0])
        keys = np.array(
            [
                deb_key(feasible=bool(f), y=float(yi), phi=float(p))
                for f, yi, p in zip(feasible, y, phi, strict=True)
            ],
            dtype=np.float64,
        )
        graph = build_plateau_graph(dense_preorder_ranks(keys), space)
        single, multiple = nested_best_exit_support(graph)
        pid_mid = int(graph.plateau_of_linear[2])
        self.assertIn(pid_mid, multiple)
        self.assertEqual(len(multiple[pid_mid]), 2)
        self.assertEqual(set(multiple[pid_mid]), {0, 1})


if __name__ == "__main__":
    unittest.main()
