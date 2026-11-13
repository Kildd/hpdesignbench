"""Deb / Fieldsend Eq. (6) comparator: F, y-on-F, phi-off-F, equal-phi ties."""
from __future__ import annotations

import inspect
import unittest

import numpy as np

from core.ioh_core.workflows.la.deb import deb_compare, deb_key
from core.ioh_core.workflows.la.preprocess import constraint_violations


class TestDebComparator(unittest.TestCase):
    def test_feasible_beats_infeasible_regardless_of_y_and_phi(self) -> None:
        self.assertEqual(
            deb_compare(
                feasible_a=True,
                y_a=100.0,
                phi_a=0.0,
                feasible_b=False,
                y_b=0.0,
                phi_b=0.001,
            ),
            -1,
        )
        self.assertEqual(
            deb_key(feasible=True, y=100.0, phi=0.0),
            (0, 100.0),
        )
        self.assertEqual(
            deb_key(feasible=False, y=0.0, phi=0.001),
            (1, 0.001),
        )

    def test_two_feasible_designs_order_by_y(self) -> None:
        self.assertEqual(
            deb_compare(
                feasible_a=True,
                y_a=1.0,
                phi_a=0.0,
                feasible_b=True,
                y_b=2.0,
                phi_b=0.0,
            ),
            -1,
        )

    def test_two_infeasible_designs_order_by_phi_only(self) -> None:
        self.assertEqual(
            deb_compare(
                feasible_a=False,
                y_a=0.0,
                phi_a=2.0,
                feasible_b=False,
                y_b=100.0,
                phi_b=1.0,
            ),
            1,
        )

    def test_equal_phi_infeasible_designs_are_tied_regardless_of_y(self) -> None:
        self.assertEqual(
            deb_compare(
                feasible_a=False,
                y_a=0.0,
                phi_a=1.5,
                feasible_b=False,
                y_b=99.0,
                phi_b=1.5,
            ),
            0,
        )
        self.assertEqual(
            deb_key(feasible=False, y=0.0, phi=1.5),
            deb_key(feasible=False, y=99.0, phi=1.5),
        )

    def test_linear_id_is_not_part_of_h(self) -> None:
        for name in ("linear_id", "linear", "index"):
            self.assertNotIn(name, inspect.signature(deb_key).parameters)
            self.assertNotIn(name, inspect.signature(deb_compare).parameters)
        self.assertEqual(
            deb_compare(
                feasible_a=False,
                y_a=1.0,
                phi_a=3.0,
                feasible_b=False,
                y_b=2.0,
                phi_b=3.0,
            ),
            0,
        )

    def test_phi_and_f_match_constraint_violations_on_keep_set(self) -> None:
        # Same selected-util recipe as GBLT / constrained landscape analysis:
        # v_i = max(u_i-1, 0) on the keep columns only (third column dropped).
        util_all = np.array(
            [
                [0.5, 0.9, 2.0],
                [1.2, 0.8, 0.0],
                [1.1, 1.4, 0.0],
            ],
            dtype=float,
        )
        keep = util_all[:, :2]
        y = np.array([10.0, 1.0, 0.0])
        _violation, phi, feasible = constraint_violations(keep)

        self.assertEqual(feasible.tolist(), [True, False, False])
        np.testing.assert_allclose(phi, [0.0, 0.2, 0.5])
        self.assertEqual(
            deb_compare(
                feasible_a=bool(feasible[0]),
                y_a=float(y[0]),
                phi_a=float(phi[0]),
                feasible_b=bool(feasible[1]),
                y_b=float(y[1]),
                phi_b=float(phi[1]),
            ),
            -1,
        )
        self.assertEqual(
            deb_compare(
                feasible_a=bool(feasible[1]),
                y_a=float(y[1]),
                phi_a=float(phi[1]),
                feasible_b=bool(feasible[2]),
                y_b=float(y[2]),
                phi_b=float(phi[2]),
            ),
            -1,
        )


if __name__ == "__main__":
    unittest.main()
