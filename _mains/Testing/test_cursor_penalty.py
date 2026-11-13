"""
Additive violation-power penalty regressions (mirrors analysis.py inline block).

    violation_i = max(0, η_i − 1)
    penalty_i   = α · violation_i^β
    y_p         = y · (1 + Σ penalty_i)

over active constraints only (Z included when active).
Default catalogue uses α = 1 and β = 1/3.
"""
from __future__ import annotations

import math
import unittest


def _penalty_inline(
    y: float,
    constraint_values: dict[str, float],
    constraints: dict[str, dict],
    *,
    alpha: float = 1.0,
    beta: float = 1.0 / 3.0,
) -> tuple[float, dict[str, float], dict[str, float], float]:
    """Replica of the penalty loop in hp_slab.analysis()."""
    violations_: dict[str, float] = {}
    penalties_: dict[str, float] = {}
    penalty_sum = 0.0
    for name, eta in constraint_values.items():
        meta = constraints.get(name)
        if meta is None or not meta.get("active", True):
            continue
        violation = max(0.0, float(eta) - 1.0)
        penalty = float(alpha) * (violation ** float(beta))
        violations_[name] = violation
        penalties_[name] = penalty
        penalty_sum += penalty
    return y * (1.0 + penalty_sum), violations_, penalties_, penalty_sum


class TestAdditiveViolationPowerPenalty(unittest.TestCase):
    """Assert violation-power penalty semantics."""

    def test_all_satisfied_leaves_objective_unchanged(self) -> None:
        y = 12.5
        values = {
            "A_bending_capacity": 0.8,
            "Z2_beam_theory_H_L_capacity": 1.0,
        }
        constraints = {name: {"active": True} for name in values}
        y_p, violations, penalties, psum = _penalty_inline(y, values, constraints)
        self.assertEqual(
            violations,
            {"A_bending_capacity": 0.0, "Z2_beam_theory_H_L_capacity": 0.0},
        )
        self.assertEqual(
            penalties,
            {"A_bending_capacity": 0.0, "Z2_beam_theory_H_L_capacity": 0.0},
        )
        self.assertAlmostEqual(psum, 0.0)
        self.assertAlmostEqual(y_p, y)

    def test_single_unit_violation_with_beta_one(self) -> None:
        y = 10.0
        values = {"A_bending_capacity": 2.0}
        constraints = {"A_bending_capacity": {"active": True}}
        y_p, violations, penalties, _ = _penalty_inline(
            y, values, constraints, beta=1.0
        )
        self.assertAlmostEqual(violations["A_bending_capacity"], 1.0)
        self.assertAlmostEqual(penalties["A_bending_capacity"], 1.0)
        self.assertAlmostEqual(y_p, 20.0)

    def test_default_beta_one_third_on_small_violation(self) -> None:
        y = 10.0
        values = {"A_bending_capacity": 1.0018819653072522}
        constraints = {"A_bending_capacity": {"active": True}}
        y_p, _, penalties, _ = _penalty_inline(y, values, constraints)
        v = 0.0018819653072522
        expected = v ** (1.0 / 3.0)
        self.assertAlmostEqual(penalties["A_bending_capacity"], expected)
        self.assertAlmostEqual(y_p, y * (1.0 + expected))

    def test_inactive_constraints_are_excluded(self) -> None:
        y = 10.0
        values = {
            "A_bending_capacity": 2.0,
            "inactive_check": 5.0,
        }
        constraints = {
            "A_bending_capacity": {"active": True},
            "inactive_check": {"active": False},
        }
        y_p, violations, _, _ = _penalty_inline(
            y, values, constraints, beta=1.0
        )
        self.assertNotIn("inactive_check", violations)
        self.assertAlmostEqual(y_p, 20.0)

    def test_z_violations_are_included_when_active(self) -> None:
        y = 10.0
        values = {
            "A_bending_capacity": 1.0,
            "Z2_beam_theory_H_L_capacity": 1.5,
        }
        constraints = {name: {"active": True} for name in values}
        y_p, violations, penalties, _ = _penalty_inline(
            y, values, constraints, beta=1.0
        )
        self.assertIn("Z2_beam_theory_H_L_capacity", violations)
        self.assertAlmostEqual(penalties["Z2_beam_theory_H_L_capacity"], 0.5)
        self.assertAlmostEqual(y_p, 10.0 * (1.0 + 0.5))

    def test_alpha_beta_shape(self) -> None:
        y = 1.0
        values = {"A_bending_capacity": 5.0}
        constraints = {"A_bending_capacity": {"active": True}}
        y_p, _, penalties, _ = _penalty_inline(
            y, values, constraints, alpha=2.0, beta=2.0
        )
        # v=4, p = 2 * 4^2 = 32
        self.assertAlmostEqual(penalties["A_bending_capacity"], 32.0)
        self.assertAlmostEqual(y_p, 33.0)


if __name__ == "__main__":
    unittest.main()
