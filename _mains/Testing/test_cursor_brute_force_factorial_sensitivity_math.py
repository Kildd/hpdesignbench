"""
C7.1 gate: exact finite-grid Sobol tensor oracle and response contracts.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.ioh_core.workflows.brute_force.analysis.factorial_sensitivity import (
    EXPRESSION_BINARY_FEASIBLE,
    EXPRESSION_COLUMN,
    METHOD_ID,
    OWEN_DOI,
    RESPONSE_FAMILY_FEASIBILITY,
    RESPONSE_FAMILY_OBJECTIVE,
    RESPONSE_FAMILY_UTILIZATION,
    VARIANCE_DDOF,
    build_factorial_response_specs,
    default_method_meta,
    exact_sobol_from_tensor,
    factor_results_from_tensor,
    materialize_response_values,
)


class TestFactorialSensitivityContracts(unittest.TestCase):
    """Method metadata and response-spec builders."""

    def test_default_method_meta_locks(self) -> None:
        meta = default_method_meta()
        self.assertEqual(meta.method_id, METHOD_ID)
        self.assertEqual(meta.method_id, "sobol_hoeffding_functional_anova")
        self.assertEqual(meta.variance_ddof, VARIANCE_DDOF)
        self.assertEqual(meta.variance_ddof, 0)
        self.assertEqual(meta.doi, OWEN_DOI)
        self.assertEqual(meta.doi, "10.1137/120876782")
        self.assertFalse(meta.publishes_sampling_ci)
        self.assertFalse(meta.publishes_second_order)

    def test_build_response_specs_order(self) -> None:
        specs = build_factorial_response_specs(["c1", "c2"])
        self.assertEqual(
            [spec.name for spec in specs],
            [
                "y_p",
                "util__c1",
                "feasible__c1",
                "util__c2",
                "feasible__c2",
            ],
        )
        self.assertEqual(specs[0].family, RESPONSE_FAMILY_OBJECTIVE)
        self.assertEqual(specs[0].expression, EXPRESSION_COLUMN)
        self.assertEqual(specs[1].family, RESPONSE_FAMILY_UTILIZATION)
        self.assertEqual(specs[2].family, RESPONSE_FAMILY_FEASIBILITY)
        self.assertEqual(specs[2].expression, EXPRESSION_BINARY_FEASIBLE)
        self.assertEqual(specs[2].source_column, "util__c1")

    def test_build_response_specs_rejects_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            build_factorial_response_specs(["c1", "c1"])

    def test_binary_feasibility_materialization(self) -> None:
        source = np.array([0.5, 1.0, 1.01, 1e6])
        binary = materialize_response_values(
            source,
            expression=EXPRESSION_BINARY_FEASIBLE,
        )
        np.testing.assert_array_equal(binary, np.array([1.0, 1.0, 0.0, 0.0]))


class TestExactSobolFromTensor(unittest.TestCase):
    """Analytic oracles for the small-grid reference implementation."""

    def test_rejects_constant_and_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero variance"):
            exact_sobol_from_tensor(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            exact_sobol_from_tensor(np.array(3.0))
        with self.assertRaises(ValueError):
            exact_sobol_from_tensor(np.array([1.0, np.nan]))

    def test_unequal_level_counts_additive_oracle(self) -> None:
        # Shape (2, 3), Y = 2*i + 5*j  => S1 = ST = (3/53, 50/53).
        grid = np.fromfunction(
            lambda i, j: 2.0 * i + 5.0 * j,
            (2, 3),
            dtype=float,
        )
        result = exact_sobol_from_tensor(grid)
        expected = np.array([3.0 / 53.0, 50.0 / 53.0])
        np.testing.assert_allclose(result.first_order, expected, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(result.total_effect, expected, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(
            result.total_effect_via_complement_mean,
            expected,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            result.identity_residual,
            np.zeros(2),
            rtol=0.0,
            atol=1e-15,
        )

    def test_additive_function_s1_equals_st(self) -> None:
        xs = np.array([-1.0, 0.0, 2.0])
        ys = np.array([-2.0, 5.0])
        zs = np.array([0.0, 1.0])
        grid = (
            xs[:, None, None]
            + 3.0 * ys[None, :, None]
            - 0.5 * zs[None, None, :]
        )
        result = exact_sobol_from_tensor(grid)
        np.testing.assert_allclose(
            result.first_order,
            result.total_effect,
            rtol=0.0,
            atol=1e-14,
        )
        self.assertAlmostEqual(float(np.sum(result.first_order)), 1.0, places=14)

    def test_centered_pure_interaction(self) -> None:
        # Y = X1 * X2 on {-1, +1}^2: S1 = 0, ST = 1 for both factors.
        xs = np.array([-1.0, 1.0])
        ys = np.array([-1.0, 1.0])
        grid = xs[:, None] * ys[None, :]
        result = exact_sobol_from_tensor(grid)
        np.testing.assert_allclose(
            result.first_order,
            np.zeros(2),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            result.total_effect,
            np.ones(2),
            rtol=0.0,
            atol=1e-15,
        )

    def test_mixed_main_interaction_oracle(self) -> None:
        # Y = x + 2y + 3xy on {-1,+1}^2
        # => S1 = (1/14, 4/14), ST = (10/14, 13/14).
        xs = np.array([-1.0, 1.0])
        ys = np.array([-1.0, 1.0])
        grid = np.empty((2, 2), dtype=float)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                grid[i, j] = x + 2.0 * y + 3.0 * x * y
        result = exact_sobol_from_tensor(grid)
        expected_s1 = np.array([1.0 / 14.0, 4.0 / 14.0])
        expected_st = np.array([10.0 / 14.0, 13.0 / 14.0])
        np.testing.assert_allclose(
            result.first_order,
            expected_s1,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            result.total_effect,
            expected_st,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            result.total_effect_via_complement_mean,
            expected_st,
            rtol=0.0,
            atol=1e-15,
        )

    def test_irrelevant_variable_has_zero_indices(self) -> None:
        xs = np.array([-1.0, 1.0])
        ys = np.array([-1.0, 1.0])
        zs = np.array([-1.0, 1.0])
        grid = np.empty((2, 2, 2), dtype=float)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, _z in enumerate(zs):
                    grid[i, j, k] = x + x * y
        result = exact_sobol_from_tensor(grid)
        self.assertAlmostEqual(float(result.first_order[2]), 0.0, places=15)
        self.assertAlmostEqual(float(result.total_effect[2]), 0.0, places=15)
        self.assertAlmostEqual(float(result.first_order[0]), 0.5, places=15)
        self.assertAlmostEqual(float(result.total_effect[0]), 1.0, places=15)
        self.assertAlmostEqual(float(result.first_order[1]), 0.0, places=15)
        self.assertAlmostEqual(float(result.total_effect[1]), 0.5, places=15)

    def test_binary_threshold_response(self) -> None:
        util = np.array(
            [
                [0.5, 1.0, 1.5],
                [0.8, 2.0, 0.2],
            ],
            dtype=float,
        )
        binary = materialize_response_values(
            util,
            expression=EXPRESSION_BINARY_FEASIBLE,
        )
        result = exact_sobol_from_tensor(binary)
        # Four of six cells are feasible => non-constant binary response.
        self.assertGreater(result.total_variance, 0.0)
        self.assertEqual(result.factor_count, 2)
        np.testing.assert_allclose(
            result.total_effect,
            result.total_effect_via_complement_mean,
            rtol=0.0,
            atol=1e-15,
        )
        # First-order effects remain inside [0, 1] and ST >= S1 numerically.
        self.assertTrue(np.all(result.first_order >= -1e-15))
        self.assertTrue(np.all(result.first_order <= 1.0 + 1e-15))
        self.assertTrue(
            np.all(result.total_effect + 1e-15 >= result.first_order)
        )

    def test_shuffled_axis_order_matches_named_factors(self) -> None:
        # Same mixed oracle with axes swapped: names must track axis meaning.
        xs = np.array([-1.0, 1.0])
        ys = np.array([-1.0, 1.0])
        grid_xy = np.empty((2, 2), dtype=float)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                grid_xy[i, j] = x + 2.0 * y + 3.0 * x * y
        grid_yx = np.transpose(grid_xy, (1, 0))

        named_xy = factor_results_from_tensor(
            grid_xy,
            variable_names=("x", "y"),
        )
        named_yx = factor_results_from_tensor(
            grid_yx,
            variable_names=("y", "x"),
        )
        by_xy = {item.variable: item for item in named_xy}
        by_yx = {item.variable: item for item in named_yx}
        self.assertAlmostEqual(
            by_xy["x"].first_order,
            by_yx["x"].first_order,
            places=15,
        )
        self.assertAlmostEqual(
            by_xy["y"].total_effect,
            by_yx["y"].total_effect,
            places=15,
        )
        self.assertAlmostEqual(by_xy["x"].first_order, 1.0 / 14.0, places=15)
        self.assertAlmostEqual(by_xy["y"].first_order, 4.0 / 14.0, places=15)

    def test_factor_results_validate_names(self) -> None:
        with self.assertRaises(ValueError):
            factor_results_from_tensor(
                np.array([[0.0, 1.0], [2.0, 3.0]]),
                variable_names=("only_one",),
            )
        with self.assertRaises(ValueError):
            factor_results_from_tensor(
                np.array([[0.0, 1.0], [2.0, 3.0]]),
                variable_names=("a", "a"),
            )


if __name__ == "__main__":
    unittest.main()
