"""
C1.1 gate: SALib Sobol analyse wrappers and packing.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.ioh_core.workflows.sensitivity_analysis.deps import (
    SensitivityDependencyError,
    import_salib,
)
from core.ioh_core.workflows.sensitivity_analysis.design import (
    sample_saltelli_unit_matrix,
)
from core.ioh_core.workflows.sensitivity_analysis.estimators import (
    estimate_sobol_indices,
    pack_saltelli_responses,
)


def _salib_available() -> bool:
    try:
        import_salib()
    except SensitivityDependencyError:
        return False
    return True


def _evaluate_packed(
    packed_unit: np.ndarray,
    function,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor_count = packed_unit.shape[1]
    step = factor_count + 2
    sample_size = packed_unit.shape[0] // step
    y_all = np.asarray(
        [function(row) for row in packed_unit],
        dtype=float,
    )
    y_a = y_all[0::step]
    y_b = y_all[(step - 1)::step]
    y_ab = np.column_stack(
        [y_all[(index + 1)::step] for index in range(factor_count)]
    )
    assert y_a.shape == (sample_size,)
    assert y_b.shape == (sample_size,)
    assert y_ab.shape == (sample_size, factor_count)
    return y_a, y_b, y_ab


def _sobol_g_analytic(
    a_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    variances = 1.0 / (3.0 * (1.0 + a_coefficients) ** 2)
    total_variance = float(np.prod(1.0 + variances) - 1.0)
    first_order = variances / total_variance
    total_effect = np.empty_like(first_order)
    for index in range(a_coefficients.size):
        others = np.prod(1.0 + np.delete(variances, index)) - 1.0
        total_effect[index] = 1.0 - others / total_variance
    return first_order, total_effect


@unittest.skipUnless(_salib_available(), 'install with: pip install -e ".[sensitivity]"')
class TestSensitivityEstimators(unittest.TestCase):
    """Packing order and SALib analyse wrappers."""

    def test_pack_saltelli_responses_order(self) -> None:
        y_a = np.array([10.0, 20.0])
        y_b = np.array([11.0, 21.0])
        y_ab = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        packed = pack_saltelli_responses(y_a, y_b, y_ab)
        expected = np.array(
            [
                10.0, 1.0, 2.0, 3.0, 11.0,
                20.0, 4.0, 5.0, 6.0, 21.0,
            ]
        )
        np.testing.assert_array_equal(packed, expected)

    def test_wrong_ab_then_packing_breaks_salib_match(self) -> None:
        """If Y were packed A,B,AB... SALib analyse would disagree."""
        coeffs = np.array([1.0, 2.0, 3.0])

        def additive(row: np.ndarray) -> float:
            return float(np.dot(coeffs, row))

        packed_unit = sample_saltelli_unit_matrix(
            sample_size=64,
            factor_count=3,
            scramble=True,
            seed=3,
        )
        y_a, y_b, y_ab = _evaluate_packed(packed_unit, additive)
        correct = estimate_sobol_indices(
            y_a,
            y_b,
            y_ab,
            compute_bootstrap_ci=False,
            num_resamples=8,
            conf_level=0.95,
            seed=1,
        )

        # Deliberately wrong pack: A, B, then AB columns.
        wrong = np.empty((y_a.size, 3 + 2), dtype=float)
        wrong[:, 0] = y_a
        wrong[:, 1] = y_b
        wrong[:, 2:] = y_ab
        from SALib.analyze import sobol as salib_sobol_analyze
        from core.ioh_core.workflows.sensitivity_analysis.design import (
            make_unit_problem,
        )

        wrong_result = salib_sobol_analyze.analyze(
            make_unit_problem(3),
            wrong.reshape(-1),
            calc_second_order=False,
            num_resamples=8,
            conf_level=0.95,
            print_to_console=False,
            parallel=False,
            seed=1,
        )
        self.assertFalse(
            np.allclose(correct.first_order, wrong_result["S1"], atol=1e-8)
        )

    def test_constant_output_fails_closed(self) -> None:
        y_a = np.ones(4)
        y_b = np.ones(4)
        y_ab = np.ones((4, 2))
        from core.ioh_core.workflows.sensitivity_analysis.estimators import (
            ConstantResponseError,
        )

        with self.assertRaisesRegex(ConstantResponseError, "zero range"):
            estimate_sobol_indices(
                y_a,
                y_b,
                y_ab,
                compute_bootstrap_ci=False,
                num_resamples=4,
                conf_level=0.95,
                seed=1,
            )

    def test_nonfinite_output_fails_closed(self) -> None:
        y_a = np.array([1.0, 2.0, 3.0, 4.0])
        y_b = np.array([1.0, 2.0, np.nan, 4.0])
        y_ab = np.arange(8, dtype=float).reshape(4, 2)
        with self.assertRaisesRegex(ValueError, "finite"):
            estimate_sobol_indices(
                y_a,
                y_b,
                y_ab,
                compute_bootstrap_ci=False,
                num_resamples=4,
                conf_level=0.95,
                seed=1,
            )

    def test_shape_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            pack_saltelli_responses(
                np.ones(3),
                np.ones(4),
                np.ones((4, 2)),
            )

    def test_ci_toggle_preserves_point_estimates(self) -> None:
        coeffs = np.array([1.5, -0.5, 2.0])

        def additive(row: np.ndarray) -> float:
            return float(np.dot(coeffs, row))

        packed_unit = sample_saltelli_unit_matrix(
            sample_size=32,
            factor_count=3,
            scramble=True,
            seed=9,
        )
        y_a, y_b, y_ab = _evaluate_packed(packed_unit, additive)
        shared = {
            "num_resamples": 32,
            "conf_level": 0.95,
            "seed": 123,
        }
        with_ci = estimate_sobol_indices(
            y_a,
            y_b,
            y_ab,
            compute_bootstrap_ci=True,
            **shared,
        )
        without_ci = estimate_sobol_indices(
            y_a,
            y_b,
            y_ab,
            compute_bootstrap_ci=False,
            **shared,
        )
        np.testing.assert_allclose(
            with_ci.first_order, without_ci.first_order, atol=0.0, rtol=0.0
        )
        np.testing.assert_allclose(
            with_ci.total_effect, without_ci.total_effect, atol=0.0, rtol=0.0
        )
        self.assertIsNotNone(with_ci.first_order_conf_half_width)
        self.assertIsNotNone(with_ci.total_effect_conf_half_width)
        self.assertIsNone(without_ci.first_order_conf_half_width)
        self.assertIsNone(without_ci.total_effect_conf_half_width)

    def test_bootstrap_repeatability(self) -> None:
        rng = np.random.default_rng(0)
        y_a = rng.normal(size=16)
        y_b = rng.normal(size=16)
        y_ab = rng.normal(size=(16, 2))
        kwargs = {
            "compute_bootstrap_ci": True,
            "num_resamples": 40,
            "conf_level": 0.9,
            "seed": 77,
        }
        first = estimate_sobol_indices(y_a, y_b, y_ab, **kwargs)
        second = estimate_sobol_indices(y_a, y_b, y_ab, **kwargs)
        np.testing.assert_allclose(
            first.first_order_conf_half_width,
            second.first_order_conf_half_width,
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            first.total_effect_conf_half_width,
            second.total_effect_conf_half_width,
            atol=0.0,
            rtol=0.0,
        )

    def test_matches_direct_salib_analyze(self) -> None:
        rng = np.random.default_rng(4)
        y_a = rng.normal(size=16)
        y_b = rng.normal(size=16)
        y_ab = rng.normal(size=(16, 3))
        estimate = estimate_sobol_indices(
            y_a,
            y_b,
            y_ab,
            compute_bootstrap_ci=True,
            num_resamples=25,
            conf_level=0.95,
            seed=5,
        )
        from SALib.analyze import sobol as salib_sobol_analyze
        from core.ioh_core.workflows.sensitivity_analysis.design import (
            make_unit_problem,
        )

        direct = salib_sobol_analyze.analyze(
            make_unit_problem(3),
            pack_saltelli_responses(y_a, y_b, y_ab),
            calc_second_order=False,
            num_resamples=25,
            conf_level=0.95,
            print_to_console=False,
            parallel=False,
            seed=5,
        )
        np.testing.assert_allclose(estimate.first_order, direct["S1"])
        np.testing.assert_allclose(estimate.total_effect, direct["ST"])
        np.testing.assert_allclose(
            estimate.first_order_conf_half_width, direct["S1_conf"]
        )
        np.testing.assert_allclose(
            estimate.total_effect_conf_half_width, direct["ST_conf"]
        )

    def test_additive_analytic_indices(self) -> None:
        coeffs = np.array([1.0, 2.0, 3.0])
        analytic = (coeffs ** 2) / np.sum(coeffs ** 2)

        def additive(row: np.ndarray) -> float:
            return float(np.dot(coeffs, row))

        packed_unit = sample_saltelli_unit_matrix(
            sample_size=256,
            factor_count=3,
            scramble=True,
            seed=42,
        )
        y_a, y_b, y_ab = _evaluate_packed(packed_unit, additive)
        estimate = estimate_sobol_indices(
            y_a,
            y_b,
            y_ab,
            compute_bootstrap_ci=False,
            num_resamples=8,
            conf_level=0.95,
            seed=1,
        )
        # Locked initial green tolerances for scrambled N=256.
        np.testing.assert_allclose(
            estimate.first_order, analytic, atol=0.05, rtol=0.0
        )
        np.testing.assert_allclose(
            estimate.total_effect, analytic, atol=0.05, rtol=0.0
        )

    def test_sobol_g_function_analytic_indices(self) -> None:
        a_coefficients = np.array([0.0, 0.5, 3.0, 9.0])
        analytic_s1, analytic_st = _sobol_g_analytic(a_coefficients)

        def sobol_g(row: np.ndarray) -> float:
            terms = (np.abs(4.0 * row - 2.0) + a_coefficients) / (
                1.0 + a_coefficients
            )
            return float(np.prod(terms))

        packed_unit = sample_saltelli_unit_matrix(
            sample_size=512,
            factor_count=4,
            scramble=True,
            seed=17,
        )
        y_a, y_b, y_ab = _evaluate_packed(packed_unit, sobol_g)
        estimate = estimate_sobol_indices(
            y_a,
            y_b,
            y_ab,
            compute_bootstrap_ci=False,
            num_resamples=8,
            conf_level=0.95,
            seed=1,
        )
        # Locked initial green tolerances for scrambled N=512.
        np.testing.assert_allclose(
            estimate.first_order, analytic_s1, atol=0.08, rtol=0.0
        )
        np.testing.assert_allclose(
            estimate.total_effect, analytic_st, atol=0.08, rtol=0.0
        )

    def test_swapped_a_b_changes_estimates(self) -> None:
        coeffs = np.array([1.0, 0.0, 0.5])

        def additive(row: np.ndarray) -> float:
            return float(np.dot(coeffs, row))

        packed_unit = sample_saltelli_unit_matrix(
            sample_size=64,
            factor_count=3,
            scramble=True,
            seed=8,
        )
        y_a, y_b, y_ab = _evaluate_packed(packed_unit, additive)
        correct = estimate_sobol_indices(
            y_a,
            y_b,
            y_ab,
            compute_bootstrap_ci=False,
            num_resamples=8,
            conf_level=0.95,
            seed=1,
        )
        swapped = estimate_sobol_indices(
            y_b,
            y_a,
            y_ab,
            compute_bootstrap_ci=False,
            num_resamples=8,
            conf_level=0.95,
            seed=1,
        )
        self.assertFalse(np.allclose(correct.first_order, swapped.first_order))


if __name__ == "__main__":
    unittest.main()
