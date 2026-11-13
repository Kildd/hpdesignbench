"""
Phase R5d: κ₀ is an intentional linearized initial-tangent estimate.

Compares production ``_calculate_kappa_0`` to the Loutfi closed form and to
a nonlinear strain-profile solve, and checks that M–κ diagrams reuse the
same κ₀ as the first stored point.
"""
from __future__ import annotations

import unittest

import numpy as np

from _mains.Testing.fixtures import hp_sections as hs
from core.analysis_core.section_methods import (
    DEFLECTION_MK_SIMPLIFICATION,
    DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
    _calculate_kappa_0,
    calculate_cracking_moment_sls_Nmm_EC,
    calculate_moment_curvature_sls_EC,
    calculate_prestress_forces_Nmm,
    sls_section_EC,
)


def _loutfi_kappa_0_1_per_mm(section, n: float = 0.0) -> float | None:
    """Loutfi estimate κ₀ = (M_p · κ_cr) / (M_cr - M_p) [1/mm]."""
    M_p, _ = calculate_prestress_forces_Nmm(section)
    m_cr = calculate_cracking_moment_sls_Nmm_EC(
        sls_section_EC(section, "FCTM_PARABOLIC"), n=n
    )
    if not m_cr.get("valid", True):
        return None
    M_cr = abs(m_cr["m_cr"])
    kappa_cr = abs(m_cr["strain_profile"][1])
    if abs(M_cr - M_p) <= 1e-3:
        return 0.0
    return (M_p * kappa_cr) / (M_cr - M_p)


def _linear_solve_residual(sls_sec, n: float = 0.0) -> float:
    """||response_0 + K_0·Δε - target||_∞ for the production κ₀ solve."""
    target = np.array([n, 0.0, 0.0], dtype=float)
    zero = np.zeros(3, dtype=float)
    calc = sls_sec.section_calculator
    response_0 = calc.integrate_strain_profile(zero, integrate="stress").asarray()
    stiffness_0 = calc.integrate_strain_profile(zero, integrate="modulus").asarray()
    delta = np.linalg.solve(stiffness_0, target - response_0)
    return float(np.max(np.abs(response_0 + stiffness_0 @ delta - target)))


class TestKappa0Contract(unittest.TestCase):
    def test_non_prestressed_returns_zero(self) -> None:
        sls = sls_section_EC(hs.hp_section_c1_3_uls, "TENSTIFF_PARABOLIC")
        self.assertEqual(_calculate_kappa_0(sls), 0.0)

    def test_non_prestressed_full_mk_does_not_prepend_kappa0(self) -> None:
        full = calculate_moment_curvature_sls_EC(
            hs.hp_section_c1_3_uls,
            constitutive_law="TENSTIFF_PARABOLIC",
            simplification=False,
            m_k_num_points=40,
        )
        # 40 linspace + cracking insert; no (κ₀, M=0) prepend when M_p = 0
        self.assertEqual(len(full.m_y), 41)

    def test_production_matches_simplified_mk_first_point(self) -> None:
        sec = hs.hp_section_c1_1_uls
        sls = sls_section_EC(sec, "TENSTIFF_PARABOLIC")
        kappa_0 = _calculate_kappa_0(sls)
        mk = calculate_moment_curvature_sls_EC(
            sec,
            constitutive_law="TENSTIFF_PARABOLIC",
            simplification=DEFLECTION_MK_SIMPLIFICATION,
            simplification_softening=DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
            m_k_num_points=40,
        )
        self.assertAlmostEqual(float(mk.chi_y[0]), kappa_0)
        self.assertAlmostEqual(float(mk.m_y[0]), 0.0)

    def test_linear_solve_residual_is_tiny(self) -> None:
        sls = sls_section_EC(hs.hp_section_c2_uls_x_0_50, "TENSTIFF_PARABOLIC")
        self.assertLess(_linear_solve_residual(sls), 1e-6)


class TestKappa0ApproximationComparisons(unittest.TestCase):
    """Quantify linearized κ₀ vs Loutfi and vs nonlinear equilibrium."""

    def test_close_to_loutfi_estimate_on_reference_sections(self) -> None:
        # Documented tolerance: linearized tangent vs Loutfi closed form.
        cases = (
            ("C1_1", hs.hp_section_c1_1_uls, 0.02),
            ("C2_x_0_50", hs.hp_section_c2_uls_x_0_50, 0.05),
        )
        for name, sec, rtol in cases:
            with self.subTest(section=name):
                sls = sls_section_EC(sec, "TENSTIFF_PARABOLIC")
                prod = _calculate_kappa_0(sls)
                loutfi = _loutfi_kappa_0_1_per_mm(sec)
                self.assertIsNotNone(loutfi)
                self.assertGreater(abs(loutfi), 0.0)
                rel = abs(prod - loutfi) / abs(loutfi)
                self.assertLessEqual(
                    rel,
                    rtol,
                    msg=f"{name}: |prod-loutfi|/|loutfi|={rel:.4f} > {rtol}",
                )

    def test_nonlinear_equilibrium_may_differ_substantially(self) -> None:
        """Guardrail: nonlinear solve exists and can diverge from κ₀.

        R5d keeps the linearized estimate; this only asserts that a
        converged nonlinear reference is available and that C1.4 still
        shows a large relative gap (so we do not silently 'fix' κ₀ by
        switching to nonlinear without noticing).
        """
        sls = sls_section_EC(hs.hp_section_c1_4_uls, "TENSTIFF_PARABOLIC")
        prod = _calculate_kappa_0(sls)
        sp = sls.section_calculator.calculate_strain_profile(
            0.0, 0.0, 0.0, max_iter=100
        )
        self.assertTrue(sp.converged)
        rel = abs(prod - sp.chi_y) / abs(sp.chi_y)
        self.assertGreater(
            rel,
            0.10,
            msg="Expected C1.4 linearized vs nonlinear gap > 10%",
        )


if __name__ == "__main__":
    unittest.main()
