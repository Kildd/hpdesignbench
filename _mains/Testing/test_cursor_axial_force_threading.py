"""
Phase R5a regressions: external axial force n threading (not prestress).
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from core.analysis_core.checks.structural_checks import (
    DeflectionLimitByDeflectionCheckEC2004DE,
    FailureAnnouncementByDeflectionCheckEC2004DE,
)
from core.analysis_core.statics.constants import SystemType
from core.unit_core import kN_to_N
from slab_construction.slab_construction import AnalysisCache, SectionResultKey


class TestAxialForceCacheKeys(unittest.TestCase):
    def test_m_cr_cache_does_not_fall_back_across_n(self) -> None:
        cache = AnalysisCache()
        n_nonzero = kN_to_N(100.0)
        key_n = SectionResultKey("M_cr", 0.5, "SLS", n_nonzero)
        key_0 = SectionResultKey("M_cr", 0.5, "SLS", 0.0)
        cache.m_cr[key_n] = {"valid": True, "m_cr": 1.0, "n": n_nonzero}

        self.assertIs(cache.require_m_cr(key_n), cache.m_cr[key_n])
        with self.assertRaises(RuntimeError) as ctx:
            cache.require_m_cr(key_0)
        self.assertIn("Missing cached M_cr", str(ctx.exception))
        self.assertNotEqual(key_n, key_0)


class TestBChecksForwardAxialForce(unittest.TestCase):
    def test_b1a_passes_n_as_N_axial_N(self) -> None:
        slab = MagicMock()
        slab.slab.L = 8000.0
        loads = MagicMock()
        n = kN_to_N(-50.0)
        with patch(
            "core.analysis_core.checks.structural_checks.DeflectionCalculator.calculate_deflection_mm_EC",
            return_value=10.0,
        ) as calc:
            DeflectionLimitByDeflectionCheckEC2004DE.calculate_utilization(
                slab,
                loads,
                system=SystemType.SIMPLE_BEAM,
                limit_factor=250.0,
                n=n,
            )
        self.assertEqual(calc.call_args.kwargs["N_axial_N"], n)

    def test_b2a_passes_n_as_N_axial_N(self) -> None:
        slab = MagicMock()
        slab.slab.L = 8000.0
        loads = MagicMock()
        n = kN_to_N(25.0)
        with patch(
            "core.analysis_core.checks.structural_checks.DeflectionCalculator.calculate_deflection_mm_EC",
            return_value=80.0,
        ) as calc:
            FailureAnnouncementByDeflectionCheckEC2004DE.calculate_utilization(
                slab,
                loads,
                system=SystemType.SIMPLE_BEAM,
                min_factor=100.0,
                n=n,
            )
        self.assertEqual(calc.call_args.kwargs["N_axial_N"], n)


class TestFullMkUsesAxialForceForCracking(unittest.TestCase):
    def test_full_mk_calls_m_cr_and_strain_profile_with_n(self) -> None:
        from core.analysis_core import section_methods as sm

        n = kN_to_N(40.0)
        section = MagicMock()
        section.section_calculator.calculate_strain_profile.return_value = MagicMock(
            chi_y=1e-5
        )
        section.geometry.point_geometries = []

        mk_results = MagicMock()
        mk_results.m_y = MagicMock()
        mk_results.chi_y = MagicMock()
        # Avoid numpy abs path by raising after the cracking setup is verified
        section.section_calculator.calculate_moment_curvature.side_effect = RuntimeError(
            "stop-after-cracking-setup"
        )

        with patch.object(
            sm,
            "calculate_bending_strength_Nmm_EC",
            return_value={
                "valid": True,
                "m_u": -1.0e7,
                "strain_profile": [0.0, 1e-4, 0.0],
                "reason": None,
            },
        ), patch.object(
            sm,
            "calculate_cracking_moment_sls_Nmm_EC",
            return_value={"valid": True, "m_cr": 2.5e7},
        ) as m_cr:
            with self.assertRaises(RuntimeError):
                sm._full_moment_curvature_method(section, n=n, m_k_num_points=5)

        m_cr.assert_called_once_with(section, n=n)
        section.section_calculator.calculate_strain_profile.assert_called_once_with(
            n, 2.5e7, 0
        )


if __name__ == "__main__":
    unittest.main()
