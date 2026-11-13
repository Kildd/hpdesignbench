"""
Coordinated simplified M–κ topology: softening extras and yield keypoints.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from core.analysis_core.section_methods import (
    InvalidSectionForMKError,
    SimplifiedMKParts,
    _assemble_simplified_mk_results,
    _build_simplified_mk_parts,
    _compute_simplified_yield_point,
    prepare_coordinated_simplified_mk_cache_sls_EC,
)


def _parts_stub(*, section=None, kappa_yield=None, M_yield=None, softening=False):
    if section is None:
        section = MagicMock()
        section.section_calculator.calculate_moment_curvature.return_value = MagicMock(
            m_y=np.array([]), chi_y=np.array([])
        )
    return SimplifiedMKParts(
        section=section,
        n=0.0,
        kappa_0=0.0,
        kappa_cr=-1e-5,
        kappa_eoc=-2e-5,
        kappa_u=-3e-5,
        M_cr_Nmm=-1e6,
        M_eoc_Nmm=-2e6,
        M_u_Nmm=-3e6,
        kappa_extra_cr_eoc=np.array([], dtype=float),
        M_extra_cr_eoc_Nmm=np.array([], dtype=float),
        kappa_extra_eoc_u=np.array([], dtype=float),
        M_extra_eoc_u_Nmm=np.array([], dtype=float),
        kappa_yield=list(kappa_yield or []),
        M_yield_Nmm=list(M_yield or []),
        softening_detected=softening,
    )


class TestComputeSimplifiedYieldPoint(unittest.TestCase):
    def test_no_yield_when_fyk_equals_ftk(self) -> None:
        section = MagicMock()
        reinf = MagicMock(fyk=2800.0, ftk=2800.0)
        with patch(
            "core.analysis_core.section_methods.get_reinforcement",
            return_value=(reinf, 0.0),
        ):
            kappa, M = _compute_simplified_yield_point(section, require=False)
        self.assertEqual(kappa, [])
        self.assertEqual(M, [])

    def test_require_raises_when_fyk_equals_ftk(self) -> None:
        section = MagicMock()
        reinf = MagicMock(fyk=2800.0, ftk=2800.0)
        with patch(
            "core.analysis_core.section_methods.get_reinforcement",
            return_value=(reinf, 0.0),
        ):
            with self.assertRaises(ValueError) as ctx:
                _compute_simplified_yield_point(section, require=True)
        self.assertIn("fyk == ftk", str(ctx.exception))


class TestAssembleYieldAffectsLength(unittest.TestCase):
    def test_yield_changes_assembled_point_count(self) -> None:
        without = _assemble_simplified_mk_results(_parts_stub())
        with_yield = _assemble_simplified_mk_results(
            _parts_stub(kappa_yield=[-2.5e-5], M_yield=[-2.5e6])
        )
        self.assertEqual(len(without.m_y) + 1, len(with_yield.m_y))


class TestCollapsedEocTopology(unittest.TestCase):
    def test_collapsed_eoc_pins_requested_extras_to_cracking(self) -> None:
        class _Law:
            eps_F_t = 1.0e-4

        law = _Law()
        concrete = MagicMock(constitutive_law=law)
        section = MagicMock()
        section.section_calculator.calculate_moment_curvature.side_effect = (
            AssertionError("collapsed cr–EOC extras must not invoke the solver")
        )

        with patch(
            "core.analysis_core.section_methods.TensionStiffeningConcreteLawEC",
            _Law,
        ), patch(
            "core.analysis_core.section_methods.get_concrete",
            return_value=concrete,
        ), patch(
            "core.analysis_core.section_methods.calculate_cracking_moment_sls_Nmm_EC",
            return_value={
                "valid": True,
                "m_cr": -1.0e6,
                "strain_profile": (0.0, -1.0e-5, 0.0),
            },
        ), patch(
            "core.analysis_core.section_methods.calculate_bending_strength_Nmm_EC",
            return_value={},
        ), patch(
            "core.analysis_core.section_methods._validate_ultimate_mk_terminal",
            return_value=(-3.0e6, -3.0e-5),
        ), patch(
            "core.analysis_core.section_methods._calculate_kappa_0",
            return_value=0.0,
        ), patch(
            "core.analysis_core.section_methods._calculate_section_state_from_bottom_strain_sls",
            return_value={"valid": False, "reason": "no equilibrium"},
        ), patch(
            "core.analysis_core.section_methods._compute_simplified_yield_point",
            return_value=([], []),
        ), patch(
            "core.analysis_core.section_methods.warnings.warn",
        ) as warn:
            parts = _build_simplified_mk_parts(
                section,
                simplification=[0.25, 0.50, 0.75],
                m_k_num_points=40,
            )

        warn.assert_called_once()
        self.assertIs(warn.call_args.args[1], RuntimeWarning)
        np.testing.assert_array_equal(
            parts.kappa_extra_cr_eoc,
            np.full(3, parts.kappa_cr),
        )
        np.testing.assert_array_equal(
            parts.M_extra_cr_eoc_Nmm,
            np.full(3, parts.M_cr_Nmm),
        )
        section.section_calculator.calculate_moment_curvature.assert_not_called()
        section.section_calculator.calculate_moment_curvature.side_effect = None
        section.section_calculator.calculate_moment_curvature.return_value = MagicMock(
            m_y=np.array([]), chi_y=np.array([])
        )
        self.assertEqual(len(_assemble_simplified_mk_results(parts).m_y), 7)


class TestCoordinateYieldTopology(unittest.TestCase):
    def test_mixed_yield_is_filled_on_all_sections(self) -> None:
        sec0 = MagicMock(name="sec0")
        sec1 = MagicMock(name="sec1")
        for sec in (sec0, sec1):
            sec.section_calculator.calculate_moment_curvature.return_value = MagicMock(
                m_y=np.array([]), chi_y=np.array([])
            )

        parts0 = _parts_stub(
            section=sec0, kappa_yield=[-2.5e-5], M_yield=[-2.5e6]
        )
        parts1 = _parts_stub(section=sec1)

        def _fake_build(section, **kwargs):
            return parts0 if section is sec0 else parts1

        with patch(
            "core.analysis_core.section_methods.sls_section_EC",
            side_effect=lambda section, _law: section,
        ), patch(
            "core.analysis_core.section_methods._build_simplified_mk_parts",
            side_effect=_fake_build,
        ), patch(
            "core.analysis_core.section_methods._compute_simplified_yield_point",
            return_value=([-2.4e-5], [-2.4e6]),
        ) as compute_yield:
            curves, _, _ = prepare_coordinated_simplified_mk_cache_sls_EC(
                {0.0: sec0, 0.5: sec1},
                simplification=True,
                simplification_softening=None,
            )

        compute_yield.assert_called_once()
        self.assertEqual(len(curves[0.0].m_y), len(curves[0.5].m_y))
        self.assertEqual(
            len(curves[0.0].m_y),
            len(_assemble_simplified_mk_results(parts0).m_y),
        )

    def test_residual_length_mismatch_fails_as_invalid_section(self) -> None:
        sec0 = MagicMock(name="sec0")
        sec1 = MagicMock(name="sec1")
        parts0 = _parts_stub(section=sec0)
        parts1 = _parts_stub(section=sec1)
        parts0.kappa_extra_cr_eoc = np.array([-1.5e-5])
        parts0.M_extra_cr_eoc_Nmm = np.array([-1.5e6])

        with patch(
            "core.analysis_core.section_methods.sls_section_EC",
            side_effect=lambda section, _law: section,
        ), patch(
            "core.analysis_core.section_methods._build_simplified_mk_parts",
            side_effect=lambda section, **_kwargs: (
                parts0 if section is sec0 else parts1
            ),
        ):
            with self.assertRaisesRegex(
                InvalidSectionForMKError,
                "same point count",
            ):
                prepare_coordinated_simplified_mk_cache_sls_EC(
                    {0.0: sec0, 0.5: sec1},
                    simplification=True,
                    simplification_softening=None,
                )


if __name__ == "__main__":
    unittest.main()
