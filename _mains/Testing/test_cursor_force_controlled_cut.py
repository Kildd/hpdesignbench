"""
Force-controlled M–κ early-cut plan tests (steps 1–6).

Step 1/4 — ``_ensure_force_controlled`` cuts lasting terminal softening at
the real peak ``(M_peak, κ_peak)`` (no fake ``(M_peak, κ_u)`` tail).

Step 2 — shared ``M_u`` / ``κ_u`` terminal validation for full and simplified.

Step 3 — isolated ``_terminal_softening_prefix`` array helper.

Step 4 — cut-then-dip-fill pipeline inside ``_ensure_force_controlled``.

Step 5 — single-section full and simplified both use the same helper via
``calculate_moment_curvature_sls_EC``; full raw stays uncut for verification.

Step 6 — multi-section caches store raw equal-length curves; force-control
runs after local spatial interpolation in deflection.

Lookup / ``AboveMkCapacityError`` contracts stay in
``test_cursor_above_mk_capacity.py``.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from structuralcodes import set_design_code
from structuralcodes.core._section_results import MomentCurvatureResults

from core.analysis_core.section_methods import (
    DEFLECTION_MK_SIMPLIFICATION,
    DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
    AboveMkCapacityError,
    InvalidSectionForMKError,
    _add_softening_extras_to_simplified_mk_parts,
    _assemble_simplified_mk_results,
    _build_simplified_mk_parts,
    _ensure_force_controlled,
    _force_controlled_mk_arrays,
    _full_moment_curvature_method,
    _terminal_softening_prefix,
    _validate_ultimate_mk_terminal,
    calculate_moment_curvature_sls_EC,
    lookup_kappa_force_controlled,
    sls_section_EC,
)
from core.analysis_core.statics.deflection import DeflectionCalculator
from core.ioh_core.import_specs import load_materials_registry
from _mains.verification.plot_mk_height_reinf_matrix import (
    HP_DIR,
    MkCase,
    build_midspan_uls_section,
    compute_mk_pair,
)


# Known terminal-softening cases from the height matrix (a_tex=1.81, nt=5, kap=60%).
SOFTENING_CASES = (
    MkCase(100.0, 1.81, 5, 60.0),
    MkCase(300.0, 1.81, 5, 60.0),
    MkCase(600.0, 1.81, 5, 60.0),
)

# Relative margin: raw peak must clearly exceed |M_u| (not noise).
PEAK_OVER_MU_RTOL = 0.05


def _abs_arrays(results) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.abs(np.asarray(results.m_y, dtype=float)),
        np.asarray(results.chi_y, dtype=float),
    )


class TestEnsureForceControlledCutThenDipFill(unittest.TestCase):
    """Step 4 gate: terminal cut at real peak; recoverable dips still filled."""

    def test_terminal_softening_ends_at_real_peak(self) -> None:
        # Signed sagging moments (negative). Peak at index 2; terminal softer.
        m = np.array([0.0, -50.0, -100.0, -80.0, -60.0], dtype=float)
        k = np.array([0.0, 0.001, 0.002, 0.003, 0.004], dtype=float)
        results = MomentCurvatureResults(m_y=m.copy(), chi_y=k.copy())
        out = _ensure_force_controlled(results)

        self.assertEqual(len(out.m_y), 3)
        np.testing.assert_allclose(out.m_y, m[:3])
        np.testing.assert_allclose(out.chi_y, k[:3])
        self.assertAlmostEqual(float(out.m_y[-1]), -100.0)
        self.assertAlmostEqual(float(out.chi_y[-1]), 0.002)

    def test_recoverable_dip_still_monotonized_without_cut(self) -> None:
        m = np.array([0.0, -50.0, -40.0, -80.0], dtype=float)
        k = np.array([0.0, 0.001, 0.002, 0.003], dtype=float)
        results = MomentCurvatureResults(m_y=m.copy(), chi_y=k.copy())
        out = _ensure_force_controlled(results)
        self.assertEqual(len(out.m_y), len(m))
        # Dip at index 2 is lifted toward the preceding peak; final |M| is larger.
        self.assertGreaterEqual(abs(float(out.m_y[2])), 50.0 - 1e-9)
        self.assertAlmostEqual(float(out.m_y[-1]), -80.0)
        self.assertAlmostEqual(float(out.chi_y[-1]), 0.003)

    def test_cut_disabled_keeps_legacy_tail_lift(self) -> None:
        m = np.array([0.0, -50.0, -100.0, -80.0, -60.0], dtype=float)
        k = np.array([0.0, 0.001, 0.002, 0.003, 0.004], dtype=float)
        results = MomentCurvatureResults(m_y=m.copy(), chi_y=k.copy())
        out = _ensure_force_controlled(results, cut_terminal_softening=False)

        self.assertEqual(len(out.m_y), len(m))
        np.testing.assert_allclose(np.abs(out.m_y[2:]), 100.0)
        np.testing.assert_allclose(out.chi_y, k)
        self.assertAlmostEqual(float(out.m_y[-1]), -100.0)
        self.assertAlmostEqual(float(out.chi_y[-1]), 0.004)


class TestKnownSofteningCasesEarlyCut(unittest.TestCase):
    """Regression: raw early peak; FC now ends at (M_peak, κ_peak)."""

    @classmethod
    def setUpClass(cls) -> None:
        set_design_code("ec2_2004")
        cls.materials = load_materials_registry(HP_DIR / "materials.csv")

    def test_raw_peaks_before_kappa_u_and_fc_ends_at_peak(self) -> None:
        for case in SOFTENING_CASES:
            with self.subTest(case=case.label):
                section = build_midspan_uls_section(case, self.materials)
                pair = compute_mk_pair(section)
                self.assertIsNotNone(pair["full_raw"], pair["full_raw_error"])
                self.assertIsNotNone(pair["full"], pair["full_error"])

                m_raw, k_raw = _abs_arrays(pair["full_raw"])
                m_fc, k_fc = _abs_arrays(pair["full"])

                peak_i = int(np.argmax(m_raw))
                M_peak = float(m_raw[peak_i])
                M_u = float(m_raw[-1])
                k_peak = float(k_raw[peak_i])
                k_u = float(k_raw[-1])

                # Raw: genuine early peak with lasting terminal softening.
                self.assertLess(peak_i, len(m_raw) - 1, msg="peak must precede κ_u")
                self.assertGreater(
                    M_peak,
                    M_u * (1.0 + PEAK_OVER_MU_RTOL),
                    msg=f"expected clear peak>M_u; got {M_peak=}, {M_u=}",
                )
                self.assertNotAlmostEqual(k_peak, k_u, places=12)

                # FC: shortened to the real peak endpoint.
                self.assertEqual(len(m_fc), peak_i + 1)
                self.assertAlmostEqual(float(m_fc[-1]), M_peak, places=3)
                self.assertAlmostEqual(float(k_fc[-1]), k_peak, places=12)
                self.assertNotAlmostEqual(float(k_fc[-1]), k_u, places=12)

    def test_force_control_mutates_in_place_and_shortens_on_cut(self) -> None:
        case = SOFTENING_CASES[0]
        section = build_midspan_uls_section(case, self.materials)
        pair = compute_mk_pair(section)
        raw = pair["full_raw"]
        self.assertIsNotNone(raw)

        before_n = len(raw.m_y)
        peak_i = int(np.argmax(np.abs(np.asarray(raw.m_y, dtype=float))))
        clone = MomentCurvatureResults(
            m_y=np.asarray(raw.m_y, dtype=float).copy(),
            chi_y=np.asarray(raw.chi_y, dtype=float).copy(),
        )
        out = _ensure_force_controlled(clone)

        self.assertIs(out, clone)
        self.assertEqual(len(out.m_y), peak_i + 1)
        self.assertLess(len(out.m_y), before_n)
        # Raw from compute_mk_pair remains the unlifted full sweep.
        self.assertEqual(len(raw.m_y), before_n)


class TestSingleCurveFullAndSimplifiedEarlyCut(unittest.TestCase):
    """Step 5 gate: shared FC helper on single full and simplified curves."""

    @classmethod
    def setUpClass(cls) -> None:
        set_design_code("ec2_2004")
        cls.materials = load_materials_registry(HP_DIR / "materials.csv")

    def test_full_entry_cuts_like_helper_and_raw_stays_uncut(self) -> None:
        for case in SOFTENING_CASES:
            with self.subTest(case=case.label):
                section = build_midspan_uls_section(case, self.materials)
                sls = sls_section_EC(section, "TENSTIFF_PARABOLIC")
                raw = _full_moment_curvature_method(
                    section=sls, n=0.0, m_k_num_points=60
                )
                prod = calculate_moment_curvature_sls_EC(
                    section,
                    n=0.0,
                    constitutive_law="TENSTIFF_PARABOLIC",
                    simplification=False,
                    m_k_num_points=60,
                )

                m_raw = np.abs(np.asarray(raw.m_y, dtype=float))
                k_raw = np.asarray(raw.chi_y, dtype=float)
                peak_i = int(np.argmax(m_raw))
                self.assertLess(peak_i, len(m_raw) - 1)
                self.assertGreater(
                    float(m_raw[peak_i]),
                    float(m_raw[-1]) * (1.0 + PEAK_OVER_MU_RTOL),
                )

                # Raw verification path is not force-controlled / cut.
                self.assertEqual(len(raw.m_y), len(m_raw))
                self.assertNotAlmostEqual(
                    float(k_raw[peak_i]), float(k_raw[-1]), places=12
                )

                # Production full entry ends at the real peak.
                self.assertEqual(len(prod.m_y), peak_i + 1)
                self.assertAlmostEqual(
                    float(np.abs(prod.m_y[-1])), float(m_raw[peak_i]), places=3
                )
                self.assertAlmostEqual(
                    float(prod.chi_y[-1]), float(k_raw[peak_i]), places=12
                )

    def test_simplified_entry_cuts_assembled_terminal_softening(self) -> None:
        for case in SOFTENING_CASES:
            with self.subTest(case=case.label):
                section = build_midspan_uls_section(case, self.materials)
                sls = sls_section_EC(section, "TENSTIFF_PARABOLIC")
                parts = _build_simplified_mk_parts(
                    sls,
                    n=0.0,
                    simplification=DEFLECTION_MK_SIMPLIFICATION,
                    m_k_num_points=60,
                )
                if parts.softening_detected:
                    _add_softening_extras_to_simplified_mk_parts(
                        parts,
                        simplification_softening=DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
                    )
                assembled = _assemble_simplified_mk_results(parts)
                m_raw = np.abs(np.asarray(assembled.m_y, dtype=float))
                k_raw = np.asarray(assembled.chi_y, dtype=float)
                peak_i = int(np.argmax(m_raw))
                self.assertLess(
                    peak_i,
                    len(m_raw) - 1,
                    msg="expected simplified assembly terminal softening",
                )
                self.assertGreater(
                    float(m_raw[peak_i]),
                    float(m_raw[-1]) * (1.0 + PEAK_OVER_MU_RTOL),
                )

                prod = calculate_moment_curvature_sls_EC(
                    section,
                    n=0.0,
                    constitutive_law="TENSTIFF_PARABOLIC",
                    simplification=DEFLECTION_MK_SIMPLIFICATION,
                    simplification_softening=DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
                    m_k_num_points=60,
                )
                self.assertEqual(len(prod.m_y), peak_i + 1)
                self.assertAlmostEqual(
                    float(np.abs(prod.m_y[-1])), float(m_raw[peak_i]), places=3
                )
                self.assertAlmostEqual(
                    float(prod.chi_y[-1]), float(k_raw[peak_i]), places=12
                )

    def test_full_and_simplified_share_default_force_control_kwargs(self) -> None:
        """Both production variants must call FC without special cut flags."""
        section = build_midspan_uls_section(SOFTENING_CASES[0], self.materials)
        calls: list[dict] = []

        real_fc = _ensure_force_controlled

        def _spy(results, **kwargs):
            calls.append(dict(kwargs))
            return real_fc(results, **kwargs)

        with patch(
            "core.analysis_core.section_methods._ensure_force_controlled",
            side_effect=_spy,
        ):
            calculate_moment_curvature_sls_EC(
                section,
                simplification=False,
                m_k_num_points=60,
            )
            calculate_moment_curvature_sls_EC(
                section,
                simplification=DEFLECTION_MK_SIMPLIFICATION,
                simplification_softening=DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
                m_k_num_points=60,
            )

        self.assertEqual(len(calls), 2)
        for kwargs in calls:
            # Default cut path: no override that disables terminal softening.
            self.assertNotIn("cut_terminal_softening", kwargs)
            self.assertNotIn("atol", kwargs)
            self.assertNotIn("rtol", kwargs)


class TestValidateUltimateMkTerminal(unittest.TestCase):
    """Gate: invalid terminals raise InvalidSectionForMKError, never KeyError."""

    def test_accepts_finite_valid_terminal(self) -> None:
        M_u, kappa_u = _validate_ultimate_mk_terminal(
            {
                "valid": True,
                "m_u": -1.5e7,
                "strain_profile": [0.0, -2e-5, 0.0],
                "reason": None,
            },
            context="test diagram",
        )
        self.assertAlmostEqual(M_u, -1.5e7)
        self.assertAlmostEqual(kappa_u, -2e-5)

    def test_rejects_invalid_flag(self) -> None:
        with self.assertRaises(InvalidSectionForMKError) as ctx:
            _validate_ultimate_mk_terminal(
                {
                    "valid": False,
                    "m_u": None,
                    "strain_profile": None,
                    "reason": "cannot be taken by section",
                },
                context="full M-K diagram",
            )
        self.assertIn("bending strength invalid", str(ctx.exception))
        self.assertIn("cannot be taken by section", str(ctx.exception))

    def test_rejects_missing_m_u(self) -> None:
        with self.assertRaises(InvalidSectionForMKError) as ctx:
            _validate_ultimate_mk_terminal(
                {
                    "valid": True,
                    "strain_profile": [0.0, -1e-5, 0.0],
                    "reason": None,
                },
                context="simplified M-K diagram",
            )
        self.assertIn("missing m_u", str(ctx.exception))

    def test_rejects_missing_strain_profile(self) -> None:
        with self.assertRaises(InvalidSectionForMKError) as ctx:
            _validate_ultimate_mk_terminal(
                {
                    "valid": True,
                    "m_u": -1.0e7,
                    "strain_profile": None,
                    "reason": None,
                },
                context="simplified M-K diagram",
            )
        self.assertIn("missing strain_profile", str(ctx.exception))

    def test_rejects_nan_and_inf(self) -> None:
        cases = (
            {"m_u": float("nan"), "kappa": -1e-5, "needle": "non-finite"},
            {"m_u": float("inf"), "kappa": -1e-5, "needle": "non-finite"},
            {"m_u": -1.0e7, "kappa": float("nan"), "needle": "non-finite"},
            {"m_u": -1.0e7, "kappa": float("-inf"), "needle": "non-finite"},
        )
        for case in cases:
            with self.subTest(m_u=case["m_u"], kappa=case["kappa"]):
                with self.assertRaises(InvalidSectionForMKError) as ctx:
                    _validate_ultimate_mk_terminal(
                        {
                            "valid": True,
                            "m_u": case["m_u"],
                            "strain_profile": [0.0, case["kappa"], 0.0],
                            "reason": None,
                        },
                        context="full M-K diagram",
                    )
                self.assertIn(case["needle"], str(ctx.exception))

    def test_rejects_unreadable_strain_profile(self) -> None:
        with self.assertRaises(InvalidSectionForMKError) as ctx:
            _validate_ultimate_mk_terminal(
                {
                    "valid": True,
                    "m_u": -1.0e7,
                    "strain_profile": [0.0],  # missing κ_u index
                    "reason": None,
                },
                context="full M-K diagram",
            )
        self.assertIn("unreadable m_u/κ_u", str(ctx.exception))
        # Must not leak as IndexError / TypeError.
        self.assertIsInstance(ctx.exception, InvalidSectionForMKError)

    def test_full_path_maps_invalid_mu_to_invalid_section(self) -> None:
        section = MagicMock()
        with patch(
            "core.analysis_core.section_methods.calculate_bending_strength_Nmm_EC",
            return_value={
                "valid": False,
                "m_u": None,
                "strain_profile": None,
                "reason": "cannot be taken by section",
            },
        ):
            with self.assertRaises(InvalidSectionForMKError) as ctx:
                _full_moment_curvature_method(section, n=0.0, m_k_num_points=5)
        self.assertIn("full M-K diagram", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, KeyError)
        self.assertNotIsInstance(ctx.exception, TypeError)


class TestTerminalSofteningPrefix(unittest.TestCase):
    """Step 3 gate: pure array cut helper."""

    def test_monotone_curve_no_cut(self) -> None:
        M = np.array([0.0, -50.0, -80.0, -100.0], dtype=float)
        k = np.array([0.0, 0.001, 0.002, 0.003], dtype=float)
        M_in = M.copy()
        k_in = k.copy()
        M_out, k_out, meta = _terminal_softening_prefix(
            M, k, atol=1e-9, rtol=1e-6
        )
        self.assertFalse(meta["cut"])
        np.testing.assert_array_equal(M_out, M)
        np.testing.assert_array_equal(k_out, k)
        self.assertEqual(meta["peak_index"], 3)
        self.assertAlmostEqual(meta["M_cap"], 100.0)
        self.assertAlmostEqual(meta["kappa_cap"], 0.003)
        np.testing.assert_array_equal(M, M_in)
        np.testing.assert_array_equal(k, k_in)

    def test_terminal_softening_cuts_at_first_peak(self) -> None:
        M = np.array([0.0, -50.0, -100.0, -80.0, -60.0], dtype=float)
        k = np.array([0.0, 0.001, 0.002, 0.003, 0.004], dtype=float)
        M_out, k_out, meta = _terminal_softening_prefix(
            M, k, atol=1e-9, rtol=1e-6
        )
        self.assertTrue(meta["cut"])
        self.assertEqual(meta["peak_index"], 2)
        np.testing.assert_array_equal(M_out, M[:3])
        np.testing.assert_array_equal(k_out, k[:3])
        self.assertAlmostEqual(meta["M_terminal"], 60.0)
        self.assertAlmostEqual(meta["kappa_terminal"], 0.004)
        self.assertAlmostEqual(meta["M_cap"], 100.0)
        self.assertAlmostEqual(meta["kappa_cap"], 0.002)
        self.assertAlmostEqual(float(M_out[-1]), -100.0)
        self.assertAlmostEqual(float(k_out[-1]), 0.002)

    def test_within_tolerance_no_cut(self) -> None:
        M = np.array([0.0, -100.0, -99.999], dtype=float)
        k = np.array([0.0, 0.002, 0.003], dtype=float)
        M_out, k_out, meta = _terminal_softening_prefix(
            M, k, atol=1e-2, rtol=0.0
        )
        self.assertFalse(meta["cut"])
        self.assertEqual(len(M_out), 3)
        np.testing.assert_array_equal(M_out, M)
        np.testing.assert_array_equal(k_out, k)
        self.assertAlmostEqual(meta["M_cap"], 99.999)

    def test_first_peak_occurrence_when_plateau(self) -> None:
        M = np.array([0.0, -100.0, -100.0, -70.0], dtype=float)
        k = np.array([0.0, 0.002, 0.003, 0.004], dtype=float)
        M_out, k_out, meta = _terminal_softening_prefix(
            M, k, atol=1e-9, rtol=1e-6
        )
        self.assertTrue(meta["cut"])
        self.assertEqual(meta["peak_index"], 1)
        np.testing.assert_array_equal(M_out, M[:2])
        np.testing.assert_array_equal(k_out, k[:2])
        self.assertAlmostEqual(meta["kappa_cap"], 0.002)

    def test_recoverable_local_dip_does_not_cut(self) -> None:
        M = np.array([0.0, -50.0, -40.0, -80.0], dtype=float)
        k = np.array([0.0, 0.001, 0.002, 0.003], dtype=float)
        M_out, k_out, meta = _terminal_softening_prefix(
            M, k, atol=1e-9, rtol=1e-6
        )
        self.assertFalse(meta["cut"])
        self.assertEqual(len(M_out), 4)
        self.assertEqual(meta["peak_index"], 3)
        self.assertAlmostEqual(meta["M_cap"], 80.0)

    def test_rejects_mismatched_or_nonfinite_inputs(self) -> None:
        with self.assertRaises(ValueError):
            _terminal_softening_prefix(
                np.array([0.0, -1.0]), np.array([0.0]), atol=0.0, rtol=0.0
            )
        with self.assertRaises(ValueError):
            _terminal_softening_prefix(
                np.array([0.0, float("nan")]),
                np.array([0.0, 0.001]),
                atol=0.0,
                rtol=0.0,
            )


class TestLocalInterpolationThenForceControl(unittest.TestCase):
    """Step 6 gate: interpolate raw equal-length curves, then cut locally."""

    def test_support_and_mid_different_peaks_yield_local_caps(self) -> None:
        # Positive sagging magnitudes as in deflection lookup units (kNm, 1/m).
        # Equal-length raw softening curves with different peak indices.
        M_support = np.array([0.0, 50.0, 100.0, 80.0, 60.0], dtype=float)
        k_support = np.array([0.0, 0.001, 0.002, 0.003, 0.004], dtype=float)
        M_mid = np.array([0.0, 40.0, 90.0, 130.0, 70.0], dtype=float)
        k_mid = np.array([0.0, 0.001, 0.002, 0.003, 0.005], dtype=float)
        self.assertEqual(len(M_support), len(M_mid))

        # At support (x=0): local curve == support → cut at index 2 (100 @ 0.002)
        M0 = DeflectionCalculator._parabolic_interpolate(M_support, M_mid, 0.0)
        k0 = DeflectionCalculator._parabolic_interpolate(k_support, k_mid, 0.0)
        M0_fc, k0_fc, meta0 = _force_controlled_mk_arrays(M0, k0)
        self.assertTrue(meta0["cut"])
        self.assertEqual(len(M0_fc), 3)
        self.assertAlmostEqual(float(M0_fc[-1]), 100.0)
        self.assertAlmostEqual(float(k0_fc[-1]), 0.002)

        # At midspan vertex (x=0.5): local curve == mid → cut at index 3
        M1 = DeflectionCalculator._parabolic_interpolate(M_support, M_mid, 0.5)
        k1 = DeflectionCalculator._parabolic_interpolate(k_support, k_mid, 0.5)
        M1_fc, k1_fc, meta1 = _force_controlled_mk_arrays(M1, k1)
        self.assertTrue(meta1["cut"])
        self.assertEqual(len(M1_fc), 4)
        self.assertAlmostEqual(float(M1_fc[-1]), 130.0)
        self.assertAlmostEqual(float(k1_fc[-1]), 0.003)

        # Different local capacities — gate for spatially varying (M_cap, κ_cap).
        self.assertNotAlmostEqual(float(M0_fc[-1]), float(M1_fc[-1]))
        self.assertNotAlmostEqual(float(k0_fc[-1]), float(k1_fc[-1]))

    def test_lookup_and_bounds_use_same_force_controlled_local_curve(self) -> None:
        M_raw = np.array([0.0, 50.0, 100.0, 80.0, 60.0], dtype=float)
        k_raw = np.array([0.0, 0.001, 0.002, 0.003, 0.004], dtype=float)
        M_fc, k_fc, meta = _force_controlled_mk_arrays(M_raw, k_raw)
        self.assertTrue(meta["cut"])

        kappa = lookup_kappa_force_controlled(90.0, M_fc, k_fc)
        M_low, M_high, k_low, k_high = DeflectionCalculator._interpolation_bounds(
            90.0, M_fc, k_fc
        )
        self.assertLessEqual(M_low, 90.0)
        self.assertGreaterEqual(M_high, 90.0)
        self.assertTrue(np.isclose(kappa, np.interp(90.0, M_fc, k_fc)))

        # Demand above local cut capacity must reject (not use raw κ_u tail).
        with self.assertRaises(AboveMkCapacityError):
            lookup_kappa_force_controlled(100.1, M_fc, k_fc)

    def test_tail_lifted_cache_would_hide_cut_but_raw_does_not(self) -> None:
        # Documents why caches must stay raw: after never-recovers lift,
        # peak ≈ terminal so cut would not fire.
        M_raw = np.array([0.0, 50.0, 100.0, 80.0, 60.0], dtype=float)
        k_raw = np.array([0.0, 0.001, 0.002, 0.003, 0.004], dtype=float)
        M_lifted, k_lifted, _ = _force_controlled_mk_arrays(
            M_raw, k_raw, cut_terminal_softening=False
        )
        _M_hid, _k_hid, meta_hid = _force_controlled_mk_arrays(
            M_lifted, k_lifted, cut_terminal_softening=True
        )
        self.assertFalse(meta_hid["cut"])

        _M_ok, k_ok, meta_ok = _force_controlled_mk_arrays(
            M_raw, k_raw, cut_terminal_softening=True
        )
        self.assertTrue(meta_ok["cut"])
        self.assertAlmostEqual(float(k_ok[-1]), 0.002)


if __name__ == "__main__":
    unittest.main()
