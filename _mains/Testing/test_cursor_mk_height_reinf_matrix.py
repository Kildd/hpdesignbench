"""
Soft-fail contract for the h=0.10/0.30/0.60 m full-vs-simplified M–κ matrix.

All 36 reinforcement cases × {support, midspan} = 72 sections must complete
without unexpected exceptions. Full or simplified MK may fail only via
``InvalidSectionForMKError``.

The expected infeasible extreme is **crush-before-crack** (high prestress +
max ``a_tex`` / ``nt``): classical ``M_cr`` is undefined under ``N = 0``,
so M–κ is refused even though ``M_u`` may still solve. That is an
intentional soft-fail, not an Early-Cut / EOC defect.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from structuralcodes import set_design_code

from core.analysis_core.section_methods import (
    calculate_bending_strength_Nmm_EC,
    calculate_cracking_moment_sls_Nmm_EC,
    sls_section_EC,
)
from core.ioh_core.import_specs import load_materials_registry
from _mains.verification.plot_mk_height_reinf_matrix import (
    HP_DIR,
    MK_POSITIONS,
    MkCase,
    X_MID,
    X_SUPPORT,
    _position_label,
    build_uls_section,
    compute_mk_pair,
    iter_cases,
)


# Needle that identifies intentional crush-before-crack soft-fails in MK errors.
_CRUSH_BEFORE_CRACK_NEEDLE = "crush before cracking"

# Known matrix extreme: max a_tex / nt / kap — crush-before-crack on thin/mid sections.
_CRUSH_BEFORE_CRACK_CASE = MkCase(100.0, 10.86, 30, 60.0)


class TestMkHeightReinfMatrix(unittest.TestCase):
    """72 support+midspan cases: MK soft-fails only via InvalidSectionForMKError."""

    @classmethod
    def setUpClass(cls) -> None:
        set_design_code("ec2_2004")
        cls.materials = load_materials_registry(HP_DIR / "materials.csv")
        cls.cases = list(iter_cases())
        cls.positions = list(MK_POSITIONS)

    def test_matrix_has_thirty_six_cases_and_two_positions(self) -> None:
        self.assertEqual(len(self.cases), 36)
        heights = {c.h_ges_mm for c in self.cases}
        self.assertEqual(heights, {100.0, 300.0, 600.0})
        self.assertEqual(set(self.positions), {X_SUPPORT, X_MID})

    def test_mk_pair_only_soft_fails_via_invalid_section(self) -> None:
        full_soft: list[str] = []
        simp_soft: list[str] = []
        for case in self.cases:
            for x in self.positions:
                label = f"{case.label} {_position_label(x)}"
                with self.subTest(case=label):
                    section = build_uls_section(case, self.materials, x=x)
                    # Must not raise unexpected exceptions (ValueError, etc.).
                    pair = compute_mk_pair(section)

                    # raw and FC full share the same InvalidSectionForMKError gate
                    if pair["full"] is None:
                        self.assertIsNone(pair["full_raw"])
                        self.assertIsInstance(pair["full_error"], str)
                        self.assertTrue(pair["full_error"])
                        self.assertEqual(pair["full_error"], pair["full_raw_error"])
                        # Soft-fails in this matrix must be cracking-gate failures
                        # (crush-before-crack), not Early-Cut / EOC / Mu issues.
                        self.assertIn(
                            "cracking moment invalid",
                            pair["full_error"],
                            msg=(
                                f"{label}: unexpected full soft-fail source; "
                                f"got {pair['full_error']!r}"
                            ),
                        )
                        self.assertIn(
                            _CRUSH_BEFORE_CRACK_NEEDLE,
                            pair["full_error"],
                            msg=(
                                f"{label}: expected crush-before-crack in full "
                                f"error, got {pair['full_error']!r}"
                            ),
                        )
                        full_soft.append(label)
                    else:
                        self.assertIsNotNone(pair["full_raw"])
                        self.assertIsNone(pair["full_error"])
                        self.assertIsNone(pair["full_raw_error"])

                    if pair["simplified"] is None:
                        self.assertIsInstance(pair["simplified_error"], str)
                        self.assertTrue(pair["simplified_error"])
                        # EOC-only failures must not abort simplified MK anymore
                        # (EOC collapses onto cracking in production).
                        self.assertNotIn(
                            "EOC solve failed",
                            pair["simplified_error"],
                            msg="EOC fallback should prevent EOC-only soft-fails",
                        )
                        self.assertIn(
                            "cracking moment invalid",
                            pair["simplified_error"],
                            msg=(
                                f"{label}: unexpected simplified soft-fail source; "
                                f"got {pair['simplified_error']!r}"
                            ),
                        )
                        self.assertIn(
                            _CRUSH_BEFORE_CRACK_NEEDLE,
                            pair["simplified_error"],
                            msg=(
                                f"{label}: expected crush-before-crack in "
                                f"simplified error, got {pair['simplified_error']!r}"
                            ),
                        )
                        simp_soft.append(label)
                    else:
                        self.assertIsNone(pair["simplified_error"])

        if full_soft:
            print(f"full soft-fails ({len(full_soft)}): {full_soft}")
        if simp_soft:
            print(f"simplified soft-fails ({len(simp_soft)}): {simp_soft}")
        # Soft-fails, if any, must be explained as crush-before-crack above.
        self.assertEqual(full_soft, simp_soft)

    def test_crush_before_crack_soft_fail_documents_mcr_gate_not_mu(self) -> None:
        """a=10.86, nt=30, kap=60%, h=100: M_cr fails, M_u may still solve.

        Documents why the matrix panel is empty: intentional M–κ refusal at
        the cracking gate (crush-before-crack), not missing ultimate strength
        and not Early-Cut / EOC.
        """
        case = _CRUSH_BEFORE_CRACK_CASE
        for x in (X_SUPPORT, X_MID):
            label = f"{case.label} {_position_label(x)}"
            with self.subTest(case=label):
                section = build_uls_section(case, self.materials, x=x)
                sls = sls_section_EC(section, "TENSTIFF_PARABOLIC")

                m_cr = calculate_cracking_moment_sls_Nmm_EC(sls, n=0.0)
                m_u = calculate_bending_strength_Nmm_EC(sls, n=0.0)

                self.assertFalse(
                    m_cr.get("valid", True),
                    msg=f"{label}: expected invalid M_cr",
                )
                self.assertIn(_CRUSH_BEFORE_CRACK_NEEDLE, str(m_cr.get("reason", "")))
                self.assertTrue(
                    m_u.get("valid", False),
                    msg=(
                        f"{label}: M_u should still solve; crush-before-crack "
                        f"is an M_cr-gate refusal, not missing ultimate capacity"
                    ),
                )
                self.assertIsNotNone(m_u.get("m_u"))

                pair = compute_mk_pair(section)
                self.assertIsNone(pair["full"])
                self.assertIsNone(pair["simplified"])
                self.assertIn(_CRUSH_BEFORE_CRACK_NEEDLE, pair["full_error"])
                self.assertIn("cracking moment invalid", pair["full_error"])
                self.assertIn(_CRUSH_BEFORE_CRACK_NEEDLE, pair["simplified_error"])
                self.assertIn("cracking moment invalid", pair["simplified_error"])

    def test_eoc_fallback_collapses_onto_cracking_for_known_case(self) -> None:
        """Former EOC-only failure: simplified MK succeeds with κ_eoc = κ_cr."""
        from core.analysis_core.section_methods import (
            DEFLECTION_MK_SIMPLIFICATION,
            _build_simplified_mk_parts,
            _calculate_section_state_from_bottom_strain_sls,
            get_concrete,
        )

        case = MkCase(100.0, 10.86, 30, 30.0)
        # Known midspan case from earlier matrix work.
        section = build_uls_section(case, self.materials, x=X_MID)
        sls = sls_section_EC(section, "TENSTIFF_PARABOLIC")
        law = get_concrete(sls).constitutive_law

        # Sanity: EOC brentq itself still has no sign change on this section.
        m_cr = calculate_cracking_moment_sls_Nmm_EC(sls, n=0.0)
        m_u = calculate_bending_strength_Nmm_EC(sls, n=0.0)
        eoc = _calculate_section_state_from_bottom_strain_sls(
            sls,
            n=0.0,
            eps_bot=law.eps_F_t,
            method="brentq",
            bracket=(m_cr["strain_profile"][1], m_u["strain_profile"][1]),
            tolerance=1e-1,
        )
        self.assertFalse(eoc.get("valid", True))

        with patch(
            "core.analysis_core.section_methods.warnings.warn",
        ) as warn:
            parts = _build_simplified_mk_parts(
                sls,
                n=0.0,
                simplification=DEFLECTION_MK_SIMPLIFICATION,
                m_k_num_points=60,
            )
        warn.assert_called_once()
        self.assertIs(warn.call_args.args[1], RuntimeWarning)
        self.assertEqual(parts.kappa_eoc, parts.kappa_cr)
        self.assertEqual(parts.M_eoc_Nmm, parts.M_cr_Nmm)
        np.testing.assert_array_equal(
            parts.kappa_extra_cr_eoc,
            np.full(3, parts.kappa_cr),
        )
        np.testing.assert_array_equal(
            parts.M_extra_cr_eoc_Nmm,
            np.full(3, parts.M_cr_Nmm),
        )

        pair = compute_mk_pair(section)
        self.assertIsNotNone(pair["simplified"])
        self.assertIsNone(pair["simplified_error"])


if __name__ == "__main__":
    unittest.main()
