"""
R6 goldens: Loutfi C.1 cracking and ultimate moments (M_cr, M_u ULS/SLS).

Source: Loutfi (2023) ch. C.1; verification tables marked passed 27.01.2026
in ``_mains/verification/mcr_verification.py``, ``mu_uls_verification.py``,
``mu_sls_verification.py``. Tolerances absorb small solver drift since then.
"""
from __future__ import annotations

import unittest

from _mains.Testing.fixtures.hp_sections import (
    hp_section_c1_1_uls,
    hp_section_c1_2_c50_uls,
    hp_section_c1_2_c80_uls,
    hp_section_c1_3_uls,
    hp_section_c1_4_uls,
)
from core.analysis_core.section_methods import (
    calculate_bending_strength_Nmm_EC,
    calculate_cracking_moment_sls_Nmm_EC,
    sls_section_EC,
)
from core.unit_core import Nmm_to_kNm

# (label, section, M_cr [kNm], M_u ULS [kNm], M_u SLS [kNm])
_LOUTFI_C1 = (
    ("C1_1", hp_section_c1_1_uls, 38.62, 62.55, 91.61),
    ("C1_2_C50", hp_section_c1_2_c50_uls, 185.10, 212.51, 309.79),
    ("C1_2_C80", hp_section_c1_2_c80_uls, 192.82, 240.86, 337.88),
    ("C1_3", hp_section_c1_3_uls, 46.34, 251.38, 374.34),
    ("C1_4", hp_section_c1_4_uls, 409.73, 490.12, 722.60),
)


class TestLoutfiC1Capacities(unittest.TestCase):
    def test_m_cr_fctm_parabolic(self) -> None:
        for name, section, m_cr_ref, _, _ in _LOUTFI_C1:
            with self.subTest(section=name):
                res = calculate_cracking_moment_sls_Nmm_EC(
                    sls_section_EC(section, "FCTM_PARABOLIC"), n=0.0
                )
                self.assertTrue(res.get("valid", True), msg=res.get("reason"))
                m_cr = Nmm_to_kNm(-res["m_cr"])
                self.assertAlmostEqual(m_cr, m_cr_ref, delta=0.5)

    def test_m_u_uls(self) -> None:
        for name, section, _, m_u_ref, _ in _LOUTFI_C1:
            with self.subTest(section=name):
                res = calculate_bending_strength_Nmm_EC(section, n=0.0)
                self.assertTrue(res.get("valid", True), msg=res.get("reason"))
                m_u = Nmm_to_kNm(-res["m_u"])
                self.assertAlmostEqual(m_u, m_u_ref, delta=0.6)

    def test_m_u_sls_none_parabolic(self) -> None:
        for name, section, _, _, m_u_ref in _LOUTFI_C1:
            with self.subTest(section=name):
                res = calculate_bending_strength_Nmm_EC(
                    sls_section_EC(section, "NONE_PARABOLIC"), n=0.0
                )
                self.assertTrue(res.get("valid", True), msg=res.get("reason"))
                m_u = Nmm_to_kNm(-res["m_u"])
                # SLS Mu drifted more than ULS/Mcr since the 27.01.2026 table.
                self.assertAlmostEqual(m_u, m_u_ref, delta=2.0)


if __name__ == "__main__":
    unittest.main()
