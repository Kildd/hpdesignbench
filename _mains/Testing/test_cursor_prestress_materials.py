"""
Phase R3 regressions: prestress catalog 0–60%/5% and 1R/2R approval card.
"""
from __future__ import annotations

import math
import unittest

from structuralcodes import set_design_code

from core.analysis_core.material_methods import get_cfrp_reinforcement_from_registry
from core.ioh_core.import_specs import (
    build_space,
    load_constraint_defaults,
    load_materials_registry,
    load_param_defaults,
    load_problems_combined,
    make_decode,
)

from _mains.Testing._paths import (
    CONSTRAINT_DEFAULTS_CSV,
    MATERIALS_CSV,
    PARAM_DEFAULTS_CSV,
    PROBLEM_LIST_CSV,
)


class TestPrestressCatalogAndCardinality(unittest.TestCase):
    """Assert prestress domain, bounds, and Problem 1 cardinality."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
        cdef = load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV)
        cls.problems = load_problems_combined(cls.pdef, cdef, PROBLEM_LIST_CSV)
        cls.info = cls.problems["L6750_B2"]
        cls.var_names, cls.lb_idx, cls.ub_idx = build_space(cls.info["model"])
        cls.decode = staticmethod(make_decode(cls.info["model"], cls.var_names))

    def test_prestress_catalog_is_0_to_60_by_5(self) -> None:
        vals = self.pdef["reinf_kap_t_percent"]["values"]
        expected = [float(p) for p in range(0, 61, 5)]
        self.assertEqual(vals, expected)
        self.assertEqual(len(vals), 13)
        self.assertNotIn(65.0, vals)
        self.assertNotIn(70.0, vals)

    def test_problem_1_prestress_bounds_cover_full_catalog(self) -> None:
        i = self.var_names.index("reinf_kap_t_percent")
        self.assertEqual(self.lb_idx[i], 0)
        self.assertEqual(self.ub_idx[i], 12)
        self.assertEqual(
            self.info["model"]["reinf_kap_t_percent"]["ub"],
            12,
        )

    def test_decode_includes_0_and_60_percent_boundaries(self) -> None:
        i = self.var_names.index("reinf_kap_t_percent")
        x0 = list(self.lb_idx)
        x0[i] = 0
        x60 = list(self.lb_idx)
        x60[i] = 12
        self.assertEqual(self.decode(x0)["reinf_kap_t_percent"], 0.0)
        self.assertEqual(self.decode(x60)["reinf_kap_t_percent"], 60.0)

    def test_problem_1_cardinality_from_build_space(self) -> None:
        cardinality = 1
        for lo, hi in zip(self.lb_idx, self.ub_idx):
            cardinality *= hi - lo + 1
        # Product of (ub-lb+1) over Problem 1 variables matching the agreed
        # 9-variable design space:
        # 11*9*6*6*12*13*4*6*4 = 53_374_464
        self.assertEqual(cardinality, 53_374_464)

    def test_geom_h_ges_catalog_is_100_to_600_by_50(self) -> None:
        vals = self.pdef["geom_h_ges_mm"]["values"]
        expected = [float(h) for h in range(100, 601, 50)]
        self.assertEqual(vals, expected)
        i = self.var_names.index("geom_h_ges_mm")
        self.assertEqual(self.lb_idx[i], 0)
        self.assertEqual(self.ub_idx[i], 10)
        self.assertEqual(self.decode(list(self.lb_idx))["geom_h_ges_mm"], 100.0)
        x_max = list(self.lb_idx)
        x_max[i] = 10
        self.assertEqual(self.decode(x_max)["geom_h_ges_mm"], 600.0)


class TestSolidianApprovalMaterialCard(unittest.TestCase):
    """Assert 1R/2R approval values, units, and ULS prestress initial strain."""

    @classmethod
    def setUpClass(cls) -> None:
        set_design_code("ec2_2004")
        cls.reg = load_materials_registry(MATERIALS_CSV)
        cls.id_1r = "solidian Roving 1R (Z-1.6-308)"
        cls.id_2r = "solidian Roving 2R (Z-1.6-308)"

    def test_1r_approval_modulus_and_strain(self) -> None:
        r1 = self.reg[self.id_1r]
        self.assertEqual(r1["E_tex"], 240000.0)
        self.assertEqual(r1["eps_y"], 12.6)
        self.assertEqual(r1["eps_u"], 12.6)
        self.assertEqual(r1["f_tk"], 3039.0)
        # f_tk is independently sourced; do not force E * epsuk.
        self.assertNotAlmostEqual(r1["f_tk"], r1["E_tex"] * r1["eps_u"] / 1000.0)

    def test_2r_approval_strain_keeps_independent_e_and_ftk(self) -> None:
        r2 = self.reg[self.id_2r]
        self.assertEqual(r2["eps_y"], 12.4)
        self.assertEqual(r2["eps_u"], 12.4)
        self.assertEqual(r2["E_tex"], 235000.0)
        self.assertEqual(r2["f_tk"], 2917.0)
        self.assertNotAlmostEqual(r2["f_tk"], r2["E_tex"] * r2["eps_u"] / 1000.0)

    def test_promille_loader_and_uls_prestress_initial_strain(self) -> None:
        r1 = get_cfrp_reinforcement_from_registry(
            self.id_1r, self.reg, prestress_percent=60.0, limit_state="ULS"
        )
        epsuk = 12.6 / 1000.0
        self.assertAlmostEqual(r1.epsuk, epsuk)
        self.assertAlmostEqual(r1.Es, 240000.0)
        self.assertAlmostEqual(float(r1.initial_strain), 0.60 * epsuk)
        # ULS constitutive law uses design strain; prestress uses characteristic epsuk.
        eps_ud = 0.83 * epsuk / 1.3
        self.assertAlmostEqual(
            float(r1.constitutive_law.wrapped_law.get_ultimate_strain()[1]),
            eps_ud,
        )
        self.assertLess(float(r1.initial_strain), eps_ud)

    def test_2r_uls_prestress_at_sixty_percent(self) -> None:
        r2 = get_cfrp_reinforcement_from_registry(
            self.id_2r, self.reg, prestress_percent=60.0, limit_state="ULS"
        )
        epsuk = 12.4 / 1000.0
        self.assertAlmostEqual(r2.epsuk, epsuk)
        self.assertAlmostEqual(float(r2.initial_strain), 0.60 * epsuk)
        self.assertTrue(math.isfinite(float(r2.initial_strain)))


if __name__ == "__main__":
    unittest.main()
