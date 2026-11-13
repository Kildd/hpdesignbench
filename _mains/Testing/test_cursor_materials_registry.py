"""
Materials registry regressions for the redesigned materials.csv.
"""
from __future__ import annotations

import unittest

from structuralcodes import set_design_code

from core.analysis_core.material_methods import (
    get_cfrp_reinforcement_from_registry,
    get_floor_material_from_registry,
    get_material_properties,
)
from core.ioh_core.import_specs import load_materials_registry

from _mains.Testing._paths import MATERIALS_CSV


class TestMaterialsRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_design_code("ec2_2004")
        cls.reg = load_materials_registry(MATERIALS_CSV)

    def test_concrete_c30_intensities_and_factors(self) -> None:
        c30 = self.reg["C30/37"]
        self.assertEqual(c30["type"], "concrete")
        self.assertEqual(c30["f_ck"], 30.0)
        self.assertEqual(c30["alpha_cc"], 0.85)
        self.assertEqual(c30["gamma_c"], 1.5)
        self.assertEqual(c30["gwp_m3"], 219.0)
        self.assertEqual(c30["cost_m3"], 85.0)
        self.assertIsNone(c30["gwp"])
        self.assertIsNone(c30["cost"])
        self.assertIn("Referenzbeton", c30["comment"] or "")

    def test_all_concrete_grades_present(self) -> None:
        expected = [
            "C12/15", "C16/20", "C20/25", "C25/30", "C30/37",
            "C35/45", "C40/50", "C45/55", "C50/60", "C55/67",
            "C60/75", "C70/85", "C80/95", "C90/105", "C100/115",
        ]
        for name in expected:
            self.assertIn(name, self.reg)
            self.assertEqual(self.reg[name]["type"], "concrete")

    def test_cfrp_uses_alpha_tex_and_promille_eps(self) -> None:
        r1 = self.reg["solidian Roving 1R (Z-1.6-308)"]
        self.assertEqual(r1["alpha_tex"], 0.83)
        self.assertNotIn("alpha_nm", r1)
        self.assertEqual(r1["eps_u_c"], 0.0)
        self.assertEqual(r1["gwp"], 23.3)
        self.assertEqual(r1["cost"], 49.0)
        reinf = get_cfrp_reinforcement_from_registry(
            "solidian Roving 1R (Z-1.6-308)", self.reg
        )
        self.assertAlmostEqual(reinf.epsuk, 12.6 / 1000)
        self.assertAlmostEqual(reinf.Es, 240000.0)

    def test_get_material_properties_and_finishes(self) -> None:
        props = get_material_properties("generic_screed", self.reg)
        self.assertEqual(props["gwp"], 0.175)
        self.assertIsNone(props["cost"])
        ins = get_floor_material_from_registry("lightweight_mineral_wool", self.reg)
        self.assertEqual(ins.density, 24.0)
        self.assertEqual(float(ins.E_dyn), 0.6)


if __name__ == "__main__":
    unittest.main()
