"""
Finish-material objective accounting (infill / insulation / screed).

GWP always uses registry per-kg intensities × mass.
Cost is included only when registry cost is present.
"""
from __future__ import annotations

import unittest

from core.analysis_core.material_methods import get_material_properties
from core.ioh_core.import_specs import load_materials_registry
from slab_construction.floor import FloorLayer, InsulationMaterial, ScreedMaterial

from _mains.Testing._paths import MATERIALS_CSV


class TestFinishObjectiveUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reg = load_materials_registry(MATERIALS_CSV)

    def test_finish_gwp_from_registry_per_kg(self) -> None:
        infill = get_material_properties("generic_infill", self.reg)
        insu = get_material_properties("lightweight_mineral_wool", self.reg)
        screed = get_material_properties("generic_screed", self.reg)

        self.assertEqual(infill["gwp"], 0.0)
        self.assertAlmostEqual(insu["gwp"], 1.4375)
        self.assertAlmostEqual(screed["gwp"], 0.175)
        self.assertIsNone(infill["cost"])
        self.assertIsNone(insu["cost"])
        self.assertIsNone(screed["cost"])

    def test_finish_mass_times_gwp_hand_calc(self) -> None:
        # 20 mm insulation @ 24 kg/m³ → 0.48 kg/m²; area 10 m² → 4.8 kg
        layer = FloorLayer(
            InsulationMaterial(density=24.0, E_dyn=0.6, name="lightweight_mineral_wool"),
            20.0,
        )
        area_m2 = 10.0
        mass_kg = layer.area_density_kg_m2() * area_m2
        gwp = get_material_properties("lightweight_mineral_wool", self.reg)["gwp"]
        self.assertAlmostEqual(mass_kg, 4.8)
        self.assertAlmostEqual(gwp * mass_kg, 1.4375 * 4.8)

    def test_screed_mass_times_gwp_hand_calc(self) -> None:
        layer = FloorLayer(
            ScreedMaterial(density=2000.0, name="generic_screed"),
            50.0,
        )
        area_m2 = 4.0
        mass_kg = layer.area_density_kg_m2() * area_m2  # 100 kg/m² × 4 = 400 kg
        gwp = get_material_properties("generic_screed", self.reg)["gwp"]
        self.assertAlmostEqual(mass_kg, 400.0)
        self.assertAlmostEqual(gwp * mass_kg, 0.175 * 400.0)

    def test_missing_finish_cost_is_omitted_not_zero_invented(self) -> None:
        """Contract: empty registry cost → omit from y_cost (list as missing)."""
        missing = []
        y_cost = 0.0
        for name, mat_id in (
            ("infill", "generic_infill"),
            ("insulation", "lightweight_mineral_wool"),
            ("screed", "generic_screed"),
        ):
            cost = get_material_properties(mat_id, self.reg).get("cost")
            if cost is None:
                missing.append(name)
            else:
                y_cost += cost * 1.0
        self.assertEqual(missing, ["infill", "insulation", "screed"])
        self.assertEqual(y_cost, 0.0)


if __name__ == "__main__":
    unittest.main()
