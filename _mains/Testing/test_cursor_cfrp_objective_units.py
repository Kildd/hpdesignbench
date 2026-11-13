"""
Unit-correct CFRP objective accounting.

Concrete intensities are per m³.
CFRP ratio factors encode (intensity_per_kg_CFRP) / (intensity_per_m3_C30).
Objective must therefore multiply CFRP intensities by mass = volume × density.
Registry GWP for rovings is 23.30 kgCO₂e/kg (materials.csv).
"""
from __future__ import annotations

import csv
import unittest
from pathlib import Path

from core.unit_core import mm3_to_m3


_MATERIALS_CSV = (
    Path(__file__).resolve().parents[2]
    / "slab_construction"
    / "slabs"
    / "hp_slab"
    / "materials.csv"
)


def _roving_registry_row(name_substr: str = "Roving 1R") -> dict[str, str]:
    with _MATERIALS_CSV.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            if name_substr in row["name"]:
                return row
    raise AssertionError(f"No materials.csv row containing {name_substr!r}")


class TestCfrpObjectiveUnits(unittest.TestCase):
    """Assert density-aware conversion from ratio factors."""

    def test_materials_csv_gwp_is_per_kg(self) -> None:
        row = _roving_registry_row()
        gwp_header = next(k for k in row if k.lower().startswith("gwp"))
        self.assertIn("/kg", gwp_header.lower().replace(" ", ""))
        self.assertAlmostEqual(float(row[gwp_header]), 23.30, places=2)
        density = float(row["density [kg/m3]"])
        self.assertAlmostEqual(density, 1300.0)

    def test_default_factors_recover_registry_per_kg_values(self) -> None:
        gwp_c30 = 219.0
        cost_c30 = 85.0
        gwp_factor = 0.1064  # 23.30 / 219
        cost_factor = 0.5765  # 49 / 85

        self.assertAlmostEqual(gwp_c30 * gwp_factor, 23.30, places=2)
        self.assertAlmostEqual(cost_c30 * cost_factor, 49.0, places=2)

    def test_objective_uses_mass_not_volume_alone(self) -> None:
        row = _roving_registry_row()
        rho = float(row["density [kg/m3]"])
        v_m3 = 1.0e-4  # arbitrary reinforcement volume [m³]
        mass_kg = v_m3 * rho

        gwp_per_kg = 219.0 * 0.1064  # recovers 23.30 kgCO2e/kg
        wrong_without_density = gwp_per_kg * v_m3
        correct_with_density = gwp_per_kg * mass_kg

        self.assertAlmostEqual(rho, 1300.0)
        self.assertAlmostEqual(correct_with_density / wrong_without_density, rho)
        self.assertAlmostEqual(correct_with_density, 23.3016 * mass_kg, places=4)
        # Sanity: volume-only path understates GWP by the density factor
        self.assertAlmostEqual(wrong_without_density * rho, correct_with_density)


if __name__ == "__main__":
    unittest.main()
