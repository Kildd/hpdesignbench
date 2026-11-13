"""
Axial-force unit conversion: CSV input is kN, section APIs expect N.

Sign convention (matches section_methods / GH): positive = tension.
"""
from __future__ import annotations

import unittest

from core.unit_core import N_to_kN, kN_to_N


class TestAxialForceUnits(unittest.TestCase):
    def test_kN_to_N_round_trip(self) -> None:
        self.assertAlmostEqual(kN_to_N(100.0), 100_000.0)
        self.assertAlmostEqual(kN_to_N(-100.0), -100_000.0)
        self.assertAlmostEqual(kN_to_N(0.0), 0.0)
        self.assertAlmostEqual(N_to_kN(kN_to_N(37.5)), 37.5)

    def test_analysis_conversion_contract(self) -> None:
        """Mirror of analysis.py: decode kN once, pass N downstream."""
        loads_N_kN = 100.0  # as from params / CSV
        n = kN_to_N(loads_N_kN)
        self.assertEqual(n, 100_000.0)
        # tension stays positive after conversion
        self.assertGreater(kN_to_N(10.0), 0.0)
        self.assertLess(kN_to_N(-10.0), 0.0)


if __name__ == "__main__":
    unittest.main()
