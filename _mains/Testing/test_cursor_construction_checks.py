"""
Construction-check regressions for GH-aligned C.1–C.3.

- C.2 spacing requirement uses 6 · d_p
- C.3 thickness requirement uses 14 · d_p (= 2d + 2·3d + 6d)
- Zero cover/spacing returns the CSV ``opt_invalid_utilization`` sentinel
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.analysis_core.checks.construction_checks import (
    MidlineConcreteCoverCheck,
    MinimumHPShellThicknessCheck,
    ReinforcementSpacingCheck,
)
from core.ioh_core.import_specs import load_param_defaults

from _mains.Testing._paths import PARAM_DEFAULTS_CSV


def _slab_with_shell(hp_shell) -> MagicMock:
    slab_construction = MagicMock()
    slab_construction.slab.hp_shell = hp_shell
    return slab_construction


class TestConstructionChecksGhAligned(unittest.TestCase):
    """Assert C.1–C.3 formulas and CSV-sourced sentinel."""

    @classmethod
    def setUpClass(cls) -> None:
        pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
        cls.invalid_utilization = float(pdef["opt_invalid_utilization"]["values"][0])

    def test_c1_cover_utilization_uses_3d(self) -> None:
        hp_shell = MagicMock()
        hp_shell.d_p.return_value = 10.0
        hp_shell.c_1_clear_concrete_cover.return_value = 30.0  # exactly 3·d

        u = MidlineConcreteCoverCheck.calculate_utilization(
            _slab_with_shell(hp_shell),
            invalid_utilization=self.invalid_utilization,
        )
        self.assertAlmostEqual(u, 1.0)

    def test_c1_zero_cover_returns_sentinel_99(self) -> None:
        hp_shell = MagicMock()
        hp_shell.d_p.return_value = 10.0
        hp_shell.c_1_clear_concrete_cover.return_value = 0.0

        u = MidlineConcreteCoverCheck.calculate_utilization(
            _slab_with_shell(hp_shell),
            invalid_utilization=self.invalid_utilization,
        )
        self.assertEqual(u, self.invalid_utilization)
        self.assertEqual(u, 99.0)

    def test_c2_spacing_utilization_uses_6d(self) -> None:
        hp_shell = MagicMock()
        hp_shell.d_p.return_value = 10.0
        hp_shell.s_min_clear_reinf_spacing.return_value = 60.0  # exactly 6·d

        u = ReinforcementSpacingCheck.calculate_utilization(
            _slab_with_shell(hp_shell),
            invalid_utilization=self.invalid_utilization,
        )
        self.assertAlmostEqual(u, 1.0)

    def test_c2_zero_spacing_returns_sentinel_99(self) -> None:
        hp_shell = MagicMock()
        hp_shell.d_p.return_value = 10.0
        hp_shell.s_min_clear_reinf_spacing.return_value = 0.0

        u = ReinforcementSpacingCheck.calculate_utilization(
            _slab_with_shell(hp_shell),
            invalid_utilization=self.invalid_utilization,
        )
        self.assertEqual(u, self.invalid_utilization)
        self.assertEqual(u, 99.0)

    def test_c3_thickness_utilization_uses_14d(self) -> None:
        # GH: t_req = 2d + 2*(3d) + 6d = 14d
        hp_shell = MagicMock()
        hp_shell.d_p.return_value = 10.0
        hp_shell.hp_geometry = SimpleNamespace(t=140.0)

        u = MinimumHPShellThicknessCheck.calculate_utilization(
            _slab_with_shell(hp_shell),
        )
        self.assertAlmostEqual(u, 1.0)

    def test_c3_below_14d_is_violated(self) -> None:
        hp_shell = MagicMock()
        hp_shell.d_p.return_value = 10.0
        hp_shell.hp_geometry = SimpleNamespace(t=137.5)

        u = MinimumHPShellThicknessCheck.calculate_utilization(
            _slab_with_shell(hp_shell),
        )
        self.assertAlmostEqual(u, 140.0 / 137.5)
        self.assertGreater(u, 1.0)


if __name__ == "__main__":
    unittest.main()
