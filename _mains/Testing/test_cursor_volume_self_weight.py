"""
Baseline volume / self-weight regressions for CURRENT HP-shell accounting.

Documents:
- objective concrete quantity uses net volume (gross − reinforcement)
- structural self-weight uses gross shell volume
- CFRP density does not affect structural self-weight
"""
from __future__ import annotations

import unittest

from _mains.Testing.fixtures.hp_slabs import hp_slab_ref
from core.unit_core import mm3_to_m3


class TestVolumeSelfWeightBaseline(unittest.TestCase):
    """Assert current net/gross volume and self-weight conventions."""

    def test_net_concrete_equals_gross_minus_reinforcement(self) -> None:
        shell = hp_slab_ref.hp_shell
        gross = shell.hp_geometry.volume()
        reinf = shell.total_reinforcement_volume()
        net = shell.net_concrete_volume()
        self.assertGreater(gross, 0.0)
        self.assertGreater(reinf, 0.0)
        self.assertAlmostEqual(net, gross - reinf)
        self.assertLess(net, gross)

    def test_self_load_uses_gross_volume_not_net(self) -> None:
        slab = hp_slab_ref
        shell = slab.hp_shell
        gross_m3 = mm3_to_m3(shell.hp_geometry.volume())
        net_m3 = mm3_to_m3(shell.net_concrete_volume())
        gamma_c = shell.concrete.density * 10.0 / 1000.0
        area_m2 = (slab.B * slab.L) * 1e-6

        expected_gross = gross_m3 * gamma_c / area_m2
        expected_net = net_m3 * gamma_c / area_m2
        actual = slab.self_load()

        self.assertAlmostEqual(actual, expected_gross)
        self.assertNotAlmostEqual(actual, expected_net)

    def test_self_load_ignores_reinforcement_area(self) -> None:
        slab = hp_slab_ref
        before = slab.self_load()
        original_area = slab.hp_shell.reinf_area
        net_before = slab.hp_shell.net_concrete_volume()

        # Changing bar area changes net concrete volume, but not gross-volume self-weight
        slab.hp_shell.reinf_area = original_area * 10.0
        try:
            after = slab.self_load()
            net_after = slab.hp_shell.net_concrete_volume()
        finally:
            slab.hp_shell.reinf_area = original_area

        self.assertNotAlmostEqual(net_before, net_after)
        self.assertAlmostEqual(before, after)


if __name__ == "__main__":
    unittest.main()
