"""
Above-capacity M–κ lookup must reject, not clamp via numpy.interp.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from core.analysis_core.checks.structural_checks import (
    DeflectionLimitByDeflectionCheckEC2004DE,
    FailureAnnouncementByDeflectionCheckEC2004DE,
)
from core.analysis_core.section_methods import (
    AboveMkCapacityError,
    _force_controlled_mk_arrays,
    lookup_kappa_force_controlled,
)
from core.analysis_core.statics.deflection import DeflectionCalculator
from core.analysis_core.statics.constants import SystemType


class TestLookupKappaForceControlled(unittest.TestCase):
    def test_interpolates_within_capacity(self) -> None:
        kappa = lookup_kappa_force_controlled(50.0, [0.0, 100.0], [0.0, 0.002])
        self.assertAlmostEqual(kappa, 0.001)

    def test_allows_exact_capacity(self) -> None:
        kappa = lookup_kappa_force_controlled(100.0, [0.0, 100.0], [0.0, 0.002])
        self.assertAlmostEqual(kappa, 0.002)

    def test_rejects_above_capacity(self) -> None:
        with self.assertRaises(AboveMkCapacityError) as ctx:
            lookup_kappa_force_controlled(100.1, [0.0, 100.0], [0.0, 0.002])
        self.assertIn("exceeds M–κ capacity", str(ctx.exception))

    def test_identical_cracking_points_remain_lookup_safe(self) -> None:
        moments = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 200.0])
        kappas = np.array([0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.003])

        M_fc, kappa_fc, _meta = _force_controlled_mk_arrays(moments, kappas)
        np.testing.assert_array_equal(M_fc, moments)
        np.testing.assert_array_equal(kappa_fc, kappas)

        self.assertAlmostEqual(
            lookup_kappa_force_controlled(99.9, M_fc, kappa_fc),
            0.000999,
        )
        self.assertAlmostEqual(
            lookup_kappa_force_controlled(100.0, M_fc, kappa_fc),
            0.001,
        )
        self.assertAlmostEqual(
            lookup_kappa_force_controlled(100.1, M_fc, kappa_fc),
            0.001002,
        )

        _M_lo, _M_hi, _kappa_lo, kappa_hi = (
            DeflectionCalculator._interpolation_bounds(
                100.0, M_fc, kappa_fc
            )
        )
        self.assertAlmostEqual(kappa_hi, 0.001)


class TestBChecksMapAboveCapacity(unittest.TestCase):
    def test_b1a_maps_above_capacity_to_invalid_utilization(self) -> None:
        slab = MagicMock()
        slab.slab.L = 8000.0
        loads = MagicMock()
        with patch(
            "core.analysis_core.checks.structural_checks.DeflectionCalculator.calculate_deflection_mm_EC",
            side_effect=AboveMkCapacityError("over capacity"),
        ):
            util = DeflectionLimitByDeflectionCheckEC2004DE.calculate_utilization(
                slab,
                loads,
                system=SystemType.SIMPLE_BEAM,
                limit_factor=250.0,
                invalid_utilization=99.0,
            )
        self.assertEqual(util, 99.0)

    def test_b2a_maps_above_capacity_to_invalid_utilization(self) -> None:
        slab = MagicMock()
        slab.slab.L = 8000.0
        loads = MagicMock()
        with patch(
            "core.analysis_core.checks.structural_checks.DeflectionCalculator.calculate_deflection_mm_EC",
            side_effect=AboveMkCapacityError("over capacity"),
        ):
            util = FailureAnnouncementByDeflectionCheckEC2004DE.calculate_utilization(
                slab,
                loads,
                system=SystemType.SIMPLE_BEAM,
                min_factor=100.0,
                invalid_utilization=99.0,
            )
        self.assertEqual(util, 99.0)


if __name__ == "__main__":
    unittest.main()
