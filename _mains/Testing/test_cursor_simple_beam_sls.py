"""
Phase R4 regressions: simply supported beam only; B1a/B2a SLS pair.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from core.analysis_core.checks.structural_checks import (
    FailureAnnouncementByDeflectionCheckEC2004DE,
)
from core.analysis_core.statics.constants import (
    MOMENT_DATA,
    MAX_X_POSITIONS,
    MomentType,
    SystemType,
)
from core.analysis_core.statics.internal_forces import InternalForces, MOMENT_FUNCTIONS
from core.ioh_core.import_specs import (
    load_constraint_defaults,
    load_param_defaults,
    load_problems_combined,
)

from _mains.Testing._paths import (
    CONSTRAINT_DEFAULTS_CSV,
    PARAM_DEFAULTS_CSV,
    PROBLEM_LIST_CSV,
)
from slab_construction.slabs.hp_slab import analysis as hp_analysis


class TestSimplySupportedStaticsOnly(unittest.TestCase):
    def test_system_and_moment_enums_are_simple_beam_only(self) -> None:
        self.assertEqual(list(SystemType), [SystemType.SIMPLE_BEAM])
        self.assertEqual(list(MomentType), [MomentType.MAX_POS_MOMENT])
        self.assertEqual(set(MAX_X_POSITIONS), {"SIMPLE_BEAM"})
        self.assertEqual(set(MOMENT_DATA), {"SIMPLE_BEAM"})
        self.assertEqual(set(MOMENT_DATA["SIMPLE_BEAM"]), {"MAX_POS_MOMENT"})
        self.assertEqual(set(MOMENT_FUNCTIONS), {SystemType.SIMPLE_BEAM})
        self.assertFalse(hasattr(SystemType, "CANTILEVER"))
        self.assertFalse(hasattr(SystemType, "TWO_SPAN"))
        self.assertFalse(hasattr(MomentType, "MAX_NEG_MOMENT"))

    def test_ql2_over_8_coefficient_and_midspan(self) -> None:
        data = InternalForces.get_moment_data(
            SystemType.SIMPLE_BEAM, MomentType.MAX_POS_MOMENT
        )
        self.assertEqual(data["coefficient"], 0.125)
        self.assertEqual(data["x_position"], 0.5)

        # Analytical M(L/2) = w L^2 / 8
        w, L = 10.0, 8.0
        self.assertAlmostEqual(
            InternalForces.moment_simple_beam(L / 2.0, w, L),
            w * L * L / 8.0,
        )

    def test_unsupported_systems_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InternalForces.get_moment_data("CANTILEVER", MomentType.MAX_POS_MOMENT)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            InternalForces.validate_x_position("TWO_SPAN", 0.5)  # type: ignore[arg-type]


class TestB1aB2aOnlyConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
        cls.cdef = load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV)
        cls.problems = load_problems_combined(cls.pdef, cls.cdef, PROBLEM_LIST_CSV)
        cls.info = cls.problems["L6750_B2"]

    def test_defl_sls_case_and_b1b_b2b_absent(self) -> None:
        self.assertNotIn("defl_sls_case", self.pdef)
        self.assertNotIn("defl_sls_case", self.info["model"])
        self.assertNotIn("B1b_deflection_by_mcr_capacity", self.cdef)
        self.assertNotIn("B2b_failure_announcement_by_mcr_capacity", self.cdef)
        self.assertNotIn("B1b_deflection_by_mcr_capacity", self.info["constraints"])
        self.assertNotIn("B2b_failure_announcement_by_mcr_capacity", self.info["constraints"])
        self.assertFalse(hasattr(hp_analysis, "resolve_active_constraints"))
        from core.analysis_core.checks import structural_checks as sc

        self.assertFalse(hasattr(sc, "DeflectionLimitByMcrCheckEC2004DE"))
        self.assertFalse(hasattr(sc, "FailureAnnouncementByMcrCheckEC2004DE"))

    def test_b1a_b2a_active_and_announce_factor_is_100(self) -> None:
        self.assertTrue(self.info["constraints"]["B1a_deflection_by_wmax_capacity"]["active"])
        self.assertTrue(
            self.info["constraints"]["B2a_failure_announcement_by_wmin_capacity"]["active"]
        )
        meta = self.pdef["defl_max_announce_failure"]
        self.assertEqual(meta["values"][meta["fixed_idx"]], 100.0)

    def test_problem_builder_result_keys_match_active_b_pair(self) -> None:
        b_keys = {
            name
            for name, conf in self.info["constraints"].items()
            if name.startswith("B") and conf.get("active", True)
        }
        self.assertEqual(
            b_keys,
            {
                "B1a_deflection_by_wmax_capacity",
                "B2a_failure_announcement_by_wmin_capacity",
            },
        )


class TestB2aL100Proxy(unittest.TestCase):
    def test_b2a_utilization_around_l_over_100(self) -> None:
        # w > 0 sagging: w = L/100 → util = 1; smaller w → util > 1; larger → util < 1
        L = 8000.0  # mm
        slab = MagicMock()
        slab.slab.L = L
        loads = MagicMock()

        cases = [
            (L / 100.0, 1.0),
            (L / 100.0 * 0.5, 2.0),
            (L / 100.0 * 2.0, 0.5),
        ]
        for w_mm, expected in cases:
            with patch(
                "core.analysis_core.checks.structural_checks.DeflectionCalculator.calculate_deflection_mm_EC",
                return_value=w_mm,
            ):
                u = FailureAnnouncementByDeflectionCheckEC2004DE.calculate_utilization(
                    slab,
                    loads,
                    system=SystemType.SIMPLE_BEAM,
                    min_factor=100.0,
                    invalid_utilization=99.0,
                )
            self.assertAlmostEqual(u, expected, places=6)

    def test_b2a_nonpositive_deflection_is_invalid(self) -> None:
        slab = MagicMock()
        slab.slab.L = 8000.0
        loads = MagicMock()
        for w_mm in (-1.0, -80.0, 0.0):
            with patch(
                "core.analysis_core.checks.structural_checks.DeflectionCalculator.calculate_deflection_mm_EC",
                return_value=w_mm,
            ):
                u = FailureAnnouncementByDeflectionCheckEC2004DE.calculate_utilization(
                    slab,
                    loads,
                    system=SystemType.SIMPLE_BEAM,
                    min_factor=100.0,
                    invalid_utilization=99.0,
                )
            self.assertEqual(u, 99.0)


if __name__ == "__main__":
    unittest.main()
