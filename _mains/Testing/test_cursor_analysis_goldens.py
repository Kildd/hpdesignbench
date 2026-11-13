"""
R6 goldens: reference-slab util checks and Problem-1 midpoint analysis.
"""
from __future__ import annotations

import math
import unittest

from _mains.Testing._paths import (
    CONSTRAINT_DEFAULTS_CSV,
    MATERIALS_CSV,
    PARAM_DEFAULTS_CSV,
    PROBLEM_LIST_CSV,
)
from _mains.Testing.fixtures.floor import ref_floor
from _mains.Testing.fixtures.hp_slabs import hp_slab_ref
from _mains.Testing.fixtures.loads import test_loads
from core.analysis_core.checks.structural_checks import (
    DeflectionLimitByDeflectionCheckEC2004DE,
    FailureAnnouncementByDeflectionCheckEC2004DE,
    UltimateMomentCheckEC2004DE,
)
from core.analysis_core.section_methods import (
    prepare_deflection_mk_cache,
    prepare_reference_analysis_cache_EC,
)
from core.analysis_core.statics.constants import MomentType, SystemType
from core.ioh_core.import_specs import (
    build_space,
    load_constraint_defaults,
    load_materials_registry,
    load_param_defaults,
    load_problems_combined,
    make_decode,
)
from slab_construction.slab_construction import SlabConstruction
from slab_construction.slabs.hp_slab.analysis import analysis


class TestReferenceSlabUtilizations(unittest.TestCase):
    """Loutfi reference HP-shell + floor; production B1a/B2a M–κ cache."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.slab = SlabConstruction(hp_slab_ref, ref_floor)
        prepare_reference_analysis_cache_EC(cls.slab, n=0.0)
        prepare_deflection_mk_cache(cls.slab, n=0.0)

    def test_uls_and_sls_utilizations(self) -> None:
        a = UltimateMomentCheckEC2004DE.calculate_utilization(
            self.slab,
            test_loads,
            system=SystemType.SIMPLE_BEAM,
            moment=MomentType.MAX_POS_MOMENT,
            n=0.0,
        )
        b1a = DeflectionLimitByDeflectionCheckEC2004DE.calculate_utilization(
            self.slab,
            test_loads,
            system=SystemType.SIMPLE_BEAM,
            limit_factor=250.0,
            n=0.0,
        )
        b2a = FailureAnnouncementByDeflectionCheckEC2004DE.calculate_utilization(
            self.slab,
            test_loads,
            system=SystemType.SIMPLE_BEAM,
            min_factor=100.0,
            n=0.0,
        )
        # Locked 2026-07-27 (R6); relative band catches silent drift.
        self.assertAlmostEqual(a, 1.0446764225179679, places=5)
        self.assertAlmostEqual(b1a, 0.27416763399573624, places=5)
        self.assertAlmostEqual(b2a, 0.7213978653506765, places=5)


class TestProblem1MidpointAnalysis(unittest.TestCase):
    """Production ``analysis()`` midpoint golden (materials.csv objective)."""

    @classmethod
    def setUpClass(cls) -> None:
        pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
        cdef = load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV)
        problems = load_problems_combined(pdef, cdef, PROBLEM_LIST_CSV)
        info = problems["L6750_B2"]
        cls.var_names, cls.lb_idx, cls.ub_idx = build_space(info["model"])
        cls.decode = staticmethod(make_decode(info["model"], cls.var_names))
        cls.constraints = {
            name: {
                "active": c.get("active", True),
                "enforced": c.get("enforced"),
                "weight": float(c.get("weight", 1)),
                "exponent": float(c.get("exponent", 1)),
            }
            for name, c in info["constraints"].items()
        }
        cls.materials = load_materials_registry(MATERIALS_CSV)

    def test_midpoint_objective_and_key_utils(self) -> None:
        mid = [int(round(0.5 * (lo + hi))) for lo, hi in zip(self.lb_idx, self.ub_idx)]
        result = analysis(self.decode(mid), self.constraints, self.materials)
        cv = result["constraint_values"]

        self.assertTrue(result["analysis_ok"])
        self.assertFalse(result["feasible"])
        self.assertTrue(math.isfinite(result["y"]))
        self.assertTrue(math.isfinite(result["y_p"]))
        self.assertGreater(result["y_p"], result["y"])

        # Locked 2026-08-03 after penalty reformulation:
        # p_i = α · v_i^β with catalogue α=1, β=1/3 (no nested square root).
        self.assertAlmostEqual(result["y"], 455.0512514305275, places=3)
        self.assertAlmostEqual(result["y_p"], 879.6436712620355, places=1)
        self.assertAlmostEqual(cv["A_bending_capacity"], 1.0776111778632105, places=4)
        self.assertAlmostEqual(
            cv["B1a_deflection_by_wmax_capacity"], 0.26509358188026905, places=4
        )
        self.assertAlmostEqual(
            cv["B2a_failure_announcement_by_wmin_capacity"],
            0.750274355046204,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
