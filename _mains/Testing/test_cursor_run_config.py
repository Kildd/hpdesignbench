"""
Phase R2 regressions: frozen AnalysisRunConfig and CSV-sourced sentinel.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from core.analysis_core.checks.construction_checks import MidlineConcreteCoverCheck
from core.analysis_core.checks.modeling_checks import (
    BeamTheoryBLRatioCheck,
    BeamTheoryHgesLRatioCheck,
    NtDyCombinationCheck,
)
from core.analysis_core.run_config import (
    AnalysisRunConfig,
    analysis_run_config_from_params,
    apply_analysis_run_config,
    reset_applied_analysis_run_config_for_tests,
)
from core.ioh_core.import_specs import (
    build_space,
    load_constraint_defaults,
    load_materials_registry,
    load_param_defaults,
    load_problems_combined,
    make_decode,
)
from slab_construction.slabs.hp_slab.analysis import (
    analysis,
    failed_evaluation_violation,
)

from _mains.Testing._paths import (
    CONSTRAINT_DEFAULTS_CSV,
    MATERIALS_CSV,
    PARAM_DEFAULTS_CSV,
    PROBLEM_LIST_CSV,
)
from _mains.Testing.test_cursor_construction_checks import _slab_with_shell


def _slab_with_geometry(**geometry) -> MagicMock:
    slab_construction = MagicMock()
    for name, value in geometry.items():
        setattr(slab_construction.slab.hp_shell.hp_geometry, name, value)
    return slab_construction


class TestAnalysisRunConfig(unittest.TestCase):
    """Assert run-config wiring and CSV-sourced invalid utilization."""

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
        cls.csv_invalid = float(pdef["opt_invalid_utilization"]["values"][0])

    def setUp(self) -> None:
        reset_applied_analysis_run_config_for_tests()

    def tearDown(self) -> None:
        reset_applied_analysis_run_config_for_tests()

    def _mid_params(self) -> dict:
        mid = [int(round(0.5 * (lo + hi))) for lo, hi in zip(self.lb_idx, self.ub_idx)]
        return self.decode(mid)

    def test_run_config_reads_sentinel_from_params_csv(self) -> None:
        params = self._mid_params()
        self.assertIn("opt_invalid_utilization", params)
        cfg = analysis_run_config_from_params(params, slab_type="hp_slab")
        self.assertEqual(cfg.invalid_utilization, self.csv_invalid)
        self.assertEqual(self.csv_invalid, 99.0)

    def test_missing_opt_invalid_utilization_raises(self) -> None:
        with self.assertRaises(KeyError):
            analysis_run_config_from_params({"geom_L_mm": 6000.0})

    def test_explicit_invalid_utilization_is_used_by_checks(self) -> None:
        hp_shell = MagicMock()
        hp_shell.d_p.return_value = 10.0
        hp_shell.c_1_clear_concrete_cover.return_value = 0.0
        u = MidlineConcreteCoverCheck.calculate_utilization(
            _slab_with_shell(hp_shell),
            invalid_utilization=55.0,
        )
        self.assertEqual(u, 55.0)

    def test_alternating_explicit_sentinels_have_no_call_order_leakage(self) -> None:
        hp_shell = MagicMock()
        hp_shell.d_p.return_value = 10.0
        hp_shell.c_1_clear_concrete_cover.return_value = 0.0
        slab = _slab_with_shell(hp_shell)
        u_a = MidlineConcreteCoverCheck.calculate_utilization(
            slab, invalid_utilization=40.0
        )
        u_b = MidlineConcreteCoverCheck.calculate_utilization(
            slab, invalid_utilization=80.0
        )
        u_a2 = MidlineConcreteCoverCheck.calculate_utilization(
            slab, invalid_utilization=40.0
        )
        self.assertEqual(u_a, 40.0)
        self.assertEqual(u_b, 80.0)
        self.assertEqual(u_a2, 40.0)

    def test_failed_evaluation_violation_tracks_explicit_sentinel(self) -> None:
        self.assertEqual(failed_evaluation_violation(55.0), 54.0)
        self.assertEqual(failed_evaluation_violation(self.csv_invalid), 98.0)

    def test_analysis_uses_csv_sentinel_via_run_config(self) -> None:
        params = self._mid_params()
        cfg = analysis_run_config_from_params(params, slab_type="hp_slab")
        result = analysis(
            params, self.constraints, self.materials, run_config=cfg
        )
        self.assertIn("feasible", result)
        self.assertEqual(cfg.invalid_utilization, self.csv_invalid)

    def test_apply_design_code_is_idempotent_for_same_value(self) -> None:
        cfg = AnalysisRunConfig(
            design_code="ec2_2004",
            invalid_utilization=self.csv_invalid,
        )
        apply_analysis_run_config(cfg)
        apply_analysis_run_config(cfg)
        self.assertEqual(cfg.design_code, "ec2_2004")


class TestModelingConstraintSentinel(unittest.TestCase):
    """Z2/Z3 keep an in-range ratio and return the sentinel above 1."""

    def test_z2_ratio_below_and_at_limit_is_unchanged(self) -> None:
        below = BeamTheoryHgesLRatioCheck.calculate_utilization(
            _slab_with_geometry(Hx=40.0, Hy=60.0, L=1000.0),
            f_b=5.0,
            invalid_utilization=99.0,
        )
        at_limit = BeamTheoryHgesLRatioCheck.calculate_utilization(
            _slab_with_geometry(Hx=100.0, Hy=100.0, L=1000.0),
            f_b=5.0,
            invalid_utilization=99.0,
        )
        self.assertAlmostEqual(below, 0.5)
        self.assertAlmostEqual(at_limit, 1.0)

    def test_z2_ratio_above_limit_returns_explicit_sentinel(self) -> None:
        slab = _slab_with_geometry(Hx=200.0, Hy=200.0, L=1000.0)
        self.assertEqual(
            BeamTheoryHgesLRatioCheck.calculate_utilization(
                slab, f_b=5.0, invalid_utilization=99.0
            ),
            99.0,
        )
        self.assertEqual(
            BeamTheoryHgesLRatioCheck.calculate_utilization(
                slab, f_b=5.0, invalid_utilization=55.0
            ),
            55.0,
        )

    def test_z3_ratio_below_and_at_limit_is_unchanged(self) -> None:
        below = BeamTheoryBLRatioCheck.calculate_utilization(
            _slab_with_geometry(B=100.0, L=1000.0),
            f_b=5.0,
            invalid_utilization=99.0,
        )
        at_limit = BeamTheoryBLRatioCheck.calculate_utilization(
            _slab_with_geometry(B=200.0, L=1000.0),
            f_b=5.0,
            invalid_utilization=99.0,
        )
        self.assertAlmostEqual(below, 0.5)
        self.assertAlmostEqual(at_limit, 1.0)

    def test_z3_ratio_above_limit_returns_explicit_sentinel(self) -> None:
        slab = _slab_with_geometry(B=400.0, L=1000.0)
        self.assertEqual(
            BeamTheoryBLRatioCheck.calculate_utilization(
                slab, f_b=5.0, invalid_utilization=99.0
            ),
            99.0,
        )
        self.assertEqual(
            BeamTheoryBLRatioCheck.calculate_utilization(
                slab, f_b=5.0, invalid_utilization=55.0
            ),
            55.0,
        )

    def test_z1_still_returns_one_or_sentinel(self) -> None:
        matched = NtDyCombinationCheck.calculate_utilization(
            _slab_with_geometry(nt=1, dy=10.0, dy_real=lambda: 10.0),
            invalid_utilization=99.0,
        )
        mismatched = NtDyCombinationCheck.calculate_utilization(
            _slab_with_geometry(nt=1, dy=50.0, dy_real=lambda: 10.0),
            invalid_utilization=55.0,
        )
        several = NtDyCombinationCheck.calculate_utilization(
            _slab_with_geometry(nt=2, dy=50.0, dy_real=lambda: 10.0),
            invalid_utilization=99.0,
        )
        self.assertEqual(matched, 1.0)
        self.assertEqual(mismatched, 55.0)
        self.assertEqual(several, 1.0)


if __name__ == "__main__":
    unittest.main()
