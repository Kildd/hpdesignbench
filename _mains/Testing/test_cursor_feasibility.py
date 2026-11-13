"""
Fail-closed feasibility and evaluation-status regressions (Phase R1).
"""
from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from core.ioh_core.cache_eval import EvalContext
from core.ioh_core.import_specs import (
    build_space,
    load_constraint_defaults,
    load_materials_registry,
    load_param_defaults,
    load_problems_combined,
    make_decode,
)
from core.ioh_core.problem_builder import _make_violation_reader
from slab_construction.slabs.hp_slab.analysis import (
    failed_evaluation_violation,
    _apply_active_penalties,
    analysis,
)

from _mains.Testing._paths import (
    CONSTRAINT_DEFAULTS_CSV,
    MATERIALS_CSV,
    PARAM_DEFAULTS_CSV,
    PROBLEM_LIST_CSV,
)


class TestPenaltySanitizeUnit(unittest.TestCase):
    """Unit tests for non-finite utilization remapping and penalties."""

    def test_eta_one_is_feasible_boundary(self) -> None:
        (
            values,
            violations,
            penalties,
            psum,
            pfactor,
            y_p,
            remapped,
            analysis_ok,
            analysis_error,
        ) = _apply_active_penalties(
            y=10.0,
            constraint_values={"A": 1.0},
            constraints={"A": {"active": True}},
            invalid_utilization=99.0,
            opt_penalty_alpha=1.0,
            opt_penalty_beta=1.0,
        )
        self.assertEqual(values["A"], 1.0)
        self.assertEqual(violations["A"], 0.0)
        self.assertEqual(penalties["A"], 0.0)
        self.assertEqual(psum, 0.0)
        self.assertEqual(pfactor, 1.0)
        self.assertEqual(y_p, 10.0)
        self.assertEqual(remapped, [])
        self.assertTrue(analysis_ok)
        self.assertIsNone(analysis_error)

    def test_eta_above_one_keeps_analysis_ok(self) -> None:
        (
            _,
            violations,
            penalties,
            _,
            _,
            y_p,
            remapped,
            analysis_ok,
            analysis_error,
        ) = _apply_active_penalties(
            y=10.0,
            constraint_values={"A": 2.0},
            constraints={"A": {"active": True}},
            invalid_utilization=99.0,
            opt_penalty_alpha=1.0,
            opt_penalty_beta=1.0,
        )
        self.assertEqual(violations["A"], 1.0)
        self.assertEqual(penalties["A"], 1.0)
        self.assertEqual(y_p, 20.0)
        self.assertEqual(remapped, [])
        self.assertTrue(analysis_ok)
        self.assertIsNone(analysis_error)

    def test_nan_and_inf_remap_to_sentinel(self) -> None:
        (
            values,
            violations,
            _,
            _,
            _,
            y_p,
            remapped,
            analysis_ok,
            analysis_error,
        ) = _apply_active_penalties(
            y=10.0,
            constraint_values={"A": float("nan"), "B": float("inf")},
            constraints={"A": {"active": True}, "B": {"active": True}},
            invalid_utilization=99.0,
            opt_penalty_alpha=1.0,
            opt_penalty_beta=1.0,
        )
        self.assertEqual(values["A"], 99.0)
        self.assertEqual(values["B"], 99.0)
        self.assertEqual(violations["A"], failed_evaluation_violation(99.0))
        self.assertEqual(violations["B"], failed_evaluation_violation(99.0))
        self.assertTrue(math.isfinite(y_p))
        self.assertGreater(y_p, 10.0)
        self.assertEqual(remapped, ["A", "B"])
        self.assertFalse(analysis_ok)
        self.assertEqual(analysis_error["stage"], "nonfinite_utilization")
        self.assertEqual(analysis_error["constraints"], ["A", "B"])

    def test_inactive_nan_is_ignored(self) -> None:
        (
            values,
            violations,
            _,
            _,
            _,
            y_p,
            remapped,
            analysis_ok,
            analysis_error,
        ) = _apply_active_penalties(
            y=5.0,
            constraint_values={"A": 0.5, "B": float("nan")},
            constraints={"A": {"active": True}, "B": {"active": False}},
            invalid_utilization=99.0,
            opt_penalty_alpha=1.0,
            opt_penalty_beta=1.0,
        )
        self.assertTrue(math.isnan(float(values["B"])))
        self.assertNotIn("B", violations)
        self.assertEqual(y_p, 5.0)
        self.assertEqual(remapped, [])
        self.assertTrue(analysis_ok)
        self.assertIsNone(analysis_error)

    def test_sentinel_ninety_nine_is_finite_penalty(self) -> None:
        (
            _,
            violations,
            penalties,
            _,
            _,
            y_p,
            remapped,
            analysis_ok,
            _,
        ) = _apply_active_penalties(
            y=10.0,
            constraint_values={"A": 99.0},
            constraints={"A": {"active": True}},
            invalid_utilization=99.0,
            opt_penalty_alpha=1.0,
            opt_penalty_beta=1.0,
        )
        self.assertEqual(violations["A"], 98.0)
        self.assertAlmostEqual(penalties["A"], 98.0)  # α=1, β=1 → v^1
        self.assertTrue(math.isfinite(y_p))
        self.assertEqual(remapped, [])
        # Intentional finite sentinel is not a remap; analysis_ok stays True
        # at this helper layer (constraint_feasible will still be False).
        self.assertTrue(analysis_ok)


class TestAnalysisFeasibilityIntegration(unittest.TestCase):
    """Integration checks against Problem 1 analysis()."""

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

    def test_midpoint_is_physically_infeasible_but_analysis_ok(self) -> None:
        mid = [int(round(0.5 * (lo + hi))) for lo, hi in zip(self.lb_idx, self.ub_idx)]
        params = self.decode(mid)
        result = analysis(params, self.constraints, self.materials)

        self.assertTrue(result["analysis_ok"])
        self.assertFalse(result["constraint_feasible"])
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["analysis_error"])
        self.assertEqual(result["remapped_utilizations"], [])
        self.assertTrue(math.isfinite(result["y"]))
        self.assertTrue(math.isfinite(result["y_p"]))
        self.assertGreater(result["y_p"], result["y"])
        self.assertGreater(result["violations_"]["A_bending_capacity"], 0.0)

    def test_material_error_sets_status_flags(self) -> None:
        params = self.decode(list(self.lb_idx))
        params["mat_reinf_id"] = "does_not_exist"
        result = analysis(params, self.constraints, self.materials)
        self.assertFalse(result["analysis_ok"])
        self.assertFalse(result["constraint_feasible"])
        self.assertFalse(result["feasible"])
        self.assertEqual(result["remapped_utilizations"], [])
        self.assertEqual(result["analysis_error"]["stage"], "reinforcement")


class TestEvalContextFailureCache(unittest.TestCase):
    """Unexpected exceptions are cached once and fail closed."""

    def test_unexpected_exception_cached_once(self) -> None:
        from core.analysis_core.run_config import AnalysisRunConfig

        calls = {"n": 0}

        def decode(x):
            return {"x": list(x)}

        def boom(_params):
            calls["n"] += 1
            raise RuntimeError("synthetic analysis failure")

        cfg = AnalysisRunConfig(invalid_utilization=99.0)
        ctx = EvalContext(decode, boom, run_config=cfg)
        ctx.ensure_constraints(["A_bending_capacity", "C1_concrete_cover_capacity"])
        ctx.ensure_params(["dummy"])

        x = [0, 1]
        r1 = ctx.get(x)
        r2 = ctx.get(x)

        self.assertEqual(calls["n"], 1)
        self.assertIs(r1, r2)
        self.assertEqual(ctx.misses, 1)
        self.assertEqual(ctx.hits, 1)
        self.assertFalse(r1["analysis_ok"])
        self.assertFalse(r1["constraint_feasible"])
        self.assertFalse(r1["feasible"])
        self.assertTrue(math.isinf(r1["y_p"]))
        self.assertEqual(r1["analysis_error"]["stage"], "unexpected_exception")
        self.assertIn("RuntimeError", r1["analysis_error"]["exception_type"])
        self.assertIn("traceback", r1["analysis_error"])
        self.assertEqual(
            r1["violations_"]["A_bending_capacity"], failed_evaluation_violation(99.0)
        )

        g_a = _make_violation_reader(ctx, "A_bending_capacity")
        g_c = _make_violation_reader(ctx, "C1_concrete_cover_capacity")
        self.assertEqual(g_a(x), failed_evaluation_violation(99.0))
        self.assertEqual(g_c(x), failed_evaluation_violation(99.0))
        self.assertEqual(calls["n"], 1)

    def test_violation_reader_fail_closed_on_reader_exception(self) -> None:
        from core.analysis_core.run_config import AnalysisRunConfig

        cfg = AnalysisRunConfig(invalid_utilization=99.0)
        ctx = EvalContext(lambda x: {}, lambda p: {}, run_config=cfg)
        g = _make_violation_reader(ctx, "A_bending_capacity")
        with patch.object(ctx, "get", side_effect=RuntimeError("boom")):
            self.assertEqual(g([0]), failed_evaluation_violation(99.0))

    def test_violation_tracks_configured_invalid_utilization(self) -> None:
        from core.analysis_core.run_config import AnalysisRunConfig

        cfg = AnalysisRunConfig(invalid_utilization=50.0)
        self.assertEqual(failed_evaluation_violation(50.0), 49.0)
        ctx = EvalContext(
            lambda x: {},
            lambda p: (_ for _ in ()).throw(RuntimeError("x")),
            run_config=cfg,
        )
        ctx.ensure_constraints(["A"])
        rec = ctx.get([0])
        self.assertEqual(rec["constraint_values"]["A"], 50.0)
        self.assertEqual(rec["violations_"]["A"], 49.0)
        self.assertEqual(_make_violation_reader(ctx, "A")([0]), 49.0)


if __name__ == "__main__":
    unittest.main()
