"""
Analysis result-shape regressions for hp_slab.analysis().

Success and material-error paths share the same key set; the error path
sets feasible=False and fills analysis_error.
"""
from __future__ import annotations

import math
import unittest

from core.ioh_core.import_specs import (
    build_space,
    load_constraint_defaults,
    load_materials_registry,
    load_param_defaults,
    load_problems_combined,
    make_decode,
)
from slab_construction.slabs.hp_slab.analysis import analysis

from _mains.Testing._paths import (
    CONSTRAINT_DEFAULTS_CSV,
    MATERIALS_CSV,
    PARAM_DEFAULTS_CSV,
    PROBLEM_LIST_CSV,
)

_RESULT_KEYS = {
    "y",
    "y_p",
    "constraint_values",
    "violations_",
    "penalties_",
    "penalty_sum",
    "penalty_factor",
    "params",
    "analysis_ok",
    "constraint_feasible",
    "feasible",
    "analysis_error",
    "remapped_utilizations",
    "cost_components_missing",
}


class TestAnalysisResultShape(unittest.TestCase):
    """Assert uniform success/error dictionary shapes."""

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

    def test_material_error_path_has_uniform_schema(self) -> None:
        params = self.decode(list(self.lb_idx))
        params["mat_reinf_id"] = "does_not_exist"

        result = analysis(params, self.constraints, self.materials)

        self.assertEqual(set(result.keys()), _RESULT_KEYS)
        self.assertTrue(math.isinf(result["y"]))
        self.assertTrue(math.isinf(result["y_p"]))
        self.assertFalse(result["analysis_ok"])
        self.assertFalse(result["constraint_feasible"])
        self.assertFalse(result["feasible"])
        self.assertEqual(result["remapped_utilizations"], [])
        self.assertIsInstance(result["analysis_error"], dict)
        self.assertEqual(result["analysis_error"]["stage"], "reinforcement")
        self.assertEqual(result["analysis_error"]["material_id"], "does_not_exist")
        self.assertEqual(result["violations_"], {})
        self.assertEqual(result["penalties_"], {})
        self.assertEqual(result["constraint_values"], {})
        self.assertAlmostEqual(result["penalty_factor"], 1.0)
        self.assertNotIn("violations", result)

    def test_success_path_has_uniform_schema(self) -> None:
        mid = [int(round(0.5 * (lo + hi))) for lo, hi in zip(self.lb_idx, self.ub_idx)]
        params = self.decode(mid)
        result = analysis(params, self.constraints, self.materials)
        self.assertEqual(set(result.keys()), _RESULT_KEYS)
        self.assertIsInstance(result["remapped_utilizations"], list)
        self.assertIn("analysis_ok", result)
        self.assertIn("constraint_feasible", result)


if __name__ == "__main__":
    unittest.main()
