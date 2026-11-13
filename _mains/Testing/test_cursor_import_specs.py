"""
Decode/domain regressions for the Grasshopper-aligned hp_slab CSV configuration.
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from core.analysis_core.statics.loads import LoadsEC
from core.ioh_core.import_specs import (
    build_space,
    load_constraint_defaults,
    load_param_defaults,
    load_problems_combined,
    make_decode,
)

from _mains.Testing._paths import CONSTRAINT_DEFAULTS_CSV, PARAM_DEFAULTS_CSV, PROBLEM_LIST_CSV


class TestHpSlabParamDefaults(unittest.TestCase):
    """Assert parameter_defaults.csv domains and roles after Step 2a."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_param_defaults(PARAM_DEFAULTS_CSV)

    def test_mat_reinf_id_is_lookup_on_a_tex(self) -> None:
        meta = self.spec["mat_reinf_id"]
        self.assertEqual(meta["role"], "lookup")
        self.assertEqual(meta["lookup_key"], "reinf_a_tex_mm2")
        self.assertEqual(
            meta["values"],
            [
                "solidian Roving 1R (Z-1.6-308)",
                "solidian Roving 2R (Z-1.6-308)",
                "solidian Roving 2R (Z-1.6-308)",
                "solidian Roving 2R (Z-1.6-308)",
            ],
        )
        self.assertEqual(
            len(meta["values"]),
            len(self.spec["reinf_a_tex_mm2"]["values"]),
        )

    def test_reinf_a_tex_catalog(self) -> None:
        vals = self.spec["reinf_a_tex_mm2"]["values"]
        self.assertEqual(vals, [1.81, 3.62, 7.24, 10.86])
        self.assertEqual(self.spec["reinf_a_tex_mm2"]["role"], "var")

    def test_geom_nt_gh_domain(self) -> None:
        vals = self.spec["geom_nt"]["values"]
        self.assertEqual(vals, [5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        self.assertEqual(self.spec["geom_nt"]["lb"], 0)
        self.assertEqual(self.spec["geom_nt"]["ub"], 5)

    def test_geom_span_grid_includes_gh_spans(self) -> None:
        vals = self.spec["geom_span_mm"]["values"]
        self.assertEqual(len(vals), 10801)
        self.assertEqual(vals[0], 0.0)
        self.assertEqual(vals[6750], 6750.0)
        self.assertEqual(vals[7425], 7425.0)
        self.assertEqual(vals[8100], 8100.0)
        self.assertEqual(vals[8775], 8775.0)
        self.assertEqual(vals[9450], 9450.0)
        self.assertEqual(vals[10125], 10125.0)
        self.assertEqual(vals[10800], 10800.0)
        self.assertEqual(self.spec["geom_span_mm"]["fixed_idx"], 6750)

    def test_optimization_controls_defaults(self) -> None:
        alpha = self.spec["opt_penalty_alpha"]
        beta = self.spec["opt_penalty_beta"]
        sentinel = self.spec["opt_invalid_utilization"]
        self.assertEqual(alpha["values"][alpha["fixed_idx"]], 1.0)
        self.assertAlmostEqual(
            beta["values"][beta["fixed_idx"]], 1.0 / 3.0
        )
        self.assertEqual(sentinel["values"][sentinel["fixed_idx"]], 99.0)

    def test_loads_category_default_is_b2(self) -> None:
        meta = self.spec["loads_category"]
        self.assertEqual(meta["values"][meta["fixed_idx"]], "B2")

    def test_acoustic_defaults_match_gh_wired_limits(self) -> None:
        rw = self.spec["insu_rw_req_db"]
        lnw = self.spec["insu_lnw_max_db"]
        self.assertEqual(rw["values"][rw["fixed_idx"]], 54.0)
        self.assertEqual(lnw["values"][lnw["fixed_idx"]], 53.0)

    def test_objective_weight_defaults_are_gwp_only(self) -> None:
        w_gwp = self.spec["weight_omega_1_gwp"]
        w_cost = self.spec["weight_omega_2_cost"]
        self.assertEqual(w_gwp["values"][w_gwp["fixed_idx"]], 1.0)
        self.assertEqual(w_cost["values"][w_cost["fixed_idx"]], 0.0)

    def test_mat_conc_id_is_var_catalog(self) -> None:
        meta = self.spec["mat_conc_id"]
        self.assertEqual(meta["role"], "var")
        self.assertEqual(meta["value_type"], "str")
        self.assertEqual(meta["values"][3], "C25/30")
        self.assertEqual(meta["values"][4], "C30/37")
        self.assertEqual(meta["values"][-1], "C100/115")
        self.assertNotIn("mat_conc_fck", self.spec)
        self.assertNotIn("mat_conc_cost_eur_m3", self.spec)
        self.assertNotIn("mat_conc_gwp_kgco2e_m3", self.spec)


class TestHpSlabL6750B2Decode(unittest.TestCase):
    """Assert L6750_B2 overlays and decode behaviour."""

    @classmethod
    def setUpClass(cls) -> None:
        pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
        cdef = load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV)
        cls.problems = load_problems_combined(pdef, cdef, PROBLEM_LIST_CSV)
        cls.info = cls.problems["L6750_B2"]
        cls.var_names, cls.lb_idx, cls.ub_idx = build_space(cls.info["model"])
        cls.decode = staticmethod(make_decode(cls.info["model"], cls.var_names))

    def test_l6750_b2_exists_with_nine_variables(self) -> None:
        self.assertIn("L6750_B2", self.problems)
        self.assertNotIn("default", self.problems)
        self.assertNotIn("1", self.problems)
        self.assertEqual(self.info["label"], "L=6.75 m B2")
        self.assertEqual(len(self.var_names), 9)
        self.assertEqual(
            self.var_names,
            [
                "geom_h_ges_mm",
                "geom_t_mm",
                "geom_nt",
                "geom_dy_mm",
                "mat_conc_id",
                "reinf_kap_t_percent",
                "reinf_a_tex_mm2",
                "geom_t_infill_mm",
                "geom_t_screed_mm",
            ],
        )
        # lookup params must not appear as optimizer variables
        self.assertNotIn("mat_reinf_id", self.var_names)

    def test_l6750_b2_geom_nt_bounds_match_gh_grid(self) -> None:
        i = self.var_names.index("geom_nt")
        self.assertEqual(self.lb_idx[i], 0)
        self.assertEqual(self.ub_idx[i], 5)

    def test_mat_reinf_id_is_lookup_not_problem_overlay(self) -> None:
        meta = self.info["model"]["mat_reinf_id"]
        self.assertEqual(meta["role"], "lookup")
        self.assertEqual(meta["lookup_key"], "reinf_a_tex_mm2")

    def test_twenty_one_instances_resolve_span_and_load(self) -> None:
        spans = (6750, 7425, 8100, 8775, 9450, 10125, 10800)
        loads = (("A1", 0, 1.0), ("B1", 3, 2.0), ("B2", 4, 3.0))
        expected_ids = tuple(
            f"L{span}_{cat}" for span in spans for cat, _idx, _qk in loads
        )
        self.assertEqual(set(self.problems), set(expected_ids))
        for span in spans:
            for cat, load_idx, qk in loads:
                problem_id = f"L{span}_{cat}"
                info = self.problems[problem_id]
                span_meta = info["model"]["geom_span_mm"]
                load_meta = info["model"]["loads_category"]
                self.assertEqual(span_meta["fixed_idx"], span)
                self.assertEqual(span_meta["values"][span], float(span))
                self.assertEqual(load_meta["fixed_idx"], load_idx)
                self.assertEqual(load_meta["values"][load_idx], cat)
                names, lb, _ub = build_space(info["model"])
                decode = make_decode(info["model"], names)
                params = decode(list(lb))
                self.assertEqual(params["geom_span_mm"], float(span))
                self.assertEqual(params["loads_category"], cat)
                loads_ec = LoadsEC.from_categories_EC0_NA_DE(params["loads_category"])
                self.assertAlmostEqual(float(loads_ec.Qk[0]), qk)

    def test_decode_mid_indices_values(self) -> None:
        mid = [(lo + (hi - lo) // 2) for lo, hi in zip(self.lb_idx, self.ub_idx)]
        params = self.decode(mid)

        self.assertEqual(params["geom_span_mm"], 6750.0)
        self.assertEqual(params["geom_b_mm"], 1350.0)
        self.assertEqual(params["loads_category"], "B2")
        self.assertEqual(params["loads_N_kN"], 0.0)
        self.assertEqual(params["insu_rw_req_db"], 54.0)
        self.assertEqual(params["insu_lnw_max_db"], 53.0)
        self.assertEqual(params["weight_omega_1_gwp"], 1.0)
        self.assertEqual(params["weight_omega_2_cost"], 0.0)
        self.assertEqual(params["opt_penalty_alpha"], 1.0)
        self.assertAlmostEqual(params["opt_penalty_beta"], 1.0 / 3.0)
        self.assertEqual(params["opt_invalid_utilization"], 99.0)

        # Concrete id from catalog (mid index within lb=3..ub=14 → idx 8)
        self.assertEqual(params["mat_conc_id"], "C50/60")
        # C30 refs come from materials registry in analysis, not decode
        self.assertNotIn("mat_conc_cost_c30_ref_eur_m3", params)
        self.assertNotIn("mat_conc_gwp_c30_ref_kgco2e_m3", params)

    def test_reinf_a_tex_selects_material_0_1_1_1(self) -> None:
        expected = [
            "solidian Roving 1R (Z-1.6-308)",
            "solidian Roving 2R (Z-1.6-308)",
            "solidian Roving 2R (Z-1.6-308)",
            "solidian Roving 2R (Z-1.6-308)",
        ]
        tex_i = self.var_names.index("reinf_a_tex_mm2")
        for tex_idx, expected_mat in enumerate(expected):
            x = list(self.lb_idx)
            x[tex_i] = tex_idx
            params = self.decode(x)
            self.assertEqual(params["mat_reinf_id"], expected_mat)
            self.assertEqual(
                params["reinf_a_tex_mm2"],
                [1.81, 3.62, 7.24, 10.86][tex_idx],
            )

    def test_unresolved_lookup_key_raises(self) -> None:
        """Catalog drift must not silently fall back to material index 0."""
        import copy

        model = copy.deepcopy(self.info["model"])

        class _DecodeOnlyList(list):
            """``values[i]`` returns an alien value; ``index`` uses the real catalog."""

            def __getitem__(self, i):
                if isinstance(i, int):
                    return 999.0
                return super().__getitem__(i)

        catalog = list(model["reinf_a_tex_mm2"]["values"])
        model["reinf_a_tex_mm2"]["values"] = _DecodeOnlyList(catalog)
        decode_broken = make_decode(model, self.var_names)
        with self.assertRaises(ValueError) as ctx:
            decode_broken(list(self.lb_idx))
        msg = str(ctx.exception)
        self.assertIn("Silent fallback to index 0 is not allowed", msg)
        self.assertIn("mat_reinf_id", msg)
        self.assertIn("999.0", msg)

    def test_default_live_load_qk_is_3(self) -> None:
        params = self.decode(list(self.lb_idx))
        loads = LoadsEC.from_categories_EC0_NA_DE(params["loads_category"])
        self.assertEqual(params["loads_category"], "B2")
        self.assertAlmostEqual(float(loads.Qk[0]), 3.0)

    def test_z_constraints_present_and_active(self) -> None:
        for name in (
            "Z1_nt_dt_combination_capacity",
            "Z2_beam_theory_H_L_capacity",
            "Z3_beam_theory_B_L_capacity",
        ):
            self.assertIn(name, self.info["constraints"])
            self.assertTrue(self.info["constraints"][name]["active"])


class TestProblemDefaultInheritance(unittest.TestCase):
    """Validate the reserved default and sparse problem overlay contract."""

    fields = [
        "problem_id", "type", "name", "role",
        "lb_idx", "ub_idx", "fixed_idx",
        "active", "enforced", "weight", "exponent",
        "label",
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls.pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
        cls.cdef = load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV)

    def _load(self, rows: list[dict[str, object]]) -> dict[str, dict]:
        """
        Write and load a temporary problem-list CSV.

        Parameters
        ----------
        rows : list of dict
            Row values keyed by :attr:`fields`; omitted cells are blank.

        Returns
        -------
        dict
            Resolved ordinary problem mapping.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "problem_list.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=self.fields,
                    delimiter=";",
                    extrasaction="raise",
                )
                writer.writeheader()
                writer.writerows(rows)
            return load_problems_combined(self.pdef, self.cdef, path)

    @staticmethod
    def _default_row(**overrides: object) -> dict[str, object]:
        """
        Return one valid default-group constraint overlay.

        Parameters
        ----------
        **overrides : object
            Cells replacing the standard default-row values.

        Returns
        -------
        dict
            CSV row mapping.
        """
        row: dict[str, object] = {
            "problem_id": "default",
            "type": "constraint",
            "name": "A_bending_capacity",
            "active": "True",
        }
        row.update(overrides)
        return row

    def test_declaration_only_problem_equals_default_and_default_is_hidden(self) -> None:
        problems = self._load(
            [
                self._default_row(label="Reference"),
                {"problem_id": "1", "label": "Exact default"},
            ]
        )

        self.assertEqual(set(problems), {"1"})
        self.assertEqual(problems["1"]["model"], self.pdef)
        self.assertEqual(problems["1"]["constraints"], self.cdef)
        self.assertEqual(problems["1"]["label"], "Exact default")

    def test_sparse_constraint_override_inherits_everything_else(self) -> None:
        problems = self._load(
            [
                self._default_row(),
                {"problem_id": "1"},
                {
                    "problem_id": "2",
                    "type": "constraint",
                    "name": "A_bending_capacity",
                    "active": "False",
                    "weight": "0",
                },
            ]
        )
        exact = problems["1"]
        sparse = problems["2"]

        self.assertEqual(sparse["model"], exact["model"])
        for name, config in exact["constraints"].items():
            if name != "A_bending_capacity":
                self.assertEqual(sparse["constraints"][name], config)
        self.assertFalse(sparse["constraints"]["A_bending_capacity"]["active"])
        self.assertEqual(sparse["constraints"]["A_bending_capacity"]["weight"], 0.0)
        self.assertTrue(exact["constraints"]["A_bending_capacity"]["active"])
        self.assertEqual(exact["constraints"]["A_bending_capacity"]["weight"], 1.0)

    def test_sparse_problems_are_independent_deep_copies(self) -> None:
        problems = self._load(
            [
                self._default_row(),
                {"problem_id": "1"},
                {"problem_id": "2"},
            ]
        )
        first_values = problems["1"]["model"]["geom_h_ges_mm"]["values"]
        second_values = problems["2"]["model"]["geom_h_ges_mm"]["values"]
        self.assertIsNot(first_values, second_values)
        first_values[0] = -999.0
        self.assertNotEqual(first_values[0], second_values[0])

    def test_missing_or_invalid_default_group_fails(self) -> None:
        cases = {
            "missing": [{"problem_id": "1", "label": "ordinary"}],
            "non_contiguous": [
                self._default_row(),
                {"problem_id": "1"},
                self._default_row(name="C1_concrete_cover_capacity"),
            ],
            "empty": [
                {"problem_id": "default", "label": "reference"},
                {"problem_id": "1"},
            ],
        }
        for name, rows in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "default"):
                    self._load(rows)

    def test_duplicate_override_and_conflicting_labels_fail(self) -> None:
        duplicate = [
            self._default_row(),
            {
                "problem_id": "1",
                "type": "constraint",
                "name": "A_bending_capacity",
                "active": "False",
            },
            {
                "problem_id": "1",
                "type": "constraint",
                "name": "A_bending_capacity",
                "active": "True",
            },
        ]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            self._load(duplicate)

        labels = [
            self._default_row(),
            {"problem_id": "1", "label": "first"},
            {
                "problem_id": "1",
                "type": "param",
                "name": "geom_h_ges_mm",
                "label": "second",
            },
        ]
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            self._load(labels)

    def test_malformed_rows_and_invalid_boolean_fail(self) -> None:
        cases = {
            "blank_id": [
                self._default_row(),
                {"problem_id": "", "label": "invalid"},
            ],
            "unknown_type": [
                self._default_row(),
                {"problem_id": "1", "type": "other", "name": "anything"},
            ],
            "unknown_parameter": [
                self._default_row(),
                {"problem_id": "1", "type": "param", "name": "unknown"},
            ],
            "unknown_constraint": [
                self._default_row(),
                {"problem_id": "1", "type": "constraint", "name": "unknown"},
            ],
            "invalid_boolean": [
                self._default_row(),
                {
                    "problem_id": "1",
                    "type": "constraint",
                    "name": "A_bending_capacity",
                    "active": "flase",
                },
            ],
        }
        for name, rows in cases.items():
            with self.subTest(name=name):
                with self.assertRaises((KeyError, ValueError)):
                    self._load(rows)


if __name__ == "__main__":
    unittest.main()
