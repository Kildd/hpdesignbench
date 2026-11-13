"""
C1.3 gate: sensitivity Parquet schema and SALib packing identity fields.
"""
from __future__ import annotations

import math
import unittest

from core.ioh_core.import_specs import (
    build_space,
    load_constraint_defaults,
    load_param_defaults,
    load_problems_combined,
)
from core.ioh_core.workflows.sensitivity_analysis.schema import (
    MATRIX_KIND_A,
    MATRIX_KIND_AB,
    MATRIX_KIND_B,
    SensitivitySchemaError,
    build_sensitivity_schema,
    flatten_sensitivity_row,
    global_index_for,
    matrix_role_for_offset,
    validate_matrix_fields,
)

from _mains.Testing._paths import (
    CONSTRAINT_DEFAULTS_CSV,
    PARAM_DEFAULTS_CSV,
    PROBLEM_LIST_CSV,
)


def _scientific_result(**overrides):
    base = {
        "y": 1.0,
        "y_p": 1.1,
        "constraint_values": {},
        "violations_": {},
        "penalties_": {},
        "penalty_sum": 0.0,
        "penalty_factor": 1.0,
        "params": {},
        "analysis_ok": True,
        "constraint_feasible": True,
        "feasible": True,
        "analysis_error": None,
        "remapped_utilizations": [],
        "cost_components_missing": [],
    }
    base.update(overrides)
    return base


class TestSensitivitySchema(unittest.TestCase):
    """Schema construction, packing roles, and flattening."""

    @classmethod
    def setUpClass(cls) -> None:
        pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
        cdef = load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV)
        problems = load_problems_combined(pdef, cdef, PROBLEM_LIST_CSV)
        cls.info = problems["L6750_B2"]
        cls.var_names, cls.lb_idx, cls.ub_idx = build_space(cls.info["model"])
        cls.factor_names = (
            "geom_h_ges_mm",
            "geom_t_mm",
            "mat_conc_id",
        )
        cls.constraint_names = tuple(cls.info["constraints"].keys())
        cls.schema_info = build_sensitivity_schema(
            var_names=cls.var_names,
            factor_names=cls.factor_names,
            model=cls.info["model"],
            constraint_names=cls.constraint_names,
        )

    def test_schema_contains_identity_and_scientific_columns(self) -> None:
        names = [field.name for field in self.schema_info.schema]
        for required in (
            "global_index",
            "shard_index",
            "sample_index",
            "matrix_offset",
            "matrix_kind",
            "factor_index",
            "factor_name",
            "problem_id",
            "y",
            "y_p",
            "idx__geom_h_ges_mm",
            "param__mat_conc_id",
        ):
            self.assertIn(required, names)
        self.assertTrue(self.schema_info.schema_hash)

    def test_matrix_roles_follow_salib_packing(self) -> None:
        k = len(self.factor_names)
        self.assertEqual(
            matrix_role_for_offset(0, factor_count=k, factor_names=self.factor_names),
            (MATRIX_KIND_A, None, None),
        )
        self.assertEqual(
            matrix_role_for_offset(1, factor_count=k, factor_names=self.factor_names),
            (MATRIX_KIND_AB, 0, "geom_h_ges_mm"),
        )
        self.assertEqual(
            matrix_role_for_offset(k, factor_count=k, factor_names=self.factor_names),
            (MATRIX_KIND_AB, k - 1, "mat_conc_id"),
        )
        self.assertEqual(
            matrix_role_for_offset(
                k + 1, factor_count=k, factor_names=self.factor_names
            ),
            (MATRIX_KIND_B, None, None),
        )
        # Not A,B,AB...: offset 1 is AB, not B.
        self.assertNotEqual(
            matrix_role_for_offset(1, factor_count=k, factor_names=self.factor_names)[0],
            MATRIX_KIND_B,
        )

    def test_validate_matrix_fields_conditional_rules(self) -> None:
        validate_matrix_fields(
            matrix_kind=MATRIX_KIND_A,
            factor_index=None,
            factor_name=None,
            factor_names=self.factor_names,
        )
        with self.assertRaisesRegex(SensitivitySchemaError, "null"):
            validate_matrix_fields(
                matrix_kind=MATRIX_KIND_A,
                factor_index=0,
                factor_name="geom_h_ges_mm",
                factor_names=self.factor_names,
            )
        with self.assertRaisesRegex(SensitivitySchemaError, "does not match"):
            validate_matrix_fields(
                matrix_kind=MATRIX_KIND_AB,
                factor_index=0,
                factor_name="geom_t_mm",
                factor_names=self.factor_names,
            )

    def test_flatten_row_for_a_and_ab(self) -> None:
        indices = list(self.lb_idx)
        result = _scientific_result(
            params={name: self.info["model"][name]["values"][0] for name in self.info["model"]},
            constraint_values={name: 0.5 for name in self.constraint_names},
            violations_={name: 0.0 for name in self.constraint_names},
            penalties_={name: 0.0 for name in self.constraint_names},
        )
        row_a = flatten_sensitivity_row(
            schema_info=self.schema_info,
            global_index=0,
            shard_index=0,
            sample_index=0,
            matrix_offset=0,
            problem_id="L6750_B2",
            evaluation_time_s=0.01,
            catalogue_indices=indices,
            result=result,
        )
        self.assertEqual(row_a["matrix_kind"], MATRIX_KIND_A)
        self.assertIsNone(row_a["factor_index"])
        self.assertIsNone(row_a["factor_name"])

        row_ab = flatten_sensitivity_row(
            schema_info=self.schema_info,
            global_index=1,
            shard_index=0,
            sample_index=0,
            matrix_offset=1,
            problem_id="L6750_B2",
            evaluation_time_s=0.01,
            catalogue_indices=indices,
            result=result,
        )
        self.assertEqual(row_ab["matrix_kind"], MATRIX_KIND_AB)
        self.assertEqual(row_ab["factor_index"], 0)
        self.assertEqual(row_ab["factor_name"], "geom_h_ges_mm")

        with self.assertRaisesRegex(SensitivitySchemaError, "global_index"):
            flatten_sensitivity_row(
                schema_info=self.schema_info,
                global_index=99,
                shard_index=0,
                sample_index=0,
                matrix_offset=0,
                problem_id="L6750_B2",
                evaluation_time_s=0.01,
                catalogue_indices=indices,
                result=result,
            )

    def test_global_index_helper(self) -> None:
        k = 3
        self.assertEqual(
            global_index_for(sample_index=2, matrix_offset=4, factor_count=k),
            2 * (k + 2) + 4,
        )

    def test_nan_y_preserved(self) -> None:
        indices = list(self.lb_idx)
        result = _scientific_result(
            y=float("nan"),
            params={name: self.info["model"][name]["values"][0] for name in self.info["model"]},
            constraint_values={name: 0.0 for name in self.constraint_names},
            violations_={name: 0.0 for name in self.constraint_names},
            penalties_={name: 0.0 for name in self.constraint_names},
        )
        row = flatten_sensitivity_row(
            schema_info=self.schema_info,
            global_index=0,
            shard_index=0,
            sample_index=0,
            matrix_offset=0,
            problem_id="L6750_B2",
            evaluation_time_s=0.0,
            catalogue_indices=indices,
            result=result,
        )
        self.assertTrue(math.isnan(row["y"]))


if __name__ == "__main__":
    unittest.main()
