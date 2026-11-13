"""
Phase B4 gate: stable brute-force Parquet schema and row flattening.
"""
from __future__ import annotations

import io
import math
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from core.ioh_core.import_specs import (
    build_space,
    load_constraint_defaults,
    load_materials_registry,
    load_param_defaults,
    load_problems_combined,
    make_decode,
)
from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.brute_force.schema import (
    BruteForceSchemaError,
    build_brute_force_schema,
    compute_schema_hash,
    flatten_evaluation_row,
)
from slab_construction.slabs.hp_slab.analysis import analysis

from _mains.Testing._paths import (
    CONSTRAINT_DEFAULTS_CSV,
    MATERIALS_CSV,
    PARAM_DEFAULTS_CSV,
    PROBLEM_LIST_CSV,
)


class TestBruteForceSchema(unittest.TestCase):
    """Schema construction, flattening, and Parquet round-trips."""

    @classmethod
    def setUpClass(cls) -> None:
        pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
        cdef = load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV)
        problems = load_problems_combined(pdef, cdef, PROBLEM_LIST_CSV)
        cls.info = problems["L6750_B2"]
        cls.var_names, cls.lb_idx, cls.ub_idx = build_space(cls.info["model"])
        cls.space = IndexSpace.from_bounds(cls.var_names, cls.lb_idx, cls.ub_idx)
        cls.constraint_names = tuple(cls.info["constraints"].keys())
        cls.schema_info = build_brute_force_schema(
            space=cls.space,
            model=cls.info["model"],
            constraint_names=cls.constraint_names,
        )
        cls.decode = staticmethod(make_decode(cls.info["model"], cls.var_names))
        cls.constraints = {
            name: {
                "active": c.get("active", True),
                "enforced": c.get("enforced"),
                "weight": float(c.get("weight", 1)),
                "exponent": float(c.get("exponent", 1)),
            }
            for name, c in cls.info["constraints"].items()
        }
        cls.materials = load_materials_registry(MATERIALS_CSV)

    def test_schema_contains_required_columns(self) -> None:
        names = [field.name for field in self.schema_info.schema]
        for required in (
            "global_index",
            "shard_index",
            "problem_id",
            "evaluation_time_s",
            "analysis_ok",
            "constraint_feasible",
            "feasible",
            "y",
            "y_p",
            "penalty_sum",
            "penalty_factor",
            "analysis_error_json",
            "cost_components_missing",
            "remapped_utilizations",
        ):
            self.assertIn(required, names)
        for var in self.var_names:
            self.assertIn(f"idx__{var}", names)
            self.assertIn(f"param__{var}", names)
        for constr in self.constraint_names:
            self.assertIn(f"util__{constr}", names)
            self.assertIn(f"violation__{constr}", names)
            self.assertIn(f"penalty__{constr}", names)
        # Fixed string parameter uses string Arrow type.
        field = self.schema_info.schema.field("param__loads_category")
        self.assertEqual(field.type, pa.string())
        # Numeric parameter uses float64.
        field = self.schema_info.schema.field("param__geom_h_ges_mm")
        self.assertEqual(field.type, pa.float64())

    def test_schema_hash_is_deterministic(self) -> None:
        again = build_brute_force_schema(
            space=self.space,
            model=self.info["model"],
            constraint_names=self.constraint_names,
        )
        self.assertEqual(self.schema_info.schema_hash, again.schema_hash)
        self.assertEqual(
            self.schema_info.schema_hash,
            compute_schema_hash(self.schema_info.schema),
        )
        self.assertEqual(len(self.schema_info.schema_hash), 64)

    def test_success_and_error_rows_share_schema_fields(self) -> None:
        mid = [
            int(round(0.5 * (lo + hi))) for lo, hi in zip(self.lb_idx, self.ub_idx)
        ]
        success = analysis(
            self.decode(mid), self.constraints, self.materials
        )
        error_params = self.decode(list(self.lb_idx))
        error_params["mat_reinf_id"] = "does_not_exist"
        error = analysis(error_params, self.constraints, self.materials)

        success_row = flatten_evaluation_row(
            schema_info=self.schema_info,
            global_index=1,
            shard_index=0,
            problem_id="L6750_B2",
            evaluation_time_s=0.1,
            catalogue_indices=mid,
            result=success,
        )
        error_row = flatten_evaluation_row(
            schema_info=self.schema_info,
            global_index=0,
            shard_index=0,
            problem_id="L6750_B2",
            evaluation_time_s=0.2,
            catalogue_indices=list(self.lb_idx),
            result=error,
        )
        self.assertEqual(set(success_row.keys()), set(error_row.keys()))
        self.assertEqual(
            set(success_row.keys()),
            {field.name for field in self.schema_info.schema},
        )
        self.assertTrue(success_row["analysis_ok"])
        self.assertFalse(error_row["analysis_ok"])
        self.assertIsNone(success_row["analysis_error_json"])
        self.assertIsNotNone(error_row["analysis_error_json"])
        self.assertEqual(error_row["analysis_error_stage"], "reinforcement")
        # Missing constraint maps stay null, not zero.
        for name in self.constraint_names:
            self.assertIsNone(error_row[f"util__{name}"])
            self.assertIsNone(error_row[f"violation__{name}"])
            self.assertIsNone(error_row[f"penalty__{name}"])

    def test_rejects_unknown_scientific_keys(self) -> None:
        mid = list(self.lb_idx)
        result = analysis(self.decode(mid), self.constraints, self.materials)
        result = dict(result)
        result["unexpected_metric"] = 1.0
        with self.assertRaisesRegex(BruteForceSchemaError, "unexpected scientific"):
            flatten_evaluation_row(
                schema_info=self.schema_info,
                global_index=0,
                shard_index=0,
                problem_id="L6750_B2",
                evaluation_time_s=0.0,
                catalogue_indices=mid,
                result=result,
            )

    def test_rejects_inconsistent_feasible_flag(self) -> None:
        mid = list(self.lb_idx)
        result = dict(
            analysis(self.decode(mid), self.constraints, self.materials)
        )
        result["feasible"] = not bool(
            result["analysis_ok"] and result["constraint_feasible"]
        )
        with self.assertRaisesRegex(BruteForceSchemaError, "feasible must equal"):
            flatten_evaluation_row(
                schema_info=self.schema_info,
                global_index=0,
                shard_index=0,
                problem_id="L6750_B2",
                evaluation_time_s=0.0,
                catalogue_indices=mid,
                result=result,
            )

    def test_infinity_and_nan_roundtrip(self) -> None:
        mid = list(self.lb_idx)
        result = dict(
            analysis(self.decode(mid), self.constraints, self.materials)
        )
        result["y"] = float("inf")
        result["y_p"] = float("nan")
        result["analysis_ok"] = False
        result["constraint_feasible"] = False
        result["feasible"] = False
        row = flatten_evaluation_row(
            schema_info=self.schema_info,
            global_index=7,
            shard_index=1,
            problem_id="L6750_B2",
            evaluation_time_s=0.01,
            catalogue_indices=mid,
            result=result,
        )
        self.assertTrue(math.isinf(row["y"]))
        self.assertTrue(math.isnan(row["y_p"]))

        # Build a one-row table with only float columns exercised via full schema.
        arrays = []
        for field in self.schema_info.schema:
            value = row[field.name]
            arrays.append(pa.array([value], type=field.type))
        table = pa.Table.from_arrays(
            arrays, names=[field.name for field in self.schema_info.schema]
        )
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression="zstd")
        buffer.seek(0)
        loaded = pq.read_table(buffer)
        y_val = loaded.column("y")[0].as_py()
        yp_val = loaded.column("y_p")[0].as_py()
        self.assertTrue(math.isinf(y_val))
        self.assertTrue(math.isnan(yp_val))


if __name__ == "__main__":
    unittest.main()
