"""
C7.2 gate: exact factorial sensitivity eligibility / coverage gate.
"""
from __future__ import annotations

import dataclasses
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.lifecycle import (
    collect_experiment,
    dispatch_experiment,
    prepare_experiment,
)
from core.ioh_core.workflows.brute_force.analysis import (
    DatasetReportOptions,
    open_brute_force_dataset,
    open_query_engine,
)
from core.ioh_core.workflows.brute_force.analysis.engine import quote_ident
from core.ioh_core.workflows.brute_force.analysis.factorial_sensitivity import (
    assess_factorial_sensitivity_eligibility,
)
from core.ioh_core.workflows.brute_force.models import IndexSpace


def _write_bf_config(
    path: Path,
    *,
    experiment_id: str,
    max_rows: int | None = 48,
    start_index: int = 0,
) -> Path:
    max_rows_line = (
        f"max_rows = {int(max_rows)}" if max_rows is not None else ""
    )
    path.write_text(
        textwrap.dedent(
            f"""\
            schema_version = 1

            [experiment]
            id = "{experiment_id}"
            workflow = "brute_force"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [brute_force]
            rows_per_shard = 24
            {max_rows_line}
            start_index = {int(start_index)}
            parquet_batch_rows = 24
            compression = "zstd"
            reuse_structural_artifacts = true
            acknowledge_large_run = true
            maximum_expected_rows = 1000000
            maximum_expected_bytes = 10000000000

            [execution]
            backend = "local"
            max_parallel_tasks = 2
            timeout_seconds = 3600
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        ),
        encoding="utf-8",
    )
    return path


def _run_tiny_bf(
    root: Path,
    *,
    experiment_id: str,
    max_rows: int | None = 48,
    start_index: int = 0,
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = _write_bf_config(
        root / f"{experiment_id}.toml",
        experiment_id=experiment_id,
        max_rows=max_rows,
        start_index=start_index,
    )
    experiment_dir = root / "experiments" / experiment_id
    prepare_experiment(config_path=config_path, output=experiment_dir)
    dispatch_experiment(experiment_dir)
    collect_experiment(experiment_dir, require_complete=True)
    return experiment_dir


def _toy_context(base_context):
    space = IndexSpace.from_bounds(("a", "b"), (0, 0), (1, 2))
    problem_info = {
        "model": {
            "a": {
                "role": "var",
                "values": [10, 20],
                "lb": 0,
                "ub": 1,
            },
            "b": {
                "role": "var",
                "values": ["x", "y", "z"],
                "lb": 0,
                "ub": 2,
            },
            "fixed_k": {
                "role": "fixed",
                "values": [42],
                "fixed_idx": 0,
                "lb": 0,
                "ub": 0,
            },
        }
    }
    return dataclasses.replace(
        base_context,
        index_space=space,
        var_names=("a", "b"),
        param_names=("a", "b", "fixed_k"),
        constraint_names=("c1",),
        problem_info=problem_info,
        fixed_params={"fixed_k": 42},
        full_catalogue_rows=6,
        planned_start_index=0,
        planned_stop_index=6,
        planned_rows=6,
        is_full_enumeration=True,
        coverage_fraction=1.0,
    )


def _install_toy_view(
    engine,
    *,
    rows: list[tuple] | None = None,
    omit_b_level: int | None = None,
) -> None:
    if rows is None:
        rows = []
        global_index = 0
        for a in (0, 1):
            for b in (0, 1, 2):
                if omit_b_level is not None and b == omit_b_level:
                    continue
                # y_p / util finite; analysis_ok true
                rows.append(
                    (
                        global_index,
                        a,
                        b,
                        42,
                        True,
                        float(a + 2 * b),
                        0.5 + 0.1 * a,
                    )
                )
                global_index += 1
    conn = engine.connection
    conn.execute(f"DROP VIEW IF EXISTS {quote_ident(engine.view_name)}")
    conn.execute("DROP TABLE IF EXISTS toy_factor_parts")
    conn.execute(
        "CREATE TEMP TABLE toy_factor_parts ("
        "global_index BIGINT, "
        '"idx__a" INTEGER, '
        '"idx__b" INTEGER, '
        '"param__fixed_k" INTEGER, '
        "analysis_ok BOOLEAN, "
        "y_p DOUBLE, "
        '"util__c1" DOUBLE'
        ")"
    )
    conn.executemany(
        "INSERT INTO toy_factor_parts VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.execute(
        f"CREATE VIEW {quote_ident(engine.view_name)} AS "
        "SELECT * FROM toy_factor_parts"
    )


class TestFactorialSensitivityEligibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        cls.golden = _run_tiny_bf(
            cls.root / "golden",
            experiment_id="bf_fs_elig_gold",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def _assess_toy(self, mutator=None, **install_kwargs):
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        with open_query_engine(context) as engine:
            _install_toy_view(engine, **install_kwargs)
            if mutator is not None:
                mutator(engine)
            return assess_factorial_sensitivity_eligibility(
                toy,
                engine,
                options=DatasetReportOptions(max_violation_examples=5),
                include_integrity=False,
            )

    def test_complete_toy_grid_passes_without_integrity(self) -> None:
        result = self._assess_toy()
        self.assertTrue(result.ok)
        self.assertEqual(result.reasons, ())
        self.assertEqual(result.cardinality, 6)
        self.assertEqual(result.variable_names, ("a", "b"))
        self.assertEqual(result.level_counts, (2, 3))
        self.assertEqual(
            [spec.name for spec in result.response_specs],
            ["y_p", "util__c1", "feasible__c1"],
        )

    def test_truncated_real_experiment_rejected(self) -> None:
        context = open_brute_force_dataset(self.golden)
        self.assertFalse(context.is_full_enumeration)
        with open_query_engine(context) as engine:
            result = assess_factorial_sensitivity_eligibility(
                context,
                engine,
                include_integrity=True,
            )
        self.assertFalse(result.ok)
        self.assertIn("not_full_enumeration", result.reasons)
        self.assertIn("row_count_mismatch", result.reasons)

    def test_missing_level_fails_coverage(self) -> None:
        result = self._assess_toy(omit_b_level=1)
        self.assertFalse(result.ok)
        self.assertIn("coverage_failed", result.reasons)
        self.assertIn("row_count_mismatch", result.reasons)

    def test_correct_marginals_but_incomplete_combinations(self) -> None:
        # 2x2 grid: each level appears twice (balanced marginals) but only two
        # distinct pairs exist instead of all four catalogue combinations.
        space = IndexSpace.from_bounds(("a", "b"), (0, 0), (1, 1))
        context = open_brute_force_dataset(self.golden)
        toy = dataclasses.replace(
            _toy_context(context),
            index_space=space,
            full_catalogue_rows=4,
            planned_stop_index=4,
            planned_rows=4,
            problem_info={
                "model": {
                    "a": {
                        "role": "var",
                        "values": [10, 20],
                        "lb": 0,
                        "ub": 1,
                    },
                    "b": {
                        "role": "var",
                        "values": ["x", "y"],
                        "lb": 0,
                        "ub": 1,
                    },
                    "fixed_k": {
                        "role": "fixed",
                        "values": [42],
                        "fixed_idx": 0,
                        "lb": 0,
                        "ub": 0,
                    },
                }
            },
        )
        rows_2x2 = [
            (0, 0, 0, 42, True, 1.0, 0.5),
            (1, 0, 0, 42, True, 2.0, 0.5),
            (2, 1, 1, 42, True, 3.0, 0.6),
            (3, 1, 1, 42, True, 4.0, 0.6),
        ]
        with open_query_engine(context) as engine:
            _install_toy_view(engine, rows=rows_2x2)
            result = assess_factorial_sensitivity_eligibility(
                toy,
                engine,
                include_integrity=False,
            )
        self.assertFalse(result.ok)
        self.assertIn("duplicate_or_incomplete_combinations", result.reasons)
        self.assertNotIn("coverage_failed", result.reasons)
        self.assertEqual(result.diagnostics["distinct_index_tuples"], 2)

    def test_analysis_not_ok_rows_fail(self) -> None:
        rows = []
        gi = 0
        for a in (0, 1):
            for b in (0, 1, 2):
                rows.append(
                    (
                        gi,
                        a,
                        b,
                        42,
                        gi != 2,  # one failed analysis
                        float(a + b),
                        0.5,
                    )
                )
                gi += 1
        result = self._assess_toy(rows=rows)
        self.assertFalse(result.ok)
        self.assertIn("analysis_not_ok_rows", result.reasons)
        self.assertEqual(
            result.diagnostics["response_quality"]["analysis_not_ok"]["count"],
            1,
        )

    def test_nonfinite_objective_fails(self) -> None:
        rows = []
        gi = 0
        for a in (0, 1):
            for b in (0, 1, 2):
                y_p = float("nan") if gi == 1 else float(a + b)
                rows.append((gi, a, b, 42, True, y_p, 0.5))
                gi += 1
        result = self._assess_toy(rows=rows)
        self.assertFalse(result.ok)
        self.assertIn("nonfinite_objective", result.reasons)

    def test_nonfinite_utilization_fails(self) -> None:
        rows = []
        gi = 0
        for a in (0, 1):
            for b in (0, 1, 2):
                util = None if gi == 4 else 0.5
                rows.append((gi, a, b, 42, True, float(a + b), util))
                gi += 1
        result = self._assess_toy(rows=rows)
        self.assertFalse(result.ok)
        self.assertIn("nonfinite_utilization", result.reasons)

    def test_missing_required_column_fails(self) -> None:
        def drop_util(engine) -> None:
            conn = engine.connection
            conn.execute(f"DROP VIEW IF EXISTS {quote_ident(engine.view_name)}")
            conn.execute(
                f"CREATE VIEW {quote_ident(engine.view_name)} AS "
                "SELECT global_index, \"idx__a\", \"idx__b\", "
                '"param__fixed_k", analysis_ok, y_p '
                "FROM toy_factor_parts"
            )

        result = self._assess_toy(mutator=drop_util)
        self.assertFalse(result.ok)
        self.assertIn("missing_response_columns", result.reasons)

    def test_global_index_gap_fails(self) -> None:
        rows = []
        gi = 0
        for a in (0, 1):
            for b in (0, 1, 2):
                # Skip index 3; duplicate index 0 at the end instead.
                index = 0 if gi == 5 else (gi if gi < 3 else gi + 1)
                rows.append((index, a, b, 42, True, float(a + b), 0.5))
                gi += 1
        result = self._assess_toy(rows=rows)
        self.assertFalse(result.ok)
        self.assertIn("global_index_incomplete", result.reasons)


if __name__ == "__main__":
    unittest.main()
