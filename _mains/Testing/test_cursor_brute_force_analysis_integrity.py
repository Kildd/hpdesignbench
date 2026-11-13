"""
Phase C3.3 gate: structural and scientific integrity analysis.
"""
from __future__ import annotations

import shutil
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from core.execution_core.lifecycle import (
    collect_experiment,
    dispatch_experiment,
    prepare_experiment,
)
from core.ioh_core.workflows.brute_force.analysis import (
    DatasetReportOptions,
    open_brute_force_dataset,
    open_query_engine,
    run_integrity_analysis,
)
from core.ioh_core.workflows.brute_force.validation import validate_dataset


def _write_bf_config(
    path: Path,
    *,
    experiment_id: str,
    rows_per_shard: int = 24,
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
            rows_per_shard = {int(rows_per_shard)}
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


def _clone_experiment(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _rewrite_first_part(experiment_dir: Path, mutator) -> None:
    part = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
    table = pq.read_table(part)
    mutated = mutator(table)
    pq.write_table(mutated, part)


def _ensure_success(experiment_dir: Path) -> None:
    marker = experiment_dir / "dataset" / "_SUCCESS"
    marker.write_text("ok\n", encoding="utf-8")


class TestBruteForceIntegrityAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        cls.golden = _run_tiny_bf(cls.root / "golden", experiment_id="bf_int_gold")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def _run(self, experiment_dir: Path, *, max_examples: int = 5):
        context = open_brute_force_dataset(experiment_dir)
        with open_query_engine(context) as engine:
            return run_integrity_analysis(
                context,
                engine,
                options=DatasetReportOptions(max_violation_examples=max_examples),
            )

    def test_valid_truncated_dataset_passes(self) -> None:
        report = self._run(self.golden)
        self.assertTrue(report["structural"]["ok"])
        self.assertTrue(report["scientific_ran"])
        self.assertTrue(report["scientific_ok"])
        self.assertTrue(report["ok"])
        self.assertEqual(report["overall_status"], "success")
        self.assertEqual(report["dataset_identity"]["planned_rows"], 48)
        self.assertEqual(report["dataset_identity"]["full_catalogue_rows"], 53374464)
        self.assertTrue(report["dataset_identity"]["duckdb_matches_b8"])
        self.assertEqual(
            report["uniqueness"]["direct_tuple_distinct_scan"], "skipped"
        )
        self.assertTrue(report["rules"]["candidate_decoder"]["ok"])
        self.assertTrue(report["rules"]["y_p_identity"]["ok"])
        self.assertIn("y", report["numeric_quality"])
        self.assertIn("nan", report["numeric_quality"]["y"])

    def test_truncated_nonzero_start_still_valid(self) -> None:
        # max_rows is an exclusive absolute stop index, not a row count.
        experiment_dir = _run_tiny_bf(
            self.root / "trunc",
            experiment_id="bf_int_trunc",
            max_rows=48,
            start_index=24,
        )
        report = self._run(experiment_dir)
        self.assertTrue(report["ok"])
        self.assertEqual(report["dataset_identity"]["planned_start_index"], 24)
        self.assertEqual(report["dataset_identity"]["planned_stop_index"], 48)
        self.assertFalse(report["dataset_identity"]["is_full_enumeration"])

    def test_structural_failure_skips_scientific(self) -> None:
        experiment_dir = _clone_experiment(
            self.golden, self.root / "mut" / "struct"
        )
        part = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
        part.write_bytes(b"not parquet")
        _ensure_success(experiment_dir)
        structural = validate_dataset(experiment_dir)
        self.assertFalse(structural.ok)
        context = open_brute_force_dataset(experiment_dir)
        # No DuckDB engine: scientific scans must not run on untrusted data.
        report = run_integrity_analysis(
            context, engine=None, structural=structural
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["overall_status"], "structural_failure")
        self.assertFalse(report["scientific_ran"])
        self.assertTrue(report["scientific"]["skipped"])
        self.assertEqual(report["rules"], {})

    def test_decoder_mismatch_fails_with_bounded_examples(self) -> None:
        experiment_dir = _clone_experiment(
            self.golden, self.root / "mut" / "decoder"
        )
        var_name = open_brute_force_dataset(self.golden).var_names[0]
        col = f"idx__{var_name}"

        def mutate(table: pa.Table) -> pa.Table:
            arrays = []
            for field in table.schema:
                if field.name != col:
                    arrays.append(table.column(field.name))
                    continue
                values = table.column(field.name).to_pylist()
                values[0] = int(values[0]) + 1
                arrays.append(pa.array(values, type=field.type))
            return pa.Table.from_arrays(arrays, schema=table.schema)

        _rewrite_first_part(experiment_dir, mutate)
        _ensure_success(experiment_dir)
        report = self._run(experiment_dir, max_examples=3)
        self.assertFalse(report["ok"])
        self.assertEqual(report["overall_status"], "scientific_integrity_failure")
        rule = report["rules"]["candidate_decoder"]
        self.assertFalse(rule["ok"])
        self.assertGreaterEqual(rule["count"], 1)
        self.assertLessEqual(len(rule["examples"]), 3)
        self.assertEqual(rule["examples"], sorted(rule["examples"]))
        self.assertEqual(
            report["uniqueness"]["direct_tuple_distinct_scan"], "not_run"
        )

    def test_feasible_boolean_inconsistency(self) -> None:
        experiment_dir = _clone_experiment(
            self.golden, self.root / "mut" / "feasible"
        )

        def mutate(table: pa.Table) -> pa.Table:
            arrays = []
            for field in table.schema:
                if field.name != "feasible":
                    arrays.append(table.column(field.name))
                    continue
                values = table.column(field.name).to_pylist()
                values[0] = not bool(values[0])
                arrays.append(pa.array(values, type=field.type))
            return pa.Table.from_arrays(arrays, schema=table.schema)

        _rewrite_first_part(experiment_dir, mutate)
        _ensure_success(experiment_dir)
        report = self._run(experiment_dir)
        rule = report["rules"]["feasible_composite"]
        self.assertFalse(rule["ok"])
        self.assertGreaterEqual(rule["count"], 1)
        self.assertEqual(rule["examples"][0], 0)

    def test_y_p_identity_mismatch(self) -> None:
        experiment_dir = _clone_experiment(
            self.golden, self.root / "mut" / "yp"
        )

        def mutate(table: pa.Table) -> pa.Table:
            arrays = []
            for field in table.schema:
                if field.name != "y_p":
                    arrays.append(table.column(field.name))
                    continue
                values = table.column(field.name).to_pylist()
                values[0] = float(values[0]) + 10.0
                arrays.append(pa.array(values, type=field.type))
            return pa.Table.from_arrays(arrays, schema=table.schema)

        _rewrite_first_part(experiment_dir, mutate)
        _ensure_success(experiment_dir)
        report = self._run(experiment_dir)
        rule = report["rules"]["y_p_identity"]
        self.assertFalse(rule["ok"])
        self.assertGreaterEqual(rule["count"], 1)

    def test_nonfinite_classification_counts(self) -> None:
        experiment_dir = _clone_experiment(
            self.golden, self.root / "mut" / "nonfinite"
        )

        def mutate(table: pa.Table) -> pa.Table:
            arrays = []
            for field in table.schema:
                if field.name != "y":
                    arrays.append(table.column(field.name))
                    continue
                values = table.column(field.name).to_pylist()
                values[0] = float("nan")
                values[1] = float("inf")
                values[2] = float("-inf")
                arrays.append(pa.array(values, type=field.type))
            return pa.Table.from_arrays(arrays, schema=table.schema)

        _rewrite_first_part(experiment_dir, mutate)
        _ensure_success(experiment_dir)
        report = self._run(experiment_dir)
        self.assertFalse(report["ok"])
        self.assertFalse(report["structural"]["ok"])
        self.assertFalse(report["scientific_ran"])
        self.assertEqual(report["numeric_quality"], {})
        joined = " ".join(report["structural"]["errors"])
        self.assertIn("non-finite y", joined)


if __name__ == "__main__":
    unittest.main()
