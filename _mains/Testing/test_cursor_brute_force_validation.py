"""
Phase B8 gate: brute-force dataset validation and fail-closed publication.
"""
from __future__ import annotations

import math
import shutil
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from core.execution_core.atomic_io import read_json, write_json_atomic
from core.execution_core.lifecycle import (
    collect_experiment,
    dispatch_experiment,
    prepare_experiment,
)
from core.ioh_core.workflows.brute_force.aggregation import (
    aggregate_brute_force_experiment,
)
from core.ioh_core.workflows.brute_force.validation import (
    BruteForceValidationError,
    validate_dataset,
    validate_dataset_or_raise,
)


def _write_bf_config(
    path: Path,
    *,
    experiment_id: str,
    rows_per_shard: int = 24,
    max_rows: int = 48,
) -> Path:
    """
    Write a tiny local brute-force TOML (two finish groups by default).
    """
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
            max_rows = {int(max_rows)}
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


def _run_tiny_bf(root: Path, *, experiment_id: str = "bf_validate") -> Path:
    """
    Prepare, dispatch, and collect a 48-row / 2-shard experiment.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = _write_bf_config(root / f"{experiment_id}.toml", experiment_id=experiment_id)
    experiment_dir = root / "experiments" / experiment_id
    prepare_experiment(config_path=config_path, output=experiment_dir)
    dispatch_experiment(experiment_dir)
    collect_experiment(experiment_dir, require_complete=True)
    return experiment_dir


def _clone_experiment(src: Path, dst: Path) -> Path:
    """
    Copy an experiment tree for mutation without re-running evaluations.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    success = dst / "dataset" / "_SUCCESS"
    if success.exists():
        success.unlink()
    return dst


def _expected_best(experiment_dir: Path) -> tuple[float, int]:
    """
    Independently compute the minimum finite y_p and its global_index.
    """
    best_y: float | None = None
    best_i: int | None = None
    for path in sorted((experiment_dir / "dataset" / "parts").glob("shard-*.parquet")):
        table = pq.read_table(path, columns=["global_index", "y_p"])
        for index, raw in zip(
            table.column("global_index").to_pylist(),
            table.column("y_p").to_pylist(),
        ):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if best_y is None or value < best_y:
                best_y = value
                best_i = int(index)
    if best_y is None or best_i is None:
        raise AssertionError("no finite y_p in dataset")
    return best_y, best_i


def _assert_no_success(experiment_dir: Path) -> None:
    path = Path(experiment_dir) / "dataset" / "_SUCCESS"
    if path.exists():
        raise AssertionError(f"_SUCCESS should be absent: {path}")


class TestBruteForceValidationAndAggregation(unittest.TestCase):
    """Valid publication and fail-closed invalid cases."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        cls.golden = _run_tiny_bf(cls.root / "golden", experiment_id="bf_golden")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def _mutated(self, name: str) -> Path:
        return _clone_experiment(self.golden, self.root / "mut" / name)

    def test_valid_dataset_publishes_success_and_global_best(self) -> None:
        summary = read_json(self.golden / "aggregate" / "summary.json")
        self.assertTrue((self.golden / "dataset" / "_SUCCESS").is_file())
        self.assertTrue(summary["success_marker_written"])
        self.assertTrue(summary["validation_ok"])
        self.assertFalse(summary["preview_only"])
        self.assertEqual(summary["total_evaluations"], 48)
        self.assertEqual(summary["planned_rows"], 48)
        expected_y, expected_i = _expected_best(self.golden)
        self.assertEqual(summary["best_global_index"], expected_i)
        self.assertTrue(
            math.isclose(float(summary["best_finite_y_p"]), expected_y, rel_tol=0.0, abs_tol=0.0)
        )
        self.assertIsInstance(summary["best_params"], dict)
        self.assertGreater(len(summary["best_params"]), 0)
        self.assertEqual(summary["analysis_ok_count"] + summary["analysis_error_count"], 48)
        result = validate_dataset(self.golden)
        self.assertTrue(result.ok)
        self.assertEqual(result.errors, ())

    def test_missing_shard_fails_without_success(self) -> None:
        experiment_dir = self._mutated("missing")
        part = experiment_dir / "dataset" / "parts" / "shard-000001.parquet"
        part.unlink()
        result = validate_dataset(experiment_dir)
        self.assertFalse(result.ok)
        self.assertTrue(any("missing part" in msg for msg in result.errors))
        with self.assertRaises(BruteForceValidationError):
            aggregate_brute_force_experiment(experiment_dir)
        _assert_no_success(experiment_dir)

    def test_extra_part_fails_without_success(self) -> None:
        experiment_dir = self._mutated("extra")
        parts = experiment_dir / "dataset" / "parts"
        shutil.copy2(parts / "shard-000000.parquet", parts / "shard-000099.parquet")
        with self.assertRaises(BruteForceValidationError):
            validate_dataset_or_raise(experiment_dir)
        with self.assertRaises(BruteForceValidationError):
            aggregate_brute_force_experiment(experiment_dir)
        _assert_no_success(experiment_dir)

    def test_wrong_row_count_fails_without_success(self) -> None:
        experiment_dir = self._mutated("wrong_count")
        part = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
        table = pq.read_table(part).slice(0, 12)
        pq.write_table(table, part)
        with self.assertRaises(BruteForceValidationError):
            aggregate_brute_force_experiment(experiment_dir)
        _assert_no_success(experiment_dir)

    def test_corrupt_footer_fails_without_success(self) -> None:
        experiment_dir = self._mutated("corrupt")
        part = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
        part.write_bytes(b"not a parquet file")
        with self.assertRaises(BruteForceValidationError):
            aggregate_brute_force_experiment(experiment_dir)
        _assert_no_success(experiment_dir)

    def test_schema_mismatch_fails_without_success(self) -> None:
        experiment_dir = self._mutated("schema")
        part = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
        bad = pa.table(
            {
                "global_index": pa.array(list(range(24)), type=pa.int64()),
                "noise": pa.array([0.0] * 24, type=pa.float64()),
            }
        )
        pq.write_table(bad, part)
        with self.assertRaises(BruteForceValidationError):
            aggregate_brute_force_experiment(experiment_dir)
        _assert_no_success(experiment_dir)

    def test_index_sequence_mismatch_fails_without_success(self) -> None:
        experiment_dir = self._mutated("indices")
        part = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
        table = pq.read_table(part)
        gi_idx = table.schema.get_field_index("global_index")
        gi_field = table.schema.field("global_index")
        values = table.column("global_index").to_pylist()
        values[0], values[1] = values[1], values[0]
        table = table.set_column(
            gi_idx,
            gi_field,
            pa.array(values, type=gi_field.type),
        )
        pq.write_table(table, part)
        with self.assertRaises(BruteForceValidationError):
            aggregate_brute_force_experiment(experiment_dir)
        _assert_no_success(experiment_dir)

    def test_overlapping_planned_ranges_fail_without_success(self) -> None:
        experiment_dir = self._mutated("overlap")
        manifest_path = experiment_dir / "manifest.json"
        manifest = read_json(manifest_path)
        # Keep part files as-is but claim overlapping planned ranges.
        manifest["tasks"][0]["payload"]["start_index"] = 0
        manifest["tasks"][0]["payload"]["stop_index"] = 24
        manifest["tasks"][1]["payload"]["start_index"] = 12
        manifest["tasks"][1]["payload"]["stop_index"] = 36
        write_json_atomic(manifest_path, manifest)

        # Rewrite parts so each individually matches the corrupt plan.
        parts = experiment_dir / "dataset" / "parts"
        for shard_index, start, stop in ((0, 0, 24), (1, 12, 36)):
            src = parts / f"shard-{shard_index:06d}.parquet"
            table = pq.read_table(src).slice(0, stop - start)
            n = stop - start
            gi_idx = table.schema.get_field_index("global_index")
            si_idx = table.schema.get_field_index("shard_index")
            gi_type = table.schema.field("global_index").type
            si_type = table.schema.field("shard_index").type
            table = table.set_column(
                gi_idx,
                table.schema.field("global_index"),
                pa.array(list(range(start, stop)), type=gi_type),
            )
            table = table.set_column(
                si_idx,
                table.schema.field("shard_index"),
                pa.array([shard_index] * n, type=si_type),
            )
            pq.write_table(table, src)

        result = validate_dataset(experiment_dir)
        self.assertFalse(result.ok)
        self.assertTrue(
            any("gap or overlap" in msg for msg in result.errors),
            msg=result.errors,
        )
        with self.assertRaises(BruteForceValidationError):
            aggregate_brute_force_experiment(experiment_dir)
        _assert_no_success(experiment_dir)

    def test_nonfinite_y_fails_without_success(self) -> None:
        experiment_dir = self._mutated("nonfinite_y")
        part = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
        table = pq.read_table(part)
        y_idx = table.schema.get_field_index("y")
        y_field = table.schema.field("y")
        values = table.column("y").to_pylist()
        values[0] = float("inf")
        table = table.set_column(
            y_idx,
            y_field,
            pa.array(values, type=y_field.type),
        )
        pq.write_table(table, part)
        result = validate_dataset(experiment_dir)
        self.assertFalse(result.ok)
        self.assertTrue(
            any("non-finite y" in msg for msg in result.errors),
            msg=result.errors,
        )
        with self.assertRaises(BruteForceValidationError):
            aggregate_brute_force_experiment(experiment_dir)
        _assert_no_success(experiment_dir)

    def test_partial_files_are_ignored(self) -> None:
        experiment_dir = self._mutated("partials")
        parts = experiment_dir / "dataset" / "parts"
        (parts / "shard-000000.parquet.partial").write_text("tmp", encoding="utf-8")
        (parts / "junk.txt").write_text("ignore", encoding="utf-8")
        result = validate_dataset(experiment_dir)
        self.assertTrue(result.ok)
        self.assertIn("shard-000000.parquet.partial", result.ignored_files)
        self.assertIn("junk.txt", result.ignored_files)
        summary = aggregate_brute_force_experiment(experiment_dir)
        self.assertTrue(summary["success_marker_written"])
        self.assertTrue((experiment_dir / "dataset" / "_SUCCESS").is_file())


if __name__ == "__main__":
    unittest.main()
