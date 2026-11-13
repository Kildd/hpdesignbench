"""
Phase B7 gate: local brute-force integration and parallel equivalence.
"""
from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow.parquet as pq

from core.execution_core.artifacts import read_task_status
from core.execution_core.atomic_io import read_json, write_json_atomic
from core.execution_core.lifecycle import (
    collect_experiment,
    dispatch_experiment,
    prepare_experiment,
    retry_failed_tasks,
)
from core.execution_core.models import TaskState
from core.ioh_core.workflows.brute_force.structural import scientific_values_equal


SCI_COMPARE_SKIP = frozenset({"evaluation_time_s"})


def _write_bf_config(
    path: Path,
    *,
    experiment_id: str,
    max_parallel_tasks: int,
    reuse_structural_artifacts: bool,
    rows_per_shard: int = 72,
    max_rows: int = 216,
) -> Path:
    """
    Write a truncated local brute-force TOML (3 shards × 72 rows by default).
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
            reuse_structural_artifacts = {"true" if reuse_structural_artifacts else "false"}
            acknowledge_large_run = true
            maximum_expected_rows = 1000000
            maximum_expected_bytes = 10000000000

            [execution]
            backend = "local"
            max_parallel_tasks = {int(max_parallel_tasks)}
            timeout_seconds = 3600
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        ),
        encoding="utf-8",
    )
    return path


def _run_bf_experiment(
    root: Path,
    *,
    experiment_id: str,
    max_parallel_tasks: int,
    reuse_structural_artifacts: bool,
) -> Path:
    """
    Prepare, dispatch, and collect one brute-force experiment.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = _write_bf_config(
        root / f"{experiment_id}.toml",
        experiment_id=experiment_id,
        max_parallel_tasks=max_parallel_tasks,
        reuse_structural_artifacts=reuse_structural_artifacts,
    )
    experiment_dir = root / "experiments" / experiment_id
    prepare_experiment(config_path=config_path, output=experiment_dir)
    dispatch_experiment(experiment_dir)
    collect_experiment(experiment_dir, require_complete=True)
    return experiment_dir


def _read_parts_sorted(experiment_dir: Path) -> dict[str, list[Any]]:
    """
    Load all dataset parts and return columns sorted by global_index.
    """
    parts_dir = Path(experiment_dir) / "dataset" / "parts"
    tables = []
    for path in sorted(parts_dir.glob("shard-*.parquet")):
        tables.append(pq.read_table(path))
    if not tables:
        raise AssertionError(f"no parquet parts under {parts_dir}")
    import pyarrow as pa

    table = pa.concat_tables(tables).sort_by("global_index")
    return {name: table.column(name).to_pylist() for name in table.column_names}


def _assert_scientific_columns_equal(
    left: dict[str, list[Any]],
    right: dict[str, list[Any]],
) -> None:
    """
    Compare deterministic scientific columns, ignoring evaluation timing.
    """
    left_keys = set(left) - SCI_COMPARE_SKIP
    right_keys = set(right) - SCI_COMPARE_SKIP
    if left_keys != right_keys:
        raise AssertionError(
            f"column mismatch: {sorted(left_keys ^ right_keys)}"
        )
    for key in sorted(left_keys):
        if not scientific_values_equal(left[key], right[key]):
            raise AssertionError(f"scientific column diverged: {key}")


class TestBruteForceLocalIntegration(unittest.TestCase):
    """Prepare/schema sealing and registration smoke."""

    def test_prepare_writes_schema_and_three_shards(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            config_path = _write_bf_config(
                root / "prep.toml",
                experiment_id="bf_prep",
                max_parallel_tasks=1,
                reuse_structural_artifacts=True,
            )
            experiment_dir = root / "experiments" / "bf_prep"
            paths = prepare_experiment(
                config_path=config_path, output=experiment_dir
            )
            schema_path = paths.root / "dataset" / "schema.json"
            self.assertTrue(schema_path.is_file())
            schema = read_json(schema_path)
            self.assertIn("schema_hash", schema)
            self.assertTrue((paths.root / "dataset" / "parts").is_dir())
            from core.execution_core.manifest import load_manifest

            manifest = load_manifest(paths.manifest)
            self.assertEqual(manifest.workflow_type, "brute_force")
            self.assertEqual(len(manifest.tasks), 3)
            self.assertEqual(manifest.worker_module, (
                "core.ioh_core.workflows.brute_force.worker"
            ))
            for task in manifest.tasks:
                self.assertEqual(
                    task.payload["schema_hash"], schema["schema_hash"]
                )
            self.assertEqual(manifest.tasks[0].payload["stop_index"], 72)
            self.assertEqual(manifest.tasks[2].payload["stop_index"], 216)


class TestBruteForceParallelEquivalence(unittest.TestCase):
    """Serial versus parallel scientific equivalence with reuse on/off."""

    def _compare_parallelism(
        self,
        *,
        reuse_structural_artifacts: bool,
        label: str,
    ) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            serial_dir = _run_bf_experiment(
                root / "serial",
                experiment_id=f"bf_serial_{label}",
                max_parallel_tasks=1,
                reuse_structural_artifacts=reuse_structural_artifacts,
            )
            parallel_dir = _run_bf_experiment(
                root / "parallel",
                experiment_id=f"bf_parallel_{label}",
                max_parallel_tasks=2,
                reuse_structural_artifacts=reuse_structural_artifacts,
            )
            left = _read_parts_sorted(serial_dir)
            right = _read_parts_sorted(parallel_dir)
            self.assertEqual(left["global_index"], list(range(216)))
            self.assertEqual(right["global_index"], list(range(216)))
            _assert_scientific_columns_equal(left, right)

            summary = read_json(parallel_dir / "aggregate" / "summary.json")
            self.assertFalse(summary["preview_only"])
            self.assertTrue(summary["validation_ok"])
            self.assertTrue(summary["success_marker_written"])
            self.assertTrue(
                (parallel_dir / "dataset" / "_SUCCESS").is_file()
            )
            self.assertEqual(summary["completed_task_count"], 3)
            self.assertEqual(summary["total_evaluations"], 216)
            self.assertEqual(summary["planned_rows"], 216)
            self.assertIsInstance(summary["best_finite_y_p"], float)
            self.assertIsInstance(summary["best_global_index"], int)
            self.assertIsInstance(summary["best_params"], dict)

    def test_serial_vs_parallel_with_reuse_enabled(self) -> None:
        self._compare_parallelism(
            reuse_structural_artifacts=True, label="reuse_on"
        )

    def test_serial_vs_parallel_with_reuse_disabled(self) -> None:
        self._compare_parallelism(
            reuse_structural_artifacts=False, label="reuse_off"
        )


class TestBruteForceFailureIsolation(unittest.TestCase):
    """Failed-shard-only retry and completed-part preservation."""

    def test_failed_shard_retry_preserves_completed_parts(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            config_path = _write_bf_config(
                root / "fail.toml",
                experiment_id="bf_fail",
                max_parallel_tasks=2,
                reuse_structural_artifacts=True,
            )
            experiment_dir = root / "experiments" / "bf_fail"
            prepare_experiment(config_path=config_path, output=experiment_dir)

            # Corrupt task 1 schema hash so the worker fails that shard only.
            manifest_path = experiment_dir / "manifest.json"
            manifest = read_json(manifest_path)
            original_hash = str(manifest["tasks"][1]["payload"]["schema_hash"])
            manifest["tasks"][1]["payload"]["schema_hash"] = "0" * 64
            write_json_atomic(manifest_path, manifest)

            dispatch_experiment(experiment_dir)

            statuses = {
                i: read_task_status(experiment_dir / "tasks" / f"task_{i:06d}")
                for i in range(3)
            }
            self.assertEqual(statuses[1]["state"], TaskState.FAILED.value)
            completed_before = {
                i
                for i, st in statuses.items()
                if st["state"] == TaskState.COMPLETED.value
            }
            self.assertIn(0, completed_before)
            self.assertIn(2, completed_before)
            part0 = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
            part2 = experiment_dir / "dataset" / "parts" / "shard-000002.parquet"
            part1 = experiment_dir / "dataset" / "parts" / "shard-000001.parquet"
            self.assertTrue(part0.is_file())
            self.assertTrue(part2.is_file())
            self.assertFalse(part1.is_file())
            stamp0 = part0.stat().st_mtime_ns
            stamp2 = part2.stat().st_mtime_ns

            # Restore a valid payload, then retry only the failed shard.
            manifest = read_json(manifest_path)
            manifest["tasks"][1]["payload"]["schema_hash"] = original_hash
            write_json_atomic(manifest_path, manifest)
            retried = retry_failed_tasks(experiment_dir)
            self.assertEqual(retried, [1])
            dispatch_experiment(experiment_dir, task_indices=[1])
            summary = collect_experiment(experiment_dir, require_complete=True)
            self.assertEqual(summary["completed_task_count"], 3)
            self.assertTrue(summary["success_marker_written"])
            self.assertTrue(
                (experiment_dir / "dataset" / "_SUCCESS").is_file()
            )
            self.assertTrue(part1.is_file())
            self.assertEqual(part0.stat().st_mtime_ns, stamp0)
            self.assertEqual(part2.stat().st_mtime_ns, stamp2)
            rows = _read_parts_sorted(experiment_dir)
            self.assertEqual(rows["global_index"], list(range(216)))


if __name__ == "__main__":
    unittest.main()
