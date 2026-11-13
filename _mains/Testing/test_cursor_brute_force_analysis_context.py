"""
Phase C3.1 gate: sealed brute-force analysis context opening.
"""
from __future__ import annotations

import dataclasses
import shutil
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from core.execution_core.config import ConfigError, parse_experiment_config
from core.execution_core.lifecycle import (
    collect_experiment,
    dispatch_experiment,
    prepare_experiment,
)
from core.ioh_core.workflows.brute_force.analysis import (
    BruteForceAnalysisError,
    DatasetReportOptions,
    open_brute_force_dataset,
)
from core.ioh_core.workflows.brute_force.planner import (
    load_index_space,
    load_problem_space_and_schema_from_csv_dir,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


def _write_bf_config(
    path: Path,
    *,
    experiment_id: str,
    rows_per_shard: int = 24,
    max_rows: int | None = 48,
    start_index: int = 0,
) -> Path:
    """
    Write a tiny local brute-force TOML.
    """
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
    experiment_id: str = "bf_analysis_ctx",
    max_rows: int | None = 48,
    start_index: int = 0,
) -> Path:
    """
    Prepare, dispatch, and collect a small sealed brute-force experiment.
    """
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


def _write_opt_config(path: Path, *, experiment_id: str) -> Path:
    path.write_text(
        textwrap.dedent(
            f"""\
            schema_version = 1

            [experiment]
            id = "{experiment_id}"
            workflow = "optimization_runs"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [[experiment.algorithms]]
            name = "RANDOM_SEARCH"
            max_evaluations = 2

            [experiment.repetitions]
            count = 1
            seed_base = 1

            [execution]
            backend = "local"
            max_parallel_tasks = 1
            timeout_seconds = 3600
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        ),
        encoding="utf-8",
    )
    return path


class TestDatasetReportOptions(unittest.TestCase):
    """Validate DatasetReportOptions defaults and guards."""

    def test_defaults(self) -> None:
        options = DatasetReportOptions()
        self.assertEqual(options.max_violation_examples, 20)
        self.assertFalse(options.include_pairwise_level_coverage)
        self.assertTrue(options.include_coviolation_matrix)
        self.assertEqual(options.duckdb_threads, 1)

    def test_rejects_non_positive_threads(self) -> None:
        with self.assertRaises(ValueError):
            DatasetReportOptions(duckdb_threads=0)


class TestBruteForceAnalysisContext(unittest.TestCase):
    """Sealed dataset context opening and coverage metadata."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.experiment_dir = _run_tiny_bf(cls.root, experiment_id="bf_ctx_ok")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_open_truncated_context_is_read_only(self) -> None:
        analysis_dir = self.experiment_dir / "aggregate" / "analysis"
        self.assertFalse(analysis_dir.exists())
        ctx = open_brute_force_dataset(self.experiment_dir)
        self.assertFalse(analysis_dir.exists())
        self.assertEqual(ctx.problem_id, "L6750_B2")
        self.assertEqual(ctx.slab_type, "hp_slab")
        self.assertEqual(ctx.planned_rows, 48)
        self.assertEqual(ctx.planned_start_index, 0)
        self.assertEqual(ctx.planned_stop_index, 48)
        self.assertFalse(ctx.is_full_enumeration)
        self.assertEqual(ctx.full_catalogue_rows, 53_374_464)
        self.assertAlmostEqual(
            ctx.coverage_fraction,
            48 / 53_374_464,
            places=18,
        )
        self.assertEqual(ctx.opt_penalty_alpha, 1.0)
        self.assertAlmostEqual(ctx.opt_penalty_beta, 1.0 / 3.0)
        self.assertEqual(ctx.run_config.invalid_utilization, 99.0)
        self.assertIsNotNone(ctx.collect_summary)
        self.assertTrue((self.experiment_dir / "dataset" / "_SUCCESS").is_file())
        self.assertIn("shard-*.parquet", ctx.parts_glob)

    def test_problem1_cardinality_from_snapshot_helper(self) -> None:
        space = load_index_space(slab_type="hp_slab", problem_id="L6750_B2")
        self.assertEqual(space.cardinality, 53_374_464)
        snapshot_dir = self.experiment_dir / "inputs" / "hp_slab"
        rebuilt_space, schema_info, info = load_problem_space_and_schema_from_csv_dir(
            slab_dir=snapshot_dir,
            problem_id="L6750_B2",
        )
        self.assertEqual(rebuilt_space.cardinality, 53_374_464)
        self.assertEqual(schema_info.var_names, space.var_names)
        self.assertIn("opt_penalty_alpha", info["model"])

    def test_missing_success_marker(self) -> None:
        clone = self.root / "bf_ctx_no_success"
        if clone.exists():
            shutil.rmtree(clone)
        shutil.copytree(self.experiment_dir, clone)
        (clone / "dataset" / "_SUCCESS").unlink()
        with self.assertRaises(BruteForceAnalysisError) as ctx:
            open_brute_force_dataset(clone)
        self.assertIn("_SUCCESS", str(ctx.exception))
        self.assertFalse((clone / "aggregate" / "analysis").exists())

    def test_missing_schema(self) -> None:
        clone = self.root / "bf_ctx_no_schema"
        if clone.exists():
            shutil.rmtree(clone)
        shutil.copytree(self.experiment_dir, clone)
        (clone / "dataset" / "schema.json").unlink()
        with self.assertRaises(BruteForceAnalysisError) as ctx:
            open_brute_force_dataset(clone)
        self.assertIn("schema.json", str(ctx.exception))

    def test_altered_snapshot_hash(self) -> None:
        clone = self.root / "bf_ctx_bad_hash"
        if clone.exists():
            shutil.rmtree(clone)
        shutil.copytree(self.experiment_dir, clone)
        csv_path = clone / "inputs" / "hp_slab" / "parameter_defaults.csv"
        csv_path.write_bytes(csv_path.read_bytes() + b"\n")
        with self.assertRaises(BruteForceAnalysisError) as ctx:
            open_brute_force_dataset(clone)
        self.assertIn("snapshot hash", str(ctx.exception).lower())

    def test_wrong_workflow(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_opt_config(
                root / "opt.toml", experiment_id="opt_wrong_wf"
            )
            experiment_dir = root / "experiments" / "opt_wrong_wf"
            prepare_experiment(config_path=config_path, output=experiment_dir)
            with self.assertRaises(BruteForceAnalysisError) as ctx:
                open_brute_force_dataset(experiment_dir)
            self.assertIn("brute_force", str(ctx.exception))

    def test_multiple_problem_ids_rejected_by_config(self) -> None:
        text = textwrap.dedent(
            """\
            schema_version = 1

            [experiment]
            id = "bf_multi"
            workflow = "brute_force"
            slab_type = "hp_slab"
            problem_ids = ["1", "2"]
            output_root = "experiments"

            [brute_force]
            rows_per_shard = 24
            max_rows = 48
            acknowledge_large_run = true
            maximum_expected_rows = 1000000
            maximum_expected_bytes = 10000000000

            [execution]
            backend = "local"
            max_parallel_tasks = 1
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        )
        with self.assertRaises(ConfigError) as ctx:
            parse_experiment_config(tomllib.loads(text))
        self.assertIn("exactly one problem_id", str(ctx.exception))

    def test_multiple_problem_ids_rejected_by_context_guard(self) -> None:
        ctx_ok = open_brute_force_dataset(self.experiment_dir)
        mutated_experiment = dataclasses.replace(
            ctx_ok.config.experiment,
            problem_ids=("L6750_B2", "L8100_B2"),
        )
        mutated_config = dataclasses.replace(
            ctx_ok.config,
            experiment=mutated_experiment,
        )
        with patch(
            "core.ioh_core.workflows.brute_force.analysis.context.load_prepared_config",
            return_value=mutated_config,
        ):
            with self.assertRaises(BruteForceAnalysisError) as ctx:
                open_brute_force_dataset(self.experiment_dir)
        self.assertIn("exactly one problem_id", str(ctx.exception))

    def test_start_index_truncated_window(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = _run_tiny_bf(
                Path(tmp),
                experiment_id="bf_ctx_window",
                max_rows=48,
                start_index=24,
            )
            ctx = open_brute_force_dataset(experiment_dir)
            self.assertEqual(ctx.planned_start_index, 24)
            self.assertEqual(ctx.planned_stop_index, 48)
            self.assertEqual(ctx.planned_rows, 24)
            self.assertFalse(ctx.is_full_enumeration)
            self.assertEqual(ctx.full_catalogue_rows, 53_374_464)


if __name__ == "__main__":
    unittest.main()
