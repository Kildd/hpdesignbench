"""
C1.3 gate: sensitivity block-aligned planning and preflight.
"""
from __future__ import annotations

import textwrap
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from core.execution_core.config import parse_experiment_config
from core.ioh_core.workflows.sensitivity_analysis.planner import (
    SENSITIVITY_TASK_KIND,
    SensitivityPreflightError,
    plan_sensitivity_shard_specs,
    plan_sensitivity_tasks,
    run_sensitivity_preflight,
    specs_to_task_records,
)


def _config_text(
    *,
    sample_size: int = 16,
    blocks_per_shard: int = 5,
    acknowledge: str = "false",
    max_evals: int = 100000,
    max_bytes: int = 2_000_000_000,
) -> str:
    return textwrap.dedent(
        f"""\
        schema_version = 1

        [experiment]
        id = "sensitivity_plan"
        workflow = "sensitivity_analysis"
        slab_type = "hp_slab"
        problem_ids = ["L6750_B2"]
        output_root = "experiments"

        [sensitivity_analysis]
        sample_size = {sample_size}
        factor_names = ["geom_h_ges_mm", "geom_t_mm", "mat_conc_id"]
        response_key = "y"
        scramble = true
        seed = 1
        blocks_per_shard = {blocks_per_shard}
        parquet_batch_rows = 32
        compression = "zstd"
        bootstrap_resamples = 10
        bootstrap_confidence_level = 0.95
        bootstrap_seed = 1
        compute_bootstrap_ci = false
        acknowledge_large_run = {acknowledge}
        maximum_expected_evaluations = {max_evals}
        maximum_expected_bytes = {max_bytes}

        [execution]
        backend = "local"
        max_parallel_tasks = 1
        cpus_per_task = 1

        [execution.local]
        python_executable = ""

        [execution.slurm]
        python_executable = ""
        partition = ""
        account = ""
        memory = "8G"
        time_limit = "01:00:00"
        array_concurrency = 4
        maximum_array_size = 1000
        """
    )


class TestSensitivityPlanner(unittest.TestCase):
    """Block shard planning and preflight gates."""

    def test_contiguous_blocks_last_shard_smaller(self) -> None:
        specs = plan_sensitivity_shard_specs(
            sample_size=16,
            factor_count=3,
            blocks_per_shard=5,
            problem_id="L6750_B2",
            slab_type="hp_slab",
            schema_hash="abc",
        )
        self.assertEqual(len(specs), 4)
        self.assertEqual(
            [(s.block_start, s.block_stop) for s in specs],
            [(0, 5), (5, 10), (10, 15), (15, 16)],
        )
        self.assertEqual(specs[-1].block_count, 1)
        self.assertEqual(specs[0].rows_per_block, 5)
        self.assertEqual(specs[0].row_count, 5 * 5)
        self.assertEqual(specs[-1].row_count, 1 * 5)

        covered = []
        for spec in specs:
            covered.extend(range(spec.block_start, spec.block_stop))
        self.assertEqual(covered, list(range(16)))

    def test_task_records_are_contiguous(self) -> None:
        specs = plan_sensitivity_shard_specs(
            sample_size=8,
            factor_count=2,
            blocks_per_shard=3,
            problem_id="L6750_B2",
            slab_type="hp_slab",
            schema_hash="hash",
        )
        tasks = specs_to_task_records(specs)
        self.assertEqual([t.task_index for t in tasks], list(range(len(specs))))
        self.assertEqual(tasks[0].kind, SENSITIVITY_TASK_KIND)
        self.assertEqual(tasks[0].payload["block_start"], 0)
        self.assertEqual(tasks[0].payload["rows_per_block"], 4)

    def test_plan_from_config(self) -> None:
        config = parse_experiment_config(tomllib.loads(_config_text()))
        tasks = plan_sensitivity_tasks(config)
        self.assertEqual(len(tasks), 4)
        self.assertTrue(tasks[0].payload["schema_hash"])
        self.assertEqual(tasks[0].payload["factor_count"], 3)

    def test_preflight_blocks_without_ack(self) -> None:
        config = parse_experiment_config(
            tomllib.loads(
                _config_text(
                    sample_size=1024,
                    max_evals=100,
                    acknowledge="false",
                )
            )
        )
        assert config.sensitivity_analysis is not None
        with self.assertRaises(SensitivityPreflightError):
            run_sensitivity_preflight(
                section=config.sensitivity_analysis,
                factor_count=3,
                var_count=9,
            )

    def test_preflight_allows_with_ack(self) -> None:
        config = parse_experiment_config(
            tomllib.loads(
                _config_text(
                    sample_size=1024,
                    max_evals=100,
                    acknowledge="true",
                )
            )
        )
        assert config.sensitivity_analysis is not None
        report = run_sensitivity_preflight(
            section=config.sensitivity_analysis,
            factor_count=3,
            var_count=9,
        )
        self.assertFalse(report.blocked)
        self.assertTrue(report.warnings)
        self.assertEqual(report.expected_evaluations, 1024 * 5)


if __name__ == "__main__":
    unittest.main()
