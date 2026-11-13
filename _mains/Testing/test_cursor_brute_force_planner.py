"""
Phase B3 gate: brute-force shard planning coverage and alignment.
"""
from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from core.execution_core.config import (
    ConfigError,
    load_experiment_config,
    parse_experiment_config,
)
from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.brute_force.planner import (
    BruteForcePlanError,
    aligned_rows_per_shard,
    finish_group_size,
    load_index_space,
    plan_brute_force_tasks,
    plan_shard_specs,
    resolve_enumeration_bounds,
)


def _toy_space() -> IndexSpace:
    """
    Return a 2 x 3 x 4 space (finish group 12).
    """
    return IndexSpace.from_bounds(("a", "b", "c"), (0, 0, 0), (1, 2, 3))


def _parse_toml(text: str):
    """
    Parse TOML text into an ExperimentConfig.
    """
    return parse_experiment_config(tomllib.loads(text))


class TestBruteForcePlanner(unittest.TestCase):
    """Shard coverage, alignment, and config planning."""

    def test_exact_coverage_no_gaps_or_overlaps(self) -> None:
        space = _toy_space()
        group = finish_group_size(space)
        self.assertEqual(group, 12)
        specs = plan_shard_specs(
            space=space,
            problem_id="L6750_B2",
            slab_type="hp_slab",
            rows_per_shard=20,
        )
        self.assertEqual(aligned_rows_per_shard(20, group), 12)
        self.assertEqual([s.start_index for s in specs], [0, 12])
        self.assertEqual([s.stop_index for s in specs], [12, 24])
        covered: list[int] = []
        for spec in specs:
            covered.extend(range(spec.start_index, spec.stop_index))
            self.assertEqual(spec.start_index % group, 0)
            self.assertEqual(spec.stop_index % group, 0)
        self.assertEqual(covered, list(range(space.cardinality)))

    def test_final_shard_smaller_when_needed(self) -> None:
        # 5 x 2 x 4 = 40; finish group = 8; rows_per_shard=24 -> step=24
        space = IndexSpace.from_bounds(("a", "b", "c"), (0, 0, 0), (4, 1, 3))
        group = finish_group_size(space)
        self.assertEqual(group, 8)
        self.assertEqual(space.cardinality, 40)
        specs = plan_shard_specs(
            space=space,
            problem_id="L6750_B2",
            slab_type="hp_slab",
            rows_per_shard=24,
        )
        self.assertEqual(aligned_rows_per_shard(24, group), 24)
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].stop_index - specs[0].start_index, 24)
        self.assertEqual(specs[1].stop_index - specs[1].start_index, 16)
        self.assertEqual(specs[-1].stop_index, space.cardinality)

    def test_small_requested_rows_use_one_finish_group(self) -> None:
        space = _toy_space()
        group = finish_group_size(space)
        specs = plan_shard_specs(
            space=space,
            problem_id="L6750_B2",
            slab_type="hp_slab",
            rows_per_shard=1,
        )
        self.assertEqual(aligned_rows_per_shard(1, group), group)
        self.assertEqual(len(specs), space.cardinality // group)
        self.assertTrue(all(s.stop_index - s.start_index == group for s in specs))

    def test_problem_1_plan_from_config(self) -> None:
        text = textwrap.dedent(
            """\
            schema_version = 1

            [experiment]
            id = "bf_plan"
            workflow = "brute_force"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [brute_force]
            rows_per_shard = 10000
            acknowledge_large_run = true

            [execution]
            backend = "local"
            max_parallel_tasks = 2
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        )
        config = _parse_toml(text)
        space = load_index_space(slab_type="hp_slab", problem_id="L6750_B2")
        self.assertEqual(finish_group_size(space), 24)
        tasks = plan_brute_force_tasks(config)
        self.assertGreater(len(tasks), 1)
        step = aligned_rows_per_shard(10000, 24)
        self.assertEqual(step, 9984)
        first = tasks[0].payload
        self.assertEqual(first["start_index"], 0)
        self.assertEqual(first["stop_index"], step)
        self.assertEqual(tasks[-1].payload["stop_index"], space.cardinality)
        covered = 0
        previous_stop = 0
        for task in tasks:
            payload = task.payload
            self.assertEqual(payload["start_index"], previous_stop)
            self.assertEqual(payload["start_index"] % 24, 0)
            self.assertEqual(payload["stop_index"] % 24, 0)
            covered += payload["stop_index"] - payload["start_index"]
            previous_stop = payload["stop_index"]
        self.assertEqual(covered, space.cardinality)

    def test_config_rejects_algorithms_on_brute_force(self) -> None:
        text = textwrap.dedent(
            """\
            schema_version = 1

            [experiment]
            id = "bf_bad"
            workflow = "brute_force"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [[experiment.algorithms]]
            name = "RANDOM_SEARCH"
            max_evaluations = 2

            [brute_force]
            rows_per_shard = 24

            [execution]
            backend = "local"
            max_parallel_tasks = 1
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        )
        with self.assertRaisesRegex(ConfigError, "algorithms is not allowed"):
            _parse_toml(text)

    def test_start_index_mid_catalogue_window(self) -> None:
        space = IndexSpace.from_bounds(("a", "b", "c"), (0, 0, 0), (4, 1, 3))
        group = finish_group_size(space)
        self.assertEqual(group, 8)
        self.assertEqual(space.cardinality, 40)
        specs = plan_shard_specs(
            space=space,
            problem_id="L6750_B2",
            slab_type="hp_slab",
            rows_per_shard=24,
            start_index=16,
            stop_index=40,
        )
        self.assertEqual(specs[0].start_index, 16)
        self.assertEqual(specs[-1].stop_index, 40)
        covered = 0
        previous = 16
        for spec in specs:
            self.assertEqual(spec.start_index, previous)
            covered += spec.stop_index - spec.start_index
            previous = spec.stop_index
        self.assertEqual(covered, 24)

    def test_resolve_enumeration_bounds_rejects_unaligned_start(self) -> None:
        with self.assertRaisesRegex(BruteForcePlanError, "multiple of finish-group"):
            resolve_enumeration_bounds(
                start_index=1,
                max_rows=24,
                group_size=12,
                cardinality=24,
            )

    def test_resolve_enumeration_bounds_rejects_empty_window(self) -> None:
        with self.assertRaisesRegex(BruteForcePlanError, "empty enumeration"):
            resolve_enumeration_bounds(
                start_index=24,
                max_rows=24,
                group_size=12,
                cardinality=48,
            )

    def test_problem_1_second_half_from_config(self) -> None:
        half = 26_687_232
        text = textwrap.dedent(
            f"""\
            schema_version = 1

            [experiment]
            id = "bf_half2"
            workflow = "brute_force"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [brute_force]
            rows_per_shard = 10000
            start_index = {half}
            acknowledge_large_run = true

            [execution]
            backend = "local"
            max_parallel_tasks = 2
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        )
        config = _parse_toml(text)
        self.assertEqual(config.brute_force.start_index, half)
        space = load_index_space(slab_type="hp_slab", problem_id="L6750_B2")
        self.assertEqual(space.cardinality, 2 * half)
        tasks = plan_brute_force_tasks(config)
        self.assertEqual(tasks[0].payload["start_index"], half)
        self.assertEqual(tasks[-1].payload["stop_index"], space.cardinality)
        covered = sum(
            t.payload["stop_index"] - t.payload["start_index"] for t in tasks
        )
        self.assertEqual(covered, half)

    def test_load_from_temp_file(self) -> None:
        text = textwrap.dedent(
            """\
            schema_version = 1

            [experiment]
            id = "bf_file"
            workflow = "brute_force"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [brute_force]
            rows_per_shard = 48
            max_rows = 96
            acknowledge_large_run = true

            [execution]
            backend = "local"
            max_parallel_tasks = 1
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bf.toml"
            path.write_text(text, encoding="utf-8")
            config = load_experiment_config(path)
            self.assertEqual(config.experiment.workflow, "brute_force")
            self.assertIsNotNone(config.brute_force)
            tasks = plan_brute_force_tasks(config)
            self.assertTrue(tasks)


if __name__ == "__main__":
    unittest.main()
