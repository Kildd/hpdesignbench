"""
Phase A3 gate: immutable manifest and environment artifacts.
"""
from __future__ import annotations

import textwrap
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.config import load_experiment_config
from core.execution_core.manifest import (
    ManifestError,
    build_environment_record,
    build_manifest,
    environment_from_dict,
    environment_to_dict,
    load_environment,
    load_manifest,
    manifest_from_dict,
    manifest_to_dict,
    prepare_experiment_artifacts,
    write_environment,
    write_manifest,
)
from core.execution_core.models import TaskRecord, format_task_id


def _write_valid_config(path: Path) -> Path:
    """
    Write a minimal valid local-backend TOML config.

    Parameters
    ----------
    path : Path
        Destination TOML path.

    Returns
    -------
    Path
        The same path after writing.
    """
    path.write_text(
        textwrap.dedent(
            """\
            schema_version = 1

            [experiment]
            id = "hp_slab_smoke"
            workflow = "optimization_runs"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [[experiment.algorithms]]
            name = "RANDOM_SEARCH"
            max_evaluations = 10

            [experiment.repetitions]
            count = 2
            seed_base = 42

            [execution]
            backend = "local"
            max_parallel_tasks = 2
            timeout_seconds = 2700
            cpus_per_task = 1

            [execution.thread_environment]
            OMP_NUM_THREADS = "1"

            [execution.local]
            python_executable = ""
            """
        ),
        encoding="utf-8",
    )
    return path


class TestManifestRoundTrip(unittest.TestCase):
    """JSON compatibility and reopen behaviour."""

    def test_manifest_json_round_trip(self) -> None:
        created = datetime(2026, 7, 28, 10, 0, tzinfo=timezone.utc)
        tasks = (
            TaskRecord(
                task_index=0,
                task_id=format_task_id(0),
                kind="optimization_run",
                payload={"problem_id": "L6750_B2", "repetition": 0},
            ),
        )
        manifest = build_manifest(
            experiment_id="hp_slab_smoke",
            workflow_type="optimization_runs",
            tasks=tasks,
            input_files={"hp_slab/problem_list.csv": "abc123"},
            created_at=created,
        )
        restored = manifest_from_dict(manifest_to_dict(manifest))
        self.assertEqual(restored, manifest)
        self.assertEqual(restored.created_at, created)
        self.assertEqual(
            restored.worker_module,
            "core.ioh_core.workflows.optimization_runs.worker",
        )

    def test_environment_json_round_trip(self) -> None:
        record = build_environment_record(
            python_executable=__import__("sys").executable,
            thread_environment={"OMP_NUM_THREADS": "1"},
        )
        restored = environment_from_dict(environment_to_dict(record))
        self.assertEqual(restored.schema_version, 1)
        self.assertEqual(restored.python_executable, record.python_executable)
        self.assertEqual(restored.thread_environment, {"OMP_NUM_THREADS": "1"})
        self.assertIn("numpy", restored.package_versions)
        self.assertIn("ioh", restored.package_versions)


class TestManifestPublish(unittest.TestCase):
    """Overwrite protection and reopen from disk."""

    def test_prepare_and_reopen(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_experiment_config(_write_valid_config(root / "smoke.toml"))
            experiment_root = root / "experiments" / config.experiment.id
            tasks = [
                TaskRecord(
                    task_index=0,
                    task_id=format_task_id(0),
                    kind="optimization_run",
                    payload={"problem_id": "L6750_B2"},
                )
            ]
            paths, manifest, environment = prepare_experiment_artifacts(
                config,
                root=experiment_root,
                tasks=tasks,
                input_files={"hp_slab/problem_list.csv": "deadbeef"},
            )

            self.assertTrue(paths.manifest.is_file())
            self.assertTrue(paths.environment.is_file())
            self.assertTrue(paths.source_config.is_file())
            self.assertEqual(load_manifest(paths.manifest), manifest)
            self.assertEqual(load_environment(paths.environment), environment)
            self.assertEqual(
                environment.thread_environment.get("OMP_NUM_THREADS"),
                "1",
            )

    def test_overwrite_protection(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiments" / "hp_slab_smoke"
            root.mkdir(parents=True)
            manifest = build_manifest(
                experiment_id="hp_slab_smoke",
                workflow_type="optimization_runs",
                tasks=(),
            )
            write_manifest(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ManifestError, "refusing to overwrite"):
                write_manifest(root / "manifest.json", manifest)

            environment = build_environment_record()
            write_environment(root / "environment.json", environment)
            with self.assertRaisesRegex(ManifestError, "refusing to overwrite"):
                write_environment(root / "environment.json", environment)

            config = load_experiment_config(_write_valid_config(Path(tmp) / "c.toml"))
            with self.assertRaisesRegex(ManifestError, "already prepared"):
                prepare_experiment_artifacts(config, root=root, tasks=())

    def test_invalid_task_indices_rejected(self) -> None:
        with self.assertRaisesRegex(ManifestError, "contiguous"):
            build_manifest(
                experiment_id="x",
                workflow_type="optimization_runs",
                tasks=[
                    TaskRecord(
                        task_index=1,
                        task_id=format_task_id(1),
                        kind="optimization_run",
                    )
                ],
            )

    def test_algorithm_parameters_round_trip_to_manifest_json(self) -> None:
        import json
        from unittest.mock import patch

        from core.ioh_core.algorithms.registry import (
            AlgorithmSpec,
            BudgetSemantics,
            CandidateMode,
            ParameterSpec,
            SeedPolicy,
            create_registry,
        )
        from core.ioh_core.workflows.optimization_runs.planner import (
            plan_optimization_tasks,
        )

        toy_registry = create_registry(
            (
                AlgorithmSpec(
                    name="TOY_ROUNDTRIP",
                    library="native",
                    constructor_ref=(
                        "core.ioh_core.algorithms.native.factory:build_random_search"
                    ),
                    parameters=(
                        ParameterSpec(
                            name="max_evaluations",
                            types=(int,),
                            required=True,
                            minimum=1,
                        ),
                        ParameterSpec(
                            name="mode",
                            types=(str,),
                            required=False,
                            default="fast",
                        ),
                    ),
                    seed_policy=SeedPolicy.REQUIRED,
                    dependency_extra=None,
                    candidate_mode=CandidateMode.INTEGER,
                    budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
                    description="toy algorithm for parameter round-trip",
                ),
            )
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "roundtrip.toml"
            config_path.write_text(
                textwrap.dedent(
                    """\
                    schema_version = 1

                    [experiment]
                    id = "param_roundtrip"
                    workflow = "optimization_runs"
                    slab_type = "hp_slab"
                    problem_ids = ["L6750_B2"]
                    output_root = "experiments"

                    [[experiment.algorithms]]
                    name = "TOY_ROUNDTRIP"
                    max_evaluations = 10
                    [experiment.algorithms.parameters]
                    mode = "careful"

                    [experiment.repetitions]
                    count = 1
                    seed_base = 42

                    [execution]
                    backend = "local"
                    max_parallel_tasks = 1
                    cpus_per_task = 1

                    [execution.local]
                    python_executable = ""
                    """
                ),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            experiment_root = root / "experiments" / config.experiment.id
            with patch(
                "core.ioh_core.algorithms.registry.ALGORITHM_REGISTRY",
                toy_registry,
            ):
                tasks = plan_optimization_tasks(config)
                paths, _manifest, _env = prepare_experiment_artifacts(
                    config,
                    root=experiment_root,
                    tasks=tasks,
                    input_files={"hp_slab/problem_list.csv": "deadbeef"},
                )

            raw = json.loads(paths.manifest.read_text(encoding="utf-8"))
            payload = raw["tasks"][0]["payload"]
            self.assertEqual(payload["algorithm_name"], "TOY_ROUNDTRIP")
            self.assertEqual(
                payload["algorithm_parameters"],
                {"max_evaluations": 10, "mode": "careful"},
            )
            self.assertIsInstance(payload["random_seed"], int)
            self.assertEqual(payload["termination_mode"], "fixed_budget")
            self.assertIsNone(payload["global_optimum"])
            self.assertIsNone(payload["allowed_deviation_percent"])
            self.assertIsNone(payload["target_reference"])
            self.assertIsNone(payload["target_threshold"])
            restored = manifest_from_dict(raw)
            self.assertEqual(
                restored.tasks[0].payload["algorithm_parameters"],
                {"max_evaluations": 10, "mode": "careful"},
            )
            self.assertEqual(
                restored.tasks[0].payload["termination_mode"],
                "fixed_budget",
            )


if __name__ == "__main__":
    unittest.main()
