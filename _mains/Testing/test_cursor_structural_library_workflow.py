"""
Phase L4 gate: structural_library workflow planning and tiny local seal.
"""
from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from core.execution_core.artifacts import reserve_attempt
from core.execution_core.config import load_experiment_config
from core.execution_core.lifecycle import collect_experiment, prepare_experiment
from core.ioh_core.structural_artifacts import open_structural_library
from core.ioh_core.workflows.registry import get_workflow, registered_workflows
from core.ioh_core.workflows.structural_library.planner import (
    STRUCTURAL_LIBRARY_TASK_KIND,
    plan_structural_id_shards,
)
from core.ioh_core.workflows.structural_library.worker import run_worker


def _write_config(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """\
            schema_version = 1

            [experiment]
            id = "sl_l4_gate"
            workflow = "structural_library"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [structural_library]
            rows_per_shard = 3
            max_ids = 6
            acknowledge_large_run = true
            maximum_expected_ids = 1000
            maximum_expected_bytes = 1000000000

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
    return path


class TestStructuralLibraryPlanning(unittest.TestCase):
    def test_registered_and_shards(self) -> None:
        self.assertIn("structural_library", registered_workflows())
        entry = get_workflow("structural_library")
        self.assertTrue(entry.worker_module.endswith("structural_library.worker"))
        specs = plan_structural_id_shards(
            id_start=0,
            id_stop=6,
            rows_per_shard=4,
            problem_id="L6750_B2",
            slab_type="hp_slab",
            snapshot_hash="x",
            run_config_fingerprint="y",
            library_structural_cardinality=6,
            full_structural_cardinality=100,
            full_candidate_cardinality=2400,
        )
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].id_stop, 4)
        self.assertEqual(specs[1].nrows, 2)
        self.assertEqual(specs[0].to_payload()["shard_index"], 0)

    def test_startup_banner_mentions_library_shards(self) -> None:
        from core.execution_core.lifecycle import format_experiment_startup_message

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root / "sl.toml")
            config = load_experiment_config(config_path)
            banner = format_experiment_startup_message(
                config,
                experiment_dir=root / "out",
                n_tasks=2,
            )
            self.assertIn("Starting sl_l4_gate", banner)
            self.assertIn("workflow    : structural_library", banner)
            self.assertIn("rows_per_shard=3", banner)
            self.assertIn("max_ids=6", banner)
            self.assertIn("tasks       : 2 shard(s)", banner)
            self.assertIn("publish     :", banner)
            self.assertNotIn("algorithms  :", banner)


class TestStructuralLibraryTinyBuild(unittest.TestCase):
    def test_prepare_worker_collect_seal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root / "sl.toml")
            experiment_dir = root / "experiments" / "sl_l4_gate"
            paths = prepare_experiment(
                config_path=config_path,
                output=experiment_dir,
            )
            config = load_experiment_config(paths.root / "source-config.toml")
            self.assertEqual(config.experiment.workflow, "structural_library")
            self.assertEqual(config.structural_library.max_ids, 6)

            from core.execution_core.manifest import load_manifest

            manifest = load_manifest(paths.manifest)
            self.assertEqual(manifest.workflow_type, "structural_library")
            self.assertEqual(len(manifest.tasks), 2)
            self.assertEqual(manifest.tasks[0].kind, STRUCTURAL_LIBRARY_TASK_KIND)

            for task in manifest.tasks:
                attempt_ref = reserve_attempt(paths.tasks / task.task_id)
                run_worker(
                    experiment_dir=paths.root,
                    task_index=task.task_index,
                    attempt=attempt_ref.attempt,
                )

            summary = collect_experiment(paths.root)
            from core.ioh_core.workflows.structural_library.validation import (
                library_directory,
            )

            lib_root = library_directory(paths.root)
            self.assertEqual(summary["nrows"], 6)
            self.assertEqual(
                lib_root,
                (root / "experiments" / "structural_library" / "baseline").resolve(),
            )
            self.assertTrue((lib_root / "_SUCCESS").is_file())
            with open_structural_library(lib_root) as lib:
                self.assertEqual(lib.manifest.nrows, 6)
                art = lib.get(0)
                self.assertEqual(art.snapshot_hash, lib.identity.snapshot_hash)

    def test_prepare_warns_when_consumer_library_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root / "sl.toml")
            lib_dir = root / "experiments" / "structural_library" / "baseline"
            lib_dir.mkdir(parents=True)
            (lib_dir / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertWarnsRegex(RuntimeWarning, "will be overwritten"):
                prepare_experiment(
                    config_path=config_path,
                    output=root / "experiments" / "sl_l4_gate",
                )


if __name__ == "__main__":
    unittest.main()
