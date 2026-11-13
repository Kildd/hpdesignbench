"""
Phase B10 gate: brute-force example configs and prepare/packaging smoke.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.backends import RecordingSlurmCommands, SlurmArrayBackend
from core.execution_core.config import load_experiment_config
from core.execution_core.lifecycle import prepare_experiment
from core.execution_core.manifest import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
BF_LOCAL = (
    REPO_ROOT
    / "_mains"
    / "studies"
    / "Execution"
    / "configs"
    / "examples"
    / "brute_force_local.toml"
)
BF_SLURM = (
    REPO_ROOT
    / "_mains"
    / "studies"
    / "Execution"
    / "configs"
    / "examples"
    / "brute_force_slurm.toml"
)
BF_WORKER = "core.ioh_core.workflows.brute_force.worker"


class TestBruteForceExampleConfigs(unittest.TestCase):
    """Example TOMLs load and match documented smoke sizes."""

    def test_local_example_loads(self) -> None:
        config = load_experiment_config(BF_LOCAL)
        self.assertEqual(config.experiment.workflow, "brute_force")
        self.assertEqual(config.execution.backend, "local")
        self.assertIsNotNone(config.brute_force)
        self.assertEqual(config.brute_force.max_rows, 72)
        self.assertEqual(config.brute_force.rows_per_shard, 24)

    def test_slurm_example_loads(self) -> None:
        config = load_experiment_config(BF_SLURM)
        self.assertEqual(config.experiment.workflow, "brute_force")
        self.assertEqual(config.execution.backend, "slurm")
        self.assertIsNotNone(config.execution.slurm)
        config.execution.slurm.require_complete()
        self.assertEqual(config.brute_force.max_rows, 288)
        self.assertEqual(config.execution.slurm.array_concurrency, 4)


class TestBruteForceLocalExamplePrepare(unittest.TestCase):
    """Prepare the packaged local example (no full enumeration required)."""

    def test_prepare_local_example(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            experiment_dir = Path(tmp) / "bf_local_smoke"
            paths = prepare_experiment(
                config_path=BF_LOCAL,
                output=experiment_dir,
            )
            manifest = load_manifest(paths.manifest)
            self.assertEqual(manifest.workflow_type, "brute_force")
            self.assertEqual(manifest.worker_module, BF_WORKER)
            self.assertEqual(len(manifest.tasks), 3)
            self.assertTrue((paths.root / "dataset" / "schema.json").is_file())
            self.assertTrue((paths.root / "dataset" / "parts").is_dir())
            self.assertFalse((paths.root / "dataset" / "_SUCCESS").exists())


class TestBruteForceSlurmExampleGeneration(unittest.TestCase):
    """Example Slurm BF config prepares and generates a mocked array."""

    def test_slurm_example_prepares_and_generates_submission(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            text = BF_SLURM.read_text(encoding="utf-8")
            text = text.replace(
                'output_root = "/shared/experiments"',
                f'output_root = "{root.as_posix()}"',
            )
            cfg_path = root / "brute_force_slurm.toml"
            cfg_path.write_text(text, encoding="utf-8")
            experiment_dir = root / "bf_slurm_smoke"
            prepare_experiment(config_path=cfg_path, output=experiment_dir)
            manifest = load_manifest(experiment_dir / "manifest.json")
            self.assertEqual(len(manifest.tasks), 12)
            self.assertEqual(manifest.worker_module, BF_WORKER)

            config = load_experiment_config(cfg_path)
            commands = RecordingSlurmCommands(sbatch_job_ids=["424242"])
            backend = SlurmArrayBackend(
                python_executable=config.execution.slurm.python_executable,
                partition=config.execution.slurm.partition,
                account=config.execution.slurm.account,
                memory=config.execution.slurm.memory,
                time_limit=config.execution.slurm.time_limit,
                cpus_per_task=config.execution.cpus_per_task,
                array_concurrency=config.execution.slurm.array_concurrency,
                maximum_array_size=config.execution.slurm.maximum_array_size,
                thread_environment=config.execution.thread_environment,
                commands=commands,
            )
            records = backend.dispatch(
                experiment_dir=experiment_dir,
                task_indices=list(range(12)),
                worker_module=manifest.worker_module,
            )
            self.assertEqual(len(records), 1)
            sub_dirs = sorted(
                p for p in (experiment_dir / "submissions").iterdir() if p.is_dir()
            )
            self.assertEqual(len(sub_dirs), 1)
            job = (sub_dirs[0] / "job.sh").read_text(encoding="utf-8")
            self.assertIn(BF_WORKER, job)
            self.assertTrue((sub_dirs[0] / "task-map.tsv").is_file())
            self.assertEqual(len(commands.sbatch_calls), 1)


if __name__ == "__main__":
    unittest.main()
