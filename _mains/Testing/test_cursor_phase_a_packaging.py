"""
Phase A12 gate: packaging entry point, example configs, and local smoke.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tomllib
import unittest
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.backends import RecordingSlurmCommands, SlurmArrayBackend
from core.execution_core.config import load_experiment_config
from core.execution_core.lifecycle import list_task_statuses, prepare_experiment
from core.ioh_core.workflows.cli import build_parser, main as cli_main


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG = (
    REPO_ROOT / "_mains" / "studies" / "Execution" / "configs" / "examples" / "optimization_local.toml"
)
SLURM_CONFIG = (
    REPO_ROOT / "_mains" / "studies" / "Execution" / "configs" / "examples" / "optimization_slurm.toml"
)
PYPROJECT = REPO_ROOT / "pyproject.toml"


class TestPhaseAPackaging(unittest.TestCase):
    """Console script declaration, help, import-outside-CWD, configs."""

    def test_pyproject_declares_hdb_experiment_script(self) -> None:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        scripts = data.get("project", {}).get("scripts", {})
        self.assertEqual(
            scripts.get("hdb-experiment"),
            "core.ioh_core.workflows.cli:main",
        )

    def test_entry_point_resolves_after_editable_install(self) -> None:
        selected = metadata.entry_points().select(group="console_scripts")
        matches = [ep for ep in selected if ep.name == "hdb-experiment"]
        self.assertTrue(
            matches,
            "hdb-experiment missing from console_scripts; run pip install -e .",
        )
        loaded = matches[0].load()
        self.assertTrue(callable(loaded))

    def test_module_help_and_parser_prog(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.prog, "hdb-experiment")
        completed = subprocess.run(
            [sys.executable, "-m", "core.ioh_core.workflows.cli", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("prepare", completed.stdout)
        self.assertIn("dispatch", completed.stdout)
        self.assertIn("collect", completed.stdout)

    def test_import_and_help_outside_repository_cwd(self) -> None:
        with TemporaryDirectory() as tmp:
            foreign = Path(tmp) / "elsewhere"
            foreign.mkdir()
            code = (
                "import core.ioh_core.workflows.cli as cli; "
                "raise SystemExit(0 if cli.build_parser().prog == 'hdb-experiment' else 1)"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(foreign),
                capture_output=True,
                text=True,
                check=False,
                env={
                    **os.environ,
                    # Editable installs normally need no PYTHONPATH; keep repo root
                    # only if the package is not yet installed into this interpreter.
                    "PYTHONPATH": os.pathsep.join(
                        [str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]
                    ).rstrip(os.pathsep),
                },
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stderr or completed.stdout,
            )

    def test_example_configs_load(self) -> None:
        local = load_experiment_config(LOCAL_CONFIG)
        self.assertEqual(local.execution.backend, "local")
        self.assertEqual(local.experiment.algorithms[0].name, "RANDOM_SEARCH")

        slurm = load_experiment_config(SLURM_CONFIG)
        self.assertEqual(slurm.execution.backend, "slurm")
        slurm.execution.slurm.require_complete()
        self.assertEqual(slurm.experiment.repetitions.count, 2)


class TestPhaseALocalSmoke(unittest.TestCase):
    """One-task local smoke using the packaged example config."""

    def test_one_task_local_smoke_from_example_config(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "local_smoke"
            code = cli_main(
                [
                    "run",
                    "--config",
                    str(LOCAL_CONFIG),
                    "--output",
                    str(experiment_dir),
                ]
            )
            self.assertEqual(code, 0)
            status = list_task_statuses(experiment_dir)
            self.assertTrue(status.complete)
            self.assertEqual(status.counts["completed"], 1)
            self.assertTrue((experiment_dir / "aggregate" / "tasks.csv").is_file())
            summary_path = (
                experiment_dir / "tasks" / "task_000000" / "attempts" / "001" / "summary.json"
            )
            self.assertTrue(summary_path.is_file())


class TestPhaseASlurmExampleGeneration(unittest.TestCase):
    """Example Slurm config generates an array without a real cluster."""

    def test_slurm_example_prepares_and_generates_submission(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Rewrite output_root to a writable temp path while keeping Slurm fields.
            text = SLURM_CONFIG.read_text(encoding="utf-8")
            text = text.replace(
                'output_root = "/shared/experiments"',
                f'output_root = "{root.as_posix()}"',
            )
            cfg_path = root / "optimization_slurm.toml"
            cfg_path.write_text(text, encoding="utf-8")
            experiment_dir = root / "slurm_smoke"
            prepare_experiment(config_path=cfg_path, output=experiment_dir)
            config = load_experiment_config(cfg_path)
            commands = RecordingSlurmCommands(sbatch_job_ids=["12345"])
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
                task_indices=[0, 1],
                worker_module="core.ioh_core.workflows.optimization_runs.worker",
            )
            self.assertEqual(len(records), 1)
            submission = experiment_dir / "submissions"
            sub_dirs = sorted(p for p in submission.iterdir() if p.is_dir())
            self.assertEqual(len(sub_dirs), 1)
            self.assertTrue((sub_dirs[0] / "job.sh").is_file())
            self.assertTrue((sub_dirs[0] / "task-map.tsv").is_file())
            self.assertTrue((sub_dirs[0] / "submission.json").is_file())
            self.assertEqual(len(commands.sbatch_calls), 1)


if __name__ == "__main__":
    unittest.main()
