"""
Phase A7 gate: bounded local subprocess backend behaviour.
"""
from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from core.execution_core.artifacts import (
    create_experiment_layout,
    create_task_layout,
    read_task_status,
)
from core.execution_core.atomic_io import read_json
from core.execution_core.backends.factory import create_execution_backend
from core.execution_core.backends.local import LocalBackendError, LocalSubprocessBackend
from core.execution_core.config import load_experiment_config
from core.execution_core.manifest import prepare_experiment_artifacts
from core.execution_core.models import TaskState
from core.ioh_core.workflows.optimization_runs.planner import plan_optimization_tasks
from core.ioh_core.workflows.optimization_runs.snapshot import snapshot_slab_inputs


FAKE_WORKER = "_mains.Testing.helpers.fake_worker"


def _prepare_n_tasks(root: Path, n: int) -> Path:
    """
    Create an experiment with ``n`` pending empty task directories.

    Parameters
    ----------
    root : Path
        Temporary parent directory.
    n : int
        Number of task directories to create.

    Returns
    -------
    Path
        Experiment root.
    """
    experiment_dir = root / "experiments" / "local_backend"
    paths = create_experiment_layout(experiment_dir)
    # Minimal sealed manifest so paths exist; tasks are created explicitly.
    write_toml = root / "cfg.toml"
    write_toml.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "[experiment]",
                'id = "local_backend"',
                'workflow = "optimization_runs"',
                'slab_type = "hp_slab"',
                'problem_ids = ["L6750_B2"]',
                'output_root = "experiments"',
                "[[experiment.algorithms]]",
                'name = "RANDOM_SEARCH"',
                "max_evaluations = 1",
                "[experiment.repetitions]",
                "count = 1",
                "seed_base = 1",
                "[execution]",
                'backend = "local"',
                "max_parallel_tasks = 1",
                "cpus_per_task = 1",
                "[execution.local]",
                'python_executable = ""',
            ]
        ),
        encoding="utf-8",
    )
    config = load_experiment_config(write_toml)
    prepare_experiment_artifacts(config, root=experiment_dir, tasks=())
    for index in range(n):
        create_task_layout(paths, index)
    return experiment_dir


class TestLocalSubprocessBackend(unittest.TestCase):
    """Concurrency, isolation, timeout, env, and failure mapping."""

    def test_strict_concurrency_and_slot_reuse(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_n_tasks(root, 3)
            backend = LocalSubprocessBackend(
                max_parallel_tasks=1,
                thread_environment={
                    "HDB_FAKE_WORKER_MODE": "sleep",
                    "HDB_FAKE_WORKER_SLEEP": "0.35",
                },
            )
            started = time.perf_counter()
            backend.dispatch(
                experiment_dir=experiment_dir,
                task_indices=[0, 1, 2],
                worker_module=FAKE_WORKER,
            )
            elapsed = time.perf_counter() - started
            self.assertGreaterEqual(elapsed, 0.9)
            for index in range(3):
                status = read_task_status(
                    experiment_dir / "tasks" / f"task_{index:06d}"
                )
                self.assertEqual(status["state"], TaskState.COMPLETED.value)

            # With three slots, the same sleeps finish much faster.
            experiment_dir2 = _prepare_n_tasks(root / "par", 3)
            backend2 = LocalSubprocessBackend(
                max_parallel_tasks=3,
                thread_environment={
                    "HDB_FAKE_WORKER_MODE": "sleep",
                    "HDB_FAKE_WORKER_SLEEP": "0.35",
                },
            )
            started = time.perf_counter()
            backend2.dispatch(
                experiment_dir=experiment_dir2,
                task_indices=[0, 1, 2],
                worker_module=FAKE_WORKER,
            )
            elapsed_par = time.perf_counter() - started
            # Parallel must clearly beat serial. Avoid absolute wall-clock ceilings:
            # Windows Python subprocess spawn can exceed one sleep interval alone.
            self.assertLess(elapsed_par, elapsed)
            self.assertLess(elapsed_par, 0.7 * elapsed)

    def test_failure_isolation_and_task_logs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_n_tasks(root, 2)
            # First task fails, second succeeds — mode is fixed per backend env,
            # so run them as separate dispatches with different envs.
            fail_backend = LocalSubprocessBackend(
                max_parallel_tasks=1,
                thread_environment={"HDB_FAKE_WORKER_MODE": "failure"},
            )
            ok_backend = LocalSubprocessBackend(
                max_parallel_tasks=1,
                thread_environment={"HDB_FAKE_WORKER_MODE": "success"},
            )
            records = fail_backend.dispatch(
                experiment_dir=experiment_dir,
                task_indices=[0],
                worker_module=FAKE_WORKER,
            )
            ok_backend.dispatch(
                experiment_dir=experiment_dir,
                task_indices=[1],
                worker_module=FAKE_WORKER,
            )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                TaskState.FAILED.value,
            )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000001")["state"],
                TaskState.COMPLETED.value,
            )
            attempt0 = experiment_dir / "tasks" / "task_000000" / "attempts" / "001"
            self.assertTrue((attempt0 / "stdout.log").is_file())
            self.assertTrue((attempt0 / "stderr.log").is_file())
            self.assertTrue((attempt0 / "failure.json").is_file())
            self.assertEqual(
                records[0].metadata["outcomes"][0]["failure_kind"],
                "nonzero_exit",
            )

    def test_timeout_sets_timed_out_after_death(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = _prepare_n_tasks(Path(tmp), 1)
            backend = LocalSubprocessBackend(
                max_parallel_tasks=1,
                timeout_seconds=0.2,
                termination_grace_seconds=0.2,
                thread_environment={
                    "HDB_FAKE_WORKER_MODE": "sleep",
                    "HDB_FAKE_WORKER_SLEEP": "5",
                },
            )
            records = backend.dispatch(
                experiment_dir=experiment_dir,
                task_indices=[0],
                worker_module=FAKE_WORKER,
            )
            status = read_task_status(experiment_dir / "tasks" / "task_000000")
            self.assertEqual(status["state"], TaskState.TIMED_OUT.value)
            self.assertEqual(
                records[0].metadata["outcomes"][0]["failure_kind"],
                "timeout",
            )

    def test_environment_injection(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = _prepare_n_tasks(Path(tmp), 1)
            backend = LocalSubprocessBackend(
                max_parallel_tasks=1,
                thread_environment={
                    "HDB_FAKE_WORKER_MODE": "env",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )
            backend.dispatch(
                experiment_dir=experiment_dir,
                task_indices=[0],
                worker_module=FAKE_WORKER,
            )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                TaskState.COMPLETED.value,
            )

    def test_fake_worker_resolves_without_ambient_pythonpath(self) -> None:
        """Workers use a private cwd; ``_mains`` must still import via PYTHONPATH."""
        with TemporaryDirectory() as tmp:
            experiment_dir = _prepare_n_tasks(Path(tmp), 1)
            cleared = {
                key: value
                for key, value in os.environ.items()
                if key != "PYTHONPATH"
            }
            with patch.dict(os.environ, cleared, clear=True):
                backend = LocalSubprocessBackend(
                    max_parallel_tasks=1,
                    thread_environment={"HDB_FAKE_WORKER_MODE": "success"},
                )
                backend.dispatch(
                    experiment_dir=experiment_dir,
                    task_indices=[0],
                    worker_module=FAKE_WORKER,
                )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                TaskState.COMPLETED.value,
            )

    def test_duplicate_dispatch_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = _prepare_n_tasks(Path(tmp), 1)
            backend = LocalSubprocessBackend(
                max_parallel_tasks=1,
                thread_environment={"HDB_FAKE_WORKER_MODE": "success"},
            )
            backend.dispatch(
                experiment_dir=experiment_dir,
                task_indices=[0],
                worker_module=FAKE_WORKER,
            )
            with self.assertRaisesRegex(LocalBackendError, "duplicate"):
                backend.dispatch(
                    experiment_dir=experiment_dir,
                    task_indices=[0],
                    worker_module=FAKE_WORKER,
                )

    def test_missing_summary_and_exit_137_map_to_failures(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_empty = _prepare_n_tasks(root / "empty", 1)
            backend_empty = LocalSubprocessBackend(
                max_parallel_tasks=1,
                thread_environment={"HDB_FAKE_WORKER_MODE": "empty_success"},
            )
            rec_empty = backend_empty.dispatch(
                experiment_dir=exp_empty,
                task_indices=[0],
                worker_module=FAKE_WORKER,
            )
            self.assertEqual(
                rec_empty[0].metadata["outcomes"][0]["failure_kind"],
                "missing_summary",
            )
            self.assertEqual(
                read_task_status(exp_empty / "tasks" / "task_000000")["state"],
                TaskState.FAILED.value,
            )

            exp_oom = _prepare_n_tasks(root / "oom", 1)
            backend_oom = LocalSubprocessBackend(
                max_parallel_tasks=1,
                thread_environment={"HDB_FAKE_WORKER_MODE": "exit_137"},
            )
            rec_oom = backend_oom.dispatch(
                experiment_dir=exp_oom,
                task_indices=[0],
                worker_module=FAKE_WORKER,
            )
            self.assertEqual(
                rec_oom[0].metadata["outcomes"][0]["failure_kind"],
                "oom_or_kill",
            )

    def test_keyboard_interrupt_cleanup(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = _prepare_n_tasks(Path(tmp), 1)
            backend = LocalSubprocessBackend(
                max_parallel_tasks=1,
                thread_environment={
                    "HDB_FAKE_WORKER_MODE": "sleep",
                    "HDB_FAKE_WORKER_SLEEP": "5",
                },
            )
            original_sleep = time.sleep
            calls = {"n": 0}

            def _sleep(seconds):
                calls["n"] += 1
                if calls["n"] >= 2:
                    raise KeyboardInterrupt
                return original_sleep(min(seconds, 0.05))

            with patch.object(backend, "_idle", side_effect=_sleep):
                with self.assertRaises(KeyboardInterrupt):
                    backend.dispatch(
                        experiment_dir=experiment_dir,
                        task_indices=[0],
                        worker_module=FAKE_WORKER,
                    )
            status = read_task_status(experiment_dir / "tasks" / "task_000000")
            self.assertEqual(status["state"], TaskState.CANCELLED.value)

    def test_factory_builds_local_backend(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "c.toml"
            cfg.write_text(
                "\n".join(
                    [
                        "schema_version = 1",
                        "[experiment]",
                        'id = "x"',
                        'workflow = "optimization_runs"',
                        'slab_type = "hp_slab"',
                        'problem_ids = ["L6750_B2"]',
                        'output_root = "experiments"',
                        "[[experiment.algorithms]]",
                        'name = "RANDOM_SEARCH"',
                        "max_evaluations = 1",
                        "[experiment.repetitions]",
                        "count = 1",
                        "seed_base = 1",
                        "[execution]",
                        'backend = "local"',
                        "max_parallel_tasks = 2",
                        "cpus_per_task = 1",
                        "[execution.local]",
                        'python_executable = ""',
                    ]
                ),
                encoding="utf-8",
            )
            backend = create_execution_backend(load_experiment_config(cfg))
            self.assertIsInstance(backend, LocalSubprocessBackend)
            self.assertEqual(backend.max_parallel_tasks, 2)


class TestLocalOptimisationSmoke(unittest.TestCase):
    """Two-task real optimisation worker smoke through the local backend."""

    def test_two_task_real_optimisation_smoke(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "smoke.toml"
            cfg.write_text(
                "\n".join(
                    [
                        "schema_version = 1",
                        "[experiment]",
                        'id = "opt_smoke"',
                        'workflow = "optimization_runs"',
                        'slab_type = "hp_slab"',
                        'problem_ids = ["L6750_B2"]',
                        'output_root = "experiments"',
                        "[[experiment.algorithms]]",
                        'name = "RANDOM_SEARCH"',
                        "max_evaluations = 2",
                        "[experiment.repetitions]",
                        "count = 2",
                        "seed_base = 7",
                        "[execution]",
                        'backend = "local"',
                        "max_parallel_tasks = 2",
                        "cpus_per_task = 1",
                        "[execution.thread_environment]",
                        'OMP_NUM_THREADS = "1"',
                        'MKL_NUM_THREADS = "1"',
                        "[execution.local]",
                        'python_executable = ""',
                    ]
                ),
                encoding="utf-8",
            )
            config = load_experiment_config(cfg)
            experiment_dir = root / "experiments" / config.experiment.id
            paths = create_experiment_layout(experiment_dir)
            hashes = snapshot_slab_inputs(
                slab_type="hp_slab",
                inputs_dir=paths.inputs,
            )
            tasks = plan_optimization_tasks(config)
            prepare_experiment_artifacts(
                config,
                root=experiment_dir,
                tasks=tasks,
                input_files=hashes,
            )
            for task in tasks:
                create_task_layout(paths, task.task_index)

            backend = LocalSubprocessBackend(
                max_parallel_tasks=2,
                thread_environment=config.execution.thread_environment,
            )
            records = backend.dispatch(
                experiment_dir=experiment_dir,
                task_indices=[0, 1],
                worker_module="core.ioh_core.workflows.optimization_runs.worker",
            )
            self.assertEqual(len(records[0].task_indices), 2)
            for index in (0, 1):
                status = read_task_status(
                    experiment_dir / "tasks" / f"task_{index:06d}"
                )
                self.assertEqual(status["state"], TaskState.COMPLETED.value)
                summary = read_json(
                    experiment_dir
                    / "tasks"
                    / f"task_{index:06d}"
                    / "attempts"
                    / "001"
                    / "summary.json"
                )
                self.assertIn("best_objective", summary)
                self.assertEqual(summary["evaluations"], 2)


if __name__ == "__main__":
    unittest.main()
