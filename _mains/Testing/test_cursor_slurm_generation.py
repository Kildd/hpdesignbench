"""
Phase A9 gate: Slurm array script generation and sbatch submission.
"""
from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.artifacts import (
    create_experiment_layout,
    create_task_layout,
    read_task_status,
    reserve_attempt,
    transition_task_status,
)
from core.execution_core.atomic_io import read_json
from core.execution_core.backends.factory import create_execution_backend
from core.execution_core.backends.slurm import (
    SlurmArrayBackend,
    SyncSlurmUnavailable,
    chunk_indices,
    render_job_script,
)
from core.execution_core.backends.slurm_commands import RecordingSlurmCommands
from core.execution_core.config import load_experiment_config
from core.execution_core.manifest import prepare_experiment_artifacts
from core.execution_core.models import TaskState


def _write_slurm_config(path: Path, *, maximum_array_size: int = 1000) -> Path:
    """
    Write a complete Slurm-backend experiment TOML.

    Parameters
    ----------
    path : Path
        Destination TOML path.
    maximum_array_size : int, optional
        Slurm maximum array size.

    Returns
    -------
    Path
        Written path.
    """
    path.write_text(
        textwrap.dedent(
            f"""\
            schema_version = 1

            [experiment]
            id = "slurm_gen"
            workflow = "optimization_runs"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [[experiment.algorithms]]
            name = "RANDOM_SEARCH"
            max_evaluations = 1

            [experiment.repetitions]
            count = 1
            seed_base = 1

            [execution]
            backend = "slurm"
            max_parallel_tasks = 8
            cpus_per_task = 2
            timeout_seconds = 2700

            [execution.thread_environment]
            OMP_NUM_THREADS = "1"
            MKL_NUM_THREADS = "1"

            [execution.slurm]
            python_executable = "/cluster/python/bin/python"
            partition = "cpu-short"
            account = "hpdesign"
            memory = "8G"
            time_limit = "00:45:00"
            array_concurrency = 64
            maximum_array_size = {maximum_array_size}
            """
        ),
        encoding="utf-8",
    )
    return path


def _prepare_n_tasks(root: Path, n: int, *, maximum_array_size: int = 1000) -> Path:
    """
    Create an experiment with ``n`` pending task directories.

    Parameters
    ----------
    root : Path
        Temporary parent directory.
    n : int
        Number of tasks.
    maximum_array_size : int, optional
        Slurm chunk size limit sealed into the config.

    Returns
    -------
    Path
        Experiment root.
    """
    config_path = _write_slurm_config(
        root / "cfg.toml",
        maximum_array_size=maximum_array_size,
    )
    config = load_experiment_config(config_path)
    # Put the experiment under a path that may contain spaces.
    experiment_dir = root / "experiments" / "slurm gen"
    create_experiment_layout(experiment_dir)
    prepare_experiment_artifacts(config, root=experiment_dir, tasks=())
    paths = create_experiment_layout(experiment_dir)
    for index in range(n):
        create_task_layout(paths, index)
    return experiment_dir


class TestSlurmChunking(unittest.TestCase):
    """Pure chunk helper behaviour."""

    def test_chunk_2500(self) -> None:
        chunks = chunk_indices(list(range(2500)), 1000)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 1000)
        self.assertEqual(len(chunks[1]), 1000)
        self.assertEqual(len(chunks[2]), 500)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[2][-1], 2499)


class TestSlurmScriptRendering(unittest.TestCase):
    """Directive and quoting checks without submission."""

    def test_directives_and_space_quoting(self) -> None:
        experiment_dir = Path("/data/my experiments/exp1")
        submission_dir = experiment_dir / "submissions" / "submission_001"
        script = render_job_script(
            experiment_dir=experiment_dir,
            worker_module="core.ioh_core.workflows.optimization_runs.worker",
            python_executable=Path("/opt/python with spaces/bin/python"),
            submission_dir=submission_dir,
            task_map_path=submission_dir / "task-map.tsv",
            partition="cpu-short",
            account="hpdesign",
            memory="16G",
            time_limit="01:00:00",
            cpus_per_task=4,
            array_concurrency=32,
            array_count=10,
            thread_environment={"OMP_NUM_THREADS": "1"},
        )
        self.assertIn("set -euo pipefail", script)
        self.assertIn("#SBATCH --partition=cpu-short", script)
        self.assertIn("#SBATCH --account=hpdesign", script)
        self.assertIn("#SBATCH --mem=16G", script)
        self.assertIn("#SBATCH --time=01:00:00", script)
        self.assertIn("#SBATCH --cpus-per-task=4", script)
        self.assertIn("#SBATCH --array=0-9%32", script)
        self.assertIn("'/data/my experiments/exp1'", script)
        self.assertIn("'/opt/python with spaces/bin/python'", script)
        self.assertIn("export OMP_NUM_THREADS=", script)
        self.assertIn("Never install packages", script)
        self.assertNotIn("pip install", script)


class TestSlurmArrayBackend(unittest.TestCase):
    """Generation, chunked sbatch, retries, and safe unavailable behaviour."""

    def test_submit_resources_and_parsable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_n_tasks(root, 3)
            commands = RecordingSlurmCommands(sbatch_job_ids=["4242"])
            backend = SlurmArrayBackend(
                python_executable="/cluster/python/bin/python",
                partition="cpu-short",
                account="hpdesign",
                memory="8G",
                time_limit="00:45:00",
                cpus_per_task=2,
                array_concurrency=64,
                maximum_array_size=1000,
                thread_environment={"OMP_NUM_THREADS": "1"},
                commands=commands,
            )
            records = backend.dispatch(
                experiment_dir=experiment_dir,
                task_indices=[0, 1, 2],
                worker_module="core.ioh_core.workflows.optimization_runs.worker",
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].metadata["job_id"], "4242")
            self.assertEqual(len(commands.sbatch_calls), 1)

            submission_dir = experiment_dir / "submissions" / "submission_001"
            self.assertTrue((submission_dir / "job.sh").is_file())
            self.assertTrue((submission_dir / "task-map.tsv").is_file())
            meta = read_json(submission_dir / "submission.json")
            self.assertEqual(meta["job_id"], "4242")
            self.assertEqual(meta["partition"], "cpu-short")
            self.assertEqual(meta["account"], "hpdesign")
            self.assertEqual(meta["memory"], "8G")
            self.assertEqual(meta["time_limit"], "00:45:00")
            self.assertEqual(meta["array_concurrency"], 64)

            job = (submission_dir / "job.sh").read_text(encoding="utf-8")
            self.assertIn("#SBATCH --array=0-2%64", job)
            self.assertIn("#SBATCH --partition=cpu-short", job)
            self.assertIn("#SBATCH --cpus-per-task=2", job)
            # Path with spaces must be shell-quoted in the generated script.
            self.assertIn("slurm gen", job)
            self.assertRegex(job, r"EXPERIMENT_DIR='[^']*slurm gen'")

            for index in range(3):
                status = read_task_status(
                    experiment_dir / "tasks" / f"task_{index:06d}"
                )
                self.assertEqual(status["state"], TaskState.DISPATCHED.value)
                self.assertEqual(status["attempt"], 1)

            task_map = (submission_dir / "task-map.tsv").read_text(encoding="utf-8")
            self.assertIn("0\t0\t1", task_map)
            self.assertIn("2\t2\t1", task_map)

    def test_chunking_2500_tasks(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_n_tasks(root, 2500, maximum_array_size=1000)
            commands = RecordingSlurmCommands(
                sbatch_job_ids=["1", "2", "3"],
            )
            backend = SlurmArrayBackend(
                python_executable="/cluster/python/bin/python",
                partition="cpu-short",
                account="hpdesign",
                memory="8G",
                time_limit="00:45:00",
                array_concurrency=64,
                maximum_array_size=1000,
                commands=commands,
            )
            records = backend.dispatch(
                experiment_dir=experiment_dir,
                task_indices=list(range(2500)),
                worker_module="core.ioh_core.workflows.optimization_runs.worker",
            )
            self.assertEqual(len(records), 3)
            self.assertEqual(len(commands.sbatch_calls), 3)
            self.assertEqual(records[0].metadata["array_count"], 1000)
            self.assertEqual(records[1].metadata["array_count"], 1000)
            self.assertEqual(records[2].metadata["array_count"], 500)
            self.assertTrue(
                (experiment_dir / "submissions" / "submission_003" / "job.sh").is_file()
            )
            job3 = (
                experiment_dir / "submissions" / "submission_003" / "job.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --array=0-499%64", job3)

    def test_arbitrary_retry_mappings(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_n_tasks(root, 6)
            # Pretend task 1 and 4 already consumed attempt 001 and are pending again.
            for index in (1, 4):
                task_dir = experiment_dir / "tasks" / f"task_{index:06d}"
                attempt = reserve_attempt(task_dir)
                transition_task_status(
                    task_dir,
                    expected={TaskState.PENDING},
                    new_state=TaskState.DISPATCHED,
                    attempt=attempt.attempt,
                )
                transition_task_status(
                    task_dir,
                    expected={TaskState.DISPATCHED},
                    new_state=TaskState.RUNNING,
                    attempt=attempt.attempt,
                )
                transition_task_status(
                    task_dir,
                    expected={TaskState.RUNNING},
                    new_state=TaskState.FAILED,
                    attempt=attempt.attempt,
                    metadata={"failure_kind": "boom"},
                )
                # Administrative reset to pending (same as lifecycle retry).
                from core.execution_core.atomic_io import write_json_atomic

                status = read_task_status(task_dir)
                write_json_atomic(
                    task_dir / "status.json",
                    {
                        "task_id": status["task_id"],
                        "task_index": status["task_index"],
                        "state": TaskState.PENDING.value,
                        "attempt": 0,
                        "updated_at": status["updated_at"],
                        "metadata": {},
                    },
                )

            commands = RecordingSlurmCommands(sbatch_job_ids=["77"])
            backend = SlurmArrayBackend(
                python_executable="/cluster/python/bin/python",
                partition="cpu-short",
                account="",
                memory="8G",
                time_limit="00:30:00",
                array_concurrency=8,
                maximum_array_size=1000,
                commands=commands,
            )
            records = backend.dispatch(
                experiment_dir=experiment_dir,
                task_indices=[1, 4, 5],
                worker_module="core.ioh_core.workflows.optimization_runs.worker",
            )
            self.assertEqual(list(records[0].task_indices), [1, 4, 5])
            attempts = records[0].metadata["attempts"]
            self.assertEqual(attempts["1"], 2)
            self.assertEqual(attempts["4"], 2)
            self.assertEqual(attempts["5"], 1)
            task_map = (
                experiment_dir / "submissions" / "submission_001" / "task-map.tsv"
            ).read_text(encoding="utf-8")
            self.assertIn("0\t1\t2", task_map)
            self.assertIn("1\t4\t2", task_map)
            self.assertIn("2\t5\t1", task_map)
            job = (
                experiment_dir / "submissions" / "submission_001" / "job.sh"
            ).read_text(encoding="utf-8")
            self.assertNotIn("#SBATCH --account=", job)

    def test_missing_slurm_does_not_mutate_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_n_tasks(root, 2)
            commands = RecordingSlurmCommands(available_flag=False)
            backend = SlurmArrayBackend(
                python_executable="/cluster/python/bin/python",
                partition="cpu-short",
                account="hpdesign",
                memory="8G",
                time_limit="00:45:00",
                commands=commands,
            )
            with self.assertRaises(SyncSlurmUnavailable):
                backend.dispatch(
                    experiment_dir=experiment_dir,
                    task_indices=[0, 1],
                    worker_module="core.ioh_core.workflows.optimization_runs.worker",
                )
            self.assertEqual(commands.sbatch_calls, [])
            self.assertFalse((experiment_dir / "submissions" / "submission_001").exists())
            for index in range(2):
                status = read_task_status(
                    experiment_dir / "tasks" / f"task_{index:06d}"
                )
                self.assertEqual(status["state"], TaskState.PENDING.value)
                self.assertEqual(status["attempt"], 0)

    def test_factory_builds_slurm_backend(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_experiment_config(_write_slurm_config(root / "cfg.toml"))
            backend = create_execution_backend(config)
            self.assertIsInstance(backend, SlurmArrayBackend)
            self.assertEqual(backend.partition, "cpu-short")
            self.assertEqual(backend.maximum_array_size, 1000)


if __name__ == "__main__":
    unittest.main()
