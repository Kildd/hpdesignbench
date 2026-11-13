"""
Phase B9 mocked gate: Slurm generation for brute-force shard experiments.
"""
from __future__ import annotations

import shutil
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.artifacts import read_task_status, reserve_attempt, transition_task_status
from core.execution_core.atomic_io import read_json, write_json_atomic
from core.execution_core.backends.factory import create_execution_backend
from core.execution_core.backends.slurm import SlurmArrayBackend
from core.execution_core.backends.slurm_commands import RecordingSlurmCommands
from core.execution_core.config import load_experiment_config
from core.execution_core.lifecycle import load_prepared_config, prepare_experiment
from core.execution_core.manifest import load_manifest
from core.execution_core.models import TaskState
from core.ioh_core.workflows.brute_force.models import BruteForceShardSpec

BF_WORKER = "core.ioh_core.workflows.brute_force.worker"


def _write_bf_slurm_config(
    path: Path,
    *,
    experiment_id: str,
    rows_per_shard: int = 24,
    max_rows: int = 288,
    maximum_array_size: int = 1000,
    array_concurrency: int = 4,
) -> Path:
    """
    Write a brute-force Slurm TOML with placeholder cluster fields.
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
            reuse_structural_artifacts = true
            acknowledge_large_run = true
            maximum_expected_rows = 1000000
            maximum_expected_bytes = 10000000000

            [execution]
            backend = "slurm"
            max_parallel_tasks = {int(array_concurrency)}
            timeout_seconds = 3600
            cpus_per_task = 1

            [execution.thread_environment]
            OMP_NUM_THREADS = "1"
            MKL_NUM_THREADS = "1"

            [execution.slurm]
            python_executable = "/placeholder/python"
            partition = "PLACEHOLDER_PARTITION"
            account = "PLACEHOLDER_ACCOUNT"
            memory = "8G"
            time_limit = "01:00:00"
            array_concurrency = {int(array_concurrency)}
            maximum_array_size = {int(maximum_array_size)}
            """
        ),
        encoding="utf-8",
    )
    return path


def _prepare_bf_slurm(
    root: Path,
    *,
    experiment_id: str,
    max_rows: int = 288,
    maximum_array_size: int = 1000,
    array_concurrency: int = 4,
) -> Path:
    """
    Prepare a sealed brute-force Slurm experiment (no real sbatch).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = _write_bf_slurm_config(
        root / f"{experiment_id}.toml",
        experiment_id=experiment_id,
        max_rows=max_rows,
        maximum_array_size=maximum_array_size,
        array_concurrency=array_concurrency,
    )
    experiment_dir = root / "experiments" / experiment_id
    prepare_experiment(config_path=config_path, output=experiment_dir)
    return experiment_dir


def _clone_pending(src: Path, dst: Path) -> Path:
    """
    Copy a prepared experiment and keep all tasks pending.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _backend_from_prepared(
    experiment_dir: Path,
    *,
    commands: RecordingSlurmCommands,
) -> SlurmArrayBackend:
    """
    Build a Slurm backend from the sealed experiment config.
    """
    config = load_prepared_config(experiment_dir)
    slurm = config.execution.slurm
    assert slurm is not None
    return SlurmArrayBackend(
        python_executable=slurm.python_executable,
        partition=slurm.partition,
        account=slurm.account,
        memory=slurm.memory,
        time_limit=slurm.time_limit,
        cpus_per_task=config.execution.cpus_per_task,
        array_concurrency=slurm.array_concurrency,
        maximum_array_size=slurm.maximum_array_size,
        thread_environment=config.execution.thread_environment,
        commands=commands,
    )


def _force_failed_then_pending(experiment_dir: Path, task_index: int) -> None:
    """
    Consume attempt 001 as failed, then reset status to pending for retry.
    """
    task_dir = Path(experiment_dir) / "tasks" / f"task_{task_index:06d}"
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
        metadata={"failure_kind": "injected"},
    )
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


class TestBruteForceSlurmMocked(unittest.TestCase):
    """Mocked Slurm submit path for brute-force shards."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        # 12 shards × 24 rows (smoke-scale floor); preflight runs 5 probes once.
        cls.golden = _prepare_bf_slurm(
            cls.root / "golden",
            experiment_id="bf_slurm_golden",
            max_rows=288,
            maximum_array_size=1000,
            array_concurrency=4,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def _fresh(self, name: str) -> Path:
        return _clone_pending(self.golden, self.root / "runs" / name)

    def test_prepare_seals_bf_worker_and_twelve_shards(self) -> None:
        manifest = load_manifest(self.golden / "manifest.json")
        self.assertEqual(manifest.workflow_type, "brute_force")
        self.assertEqual(manifest.worker_module, BF_WORKER)
        self.assertEqual(len(manifest.tasks), 12)
        for task in manifest.tasks:
            spec = BruteForceShardSpec.from_payload(task.payload)
            self.assertEqual(spec.stop_index - spec.start_index, 24)
        self.assertTrue((self.golden / "dataset" / "schema.json").is_file())
        self.assertTrue((self.golden / "dataset" / "parts").is_dir())

    def test_dispatch_uses_bf_worker_and_shard_task_map(self) -> None:
        experiment_dir = self._fresh("submit_all")
        commands = RecordingSlurmCommands(sbatch_job_ids=["9001"])
        backend = _backend_from_prepared(experiment_dir, commands=commands)
        records = backend.dispatch(
            experiment_dir=experiment_dir,
            task_indices=list(range(12)),
            worker_module=BF_WORKER,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(len(commands.sbatch_calls), 1)

        submission_dir = experiment_dir / "submissions" / "submission_001"
        job = (submission_dir / "job.sh").read_text(encoding="utf-8")
        self.assertIn(BF_WORKER, job)
        self.assertIn("#SBATCH --partition=PLACEHOLDER_PARTITION", job)
        self.assertIn("#SBATCH --account=PLACEHOLDER_ACCOUNT", job)
        self.assertIn("#SBATCH --array=0-11%4", job)
        self.assertIn("PYTHON=/placeholder/python", job)
        self.assertIn(f"WORKER_MODULE={BF_WORKER}", job)

        task_map = (submission_dir / "task-map.tsv").read_text(encoding="utf-8")
        self.assertIn("0\t0\t1", task_map)
        self.assertIn("11\t11\t1", task_map)
        meta = read_json(submission_dir / "submission.json")
        self.assertEqual(meta["worker_module"], BF_WORKER)
        self.assertEqual(meta["job_id"], "9001")

        for index in range(12):
            status = read_task_status(
                experiment_dir / "tasks" / f"task_{index:06d}"
            )
            self.assertEqual(status["state"], TaskState.DISPATCHED.value)
            self.assertEqual(status["attempt"], 1)

    def test_noncontiguous_retry_task_map(self) -> None:
        experiment_dir = self._fresh("retry")
        for index in (1, 4, 9):
            _force_failed_then_pending(experiment_dir, index)

        commands = RecordingSlurmCommands(sbatch_job_ids=["9002"])
        backend = _backend_from_prepared(experiment_dir, commands=commands)
        records = backend.dispatch(
            experiment_dir=experiment_dir,
            task_indices=[1, 4, 9],
            worker_module=BF_WORKER,
        )
        self.assertEqual(list(records[0].task_indices), [1, 4, 9])
        attempts = records[0].metadata["attempts"]
        self.assertEqual(attempts["1"], 2)
        self.assertEqual(attempts["4"], 2)
        self.assertEqual(attempts["9"], 2)

        task_map = (
            experiment_dir / "submissions" / "submission_001" / "task-map.tsv"
        ).read_text(encoding="utf-8")
        self.assertIn("0\t1\t2", task_map)
        self.assertIn("1\t4\t2", task_map)
        self.assertIn("2\t9\t2", task_map)
        job = (
            experiment_dir / "submissions" / "submission_001" / "job.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(BF_WORKER, job)
        self.assertIn("#SBATCH --array=0-2%4", job)

        # Untouched shards stay pending and unpublished.
        for index in (0, 2, 3):
            status = read_task_status(
                experiment_dir / "tasks" / f"task_{index:06d}"
            )
            self.assertEqual(status["state"], TaskState.PENDING.value)
            self.assertEqual(status["attempt"], 0)

    def test_factory_builds_slurm_backend_from_bf_config(self) -> None:
        config = load_experiment_config(
            self.root / "golden" / "bf_slurm_golden.toml"
        )
        backend = create_execution_backend(config)
        self.assertIsInstance(backend, SlurmArrayBackend)
        self.assertEqual(backend.partition, "PLACEHOLDER_PARTITION")
        self.assertEqual(backend.maximum_array_size, 1000)


class TestBruteForceSlurmChunking(unittest.TestCase):
    """BF task counts above maximum_array_size create multiple arrays."""

    def test_chunking_across_multiple_arrays(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            # 25 shards with max array size 10 → three sbatch calls.
            experiment_dir = _prepare_bf_slurm(
                root,
                experiment_id="bf_slurm_chunk",
                max_rows=600,
                maximum_array_size=10,
                array_concurrency=4,
            )
            manifest = load_manifest(experiment_dir / "manifest.json")
            self.assertEqual(len(manifest.tasks), 25)
            self.assertEqual(manifest.worker_module, BF_WORKER)

            commands = RecordingSlurmCommands(
                sbatch_job_ids=["1", "2", "3"],
            )
            backend = _backend_from_prepared(experiment_dir, commands=commands)
            records = backend.dispatch(
                experiment_dir=experiment_dir,
                task_indices=list(range(25)),
                worker_module=manifest.worker_module,
            )
            self.assertEqual(len(records), 3)
            self.assertEqual(len(commands.sbatch_calls), 3)
            self.assertEqual(records[0].metadata["array_count"], 10)
            self.assertEqual(records[1].metadata["array_count"], 10)
            self.assertEqual(records[2].metadata["array_count"], 5)

            job3 = (
                experiment_dir / "submissions" / "submission_003" / "job.sh"
            ).read_text(encoding="utf-8")
            self.assertIn(BF_WORKER, job3)
            self.assertIn("#SBATCH --array=0-4%4", job3)
            map3 = (
                experiment_dir / "submissions" / "submission_003" / "task-map.tsv"
            ).read_text(encoding="utf-8")
            self.assertIn("0\t20\t1", map3)
            self.assertIn("4\t24\t1", map3)


if __name__ == "__main__":
    unittest.main()
