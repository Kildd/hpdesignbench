"""
C1.6 gate: registered sensitivity lifecycle (local cheap E2E + mocked Slurm).
"""
from __future__ import annotations

import shutil
import textwrap
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow.parquet as pq

from core.execution_core.artifacts import (
    read_task_status,
    reserve_attempt,
    transition_task_status,
)
from core.execution_core.atomic_io import read_json
from core.execution_core.output_layout import (
    resolve_results_root,
    saltelli_publication_dir,
)
from core.execution_core.backends.factory import create_execution_backend
from core.execution_core.backends.slurm import SlurmArrayBackend
from core.execution_core.backends.slurm_commands import RecordingSlurmCommands
from core.execution_core.config import load_experiment_config
from core.execution_core.lifecycle import (
    collect_experiment,
    load_prepared_config,
    pending_task_indices,
    prepare_experiment,
    retry_failed_tasks,
)
from core.execution_core.manifest import load_manifest
from core.execution_core.models import TaskState
from core.ioh_core.problem_builder import build_problem_from_snapshot
from core.ioh_core.workflows.registry import get_workflow, registered_workflows
from core.ioh_core.workflows.sensitivity_analysis.deps import (
    SensitivityDependencyError,
    import_salib,
)
from core.ioh_core.workflows.sensitivity_analysis.worker import run_worker

SA_WORKER = "core.ioh_core.workflows.sensitivity_analysis.worker"

_SCIENCE_COLUMNS = (
    "global_index",
    "sample_index",
    "matrix_offset",
    "matrix_kind",
    "factor_index",
    "factor_name",
    "shard_index",
    "problem_id",
    "analysis_ok",
    "y",
    "y_p",
)


def _salib_available() -> bool:
    try:
        import_salib()
    except SensitivityDependencyError:
        return False
    return True


def _cheap_analysis(params: dict) -> dict:
    y = float(params["geom_h_ges_mm"]) + 0.01 * float(params["geom_t_mm"])
    return {
        "y": y,
        "y_p": y,
        "constraint_values": {},
        "violations_": {},
        "penalties_": {},
        "penalty_sum": 0.0,
        "penalty_factor": 1.0,
        "params": dict(params),
        "analysis_ok": True,
        "constraint_feasible": True,
        "feasible": True,
        "analysis_error": None,
        "remapped_utilizations": [],
        "cost_components_missing": [],
    }


def _wrap_bundle_with_cheap_analysis(bundle: dict) -> dict:
    names = tuple(bundle["constraints"].keys())

    def analysis_fn(params: dict) -> dict:
        result = _cheap_analysis(params)
        result["constraint_values"] = {name: 0.5 for name in names}
        result["violations_"] = {name: 0.0 for name in names}
        result["penalties_"] = {name: 0.0 for name in names}
        return result

    wrapped = dict(bundle)
    wrapped["analysis_fn"] = analysis_fn
    return wrapped


def _write_local_config(
    path: Path,
    *,
    experiment_id: str,
    max_parallel_tasks: int,
    sample_size: int = 4,
    blocks_per_shard: int = 2,
) -> Path:
    path.write_text(
        textwrap.dedent(
            f"""\
            schema_version = 1

            [experiment]
            id = "{experiment_id}"
            workflow = "sensitivity_analysis"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [sensitivity_analysis]
            sample_size = {sample_size}
            factor_names = ["geom_h_ges_mm", "geom_t_mm"]
            response_key = "y"
            scramble = true
            seed = 11
            blocks_per_shard = {blocks_per_shard}
            parquet_batch_rows = 4
            compression = "zstd"
            bootstrap_resamples = 8
            bootstrap_confidence_level = 0.95
            bootstrap_seed = 1
            compute_bootstrap_ci = false
            acknowledge_large_run = true
            maximum_expected_evaluations = 100000
            maximum_expected_bytes = 2000000000

            [execution]
            backend = "local"
            max_parallel_tasks = {max_parallel_tasks}
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        ),
        encoding="utf-8",
    )
    return path


def _write_slurm_config(
    path: Path,
    *,
    experiment_id: str,
    sample_size: int = 8,
    blocks_per_shard: int = 2,
    maximum_array_size: int = 1000,
    array_concurrency: int = 4,
) -> Path:
    path.write_text(
        textwrap.dedent(
            f"""\
            schema_version = 1

            [experiment]
            id = "{experiment_id}"
            workflow = "sensitivity_analysis"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [sensitivity_analysis]
            sample_size = {sample_size}
            factor_names = ["geom_h_ges_mm", "geom_t_mm"]
            response_key = "y"
            scramble = true
            seed = 11
            blocks_per_shard = {blocks_per_shard}
            parquet_batch_rows = 4
            compression = "zstd"
            bootstrap_resamples = 8
            bootstrap_confidence_level = 0.95
            bootstrap_seed = 1
            compute_bootstrap_ci = false
            acknowledge_large_run = true
            maximum_expected_evaluations = 100000
            maximum_expected_bytes = 2000000000

            [execution]
            backend = "slurm"
            max_parallel_tasks = {array_concurrency}
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
            array_concurrency = {array_concurrency}
            maximum_array_size = {maximum_array_size}
            """
        ),
        encoding="utf-8",
    )
    return path


def _dispatch_cheap_inprocess(
    experiment_dir: Path,
    *,
    max_workers: int,
    task_indices: list[int] | None = None,
) -> None:
    """
    Cheap local dispatch: same attempt ownership as the local backend, but
    evaluate in-process with a deterministic fake analysis (subprocess workers
    cannot see unittest mocks).
    """
    indices = (
        list(task_indices)
        if task_indices is not None
        else pending_task_indices(experiment_dir)
    )
    if not indices:
        return

    real_build = build_problem_from_snapshot

    def fake_build(**kwargs):
        return _wrap_bundle_with_cheap_analysis(real_build(**kwargs))

    def _run_one(task_index: int) -> None:
        task_dir = experiment_dir / "tasks" / f"task_{task_index:06d}"
        attempt = reserve_attempt(task_dir)
        transition_task_status(
            task_dir,
            expected={TaskState.PENDING},
            new_state=TaskState.DISPATCHED,
            attempt=attempt.attempt,
        )
        run_worker(
            experiment_dir=experiment_dir,
            task_index=task_index,
            attempt=attempt.attempt,
        )

    with patch(
        "core.ioh_core.workflows.sensitivity_analysis.worker.build_problem_from_snapshot",
        side_effect=fake_build,
    ):
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            futures = [pool.submit(_run_one, index) for index in indices]
            for future in as_completed(futures):
                future.result()


def _run_cheap_lifecycle(
    root: Path,
    *,
    experiment_id: str,
    max_parallel_tasks: int,
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = _write_local_config(
        root / f"{experiment_id}.toml",
        experiment_id=experiment_id,
        max_parallel_tasks=max_parallel_tasks,
    )
    experiment_dir = root / "experiments" / experiment_id
    prepare_experiment(config_path=config_path, output=experiment_dir)
    _dispatch_cheap_inprocess(
        experiment_dir, max_workers=max_parallel_tasks
    )
    collect_experiment(experiment_dir, require_complete=True)
    return experiment_dir


def _scientific_rows(experiment_dir: Path) -> list[dict]:
    rows: list[dict] = []
    parts = sorted((experiment_dir / "dataset" / "parts").glob("shard-*.parquet"))
    for part in parts:
        table = pq.read_table(part, columns=list(_SCIENCE_COLUMNS))
        # Also capture factor idx columns present in the table.
        idx_cols = [
            name
            for name in pq.read_schema(part).names
            if name.startswith("idx__")
        ]
        if idx_cols:
            table = pq.read_table(
                part, columns=list(_SCIENCE_COLUMNS) + idx_cols
            )
        for index in range(table.num_rows):
            row = {
                name: table.column(name)[index].as_py()
                for name in table.column_names
            }
            rows.append(row)
    rows.sort(key=lambda item: int(item["global_index"]))
    return rows


def _design_hashes(experiment_dir: Path) -> dict[str, str]:
    manifest = load_manifest(experiment_dir / "manifest.json")
    return {
        key: digest
        for key, digest in manifest.input_files.items()
        if key.startswith("sensitivity/")
    }


@unittest.skipUnless(_salib_available(), 'install with: pip install -e ".[sensitivity]"')
class TestSensitivityWorkflowLocal(unittest.TestCase):
    """Registry-backed prepare/collect with cheap in-process workers."""

    def test_registry_lists_sensitivity(self) -> None:
        self.assertIn("sensitivity_analysis", registered_workflows())
        entry = get_workflow("sensitivity_analysis")
        self.assertEqual(entry.worker_module, SA_WORKER)

    def test_parallel_one_and_two_match_scientifically(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            first = _run_cheap_lifecycle(
                root / "p1", experiment_id="sa_p1", max_parallel_tasks=1
            )
            second = _run_cheap_lifecycle(
                root / "p2", experiment_id="sa_p2", max_parallel_tasks=2
            )

            self.assertEqual(_design_hashes(first), _design_hashes(second))
            self.assertEqual(_scientific_rows(first), _scientific_rows(second))

            indices1 = read_json(
                saltelli_publication_dir(
                    resolve_results_root(first), first.name
                )
                / "sensitivity_indices.json"
            )
            indices2 = read_json(
                saltelli_publication_dir(
                    resolve_results_root(second), second.name
                )
                / "sensitivity_indices.json"
            )
            self.assertEqual(indices1, indices2)
            self.assertTrue((first / "dataset" / "_SUCCESS").is_file())
            self.assertTrue((first / "aggregate" / "_SUCCESS").is_file())
            self.assertTrue((second / "dataset" / "_SUCCESS").is_file())
            self.assertTrue((second / "aggregate" / "_SUCCESS").is_file())

    def test_failed_shard_retry_then_collect(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            config_path = _write_local_config(
                root / "sa_retry.toml",
                experiment_id="sa_retry",
                max_parallel_tasks=1,
            )
            experiment_dir = root / "experiments" / "sa_retry"
            prepare_experiment(config_path=config_path, output=experiment_dir)

            # Complete shard 0; fail shard 1 without publishing a part.
            _dispatch_cheap_inprocess(
                experiment_dir, max_workers=1, task_indices=[0]
            )
            task1 = experiment_dir / "tasks" / "task_000001"
            attempt = reserve_attempt(task1)
            transition_task_status(
                task1,
                expected={TaskState.PENDING},
                new_state=TaskState.DISPATCHED,
                attempt=attempt.attempt,
            )
            transition_task_status(
                task1,
                expected={TaskState.DISPATCHED},
                new_state=TaskState.RUNNING,
                attempt=attempt.attempt,
            )
            transition_task_status(
                task1,
                expected={TaskState.RUNNING},
                new_state=TaskState.FAILED,
                attempt=attempt.attempt,
                metadata={"failure_kind": "injected"},
            )
            self.assertFalse(
                (experiment_dir / "dataset" / "parts" / "shard-000001.parquet").exists()
            )

            with self.assertRaises(Exception):
                collect_experiment(experiment_dir, require_complete=True)

            reset = retry_failed_tasks(experiment_dir)
            self.assertEqual(reset, [1])
            _dispatch_cheap_inprocess(
                experiment_dir, max_workers=1, task_indices=[1]
            )
            summary = collect_experiment(experiment_dir, require_complete=True)
            self.assertTrue(summary["scientific_ok"])
            self.assertTrue((experiment_dir / "aggregate" / "_SUCCESS").is_file())
            self.assertEqual(
                read_task_status(task1)["state"], TaskState.COMPLETED.value
            )
            self.assertEqual(read_task_status(task1)["attempt"], 2)


@unittest.skipUnless(_salib_available(), 'install with: pip install -e ".[sensitivity]"')
class TestSensitivityWorkflowSlurmMocked(unittest.TestCase):
    """Mocked Slurm generation for the registered sensitivity worker."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        config_path = _write_slurm_config(
            cls.root / "sa_slurm.toml",
            experiment_id="sa_slurm_golden",
            sample_size=8,
            blocks_per_shard=2,
            maximum_array_size=3,
            array_concurrency=2,
        )
        cls.golden = cls.root / "experiments" / "sa_slurm_golden"
        prepare_experiment(config_path=config_path, output=cls.golden)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def _fresh(self, name: str) -> Path:
        dst = self.root / "runs" / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.golden, dst)
        return dst

    def test_prepare_seals_sensitivity_worker_and_design(self) -> None:
        manifest = load_manifest(self.golden / "manifest.json")
        self.assertEqual(manifest.workflow_type, "sensitivity_analysis")
        self.assertEqual(manifest.worker_module, SA_WORKER)
        self.assertEqual(len(manifest.tasks), 4)  # N=8, blocks_per_shard=2
        self.assertTrue(
            (self.golden / "inputs" / "sensitivity" / "design.json").is_file()
        )
        self.assertTrue((self.golden / "dataset" / "schema.json").is_file())
        for key in (
            "sensitivity/A_unit.npy",
            "sensitivity/B_unit.npy",
            "sensitivity/A_index.npy",
            "sensitivity/B_index.npy",
            "sensitivity/design.json",
        ):
            self.assertIn(key, manifest.input_files)

    def test_dispatch_uses_sensitivity_worker_and_splits_arrays(self) -> None:
        experiment_dir = self._fresh("submit_all")
        commands = RecordingSlurmCommands(sbatch_job_ids=["9101", "9102"])
        config = load_prepared_config(experiment_dir)
        slurm = config.execution.slurm
        assert slurm is not None
        backend = SlurmArrayBackend(
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
        records = backend.dispatch(
            experiment_dir=experiment_dir,
            task_indices=list(range(4)),
            worker_module=SA_WORKER,
        )
        # maximum_array_size=3 → two array jobs (3 + 1).
        self.assertEqual(len(records), 2)
        self.assertEqual(len(commands.sbatch_calls), 2)

        first = experiment_dir / "submissions" / "submission_001"
        job = (first / "job.sh").read_text(encoding="utf-8")
        self.assertIn(SA_WORKER, job)
        self.assertIn("#SBATCH --array=0-2%2", job)
        self.assertIn(f"WORKER_MODULE={SA_WORKER}", job)

        second = experiment_dir / "submissions" / "submission_002"
        job2 = (second / "job.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-0%2", job2)

        for index in range(4):
            status = read_task_status(
                experiment_dir / "tasks" / f"task_{index:06d}"
            )
            self.assertEqual(status["state"], TaskState.DISPATCHED.value)

    def test_factory_builds_slurm_backend_from_sa_config(self) -> None:
        config = load_experiment_config(self.root / "sa_slurm.toml")
        backend = create_execution_backend(config)
        self.assertIsInstance(backend, SlurmArrayBackend)
        self.assertEqual(backend.partition, "PLACEHOLDER_PARTITION")
        self.assertEqual(backend.maximum_array_size, 3)


if __name__ == "__main__":
    unittest.main()
