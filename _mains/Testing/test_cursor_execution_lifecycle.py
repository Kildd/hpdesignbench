"""
Phase A8 gate: lifecycle prepare/dispatch/collect/retry/reconcile.
"""
from __future__ import annotations

import textwrap
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.artifacts import (
    read_task_status,
    reserve_attempt,
    transition_task_status,
)
from core.execution_core.atomic_io import write_json_atomic
from core.execution_core.lifecycle import (
    LifecycleError,
    cancel_tasks,
    collect_experiment,
    dispatch_experiment,
    list_task_statuses,
    prepare_experiment,
    reconcile_stale_tasks,
    retry_failed_tasks,
    touch_attempt_heartbeat,
    write_attempt_owner,
)
from core.execution_core.models import TaskState


def _write_config(path: Path, *, repetitions: int = 2, max_evaluations: int = 2) -> Path:
    """
    Write a tiny local optimisation TOML.

    Parameters
    ----------
    path : Path
        Destination TOML path.
    repetitions : int, optional
        Repetition count.
    max_evaluations : int, optional
        Evaluation budget.

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
            id = "lifecycle_smoke"
            workflow = "optimization_runs"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [[experiment.algorithms]]
            name = "RANDOM_SEARCH"
            max_evaluations = {max_evaluations}

            [experiment.repetitions]
            count = {repetitions}
            seed_base = 7

            [execution]
            backend = "local"
            max_parallel_tasks = 2
            timeout_seconds = 120
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        ),
        encoding="utf-8",
    )
    return path


class TestLifecyclePrepareDispatchCollect(unittest.TestCase):
    """End-to-end local lifecycle behaviours."""

    def test_prepare_dispatch_collect_and_retry(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root / "cfg.toml", repetitions=2)
            experiment_dir = root / "exp"
            paths = prepare_experiment(config_path=config_path, output=experiment_dir)
            self.assertTrue(paths.manifest.is_file())
            status = list_task_statuses(experiment_dir)
            self.assertEqual(status.counts["pending"], 2)
            self.assertFalse(status.complete)

            dispatch_experiment(experiment_dir)
            status = list_task_statuses(experiment_dir)
            self.assertTrue(status.complete)
            self.assertEqual(status.counts["completed"], 2)

            summary = collect_experiment(experiment_dir, require_complete=True)
            self.assertTrue(summary["complete"])
            self.assertTrue((experiment_dir / "aggregate" / "tasks.csv").is_file())

            # Completed tasks are not eligible for retry.
            reset = retry_failed_tasks(experiment_dir)
            self.assertEqual(reset, [])
            self.assertEqual(list_task_statuses(experiment_dir).counts["completed"], 2)

    def test_incomplete_collect_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root / "cfg.toml", repetitions=1)
            experiment_dir = root / "exp"
            prepare_experiment(config_path=config_path, output=experiment_dir)
            with self.assertRaises(LifecycleError):
                collect_experiment(experiment_dir, require_complete=True)
            # Partial aggregate still allowed.
            summary = collect_experiment(experiment_dir, require_complete=False)
            self.assertFalse(summary["complete"])

    def test_retry_preserves_attempts_and_skips_completed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root / "cfg.toml", repetitions=2)
            experiment_dir = root / "exp"
            prepare_experiment(config_path=config_path, output=experiment_dir)

            # Force task 0 into failed after reserving attempt 001.
            task0 = experiment_dir / "tasks" / "task_000000"
            attempt = reserve_attempt(task0)
            transition_task_status(
                task0,
                expected={TaskState.PENDING},
                new_state=TaskState.DISPATCHED,
                attempt=attempt.attempt,
            )
            transition_task_status(
                task0,
                expected={TaskState.DISPATCHED},
                new_state=TaskState.RUNNING,
                attempt=attempt.attempt,
            )
            transition_task_status(
                task0,
                expected={TaskState.RUNNING},
                new_state=TaskState.FAILED,
                attempt=attempt.attempt,
                metadata={"failure_kind": "boom"},
            )

            # Complete task 1 for immutability check.
            task1 = experiment_dir / "tasks" / "task_000001"
            attempt1 = reserve_attempt(task1)
            transition_task_status(
                task1,
                expected={TaskState.PENDING},
                new_state=TaskState.DISPATCHED,
                attempt=attempt1.attempt,
            )
            transition_task_status(
                task1,
                expected={TaskState.DISPATCHED},
                new_state=TaskState.RUNNING,
                attempt=attempt1.attempt,
            )
            write_json_atomic(
                attempt1.attempt_dir / "summary.json",
                {"best_objective": 1.0, "best_x": [0]},
            )
            transition_task_status(
                task1,
                expected={TaskState.RUNNING},
                new_state=TaskState.COMPLETED,
                attempt=attempt1.attempt,
            )

            reset = retry_failed_tasks(experiment_dir)
            self.assertEqual(reset, [0])
            self.assertEqual(read_task_status(task0)["state"], "pending")
            self.assertTrue((task0 / "attempts" / "001").is_dir())
            self.assertEqual(read_task_status(task1)["state"], "completed")

            # Next dispatch must reserve attempt 002 for the retried task only.
            dispatch_experiment(experiment_dir, task_indices=[0])
            self.assertTrue((task0 / "attempts" / "002").is_dir())
            self.assertEqual(read_task_status(task0)["state"], "completed")
            self.assertEqual(read_task_status(task1)["state"], "completed")

    def test_cancel_pending(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root / "cfg.toml", repetitions=1)
            experiment_dir = root / "exp"
            prepare_experiment(config_path=config_path, output=experiment_dir)
            cancelled = cancel_tasks(experiment_dir)
            self.assertEqual(cancelled, [0])
            self.assertEqual(read_task_status(experiment_dir / "tasks" / "task_000000")["state"], "cancelled")


class TestStaleReconciliation(unittest.TestCase):
    """Owner/heartbeat reconciliation rules."""

    def _prepare_running_attempt(self, experiment_dir: Path) -> Path:
        """
        Create one running attempt directory and return it.

        Parameters
        ----------
        experiment_dir : Path
            Prepared experiment root.

        Returns
        -------
        Path
            Attempt directory.
        """
        task_dir = experiment_dir / "tasks" / "task_000000"
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
        return attempt.attempt_dir

    def test_adopt_valid_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root / "cfg.toml", repetitions=1)
            experiment_dir = root / "exp"
            prepare_experiment(config_path=config_path, output=experiment_dir)
            attempt_dir = self._prepare_running_attempt(experiment_dir)
            write_json_atomic(
                attempt_dir / "summary.json",
                {"best_objective": 3.5, "best_x": [1, 2]},
            )
            actions = reconcile_stale_tasks(experiment_dir, grace_seconds=0.0)
            self.assertEqual(actions[0]["action"], "adopted")
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                "completed",
            )

    def test_stale_heartbeat_dead_owner(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root / "cfg.toml", repetitions=1)
            experiment_dir = root / "exp"
            prepare_experiment(config_path=config_path, output=experiment_dir)
            attempt_dir = self._prepare_running_attempt(experiment_dir)
            write_attempt_owner(attempt_dir, backend="local", pid=999_999_999)
            old = datetime.now(timezone.utc) - timedelta(seconds=10_000)
            touch_attempt_heartbeat(attempt_dir, pid=999_999_999, updated_at=old)
            actions = reconcile_stale_tasks(
                experiment_dir,
                timeout_seconds=1.0,
                grace_seconds=1.0,
            )
            self.assertEqual(actions[0]["action"], "stale")
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                "failed",
            )

    def test_live_owner_preserved(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root / "cfg.toml", repetitions=1)
            experiment_dir = root / "exp"
            prepare_experiment(config_path=config_path, output=experiment_dir)
            attempt_dir = self._prepare_running_attempt(experiment_dir)
            import os

            write_attempt_owner(attempt_dir, backend="local", pid=os.getpid())
            old = datetime.now(timezone.utc) - timedelta(seconds=10_000)
            touch_attempt_heartbeat(attempt_dir, pid=os.getpid(), updated_at=old)
            actions = reconcile_stale_tasks(
                experiment_dir,
                timeout_seconds=1.0,
                grace_seconds=1.0,
            )
            self.assertEqual(actions, [])
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                "running",
            )

    def test_unknown_remote_owner_preserved(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root / "cfg.toml", repetitions=1)
            experiment_dir = root / "exp"
            prepare_experiment(config_path=config_path, output=experiment_dir)
            attempt_dir = self._prepare_running_attempt(experiment_dir)
            write_attempt_owner(
                attempt_dir,
                backend="slurm",
                pid=None,
                slurm_job_id="12345",
                slurm_task_id="0",
            )
            old = datetime.now(timezone.utc) - timedelta(seconds=10_000)
            touch_attempt_heartbeat(attempt_dir, pid=1, updated_at=old)
            actions = reconcile_stale_tasks(
                experiment_dir,
                timeout_seconds=1.0,
                grace_seconds=1.0,
            )
            self.assertEqual(actions, [])
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                "running",
            )

    def test_forced_stale_recovery(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root / "cfg.toml", repetitions=1)
            experiment_dir = root / "exp"
            prepare_experiment(config_path=config_path, output=experiment_dir)
            attempt_dir = self._prepare_running_attempt(experiment_dir)
            write_attempt_owner(
                attempt_dir,
                backend="slurm",
                pid=None,
                slurm_job_id="999",
            )
            old = datetime.now(timezone.utc) - timedelta(seconds=10_000)
            touch_attempt_heartbeat(attempt_dir, pid=1, updated_at=old)
            actions = reconcile_stale_tasks(
                experiment_dir,
                force_stale_seconds=100.0,
            )
            self.assertEqual(actions[0]["reason"], "forced_stale")
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                "failed",
            )


class TestPrepareAlgorithmValidation(unittest.TestCase):
    """ALG3.2: registry validation before manifest publication."""

    def test_unknown_algorithm_fails_before_manifest_publication(self) -> None:
        from core.ioh_core.algorithm_factory import AlgorithmFactoryError

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "cfg.toml"
            config_path.write_text(
                textwrap.dedent(
                    """\
                    schema_version = 1

                    [experiment]
                    id = "unknown_algo"
                    workflow = "optimization_runs"
                    slab_type = "hp_slab"
                    problem_ids = ["L6750_B2"]
                    output_root = "experiments"

                    [[experiment.algorithms]]
                    name = "NOT_A_REAL_ALGORITHM"
                    max_evaluations = 2

                    [experiment.repetitions]
                    count = 1
                    seed_base = 1

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
            experiment_dir = root / "exp"
            with self.assertRaises(AlgorithmFactoryError) as ctx:
                prepare_experiment(config_path=config_path, output=experiment_dir)
            self.assertIn("unknown algorithm", str(ctx.exception))
            self.assertFalse((experiment_dir / "manifest.json").is_file())
            self.assertTrue((experiment_dir / "inputs").is_dir())

    def test_unknown_algorithm_parameter_fails_before_manifest_publication(self) -> None:
        from core.ioh_core.algorithm_factory import AlgorithmFactoryError

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "cfg.toml"
            config_path.write_text(
                textwrap.dedent(
                    """\
                    schema_version = 1

                    [experiment]
                    id = "unknown_param"
                    workflow = "optimization_runs"
                    slab_type = "hp_slab"
                    problem_ids = ["L6750_B2"]
                    output_root = "experiments"

                    [[experiment.algorithms]]
                    name = "RANDOM_SEARCH"
                    max_evaluations = 2
                    [experiment.algorithms.parameters]
                    population = 10

                    [experiment.repetitions]
                    count = 1
                    seed_base = 1

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
            experiment_dir = root / "exp"
            with self.assertRaises(AlgorithmFactoryError) as ctx:
                prepare_experiment(config_path=config_path, output=experiment_dir)
            self.assertIn("unknown parameters", str(ctx.exception))
            self.assertFalse((experiment_dir / "manifest.json").is_file())

    def test_user_parameters_cannot_override_max_evaluations(self) -> None:
        from core.ioh_core.algorithm_factory import AlgorithmFactoryError

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "cfg.toml"
            config_path.write_text(
                textwrap.dedent(
                    """\
                    schema_version = 1

                    [experiment]
                    id = "budget_override"
                    workflow = "optimization_runs"
                    slab_type = "hp_slab"
                    problem_ids = ["L6750_B2"]
                    output_root = "experiments"

                    [[experiment.algorithms]]
                    name = "RANDOM_SEARCH"
                    max_evaluations = 2
                    [experiment.algorithms.parameters]
                    max_evaluations = 99

                    [experiment.repetitions]
                    count = 1
                    seed_base = 1

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
            experiment_dir = root / "exp"
            with self.assertRaises(AlgorithmFactoryError) as ctx:
                prepare_experiment(config_path=config_path, output=experiment_dir)
            self.assertIn("max_evaluations", str(ctx.exception))
            self.assertFalse((experiment_dir / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
