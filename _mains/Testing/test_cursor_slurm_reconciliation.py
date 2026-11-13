"""
Phase A10 gate: Slurm reconciliation and cancellation.
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
from core.execution_core.atomic_io import write_json_atomic
from core.execution_core.backends.slurm import SlurmArrayBackend
from core.execution_core.backends.slurm_commands import (
    RecordingSlurmCommands,
    SlurmCommandError,
)
from core.execution_core.backends.slurm_status import (
    cancel_slurm_submission,
    cancel_slurm_task,
    classify_slurm_state,
    parse_sacct_parsable,
    parse_squeue_parsable,
    reconcile_slurm_experiment,
)
from core.execution_core.config import load_experiment_config
from core.execution_core.manifest import prepare_experiment_artifacts
from core.execution_core.models import TaskState

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "slurm"


def _write_slurm_config(path: Path) -> Path:
    """
    Write a complete Slurm-backend TOML.

    Parameters
    ----------
    path : Path
        Destination path.

    Returns
    -------
    Path
        Written path.
    """
    path.write_text(
        textwrap.dedent(
            """\
            schema_version = 1

            [experiment]
            id = "slurm_recon"
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
            max_parallel_tasks = 4
            cpus_per_task = 1

            [execution.slurm]
            python_executable = "/cluster/python/bin/python"
            partition = "cpu-short"
            account = "hpdesign"
            memory = "8G"
            time_limit = "00:45:00"
            array_concurrency = 64
            maximum_array_size = 1000
            """
        ),
        encoding="utf-8",
    )
    return path


def _prepare_dispatched(root: Path, n: int, *, job_id: str = "12345") -> Path:
    """
    Prepare an experiment, dispatch via recording sbatch, return experiment dir.

    Parameters
    ----------
    root : Path
        Temporary parent.
    n : int
        Number of pending tasks.
    job_id : str, optional
        Fake sbatch job id.

    Returns
    -------
    Path
        Experiment root.
    """
    config = load_experiment_config(_write_slurm_config(root / "cfg.toml"))
    experiment_dir = root / "experiments" / "slurm_recon"
    create_experiment_layout(experiment_dir)
    prepare_experiment_artifacts(config, root=experiment_dir, tasks=())
    paths = create_experiment_layout(experiment_dir)
    for index in range(n):
        create_task_layout(paths, index)
    commands = RecordingSlurmCommands(sbatch_job_ids=[job_id])
    backend = SlurmArrayBackend(
        python_executable="/cluster/python/bin/python",
        partition="cpu-short",
        account="hpdesign",
        memory="8G",
        time_limit="00:45:00",
        commands=commands,
    )
    backend.dispatch(
        experiment_dir=experiment_dir,
        task_indices=list(range(n)),
        worker_module="core.ioh_core.workflows.optimization_runs.worker",
    )
    return experiment_dir


class TestSlurmParsing(unittest.TestCase):
    """Fixture-backed squeue/sacct parsing."""

    def test_squeue_pending_running_ignores_steps(self) -> None:
        text = (FIXTURES / "squeue_pending_running.txt").read_text(encoding="utf-8")
        records = parse_squeue_parsable(text)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].array_index, 0)
        self.assertEqual(classify_slurm_state(records[0].state), "active")
        self.assertEqual(classify_slurm_state(records[1].state), "active")

    def test_sacct_mixed_and_duplicates(self) -> None:
        text = (FIXTURES / "sacct_mixed_terminal.txt").read_text(encoding="utf-8")
        records = {r.array_index: r for r in parse_sacct_parsable(text)}
        self.assertEqual(classify_slurm_state(records[0].state), "completed")
        self.assertEqual(classify_slurm_state(records[1].state), "failed")
        self.assertEqual(classify_slurm_state(records[2].state), "failed")
        self.assertEqual(records[2].state, "OUT_OF_MEMORY")
        self.assertEqual(classify_slurm_state(records[3].state), "timed_out")
        self.assertEqual(classify_slurm_state(records[4].state), "cancelled")
        # Duplicate active row must not override terminal COMPLETED.
        self.assertEqual(records[5].state, "COMPLETED")
        # .batch / .extern ignored
        self.assertTrue(all(not r.raw_job_id.endswith(".batch") for r in records.values()))


class TestSlurmReconciliation(unittest.TestCase):
    """Scheduler → task state mapping."""

    def test_terminal_mapping_and_missing_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_dispatched(root, 6, job_id="12345")
            # Valid summary on task 0 should win over COMPLETED.
            attempt0 = experiment_dir / "tasks" / "task_000000" / "attempts" / "001"
            write_json_atomic(
                attempt0 / "summary.json",
                {"best_objective": 1.25, "best_x": [0]},
            )
            sacct = (FIXTURES / "sacct_mixed_terminal.txt").read_text(encoding="utf-8")
            commands = RecordingSlurmCommands(sacct_output=sacct, squeue_output="")
            actions = reconcile_slurm_experiment(experiment_dir, commands=commands)
            self.assertTrue(any(a.get("action") == "adopted" for a in actions))
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                "completed",
            )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000001")["state"],
                "failed",
            )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000002")["state"],
                "failed",
            )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000002")["metadata"][
                    "failure_kind"
                ],
                "out_of_memory",
            )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000003")["state"],
                "timed_out",
            )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000004")["state"],
                "cancelled",
            )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000005")["state"],
                "failed",
            )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000005")["metadata"][
                    "failure_kind"
                ],
                "missing_summary",
            )

    def test_pending_and_running_remain_active(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_dispatched(root, 3, job_id="12345")
            squeue = (FIXTURES / "squeue_pending_running.txt").read_text(encoding="utf-8")
            commands = RecordingSlurmCommands(squeue_output=squeue, sacct_output="")
            reconcile_slurm_experiment(experiment_dir, commands=commands)
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                "dispatched",
            )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000001")["state"],
                "running",
            )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000002")["state"],
                "running",
            )

    def test_boot_and_node_fail(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_dispatched(root, 2, job_id="999")
            sacct = (FIXTURES / "sacct_boot_node_fail.txt").read_text(encoding="utf-8")
            commands = RecordingSlurmCommands(sacct_output=sacct)
            reconcile_slurm_experiment(experiment_dir, commands=commands)
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["metadata"][
                    "failure_kind"
                ],
                "boot_fail",
            )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000001")["metadata"][
                    "failure_kind"
                ],
                "node_fail",
            )


class TestSlurmCancellation(unittest.TestCase):
    """scancel mapping and failure safety."""

    def test_cancel_submission_then_reconcile(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_dispatched(root, 2, job_id="555")
            commands = RecordingSlurmCommands(
                sacct_output=(
                    "JobID|State|ExitCode|Reason\n"
                    "555_0|CANCELLED|0:0|\n"
                    "555_1|CANCELLED|0:0|\n"
                )
            )
            submission_dir = experiment_dir / "submissions" / "submission_001"
            actions = cancel_slurm_submission(
                experiment_dir=experiment_dir,
                submission_dir=submission_dir,
                commands=commands,
            )
            self.assertEqual(commands.scancel_calls, [["555"]])
            self.assertEqual(len(actions), 2)
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                "cancelled",
            )

    def test_cancel_single_task_mapping(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_dispatched(root, 3, job_id="777")
            commands = RecordingSlurmCommands(
                sacct_output="JobID|State|ExitCode|Reason\n777_1|CANCELLED|0:0|\n"
            )
            cancel_slurm_task(
                experiment_dir=experiment_dir,
                task_index=1,
                commands=commands,
            )
            self.assertEqual(commands.scancel_calls, [["777_1"]])
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000001")["state"],
                "cancelled",
            )
            # Others untouched by this sacct stub.
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                "dispatched",
            )

    def test_scancel_failure_does_not_mutate(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_dispatched(root, 1, job_id="42")
            commands = RecordingSlurmCommands(scancel_error="permission denied")
            with self.assertRaises(SlurmCommandError):
                cancel_slurm_submission(
                    experiment_dir=experiment_dir,
                    submission_dir=experiment_dir / "submissions" / "submission_001",
                    commands=commands,
                )
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                "dispatched",
            )
            self.assertEqual(commands.sacct_calls, [])


if __name__ == "__main__":
    unittest.main()
