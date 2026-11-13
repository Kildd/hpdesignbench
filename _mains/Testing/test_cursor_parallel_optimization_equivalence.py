"""
Phase A11 gate: local parallel equivalence and failure isolation.

Runs the same seeded RandomSearch manifest with parallelism one and two,
compares outcomes by semantic task identity, and checks failure/retry
isolation for the real optimisation worker path.
"""
from __future__ import annotations

import csv
import math
import os
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from core.execution_core.artifacts import read_task_status
from core.execution_core.atomic_io import read_json, write_json_atomic
from core.execution_core.lifecycle import (
    collect_experiment,
    dispatch_experiment,
    list_task_statuses,
    prepare_experiment,
    retry_failed_tasks,
)
from core.execution_core.models import TaskState
from core.ioh_core.workflows.optimization_runs.aggregation import CSV_FIELDS


def _write_config(
    path: Path,
    *,
    experiment_id: str,
    max_parallel_tasks: int,
    algorithms: list[tuple[str, int]],
    repetitions: int = 2,
    seed_base: int = 42,
    termination: str = "",
) -> Path:
    """
    Write a tiny local optimisation TOML.

    Parameters
    ----------
    path : Path
        Destination TOML path.
    experiment_id : str
        Experiment id.
    max_parallel_tasks : int
        Local concurrency cap.
    algorithms : list of (name, max_evaluations)
        Algorithm entries.
    repetitions : int, optional
        Repetition count.
    seed_base : int, optional
        Seed base for stochastic algorithms.
    termination : str, optional
        Optional ``[experiment.termination]`` TOML fragment.

    Returns
    -------
    Path
        Written path.
    """
    algo_blocks: list[str] = []
    for name, budget in algorithms:
        algo_blocks.append(
            textwrap.dedent(
                f"""\
                [[experiment.algorithms]]
                name = "{name}"
                max_evaluations = {budget}
                """
            )
        )
    body = textwrap.dedent(
        f"""\
        schema_version = 1

        [experiment]
        id = "{experiment_id}"
        workflow = "optimization_runs"
        slab_type = "hp_slab"
        problem_ids = ["L6750_B2"]
        output_root = "experiments"

        {"".join(algo_blocks)}
        [experiment.repetitions]
        count = {repetitions}
        seed_base = {seed_base}

        [execution]
        backend = "local"
        max_parallel_tasks = {max_parallel_tasks}
        timeout_seconds = 600
        cpus_per_task = 1

        [execution.local]
        python_executable = ""
        """
    )
    if termination.strip():
        body = body + "\n" + termination.strip() + "\n"
    path.write_text(body, encoding="utf-8")
    return path


def _semantic_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """
    Build a finish-order-independent identity for one aggregate row.

    Parameters
    ----------
    row : dict
        Aggregate row or summary fields.

    Returns
    -------
    tuple
        ``(problem_id, algorithm_name, repetition, random_seed)``.
    """
    return (
        str(row["problem_id"]),
        str(row["algorithm_name"]),
        int(row["repetition"]),
        row.get("random_seed"),
    )


def _outcome_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """
    Extract comparable scientific fields from a task summary.

    Parameters
    ----------
    summary : Mapping[str, Any]
        Worker ``summary.json`` contents.

    Returns
    -------
    dict
        Seeds, best vector/objective, and evaluation count.
    """
    return {
        "random_seed": summary.get("random_seed"),
        "best_x": list(summary["best_x"]),
        "best_objective": float(summary["best_objective"]),
        "evaluations": int(summary["evaluations"]),
        "problem_id": summary["problem_id"],
        "algorithm_name": summary["algorithm_name"],
        "repetition": int(summary["repetition"]),
        "target_hit": summary.get("target_hit"),
        "evaluations_to_target": summary.get("evaluations_to_target"),
        "stop_reason": summary.get("stop_reason"),
    }


def _outcomes_by_identity(experiment_dir: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    """
    Load completed-task outcomes keyed by semantic identity.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root after dispatch/collect.

    Returns
    -------
    dict
        Mapping from semantic key to comparable outcome fields.
    """
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    tasks_root = experiment_dir / "tasks"
    for task_dir in sorted(tasks_root.glob("task_*")):
        status = read_task_status(task_dir)
        if status["state"] != TaskState.COMPLETED.value:
            continue
        attempt = int(status["attempt"])
        summary = read_json(task_dir / "attempts" / f"{attempt:03d}" / "summary.json")
        outcome = _outcome_from_summary(summary)
        out[_semantic_key(outcome)] = outcome
    return out


def _poison_task_problem_id(experiment_dir: Path, task_index: int, bad_id: str) -> str:
    """
    Rewrite one sealed task payload ``problem_id`` and return the old value.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.
    task_index : int
        Task to mutate.
    bad_id : str
        Invalid problem id that will fail at build time.

    Returns
    -------
    str
        Previous problem id.
    """
    manifest_path = experiment_dir / "manifest.json"
    manifest = read_json(manifest_path)
    task = manifest["tasks"][task_index]
    old = str(task["payload"]["problem_id"])
    task["payload"]["problem_id"] = bad_id
    write_json_atomic(manifest_path, manifest)
    return old


def _restore_task_problem_id(experiment_dir: Path, task_index: int, problem_id: str) -> None:
    """
    Restore one sealed task payload ``problem_id``.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root.
    task_index : int
        Task index.
    problem_id : str
        Value to restore.
    """
    manifest_path = experiment_dir / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["tasks"][task_index]["payload"]["problem_id"] = problem_id
    write_json_atomic(manifest_path, manifest)


class TestParallelOptimizationEquivalence(unittest.TestCase):
    """Seeded RandomSearch parity across local concurrency settings."""

    def test_parallelism_one_vs_two_semantic_equivalence(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            algorithms = [("RANDOM_SEARCH", 3)]
            outcomes: dict[int, dict[tuple[Any, ...], dict[str, Any]]] = {}
            for parallel in (1, 2):
                exp_dir = root / f"eq_p{parallel}"
                config_path = _write_config(
                    root / f"eq_p{parallel}.toml",
                    experiment_id=f"eq_p{parallel}",
                    max_parallel_tasks=parallel,
                    algorithms=algorithms,
                    repetitions=2,
                    seed_base=42,
                )
                prepare_experiment(config_path=config_path, output=exp_dir)
                dispatch_experiment(exp_dir)
                status = list_task_statuses(exp_dir)
                self.assertTrue(status.complete)
                self.assertEqual(status.counts["completed"], 2)
                collect_experiment(exp_dir, require_complete=True)
                outcomes[parallel] = _outcomes_by_identity(exp_dir)

            self.assertEqual(set(outcomes[1]), set(outcomes[2]))
            for key in outcomes[1]:
                left = outcomes[1][key]
                right = outcomes[2][key]
                self.assertEqual(left["random_seed"], right["random_seed"])
                self.assertEqual(left["best_x"], right["best_x"])
                self.assertAlmostEqual(
                    left["best_objective"],
                    right["best_objective"],
                    places=12,
                )
                self.assertEqual(left["evaluations"], right["evaluations"])
                self.assertEqual(left["evaluations"], 3)

    def test_fixed_target_parallelism_one_vs_two_equivalence(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            termination = textwrap.dedent(
                """\
                [experiment.termination]
                mode = "fixed_target"
                global_optimum = 1.0e20
                allowed_deviation_percent = 0.0
                target_reference = "all"
                """
            )
            outcomes: dict[int, dict[tuple[Any, ...], dict[str, Any]]] = {}
            for parallel in (1, 2):
                exp_dir = root / f"ft_p{parallel}"
                config_path = _write_config(
                    root / f"ft_p{parallel}.toml",
                    experiment_id=f"ft_p{parallel}",
                    max_parallel_tasks=parallel,
                    algorithms=[("RANDOM_SEARCH", 5)],
                    repetitions=2,
                    seed_base=42,
                    termination=termination,
                )
                prepare_experiment(config_path=config_path, output=exp_dir)
                dispatch_experiment(exp_dir)
                status = list_task_statuses(exp_dir)
                self.assertTrue(status.complete)
                self.assertEqual(status.counts["completed"], 2)
                collect_experiment(exp_dir, require_complete=True)
                outcomes[parallel] = _outcomes_by_identity(exp_dir)

            self.assertEqual(set(outcomes[1]), set(outcomes[2]))
            for key in outcomes[1]:
                left = outcomes[1][key]
                right = outcomes[2][key]
                self.assertEqual(left["target_hit"], right["target_hit"])
                self.assertEqual(
                    left["evaluations_to_target"],
                    right["evaluations_to_target"],
                )
                self.assertEqual(left["stop_reason"], right["stop_reason"])
                self.assertEqual(left["best_x"], right["best_x"])
                self.assertAlmostEqual(
                    left["best_objective"],
                    right["best_objective"],
                    places=12,
                )
                self.assertTrue(left["target_hit"])
                self.assertEqual(left["stop_reason"], "target_hit")
                self.assertEqual(left["evaluations"], 1)

    def test_aggregate_order_independent_of_finish_order(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "agg_order"
            prepare_experiment(
                config_path=_write_config(
                    root / "agg.toml",
                    experiment_id="agg_order",
                    max_parallel_tasks=2,
                    algorithms=[("RANDOM_SEARCH", 2)],
                    repetitions=3,
                    seed_base=11,
                ),
                output=exp_dir,
            )
            dispatch_experiment(exp_dir)
            status = list_task_statuses(exp_dir)
            self.assertTrue(status.complete)
            self.assertEqual(status.counts["completed"], 3)
            self.assertEqual(status.counts["failed"], 0)
            summary = collect_experiment(exp_dir, require_complete=True)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["counts"]["completed"], 3)

            csv_path = exp_dir / "aggregate" / "tasks.csv"
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(CSV_FIELDS))
                rows = list(reader)
            self.assertEqual([int(r["task_index"]) for r in rows], [0, 1, 2])
            self.assertTrue(all(r["state"] == "completed" for r in rows))
            self.assertTrue(all(r["best_objective"] != "" for r in rows))
            # Semantic identity still unique and seeds present.
            keys = [
                (
                    r["problem_id"],
                    r["algorithm_name"],
                    int(r["repetition"]),
                    int(r["random_seed"]),
                )
                for r in rows
            ]
            self.assertEqual(len(keys), len(set(keys)))


class TestWorkerIsolationAndRetry(unittest.TestCase):
    """CWD independence, IOH isolation, failure isolation, and retry scope."""

    def test_cwd_independence_and_private_ioh_folders(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "cwd_iso"
            prepare_experiment(
                config_path=_write_config(
                    root / "cwd.toml",
                    experiment_id="cwd_iso",
                    max_parallel_tasks=2,
                    algorithms=[("RANDOM_SEARCH", 2)],
                    repetitions=2,
                    seed_base=9,
                ),
                output=exp_dir,
            )
            other = root / "unrelated_cwd"
            other.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(other)
                dispatch_experiment(exp_dir)
            finally:
                os.chdir(previous)

            status = list_task_statuses(exp_dir)
            self.assertTrue(status.complete)
            collect_experiment(exp_dir, require_complete=True)

            ioh_dirs: list[Path] = []
            for index in (0, 1):
                attempt_dir = exp_dir / "tasks" / f"task_{index:06d}" / "attempts" / "001"
                ioh_dir = attempt_dir / "ioh"
                self.assertTrue(ioh_dir.is_dir())
                self.assertTrue(list(ioh_dir.rglob("IOHprofiler_f*.json")))
                ioh_dirs.append(ioh_dir.resolve())
                # Worker must not have written under the foreign CWD.
                self.assertFalse(any(other.iterdir()))
            self.assertEqual(len(ioh_dirs), len(set(ioh_dirs)))
            # Folders must not share the same resolved tree.
            self.assertNotEqual(ioh_dirs[0], ioh_dirs[1])

    def test_failed_task_does_not_stop_others_and_retry_is_scoped(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "fail_iso"
            prepare_experiment(
                config_path=_write_config(
                    root / "fail.toml",
                    experiment_id="fail_iso",
                    max_parallel_tasks=2,
                    algorithms=[("RANDOM_SEARCH", 3)],
                    repetitions=2,
                    seed_base=21,
                ),
                output=exp_dir,
            )
            original_problem = _poison_task_problem_id(exp_dir, 0, "missing_problem_xyz")
            dispatch_experiment(exp_dir)

            status = list_task_statuses(exp_dir)
            self.assertEqual(status.counts["failed"], 1)
            self.assertEqual(status.counts["completed"], 1)
            self.assertEqual(
                read_task_status(exp_dir / "tasks" / "task_000000")["state"],
                TaskState.FAILED.value,
            )
            self.assertEqual(
                read_task_status(exp_dir / "tasks" / "task_000001")["state"],
                TaskState.COMPLETED.value,
            )

            attempt0 = exp_dir / "tasks" / "task_000000" / "attempts" / "001"
            attempt1 = exp_dir / "tasks" / "task_000001" / "attempts" / "001"
            self.assertTrue((attempt0 / "failure.json").is_file())
            self.assertFalse((attempt0 / "summary.json").exists())
            ok_summary = read_json(attempt1 / "summary.json")
            self.assertEqual(ok_summary["evaluations"], 3)

            # Restore a valid payload, then retry only the failed task.
            _restore_task_problem_id(exp_dir, 0, original_problem)
            reset = retry_failed_tasks(exp_dir)
            self.assertEqual(reset, [0])
            self.assertEqual(
                read_task_status(exp_dir / "tasks" / "task_000000")["state"],
                TaskState.PENDING.value,
            )
            self.assertEqual(
                read_task_status(exp_dir / "tasks" / "task_000001")["state"],
                TaskState.COMPLETED.value,
            )
            # Previous failed attempt directory remains untouched.
            self.assertTrue((attempt0 / "failure.json").is_file())

            dispatch_experiment(exp_dir, task_indices=[0])
            self.assertEqual(
                read_task_status(exp_dir / "tasks" / "task_000000")["state"],
                TaskState.COMPLETED.value,
            )
            self.assertTrue(
                (exp_dir / "tasks" / "task_000000" / "attempts" / "002" / "summary.json").is_file()
            )
            # Successful sibling attempt artifacts are immutable.
            self.assertEqual(read_json(attempt1 / "summary.json"), ok_summary)
            self.assertFalse(
                (exp_dir / "tasks" / "task_000001" / "attempts" / "002").exists()
            )

            summary = collect_experiment(exp_dir, require_complete=True)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["counts"]["completed"], 2)


class TestDirectLInvariants(unittest.TestCase):
    """DIRECT-L finite/invariant smoke (no cross-platform bitwise claim)."""

    def test_direct_l_finite_outputs_and_budget(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "direct"
            prepare_experiment(
                config_path=_write_config(
                    root / "direct.toml",
                    experiment_id="direct",
                    max_parallel_tasks=1,
                    algorithms=[("NLOPT_DIRECT_L", 5)],
                    repetitions=1,
                    seed_base=1,
                ),
                output=exp_dir,
            )
            dispatch_experiment(exp_dir)
            status = list_task_statuses(exp_dir)
            self.assertTrue(status.complete)
            self.assertEqual(status.counts.get("failed", 0), 0)
            summary = read_json(
                exp_dir / "tasks" / "task_000000" / "attempts" / "001" / "summary.json"
            )
            self.assertEqual(summary["algorithm_name"], "NLOPT_DIRECT_L")
            self.assertIsNone(summary["random_seed"])
            self.assertTrue(summary["finite_best"])
            self.assertTrue(math.isfinite(float(summary["best_objective"])))
            self.assertGreaterEqual(int(summary["evaluations"]), 1)
            self.assertLessEqual(int(summary["evaluations"]), 5)
            self.assertIsInstance(summary["best_x"], list)
            self.assertTrue(summary["best_x"])
            self.assertTrue(all(isinstance(v, int) for v in summary["best_x"]))
            # Manifest still records a null seed for the deterministic method.
            manifest = read_json(exp_dir / "manifest.json")
            self.assertIsNone(manifest["tasks"][0]["payload"]["random_seed"])


if __name__ == "__main__":
    unittest.main()
