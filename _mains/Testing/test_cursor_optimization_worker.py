"""
Phase A6 gate: isolated logged optimisation and worker contract.
"""
from __future__ import annotations

import json
import os
import shutil
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from core.execution_core.artifacts import (
    create_experiment_layout,
    create_task_layout,
    read_task_status,
    reserve_attempt,
    transition_task_status,
)
from core.execution_core.atomic_io import read_json
from core.execution_core.config import load_experiment_config
from core.execution_core.manifest import prepare_experiment_artifacts
from core.execution_core.models import TaskState
from core.ioh_core.algorithm_factory import create_algorithm
from core.ioh_core.experiment import SUITE_NAME, run_experiment
from core.ioh_core.problem_builder import build_problem_from_snapshot, build_problems_for_slab_type
from core.ioh_core.workflows.optimization_runs.io_logging import run_logged_optimization
from core.ioh_core.workflows.optimization_runs.planner import plan_optimization_tasks
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
    snapshot_slab_inputs,
)
from core.ioh_core.workflows.optimization_runs.worker import main as worker_main


def _write_smoke_config(path: Path) -> Path:
    """
    Write a tiny local optimisation TOML for worker tests.

    Parameters
    ----------
    path : Path
        Destination TOML path.

    Returns
    -------
    Path
        The written path.
    """
    path.write_text(
        textwrap.dedent(
            """\
            schema_version = 1

            [experiment]
            id = "worker_smoke"
            workflow = "optimization_runs"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [[experiment.algorithms]]
            name = "RANDOM_SEARCH"
            max_evaluations = 3

            [experiment.repetitions]
            count = 1
            seed_base = 42

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


def _prepare_experiment(root: Path) -> tuple[Path, object]:
    """
    Prepare a sealed experiment with one RANDOM_SEARCH task.

    Parameters
    ----------
    root : Path
        Temporary directory used as parent for the experiment root.

    Returns
    -------
    tuple
        ``(experiment_dir, config)``.
    """
    config_path = _write_smoke_config(root / "smoke.toml")
    config = load_experiment_config(config_path)
    experiment_dir = root / "experiments" / config.experiment.id
    create_experiment_layout(experiment_dir)
    hashes = snapshot_slab_inputs(
        slab_type=config.experiment.slab_type,
        inputs_dir=experiment_dir / "inputs",
    )
    tasks = plan_optimization_tasks(config)
    paths, _manifest, _env = prepare_experiment_artifacts(
        config,
        root=experiment_dir,
        tasks=tasks,
        input_files=hashes,
    )
    for task in tasks:
        create_task_layout(paths, task.task_index)
    return experiment_dir, config


def _dispatch_attempt(experiment_dir: Path, task_index: int = 0):
    """
    Reserve an attempt and mark the task dispatched.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.
    task_index : int, optional
        Task index to claim.

    Returns
    -------
    AttemptRef
        Reserved attempt reference.
    """
    from core.ioh_core.workflows.optimization_runs.worker import (
        experiment_paths,
        task_directory,
    )

    paths = experiment_paths(experiment_dir)
    task_dir = task_directory(paths, task_index)
    attempt = reserve_attempt(task_dir)
    transition_task_status(
        task_dir,
        expected={TaskState.PENDING},
        new_state=TaskState.DISPATCHED,
        attempt=attempt.attempt,
    )
    return attempt


class TestRunLoggedOptimization(unittest.TestCase):
    """Private IOH logging helper behaviour."""

    def test_deb_mlsl_persists_algorithm_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "ioh_out"
            bundle = build_problems_for_slab_type("hp_slab")["L6750_B2"]
            algorithm = create_algorithm(
                "DISCRETE_MLSL_DEB_SI_DESC_COLD_START",
                {"max_evaluations": 1, "batch_size": 1},
                random_seed=123,
            )
            summary = run_logged_optimization(
                bundle=bundle,
                algorithm=algorithm,
                output_directory=out,
            )
            diagnostics = summary["algorithm_diagnostics"]
            self.assertEqual(diagnostics["label"], "dMLSL-L1-Deb")
            self.assertEqual(diagnostics["n_evaluations"], 1)
            self.assertEqual(
                diagnostics["n_evaluations"],
                diagnostics["n_failed_analysis_evaluations"]
                + diagnostics["n_feasible_evaluations"]
                + diagnostics["n_infeasible_evaluations"],
            )
            names = bundle["active_constraint_names"]
            self.assertEqual(
                set(diagnostics["sentinel_evaluations_by_constraint"]),
                set(names),
            )
            self.assertEqual(len(names), 11)
            self.assertIn("Z1_nt_dt_combination_capacity", names)
            self.assertIn("Z2_beam_theory_H_L_capacity", names)
            self.assertIn("Z3_beam_theory_B_L_capacity", names)

    def test_historical_mlsl_omits_algorithm_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "ioh_out"
            bundle = build_problems_for_slab_type("hp_slab")["L6750_B2"]
            algorithm = create_algorithm(
                "DISCRETE_MLSL_PEN_SI_DESC_COLD_START",
                {"max_evaluations": 1, "batch_size": 1},
                random_seed=123,
            )
            summary = run_logged_optimization(
                bundle=bundle,
                algorithm=algorithm,
                output_directory=out,
            )
            self.assertNotIn("algorithm_diagnostics", summary)

    def test_deb_pbh_persists_algorithm_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "ioh_out"
            bundle = build_problems_for_slab_type("hp_slab")["L6750_B2"]
            algorithm = create_algorithm(
                "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
                {"max_evaluations": 1, "population_size": 1},
                random_seed=123,
            )

            summary = run_logged_optimization(
                bundle=bundle,
                algorithm=algorithm,
                output_directory=out,
            )

            diagnostics = summary["algorithm_diagnostics"]
            self.assertEqual(diagnostics["label"], "PBH-L1-Deb")
            self.assertEqual(diagnostics["n_evaluations"], 1)
            self.assertEqual(
                diagnostics["n_evaluations"],
                diagnostics["n_failed_analysis_evaluations"]
                + diagnostics["n_feasible_evaluations"]
                + diagnostics["n_infeasible_evaluations"],
            )

    def test_deb_ams_persists_algorithm_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "ioh_out"
            bundle = build_problems_for_slab_type("hp_slab")["L6750_B2"]
            algorithm = create_algorithm(
                "K_ELITE_AMS_DEB_SI_DESC_COLD_START",
                {"max_evaluations": 1, "n_parents": 1, "archive_size": 1},
                random_seed=123,
            )
            summary = run_logged_optimization(
                bundle=bundle,
                algorithm=algorithm,
                output_directory=out,
            )
            diagnostics = summary["algorithm_diagnostics"]
            self.assertEqual(diagnostics["label"], "k-elite AMS-Deb")
            self.assertNotIn("final_archive", diagnostics)
            self.assertEqual(diagnostics["n_evaluations"], 1)
            names = bundle["active_constraint_names"]
            self.assertEqual(
                set(diagnostics["sentinel_evaluations_by_constraint"]),
                set(names),
            )
            self.assertIn("Z1_nt_dt_combination_capacity", names)
            self.assertIn("Z2_beam_theory_H_L_capacity", names)
            self.assertIn("Z3_beam_theory_B_L_capacity", names)
            self.assertEqual(len(names), 11)

    def test_historical_ams_omits_algorithm_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "ioh_out"
            bundle = build_problems_for_slab_type("hp_slab")["L6750_B2"]
            algorithm = create_algorithm(
                "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
                {"max_evaluations": 1, "n_parents": 1, "archive_size": 1},
                random_seed=123,
            )
            summary = run_logged_optimization(
                bundle=bundle,
                algorithm=algorithm,
                output_directory=out,
            )
            self.assertNotIn("algorithm_diagnostics", summary)

    def test_tiny_seeded_run_summary_and_suite_patch(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "ioh_out"
            bundle = build_problems_for_slab_type("hp_slab")["L6750_B2"]
            algorithm = create_algorithm(
                "RANDOM_SEARCH",
                {"max_evaluations": 3},
                random_seed=123,
            )
            summary = run_logged_optimization(
                bundle=bundle,
                algorithm=algorithm,
                output_directory=out,
            )
            required = {
                "best_objective",
                "best_x",
                "best_params",
                "hits",
                "misses",
                "evaluations",
                "wall_time_s",
                "finite_best",
                "algorithm_name",
                "termination_mode",
                "global_optimum",
                "allowed_deviation_percent",
                "target_threshold",
                "target_hit",
                "evaluations_to_target",
                "budget_exhausted",
                "stop_reason",
            }
            self.assertTrue(required.issubset(summary))
            self.assertEqual(summary["evaluations"], 3)
            self.assertEqual(summary["algorithm_name"], "RANDOM_SEARCH")
            self.assertTrue(summary["finite_best"])
            self.assertEqual(len(summary["best_x"]), len(bundle["var_names"]))
            self.assertEqual(summary["termination_mode"], "fixed_budget")
            self.assertIsNone(summary["global_optimum"])
            self.assertIsNone(summary["allowed_deviation_percent"])
            self.assertIsNone(summary["target_threshold"])
            self.assertFalse(summary["target_hit"])
            self.assertIsNone(summary["evaluations_to_target"])
            self.assertTrue(summary["budget_exhausted"])
            self.assertEqual(summary["stop_reason"], "budget_exhausted")
            self.assertNotIn("algorithm_diagnostics", summary)

            jsons = list(out.rglob("IOHprofiler_f*.json"))
            self.assertTrue(jsons)
            for jf in jsons:
                data = json.loads(jf.read_text(encoding="utf-8"))
                self.assertEqual(data.get("suite"), SUITE_NAME)
                self.assertIn("t_eval_s", data.get("attributes", []))

            dats = list(out.rglob("*.dat"))
            self.assertTrue(dats)
            header = dats[0].read_text(encoding="utf-8").splitlines()[0].split()
            self.assertGreaterEqual(len(header), 3)
            self.assertEqual(header[0], "evaluations")
            self.assertEqual(header[1], "raw_y")
            self.assertEqual(header[2], "t_eval_s")
            t_idx = header.index("t_eval_s")
            times: list[float] = []
            for line in dats[0].read_text(encoding="utf-8").splitlines()[1:]:
                if not line.strip():
                    continue
                parts = line.split()
                times.append(float(parts[t_idx]))
            self.assertGreaterEqual(len(times), 3)
            self.assertGreaterEqual(times[0], 0.0)
            for earlier, later in zip(times, times[1:]):
                self.assertGreaterEqual(later, earlier)

    def test_ioh_logger_writes_through_non_ascii_destination(self) -> None:
        """IOH C++ logger is staged when the destination path has umlauts."""
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "Universität" / "ioh_out"
            bundle = build_problems_for_slab_type("hp_slab")["L6750_B2"]
            algorithm = create_algorithm(
                "RANDOM_SEARCH",
                {"max_evaluations": 3},
                random_seed=7,
            )
            summary = run_logged_optimization(
                bundle=bundle,
                algorithm=algorithm,
                output_directory=out,
            )
            self.assertEqual(summary["evaluations"], 3)
            self.assertEqual(summary["output_directory"], str(out.resolve()))
            jsons = list(out.rglob("IOHprofiler_f*.json"))
            self.assertTrue(
                jsons,
                "expected IOHprofiler JSON under non-ASCII destination",
            )
            dats = list(out.rglob("*.dat"))
            self.assertTrue(dats, "expected per-evaluation .dat traces")
            for jf in jsons:
                data = json.loads(jf.read_text(encoding="utf-8"))
                self.assertEqual(data.get("suite"), SUITE_NAME)

    def test_same_seed_reproducibility_and_ioh_sourced_counts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_hashes = snapshot_slab_inputs(
                slab_type="hp_slab",
                inputs_dir=root / "inputs",
            )
            self.assertTrue(snap_hashes)
            snapshot_dir = resolve_snapshot_slab_dir(root / "inputs", "hp_slab")

            summaries = []
            for name in ("a", "b"):
                bundle = build_problem_from_snapshot(
                    slab_type="hp_slab",
                    problem_id="L6750_B2",
                    snapshot_dir=snapshot_dir,
                )
                algorithm = create_algorithm(
                    "RANDOM_SEARCH",
                    {"max_evaluations": 4},
                    random_seed=7,
                )
                summaries.append(
                    run_logged_optimization(
                        bundle=bundle,
                        algorithm=algorithm,
                        output_directory=root / name,
                    )
                )
            self.assertEqual(summaries[0]["best_x"], summaries[1]["best_x"])
            self.assertAlmostEqual(
                summaries[0]["best_objective"],
                summaries[1]["best_objective"],
            )
            self.assertEqual(summaries[0]["evaluations"], 4)
            self.assertEqual(summaries[0]["evaluations"], summaries[1]["evaluations"])

            # DIRECT-L also reports evaluations from IOH state.
            direct_bundle = build_problem_from_snapshot(
                slab_type="hp_slab",
                problem_id="L6750_B2",
                snapshot_dir=snapshot_dir,
            )
            direct = create_algorithm(
                "NLOPT_DIRECT_L",
                {"max_evaluations": 5},
                random_seed=None,
            )
            direct_summary = run_logged_optimization(
                bundle=direct_bundle,
                algorithm=direct,
                output_directory=root / "direct",
            )
            self.assertGreaterEqual(direct_summary["evaluations"], 1)
            self.assertIn("best_objective", direct_summary)
            self.assertEqual(direct_summary["termination_mode"], "fixed_budget")
            self.assertEqual(direct_summary["stop_reason"], "budget_exhausted")


class TestLoggedOptimizationTermination(unittest.TestCase):
    """C4.7: fixed-target outcomes recorded by run_logged_optimization."""

    def test_target_hit_stops_and_logs_each_evaluation(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "ioh_hit"
            bundle = build_problems_for_slab_type("hp_slab")["L6750_B2"]
            algorithm = create_algorithm(
                "RANDOM_SEARCH",
                {"max_evaluations": 8},
                random_seed=7,
            )
            summary = run_logged_optimization(
                bundle=bundle,
                algorithm=algorithm,
                output_directory=out,
                max_evaluations=8,
                termination_mode="fixed_target",
                global_optimum=1.0e20,
                allowed_deviation_percent=0.0,
                target_reference="all",
                target_threshold=1.0e20,
            )
            self.assertTrue(summary["target_hit"])
            self.assertEqual(summary["stop_reason"], "target_hit")
            self.assertEqual(summary["evaluations_to_target"], 1)
            self.assertEqual(summary["evaluations"], 1)
            self.assertLessEqual(summary["best_objective"], 1.0e20)
            dats = list(out.rglob("*.dat"))
            self.assertTrue(dats)
            rows = [
                line
                for line in dats[0].read_text(encoding="utf-8").splitlines()[1:]
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)

    def test_target_miss_uses_hard_budget(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "ioh_miss"
            bundle = build_problems_for_slab_type("hp_slab")["L6750_B2"]
            algorithm = create_algorithm(
                "RANDOM_SEARCH",
                {"max_evaluations": 4},
                random_seed=11,
            )
            summary = run_logged_optimization(
                bundle=bundle,
                algorithm=algorithm,
                output_directory=out,
                max_evaluations=4,
                termination_mode="fixed_target",
                global_optimum=0.0,
                allowed_deviation_percent=0.0,
                target_reference="all",
                target_threshold=0.0,
            )
            self.assertFalse(summary["target_hit"])
            self.assertIsNone(summary["evaluations_to_target"])
            self.assertEqual(summary["evaluations"], 4)
            self.assertTrue(summary["budget_exhausted"])
            self.assertEqual(summary["stop_reason"], "budget_exhausted")

    def test_target_hit_exactly_at_budget(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "ioh_cap_hit"
            bundle = build_problems_for_slab_type("hp_slab")["L6750_B2"]
            algorithm = create_algorithm(
                "RANDOM_SEARCH",
                {"max_evaluations": 1},
                random_seed=3,
            )
            summary = run_logged_optimization(
                bundle=bundle,
                algorithm=algorithm,
                output_directory=out,
                max_evaluations=1,
                termination_mode="fixed_target",
                global_optimum=1.0e20,
                allowed_deviation_percent=0.0,
                target_reference="all",
                target_threshold=1.0e20,
            )
            self.assertTrue(summary["target_hit"])
            self.assertTrue(summary["budget_exhausted"])
            self.assertEqual(summary["evaluations"], 1)
            self.assertEqual(summary["evaluations_to_target"], 1)
            self.assertEqual(summary["stop_reason"], "target_hit")

    def test_non_finite_observations_never_hit(self) -> None:
        from core.ioh_core.algorithms.common import RunController
        from core.ioh_core.workflows.optimization_runs.io_logging import (
            summarize_termination,
        )
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0], [0])
        controller = RunController(
            problem,
            3,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        for _ in range(3):
            controller.require_evaluation()
            problem.state.evaluations += 1
            controller.observe_evaluation(float("inf"))
        fields = summarize_termination(
            controller,
            global_optimum=0.0,
            allowed_deviation_percent=0.0,
        )
        self.assertFalse(fields["target_hit"])
        self.assertIsNone(fields["evaluations_to_target"])
        self.assertTrue(fields["budget_exhausted"])
        self.assertEqual(fields["stop_reason"], "budget_exhausted")

    def test_controller_summary_mismatch_rejected(self) -> None:
        from core.ioh_core.workflows.optimization_runs.io_logging import (
            validate_termination_summary,
        )

        summary = {
            "termination_mode": "fixed_target",
            "global_optimum": 10.0,
            "allowed_deviation_percent": 0.0,
            "target_threshold": 10.0,
            "target_hit": True,
            "evaluations_to_target": 1,
            "budget_exhausted": False,
            "stop_reason": "target_hit",
            "evaluations": 1,
            "best_objective": 5.0,
        }
        with self.assertRaises(ValueError):
            validate_termination_summary(
                summary,
                sealed_termination_mode="fixed_target",
                sealed_global_optimum=10.0,
                sealed_allowed_deviation_percent=0.0,
                sealed_target_threshold=9.0,
                max_evaluations=5,
            )
        with self.assertRaises(ValueError):
            validate_termination_summary(
                {
                    **summary,
                    "best_objective": 11.0,
                },
                sealed_termination_mode="fixed_target",
                sealed_global_optimum=10.0,
                sealed_allowed_deviation_percent=0.0,
                sealed_target_threshold=10.0,
                max_evaluations=5,
            )

    def test_logger_cleanup_after_cooperative_target_stop(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "ioh_cleanup"
            bundle = build_problems_for_slab_type("hp_slab")["L6750_B2"]
            problem = bundle["problem"]
            algorithm = create_algorithm(
                "RANDOM_SEARCH",
                {"max_evaluations": 5},
                random_seed=1,
            )
            summary = run_logged_optimization(
                bundle=bundle,
                algorithm=algorithm,
                output_directory=out,
                max_evaluations=5,
                termination_mode="fixed_target",
                global_optimum=1.0e20,
                allowed_deviation_percent=0.0,
                target_reference="all",
                target_threshold=1.0e20,
            )
            self.assertEqual(summary["stop_reason"], "target_hit")
            from core.ioh_core.algorithms.common import _EVAL_CONTEXT_BY_PROBLEM_ID

            self.assertNotIn(id(problem), _EVAL_CONTEXT_BY_PROBLEM_ID)


class TestOptimizationWorker(unittest.TestCase):
    """CLI worker ownership, status, and failure behaviour."""

    def test_cwd_independence_and_complete_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir, _ = _prepare_experiment(root)
            attempt = _dispatch_attempt(experiment_dir)
            other = root / "other_cwd"
            other.mkdir()
            old = Path.cwd()
            try:
                os.chdir(other)
                code = worker_main(
                    [
                        "--experiment-dir",
                        str(experiment_dir),
                        "--task-index",
                        "0",
                        "--attempt",
                        str(attempt.attempt),
                    ]
                )
            finally:
                os.chdir(old)
            self.assertEqual(code, 0)
            summary = read_json(attempt.attempt_dir / "summary.json")
            self.assertEqual(summary["task_index"], 0)
            self.assertEqual(summary["evaluations"], 3)
            self.assertEqual(summary["termination_mode"], "fixed_budget")
            self.assertFalse(summary["target_hit"])
            self.assertEqual(summary["stop_reason"], "budget_exhausted")
            self.assertIsNone(summary["target_reference"])
            self.assertTrue((attempt.attempt_dir / "ioh").is_dir())
            status = read_task_status(attempt.task_dir)
            self.assertEqual(status["state"], TaskState.COMPLETED.value)

    def test_old_payload_without_termination_keys_still_runs(self) -> None:
        from core.execution_core.atomic_io import write_json_atomic
        from core.execution_core.manifest import load_manifest, manifest_to_dict

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir, _ = _prepare_experiment(root)
            paths = experiment_dir
            manifest_path = paths / "manifest.json"
            manifest = load_manifest(manifest_path)
            payload = dict(manifest.tasks[0].payload)
            for key in (
                "termination_mode",
                "global_optimum",
                "allowed_deviation_percent",
                "target_reference",
                "target_threshold",
            ):
                payload.pop(key, None)
            data = manifest_to_dict(manifest)
            data["tasks"][0]["payload"] = payload
            write_json_atomic(manifest_path, data)

            attempt = _dispatch_attempt(experiment_dir)
            code = worker_main(
                [
                    "--experiment-dir",
                    str(experiment_dir),
                    "--task-index",
                    "0",
                    "--attempt",
                    str(attempt.attempt),
                ]
            )
            self.assertEqual(code, 0)
            summary = read_json(attempt.attempt_dir / "summary.json")
            self.assertEqual(summary["termination_mode"], "fixed_budget")
            self.assertFalse(summary["target_hit"])
            self.assertEqual(summary["stop_reason"], "budget_exhausted")
            self.assertIsNone(summary["global_optimum"])
            self.assertIsNone(summary["target_threshold"])
            self.assertTrue((attempt.attempt_dir / "ioh").is_dir())
            status = read_task_status(attempt.task_dir)
            self.assertEqual(status["state"], TaskState.COMPLETED.value)

    def test_previous_attempt_unchanged_and_failure_record(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir, _ = _prepare_experiment(root)
            first = _dispatch_attempt(experiment_dir)
            self.assertEqual(
                worker_main(
                    [
                        "--experiment-dir",
                        str(experiment_dir),
                        "--task-index",
                        "0",
                        "--attempt",
                        str(first.attempt),
                    ]
                ),
                0,
            )
            first_summary = read_json(first.attempt_dir / "summary.json")

            # Fresh experiment for an infrastructure failure after running starts.
            fail_root = root / "fail_case"
            fail_root.mkdir()
            experiment_dir2, _ = _prepare_experiment(fail_root)
            attempt = _dispatch_attempt(experiment_dir2)
            with patch(
                "core.ioh_core.workflows.optimization_runs.worker.build_problem_from_snapshot",
                side_effect=RuntimeError("boom"),
            ):
                code = worker_main(
                    [
                        "--experiment-dir",
                        str(experiment_dir2),
                        "--task-index",
                        "0",
                        "--attempt",
                        str(attempt.attempt),
                    ]
                )
            self.assertEqual(code, 1)
            failure = read_json(attempt.attempt_dir / "failure.json")
            self.assertEqual(failure["error_type"], "RuntimeError")
            self.assertIn("boom", failure["message"])
            self.assertIn("Traceback", failure["traceback"])
            self.assertFalse((attempt.attempt_dir / "summary.json").exists())
            status = read_task_status(attempt.task_dir)
            self.assertEqual(status["state"], TaskState.FAILED.value)

            # First successful attempt artifacts remain untouched.
            self.assertEqual(
                read_json(first.attempt_dir / "summary.json"),
                first_summary,
            )

    def test_invalid_task_isolation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir, _ = _prepare_experiment(root)
            attempt = _dispatch_attempt(experiment_dir)
            code = worker_main(
                [
                    "--experiment-dir",
                    str(experiment_dir),
                    "--task-index",
                    "99",
                    "--attempt",
                    str(attempt.attempt),
                ]
            )
            self.assertEqual(code, 1)
            # Ownership attempt for task 0 remains pending/dispatched, not completed.
            status = read_task_status(attempt.task_dir)
            self.assertEqual(status["state"], TaskState.DISPATCHED.value)
            self.assertFalse((attempt.attempt_dir / "summary.json").exists())

    def test_worker_constructs_new_registry_algorithm_without_branch(self) -> None:
        import inspect

        from core.ioh_core.algorithms.registry import (
            AlgorithmSpec,
            BudgetSemantics,
            CandidateMode,
            ParameterSpec,
            SeedPolicy,
            create_registry,
        )
        import core.ioh_core.workflows.optimization_runs.worker as worker_mod

        algorithm_name = "TOY_WORKER_ALGO"
        self.assertNotIn(algorithm_name, inspect.getsource(worker_mod))
        self.assertNotIn(algorithm_name, inspect.getsource(create_algorithm))

        toy_registry = create_registry(
            (
                AlgorithmSpec(
                    name=algorithm_name,
                    library="native",
                    constructor_ref=(
                        "core.ioh_core.algorithms.native.factory:build_random_search"
                    ),
                    parameters=(
                        ParameterSpec(
                            name="max_evaluations",
                            types=(int,),
                            required=True,
                            minimum=1,
                        ),
                    ),
                    seed_policy=SeedPolicy.REQUIRED,
                    dependency_extra=None,
                    candidate_mode=CandidateMode.INTEGER,
                    budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
                    description="toy algorithm proving registry-driven workers",
                ),
            )
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "toy.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""\
                    schema_version = 1

                    [experiment]
                    id = "worker_toy"
                    workflow = "optimization_runs"
                    slab_type = "hp_slab"
                    problem_ids = ["L6750_B2"]
                    output_root = "experiments"

                    [[experiment.algorithms]]
                    name = "{algorithm_name}"
                    max_evaluations = 3

                    [experiment.repetitions]
                    count = 1
                    seed_base = 42

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
            with patch(
                "core.ioh_core.algorithms.registry.ALGORITHM_REGISTRY",
                toy_registry,
            ), patch(
                "core.ioh_core.algorithm_factory.ALGORITHM_REGISTRY",
                toy_registry,
            ):
                config = load_experiment_config(config_path)
                experiment_dir = root / "experiments" / config.experiment.id
                create_experiment_layout(experiment_dir)
                hashes = snapshot_slab_inputs(
                    slab_type=config.experiment.slab_type,
                    inputs_dir=experiment_dir / "inputs",
                )
                tasks = plan_optimization_tasks(config)
                paths, _manifest, _env = prepare_experiment_artifacts(
                    config,
                    root=experiment_dir,
                    tasks=tasks,
                    input_files=hashes,
                )
                for task in tasks:
                    create_task_layout(paths, task.task_index)
                attempt = _dispatch_attempt(experiment_dir)
                code = worker_main(
                    [
                        "--experiment-dir",
                        str(experiment_dir),
                        "--task-index",
                        "0",
                        "--attempt",
                        str(attempt.attempt),
                    ]
                )
            self.assertEqual(code, 0)
            summary = read_json(attempt.attempt_dir / "summary.json")
            # Instance identity comes from the shared RandomSearch builder; the
            # registry token is what the worker looked up without a local branch.
            from core.execution_core.manifest import load_manifest

            manifest = load_manifest(experiment_dir / "manifest.json")
            self.assertEqual(
                manifest.tasks[0].payload["algorithm_name"],
                algorithm_name,
            )
            self.assertEqual(summary["evaluations"], 3)
            self.assertEqual(
                summary["algorithm_parameters"],
                {"max_evaluations": 3},
            )
            status = read_task_status(attempt.task_dir)
            self.assertEqual(status["state"], TaskState.COMPLETED.value)

    def test_worker_runs_tiny_pymoo_task_without_worker_branch(self) -> None:
        import inspect

        import core.ioh_core.workflows.optimization_runs.worker as worker_mod

        self.assertNotIn("PYMOO_GA", inspect.getsource(worker_mod))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "pymoo.toml"
            config_path.write_text(
                textwrap.dedent(
                    """\
                    schema_version = 1

                    [experiment]
                    id = "worker_pymoo"
                    workflow = "optimization_runs"
                    slab_type = "hp_slab"
                    problem_ids = ["L6750_B2"]
                    output_root = "experiments"

                    [[experiment.algorithms]]
                    name = "PYMOO_GA"
                    max_evaluations = 8
                    [experiment.algorithms.parameters]
                    population_size = 4

                    [experiment.repetitions]
                    count = 1
                    seed_base = 42

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
            config = load_experiment_config(config_path)
            experiment_dir = root / "experiments" / config.experiment.id
            create_experiment_layout(experiment_dir)
            hashes = snapshot_slab_inputs(
                slab_type=config.experiment.slab_type,
                inputs_dir=experiment_dir / "inputs",
            )
            tasks = plan_optimization_tasks(config)
            paths, _manifest, _env = prepare_experiment_artifacts(
                config,
                root=experiment_dir,
                tasks=tasks,
                input_files=hashes,
            )
            for task in tasks:
                create_task_layout(paths, task.task_index)
            attempt = _dispatch_attempt(experiment_dir)
            code = worker_main(
                [
                    "--experiment-dir",
                    str(experiment_dir),
                    "--task-index",
                    "0",
                    "--attempt",
                    str(attempt.attempt),
                ]
            )
            self.assertEqual(code, 0)
            summary = read_json(attempt.attempt_dir / "summary.json")
            self.assertEqual(summary["algorithm_name"], "PYMOO_GA")
            self.assertLessEqual(summary["evaluations"], 8)
            self.assertGreaterEqual(summary["evaluations"], 1)
            self.assertEqual(
                summary["algorithm_parameters"]["max_evaluations"],
                8,
            )
            self.assertEqual(
                summary["algorithm_parameters"]["population_size"],
                4,
            )
            status = read_task_status(attempt.task_dir)
            self.assertEqual(status["state"], TaskState.COMPLETED.value)

    def test_worker_runs_tiny_hyperopt_task_without_worker_branch(self) -> None:
        import inspect

        import core.ioh_core.workflows.optimization_runs.worker as worker_mod

        self.assertNotIn("HYPEROPT_TPE", inspect.getsource(worker_mod))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "hyperopt.toml"
            config_path.write_text(
                textwrap.dedent(
                    """\
                    schema_version = 1

                    [experiment]
                    id = "worker_hyperopt"
                    workflow = "optimization_runs"
                    slab_type = "hp_slab"
                    problem_ids = ["L6750_B2"]
                    output_root = "experiments"

                    [[experiment.algorithms]]
                    name = "HYPEROPT_TPE"
                    max_evaluations = 8

                    [experiment.repetitions]
                    count = 1
                    seed_base = 42

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
            config = load_experiment_config(config_path)
            experiment_dir = root / "experiments" / config.experiment.id
            create_experiment_layout(experiment_dir)
            hashes = snapshot_slab_inputs(
                slab_type=config.experiment.slab_type,
                inputs_dir=experiment_dir / "inputs",
            )
            tasks = plan_optimization_tasks(config)
            paths, _manifest, _env = prepare_experiment_artifacts(
                config,
                root=experiment_dir,
                tasks=tasks,
                input_files=hashes,
            )
            for task in tasks:
                create_task_layout(paths, task.task_index)
            attempt = _dispatch_attempt(experiment_dir)
            code = worker_main(
                [
                    "--experiment-dir",
                    str(experiment_dir),
                    "--task-index",
                    "0",
                    "--attempt",
                    str(attempt.attempt),
                ]
            )
            self.assertEqual(code, 0)
            summary = read_json(attempt.attempt_dir / "summary.json")
            self.assertEqual(summary["algorithm_name"], "HYPEROPT_TPE")
            self.assertLessEqual(summary["evaluations"], 8)
            self.assertGreaterEqual(summary["evaluations"], 1)
            self.assertEqual(
                summary["algorithm_parameters"],
                {"max_evaluations": 8},
            )
            status = read_task_status(attempt.task_dir)
            self.assertEqual(status["state"], TaskState.COMPLETED.value)

    def test_worker_runs_tiny_optuna_task_without_worker_branch(self) -> None:
        import inspect

        import core.ioh_core.workflows.optimization_runs.worker as worker_mod

        self.assertNotIn("OPTUNA_TPE", inspect.getsource(worker_mod))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "optuna.toml"
            config_path.write_text(
                textwrap.dedent(
                    """\
                    schema_version = 1

                    [experiment]
                    id = "worker_optuna"
                    workflow = "optimization_runs"
                    slab_type = "hp_slab"
                    problem_ids = ["L6750_B2"]
                    output_root = "experiments"

                    [[experiment.algorithms]]
                    name = "OPTUNA_TPE"
                    max_evaluations = 8

                    [experiment.repetitions]
                    count = 1
                    seed_base = 42

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
            config = load_experiment_config(config_path)
            experiment_dir = root / "experiments" / config.experiment.id
            create_experiment_layout(experiment_dir)
            hashes = snapshot_slab_inputs(
                slab_type=config.experiment.slab_type,
                inputs_dir=experiment_dir / "inputs",
            )
            tasks = plan_optimization_tasks(config)
            paths, _manifest, _env = prepare_experiment_artifacts(
                config,
                root=experiment_dir,
                tasks=tasks,
                input_files=hashes,
            )
            for task in tasks:
                create_task_layout(paths, task.task_index)
            attempt = _dispatch_attempt(experiment_dir)
            code = worker_main(
                [
                    "--experiment-dir",
                    str(experiment_dir),
                    "--task-index",
                    "0",
                    "--attempt",
                    str(attempt.attempt),
                ]
            )
            self.assertEqual(code, 0)
            summary = read_json(attempt.attempt_dir / "summary.json")
            self.assertEqual(summary["algorithm_name"], "OPTUNA_TPE")
            self.assertLessEqual(summary["evaluations"], 8)
            self.assertGreaterEqual(summary["evaluations"], 1)
            self.assertEqual(
                summary["algorithm_parameters"],
                {"max_evaluations": 8},
            )
            status = read_task_status(attempt.task_dir)
            self.assertEqual(status["state"], TaskState.COMPLETED.value)

    def test_worker_records_missing_rbfopt_solver_failure(self) -> None:
        import inspect

        import core.ioh_core.workflows.optimization_runs.worker as worker_mod

        self.assertNotIn("RBFOPT", inspect.getsource(worker_mod))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "rbfopt_missing.toml"
            config_path.write_text(
                textwrap.dedent(
                    """\
                    schema_version = 1

                    [experiment]
                    id = "worker_rbfopt_missing"
                    workflow = "optimization_runs"
                    slab_type = "hp_slab"
                    problem_ids = ["L6750_B2"]
                    output_root = "experiments"

                    [[experiment.algorithms]]
                    name = "RBFOPT"
                    max_evaluations = 8
                    [experiment.algorithms.parameters]
                    ga_base_population_size = 20
                    minlp_solver_path = "definitely-missing-bonmin-xyz"
                    nlp_solver_path = "definitely-missing-ipopt-xyz"

                    [experiment.repetitions]
                    count = 1
                    seed_base = 42

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
            config = load_experiment_config(config_path)
            experiment_dir = root / "experiments" / config.experiment.id
            create_experiment_layout(experiment_dir)
            hashes = snapshot_slab_inputs(
                slab_type=config.experiment.slab_type,
                inputs_dir=experiment_dir / "inputs",
            )
            tasks = plan_optimization_tasks(config)
            paths, _manifest, _env = prepare_experiment_artifacts(
                config,
                root=experiment_dir,
                tasks=tasks,
                input_files=hashes,
            )
            for task in tasks:
                create_task_layout(paths, task.task_index)
            attempt = _dispatch_attempt(experiment_dir)
            code = worker_main(
                [
                    "--experiment-dir",
                    str(experiment_dir),
                    "--task-index",
                    "0",
                    "--attempt",
                    str(attempt.attempt),
                ]
            )
            self.assertEqual(code, 1)
            failure = read_json(attempt.attempt_dir / "failure.json")
            self.assertEqual(failure["error_type"], "AlgorithmDependencyError")
            self.assertIn("minlp_solver_path", failure["message"])
            self.assertIn("definitely-missing-bonmin-xyz", failure["message"])
            self.assertFalse((attempt.attempt_dir / "summary.json").exists())
            status = read_task_status(attempt.task_dir)
            self.assertEqual(status["state"], TaskState.FAILED.value)

    @unittest.skipUnless(
        shutil.which("bonmin") and shutil.which("ipopt"),
        "bonmin and ipopt executables are not available",
    )
    def test_worker_runs_tiny_rbfopt_when_solvers_available(self) -> None:
        import inspect

        import core.ioh_core.workflows.optimization_runs.worker as worker_mod

        self.assertNotIn("RBFOPT", inspect.getsource(worker_mod))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "rbfopt.toml"
            config_path.write_text(
                textwrap.dedent(
                    """\
                    schema_version = 1

                    [experiment]
                    id = "worker_rbfopt"
                    workflow = "optimization_runs"
                    slab_type = "hp_slab"
                    problem_ids = ["L6750_B2"]
                    output_root = "experiments"

                    [[experiment.algorithms]]
                    name = "RBFOPT"
                    max_evaluations = 8
                    [experiment.algorithms.parameters]
                    ga_base_population_size = 20
                    minlp_solver_path = "bonmin"
                    nlp_solver_path = "ipopt"

                    [experiment.repetitions]
                    count = 1
                    seed_base = 42

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
            config = load_experiment_config(config_path)
            experiment_dir = root / "experiments" / config.experiment.id
            create_experiment_layout(experiment_dir)
            hashes = snapshot_slab_inputs(
                slab_type=config.experiment.slab_type,
                inputs_dir=experiment_dir / "inputs",
            )
            tasks = plan_optimization_tasks(config)
            paths, _manifest, _env = prepare_experiment_artifacts(
                config,
                root=experiment_dir,
                tasks=tasks,
                input_files=hashes,
            )
            for task in tasks:
                create_task_layout(paths, task.task_index)
            attempt = _dispatch_attempt(experiment_dir)
            code = worker_main(
                [
                    "--experiment-dir",
                    str(experiment_dir),
                    "--task-index",
                    "0",
                    "--attempt",
                    str(attempt.attempt),
                ]
            )
            self.assertEqual(code, 0)
            summary = read_json(attempt.attempt_dir / "summary.json")
            self.assertEqual(summary["algorithm_name"], "RBFOPT")
            self.assertLessEqual(summary["evaluations"], 8)
            self.assertGreaterEqual(summary["evaluations"], 1)
            self.assertEqual(
                summary["algorithm_parameters"]["max_evaluations"],
                8,
            )
            self.assertEqual(
                summary["algorithm_parameters"]["ga_base_population_size"],
                20,
            )
            status = read_task_status(attempt.task_dir)
            self.assertEqual(status["state"], TaskState.COMPLETED.value)


class TestLegacyRunnerCompatibility(unittest.TestCase):
    """Legacy run_experiment remains usable and still patches suite metadata."""

    def test_legacy_runner_output_still_patches_suite(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "legacy"
            bundle = {"L6750_B2": build_problems_for_slab_type("hp_slab")["L6750_B2"]}
            algorithm = create_algorithm(
                "RANDOM_SEARCH",
                {"max_evaluations": 2},
                random_seed=1,
            )
            run_experiment(
                problem_bundle=bundle,
                algorithm=algorithm,
                n_runs=1,
                name="legacy_compat",
                zip_results=False,
                output_root=out,
            )
            jsons = list(out.rglob("IOHprofiler_f*.json"))
            self.assertTrue(jsons)
            for jf in jsons:
                data = json.loads(jf.read_text(encoding="utf-8"))
                self.assertEqual(data.get("suite"), SUITE_NAME)


if __name__ == "__main__":
    unittest.main()
