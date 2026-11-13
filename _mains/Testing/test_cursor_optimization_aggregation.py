"""
Phase A8 / C4.8 gate: deterministic optimisation aggregation outputs.
"""
from __future__ import annotations

import csv
import json
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.artifacts import (
    read_task_status,
    reserve_attempt,
    transition_task_status,
)
from core.execution_core.atomic_io import write_json_atomic
from core.execution_core.lifecycle import prepare_experiment
from core.execution_core.models import TaskState
from core.ioh_core.workflows.optimization_runs.aggregation import (
    CSV_FIELDS,
    AggregationError,
    aggregate_optimization_experiment,
    collect_optimization_rows,
    normalize_summary_termination,
)
from core.ioh_core.workflows.optimization_runs.fixed_target_feasibility_audit import (
    audit_catalogue_thresholds,
    write_audit_artifacts,
)


def _write_config(path: Path, *, termination: str = "") -> Path:
    """
    Write a two-task local optimisation TOML.

    Parameters
    ----------
    path : Path
        Destination path.
    termination : str, optional
        Optional ``[experiment.termination]`` fragment.

    Returns
    -------
    Path
        Written path.
    """
    body = textwrap.dedent(
        """\
        schema_version = 1

        [experiment]
        id = "agg_smoke"
        workflow = "optimization_runs"
        slab_type = "hp_slab"
        problem_ids = ["L6750_B2"]
        output_root = "experiments"

        [[experiment.algorithms]]
        name = "RANDOM_SEARCH"
        max_evaluations = 2

        [experiment.repetitions]
        count = 2
        seed_base = 11

        [execution]
        backend = "local"
        max_parallel_tasks = 1
        cpus_per_task = 1

        [execution.local]
        python_executable = ""
        """
    )
    if termination.strip():
        body = body + "\n" + termination.strip() + "\n"
    path.write_text(body, encoding="utf-8")
    return path


def _complete_task(
    experiment_dir: Path,
    task_index: int,
    summary: dict,
) -> None:
    """Mark one task completed with the given summary payload."""
    task_dir = experiment_dir / "tasks" / f"task_{task_index:06d}"
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
    write_json_atomic(attempt.attempt_dir / "summary.json", summary)
    transition_task_status(
        task_dir,
        expected={TaskState.RUNNING},
        new_state=TaskState.COMPLETED,
        attempt=attempt.attempt,
    )


class TestOptimizationAggregation(unittest.TestCase):
    """CSV/JSON aggregation ordering and field coverage."""

    def test_deterministic_order_and_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = root / "exp"
            prepare_experiment(
                config_path=_write_config(root / "cfg.toml"),
                output=experiment_dir,
            )

            # Finish task 1 first, then task 0, to prove order is by index.
            for task_index, objective in ((1, 20.0), (0, 10.0)):
                _complete_task(
                    experiment_dir,
                    task_index,
                    {
                        "best_objective": objective,
                        "best_x": [task_index, 1],
                        "hits": 1,
                        "misses": 2,
                        "evaluations": 3,
                        "wall_time_s": 0.5,
                    },
                )

            rows = collect_optimization_rows(experiment_dir)
            self.assertEqual([row["task_index"] for row in rows], [0, 1])
            self.assertEqual(rows[0]["best_objective"], 10.0)
            self.assertEqual(rows[1]["best_objective"], 20.0)
            self.assertEqual(json.loads(rows[0]["best_x_json"]), [0, 1])
            # Legacy summaries normalize to fixed-budget termination fields.
            self.assertEqual(rows[0]["termination_mode"], "fixed_budget")
            self.assertEqual(rows[0]["target_hit"], False)
            self.assertEqual(rows[0]["stop_reason"], "budget_exhausted")
            self.assertEqual(rows[0]["budget_exhausted"], True)

            summary = aggregate_optimization_experiment(experiment_dir)
            self.assertIsNone(summary["feasibility_guarantee"])
            self.assertEqual(summary["feasibility_guarantee_pairs"], [])
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["fixed_target_task_count"], 0)
            self.assertIsNone(summary["target_hit_share"])
            self.assertIsNone(summary["evaluations_to_target_min"])
            csv_path = experiment_dir / "aggregate" / "tasks.csv"
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(CSV_FIELDS))
                csv_rows = list(reader)
            self.assertEqual([int(r["task_index"]) for r in csv_rows], [0, 1])
            self.assertEqual(float(csv_rows[0]["best_objective"]), 10.0)
            self.assertTrue((experiment_dir / "aggregate" / "summary.json").is_file())
            # Status immutability of completed tasks still holds.
            self.assertEqual(
                read_task_status(experiment_dir / "tasks" / "task_000000")["state"],
                "completed",
            )

    def test_old_summary_budget_exhausted_stop_reason(self) -> None:
        fields = normalize_summary_termination(
            {"best_objective": 1.0, "evaluations": 2},
            {"algorithm_parameters": {"max_evaluations": 2}},
        )
        self.assertEqual(fields["termination_mode"], "fixed_budget")
        self.assertFalse(fields["target_hit"])
        self.assertTrue(fields["budget_exhausted"])
        self.assertEqual(fields["stop_reason"], "budget_exhausted")

    def test_fixed_target_rollups_and_csv_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = root / "exp_ft"
            prepare_experiment(
                config_path=_write_config(
                    root / "cfg.toml",
                    termination=textwrap.dedent(
                        """\
                        [experiment.termination]
                        mode = "fixed_target"
                        global_optimum = 100.0
                        allowed_deviation_percent = 0.0
                        target_reference = "all"
                        """
                    ),
                ),
                output=experiment_dir,
            )
            _complete_task(
                experiment_dir,
                0,
                {
                    "best_objective": 90.0,
                    "best_x": [0],
                    "evaluations": 1,
                    "termination_mode": "fixed_target",
                    "global_optimum": 100.0,
                    "allowed_deviation_percent": 0.0,
                    "target_threshold": 100.0,
                    "target_hit": True,
                    "evaluations_to_target": 1,
                    "budget_exhausted": False,
                    "stop_reason": "target_hit",
                },
            )
            _complete_task(
                experiment_dir,
                1,
                {
                    "best_objective": 150.0,
                    "best_x": [1],
                    "evaluations": 2,
                    "termination_mode": "fixed_target",
                    "global_optimum": 100.0,
                    "allowed_deviation_percent": 0.0,
                    "target_threshold": 100.0,
                    "target_hit": False,
                    "evaluations_to_target": None,
                    "budget_exhausted": True,
                    "stop_reason": "budget_exhausted",
                },
            )
            summary = aggregate_optimization_experiment(experiment_dir)
            self.assertEqual(summary["fixed_target_task_count"], 2)
            self.assertEqual(summary["target_hit_count"], 1)
            self.assertEqual(summary["target_miss_count"], 1)
            self.assertAlmostEqual(summary["target_hit_share"], 0.5)
            self.assertAlmostEqual(summary["target_miss_share"], 0.5)
            self.assertEqual(summary["evaluations_to_target_min"], 1)
            self.assertEqual(summary["evaluations_to_target_max"], 1)
            self.assertEqual(summary["evaluations_to_target_median"], 1.0)
            self.assertEqual(summary["budget_exhausted_count"], 1)
            self.assertEqual(summary["feasibility_guarantee"], "unverified")
            self.assertEqual(
                [pair["state"] for pair in summary["feasibility_guarantee_pairs"]],
                ["unverified", "unverified"],
            )
            self.assertEqual(summary["target_hit_count"], 1)

            csv_path = experiment_dir / "aggregate" / "tasks.csv"
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames[-8:],
                    list(
                        (
                            "termination_mode",
                            "global_optimum",
                            "allowed_deviation_percent",
                            "target_threshold",
                            "target_hit",
                            "evaluations_to_target",
                            "budget_exhausted",
                            "stop_reason",
                        )
                    ),
                )
                rows = list(reader)
            self.assertEqual(rows[0]["stop_reason"], "target_hit")
            self.assertEqual(rows[0]["evaluations_to_target"], "1")
            self.assertEqual(rows[1]["evaluations_to_target"], "")

    def test_malformed_fixed_target_summary_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = root / "exp_bad"
            prepare_experiment(
                config_path=_write_config(
                    root / "cfg.toml",
                    termination=textwrap.dedent(
                        """\
                        [experiment.termination]
                        mode = "fixed_target"
                        global_optimum = 10.0
                        allowed_deviation_percent = 0.0
                        target_reference = "all"
                        """
                    ),
                ),
                output=experiment_dir,
            )
            _complete_task(
                experiment_dir,
                0,
                {
                    "best_objective": 11.0,
                    "best_x": [0],
                    "evaluations": 1,
                    "termination_mode": "fixed_target",
                    "global_optimum": 10.0,
                    "allowed_deviation_percent": 0.0,
                    "target_threshold": 10.0,
                    "target_hit": True,
                    "evaluations_to_target": 1,
                    "budget_exhausted": False,
                    "stop_reason": "target_hit",
                },
            )
            with self.assertRaises(AggregationError):
                collect_optimization_rows(experiment_dir)

    def test_failed_audit_does_not_change_target_hit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = root / "exp_fail"
            prepare_experiment(
                config_path=_write_config(
                    root / "cfg.toml",
                    termination=textwrap.dedent(
                        """\
                        [experiment.termination]
                        mode = "fixed_target"
                        global_optimum = 100.0
                        allowed_deviation_percent = 0.0
                        target_reference = "all"
                        """
                    ),
                ),
                output=experiment_dir,
            )
            _complete_task(
                experiment_dir,
                0,
                {
                    "best_objective": 90.0,
                    "best_x": [0],
                    "evaluations": 1,
                    "termination_mode": "fixed_target",
                    "global_optimum": 100.0,
                    "allowed_deviation_percent": 0.0,
                    "target_threshold": 100.0,
                    "target_hit": True,
                    "evaluations_to_target": 1,
                    "budget_exhausted": False,
                    "stop_reason": "target_hit",
                },
            )
            _complete_task(
                experiment_dir,
                1,
                {
                    "best_objective": 150.0,
                    "best_x": [1],
                    "evaluations": 2,
                    "termination_mode": "fixed_target",
                    "global_optimum": 100.0,
                    "allowed_deviation_percent": 0.0,
                    "target_threshold": 100.0,
                    "target_hit": False,
                    "evaluations_to_target": None,
                    "budget_exhausted": True,
                    "stop_reason": "budget_exhausted",
                },
            )
            audit = audit_catalogue_thresholds(
                [
                    {
                        "problem_id": "L6750_B2",
                        "cardinality": 2,
                        "rows": [
                            {
                                "global_index": 0,
                                "analysis_ok": True,
                                "constraint_feasible": True,
                                "feasible": True,
                                "y_p": 100.0,
                            },
                            {
                                "global_index": 1,
                                "analysis_ok": False,
                                "constraint_feasible": False,
                                "feasible": False,
                                "y_p": 100.0,
                            },
                        ],
                        "constraint_names": ["g"],
                        "penalty_alpha": 1.0,
                        "penalty_beta": 1.0 / 3.0,
                        "objective_identity": "y_p",
                        "source_path": "synthetic.duckdb",
                        "table_name": "catalogue",
                        "thresholds": [
                            {
                                "target_threshold": 100.0,
                                "allowed_deviation_percent": 0.0,
                            }
                        ],
                    }
                ],
                campaign_problem_ids=("L6750_B2",),
                campaign_constraint_names=("g",),
                campaign_penalty_alpha=1.0,
                campaign_penalty_beta=1.0 / 3.0,
                campaign_objective_identity="y_p",
                code_revision="test",
                audited_at="2026-01-01T00:00:00Z",
            )
            write_audit_artifacts(audit, experiment_dir)
            summary = aggregate_optimization_experiment(experiment_dir)
            self.assertEqual(summary["feasibility_guarantee"], "fail")
            self.assertEqual(summary["target_hit_count"], 1)
            self.assertEqual(summary["target_miss_count"], 1)
            self.assertEqual(
                [pair["state"] for pair in summary["feasibility_guarantee_pairs"]],
                ["fail", "fail"],
            )


if __name__ == "__main__":
    unittest.main()
