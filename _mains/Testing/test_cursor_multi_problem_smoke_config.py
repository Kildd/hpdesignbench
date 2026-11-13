"""Contract test for the runnable multi-problem optimization smoke config."""
from __future__ import annotations

import unittest
from pathlib import Path

from core.execution_core.config import load_experiment_config
from core.execution_core.lifecycle import format_experiment_startup_message
from core.ioh_core.workflows.optimization_runs.models import OptimizationTaskSpec
from core.ioh_core.workflows.optimization_runs.planner import (
    plan_optimization_tasks,
)
from core.ioh_core.workflows.registry import get_workflow


_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = (
    _ROOT
    / "_mains"
    / "studies"
    / "Execution"
    / "configs"
    / "examples"
    / "optimization_multi_problem_smoke.toml"
)
_REFERENCE4_CONFIG = (
    _ROOT
    / "_mains"
    / "studies"
    / "Execution"
    / "configs"
    / "study_ws_cmp_reference4_all.toml"
)


class TestMultiProblemSmokeConfig(unittest.TestCase):
    def test_plans_expected_cross_problem_tasks(self) -> None:
        config = load_experiment_config(_CONFIG)
        tasks = plan_optimization_tasks(config)

        self.assertEqual(len(tasks), 8)
        self.assertEqual(
            [task.task_index for task in tasks],
            list(range(8)),
        )
        specs = [
            OptimizationTaskSpec.from_payload(task.payload)
            for task in tasks
        ]
        self.assertEqual(
            [
                (spec.problem_id, spec.algorithm_name, spec.repetition)
                for spec in specs
            ],
            [
                ("L6750_B2", "RANDOM_SEARCH", 0),
                ("L6750_B2", "RANDOM_SEARCH", 1),
                ("L6750_B2", "LATIN_HYPERCUBE", 0),
                ("L6750_B2", "SCIPY_DIRECT_L", 0),
                ("L8100_B2", "RANDOM_SEARCH", 0),
                ("L8100_B2", "RANDOM_SEARCH", 1),
                ("L8100_B2", "LATIN_HYPERCUBE", 0),
                ("L8100_B2", "SCIPY_DIRECT_L", 0),
            ],
        )

        optima = {
            "L6750_B2": 353.88457837247404,
            "L8100_B2": 458.03280562909964,
        }
        for spec in specs:
            self.assertEqual(
                spec.algorithm_parameters["max_evaluations"],
                10,
            )
            self.assertEqual(spec.global_optimum, optima[spec.problem_id])
            self.assertAlmostEqual(
                spec.target_threshold,
                optima[spec.problem_id] * 1.01,
            )
            self.assertNotIn("global_optima", spec.to_payload())
            if spec.algorithm_name == "SCIPY_DIRECT_L":
                self.assertIsNone(spec.random_seed)
            else:
                self.assertIsInstance(spec.random_seed, int)

        entry = get_workflow("optimization_runs")
        assert entry.validator is not None
        entry.validator(config, tasks)

        banner = format_experiment_startup_message(
            config,
            experiment_dir=_ROOT / "experiments" / "smoke",
        )
        self.assertIn(
            "LATIN_HYPERCUBE (max_evaluations=10, repetitions=1)",
            banner,
        )
        self.assertIn("tasks       : 8", banner)
        self.assertIn(
            "2 problem(s) x 4 effective algorithm repetition(s) per problem",
            banner,
        )

    def test_reference4_campaign_plans_complete_roster(self) -> None:
        config = load_experiment_config(_REFERENCE4_CONFIG)
        tasks = plan_optimization_tasks(config)

        self.assertEqual(len(config.experiment.problem_ids), 4)
        self.assertEqual(len(config.experiment.algorithms), 44)
        self.assertEqual(len(tasks), 3064)
        counts = {
            algorithm.name: (
                algorithm.repetitions
                if algorithm.repetitions is not None
                else config.experiment.repetitions.count
            )
            for algorithm in config.experiment.algorithms
        }
        self.assertEqual(
            sum(counts.values()),
            38 * 20 + 6,
        )
        self.assertEqual(
            sum(value == 1 for value in counts.values()),
            6,
        )
        self.assertIn("OPTUNA_TPE", counts)
        self.assertIn("OPTUNA_TPE_WARM_START", counts)
        self.assertIn("SCIPY_DUAL_ANNEALING", counts)
        self.assertIn("SCIPY_DUAL_ANNEALING_WARM_START", counts)
        self.assertIn("K_ELITE_AMS_PEN_SI_DESC_COLD_START", counts)
        self.assertIn("K_ELITE_AMS_PEN_SI_DESC_WARM_START", counts)
        self.assertNotIn("HYPEROPT_TPE", counts)
        self.assertNotIn("HYPEROPT_TPE_WARM_START", counts)

        entry = get_workflow("optimization_runs")
        assert entry.validator is not None
        entry.validator(config, tasks)


if __name__ == "__main__":
    unittest.main()
