"""
ALG8.1: configured sequential experiment construction.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from core.execution_core.lifecycle import (
    collect_experiment,
    dispatch_experiment,
    list_task_statuses,
    prepare_experiment,
)
from core.ioh_core.algorithm_factory import create_algorithm
from core.ioh_core.experiment import run_configured_experiment, run_experiment
from core.ioh_core.problem_builder import build_problems_for_slab_type
from core.ioh_core.workflows.optimization_runs.planner import derive_random_seed
from _mains.Testing.test_cursor_parallel_optimization_equivalence import (
    _outcomes_by_identity,
    _write_config,
)


class TestConfiguredExperiment(unittest.TestCase):
    """Fresh construction, planner seeds, and legacy reuse."""

    def test_configured_experiment_constructs_fresh_instance_per_run(self) -> None:
        created: list[Any] = []
        real_create = create_algorithm

        def tracking_create(name, parameters, random_seed=None):
            obj = real_create(name, parameters, random_seed=random_seed)
            created.append(obj)
            return obj

        with TemporaryDirectory() as tmp:
            bundle = {"L6750_B2": build_problems_for_slab_type("hp_slab")["L6750_B2"]}
            with patch(
                "core.ioh_core.experiment.create_algorithm",
                side_effect=tracking_create,
            ):
                run_configured_experiment(
                    bundle,
                    algorithm_name="RANDOM_SEARCH",
                    algorithm_parameters={"max_evaluations": 2},
                    repetitions=2,
                    seed_base=7,
                    slab_type="hp_slab",
                    name="fresh_instances",
                    zip_results=False,
                    output_root=Path(tmp) / "out",
                )

        self.assertEqual(len(created), 2)
        self.assertIsNot(created[0], created[1])

    def test_configured_experiment_uses_planner_seed_derivation(self) -> None:
        seen_seeds: list[int | None] = []
        real_create = create_algorithm
        seed_base = 42
        slab_type = "hp_slab"
        algorithm_name = "RANDOM_SEARCH"
        problem_id = "L6750_B2"
        repetitions = 2

        def tracking_create(name, parameters, random_seed=None):
            seen_seeds.append(random_seed)
            return real_create(name, parameters, random_seed=random_seed)

        expected = [
            derive_random_seed(
                seed_base=seed_base,
                slab_type=slab_type,
                problem_id=problem_id,
                algorithm_name=algorithm_name,
                repetition=rep,
            )
            for rep in range(repetitions)
        ]

        with TemporaryDirectory() as tmp:
            bundle = {problem_id: build_problems_for_slab_type(slab_type)[problem_id]}
            with patch(
                "core.ioh_core.experiment.create_algorithm",
                side_effect=tracking_create,
            ):
                run_configured_experiment(
                    bundle,
                    algorithm_name=algorithm_name,
                    algorithm_parameters={"max_evaluations": 2},
                    repetitions=repetitions,
                    seed_base=seed_base,
                    slab_type=slab_type,
                    name="seed_derivation",
                    zip_results=False,
                    output_root=Path(tmp) / "out",
                )

        self.assertEqual(seen_seeds, expected)

    def test_configured_and_parallel_random_search_are_semantically_equivalent(
        self,
    ) -> None:
        seed_base = 42
        repetitions = 2
        budget = 3
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "parallel"
            prepare_experiment(
                config_path=_write_config(
                    root / "eq.toml",
                    experiment_id="cfg_vs_par",
                    max_parallel_tasks=1,
                    algorithms=[("RANDOM_SEARCH", budget)],
                    repetitions=repetitions,
                    seed_base=seed_base,
                ),
                output=exp_dir,
            )
            dispatch_experiment(exp_dir)
            status = list_task_statuses(exp_dir)
            self.assertTrue(status.complete)
            collect_experiment(exp_dir, require_complete=True)
            parallel = _outcomes_by_identity(exp_dir)

            bundle = {"L6750_B2": build_problems_for_slab_type("hp_slab")["L6750_B2"]}
            summary = run_configured_experiment(
                bundle,
                algorithm_name="RANDOM_SEARCH",
                algorithm_parameters={"max_evaluations": budget},
                repetitions=repetitions,
                seed_base=seed_base,
                slab_type="hp_slab",
                name="cfg_eq",
                zip_results=False,
                return_summary=True,
                output_root=root / "configured",
            )

            self.assertEqual(len(summary["runs"]), repetitions)
            for row in summary["runs"]:
                repetition = int(row["run"]) - 1
                seed = derive_random_seed(
                    seed_base=seed_base,
                    slab_type="hp_slab",
                    problem_id="L6750_B2",
                    algorithm_name="RANDOM_SEARCH",
                    repetition=repetition,
                )
                key = ("L6750_B2", "RANDOM_SEARCH", repetition, seed)
                self.assertIn(key, parallel)
                left = parallel[key]
                self.assertEqual(left["best_x"], [int(v) for v in row["best_x"]])
                self.assertAlmostEqual(
                    left["best_objective"],
                    float(row["best_y"]),
                    places=12,
                )
                self.assertEqual(left["evaluations"], budget)

    def test_configured_experiment_uses_null_seed_for_deterministic_algorithm(
        self,
    ) -> None:
        seen_seeds: list[int | None] = []
        real_create = create_algorithm

        def tracking_create(name, parameters, random_seed=None):
            seen_seeds.append(random_seed)
            return real_create(name, parameters, random_seed=random_seed)

        with TemporaryDirectory() as tmp:
            bundle = {"L6750_B2": build_problems_for_slab_type("hp_slab")["L6750_B2"]}
            with patch(
                "core.ioh_core.experiment.create_algorithm",
                side_effect=tracking_create,
            ):
                run_configured_experiment(
                    bundle,
                    algorithm_name="NLOPT_DIRECT_L",
                    algorithm_parameters={"max_evaluations": 2},
                    repetitions=1,
                    seed_base=99,
                    slab_type="hp_slab",
                    name="null_seed",
                    zip_results=False,
                    output_root=Path(tmp) / "out",
                )

        self.assertEqual(seen_seeds, [None])

    def test_legacy_run_experiment_reuses_supplied_object(self) -> None:
        call_ids: list[int] = []
        algorithm = create_algorithm(
            "RANDOM_SEARCH",
            {"max_evaluations": 2},
            random_seed=1,
        )

        class ReusedWrapper:
            name = algorithm.name
            info = algorithm.info

            def __call__(self, problem):
                call_ids.append(id(self))
                return algorithm(problem)

        wrapper = ReusedWrapper()
        with TemporaryDirectory() as tmp:
            bundle = {"L6750_B2": build_problems_for_slab_type("hp_slab")["L6750_B2"]}
            run_experiment(
                problem_bundle=bundle,
                algorithm=wrapper,
                n_runs=2,
                name="legacy_reuse",
                zip_results=False,
                output_root=Path(tmp) / "out",
            )

        self.assertEqual(call_ids, [id(wrapper), id(wrapper)])


if __name__ == "__main__":
    unittest.main()
