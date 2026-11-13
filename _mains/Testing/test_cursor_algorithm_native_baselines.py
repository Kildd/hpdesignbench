"""
Phase ALG7.2 / C4.4 gate: native random-search and Latin-hypercube baselines.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.ioh_core.algorithm_factory import create_algorithm
from core.ioh_core.algorithms.common import RunController
from core.ioh_core.algorithms.registry import (
    BudgetSemantics,
    CandidateMode,
    SeedPolicy,
    get_algorithm_spec,
    normalize_algorithm_parameters,
)
from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem


class TestRandomSearchFixedTarget(unittest.TestCase):
    """RANDOM_SEARCH fixed-target stopping."""

    def test_target_hit_stops_without_extra_call(self) -> None:
        problem = FakeIOHProblem([0, 0], [0, 0])
        controller = RunController(
            problem,
            8,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "RANDOM_SEARCH",
            {"max_evaluations": 8},
            random_seed=1,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 1)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.evaluations_to_target, 1)
        self.assertEqual(controller.stop_reason(), "target_hit")

    def test_target_miss_uses_exact_hard_ceiling(self) -> None:
        problem = FakeIOHProblem([2, 2], [4, 4])
        controller = RunController(
            problem,
            5,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "RANDOM_SEARCH",
            {"max_evaluations": 5},
            random_seed=2,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 5)
        self.assertFalse(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")

    def test_finite_objective_inside_margin_stops(self) -> None:
        problem = FakeIOHProblem([0, 0], [0, 0])
        problem.next_feasible = False
        controller = RunController(
            problem,
            3,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "RANDOM_SEARCH",
            {"max_evaluations": 3},
            random_seed=3,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 1)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "target_hit")


class TestLatinHypercubeSampling(unittest.TestCase):
    """Registered LATIN_HYPERCUBE adapter behaviour."""

    def test_lhs_registry_spec(self) -> None:
        spec = get_algorithm_spec("LATIN_HYPERCUBE")
        self.assertEqual(spec.library, "native")
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.native.factory:build_latin_hypercube",
        )
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertIsNone(spec.dependency_extra)
        self.assertFalse(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations"],
        )
        normalized = normalize_algorithm_parameters(
            "LATIN_HYPERCUBE",
            {"max_evaluations": 10},
        )
        self.assertEqual(normalized, {"max_evaluations": 10})

    def test_lhs_same_seed_same_candidate_sequence(self) -> None:
        problem_a = FakeIOHProblem([0, 1], [3, 4])
        problem_b = FakeIOHProblem([0, 1], [3, 4])
        create_algorithm(
            "LATIN_HYPERCUBE",
            {"max_evaluations": 8},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            "LATIN_HYPERCUBE",
            {"max_evaluations": 8},
            random_seed=17,
        )(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)
        self.assertEqual(len(problem_a.calls), 8)

    def test_lhs_strata_are_permuted_per_dimension(self) -> None:
        # Use one level per sample so classic LHS uniqueness is observable.
        sample_count = 5
        problem = FakeIOHProblem([0, 0], [4, 4])
        create_algorithm(
            "LATIN_HYPERCUBE",
            {"max_evaluations": sample_count},
            random_seed=3,
        )(problem)
        design = np.asarray(problem.calls, dtype=int)
        self.assertEqual(design.shape, (sample_count, 2))
        for column in design.T:
            self.assertEqual(sorted(column.tolist()), list(range(sample_count)))
        # Independent permutations: columns should not be identical in general.
        self.assertFalse(np.array_equal(design[:, 0], design[:, 1]))

    def test_lhs_degenerate_dimension(self) -> None:
        problem = FakeIOHProblem([2, 0], [2, 3])
        create_algorithm(
            "LATIN_HYPERCUBE",
            {"max_evaluations": 6},
            random_seed=9,
        )(problem)
        self.assertEqual(len(problem.calls), 6)
        self.assertTrue(all(candidate[0] == 2 for candidate in problem.calls))
        second = [candidate[1] for candidate in problem.calls]
        self.assertTrue(all(0 <= value <= 3 for value in second))
        self.assertTrue(all(isinstance(value, int) for value in second))

    def test_lhs_exact_budget_and_no_best_re_evaluation(self) -> None:
        problem = FakeIOHProblem([0, 0], [3, 3])
        algo = create_algorithm(
            "LATIN_HYPERCUBE",
            {"max_evaluations": 7},
            random_seed=5,
        )
        algo(problem)
        call_count = len(problem.calls)
        self.assertEqual(call_count, 7)
        self.assertEqual(problem.state.evaluations, 7)
        self.assertIn(problem.state.current_best.x, problem.calls)
        self.assertEqual(len(problem.calls), call_count)
        self.assertEqual(problem.state.evaluations, call_count)

    def test_lhs_target_hit_uses_design_prefix(self) -> None:
        full = FakeIOHProblem([0, 0], [0, 0])
        create_algorithm(
            "LATIN_HYPERCUBE",
            {"max_evaluations": 6},
            random_seed=11,
        )(full)
        self.assertEqual(len(full.calls), 6)

        early = FakeIOHProblem([0, 0], [0, 0])
        controller = RunController(
            early,
            6,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "LATIN_HYPERCUBE",
            {"max_evaluations": 6},
            random_seed=11,
        )(early, run_controller=controller)
        self.assertEqual(len(early.calls), 1)
        self.assertEqual(early.calls, full.calls[:1])
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "target_hit")

    def test_lhs_target_miss_exact_ceiling(self) -> None:
        problem = FakeIOHProblem([2, 2], [4, 4])
        controller = RunController(
            problem,
            4,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "LATIN_HYPERCUBE",
            {"max_evaluations": 4},
            random_seed=12,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 4)
        self.assertFalse(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
