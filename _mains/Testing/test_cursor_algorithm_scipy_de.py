"""
Phase C2.1 gate: SciPy differential evolution with integer integrality.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from core.ioh_core.algorithm_factory import (
    AlgorithmFactoryError,
    create_algorithm,
    validate_algorithm_parameters,
)
from core.ioh_core.algorithms import DifferentialEvolution
from core.ioh_core.algorithms.common import RunController
from core.ioh_core.algorithms.registry import (
    BudgetSemantics,
    CandidateMode,
    SeedPolicy,
    get_algorithm_spec,
    normalize_algorithm_parameters,
)
from core.ioh_core.algorithms.scipy.differential_evolution import (
    DifferentialEvolution as PackageDifferentialEvolution,
)
from core.ioh_core.algorithms.scipy.factory import (
    build_scipy_differential_evolution,
)
from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem


class TestScipyDifferentialEvolution(unittest.TestCase):
    """Registered SCIPY_DIFFERENTIAL_EVOLUTION adapter behaviour."""

    def test_registry_spec_and_defaults(self) -> None:
        spec = get_algorithm_spec("SCIPY_DIFFERENTIAL_EVOLUTION")
        self.assertEqual(spec.library, "scipy")
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.scipy.factory:"
            "build_scipy_differential_evolution",
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
            "SCIPY_DIFFERENTIAL_EVOLUTION",
            {"max_evaluations": 20},
        )
        self.assertEqual(normalized, {"max_evaluations": 20})

    def test_package_export_matches_algorithms_export(self) -> None:
        self.assertIs(DifferentialEvolution, PackageDifferentialEvolution)

    def test_calls_differential_evolution_with_locked_arguments(self) -> None:
        captured: dict = {}

        def _fake_de(func, bounds, **kwargs):  # noqa: ANN001
            captured["func"] = func
            captured["bounds"] = bounds
            captured["kwargs"] = kwargs
            # Drive a few objective calls so the adapter path is exercised.
            for point in ([0.0, 0.0], [1.0, 2.0], [0.0, 1.0]):
                try:
                    func(point)
                except Exception:
                    break
            return None

        problem = FakeIOHProblem([0, 0], [3, 4])
        with patch(
            "scipy.optimize.differential_evolution",
            _fake_de,
        ):
            algo = create_algorithm(
                "SCIPY_DIFFERENTIAL_EVOLUTION",
                {"max_evaluations": 8},
                random_seed=11,
            )
            result = algo(problem)

        self.assertIs(result, algo)
        self.assertEqual(captured["bounds"], [(0.0, 3.0), (0.0, 4.0)])
        kwargs = captured["kwargs"]
        self.assertEqual(kwargs["strategy"], "best1bin")
        self.assertEqual(kwargs["maxiter"], 8)
        self.assertEqual(kwargs["popsize"], 15)
        self.assertEqual(kwargs["tol"], 0.01)
        self.assertEqual(kwargs["mutation"], (0.5, 1))
        self.assertEqual(kwargs["recombination"], 0.7)
        self.assertEqual(kwargs["rng"], 11)
        self.assertIs(kwargs["polish"], False)
        self.assertEqual(kwargs["init"], "latinhypercube")
        self.assertEqual(kwargs["atol"], 0)
        self.assertEqual(kwargs["updating"], "immediate")
        self.assertEqual(kwargs["workers"], 1)
        self.assertIs(kwargs["integrality"], True)
        self.assertEqual(len(problem.calls), 3)
        self.assertTrue(
            all(isinstance(value, int) for call in problem.calls for value in call)
        )

    def test_exact_budget(self) -> None:
        problem = FakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            "SCIPY_DIFFERENTIAL_EVOLUTION",
            {"max_evaluations": 25},
            random_seed=8,
        )(problem)
        self.assertEqual(len(problem.calls), 25)
        self.assertEqual(problem.state.evaluations, 25)

    def test_same_seed_same_candidate_sequence(self) -> None:
        problem_a = FakeIOHProblem([0, 1], [3, 4])
        problem_b = FakeIOHProblem([0, 1], [3, 4])
        create_algorithm(
            "SCIPY_DIFFERENTIAL_EVOLUTION",
            {"max_evaluations": 20},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            "SCIPY_DIFFERENTIAL_EVOLUTION",
            {"max_evaluations": 20},
            random_seed=17,
        )(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)
        self.assertGreaterEqual(len(problem_a.calls), 1)
        self.assertTrue(
            all(isinstance(value, int) for call in problem_a.calls for value in call)
        )

    def test_different_seed_different_candidate_sequence(self) -> None:
        problem_a = FakeIOHProblem([0, 1], [5, 6])
        problem_b = FakeIOHProblem([0, 1], [5, 6])
        create_algorithm(
            "SCIPY_DIFFERENTIAL_EVOLUTION",
            {"max_evaluations": 30},
            random_seed=3,
        )(problem_a)
        create_algorithm(
            "SCIPY_DIFFERENTIAL_EVOLUTION",
            {"max_evaluations": 30},
            random_seed=99,
        )(problem_b)
        self.assertNotEqual(problem_a.calls, problem_b.calls)

    def test_does_not_re_evaluate_best(self) -> None:
        problem = FakeIOHProblem([0, 0], [2, 2])
        create_algorithm(
            "SCIPY_DIFFERENTIAL_EVOLUTION",
            {"max_evaluations": 18},
            random_seed=5,
        )(problem)
        call_count = len(problem.calls)
        self.assertEqual(problem.state.evaluations, call_count)
        self.assertEqual(call_count, 18)
        self.assertIn(problem.state.current_best.x, problem.calls)

    def test_integer_candidates_only(self) -> None:
        problem = FakeIOHProblem([0, 2], [4, 6])
        create_algorithm(
            "SCIPY_DIFFERENTIAL_EVOLUTION",
            {"max_evaluations": 16},
            random_seed=2,
        )(problem)
        for candidate in problem.calls:
            self.assertEqual(len(candidate), 2)
            self.assertTrue(all(isinstance(value, int) for value in candidate))
            self.assertGreaterEqual(candidate[0], 0)
            self.assertLessEqual(candidate[0], 4)
            self.assertGreaterEqual(candidate[1], 2)
            self.assertLessEqual(candidate[1], 6)

    def test_degenerate_dimension_stays_fixed(self) -> None:
        problem = FakeIOHProblem([2, 0], [2, 3])
        create_algorithm(
            "SCIPY_DIFFERENTIAL_EVOLUTION",
            {"max_evaluations": 12},
            random_seed=4,
        )(problem)
        self.assertTrue(all(candidate[0] == 2 for candidate in problem.calls))

    def test_target_hit_stops_early(self) -> None:
        problem = FakeIOHProblem([0, 0], [0, 0])
        controller = RunController(
            problem,
            20,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "SCIPY_DIFFERENTIAL_EVOLUTION",
            {"max_evaluations": 20},
            random_seed=4,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 1)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "target_hit")

    def test_target_miss_exact_budget(self) -> None:
        problem = FakeIOHProblem([2, 2], [4, 4])
        controller = RunController(
            problem,
            14,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "SCIPY_DIFFERENTIAL_EVOLUTION",
            {"max_evaluations": 14},
            random_seed=6,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 14)
        self.assertFalse(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")

    def test_missing_seed_rejected(self) -> None:
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "SCIPY_DIFFERENTIAL_EVOLUTION",
                {"max_evaluations": 5},
                random_seed=None,
            )
        with self.assertRaises(ValueError):
            build_scipy_differential_evolution(
                {"max_evaluations": 5},
                random_seed=None,
            )

    def test_unknown_parameter_rejected(self) -> None:
        with self.assertRaises(AlgorithmFactoryError):
            validate_algorithm_parameters(
                "SCIPY_DIFFERENTIAL_EVOLUTION",
                {"max_evaluations": 5, "popsize": 20},
                random_seed=1,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "SCIPY_DIFFERENTIAL_EVOLUTION",
                {"max_evaluations": 5, "popsize": 20},
                random_seed=1,
            )


if __name__ == "__main__":
    unittest.main()
