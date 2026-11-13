"""
Phase L1 gate: LEAP package laziness, API contract, and LEAP_GA adapter.
"""
from __future__ import annotations

import importlib
import sys
import unittest

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version


_LOCKED_LEAP_SYMBOLS = (
    ("leap_ec.algorithm", "generational_ea"),
    ("leap_ec.problem", "ScalarProblem"),
    ("leap_ec.representation", "Representation"),
    ("leap_ec.decoder", "IdentityDecoder"),
    ("leap_ec.int_rep.initializers", "create_int_vector"),
    ("leap_ec.int_rep.ops", "mutate_randint"),
    ("leap_ec.ops", "tournament_selection"),
    ("leap_ec.ops", "clone"),
    ("leap_ec.ops", "UniformCrossover"),
    ("leap_ec.ops", "evaluate"),
    ("leap_ec.ops", "pool"),
)


class TestLeapPackageScaffold(unittest.TestCase):
    """Optional LEAP package remains lazy and API-compatible."""

    def test_leap_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name == "leap_ec" or name.startswith("leap_ec."):
                sys.modules.pop(name, None)
            if name.startswith("core.ioh_core.algorithms.leap."):
                sys.modules.pop(name, None)
        sys.modules.pop("core.ioh_core.algorithms.leap", None)

        mod = importlib.import_module("core.ioh_core.algorithms.leap")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn("leap_ec", sys.modules)

    def test_leap_api_contract_082(self) -> None:
        try:
            installed = version("leap-ec")
        except PackageNotFoundError as exc:
            self.fail(f"leap-ec extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("0.8.2"))
        self.assertLess(parsed, Version("0.9"))

        for module_name, attr_name in _LOCKED_LEAP_SYMBOLS:
            module = importlib.import_module(module_name)
            self.assertTrue(
                hasattr(module, attr_name),
                msg=f"missing locked symbol {module_name}:{attr_name}",
            )
            self.assertTrue(callable(getattr(module, attr_name)))

        from leap_ec.int_rep.initializers import create_int_vector
        from leap_ec.ops import UniformCrossover

        initializer = create_int_vector([(0, 3), (1, 4)])
        genome = initializer()
        self.assertEqual(len(genome), 2)
        self.assertGreaterEqual(int(genome[0]), 0)
        self.assertLessEqual(int(genome[0]), 3)
        self.assertGreaterEqual(int(genome[1]), 1)
        self.assertLessEqual(int(genome[1]), 4)

        crossover = UniformCrossover(p_swap=0.2)
        self.assertEqual(crossover.p_swap, 0.2)


class TestLeapIOHIntegerProblem(unittest.TestCase):
    """IOHLeapIntegerProblem bounds, validation, and budget sentinel."""

    def test_leap_problem_declares_minimization_and_inclusive_bounds(self) -> None:
        from core.ioh_core.algorithms.leap.problem import IOHLeapIntegerProblem
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 2], [4, 6])
        adapter = IOHLeapIntegerProblem(problem, max_evaluations=5)
        self.assertFalse(adapter.maximize)
        self.assertEqual(adapter.lower_bounds, [0, 2])
        self.assertEqual(adapter.upper_bounds, [4, 6])
        self.assertEqual(adapter.bounds, [(0, 4), (2, 6)])

    def test_leap_problem_validates_candidates_before_ioh(self) -> None:
        from core.ioh_core.algorithms.common import CandidateError
        from core.ioh_core.algorithms.leap.problem import IOHLeapIntegerProblem
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])
        adapter = IOHLeapIntegerProblem(problem, max_evaluations=3)
        with self.assertRaises(CandidateError):
            adapter.evaluate([0.5, 1])
        self.assertEqual(problem.calls, [])
        self.assertEqual(problem.state.evaluations, 0)

        with self.assertRaises(CandidateError):
            adapter.evaluate([0, 9])
        self.assertEqual(problem.calls, [])

    def test_leap_problem_uses_finite_sentinel_after_budget(self) -> None:
        import numpy as np

        from core.ioh_core.algorithms.leap.problem import (
            LEAP_BUDGET_EXHAUSTED_FITNESS,
            IOHLeapIntegerProblem,
        )
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0], [3])
        adapter = IOHLeapIntegerProblem(problem, max_evaluations=2)
        self.assertEqual(adapter.evaluate([1]), 1.0)
        self.assertEqual(adapter.evaluate([2]), 4.0)
        self.assertEqual(problem.state.evaluations, 2)

        sentinel = adapter.evaluate([0])
        self.assertEqual(sentinel, LEAP_BUDGET_EXHAUSTED_FITNESS)
        self.assertEqual(problem.state.evaluations, 2)
        self.assertEqual(len(problem.calls), 2)
        self.assertTrue(np.isfinite(sentinel))
        self.assertNotEqual(sentinel, float("inf"))

    def test_leap_problem_population_larger_than_budget(self) -> None:
        from core.ioh_core.algorithms.leap.problem import (
            LEAP_BUDGET_EXHAUSTED_FITNESS,
            IOHLeapIntegerProblem,
        )
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])
        budget = 3
        adapter = IOHLeapIntegerProblem(problem, max_evaluations=budget)
        fitnesses: list[float] = []
        for candidate in (
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 2],
            [2, 0],
        ):
            fitnesses.append(float(adapter.evaluate(candidate)))

        self.assertEqual(problem.state.evaluations, budget)
        self.assertEqual(len(problem.calls), budget)
        self.assertEqual(
            fitnesses[budget:],
            [LEAP_BUDGET_EXHAUSTED_FITNESS] * (len(fitnesses) - budget),
        )


class TestLeapGeneticAlgorithm(unittest.TestCase):
    """Registered LEAP_GA adapter behaviour."""

    def test_leap_ga_registry_spec(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec("LEAP_GA")
        self.assertEqual(spec.library, "leap")
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.leap.factory:build_leap_ga",
        )
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-leap")
        self.assertTrue(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations", "population_size"],
        )
        normalized = normalize_algorithm_parameters(
            "LEAP_GA",
            {"max_evaluations": 10},
        )
        self.assertEqual(
            normalized,
            {
                "max_evaluations": 10,
                "population_size": 50,
            },
        )

    def test_leap_ga_uses_locked_integer_pipeline(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithm_factory import create_algorithm
        from core.ioh_core.algorithms.leap import algorithms as leap_algorithms
        from leap_ec.ops import UniformCrossover
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        captured: dict = {}

        def _fake_generational_ea(**kwargs):
            captured["kwargs"] = kwargs
            return []

        with patch.object(
            leap_algorithms,
            "generational_ea",
            side_effect=_fake_generational_ea,
        ):
            algo = create_algorithm(
                "LEAP_GA",
                {"max_evaluations": 8, "population_size": 4},
                random_seed=3,
            )
            algo(FakeIOHProblem([0, 0], [2, 2]))

        kwargs = captured["kwargs"]
        self.assertEqual(kwargs["max_generations"], float("inf"))
        self.assertEqual(kwargs["pop_size"], 4)
        self.assertEqual(kwargs["k_elites"], 1)
        self.assertTrue(callable(kwargs["stop"]))
        self.assertFalse(kwargs["problem"].maximize)
        self.assertEqual(kwargs["problem"].bounds, [(0, 2), (0, 2)])

        pipeline = kwargs["pipeline"]
        self.assertIs(pipeline[0], leap_algorithms.tournament_selection)
        self.assertIs(pipeline[1], leap_algorithms.clone)
        self.assertTrue(callable(pipeline[2]))
        self.assertIsInstance(pipeline[3], UniformCrossover)
        self.assertEqual(pipeline[3].p_swap, 0.2)
        self.assertIs(pipeline[4], leap_algorithms.evaluate)
        self.assertTrue(callable(pipeline[5]))

    def test_leap_ga_same_seed_same_candidate_sequence(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem_a = FakeIOHProblem([0, 1], [3, 4])
        problem_b = FakeIOHProblem([0, 1], [3, 4])
        create_algorithm(
            "LEAP_GA",
            {"max_evaluations": 12, "population_size": 4},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            "LEAP_GA",
            {"max_evaluations": 12, "population_size": 4},
            random_seed=17,
        )(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)
        self.assertGreaterEqual(len(problem_a.calls), 1)
        self.assertTrue(
            all(isinstance(v, int) for c in problem_a.calls for v in c)
        )

    def test_leap_ga_different_seed_different_candidate_sequence(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem_a = FakeIOHProblem([0, 1], [5, 6])
        problem_b = FakeIOHProblem([0, 1], [5, 6])
        create_algorithm(
            "LEAP_GA",
            {"max_evaluations": 20, "population_size": 4},
            random_seed=3,
        )(problem_a)
        create_algorithm(
            "LEAP_GA",
            {"max_evaluations": 20, "population_size": 4},
            random_seed=99,
        )(problem_b)
        self.assertNotEqual(problem_a.calls, problem_b.calls)

    def test_leap_ga_exact_budget(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            "LEAP_GA",
            {"max_evaluations": 10, "population_size": 4},
            random_seed=8,
        )(problem)
        self.assertEqual(len(problem.calls), 10)
        self.assertEqual(problem.state.evaluations, 10)

    def test_leap_ga_does_not_re_evaluate_best(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])
        algo = create_algorithm(
            "LEAP_GA",
            {"max_evaluations": 10, "population_size": 4},
            random_seed=5,
        )
        algo(problem)
        call_count = len(problem.calls)
        self.assertEqual(problem.state.evaluations, call_count)
        self.assertLessEqual(call_count, 10)
        self.assertIn(problem.state.current_best.x, problem.calls)
        self.assertEqual(len(problem.calls), call_count)
        self.assertEqual(problem.state.evaluations, call_count)

    def test_leap_ga_missing_seed_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
        )
        from core.ioh_core.algorithms.leap.factory import build_leap_ga

        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "LEAP_GA",
                {"max_evaluations": 5, "population_size": 4},
                random_seed=None,
            )
        with self.assertRaises(ValueError):
            build_leap_ga(
                {"max_evaluations": 5, "population_size": 4},
                random_seed=None,
            )

    def test_leap_ga_unknown_parameter_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
            validate_algorithm_parameters,
        )

        with self.assertRaises(AlgorithmFactoryError):
            validate_algorithm_parameters(
                "LEAP_GA",
                {
                    "max_evaluations": 5,
                    "population_size": 4,
                    "tournament_k": 3,
                },
                random_seed=1,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "LEAP_GA",
                {
                    "max_evaluations": 5,
                    "population_size": 4,
                    "tournament_k": 3,
                },
                random_seed=1,
            )

    def test_leap_ga_missing_dependency_uses_install_hint(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.leap.factory import build_leap_ga

        with patch(
            "core.ioh_core.algorithms.leap.factory.import_optional_dependency",
            side_effect=AlgorithmDependencyError(
                "LEAP_GA requires optional dependency 'leap_ec'; install with:\n"
                'pip install -e ".[algorithms-leap]"'
            ),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                build_leap_ga(
                    {"max_evaluations": 5, "population_size": 4},
                    random_seed=0,
                )
        self.assertIn(
            'pip install -e ".[algorithms-leap]"',
            str(ctx.exception),
        )


if __name__ == "__main__":
    unittest.main()
