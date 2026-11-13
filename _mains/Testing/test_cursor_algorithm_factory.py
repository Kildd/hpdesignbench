"""
Phase A5 / ALG2 gate: strict algorithm factory validation and instance isolation.
"""
from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import nlopt

from core.ioh_core.algorithm_factory import (
    AlgorithmFactoryError,
    create_algorithm,
    validate_algorithm_parameters,
)
from core.ioh_core.algorithms import (
    NLoptDirectLocalSearch,
    RandomSearch,
)
from core.ioh_core.algorithms.native.random_search import (
    RandomSearch as NativeRandomSearch,
)
from core.ioh_core.algorithms.nlopt.algorithms import (
    NLoptDirectLocalSearch as PackageNLoptDirectLocalSearch,
)


class _FakeBounds:
    """Minimal inclusive integer bounds."""

    def __init__(self, lb: list[int], ub: list[int]) -> None:
        self.lb = lb
        self.ub = ub


class _FakeProblem:
    """Tiny IOH-like problem that records every candidate."""

    def __init__(self, lb: list[int], ub: list[int]) -> None:
        self.meta_data = SimpleNamespace(n_variables=len(lb))
        self.bounds = _FakeBounds(lb, ub)
        self.state = SimpleNamespace(evaluations=0)
        self.calls: list[list[int]] = []

    def __call__(self, x: Any) -> float:
        candidate = [int(v) for v in x]
        self.calls.append(candidate)
        self.state.evaluations += 1
        return float(sum(candidate))


class TestAlgorithmFactory(unittest.TestCase):
    """Supported algorithms, invalid definitions, and independent instances."""

    def test_create_random_search_and_direct_l(self) -> None:
        rs = create_algorithm(
            "RANDOM_SEARCH",
            {"max_evaluations": 7},
            random_seed=123,
        )
        direct = create_algorithm(
            "NLOPT_DIRECT_L",
            {"max_evaluations": 11},
            random_seed=None,
        )
        self.assertIsInstance(rs, RandomSearch)
        self.assertEqual(rs.evaluations, 7)
        self.assertIsInstance(direct, NLoptDirectLocalSearch)
        self.assertEqual(direct.max_evaluations, 11)

    def test_independent_algorithm_instances(self) -> None:
        first = create_algorithm(
            "RANDOM_SEARCH",
            {"max_evaluations": 3},
            random_seed=1,
        )
        second = create_algorithm(
            "RANDOM_SEARCH",
            {"max_evaluations": 3},
            random_seed=1,
        )
        self.assertIsNot(first, second)
        self.assertIsNot(first.rng, second.rng)

        d1 = create_algorithm("NLOPT_DIRECT_L", {"max_evaluations": 4})
        d2 = create_algorithm("NLOPT_DIRECT_L", {"max_evaluations": 4})
        self.assertIsNot(d1, d2)

    def test_invalid_definitions_are_rejected(self) -> None:
        with self.assertRaisesRegex(AlgorithmFactoryError, "unknown algorithm"):
            create_algorithm("GA", {"max_evaluations": 1}, random_seed=0)

        with self.assertRaisesRegex(AlgorithmFactoryError, "unknown algorithm"):
            create_algorithm(
                "SCIPY_SOBOL",
                {"max_evaluations": 1},
                random_seed=0,
            )

        with self.assertRaisesRegex(AlgorithmFactoryError, "unknown parameters"):
            create_algorithm(
                "RANDOM_SEARCH",
                {"max_evaluations": 1, "population": 10},
                random_seed=0,
            )

        with self.assertRaisesRegex(AlgorithmFactoryError, "max_evaluations"):
            create_algorithm("RANDOM_SEARCH", {}, random_seed=0)

        with self.assertRaisesRegex(AlgorithmFactoryError, "requires a random_seed"):
            validate_algorithm_parameters(
                "RANDOM_SEARCH",
                {"max_evaluations": 1},
                random_seed=None,
            )

        with self.assertRaisesRegex(AlgorithmFactoryError, "seedless"):
            create_algorithm(
                "NLOPT_DIRECT_L",
                {"max_evaluations": 1},
                random_seed=99,
            )

    def test_random_search_compatibility_export_and_signature(self) -> None:
        self.assertIs(RandomSearch, NativeRandomSearch)
        signature = inspect.signature(RandomSearch.__init__)
        self.assertEqual(
            list(signature.parameters),
            ["self", "max_evaluations", "seed"],
        )
        algo = RandomSearch(max_evaluations=5, seed=0)
        self.assertEqual(algo.name, "RANDOM_SEARCH")
        self.assertTrue(algo.info)
        self.assertEqual(algo.evaluations, 5)
        self.assertIsNotNone(algo.rng)

    def test_random_search_same_seed_candidate_sequence(self) -> None:
        problem_a = _FakeProblem([0, 1], [3, 4])
        problem_b = _FakeProblem([0, 1], [3, 4])
        first = RandomSearch(max_evaluations=8, seed=42)
        second = RandomSearch(max_evaluations=8, seed=42)
        first(problem_a)
        second(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)
        self.assertEqual(first.best_x_, second.best_x_)
        self.assertEqual(first.best_y_, second.best_y_)

    def test_random_search_different_seed_candidate_sequence(self) -> None:
        problem_a = _FakeProblem([0, 1], [3, 4])
        problem_b = _FakeProblem([0, 1], [3, 4])
        RandomSearch(max_evaluations=8, seed=1)(problem_a)
        RandomSearch(max_evaluations=8, seed=2)(problem_b)
        self.assertNotEqual(problem_a.calls, problem_b.calls)

    def test_direct_l_candidates_are_integer_and_bounded(self) -> None:
        self.assertIs(NLoptDirectLocalSearch, PackageNLoptDirectLocalSearch)
        problem = _FakeProblem([0, 2], [4, 6])
        f_best, x_best = NLoptDirectLocalSearch(max_evaluations=12)(problem)
        self.assertIsInstance(f_best, float)
        self.assertEqual(len(x_best), 2)
        self.assertTrue(all(isinstance(value, int) for value in x_best))
        self.assertGreaterEqual(len(problem.calls), 1)
        for candidate in problem.calls:
            self.assertEqual(len(candidate), 2)
            self.assertTrue(all(isinstance(value, int) for value in candidate))
            self.assertGreaterEqual(candidate[0], 0)
            self.assertLessEqual(candidate[0], 4)
            self.assertGreaterEqual(candidate[1], 2)
            self.assertLessEqual(candidate[1], 6)
        self.assertIn(x_best, problem.calls)

    def test_direct_l_roundoff_uses_tracked_evaluated_best(self) -> None:
        problem = _FakeProblem([0, 0], [5, 5])
        real_opt = nlopt.opt

        class _RoundoffOpt:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._inner = real_opt(*args, **kwargs)
                self._objective = None

            def set_lower_bounds(self, *args: Any, **kwargs: Any) -> None:
                self._inner.set_lower_bounds(*args, **kwargs)

            def set_upper_bounds(self, *args: Any, **kwargs: Any) -> None:
                self._inner.set_upper_bounds(*args, **kwargs)

            def set_min_objective(self, objective: Any) -> None:
                self._objective = objective
                self._inner.set_min_objective(objective)

            def set_maxeval(self, value: int) -> None:
                self._inner.set_maxeval(value)

            def set_stopval(self, value: float) -> None:
                self._inner.set_stopval(value)

            def optimize(self, x0: Any) -> list[float]:
                assert self._objective is not None
                self._objective([1.4, 2.6], None)
                self._objective([0.2, 0.8], None)
                raise nlopt.RoundoffLimited()

            def last_optimize_result(self) -> int:
                # Status-like integer; must never be interpreted as a vector.
                return 42

            def last_optimum_value(self) -> float:
                return -12345.0

        with patch("core.ioh_core.algorithms.nlopt.algorithms.nlopt.opt", _RoundoffOpt):
            f_best, x_best = NLoptDirectLocalSearch(max_evaluations=10)(problem)

        self.assertEqual(problem.calls, [[1, 3], [0, 1]])
        self.assertEqual(x_best, [0, 1])
        self.assertEqual(f_best, 1.0)
        self.assertNotEqual(x_best, [42])

    def test_direct_l_does_not_re_evaluate_best(self) -> None:
        problem = _FakeProblem([0, 0], [3, 3])
        algo = NLoptDirectLocalSearch(max_evaluations=9)
        f_best, x_best = algo(problem)
        call_count = len(problem.calls)
        self.assertEqual(call_count, algo.n_optimizer_evaluations_)
        self.assertIn(x_best, problem.calls)
        self.assertEqual(f_best, float(sum(x_best)))
        self.assertEqual(len(problem.calls), call_count)

    def test_factory_translates_registry_validation_errors(self) -> None:
        with self.assertRaises(AlgorithmFactoryError) as ctx:
            create_algorithm("GA", {"max_evaluations": 1}, random_seed=0)
        self.assertIn("unknown algorithm", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, Exception)
        self.assertEqual(type(ctx.exception.__cause__).__name__, "RegistryError")

        with self.assertRaises(AlgorithmFactoryError) as ctx:
            validate_algorithm_parameters(
                "RANDOM_SEARCH",
                {"max_evaluations": 1},
                random_seed=None,
            )
        self.assertIn("requires a random_seed", str(ctx.exception))
        self.assertEqual(type(ctx.exception.__cause__).__name__, "RegistryError")

    def test_factory_does_not_translate_dependency_errors(self) -> None:
        class AlgorithmDependencyError(ImportError):
            """Stand-in until ALG4.1 adds the real error type."""

        with patch(
            "core.ioh_core.algorithm_factory.construct_algorithm",
            side_effect=AlgorithmDependencyError("missing optional dependency"),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                create_algorithm(
                    "RANDOM_SEARCH",
                    {"max_evaluations": 1},
                    random_seed=0,
                )
        self.assertNotIsInstance(ctx.exception, AlgorithmFactoryError)
        self.assertIn("missing optional dependency", str(ctx.exception))

    def test_user_parameters_cannot_override_max_evaluations(self) -> None:
        import textwrap
        from tempfile import TemporaryDirectory

        from core.execution_core.config import AlgorithmConfig, load_experiment_config
        from core.ioh_core.workflows.registry import _validate_optimization

        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "cfg.toml"
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
            config = load_experiment_config(config_path)
            algo = config.experiment.algorithms[0]
            self.assertIsInstance(algo, AlgorithmConfig)
            self.assertIn("max_evaluations", algo.parameters)
            with self.assertRaises(AlgorithmFactoryError) as ctx:
                _validate_optimization(config, tasks=())
            self.assertIn("max_evaluations", str(ctx.exception))
            self.assertIn("must not include", str(ctx.exception))

    def test_worker_constructs_new_registry_algorithm_without_branch(self) -> None:
        """Factory constructs a patched registry algorithm with no name branches."""
        import inspect
        from unittest.mock import patch

        from core.ioh_core.algorithms.registry import (
            AlgorithmSpec,
            BudgetSemantics,
            CandidateMode,
            ParameterSpec,
            SeedPolicy,
            create_registry,
        )
        import core.ioh_core.workflows.optimization_runs.worker as worker_mod

        algorithm_name = "TOY_FACTORY_ALGO"
        self.assertNotIn(algorithm_name, inspect.getsource(worker_mod))

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
                    description="toy algorithm for factory registry proof",
                ),
            )
        )
        with patch(
            "core.ioh_core.algorithms.registry.ALGORITHM_REGISTRY",
            toy_registry,
        ), patch(
            "core.ioh_core.algorithm_factory.ALGORITHM_REGISTRY",
            toy_registry,
        ):
            algo = create_algorithm(
                algorithm_name,
                {"max_evaluations": 4},
                random_seed=11,
            )
        self.assertEqual(algo.name, "RANDOM_SEARCH")
        self.assertEqual(algo.evaluations, 4)


if __name__ == "__main__":
    unittest.main()
