"""
Phase SP1 gate: starpso package laziness, API contract, and IntegerPSO adapter.
"""
from __future__ import annotations

import importlib
import sys
import unittest

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version


class TestStarPsoPackageScaffold(unittest.TestCase):
    """Optional starpso package remains lazy and API-compatible."""

    def test_starpso_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name == "star_pso" or name.startswith("star_pso."):
                sys.modules.pop(name, None)
            if name.startswith("core.ioh_core.algorithms.starpso."):
                sys.modules.pop(name, None)
        sys.modules.pop("core.ioh_core.algorithms.starpso", None)

        mod = importlib.import_module("core.ioh_core.algorithms.starpso")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn("star_pso", sys.modules)

    def test_starpso_integer_pso_api_contract_013(self) -> None:
        try:
            installed = version("starpso")
        except PackageNotFoundError as exc:
            self.fail(f"starpso extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("0.1.3"))
        self.assertLess(parsed, Version("0.2"))

        import numpy as np
        from star_pso.engines.generic_pso import GenericPSO
        from star_pso.engines.integer_pso import IntegerPSO
        from star_pso.population.particle import Particle
        from star_pso.population.swarm import Swarm
        from star_pso.utils.auxiliary import cost_function

        self.assertTrue(callable(GenericPSO.set_seed))
        self.assertTrue(callable(cost_function))

        @cost_function(minimize=True)
        def sphere(x, **_kwargs):
            return float(np.sum(np.asarray(x) ** 2))

        if "rng" in IntegerPSO.__dict__:
            delattr(IntegerPSO, "rng")
        GenericPSO.set_seed(0)
        lb = np.array([0, 0], dtype=int)
        ub = np.array([3, 3], dtype=int)
        positions = GenericPSO.rng.integers(
            low=lb, high=ub, endpoint=True, size=(4, 2), dtype=int
        )
        swarm = Swarm([Particle(row) for row in positions])
        optimizer = IntegerPSO(
            x_min=lb,
            x_max=ub,
            initial_swarm=swarm,
            obj_func=sphere,
            n_cpus=1,
        )
        optimizer.run(
            max_it=2,
            options={"w0": 0.70, "c1": 1.50, "c2": 1.50, "mode": "g_best"},
            parallel=False,
            adapt_params=False,
            verbose=False,
        )
        _i_opt, f_opt, x_opt = optimizer.get_optimal_values()
        self.assertTrue(np.isfinite(f_opt))
        self.assertEqual(len(x_opt), 2)


class TestPyStarPsoIntegerPso(unittest.TestCase):
    """Registered PYSTARPSO_INTEGER_PSO adapter behaviour."""

    def test_pystarpso_integer_pso_registry_metadata(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec("PYSTARPSO_INTEGER_PSO")
        self.assertEqual(spec.library, "starpso")
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.starpso.factory:"
            "build_pystarpso_integer_pso",
        )
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-starpso")
        self.assertTrue(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations", "n_pop"],
        )
        normalized = normalize_algorithm_parameters(
            "PYSTARPSO_INTEGER_PSO",
            {"max_evaluations": 10},
        )
        self.assertEqual(
            normalized,
            {
                "max_evaluations": 10,
                "n_pop": 50,
            },
        )

    def test_pystarpso_integer_pso_respects_hard_budget(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            "PYSTARPSO_INTEGER_PSO",
            {"max_evaluations": 10, "n_pop": 4},
            random_seed=8,
        )(problem)
        self.assertEqual(len(problem.calls), 10)
        self.assertEqual(problem.state.evaluations, 10)

    def test_pystarpso_integer_pso_deterministic_with_seed(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem_a = FakeIOHProblem([0, 1], [3, 4])
        problem_b = FakeIOHProblem([0, 1], [3, 4])
        create_algorithm(
            "PYSTARPSO_INTEGER_PSO",
            {"max_evaluations": 12, "n_pop": 4},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            "PYSTARPSO_INTEGER_PSO",
            {"max_evaluations": 12, "n_pop": 4},
            random_seed=17,
        )(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)
        self.assertGreaterEqual(len(problem_a.calls), 1)
        self.assertTrue(
            all(isinstance(v, int) for c in problem_a.calls for v in c)
        )

    def test_pystarpso_integer_pso_different_seed_different_sequence(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem_a = FakeIOHProblem([0, 1], [5, 6])
        problem_b = FakeIOHProblem([0, 1], [5, 6])
        create_algorithm(
            "PYSTARPSO_INTEGER_PSO",
            {"max_evaluations": 20, "n_pop": 4},
            random_seed=3,
        )(problem_a)
        create_algorithm(
            "PYSTARPSO_INTEGER_PSO",
            {"max_evaluations": 20, "n_pop": 4},
            random_seed=99,
        )(problem_b)
        self.assertNotEqual(problem_a.calls, problem_b.calls)

    def test_pystarpso_integer_pso_does_not_re_evaluate_best(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])
        algo = create_algorithm(
            "PYSTARPSO_INTEGER_PSO",
            {"max_evaluations": 10, "n_pop": 4},
            random_seed=5,
        )
        algo(problem)
        call_count = len(problem.calls)
        self.assertEqual(problem.state.evaluations, call_count)
        self.assertLessEqual(call_count, 10)
        self.assertIn(problem.state.current_best.x, problem.calls)
        self.assertEqual(len(problem.calls), call_count)
        self.assertEqual(problem.state.evaluations, call_count)

    def test_pystarpso_integer_pso_missing_seed_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
        )
        from core.ioh_core.algorithms.starpso.factory import (
            build_pystarpso_integer_pso,
        )

        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "PYSTARPSO_INTEGER_PSO",
                {"max_evaluations": 5, "n_pop": 4},
                random_seed=None,
            )
        with self.assertRaises(ValueError):
            build_pystarpso_integer_pso(
                {"max_evaluations": 5, "n_pop": 4},
                random_seed=None,
            )

    def test_pystarpso_integer_pso_unknown_parameter_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
            validate_algorithm_parameters,
        )

        with self.assertRaises(AlgorithmFactoryError):
            validate_algorithm_parameters(
                "PYSTARPSO_INTEGER_PSO",
                {
                    "max_evaluations": 5,
                    "n_pop": 4,
                    "w0": 0.5,
                },
                random_seed=1,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "PYSTARPSO_INTEGER_PSO",
                {
                    "max_evaluations": 5,
                    "n_pop": 4,
                    "w0": 0.5,
                },
                random_seed=1,
            )

    def test_pystarpso_integer_pso_missing_dependency_uses_install_hint(
        self,
    ) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.starpso.factory import (
            build_pystarpso_integer_pso,
        )

        with patch(
            "core.ioh_core.algorithms.starpso.factory."
            "import_optional_dependency",
            side_effect=AlgorithmDependencyError(
                "PYSTARPSO_INTEGER_PSO requires optional dependency "
                "'star_pso'; install with:\n"
                'pip install -e ".[algorithms-starpso]"'
            ),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                build_pystarpso_integer_pso(
                    {"max_evaluations": 5, "n_pop": 4},
                    random_seed=0,
                )
        self.assertIn(
            'pip install -e ".[algorithms-starpso]"',
            str(ctx.exception),
        )


if __name__ == "__main__":
    unittest.main()
