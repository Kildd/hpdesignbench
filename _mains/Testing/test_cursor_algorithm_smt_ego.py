"""
Phase SMT1 gate: smt package laziness, API contract, and SMT_EGO adapter.
"""
from __future__ import annotations

import importlib
import sys
import unittest

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version


class TestSmtPackageScaffold(unittest.TestCase):
    """Optional smt package remains lazy and API-compatible."""

    def test_smt_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name == "smt" or name.startswith("smt."):
                sys.modules.pop(name, None)
            if name.startswith("core.ioh_core.algorithms.smt."):
                sys.modules.pop(name, None)
        sys.modules.pop("core.ioh_core.algorithms.smt", None)

        mod = importlib.import_module("core.ioh_core.algorithms.smt")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn("smt", sys.modules)

    def test_smt_ego_api_contract_2141(self) -> None:
        try:
            installed = version("smt")
        except PackageNotFoundError as exc:
            self.fail(f"smt extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("2.14.1"))
        self.assertLess(parsed, Version("2.15"))

        import numpy as np
        from smt.applications import EGO
        from smt.design_space import DesignSpace, IntegerVariable
        from smt.surrogate_models import KRG

        design_space = DesignSpace(
            [IntegerVariable(0, 3), IntegerVariable(1, 4)],
            seed=0,
        )

        def fun(x):
            x = np.atleast_2d(x)
            return np.sum(x * x, axis=1, keepdims=True)

        ego = EGO(
            n_iter=2,
            n_doe=3,
            criterion="EI",
            surrogate=KRG(design_space=design_space, print_global=False),
            seed=0,
            n_parallel=1,
            verbose=False,
        )
        x_opt, y_opt, _, x_data, y_data = ego.optimize(fun=fun)
        self.assertEqual(x_data.shape[0], 5)
        self.assertEqual(y_data.shape[0], 5)
        self.assertEqual(x_opt.shape[-1], 2)


class TestSmtSpaceHelper(unittest.TestCase):
    """Integer design-space builder and n_doe default formula."""

    def test_default_n_doe_formula(self) -> None:
        from core.ioh_core.algorithms.smt.space import default_n_doe

        self.assertEqual(default_n_doe(1), 1)
        self.assertEqual(default_n_doe(2), 1)
        self.assertEqual(default_n_doe(10), 5)
        self.assertEqual(default_n_doe(50), 10)
        self.assertEqual(default_n_doe(100), 10)

    def test_build_integer_design_space_types(self) -> None:
        from smt.design_space import IntegerVariable

        from core.ioh_core.algorithms.smt.space import (
            build_integer_design_space,
        )

        design_space = build_integer_design_space([0, 2], [3, 4], seed=1)
        variables = design_space.design_variables
        self.assertEqual(len(variables), 2)
        self.assertIsInstance(variables[0], IntegerVariable)
        self.assertIsInstance(variables[1], IntegerVariable)
        self.assertEqual(variables[0].lower, 0)
        self.assertEqual(variables[0].upper, 3)
        self.assertEqual(variables[1].lower, 2)
        self.assertEqual(variables[1].upper, 4)

    def test_build_integer_design_space_rejects_degenerate(self) -> None:
        from core.ioh_core.algorithms.smt.space import (
            build_integer_design_space,
        )

        with self.assertRaises(ValueError):
            build_integer_design_space([0, 5], [3, 5], seed=0)

    def test_build_integer_design_space_rejects_inverted_bounds(self) -> None:
        from core.ioh_core.algorithms.smt.space import (
            build_integer_design_space,
        )

        with self.assertRaises(ValueError):
            build_integer_design_space([0, 3], [2, 1], seed=0)


class TestSmtEgoAdapter(unittest.TestCase):
    """Registered SMT_EGO adapter behaviour."""

    token = "SMT_EGO"

    def test_registry_metadata_and_n_doe_default(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )
        from core.ioh_core.algorithms.registry import RegistryError

        spec = get_algorithm_spec(self.token)
        self.assertEqual(spec.library, "smt")
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-smt")
        self.assertTrue(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations", "n_doe"],
        )
        normalized = normalize_algorithm_parameters(
            self.token,
            {"max_evaluations": 50},
        )
        self.assertEqual(normalized, {"max_evaluations": 50, "n_doe": 10})
        normalized_small = normalize_algorithm_parameters(
            self.token,
            {"max_evaluations": 10},
        )
        self.assertEqual(
            normalized_small,
            {"max_evaluations": 10, "n_doe": 5},
        )
        explicit = normalize_algorithm_parameters(
            self.token,
            {"max_evaluations": 10, "n_doe": 4},
        )
        self.assertEqual(explicit, {"max_evaluations": 10, "n_doe": 4})
        with self.assertRaises(RegistryError):
            normalize_algorithm_parameters(
                self.token,
                {"max_evaluations": 10, "n_doe": 11},
            )

    def test_respects_hard_budget(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            self.token,
            {"max_evaluations": 10, "n_doe": 4},
            random_seed=8,
        )(problem)
        self.assertEqual(len(problem.calls), 10)
        self.assertEqual(problem.state.evaluations, 10)

    def test_deterministic_with_seed(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem_a = FakeIOHProblem([0, 1], [3, 4])
        problem_b = FakeIOHProblem([0, 1], [3, 4])
        create_algorithm(
            self.token,
            {"max_evaluations": 12, "n_doe": 4},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            self.token,
            {"max_evaluations": 12, "n_doe": 4},
            random_seed=17,
        )(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)
        self.assertGreaterEqual(len(problem_a.calls), 1)
        self.assertTrue(
            all(isinstance(v, int) for c in problem_a.calls for v in c)
        )

    def test_different_seed_different_sequence(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem_a = FakeIOHProblem([0, 1], [5, 6])
        problem_b = FakeIOHProblem([0, 1], [5, 6])
        create_algorithm(
            self.token,
            {"max_evaluations": 16, "n_doe": 6},
            random_seed=3,
        )(problem_a)
        create_algorithm(
            self.token,
            {"max_evaluations": 16, "n_doe": 6},
            random_seed=99,
        )(problem_b)
        self.assertNotEqual(problem_a.calls, problem_b.calls)

    def test_does_not_re_evaluate_best(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])
        algo = create_algorithm(
            self.token,
            {"max_evaluations": 10, "n_doe": 4},
            random_seed=5,
        )
        algo(problem)
        call_count = len(problem.calls)
        self.assertEqual(problem.state.evaluations, call_count)
        self.assertLessEqual(call_count, 10)
        self.assertIn(problem.state.current_best.x, problem.calls)

    def test_missing_seed_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
        )

        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                self.token,
                {"max_evaluations": 5},
                random_seed=None,
            )

    def test_unknown_parameter_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
            validate_algorithm_parameters,
        )

        with self.assertRaises(AlgorithmFactoryError):
            validate_algorithm_parameters(
                self.token,
                {"max_evaluations": 5, "criterion": "SBO"},
                random_seed=1,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                self.token,
                {"max_evaluations": 5, "criterion": "SBO"},
                random_seed=1,
            )

    def test_constructor_ref_and_factory_seed(self) -> None:
        from core.ioh_core.algorithms.smt.factory import build_smt_ego
        from core.ioh_core.algorithms.registry import get_algorithm_spec

        spec = get_algorithm_spec(self.token)
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.smt.factory:build_smt_ego",
        )
        with self.assertRaises(ValueError):
            build_smt_ego(
                {"max_evaluations": 5, "n_doe": 2},
                random_seed=None,
            )

    def test_missing_dependency_uses_install_hint(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.smt.factory import build_smt_ego

        with patch(
            "core.ioh_core.algorithms.smt.factory."
            "import_optional_dependency",
            side_effect=AlgorithmDependencyError(
                "SMT_EGO requires optional dependency "
                "'smt'; install with:\n"
                'pip install -e ".[algorithms-smt]"'
            ),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                build_smt_ego(
                    {"max_evaluations": 5, "n_doe": 2},
                    random_seed=0,
                )
        self.assertIn(
            'pip install -e ".[algorithms-smt]"',
            str(ctx.exception),
        )


if __name__ == "__main__":
    unittest.main()
