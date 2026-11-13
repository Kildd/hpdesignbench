"""
Phase NG1 gate: nevergrad package laziness, API contract, and MetaModel adapters.
"""
from __future__ import annotations

import importlib
import sys
import unittest

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version


class TestNevergradPackageScaffold(unittest.TestCase):
    """Optional nevergrad package remains lazy and API-compatible."""

    def test_nevergrad_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name == "nevergrad" or name.startswith("nevergrad."):
                sys.modules.pop(name, None)
            if name.startswith("core.ioh_core.algorithms.nevergrad."):
                sys.modules.pop(name, None)
        sys.modules.pop("core.ioh_core.algorithms.nevergrad", None)

        mod = importlib.import_module("core.ioh_core.algorithms.nevergrad")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn("nevergrad", sys.modules)

    def test_nevergrad_metamodel_api_contract_1012(self) -> None:
        try:
            installed = version("nevergrad")
        except PackageNotFoundError as exc:
            self.fail(f"nevergrad extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("1.0.12"))
        self.assertLess(parsed, Version("1.1"))

        import nevergrad as ng
        from nevergrad.optimization import registry

        self.assertIn("MetaModel", registry)
        self.assertIn("MetaModelOnePlusOne", registry)

        parametrization = ng.p.Tuple(
            ng.p.TransitionChoice([0, 1, 2]),
            ng.p.TransitionChoice([1, 2, 3]),
        )
        parametrization.random_state.seed(0)
        optimizer = registry["MetaModel"](
            parametrization=parametrization,
            budget=4,
            num_workers=1,
        )
        candidate = optimizer.ask()
        values = [int(v) for v in candidate.value]
        self.assertEqual(len(values), 2)
        optimizer.tell(candidate, float(sum(v * v for v in values)))


class TestNevergradSpaceHelper(unittest.TestCase):
    """TransitionChoice catalogue space builder."""

    def test_build_transition_choice_tuple_cardinalities(self) -> None:
        from core.ioh_core.algorithms.nevergrad.space import (
            build_transition_choice_tuple,
        )

        parametrization = build_transition_choice_tuple([0, 2], [3, 4])
        self.assertEqual(len(parametrization), 2)
        self.assertEqual(len(parametrization[0].choices), 4)
        self.assertEqual(len(parametrization[1].choices), 3)

    def test_build_transition_choice_tuple_rejects_inverted_bounds(self) -> None:
        from core.ioh_core.algorithms.nevergrad.space import (
            build_transition_choice_tuple,
        )

        with self.assertRaises(ValueError):
            build_transition_choice_tuple([0, 3], [2, 1])


class _MetaModelAdapterTests:
    """Shared behaviour checks for both Nevergrad MetaModel tokens."""

    token: str

    def test_registry_metadata(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec(self.token)
        self.assertEqual(spec.library, "nevergrad")
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-nevergrad")
        self.assertFalse(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations"],
        )
        normalized = normalize_algorithm_parameters(
            self.token,
            {"max_evaluations": 10},
        )
        self.assertEqual(normalized, {"max_evaluations": 10})

    def test_respects_hard_budget(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            self.token,
            {"max_evaluations": 10},
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
            {"max_evaluations": 12},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            self.token,
            {"max_evaluations": 12},
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
            {"max_evaluations": 20},
            random_seed=3,
        )(problem_a)
        create_algorithm(
            self.token,
            {"max_evaluations": 20},
            random_seed=99,
        )(problem_b)
        self.assertNotEqual(problem_a.calls, problem_b.calls)

    def test_does_not_re_evaluate_best(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])
        algo = create_algorithm(
            self.token,
            {"max_evaluations": 10},
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
                {"max_evaluations": 5, "frequency_ratio": 0.5},
                random_seed=1,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                self.token,
                {"max_evaluations": 5, "frequency_ratio": 0.5},
                random_seed=1,
            )


class TestNevergradMetaModel(_MetaModelAdapterTests, unittest.TestCase):
    """Registered NEVERGRAD_METAMODEL adapter behaviour."""

    token = "NEVERGRAD_METAMODEL"

    def test_constructor_ref_and_factory_seed(self) -> None:
        from core.ioh_core.algorithms.nevergrad.factory import (
            build_nevergrad_metamodel,
        )
        from core.ioh_core.algorithms.registry import get_algorithm_spec

        spec = get_algorithm_spec(self.token)
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.nevergrad.factory:"
            "build_nevergrad_metamodel",
        )
        with self.assertRaises(ValueError):
            build_nevergrad_metamodel({"max_evaluations": 5}, random_seed=None)

    def test_missing_dependency_uses_install_hint(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.nevergrad.factory import (
            build_nevergrad_metamodel,
        )

        with patch(
            "core.ioh_core.algorithms.nevergrad.factory."
            "import_optional_dependency",
            side_effect=AlgorithmDependencyError(
                "NEVERGRAD_METAMODEL requires optional dependency "
                "'nevergrad'; install with:\n"
                'pip install -e ".[algorithms-nevergrad]"'
            ),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                build_nevergrad_metamodel(
                    {"max_evaluations": 5},
                    random_seed=0,
                )
        self.assertIn(
            'pip install -e ".[algorithms-nevergrad]"',
            str(ctx.exception),
        )


class TestNevergradMetaModelOnePlusOne(_MetaModelAdapterTests, unittest.TestCase):
    """Registered NEVERGRAD_METAMODEL_ONEPLUSONE adapter behaviour."""

    token = "NEVERGRAD_METAMODEL_ONEPLUSONE"

    def test_constructor_ref_and_factory_seed(self) -> None:
        from core.ioh_core.algorithms.nevergrad.factory import (
            build_nevergrad_metamodel_oneplusone,
        )
        from core.ioh_core.algorithms.registry import get_algorithm_spec

        spec = get_algorithm_spec(self.token)
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.nevergrad.factory:"
            "build_nevergrad_metamodel_oneplusone",
        )
        with self.assertRaises(ValueError):
            build_nevergrad_metamodel_oneplusone(
                {"max_evaluations": 5},
                random_seed=None,
            )

    def test_missing_dependency_uses_install_hint(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.nevergrad.factory import (
            build_nevergrad_metamodel_oneplusone,
        )

        with patch(
            "core.ioh_core.algorithms.nevergrad.factory."
            "import_optional_dependency",
            side_effect=AlgorithmDependencyError(
                "NEVERGRAD_METAMODEL_ONEPLUSONE requires optional dependency "
                "'nevergrad'; install with:\n"
                'pip install -e ".[algorithms-nevergrad]"'
            ),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                build_nevergrad_metamodel_oneplusone(
                    {"max_evaluations": 5},
                    random_seed=0,
                )
        self.assertIn(
            'pip install -e ".[algorithms-nevergrad]"',
            str(ctx.exception),
        )


if __name__ == "__main__":
    unittest.main()
