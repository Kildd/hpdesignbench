"""
Phase BLACKBOXOPT1 gate: blackboxoptim package laziness, API contract, and
BLACKBOXOPT_MSRS adapter.
"""
from __future__ import annotations

import importlib
import sys
import unittest

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version


class TestBlackboxoptimPackageScaffold(unittest.TestCase):
    """Optional blackboxoptim package remains lazy and API-compatible."""

    def test_blackboxoptim_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name == "blackboxoptim" or name.startswith("blackboxoptim."):
                sys.modules.pop(name, None)
            if name.startswith("core.ioh_core.algorithms.blackboxoptim."):
                sys.modules.pop(name, None)
        sys.modules.pop("core.ioh_core.algorithms.blackboxoptim", None)

        mod = importlib.import_module("core.ioh_core.algorithms.blackboxoptim")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn("blackboxoptim", sys.modules)

    def test_multistart_msrs_api_contract_110(self) -> None:
        try:
            installed = version("blackboxoptim")
        except PackageNotFoundError as exc:
            self.fail(f"blackboxoptim extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("1.1.0"))
        self.assertLess(parsed, Version("1.2"))

        import numpy as np
        from blackboxoptim.optimize import multistart_msrs
        from blackboxoptim.rbf import MedianLpfFilter, RbfModel

        bounds = [[0, 3], [0, 3]]
        calls: list[list[float]] = []

        def fun(x):
            batch = np.atleast_2d(x)
            values = np.sum(batch * batch, axis=1)
            calls.extend(batch.tolist())
            return values

        model = RbfModel(iindex=(0, 1), filter=MedianLpfFilter())
        np.random.seed(4)
        result = multistart_msrs(fun, bounds, 15, surrogateModel=model)
        self.assertGreaterEqual(result.nfev, 1)
        self.assertEqual(result.sample.shape[0], 15)
        self.assertTrue(np.isfinite(result.fx))


class TestBlackboxoptMsrsAdapter(unittest.TestCase):
    """Registered BLACKBOXOPT_MSRS adapter behaviour."""

    token = "BLACKBOXOPT_MSRS"

    def test_registry_metadata(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec(self.token)
        self.assertEqual(spec.library, "blackboxoptim")
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-blackboxoptim")
        self.assertTrue(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations"],
        )
        normalized = normalize_algorithm_parameters(
            self.token,
            {"max_evaluations": 50},
        )
        self.assertEqual(normalized, {"max_evaluations": 50})

    def test_respects_hard_budget(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            self.token,
            {"max_evaluations": 10},
            random_seed=4,
        )(problem)
        self.assertEqual(len(problem.calls), 10)
        self.assertEqual(problem.state.evaluations, 10)

    def test_deterministic_with_seed(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem_a = FakeIOHProblem([0, 0], [3, 3])
        problem_b = FakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            self.token,
            {"max_evaluations": 10},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            self.token,
            {"max_evaluations": 10},
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

        problem_a = FakeIOHProblem([0, 1], [3, 4])
        problem_b = FakeIOHProblem([0, 1], [3, 4])
        create_algorithm(
            self.token,
            {"max_evaluations": 12},
            random_seed=3,
        )(problem_a)
        create_algorithm(
            self.token,
            {"max_evaluations": 12},
            random_seed=5,
        )(problem_b)
        self.assertNotEqual(problem_a.calls, problem_b.calls)

    def test_does_not_re_evaluate_best(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            self.token,
            {"max_evaluations": 10},
            random_seed=4,
        )(problem)
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
                {"max_evaluations": 5, "batchSize": 2},
                random_seed=1,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                self.token,
                {"max_evaluations": 5, "batchSize": 2},
                random_seed=1,
            )

    def test_constructor_ref_and_factory_seed(self) -> None:
        from core.ioh_core.algorithms.blackboxoptim.factory import (
            build_blackboxopt_msrs,
        )
        from core.ioh_core.algorithms.registry import get_algorithm_spec

        spec = get_algorithm_spec(self.token)
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.blackboxoptim.factory:"
            "build_blackboxopt_msrs",
        )
        with self.assertRaises(ValueError):
            build_blackboxopt_msrs({"max_evaluations": 5}, random_seed=None)

    def test_missing_dependency_uses_install_hint(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.blackboxoptim.factory import (
            build_blackboxopt_msrs,
        )

        with patch(
            "core.ioh_core.algorithms.blackboxoptim.factory."
            "import_optional_dependency",
            side_effect=AlgorithmDependencyError(
                "BLACKBOXOPT_MSRS requires optional dependency "
                "'blackboxoptim'; install with:\n"
                'pip install -e ".[algorithms-blackboxoptim]"'
            ),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                build_blackboxopt_msrs({"max_evaluations": 5}, random_seed=0)
        self.assertIn(
            'pip install -e ".[algorithms-blackboxoptim]"',
            str(ctx.exception),
        )


if __name__ == "__main__":
    unittest.main()
