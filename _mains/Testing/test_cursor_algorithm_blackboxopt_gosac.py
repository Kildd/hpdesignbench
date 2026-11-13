"""
Phase BLACKBOXOPT3 gate: GOSAC API contract and BLACKBOXOPT_GOSAC adapter.
"""
from __future__ import annotations

import unittest

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version

from _mains.Testing.test_cursor_algorithm_scipy_cobyla import (
    _ConstrainedFakeIOHProblem,
)


class TestBlackboxoptimGosacApi(unittest.TestCase):
    """blackboxoptim GOSAC remains API-compatible at the locked extra."""

    def test_gosac_api_contract_110(self) -> None:
        try:
            installed = version("blackboxoptim")
        except PackageNotFoundError as exc:
            self.fail(f"blackboxoptim extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("1.1.0"))
        self.assertLess(parsed, Version("1.2"))

        import numpy as np
        from blackboxoptim.optimize import gosac
        from blackboxoptim.rbf import MedianLpfFilter, RbfModel

        from core.ioh_core.algorithms.blackboxoptim.evaluator import (
            ensure_numpy_inf_alias,
        )

        ensure_numpy_inf_alias()

        bounds = [[0, 3], [0, 3]]

        def fun(x):
            batch = np.atleast_2d(x)
            return np.sum(batch * batch, axis=1)

        def gfun(x):
            batch = np.atleast_2d(x)
            return batch[:, :1] - 1.0

        models = [RbfModel(iindex=(0, 1), filter=MedianLpfFilter())]
        np.random.seed(4)
        result = gosac(fun, gfun, bounds, 12, models)
        self.assertGreaterEqual(result.nfev, 1)
        self.assertLessEqual(result.nfev, 12)
        self.assertEqual(result.sample.shape[1], 2)


class TestBlackboxoptGosacAdapter(unittest.TestCase):
    """Registered BLACKBOXOPT_GOSAC adapter behaviour."""

    token = "BLACKBOXOPT_GOSAC"

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
        self.assertIn("constrained", spec.description.lower())
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations"],
        )
        normalized = normalize_algorithm_parameters(
            self.token,
            {"max_evaluations": 50},
        )
        self.assertEqual(normalized, {"max_evaluations": 50})

    def test_gosac_evaluator_uses_unpenalized_y_and_signed_slacks(self) -> None:
        import numpy as np

        from core.ioh_core.algorithms.blackboxoptim.evaluator import (
            GosacEvaluator,
        )
        from core.ioh_core.algorithms.common import resolve_run_controller

        problem = _ConstrainedFakeIOHProblem(
            [0, 0],
            [3, 3],
            y_scale=2.0,
        )
        controller = resolve_run_controller(problem, max_evaluations=5)
        evaluator = GosacEvaluator(
            problem,
            controller,
            [0, 0],
            [3, 3],
            1,
        )
        point = np.asarray([[2, 0]], dtype=float)
        y = evaluator.objective(point)
        g = evaluator.constraints(point)
        self.assertEqual(len(problem.calls), 1)
        self.assertAlmostEqual(float(y[0]), 8.0)
        self.assertAlmostEqual(float(g[0, 0]), 1.0)
        self.assertNotEqual(float(y[0]), float(problem.calls[0][0] ** 2))

    def test_rejects_zero_constraints(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem(
            [0],
            [2],
            constraint_names=[],
        )
        algo = create_algorithm(
            self.token,
            {"max_evaluations": 5},
            random_seed=1,
        )
        with self.assertRaises(ValueError):
            algo(problem)

    def test_respects_hard_budget(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            self.token,
            {"max_evaluations": 10},
            random_seed=4,
        )(problem)
        self.assertLessEqual(len(problem.calls), 10)
        self.assertGreaterEqual(len(problem.calls), 1)
        self.assertEqual(problem.state.evaluations, len(problem.calls))

    def test_eval_cache_does_not_double_count(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            self.token,
            {"max_evaluations": 10},
            random_seed=4,
        )(problem)
        unique = {tuple(c) for c in problem.calls}
        self.assertEqual(len(problem.calls), len(unique))

    def test_deterministic_with_seed(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem_a = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        problem_b = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
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

        problem_a = _ConstrainedFakeIOHProblem([0, 1], [3, 4])
        problem_b = _ConstrainedFakeIOHProblem([0, 1], [3, 4])
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

        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
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
            build_blackboxopt_gosac,
        )
        from core.ioh_core.algorithms.registry import get_algorithm_spec

        spec = get_algorithm_spec(self.token)
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.blackboxoptim.factory:"
            "build_blackboxopt_gosac",
        )
        with self.assertRaises(ValueError):
            build_blackboxopt_gosac({"max_evaluations": 5}, random_seed=None)

    def test_missing_dependency_uses_install_hint(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.blackboxoptim.factory import (
            build_blackboxopt_gosac,
        )

        with patch(
            "core.ioh_core.algorithms.blackboxoptim.factory."
            "import_optional_dependency",
            side_effect=AlgorithmDependencyError(
                "BLACKBOXOPT_GOSAC requires optional dependency "
                "'blackboxoptim'; install with:\n"
                'pip install -e ".[algorithms-blackboxoptim]"'
            ),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                build_blackboxopt_gosac({"max_evaluations": 5}, random_seed=0)
        self.assertIn(
            'pip install -e ".[algorithms-blackboxoptim]"',
            str(ctx.exception),
        )


if __name__ == "__main__":
    unittest.main()
