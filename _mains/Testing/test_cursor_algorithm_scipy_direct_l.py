"""
Phase C2.3 gate: SciPy DIRECT-L continuous-rounded adapter.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from core.ioh_core.algorithm_factory import (
    AlgorithmFactoryError,
    create_algorithm,
    validate_algorithm_parameters,
)
from core.ioh_core.algorithms import DirectL
from core.ioh_core.algorithms.common import RunController
from core.ioh_core.algorithms.registry import (
    BudgetSemantics,
    CandidateMode,
    SeedPolicy,
    get_algorithm_spec,
    normalize_algorithm_parameters,
)
from core.ioh_core.algorithms.scipy.direct_l import DirectL as PackageDirectL
from core.ioh_core.algorithms.scipy.direct_l import _ScipyBudgetStop
from core.ioh_core.algorithms.scipy.factory import build_scipy_direct_l
from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem


class TestScipyDirectL(unittest.TestCase):
    """Registered SCIPY_DIRECT_L adapter behaviour."""

    def test_registry_spec_and_defaults(self) -> None:
        spec = get_algorithm_spec("SCIPY_DIRECT_L")
        self.assertEqual(spec.library, "scipy")
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.scipy.factory:build_scipy_direct_l",
        )
        self.assertEqual(spec.seed_policy, SeedPolicy.FORBIDDEN)
        self.assertEqual(spec.candidate_mode, CandidateMode.CONTINUOUS_ROUNDED)
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
            "SCIPY_DIRECT_L",
            {"max_evaluations": 20},
        )
        self.assertEqual(normalized, {"max_evaluations": 20})

    def test_distinct_from_nlopt_direct_l(self) -> None:
        scipy_spec = get_algorithm_spec("SCIPY_DIRECT_L")
        nlopt_spec = get_algorithm_spec("NLOPT_DIRECT_L")
        self.assertNotEqual(scipy_spec.name, nlopt_spec.name)
        self.assertEqual(scipy_spec.library, "scipy")
        self.assertEqual(nlopt_spec.library, "nlopt")
        self.assertNotEqual(scipy_spec.constructor_ref, nlopt_spec.constructor_ref)

    def test_package_export_matches_algorithms_export(self) -> None:
        self.assertIs(DirectL, PackageDirectL)

    def test_calls_direct_with_locked_arguments(self) -> None:
        captured: dict = {}

        def _fake_direct(func, bounds, **kwargs):  # noqa: ANN001
            captured["func"] = func
            captured["bounds"] = bounds
            captured["kwargs"] = kwargs
            for point in ([1.2, 2.7], [0.4, 1.1]):
                try:
                    func(point)
                except Exception:
                    break
            return None

        problem = FakeIOHProblem([0, 0], [4, 5])
        with patch("scipy.optimize.direct", _fake_direct):
            algo = create_algorithm(
                "SCIPY_DIRECT_L",
                {"max_evaluations": 9},
                random_seed=None,
            )
            result = algo(problem)

        self.assertIs(result, algo)
        self.assertEqual(captured["bounds"], [(0.0, 4.0), (0.0, 5.0)])
        kwargs = captured["kwargs"]
        self.assertEqual(kwargs["maxfun"], 9)
        self.assertEqual(kwargs["maxiter"], 1000)
        self.assertIs(kwargs["locally_biased"], True)
        self.assertEqual(problem.calls[0], [1, 3])
        self.assertEqual(problem.calls[1], [0, 1])

    def test_ioh_budget_hard_cap_despite_maxfun_overshoot(self) -> None:
        problem = FakeIOHProblem([0, 0], [5, 5])
        create_algorithm(
            "SCIPY_DIRECT_L",
            {"max_evaluations": 12},
            random_seed=None,
        )(problem)
        self.assertLessEqual(len(problem.calls), 12)
        self.assertEqual(problem.state.evaluations, len(problem.calls))
        self.assertGreaterEqual(len(problem.calls), 1)

    def test_in_bounds_integer_candidates(self) -> None:
        problem = FakeIOHProblem([0, 2], [4, 6])
        create_algorithm(
            "SCIPY_DIRECT_L",
            {"max_evaluations": 15},
            random_seed=None,
        )(problem)
        for candidate in problem.calls:
            self.assertEqual(len(candidate), 2)
            self.assertTrue(all(isinstance(value, int) for value in candidate))
            self.assertGreaterEqual(candidate[0], 0)
            self.assertLessEqual(candidate[0], 4)
            self.assertGreaterEqual(candidate[1], 2)
            self.assertLessEqual(candidate[1], 6)

    def test_does_not_re_evaluate_best(self) -> None:
        problem = FakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            "SCIPY_DIRECT_L",
            {"max_evaluations": 10},
            random_seed=None,
        )(problem)
        call_count = len(problem.calls)
        self.assertEqual(problem.state.evaluations, call_count)
        self.assertIn(problem.state.current_best.x, problem.calls)

    def test_degenerate_dimension_stays_fixed(self) -> None:
        problem = FakeIOHProblem([2, 0], [2, 3])
        create_algorithm(
            "SCIPY_DIRECT_L",
            {"max_evaluations": 12},
            random_seed=None,
        )(problem)
        self.assertGreaterEqual(len(problem.calls), 1)
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
            "SCIPY_DIRECT_L",
            {"max_evaluations": 20},
            random_seed=None,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 1)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "target_hit")

    def test_target_miss_respects_budget(self) -> None:
        problem = FakeIOHProblem([2, 2], [4, 4])
        controller = RunController(
            problem,
            10,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "SCIPY_DIRECT_L",
            {"max_evaluations": 10},
            random_seed=None,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 10)
        self.assertFalse(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")

    def test_intentional_system_error_is_swallowed(self) -> None:
        problem = FakeIOHProblem([0, 0], [2, 2])

        def _fake_direct(func, bounds, **kwargs):  # noqa: ANN001
            func([0.0, 0.0])
            try:
                func([1.0, 1.0])
            except _ScipyBudgetStop:
                raise SystemError(
                    "<built-in function direct> returned a result with an "
                    "exception set"
                ) from None
            return None

        controller = RunController(
            problem,
            1,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        with patch("scipy.optimize.direct", _fake_direct):
            create_algorithm(
                "SCIPY_DIRECT_L",
                {"max_evaluations": 1},
                random_seed=None,
            )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 1)
        self.assertTrue(controller.target_hit)

    def test_accidental_system_error_is_reraised(self) -> None:
        problem = FakeIOHProblem([0, 0], [2, 2])

        def _fake_direct(func, bounds, **kwargs):  # noqa: ANN001
            func([0.0, 0.0])
            raise SystemError("unexpected binding failure")

        with patch("scipy.optimize.direct", _fake_direct):
            with self.assertRaises(SystemError):
                create_algorithm(
                    "SCIPY_DIRECT_L",
                    {"max_evaluations": 5},
                    random_seed=None,
                )(problem)
        self.assertEqual(len(problem.calls), 1)

    def test_zero_eval_raises(self) -> None:
        problem = FakeIOHProblem([0, 0], [2, 2])

        def _fake_direct(func, bounds, **kwargs):  # noqa: ANN001
            return None

        with patch("scipy.optimize.direct", _fake_direct):
            with self.assertRaisesRegex(RuntimeError, "without evaluating"):
                create_algorithm(
                    "SCIPY_DIRECT_L",
                    {"max_evaluations": 5},
                    random_seed=None,
                )(problem)

    def test_seed_rejected(self) -> None:
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "SCIPY_DIRECT_L",
                {"max_evaluations": 5},
                random_seed=1,
            )
        with self.assertRaises(ValueError):
            build_scipy_direct_l({"max_evaluations": 5}, random_seed=1)

    def test_unknown_parameter_rejected(self) -> None:
        with self.assertRaises(AlgorithmFactoryError):
            validate_algorithm_parameters(
                "SCIPY_DIRECT_L",
                {"max_evaluations": 5, "locally_biased": False},
                random_seed=None,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "SCIPY_DIRECT_L",
                {"max_evaluations": 5, "eps": 1e-3},
                random_seed=None,
            )


if __name__ == "__main__":
    unittest.main()
