"""
Phase ALG1 gate: candidate conversion, bounds, and evaluation-budget helpers.
"""
from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from core.ioh_core.algorithms.common import (
    BudgetError,
    CandidateError,
    EvaluationBudget,
    RunController,
    clip_integer_candidate,
    continuous_to_integer_candidate,
    read_problem_bounds,
    resolve_run_controller,
    round_half_away_from_zero,
    validate_integer_candidate,
)
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem


class _ProtocolAlgo:
    name = "FAKE"
    info = "fake"

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        return resolve_run_controller(
            problem,
            max_evaluations=1,
            run_controller=run_controller,
        )


class TestFakeProblemContract(unittest.TestCase):
    """Fake problem shape and protocol smoke."""

    def test_fake_problem_and_protocol(self) -> None:
        problem = FakeIOHProblem([0, -2], [3, 4])
        self.assertEqual(read_problem_bounds(problem), (2, [0, -2], [3, 4]))
        y = problem([1, 0])
        self.assertEqual(y, 1.0)
        self.assertEqual(problem.state.evaluations, 1)
        self.assertEqual(problem.state.current_best.x, [1, 0])
        algo = _ProtocolAlgo()
        self.assertIsInstance(algo, IOHAlgorithm)
        controller = algo(problem)
        self.assertIsInstance(controller, RunController)
        self.assertEqual(controller.mode, "fixed_budget")


class TestRunControllerContract(unittest.TestCase):
    """Shared RunController budget and resolver smoke."""

    def test_fixed_budget_controller_tracks_evaluations(self) -> None:
        problem = FakeIOHProblem([0], [3])
        controller = RunController(problem, 2)
        controller.require_evaluation()
        problem([1])
        controller.observe_evaluation(4.0)
        self.assertEqual(controller.used, 1)
        self.assertFalse(controller.target_hit)
        self.assertEqual(controller.best_finite_y_p, 4.0)
        controller.require_evaluation()
        problem([2])
        controller.observe_evaluation(1.0)
        self.assertTrue(controller.budget_exhausted)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")
        with self.assertRaises(BudgetError):
            controller.require_evaluation()

    def test_resolve_rejects_problem_mismatch(self) -> None:
        problem_a = FakeIOHProblem([0], [1])
        problem_b = FakeIOHProblem([0], [1])
        supplied = RunController(problem_a, 2)
        with self.assertRaises(ValueError):
            resolve_run_controller(
                problem_b,
                max_evaluations=2,
                run_controller=supplied,
            )


class TestRoundingAndCandidates(unittest.TestCase):
    """Continuous conversion, integer validation, clipping."""

    def test_half_step_tie_rule(self) -> None:
        self.assertEqual(round_half_away_from_zero(0.5), 1)
        self.assertEqual(round_half_away_from_zero(-0.5), -1)
        self.assertEqual(round_half_away_from_zero(1.5), 2)
        self.assertEqual(round_half_away_from_zero(2.5), 3)
        self.assertEqual(round_half_away_from_zero(-1.5), -2)

    def test_continuous_conversion_clipping_and_numpy_scalars(self) -> None:
        lower = [0, 0, 0]
        upper = [2, 2, 2]
        converted = continuous_to_integer_candidate(
            [0.5, np.float64(1.2), 2.6],
            lower_bounds=lower,
            upper_bounds=upper,
        )
        self.assertEqual(converted, [1, 1, 2])

        clipped = continuous_to_integer_candidate(
            [-10.0, 1.0, 99.0],
            lower_bounds=lower,
            upper_bounds=upper,
        )
        self.assertEqual(clipped, [0, 1, 2])
        self.assertEqual(
            clip_integer_candidate([-1, 5, 1], lower_bounds=lower, upper_bounds=upper),
            [0, 2, 1],
        )

    def test_integer_validation_and_errors(self) -> None:
        lower = [0, 1]
        upper = [2, 4]
        self.assertEqual(
            validate_integer_candidate([0, 4], lower_bounds=lower, upper_bounds=upper),
            [0, 4],
        )
        with self.assertRaisesRegex(CandidateError, "dimension"):
            validate_integer_candidate([0], lower_bounds=lower, upper_bounds=upper)
        with self.assertRaisesRegex(CandidateError, "integer index"):
            validate_integer_candidate([0.5, 1], lower_bounds=lower, upper_bounds=upper)
        with self.assertRaisesRegex(CandidateError, "outside inclusive"):
            validate_integer_candidate([0, 5], lower_bounds=lower, upper_bounds=upper)
        with self.assertRaisesRegex(CandidateError, "numeric"):
            continuous_to_integer_candidate(
                ["a", 1],
                lower_bounds=lower,
                upper_bounds=upper,
            )


class TestEvaluationBudget(unittest.TestCase):
    """Hard IOH evaluation budget helper."""

    def test_budget_cannot_be_exceeded(self) -> None:
        problem = FakeIOHProblem([0], [5])
        budget = EvaluationBudget(3, problem)
        self.assertEqual(budget.remaining, 3)
        self.assertTrue(budget.can_evaluate(3))

        for _ in range(3):
            budget.require(1)
            problem([1])
        self.assertEqual(budget.remaining, 0)
        self.assertFalse(budget.can_evaluate(1))
        with self.assertRaisesRegex(BudgetError, "exhausted"):
            budget.require(1)
        # IOH state remains the authority for used count.
        self.assertEqual(problem.state.evaluations, 3)
        self.assertEqual(budget.used, 3)

    def test_invalid_budget_arguments(self) -> None:
        problem = FakeIOHProblem([0], [1])
        with self.assertRaises(BudgetError):
            EvaluationBudget(0, problem)
        with self.assertRaises(BudgetError):
            EvaluationBudget(True, problem)  # type: ignore[arg-type]
        budget = EvaluationBudget(2, problem)
        with self.assertRaises(BudgetError):
            budget.can_evaluate(0)


class TestLastEvaluationRecordHelpers(unittest.TestCase):
    """PM1.1: read last y / ordered violations from attached EvalContext."""

    def tearDown(self) -> None:
        from core.ioh_core.algorithms.common import detach_eval_context

        problem = getattr(self, "_problem", None)
        if problem is not None:
            detach_eval_context(problem)

    def _make_context(
        self,
        *,
        y: float,
        y_p: float,
        violations: dict[str, float],
        names: list[str],
        feasible: bool = True,
    ) -> Any:
        return type(
            "StubEvalContext",
            (),
            {
                "_c_names": list(names),
                "last_data": {
                    "y": y,
                    "y_p": y_p,
                    "violations_": dict(violations),
                    "feasible": feasible,
                },
            },
        )()

    def test_reads_y_and_ordered_violations_from_attached_context(self) -> None:
        from core.ioh_core.algorithms.common import (
            attach_eval_context,
            read_last_evaluation_feasibility,
            read_last_evaluation_objective_y,
            read_last_evaluation_record,
            read_last_evaluation_violations,
        )

        self._problem = FakeIOHProblem([0], [2])
        ctx = self._make_context(
            y=10.0,
            y_p=12.5,
            violations={"b": 0.5, "a": 0.0},
            names=["a", "b"],
            feasible=False,
        )
        attach_eval_context(self._problem, ctx)

        record = read_last_evaluation_record(self._problem)
        self.assertEqual(record["y"], 10.0)
        self.assertEqual(record["y_p"], 12.5)
        self.assertEqual(read_last_evaluation_objective_y(self._problem), 10.0)
        self.assertEqual(
            read_last_evaluation_violations(self._problem),
            [0.0, 0.5],
        )
        self.assertFalse(read_last_evaluation_feasibility(self._problem))

    def test_missing_context_raises(self) -> None:
        from core.ioh_core.algorithms.common import (
            read_last_evaluation_objective_y,
            read_last_evaluation_record,
            read_last_evaluation_violations,
        )

        problem = FakeIOHProblem([0], [1])
        with self.assertRaises(ValueError):
            read_last_evaluation_record(problem)
        with self.assertRaises(ValueError):
            read_last_evaluation_objective_y(problem)
        with self.assertRaises(ValueError):
            read_last_evaluation_violations(problem)

    def test_missing_violation_key_becomes_zero(self) -> None:
        from core.ioh_core.algorithms.common import (
            attach_eval_context,
            read_last_evaluation_violations,
        )

        self._problem = FakeIOHProblem([0], [1])
        ctx = self._make_context(
            y=1.0,
            y_p=1.0,
            violations={"a": 0.25},
            names=["a", "missing"],
        )
        attach_eval_context(self._problem, ctx)
        self.assertEqual(
            read_last_evaluation_violations(self._problem),
            [0.25, 0.0],
        )

    def test_negative_or_nonfinite_violation_rejected(self) -> None:
        from core.ioh_core.algorithms.common import (
            attach_eval_context,
            read_last_evaluation_violations,
        )

        self._problem = FakeIOHProblem([0], [1])
        ctx = self._make_context(
            y=1.0,
            y_p=1.0,
            violations={"a": -0.1},
            names=["a"],
        )
        attach_eval_context(self._problem, ctx)
        with self.assertRaises(ValueError):
            read_last_evaluation_violations(self._problem)

    def test_feasibility_helper_still_defaults_without_context(self) -> None:
        from core.ioh_core.algorithms.common import (
            read_last_evaluation_feasibility,
        )

        problem = FakeIOHProblem([0], [1])
        self.assertTrue(read_last_evaluation_feasibility(problem))
        problem.hdb_last_feasible = False
        self.assertFalse(read_last_evaluation_feasibility(problem))


if __name__ == "__main__":
    unittest.main()
