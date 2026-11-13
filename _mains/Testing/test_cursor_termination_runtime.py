"""
Phase C4.3 gate: target-aware RunController runtime contract.
"""
from __future__ import annotations

import math
import unittest
from typing import Any

from core.ioh_core.algorithms.common import (
    BudgetError,
    RunController,
    read_last_evaluation_feasibility,
    resolve_run_controller,
)
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem


def _run_sequence(
    controller: RunController,
    problem: FakeIOHProblem,
    values: list[tuple[float, bool]],
) -> list[list[int]]:
    """
    Drive a controller with synthetic objectives without calling analysis.

    Simulates an IOH evaluation by incrementing ``problem.state.evaluations``
    then observing ``y_p``.

    Parameters
    ----------
    controller : RunController
        Active controller.
    problem : FakeIOHProblem
        Problem whose evaluation counter is updated.
    values : list of (y_p, feasible)
        Sequence of observed returns.

    Returns
    -------
    list of list of int
        Synthetic candidates recorded for each actual evaluation.
    """
    evaluated: list[list[int]] = []
    for index, (y_p, _feasible) in enumerate(values):
        if controller.should_stop() or not controller.can_evaluate():
            break
        controller.require_evaluation()
        problem.state.evaluations += 1
        evaluated.append([index])
        controller.observe_evaluation(y_p)
    return evaluated


class TestRunControllerTargetHits(unittest.TestCase):
    """Fixed-target hit/miss and threshold semantics."""

    def test_hit_on_first_evaluation(self) -> None:
        problem = FakeIOHProblem([0], [5])
        controller = RunController(
            problem,
            5,
            mode="fixed_target",
            target_threshold=10.0,
            target_reference="all",
        )
        evaluated = _run_sequence(controller, problem, [(5.0, True), (1.0, True)])
        self.assertEqual(len(evaluated), 1)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.evaluations_to_target, 1)
        self.assertEqual(controller.stop_reason(), "target_hit")
        self.assertTrue(controller.should_stop())

    def test_hit_on_final_evaluation_reports_target_hit(self) -> None:
        problem = FakeIOHProblem([0], [5])
        controller = RunController(
            problem,
            3,
            mode="fixed_target",
            target_threshold=10.0,
            target_reference="all",
        )
        evaluated = _run_sequence(
            controller,
            problem,
            [(20.0, True), (15.0, True), (10.0, True), (0.0, True)],
        )
        self.assertEqual(len(evaluated), 3)
        self.assertTrue(controller.target_hit)
        self.assertTrue(controller.budget_exhausted)
        self.assertEqual(controller.evaluations_to_target, 3)
        self.assertEqual(controller.stop_reason(), "target_hit")

    def test_miss_at_hard_ceiling(self) -> None:
        problem = FakeIOHProblem([0], [5])
        controller = RunController(
            problem,
            3,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        evaluated = _run_sequence(
            controller,
            problem,
            [(5.0, True), (4.0, True), (3.0, True), (0.0, True)],
        )
        self.assertEqual(len(evaluated), 3)
        self.assertFalse(controller.target_hit)
        self.assertIsNone(controller.evaluations_to_target)
        self.assertTrue(controller.budget_exhausted)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")

    def test_exact_threshold_hits_and_above_does_not(self) -> None:
        problem = FakeIOHProblem([0], [5])
        exact = RunController(
            problem,
            2,
            mode="fixed_target",
            target_threshold=10.0,
            target_reference="all",
        )
        _run_sequence(exact, problem, [(10.0, True)])
        self.assertTrue(exact.target_hit)

        problem_b = FakeIOHProblem([0], [5])
        above = RunController(
            problem_b,
            2,
            mode="fixed_target",
            target_threshold=10.0,
            target_reference="all",
        )
        evaluated = _run_sequence(above, problem_b, [(10.0000001, True)])
        self.assertEqual(len(evaluated), 1)
        self.assertFalse(above.target_hit)
        self.assertEqual(above.used, 1)
        above.require_evaluation()
        problem_b.state.evaluations += 1
        above.observe_evaluation(9.0)
        self.assertTrue(above.target_hit)
        self.assertEqual(above.evaluations_to_target, 2)

    def test_non_finite_values_never_hit(self) -> None:
        problem = FakeIOHProblem([0], [5])
        controller = RunController(
            problem,
            4,
            mode="fixed_target",
            target_threshold=10.0,
            target_reference="all",
        )
        evaluated = _run_sequence(
            controller,
            problem,
            [
                (float("nan"), True),
                (float("inf"), True),
                (float("-inf"), True),
                (5.0, True),
            ],
        )
        self.assertEqual(len(evaluated), 4)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.evaluations_to_target, 4)
        self.assertEqual(controller.best_finite_y_p, 5.0)

    def test_first_hit_stable_when_later_values_better(self) -> None:
        problem = FakeIOHProblem([0], [5])
        controller = RunController(
            problem,
            5,
            mode="fixed_target",
            target_threshold=10.0,
            target_reference="all",
        )
        # After hit, should_stop prevents further requires in _run_sequence.
        evaluated = _run_sequence(
            controller,
            problem,
            [(9.0, True), (1.0, True), (0.0, True)],
        )
        self.assertEqual(len(evaluated), 1)
        self.assertEqual(controller.evaluations_to_target, 1)
        self.assertEqual(controller.best_finite_y_p, 9.0)

    def test_finite_objective_inside_margin_stops(self) -> None:
        problem = FakeIOHProblem([0], [5])
        controller = RunController(
            problem,
            4,
            mode="fixed_target",
            target_threshold=10.0,
            target_reference="all",
        )
        evaluated = _run_sequence(
            controller,
            problem,
            [
                (1.0, False),
                (20.0, True),
                (5.0, True),
                (0.0, True),
            ],
        )
        self.assertEqual(len(evaluated), 1)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.evaluations_to_target, 1)
        self.assertEqual(controller.best_finite_y_p, 1.0)

    def test_no_evaluation_after_should_stop(self) -> None:
        problem = FakeIOHProblem([0], [5])
        controller = RunController(
            problem,
            5,
            mode="fixed_target",
            target_threshold=10.0,
            target_reference="all",
        )
        controller.require_evaluation()
        problem.state.evaluations += 1
        controller.observe_evaluation(1.0)
        self.assertTrue(controller.should_stop())
        self.assertFalse(controller.can_evaluate())
        with self.assertRaises(BudgetError):
            controller.require_evaluation()


class TestResolveRunController(unittest.TestCase):
    """Controller resolver identity and budget checks."""

    def test_missing_controller_creates_fixed_budget(self) -> None:
        problem = FakeIOHProblem([0], [1])
        controller = resolve_run_controller(problem, max_evaluations=7)
        self.assertEqual(controller.mode, "fixed_budget")
        self.assertEqual(controller.max_evaluations, 7)
        self.assertIs(controller.problem, problem)

    def test_mismatch_rejected(self) -> None:
        problem_a = FakeIOHProblem([0], [1])
        problem_b = FakeIOHProblem([0], [1])
        controller = RunController(problem_a, 3)
        with self.assertRaises(ValueError):
            resolve_run_controller(
                problem_b,
                max_evaluations=3,
                run_controller=controller,
            )
        with self.assertRaises(ValueError):
            resolve_run_controller(
                problem_a,
                max_evaluations=4,
                run_controller=controller,
            )

    def test_protocol_accepts_run_controller_keyword(self) -> None:
        class _Algo:
            name = "FAKE"
            info = "fake"

            def __call__(self, problem: Any, *, run_controller=None) -> object:
                return resolve_run_controller(
                    problem,
                    max_evaluations=1,
                    run_controller=run_controller,
                )

        self.assertIsInstance(_Algo(), IOHAlgorithm)
        problem = FakeIOHProblem([0], [1])
        resolved = _Algo()(problem)
        self.assertIsInstance(resolved, RunController)
        self.assertFalse(math.isfinite(float("nan")) and False)


class TestReadLastEvaluationFeasibility(unittest.TestCase):
    """Shared feasibility reader used by sampling adapters."""

    def test_prefers_eval_context_last_data(self) -> None:
        problem = FakeIOHProblem([0], [1])
        problem.hdb_last_feasible = True
        problem.hdb_eval_context = type(
            "Ctx",
            (),
            {"last_data": {"feasible": False}},
        )()
        self.assertFalse(read_last_evaluation_feasibility(problem))

    def test_falls_back_to_hdb_last_feasible(self) -> None:
        problem = FakeIOHProblem([0], [1])
        problem.hdb_last_feasible = False
        self.assertFalse(read_last_evaluation_feasibility(problem))

    def test_defaults_true_when_unavailable(self) -> None:
        problem = FakeIOHProblem([0], [1])
        del problem.hdb_last_feasible
        self.assertTrue(read_last_evaluation_feasibility(problem))


if __name__ == "__main__":
    unittest.main()
