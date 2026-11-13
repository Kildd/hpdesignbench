"""
IOH problem adapter for LEAP integer ScalarProblem evaluation.
"""
from __future__ import annotations

from typing import Any

from leap_ec.problem import ScalarProblem

from core.ioh_core.algorithms.common import (
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)

LEAP_BUDGET_EXHAUSTED_FITNESS = 1.0e30


class IOHLeapIntegerProblem(ScalarProblem):
    """
    LEAP scalar problem that minimizes IOH integer catalogue indices.

    Candidates beyond the hard IOH budget or after a fixed-target hit receive
    :data:`LEAP_BUDGET_EXHAUSTED_FITNESS` and do not call the IOH problem.

    Parameters
    ----------
    problem : Any
        IOH-like problem with bounds and callable evaluation.
    max_evaluations : int
        Hard upper bound on IOH evaluations for this run.
    run_controller : optional
        Optional target-aware evaluation controller.
    """

    def __init__(
        self,
        problem: Any,
        max_evaluations: int,
        *,
        run_controller=None,
    ) -> None:
        super().__init__(maximize=False)
        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        self.ioh_problem = problem
        self.lower_bounds = list(lower_bounds)
        self.upper_bounds = list(upper_bounds)
        self.bounds = [
            (int(lo), int(hi))
            for lo, hi in zip(self.lower_bounds, self.upper_bounds)
        ]
        self.controller = resolve_run_controller(
            problem,
            max_evaluations=max_evaluations,
            run_controller=run_controller,
        )

    def evaluate(self, phenome: Any, *args: Any, **kwargs: Any) -> float:
        """
        Evaluate one phenome under the hard IOH budget.

        Parameters
        ----------
        phenome : Any
            Integer-valued genome / phenotype from LEAP.
        *args, **kwargs
            Accepted for LEAP ``Problem.evaluate`` compatibility; unused.

        Returns
        -------
        float
            Penalized IOH objective, or the finite budget-exhausted sentinel.
        """
        if not self.controller.can_evaluate():
            return LEAP_BUDGET_EXHAUSTED_FITNESS
        candidate = validate_integer_candidate(
            phenome,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )
        self.controller.require_evaluation()
        y = float(self.ioh_problem(candidate))
        self.controller.observe_evaluation(
            y,
        )
        return y
