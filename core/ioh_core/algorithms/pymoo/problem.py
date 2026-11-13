"""
IOH problem adapter for Pymoo integer elementwise evaluation.
"""
from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from core.ioh_core.algorithms.common import (
    get_attached_eval_context,
    read_last_evaluation_constraint_slacks,
    read_last_evaluation_objective_y,
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)

PYMOO_BUDGET_EXHAUSTED_FITNESS = 1.0e30


def _count_ioh_constraints(problem: Any) -> int:
    """
    Return the number of active IOH inequality constraints for ``problem``.

    Prefers attached EvalContext ``_c_names``; falls back to
    ``problem.constraints.n()``.

    Parameters
    ----------
    problem : Any
        IOH-like problem with an attached EvalContext and/or constraint set.

    Returns
    -------
    int
        Non-negative constraint count.

    Raises
    ------
    ValueError
        If the constraint count cannot be determined.
    """
    ctx = get_attached_eval_context(problem)
    if ctx is not None:
        names = getattr(ctx, "_c_names", None)
        if isinstance(names, list):
            return len(names)

    constraints = getattr(problem, "constraints", None)
    if constraints is not None:
        n_method = getattr(constraints, "n", None)
        if callable(n_method):
            try:
                return int(n_method())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "problem.constraints.n() did not return an integer "
                    f"constraint count: {exc}"
                ) from exc
        if isinstance(n_method, int) and not isinstance(n_method, bool):
            return int(n_method)

    raise ValueError(
        "cannot determine IOH constraint count; attach an EvalContext with "
        "ensure_constraints(...) or expose problem.constraints.n()"
    )


class IOHIntegerProblem(ElementwiseProblem):
    """
    Pymoo elementwise problem that evaluates IOH integer catalogue indices.

    Candidates beyond the hard IOH budget or after a fixed-target hit receive
    :data:`PYMOO_BUDGET_EXHAUSTED_FITNESS` and do not call the IOH problem.

    Parameters
    ----------
    problem : Any
        IOH-like problem with bounds and callable evaluation.
    max_evaluations : int
        Hard upper bound on IOH evaluations for this run.
    run_controller : optional
        Optional target-aware evaluation controller.
    stop_flag : mutable mapping or None, optional
        Optional ``{\"stop\": bool}`` updated after each observation for
        :class:`~core.ioh_core.algorithms.pymoo.termination.ControllerAwareTermination`.
    """

    def __init__(
        self,
        problem: Any,
        max_evaluations: int,
        *,
        run_controller=None,
        stop_flag: MutableMapping[str, bool] | None = None,
    ) -> None:
        dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        self.ioh_problem = problem
        self.lower_bounds = list(lower_bounds)
        self.upper_bounds = list(upper_bounds)
        self.stop_flag = stop_flag
        self.controller = resolve_run_controller(
            problem,
            max_evaluations=max_evaluations,
            run_controller=run_controller,
        )
        # Compatibility alias for older focused tests / helpers.
        self.budget = self.controller._budget
        super().__init__(
            n_var=dimension,
            n_obj=1,
            n_ieq_constr=0,
            xl=np.asarray(self.lower_bounds, dtype=int),
            xu=np.asarray(self.upper_bounds, dtype=int),
            vtype=int,
        )

    def _evaluate(
        self,
        x: Any,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not self.controller.can_evaluate():
            out["F"] = PYMOO_BUDGET_EXHAUSTED_FITNESS
            if self.stop_flag is not None:
                self.stop_flag["stop"] = True
            return
        candidate = validate_integer_candidate(
            x,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )
        self.controller.require_evaluation()
        y = float(self.ioh_problem(candidate))
        self.controller.observe_evaluation(
            y,
        )
        if self.stop_flag is not None:
            self.stop_flag["stop"] = bool(self.controller.should_stop())
        out["F"] = y


class IOHIntegerConstrainedProblem(ElementwiseProblem):
    """
    Pymoo elementwise problem minimizing unpenalized ``y`` with native ``G``.

    Each counted evaluation calls the IOH problem (penalized ``y_p`` for
    logging / ``current_best``), then reads unpenalized ``y`` and ordered
    signed slacks ``η − 1`` from the attached EvalContext for Pymoo ``F`` /
    ``G`` (pymoo convention: feasible iff all ``G ≤ 0``; CV uses
    ``max(0, G)`` internally).

    Candidates beyond the hard IOH budget or after a fixed-target hit receive
    :data:`PYMOO_BUDGET_EXHAUSTED_FITNESS` for ``F`` and every ``G`` entry and
    do not call the IOH problem.

    Parameters
    ----------
    problem : Any
        IOH-like problem with bounds, callable evaluation, and attached
        EvalContext exposing active constraint names.
    max_evaluations : int
        Hard upper bound on IOH evaluations for this run.
    run_controller : optional
        Optional target-aware evaluation controller.
    stop_flag : mutable mapping or None, optional
        Optional ``{\"stop\": bool}`` updated after each observation for
        :class:`~core.ioh_core.algorithms.pymoo.termination.ControllerAwareTermination`.
    """

    def __init__(
        self,
        problem: Any,
        max_evaluations: int,
        *,
        run_controller=None,
        stop_flag: MutableMapping[str, bool] | None = None,
    ) -> None:
        dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        n_ieq = _count_ioh_constraints(problem)
        if n_ieq < 1:
            raise ValueError(
                "PYMOO_GA_CONSTRAINED requires at least one active IOH "
                f"constraint; got n_ieq_constr={n_ieq}"
            )
        self.ioh_problem = problem
        self.lower_bounds = list(lower_bounds)
        self.upper_bounds = list(upper_bounds)
        self.stop_flag = stop_flag
        self.controller = resolve_run_controller(
            problem,
            max_evaluations=max_evaluations,
            run_controller=run_controller,
        )
        self.budget = self.controller._budget
        super().__init__(
            n_var=dimension,
            n_obj=1,
            n_ieq_constr=n_ieq,
            xl=np.asarray(self.lower_bounds, dtype=int),
            xu=np.asarray(self.upper_bounds, dtype=int),
            vtype=int,
        )

    def _evaluate(
        self,
        x: Any,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not self.controller.can_evaluate():
            out["F"] = PYMOO_BUDGET_EXHAUSTED_FITNESS
            out["G"] = [PYMOO_BUDGET_EXHAUSTED_FITNESS] * int(self.n_ieq_constr)
            if self.stop_flag is not None:
                self.stop_flag["stop"] = True
            return
        candidate = validate_integer_candidate(
            x,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )
        self.controller.require_evaluation()
        y_p = float(self.ioh_problem(candidate))
        y = read_last_evaluation_objective_y(self.ioh_problem)
        g = read_last_evaluation_constraint_slacks(self.ioh_problem)
        if len(g) != int(self.n_ieq_constr):
            raise ValueError(
                f"expected {self.n_ieq_constr} constraint slacks, got {len(g)}"
            )
        self.controller.observe_evaluation(
            y_p,
        )
        if self.stop_flag is not None:
            self.stop_flag["stop"] = bool(self.controller.should_stop())
        out["F"] = y
        out["G"] = g
