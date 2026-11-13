"""
SciPy COBYLA on a constrained continuous-to-integer rounded IOH surface.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from core.ioh_core.algorithms.common import (
    BudgetError,
    CandidateError,
    continuous_to_integer_candidate,
    get_attached_eval_context,
    read_last_evaluation_constraint_slacks,
    read_last_evaluation_objective_y,
    read_problem_bounds,
    resolve_run_controller,
    seeded_uniform_continuous_start,
)


class _ScipyBudgetStop(Exception):
    """
    Internal signal to leave SciPy ``minimize`` / COBYLA early.

    Raised from the objective or constraint callable when the shared run
    controller forbids further IOH evaluations (hard budget or fixed-target
    hit). Caught only around the SciPy call; never treated as a solver result.
    """


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


class Cobyla:
    """
    SciPy COBYLA on a constrained continuous-to-integer rounded surface.

    Continuous proposals are converted with the shared
    ``continuous_to_integer_candidate`` policy. Searches on unpenalized ``y``
    with ``NonlinearConstraint`` fed signed ``η − 1`` (``ub=0``). Objective and
    constraint callables share an integer-keyed eval cache so SciPy's separate
    callbacks do not double-count IOH evaluations. Hard IOH budgeting and
    fixed-target stops are enforced through
    :class:`~core.ioh_core.algorithms.common.RunController`.

    Parameters
    ----------
    max_evaluations : int
        Hard upper bound on counted IOH evaluations (also soft COBYLA
        ``maxiter``).
    seed : int
        Task seed used once to draw a uniform continuous start in the
        catalogue-index box (multistart across repetitions).
    """

    name = "SCIPY_COBYLA"
    info = (
        "SciPy COBYLA on continuous-rounded catalogue indices; minimizes "
        "unpenalized y subject to signed g=η−1; seeded random start"
    )

    def __init__(self, max_evaluations: int, seed: int) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run COBYLA on a constrained IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds, callable evaluation, and attached
            EvalContext with at least one active constraint.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        Cobyla
            ``self`` after the run; best values live on the IOH problem state.

        Raises
        ------
        ValueError
            If the problem has no active constraints.
        """
        from scipy.optimize import NonlinearConstraint, minimize

        mm = _count_ioh_constraints(problem)
        if mm < 1:
            raise ValueError(
                "SCIPY_COBYLA requires at least one active IOH "
                f"constraint; got mm={mm}"
            )

        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        bounds = [
            (float(lower), float(upper))
            for lower, upper in zip(lower_bounds, upper_bounds)
        ]
        x0 = seeded_uniform_continuous_start(
            lower_bounds,
            upper_bounds,
            self.seed,
        )
        eval_cache: dict[tuple[int, ...], tuple[float, float, list[float]]] = {}

        def evaluate_once(x: Any) -> tuple[float, float, list[float]]:
            try:
                x_int = continuous_to_integer_candidate(
                    x,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                )
            except CandidateError as exc:
                # SciPy/PRIMA may probe malformed vectors after a stop; treat
                # as a soft exit rather than leaking CandidateError.
                raise _ScipyBudgetStop() from exc
            key = tuple(x_int)
            cached = eval_cache.get(key)
            if cached is not None:
                return cached
            if not controller.can_evaluate():
                raise _ScipyBudgetStop()
            try:
                controller.require_evaluation()
            except BudgetError as exc:
                raise _ScipyBudgetStop() from exc
            y_p = float(problem(x_int))
            y = read_last_evaluation_objective_y(problem)
            g = read_last_evaluation_constraint_slacks(problem)
            if len(g) != mm:
                raise ValueError(
                    f"expected {mm} constraint slacks, got {len(g)}"
                )
            controller.observe_evaluation(
                y_p,
            )
            record = (y_p, float(y), list(g))
            eval_cache[key] = record
            return record

        def objective(x: Any) -> float:
            _y_p, y, _g = evaluate_once(x)
            return float(y)

        def constraint_fun(x: Any) -> np.ndarray:
            _y_p, _y, g = evaluate_once(x)
            return np.asarray(g, dtype=float)

        nonlinear = NonlinearConstraint(
            constraint_fun,
            lb=-np.inf,
            ub=0.0,
        )

        try:
            minimize(
                objective,
                x0=x0,
                method="COBYLA",
                bounds=bounds,
                constraints=[nonlinear],
                options={"maxiter": self.max_evaluations},
            )
        except _ScipyBudgetStop:
            pass
        return self


class CobylaWarmStart:
    """
    SciPy COBYLA from rank 1 of a ranked InitialDesign.

    The search matches :class:`Cobyla` except ``x0``, which is the loaded
    catalogue-index vector as floats. No seed is used.
    """

    name = "SCIPY_COBYLA_WARM_START"
    info = (
        "SciPy COBYLA on continuous-rounded catalogue indices; minimizes "
        "unpenalized y subject to signed g=η−1; rank-1 InitialDesign start"
    )

    def __init__(self, max_evaluations: int, initial_design_path: str) -> None:
        self.max_evaluations = int(max_evaluations)
        self.initial_design_path = str(initial_design_path)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run COBYLA from the loaded rank-1 vector.

        Parameters
        ----------
        problem : Any
            Constrained IOH-like problem.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        CobylaWarmStart
            ``self`` after the run.

        Raises
        ------
        ValueError
            If the problem has no active constraints, the InitialDesign is
            missing, or the budget cannot pay for the start.
        """
        from scipy.optimize import NonlinearConstraint, minimize

        from core.ioh_core.algorithms.warm_start import rank1_start_vector

        mm = _count_ioh_constraints(problem)
        if mm < 1:
            raise ValueError(
                "SCIPY_COBYLA_WARM_START requires at least one active IOH "
                f"constraint; got mm={mm}"
            )

        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        bounds = [
            (float(lower), float(upper))
            for lower, upper in zip(lower_bounds, upper_bounds)
        ]
        x0 = [
            float(value)
            for value in rank1_start_vector(
                problem,
                algorithm_name=self.name,
                initial_design_path=self.initial_design_path,
                max_evaluations=self.max_evaluations,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
        ]
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        eval_cache: dict[tuple[int, ...], tuple[float, float, list[float]]] = {}

        def evaluate_once(x: Any) -> tuple[float, float, list[float]]:
            try:
                x_int = continuous_to_integer_candidate(
                    x,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                )
            except CandidateError as exc:
                raise _ScipyBudgetStop() from exc
            key = tuple(x_int)
            cached = eval_cache.get(key)
            if cached is not None:
                return cached
            if not controller.can_evaluate():
                raise _ScipyBudgetStop()
            try:
                controller.require_evaluation()
            except BudgetError as exc:
                raise _ScipyBudgetStop() from exc
            y_p = float(problem(x_int))
            y = read_last_evaluation_objective_y(problem)
            g = read_last_evaluation_constraint_slacks(problem)
            if len(g) != mm:
                raise ValueError(
                    f"expected {mm} constraint slacks, got {len(g)}"
                )
            controller.observe_evaluation(
                y_p,
            )
            record = (y_p, float(y), list(g))
            eval_cache[key] = record
            return record

        def objective(x: Any) -> float:
            _y_p, y, _g = evaluate_once(x)
            return float(y)

        def constraint_fun(x: Any) -> np.ndarray:
            _y_p, _y, g = evaluate_once(x)
            return np.asarray(g, dtype=float)

        nonlinear = NonlinearConstraint(
            constraint_fun,
            lb=-np.inf,
            ub=0.0,
        )

        try:
            minimize(
                objective,
                x0=x0,
                method="COBYLA",
                bounds=bounds,
                constraints=[nonlinear],
                options={"maxiter": self.max_evaluations},
            )
        except _ScipyBudgetStop:
            pass
        return self
