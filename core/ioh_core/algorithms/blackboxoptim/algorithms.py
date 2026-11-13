"""
blackboxoptim wrappers for IOHexperimenter catalogue-index problems.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from core.ioh_core.algorithms.blackboxoptim.evaluator import (
    BlackboxoptimEvaluator,
    GosacEvaluator,
    build_blackboxoptim_bounds,
    build_integer_rbf_model,
    count_ioh_constraints,
    ensure_numpy_inf_alias,
    make_budget_stop_callback,
)
from core.ioh_core.algorithms.common import (
    read_problem_bounds,
    resolve_run_controller,
)


def run_blackboxoptim_yp(
    problem: Any,
    *,
    max_evaluations: int,
    seed: int,
    run_controller: Any,
    solver: Callable[..., Any],
) -> None:
    """
    Run a blackboxoptim scalar solver on unconstrained IOH ``y_p``.

    Authoritative evaluation counts and best solutions live on the IOH
    problem state. The library ``OptimizeResult`` is ignored.
    """
    controller = resolve_run_controller(
        problem,
        max_evaluations=max_evaluations,
        run_controller=run_controller,
    )
    if not controller.can_evaluate():
        return

    dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
    bounds = build_blackboxoptim_bounds(lower_bounds, upper_bounds)
    evaluator = BlackboxoptimEvaluator(
        problem,
        controller,
        lower_bounds,
        upper_bounds,
    )
    surrogate_model = build_integer_rbf_model(dimension)
    callback = make_budget_stop_callback(controller, evaluator)

    np.random.seed(int(seed))
    try:
        solver(
            evaluator,
            bounds,
            int(max_evaluations),
            surrogateModel=surrogate_model,
            callback=callback,
        )
    except np.linalg.LinAlgError:
        # blackboxoptim RBF fits can fail on tiny budgets or degenerate
        # samples; IOH state remains authoritative for partial runs.
        pass


class BlackboxoptMsrs:
    """
    Multistart MSRS from blackboxoptim on integer catalogue indices.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Best-effort NumPy RNG seed applied immediately before the run.
    """

    name = "BLACKBOXOPT_MSRS"
    info = (
        "blackboxoptim multistart MSRS on integer catalogue indices; "
        "minimizes unconstrained y_p"
    )

    def __init__(
        self,
        max_evaluations: int,
        seed: int,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run multistart MSRS on an IOH-like problem.

        Returns
        -------
        BlackboxoptMsrs
            ``self`` after the run; best values live on the IOH problem state.
        """
        from blackboxoptim.optimize import multistart_msrs

        run_blackboxoptim_yp(
            problem,
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            run_controller=run_controller,
            solver=multistart_msrs,
        )
        return self


class BlackboxoptDycors:
    """
    DYCORS from blackboxoptim on integer catalogue indices.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Best-effort NumPy RNG seed applied immediately before the run.
    """

    name = "BLACKBOXOPT_DYCORS"
    info = (
        "blackboxoptim DYCORS on integer catalogue indices; "
        "minimizes unconstrained y_p"
    )

    def __init__(
        self,
        max_evaluations: int,
        seed: int,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run DYCORS on an IOH-like problem.

        Returns
        -------
        BlackboxoptDycors
            ``self`` after the run; best values live on the IOH problem state.
        """
        from blackboxoptim.optimize import dycors

        run_blackboxoptim_yp(
            problem,
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            run_controller=run_controller,
            solver=dycors,
        )
        return self


class BlackboxoptGosac:
    """
    GOSAC from blackboxoptim on integer catalogue indices.

    Constrained-only. Searches unpenalized ``y`` with per-constraint RBF
    models of signed slacks ``g = η − 1``. One IOH evaluation is one budget
    unit (``fun`` / ``gfun`` share a cache).

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Best-effort NumPy RNG seed applied immediately before the run.
    """

    name = "BLACKBOXOPT_GOSAC"
    info = (
        "blackboxoptim GOSAC on integer catalogue indices; minimizes "
        "unpenalized y subject to signed g=η−1"
    )

    def __init__(
        self,
        max_evaluations: int,
        seed: int,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run GOSAC on a constrained IOH-like problem.

        Returns
        -------
        BlackboxoptGosac
            ``self`` after the run; best values live on the IOH problem state.

        Raises
        ------
        ValueError
            If the problem has no active constraints.
        """
        from blackboxoptim.optimize import gosac

        ensure_numpy_inf_alias()

        mm = count_ioh_constraints(problem)
        if mm < 1:
            raise ValueError(
                "BLACKBOXOPT_GOSAC requires at least one active IOH "
                f"constraint; got mm={mm}"
            )

        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        if not controller.can_evaluate():
            return self

        dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        bounds = build_blackboxoptim_bounds(lower_bounds, upper_bounds)
        evaluator = GosacEvaluator(
            problem,
            controller,
            lower_bounds,
            upper_bounds,
            mm,
        )
        surrogate_models = [
            build_integer_rbf_model(dimension) for _ in range(mm)
        ]
        callback = make_budget_stop_callback(controller, evaluator)

        np.random.seed(self.seed)
        try:
            gosac(
                evaluator.objective,
                evaluator.constraints,
                bounds,
                self.max_evaluations,
                surrogate_models,
                callback=callback,
            )
        except np.linalg.LinAlgError:
            # blackboxoptim RBF fits can fail on tiny budgets or degenerate
            # samples; IOH state remains authoritative for partial runs.
            pass
        return self
