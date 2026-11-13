"""
StarPSO IntegerPSO wrapper for IOHexperimenter catalogue-index problems.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from star_pso.engines.generic_pso import GenericPSO
from star_pso.engines.integer_pso import IntegerPSO
from star_pso.population.particle import Particle
from star_pso.population.swarm import Swarm
from star_pso.utils.auxiliary import cost_function

from core.ioh_core.algorithms.common import (
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)

STARPSO_BUDGET_EXHAUSTED_FITNESS = 1.0e30

_LOCKED_OPTIONS = {
    "w0": 0.70,
    "c1": 1.50,
    "c2": 1.50,
    "mode": "g_best",
}


class PyStarPsoIntegerPso:
    """
    StarPSO IntegerPSO search over IOH catalogue indices.

    Authoritative evaluation counts and best solutions come from the IOH
    problem state. StarPSO's internal ranking is ignored for reporting.

    Minimizes the unconstrained penalized IOH objective ``y_p`` via
    ``@cost_function(minimize=True)``. Hard IOH budgets are enforced with a
    finite sentinel and ``solution_is_found`` stop — native ``f_max_eval`` is
    not used because it overshoots after full-swarm iterations.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    n_pop : int, optional
        Number of particles (>= 2). Default is ``50``.
    seed : int or None, optional
        Deterministic seed required by the registry seed policy.
    """

    name = "PYSTARPSO_INTEGER_PSO"
    info = (
        "StarPSO IntegerPSO on integer catalogue indices; "
        "minimizes unconstrained y_p"
    )

    def __init__(
        self,
        max_evaluations: int,
        n_pop: int = 50,
        seed: int | None = None,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.n_pop = int(n_pop)
        self.seed = seed

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run IntegerPSO on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        PyStarPsoIntegerPso
            ``self`` after the run; best values live on the IOH problem state.
        """
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        lb = np.asarray(lower_bounds, dtype=int)
        ub = np.asarray(upper_bounds, dtype=int)

        # IntegerPSO.set_seed would shadow GenericPSO.rng on the subclass while
        # IntegerPSO still samples via GenericPSO.rng — seed the base class.
        if "rng" in IntegerPSO.__dict__:
            delattr(IntegerPSO, "rng")
        GenericPSO.set_seed(int(self.seed) if self.seed is not None else None)
        positions = GenericPSO.rng.integers(
            low=lb,
            high=ub,
            endpoint=True,
            size=(self.n_pop, dimension),
            dtype=int,
        )
        swarm = Swarm([Particle(row) for row in positions])

        @cost_function(minimize=True)
        def objective(x: Any, **_kwargs: Any) -> tuple[float, bool]:
            if not controller.can_evaluate():
                return STARPSO_BUDGET_EXHAUSTED_FITNESS, True
            candidate = validate_integer_candidate(
                x,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            controller.require_evaluation()
            y = float(problem(candidate))
            controller.observe_evaluation(
                y,
            )
            return y, bool(controller.should_stop())

        optimizer = IntegerPSO(
            x_min=lb,
            x_max=ub,
            initial_swarm=swarm,
            obj_func=objective,
            n_cpus=1,
        )
        optimizer.run(
            max_it=max(1, self.max_evaluations),
            options=dict(_LOCKED_OPTIONS),
            parallel=False,
            reset_swarm=False,
            f_tol=None,
            f_max_eval=None,
            adapt_params=False,
            verbose=False,
        )
        return self


class PyStarPsoIntegerPsoWarmStart:
    """
    IntegerPSO whose initial swarm is a ranked InitialDesign prefix.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    n_pop : int, optional
        Prefix length and swarm size. Default is ``50``.
    seed : int or None, optional
        Deterministic seed required by the registry seed policy.
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    name = "PYSTARPSO_INTEGER_PSO_WARM_START"
    info = (
        "StarPSO IntegerPSO from a ranked InitialDesign prefix "
        "(unconstrained y_p)"
    )

    def __init__(
        self,
        max_evaluations: int,
        n_pop: int = 50,
        seed: int | None = None,
        *,
        initial_design_path: str,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.n_pop = int(n_pop)
        self.seed = seed
        self.initial_design_path = initial_design_path

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run warm-start IntegerPSO on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        PyStarPsoIntegerPsoWarmStart
            ``self`` after the run.
        """
        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=self.n_pop,
            max_evaluations=self.max_evaluations,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        lb = np.asarray(lower_bounds, dtype=int)
        ub = np.asarray(upper_bounds, dtype=int)
        if "rng" in IntegerPSO.__dict__:
            delattr(IntegerPSO, "rng")
        GenericPSO.set_seed(int(self.seed) if self.seed is not None else None)
        positions = np.asarray(prefix, dtype=int)
        if positions.shape != (self.n_pop, dimension):
            raise ValueError(
                "PYSTARPSO_INTEGER_PSO_WARM_START prefix shape "
                f"{positions.shape} does not match {(self.n_pop, dimension)}"
            )
        swarm = Swarm([Particle(row) for row in positions])

        @cost_function(minimize=True)
        def objective(x: Any, **_kwargs: Any) -> tuple[float, bool]:
            if not controller.can_evaluate():
                return STARPSO_BUDGET_EXHAUSTED_FITNESS, True
            candidate = validate_integer_candidate(
                x,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            controller.require_evaluation()
            y = float(problem(candidate))
            controller.observe_evaluation(
                y,
            )
            return y, bool(controller.should_stop())

        optimizer = IntegerPSO(
            x_min=lb,
            x_max=ub,
            initial_swarm=swarm,
            obj_func=objective,
            n_cpus=1,
        )
        optimizer.run(
            max_it=max(1, self.max_evaluations),
            options=dict(_LOCKED_OPTIONS),
            parallel=False,
            reset_swarm=False,
            f_tol=None,
            f_max_eval=None,
            adapt_params=False,
            verbose=False,
        )
        return self
