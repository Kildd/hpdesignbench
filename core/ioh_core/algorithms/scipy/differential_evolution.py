"""
SciPy differential evolution over IOH integer catalogue-index bounds.
"""
from __future__ import annotations

from typing import Any

from core.ioh_core.algorithms.common import (
    BudgetError,
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)


class _ScipyBudgetStop(Exception):
    """
    Internal signal to leave ``differential_evolution`` early.

    Raised from the objective when the shared run controller forbids further
    IOH evaluations (hard budget or fixed-target hit). Caught only around the
    SciPy call; never treated as a solver result.
    """


class DifferentialEvolution:
    """
    SciPy differential evolution with integer integrality on catalogue indices.

    Uses ``integrality=True`` so SciPy proposes integer-valued coordinates that
    are validated with ``validate_integer_candidate`` (no continuous rounding).
    Hard IOH budgeting and fixed-target stops are enforced through
    :class:`~core.ioh_core.algorithms.common.RunController`.

    Parameters
    ----------
    max_evaluations : int
        Hard upper bound on counted IOH evaluations.
    seed : int
        Deterministic seed required by the registry seed policy (passed as
        SciPy ``rng``).
    """

    name = "SCIPY_DIFFERENTIAL_EVOLUTION"
    info = (
        "SciPy differential evolution with integer integrality on catalogue "
        "indices"
    )

    def __init__(self, max_evaluations: int, seed: int) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run differential evolution on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        DifferentialEvolution
            ``self`` after the run; best values live on the IOH problem state.
        """
        from scipy.optimize import differential_evolution

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

        def objective(x: Any) -> float:
            if not controller.can_evaluate():
                raise _ScipyBudgetStop()
            try:
                controller.require_evaluation()
            except BudgetError as exc:
                raise _ScipyBudgetStop() from exc

            x_int = validate_integer_candidate(
                x,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            y = float(problem(x_int))
            controller.observe_evaluation(
                y,
            )
            if controller.should_stop():
                raise _ScipyBudgetStop()
            return y

        try:
            differential_evolution(
                objective,
                bounds,
                strategy="best1bin",
                maxiter=self.max_evaluations,
                popsize=15,
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.7,
                rng=self.seed,
                polish=False,
                init="latinhypercube",
                atol=0,
                updating="immediate",
                workers=1,
                integrality=True,
            )
        except _ScipyBudgetStop:
            pass
        return self


class DifferentialEvolutionWarmStart:
    """
    Differential evolution whose initial population is a ranked prefix.

    ``init`` has shape ``(15 * dimension, dimension)`` and overrides the
    population size implied by ``popsize``.

    Parameters
    ----------
    max_evaluations : int
        Hard upper bound on counted IOH evaluations.
    seed : int
        Deterministic seed passed as SciPy ``rng``.
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    name = "SCIPY_DIFFERENTIAL_EVOLUTION_WARM_START"
    info = (
        "SciPy differential evolution from a ranked InitialDesign prefix"
    )

    def __init__(
        self,
        max_evaluations: int,
        seed: int,
        *,
        initial_design_path: str,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)
        self.initial_design_path = initial_design_path

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run warm-start differential evolution on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        DifferentialEvolutionWarmStart
            ``self`` after the run.
        """
        import numpy as np

        from scipy.optimize import differential_evolution

        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=15 * dimension,
            max_evaluations=self.max_evaluations,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        bounds = [
            (float(lower), float(upper))
            for lower, upper in zip(lower_bounds, upper_bounds)
        ]

        def objective(x: Any) -> float:
            if not controller.can_evaluate():
                raise _ScipyBudgetStop()
            try:
                controller.require_evaluation()
            except BudgetError as exc:
                raise _ScipyBudgetStop() from exc

            x_int = validate_integer_candidate(
                x,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            y = float(problem(x_int))
            controller.observe_evaluation(
                y,
            )
            if controller.should_stop():
                raise _ScipyBudgetStop()
            return y

        try:
            differential_evolution(
                objective,
                bounds,
                strategy="best1bin",
                maxiter=self.max_evaluations,
                popsize=15,
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.7,
                rng=self.seed,
                polish=False,
                init=np.asarray(prefix, dtype=float),
                atol=0,
                updating="immediate",
                workers=1,
                integrality=True,
            )
        except _ScipyBudgetStop:
            pass
        return self
