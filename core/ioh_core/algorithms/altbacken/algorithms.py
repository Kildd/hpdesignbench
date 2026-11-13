"""
Altbacken simulated annealing for IOHexperimenter catalogue-index problems.
"""
from __future__ import annotations

from random import Random
from typing import Any, Sequence

from core.ioh_core.algorithms.altbacken.neighbourhood import (
    CatalogueResetNeighbourhood,
    sample_integer_vector,
)
from core.ioh_core.algorithms.common import (
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)

_INITIAL_TEMPERATURE = 10.0
_COOLING_RATE = 0.95


class BudgetExhausted(Exception):
    """
    Raised when the hard IOH budget or fixed-target stop forbids further
    evaluations.
    """


class AltbackenSA:
    """
    Altbacken simulated annealing on integer catalogue indices.

    Authoritative evaluation counts and best solutions come from the IOH
    problem state. The annealer's returned optimum is ignored for reporting.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Deterministic seed for initial point, neighbourhood, and acceptance.
    """

    name = "ALTBACKEN_SA"
    info = (
        "Altbacken SA with catalogue-reset neighbourhood; "
        "minimizes unconstrained y_p"
    )

    def __init__(self, max_evaluations: int, seed: int) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run Altbacken SA on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        AltbackenSA
            ``self`` after the run; best values live on the IOH problem state.
        """
        from altbacken.external.acceptance import BoltzmannAcceptance
        from altbacken.external.annealing import SimpleSimulatedAnnealing
        from altbacken.external.temperature import ExponentialCooling

        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        rng = Random(self.seed)
        neighbourhood = CatalogueResetNeighbourhood(
            lower_bounds,
            upper_bounds,
            rng,
        )

        def fitness(candidate: Sequence[int]) -> float:
            if not controller.can_evaluate():
                raise BudgetExhausted(
                    "ALTBACKEN_SA hard IOH evaluation budget exhausted"
                )
            values = validate_integer_candidate(
                candidate,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            controller.require_evaluation()
            y_p = float(problem(values))
            controller.observe_evaluation(
                y_p,
            )
            return y_p

        def stop_condition(_state: object) -> bool:
            return bool(controller.should_stop())

        annealer = SimpleSimulatedAnnealing(
            fitness=fitness,
            neighbourhood=neighbourhood,
            temperature=ExponentialCooling(
                initial_temperature=_INITIAL_TEMPERATURE,
                cooling_rate=_COOLING_RATE,
            ),
            stop=stop_condition,
            acceptance=BoltzmannAcceptance(),
            random=rng.random,
        )
        initial = sample_integer_vector(lower_bounds, upper_bounds, rng)
        try:
            annealer.simulate(initial)
        except BudgetExhausted:
            pass
        return self


class AltbackenSAWarmStart:
    """
    Altbacken SA from rank 1 of a ranked InitialDesign.

    The search matches :class:`AltbackenSA`. The loaded vector is passed to
    ``simulate``. ``sample_integer_vector`` is still called on the same
    ``Random(seed)`` and the result is discarded, so later ``randrange``,
    ``choice``, and ``random`` draws stay aligned with a cold run of the
    same seed. That discarded draw is not an evaluation.
    """

    name = "ALTBACKEN_SA_WARM_START"
    info = (
        "Altbacken SA from rank 1 of a ranked InitialDesign; "
        "catalogue-reset neighbourhood; minimizes unconstrained y_p"
    )

    def __init__(self, max_evaluations: int, seed: int, initial_design_path: str) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)
        self.initial_design_path = str(initial_design_path)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run Altbacken SA from the loaded rank-1 vector.

        Parameters
        ----------
        problem : Any
            IOH-like problem.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        AltbackenSAWarmStart
            ``self`` after the run.
        """
        from altbacken.external.acceptance import BoltzmannAcceptance
        from altbacken.external.annealing import SimpleSimulatedAnnealing
        from altbacken.external.temperature import ExponentialCooling

        from core.ioh_core.algorithms.warm_start import rank1_start_vector

        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        initial = list(
            rank1_start_vector(
                problem,
                algorithm_name=self.name,
                initial_design_path=self.initial_design_path,
                max_evaluations=self.max_evaluations,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
        )
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        rng = Random(self.seed)
        neighbourhood = CatalogueResetNeighbourhood(
            lower_bounds,
            upper_bounds,
            rng,
        )

        def fitness(candidate: Sequence[int]) -> float:
            if not controller.can_evaluate():
                raise BudgetExhausted(
                    "ALTBACKEN_SA_WARM_START hard IOH evaluation budget exhausted"
                )
            values = validate_integer_candidate(
                candidate,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            controller.require_evaluation()
            y_p = float(problem(values))
            controller.observe_evaluation(
                y_p,
            )
            return y_p

        def stop_condition(_state: object) -> bool:
            return bool(controller.should_stop())

        annealer = SimpleSimulatedAnnealing(
            fitness=fitness,
            neighbourhood=neighbourhood,
            temperature=ExponentialCooling(
                initial_temperature=_INITIAL_TEMPERATURE,
                cooling_rate=_COOLING_RATE,
            ),
            stop=stop_condition,
            acceptance=BoltzmannAcceptance(),
            random=rng.random,
        )
        sample_integer_vector(lower_bounds, upper_bounds, rng)
        try:
            annealer.simulate(initial)
        except BudgetExhausted:
            pass
        return self
