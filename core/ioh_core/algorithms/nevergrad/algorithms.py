"""
Nevergrad MetaModel wrappers for IOHexperimenter catalogue-index problems.
"""
from __future__ import annotations

from typing import Any

from nevergrad.optimization import registry as nevergrad_registry

from core.ioh_core.algorithms.common import (
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)
from core.ioh_core.algorithms.nevergrad.space import build_transition_choice_tuple


def _run_nevergrad_ask_tell(
    *,
    problem: Any,
    max_evaluations: int,
    seed: int,
    engine_name: str,
    run_controller=None,
    prefix: tuple[tuple[int, ...], ...] | None = None,
) -> None:
    """
    Run one Nevergrad registry engine with hard IOH ask–tell control.

    Parameters
    ----------
    problem : Any
        IOH-like problem with bounds and callable evaluation.
    max_evaluations : int
        Hard IOH evaluation budget (also Nevergrad constructor budget).
    seed : int
        Deterministic seed for the parametrization RNG.
    engine_name : str
        Exact Nevergrad registry name (``MetaModel`` or
        ``MetaModelOnePlusOne``).
    run_controller : optional
        Optional target-aware evaluation controller.
    """
    controller = resolve_run_controller(
        problem,
        max_evaluations=max_evaluations,
        run_controller=run_controller,
    )
    _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
    parametrization = build_transition_choice_tuple(
        lower_bounds,
        upper_bounds,
    )
    parametrization.random_state.seed(int(seed))
    optimizer = nevergrad_registry[engine_name](
        parametrization=parametrization,
        budget=int(max_evaluations),
        num_workers=1,
    )
    queued = tuple(prefix or ())
    evaluation_index = 0

    while not controller.should_stop():
        if not controller.can_evaluate():
            break
        if evaluation_index < len(queued):
            optimizer.suggest(
                tuple(int(value) for value in queued[evaluation_index])
            )
        evaluation_index += 1
        candidate_ng = optimizer.ask()
        candidate = validate_integer_candidate(
            candidate_ng.value,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        controller.require_evaluation()
        y = float(problem(candidate))
        controller.observe_evaluation(
            y,
        )
        optimizer.tell(candidate_ng, y)


class NevergradMetaModel:
    """
    Nevergrad ``MetaModel`` on discrete catalogue-index ``TransitionChoice``s.

    Authoritative evaluation counts and best solutions come from the IOH
    problem state. Nevergrad recommendations are ignored for reporting.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Deterministic seed required by the registry seed policy.
    """

    name = "NEVERGRAD_METAMODEL"
    info = (
        "Nevergrad MetaModel on discrete catalogue indices; "
        "minimizes unconstrained y_p"
    )

    def __init__(self, max_evaluations: int, seed: int) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run MetaModel on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        NevergradMetaModel
            ``self`` after the run; best values live on the IOH problem state.
        """
        _run_nevergrad_ask_tell(
            problem=problem,
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            engine_name="MetaModel",
            run_controller=run_controller,
        )
        return self


class NevergradMetaModelOnePlusOne:
    """
    Nevergrad ``MetaModelOnePlusOne`` on discrete catalogue indices.

    Same ask–tell / ``y_p`` contract as :class:`NevergradMetaModel`, with
    ``OnePlusOne`` as the multivariate base optimizer.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Deterministic seed required by the registry seed policy.
    """

    name = "NEVERGRAD_METAMODEL_ONEPLUSONE"
    info = (
        "Nevergrad MetaModelOnePlusOne on discrete catalogue indices; "
        "minimizes unconstrained y_p"
    )

    def __init__(self, max_evaluations: int, seed: int) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run MetaModelOnePlusOne on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        NevergradMetaModelOnePlusOne
            ``self`` after the run; best values live on the IOH problem state.
        """
        _run_nevergrad_ask_tell(
            problem=problem,
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            engine_name="MetaModelOnePlusOne",
            run_controller=run_controller,
        )
        return self


_NEVERGRAD_PREFIX_LENGTH = 50


def _run_nevergrad_warm_start(
    problem: Any,
    *,
    name: str,
    engine_name: str,
    max_evaluations: int,
    seed: int,
    initial_design_path: str,
    run_controller: Any,
) -> None:
    """Load a length-50 prefix, then interleave suggest and ask."""
    from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

    _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
    prefix = load_warm_start_prefix(
        problem,
        algorithm_name=name,
        initial_design_path=initial_design_path,
        count=_NEVERGRAD_PREFIX_LENGTH,
        max_evaluations=max_evaluations,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    _run_nevergrad_ask_tell(
        problem=problem,
        max_evaluations=max_evaluations,
        seed=seed,
        engine_name=engine_name,
        run_controller=run_controller,
        prefix=prefix,
    )


class NevergradMetaModelWarmStart:
    """
    MetaModel whose first 50 asks are a ranked InitialDesign prefix.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Deterministic seed.
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    name = "NEVERGRAD_METAMODEL_WARM_START"
    info = "Nevergrad MetaModel from a ranked InitialDesign prefix"

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
        """Run warm-start MetaModel."""
        _run_nevergrad_warm_start(
            problem,
            name=self.name,
            engine_name="MetaModel",
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            initial_design_path=self.initial_design_path,
            run_controller=run_controller,
        )
        return self


class NevergradMetaModelOnePlusOneWarmStart:
    """
    MetaModelOnePlusOne whose first 50 asks are a ranked prefix.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Deterministic seed.
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    name = "NEVERGRAD_METAMODEL_ONEPLUSONE_WARM_START"
    info = "Nevergrad MetaModelOnePlusOne from a ranked InitialDesign prefix"

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
        """Run warm-start MetaModelOnePlusOne."""
        _run_nevergrad_warm_start(
            problem,
            name=self.name,
            engine_name="MetaModelOnePlusOne",
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            initial_design_path=self.initial_design_path,
            run_controller=run_controller,
        )
        return self
