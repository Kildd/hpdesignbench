"""
SciPy dual annealing on a continuous-rounded catalogue surface.

The adapter runs SciPy's global phase only (generalised simulated annealing).
Local search stays off. The objective is penalized ``y_p``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from core.ioh_core.algorithms.common import (
    BudgetError,
    continuous_to_integer_candidate,
    read_problem_bounds,
    resolve_run_controller,
)

_INITIAL_TEMP = 5230.0
_RESTART_TEMP_RATIO = 2.0e-5
_VISIT = 2.62
_ACCEPT = -5.0


class _ScipyBudgetStop(Exception):
    """
    Internal signal to leave ``dual_annealing`` early.

    Raised from the objective when the shared run controller forbids further
    IOH evaluations (hard budget or fixed-target hit). Caught only around the
    SciPy call; never treated as a solver result.
    """


def _require_nondegenerate_bounds(
    lower_bounds: list[int],
    upper_bounds: list[int],
) -> None:
    """
    Reject a coordinate whose continuous interval would be empty.

    Parameters
    ----------
    lower_bounds, upper_bounds : list of int
        Inclusive catalogue-index bounds.

    Raises
    ------
    ValueError
        If any coordinate has ``upper <= lower``. SciPy is not called.
    """
    for index, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
        if int(upper) <= int(lower):
            raise ValueError(
                "dual_annealing requires non-degenerate continuous bounds;\n"
                f"coordinate {index} has bounds [{int(lower)}, {int(upper)}]"
            )


def _run_dual_annealing(
    problem: Any,
    *,
    max_evaluations: int,
    seed: int,
    x0: np.ndarray | None,
    run_controller: Any,
) -> None:
    """
    Run SciPy dual annealing with the locked global-phase configuration.

    Parameters
    ----------
    problem : Any
        IOH-like problem exposing bounds and callable evaluation.
    max_evaluations : int
        Hard upper bound on counted IOH evaluations.
    seed : int
        Task seed passed as SciPy ``rng``.
    x0 : ndarray or None
        Explicit start. ``None`` leaves initialisation to SciPy.
    run_controller : optional
        Optional target-aware evaluation controller.
    """
    from scipy.optimize import dual_annealing

    _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
    _require_nondegenerate_bounds(lower_bounds, upper_bounds)
    controller = resolve_run_controller(
        problem,
        max_evaluations=max_evaluations,
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
        x_int = continuous_to_integer_candidate(
            x,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        y_p = float(problem(x_int))
        controller.observe_evaluation(
            y_p,
        )
        if controller.should_stop():
            raise _ScipyBudgetStop()
        return y_p

    try:
        dual_annealing(
            objective,
            bounds,
            maxiter=int(max_evaluations),
            minimizer_kwargs=None,
            initial_temp=_INITIAL_TEMP,
            restart_temp_ratio=_RESTART_TEMP_RATIO,
            visit=_VISIT,
            accept=_ACCEPT,
            maxfun=int(max_evaluations),
            rng=int(seed),
            no_local_search=True,
            x0=x0,
        )
    except _ScipyBudgetStop:
        pass


class DualAnnealing:
    """
    SciPy dual annealing with no supplied start and local search disabled.

    ``x0`` is ``None``. SciPy initialises the search from ``rng``. Each
    continuous proposal is rounded onto the catalogue and costs one IOH
    evaluation, including a repeat of an integer cell.

    Parameters
    ----------
    max_evaluations : int
        Hard upper bound on counted IOH evaluations. Also passed as
        ``maxfun`` and ``maxiter``.
    seed : int
        Deterministic seed passed as SciPy ``rng``.
    """

    name = "SCIPY_DUAL_ANNEALING"
    info = (
        "SciPy dual annealing (global phase only; local search disabled) "
        "on continuous-rounded catalogue indices; minimizes y_p; "
        "no initial point supplied"
    )

    def __init__(self, max_evaluations: int, seed: int) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run cold dual annealing on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        DualAnnealing
            ``self`` after the run; best values live on the IOH problem state.
        """
        _run_dual_annealing(
            problem,
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            x0=None,
            run_controller=run_controller,
        )
        return self


class DualAnnealingWarmStart:
    """
    SciPy dual annealing from rank 1 of a ranked InitialDesign.

    The search matches :class:`DualAnnealing` except ``x0``, which is the
    loaded catalogue-index vector as floats. The task seed still drives the
    visiting distribution. A temperature restart inside SciPy may leave
    that point; evaluation 1 is rank 1.

    Parameters
    ----------
    max_evaluations : int
        Hard upper bound on counted IOH evaluations.
    seed : int
        Deterministic seed passed as SciPy ``rng``.
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    name = "SCIPY_DUAL_ANNEALING_WARM_START"
    info = (
        "SciPy dual annealing (global phase only; local search disabled) "
        "from rank 1 of a ranked InitialDesign; minimizes y_p; seeded"
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
        self.initial_design_path = str(initial_design_path)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run warm-start dual annealing on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        DualAnnealingWarmStart
            ``self`` after the run.
        """
        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=1,
            max_evaluations=self.max_evaluations,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        _run_dual_annealing(
            problem,
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            x0=np.asarray(prefix[0], dtype=float),
            run_controller=run_controller,
        )
        return self
