"""
CatCMAwM wrapper for IOHexperimenter catalogue-index problems.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from cmaes import CatCMAwM

from core.ioh_core.algorithms.common import (
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)

CMAES_BUDGET_EXHAUSTED_FITNESS = 1.0e30


def build_integer_z_space(
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> list[list[int]]:
    """
    Build CatCMAwM ``z_space`` from inclusive IOH catalogue-index bounds.

    Parameters
    ----------
    lower_bounds : sequence of int
        Inclusive lower index per dimension.
    upper_bounds : sequence of int
        Inclusive upper index per dimension.

    Returns
    -------
    list of list of int
        Per-dimension sorted allowed integer values.

    Raises
    ------
    ValueError
        If any dimension has fewer than two inclusive indices.
    """
    if len(lower_bounds) != len(upper_bounds):
        raise ValueError("lower/upper bound lengths disagree")
    z_space: list[list[int]] = []
    for index, (lo, hi) in enumerate(zip(lower_bounds, upper_bounds)):
        lo_i = int(lo)
        hi_i = int(hi)
        if hi_i <= lo_i:
            raise ValueError(
                "CMAES_CATCMAWM requires at least two inclusive catalogue "
                f"indices per dimension; dimension {index} has "
                f"lb={lo_i}, ub={hi_i}"
            )
        z_space.append(list(range(lo_i, hi_i + 1)))
    return z_space


def integer_box_centre_covariance(
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> np.ndarray:
    """
    Return the diagonal covariance a cold CatCMAwM builds from the box centre.

    For an integer-only space the library uses ``width / 5``, where ``width``
    is the distance from the centre to the nearer bound. A warm start must
    pass this matrix explicitly so a mean on a bound does not freeze that
    coordinate.

    Parameters
    ----------
    lower_bounds, upper_bounds : sequence of int
        Inclusive catalogue-index bounds.

    Returns
    -------
    numpy.ndarray
        Diagonal covariance of shape ``(dimension, dimension)``.
    """
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    centre = (lower + upper) / 2.0
    width = np.minimum(centre - lower, upper - centre) / 5.0
    return np.diag(width**2)


class CatCmaWMAlgorithm:
    """
    CatCMAwM ask–tell search over IOH catalogue indices.

    Authoritative evaluation counts and best solutions come from the IOH
    problem state. CatCMAwM's internal ranking is ignored for reporting.

    Minimizes the unconstrained penalized IOH objective ``y_p``. Mid-batch
    budget exhaustion pads remaining ``ask`` solutions with
    :data:`CMAES_BUDGET_EXHAUSTED_FITNESS` so ``tell`` still receives a full
    population.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    population_size : int, optional
        CatCMAwM population size (>= 2). Default is ``50``.
    seed : int or None, optional
        Deterministic seed required by the registry seed policy.
    """

    name = "CMAES_CATCMAWM"
    info = (
        "CatCMAwM (cmaes) on integer catalogue indices; "
        "minimizes unconstrained y_p"
    )

    def __init__(
        self,
        max_evaluations: int,
        population_size: int = 50,
        seed: int | None = None,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.population_size = int(population_size)
        self.seed = seed

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run CatCMAwM on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        CatCmaWMAlgorithm
            ``self`` after the run; best values live on the IOH problem state.
        """
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        z_space = build_integer_z_space(lower_bounds, upper_bounds)
        optimizer = CatCMAwM(
            z_space=z_space,
            population_size=self.population_size,
            seed=int(self.seed) if self.seed is not None else None,
        )

        while not controller.should_stop():
            if not controller.can_evaluate():
                break
            solutions: list[tuple[Any, float]] = []
            for _ in range(optimizer.population_size):
                sol = optimizer.ask()
                if not controller.can_evaluate():
                    solutions.append((sol, CMAES_BUDGET_EXHAUSTED_FITNESS))
                    continue
                candidate = validate_integer_candidate(
                    sol.z,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                )
                controller.require_evaluation()
                y = float(problem(candidate))
                controller.observe_evaluation(y)
                solutions.append((sol, y))
            optimizer.tell(solutions)

        return self


class CatCmaWMWarmStart:
    """
    CatCMAwM whose initial mean is rank 1 of a ranked InitialDesign.

    The covariance is the cold diagonal from the box centre, not the
    variance the library would derive from that mean. Rank 1 is distribution
    state and is not evaluated before the first ``ask``.
    """

    name = "CMAES_CATCMAWM_WARM_START"
    info = (
        "CatCMAwM from rank 1 as the initial mean; cold box-centre "
        "covariance; minimizes unconstrained y_p"
    )

    def __init__(
        self,
        max_evaluations: int,
        population_size: int = 50,
        seed: int | None = None,
        *,
        initial_design_path: str,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.population_size = int(population_size)
        self.seed = seed
        self.initial_design_path = str(initial_design_path)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run CatCMAwM from the loaded mean.

        Parameters
        ----------
        problem : Any
            IOH-like problem.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        CatCmaWMWarmStart
            ``self`` after the run.
        """
        from core.ioh_core.algorithms.warm_start import (
            initial_design_prefix,
            warm_start_problem_id,
            validate_prefix_bounds,
        )

        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        problem_id = warm_start_problem_id(problem)
        prefix = initial_design_prefix(
            self.initial_design_path,
            1,
            algorithm_name=self.name,
            problem_id=problem_id,
        )
        validate_prefix_bounds(prefix, lower_bounds, upper_bounds)
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        z_space = build_integer_z_space(lower_bounds, upper_bounds)
        optimizer = CatCMAwM(
            z_space=z_space,
            population_size=self.population_size,
            mean=np.asarray(prefix[0], dtype=float),
            cov=integer_box_centre_covariance(lower_bounds, upper_bounds),
            seed=int(self.seed) if self.seed is not None else None,
        )

        while not controller.should_stop():
            if not controller.can_evaluate():
                break
            solutions: list[tuple[Any, float]] = []
            for _ in range(optimizer.population_size):
                sol = optimizer.ask()
                if not controller.can_evaluate():
                    solutions.append((sol, CMAES_BUDGET_EXHAUSTED_FITNESS))
                    continue
                candidate = validate_integer_candidate(
                    sol.z,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                )
                controller.require_evaluation()
                y = float(problem(candidate))
                controller.observe_evaluation(y)
                solutions.append((sol, y))
            optimizer.tell(solutions)

        return self
