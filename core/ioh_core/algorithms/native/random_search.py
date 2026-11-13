"""
Native random search over IOH integer catalogue-index bounds.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from core.ioh_core.algorithms.common import (
    resolve_run_controller,
)


class RandomSearch:
    """
    Simple random search over integer index bounds.
    Author: Max Dombrowski

    Samples uniformly from the integer index ranges defined by the IOH
    problem bounds and tracks the best objective value found.

    Attributes
    ----------
    name : str
        Algorithm identifier string used by the IOH logger.
    info : str
        Human-readable algorithm description.
    evaluations : int
        Number of random samples to draw per call [-].
    rng : numpy.random.Generator
        Random number generator instance.
    best_y_ : float
        Best objective value found in the last call (set after ``__call__``).
    best_x_ : list[int]
        Integer index vector achieving ``best_y_`` (set after ``__call__``).

    """

    def __init__(self, max_evaluations: int, seed: int | None = None):
        """
        Parameters
        ----------
        max_evaluations : int
            Number of random samples to draw per optimization call.
        seed : int or None, optional
            Seed for the random number generator. ``None`` uses a
            non-deterministic seed. Default is ``None``.
        """
        self.name = "RANDOM_SEARCH"
        self.info = "Very simple random search over integer index bounds"
        self.evaluations = int(max_evaluations)
        self.rng = np.random.default_rng(seed)

    def __call__(self, problem: Any, *, run_controller=None):
        """
        Run the random search on an IOH problem.

        Parameters
        ----------
        problem : ioh.Problem
            An IOHexperimenter problem instance exposing
            ``bounds.lb``, ``bounds.ub``, ``meta_data.n_variables``,
            and callable evaluation via ``problem(x)``.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        RandomSearch
            Returns ``self`` with :attr:`best_y_` and :attr:`best_x_`
            updated to the best solution found.
        """
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.evaluations,
            run_controller=run_controller,
        )
        lb = np.asarray(problem.bounds.lb, dtype=int)
        ub = np.asarray(problem.bounds.ub, dtype=int)
        dim = len(lb)

        # safety
        assert dim == problem.meta_data.n_variables

        # sample uniformly over integer index ranges
        best_y = None
        best_x = None
        for _ in range(self.evaluations):
            if not controller.can_evaluate():
                break
            controller.require_evaluation()
            x = self.rng.integers(low=lb, high=ub + 1)  # inclusive upper bound for indices
            y = problem(x)  # IOH problem is callable; evaluates objective (+ constraints internally)
            controller.observe_evaluation(
                y,
            )
            if (best_y is None) or (y < best_y):
                best_y, best_x = y, x.copy()
            if controller.should_stop():
                break

        # store for convenience
        if best_y is None or best_x is None:
            self.best_y_ = float("inf")
            self.best_x_ = [int(v) for v in lb]
        else:
            self.best_y_ = float(best_y)
            self.best_x_ = [int(v) for v in best_x]
        return self
