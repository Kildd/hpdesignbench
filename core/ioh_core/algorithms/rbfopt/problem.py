"""
IOH problem adapter for RBFOpt native-integer black-box evaluation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from rbfopt import RbfoptBlackBox

from core.ioh_core.algorithms.common import (
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)

RBFOPT_POST_STOP_SENTINEL = 1.0e30


class IOHRbfoptBlackBox(RbfoptBlackBox):
    """
    RBFOpt black box over inclusive IOH catalogue-index variables.

    Every variable is declared as integer type ``I``. Noisy evaluation is
    disabled. Authoritative evaluation counts live on the IOH problem state.
    After the shared controller stops (budget or target), further library
    requests receive :data:`RBFOPT_POST_STOP_SENTINEL` without an IOH call.

    Parameters
    ----------
    problem : Any
        IOH-like problem with bounds and callable evaluation.
    max_evaluations : int
        Hard upper bound on IOH evaluations for this run.
    run_controller : optional
        Optional target-aware evaluation controller.
    """

    def __init__(
        self,
        problem: Any,
        max_evaluations: int,
        *,
        run_controller=None,
    ) -> None:
        dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        self.ioh_problem = problem
        self.dimension = int(dimension)
        self.lower_bounds = list(lower_bounds)
        self.upper_bounds = list(upper_bounds)
        self.controller = resolve_run_controller(
            problem,
            max_evaluations=max_evaluations,
            run_controller=run_controller,
        )
        self.budget = self.controller._budget
        self._var_lower = np.asarray(self.lower_bounds, dtype=float)
        self._var_upper = np.asarray(self.upper_bounds, dtype=float)
        self._var_type = np.asarray(["I"] * self.dimension)

    def get_dimension(self) -> int:
        """
        Return the number of decision variables.

        Returns
        -------
        int
            Problem dimension.
        """
        return self.dimension

    def get_var_lower(self) -> np.ndarray:
        """
        Return a copy of the inclusive lower bounds.

        Returns
        -------
        numpy.ndarray
            Float lower bounds of length ``dimension``.
        """
        return self._var_lower.copy()

    def get_var_upper(self) -> np.ndarray:
        """
        Return a copy of the inclusive upper bounds.

        Returns
        -------
        numpy.ndarray
            Float upper bounds of length ``dimension``.
        """
        return self._var_upper.copy()

    def get_var_type(self) -> np.ndarray:
        """
        Return a copy of the variable-type vector (all ``I``).

        Returns
        -------
        numpy.ndarray
            Character codes of length ``dimension``.
        """
        return self._var_type.copy()

    def evaluate(self, x: np.ndarray) -> float:
        """
        Validate ``x`` as an integer catalogue vector and evaluate IOH once.

        Parameters
        ----------
        x : numpy.ndarray
            Candidate coordinates (may be integral floats).

        Returns
        -------
        float
            IOH objective value, or :data:`RBFOPT_POST_STOP_SENTINEL` when the
            controller has already stopped.
        """
        if not self.controller.can_evaluate():
            return float(RBFOPT_POST_STOP_SENTINEL)
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
        return y

    def evaluate_noisy(self, x: np.ndarray) -> list[float]:
        """
        Noisy evaluation is disabled for this adapter.

        Parameters
        ----------
        x : numpy.ndarray
            Unused candidate; present for the RBFOpt API.

        Raises
        ------
        NotImplementedError
            Always; noisy evaluation is not enabled.
        """
        raise NotImplementedError(
            "IOHRbfoptBlackBox does not implement noisy evaluation"
        )

    def has_evaluate_noisy(self) -> bool:
        """
        Report whether noisy evaluation is available.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False
