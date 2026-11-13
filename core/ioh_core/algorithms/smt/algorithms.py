"""
SMT EGO wrapper for IOHexperimenter catalogue-index problems.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from smt.applications.ego import Evaluator

from core.ioh_core.algorithms.common import (
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)
from core.ioh_core.algorithms.smt.space import build_integer_design_space


class BudgetExhausted(Exception):
    """
    Raised by :class:`BudgetAwareEvaluator` when the hard IOH budget or
    fixed-target stop forbids further evaluations.
    """


class BudgetAwareEvaluator(Evaluator):
    """
    SMT ``Evaluator`` that gates each row through ``RunController``.

    Parameters
    ----------
    problem : Any
        IOH-like problem callable.
    controller : Any
        Target-aware evaluation controller.
    lower_bounds : sequence of int
        Inclusive lower catalogue bounds.
    upper_bounds : sequence of int
        Inclusive upper catalogue bounds.
    """

    def __init__(
        self,
        problem: Any,
        controller: Any,
        lower_bounds: Sequence[int],
        upper_bounds: Sequence[int],
    ) -> None:
        super().__init__()
        self._problem = problem
        self._controller = controller
        self._lower_bounds = list(lower_bounds)
        self._upper_bounds = list(upper_bounds)

    def run(
        self,
        fun: Any,
        x: np.ndarray,
        design_space: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Evaluate ``x`` row-by-row under the hard IOH controller.

        Parameters
        ----------
        fun : Any
            Unused SMT objective handle (evaluation goes through ``problem``).
        x : ndarray
            Candidate batch of shape ``(n_samples, nx)``.
        design_space : optional
            Optional SMT design space for mixed-integer correction.

        Returns
        -------
        ndarray
            Column vector of ``y_p`` values, shape ``(n_samples, 1)``.

        Raises
        ------
        BudgetExhausted
            When no further IOH evaluations are allowed.
        """
        del fun  # IOH problem is authoritative; do not call SMT's fun.
        points = np.atleast_2d(np.asarray(x, dtype=float))
        if design_space is not None and hasattr(
            design_space, "correct_get_acting"
        ):
            points, _ = design_space.correct_get_acting(points)

        objectives = np.empty((points.shape[0], 1), dtype=float)
        for row_index, row in enumerate(points):
            if not self._controller.can_evaluate():
                raise BudgetExhausted(
                    "SMT_EGO hard IOH evaluation budget exhausted"
                )
            candidate = validate_integer_candidate(
                row,
                lower_bounds=self._lower_bounds,
                upper_bounds=self._upper_bounds,
            )
            self._controller.require_evaluation()
            y_p = float(self._problem(candidate))
            self._controller.observe_evaluation(
                y_p,
            )
            objectives[row_index, 0] = y_p
            if self._controller.should_stop():
                raise BudgetExhausted(
                    "SMT_EGO stopped by RunController "
                    "(budget or fixed-target)"
                )
        return objectives


class SmtEgo:
    """
    SMT Efficient Global Optimization on integer catalogue indices.

    Authoritative evaluation counts and best solutions come from the IOH
    problem state. SMT's returned optimum is ignored for reporting.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Deterministic seed for DesignSpace and EGO.
    n_doe : int
        Initial LHS DOE size; enrichment iterations use the remainder.
    """

    name = "SMT_EGO"
    info = (
        "SMT EGO (EI) on integer catalogue indices; "
        "minimizes unconstrained y_p"
    )

    def __init__(
        self,
        max_evaluations: int,
        seed: int,
        n_doe: int,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)
        self.n_doe = int(n_doe)
        if self.n_doe < 1:
            raise ValueError(f"n_doe must be >= 1, got {self.n_doe}")
        if self.n_doe > self.max_evaluations:
            raise ValueError(
                f"n_doe ({self.n_doe}) must be <= max_evaluations "
                f"({self.max_evaluations})"
            )
        self.n_iter = self.max_evaluations - self.n_doe

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run SMT EGO on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        SmtEgo
            ``self`` after the run; best values live on the IOH problem state.
        """
        from smt.applications import EGO
        from smt.surrogate_models import KRG

        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        design_space = build_integer_design_space(
            lower_bounds,
            upper_bounds,
            seed=self.seed,
        )
        evaluator = BudgetAwareEvaluator(
            problem,
            controller,
            lower_bounds,
            upper_bounds,
        )
        ego = EGO(
            n_iter=self.n_iter,
            n_doe=self.n_doe,
            criterion="EI",
            surrogate=KRG(design_space=design_space, print_global=False),
            seed=self.seed,
            n_parallel=1,
            n_start=20,
            n_max_optim=20,
            enable_tunneling=False,
            verbose=False,
            evaluator=evaluator,
        )
        try:
            ego.optimize(fun=lambda x: x)
        except BudgetExhausted:
            pass
        return self


class SmtEgoWarmStart:
    """
    SMT EGO whose DOE is a ranked InitialDesign prefix.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Deterministic seed for DesignSpace and EGO.
    n_doe : int
        Prefix length. Enrichment uses the remainder of the budget.
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    name = "SMT_EGO_WARM_START"
    info = "SMT EGO from a ranked InitialDesign DOE"

    def __init__(
        self,
        max_evaluations: int,
        seed: int,
        n_doe: int,
        *,
        initial_design_path: str,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)
        self.n_doe = int(n_doe)
        self.initial_design_path = initial_design_path

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """Run warm-start SMT EGO."""
        from smt.applications import EGO
        from smt.surrogate_models import KRG

        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=self.n_doe,
            max_evaluations=self.max_evaluations,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        design_space = build_integer_design_space(
            lower_bounds,
            upper_bounds,
            seed=self.seed,
        )
        evaluator = BudgetAwareEvaluator(
            problem,
            controller,
            lower_bounds,
            upper_bounds,
        )
        ego = EGO(
            n_iter=self.max_evaluations - len(prefix),
            n_doe=len(prefix),
            criterion="EI",
            surrogate=KRG(design_space=design_space, print_global=False),
            seed=self.seed,
            n_parallel=1,
            n_start=20,
            n_max_optim=20,
            enable_tunneling=False,
            verbose=False,
            evaluator=evaluator,
            xdoe=np.atleast_2d(np.asarray(prefix, dtype=float)),
        )
        try:
            ego.optimize(fun=lambda x: x)
        except BudgetExhausted:
            pass
        return self
