"""
SciPy locally-biased DIRECT-L on a continuous-to-integer rounded surface.
"""
from __future__ import annotations

from typing import Any

from core.ioh_core.algorithms.common import (
    BudgetError,
    continuous_to_integer_candidate,
    read_problem_bounds,
    resolve_run_controller,
)


class _ScipyBudgetStop(Exception):
    """
    Internal signal to leave SciPy ``direct`` early.

    Raised from the objective when the shared run controller forbids further
    IOH evaluations. SciPy's C extension may surface this as ``SystemError``;
    the adapter catches that only when an intentional stop was requested.
    """


class DirectL:
    """
    SciPy locally-biased DIRECT-L on a continuous-to-integer rounded surface.

    Uses ``scipy.optimize.direct`` with ``locally_biased=True`` (DIRECT-L).
    Continuous proposals are converted with
    ``continuous_to_integer_candidate``. SciPy ``maxfun`` is best-effort only;
    hard IOH budgeting uses the shared run controller plus a flag-guarded
    ``SystemError`` / ``_ScipyBudgetStop`` catch.

    This token is distinct from ``NLOPT_DIRECT_L``.

    Parameters
    ----------
    max_evaluations : int
        Hard upper bound on counted IOH evaluations.
    """

    name = "SCIPY_DIRECT_L"
    info = (
        "SciPy locally-biased DIRECT-L on a continuous-to-integer rounded "
        "surface"
    )

    def __init__(self, max_evaluations: int) -> None:
        self.max_evaluations = int(max_evaluations)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run SciPy DIRECT-L on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        DirectL
            ``self`` after the run; best values live on the IOH problem state.

        Raises
        ------
        RuntimeError
            If SciPy finishes without any IOH evaluation.
        SystemError
            If SciPy raises ``SystemError`` without an intentional budget stop.
        """
        from scipy.optimize import direct

        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        # SciPy direct requires strict lb < ub. Widen degenerate catalogue
        # dimensions for the solver only; continuous_to_integer_candidate still
        # clips back to the fixed inclusive integer index.
        bounds = []
        for lower, upper in zip(lower_bounds, upper_bounds):
            lo = float(lower)
            hi = float(upper)
            if lo == hi:
                bounds.append((lo - 0.5, hi + 0.5))
            else:
                bounds.append((lo, hi))

        stop_requested = False
        n_ioh_evaluations = 0

        def objective(x: Any) -> float:
            nonlocal stop_requested, n_ioh_evaluations
            if not controller.can_evaluate():
                stop_requested = True
                raise _ScipyBudgetStop()
            try:
                controller.require_evaluation()
            except BudgetError as exc:
                stop_requested = True
                raise _ScipyBudgetStop() from exc

            x_int = continuous_to_integer_candidate(
                x,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            n_ioh_evaluations += 1
            y = float(problem(x_int))
            controller.observe_evaluation(
                y,
            )
            if controller.should_stop():
                stop_requested = True
                raise _ScipyBudgetStop()
            return y

        try:
            direct(
                objective,
                bounds,
                maxfun=self.max_evaluations,
                maxiter=max(self.max_evaluations, 1000),
                locally_biased=True,
            )
        except _ScipyBudgetStop:
            pass
        except SystemError:
            if not stop_requested:
                raise

        if n_ioh_evaluations == 0:
            raise RuntimeError(
                "SCIPY_DIRECT_L finished without evaluating any candidate"
            )
        return self
