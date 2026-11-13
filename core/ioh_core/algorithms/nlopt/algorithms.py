"""
NLopt algorithm wrappers for IOHexperimenter problems.
"""
from __future__ import annotations

from typing import Any

import nlopt

from core.ioh_core.algorithms.common import (
    BudgetError,
    continuous_to_integer_candidate,
    read_problem_bounds,
    resolve_run_controller,
    seeded_uniform_continuous_start,
)


def _optimize_rounded_surface(
    *,
    problem: Any,
    max_evaluations: int,
    algorithm: int,
    run_controller,
    empty_error: str,
    x0: list[float] | None = None,
) -> tuple[float, list[int], int]:
    """
    Run one NLopt algorithm on a continuous-to-integer rounded surface.

    Parameters
    ----------
    problem : Any
        IOH-like problem with bounds and callable evaluation.
    max_evaluations : int
        Hard upper bound on IOH evaluations.
    algorithm : int
        NLopt algorithm code (for example ``nlopt.GN_DIRECT_L``).
    run_controller : optional
        Optional target-aware evaluation controller.
    empty_error : str
        RuntimeError message when no candidate was evaluated.
    x0 : list of float or None, optional
        Continuous start vector. When ``None``, use the bounds midpoint
        (deterministic global methods such as DIRECT-L).

    Returns
    -------
    tuple[float, list[int], int]
        Best objective, best integer candidate, and IOH evaluation count.

    Raises
    ------
    RuntimeError
        If NLopt finishes without evaluating any candidate.
    """
    controller = resolve_run_controller(
        problem,
        max_evaluations=max_evaluations,
        run_controller=run_controller,
    )
    dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
    lb = [float(value) for value in lower_bounds]
    ub = [float(value) for value in upper_bounds]

    opt = nlopt.opt(algorithm, dimension)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_maxeval(max_evaluations)
    # NLopt stopval is numeric. The shared controller records the hit and
    # refuses further evaluations.
    if (
        controller.mode == "fixed_target"
        and controller.target_threshold is not None
    ):
        opt.set_stopval(float(controller.target_threshold))

    n_optimizer_evaluations = 0
    best_x: list[int] | None = None
    best_y: float | None = None

    def objective(x, grad):
        nonlocal best_x, best_y, n_optimizer_evaluations
        if not controller.can_evaluate():
            raise nlopt.ForcedStop()
        try:
            controller.require_evaluation()
        except BudgetError as exc:
            raise nlopt.ForcedStop() from exc

        x_int = continuous_to_integer_candidate(
            x,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        n_optimizer_evaluations += 1
        y = float(problem(x_int))
        controller.observe_evaluation(
            y,
        )
        if best_y is None or y < best_y:
            best_y = y
            best_x = list(x_int)
        # Return normally after a completed evaluation. Raising ForcedStop here
        # can become SystemError in the NLopt Python bindings. Early stop on a
        # hit uses set_stopval and ForcedStop on the
        # next callback when can_evaluate() is False.
        return y

    opt.set_min_objective(objective)
    if x0 is None:
        start = [(lo + hi) / 2.0 for lo, hi in zip(lb, ub)]
    else:
        if len(x0) != dimension:
            raise ValueError(
                f"x0 length {len(x0)} does not match dimension {dimension}"
            )
        start = [float(v) for v in x0]

    try:
        opt.optimize(start)
    except (nlopt.RoundoffLimited, nlopt.ForcedStop):
        # Keep the best actually evaluated integer candidate; never treat
        # last_optimize_result() (a status code) as a vector.
        pass

    if best_x is None or best_y is None:
        raise RuntimeError(empty_error)

    return float(best_y), list(best_x), n_optimizer_evaluations


class NLoptDirectLocalSearch:
    """
    NLopt DIRECT-L wrapper around an IOHexperimenter problem.
    Adapted from: Max Dombrowski

    Uses the locally biased variant of the DIviding RECTangles (DIRECT)
    algorithm (``nlopt.GN_DIRECT_L``) for derivative-free global
    optimization over integer index bounds.

    Continuous NLopt proposals are converted with the shared
    continuous-to-integer policy. The best returned point is always an
    actually evaluated integer candidate; the wrapper never re-evaluates
    that best point and never treats NLopt status codes as vectors.

    Attributes
    ----------
    max_evaluations : int
        Maximum number of objective function evaluations [-].
    name : str
        Algorithm identifier string used by the IOH logger.
    info : str
        Human-readable algorithm description.
    """

    def __init__(self, max_evaluations: int = 50):
        """
        Parameters
        ----------
        max_evaluations : int, optional
            Maximum number of objective function evaluations.
            Default is ``50``.
        """
        self.max_evaluations = int(max_evaluations)
        self.name = "NLOPT_DIRECT_L"
        self.info = (
            "locally biased variant of DIviding RECTangles (DIRECT) "
            "algorithm for global optimization"
        )  # (Added by: Elliot Melcer)
        self.n_optimizer_evaluations_ = 0

    def __call__(self, problem, *, run_controller=None):
        """
        Run the DIRECT-L optimizer on an IOH problem.

        Parameters
        ----------
        problem : ioh.Problem
            An IOHexperimenter problem instance exposing
            ``meta_data.n_variables``, ``bounds.lb``, ``bounds.ub``,
            and callable evaluation via ``problem(x_int)``.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        tuple[float, list[int]]
            ``(f_best, x_best_int)`` — the best objective value found and
            the corresponding integer index vector.
        """
        best_y, best_x, n_evals = _optimize_rounded_surface(
            problem=problem,
            max_evaluations=self.max_evaluations,
            algorithm=nlopt.GN_DIRECT_L,
            run_controller=run_controller,
            empty_error=(
                "NLOPT_DIRECT_L finished without evaluating any candidate"
            ),
        )
        self.n_optimizer_evaluations_ = n_evals
        return best_y, best_x


class NLoptSbplx:
    """
    NLopt SBPLX compatibility search on a rounded integer surface.

    Uses ``nlopt.LN_SBPLX`` with the shared continuous-to-integer conversion.
    This is a compatibility benchmark on a discontinuous rounded surface, not
    native integer optimization. The returned best is always an actually
    evaluated integer candidate. The continuous start is drawn uniformly in
    the bounds box from ``seed`` (multistart across repetitions).

    Attributes
    ----------
    name : str
        Registry token.
    info : str
        Human-readable description.
    max_evaluations : int
        Hard upper bound on objective callbacks / IOH evaluations.
    seed : int
        Task seed for the uniform continuous start.
    """

    name = "NLOPT_SBPLX"
    info = (
        "NLopt SBPLX compatibility search on a rounded integer surface; "
        "seeded random start"
    )

    def __init__(self, max_evaluations: int = 50, *, seed: int) -> None:
        """
        Parameters
        ----------
        max_evaluations : int, optional
            Maximum number of objective function evaluations.
            Default is ``50``.
        seed : int
            Deterministic seed for the continuous start point.
        """
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)
        self.n_optimizer_evaluations_ = 0

    def __call__(
        self, problem: Any, *, run_controller=None
    ) -> tuple[float, list[int]]:
        """
        Run SBPLX on an IOH problem with continuous-to-integer rounding.

        Parameters
        ----------
        problem : Any
            IOH-like problem with bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        tuple[float, list[int]]
            ``(f_best, x_best_int)`` from evaluated candidates only.

        Raises
        ------
        RuntimeError
            If NLopt finishes without evaluating any candidate.
        """
        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        x0 = seeded_uniform_continuous_start(
            lower_bounds,
            upper_bounds,
            self.seed,
        )
        best_y, best_x, n_evals = _optimize_rounded_surface(
            problem=problem,
            max_evaluations=self.max_evaluations,
            algorithm=nlopt.LN_SBPLX,
            run_controller=run_controller,
            empty_error=(
                "NLOPT_SBPLX finished without evaluating any candidate"
            ),
            x0=x0,
        )
        self.n_optimizer_evaluations_ = n_evals
        return best_y, best_x


class NLoptSbplxWarmStart:
    """
    NLopt SBPLX from rank 1 of a ranked InitialDesign.

    The search matches :class:`NLoptSbplx` except ``x0``, which is the
    loaded catalogue-index vector as floats. No seed is used.
    """

    name = "NLOPT_SBPLX_WARM_START"
    info = (
        "NLopt SBPLX compatibility search on a rounded integer surface; "
        "rank-1 InitialDesign start"
    )

    def __init__(
        self,
        max_evaluations: int = 50,
        *,
        initial_design_path: str,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.initial_design_path = str(initial_design_path)
        self.n_optimizer_evaluations_ = 0

    def __call__(
        self, problem: Any, *, run_controller=None
    ) -> tuple[float, list[int]]:
        """
        Run SBPLX from the loaded rank-1 vector.

        Parameters
        ----------
        problem : Any
            IOH-like problem with bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        tuple[float, list[int]]
            ``(f_best, x_best_int)`` from evaluated candidates only.
        """
        from core.ioh_core.algorithms.warm_start import rank1_start_vector

        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        x0 = [
            float(value)
            for value in rank1_start_vector(
                problem,
                algorithm_name=self.name,
                initial_design_path=self.initial_design_path,
                max_evaluations=self.max_evaluations,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
        ]
        best_y, best_x, n_evals = _optimize_rounded_surface(
            problem=problem,
            max_evaluations=self.max_evaluations,
            algorithm=nlopt.LN_SBPLX,
            run_controller=run_controller,
            empty_error=(
                "NLOPT_SBPLX_WARM_START finished without evaluating any "
                "candidate"
            ),
            x0=x0,
        )
        self.n_optimizer_evaluations_ = n_evals
        return best_y, best_x
