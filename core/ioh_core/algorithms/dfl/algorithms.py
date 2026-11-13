"""
DFLINT BBOA adapter for constrained IOHexperimenter catalogue-index problems.
"""
from __future__ import annotations

import importlib
from random import Random
from typing import Any, Sequence

import numpy as np

from core.ioh_core.algorithms.common import (
    CandidateError,
    get_attached_eval_context,
    read_last_evaluation_constraint_slacks,
    read_last_evaluation_objective_y,
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)
from core.ioh_core.algorithms.warm_start import validate_warm_start_candidate

_NONMONOTONE_MEMORY = 4


class BudgetExhausted(Exception):
    """
    Raised when the hard IOH budget or fixed-target stop forbids further
    evaluations.
    """


def _count_ioh_constraints(problem: Any) -> int:
    """
    Return the number of active IOH inequality constraints for ``problem``.

    Prefers attached EvalContext ``_c_names``; falls back to
    ``problem.constraints.n()``.

    Parameters
    ----------
    problem : Any
        IOH-like problem with an attached EvalContext and/or constraint set.

    Returns
    -------
    int
        Non-negative constraint count.

    Raises
    ------
    ValueError
        If the constraint count cannot be determined.
    """
    ctx = get_attached_eval_context(problem)
    if ctx is not None:
        names = getattr(ctx, "_c_names", None)
        if isinstance(names, list):
            return len(names)

    constraints = getattr(problem, "constraints", None)
    if constraints is not None:
        n_method = getattr(constraints, "n", None)
        if callable(n_method):
            try:
                return int(n_method())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "problem.constraints.n() did not return an integer "
                    f"constraint count: {exc}"
                ) from exc
        if isinstance(n_method, int) and not isinstance(n_method, bool):
            return int(n_method)

    raise ValueError(
        "cannot determine IOH constraint count; attach an EvalContext with "
        "ensure_constraints(...) or expose problem.constraints.n()"
    )


def _sample_integer_vector(
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    rng: Random,
) -> list[int]:
    """
    Draw one inclusive integer vector from ``rng``.

    Parameters
    ----------
    lower_bounds : sequence of int
        Inclusive lower bounds.
    upper_bounds : sequence of int
        Inclusive upper bounds.
    rng : Random
        Seeded RNG.

    Returns
    -------
    list of int
        One candidate.
    """
    return [
        rng.randint(int(lo), int(hi))
        for lo, hi in zip(lower_bounds, upper_bounds)
    ]


def _resolve_dflint_start(
    problem: Any,
    *,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    seed: int | None,
    initial_x: Sequence[int] | None,
    algorithm_name: str,
) -> list[int]:
    """
    Return the integer start: a loaded vector or a seeded sample.

    A loaded vector is checked against the catalogue bounds.
    """
    if initial_x is None:
        if seed is None:
            raise ValueError(f"{algorithm_name} requires a seed or a sealed initial_x")
        return _sample_integer_vector(lower_bounds, upper_bounds, Random(seed))

    return validate_warm_start_candidate(
        problem,
        initial_x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        algorithm_name=algorithm_name,
    )


def _load_dflint_module() -> Any:
    """
    Import the vendored DFLINT module (lazy).

    Returns
    -------
    module
        Vendored ``DFLINT`` module exposing ``DFLINT`` and ``func_f``.
    """
    return importlib.import_module(
        "core.ioh_core.algorithms.dfl._vendor.DFLINT"
    )


class DflintBboa:
    """
    DFLINT BBOA on integer catalogue indices with native inequality constraints.

    Authoritative evaluation counts and best solutions come from the IOH
    problem state. The optimiser's returned ``(x, f)`` is ignored for
    reporting.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int or None
        Deterministic seed for a random initial integer vector. Required
        unless ``initial_x`` is set.
    initial_x : sequence of int or None, optional
        Sealed catalogue-index start. When set, ``seed`` is unused.
    initial_design_path : str or None, optional
        Ranked InitialDesign JSON. Rank 1 is loaded at call time.
        Mutually exclusive with ``initial_x``.
    name : str, optional
        Registry token stored on the instance.
    """

    name = "DFLINT_BBOA"
    info = (
        "DFLINT BBOA (Derivative-Free Library); minimizes unpenalized y "
        "subject to signed g=η−1 constraints"
    )

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None,
        *,
        initial_x: Sequence[int] | None = None,
        initial_design_path: str | None = None,
        name: str | None = None,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = None if seed is None else int(seed)
        self._initial_x = (
            None if initial_x is None else tuple(int(v) for v in initial_x)
        )
        self._initial_design_path = (
            None if initial_design_path is None else str(initial_design_path)
        )
        if name is not None:
            self.name = name
        if self._initial_x is not None and self._initial_design_path is not None:
            raise ValueError(
                f"{self.name} accepts either initial_x or "
                "initial_design_path, not both"
            )
        if (
            self._initial_x is None
            and self._initial_design_path is None
            and self.seed is None
        ):
            raise ValueError(
                f"{self.name} requires a seed, a sealed initial_x, or "
                "initial_design_path"
            )
        if self._initial_x is not None or self._initial_design_path is not None:
            self.info = (
                "DFLINT BBOA from rank 1 of a ranked InitialDesign; "
                "minimizes unpenalized y subject to signed g=η−1 constraints"
            )

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run DFLINT BBOA on a constrained IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds, callable evaluation, and attached
            EvalContext with at least one active constraint.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        DflintBboa
            ``self`` after the run; best values live on the IOH problem state.

        Raises
        ------
        ValueError
            If the problem has no active constraints, or a loaded start
            is incompatible with the problem.
        CandidateError
            If the loaded start is outside inclusive bounds.
        """
        mm = _count_ioh_constraints(problem)
        if mm < 1:
            raise ValueError(
                f"{self.name} requires at least one active IOH "
                f"constraint; got mm={mm}"
            )

        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        lb = np.asarray(lower_bounds, dtype=float)
        ub = np.asarray(upper_bounds, dtype=float)
        if self._initial_design_path is not None:
            from core.ioh_core.algorithms.warm_start import rank1_start_vector

            initial_x = rank1_start_vector(
                problem,
                algorithm_name=self.name,
                initial_design_path=self._initial_design_path,
                max_evaluations=self.max_evaluations,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
        else:
            initial_x = self._initial_x
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        x0 = np.asarray(
            _resolve_dflint_start(
                problem,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                seed=self.seed,
                initial_x=initial_x,
                algorithm_name=self.name,
            ),
            dtype=float,
        )

        dflint = _load_dflint_module()

        def func_f(x: Any) -> tuple[float, np.ndarray]:
            if not controller.can_evaluate():
                raise BudgetExhausted(
                    f"{self.name} hard IOH evaluation budget exhausted"
                )
            candidate = validate_integer_candidate(
                x,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            controller.require_evaluation()
            y_p = float(problem(candidate))
            y = read_last_evaluation_objective_y(problem)
            g = read_last_evaluation_constraint_slacks(problem)
            if len(g) != mm:
                raise ValueError(
                    f"expected {mm} constraint slacks, got {len(g)}"
                )
            controller.observe_evaluation(y_p)
            return float(y), np.asarray(g, dtype=float)

        previous = getattr(dflint, "func_f", None)
        dflint.func_f = func_f
        try:
            dflint.DFLINT(
                "BBOA",
                int(mm),
                _NONMONOTONE_MEMORY,
                x0,
                lb,
                ub,
                int(self.max_evaluations),
                0,
            )
        except BudgetExhausted:
            pass
        finally:
            if previous is None:
                try:
                    delattr(dflint, "func_f")
                except AttributeError:
                    dflint.func_f = None
            else:
                dflint.func_f = previous
        return self
