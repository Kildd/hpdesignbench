"""
Optuna TPE adapter for constrained IOHexperimenter catalogue-index problems.
"""
from __future__ import annotations

from typing import Any, Sequence

from core.ioh_core.algorithms.common import (
    get_attached_eval_context,
    read_last_evaluation_constraint_slacks,
    read_last_evaluation_objective_y,
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)

_CONSTRAINTS_USER_ATTR = "constraints"


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


def optuna_dimension_label(index: int) -> str:
    """
    Return the stable Optuna parameter label for dimension ``index``.

    Parameters
    ----------
    index : int
        Zero-based variable index.

    Returns
    -------
    str
        Label of the form ``x_0000``, ``x_0001``, ...
    """
    return f"x_{int(index):04d}"


def candidate_from_optuna_trial(
    trial: Any,
    *,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> list[int]:
    """
    Suggest and validate an integer candidate from an Optuna trial.

    Non-degenerate dimensions use ``trial.suggest_int`` with inclusive IOH
    bounds (never ``upper + 1``). Degenerate bounds (``lower == upper``) are
    taken as integer constants without calling ``suggest_int``.

    Parameters
    ----------
    trial : Any
        Optuna trial exposing ``suggest_int``.
    lower_bounds : sequence of int
        Inclusive lower catalogue indices.
    upper_bounds : sequence of int
        Inclusive upper catalogue indices.

    Returns
    -------
    list of int
        Validated integer candidate vector in dimension index order.

    Raises
    ------
    ValueError
        If bound lengths disagree or a lower bound exceeds its upper bound.
    """
    dimension = len(lower_bounds)
    if len(upper_bounds) != dimension:
        raise ValueError(
            f"bound lengths disagree: lb={dimension} ub={len(upper_bounds)}"
        )
    raw: list[int] = []
    for index in range(dimension):
        lower = int(lower_bounds[index])
        upper = int(upper_bounds[index])
        if lower > upper:
            raise ValueError(
                f"bound[{index}] lower {lower} exceeds upper {upper}"
            )
        if lower == upper:
            raw.append(lower)
        else:
            raw.append(
                int(
                    trial.suggest_int(
                        optuna_dimension_label(index),
                        lower,
                        upper,
                    )
                )
            )
    return validate_integer_candidate(
        raw,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


class OptunaTPE:
    """
    Optuna TPE on integer catalogue indices with native inequality constraints.

    Searches on unpenalized ``y`` with ``TPESampler(constraints_func=...)``
    fed signed ``η − 1``. Authoritative evaluation counts and best solutions
    come from the IOH problem state (penalized ``y_p``). Optuna's best trial
    is ignored and must not be re-evaluated.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget (also Optuna ``n_trials``).
    seed : int
        Deterministic seed required by the registry seed policy.
    """

    name = "OPTUNA_TPE"
    info = (
        "Optuna TPE on integer catalogue indices; minimizes unpenalized y "
        "subject to signed g=η−1"
    )

    def __init__(self, max_evaluations: int, seed: int) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run Optuna TPE on a constrained IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds, callable evaluation, and attached
            EvalContext with at least one active constraint.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        OptunaTPE
            ``self`` after the run; best values live on the IOH problem state.

        Raises
        ------
        ValueError
            If the problem has no active constraints.
        """
        import optuna
        from optuna.samplers import TPESampler

        mm = _count_ioh_constraints(problem)
        if mm < 1:
            raise ValueError(
                "OPTUNA_TPE requires at least one active IOH "
                f"constraint; got mm={mm}"
            )

        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def constraints_func(trial: optuna.trial.FrozenTrial) -> Sequence[float]:
            stored = trial.user_attrs.get(_CONSTRAINTS_USER_ATTR)
            if stored is None:
                raise ValueError(
                    "OPTUNA_TPE constraints_func missing user attr "
                    f"{_CONSTRAINTS_USER_ATTR!r}; objective must store "
                    "signed slacks after each successful evaluation"
                )
            if not isinstance(stored, (list, tuple)):
                raise ValueError(
                    "OPTUNA_TPE constraints user attr must be a sequence, "
                    f"got {type(stored).__name__}"
                )
            if len(stored) != mm:
                raise ValueError(
                    f"expected {mm} constraint slacks, got {len(stored)}"
                )
            return tuple(float(value) for value in stored)

        sampler = TPESampler(
            seed=int(self.seed),
            constraints_func=constraints_func,
        )
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        def objective(trial: optuna.trial.Trial) -> float:
            candidate = candidate_from_optuna_trial(
                trial,
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
            trial.set_user_attr(_CONSTRAINTS_USER_ATTR, tuple(g))
            controller.observe_evaluation(
                y_p,
            )
            if controller.should_stop():
                study.stop()
            return float(y)

        study.optimize(
            objective,
            n_trials=int(self.max_evaluations),
            n_jobs=1,
            catch=(),
            show_progress_bar=False,
        )
        return self


_OPTUNA_PREFIX_LENGTH = 50


class OptunaTPEWarmStart:
    """
    Optuna TPE whose first trials are a ranked InitialDesign prefix.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Deterministic seed.
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    name = "OPTUNA_TPE_WARM_START"
    info = "Optuna TPE from a ranked InitialDesign prefix"

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
        """Run warm-start TPE. Queued trials count inside one optimize call."""
        import optuna
        from optuna.samplers import TPESampler

        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        mm = _count_ioh_constraints(problem)
        if mm < 1:
            raise ValueError(
                "OPTUNA_TPE_WARM_START requires at least one active IOH "
                f"constraint; got mm={mm}"
            )

        dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=_OPTUNA_PREFIX_LENGTH,
            max_evaluations=self.max_evaluations,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def constraints_func(trial: optuna.trial.FrozenTrial) -> Sequence[float]:
            stored = trial.user_attrs.get(_CONSTRAINTS_USER_ATTR)
            if stored is None:
                raise ValueError(
                    "OPTUNA_TPE_WARM_START constraints_func missing user attr "
                    f"{_CONSTRAINTS_USER_ATTR!r}; objective must store "
                    "signed slacks after each successful evaluation"
                )
            if not isinstance(stored, (list, tuple)):
                raise ValueError(
                    "OPTUNA_TPE_WARM_START constraints user attr must be a "
                    f"sequence, got {type(stored).__name__}"
                )
            if len(stored) != mm:
                raise ValueError(
                    f"expected {mm} constraint slacks, got {len(stored)}"
                )
            return tuple(float(value) for value in stored)

        sampler = TPESampler(
            seed=int(self.seed),
            constraints_func=constraints_func,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        for vector in prefix:
            study.enqueue_trial(
                {
                    optuna_dimension_label(index): int(vector[index])
                    for index in range(dimension)
                }
            )

        def objective(trial: optuna.trial.Trial) -> float:
            candidate = candidate_from_optuna_trial(
                trial,
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
            trial.set_user_attr(_CONSTRAINTS_USER_ATTR, tuple(g))
            controller.observe_evaluation(
                y_p,
            )
            if controller.should_stop():
                study.stop()
            return float(y)

        study.optimize(
            objective,
            n_trials=int(self.max_evaluations),
            n_jobs=1,
            catch=(),
            show_progress_bar=False,
        )
        return self
