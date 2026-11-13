"""
Hyperopt search-space helpers and TPE adapter for IOHexperimenter problems.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.ioh_core.algorithms.common import (
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)


def hyperopt_dimension_label(index: int) -> str:
    """
    Return the stable Hyperopt label for dimension ``index``.

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


def build_quantized_integer_space(
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> dict[str, Any]:
    """
    Build a Hyperopt space over inclusive IOH catalogue-index bounds.

    Each non-degenerate dimension uses ``hp.quniform(..., q=1)`` with the
    inclusive upper bound as ``high`` (never ``upper + 1``). Degenerate
    bounds (``lower == upper``) are encoded as integer constants.

    Parameters
    ----------
    lower_bounds : sequence of int
        Inclusive lower catalogue indices.
    upper_bounds : sequence of int
        Inclusive upper catalogue indices.

    Returns
    -------
    dict
        Mapping from stable ``x_XXXX`` labels to Hyperopt nodes or ints.

    Raises
    ------
    ValueError
        If bound lengths disagree or a lower bound exceeds its upper bound.
    """
    from hyperopt import hp

    dimension = len(lower_bounds)
    if len(upper_bounds) != dimension:
        raise ValueError(
            f"bound lengths disagree: lb={dimension} ub={len(upper_bounds)}"
        )
    space: dict[str, Any] = {}
    for index in range(dimension):
        lower = int(lower_bounds[index])
        upper = int(upper_bounds[index])
        if lower > upper:
            raise ValueError(
                f"bound[{index}] lower {lower} exceeds upper {upper}"
            )
        label = hyperopt_dimension_label(index)
        if lower == upper:
            space[label] = lower
        else:
            space[label] = hp.quniform(label, lower, upper, 1)
    return space


def candidate_from_hyperopt_sample(
    sample: Mapping[str, Any],
    *,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> list[int]:
    """
    Reconstruct a validated integer candidate from a Hyperopt sample mapping.

    Coordinates are read by stable dimension labels in index order; mapping
    iteration order is never used.

    Parameters
    ----------
    sample : mapping
        Hyperopt sample with ``x_XXXX`` keys.
    lower_bounds : sequence of int
        Inclusive lower catalogue indices.
    upper_bounds : sequence of int
        Inclusive upper catalogue indices.

    Returns
    -------
    list of int
        Validated integer candidate vector.
    """
    dimension = len(lower_bounds)
    raw = [sample[hyperopt_dimension_label(index)] for index in range(dimension)]
    return validate_integer_candidate(
        raw,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


class HyperoptTPE:
    """
    Hyperopt TPE over an inclusive integer catalogue-index grid.

    Authoritative evaluation counts and best solutions come from the IOH
    problem state. Hyperopt's returned best mapping is ignored and must not be
    re-evaluated.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget (also Hyperopt ``max_evals``).
    seed : int or None, optional
        Deterministic seed required by the registry seed policy.
    """

    name = "HYPEROPT_TPE"
    info = "Hyperopt tree-structured Parzen estimator on an integer grid"

    def __init__(self, max_evaluations: int, seed: int | None = None) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = seed

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run TPE on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        HyperoptTPE
            ``self`` after the run; best values live on the IOH problem state.
        """
        import numpy as np
        from hyperopt import STATUS_OK, Trials, fmin, tpe

        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        space = build_quantized_integer_space(lower_bounds, upper_bounds)
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )

        def objective(sample: Mapping[str, Any]) -> dict[str, Any]:
            candidate = candidate_from_hyperopt_sample(
                sample,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            controller.require_evaluation()
            y = float(problem(candidate))
            controller.observe_evaluation(
                y,
            )
            return {"loss": y, "status": STATUS_OK}

        def early_stop_fn(trials: Any, *args: Any) -> tuple[bool, list[Any]]:
            del trials
            return bool(controller.should_stop()), list(args)

        trials = Trials()
        fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evaluations,
            trials=trials,
            rstate=np.random.default_rng(self.seed),
            show_progressbar=False,
            verbose=False,
            early_stop_fn=early_stop_fn,
        )
        return self


_HYPEROPT_PREFIX_LENGTH = 50


class HyperoptTPEWarmStart:
    """
    Hyperopt TPE whose first trials are a ranked InitialDesign prefix.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int or None, optional
        Deterministic seed.
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    name = "HYPEROPT_TPE_WARM_START"
    info = "Hyperopt TPE from a ranked InitialDesign prefix"

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        *,
        initial_design_path: str,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = seed
        self.initial_design_path = initial_design_path

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """Run warm-start TPE. Injected trials are evaluated inside fmin."""
        import numpy as np
        from hyperopt import STATUS_OK, fmin, tpe

        try:
            from hyperopt.fmin import generate_trials_to_calculate
        except ImportError as exc:
            import hyperopt

            raise ImportError(
                "hyperopt.fmin.generate_trials_to_calculate is missing in "
                f"hyperopt {getattr(hyperopt, '__version__', 'unknown')}"
            ) from exc

        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=_HYPEROPT_PREFIX_LENGTH,
            max_evaluations=self.max_evaluations,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        space = build_quantized_integer_space(lower_bounds, upper_bounds)
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )

        def objective(sample: Mapping[str, Any]) -> dict[str, Any]:
            candidate = candidate_from_hyperopt_sample(
                sample,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            controller.require_evaluation()
            y = float(problem(candidate))
            controller.observe_evaluation(
                y,
            )
            return {"loss": y, "status": STATUS_OK}

        def early_stop_fn(trials: Any, *args: Any) -> tuple[bool, list[Any]]:
            del trials
            return bool(controller.should_stop()), list(args)

        points = [
            {
                hyperopt_dimension_label(index): int(vector[index])
                for index in range(dimension)
            }
            for vector in prefix
        ]
        trials = generate_trials_to_calculate(points)
        fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evaluations,
            trials=trials,
            rstate=np.random.default_rng(self.seed),
            show_progressbar=False,
            verbose=False,
            early_stop_fn=early_stop_fn,
        )
        return self
