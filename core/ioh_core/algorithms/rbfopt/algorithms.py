"""
RBFOpt surrogate search wrapper for IOHexperimenter problems.
"""
from __future__ import annotations

from typing import Any

from core.ioh_core.algorithms.common import resolve_run_controller

# Discarded proposals and reused pre-restart values increment RBFOpt's
# iteration and cycle counters without a catalogue call. A headroom of
# ten keeps max_evaluations as the binding budget at the waste rate seen
# on the reference slabs, and it still stops a discard storm. The library
# defaults (1,000 iterations and 1,000 cycles) stop a 15,000-evaluation
# run after roughly 1,000 catalogue calls.
_NON_EVALUATION_HEADROOM = 10


def _settings_kwargs(
    *,
    max_evaluations: int,
    ga_base_population_size: int,
    minlp_solver_path: str,
    nlp_solver_path: str,
    seed: int,
) -> dict[str, Any]:
    """RBFOpt settings whose evaluation counter is the binding budget."""
    budget = int(max_evaluations)
    ceiling = _NON_EVALUATION_HEADROOM * budget
    return {
        "max_evaluations": budget,
        "max_iterations": ceiling,
        "max_cycles": ceiling,
        "max_noisy_evaluations": 0,
        "ga_base_population_size": int(ga_base_population_size),
        "minlp_solver_path": str(minlp_solver_path),
        "nlp_solver_path": str(nlp_solver_path),
        "rand_seed": int(seed),
        "num_cpus": 1,
    }


class RBFOptSearch:
    """
    RBFOpt search on native integer catalogue-index variables.

    Authoritative evaluation counts and best solutions come from the IOH
    problem state. The ``optimize()`` return tuple is ignored and must not be
    re-evaluated.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget (also RBFOpt ``max_evaluations``).
        ``max_iterations`` and ``max_cycles`` are set to ten times this
        budget so neither library default stops the run earlier.
    ga_base_population_size : int
        RBFOpt genetic-algorithm base population size (>= 2).
    minlp_solver_path : str
        Resolved Bonmin (or compatible MINLP) executable path.
    nlp_solver_path : str
        Resolved Ipopt (or compatible NLP) executable path.
    seed : int
        Deterministic seed forwarded as ``rand_seed``.
    """

    name = "RBFOPT"
    info = "RBFOpt surrogate search on native integer variables"

    def __init__(
        self,
        *,
        max_evaluations: int,
        ga_base_population_size: int,
        minlp_solver_path: str,
        nlp_solver_path: str,
        seed: int,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.ga_base_population_size = int(ga_base_population_size)
        self.minlp_solver_path = str(minlp_solver_path)
        self.nlp_solver_path = str(nlp_solver_path)
        self.seed = int(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run RBFOpt on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        RBFOptSearch
            ``self`` after the run; best values live on the IOH problem state.
        """
        from rbfopt import RbfoptAlgorithm, RbfoptSettings

        from core.ioh_core.algorithms.rbfopt.problem import IOHRbfoptBlackBox

        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        black_box = IOHRbfoptBlackBox(
            problem,
            self.max_evaluations,
            run_controller=controller,
        )
        settings_kwargs: dict[str, Any] = _settings_kwargs(
            max_evaluations=self.max_evaluations,
            ga_base_population_size=self.ga_base_population_size,
            minlp_solver_path=self.minlp_solver_path,
            nlp_solver_path=self.nlp_solver_path,
            seed=self.seed,
        )
        # Public stop uses gap <= eps_opt with
        # gap = (fmin - target_objval) / max(|target_objval|, eps_zero).
        # eps_opt=0 yields absolute fmin <= target_objval, matching our
        # threshold.
        if (
            controller.mode == "fixed_target"
            and controller.target_threshold is not None
        ):
            settings_kwargs["target_objval"] = float(controller.target_threshold)
            settings_kwargs["eps_opt"] = 0.0
        settings = RbfoptSettings(**settings_kwargs)
        algorithm = RbfoptAlgorithm(settings, black_box)
        algorithm.optimize()
        return self


class RBFOptSearchWarmStart:
    """
    RBFOpt whose initial nodes are a ranked InitialDesign prefix.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget. ``max_iterations`` and ``max_cycles``
        are set to ten times this budget, as in :class:`RBFOptSearch`.
    ga_base_population_size : int
        RBFOpt genetic-algorithm base population size.
    minlp_solver_path : str
        Resolved MINLP executable path.
    nlp_solver_path : str
        Resolved NLP executable path.
    seed : int
        Deterministic seed forwarded as ``rand_seed``.
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    name = "RBFOPT_WARM_START"
    info = "RBFOpt from a ranked InitialDesign prefix"

    def __init__(
        self,
        *,
        max_evaluations: int,
        ga_base_population_size: int,
        minlp_solver_path: str,
        nlp_solver_path: str,
        seed: int,
        initial_design_path: str,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.ga_base_population_size = int(ga_base_population_size)
        self.minlp_solver_path = str(minlp_solver_path)
        self.nlp_solver_path = str(nlp_solver_path)
        self.seed = int(seed)
        self.initial_design_path = initial_design_path

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """Run warm-start RBFOpt. The library evaluates the prefix."""
        import numpy as np
        from rbfopt import RbfoptAlgorithm, RbfoptSettings
        from rbfopt.rbfopt_utils import get_num_init_samples

        from core.ioh_core.algorithms.common import read_problem_bounds
        from core.ioh_core.algorithms.rbfopt.problem import IOHRbfoptBlackBox
        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        settings_kwargs: dict[str, Any] = _settings_kwargs(
            max_evaluations=self.max_evaluations,
            ga_base_population_size=self.ga_base_population_size,
            minlp_solver_path=self.minlp_solver_path,
            nlp_solver_path=self.nlp_solver_path,
            seed=self.seed,
        )
        settings = RbfoptSettings(**settings_kwargs).set_auto_parameters(
            dimension,
            np.asarray(lower_bounds, dtype=float),
            np.asarray(upper_bounds, dtype=float),
            np.asarray(list(range(dimension)), dtype=int),
        )
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=int(get_num_init_samples(settings, dimension)),
            max_evaluations=self.max_evaluations,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        if (
            controller.mode == "fixed_target"
            and controller.target_threshold is not None
        ):
            settings.target_objval = float(controller.target_threshold)
            settings.eps_opt = 0.0
        black_box = IOHRbfoptBlackBox(
            problem,
            self.max_evaluations,
            run_controller=controller,
        )
        algorithm = RbfoptAlgorithm(
            settings,
            black_box,
            init_node_pos=np.asarray(prefix, dtype=float),
            init_node_val=None,
            do_init_strategy=False,
        )
        algorithm.optimize()
        return self
