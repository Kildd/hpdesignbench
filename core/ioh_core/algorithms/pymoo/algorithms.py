"""
Pymoo genetic algorithm wrapper for IOHexperimenter problems.
"""
from __future__ import annotations

from typing import Any

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.collection import TerminationCollection
from pymoo.termination.max_eval import MaximumFunctionCallTermination

from core.ioh_core.algorithms.common import read_problem_bounds, resolve_run_controller
from core.ioh_core.algorithms.pymoo.problem import (
    IOHIntegerConstrainedProblem,
    IOHIntegerProblem,
)
from core.ioh_core.algorithms.pymoo.termination import ControllerAwareTermination


class PymooGeneticAlgorithm:
    """
    Pymoo GA with integer sampling/operators over IOH catalogue indices.

    Authoritative evaluation counts and best solutions come from the IOH
    problem state. Pymoo's ``result.X`` / ``result.F`` / ``n_eval`` are ignored
    for experiment reporting.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    population_size : int, optional
        GA population size (>= 2). Default is ``50``.
    eliminate_duplicates : bool, optional
        Forwarded to Pymoo ``GA``. Default is ``True``.
    seed : int or None, optional
        Deterministic seed required by the registry seed policy.
    """

    name = "PYMOO_GA"
    info = "Pymoo genetic algorithm with integer operators"

    def __init__(
        self,
        max_evaluations: int,
        population_size: int = 50,
        eliminate_duplicates: bool = True,
        seed: int | None = None,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.population_size = int(population_size)
        self.eliminate_duplicates = bool(eliminate_duplicates)
        self.seed = seed

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run the GA on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        PymooGeneticAlgorithm
            ``self`` after the run; best values live on the IOH problem state.
        """
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        stop_flag = {"stop": False}
        adapter = IOHIntegerProblem(
            problem,
            self.max_evaluations,
            run_controller=controller,
            stop_flag=stop_flag,
        )
        algorithm = GA(
            pop_size=self.population_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(
                prob=1.0,
                eta=3.0,
                vtype=float,
                repair=RoundingRepair(),
            ),
            mutation=PM(
                eta=3.0,
                vtype=float,
                repair=RoundingRepair(),
            ),
            eliminate_duplicates=self.eliminate_duplicates,
        )
        termination = TerminationCollection(
            MaximumFunctionCallTermination(self.max_evaluations),
            ControllerAwareTermination(stop_flag),
        )
        minimize(
            adapter,
            algorithm,
            termination=termination,
            seed=self.seed,
            verbose=False,
        )
        return self


class PymooGeneticAlgorithmWarmStart:
    """
    Pymoo GA whose initial population is a ranked InitialDesign prefix.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    population_size : int, optional
        Prefix length and GA population size. Default is ``50``.
    eliminate_duplicates : bool, optional
        Forwarded to Pymoo ``GA``. Default is ``True``.
    seed : int or None, optional
        Deterministic seed required by the registry seed policy.
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    name = "PYMOO_GA_WARM_START"
    info = (
        "Pymoo genetic algorithm from a ranked InitialDesign prefix "
        "(initial population)"
    )

    def __init__(
        self,
        max_evaluations: int,
        population_size: int = 50,
        eliminate_duplicates: bool = True,
        seed: int | None = None,
        *,
        initial_design_path: str,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.population_size = int(population_size)
        self.eliminate_duplicates = bool(eliminate_duplicates)
        self.seed = seed
        self.initial_design_path = initial_design_path

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run the warm-start GA on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        PymooGeneticAlgorithmWarmStart
            ``self`` after the run.
        """
        import numpy as np

        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=self.population_size,
            max_evaluations=self.max_evaluations,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        stop_flag = {"stop": False}
        adapter = IOHIntegerProblem(
            problem,
            self.max_evaluations,
            run_controller=controller,
            stop_flag=stop_flag,
        )
        algorithm = GA(
            pop_size=self.population_size,
            sampling=np.asarray(prefix, dtype=int),
            crossover=SBX(
                prob=1.0,
                eta=3.0,
                vtype=float,
                repair=RoundingRepair(),
            ),
            mutation=PM(
                eta=3.0,
                vtype=float,
                repair=RoundingRepair(),
            ),
            eliminate_duplicates=self.eliminate_duplicates,
        )
        termination = TerminationCollection(
            MaximumFunctionCallTermination(self.max_evaluations),
            ControllerAwareTermination(stop_flag),
        )
        minimize(
            adapter,
            algorithm,
            termination=termination,
            seed=self.seed,
            verbose=False,
        )
        return self


class PymooConstrainedGeneticAlgorithm:
    """
    Pymoo GA minimizing unpenalized ``y`` subject to native inequality constraints.

    Authoritative evaluation counts and best solutions come from the IOH
    problem state (penalized ``y_p``). Pymoo's ``result.X`` / ``result.F`` /
    ``n_eval`` are ignored for experiment reporting.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    population_size : int, optional
        GA population size (>= 2). Default is ``50``.
    eliminate_duplicates : bool, optional
        Forwarded to Pymoo ``GA``. Default is ``True``.
    seed : int or None, optional
        Deterministic seed required by the registry seed policy.
    warm_start : bool, optional
        When ``True``, the initial population is a ranked InitialDesign
        prefix of length ``population_size``.
    initial_design_path : str or None, optional
        Required when ``warm_start`` is true.
    name : str or None, optional
        Registry token stored on the instance.
    """

    name = "PYMOO_GA_CONSTRAINED"
    info = (
        "Pymoo genetic algorithm with integer operators; "
        "minimizes unpenalized y subject to signed g=η−1"
    )

    def __init__(
        self,
        max_evaluations: int,
        population_size: int = 50,
        eliminate_duplicates: bool = True,
        seed: int | None = None,
        *,
        warm_start: bool = False,
        initial_design_path: str | None = None,
        name: str | None = None,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.population_size = int(population_size)
        self.eliminate_duplicates = bool(eliminate_duplicates)
        self.seed = seed
        self._warm_start = bool(warm_start)
        self.initial_design_path = initial_design_path
        if name is not None:
            self.name = name
        if self._warm_start:
            self.info = (
                "Pymoo constrained GA from a ranked InitialDesign prefix "
                "(initial population); minimizes unpenalized y subject to "
                "signed g=η−1"
            )

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run the constrained GA on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds, callable evaluation, and attached
            EvalContext with at least one active constraint.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        PymooConstrainedGeneticAlgorithm
            ``self`` after the run; best values live on the IOH problem state.
        """
        if self._warm_start:
            import numpy as np

            from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

            if not self.initial_design_path:
                raise ValueError(
                    f"{self.name} requires initial_design_path"
                )
            _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
            prefix = load_warm_start_prefix(
                problem,
                algorithm_name=self.name,
                initial_design_path=self.initial_design_path,
                count=self.population_size,
                max_evaluations=self.max_evaluations,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            sampling = np.asarray(prefix, dtype=int)
        else:
            sampling = IntegerRandomSampling()
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        stop_flag = {"stop": False}
        adapter = IOHIntegerConstrainedProblem(
            problem,
            self.max_evaluations,
            run_controller=controller,
            stop_flag=stop_flag,
        )
        algorithm = GA(
            pop_size=self.population_size,
            sampling=sampling,
            crossover=SBX(
                prob=1.0,
                eta=3.0,
                vtype=float,
                repair=RoundingRepair(),
            ),
            mutation=PM(
                eta=3.0,
                vtype=float,
                repair=RoundingRepair(),
            ),
            eliminate_duplicates=self.eliminate_duplicates,
        )
        termination = TerminationCollection(
            MaximumFunctionCallTermination(self.max_evaluations),
            ControllerAwareTermination(stop_flag),
        )
        minimize(
            adapter,
            algorithm,
            termination=termination,
            seed=self.seed,
            verbose=False,
        )
        return self
