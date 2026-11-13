"""
LEAP genetic algorithm wrapper for IOHexperimenter problems.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from leap_ec.algorithm import generational_ea
from leap_ec.int_rep.initializers import create_int_vector
from leap_ec.int_rep.ops import mutate_randint
from leap_ec.ops import (
    UniformCrossover,
    clone,
    evaluate,
    pool,
    tournament_selection,
)
from leap_ec.representation import Representation

from core.ioh_core.algorithms.common import resolve_run_controller
from core.ioh_core.algorithms.leap.problem import IOHLeapIntegerProblem


class LeapGeneticAlgorithm:
    """
    LEAP generational GA with integer operators over IOH catalogue indices.

    Authoritative evaluation counts and best solutions come from the IOH
    problem state. The population returned by ``generational_ea`` is ignored
    for experiment reporting.

    LEAP 0.8.2 integer operators use the global ``random`` and ``np.random``
    modules. Each ``__call__`` therefore seeds both from the task seed before
    creating the initial population.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    population_size : int, optional
        GA population size (>= 2). Default is ``50``.
    seed : int or None, optional
        Deterministic seed required by the registry seed policy.
    """

    name = "LEAP_GA"
    info = "LEAP generational genetic algorithm with integer operators"

    def __init__(
        self,
        max_evaluations: int,
        population_size: int = 50,
        seed: int | None = None,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.population_size = int(population_size)
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
        LeapGeneticAlgorithm
            ``self`` after the run; best values live on the IOH problem state.
        """
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        adapter = IOHLeapIntegerProblem(
            problem,
            self.max_evaluations,
            run_controller=controller,
        )
        random.seed(int(self.seed))
        np.random.seed(int(self.seed))

        bounds = adapter.bounds
        generational_ea(
            max_generations=float("inf"),
            pop_size=self.population_size,
            problem=adapter,
            representation=Representation(
                initialize=create_int_vector(bounds),
            ),
            pipeline=[
                tournament_selection,
                clone,
                mutate_randint(expected_num_mutations=1, bounds=bounds),
                UniformCrossover(p_swap=0.2),
                evaluate,
                pool(size=self.population_size),
            ],
            stop=lambda _population: controller.should_stop(),
            k_elites=1,
        )
        return self


class LeapGeneticAlgorithmWarmStart:
    """
    LEAP GA whose initial population is a ranked InitialDesign prefix.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    population_size : int, optional
        Prefix length and population size. Default is ``50``.
    seed : int or None, optional
        Deterministic seed required by the registry seed policy.
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    name = "LEAP_GA_WARM_START"
    info = "LEAP generational GA from a ranked InitialDesign prefix"

    def __init__(
        self,
        max_evaluations: int,
        population_size: int = 50,
        seed: int | None = None,
        *,
        initial_design_path: str,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.population_size = int(population_size)
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
        LeapGeneticAlgorithmWarmStart
            ``self`` after the run.
        """
        from core.ioh_core.algorithms.common import read_problem_bounds
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
        genomes = [np.array(vector, dtype=int) for vector in prefix]
        cursor = {"index": 0}

        def initialize() -> np.ndarray:
            genome = genomes[cursor["index"]]
            cursor["index"] += 1
            return genome

        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        adapter = IOHLeapIntegerProblem(
            problem,
            self.max_evaluations,
            run_controller=controller,
        )
        random.seed(int(self.seed))
        np.random.seed(int(self.seed))
        bounds = adapter.bounds
        generational_ea(
            max_generations=float("inf"),
            pop_size=self.population_size,
            problem=adapter,
            representation=Representation(initialize=initialize),
            pipeline=[
                tournament_selection,
                clone,
                mutate_randint(expected_num_mutations=1, bounds=bounds),
                UniformCrossover(p_swap=0.2),
                evaluate,
                pool(size=self.population_size),
            ],
            stop=lambda _population: controller.should_stop(),
            k_elites=1,
        )
        return self
