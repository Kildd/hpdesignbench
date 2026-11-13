"""
Native Latin hypercube sampling over IOH integer catalogue-index bounds.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from core.ioh_core.algorithms.common import (
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)


class LatinHypercubeSampling:
    """
    Latin hypercube sampling baseline in inclusive integer index space.

    For each dimension, draws one jittered stratum value per sample, permutes
    those strata, and maps ``[0, 1)`` values onto inclusive catalogue indices.
    Generates exactly ``max_evaluations`` candidates in fixed-budget mode.
    Fixed-target mode may stop after a design prefix.

    Parameters
    ----------
    max_evaluations : int
        Hard upper bound on IOH evaluations / LHS samples.
    seed : int or None, optional
        Deterministic seed required by the registry seed policy.
    """

    name = "LATIN_HYPERCUBE"
    info = "Latin hypercube sampling baseline in integer index space"

    def __init__(self, max_evaluations: int, seed: int | None = None) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Evaluate an Latin-hypercube design on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        LatinHypercubeSampling
            ``self`` after the run; best values live on the IOH problem state.
        """
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        sample_count = self.max_evaluations
        design = self._integer_design(
            sample_count=sample_count,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        for candidate in design:
            if not controller.can_evaluate():
                break
            validated = validate_integer_candidate(
                candidate,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            controller.require_evaluation()
            y = problem(validated)
            controller.observe_evaluation(
                y,
            )
            if controller.should_stop():
                break
        return self

    def _integer_design(
        self,
        *,
        sample_count: int,
        lower_bounds: list[int],
        upper_bounds: list[int],
    ) -> list[list[int]]:
        """
        Build an integer LHS design with one row per sample.

        Parameters
        ----------
        sample_count : int
            Number of samples / strata.
        lower_bounds : list of int
            Inclusive lower catalogue indices.
        upper_bounds : list of int
            Inclusive upper catalogue indices.

        Returns
        -------
        list of list of int
            Candidate matrix with shape ``(sample_count, dimension)``.
        """
        dimension = len(lower_bounds)
        columns: list[np.ndarray] = []
        for index in range(dimension):
            lower = int(lower_bounds[index])
            upper = int(upper_bounds[index])
            if lower == upper:
                columns.append(np.full(sample_count, lower, dtype=int))
                continue
            level_count = upper - lower + 1
            jitters = self.rng.random(sample_count)
            strata = (np.arange(sample_count, dtype=float) + jitters) / float(
                sample_count
            )
            self.rng.shuffle(strata)
            mapped = lower + np.floor(strata * float(level_count)).astype(int)
            mapped = np.clip(mapped, lower, upper)
            columns.append(mapped)

        design = np.column_stack(columns) if columns else np.empty((sample_count, 0))
        return [[int(value) for value in row] for row in design]
