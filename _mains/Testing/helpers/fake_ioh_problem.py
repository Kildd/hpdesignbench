"""
Shared fake IOH problem and algorithm conformance checks for adapter tests.
"""
from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from core.ioh_core.algorithm_factory import AlgorithmFactoryError
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import SeedPolicy, get_algorithm_spec


@dataclass
class FakeBest:
    """Minimal IOH current-best record."""

    x: list[int] = field(default_factory=list)
    y: float = math.inf


@dataclass
class FakeState:
    """Minimal IOH problem state."""

    evaluations: int = 0
    current_best: FakeBest = field(default_factory=FakeBest)


class FakeIOHProblem:
    """
    Tiny IOH-like problem for algorithm adapter and conformance tests.

    Records every candidate, updates evaluations/current best, and exposes the
    attributes used by shared helpers.
    """

    def __init__(self, lower: list[int], upper: list[int]) -> None:
        if len(lower) != len(upper):
            raise ValueError("bound lengths must match")
        self.meta_data = SimpleNamespace(n_variables=len(lower))
        self.bounds = SimpleNamespace(lb=list(lower), ub=list(upper))
        self.state = FakeState()
        self.calls: list[list[int]] = []
        self.hdb_last_feasible = True
        self.next_feasible = True

    def __call__(self, x: Any) -> float:
        candidate = [int(v) for v in x]
        self.calls.append(candidate)
        self.state.evaluations += 1
        self.hdb_last_feasible = bool(self.next_feasible)
        # Sphere-like objective on integer indices.
        value = float(sum(v * v for v in candidate))
        if value < self.state.current_best.y:
            self.state.current_best = FakeBest(x=list(candidate), y=value)
        return value


def _declared_budget(algorithm: IOHAlgorithm) -> int:
    """
    Return the declared evaluation budget from a constructed algorithm.

    Parameters
    ----------
    algorithm : IOHAlgorithm
        Constructed algorithm instance.

    Returns
    -------
    int
        Positive evaluation budget.
    """
    for attr in ("evaluations", "max_evaluations"):
        value = getattr(algorithm, attr, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
    raise AssertionError(
        "algorithm {!r} does not expose evaluations/max_evaluations".format(
            getattr(algorithm, "name", type(algorithm).__name__)
        )
    )


def _assert_candidates_valid(
    testcase: unittest.TestCase,
    problem: FakeIOHProblem,
    *,
    dimension: int,
    lower: list[int],
    upper: list[int],
) -> None:
    """
    Assert every recorded candidate is an in-bounds integer vector.

    Parameters
    ----------
    testcase : unittest.TestCase
        Active test case.
    problem : FakeIOHProblem
        Problem that recorded candidates.
    dimension : int
        Expected candidate length.
    lower : list of int
        Inclusive lower bounds.
    upper : list of int
        Inclusive upper bounds.
    """
    testcase.assertGreaterEqual(len(problem.calls), 1)
    for candidate in problem.calls:
        testcase.assertEqual(len(candidate), dimension)
        testcase.assertTrue(all(isinstance(value, int) for value in candidate))
        for value, lo, hi in zip(candidate, lower, upper):
            testcase.assertGreaterEqual(value, lo)
            testcase.assertLessEqual(value, hi)


def assert_algorithm_conformance(
    testcase: unittest.TestCase,
    *,
    algorithm_name: str,
    build_algorithm: Callable[[int | None], IOHAlgorithm],
    seed: int | None,
    second_seed: int | None,
    make_problem: Callable[[list[int], list[int]], Any] | None = None,
    lower: list[int] | None = None,
    upper: list[int] | None = None,
    require_seed_divergence: bool = True,
) -> None:
    """
    Assert shared adapter contracts for one registered algorithm.

    Parameters
    ----------
    testcase : unittest.TestCase
        Active test case used for assertions.
    algorithm_name : str
        Registry token expected on the constructed instance.
    build_algorithm : callable
        ``(seed) -> IOHAlgorithm`` factory for fresh instances.
    seed : int or None
        Primary seed for required-seed algorithms; ``None`` when forbidden.
    second_seed : int or None
        Distinct seed for stochastic divergence checks; ``None`` when forbidden.
    make_problem : callable or None, optional
        Optional ``(lower, upper) -> problem`` factory. Defaults to
        :class:`FakeIOHProblem`. Constrained adapters may pass a fake that
        attaches an EvalContext with active constraints.
    lower, upper : list of int or None, optional
        Inclusive catalogue-index bounds for the fake problem. Defaults to
        ``[0, 1]`` / ``[3, 4]``.
    require_seed_divergence : bool, optional
        When ``False``, skip the different-seed sequence check (sealed
        starts can make short integer MADS trajectories seed-invariant).
    """
    spec = get_algorithm_spec(algorithm_name)
    if lower is None:
        lower = [0, 1]
    if upper is None:
        upper = [3, 4]
    dimension = len(lower)
    problem_factory = make_problem or FakeIOHProblem

    first = build_algorithm(seed)
    second = build_algorithm(seed)
    testcase.assertIsInstance(first, IOHAlgorithm)
    testcase.assertIsNot(first, second)
    testcase.assertEqual(first.name, algorithm_name)
    testcase.assertTrue(str(first.info).strip())

    budget = _declared_budget(first)
    problem = problem_factory(lower, upper)
    first(problem)
    _assert_candidates_valid(
        testcase,
        problem,
        dimension=dimension,
        lower=lower,
        upper=upper,
    )
    testcase.assertLessEqual(len(problem.calls), budget)
    testcase.assertEqual(problem.state.evaluations, len(problem.calls))
    testcase.assertIn(problem.state.current_best.x, problem.calls)
    optimizer_evals = getattr(first, "n_optimizer_evaluations_", None)
    if optimizer_evals is not None:
        testcase.assertEqual(optimizer_evals, len(problem.calls))

    if spec.seed_policy is SeedPolicy.REQUIRED:
        testcase.assertIsNotNone(seed)
        testcase.assertIsNotNone(second_seed)
        testcase.assertNotEqual(seed, second_seed)

        problem_a = problem_factory(lower, upper)
        problem_b = problem_factory(lower, upper)
        build_algorithm(seed)(problem_a)
        build_algorithm(seed)(problem_b)
        testcase.assertEqual(problem_a.calls, problem_b.calls)

        problem_c = problem_factory(lower, upper)
        build_algorithm(second_seed)(problem_c)
        if require_seed_divergence and budget >= 4:
            testcase.assertNotEqual(problem_a.calls, problem_c.calls)

        with testcase.assertRaises(AlgorithmFactoryError):
            build_algorithm(None)
    else:
        testcase.assertIsNone(seed)
        testcase.assertIsNone(second_seed)
        with testcase.assertRaises(AlgorithmFactoryError):
            build_algorithm(0)
