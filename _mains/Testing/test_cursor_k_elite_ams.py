"""
k-elite adaptive multi-start.

Covers rank-weighted parent selection, coordinate inheritance, valid
ordinal mutation, archive bookkeeping, and the cold-start IOH loop.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import numpy as np

from core.ioh_core.algorithm_factory import AlgorithmFactoryError, create_algorithm
from core.ioh_core.algorithms.common import RunController
from core.ioh_core.algorithms.native.discrete_search import (
    DiscreteSearchSpace,
    EvaluatedPoint,
    EvaluationManager,
    LocalSearchResult,
)
from core.ioh_core.algorithms.native.factory import (
    build_k_elite_ams,
    build_k_elite_ams_warm_start,
)
from core.ioh_core.algorithms.native.k_elite_ams import (
    ArchiveBook,
    BudgetExhausted,
    EliteMember,
    _CONSECUTIVE_DUPLICATE_START_CAP,
    inherit_parent_coordinates,
    initial_local_searches,
    mutate_ordinal,
    run_k_elite_loop,
    select_rank_weighted,
    take_unseen_start,
    update_elite_archive,
)
from core.ioh_core.algorithms.registry import (
    BudgetSemantics,
    CandidateMode,
    SeedPolicy,
    get_algorithm_spec,
)
from _mains.Testing.helpers.fake_ioh_problem import (
    FakeBest,
    FakeIOHProblem,
    assert_algorithm_conformance,
)


class ScriptedRng:
    """Deterministic stand-in for the few Generator methods AMS uses."""

    def __init__(
        self,
        *,
        integers: list[Any] | None = None,
        randoms: list[float] | None = None,
        choices: list[Any] | None = None,
    ) -> None:
        self.integers_queue = list(integers or [])
        self.randoms = list(randoms or [])
        self.choices = list(choices or [])
        self.random_calls = 0

    def integers(self, low: Any, high: Any = None, size: Any = None) -> Any:
        """Return the next scripted draw."""
        del low, high, size
        value = self.integers_queue.pop(0)
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        return value

    def random(self) -> float:
        """Return the next scripted Bernoulli draw."""
        self.random_calls += 1
        return float(self.randoms.pop(0))

    def choice(self, a: Any, p: Any = None) -> Any:
        """Return the next scripted choice."""
        del a, p
        return self.choices.pop(0)


class RecordingRng(ScriptedRng):
    """Records the probability vector passed to ``choice``."""

    def __init__(self) -> None:
        super().__init__()
        self.probabilities: list[np.ndarray] = []

    def choice(self, a: Any, p: Any = None) -> Any:
        """Keep ``p`` and return the first strictly positive index."""
        del a
        probs = np.asarray(p, dtype=np.float64)
        self.probabilities.append(probs.copy())
        return int(np.flatnonzero(probs > 0.0)[0])


class FlatProblem(FakeIOHProblem):
    """Every index is a certified local minimum."""

    def __call__(self, x: object) -> float:
        candidate = [int(value) for value in list(x)]
        self.calls.append(candidate)
        self.state.evaluations += 1
        self.hdb_last_feasible = True
        if self.state.current_best.y > 0.0:
            self.state.current_best = FakeBest(x=list(candidate), y=0.0)
        return 0.0


def _member(minimum_id: int, x: tuple[int, ...], value: float) -> EliteMember:
    """Build one elite record."""
    return EliteMember(minimum_id, EvaluatedPoint(x, value))


def _certified(x: tuple[int, ...], value: float) -> LocalSearchResult:
    """Build a certified local-search result."""
    return LocalSearchResult(
        point=EvaluatedPoint(x, value),
        certified_local_optimum=True,
        n_steps=0,
        n_evaluations=0,
    )


def _uncertified(x: tuple[int, ...], value: float) -> LocalSearchResult:
    """Build an interrupted local-search result."""
    return LocalSearchResult(
        point=EvaluatedPoint(x, value),
        certified_local_optimum=False,
        n_steps=0,
        n_evaluations=1,
    )


class TestKEliteOperators(unittest.TestCase):
    """Pure operator tests with an injected generator."""

    def test_inheritance_follows_scripted_parent_indices(self) -> None:
        parents = [
            _member(0, (4,), 0.0),
            _member(1, (4,), 1.0),
            _member(2, (4,), 2.0),
            _member(3, (5,), 3.0),
            _member(4, (4,), 4.0),
        ]
        first = inherit_parent_coordinates(parents, ScriptedRng(integers=[0]))
        second = inherit_parent_coordinates(parents, ScriptedRng(integers=[3]))
        self.assertEqual(first, (4,))
        self.assertEqual(second, (5,))

    def test_rank_weights_follow_objective_then_index(self) -> None:
        members = [
            _member(0, (1,), 1.0),
            _member(1, (0,), 1.0),
            _member(2, (2,), 3.0),
        ]
        rng = RecordingRng()
        selected = select_rank_weighted(members, 1, rng)
        self.assertEqual(selected[0].point.x, (0,))
        expected = np.array([1.0, 0.5, 1.0 / 3.0])
        expected /= expected.sum()
        np.testing.assert_allclose(rng.probabilities[0], expected)

    def test_selection_uses_the_requested_parent_count(self) -> None:
        members = [
            _member(0, (0,), 0.0),
            _member(1, (1,), 1.0),
            _member(2, (2,), 2.0),
        ]
        selected = select_rank_weighted(members, 2, RecordingRng())
        self.assertEqual(len(selected), 2)
        with self.assertRaises(ValueError):
            select_rank_weighted(members, 4, RecordingRng())

    def test_exact_archive_is_the_whole_parent_pool(self) -> None:
        members = [_member(index, (index,), float(index)) for index in range(5)]
        selected = select_rank_weighted(members, 5, RecordingRng())
        self.assertEqual(
            {member.minimum_id for member in selected},
            {0, 1, 2, 3, 4},
        )

    def test_mutation_uses_only_legal_steps(self) -> None:
        space = DiscreteSearchSpace([0, 0, 0], [0, 4, 5])
        mutated = mutate_ordinal(
            (0, 2, 5),
            space,
            ScriptedRng(randoms=[0.0, 0.0], choices=[1]),
            0.05,
        )
        self.assertEqual(mutated, (0, 3, 4))

        lower = mutate_ordinal(
            (0, 2),
            DiscreteSearchSpace([0, 0], [3, 3]),
            ScriptedRng(randoms=[0.0, 1.0]),
            0.05,
        )
        self.assertEqual(lower, (1, 2))

        fixed = mutate_ordinal(
            (3,),
            DiscreteSearchSpace([3], [3]),
            ScriptedRng(),
            1.0,
        )
        self.assertEqual(fixed, (3,))

    def test_fixed_axis_consumes_no_bernoulli_draw(self) -> None:
        rng = ScriptedRng()
        mutate_ordinal((4,), DiscreteSearchSpace([4], [4]), rng, 1.0)
        self.assertEqual(rng.random_calls, 0)

    def test_archive_admits_rejects_and_keeps_ids(self) -> None:
        book = ArchiveBook()
        update_elite_archive(book, _uncertified((9,), 0.0), 2)
        self.assertEqual(book.n_admissions, 0)
        self.assertEqual(book.n_duplicates, 0)
        self.assertEqual(book.n_rejections, 0)
        self.assertEqual(book.ids_by_x, {})

        update_elite_archive(book, _certified((1,), 1.0), 1)
        update_elite_archive(book, _certified((1,), 1.0), 1)
        self.assertEqual(book.n_duplicates, 1)
        self.assertEqual(len(book.members), 1)

        update_elite_archive(book, _certified((0,), 0.0), 1)
        self.assertEqual(book.members[0].point.x, (0,))
        self.assertEqual(book.n_rejections, 0)
        evicted_id = book.ids_by_x[(1,)]
        seen = len(book.ids_by_x)

        update_elite_archive(book, _certified((1,), 1.0), 1)
        self.assertEqual(book.n_duplicates, 1)
        self.assertEqual(book.n_rejections, 1)
        self.assertEqual(book.ids_by_x[(1,)], evicted_id)
        self.assertEqual(len(book.ids_by_x), seen)
        self.assertNotIn((1,), [member.point.x for member in book.members])

        admissions = book.n_admissions
        update_elite_archive(book, _certified((1,), 1.0), 2)
        self.assertEqual(book.n_admissions, admissions + 1)
        self.assertEqual(book.ids_by_x[(1,)], evicted_id)
        self.assertEqual(len(book.ids_by_x), seen)

    def test_duplicate_start_counter_resets_after_an_unseen_draw(self) -> None:
        space = DiscreteSearchSpace([0], [5])
        seen = {(0,)}
        rng = ScriptedRng(integers=[[0], [0], [2], [2], [2], [3]])
        first, consecutive = take_unseen_start(space, rng, seen, 0)
        self.assertEqual(first, (2,))
        self.assertEqual(consecutive, 0)
        second, consecutive = take_unseen_start(space, rng, seen, consecutive)
        self.assertEqual(second, (3,))
        self.assertEqual(consecutive, 0)

        capped = ScriptedRng(integers=[[0]] * _CONSECUTIVE_DUPLICATE_START_CAP)
        missing, count = take_unseen_start(space, capped, {(0,)}, 0)
        self.assertIsNone(missing)
        self.assertEqual(count, _CONSECUTIVE_DUPLICATE_START_CAP)


class TestKEliteAMS(unittest.TestCase):
    """Algorithm, registry, and budget behaviour."""

    def test_registry_and_factory(self) -> None:
        spec = get_algorithm_spec("K_ELITE_AMS_PEN_SI_DESC_COLD_START")
        self.assertEqual(spec.library, "native")
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(spec.budget_semantics, BudgetSemantics.HARD_IOH_EVALUATIONS)
        self.assertIsNone(spec.dependency_extra)
        defaults = {parameter.name: parameter.default for parameter in spec.parameters}
        self.assertEqual(defaults["archive_size"], 10)
        self.assertEqual(defaults["n_parents"], 5)
        self.assertEqual(defaults["mutation_probability"], 0.05)
        self.assertEqual(defaults["random_restart_probability"], 0.10)
        self.assertNotIn("initial_sampler", defaults)
        self.assertNotIn("initial_design_path", defaults)

        with self.assertRaises(ValueError):
            build_k_elite_ams({"max_evaluations": 4}, None)
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
                {"max_evaluations": 4, "n_parents": 6, "archive_size": 4},
                random_seed=1,
            )

        algorithm = build_k_elite_ams(
            {
                "max_evaluations": 4,
                "archive_size": 10,
                "n_parents": 5,
                "mutation_probability": 0.05,
                "random_restart_probability": 0.10,
                "initial_design_path": "missing.json",
            },
            3,
        )
        problem = FlatProblem([0], [2])
        algorithm(problem)
        self.assertGreaterEqual(len(problem.calls), 1)

    def test_conformance(self) -> None:
        def build(seed: int | None):
            return create_algorithm(
                "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
                {"max_evaluations": 12, "n_parents": 2, "archive_size": 4},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="K_ELITE_AMS_PEN_SI_DESC_COLD_START",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_same_seed_repeats_the_trace(self) -> None:
        def run(seed: int) -> list[list[int]]:
            problem = FlatProblem([0], [6])
            create_algorithm(
                "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
                {
                    "max_evaluations": 30,
                    "n_parents": 2,
                    "archive_size": 4,
                    "mutation_probability": 0.0,
                    "random_restart_probability": 0.0,
                },
                random_seed=seed,
            )(problem)
            return problem.calls

        self.assertEqual(run(5), run(5))
        self.assertNotEqual(run(5), run(6))

    def test_one_evaluation_budget_does_not_start_the_loop(self) -> None:
        problem = FlatProblem([0], [4])
        problem.state.evaluations = 9
        algorithm = create_algorithm(
            "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
            {"max_evaluations": 1, "n_parents": 5},
            random_seed=1,
        )
        returned = algorithm(problem)
        self.assertIs(returned, algorithm)
        self.assertEqual(len(problem.calls), 1)
        stats = algorithm.last_run_stats
        self.assertEqual(stats["n_evaluations"], 1)
        self.assertEqual(problem.state.evaluations, 10)
        self.assertEqual(stats["n_k_elite_proposals"], 0)
        self.assertEqual(stats["n_random_restarts"], 0)
        self.assertEqual(stats["initial_archive_size"], 0)
        self.assertEqual(stats["initialisation_evaluations"], 1)

    def test_initialisation_evaluations_include_neighbour_calls(self) -> None:
        problem = FlatProblem([0], [20])
        algorithm = create_algorithm(
            "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
            {
                "max_evaluations": 80,
                "n_parents": 5,
                "archive_size": 10,
                "mutation_probability": 0.0,
                "random_restart_probability": 0.0,
            },
            random_seed=2,
        )
        algorithm(problem)
        stats = algorithm.last_run_stats
        self.assertEqual(stats["initial_archive_size"], 5)
        self.assertGreater(
            stats["initialisation_evaluations"],
            stats["n_initial_local_searches"],
        )
        self.assertEqual(stats["n_evaluations"], len(problem.calls))
        members, cursor = _replay_flat_starts(problem.calls, 0, 20, 5)
        self.assertEqual(cursor, stats["initialisation_evaluations"])
        self.assertEqual(len(members), 5)
        _assert_later_starts_use_parents(
            self,
            problem.calls,
            cursor,
            0,
            20,
            members,
        )
        self.assertLess(stats["initial_archive_size"], 10)

    def test_archive_grows_after_startup(self) -> None:
        problem = FlatProblem([0], [30])
        algorithm = create_algorithm(
            "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
            {
                "max_evaluations": 250,
                "n_parents": 5,
                "archive_size": 8,
                "mutation_probability": 1.0,
                "random_restart_probability": 0.0,
            },
            random_seed=4,
        )
        algorithm(problem)
        stats = algorithm.last_run_stats
        archive = stats["final_archive"]
        self.assertEqual(stats["initial_archive_size"], 5)
        self.assertGreater(len(archive), 5)
        self.assertLessEqual(len(archive), 8)
        keys = [(row["f"], tuple(row["x"])) for row in archive]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len({tuple(row["x"]) for row in archive}), len(archive))

    def test_loop_with_five_members_uses_those_parents(self) -> None:
        problem = FlatProblem([0, 0], [9, 9])
        controller = RunController(problem, max_evaluations=1)
        space = DiscreteSearchSpace([0, 0], [9, 9])
        evaluator = EvaluationManager(problem, controller, space)
        book = ArchiveBook()
        vectors = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 0)]
        for index, vector in enumerate(vectors):
            book.members.append(_member(index, vector, float(index)))
            book.ids_by_x[vector] = index
        book.next_id = 5
        stats = {
            "n_local_searches": 0,
            "n_certified_local_searches": 0,
            "n_initial_local_searches": 0,
            "n_k_elite_proposals": 0,
            "n_random_restarts": 0,
        }
        run_k_elite_loop(
            book,
            evaluator,
            space,
            np.random.default_rng(1),
            n_parents=5,
            archive_size=10,
            mutation_probability=0.0,
            random_restart_probability=0.0,
            stats=stats,
        )
        self.assertEqual(stats["n_k_elite_proposals"], 1)
        self.assertEqual(stats["n_random_restarts"], 0)
        child = tuple(problem.calls[0])
        for axis in range(2):
            self.assertIn(child[axis], {vector[axis] for vector in vectors})

    def test_fixed_target_stops_the_run(self) -> None:
        problem = FakeIOHProblem([0], [5])
        controller = RunController(
            problem,
            max_evaluations=100,
            mode="fixed_target",
            target_threshold=0.0,
            target_reference="all",
        )
        algorithm = create_algorithm(
            "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
            {"max_evaluations": 100, "n_parents": 1, "archive_size": 3},
            random_seed=1,
        )
        algorithm(problem, run_controller=controller)
        self.assertTrue(controller.target_hit)
        self.assertEqual(problem.state.current_best.x, [0])
        self.assertLess(len(problem.calls), 100)

    def test_one_point_box_stops_before_a_short_parent_set(self) -> None:
        problem = FlatProblem([4], [4])
        algorithm = create_algorithm(
            "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
            {"max_evaluations": 30, "n_parents": 5, "archive_size": 10},
            random_seed=1,
        )
        algorithm(problem)
        stats = algorithm.last_run_stats
        self.assertEqual(len(problem.calls), 1)
        self.assertEqual(stats["n_k_elite_proposals"], 0)
        self.assertEqual(stats["n_random_restarts"], 0)
        self.assertEqual(stats["initial_archive_size"], 0)
        self.assertEqual(stats["n_distinct_certified_minima_seen"], 1)
        self.assertEqual(stats["n_evaluations"], 1)
        self.assertLess(stats["n_evaluations"], 30)

    def test_reaches_the_unique_minimum(self) -> None:
        problem = FakeIOHProblem([0], [5])
        algorithm = create_algorithm(
            "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
            {"max_evaluations": 80, "n_parents": 5, "archive_size": 10},
            random_seed=7,
        )
        algorithm(problem)
        stats = algorithm.last_run_stats
        self.assertEqual(stats["best_x"], [0])
        archive = stats["final_archive"]
        self.assertLessEqual(len(archive), 10)
        self.assertGreaterEqual(len(archive), 1)
        keys = [(row["f"], tuple(row["x"])) for row in archive]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len({tuple(row["x"]) for row in archive}), len(archive))
        self.assertIn([0], [row["x"] for row in archive])

    def test_short_parent_loop_refuses_to_run(self) -> None:
        problem = FlatProblem([0], [3])
        controller = RunController(problem, max_evaluations=5)
        space = DiscreteSearchSpace([0], [3])
        evaluator = EvaluationManager(problem, controller, space)
        with self.assertRaises(ValueError):
            run_k_elite_loop(
                ArchiveBook(),
                evaluator,
                space,
                np.random.default_rng(0),
                n_parents=5,
                archive_size=10,
                mutation_probability=0.0,
                random_restart_probability=0.0,
                stats={
                    "n_local_searches": 0,
                    "n_certified_local_searches": 0,
                    "n_initial_local_searches": 0,
                    "n_k_elite_proposals": 0,
                    "n_random_restarts": 0,
                },
            )
        self.assertEqual(problem.calls, [])

    def test_budget_abort_keeps_the_initialisation_count(self) -> None:
        problem = FlatProblem([0], [8])
        algorithm = create_algorithm(
            "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
            {
                "max_evaluations": 40,
                "n_parents": 2,
                "archive_size": 4,
                "mutation_probability": 0.0,
                "random_restart_probability": 0.0,
            },
            random_seed=1,
        )

        def raising(
            book,
            evaluator,
            space,
            rng,
            n_parents,
            archive_size,
            mutation_probability,
            random_restart_probability,
            stats,
            local_search=None,
        ):
            del book, rng, n_parents, archive_size, mutation_probability
            del random_restart_probability, stats, local_search
            evaluator.evaluate(tuple(int(value) for value in space.lower))
            raise BudgetExhausted("stop")

        with patch(
            "core.ioh_core.algorithms.native.k_elite_ams.run_k_elite_loop",
            raising,
        ):
            algorithm(problem)
        stats = algorithm.last_run_stats
        self.assertGreater(
            stats["n_evaluations"],
            stats["initialisation_evaluations"],
        )


def _replay_flat_starts(
    calls: list[list[int]],
    low: int,
    high: int,
    n_parents: int,
) -> tuple[list[int], int]:
    """
    Recover certified flat 1-D starts from the evaluation trace.

    On a flat objective the endpoint is the start. Neighbours are the
    in-bounds L1 steps, ``-1`` then ``+1``.
    """
    cursor = 0
    members: list[int] = []
    seen: set[int] = set()
    while len(members) < n_parents:
        start = calls[cursor][0]
        cursor += 1
        for step in (-1, 1):
            neighbour = start + step
            if low <= neighbour <= high:
                if calls[cursor] != [neighbour]:
                    raise AssertionError(
                        f"expected neighbour {[neighbour]} at call {cursor}, "
                        f"got {calls[cursor]}"
                    )
                cursor += 1
        if start not in seen:
            seen.add(start)
            members.append(start)
    return members, cursor


def _assert_later_starts_use_parents(
    test: unittest.TestCase,
    calls: list[list[int]],
    cursor: int,
    low: int,
    high: int,
    members: list[int],
) -> None:
    """Every main-loop start on a flat line lies in the startup archive."""
    allowed = set(members)
    while cursor < len(calls):
        start = calls[cursor][0]
        test.assertIn(start, allowed)
        cursor += 1
        for step in (-1, 1):
            neighbour = start + step
            if neighbour < low or neighbour > high or cursor >= len(calls):
                continue
            if calls[cursor] != [neighbour]:
                break
            cursor += 1


class TestInitialSearchStopsAtParentCount(unittest.TestCase):
    """Initialisation must not keep sampling once ``n_parents`` exist."""

    def test_stops_when_the_archive_can_supply_parents(self) -> None:
        problem = FlatProblem([0], [15])
        controller = RunController(problem, max_evaluations=200)
        space = DiscreteSearchSpace([0], [15])
        evaluator = EvaluationManager(problem, controller, space)
        book = ArchiveBook()
        stats = {
            "n_local_searches": 0,
            "n_certified_local_searches": 0,
            "n_initial_local_searches": 0,
        }
        initial_local_searches(
            book,
            evaluator,
            space,
            np.random.default_rng(0),
            n_parents=3,
            archive_size=10,
            stats=stats,
        )
        self.assertEqual(len(book.members), 3)
        self.assertLess(stats["n_initial_local_searches"], 10)
        self.assertTrue(evaluator.has_budget())


def _nine(index: int) -> list[int]:
    """One catalogue vector that varies only in the first coordinate."""
    return [index, 0, 0, 0, 0, 0, 0, 0, 0]


def _write_initial_design(path: Path, vectors: list[list[int]]) -> None:
    """Write a schema-2 InitialDesign the production loader accepts."""
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "problem_id": "L8100_B2",
                "n": len(vectors),
                "vectors": vectors,
            }
        ),
        encoding="utf-8",
    )


def _flat_neighbourhood_end(
    calls: list[list[int]],
    cursor: int,
    start: list[int],
    low: list[int],
    high: list[int],
) -> int:
    """Advance past the unevaluated-improvement neighbourhood of ``start``."""
    for dimension, value in enumerate(start):
        for step in (-1, 1):
            neighbour = list(start)
            neighbour[dimension] = value + step
            if neighbour[dimension] < low[dimension] or neighbour[dimension] > high[dimension]:
                continue
            if cursor >= len(calls) or calls[cursor] != neighbour:
                return cursor
            cursor += 1
    return cursor


def _assert_warm_restarts_use_parent_coordinates(
    test: unittest.TestCase,
    calls: list[list[int]],
    cursor: int,
    low: list[int],
    high: list[int],
    parents: list[list[int]],
) -> None:
    """Every main-loop start coordinate is a value held by the prefix."""
    allowed = [set() for _ in parents[0]]
    for parent in parents:
        for dimension, value in enumerate(parent):
            allowed[dimension].add(value)
    while cursor < len(calls):
        start = calls[cursor]
        for dimension, value in enumerate(start):
            test.assertIn(value, allowed[dimension])
        cursor = _flat_neighbourhood_end(calls, cursor + 1, start, low, high)


class TestKEliteAMSWarmStart(unittest.TestCase):
    """Ranked-prefix startup for ``K_ELITE_AMS_PEN_SI_DESC_WARM_START``."""

    def _run(
        self,
        vectors: list[list[int]],
        *,
        budget: int,
        n_parents: int = 5,
        archive_size: int = 10,
        seed: int = 1,
        mutation_probability: float = 0.0,
        random_restart_probability: float = 0.0,
        lower: list[int] | None = None,
        upper: list[int] | None = None,
        problem_cls: type = FlatProblem,
    ):
        lower = [0] * 9 if lower is None else lower
        upper = [4] * 9 if upper is None else upper
        problem = problem_cls(lower, upper)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            _write_initial_design(path, vectors)
            algorithm = create_algorithm(
                "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
                {
                    "max_evaluations": budget,
                    "n_parents": n_parents,
                    "archive_size": archive_size,
                    "mutation_probability": mutation_probability,
                    "random_restart_probability": random_restart_probability,
                    "initial_design_path": str(path),
                },
                random_seed=seed,
            )
            algorithm(problem)
        return problem, algorithm.last_run_stats

    def test_factory_requires_seed_and_path(self) -> None:
        with self.assertRaises(ValueError):
            build_k_elite_ams_warm_start(
                {"max_evaluations": 4, "initial_design_path": "design.json"},
                None,
            )
        with self.assertRaises(ValueError):
            build_k_elite_ams_warm_start({"max_evaluations": 4}, 1)
        with self.assertRaises(ValueError):
            build_k_elite_ams_warm_start(
                {"max_evaluations": 4, "initial_design_path": ""},
                1,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
                {"max_evaluations": 4, "initial_design_path": "missing.json"},
                random_seed=None,
            )

    def test_missing_file_evaluates_nothing(self) -> None:
        problem = FlatProblem([0] * 9, [4] * 9)
        algorithm = create_algorithm(
            "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
            {
                "max_evaluations": 8,
                "n_parents": 2,
                "initial_design_path": "missing-initial-design.json",
            },
            random_seed=1,
        )
        with self.assertRaises(ValueError) as raised:
            algorithm(problem)
        self.assertIn("has no InitialDesign", str(raised.exception))
        self.assertEqual(problem.calls, [])

    def test_short_file_evaluates_nothing(self) -> None:
        problem = FlatProblem([0] * 9, [4] * 9)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            _write_initial_design(path, [_nine(0), _nine(1)])
            algorithm = create_algorithm(
                "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
                {
                    "max_evaluations": 8,
                    "n_parents": 3,
                    "initial_design_path": str(path),
                },
                random_seed=1,
            )
            with self.assertRaises(ValueError) as raised:
                algorithm(problem)
        message = str(raised.exception)
        self.assertIn("needs 3 initial designs", message)
        self.assertIn("contains 2", message)
        self.assertEqual(problem.calls, [])

    def test_short_budget_evaluates_nothing(self) -> None:
        problem = FlatProblem([0] * 9, [4] * 9)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            _write_initial_design(path, [_nine(index) for index in range(4)])
            algorithm = create_algorithm(
                "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
                {
                    "max_evaluations": 2,
                    "n_parents": 3,
                    "initial_design_path": str(path),
                },
                random_seed=1,
            )
            with self.assertRaises(ValueError) as raised:
                algorithm(problem)
        self.assertIn("needs 3 initial evaluations", str(raised.exception))
        self.assertEqual(problem.calls, [])

    def test_out_of_bounds_prefix_evaluates_nothing(self) -> None:
        problem = FlatProblem([0] * 9, [4] * 9)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            _write_initial_design(path, [_nine(0), _nine(9)])
            algorithm = create_algorithm(
                "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
                {
                    "max_evaluations": 4,
                    "n_parents": 2,
                    "initial_design_path": str(path),
                },
                random_seed=1,
            )
            with self.assertRaises(ValueError):
                algorithm(problem)
        self.assertEqual(problem.calls, [])

    def test_budget_equal_to_prefix_evaluates_only_the_prefix(self) -> None:
        vectors = [_nine(index) for index in range(3)]
        problem, stats = self._run(vectors, budget=3, n_parents=3)
        self.assertEqual(problem.calls, vectors)
        self.assertEqual(stats["n_initial_local_searches"], 0)
        self.assertEqual(stats["n_k_elite_proposals"], 0)
        self.assertEqual(stats["initial_archive_size"], 0)

    def test_prefix_is_evaluated_before_the_first_neighbour(self) -> None:
        vectors = [_nine(index) for index in range(4)]
        problem, stats = self._run(vectors, budget=6, n_parents=3, archive_size=10)
        self.assertEqual(problem.calls[:3], vectors[:3])
        self.assertEqual(problem.calls[3], [1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(stats["n_initial_local_searches"], 1)
        self.assertGreaterEqual(stats["initialisation_evaluations"], 4)

    def test_parent_override_sets_the_prefix_length(self) -> None:
        vectors = [_nine(index) for index in range(4)]
        problem, _stats = self._run(
            vectors,
            budget=3,
            n_parents=2,
            archive_size=10,
        )
        self.assertEqual(problem.calls[:2], vectors[:2])
        self.assertNotEqual(problem.calls[2], vectors[2])

    def test_flat_prefix_enters_the_loop_inside_the_parents(self) -> None:
        vectors = [_nine(index) for index in range(5)]
        low = [0] * 9
        high = [4] * 9
        problem, stats = self._run(
            vectors,
            budget=120,
            n_parents=5,
            mutation_probability=0.0,
            random_restart_probability=0.0,
            lower=low,
            upper=high,
        )
        self.assertEqual(stats["initial_archive_size"], 5)
        self.assertEqual(problem.calls[:5], vectors)
        _assert_warm_restarts_use_parent_coordinates(
            self,
            problem.calls,
            stats["initialisation_evaluations"],
            low,
            high,
            vectors,
        )
        self.assertGreater(stats["n_local_searches"], stats["n_initial_local_searches"])

    def test_one_point_box_does_not_start_the_loop(self) -> None:
        vectors = [[4] * 9 for _ in range(5)]
        problem, stats = self._run(
            vectors,
            budget=20,
            n_parents=5,
            lower=[4] * 9,
            upper=[4] * 9,
        )
        self.assertEqual(problem.calls[:5], vectors)
        self.assertEqual(problem.calls[5:], [vectors[0]])
        self.assertEqual(stats["n_initial_local_searches"], 6)
        self.assertEqual(stats["initial_archive_size"], 0)
        self.assertEqual(stats["n_k_elite_proposals"], 0)
        self.assertEqual(stats["n_random_restarts"], 0)
        self.assertEqual(len(stats["final_archive"]), 1)
        self.assertEqual(stats["n_archive_duplicates"], 5)

    def test_duplicate_row_is_evaluated_and_counted(self) -> None:
        vectors = [_nine(0), _nine(0)]
        problem, stats = self._run(
            vectors,
            budget=30,
            n_parents=2,
            lower=[0] * 9,
            upper=[0] * 9,
        )
        self.assertEqual(problem.calls[:2], vectors)
        self.assertEqual(problem.calls[2:], [vectors[0]])
        self.assertEqual(stats["n_archive_duplicates"], 2)
        self.assertEqual(len(stats["final_archive"]), 1)
        self.assertEqual(stats["n_k_elite_proposals"], 0)

    def test_budget_abort_keeps_the_initialisation_count(self) -> None:
        vectors = [_nine(index) for index in range(3)]
        problem = FlatProblem([0] * 9, [4] * 9)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            _write_initial_design(path, vectors)

            def raising(
                book,
                evaluator,
                space,
                rng,
                n_parents,
                archive_size,
                mutation_probability,
                random_restart_probability,
                stats,
                local_search=None,
            ):
                del book, rng, n_parents, archive_size, mutation_probability
                del random_restart_probability, stats, local_search
                evaluator.evaluate(tuple(int(value) for value in space.lower))
                raise BudgetExhausted("stop")

            algorithm = create_algorithm(
                "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
                {
                    "max_evaluations": 80,
                    "n_parents": 2,
                    "archive_size": 4,
                    "mutation_probability": 0.0,
                    "random_restart_probability": 0.0,
                    "initial_design_path": str(path),
                },
                random_seed=1,
            )
            with patch(
                "core.ioh_core.algorithms.native.k_elite_ams.run_k_elite_loop",
                raising,
            ):
                algorithm(problem)
        stats = algorithm.last_run_stats
        self.assertGreater(stats["n_evaluations"], stats["initialisation_evaluations"])

    def test_same_seed_repeats_and_startup_ignores_the_seed(self) -> None:
        vectors = [_nine(index) for index in range(5)]
        first, first_stats = self._run(vectors, budget=80, seed=4)
        second, second_stats = self._run(vectors, budget=80, seed=4)
        other, other_stats = self._run(vectors, budget=80, seed=5)
        self.assertEqual(first.calls, second.calls)
        startup = first_stats["initialisation_evaluations"]
        self.assertEqual(startup, other_stats["initialisation_evaluations"])
        self.assertEqual(first.calls[:startup], other.calls[:startup])
        self.assertNotEqual(first.calls, other.calls)
        low = [0] * 9
        high = [4] * 9
        _assert_warm_restarts_use_parent_coordinates(
            self,
            other.calls,
            other_stats["initialisation_evaluations"],
            low,
            high,
            vectors,
        )

    def test_short_prefix_archive_is_filled_before_the_loop(self) -> None:
        origin = [0] * 9
        vectors = [list(origin) for _ in range(3)]
        problem, stats = self._run(
            vectors,
            budget=250,
            n_parents=3,
            lower=[0] * 9,
            upper=[2] * 9,
            seed=2,
        )
        self.assertEqual(problem.calls[:3], vectors)
        self.assertEqual(stats["initial_archive_size"], 3)
        self.assertGreaterEqual(len(stats["final_archive"]), 3)
        self.assertGreater(
            stats["n_local_searches"],
            stats["n_initial_local_searches"],
        )

    def test_conformance(self) -> None:
        vectors = [_nine(index) for index in range(4)]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            _write_initial_design(path, vectors)

            def build(seed: int | None):
                return create_algorithm(
                    "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
                    {
                        "max_evaluations": 40,
                        "n_parents": 2,
                        "archive_size": 4,
                        "initial_design_path": str(path),
                    },
                    random_seed=seed,
                )

            assert_algorithm_conformance(
                self,
                algorithm_name="K_ELITE_AMS_PEN_SI_DESC_WARM_START",
                build_algorithm=build,
                seed=11,
                second_seed=29,
                make_problem=FlatProblem,
                lower=[0] * 9,
                upper=[4] * 9,
            )
