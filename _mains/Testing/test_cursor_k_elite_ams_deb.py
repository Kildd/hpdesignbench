"""Constraint-aware k-elite AMS unit, lifecycle, and HP-slab smoke tests."""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem
from core.ioh_core.algorithm_factory import create_algorithm
from core.ioh_core.algorithms.common import RunController, attach_eval_context
from core.ioh_core.algorithms.native.discrete_search import (
    DebEvaluatedPoint,
    DebEvaluationManager,
    DebLocalSearchResult,
    DiscreteSearchSpace,
    EvaluatedPoint,
    PBHDebEvaluatedPoint,
    PBHDebEvaluationManager,
    PBHDebLocalSearchResult,
    deb_best_improvement_local_search,
    deb_point_key,
    deb_strictly_better,
    pbh_deb_best_improvement_local_search,
    pbh_deb_key,
    pbh_strictly_better,
)
from core.ioh_core.algorithms.native.k_elite_ams import (
    DebArchiveBook,
    DebEliteMember,
    EliteMember,
    KEliteAMSDeb,
    KEliteAMSDebWarmStart,
    _CONSECUTIVE_DUPLICATE_START_CAP,
    initial_local_searches_deb,
    select_rank_weighted,
    select_rank_weighted_deb,
    update_elite_archive_deb,
    validate_k_elite_ams_deb_parameters,
    validate_k_elite_ams_parameters,
)
from core.ioh_core.algorithms.registry import RegistryError, get_algorithm_spec
from core.ioh_core.problem_builder import build_problems_for_slab_type
from core.ioh_core.workflows.la.deb import deb_key


class _DebProblem:
    """IOH-like problem with an explicit auxiliary record per index."""

    def __init__(
        self,
        records: dict[tuple[int, ...], dict[str, Any]],
        *,
        lower: list[int],
        upper: list[int],
        constraint_names: list[str] | None = None,
        invalid_utilization: float = 99.0,
        on_call: Any = None,
    ) -> None:
        self.records = records
        self.bounds = SimpleNamespace(lb=lower, ub=upper)
        self.meta_data = SimpleNamespace(n_variables=len(lower))
        self.state = SimpleNamespace(
            evaluations=0,
            current_best=SimpleNamespace(y=math.inf, x=[]),
        )
        self.calls: list[tuple[int, ...]] = []
        self.on_call = on_call
        self.hdb_eval_context = SimpleNamespace(
            _c_names=["g"] if constraint_names is None else list(constraint_names),
            run_config=SimpleNamespace(invalid_utilization=invalid_utilization),
            last_data=None,
        )

    def __call__(self, x: Any) -> float:
        point = tuple(int(value) for value in x)
        self.calls.append(point)
        if self.on_call is not None:
            self.on_call(point)
        self.state.evaluations += 1
        if point not in self.records:
            self.records[point] = _record(y_p=1.0, y=1.0)
        record = dict(self.records[point])
        self.hdb_eval_context.last_data = record
        y_p = float(record["y_p"])
        if math.isfinite(y_p) and y_p < self.state.current_best.y:
            self.state.current_best = SimpleNamespace(y=y_p, x=list(point))
        return y_p


class ScriptedRng:
    """Queue stand-in for the Generator methods AMS draws."""

    def __init__(
        self,
        *,
        integers: list[Any] | None = None,
        randoms: list[float] | None = None,
        choices: list[int] | None = None,
    ) -> None:
        self.integers_queue = list(integers or [])
        self.randoms = list(randoms or [])
        self.choices = list(choices or [])
        self.draws = 0

    def integers(self, low: Any, high: Any = None, size: Any = None) -> Any:
        del low, high, size
        self.draws += 1
        value = self.integers_queue.pop(0)
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        return value

    def random(self) -> float:
        self.draws += 1
        if self.randoms:
            return float(self.randoms.pop(0))
        return 1.0

    def choice(self, a: Any, p: Any = None) -> Any:
        del a
        self.draws += 1
        if self.choices:
            return int(self.choices.pop(0))
        probs = np.asarray(p, dtype=np.float64)
        return int(np.flatnonzero(probs > 0.0)[0])


class RecordingRng(ScriptedRng):
    """Records the probability vector passed to ``choice``."""

    def __init__(self) -> None:
        super().__init__()
        self.probabilities: list[np.ndarray] = []

    def choice(self, a: Any, p: Any = None) -> Any:
        del a
        self.draws += 1
        probs = np.asarray(p, dtype=np.float64)
        self.probabilities.append(probs.copy())
        return int(np.flatnonzero(probs > 0.0)[0])


def _record(
    *,
    y_p: float,
    y: float,
    violation: float = 0.0,
    analysis_ok: bool = True,
    feasible: bool | None = None,
    utilization: float | None = None,
    violations: dict[str, float] | None = None,
    constraint_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    if feasible is None:
        feasible = bool(analysis_ok and violation == 0.0 and not violations)
    if violations is None:
        violations = {"g": violation}
    if constraint_values is None:
        eta = violation + 1.0 if utilization is None else utilization
        constraint_values = {name: eta for name in violations}
    return {
        "y_p": y_p,
        "y": y,
        "analysis_ok": analysis_ok,
        "feasible": feasible,
        "violations_": violations,
        "constraint_values": constraint_values,
    }


def _point(
    x: tuple[int, ...],
    *,
    y: float,
    phi: float,
    feasible: bool,
    y_p: float | None = None,
) -> DebEvaluatedPoint:
    return DebEvaluatedPoint(
        x=x,
        y_p=y if y_p is None else y_p,
        y=y,
        phi=phi,
        feasible=feasible,
        analysis_ok=True,
        sentinel_constraints=(),
    )


def _member(minimum_id: int, point: DebEvaluatedPoint) -> DebEliteMember:
    return DebEliteMember(minimum_id, point)


def _certified(point: DebEvaluatedPoint) -> DebLocalSearchResult:
    return DebLocalSearchResult(
        point=point,
        certified_local_optimum=True,
        n_steps=0,
        n_evaluations=0,
    )


def _manager(
    problem: _DebProblem,
    *,
    budget: int = 20,
    mode: str = "fixed_budget",
    target: float | None = None,
    label: str = "k-elite AMS-Deb",
) -> tuple[DebEvaluationManager, RunController]:
    controller = RunController(
        problem,
        budget,
        mode=mode,
        target_threshold=target,
        target_reference="all" if mode == "fixed_target" else None,
    )
    space = DiscreteSearchSpace(problem.bounds.lb, problem.bounds.ub)
    return (
        DebEvaluationManager(problem, controller, space, algorithm_label=label),
        controller,
    )


def _assert_identities(test: unittest.TestCase, stats: dict[str, Any]) -> None:
    test.assertEqual(
        stats["n_evaluations"],
        stats["n_failed_analysis_evaluations"]
        + stats["n_feasible_evaluations"]
        + stats["n_infeasible_evaluations"],
    )
    test.assertEqual(
        stats["n_valid_evaluations"],
        stats["n_feasible_evaluations"] + stats["n_infeasible_evaluations"],
    )
    test.assertEqual(
        stats["final_archive_size"],
        stats["final_feasible_archive_size"] + stats["final_infeasible_archive_size"],
    )
    test.assertEqual(
        stats["n_evaluations"],
        stats["n_start_evaluations"] + stats["n_local_search_evaluations"],
    )
    test.assertEqual(
        stats["n_start_evaluations"],
        stats["n_initial_start_evaluations"] + stats["n_main_start_evaluations"],
    )
    test.assertEqual(
        stats["n_local_search_evaluations"],
        stats["n_initial_local_search_evaluations"]
        + stats["n_main_local_search_evaluations"],
    )
    test.assertLessEqual(
        stats["n_certification_evaluations"],
        stats["n_local_search_evaluations"],
    )
    test.assertEqual(
        stats["initialisation_evaluations"],
        stats["n_initial_start_evaluations"]
        + stats["n_initial_local_search_evaluations"],
    )
    test.assertEqual(
        stats["n_local_searches"],
        stats["n_certified_local_searches"] + stats["n_uncertified_local_searches"],
    )
    test.assertEqual(
        stats["startup_archive_size"],
        stats["startup_feasible_archive_size"]
        + stats["startup_infeasible_archive_size"],
    )
    test.assertEqual(
        stats["initial_archive_size"],
        stats["initial_feasible_archive_size"]
        + stats["initial_infeasible_archive_size"],
    )


def _design(path: Path, vectors: list[list[int]], *, schema: int = 1) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": schema,
                "problem_id": "L8100_B2",
                "n": len(vectors),
                "vectors": vectors,
            }
        ),
        encoding="utf-8",
    )


class TestNeutralDebCore(unittest.TestCase):
    def test_pbh_aliases_are_the_neutral_objects(self) -> None:
        self.assertIs(PBHDebEvaluatedPoint, DebEvaluatedPoint)
        self.assertIs(PBHDebLocalSearchResult, DebLocalSearchResult)
        self.assertIs(PBHDebEvaluationManager, DebEvaluationManager)
        self.assertIs(pbh_deb_key, deb_point_key)
        self.assertIs(pbh_strictly_better, deb_strictly_better)
        self.assertIs(
            pbh_deb_best_improvement_local_search,
            deb_best_improvement_local_search,
        )

    def test_key_matches_landscape_definition(self) -> None:
        point = _point((1,), y=4.0, phi=0.2, feasible=False)
        self.assertEqual(
            deb_point_key(point),
            deb_key(feasible=False, y=4.0, phi=0.2),
        )

    def test_ams_and_pbh_error_prefixes_stay_distinct(self) -> None:
        problem = FakeIOHProblem([0], [1])
        controller = RunController(problem, 2)
        space = DiscreteSearchSpace([0], [1])
        with self.assertRaisesRegex(ValueError, "PBH-L1-Deb"):
            PBHDebEvaluationManager(problem, controller, space)
        with self.assertRaisesRegex(ValueError, "k-elite AMS-Deb"):
            DebEvaluationManager(
                problem,
                controller,
                space,
                algorithm_label="k-elite AMS-Deb",
            )

    def test_invalid_start_is_searched_and_all_invalid_is_not_certified(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(
                    y_p=math.inf, y=math.inf, analysis_ok=False, feasible=False
                ),
                (1,): _record(y_p=3.0, y=3.0, violation=0.0),
            },
            lower=[0],
            upper=[1],
        )
        manager, _controller = _manager(problem)
        space = DiscreteSearchSpace([0], [1])
        start = manager.evaluate((0,))
        result = deb_best_improvement_local_search(start, manager, space)
        self.assertEqual(result.point.x, (1,))
        self.assertTrue(result.certified_local_optimum)
        book = DebArchiveBook()
        update_elite_archive_deb(book, result, 2)
        self.assertEqual([member.point.x for member in book.members], [(1,)])

        failed = _DebProblem(
            {
                (0,): _record(
                    y_p=math.inf, y=math.inf, analysis_ok=False, feasible=False
                ),
                (1,): _record(
                    y_p=math.inf, y=math.inf, analysis_ok=False, feasible=False
                ),
            },
            lower=[0],
            upper=[1],
        )
        failed_manager, _controller = _manager(failed)
        failed_start = failed_manager.evaluate((0,))
        failed_result = deb_best_improvement_local_search(
            failed_start,
            failed_manager,
            space,
        )
        self.assertFalse(failed_result.certified_local_optimum)
        empty = DebArchiveBook()
        update_elite_archive_deb(empty, failed_result, 2)
        self.assertEqual(empty.members, [])
        self.assertEqual(empty.n_admissions, 0)

    def test_fatal_errors_are_not_counted_as_failed_analyses(self) -> None:
        problem = _DebProblem(
            {(0,): _record(y_p=1.0, y=1.0, violation=0.0)},
            lower=[0],
            upper=[0],
        )
        problem.records[(0,)]["y_p"] = math.nan
        manager, _controller = _manager(problem)
        with self.assertRaises(RuntimeError):
            manager.evaluate((0,))
        self.assertEqual(manager.diagnostics["n_failed_analysis_evaluations"], 0)

        broken = _DebProblem(
            {(0,): {"y_p": 1.0, "y": 1.0, "analysis_ok": True, "feasible": True}},
            lower=[0],
            upper=[0],
        )
        broken_manager, _controller = _manager(broken)
        with self.assertRaises(ValueError):
            broken_manager.evaluate((0,))
        self.assertEqual(
            broken_manager.diagnostics["n_failed_analysis_evaluations"],
            0,
        )

        encoded = _DebProblem(
            {
                (0,): _record(
                    y_p=math.inf, y=math.inf, analysis_ok=False, feasible=False
                )
            },
            lower=[0],
            upper=[0],
        )
        encoded_manager, _controller = _manager(encoded)
        point = encoded_manager.evaluate((0,))
        self.assertFalse(point.analysis_ok)
        self.assertEqual(encoded_manager.diagnostics["n_failed_analysis_evaluations"], 1)
        self.assertIsNone(encoded_manager.best_deb_point)

    def test_phi_uses_every_active_name_and_rejects_undeclared_keys(self) -> None:
        names = ["g", "Z1_nt_dt_combination_capacity"]
        problem = _DebProblem(
            {
                (0,): _record(
                    y_p=4.0,
                    y=2.0,
                    violations={"g": 0.2, "Z1_nt_dt_combination_capacity": 0.3},
                    constraint_values={
                        "g": 1.2,
                        "Z1_nt_dt_combination_capacity": 1.3,
                    },
                    feasible=False,
                )
            },
            lower=[0],
            upper=[0],
            constraint_names=names,
        )
        manager, _controller = _manager(problem)
        point = manager.evaluate((0,))
        self.assertEqual(point.phi, 0.5)

        extra = _DebProblem(
            {
                (0,): _record(
                    y_p=4.0,
                    y=2.0,
                    violations={
                        "g": 0.2,
                        "Z1_nt_dt_combination_capacity": 0.3,
                        "undeclared": 1.0,
                    },
                    constraint_values={
                        "g": 1.2,
                        "Z1_nt_dt_combination_capacity": 1.3,
                    },
                    feasible=False,
                )
            },
            lower=[0],
            upper=[0],
            constraint_names=names,
        )
        extra_manager, _controller = _manager(extra)
        with self.assertRaisesRegex(ValueError, "exactly"):
            extra_manager.evaluate((0,))

    def test_equal_phi_sentinel_neighbours_do_not_move(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(
                    y_p=9.0,
                    y=9.0,
                    violation=1.0,
                    utilization=99.0,
                    feasible=False,
                ),
                (1,): _record(
                    y_p=1.0,
                    y=1.0,
                    violation=1.0,
                    utilization=99.0,
                    feasible=False,
                ),
            },
            lower=[0],
            upper=[1],
        )
        manager, _controller = _manager(problem)
        start = manager.evaluate((0,))
        result = deb_best_improvement_local_search(
            start,
            manager,
            DiscreteSearchSpace([0], [1]),
        )
        self.assertEqual(result.point.x, (0,))
        self.assertTrue(result.certified_local_optimum)
        self.assertEqual(manager.diagnostics["n_infeasible_plateau_terminations"], 1)
        self.assertEqual(
            manager.diagnostics["n_sentinel_infeasible_plateau_terminations"],
            1,
        )

    def test_certification_counts_only_the_terminal_neighbourhood(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(y_p=3.0, y=3.0),
                (1,): _record(y_p=1.0, y=1.0),
                (2,): _record(y_p=4.0, y=4.0),
            },
            lower=[0],
            upper=[2],
        )
        manager, _controller = _manager(problem)
        start = manager.evaluate((0,))
        result = deb_best_improvement_local_search(
            start,
            manager,
            DiscreteSearchSpace([0], [2]),
        )
        self.assertEqual(result.point.x, (1,))
        self.assertTrue(result.certified_local_optimum)
        self.assertEqual(result.n_evaluations, 2)
        self.assertEqual(result.n_certification_evaluations, 1)

        short = _DebProblem(
            {
                (0,): _record(y_p=3.0, y=3.0),
                (1,): _record(y_p=1.0, y=1.0),
            },
            lower=[0],
            upper=[1],
        )
        short_manager, _controller = _manager(short, budget=1)
        short_start = short_manager.evaluate((0,))
        interrupted = deb_best_improvement_local_search(
            short_start,
            short_manager,
            DiscreteSearchSpace([0], [1]),
        )
        self.assertFalse(interrupted.certified_local_optimum)
        self.assertEqual(interrupted.n_certification_evaluations, 0)

        targeted = _DebProblem(
            {
                (0,): _record(y_p=5.0, y=5.0),
                (1,): _record(y_p=0.5, y=0.5),
            },
            lower=[0],
            upper=[1],
        )
        target_manager, _controller = _manager(
            targeted,
            budget=5,
            mode="fixed_target",
            target=1.0,
        )
        target_start = target_manager.evaluate((0,))
        hit = deb_best_improvement_local_search(
            target_start,
            target_manager,
            DiscreteSearchSpace([0], [1]),
        )
        self.assertFalse(hit.certified_local_optimum)
        self.assertEqual(hit.n_certification_evaluations, 0)


class TestDenseRankSelection(unittest.TestCase):
    def test_deb_order_and_equal_tie_weights(self) -> None:
        members = [
            _member(0, _point((0,), y=1.0, phi=0.0, feasible=True, y_p=50.0)),
            _member(1, _point((1,), y=1.0, phi=0.0, feasible=True, y_p=0.0)),
            _member(2, _point((0,), y=2.0, phi=0.0, feasible=True)),
            _member(3, _point((0,), y=9.0, phi=0.1, feasible=False, y_p=0.0)),
            _member(4, _point((1,), y=0.0, phi=0.1, feasible=False, y_p=80.0)),
        ]
        rng = RecordingRng()
        select_rank_weighted_deb(members, 1, rng)
        weights = np.array([1.0, 1.0, 0.5, 1.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_allclose(rng.probabilities[0], weights / weights.sum())

    def test_no_tie_selection_matches_historical_ids(self) -> None:
        historical = [
            EliteMember(0, EvaluatedPoint((2,), 3.0)),
            EliteMember(1, EvaluatedPoint((0,), 1.0)),
            EliteMember(2, EvaluatedPoint((1,), 2.0)),
        ]
        deb_members = [
            _member(0, _point((2,), y=3.0, phi=0.0, feasible=True)),
            _member(1, _point((0,), y=1.0, phi=0.0, feasible=True)),
            _member(2, _point((1,), y=2.0, phi=0.0, feasible=True)),
        ]
        left = select_rank_weighted(historical, 3, np.random.default_rng(7))
        right = select_rank_weighted_deb(deb_members, 3, np.random.default_rng(7))
        self.assertEqual(
            [member.minimum_id for member in left],
            [member.minimum_id for member in right],
        )

    def test_x_orders_ties_without_changing_weights_and_short_archive_fails(self) -> None:
        tied = [
            _member(1, _point((2,), y=1.0, phi=0.2, feasible=False)),
            _member(0, _point((0,), y=5.0, phi=0.2, feasible=False)),
        ]
        rng = RecordingRng()
        chosen = select_rank_weighted_deb(tied, 2, rng)
        self.assertEqual([member.minimum_id for member in chosen], [0, 1])
        np.testing.assert_allclose(rng.probabilities[0], [0.5, 0.5])
        with self.assertRaisesRegex(ValueError, "archive members"):
            select_rank_weighted_deb(tied[:1], 2, RecordingRng())


class TestDebArchive(unittest.TestCase):
    def test_bookkeeping_boundary_ties_and_feasibility(self) -> None:
        book = DebArchiveBook()
        update_elite_archive_deb(
            book,
            DebLocalSearchResult(
                point=_point((0,), y=1.0, phi=1.0, feasible=False),
                certified_local_optimum=False,
                n_steps=0,
                n_evaluations=1,
            ),
            2,
        )
        self.assertEqual(book.n_admissions, 0)
        self.assertEqual(book.ids_by_x, {})

        first = _point((1,), y=2.0, phi=0.0, feasible=True)
        second = _point((0,), y=2.0, phi=0.0, feasible=True)
        update_elite_archive_deb(book, _certified(first), 2)
        update_elite_archive_deb(book, _certified(second), 2)
        self.assertEqual([member.point.x for member in book.members], [(0,), (1,)])
        update_elite_archive_deb(book, _certified(second), 2)
        self.assertEqual(book.n_duplicates, 1)

        incoming = _point((2,), y=2.0, phi=0.0, feasible=True)
        update_elite_archive_deb(book, _certified(incoming), 2)
        self.assertEqual(book.n_boundary_tie_events, 1)
        self.assertEqual(book.n_boundary_tie_candidate_rejections, 1)
        self.assertEqual(book.n_boundary_tie_existing_evictions, 0)
        self.assertEqual([member.point.x for member in book.members], [(0,), (1,)])
        self.assertIn((2,), book.ids_by_x)

        mixed = DebArchiveBook()
        update_elite_archive_deb(
            mixed,
            _certified(_point((2,), y=1.0, phi=0.0, feasible=True)),
            2,
        )
        update_elite_archive_deb(
            mixed,
            _certified(_point((4,), y=1.0, phi=0.0, feasible=True)),
            2,
        )
        update_elite_archive_deb(
            mixed,
            _certified(_point((3,), y=1.0, phi=0.0, feasible=True)),
            2,
        )
        self.assertEqual(mixed.n_boundary_tie_existing_evictions, 1)
        self.assertEqual(
            [member.point.x for member in mixed.members],
            [(2,), (3,)],
        )

        feasible_only = DebArchiveBook()
        update_elite_archive_deb(
            feasible_only,
            _certified(_point((0,), y=1.0, phi=0.0, feasible=True)),
            1,
        )
        update_elite_archive_deb(
            feasible_only,
            _certified(_point((1,), y=0.0, phi=0.4, feasible=False, y_p=0.0)),
            1,
        )
        self.assertEqual(feasible_only.n_rejections, 1)
        self.assertEqual(feasible_only.n_boundary_tie_events, 0)
        self.assertTrue(feasible_only.members[0].point.feasible)

        open_book = DebArchiveBook()
        update_elite_archive_deb(
            open_book,
            _certified(_point((0,), y=3.0, phi=0.0, feasible=True)),
            3,
        )
        update_elite_archive_deb(
            open_book,
            _certified(_point((1,), y=0.0, phi=0.2, feasible=False)),
            3,
        )
        self.assertEqual(len(open_book.members), 2)
        self.assertEqual(
            [member.point.feasible for member in open_book.members],
            [True, False],
        )

        evicted = DebArchiveBook()
        update_elite_archive_deb(
            evicted,
            _certified(_point((5,), y=4.0, phi=0.0, feasible=True)),
            1,
        )
        original = evicted.ids_by_x[(5,)]
        update_elite_archive_deb(
            evicted,
            _certified(_point((1,), y=1.0, phi=0.0, feasible=True)),
            1,
        )
        update_elite_archive_deb(
            evicted,
            _certified(_point((5,), y=0.5, phi=0.0, feasible=True)),
            1,
        )
        self.assertEqual(evicted.ids_by_x[(5,)], original)
        self.assertEqual(evicted.members[0].point.x, (5,))


class TestColdLifecycle(unittest.TestCase):
    def test_missing_context_fails_before_evaluation(self) -> None:
        problem = FakeIOHProblem([0], [1])
        algorithm = KEliteAMSDeb(max_evaluations=3, seed=1, n_parents=1, archive_size=1)
        with self.assertRaisesRegex(ValueError, "EvalContext"):
            algorithm(problem)
        self.assertEqual(problem.calls, [])

    def test_same_seed_replays_and_initialization_stops_at_n_parents(self) -> None:
        def problem() -> _DebProblem:
            records = {
                (index,): _record(y_p=float(index), y=float(index))
                for index in range(6)
            }
            return _DebProblem(records, lower=[0], upper=[5])

        first = problem()
        second = problem()
        third = problem()
        kwargs = {
            "max_evaluations": 30,
            "n_parents": 1,
            "archive_size": 3,
            "mutation_probability": 0.0,
            "random_restart_probability": 0.0,
        }
        KEliteAMSDeb(seed=0, **kwargs)(first)
        KEliteAMSDeb(seed=0, **kwargs)(second)
        KEliteAMSDeb(seed=1, **kwargs)(third)
        self.assertEqual(first.calls, second.calls)
        self.assertNotEqual(first.calls, third.calls)
        self.assertLessEqual(
            first.calls.count(first.calls[0]),
            len(first.calls),
        )

        wide = _DebProblem(
            {
                (index,): _record(y_p=float(10 + index), y=float(10 + index))
                for index in range(8)
            },
            lower=[0],
            upper=[7],
        )
        algorithm = KEliteAMSDeb(
            max_evaluations=12,
            seed=1,
            n_parents=1,
            archive_size=4,
            mutation_probability=0.0,
            random_restart_probability=1.0,
        )
        algorithm(wide)
        self.assertEqual(algorithm.last_run_stats["startup_archive_size"], 1)
        self.assertLess(algorithm.last_run_stats["startup_archive_size"], 4)
        _assert_identities(self, algorithm.last_run_stats)

    def test_target_budget_and_opposing_incumbents(self) -> None:
        records = {
            (0,): _record(y_p=10.0, y=1.0, violation=0.1, feasible=False),
            (1,): _record(y_p=1.0, y=8.0, violation=0.5, feasible=False),
        }
        problem = _DebProblem(records, lower=[0], upper=[1])
        manager, _controller = _manager(problem, budget=10)
        stats: dict[str, Any] = {
            "n_start_evaluations": 0,
            "n_initial_start_evaluations": 0,
            "n_main_start_evaluations": 0,
            "n_local_searches": 0,
            "n_local_search_evaluations": 0,
            "n_initial_local_searches": 0,
            "n_initial_local_search_evaluations": 0,
            "n_main_local_search_evaluations": 0,
            "n_certification_evaluations": 0,
            "n_certified_local_searches": 0,
            "n_uncertified_local_searches": 0,
        }
        book = DebArchiveBook()
        initial_local_searches_deb(
            book,
            manager,
            DiscreteSearchSpace([0], [1]),
            ScriptedRng(integers=[[1]]),
            1,
            2,
            stats,
        )
        self.assertEqual(book.members[0].point.x, (0,))
        self.assertEqual(manager.best_deb_point.x, (0,))
        self.assertEqual(manager.best_benchmark_point.x, (1,))
        self.assertEqual(
            (
                manager.best_deb_point.x,
                manager.best_deb_point.y,
                manager.best_deb_point.phi,
                manager.best_deb_point.feasible,
            ),
            ((0,), 1.0, 0.1, False),
        )

        hit_problem = _DebProblem(
            {(0,): _record(y_p=0.2, y=0.2), (1,): _record(y_p=5.0, y=5.0)},
            lower=[0],
            upper=[1],
        )
        algorithm = KEliteAMSDeb(
            max_evaluations=8,
            seed=0,
            n_parents=1,
            archive_size=1,
        )
        controller = RunController(
            hit_problem,
            8,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        with patch(
            "numpy.random.default_rng",
            return_value=ScriptedRng(integers=[[0]]),
        ):
            algorithm(hit_problem, run_controller=controller)
        self.assertEqual(hit_problem.calls, [(0,)])
        self.assertEqual(algorithm.last_run_stats["n_local_searches"], 0)
        self.assertEqual(algorithm.last_run_stats["final_archive"], [])
        _assert_identities(self, algorithm.last_run_stats)

    def test_duplicate_proposal_and_distinct_starts_do_not_use_the_cap(self) -> None:
        only = _DebProblem(
            {(0,): _record(y_p=1.0, y=1.0)},
            lower=[0],
            upper=[0],
        )
        algorithm = KEliteAMSDeb(
            max_evaluations=4,
            seed=1,
            n_parents=1,
            archive_size=1,
            random_restart_probability=1.0,
        )
        algorithm(only)
        self.assertGreater(len(only.calls), 1)
        self.assertGreaterEqual(algorithm.last_run_stats["n_archive_duplicates"], 1)
        self.assertEqual(algorithm.last_run_stats["final_archive_size"], 1)
        _assert_identities(self, algorithm.last_run_stats)

        basin = {
            (index,): _record(y_p=float(index + 1), y=float(index + 1))
            for index in range(30)
        }
        basin[(0,)] = _record(y_p=0.5, y=0.5)
        converging = _DebProblem(basin, lower=[0], upper=[29])
        manager, _controller = _manager(converging, budget=18)
        stats = {name: 0 for name in (
            "n_start_evaluations",
            "n_initial_start_evaluations",
            "n_main_start_evaluations",
            "n_local_searches",
            "n_local_search_evaluations",
            "n_initial_local_searches",
            "n_initial_local_search_evaluations",
            "n_main_local_search_evaluations",
            "n_certification_evaluations",
            "n_certified_local_searches",
            "n_uncertified_local_searches",
        )}
        book = DebArchiveBook()
        initial_local_searches_deb(
            book,
            manager,
            DiscreteSearchSpace([0], [29]),
            ScriptedRng(integers=[[index] for index in range(2, 20)]),
            2,
            4,
            stats,
        )
        self.assertEqual(len(book.members), 1)
        self.assertEqual(book.members[0].point.x, (0,))
        self.assertGreaterEqual(stats["n_local_searches"], 2)
        self.assertLess(stats["n_local_searches"], _CONSECUTIVE_DUPLICATE_START_CAP)
        self.assertEqual(manager.n_evaluations, 18)

    def test_scripted_duplicate_starts_hit_the_existing_cap(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(y_p=1.0, y=1.0),
                (1,): _record(y_p=2.0, y=2.0),
            },
            lower=[0],
            upper=[1],
        )
        manager, _controller = _manager(problem, budget=5)
        stats = {name: 0 for name in (
            "n_start_evaluations",
            "n_initial_start_evaluations",
            "n_main_start_evaluations",
            "n_local_searches",
            "n_local_search_evaluations",
            "n_initial_local_searches",
            "n_initial_local_search_evaluations",
            "n_main_local_search_evaluations",
            "n_certification_evaluations",
            "n_certified_local_searches",
            "n_uncertified_local_searches",
        )}
        book = DebArchiveBook()
        draws = [[0]] + [[0]] * _CONSECUTIVE_DUPLICATE_START_CAP
        initial_local_searches_deb(
            book,
            manager,
            DiscreteSearchSpace([0], [1]),
            ScriptedRng(integers=draws),
            2,
            2,
            stats,
        )
        self.assertEqual(len(book.members), 1)
        self.assertLess(manager.n_evaluations, 5)

        full = _DebProblem({(0,): _record(y_p=1.0, y=1.0)}, lower=[0], upper=[0])
        capped = KEliteAMSDeb(
            max_evaluations=20,
            seed=1,
            n_parents=2,
            archive_size=2,
        )
        capped(full)
        self.assertEqual(capped.last_run_stats["n_local_searches"], 1)
        self.assertEqual(capped.last_run_stats["initial_archive_size"], 0)
        self.assertEqual(capped.last_run_stats["startup_archive_size"], 1)

    def test_full_budget_is_consumed_when_the_loop_runs(self) -> None:
        records = {
            (index,): _record(y_p=float(index), y=float(index))
            for index in range(11)
        }
        # Neighbours are not strictly better when every raw value is equal.
        for index in range(11):
            records[(index,)] = _record(y_p=1.0, y=1.0)
        problem = _DebProblem(records, lower=[0], upper=[10])
        algorithm = KEliteAMSDeb(
            max_evaluations=40,
            seed=3,
            n_parents=2,
            archive_size=4,
            mutation_probability=0.0,
            random_restart_probability=1.0,
        )
        algorithm(problem)
        stats = algorithm.last_run_stats
        self.assertGreaterEqual(stats["initial_archive_size"], 2)
        self.assertEqual(stats["n_evaluations"], 40)
        _assert_identities(self, stats)
        self.assertLessEqual(stats["final_archive_size"], 4)

    def test_validator_tokens_differ(self) -> None:
        parameters = {
            "archive_size": 2,
            "n_parents": 3,
            "mutation_probability": 0.05,
            "random_restart_probability": 0.1,
        }
        with self.assertRaisesRegex(RegistryError, "K_ELITE_AMS_PEN_SI_DESC_COLD_START:"):
            validate_k_elite_ams_parameters(parameters)
        with self.assertRaisesRegex(RegistryError, "K_ELITE_AMS_DEB_SI_DESC_COLD_START:"):
            validate_k_elite_ams_deb_parameters(parameters)


class TestWarmLifecycle(unittest.TestCase):
    def _algorithm(
        self,
        path: str,
        *,
        budget: int = 6,
        n_parents: int = 2,
        archive_size: int = 4,
    ) -> KEliteAMSDebWarmStart:
        return KEliteAMSDebWarmStart(
            max_evaluations=budget,
            seed=1,
            n_parents=n_parents,
            archive_size=archive_size,
            initial_design_path=path,
            mutation_probability=0.0,
            random_restart_probability=1.0,
        )

    def test_bad_designs_fail_before_evaluation(self) -> None:
        problem = _DebProblem({}, lower=[0] * 9, upper=[2] * 9)
        with TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.json")
            with self.assertRaises(ValueError):
                self._algorithm(missing)(problem)
            self.assertEqual(problem.calls, [])

            short = Path(tmp) / "short.json"
            _design(short, [[0] * 9])
            with self.assertRaisesRegex(ValueError, "needs 2"):
                self._algorithm(str(short))(problem)
            self.assertEqual(problem.calls, [])

            schema = Path(tmp) / "schema.json"
            _design(schema, [[0] * 9, [1, 0, 0, 0, 0, 0, 0, 0, 0]], schema=2)
            with self.assertRaises(ValueError):
                self._algorithm(str(schema))(problem)

            bounds = Path(tmp) / "bounds.json"
            _design(bounds, [[0] * 9, [9, 0, 0, 0, 0, 0, 0, 0, 0]])
            with self.assertRaises(ValueError):
                self._algorithm(str(bounds))(problem)

            budget = Path(tmp) / "budget.json"
            _design(budget, [[0] * 9, [1, 0, 0, 0, 0, 0, 0, 0, 0]])
            with self.assertRaisesRegex(ValueError, "initial evaluations"):
                self._algorithm(str(budget), budget=1)(problem)
            self.assertEqual(problem.calls, [])

    def test_prefix_keeps_file_order_and_stops_on_target(self) -> None:
        def vector(index: int) -> tuple[int, ...]:
            return (index, 0, 0, 0, 0, 0, 0, 0, 0)

        records = {
            vector(0): _record(y_p=5.0, y=1.0, violation=0.4, feasible=False),
            vector(1): _record(y_p=4.0, y=2.0),
            vector(2): _record(y_p=0.1, y=0.1),
        }
        problem = _DebProblem(records, lower=[0] * 9, upper=[2] * 9)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            _design(path, [list(vector(0)), list(vector(1)), list(vector(2))])
            algorithm = self._algorithm(str(path), budget=2, n_parents=2, archive_size=8)
            counted = ScriptedRng()
            with patch("numpy.random.default_rng", return_value=counted):
                algorithm(problem)
            self.assertEqual(problem.calls, [vector(0), vector(1)])
            self.assertEqual(counted.draws, 0)
            self.assertEqual(algorithm.last_run_stats["final_archive"], [])

            hit = _DebProblem(records, lower=[0] * 9, upper=[2] * 9)
            controller = RunController(
                hit,
                6,
                mode="fixed_target",
                target_threshold=1.0,
                target_reference="all",
            )
            warm = self._algorithm(str(path), budget=6, n_parents=3)
            warm(hit, run_controller=controller)
            self.assertEqual(hit.calls, [vector(0), vector(1), vector(2)])
            self.assertEqual(warm.last_run_stats["n_local_searches"], 0)
            self.assertEqual(warm.last_run_stats["final_archive_size"], 0)

    def test_duplicate_prefix_and_cold_fallback(self) -> None:
        zero = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        problem = _DebProblem({}, lower=[0] * 9, upper=[1, 0, 0, 0, 0, 0, 0, 0, 0])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            _design(path, [list(zero), list(zero), [1, 0, 0, 0, 0, 0, 0, 0, 0]])
            algorithm = self._algorithm(str(path), budget=12, n_parents=2, archive_size=4)
            algorithm(problem)
            self.assertEqual(problem.calls[0], zero)
            self.assertEqual(problem.calls[1], zero)
            self.assertGreaterEqual(
                algorithm.last_run_stats["n_archive_duplicates"],
                1,
            )
            self.assertGreaterEqual(algorithm.last_run_stats["startup_archive_size"], 1)
            _assert_identities(self, algorithm.last_run_stats)

            short = _DebProblem({}, lower=[0] * 9, upper=[1, 0, 0, 0, 0, 0, 0, 0, 0])
            one = Path(tmp) / "one.json"
            _design(one, [list(zero), list(zero)])
            fallback = self._algorithm(str(one), budget=30, n_parents=2, archive_size=4)
            fallback(short)
            self.assertGreaterEqual(fallback.last_run_stats["n_initial_local_searches"], 3)
            self.assertGreaterEqual(fallback.last_run_stats["startup_archive_size"], 2)


class TestHpSlabSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = build_problems_for_slab_type("hp_slab")["L6750_B2"]

    def _run(self, token: str, budget: int, seed: int, *, design: str | None = None) -> Any:
        problem = self.bundle["problem"]
        context = self.bundle["ctx"]
        problem.reset()
        context.reset()
        attach_eval_context(problem, context)
        parameters: dict[str, Any] = {
            "max_evaluations": budget,
            "n_parents": 2,
            "archive_size": 4,
        }
        if design is not None:
            parameters["initial_design_path"] = design
        algorithm = create_algorithm(token, parameters, random_seed=seed)
        algorithm(problem)
        return algorithm

    def test_budgets_partitions_and_active_constraints(self) -> None:
        names = list(self.bundle["active_constraint_names"])
        self.assertEqual(len(names), 11)
        self.assertIn("Z1_nt_dt_combination_capacity", names)
        self.assertIn("Z2_beam_theory_H_L_capacity", names)
        self.assertIn("Z3_beam_theory_B_L_capacity", names)
        origin = [int(value) for value in self.bundle["lb_idx"]]
        with TemporaryDirectory() as tmp:
            design = Path(tmp) / "initial_design.json"
            design.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "problem_id": self.bundle["ctx"].problem_id,
                        "n": 2,
                        "vectors": [origin, origin],
                    }
                ),
                encoding="utf-8",
            )
            for token in ("K_ELITE_AMS_DEB_SI_DESC_COLD_START", "K_ELITE_AMS_DEB_SI_DESC_WARM_START"):
                for budget in (100, 250, 500):
                    with self.subTest(token=token, budget=budget):
                        algorithm = self._run(
                            token,
                            budget,
                            seed=11,
                            design=None if token == "K_ELITE_AMS_DEB_SI_DESC_COLD_START" else str(design),
                        )
                        stats = algorithm.last_run_stats
                        diagnostics = algorithm.summary_diagnostics()
                        self.assertGreaterEqual(stats["n_evaluations"], 0)
                        self.assertLessEqual(stats["n_evaluations"], budget)
                        _assert_identities(self, stats)
                        self.assertLessEqual(stats["final_archive_size"], 4)
                        self.assertEqual(
                            set(diagnostics["sentinel_evaluations_by_constraint"]),
                            set(names),
                        )
                        self.assertNotIn("final_archive", diagnostics)
                        json.dumps(diagnostics)

        replay_a = self._run("K_ELITE_AMS_DEB_SI_DESC_COLD_START", 100, seed=11)
        replay_b = self._run("K_ELITE_AMS_DEB_SI_DESC_COLD_START", 100, seed=11)
        self.assertEqual(
            replay_a.summary_diagnostics(),
            replay_b.summary_diagnostics(),
        )


class TestRegistryMetadata(unittest.TestCase):
    def test_deb_specs_match_historical_shape(self) -> None:
        cold = get_algorithm_spec("K_ELITE_AMS_PEN_SI_DESC_COLD_START")
        deb = get_algorithm_spec("K_ELITE_AMS_DEB_SI_DESC_COLD_START")
        warm = get_algorithm_spec("K_ELITE_AMS_DEB_SI_DESC_WARM_START")
        self.assertEqual(deb.seed_policy, cold.seed_policy)
        self.assertEqual(deb.candidate_mode, cold.candidate_mode)
        self.assertFalse(deb.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in deb.parameters],
            [parameter.name for parameter in cold.parameters],
        )
        self.assertEqual(
            [parameter.name for parameter in warm.parameters][-1],
            "initial_design_path",
        )
        created = create_algorithm(
            "K_ELITE_AMS_DEB_SI_DESC_COLD_START",
            {"max_evaluations": 4},
            random_seed=1,
        )
        self.assertTrue(created.info.startswith("k-elite AMS-Deb "))
        self.assertEqual(created.name, "K_ELITE_AMS_DEB_SI_DESC_COLD_START")


if __name__ == "__main__":
    unittest.main()
