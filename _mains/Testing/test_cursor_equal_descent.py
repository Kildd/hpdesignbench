"""Equal descent and the six penalty PBH, AMS, and MLSL algorithms."""
from __future__ import annotations

import inspect
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from core.ioh_core.algorithm_factory import create_algorithm
from core.ioh_core.algorithms.common import RunController
from core.ioh_core.algorithms.native.discrete_search import (
    DiscreteMLSL,
    DiscreteMLSLDeb,
    DiscreteMLSLEquDesc,
    DiscreteMLSLEquDescWarmStart,
    DiscreteSampler,
    DiscreteSearchSpace,
    EvaluatedPoint,
    EvaluationManager,
    PopulationBasinHopping,
    PopulationBasinHoppingDeb,
    PopulationBasinHoppingEquDesc,
    PopulationBasinHoppingEquDescWarmStart,
    best_improvement_local_search,
    deb_best_improvement_local_search,
    equal_descent_local_search,
    pbh_deb_best_improvement_local_search,
    run_discrete_mlsl_deb,
)
from core.ioh_core.algorithms.native.k_elite_ams import (
    KEliteAMS,
    KEliteAMSDeb,
    KEliteAMSEquDesc,
    KEliteAMSEquDescWarmStart,
    _uniform_index,
    continue_after_initialisation_deb,
    initial_local_searches_deb,
    run_k_elite_loop_deb,
    search_ranked_prefix_deb,
)
from core.ioh_core.algorithms.registry import ALGORITHM_REGISTRY, get_algorithm_spec
from core.ioh_core.algorithms.warm_start import (
    DEB_WARM_START_TOKENS,
    INITIAL_DESIGN_PATH_TOKENS,
    WARM_START_TOKENS,
)
from core.visualization_core.optimization.algorithm_presentation import (
    legend_handling,
    local_search_label,
    presentation_for,
)
from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

_EQU_TOKENS = (
    "POPULATION_BASIN_HOPPING_PEN_EQU_DESC_COLD_START",
    "POPULATION_BASIN_HOPPING_PEN_EQU_DESC_WARM_START",
    "K_ELITE_AMS_PEN_EQU_DESC_COLD_START",
    "K_ELITE_AMS_PEN_EQU_DESC_WARM_START",
    "DISCRETE_MLSL_PEN_EQU_DESC_COLD_START",
    "DISCRETE_MLSL_PEN_EQU_DESC_WARM_START",
)
_DEB_EQU_TOKENS = (
    "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_COLD_START",
    "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START",
    "K_ELITE_AMS_DEB_EQU_DESC_COLD_START",
    "K_ELITE_AMS_DEB_EQU_DESC_WARM_START",
    "DISCRETE_MLSL_DEB_EQU_DESC_COLD_START",
    "DISCRETE_MLSL_DEB_EQU_DESC_WARM_START",
)


class MapProblem(FakeIOHProblem):
    """Integer objective keyed by index vector. Missing points use the default."""

    def __init__(
        self,
        lower: list[int],
        upper: list[int],
        values: dict[tuple[int, ...], float],
        default: float = 100.0,
    ) -> None:
        super().__init__(lower, upper)
        self.values = values
        self.default = default

    def __call__(self, x: object) -> float:
        candidate = tuple(int(value) for value in list(x))
        self.calls.append(list(candidate))
        self.state.evaluations += 1
        self.hdb_last_feasible = True
        value = float(self.values.get(candidate, self.default))
        if math.isfinite(value) and value < self.state.current_best.y:
            from _mains.Testing.helpers.fake_ioh_problem import FakeBest

            self.state.current_best = FakeBest(x=list(candidate), y=value)
        return value


def _search(values, start, budget, function, lower, upper):
    """Run one local search. The start is already evaluated and is not a call."""
    problem = MapProblem(lower, upper, values)
    controller = RunController(problem, budget)
    space = DiscreteSearchSpace(problem.bounds.lb, problem.bounds.ub)
    manager = EvaluationManager(problem, controller, space)
    result = function(
        EvaluatedPoint(tuple(start), float(values[tuple(start)])),
        manager,
        space,
    )
    return problem, result


class TestEqualDescentExamples(unittest.TestCase):
    """Worked examples from the approved implementation plan."""

    def test_example_7_1_scans_both_equal_exits(self) -> None:
        values = {(1, 1): 10.0, (0, 1): 8.0, (2, 1): 8.0, (1, 0): 9.0, (1, 2): 11.0}
        bounds = ([0, 0], [3, 3])
        strict_problem, strict = _search(
            values, (1, 1), 100, best_improvement_local_search, *bounds
        )
        equal_problem, equal = _search(
            values, (1, 1), 100, equal_descent_local_search, *bounds
        )
        self.assertEqual(strict.point.x, (0, 1))
        self.assertEqual(equal.point.x, (0, 1))
        self.assertTrue(strict.certified_local_optimum)
        self.assertTrue(equal.certified_local_optimum)
        self.assertEqual(strict.n_steps, 1)
        self.assertEqual(equal.n_steps, 1)
        extra = {(3, 1), (2, 0), (2, 2)}
        strict_calls = {tuple(call) for call in strict_problem.calls}
        equal_calls = {tuple(call) for call in equal_problem.calls}
        self.assertTrue(extra.isdisjoint(strict_calls))
        self.assertTrue(extra <= equal_calls)
        self.assertFalse(getattr(equal, "early_certificate", False))

    def test_example_7_2_follows_the_strictly_better_line(self) -> None:
        values = {
            (1, 1): 10.0,
            (0, 1): 8.0,
            (2, 1): 8.0,
            (1, 0): 9.0,
            (1, 2): 11.0,
            (2, 2): 7.0,
        }
        bounds = ([0, 0], [3, 3])
        _strict_problem, strict = _search(
            values, (1, 1), 100, best_improvement_local_search, *bounds
        )
        equal_problem, equal = _search(
            values, (1, 1), 100, equal_descent_local_search, *bounds
        )
        self.assertEqual(strict.point.x, (0, 1))
        self.assertEqual(equal.point.x, (2, 2))
        self.assertTrue(equal.certified_local_optimum)
        self.assertEqual(equal.n_steps, 2)
        self.assertIn([2, 2], equal_problem.calls)

    def test_example_7_3_keeps_the_arrival_design(self) -> None:
        values = {(2, 1): 8.0, (1, 1): 8.0}
        bounds = ([0, 0], [3, 2])
        _strict_problem, strict = _search(
            values, (2, 1), 100, best_improvement_local_search, *bounds
        )
        _equal_problem, equal = _search(
            values, (2, 1), 100, equal_descent_local_search, *bounds
        )
        self.assertEqual(strict.point.x, (2, 1))
        self.assertEqual(equal.point.x, (2, 1))
        self.assertTrue(equal.certified_local_optimum)
        self.assertEqual(equal.n_steps, 0)

    def test_example_7_4_drops_the_worse_line_before_its_terminal(self) -> None:
        values = {
            (2, 2): 10.0,
            (1, 2): 8.0,
            (3, 2): 8.0,
            (1, 1): 5.0,
            (3, 1): 5.0,
            (1, 0): 3.0,
            (3, 0): 4.0,
            (4, 0): 1.0,
        }
        _problem, result = _search(
            values,
            (2, 2),
            100,
            equal_descent_local_search,
            [0, 0],
            [4, 4],
        )
        self.assertEqual(result.point.x, (1, 0))
        self.assertTrue(result.certified_local_optimum)
        self.assertEqual(result.n_steps, 3)
        self.assertIn([3, 0], _problem.calls)
        self.assertNotIn([4, 0], _problem.calls)

    def test_example_7_5_budget_cases(self) -> None:
        bounds = ([0, 0], [3, 3])
        tied = {(1, 1): 10.0, (0, 1): 8.0, (2, 1): 8.0, (1, 0): 9.0, (1, 2): 11.0}
        for function in (best_improvement_local_search, equal_descent_local_search):
            _problem, stopped = _search(tied, (1, 1), 1, function, *bounds)
            self.assertEqual(stopped.point.x, (0, 1))
            self.assertFalse(stopped.certified_local_optimum)
            self.assertEqual(stopped.n_steps, 1)

        plateau = {(2, 1): 8.0, (1, 1): 8.0}
        for function in (best_improvement_local_search, equal_descent_local_search):
            _problem, stopped = _search(plateau, (2, 1), 1, function, *bounds)
            self.assertEqual(stopped.point.x, (2, 1))
            self.assertFalse(stopped.certified_local_optimum)
            self.assertEqual(stopped.n_steps, 0)
            self.assertFalse(getattr(stopped, "early_certificate", False))

        _problem, early = _search(
            plateau, (2, 1), 5, equal_descent_local_search, *bounds
        )
        self.assertEqual(early.point.x, (2, 1))
        self.assertTrue(early.certified_local_optimum)
        self.assertEqual(early.n_steps, 0)
        self.assertTrue(early.early_certificate)
        self.assertEqual(early.n_evaluations, 5)

        partial_exit = {(2, 1): 8.0, (1, 1): 8.0, (0, 1): 7.0}
        _problem, held = _search(
            partial_exit, (2, 1), 5, equal_descent_local_search, *bounds
        )
        self.assertEqual(held.point.x, (2, 1))
        self.assertFalse(held.certified_local_optimum)
        self.assertEqual(held.n_steps, 0)
        self.assertFalse(getattr(held, "early_certificate", False))

        strict_during_first = {(2, 1): 8.0, (1, 1): 7.0}
        for function in (best_improvement_local_search, equal_descent_local_search):
            _problem, moved = _search(
                strict_during_first, (2, 1), 1, function, *bounds
            )
            self.assertEqual(moved.point.x, (1, 1))
            self.assertFalse(moved.certified_local_optimum)
            self.assertEqual(moved.n_steps, 1)

        later = {
            (1, 1): 10.0,
            (0, 1): 8.0,
            (2, 1): 8.0,
            (1, 0): 9.0,
            (1, 2): 11.0,
            (2, 2): 7.0,
            (3, 2): 9.0,
        }
        _problem, stayed = _search(
            later, (1, 1), 10, equal_descent_local_search, *bounds
        )
        self.assertEqual(stayed.point.x, (2, 2))
        self.assertNotEqual(stayed.point.x, (0, 1))
        self.assertFalse(stayed.certified_local_optimum)
        self.assertEqual(stayed.n_steps, 2)

        later_better = dict(later)
        later_better[(3, 2)] = 6.0
        _problem, moved_later = _search(
            later_better, (1, 1), 10, equal_descent_local_search, *bounds
        )
        self.assertEqual(moved_later.point.x, (3, 2))
        self.assertFalse(moved_later.certified_local_optimum)
        self.assertEqual(moved_later.n_steps, 3)


class TestEqualDescentOperator(unittest.TestCase):
    """Plateau, cycle, non-finite, and cache behaviour."""

    def test_unique_best_matches_strict_improvement(self) -> None:
        values = {
            (1, 1): 10.0,
            (0, 1): 8.0,
            (2, 1): 9.0,
            (1, 0): 9.0,
            (1, 2): 11.0,
            (0, 0): 6.0,
            (0, 2): 9.0,
        }
        bounds = ([0, 0], [2, 2])
        strict_problem, strict = _search(
            values, (1, 1), 100, best_improvement_local_search, *bounds
        )
        equal_problem, equal = _search(
            values, (1, 1), 100, equal_descent_local_search, *bounds
        )
        self.assertEqual(strict_problem.calls, equal_problem.calls)
        self.assertEqual(strict.point.x, equal.point.x)
        self.assertEqual(strict.point.x, (0, 0))
        self.assertEqual(strict.certified_local_optimum, equal.certified_local_optimum)
        self.assertEqual(strict.n_steps, equal.n_steps)
        self.assertEqual(strict.n_evaluations, equal.n_evaluations)
        self.assertFalse(getattr(equal, "early_certificate", False))

    def test_multi_step_plateau_reaches_the_exit_two_steps_away(self) -> None:
        values = {(1, 1): 5.0, (1, 2): 5.0, (1, 3): 4.0}
        problem, result = _search(
            values, (1, 1), 100, equal_descent_local_search, [0, 0], [2, 4]
        )
        self.assertEqual(result.point.x, (1, 3))
        self.assertTrue(result.certified_local_optimum)
        self.assertIn([1, 2], problem.calls)
        self.assertEqual(result.n_steps, 1)

    def test_two_exits_from_different_plateau_members_both_stay(self) -> None:
        values = {(1, 1): 5.0, (2, 1): 5.0, (1, 0): 3.0, (2, 0): 3.0}
        problem, result = _search(
            values, (1, 1), 100, equal_descent_local_search, [0, 0], [3, 2]
        )
        self.assertIn([2, 0], problem.calls)
        self.assertEqual(result.point.x, (1, 0))
        self.assertTrue(result.certified_local_optimum)

    def test_equal_component_is_evaluated_once_per_call(self) -> None:
        values = {(1, 1): 5.0, (2, 1): 5.0}
        problem = MapProblem([0, 0], [3, 2], values)
        space = DiscreteSearchSpace(problem.bounds.lb, problem.bounds.ub)
        for _ in range(2):
            controller = RunController(problem, 100)
            manager = EvaluationManager(problem, controller, space)
            equal_descent_local_search(
                EvaluatedPoint((1, 1), 5.0),
                manager,
                space,
            )
        self.assertEqual(problem.calls.count([2, 1]), 2)
        self.assertEqual(problem.calls[: problem.calls.index([2, 1]) + 1].count([2, 1]), 1)

    def test_non_finite_values_do_not_join_a_component(self) -> None:
        values = {
            (1, 1): 5.0,
            (0, 1): math.inf,
            (2, 1): 4.0,
            (0, 0): 0.0,
            (0, 2): 0.0,
        }
        problem, result = _search(
            values, (1, 1), 100, equal_descent_local_search, [0, 0], [3, 3]
        )
        self.assertEqual(result.point.x, (2, 1))
        self.assertIn([0, 1], problem.calls)
        self.assertNotIn([0, 0], problem.calls)
        self.assertNotIn([0, 2], problem.calls)

        inf_start = {
            (1, 1): math.inf,
            (0, 1): 3.0,
            (2, 1): math.inf,
            (1, 0): math.inf,
            (1, 2): math.inf,
            (2, 0): 1.0,
        }
        problem, result = _search(
            inf_start, (1, 1), 100, equal_descent_local_search, [0, 0], [3, 3]
        )
        self.assertEqual(result.point.x, (0, 1))
        self.assertTrue(result.certified_local_optimum)
        self.assertNotIn([2, 0], problem.calls)

        only_inf = {
            (1, 1): math.inf,
            (0, 1): math.inf,
            (2, 1): math.inf,
            (1, 0): math.inf,
            (1, 2): math.inf,
            (2, 0): 1.0,
        }
        problem, result = _search(
            only_inf, (1, 1), 100, equal_descent_local_search, [0, 0], [3, 3]
        )
        self.assertEqual(result.point.x, (1, 1))
        self.assertTrue(result.certified_local_optimum)
        self.assertEqual(result.n_steps, 0)
        self.assertNotIn([2, 0], problem.calls)

    def test_target_hit_stops_the_search(self) -> None:
        values = {(1, 1): 10.0, (0, 1): 8.0, (2, 1): 9.0, (1, 0): 9.0, (1, 2): 9.0}
        problem = MapProblem([0, 0], [2, 2], values)
        controller = RunController(
            problem,
            20,
            mode="fixed_target",
            target_threshold=8.0,
            target_reference="all",
        )
        space = DiscreteSearchSpace(problem.bounds.lb, problem.bounds.ub)
        manager = EvaluationManager(problem, controller, space)
        result = equal_descent_local_search(
            EvaluatedPoint((1, 1), 10.0),
            manager,
            space,
        )
        self.assertTrue(controller.target_hit)
        self.assertEqual(result.point.x, (0, 1))
        self.assertFalse(result.certified_local_optimum)
        self.assertEqual(problem.calls, [[0, 1]])

    def test_empty_neighbourhood_certifies_the_start(self) -> None:
        problem, result = _search(
            {(0,): 4.0}, (0,), 10, equal_descent_local_search, [0], [0]
        )
        self.assertEqual(result.point.x, (0,))
        self.assertTrue(result.certified_local_optimum)
        self.assertEqual(result.n_steps, 0)
        self.assertEqual(result.n_evaluations, 0)
        self.assertEqual(problem.calls, [])

    def test_interrupted_level_does_not_adopt_another_members_exit(self) -> None:
        values = {
            (1, 1): 10.0,
            (0, 1): 8.0,
            (2, 1): 8.0,
            (1, 0): 9.0,
            (1, 2): 9.0,
            (0, 0): 6.0,
            (0, 2): 9.0,
            (3, 1): 7.0,
        }
        _problem, result = _search(
            values, (1, 1), 7, equal_descent_local_search, [0, 0], [3, 2]
        )
        self.assertEqual(result.point.x, (3, 1))
        self.assertNotEqual(result.point.x, (0, 0))
        self.assertNotEqual(result.point.x, (1, 1))
        self.assertFalse(result.certified_local_optimum)
        self.assertEqual(result.n_steps, 2)


def _fork_values(space: DiscreteSearchSpace, start: tuple[int, ...]):
    """Two equal-best exits, plus a neighbour only the second exit can open."""
    neighbours = space.neighbours_l1(start)
    first, second = neighbours[0], neighbours[1]
    private = next(
        candidate
        for candidate in space.neighbours_l1(second)
        if candidate != start and candidate not in neighbours
    )
    values = {start: 10.0, first: 8.0, second: 8.0, private: 9.0}
    return values, private


def _measured_budget(space, start, values, lower, upper) -> int:
    _problem, result = _search(
        values, start, 1000, equal_descent_local_search, lower, upper
    )
    return 1 + result.n_evaluations


def _write_prefix(path: Path, vectors: list[list[int]]) -> None:
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


class TestEqualDescentAlgorithms(unittest.TestCase):
    """Registry, presentation, and outer-loop wiring."""

    def test_registry_presentation_and_warm_start_membership(self) -> None:
        names = ALGORITHM_REGISTRY.names()
        for token in _EQU_TOKENS:
            self.assertIn(token, names)
            spec = get_algorithm_spec(token)
            self.assertEqual(spec.library, "native")
            self.assertFalse(spec.batch_evaluations)
            presented = presentation_for(token)
            self.assertEqual(presented.qualifier, "EQU-DESC")
            self.assertEqual(local_search_label(presented), "Equal descent")
            self.assertEqual(
                legend_handling(presented),
                "Penalty-based, Equal descent",
            )
            self.assertFalse(presented.constraint_aware)
            self.assertEqual(
                presented.family,
                token.removesuffix("_WARM_START").removesuffix("_COLD_START")
                + "_COLD_START",
            )
        for token in _DEB_EQU_TOKENS:
            self.assertIn(token, names)
        for stem in (
            "POPULATION_BASIN_HOPPING_PEN_SI_DESC",
            "K_ELITE_AMS_PEN_SI_DESC",
            "DISCRETE_MLSL_PEN_SI_DESC",
        ):
            penalty = presentation_for(f"{stem}_COLD_START")
            self.assertEqual(penalty.qualifier, "SI-DESC")
            self.assertEqual(local_search_label(penalty), "Strict improvement")
            self.assertEqual(
                legend_handling(penalty),
                "Penalty-based, Strict improvement",
            )
            deb_stem = stem.replace("_PEN_", "_DEB_")
            deb = presentation_for(f"{deb_stem}_COLD_START")
            self.assertEqual(deb.qualifier, "")
            self.assertEqual(local_search_label(deb), "Strict improvement")
            self.assertEqual(
                legend_handling(deb),
                "Constraint-aware, Strict improvement",
            )
        warm = {token for token in _EQU_TOKENS if token.endswith("_WARM_START")}
        self.assertTrue(warm <= WARM_START_TOKENS)
        self.assertTrue(warm <= INITIAL_DESIGN_PATH_TOKENS)
        self.assertTrue(warm.isdisjoint(DEB_WARM_START_TOKENS))

    def test_classes_use_equal_descent_and_the_info_prefix(self) -> None:
        cold = (
            (
                PopulationBasinHoppingEquDesc(5, seed=1, population_size=1),
                "PBH-L1-EquDesc",
            ),
            (
                KEliteAMSEquDesc(5, seed=1, archive_size=1, n_parents=1),
                "k-elite AMS-EquDesc",
            ),
            (DiscreteMLSLEquDesc(5, seed=1, batch_size=1), "dMLSL-L1-EquDesc-Sobol"),
            (
                DiscreteMLSLEquDesc(5, seed=1, batch_size=1, sampler="uniform"),
                "dMLSL-L1-EquDesc-uniform",
            ),
        )
        for algorithm, prefix in cold:
            self.assertIs(algorithm.local_search, equal_descent_local_search)
            self.assertTrue(algorithm.info.startswith(prefix))
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "initial_design.json")
            warm = (
                PopulationBasinHoppingEquDescWarmStart(
                    5,
                    seed=1,
                    population_size=1,
                    initial_design_path=path,
                ),
                KEliteAMSEquDescWarmStart(
                    5,
                    seed=1,
                    archive_size=1,
                    n_parents=1,
                    initial_design_path=path,
                ),
                DiscreteMLSLEquDescWarmStart(
                    5,
                    seed=1,
                    batch_size=1,
                    initial_design_path=path,
                ),
            )
            for algorithm in warm:
                self.assertIs(algorithm.local_search, equal_descent_local_search)
                self.assertTrue(algorithm.name.endswith("_WARM_START"))

    def test_outer_loops_evaluate_the_second_branch(self) -> None:
        lower = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        upper = [2, 2, 0, 0, 0, 0, 0, 0, 0]
        space = DiscreteSearchSpace(lower, upper)
        warm_start = (1, 1, 0, 0, 0, 0, 0, 0, 0)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            _write_prefix(path, [list(warm_start)])
            cases = []
            pbh_start = DiscreteSampler(
                space, "uniform", 7, np.random.default_rng(7)
            ).draw()
            ams_start = _uniform_index(space, np.random.default_rng(7))
            mlsl_start = DiscreteSampler(
                space, "uniform", 7, np.random.default_rng(7)
            ).next_unique(set())
            assert mlsl_start is not None
            for start, builder in (
                (
                    pbh_start,
                    lambda budget: PopulationBasinHoppingEquDesc(
                        budget,
                        seed=7,
                        population_size=1,
                        initial_sampler="uniform",
                    ),
                ),
                (
                    ams_start,
                    lambda budget: KEliteAMSEquDesc(
                        budget,
                        seed=7,
                        archive_size=1,
                        n_parents=1,
                    ),
                ),
                (
                    mlsl_start,
                    lambda budget: DiscreteMLSLEquDesc(
                        budget,
                        seed=7,
                        batch_size=1,
                        sampler="uniform",
                    ),
                ),
                (
                    warm_start,
                    lambda budget: PopulationBasinHoppingEquDescWarmStart(
                        budget,
                        seed=7,
                        population_size=1,
                        initial_sampler="uniform",
                        initial_design_path=str(path),
                    ),
                ),
                (
                    warm_start,
                    lambda budget: KEliteAMSEquDescWarmStart(
                        budget,
                        seed=7,
                        archive_size=1,
                        n_parents=1,
                        initial_design_path=str(path),
                    ),
                ),
                (
                    warm_start,
                    lambda budget: DiscreteMLSLEquDescWarmStart(
                        budget,
                        seed=7,
                        batch_size=1,
                        sampler="uniform",
                        initial_design_path=str(path),
                    ),
                ),
            ):
                values, private = _fork_values(space, start)
                budget = _measured_budget(space, start, values, lower, upper)
                algorithm = builder(budget)
                problem = MapProblem(lower, upper, values)
                algorithm(problem)
                self.assertIs(algorithm.local_search, equal_descent_local_search)
                self.assertIn(list(private), problem.calls)
                self.assertNotIn(
                    "n_equal_descent_early_certificates",
                    algorithm.last_run_stats,
                )
                cases.append(algorithm.name)
            self.assertEqual(len(cases), 6)

    def test_early_certificate_is_counted_only_for_equal_descent(self) -> None:
        from core.ioh_core.algorithms.native.discrete_search import (
            PopulationBasinHopping,
        )

        strict = PopulationBasinHopping(
            6,
            seed=1,
            population_size=1,
            initial_sampler="uniform",
        )
        strict(MapProblem([0, 0], [2, 2], {(0, 0): 1.0}))
        self.assertNotIn(
            "n_equal_descent_early_certificates",
            strict.last_run_stats,
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            start = (2, 1, 0, 0, 0, 0, 0, 0, 0)
            equal = (1, 1, 0, 0, 0, 0, 0, 0, 0)
            _write_prefix(path, [list(start)])
            algorithm = PopulationBasinHoppingEquDescWarmStart(
                6,
                seed=1,
                population_size=1,
                initial_design_path=str(path),
            )
            algorithm(
                MapProblem(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [3, 2, 0, 0, 0, 0, 0, 0, 0],
                    {start: 8.0, equal: 8.0},
                )
            )
        self.assertEqual(
            algorithm.last_run_stats["n_equal_descent_early_certificates"],
            1,
        )

    def test_certified_return_has_no_strictly_better_resolved_neighbour(self) -> None:
        values = {
            (1, 1): 10.0,
            (0, 1): 8.0,
            (2, 1): 8.0,
            (1, 0): 9.0,
            (1, 2): 11.0,
            (2, 2): 7.0,
        }
        bounds = ([0, 0], [3, 3])
        problem, result = _search(
            values, (1, 1), 100, equal_descent_local_search, *bounds
        )
        self.assertTrue(result.certified_local_optimum)
        space = DiscreteSearchSpace(*bounds)
        resolved = {tuple(call) for call in problem.calls}
        resolved.add((1, 1))
        for neighbour in space.neighbours_l1(result.point.x):
            if neighbour not in resolved:
                continue
            value = float(values.get(neighbour, problem.default))
            if math.isfinite(value):
                self.assertGreaterEqual(value, result.point.f)

    def test_strict_classes_match_a_direct_strict_search(self) -> None:
        lower = [0, 0]
        upper = [2, 2]
        space = DiscreteSearchSpace(lower, upper)
        values = {
            (x0, x1): float(x0 + x1) for x0 in range(3) for x1 in range(3)
        }
        starts = {
            "pbh": DiscreteSampler(
                space, "uniform", 7, np.random.default_rng(7)
            ).draw(),
            "ams": _uniform_index(space, np.random.default_rng(7)),
            "mlsl": DiscreteSampler(
                space, "uniform", 7, np.random.default_rng(7)
            ).next_unique(set()),
        }
        assert starts["mlsl"] is not None
        builders = {
            "pbh": lambda budget: PopulationBasinHopping(
                budget,
                seed=7,
                population_size=1,
                initial_sampler="uniform",
            ),
            "ams": lambda budget: KEliteAMS(
                budget,
                seed=7,
                archive_size=1,
                n_parents=1,
            ),
            "mlsl": lambda budget: DiscreteMLSL(
                budget,
                seed=7,
                batch_size=1,
                sampler="uniform",
            ),
        }
        for key, builder in builders.items():
            start = starts[key]
            _direct_problem, direct = _search(
                values, start, 100, best_improvement_local_search, lower, upper
            )
            algorithm = builder(1 + direct.n_evaluations)
            self.assertIs(algorithm.local_search, best_improvement_local_search)
            problem = MapProblem(lower, upper, values)
            algorithm(problem)
            self.assertEqual([tuple(call) for call in problem.calls][0], start)
            self.assertEqual(
                [tuple(call) for call in problem.calls][1:],
                [tuple(call) for call in _direct_problem.calls],
            )
            self.assertEqual(
                tuple(algorithm.last_run_stats["best_x"]),
                direct.point.x,
            )
            self.assertEqual(bool(direct.certified_local_optimum), True)
            self.assertGreaterEqual(
                algorithm.last_run_stats["n_certified_local_searches"],
                1,
            )

    def test_strict_deb_search_stays_the_default(self) -> None:
        self.assertIs(
            pbh_deb_best_improvement_local_search,
            deb_best_improvement_local_search,
        )
        self.assertIs(
            PopulationBasinHoppingDeb(4, seed=1, population_size=1).local_search,
            deb_best_improvement_local_search,
        )
        self.assertIs(
            KEliteAMSDeb(4, seed=1, n_parents=1, archive_size=1).local_search,
            deb_best_improvement_local_search,
        )
        self.assertIs(
            DiscreteMLSLDeb(4, seed=1, batch_size=1).local_search,
            deb_best_improvement_local_search,
        )
        for function in (
            run_discrete_mlsl_deb,
            initial_local_searches_deb,
            run_k_elite_loop_deb,
            search_ranked_prefix_deb,
        ):
            parameter = inspect.signature(function).parameters["local_search"]
            self.assertIsNone(parameter.default)
            source = inspect.getsource(function)
            self.assertIn("deb_best_improvement_local_search", source)
            self.assertNotIn("deb_equal_descent_local_search", source)
        forwarded = inspect.getsource(continue_after_initialisation_deb)
        self.assertIn("local_search=local_search", forwarded)
        self.assertIsNone(
            inspect.signature(continue_after_initialisation_deb)
            .parameters["local_search"]
            .default
        )

    def test_factory_constructs_the_cold_token(self) -> None:
        algorithm = create_algorithm(
            "K_ELITE_AMS_PEN_EQU_DESC_COLD_START",
            {"max_evaluations": 4, "archive_size": 1, "n_parents": 1},
            random_seed=3,
        )
        self.assertIs(algorithm.local_search, equal_descent_local_search)
        self.assertEqual(algorithm.name, "K_ELITE_AMS_PEN_EQU_DESC_COLD_START")


if __name__ == "__main__":
    unittest.main()
