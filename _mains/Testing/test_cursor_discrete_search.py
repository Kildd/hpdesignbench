"""
Discrete MLSL and population basin hopping.

Covers the shared integer search, linkage rules, replacement edges, the IOH
trace, and an integrity smoke on the HP slab.
"""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ioh
import numpy as np

from core.ioh_core.algorithm_factory import AlgorithmFactoryError, create_algorithm
from core.ioh_core.algorithms.common import RunController
from core.ioh_core.algorithms.native.discrete_search import (
    DiscreteMLSL,
    DiscreteMLSLWarmStart,
    DiscreteSampler,
    SobolMultistart,
    SobolMultistartEquDesc,
    SobolSample,
    equal_descent_local_search,
    DiscreteSearchSpace,
    EvaluatedPoint,
    EvaluationManager,
    LocalSearchResult,
    PopulationBasinHopping,
    PopulationBasinHoppingWarmStart,
    apply_population_replacement,
    best_improvement_local_search,
    box_perturbation,
    critical_radius,
    better_sample_within_radius,
    mean_pairwise_distance,
    run_pbh_generation,
)
from core.ioh_core.algorithms.registry import (
    BudgetSemantics,
    CandidateMode,
    SeedPolicy,
    get_algorithm_spec,
)
from core.ioh_core.problem_builder import build_problem_from_snapshot
from core.ioh_core.workflows.optimization_runs.io_logging import run_logged_optimization
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
    snapshot_slab_inputs,
)
from _mains.Testing.helpers.fake_ioh_problem import (
    FakeBest,
    FakeIOHProblem,
    assert_algorithm_conformance,
)

_IOH_TOY_NAME = "hdb_discrete_search_sum"
_IOH_RECORDED: list[tuple[list[int], float]] = []


def _ensure_sum_ioh_problem() -> None:
    """Register one cheap integer IOH problem for the trace fixture."""
    names = set(ioh.problem.IntegerSingleObjective.problems.values())
    if _IOH_TOY_NAME in names:
        return

    def objective(x: list[int]) -> float:
        row = [int(value) for value in x]
        value = float(sum(row))
        _IOH_RECORDED.append((row, value))
        return value

    ioh.problem.wrap_integer_problem(objective, _IOH_TOY_NAME, lb=0, ub=20)


class QuadraticProblem(FakeIOHProblem):
    """Integer quadratic with a unique minimum at (2, 3)."""

    def __call__(self, x: object) -> float:
        candidate = [int(v) for v in list(x)]
        self.calls.append(candidate)
        self.state.evaluations += 1
        self.hdb_last_feasible = True
        value = float((candidate[0] - 2) ** 2 + (candidate[1] - 3) ** 2)
        if value < self.state.current_best.y:
            self.state.current_best = FakeBest(x=list(candidate), y=value)
        return value


class FlatProblem(FakeIOHProblem):
    """Constant objective. Every point is a strict local optimum."""

    def __call__(self, x: object) -> float:
        candidate = [int(v) for v in list(x)]
        self.calls.append(candidate)
        self.state.evaluations += 1
        self.hdb_last_feasible = True
        value = 1.0
        if value < self.state.current_best.y:
            self.state.current_best = FakeBest(x=list(candidate), y=value)
        return value


class ScriptedProblem(FakeIOHProblem):
    """Returns a prepared sequence, then zeroes."""

    def __init__(self, lower: list[object], upper: list[object], values: list[float]) -> None:
        super().__init__(lower, upper)
        self._values = list(values)

    def __call__(self, x: object) -> float:
        candidate = [int(v) for v in list(x)]
        self.calls.append(candidate)
        self.state.evaluations += 1
        self.hdb_last_feasible = True
        value = self._values.pop(0) if self._values else 0.0
        if math.isfinite(value) and value < self.state.current_best.y:
            self.state.current_best = FakeBest(x=list(candidate), y=value)
        return value


def _manager(
    problem: FakeIOHProblem,
    budget: int,
) -> EvaluationManager:
    """Build an evaluation manager on the problem's integer bounds."""
    controller = RunController(problem, budget)
    space = DiscreteSearchSpace(problem.bounds.lb, problem.bounds.ub)
    return EvaluationManager(problem, controller, space)


def _raw_objectives(directory: Path) -> list[float]:
    """Read the raw objective column from an IOH Analyzer ``.dat`` trace."""
    rows: list[float] = []
    for path in sorted(directory.rglob("*.dat")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if (not stripped) or stripped.startswith("%") or stripped.startswith('"'):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                int(parts[0])
                value = float(parts[1])
            except ValueError:
                continue
            rows.append(value)
    return rows


class TestExistingOptimisers(unittest.TestCase):
    """Ticket 1: factory and seed policy of the two reference algorithms."""

    def test_random_search_runs_and_direct_l_rejects_a_seed(self) -> None:
        problem = FakeIOHProblem([0, 0], [3, 3])
        algorithm = create_algorithm(
            "RANDOM_SEARCH",
            {"max_evaluations": 4},
            random_seed=1,
        )
        algorithm(problem)
        self.assertEqual(len(problem.calls), 4)
        self.assertEqual(problem.state.evaluations, 4)
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "NLOPT_DIRECT_L",
                {"max_evaluations": 4},
                random_seed=1,
            )


class TestDiscreteSearchSpace(unittest.TestCase):
    """Ticket 2: bounds, distance, and the L1 neighbourhood."""

    def test_normalisation_distance_and_neighbours(self) -> None:
        space = DiscreteSearchSpace([0, 2], [4, 2])
        self.assertEqual(space.normalise((0, 2)).tolist(), [0.0, 0.0])
        self.assertEqual(space.normalise((4, 2)).tolist(), [1.0, 0.0])
        self.assertEqual(space.distance((1, 2), (1, 2)), 0.0)
        self.assertEqual(
            space.distance((0, 2), (4, 2)),
            space.distance((4, 2), (0, 2)),
        )
        interior = DiscreteSearchSpace([0, 0], [4, 5])
        self.assertEqual(len(interior.neighbours_l1((2, 3))), 4)
        corner = interior.neighbours_l1((0, 0))
        self.assertLessEqual(len(corner), 4)
        self.assertEqual(corner, [(1, 0), (0, 1)])
        for neighbour in corner:
            interior.validate(neighbour)
        fixed = DiscreteSearchSpace([0, 5], [2, 5])
        self.assertEqual(fixed.neighbours_l1((1, 5)), [(0, 5), (2, 5)])
        self.assertEqual(fixed.n_free_dimensions, 1)

    def test_rejects_bad_candidates_and_bounds(self) -> None:
        space = DiscreteSearchSpace([0, 0], [2, 2])
        with self.assertRaises(ValueError):
            space.validate((1.5, 0))
        with self.assertRaises(ValueError):
            space.validate((True, 0))
        with self.assertRaises(ValueError):
            space.validate((3, 0))
        with self.assertRaises(ValueError):
            space.validate((0,))
        with self.assertRaises(ValueError):
            DiscreteSearchSpace([0], [True])
        with self.assertRaises(ValueError):
            DiscreteSearchSpace([1.5], [3])
        with self.assertRaises(ValueError):
            DiscreteSearchSpace([2], [1])
        with self.assertRaises(ValueError):
            DiscreteSearchSpace([], [])

    def test_whole_number_float_bounds_from_the_problem(self) -> None:
        problem = QuadraticProblem([0.0, 0.0], [4.0, 5.0])
        algorithm = DiscreteMLSL(
            max_evaluations=4,
            seed=1,
            batch_size=4,
        )
        algorithm(problem)
        self.assertEqual(len(problem.calls), 4)
        for call in problem.calls:
            self.assertTrue(all(isinstance(value, int) for value in call))


class TestLocalSearch(unittest.TestCase):
    """Ticket 2: strict L1 best improvement."""

    def test_quadratic_reaches_the_minimum_from_every_start(self) -> None:
        lower = [0, 0]
        upper = [4, 5]
        for start in ((0, 0), (4, 0), (0, 5), (4, 5), (2, 3)):
            problem = QuadraticProblem(lower, upper)
            manager = _manager(problem, 80)
            origin = EvaluatedPoint(start, float((start[0] - 2) ** 2 + (start[1] - 3) ** 2))
            result = best_improvement_local_search(origin, manager, manager.space)
            self.assertEqual(result.point.x, (2, 3))
            self.assertTrue(result.certified_local_optimum)
            self.assertEqual(result.n_evaluations, len(problem.calls))
            for neighbour in manager.space.neighbours_l1(result.point.x):
                value = float((neighbour[0] - 2) ** 2 + (neighbour[1] - 3) ** 2)
                self.assertGreaterEqual(value, result.point.f)

    def test_predecessor_is_not_evaluated_again(self) -> None:
        problem = QuadraticProblem([0, 0], [5, 5])
        manager = _manager(problem, 40)
        start = EvaluatedPoint((0, 0), float(4 + 9))
        result = best_improvement_local_search(start, manager, manager.space)
        self.assertEqual(result.point.x, (2, 3))
        self.assertNotIn([0, 0], problem.calls)

    def test_boundary_optimum_is_certified(self) -> None:
        problem = QuadraticProblem([0, 0], [1, 1])
        manager = _manager(problem, 20)
        start = EvaluatedPoint((0, 0), float(4 + 9))
        result = best_improvement_local_search(start, manager, manager.space)
        self.assertEqual(result.point.x, (1, 1))
        self.assertTrue(result.certified_local_optimum)

    def test_plateau_stops_at_the_start(self) -> None:
        problem = FlatProblem([0, 0], [2, 2])
        manager = _manager(problem, 20)
        start = EvaluatedPoint((1, 1), 1.0)
        result = best_improvement_local_search(start, manager, manager.space)
        self.assertEqual(result.point.x, (1, 1))
        self.assertTrue(result.certified_local_optimum)
        self.assertEqual(result.n_steps, 0)

    def test_budget_inside_a_neighbourhood_is_not_certified(self) -> None:
        problem = QuadraticProblem([0, 0], [4, 4])
        manager = _manager(problem, 2)
        start = EvaluatedPoint((2, 2), 8.0)
        result = best_improvement_local_search(start, manager, manager.space)
        self.assertFalse(result.certified_local_optimum)
        self.assertEqual(len(problem.calls), 2)
        self.assertEqual(result.n_evaluations, 2)
        self.assertEqual(result.point.x, (1, 2))
        self.assertEqual(manager.controller.used, 2)

    def test_nonfinite_values_and_exceptions(self) -> None:
        first_inf = ScriptedProblem([0, 0], [3, 3], [math.inf, 4.0])
        algorithm = DiscreteMLSL(max_evaluations=2, seed=1, batch_size=8)
        algorithm(first_inf)
        self.assertEqual(algorithm.last_run_stats["best_f"], 4.0)
        self.assertEqual(algorithm.last_run_stats["best_x"], first_inf.calls[1])

        only_inf = ScriptedProblem([0, 0], [3, 3], [math.inf, math.inf, math.inf])
        only_algorithm = DiscreteMLSL(max_evaluations=3, seed=1, batch_size=8)
        only_algorithm(only_inf)
        self.assertIsNone(only_algorithm.last_run_stats["best_f"])
        self.assertIsNone(only_algorithm.last_run_stats["best_x"])
        self.assertEqual(only_algorithm.last_run_stats["n_evaluations"], 3)

        nan_problem = ScriptedProblem([0, 0], [3, 3], [math.nan])
        with self.assertRaises(RuntimeError):
            algorithm(nan_problem)
        self.assertEqual(nan_problem.state.evaluations, 1)

        negative = ScriptedProblem([0, 0], [3, 3], [float("-inf")])
        with self.assertRaises(RuntimeError):
            DiscreteMLSL(max_evaluations=2, seed=1, batch_size=8)(negative)
        self.assertEqual(negative.state.evaluations, 1)

        class Boom(FakeIOHProblem):
            def __call__(self, x: object) -> float:
                raise ValueError("objective failed")

        with self.assertRaises(ValueError):
            DiscreteMLSL(max_evaluations=2, seed=1)(Boom([0, 0], [1, 1]))


class TestSamplerAndMlslToy(unittest.TestCase):
    """Tickets 2 and 3: projection, linkage, and a unimodal MLSL run."""

    def test_projection_is_deterministic_and_clips(self) -> None:
        space = DiscreteSearchSpace([0, 2], [3, 2])
        sampler = DiscreteSampler(space, "sobol", 0, np.random.default_rng(0))
        sampler._sobol.random = lambda n: np.array([[0.0, 0.5]])
        self.assertEqual(sampler.draw(), (0, 2))
        sampler._sobol.random = lambda n: np.array([[0.999, 0.0]])
        self.assertEqual(sampler.draw(), (3, 2))
        sampler._sobol.random = lambda n: np.array([[1.0, 0.0]])
        self.assertEqual(sampler.draw(), (3, 2))

        left = DiscreteSampler(space, "sobol", 4, np.random.default_rng(4))
        right = DiscreteSampler(space, "sobol", 4, np.random.default_rng(4))
        self.assertEqual([left.draw() for _ in range(5)], [right.draw() for _ in range(5)])
        full = DiscreteSearchSpace([0], [1])
        blocked = DiscreteSampler(full, "uniform", 1, np.random.default_rng(1))
        self.assertIsNone(blocked.next_unique({(0,), (1,)}))

    def test_duplicate_projection_is_not_evaluated(self) -> None:
        drawn: list[tuple[int, ...]] = []
        planned = [(0, 0), (0, 0), (1, 0)]

        def draw(self: DiscreteSampler) -> tuple[int, ...]:
            point = planned[len(drawn)]
            drawn.append(point)
            return point

        problem = FlatProblem([0, 0], [1, 1])
        with patch.object(DiscreteSampler, "draw", draw):
            DiscreteMLSL(
                max_evaluations=2,
                seed=1,
                batch_size=2,
                reduction_fraction=1.0,
            )(problem)
        self.assertEqual(drawn, planned)
        self.assertEqual(problem.calls, [[0, 0], [1, 0]])

    def test_critical_radius_matches_reference_values(self) -> None:
        self.assertTrue(math.isinf(critical_radius(1, 2, 2.0)))
        self.assertTrue(math.isinf(critical_radius(4, 0, 2.0)))
        self.assertAlmostEqual(critical_radius(2, 1, 2.0), 0.34657359027997264)
        self.assertAlmostEqual(critical_radius(3, 1, 2.0), 0.3662040962227032)
        self.assertAlmostEqual(critical_radius(4, 1, 2.0), 0.34657359027997264)
        self.assertGreater(critical_radius(3, 1, 2.0), critical_radius(2, 1, 2.0))
        for dimension in (1, 4):
            for count in range(3, 12):
                self.assertLess(
                    critical_radius(count + 1, dimension, 2.0),
                    critical_radius(count, dimension, 2.0),
                )

    def test_suppression_uses_strict_improvement_and_strict_radius(self) -> None:
        space = DiscreteSearchSpace([0], [10])
        candidate = EvaluatedPoint((5,), 5.0)
        self.assertTrue(
            better_sample_within_radius(
                candidate,
                [EvaluatedPoint((4,), 1.0)],
                0.2,
                space,
            )
        )
        self.assertFalse(
            better_sample_within_radius(
                candidate,
                [EvaluatedPoint((6,), 9.0)],
                0.2,
                space,
            )
        )
        self.assertFalse(
            better_sample_within_radius(
                candidate,
                [EvaluatedPoint((0,), 1.0)],
                0.2,
                space,
            )
        )
        self.assertFalse(
            better_sample_within_radius(
                candidate,
                [EvaluatedPoint((4,), 5.0)],
                1.0,
                space,
            )
        )
        self.assertFalse(
            better_sample_within_radius(
                candidate,
                [EvaluatedPoint((4,), 1.0)],
                space.distance((5,), (4,)),
                space,
            )
        )

    def test_quadratic_run_is_reproducible_and_within_budget(self) -> None:
        def run(seed: int) -> list[list[int]]:
            problem = QuadraticProblem([0, 0], [4, 5])
            algorithm = DiscreteMLSL(
                max_evaluations=40,
                seed=seed,
                batch_size=4,
                reduction_fraction=1.0,
                sigma=2.0,
            )
            algorithm(problem)
            self.assertLessEqual(len(problem.calls), 40)
            self.assertEqual(algorithm.last_run_stats["best_x"], [2, 3])
            self.assertEqual(algorithm.last_run_stats["best_f"], 0.0)
            self.assertEqual(
                algorithm.last_run_stats["best_x"],
                list(problem.state.current_best.x),
            )
            return problem.calls

        self.assertEqual(run(3), run(3))
        again = DiscreteMLSL(
            max_evaluations=40,
            seed=3,
            batch_size=4,
            reduction_fraction=1.0,
        )
        first = QuadraticProblem([0, 0], [4, 5])
        second = QuadraticProblem([0, 0], [4, 5])
        again(first)
        again(second)
        self.assertEqual(first.calls, second.calls)

    def test_spent_batch_does_not_start_local_search(self) -> None:
        problem = FlatProblem([0, 0], [6, 6])
        algorithm = DiscreteMLSL(max_evaluations=10, seed=2, batch_size=128)
        algorithm(problem)
        stats = algorithm.last_run_stats
        self.assertEqual(stats["n_evaluations"], 10)
        self.assertEqual(stats["n_mlsl_iterations"], 0)
        self.assertEqual(stats["evaluations_per_local_search"], [])
        self.assertEqual(stats["n_global_samples"], 10)

    def test_reduced_sample_launches_the_best_prefix_once(self) -> None:
        problem = QuadraticProblem([0, 0], [8, 8])
        algorithm = DiscreteMLSL(
            max_evaluations=6,
            seed=5,
            batch_size=5,
            reduction_fraction=0.2,
        )
        algorithm(problem)
        stats = algorithm.last_run_stats
        scored = [
            ((call[0] - 2) ** 2 + (call[1] - 3) ** 2, tuple(call))
            for call in problem.calls[:5]
        ]
        best = min(scored)[1]
        self.assertEqual(stats["n_global_samples"], 5)
        self.assertEqual(stats["n_mlsl_iterations"], 1)
        self.assertEqual(
            [tuple(point) for point in stats["local_search_start_points"]],
            [best],
        )
        self.assertEqual(
            stats["n_evaluations"],
            stats["n_global_samples"] + sum(stats["evaluations_per_local_search"]),
        )

    def test_full_index_set_gets_one_local_phase(self) -> None:
        problem = FlatProblem([0, 0], [1, 1])
        algorithm = DiscreteMLSL(
            max_evaluations=40,
            seed=1,
            batch_size=3,
            reduction_fraction=1.0,
        )
        algorithm(problem)
        stats = algorithm.last_run_stats
        self.assertEqual(stats["n_global_samples"], 4)
        self.assertEqual(stats["n_mlsl_iterations"], 2)
        starts = [tuple(point) for point in stats["local_search_start_points"]]
        self.assertEqual(len(starts), len(set(starts)))
        self.assertGreater(stats["n_candidates_already_started"], 0)
        self.assertEqual(
            stats["n_evaluations"],
            stats["n_global_samples"] + sum(stats["evaluations_per_local_search"]),
        )
        self.assertLessEqual(stats["n_evaluations"], 40)

    def test_fixed_target_stops_on_the_hitting_evaluation(self) -> None:
        problem = QuadraticProblem([2, 3], [2, 3])
        controller = RunController(
            problem,
            12,
            mode="fixed_target",
            target_threshold=0.0,
            target_reference="all",
        )
        DiscreteMLSL(max_evaluations=12, seed=1, batch_size=4)(
            problem,
            run_controller=controller,
        )
        self.assertEqual(len(problem.calls), 1)
        self.assertTrue(controller.target_hit)


class TestPopulationBasinHopping(unittest.TestCase):
    """Tickets 4 and 5: population, replacement, and the initialisation edges."""

    def test_registry_and_labels(self) -> None:
        mlsl = get_algorithm_spec("DISCRETE_MLSL_PEN_SI_DESC_COLD_START")
        pbh = get_algorithm_spec("POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START")
        for spec in (mlsl, pbh):
            self.assertEqual(spec.library, "native")
            self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
            self.assertEqual(spec.budget_semantics, BudgetSemantics.HARD_IOH_EVALUATIONS)
            self.assertFalse(spec.batch_evaluations)
            self.assertIsNone(spec.dependency_extra)
            self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        sobol = DiscreteMLSL(max_evaluations=4, seed=1)
        self.assertIn("Sobol", sobol.info)
        uniform = DiscreteMLSL(max_evaluations=4, seed=1, sampler="uniform")
        self.assertIn("uniform", uniform.info)
        self.assertNotIn("Sobol", uniform.info)
        hopper = PopulationBasinHopping(
            max_evaluations=4,
            seed=1,
            initial_sampler="uniform",
        )
        self.assertIn("uniform", hopper.info)
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "DISCRETE_MLSL_PEN_SI_DESC_COLD_START",
                {"max_evaluations": 4, "reduction_fraction": 0},
                random_seed=1,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "DISCRETE_MLSL_PEN_SI_DESC_COLD_START",
                {"max_evaluations": 4, "sigma": 0},
                random_seed=1,
            )
        with self.assertRaises(ValueError):
            PopulationBasinHopping(max_evaluations=4, dcut_factor=0)
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                {"max_evaluations": 4, "dcut_factor": 0},
                random_seed=1,
            )

    def test_seeded_cold_and_warm_evaluation_traces_are_frozen(self) -> None:
        cold_problem = FlatProblem([0, 0], [2, 2])
        cold = PopulationBasinHopping(
            max_evaluations=12,
            seed=7,
            population_size=2,
            perturbation_fraction=0.5,
            initial_sampler="uniform",
        )
        cold(cold_problem)
        self.assertEqual(
            cold_problem.calls,
            [
                [2, 1], [1, 1], [2, 0], [2, 2], [2, 2], [1, 2],
                [2, 1], [2, 2], [1, 2], [2, 1], [2, 1], [1, 1],
            ],
        )
        self.assertEqual(
            cold.last_run_stats["evaluations_per_local_search"],
            [3, 2, 2, 1],
        )
        self.assertEqual(cold.last_run_stats["best_f"], 1.0)
        self.assertEqual(cold.last_run_stats["initial_population_size"], 2)
        self.assertEqual(cold.last_run_stats["n_generations"], 1)
        self.assertEqual(cold.info.split(" ", 1)[0], "PBH-L1")

        first = [0] * 9
        second = [1] + [0] * 8
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "problem_id": "L8100_B2",
                        "n": 2,
                        "vectors": [first, second],
                    }
                ),
                encoding="utf-8",
            )
            warm_problem = FlatProblem([0] * 9, [1] * 9)
            warm = PopulationBasinHoppingWarmStart(
                max_evaluations=12,
                seed=7,
                population_size=2,
                perturbation_fraction=0.5,
                initial_sampler="uniform",
                initial_design_path=str(path),
            )
            warm(warm_problem)
        expected_warm = [
            first,
            second,
            second,
            *[
                [int(index == axis) for index in range(9)]
                for axis in range(1, 9)
            ],
            first,
        ]
        self.assertEqual(warm_problem.calls, expected_warm)
        self.assertEqual(
            warm.last_run_stats["evaluations_per_local_search"],
            [9, 1],
        )
        self.assertEqual(warm.last_run_stats["best_f"], 1.0)
        self.assertEqual(warm.last_run_stats["initial_population_size"], 1)
        self.assertEqual(warm.last_run_stats["n_generations"], 0)
        self.assertEqual(warm.name, "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START")
        self.assertEqual(warm.info.split(" ", 1)[0], "PBH-L1")

    def test_conformance(self) -> None:
        def build_mlsl(seed: int | None) -> DiscreteMLSL:
            return create_algorithm(
                "DISCRETE_MLSL_PEN_SI_DESC_COLD_START",
                {"max_evaluations": 8, "batch_size": 4, "reduction_fraction": 1},
                random_seed=seed,
            )

        def build_pbh(seed: int | None) -> PopulationBasinHopping:
            return create_algorithm(
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                {"max_evaluations": 8, "population_size": 3},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="DISCRETE_MLSL_PEN_SI_DESC_COLD_START",
            build_algorithm=build_mlsl,
            seed=11,
            second_seed=29,
        )
        assert_algorithm_conformance(
            self,
            algorithm_name="POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
            build_algorithm=build_pbh,
            seed=11,
            second_seed=29,
        )

    def test_replacement_rules(self) -> None:
        space = DiscreteSearchSpace([0], [10])
        population = [EvaluatedPoint((0,), 5.0), EvaluatedPoint((10,), 3.0)]
        self.assertEqual(
            apply_population_replacement(EvaluatedPoint((1,), 0.0), population, space, 0.5),
            "near",
        )
        self.assertEqual(population[0].x, (1,))

        population = [EvaluatedPoint((0,), 5.0), EvaluatedPoint((10,), 3.0)]
        self.assertEqual(
            apply_population_replacement(EvaluatedPoint((1,), 9.0), population, space, 0.5),
            "rejected",
        )
        self.assertEqual([member.x for member in population], [(0,), (10,)])

        population = [EvaluatedPoint((0,), 1.0), EvaluatedPoint((10,), 5.0)]
        self.assertEqual(
            apply_population_replacement(EvaluatedPoint((9,), 0.0), population, space, 0.05),
            "diverse",
        )
        self.assertEqual([member.x for member in population], [(0,), (9,)])

        population = [EvaluatedPoint((0,), 1.0), EvaluatedPoint((10,), 5.0)]
        self.assertEqual(
            apply_population_replacement(EvaluatedPoint((9,), 6.0), population, space, 0.05),
            "rejected",
        )
        self.assertEqual(
            apply_population_replacement(EvaluatedPoint((9,), 5.0), population, space, 0.05),
            "rejected",
        )

        population = [EvaluatedPoint((0,), 0.0), EvaluatedPoint((10,), 0.0)]
        self.assertEqual(
            apply_population_replacement(EvaluatedPoint((5,), -1.0), population, space, 1.0),
            "near",
        )
        self.assertEqual([member.x for member in population], [(5,), (10,)])

        population = [EvaluatedPoint((0,), 5.0), EvaluatedPoint((3,), 5.0)]
        self.assertEqual(
            apply_population_replacement(EvaluatedPoint((10,), 0.0), population, space, 0.1),
            "diverse",
        )
        self.assertEqual([member.x for member in population], [(0,), (10,)])

        population = [EvaluatedPoint((0,), 1.0), EvaluatedPoint((10,), 4.0)]
        self.assertEqual(
            apply_population_replacement(EvaluatedPoint((5,), 0.0), population, space, 0.5),
            "diverse",
        )
        self.assertEqual([member.x for member in population], [(0,), (5,)])

        population = [EvaluatedPoint((0,), 1.0), EvaluatedPoint((10,), 5.0)]
        self.assertEqual(
            apply_population_replacement(EvaluatedPoint((0,), -100.0), population, space, 0.0),
            "rejected",
        )
        self.assertEqual([member.x for member in population], [(0,), (10,)])

        single = [EvaluatedPoint((4,), 3.0)]
        self.assertEqual(
            apply_population_replacement(EvaluatedPoint((1,), 1.0), single, space, math.inf),
            "near",
        )
        self.assertEqual(single[0].x, (1,))
        self.assertEqual(
            apply_population_replacement(EvaluatedPoint((2,), 4.0), single, space, math.inf),
            "rejected",
        )

    def test_duplicate_initial_draw_counts_and_is_not_evaluated(self) -> None:
        sequence = iter([(0, 0), (0, 0), (1, 0), (9, 0)])

        def draw(self: DiscreteSampler) -> tuple[int, ...]:
            return next(sequence)

        problem = FlatProblem([0, 0], [1, 0])
        with patch.object(DiscreteSampler, "draw", draw):
            algorithm = PopulationBasinHopping(
                max_evaluations=4,
                seed=1,
                population_size=5,
                perturbation_fraction=1.0,
            )
            algorithm(problem)
        self.assertEqual(problem.calls, [[0, 0], [1, 0], [1, 0], [0, 0]])
        self.assertEqual(algorithm.last_run_stats["initial_population_size"], 2)
        self.assertEqual(next(sequence), (9, 0))

    def test_initialisation_stops_when_the_index_set_is_exhausted(self) -> None:
        draws = {"n": 0}

        def draw(self: DiscreteSampler) -> tuple[int, ...]:
            draws["n"] += 1
            return (draws["n"] % 2,)

        problem = FlatProblem([0], [1])
        algorithm = PopulationBasinHopping(
            max_evaluations=40,
            seed=1,
            population_size=8,
            perturbation_fraction=1.0,
        )
        with patch.object(DiscreteSampler, "draw", draw):
            algorithm(problem)
        stats = algorithm.last_run_stats
        self.assertEqual(draws["n"], 2)
        self.assertEqual(stats["initial_population_size"], 2)

    def test_repeated_draw_does_not_loop_and_sets_infinite_dcut(self) -> None:
        problem = FlatProblem([0, 0], [3, 3])
        with patch.object(DiscreteSampler, "draw", lambda self: (1, 1)):
            algorithm = PopulationBasinHopping(
                max_evaluations=30,
                seed=1,
                population_size=4,
            )
            algorithm(problem)
        stats = algorithm.last_run_stats
        self.assertEqual(stats["initial_population_size"], 1)
        self.assertTrue(math.isinf(stats["dcut"]))
        self.assertIsNone(stats["initial_mean_distance"])
        self.assertEqual(stats["n_evaluations"], 30)
        self.assertLess(problem.calls.count([1, 1]), 10)
        self.assertGreaterEqual(stats["n_generations"], 1)

    def test_empty_population_does_not_open_a_generation(self) -> None:
        def uncertified(
            start: EvaluatedPoint,
            evaluator: EvaluationManager,
            space: DiscreteSearchSpace,
        ) -> LocalSearchResult:
            return LocalSearchResult(start, False, 0, 0)

        problem = FlatProblem([0, 0], [5, 5])
        with patch(
            "core.ioh_core.algorithms.native.discrete_search.best_improvement_local_search",
            uncertified,
        ):
            algorithm = PopulationBasinHopping(
                max_evaluations=80,
                seed=1,
                population_size=2,
            )
            algorithm(problem)
        stats = algorithm.last_run_stats
        self.assertEqual(stats["initial_population_size"], 0)
        self.assertIsNone(stats["dcut"])
        self.assertEqual(stats["n_generations"], 0)
        self.assertLess(stats["n_evaluations"], 80)
        self.assertNotIn("_unique_optima", stats)

    def test_generation_snapshot_and_immovable_parent(self) -> None:
        problem = FlatProblem([0, 0], [4, 4])
        manager = _manager(problem, 10)
        population = [EvaluatedPoint((0, 0), 5.0), EvaluatedPoint((4, 4), 5.0)]
        seen: list[tuple[int, ...]] = []

        def perturb(
            x: object,
            space: DiscreteSearchSpace,
            rng: np.random.Generator,
            perturbation_fraction: float,
        ) -> tuple[int, ...] | None:
            seen.append(tuple(x))
            if len(seen) == 1:
                return (1, 0)
            return None

        def certified(
            start: EvaluatedPoint,
            evaluator: EvaluationManager,
            space: DiscreteSearchSpace,
        ) -> LocalSearchResult:
            return LocalSearchResult(EvaluatedPoint((1, 0), 0.0), True, 1, 0)

        stats = {
            "n_generations": 0,
            "n_children": 0,
            "n_near_replacements": 0,
            "n_diverse_replacements": 0,
            "n_rejected_children": 0,
            "n_local_searches": 0,
            "n_local_search_steps": 0,
            "n_certified_local_searches": 0,
            "n_unique_certified_local_optima": 0,
            "evaluations_per_local_search": [],
        }
        with (
            patch(
                "core.ioh_core.algorithms.native.discrete_search.box_perturbation",
                perturb,
            ),
            patch(
                "core.ioh_core.algorithms.native.discrete_search.best_improvement_local_search",
                certified,
            ),
        ):
            immovable = run_pbh_generation(
                population,
                manager,
                manager.space,
                np.random.default_rng(0),
                0.15,
                0.01,
                stats,
            )
        self.assertTrue(immovable)
        self.assertEqual(seen, [(0, 0), (4, 4)])
        self.assertEqual(len(problem.calls), 1)
        self.assertEqual([member.x for member in population], [(0, 0), (1, 0)])
        self.assertEqual(stats["n_generations"], 1)
        self.assertEqual(stats["n_children"], 1)

    def test_half_to_even_perturbation_window(self) -> None:
        space = DiscreteSearchSpace([0], [10])
        rng = np.random.default_rng(0)
        seen: set[int] = set()
        for _ in range(200):
            draw = box_perturbation((5,), space, rng, 0.25)
            self.assertIsNotNone(draw)
            assert draw is not None
            self.assertGreaterEqual(draw[0], 3)
            self.assertLessEqual(draw[0], 7)
            seen.add(draw[0])
        self.assertIn(3, seen)
        self.assertIn(7, seen)
        self.assertIsNone(
            box_perturbation((3,), DiscreteSearchSpace([3], [3]), np.random.default_rng(1), 0.15)
        )

    def test_mean_distance_matches_the_pair_average(self) -> None:
        space = DiscreteSearchSpace([0], [10])
        self.assertIsNone(mean_pairwise_distance([EvaluatedPoint((1,), 0.0)], space))
        points = [EvaluatedPoint((0,), 0.0), EvaluatedPoint((10,), 1.0)]
        self.assertAlmostEqual(mean_pairwise_distance(points, space), 1.0)


class TestIohTrace(unittest.TestCase):
    """Ticket 6: fixed-budget IOH trace on a space that cannot be exhausted."""

    def test_both_algorithms_log_one_hundred_evaluations(self) -> None:
        _ensure_sum_ioh_problem()
        cases = (
            (
                "DISCRETE_MLSL_PEN_SI_DESC_COLD_START",
                {"max_evaluations": 100, "batch_size": 128},
            ),
            (
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                {"max_evaluations": 100, "population_size": 4, "perturbation_fraction": 0.5},
            ),
        )
        for name, parameters in cases:
            with TemporaryDirectory() as tmp:
                _IOH_RECORDED.clear()
                problem = ioh.problem.IntegerSingleObjective.create(_IOH_TOY_NAME, 1, 4)
                # Default Analyzer. An inline ``[trigger.Each(1)]`` is freed
                # after this call: IOH 0.3.22 does not retain the trigger
                # objects, and the next evaluation access-violates. A named
                # list kept for the logger's lifetime does not. The default
                # trigger still stores the evaluation count; on a wrapped
                # integer problem the ``.dat`` rows are first and last.
                logger = ioh.logger.Analyzer(
                    root=tmp,
                    folder_name="run",
                    algorithm_name=name,
                    algorithm_info="trace",
                    store_positions=True,
                )
                algorithm = create_algorithm(name, parameters, random_seed=7)
                problem.attach_logger(logger)
                try:
                    returned = algorithm(problem)
                    evaluations = int(problem.state.evaluations)
                finally:
                    problem.detach_logger()
                    logger.close()
                self.assertIs(returned, algorithm)
                self.assertEqual(evaluations, 100)
                self.assertEqual(algorithm.last_run_stats["n_evaluations"], 100)
                self.assertEqual(len(_IOH_RECORDED), 100)
                for candidate, _value in _IOH_RECORDED:
                    self.assertEqual(len(candidate), 4)
                    self.assertTrue(all(0 <= coordinate <= 20 for coordinate in candidate))
                best_so_far = []
                running = math.inf
                for _candidate, value in _IOH_RECORDED:
                    running = min(running, value)
                    best_so_far.append(running)
                self.assertTrue(
                    all(best_so_far[i + 1] <= best_so_far[i] for i in range(99))
                )
                jsons = list(Path(tmp).rglob("IOHprofiler_f*.json"))
                self.assertTrue(jsons)
                for path in jsons:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    runs = payload["scenarios"][0]["runs"]
                    self.assertEqual(runs[0]["evals"], 100)
        mlsl = create_algorithm(
            "DISCRETE_MLSL_PEN_SI_DESC_COLD_START",
            {"max_evaluations": 100},
            random_seed=7,
        )
        self.assertIn("batch=128", mlsl.info)


class TestHpSlabSmoke(unittest.TestCase):
    """Ticket 7: integrity only, on the HP slab at three budgets."""

    def test_budgets_are_repeatable_and_respect_the_trace(self) -> None:
        budgets = (100, 250, 500)
        specs = (
            ("DISCRETE_MLSL_PEN_SI_DESC_COLD_START", {}),
            (
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                {"population_size": 4, "perturbation_fraction": 0.5},
            ),
            (
                "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
                {"population_size": 4, "perturbation_fraction": 0.5},
            ),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot_slab_inputs(slab_type="hp_slab", inputs_dir=root / "inputs")
            snapshot_dir = resolve_snapshot_slab_dir(root / "inputs", "hp_slab")
            for name, extra in specs:
                for budget in budgets:
                    sequences = []
                    summaries = []
                    stats = []
                    for copy in (0, 1):
                        bundle = build_problem_from_snapshot(
                            slab_type="hp_slab",
                            problem_id="L6750_B2",
                            snapshot_dir=snapshot_dir,
                        )
                        algorithm = create_algorithm(
                            name,
                            {"max_evaluations": budget, **extra},
                            random_seed=11,
                        )
                        summary = run_logged_optimization(
                            bundle=bundle,
                            algorithm=algorithm,
                            output_directory=root / f"{name}-{budget}-{copy}",
                        )
                        summaries.append(summary)
                        stats.append(algorithm.last_run_stats)
                        sequences.append(
                            _raw_objectives(root / f"{name}-{budget}-{copy}")
                        )
                    self.assertEqual(sequences[0], sequences[1])
                    self.assertEqual(summaries[0]["evaluations"], budget)
                    self.assertEqual(stats[0]["n_evaluations"], budget)
                    self.assertEqual(len(sequences[0]), budget)
                    incumbent = math.inf
                    previous = math.inf
                    for value in sequences[0]:
                        incumbent = min(incumbent, value)
                        self.assertLessEqual(incumbent, previous)
                        previous = incumbent
                    best_x = summaries[0]["best_x"]
                    self.assertTrue(all(isinstance(value, int) for value in best_x))
                    if name == "DISCRETE_MLSL_PEN_SI_DESC_COLD_START" and budget == 100:
                        self.assertEqual(stats[0]["n_mlsl_iterations"], 0)
                        self.assertEqual(stats[0]["evaluations_per_local_search"], [])
                    if name == "DISCRETE_MLSL_PEN_SI_DESC_COLD_START" and budget in (250, 500):
                        self.assertGreaterEqual(stats[0]["n_mlsl_iterations"], 1)
                        self.assertTrue(stats[0]["evaluations_per_local_search"])
                    self.assertIn("best_f", stats[0])
                    self.assertIn("seed", stats[0])
                    if name == "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START":
                        diagnostics = summaries[0]["algorithm_diagnostics"]
                        self.assertEqual(diagnostics["label"], "PBH-L1-Deb")
                        self.assertEqual(diagnostics["n_evaluations"], budget)


# Captured once from DiscreteMLSL on the quadratic fixture. If this literal
# fails, restore the historical ordering rather than editing the oracle.
_COLD_MLSL_TRACE: tuple[tuple[tuple[int, ...], float], ...] = (
    ((3, 3), 1.0),
    ((1, 0), 10.0),
    ((0, 5), 8.0),
    ((4, 2), 5.0),
    ((2, 3), 0.0),
    ((4, 3), 4.0),
    ((3, 2), 2.0),
    ((3, 4), 2.0),
    ((1, 3), 1.0),
    ((2, 2), 1.0),
    ((2, 4), 1.0),
    ((1, 5), 5.0),
    ((0, 4), 5.0),
    ((1, 4), 2.0),
    ((0, 3), 4.0),
    ((2, 4), 1.0),
    ((1, 3), 1.0),
    ((2, 3), 0.0),
    ((1, 2), 2.0),
    ((3, 3), 1.0),
    ((2, 2), 1.0),
    ((0, 0), 13.0),
    ((2, 0), 9.0),
    ((1, 1), 5.0),
    ((0, 1), 8.0),
    ((2, 1), 4.0),
    ((1, 2), 2.0),
    ((0, 2), 5.0),
    ((2, 2), 1.0),
    ((1, 3), 1.0),
    ((0, 3), 4.0),
    ((2, 3), 0.0),
    ((1, 4), 2.0),
    ((3, 3), 1.0),
    ((2, 4), 1.0),
    ((4, 4), 5.0),
    ((0, 1), 8.0),
    ((2, 4), 1.0),
    ((3, 0), 10.0),
    ((1, 4), 2.0),
)

# InitialDesign vectors are length 9, so this warm fixture embeds the cold
# quadratic in the first two coordinates and pins the other seven at 0.
_WARM_MLSL_TRACE: tuple[tuple[tuple[int, ...], float], ...] = (
    ((2, 3, 0, 0, 0, 0, 0, 0, 0), 0.0),
    ((2, 3, 0, 0, 0, 0, 0, 0, 0), 0.0),
    ((0, 0, 0, 0, 0, 0, 0, 0, 0), 13.0),
    ((1, 1, 0, 0, 0, 0, 0, 0, 0), 5.0),
    ((1, 3, 0, 0, 0, 0, 0, 0, 0), 1.0),
    ((3, 3, 0, 0, 0, 0, 0, 0, 0), 1.0),
    ((2, 2, 0, 0, 0, 0, 0, 0, 0), 1.0),
    ((2, 4, 0, 0, 0, 0, 0, 0, 0), 1.0),
    ((0, 1, 0, 0, 0, 0, 0, 0, 0), 8.0),
    ((2, 1, 0, 0, 0, 0, 0, 0, 0), 4.0),
    ((1, 0, 0, 0, 0, 0, 0, 0, 0), 10.0),
    ((1, 2, 0, 0, 0, 0, 0, 0, 0), 2.0),
    ((0, 2, 0, 0, 0, 0, 0, 0, 0), 5.0),
    ((2, 2, 0, 0, 0, 0, 0, 0, 0), 1.0),
    ((1, 3, 0, 0, 0, 0, 0, 0, 0), 1.0),
    ((0, 3, 0, 0, 0, 0, 0, 0, 0), 4.0),
    ((2, 3, 0, 0, 0, 0, 0, 0, 0), 0.0),
    ((1, 4, 0, 0, 0, 0, 0, 0, 0), 2.0),
    ((3, 3, 0, 0, 0, 0, 0, 0, 0), 1.0),
    ((2, 4, 0, 0, 0, 0, 0, 0, 0), 1.0),
    ((3, 3, 0, 0, 0, 0, 0, 0, 0), 1.0),
    ((2, 1, 0, 0, 0, 0, 0, 0, 0), 4.0),
    ((0, 4, 0, 0, 0, 0, 0, 0, 0), 5.0),
    ((3, 2, 0, 0, 0, 0, 0, 0, 0), 2.0),
)


def _observed_trace(problem: QuadraticProblem) -> list[tuple[tuple[int, ...], float]]:
    return [
        (
            tuple(call),
            float((call[0] - 2) ** 2 + (call[1] - 3) ** 2),
        )
        for call in problem.calls
    ]


class TestHistoricalMlslTraces(unittest.TestCase):
    def test_cold_quadratic_trace_is_literal(self) -> None:
        def run() -> tuple[list[tuple[tuple[int, ...], float]], dict[str, object]]:
            problem = QuadraticProblem([0, 0], [4, 5])
            algorithm = DiscreteMLSL(
                max_evaluations=40,
                seed=3,
                batch_size=4,
                reduction_fraction=1.0,
                sigma=2.0,
            )
            algorithm(problem)
            return _observed_trace(problem), algorithm.last_run_stats

        first, stats = run()
        second, _repeated = run()
        self.assertEqual(first, list(_COLD_MLSL_TRACE))
        self.assertEqual(second, list(_COLD_MLSL_TRACE))
        self.assertEqual(stats["n_evaluations"], len(_COLD_MLSL_TRACE))
        self.assertEqual(stats["best_x"], [2, 3])
        self.assertEqual(stats["best_f"], 0.0)

        again = DiscreteMLSL(
            max_evaluations=40,
            seed=3,
            batch_size=4,
            reduction_fraction=1.0,
            sigma=2.0,
        )
        first_problem = QuadraticProblem([0, 0], [4, 5])
        second_problem = QuadraticProblem([0, 0], [4, 5])
        again(first_problem)
        again(second_problem)
        self.assertEqual(_observed_trace(first_problem), list(_COLD_MLSL_TRACE))
        self.assertEqual(_observed_trace(second_problem), list(_COLD_MLSL_TRACE))

    def test_warm_prefix_trace_is_literal(self) -> None:
        """Pin the historical warm trace, including one duplicated prefix row.

        ``InitialDesign`` stores length-9 catalogue vectors, so this fixture
        uses the cold quadratic on the first two coordinates and fixes the
        other seven coordinates at 0. A length-2 vector is rejected before
        evaluation.
        """
        lower = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        upper = [4, 5, 0, 0, 0, 0, 0, 0, 0]
        prefix = [
            [2, 3, 0, 0, 0, 0, 0, 0, 0],
            [2, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "problem_id": "L8100_B2",
                        "n": 4,
                        "vectors": prefix,
                    }
                ),
                encoding="utf-8",
            )

            def run() -> tuple[list[tuple[tuple[int, ...], float]], dict[str, object]]:
                problem = QuadraticProblem(lower, upper)
                algorithm = DiscreteMLSLWarmStart(
                    max_evaluations=24,
                    seed=3,
                    batch_size=4,
                    reduction_fraction=1.0,
                    sigma=2.0,
                    initial_design_path=str(path),
                )
                algorithm(problem)
                return _observed_trace(problem), algorithm.last_run_stats

            first, stats = run()
            second, _repeated = run()

        self.assertEqual(first, list(_WARM_MLSL_TRACE))
        self.assertEqual(second, list(_WARM_MLSL_TRACE))
        self.assertEqual(first[0], first[1])
        self.assertEqual(stats["n_evaluations"], len(_WARM_MLSL_TRACE))
        self.assertEqual(stats["n_global_samples"], 8)
        self.assertEqual(stats["n_candidates_already_started"], 1)
        starts = [tuple(point) for point in stats["local_search_start_points"]]
        self.assertEqual(len(starts), len(set(starts)))
        self.assertEqual(stats["best_x"], [2, 3, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(stats["best_f"], 0.0)


class TestSobolMultistart(unittest.TestCase):
    """Unfiltered Sobol sampling followed by the shared local searches."""

    def test_local_search_functions_are_the_shared_ones(self) -> None:
        strict = SobolMultistart(max_evaluations=4, seed=1)
        equal = SobolMultistartEquDesc(max_evaluations=4, seed=1)
        self.assertIs(strict.local_search, best_improvement_local_search)
        self.assertIs(equal.local_search, equal_descent_local_search)
        self.assertIs(
            DiscreteMLSL(max_evaluations=4, seed=1).local_search,
            best_improvement_local_search,
        )
        problem = FlatProblem([0, 0], [3, 3])
        space = DiscreteSearchSpace(problem.bounds.lb, problem.bounds.ub)
        start = EvaluatedPoint((1, 1), 1.0)
        direct = best_improvement_local_search(start, _manager(problem, 20), space)
        again = FlatProblem([0, 0], [3, 3])
        shared = strict.local_search(start, _manager(again, 20), space)
        self.assertEqual(direct.point.x, shared.point.x)
        self.assertEqual(direct.certified_local_optimum, shared.certified_local_optimum)
        self.assertEqual(direct.n_evaluations, shared.n_evaluations)

    def test_every_accepted_sample_starts_a_search(self) -> None:
        problem = FlatProblem([0, 0], [6, 6])
        algorithm = SobolMultistart(max_evaluations=40, seed=2)
        algorithm(problem)
        stats = algorithm.last_run_stats
        self.assertEqual(stats["n_evaluations"], 40)
        self.assertGreater(stats["n_global_samples"], 1)
        self.assertEqual(stats["n_candidates_radius_rejected"], 0)
        self.assertEqual(stats["n_reduced_candidates"], 0)
        self.assertEqual(stats["selection"], "none")
        # A sample that consumes the last evaluation cannot open a neighbourhood.
        self.assertIn(
            stats["n_global_samples"] - stats["n_local_searches"],
            (0, 1),
        )
        self.assertEqual(
            len(stats["local_search_start_points"]),
            stats["n_local_searches"],
        )
        filtered = DiscreteMLSL(
            max_evaluations=40,
            seed=2,
            batch_size=8,
            reduction_fraction=0.10,
        )
        filtered(FlatProblem([0, 0], [6, 6]))
        self.assertLess(
            filtered.last_run_stats["n_local_searches"],
            filtered.last_run_stats["n_global_samples"],
        )

    def test_first_sample_matches_mlsl_and_then_descends(self) -> None:
        mlsl = DiscreteMLSL(max_evaluations=1, seed=7, batch_size=8)
        sobol = SobolMultistart(max_evaluations=1, seed=7)
        mlsl_problem = QuadraticProblem([0, 0], [4, 5])
        sobol_problem = QuadraticProblem([0, 0], [4, 5])
        mlsl(mlsl_problem)
        sobol(sobol_problem)
        self.assertEqual(mlsl_problem.calls, sobol_problem.calls)
        self.assertEqual(sobol.last_run_stats["n_local_searches"], 0)

        descended = SobolMultistart(max_evaluations=6, seed=7)
        descended_problem = QuadraticProblem([0, 0], [4, 5])
        descended(descended_problem)
        self.assertEqual(descended_problem.calls[0], sobol_problem.calls[0])
        self.assertNotEqual(descended_problem.calls[1], mlsl_problem.calls[0])
        space = DiscreteSearchSpace([0, 0], [4, 5])
        self.assertIn(
            tuple(descended_problem.calls[1]),
            space.neighbours_l1(tuple(descended_problem.calls[0])),
        )
        self.assertLessEqual(len(descended_problem.calls), 6)

    def test_duplicate_projection_is_not_evaluated(self) -> None:
        sequence = iter([(1, 1), (1, 1), (2, 2)])

        def draw(self: DiscreteSampler) -> tuple[int, ...]:
            return next(sequence)

        with patch.object(DiscreteSampler, "draw", draw):
            problem = FlatProblem([0, 0], [3, 3])
            algorithm = SobolMultistart(max_evaluations=10, seed=1)
            algorithm(problem)
        stats = algorithm.last_run_stats
        self.assertEqual(stats["n_duplicate_projections_skipped"], 1)
        self.assertEqual(stats["n_global_samples"], 2)
        self.assertEqual(stats["n_local_searches"], 2)
        self.assertEqual(problem.calls.count([1, 1]), 1)

    def test_budget_target_and_reproduction(self) -> None:
        def calls(seed: int) -> list[list[int]]:
            problem = QuadraticProblem([0, 0], [4, 5])
            SobolMultistart(max_evaluations=25, seed=seed)(problem)
            self.assertLessEqual(len(problem.calls), 25)
            return problem.calls

        self.assertEqual(calls(4), calls(4))
        self.assertNotEqual(calls(4), calls(5))

        problem = QuadraticProblem([0, 0], [4, 5])
        controller = RunController(
            problem,
            80,
            mode="fixed_target",
            target_threshold=0.0,
            target_reference="all",
        )
        SobolMultistart(max_evaluations=80, seed=3)(
            problem,
            run_controller=controller,
        )
        self.assertEqual(controller.stop_reason(), "target_hit")
        self.assertLess(controller.used, 80)
        self.assertEqual(problem.state.current_best.y, 0.0)

    def test_factory_and_diagnostics_are_log_compatible(self) -> None:
        algorithm = create_algorithm(
            "SOBOL_MULTISTART_PEN_SI_DESC_COLD_START",
            {"max_evaluations": 12},
            random_seed=1,
        )
        problem = QuadraticProblem([0, 0], [4, 5])
        algorithm(problem)
        diagnostics = algorithm.summary_diagnostics()
        self.assertEqual(diagnostics["selection"], "none")
        self.assertEqual(diagnostics["n_candidates_radius_rejected"], 0)
        self.assertEqual(
            diagnostics["n_evaluations"],
            diagnostics["n_global_samples"]
            + sum(diagnostics["evaluations_per_local_search"]),
        )
        json.dumps(diagnostics)
        equal = create_algorithm(
            "SOBOL_MULTISTART_PEN_EQU_DESC_COLD_START",
            {"max_evaluations": 12},
            random_seed=1,
        )
        self.assertIs(equal.local_search, equal_descent_local_search)


class TestSobolSample(unittest.TestCase):
    """Scrambled Sobol with the multistart stream and no local search."""

    def test_samples_follow_the_shared_sobol_stream(self) -> None:
        problem = QuadraticProblem([0, 0], [4, 5])
        algorithm = SobolSample(max_evaluations=4, seed=11)
        algorithm(problem)
        space = DiscreteSearchSpace(problem.bounds.lb, problem.bounds.ub)
        sampler = DiscreteSampler(space, "sobol", 11, np.random.default_rng(11))
        expected = [list(sampler.draw()) for _ in range(4)]
        self.assertEqual(problem.calls, expected)
        stats = algorithm.summary_diagnostics()
        self.assertEqual(stats["n_evaluations"], 4)
        self.assertEqual(stats["n_global_samples"], 4)
        self.assertEqual(stats["n_local_searches"], 0)
        self.assertEqual(stats["local_search"], "none")
        self.assertEqual(stats["local_search_start_points"], [])
        self.assertEqual(stats["selection"], "none")
        json.dumps(stats)

    def test_first_sample_matches_multistart_and_no_neighbour_is_opened(self) -> None:
        sampled = QuadraticProblem([0, 0], [4, 5])
        descended = QuadraticProblem([0, 0], [4, 5])
        SobolSample(max_evaluations=1, seed=9)(sampled)
        SobolMultistart(max_evaluations=1, seed=9)(descended)
        self.assertEqual(sampled.calls, descended.calls)

        longer = QuadraticProblem([0, 0], [4, 5])
        with patch(
            "core.ioh_core.algorithms.native.discrete_search.best_improvement_local_search",
            side_effect=AssertionError("local search must not run"),
        ):
            SobolSample(max_evaluations=3, seed=9)(longer)
        self.assertEqual(len(longer.calls), 3)
        self.assertEqual(longer.calls[0], sampled.calls[0])
        space = DiscreteSearchSpace(longer.bounds.lb, longer.bounds.ub)
        self.assertNotIn(tuple(longer.calls[1]), space.neighbours_l1(tuple(longer.calls[0])))

    def test_duplicate_projection_is_not_evaluated(self) -> None:
        sequence = iter([(1, 1), (1, 1), (2, 2)])

        def draw(self: DiscreteSampler) -> tuple[int, ...]:
            return next(sequence)

        with patch.object(DiscreteSampler, "draw", draw):
            problem = FlatProblem([0, 0], [3, 3])
            algorithm = SobolSample(max_evaluations=2, seed=1)
            algorithm(problem)
        stats = algorithm.summary_diagnostics()
        self.assertEqual(stats["n_duplicate_projections_skipped"], 1)
        self.assertEqual(stats["n_global_samples"], 2)
        self.assertEqual(stats["n_evaluations"], 2)
        self.assertEqual(problem.calls, [[1, 1], [2, 2]])

    def test_budget_target_and_distinct_token_seeds(self) -> None:
        problem = QuadraticProblem([0, 0], [4, 5])
        first = SobolSample(max_evaluations=8, seed=4)
        first(problem)
        again = QuadraticProblem([0, 0], [4, 5])
        SobolSample(max_evaluations=8, seed=4)(again)
        other = QuadraticProblem([0, 0], [4, 5])
        SobolSample(max_evaluations=8, seed=5)(other)
        self.assertEqual(problem.calls, again.calls)
        self.assertNotEqual(problem.calls, other.calls)
        self.assertLessEqual(len(problem.calls), 8)

        targeted = QuadraticProblem([0, 0], [4, 5])
        controller = RunController(
            targeted,
            80,
            mode="fixed_target",
            target_threshold=0.0,
            target_reference="all",
        )
        SobolSample(max_evaluations=80, seed=3)(targeted, run_controller=controller)
        self.assertEqual(controller.stop_reason(), "target_hit")
        self.assertLess(controller.used, 80)
        self.assertEqual(targeted.state.current_best.y, 0.0)

        from core.ioh_core.workflows.optimization_runs.planner import derive_random_seed

        tokens = (
            "SOBOL_PEN_COLD_START",
            "SOBOL_MULTISTART_PEN_SI_DESC_COLD_START",
            "SOBOL_MULTISTART_PEN_EQU_DESC_COLD_START",
            "DISCRETE_MLSL_PEN_SI_DESC_COLD_START",
        )
        seeds = [
            derive_random_seed(
                seed_base=42,
                slab_type="hp_slab",
                problem_id="L6750_B2",
                algorithm_name=token,
                repetition=0,
            )
            for token in tokens
        ]
        self.assertEqual(len(set(seeds)), len(tokens))
        created = create_algorithm("SOBOL_PEN_COLD_START", {"max_evaluations": 4}, random_seed=seeds[0])
        self.assertIsInstance(created, SobolSample)
        self.assertFalse(hasattr(created, "local_search"))


if __name__ == "__main__":
    unittest.main()
