"""Constraint-aware PBH-L1-Deb unit and boundary tests."""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from core.ioh_core.algorithm_factory import create_algorithm
from core.ioh_core.algorithms.common import RunController
from core.ioh_core.algorithms.native.discrete_search import (
    DiscreteSearchSpace,
    PBHDebEvaluatedPoint,
    PBHDebEvaluationManager,
    PopulationBasinHoppingDeb,
    PopulationBasinHoppingDebWarmStart,
    apply_population_replacement_deb,
    pbh_deb_best_improvement_local_search,
    pbh_deb_key,
    pbh_strictly_better,
    validate_population_basin_hopping_deb_parameters,
    validate_population_basin_hopping_parameters,
)
from core.ioh_core.algorithms.registry import RegistryError, get_algorithm_spec
from core.ioh_core.workflows.la.deb import deb_key
from core.ioh_core.workflows.optimization_runs.io_logging import (
    _validated_algorithm_diagnostics,
)


class _DebProblem:
    """Small IOH-like problem publishing configurable auxiliary records."""

    def __init__(
        self,
        records: dict[tuple[int, ...], dict[str, Any]],
        *,
        lower: list[int],
        upper: list[int],
        constraint_names: list[str] | None = None,
        invalid_utilization: float = 99.0,
    ) -> None:
        self.records = records
        self.bounds = SimpleNamespace(lb=lower, ub=upper)
        self.meta_data = SimpleNamespace(n_variables=len(lower))
        self.state = SimpleNamespace(
            evaluations=0,
            current_best=SimpleNamespace(y=math.inf, x=[]),
        )
        self.calls: list[tuple[int, ...]] = []
        self.hdb_eval_context = SimpleNamespace(
            _c_names=["g"] if constraint_names is None else constraint_names,
            run_config=SimpleNamespace(invalid_utilization=invalid_utilization),
            last_data=None,
        )

    def __call__(self, x: Any) -> float:
        point = tuple(int(value) for value in x)
        self.calls.append(point)
        self.state.evaluations += 1
        record = dict(self.records[point])
        self.hdb_eval_context.last_data = record
        y_p = float(record["y_p"])
        if math.isfinite(y_p) and y_p < self.state.current_best.y:
            self.state.current_best = SimpleNamespace(y=y_p, x=list(point))
        return y_p


def _record(
    *,
    y_p: float,
    y: float,
    violation: float,
    analysis_ok: bool = True,
    feasible: bool | None = None,
    utilization: float | None = None,
) -> dict[str, Any]:
    if feasible is None:
        feasible = bool(analysis_ok and violation == 0.0)
    if utilization is None:
        utilization = violation + 1.0
    return {
        "y_p": y_p,
        "y": y,
        "analysis_ok": analysis_ok,
        "feasible": feasible,
        "violations_": {"g": violation},
        "constraint_values": {"g": utilization},
    }


def _point(
    x: tuple[int, ...],
    *,
    y: float,
    phi: float,
    feasible: bool,
) -> PBHDebEvaluatedPoint:
    return PBHDebEvaluatedPoint(
        x=x,
        y_p=y,
        y=y,
        phi=phi,
        feasible=feasible,
        analysis_ok=True,
        sentinel_constraints=(),
    )


class TestDebOrdering(unittest.TestCase):
    def test_feasibility_rules_and_strict_plateau_tie(self) -> None:
        feasible = _point((2,), y=50.0, phi=0.0, feasible=True)
        infeasible = _point((0,), y=1.0, phi=0.1, feasible=False)
        lower_phi = _point((1,), y=100.0, phi=0.05, feasible=False)
        tied_phi = _point((3,), y=0.0, phi=0.1, feasible=False)

        self.assertEqual(pbh_deb_key(feasible), (0, 50.0))
        self.assertEqual(pbh_deb_key(infeasible), (1, 0.1))
        self.assertTrue(pbh_strictly_better(feasible, infeasible))
        self.assertTrue(pbh_strictly_better(lower_phi, infeasible))
        self.assertFalse(pbh_strictly_better(tied_phi, infeasible))

    def test_key_matches_landscape_analysis_definition(self) -> None:
        for point in (
            _point((0,), y=3.0, phi=0.0, feasible=True),
            _point((1,), y=1.0, phi=0.4, feasible=False),
        ):
            self.assertEqual(
                pbh_deb_key(point),
                deb_key(
                    feasible=point.feasible,
                    y=point.y,
                    phi=point.phi,
                ),
            )


class TestDebEvaluationManager(unittest.TestCase):
    def _manager(
        self,
        problem: _DebProblem,
        *,
        budget: int = 10,
        mode: str = "fixed_budget",
        target: float | None = None,
    ) -> tuple[PBHDebEvaluationManager, RunController]:
        controller = RunController(
            problem,
            budget,
            mode=mode,
            target_threshold=target,
            target_reference="all" if mode == "fixed_target" else None,
        )
        space = DiscreteSearchSpace(problem.bounds.lb, problem.bounds.ub)
        return PBHDebEvaluationManager(problem, controller, space), controller

    def test_valid_records_update_benchmark_and_deb_incumbents(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(y_p=1.0, y=10.0, violation=0.4),
                (1,): _record(y_p=2.0, y=20.0, violation=0.0),
            },
            lower=[0],
            upper=[1],
        )
        manager, _controller = self._manager(problem)

        first = manager.evaluate((0,))
        second = manager.evaluate((1,))

        self.assertFalse(first.feasible)
        self.assertEqual(first.phi, 0.4)
        self.assertIs(manager.best_benchmark_point, first)
        self.assertIs(manager.best_deb_point, second)
        self.assertEqual(manager.diagnostics["n_infeasible_evaluations"], 1)
        self.assertEqual(manager.diagnostics["n_feasible_evaluations"], 1)

    def test_active_constraint_names_must_be_a_nonempty_list(self) -> None:
        problem = _DebProblem(
            {(0,): _record(y_p=1.0, y=1.0, violation=0.0)},
            lower=[0],
            upper=[0],
        )
        problem.hdb_eval_context._c_names = ("g",)
        with self.assertRaisesRegex(ValueError, "active constraint name"):
            self._manager(problem)
        self.assertEqual(problem.calls, [])

    def test_failed_analysis_is_not_a_deb_incumbent(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(
                    y_p=7.0,
                    y=4.0,
                    violation=98.0,
                    analysis_ok=False,
                )
            },
            lower=[0],
            upper=[0],
        )
        manager, _controller = self._manager(problem)

        point = manager.evaluate((0,))

        self.assertFalse(point.analysis_ok)
        self.assertTrue(math.isinf(point.phi))
        self.assertIsNone(manager.best_deb_point)
        self.assertEqual(manager.diagnostics["n_failed_analysis_evaluations"], 1)
        self.assertEqual(manager.diagnostics["n_infeasible_evaluations"], 0)

    def test_scalar_is_validated_before_controller_observation(self) -> None:
        problem = _DebProblem(
            {(0,): _record(y_p=math.nan, y=1.0, violation=0.0)},
            lower=[0],
            upper=[0],
        )
        manager, controller = self._manager(
            problem,
            mode="fixed_target",
            target=10.0,
        )

        with self.assertRaisesRegex(RuntimeError, "NaN or -inf"):
            manager.evaluate((0,))
        self.assertFalse(controller.target_hit)

    def test_positive_infinity_is_allowed_and_ignored_by_controller(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(
                    y_p=math.inf,
                    y=math.inf,
                    violation=98.0,
                    analysis_ok=False,
                )
            },
            lower=[0],
            upper=[0],
        )
        manager, controller = self._manager(problem)

        point = manager.evaluate((0,))

        self.assertTrue(math.isinf(point.y_p))
        self.assertIsNone(manager.best_benchmark_point)
        self.assertEqual(controller.best_finite_y_p, math.inf)

    def test_exact_active_constraint_contract_is_required(self) -> None:
        record = _record(y_p=1.0, y=1.0, violation=0.0)
        record["violations_"] = {}
        problem = _DebProblem({(0,): record}, lower=[0], upper=[0])
        manager, _controller = self._manager(problem)
        with self.assertRaisesRegex(ValueError, "exactly"):
            manager.evaluate((0,))

        record = _record(y_p=1.0, y=1.0, violation=0.0)
        record["violations_"]["extra"] = 0.0
        problem = _DebProblem({(0,): record}, lower=[0], upper=[0])
        manager, _controller = self._manager(problem)
        with self.assertRaisesRegex(ValueError, "exactly"):
            manager.evaluate((0,))

    def test_analysis_valid_malformed_violations_remain_fatal(self) -> None:
        record = _record(y_p=1.0, y=1.0, violation=0.0)
        record["violations_"] = {}
        problem = _DebProblem({(0,): record}, lower=[0], upper=[0])
        manager, _controller = self._manager(problem)
        with self.assertRaisesRegex(ValueError, "exactly"):
            manager.evaluate((0,))
        self.assertIsNone(manager.best_deb_point)

    def test_malformed_failed_analysis_contents_are_ignored(self) -> None:
        record = _record(
            y_p=1.5,
            y=4.0,
            violation=98.0,
            analysis_ok=False,
            feasible=False,
        )
        record["violations_"] = {}
        record["constraint_values"] = {"extra": "not-a-number"}
        problem = _DebProblem({(0,): record}, lower=[0], upper=[0])
        manager, _controller = self._manager(problem)

        point = manager.evaluate((0,))

        self.assertEqual(manager.n_evaluations, 1)
        self.assertFalse(point.analysis_ok)
        self.assertTrue(math.isinf(point.phi))
        self.assertIsNone(manager.best_deb_point)
        self.assertIs(manager.best_benchmark_point, point)
        self.assertEqual(point.y_p, 1.5)
        self.assertEqual(manager.diagnostics["n_sentinel_evaluations"], 0)
        self.assertEqual(
            manager.diagnostics["sentinel_evaluations_by_constraint"]["g"],
            0,
        )
        self.assertEqual(manager.diagnostics["n_failed_analysis_evaluations"], 1)

    def test_failed_analysis_with_feasible_true_is_fatal(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(
                    y_p=1.0,
                    y=1.0,
                    violation=98.0,
                    analysis_ok=False,
                    feasible=True,
                )
            },
            lower=[0],
            upper=[0],
        )
        manager, _controller = self._manager(problem)
        with self.assertRaisesRegex(ValueError, "feasible"):
            manager.evaluate((0,))
        self.assertIsNone(manager.best_deb_point)
        self.assertIsNone(manager.best_benchmark_point)

    def test_feasible_flag_must_match_valid_violations(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(
                    y_p=1.0,
                    y=1.0,
                    violation=0.2,
                    feasible=True,
                )
            },
            lower=[0],
            upper=[0],
        )
        manager, _controller = self._manager(problem)
        with self.assertRaisesRegex(ValueError, "feasible"):
            manager.evaluate((0,))

    def test_sentinel_constraints_are_diagnosed_without_changing_phi(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(
                    y_p=1.0,
                    y=1.0,
                    violation=98.0,
                    utilization=99.0,
                )
            },
            lower=[0],
            upper=[0],
        )
        manager, _controller = self._manager(problem)
        point = manager.evaluate((0,))
        self.assertEqual(point.phi, 98.0)
        self.assertEqual(point.sentinel_constraints, ("g",))
        self.assertEqual(manager.diagnostics["n_sentinel_evaluations"], 1)


class TestDebLocalSearch(unittest.TestCase):
    def test_violation_descent_crosses_to_feasible_and_reuses_cache(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(y_p=40.0, y=1.0, violation=0.4),
                (1,): _record(y_p=30.0, y=100.0, violation=0.2),
                (2,): _record(y_p=20.0, y=10.0, violation=0.0),
                (3,): _record(y_p=10.0, y=5.0, violation=0.0),
            },
            lower=[0],
            upper=[3],
        )
        manager, _controller = TestDebEvaluationManager()._manager(
            problem,
            budget=10,
        )
        result = pbh_deb_best_improvement_local_search(
            manager.evaluate((0,)),
            manager,
            manager.space,
        )
        self.assertEqual(result.point.x, (3,))
        self.assertEqual(result.n_steps, 3)
        self.assertTrue(result.certified_local_optimum)
        self.assertEqual(problem.calls, [(0,), (1,), (2,), (3,)])

    def test_incomplete_neighbourhood_keeps_resolved_winner(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(y_p=1.0, y=1.0, violation=0.0),
                (1,): _record(y_p=5.0, y=5.0, violation=0.0),
                (2,): _record(y_p=9.0, y=9.0, violation=0.0),
            },
            lower=[0],
            upper=[2],
        )
        manager, _controller = TestDebEvaluationManager()._manager(
            problem,
            budget=2,
        )
        result = pbh_deb_best_improvement_local_search(
            manager.evaluate((1,)),
            manager,
            manager.space,
        )
        self.assertEqual(result.point.x, (0,))
        self.assertFalse(result.certified_local_optimum)
        self.assertEqual(problem.calls, [(1,), (0,)])

    def test_equal_phi_does_not_move_and_counts_plateau(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(y_p=5.0, y=5.0, violation=0.2),
                (1,): _record(y_p=1.0, y=1.0, violation=0.2),
            },
            lower=[0],
            upper=[1],
        )
        controller = RunController(problem, 2)
        manager = PBHDebEvaluationManager(
            problem,
            controller,
            DiscreteSearchSpace([0], [1]),
        )
        start = manager.evaluate((0,))

        result = pbh_deb_best_improvement_local_search(
            start,
            manager,
            DiscreteSearchSpace([0], [1]),
        )

        self.assertTrue(result.certified_local_optimum)
        self.assertEqual(result.point.x, (0,))
        self.assertEqual(manager.diagnostics["n_infeasible_plateau_terminations"], 1)

    def test_final_budget_evaluation_can_certify_completed_neighbourhood(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(y_p=1.0, y=1.0, violation=0.0),
                (1,): _record(y_p=2.0, y=2.0, violation=0.0),
            },
            lower=[0],
            upper=[1],
        )
        controller = RunController(problem, 2)
        manager = PBHDebEvaluationManager(
            problem,
            controller,
            DiscreteSearchSpace([0], [1]),
        )
        start = manager.evaluate((0,))
        result = pbh_deb_best_improvement_local_search(
            start,
            manager,
            DiscreteSearchSpace([0], [1]),
        )
        self.assertTrue(result.certified_local_optimum)

    def test_final_budget_improvement_opens_uncertified_neighbourhood(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(y_p=2.0, y=2.0, violation=0.0),
                (1,): _record(y_p=1.0, y=1.0, violation=0.0),
            },
            lower=[0],
            upper=[1],
        )
        manager, _controller = TestDebEvaluationManager()._manager(
            problem,
            budget=2,
        )

        result = pbh_deb_best_improvement_local_search(
            manager.evaluate((0,)),
            manager,
            manager.space,
        )

        self.assertEqual(result.point.x, (1,))
        self.assertEqual(result.n_steps, 1)
        self.assertFalse(result.certified_local_optimum)

    def test_target_hit_inside_neighbourhood_is_uncertified(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(y_p=5.0, y=5.0, violation=0.0),
                (1,): _record(y_p=1.0, y=1.0, violation=0.0),
            },
            lower=[0],
            upper=[1],
        )
        controller = RunController(
            problem,
            5,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        manager = PBHDebEvaluationManager(
            problem,
            controller,
            DiscreteSearchSpace([0], [1]),
        )
        start = manager.evaluate((0,))
        result = pbh_deb_best_improvement_local_search(
            start,
            manager,
            DiscreteSearchSpace([0], [1]),
        )
        self.assertTrue(controller.target_hit)
        self.assertFalse(result.certified_local_optimum)
        self.assertEqual(problem.calls, [(0,), (1,)])

    def test_failed_endpoint_is_never_certified(self) -> None:
        problem = _DebProblem(
            {
                (0,): _record(
                    y_p=2.0,
                    y=2.0,
                    violation=98.0,
                    analysis_ok=False,
                )
            },
            lower=[0],
            upper=[0],
        )
        controller = RunController(problem, 1)
        manager = PBHDebEvaluationManager(
            problem,
            controller,
            DiscreteSearchSpace([0], [0]),
        )
        start = manager.evaluate((0,))
        result = pbh_deb_best_improvement_local_search(
            start,
            manager,
            DiscreteSearchSpace([0], [0]),
        )
        self.assertFalse(result.certified_local_optimum)


class TestDebPopulationReplacement(unittest.TestCase):
    def test_near_and_diverse_replacement_use_deb_order(self) -> None:
        space = DiscreteSearchSpace([0], [10])
        infeasible = _point((0,), y=1.0, phi=1.0, feasible=False)
        feasible_worse_y = _point((10,), y=100.0, phi=0.0, feasible=True)
        child = _point((1,), y=50.0, phi=0.0, feasible=True)
        population = [infeasible, feasible_worse_y]

        self.assertEqual(
            apply_population_replacement_deb(child, population, space, 0.2),
            "near",
        )
        self.assertIn(child, population)

        better_feasible = _point((5,), y=40.0, phi=0.0, feasible=True)
        self.assertEqual(
            apply_population_replacement_deb(
                better_feasible,
                population,
                space,
                0.0,
            ),
            "diverse",
        )

    def test_one_member_population_uses_near_branch_with_infinite_dcut(self) -> None:
        space = DiscreteSearchSpace([0], [2])
        population = [_point((0,), y=5.0, phi=0.0, feasible=True)]
        child = _point((2,), y=4.0, phi=0.0, feasible=True)
        self.assertEqual(
            apply_population_replacement_deb(
                child,
                population,
                space,
                math.inf,
            ),
            "near",
        )

    def test_dcut_equality_worst_tie_and_rejection_are_strict(self) -> None:
        space = DiscreteSearchSpace([0], [10])
        left = _point((0,), y=5.0, phi=0.0, feasible=True)
        right = _point((3,), y=5.0, phi=0.0, feasible=True)
        child = _point((10,), y=4.0, phi=0.0, feasible=True)
        population = [left, right]
        self.assertEqual(
            apply_population_replacement_deb(child, population, space, 0.7),
            "diverse",
        )
        self.assertEqual([point.x for point in population], [(0,), (10,)])
        self.assertEqual(
            apply_population_replacement_deb(
                _point((5,), y=5.0, phi=0.0, feasible=True),
                population,
                space,
                0.0,
            ),
            "rejected",
        )


class TestPopulationBasinHoppingDeb(unittest.TestCase):
    def test_registry_constructs_both_new_algorithms(self) -> None:
        cold = create_algorithm(
            "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
            {"max_evaluations": 2, "population_size": 1},
            random_seed=1,
        )
        warm = create_algorithm(
            "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
            {
                "max_evaluations": 2,
                "population_size": 1,
                "initial_design_path": "initial_design.json",
            },
            random_seed=1,
        )
        self.assertIsInstance(cold, PopulationBasinHoppingDeb)
        self.assertIsInstance(warm, PopulationBasinHoppingDebWarmStart)
        self.assertTrue(cold.info.startswith("PBH-L1-Deb "))
        self.assertTrue(warm.info.startswith("PBH-L1-Deb-WarmStart "))
        self.assertEqual(
            get_algorithm_spec("POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START").validator_ref,
            "core.ioh_core.algorithms.native.discrete_search:"
            "validate_population_basin_hopping_deb_parameters",
        )

    def test_shared_registry_validator_preserves_token_messages(self) -> None:
        parameters = {"perturbation_fraction": 0, "dcut_factor": 1}
        with self.assertRaisesRegex(
            RegistryError,
            "^POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START: perturbation_fraction",
        ):
            validate_population_basin_hopping_parameters(parameters)
        with self.assertRaisesRegex(
            RegistryError,
            "^POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START: perturbation_fraction",
        ):
            validate_population_basin_hopping_deb_parameters(parameters)

    def test_cold_target_hit_stops_before_local_search(self) -> None:
        records = {
            (0,): _record(y_p=1.0, y=1.0, violation=0.0),
            (1,): _record(y_p=1.0, y=1.0, violation=0.0),
        }
        problem = _DebProblem(records, lower=[0], upper=[1])
        controller = RunController(
            problem,
            5,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        algorithm = PopulationBasinHoppingDeb(
            max_evaluations=5,
            seed=3,
            population_size=1,
            initial_sampler="uniform",
        )
        algorithm(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 1)
        self.assertEqual(algorithm.last_run_stats["n_local_searches"], 0)
        self.assertEqual(
            algorithm.summary_diagnostics()["initial_population_size"],
            0,
        )

    def test_target_hit_during_warm_prefix_stops_without_population(self) -> None:
        first = (0,) * 9
        second = (1,) + (0,) * 8
        records = {
            first: _record(y_p=1.0, y=1.0, violation=0.0),
            second: _record(y_p=2.0, y=2.0, violation=0.0),
        }
        problem = _DebProblem(records, lower=[0] * 9, upper=[1] * 9)
        controller = RunController(
            problem,
            5,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "problem_id": "L8100_B2",
                        "n": 2,
                        "vectors": [list(first), list(second)],
                    }
                ),
                encoding="utf-8",
            )
            algorithm = PopulationBasinHoppingDebWarmStart(
                max_evaluations=5,
                seed=1,
                population_size=2,
                initial_design_path=str(path),
            )
            algorithm(problem, run_controller=controller)

        self.assertEqual(problem.calls, [first])
        self.assertTrue(controller.target_hit)
        self.assertEqual(
            algorithm.summary_diagnostics()["initial_population_size"],
            0,
        )
        self.assertEqual(
            algorithm.summary_diagnostics()["label"],
            "PBH-L1-Deb",
        )

    def test_all_failed_run_has_no_deb_incumbent_or_population(self) -> None:
        records = {
            (0,): _record(
                y_p=7.0,
                y=7.0,
                violation=98.0,
                analysis_ok=False,
            ),
            (1,): _record(
                y_p=8.0,
                y=8.0,
                violation=98.0,
                analysis_ok=False,
            ),
        }
        problem = _DebProblem(records, lower=[0], upper=[1])
        algorithm = PopulationBasinHoppingDeb(
            max_evaluations=2,
            seed=3,
            population_size=1,
            initial_sampler="uniform",
        )

        algorithm(problem)
        diagnostics = algorithm.summary_diagnostics()

        self.assertIsNone(diagnostics["best_deb_x"])
        self.assertIsNone(diagnostics["best_deb_y"])
        self.assertIsNone(diagnostics["best_deb_phi"])
        self.assertIsNone(diagnostics["best_deb_feasible"])
        self.assertEqual(diagnostics["initial_population_size"], 0)
        self.assertEqual(diagnostics["n_failed_analysis_evaluations"], 2)

    def test_summary_is_compact_finite_and_partitions_evaluations(self) -> None:
        records = {
            (0,): _record(y_p=3.0, y=3.0, violation=0.0),
            (1,): _record(y_p=4.0, y=4.0, violation=0.2),
        }
        problem = _DebProblem(records, lower=[0], upper=[1])
        algorithm = PopulationBasinHoppingDeb(
            max_evaluations=2,
            seed=1,
            population_size=1,
            initial_sampler="uniform",
        )
        algorithm(problem)

        diagnostics = _validated_algorithm_diagnostics(
            algorithm.summary_diagnostics()
        )
        self.assertEqual(diagnostics["label"], "PBH-L1-Deb")
        self.assertEqual(
            diagnostics["n_evaluations"],
            diagnostics["n_failed_analysis_evaluations"]
            + diagnostics["n_feasible_evaluations"]
            + diagnostics["n_infeasible_evaluations"],
        )
        self.assertNotIn("population_mean_distance_history", diagnostics)
        self.assertNotIn("evaluations_per_local_search", diagnostics)

    def test_generic_diagnostics_validation_rejects_bad_payloads(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            _validated_algorithm_diagnostics({"bad": math.inf})
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            _validated_algorithm_diagnostics({"": 1})
        with self.assertRaisesRegex(ValueError, "JSON-compatible"):
            _validated_algorithm_diagnostics({"bad": (1, 2)})


if __name__ == "__main__":
    unittest.main()
