"""Constraint-aware equal descent and the six Deb EQU_DESC algorithms."""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import numpy as np

from core.ioh_core.algorithms.common import RunController
from core.ioh_core.algorithms.native.deb_record import (
    deb_quality_key,
    validate_deb_auxiliary_record,
)
from core.ioh_core.algorithms.native.discrete_search import (
    DebEvaluatedPoint,
    DebEvaluationManager,
    DiscreteMLSL,
    DiscreteMLSLDeb,
    DiscreteMLSLDebEquDesc,
    DiscreteMLSLDebEquDescWarmStart,
    DiscreteMLSLEquDesc,
    DiscreteSampler,
    DiscreteSearchSpace,
    PopulationBasinHopping,
    PopulationBasinHoppingDeb,
    PopulationBasinHoppingDebEquDesc,
    PopulationBasinHoppingDebEquDescWarmStart,
    PopulationBasinHoppingEquDesc,
    best_improvement_local_search,
    deb_best_improvement_local_search,
    deb_equal_descent_local_search,
    equal_descent_local_search,
    pbh_deb_best_improvement_local_search,
)
from core.ioh_core.algorithms.native.factory import (
    build_discrete_mlsl_deb_equ_desc,
    build_discrete_mlsl_deb_equ_desc_warm_start,
    build_k_elite_ams_deb_equ_desc,
    build_k_elite_ams_deb_equ_desc_warm_start,
    build_population_basin_hopping_deb_equ_desc,
    build_population_basin_hopping_deb_equ_desc_warm_start,
)
from core.ioh_core.algorithms.native.k_elite_ams import (
    KEliteAMS,
    KEliteAMSDeb,
    KEliteAMSDebEquDesc,
    KEliteAMSDebWarmStart,
    KEliteAMSDebEquDescWarmStart,
    KEliteAMSEquDesc,
    _uniform_index,
)
from core.ioh_core.algorithms.registry import ALGORITHM_REGISTRY, get_algorithm_spec
from core.ioh_core.algorithms.warm_start import (
    DEB_WARM_START_TOKENS,
    INITIAL_DESIGN_PATH_TOKENS,
    RANK1_SEEDLESS_TOKENS,
    WARM_START_TOKENS,
)
from core.visualization_core.optimization.algorithm_presentation import (
    legend_handling,
    local_search_label,
    presentation_for,
)

_NAMES = ("g", "h")
_STRICT = deb_best_improvement_local_search
_EQUAL = deb_equal_descent_local_search
_DEB_EQU = (
    "DISCRETE_MLSL_DEB_EQU_DESC_COLD_START",
    "DISCRETE_MLSL_DEB_EQU_DESC_WARM_START",
    "K_ELITE_AMS_DEB_EQU_DESC_COLD_START",
    "K_ELITE_AMS_DEB_EQU_DESC_WARM_START",
    "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_COLD_START",
    "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START",
)


def _penalty(y: float, *violations: float) -> float:
    return float(y) * (1.0 + sum(value ** (1.0 / 3.0) for value in violations))


def _record(
    y: float,
    y_p: float,
    violations: dict[str, float],
    *,
    names: tuple[str, ...] = ("g",),
    analysis_ok: bool = True,
    utilizations: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not analysis_ok:
        return {
            "y_p": y_p,
            "y": y,
            "analysis_ok": False,
            "feasible": False,
            "violations_": {name: 0.0 for name in names},
            "constraint_values": {name: 0.0 for name in names},
        }
    ordered = {name: float(violations[name]) for name in names}
    phi = math.fsum(ordered[name] for name in names)
    if utilizations is None:
        shown = {name: ordered[name] + 1.0 for name in names}
    else:
        shown = {name: float(utilizations[name]) for name in names}
    return {
        "y_p": y_p,
        "y": y,
        "analysis_ok": True,
        "feasible": phi == 0.0,
        "violations_": ordered,
        "constraint_values": shown,
    }


def _feasible(y: float, *, names: tuple[str, ...] = ("g",)) -> dict[str, Any]:
    return _record(y, y, {name: 0.0 for name in names}, names=names)


def _phi(
    phi: float,
    y: float,
    y_p: float,
    *,
    names: tuple[str, ...] = ("g",),
    utilizations: dict[str, float] | None = None,
) -> dict[str, Any]:
    violations = {names[0]: phi}
    for name in names[1:]:
        violations[name] = 0.0
    return _record(y, y_p, violations, names=names, utilizations=utilizations)


def _failed(*, names: tuple[str, ...] = ("g",)) -> dict[str, Any]:
    return _record(math.inf, math.inf, {}, names=names, analysis_ok=False)


def _split(
    first: float,
    second: float,
    y: float,
    y_p: float,
) -> dict[str, Any]:
    return _record(y, y_p, {"g": first, "h": second}, names=_NAMES)


class _Map:
    """Integer problem publishing one auxiliary record per index vector."""

    def __init__(
        self,
        records: dict[tuple[int, ...], dict[str, Any]],
        lower: list[int],
        upper: list[int],
        *,
        names: tuple[str, ...],
        default: dict[str, Any],
        invalid_utilization: float = 99.0,
    ) -> None:
        self.records = records
        self.default = default
        self.bounds = SimpleNamespace(lb=lower, ub=upper)
        self.meta_data = SimpleNamespace(n_variables=len(lower))
        self.state = SimpleNamespace(
            evaluations=0,
            current_best=SimpleNamespace(y=math.inf, x=[]),
        )
        self.calls: list[tuple[int, ...]] = []
        self.hdb_eval_context = SimpleNamespace(
            _c_names=list(names),
            run_config=SimpleNamespace(invalid_utilization=invalid_utilization),
            last_data=None,
        )

    def __call__(self, x: Any) -> float:
        point = tuple(int(value) for value in x)
        self.calls.append(point)
        self.state.evaluations += 1
        record = dict(self.records.get(point, self.default))
        self.hdb_eval_context.last_data = record
        y_p = float(record["y_p"])
        if math.isfinite(y_p) and y_p < self.state.current_best.y:
            self.state.current_best = SimpleNamespace(y=y_p, x=list(point))
        return y_p


def _as_point(
    x: tuple[int, ...],
    record: dict[str, Any],
    names: tuple[str, ...],
    invalid_utilization: float = 99.0,
) -> DebEvaluatedPoint:
    channels = validate_deb_auxiliary_record(
        record,
        active_constraint_names=names,
        invalid_utilization=invalid_utilization,
    )
    return DebEvaluatedPoint(
        x=tuple(x),
        y_p=float(record["y_p"]),
        y=channels.y,
        phi=channels.phi,
        feasible=channels.feasible,
        analysis_ok=channels.analysis_ok,
        sentinel_constraints=channels.sentinel_constraints,
    )


def _search(
    records: dict[tuple[int, ...], dict[str, Any]],
    start: tuple[int, ...],
    budget: int,
    function: Any,
    lower: list[int],
    upper: list[int],
    *,
    names: tuple[str, ...] = ("g",),
    default: dict[str, Any] | None = None,
    mode: str = "fixed_budget",
    target: float | None = None,
    invalid_utilization: float = 99.0,
):
    """Run one Deb local search. The start is already evaluated."""
    if default is None:
        default = _phi(100.0, 100.0, 200.0, names=names)
    problem = _Map(
        records,
        lower,
        upper,
        names=names,
        default=default,
        invalid_utilization=invalid_utilization,
    )
    controller = RunController(
        problem,
        budget,
        mode=mode,
        target_threshold=target,
        target_reference="all" if mode == "fixed_target" else None,
    )
    space = DiscreteSearchSpace(lower, upper)
    manager = DebEvaluationManager(problem, controller, space)
    result = function(
        _as_point(
            start,
            records[start],
            names,
            invalid_utilization,
        ),
        manager,
        space,
    )
    return problem, result, manager, controller


def _key(record: dict[str, Any], names: tuple[str, ...]) -> tuple[int, float]:
    channels = validate_deb_auxiliary_record(
        record,
        active_constraint_names=names,
        invalid_utilization=99.0,
    )
    return deb_quality_key(
        feasible=channels.feasible,
        y=channels.y,
        phi=channels.phi,
    )


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


def _fork_records(space: DiscreteSearchSpace, start: tuple[int, ...]):
    neighbours = space.neighbours_l1(start)
    first, second = neighbours[0], neighbours[1]
    private = next(
        candidate
        for candidate in space.neighbours_l1(second)
        if candidate != start and candidate not in neighbours
    )
    records = {
        start: _feasible(10.0),
        first: _feasible(8.0),
        second: _feasible(8.0),
        private: _feasible(9.0),
    }
    return records, private


def _measured_budget(space, start, records, lower, upper) -> int:
    _problem, result, _manager, _controller = _search(
        records,
        start,
        1000,
        _EQUAL,
        lower,
        upper,
        default=_feasible(100.0),
    )
    return 1 + result.n_evaluations


class TestDebEqualDescentExamples(unittest.TestCase):
    """Worked examples from the approved Deb equal-descent plan."""

    def test_example_6_1_scans_both_equal_exits(self) -> None:
        records = {
            (1, 1): _split(27.0, 0.0, 10.0, _penalty(10.0, 27.0)),
            (0, 1): _split(8.0, 0.0, 8.0, _penalty(8.0, 8.0)),
            (2, 1): _split(8.0, 0.0, 12.0, _penalty(12.0, 8.0)),
            (1, 0): _split(27.0, 1.0, 9.0, _penalty(9.0, 27.0, 1.0)),
            (1, 2): _split(27.0, 8.0, 11.0, _penalty(11.0, 27.0, 8.0)),
        }
        bounds = ([0, 0], [3, 3])
        strict_problem, strict, strict_manager, _controller = _search(
            records, (1, 1), 100, _STRICT, *bounds, names=_NAMES
        )
        equal_problem, equal, equal_manager, _controller = _search(
            records, (1, 1), 100, _EQUAL, *bounds, names=_NAMES
        )
        self.assertEqual(strict.point.x, (0, 1))
        self.assertEqual(equal.point.x, (0, 1))
        self.assertTrue(strict.certified_local_optimum)
        self.assertTrue(equal.certified_local_optimum)
        self.assertEqual(strict.n_steps, 1)
        self.assertEqual(equal.n_steps, 1)
        extra = {(3, 1), (2, 0), (2, 2)}
        self.assertTrue(extra.isdisjoint(strict_problem.calls))
        self.assertTrue(extra <= set(equal_problem.calls))
        self.assertEqual(strict_manager.diagnostics["n_infeasible_plateau_terminations"], 0)
        self.assertEqual(equal_manager.diagnostics["n_infeasible_plateau_terminations"], 0)
        self.assertFalse(getattr(equal, "early_certificate", False))

        for function in (_STRICT, _EQUAL):
            _problem, stopped, _manager, _controller = _search(
                records, (1, 1), 1, function, *bounds, names=_NAMES
            )
            self.assertEqual(stopped.point.x, (0, 1))
            self.assertFalse(stopped.certified_local_optimum)
            self.assertEqual(stopped.n_steps, 1)
            self.assertEqual(stopped.n_certification_evaluations, 0)

    def test_example_6_2_follows_the_feasible_exit(self) -> None:
        records = {
            (1, 1): _split(27.0, 0.0, 10.0, _penalty(10.0, 27.0)),
            (0, 1): _split(8.0, 0.0, 8.0, _penalty(8.0, 8.0)),
            (2, 1): _split(8.0, 0.0, 12.0, _penalty(12.0, 8.0)),
            (1, 0): _split(27.0, 1.0, 9.0, _penalty(9.0, 27.0, 1.0)),
            (1, 2): _split(27.0, 8.0, 11.0, _penalty(11.0, 27.0, 8.0)),
            (2, 2): _split(0.0, 0.0, 20.0, 20.0),
        }
        bounds = ([0, 0], [3, 3])
        _strict_problem, strict, _manager, _controller = _search(
            records, (1, 1), 100, _STRICT, *bounds, names=_NAMES
        )
        equal_problem, equal, _manager, _controller = _search(
            records, (1, 1), 100, _EQUAL, *bounds, names=_NAMES
        )
        self.assertEqual(strict.point.x, (0, 1))
        self.assertTrue(strict.certified_local_optimum)
        self.assertEqual(equal.point.x, (2, 2))
        self.assertTrue(equal.certified_local_optimum)
        self.assertEqual(equal.n_steps, 2)
        self.assertIn((2, 2), equal_problem.calls)
        self.assertEqual(equal.point.y, 20.0)

    def test_example_6_3_keeps_the_arrival_design_on_the_equal_level(self) -> None:
        records = {
            (1, 1): _phi(8.0, 10.0, _penalty(10.0, 8.0)),
            (0, 1): _phi(1.0, 20.0, _penalty(20.0, 1.0)),
            (2, 1): _phi(8.0, 4.0, _penalty(4.0, 8.0)),
            (1, 0): _feasible(40.0),
            (1, 2): _phi(27.0, 10.0, _penalty(10.0, 27.0)),
        }
        bounds = ([0, 0], [3, 3])
        strict_problem, strict, _manager, _controller = _search(
            records, (1, 1), 100, _STRICT, *bounds
        )
        equal_problem, equal, _manager, _controller = _search(
            records, (1, 1), 100, _EQUAL, *bounds
        )
        self.assertEqual(strict.point.x, (1, 0))
        self.assertEqual(equal.point.x, (1, 0))
        self.assertTrue(strict.certified_local_optimum)
        self.assertTrue(equal.certified_local_optimum)
        self.assertIn((2, 0), strict_problem.calls)
        self.assertIn((2, 0), equal_problem.calls)
        self.assertTrue({(3, 1), (2, 2)} <= set(equal_problem.calls))
        self.assertTrue({(3, 1), (2, 2)}.isdisjoint(strict_problem.calls))
        self.assertNotIn((0, 2), strict_problem.calls)
        self.assertNotIn((0, 2), equal_problem.calls)

    def test_example_6_4_plateau_counter_is_a_termination_diagnostic(self) -> None:
        records = {
            (2, 1): _phi(8.0, 10.0, _penalty(10.0, 8.0)),
            (1, 1): _phi(8.0, 12.0, _penalty(12.0, 8.0)),
            (1, 0): _feasible(20.0),
        }
        bounds = ([0, 0], [3, 3])
        strict_problem, strict, strict_manager, _controller = _search(
            records, (2, 1), 100, _STRICT, *bounds
        )
        equal_problem, equal, equal_manager, _controller = _search(
            records, (2, 1), 100, _EQUAL, *bounds
        )
        self.assertEqual(strict.point.x, (2, 1))
        self.assertTrue(strict.certified_local_optimum)
        self.assertNotIn((1, 0), strict_problem.calls)
        self.assertEqual(strict_manager.diagnostics["n_infeasible_plateau_terminations"], 1)
        self.assertEqual(equal.point.x, (1, 0))
        self.assertTrue(equal.certified_local_optimum)
        self.assertIn((1, 0), equal_problem.calls)
        self.assertEqual(equal_manager.diagnostics["n_infeasible_plateau_terminations"], 0)
        self.assertEqual(
            equal_manager.diagnostics["n_sentinel_infeasible_plateau_terminations"],
            0,
        )

    def test_sentinel_elsewhere_does_not_increment_the_sentinel_counter(self) -> None:
        records = {
            (2, 1): _phi(8.0, 10.0, _penalty(10.0, 8.0)),
            (1, 1): _phi(8.0, 12.0, _penalty(12.0, 8.0)),
            (0, 1): _phi(
                8.0,
                14.0,
                _penalty(14.0, 8.0),
                utilizations={"g": 99.0},
            ),
        }
        bounds = ([0, 0], [3, 3])
        strict_problem, strict, strict_manager, _controller = _search(
            records, (2, 1), 100, _STRICT, *bounds
        )
        equal_problem, equal, equal_manager, _controller = _search(
            records, (2, 1), 100, _EQUAL, *bounds
        )
        self.assertEqual(strict.point.x, (2, 1))
        self.assertEqual(equal.point.x, (2, 1))
        self.assertTrue(strict.certified_local_optimum)
        self.assertTrue(equal.certified_local_optimum)
        self.assertNotIn((0, 1), strict_problem.calls)
        self.assertIn((0, 1), equal_problem.calls)
        for manager in (strict_manager, equal_manager):
            self.assertEqual(manager.diagnostics["n_infeasible_plateau_terminations"], 1)
            self.assertEqual(
                manager.diagnostics["n_sentinel_infeasible_plateau_terminations"],
                0,
            )

    def test_sentinel_on_the_incumbent_counts_both(self) -> None:
        records = {
            (2, 1): _phi(
                8.0,
                10.0,
                _penalty(10.0, 8.0),
                utilizations={"g": 99.0},
            ),
            (1, 1): _phi(8.0, 12.0, _penalty(12.0, 8.0)),
        }
        bounds = ([0, 0], [3, 3])
        for function in (_STRICT, _EQUAL):
            _problem, result, manager, _controller = _search(
                records, (2, 1), 100, function, *bounds
            )
            self.assertEqual(result.point.x, (2, 1))
            self.assertTrue(result.certified_local_optimum)
            self.assertEqual(result.point.sentinel_constraints, ("g",))
            self.assertEqual(manager.diagnostics["n_infeasible_plateau_terminations"], 1)
            self.assertEqual(
                manager.diagnostics["n_sentinel_infeasible_plateau_terminations"],
                1,
            )

    def test_sentinel_on_an_equal_neighbour_counts_both(self) -> None:
        records = {
            (2, 1): _phi(8.0, 10.0, _penalty(10.0, 8.0)),
            (1, 1): _phi(
                8.0,
                12.0,
                _penalty(12.0, 8.0),
                utilizations={"g": 99.0},
            ),
        }
        bounds = ([0, 0], [3, 3])
        for function in (_STRICT, _EQUAL):
            _problem, result, manager, _controller = _search(
                records, (2, 1), 100, function, *bounds
            )
            self.assertEqual(result.point.x, (2, 1))
            self.assertTrue(result.certified_local_optimum)
            self.assertEqual(manager.diagnostics["n_infeasible_plateau_terminations"], 1)
            self.assertEqual(
                manager.diagnostics["n_sentinel_infeasible_plateau_terminations"],
                1,
            )

    def test_example_6_5_failed_analysis_is_not_a_component_member(self) -> None:
        records = {
            (1, 1): _feasible(5.0),
            (0, 1): _failed(),
            (2, 1): _failed(),
            (1, 0): _phi(1.0, 4.0, _penalty(4.0, 1.0)),
        }
        bounds = ([0, 0], [3, 3])
        for function in (_STRICT, _EQUAL):
            problem, result, _manager, _controller = _search(
                records, (1, 1), 100, function, *bounds
            )
            self.assertEqual(result.point.x, (1, 1))
            self.assertTrue(result.certified_local_optimum)
            self.assertEqual(result.n_steps, 0)
            self.assertEqual(problem.calls.count((0, 1)), 1)
            self.assertEqual(problem.calls.count((2, 1)), 1)
            self.assertNotIn((0, 0), problem.calls)
            self.assertNotIn((3, 1), problem.calls)

        failed_start = {
            (1, 1): _failed(),
            (0, 1): _failed(),
            (2, 1): _feasible(9.0),
            (1, 0): _phi(1.0, 4.0, _penalty(4.0, 1.0)),
        }
        for function in (_STRICT, _EQUAL):
            problem, result, _manager, _controller = _search(
                failed_start, (1, 1), 100, function, *bounds
            )
            self.assertEqual(result.point.x, (2, 1))
            self.assertTrue(result.certified_local_optimum)
            self.assertEqual(result.n_steps, 1)
            self.assertNotIn((0, 0), problem.calls)
            self.assertFalse(result.point.analysis_ok is False)

        all_failed = {
            (1, 1): _failed(),
            (0, 1): _failed(),
            (2, 1): _failed(),
            (1, 0): _failed(),
            (1, 2): _failed(),
        }
        _problem, result, _manager, _controller = _search(
            all_failed, (1, 1), 100, _EQUAL, *bounds
        )
        self.assertEqual(result.point.x, (1, 1))
        self.assertFalse(result.certified_local_optimum)
        self.assertEqual(result.n_steps, 0)
        self.assertFalse(getattr(result, "early_certificate", False))
        self.assertFalse(result.point.analysis_ok)

        for function in (_STRICT, _EQUAL):
            _problem, moved, _manager, _controller = _search(
                failed_start, (1, 1), 2, function, *bounds
            )
            self.assertEqual(moved.point.x, (2, 1))
            self.assertFalse(moved.certified_local_optimum)
            self.assertEqual(moved.n_steps, 1)
            _problem, stayed, _manager, _controller = _search(
                failed_start, (1, 1), 1, function, *bounds
            )
            self.assertEqual(stayed.point.x, (1, 1))
            self.assertFalse(stayed.certified_local_optimum)
            self.assertEqual(stayed.n_steps, 0)

    def test_sentinel_violation_stays_expandable(self) -> None:
        y_p = 10.0 * (1.0 + 98.0 ** (1.0 / 3.0))
        records = {
            (1, 1): _phi(98.0, 10.0, y_p, utilizations={"g": 99.0}),
            (2, 1): _phi(98.0, 12.0, 12.0 * (1.0 + 98.0 ** (1.0 / 3.0))),
            (0, 1): _failed(),
        }
        bounds = ([0, 0], [3, 3])
        start = _as_point((1, 1), records[(1, 1)], ("g",))
        self.assertTrue(start.analysis_ok)
        self.assertEqual(start.phi, 98.0)
        self.assertEqual(start.sentinel_constraints, ("g",))
        self.assertEqual(start.y_p, y_p)
        strict_problem, strict, _manager, _controller = _search(
            records, (1, 1), 100, _STRICT, *bounds
        )
        equal_problem, equal, _manager, _controller = _search(
            records, (1, 1), 100, _EQUAL, *bounds
        )
        self.assertEqual(strict.point.x, (1, 1))
        self.assertEqual(equal.point.x, (1, 1))
        self.assertTrue(equal.certified_local_optimum)
        self.assertIn((3, 1), equal_problem.calls)
        self.assertNotIn((3, 1), strict_problem.calls)
        self.assertIn((0, 1), equal_problem.calls)
        self.assertIn((0, 1), strict_problem.calls)
        self.assertNotIn((0, 0), equal_problem.calls)
        self.assertNotIn((0, 0), strict_problem.calls)

    def test_example_6_6_budget_and_early_certificate(self) -> None:
        records = {(2, 1): _feasible(8.0), (1, 1): _feasible(8.0)}
        bounds = ([0, 0], [3, 3])
        for function in (_STRICT, _EQUAL):
            _problem, result, manager, _controller = _search(
                records, (2, 1), 100, function, *bounds, default=_feasible(100.0)
            )
            self.assertEqual(result.point.x, (2, 1))
            self.assertNotEqual(result.point.x, (1, 1))
            self.assertTrue(result.certified_local_optimum)
            self.assertEqual(result.n_steps, 0)
            self.assertEqual(manager.diagnostics["n_infeasible_plateau_terminations"], 0)

        for function in (_STRICT, _EQUAL):
            _problem, stopped, manager, _controller = _search(
                records, (2, 1), 1, function, *bounds, default=_feasible(100.0)
            )
            self.assertEqual(stopped.point.x, (2, 1))
            self.assertFalse(stopped.certified_local_optimum)
            self.assertEqual(stopped.n_steps, 0)
            self.assertFalse(getattr(stopped, "early_certificate", False))
            self.assertEqual(manager.diagnostics["n_infeasible_plateau_terminations"], 0)
            self.assertEqual(stopped.n_certification_evaluations, 0)

        _problem, early, _manager, _controller = _search(
            records, (2, 1), 5, _EQUAL, *bounds, default=_feasible(100.0)
        )
        self.assertEqual(early.point.x, (2, 1))
        self.assertTrue(early.certified_local_optimum)
        self.assertTrue(early.early_certificate)
        self.assertEqual(early.n_steps, 0)
        self.assertEqual(early.n_certification_evaluations, 4)
        self.assertEqual(early.n_evaluations, 5)
        _problem, strict, _manager, _controller = _search(
            records, (2, 1), 5, _STRICT, *bounds, default=_feasible(100.0)
        )
        self.assertTrue(strict.certified_local_optimum)
        self.assertFalse(getattr(strict, "early_certificate", False))
        self.assertEqual(strict.n_evaluations, 4)

        partial = {(2, 1): _feasible(8.0), (1, 1): _feasible(8.0), (0, 1): _feasible(7.0)}
        _problem, held, _manager, _controller = _search(
            partial, (2, 1), 5, _EQUAL, *bounds, default=_feasible(100.0)
        )
        self.assertEqual(held.point.x, (2, 1))
        self.assertNotEqual(held.point.x, (0, 1))
        self.assertFalse(held.certified_local_optimum)
        self.assertEqual(held.n_steps, 0)
        self.assertEqual(held.n_certification_evaluations, 0)
        self.assertFalse(getattr(held, "early_certificate", False))

        for function in (_STRICT, _EQUAL):
            _problem, moved, _manager, _controller = _search(
                {(2, 1): _feasible(8.0), (1, 1): _feasible(7.0)},
                (2, 1),
                1,
                function,
                *bounds,
                default=_feasible(100.0),
            )
            self.assertEqual(moved.point.x, (1, 1))
            self.assertFalse(moved.certified_local_optimum)
            self.assertEqual(moved.n_steps, 1)
            self.assertEqual(moved.n_certification_evaluations, 0)

    def test_example_6_7_drops_the_worse_line(self) -> None:
        named = {
            (2, 2): 10.0,
            (1, 2): 8.0,
            (3, 2): 8.0,
            (1, 1): 5.0,
            (3, 1): 5.0,
            (1, 0): 3.0,
            (3, 0): 4.0,
            (4, 0): 1.0,
        }
        bounds = ([0, 0], [4, 4])
        feasible = {point: _feasible(value) for point, value in named.items()}
        _problem, result, _manager, _controller = _search(
            feasible, (2, 2), 100, _EQUAL, *bounds, default=_feasible(100.0)
        )
        self.assertEqual(result.point.x, (1, 0))
        self.assertTrue(result.certified_local_optimum)
        self.assertEqual(result.n_steps, 3)
        self.assertIn((3, 0), _problem.calls)
        self.assertNotIn((4, 0), _problem.calls)

        infeasible = {
            point: _phi(value, 50.0, 80.0) for point, value in named.items()
        }
        _problem, result, _manager, _controller = _search(
            infeasible, (2, 2), 100, _EQUAL, *bounds, default=_phi(100.0, 50.0, 80.0)
        )
        self.assertEqual(result.point.x, (1, 0))
        self.assertTrue(result.certified_local_optimum)
        self.assertEqual(result.n_steps, 3)
        self.assertIn((3, 0), _problem.calls)
        self.assertNotIn((4, 0), _problem.calls)
        self.assertEqual(result.point.phi, 3.0)
        self.assertNotEqual(result.point.y, 3.0)

    def test_example_6_8_interrupted_level_and_target_hit(self) -> None:
        base = {
            (1, 1): _split(27.0, 0.0, 10.0, _penalty(10.0, 27.0)),
            (0, 1): _split(8.0, 0.0, 8.0, _penalty(8.0, 8.0)),
            (2, 1): _split(8.0, 0.0, 12.0, _penalty(12.0, 8.0)),
            (1, 0): _split(27.0, 1.0, 9.0, _penalty(9.0, 27.0, 1.0)),
            (1, 2): _split(27.0, 8.0, 11.0, _penalty(11.0, 27.0, 8.0)),
            (2, 2): _split(0.0, 0.0, 20.0, 20.0),
        }
        bounds = ([0, 0], [3, 3])
        better = dict(base)
        better[(3, 2)] = _split(0.0, 0.0, 1.0, 1.0)
        _problem, moved, _manager, _controller = _search(
            better, (1, 1), 10, _EQUAL, *bounds, names=_NAMES
        )
        self.assertEqual(moved.point.x, (3, 2))
        self.assertNotEqual(moved.point.x, (0, 1))
        self.assertFalse(moved.certified_local_optimum)
        self.assertEqual(moved.n_steps, 3)
        self.assertEqual(moved.n_evaluations, 10)
        self.assertEqual(moved.n_certification_evaluations, 0)

        worse = dict(base)
        worse[(3, 2)] = _split(0.0, 0.0, 30.0, 30.0)
        _problem, stayed, _manager, _controller = _search(
            worse, (1, 1), 10, _EQUAL, *bounds, names=_NAMES
        )
        self.assertEqual(stayed.point.x, (2, 2))
        self.assertNotEqual(stayed.point.x, (0, 1))
        self.assertFalse(stayed.certified_local_optimum)
        self.assertEqual(stayed.n_steps, 2)
        self.assertEqual(stayed.n_certification_evaluations, 0)

        hit_records = {
            (1, 1): _feasible(10.0),
            (0, 1): _feasible(12.0),
            (2, 1): _feasible(12.0),
            (1, 0): _feasible(12.0),
            (1, 2): _feasible(4.0),
        }
        small = ([0, 0], [2, 2])
        for function in (_STRICT, _EQUAL):
            problem, hit, _manager, controller = _search(
                hit_records,
                (1, 1),
                10,
                function,
                *small,
                default=_feasible(12.0),
                mode="fixed_target",
                target=4.0,
            )
            self.assertEqual(hit.point.x, (1, 2))
            self.assertFalse(hit.certified_local_optimum)
            self.assertEqual(hit.n_steps, 1)
            self.assertEqual(hit.n_certification_evaluations, 0)
            self.assertNotIn((0, 2), problem.calls)
            self.assertNotIn((2, 2), problem.calls)
            self.assertTrue(controller.target_hit)

        non_improving = {
            (1, 1): _feasible(20.0),
            (0, 1): _phi(1.0, 6.0, 12.0),
        }
        for function in (_STRICT, _EQUAL):
            problem, held, manager, controller = _search(
                non_improving,
                (1, 1),
                10,
                function,
                *small,
                default=_feasible(30.0),
                mode="fixed_target",
                target=12.0,
            )
            self.assertEqual(held.point.x, (1, 1))
            self.assertFalse(held.certified_local_optimum)
            self.assertEqual(held.n_steps, 0)
            self.assertEqual(held.n_certification_evaluations, 0)
            self.assertEqual(problem.calls, [(0, 1)])
            self.assertEqual(controller.best_finite_y_p, 12.0)
            self.assertIsNotNone(manager.best_benchmark_point)
            self.assertEqual(manager.best_benchmark_point.y_p, 12.0)

    def test_example_6_9_certification_count_can_differ(self) -> None:
        records = {
            (0, 1): _feasible(10.0),
            (1, 1): _feasible(10.0),
            (0, 0): _feasible(8.0),
            (1, 0): _feasible(9.0),
        }
        bounds = ([0, 0], [1, 1])
        strict_problem, strict, _manager, _controller = _search(
            records, (0, 1), 100, _STRICT, *bounds, default=_feasible(100.0)
        )
        equal_problem, equal, _manager, _controller = _search(
            records, (0, 1), 100, _EQUAL, *bounds, default=_feasible(100.0)
        )
        self.assertEqual(strict.point.x, (0, 0))
        self.assertEqual(equal.point.x, (0, 0))
        self.assertTrue(strict.certified_local_optimum)
        self.assertTrue(equal.certified_local_optimum)
        self.assertEqual(strict.n_certification_evaluations, 1)
        self.assertEqual(equal.n_certification_evaluations, 0)
        self.assertEqual(equal_problem.calls.count((1, 0)), 1)
        self.assertEqual(strict_problem.calls.count((1, 0)), 1)
        second, _result, _manager, _controller = _search(
            records, (0, 1), 100, _EQUAL, *bounds, default=_feasible(100.0)
        )
        self.assertIn((1, 0), second.calls)

    def test_unique_best_slope_matches_strict_search(self) -> None:
        records = {
            (x0, x1): _feasible(10.0 + x0 + 3.0 * x1)
            for x0 in range(3)
            for x1 in range(3)
        }
        bounds = ([0, 0], [2, 2])
        strict_problem, strict, _manager, _controller = _search(
            records, (2, 2), 100, _STRICT, *bounds, default=_feasible(100.0)
        )
        equal_problem, equal, _manager, _controller = _search(
            records, (2, 2), 100, _EQUAL, *bounds, default=_feasible(100.0)
        )
        self.assertEqual(equal.point.x, strict.point.x)
        self.assertEqual(equal_problem.calls, strict_problem.calls)
        self.assertEqual(equal.certified_local_optimum, strict.certified_local_optimum)
        self.assertEqual(equal.n_steps, strict.n_steps)
        self.assertEqual(
            equal.n_certification_evaluations,
            strict.n_certification_evaluations,
        )

    def test_empty_neighbourhood_is_certified_without_evaluations(self) -> None:
        records = {(0,): _feasible(4.0)}
        _problem, result, _manager, _controller = _search(
            records, (0,), 10, _EQUAL, [0], [0]
        )
        self.assertEqual(result.point.x, (0,))
        self.assertTrue(result.certified_local_optimum)
        self.assertEqual(result.n_steps, 0)
        self.assertEqual(result.n_evaluations, 0)
        self.assertEqual(result.n_certification_evaluations, 0)
        self.assertFalse(getattr(result, "early_certificate", False))

    def test_a_better_resolved_neighbour_is_never_certified(self) -> None:
        records = {
            (1, 1): _split(27.0, 0.0, 10.0, _penalty(10.0, 27.0)),
            (0, 1): _split(8.0, 0.0, 8.0, _penalty(8.0, 8.0)),
            (2, 1): _split(8.0, 0.0, 12.0, _penalty(12.0, 8.0)),
            (1, 0): _split(27.0, 1.0, 9.0, _penalty(9.0, 27.0, 1.0)),
            (1, 2): _split(27.0, 8.0, 11.0, _penalty(11.0, 27.0, 8.0)),
            (2, 2): _split(0.0, 0.0, 20.0, 20.0),
        }
        bounds = ([0, 0], [3, 3])
        _full, complete, _manager, _controller = _search(
            records, (1, 1), 100, _EQUAL, *bounds, names=_NAMES
        )
        for budget in range(1, complete.n_evaluations + 1):
            problem, result, _manager, _controller = _search(
                records, (1, 1), budget, _EQUAL, *bounds, names=_NAMES
            )
            returned = deb_quality_key(
                feasible=result.point.feasible,
                y=result.point.y,
                phi=result.point.phi,
            )
            for call in problem.calls:
                record = records.get(call, _phi(100.0, 100.0, 200.0, names=_NAMES))
                if _key(record, _NAMES) < returned:
                    self.assertFalse(result.certified_local_optimum)


class TestDebEqualDescentWiring(unittest.TestCase):
    """Registry, presentation, and outer-loop wiring for the six tokens."""

    def test_alias_stays_on_strict_search(self) -> None:
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

    def test_penalty_classes_keep_their_own_searches(self) -> None:
        self.assertIs(
            PopulationBasinHoppingEquDesc(4, seed=1, population_size=1).local_search,
            equal_descent_local_search,
        )
        self.assertIs(
            KEliteAMSEquDesc(4, seed=1, n_parents=1, archive_size=1).local_search,
            equal_descent_local_search,
        )
        self.assertIs(
            DiscreteMLSLEquDesc(4, seed=1, batch_size=1).local_search,
            equal_descent_local_search,
        )
        self.assertIs(
            PopulationBasinHopping(4, seed=1, population_size=1).local_search,
            best_improvement_local_search,
        )
        self.assertIs(
            KEliteAMS(4, seed=1, n_parents=1, archive_size=1).local_search,
            best_improvement_local_search,
        )
        self.assertIs(
            DiscreteMLSL(4, seed=1, batch_size=1).local_search,
            best_improvement_local_search,
        )

    def test_registry_order_parameters_and_descriptions(self) -> None:
        names = ALGORITHM_REGISTRY.names()
        for stem in (
            "DISCRETE_MLSL",
            "K_ELITE_AMS",
            "POPULATION_BASIN_HOPPING",
        ):
            equ_cold = names.index(f"{stem}_DEB_EQU_DESC_COLD_START")
            equ_warm = names.index(f"{stem}_DEB_EQU_DESC_WARM_START")
            si_cold = names.index(f"{stem}_DEB_SI_DESC_COLD_START")
            si_warm = names.index(f"{stem}_DEB_SI_DESC_WARM_START")
            pen_equ = names.index(f"{stem}_PEN_EQU_DESC_COLD_START")
            self.assertLess(equ_cold, equ_warm)
            self.assertLess(equ_warm, si_cold)
            self.assertLess(si_cold, si_warm)
            self.assertLess(equ_cold, pen_equ)
            equ = get_algorithm_spec(f"{stem}_DEB_EQU_DESC_COLD_START")
            strict = get_algorithm_spec(f"{stem}_DEB_SI_DESC_COLD_START")
            self.assertEqual(equ.parameters, strict.parameters)
            self.assertEqual(equ.seed_policy, strict.seed_policy)
            self.assertEqual(equ.candidate_mode, strict.candidate_mode)
            self.assertEqual(equ.budget_semantics, strict.budget_semantics)
            self.assertEqual(equ.validator_ref, strict.validator_ref)
            self.assertEqual(equ.library, "native")
            self.assertIn("equal", equ.description.lower())
            warm = get_algorithm_spec(f"{stem}_DEB_EQU_DESC_WARM_START")
            self.assertIn("InitialDesign", warm.description)
            self.assertIn("Deb-ranked", warm.description)

    def test_builders_name_the_new_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "DISCRETE_MLSL_DEB_EQU_DESC_COLD_START"):
            build_discrete_mlsl_deb_equ_desc({}, None)
        with self.assertRaisesRegex(ValueError, "K_ELITE_AMS_DEB_EQU_DESC_COLD_START"):
            build_k_elite_ams_deb_equ_desc({}, None)
        with self.assertRaisesRegex(
            ValueError,
            "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_COLD_START",
        ):
            build_population_basin_hopping_deb_equ_desc({}, None)
        with self.assertRaisesRegex(ValueError, "DISCRETE_MLSL_DEB_EQU_DESC_WARM_START"):
            build_discrete_mlsl_deb_equ_desc_warm_start(
                {
                    "max_evaluations": 1,
                    "batch_size": 1,
                    "reduction_fraction": 0.1,
                    "sigma": 2.0,
                    "sampler": "sobol",
                },
                1,
            )
        with self.assertRaisesRegex(ValueError, "K_ELITE_AMS_DEB_EQU_DESC_WARM_START"):
            build_k_elite_ams_deb_equ_desc_warm_start(
                {
                    "max_evaluations": 1,
                    "archive_size": 1,
                    "n_parents": 1,
                    "mutation_probability": 0.05,
                    "random_restart_probability": 0.1,
                },
                1,
            )
        with self.assertRaisesRegex(
            ValueError,
            "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START",
        ):
            build_population_basin_hopping_deb_equ_desc_warm_start(
                {
                    "max_evaluations": 1,
                    "population_size": 1,
                    "perturbation_fraction": 0.15,
                    "dcut_factor": 0.5,
                    "initial_sampler": "uniform",
                },
                1,
            )

    def test_presentation_and_warm_start_membership(self) -> None:
        for token in _DEB_EQU:
            presented = presentation_for(token)
            self.assertTrue(presented.constraint_aware)
            self.assertEqual(presented.qualifier, "EQU-DESC")
            self.assertEqual(local_search_label(presented), "Equal descent")
            self.assertEqual(legend_handling(presented), "Constraint-aware, Equal descent")
            if token.endswith("_WARM_START"):
                cold = token.replace("_WARM_START", "_COLD_START")
                self.assertEqual(presented.family, cold)
        for stem in (
            "DISCRETE_MLSL_DEB_SI_DESC",
            "K_ELITE_AMS_DEB_SI_DESC",
            "POPULATION_BASIN_HOPPING_DEB_SI_DESC",
        ):
            presented = presentation_for(f"{stem}_COLD_START")
            self.assertTrue(presented.constraint_aware)
            self.assertEqual(local_search_label(presented), "Strict improvement")
        for token in (
            "DISCRETE_MLSL_DEB_EQU_DESC_WARM_START",
            "K_ELITE_AMS_DEB_EQU_DESC_WARM_START",
            "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START",
        ):
            self.assertIn(token, INITIAL_DESIGN_PATH_TOKENS)
            self.assertIn(token, WARM_START_TOKENS)
            self.assertIn(token, DEB_WARM_START_TOKENS)
            self.assertNotIn(token, RANK1_SEEDLESS_TOKENS)

    def test_classes_use_deb_equal_descent(self) -> None:
        cold = (
            (
                PopulationBasinHoppingDebEquDesc(5, seed=1, population_size=1),
                "PBH-L1-Deb-EquDesc",
            ),
            (
                KEliteAMSDebEquDesc(5, seed=1, archive_size=1, n_parents=1),
                "k-elite AMS-Deb-EquDesc",
            ),
            (
                DiscreteMLSLDebEquDesc(5, seed=1, batch_size=1),
                "dMLSL-L1-Deb-EquDesc-Sobol",
            ),
            (
                DiscreteMLSLDebEquDesc(5, seed=1, batch_size=1, sampler="uniform"),
                "dMLSL-L1-Deb-EquDesc-uniform",
            ),
        )
        for algorithm, prefix in cold:
            self.assertIs(algorithm.local_search, deb_equal_descent_local_search)
            self.assertTrue(algorithm.info.startswith(prefix))
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "initial_design_deb.json")
            warm = (
                (
                    PopulationBasinHoppingDebEquDescWarmStart(
                        5,
                        seed=1,
                        population_size=1,
                        initial_design_path=path,
                    ),
                    "PBH-L1-Deb-EquDesc-WarmStart",
                ),
                (
                    KEliteAMSDebEquDescWarmStart(
                        5,
                        seed=1,
                        archive_size=1,
                        n_parents=1,
                        initial_design_path=path,
                    ),
                    "k-elite AMS-Deb-EquDesc-WarmStart",
                ),
                (
                    DiscreteMLSLDebEquDescWarmStart(
                        5,
                        seed=1,
                        batch_size=1,
                        initial_design_path=path,
                    ),
                    "dMLSL-L1-Deb-EquDesc-WarmStart-Sobol",
                ),
            )
            for algorithm, prefix in warm:
                self.assertIs(algorithm.local_search, deb_equal_descent_local_search)
                self.assertTrue(algorithm.info.startswith(prefix))
                self.assertTrue(algorithm.name.endswith("_WARM_START"))

    def test_outer_loops_evaluate_the_second_branch(self) -> None:
        lower = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        upper = [2, 2, 0, 0, 0, 0, 0, 0, 0]
        space = DiscreteSearchSpace(lower, upper)
        warm_start = (1, 1, 0, 0, 0, 0, 0, 0, 0)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design_deb.json"
            _write_prefix(path, [list(warm_start)])
            pbh_start = DiscreteSampler(
                space, "uniform", 7, np.random.default_rng(7)
            ).draw()
            ams_start = _uniform_index(space, np.random.default_rng(7))
            mlsl_start = DiscreteSampler(
                space, "uniform", 7, np.random.default_rng(7)
            ).next_unique(set())
            assert mlsl_start is not None
            builders = (
                (
                    pbh_start,
                    lambda budget: PopulationBasinHoppingDebEquDesc(
                        budget,
                        seed=7,
                        population_size=1,
                        initial_sampler="uniform",
                    ),
                ),
                (
                    ams_start,
                    lambda budget: KEliteAMSDebEquDesc(
                        budget,
                        seed=7,
                        archive_size=1,
                        n_parents=1,
                    ),
                ),
                (
                    mlsl_start,
                    lambda budget: DiscreteMLSLDebEquDesc(
                        budget,
                        seed=7,
                        batch_size=1,
                        sampler="uniform",
                    ),
                ),
                (
                    warm_start,
                    lambda budget: PopulationBasinHoppingDebEquDescWarmStart(
                        budget,
                        seed=7,
                        population_size=1,
                        initial_sampler="uniform",
                        initial_design_path=str(path),
                    ),
                ),
                (
                    warm_start,
                    lambda budget: KEliteAMSDebEquDescWarmStart(
                        budget,
                        seed=7,
                        archive_size=1,
                        n_parents=1,
                        initial_design_path=str(path),
                    ),
                ),
                (
                    warm_start,
                    lambda budget: DiscreteMLSLDebEquDescWarmStart(
                        budget,
                        seed=7,
                        batch_size=1,
                        sampler="uniform",
                        initial_design_path=str(path),
                    ),
                ),
            )
            names = []
            for start, builder in builders:
                records, private = _fork_records(space, start)
                budget = _measured_budget(space, start, records, lower, upper)
                algorithm = builder(budget)
                problem = _Map(
                    records,
                    lower,
                    upper,
                    names=("g",),
                    default=_feasible(100.0),
                )
                algorithm(problem)
                self.assertIs(algorithm.local_search, deb_equal_descent_local_search)
                self.assertIn(private, problem.calls)
                self.assertNotIn(
                    "n_equal_descent_early_certificates",
                    algorithm.last_run_stats,
                )
                names.append(algorithm.name)
            self.assertEqual(len(names), 6)
            self.assertEqual(len(set(names)), 6)

    def test_ams_early_certificate_is_counted_only_for_equal_descent(self) -> None:
        lower = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        upper = [2, 2, 0, 0, 0, 0, 0, 0, 0]
        start = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        records = {
            start: _feasible(8.0),
            (1, 0, 0, 0, 0, 0, 0, 0, 0): _feasible(8.0),
            (0, 1, 0, 0, 0, 0, 0, 0, 0): _feasible(12.0),
            (2, 0, 0, 0, 0, 0, 0, 0, 0): _feasible(12.0),
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design_deb.json"
            _write_prefix(path, [list(start)])
            strict = KEliteAMSDebWarmStart(
                3,
                seed=1,
                archive_size=1,
                n_parents=1,
                initial_design_path=str(path),
            )
            strict(
                _Map(records, lower, upper, names=("g",), default=_feasible(100.0))
            )
            self.assertNotIn(
                "n_equal_descent_early_certificates",
                strict.last_run_stats,
            )
            equal = KEliteAMSDebEquDescWarmStart(
                4,
                seed=1,
                archive_size=1,
                n_parents=1,
                initial_design_path=str(path),
            )
            equal(
                _Map(records, lower, upper, names=("g",), default=_feasible(100.0))
            )
        self.assertEqual(equal.last_run_stats["n_equal_descent_early_certificates"], 1)

    def test_strict_classes_match_a_direct_strict_search(self) -> None:
        lower = [0, 0]
        upper = [2, 2]
        space = DiscreteSearchSpace(lower, upper)
        records = {
            (x0, x1): _feasible(4.0 + x0 + 3.0 * x1)
            for x0 in range(3)
            for x1 in range(3)
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
            "pbh": lambda budget: PopulationBasinHoppingDeb(
                budget,
                seed=7,
                population_size=1,
                initial_sampler="uniform",
            ),
            "ams": lambda budget: KEliteAMSDeb(
                budget,
                seed=7,
                archive_size=1,
                n_parents=1,
            ),
            "mlsl": lambda budget: DiscreteMLSLDeb(
                budget,
                seed=7,
                batch_size=1,
                sampler="uniform",
            ),
        }
        for key, builder in builders.items():
            start = starts[key]
            _direct_problem, direct, direct_manager, _controller = _search(
                records,
                start,
                1000,
                deb_best_improvement_local_search,
                lower,
                upper,
                default=_feasible(100.0),
            )
            algorithm = builder(1 + direct.n_evaluations)
            problem = _Map(
                records,
                lower,
                upper,
                names=("g",),
                default=_feasible(100.0),
            )
            algorithm(problem)
            self.assertEqual(problem.calls[0], start)
            self.assertEqual(problem.calls[1:], _direct_problem.calls)
            self.assertEqual(tuple(algorithm.last_run_stats["best_x"]), direct.point.x)
            self.assertTrue(direct.certified_local_optimum)
            self.assertEqual(direct_manager.diagnostics["n_infeasible_plateau_terminations"], 0)
            self.assertEqual(
                direct_manager.diagnostics["n_sentinel_infeasible_plateau_terminations"],
                0,
            )
            self.assertEqual(algorithm.last_run_stats["n_infeasible_plateau_terminations"], 0)
            self.assertEqual(
                algorithm.last_run_stats["n_sentinel_infeasible_plateau_terminations"],
                0,
            )
            self.assertGreaterEqual(algorithm.last_run_stats["n_certified_local_searches"], 1)
            if key == "pbh":
                self.assertNotIn(
                    "n_certification_evaluations",
                    algorithm.last_run_stats,
                )
            else:
                self.assertEqual(
                    algorithm.last_run_stats["n_certification_evaluations"],
                    direct.n_certification_evaluations,
                )
