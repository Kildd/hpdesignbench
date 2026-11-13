"""Constraint-aware discrete MLSL unit and lifecycle tests."""
from __future__ import annotations

import ast
import math
import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.ioh_core.algorithm_factory import create_algorithm
from core.ioh_core.algorithms.common import RunController
from core.ioh_core.algorithms.native.discrete_search import (
    DebEvaluatedPoint,
    DebEvaluationManager,
    DiscreteMLSLDeb,
    DiscreteMLSLDebWarmStart,
    DiscreteSearchSpace,
    better_deb_sample_within_radius,
    deb_best_improvement_local_search,
    reduce_samples_deb,
    validate_discrete_mlsl_deb_parameters,
    validate_discrete_mlsl_parameters,
)
from core.ioh_core.algorithms.registry import RegistryError
from core.ioh_core.problem_builder import build_problem_from_snapshot
from core.ioh_core.workflows.optimization_runs.io_logging import (
    _validated_algorithm_diagnostics,
    run_logged_optimization,
)
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
    snapshot_slab_inputs,
)
from _mains.Testing.test_cursor_population_basin_hopping_deb import (
    _DebProblem,
    _record,
)


def _point(
    x: tuple[int, ...],
    *,
    y: float,
    phi: float,
    feasible: bool,
    y_p: float | None = None,
) -> DebEvaluatedPoint:
    """Build one analysis-valid Deb evaluation."""
    return DebEvaluatedPoint(
        x=x,
        y_p=y if y_p is None else y_p,
        y=y,
        phi=phi,
        feasible=feasible,
        analysis_ok=True,
        sentinel_constraints=(),
    )


class TestDebReduction(unittest.TestCase):
    """Reduced samples use Deb quality and deterministic representatives."""

    def test_orders_feasibility_y_and_phi_without_mutating_input(self) -> None:
        samples = [
            _point((5,), y=1.0, phi=0.4, feasible=False, y_p=0.0),
            _point((4,), y=9.0, phi=0.1, feasible=False, y_p=100.0),
            _point((3,), y=3.0, phi=0.0, feasible=True, y_p=50.0),
            _point((2,), y=2.0, phi=0.0, feasible=True, y_p=80.0),
        ]
        original = list(samples)

        reduced = reduce_samples_deb(samples, 1.0)

        self.assertEqual([point.x for point in reduced], [(2,), (3,), (4,), (5,)])
        self.assertEqual(samples, original)

    def test_equal_phi_uses_x_only_for_exact_capacity(self) -> None:
        samples = [
            _point((3,), y=0.0, phi=0.2, feasible=False, y_p=0.0),
            _point((1,), y=99.0, phi=0.2, feasible=False, y_p=99.0),
            _point((2,), y=5.0, phi=0.2, feasible=False, y_p=5.0),
        ]

        reduced = reduce_samples_deb(samples, 0.5)

        self.assertEqual(len(reduced), 2)
        self.assertEqual([point.x for point in reduced], [(1,), (2,)])

    def test_reduction_size_is_ceil_with_minimum_one(self) -> None:
        samples = [
            _point((index,), y=float(index), phi=0.0, feasible=True)
            for index in range(5)
        ]
        self.assertEqual(len(reduce_samples_deb(samples, 0.01)), 1)
        self.assertEqual(len(reduce_samples_deb(samples, 0.41)), 3)


class TestDebRadiusSuppression(unittest.TestCase):
    """Only a strictly Deb-better sample inside the strict radius suppresses."""

    def setUp(self) -> None:
        self.space = DiscreteSearchSpace([0], [10])

    def test_feasible_and_lower_phi_samples_suppress(self) -> None:
        infeasible = _point((5,), y=1.0, phi=0.5, feasible=False, y_p=1.0)
        feasible = _point((4,), y=99.0, phi=0.0, feasible=True, y_p=99.0)
        lower_phi = _point((6,), y=100.0, phi=0.2, feasible=False, y_p=100.0)

        self.assertTrue(
            better_deb_sample_within_radius(
                infeasible, [feasible], 0.2, self.space
            )
        )
        self.assertTrue(
            better_deb_sample_within_radius(
                infeasible, [lower_phi], 0.2, self.space
            )
        )

    def test_feasible_samples_use_y(self) -> None:
        candidate = _point((5,), y=5.0, phi=0.0, feasible=True, y_p=1.0)
        lower_y = _point((4,), y=4.0, phi=0.0, feasible=True, y_p=100.0)
        higher_y = _point((6,), y=6.0, phi=0.0, feasible=True, y_p=0.0)
        self.assertTrue(
            better_deb_sample_within_radius(
                candidate, [lower_y], 0.2, self.space
            )
        )
        self.assertFalse(
            better_deb_sample_within_radius(
                candidate, [higher_y], 0.2, self.space
            )
        )

    def test_equal_phi_y_y_p_and_x_do_not_create_strict_better(self) -> None:
        candidate = _point((5,), y=50.0, phi=0.2, feasible=False, y_p=50.0)
        tied = [
            _point((4,), y=1.0, phi=0.2, feasible=False, y_p=1.0),
            _point((6,), y=99.0, phi=0.2, feasible=False, y_p=99.0),
        ]
        self.assertFalse(
            better_deb_sample_within_radius(
                candidate, tied, math.inf, self.space
            )
        )

    def test_radius_is_strict_and_infinity_needs_a_better_sample(self) -> None:
        candidate = _point((5,), y=5.0, phi=0.0, feasible=True)
        better_at_boundary = _point((4,), y=4.0, phi=0.0, feasible=True)
        exact = self.space.distance(candidate.x, better_at_boundary.x)
        self.assertFalse(
            better_deb_sample_within_radius(
                candidate, [better_at_boundary], exact, self.space
            )
        )
        self.assertTrue(
            better_deb_sample_within_radius(
                candidate, [better_at_boundary], exact + 1e-12, self.space
            )
        )
        self.assertTrue(
            better_deb_sample_within_radius(
                candidate, [better_at_boundary], math.inf, self.space
            )
        )
        self.assertFalse(
            better_deb_sample_within_radius(candidate, [candidate], math.inf, self.space)
        )


class TestDiscreteMLSLDebLifecycle(unittest.TestCase):
    """Cold and warm variants preserve budgets, targets, and diagnostics."""

    @staticmethod
    def _records(size: int) -> dict[tuple[int, ...], dict[str, object]]:
        return {
            (index,): _record(
                y_p=float(index + 1),
                y=float(index + 1),
                violation=0.0 if index == 0 else float(index) / 10.0,
            )
            for index in range(size)
        }

    def test_registry_constructs_both_variants(self) -> None:
        cold = create_algorithm(
            "DISCRETE_MLSL_DEB_SI_DESC_COLD_START",
            {"max_evaluations": 4, "batch_size": 2},
            random_seed=7,
        )
        warm = create_algorithm(
            "DISCRETE_MLSL_DEB_SI_DESC_WARM_START",
            {
                "max_evaluations": 4,
                "batch_size": 2,
                "initial_design_path": "initial_design.json",
            },
            random_seed=7,
        )
        self.assertIsInstance(cold, DiscreteMLSLDeb)
        self.assertIsInstance(warm, DiscreteMLSLDebWarmStart)
        self.assertTrue(cold.info.startswith("dMLSL-L1-Deb-Sobol "))
        self.assertTrue(
            warm.info.startswith("dMLSL-L1-Deb-WarmStart-Sobol ")
        )

    def test_shared_validator_preserves_algorithm_identity(self) -> None:
        parameters = {"reduction_fraction": 0, "sigma": 1}
        with self.assertRaisesRegex(
            RegistryError,
            "^DISCRETE_MLSL_PEN_SI_DESC_COLD_START: reduction_fraction",
        ):
            validate_discrete_mlsl_parameters(parameters)
        with self.assertRaisesRegex(
            RegistryError,
            "^DISCRETE_MLSL_DEB_SI_DESC_COLD_START: reduction_fraction",
        ):
            validate_discrete_mlsl_deb_parameters(parameters)

    def test_missing_eval_context_fails_before_evaluation(self) -> None:
        problem = _DebProblem(self._records(2), lower=[0], upper=[1])
        del problem.hdb_eval_context
        algorithm = DiscreteMLSLDeb(
            max_evaluations=2,
            seed=1,
            batch_size=1,
            sampler="uniform",
        )
        with self.assertRaisesRegex(ValueError, "requires an EvalContext"):
            algorithm(problem)
        self.assertEqual(problem.calls, [])
        self.assertEqual(algorithm.summary_diagnostics()["n_evaluations"], 0)

    def test_finite_failed_target_during_global_sampling_starts_no_search(
        self,
    ) -> None:
        first = (0,) * 9
        second = (1,) + (0,) * 8
        records = {
            first: _record(
                y_p=1.0,
                y=10.0,
                violation=98.0,
                analysis_ok=False,
            ),
            second: _record(y_p=2.0, y=2.0, violation=0.0),
        }
        problem = _DebProblem(records, lower=[0] * 9, upper=[1] * 9)
        controller = RunController(
            problem,
            4,
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
            algorithm = DiscreteMLSLDebWarmStart(
                max_evaluations=4,
                seed=3,
                batch_size=2,
                sampler="uniform",
                initial_design_path=str(path),
            )
            algorithm(problem, run_controller=controller)
        self.assertTrue(controller.target_hit)
        self.assertEqual(problem.calls, [first])
        self.assertEqual(algorithm.last_run_stats["n_local_searches"], 0)
        self.assertEqual(algorithm.summary_diagnostics()["best_f"], 1.0)
        self.assertIsNone(algorithm.summary_diagnostics()["best_deb_x"])

    def test_deterministic_trace_and_accounting_identities(self) -> None:
        traces: list[list[tuple[int, ...]]] = []
        summaries: list[dict[str, object]] = []
        for _ in range(2):
            problem = _DebProblem(self._records(7), lower=[0], upper=[6])
            algorithm = DiscreteMLSLDeb(
                max_evaluations=7,
                seed=19,
                batch_size=2,
                reduction_fraction=0.5,
                sampler="uniform",
            )
            algorithm(problem)
            traces.append(problem.calls)
            summaries.append(dict(algorithm.summary_diagnostics()))
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(summaries[0], summaries[1])
        diagnostics = _validated_algorithm_diagnostics(summaries[0])
        self.assertEqual(
            diagnostics["n_evaluations"],
            diagnostics["n_global_samples"]
            + diagnostics["n_local_search_evaluations"],
        )
        self.assertEqual(
            diagnostics["n_evaluations"],
            diagnostics["n_failed_analysis_evaluations"]
            + diagnostics["n_feasible_evaluations"]
            + diagnostics["n_infeasible_evaluations"],
        )

    def test_budget_interrupts_local_search_with_complete_accounting(self) -> None:
        records = {
            (a, b, c): _record(
                y_p=float(1 + a + b + c),
                y=float(1 + a + b + c),
                violation=0.0,
            )
            for a in range(2)
            for b in range(2)
            for c in range(2)
        }
        problem = _DebProblem(records, lower=[0, 0, 0], upper=[1, 1, 1])
        algorithm = DiscreteMLSLDeb(
            max_evaluations=3,
            seed=11,
            batch_size=1,
            sampler="uniform",
        )
        algorithm(problem)
        diagnostics = algorithm.summary_diagnostics()
        self.assertEqual(diagnostics["n_global_samples"], 1)
        self.assertEqual(diagnostics["n_local_search_evaluations"], 2)
        self.assertEqual(diagnostics["n_evaluations"], 3)
        self.assertEqual(diagnostics["n_uncertified_local_searches"], 1)

    def test_nonempty_exhausting_batch_gets_one_final_local_phase(self) -> None:
        problem = _DebProblem(self._records(2), lower=[0], upper=[1])
        algorithm = DiscreteMLSLDeb(
            max_evaluations=8,
            seed=13,
            batch_size=3,
            sampler="uniform",
        )
        algorithm(problem)
        self.assertEqual(algorithm.last_run_stats["n_global_samples"], 2)
        self.assertEqual(algorithm.last_run_stats["n_mlsl_iterations"], 1)
        self.assertGreater(algorithm.last_run_stats["n_local_searches"], 0)

    def test_equal_phi_plateau_is_pointwise_certified(self) -> None:
        records = {
            (0,): _record(y_p=9.0, y=9.0, violation=0.2),
            (1,): _record(y_p=8.0, y=8.0, violation=0.2),
            (2,): _record(y_p=7.0, y=7.0, violation=0.1),
        }
        problem = _DebProblem(records, lower=[0], upper=[2])
        controller = RunController(problem, 3)
        space = DiscreteSearchSpace([0], [2])
        evaluator = DebEvaluationManager(
            problem,
            controller,
            space,
            algorithm_label="dMLSL-L1-Deb",
        )
        start = evaluator.evaluate((0,))
        result = deb_best_improvement_local_search(start, evaluator, space)
        self.assertTrue(result.certified_local_optimum)
        self.assertEqual(result.point.x, (0,))
        self.assertNotIn((2,), problem.calls)
        self.assertGreaterEqual(
            evaluator.diagnostics["n_infeasible_plateau_terminations"],
            1,
        )

    def test_completed_final_neighbourhood_certifies_but_final_move_does_not(
        self,
    ) -> None:
        for neighbour_y, expected_certified in ((2.0, True), (0.5, False)):
            problem = _DebProblem(
                {
                    (0,): _record(y_p=1.0, y=1.0, violation=0.0),
                    (1,): _record(
                        y_p=neighbour_y,
                        y=neighbour_y,
                        violation=0.0,
                    ),
                },
                lower=[0],
                upper=[1],
            )
            controller = RunController(problem, 2)
            space = DiscreteSearchSpace([0], [1])
            evaluator = DebEvaluationManager(
                problem,
                controller,
                space,
                algorithm_label="dMLSL-L1-Deb",
            )
            start = evaluator.evaluate((0,))
            result = deb_best_improvement_local_search(
                start,
                evaluator,
                space,
            )
            self.assertEqual(
                result.certified_local_optimum,
                expected_certified,
            )
            self.assertEqual(result.n_evaluations, 1)

    def test_target_hit_during_local_search_is_uncertified(self) -> None:
        first = (0,) * 9
        neighbour = (1,) + (0,) * 8
        records = {
            first: _record(y_p=10.0, y=10.0, violation=0.0),
            neighbour: _record(y_p=1.0, y=1.0, violation=0.0),
        }
        problem = _DebProblem(
            records,
            lower=[0] * 9,
            upper=[1] + [0] * 8,
        )
        controller = RunController(
            problem,
            3,
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
                        "n": 1,
                        "vectors": [list(first)],
                    }
                ),
                encoding="utf-8",
            )
            algorithm = DiscreteMLSLDebWarmStart(
                max_evaluations=3,
                seed=3,
                batch_size=1,
                sampler="uniform",
                initial_design_path=str(path),
            )
            algorithm(problem, run_controller=controller)
        self.assertEqual(problem.calls, [first, neighbour])
        self.assertEqual(
            algorithm.summary_diagnostics()["n_uncertified_local_searches"],
            1,
        )
        self.assertEqual(
            algorithm.summary_diagnostics()["n_local_search_evaluations"],
            1,
        )

    def test_all_failed_run_has_no_deb_incumbent(self) -> None:
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
        algorithm = DiscreteMLSLDeb(
            max_evaluations=2,
            seed=4,
            batch_size=2,
            sampler="uniform",
        )
        algorithm(problem)
        diagnostics = algorithm.summary_diagnostics()
        self.assertIsNone(diagnostics["best_deb_x"])
        self.assertEqual(diagnostics["n_failed_analysis_evaluations"], 2)
        self.assertEqual(diagnostics["n_global_failed_analysis_samples"], 2)

    def test_warm_duplicates_are_counted_and_invalid_reuse_resets_state(self) -> None:
        first = (0,) * 9
        records = {first: _record(y_p=1.0, y=1.0, violation=0.0)}
        problem = _DebProblem(records, lower=[0] * 9, upper=[1] * 9)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "problem_id": "L8100_B2",
                        "n": 2,
                        "vectors": [list(first), list(first)],
                    }
                ),
                encoding="utf-8",
            )
            algorithm = DiscreteMLSLDebWarmStart(
                max_evaluations=2,
                seed=5,
                batch_size=2,
                sampler="uniform",
                initial_design_path=str(path),
            )
            algorithm(problem)
            self.assertEqual(problem.calls, [first, first])
            diagnostics = algorithm.summary_diagnostics()
            self.assertEqual(diagnostics["n_global_samples"], 2)
            self.assertEqual(diagnostics["n_unique_global_sample_points"], 1)

            path.unlink()
            with self.assertRaisesRegex(ValueError, "no InitialDesign"):
                algorithm(problem)
            reset = algorithm.summary_diagnostics()
            self.assertEqual(reset["n_evaluations"], 0)
            self.assertIsNone(reset["best_f"])

    def test_default_warm_batch_rejects_budget_100_before_evaluation(self) -> None:
        first = (0,) * 9
        problem = _DebProblem(
            {first: _record(y_p=1.0, y=1.0, violation=0.0)},
            lower=[0] * 9,
            upper=[1] * 9,
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "problem_id": "L8100_B2",
                        "n": 128,
                        "vectors": [list(first)] * 128,
                    }
                ),
                encoding="utf-8",
            )
            algorithm = DiscreteMLSLDebWarmStart(
                max_evaluations=100,
                seed=5,
                initial_design_path=str(path),
            )
            with self.assertRaisesRegex(ValueError, "needs 128"):
                algorithm(problem)
        self.assertEqual(problem.calls, [])
        self.assertEqual(algorithm.summary_diagnostics()["n_evaluations"], 0)


class TestDiscreteMLSLDebHpSlabSmoke(unittest.TestCase):
    """Production-record integrity at the three campaign smoke budgets."""

    def test_cold_and_warm_budget_matrix_is_repeatable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot_slab_inputs(
                slab_type="hp_slab",
                inputs_dir=root / "inputs",
            )
            snapshot_dir = resolve_snapshot_slab_dir(
                root / "inputs",
                "hp_slab",
            )
            probe = build_problem_from_snapshot(
                slab_type="hp_slab",
                problem_id="L6750_B2",
                snapshot_dir=snapshot_dir,
            )
            origin = [int(value) for value in probe["lb_idx"]]
            design = root / "initial_design.json"
            design.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "problem_id": "L6750_B2",
                        "n": 64,
                        "vectors": [origin] * 64,
                    }
                ),
                encoding="utf-8",
            )
            variants = (
                ("DISCRETE_MLSL_DEB_SI_DESC_COLD_START", {}),
                (
                    "DISCRETE_MLSL_DEB_SI_DESC_WARM_START",
                    {"initial_design_path": str(design)},
                ),
            )
            for name, extra in variants:
                for budget in (100, 250, 500):
                    diagnostics: list[dict[str, object]] = []
                    for copy in range(2):
                        bundle = build_problem_from_snapshot(
                            slab_type="hp_slab",
                            problem_id="L6750_B2",
                            snapshot_dir=snapshot_dir,
                        )
                        algorithm = create_algorithm(
                            name,
                            {
                                "max_evaluations": budget,
                                "batch_size": 64,
                                **extra,
                            },
                            random_seed=11,
                        )
                        summary = run_logged_optimization(
                            bundle=bundle,
                            algorithm=algorithm,
                            output_directory=(
                                root / f"{name}-{budget}-{copy}"
                            ),
                        )
                        current = dict(summary["algorithm_diagnostics"])
                        diagnostics.append(current)
                        self._assert_identities(current, budget)
                        self.assertNotIn(
                            "critical_radius_history",
                            current,
                        )
                        self.assertNotIn(
                            "evaluations_per_local_search",
                            current,
                        )
                    self.assertEqual(diagnostics[0], diagnostics[1])

    def test_controlled_case_consumes_full_budget(self) -> None:
        records = {
            (a, b, c): _record(
                y_p=float(1 + a + b + c),
                y=float(1 + a + b + c),
                violation=0.0,
            )
            for a in range(3)
            for b in range(3)
            for c in range(3)
        }
        problem = _DebProblem(
            records,
            lower=[0, 0, 0],
            upper=[2, 2, 2],
        )
        algorithm = DiscreteMLSLDeb(
            max_evaluations=17,
            seed=17,
            batch_size=4,
            sampler="uniform",
        )
        algorithm(problem)
        diagnostics = dict(algorithm.summary_diagnostics())
        self.assertEqual(diagnostics["n_evaluations"], 17)
        self.assertGreater(diagnostics["n_local_searches"], 0)
        self._assert_identities(diagnostics, 17)

    def _assert_identities(
        self,
        diagnostics: dict[str, object],
        budget: int,
    ) -> None:
        self.assertLessEqual(int(diagnostics["n_evaluations"]), budget)
        self.assertEqual(
            diagnostics["n_evaluations"],
            int(diagnostics["n_failed_analysis_evaluations"])
            + int(diagnostics["n_feasible_evaluations"])
            + int(diagnostics["n_infeasible_evaluations"]),
        )
        self.assertEqual(
            diagnostics["n_global_samples"],
            int(diagnostics["n_global_failed_analysis_samples"])
            + int(diagnostics["n_global_feasible_samples"])
            + int(diagnostics["n_global_infeasible_samples"]),
        )
        self.assertEqual(
            diagnostics["n_evaluations"],
            int(diagnostics["n_global_samples"])
            + int(diagnostics["n_local_search_evaluations"]),
        )
        self.assertEqual(
            diagnostics["n_local_searches"],
            int(diagnostics["n_certified_local_searches"])
            + int(diagnostics["n_uncertified_local_searches"]),
        )


class TestMlslPlanSignatures(unittest.TestCase):
    def test_fenced_python_stubs_parse(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "_docs"
            / "constraint_aware_discrete_mlsl_implementation_plan.md"
        )
        text = path.read_text(encoding="utf-8")
        fences = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
        found: dict[str, ast.FunctionDef] = {}
        for fence in fences:
            if (
                "def better_deb_sample_within_radius" not in fence
                and "def run_discrete_mlsl_deb" not in fence
            ):
                continue
            tree = ast.parse(fence)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in {
                    "better_deb_sample_within_radius",
                    "run_discrete_mlsl_deb",
                }:
                    found[node.name] = node
        self.assertEqual(
            set(found),
            {"better_deb_sample_within_radius", "run_discrete_mlsl_deb"},
        )
        radius = found["better_deb_sample_within_radius"]
        self.assertEqual(
            [arg.arg for arg in radius.args.args],
            ["candidate", "samples", "radius", "space"],
        )
        runner = found["run_discrete_mlsl_deb"]
        self.assertEqual(runner.args.args, [])
        self.assertEqual(
            [arg.arg for arg in runner.args.kwonlyargs],
            [
                "evaluator",
                "space",
                "sampler",
                "batch_size",
                "reduction_fraction",
                "sigma",
                "stats",
                "first_batch",
            ],
        )


if __name__ == "__main__":
    unittest.main()
