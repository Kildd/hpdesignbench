"""
Phase C2.2 / SC1 gate: SciPy COBYLA constrained continuous-rounded adapter.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.ioh_core.algorithm_factory import (
    AlgorithmFactoryError,
    create_algorithm,
    validate_algorithm_parameters,
)
from core.ioh_core.algorithms import Cobyla
from core.ioh_core.algorithms.common import RunController
from core.ioh_core.algorithms.registry import (
    BudgetSemantics,
    CandidateMode,
    SeedPolicy,
    get_algorithm_spec,
    normalize_algorithm_parameters,
)
from core.ioh_core.algorithms.scipy.cobyla import Cobyla as PackageCobyla
from core.ioh_core.algorithms.scipy.factory import build_scipy_cobyla
from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem


class _ConstrainedFakeIOHProblem(FakeIOHProblem):
    """
    Fake IOH problem that publishes unpenalized ``y`` and utilizations.

    Constraint ``g0`` uses utilization ``η = x[0]`` (feasible iff ``x[0] ≤ 1``).
    """

    def __init__(
        self,
        lower: list[int],
        upper: list[int],
        *,
        constraint_names: list[str] | None = None,
        y_scale: float = 1.0,
    ) -> None:
        super().__init__(lower, upper)
        names = list(constraint_names) if constraint_names is not None else ["g0"]
        self._constraint_names = names
        self._y_scale = float(y_scale)
        self.hdb_eval_context = type(
            "StubEvalContext",
            (),
            {"_c_names": list(names), "last_data": None},
        )()

    def __call__(self, x):  # type: ignore[override]
        y_p = float(super().__call__(x))
        candidate = [int(v) for v in x]
        y = float(self._y_scale * sum(v * v for v in candidate))
        constraint_values: dict[str, float] = {}
        violations: dict[str, float] = {}
        for index, name in enumerate(self._constraint_names):
            if index == 0 and candidate:
                eta = float(candidate[0])
            else:
                eta = 0.0
            constraint_values[name] = eta
            violations[name] = float(max(0.0, eta - 1.0))
        feasible = all(value <= 1.0 for value in constraint_values.values())
        self.hdb_eval_context.last_data = {
            "y": y,
            "y_p": y_p,
            "constraint_values": constraint_values,
            "violations_": violations,
            "feasible": feasible,
        }
        self.hdb_last_feasible = feasible
        return y_p


class TestScipyCobyla(unittest.TestCase):
    """Registered SCIPY_COBYLA adapter behaviour."""

    def test_registry_spec_and_defaults(self) -> None:
        spec = get_algorithm_spec("SCIPY_COBYLA")
        self.assertEqual(spec.library, "scipy")
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.scipy.factory:build_scipy_cobyla",
        )
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.CONTINUOUS_ROUNDED)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertIsNone(spec.dependency_extra)
        self.assertFalse(spec.batch_evaluations)
        self.assertIn("constrained", spec.description.lower())
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations"],
        )
        normalized = normalize_algorithm_parameters(
            "SCIPY_COBYLA",
            {"max_evaluations": 20},
        )
        self.assertEqual(normalized, {"max_evaluations": 20})

    def test_package_export_matches_algorithms_export(self) -> None:
        self.assertIs(Cobyla, PackageCobyla)

    def test_rejects_zero_constraints(self) -> None:
        problem = _ConstrainedFakeIOHProblem(
            [0],
            [2],
            constraint_names=[],
        )
        algo = create_algorithm(
            "SCIPY_COBYLA",
            {"max_evaluations": 5},
            random_seed=7,
        )
        with self.assertRaisesRegex(ValueError, "at least one active"):
            algo(problem)

    def test_calls_minimize_with_locked_arguments(self) -> None:
        captured: dict = {}

        def _fake_minimize(fun, x0, **kwargs):  # noqa: ANN001
            captured["fun"] = fun
            captured["x0"] = list(x0)
            captured["kwargs"] = kwargs
            constraints = kwargs["constraints"]
            self.assertEqual(len(constraints), 1)
            constraint_fun = constraints[0].fun
            for point in ([1.2, 2.7], [0.4, 1.1], [2.0, 3.0]):
                try:
                    y = fun(point)
                    g = constraint_fun(point)
                    captured.setdefault("pairs", []).append((y, np.asarray(g)))
                except Exception:
                    break
            return None

        problem = _ConstrainedFakeIOHProblem([0, 0], [4, 5], y_scale=2.0)
        from core.ioh_core.algorithms.common import seeded_uniform_continuous_start

        expected_x0 = seeded_uniform_continuous_start([0, 0], [4, 5], 7)
        with patch("scipy.optimize.minimize", _fake_minimize):
            algo = create_algorithm(
                "SCIPY_COBYLA",
                {"max_evaluations": 9},
                random_seed=7,
            )
            result = algo(problem)

        self.assertIs(result, algo)
        self.assertEqual(captured["x0"], expected_x0)
        kwargs = captured["kwargs"]
        self.assertEqual(kwargs["method"], "COBYLA")
        self.assertEqual(kwargs["bounds"], [(0.0, 4.0), (0.0, 5.0)])
        self.assertEqual(kwargs["options"], {"maxiter": 9})
        self.assertEqual(len(kwargs["constraints"]), 1)
        # Obj + cons share one IOH eval per rounded integer point.
        self.assertEqual(len(problem.calls), 3)
        self.assertEqual(problem.calls[0], [1, 3])
        self.assertEqual(problem.calls[1], [0, 1])
        self.assertEqual(problem.calls[2], [2, 3])
        # First point [1,3]: y = 2*(1+9)=20; η=1 → g=0
        self.assertEqual(captured["pairs"][0][0], 20.0)
        np.testing.assert_allclose(captured["pairs"][0][1], [0.0])
        # Authoritative best remains y_p; best among calls is [0,1] → 1.0
        self.assertEqual(problem.state.current_best.y, 1.0)
        self.assertEqual(problem.state.current_best.x, [0, 1])

    def test_obj_and_constraint_share_one_ioh_eval(self) -> None:
        captured: dict = {}

        def _fake_minimize(fun, x0, **kwargs):  # noqa: ANN001
            constraint_fun = kwargs["constraints"][0].fun
            point = [1.4, 0.2]
            captured["y"] = fun(point)
            captured["g"] = np.asarray(constraint_fun(point))
            captured["y_again"] = fun(point)
            return None

        problem = _ConstrainedFakeIOHProblem(
            [0, 0],
            [3, 3],
            constraint_names=["g0"],
            y_scale=3.0,
        )
        with patch("scipy.optimize.minimize", _fake_minimize):
            create_algorithm(
                "SCIPY_COBYLA",
                {"max_evaluations": 5},
                random_seed=7,
            )(problem)
        self.assertEqual(len(problem.calls), 1)
        self.assertEqual(problem.calls[0], [1, 0])
        # y = 3*(1^2+0)=3; η=1 → g=0; y_p = 1 for current_best
        self.assertEqual(captured["y"], 3.0)
        self.assertEqual(captured["y_again"], 3.0)
        np.testing.assert_allclose(captured["g"], [0.0])
        self.assertEqual(problem.state.current_best.y, 1.0)

    def test_never_exceeds_budget(self) -> None:
        problem = _ConstrainedFakeIOHProblem([0, 0], [5, 5])
        create_algorithm(
            "SCIPY_COBYLA",
            {"max_evaluations": 15},
            random_seed=7,
        )(problem)
        self.assertLessEqual(len(problem.calls), 15)
        self.assertEqual(problem.state.evaluations, len(problem.calls))
        self.assertGreaterEqual(len(problem.calls), 1)

    def test_controller_enforces_exact_budget_when_solver_keeps_asking(self) -> None:
        """Force many distinct proposals so RunController, not SciPy, stops."""

        def _fake_minimize(fun, x0, **kwargs):  # noqa: ANN001
            constraint_fun = kwargs["constraints"][0].fun
            for index in range(200):
                # Unique integer grid points in [0,8]^2 (row-major).
                point = [float(index % 9), float((index // 9) % 9)]
                try:
                    fun(point)
                    constraint_fun(point)
                except Exception:
                    break
            return None

        problem = _ConstrainedFakeIOHProblem([0, 0], [8, 8])
        with patch("scipy.optimize.minimize", _fake_minimize):
            create_algorithm(
                "SCIPY_COBYLA",
                {"max_evaluations": 12},
                random_seed=7,
            )(problem)
        self.assertEqual(len(problem.calls), 12)
        self.assertEqual(problem.state.evaluations, 12)

    def test_seeded_start_deterministic_and_in_bounds(self) -> None:
        from core.ioh_core.algorithms.common import (
            continuous_to_integer_candidate,
            seeded_uniform_continuous_start,
        )

        problem = _ConstrainedFakeIOHProblem([0, 2], [4, 6])
        expected_x0 = seeded_uniform_continuous_start([0, 2], [4, 6], 7)
        expected_first = continuous_to_integer_candidate(
            expected_x0,
            lower_bounds=[0, 2],
            upper_bounds=[4, 6],
        )
        create_algorithm(
            "SCIPY_COBYLA",
            {"max_evaluations": 10},
            random_seed=7,
        )(problem)
        self.assertEqual(problem.calls[0], expected_first)
        create_algorithm(
            "SCIPY_COBYLA",
            {"max_evaluations": 10},
            random_seed=7,
        )(_ConstrainedFakeIOHProblem([0, 2], [4, 6]))
        # Same seed → same continuous start → same first integer candidate.
        problem_b = _ConstrainedFakeIOHProblem([0, 2], [4, 6])
        create_algorithm(
            "SCIPY_COBYLA",
            {"max_evaluations": 10},
            random_seed=7,
        )(problem_b)
        self.assertEqual(problem_b.calls[0], expected_first)
        for candidate in problem.calls:
            self.assertEqual(len(candidate), 2)
            self.assertTrue(all(isinstance(value, int) for value in candidate))
            self.assertGreaterEqual(candidate[0], 0)
            self.assertLessEqual(candidate[0], 4)
            self.assertGreaterEqual(candidate[1], 2)
            self.assertLessEqual(candidate[1], 6)

    def test_does_not_re_evaluate_best(self) -> None:
        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            "SCIPY_COBYLA",
            {"max_evaluations": 10},
            random_seed=7,
        )(problem)
        call_count = len(problem.calls)
        self.assertEqual(problem.state.evaluations, call_count)
        self.assertIn(problem.state.current_best.x, problem.calls)

    def test_target_hit_stops_early(self) -> None:
        problem = _ConstrainedFakeIOHProblem([0, 0], [0, 0])
        controller = RunController(
            problem,
            20,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "SCIPY_COBYLA",
            {"max_evaluations": 20},
            random_seed=7,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 1)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "target_hit")

    def test_target_miss_respects_budget(self) -> None:
        def _fake_minimize(fun, x0, **kwargs):  # noqa: ANN001
            constraint_fun = kwargs["constraints"][0].fun
            # Skip the origin so y_p never hits threshold 0 under fixed_target.
            for index in range(1, 200):
                point = [float(index % 9), float((index // 9) % 9)]
                try:
                    fun(point)
                    constraint_fun(point)
                except Exception:
                    break
            return None

        problem = _ConstrainedFakeIOHProblem([0, 0], [8, 8])
        controller = RunController(
            problem,
            11,
            mode="fixed_target",
            target_threshold=0.0,
            target_reference="all",
        )
        with patch("scipy.optimize.minimize", _fake_minimize):
            create_algorithm(
                "SCIPY_COBYLA",
                {"max_evaluations": 11},
                random_seed=7,
            )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 11)
        self.assertFalse(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")

    def test_seed_required(self) -> None:
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "SCIPY_COBYLA",
                {"max_evaluations": 5},
                random_seed=None,
            )
        with self.assertRaises(ValueError):
            build_scipy_cobyla({"max_evaluations": 5}, random_seed=None)

    def test_unknown_parameter_rejected(self) -> None:
        with self.assertRaises(AlgorithmFactoryError):
            validate_algorithm_parameters(
                "SCIPY_COBYLA",
                {"max_evaluations": 5, "rhobeg": 0.5},
                random_seed=7,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "SCIPY_COBYLA",
                {"max_evaluations": 5, "rhobeg": 0.5},
                random_seed=7,
            )


class TestScipyCobylaWarmStart(unittest.TestCase):
    """COBYLA starts at rank 1 and the cold class keeps its seeded start."""

    def test_cold_class_still_draws_a_seeded_start(self) -> None:
        from core.ioh_core.algorithms.common import (
            continuous_to_integer_candidate,
            seeded_uniform_continuous_start,
        )

        lower = [0, 0]
        upper = [4, 4]
        problem = _ConstrainedFakeIOHProblem(lower, upper)
        Cobyla(max_evaluations=1, seed=3)(problem)
        expected = continuous_to_integer_candidate(
            seeded_uniform_continuous_start(lower, upper, 3),
            lower_bounds=lower,
            upper_bounds=upper,
        )
        self.assertEqual(problem.calls[0], expected)

    def test_rank1_is_the_first_evaluation(self) -> None:
        import json
        from tempfile import TemporaryDirectory

        from core.ioh_core.algorithms.scipy.cobyla import CobylaWarmStart

        vector = [1, 0, 0, 0, 0, 0, 0, 0, 2]
        lower = [0] * 9
        upper = [5] * 9
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "problem_id": "L8100_B2",
                        "n": 1,
                        "vectors": [vector],
                    }
                ),
                encoding="utf-8",
            )
            problem = _ConstrainedFakeIOHProblem(lower, upper)
            CobylaWarmStart(1, str(path))(problem)
            self.assertEqual(problem.calls, [vector])
            longer = _ConstrainedFakeIOHProblem(lower, upper)
            CobylaWarmStart(2, str(path))(longer)
            self.assertEqual(longer.calls[0], vector)
            self.assertGreaterEqual(len(longer.calls), 2)

    def test_missing_file_and_short_budget_evaluate_nothing(self) -> None:
        import json
        from tempfile import TemporaryDirectory

        from core.ioh_core.algorithms.scipy.cobyla import CobylaWarmStart

        lower = [0] * 9
        upper = [5] * 9
        missing = _ConstrainedFakeIOHProblem(lower, upper)
        with self.assertRaisesRegex(ValueError, "has no InitialDesign"):
            CobylaWarmStart(1, "missing-initial-design.json")(missing)
        self.assertEqual(missing.calls, [])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "problem_id": "L8100_B2",
                        "n": 1,
                        "vectors": [[1, 0, 0, 0, 0, 0, 0, 0, 0]],
                    }
                ),
                encoding="utf-8",
            )
            problem = _ConstrainedFakeIOHProblem(lower, upper)
            with self.assertRaisesRegex(
                ValueError,
                "needs 1 initial evaluations but max_evaluations is 0",
            ):
                CobylaWarmStart(0, str(path))(problem)
        self.assertEqual(problem.calls, [])


if __name__ == "__main__":
    unittest.main()
