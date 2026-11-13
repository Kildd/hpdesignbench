"""
Phase ALG7.1 / C4.5 gate: NLopt DIRECT-L and SBPLX adapters.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import nlopt

from core.ioh_core.algorithm_factory import create_algorithm
from core.ioh_core.algorithms.common import RunController
from core.ioh_core.algorithms.registry import (
    BudgetSemantics,
    CandidateMode,
    SeedPolicy,
    get_algorithm_spec,
    normalize_algorithm_parameters,
)
from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem


def _fixed_target_controller(
    problem: FakeIOHProblem,
    max_evaluations: int,
    *,
    threshold: float,
    target_reference: str = "all",
) -> RunController:
    """Build a fixed-target controller for NLopt adapter tests."""
    return RunController(
        problem,
        max_evaluations,
        mode="fixed_target",
        target_threshold=threshold,
        target_reference=target_reference,
    )


class TestNLoptDirectFixedTarget(unittest.TestCase):
    """NLOPT_DIRECT_L fixed-target stopping."""

    def test_direct_l_target_hit_stops_without_extra_call(self) -> None:
        problem = FakeIOHProblem([0, 0], [0, 0])
        controller = _fixed_target_controller(problem, 8, threshold=1.0)
        f_best, x_best = create_algorithm(
            "NLOPT_DIRECT_L",
            {"max_evaluations": 8},
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 1)
        self.assertEqual(x_best, [0, 0])
        self.assertEqual(f_best, 0.0)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.evaluations_to_target, 1)
        self.assertEqual(controller.stop_reason(), "target_hit")

    def test_direct_l_target_miss_uses_exact_hard_ceiling(self) -> None:
        problem = FakeIOHProblem([2, 2], [4, 4])
        controller = _fixed_target_controller(problem, 5, threshold=1.0)
        create_algorithm(
            "NLOPT_DIRECT_L",
            {"max_evaluations": 5},
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 5)
        self.assertFalse(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")

    def test_direct_l_all_reference_accepts_finite_hits(self) -> None:
        problem = FakeIOHProblem([2, 2], [4, 4])
        problem.next_feasible = False
        controller = _fixed_target_controller(
            problem,
            4,
            threshold=100.0,
            target_reference="all",
        )
        create_algorithm(
            "NLOPT_DIRECT_L",
            {"max_evaluations": 4},
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 1)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "target_hit")

    def test_direct_l_forced_stop_is_successful_completion(self) -> None:
        from core.ioh_core.algorithms.nlopt import algorithms as nlopt_algorithms
        from core.ioh_core.algorithms.nlopt.algorithms import NLoptDirectLocalSearch

        problem = FakeIOHProblem([0, 0], [3, 3])
        adapter = NLoptDirectLocalSearch(max_evaluations=5)

        class _FakeOpt:
            def __init__(self, *args, **kwargs):
                self._objective = None

            def set_lower_bounds(self, lb):
                return None

            def set_upper_bounds(self, ub):
                return None

            def set_min_objective(self, objective):
                self._objective = objective

            def set_maxeval(self, maxeval):
                return None

            def set_stopval(self, stopval):
                return None

            def optimize(self, x0):
                self._objective(x0, None)
                raise nlopt.ForcedStop()

        with patch.object(nlopt_algorithms.nlopt, "opt", side_effect=_FakeOpt):
            f_best, x_best = adapter(problem)

        self.assertEqual(len(problem.calls), 1)
        self.assertEqual(list(x_best), problem.calls[0])
        self.assertEqual(f_best, float(sum(v * v for v in x_best)))

    def test_direct_l_unrelated_callback_failure_propagates(self) -> None:
        from core.ioh_core.algorithms.nlopt import algorithms as nlopt_algorithms
        from core.ioh_core.algorithms.nlopt.algorithms import NLoptDirectLocalSearch

        problem = FakeIOHProblem([0, 0], [3, 3])
        adapter = NLoptDirectLocalSearch(max_evaluations=5)

        class _FakeOpt:
            def __init__(self, *args, **kwargs):
                self._objective = None

            def set_lower_bounds(self, lb):
                return None

            def set_upper_bounds(self, ub):
                return None

            def set_min_objective(self, objective):
                self._objective = objective

            def set_maxeval(self, maxeval):
                return None

            def set_stopval(self, stopval):
                return None

            def optimize(self, x0):
                raise RuntimeError("unexpected callback failure")

        with patch.object(nlopt_algorithms.nlopt, "opt", side_effect=_FakeOpt):
            with self.assertRaises(RuntimeError) as ctx:
                adapter(problem)
        self.assertIn("unexpected callback failure", str(ctx.exception))


class TestNLoptSbplx(unittest.TestCase):
    """Registered NLOPT_SBPLX adapter behaviour."""

    def test_sbplx_registry_spec(self) -> None:
        spec = get_algorithm_spec("NLOPT_SBPLX")
        self.assertEqual(spec.library, "nlopt")
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.nlopt.factory:build_sbplx",
        )
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.CONTINUOUS_ROUNDED)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertIsNone(spec.dependency_extra)
        self.assertFalse(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations"],
        )
        normalized = normalize_algorithm_parameters(
            "NLOPT_SBPLX",
            {"max_evaluations": 10},
        )
        self.assertEqual(normalized, {"max_evaluations": 10})

    def test_sbplx_uses_seeded_start_and_integer_candidates(self) -> None:
        from core.ioh_core.algorithms.common import seeded_uniform_continuous_start
        from core.ioh_core.algorithms.nlopt import algorithms as nlopt_algorithms

        captured: dict = {}
        real_opt = nlopt.opt
        seed = 7
        expected_x0 = seeded_uniform_continuous_start([0, 2], [4, 6], seed)

        class _TrackingOpt:
            def __init__(self, algorithm, dimension):
                self._opt = real_opt(algorithm, dimension)
                captured["algorithm"] = algorithm
                captured["dimension"] = dimension

            def set_lower_bounds(self, lb):
                captured["lb"] = list(lb)
                return self._opt.set_lower_bounds(lb)

            def set_upper_bounds(self, ub):
                captured["ub"] = list(ub)
                return self._opt.set_upper_bounds(ub)

            def set_min_objective(self, objective):
                captured["objective"] = objective
                return self._opt.set_min_objective(objective)

            def set_maxeval(self, maxeval):
                captured["maxeval"] = maxeval
                return self._opt.set_maxeval(maxeval)

            def set_stopval(self, stopval):
                captured["stopval"] = stopval
                return self._opt.set_stopval(stopval)

            def optimize(self, x0):
                captured["x0"] = list(x0)
                return self._opt.optimize(x0)

        problem = FakeIOHProblem([0, 2], [4, 6])
        with patch.object(nlopt_algorithms.nlopt, "opt", side_effect=_TrackingOpt):
            algo = create_algorithm(
                "NLOPT_SBPLX",
                {"max_evaluations": 8},
                random_seed=seed,
            )
            f_best, x_best = algo(problem)

        self.assertEqual(captured["algorithm"], nlopt.LN_SBPLX)
        self.assertEqual(captured["maxeval"], 8)
        self.assertNotIn("stopval", captured)
        self.assertEqual(captured["x0"], expected_x0)
        self.assertGreaterEqual(len(problem.calls), 1)
        self.assertTrue(all(isinstance(v, int) for c in problem.calls for v in c))
        self.assertIn(list(x_best), problem.calls)
        self.assertIsInstance(f_best, float)

    def test_sbplx_tracks_only_evaluated_best(self) -> None:
        from core.ioh_core.algorithms.common import seeded_uniform_continuous_start
        from core.ioh_core.algorithms.nlopt import algorithms as nlopt_algorithms
        from core.ioh_core.algorithms.nlopt.algorithms import NLoptSbplx

        seed = 7
        problem = FakeIOHProblem([0, 0], [3, 3])
        expected_x0 = seeded_uniform_continuous_start([0, 0], [3, 3], seed)
        adapter = NLoptSbplx(max_evaluations=5, seed=seed)
        captured_x0: list[list[float]] = []

        class _FakeOpt:
            def __init__(self, *args, **kwargs):
                self._objective = None

            def set_lower_bounds(self, lb):
                return None

            def set_upper_bounds(self, ub):
                return None

            def set_min_objective(self, objective):
                self._objective = objective

            def set_maxeval(self, maxeval):
                return None

            def set_stopval(self, stopval):
                return None

            def optimize(self, x0):
                captured_x0.append(list(x0))
                self._objective(x0, None)
                self._objective([1.0, 2.0], None)
                raise nlopt.RoundoffLimited()

        with patch.object(nlopt_algorithms.nlopt, "opt", side_effect=_FakeOpt):
            f_best, x_best = adapter(problem)

        self.assertEqual(captured_x0, [expected_x0])
        from core.ioh_core.algorithms.common import continuous_to_integer_candidate

        first = continuous_to_integer_candidate(
            expected_x0,
            lower_bounds=[0, 0],
            upper_bounds=[3, 3],
        )
        # First evaluated point from seeded x0; second [1, 2] (y=5).
        self.assertEqual(problem.calls[0], first)
        self.assertEqual(problem.calls[1], [1, 2])
        self.assertEqual(x_best, [1, 2])
        self.assertEqual(f_best, 5.0)
        self.assertNotIn([9, 9], problem.calls)

    def test_sbplx_does_not_re_evaluate_best(self) -> None:
        problem = FakeIOHProblem([0, 0], [3, 3])
        algo = create_algorithm(
            "NLOPT_SBPLX",
            {"max_evaluations": 10},
            random_seed=7,
        )
        f_best, x_best = algo(problem)
        call_count = len(problem.calls)
        self.assertEqual(problem.state.evaluations, call_count)
        self.assertLessEqual(call_count, 10)
        self.assertEqual(algo.n_optimizer_evaluations_, call_count)
        self.assertIn(list(x_best), problem.calls)
        self.assertEqual(f_best, problem.state.current_best.y)
        # No post-optimize re-evaluation of the returned best.
        self.assertEqual(len(problem.calls), call_count)
        self.assertEqual(problem.state.evaluations, call_count)

    def test_sbplx_target_hit_stops_without_extra_call(self) -> None:
        problem = FakeIOHProblem([0, 0], [0, 0])
        controller = _fixed_target_controller(problem, 8, threshold=1.0)
        f_best, x_best = create_algorithm(
            "NLOPT_SBPLX",
            {"max_evaluations": 8},
            random_seed=7,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 1)
        self.assertEqual(x_best, [0, 0])
        self.assertEqual(f_best, 0.0)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "target_hit")

    def test_sbplx_target_miss_uses_exact_hard_ceiling(self) -> None:
        problem = FakeIOHProblem([2, 2], [4, 4])
        controller = _fixed_target_controller(problem, 5, threshold=1.0)
        create_algorithm(
            "NLOPT_SBPLX",
            {"max_evaluations": 5},
            random_seed=7,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 5)
        self.assertFalse(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")

    def test_sbplx_sets_stopval_for_all_reference(self) -> None:
        from core.ioh_core.algorithms.nlopt import algorithms as nlopt_algorithms

        captured: dict = {}
        real_opt = nlopt.opt

        class _TrackingOpt:
            def __init__(self, algorithm, dimension):
                self._opt = real_opt(algorithm, dimension)

            def set_lower_bounds(self, lb):
                return self._opt.set_lower_bounds(lb)

            def set_upper_bounds(self, ub):
                return self._opt.set_upper_bounds(ub)

            def set_min_objective(self, objective):
                return self._opt.set_min_objective(objective)

            def set_maxeval(self, maxeval):
                return self._opt.set_maxeval(maxeval)

            def set_stopval(self, stopval):
                captured["stopval"] = stopval
                return self._opt.set_stopval(stopval)

            def optimize(self, x0):
                return self._opt.optimize(x0)

        problem = FakeIOHProblem([0, 0], [0, 0])
        controller = _fixed_target_controller(problem, 3, threshold=2.5)
        with patch.object(nlopt_algorithms.nlopt, "opt", side_effect=_TrackingOpt):
            create_algorithm(
                "NLOPT_SBPLX",
                {"max_evaluations": 3},
                random_seed=7,
            )(problem, run_controller=controller)
        self.assertEqual(captured["stopval"], 2.5)


class TestNLoptSbplxWarmStart(unittest.TestCase):
    """SBPLX starts at rank 1 and the cold class keeps its seeded start."""

    def test_cold_class_still_draws_a_seeded_start(self) -> None:
        from core.ioh_core.algorithms.common import (
            continuous_to_integer_candidate,
            seeded_uniform_continuous_start,
        )
        from core.ioh_core.algorithms.nlopt.algorithms import NLoptSbplx

        lower = [0, 0]
        upper = [4, 4]
        problem = FakeIOHProblem(lower, upper)
        NLoptSbplx(max_evaluations=1, seed=3)(problem)
        expected = continuous_to_integer_candidate(
            seeded_uniform_continuous_start(lower, upper, 3),
            lower_bounds=lower,
            upper_bounds=upper,
        )
        self.assertEqual(problem.calls[0], expected)

    def test_rank1_missing_file_and_short_budget(self) -> None:
        import json
        from tempfile import TemporaryDirectory

        from core.ioh_core.algorithms.nlopt.algorithms import NLoptSbplxWarmStart

        vector = [1, 0, 0, 0, 0, 0, 0, 0, 2]
        lower = [0] * 9
        upper = [5] * 9
        missing = FakeIOHProblem(lower, upper)
        with self.assertRaisesRegex(ValueError, "has no InitialDesign"):
            NLoptSbplxWarmStart(
                max_evaluations=1,
                initial_design_path="missing-initial-design.json",
            )(missing)
        self.assertEqual(missing.calls, [])
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
            problem = FakeIOHProblem(lower, upper)
            NLoptSbplxWarmStart(
                max_evaluations=1,
                initial_design_path=str(path),
            )(problem)
            self.assertEqual(problem.calls[0], vector)
            short = FakeIOHProblem(lower, upper)
            with self.assertRaisesRegex(
                ValueError,
                "needs 1 initial evaluations but max_evaluations is 0",
            ):
                NLoptSbplxWarmStart(
                    max_evaluations=0,
                    initial_design_path=str(path),
                )(short)
            self.assertEqual(short.calls, [])


if __name__ == "__main__":
    unittest.main()
