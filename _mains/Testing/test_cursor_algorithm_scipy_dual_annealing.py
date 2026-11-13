"""
SciPy dual annealing: global phase only, continuous-rounded, y_p only.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from core.ioh_core.algorithm_factory import (
    AlgorithmFactoryError,
    create_algorithm,
)
from core.ioh_core.algorithms import DualAnnealing
from core.ioh_core.algorithms.common import (
    RunController,
    continuous_to_integer_candidate,
)
from core.ioh_core.algorithms.registry import (
    BudgetSemantics,
    CandidateMode,
    SeedPolicy,
    get_algorithm_spec,
)
from core.ioh_core.algorithms.scipy.dual_annealing import (
    DualAnnealing as PackageDualAnnealing,
)
from core.ioh_core.algorithms.warm_start import RANK1_SEEDLESS_TOKENS
from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem


def _write_design(path: Path, vectors: list[list[int]]) -> None:
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


class TestScipyDualAnnealing(unittest.TestCase):
    """Registered SCIPY_DUAL_ANNEALING adapter behaviour."""

    def test_registry_spec_and_seed_policy(self) -> None:
        for name in ("SCIPY_DUAL_ANNEALING", "SCIPY_DUAL_ANNEALING_WARM_START"):
            spec = get_algorithm_spec(name)
            self.assertEqual(spec.library, "scipy")
            self.assertIsNone(spec.dependency_extra)
            self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
            self.assertEqual(spec.candidate_mode, CandidateMode.CONTINUOUS_ROUNDED)
            self.assertEqual(
                spec.budget_semantics,
                BudgetSemantics.HARD_IOH_EVALUATIONS,
            )
            self.assertFalse(spec.batch_evaluations)
            self.assertIn("y_p", spec.description)
            self.assertIn("local search", spec.description)
        self.assertNotIn(
            "SCIPY_DUAL_ANNEALING_WARM_START",
            RANK1_SEEDLESS_TOKENS,
        )
        cold = get_algorithm_spec("SCIPY_DUAL_ANNEALING")
        self.assertEqual(
            cold.constructor_ref,
            "core.ioh_core.algorithms.scipy.factory:build_scipy_dual_annealing",
        )
        self.assertEqual(
            [parameter.name for parameter in cold.parameters],
            ["max_evaluations"],
        )
        warm = get_algorithm_spec("SCIPY_DUAL_ANNEALING_WARM_START")
        self.assertEqual(
            [parameter.name for parameter in warm.parameters],
            ["max_evaluations", "initial_design_path"],
        )

    def test_package_export_matches_algorithms_export(self) -> None:
        self.assertIs(DualAnnealing, PackageDualAnnealing)

    def test_cold_and_warm_require_a_seed(self) -> None:
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "SCIPY_DUAL_ANNEALING",
                {"max_evaluations": 4},
                random_seed=None,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "SCIPY_DUAL_ANNEALING_WARM_START",
                {
                    "max_evaluations": 4,
                    "initial_design_path": "missing.json",
                },
                random_seed=None,
            )

    def test_warm_requires_initial_design_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial_design_path"):
            create_algorithm(
                "SCIPY_DUAL_ANNEALING_WARM_START",
                {"max_evaluations": 4},
                random_seed=3,
            )

    def test_locked_scipy_arguments_and_no_supplied_cold_start(self) -> None:
        captured: dict = {}

        def _fake_dual_annealing(fun, bounds, **kwargs):  # noqa: ANN001
            captured["fun"] = fun
            captured["bounds"] = bounds
            captured["kwargs"] = kwargs
            fun([0.2, 1.2])
            return None

        problem = FakeIOHProblem([0, 0], [4, 5])
        with patch("scipy.optimize.dual_annealing", _fake_dual_annealing):
            algo = create_algorithm(
                "SCIPY_DUAL_ANNEALING",
                {"max_evaluations": 9},
                random_seed=7,
            )
            result = algo(problem)

        self.assertIs(result, algo)
        kwargs = captured["kwargs"]
        self.assertIsNone(kwargs["x0"])
        self.assertTrue(kwargs["no_local_search"])
        self.assertEqual(kwargs["maxfun"], 9)
        self.assertEqual(kwargs["maxiter"], 9)
        self.assertEqual(kwargs["rng"], 7)
        self.assertEqual(kwargs["initial_temp"], 5230.0)
        self.assertEqual(kwargs["restart_temp_ratio"], 2e-5)
        self.assertEqual(kwargs["visit"], 2.62)
        self.assertEqual(kwargs["accept"], -5.0)
        self.assertEqual(captured["bounds"], [(0.0, 4.0), (0.0, 5.0)])
        self.assertEqual(captured["fun"]([1.0, 1.0]), 2.0)

    def test_objective_is_y_p_and_slacks_are_not_read(self) -> None:
        def _fake_dual_annealing(fun, bounds, **kwargs):  # noqa: ANN001
            fun([1.2, 0.2])
            return None

        problem = FakeIOHProblem([0, 0], [3, 3])
        with (
            patch("scipy.optimize.dual_annealing", _fake_dual_annealing),
            patch(
                "core.ioh_core.algorithms.common.read_last_evaluation_constraint_slacks"
            ) as slacks,
        ):
            create_algorithm(
                "SCIPY_DUAL_ANNEALING",
                {"max_evaluations": 4},
                random_seed=1,
            )(problem)
        slacks.assert_not_called()
        self.assertEqual(problem.calls, [[1, 0]])
        self.assertEqual(problem.state.current_best.y, 1.0)

    def test_repeated_rounded_cells_count_separately(self) -> None:
        def _fake_dual_annealing(fun, bounds, **kwargs):  # noqa: ANN001
            fun([0.1, 0.2])
            fun([0.4, 0.2])
            return None

        class _LoggingProblem(FakeIOHProblem):
            def __init__(self) -> None:
                super().__init__([0, 0], [3, 3])
                self.log: list[list[int]] = []

            def __call__(self, candidate):  # noqa: ANN001
                self.log.append(list(candidate))
                return super().__call__(candidate)

        problem = _LoggingProblem()
        with patch("scipy.optimize.dual_annealing", _fake_dual_annealing):
            create_algorithm(
                "SCIPY_DUAL_ANNEALING",
                {"max_evaluations": 4},
                random_seed=1,
            )(problem)
        self.assertEqual(problem.calls, [[0, 0], [0, 0]])
        self.assertEqual(problem.log, [[0, 0], [0, 0]])
        self.assertEqual(problem.state.evaluations, 2)

    def test_rounding_matches_the_shared_policy(self) -> None:
        recorded: list = []

        def _fake_dual_annealing(fun, bounds, **kwargs):  # noqa: ANN001
            recorded.append(fun([1.5, 4.8]))
            recorded.append(fun([-0.2, 0.4]))
            return None

        lower = [0, 0]
        upper = [3, 4]
        problem = FakeIOHProblem(lower, upper)
        with patch("scipy.optimize.dual_annealing", _fake_dual_annealing):
            create_algorithm(
                "SCIPY_DUAL_ANNEALING",
                {"max_evaluations": 4},
                random_seed=1,
            )(problem)
        self.assertEqual(
            problem.calls[0],
            continuous_to_integer_candidate(
                [1.5, 4.8],
                lower_bounds=lower,
                upper_bounds=upper,
            ),
        )
        self.assertEqual(
            problem.calls[1],
            continuous_to_integer_candidate(
                [-0.2, 0.4],
                lower_bounds=lower,
                upper_bounds=upper,
            ),
        )
        self.assertEqual(recorded[0], float(sum(v * v for v in problem.calls[0])))

    def test_degenerate_bounds_fail_before_scipy(self) -> None:
        def _fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("SciPy dual_annealing was called")

        problem = FakeIOHProblem([0, 0, 0, 0, 3], [2, 2, 2, 2, 3])
        with patch("scipy.optimize.dual_annealing", _fail_if_called):
            with self.assertRaisesRegex(
                ValueError,
                r"coordinate 4 has bounds \[3, 3\]",
            ):
                create_algorithm(
                    "SCIPY_DUAL_ANNEALING",
                    {"max_evaluations": 4},
                    random_seed=1,
                )(problem)
        self.assertEqual(problem.calls, [])

    def test_never_exceeds_budget_and_a_representative_run_uses_it_all(self) -> None:
        problem = FakeIOHProblem([0, 0], [6, 6])
        create_algorithm(
            "SCIPY_DUAL_ANNEALING",
            {"max_evaluations": 20},
            random_seed=7,
        )(problem)
        self.assertEqual(len(problem.calls), 20)
        self.assertEqual(problem.state.evaluations, 20)

    def test_fixed_target_stops_early(self) -> None:
        problem = FakeIOHProblem([0, 0], [5, 5])
        controller = RunController(
            problem,
            30,
            mode="fixed_target",
            target_threshold=1.0e9,
            target_reference="all",
        )
        result = create_algorithm(
            "SCIPY_DUAL_ANNEALING",
            {"max_evaluations": 30},
            random_seed=7,
        )(problem, run_controller=controller)
        self.assertIsInstance(result, DualAnnealing)
        self.assertEqual(controller.stop_reason(), "target_hit")
        self.assertLess(len(problem.calls), 30)
        self.assertGreaterEqual(len(problem.calls), 1)

    def test_cold_seed_reproduces_and_a_second_seed_may_diverge(self) -> None:
        def run(seed: int) -> list[list[int]]:
            problem = FakeIOHProblem([0, 0], [8, 8])
            create_algorithm(
                "SCIPY_DUAL_ANNEALING",
                {"max_evaluations": 12},
                random_seed=seed,
            )(problem)
            return problem.calls

        self.assertEqual(run(11), run(11))
        self.assertNotEqual(run(11), run(29))

    def test_warm_start_first_evaluation_is_rank_1(self) -> None:
        rank1 = [4, 2, 1, 0, 3, 1, 2, 0, 1]
        zeros = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            _write_design(path, [rank1, zeros])
            problem = FakeIOHProblem(zeros, [8] * 9)
            create_algorithm(
                "SCIPY_DUAL_ANNEALING_WARM_START",
                {
                    "max_evaluations": 8,
                    "initial_design_path": str(path),
                },
                random_seed=5,
            )(problem)
        self.assertEqual(problem.calls[0], rank1)

    def test_warm_seed_reproduces_and_a_second_seed_may_diverge(self) -> None:
        rank1 = [3, 1, 0, 2, 1, 4, 0, 2, 1]
        zeros = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        def run(seed: int) -> list[list[int]]:
            with TemporaryDirectory() as tmp:
                path = Path(tmp) / "initial_design.json"
                _write_design(path, [rank1, zeros])
                problem = FakeIOHProblem(zeros, [8] * 9)
                create_algorithm(
                    "SCIPY_DUAL_ANNEALING_WARM_START",
                    {
                        "max_evaluations": 12,
                        "initial_design_path": str(path),
                    },
                    random_seed=seed,
                )(problem)
            return problem.calls

        first = run(11)
        self.assertEqual(first[0], rank1)
        self.assertEqual(first, run(11))
        self.assertNotEqual(first, run(29))
        self.assertEqual(run(29)[0], rank1)


if __name__ == "__main__":
    unittest.main()
