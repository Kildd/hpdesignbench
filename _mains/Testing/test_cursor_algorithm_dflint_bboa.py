"""
Phase DF1 gate: DFL package laziness, Halton shim, and DFLINT_BBOA.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem


class _ConstrainedFakeIOHProblem(FakeIOHProblem):
    """
    Fake IOH problem publishing unpenalized ``y`` and utilizations η.

    Sphere objective is used as both ``y`` and ``y_p`` unless ``y_scale``
    multiplies the unpenalized value. Constraint ``g0`` uses utilization
    ``η = x[0]`` so signed slack is ``η − 1 = x[0] − 1`` (feasible for
    ``x[0] ≤ 1``).
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


class TestDflPackageScaffold(unittest.TestCase):
    """Optional dfl package remains lazy and Halton/DFLINT smoke cleanly."""

    def test_dfl_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name.startswith("core.ioh_core.algorithms.dfl"):
                sys.modules.pop(name, None)

        mod = importlib.import_module("core.ioh_core.algorithms.dfl")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn(
            "core.ioh_core.algorithms.dfl._vendor.DFLINT",
            sys.modules,
        )

    def test_halton_shim_and_dflint_smoke(self) -> None:
        import numpy as np

        from core.ioh_core.algorithms.dfl.halton import Halton

        points = Halton(2).get(4)
        self.assertEqual(len(points), 4)
        self.assertEqual(len(points[0]), 2)

        dflint = importlib.import_module(
            "core.ioh_core.algorithms.dfl._vendor.DFLINT"
        )
        calls: list[tuple[int, float, float]] = []

        def func_f(x):
            x = np.asarray(x).reshape(-1)
            f = float((x[0] - 2.0) ** 2)
            g = np.array([x[0] - 3.0], dtype=float)
            calls.append((int(round(float(x[0]))), f, float(g[0])))
            return f, g

        dflint.func_f = func_f
        x, f, stop, _d = dflint.DFLINT(
            "BBOA",
            1,
            4,
            np.array([5.0]),
            np.array([0.0]),
            np.array([5.0]),
            40,
            0,
        )
        best = float(np.asarray(x).reshape(-1)[0])
        self.assertEqual(best, 2.0)
        self.assertEqual(float(f), 0.0)
        self.assertGreaterEqual(len(calls), 1)


class TestConstraintSlacksHelper(unittest.TestCase):
    """Signed η−1 helper for DFL-style constraints."""

    def test_reads_signed_slacks_not_clipped_violations(self) -> None:
        from core.ioh_core.algorithms.common import (
            read_last_evaluation_constraint_slacks,
            read_last_evaluation_violations,
        )

        problem = _ConstrainedFakeIOHProblem([0], [3])
        problem([0])
        slacks = read_last_evaluation_constraint_slacks(problem)
        violations = read_last_evaluation_violations(problem)
        self.assertEqual(slacks, [-1.0])
        self.assertEqual(violations, [0.0])

        problem([3])
        slacks = read_last_evaluation_constraint_slacks(problem)
        violations = read_last_evaluation_violations(problem)
        self.assertEqual(slacks, [2.0])
        self.assertEqual(violations, [2.0])


class TestDflintBboaAdapter(unittest.TestCase):
    """Registered DFLINT_BBOA adapter behaviour."""

    token = "DFLINT_BBOA"

    def test_registry_metadata(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec(self.token)
        self.assertEqual(spec.library, "dfl")
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-dfl")
        self.assertFalse(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations"],
        )
        normalized = normalize_algorithm_parameters(
            self.token,
            {"max_evaluations": 10},
        )
        self.assertEqual(normalized, {"max_evaluations": 10})

    def test_respects_hard_budget(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            self.token,
            {"max_evaluations": 10},
            random_seed=8,
        )(problem)
        self.assertLessEqual(len(problem.calls), 10)
        self.assertGreaterEqual(len(problem.calls), 1)
        self.assertEqual(problem.state.evaluations, len(problem.calls))

    def test_rejects_zero_constraints(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem(
            [0],
            [2],
            constraint_names=[],
        )
        algo = create_algorithm(
            self.token,
            {"max_evaluations": 5},
            random_seed=1,
        )
        with self.assertRaisesRegex(ValueError, "at least one active"):
            algo(problem)

    def test_deterministic_with_seed(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem_a = _ConstrainedFakeIOHProblem([0, 1], [3, 4])
        problem_b = _ConstrainedFakeIOHProblem([0, 1], [3, 4])
        create_algorithm(
            self.token,
            {"max_evaluations": 12},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            self.token,
            {"max_evaluations": 12},
            random_seed=17,
        )(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)
        self.assertGreaterEqual(len(problem_a.calls), 1)
        self.assertTrue(
            all(isinstance(v, int) for c in problem_a.calls for v in c)
        )

    def test_different_seed_different_sequence(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem_a = _ConstrainedFakeIOHProblem([0, 1], [5, 6])
        problem_b = _ConstrainedFakeIOHProblem([0, 1], [5, 6])
        create_algorithm(
            self.token,
            {"max_evaluations": 20},
            random_seed=3,
        )(problem_a)
        create_algorithm(
            self.token,
            {"max_evaluations": 20},
            random_seed=99,
        )(problem_b)
        self.assertNotEqual(problem_a.calls, problem_b.calls)

    def test_missing_seed_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
        )

        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                self.token,
                {"max_evaluations": 5},
                random_seed=None,
            )

    def test_unknown_parameter_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
        )

        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                self.token,
                {"max_evaluations": 5, "mm": 2},
                random_seed=1,
            )

    def test_uses_unpenalized_y_scale(self) -> None:
        """Search sees scaled y; IOH current_best tracks unscaled y_p."""
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem(
            [0],
            [3],
            y_scale=2.0,
        )
        create_algorithm(
            self.token,
            {"max_evaluations": 15},
            random_seed=4,
        )(problem)
        self.assertGreaterEqual(len(problem.calls), 1)
        # Best y_p is sphere on indices; unpenalized y would be 2x that.
        best_x = problem.state.current_best.x
        self.assertEqual(
            problem.state.current_best.y,
            float(sum(v * v for v in best_x)),
        )

    def test_nonmonotone_memory_reset_stays_1d(self) -> None:
        """
        Regression: upstream reset used ``W`` shape ``(1, 4)``, so ``W[0]``
        became a length-4 array and linesearch raised ambiguous-truth
        ``ValueError`` after the first penalty-``eps`` tightening.
        """
        import numpy as np
        from pathlib import Path

        from core.ioh_core.algorithms.dfl._vendor import DFLINT as dflint

        source = Path(dflint.__file__).read_text(encoding="utf-8")
        self.assertNotIn("np.nan*np.ones((1,4))", source)
        self.assertIn("np.nan*np.ones(M)", source)

        # Document the failure mode of the old shape.
        bad_w = np.nan * np.ones((1, 4))
        bad_w[0] = 1.5
        f_ref = bad_w[0]
        with self.assertRaises(ValueError) as ctx:
            _ = bool((2.0 < f_ref) and True)
        self.assertIn("ambiguous", str(ctx.exception))

        good_w = np.nan * np.ones(4)
        good_w[0] = 1.5
        f_ref = float(max(good_w[~np.isnan(good_w)]))
        self.assertTrue((0.0 < f_ref) and (f_ref < 2.0))


_P2_LOWER = [0, 0, 0, 0, 3, 0, 0, 0, 0]
_P2_UPPER = [10, 8, 5, 5, 14, 12, 3, 5, 3]


class TestDflintBboaWarmStart(unittest.TestCase):
    """DFLINT starts at rank 1 of a ranked InitialDesign."""

    token = "DFLINT_BBOA_WARM_START"

    def test_registry_metadata(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec(self.token)
        cold = get_algorithm_spec("DFLINT_BBOA")
        self.assertEqual(spec.library, "dfl")
        self.assertEqual(spec.seed_policy, SeedPolicy.FORBIDDEN)
        self.assertEqual(spec.candidate_mode, cold.candidate_mode)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(spec.budget_semantics, cold.budget_semantics)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, cold.dependency_extra)
        self.assertEqual(spec.dependency_extra, "algorithms-dfl")
        self.assertFalse(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations", "initial_design_path"],
        )
        self.assertFalse(spec.parameters[1].has_default)
        normalized = normalize_algorithm_parameters(
            self.token,
            {"max_evaluations": 10},
        )
        self.assertEqual(normalized, {"max_evaluations": 10})

    def test_first_evaluation_is_rank1(self) -> None:
        import json
        from tempfile import TemporaryDirectory

        from core.ioh_core.algorithm_factory import create_algorithm

        vector = [1, 2, 0, 0, 3, 0, 0, 0, 1]
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
            problem = _ConstrainedFakeIOHProblem(_P2_LOWER, _P2_UPPER)
            create_algorithm(
                self.token,
                {"max_evaluations": 8, "initial_design_path": str(path)},
                random_seed=None,
            )(problem)
        self.assertEqual(problem.calls[0], vector)
        self.assertGreaterEqual(len(problem.calls), 1)
        self.assertLessEqual(len(problem.calls), 8)

    def test_missing_file_does_not_evaluate(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem(_P2_LOWER, _P2_UPPER)
        algo = create_algorithm(
            self.token,
            {
                "max_evaluations": 5,
                "initial_design_path": "missing-initial-design.json",
            },
            random_seed=None,
        )
        with self.assertRaisesRegex(ValueError, "has no InitialDesign"):
            algo(problem)
        self.assertEqual(problem.calls, [])

    def test_short_budget_does_not_evaluate(self) -> None:
        import json
        from tempfile import TemporaryDirectory

        from core.ioh_core.algorithms.dfl.algorithms import DflintBboa

        vector = [1, 0, 0, 0, 3, 0, 0, 0, 0]
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
            problem = _ConstrainedFakeIOHProblem(_P2_LOWER, _P2_UPPER)
            algo = DflintBboa(
                max_evaluations=0,
                seed=None,
                initial_design_path=str(path),
                name=self.token,
            )
            with self.assertRaisesRegex(
                ValueError,
                "needs 1 initial evaluations but max_evaluations is 0",
            ):
                algo(problem)
        self.assertEqual(problem.calls, [])

    def test_rejects_wrong_dimension(self) -> None:
        import json
        from tempfile import TemporaryDirectory

        from core.ioh_core.algorithm_factory import create_algorithm
        from core.ioh_core.algorithms.common import CandidateError

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "problem_id": "L8100_B2",
                        "n": 1,
                        "vectors": [[1, 0, 0, 0, 3, 0, 0, 0, 0]],
                    }
                ),
                encoding="utf-8",
            )
            problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
            algo = create_algorithm(
                self.token,
                {"max_evaluations": 5, "initial_design_path": str(path)},
                random_seed=None,
            )
            with self.assertRaises(CandidateError):
                algo(problem)
        self.assertEqual(problem.calls, [])

    def test_rejects_decode_without_problem_id(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem(_P2_LOWER, _P2_UPPER)
        problem.hdb_eval_context.decode = lambda _x: {"geom_span_mm": 1234.0}
        algo = create_algorithm(
            self.token,
            {
                "max_evaluations": 5,
                "initial_design_path": "missing-initial-design.json",
            },
            random_seed=None,
        )
        with self.assertRaisesRegex(ValueError, "no problem_id"):
            algo(problem)
        self.assertEqual(problem.calls, [])

    def test_seed_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
        )

        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                self.token,
                {"max_evaluations": 5, "initial_design_path": "x.json"},
                random_seed=1,
            )

    def test_runs_are_identical(self) -> None:
        import json
        from tempfile import TemporaryDirectory

        from core.ioh_core.algorithm_factory import create_algorithm

        vector = [1, 0, 0, 0, 3, 0, 0, 0, 1]
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
            problem_a = _ConstrainedFakeIOHProblem(_P2_LOWER, _P2_UPPER)
            problem_b = _ConstrainedFakeIOHProblem(_P2_LOWER, _P2_UPPER)
            parameters = {
                "max_evaluations": 12,
                "initial_design_path": str(path),
            }
            create_algorithm(self.token, parameters, random_seed=None)(problem_a)
            create_algorithm(self.token, parameters, random_seed=None)(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)
        self.assertEqual(problem_a.calls[0], vector)


if __name__ == "__main__":
    unittest.main()
