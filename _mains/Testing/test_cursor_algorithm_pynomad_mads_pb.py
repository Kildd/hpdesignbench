"""
Phase NM1 gate: nomad package laziness, PyNomad API, and PYNOMAD_MADS_PB.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version

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


class TestNomadPackageScaffold(unittest.TestCase):
    """Optional nomad package remains lazy and PyNomad API is usable."""

    def test_nomad_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name == "PyNomad" or name.startswith("PyNomad."):
                sys.modules.pop(name, None)
            if name.startswith("core.ioh_core.algorithms.nomad"):
                sys.modules.pop(name, None)

        mod = importlib.import_module("core.ioh_core.algorithms.nomad")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn("PyNomad", sys.modules)

    def test_pynomad_api_contract_451(self) -> None:
        try:
            installed = version("PyNomadBBO")
        except PackageNotFoundError as exc:
            self.fail(f"algorithms-nomad extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("4.5.1"))
        self.assertLess(parsed, Version("4.6"))

        import PyNomad

        self.assertTrue(callable(getattr(PyNomad, "optimize", None)))

        calls: list[float] = []

        def bb(x):
            coords = [x.get_coord(i) for i in range(x.size())]
            f = (coords[0] - 2.0) ** 2
            g = coords[0] - 3.0
            calls.append(coords[0])
            x.setBBO(f"{f:.16g} {g:.16g}".encode("UTF-8"))
            return 1

        result = PyNomad.optimize(
            bb,
            [5.0],
            [0.0],
            [5.0],
            [
                "DIMENSION 1",
                "BB_INPUT_TYPE * I",
                "BB_OUTPUT_TYPE OBJ PB",
                "MAX_BB_EVAL 30",
                "MAX_EVAL 30",
                "SEED 7",
                "DISPLAY_DEGREE 0",
                "DISPLAY_ALL_EVAL false",
                "NB_THREADS_PARALLEL_EVAL 1",
            ],
        )
        self.assertIsInstance(result, dict)
        self.assertIn("x_best", result)
        self.assertEqual(float(result["x_best"][0]), 2.0)
        self.assertEqual(float(result["f_best"]), 0.0)
        self.assertGreaterEqual(len(calls), 1)


class TestPyNomadMadsPbAdapter(unittest.TestCase):
    """Registered PYNOMAD_MADS_PB adapter behaviour."""

    token = "PYNOMAD_MADS_PB"

    def test_registry_metadata(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec(self.token)
        self.assertEqual(spec.library, "nomad")
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-nomad")
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
                {"max_evaluations": 5, "direction_type": "ORTHO"},
                random_seed=1,
            )

    def test_missing_dependency_hint(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.nomad.factory import build_pynomad_mads_pb

        with patch(
            "core.ioh_core.algorithms.errors.importlib.import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'PyNomad'", name="PyNomad"
            ),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                build_pynomad_mads_pb({"max_evaluations": 5}, random_seed=1)
        message = str(ctx.exception)
        self.assertIn("PYNOMAD_MADS_PB", message)
        self.assertIn("algorithms-nomad", message)

    def test_nomad_signed_seed_masks_uint32_to_signed32(self) -> None:
        from core.ioh_core.algorithms.nomad.algorithms import nomad_signed_seed

        self.assertEqual(nomad_signed_seed(0), 0)
        self.assertEqual(nomad_signed_seed(2147483647), 2147483647)
        self.assertEqual(nomad_signed_seed(2147483648), 0)
        self.assertEqual(nomad_signed_seed(2301619403), 2301619403 & 0x7FFFFFFF)
        self.assertEqual(nomad_signed_seed(4061050216), 4061050216 & 0x7FFFFFFF)
        self.assertLessEqual(nomad_signed_seed(2**32 - 1), 2**31 - 1)

    def test_large_uint32_seed_masks_before_setseed(self) -> None:
        """Regression: planner seeds may exceed signed 32-bit C long."""
        from unittest.mock import MagicMock, patch

        from core.ioh_core.algorithms.nomad.algorithms import (
            PyNomadMadsPb,
            nomad_signed_seed,
        )

        large_seed = 2301619403  # > 2**31 - 1; previously OverflowError
        expected = nomad_signed_seed(large_seed)
        self.assertLessEqual(expected, 2**31 - 1)

        fake_py_nomad = MagicMock()
        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        with patch.dict("sys.modules", {"PyNomad": fake_py_nomad}):
            PyNomadMadsPb(max_evaluations=5, seed=large_seed)(problem)

        fake_py_nomad.setSeed.assert_called_once_with(expected)
        params = fake_py_nomad.optimize.call_args.args[4]
        self.assertIn(f"SEED {expected}", params)
        self.assertNotIn(f"SEED {large_seed}", params)

    def test_mads_pb_params_omit_vns_mads_keys(self) -> None:
        """MADS-PB must not enable VNS search (identity of this token)."""
        from unittest.mock import MagicMock, patch

        from core.ioh_core.algorithms.nomad.algorithms import PyNomadMadsPb

        fake_py_nomad = MagicMock()
        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        with patch.dict("sys.modules", {"PyNomad": fake_py_nomad}):
            PyNomadMadsPb(max_evaluations=5, seed=3)(problem)

        params = fake_py_nomad.optimize.call_args.args[4]
        vns_lines = [line for line in params if "VNS_MADS" in line]
        self.assertEqual(vns_lines, [])

    def test_sets_max_eval_to_unique_budget(self) -> None:
        """Granular NOMAD otherwise injects MAX_EVAL 100000 and spins after a target hit."""
        from unittest.mock import MagicMock, patch

        from core.ioh_core.algorithms.nomad.algorithms import PyNomadMadsPb

        fake_py_nomad = MagicMock()
        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        with patch.dict("sys.modules", {"PyNomad": fake_py_nomad}):
            PyNomadMadsPb(max_evaluations=15, seed=3)(problem)

        params = fake_py_nomad.optimize.call_args.args[4]
        self.assertIn("MAX_BB_EVAL 15", params)
        self.assertIn("MAX_EVAL 15", params)

    def test_live_optimize_returns_after_early_target_hit(self) -> None:
        """PyNomad.cb swallows Python exceptions; MAX_EVAL must bound the dummy loop."""
        import threading
        import time

        from core.ioh_core.algorithms.common import RunController
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        controller = RunController(
            problem,
            10000,
            mode="fixed_target",
            target_threshold=1e9,
            target_reference="all",
        )
        finished: list[bool] = []

        def _run() -> None:
            create_algorithm(
                self.token,
                {"max_evaluations": 10000},
                random_seed=8,
            )(problem, run_controller=controller)
            finished.append(True)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(8.0)
        self.assertFalse(thread.is_alive(), "optimize still running after target hit")
        self.assertEqual(finished, [True])
        self.assertTrue(controller.target_hit)
        self.assertGreaterEqual(len(problem.calls), 1)
        self.assertLess(len(problem.calls), 50)

    def test_optimize_stops_when_controller_refuses(self) -> None:
        """Refuse-eval must abort optimize, not return 0 and spin."""
        from unittest.mock import MagicMock, patch

        from core.ioh_core.algorithms.nomad.algorithms import PyNomadMadsPb

        class _EvalPoint:
            def size(self) -> int:
                return 2

            def get_coord(self, index: int) -> float:
                del index
                return 0.0

            def setBBO(self, *_args: object, **_kwargs: object) -> None:
                return None

        def _fake_optimize(blackbox, x0, lb, ub, params):
            del x0, lb, ub, params
            point = _EvalPoint()
            for _ in range(500):
                blackbox(point)

        fake_py_nomad = MagicMock()
        fake_py_nomad.optimize.side_effect = _fake_optimize
        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        with patch.dict("sys.modules", {"PyNomad": fake_py_nomad}):
            PyNomadMadsPb(max_evaluations=3, seed=3)(problem)
        self.assertEqual(len(problem.calls), 3)
        self.assertEqual(problem.state.evaluations, 3)


_P2_LOWER = [0, 0, 0, 0, 3, 0, 0, 0, 0]
_P2_UPPER = [10, 8, 5, 5, 14, 12, 3, 5, 3]


class TestPyNomadMadsPbWarmStart(unittest.TestCase):
    """PYNOMAD_MADS_PB_WARM_START uses rank 1 as x0 and keeps the seed."""

    token = "PYNOMAD_MADS_PB_WARM_START"

    def test_registry_keeps_required_seed(self) -> None:
        from core.ioh_core.algorithms.registry import SeedPolicy, get_algorithm_spec

        spec = get_algorithm_spec(self.token)
        cold = get_algorithm_spec("PYNOMAD_MADS_PB")
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.dependency_extra, cold.dependency_extra)
        self.assertEqual(spec.candidate_mode, cold.candidate_mode)
        self.assertEqual(spec.budget_semantics, cold.budget_semantics)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations", "initial_design_path"],
        )
        self.assertFalse(spec.parameters[1].has_default)

    def test_optimize_x0_is_rank1(self) -> None:
        import json
        from tempfile import TemporaryDirectory
        from unittest.mock import MagicMock, patch

        from core.ioh_core.algorithm_factory import create_algorithm

        vector = [1, 0, 0, 0, 3, 0, 0, 0, 1]
        fake_py_nomad = MagicMock()
        problem = _ConstrainedFakeIOHProblem(_P2_LOWER, _P2_UPPER)
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
            with patch.dict("sys.modules", {"PyNomad": fake_py_nomad}):
                create_algorithm(
                    self.token,
                    {"max_evaluations": 5, "initial_design_path": str(path)},
                    random_seed=3,
                )(problem)
                other = _ConstrainedFakeIOHProblem(_P2_LOWER, _P2_UPPER)
                create_algorithm(
                    self.token,
                    {"max_evaluations": 5, "initial_design_path": str(path)},
                    random_seed=9,
                )(other)
        x0 = [int(v) for v in fake_py_nomad.optimize.call_args_list[0].args[1]]
        x0_other = [int(v) for v in fake_py_nomad.optimize.call_args_list[1].args[1]]
        self.assertEqual(x0, vector)
        self.assertEqual(x0_other, vector)
        self.assertTrue(fake_py_nomad.setSeed.called)

    def test_missing_file_does_not_optimize(self) -> None:
        from unittest.mock import MagicMock, patch

        from core.ioh_core.algorithms.nomad.algorithms import PyNomadMadsPb

        fake_py_nomad = MagicMock()
        problem = _ConstrainedFakeIOHProblem(_P2_LOWER, _P2_UPPER)
        algo = PyNomadMadsPb(
            max_evaluations=0,
            seed=3,
            initial_design_path="missing-initial-design.json",
            name=self.token,
        )
        with patch.dict("sys.modules", {"PyNomad": fake_py_nomad}):
            with self.assertRaisesRegex(ValueError, "has no InitialDesign"):
                algo(problem)
        self.assertEqual(problem.calls, [])
        self.assertFalse(fake_py_nomad.optimize.called)


if __name__ == "__main__":
    unittest.main()
