"""
Phase NM2 gate: PYNOMAD_VNS_MADS_PB (MADS-PB plus VNS search).
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version

from _mains.Testing.test_cursor_algorithm_pynomad_mads_pb import (
    _ConstrainedFakeIOHProblem,
)


class TestPyNomadVnsMadsSearchApi(unittest.TestCase):
    """Live PyNomad accepts VNS Mads search param lines."""

    def test_pynomad_optimize_accepts_vns_mads_params(self) -> None:
        try:
            installed = version("PyNomadBBO")
        except PackageNotFoundError as exc:
            self.fail(f"algorithms-nomad extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("4.5.1"))
        self.assertLess(parsed, Version("4.6"))

        import PyNomad

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
                "VNS_MADS_SEARCH yes",
                "VNS_MADS_SEARCH_TRIGGER 0.75",
            ],
        )
        self.assertIsInstance(result, dict)
        self.assertIn("x_best", result)
        self.assertGreaterEqual(len(calls), 1)


class TestPyNomadVnsMadsPbAdapter(unittest.TestCase):
    """Registered PYNOMAD_VNS_MADS_PB adapter behaviour."""

    token = "PYNOMAD_VNS_MADS_PB"

    def test_registry_metadata_and_trigger_default(self) -> None:
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
            ["max_evaluations", "vns_mads_search_trigger"],
        )
        normalized = normalize_algorithm_parameters(
            self.token,
            {"max_evaluations": 10},
        )
        self.assertEqual(normalized["max_evaluations"], 10)
        self.assertEqual(normalized["vns_mads_search_trigger"], 0.75)

    def test_trigger_accepts_integer_one(self) -> None:
        from core.ioh_core.algorithms.registry import (
            normalize_algorithm_parameters,
        )

        normalized = normalize_algorithm_parameters(
            self.token,
            {"max_evaluations": 10, "vns_mads_search_trigger": 1},
        )
        self.assertEqual(normalized["vns_mads_search_trigger"], 1.0)

    def test_trigger_rejects_zero(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
        )

        with self.assertRaises(AlgorithmFactoryError) as ctx:
            create_algorithm(
                self.token,
                {"max_evaluations": 5, "vns_mads_search_trigger": 0},
                random_seed=1,
            )
        self.assertIn("vns_mads_search_trigger", str(ctx.exception))
        self.assertIn("> 0", str(ctx.exception))

        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                self.token,
                {"max_evaluations": 5, "vns_mads_search_trigger": 0.0},
                random_seed=1,
            )

    def test_trigger_rejects_above_one(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
        )

        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                self.token,
                {"max_evaluations": 5, "vns_mads_search_trigger": 1.1},
                random_seed=1,
            )

    def test_optimize_params_enable_vns_search(self) -> None:
        from core.ioh_core.algorithms.nomad.algorithms import PyNomadVnsMadsPb

        fake_py_nomad = MagicMock()
        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        with patch.dict("sys.modules", {"PyNomad": fake_py_nomad}):
            PyNomadVnsMadsPb(
                max_evaluations=5,
                vns_mads_search_trigger=0.75,
                seed=3,
            )(problem)

        params = fake_py_nomad.optimize.call_args.args[4]
        self.assertIn("VNS_MADS_SEARCH yes", params)
        self.assertIn("VNS_MADS_SEARCH_TRIGGER 0.75", params)
        self.assertIn("MAX_BB_EVAL 5", params)
        self.assertIn("MAX_EVAL 5", params)

    def test_explicit_trigger_is_forwarded(self) -> None:
        from core.ioh_core.algorithms.nomad.algorithms import PyNomadVnsMadsPb

        fake_py_nomad = MagicMock()
        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        with patch.dict("sys.modules", {"PyNomad": fake_py_nomad}):
            PyNomadVnsMadsPb(
                max_evaluations=5,
                vns_mads_search_trigger=0.5,
                seed=3,
            )(problem)

        params = fake_py_nomad.optimize.call_args.args[4]
        self.assertIn("VNS_MADS_SEARCH_TRIGGER 0.5", params)

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

    def test_optimize_stops_when_controller_refuses(self) -> None:
        """VNS uses the same runner; refuse-eval must abort optimize."""
        from unittest.mock import MagicMock, patch

        from core.ioh_core.algorithms.nomad.algorithms import PyNomadVnsMadsPb

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
            PyNomadVnsMadsPb(
                max_evaluations=3,
                vns_mads_search_trigger=0.75,
                seed=3,
            )(problem)
        self.assertEqual(len(problem.calls), 3)
        self.assertEqual(problem.state.evaluations, 3)

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
        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.nomad.factory import (
            build_pynomad_vns_mads_pb,
        )

        with patch(
            "core.ioh_core.algorithms.errors.importlib.import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'PyNomad'", name="PyNomad"
            ),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                build_pynomad_vns_mads_pb(
                    {
                        "max_evaluations": 5,
                        "vns_mads_search_trigger": 0.75,
                    },
                    random_seed=1,
                )
        message = str(ctx.exception)
        self.assertIn("PYNOMAD_VNS_MADS_PB", message)
        self.assertIn("algorithms-nomad", message)


class TestPyNomadVnsMadsPbWarmStart(unittest.TestCase):
    """PYNOMAD_VNS_MADS_PB_WARM_START uses rank 1 of the InitialDesign as x0."""

    def test_optimize_x0_is_rank1(self) -> None:
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from core.ioh_core.algorithm_factory import create_algorithm

        vector = [1, 0, 0, 0, 3, 0, 0, 0, 1]
        fake_py_nomad = MagicMock()
        problem = _ConstrainedFakeIOHProblem(
            [0, 0, 0, 0, 3, 0, 0, 0, 0],
            [10, 8, 5, 5, 14, 12, 3, 5, 3],
        )
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
                    "PYNOMAD_VNS_MADS_PB_WARM_START",
                    {"max_evaluations": 5, "initial_design_path": str(path)},
                    random_seed=3,
                )(problem)
        x0 = fake_py_nomad.optimize.call_args.args[1]
        params = fake_py_nomad.optimize.call_args.args[4]
        self.assertEqual([int(v) for v in x0], vector)
        self.assertTrue(any("VNS_MADS_SEARCH" in line for line in params))


if __name__ == "__main__":
    unittest.main()
