"""
Phase ALG6.4 gate: RBFOpt package laziness, API contract, and solver preflight.
"""
from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version


_LOCKED_RBFOPT_SYMBOLS = (
    ("rbfopt", "RbfoptBlackBox"),
    ("rbfopt", "RbfoptSettings"),
    ("rbfopt", "RbfoptAlgorithm"),
)


class TestRbfoptPackageScaffold(unittest.TestCase):
    """Optional RBFOpt package remains lazy and API-compatible."""

    def test_rbfopt_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name == "rbfopt" or name.startswith("rbfopt."):
                sys.modules.pop(name, None)
            # Drop adapter implementation modules so a later import cannot keep a
            # stale RbfoptBlackBox base class after rbfopt is reloaded.
            if name.startswith("core.ioh_core.algorithms.rbfopt."):
                sys.modules.pop(name, None)

        mod = importlib.import_module("core.ioh_core.algorithms.rbfopt")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn("rbfopt", sys.modules)

    def test_rbfopt_api_contract_43(self) -> None:
        try:
            installed = version("rbfopt")
        except PackageNotFoundError as exc:
            self.fail(f"rbfopt extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("4.3.0"))
        self.assertLess(parsed, Version("4.4"))

        for module_name, attr_name in _LOCKED_RBFOPT_SYMBOLS:
            module = importlib.import_module(module_name)
            self.assertTrue(
                hasattr(module, attr_name),
                msg=f"missing locked symbol {module_name}:{attr_name}",
            )
            self.assertTrue(callable(getattr(module, attr_name)))


class TestResolveSolverExecutable(unittest.TestCase):
    """External Bonmin/Ipopt path preflight."""

    def test_resolve_solver_accepts_bare_name(self) -> None:
        from core.ioh_core.algorithms.rbfopt.factory import resolve_solver_executable

        with TemporaryDirectory() as tmp:
            fake = Path(tmp) / "bonmin"
            if os.name == "nt":
                fake = fake.with_suffix(".exe")
            fake.write_text("", encoding="utf-8")
            with patch(
                "core.ioh_core.algorithms.rbfopt.factory.shutil.which",
                return_value=str(fake),
            ) as mocked:
                resolved = resolve_solver_executable(
                    "bonmin",
                    parameter_name="minlp_solver_path",
                )
            mocked.assert_called_once_with("bonmin")
            self.assertEqual(resolved, str(fake.resolve()))

    def test_resolve_solver_accepts_path_with_spaces(self) -> None:
        from core.ioh_core.algorithms.rbfopt.factory import resolve_solver_executable

        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "solver tools"
            folder.mkdir()
            if os.name == "nt":
                executable = folder / "ipopt.bat"
                executable.write_text("@echo off\r\n", encoding="utf-8")
            else:
                executable = folder / "ipopt"
                executable.write_text("#!/bin/sh\n", encoding="utf-8")
                executable.chmod(0o755)

            resolved = resolve_solver_executable(
                str(executable),
                parameter_name="nlp_solver_path",
            )
            self.assertEqual(resolved, str(executable.resolve()))
            self.assertIn(" ", resolved)

    def test_resolve_solver_rejects_missing_executable(self) -> None:
        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.rbfopt.factory import resolve_solver_executable

        with self.assertRaises(AlgorithmDependencyError) as ctx:
            resolve_solver_executable(
                "definitely-not-a-real-solver-xyz",
                parameter_name="minlp_solver_path",
            )
        self.assertEqual(
            str(ctx.exception),
            "RBFOPT requires executable for 'minlp_solver_path', "
            "got 'definitely-not-a-real-solver-xyz'",
        )

        with self.assertRaises(AlgorithmDependencyError) as ctx_blank:
            resolve_solver_executable("   ", parameter_name="nlp_solver_path")
        self.assertIn("nlp_solver_path", str(ctx_blank.exception))

        with self.assertRaises(AlgorithmDependencyError) as ctx_type:
            resolve_solver_executable(123, parameter_name="minlp_solver_path")  # type: ignore[arg-type]
        self.assertIn("minlp_solver_path", str(ctx_type.exception))


class TestIOHRbfoptBlackBox(unittest.TestCase):
    """Native-integer IOH black-box adapter for RBFOpt."""

    def test_rbfopt_black_box_declares_all_variables_integer(self) -> None:
        import numpy as np

        from core.ioh_core.algorithms.rbfopt.problem import IOHRbfoptBlackBox
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 2, 5], [4, 6, 9])
        box = IOHRbfoptBlackBox(problem, max_evaluations=5)
        self.assertEqual(box.get_dimension(), 3)
        types = box.get_var_type()
        self.assertEqual(list(types), ["I", "I", "I"])
        self.assertFalse(np.any(types == "C"))
        self.assertFalse(np.any(types == "R"))
        np.testing.assert_array_equal(box.get_var_lower(), np.asarray([0, 2, 5], dtype=float))
        np.testing.assert_array_equal(box.get_var_upper(), np.asarray([4, 6, 9], dtype=float))

    def test_rbfopt_black_box_returns_defensive_bound_copies(self) -> None:
        import numpy as np

        from core.ioh_core.algorithms.rbfopt.problem import IOHRbfoptBlackBox
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        box = IOHRbfoptBlackBox(FakeIOHProblem([0, 1], [3, 4]), max_evaluations=2)
        lower = box.get_var_lower()
        upper = box.get_var_upper()
        types = box.get_var_type()
        lower[0] = 99.0
        upper[1] = -1.0
        types[0] = "R"
        np.testing.assert_array_equal(box.get_var_lower(), np.asarray([0, 1], dtype=float))
        np.testing.assert_array_equal(box.get_var_upper(), np.asarray([3, 4], dtype=float))
        self.assertEqual(list(box.get_var_type()), ["I", "I"])

    def test_rbfopt_black_box_rejects_fractional_candidates(self) -> None:
        import numpy as np

        from core.ioh_core.algorithms.common import CandidateError
        from core.ioh_core.algorithms.rbfopt.problem import IOHRbfoptBlackBox
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])
        box = IOHRbfoptBlackBox(problem, max_evaluations=3)
        with self.assertRaises(CandidateError):
            box.evaluate(np.asarray([0.5, 1.0]))
        self.assertEqual(problem.calls, [])
        self.assertEqual(problem.state.evaluations, 0)

        # Integral floats are accepted and cast.
        value = box.evaluate(np.asarray([1.0, 2.0]))
        self.assertEqual(value, 5.0)
        self.assertEqual(problem.calls, [[1, 2]])

    def test_rbfopt_black_box_enforces_hard_budget(self) -> None:
        import numpy as np

        from core.ioh_core.algorithms.rbfopt.problem import (
            RBFOPT_POST_STOP_SENTINEL,
            IOHRbfoptBlackBox,
        )
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0], [3])
        box = IOHRbfoptBlackBox(problem, max_evaluations=2)
        self.assertEqual(box.evaluate(np.asarray([1.0])), 1.0)
        self.assertEqual(box.evaluate(np.asarray([2.0])), 4.0)
        self.assertEqual(box.evaluate(np.asarray([0.0])), RBFOPT_POST_STOP_SENTINEL)
        self.assertEqual(problem.state.evaluations, 2)
        self.assertEqual(len(problem.calls), 2)

    def test_rbfopt_black_box_disables_noisy_evaluation(self) -> None:
        import numpy as np

        from core.ioh_core.algorithms.rbfopt.problem import IOHRbfoptBlackBox
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0], [2])
        box = IOHRbfoptBlackBox(problem, max_evaluations=3)
        self.assertFalse(box.has_evaluate_noisy())
        with self.assertRaises(NotImplementedError):
            box.evaluate_noisy(np.asarray([1.0]))
        self.assertEqual(problem.calls, [])
        self.assertEqual(problem.state.evaluations, 0)


class TestRBFOptSearch(unittest.TestCase):
    """Registered RBFOPT adapter behaviour."""

    def test_rbfopt_registry_spec_uses_actual_setting_names(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            ParameterKind,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec("RBFOPT")
        self.assertEqual(spec.library, "rbfopt")
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.rbfopt.factory:build_rbfopt",
        )
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-rbfopt")
        self.assertFalse(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            [
                "max_evaluations",
                "ga_base_population_size",
                "minlp_solver_path",
                "nlp_solver_path",
            ],
        )
        by_name = {parameter.name: parameter for parameter in spec.parameters}
        self.assertEqual(by_name["minlp_solver_path"].kind, ParameterKind.EXECUTABLE_PATH)
        self.assertEqual(by_name["nlp_solver_path"].kind, ParameterKind.EXECUTABLE_PATH)
        normalized = normalize_algorithm_parameters(
            "RBFOPT",
            {
                "max_evaluations": 10,
                "minlp_solver_path": "bonmin",
                "nlp_solver_path": "ipopt",
            },
        )
        self.assertEqual(
            normalized,
            {
                "max_evaluations": 10,
                "ga_base_population_size": 400,
                "minlp_solver_path": "bonmin",
                "nlp_solver_path": "ipopt",
            },
        )

    def test_rbfopt_factory_resolves_both_solvers(self) -> None:
        from core.ioh_core.algorithms.rbfopt.factory import build_rbfopt

        with TemporaryDirectory() as tmp:
            minlp = Path(tmp) / "bonmin"
            nlp = Path(tmp) / "ipopt"
            if os.name == "nt":
                minlp = minlp.with_suffix(".exe")
                nlp = nlp.with_suffix(".exe")
            minlp.write_text("", encoding="utf-8")
            nlp.write_text("", encoding="utf-8")

            def _which(value: str) -> str | None:
                if value in {"bonmin", str(minlp)}:
                    return str(minlp)
                if value in {"ipopt", str(nlp)}:
                    return str(nlp)
                return None

            with patch(
                "core.ioh_core.algorithms.rbfopt.factory.shutil.which",
                side_effect=_which,
            ):
                algo = build_rbfopt(
                    {
                        "max_evaluations": 5,
                        "ga_base_population_size": 20,
                        "minlp_solver_path": "bonmin",
                        "nlp_solver_path": "ipopt",
                    },
                    random_seed=7,
                )

        self.assertEqual(algo.minlp_solver_path, str(minlp.resolve()))
        self.assertEqual(algo.nlp_solver_path, str(nlp.resolve()))
        self.assertEqual(algo.max_evaluations, 5)
        self.assertEqual(algo.ga_base_population_size, 20)
        self.assertEqual(algo.seed, 7)

    def test_rbfopt_search_constructs_locked_settings(self) -> None:
        from core.ioh_core.algorithms.rbfopt.algorithms import RBFOptSearch
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        captured: dict = {}

        class _FakeAlgorithm:
            def __init__(self, settings, black_box):
                captured["settings"] = settings
                captured["black_box"] = black_box

            def optimize(self):
                return (0.0, [0.0, 0.0])

        with patch("rbfopt.RbfoptSettings", side_effect=lambda **kwargs: kwargs) as settings_ctor, patch(
            "rbfopt.RbfoptAlgorithm",
            side_effect=_FakeAlgorithm,
        ):
            algo = RBFOptSearch(
                max_evaluations=8,
                ga_base_population_size=20,
                minlp_solver_path="/tmp/bonmin",
                nlp_solver_path="/tmp/ipopt",
                seed=13,
            )
            algo(FakeIOHProblem([0, 0], [2, 2]))

        settings_ctor.assert_called_once_with(
            max_evaluations=8,
            max_iterations=80,
            max_cycles=80,
            max_noisy_evaluations=0,
            ga_base_population_size=20,
            minlp_solver_path="/tmp/bonmin",
            nlp_solver_path="/tmp/ipopt",
            rand_seed=13,
            num_cpus=1,
        )
        self.assertEqual(captured["settings"]["max_evaluations"], 8)
        self.assertEqual(captured["black_box"].get_dimension(), 2)

    def test_rbfopt_target_objval_semantics_absolute_with_eps_opt_zero(self) -> None:
        from rbfopt import RbfoptSettings
        from rbfopt.rbfopt_utils import compute_gap

        settings = RbfoptSettings(target_objval=10.0, eps_opt=0.0)
        # Public gap formula: (fmin - target) / max(|target|, eps_zero).
        self.assertGreater(compute_gap(settings, 10.1), 0.0)
        self.assertEqual(compute_gap(settings, 10.0), 0.0)
        self.assertLess(compute_gap(settings, 9.9), 0.0)

    def test_rbfopt_sets_target_objval_for_all_reference(self) -> None:
        from core.ioh_core.algorithms.common import RunController
        from core.ioh_core.algorithms.rbfopt.algorithms import RBFOptSearch
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        captured: dict = {}

        class _FakeAlgorithm:
            def __init__(self, settings, black_box):
                captured["settings"] = settings

            def optimize(self):
                return (0.0, [0.0, 0.0])

        problem = FakeIOHProblem([0, 0], [2, 2])
        controller = RunController(
            problem,
            8,
            mode="fixed_target",
            target_threshold=12.5,
            target_reference="all",
        )
        with patch(
            "rbfopt.RbfoptSettings", side_effect=lambda **kwargs: kwargs
        ) as settings_ctor, patch(
            "rbfopt.RbfoptAlgorithm",
            side_effect=_FakeAlgorithm,
        ):
            RBFOptSearch(
                max_evaluations=8,
                ga_base_population_size=20,
                minlp_solver_path="/tmp/bonmin",
                nlp_solver_path="/tmp/ipopt",
                seed=13,
            )(problem, run_controller=controller)

        settings_ctor.assert_called_once_with(
            max_evaluations=8,
            max_iterations=80,
            max_cycles=80,
            max_noisy_evaluations=0,
            ga_base_population_size=20,
            minlp_solver_path="/tmp/bonmin",
            nlp_solver_path="/tmp/ipopt",
            rand_seed=13,
            num_cpus=1,
            target_objval=12.5,
            eps_opt=0.0,
        )
        self.assertEqual(captured["settings"]["target_objval"], 12.5)
        self.assertEqual(captured["settings"]["eps_opt"], 0.0)

    def test_warm_start_keeps_evaluation_headroom(self) -> None:
        from core.ioh_core.algorithms.rbfopt.algorithms import RBFOptSearchWarmStart
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        captured: dict = {}

        class _Settings:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

            def set_auto_parameters(self, *_args: object, **_kwargs: object) -> object:
                return self

        class _FakeAlgorithm:
            def __init__(
                self,
                settings: object,
                black_box: object,
                **kwargs: object,
            ) -> None:
                captured["settings"] = settings
                captured["init"] = kwargs

            def optimize(self) -> tuple[float, list[float]]:
                return (0.0, [0.0, 0.0])

        with patch("rbfopt.RbfoptSettings", _Settings), patch(
            "rbfopt.RbfoptAlgorithm",
            _FakeAlgorithm,
        ), patch(
            "rbfopt.rbfopt_utils.get_num_init_samples",
            return_value=1,
        ), patch(
            "core.ioh_core.algorithms.warm_start.load_warm_start_prefix",
            return_value=[[0, 0]],
        ):
            RBFOptSearchWarmStart(
                max_evaluations=8,
                ga_base_population_size=20,
                minlp_solver_path="/tmp/bonmin",
                nlp_solver_path="/tmp/ipopt",
                seed=13,
                initial_design_path="unused.json",
            )(FakeIOHProblem([0, 0], [2, 2]))

        settings = captured["settings"]
        self.assertEqual(settings.max_evaluations, 8)
        self.assertEqual(settings.max_iterations, 80)
        self.assertEqual(settings.max_cycles, 80)
        self.assertEqual(captured["init"]["do_init_strategy"], False)

    def test_rbfopt_black_box_sentinel_after_target_hit(self) -> None:
        import numpy as np

        from core.ioh_core.algorithms.common import RunController
        from core.ioh_core.algorithms.rbfopt.problem import (
            RBFOPT_POST_STOP_SENTINEL,
            IOHRbfoptBlackBox,
        )
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0], [0])
        controller = RunController(
            problem,
            5,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        box = IOHRbfoptBlackBox(problem, 5, run_controller=controller)
        self.assertEqual(box.evaluate(np.asarray([0.0])), 0.0)
        self.assertTrue(controller.target_hit)
        self.assertEqual(box.evaluate(np.asarray([0.0])), RBFOPT_POST_STOP_SENTINEL)
        self.assertEqual(len(problem.calls), 1)

    def test_rbfopt_search_ignores_optimize_tuple(self) -> None:
        from core.ioh_core.algorithms.rbfopt.algorithms import RBFOptSearch
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])

        class _FakeAlgorithm:
            def __init__(self, settings, black_box):
                self.black_box = black_box

            def optimize(self):
                # Pretend RBFOpt found a different best than IOH observed.
                return (999.0, [9.0, 9.0])

        with patch("rbfopt.RbfoptSettings", return_value=object()), patch(
            "rbfopt.RbfoptAlgorithm",
            side_effect=_FakeAlgorithm,
        ):
            RBFOptSearch(
                max_evaluations=4,
                ga_base_population_size=20,
                minlp_solver_path="/tmp/bonmin",
                nlp_solver_path="/tmp/ipopt",
                seed=3,
            )(problem)

        self.assertEqual(problem.calls, [])
        self.assertEqual(problem.state.evaluations, 0)


class _SeededFakeRbfoptAlgorithm:
    """
    Deterministic stand-in for ``RbfoptAlgorithm`` used by conformance tests.

    Evaluates the black box with a seed-derived integer sequence so shared
    conformance checks can run without Bonmin/Ipopt.
    """

    def __init__(self, settings, black_box) -> None:
        self.settings = settings
        self.black_box = black_box

    def optimize(self):
        import numpy as np

        rng = np.random.default_rng(int(self.settings.rand_seed))
        lower = self.black_box.get_var_lower()
        upper = self.black_box.get_var_upper()
        dimension = int(self.black_box.get_dimension())
        for _ in range(int(self.settings.max_evaluations)):
            candidate = np.asarray(
                [
                    float(rng.integers(int(lower[index]), int(upper[index]) + 1))
                    for index in range(dimension)
                ],
                dtype=float,
            )
            self.black_box.evaluate(candidate)
        return (0.0, [0.0] * dimension)


class TestRBFOptConformance(unittest.TestCase):
    """Shared adapter contract with mocked RBFOpt solver backend."""

    def test_rbfopt_conformance(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import assert_algorithm_conformance

        with TemporaryDirectory() as tmp:
            minlp = Path(tmp) / "bonmin"
            nlp = Path(tmp) / "ipopt"
            if os.name == "nt":
                minlp = minlp.with_suffix(".bat")
                nlp = nlp.with_suffix(".bat")
                minlp.write_text("@echo off\r\n", encoding="utf-8")
                nlp.write_text("@echo off\r\n", encoding="utf-8")
            else:
                minlp.write_text("#!/bin/sh\n", encoding="utf-8")
                nlp.write_text("#!/bin/sh\n", encoding="utf-8")
                minlp.chmod(0o755)
                nlp.chmod(0o755)

            parameters = {
                "max_evaluations": 8,
                "ga_base_population_size": 20,
                "minlp_solver_path": str(minlp),
                "nlp_solver_path": str(nlp),
            }

            def build(seed: int | None):
                return create_algorithm(
                    "RBFOPT",
                    parameters,
                    random_seed=seed,
                )

            with patch("rbfopt.RbfoptAlgorithm", _SeededFakeRbfoptAlgorithm):
                assert_algorithm_conformance(
                    self,
                    algorithm_name="RBFOPT",
                    build_algorithm=build,
                    seed=11,
                    second_seed=29,
                )


if __name__ == "__main__":
    unittest.main()
