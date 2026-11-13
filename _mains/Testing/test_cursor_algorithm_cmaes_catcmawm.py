"""
Phase CM1 gate: cmaes package laziness, API contract, and CMAES_CATCMAWM.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version


class TestCmaesPackageScaffold(unittest.TestCase):
    """Optional cmaes package remains lazy and API-compatible."""

    def test_cmaes_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name == "cmaes" or name.startswith("cmaes."):
                sys.modules.pop(name, None)
            if name.startswith("core.ioh_core.algorithms.cmaes."):
                sys.modules.pop(name, None)
        sys.modules.pop("core.ioh_core.algorithms.cmaes", None)

        mod = importlib.import_module("core.ioh_core.algorithms.cmaes")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn("cmaes", sys.modules)

    def test_cmaes_catcmawm_api_contract_013(self) -> None:
        try:
            installed = version("cmaes")
        except PackageNotFoundError as exc:
            self.fail(f"cmaes extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("0.13.0"))
        self.assertLess(parsed, Version("0.14"))

        from cmaes import CatCMAwM

        optimizer = CatCMAwM(
            z_space=[[0, 1, 2], [0, 1]],
            population_size=4,
            seed=0,
        )
        sol = optimizer.ask()
        self.assertIsNone(sol.x)
        self.assertIsNone(sol.c)
        self.assertEqual(len(sol.z), 2)
        self.assertTrue(all(int(v) == v for v in sol.z))


class TestCatCmaWMAlgorithm(unittest.TestCase):
    """Registered CMAES_CATCMAWM adapter behaviour."""

    def test_cmaes_catcmawm_registry_metadata(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec("CMAES_CATCMAWM")
        self.assertEqual(spec.library, "cmaes")
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.cmaes.factory:build_cmaes_catcmawm",
        )
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-cmaes")
        self.assertTrue(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations", "population_size"],
        )
        normalized = normalize_algorithm_parameters(
            "CMAES_CATCMAWM",
            {"max_evaluations": 10},
        )
        self.assertEqual(
            normalized,
            {
                "max_evaluations": 10,
                "population_size": 50,
            },
        )

    def test_cmaes_catcmawm_respects_hard_budget(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            "CMAES_CATCMAWM",
            {"max_evaluations": 10, "population_size": 4},
            random_seed=8,
        )(problem)
        self.assertEqual(len(problem.calls), 10)
        self.assertEqual(problem.state.evaluations, 10)

    def test_cmaes_catcmawm_deterministic_with_seed(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem_a = FakeIOHProblem([0, 1], [3, 4])
        problem_b = FakeIOHProblem([0, 1], [3, 4])
        create_algorithm(
            "CMAES_CATCMAWM",
            {"max_evaluations": 12, "population_size": 4},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            "CMAES_CATCMAWM",
            {"max_evaluations": 12, "population_size": 4},
            random_seed=17,
        )(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)
        self.assertGreaterEqual(len(problem_a.calls), 1)
        self.assertTrue(
            all(isinstance(v, int) for c in problem_a.calls for v in c)
        )

    def test_cmaes_catcmawm_different_seed_different_sequence(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem_a = FakeIOHProblem([0, 1], [5, 6])
        problem_b = FakeIOHProblem([0, 1], [5, 6])
        create_algorithm(
            "CMAES_CATCMAWM",
            {"max_evaluations": 20, "population_size": 4},
            random_seed=3,
        )(problem_a)
        create_algorithm(
            "CMAES_CATCMAWM",
            {"max_evaluations": 20, "population_size": 4},
            random_seed=99,
        )(problem_b)
        self.assertNotEqual(problem_a.calls, problem_b.calls)

    def test_cmaes_catcmawm_rejects_singleton_z_space_dimension(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from core.ioh_core.algorithms.cmaes.algorithms import build_integer_z_space
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        with self.assertRaises(ValueError):
            build_integer_z_space([0, 1], [2, 1])

        algo = create_algorithm(
            "CMAES_CATCMAWM",
            {"max_evaluations": 8, "population_size": 4},
            random_seed=1,
        )
        with self.assertRaises(ValueError):
            algo(FakeIOHProblem([0, 2], [3, 2]))

    def test_cmaes_catcmawm_does_not_re_evaluate_best(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])
        algo = create_algorithm(
            "CMAES_CATCMAWM",
            {"max_evaluations": 10, "population_size": 4},
            random_seed=5,
        )
        algo(problem)
        call_count = len(problem.calls)
        self.assertEqual(problem.state.evaluations, call_count)
        self.assertLessEqual(call_count, 10)
        self.assertIn(problem.state.current_best.x, problem.calls)
        self.assertEqual(len(problem.calls), call_count)
        self.assertEqual(problem.state.evaluations, call_count)

    def test_cmaes_catcmawm_missing_seed_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
        )
        from core.ioh_core.algorithms.cmaes.factory import build_cmaes_catcmawm

        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "CMAES_CATCMAWM",
                {"max_evaluations": 5, "population_size": 4},
                random_seed=None,
            )
        with self.assertRaises(ValueError):
            build_cmaes_catcmawm(
                {"max_evaluations": 5, "population_size": 4},
                random_seed=None,
            )

    def test_cmaes_catcmawm_unknown_parameter_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
            validate_algorithm_parameters,
        )

        with self.assertRaises(AlgorithmFactoryError):
            validate_algorithm_parameters(
                "CMAES_CATCMAWM",
                {
                    "max_evaluations": 5,
                    "population_size": 4,
                    "sigma": 1.0,
                },
                random_seed=1,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "CMAES_CATCMAWM",
                {
                    "max_evaluations": 5,
                    "population_size": 4,
                    "sigma": 1.0,
                },
                random_seed=1,
            )

    def test_cmaes_catcmawm_missing_dependency_uses_install_hint(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.cmaes.factory import build_cmaes_catcmawm

        with patch(
            "core.ioh_core.algorithms.cmaes.factory.import_optional_dependency",
            side_effect=AlgorithmDependencyError(
                "CMAES_CATCMAWM requires optional dependency 'cmaes'; "
                "install with:\n"
                'pip install -e ".[algorithms-cmaes]"'
            ),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                build_cmaes_catcmawm(
                    {"max_evaluations": 5, "population_size": 4},
                    random_seed=0,
                )
        self.assertIn(
            'pip install -e ".[algorithms-cmaes]"',
            str(ctx.exception),
        )


class TestCatCmaWMWarmStart(unittest.TestCase):
    """Rank 1 is the mean. The cold box-centre covariance stays positive."""

    def test_bound_mean_keeps_a_positive_diagonal(self) -> None:
        import numpy as np
        from cmaes import CatCMAwM

        from core.ioh_core.algorithms.cmaes.algorithms import (
            build_integer_z_space,
            integer_box_centre_covariance,
        )

        lower = [0, 0, 0, 0]
        upper = [20, 20, 20, 20]
        mean = np.asarray([0, 20, 0, 20], dtype=float)
        cov = integer_box_centre_covariance(lower, upper)
        optimizer = CatCMAwM(
            z_space=build_integer_z_space(lower, upper),
            population_size=24,
            mean=mean,
            cov=cov,
            seed=1,
        )
        diagonal = np.diag(optimizer._C)
        self.assertTrue(np.all(diagonal > 0))
        asked = [optimizer.ask().z for _ in range(optimizer.population_size)]
        for index in range(4):
            values = {int(row[index]) for row in asked}
            self.assertGreater(len(values), 1)

    def test_one_population_does_not_require_the_mean(self) -> None:
        import json
        from tempfile import TemporaryDirectory

        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem
        from core.ioh_core.algorithms.cmaes.algorithms import CatCmaWMWarmStart

        vector = [1, 0, 0, 0, 0, 0, 0, 0, 4]
        lower = [0] * 9
        upper = [8] * 9
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
            CatCmaWMWarmStart(
                max_evaluations=4,
                population_size=4,
                seed=3,
                initial_design_path=str(path),
            )(problem)
            missing = FakeIOHProblem(lower, upper)
            with self.assertRaisesRegex(ValueError, "has no InitialDesign"):
                CatCmaWMWarmStart(
                    max_evaluations=4,
                    population_size=4,
                    seed=3,
                    initial_design_path="missing-initial-design.json",
                )(missing)
        self.assertEqual(len(problem.calls), 4)
        self.assertEqual(missing.calls, [])


if __name__ == "__main__":
    unittest.main()
