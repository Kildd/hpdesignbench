"""
Phase ALG6.1 gate: Hyperopt package laziness, API contract, and quantized space.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from collections import OrderedDict
from unittest.mock import patch

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version


_LOCKED_HYPEROPT_CALLABLES = (
    ("hyperopt", "Trials"),
    ("hyperopt", "fmin"),
    ("hyperopt.hp", "quniform"),
    ("hyperopt.tpe", "suggest"),
)


class TestHyperoptPackageScaffold(unittest.TestCase):
    """Optional Hyperopt package remains lazy and API-compatible."""

    def test_hyperopt_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name == "hyperopt" or name.startswith("hyperopt."):
                sys.modules.pop(name, None)
            if name.startswith("core.ioh_core.algorithms.hyperopt."):
                sys.modules.pop(name, None)

        mod = importlib.import_module("core.ioh_core.algorithms.hyperopt")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn("hyperopt", sys.modules)

    def test_hyperopt_api_contract_03(self) -> None:
        try:
            installed = version("hyperopt")
        except PackageNotFoundError as exc:
            self.fail(f"hyperopt extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("0.3.0"))
        self.assertLess(parsed, Version("0.4"))

        hyperopt = importlib.import_module("hyperopt")
        self.assertEqual(getattr(hyperopt, "STATUS_OK"), "ok")

        for module_name, attr_name in _LOCKED_HYPEROPT_CALLABLES:
            module = importlib.import_module(module_name)
            self.assertTrue(
                hasattr(module, attr_name),
                msg=f"missing locked symbol {module_name}:{attr_name}",
            )
            self.assertTrue(callable(getattr(module, attr_name)))


class TestHyperoptQuantizedSpace(unittest.TestCase):
    """Inclusive-bound quantized integer Hyperopt search space."""

    def test_hyperopt_space_uses_stable_labels_and_inclusive_bounds(self) -> None:
        from core.ioh_core.algorithms.hyperopt.algorithms import (
            build_quantized_integer_space,
            hyperopt_dimension_label,
        )

        lower = [0, 2, 5]
        upper = [4, 6, 9]
        with patch(
            "hyperopt.hp.quniform",
            side_effect=lambda label, low, high, q: (label, low, high, q),
        ) as mocked:
            space = build_quantized_integer_space(lower, upper)

        self.assertEqual(
            list(space.keys()),
            [hyperopt_dimension_label(i) for i in range(3)],
        )
        self.assertEqual(mocked.call_count, 3)
        mocked.assert_any_call("x_0000", 0, 4, 1)
        mocked.assert_any_call("x_0001", 2, 6, 1)
        mocked.assert_any_call("x_0002", 5, 9, 1)
        self.assertEqual(space["x_0000"], ("x_0000", 0, 4, 1))
        self.assertEqual(space["x_0001"], ("x_0001", 2, 6, 1))
        self.assertEqual(space["x_0002"], ("x_0002", 5, 9, 1))

    def test_hyperopt_space_does_not_add_one_to_upper_bound(self) -> None:
        from core.ioh_core.algorithms.hyperopt.algorithms import (
            build_quantized_integer_space,
        )

        with patch(
            "hyperopt.hp.quniform",
            side_effect=lambda label, low, high, q: (label, low, high, q),
        ) as mocked:
            build_quantized_integer_space([0], [3])

        mocked.assert_called_once_with("x_0000", 0, 3, 1)
        _label, _low, high, _q = mocked.call_args.args
        self.assertEqual(high, 3)
        self.assertNotEqual(high, 4)

    def test_hyperopt_space_uses_constant_for_degenerate_bound(self) -> None:
        from core.ioh_core.algorithms.hyperopt.algorithms import (
            build_quantized_integer_space,
        )

        with patch("hyperopt.hp.quniform") as mocked:
            space = build_quantized_integer_space([2, 0], [2, 5])

        self.assertEqual(space["x_0000"], 2)
        self.assertIsInstance(space["x_0000"], int)
        mocked.assert_called_once_with("x_0001", 0, 5, 1)

    def test_hyperopt_candidate_reconstruction_uses_index_order(self) -> None:
        from core.ioh_core.algorithms.hyperopt.algorithms import (
            candidate_from_hyperopt_sample,
        )

        # Insertion order deliberately reversed relative to dimension index.
        sample = OrderedDict(
            [
                ("x_0002", 9.0),
                ("x_0000", 1.0),
                ("x_0001", 4.0),
            ]
        )
        candidate = candidate_from_hyperopt_sample(
            sample,
            lower_bounds=[0, 2, 5],
            upper_bounds=[4, 6, 9],
        )
        self.assertEqual(candidate, [1, 4, 9])


class TestHyperoptTPE(unittest.TestCase):
    """Registered HYPEROPT_TPE adapter behaviour."""

    def test_hyperopt_tpe_registry_spec(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec("HYPEROPT_TPE")
        self.assertEqual(spec.library, "hyperopt")
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.hyperopt.factory:build_hyperopt_tpe",
        )
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.QUANTIZED_GRID)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-hyperopt")
        self.assertFalse(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations"],
        )
        normalized = normalize_algorithm_parameters(
            "HYPEROPT_TPE",
            {"max_evaluations": 10},
        )
        self.assertEqual(normalized, {"max_evaluations": 10})

    def test_hyperopt_tpe_calls_fmin_with_locked_arguments(self) -> None:
        import numpy as np
        from hyperopt import Trials, tpe

        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        captured: dict = {}

        def _fake_fmin(**kwargs):
            captured["fmin"] = kwargs
            return {"x_0000": 0.0, "x_0001": 0.0}

        with patch("hyperopt.fmin", side_effect=_fake_fmin):
            algo = create_algorithm(
                "HYPEROPT_TPE",
                {"max_evaluations": 7},
                random_seed=11,
            )
            algo(FakeIOHProblem([0, 0], [2, 2]))

        kwargs = captured["fmin"]
        self.assertIs(kwargs["algo"], tpe.suggest)
        self.assertEqual(kwargs["max_evals"], 7)
        self.assertFalse(kwargs["show_progressbar"])
        self.assertFalse(kwargs["verbose"])
        self.assertIsInstance(kwargs["trials"], Trials)
        self.assertIsInstance(kwargs["rstate"], np.random.Generator)
        self.assertEqual(set(kwargs["space"]), {"x_0000", "x_0001"})
        self.assertTrue(callable(kwargs["fn"]))
        self.assertTrue(callable(kwargs["early_stop_fn"]))
        # Returned best mapping must not be re-evaluated.
        problem = FakeIOHProblem([0, 0], [2, 2])
        with patch("hyperopt.fmin", side_effect=_fake_fmin):
            create_algorithm(
                "HYPEROPT_TPE",
                {"max_evaluations": 3},
                random_seed=1,
            )(problem)
        self.assertEqual(problem.calls, [])
        self.assertEqual(problem.state.evaluations, 0)

    def test_hyperopt_tpe_same_seed_same_candidate_sequence(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem_a = FakeIOHProblem([0, 1], [3, 4])
        problem_b = FakeIOHProblem([0, 1], [3, 4])
        create_algorithm(
            "HYPEROPT_TPE",
            {"max_evaluations": 12},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            "HYPEROPT_TPE",
            {"max_evaluations": 12},
            random_seed=17,
        )(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)
        self.assertGreaterEqual(len(problem_a.calls), 1)
        self.assertTrue(all(isinstance(v, int) for c in problem_a.calls for v in c))

    def test_hyperopt_tpe_does_not_re_evaluate_best(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])
        algo = create_algorithm(
            "HYPEROPT_TPE",
            {"max_evaluations": 10},
            random_seed=5,
        )
        algo(problem)
        call_count = len(problem.calls)
        self.assertEqual(problem.state.evaluations, call_count)
        self.assertEqual(call_count, 10)
        self.assertIn(problem.state.current_best.x, problem.calls)
        # No post-fmin re-evaluation of the returned best mapping.
        self.assertEqual(len(problem.calls), call_count)
        self.assertEqual(problem.state.evaluations, call_count)

    def test_hyperopt_tpe_target_hit_stops_early(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from core.ioh_core.algorithms.common import RunController
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [0, 0])
        controller = RunController(
            problem,
            12,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "HYPEROPT_TPE",
            {"max_evaluations": 12},
            random_seed=4,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 1)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "target_hit")

    def test_hyperopt_tpe_target_miss_exact_budget(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from core.ioh_core.algorithms.common import RunController
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([2, 2], [4, 4])
        controller = RunController(
            problem,
            7,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "HYPEROPT_TPE",
            {"max_evaluations": 7},
            random_seed=5,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 7)
        self.assertFalse(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
