"""
Phase ALG5.1 gate: Pymoo package laziness and locked 0.6.2 API smoke.
"""
from __future__ import annotations

import importlib
import sys
import unittest

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version

from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem


_LOCKED_PYMOO_SYMBOLS = (
    ("pymoo.algorithms.soo.nonconvex.ga", "GA"),
    ("pymoo.core.problem", "ElementwiseProblem"),
    ("pymoo.operators.crossover.sbx", "SBX"),
    ("pymoo.operators.mutation.pm", "PM"),
    ("pymoo.operators.repair.rounding", "RoundingRepair"),
    ("pymoo.operators.sampling.rnd", "IntegerRandomSampling"),
    ("pymoo.optimize", "minimize"),
)


class TestPymooPackageScaffold(unittest.TestCase):
    """Optional Pymoo package remains lazy and API-compatible."""

    def test_pymoo_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name == "pymoo" or name.startswith("pymoo."):
                sys.modules.pop(name, None)
            if name.startswith("core.ioh_core.algorithms.pymoo."):
                sys.modules.pop(name, None)

        mod = importlib.import_module("core.ioh_core.algorithms.pymoo")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn("pymoo", sys.modules)

    def test_pymoo_api_contract_062(self) -> None:
        try:
            installed = version("pymoo")
        except PackageNotFoundError as exc:
            self.fail(f"pymoo extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("0.6.2"))
        self.assertLess(parsed, Version("0.7"))

        for module_name, attr_name in _LOCKED_PYMOO_SYMBOLS:
            module = importlib.import_module(module_name)
            self.assertTrue(
                hasattr(module, attr_name),
                msg=f"missing locked symbol {module_name}:{attr_name}",
            )
            self.assertTrue(callable(getattr(module, attr_name)))


class TestPymooIOHIntegerProblem(unittest.TestCase):
    """IOHIntegerProblem bounds, validation, and budget sentinel."""

    def test_pymoo_problem_declares_integer_bounds_and_one_objective(self) -> None:
        import numpy as np

        from core.ioh_core.algorithms.pymoo.problem import IOHIntegerProblem
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 2], [4, 6])
        adapter = IOHIntegerProblem(problem, max_evaluations=5)
        self.assertEqual(adapter.n_var, 2)
        self.assertEqual(adapter.n_obj, 1)
        self.assertEqual(adapter.n_ieq_constr, 0)
        self.assertEqual(adapter.vtype, int)
        np.testing.assert_array_equal(adapter.xl, np.asarray([0, 2], dtype=int))
        np.testing.assert_array_equal(adapter.xu, np.asarray([4, 6], dtype=int))
        self.assertEqual(adapter.lower_bounds, [0, 2])
        self.assertEqual(adapter.upper_bounds, [4, 6])

    def test_pymoo_problem_validates_candidates_before_ioh(self) -> None:
        from core.ioh_core.algorithms.common import CandidateError
        from core.ioh_core.algorithms.pymoo.problem import IOHIntegerProblem
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])
        adapter = IOHIntegerProblem(problem, max_evaluations=3)
        out: dict = {}
        with self.assertRaises(CandidateError):
            adapter._evaluate([0.5, 1], out)
        self.assertEqual(problem.calls, [])
        self.assertEqual(problem.state.evaluations, 0)

        with self.assertRaises(CandidateError):
            adapter._evaluate([0, 9], out)
        self.assertEqual(problem.calls, [])

    def test_pymoo_problem_uses_finite_sentinel_after_budget(self) -> None:
        import numpy as np

        from core.ioh_core.algorithms.pymoo.problem import (
            PYMOO_BUDGET_EXHAUSTED_FITNESS,
            IOHIntegerProblem,
        )
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0], [3])
        adapter = IOHIntegerProblem(problem, max_evaluations=2)
        out: dict = {}
        adapter._evaluate([1], out)
        self.assertEqual(out["F"], 1.0)
        adapter._evaluate([2], out)
        self.assertEqual(out["F"], 4.0)
        self.assertEqual(problem.state.evaluations, 2)

        adapter._evaluate([0], out)
        self.assertEqual(out["F"], PYMOO_BUDGET_EXHAUSTED_FITNESS)
        self.assertEqual(problem.state.evaluations, 2)
        self.assertEqual(len(problem.calls), 2)
        self.assertTrue(np.isfinite(out["F"]))
        self.assertNotEqual(out["F"], float("inf"))

    def test_pymoo_problem_population_larger_than_budget(self) -> None:
        from core.ioh_core.algorithms.pymoo.problem import (
            PYMOO_BUDGET_EXHAUSTED_FITNESS,
            IOHIntegerProblem,
        )
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])
        budget = 3
        adapter = IOHIntegerProblem(problem, max_evaluations=budget)
        fitnesses: list[float] = []
        for candidate in (
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 2],
            [2, 0],
        ):
            out: dict = {}
            adapter._evaluate(candidate, out)
            fitnesses.append(float(out["F"]))

        self.assertEqual(problem.state.evaluations, budget)
        self.assertEqual(len(problem.calls), budget)
        self.assertEqual(
            fitnesses[budget:],
            [PYMOO_BUDGET_EXHAUSTED_FITNESS] * (len(fitnesses) - budget),
        )


class TestPymooGeneticAlgorithm(unittest.TestCase):
    """Registered PYMOO_GA adapter behaviour."""

    def test_pymoo_ga_registry_spec(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec("PYMOO_GA")
        self.assertEqual(spec.library, "pymoo")
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.pymoo.factory:build_pymoo_ga",
        )
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-pymoo")
        self.assertTrue(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations", "population_size", "eliminate_duplicates"],
        )
        normalized = normalize_algorithm_parameters(
            "PYMOO_GA",
            {"max_evaluations": 10},
        )
        self.assertEqual(
            normalized,
            {
                "max_evaluations": 10,
                "population_size": 50,
                "eliminate_duplicates": True,
            },
        )

    def test_pymoo_ga_uses_locked_integer_operators(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithm_factory import create_algorithm
        from core.ioh_core.algorithms.pymoo import algorithms as pymoo_algorithms
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        captured: dict = {}

        def _fake_ga(**kwargs):
            captured["ga"] = kwargs
            return object()

        def _fake_minimize(adapter, algorithm, termination, seed, verbose):
            captured["minimize"] = {
                "adapter": adapter,
                "algorithm": algorithm,
                "termination": termination,
                "seed": seed,
                "verbose": verbose,
            }
            return object()

        with patch.object(pymoo_algorithms, "GA", side_effect=_fake_ga), patch.object(
            pymoo_algorithms,
            "minimize",
            side_effect=_fake_minimize,
        ):
            algo = create_algorithm(
                "PYMOO_GA",
                {"max_evaluations": 8, "population_size": 4},
                random_seed=3,
            )
            algo(FakeIOHProblem([0, 0], [2, 2]))

        ga_kwargs = captured["ga"]
        self.assertEqual(ga_kwargs["pop_size"], 4)
        self.assertIsInstance(
            ga_kwargs["sampling"],
            pymoo_algorithms.IntegerRandomSampling,
        )
        self.assertIsInstance(ga_kwargs["crossover"], pymoo_algorithms.SBX)
        self.assertIsInstance(ga_kwargs["mutation"], pymoo_algorithms.PM)
        self.assertIsInstance(
            ga_kwargs["crossover"].repair,
            pymoo_algorithms.RoundingRepair,
        )
        self.assertIsInstance(
            ga_kwargs["mutation"].repair,
            pymoo_algorithms.RoundingRepair,
        )
        self.assertTrue(ga_kwargs["eliminate_duplicates"])
        termination = captured["minimize"]["termination"]
        self.assertIsInstance(
            termination,
            pymoo_algorithms.TerminationCollection,
        )
        self.assertEqual(len(termination.terminations), 2)
        self.assertIsInstance(
            termination.terminations[0],
            pymoo_algorithms.MaximumFunctionCallTermination,
        )
        self.assertEqual(termination.terminations[0].n_max_evals, 8)
        from core.ioh_core.algorithms.pymoo.termination import (
            ControllerAwareTermination,
        )

        self.assertIsInstance(
            termination.terminations[1],
            ControllerAwareTermination,
        )
        self.assertEqual(captured["minimize"]["seed"], 3)
        self.assertFalse(captured["minimize"]["verbose"])

    def test_pymoo_ga_same_seed_same_candidate_sequence(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem_a = FakeIOHProblem([0, 1], [3, 4])
        problem_b = FakeIOHProblem([0, 1], [3, 4])
        create_algorithm(
            "PYMOO_GA",
            {"max_evaluations": 12, "population_size": 4},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            "PYMOO_GA",
            {"max_evaluations": 12, "population_size": 4},
            random_seed=17,
        )(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)
        self.assertGreaterEqual(len(problem_a.calls), 1)
        self.assertTrue(all(isinstance(v, int) for c in problem_a.calls for v in c))

    def test_pymoo_ga_does_not_re_evaluate_result(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])
        algo = create_algorithm(
            "PYMOO_GA",
            {"max_evaluations": 10, "population_size": 4},
            random_seed=5,
        )
        algo(problem)
        call_count = len(problem.calls)
        self.assertEqual(problem.state.evaluations, call_count)
        self.assertLessEqual(call_count, 10)
        self.assertIn(problem.state.current_best.x, problem.calls)
        # No post-minimize re-evaluation of result.X.
        self.assertEqual(len(problem.calls), call_count)
        self.assertEqual(problem.state.evaluations, call_count)

    def test_pymoo_missing_dependency_uses_install_hint(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.pymoo.factory import build_pymoo_ga

        with patch(
            "core.ioh_core.algorithms.pymoo.factory.import_optional_dependency",
            side_effect=AlgorithmDependencyError(
                "PYMOO_GA requires optional dependency 'pymoo'; install with:\n"
                'pip install -e ".[algorithms-pymoo]"'
            ),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                build_pymoo_ga({"max_evaluations": 5}, random_seed=0)
        self.assertIn("pip install -e \".[algorithms-pymoo]\"", str(ctx.exception))

    def test_pymoo_ga_tiny_smoke_against_fake_problem(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [3, 3])
        algo = create_algorithm(
            "PYMOO_GA",
            {"max_evaluations": 10, "population_size": 5},
            random_seed=9,
        )
        algo(problem)
        self.assertGreaterEqual(len(problem.calls), 1)
        self.assertLessEqual(problem.state.evaluations, 10)
        self.assertIn(problem.state.current_best.x, problem.calls)

    def test_pymoo_ga_target_hit_stops_without_ioh_overshoot(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from core.ioh_core.algorithms.common import RunController
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [0, 0])
        controller = RunController(
            problem,
            20,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "PYMOO_GA",
            {"max_evaluations": 20, "population_size": 8},
            random_seed=1,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 1)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "target_hit")

    def test_pymoo_ga_target_miss_uses_hard_ceiling(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from core.ioh_core.algorithms.common import RunController
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([2, 2], [4, 4])
        controller = RunController(
            problem,
            6,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "PYMOO_GA",
            {"max_evaluations": 6, "population_size": 4},
            random_seed=2,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 6)
        self.assertFalse(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")

    def test_pymoo_problem_sentinel_after_target_hit(self) -> None:
        from core.ioh_core.algorithms.common import RunController
        from core.ioh_core.algorithms.pymoo.problem import (
            PYMOO_BUDGET_EXHAUSTED_FITNESS,
            IOHIntegerProblem,
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
        adapter = IOHIntegerProblem(
            problem,
            5,
            run_controller=controller,
        )
        out: dict = {}
        adapter._evaluate([0], out)
        self.assertEqual(out["F"], 0.0)
        self.assertTrue(controller.target_hit)
        adapter._evaluate([0], out)
        self.assertEqual(out["F"], PYMOO_BUDGET_EXHAUSTED_FITNESS)
        self.assertEqual(len(problem.calls), 1)


class _ConstrainedFakeIOHProblem(FakeIOHProblem):
    """
    Fake IOH problem that publishes unpenalized ``y`` and utilizations.

    Sphere objective is used as both ``y`` and ``y_p`` unless
    ``y_scale`` multiplies the unpenalized value. Constraint ``g0`` uses
    utilization ``η = x[0]`` (feasible iff ``x[0] ≤ 1`` ⇒ signed slack
    ``η − 1``).
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
            {
                "_c_names": list(names),
                "last_data": None,
                "run_config": type(
                    "StubRunConfig",
                    (),
                    {"invalid_utilization": 99.0},
                )(),
            },
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
            "analysis_ok": True,
            "feasible": feasible,
        }
        self.hdb_last_feasible = feasible
        return y_p


class TestPymooConstrainedIntegerProblem(unittest.TestCase):
    """IOHIntegerConstrainedProblem bounds, y/G mapping, and sentinel."""

    def test_pymoo_constrained_problem_declares_ieq_constraints(self) -> None:
        import numpy as np

        from core.ioh_core.algorithms.pymoo.problem import (
            IOHIntegerConstrainedProblem,
        )

        problem = _ConstrainedFakeIOHProblem(
            [0, 2],
            [4, 6],
            constraint_names=["a", "b"],
        )
        adapter = IOHIntegerConstrainedProblem(problem, max_evaluations=5)
        self.assertEqual(adapter.n_var, 2)
        self.assertEqual(adapter.n_obj, 1)
        self.assertEqual(adapter.n_ieq_constr, 2)
        self.assertEqual(adapter.vtype, int)
        np.testing.assert_array_equal(adapter.xl, np.asarray([0, 2], dtype=int))
        np.testing.assert_array_equal(adapter.xu, np.asarray([4, 6], dtype=int))

    def test_pymoo_constrained_problem_rejects_zero_constraints(self) -> None:
        from core.ioh_core.algorithms.pymoo.problem import (
            IOHIntegerConstrainedProblem,
        )

        problem = _ConstrainedFakeIOHProblem(
            [0],
            [2],
            constraint_names=[],
        )
        with self.assertRaisesRegex(ValueError, "at least one active"):
            IOHIntegerConstrainedProblem(problem, max_evaluations=3)

    def test_pymoo_constrained_problem_uses_y_not_y_p_for_F(self) -> None:
        from core.ioh_core.algorithms.pymoo.problem import (
            IOHIntegerConstrainedProblem,
        )

        problem = _ConstrainedFakeIOHProblem(
            [0],
            [3],
            constraint_names=["g0"],
            y_scale=2.0,
        )
        adapter = IOHIntegerConstrainedProblem(problem, max_evaluations=3)
        out: dict = {}
        adapter._evaluate([2], out)
        # y_p = 2^2 = 4 from FakeIOHProblem; y = 2 * 4 = 8 via y_scale.
        self.assertEqual(out["F"], 8.0)
        self.assertEqual(problem.state.current_best.y, 4.0)
        # Signed slack η−1 with η = x[0] = 2 → G = [1.0].
        self.assertEqual(out["G"], [1.0])

    def test_pymoo_constrained_problem_maps_signed_slacks_to_G(self) -> None:
        from core.ioh_core.algorithms.pymoo.problem import (
            IOHIntegerConstrainedProblem,
        )

        problem = _ConstrainedFakeIOHProblem(
            [0, 0],
            [3, 3],
            constraint_names=["g0", "g1"],
        )
        adapter = IOHIntegerConstrainedProblem(problem, max_evaluations=3)
        out: dict = {}
        adapter._evaluate([0, 1], out)
        self.assertEqual(out["F"], 1.0)
        # g0: η=0 → −1; g1: η=0 → −1 (feasible slack, not clipped 0).
        self.assertEqual(out["G"], [-1.0, -1.0])
        adapter._evaluate([3, 0], out)
        self.assertEqual(out["G"], [2.0, -1.0])

    def test_pymoo_constrained_problem_sentinel_sets_F_and_G(self) -> None:
        from core.ioh_core.algorithms.pymoo.problem import (
            PYMOO_BUDGET_EXHAUSTED_FITNESS,
            IOHIntegerConstrainedProblem,
        )

        problem = _ConstrainedFakeIOHProblem(
            [0],
            [3],
            constraint_names=["g0", "g1"],
        )
        adapter = IOHIntegerConstrainedProblem(problem, max_evaluations=1)
        out: dict = {}
        adapter._evaluate([1], out)
        self.assertEqual(problem.state.evaluations, 1)
        adapter._evaluate([2], out)
        self.assertEqual(out["F"], PYMOO_BUDGET_EXHAUSTED_FITNESS)
        self.assertEqual(
            out["G"],
            [PYMOO_BUDGET_EXHAUSTED_FITNESS, PYMOO_BUDGET_EXHAUSTED_FITNESS],
        )
        self.assertEqual(problem.state.evaluations, 1)
        self.assertEqual(len(problem.calls), 1)

    def test_pymoo_constrained_problem_population_larger_than_budget(self) -> None:
        from core.ioh_core.algorithms.pymoo.problem import (
            PYMOO_BUDGET_EXHAUSTED_FITNESS,
            IOHIntegerConstrainedProblem,
        )

        problem = _ConstrainedFakeIOHProblem(
            [0, 0],
            [2, 2],
            constraint_names=["g0"],
        )
        budget = 3
        adapter = IOHIntegerConstrainedProblem(problem, max_evaluations=budget)
        fitnesses: list[float] = []
        for candidate in (
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 2],
        ):
            out: dict = {}
            adapter._evaluate(candidate, out)
            fitnesses.append(float(out["F"]))

        self.assertEqual(problem.state.evaluations, budget)
        self.assertEqual(len(problem.calls), budget)
        self.assertEqual(
            fitnesses[budget:],
            [PYMOO_BUDGET_EXHAUSTED_FITNESS] * (len(fitnesses) - budget),
        )


class TestPymooConstrainedGeneticAlgorithm(unittest.TestCase):
    """Registered PYMOO_GA_CONSTRAINED adapter behaviour."""

    def test_pymoo_ga_constrained_registry_spec(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec("PYMOO_GA_CONSTRAINED")
        self.assertEqual(spec.library, "pymoo")
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.pymoo.factory:build_pymoo_ga_constrained",
        )
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-pymoo")
        self.assertTrue(spec.batch_evaluations)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations", "population_size", "eliminate_duplicates"],
        )
        normalized = normalize_algorithm_parameters(
            "PYMOO_GA_CONSTRAINED",
            {"max_evaluations": 10},
        )
        self.assertEqual(
            normalized,
            {
                "max_evaluations": 10,
                "population_size": 50,
                "eliminate_duplicates": True,
            },
        )

    def test_pymoo_ga_constrained_uses_constrained_problem(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithm_factory import create_algorithm
        from core.ioh_core.algorithms.pymoo import algorithms as pymoo_algorithms
        from core.ioh_core.algorithms.pymoo.problem import (
            IOHIntegerConstrainedProblem,
        )

        captured: dict = {}

        def _fake_minimize(adapter, algorithm, termination, seed, verbose):
            captured["adapter"] = adapter
            captured["seed"] = seed
            return object()

        with patch.object(
            pymoo_algorithms,
            "minimize",
            side_effect=_fake_minimize,
        ):
            algo = create_algorithm(
                "PYMOO_GA_CONSTRAINED",
                {"max_evaluations": 8, "population_size": 4},
                random_seed=3,
            )
            algo(_ConstrainedFakeIOHProblem([0, 0], [2, 2]))

        self.assertIsInstance(captured["adapter"], IOHIntegerConstrainedProblem)
        self.assertEqual(captured["adapter"].n_ieq_constr, 1)
        self.assertEqual(captured["seed"], 3)

    def test_pymoo_ga_constrained_same_seed_same_candidate_sequence(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem_a = _ConstrainedFakeIOHProblem([0, 1], [3, 4])
        problem_b = _ConstrainedFakeIOHProblem([0, 1], [3, 4])
        create_algorithm(
            "PYMOO_GA_CONSTRAINED",
            {"max_evaluations": 12, "population_size": 4},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            "PYMOO_GA_CONSTRAINED",
            {"max_evaluations": 12, "population_size": 4},
            random_seed=17,
        )(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)
        self.assertGreaterEqual(len(problem_a.calls), 1)

    def test_pymoo_ga_constrained_exact_budget(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            "PYMOO_GA_CONSTRAINED",
            {"max_evaluations": 10, "population_size": 4},
            random_seed=8,
        )(problem)
        self.assertEqual(len(problem.calls), 10)
        self.assertEqual(problem.state.evaluations, 10)

    def test_pymoo_ga_constrained_does_not_re_evaluate_best(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem([0, 0], [2, 2])
        algo = create_algorithm(
            "PYMOO_GA_CONSTRAINED",
            {"max_evaluations": 10, "population_size": 4},
            random_seed=5,
        )
        algo(problem)
        call_count = len(problem.calls)
        self.assertEqual(problem.state.evaluations, call_count)
        self.assertLessEqual(call_count, 10)
        self.assertIn(problem.state.current_best.x, problem.calls)
        self.assertEqual(len(problem.calls), call_count)

    def test_pymoo_ga_constrained_missing_seed_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
        )
        from core.ioh_core.algorithms.pymoo.factory import (
            build_pymoo_ga_constrained,
        )

        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "PYMOO_GA_CONSTRAINED",
                {"max_evaluations": 5, "population_size": 4},
                random_seed=None,
            )
        with self.assertRaises(ValueError):
            build_pymoo_ga_constrained(
                {"max_evaluations": 5, "population_size": 4},
                random_seed=None,
            )

    def test_pymoo_ga_constrained_unknown_parameter_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
            validate_algorithm_parameters,
        )

        with self.assertRaises(AlgorithmFactoryError):
            validate_algorithm_parameters(
                "PYMOO_GA_CONSTRAINED",
                {
                    "max_evaluations": 5,
                    "population_size": 4,
                    "constraint_mode": "native",
                },
                random_seed=1,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "PYMOO_GA_CONSTRAINED",
                {
                    "max_evaluations": 5,
                    "population_size": 4,
                    "constraint_mode": "native",
                },
                random_seed=1,
            )

    def test_pymoo_ga_constrained_missing_dependency_uses_install_hint(
        self,
    ) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.pymoo.factory import (
            build_pymoo_ga_constrained,
        )

        with patch(
            "core.ioh_core.algorithms.pymoo.factory.import_optional_dependency",
            side_effect=AlgorithmDependencyError(
                "PYMOO_GA_CONSTRAINED requires optional dependency 'pymoo'; "
                'install with:\npip install -e ".[algorithms-pymoo]"'
            ),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                build_pymoo_ga_constrained(
                    {"max_evaluations": 5, "population_size": 4},
                    random_seed=0,
                )
        self.assertIn(
            'pip install -e ".[algorithms-pymoo]"',
            str(ctx.exception),
        )


_P2_LOWER = [0, 0, 0, 0, 3, 0, 0, 0, 0]
_P2_UPPER = [10, 8, 5, 5, 14, 12, 3, 5, 3]


class TestPymooConstrainedGeneticAlgorithmWarmStart(unittest.TestCase):
    """PYMOO_GA_CONSTRAINED_WARM_START uses a ranked InitialDesign prefix."""

    token = "PYMOO_GA_CONSTRAINED_WARM_START"

    def test_registry_metadata(self) -> None:
        from core.ioh_core.algorithms.registry import (
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec(self.token)
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            [
                "max_evaluations",
                "population_size",
                "eliminate_duplicates",
                "initial_design_path",
            ],
        )
        self.assertIsNone(spec.parameters[1].maximum)
        normalized = normalize_algorithm_parameters(
            self.token,
            {
                "max_evaluations": 10,
                "population_size": 4,
                "initial_design_path": "design.json",
            },
        )
        self.assertEqual(normalized["population_size"], 4)

    def test_initial_population_is_ranked_prefix(self) -> None:
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from core.ioh_core.algorithm_factory import create_algorithm

        vectors = [
            [index, 0, 0, 0, 3, 0, 0, 0, 0] for index in range(5)
        ]
        problem = _ConstrainedFakeIOHProblem(_P2_LOWER, _P2_UPPER)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
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
            create_algorithm(
                self.token,
                {
                    "max_evaluations": 4,
                    "population_size": 4,
                    "initial_design_path": str(path),
                },
                random_seed=3,
            )(problem)
        self.assertEqual(problem.calls[:4], vectors[:4])


if __name__ == "__main__":
    unittest.main()
