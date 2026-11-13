"""
Phase OT1/OT2 gate: Optuna package laziness, API contract, and OPTUNA_TPE.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from unittest.mock import MagicMock, patch

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version

from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem


_LOCKED_OPTUNA_CALLABLES = (
    ("optuna", "create_study"),
    ("optuna.samplers", "TPESampler"),
    ("optuna.logging", "set_verbosity"),
)


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


class TestOptunaPackageScaffold(unittest.TestCase):
    """Optional Optuna package remains lazy and API-compatible."""

    def test_optuna_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name == "optuna" or name.startswith("optuna."):
                sys.modules.pop(name, None)
            if name.startswith("core.ioh_core.algorithms.optuna."):
                sys.modules.pop(name, None)
        sys.modules.pop("core.ioh_core.algorithms.optuna", None)

        mod = importlib.import_module("core.ioh_core.algorithms.optuna")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn("optuna", sys.modules)

    def test_optuna_api_contract_4x(self) -> None:
        try:
            installed = version("optuna")
        except PackageNotFoundError as exc:
            self.fail(f"optuna extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("4.0.0"))
        self.assertLess(parsed, Version("5"))

        for module_name, attr_name in _LOCKED_OPTUNA_CALLABLES:
            module = importlib.import_module(module_name)
            self.assertTrue(
                hasattr(module, attr_name),
                msg=f"missing locked symbol {module_name}:{attr_name}",
            )
            self.assertTrue(callable(getattr(module, attr_name)))

        import optuna
        from optuna.samplers import TPESampler

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=0),
        )

        def objective(trial: optuna.trial.Trial) -> float:
            value = trial.suggest_int("x_0000", 0, 3)
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 3)
            return float(value)

        study.optimize(
            objective,
            n_trials=1,
            n_jobs=1,
            catch=(),
            show_progress_bar=False,
        )
        self.assertEqual(len(study.trials), 1)
        self.assertIsInstance(study.trials[0].params["x_0000"], int)


class TestOptunaIntegerSpace(unittest.TestCase):
    """Inclusive-bound native integer Optuna search space helpers."""

    def test_optuna_suggest_int_uses_inclusive_bounds(self) -> None:
        from core.ioh_core.algorithms.optuna.tpe import candidate_from_optuna_trial

        trial = MagicMock()
        trial.suggest_int.return_value = 0
        candidate = candidate_from_optuna_trial(
            trial,
            lower_bounds=[0],
            upper_bounds=[3],
        )
        trial.suggest_int.assert_called_once_with("x_0000", 0, 3)
        self.assertEqual(candidate, [0])

    def test_optuna_degenerate_bound_skips_suggest_int(self) -> None:
        from core.ioh_core.algorithms.optuna.tpe import candidate_from_optuna_trial

        trial = MagicMock()
        trial.suggest_int.return_value = 4
        candidate = candidate_from_optuna_trial(
            trial,
            lower_bounds=[2, 0],
            upper_bounds=[2, 5],
        )
        trial.suggest_int.assert_called_once_with("x_0001", 0, 5)
        self.assertEqual(candidate, [2, 4])

    def test_optuna_candidate_reconstruction_uses_index_order(self) -> None:
        from core.ioh_core.algorithms.optuna.tpe import (
            candidate_from_optuna_trial,
            optuna_dimension_label,
        )

        trial = MagicMock()

        def _suggest(name: str, low: int, high: int) -> int:
            values = {
                optuna_dimension_label(0): 1,
                optuna_dimension_label(1): 4,
                optuna_dimension_label(2): 9,
            }
            return values[name]

        trial.suggest_int.side_effect = _suggest
        candidate = candidate_from_optuna_trial(
            trial,
            lower_bounds=[0, 2, 5],
            upper_bounds=[4, 6, 9],
        )
        self.assertEqual(candidate, [1, 4, 9])
        self.assertEqual(
            [call.args[0] for call in trial.suggest_int.call_args_list],
            ["x_0000", "x_0001", "x_0002"],
        )


class TestOptunaTPE(unittest.TestCase):
    """Registered OPTUNA_TPE adapter behaviour."""

    def test_optuna_tpe_registry_spec(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec("OPTUNA_TPE")
        self.assertEqual(spec.library, "optuna")
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.optuna.factory:build_optuna_tpe",
        )
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-optuna")
        self.assertFalse(spec.batch_evaluations)
        self.assertIn("constrained", spec.description.lower())
        self.assertEqual(
            [parameter.name for parameter in spec.parameters],
            ["max_evaluations"],
        )
        normalized = normalize_algorithm_parameters(
            "OPTUNA_TPE",
            {"max_evaluations": 10},
        )
        self.assertEqual(normalized, {"max_evaluations": 10})

    def test_optuna_tpe_rejects_zero_constraints(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem(
            [0],
            [2],
            constraint_names=[],
        )
        algo = create_algorithm(
            "OPTUNA_TPE",
            {"max_evaluations": 3},
            random_seed=1,
        )
        with self.assertRaisesRegex(ValueError, "at least one active"):
            algo(problem)

    def test_optuna_tpe_returns_y_not_y_p_and_stores_slacks(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem(
            [0],
            [3],
            constraint_names=["g0"],
            y_scale=2.0,
        )
        create_algorithm(
            "OPTUNA_TPE",
            {"max_evaluations": 1},
            random_seed=0,
        )(problem)
        self.assertEqual(len(problem.calls), 1)
        # Authoritative current_best stays on y_p (= x^2).
        x0 = int(problem.calls[0][0])
        self.assertEqual(problem.state.current_best.y, float(x0 * x0))

        import optuna
        from optuna.samplers import TPESampler

        captured: dict = {}

        def _fake_optimize(self, func, **kwargs):  # noqa: ANN001
            trial = self.ask()
            value = func(trial)
            captured["objective_value"] = value
            captured["user_attrs"] = dict(trial.user_attrs)
            self.tell(trial, value)
            return None

        problem2 = _ConstrainedFakeIOHProblem(
            [2],
            [2],
            constraint_names=["g0"],
            y_scale=2.0,
        )
        with patch("optuna.study.Study.optimize", _fake_optimize):
            create_algorithm(
                "OPTUNA_TPE",
                # Budget > 1 so objective does not call study.stop() under the
                # patched ask/tell harness (stop only works inside optimize).
                {"max_evaluations": 2},
                random_seed=2,
            )(problem2)
        # y = 2 * (2^2) = 8; y_p = 4; slack η−1 with η=2 → 1.0
        self.assertEqual(captured["objective_value"], 8.0)
        self.assertEqual(captured["user_attrs"]["constraints"], (1.0,))
        self.assertEqual(problem2.state.current_best.y, 4.0)
        self.assertIsNotNone(optuna)
        self.assertIsNotNone(TPESampler)

    def test_optuna_tpe_calls_optimize_with_locked_arguments(self) -> None:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.study import StudyDirection

        from core.ioh_core.algorithm_factory import create_algorithm

        captured: dict = {}

        def _fake_optimize(self, func, **kwargs):  # noqa: ANN001
            captured["optimize"] = kwargs
            captured["study"] = self
            return None

        with patch("optuna.study.Study.optimize", _fake_optimize):
            with patch(
                "optuna.samplers.TPESampler",
                wraps=TPESampler,
            ) as mocked_sampler:
                algo = create_algorithm(
                    "OPTUNA_TPE",
                    {"max_evaluations": 7},
                    random_seed=11,
                )
                algo(_ConstrainedFakeIOHProblem([0, 0], [2, 2]))

        kwargs = captured["optimize"]
        self.assertEqual(kwargs["n_trials"], 7)
        self.assertEqual(kwargs["n_jobs"], 1)
        self.assertEqual(kwargs["catch"], ())
        self.assertFalse(kwargs["show_progress_bar"])
        self.assertEqual(captured["study"].direction, StudyDirection.MINIMIZE)
        mocked_sampler.assert_called_once()
        call_kwargs = mocked_sampler.call_args.kwargs
        self.assertEqual(call_kwargs["seed"], 11)
        self.assertTrue(callable(call_kwargs["constraints_func"]))

        problem = _ConstrainedFakeIOHProblem([0, 0], [2, 2])
        with patch("optuna.study.Study.optimize", _fake_optimize):
            create_algorithm(
                "OPTUNA_TPE",
                {"max_evaluations": 3},
                random_seed=1,
            )(problem)
        self.assertEqual(problem.calls, [])
        self.assertEqual(problem.state.evaluations, 0)
        self.assertIsNotNone(optuna)

    def test_optuna_tpe_same_seed_same_candidate_sequence(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem_a = _ConstrainedFakeIOHProblem([0, 1], [3, 4])
        problem_b = _ConstrainedFakeIOHProblem([0, 1], [3, 4])
        create_algorithm(
            "OPTUNA_TPE",
            {"max_evaluations": 12},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            "OPTUNA_TPE",
            {"max_evaluations": 12},
            random_seed=17,
        )(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)
        self.assertGreaterEqual(len(problem_a.calls), 1)
        self.assertTrue(
            all(isinstance(v, int) for c in problem_a.calls for v in c)
        )

    def test_optuna_tpe_different_seed_different_candidate_sequence(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem_a = _ConstrainedFakeIOHProblem([0, 1], [5, 6])
        problem_b = _ConstrainedFakeIOHProblem([0, 1], [5, 6])
        create_algorithm(
            "OPTUNA_TPE",
            {"max_evaluations": 20},
            random_seed=3,
        )(problem_a)
        create_algorithm(
            "OPTUNA_TPE",
            {"max_evaluations": 20},
            random_seed=99,
        )(problem_b)
        self.assertNotEqual(problem_a.calls, problem_b.calls)

    def test_optuna_tpe_does_not_re_evaluate_best(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem([0, 0], [2, 2])
        algo = create_algorithm(
            "OPTUNA_TPE",
            {"max_evaluations": 10},
            random_seed=5,
        )
        algo(problem)
        call_count = len(problem.calls)
        self.assertEqual(problem.state.evaluations, call_count)
        self.assertEqual(call_count, 10)
        self.assertIn(problem.state.current_best.x, problem.calls)
        self.assertEqual(len(problem.calls), call_count)
        self.assertEqual(problem.state.evaluations, call_count)

    def test_optuna_tpe_exact_budget(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            "OPTUNA_TPE",
            {"max_evaluations": 15},
            random_seed=8,
        )(problem)
        self.assertEqual(len(problem.calls), 15)
        self.assertEqual(problem.state.evaluations, 15)

    def test_optuna_tpe_target_hit_stops_early(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from core.ioh_core.algorithms.common import RunController

        problem = _ConstrainedFakeIOHProblem([0, 0], [0, 0])
        controller = RunController(
            problem,
            12,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "OPTUNA_TPE",
            {"max_evaluations": 12},
            random_seed=4,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 1)
        self.assertTrue(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "target_hit")

    def test_optuna_tpe_target_miss_exact_budget(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from core.ioh_core.algorithms.common import RunController

        problem = _ConstrainedFakeIOHProblem([2, 2], [4, 4])
        controller = RunController(
            problem,
            9,
            mode="fixed_target",
            target_threshold=1.0,
            target_reference="all",
        )
        create_algorithm(
            "OPTUNA_TPE",
            {"max_evaluations": 9},
            random_seed=6,
        )(problem, run_controller=controller)
        self.assertEqual(len(problem.calls), 9)
        self.assertFalse(controller.target_hit)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")

    def test_optuna_tpe_missing_seed_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
        )
        from core.ioh_core.algorithms.optuna.factory import build_optuna_tpe

        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm("OPTUNA_TPE", {"max_evaluations": 5}, random_seed=None)
        with self.assertRaises(ValueError):
            build_optuna_tpe({"max_evaluations": 5}, random_seed=None)

    def test_optuna_tpe_unknown_parameter_rejected(self) -> None:
        from core.ioh_core.algorithm_factory import (
            AlgorithmFactoryError,
            create_algorithm,
            validate_algorithm_parameters,
        )

        with self.assertRaises(AlgorithmFactoryError):
            validate_algorithm_parameters(
                "OPTUNA_TPE",
                {"max_evaluations": 5, "multivariate": True},
                random_seed=1,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                "OPTUNA_TPE",
                {"max_evaluations": 5, "multivariate": True},
                random_seed=1,
            )


if __name__ == "__main__":
    unittest.main()
