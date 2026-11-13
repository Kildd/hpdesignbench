"""
Phase PFN1 gate: pfns4bo package laziness, model load, PFN_PEN / PFN_CEI.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import unittest
from unittest.mock import patch

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


@unittest.skipUnless(
    importlib.util.find_spec("pfns4bo") is not None
    and importlib.util.find_spec("torch") is not None,
    "algorithms-pfns4bo extra not installed",
)
class TestPfns4boPackageScaffold(unittest.TestCase):
    """Optional pfns4bo package remains lazy; model path resolves."""

    def test_pfns4bo_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name.startswith("core.ioh_core.algorithms.pfns4bo"):
                sys.modules.pop(name, None)

        mod = importlib.import_module("core.ioh_core.algorithms.pfns4bo")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn(
            "core.ioh_core.algorithms.pfns4bo.algorithms",
            sys.modules,
        )
        self.assertNotIn(
            "core.ioh_core.algorithms.pfns4bo._vendor.pfn_cei",
            sys.modules,
        )

    def test_model_path_and_suggest_smoke(self) -> None:
        import torch

        from core.ioh_core.algorithms.pfns4bo.algorithms import (
            _ensure_hebo_plus_model_path,
            _patch_pfns4bo_torch_compat,
        )

        path = _ensure_hebo_plus_model_path()
        self.assertTrue(path.endswith(".pt"))
        _patch_pfns4bo_torch_compat()
        from pfns4bo.scripts.acquisition_functions import TransformerBOMethod

        model = torch.load(path, map_location="cpu", weights_only=False)
        pfn = TransformerBOMethod(model, device="cpu:0")
        x_obs = torch.rand(4, 2)
        y_obs = torch.rand(4)
        x_pen = torch.rand(8, 2)
        index = pfn.observe_and_suggest(x_obs, y_obs, x_pen)
        self.assertIsInstance(index, int)
        self.assertGreaterEqual(index, 0)
        self.assertLess(index, 8)


@unittest.skipUnless(
    importlib.util.find_spec("pfns4bo") is not None
    and importlib.util.find_spec("torch") is not None,
    "algorithms-pfns4bo extra not installed",
)
class TestPfnPenAdapter(unittest.TestCase):
    """Focused PFN_PEN behaviour."""

    token = "PFN_PEN"

    def test_registry_metadata(self) -> None:
        from core.ioh_core.algorithms.registry import (
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
        )

        spec = get_algorithm_spec(self.token)
        self.assertEqual(spec.library, "pfns4bo")
        self.assertEqual(spec.dependency_extra, "algorithms-pfns4bo")
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)

    def test_hard_budget_respected(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = FakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            self.token,
            {"max_evaluations": 8},
            random_seed=8,
        )(problem)
        self.assertLessEqual(len(problem.calls), 8)
        self.assertGreaterEqual(len(problem.calls), 1)
        self.assertEqual(problem.state.evaluations, len(problem.calls))

    def test_pen_does_not_read_constraint_slacks(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = FakeIOHProblem([0, 0], [2, 2])
        with patch(
            "core.ioh_core.algorithms.pfns4bo.algorithms."
            "read_last_evaluation_constraint_slacks",
            side_effect=AssertionError("PEN must not read slacks"),
        ):
            create_algorithm(
                self.token,
                {"max_evaluations": 6},
                random_seed=3,
            )(problem)
        self.assertGreaterEqual(len(problem.calls), 1)

    def test_deterministic_with_seed(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem_a = FakeIOHProblem([0, 1], [3, 4])
        problem_b = FakeIOHProblem([0, 1], [3, 4])
        create_algorithm(
            self.token,
            {"max_evaluations": 8},
            random_seed=17,
        )(problem_a)
        create_algorithm(
            self.token,
            {"max_evaluations": 8},
            random_seed=17,
        )(problem_b)
        self.assertEqual(problem_a.calls, problem_b.calls)

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
                {"max_evaluations": 5, "n_pending": 10},
                random_seed=1,
            )


@unittest.skipUnless(
    importlib.util.find_spec("pfns4bo") is not None
    and importlib.util.find_spec("torch") is not None,
    "algorithms-pfns4bo extra not installed",
)
class TestPfnCeiAdapter(unittest.TestCase):
    """Focused PFN_CEI behaviour."""

    token = "PFN_CEI"

    def test_registry_metadata(self) -> None:
        from core.ioh_core.algorithms.registry import (
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
        )

        spec = get_algorithm_spec(self.token)
        self.assertEqual(spec.library, "pfns4bo")
        self.assertEqual(spec.dependency_extra, "algorithms-pfns4bo")
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)

    def test_hard_budget_respected(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = _ConstrainedFakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            self.token,
            {"max_evaluations": 8},
            random_seed=8,
        )(problem)
        self.assertLessEqual(len(problem.calls), 8)
        self.assertGreaterEqual(len(problem.calls), 1)

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

    def test_uses_signed_slacks_not_clipped_violations(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from core.ioh_core.algorithms.common import (
            read_last_evaluation_constraint_slacks,
        )

        problem = _ConstrainedFakeIOHProblem([0, 0], [2, 2])
        create_algorithm(
            self.token,
            {"max_evaluations": 6},
            random_seed=4,
        )(problem)
        # Last evaluation must expose signed slack x0-1 (may be negative).
        slacks = read_last_evaluation_constraint_slacks(problem)
        self.assertEqual(len(slacks), 1)
        last_x0 = problem.calls[-1][0]
        self.assertAlmostEqual(slacks[0], float(last_x0) - 1.0)

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


if __name__ == "__main__":
    unittest.main()
