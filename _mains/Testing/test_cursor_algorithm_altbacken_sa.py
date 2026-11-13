"""
Phase AB1 gate: altbacken package laziness, API contract, and ALTBACKEN_SA.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from random import Random

from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version


class TestAltbackenPackageScaffold(unittest.TestCase):
    """Optional altbacken package remains lazy and API-compatible."""

    def test_altbacken_package_import_is_lazy(self) -> None:
        for name in list(sys.modules):
            if name == "altbacken" or name.startswith("altbacken."):
                sys.modules.pop(name, None)
            if name.startswith("core.ioh_core.algorithms.altbacken."):
                sys.modules.pop(name, None)
        sys.modules.pop("core.ioh_core.algorithms.altbacken", None)

        mod = importlib.import_module("core.ioh_core.algorithms.altbacken")
        self.assertEqual(mod.__all__, ())
        self.assertNotIn("altbacken", sys.modules)

    def test_altbacken_sa_api_contract_0151(self) -> None:
        try:
            installed = version("altbacken")
        except PackageNotFoundError as exc:
            self.fail(f"altbacken extra is not installed: {exc}")

        parsed = Version(installed)
        self.assertGreaterEqual(parsed, Version("0.15.1"))
        self.assertLess(parsed, Version("0.16"))

        from altbacken.core.state import AnnealingState
        from altbacken.external.acceptance import BoltzmannAcceptance
        from altbacken.external.annealing import SimpleSimulatedAnnealing
        from altbacken.external.temperature import ExponentialCooling

        from core.ioh_core.algorithms.altbacken.neighbourhood import (
            CatalogueResetNeighbourhood,
            sample_integer_vector,
        )

        rng = Random(0)
        lb, ub = [0, 0], [4, 4]
        neighbourhood = CatalogueResetNeighbourhood(lb, ub, rng)

        def fitness(vector):
            return float(sum(int(v) * int(v) for v in vector))

        initial = sample_integer_vector(lb, ub, rng)
        annealer = SimpleSimulatedAnnealing(
            fitness=fitness,
            neighbourhood=neighbourhood,
            temperature=ExponentialCooling(
                initial_temperature=10.0,
                cooling_rate=0.95,
            ),
            stop=25,
            acceptance=BoltzmannAcceptance(),
            random=rng.random,
        )
        best, value = annealer.simulate(initial)
        self.assertEqual(len(best), 2)
        self.assertTrue(all(isinstance(v, int) for v in best))
        self.assertIsInstance(value, float)
        # Neighbourhood protocol accepts AnnealingState.
        state = AnnealingState(0, 1.0, [1, 2], 5.0, [1, 2], 5.0)
        neighbour = neighbourhood(state)
        self.assertEqual(len(neighbour), 2)
        self.assertTrue(
            all(lo <= v <= hi for v, lo, hi in zip(neighbour, lb, ub))
        )


class TestCatalogueResetNeighbourhood(unittest.TestCase):
    """Bound-aware single-coordinate reset neighbourhood."""

    def test_neighbours_stay_in_bounds_and_change_one_coord(self) -> None:
        from core.ioh_core.algorithms.altbacken.neighbourhood import (
            CatalogueResetNeighbourhood,
        )
        from altbacken.core.state import AnnealingState

        rng = Random(7)
        neighbourhood = CatalogueResetNeighbourhood([0, 2], [3, 5], rng)
        current = [1, 4]
        for _ in range(40):
            state = AnnealingState(0, 1.0, current, 0.0, current, 0.0)
            neighbour = neighbourhood(state)
            self.assertEqual(len(neighbour), 2)
            self.assertTrue(0 <= neighbour[0] <= 3)
            self.assertTrue(2 <= neighbour[1] <= 5)
            diffs = sum(
                int(a != b) for a, b in zip(neighbour, current)
            )
            self.assertLessEqual(diffs, 1)
            current = neighbour

    def test_rejects_inverted_bounds(self) -> None:
        from core.ioh_core.algorithms.altbacken.neighbourhood import (
            CatalogueResetNeighbourhood,
        )

        with self.assertRaises(ValueError):
            CatalogueResetNeighbourhood([0, 3], [2, 1], Random(0))


class TestAltbackenSAAdapter(unittest.TestCase):
    """Registered ALTBACKEN_SA adapter behaviour."""

    token = "ALTBACKEN_SA"

    def test_registry_metadata(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        spec = get_algorithm_spec(self.token)
        self.assertEqual(spec.library, "altbacken")
        self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(spec.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            spec.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertEqual(spec.dependency_extra, "algorithms-altbacken")
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
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [3, 3])
        create_algorithm(
            self.token,
            {"max_evaluations": 10},
            random_seed=8,
        )(problem)
        self.assertEqual(len(problem.calls), 10)
        self.assertEqual(problem.state.evaluations, 10)

    def test_deterministic_with_seed(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem_a = FakeIOHProblem([0, 1], [3, 4])
        problem_b = FakeIOHProblem([0, 1], [3, 4])
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
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem_a = FakeIOHProblem([0, 1], [5, 6])
        problem_b = FakeIOHProblem([0, 1], [5, 6])
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

    def test_does_not_re_evaluate_best(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm
        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem

        problem = FakeIOHProblem([0, 0], [2, 2])
        algo = create_algorithm(
            self.token,
            {"max_evaluations": 10},
            random_seed=5,
        )
        algo(problem)
        call_count = len(problem.calls)
        self.assertEqual(problem.state.evaluations, call_count)
        self.assertLessEqual(call_count, 10)
        self.assertIn(problem.state.current_best.x, problem.calls)

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
            validate_algorithm_parameters,
        )

        with self.assertRaises(AlgorithmFactoryError):
            validate_algorithm_parameters(
                self.token,
                {"max_evaluations": 5, "cooling_rate": 0.5},
                random_seed=1,
            )
        with self.assertRaises(AlgorithmFactoryError):
            create_algorithm(
                self.token,
                {"max_evaluations": 5, "cooling_rate": 0.5},
                random_seed=1,
            )

    def test_constructor_ref_and_factory_seed(self) -> None:
        from core.ioh_core.algorithms.altbacken.factory import (
            build_altbacken_sa,
        )
        from core.ioh_core.algorithms.registry import get_algorithm_spec

        spec = get_algorithm_spec(self.token)
        self.assertEqual(
            spec.constructor_ref,
            "core.ioh_core.algorithms.altbacken.factory:build_altbacken_sa",
        )
        with self.assertRaises(ValueError):
            build_altbacken_sa({"max_evaluations": 5}, random_seed=None)

    def test_missing_dependency_uses_install_hint(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.altbacken.factory import (
            build_altbacken_sa,
        )

        with patch(
            "core.ioh_core.algorithms.altbacken.factory."
            "import_optional_dependency",
            side_effect=AlgorithmDependencyError(
                "ALTBACKEN_SA requires optional dependency "
                "'altbacken'; install with:\n"
                'pip install -e ".[algorithms-altbacken]" '
                "--ignore-requires-python"
            ),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                build_altbacken_sa(
                    {"max_evaluations": 5},
                    random_seed=0,
                )
        self.assertIn(
            'pip install -e ".[algorithms-altbacken]" '
            "--ignore-requires-python",
            str(ctx.exception),
        )


class _RecordingRandom(Random):
    """Random that records later neighbourhood and acceptance draws."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.events: list[tuple] = []

    def randint(self, a: int, b: int) -> int:
        value = super().randint(a, b)
        self.events.append(("randint", value))
        return value

    def randrange(self, *args: int) -> int:
        value = super().randrange(*args)
        self.events.append(("randrange", args, value))
        return value

    def choice(self, seq):
        value = super().choice(seq)
        self.events.append(("choice",))
        return value

    def random(self) -> float:
        value = super().random()
        self.events.append(("random", value))
        return value


class TestAltbackenSAWarmStart(unittest.TestCase):
    """Rank 1 is evaluated. The discarded draw keeps the cold random stream."""

    def test_later_draws_match_the_cold_stream(self) -> None:
        import json
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem
        from core.ioh_core.algorithms.altbacken import algorithms as sa_module
        from core.ioh_core.algorithms.altbacken.algorithms import (
            AltbackenSA,
            AltbackenSAWarmStart,
        )

        created: list[_RecordingRandom] = []

        class _CapturingRandom(_RecordingRandom):
            def __init__(self, seed: int) -> None:
                super().__init__(seed)
                created.append(self)

        vector = [2, 1, 0, 3, 1, 0, 4, 0, 1]
        lower = [0] * 9
        upper = [5] * 9

        class _FlatProblem(FakeIOHProblem):
            def __call__(self, x):
                candidate = [int(value) for value in x]
                self.calls.append(candidate)
                self.state.evaluations += 1
                return 1.0

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
            cold = _FlatProblem(lower, upper)
            warm = _FlatProblem(lower, upper)
            with patch.object(sa_module, "Random", _CapturingRandom):
                AltbackenSA(max_evaluations=4, seed=11)(cold)
                AltbackenSAWarmStart(4, 11, str(path))(warm)
        self.assertEqual(warm.calls[0], vector)
        self.assertNotEqual(warm.calls[0], cold.calls[0])
        self.assertEqual(len(created), 2)

        def _after_initial_sample(events: list[tuple]) -> list[tuple]:
            seen = 0
            for index, event in enumerate(events):
                if event[0] == "randint":
                    seen += 1
                    if seen == 9:
                        return events[index + 1 :]
            raise AssertionError("expected 9 initial randint draws")

        cold_later = _after_initial_sample(created[0].events)
        warm_later = _after_initial_sample(created[1].events)
        self.assertEqual(cold_later, warm_later)
        self.assertTrue(any(event[0] == "randrange" for event in cold_later))
        self.assertTrue(any(event[0] == "choice" for event in cold_later))
        self.assertTrue(any(event[0] == "random" for event in cold_later))

    def test_missing_file_and_short_budget_evaluate_nothing(self) -> None:
        import json
        from tempfile import TemporaryDirectory

        from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem
        from core.ioh_core.algorithms.altbacken.algorithms import (
            AltbackenSAWarmStart,
        )

        lower = [0] * 9
        upper = [5] * 9
        missing = FakeIOHProblem(lower, upper)
        with self.assertRaisesRegex(ValueError, "has no InitialDesign"):
            AltbackenSAWarmStart(1, 1, "missing-initial-design.json")(missing)
        self.assertEqual(missing.calls, [])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "problem_id": "L8100_B2",
                        "n": 1,
                        "vectors": [[1, 0, 0, 0, 0, 0, 0, 0, 0]],
                    }
                ),
                encoding="utf-8",
            )
            problem = FakeIOHProblem(lower, upper)
            with self.assertRaisesRegex(
                ValueError,
                "needs 1 initial evaluations but max_evaluations is 0",
            ):
                AltbackenSAWarmStart(0, 1, str(path))(problem)
        self.assertEqual(problem.calls, [])


if __name__ == "__main__":
    unittest.main()
