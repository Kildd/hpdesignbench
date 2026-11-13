"""
Optax discrete-gradient adapters: gradient, budget, and update contracts.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from _mains.Testing.helpers.fake_ioh_problem import (
    FakeBest,
    FakeIOHProblem,
    FakeState,
    assert_algorithm_conformance,
)
from core.ioh_core.algorithms.common import (
    RunController,
    continuous_to_integer_candidate,
)
from core.ioh_core.algorithms.optax.engine import compute_index_gradient


class ValueProblem:
    """
    Fake IOH problem with a caller-supplied objective.

    Attributes
    ----------
    calls : list of list of int
        Catalogue-index vectors passed to ``problem``.
    """

    def __init__(
        self,
        lower: Sequence[int],
        upper: Sequence[int],
        objective: Callable[[Sequence[int]], float],
    ) -> None:
        self.bounds = SimpleNamespace(lb=list(lower), ub=list(upper))
        self.meta_data = SimpleNamespace(n_variables=len(lower))
        self.state = FakeState()
        self.calls: list[list[int]] = []
        self.hdb_last_feasible = True
        self._objective = objective

    def __call__(self, x: Any) -> float:
        candidate = [int(value) for value in x]
        self.calls.append(candidate)
        self.state.evaluations += 1
        value = float(self._objective(candidate))
        if math.isfinite(value) and value < self.state.current_best.y:
            self.state.current_best = FakeBest(x=list(candidate), y=value)
        return value


def _controller(problem: Any, max_evaluations: int = 100) -> RunController:
    return RunController(problem, max_evaluations)


class TestHalfAwayRounding(unittest.TestCase):
    """Shared integer projection used by every Optax token."""

    def test_half_away_from_zero_examples(self) -> None:
        lower = [-10]
        upper = [10]
        expected = {
            2.49: 2,
            2.50: 3,
            2.51: 3,
            -2.49: -2,
            -2.50: -3,
        }
        for value, projected in expected.items():
            got = continuous_to_integer_candidate(
                [value],
                lower_bounds=lower,
                upper_bounds=upper,
            )
            self.assertEqual(got, [projected])


class TestIndexGradient(unittest.TestCase):
    """Finite-difference gradient on catalogue indices."""

    def test_interior_linear_partials(self) -> None:
        problem = ValueProblem(
            [0, 0, 0],
            [10, 10, 10],
            lambda x: 2 * x[0] - 3 * x[1] + 5 * x[2],
        )
        gradient = compute_index_gradient(
            problem,
            _controller(problem),
            [5, 5, 5],
            [0, 0, 0],
            [10, 10, 10],
        )
        self.assertEqual(gradient, [2.0, -3.0, 5.0])

    def test_lower_and_upper_boundaries_match_interior_slope(self) -> None:
        def objective(x: Sequence[int]) -> float:
            return 2.0 * x[0]

        lower_problem = ValueProblem([0], [5], objective)
        lower_gradient = compute_index_gradient(
            lower_problem,
            _controller(lower_problem),
            [0],
            [0],
            [5],
        )
        upper_problem = ValueProblem([0], [5], objective)
        upper_gradient = compute_index_gradient(
            upper_problem,
            _controller(upper_problem),
            [5],
            [0],
            [5],
        )
        self.assertEqual(lower_gradient, [2.0])
        self.assertEqual(upper_gradient, [2.0])

    def test_evaluation_order_is_center_then_minus_then_plus(self) -> None:
        problem = ValueProblem([0, 0], [10, 10], lambda _x: 0.0)
        compute_index_gradient(
            problem,
            _controller(problem),
            [5, 5],
            [0, 0],
            [10, 10],
        )
        self.assertEqual(
            problem.calls,
            [
                [5, 5],
                [4, 5],
                [6, 5],
                [5, 4],
                [5, 6],
            ],
        )

    def test_singleton_axis_has_zero_gradient_and_no_neighbour_call(self) -> None:
        problem = ValueProblem(
            [0, 0, 4],
            [10, 10, 4],
            lambda x: float(x[0] + x[1] + x[2]),
        )
        gradient = compute_index_gradient(
            problem,
            _controller(problem),
            [5, 5, 4],
            [0, 0, 4],
            [10, 10, 4],
        )
        self.assertEqual(gradient[2], 0.0)
        self.assertTrue(all(call[2] == 4 for call in problem.calls))
        self.assertNotIn([5, 5, 3], problem.calls)
        self.assertNotIn([5, 5, 5], problem.calls)

    def test_nonfinite_side_zeros_only_that_component(self) -> None:
        def objective(x: Sequence[int]) -> float:
            if x[0] == 6:
                return math.inf
            return 2 * x[0] - 3 * x[1] + 5 * x[2]

        problem = ValueProblem([0, 0, 0], [10, 10, 10], objective)
        gradient = compute_index_gradient(
            problem,
            _controller(problem),
            [5, 5, 5],
            [0, 0, 0],
            [10, 10, 10],
        )
        self.assertEqual(gradient[0], 0.0)
        self.assertEqual(gradient[1], -3.0)
        self.assertEqual(gradient[2], 5.0)

    def test_nonfinite_boundary_neighbour_zeros_that_component(self) -> None:
        problem = ValueProblem(
            [0],
            [5],
            lambda x: math.inf if x[0] == 1 else float(x[0]),
        )
        gradient = compute_index_gradient(
            problem,
            _controller(problem),
            [0],
            [0],
            [5],
        )
        self.assertEqual(gradient, [0.0])

    def test_repeated_gradient_re_evaluates_every_probe(self) -> None:
        problem = ValueProblem([0] * 9, [10] * 9, lambda _x: 0.0)
        point = [5] * 9
        bounds = [0] * 9
        upper = [10] * 9
        compute_index_gradient(problem, _controller(problem), point, bounds, upper)
        compute_index_gradient(problem, _controller(problem), point, bounds, upper)
        self.assertEqual(len(problem.calls), 38)


class _CountingOptimizer:
    """Optax-like transform that records how often ``update`` runs."""

    def __init__(self) -> None:
        self.update_count = 0

    def build(self, optax_module: Any, jnp_module: Any) -> Any:
        base = optax_module.adam(learning_rate=0.3, b1=0.9, b2=0.999, eps=1e-8)

        def update(grads: Any, state: Any, params: Any = None, **kwargs: Any) -> Any:
            self.update_count += 1
            return base.update(grads, state, params, **kwargs)

        return SimpleNamespace(init=base.init, update=update)


class TestOptaxLoopControls(unittest.TestCase):
    """Budget, target, clipping, and continuous accumulation."""

    def test_budget_stops_inside_the_gradient_without_an_update(self) -> None:
        from core.ioh_core.algorithms.optax.adam import OptaxAdam

        counter = _CountingOptimizer()

        class BudgetAdam(OptaxAdam):
            def initial_continuous_state(self, lower_bounds, upper_bounds):
                return [5.0, 5.0]

            def build_optimizer(self, optax_module, jnp_module):
                return counter.build(optax_module, jnp_module)

        problem = ValueProblem([0, 0], [10, 10], lambda _x: 1.0)
        controller = RunController(problem, 3)
        BudgetAdam(max_evaluations=3, seed=1)(problem, run_controller=controller)
        self.assertEqual(
            problem.calls,
            [[5, 5], [4, 5], [6, 5]],
        )
        self.assertEqual(counter.update_count, 0)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")
        self.assertEqual(problem.state.evaluations, 3)

    def test_target_hit_on_a_neighbour_skips_the_update(self) -> None:
        from core.ioh_core.algorithms.optax.adam import OptaxAdam

        counter = _CountingOptimizer()

        class TargetAdam(OptaxAdam):
            def initial_continuous_state(self, lower_bounds, upper_bounds):
                return [5.0, 5.0]

            def build_optimizer(self, optax_module, jnp_module):
                return counter.build(optax_module, jnp_module)

        def objective(x: Sequence[int]) -> float:
            if x == [6, 5]:
                return 0.0
            return 5.0

        problem = ValueProblem([0, 0], [10, 10], objective)
        controller = RunController(
            problem,
            20,
            mode="fixed_target",
            target_threshold=0.0,
            target_reference="all",
        )
        TargetAdam(max_evaluations=20, seed=1)(problem, run_controller=controller)
        self.assertEqual(problem.calls, [[5, 5], [4, 5], [6, 5]])
        self.assertEqual(controller.evaluations_to_target, 3)
        self.assertEqual(controller.stop_reason(), "target_hit")
        self.assertEqual(counter.update_count, 0)

    def test_continuous_fraction_is_kept_until_it_crosses_an_integer(self) -> None:
        from core.ioh_core.algorithms.optax.adam import OptaxAdam

        class DriftAdam(OptaxAdam):
            def initial_continuous_state(self, lower_bounds, upper_bounds):
                return [3.0]

            def build_optimizer(self, optax_module, jnp_module):
                def init(params: Any) -> tuple[()]:
                    return ()

                def update(
                    grads: Any,
                    state: Any,
                    params: Any = None,
                    **kwargs: Any,
                ) -> Any:
                    return jnp_module.full_like(params, 0.2), state

                return SimpleNamespace(init=init, update=update)

        problem = ValueProblem([0], [10], lambda _x: 0.0)
        DriftAdam(max_evaluations=12, seed=1)(problem)
        centers = [problem.calls[index] for index in (0, 3, 6, 9)]
        self.assertEqual(centers, [[3], [3], [3], [4]])

    def test_box_clip_keeps_the_optimizer_state(self) -> None:
        from core.ioh_core.algorithms.optax.adam import OptaxAdam

        class ClipAdam(OptaxAdam):
            def initial_continuous_state(self, lower_bounds, upper_bounds):
                return [4.8]

            def build_optimizer(self, optax_module, jnp_module):
                def init(params: Any) -> dict[str, Any]:
                    return {"n": jnp_module.array(0, dtype=jnp_module.int32)}

                def update(
                    grads: Any,
                    state: Any,
                    params: Any = None,
                    **kwargs: Any,
                ) -> Any:
                    count = state["n"] + 1
                    step = jnp_module.where(count == 1, 2.0, -0.6)
                    return (
                        jnp_module.full_like(params, step),
                        {"n": count},
                    )

                return SimpleNamespace(init=init, update=update)

        problem = ValueProblem([0], [5], lambda _x: 0.0)
        ClipAdam(max_evaluations=6, seed=1)(problem)
        centers = [problem.calls[index] for index in (0, 2, 4)]
        self.assertEqual(centers, [[5], [5], [4]])

    def test_nonfinite_update_raises_and_restores_x64(self) -> None:
        import jax

        from core.ioh_core.algorithms.optax.adam import OptaxAdam

        class NanAdam(OptaxAdam):
            def initial_continuous_state(self, lower_bounds, upper_bounds):
                return [1.0]

            def build_optimizer(self, optax_module, jnp_module):
                def init(params: Any) -> tuple[()]:
                    return ()

                def update(
                    grads: Any,
                    state: Any,
                    params: Any = None,
                    **kwargs: Any,
                ) -> Any:
                    return jnp_module.full_like(params, jnp_module.nan), state

                return SimpleNamespace(init=init, update=update)

        jax.config.update("jax_enable_x64", False)
        problem = ValueProblem([0], [4], lambda x: float(x[0]))
        with self.assertRaises(RuntimeError):
            NanAdam(max_evaluations=8, seed=1)(problem)
        self.assertFalse(bool(jax.config.jax_enable_x64))

    def test_x64_flag_is_restored(self) -> None:
        import jax

        from core.ioh_core.algorithms.optax.adam import OptaxAdam

        class SeenAdam(OptaxAdam):
            dtype_name = ""

            def build_optimizer(self, optax_module, jnp_module):
                base = super().build_optimizer(optax_module, jnp_module)

                def init(params: Any) -> Any:
                    SeenAdam.dtype_name = str(params.dtype)
                    return base.init(params)

                return SimpleNamespace(init=init, update=base.update)

        jax.config.update("jax_enable_x64", False)
        problem = ValueProblem([0], [3], lambda x: float(x[0]))
        SeenAdam(max_evaluations=1, seed=2)(problem)
        self.assertIn("64", SeenAdam.dtype_name)
        self.assertFalse(bool(jax.config.jax_enable_x64))

        jax.config.update("jax_enable_x64", True)
        try:
            SeenAdam(max_evaluations=1, seed=2)(
                ValueProblem([0], [3], lambda x: float(x[0]))
            )
            self.assertTrue(bool(jax.config.jax_enable_x64))
        finally:
            jax.config.update("jax_enable_x64", False)


class TestOptaxUpdates(unittest.TestCase):
    """Frozen Optax transforms, independent of the structural analysis."""

    def _under_x64(self, callback: Callable[[Any, Any], None]) -> None:
        import jax
        import jax.numpy as jnp
        import optax

        previous = bool(jax.config.jax_enable_x64)
        jax.config.update("jax_enable_x64", True)
        try:
            callback(optax, jnp)
        finally:
            jax.config.update("jax_enable_x64", previous)

    def test_adam_first_step_descends(self) -> None:
        from core.ioh_core.algorithms.optax.adam import OptaxAdam

        def check(optax_module: Any, jnp_module: Any) -> None:
            optimizer = OptaxAdam(max_evaluations=1, seed=0).build_optimizer(
                optax_module,
                jnp_module,
            )
            z = jnp_module.zeros((3,), dtype=jnp_module.float64)
            state = optimizer.init(z)
            updates, _state = optimizer.update(
                jnp_module.asarray([1.0, -1.0, 0.0], dtype=jnp_module.float64),
                state,
                z,
            )
            values = [float(item) for item in updates.tolist()]
            self.assertAlmostEqual(values[0], -0.3, places=4)
            self.assertAlmostEqual(values[1], 0.3, places=4)
            self.assertAlmostEqual(values[2], 0.0, places=8)

        self._under_x64(check)

    def test_rprop_initial_step_then_grows(self) -> None:
        from core.ioh_core.algorithms.optax.rprop import OptaxRProp

        def check(optax_module: Any, jnp_module: Any) -> None:
            optimizer = OptaxRProp(max_evaluations=1, seed=0).build_optimizer(
                optax_module,
                jnp_module,
            )
            z = jnp_module.zeros((3,), dtype=jnp_module.float64)
            state = optimizer.init(z)
            gradient = jnp_module.asarray(
                [1.0, -1.0, 0.0],
                dtype=jnp_module.float64,
            )
            first, state = optimizer.update(gradient, state, z)
            second, state = optimizer.update(gradient, state, z)
            third, _state = optimizer.update(gradient, state, z)
            first_values = [float(item) for item in first.tolist()]
            second_values = [float(item) for item in second.tolist()]
            third_values = [float(item) for item in third.tolist()]
            # Optax stores the step and applies it on the next update.
            self.assertAlmostEqual(first_values[0], 0.0, places=8)
            self.assertAlmostEqual(second_values[0], -0.1, places=6)
            self.assertAlmostEqual(second_values[1], 0.1, places=6)
            self.assertAlmostEqual(second_values[2], 0.0, places=8)
            self.assertAlmostEqual(third_values[0], -0.12, places=6)

        self._under_x64(check)

    def test_sgd_sign_and_first_nesterov_step(self) -> None:
        from core.ioh_core.algorithms.optax.sgd_nesterov import OptaxSGDNesterov

        raw = [100.0, -0.0001, 0.0]
        signed = OptaxSGDNesterov(max_evaluations=1, seed=0).preprocess_gradient(raw)
        self.assertEqual(signed, [1.0, -1.0, 0.0])

        def check(optax_module: Any, jnp_module: Any) -> None:
            optimizer = OptaxSGDNesterov(
                max_evaluations=1,
                seed=0,
            ).build_optimizer(optax_module, jnp_module)
            z = jnp_module.zeros((3,), dtype=jnp_module.float64)
            state = optimizer.init(z)
            updates, _state = optimizer.update(
                jnp_module.asarray(signed, dtype=jnp_module.float64),
                state,
                z,
            )
            values = [float(item) for item in updates.tolist()]
            self.assertAlmostEqual(values[0], -0.057, places=6)
            self.assertAlmostEqual(values[1], 0.057, places=6)
            self.assertAlmostEqual(values[2], 0.0, places=8)

        self._under_x64(check)


class TestOptaxRegistryAndRuns(unittest.TestCase):
    """Registry metadata, reproducibility, and the hard evaluation budget."""

    def test_package_import_is_lazy(self) -> None:
        import subprocess

        code = (
            "import importlib, sys\n"
            "import core.ioh_core.algorithms.registry as registry\n"
            "mod = importlib.import_module('core.ioh_core.algorithms.optax')\n"
            "assert mod.__all__ == ()\n"
            "assert 'OPTAX_ADAM' in registry.ALGORITHM_REGISTRY.names()\n"
            "assert 'optax' not in sys.modules\n"
            "assert 'jax' not in sys.modules\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stderr or completed.stdout,
        )

    def test_registry_metadata_freezes_max_evaluations_only(self) -> None:
        from core.ioh_core.algorithms.registry import (
            BudgetSemantics,
            CandidateMode,
            RegistryError,
            SeedPolicy,
            get_algorithm_spec,
            normalize_algorithm_parameters,
        )

        expected = {
            "OPTAX_ADAM": "build_optax_adam",
            "OPTAX_RPROP": "build_optax_rprop",
            "OPTAX_SGD_NESTEROV": "build_optax_sgd_nesterov",
        }
        for name, builder in expected.items():
            spec = get_algorithm_spec(name)
            self.assertEqual(spec.library, "optax")
            self.assertEqual(spec.seed_policy, SeedPolicy.REQUIRED)
            self.assertEqual(spec.candidate_mode, CandidateMode.CONTINUOUS_ROUNDED)
            self.assertEqual(
                spec.budget_semantics,
                BudgetSemantics.HARD_IOH_EVALUATIONS,
            )
            self.assertEqual(spec.dependency_extra, "algorithms-optax")
            self.assertFalse(spec.batch_evaluations)
            self.assertEqual(
                [parameter.name for parameter in spec.parameters],
                ["max_evaluations"],
            )
            self.assertTrue(spec.constructor_ref.endswith(builder))
            self.assertNotIn("analytic", spec.description.lower())
            self.assertNotIn("automatic differentiation", spec.description.lower())
            with self.assertRaises(RegistryError):
                normalize_algorithm_parameters(
                    name,
                    {"max_evaluations": 8, "learning_rate": 0.1},
                )

    def test_missing_extra_names_algorithms_optax(self) -> None:
        from unittest.mock import patch

        from core.ioh_core.algorithms.errors import AlgorithmDependencyError
        from core.ioh_core.algorithms.optax.factory import (
            build_optax_adam,
            build_optax_rprop,
            build_optax_sgd_nesterov,
        )

        builders = (
            build_optax_adam,
            build_optax_rprop,
            build_optax_sgd_nesterov,
        )
        with patch(
            "core.ioh_core.algorithms.optax.factory.import_optional_dependency",
            side_effect=AlgorithmDependencyError(
                "OPTAX_ADAM requires optional dependency 'optax'; "
                "install with:\n"
                'pip install -e ".[algorithms-optax]"'
            ),
        ):
            for builder in builders:
                with self.assertRaises(AlgorithmDependencyError) as ctx:
                    builder({"max_evaluations": 4}, random_seed=1)
                self.assertIn(
                    'pip install -e ".[algorithms-optax]"',
                    str(ctx.exception),
                )

    def test_same_seed_reproduces_the_call_sequence(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        first = ValueProblem([0, 0], [6, 6], lambda x: float(x[0] + 2 * x[1]))
        second = ValueProblem([0, 0], [6, 6], lambda x: float(x[0] + 2 * x[1]))
        create_algorithm("OPTAX_ADAM", {"max_evaluations": 20}, random_seed=7)(first)
        create_algorithm("OPTAX_ADAM", {"max_evaluations": 20}, random_seed=7)(second)
        self.assertEqual(first.calls, second.calls)
        self.assertEqual(len(first.calls), 20)

    def test_hard_budget_is_exact(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        problem = ValueProblem([0, 1], [8, 9], lambda x: float(sum(x)))
        controller = RunController(problem, 64)
        create_algorithm(
            "OPTAX_RPROP",
            {"max_evaluations": 64},
            random_seed=3,
        )(problem, run_controller=controller)
        self.assertEqual(problem.state.evaluations, 64)
        self.assertEqual(controller.stop_reason(), "budget_exhausted")

    def test_task_seeds_differ_by_algorithm_name(self) -> None:
        from core.ioh_core.workflows.optimization_runs.planner import (
            derive_random_seed,
        )

        seeds = [
            derive_random_seed(
                seed_base=42,
                slab_type="hp_slab",
                problem_id="L6750_B2",
                algorithm_name=name,
                repetition=0,
            )
            for name in ("OPTAX_ADAM", "OPTAX_RPROP", "OPTAX_SGD_NESTEROV")
        ]
        self.assertEqual(len(set(seeds)), 3)

    def test_conformance_for_each_token(self) -> None:
        from core.ioh_core.algorithm_factory import create_algorithm

        for name in ("OPTAX_ADAM", "OPTAX_RPROP", "OPTAX_SGD_NESTEROV"):

            def build(seed: int | None, token: str = name) -> Any:
                return create_algorithm(
                    token,
                    {"max_evaluations": 12},
                    random_seed=seed,
                )

            assert_algorithm_conformance(
                self,
                algorithm_name=name,
                build_algorithm=build,
                seed=11,
                second_seed=29,
            )


class TestOptaxWarmStart(unittest.TestCase):
    """Each Optax warm start evaluates rank 1 first. Cold starts stay seeded."""

    def test_cold_adam_still_draws_a_seeded_start(self) -> None:
        from core.ioh_core.algorithms.common import (
            continuous_to_integer_candidate,
            seeded_uniform_continuous_start,
        )
        from core.ioh_core.algorithms.optax.adam import OptaxAdam

        lower = [0, 0]
        upper = [4, 4]
        problem = ValueProblem(lower, upper, lambda _x: 1.0)
        OptaxAdam(max_evaluations=1, seed=3)(problem)
        expected = continuous_to_integer_candidate(
            seeded_uniform_continuous_start(lower, upper, 3),
            lower_bounds=lower,
            upper_bounds=upper,
        )
        self.assertEqual(problem.calls[0], expected)

    def test_rank1_missing_file_and_short_budget(self) -> None:
        import json
        from tempfile import TemporaryDirectory

        from core.ioh_core.algorithms.optax.engine import (
            OptaxAdamWarmStart,
            OptaxRPropWarmStart,
            OptaxSGDNesterovWarmStart,
        )

        vector = [1, 0, 0, 0, 0, 0, 0, 0, 2]
        lower = [0] * 9
        upper = [4] * 9
        classes = (
            OptaxAdamWarmStart,
            OptaxRPropWarmStart,
            OptaxSGDNesterovWarmStart,
        )
        for cls in classes:
            with self.subTest(token=cls.name):
                missing = ValueProblem(lower, upper, lambda _x: 1.0)
                with self.assertRaisesRegex(ValueError, "has no InitialDesign"):
                    cls(1, "missing-initial-design.json")(missing)
                self.assertEqual(missing.calls, [])
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
            for cls in classes:
                with self.subTest(token=cls.name, case="rank1"):
                    problem = ValueProblem(lower, upper, lambda _x: 1.0)
                    cls(1, str(path))(problem)
                    self.assertEqual(problem.calls[0], vector)
                with self.subTest(token=cls.name, case="budget"):
                    problem = ValueProblem(lower, upper, lambda _x: 1.0)
                    with self.assertRaisesRegex(
                        ValueError,
                        "needs 1 initial evaluations but max_evaluations is 0",
                    ):
                        cls(0, str(path))(problem)
                    self.assertEqual(problem.calls, [])


if __name__ == "__main__":
    unittest.main()
