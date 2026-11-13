"""
Phase A5 gate: optimisation task planning order and deterministic seeds.
"""
from __future__ import annotations

import textwrap
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.config import load_experiment_config
from core.execution_core.models import format_task_id
from core.ioh_core.algorithm_factory import AlgorithmFactoryError
from core.ioh_core.workflows.optimization_runs.planner import (
    OPTIMIZATION_TASK_KIND,
    apply_warm_start_parameters,
    derive_random_seed,
    natural_sort_key,
    plan_optimization_specs,
    plan_optimization_tasks,
)
from core.ioh_core.workflows.predesign_paths import (
    predesign_initial_design_deb_path,
    predesign_initial_design_path,
)


def _write_config(
    path: Path,
    *,
    problem_ids: str = '["2", "10", "1"]',
    algorithms: str | None = None,
    repetitions: str | None = None,
    termination: str = "",
) -> Path:
    """
    Write a local optimisation TOML config for planner tests.

    Parameters
    ----------
    path : Path
        Destination TOML path.
    problem_ids : str, optional
        Literal TOML array text for ``problem_ids``.
    algorithms : str or None, optional
        Optional replacement for the algorithms section.
    repetitions : str or None, optional
        Optional replacement for the repetitions section.
    termination : str, optional
        Optional ``[experiment.termination]`` TOML fragment.

    Returns
    -------
    Path
        The written config path.
    """
    if algorithms is None:
        algorithms = textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "RANDOM_SEARCH"
            max_evaluations = 10

            [[experiment.algorithms]]
            name = "NLOPT_DIRECT_L"
            max_evaluations = 20
            """
        )
    if repetitions is None:
        repetitions = textwrap.dedent(
            """\
            [experiment.repetitions]
            count = 2
            seed_base = 42
            """
        )
    parts = [
        "schema_version = 1",
        "",
        "[experiment]",
        'id = "planner_smoke"',
        'workflow = "optimization_runs"',
        'slab_type = "hp_slab"',
        f"problem_ids = {problem_ids}",
        'output_root = "experiments"',
        "",
        algorithms,
        repetitions,
    ]
    if termination.strip():
        parts.append(termination.strip())
    parts.append(
        textwrap.dedent(
            """\
            [execution]
            backend = "local"
            max_parallel_tasks = 1
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        )
    )
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


class TestOptimizationPlanner(unittest.TestCase):
    """Task expansion, order, and seed stability."""

    def test_natural_sort_orders_problem_ids(self) -> None:
        self.assertEqual(
            sorted(["10", "2", "1"], key=natural_sort_key),
            ["1", "2", "10"],
        )

    def test_task_count_order_and_payloads(self) -> None:
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(Path(tmp) / "exp.toml")
            )
            specs = plan_optimization_specs(config)
            tasks = plan_optimization_tasks(config)

            # 3 problems × 2 algorithms × 2 repetitions
            self.assertEqual(len(specs), 12)
            self.assertEqual(len(tasks), 12)

            expected_order = []
            for problem_id in ("1", "2", "10"):
                for algorithm_name in ("RANDOM_SEARCH", "NLOPT_DIRECT_L"):
                    for repetition in (0, 1):
                        expected_order.append(
                            (problem_id, algorithm_name, repetition)
                        )
            actual_order = [
                (s.problem_id, s.algorithm_name, s.repetition) for s in specs
            ]
            self.assertEqual(actual_order, expected_order)

            for index, (spec, task) in enumerate(zip(specs, tasks)):
                self.assertEqual(task.task_index, index)
                self.assertEqual(task.task_id, format_task_id(index))
                self.assertEqual(task.kind, OPTIMIZATION_TASK_KIND)
                self.assertEqual(task.payload, spec.to_payload())
                if spec.algorithm_name == "RANDOM_SEARCH":
                    self.assertIsInstance(spec.random_seed, int)
                    self.assertGreaterEqual(spec.random_seed, 0)
                    self.assertLessEqual(spec.random_seed, 0xFFFFFFFF)
                else:
                    self.assertIsNone(spec.random_seed)

    def test_seeds_are_stable_and_distinct_across_repetitions(self) -> None:
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["1"]',
                    algorithms=textwrap.dedent(
                        """\
                        [[experiment.algorithms]]
                        name = "RANDOM_SEARCH"
                        max_evaluations = 5
                        """
                    ),
                    repetitions=textwrap.dedent(
                        """\
                        [experiment.repetitions]
                        count = 3
                        seed_base = 42
                        """
                    ),
                )
            )
            first = plan_optimization_specs(config)
            second = plan_optimization_specs(config)
            self.assertEqual(
                [s.random_seed for s in first],
                [s.random_seed for s in second],
            )
            seeds = [s.random_seed for s in first]
            self.assertEqual(len(seeds), len(set(seeds)))
            self.assertEqual(
                first[0].random_seed,
                derive_random_seed(
                    seed_base=42,
                    slab_type="hp_slab",
                    problem_id="1",
                    algorithm_name="RANDOM_SEARCH",
                    repetition=0,
                ),
            )

    def test_planner_uses_registry_seed_policy(self) -> None:
        from core.ioh_core.algorithms.registry import SeedPolicy, get_algorithm_spec

        self.assertEqual(
            get_algorithm_spec("RANDOM_SEARCH").seed_policy,
            SeedPolicy.REQUIRED,
        )
        self.assertEqual(
            get_algorithm_spec("NLOPT_DIRECT_L").seed_policy,
            SeedPolicy.FORBIDDEN,
        )
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(_write_config(Path(tmp) / "exp.toml"))
            specs = plan_optimization_specs(config)
            for spec in specs:
                if spec.algorithm_name == "RANDOM_SEARCH":
                    self.assertIsInstance(spec.random_seed, int)
                else:
                    self.assertIsNone(spec.random_seed)

    def test_planner_materializes_registry_defaults(self) -> None:
        from types import MappingProxyType
        from unittest.mock import patch

        from core.execution_core.config import AlgorithmConfig
        from core.ioh_core.algorithms.registry import (
            AlgorithmSpec,
            BudgetSemantics,
            CandidateMode,
            ParameterSpec,
            SeedPolicy,
            create_registry,
        )
        from core.ioh_core.workflows.optimization_runs.planner import (
            algorithm_parameters_for,
        )

        toy_registry = create_registry(
            (
                AlgorithmSpec(
                    name="TOY_DEFAULTS",
                    library="native",
                    constructor_ref=(
                        "_mains.Testing.test_cursor_algorithm_registry:"
                        "build_toy_algorithm"
                    ),
                    parameters=(
                        ParameterSpec(
                            name="max_evaluations",
                            types=(int,),
                            required=True,
                            minimum=1,
                        ),
                        ParameterSpec(
                            name="mode",
                            types=(str,),
                            required=False,
                            default="fast",
                        ),
                    ),
                    seed_policy=SeedPolicy.REQUIRED,
                    dependency_extra=None,
                    candidate_mode=CandidateMode.INTEGER,
                    budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
                    description="toy algorithm with a defaulted mode parameter",
                ),
            )
        )
        with patch(
            "core.ioh_core.algorithms.registry.ALGORITHM_REGISTRY",
            toy_registry,
        ):
            params = algorithm_parameters_for(
                AlgorithmConfig(
                    name="TOY_DEFAULTS",
                    max_evaluations=7,
                    parameters={},
                )
            )
            self.assertEqual(params, {"max_evaluations": 7, "mode": "fast"})
            self.assertIsInstance(params, dict)
            self.assertNotIsInstance(params, MappingProxyType)

            with TemporaryDirectory() as tmp:
                config = load_experiment_config(
                    _write_config(
                        Path(tmp) / "exp.toml",
                        problem_ids='["1"]',
                        algorithms=textwrap.dedent(
                            """\
                            [[experiment.algorithms]]
                            name = "TOY_DEFAULTS"
                            max_evaluations = 7
                            """
                        ),
                        repetitions=textwrap.dedent(
                            """\
                            [experiment.repetitions]
                            count = 1
                            seed_base = 1
                            """
                        ),
                    )
                )
                specs = plan_optimization_specs(config)
            self.assertEqual(len(specs), 1)
            self.assertEqual(
                specs[0].algorithm_parameters,
                {"max_evaluations": 7, "mode": "fast"},
            )
            self.assertIsInstance(specs[0].algorithm_parameters, dict)
    def test_planner_rejects_budget_override(self) -> None:
        from core.execution_core.config import AlgorithmConfig
        from core.ioh_core.algorithm_factory import AlgorithmFactoryError
        from core.ioh_core.workflows.optimization_runs.planner import (
            algorithm_parameters_for,
        )

        with self.assertRaises(AlgorithmFactoryError) as ctx:
            algorithm_parameters_for(
                AlgorithmConfig(
                    name="RANDOM_SEARCH",
                    max_evaluations=5,
                    parameters={"max_evaluations": 99},
                )
            )
        self.assertIn("max_evaluations", str(ctx.exception))
        self.assertIn("must not include", str(ctx.exception))

    def test_planner_payload_is_plain_json_data(self) -> None:
        import json
        from types import MappingProxyType

        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["1"]',
                    algorithms=textwrap.dedent(
                        """\
                        [[experiment.algorithms]]
                        name = "RANDOM_SEARCH"
                        max_evaluations = 5
                        """
                    ),
                    repetitions=textwrap.dedent(
                        """\
                        [experiment.repetitions]
                        count = 1
                        seed_base = 1
                        """
                    ),
                )
            )
            tasks = plan_optimization_tasks(config)
            payload = tasks[0].payload
            self.assertIsInstance(payload, dict)
            self.assertIsInstance(payload["algorithm_parameters"], dict)
            self.assertNotIsInstance(payload["algorithm_parameters"], MappingProxyType)
            json.dumps(payload)

    def test_dflint_warm_start_uses_repetition_as_start_index(self) -> None:
        algorithms = textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "DFLINT_BBOA_WARM_START"
            max_evaluations = 10
            """
        )
        repetitions = textwrap.dedent(
            """\
            [experiment.repetitions]
            count = 1
            seed_base = 42
            """
        )
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["2"]',
                    algorithms=algorithms,
                    repetitions=repetitions,
                )
            )
            specs = plan_optimization_specs(config)
            self.assertEqual(len(specs), 1)
            spec = specs[0]
            self.assertEqual(spec.repetition, 0)
            self.assertIsNone(spec.random_seed)
            self.assertNotIn("start_index", spec.algorithm_parameters)
            self.assertEqual(
                spec.algorithm_parameters["initial_design_path"],
                str(predesign_initial_design_deb_path("experiments", "2")),
            )

    def test_dflint_warm_start_rejects_more_runs_than_design(self) -> None:
        from core.ioh_core.algorithm_factory import AlgorithmFactoryError

        algorithms = textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "DFLINT_BBOA_WARM_START"
            max_evaluations = 10
            """
        )
        repetitions = textwrap.dedent(
            """\
            [experiment.repetitions]
            count = 2
            seed_base = 42
            """
        )
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["2"]',
                    algorithms=algorithms,
                    repetitions=repetitions,
                )
            )
            with self.assertRaisesRegex(AlgorithmFactoryError, "repetitions.count"):
                plan_optimization_specs(config)

    def test_algorithm_repetition_override_combines_seeded_and_seedless(self) -> None:
        algorithms = textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "RANDOM_SEARCH"
            max_evaluations = 10

            [[experiment.algorithms]]
            name = "PYNOMAD_MADS_PB_WARM_START"
            max_evaluations = 10

            [[experiment.algorithms]]
            name = "DFLINT_BBOA_WARM_START"
            max_evaluations = 10
            repetitions = 1
            """
        )
        repetitions = textwrap.dedent(
            """\
            [experiment.repetitions]
            count = 2
            seed_base = 42
            """
        )
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["1", "2"]',
                    algorithms=algorithms,
                    repetitions=repetitions,
                    termination=textwrap.dedent(
                        """\
                        [experiment.termination]
                        mode = "fixed_target"
                        allowed_deviation_percent = 5.0
                        target_reference = "all"
                        [experiment.termination.global_optima]
                        1 = 100.0
                        2 = 200.0
                        """
                    ),
                )
            )
            specs = plan_optimization_specs(config)

        self.assertEqual(len(specs), 10)
        for problem_id in ("1", "2"):
            seeded = [
                spec
                for spec in specs
                if spec.problem_id == problem_id
                and spec.algorithm_name == "RANDOM_SEARCH"
            ]
            seeded_warm = [
                spec
                for spec in specs
                if spec.problem_id == problem_id
                and spec.algorithm_name == "PYNOMAD_MADS_PB_WARM_START"
            ]
            seedless = [
                spec
                for spec in specs
                if spec.problem_id == problem_id
                and spec.algorithm_name == "DFLINT_BBOA_WARM_START"
            ]
            self.assertEqual([spec.repetition for spec in seeded], [0, 1])
            self.assertTrue(all(spec.random_seed is not None for spec in seeded))
            self.assertEqual(
                [spec.repetition for spec in seeded_warm],
                [0, 1],
            )
            self.assertTrue(
                all(spec.random_seed is not None for spec in seeded_warm)
            )
            self.assertTrue(
                all(
                    spec.algorithm_parameters["initial_design_path"]
                    == str(
                        predesign_initial_design_deb_path(
                            "experiments", problem_id
                        )
                    )
                    for spec in seeded_warm
                )
            )
            self.assertEqual(len(seedless), 1)
            self.assertEqual(seedless[0].repetition, 0)
            self.assertIsNone(seedless[0].random_seed)
            self.assertEqual(
                seedless[0].algorithm_parameters["initial_design_path"],
                str(predesign_initial_design_deb_path("experiments", problem_id)),
            )
            expected_optimum = 100.0 if problem_id == "1" else 200.0
            for spec in (*seeded, *seeded_warm, *seedless):
                self.assertEqual(spec.global_optimum, expected_optimum)
                self.assertEqual(
                    spec.target_threshold,
                    expected_optimum * 1.05,
                )
                self.assertNotIn("global_optima", spec.to_payload())

        from core.ioh_core.workflows.optimization_runs.planner import (
            plan_optimization_tasks,
        )
        from core.ioh_core.workflows.registry import get_workflow

        tasks = plan_optimization_tasks(config)
        self.assertEqual(
            [task.task_index for task in tasks],
            list(range(10)),
        )
        entry = get_workflow("optimization_runs")
        assert entry.validator is not None
        entry.validator(config, tasks)

    def test_dflint_warm_start_rejects_toml_start_index(self) -> None:
        from core.ioh_core.algorithm_factory import AlgorithmFactoryError

        algorithms = textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "DFLINT_BBOA_WARM_START"
            max_evaluations = 10

            [experiment.algorithms.parameters]
            start_index = 3
            """
        )
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["2"]',
                    algorithms=algorithms,
                )
            )
            with self.assertRaisesRegex(AlgorithmFactoryError, "start_index"):
                plan_optimization_specs(config)

    def test_mads_warm_start_keeps_seed_and_sets_start_index(self) -> None:
        algorithms = textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "PYNOMAD_MADS_PB_WARM_START"
            max_evaluations = 10
            """
        )
        repetitions = textwrap.dedent(
            """\
            [experiment.repetitions]
            count = 3
            seed_base = 42
            """
        )
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["2"]',
                    algorithms=algorithms,
                    repetitions=repetitions,
                )
            )
            specs = plan_optimization_specs(config)
            self.assertEqual(len(specs), 3)
            seeds = {spec.random_seed for spec in specs}
            self.assertEqual(len(seeds), 3)
            for index, spec in enumerate(specs):
                self.assertIsInstance(spec.random_seed, int)
                self.assertNotIn("start_index", spec.algorithm_parameters)
                self.assertEqual(
                    spec.algorithm_parameters["initial_design_path"],
                    str(predesign_initial_design_deb_path("experiments", "2")),
                )


class TestOptimizationTerminationSealing(unittest.TestCase):
    """C4.2: termination fields are sealed into every task payload."""

    def test_fixed_budget_default_payload_round_trip(self) -> None:
        from core.ioh_core.workflows.optimization_runs.models import (
            OptimizationTaskSpec,
        )

        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["1"]',
                    algorithms=textwrap.dedent(
                        """\
                        [[experiment.algorithms]]
                        name = "RANDOM_SEARCH"
                        max_evaluations = 5
                        """
                    ),
                    repetitions=textwrap.dedent(
                        """\
                        [experiment.repetitions]
                        count = 1
                        seed_base = 1
                        """
                    ),
                )
            )
            specs = plan_optimization_specs(config)
            self.assertEqual(len(specs), 1)
            spec = specs[0]
            self.assertEqual(spec.termination_mode, "fixed_budget")
            self.assertIsNone(spec.target_reference)
            self.assertIsNone(spec.target_threshold)
            restored = OptimizationTaskSpec.from_payload(spec.to_payload())
            self.assertEqual(restored, spec)
            for key in (
                "termination_mode",
                "global_optimum",
                "allowed_deviation_percent",
                "target_reference",
                "target_threshold",
            ):
                self.assertNotIn(key, spec.algorithm_parameters)

    def test_fixed_target_resolved_per_problem(self) -> None:
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["1", "2"]',
                    algorithms=textwrap.dedent(
                        """\
                        [[experiment.algorithms]]
                        name = "RANDOM_SEARCH"
                        max_evaluations = 5
                        """
                    ),
                    repetitions=textwrap.dedent(
                        """\
                        [experiment.repetitions]
                        count = 1
                        seed_base = 1
                        """
                    ),
                    termination=textwrap.dedent(
                        """\
                        [experiment.termination]
                        mode = "fixed_target"
                        allowed_deviation_percent = 5.0
                        target_reference = "all"
                        [experiment.termination.global_optima]
                        1 = 100.0
                        2 = 200.0
                        """
                    ),
                )
            )
            specs = plan_optimization_specs(config)
            self.assertEqual(len(specs), 2)
            by_problem = {spec.problem_id: spec for spec in specs}
            self.assertEqual(by_problem["1"].global_optimum, 100.0)
            self.assertEqual(by_problem["1"].target_threshold, 105.0)
            self.assertEqual(by_problem["2"].global_optimum, 200.0)
            self.assertEqual(by_problem["2"].target_threshold, 210.0)
            for spec in specs:
                self.assertEqual(spec.termination_mode, "fixed_target")
                self.assertEqual(spec.allowed_deviation_percent, 5.0)
                self.assertEqual(spec.target_reference, "all")

    def test_old_payload_without_termination_is_fixed_budget(self) -> None:
        from core.ioh_core.workflows.optimization_runs.models import (
            OptimizationTaskSpec,
        )

        restored = OptimizationTaskSpec.from_payload(
            {
                "problem_id": "1",
                "slab_type": "hp_slab",
                "algorithm_name": "RANDOM_SEARCH",
                "algorithm_parameters": {"max_evaluations": 5},
                "repetition": 0,
                "random_seed": 1,
            }
        )
        self.assertEqual(restored.termination_mode, "fixed_budget")
        self.assertIsNone(restored.target_threshold)

    def test_partial_and_altered_threshold_rejected(self) -> None:
        from core.ioh_core.workflows.optimization_runs.models import (
            OptimizationTaskSpec,
        )

        with self.assertRaises(ValueError):
            OptimizationTaskSpec.from_payload(
                {
                    "problem_id": "1",
                    "slab_type": "hp_slab",
                    "algorithm_name": "RANDOM_SEARCH",
                    "algorithm_parameters": {"max_evaluations": 5},
                    "repetition": 0,
                    "random_seed": 1,
                    "termination_mode": "fixed_target",
                    "global_optimum": 500.0,
                }
            )
        with self.assertRaises(ValueError):
            OptimizationTaskSpec(
                problem_id="1",
                slab_type="hp_slab",
                algorithm_name="RANDOM_SEARCH",
                algorithm_parameters={"max_evaluations": 5},
                repetition=0,
                random_seed=1,
                termination_mode="fixed_target",
                global_optimum=500.0,
                allowed_deviation_percent=5.0,
                target_reference="all",
                target_threshold=999.0,
            )


class TestWarmStartPlannerParameters(unittest.TestCase):
    def test_apply_warm_start_parameters_sets_the_predesign_path(self) -> None:
        path_tokens = frozenset({"FAKE_WARM_START"})
        params = apply_warm_start_parameters(
            algorithm_name="FAKE_WARM_START",
            parameters={"max_evaluations": 10},
            output_root="experiments",
            problem_id="L8100_B2",
            repetition=0,
            repetitions_count=1,
            initial_design_path_tokens=path_tokens,
            rank1_seedless_tokens=frozenset(),
        )
        self.assertEqual(
            params["initial_design_path"],
            str(predesign_initial_design_path("experiments", "L8100_B2")),
        )
        self.assertNotIn("start_index", params)
        with self.assertRaises(AlgorithmFactoryError) as raised:
            apply_warm_start_parameters(
                algorithm_name="FAKE_WARM_START",
                parameters={"max_evaluations": 10},
                output_root="experiments",
                problem_id="L8100_B2",
                repetition=0,
                repetitions_count=2,
                initial_design_path_tokens=path_tokens,
                rank1_seedless_tokens=path_tokens,
            )
        self.assertIn("repetitions.count", str(raised.exception))
        self.assertIn("1", str(raised.exception))

    def test_existing_warm_starts_keep_start_index(self) -> None:
        algorithms = textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "PYNOMAD_MADS_PB_WARM_START"
            max_evaluations = 10

            [[experiment.algorithms]]
            name = "PYNOMAD_VNS_MADS_PB_WARM_START"
            max_evaluations = 10
            """
        )
        repetitions = textwrap.dedent(
            """\
            [experiment.repetitions]
            count = 2
            seed_base = 1
            """
        )
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["L8100_B2"]',
                    algorithms=algorithms,
                    repetitions=repetitions,
                )
            )
            specs = plan_optimization_specs(config)
        self.assertEqual(len(specs), 4)
        for spec in specs:
            self.assertNotIn("start_index", spec.algorithm_parameters)
            self.assertEqual(
                spec.algorithm_parameters["initial_design_path"],
                str(predesign_initial_design_deb_path("experiments", "L8100_B2")),
            )
            self.assertIsInstance(spec.random_seed, int)

    def test_deb_warm_starts_receive_the_deb_initial_design(self) -> None:
        from core.ioh_core.algorithms.warm_start import (
            DEB_WARM_START_TOKENS,
            INITIAL_DESIGN_PATH_TOKENS,
            WARM_START_TOKENS,
            initial_design_prefix,
        )
        from core.ioh_core.workflows.predesign_paths import (
            predesign_initial_design_deb_path,
        )

        self.assertEqual(
            DEB_WARM_START_TOKENS,
            frozenset(
                {
                    "DFLINT_BBOA_WARM_START",
                    "DISCRETE_MLSL_DEB_EQU_DESC_WARM_START",
                    "DISCRETE_MLSL_DEB_SI_DESC_WARM_START",
                    "K_ELITE_AMS_DEB_EQU_DESC_WARM_START",
                    "K_ELITE_AMS_DEB_SI_DESC_WARM_START",
                    "OPTUNA_TPE_WARM_START",
                    "PFN_CEI_WARM_START",
                    "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START",
                    "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                    "PYMOO_GA_CONSTRAINED_WARM_START",
                    "PYNOMAD_MADS_PB_WARM_START",
                    "PYNOMAD_VNS_MADS_PB_WARM_START",
                    "SCIPY_COBYLA_WARM_START",
                }
            ),
        )
        self.assertTrue(DEB_WARM_START_TOKENS <= INITIAL_DESIGN_PATH_TOKENS)
        self.assertTrue(DEB_WARM_START_TOKENS <= WARM_START_TOKENS)
        for token in sorted(DEB_WARM_START_TOKENS):
            params = apply_warm_start_parameters(
                algorithm_name=token,
                parameters={"max_evaluations": 10},
                output_root="experiments",
                problem_id="L8100_B2",
                repetition=0,
                repetitions_count=1,
            )
            expected = predesign_initial_design_deb_path(
                "experiments",
                "L8100_B2",
            )
            self.assertEqual(params["initial_design_path"], str(expected))
        legacy = apply_warm_start_parameters(
            algorithm_name="DISCRETE_MLSL_PEN_SI_DESC_WARM_START",
            parameters={"max_evaluations": 10},
            output_root="experiments",
            problem_id="L8100_B2",
            repetition=0,
            repetitions_count=1,
        )
        self.assertEqual(
            legacy["initial_design_path"],
            str(predesign_initial_design_path("experiments", "L8100_B2")),
        )
        penalty_file = str(
            predesign_initial_design_path("experiments", "L8100_B2")
        )
        for token in ("PYMOO_GA_WARM_START", "PFN_PEN_WARM_START"):
            twin = apply_warm_start_parameters(
                algorithm_name=token,
                parameters={"max_evaluations": 10},
                output_root="experiments",
                problem_id="L8100_B2",
                repetition=0,
                repetitions_count=1,
            )
            self.assertEqual(
                twin["initial_design_path"],
                penalty_file,
                token,
            )
        missing = predesign_initial_design_deb_path(
            "experiments",
            "L8100_B2",
        )
        with self.assertRaises(ValueError) as raised:
            initial_design_prefix(
                missing,
                1,
                algorithm_name="DISCRETE_MLSL_DEB_SI_DESC_WARM_START",
                problem_id="L8100_B2",
            )
        message = str(raised.exception)
        self.assertIn("initial_design_deb.json", message)
        self.assertIn("predesign grid", message)
        historical = INITIAL_DESIGN_PATH_TOKENS - DEB_WARM_START_TOKENS
        for token in sorted(historical):
            params = apply_warm_start_parameters(
                algorithm_name=token,
                parameters={"max_evaluations": 10},
                output_root="experiments",
                problem_id="L8100_B2",
                repetition=0,
                repetitions_count=1,
            )
            self.assertEqual(
                params["initial_design_path"],
                str(predesign_initial_design_path("experiments", "L8100_B2")),
                token,
            )
        from core.ioh_core.algorithms.registry import get_algorithm_spec

        prefix_defaults = {
            "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START": ("population_size", 8),
            "K_ELITE_AMS_DEB_SI_DESC_WARM_START": ("n_parents", 5),
            "DISCRETE_MLSL_DEB_SI_DESC_WARM_START": ("batch_size", 128),
        }
        for token, (name, expected) in prefix_defaults.items():
            spec = get_algorithm_spec(token)
            default = {
                parameter.name: parameter.default
                for parameter in spec.parameters
            }[name]
            self.assertEqual(default, expected, token)

    def test_cobyla_warm_start_rejects_repeated_runs(self) -> None:
        algorithms = textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "SCIPY_COBYLA_WARM_START"
            max_evaluations = 10
            """
        )
        repetitions = textwrap.dedent(
            """\
            [experiment.repetitions]
            count = 2
            seed_base = 1
            """
        )
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["L8100_B2"]',
                    algorithms=algorithms,
                    repetitions=repetitions,
                )
            )
            with self.assertRaisesRegex(AlgorithmFactoryError, "repetitions.count"):
                plan_optimization_specs(config)

    def test_dual_annealing_warm_start_allows_two_repetitions(self) -> None:
        algorithms = textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "SCIPY_DUAL_ANNEALING_WARM_START"
            max_evaluations = 10
            """
        )
        repetitions = textwrap.dedent(
            """\
            [experiment.repetitions]
            count = 2
            seed_base = 1
            """
        )
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["L8100_B2"]',
                    algorithms=algorithms,
                    repetitions=repetitions,
                )
            )
            specs = plan_optimization_specs(config)
        self.assertEqual(len(specs), 2)
        self.assertEqual(len({spec.random_seed for spec in specs}), 2)
        for spec in specs:
            self.assertNotIn("start_index", spec.algorithm_parameters)
            self.assertIn("initial_design_path", spec.algorithm_parameters)

    def test_dual_annealing_warm_start_rejects_toml_path(self) -> None:
        algorithms = textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "SCIPY_DUAL_ANNEALING_WARM_START"
            max_evaluations = 10

            [experiment.algorithms.parameters]
            initial_design_path = "C:/not/used.json"
            """
        )
        repetitions = textwrap.dedent(
            """\
            [experiment.repetitions]
            count = 1
            seed_base = 1
            """
        )
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["L8100_B2"]',
                    algorithms=algorithms,
                    repetitions=repetitions,
                )
            )
            with self.assertRaisesRegex(AlgorithmFactoryError, "initial_design_path"):
                plan_optimization_specs(config)

    def test_dual_annealing_warm_start_rejects_toml_start_index(self) -> None:
        algorithms = textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "SCIPY_DUAL_ANNEALING_WARM_START"
            max_evaluations = 10

            [experiment.algorithms.parameters]
            start_index = 3
            """
        )
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["L8100_B2"]',
                    algorithms=algorithms,
                )
            )
            with self.assertRaisesRegex(AlgorithmFactoryError, "start_index"):
                plan_optimization_specs(config)

    def test_sa_warm_start_allows_two_repetitions(self) -> None:
        algorithms = textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "ALTBACKEN_SA_WARM_START"
            max_evaluations = 10
            """
        )
        repetitions = textwrap.dedent(
            """\
            [experiment.repetitions]
            count = 2
            seed_base = 1
            """
        )
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(
                    Path(tmp) / "exp.toml",
                    problem_ids='["L8100_B2"]',
                    algorithms=algorithms,
                    repetitions=repetitions,
                )
            )
            specs = plan_optimization_specs(config)
        self.assertEqual(len(specs), 2)
        self.assertEqual(len({spec.random_seed for spec in specs}), 2)
        for spec in specs:
            self.assertNotIn("start_index", spec.algorithm_parameters)
            self.assertIn("initial_design_path", spec.algorithm_parameters)

    def test_absent_seed_manifest_keeps_token_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            config = load_experiment_config(
                _write_config(Path(tmp) / "exp.toml")
            )
            specs = plan_optimization_specs(config)
            self.assertIsNone(config.experiment.repetitions.seed_manifest_path)
            self.assertEqual(
                specs[0].random_seed,
                derive_random_seed(
                    seed_base=config.experiment.repetitions.seed_base,
                    slab_type="hp_slab",
                    problem_id=specs[0].problem_id,
                    algorithm_name=specs[0].algorithm_name,
                    repetition=specs[0].repetition,
                ),
            )

    def test_seed_manifest_overrides_every_planned_task(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "seeds.jsonl"
            rows = []
            for problem_id in ("1", "2", "10"):
                for repetition in range(3):
                    rows.append(
                        {
                            "algorithm_name": "RANDOM_SEARCH",
                            "problem_id": problem_id,
                            "repetition": repetition,
                            "start": "Cold",
                            "random_seed": 1000 + repetition,
                        }
                    )
            manifest.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            repetitions = textwrap.dedent(
                f"""\
                [experiment.repetitions]
                count = 3
                seed_base = 42
                seed_manifest_path = "{manifest.as_posix()}"
                """
            )
            algorithms = textwrap.dedent(
                """\
                [[experiment.algorithms]]
                name = "RANDOM_SEARCH"
                max_evaluations = 10
                """
            )
            config = load_experiment_config(
                _write_config(
                    root / "exp.toml",
                    algorithms=algorithms,
                    repetitions=repetitions,
                )
            )
            specs = plan_optimization_specs(config)
            self.assertEqual(len(specs), 9)
            for spec in specs:
                self.assertEqual(spec.random_seed, 1000 + spec.repetition)
                self.assertNotEqual(
                    spec.random_seed,
                    derive_random_seed(
                        seed_base=42,
                        slab_type="hp_slab",
                        problem_id=spec.problem_id,
                        algorithm_name=spec.algorithm_name,
                        repetition=spec.repetition,
                    ),
                )

    def test_seed_manifest_rejects_a_missing_or_duplicate_task(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "seeds.jsonl"
            row = {
                "algorithm_name": "RANDOM_SEARCH",
                "problem_id": "1",
                "repetition": 0,
                "random_seed": 7,
            }
            manifest.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            repetitions = textwrap.dedent(
                f"""\
                [experiment.repetitions]
                count = 1
                seed_base = 42
                seed_manifest_path = "{manifest.as_posix()}"
                """
            )
            config = load_experiment_config(
                _write_config(
                    root / "exp.toml",
                    problem_ids='["1"]',
                    repetitions=repetitions,
                )
            )
            with self.assertRaisesRegex(AlgorithmFactoryError, "duplicates"):
                plan_optimization_specs(config)
            missing = root / "missing.jsonl"
            missing.write_text(json.dumps(row) + "\n", encoding="utf-8")
            missing_repetitions = textwrap.dedent(
                f"""\
                [experiment.repetitions]
                count = 2
                seed_base = 42
                seed_manifest_path = "{missing.as_posix()}"
                """
            )
            missing_config = load_experiment_config(
                _write_config(
                    root / "missing.toml",
                    problem_ids='["1"]',
                    algorithms=textwrap.dedent(
                        """\
                        [[experiment.algorithms]]
                        name = "RANDOM_SEARCH"
                        max_evaluations = 5
                        """
                    ),
                    repetitions=missing_repetitions,
                )
            )
            with self.assertRaisesRegex(AlgorithmFactoryError, "missing"):
                plan_optimization_specs(missing_config)


if __name__ == "__main__":
    unittest.main()
