"""
Cross-checks for ranked InitialDesign warm starts.
"""
from __future__ import annotations

import json
import os
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from _mains.Testing.helpers.fake_ioh_problem import FakeIOHProblem
from _mains.Testing.test_cursor_algorithm_pymoo import _ConstrainedFakeIOHProblem
from _mains.Testing.test_cursor_optimization_planner import _write_config
from core.execution_core.config import load_experiment_config
from core.ioh_core.algorithm_factory import AlgorithmFactoryError, create_algorithm
from core.ioh_core.algorithms.registry import (
    ALGORITHM_REGISTRY,
    SeedPolicy,
    get_algorithm_spec,
)
from core.ioh_core.algorithms.warm_start import (
    RANK1_SEEDLESS_TOKENS,
    WARM_START_TOKENS,
)
from core.ioh_core.workflows.optimization_runs.planner import plan_optimization_specs
from core.ioh_core.workflows.predesign_paths import (
    predesign_initial_design_deb_path,
    predesign_initial_design_path,
)


def _cold_token(name: str) -> str:
    """Return the cold token paired with a warm-start token."""
    stem = name.removesuffix("_WARM_START")
    if stem.endswith("_SI_DESC") or stem.endswith("_EQU_DESC"):
        return f"{stem}_COLD_START"
    return stem


_SECTION_5_NAMES = frozenset(
    {
        "ALTBACKEN_SA_WARM_START",
        "CMAES_CATCMAWM_WARM_START",
        "DFLINT_BBOA_WARM_START",
        "DISCRETE_MLSL_DEB_EQU_DESC_WARM_START",
        "DISCRETE_MLSL_DEB_SI_DESC_WARM_START",
        "DISCRETE_MLSL_PEN_EQU_DESC_WARM_START",
        "DISCRETE_MLSL_PEN_SI_DESC_WARM_START",
        "HYPEROPT_TPE_WARM_START",
        "K_ELITE_AMS_DEB_EQU_DESC_WARM_START",
        "K_ELITE_AMS_DEB_SI_DESC_WARM_START",
        "K_ELITE_AMS_PEN_EQU_DESC_WARM_START",
        "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
        "LEAP_GA_WARM_START",
        "NEVERGRAD_METAMODEL_ONEPLUSONE_WARM_START",
        "NEVERGRAD_METAMODEL_WARM_START",
        "NLOPT_SBPLX_WARM_START",
        "OPTAX_ADAM_WARM_START",
        "OPTAX_RPROP_WARM_START",
        "OPTAX_SGD_NESTEROV_WARM_START",
        "OPTUNA_TPE_WARM_START",
        "PFN_CEI_WARM_START",
        "PFN_PEN_WARM_START",
        "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START",
        "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
        "POPULATION_BASIN_HOPPING_PEN_EQU_DESC_WARM_START",
        "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
        "PYMOO_GA_CONSTRAINED_WARM_START",
        "PYMOO_GA_WARM_START",
        "PYNOMAD_MADS_PB_WARM_START",
        "PYNOMAD_VNS_MADS_PB_WARM_START",
        "PYSTARPSO_INTEGER_PSO_WARM_START",
        "RBFOPT_WARM_START",
        "SCIPY_COBYLA_WARM_START",
        "SCIPY_DIFFERENTIAL_EVOLUTION_WARM_START",
        "SCIPY_DUAL_ANNEALING_WARM_START",
        "SMT_EGO_WARM_START",
    }
)


def _vectors(count: int) -> list[list[int]]:
    return [[index, 0, 0, 0, 0, 0, 0, 0, 0] for index in range(count)]


def _write_design(path: Path, vectors: list[list[int]]) -> None:
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


def _problem(*, constrained: bool) -> FakeIOHProblem:
    lower = [0] * 9
    upper = [200] * 9
    if constrained:
        return _ConstrainedFakeIOHProblem(lower, upper)
    return FakeIOHProblem(lower, upper)


class TestWarmStartRegistry(unittest.TestCase):
    def test_warm_names_match_section_5(self) -> None:
        names = {
            name
            for name in ALGORITHM_REGISTRY.names()
            if name.endswith("_WARM_START")
        }
        self.assertEqual(names, WARM_START_TOKENS)
        self.assertEqual(WARM_START_TOKENS, _SECTION_5_NAMES)

    def test_metadata_matches_the_cold_token(self) -> None:
        for name in sorted(WARM_START_TOKENS):
            cold_name = _cold_token(name)
            warm = get_algorithm_spec(name)
            cold = get_algorithm_spec(cold_name)
            self.assertEqual(warm.dependency_extra, cold.dependency_extra, name)
            self.assertEqual(warm.candidate_mode, cold.candidate_mode, name)
            self.assertEqual(warm.budget_semantics, cold.budget_semantics, name)
            if name in RANK1_SEEDLESS_TOKENS:
                self.assertNotEqual(warm.seed_policy, cold.seed_policy, name)
                self.assertEqual(warm.seed_policy, SeedPolicy.FORBIDDEN, name)
            else:
                self.assertEqual(warm.seed_policy, cold.seed_policy, name)
            warm_parameters = [
                parameter
                for parameter in warm.parameters
                if parameter.name != "initial_design_path"
            ]
            self.assertEqual(warm_parameters, list(cold.parameters), name)
            self.assertIn(
                "initial_design_path",
                [parameter.name for parameter in warm.parameters],
            )

    def test_cold_factory_rejects_initial_design_path(self) -> None:
        for name in sorted(WARM_START_TOKENS):
            cold_name = _cold_token(name)
            cold = get_algorithm_spec(cold_name)
            seed = None if cold.seed_policy is SeedPolicy.FORBIDDEN else 1
            parameters: dict = {
                "max_evaluations": 4,
                "initial_design_path": "missing.json",
            }
            for parameter in cold.parameters:
                if parameter.required and parameter.name not in parameters:
                    parameters[parameter.name] = "solver"
            with self.assertRaises(AlgorithmFactoryError):
                create_algorithm(cold_name, parameters, random_seed=seed)


class TestWarmStartPrefixes(unittest.TestCase):
    def _run(
        self,
        token: str,
        *,
        count: int,
        budget: int,
        extra: dict | None = None,
        constrained: bool = False,
        file_count: int | None = None,
    ) -> list:
        vectors = _vectors(file_count or (count + 1))
        problem = _problem(constrained=constrained)
        spec = get_algorithm_spec(token)
        seed = None if spec.seed_policy is SeedPolicy.FORBIDDEN else 1
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            _write_design(path, vectors)
            parameters = {
                "max_evaluations": budget,
                "initial_design_path": str(path),
            }
            if extra:
                parameters.update(extra)
            create_algorithm(token, parameters, random_seed=seed)(problem)
        return problem.calls

    def test_population_and_doe_prefixes_are_in_order(self) -> None:
        cases = [
            ("PYMOO_GA_WARM_START", 3, {"population_size": 3}, False),
            (
                "PYMOO_GA_CONSTRAINED_WARM_START",
                3,
                {"population_size": 3},
                True,
            ),
            ("LEAP_GA_WARM_START", 3, {"population_size": 3}, False),
            ("PYSTARPSO_INTEGER_PSO_WARM_START", 3, {"n_pop": 3}, False),
            (
                "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                3,
                {"population_size": 3},
                True,
            ),
            (
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
                3,
                {"population_size": 3},
                False,
            ),
            ("K_ELITE_AMS_DEB_SI_DESC_WARM_START", 3, {"n_parents": 3}, True),
            ("K_ELITE_AMS_PEN_SI_DESC_WARM_START", 3, {"n_parents": 3}, False),
            ("DISCRETE_MLSL_PEN_SI_DESC_WARM_START", 4, {"batch_size": 4}, False),
            ("SCIPY_DIFFERENTIAL_EVOLUTION_WARM_START", 135, {}, False),
            ("SMT_EGO_WARM_START", 3, {"n_doe": 3}, False),
            ("PFN_PEN_WARM_START", 2, {}, False),
            ("PFN_CEI_WARM_START", 2, {}, True),
        ]
        for token, count, extra, constrained in cases:
            with self.subTest(token=token):
                budget = 4 if token.startswith("PFN_") else count
                calls = self._run(
                    token,
                    count=count,
                    budget=budget,
                    extra=extra,
                    constrained=constrained,
                )
                self.assertEqual(calls[:count], _vectors(count))

    def test_fixed_length_prefixes_are_in_order(self) -> None:
        for token, constrained in (
            ("NEVERGRAD_METAMODEL_WARM_START", False),
            ("NEVERGRAD_METAMODEL_ONEPLUSONE_WARM_START", False),
            ("HYPEROPT_TPE_WARM_START", False),
            ("OPTUNA_TPE_WARM_START", True),
        ):
            with self.subTest(token=token):
                calls = self._run(
                    token,
                    count=50,
                    budget=50,
                    constrained=constrained,
                )
                self.assertEqual(calls[:50], _vectors(50))

    def test_short_file_raises_before_evaluation(self) -> None:
        problem = FakeIOHProblem([0] * 9, [200] * 9)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            _write_design(path, _vectors(3))
            algorithm = create_algorithm(
                "DISCRETE_MLSL_PEN_SI_DESC_WARM_START",
                {
                    "max_evaluations": 8,
                    "batch_size": 4,
                    "initial_design_path": str(path),
                },
                random_seed=1,
            )
            with self.assertRaises(ValueError) as raised:
                algorithm(problem)
        message = str(raised.exception)
        self.assertIn("needs 4 initial designs", message)
        self.assertIn("contains 3", message)
        self.assertEqual(problem.calls, [])

    def test_mlsl_later_batch_is_not_the_next_design_vector(self) -> None:
        calls = self._run(
            "DISCRETE_MLSL_PEN_SI_DESC_WARM_START",
            count=4,
            budget=6,
            extra={"batch_size": 4},
            file_count=8,
        )
        self.assertEqual(calls[:4], _vectors(4))
        self.assertNotEqual(calls[4], [4, 0, 0, 0, 0, 0, 0, 0, 0])


class TestWarmStartBudgets(unittest.TestCase):
    def _expect_short(self, token: str, budget: int, extra: dict) -> None:
        problem = FakeIOHProblem([0] * 9, [200] * 9)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "initial_design.json"
            _write_design(path, _vectors(200))
            parameters = {
                "max_evaluations": budget,
                "initial_design_path": str(path),
                **extra,
            }
            algorithm = create_algorithm(token, parameters, random_seed=1)
            with self.assertRaises(ValueError) as raised:
                algorithm(problem)
        self.assertIn("initial evaluations", str(raised.exception))
        self.assertEqual(problem.calls, [])

    def test_basin_hopping_budgets(self) -> None:
        token = "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START"
        extra = {"population_size": 8}
        self._expect_short(token, 7, extra)
        calls = TestWarmStartPrefixes()._run(
            token, count=8, budget=8, extra=extra
        )
        self.assertEqual(calls, _vectors(8))
        calls = TestWarmStartPrefixes()._run(
            token, count=8, budget=9, extra=extra
        )
        self.assertEqual(calls[:8], _vectors(8))
        self.assertGreaterEqual(len(calls), 9)

    def test_differential_evolution_budgets(self) -> None:
        token = "SCIPY_DIFFERENTIAL_EVOLUTION_WARM_START"
        self._expect_short(token, 134, {})
        calls = TestWarmStartPrefixes()._run(
            token, count=135, budget=135
        )
        self.assertEqual(calls[:135], _vectors(135))
        calls = TestWarmStartPrefixes()._run(
            token, count=135, budget=136
        )
        self.assertEqual(calls[:135], _vectors(135))
        self.assertGreaterEqual(len(calls), 31)

    def test_genetic_algorithm_budgets(self) -> None:
        token = "PYMOO_GA_WARM_START"
        extra = {"population_size": 50}
        self._expect_short(token, 49, extra)
        calls = TestWarmStartPrefixes()._run(
            token, count=50, budget=50, extra=extra
        )
        self.assertEqual(calls[:50], _vectors(50))
        calls = TestWarmStartPrefixes()._run(
            token, count=50, budget=51, extra=extra
        )
        self.assertEqual(calls[:50], _vectors(50))
        self.assertGreaterEqual(len(calls), 51)

    def test_mlsl_budgets(self) -> None:
        token = "DISCRETE_MLSL_PEN_SI_DESC_WARM_START"
        extra = {"batch_size": 4}
        self._expect_short(token, 3, extra)
        calls = TestWarmStartPrefixes()._run(
            token, count=4, budget=4, extra=extra, file_count=8
        )
        self.assertEqual(calls, _vectors(4))
        calls = TestWarmStartPrefixes()._run(
            token, count=4, budget=5, extra=extra, file_count=8
        )
        self.assertEqual(calls[:4], _vectors(4))
        self.assertGreaterEqual(len(calls), 5)

    def test_rbfopt_budgets(self) -> None:
        token = "RBFOPT_WARM_START"
        with TemporaryDirectory() as tmp:
            minlp = Path(tmp) / "bonmin"
            nlp = Path(tmp) / "ipopt"
            if os.name == "nt":
                minlp = minlp.with_suffix(".exe")
                nlp = nlp.with_suffix(".exe")
            minlp.write_text("", encoding="utf-8")
            nlp.write_text("", encoding="utf-8")

            def _which(value: str) -> str | None:
                if "bonmin" in value:
                    return str(minlp)
                if "ipopt" in value:
                    return str(nlp)
                return None

            extra = {
                "ga_base_population_size": 20,
                "minlp_solver_path": str(minlp),
                "nlp_solver_path": str(nlp),
            }
            with patch(
                "core.ioh_core.algorithms.rbfopt.factory.shutil.which",
                side_effect=_which,
            ):
                self._expect_short(token, 4, extra)
                runner = TestWarmStartPrefixes()
                calls = runner._run(token, count=5, budget=5, extra=extra)
                self.assertEqual(calls[:5], _vectors(5))
                calls = runner._run(token, count=5, budget=6, extra=extra)
        self.assertEqual(calls[:5], _vectors(5))
        self.assertGreaterEqual(len(calls), 6)


class TestWarmStartPlannerChain(unittest.TestCase):
    def test_cobyla_task_path_is_the_first_evaluated_point(self) -> None:
        algorithms = textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "SCIPY_COBYLA_WARM_START"
            max_evaluations = 4
            """
        )
        repetitions = textwrap.dedent(
            """\
            [experiment.repetitions]
            count = 1
            seed_base = 1
            """
        )
        vectors = _vectors(2)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_experiment_config(
                _write_config(
                    root / "exp.toml",
                    problem_ids='["L8100_B2"]',
                    algorithms=algorithms,
                    repetitions=repetitions,
                )
            )
            specs = plan_optimization_specs(config)
            self.assertEqual(len(specs), 1)
            expected = str(
                predesign_initial_design_deb_path("experiments", "L8100_B2")
            )
            self.assertEqual(
                specs[0].algorithm_parameters["initial_design_path"],
                expected,
            )
            design = root / "initial_design.json"
            _write_design(design, vectors)
            parameters = dict(specs[0].algorithm_parameters)
            parameters["initial_design_path"] = str(design)
            problem = _ConstrainedFakeIOHProblem([0] * 9, [200] * 9)
            create_algorithm(
                "SCIPY_COBYLA_WARM_START",
                parameters,
                random_seed=None,
            )(problem)
        self.assertEqual(problem.calls[0], vectors[0])


class TestDebWarmStartFile(unittest.TestCase):
    """Deb warm starts read the Deb file and fail closed without it."""

    def test_prefix_order_is_the_evaluation_order(self) -> None:
        from core.ioh_core.workflows.optimization_runs.planner import (
            apply_warm_start_parameters,
        )
        from core.ioh_core.workflows.predesign_paths import (
            predesign_initial_design_deb_path,
        )

        legacy_vectors = _vectors(4)
        deb_vectors = list(reversed(legacy_vectors))
        cases = (
            (
                "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                3,
                {"population_size": 3},
                True,
                deb_vectors,
            ),
            (
                "K_ELITE_AMS_DEB_SI_DESC_WARM_START",
                3,
                {"n_parents": 3, "archive_size": 6},
                True,
                deb_vectors,
            ),
            (
                "DISCRETE_MLSL_DEB_SI_DESC_WARM_START",
                4,
                {"batch_size": 4},
                True,
                deb_vectors,
            ),
            (
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
                3,
                {"population_size": 3},
                False,
                legacy_vectors,
            ),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_path = predesign_initial_design_path(root, "L8100_B2")
            deb_path = predesign_initial_design_deb_path(root, "L8100_B2")
            legacy_path.parent.mkdir(parents=True)
            _write_design(legacy_path, legacy_vectors)
            _write_design(deb_path, deb_vectors)
            for token, count, extra, constrained, expected in cases:
                with self.subTest(token=token):
                    planned = apply_warm_start_parameters(
                        algorithm_name=token,
                        parameters={"max_evaluations": count, **extra},
                        output_root=str(root),
                        problem_id="L8100_B2",
                        repetition=0,
                        repetitions_count=1,
                    )
                    problem = _problem(constrained=constrained)
                    spec = get_algorithm_spec(token)
                    seed = (
                        None
                        if spec.seed_policy is SeedPolicy.FORBIDDEN
                        else 1
                    )
                    create_algorithm(
                        token,
                        planned,
                        random_seed=seed,
                    )(problem)
                    self.assertEqual(problem.calls[:count], expected[:count])

    def test_missing_deb_sidecar_fails_before_evaluation(self) -> None:
        from core.ioh_core.workflows.optimization_runs.planner import (
            apply_warm_start_parameters,
        )

        problem = _problem(constrained=True)
        with TemporaryDirectory() as tmp:
            planned = apply_warm_start_parameters(
                algorithm_name="DISCRETE_MLSL_DEB_SI_DESC_WARM_START",
                parameters={"max_evaluations": 4, "batch_size": 2},
                output_root=tmp,
                problem_id="L8100_B2",
                repetition=0,
                repetitions_count=1,
            )
            with self.assertRaises(ValueError) as raised:
                create_algorithm(
                    "DISCRETE_MLSL_DEB_SI_DESC_WARM_START",
                    planned,
                    random_seed=1,
                )(problem)
        self.assertEqual(problem.calls, [])
        self.assertIn("initial_design_deb.json", str(raised.exception))
        self.assertIn("predesign grid", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
