"""
Phase ALG1 / ALG1.1 gate: algorithm registry metadata, validation, lazy ctors.
"""
from __future__ import annotations

import json
import sys
import unittest
from typing import Any, Mapping

from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import (
    ALGORITHM_REGISTRY,
    EMPTY_REGISTRY,
    AlgorithmSpec,
    BudgetSemantics,
    CandidateMode,
    ParameterKind,
    ParameterSpec,
    RegistryError,
    SeedPolicy,
    construct_algorithm,
    create_registry,
    get_algorithm_spec,
    list_algorithm_specs,
    normalize_algorithm_parameters,
    normalize_parameters,
    resolve_constructor,
    resolve_parameter_validator,
    validate_seed,
)


class _ToyAlgorithm:
    """Minimal algorithm object satisfying :class:`IOHAlgorithm`."""

    def __init__(self, name: str, info: str, parameters: Mapping[str, Any], seed: int | None):
        self.name = name
        self.info = info
        self.parameters = dict(parameters)
        self.seed = seed

    def __call__(self, problem: Any) -> object:
        return self


def build_toy_algorithm(parameters: Mapping[str, Any], random_seed: int | None) -> _ToyAlgorithm:
    """
    Test constructor matching the registry callable signature.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters.
    random_seed : int or None
        Task seed.

    Returns
    -------
    _ToyAlgorithm
        Fresh toy instance.
    """
    return _ToyAlgorithm("TOY", "toy algorithm", parameters, random_seed)


def toy_validator_passthrough(
    parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    """
    Cross-field validator that returns a shallow copy.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters.

    Returns
    -------
    mapping
        Copied parameters.
    """
    return dict(parameters)


def toy_validator_sets_mode_slow(
    parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    """
    Cross-field validator that overrides ``mode`` to ``slow``.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters.

    Returns
    -------
    mapping
        Parameters with ``mode=\"slow\"``.
    """
    out = dict(parameters)
    out["mode"] = "slow"
    return out


def toy_validator_returns_invalid_mode(
    parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    """
    Cross-field validator that returns an illegal ``mode`` value.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters.

    Returns
    -------
    mapping
        Parameters with invalid ``mode``.
    """
    out = dict(parameters)
    out["mode"] = "medium"
    return out


def toy_validator_mutates_input(
    parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    """
    Cross-field validator that attempts to mutate its input mapping.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters (expected immutable).

    Returns
    -------
    mapping
        Copied parameters.
    """
    try:
        parameters["mode"] = "slow"  # type: ignore[index]
    except TypeError:
        return dict(parameters)
    raise AssertionError("validator received a mutable mapping")


def _toy_spec(**overrides: Any) -> AlgorithmSpec:
    """
    Build a TOY AlgorithmSpec with optional field overrides.

    Returns
    -------
    AlgorithmSpec
        Spec pointing at :func:`build_toy_algorithm`.
    """
    base = dict(
        name="TOY",
        library="test",
        constructor_ref=(
            "_mains.Testing.test_cursor_algorithm_registry:build_toy_algorithm"
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
                choices=("fast", "slow"),
            ),
            ParameterSpec(
                name="flag",
                types=(bool,),
                required=False,
                default=False,
            ),
        ),
        seed_policy=SeedPolicy.REQUIRED,
        dependency_extra=None,
        candidate_mode=CandidateMode.INTEGER,
        budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
        description="toy",
    )
    base.update(overrides)
    return AlgorithmSpec(**base)


class TestProtocol(unittest.TestCase):
    """Runtime-checkable IOHAlgorithm protocol."""

    def test_protocol_accepts_toy_and_rejects_incomplete(self) -> None:
        algo = _ToyAlgorithm("TOY", "info", {}, 1)
        self.assertIsInstance(algo, IOHAlgorithm)

        class Incomplete:
            name = "X"

            def __call__(self, problem: Any) -> object:
                return None

        self.assertNotIsInstance(Incomplete(), IOHAlgorithm)


class TestRegistryCore(unittest.TestCase):
    """Registry creation, lookup, and listing."""

    def test_duplicate_names_rejected_and_lookup_lists_supported(self) -> None:
        spec = _toy_spec()
        with self.assertRaisesRegex(RegistryError, "duplicate"):
            create_registry([spec, spec])

        registry = create_registry([spec, _toy_spec(name="OTHER")])
        self.assertEqual(registry.names(), ["OTHER", "TOY"])
        self.assertEqual(registry.get("TOY").library, "test")
        with self.assertRaisesRegex(RegistryError, "unknown algorithm"):
            registry.get("MISSING")
        self.assertEqual(EMPTY_REGISTRY.names(), [])


class TestParameterAndSeedValidation(unittest.TestCase):
    """Parameter normalization and seed policies."""

    def test_defaults_unknown_types_ranges_choices_bool_vs_int(self) -> None:
        spec = _toy_spec()
        original = {"max_evaluations": 3}
        normalized = normalize_parameters(spec, original)
        self.assertEqual(normalized["max_evaluations"], 3)
        self.assertEqual(normalized["mode"], "fast")
        self.assertEqual(normalized["flag"], False)
        self.assertEqual(original, {"max_evaluations": 3})

        with self.assertRaisesRegex(RegistryError, "missing required"):
            normalize_parameters(spec, {})
        with self.assertRaisesRegex(RegistryError, "unknown parameters"):
            normalize_parameters(spec, {"max_evaluations": 1, "population": 10})
        with self.assertRaisesRegex(RegistryError, "must be >= 1"):
            normalize_parameters(spec, {"max_evaluations": 0})
        with self.assertRaisesRegex(RegistryError, "one of"):
            normalize_parameters(spec, {"max_evaluations": 1, "mode": "medium"})
        with self.assertRaisesRegex(RegistryError, "must be one of"):
            # bool must not satisfy int
            normalize_parameters(spec, {"max_evaluations": True})

    def test_seed_policies_and_unsigned_bounds(self) -> None:
        required = _toy_spec(seed_policy=SeedPolicy.REQUIRED)
        forbidden = _toy_spec(name="DET", seed_policy=SeedPolicy.FORBIDDEN)

        validate_seed(required, 0)
        validate_seed(required, 0xFFFFFFFF)
        with self.assertRaisesRegex(RegistryError, "requires a random_seed"):
            validate_seed(required, None)
        with self.assertRaisesRegex(RegistryError, "unsigned 32-bit"):
            validate_seed(required, -1)
        with self.assertRaisesRegex(RegistryError, "unsigned 32-bit"):
            validate_seed(required, 0x100000000)

        validate_seed(forbidden, None)
        with self.assertRaisesRegex(RegistryError, "seedless"):
            validate_seed(forbidden, 1)


class TestAlg11ContractHardening(unittest.TestCase):
    """ALG1.1 enums, batch flag, and lazy cross-field validators."""

    def test_algorithm_spec_rejects_invalid_enum_metadata(self) -> None:
        with self.assertRaisesRegex(RegistryError, "candidate_mode"):
            _toy_spec(candidate_mode="integer")
        with self.assertRaisesRegex(RegistryError, "budget_semantics"):
            _toy_spec(budget_semantics="hard_ioh_evaluations")
        with self.assertRaisesRegex(RegistryError, "kind must be ParameterKind"):
            ParameterSpec(
                name="path",
                types=(str,),
                kind="executable_path",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(RegistryError, "batch_evaluations must be bool"):
            _toy_spec(batch_evaluations=1)  # type: ignore[arg-type]

    def test_algorithm_metadata_enums_are_json_strings(self) -> None:
        spec = _toy_spec()
        payload = {
            "candidate_mode": spec.candidate_mode,
            "budget_semantics": spec.budget_semantics,
            "parameter_kind": ParameterKind.EXECUTABLE_PATH,
            "seed_policy": spec.seed_policy,
        }
        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["candidate_mode"], "integer")
        self.assertEqual(decoded["budget_semantics"], "hard_ioh_evaluations")
        self.assertEqual(decoded["parameter_kind"], "executable_path")
        self.assertEqual(decoded["seed_policy"], "required")

    def test_batch_evaluations_defaults_false(self) -> None:
        spec = _toy_spec()
        self.assertIs(spec.batch_evaluations, False)
        self.assertIs(_toy_spec(batch_evaluations=True).batch_evaluations, True)

    def test_validator_ref_is_lazy(self) -> None:
        with self.assertRaisesRegex(RegistryError, "malformed.*validator_ref"):
            _toy_spec(validator_ref="not_a_ref")
        with self.assertRaisesRegex(RegistryError, "malformed.*validator_ref"):
            _toy_spec(validator_ref=":missing_module")
        with self.assertRaisesRegex(RegistryError, "malformed.*validator_ref"):
            _toy_spec(validator_ref="pkg.mod:attr.nested")

        missing_module = _toy_spec(
            validator_ref="package.that.does.not.exist:validate"
        )
        with self.assertRaisesRegex(RegistryError, "cannot import validator"):
            normalize_parameters(missing_module, {"max_evaluations": 1})

        missing_attr = _toy_spec(
            validator_ref=(
                "_mains.Testing.test_cursor_algorithm_registry:missing_validator"
            )
        )
        with self.assertRaisesRegex(RegistryError, "validator .* not found"):
            normalize_parameters(missing_attr, {"max_evaluations": 1})

        resolve_parameter_validator(
            "_mains.Testing.test_cursor_algorithm_registry:toy_validator_passthrough"
        )

    def test_validator_result_is_copied_and_revalidated(self) -> None:
        mutating = _toy_spec(
            validator_ref=(
                "_mains.Testing.test_cursor_algorithm_registry:"
                "toy_validator_mutates_input"
            )
        )
        original = {"max_evaluations": 2, "mode": "fast"}
        normalized = normalize_parameters(mutating, original)
        self.assertEqual(normalized["mode"], "fast")
        self.assertEqual(original["mode"], "fast")

        override = _toy_spec(
            validator_ref=(
                "_mains.Testing.test_cursor_algorithm_registry:"
                "toy_validator_sets_mode_slow"
            )
        )
        overridden = normalize_parameters(override, {"max_evaluations": 2})
        self.assertEqual(overridden["mode"], "slow")

        invalid = _toy_spec(
            validator_ref=(
                "_mains.Testing.test_cursor_algorithm_registry:"
                "toy_validator_returns_invalid_mode"
            )
        )
        with self.assertRaisesRegex(RegistryError, "one of"):
            normalize_parameters(invalid, {"max_evaluations": 2})


class TestLazyConstructors(unittest.TestCase):
    """Lazy constructor resolution and construction."""

    def test_resolve_and_construct(self) -> None:
        registry = create_registry([_toy_spec()])
        algo = construct_algorithm(
            registry,
            "TOY",
            {"max_evaluations": 4},
            random_seed=7,
        )
        self.assertIsInstance(algo, IOHAlgorithm)
        self.assertEqual(algo.parameters["max_evaluations"], 4)
        self.assertEqual(algo.parameters["mode"], "fast")
        self.assertEqual(algo.seed, 7)

        ctor = resolve_constructor(
            "_mains.Testing.test_cursor_algorithm_registry:build_toy_algorithm"
        )
        self.assertTrue(callable(ctor))

        with self.assertRaisesRegex(RegistryError, "malformed"):
            resolve_constructor("not_a_ref")
        with self.assertRaisesRegex(RegistryError, "cannot import"):
            resolve_constructor("package.that.does.not.exist:factory")
        with self.assertRaisesRegex(RegistryError, "not found"):
            resolve_constructor(
                "_mains.Testing.test_cursor_algorithm_registry:missing_symbol"
            )


class TestProductionRegistry(unittest.TestCase):
    """ALG2.4: production registry exposes the migrated algorithms."""

    def test_production_registry_contains_existing_algorithms(self) -> None:
        self.assertEqual(
            ALGORITHM_REGISTRY.names(),
            [
                "ALTBACKEN_SA",
                "ALTBACKEN_SA_WARM_START",
                "BLACKBOXOPT_DYCORS",
                "BLACKBOXOPT_GOSAC",
                "BLACKBOXOPT_MSRS",
                "CMAES_CATCMAWM",
                "CMAES_CATCMAWM_WARM_START",
                "DFLINT_BBOA",
                "DFLINT_BBOA_WARM_START",
                "DISCRETE_MLSL_DEB_EQU_DESC_COLD_START",
                "DISCRETE_MLSL_DEB_EQU_DESC_WARM_START",
                "DISCRETE_MLSL_DEB_SI_DESC_COLD_START",
                "DISCRETE_MLSL_DEB_SI_DESC_WARM_START",
                "DISCRETE_MLSL_PEN_EQU_DESC_COLD_START",
                "DISCRETE_MLSL_PEN_EQU_DESC_WARM_START",
                "DISCRETE_MLSL_PEN_SI_DESC_COLD_START",
                "DISCRETE_MLSL_PEN_SI_DESC_WARM_START",
                "HYPEROPT_TPE",
                "HYPEROPT_TPE_WARM_START",
                "K_ELITE_AMS_DEB_EQU_DESC_COLD_START",
                "K_ELITE_AMS_DEB_EQU_DESC_WARM_START",
                "K_ELITE_AMS_DEB_SI_DESC_COLD_START",
                "K_ELITE_AMS_DEB_SI_DESC_WARM_START",
                "K_ELITE_AMS_PEN_EQU_DESC_COLD_START",
                "K_ELITE_AMS_PEN_EQU_DESC_WARM_START",
                "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
                "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
                "LATIN_HYPERCUBE",
                "LEAP_GA",
                "LEAP_GA_WARM_START",
                "NEVERGRAD_METAMODEL",
                "NEVERGRAD_METAMODEL_ONEPLUSONE",
                "NEVERGRAD_METAMODEL_ONEPLUSONE_WARM_START",
                "NEVERGRAD_METAMODEL_WARM_START",
                "NLOPT_DIRECT_L",
                "NLOPT_SBPLX",
                "NLOPT_SBPLX_WARM_START",
                "OPTAX_ADAM",
                "OPTAX_ADAM_WARM_START",
                "OPTAX_RPROP",
                "OPTAX_RPROP_WARM_START",
                "OPTAX_SGD_NESTEROV",
                "OPTAX_SGD_NESTEROV_WARM_START",
                "OPTUNA_TPE",
                "OPTUNA_TPE_WARM_START",
                "PFN_CEI",
                "PFN_CEI_WARM_START",
                "PFN_PEN",
                "PFN_PEN_WARM_START",
                "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_COLD_START",
                "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START",
                "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
                "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                "POPULATION_BASIN_HOPPING_PEN_EQU_DESC_COLD_START",
                "POPULATION_BASIN_HOPPING_PEN_EQU_DESC_WARM_START",
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
                "PYMOO_GA",
                "PYMOO_GA_CONSTRAINED",
                "PYMOO_GA_CONSTRAINED_WARM_START",
                "PYMOO_GA_WARM_START",
                "PYNOMAD_MADS_PB",
                "PYNOMAD_MADS_PB_WARM_START",
                "PYNOMAD_VNS_MADS_PB",
                "PYNOMAD_VNS_MADS_PB_WARM_START",
                "PYSTARPSO_INTEGER_PSO",
                "PYSTARPSO_INTEGER_PSO_WARM_START",
                "RANDOM_SEARCH",
                "RBFOPT",
                "RBFOPT_WARM_START",
                "SCIPY_COBYLA",
                "SCIPY_COBYLA_WARM_START",
                "SCIPY_DIFFERENTIAL_EVOLUTION",
                "SCIPY_DIFFERENTIAL_EVOLUTION_WARM_START",
                "SCIPY_DIRECT_L",
                "SCIPY_DUAL_ANNEALING",
                "SCIPY_DUAL_ANNEALING_WARM_START",
                "SMT_EGO",
                "SMT_EGO_WARM_START",
                "SOBOL_MULTISTART_PEN_EQU_DESC_COLD_START",
                "SOBOL_MULTISTART_PEN_SI_DESC_COLD_START",
                "SOBOL_PEN_COLD_START",
            ],
        )
        specs = list_algorithm_specs()
        self.assertEqual([spec.name for spec in specs], ALGORITHM_REGISTRY.names())

        random_search = get_algorithm_spec("RANDOM_SEARCH")
        self.assertEqual(random_search.library, "native")
        self.assertEqual(
            random_search.constructor_ref,
            "core.ioh_core.algorithms.native.factory:build_random_search",
        )
        self.assertEqual(random_search.seed_policy, SeedPolicy.REQUIRED)
        self.assertEqual(random_search.candidate_mode, CandidateMode.INTEGER)
        self.assertEqual(
            random_search.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertIsNone(random_search.dependency_extra)
        self.assertEqual(len(random_search.parameters), 1)
        self.assertEqual(random_search.parameters[0].name, "max_evaluations")
        self.assertEqual(random_search.parameters[0].types, (int,))
        self.assertTrue(random_search.parameters[0].required)
        self.assertEqual(random_search.parameters[0].minimum, 1)

        direct_l = get_algorithm_spec("NLOPT_DIRECT_L")
        self.assertEqual(direct_l.library, "nlopt")
        self.assertEqual(
            direct_l.constructor_ref,
            "core.ioh_core.algorithms.nlopt.factory:build_direct_l",
        )
        self.assertEqual(direct_l.seed_policy, SeedPolicy.FORBIDDEN)
        self.assertEqual(direct_l.candidate_mode, CandidateMode.CONTINUOUS_ROUNDED)
        self.assertEqual(
            direct_l.budget_semantics,
            BudgetSemantics.HARD_IOH_EVALUATIONS,
        )
        self.assertIsNone(direct_l.dependency_extra)
        self.assertEqual(len(direct_l.parameters), 1)
        self.assertEqual(direct_l.parameters[0].name, "max_evaluations")

        normalized = normalize_algorithm_parameters(
            "RANDOM_SEARCH",
            {"max_evaluations": 3},
        )
        self.assertEqual(normalized, {"max_evaluations": 3})


class TestNoOptionalImports(unittest.TestCase):
    """Registry/common imports must not pull optional optimiser libraries."""

    def test_registry_and_common_do_not_import_optional_libs(self) -> None:
        for name in ("pymoo", "hyperopt", "rbfopt"):
            sys.modules.pop(name, None)
        # Importing the contract modules must not load optional optimisers.
        import core.ioh_core.algorithms.common  # noqa: F401
        import core.ioh_core.algorithms.registry  # noqa: F401

        self.assertNotIn("pymoo", sys.modules)
        self.assertNotIn("hyperopt", sys.modules)
        self.assertNotIn("rbfopt", sys.modules)


if __name__ == "__main__":
    unittest.main()
