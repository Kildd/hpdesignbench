"""
Immutable algorithm registry: metadata, parameter/seed validation, lazy ctors.

Importing this module must not import optional optimiser libraries such as
Pymoo, Hyperopt, or RBFOpt. Constructor modules are loaded only when an
algorithm is actually constructed.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from core.ioh_core.algorithms.protocol import IOHAlgorithm

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list[Any] | dict[str, Any]
Constructor = Callable[[Mapping[str, JSONValue], int | None], IOHAlgorithm]
ParameterValidator = Callable[[Mapping[str, JSONValue]], Mapping[str, JSONValue]]

# Sentinel: parameter has no default and must be omitted or supplied by the user.
_NO_DEFAULT = object()


class SeedPolicy(str, Enum):
    """
    Whether a deterministic task seed is required or forbidden.

    Attributes
    ----------
    REQUIRED : str
        Stochastic algorithm; ``random_seed`` must be an unsigned 32-bit int.
    FORBIDDEN : str
        Deterministic algorithm; ``random_seed`` must be ``None``.
    """

    REQUIRED = "required"
    FORBIDDEN = "forbidden"


class CandidateMode(str, Enum):
    """
    How an adapter maps library proposals onto IOH catalogue-index vectors.

    Attributes
    ----------
    INTEGER : str
        Adapter evaluates integer catalogue-index vectors and must not call
        ``continuous_to_integer_candidate``. Examples: Optuna ``suggest_int``,
        Pymoo integer operators, SciPy DE ``integrality=True``, and integer
        sampling baselines. Integer-valued floats (for example ``3.0``) may be
        coerced via ``validate_integer_candidate``.
    CONTINUOUS_ROUNDED : str
        Continuous proposals must be converted with the shared
        ``continuous_to_integer_candidate`` policy (round-half-away-from-zero
        then clip).
    QUANTIZED_GRID : str
        Library-specific quantized float grid (for example Hyperopt). Does not
        use ``continuous_to_integer_candidate``; keep its own grid mapping and
        do not collapse this mode into ``INTEGER``.
    """

    INTEGER = "integer"
    CONTINUOUS_ROUNDED = "continuous_rounded"
    QUANTIZED_GRID = "quantized_grid"


class BudgetSemantics(str, Enum):
    """
    Meaning of ``max_evaluations`` for an algorithm.

    Attributes
    ----------
    HARD_IOH_EVALUATIONS : str
        Hard upper bound on counted IOH ``problem(...)`` calls.
    """

    HARD_IOH_EVALUATIONS = "hard_ioh_evaluations"


class ParameterKind(str, Enum):
    """
    Intent metadata for one hyperparameter.

    Path kinds record intent only. Existence/executability checks belong in
    the library constructor inside the worker environment, not at registry
    import time.

    Attributes
    ----------
    SCALAR : str
        Ordinary JSON scalar / nested JSON value.
    FILE_PATH : str
        Path to a readable file (validated later by the constructor).
    EXECUTABLE_PATH : str
        Path or bare name of an executable (validated later by the constructor).
    """

    SCALAR = "scalar"
    FILE_PATH = "file_path"
    EXECUTABLE_PATH = "executable_path"


class RegistryError(ValueError):
    """
    Raised for invalid registry metadata or lookup/validation failures.
    """


@dataclass(frozen=True)
class ParameterSpec:
    """
    One declared algorithm hyperparameter.

    Attributes
    ----------
    name : str
        Parameter key in TOML / task payloads.
    types : tuple of type
        Accepted Python types after JSON decoding. Boolean values are accepted
        only when ``bool`` is listed explicitly (they never satisfy ``int``).
    required : bool, optional
        When ``True`` and no default is set, the parameter must be present.
    default : JSON scalar or the internal no-default sentinel, optional
        Default inserted when the key is omitted. Leave unset (sentinel) when
        there is no default. Compare with :attr:`has_default`.
    minimum : float or None, optional
        Inclusive numeric lower bound.
    maximum : float or None, optional
        Inclusive numeric upper bound.
    choices : tuple of JSON scalars, optional
        Closed set of allowed values when non-empty.
    kind : ParameterKind, optional
        Intent metadata (scalar vs path/executable). Default is scalar.
    """

    name: str
    types: tuple[type, ...]
    required: bool = False
    default: Any = _NO_DEFAULT
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[JSONScalar, ...] = ()
    kind: ParameterKind = ParameterKind.SCALAR

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise RegistryError("ParameterSpec.name must be non-empty")
        if not self.types:
            raise RegistryError(f"ParameterSpec {self.name!r}: types must be non-empty")
        if not isinstance(self.kind, ParameterKind):
            raise RegistryError(
                f"ParameterSpec {self.name!r}: kind must be ParameterKind, "
                f"got {self.kind!r}"
            )

    @property
    def has_default(self) -> bool:
        """
        Return whether a default value was declared.

        Returns
        -------
        bool
            ``True`` when ``default`` is not the no-default sentinel.
        """
        return self.default is not _NO_DEFAULT


@dataclass(frozen=True)
class AlgorithmSpec:
    """
    Registry metadata for one algorithm token.

    Attributes
    ----------
    name : str
        Stable uppercase token used in TOML/manifests.
    library : str
        Library family label such as ``\"native\"`` or ``\"nlopt\"``.
    constructor_ref : str
        Lazy ``module:callable`` reference resolved only at construction time.
    parameters : tuple of ParameterSpec
        Accepted hyperparameters (excluding seed, which uses ``seed_policy``).
    seed_policy : SeedPolicy
        Required vs forbidden deterministic seed.
    dependency_extra : str or None
        Optional packaging extra name, or ``None`` when always available.
    candidate_mode : CandidateMode
        How candidates are mapped onto IOH integer indices.
    budget_semantics : BudgetSemantics
        Meaning of the evaluation budget.
    description : str
        Short human-readable description.
    batch_evaluations : bool, optional
        When ``True``, the algorithm may propose population batches; hard IOH
        budgeting must still be enforced. Default is ``False``.
    validator_ref : str or None, optional
        Optional lazy ``module:function`` cross-field validator. Receives the
        normalized parameter mapping and returns a new mapping.
    """

    name: str
    library: str
    constructor_ref: str
    parameters: tuple[ParameterSpec, ...]
    seed_policy: SeedPolicy
    dependency_extra: str | None
    candidate_mode: CandidateMode
    budget_semantics: BudgetSemantics
    description: str
    batch_evaluations: bool = False
    validator_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise RegistryError("AlgorithmSpec.name must be non-empty")
        _parse_module_callable_ref(
            self.constructor_ref,
            what=f"{self.name} constructor_ref",
        )
        if not isinstance(self.candidate_mode, CandidateMode):
            raise RegistryError(
                f"{self.name}: candidate_mode must be CandidateMode, "
                f"got {self.candidate_mode!r}"
            )
        if not isinstance(self.budget_semantics, BudgetSemantics):
            raise RegistryError(
                f"{self.name}: budget_semantics must be BudgetSemantics, "
                f"got {self.budget_semantics!r}"
            )
        if not isinstance(self.batch_evaluations, bool):
            raise RegistryError(
                f"{self.name}: batch_evaluations must be bool, "
                f"got {self.batch_evaluations!r}"
            )
        if self.validator_ref is not None:
            _parse_module_callable_ref(
                self.validator_ref,
                what=f"{self.name} validator_ref",
            )
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise RegistryError(f"{self.name}: duplicate parameter names in spec")


@dataclass(frozen=True)
class AlgorithmRegistry:
    """
    Immutable mapping from algorithm name tokens to :class:`AlgorithmSpec`.

    Parameters
    ----------
    _specs : mapping of str to AlgorithmSpec
        Name-keyed specifications.
    """

    _specs: Mapping[str, AlgorithmSpec]

    def get(self, name: str) -> AlgorithmSpec:
        """
        Return the spec for ``name``.

        Parameters
        ----------
        name : str
            Algorithm token.

        Returns
        -------
        AlgorithmSpec
            Matching specification.

        Raises
        ------
        RegistryError
            If the name is unknown.
        """
        try:
            return self._specs[name]
        except KeyError as exc:
            supported = ", ".join(self.names()) or "(none)"
            raise RegistryError(
                f"unknown algorithm {name!r}; supported: {supported}"
            ) from exc

    def names(self) -> list[str]:
        """
        Return sorted registered algorithm names.

        Returns
        -------
        list of str
            Sorted tokens.
        """
        return sorted(self._specs)

    def list_specs(self) -> tuple[AlgorithmSpec, ...]:
        """
        Return all specs in name order.

        Returns
        -------
        tuple of AlgorithmSpec
            Ordered specifications.
        """
        return tuple(self._specs[name] for name in self.names())


def create_registry(specs: Iterable[AlgorithmSpec]) -> AlgorithmRegistry:
    """
    Build an immutable registry, rejecting duplicate names.

    Parameters
    ----------
    specs : iterable of AlgorithmSpec
        Algorithm specifications.

    Returns
    -------
    AlgorithmRegistry
        Frozen registry.

    Raises
    ------
    RegistryError
        If two specs share a name.
    """
    mapping: dict[str, AlgorithmSpec] = {}
    for spec in specs:
        if spec.name in mapping:
            raise RegistryError(f"duplicate algorithm name {spec.name!r}")
        mapping[spec.name] = spec
    return AlgorithmRegistry(_specs=dict(mapping))


def _parse_module_callable_ref(ref: str, *, what: str) -> tuple[str, str]:
    """
    Parse a ``module:callable`` reference.

    Parameters
    ----------
    ref : str
        Import reference.
    what : str
        Label used in error messages.

    Returns
    -------
    tuple of str
        ``(module_name, attr_name)``.

    Raises
    ------
    RegistryError
        If the reference is malformed.
    """
    if not isinstance(ref, str) or ":" not in ref:
        raise RegistryError(
            f"malformed {what} {ref!r}; expected 'module:callable'"
        )
    module_name, _, attr_name = ref.partition(":")
    if not module_name.strip() or not attr_name.strip():
        raise RegistryError(
            f"malformed {what} {ref!r}; expected 'module:callable'"
        )
    if ":" in attr_name:
        raise RegistryError(
            f"malformed {what} {ref!r}; expected exactly one 'module:callable' pair"
        )
    if "." in attr_name:
        raise RegistryError(
            f"malformed {what} {ref!r}; callable name must not contain '.'"
        )
    return module_name, attr_name


def _type_allowed(value: Any, allowed: tuple[type, ...]) -> bool:
    """
    Return whether ``value`` matches one of ``allowed`` without bool/int bleed.

    Parameters
    ----------
    value : Any
        Candidate value.
    allowed : tuple of type
        Accepted types.

    Returns
    -------
    bool
        ``True`` when the value is acceptable.
    """
    if value is None:
        return type(None) in allowed
    if isinstance(value, bool):
        return bool in allowed
    return any(isinstance(value, typ) for typ in allowed if typ not in (bool, type(None)))


def _materialize_and_validate_parameters(
    spec: AlgorithmSpec,
    parameters: Mapping[str, Any],
) -> dict[str, JSONValue]:
    """
    Insert defaults and validate each declared parameter.

    Parameters
    ----------
    spec : AlgorithmSpec
        Algorithm metadata.
    parameters : mapping
        Declared hyperparameters (may omit defaults).

    Returns
    -------
    dict
        New normalized parameter mapping.

    Raises
    ------
    RegistryError
        If unknown keys, missing required fields, or type/range/choice checks
        fail.
    """
    allowed = {parameter.name: parameter for parameter in spec.parameters}
    unknown = sorted(set(parameters) - set(allowed))
    if unknown:
        raise RegistryError(
            f"{spec.name}: unknown parameters {unknown}; "
            f"allowed: {sorted(allowed)}"
        )

    out: dict[str, JSONValue] = {}
    for name, parameter in allowed.items():
        if name in parameters:
            value: Any = parameters[name]
        elif parameter.has_default:
            value = parameter.default
        elif parameter.required:
            raise RegistryError(
                f"{spec.name}: missing required parameter {name!r}"
            )
        else:
            continue

        if not _type_allowed(value, parameter.types):
            raise RegistryError(
                f"{spec.name}: parameter {name!r} must be one of "
                f"{tuple(t.__name__ for t in parameter.types)}, got {value!r}"
            )

        if parameter.choices and value not in parameter.choices:
            raise RegistryError(
                f"{spec.name}: parameter {name!r} must be one of "
                f"{list(parameter.choices)}, got {value!r}"
            )

        if parameter.minimum is not None or parameter.maximum is not None:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RegistryError(
                    f"{spec.name}: parameter {name!r} must be numeric for range checks"
                )
            number = float(value)
            if parameter.minimum is not None and number < parameter.minimum:
                raise RegistryError(
                    f"{spec.name}: parameter {name!r} must be >= {parameter.minimum}, "
                    f"got {value!r}"
                )
            if parameter.maximum is not None and number > parameter.maximum:
                raise RegistryError(
                    f"{spec.name}: parameter {name!r} must be <= {parameter.maximum}, "
                    f"got {value!r}"
                )

        out[name] = value  # type: ignore[assignment]

    return out


def normalize_parameters(
    spec: AlgorithmSpec,
    parameters: Mapping[str, Any],
) -> dict[str, JSONValue]:
    """
    Validate and normalize hyperparameters for ``spec``.

    Does not mutate ``parameters``. Inserts declared defaults for omitted keys.
    When ``validator_ref`` is set, resolves and applies the cross-field
    validator, then revalidates the returned mapping without calling the
    validator again.

    Parameters
    ----------
    spec : AlgorithmSpec
        Algorithm metadata.
    parameters : mapping
        Declared hyperparameters (may omit defaults).

    Returns
    -------
    dict
        New normalized parameter mapping.

    Raises
    ------
    RegistryError
        If unknown keys, missing required fields, type/range/choice checks,
        or the optional cross-field validator fail.
    """
    if not isinstance(parameters, Mapping):
        raise RegistryError(f"{spec.name}: parameters must be a mapping")

    out = _materialize_and_validate_parameters(spec, parameters)
    if spec.validator_ref is None:
        return out

    validator = resolve_parameter_validator(spec.validator_ref)
    try:
        validated = validator(MappingProxyType(dict(out)))
    except RegistryError:
        raise
    except Exception as exc:
        raise RegistryError(
            f"{spec.name}: parameter validator {spec.validator_ref!r} failed: {exc}"
        ) from exc

    if not isinstance(validated, Mapping):
        raise RegistryError(
            f"{spec.name}: parameter validator must return a mapping, "
            f"got {type(validated)!r}"
        )

    return _materialize_and_validate_parameters(spec, dict(validated))


def validate_seed(spec: AlgorithmSpec, random_seed: int | None) -> None:
    """
    Enforce ``seed_policy`` and unsigned 32-bit bounds.

    Parameters
    ----------
    spec : AlgorithmSpec
        Algorithm metadata.
    random_seed : int or None
        Task seed from the planner/payload.

    Raises
    ------
    RegistryError
        If the seed violates the policy or range.
    """
    if spec.seed_policy == SeedPolicy.REQUIRED:
        if random_seed is None:
            raise RegistryError(
                f"{spec.name} requires a random_seed (unsigned 32-bit int)"
            )
        if not isinstance(random_seed, int) or isinstance(random_seed, bool):
            raise RegistryError(
                f"random_seed must be an unsigned 32-bit int, got {random_seed!r}"
            )
        if random_seed < 0 or random_seed > 0xFFFFFFFF:
            raise RegistryError(
                f"random_seed must be an unsigned 32-bit int, got {random_seed}"
            )
        return

    if spec.seed_policy == SeedPolicy.FORBIDDEN:
        if random_seed is not None:
            raise RegistryError(
                f"{spec.name} is seedless; random_seed must be null"
            )
        return

    raise RegistryError(f"{spec.name}: unknown seed_policy {spec.seed_policy!r}")


def resolve_constructor(constructor_ref: str) -> Constructor:
    """
    Lazily import ``module:callable`` and return the constructor.

    Parameters
    ----------
    constructor_ref : str
        Import reference such as ``\"pkg.mod:build_algo\"``.

    Returns
    -------
    callable
        Constructor ``(parameters, random_seed) -> IOHAlgorithm``.

    Raises
    ------
    RegistryError
        If the reference is malformed or cannot be imported/resolved.
    """
    module_name, attr_name = _parse_module_callable_ref(
        constructor_ref,
        what="constructor_ref",
    )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RegistryError(
            f"cannot import constructor module {module_name!r}: {exc}"
        ) from exc
    try:
        ctor = getattr(module, attr_name)
    except AttributeError as exc:
        raise RegistryError(
            f"constructor {attr_name!r} not found in module {module_name!r}"
        ) from exc
    if not callable(ctor):
        raise RegistryError(
            f"constructor_ref {constructor_ref!r} did not resolve to a callable"
        )
    return ctor  # type: ignore[return-value]


def resolve_parameter_validator(validator_ref: str) -> ParameterValidator:
    """
    Lazily import ``module:callable`` and return a parameter validator.

    Parameters
    ----------
    validator_ref : str
        Import reference such as ``\"pkg.mod:validate_params\"``.

    Returns
    -------
    callable
        Validator ``(parameters) -> Mapping``.

    Raises
    ------
    RegistryError
        If the reference is malformed or cannot be imported/resolved.
    """
    module_name, attr_name = _parse_module_callable_ref(
        validator_ref,
        what="validator_ref",
    )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RegistryError(
            f"cannot import validator module {module_name!r}: {exc}"
        ) from exc
    try:
        validator = getattr(module, attr_name)
    except AttributeError as exc:
        raise RegistryError(
            f"validator {attr_name!r} not found in module {module_name!r}"
        ) from exc
    if not callable(validator):
        raise RegistryError(
            f"validator_ref {validator_ref!r} did not resolve to a callable"
        )
    return validator  # type: ignore[return-value]


def construct_algorithm(
    registry: AlgorithmRegistry,
    name: str,
    parameters: Mapping[str, Any],
    *,
    random_seed: int | None = None,
) -> IOHAlgorithm:
    """
    Validate parameters/seed and construct a fresh algorithm instance.

    Parameters
    ----------
    registry : AlgorithmRegistry
        Algorithm registry.
    name : str
        Algorithm token.
    parameters : mapping
        Declared hyperparameters.
    random_seed : int or None, optional
        Task seed.

    Returns
    -------
    IOHAlgorithm
        New algorithm instance.
    """
    spec = registry.get(name)
    normalized = normalize_parameters(spec, parameters)
    validate_seed(spec, random_seed)
    ctor = resolve_constructor(spec.constructor_ref)
    return ctor(normalized, random_seed)


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    """
    Return the production registry spec for ``name``.

    Parameters
    ----------
    name : str
        Algorithm token.

    Returns
    -------
    AlgorithmSpec
        Matching production specification.

    Raises
    ------
    RegistryError
        If the name is unknown.
    """
    return ALGORITHM_REGISTRY.get(name)


def list_algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    """
    Return all production algorithm specs in name order.

    Returns
    -------
    tuple of AlgorithmSpec
        Ordered production specifications.
    """
    return ALGORITHM_REGISTRY.list_specs()


def normalize_algorithm_parameters(
    name: str,
    parameters: Mapping[str, Any],
) -> dict[str, JSONValue]:
    """
    Normalize hyperparameters against the production registry.

    Parameters
    ----------
    name : str
        Algorithm token.
    parameters : mapping
        Declared hyperparameters.

    Returns
    -------
    dict
        New normalized parameter mapping.

    Raises
    ------
    RegistryError
        If the name or parameters are invalid.
    """
    return normalize_parameters(ALGORITHM_REGISTRY.get(name), parameters)


def _max_evaluations_spec() -> ParameterSpec:
    """
    Return the shared required ``max_evaluations`` parameter spec.

    Returns
    -------
    ParameterSpec
        Positive integer evaluation budget.
    """
    return ParameterSpec(
        name="max_evaluations",
        types=(int,),
        required=True,
        minimum=1,
    )


def _initial_design_path_spec() -> ParameterSpec:
    """
    Planner-injected InitialDesign path.

    Optional and without a default so a direct factory call that omits it
    fails closed. TOML must not set it.
    """
    return ParameterSpec(
        name="initial_design_path",
        types=(str,),
        required=False,
        kind=ParameterKind.FILE_PATH,
    )


def _k_elite_ams_parameter_specs(
    *,
    warm_start: bool,
) -> tuple[ParameterSpec, ...]:
    """Return the shared cold/warm k-elite AMS parameter schema."""
    parameters = [
        _max_evaluations_spec(),
        ParameterSpec(
            name="archive_size",
            types=(int,),
            required=False,
            default=10,
            minimum=1,
        ),
        ParameterSpec(
            name="n_parents",
            types=(int,),
            required=False,
            default=5,
            minimum=1,
        ),
        ParameterSpec(
            name="mutation_probability",
            types=(int, float),
            required=False,
            default=0.05,
            minimum=0.0,
            maximum=1.0,
        ),
        ParameterSpec(
            name="random_restart_probability",
            types=(int, float),
            required=False,
            default=0.10,
            minimum=0.0,
            maximum=1.0,
        ),
    ]
    if warm_start:
        parameters.append(_initial_design_path_spec())
    return tuple(parameters)


def _population_basin_hopping_parameter_specs(
    *,
    warm_start: bool,
) -> tuple[ParameterSpec, ...]:
    """Return the common cold/warm PBH parameter schema."""
    parameters = (
        _max_evaluations_spec(),
        ParameterSpec(
            name="population_size",
            types=(int,),
            required=False,
            default=8,
            minimum=1,
        ),
        ParameterSpec(
            name="perturbation_fraction",
            types=(int, float),
            required=False,
            default=0.15,
            maximum=1.0,
        ),
        ParameterSpec(
            name="dcut_factor",
            types=(int, float),
            required=False,
            default=0.5,
        ),
        ParameterSpec(
            name="initial_sampler",
            types=(str,),
            required=False,
            default="sobol",
            choices=("sobol", "uniform"),
        ),
    )
    if warm_start:
        return (*parameters, _initial_design_path_spec())
    return parameters


def _discrete_mlsl_parameter_specs(
    *,
    warm_start: bool,
) -> tuple[ParameterSpec, ...]:
    parameters = (
        _max_evaluations_spec(),
        ParameterSpec(
            name="batch_size",
            types=(int,),
            required=False,
            default=128,
            minimum=1,
        ),
        ParameterSpec(
            name="reduction_fraction",
            types=(int, float),
            required=False,
            default=0.10,
            maximum=1.0,
        ),
        ParameterSpec(
            name="sigma",
            types=(int, float),
            required=False,
            default=2.0,
        ),
        ParameterSpec(
            name="sampler",
            types=(str,),
            required=False,
            default="sobol",
            choices=("sobol", "uniform"),
        ),
    )
    if warm_start:
        return (*parameters, _initial_design_path_spec())
    return parameters


# Empty registry retained for ALG1 tests that assert an empty baseline.
EMPTY_REGISTRY = create_registry(())

# Production registry for workers/factory. Constructors load lazily.
ALGORITHM_REGISTRY = create_registry(
    (
        AlgorithmSpec(
            name="RANDOM_SEARCH",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:build_random_search"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description="Very simple random search over integer index bounds",
        ),
        AlgorithmSpec(
            name="LATIN_HYPERCUBE",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:build_latin_hypercube"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description="Latin hypercube sampling baseline in integer index space",
        ),
        AlgorithmSpec(
            name="DISCRETE_MLSL_PEN_SI_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:build_discrete_mlsl"
            ),
            parameters=_discrete_mlsl_parameter_specs(warm_start=False),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Discrete multi-level single-linkage with strict L1 local search"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_discrete_mlsl_parameters"
            ),
        ),
        AlgorithmSpec(
            name="SOBOL_MULTISTART_PEN_SI_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:build_sobol_multistart"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Scrambled Sobol multistart. Every new sample starts the "
                "shared strict-improvement local search. No reduction and "
                "no critical-radius filter."
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="SOBOL_MULTISTART_PEN_EQU_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_sobol_multistart_equ_desc"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Scrambled Sobol multistart. Every new sample starts the "
                "shared equal-descent local search. No reduction and no "
                "critical-radius filter."
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="SOBOL_PEN_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:build_sobol_sample"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Scrambled Sobol sampling with no local search. The same "
                "stream and duplicate-projection rule as Sobol multistart."
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="DISCRETE_MLSL_DEB_EQU_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_discrete_mlsl_deb_equ_desc"
            ),
            parameters=_discrete_mlsl_parameter_specs(warm_start=False),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Constraint-aware discrete MLSL with Deb reduction, linkage, "
                "and equal-descent L1 local search"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_discrete_mlsl_deb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="DISCRETE_MLSL_DEB_EQU_DESC_WARM_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_discrete_mlsl_deb_equ_desc_warm_start"
            ),
            parameters=_discrete_mlsl_parameter_specs(warm_start=True),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Constraint-aware equal-descent discrete MLSL from a "
                "Deb-ranked InitialDesign prefix"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_discrete_mlsl_deb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="DISCRETE_MLSL_DEB_SI_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_discrete_mlsl_deb"
            ),
            parameters=_discrete_mlsl_parameter_specs(warm_start=False),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Constraint-aware discrete MLSL with Deb reduction, linkage, "
                "and strict L1 local search"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_discrete_mlsl_deb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="DISCRETE_MLSL_DEB_SI_DESC_WARM_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_discrete_mlsl_deb_warm_start"
            ),
            parameters=_discrete_mlsl_parameter_specs(warm_start=True),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Constraint-aware discrete MLSL with a ranked InitialDesign "
                "first batch"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_discrete_mlsl_deb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="DISCRETE_MLSL_PEN_SI_DESC_WARM_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_discrete_mlsl_warm_start"
            ),
            parameters=_discrete_mlsl_parameter_specs(warm_start=True),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Discrete MLSL whose first batch is a ranked InitialDesign prefix"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_discrete_mlsl_parameters"
            ),
        ),
        AlgorithmSpec(
            name="DISCRETE_MLSL_PEN_EQU_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_discrete_mlsl_equ_desc"
            ),
            parameters=_discrete_mlsl_parameter_specs(warm_start=False),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Discrete multi-level single-linkage with equal-descent L1 local search"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_discrete_mlsl_parameters"
            ),
        ),
        AlgorithmSpec(
            name="DISCRETE_MLSL_PEN_EQU_DESC_WARM_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_discrete_mlsl_equ_desc_warm_start"
            ),
            parameters=_discrete_mlsl_parameter_specs(warm_start=True),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Equal-descent discrete MLSL from a ranked InitialDesign prefix"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_discrete_mlsl_parameters"
            ),
        ),
        AlgorithmSpec(
            name="K_ELITE_AMS_PEN_SI_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:build_k_elite_ams"
            ),
            parameters=_k_elite_ams_parameter_specs(warm_start=False),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "k-elite adaptive multi-start on certified L1 local minima"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.k_elite_ams:"
                "validate_k_elite_ams_parameters"
            ),
        ),
        AlgorithmSpec(
            name="K_ELITE_AMS_DEB_EQU_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_k_elite_ams_deb_equ_desc"
            ),
            parameters=_k_elite_ams_parameter_specs(warm_start=False),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "k-elite adaptive multi-start with equal-descent local search "
                "under Deb ordering"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.k_elite_ams:"
                "validate_k_elite_ams_deb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="K_ELITE_AMS_DEB_EQU_DESC_WARM_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_k_elite_ams_deb_equ_desc_warm_start"
            ),
            parameters=_k_elite_ams_parameter_specs(warm_start=True),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Equal-descent k-elite adaptive multi-start from a "
                "Deb-ranked InitialDesign prefix"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.k_elite_ams:"
                "validate_k_elite_ams_deb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="K_ELITE_AMS_DEB_SI_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:build_k_elite_ams_deb"
            ),
            parameters=_k_elite_ams_parameter_specs(warm_start=False),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "k-elite adaptive multi-start on certified L1 minima "
                "under Deb ordering"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.k_elite_ams:"
                "validate_k_elite_ams_deb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="K_ELITE_AMS_DEB_SI_DESC_WARM_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_k_elite_ams_deb_warm_start"
            ),
            parameters=_k_elite_ams_parameter_specs(warm_start=True),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Deb k-elite adaptive multi-start from a y_p-ranked "
                "InitialDesign prefix"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.k_elite_ams:"
                "validate_k_elite_ams_deb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="K_ELITE_AMS_PEN_SI_DESC_WARM_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_k_elite_ams_warm_start"
            ),
            parameters=_k_elite_ams_parameter_specs(warm_start=True),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "k-elite adaptive multi-start from a ranked InitialDesign prefix"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.k_elite_ams:"
                "validate_k_elite_ams_parameters"
            ),
        ),
        AlgorithmSpec(
            name="K_ELITE_AMS_PEN_EQU_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:build_k_elite_ams_equ_desc"
            ),
            parameters=_k_elite_ams_parameter_specs(warm_start=False),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "k-elite adaptive multi-start with equal-descent local search"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.k_elite_ams:"
                "validate_k_elite_ams_parameters"
            ),
        ),
        AlgorithmSpec(
            name="K_ELITE_AMS_PEN_EQU_DESC_WARM_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_k_elite_ams_equ_desc_warm_start"
            ),
            parameters=_k_elite_ams_parameter_specs(warm_start=True),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Equal-descent k-elite adaptive multi-start from a ranked "
                "InitialDesign prefix"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.k_elite_ams:"
                "validate_k_elite_ams_parameters"
            ),
        ),
        AlgorithmSpec(
            name="POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_population_basin_hopping"
            ),
            parameters=_population_basin_hopping_parameter_specs(
                warm_start=False
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Discrete population basin hopping on certified L1 local optima"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_population_basin_hopping_parameters"
            ),
        ),
        AlgorithmSpec(
            name="POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_population_basin_hopping_warm_start"
            ),
            parameters=_population_basin_hopping_parameter_specs(
                warm_start=True
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Discrete population basin hopping from a ranked InitialDesign prefix"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_population_basin_hopping_parameters"
            ),
        ),
        AlgorithmSpec(
            name="POPULATION_BASIN_HOPPING_PEN_EQU_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_population_basin_hopping_equ_desc"
            ),
            parameters=_population_basin_hopping_parameter_specs(
                warm_start=False
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Discrete population basin hopping with equal-descent local search"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_population_basin_hopping_parameters"
            ),
        ),
        AlgorithmSpec(
            name="POPULATION_BASIN_HOPPING_PEN_EQU_DESC_WARM_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_population_basin_hopping_equ_desc_warm_start"
            ),
            parameters=_population_basin_hopping_parameter_specs(
                warm_start=True
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Equal-descent population basin hopping from a ranked "
                "InitialDesign prefix"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_population_basin_hopping_parameters"
            ),
        ),
        AlgorithmSpec(
            name="POPULATION_BASIN_HOPPING_DEB_EQU_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_population_basin_hopping_deb_equ_desc"
            ),
            parameters=_population_basin_hopping_parameter_specs(
                warm_start=False
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Constraint-aware population basin hopping with "
                "equal-descent local search"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_population_basin_hopping_deb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_population_basin_hopping_deb_equ_desc_warm_start"
            ),
            parameters=_population_basin_hopping_parameter_specs(
                warm_start=True
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Constraint-aware equal-descent population basin hopping "
                "from a Deb-ranked InitialDesign prefix"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_population_basin_hopping_deb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_population_basin_hopping_deb"
            ),
            parameters=_population_basin_hopping_parameter_specs(
                warm_start=False
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Constraint-aware population basin hopping with strict "
                "Deb-ordered L1 local search"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_population_basin_hopping_deb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
            library="native",
            constructor_ref=(
                "core.ioh_core.algorithms.native.factory:"
                "build_population_basin_hopping_deb_warm_start"
            ),
            parameters=_population_basin_hopping_parameter_specs(
                warm_start=True
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Constraint-aware population basin hopping from a ranked "
                "InitialDesign prefix"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.native.discrete_search:"
                "validate_population_basin_hopping_deb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="SCIPY_DIFFERENTIAL_EVOLUTION",
            library="scipy",
            constructor_ref=(
                "core.ioh_core.algorithms.scipy.factory:"
                "build_scipy_differential_evolution"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "SciPy differential evolution with integer integrality on "
                "catalogue indices"
            ),
        ),
        AlgorithmSpec(
            name="SCIPY_DIFFERENTIAL_EVOLUTION_WARM_START",
            library="scipy",
            constructor_ref=(
                "core.ioh_core.algorithms.scipy.factory:"
                "build_scipy_differential_evolution_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "SciPy differential evolution from a ranked InitialDesign prefix"
            ),
        ),
        AlgorithmSpec(
            name="SCIPY_COBYLA",
            library="scipy",
            constructor_ref=(
                "core.ioh_core.algorithms.scipy.factory:build_scipy_cobyla"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "SciPy COBYLA on continuous-rounded catalogue indices "
                "(constrained y + signed g=η−1; seeded random start)"
            ),
        ),
        AlgorithmSpec(
            name="SCIPY_COBYLA_WARM_START",
            library="scipy",
            constructor_ref=(
                "core.ioh_core.algorithms.scipy.factory:"
                "build_scipy_cobyla_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.FORBIDDEN,
            dependency_extra=None,
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "SciPy COBYLA from rank 1 of a ranked InitialDesign "
                "(constrained y + signed g=η−1)"
            ),
        ),
        AlgorithmSpec(
            name="SCIPY_DIRECT_L",
            library="scipy",
            constructor_ref=(
                "core.ioh_core.algorithms.scipy.factory:build_scipy_direct_l"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.FORBIDDEN,
            dependency_extra=None,
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "SciPy locally-biased DIRECT-L on a continuous-to-integer "
                "rounded surface"
            ),
        ),
        AlgorithmSpec(
            name="SCIPY_DUAL_ANNEALING",
            library="scipy",
            constructor_ref=(
                "core.ioh_core.algorithms.scipy.factory:"
                "build_scipy_dual_annealing"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "SciPy dual annealing (global phase only; local search "
                "disabled) on continuous-rounded catalogue indices; "
                "minimizes y_p; no initial point supplied"
            ),
        ),
        AlgorithmSpec(
            name="SCIPY_DUAL_ANNEALING_WARM_START",
            library="scipy",
            constructor_ref=(
                "core.ioh_core.algorithms.scipy.factory:"
                "build_scipy_dual_annealing_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "SciPy dual annealing (global phase only; local search "
                "disabled) from rank 1 of a ranked InitialDesign; "
                "minimizes y_p; seeded"
            ),
        ),
        AlgorithmSpec(
            name="NLOPT_DIRECT_L",
            library="nlopt",
            constructor_ref=(
                "core.ioh_core.algorithms.nlopt.factory:build_direct_l"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.FORBIDDEN,
            dependency_extra=None,
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "locally biased variant of DIviding RECTangles (DIRECT) "
                "algorithm for global optimization"
            ),
        ),
        AlgorithmSpec(
            name="NLOPT_SBPLX",
            library="nlopt",
            constructor_ref=(
                "core.ioh_core.algorithms.nlopt.factory:build_sbplx"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra=None,
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "NLopt SBPLX compatibility search on a rounded integer "
                "surface (seeded random start)"
            ),
        ),
        AlgorithmSpec(
            name="NLOPT_SBPLX_WARM_START",
            library="nlopt",
            constructor_ref=(
                "core.ioh_core.algorithms.nlopt.factory:build_sbplx_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.FORBIDDEN,
            dependency_extra=None,
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "NLopt SBPLX from rank 1 of a ranked InitialDesign "
                "on a rounded integer surface"
            ),
        ),
        AlgorithmSpec(
            name="LEAP_GA",
            library="leap",
            constructor_ref=(
                "core.ioh_core.algorithms.leap.factory:build_leap_ga"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="population_size",
                    types=(int,),
                    required=False,
                    default=50,
                    minimum=2,
                ),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-leap",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "LEAP generational genetic algorithm with integer operators"
            ),
            batch_evaluations=True,
        ),
        AlgorithmSpec(
            name="LEAP_GA_WARM_START",
            library="leap",
            constructor_ref=(
                "core.ioh_core.algorithms.leap.factory:build_leap_ga_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="population_size",
                    types=(int,),
                    required=False,
                    default=50,
                    minimum=2,
                ),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-leap",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "LEAP generational GA from a ranked InitialDesign prefix"
            ),
            batch_evaluations=True,
        ),
        AlgorithmSpec(
            name="CMAES_CATCMAWM",
            library="cmaes",
            constructor_ref=(
                "core.ioh_core.algorithms.cmaes.factory:build_cmaes_catcmawm"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="population_size",
                    types=(int,),
                    required=False,
                    default=50,
                    minimum=2,
                ),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-cmaes",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "CatCMAwM from cmaes on integer catalogue indices "
                "(unconstrained y_p)"
            ),
            batch_evaluations=True,
        ),
        AlgorithmSpec(
            name="CMAES_CATCMAWM_WARM_START",
            library="cmaes",
            constructor_ref=(
                "core.ioh_core.algorithms.cmaes.factory:"
                "build_cmaes_catcmawm_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="population_size",
                    types=(int,),
                    required=False,
                    default=50,
                    minimum=2,
                ),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-cmaes",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "CatCMAwM with rank 1 as the initial mean and the cold "
                "box-centre covariance (unconstrained y_p)"
            ),
            batch_evaluations=True,
        ),
        AlgorithmSpec(
            name="PYMOO_GA",
            library="pymoo",
            constructor_ref=(
                "core.ioh_core.algorithms.pymoo.factory:build_pymoo_ga"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="population_size",
                    types=(int,),
                    required=False,
                    default=50,
                    minimum=2,
                ),
                ParameterSpec(
                    name="eliminate_duplicates",
                    types=(bool,),
                    required=False,
                    default=True,
                ),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-pymoo",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description="Pymoo genetic algorithm with integer operators",
            batch_evaluations=True,
        ),
        AlgorithmSpec(
            name="PYMOO_GA_WARM_START",
            library="pymoo",
            constructor_ref=(
                "core.ioh_core.algorithms.pymoo.factory:build_pymoo_ga_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="population_size",
                    types=(int,),
                    required=False,
                    default=50,
                    minimum=2,
                ),
                ParameterSpec(
                    name="eliminate_duplicates",
                    types=(bool,),
                    required=False,
                    default=True,
                ),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-pymoo",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Pymoo genetic algorithm from a ranked InitialDesign prefix"
            ),
            batch_evaluations=True,
        ),
        AlgorithmSpec(
            name="PYMOO_GA_CONSTRAINED",
            library="pymoo",
            constructor_ref=(
                "core.ioh_core.algorithms.pymoo.factory:"
                "build_pymoo_ga_constrained"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="population_size",
                    types=(int,),
                    required=False,
                    default=50,
                    minimum=2,
                ),
                ParameterSpec(
                    name="eliminate_duplicates",
                    types=(bool,),
                    required=False,
                    default=True,
                ),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-pymoo",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Pymoo GA minimizing unpenalized y with native inequality "
                "constraints (signed g=η−1)"
            ),
            batch_evaluations=True,
        ),
        AlgorithmSpec(
            name="PYMOO_GA_CONSTRAINED_WARM_START",
            library="pymoo",
            constructor_ref=(
                "core.ioh_core.algorithms.pymoo.factory:"
                "build_pymoo_ga_constrained_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="population_size",
                    types=(int,),
                    required=False,
                    default=50,
                    minimum=2,
                ),
                ParameterSpec(
                    name="eliminate_duplicates",
                    types=(bool,),
                    required=False,
                    default=True,
                ),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-pymoo",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Pymoo constrained GA from a ranked InitialDesign prefix "
                "(initial population; signed g=η−1)"
            ),
            batch_evaluations=True,
        ),
        AlgorithmSpec(
            name="PYSTARPSO_INTEGER_PSO",
            library="starpso",
            constructor_ref=(
                "core.ioh_core.algorithms.starpso.factory:"
                "build_pystarpso_integer_pso"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="n_pop",
                    types=(int,),
                    required=False,
                    default=50,
                    minimum=2,
                ),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-starpso",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "StarPSO IntegerPSO on integer catalogue indices "
                "(unconstrained y_p)"
            ),
            batch_evaluations=True,
        ),
        AlgorithmSpec(
            name="PYSTARPSO_INTEGER_PSO_WARM_START",
            library="starpso",
            constructor_ref=(
                "core.ioh_core.algorithms.starpso.factory:"
                "build_pystarpso_integer_pso_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="n_pop",
                    types=(int,),
                    required=False,
                    default=50,
                    minimum=2,
                ),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-starpso",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "StarPSO IntegerPSO from a ranked InitialDesign prefix "
                "(unconstrained y_p)"
            ),
            batch_evaluations=True,
        ),
        AlgorithmSpec(
            name="NEVERGRAD_METAMODEL",
            library="nevergrad",
            constructor_ref=(
                "core.ioh_core.algorithms.nevergrad.factory:"
                "build_nevergrad_metamodel"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-nevergrad",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Nevergrad MetaModel on discrete catalogue indices "
                "(unconstrained y_p)"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="NEVERGRAD_METAMODEL_WARM_START",
            library="nevergrad",
            constructor_ref=(
                "core.ioh_core.algorithms.nevergrad.factory:"
                "build_nevergrad_metamodel_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-nevergrad",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Nevergrad MetaModel from a ranked InitialDesign prefix"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="NEVERGRAD_METAMODEL_ONEPLUSONE",
            library="nevergrad",
            constructor_ref=(
                "core.ioh_core.algorithms.nevergrad.factory:"
                "build_nevergrad_metamodel_oneplusone"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-nevergrad",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Nevergrad MetaModelOnePlusOne on discrete catalogue indices "
                "(unconstrained y_p)"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="NEVERGRAD_METAMODEL_ONEPLUSONE_WARM_START",
            library="nevergrad",
            constructor_ref=(
                "core.ioh_core.algorithms.nevergrad.factory:"
                "build_nevergrad_metamodel_oneplusone_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-nevergrad",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Nevergrad MetaModelOnePlusOne from a ranked InitialDesign prefix"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="HYPEROPT_TPE",
            library="hyperopt",
            constructor_ref=(
                "core.ioh_core.algorithms.hyperopt.factory:build_hyperopt_tpe"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-hyperopt",
            candidate_mode=CandidateMode.QUANTIZED_GRID,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Hyperopt tree-structured Parzen estimator on an integer grid"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="HYPEROPT_TPE_WARM_START",
            library="hyperopt",
            constructor_ref=(
                "core.ioh_core.algorithms.hyperopt.factory:"
                "build_hyperopt_tpe_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-hyperopt",
            candidate_mode=CandidateMode.QUANTIZED_GRID,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Hyperopt TPE from a ranked InitialDesign prefix"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="OPTUNA_TPE",
            library="optuna",
            constructor_ref=(
                "core.ioh_core.algorithms.optuna.factory:build_optuna_tpe"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-optuna",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Optuna TPE on integer catalogue indices "
                "(constrained y + signed g=η−1)"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="OPTUNA_TPE_WARM_START",
            library="optuna",
            constructor_ref=(
                "core.ioh_core.algorithms.optuna.factory:"
                "build_optuna_tpe_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-optuna",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Optuna TPE from a ranked InitialDesign prefix"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="RBFOPT",
            library="rbfopt",
            constructor_ref=(
                "core.ioh_core.algorithms.rbfopt.factory:build_rbfopt"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="ga_base_population_size",
                    types=(int,),
                    required=False,
                    default=400,
                    minimum=2,
                ),
                ParameterSpec(
                    name="minlp_solver_path",
                    types=(str,),
                    required=True,
                    kind=ParameterKind.EXECUTABLE_PATH,
                ),
                ParameterSpec(
                    name="nlp_solver_path",
                    types=(str,),
                    required=True,
                    kind=ParameterKind.EXECUTABLE_PATH,
                ),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-rbfopt",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description="RBFOpt surrogate search on native integer variables",
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="RBFOPT_WARM_START",
            library="rbfopt",
            constructor_ref=(
                "core.ioh_core.algorithms.rbfopt.factory:build_rbfopt_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="ga_base_population_size",
                    types=(int,),
                    required=False,
                    default=400,
                    minimum=2,
                ),
                ParameterSpec(
                    name="minlp_solver_path",
                    types=(str,),
                    required=True,
                    kind=ParameterKind.EXECUTABLE_PATH,
                ),
                ParameterSpec(
                    name="nlp_solver_path",
                    types=(str,),
                    required=True,
                    kind=ParameterKind.EXECUTABLE_PATH,
                ),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-rbfopt",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description="RBFOpt from a ranked InitialDesign prefix",
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="SMT_EGO",
            library="smt",
            constructor_ref=(
                "core.ioh_core.algorithms.smt.factory:build_smt_ego"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="n_doe",
                    types=(int,),
                    required=False,
                    minimum=1,
                ),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-smt",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "SMT EGO (EI) on integer catalogue indices "
                "(unconstrained y_p)"
            ),
            batch_evaluations=True,
            validator_ref=(
                "core.ioh_core.algorithms.smt.factory:"
                "validate_smt_ego_parameters"
            ),
        ),
        AlgorithmSpec(
            name="SMT_EGO_WARM_START",
            library="smt",
            constructor_ref=(
                "core.ioh_core.algorithms.smt.factory:build_smt_ego_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="n_doe",
                    types=(int,),
                    required=False,
                    minimum=1,
                ),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-smt",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description="SMT EGO from a ranked InitialDesign DOE",
            batch_evaluations=True,
            validator_ref=(
                "core.ioh_core.algorithms.smt.factory:"
                "validate_smt_ego_warm_start_parameters"
            ),
        ),
        AlgorithmSpec(
            name="BLACKBOXOPT_DYCORS",
            library="blackboxoptim",
            constructor_ref=(
                "core.ioh_core.algorithms.blackboxoptim.factory:"
                "build_blackboxopt_dycors"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-blackboxoptim",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "blackboxoptim DYCORS on integer catalogue indices "
                "(unconstrained y_p)"
            ),
            batch_evaluations=True,
        ),
        AlgorithmSpec(
            name="BLACKBOXOPT_GOSAC",
            library="blackboxoptim",
            constructor_ref=(
                "core.ioh_core.algorithms.blackboxoptim.factory:"
                "build_blackboxopt_gosac"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-blackboxoptim",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "blackboxoptim GOSAC on integer catalogue indices "
                "(constrained-only; unpenalized y with signed g=η−1)"
            ),
            batch_evaluations=True,
        ),
        AlgorithmSpec(
            name="BLACKBOXOPT_MSRS",
            library="blackboxoptim",
            constructor_ref=(
                "core.ioh_core.algorithms.blackboxoptim.factory:"
                "build_blackboxopt_msrs"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-blackboxoptim",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "blackboxoptim multistart MSRS on integer catalogue indices "
                "(unconstrained y_p)"
            ),
            batch_evaluations=True,
        ),
        AlgorithmSpec(
            name="ALTBACKEN_SA",
            library="altbacken",
            constructor_ref=(
                "core.ioh_core.algorithms.altbacken.factory:"
                "build_altbacken_sa"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-altbacken",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Altbacken SA with catalogue-reset neighbourhood "
                "(unconstrained y_p)"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="ALTBACKEN_SA_WARM_START",
            library="altbacken",
            constructor_ref=(
                "core.ioh_core.algorithms.altbacken.factory:"
                "build_altbacken_sa_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-altbacken",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Altbacken SA from rank 1 of a ranked InitialDesign "
                "(unconstrained y_p)"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="DFLINT_BBOA",
            library="dfl",
            constructor_ref=(
                "core.ioh_core.algorithms.dfl.factory:"
                "build_dflint_bboa"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-dfl",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "DFLINT BBOA on integer catalogue indices "
                "(constrained y + signed g=η−1)"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="DFLINT_BBOA_WARM_START",
            library="dfl",
            constructor_ref=(
                "core.ioh_core.algorithms.dfl.factory:"
                "build_dflint_bboa_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.FORBIDDEN,
            dependency_extra="algorithms-dfl",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "DFLINT BBOA from rank 1 of a ranked InitialDesign "
                "(constrained y + signed g=η−1)"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="PYNOMAD_MADS_PB",
            library="nomad",
            constructor_ref=(
                "core.ioh_core.algorithms.nomad.factory:"
                "build_pynomad_mads_pb"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-nomad",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "PyNomad MADS Progressive Barrier on integer indices "
                "(constrained y + signed g=η−1)"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="PYNOMAD_MADS_PB_WARM_START",
            library="nomad",
            constructor_ref=(
                "core.ioh_core.algorithms.nomad.factory:"
                "build_pynomad_mads_pb_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-nomad",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "PyNomad MADS-PB from rank 1 of a ranked InitialDesign "
                "(signed g=η−1)"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="PYNOMAD_VNS_MADS_PB",
            library="nomad",
            constructor_ref=(
                "core.ioh_core.algorithms.nomad.factory:"
                "build_pynomad_vns_mads_pb"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="vns_mads_search_trigger",
                    types=(int, float),
                    required=False,
                    default=0.75,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-nomad",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "PyNomad MADS Progressive Barrier with VNS search on "
                "integer indices (constrained y + signed g=η−1)"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.nomad.factory:"
                "validate_pynomad_vns_mads_pb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="PYNOMAD_VNS_MADS_PB_WARM_START",
            library="nomad",
            constructor_ref=(
                "core.ioh_core.algorithms.nomad.factory:"
                "build_pynomad_vns_mads_pb_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                ParameterSpec(
                    name="vns_mads_search_trigger",
                    types=(int, float),
                    required=False,
                    default=0.75,
                    minimum=0.0,
                    maximum=1.0,
                ),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-nomad",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "PyNomad VNS-MADS-PB from rank 1 of a ranked InitialDesign "
                "(signed g=η−1)"
            ),
            batch_evaluations=False,
            validator_ref=(
                "core.ioh_core.algorithms.nomad.factory:"
                "validate_pynomad_vns_mads_pb_parameters"
            ),
        ),
        AlgorithmSpec(
            name="PFN_PEN",
            library="pfns4bo",
            constructor_ref=(
                "core.ioh_core.algorithms.pfns4bo.factory:build_pfn_pen"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-pfns4bo",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "PFNs4BO HEBO+ EI on integer catalogue indices "
                "(constraint-blind; maximizes -y_p)"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="PFN_PEN_WARM_START",
            library="pfns4bo",
            constructor_ref=(
                "core.ioh_core.algorithms.pfns4bo.factory:build_pfn_pen_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-pfns4bo",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description="PFN-PEN from a ranked InitialDesign DOE",
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="PFN_CEI",
            library="pfns4bo",
            constructor_ref=(
                "core.ioh_core.algorithms.pfns4bo.factory:build_pfn_cei"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-pfns4bo",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "PFNs4BO HEBO+ CEI on integer catalogue indices "
                "(constrained -y + signed g=η−1)"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="PFN_CEI_WARM_START",
            library="pfns4bo",
            constructor_ref=(
                "core.ioh_core.algorithms.pfns4bo.factory:build_pfn_cei_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-pfns4bo",
            candidate_mode=CandidateMode.INTEGER,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description="PFN-CEI from a ranked InitialDesign DOE",
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="OPTAX_ADAM",
            library="optax",
            constructor_ref=(
                "core.ioh_core.algorithms.optax.factory:build_optax_adam"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-optax",
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Adam on y_p with a discrete finite-difference gradient "
                "in catalogue-index space"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="OPTAX_ADAM_WARM_START",
            library="optax",
            constructor_ref=(
                "core.ioh_core.algorithms.optax.factory:"
                "build_optax_adam_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.FORBIDDEN,
            dependency_extra="algorithms-optax",
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Adam on y_p from rank 1 of a ranked InitialDesign"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="OPTAX_RPROP",
            library="optax",
            constructor_ref=(
                "core.ioh_core.algorithms.optax.factory:build_optax_rprop"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-optax",
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "RProp on y_p with a discrete finite-difference gradient "
                "in catalogue-index space"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="OPTAX_RPROP_WARM_START",
            library="optax",
            constructor_ref=(
                "core.ioh_core.algorithms.optax.factory:"
                "build_optax_rprop_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.FORBIDDEN,
            dependency_extra="algorithms-optax",
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "RProp on y_p from rank 1 of a ranked InitialDesign"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="OPTAX_SGD_NESTEROV",
            library="optax",
            constructor_ref=(
                "core.ioh_core.algorithms.optax.factory:"
                "build_optax_sgd_nesterov"
            ),
            parameters=(_max_evaluations_spec(),),
            seed_policy=SeedPolicy.REQUIRED,
            dependency_extra="algorithms-optax",
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Sign-normalised Nesterov SGD on y_p with a discrete "
                "finite-difference gradient in catalogue-index space"
            ),
            batch_evaluations=False,
        ),
        AlgorithmSpec(
            name="OPTAX_SGD_NESTEROV_WARM_START",
            library="optax",
            constructor_ref=(
                "core.ioh_core.algorithms.optax.factory:"
                "build_optax_sgd_nesterov_warm_start"
            ),
            parameters=(
                _max_evaluations_spec(),
                _initial_design_path_spec(),
            ),
            seed_policy=SeedPolicy.FORBIDDEN,
            dependency_extra="algorithms-optax",
            candidate_mode=CandidateMode.CONTINUOUS_ROUNDED,
            budget_semantics=BudgetSemantics.HARD_IOH_EVALUATIONS,
            description=(
                "Sign-normalised Nesterov SGD on y_p from rank 1 of a "
                "ranked InitialDesign"
            ),
            batch_evaluations=False,
        ),
    )
)
