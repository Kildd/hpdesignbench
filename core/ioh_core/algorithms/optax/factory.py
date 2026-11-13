"""
Import-safe constructors for Optax discrete-gradient adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.errors import import_optional_dependency
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def _require_optax_extra(algorithm_name: str) -> None:
    """
    Import Optax and JAX or raise an install hint.

    Parameters
    ----------
    algorithm_name : str
        Registry token used in the error message.

    Raises
    ------
    AlgorithmDependencyError
        If ``optax`` or ``jax`` is not installed.
    """
    import_optional_dependency(
        "optax",
        algorithm_name=algorithm_name,
        extra="algorithms-optax",
    )
    import_optional_dependency(
        "jax",
        algorithm_name=algorithm_name,
        extra="algorithms-optax",
    )


def _require_seed(algorithm_name: str, random_seed: int | None) -> int:
    """
    Return the task seed or raise when it is missing.

    Parameters
    ----------
    algorithm_name : str
        Registry token used in the error message.
    random_seed : int or None
        Unsigned 32-bit task seed.

    Returns
    -------
    int
        Validated seed.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None``.
    """
    if random_seed is None:
        raise ValueError(
            f"{algorithm_name} requires a random_seed (unsigned 32-bit int)"
        )
    return int(random_seed)


def build_optax_adam(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`OptaxAdam` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters; must include ``max_evaluations``.
    random_seed : int or None
        Required seed for the uniform continuous start.

    Returns
    -------
    IOHAlgorithm
        New independent Adam instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional Optax extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    _require_optax_extra("OPTAX_ADAM")
    seed = _require_seed("OPTAX_ADAM", random_seed)
    from core.ioh_core.algorithms.optax.adam import OptaxAdam

    return OptaxAdam(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=seed,
    )


def build_optax_rprop(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`OptaxRProp` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters; must include ``max_evaluations``.
    random_seed : int or None
        Required seed for the uniform continuous start.

    Returns
    -------
    IOHAlgorithm
        New independent RProp instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional Optax extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    _require_optax_extra("OPTAX_RPROP")
    seed = _require_seed("OPTAX_RPROP", random_seed)
    from core.ioh_core.algorithms.optax.rprop import OptaxRProp

    return OptaxRProp(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=seed,
    )


def build_optax_sgd_nesterov(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`OptaxSGDNesterov` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters; must include ``max_evaluations``.
    random_seed : int or None
        Required seed for the uniform continuous start.

    Returns
    -------
    IOHAlgorithm
        New independent sign-normalised Nesterov SGD instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional Optax extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    _require_optax_extra("OPTAX_SGD_NESTEROV")
    seed = _require_seed("OPTAX_SGD_NESTEROV", random_seed)
    from core.ioh_core.algorithms.optax.sgd_nesterov import OptaxSGDNesterov

    return OptaxSGDNesterov(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=seed,
    )


def _require_rank1_path(algorithm_name: str, parameters: Mapping[str, JSONValue]) -> str:
    """
    Return the planner-injected InitialDesign path.

    Parameters
    ----------
    algorithm_name : str
        Warm-start token.
    parameters : mapping
        Normalized hyperparameters.

    Returns
    -------
    str
        Path of ``initial_design.json``.

    Raises
    ------
    ValueError
        If the path is missing or a seed was supplied. The seed check is
        the caller's responsibility.
    """
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or path == "":
        raise ValueError(f"{algorithm_name} requires initial_design_path")
    return path


def _require_seedless(algorithm_name: str, random_seed: int | None) -> None:
    if random_seed is not None:
        raise ValueError(
            f"{algorithm_name} is seedless; random_seed must be None"
        )


def build_optax_adam_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct Adam starting at rank 1 of an InitialDesign.

    Parameters
    ----------
    parameters : mapping
        Must include ``max_evaluations`` and ``initial_design_path``.
    random_seed : int or None
        Must be ``None``.

    Returns
    -------
    IOHAlgorithm
        New independent warm-start Adam instance.
    """
    _require_optax_extra("OPTAX_ADAM_WARM_START")
    _require_seedless("OPTAX_ADAM_WARM_START", random_seed)
    from core.ioh_core.algorithms.optax.engine import OptaxAdamWarmStart

    return OptaxAdamWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        initial_design_path=_require_rank1_path(
            "OPTAX_ADAM_WARM_START",
            parameters,
        ),
    )


def build_optax_rprop_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct RProp starting at rank 1 of an InitialDesign.

    Parameters
    ----------
    parameters : mapping
        Must include ``max_evaluations`` and ``initial_design_path``.
    random_seed : int or None
        Must be ``None``.

    Returns
    -------
    IOHAlgorithm
        New independent warm-start RProp instance.
    """
    _require_optax_extra("OPTAX_RPROP_WARM_START")
    _require_seedless("OPTAX_RPROP_WARM_START", random_seed)
    from core.ioh_core.algorithms.optax.engine import OptaxRPropWarmStart

    return OptaxRPropWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        initial_design_path=_require_rank1_path(
            "OPTAX_RPROP_WARM_START",
            parameters,
        ),
    )


def build_optax_sgd_nesterov_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct sign-normalised Nesterov SGD from rank 1.

    Parameters
    ----------
    parameters : mapping
        Must include ``max_evaluations`` and ``initial_design_path``.
    random_seed : int or None
        Must be ``None``.

    Returns
    -------
    IOHAlgorithm
        New independent warm-start SGD instance.
    """
    _require_optax_extra("OPTAX_SGD_NESTEROV_WARM_START")
    _require_seedless("OPTAX_SGD_NESTEROV_WARM_START", random_seed)
    from core.ioh_core.algorithms.optax.engine import OptaxSGDNesterovWarmStart

    return OptaxSGDNesterovWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        initial_design_path=_require_rank1_path(
            "OPTAX_SGD_NESTEROV_WARM_START",
            parameters,
        ),
    )
