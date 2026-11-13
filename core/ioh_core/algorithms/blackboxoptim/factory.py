"""
Import-safe constructors for blackboxoptim adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.errors import import_optional_dependency
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def build_blackboxopt_msrs(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`BlackboxoptMsrs` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent MSRS instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional blackboxoptim extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "blackboxoptim",
        algorithm_name="BLACKBOXOPT_MSRS",
        extra="algorithms-blackboxoptim",
    )
    if random_seed is None:
        raise ValueError(
            "BLACKBOXOPT_MSRS requires a random_seed (unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.blackboxoptim.algorithms import BlackboxoptMsrs

    return BlackboxoptMsrs(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_blackboxopt_dycors(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`BlackboxoptDycors` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent DYCORS instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional blackboxoptim extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "blackboxoptim",
        algorithm_name="BLACKBOXOPT_DYCORS",
        extra="algorithms-blackboxoptim",
    )
    if random_seed is None:
        raise ValueError(
            "BLACKBOXOPT_DYCORS requires a random_seed (unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.blackboxoptim.algorithms import (
        BlackboxoptDycors,
    )

    return BlackboxoptDycors(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_blackboxopt_gosac(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`BlackboxoptGosac` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent GOSAC instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional blackboxoptim extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "blackboxoptim",
        algorithm_name="BLACKBOXOPT_GOSAC",
        extra="algorithms-blackboxoptim",
    )
    if random_seed is None:
        raise ValueError(
            "BLACKBOXOPT_GOSAC requires a random_seed (unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.blackboxoptim.algorithms import (
        BlackboxoptGosac,
    )

    return BlackboxoptGosac(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )
