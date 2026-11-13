"""
Strict factory for optimisation algorithms used by parallel workers.

Public facade over the production algorithm registry. Selection and parameter
validation are registry-driven; concrete implementations remain under
``core.ioh_core.algorithms``.
"""
from __future__ import annotations

from typing import Any, Mapping

from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import (
    ALGORITHM_REGISTRY,
    RegistryError,
    construct_algorithm,
    normalize_algorithm_parameters,
    validate_seed,
)


class AlgorithmFactoryError(ValueError):
    """
    Raised when an algorithm name or parameter set is invalid.

    Typical causes: unknown algorithm names, missing ``max_evaluations``,
    unknown parameter keys, or a seed supplied for a seedless algorithm.
    """


def validate_algorithm_parameters(
    name: str,
    parameters: Mapping[str, Any],
    *,
    random_seed: int | None = None,
) -> None:
    """
    Reject unknown algorithms and illegal parameter/seed combinations.

    Parameters
    ----------
    name : str
        Algorithm factory name.
    parameters : Mapping[str, Any]
        Declared hyperparameters.
    random_seed : int or None, optional
        Seed from the task payload. Must be ``None`` for seedless algorithms.

    Raises
    ------
    AlgorithmFactoryError
        If validation fails.
    """
    try:
        normalize_algorithm_parameters(name, parameters)
        validate_seed(ALGORITHM_REGISTRY.get(name), random_seed)
    except RegistryError as exc:
        raise AlgorithmFactoryError(str(exc)) from exc


def create_algorithm(
    name: str,
    parameters: Mapping[str, Any],
    *,
    random_seed: int | None = None,
) -> IOHAlgorithm:
    """
    Construct a fresh algorithm instance for one optimisation task.

    Parameters
    ----------
    name : str
        Registered algorithm token such as ``\"RANDOM_SEARCH\"``.
    parameters : Mapping[str, Any]
        Strict hyperparameters; currently only ``max_evaluations``.
    random_seed : int or None, optional
        Required for seeded algorithms; must be ``None`` when forbidden.

    Returns
    -------
    IOHAlgorithm
        New independent algorithm instance.

    Raises
    ------
    AlgorithmFactoryError
        If the name/parameters/seed combination is invalid.
    """
    try:
        return construct_algorithm(
            ALGORITHM_REGISTRY,
            name,
            parameters,
            random_seed=random_seed,
        )
    except RegistryError as exc:
        raise AlgorithmFactoryError(str(exc)) from exc
