"""
Import-safe constructors for Nevergrad algorithm adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.errors import import_optional_dependency
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def build_nevergrad_metamodel(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`NevergradMetaModel` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent MetaModel instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional nevergrad extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "nevergrad",
        algorithm_name="NEVERGRAD_METAMODEL",
        extra="algorithms-nevergrad",
    )
    if random_seed is None:
        raise ValueError(
            "NEVERGRAD_METAMODEL requires a random_seed "
            "(unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.nevergrad.algorithms import NevergradMetaModel

    return NevergradMetaModel(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_nevergrad_metamodel_oneplusone(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`NevergradMetaModelOnePlusOne` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent MetaModelOnePlusOne instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional nevergrad extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "nevergrad",
        algorithm_name="NEVERGRAD_METAMODEL_ONEPLUSONE",
        extra="algorithms-nevergrad",
    )
    if random_seed is None:
        raise ValueError(
            "NEVERGRAD_METAMODEL_ONEPLUSONE requires a random_seed "
            "(unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.nevergrad.algorithms import (
        NevergradMetaModelOnePlusOne,
    )

    return NevergradMetaModelOnePlusOne(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def _build_nevergrad_warm(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
    *,
    name: str,
    cls_name: str,
) -> IOHAlgorithm:
    import_optional_dependency(
        "nevergrad",
        algorithm_name=name,
        extra="algorithms-nevergrad",
    )
    if random_seed is None:
        raise ValueError(f"{name} requires a random_seed (unsigned 32-bit int)")
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{name} requires initial_design_path")
    from core.ioh_core.algorithms.nevergrad import algorithms as nevergrad_algorithms

    cls = getattr(nevergrad_algorithms, cls_name)
    return cls(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        initial_design_path=path,
    )


def build_nevergrad_metamodel_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct MetaModel from a ranked InitialDesign prefix."""
    return _build_nevergrad_warm(
        parameters,
        random_seed,
        name="NEVERGRAD_METAMODEL_WARM_START",
        cls_name="NevergradMetaModelWarmStart",
    )


def build_nevergrad_metamodel_oneplusone_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct MetaModelOnePlusOne from a ranked InitialDesign prefix."""
    return _build_nevergrad_warm(
        parameters,
        random_seed,
        name="NEVERGRAD_METAMODEL_ONEPLUSONE_WARM_START",
        cls_name="NevergradMetaModelOnePlusOneWarmStart",
    )
