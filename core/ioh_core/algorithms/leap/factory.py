"""
Import-safe constructors for LEAP algorithm adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.errors import import_optional_dependency
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def build_leap_ga(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`LeapGeneticAlgorithm` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent GA instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional LEAP extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "leap_ec",
        algorithm_name="LEAP_GA",
        extra="algorithms-leap",
    )
    if random_seed is None:
        raise ValueError("LEAP_GA requires a random_seed (unsigned 32-bit int)")
    from core.ioh_core.algorithms.leap.algorithms import LeapGeneticAlgorithm

    return LeapGeneticAlgorithm(
        max_evaluations=int(parameters["max_evaluations"]),
        population_size=int(parameters["population_size"]),
        seed=int(random_seed),
    )


def build_leap_ga_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a LEAP GA from a ranked InitialDesign prefix.
    """
    import_optional_dependency(
        "leap_ec",
        algorithm_name="LEAP_GA_WARM_START",
        extra="algorithms-leap",
    )
    if random_seed is None:
        raise ValueError(
            "LEAP_GA_WARM_START requires a random_seed (unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError("LEAP_GA_WARM_START requires initial_design_path")
    from core.ioh_core.algorithms.leap.algorithms import (
        LeapGeneticAlgorithmWarmStart,
    )

    return LeapGeneticAlgorithmWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        population_size=int(parameters["population_size"]),
        seed=int(random_seed),
        initial_design_path=path,
    )
