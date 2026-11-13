"""
Import-safe constructors for Pymoo algorithm adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.errors import import_optional_dependency
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def build_pymoo_ga(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`PymooGeneticAlgorithm` instance.

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
        If the optional Pymoo extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "pymoo",
        algorithm_name="PYMOO_GA",
        extra="algorithms-pymoo",
    )
    if random_seed is None:
        raise ValueError("PYMOO_GA requires a random_seed (unsigned 32-bit int)")
    from core.ioh_core.algorithms.pymoo.algorithms import PymooGeneticAlgorithm

    return PymooGeneticAlgorithm(
        max_evaluations=int(parameters["max_evaluations"]),
        population_size=int(parameters["population_size"]),
        eliminate_duplicates=bool(parameters["eliminate_duplicates"]),
        seed=int(random_seed),
    )


def build_pymoo_ga_constrained(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`PymooConstrainedGeneticAlgorithm` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent constrained GA instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional Pymoo extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "pymoo",
        algorithm_name="PYMOO_GA_CONSTRAINED",
        extra="algorithms-pymoo",
    )
    if random_seed is None:
        raise ValueError(
            "PYMOO_GA_CONSTRAINED requires a random_seed (unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.pymoo.algorithms import (
        PymooConstrainedGeneticAlgorithm,
    )

    return PymooConstrainedGeneticAlgorithm(
        max_evaluations=int(parameters["max_evaluations"]),
        population_size=int(parameters["population_size"]),
        eliminate_duplicates=bool(parameters["eliminate_duplicates"]),
        seed=int(random_seed),
    )


def build_pymoo_ga_constrained_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct constrained Pymoo GA from a ranked InitialDesign prefix.
    """
    import_optional_dependency(
        "pymoo",
        algorithm_name="PYMOO_GA_CONSTRAINED_WARM_START",
        extra="algorithms-pymoo",
    )
    if random_seed is None:
        raise ValueError(
            "PYMOO_GA_CONSTRAINED_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "PYMOO_GA_CONSTRAINED_WARM_START requires initial_design_path"
        )
    from core.ioh_core.algorithms.pymoo.algorithms import (
        PymooConstrainedGeneticAlgorithm,
    )

    return PymooConstrainedGeneticAlgorithm(
        max_evaluations=int(parameters["max_evaluations"]),
        population_size=int(parameters["population_size"]),
        eliminate_duplicates=bool(parameters["eliminate_duplicates"]),
        seed=int(random_seed),
        warm_start=True,
        initial_design_path=path,
        name="PYMOO_GA_CONSTRAINED_WARM_START",
    )


def build_pymoo_ga_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct unconstrained Pymoo GA from a ranked InitialDesign prefix.
    """
    import_optional_dependency(
        "pymoo",
        algorithm_name="PYMOO_GA_WARM_START",
        extra="algorithms-pymoo",
    )
    if random_seed is None:
        raise ValueError(
            "PYMOO_GA_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError("PYMOO_GA_WARM_START requires initial_design_path")
    from core.ioh_core.algorithms.pymoo.algorithms import (
        PymooGeneticAlgorithmWarmStart,
    )

    return PymooGeneticAlgorithmWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        population_size=int(parameters["population_size"]),
        eliminate_duplicates=bool(parameters["eliminate_duplicates"]),
        seed=int(random_seed),
        initial_design_path=path,
    )
