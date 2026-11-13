"""
Import-safe constructors for cmaes algorithm adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.errors import import_optional_dependency
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def build_cmaes_catcmawm(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`CatCmaWMAlgorithm` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent CatCMAwM instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional cmaes extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "cmaes",
        algorithm_name="CMAES_CATCMAWM",
        extra="algorithms-cmaes",
    )
    if random_seed is None:
        raise ValueError(
            "CMAES_CATCMAWM requires a random_seed (unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.cmaes.algorithms import CatCmaWMAlgorithm

    return CatCmaWMAlgorithm(
        max_evaluations=int(parameters["max_evaluations"]),
        population_size=int(parameters["population_size"]),
        seed=int(random_seed),
    )


def build_cmaes_catcmawm_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct CatCMAwM with rank 1 as the initial mean.

    Parameters
    ----------
    parameters : mapping
        Must include ``max_evaluations`` and ``initial_design_path``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent warm-start CatCMAwM instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None`` or the path is missing.
    """
    import_optional_dependency(
        "cmaes",
        algorithm_name="CMAES_CATCMAWM_WARM_START",
        extra="algorithms-cmaes",
    )
    if random_seed is None:
        raise ValueError(
            "CMAES_CATCMAWM_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or path == "":
        raise ValueError(
            "CMAES_CATCMAWM_WARM_START requires initial_design_path"
        )
    from core.ioh_core.algorithms.cmaes.algorithms import CatCmaWMWarmStart

    return CatCmaWMWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        population_size=int(parameters["population_size"]),
        seed=int(random_seed),
        initial_design_path=path,
    )
