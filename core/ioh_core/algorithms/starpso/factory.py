"""
Import-safe constructors for StarPSO algorithm adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.errors import import_optional_dependency
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def build_pystarpso_integer_pso(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`PyStarPsoIntegerPso` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent IntegerPSO instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional starpso extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "star_pso",
        algorithm_name="PYSTARPSO_INTEGER_PSO",
        extra="algorithms-starpso",
    )
    if random_seed is None:
        raise ValueError(
            "PYSTARPSO_INTEGER_PSO requires a random_seed "
            "(unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.starpso.algorithms import PyStarPsoIntegerPso

    return PyStarPsoIntegerPso(
        max_evaluations=int(parameters["max_evaluations"]),
        n_pop=int(parameters["n_pop"]),
        seed=int(random_seed),
    )


def build_pystarpso_integer_pso_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct IntegerPSO from a ranked InitialDesign prefix.
    """
    import_optional_dependency(
        "star_pso",
        algorithm_name="PYSTARPSO_INTEGER_PSO_WARM_START",
        extra="algorithms-starpso",
    )
    if random_seed is None:
        raise ValueError(
            "PYSTARPSO_INTEGER_PSO_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "PYSTARPSO_INTEGER_PSO_WARM_START requires initial_design_path"
        )
    from core.ioh_core.algorithms.starpso.algorithms import (
        PyStarPsoIntegerPsoWarmStart,
    )

    return PyStarPsoIntegerPsoWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        n_pop=int(parameters["n_pop"]),
        seed=int(random_seed),
        initial_design_path=path,
    )
