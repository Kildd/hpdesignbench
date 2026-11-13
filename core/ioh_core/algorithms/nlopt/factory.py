"""
Import-safe constructors for NLopt algorithm adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.nlopt.algorithms import (
    NLoptDirectLocalSearch,
    NLoptSbplx,
    NLoptSbplxWarmStart,
)
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def build_direct_l(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`NLoptDirectLocalSearch` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters; must include ``max_evaluations``.
    random_seed : int or None
        Must be ``None`` for this deterministic algorithm.

    Returns
    -------
    IOHAlgorithm
        New DIRECT-L instance.

    Raises
    ------
    ValueError
        If a non-null seed is supplied.
    """
    if random_seed is not None:
        raise ValueError("NLOPT_DIRECT_L is seedless; random_seed must be null")
    return NLoptDirectLocalSearch(
        max_evaluations=int(parameters["max_evaluations"]),
    )


def build_sbplx(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`NLoptSbplx` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters; must include ``max_evaluations``.
    random_seed : int or None
        Required seed for the uniform continuous start point.

    Returns
    -------
    IOHAlgorithm
        New SBPLX instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None``.
    """
    if random_seed is None:
        raise ValueError(
            "NLOPT_SBPLX requires a random_seed (unsigned 32-bit int)"
        )
    return NLoptSbplx(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_sbplx_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct SBPLX starting at rank 1 of an InitialDesign.

    Parameters
    ----------
    parameters : mapping
        Must include ``max_evaluations`` and ``initial_design_path``.
    random_seed : int or None
        Must be ``None``.

    Returns
    -------
    IOHAlgorithm
        New independent warm-start SBPLX instance.

    Raises
    ------
    ValueError
        If a seed is set or ``initial_design_path`` is missing.
    """
    if random_seed is not None:
        raise ValueError(
            "NLOPT_SBPLX_WARM_START is seedless; random_seed must be None"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or path == "":
        raise ValueError("NLOPT_SBPLX_WARM_START requires initial_design_path")
    return NLoptSbplxWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        initial_design_path=path,
    )
