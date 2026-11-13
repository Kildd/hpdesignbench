"""
Import-safe constructors for SciPy algorithm adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue
from core.ioh_core.algorithms.scipy.cobyla import Cobyla, CobylaWarmStart
from core.ioh_core.algorithms.scipy.differential_evolution import (
    DifferentialEvolution,
)
from core.ioh_core.algorithms.scipy.direct_l import DirectL
from core.ioh_core.algorithms.scipy.dual_annealing import (
    DualAnnealing,
    DualAnnealingWarmStart,
)


def build_scipy_cobyla(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`Cobyla` instance.

    The adapter is constrained-only (``mm ≥ 1``); validation happens at
    ``__call__`` time when the IOH problem is available.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters; must include ``max_evaluations``.
    random_seed : int or None
        Required seed for the uniform continuous start point.

    Returns
    -------
    IOHAlgorithm
        New independent COBYLA instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None``.
    """
    if random_seed is None:
        raise ValueError(
            "SCIPY_COBYLA requires a random_seed (unsigned 32-bit int)"
        )
    return Cobyla(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_scipy_cobyla_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct COBYLA starting at rank 1 of an InitialDesign.

    Parameters
    ----------
    parameters : mapping
        Must include ``max_evaluations`` and ``initial_design_path``.
    random_seed : int or None
        Must be ``None``.

    Returns
    -------
    IOHAlgorithm
        New independent warm-start COBYLA instance.

    Raises
    ------
    ValueError
        If a seed is set or ``initial_design_path`` is missing.
    """
    if random_seed is not None:
        raise ValueError(
            "SCIPY_COBYLA_WARM_START is seedless; random_seed must be None"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or path == "":
        raise ValueError(
            "SCIPY_COBYLA_WARM_START requires initial_design_path"
        )
    return CobylaWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        initial_design_path=path,
    )


def build_scipy_dual_annealing(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`DualAnnealing` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters; must include ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed passed as SciPy ``rng``.

    Returns
    -------
    IOHAlgorithm
        New independent dual-annealing instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None``.
    """
    if random_seed is None:
        raise ValueError(
            "SCIPY_DUAL_ANNEALING requires a random_seed "
            "(unsigned 32-bit int)"
        )
    return DualAnnealing(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_scipy_dual_annealing_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct dual annealing from rank 1 of a ranked InitialDesign.

    Parameters
    ----------
    parameters : mapping
        Must include ``max_evaluations`` and ``initial_design_path``.
    random_seed : int or None
        Required deterministic seed. The warm start is not seedless.

    Returns
    -------
    IOHAlgorithm
        New independent warm-start dual-annealing instance.

    Raises
    ------
    ValueError
        If a seed is missing or ``initial_design_path`` is missing.
    """
    if random_seed is None:
        raise ValueError(
            "SCIPY_DUAL_ANNEALING_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or path == "":
        raise ValueError(
            "SCIPY_DUAL_ANNEALING_WARM_START requires initial_design_path"
        )
    return DualAnnealingWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        initial_design_path=path,
    )


def build_scipy_direct_l(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`DirectL` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters; must include ``max_evaluations``.
    random_seed : int or None
        Must be ``None`` for this deterministic algorithm.

    Returns
    -------
    IOHAlgorithm
        New independent SciPy DIRECT-L instance.

    Raises
    ------
    ValueError
        If a non-null seed is supplied.
    """
    if random_seed is not None:
        raise ValueError(
            "SCIPY_DIRECT_L is seedless; random_seed must be null"
        )
    return DirectL(max_evaluations=int(parameters["max_evaluations"]))


def build_scipy_differential_evolution(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`DifferentialEvolution` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters; must include ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent differential-evolution instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None``.
    """
    if random_seed is None:
        raise ValueError(
            "SCIPY_DIFFERENTIAL_EVOLUTION requires a random_seed "
            "(unsigned 32-bit int)"
        )
    return DifferentialEvolution(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_scipy_differential_evolution_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct differential evolution from a ranked InitialDesign prefix.
    """
    if random_seed is None:
        raise ValueError(
            "SCIPY_DIFFERENTIAL_EVOLUTION_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "SCIPY_DIFFERENTIAL_EVOLUTION_WARM_START requires "
            "initial_design_path"
        )
    from core.ioh_core.algorithms.scipy.differential_evolution import (
        DifferentialEvolutionWarmStart,
    )

    return DifferentialEvolutionWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        initial_design_path=path,
    )
