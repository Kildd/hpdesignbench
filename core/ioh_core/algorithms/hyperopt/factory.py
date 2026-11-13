"""
Import-safe constructors for Hyperopt algorithm adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.errors import import_optional_dependency
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def build_hyperopt_tpe(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`HyperoptTPE` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent TPE instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional Hyperopt extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "hyperopt",
        algorithm_name="HYPEROPT_TPE",
        extra="algorithms-hyperopt",
    )
    if random_seed is None:
        raise ValueError("HYPEROPT_TPE requires a random_seed (unsigned 32-bit int)")
    from core.ioh_core.algorithms.hyperopt.algorithms import HyperoptTPE

    return HyperoptTPE(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_hyperopt_tpe_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct Hyperopt TPE from a ranked InitialDesign prefix."""
    import_optional_dependency(
        "hyperopt",
        algorithm_name="HYPEROPT_TPE_WARM_START",
        extra="algorithms-hyperopt",
    )
    if random_seed is None:
        raise ValueError(
            "HYPEROPT_TPE_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError("HYPEROPT_TPE_WARM_START requires initial_design_path")
    from core.ioh_core.algorithms.hyperopt.algorithms import HyperoptTPEWarmStart

    return HyperoptTPEWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        initial_design_path=path,
    )
