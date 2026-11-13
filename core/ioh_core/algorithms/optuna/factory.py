"""
Import-safe constructors for Optuna algorithm adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.errors import import_optional_dependency
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def build_optuna_tpe(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`OptunaTPE` instance.

    The adapter is constrained-only (``mm ≥ 1``); validation happens at
    ``__call__`` time when the IOH problem is available.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent Optuna TPE instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional Optuna extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "optuna",
        algorithm_name="OPTUNA_TPE",
        extra="algorithms-optuna",
    )
    if random_seed is None:
        raise ValueError("OPTUNA_TPE requires a random_seed (unsigned 32-bit int)")
    from core.ioh_core.algorithms.optuna.tpe import OptunaTPE

    return OptunaTPE(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_optuna_tpe_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct Optuna TPE from a ranked InitialDesign prefix."""
    import_optional_dependency(
        "optuna",
        algorithm_name="OPTUNA_TPE_WARM_START",
        extra="algorithms-optuna",
    )
    if random_seed is None:
        raise ValueError(
            "OPTUNA_TPE_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError("OPTUNA_TPE_WARM_START requires initial_design_path")
    from core.ioh_core.algorithms.optuna.tpe import OptunaTPEWarmStart

    return OptunaTPEWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        initial_design_path=path,
    )
