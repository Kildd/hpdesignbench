"""
Import-safe constructors for DFL / DFLINT algorithm adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.registry import JSONValue
from core.ioh_core.algorithms.protocol import IOHAlgorithm


def build_dflint_bboa(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`DflintBboa` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent DFLINT BBOA instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None``.
    """
    if random_seed is None:
        raise ValueError(
            "DFLINT_BBOA requires a random_seed (unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.dfl.algorithms import DflintBboa

    return DflintBboa(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_dflint_bboa_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct DFLINT BBOA starting at rank 1 of an InitialDesign.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations`` and
        ``initial_design_path``.
    random_seed : int or None
        Must be ``None``.

    Returns
    -------
    IOHAlgorithm
        New independent DFLINT BBOA instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is not ``None`` or the path is missing.
    """
    if random_seed is not None:
        raise ValueError(
            "DFLINT_BBOA_WARM_START is seedless; random_seed must be None"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or path == "":
        raise ValueError("DFLINT_BBOA_WARM_START requires initial_design_path")
    from core.ioh_core.algorithms.dfl.algorithms import DflintBboa

    return DflintBboa(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=None,
        initial_design_path=path,
        name="DFLINT_BBOA_WARM_START",
    )
