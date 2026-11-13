"""
Import-safe constructors for Altbacken algorithm adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.errors import import_optional_dependency
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def build_altbacken_sa(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`AltbackenSA` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent Altbacken SA instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional altbacken extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "altbacken",
        algorithm_name="ALTBACKEN_SA",
        extra="algorithms-altbacken",
        pip_extra_args="--ignore-requires-python",
    )
    if random_seed is None:
        raise ValueError(
            "ALTBACKEN_SA requires a random_seed (unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.altbacken.algorithms import AltbackenSA

    return AltbackenSA(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_altbacken_sa_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct Altbacken SA starting at rank 1 of an InitialDesign.

    Parameters
    ----------
    parameters : mapping
        Must include ``max_evaluations`` and ``initial_design_path``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent warm-start SA instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None`` or the path is missing.
    """
    import_optional_dependency(
        "altbacken",
        algorithm_name="ALTBACKEN_SA_WARM_START",
        extra="algorithms-altbacken",
        pip_extra_args="--ignore-requires-python",
    )
    if random_seed is None:
        raise ValueError(
            "ALTBACKEN_SA_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or path == "":
        raise ValueError("ALTBACKEN_SA_WARM_START requires initial_design_path")
    from core.ioh_core.algorithms.altbacken.algorithms import AltbackenSAWarmStart

    return AltbackenSAWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        initial_design_path=path,
    )
