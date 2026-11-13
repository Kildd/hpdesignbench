"""
Import-safe constructors for PFNs4BO algorithm adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.errors import import_optional_dependency
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def _require_seed(algorithm_name: str, random_seed: int | None) -> int:
    """
    Require a non-None seed for PFNs4BO adapters.

    Parameters
    ----------
    algorithm_name : str
        Registry token for the error message.
    random_seed : int or None
        Seed from the factory.

    Returns
    -------
    int
        Validated seed.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None``.
    """
    if random_seed is None:
        raise ValueError(
            f"{algorithm_name} requires a random_seed (unsigned 32-bit int)"
        )
    return int(random_seed)


def build_pfn_pen(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`PfnPen` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent PFN-PEN instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional pfns4bo extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "pfns4bo",
        algorithm_name="PFN_PEN",
        extra="algorithms-pfns4bo",
    )
    import_optional_dependency(
        "torch",
        algorithm_name="PFN_PEN",
        extra="algorithms-pfns4bo",
    )
    seed = _require_seed("PFN_PEN", random_seed)
    from core.ioh_core.algorithms.pfns4bo.algorithms import PfnPen

    return PfnPen(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=seed,
    )


def build_pfn_cei(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`PfnCei` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent PFN-CEI instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional pfns4bo extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "pfns4bo",
        algorithm_name="PFN_CEI",
        extra="algorithms-pfns4bo",
    )
    import_optional_dependency(
        "torch",
        algorithm_name="PFN_CEI",
        extra="algorithms-pfns4bo",
    )
    seed = _require_seed("PFN_CEI", random_seed)
    from core.ioh_core.algorithms.pfns4bo.algorithms import PfnCei

    return PfnCei(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=seed,
    )


def build_pfn_pen_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct PFN-PEN from a ranked InitialDesign DOE."""
    import_optional_dependency(
        "pfns4bo",
        algorithm_name="PFN_PEN_WARM_START",
        extra="algorithms-pfns4bo",
    )
    import_optional_dependency(
        "torch",
        algorithm_name="PFN_PEN_WARM_START",
        extra="algorithms-pfns4bo",
    )
    seed = _require_seed("PFN_PEN_WARM_START", random_seed)
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError("PFN_PEN_WARM_START requires initial_design_path")
    from core.ioh_core.algorithms.pfns4bo.algorithms import PfnPenWarmStart

    return PfnPenWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=seed,
        initial_design_path=path,
    )


def build_pfn_cei_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct PFN-CEI from a ranked InitialDesign DOE."""
    import_optional_dependency(
        "pfns4bo",
        algorithm_name="PFN_CEI_WARM_START",
        extra="algorithms-pfns4bo",
    )
    import_optional_dependency(
        "torch",
        algorithm_name="PFN_CEI_WARM_START",
        extra="algorithms-pfns4bo",
    )
    seed = _require_seed("PFN_CEI_WARM_START", random_seed)
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError("PFN_CEI_WARM_START requires initial_design_path")
    from core.ioh_core.algorithms.pfns4bo.algorithms import PfnCeiWarmStart

    return PfnCeiWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=seed,
        initial_design_path=path,
    )
