"""
Import-safe constructors and parameter validation for SMT adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.errors import import_optional_dependency
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue, RegistryError
from core.ioh_core.algorithms.smt.space import default_n_doe


def validate_smt_ego_parameters(
    parameters: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    """
    Fill default ``n_doe`` and validate it against ``max_evaluations``.

    Parameters
    ----------
    parameters : mapping
        Partially normalized hyperparameters (defaults already applied for
        keys that declare static defaults).

    Returns
    -------
    dict
        Mapping that always includes ``n_doe``.

    Raises
    ------
    RegistryError
        If ``n_doe`` is outside ``[1, max_evaluations]``.
    """
    out = dict(parameters)
    max_evaluations = int(out["max_evaluations"])
    if "n_doe" not in out:
        out["n_doe"] = default_n_doe(max_evaluations)
    n_doe = int(out["n_doe"])
    if n_doe < 1:
        raise RegistryError(
            f"SMT_EGO: parameter 'n_doe' must be >= 1, got {n_doe!r}"
        )
    if n_doe > max_evaluations:
        raise RegistryError(
            f"SMT_EGO: parameter 'n_doe' must be <= max_evaluations "
            f"({max_evaluations}), got {n_doe!r}"
        )
    out["n_doe"] = n_doe
    return out


def build_smt_ego(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`SmtEgo` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations`` and
        ``n_doe``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent SMT EGO instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional smt extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "smt",
        algorithm_name="SMT_EGO",
        extra="algorithms-smt",
    )
    if random_seed is None:
        raise ValueError(
            "SMT_EGO requires a random_seed (unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.smt.algorithms import SmtEgo

    return SmtEgo(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        n_doe=int(parameters["n_doe"]),
    )


def validate_smt_ego_warm_start_parameters(
    parameters: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    """
    Fill default ``n_doe`` without the cold budget cap.

    A prefix longer than ``max_evaluations`` is rejected later by
    ``require_warm_start_budget``.
    """
    out = dict(parameters)
    max_evaluations = int(out["max_evaluations"])
    if "n_doe" not in out:
        out["n_doe"] = default_n_doe(max_evaluations)
    n_doe = int(out["n_doe"])
    if n_doe < 1:
        raise RegistryError(
            f"SMT_EGO_WARM_START: parameter 'n_doe' must be >= 1, got {n_doe!r}"
        )
    out["n_doe"] = n_doe
    return out


def build_smt_ego_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct SMT EGO from a ranked InitialDesign DOE."""
    import_optional_dependency(
        "smt",
        algorithm_name="SMT_EGO_WARM_START",
        extra="algorithms-smt",
    )
    if random_seed is None:
        raise ValueError(
            "SMT_EGO_WARM_START requires a random_seed (unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError("SMT_EGO_WARM_START requires initial_design_path")
    from core.ioh_core.algorithms.smt.algorithms import SmtEgoWarmStart

    return SmtEgoWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        n_doe=int(parameters["n_doe"]),
        initial_design_path=path,
    )
