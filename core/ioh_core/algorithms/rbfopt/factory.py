"""
Import-safe constructors and solver-path preflight for RBFOpt adapters.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping

from core.ioh_core.algorithms.errors import (
    AlgorithmDependencyError,
    import_optional_dependency,
)
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def resolve_solver_executable(value: str, *, parameter_name: str) -> str:
    """
    Resolve an external solver executable path for RBFOpt.

    Accepts a bare command name or a filesystem path (including paths that
    contain spaces). Uses :func:`shutil.which` only — no shell, no argv
    splitting.

    Parameters
    ----------
    value : str
        Bare executable name or path string from sealed task parameters.
    parameter_name : str
        Algorithm parameter name used in error messages
        (``minlp_solver_path`` or ``nlp_solver_path``).

    Returns
    -------
    str
        Absolute resolved executable path.

    Raises
    ------
    AlgorithmDependencyError
        If ``value`` is not a non-empty string or cannot be resolved to an
        executable.
    """
    if not isinstance(value, str) or not value.strip():
        raise AlgorithmDependencyError(
            f"RBFOPT requires executable for '{parameter_name}', got '{value}'"
        )
    resolved = shutil.which(value)
    if resolved is None:
        raise AlgorithmDependencyError(
            f"RBFOPT requires executable for '{parameter_name}', got '{value}'"
        )
    return str(Path(resolved).resolve())


def build_rbfopt(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`RBFOptSearch` instance.

    Resolves both external solver executables in the worker environment before
    returning the adapter. Registry/config import never performs this check.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including solver paths.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent RBFOpt instance with resolved solver paths.

    Raises
    ------
    AlgorithmDependencyError
        If the optional RBFOpt extra is missing or a solver path cannot be
        resolved.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "rbfopt",
        algorithm_name="RBFOPT",
        extra="algorithms-rbfopt",
    )
    if random_seed is None:
        raise ValueError("RBFOPT requires a random_seed (unsigned 32-bit int)")

    resolved_minlp = resolve_solver_executable(
        str(parameters["minlp_solver_path"]),
        parameter_name="minlp_solver_path",
    )
    resolved_nlp = resolve_solver_executable(
        str(parameters["nlp_solver_path"]),
        parameter_name="nlp_solver_path",
    )
    from core.ioh_core.algorithms.rbfopt.algorithms import RBFOptSearch

    return RBFOptSearch(
        max_evaluations=int(parameters["max_evaluations"]),
        ga_base_population_size=int(parameters["ga_base_population_size"]),
        minlp_solver_path=resolved_minlp,
        nlp_solver_path=resolved_nlp,
        seed=int(random_seed),
    )


def build_rbfopt_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct RBFOpt from a ranked InitialDesign prefix."""
    import_optional_dependency(
        "rbfopt",
        algorithm_name="RBFOPT_WARM_START",
        extra="algorithms-rbfopt",
    )
    if random_seed is None:
        raise ValueError(
            "RBFOPT_WARM_START requires a random_seed (unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError("RBFOPT_WARM_START requires initial_design_path")
    resolved_minlp = resolve_solver_executable(
        str(parameters["minlp_solver_path"]),
        parameter_name="minlp_solver_path",
    )
    resolved_nlp = resolve_solver_executable(
        str(parameters["nlp_solver_path"]),
        parameter_name="nlp_solver_path",
    )
    from core.ioh_core.algorithms.rbfopt.algorithms import RBFOptSearchWarmStart

    return RBFOptSearchWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        ga_base_population_size=int(parameters["ga_base_population_size"]),
        minlp_solver_path=resolved_minlp,
        nlp_solver_path=resolved_nlp,
        seed=int(random_seed),
        initial_design_path=path,
    )
