"""
Optional-dependency errors for library-backed optimisation algorithms.

Adapters call :func:`import_optional_dependency` so missing extras produce an
actionable install hint instead of a bare ``ModuleNotFoundError``.
"""
from __future__ import annotations

import importlib
from types import ModuleType


class AlgorithmDependencyError(ImportError):
    """
    Raised when an optional optimiser library is not installed.

    Subclasses ``ImportError`` so callers can distinguish missing extras from
    invalid algorithm configuration (``AlgorithmFactoryError``).
    """


def import_optional_dependency(
    module_name: str,
    *,
    algorithm_name: str,
    extra: str,
    pip_extra_args: str = "",
) -> ModuleType:
    """
    Import ``module_name`` or raise an actionable dependency error.

    Parameters
    ----------
    module_name : str
        Top-level module such as ``\"pymoo\"``.
    algorithm_name : str
        Registry token used in the error message.
    extra : str
        ``pyproject.toml`` optional-extra name such as ``\"algorithms-pymoo\"``.
    pip_extra_args : str, optional
        Extra flags appended to the install hint (for example
        ``\"--ignore-requires-python\"``).

    Returns
    -------
    module
        Imported top-level module.

    Raises
    ------
    AlgorithmDependencyError
        If the module cannot be imported because it is missing.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or module_name
        install = f'pip install -e ".[{extra}]"'
        if pip_extra_args.strip():
            install = f"{install} {pip_extra_args.strip()}"
        raise AlgorithmDependencyError(
            f"{algorithm_name} requires optional dependency '{missing}'; "
            f"install with:\n{install}"
        ) from exc
