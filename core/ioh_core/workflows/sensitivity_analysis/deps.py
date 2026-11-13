"""
Optional dependencies for the sensitivity-analysis workflow.

Call :func:`import_salib` from sampling / analyse code so a missing
``sensitivity`` extra yields an actionable install hint instead of a bare
``ModuleNotFoundError``.

Importing this module must not load SALib or PyArrow.
"""
from __future__ import annotations

import importlib
from types import ModuleType


class SensitivityDependencyError(ImportError):
    """
    Raised when an optional sensitivity-analysis extra is not installed.

    Subclasses ``ImportError`` so callers can distinguish a missing extra from
    invalid sensitivity configuration.
    """


def import_salib() -> ModuleType:
    """
    Import ``SALib`` or raise an actionable dependency error.

    Returns
    -------
    module
        Imported ``SALib`` top-level module.

    Raises
    ------
    SensitivityDependencyError
        If ``SALib`` cannot be imported because it is missing.
    """
    try:
        return importlib.import_module("SALib")
    except ModuleNotFoundError as exc:
        missing = exc.name or "SALib"
        raise SensitivityDependencyError(
            f"sensitivity_analysis requires optional dependency '{missing}'; "
            f'install with:\npip install -e ".[sensitivity]"'
        ) from exc


def import_pyarrow() -> ModuleType:
    """
    Import ``pyarrow`` or raise an actionable dependency error.

    Returns
    -------
    module
        Imported ``pyarrow`` top-level module.

    Raises
    ------
    SensitivityDependencyError
        If ``pyarrow`` cannot be imported because it is missing.
    """
    try:
        return importlib.import_module("pyarrow")
    except ModuleNotFoundError as exc:
        missing = exc.name or "pyarrow"
        raise SensitivityDependencyError(
            f"sensitivity_analysis requires optional dependency '{missing}'; "
            f'install with:\npip install -e ".[sensitivity]"'
        ) from exc
