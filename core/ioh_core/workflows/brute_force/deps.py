"""
Optional dependencies for the brute-force workflow and analysis toolbox.

Call :func:`import_pyarrow` from Parquet-facing code so a missing
``bruteforce`` extra yields an actionable install hint instead of a bare
``ModuleNotFoundError``. Call :func:`import_duckdb` from analysis code so a
missing ``bruteforce-analysis`` extra yields a similar actionable hint.

Importing this module must not load PyArrow or DuckDB.
"""
from __future__ import annotations

import importlib
from types import ModuleType


class BruteForceDependencyError(ImportError):
    """
    Raised when an optional brute-force / analysis extra is not installed.

    Subclasses ``ImportError`` so callers can distinguish a missing extra from
    invalid brute-force configuration.
    """


def import_pyarrow() -> ModuleType:
    """
    Import ``pyarrow`` or raise an actionable dependency error.

    Returns
    -------
    module
        Imported ``pyarrow`` top-level module.

    Raises
    ------
    BruteForceDependencyError
        If ``pyarrow`` cannot be imported because it is missing.
    """
    try:
        return importlib.import_module("pyarrow")
    except ModuleNotFoundError as exc:
        missing = exc.name or "pyarrow"
        raise BruteForceDependencyError(
            f"brute_force requires optional dependency '{missing}'; "
            f'install with:\npip install -e ".[bruteforce]"'
        ) from exc


def import_duckdb() -> ModuleType:
    """
    Import ``duckdb`` or raise an actionable dependency error.

    Returns
    -------
    module
        Imported ``duckdb`` top-level module.

    Raises
    ------
    BruteForceDependencyError
        If ``duckdb`` cannot be imported because it is missing.
    """
    try:
        return importlib.import_module("duckdb")
    except ModuleNotFoundError as exc:
        missing = exc.name or "duckdb"
        raise BruteForceDependencyError(
            f"brute_force analysis requires optional dependency '{missing}'; "
            f'install with:\npip install -e ".[bruteforce-analysis]"'
        ) from exc
