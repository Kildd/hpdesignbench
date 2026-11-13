"""
Optional dependencies for the ELA analysis package.

Call :func:`import_pyarrow` and :func:`import_duckdb` from Parquet/SQL-facing
code so a missing ``la`` extra yields an actionable install hint instead of
a bare ``ModuleNotFoundError``. Call :func:`import_numdifftools` and
:func:`import_pflacco` from native-family code paths only.

The ``la`` extra is pyarrow, duckdb, and
``numdifftools>=0.9.40,<1``. It does **not** list ``pflacco``: a normal
``pflacco==1.2.2`` extra would pull ``numpy~=1.24`` / ``pandas<2.1`` and
break this project. Native sampled-ELA families need a second step::

    pip install -e ".[la]"
    pip install pflacco==1.2.2 --no-deps

Do not ``pip install pflacco`` with dependencies. Pickle-first qualification
against ``Reiyan/pflacco`` tag ``v1.2.2`` (commit
``4689566adcd20b86b4064d891d5e4e37c6e2ceca``) on the project Python 3.12
stack matched historical classical features to float noise (test gate
``np.allclose`` with ``atol=1e-9`` and ``rtol=1e-12``, excluding
``*.costs_runtime``). Cell-mapping public
APIs are not used; see Phase 17.

Importing this module must not load PyArrow, DuckDB, pflacco, numdifftools,
or SALib.
"""
from __future__ import annotations

import importlib
from types import ModuleType


class ElaDependencyError(ImportError):
    """
    Raised when an optional ELA extra is not installed.

    Subclasses ``ImportError`` so callers can distinguish a missing extra from
    invalid ELA configuration.
    """


_LA_EXTRA_HINT = 'pip install -e ".[la]"'
_PFLACCO_HINT = (
    "sampled ELA families require pflacco==1.2.2 installed without its "
    "declared pins:\n"
    'pip install -e ".[la]"\n'
    "pip install pflacco==1.2.2 --no-deps\n"
    "Do not pip install pflacco with dependencies "
    "(that pulls numpy~=1.24)."
)


def import_pyarrow() -> ModuleType:
    """
    Import ``pyarrow`` or raise an actionable dependency error.

    Returns
    -------
    module
        Imported ``pyarrow`` top-level module.

    Raises
    ------
    ElaDependencyError
        If ``pyarrow`` cannot be imported because it is missing.
    """
    try:
        return importlib.import_module("pyarrow")
    except ModuleNotFoundError as exc:
        missing = exc.name or "pyarrow"
        raise ElaDependencyError(
            f"landscape analysis requires optional dependency '{missing}'; "
            f"install with:\n{_LA_EXTRA_HINT}"
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
    ElaDependencyError
        If ``duckdb`` cannot be imported because it is missing.
    """
    try:
        return importlib.import_module("duckdb")
    except ModuleNotFoundError as exc:
        missing = exc.name or "duckdb"
        raise ElaDependencyError(
            f"landscape analysis requires optional dependency '{missing}'; "
            f"install with:\n{_LA_EXTRA_HINT}"
        ) from exc


def import_numdifftools() -> ModuleType:
    """
    Import ``numdifftools`` or raise an actionable dependency error.

    ``pflacco.classical_ela_features`` imports this at module top.
    It is part of the ``ela`` extra, not a pflacco ``Requires-Dist`` install.

    Returns
    -------
    module
        Imported ``numdifftools`` top-level module.

    Raises
    ------
    ElaDependencyError
        If ``numdifftools`` cannot be imported because it is missing.
    """
    try:
        return importlib.import_module("numdifftools")
    except ModuleNotFoundError as exc:
        missing = exc.name or "numdifftools"
        raise ElaDependencyError(
            f"landscape analysis requires optional dependency '{missing}'; "
            f"install with:\n{_LA_EXTRA_HINT}"
        ) from exc


def import_pflacco() -> ModuleType:
    """
    Import ``pflacco`` or raise an actionable ``--no-deps`` install error.

    Does not import ``pflacco.sampling``. Does not satisfy upstream
    ``Requires-Dist`` (numpy 1.24 / pandas < 2.1).

    Returns
    -------
    module
        Imported ``pflacco`` top-level module.

    Raises
    ------
    ElaDependencyError
        If ``pflacco`` cannot be imported because it is missing.
    """
    try:
        return importlib.import_module("pflacco")
    except ModuleNotFoundError as exc:
        raise ElaDependencyError(_PFLACCO_HINT) from exc
