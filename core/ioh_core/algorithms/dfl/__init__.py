"""
DFL-backed optimisation algorithms (optional extra ``algorithms-dfl``).

DFLINT is vendored under ``_vendor/`` (GPL-3.0). Halton directions use the
SciPy shim in :mod:`core.ioh_core.algorithms.dfl.halton` when PyPI
``ghalton`` is not installed. Concrete adapters load only through lazy
registry constructors.
"""
from __future__ import annotations

__all__: tuple[str, ...] = ()
