"""
NOMAD / PyNomad-backed optimisation algorithms (optional extra ``algorithms-nomad``).

This package stays importable when PyNomadBBO is absent. Concrete adapters and
the third-party library are loaded only through lazy registry constructors.

Install::

    pip install -e ".[algorithms-nomad]"
"""
from __future__ import annotations

__all__: tuple[str, ...] = ()
