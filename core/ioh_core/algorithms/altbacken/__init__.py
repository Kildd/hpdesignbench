"""
Altbacken-backed optimisation algorithms (optional extra ``algorithms-altbacken``).

This package stays importable when altbacken is absent. Concrete adapters and
the third-party library are loaded only through lazy registry constructors.

On Python 3.12, install with::

    pip install -e ".[algorithms-altbacken]" --ignore-requires-python
"""
from __future__ import annotations

__all__: tuple[str, ...] = ()
