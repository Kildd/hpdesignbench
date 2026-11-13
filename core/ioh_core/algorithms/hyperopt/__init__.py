"""
Hyperopt-backed optimisation algorithms (optional extra ``algorithms-hyperopt``).

This package stays importable when Hyperopt is absent. Concrete adapters and the
third-party library are loaded only through lazy registry constructors.
"""
from __future__ import annotations

__all__: tuple[str, ...] = ()
