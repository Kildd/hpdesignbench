"""
RBFOpt-backed optimisation algorithms (optional extra ``algorithms-rbfopt``).

This package stays importable when RBFOpt is absent. Concrete adapters and the
third-party library are loaded only through lazy registry constructors.
External Bonmin/Ipopt solvers are resolved at construction time, never at
package or registry import.
"""
from __future__ import annotations

__all__: tuple[str, ...] = ()
