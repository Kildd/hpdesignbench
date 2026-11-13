"""
PFNs4BO-backed optimisation algorithms (optional extra ``algorithms-pfns4bo``).

Requires ``pip install -e ".[algorithms-pfns4bo]"`` (``pfns4bo``, ``torch``,
``torchvision``). HEBO+ model weights download on first use via
``pfns4bo.prepare_models()``. Concrete adapters load only through lazy
registry constructors.
"""
from __future__ import annotations

__all__: tuple[str, ...] = ()
