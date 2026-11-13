"""
Runtime protocol for optimisation algorithms that evaluate IOH problems.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IOHAlgorithm(Protocol):
    """
    Minimal callable contract used by runners and workers.

    Attributes
    ----------
    name : str
        Stable algorithm token written into IOH logger metadata.
    info : str
        Human-readable algorithm description.

    Notes
    -----
    ``__call__`` may return any object for backward compatibility. Authoritative
    results are always read from ``problem.state`` after the call.

    ``run_controller`` is optional. When omitted, adapters create a fixed-budget
    controller from their own ``max_evaluations``. Workers in Phase C4 pass an
    explicit controller for fixed-target runs.
    """

    name: str
    info: str

    def __call__(
        self,
        problem: Any,
        *,
        run_controller: Any | None = None,
    ) -> object:
        """
        Run the optimiser on one IOH integer problem instance.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds, metadata, callable evaluation, and
            ``state``.
        run_controller : Any or None, optional
            Optional target-aware evaluation controller. When ``None``, the
            adapter must resolve a fixed-budget controller from its own
            ``max_evaluations``.

        Returns
        -------
        object
            Non-authoritative library/wrapper result (may be ``None`` or
            ``self``).
        """
        ...
