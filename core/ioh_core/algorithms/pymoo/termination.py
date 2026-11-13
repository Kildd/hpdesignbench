"""
Pymoo termination that tracks the shared IOH run controller.
"""
from __future__ import annotations

from typing import Any, MutableMapping

from pymoo.core.termination import Termination


class ControllerAwareTermination(Termination):
    """
    Terminate a Pymoo run after the shared controller requests a stop.

    Holds only a picklable ``stop_flag`` mapping updated by
    :class:`~core.ioh_core.algorithms.pymoo.problem.IOHIntegerProblem` so the
    IOH C++ problem object is never pickled by Pymoo.
    """

    def __init__(self, stop_flag: MutableMapping[str, bool]) -> None:
        """
        Parameters
        ----------
        stop_flag : mutable mapping
            Mapping with a boolean ``\"stop\"`` entry set by the problem
            adapter after each evaluation.
        """
        super().__init__()
        self.stop_flag = stop_flag

    def _update(self, algorithm: Any) -> float:
        """
        Return termination progress for the current algorithm state.

        Parameters
        ----------
        algorithm : Any
            Pymoo algorithm instance (unused; IOH state is authoritative).

        Returns
        -------
        float
            ``1.0`` when the stop flag is set, otherwise ``0.0``.
        """
        del algorithm
        return 1.0 if bool(self.stop_flag.get("stop")) else 0.0
