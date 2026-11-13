"""
SciPy algorithm adapters for IOHexperimenter problems.
"""
from __future__ import annotations

from core.ioh_core.algorithms.scipy.cobyla import Cobyla
from core.ioh_core.algorithms.scipy.differential_evolution import (
    DifferentialEvolution,
)
from core.ioh_core.algorithms.scipy.direct_l import DirectL
from core.ioh_core.algorithms.scipy.dual_annealing import DualAnnealing

__all__ = [
    "Cobyla",
    "DifferentialEvolution",
    "DirectL",
    "DualAnnealing",
]
