"""
Native optimisation algorithms.

``RandomSearch`` and ``LatinHypercubeSampling`` use only NumPy. Discrete
MLSL and population basin hopping also use SciPy for Sobol draws and the
MLSL gamma factor. None of them need an optional dependency extra.
"""
from __future__ import annotations

from core.ioh_core.algorithms.native.discrete_search import (
    DiscreteMLSL,
    DiscreteMLSLDeb,
    PopulationBasinHopping,
    PopulationBasinHoppingDeb,
    PopulationBasinHoppingDebWarmStart,
)
from core.ioh_core.algorithms.native.k_elite_ams import KEliteAMS, KEliteAMSDeb
from core.ioh_core.algorithms.native.latin_hypercube import LatinHypercubeSampling
from core.ioh_core.algorithms.native.random_search import RandomSearch

__all__ = [
    "DiscreteMLSL",
    "DiscreteMLSLDeb",
    "KEliteAMS",
    "KEliteAMSDeb",
    "LatinHypercubeSampling",
    "PopulationBasinHopping",
    "PopulationBasinHoppingDeb",
    "PopulationBasinHoppingDebWarmStart",
    "RandomSearch",
]
