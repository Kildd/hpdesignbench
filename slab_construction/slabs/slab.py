"""
Abstract base class for slab elements.
"""
from abc import ABC, abstractmethod


class Slab(ABC):
    """Abstract base class for slab elements."""

    @abstractmethod
    def self_load(self) -> float:
        """
        Compute the self-weight area load of the slab.

        Returns
        -------
        float
            Self-weight load [kN/m²].
        """
        pass

    @abstractmethod
    def infill_load(self) -> float:
        """
        Compute the area load due to infill on the slab.

        Returns
        -------
        float
            Infill load [kN/m²].
        """
        pass