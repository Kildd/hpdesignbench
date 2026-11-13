"""
Abstract base class for one-way slab elements.

Author: Elliot Melcer
"""
from abc import ABC, abstractmethod

from structuralcodes.sections import BeamSection

from slab_construction.slabs.slab import Slab


class OneWaySlab(Slab, ABC):
    """
    Abstract base class for one-way slabs.

    Concrete subclasses must implement :attr:`L`, :attr:`B`, and
    :meth:`section_at`.
    """

    @property
    @abstractmethod
    def L(self) -> float:
        """Span length [mm]."""
        pass

    @property
    @abstractmethod
    def B(self) -> float:
        """Width [mm]."""
        pass

    @abstractmethod
    def section_at(self, x: float) -> BeamSection:
        r"""
        Return the structural cross-section at a normalized position.

        The position convention is::

              |-> x
              0      0.5      1      1.5      2

              ====================================|...
             /_\             /_\             /_\

        x = 0 at the first support, x = 1 at the second support, etc.

        Parameters
        ----------
        x : float
            Normalized longitudinal coordinate [-].

        Returns
        -------
        BeamSection
            Cross-section at position x · L.
        """
        pass