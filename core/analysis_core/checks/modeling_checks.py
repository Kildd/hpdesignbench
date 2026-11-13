"""
Modeling validity checks for HP-shell slab constructions.
Numbering according to Loutfi :cite:`loutfi_2023`

Verifies that the chosen geometry lies within the validity domain of the
underlying beam-theory model: consistent nt/dy parametrization and
slenderness ratios H_ges/L and B/L.

Adapted from: Jamila Loutfi :cite:`loutfi_2023`
"""
from abc import ABC, abstractmethod

from slab_construction.slab_construction import SlabConstruction
from core.analysis_core.run_config import require_invalid_utilization


class ModelingCheck(ABC):
    """
    Abstract base class for modeling validity checks.
    Author: Elliot Melcer

    All ModelingCheck subclasses must implement :meth:`calculate_utilization`,
    which returns a dimensionless utilization ratio where values ≤ 1.0
    indicate that the geometry is within the model's validity range.
    """

    @staticmethod
    @abstractmethod
    def calculate_utilization(
            slab_construction: SlabConstruction
    ) -> float:
        """
        Compute the modeling validity utilization ratio for a slab construction.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.

        Returns
        -------
        float
            Utilization ratio [-]. Values ≤ 1.0 indicate a valid model
            configuration.
        """
        raise NotImplementedError


"""Z.1. Combination of nt and dy """


class NtDyCombinationCheck(ModelingCheck):
    """
    Consistency check for the nt / dy parameter combination (Z.1).

    When ``nt = 1``, the geometric parameter ``dy`` must match the
    value computed by ``dy_real()`` within a tolerance of 0.01. If the
    combination is inconsistent, ``opt_invalid_utilization`` is returned.
    For ``nt ≠ 1`` the check always passes (utilization 1.0).
    """

    @staticmethod
    def calculate_utilization(
        slab_construction: SlabConstruction,
        *,
        invalid_utilization: float | None = None,
    ) -> float:
        """
        Compute the nt/dy consistency utilization ratio.
        Adapted from: Jamila Loutfi :cite:`loutfi_2023`

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        invalid_utilization : float or None, optional
            Explicit invalid-utilization sentinel from the run config.
            Defaults to ``opt_invalid_utilization``.

        Returns
        -------
        float
            ``1.0`` if valid, invalid-utilization sentinel if inconsistent
            (only when ``nt = 1``).
        """
        hp_geometry = slab_construction.slab.hp_shell.hp_geometry

        nt = hp_geometry.nt
        dy = hp_geometry.dy
        dy_real = hp_geometry.dy_real()

        if nt == 1:
            if abs(dy - dy_real) > 0.01:
                utilization = require_invalid_utilization(invalid_utilization)
            else:
                utilization = 1.0
        else:
            utilization = 1.0

        return utilization


"""Z.2. Beam Theory H_ges / L - Ratio """


class BeamTheoryHgesLRatioCheck(ModelingCheck):
    """
    Beam-theory validity check for the (H_x + H_y) / L slenderness ratio (Z.2).

    Verifies that the total rise height (H_x + H_y) does not exceed L / f_b,
    the upper bound for beam-theory applicability. A ratio ≤ 1.0 is returned
    unchanged. Any larger ratio is outside the model and returns
    ``opt_invalid_utilization``.
    """

    @staticmethod
    def calculate_utilization(
            slab_construction: SlabConstruction,
            f_b: float = 5.0,
            *,
            invalid_utilization: float | None = None,
    ) -> float:
        """
        Compute the H_ges / L slenderness utilization ratio.
        Adapted from: Jamila Loutfi :cite:`loutfi_2023`

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        f_b : float, optional
            Slenderness limit factor such that the allowable rise height is
            L / f_b [mm/mm]. Default is ``5.0`` [-].
        invalid_utilization : float or None, optional
            Sentinel returned when the ratio exceeds 1. Required for an
            out-of-range geometry.

        Returns
        -------
        float
            Utilization ratio (H_x + H_y) / (L / f_b) when that ratio is
            at most 1. Otherwise ``opt_invalid_utilization``.
        """
        # get geometry
        hp_geometry = slab_construction.slab.hp_shell.hp_geometry

        Hx = hp_geometry.Hx
        Hy = hp_geometry.Hy
        L = hp_geometry.L

        utilization = (Hx + Hy) / (L / f_b)
        if utilization > 1.0:
            return require_invalid_utilization(invalid_utilization)
        return utilization


"""Z.3. Beam Theory B / L - Ratio """


class BeamTheoryBLRatioCheck(ModelingCheck):
    """
    Beam-theory validity check for the B / L aspect ratio (Z.3).

    Verifies that the slab width B does not exceed L / f_b, the upper
    bound for beam-theory applicability. A ratio ≤ 1.0 is returned
    unchanged. Any larger ratio is outside the model and returns
    ``opt_invalid_utilization``.
    """

    @staticmethod
    def calculate_utilization(
            slab_construction: SlabConstruction,
            f_b: float = 5.0,
            *,
            invalid_utilization: float | None = None,
    ) -> float:
        """
        Compute the B / L aspect-ratio utilization ratio.
        Adapted from: Jamila Loutfi :cite:`loutfi_2023`

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        f_b : float, optional
            Aspect-ratio limit factor such that the allowable width is
            L / f_b [mm/mm]. Default is ``5.0`` [-].
        invalid_utilization : float or None, optional
            Sentinel returned when the ratio exceeds 1. Required for an
            out-of-range geometry.

        Returns
        -------
        float
            Utilization ratio B / (L / f_b) when that ratio is at most 1.
            Otherwise ``opt_invalid_utilization``.
        """
        # get geometry
        hp_geometry = slab_construction.slab.hp_shell.hp_geometry

        L = hp_geometry.L
        B = hp_geometry.B

        utilization = B / (L / f_b)
        if utilization > 1.0:
            return require_invalid_utilization(invalid_utilization)
        return utilization
