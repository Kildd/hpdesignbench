"""
Construction feasibility checks for HP-shell slab constructions
Numbering according to Loutfi :cite:`loutfi_2023`

Verifies that the chosen geometry and reinforcement layout are
constructable: sufficient concrete cover, adequate bar spacing, and
minimum shell thickness.

Adapted from: Jamila Loutfi :cite:`loutfi_2023`
"""
from abc import ABC, abstractmethod

from slab_construction.slab_construction import SlabConstruction
from core.analysis_core.run_config import require_invalid_utilization


class ConstructionCheck(ABC):
    """
    Abstract base class for construction feasibility checks.
    Author: Elliot Melcer

    All ConstructionCheck subclasses must implement :meth:`calculate_utilization`,
    which returns a dimensionless utilization ratio where values ≤ 1.0
    indicate a constructable design.
    """

    @staticmethod
    @abstractmethod
    def calculate_utilization(
            slab_construction: SlabConstruction) -> float:
        """
        Compute the construction utilization ratio for a slab construction.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.

        Returns
        -------
        float
            Utilization ratio [-]. Values ≤ 1.0 indicate a passing design.
        """
        raise NotImplementedError


"""C.1. Check for Sufficient Concrete Cover from the outermost Reinforcement to the Edge along the HP-Shell Midline """


class MidlineConcreteCoverCheck(ConstructionCheck):
    """
    Concrete cover check along the HP-shell midline (C.1).

    Verifies that the available clear concrete cover c_1 satisfies the
    minimum requirement c_nom,req = 3 · d_p, where d_p is the equivalent
    bar diameter of the outermost reinforcement layer.

    A utilization ≤ 1.0 means the concrete cover is sufficient. Infeasible
    designs (c_1 = 0) return ``opt_invalid_utilization``.
    """

    @staticmethod
    def calculate_utilization(
        slab_construction: SlabConstruction,
        *,
        invalid_utilization: float | None = None,
    ) -> float:
        """
        Compute the concrete cover utilization ratio c_nom,req / c_1.

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
            Utilization ratio c_nom,req / c_1 [-].
        """
        hp_shell = slab_construction.slab.hp_shell

        d_p_mm = hp_shell.d_p()
        c_nom_c1_req = 3 * d_p_mm
        c_nom_c1 = hp_shell.c_1_clear_concrete_cover()

        if c_nom_c1 == 0:
            return require_invalid_utilization(invalid_utilization)

        return c_nom_c1_req / c_nom_c1


"""C.2. Check for Sufficient Clear Spacing between Reinforcement along the HP-Shell Midline """


class ReinforcementSpacingCheck(ConstructionCheck):
    """
    Clear reinforcement spacing check along the HP-shell midline (C.2).

    Verifies that the minimum available spacing s_min between
    reinforcement bars satisfies s_req = 6 · d_p.

    A utilization ≤ 1.0 means the spacing is sufficient. Infeasible
    designs (s_min = 0) return ``opt_invalid_utilization``.
    """

    @staticmethod
    def calculate_utilization(
        slab_construction: SlabConstruction,
        debug_print: bool = False,
        *,
        invalid_utilization: float | None = None,
    ) -> float:
        """
        Compute the reinforcement spacing utilization ratio s_req / s_min.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        debug_print : bool, optional
            If ``True``, prints the minimum available clear spacing.
        invalid_utilization : float or None, optional
            Explicit invalid-utilization sentinel from the run config.
            Defaults to ``opt_invalid_utilization``.

        Returns
        -------
        float
            Utilization ratio s_req / s_min [-].
        """
        hp_shell = slab_construction.slab.hp_shell

        d_p_mm = hp_shell.d_p()
        s_req = 6.0 * d_p_mm
        s_min = hp_shell.s_min_clear_reinf_spacing()

        if debug_print:
            print(f"s_min = {s_min}")

        if s_min == 0:
            return require_invalid_utilization(invalid_utilization)

        return s_req / s_min


"""C.3. Check for Minimum Shell Thickness """


class MinimumHPShellThicknessCheck(ConstructionCheck):
    """
    Minimum HP-shell thickness check (C.3).

    Required thickness from C.1 and C.2 with GH spacing:

        t_req = 2·d_p + 2·(3·d_p) + 6·d_p = 14·d_p

    A utilization ≤ 1.0 means the shell is thick enough.
    """

    @staticmethod
    def calculate_utilization(slab_construction: SlabConstruction) -> float:
        """
        Compute the shell thickness utilization ratio t_req / t.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.

        Returns
        -------
        float
            Utilization ratio t_req / t [-].
        """
        hp_shell = slab_construction.slab.hp_shell

        d_p_mm = hp_shell.d_p()
        c_nom_c1_req = 3 * d_p_mm
        s_c2_req = 6.0 * d_p_mm
        t_req = 2 * d_p_mm + 2 * c_nom_c1_req + s_c2_req

        t = hp_shell.hp_geometry.t
        return t_req / t
