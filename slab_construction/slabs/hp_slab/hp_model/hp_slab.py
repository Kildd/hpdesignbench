"""
HP-slab class combining an HP shell with infill material.

Extends :class:`OneWaySlab` with HP-shell-specific load and volume
calculations.

Author: Elliot Melcer
"""
from typing import Optional
from structuralcodes.sections import BeamSection
from core.unit_core import mm2_to_m2, mm3_to_m3
from slab_construction.floor import InfillMaterial
from slab_construction.slabs.one_way_slab import OneWaySlab
from slab_construction.slabs.hp_slab.hp_model.hp_shell import HPShell


class HPSlab(OneWaySlab):
    """
    Hyperbolic paraboloid slab consisting of an HP shell and infill material.

    Extends :class:`OneWaySlab` and exposes the span L and width B from
    the underlying :class:`HPShell` geometry.

    Attributes
    ----------
    hp_shell : HPShell
        HP shell object providing geometry and section methods.
    infill_material : InfillMaterial
        Infill material used to level the top surface of the HP shell.
    name : str or None
        Optional label for the slab instance.
    """

    def __init__(
            self,
            hp_shell: HPShell,
            infill_material: InfillMaterial,
            name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        hp_shell : HPShell
            HP shell object.
        infill_material : InfillMaterial
            Infill material for levelling the shell's top surface.
        name : str or None, optional
            Optional label. Default is ``None``.
        """
        self.hp_shell = hp_shell
        self.infill_material = infill_material
        self.name = name

    @property
    def L(self) -> float:
        """Span length of the HP shell [mm]."""
        return self.hp_shell.hp_geometry.L

    @property
    def B(self) -> float:
        """Width of the HP shell [mm]."""
        return self.hp_shell.hp_geometry.B

    def minimum_infill_volume(self) -> float:
        """
        Compute the minimum infill volume required fill the void between the
        flat reference plane and the curved top surface of the shell.

        Returns
        -------
        float
            Minimum infill volume [mm³].

        Notes
        -----
        Adopted from: Jamila Loutfi.
        """
        mid_surface_volume = abs(self.B * self.L * (-2 / 3 * self.hp_shell.hp_geometry.Hy - 1 / 3 * self.hp_shell.hp_geometry.Hx))

        min_infill_volume = mid_surface_volume - self.hp_shell.hp_geometry.volume() / 2

        return min_infill_volume

    def section_at(self, _x: float, name: Optional[str] = None) -> BeamSection:
        """
        Return the structural cross-section at a normalized position.

        Delegates directly to :meth:`HPShell.section_at`.

        Parameters
        ----------
        _x : float
            Normalized longitudinal coordinate, x ∈ [0, 1] [-].
        name : str or None, optional
            Label for the section. Default is ``None``.

        Returns
        -------
        BeamSection
            Cross-section at position _x · L.
        """
        return self.hp_shell.section_at(_x, name)

    def self_load(self) -> float:
        """
        Compute the self-weight area load of the concrete shell.

        The self-weight of the CFRP reinforcement is negligible and not
        included.

        Returns
        -------
        float
            Self-weight load of the concrete shell [kN/m²].
        """
        concrete_volume_m3 = mm3_to_m3(self.hp_shell.hp_geometry.volume())  # [m³]
        gamma_c = self.hp_shell.concrete.density * 10 / 1000                # [kN/m³]
        net_area = mm2_to_m2(self.B * self.L)                               # [m²]

        return concrete_volume_m3 * gamma_c / net_area                      # [kN/m²]

    def infill_load(self) -> float:
        """
        Compute the area load due to minimum infill on the slab.

        Returns
        -------
        float
            Infill area load [kN/m²].
        """
        infill_volume_m3 = mm3_to_m3(self.minimum_infill_volume())  # [m³]
        gamma_c = self.infill_material.density * 10 / 1000          # [kN/m³]
        net_area = mm2_to_m2(self.B * self.L)                       # [m²]

        return infill_volume_m3 * gamma_c / net_area                # [kN/m²]