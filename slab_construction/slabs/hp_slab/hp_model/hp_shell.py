"""
HP-shell structural section class.

Combines an :class:`HPGeometry` with concrete and reinforcement material
objects to build position-dependent :class:`BeamSection` instances
for moment-curvature and strength calculations.

Author: Elliot Melcer (unless stated otherwise)
"""
import math
from itertools import pairwise
from typing import Literal, Optional

import numpy as np
from numpy import sqrt
from shapely import LineString, Polygon
from structuralcodes.geometry import SurfaceGeometry, add_reinforcement
from structuralcodes.materials.concrete import Concrete
from structuralcodes.materials.reinforcement import Reinforcement
from structuralcodes.sections import BeamSection

from slab_construction.slabs.hp_slab.hp_model.hp_geometry import HPGeometry


class HPShell:
    """
    Hyperbolic paraboloid shell combining geometry and material properties.

    Builds :class:`~structuralcodes.sections.BeamSection` objects at
    arbitrary longitudinal positions for use in section analysis routines.

    Attributes
    ----------
    hp_geometry : HPGeometry
        HP-shell geometry object.
    concrete : Concrete
        Concrete material object.
    reinforcement : Reinforcement
        Reinforcement material object.
    reinf_area : float
        Cross-sectional area of a single reinforcement bar [mm²].
    name : str or None
        Optional label for the shell instance.
    integrator : {"fiber", "marin"}
        StructuralCodes section integrator used by :meth:`section_at`.
        Default is ``"fiber"``.
    mesh_size : float
        Fiber mesh size passed to :class:`BeamSection`, as a fraction of
        each polygon area in ``(0, 1]``. Default is ``0.01``. Marin does
        not use it; SLS rebuilds still copy the stored value.
    """

    def __init__(
            self,
            hp_geometry: HPGeometry,
            concrete: Concrete,
            reinforcement: Reinforcement,
            reinf_area: float,
            name: Optional[str] = None,
            *,
            integrator: Literal["fiber", "marin"] = "fiber",
            mesh_size: float = 0.01,
    ):
        """
        Parameters
        ----------
        hp_geometry : HPGeometry
            HP-shell geometry object.
        concrete : Concrete
            Concrete material object.
        reinforcement : Reinforcement
            Reinforcement material object (applied to all tendons).
        reinf_area : float
            Cross-sectional area of a single reinforcement bar [mm²].
        name : str or None, optional
            Optional label for the shell instance. Default is ``None``.
        integrator : {"fiber", "marin"}, optional
            StructuralCodes integrator for sections from :meth:`section_at`.
            Default is ``"fiber"``.
        mesh_size : float, optional
            Maximum triangle area as a fraction of each polygon area, in
            ``(0, 1]``. Default is ``0.01``.

        Raises
        ------
        ValueError
            If ``integrator`` is not ``"fiber"`` or ``"marin"``, or if
            ``mesh_size`` is outside ``(0, 1]``.
        """
        if integrator not in ("fiber", "marin"):
            raise ValueError(
                f"integrator must be 'fiber' or 'marin', got {integrator!r}"
            )
        if isinstance(mesh_size, bool) or not isinstance(mesh_size, (int, float)):
            raise ValueError(
                f"mesh_size must be a float in (0, 1], got {mesh_size!r}"
            )
        mesh_size_f = float(mesh_size)
        if not 0.0 < mesh_size_f <= 1.0:
            raise ValueError(
                f"mesh_size must be in (0, 1], got {mesh_size_f}"
            )
        self.hp_geometry = hp_geometry
        self.concrete = concrete
        self.reinforcement = reinforcement
        self.reinf_area = reinf_area
        self.name = name
        self.integrator = integrator
        self.mesh_size = mesh_size_f

    def section_at(self, x: float, name: Optional[str] = None) -> BeamSection:
        """
        Build the structural cross-section at a normalized longitudinal position.

        The concrete cross-section polygon is taken from
        :meth:`HPGeometry.polygon_section_at` and reinforcement bars are
        placed at the tendon coordinates from
        :meth:`HPGeometry.tendon_coords_at_x`. All 2 · nt tendons are
        given equal area :attr:`reinf_area`.

        Parameters
        ----------
        x : float
            Normalized longitudinal coordinate, x ∈ [0, 1], with 0.0 at
            the first support and 1.0 at the second support [-].
        name : str or None, optional
            Label for the returned section. Falls back to :attr:`self.name`
            if ``None``. Default is ``None``.

        Returns
        -------
        BeamSection
            Section object at position x · L.

        Raises
        ------
        ValueError
            If ``x`` is outside [0.0, 1.0].

        Notes
        -----
        Internally, x is shifted to the centred coordinate
        x_internal = x − 0.5 ∈ [−0.5, 0.5] before passing to the
        geometry methods.
        """
        # --- Input validation ---
        if not 0.0 <= x <= 1.0:
            raise ValueError(
                f"x must be between 0.0 and 1.0 (inclusive). Received {x}."
            )

        # Coordinate Transformation
        # External API uses x ∈ [0 ; 1], but internal geometry calculations use x ∈ [-0.5, 0.5]
        x_internal = x - 0.5

        # Concrete Geometry
        hp_geometry = SurfaceGeometry(
            poly=self.hp_geometry.polygon_section_at(x=x_internal, n=100), material=self.concrete
        )

        # Reinforcement Geometry
        reinforcement_points = self.hp_geometry.tendon_coords_at_x(x=x_internal)
        d = np.sqrt(4 * self.reinf_area / np.pi)

        # Add Reinforcement to Concrete Geometry
        for pt in reinforcement_points:
            hp_geometry = add_reinforcement(
                hp_geometry,
                pt,  # reinforcement points
                d,  # diameter [mm]
                self.reinforcement  # reinforcement material
            )

        section_name = self.name if name is None else name
        return BeamSection(
            hp_geometry,
            name=section_name,
            integrator=self.integrator,
            mesh_size=self.mesh_size,
        )

    def total_reinforcement_volume(self) -> float:
        """
        Compute the total reinforcement volume of the HP shell.

        Each tendon is modeled as a straight line segment with constant
        cross-sectional area :attr:`reinf_area`; the volume is the product
        of Euclidean tendon length and bar area, summed over all 2 · nt
        tendons.

        Returns
        -------
        float
            Total reinforcement volume [mm³].
        """
        total_tendon_length = 0.0
        for (xs, ys, zs), (xe, ye, ze) in self.hp_geometry.tendons():
            total_tendon_length += math.dist((xs, ys, zs), (xe, ye, ze))  # Euclidean distance

        volume = total_tendon_length * self.reinf_area
        return volume

    def net_concrete_volume(self) -> float:
        """
        Compute the net concrete volume (gross volume minus reinforcement volume).

        Returns
        -------
        float
            Net concrete volume [mm³].
        """
        net_concrete_volume = self.hp_geometry.volume() - self.total_reinforcement_volume()

        return net_concrete_volume

    def d_p(self) -> float:
        """
        Compute the equivalent circular diameter of the reinforcement bar.

        Derived from :attr:`reinf_area` assuming a circular cross-section:

        Returns
        -------
        float
            Equivalent bar diameter d_p [mm].
        """
        d_reinf = np.sqrt(4 * self.reinf_area / np.pi)

        return d_reinf

    def c_1_clear_concrete_cover(self) -> float:
        """
        Compute the available clear concrete cover along the HP-shell midline
        at the support cross-section.
        Adopted from Loutfi :cite:`loutfi_2023`

        The cover is measured as the arc length from the shell edge (y = −B/2)
        to the outermost tendon position, minus the tendon half-diameter.
        Returns ``0.0`` if the tendon overlaps the edge.

        Returns
        -------
        float
            Clear concrete cover c_1 along the midline [mm].
        """
        B = self.hp_geometry.B
        d_p = self.d_p()
        y_starts, _ = self.hp_geometry.gt_y()

        # include boundary y=-B/2 as starting point
        s_start = self.arc_length(-B / 2)

        s_first = self.arc_length(y_starts[0])

        c_1 = abs(s_start - s_first)

        if c_1 - d_p/2 < 0:
            c_1_clear = 0
        else:
            c_1_clear = c_1 - d_p/2

        return c_1_clear

    def s_min_clear_reinf_spacing(self) -> float:
        """
        Compute the minimum clear spacing between adjacent reinforcement bars
        along the HP-shell midline.
        Adopted from Loutfi :cite:`loutfi_2023`

        Arc lengths are computed for all 2 · nt tendon positions (both the
        regular and mirrored group). The clear spacing is the minimum
        pairwise arc-length difference minus the bar diameter. Returns ``0.0``
        if any two bars overlap.

        Returns
        -------
        float
            Minimum clear reinforcement spacing s_min along the midline [mm].
        """
        d_p = self.d_p()
        y_starts, _ = self.hp_geometry.gt_y()
        y_starts_mirrored = [-y for y in y_starts[::-1]]
        y_starts_complete = y_starts + y_starts_mirrored

        # arc-lengths relative to y=0
        s_vals = [self.arc_length(yi) for yi in y_starts_complete]

        # pairwise distances: y[0]->y[1], ... , y[n-1]->y[n]
        s = [abs(b - a) for a, b in pairwise(s_vals)]

        s_min = min(s)

        if s_min - d_p < 0:
            s_min_clear = 0
        else:
            s_min_clear = s_min - d_p

        return s_min_clear

    def arc_length(self, y: float) -> float:
        """
        Compute the arc length from the neutral axis (y = 0) to a given
        transverse coordinate y along the HP mid-surface parabola.
        Adopted from Loutfi :cite:`loutfi_2023`

        The sign of the returned value follows the sign of y (negative for
        y < 0, positive for y > 0).

        Parameters
        ----------
        y : float
            Transverse coordinate [mm].

        Returns
        -------
        float
            Signed arc length s(y) along the parabolic midline [mm].
        """
        b = self.hp_geometry.param_b()

        s_y = 1 / 4 * (2 * y * math.sqrt(((4 * y ** 2) / b ** 4) + 1) + b ** 2 * math.asinh((2 * y) / b ** 2))

        return s_y