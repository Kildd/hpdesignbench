"""
Geometry class for hyperbolic paraboloid (HP) shells.

Provides parametric computation of HP-shell geometry including tendon
coordinates, cross-section polygons, and concrete volume.

Adapted from: Jamila Loutfi :cite:`loutfi_2023`

Translated into Python by: Elliot Melcer
"""
import numpy as np
from numpy import sqrt
from shapely import LineString, Polygon


class HPGeometry:
    """
    Parametric representation of an HP-shell geometry.

    The shell is described by its plan dimensions B (width) and L (span),
    rise heights H_x and H_y in the two principal directions, shell
    thickness t, and tendon layout parameters dy and nt.

    The coordinate system is centred at the shell midpoint:
    x ∈ [−L/2, L/2], y ∈ [−B/2, B/2], z upward.

    Attributes
    ----------
    B : float
        Shell width [mm].
    L : float
        Shell span [mm].
    Hx : float
        Rise in the x-direction (hog along span) [mm].
    Hy : float
        Rise in the y-direction (sag across width) [mm].
    t : float
        Shell thickness [mm].
    dy : float
        Distance from the outermost tendon to the shell edge [mm].
    nt : int
        Number of tendons per tendon group; total tendon count is 2 · nt [-].
    """

    def __init__(
            self,
            B: float,
            L: float,
            Hx: float,
            Hy: float,
            t: float,
            dy: float,
            nt: int,
    ):
        """
        Parameters
        ----------
        B : float
            Shell width [mm].
        L : float
            Shell span [mm].
        Hx : float
            Rise in the x-direction [mm].
        Hy : float
            Rise in the y-direction [mm].
        t : float
            Shell thickness [mm].
        dy : float
            Distance from the outermost tendon to the shell edge [mm].
        nt : int
            Number of tendons per tendon group [-].
        """
        self.B = float(B)
        self.L = float(L)
        self.Hx = float(Hx)
        self.Hy = float(Hy)
        self.t = float(t)
        self.dy = float(dy)
        self.nt = nt

    def param_a(self) -> float:
        """
        Compute the HP-surface shape parameter a.
        Adapted from: Jamila Loutfi

        Returns
        -------
        float
            Shape parameter a [mm^{1/2}].
        """
        a = self.L / (2 * sqrt(self.Hx))
        return a

    def param_b(self) -> float:
        """
        Compute the HP-surface shape parameter b.
        Adapted from: Jamila Loutfi

        Returns
        -------
        float
            Shape parameter b [mm^{1/2}].
        """
        b = self.B / (2 * sqrt(self.Hy))
        return b

    def _z(self, x: float, y: float) -> float:
        """
        Compute the z-coordinate of the HP mid-surface at (x, y).
        Adapted from: Jamila Loutfi

        Parameters
        ----------
        x : float
            Longitudinal coordinate (centred at shell mid-point) [mm].
        y : float
            Transverse coordinate (centred at shell mid-point) [mm].

        Returns
        -------
        float
            Mid-surface elevation z [mm].
        """
        z = y ** 2 / self.param_b() ** 2 - x ** 2 / self.param_a() ** 2
        return z

    def x_p(self) -> float:
        """
        Compute the x-coordinate of the four HP corner points.
        Adapted from: Jamila Loutfi

        Returns
        -------
        float
            x-coordinate of corner points [mm].
        """
        x_p = self.L/2 * (1 + sqrt(self.Hy)/sqrt(self.Hx))
        return x_p

    def y_p(self) -> float:
        """
        Compute the y-coordinate of the four HP corner points.
        Adapted from: Jamila Loutfi

        Returns
        -------
        float
            y-coordinate of corner points [mm].
        """
        y_p = self.B/2 * (1 + sqrt(self.Hx)/sqrt(self.Hy))
        return y_p

    def z_p(self) -> float:
        """
        Compute the z-coordinate of the four HP corner points.
        Adapted from: Jamila Loutfi

        Returns
        -------
        float
            z-coordinate of corner points [mm].
        """
        z_p = (sqrt(self.Hx)+sqrt(self.Hy))**2
        return z_p

    def dy_real(self) -> float:
        """
        Return the effective edge-tendon offset dy.
        Adapted from: Jamila Loutfi

        For ``nt = 1`` the single tendon is placed at the geometric midpoint
        of the tendon group (α = 0.5) and dy is back-calculated from that
        position. For ``nt > 1`` the user-specified :attr:`dy` is returned
        unchanged.

        Returns
        -------
        float
            Effective edge-tendon offset [mm].
        """
        alpha_nt_1 = 0.5

        if self.nt == 1:
            dy_real = self.B / 2 + ((-self.L / 2) / self.x_p() + 2 * alpha_nt_1 - 1) * self.y_p()
            return dy_real
        else:
            return self.dy

    def alpha_edge(self) -> float:
        """
        Compute the normalized α-coordinate of the outermost tendon near
        the shell edge.
        Adapted from: Jamila Loutfi

        α is the parametric coordinate along the tendon group direction,
        with α = 0 at one edge and α = 1 at the opposite edge.

        Returns
        -------
        float
            α-coordinate of the outermost edge tendon [-].
        """
        alpha_edge = 1/2 * ( (-self.B/2 + self.dy)/self.y_p() + (self.L/2)/self.x_p() + 1)
        return alpha_edge

    def alpha_edge_bar(self) -> float:
        """
        Compute the complementary α-coordinate of the outermost tendon.
        Adapted from: Jamila Loutfi

        Defined as 1 − α_edge, clamped to 0.5 if α_edge > 0.5 (symmetric
        layout).

        Returns
        -------
        float
            Complementary α-coordinate [-].
        """
        alpha_edge_bar = 1-self.alpha_edge()

        if self.alpha_edge() > 0.5:
            alpha_edge_bar = 0.5
        return alpha_edge_bar

    def delta_alpha(self) -> float:
        """
        Compute the uniform α-spacing between adjacent tendons in one group.
        Adapted from: Jamila Loutfi

        Returns
        -------
        float
            α-spacing between tendons [-].
        """
        delta_alpha = (self.alpha_edge_bar() - self.alpha_edge()) / (self.nt - 1)
        return delta_alpha

    def alpha_list(self) -> list[float]:
        """
        Return the α-coordinates of all tendons in one tendon group.
        Adapted from: Jamila Loutfi

        For ``nt = 1`` the single tendon is placed at α = 0.5 and
        Δα = 0. For ``nt > 1`` tendons are distributed uniformly from
        :meth:`alpha_edge` to :meth:`alpha_edge_bar`.

        Returns
        -------
        list[float]
            α-coordinates for the nt tendons in one group [-].
        """
        _alpha = self.alpha_edge() # local alpha variable that is subject to change if nt = 1

        if self.nt == 1:
            # if there is only one tendon, correct alpha value
            # 0.5 ist der richtige Wert
            _alpha = 0.5
            delta_alpha = 0
        else:
            delta_alpha = self.delta_alpha()

        alpha_list = []
        for i in range(self.nt):
            alpha_i = _alpha + delta_alpha * i
            alpha_list.append(alpha_i)

        return alpha_list

    def gt_x(self) -> tuple[list[float], list[float]]:
        """
        Return the start and end x-coordinates for all tendons in one group.
        Adapted from: Jamila Loutfi

        All tendons span the full length, so x_start = −L/2 and
        x_end = +L/2 for each tendon.

        Returns
        -------
        tuple[list[float], list[float]]
            ``(gt_x_start, gt_x_end)`` — lists of length nt [mm].
        """
        gt_x_start = [-self.L/2] * self.nt
        gt_x_end = [self.L/2] * self.nt
        return   gt_x_start, gt_x_end

    def gt_y(self) -> tuple[list[float], list[float]]:
        """
        Return the start and end y-coordinates for all tendons in one group.
        Adapted from: Jamila Loutfi

        y-coordinates are computed from the α-parametrization and the
        corner point y_p.

        Returns
        -------
        tuple[list[float], list[float]]
            ``(gt_y_start, gt_y_end)`` — lists of length nt [mm].
        """
        gt_x_start, gt_x_end = self.gt_x()
        x_start, x_end = gt_x_start[0], gt_x_end[0]

        gt_y_start = []
        gt_y_end = []

        for alpha in self.alpha_list():
            y_st = (x_start / self.x_p() + 2 * alpha - 1) * self.y_p()
            gt_y_start.append(y_st)

            y_end = (x_end / self.x_p() + 2 * alpha - 1) * self.y_p()
            gt_y_end.append(y_end)

        return gt_y_start, gt_y_end

    def gt_z(self) -> tuple[list[float], list[float]]:
        """
        Return the start and end z-coordinates for all tendons in one group.
        Adapted from: Jamila Loutfi

        z-coordinates follow the HP mid-surface parametrized by α.

        Returns
        -------
        tuple[list[float], list[float]]
            ``(gt_z_start, gt_z_end)`` — lists of length nt [mm].
        """
        gt_x_start, gt_x_end = self.gt_x()
        x_start, x_end = gt_x_start[0], gt_x_end[0]

        gt_z_start = []
        gt_z_end = []

        for alpha in self.alpha_list():
            z_st = (4 * alpha * x_start / self.x_p() - 2 * x_start / self.x_p() + 4 * alpha**2 - 4 * alpha + 1) * self.z_p()
            gt_z_start.append(z_st)

            z_end = (4 * alpha * x_end   / self.x_p() - 2 * x_end   / self.x_p() + 4 * alpha**2 - 4 * alpha + 1) * self.z_p()
            gt_z_end.append(z_end)

        return gt_z_start, gt_z_end

    def tendons(self) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """
        Return all tendons as start/end 3-D coordinate pairs.
        Author: Elliot Melcer

        The list contains the regular tendon group followed by the y-mirrored
        group, giving 2 · nt tendons in total.

        Returns
        -------
        list[tuple[tuple[float, float, float], tuple[float, float, float]]]
            Each entry is ``((x_start, y_start, z_start), (x_end, y_end, z_end))``
            [mm].
        """
        gt_x_start, gt_x_end = self.gt_x()
        gt_y_start, gt_y_end  = self.gt_y()
        gt_z_start, gt_z_end = self.gt_z()

        tendon_list = []

        # regular tendon group
        for xs, xe, ys, ye, zs, ze in zip(gt_x_start, gt_x_end, gt_y_start, gt_y_end, gt_z_start, gt_z_end):
            start_point = (xs, ys, zs)
            end_point = (xe, ye, ze)
            tendon_list.append((start_point, end_point))

        # mirrored tendon group
        for xs_m, xe_m, ys_m, ye_m, zs_m, ze_m in zip(reversed(gt_x_start), reversed(gt_x_end), reversed(gt_y_start), reversed(gt_y_end), reversed(gt_z_start), reversed(gt_z_end)):
            start_point_m = (xs_m, -ys_m, zs_m)
            end_point_m = (xe_m, -ye_m, ze_m)
            tendon_list.append((start_point_m, end_point_m))

        return tendon_list

    def tendon_coords_at_x(self, x: float) -> list[tuple[float, float]]:
        """
        Return tendon (y, z) coordinates in the cross-section plane at a
        given longitudinal coordinate x via linear interpolation.
        Adapted from: Jamila Loutfi
        Author: Elliot Melcer

        Parameters
        ----------
        x : float
            Normalized longitudinal coordinate, x ∈ [−0.5, 0.5] [-].

        Returns
        -------
        list[tuple[float, float]]
            ``(y, z)`` coordinates for all 2 · nt tendons in the
            cross-section plane [mm].
        """
        tendon_list = self.tendons()

        coords = []
        for (x0, y0, z0), (x1, y1, z1) in tendon_list:

            t = (x*self.L - x0) / (x1 - x0)  # linear interpolation parameter

            y = y0 + t * (y1 - y0)
            z = z0 + t * (z1 - z0)

            coords.append((y, z))

        return coords

    def midline(self, x: float, n: int) -> LineString:
        """
        Return the HP mid-surface polyline in the cross-section plane at x.
        Author: Elliot Melcer

        Parameters
        ----------
        x : float
            Longitudinal coordinate, centred at x = 0 [mm].
        n : int
            Number of sample points along the polyline [-].

        Returns
        -------
        LineString
            Shapely :class:`~shapely.LineString` of (y, z) coordinates [mm].
        """

        # Half-span in y at this x
        y_max = self.B / 2

        # Sample nt points along y
        ys = [(-y_max + 2 * y_max * i / (n - 1)) for i in range(n)]

        # Compute z(y)
        zs = [self._z(x, y) for y in ys]

        # Build LineString
        return LineString(zip(ys, zs))

    def polygon_section_at(self, x: float, n: int) -> Polygon:
        """
        Return the cross-section polygon at a normalized longitudinal position x.
        Author: Elliot Melcer

        The shell thickness t is applied perpendicular to the mid-surface.
        Bottom and top edges are each sampled at n points; the polygon is
        ordered bottom left-to-right, then top right-to-left.

        Parameters
        ----------
        x : float
            Normalized longitudinal coordinate, x ∈ [−0.5, 0.5] [-].
        n : int
            Number of sample points per edge [-].

        Returns
        -------
        Polygon
            Shapely :class:`~shapely.Polygon` representing the shell
            cross-section [mm].
        """
        # Compute local half-span in y for this x
        b = self.param_b()

        # y max from shell boundary
        y_max = self.B / 2

        # Sample nt points along y
        ys = [(-y_max + 2 * y_max * i / (n - 1)) for i in range(n)]

        # Mid-surface z-values
        zs_mid = [self._z(x*self.L, y) for y in ys]

        # Normal directions in 2D (y,z) plane
        normals = []
        for y in ys:
            dzdy = (2 * y) / (b**2)
            length = sqrt(dzdy**2 + 1)
            ny = -dzdy / length   # y-component of unit normal
            nz = 1 / length       # z-component of unit normal
            normals.append((ny, nz))

        # Offset points for bottom and top layers (± t/2)
        t2 = self.t / 2
        bottom = [(ys[i] - normals[i][0] * t2,
                   zs_mid[i] - normals[i][1] * t2)
                  for i in range(n)]

        top = [(ys[i] + normals[i][0] * t2,
                zs_mid[i] + normals[i][1] * t2)
               for i in range(n)]

        # Polygon ordering: bottom L→R, then top R→L
        poly_points = bottom + top[::-1]

        return Polygon(poly_points)

    def volume(self) -> float:
        """
        Compute the concrete volume of the HP shell.
        Adapted from Loutfi's Grasshopper script ``volumen.gh``.

        The volume is calculated as the arc-length-corrected product of the
        curved surface area and the shell thickness t, using the closed-form
        integral of the HP parabolic arclength in both directions.

        Returns
        -------
        float
            Concrete volume of the HP shell [mm³].

        """
        y1 = self.B / 2
        y2 = -self.B / 2

        b1 = (8 * self.Hy * y1 * np.sqrt(64 * self.Hy ** 2 * y1 * y1 / (self.B ** 4) + 1) + self.B ** 2 * np.asinh(
            8 * self.Hy * y1 / (self.B ** 2))) / (16 * self.Hy)
        b2 = (8 * self.Hy * y2 * np.sqrt(64 * self.Hy ** 2 * y2 ** 2 / (self.B ** 4) + 1) + self.B ** 2 * np.asinh(
            8 * self.Hy * y2 / (self.B ** 2))) / (16 * self.Hy)

        b = b1 - b2

        x1 = self.L / 2
        x2 = -self.L / 2

        l1 = (8 * self.Hx * x1 * np.sqrt(64 * self.Hx ** 2 * x1 * x1 / (self.L ** 4) + 1) + self.L **2 * np.asinh(
            8 * self.Hx * x1 / (self.L ** 2))) / (16 * self.Hx)
        l2 = (8 * self.Hx * x2 * np.sqrt(64 * self.Hx ** 2 * x2 * x2 / (self.L ** 4) + 1) + self.L ** 2 * np.asinh(
            8 * self.Hx * x2 / (self.L ** 2))) / (16 * self.Hx)

        l = l1 - l2

        volume = l * b * self.t

        return volume