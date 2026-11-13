"""
Matplotlib-based visualization utilities for structural section and
moment-curvature analysis results.

Author: Elliot Melcer
"""
import numpy as np
from matplotlib.collections import PolyCollection
from shapely.geometry.polygon import Polygon
from structuralcodes.core._section_results import MomentCurvatureResults
from structuralcodes.core.base import ConstitutiveLaw
from structuralcodes.geometry import Geometry
from structuralcodes.materials.concrete import Concrete
from structuralcodes.materials.constitutive_laws import Elastic
from structuralcodes.materials.reinforcement import Reinforcement
from structuralcodes.sections import BeamSection
from tabulate import tabulate
import typing as t
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.axes
from matplotlib.ticker import FuncFormatter

from core.analysis_core.material_methods import CrackingConcreteLawEC, TensionStiffeningConcreteLawEC
from core.analysis_core.section_methods import get_strain_at_point

TU_COLORS = {
    "BLACK":        "#000000",
    "DARK GREY":    "#434343",
    "LIGHT GREY":   "#b2b2b2",
    "RED":          "#c40d20",
    "ORANGE":       "#ff6e00",
    "VIOLET":       "#8f13fc",
    "BLUE":         "#1f91cc",
    "GREEN":        "#47cb3f",
}

# =====================================================================================================
# Cross Section
# =====================================================================================================

def plot_cross_section(gs: BeamSection, ax=None, x=None, title = "", **kwargs):
    """
    Plot the section geometry, centroid, and local coordinate axes.

    Parameters
    ----------
    gs : BeamSection
        Section geometry to plot.
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new figure is created when ``None``.
    x : float, optional
        Normalized longitudinal position, shown in the title [-].
    title : str, optional
        Additional title text. Default is ``""``.
    **kwargs
        Extra keyword arguments forwarded to the geometry plotter.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # ---- 1. Plot the geometry using geometry method ----
    _plot_geometry(gs.geometry, ax=ax, x=x, **kwargs)

    # ---- 2. Plot the section centroid ----
    cy = gs.gross_properties.cy
    cz = gs.gross_properties.cz

    ax.scatter(cy, cz, color="red", s=10, zorder=10)
    ax.text(cy, cz, f" C({cy:.2f}, {cz:.2f})",
            color="red", va="bottom", ha="left")

    # ---- 3. Plot local coordinate system at (0,0) ----
    x_min, x_max, y_min, y_max = gs.geometry.calculate_extents()
    L = 0.1 * (y_max-y_min)

    # y-axis → positive x direction
    ax.arrow(0, 0, L, 0,
             head_width=L * 0.3, head_length=L * 0.3,
             fc="black", ec="black", alpha = 0.4)
    ax.text(L * 1.5, 0, "y", va="center", ha="left", color="black", alpha = 0.4)

    # z-axis → positive y direction
    ax.arrow(0, 0, 0, L,
             head_width=L * 0.3, head_length=L * 0.3,
             fc="black", ec="black", alpha = 0.4)
    ax.text(0, L * 1.5, "z", va="bottom", ha="center", color="black", alpha = 0.4)

    # ---- 4. Final formatting ----
    ax.set_title(f"{gs.name} at x = {x} * L" if x is not None else f"{gs.name} \n {title}")
    ax.set_aspect("equal")

    return fig, ax

# =====================================================================================================
# Geometry
# =====================================================================================================

def _plot_geometry(geo: Geometry, ax=None, x = None, **kwargs):
    """
    Plot a :class:`Geometry`, :class:`SurfaceGeometry`, :class:`PointGeometry`,
    or :class:`CompoundGeometry` object recursively.

    Parameters
    ----------
    geo : Geometry
        Geometry object to plot.
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new figure is created when ``None``.
    x : float, optional
        Normalized longitudinal position, shown in the title [-].
    **kwargs
        Extra keyword arguments forwarded to the polygon plotter.

    Returns
    -------
    matplotlib.axes.Axes
    """

    if ax is None:
        fig, ax = plt.subplots()

    # Surface geometry
    if hasattr(geo, "polygon"):
        poly = geo.polygon
        _plot_polygon(poly, ax, **kwargs, edgecolor="grey", facecolor="lightgrey")

    # Point geometry
    if hasattr(geo, "point"):
        circ = geo.point.buffer(geo.diameter / 2)
        _plot_polygon(circ, ax, **kwargs, edgecolor="black", facecolor="black")

    # Compound geometry
    if hasattr(geo, "geometries"):
        for g in geo.geometries:
            _plot_geometry(g, ax=ax, show=False, **kwargs)
        for p in geo.point_geometries:
            _plot_geometry(p, ax=ax, show=False, **kwargs)

    ax.set_aspect("equal")

    # ---- Add title here ----
    ax.set_title(f"Cross-Section at x = {x} * L")

    # remove frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ax

def _plot_polygon(poly: Polygon, ax, edgecolor="black", facecolor="lightgrey", **kwargs):
    """
    Fill and outline a Shapely :class:`~shapely.Polygon`, including interior
    holes (rendered in white).

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Polygon to draw.
    ax : matplotlib.axes.Axes
        Axes to draw on.
    edgecolor : str, optional
        Edge color. Default is ``"black"``.
    facecolor : str, optional
        Fill color for the exterior. Default is ``"lightgrey"``.
    **kwargs
        Additional keyword arguments (currently unused).
    """
    # --- Fill exterior ---
    x, y = poly.exterior.xy
    ax.fill(x, y, facecolor=facecolor, edgecolor=edgecolor)

    # --- Fill holes (white) ---
    for hole in poly.interiors:
        hx, hy = hole.xy
        ax.fill(hx, hy, facecolor="white", edgecolor=edgecolor, linestyle="--")

# =====================================================================================================
# Strain Profile
# =====================================================================================================

def plot_strain_profile(results: dict, title: str = None):
    """
    Plot the strain profile across the cross-section depth.

    Failure locations (concrete crushing or reinforcement fracture) are
    colored red. Fiber strains for all reinforcement bars are annotated.

    Parameters
    ----------
    results : dict
        Results dict containing:

        - ``"section"`` — :class:`BeamSection`.
        - ``"strain_profile"`` — strain field object compatible with
          :func:`get_strain_at_point`.

    title : str or None, optional
        Plot title. Falls back to ``"Strain Profile for {section.name}"``
        when ``None``. Default is ``None``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """

    section = results['section']
    strain_profile = results["strain_profile"]

    # Section extents
    _, _, zmin, zmax = section.geometry.calculate_extents()
    depth = zmax - zmin

    # Section centroid
    cz = section.gross_properties.cz

    # Top & bottom fiber strains
    eps_top = get_strain_at_point(strain_profile, 0, zmax)
    eps_bot = get_strain_at_point(strain_profile, 0, zmin)

    # Reinforcement z-coordinates
    z_reinf = [pg.point.y for pg in section.geometry.point_geometries]

    # Reinforcement strains (from strain field, no prestress)
    eps_reinf = [
        get_strain_at_point(strain_profile, 0, z_s)
        for z_s in z_reinf
    ]

    # Check Failure
    epsilon = 0.01e-3 # account for rounding error

    concrete_fail = eps_top <= -3.5e-3 + epsilon
    reinf_failures = []
    for pg, eps_s in zip(section.geometry.point_geometries, eps_reinf):
        eps_u_neg, eps_u_pos = pg.material.constitutive_law.get_ultimate_strain()
        reinf_failures.append(eps_s >= eps_u_pos-epsilon or eps_s <= eps_u_neg+epsilon)

    # --- X-axis strain limits (‰) with padding ------------------------
    eps_vals = [0.0, eps_top * 1e3, eps_bot * 1e3]
    eps_min = min(eps_vals) - 2
    eps_max = max(eps_vals) + 2

    # --- Y-axis padding (5% of section depth) -------------------------
    z_pad = 0.05 * depth

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 8))

    # Thick vertical extent line
    ax.vlines(
        x=0.0,
        ymin=zmin,
        ymax=zmax,
        color="black",
        linewidth=2
    )

    # Concrete strain profile
    ax.plot(
        [eps_top * 1e3, eps_bot * 1e3],
        [zmax, zmin],
        color="black",
        linewidth=1.25
    )

    # Top & bottom strain lines (thick)
    ax.hlines(
        y=zmax,
        xmin=min(0.0, eps_top * 1e3),
        xmax=max(0.0, eps_top * 1e3),
        color="red" if concrete_fail else "black",
        linewidth=1.25
    )

    ax.hlines(
        y=zmin,
        xmin=min(0.0, eps_bot * 1e3),
        xmax=max(0.0, eps_bot * 1e3),
        color="black",
        linewidth=1.25
    )

    # Centroid line (dash-dot)
    ax.hlines(
        y=cz,
        xmin=eps_min,
        xmax=eps_max,
        color="black",
        linewidth=1,
        linestyle="-."
    )

    ax.annotate(
        "centroid",
        (eps_max, cz),
        textcoords="offset points",
        xytext=(-5, 4),
        ha="right",
        va="bottom",
        fontsize=8,
        color="black",
        fontfamily="serif",
    )

    # Reinforcement strains (colored by failure)
    for z_s, eps_s, failed in zip(z_reinf, eps_reinf, reinf_failures):
        color = "red" if failed else "black"

        ax.hlines(
            y=z_s,
            xmin=0.0,
            xmax=eps_s * 1e3,
            color=color,
            linewidth=1.5
        )

        ax.annotate(
            f"{eps_s * 1e3:+.2f}‰",
            (eps_s * 1e3, z_s),
            textcoords="offset points",
            xytext=(5, 0),
            va="center",
            color=color
        )

    # Top strain label (red if concrete fails)
    ax.annotate(
        f"{eps_top * 1e3:+.2f}‰",
        (eps_top * 1e3, zmax),
        textcoords="offset points",
        xytext=(-5, 0),
        ha="right",
        va="center",
        color="red" if concrete_fail else "black"
    )

    # Bottom strain label (right)
    ax.annotate(
        f"{eps_bot * 1e3:+.2f}‰",
        (eps_bot * 1e3, zmin),
        textcoords="offset points",
        xytext=(5, 0),
        va="center",
        color="black"
    )

    # Axes formatting
    # ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Strain ε [‰]")
    ax.set_ylabel("z [mm]")
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Strain Profile for {section.name}")

    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)

    # Apply padded limits
    ax.set_xlim(eps_min, eps_max)
    ax.set_ylim(zmin - z_pad, zmax + z_pad)

    return fig, ax

# =====================================================================================================
# Moment-Curvature Diagram
# =====================================================================================================

def plot_moment_curvature(
        m_c_res: MomentCurvatureResults,
        x = None,
        ax=None,
        title = "",
        show_points:bool = False,
        show_ultimate_point: bool = False
):
    """
    Plot a moment–curvature (M–K) diagram with optional point and ultimate
    point annotations.

    Parameters
    ----------
    m_c_res : MomentCurvatureResults
        Results object from the structuralcodes library.
    x : float, optional
        Normalized position along the beam x ∈ [0, 1], shown in the
        title [-].
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. A new figure is created if ``None``.
    title : str, optional
        Plot title prefix. Default is ``""``.
    show_points : bool, optional
        If ``True``, each M–K point is shown as a dot. Default is ``False``.
    show_ultimate_point : bool, optional
        If ``True``, the last point (M_u, K_u) is highlighted in red.
        Default is ``False``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """

    import matplotlib.pyplot as plt

    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # ============================================================
    #              Plot continuous M–K curve
    # ============================================================
    ax.plot(-m_c_res.chi_y * 1e6, -m_c_res.m_y / 1e6,
            color="black", linewidth=1.5, label="M–K curve")

    # ============================================================
    #              Plot ALL points as small purple dots
    # ============================================================
    if show_points:
        ax.scatter(-m_c_res.chi_y * 1e6, -m_c_res.m_y / 1e6,
                   s=6, color="black", label="M–K points")

    # ============================================================
    #                    ULTIMATE POINT (Mu)
    # ============================================================
    if show_ultimate_point:
        x_u = -m_c_res.chi_y[-1] * 1e6
        y_u = -m_c_res.m_y[-1] / 1e6

        # Dot
        ax.plot(x_u, y_u, 'ro', markersize=5)

        # Label
        label_u = (
            f"(K_u = {-m_c_res.chi_y[-1] * 1e6 :.3e},\n"
            f" M_u = {-m_c_res.m_y[-1] / 1e6:.3f} kNm)"
        )
        ax.text(
            x_u, y_u, label_u,
            fontsize=10, color="red",
            ha="right", va="bottom"
        )

    # --- Axis labels ---
    ax.set_xlabel("K [1/1000m]")
    ax.set_ylabel("My [kNm]")

    # --- Grid ---
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # ---- Add title here ----
    ax.set_title(f"{title} \n M-K-Diagram at x = {x} * L")

    return fig, ax

def plot_moment_curvature_with_reference(
    m_c_res: MomentCurvatureResults,
    ref_curvatures,
    ref_moments,
    x=None,
    ax=None,
    title="",
    ref_label="",
):
    """
    Plot an M–K diagram from ``m_c_res`` and superimpose a reference
    dataset given as curvature and moment arrays.

    Parameters
    ----------
    m_c_res : MomentCurvatureResults
        Primary results object with ``chi_y`` and ``m_y`` arrays.
    ref_curvatures : array-like
        Reference curvature values [1/1000m].
    ref_moments : array-like
        Reference moment values [kNm].
    x : float, optional
        Normalized position shown in the title [-].
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. A new figure is created if ``None``.
    title : str, optional
        Plot title prefix. Default is ``""``.
    ref_label : str, optional
        Legend label for the reference dataset. Default is ``""``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Raises
    ------
    ValueError
        If ``ref_curvatures`` and ``ref_moments`` have different lengths.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Convert reference data to arrays
    ref_curvatures = np.asarray(ref_curvatures, dtype=float)
    ref_moments = np.asarray(ref_moments, dtype=float)

    if len(ref_curvatures) != len(ref_moments):
        raise ValueError("curvatures and moments must have the same length.")

    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # ============================================================
    #                  Plot main M–K curve
    # ============================================================
    ax.plot(
        -m_c_res.chi_y * 1e6,
        -m_c_res.m_y / 1e6,
        color="black",
        linewidth=1.0,
        label="M–K curve (Python)",
    )

    ax.scatter(
        -m_c_res.chi_y * 1e6,
        -m_c_res.m_y / 1e6,
        s=2,
        color="black",
    )

    # ============================================================
    #                  Plot reference M–K dataset
    # ============================================================

    if len(ref_curvatures) == 1:
        ref_marker_size = 6.0
    else:
        ref_marker_size = 0.0

    ax.plot(
        ref_curvatures * 1e3,
        ref_moments,
        linewidth=1.0,
        marker="x",
        markersize=ref_marker_size,
        label=ref_label,
    )

    # ============================================================
    #                    ULTIMATE POINT (Mu)
    # ============================================================
    x_u = -m_c_res.chi_y[-1] * 1e6
    y_u = -m_c_res.m_y[-1] / 1e6

    ax.plot(x_u, y_u, "ro", markersize=5)

    label_u = (
        f"(K_u = {-m_c_res.chi_y[-1] * 1e6:.3e},\n"
        f" M_u = {-m_c_res.m_y[-1] / 1e6:.3f} kNm)"
    )
    ax.text(
        x_u,
        y_u,
        label_u,
        fontsize=10,
        color="red",
        ha="right",
        va="bottom",
    )

    # --- Axis labels ---
    ax.set_xlabel("K [1/1000m]")
    ax.set_ylabel("My [kNm]")

    # --- Grid ---
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # --- Legend ---
    ax.legend()

    # --- Title ---
    ax.set_title(f"{title}\nM-K-Diagram at x = {x} * L")

    return fig, ax

@dataclass
class PlotLine:
    """
    Container for a single dataset to be plotted.

    Used as the input type for the multi-line M–K plot functions and
    :func:`mirror_plot`.

    Attributes
    ----------
    moments : np.ndarray
        x-axis values (moment or any quantity mapped to x) [kNm].
    curvatures : np.ndarray
        y-axis values (curvature or any quantity mapped to y) [1/1000m].
    name : str
        Legend label.
    color : str
        Matplotlib color string (e.g. ``"black"``, ``"#FF0000"``).
        Default is ``"black"``.
    linestyle : str
        Matplotlib linestyle (e.g. ``"solid"``, ``"dashed"``).
        Default is ``"solid"``.
    linewidth : float
        Line width [-]. Default is ``1.0``.
    marker : str
        Matplotlib marker (e.g. ``"x"``, ``"o"``, ``""``).
        Default is ``""``.
    markersize : float
        Marker size [-]. Default is ``4.0``.
    """

    moments: list
    curvatures: list
    name: str
    color: str = "black"
    linestyle: str = "solid"
    linewidth: float = 1.0
    marker: str = ""
    markersize: float = 4.0

    def __post_init__(self):
        self.moments = np.asarray(self.moments, dtype=float)
        self.curvatures = np.asarray(self.curvatures, dtype=float)
        if len(self.moments) != len(self.curvatures):
            raise ValueError(
                f"PlotLine '{self.name}': "
                "'moments' and 'curvatures' must have the same length "
                f"({len(self.moments)} vs {len(self.curvatures)})."
            )

    @classmethod
    def from_results(
            cls,
            m_c_res: MomentCurvatureResults,
            name: str,
            color: str = "black",
            linestyle: str = "solid",
            linewidth: float = 1.0,
            marker: str = "",
            markersize: float = 4.0,
    ) -> "PlotLine":
        """
        Construct a :class:`PlotLine` from a :class:`MomentCurvatureResults`
        object, applying standard unit conversions.

        Curvatures: ``−chi_y · 1e6`` [1/1000m].
        Moments: ``−m_y / 1e6`` [kNm].

        Parameters
        ----------
        m_c_res : MomentCurvatureResults
            Raw results object with ``chi_y`` [1/mm] and ``m_y`` [Nmm]
            arrays.
        name : str
            Legend label for this line.
        color, linestyle, linewidth, marker, markersize
            Forwarded directly to :class:`PlotLine`.

        Returns
        -------
        PlotLine
        """
        return cls(
            moments=-m_c_res.m_y / 1e6,
            curvatures=-m_c_res.chi_y * 1e6,
            name=name,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
        )


def plot_moment_curvature_multiple(
    lines: list[PlotLine],
    ax: Optional[matplotlib.axes.Axes] = None,
    title: str = "",
    x: Optional[float] = None,
    xlabel: str = "Krümmung κ [mm/m]",
    ylabel: str = "My [kNm]",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    xmarker: float = 5.0,
    ymarker: float = 5.0,
    figsize: Optional[tuple[float, float]] = None,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """
    Plot multiple M–K datasets on a single axes.

    Parameters
    ----------
    lines : list[PlotLine]
        One or more datasets, drawn in list order.
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new figure is created when ``None``.
    title : str, optional
        Plot title prefix. Default is ``""``.
    x : float, optional
        Normalized position appended to the title [-].
    xlabel : str, optional
        Horizontal axis label. Default is ``"Krümmung κ [mm/m]"``.
    ylabel : str, optional
        Vertical axis label. Default is ``"My [kNm]"``.
    xlim : tuple[float, float], optional
        x-axis limits.
    ylim : tuple[float, float], optional
        y-axis limits.
    xmarker : float, optional
        Major tick spacing on the x-axis [-]. Default is ``5.0``.
    ymarker : float, optional
        Major tick spacing on the y-axis [-]. Default is ``5.0``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Raises
    ------
    ValueError
        If ``lines`` is empty.
    """

    if not lines:
        raise ValueError("'lines' must contain at least one PlotLine.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- Zero axes ---
    ax.axhline(0, color="#b0b0b0", linewidth=0.7, linestyle="solid", zorder=1)
    ax.axvline(0, color="#b0b0b0", linewidth=0.7, linestyle="solid", zorder=1)

    # --- Data lines ---
    for line in lines:
        ax.plot(
            line.curvatures,
            line.moments,
            color=line.color,
            linestyle=line.linestyle,
            linewidth=line.linewidth,
            marker=line.marker,
            markersize=line.markersize,
            label=line.name,
            zorder=2,
        )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.xaxis.set_major_locator(plt.MultipleLocator(xmarker))
    ax.yaxis.set_major_locator(plt.MultipleLocator(ymarker))
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
    ax.legend(loc="lower right")

    full_title = title
    if x is not None:
        full_title = f"{title}\nM-K-Diagram at x = {x} * L"
    ax.set_title(full_title)

    return fig, ax

def plot_moment_curvature_multiple_and_differences(
    lines: list[PlotLine],
    title: str = "",
    x: Optional[float] = None,
    xlabel: str = "Krümmung κ [1/1000m]",
    ylabel: str = "My [kNm]",
    ylabel_diff: str = r"$\mathrm{\Delta M_y}$ [kNm]",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    ylim_diff: Optional[tuple[float, float]] = None,
    main_height: float = 3.0,
    diff_height: float = 0.6,
    figsize: tuple[float, float] = (6.4, 8.0),
) -> tuple[plt.Figure, matplotlib.axes.Axes, list[matplotlib.axes.Axes]]:
    """
    Plot multiple M–K datasets with one difference panel per comparison line.

    Layout: row 0 is the full M–K plot; rows 1…n each show
    ΔM_y = M_{y,i} − M_{y,ref} for one comparison line. All panels share
    the x-axis. ``lines[-1]`` is the reference. Differences are computed
    by linearly interpolating each comparison line onto the reference
    curvature grid; points outside a line's range are masked as NaN.

    Parameters
    ----------
    lines : list[PlotLine]
        At least two datasets. ``lines[-1]`` is the reference.
    title : str, optional
        Plot title prefix. Default is ``""``.
    x : float, optional
        Normalized position appended to the title [-].
    xlabel : str, optional
        Label for the shared horizontal axis (bottom panel only).
        Default is ``"Krümmung κ [1/1000m]"``.
    ylabel : str, optional
        Vertical axis label for the primary panel. Default is
        ``"My [kNm]"``.
    ylabel_diff : str, optional
        Vertical axis label for every difference panel. Default is
        ``ΔM_y [kNm]`` (LaTeX).
    xlim : tuple[float, float], optional
        x-axis limits shared across all panels.
    ylim : tuple[float, float], optional
        y-axis limits for the primary panel.
    ylim_diff : tuple[float, float], optional
        y-axis limits for every difference panel.
    main_height : float, optional
        Relative height of the primary panel [-]. Default is ``3.0``.
    diff_height : float, optional
        Relative height of each difference panel [-]. Default is ``0.6``.
    figsize : tuple[float, float], optional
        Figure size (width, height) [inches]. Default is ``(6.4, 8.0)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_main : matplotlib.axes.Axes
        The primary M–K axes (row 0).
    ax_diffs : list[matplotlib.axes.Axes]
        One axes per comparison line, in the same order as
        ``lines[:-1]``.

    Raises
    ------
    ValueError
        If ``lines`` contains fewer than two entries.
    """
    if len(lines) < 2:
        raise ValueError(
            "plot_moment_curvature_multiple_and_differences requires at least "
            "two PlotLine objects (one reference + one comparison)."
        )

    def _mathtext_name(name: str) -> str:
        """Escape underscores so they are not interpreted as subscript operators."""
        return name.replace("_", r"\_")

    n_diffs = len(lines) - 1
    height_ratios = [main_height] + [diff_height] * n_diffs

    fig, axes = plt.subplots(
        1 + n_diffs, 1,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
        figsize=figsize,
    )
    fig.subplots_adjust(hspace=0.08, right=0.88)

    ax_main = axes[0]
    ax_diffs = list(axes[1:])

    ref = lines[-1]

    # ----------------------------------------------------------- primary panel
    ax_main.axhline(0, color="#b0b0b0", linewidth=0.7, zorder=1)
    ax_main.axvline(0, color="#b0b0b0", linewidth=0.7, zorder=1)

    for line in lines:
        ax_main.plot(
            line.curvatures,
            line.moments,
            color=line.color,
            linestyle=line.linestyle,
            linewidth=line.linewidth,
            marker=line.marker,
            markersize=line.markersize,
            label=line.name,
            zorder=2,
        )

    if ylim is not None:
        ax_main.set_ylim(ylim)

    ax_main.set_ylabel(ylabel)
    ax_main.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax_main.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
    ax_main.legend(loc="lower right")
    ax_main.tick_params(labelbottom=False)

    full_title = title
    if x is not None:
        full_title = f"{title}\nM-K-Diagram at x = {x} * L"
    ax_main.set_title(full_title)

    # ----------------------------------------- one difference panel per line
    ref_kappa = ref.curvatures

    for ax_d, line in zip(ax_diffs, lines[:-1]):
        ax_d.axhline(0, color="#b0b0b0", linewidth=0.7, zorder=1)
        ax_d.axvline(0, color="#b0b0b0", linewidth=0.7, zorder=1)

        m_interp = np.interp(ref_kappa, line.curvatures, line.moments)

        lo, hi = line.curvatures.min(), line.curvatures.max()
        mask = (ref_kappa >= lo) & (ref_kappa <= hi)
        delta = np.where(mask, m_interp - ref.moments, np.nan)

        ax_d.plot(
            ref_kappa,
            delta,
            color=line.color,
            linestyle=line.linestyle,
            linewidth=line.linewidth,
            marker=line.marker,
            markersize=line.markersize,
            zorder=2,
        )

        if ylim_diff is not None:
            ax_d.set_ylim(ylim_diff)

        ax_d.set_ylabel(ylabel_diff, fontsize="small")
        ax_d.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)

        if ax_d is ax_diffs[-1]:
            ax_d.set_xlabel(xlabel)
        else:
            ax_d.tick_params(labelbottom=False)

    # --- annotation to the right, vertically centred across all diff panels ---
    # Force layout so bounding boxes are accurate before reading positions.
    fig.canvas.draw()
    top    = ax_diffs[0].get_position().y1
    bottom = ax_diffs[-1].get_position().y0
    mid_y  = (top + bottom) / 2

    ref_mt = _mathtext_name(ref.name)
    fig.text(
        0.90, mid_y,
        rf"$\mathrm{{\Delta M_{{y,i}} = M_{{y,i}} - M_{{y,\,{ref_mt}}}}}$",
        fontsize="small",
        color="#444444",
        ha="left", va="center",
        rotation=90,
    )

    if xlim is not None:
        ax_main.set_xlim(xlim)

    return fig, ax_main, ax_diffs


def table_moment_curvature(m_c_res: MomentCurvatureResults):
    """
    Return a tabulated string of moment–curvature results.

    Returns
    -------
    str
        Table formatted with ``fancy_grid`` style via tabulate. Columns
        are ``i``, ``chi_y``, ``chi_z``, ``eps_a``, ``m_y``,
        ``m_z``; values in scientific notation with 3 decimal places.
    """
    # Convert to numpy arrays for safety
    chi_y = np.asarray(m_c_res.chi_y)
    chi_z = np.asarray(m_c_res.chi_z) if m_c_res.chi_z is not None else None
    eps_a = np.asarray(m_c_res.eps_a)
    m_y = np.asarray(m_c_res.m_y)
    m_z = np.asarray(m_c_res.m_z) if m_c_res.m_z is not None else None

    # Build table rows
    rows = []
    for i in range(len(chi_y)):
        rows.append([
            i,
            chi_y[i],
            chi_z[i] if chi_z is not None else None,
            eps_a[i],
            m_y[i],
            m_z[i] if m_z is not None else None,
        ])

    headers = ["i", "chi_y", "chi_z", "eps_a", "m_y", "m_z"]

    return tabulate(rows, headers=headers, floatfmt=".3e", tablefmt="fancy_grid")

# =====================================================================================================
# Concrete Constitutive Law
# =====================================================================================================

def plot_constitutive_law_concrete(concrete: Concrete, n: int = 100, debug: bool = False, show_markers: bool = False):
    """
    Plot the stress–strain constitutive law for a concrete material.

    The compression branch is sampled from ``eps_cu`` to 0; the tension
    branch adapts to the attached law type
    (:class:`TensionStiffeningConcreteLawEC`, :class:`CrackingConcreteLawEC`,
    or :class:`Elastic`). Both axes are sign-negated so compression is
    plotted in the positive quadrant.

    Parameters
    ----------
    concrete : Concrete
        Concrete material instance with an attached constitutive law.
    n : int, optional
        Number of sample points per branch [-]. Default is ``100``.
    debug : bool, optional
        If ``True``, prints the strain and stress arrays to console.
        Default is ``False``.
    show_markers : bool, optional
        If ``True``, adds circle markers at each sample point.
        Default is ``False``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Raises
    ------
    ValueError
        If no constitutive law is attached to ``concrete``.
    """

    if concrete.constitutive_law is None:
        raise ValueError("No constitutive law is attached to this Concrete instance.")

    fctm = concrete.fctm
    Ecm = concrete.Ecm
    eps_ctm = fctm / Ecm

    law = concrete.constitutive_law

    # Build strain range based on law parameters if present
    # compression range
    eps_min, _ = law.get_ultimate_strain()
    eps_c = np.linspace(eps_min, 0, n, endpoint = False)

    # 0,0
    eps_0 = [0.00]

    # tension range
    if isinstance(law, TensionStiffeningConcreteLawEC):
        eps_P_t = eps_ctm + 0.001e-3
        eps_S_t = law.eps_S_t
        eps_F_t = law.eps_F_t

        eps_t = [eps_ctm, eps_P_t, eps_S_t, eps_F_t]
    elif isinstance(law, CrackingConcreteLawEC):
        eps_t = [eps_ctm]
    elif isinstance(law, Elastic):
        eps_t = np.flip(np.linspace(-eps_min, 0, n, endpoint = False))
    else:
        eps_t = [0.0]

    eps = np.concatenate((eps_c, eps_0, eps_t))
    sig = law.get_stress(eps)

    if debug:
        print(f"eps: {eps*1000}")
        print(f"sig: {sig}")

    # === FLIP OVER X AND Y AXIS ===
    eps_plot = -eps
    sig_plot = -sig

    fig, ax = plt.subplots()

    plot_kwargs = dict(linewidth=1.8, color="black", label=law.name)
    if show_markers:
        plot_kwargs.update(marker="o", markersize=4, markerfacecolor="white", markeredgewidth=1.0)

    ax.plot(eps_plot, sig_plot, **plot_kwargs)

    ax.axhline(0, color="#b0b0b0", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#b0b0b0", linewidth=0.8, zorder=1)

    # Reverse the sign of tick labels on both axes
    negate = FuncFormatter(lambda val, _: f"{-val:g}")
    ax.xaxis.set_major_formatter(negate)
    ax.yaxis.set_major_formatter(negate)

    ax.set_xlabel("Strain [-]")
    ax.set_ylabel("Stress [MPa]")
    ax.set_title(f"Constitutive Law of {concrete.name}")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="lower right")

    return fig, ax

# =====================================================================================================
# Reinforcement Constitutive Law
# =====================================================================================================

def plot_constitutive_law_reinforcement(reinforcement: Reinforcement, n: int = 100):
    """
    Plot the stress–strain constitutive law for a reinforcement material.

    The strain domain covers [−ε_u, ε_u] sampled in three branches:
    post-negative, yield plateau, and post-positive.

    Parameters
    ----------
    reinforcement : Reinforcement
        Reinforcement material instance with an attached constitutive law.
    n : int, optional
        Number of sample points per branch [-]. Default is ``100``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Raises
    ------
    ValueError
        If no constitutive law is attached to ``reinforcement``.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Check constitutive law
    if reinforcement.constitutive_law is None:
        raise ValueError(
            "No constitutive law is attached to this Reinforcement instance."
        )

    law = reinforcement.constitutive_law

    # ------------------------------------------------------------
    #   Build strain domain for steel
    # ------------------------------------------------------------
    eps_y = reinforcement.epsyk  # yield strain
    eps_u = reinforcement.epsud()  # ultimate strain

    # Positive strain range typical for reinforcement
    eps_post_neg = np.linspace(-eps_u, -eps_y, n, endpoint=False)
    eps_yiel_yiel = np.linspace(-eps_y, eps_y, n, endpoint=False)
    eps_post_pos = np.linspace(eps_y, eps_u, n)

    eps = np.concatenate((
        eps_post_neg,
        eps_yiel_yiel,
        eps_post_pos
    ))

    # Compute stresses (in MPa)
    sig = law.get_stress(eps)

    # ------------------------------------------------------------
    #   Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(
        eps, sig,
        color="black", linewidth=1.8,
        label=f"{law.name}"
    )

    ax.axhline(0, color="#b0b0b0", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#b0b0b0", linewidth=0.8, zorder=1)

    # ---------------------------------------------
    #   Axes & grid
    # ---------------------------------------------
    ax.set_xlabel("Strain [-]")
    ax.set_ylabel("Stress [MPa]")
    ax.set_title("Reinforcement Constitutive Law")
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
    ax.legend(loc="lower right")

    return fig, ax

# =====================================================================================================
# Mirrored Plot used for moment, curvature, stiffness and deflection plots
# =====================================================================================================

def mirror_plot(
    lines: list[PlotLine],
    ax: Optional[matplotlib.axes.Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    xmarker: float = 5.0,
    ymarker: float = 5.0,
    figsize: tuple[float, float] = (12, 5),
    flip_y_axis: bool = False,
    coordinate_axes: bool = True,
    show_x_numbers: bool = True,
    show_x_ticks: bool = True,
    x_number_position: str = "bottom",
    show_x_axis_label: bool = True,
    show_legend: bool = True,
    x_number_pad: float = 8.0,
    axis_arrows: bool = True,
    show_vertical_grid: bool = True,
    show_horizontal_grid: bool = True,
    limit_grid_to_xlim: bool = True,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """
    Plot each :class:`PlotLine` together with its x-axis mirror image.

    Draws each dataset and its mirror (reflected about the last x-value)
    on a single axes. Primarily used for moment, curvature, stiffness, and
    deflection diagrams of symmetric beams where only the half-span is
    stored but the full-span plot is desired.

    Parameters
    ----------
    lines : list[PlotLine]
        One or more datasets to plot.
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new figure is created when ``None``.
    title : str, optional
        Title text, placed left-aligned below the plot area. Default is
        ``""``.
    xlabel : str, optional
        Horizontal axis label. Default is ``""``.
    ylabel : str, optional
        Vertical axis label. Default is ``""``.
    xlim : tuple[float, float], optional
        x-axis limits.
    ylim : tuple[float, float], optional
        y-axis limits.
    xmarker : float, optional
        Major tick spacing on the x-axis [-]. Default is ``5.0``.
    ymarker : float, optional
        Major tick spacing on the y-axis [-]. Default is ``5.0``.
    figsize : tuple[float, float], optional
        Figure size (width, height) [inches]. Default is ``(12, 5)``.
    flip_y_axis : bool, optional
        If ``True``, invert the y-axis (useful for deflection diagrams).
        Default is ``False``.
    coordinate_axes : bool, optional
        If ``True``, use coordinate-axis spine style (top/right spines
        hidden). Default is ``True``.
    show_x_numbers : bool, optional
        Show or hide x-axis tick labels. Default is ``True``.
    show_x_ticks : bool, optional
        Show or hide x-axis tick marks (independent of labels).
        Default is ``True``.
    x_number_position : str, optional
        Position of x-axis tick labels: ``"bottom"`` or ``"top"``.
        Default is ``"bottom"``.
    show_x_axis_label : bool, optional
        Show the ``"x_norm"`` text label at the arrow end of the x-axis.
        Default is ``True``.
    show_legend : bool, optional
        Show the legend. Default is ``True``.
    x_number_pad : float, optional
        Padding between x-axis tick marks and labels [points].
        Default is ``8.0``.
    axis_arrows : bool, optional
        Draw arrow heads on the coordinate axes. Only effective when
        ``coordinate_axes=True``. Default is ``True``.
    show_vertical_grid : bool, optional
        Draw vertical grid lines. Default is ``True``.
    show_horizontal_grid : bool, optional
        Draw horizontal grid lines. Default is ``True``.
    limit_grid_to_xlim : bool, optional
        If ``True`` and ``xlim`` is set, clip grid lines to ``xlim``.
        Default is ``True``.
    x_scale : float, optional
        Multiplicative scale applied to all x-values (``line.moments``)
        before plotting [-]. Default is ``1.0``.
    y_scale : float, optional
        Multiplicative scale applied to all y-values (``line.curvatures``)
        before plotting [-]. Default is ``1.0``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Raises
    ------
    ValueError
        If ``lines`` is empty, or if any line has mismatched or empty
        data arrays.
    """

    if not lines:
        raise ValueError("'lines' must contain at least one PlotLine.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- Data lines and mirrored lines ---
    for line in lines:
        x = [value * x_scale for value in line.moments]
        y = [value * y_scale for value in line.curvatures]

        if len(x) != len(y):
            raise ValueError(
                f"Length mismatch for line {line.name!r}: "
                f"{len(x)} x-values but {len(y)} y-values."
            )

        if not x:
            raise ValueError(f"Line {line.name!r} contains no data.")

        mirror_axis = x[-1]

        x_mirrored = [
            2 * mirror_axis - x_value
            for x_value in reversed(x)
        ]
        y_mirrored = list(reversed(y))

        ax.plot(
            x,
            y,
            color=line.color,
            linestyle=line.linestyle,
            linewidth=line.linewidth,
            marker=line.marker,
            markersize=line.markersize,
            label=line.name,
            zorder=2,
        )

        ax.plot(
            x_mirrored,
            y_mirrored,
            color=line.color,
            linestyle=line.linestyle,
            linewidth=line.linewidth,
            marker=line.marker,
            markersize=line.markersize,
            label=None,
            zorder=2,
        )

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if flip_y_axis:
        ax.invert_yaxis()

    # --- Coordinate-axis style ---
    if coordinate_axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if axis_arrows:
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
        else:
            ax.spines["bottom"].set_position(("data", 0))
            ax.spines["left"].set_position(("data", 0))

            ax.spines["bottom"].set_color("#606060")
            ax.spines["left"].set_color("#606060")

            ax.spines["bottom"].set_linewidth(1.0)
            ax.spines["left"].set_linewidth(1.0)

        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.tick_params(axis="both", direction="out")

    else:
        ax.axhline(0, color="#b0b0b0", linewidth=0.7, linestyle="solid", zorder=1)
        ax.axvline(0, color="#b0b0b0", linewidth=0.7, linestyle="solid", zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.xaxis.set_major_locator(plt.MultipleLocator(xmarker * x_scale))
    ax.yaxis.set_major_locator(plt.MultipleLocator(ymarker * y_scale))

    # --- X-axis tick marks and labels (controlled independently) ---
    # Resolve which sides carry ticks / labels based on x_number_position
    tick_bottom = show_x_ticks and (x_number_position == "bottom")
    tick_top    = show_x_ticks and (x_number_position == "top")
    label_bottom = show_x_numbers and (x_number_position == "bottom")
    label_top    = show_x_numbers and (x_number_position == "top")

    ax.tick_params(
        axis="x",
        which="both",
        bottom=tick_bottom,
        top=tick_top,
        labelbottom=label_bottom,
        labeltop=label_top,
        direction="out",
        pad=x_number_pad,
    )

    # --- Grid ---
    if show_vertical_grid or show_horizontal_grid:
        if limit_grid_to_xlim and xlim is not None:
            y_min, y_max = ax.get_ylim()

            if show_vertical_grid:
                for x_tick in ax.get_xticks():
                    if xlim[0] <= x_tick <= xlim[1]:
                        ax.plot(
                            [x_tick, x_tick],
                            [y_min, y_max],
                            color="#b0b0b0",
                            linestyle="-",
                            linewidth=0.5,
                            alpha=0.7,
                            zorder=0,
                        )

            if show_horizontal_grid:
                for y_tick in ax.get_yticks():
                    ax.plot(
                        [xlim[0], xlim[1]],
                        [y_tick, y_tick],
                        color="#b0b0b0",
                        linestyle="-",
                        linewidth=0.5,
                        alpha=0.7,
                        zorder=0,
                    )
        else:
            ax.grid(
                True,
                axis="both",
                linestyle="-",
                linewidth=0.5,
                alpha=0.7,
            )

            if not show_vertical_grid:
                ax.xaxis.grid(False)

            if not show_horizontal_grid:
                ax.yaxis.grid(False)

    # --- Axis arrows ---
    if axis_arrows and coordinate_axes:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        ax.annotate(
            "",
            xy=(x_max, 0),
            xytext=(x_min, 0),
            arrowprops=dict(
                arrowstyle="->",
                color="#000000",
                linewidth=1.0,
                shrinkA=0,
                shrinkB=0,
            ),
            annotation_clip=False,
            zorder=3,
        )

        if flip_y_axis:
            y_arrow_end = y_min
            y_arrow_start = y_max
        else:
            y_arrow_end = y_max
            y_arrow_start = y_min

        ax.annotate(
            "",
            xy=(0, y_arrow_end),
            xytext=(0, y_arrow_start),
            arrowprops=dict(
                arrowstyle="->",
                color="#000000",
                linewidth=1.0,
                shrinkA=0,
                shrinkB=0,
            ),
            annotation_clip=False,
            zorder=3,
        )

    if show_legend:
        ax.legend(loc="lower right")

    # --- Title left-aligned below plot ---
    ax.text(
        0.01,
        0.00,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=plt.rcParams["axes.titlesize"],
        fontweight=plt.rcParams["axes.titleweight"],
    )

    # --- Axis label ---
    if show_x_axis_label:
        from matplotlib.transforms import blended_transform_factory
        blend = blended_transform_factory(ax.transAxes, ax.transData)

        x_label_va = "top" if x_number_position == "bottom" else "bottom"

        ax.text(
            1.01,
            0.04,  # data y=0 → always on the x-axis line
            "x_norm",
            transform=blend,
            ha="left",
            va=x_label_va,
            clip_on=False,
        )

    return fig, ax

# =====================================================================================================
# Triangulated Cross Section for Fiber Integrator (currently not used in HpDesignBench)
# =====================================================================================================

def plot_triangulated_mesh(triangulated_data: t.List[t.Tuple[np.ndarray, np.ndarray, np.ndarray, ConstitutiveLaw]], show_centroids=True):
    """
    Visualize triangulated fibers returned by ``FiberIntegrator.triangulate()``.

    Parameters
    ----------
    triangulated_data : list[tuple[np.ndarray, np.ndarray, np.ndarray, ConstitutiveLaw]]
        List of ``(x, y, area, constitutive_law)`` tuples, one per
        triangulated surface.
    show_centroids : bool, optional
        If ``True``, draw a dot at each fiber centroid. Default is
        ``True``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """

    fig, ax = plt.subplots()

    # Map each material to a color index
    materials = {}
    cmap = plt.cm.get_cmap('tab10')
    color_index = 0

    for x, y, area, material in triangulated_data:

        # each "set" (x,y,area) contains multiple fibers but they all
        # come from one triangulated surface with one material
        if material not in materials:
            materials[material] = cmap(color_index)
            color_index += 1

        col = materials[material]

        # draw small scatter markers for centroids
        if show_centroids:
            ax.scatter(x, y, s=10, color=col)

    # legend by material name
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                   markersize=8, label=str(m))
        for m, c in materials.items()
    ]
    ax.legend(handles=legend_elements, title="Materials")

    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Triangulated Fibers (centroids)")

    return fig, ax

def plot_mesh_with_triangles(triangulated_data):
    """
    Visualize the triangle mesh from ``FiberIntegrator.triangulate()``.

    Draws each triangle as an outlined polygon and optionally marks the
    fiber centroids.

    Parameters
    ----------
    triangulated_data : list[tuple]
        List of ``(x, y, area, material, mesh)`` tuples where ``mesh``
        is a dict with ``"vertices"`` and ``"triangles"`` arrays.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    fig, ax = plt.subplots()

    for (x, y, area, material, mesh) in triangulated_data:
        verts = mesh['vertices']
        tris = mesh['triangles']

        # create polygon array for PolyCollection
        polys = [verts[tri] for tri in tris]

        pc = PolyCollection(polys,
                            facecolors='none',
                            edgecolors='k',
                            linewidths=0.6)
        ax.add_collection(pc)

        # optional – show centroids
        ax.scatter(x, y, s=5, color='red')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Triangulated mesh")

    return fig, ax