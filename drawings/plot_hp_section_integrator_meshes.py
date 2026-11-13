"""HP-shell section: fiber triangle mesh and Marin strain bands.

Production fiber sections use ``mesh_size=0.01``. Marin does not triangulate;
it splits the same polygon into strain bands of the constitutive law.
The bands drawn here use one bending strain plane that crosses the
parabola-rectangle breakpoints, so every branch that can cut this section
is visible. That plane is not an equilibrium solution.

Run from the repo root::

    python _mains/verification/plot_hp_section_integrator_meshes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import triangle
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from structuralcodes import set_design_code
from structuralcodes.materials.concrete import create_concrete
from structuralcodes.sections.section_integrators import (
    FiberIntegrator,
    MarinIntegrator,
)

from core.analysis_core.material_methods import (
    get_cfrp_reinforcement_from_registry,
    get_material_properties,
)
from core.ioh_core.import_specs import load_materials_registry
from slab_construction.slabs.hp_slab.hp_model.hp_geometry import HPGeometry
from slab_construction.slabs.hp_slab.hp_model.hp_shell import HPShell

# Same companions as the fiber–Marin verification matrix.
SPAN_MM = 6750.0
B_MM = 1350.0
HX_HGES_RATIO = 0.2
T_MM = 20.0
DY_MM = 150.0
CONC_ID = "C50/60"
X_MID = 0.5
REINF_ID = "solidian Roving 2R (Z-1.6-308)"
A_TEX_MM2 = 10.86
NT = 5
H_GES_MM = 300.0

set_design_code("ec2_2004")

HP_DIR = Path("slab_construction/slabs/hp_slab")
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "hp_section_meshes"
MESH_SIZE = 0.01


def _uls_section(materials: dict):
    """ULS midspan section with the production fiber settings."""
    conc_props = get_material_properties(CONC_ID, materials)
    concrete = create_concrete(
        fck=conc_props["f_ck"],
        constitutive_law="parabolarectangle",
        alpha_cc=conc_props["alpha_cc"],
        gamma_c=conc_props["gamma_c"],
        name=f"{CONC_ID} ULS",
    )
    reinforcement = get_cfrp_reinforcement_from_registry(
        mat_id=REINF_ID,
        materials=materials,
        prestress_percent=0.0,
        limit_state="ULS",
    )
    geometry = HPGeometry(
        B=B_MM,
        L=SPAN_MM,
        Hx=HX_HGES_RATIO * H_GES_MM,
        Hy=(1.0 - HX_HGES_RATIO) * H_GES_MM,
        t=T_MM,
        dy=DY_MM,
        nt=NT,
    )
    shell = HPShell(
        geometry,
        concrete,
        reinforcement,
        reinf_area=A_TEX_MM2,
        integrator="fiber",
        mesh_size=MESH_SIZE,
    )
    return shell.section_at(X_MID, name="midspan")


def _fiber_meshes(section) -> list[dict]:
    """Return the triangle meshes ``FiberIntegrator.triangulate`` builds."""
    integrator = FiberIntegrator()
    meshes = []
    for surface in section.geometry.geometries:
        prepared = integrator.prepare_triangulation(surface)
        max_area = surface.area * MESH_SIZE
        mesh = triangle.triangulate(prepared, f"pq{30:.1f}Aa{max_area}o1")
        meshes.append(mesh)
    return meshes


def _marin_bands(section, strain: tuple[float, float]) -> list:
    """Return polygons Marin integrates for one uniaxial strain plane."""
    integrator = MarinIntegrator()
    bands: list = []

    def _capture(polygon, coeffs, marin_input) -> None:
        bands.append((polygon, tuple(coeffs)))
        MarinIntegrator._get_input_polygon(integrator, polygon, coeffs, marin_input)

    integrator._get_input_polygon = _capture
    integrator._process_surface_geometries(
        section.geometry,
        np.asarray(strain, dtype=float),
        [],
        integrate="stress",
    )
    return bands


def _bending_strain(section) -> tuple[float, float, float, float]:
    """Strain plane from slightly tensile at z_min to eps_u at z_max."""
    law = section.geometry.geometries[0].material.constitutive_law
    _, _, zmin, zmax = section.geometry.calculate_extents()
    eps_u = float(law._eps_u)
    eps_tensile = 1.0e-4
    chi = (eps_u - eps_tensile) / (zmax - zmin)
    eps_a = eps_tensile - chi * zmin
    return eps_a, chi, zmin, zmax


def _tendon_circles(section):
    circles = []
    for point in section.geometry.point_geometries:
        y, z = point.point.coords[0]
        circles.append((y, z, point.diameter / 2.0))
    return circles


def _draw_fiber(ax, meshes, tendons, *, linewidth: float) -> int:
    n_triangles = 0
    for mesh in meshes:
        vertices = mesh["vertices"]
        triangles = mesh["triangles"]
        n_triangles += len(triangles)
        polygons = [vertices[tri] for tri in triangles]
        ax.add_collection(
            PolyCollection(
                polygons,
                facecolors="#d9d9d9",
                edgecolors="#222222",
                linewidths=linewidth,
            )
        )
    for y, z, radius in tendons:
        ax.add_patch(
            plt.Circle((y, z), radius, facecolor="#0072B2", edgecolor="#0072B2", lw=0.4, zorder=3)
        )
    return n_triangles


def _draw_marin(ax, bands, tendons) -> int:
    hatches = ("", "////", "....", "xxxx")
    faces = ("#f0e6c8", "#c8d8ee", "#e8d0d0", "#d5ead0")
    seen: dict[tuple, int] = {}
    for polygon, coeffs in bands:
        key = tuple(np.round(coeffs, 6))
        if key not in seen:
            seen[key] = len(seen)
        index = seen[key] % len(faces)
        xy = np.column_stack(polygon.exterior.xy)
        ax.add_patch(
            MplPolygon(
                xy,
                closed=True,
                facecolor=faces[index],
                edgecolor="#222222",
                linewidth=0.4,
                hatch=hatches[index],
                zorder=1,
            )
        )
    for y, z, radius in tendons:
        ax.add_patch(
            plt.Circle((y, z), radius, facecolor="#0072B2", edgecolor="#0072B2", lw=0.4, zorder=3)
        )
    return len(bands)


def _style(ax, title: str) -> None:
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    ax.tick_params(labelsize=8)


def main() -> None:
    materials = load_materials_registry(HP_DIR / "materials.csv")
    section = _uls_section(materials)
    meshes = _fiber_meshes(section)
    eps_a, chi, zmin, zmax = _bending_strain(section)
    bands = _marin_bands(section, (eps_a, chi))
    tendons = _tendon_circles(section)

    right_band = max(bands, key=lambda item: item[0].centroid.x)
    minx, miny, maxx, maxy = right_band[0].bounds
    zoom = (minx - 90.0, maxx + 25.0, miny - 20.0, maxy + 20.0)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), layout="constrained")
    n_tri = _draw_fiber(axes[0, 0], meshes, tendons, linewidth=0.25)
    _draw_fiber(axes[1, 0], meshes, tendons, linewidth=0.45)
    n_bands = _draw_marin(axes[0, 1], bands, tendons)
    _draw_marin(axes[1, 1], bands, tendons)

    for ax in axes[0, :]:
        ax.set_xlim(-B_MM / 2 - 40, B_MM / 2 + 40)
        ax.set_ylim(zmin - 30, zmax + 30)
        ax.add_patch(
            Rectangle(
                (zoom[0], zoom[2]),
                zoom[1] - zoom[0],
                zoom[3] - zoom[2],
                fill=False,
                edgecolor="#D55E00",
                linewidth=1.0,
                zorder=4,
            )
        )
    for ax in axes[1, :]:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])

    _style(axes[0, 0], f"Fasernetz, ganzer Querschnitt ({n_tri} Dreiecke)")
    _style(axes[0, 1], f"Marin-Dehnungsstreifen, ganzer Querschnitt ({n_bands} Polygone)")
    _style(axes[1, 0], "Fasernetz, rechter Ausschnitt")
    _style(axes[1, 1], "Marin-Dehnungsstreifen, selber Ausschnitt")

    fig.suptitle(
        "HP-Schale in Feldmitte, h = 300 mm, t = 20 mm, "
        f"B = {B_MM:.0f} mm, nt = 5",
        fontsize=12,
    )
    fig.supxlabel(
        "Links: 198 Dreiecke, mesh_size = 0.01, Mindestwinkel 30°. "
        "Rechts: drei Dehnungsstreifen, kein Dreiecksnetz. "
        "Blaue Punkte: Spannglieder. Oranger Rahmen: Ausschnitt.",
        fontsize=8,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "hp_midspan_fiber_and_marin.png"
    svg = OUT_DIR / "hp_midspan_fiber_and_marin.svg"
    fig.savefig(png, dpi=160)
    fig.savefig(svg)
    print(f"triangles={n_tri} marin_polygons={n_bands}")
    print(f"strain eps_a={eps_a:.6e} chi={chi:.6e}")
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
