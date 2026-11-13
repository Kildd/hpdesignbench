"""Simple ULS/SLS material curves and one HP section with tendons in red.

Concrete is C50/60. CFRP is solidian Roving 2R. The section is the
midspan cut used for the mesh figure (h = 300 mm, t = 20 mm, nt = 5).
Compression on the concrete plots is positive and points right.
CFRP is tension only. Tendon markers are enlarged so the positions
remain visible; the shell outline is to scale.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from structuralcodes import set_design_code
from structuralcodes.materials.concrete import create_concrete

from core.analysis_core.material_methods import (
    create_sls_concrete_EC,
    get_cfrp_reinforcement_from_registry,
    get_material_properties,
)
from core.ioh_core.import_specs import load_materials_registry
from slab_construction.slabs.hp_slab.hp_model.hp_geometry import HPGeometry

set_design_code("ec2_2004")

HP_DIR = Path("slab_construction/slabs/hp_slab")
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "uls_sls_sketches"
CONC_ID = "C50/60"
REINF_ID = "solidian Roving 2R (Z-1.6-308)"


def _stress(law, strain: np.ndarray) -> np.ndarray:
    if hasattr(law, "get_stress"):
        return np.array([float(law.get_stress(float(eps))) for eps in strain])
    return np.array([float(law(float(eps))) for eps in strain])


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0.0, color="0.75", lw=0.6)
    ax.axvline(0.0, color="0.75", lw=0.6)
    ax.grid(True, color="0.9", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=140)
    fig.savefig(OUT_DIR / f"{stem}.svg")
    plt.close(fig)
    print(OUT_DIR / f"{stem}.png")


def _plot_line(strain, stress, title: str, stem: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8), layout="constrained")
    ax.plot(strain, stress, color="black", lw=2.0)
    _style(ax, title, xlabel, ylabel)
    _save(fig, stem)


def main() -> None:
    materials = load_materials_registry(HP_DIR / "materials.csv")
    props = get_material_properties(CONC_ID, materials)
    concrete_uls = create_concrete(
        fck=props["f_ck"],
        constitutive_law="parabolarectangle",
        alpha_cc=props["alpha_cc"],
        gamma_c=props["gamma_c"],
        name=f"{CONC_ID} ULS",
    )
    concrete_sls = create_sls_concrete_EC(concrete_uls, "TENSTIFF_PARABOLIC")
    cfrp_uls = get_cfrp_reinforcement_from_registry(
        REINF_ID, materials, prestress_percent=0.0, limit_state="ULS"
    )
    cfrp_sls = get_cfrp_reinforcement_from_registry(
        REINF_ID, materials, prestress_percent=0.0, limit_state="SLS"
    )

    uls_law = concrete_uls.constitutive_law
    eps_cu = float(uls_law._eps_u)
    eps_c = np.linspace(eps_cu, 0.0, 300)
    # Druck positiv: Vorzeichen der Bibliothek umdrehen.
    _plot_line(
        -eps_c,
        -_stress(uls_law, eps_c),
        "Beton ULS, C50/60, Parabel-Rechteck",
        "concrete_uls",
        r"Druckstauchung $\varepsilon_c$ [–]",
        r"Druckspannung $\sigma_c$ [N/mm$^2$]",
    )

    sls_law = concrete_sls.constitutive_law
    eps_s = np.asarray(sls_law._x, dtype=float)
    sig_s = np.asarray(sls_law._y, dtype=float)
    order = np.argsort(-eps_s)
    fig, ax = plt.subplots(figsize=(5.2, 3.8), layout="constrained")
    ax.plot(-eps_s[order], -sig_s[order], color="black", lw=2.0)
    _style(
        ax,
        "Beton SLS, C50/60, Sargin und Tension Stiffening",
        r"$\varepsilon$ [–], Druck positiv",
        r"$\sigma$ [N/mm$^2$], Druck positiv",
    )
    ax.annotate(
        "Zugast",
        xy=(-0.0002, -3.0),
        xytext=(-0.00105, 12.0),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 0.8},
    )
    _save(fig, "concrete_sls")

    for reinf, title, stem in (
        (cfrp_uls, "CFK ULS, solidian 2R, Bemessungswerte", "cfrp_uls"),
        (cfrp_sls, "CFK SLS, solidian 2R, charakteristische Werte", "cfrp_sls"),
    ):
        law = reinf.constitutive_law.wrapped_law
        eps_u = float(law.get_ultimate_strain()[1])
        eps = np.linspace(0.0, eps_u, 50)
        _plot_line(
            eps,
            _stress(law, eps),
            title,
            stem,
            r"Dehnung $\varepsilon$ [–]",
            r"Spannung $\sigma$ [N/mm$^2$]",
        )

    # Midspan outline is about B wide and Hy tall. Hy is chosen so the
    # bounding-box height is two-thirds of the width.
    geometry = HPGeometry(
        B=800.0,
        L=4000.0,
        Hx=80.0,
        Hy=530.0,
        t=28.0,
        dy=90.0,
        nt=8,
    )
    polygon = geometry.polygon_section_at(0.0, 100)
    tendons = geometry.tendon_coords_at_x(0.0)
    ys, zs = polygon.exterior.xy

    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    ax.fill(ys, zs, facecolor="#d9d9d9", edgecolor="#222222", lw=1.0, zorder=1)
    ty = [pt[0] for pt in tendons]
    tz = [pt[1] for pt in tendons]
    ax.scatter(ty, tz, s=28, c="red", zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(2 / 3)
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    ax.margins(0.03)
    ax.set_title(
        "HP-Querschnitt in Feldmitte, H/B = 2/3, nt = 8\n"
        "rote Punkte: Spannglieder, Durchmesser vergrößert"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "hp_section_reinforcement")


if __name__ == "__main__":
    main()
