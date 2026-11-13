"""SLS concrete stress–strain diagram for C25/30 through C100/115.

Two SVGs in ``outputs/mk_height_matrix/``: the full law, and the tension
branch alone. Each curve is the production
serviceability law: Sargin compression and Nayal–Rasheed tension
stiffening. The tension-stiffening factors are the same for every grade;
``fctm`` and ``ε_ctm = fctm / Ecm`` are not, so each grade has its own
tension branch.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from structuralcodes import set_design_code
from structuralcodes.materials.concrete import create_concrete

from core.analysis_core.material_methods import tenstiff_parabolic_law_EC
from core.ioh_core.import_specs import load_materials_registry

set_design_code("ec2_2004")

HP_DIR = Path("slab_construction/slabs/hp_slab")
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "mk_height_matrix"
FCK_MIN = 25.0
FCK_MAX = 100.0


def _grades(materials: dict) -> list[tuple[str, float]]:
    grades = [
        (name, float(spec["f_ck"]))
        for name, spec in materials.items()
        if spec["type"] == "concrete" and FCK_MIN <= float(spec["f_ck"]) <= FCK_MAX
    ]
    grades.sort(key=lambda item: item[1])
    return grades


def _curve(fck: float) -> tuple[np.ndarray, np.ndarray]:
    concrete = create_concrete(fck=fck, constitutive_law="sargin")
    law = tenstiff_parabolic_law_EC(concrete)
    strain = np.asarray(law._x, dtype=float)
    stress = np.asarray(law._y, dtype=float)
    order = np.argsort(strain)
    return strain[order], stress[order]


def main() -> None:
    materials = load_materials_registry(HP_DIR / "materials.csv")
    grades = _grades(materials)
    curves = [(name, *_curve(fck)) for name, fck in grades]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    greys = np.linspace(0.72, 0.0, len(curves))
    for (name, strain, stress), grey in zip(curves, greys):
        ax.plot(strain, stress, color=str(grey), lw=1.35, label=name, zorder=2)

    ax.axhline(0.0, color="0.75", lw=0.6)
    ax.axvline(0.0, color="0.75", lw=0.6)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(r"$\varepsilon$ [–]")
    ax.set_ylabel(r"$\sigma$ [N/mm$^2$]")
    ax.set_title("SLS concrete, C25/30 to C100/115", fontsize=11)
    ax.grid(True, color="0.88", lw=0.6)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "concrete_sls_stress_strain.svg"
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path} ({len(curves)} grades)", flush=True)

    fig_t, ax_t = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for (name, strain, stress), grey in zip(curves, greys):
        tension = strain >= 0.0
        ax_t.plot(
            strain[tension],
            stress[tension],
            color=str(grey),
            lw=1.35,
            label=name,
            zorder=2,
        )
    ax_t.set_xlim(0.0, 0.00135)
    ax_t.set_ylim(0.0, 6.0)
    ax_t.set_xlabel(r"$\varepsilon$ [–]")
    ax_t.set_ylabel(r"$\sigma$ [N/mm$^2$]")
    ax_t.set_title("SLS concrete tension, C25/30 to C100/115", fontsize=11)
    ax_t.grid(True, color="0.88", lw=0.6)
    ax_t.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=False,
    )
    tension_path = OUT_DIR / "concrete_sls_tension_stress_strain.svg"
    fig_t.savefig(tension_path)
    plt.close(fig_t)
    print(f"Wrote {tension_path}", flush=True)


if __name__ == "__main__":
    main()
