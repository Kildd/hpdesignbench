"""Characteristic CFRP stress–strain diagrams.

One SVG per roving, plus one SVG with both rovings, in
``outputs/mk_height_matrix/``. The curve is the production SLS law:
linear elastic, ``σ = E ε``, truncated at ``ε_uk``. ``σ_uk`` is the stress
that law returns at ``ε_uk`` (``E · ε_uk``), not the stored ``f_tk``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.ioh_core.import_specs import load_materials_registry

HP_DIR = Path("slab_construction/slabs/hp_slab")
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "mk_height_matrix"

ROVINGS = (
    (
        "solidian Roving 1R (Z-1.6-308)",
        "cfrp_1r_stress_strain.svg",
    ),
    (
        "solidian Roving 2R (Z-1.6-308)",
        "cfrp_2r_stress_strain.svg",
    ),
)

# Shared so the two products can be compared. Includes the origin and room
# for the written-out ultimate point above the steeper 1R line.
XLIM = (0.0, 0.018)
YLIM = (0.0, 4200.0)


def _law(materials: dict, mat_id: str) -> tuple[float, float, float]:
    mat = materials[mat_id]
    modulus = float(mat["E_tex"])  # N/mm²
    eps_uk = float(mat["eps_u"]) / 1000.0
    sigma_uk = modulus * eps_uk
    return eps_uk, sigma_uk, modulus


def _plot(mat_id: str, eps_uk: float, sigma_uk: float, modulus: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    ax.plot([0.0, eps_uk], [0.0, sigma_uk], color="black", lw=1.8, zorder=2)
    ax.plot(eps_uk, sigma_uk, "o", color="black", ms=5.5, zorder=3)
    ax.plot([eps_uk, eps_uk], [0.0, sigma_uk], ls=":", color="0.45", lw=1.0, zorder=1)
    ax.plot([0.0, eps_uk], [sigma_uk, sigma_uk], ls=":", color="0.45", lw=1.0, zorder=1)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r"$\varepsilon$ [–]")
    ax.set_ylabel(r"$\sigma$ [N/mm$^2$]")
    ax.set_title(mat_id, fontsize=11)
    ax.grid(True, color="0.85", lw=0.6)
    ax.text(
        0.04,
        0.96,
        "\n".join(
            [
                rf"$E = {modulus:.0f}\,\mathrm{{N/mm^2}}$",
                rf"$\varepsilon_{{\mathrm{{uk}}}} = {eps_uk * 1000:.1f}\,\mathrm{{\perthousand}}$",
                rf"$\sigma_{{\mathrm{{uk}}}} = {sigma_uk:.0f}\,\mathrm{{N/mm^2}}$",
            ]
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        color="black",
    )
    fig.savefig(path)
    plt.close(fig)


def _plot_both(curves: list[tuple[str, float, float, float]], path: Path) -> None:
    """Both rovings on one pair of axes. 1R solid black, 2R dashed black."""
    styles = (
        {"color": "black", "ls": "-"},
        {"color": "black", "ls": "--"},
    )
    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    notes: list[tuple[str, dict]] = []
    for (mat_id, eps_uk, sigma_uk, modulus), style in zip(curves, styles):
        short = "1R" if "1R" in mat_id else "2R"
        ax.plot(
            [0.0, eps_uk],
            [0.0, sigma_uk],
            color=style["color"],
            ls=style["ls"],
            lw=1.8,
            zorder=2,
            label=short,
        )
        ax.plot(
            [eps_uk, eps_uk],
            [0.0, sigma_uk],
            ls=":",
            color=style["color"],
            lw=1.0,
            zorder=1,
        )
        ax.plot(
            [0.0, eps_uk],
            [sigma_uk, sigma_uk],
            ls=":",
            color=style["color"],
            lw=1.0,
            zorder=1,
        )
        note = "\n".join(
            [
                short,
                rf"$E = {modulus:.0f}\,\mathrm{{N/mm^2}}$",
                rf"$\varepsilon_{{\mathrm{{uk}}}} = {eps_uk * 1000:.1f}\,\mathrm{{\perthousand}}$",
                rf"$\sigma_{{\mathrm{{uk}}}} = {sigma_uk:.0f}\,\mathrm{{N/mm^2}}$",
            ]
        )
        if short == "1R":
            notes.append((note, {"x": 0.04, "ha": "left"}))
        else:
            notes.append((note, {"x": 0.96, "ha": "right"}))

    ax.set_xlim(*XLIM)
    ax.set_ylim(0.0, 5200.0)
    ax.set_xlabel(r"$\varepsilon$ [–]")
    ax.set_ylabel(r"$\sigma$ [N/mm$^2$]")
    ax.set_title("solidian Roving 1R and 2R (Z-1.6-308)", fontsize=11)
    ax.grid(True, color="0.85", lw=0.6)
    ax.legend(loc="lower right", frameon=False)
    for note, place in notes:
        ax.text(
            place["x"],
            0.96,
            note,
            transform=ax.transAxes,
            va="top",
            ha=place["ha"],
            fontsize=10,
            color="black",
        )
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    materials = load_materials_registry(HP_DIR / "materials.csv")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves: list[tuple[str, float, float, float]] = []
    for mat_id, filename in ROVINGS:
        eps_uk, sigma_uk, modulus = _law(materials, mat_id)
        curves.append((mat_id, eps_uk, sigma_uk, modulus))
        path = OUT_DIR / filename
        _plot(mat_id, eps_uk, sigma_uk, modulus, path)
        print(
            f"{mat_id}: eps_uk={eps_uk:.6f} sigma_uk={sigma_uk:.1f} "
            f"E={modulus:.0f} -> {path}",
            flush=True,
        )
    both = OUT_DIR / "cfrp_1r_2r_stress_strain.svg"
    _plot_both(curves, both)
    print(f"both -> {both}", flush=True)


if __name__ == "__main__":
    main()
