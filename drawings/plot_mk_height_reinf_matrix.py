"""
Full vs simplified M–κ matrix for h_ges ∈ {100, 300, 600} mm.

Sweeps:
  a_tex ∈ {1.81, 10.86} mm²
  nt    ∈ {5, 30}
  kap   ∈ {0, 30, 60} %
  x     ∈ {0.0 (support), 0.5 (midspan)}

Each panel overlays:
  - full M–κ **without** force-control (raw curvature sweep)
  - full M–κ **with** force-control (production single-section path)
  - simplified M–κ (production deflection settings, force-controlled)

Writes one 3×4 figure per (height, position) under
``_mains/verification/outputs/mk_height_matrix/`` as ``.svg`` and ``.png``.
Each (``h_ges``, position) figure has its own x- and y-limits, shared by
the twelve panels in that figure. Support and midspan are not forced onto
the same scale, and limits are not shared across heights.

Extreme prestress + max ``a_tex`` / ``nt`` may soft-fail full or simplified
MK via ``InvalidSectionForMKError``. That is usually **crush-before-crack**:
no ``N = 0`` equilibrium at tension-face ``ε_ctm`` inside the allowable
compression range, so ``M_cr`` / ``κ_cr`` are undefined for our M–κ
topology. ``M_u`` may still be computable; we still exclude the case
because Full and Simplified both require a cracking keypoint. Panels are
annotated and the run continues.

Run from repo root::

    python _mains/verification/plot_mk_height_reinf_matrix.py
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from structuralcodes import set_design_code
from structuralcodes.materials.concrete import create_concrete

from core.analysis_core.material_methods import (
    get_cfrp_reinforcement_from_registry,
    get_material_properties,
)
from core.analysis_core.section_methods import (
    DEFLECTION_MK_SIMPLIFICATION,
    DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
    InvalidSectionForMKError,
    _ensure_force_controlled,
    _full_moment_curvature_method,
    calculate_moment_curvature_sls_EC,
    sls_section_EC,
)
from core.ioh_core.import_specs import load_materials_registry
from slab_construction.slabs.hp_slab.hp_model.hp_geometry import HPGeometry
from slab_construction.slabs.hp_slab.hp_model.hp_shell import HPShell

set_design_code("ec2_2004")

HP_DIR = Path("slab_construction/slabs/hp_slab")
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "mk_height_matrix"

H_MM = (100.0, 300.0, 600.0)
A_TEX_MM2 = (1.81, 10.86)
NT = (5, 30)
KAP_PERCENT = (0.0, 30.0, 60.0)

# Fixed companions (Problem-1-ish).
SPAN_MM = 6750.0
B_MM = 1350.0
HX_HGES_RATIO = 0.2
T_MM = 20.0
DY_MM = 150.0
CONC_ID = "C50/60"
X_SUPPORT = 0.0
X_MID = 0.5
MK_POSITIONS = (X_SUPPORT, X_MID)
N_AXIAL = 0.0
M_K_NUM_POINTS = 60

A_TEX_TO_REINF_ID = {
    1.81: "solidian Roving 1R (Z-1.6-308)",
    10.86: "solidian Roving 2R (Z-1.6-308)",
}


def _position_label(x: float) -> str:
    if abs(x - X_SUPPORT) < 1e-12:
        return "support"
    if abs(x - X_MID) < 1e-12:
        return "midspan"
    return f"x={x:g}"


@dataclass(frozen=True)
class MkCase:
    h_ges_mm: float
    a_tex_mm2: float
    nt: int
    kap_percent: float

    @property
    def label(self) -> str:
        return (
            f"h={self.h_ges_mm:.0f} a={self.a_tex_mm2:g} "
            f"nt={self.nt} kap={self.kap_percent:.0f}%"
        )


def iter_cases() -> Iterator[MkCase]:
    for h in H_MM:
        for a_tex in A_TEX_MM2:
            for nt in NT:
                for kap in KAP_PERCENT:
                    yield MkCase(
                        h_ges_mm=float(h),
                        a_tex_mm2=float(a_tex),
                        nt=int(nt),
                        kap_percent=float(kap),
                    )


def build_uls_section(
    case: MkCase,
    materials: dict[str, Any],
    *,
    x: float = X_MID,
):
    """Build ULS BeamSection at normalized position ``x`` for one matrix case."""
    conc_props = get_material_properties(CONC_ID, materials)
    concrete = create_concrete(
        fck=conc_props["f_ck"],
        constitutive_law="parabolarectangle",
        alpha_cc=conc_props["alpha_cc"],
        gamma_c=conc_props["gamma_c"],
        name=f"{CONC_ID} ULS",
    )
    reinf_id = A_TEX_TO_REINF_ID[case.a_tex_mm2]
    reinforcement = get_cfrp_reinforcement_from_registry(
        mat_id=reinf_id,
        materials=materials,
        prestress_percent=case.kap_percent,
        limit_state="ULS",
    )
    Hx = HX_HGES_RATIO * case.h_ges_mm
    Hy = (1.0 - HX_HGES_RATIO) * case.h_ges_mm
    geom = HPGeometry(
        B=B_MM,
        L=SPAN_MM,
        Hx=Hx,
        Hy=Hy,
        t=T_MM,
        dy=DY_MM,
        nt=case.nt,
    )
    shell = HPShell(
        geom,
        concrete,
        reinforcement,
        reinf_area=case.a_tex_mm2,
        name=case.label,
    )
    return shell.section_at(x, name=f"{case.label} @ x={x:g}")


def build_midspan_uls_section(case: MkCase, materials: dict[str, Any]):
    """Build ULS midspan BeamSection for one matrix case."""
    return build_uls_section(case, materials, x=X_MID)


def build_support_uls_section(case: MkCase, materials: dict[str, Any]):
    """Build ULS support BeamSection for one matrix case."""
    return build_uls_section(case, materials, x=X_SUPPORT)


def _signed_mk(results) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(results.m_y, dtype=float) / 1e6  # kNm
    k = np.asarray(results.chi_y, dtype=float) * 1000.0  # 1/m
    return m, k


def compute_mk_pair(section, *, n: float = N_AXIAL) -> dict[str, Any]:
    """
    Compute raw full, force-controlled full, and simplified M–κ.

    Returns
    -------
    dict
        ``full_raw`` — full sweep **without** ``_ensure_force_controlled``.
        ``full`` — same sweep **with** force-control (production).
        ``simplified`` — simplified production path (also force-controlled).
        Matching ``*_error`` strings when ``InvalidSectionForMKError`` occurs.
    """
    out: dict[str, Any] = {
        "full_raw": None,
        "full_raw_error": None,
        "full": None,
        "full_error": None,
        "simplified": None,
        "simplified_error": None,
    }
    try:
        sls = sls_section_EC(section, "TENSTIFF_PARABOLIC")
        # Two independent sweeps: force-control mutates results in place.
        out["full_raw"] = _full_moment_curvature_method(
            section=sls,
            n=n,
            m_k_num_points=M_K_NUM_POINTS,
        )
        out["full"] = _ensure_force_controlled(
            _full_moment_curvature_method(
                section=sls,
                n=n,
                m_k_num_points=M_K_NUM_POINTS,
            )
        )
    except InvalidSectionForMKError as exc:
        msg = str(exc)
        out["full_raw_error"] = msg
        out["full_error"] = msg

    try:
        out["simplified"] = calculate_moment_curvature_sls_EC(
            section,
            n=n,
            constitutive_law="TENSTIFF_PARABOLIC",
            simplification=DEFLECTION_MK_SIMPLIFICATION,
            simplification_softening=DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
            m_k_num_points=M_K_NUM_POINTS,
        )
    except InvalidSectionForMKError as exc:
        out["simplified_error"] = str(exc)
    return out


def _panel_summary(
    case: MkCase,
    pair: dict[str, Any],
    *,
    x: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        **asdict(case),
        "x": float(x),
        "position": _position_label(x),
        "label": f"{case.label} {_position_label(x)}",
        "full_raw_ok": pair["full_raw"] is not None,
        "full_raw_error": pair["full_raw_error"],
        "full_ok": pair["full"] is not None,
        "full_error": pair["full_error"],
        "simplified_ok": pair["simplified"] is not None,
        "simplified_error": pair["simplified_error"],
    }
    if pair["full_raw"] is not None:
        m_r, k_r = _signed_mk(pair["full_raw"])
        row.update(
            {
                "full_raw_n": int(m_r.size),
                "full_raw_M0_kNm": float(m_r[0]),
                "full_raw_kappa0_1_per_m": float(k_r[0]),
                "full_raw_M_cap_kNm": float(np.max(-m_r)),
            }
        )
    if pair["full"] is not None:
        m_f, k_f = _signed_mk(pair["full"])
        row.update(
            {
                "full_n": int(m_f.size),
                "full_M0_kNm": float(m_f[0]),
                "full_kappa0_1_per_m": float(k_f[0]),
                "full_M_cap_kNm": float(np.max(-m_f)),
            }
        )
    if pair["simplified"] is not None:
        m_s, k_s = _signed_mk(pair["simplified"])
        row.update(
            {
                "simp_n": int(m_s.size),
                "simp_M0_kNm": float(m_s[0]),
                "simp_kappa0_1_per_m": float(k_s[0]),
                "simp_M_cap_kNm": float(np.max(-m_s)),
            }
        )
    return row


def _soft_fail_panel_notes(pair: dict[str, Any]) -> list[str]:
    """Human-readable panel annotation for InvalidSectionForMKError soft-fails."""
    errs = [
        pair.get("full_raw_error"),
        pair.get("full_error"),
        pair.get("simplified_error"),
    ]
    present = [e for e in errs if e]
    if not present:
        return ["MK invalid"]

    crush = all("crush before cracking" in e for e in present)
    if crush and len(present) >= 2:
        return [
            "crush-before-crack",
            "M_cr unreachable under N=0",
            "(M_u may still solve;",
            "MK refused at cracking gate)",
        ]

    notes: list[str] = []
    labels = ("full raw", "full FC", "simplified")
    for label, err in zip(labels, errs):
        if not err:
            continue
        if "crush before cracking" in err:
            notes.append(f"{label}: crush-before-crack")
        elif "cracking moment invalid" in err:
            notes.append(f"{label}: M_cr invalid")
        else:
            notes.append(f"{label}: MK invalid")
    if any("crush before cracking" in (e or "") for e in errs):
        notes.append("M_cr unreachable under N=0")
        notes.append("(M_u may still solve; MK refused)")
    return notes


def _curve_limits(pairs: list[dict[str, Any]]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Shared plotting limits for one (rise, position) figure.

    Limits cover every successfully computed raw, force-controlled, and
    simplified curve in that figure, and include the origin. Failed panels
    do not set the scale.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for pair in pairs:
        for key in ("full_raw", "full", "simplified"):
            results = pair.get(key)
            if results is None:
                continue
            moment_kNm, kappa_per_m = _signed_mk(results)
            xs.append(-kappa_per_m)
            ys.append(-moment_kNm)
    if not xs:
        return (-0.01, 0.01), (0.0, 1.0)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    if finite_x.size == 0 or finite_y.size == 0:
        return (-0.01, 0.01), (0.0, 1.0)

    def _pad(low: float, high: float) -> tuple[float, float]:
        span = high - low
        if span <= 0.0:
            span = abs(high) if high != 0.0 else 1.0
        return low - 0.02 * span, high + 0.06 * span

    x_low, x_high = _pad(min(0.0, float(np.min(finite_x))), float(np.max(finite_x)))
    y_low, y_high = _pad(min(0.0, float(np.min(finite_y))), float(np.max(finite_y)))
    return (x_low, x_high), (y_low, y_high)


def _plot_height_figure(
    h_ges_mm: float,
    panels: dict[tuple[float, float, int, float], dict[str, Any]],
    *,
    x: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> Path:
    """One 3×4 figure: rows=prestress, cols=(a_tex, nt)."""
    pos = _position_label(x)
    col_specs = [(a, nt) for a in A_TEX_MM2 for nt in NT]
    fig, axes = plt.subplots(
        nrows=len(KAP_PERCENT),
        ncols=len(col_specs),
        figsize=(14, 9),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    fig.suptitle(
        f"Deformation-controlled, detailed, and simplified M–κ — h_ges = {h_ges_mm:.0f} mm, "
        f"{pos} (x={x:g}) "
        f"(L={SPAN_MM:.0f}, B={B_MM:.0f}, t={T_MM:.0f}, dy={DY_MM:.0f}, "
        f"{CONC_ID}, n_full={M_K_NUM_POINTS})",
        fontsize=11,
    )

    for i_kap, kap in enumerate(KAP_PERCENT):
        for j_col, (a_tex, nt) in enumerate(col_specs):
            ax = axes[i_kap, j_col]
            pair = panels[(float(x), float(a_tex), int(nt), float(kap))]

            ax.axhline(0.0, color="0.75", lw=0.6)
            ax.axvline(0.0, color="0.75", lw=0.6)
            plotted_any = False
            grey = "0.55"
            if pair["full_raw"] is not None:
                m_r, k_r = _signed_mk(pair["full_raw"])
                ax.plot(
                    -k_r,
                    -m_r,
                    ":",
                    lw=1.6,
                    color=grey,
                    label="deformation-controlled",
                )
                plotted_any = True
            if pair["full"] is not None:
                m_f, k_f = _signed_mk(pair["full"])
                ax.plot(
                    -k_f,
                    -m_f,
                    "-",
                    lw=1.3,
                    color=grey,
                    label="detailed",
                )
                plotted_any = True
            if pair["simplified"] is not None:
                m_s, k_s = _signed_mk(pair["simplified"])
                ax.plot(
                    -k_s,
                    -m_s,
                    "--",
                    lw=1.4,
                    color="black",
                    label="simplified",
                )
                plotted_any = True
            if not plotted_any:
                notes = _soft_fail_panel_notes(pair)
                ax.text(
                    0.5,
                    0.55,
                    "\n".join(notes) if notes else "MK invalid",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.35",
                        facecolor="white",
                        edgecolor="0.4",
                        alpha=0.92,
                    ),
                )
            elif pair["full"] is None or pair["simplified"] is None:
                # Mixed success/failure (unexpected in this matrix, still annotate).
                notes = _soft_fail_panel_notes(pair)
                if notes:
                    ax.text(
                        0.02,
                        0.98,
                        "\n".join(notes),
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=6,
                        color="black",
                    )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            if i_kap == 0:
                ax.set_title(f"a={a_tex:g}\nnt={nt}", fontsize=9)
            if j_col == 0:
                ax.set_ylabel(f"kap={kap:.0f}%\n-M [kNm]", fontsize=8)
            else:
                ax.set_ylabel("-M [kNm]", fontsize=8)
            if i_kap == len(KAP_PERCENT) - 1:
                ax.set_xlabel("-κ [1/m]", fontsize=8)
            if i_kap == 0 and j_col == len(col_specs) - 1:
                ax.legend(fontsize=6, loc="best")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"mk_full_vs_simp_h{int(h_ges_mm)}mm_{pos}"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return stem.with_suffix(".svg")


def _compute_height_panels(
    h_ges_mm: float,
    materials: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> dict[tuple[float, float, int, float], dict[str, Any]]:
    """Solve every panel at one rise, for support and midspan."""
    panels: dict[tuple[float, float, int, float], dict[str, Any]] = {}
    for x in MK_POSITIONS:
        for a_tex in A_TEX_MM2:
            for nt in NT:
                for kap in KAP_PERCENT:
                    case = MkCase(
                        h_ges_mm=h_ges_mm,
                        a_tex_mm2=a_tex,
                        nt=nt,
                        kap_percent=kap,
                    )
                    section = build_uls_section(case, materials, x=x)
                    pair = compute_mk_pair(section)
                    summary_rows.append(_panel_summary(case, pair, x=x))
                    panels[(float(x), float(a_tex), int(nt), float(kap))] = pair
                    print(f"  solved {case.label} {_position_label(x)}", flush=True)
    return panels


def main() -> None:
    materials = load_materials_registry(HP_DIR / "materials.csv")
    summary_rows: list[dict[str, Any]] = []
    paths = []
    for h in H_MM:
        print(f"Height {h:.0f} mm", flush=True)
        panels = _compute_height_panels(h, materials, summary_rows)
        for x in MK_POSITIONS:
            subset = [
                pair
                for (px, _a, _nt, _kap), pair in panels.items()
                if abs(px - x) < 1e-12
            ]
            xlim, ylim = _curve_limits(subset)
            print(
                f"  {_position_label(x)} limits x={xlim} y={ylim}",
                flush=True,
            )
            path = _plot_height_figure(
                h, panels, x=x, xlim=xlim, ylim=ylim,
            )
            paths.append(path)
            print(f"Wrote {path}", flush=True)

    summary_path = OUT_DIR / "mk_height_matrix_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    n_full_soft = sum(1 for r in summary_rows if not r["full_ok"])
    n_simp_soft = sum(1 for r in summary_rows if not r["simplified_ok"])
    print(
        f"Wrote {summary_path} ({len(summary_rows)} cases, "
        f"{n_full_soft} full soft-fails, {n_simp_soft} simplified soft-fails)"
    )
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
