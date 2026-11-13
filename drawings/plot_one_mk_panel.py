"""One uncut panel from the 12-panel M–κ matrix.

h = 300 mm, midspan, a_tex = 1.81 mm², nt = 5, prestress = 0 %.
The deformation-controlled sweep dips after cracking and later recovers.
Force control fills that dip and keeps the same end point, so the
terminal-softening cut does not fire.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_mk_path = Path(__file__).resolve().parent / "plot_mk_height_reinf_matrix.py"
_spec = importlib.util.spec_from_file_location("plot_mk_height_reinf_matrix", _mk_path)
_mk = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mk
_spec.loader.exec_module(_mk)

OUT = Path(__file__).resolve().parent / "outputs" / "mk_height_matrix" / "mk_panel_h300_midspan_a1_81_nt5_kap0"


def main() -> None:
    materials = _mk.load_materials_registry(_mk.HP_DIR / "materials.csv")
    case = _mk.MkCase(h_ges_mm=300.0, a_tex_mm2=1.81, nt=5, kap_percent=0.0)
    section = _mk.build_uls_section(case, materials, x=_mk.X_MID)
    pair = _mk.compute_mk_pair(section)
    if pair["full"] is None or pair["full_raw"] is None or pair["simplified"] is None:
        raise RuntimeError(pair)

    m_raw, k_raw = _mk._signed_mk(pair["full_raw"])
    m_fc, k_fc = _mk._signed_mk(pair["full"])
    m_s, k_s = _mk._signed_mk(pair["simplified"])
    if m_fc.size != m_raw.size:
        raise RuntimeError(
            f"force-controlled curve was cut: {m_fc.size} vs {m_raw.size} points"
        )

    fig, ax = plt.subplots(figsize=(5.6, 4.2), layout="constrained")
    ax.axhline(0.0, color="0.75", lw=0.6)
    ax.axvline(0.0, color="0.75", lw=0.6)
    grey = "0.45"
    ax.plot(-k_raw, -m_raw, ":", lw=1.8, color=grey, label="deformation-controlled")
    ax.plot(-k_fc, -m_fc, "-", lw=1.5, color=grey, label="detailed, force-controlled")
    ax.plot(-k_s, -m_s, "--", lw=1.6, color="black", label="simplified, force-controlled")
    ax.set_xlabel(r"$-\kappa$ [1/m]")
    ax.set_ylabel(r"$-M$ [kNm]")
    ax.set_title(
        "h = 300 mm, midspan, a = 1.81 mm², nt = 5, kap = 0 %"
    )
    ax.grid(True, color="0.88", lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=150)
    fig.savefig(OUT.with_suffix(".svg"))
    print(OUT.with_suffix(".png"))
    print(f"points raw={m_raw.size} detailed={m_fc.size} simplified={m_s.size}")


if __name__ == "__main__":
    main()
