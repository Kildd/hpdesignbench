"""
Shared matplotlib styling for algorithm-benchmarking plots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=False)

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np

# TU corporate palette (copied hex values; do not import structural visualization.py).
TU_COLORS: dict[str, str] = {
    "BLACK": "#000000",
    "DARK GREY": "#434343",
    "LIGHT GREY": "#b2b2b2",
    "RED": "#c40d20",
    "ORANGE": "#ff6e00",
    "VIOLET": "#8f13fc",
    "BLUE": "#1f91cc",
    "GREEN": "#47cb3f",
}

_OVERLAY_CYCLE: tuple[str, ...] = (
    TU_COLORS["RED"],
    TU_COLORS["BLUE"],
    TU_COLORS["GREEN"],
    TU_COLORS["ORANGE"],
    TU_COLORS["VIOLET"],
    TU_COLORS["DARK GREY"],
    TU_COLORS["BLACK"],
    TU_COLORS["LIGHT GREY"],
)

# Distinct unfilled-friendly markers (open faces applied in plot kwargs).
_OVERLAY_MARKERS: tuple[str, ...] = (
    "o",  # circle
    "s",  # square
    "^",  # triangle up
    "D",  # diamond
    "v",  # triangle down
    "<",  # triangle left
    ">",  # triangle right
    "p",  # pentagon
)

# Thin line / marker defaults for line plots.
LINEWIDTH_MAIN = 0.55
LINEWIDTH_INDIVIDUAL = 0.3
MARKERSIZE_MAIN = 3.8
MARKERSIZE_ERT = 4.8
MARKEREDGEWIDTH = 0.5
MARKER_FACECOLOR = "none"
MARKER_GAP_FACTOR = 1.1


def overlay_color(index: int) -> str:
    """
    Return a stable overlay colour by series index.

    Parameters
    ----------
    index : int
        Zero-based overlay order.

    Returns
    -------
    str
        Hex colour.
    """
    return _OVERLAY_CYCLE[int(index) % len(_OVERLAY_CYCLE)]


def overlay_marker(index: int) -> str:
    """
    Return a stable overlay marker by series index.

    Parameters
    ----------
    index : int
        Zero-based overlay order.

    Returns
    -------
    str
        Matplotlib marker code.
    """
    return _OVERLAY_MARKERS[int(index) % len(_OVERLAY_MARKERS)]


def unfilled_marker_kwargs(
    color: str,
    marker: str,
    markersize: float,
) -> dict[str, object]:
    """
    Keyword arguments for open (unfilled) markers on a line plot.

    Parameters
    ----------
    color : str
        Edge / line colour.
    marker : str
        Matplotlib marker code.
    markersize : float
        Marker size in points.

    Returns
    -------
    dict
        kwargs for ``Axes.plot`` / legend proxies.
    """
    return {
        "marker": marker,
        "markersize": float(markersize),
        "fillstyle": "none",
        "markerfacecolor": MARKER_FACECOLOR,
        "markeredgecolor": color,
        "markeredgewidth": MARKEREDGEWIDTH,
    }


def _marker_index_spacing(
    line,
    *,
    markersize: float,
    gap_factor: float,
    dpi: float,
) -> list[int]:
    """
    Indices whose horizontal display spacing is at least
    ``markersize * gap_factor`` pixels.

    Horizontal (x) spacing matches the usual paper convention that symbols
    repeat every ~1.1 symbol widths across the axes, without packing markers
    on steep vertical segments.
    """
    x_data, y_data = line.get_data(orig=False)
    x_arr = np.asarray(x_data, dtype=float).reshape(-1)
    y_arr = np.asarray(y_data, dtype=float).reshape(-1)
    n = int(min(x_arr.size, y_arr.size))
    if n <= 0:
        return []
    if n == 1:
        return [0]
    display = line.get_transform().transform(
        np.column_stack([x_arr[:n], y_arr[:n]])
    )
    min_gap = float(markersize) * float(dpi) / 72.0 * float(gap_factor)
    chosen: list[int] = [0]
    last_x = float(display[0, 0])
    for i in range(1, n):
        if abs(float(display[i, 0]) - last_x) >= min_gap:
            chosen.append(i)
            last_x = float(display[i, 0])
    return chosen


def apply_symbol_spaced_markevery(
    fig: Figure,
    marked_lines: Sequence[tuple[object, float]],
    *,
    gap_factor: float = MARKER_GAP_FACTOR,
) -> None:
    """
    Place markers every ``markersize * gap_factor`` in horizontal display pixels.

    Parameters
    ----------
    fig : Figure
        Figure owning the lines (must be drawable).
    marked_lines : sequence of (Line2D, markersize)
        Lines that should receive spaced open markers.
    gap_factor : float
        Multiplier on symbol width (default ``1.1``).
    """
    if not marked_lines:
        return
    fig.canvas.draw()
    dpi = float(fig.dpi)
    for line, markersize in marked_lines:
        indices = _marker_index_spacing(
            line,
            markersize=float(markersize),
            gap_factor=float(gap_factor),
            dpi=dpi,
        )
        if not indices:
            line.set_markevery(1)
        else:
            line.set_markevery(indices)


def save_figure(fig: Figure, out_dir: Path, stem: str) -> list[Path]:
    """
    Write PNG and PDF for one figure and close it.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure.
    out_dir : Path
        Output directory (created if needed).
    stem : str
        Filename stem without extension.

    Returns
    -------
    list of Path
        Written ``.png`` and ``.pdf`` paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def annotate_empty(ax, message: str = "no data") -> None:
    """Place a centred annotation when a panel has no samples."""
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=TU_COLORS["DARK GREY"],
    )
    ax.set_xticks([])
    ax.set_yticks([])
