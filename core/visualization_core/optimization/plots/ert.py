"""Hansen ERT plot renderer."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator

from core.visualization_core.optimization.aggregate import hansen_ert_grid
from core.visualization_core.optimization.models import ExperimentBundle, PlotConfig
from core.visualization_core.optimization.plots.style import (
    LINEWIDTH_MAIN,
    MARKERSIZE_ERT,
    annotate_empty,
    apply_symbol_spaced_markevery,
    overlay_color,
    overlay_marker,
    save_figure,
    unfilled_marker_kwargs,
)


def apply_ert_log_yaxis(ax, erts: Sequence[float]) -> None:
    """
    Apply the default Hansen-ERT evaluation axis: log₁₀ with decade ticks.

    Parameters
    ----------
    ax :
        Matplotlib axes.
    erts : sequence of float
        Finite ERT values that will be plotted (must be positive).
    """
    values = [float(v) for v in erts if v is not None and float(v) > 0.0]
    if not values:
        ax.grid(True, axis="y", alpha=0.3)
        return
    ax.set_yscale("log")
    y_min = max(1.0, min(values))
    y_max = max(values)
    bottom = 10.0 ** np.floor(np.log10(y_min))
    top = 10.0 ** np.ceil(np.log10(max(y_max, y_min * 10.0)))
    ax.set_ylim(bottom, top)
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto"))
    ax.grid(True, axis="y", which="major", alpha=0.35)
    ax.grid(True, axis="y", which="minor", alpha=0.15)


def apply_ert_epsilon_xticks(ax, percents: Sequence[float]) -> None:
    """Set target-deviation ticks; rotate labels when the grid is dense."""
    if not percents:
        return
    values = [float(p) for p in percents]
    ax.set_xticks(values)
    ax.set_xticklabels([f"{p:g}" for p in values])
    if len(values) > 6:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")


def render_ert(
    bundle: ExperimentBundle,
    config: PlotConfig,
    out_dir: Path,
    *,
    stem: str = "ert",
) -> list[Path]:
    """
    Render Hansen ERT versus target deviation percent.

    Default style: denser ``PlotConfig.target_percents`` grid near the
    optimum, logarithmic evaluation axis, legend labels without per-target
    success shares (those vary across ``ε``).

    Parameters
    ----------
    bundle : ExperimentBundle
        Loaded experiments.
    config : PlotConfig
        Target / reference settings.
    out_dir : Path
        Output directory.
    stem : str
        Output filename stem.

    Returns
    -------
    list of Path
        Written PNG/PDF paths.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    labels = bundle.overlay_labels()
    percents = tuple(float(p) for p in config.target_percents)
    if not labels:
        annotate_empty(ax, "no runs")
    any_point = False
    all_erts: list[float] = []
    marked_lines: list[tuple[object, float]] = []
    for idx, label in enumerate(labels):
        runs = bundle.runs_for(label)
        results = hansen_ert_grid(
            runs,
            percents,
            global_optimum=config.global_optimum,
            target_reference=config.target_reference,
        )
        xs: list[float] = []
        ys: list[float] = []
        for result in results:
            if not result.defined or result.ert is None:
                continue
            xs.append(float(result.percent))
            ys.append(float(result.ert))
            all_erts.append(float(result.ert))
        color = overlay_color(idx)
        marker = overlay_marker(idx)
        if xs:
            any_point = True
            arts = ax.plot(
                xs,
                ys,
                color=color,
                linewidth=LINEWIDTH_MAIN,
                label=str(label),
                **unfilled_marker_kwargs(color, marker, MARKERSIZE_ERT),
            )
            if arts:
                marked_lines.append((arts[0], MARKERSIZE_ERT))
        else:
            ax.plot(
                [],
                [],
                color=color,
                linewidth=LINEWIDTH_MAIN,
                label=str(label),
                **unfilled_marker_kwargs(color, marker, MARKERSIZE_ERT),
            )

    ax.set_xlabel(r"Target deviation $\varepsilon$ [%]")
    ax.set_ylabel("ERT [evaluations]")
    ax.set_title("Hansen ERT")
    apply_ert_epsilon_xticks(ax, percents)
    if any_point and all_erts:
        apply_ert_log_yaxis(ax, all_erts)
    else:
        ax.grid(True, axis="y", alpha=0.3)
    if labels:
        ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    apply_symbol_spaced_markevery(fig, marked_lines)
    return save_figure(fig, out_dir, stem)
