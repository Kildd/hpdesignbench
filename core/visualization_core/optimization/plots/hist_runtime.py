"""Histogram of fixed-target runtimes (seaborn)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from core.visualization_core.optimization.models import ExperimentBundle, PlotConfig
from core.visualization_core.optimization.plots.frames import (
    algorithm_palette,
    runtime_frame,
)
from core.visualization_core.optimization.plots.style import annotate_empty, save_figure


def _binwidth_for_half_cm(ax, x_span: float, *, half_cm: float = 0.5) -> float:
    """
    Choose a data bin width that is about ``half_cm`` centimetres on screen.

    Parameters
    ----------
    ax : Axes
        Target axes (must already have a figure size).
    x_span : float
        Data range on the x-axis (max - min), strictly positive.
    half_cm : float
        Desired bar width in centimetres.

    Returns
    -------
    float
        Bin width in evaluation units.
    """
    fig = ax.get_figure()
    fig_w_in, _ = fig.get_size_inches()
    # ~80% of figure width is the axes area after labels/legend.
    axes_w_cm = max(fig_w_in * 2.54 * 0.80, 1.0)
    n_bars = max(int(round(axes_w_cm / half_cm)), 8)
    return max(float(x_span) / float(n_bars), 1.0)


def render_hist_runtime(
    bundle: ExperimentBundle,
    config: PlotConfig,
    out_dir: Path,
    *,
    stem: str = "hist_runtime",
    percent: float | None = None,
) -> list[Path]:
    """
    Render dodge-bar histograms of hitting times at one target percent.

    Uses narrow bins (~0.5 cm on the figure) and ``multiple='dodge'`` so each
    algorithm's counts remain visible. COBYLA-style overlays with zero
    successes are omitted from the bars (they have no hitting times under
    ``successes``) but remain listed in the title annotation when present.

    Parameters
    ----------
    bundle : ExperimentBundle
        Loaded experiments.
    config : PlotConfig
        Success filter and target settings.
    out_dir : Path
        Output directory.
    stem : str
        Output filename stem.
    percent : float or None
        Target percent; defaults to the first entry in ``config.target_percents``.

    Returns
    -------
    list of Path
        Written PNG/PDF paths.
    """
    target = float(percent if percent is not None else config.target_percents[0])
    labels = bundle.overlay_labels()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if not labels:
        annotate_empty(ax, "no runs")
        return save_figure(fig, out_dir, stem)

    frame = runtime_frame(bundle, config, percents=(target,))
    missing = [
        label
        for label in labels
        if frame.empty or label not in set(frame["algorithm"].tolist())
    ]
    if frame.empty:
        annotate_empty(ax, "no successes")
    else:
        x_min = float(frame["evaluations"].min())
        x_max = float(frame["evaluations"].max())
        span = max(x_max - x_min, 1.0)
        binwidth = _binwidth_for_half_cm(ax, span, half_cm=0.5)
        # Pad edges so the first/last points are not clipped by bin edges.
        start = x_min - 0.5 * binwidth
        stop = x_max + 0.5 * binwidth
        bins = np.arange(start, stop + binwidth, binwidth)
        sns.histplot(
            data=frame,
            x="evaluations",
            hue="algorithm",
            hue_order=[lab for lab in labels if lab not in missing],
            palette=algorithm_palette(bundle),
            element="bars",
            multiple="dodge",
            shrink=0.85,
            bins=bins,
            stat="count",
            ax=ax,
        )
        # Explicit success counts in the legend labels.
        handles, leg_labels = ax.get_legend_handles_labels()
        counts = frame.groupby("algorithm")["evaluations"].size().to_dict()
        new_labels = [
            f"{name} (n={int(counts.get(name, 0))})" for name in leg_labels
        ]
        if handles:
            ax.legend(handles, new_labels, loc="best", frameon=False, title=None)
        ax.set_xlabel(f"Evaluations to {target:g}% target")
        ax.set_ylabel("Count")
        ax.set_ylim(bottom=0)

    note = ""
    if missing:
        note = f"; no successes: {', '.join(missing)}"
    ax.set_title(f"Runtime histogram ({config.success_filter}){note}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, out_dir, stem)
