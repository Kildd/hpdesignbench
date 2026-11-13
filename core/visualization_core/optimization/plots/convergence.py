"""Convergence (anytime) plot renderer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend import Legend

from core.visualization_core.optimization.aggregate import (
    AggregationError,
    aggregate_convergence,
    median_iqr_band,
)
from core.visualization_core.optimization.models import (
    ExperimentBundle,
    PlotConfig,
)
from core.visualization_core.optimization.plots.style import (
    LINEWIDTH_INDIVIDUAL,
    LINEWIDTH_MAIN,
    MARKERSIZE_MAIN,
    annotate_empty,
    apply_symbol_spaced_markevery,
    overlay_color,
    overlay_marker,
    save_figure,
    unfilled_marker_kwargs,
)


# Relative scale: 100% = global optimum; default axis runs up to 300%.
# Bottom sits slightly below 100% so the optimum baseline stays visible.
_RELATIVE_Y_BOTTOM = 95.0
_RELATIVE_Y_OPTIMUM = 100.0
_RELATIVE_Y_TOP_DEFAULT = 300.0


def _to_relative_percent(values, y_star: float):
    """Map absolute ``y_p`` to percent of the global optimum (100% = ``y_star``)."""
    return 100.0 * np.asarray(values, dtype=float) / float(y_star)


def _legend_anchor_axes_fraction(ax, legend: Legend | None) -> tuple[float, float]:
    """
    Return an axes-fraction point just below the legend (or top-right fallback).
    """
    if legend is None:
        return 0.98, 0.98
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer)
    inv = ax.transAxes.inverted()
    x0, y0 = inv.transform((bbox.x0, bbox.y0))
    return float(x0), float(y0) - 0.025


def _draw_target_thresholds(
    ax,
    *,
    y_star: float,
    percents: tuple[float, ...],
    legend: Legend | None,
    relative: bool,
) -> None:
    """
    Draw thin dashed target lines; put formula + ε labels under the legend.

    Target definition:

    ``y_p = y_{p,best} (1 + ε)`` with labels ``ε = 1%``, …

    On the relative axis those levels sit at ``100% · (1 + ε)``.

    All annotation text stays inside the axes so ``bbox_inches='tight'`` does
    not shrink the data panel.
    """

    def _level(percent: float) -> float:
        if relative:
            return 100.0 * (1.0 + float(percent) / 100.0)
        return float(y_star) * (1.0 + float(percent) / 100.0)

    for percent in percents:
        ax.axhline(
            _level(percent),
            color="#8a8a8a",
            linestyle="--",
            linewidth=0.35,
            alpha=0.9,
            zorder=1,
        )

    y_lo, y_hi = ax.get_ylim()
    items = [
        (float(percent), _level(percent))
        for percent in percents
        if y_lo <= _level(percent) <= y_hi
    ]
    if not items:
        return
    # Top-to-bottom under the legend: larger ε first.
    items.sort(key=lambda item: item[1], reverse=True)

    anchor_x, anchor_y = _legend_anchor_axes_fraction(ax, legend)
    # Keep the note under the legend, inside the axes.
    anchor_x = min(max(anchor_x, 0.02), 0.70)
    anchor_y = min(max(anchor_y, 0.35), 0.92)
    line_h = 0.042

    ax.text(
        anchor_x,
        anchor_y,
        r"target levels: $y_p = y_{p,\mathrm{best}}(1+\varepsilon)$",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
        ha="left",
        va="top",
        zorder=5,
    )
    ax.text(
        anchor_x,
        anchor_y - line_h,
        rf"($y_{{p,\mathrm{{best}}}} = {y_star:.4g}$)",
        transform=ax.transAxes,
        fontsize=7,
        color="#666666",
        ha="left",
        va="top",
        zorder=5,
    )

    label_top = anchor_y - 2.1 * line_h
    for i, (percent, tau) in enumerate(items):
        lab_y_axes = label_top - i * line_h
        ax.annotate(
            rf"$\varepsilon={percent:g}\%$",
            xy=(0.995, tau),
            xycoords=("axes fraction", "data"),
            xytext=(anchor_x, lab_y_axes),
            textcoords=("axes fraction", "axes fraction"),
            fontsize=8,
            color="#555555",
            va="top",
            ha="left",
            zorder=5,
            arrowprops={
                "arrowstyle": "-",
                "color": "#888888",
                "lw": 0.45,
                "shrinkA": 2,
                "shrinkB": 0,
            },
        )


def render_convergence(
    bundle: ExperimentBundle,
    config: PlotConfig,
    out_dir: Path,
    *,
    stem: str = "convergence",
) -> list[Path]:
    """
    Render multi-algorithm best-so-far convergence curves.

    When a known positive global optimum is available, the y-axis is relative:
    ``100%`` is the optimum and the default upper limit is ``300%``. Without a
    usable optimum the plot falls back to absolute ``y_p``.

    When ``display="median"``, ``extend_short_runs`` is true by default so
    early-stopped fixed-target runs are carried forward to the longest run
    length. IQR ribbons are drawn when ``config.show_iqr`` is true.

    Parameters
    ----------
    bundle : ExperimentBundle
        Loaded experiments.
    config : PlotConfig
        Display / optimum settings.
    out_dir : Path
        Output directory.
    stem : str
        Output filename stem.

    Returns
    -------
    list of Path
        Written PNG/PDF paths.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    labels = bundle.overlay_labels()

    y_star = config.global_optimum
    if y_star is None and bundle.runs:
        y_star = bundle.runs[0].global_optimum
    relative = (
        y_star is not None
        and np.isfinite(float(y_star))
        and float(y_star) > 0.0
    )
    y_star_f = float(y_star) if relative else None

    if not labels:
        annotate_empty(ax, "no runs")
    marked_lines: list[tuple[object, float]] = []
    for idx, label in enumerate(labels):
        runs = bundle.runs_for(label)
        color = overlay_color(idx)
        marker = overlay_marker(idx)
        if (
            config.display == "median"
            and config.show_iqr
            and len(runs) >= 2
        ):
            try:
                band = median_iqr_band(
                    runs,
                    extend_short_runs=config.extend_short_runs,
                    overlay_label=label,
                )
            except AggregationError:
                band = None
            if band is not None:
                q25 = band.q25
                q75 = band.q75
                if relative:
                    q25 = _to_relative_percent(q25, y_star_f)
                    q75 = _to_relative_percent(q75, y_star_f)
                ax.fill_between(
                    band.evaluations,
                    q25,
                    q75,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                    zorder=2,
                )
        series_list = aggregate_convergence(
            runs,
            display=config.display,
            extend_short_runs=config.extend_short_runs,
            overlay_label=label,
        )
        for series in series_list:
            is_individual = series.kind == "individual"
            alpha = 0.22 if is_individual else 1.0
            linewidth = (
                LINEWIDTH_INDIVIDUAL if is_individual else LINEWIDTH_MAIN
            )
            legend_label = label if not is_individual else None
            y_vals = series.y_p_best_so_far
            if relative:
                y_vals = _to_relative_percent(y_vals, y_star_f)
            plot_kwargs: dict = {
                "color": color,
                "alpha": alpha,
                "linewidth": linewidth,
                "label": legend_label,
                "zorder": 3,
            }
            if not is_individual:
                plot_kwargs.update(
                    unfilled_marker_kwargs(color, marker, MARKERSIZE_MAIN)
                )
            arts = ax.plot(series.evaluations, y_vals, **plot_kwargs)
            if not is_individual and arts:
                marked_lines.append((arts[0], MARKERSIZE_MAIN))
        if config.display == "all":
            ax.plot(
                [],
                [],
                color=color,
                linewidth=LINEWIDTH_MAIN,
                label=label,
                **unfilled_marker_kwargs(color, marker, MARKERSIZE_MAIN),
            )

    ax.set_xlabel("Evaluations")
    if relative:
        ax.set_ylabel(r"Best-so-far $y_p$ / $y_{p,\mathrm{best}}$ [%]")
        top = (
            float(config.y_max)
            if config.y_max is not None
            else _RELATIVE_Y_TOP_DEFAULT
        )
        ax.set_ylim(bottom=_RELATIVE_Y_BOTTOM, top=top)
    else:
        ax.set_ylabel(r"Best-so-far $y_p$")
        if config.y_max is not None:
            ax.set_ylim(bottom=0.0, top=float(config.y_max))

    if config.display == "median" and config.show_iqr:
        ax.set_title("Convergence (median with IQR band)")
    else:
        ax.set_title(f"Convergence ({config.display})")

    legend: Legend | None = None
    if labels:
        # Keep algorithms in the upper-right so the target note can sit under it.
        legend = ax.legend(
            loc="upper right",
            frameon=False,
            handlelength=2.2,
            markerscale=0.95,
        )

    if relative:
        # Solid optimum baseline (100%), distinct from dashed ε targets.
        ax.axhline(
            _RELATIVE_Y_OPTIMUM,
            color="#444444",
            linestyle="-",
            linewidth=0.55,
            alpha=0.85,
            zorder=1.5,
        )
        _draw_target_thresholds(
            ax,
            y_star=y_star_f,
            percents=tuple(float(p) for p in config.target_percents),
            legend=legend,
            relative=True,
        )

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    apply_symbol_spaced_markevery(fig, marked_lines)
    return save_figure(fig, out_dir, stem)
