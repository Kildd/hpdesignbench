"""
Multi-study / multi-problem faceted comparison for optimisation plot suites.

Each panel is one loaded experiment (typically one problem) with its own
``PlotConfig`` (e.g. distinct ``global_optimum``). Algorithm colours are
stable across panels.
"""
from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from core.visualization_core.optimization.aggregate import (
    AggregationError,
    aggregate_convergence,
    algorithm_feasibility_summaries,
    algorithm_target_summaries,
    algorithm_wall_time_summaries,
    hansen_ert_grid,
    median_iqr_band,
    success_probability_curve,
)
from core.visualization_core.optimization.models import ExperimentBundle, PlotConfig
from core.visualization_core.optimization.plots.ert import (
    apply_ert_epsilon_xticks,
    apply_ert_log_yaxis,
)
from core.visualization_core.optimization.plots.frames import runtime_frame
from core.visualization_core.optimization.plots.style import (
    LINEWIDTH_INDIVIDUAL,
    LINEWIDTH_MAIN,
    MARKERSIZE_ERT,
    MARKERSIZE_MAIN,
    annotate_empty,
    apply_symbol_spaced_markevery,
    overlay_color,
    overlay_marker,
    save_figure,
    unfilled_marker_kwargs,
)
from core.visualization_core.optimization.plots.summary_table import (
    _COLUMNS,
    _row_dict,
)

ComparePanel = tuple[str, ExperimentBundle, PlotConfig]

_RELATIVE_Y_BOTTOM = 95.0
_RELATIVE_Y_OPTIMUM = 100.0
_RELATIVE_Y_TOP_DEFAULT = 300.0


def _to_relative_percent(values, y_star: float):
    return 100.0 * np.asarray(values, dtype=float) / float(y_star)


def algorithm_color_map(
    panels: Sequence[ComparePanel],
) -> dict[str, str]:
    """
    Assign stable TU overlay colours by ``algorithm_name`` across panels.

    Colour indices follow sorted algorithm names so the same algorithm keeps
    the same colour in every facet, independent of panel order or per-panel
    subset membership.

    Parameters
    ----------
    panels : sequence of ComparePanel
        Panel definitions.

    Returns
    -------
    dict[str, str]
        Algorithm name → hex colour.
    """
    names: set[str] = set()
    for _title, bundle, _config in panels:
        for run in bundle.runs:
            names.add(run.algorithm_name)
    order = sorted(names)
    return {name: overlay_color(idx) for idx, name in enumerate(order)}


def algorithm_marker_map(
    panels: Sequence[ComparePanel],
) -> dict[str, str]:
    """Stable markers by sorted ``algorithm_name`` across panels."""
    names: set[str] = set()
    for _title, bundle, _config in panels:
        for run in bundle.runs:
            names.add(run.algorithm_name)
    order = sorted(names)
    return {name: overlay_marker(idx) for idx, name in enumerate(order)}


def _facet_grid(n_panels: int):
    ncols = 2 if n_panels > 1 else 1
    nrows = int(math.ceil(n_panels / ncols))
    fig_w = 7.2 * ncols
    fig_h = 4.6 * nrows + 0.55
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    return fig, axes, nrows, ncols


def _hide_extra_axes(axes, n_panels: int, nrows: int, ncols: int) -> None:
    for index in range(n_panels, nrows * ncols):
        row, col = divmod(index, ncols)
        axes[row][col].axis("off")


def _algorithm_name_for_label(bundle: ExperimentBundle, label: str) -> str:
    runs = bundle.runs_for(label)
    if runs:
        return runs[0].algorithm_name
    return label


def _draw_convergence_ax(
    ax,
    bundle: ExperimentBundle,
    config: PlotConfig,
    *,
    colors: Mapping[str, str],
    markers: Mapping[str, str],
    show_legend: bool,
) -> list[tuple[object, float]]:
    labels = bundle.overlay_labels()
    labels = tuple(
        sorted(labels, key=lambda lab: _algorithm_name_for_label(bundle, lab))
    )
    y_star = config.global_optimum
    if y_star is None and bundle.runs:
        y_star = bundle.runs[0].global_optimum
    relative = (
        y_star is not None
        and np.isfinite(float(y_star))
        and float(y_star) > 0.0
    )
    y_star_f = float(y_star) if relative else None
    marked_lines: list[tuple[object, float]] = []
    if not labels:
        annotate_empty(ax, "no runs")
        return marked_lines

    for label in labels:
        algo = _algorithm_name_for_label(bundle, label)
        color = colors.get(algo, overlay_color(0))
        marker = markers.get(algo, overlay_marker(0))
        runs = bundle.runs_for(label)
        if config.display == "median" and config.show_iqr and len(runs) >= 2:
            try:
                band = median_iqr_band(
                    runs,
                    extend_short_runs=config.extend_short_runs,
                    overlay_label=label,
                )
            except AggregationError:
                band = None
            if band is not None:
                q25, q75 = band.q25, band.q75
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
        for series in aggregate_convergence(
            runs,
            display=config.display,
            extend_short_runs=config.extend_short_runs,
            overlay_label=label,
        ):
            is_individual = series.kind == "individual"
            y_vals = series.y_p_best_so_far
            if relative:
                y_vals = _to_relative_percent(y_vals, y_star_f)
            plot_kwargs: dict = {
                "color": color,
                "alpha": 0.22 if is_individual else 1.0,
                "linewidth": (
                    LINEWIDTH_INDIVIDUAL if is_individual else LINEWIDTH_MAIN
                ),
                "label": label if not is_individual else None,
                "zorder": 3,
            }
            if not is_individual:
                plot_kwargs.update(
                    unfilled_marker_kwargs(color, marker, MARKERSIZE_MAIN)
                )
            arts = ax.plot(series.evaluations, y_vals, **plot_kwargs)
            if not is_individual and arts:
                marked_lines.append((arts[0], MARKERSIZE_MAIN))

    ax.set_xlabel("Evaluations")
    if relative:
        ax.set_ylabel(r"Best-so-far $y_p$ / $y_{p,\mathrm{best}}$ [%]")
        top = (
            float(config.y_max)
            if config.y_max is not None
            else _RELATIVE_Y_TOP_DEFAULT
        )
        ax.set_ylim(bottom=_RELATIVE_Y_BOTTOM, top=top)
        ax.axhline(
            _RELATIVE_Y_OPTIMUM,
            color="#444444",
            linestyle="-",
            linewidth=0.55,
            alpha=0.85,
            zorder=1.5,
        )
        for percent in config.target_percents:
            ax.axhline(
                100.0 * (1.0 + float(percent) / 100.0),
                color="#8a8a8a",
                linestyle="--",
                linewidth=0.35,
                alpha=0.9,
                zorder=1,
            )
    else:
        ax.set_ylabel(r"Best-so-far $y_p$")
        if config.y_max is not None:
            ax.set_ylim(bottom=0.0, top=float(config.y_max))
    if show_legend and labels:
        ax.legend(loc="upper right", frameon=False, fontsize=7, handlelength=1.8)
    ax.grid(True, alpha=0.3)
    return marked_lines


def _draw_ert_ax(
    ax,
    bundle: ExperimentBundle,
    config: PlotConfig,
    *,
    colors: Mapping[str, str],
    markers: Mapping[str, str],
    show_legend: bool,
) -> list[tuple[object, float]]:
    labels = bundle.overlay_labels()
    labels = tuple(
        sorted(labels, key=lambda lab: _algorithm_name_for_label(bundle, lab))
    )
    percents = tuple(float(p) for p in config.target_percents)
    marked_lines: list[tuple[object, float]] = []
    if not labels:
        annotate_empty(ax, "no runs")
        return marked_lines
    all_erts: list[float] = []
    for label in labels:
        algo = _algorithm_name_for_label(bundle, label)
        color = colors.get(algo, overlay_color(0))
        marker = markers.get(algo, overlay_marker(0))
        results = hansen_ert_grid(
            bundle.runs_for(label),
            percents,
            global_optimum=config.global_optimum,
            target_reference=config.target_reference,
        )
        xs, ys = [], []
        for result in results:
            if not result.defined or result.ert is None:
                continue
            xs.append(float(result.percent))
            ys.append(float(result.ert))
            all_erts.append(float(result.ert))
        arts = ax.plot(
            xs,
            ys,
            color=color,
            linewidth=LINEWIDTH_MAIN,
            label=str(label),
            **unfilled_marker_kwargs(color, marker, MARKERSIZE_ERT),
        )
        if xs and arts:
            marked_lines.append((arts[0], MARKERSIZE_ERT))
    ax.set_xlabel("Target deviation [%]")
    ax.set_ylabel("Hansen ERT [evaluations]")
    apply_ert_epsilon_xticks(ax, percents)
    if all_erts:
        apply_ert_log_yaxis(ax, all_erts)
    else:
        ax.grid(True, alpha=0.3)
    if show_legend and labels:
        ax.legend(loc="best", frameon=False, fontsize=7)
    return marked_lines


def _draw_ecdf_success_ax(
    ax,
    bundle: ExperimentBundle,
    config: PlotConfig,
    *,
    colors: Mapping[str, str],
    show_legend: bool,
    percent: float | None = None,
) -> None:
    target = float(percent if percent is not None else config.target_percents[0])
    labels = bundle.overlay_labels()
    if not labels:
        annotate_empty(ax, "no runs")
        return
    # Legend / draw order: success-rate high → low (stable colour by algorithm).
    scored: list[tuple[str, float, object]] = []
    for label in labels:
        curve = success_probability_curve(
            bundle.runs_for(label),
            target,
            global_optimum=config.global_optimum,
            target_reference=config.target_reference,
        )
        scored.append((label, curve.success_rate, curve))
    scored.sort(
        key=lambda item: (
            -item[1],
            _algorithm_name_for_label(bundle, item[0]),
        )
    )

    x_right = 1.0
    for label, _rate, curve in scored:
        algo = _algorithm_name_for_label(bundle, label)
        color = colors.get(algo, overlay_color(0))
        x_right = max(x_right, float(curve.x_max))
        legend = f"{label} ({curve.n_success}/{curve.n_runs})"
        rate = curve.success_rate
        if curve.n_success == 0:
            ax.hlines(
                0.0,
                0.0,
                float(curve.x_max),
                colors=color,
                linewidth=LINEWIDTH_MAIN,
                label=legend,
            )
        else:
            hit_times = np.asarray(curve.hit_times, dtype=float)
            xs = np.concatenate([[0.0], hit_times, [float(curve.x_max)]])
            ys = np.concatenate(
                [
                    [0.0],
                    np.arange(1, curve.n_success + 1, dtype=float)
                    / float(curve.n_runs),
                    [rate],
                ]
            )
            ax.step(
                xs,
                ys,
                where="post",
                color=color,
                linewidth=LINEWIDTH_MAIN,
                label=legend,
            )
    ax.set_xlim(left=0.0, right=x_right)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(f"Evaluations to {target:g}% target")
    ax.set_ylabel(r"Fraction of all runs that hit ($n_{\mathrm{hit}}/n$)")
    if show_legend:
        ax.legend(loc="lower right", frameon=False, fontsize=6)
    ax.grid(True, alpha=0.3)


def _draw_runtime_seaborn_ax(
    ax,
    bundle: ExperimentBundle,
    config: PlotConfig,
    *,
    colors: Mapping[str, str],
    kind: str,
    percent: float,
    show_legend: bool = True,
) -> None:
    """Draw box / hist / violin / ecdf_runtime for a single target percent."""
    labels = bundle.overlay_labels()
    if not labels:
        annotate_empty(ax, "no runs")
        return
    narrow = PlotConfig(
        display=config.display,
        success_filter=config.success_filter,
        target_percents=(float(percent),),
        global_optimum=config.global_optimum,
        target_reference=config.target_reference,
        extend_short_runs=config.extend_short_runs,
        show_iqr=config.show_iqr,
        y_max=config.y_max,
    )
    frame = runtime_frame(bundle, narrow)
    if frame.empty:
        annotate_empty(ax, "no successes")
        return
    # Stable colour by algorithm name; legend order depends on plot kind.
    label_algo = {label: _algorithm_name_for_label(bundle, label) for label in labels}
    if kind == "ecdf":
        scored: list[tuple[str, float]] = []
        for label in labels:
            curve = success_probability_curve(
                bundle.runs_for(label),
                float(percent),
                global_optimum=config.global_optimum,
                target_reference=config.target_reference,
            )
            scored.append((label, curve.success_rate))
        scored.sort(key=lambda item: (-item[1], label_algo[item[0]]))
        ordered_labels = [label for label, _ in scored]
    else:
        ordered_labels = sorted(labels, key=lambda lab: label_algo[lab])
    palette = {
        label: colors.get(label_algo[label], overlay_color(0))
        for label in ordered_labels
    }
    if kind == "box":
        sns.boxplot(
            data=frame,
            x="algorithm",
            y="evaluations",
            hue="algorithm",
            order=ordered_labels,
            hue_order=ordered_labels,
            palette=palette,
            legend=show_legend,
            ax=ax,
        )
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("Evaluations to target")
    elif kind == "violin":
        sns.violinplot(
            data=frame,
            x="algorithm",
            y="evaluations",
            hue="algorithm",
            order=ordered_labels,
            hue_order=ordered_labels,
            palette=palette,
            legend=show_legend,
            cut=0,
            ax=ax,
        )
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("Evaluations to target")
    elif kind == "hist":
        sns.histplot(
            data=frame,
            x="evaluations",
            hue="algorithm",
            hue_order=ordered_labels,
            palette=palette,
            multiple="dodge",
            bins=20,
            legend=show_legend,
            ax=ax,
        )
        ax.set_xlabel("Evaluations to target")
        ax.set_ylabel("Count")
    elif kind == "ecdf":
        sns.ecdfplot(
            data=frame,
            x="evaluations",
            hue="algorithm",
            hue_order=ordered_labels,
            palette=palette,
            ax=ax,
        )
        ax.set_xlabel("Evaluations to target")
        ax.set_ylabel("ECDF")
        if not show_legend:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
    else:
        raise ValueError(f"unknown runtime kind {kind!r}")
    if show_legend:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            for text in legend.get_texts():
                text.set_fontsize(6)
    ax.grid(True, axis="y", alpha=0.3)


def render_convergence_compare(
    panels: Sequence[ComparePanel],
    out_dir: Path,
    *,
    stem: str = "convergence_compare",
    title: str | None = None,
) -> list[Path]:
    """Faceted convergence comparison across studies."""
    if not panels:
        raise ValueError("render_convergence_compare requires at least one panel")
    colors = algorithm_color_map(panels)
    markers = algorithm_marker_map(panels)
    fig, axes, nrows, ncols = _facet_grid(len(panels))
    marked_all: list[tuple[object, float]] = []
    for index, (panel_title, bundle, config) in enumerate(panels):
        row, col = divmod(index, ncols)
        ax = axes[row][col]
        marked_all.extend(
            _draw_convergence_ax(
                ax,
                bundle,
                config,
                colors=colors,
                markers=markers,
                show_legend=True,
            )
        )
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=4)
    _hide_extra_axes(axes, len(panels), nrows, ncols)
    fig.suptitle(
        title or "Convergence across problems",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    apply_symbol_spaced_markevery(fig, marked_all)
    return save_figure(fig, out_dir, stem)


def render_ert_compare(
    panels: Sequence[ComparePanel],
    out_dir: Path,
    *,
    stem: str = "ert_compare",
    title: str | None = None,
) -> list[Path]:
    """Faceted Hansen ERT comparison across studies."""
    if not panels:
        raise ValueError("render_ert_compare requires at least one panel")
    colors = algorithm_color_map(panels)
    markers = algorithm_marker_map(panels)
    fig, axes, nrows, ncols = _facet_grid(len(panels))
    marked_all: list[tuple[object, float]] = []
    for index, (panel_title, bundle, config) in enumerate(panels):
        row, col = divmod(index, ncols)
        ax = axes[row][col]
        marked_all.extend(
            _draw_ert_ax(
                ax,
                bundle,
                config,
                colors=colors,
                markers=markers,
                show_legend=True,
            )
        )
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=4)
    _hide_extra_axes(axes, len(panels), nrows, ncols)
    fig.suptitle(
        title or "Hansen ERT across problems",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    apply_symbol_spaced_markevery(fig, marked_all)
    return save_figure(fig, out_dir, stem)


def render_ecdf_success_compare(
    panels: Sequence[ComparePanel],
    out_dir: Path,
    *,
    stem: str = "ecdf_success_compare",
    title: str | None = None,
) -> list[Path]:
    """Faceted success-probability ECDF comparison (sealed/first target %)."""
    if not panels:
        raise ValueError("render_ecdf_success_compare requires at least one panel")
    colors = algorithm_color_map(panels)
    fig, axes, nrows, ncols = _facet_grid(len(panels))
    for index, (panel_title, bundle, config) in enumerate(panels):
        row, col = divmod(index, ncols)
        ax = axes[row][col]
        _draw_ecdf_success_ax(
            ax,
            bundle,
            config,
            colors=colors,
            show_legend=True,
        )
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=4)
    _hide_extra_axes(axes, len(panels), nrows, ncols)
    fig.suptitle(
        title or "Success-probability ECDF across problems",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, out_dir, stem)


def _render_runtime_kind_compare(
    panels: Sequence[ComparePanel],
    out_dir: Path,
    *,
    kind: str,
    stem: str,
    title: str,
) -> list[Path]:
    if not panels:
        raise ValueError(f"{stem} requires at least one panel")
    colors = algorithm_color_map(panels)
    fig, axes, nrows, ncols = _facet_grid(len(panels))
    for index, (panel_title, bundle, config) in enumerate(panels):
        row, col = divmod(index, ncols)
        ax = axes[row][col]
        # Facets show the first configured target percent (typically 1%).
        percent = float(config.target_percents[0])
        _draw_runtime_seaborn_ax(
            ax,
            bundle,
            config,
            colors=colors,
            kind=kind,
            percent=percent,
            show_legend=True,
        )
        ax.set_title(
            f"{panel_title} ({percent:g}% target)",
            fontsize=11,
            fontweight="bold",
            pad=4,
        )
    _hide_extra_axes(axes, len(panels), nrows, ncols)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, out_dir, stem)


def render_box_runtime_compare(
    panels: Sequence[ComparePanel],
    out_dir: Path,
    *,
    stem: str = "box_runtime_compare",
    title: str | None = None,
) -> list[Path]:
    """Faceted box plots at each panel's first target percent."""
    return _render_runtime_kind_compare(
        panels,
        out_dir,
        kind="box",
        stem=stem,
        title=title or "Runtime box plots across problems",
    )


def render_hist_runtime_compare(
    panels: Sequence[ComparePanel],
    out_dir: Path,
    *,
    stem: str = "hist_runtime_compare",
    title: str | None = None,
) -> list[Path]:
    """Faceted histograms at each panel's first target percent."""
    return _render_runtime_kind_compare(
        panels,
        out_dir,
        kind="hist",
        stem=stem,
        title=title or "Runtime histograms across problems",
    )


def render_violin_runtime_compare(
    panels: Sequence[ComparePanel],
    out_dir: Path,
    *,
    stem: str = "violin_runtime_compare",
    title: str | None = None,
) -> list[Path]:
    """Faceted violins at each panel's first target percent."""
    return _render_runtime_kind_compare(
        panels,
        out_dir,
        kind="violin",
        stem=stem,
        title=title or "Runtime violin plots across problems",
    )


def render_ecdf_runtime_compare(
    panels: Sequence[ComparePanel],
    out_dir: Path,
    *,
    stem: str = "ecdf_runtime_compare",
    title: str | None = None,
) -> list[Path]:
    """Faceted successes-only ECDFs at each panel's first target percent."""
    return _render_runtime_kind_compare(
        panels,
        out_dir,
        kind="ecdf",
        stem=stem,
        title=title or "Runtime ECDF across problems",
    )


def _draw_bar_first_feasible_ax(
    ax,
    bundle: ExperimentBundle,
    *,
    colors: Mapping[str, str],
) -> None:
    """Draw mean first-feasible FE bars for one compare facet."""
    summaries = algorithm_feasibility_summaries(bundle)
    labels: list[str] = []
    values: list[float] = []
    bar_colors: list[str] = []
    for summary in summaries:
        if summary.mean_first_feasible_fe is None:
            continue
        name = _algorithm_name_for_label(bundle, summary.algorithm)
        labels.append(name)
        values.append(float(summary.mean_first_feasible_fe))
        bar_colors.append(colors.get(name, overlay_color(0)))
    if not labels:
        annotate_empty(ax, "no feasible evaluations")
        return
    order = sorted(range(len(labels)), key=lambda i: values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    bar_colors = [bar_colors[i] for i in order]
    ax.bar(range(len(labels)), values, color=bar_colors, edgecolor="none")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Mean first feasible FE")
    ax.grid(True, axis="y", alpha=0.3)


def render_bar_first_feasible_compare(
    panels: Sequence[ComparePanel],
    out_dir: Path,
    *,
    stem: str = "bar_first_feasible_compare",
    title: str | None = None,
) -> list[Path]:
    """Faceted mean first-feasible FE bars across studies."""
    if not panels:
        raise ValueError(
            "render_bar_first_feasible_compare requires at least one panel"
        )
    colors = algorithm_color_map(panels)
    fig, axes, nrows, ncols = _facet_grid(len(panels))
    for index, (panel_title, bundle, _config) in enumerate(panels):
        row, col = divmod(index, ncols)
        ax = axes[row][col]
        _draw_bar_first_feasible_ax(ax, bundle, colors=colors)
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=4)
    _hide_extra_axes(axes, len(panels), nrows, ncols)
    fig.suptitle(
        title or "Mean first feasible evaluation across problems",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, out_dir, stem)


def _draw_bar_wall_time_ax(
    ax,
    bundle: ExperimentBundle,
    *,
    colors: Mapping[str, str],
) -> None:
    """Draw mean wall-time bars for one compare facet."""
    summaries = algorithm_wall_time_summaries(bundle)
    labels: list[str] = []
    values: list[float] = []
    bar_colors: list[str] = []
    for summary in summaries:
        if summary.mean_wall_time_s is None:
            continue
        name = _algorithm_name_for_label(bundle, summary.algorithm)
        labels.append(name)
        values.append(float(summary.mean_wall_time_s))
        bar_colors.append(colors.get(name, overlay_color(0)))
    if not labels:
        annotate_empty(ax, "no wall times")
        return
    order = sorted(range(len(labels)), key=lambda i: values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    bar_colors = [bar_colors[i] for i in order]
    ax.bar(range(len(labels)), values, color=bar_colors, edgecolor="none")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Mean wall time [s]")
    ax.grid(True, axis="y", alpha=0.3)


def render_bar_wall_time_compare(
    panels: Sequence[ComparePanel],
    out_dir: Path,
    *,
    stem: str = "bar_wall_time_compare",
    title: str | None = None,
) -> list[Path]:
    """Faceted mean wall-clock runtime bars across studies."""
    if not panels:
        raise ValueError(
            "render_bar_wall_time_compare requires at least one panel"
        )
    colors = algorithm_color_map(panels)
    fig, axes, nrows, ncols = _facet_grid(len(panels))
    for index, (panel_title, bundle, _config) in enumerate(panels):
        row, col = divmod(index, ncols)
        ax = axes[row][col]
        _draw_bar_wall_time_ax(ax, bundle, colors=colors)
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=4)
    _hide_extra_axes(axes, len(panels), nrows, ncols)
    fig.suptitle(
        title or "Mean wall-clock runtime across problems",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, out_dir, stem)


def _pool_panels_by_algorithm(
    panels: Sequence[ComparePanel],
) -> tuple[ExperimentBundle, tuple[float, ...]]:
    """
    Merge panel runs by ``algorithm_name`` for a multi-problem summary.

    Each run keeps its sealed optimum / reference so fixed-target metrics stay
    problem-correct while ``n_runs`` sums across panels (typically 4×20).
    """
    if not panels:
        raise ValueError("at least one panel required")
    percents = panels[0][2].target_percents
    for _title, _bundle, config in panels[1:]:
        if tuple(config.target_percents) != tuple(percents):
            raise ValueError(
                "compare summary pooling requires identical target_percents "
                f"across panels; got {percents!r} vs {config.target_percents!r}"
            )
    pooled_runs = []
    dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    for _title, bundle, _config in panels:
        for experiment_dir in bundle.experiment_dirs:
            if experiment_dir not in seen_dirs:
                seen_dirs.add(experiment_dir)
                dirs.append(experiment_dir)
        for run in bundle.runs:
            pooled_runs.append(
                replace(run, overlay_label=run.algorithm_name)
            )
    # Stable algorithm order: sorted by name for pooled tables.
    pooled_runs.sort(key=lambda r: (r.algorithm_name, str(r.experiment_dir), r.task_index))
    return (
        ExperimentBundle(runs=tuple(pooled_runs), experiment_dirs=tuple(dirs)),
        tuple(float(p) for p in percents),
    )


def render_summary_compare(
    panels: Sequence[ComparePanel],
    out_dir: Path,
    *,
    stem: str = "summary_by_target_compare",
) -> list[Path]:
    """
    Write a multi-problem summary pooled by algorithm across all panels.

    ``n_runs`` is the sum of completed plottable runs over P1–P4 (typically
    80 when an algorithm appears in every panel with 20 reps each). Fixed-
    target hits use each run's own sealed optimum.
    """
    if not panels:
        raise ValueError("render_summary_compare requires at least one panel")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pooled, percents = _pool_panels_by_algorithm(panels)
    # Prefer sealed per-run optimum/reference rather than a single override.
    summaries = algorithm_target_summaries(
        pooled,
        percents,
        global_optimum=None,
        target_reference=None,
    )
    rows_out = [_row_dict(row) for row in summaries]
    columns = _COLUMNS

    csv_path = out / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows_out:
            writer.writerow({key: row.get(key, "") for key in columns})

    md_path = out / f"{stem}.md"
    lines = [
        "# Fixed-target algorithm summary (multi-problem, pooled)",
        "",
        "Rows are pooled across all compare panels by algorithm. `n_runs` "
        "sums completed trajectories (typically 4×20 when an algorithm "
        "appears in every problem). Hitting times use each run's sealed "
        "optimum. Also includes `n_runs_with_feasible`, "
        "`mean_first_feasible_fe`, `mean_share_feasible`, and "
        "`mean_wall_time_s`.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows_out:
        lines.append("| " + " | ".join(row.get(c, "") for c in columns) + " |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return [csv_path, md_path]
