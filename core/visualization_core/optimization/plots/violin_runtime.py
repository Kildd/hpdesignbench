"""Violin plot of evaluations-to-target (seaborn)."""
from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns

from core.visualization_core.optimization.models import ExperimentBundle, PlotConfig
from core.visualization_core.optimization.plots.frames import (
    algorithm_palette,
    runtime_frame,
)
from core.visualization_core.optimization.plots.style import annotate_empty, save_figure


def render_violin_runtime(
    bundle: ExperimentBundle,
    config: PlotConfig,
    out_dir: Path,
    *,
    stem: str = "violin_runtime",
) -> list[Path]:
    """
    Render seaborn violin plots of hitting times for each target percent.

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

    Returns
    -------
    list of Path
        Written PNG/PDF paths.
    """
    percents = config.target_percents
    n_panels = max(1, len(percents))
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(3.2 * n_panels, 4.5),
        sharey=True,
        squeeze=False,
    )
    axes_flat = list(axes[0])
    labels = bundle.overlay_labels()
    palette = algorithm_palette(bundle)
    frame = runtime_frame(bundle, config)

    for panel_idx, percent in enumerate(percents):
        ax = axes_flat[panel_idx]
        if not labels:
            annotate_empty(ax)
            continue
        panel = frame[frame["percent"] == float(percent)]
        if panel.empty:
            annotate_empty(ax, "no successes")
        else:
            # Violin needs enough points per category; fall back to strip+box
            # style density when a group has a single observation.
            counts = panel.groupby("algorithm")["evaluations"].transform("size")
            if int(counts.min()) < 2:
                sns.boxplot(
                    data=panel,
                    x="algorithm",
                    y="evaluations",
                    hue="algorithm",
                    palette=palette,
                    legend=False,
                    ax=ax,
                )
                sns.stripplot(
                    data=panel,
                    x="algorithm",
                    y="evaluations",
                    hue="algorithm",
                    palette=palette,
                    legend=False,
                    ax=ax,
                    size=6,
                )
            else:
                sns.violinplot(
                    data=panel,
                    x="algorithm",
                    y="evaluations",
                    hue="algorithm",
                    palette=palette,
                    legend=False,
                    cut=0,
                    inner="quartile",
                    ax=ax,
                )
        ax.set_title(f"{percent:g}% target")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)

    axes_flat[0].set_ylabel("Evaluations to target")
    fig.suptitle(f"Runtime violin plots ({config.success_filter})", y=1.02)
    fig.tight_layout()
    return save_figure(fig, out_dir, stem)
