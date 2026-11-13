"""ECDF of fixed-target runtimes (seaborn)."""
from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns

from core.visualization_core.optimization.aggregate import success_probability_curve
from core.visualization_core.optimization.models import ExperimentBundle, PlotConfig
from core.visualization_core.optimization.plots.frames import (
    algorithm_palette,
    runtime_frame,
)
from core.visualization_core.optimization.plots.style import annotate_empty, save_figure


def _hue_order_by_success_rate(
    bundle: ExperimentBundle,
    target: float,
    config: PlotConfig,
) -> list[str]:
    """Overlay labels ordered by success-rate descending (ties by name)."""
    scored: list[tuple[str, float]] = []
    for label in bundle.overlay_labels():
        curve = success_probability_curve(
            bundle.runs_for(label),
            target,
            global_optimum=config.global_optimum,
            target_reference=config.target_reference,
        )
        scored.append((label, curve.success_rate))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [label for label, _ in scored]


def render_ecdf_runtime(
    bundle: ExperimentBundle,
    config: PlotConfig,
    out_dir: Path,
    *,
    stem: str = "ecdf_runtime",
    percent: float | None = None,
) -> list[Path]:
    """
    Render seaborn empirical CDFs of hitting times at one target percent.

    Legend / hue order follows success-rate descending (share of runs that
    hit the target among all completed runs).

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
    if frame.empty:
        annotate_empty(ax, "no successes")
    else:
        present = set(frame["algorithm"].astype(str))
        hue_order = [
            label
            for label in _hue_order_by_success_rate(bundle, target, config)
            if label in present
        ]
        sns.ecdfplot(
            data=frame,
            x="evaluations",
            hue="algorithm",
            hue_order=hue_order,
            palette=algorithm_palette(bundle),
            ax=ax,
        )
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(f"Evaluations to {target:g}% target")
        ax.set_ylabel("ECDF")

    ax.set_title(f"Runtime ECDF ({config.success_filter})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, out_dir, stem)
