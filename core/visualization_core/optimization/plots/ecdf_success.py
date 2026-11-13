"""Success-probability ECDF over all runs (asymptote = hit rate)."""
from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from core.visualization_core.optimization.aggregate import (
    SuccessProbabilityCurve,
    success_probability_curve,
)
from core.visualization_core.optimization.models import ExperimentBundle, PlotConfig
from core.visualization_core.optimization.plots.style import (
    LINEWIDTH_MAIN,
    annotate_empty,
    overlay_color,
    save_figure,
)


def _labels_by_success_rate(
    bundle: ExperimentBundle,
    target: float,
    *,
    global_optimum: float | None,
    target_reference: str | None,
) -> list[tuple[str, SuccessProbabilityCurve]]:
    """
    Return ``(overlay_label, curve)`` sorted by success rate (high → low).

    Ties break by overlay label for a stable legend.
    """
    rows: list[tuple[str, SuccessProbabilityCurve]] = []
    for label in bundle.overlay_labels():
        curve = success_probability_curve(
            bundle.runs_for(label),
            target,
            global_optimum=global_optimum,
            target_reference=target_reference,  # type: ignore[arg-type]
        )
        rows.append((label, curve))
    rows.sort(key=lambda item: (-item[1].success_rate, item[0]))
    return rows


def render_ecdf_success(
    bundle: ExperimentBundle,
    config: PlotConfig,
    out_dir: Path,
    *,
    stem: str = "ecdf_success",
    percent: float | None = None,
) -> list[Path]:
    """
    Render F(t) = (# runs that hit by t) / (# all runs).

    Unlike ``ecdf_runtime`` (successes-only → asymptote 1 among hits, or
    all-runs censored at B → asymptote 1), this curve plateaus at the
    empirical success rate (e.g. 10/20 → 0.5).

    Legend order is success-rate descending (share of successful runs).

    Parameters
    ----------
    bundle : ExperimentBundle
        Loaded experiments.
    config : PlotConfig
        Target settings (``success_filter`` is ignored).
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
    color_index = {label: idx for idx, label in enumerate(labels)}
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if not labels:
        annotate_empty(ax, "no runs")
        return save_figure(fig, out_dir, stem)

    ordered = _labels_by_success_rate(
        bundle,
        target,
        global_optimum=config.global_optimum,
        target_reference=config.target_reference,
    )
    x_right = 1.0
    any_curve = False
    for label, curve in ordered:
        color = overlay_color(color_index[label])
        x_right = max(x_right, float(curve.x_max))
        rate = curve.success_rate
        legend = f"{label} ({curve.n_success}/{curve.n_runs})"
        if curve.n_success == 0:
            ax.hlines(
                0.0,
                0.0,
                float(curve.x_max),
                colors=color,
                linewidth=LINEWIDTH_MAIN,
                label=legend,
            )
            any_curve = True
            continue
        hit_times = np.asarray(curve.hit_times, dtype=float)
        xs = np.concatenate([[0.0], hit_times, [float(curve.x_max)]])
        ys = np.concatenate(
            [
                [0.0],
                np.arange(1, curve.n_success + 1, dtype=float) / float(curve.n_runs),
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
        any_curve = True

    if not any_curve:
        annotate_empty(ax, "no runs")
    else:
        ax.set_xlim(left=0.0, right=x_right)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(f"Evaluations to {target:g}% target")
        ax.set_ylabel(r"Fraction of all runs that hit ($n_{\mathrm{hit}}/n$)")
        ax.legend(loc="lower right", frameon=False, fontsize=7.5)

    ax.set_title("Success-probability ECDF (all runs)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, out_dir, stem)
