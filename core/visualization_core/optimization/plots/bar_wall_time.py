"""Bar plot of mean wall-clock runtime (seconds) by algorithm."""
from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from core.visualization_core.optimization.aggregate import (
    algorithm_wall_time_summaries,
)
from core.visualization_core.optimization.models import ExperimentBundle, PlotConfig
from core.visualization_core.optimization.plots.frames import algorithm_palette
from core.visualization_core.optimization.plots.style import annotate_empty, save_figure


def _wall_time_bar_frame(bundle: ExperimentBundle) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    palette = algorithm_palette(bundle)
    for summary in algorithm_wall_time_summaries(bundle):
        if summary.mean_wall_time_s is None:
            continue
        rows.append(
            {
                "algorithm": summary.algorithm,
                "mean_wall_time_s": float(summary.mean_wall_time_s),
                "color": palette.get(summary.algorithm, "#000000"),
            }
        )
    return pd.DataFrame(
        rows, columns=["algorithm", "mean_wall_time_s", "color"]
    )


def render_bar_wall_time(
    bundle: ExperimentBundle,
    config: PlotConfig,
    out_dir: Path,
    *,
    stem: str = "bar_wall_time",
) -> list[Path]:
    """
    Render a bar chart of mean wall-clock seconds per algorithm.

    Parameters
    ----------
    bundle : ExperimentBundle
        Loaded experiments.
    config : PlotConfig
        Unused (kept for suite renderer signature parity).
    out_dir : Path
        Output directory.
    stem : str
        Output filename stem.

    Returns
    -------
    list of Path
        Written PNG/PDF paths.
    """
    del config
    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.55 * max(1, len(bundle.overlay_labels()))), 4.5)
    )
    frame = _wall_time_bar_frame(bundle)
    if frame.empty:
        annotate_empty(ax, "no wall times")
    else:
        order = list(frame.sort_values("mean_wall_time_s")["algorithm"])
        palette = {
            row.algorithm: row.color for row in frame.itertuples(index=False)
        }
        sns.barplot(
            data=frame,
            x="algorithm",
            y="mean_wall_time_s",
            order=order,
            hue="algorithm",
            hue_order=order,
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylabel("Mean wall time [s]")
    fig.suptitle("Mean wall-clock runtime", y=1.02)
    fig.tight_layout()
    return save_figure(fig, out_dir, stem)
