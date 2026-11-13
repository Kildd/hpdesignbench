"""Long-form frames for seaborn statistical runtime plots."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from core.visualization_core.optimization.aggregate import runtime_sample
from core.visualization_core.optimization.models import (
    ExperimentBundle,
    PlotConfig,
)
from core.visualization_core.optimization.plots.style import overlay_color


def runtime_frame(
    bundle: ExperimentBundle,
    config: PlotConfig,
    *,
    percents: Sequence[float] | None = None,
) -> pd.DataFrame:
    """
    Build a tidy runtime table for seaborn box/hist/ECDF/violin plots.

    Columns: ``algorithm``, ``percent``, ``evaluations``, ``color``.

    Parameters
    ----------
    bundle : ExperimentBundle
        Loaded experiments.
    config : PlotConfig
        Success filter and target settings.
    percents : sequence of float or None
        Target percents; defaults to ``config.target_percents``.

    Returns
    -------
    DataFrame
        One row per runtime sample (may be empty).
    """
    targets = tuple(percents) if percents is not None else config.target_percents
    rows: list[dict[str, object]] = []
    for idx, label in enumerate(bundle.overlay_labels()):
        color = overlay_color(idx)
        for percent in targets:
            times, _, _ = runtime_sample(
                bundle.runs_for(label),
                float(percent),
                success_filter=config.success_filter,
                global_optimum=config.global_optimum,
                target_reference=config.target_reference,
            )
            for value in times:
                rows.append(
                    {
                        "algorithm": label,
                        "percent": float(percent),
                        "evaluations": float(value),
                        "color": color,
                    }
                )
    return pd.DataFrame(
        rows, columns=["algorithm", "percent", "evaluations", "color"]
    )


def algorithm_palette(bundle: ExperimentBundle) -> dict[str, str]:
    """Map overlay labels to TU overlay colours."""
    return {
        label: overlay_color(idx)
        for idx, label in enumerate(bundle.overlay_labels())
    }
