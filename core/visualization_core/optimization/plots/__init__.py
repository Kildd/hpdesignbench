"""Plot renderers for algorithm-benchmarking optimisation results."""
from __future__ import annotations

from core.visualization_core.optimization.plots.bar_first_feasible import (
    render_bar_first_feasible,
)
from core.visualization_core.optimization.plots.bar_wall_time import (
    render_bar_wall_time,
)
from core.visualization_core.optimization.plots.box_runtime import render_box_runtime
from core.visualization_core.optimization.plots.compare import (
    render_bar_first_feasible_compare,
    render_bar_wall_time_compare,
    render_box_runtime_compare,
    render_convergence_compare,
    render_ecdf_runtime_compare,
    render_ecdf_success_compare,
    render_ert_compare,
    render_hist_runtime_compare,
    render_summary_compare,
    render_violin_runtime_compare,
)
from core.visualization_core.optimization.plots.convergence import render_convergence
from core.visualization_core.optimization.plots.ecdf_runtime import render_ecdf_runtime
from core.visualization_core.optimization.plots.ecdf_success import render_ecdf_success
from core.visualization_core.optimization.plots.ert import render_ert
from core.visualization_core.optimization.plots.hist_runtime import render_hist_runtime
from core.visualization_core.optimization.plots.success_heatmap import (
    render_success_heatmap,
)
from core.visualization_core.optimization.plots.summary_table import render_summary_table
from core.visualization_core.optimization.plots.violin_runtime import render_violin_runtime

__all__ = [
    "render_bar_first_feasible",
    "render_bar_first_feasible_compare",
    "render_bar_wall_time",
    "render_bar_wall_time_compare",
    "render_box_runtime",
    "render_box_runtime_compare",
    "render_convergence",
    "render_convergence_compare",
    "render_ecdf_runtime",
    "render_ecdf_runtime_compare",
    "render_ecdf_success",
    "render_ecdf_success_compare",
    "render_ert",
    "render_ert_compare",
    "render_hist_runtime",
    "render_hist_runtime_compare",
    "render_success_heatmap",
    "render_summary_compare",
    "render_summary_table",
    "render_violin_runtime",
    "render_violin_runtime_compare",
]
