"""
High-level API for algorithm-benchmarking optimisation plots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from core.visualization_core.optimization.completion import (
    prepare_plot_bundle,
    scheduled_algorithm_names,
)
from core.visualization_core.optimization.models import ExperimentBundle, PlotConfig
from core.visualization_core.optimization.plots.bar_first_feasible import (
    render_bar_first_feasible,
)
from core.visualization_core.optimization.plots.bar_wall_time import (
    render_bar_wall_time,
)
from core.visualization_core.optimization.plots.box_runtime import render_box_runtime
from core.visualization_core.optimization.plots.compare import (
    ComparePanel,
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
from core.visualization_core.optimization.plots.summary_table import render_summary_table
from core.visualization_core.optimization.plots.violin_runtime import render_violin_runtime

PlotName = str

_RENDERERS: dict[
    str,
    Callable[[ExperimentBundle, PlotConfig, Path], list[Path]],
] = {
    "convergence": render_convergence,
    "ert": render_ert,
    "box": render_box_runtime,
    "hist": render_hist_runtime,
    "ecdf": render_ecdf_runtime,
    "ecdf_success": render_ecdf_success,
    "violin": render_violin_runtime,
    "bar_first_feasible": render_bar_first_feasible,
    "bar_wall_time": render_bar_wall_time,
    "summary": render_summary_table,
}

_COMPARE_RENDERERS: dict[
    str,
    Callable[..., list[Path]],
] = {
    "convergence": render_convergence_compare,
    "ert": render_ert_compare,
    "box": render_box_runtime_compare,
    "hist": render_hist_runtime_compare,
    "ecdf": render_ecdf_runtime_compare,
    "ecdf_success": render_ecdf_success_compare,
    "violin": render_violin_runtime_compare,
    "bar_first_feasible": render_bar_first_feasible_compare,
    "bar_wall_time": render_bar_wall_time_compare,
    "summary": render_summary_compare,
}

DEFAULT_PLOTS: tuple[str, ...] = (
    "convergence",
    "ert",
    "box",
    "hist",
    "ecdf_success",
    "violin",
    "bar_first_feasible",
    "bar_wall_time",
    "summary",
)


class PlotApiError(ValueError):
    """Raised when plot suite arguments are invalid."""


def render_suite(
    bundle: ExperimentBundle,
    config: PlotConfig,
    out_dir: Path,
    *,
    which: Sequence[str] | None = None,
    exclude_algorithms: Sequence[str] = (),
    problem_id: str | None = None,
) -> list[Path]:
    """
    Render selected benchmarking plots for a loaded experiment bundle.

    Parameters
    ----------
    bundle : ExperimentBundle
        One or more loaded experiment directories.
    config : PlotConfig
        Display / filter / target settings.
    out_dir : Path
        Output directory for PNG/PDF files.
    which : sequence of str or None
        Plot names from ``convergence``, ``ert``, ``box``, ``hist``,
        ``ecdf``, ``ecdf_success``, ``violin``, ``bar_first_feasible``,
        ``bar_wall_time``, ``summary``. ``None`` renders the full default
        suite.
    exclude_algorithms : sequence of str
        Sealed algorithm names to omit before the completed-repetition check.
    problem_id : str or None
        When set, keep this problem after the unfinished-task scan. Pass the
        full loaded bundle so unfinished tasks on other problems are visible.

    Returns
    -------
    list of Path
        All written figure paths in render order.

    Raises
    ------
    ValueError
        If a remaining algorithm has an unfinished task, a plotted run has
        no positive sealed optimum, or exclusion removes every completed run.
    """
    bundle = prepare_plot_bundle(
        bundle,
        exclude_algorithms,
        problem_id=problem_id,
    )
    selected = tuple(which) if which is not None else DEFAULT_PLOTS
    if not selected:
        raise PlotApiError("render_suite requires at least one plot name")
    unknown = [name for name in selected if name not in _RENDERERS]
    if unknown:
        raise PlotApiError(
            f"unknown plot name(s) {unknown!r}; "
            f"allowed: {sorted(_RENDERERS)}"
        )
    problem_ids = bundle.problem_ids()
    if len(problem_ids) > 1:
        raise PlotApiError(
            "render_suite cannot pool runs from multiple problems with "
            f"different absolute objectives; select one problem_id first "
            f"(available: {list(problem_ids)!r})"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in selected:
        written.extend(_RENDERERS[name](bundle, config, out))
    return written


def render_suite_compare(
    panels: Sequence[ComparePanel],
    out_dir: Path,
    *,
    which: Sequence[str] | None = None,
    title: str | None = None,
    exclude_algorithms: Sequence[str] = (),
    problem_id: str | None = None,
) -> list[Path]:
    """
    Render faceted multi-study / multi-problem comparison plots.

    Each panel is ``(panel_title, bundle, config)``. Unlike multi-directory
    overlays in :func:`render_suite` (one shared axes with disambiguated
    labels), this keeps one facet per study so each problem keeps its own
    optimum / trajectories.

    Runtime seaborn facets (box/hist/ecdf/violin) show each panel's **first**
    ``target_percents`` entry (typically the sealed 1% study target).
    Default ``which`` follows :data:`DEFAULT_PLOTS` (includes ``ecdf_success``,
    not successes-only ``ecdf``).

    Parameters
    ----------
    panels : sequence of ComparePanel
        Non-empty ordered ``(title, bundle, config)`` panels.
    out_dir : Path
        Output directory.
    which : sequence of str or None
        Plot names; ``None`` uses :data:`DEFAULT_PLOTS`.
    title : str or None
        Optional figure-level title override for line/runtime compares.
    exclude_algorithms : sequence of str
        Sealed algorithm names to omit from every panel before the
        completed-repetition check. A name present on any panel is known.
    problem_id : str or None
        When set, keep this problem on every panel after the unfinished-task
        scan. Pass each panel's full bundle so unfinished tasks on other
        problems stay visible.

    Returns
    -------
    list of Path
        Written artifact paths.

    Raises
    ------
    ValueError
        If a remaining algorithm has an unfinished task, a plotted run has
        no positive sealed optimum, or exclusion removes every completed run
        from a panel.
    """
    known: list[str] = []
    for _title, bundle, _config in panels:
        for name in scheduled_algorithm_names(bundle):
            if name not in known:
                known.append(name)
    panels = tuple(
        (
            panel_title,
            prepare_plot_bundle(
                bundle,
                exclude_algorithms,
                known_algorithms=known,
                problem_id=problem_id,
            ),
            config,
        )
        for panel_title, bundle, config in panels
    )
    selected = tuple(which) if which is not None else DEFAULT_PLOTS
    if not selected:
        raise PlotApiError("render_suite_compare requires at least one plot name")
    if not panels:
        raise PlotApiError("render_suite_compare requires at least one panel")
    unknown = [name for name in selected if name not in _COMPARE_RENDERERS]
    if unknown:
        raise PlotApiError(
            f"unknown compare plot name(s) {unknown!r}; "
            f"allowed: {sorted(_COMPARE_RENDERERS)}"
        )
    for panel_title, bundle, _config in panels:
        problem_ids = bundle.problem_ids()
        if len(problem_ids) > 1:
            raise PlotApiError(
                "render_suite_compare requires one problem per panel; "
                f"panel {panel_title!r} contains {list(problem_ids)!r}"
            )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in selected:
        renderer = _COMPARE_RENDERERS[name]
        if name == "summary":
            written.extend(renderer(panels, out))
        else:
            written.extend(renderer(panels, out, title=title))
    return written
