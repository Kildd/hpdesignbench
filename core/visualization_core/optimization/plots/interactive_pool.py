"""
Interactive plots that pool every problem into one series per algorithm.

Static suite overlays keep one series per problem. The pooled figures group
by ``algorithm_name`` instead, so 20 repetitions on 4 problems are 80 runs.
The same command also writes one pair of figures per experiment directory,
where 100% is every run of that problem, plus a per-problem figure whose
menu selects one algorithm and draws every repetition of that algorithm.
The pooled success curve is the only interactive ECDF. Its menus choose
the problem, the target tolerance, and whether colour encodes the
algorithm, the algorithm class, or cold versus warm start. Cold and warm
starts of one algorithm stay together. A pair is ordered by the better
variant's success share, then by the evaluation at which that variant's
curve reaches the share. The pooled expected-runtime figure uses that
same order, palette, and comparison menu. Its horizontal axis is the
relative target tolerance, and the budget menu rebuilds the Hansen ERT.
The success heatmap is written from the same loaded runs. ``which``
selects any subset of ``convergence``, ``ecdf``, ``ert``, and ``heatmap``.
Pass ``exclude_algorithms`` to omit named algorithms from every figure.
The pooled convergence figure has the same problem menu, plus a choice
between the median of every algorithm and every run of one algorithm.
Convergence is the median of each run's best-so-far divided by that run's
own sealed optimum. The success ECDF uses that same per-run optimum, and
100% means every pooled run hit the target.
"""
from __future__ import annotations

import colorsys
import html
import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from core.visualization_core.optimization.algorithm_presentation import (
    CLASS_ORDER,
    AlgorithmPresentationError,
    constraint_label,
    local_search_label,
    presentation_for,
    qualifiers_needed,
    start_label,
    tooltip_subtitle,
)
from core.visualization_core.optimization.aggregate import (
    _aligned_matrix,
    hansen_ert_from_outcomes,
    success_probability_curve,
)
from core.visualization_core.optimization.trajectories import hit_time_at_percent
from core.visualization_core.optimization.completion import prepare_plot_bundle
from core.visualization_core.optimization.models import (
    ExperimentBundle,
    RunTrajectory,
)

INTERACTIVE_PLOTS: tuple[str, ...] = ("convergence", "ecdf", "ert", "heatmap")

_DASHES = ("solid", "dash", "dot", "dashdot")
_FONT = "DejaVu Sans"
_FULL_BUDGET = 15000
_BUDGETS = (2000, 5000, 10000, 15000)
_ECDF_TITLE = "Cumulative success rate"
_RUN_CURVE_POINTS = 400
_POOLED_ECDF_PERCENTS = (1.0, 2.0, 3.0, 4.0, 5.0, 10.0)
_ALL_PROBLEMS = "all"
_MEDIAN_VIEW = "median"
_PROBLEM_ID = re.compile(r"L\d+_B\d+")
_SPAN_ID = re.compile(r"L\d+")
ProgressCallback = Callable[[int, int, str], None]


def runs_by_experiment(
    bundle: ExperimentBundle,
) -> tuple[tuple[Path, dict[str, tuple[RunTrajectory, ...]]], ...]:
    """
    Split a multi-problem bundle into one algorithm grouping per experiment.

    Parameters
    ----------
    bundle : ExperimentBundle
        Loaded experiments, in ``--exp`` order.

    Returns
    -------
    tuple
        ``(experiment_dir, algorithm grouping)`` for every directory that
        still has at least one completed run.
    """
    slices: list[tuple[Path, dict[str, tuple[RunTrajectory, ...]]]] = []
    for experiment_dir in bundle.experiment_dirs:
        runs = tuple(
            run
            for run in bundle.runs
            if Path(run.experiment_dir) == Path(experiment_dir)
        )
        if not runs:
            continue
        slices.append(
            (
                experiment_dir,
                runs_by_algorithm(
                    ExperimentBundle(runs=runs, experiment_dirs=(experiment_dir,))
                ),
            )
        )
    return tuple(slices)


def runs_by_algorithm(
    bundle: ExperimentBundle,
) -> dict[str, tuple[RunTrajectory, ...]]:
    """
    Group completed runs by sealed algorithm name, ignoring problem labels.

    Parameters
    ----------
    bundle : ExperimentBundle
        Loaded experiments. The same algorithm may appear in several of them.

    Returns
    -------
    dict
        Algorithm name to its runs, in first-seen algorithm order.
    """
    grouped: dict[str, list[RunTrajectory]] = {}
    for run in bundle.runs:
        grouped.setdefault(run.algorithm_name, []).append(run)
    return {name: tuple(runs) for name, runs in grouped.items()}


def without_excluded_algorithms(
    bundle: ExperimentBundle,
    excluded: tuple[str, ...] | list[str],
) -> ExperimentBundle:
    """
    Drop runs whose sealed algorithm name is listed in ``excluded``.

    Parameters
    ----------
    bundle : ExperimentBundle
        Loaded experiments.
    excluded : sequence of str
        Algorithm names to omit. Blank entries are ignored. Repeated names
        are ignored after the first. A name that is not in the bundle raises
        ``ValueError``.

    Returns
    -------
    ExperimentBundle
        The same experiment directories, without the excluded runs.
    """
    names = tuple(dict.fromkeys(name.strip() for name in excluded if name.strip()))
    if not names:
        return bundle
    present = list(runs_by_algorithm(bundle))
    missing = [name for name in names if name not in present]
    if missing:
        raise ValueError(f"unknown algorithm(s) {missing}; available={present}")
    blocked = set(names)
    return ExperimentBundle(
        runs=tuple(run for run in bundle.runs if run.algorithm_name not in blocked),
        experiment_dirs=bundle.experiment_dirs,
    )


def _alphabetical_rank(family: str) -> tuple[str, str, str]:
    """Display name, constraint handling, then qualifier. Unknown tokens stay literal."""
    try:
        item = presentation_for(family)
    except KeyError:
        return (family.casefold(), "", "")
    handling = "constraint-aware" if item.constraint_aware else "penalty-based"
    return (item.display_name.casefold(), handling, item.qualifier.casefold())


def better_family_sort_key(
    variant_ranks: list[tuple[float, float]],
    family_id: str,
) -> tuple[float, float, str, str, str]:
    """
    Order one family by its better variant, then by name.

    The better variant has the higher success share. The share is broken by
    the number of evaluations that variant needed to reach it, fewer first.
    """
    if variant_ranks:
        rate, reach = min(variant_ranks, key=lambda item: (-item[0], item[1]))
    else:
        rate, reach = 0.0, math.inf
    name, handling, qualifier = _alphabetical_rank(family_id)
    return (-rate, reach, name, handling, qualifier)


def ecdf_algorithm_order(
    grouped: dict[str, tuple[RunTrajectory, ...]],
    percent: float,
    *,
    family_of: Callable[[str], str] | None = None,
) -> list[str]:
    """
    Order cold/warm pairs by the better variant.

    The better variant has the higher success share. An equal share is
    broken by the number of evaluations that variant needed to reach it,
    fewer first. Equal share and evaluations are ordered alphabetically by
    display name. The cold start is listed before its warm start.

    Parameters
    ----------
    grouped : dict
        Algorithm name to its runs.
    percent : float
        Success-ECDF target, as a percent above each run's sealed optimum.
    family_of : callable or None
        Maps an algorithm token to its cold/warm family. ``None`` uses the
        display metadata.

    Returns
    -------
    list of str
        Algorithm names in legend order.
    """
    resolve = family_of if family_of is not None else (
        lambda name: presentation_for(name).family
    )
    scores = {
        name: _success_rank(runs, percent) for name, runs in grouped.items()
    }
    families: dict[str, list[str]] = {}
    for name in grouped:
        families.setdefault(resolve(name), []).append(name)

    def _family_key(family: str) -> tuple[float, float, str, str, str]:
        return better_family_sort_key(
            [scores[name] for name in families[family]],
            family,
        )

    ordered: list[str] = []
    for family in sorted(families, key=_family_key):
        members = sorted(
            families[family],
            key=lambda name: (presentation_start_rank(name), name),
        )
        ordered.extend(members)
    return ordered


def presentation_start_rank(name: str) -> int:
    """Return 0 for a cold start and 1 for a warm start, when metadata exists."""
    try:
        return 0 if presentation_for(name).start == "cold" else 1
    except KeyError:
        return 1 if name.endswith("_WARM_START") else 0


def _success_rank(
    runs: tuple[RunTrajectory, ...],
    percent: float,
) -> tuple[float, float]:
    """Success share, then the evaluation where the curve reaches that share."""
    curve = success_probability_curve(runs, float(percent))
    if int(curve.n_success) <= 0 or curve.hit_times.size == 0:
        return (float(curve.success_rate), math.inf)
    return (float(curve.success_rate), float(curve.hit_times[-1]))


def _problem_sort_key(problem_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"L(\d+)_(.+)", problem_id)
    if match is None:
        return (10**9, problem_id)
    return (int(match.group(1)), match.group(2))


# The four B2 slabs used as the reference set. Every other L####_A1/B1/B2
# instance in a combined figure is a validation instance.
_REFERENCE_PROBLEM_IDS = frozenset(
    {"L6750_B2", "L8100_B2", "L9450_B2", "L10800_B2"}
)

_PROBLEM_FACTORIAL = (
    "Each instance is one span combined with one imposed load class. "
    "The benchmark crosses seven spans, 6.750, 7.425, 8.100, 8.775, 9.450, "
    "10.125, and 10.800 m, with three imposed load classes, A1, B1, and B2, "
    "at 1, 2, and 3 kN/m², giving 21 instances. "
    "The four reference instances are imposed load class B2 at 6.750, 8.100, "
    "9.450, and 10.800 m. The 17 validation instances are the other "
    "combinations: A1 and B1 at all seven spans, and B2 at 7.425, 8.775, and "
    "10.125 m."
)


def _is_pooled_scope(scope: str) -> bool:
    """True for the all-problems scope of a figure or of one problem set."""
    return scope == _ALL_PROBLEMS or scope.endswith("/all")


def _scopes_for_runs(
    runs: tuple[RunTrajectory, ...] | list[RunTrajectory],
    *,
    prefix: str = "",
) -> list[tuple[str, str, dict[str, tuple[RunTrajectory, ...]]]]:
    """All-problems scope, then one scope per problem id."""
    material = tuple(runs)
    all_grouped = runs_by_algorithm(
        ExperimentBundle(runs=material, experiment_dirs=())
    )
    all_id = f"{prefix}/all" if prefix else "all"
    scopes: list[tuple[str, str, dict[str, tuple[RunTrajectory, ...]]]] = [
        (all_id, "All problems", all_grouped)
    ]
    by_problem: dict[str, list[RunTrajectory]] = {}
    for run in material:
        by_problem.setdefault(run.problem_id, []).append(run)
    for problem_id in sorted(by_problem, key=_problem_sort_key):
        subset = tuple(by_problem[problem_id])
        scope_id = f"{prefix}/{problem_id}" if prefix else problem_id
        scopes.append(
            (
                scope_id,
                problem_id,
                runs_by_algorithm(
                    ExperimentBundle(runs=subset, experiment_dirs=())
                ),
            )
        )
    return scopes


def _study_run_sets(
    runs: tuple[RunTrajectory, ...] | list[RunTrajectory],
) -> tuple[tuple[str, tuple[RunTrajectory, ...]], ...] | None:
    """Reference, validation, and full collections, or None when one side is absent."""
    material = tuple(runs)
    ids = {run.problem_id for run in material}
    reference_ids = [
        problem_id
        for problem_id in sorted(ids, key=_problem_sort_key)
        if problem_id in _REFERENCE_PROBLEM_IDS
    ]
    other_ids = [
        problem_id
        for problem_id in sorted(ids, key=_problem_sort_key)
        if problem_id not in _REFERENCE_PROBLEM_IDS
    ]
    if not reference_ids or not other_ids:
        return None
    by_id: dict[str, list[RunTrajectory]] = {}
    for run in material:
        by_id.setdefault(run.problem_id, []).append(run)

    def take(problem_ids: list[str]) -> tuple[RunTrajectory, ...]:
        chosen: list[RunTrajectory] = []
        for problem_id in problem_ids:
            chosen.extend(by_id[problem_id])
        return tuple(chosen)

    all_ids = sorted(ids, key=_problem_sort_key)
    return (
        ("reference", take(reference_ids)),
        ("validation", take(other_ids)),
        ("all21", take(all_ids)),
    )


def _instance_set_names(set_id: str, n_problems: int) -> tuple[str, str]:
    """Menu label and the title phrase for one collection of problem instances."""
    kind = {"reference": "reference", "validation": "validation"}.get(set_id, "complete")
    label = f"{kind.capitalize()} set ({n_problems} instances)"
    return label, f"across the {kind} set ({n_problems} instances)"


def _problem_set_groups(
    scopes: tuple[tuple[str, str, dict[str, tuple[RunTrajectory, ...]]], ...],
) -> list[dict[str, object]] | None:
    """Menu groups when scopes are prefixed ``reference/``, ``validation/``, or ``all21/``."""
    if not scopes or "/" not in scopes[0][0]:
        return None
    order: list[str] = []
    buckets: dict[str, list[tuple[str, str]]] = {}
    for scope, label, _grouped in scopes:
        set_id, _rest = scope.split("/", 1)
        if set_id not in buckets:
            order.append(set_id)
            buckets[set_id] = []
        buckets[set_id].append((scope, label))
    groups: list[dict[str, object]] = []
    for set_id in order:
        options = buckets[set_id]
        n_problems = sum(1 for scope, _label in options if not scope.endswith("/all"))
        label, pooled = _instance_set_names(set_id, n_problems)
        groups.append(
            {
                "id": set_id,
                "label": label,
                "pooled": pooled,
                "options": options,
            }
        )
    return groups


def _shown_scope_options(
    scopes: tuple[tuple[str, str, dict[str, tuple[RunTrajectory, ...]]], ...],
    groups: list[dict[str, object]] | None,
) -> list[tuple[str, str]]:
    """Problem menu for the first problem set, or every scope when there is one set."""
    if groups:
        return list(groups[0]["options"])  # type: ignore[arg-type]
    return [(scope, label) for scope, label, _grouped in scopes]


def ecdf_problem_scopes(
    runs: tuple[RunTrajectory, ...] | list[RunTrajectory],
    *,
    split_sets: bool = False,
) -> tuple[tuple[str, str, dict[str, tuple[RunTrajectory, ...]]], ...]:
    """
    Build the All-problems scope and one scope per problem id.

    Parameters
    ----------
    runs : sequence of RunTrajectory
        Completed runs already limited to the plotted algorithms.
    split_sets : bool
        When the runs contain both reference and other instances, also build
        the four-instance, validation, and full collections. Each scope id
        is ``set/problem``.

    Returns
    -------
    tuple
        ``(scope id, menu label, algorithm grouping)``.
    """
    material = tuple(runs)
    if split_sets:
        sets = _study_run_sets(material)
        if sets is not None:
            scopes: list[tuple[str, str, dict[str, tuple[RunTrajectory, ...]]]] = []
            for set_id, subset in sets:
                scopes.extend(_scopes_for_runs(subset, prefix=set_id))
            return tuple(scopes)
    return tuple(_scopes_for_runs(material))


def _ecdf_legend_layout(labels: list[str]) -> dict:
    """Axes pane. Algorithm cards are HTML beside the plot, not a Plotly legend."""
    del labels
    return {
        "showlegend": False,
        "autosize": True,
        "height": 560,
        "margin": {"l": 78, "r": 28, "t": 24, "b": 64},
    }


def relative_median_percent(
    runs: tuple[RunTrajectory, ...] | list[RunTrajectory],
    *,
    extend_short_runs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Median and IQR of best-so-far as a percent of each run's own optimum.

    Parameters
    ----------
    runs : sequence of RunTrajectory
        One algorithm's runs. Each must carry a positive sealed optimum.
    extend_short_runs : bool
        Carry an early-stopped run forward at its last best-so-far value.

    Returns
    -------
    evaluations, median, q25, q75
        Common evaluation grid and pointwise percent-of-optimum curves.

    Raises
    ------
    ValueError
        If a run has no positive sealed optimum, or fewer than two runs.
    """
    if len(runs) < 2:
        raise ValueError("relative median requires at least two runs")
    evaluations, matrix = _aligned_matrix(
        runs, extend_short_runs=extend_short_runs
    )
    for index, run in enumerate(runs):
        optimum = run.global_optimum
        if optimum is None or not np.isfinite(optimum) or float(optimum) <= 0.0:
            raise ValueError(
                f"{run.algorithm_name} run task_index={run.task_index} "
                "has no positive sealed optimum"
            )
        matrix[index, :] = 100.0 * matrix[index, :] / float(optimum)
    return (
        evaluations,
        np.median(matrix, axis=0),
        np.percentile(matrix, 25, axis=0),
        np.percentile(matrix, 75, axis=0),
    )


def relative_percent_curve(
    run: RunTrajectory,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Best-so-far of one run as a percent of that run's sealed optimum.

    Parameters
    ----------
    run : RunTrajectory
        One completed attempt. Must carry a positive sealed optimum.

    Returns
    -------
    evaluations, percent
        Logged evaluation indices and ``100 * best-so-far / y_p*``.

    Raises
    ------
    ValueError
        If the run has no positive sealed optimum.
    """
    optimum = run.global_optimum
    if optimum is None or not np.isfinite(optimum) or float(optimum) <= 0.0:
        raise ValueError(
            f"{run.algorithm_name} run task_index={run.task_index} "
            "has no positive sealed optimum"
        )
    evaluations = np.asarray(run.evaluations, dtype=int)
    percent = 100.0 * np.asarray(run.y_p_best_so_far_all, dtype=float) / float(optimum)
    return evaluations, percent


def downsample_curve(
    evaluations: np.ndarray,
    values: np.ndarray,
    *,
    max_points: int = _RUN_CURVE_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep at most ``max_points`` samples, including the first and last.

    Parameters
    ----------
    evaluations : ndarray
        Evaluation indices.
    values : ndarray
        Curve values aligned with ``evaluations``.
    max_points : int
        Maximum number of samples to retain.

    Returns
    -------
    evaluations, values
        Subsampled arrays. Unchanged when already short enough.
    """
    count = int(evaluations.shape[0])
    if count <= int(max_points):
        return evaluations, values
    indices = np.unique(np.linspace(0, count - 1, int(max_points), dtype=int))
    return evaluations[indices], values[indices]


def _series_color(index: int, count: int) -> str:
    """Distinct hue for series ``index`` of ``count`` (golden-angle steps)."""
    del count
    hue = ((int(index) * 137.508) % 360.0) / 360.0
    lightness = 0.42 if int(index) % 2 == 0 else 0.58
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, 0.70)
    return f"rgb({int(red * 255)}, {int(green * 255)}, {int(blue * 255)})"


def _translucent(rgb: str, alpha: float) -> str:
    return rgb.replace("rgb(", "rgba(").replace(")", f", {alpha})")


def selected_interactive_plots(which: Sequence[str] | None) -> tuple[str, ...]:
    """Plot names to write, in the requested order, with duplicates removed."""
    requested = tuple(which) if which is not None else INTERACTIVE_PLOTS
    if not requested:
        raise ValueError("interactive plots require at least one plot name")
    unknown = [name for name in requested if name not in INTERACTIVE_PLOTS]
    if unknown:
        raise ValueError(
            f"unknown interactive plot name(s) {unknown!r}; "
            f"allowed: {list(INTERACTIVE_PLOTS)}"
        )
    return tuple(dict.fromkeys(requested))


def render_interactive_pool(
    bundle: ExperimentBundle,
    out_dir: Path,
    *,
    percent: float = 1.0,
    extend_short_runs: bool = True,
    exclude_algorithms: tuple[str, ...] | list[str] = (),
    progress: ProgressCallback | None = None,
    downloads: bool = True,
    which: Sequence[str] | None = None,
    percents: Sequence[float] | None = None,
    problem_sets: bool = False,
) -> list[Path]:
    """
    Write hover-only HTML figures pooled across problems and once per problem.

    Parameters
    ----------
    bundle : ExperimentBundle
        One or more experiment directories.
    out_dir : Path
        Directory for the HTML files.
    percent : float
        Tolerance shown when the ECDF opens, as a percent above each run's
        sealed optimum. The default menu offers 1%, 2%, 3%, 4%, 5%, and 10%.
        ``percents`` replaces that menu when only some of those tolerances
        were stored.
    extend_short_runs : bool
        Carry early-stopped convergence curves forward.
    exclude_algorithms : sequence of str
        Algorithm names to omit from every figure written here.
    progress : callable or None
        Optional ``(done, total, message)`` reporter. ``done`` counts HTML
        files started, and ECDF messages name the problem and margin being
        built.
    downloads : bool
        When true, the success, expected-runtime, and heatmap figures offer
        PNG, SVG, and data buttons.
    which : sequence of str or None
        Names from :data:`INTERACTIVE_PLOTS`. ``None`` writes all of them.
        ``convergence`` includes the pooled figure and both per-problem
        figures.
    percents : sequence of float or None
        Target tolerances, in percent, offered by the ECDF, ERT, and
        heatmap. ``None`` uses 1, 2, 3, 4, 5, and 10.

    Returns
    -------
    list of Path
        Written HTML paths.

    Raises
    ------
    ValueError
        If plotly is not installed, a plot name is unknown, no plot is
        requested, a remaining algorithm is unfinished, or a run lacks a
        positive sealed optimum.
    """
    selected = selected_interactive_plots(which)
    bundle = prepare_plot_bundle(bundle, exclude_algorithms)
    needs_plotly = any(name != "heatmap" for name in selected)
    if needs_plotly:
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ValueError(
                "interactive plots require plotly; install with: "
                'pip install -e ".[visualization-interactive]"'
            ) from exc
    else:
        go = None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = runs_by_algorithm(bundle)
    if not grouped:
        raise ValueError("interactive plots require at least one completed run")
    color_index = {name: index for index, name in enumerate(grouped)}
    scopes = _ecdf_scopes(bundle, grouped) if "convergence" in selected else ()
    problem_slices = (
        runs_by_experiment(bundle) if "convergence" in selected else ()
    )
    curve_scopes = (
        ecdf_problem_scopes(bundle.runs, split_sets=problem_sets)
        if "ecdf" in selected or "ert" in selected
        else ()
    )
    n_files = sum(
        1 + 2 * len(problem_slices) if name == "convergence" else 1
        for name in selected
    )
    step = 0

    def _report(message: str) -> None:
        if progress is not None:
            progress(step, n_files, message)

    def _begin(name: str) -> None:
        nonlocal step
        step += 1
        _report(f"writing {name}")

    written: list[Path] = []
    if "heatmap" in selected:
        from core.visualization_core.optimization.plots.success_heatmap import (
            render_success_heatmap,
        )

    def _ecdf_detail(detail: str) -> None:
        _report(f"writing ecdf_success_pooled.html: {detail}")

    def _ert_detail(detail: str) -> None:
        _report(f"writing ert_pooled.html: {detail}")

    requested = float(percent)
    shown = (
        tuple(float(item) for item in _POOLED_ECDF_PERCENTS)
        if percents is None
        else tuple(float(item) for item in percents)
    )
    if requested not in shown:
        shown = tuple(sorted((*shown, requested)))

    for name in selected:
        if name == "convergence":
            _begin("convergence_pooled.html")
            written.append(
                _write_convergence_selectable(
                    go,
                    scopes,
                    out_dir / "convergence_pooled.html",
                    extend_short_runs=extend_short_runs,
                    color_index=color_index,
                )
            )
            for experiment_dir, problem_group in problem_slices:
                problem_dir = out_dir / experiment_dir.name
                problem_dir.mkdir(parents=True, exist_ok=True)
                label = experiment_dir.name
                _begin(f"{experiment_dir.name}/convergence.html")
                written.append(
                    _write_convergence(
                        go,
                        problem_group,
                        problem_dir / "convergence.html",
                        extend_short_runs=extend_short_runs,
                        color_index=color_index,
                        title=(
                            f"Median best-so-far on {label} "
                            "(each run divided by this problem's y_p*)"
                        ),
                    )
                )
                _begin(f"{experiment_dir.name}/convergence_runs.html")
                written.append(
                    _write_convergence_runs(
                        go,
                        problem_group,
                        problem_dir / "convergence_runs.html",
                        label=label,
                    )
                )
        elif name == "ecdf":
            _begin("ecdf_success_pooled.html")
            written.append(
                _write_ecdf_margins(
                    go,
                    curve_scopes,
                    out_dir / "ecdf_success_pooled.html",
                    percents=shown,
                    initial_percent=requested,
                    on_step=_ecdf_detail if progress is not None else None,
                    downloads=downloads,
                )
            )
        elif name == "ert":
            _begin("ert_pooled.html")
            written.append(
                _write_ert(
                    go,
                    curve_scopes,
                    out_dir / "ert_pooled.html",
                    percents=shown,
                    budgets=_BUDGETS,
                    on_step=_ert_detail if progress is not None else None,
                    downloads=downloads,
                )
            )
        else:
            _begin("success_heatmap.html")
            written.extend(
                render_success_heatmap(
                    bundle,
                    out_dir,
                    downloads=downloads,
                    margins=shown,
                )
            )
    return written


def _shown_algorithm(token: str) -> str:
    """Paper name and start, shared by every interactive figure."""
    item = presentation_for(token)
    start = "Warm start" if item.start == "warm" else "Cold start"
    return f"{item.display_name} · {start}"


def _write_convergence(
    go,
    grouped,
    path: Path,
    *,
    extend_short_runs: bool,
    color_index: dict[str, int],
    title: str,
) -> Path:
    figure = go.Figure()
    names = list(grouped)
    for name in names:
        evaluations, median, q25, q75 = relative_median_percent(
            grouped[name], extend_short_runs=extend_short_runs
        )
        series_index = color_index[name]
        color = _series_color(series_index, len(color_index))
        dash = _DASHES[series_index % len(_DASHES)]
        n_runs = len(grouped[name])
        shown = _shown_algorithm(name)
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([evaluations, evaluations[::-1]]),
                y=np.concatenate([q75, q25[::-1]]),
                fill="toself",
                fillcolor=_translucent(color, 0.15),
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=evaluations,
                y=median,
                mode="lines",
                name=shown,
                line={"color": color, "width": 2, "dash": dash},
                hovertemplate=(
                    f"{shown}<br>median of {n_runs} runs"
                    "<br>evaluation %{x}"
                    "<br>%{y:.1f}% of y_p*"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="evaluations",
        yaxis_title="best-so-far / y_p* [%]",
        showlegend=False,
        hovermode="closest",
        template="plotly_white",
    )
    figure.write_html(path, include_plotlyjs=True)
    _append_local_search_note(path, names)
    return path


def selected_view_mask(
    keys: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    scope: str,
    view: str,
) -> list[bool]:
    """
    Visibility of convergence traces for one problem and one display choice.

    Parameters
    ----------
    keys : sequence of (scope, view)
        One entry per trace, in trace order. ``view`` is ``median`` or an
        algorithm name.
    scope : str
        Selected scope id. ``all`` is every loaded problem.
    view : str
        ``median`` for every algorithm's median, or one algorithm name for
        that algorithm's individual runs.

    Returns
    -------
    list of bool
        ``True`` where both the scope and the view match.
    """
    return [
        item_scope == scope and item_view == view
        for item_scope, item_view in keys
    ]


def _downsample_band(
    evaluations: np.ndarray,
    median: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep the same samples of a median curve and its quartile band."""
    count = int(evaluations.shape[0])
    if count <= _RUN_CURVE_POINTS:
        return evaluations, median, q25, q75
    indices = np.unique(np.linspace(0, count - 1, _RUN_CURVE_POINTS, dtype=int))
    return evaluations[indices], median[indices], q25[indices], q75[indices]


def _convergence_view_title(scope: str, label: str, view: str, n_runs: int) -> str:
    if view == _MEDIAN_VIEW:
        if scope == _ALL_PROBLEMS:
            return (
                "Median best-so-far across all loaded problems "
                "(each run divided by its own y_p*)"
            )
        return (
            f"Median best-so-far on {label} "
            "(each run divided by this problem's y_p*)"
        )
    if scope == _ALL_PROBLEMS:
        return (
            f"{view} across all problems: {n_runs} runs "
            "(each divided by its own y_p*)"
        )
    return (
        f"{view} on {label}: {n_runs} runs "
        "(each divided by this problem's y_p*)"
    )


def _write_convergence_selectable(
    go,
    scopes: tuple[tuple[str, str, dict[str, tuple[RunTrajectory, ...]]], ...],
    path: Path,
    *,
    extend_short_runs: bool,
    color_index: dict[str, int],
) -> Path:
    """Median curves, or one algorithm's runs, for all problems or one problem."""
    figure = go.Figure()
    names = list(color_index)
    keys: list[tuple[str, str]] = []
    titles: dict[str, str] = {}
    first_scope = scopes[0][0]
    for scope, label, grouped in scopes:
        titles[f"{scope}|{_MEDIAN_VIEW}"] = _convergence_view_title(
            scope, label, _MEDIAN_VIEW, 0
        )
        for name in names:
            runs = grouped.get(name, ())
            if len(runs) < 2:
                continue
            _add_median_traces(
                go,
                figure,
                name,
                runs,
                extend_short_runs=extend_short_runs,
                color_index=color_index,
                visible=(scope == first_scope),
            )
            keys.extend([(scope, _MEDIAN_VIEW), (scope, _MEDIAN_VIEW)])
        for name in names:
            runs = tuple(
                sorted(
                    grouped.get(name, ()),
                    key=lambda run: (run.repetition, run.task_index),
                )
            )
            if not runs:
                continue
            titles[f"{scope}|{name}"] = _convergence_view_title(
                scope, label, _shown_algorithm(name), len(runs)
            )
            for run_index, run in enumerate(runs):
                _add_run_trace(
                    go,
                    figure,
                    run,
                    color=_series_color(run_index, max(len(runs), 1)),
                    visible=False,
                )
                keys.append((scope, name))
    figure.update_layout(
        title=titles[f"{first_scope}|{_MEDIAN_VIEW}"],
        xaxis_title="evaluations",
        yaxis_title="best-so-far / y_p* [%]",
        showlegend=False,
        hovermode="closest",
        template="plotly_white",
    )
    scope_options = [(scope, menu_label) for scope, menu_label, _grouped in scopes]
    view_options = [(_MEDIAN_VIEW, "all algorithms (median)")]
    view_options.extend((name, _shown_algorithm(name)) for name in names)
    html_doc = figure.to_html(
        full_html=True,
        include_plotlyjs=True,
        div_id="convergence-pooled",
    )
    path.write_text(
        _inject_convergence_menus(
            html_doc,
            scope_options=scope_options,
            view_options=view_options,
            keys=keys,
            titles=titles,
        ),
        encoding="utf-8",
    )
    return path


def _add_median_traces(
    go,
    figure,
    name: str,
    runs: tuple[RunTrajectory, ...],
    *,
    extend_short_runs: bool,
    color_index: dict[str, int],
    visible: bool,
) -> None:
    evaluations, median, q25, q75 = _downsample_band(
        *relative_median_percent(runs, extend_short_runs=extend_short_runs)
    )
    series_index = color_index[name]
    color = _series_color(series_index, len(color_index))
    dash = _DASHES[series_index % len(_DASHES)]
    n_runs = len(runs)
    shown = _shown_algorithm(name)
    figure.add_trace(
        go.Scatter(
            x=np.concatenate([evaluations, evaluations[::-1]]),
            y=np.concatenate([q75, q25[::-1]]),
            fill="toself",
            fillcolor=_translucent(color, 0.15),
            line={"width": 0},
            hoverinfo="skip",
            visible=visible,
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=evaluations,
            y=median,
            mode="lines",
            name=shown,
            visible=visible,
            line={"color": color, "width": 2, "dash": dash},
            hovertemplate=(
                f"{shown}<br>median of {n_runs} runs"
                "<br>evaluation %{x}"
                "<br>%{y:.1f}% of y_p*"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )


def _add_run_trace(go, figure, run: RunTrajectory, *, color: str, visible: bool) -> None:
    evaluations, percent = downsample_curve(*relative_percent_curve(run))
    problem = problem_label(run.experiment_dir)
    shown = _shown_algorithm(run.algorithm_name)
    figure.add_trace(
        go.Scatter(
            x=evaluations,
            y=percent,
            mode="lines",
            name=shown,
            visible=visible,
            line={"color": color, "width": 1.5},
            hovertemplate=(
                f"{shown}<br>{problem}<br>repetition {run.repetition}"
                "<br>evaluation %{x}"
                "<br>%{y:.1f}% of y_p*"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )


def _inject_convergence_menus(
    html_doc: str,
    *,
    scope_options: list[tuple[str, str]],
    view_options: list[tuple[str, str]],
    keys: list[tuple[str, str]],
    titles: dict[str, str],
) -> str:
    """Add problem and algorithm menus that show one matching set of traces."""
    scope_html = "\n".join(
        f'<option value="{html.escape(scope)}">{html.escape(label)}</option>'
        for scope, label in scope_options
    )
    view_html = "\n".join(
        f'<option value="{html.escape(view)}">{html.escape(label)}</option>'
        for view, label in view_options
    )
    payload = json.dumps(
        {
            "keys": [{"scope": scope, "view": view} for scope, view in keys],
            "titles": titles,
        }
    )
    note = local_search_caption(
        _labels_for_names(view for view, _label in view_options)
    )
    note_html = ""
    if note:
        note_html = (
            '<p style="font-family: sans-serif; font-size: 13px; '
            "line-height: 1.45; margin: 8px 0 0 40px; max-width: 920px;\">"
            f"<b>Local search.</b> {html.escape(note)}</p>"
        )
    controls = f"""
<div style="font-family: sans-serif; margin: 12px 0 0 40px;">
<label>Problem <select id="conv-scope">
{scope_html}
</select></label>
<label style="margin-left: 16px;">Algorithms <select id="conv-view">
{view_html}
</select></label>
</div>
{note_html}
"""
    script = f"""
<script>
(function () {{
  var payload = {payload};
  function apply() {{
    var gd = document.getElementById("convergence-pooled");
    var scope = document.getElementById("conv-scope").value;
    var view = document.getElementById("conv-view").value;
    var visible = payload.keys.map(function (item) {{
      return item.scope === scope && item.view === view;
    }});
    var traces = visible.map(function (_, index) {{ return index; }});
    Plotly.restyle(gd, {{visible: visible}}, traces);
    Plotly.relayout(gd, {{"title.text": payload.titles[scope + "|" + view]}});
  }}
  document.getElementById("conv-scope").addEventListener("change", apply);
  document.getElementById("conv-view").addEventListener("change", apply);
}})();
</script>
"""
    marker = '<div id="convergence-pooled"'
    if marker not in html_doc:
        raise ValueError("interactive convergence html is missing the plot div")
    html_doc = html_doc.replace(marker, controls + marker, 1)
    if "</body>" in html_doc:
        return html_doc.replace("</body>", script + "</body>", 1)
    return html_doc + script


def _write_convergence_runs(go, grouped, path: Path, *, label: str) -> Path:
    """One dropdown: the selected algorithm's individual runs on this problem."""
    figure = go.Figure()
    names = list(grouped)
    trace_counts: list[tuple[str, int]] = []
    for name in names:
        runs = tuple(
            sorted(grouped[name], key=lambda run: (run.repetition, run.task_index))
        )
        shown = _shown_algorithm(name)
        trace_counts.append((name, shown, len(runs)))
        for run_index, run in enumerate(runs):
            evaluations, percent = downsample_curve(*relative_percent_curve(run))
            color = _series_color(run_index, max(len(runs), 1))
            figure.add_trace(
                go.Scatter(
                    x=evaluations,
                    y=percent,
                    mode="lines",
                    name=shown,
                    visible=(name == names[0]),
                    line={"color": color, "width": 1.5},
                    hovertemplate=(
                        f"{shown}<br>repetition {run.repetition}"
                        "<br>evaluation %{x}"
                        "<br>%{y:.1f}% of y_p*"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
    total = sum(count for _name, _shown, count in trace_counts)
    buttons = []
    cursor = 0
    for _name, shown, count in trace_counts:
        visible = [False] * total
        visible[cursor : cursor + count] = [True] * count
        cursor += count
        buttons.append(
            {
                "label": shown,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": (
                            f"{shown} on {label}: {count} runs "
                            "(each divided by this problem's y_p*)"
                        )
                    },
                ],
            }
        )
    _first_name, first_shown, first_count = trace_counts[0]
    figure.update_layout(
        title=(
            f"{first_shown} on {label}: {first_count} runs "
            "(each divided by this problem's y_p*)"
        ),
        xaxis_title="evaluations",
        yaxis_title="best-so-far / y_p* [%]",
        showlegend=False,
        hovermode="closest",
        template="plotly_white",
        margin={"t": 80},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
    )
    figure.write_html(path, include_plotlyjs=True)
    _append_local_search_note(path, names)
    return path


def problem_label(experiment_dir: Path) -> str:
    """
    Short problem name taken from an experiment directory.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root, for example ``study_ws_cmp_p1_L6750_B2_r20``.

    Returns
    -------
    str
        The ``L####_B#`` token when the directory name contains one,
        otherwise an ``L####`` span token, otherwise the directory name.
    """
    match = _PROBLEM_ID.search(Path(experiment_dir).name)
    if match is not None:
        return match.group(0)
    span = _SPAN_ID.search(Path(experiment_dir).name)
    if span is not None:
        return span.group(0)
    return Path(experiment_dir).name


def selected_trace_mask(
    keys: list[tuple[str, float]] | tuple[tuple[str, float], ...],
    scope: str,
    percent: float,
) -> list[bool]:
    """
    Visibility of ECDF traces for one problem scope and one margin.

    Parameters
    ----------
    keys : sequence of (scope, percent)
        One entry per trace, in trace order.
    scope : str
        Selected scope id. ``all`` is every loaded problem.
    percent : float
        Selected margin.

    Returns
    -------
    list of bool
        ``True`` where both the scope and the margin match.
    """
    return [
        item_scope == scope and float(item_percent) == float(percent)
        for item_scope, item_percent in keys
    ]


def _ecdf_scopes(
    bundle: ExperimentBundle,
    grouped: dict[str, tuple[RunTrajectory, ...]],
) -> tuple[tuple[str, str, dict[str, tuple[RunTrajectory, ...]]], ...]:
    """Scope id, menu label, and runs. The first scope is every problem."""
    scopes: list[tuple[str, str, dict[str, tuple[RunTrajectory, ...]]]] = [
        (_ALL_PROBLEMS, "all problems", grouped)
    ]
    for experiment_dir, problem_group in runs_by_experiment(bundle):
        label = problem_label(experiment_dir)
        scopes.append((label, label, problem_group))
    return tuple(scopes)


def _ecdf_heading(
    scope: str,
    label: str,
    percent: float,
    mode: str = "algorithm",
    budget: int = _FULL_BUDGET,
) -> str:
    """Title above the menus: what is drawn, where, the tolerance, and the budget."""
    if _is_pooled_scope(scope):
        where = "across optimisation problems"
    else:
        where = f"on {html_escape(label)}"
    subject = {
        "algorithm": "algorithms",
        "class": "algorithm classes",
        "start": "cold and warm start algorithm variants",
    }.get(mode, "algorithms")
    return (
        f"{_ECDF_TITLE} of {subject} {where}"
        '<span class="ecdf-meta">'
        '<span class="ecdf-tolerance">relative target tolerance '
        f"ε<sub>X</sub> = {percent:g}%</span>"
        f'<span class="ecdf-budget">{int(budget):,} function evaluations</span>'
        "</span>"
    )


def local_search_caption(labels: object) -> str:
    """Explain the local search when one of its algorithms is on the figure."""
    present = {
        str(label)
        for label in labels
        if str(label) in {"Strict improvement", "Equal descent"}
    }
    if not present:
        return ""
    return (
        "Population Basin Hopping, Adaptive Multi-Start, Discrete "
        "Multi-Level Single Linkage, and SOBOL + local descent use the L1 "
        "neighbourhood: one catalogue "
        "index changes by one step. Strict improvement moves only to a "
        "strictly better neighbour. An equal neighbour is evaluated and is "
        "not a move. Equal descent also searches from equal neighbours and "
        "takes the best strict improvement on that plateau."
    )


def _local_search_item(labels: object) -> str:
    """Return the definitions-list entry, or an empty string."""
    text = local_search_caption(labels)
    if not text:
        return ""
    return f"<li><b>Local search.</b> {html.escape(text)}</li>"


def _curve_local_searches(curves: object) -> tuple[str, ...]:
    """Local-search captions stored on interactive curve records."""
    found: list[str] = []
    for curve in curves:
        if not isinstance(curve, dict):
            continue
        meta = curve.get("meta")
        if not isinstance(meta, dict):
            continue
        label = str(meta.get("local_search") or "")
        if label and label not in found:
            found.append(label)
    return tuple(found)


def _labels_for_names(names: object) -> tuple[str, ...]:
    """Local-search captions for algorithm tokens, skipping unknown names."""
    found: list[str] = []
    for name in names:
        try:
            label = local_search_label(presentation_for(str(name)))
        except AlgorithmPresentationError:
            continue
        if label and label not in found:
            found.append(label)
    return tuple(found)


def _append_local_search_note(path: Path, names: object) -> None:
    """Add the local-search paragraph under a written convergence figure."""
    note = local_search_caption(_labels_for_names(names))
    if not note:
        return
    html_doc = path.read_text(encoding="utf-8")
    paragraph = (
        '<p style="font-family: sans-serif; font-size: 13px; '
        "line-height: 1.45; margin: 12px 40px; max-width: 920px;\">"
        f"<b>Local search.</b> {html.escape(note)}</p>"
    )
    if "</body>" in html_doc:
        html_doc = html_doc.replace("</body>", paragraph + "</body>", 1)
    else:
        html_doc += paragraph
    path.write_text(html_doc, encoding="utf-8")


def _target_note() -> str:
    """Shared wording for y_p* under the tolerance formula."""
    return (
        "where y<sub>p</sub>* is the minimum stored penalised objective "
        "for that problem instance. A target hit need not satisfy all constraints."
    )


def _rank_sentence() -> str:
    """Shared family-order rule. Each figure says which selection updates it."""
    return (
        "Algorithm families are ranked by the higher overall success rate of "
        "their cold-start and warm-start variants, then by the earliest "
        "evaluation at which that rate is reached, and finally alphabetically."
    )


_PREDESIGN_EVALUATIONS = 2700

_PREDESIGN_SENTENCE = (
    "Predesign evaluations used to construct warm starts are excluded "
    "from the evaluation counts."
)

_PUBLICATION = (
    '<p style="margin:14px 0 4px;">The results in this interactive graphic are '
    "part of the following publications:</p>"
    '<p style="margin:0;"><i>Optimization methods for structural design problems '
    "with small feasible regions: A case study of hp shell slabs</i><br>"
    "Dombrowski M, Melcer E, Merz P, Schlaich M, Wortmann T</p>"
)

_HANSEN_REFERENCE = (
    '<p style="margin:14px 0 4px; font-weight:700; color:#222;">References</p>'
    '<p style="margin:0;">[1] Hansen, N., Auger, A., Finck, S., &amp; Ros, R. '
    "(2009). <i>Real-Parameter Black-Box Optimization Benchmarking 2009: "
    "Experimental Setup</i>. INRIA Research Report RR-6828. "
    '<a href="https://hal.inria.fr/inria-00362649/en">'
    "https://hal.inria.fr/inria-00362649/en</a></p>"
)


def _problem_coverage(
    n_problems: int,
    *,
    with_sets: bool,
    mention_menu: bool = True,
) -> str:
    """Which runs the figure is describing, plus the span and load factorial."""
    if with_sets:
        shown = (
            "the set of problem instances selected above, or one problem "
            "instance from that set"
        )
    elif n_problems == 1:
        shown = "the problem instance in this figure"
    elif n_problems > 1:
        shown = f"all {n_problems} problem instances or a selected problem instance"
    else:
        shown = "the runs in this figure"
    menu = ""
    if with_sets and mention_menu:
        menu = (
            " The Set of problem instances menu selects the Reference set "
            "(4 instances), the Validation set (17 instances), or the Complete "
            "set (21 instances). The algorithms and runs are those recorded "
            "for the problem instances in the selected set."
        )
    return f"Curves represent results from {shown}. {_PROBLEM_FACTORIAL}{menu}"


def _ecdf_description(
    n_problems: int,
    local_search_labels: object = (),
    *,
    with_sets: bool = False,
) -> str:
    """Definitions under the plot."""
    coverage = _problem_coverage(
        n_problems, with_sets=with_sets, mention_menu=False
    )
    return (
        '<p style="margin:0 0 6px; font-weight:700; color:#222;">Definitions</p>'
        "<p style=\"margin:0 0 8px;\">The vertical axis is the cumulative "
        "success rate: the percentage of completed runs that have reached the "
        "selected relative target tolerance ε<sub>X</sub> by the evaluation on "
        "the horizontal axis. The benchmarking literature calls this curve an "
        "Empirical Cumulative Distribution Function (ECDF) of the runtime [1]. "
        "That name is kept for the method. It is not used as the axis title, "
        "because runs that miss the target stay in the total, so the curve "
        "ends at the success rate of the selected runs rather than at 100% "
        "unless every run succeeds. The curve describes those runs. It is not "
        "an estimate for a larger population of problems.</p>"
        '<ul style="margin:0; padding-left:18px;">'
        "<li><b>Cumulative success rate.</b> The proportion of completed "
        "runs that first reach the target at or before a given number of "
        "function evaluations [1]. Runs that never reach the target remain in the "
        "total. The plot shows this proportion as a percentage.</li>"
        "<li><b>Relative target tolerance.</b> A run reaches the target when "
        "its best penalised objective satisfies"
        '<p class="ecdf-formula">y<sub>p,best</sub> ≤ (1 + ε<sub>X</sub>) '
        "y<sub>p</sub>*,</p>"
        f'<p class="formula-note">{_target_note()}</p></li>'
        "<li><b>Successful run.</b> A completed run that first reaches the "
        "relative target tolerance at or before the selected evaluation "
        "budget. Its first-hit evaluation determines when it contributes to "
        "the cumulative success rate.</li>"
        f"<li><b>Problems.</b> {coverage} Pooled cumulative success "
        "rates count every included run.</li>"
        "<li><b>Algorithms.</b> Each algorithm has a cold-start and a "
        "warm-start variant. In the algorithm comparison they share a colour, "
        "with a solid line for the cold start and a dashed line for the warm "
        f"start.</li>{_local_search_item(local_search_labels)}"
        "<li><b>Function evaluation.</b> The algorithm submits a candidate "
        "design to the analysis model and receives objective and constraint "
        "values. Depending on the algorithm, optimisation uses the penalised "
        f"or unpenalised objective and constraint information. {_PREDESIGN_SENTENCE}</li>"
        "<li><b>Evaluation budget.</b> The maximum number of function "
        "evaluations available to a run to reach the target. The selected "
        "budget determines the horizontal evaluation range and the success "
        "rates reported in the legend. The evaluation count in each legend "
        "cell is the last target hit within that budget.</li>"
        "<li><b>Deployment-cost reading.</b> The curves in this figure use "
        "solver evaluations. The solver budget is 15,000 function evaluations. "
        "A separate reading shifts each warm-start first-hit evaluation by "
        "2,700 predesign evaluations, so a warm-start deployment cost can "
        "reach 17,700. The interactive curves are not shifted. The solver "
        "budget and the deployment total are different quantities.</li>"
        "<li><b>Legend ordering.</b> "
        f"{_rank_sentence()} The order follows the selected set of problem instances, "
        "problem instance, relative target tolerance, and evaluation budget. Cold start "
        "stays before warm start.</li>"
        "</ul>"
        f"{_HANSEN_REFERENCE}{_PUBLICATION}"
    )


# Plotly Dark24, in palette order. The first ranks keep the first colours.
_DARK24: tuple[str, ...] = (
    "#2E91E5",
    "#E15F99",
    "#1CA71C",
    "#FB0D0D",
    "#DA16FF",
    "#222A2A",
    "#B68100",
    "#750D86",
    "#EB663B",
    "#511CFB",
    "#00A08B",
    "#FB00D1",
    "#FC0080",
    "#B2828D",
    "#6C7C32",
    "#778AAE",
    "#862A16",
    "#A777F1",
    "#620042",
    "#1616A7",
    "#DA60CA",
    "#6C4516",
    "#0D2A63",
    "#AF0038",
)


def _dark24_colors(count: int) -> list[str]:
    """Return the first ``count`` Dark24 colours."""
    if count < 0:
        raise ValueError(f"colour count must be non-negative, got {count}")
    if count > len(_DARK24):
        raise ValueError(f"Dark24 has {len(_DARK24)} colours, asked for {count}")
    return list(_DARK24[:count])


def _comparison_palette(order: list[str]) -> dict[str, dict[str, str]]:
    """Map each token to Dark24 colours, one prefix per comparison."""
    families: list[str] = []
    classes: list[str] = []
    starts: list[str] = []
    items = []
    for name in order:
        item = presentation_for(name)
        items.append((name, item))
        if item.family not in families:
            families.append(item.family)
        if item.algorithm_class not in classes:
            classes.append(item.algorithm_class)
        if item.start not in starts:
            starts.append(item.start)
    rank = {label: index for index, label in enumerate(CLASS_ORDER)}
    classes.sort(key=lambda label: (rank.get(label, len(CLASS_ORDER)), label))
    starts = [start for start in ("cold", "warm") if start in starts]
    family_color = dict(zip(families, _dark24_colors(len(families))))
    class_color = dict(zip(classes, _dark24_colors(len(classes))))
    start_color = dict(zip(starts, _dark24_colors(len(starts))))
    return {
        name: {
            "algorithm": family_color[item.family],
            "class": class_color[item.algorithm_class],
            "start": start_color[item.start],
        }
        for name, item in items
    }


def _axis_style(title: str) -> dict:
    return {
        "title": {
            "text": title,
            "font": {"family": _FONT, "size": 16, "color": "#222222"},
            "standoff": 14,
        },
        "showgrid": True,
        "gridcolor": "#ececec",
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linewidth": 1,
        "linecolor": "#444444",
        "mirror": False,
        "ticks": "outside",
        "tickfont": {"family": _FONT, "size": 13, "color": "#222222"},
    }


def _write_ecdf_margins(
    go,
    scopes: tuple[tuple[str, str, dict[str, tuple[RunTrajectory, ...]]], ...],
    path: Path,
    *,
    percents: tuple[float, ...],
    initial_percent: float | None = None,
    on_step: Callable[[str], None] | None = None,
    downloads: bool = True,
) -> Path:
    """The one interactive ECDF: problem, tolerance, and comparison menus."""
    figure = go.Figure()
    curves: list[dict[str, object]] = []
    sizing_labels: list[str] = []
    first_scope, _first_label, first_group = scopes[0]
    shown = float(percents[0] if initial_percent is None else initial_percent)
    anchor_percent = 1.0 if any(float(item) == 1.0 for item in percents) else shown
    palette_order = ecdf_algorithm_order(first_group, anchor_percent)
    known = set(palette_order)
    for _scope, _label, grouped in scopes:
        for name in grouped:
            if name not in known:
                palette_order.append(name)
                known.add(name)
    palette = _comparison_palette(palette_order)
    present_classes = _present_classes(palette_order)
    for scope, label, grouped in scopes:
        crowded = qualifiers_needed(tuple(grouped))
        for percent in percents:
            if on_step is not None:
                on_step(f"{label}, {float(percent):g}%")
            view_labels: list[str] = []
            for name in palette_order:
                if name not in grouped:
                    continue
                record = _ecdf_curve_record(
                        name,
                        grouped[name],
                    scope=scope,
                        percent=float(percent),
                    colors=palette[name],
                    show_qualifier=name in crowded,
                )
                curves.append(record)
                item = presentation_for(name)
                view_labels.append(f"{item.display_name} · {start_label(item)}")
                if scope == first_scope and float(percent) == shown:
                    _add_ecdf_trace(go, figure, record)
            if len(view_labels) > len(sizing_labels):
                sizing_labels = view_labels
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "symbol": "x",
                "size": 12,
                "color": "#111111",
                "line": {"width": 0.5, "color": "#111111"},
            },
            hoverinfo="skip",
            showlegend=False,
            visible=False,
            meta={"cursor": True},
            name="",
        )
    )
    xaxis = _axis_style("Function evaluations [FE]")
    xaxis.update(
        {
            "tickmode": "linear",
            "tick0": 2000,
            "dtick": 2000,
            "tickformat": "d",
            "exponentformat": "none",
            "range": [0, _FULL_BUDGET],
            "autorange": False,
        }
                )
    figure.update_layout(
        shapes=[
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": _FULL_BUDGET,
                "x1": _FULL_BUDGET,
                "y0": 0,
                "y1": 100,
                "line": {"color": "#444444", "width": 1, "dash": "dot"},
            }
        ],
        font={"family": _FONT, "size": 13, "color": "#222222"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=xaxis,
        yaxis=_axis_style("Cumulative success rate [%]"),
        hovermode="closest",
        **_ecdf_legend_layout(sizing_labels),
    )
    figure.update_yaxes(range=[0, 100])
    if on_step is not None:
        on_step("HTML file")
    groups = _problem_set_groups(scopes)
    scope_options = _shown_scope_options(scopes, groups)
    html = figure.to_html(
        full_html=True,
        include_plotlyjs=True,
        div_id="ecdf-pooled",
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    path.write_text(
        _inject_ecdf_menus(
            html,
            scope_options=scope_options,
            percents=tuple(float(percent) for percent in percents),
            initial_percent=shown,
            curves=curves,
            algorithm_classes=present_classes,
            downloads=downloads,
            problem_sets=groups,
        ),
        encoding="utf-8",
    )
    return path


def _present_classes(names: list[str]) -> tuple[str, ...]:
    """Return study classes that appear, in the overview-table order."""
    present: list[str] = []
    for name in names:
        label = presentation_for(name).algorithm_class
        if label not in present:
            present.append(label)
    rank = {label: index for index, label in enumerate(CLASS_ORDER)}
    present.sort(key=lambda label: (rank.get(label, len(CLASS_ORDER)), label))
    return tuple(present)


def _dejavu_face() -> str:
    """Embed the Matplotlib DejaVu faces so the figure matches the static plots."""
    import base64
    import matplotlib

    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    faces: list[str] = []
    for weight, filename in ((400, "DejaVuSans.ttf"), (700, "DejaVuSans-Bold.ttf")):
        payload = base64.b64encode((font_dir / filename).read_bytes()).decode("ascii")
        faces.append(
            "@font-face{"
            "font-family:'DejaVu Sans';"
            f"src:url(data:font/ttf;base64,{payload}) format('truetype');"
            f"font-weight:{weight};font-style:normal;}}"
        )
    return "".join(faces)


def _problem_set_control(
    groups: list[dict[str, object]] | None,
    select_id: str,
) -> tuple[str, str]:
    """First menu, and the JSON the page uses to refill the problem menu."""
    if not groups:
        return "", "[]"
    options = "\n".join(
        f'<option value="{html_escape(str(group["id"]))}">'
        f'{html_escape(str(group["label"]))}</option>'
        for group in groups
    )
    menu = (
        f'<label>Set of problem instances <select id="{select_id}">\n'
        f"{options}\n</select></label>\n"
    )
    payload = [
        {
            "id": group["id"],
            "pooled": group["pooled"],
            "options": [
                {"value": value, "label": label}
                for value, label in group["options"]  # type: ignore[union-attr]
            ],
        }
        for group in groups
    ]
    return menu, json.dumps(payload)


def _inject_ecdf_menus(
    html: str,
    *,
    scope_options: list[tuple[str, str]],
    percents: tuple[float, ...],
    curves: list[dict[str, object]] | tuple[dict[str, object], ...] = (),
    algorithm_classes: tuple[str, ...] = (),
    initial_percent: float | None = None,
    downloads: bool = True,
    problem_sets: list[dict[str, object]] | None = None,
) -> str:
    """Add menus, algorithm cards, the caption, and the hover card."""
    scope_html = "\n".join(
        f'<option value="{html_escape(scope)}">{html_escape(label)}</option>'
        for scope, label in scope_options
    )
    shown = float(percents[0] if initial_percent is None else initial_percent)
    budget_html = "\n".join(
        (
            f'<option value="{budget}"'
            f'{" selected" if budget == _FULL_BUDGET else ""}>'
            f"{budget:,} FE</option>"
        )
        for budget in _BUDGETS
    )
    algorithm_html = "\n".join(
        [
            '<option value="all">All algorithms</option>',
            '<option value="top:3">Top 3</option>',
            '<option value="top:5">Top 5</option>',
            '<option value="top:10">Top 10</option>',
        ]
        + [
            f'<option value="class:{html_escape(label)}">{html_escape(label)}</option>'
            for label in algorithm_classes
        ]
    )
    margin_html = "\n".join(
        (
            f'<option value="{percent:g}"'
            f'{" selected" if float(percent) == shown else ""}>'
            f"{percent:g}%</option>"
        )
        for percent in percents
    )
    payload = json.dumps(
        {
            "curves": list(curves),
            "classes": list(CLASS_ORDER),
        }
    )
    n_problems = sum(
        1 for scope, _label in scope_options if not _is_pooled_scope(scope)
    )
    description = _ecdf_description(
        n_problems,
        _curve_local_searches(curves),
        with_sets=bool(problem_sets),
    )
    set_menu, set_json = _problem_set_control(problem_sets, "ecdf-set")
    comparison_style = ' style="margin-left: 16px;"' if set_menu else ""
    face = _dejavu_face()
    heading = _ecdf_heading(scope_options[0][0], scope_options[0][1], shown)
    export_buttons = ""
    export_script = ""
    if downloads:
        export_buttons = (
            '<button type="button" id="ecdf-png">PNG</button>\n'
            '<button type="button" id="ecdf-svg">SVG</button>\n'
            '<button type="button" id="ecdf-hits">CSV</button>'
        )
        export_script = """
  document.getElementById("ecdf-png").addEventListener("click", function () {
    downloadFigure("png");
  });
  document.getElementById("ecdf-svg").addEventListener("click", function () {
    downloadFigure("svg");
  });
  document.getElementById("ecdf-hits").addEventListener("click", function () {
    downloadText("ecdf_first_hits.csv", firstHitCsv(), "text/csv");
  });"""
    controls = f"""
<style>
{face}
#ecdf-title, #ecdf-menus, #ecdf-menus label, #ecdf-menus select, #ecdf-reset,
#ecdf-caption, #ecdf-shell, #ecdf-tip, #ecdf-pooled text {{
  font-family:'DejaVu Sans', Arial, Helvetica, sans-serif;
}}
#ecdf-title {{font-size:18px; font-weight:700; line-height:1.35; margin:8px 4px 10px; color:#222;}}
#ecdf-title .ecdf-meta {{display:block; margin-top:2px; font-size:15px; font-weight:400; line-height:1.35;}}
#ecdf-title .ecdf-tolerance, #ecdf-title .ecdf-budget {{white-space:nowrap;}}
#ecdf-title .ecdf-budget {{margin-left:0.75em;}}
#ecdf-shell {{display:flex; align-items:flex-start; gap:20px; color:#222;}}
#ecdf-main {{flex:1 1 auto; min-width:640px; overflow:hidden;}}
#ecdf-menus {{font-size:13px; margin:8px 0 4px 4px; color:#222;}}
#ecdf-menus select, #ecdf-reset, #ecdf-png, #ecdf-svg, #ecdf-hits {{font-size:13px; color:#222; background:#fff;}}
#ecdf-reset, #ecdf-png, #ecdf-svg, #ecdf-hits {{margin-left:16px; border:1px solid #bbb; border-radius:2px; padding:1px 8px; cursor:pointer;}}
#ecdf-caption {{font-size:13px; line-height:1.45; margin:12px 0 16px; width:100%; color:#222;}}
#ecdf-caption .ecdf-formula {{margin:8px 0 6px;}}
#ecdf-caption .formula-note {{margin:0 0 2px;}}
#ecdf-caption .frac {{display:inline-block; vertical-align:middle; text-align:center; margin:0 4px;}}
#ecdf-caption .frac .num {{display:block; border-bottom:1px solid #222; padding:0 8px 1px;}}
#ecdf-caption .frac .den {{display:block; padding-top:1px;}}
#ecdf-caption .sum {{display:inline-grid; grid-template-rows:auto auto auto; justify-items:center; vertical-align:middle; margin:0 0.2em; line-height:1; row-gap:0.04em;}}
#ecdf-caption .sum-op {{font-size:1.55em; line-height:0.72;}}
#ecdf-caption .sum-lim {{display:inline-flex; align-items:flex-start; font-size:0.72em; line-height:1; white-space:nowrap;}}
#ecdf-caption .sum .sub {{font-size:0.7em; line-height:1; margin-top:0.45em;}}
#ecdf-legend {{width:520px; max-height:640px; overflow-y:auto; flex:0 0 520px; background:#fff; position:relative; z-index:2;}}
@media (max-width: 1100px) {{
  #ecdf-shell {{flex-direction: column;}}
  #ecdf-main {{min-width:0; width:100%;}}
  #ecdf-legend {{width:100%; flex:1 1 auto; max-height:420px;}}
}}
#ecdf-pooled {{width:100% !important; height:560px !important;}}
#ecdf-pooled .hoverlayer {{display:none !important;}}
.ecdf-compact .head, .ecdf-row {{display:grid; grid-template-columns:minmax(0,1fr) 92px 92px; column-gap:8px; align-items:start;}}
.ecdf-compact .head .subhead {{display:block; font-weight:400; color:#666; font-size:10px; line-height:1.2;}}
.ecdf-compact .head {{color:#666; font-size:11px; margin:0 0 6px;}}
.ecdf-row {{padding:4px 0; font-size:12px;}}
.ecdf-row .name {{min-width:0;}}
.ecdf-row .label {{display:inline-block; vertical-align:top; max-width:calc(100% - 16px);}}
.ecdf-row .handling {{display:block; color:#666; font-size:11px;}}
.dot {{width:8px; height:8px; border-radius:50%; display:inline-block; margin:0 6px 1px 0; vertical-align:middle;}}
.ecdf-row .pct, .ecdf-key .pct {{font-weight:700;}}
.fe {{color:#555; font-size:11px;}}
.miss {{color:#888;}}
.ecdf-key .head {{color:#666; font-size:11px; margin:0 0 6px;}}
.ecdf-key-item {{display:flex; align-items:center; gap:8px; font-size:12px; padding:3px 0;}}
.ecdf-style {{display:flex; gap:18px; margin:0 0 10px; padding-bottom:8px; border-bottom:1px solid #ececec;}}
.ecdf-style-item {{display:flex; align-items:center; gap:8px; font-size:12px;}}
.style-line {{width:28px; border-top:2px solid #111; display:inline-block; flex:0 0 28px;}}
.style-line.dashed {{border:0; height:2px; background:repeating-linear-gradient(to right, #111 0 3px, transparent 3px 5px);}}
.swatch-line {{width:22px; border-top:2px solid; display:inline-block; flex:0 0 22px;}}
#ecdf-tip .name {{font-weight:700; font-size:13px; margin-bottom:2px;}}
#ecdf-tip .sub {{color:#555; font-size:11px; margin-bottom:2px;}}
#ecdf-tip .section {{margin-top:8px; font-size:11px; font-weight:700; color:#555;}}
</style>
<h1 id="ecdf-title">{heading}</h1>
<div id="ecdf-menus">
{set_menu}<label{comparison_style}>Comparison <select id="ecdf-comparison">
<option value="algorithm">Algorithm</option>
<option value="class">Algorithm class</option>
<option value="start">Cold and warm start</option>
</select></label>
<label style="margin-left: 16px;">Algorithms <select id="ecdf-algorithms">
{algorithm_html}
</select></label>
<label style="margin-left: 16px;">Problems <select id="ecdf-scope">
{scope_html}
</select></label>
<label style="margin-left: 16px;">Relative target tolerance ε<sub>X</sub> <select id="ecdf-margin">
{margin_html}
</select></label>
<label style="margin-left: 16px;">Evaluation budget <select id="ecdf-budget">
{budget_html}
</select></label>
<button type="button" id="ecdf-reset">Reset view</button>
{export_buttons}
</div>
<div id="ecdf-shell">
<div id="ecdf-main"></div>
<div id="ecdf-legend"></div>
</div>
<div id="ecdf-caption">{description}</div>
<div id="ecdf-tip" style="display:none; position:fixed; z-index:20; pointer-events:none; max-width:320px; background:#f7f7f7; color:#222; border:1px solid #d5d5d5; border-left-width:4px; border-radius:2px; padding:8px 10px 8px 12px; font-size:12px; line-height:1.35; box-shadow:0 1px 4px rgba(0,0,0,0.12);"></div>
"""
    script = f"""
<script>
(function () {{
  var payload = {payload};
  var problemSets = {set_json};
    var gd = document.getElementById("ecdf-pooled");
  var tip = document.getElementById("ecdf-tip");
  var legend = document.getElementById("ecdf-legend");
  var main = document.getElementById("ecdf-main");
  var active = null;
  var timer = null;
  var initialScope = document.getElementById("ecdf-scope").value;
  var initialMargin = document.getElementById("ecdf-margin").value;
  if (gd && main) main.appendChild(gd);
  function formatInt(value) {{
    return Math.round(value).toLocaleString("en-US");
  }}
  function formatPct(count, total) {{
    var pct = total ? (100 * count / total) : 0;
    return pct.toLocaleString("en-US", {{minimumFractionDigits: 1, maximumFractionDigits: 1}}) + "%";
  }}
  function esc(value) {{
    return String(value).replace(/[&<>"]/g, function (ch) {{
      return {{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}}[ch];
    }});
  }}
  function selectedSet() {{
    var found = null;
    var menu = document.getElementById("ecdf-set");
    if (!menu) return null;
    problemSets.forEach(function (item) {{
      if (item.id === menu.value) found = item;
    }});
    return found;
  }}
  function fillEcdfProblems(setId) {{
    var select = document.getElementById("ecdf-scope");
    var found = null;
    problemSets.forEach(function (item) {{
      if (item.id === setId) found = item;
    }});
    if (!found) return;
    select.innerHTML = found.options.map(function (item) {{
      return '<option value="' + esc(item.value) + '">' + esc(item.label) + "</option>";
    }}).join("");
  }}
  function outcomeAt(x, counts, budget) {{
    var count = 0;
    var hit = null;
    var xs = x || [];
    var cs = counts || [];
    var i;
    for (i = 0; i < xs.length; i++) {{
      var xi = Number(xs[i]);
      if (!(xi <= budget)) continue;
      var next = Number(cs[i]);
      if (next > count) {{
        count = next;
        hit = xi;
      }}
    }}
    return {{count: count, hit: hit}};
  }}
  function compareText(left, right) {{
    var n = Math.max(left.length, right.length);
    var i;
    for (i = 0; i < n; i++) {{
      var a = left[i] || "";
      var b = right[i] || "";
      if (a < b) return -1;
      if (a > b) return 1;
    }}
    return 0;
  }}
  function ecdfFamilyRank(curves) {{
    var bestRate = -1;
    var bestHit = Infinity;
    curves.forEach(function (curve) {{
      var total = Number(curve.meta.n_runs) || 0;
      var at = outcomeAt(curve.x, curve.counts, selectedBudget());
      var rate = total ? at.count / total : 0;
      var hit = at.count && at.hit != null ? Number(at.hit) : Infinity;
      if (rate > bestRate || (rate === bestRate && hit < bestHit)) {{
        bestRate = rate;
        bestHit = hit;
      }}
    }});
    return {{rate: bestRate < 0 ? 0 : bestRate, hit: bestHit}};
  }}
  function byEcdfFamily(a, b) {{
    var left = ecdfFamilyRank(a.curves);
    var right = ecdfFamilyRank(b.curves);
    if (left.rate !== right.rate) return right.rate - left.rate;
    if (left.hit !== right.hit) return left.hit - right.hit;
    return compareText(
      (a.curves[0].meta && a.curves[0].meta.alpha) || [],
      (b.curves[0].meta && b.curves[0].meta.alpha) || []
    );
  }}
  function startRank(curve) {{
    return curve.meta && curve.meta.start === "warm" ? 1 : 0;
  }}
  function selectedCurves() {{
    var scope = document.getElementById("ecdf-scope").value;
    var margin = document.getElementById("ecdf-margin").value;
    var filter = document.getElementById("ecdf-algorithms").value;
    var groups = [];
    var index = {{}};
    (payload.curves || []).forEach(function (curve) {{
      if (curve.scope !== scope || String(curve.margin) !== margin) return;
      if (!curve.meta || !curve.meta.family) return;
      var family = curve.meta.family;
      if (!(family in index)) {{
        index[family] = groups.length;
        groups.push({{family: family, curves: []}});
      }}
      groups[index[family]].curves.push(curve);
    }});
    groups.sort(byEcdfFamily);
    if (filter.indexOf("class:") === 0) {{
      var className = filter.slice(6);
      groups = groups.filter(function (group) {{
        return group.curves[0].meta.category === className;
      }});
    }} else if (filter.indexOf("top:") === 0) {{
      groups = groups.slice(0, Number(filter.slice(4)));
    }}
    var selected = [];
    groups.forEach(function (group) {{
      var members = group.curves.slice().sort(function (a, b) {{
        return startRank(a) - startRank(b);
      }});
      selected = selected.concat(members);
    }});
    return selected;
  }}
  function scatterOf(curve) {{
    var mode = document.getElementById("ecdf-comparison").value;
    var meta = curve.meta;
    var palette = meta.colors || {{}};
    var start = meta.start === "warm" ? "Warm start" : "Cold start";
    return {{
      x: curve.x,
      y: curve.y,
      customdata: curve.counts,
      mode: "lines",
      name: meta.display_name + " · " + start,
      meta: meta,
      line: {{
        color: palette[mode] || "#444444",
        width: 2,
        dash: mode === "algorithm" && meta.start === "warm" ? "3px,2px" : "solid",
        shape: "hv"
      }},
      hovertemplate: " ",
      hoverlabel: {{
        bgcolor: "rgba(0,0,0,0)",
        bordercolor: "rgba(0,0,0,0)",
        font: {{color: "rgba(0,0,0,0)", size: 1}}
      }},
      showlegend: false
    }};
  }}
  function cursorTrace() {{
    return {{
      x: [null],
      y: [null],
      mode: "markers",
      marker: {{
        symbol: "x",
        size: 12,
        color: "#111111",
        line: {{width: 0.5, color: "#111111"}}
      }},
      hoverinfo: "skip",
      showlegend: false,
      visible: false,
      meta: {{cursor: true}},
      name: ""
    }};
  }}
  function drawCurves() {{
    var data = selectedCurves().map(scatterOf);
    data.push(cursorTrace());
    return Plotly.react(gd, data, gd.layout);
  }}
  function cursorIndex() {{
    return gd.data.length - 1;
  }}
  function hideCursor() {{
    if (!gd || !gd.data || !gd.data.length) return;
    Plotly.restyle(gd, {{visible: false}}, [cursorIndex()]);
  }}
  function showCursor(point) {{
    Plotly.restyle(gd, {{
      x: [[point.x]],
      y: [[point.y]],
      visible: true
    }}, [cursorIndex()]);
  }}
  function isActive(kind, key) {{
    return active && active.kind === kind && active.key === key;
  }}
  function paint(kind, key) {{
    var opacity = [];
    var width = [];
    var indices = [];
    gd.data.forEach(function (trace, index) {{
      if (!trace.meta || !trace.meta.family) return;
      var value = trace.meta.family;
      if (kind === "category") value = trace.meta.category;
      if (kind === "start") value = trace.meta.start;
      var on = !key || value === key;
      opacity.push(on ? 1 : 0.15);
      width.push(key && on ? 3.5 : 2);
      indices.push(index);
    }});
    if (indices.length) {{
      Plotly.restyle(gd, {{opacity: opacity, "line.width": width}}, indices);
    }}
    legend.querySelectorAll("[data-highlight]").forEach(function (row) {{
      var on = !key || (
        row.getAttribute("data-highlight") === kind
        && row.getAttribute("data-key") === key
      );
      row.style.opacity = on ? "1" : "0.45";
    }});
  }}
  function applyColors() {{
    var mode = document.getElementById("ecdf-comparison").value;
    var colors = [];
    var dashes = [];
    var indices = [];
    gd.data.forEach(function (trace, index) {{
      if (!trace.meta || !trace.meta.family) return;
      var palette = trace.meta.colors;
      colors.push(palette && palette[mode] ? palette[mode] : "#444444");
      dashes.push(mode === "algorithm" && trace.meta.start === "warm" ? "3px,2px" : "solid");
      indices.push(index);
    }});
    if (indices.length) {{
      Plotly.restyle(gd, {{"line.color": colors, "line.dash": dashes}}, indices);
    }}
    if (tip.style.display === "block" && active) {{
      var match = null;
      gd.data.forEach(function (trace) {{
        if (!match && trace.meta && trace.meta.family === active.key && active.kind === "family") {{
          match = trace;
        }}
      }});
      if (match && match.meta.colors && match.meta.colors[mode]) {{
        tip.style.borderLeftColor = match.meta.colors[mode];
      }}
    }}
  }}
  function selectedBudget() {{
    return Number(document.getElementById("ecdf-budget").value);
  }}
  function series(value) {{
    if (!value) return [];
    if (typeof value.length === "number") return value;
    if (value._inputArray && typeof value._inputArray.length === "number") {{
      return value._inputArray;
    }}
    if (!value.bdata || !value.dtype) return [];
    var raw = atob(value.bdata);
    var bytes = new Uint8Array(raw.length);
    var i;
    for (i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    var view = {{
      f8: Float64Array, f4: Float32Array,
      i1: Int8Array, i2: Int16Array, i4: Int32Array,
      u1: Uint8Array, u2: Uint16Array, u4: Uint32Array
    }}[value.dtype];
    return view ? new view(bytes.buffer) : [];
  }}
  function valueAtBudget(trace, budget) {{
    return outcomeAt(series(trace.x), series(trace.customdata), budget);
  }}
  function resultCell(trace) {{
    if (!trace || !trace.meta) return '<div class="miss">—</div>';
    var total = Number(trace.meta.n_runs) || 0;
    var at = valueAtBudget(trace, selectedBudget());
    var pct = '<div class="pct">' + formatPct(at.count, total) + "</div>";
    if (!at.count || at.hit == null) {{
      return pct + '<div class="miss">—</div>';
    }}
    return pct + '<div class="fe">' + formatInt(at.hit) + " FE</div>";
  }}
  function drawnTraces() {{
    var traces = [];
    gd.data.forEach(function (trace) {{
      if (!trace.meta || !trace.meta.family) return;
      traces.push(trace);
    }});
    return traces;
  }}
  function renderAlgorithmLegend(traces) {{
    var groups = [];
    var index = {{}};
    traces.forEach(function (trace) {{
      var key = trace.meta.family;
      if (!(key in index)) {{
        index[key] = groups.length;
        groups.push({{
          family: key,
          name: trace.meta.display_name,
          qualifier: trace.meta.qualifier || "",
          localSearch: trace.meta.local_search || "",
          handling: trace.meta.handling || "PEN",
          color: (trace.meta.colors && trace.meta.colors.algorithm) || "#444444",
          cold: null,
          warm: null
        }});
      }}
      if (trace.meta.start === "warm") groups[index[key]].warm = trace;
      else groups[index[key]].cold = trace;
    }});
    var rows = groups.map(function (group) {{
      var name = group.name;
      if (group.qualifier) name += " (" + group.qualifier + ")";
      var handling = group.handling === "CON" ? "Constraint-aware" : "Penalty-based";
      if (group.localSearch) handling += ", " + group.localSearch;
      return '<div class="ecdf-row" data-highlight="family" data-key="' + esc(group.family) + '">'
        + '<div class="name"><span class="dot" style="background:' + group.color + ';"></span>'
        + '<span class="label">' + esc(name)
        + '<span class="handling">' + esc(handling) + "</span></span></div>"
        + "<div>" + resultCell(group.cold) + "</div>"
        + "<div>" + resultCell(group.warm) + "</div></div>";
    }}).join("");
    return '<div class="ecdf-style">'
      + '<span class="ecdf-style-item"><span class="style-line"></span>Cold start</span>'
      + '<span class="ecdf-style-item"><span class="style-line dashed"></span>Warm start</span>'
      + '</div><div class="ecdf-compact"><div class="head"><span>Algorithms</span>'
      + '<span>Cold<span class="subhead">Last target hit</span></span>'
      + '<span>Warm<span class="subhead">Last target hit</span></span></div>'
      + rows + "</div>";
  }}
  function renderCategoryLegend(traces) {{
    var colors = {{}};
    traces.forEach(function (trace) {{
      var category = trace.meta.category;
      if (!category || (category in colors)) return;
      colors[category] = (trace.meta.colors && trace.meta.colors.class) || "#444444";
    }});
    var items = (payload.classes || []).filter(function (category) {{
      return category in colors;
    }}).map(function (category) {{
      return '<div class="ecdf-key-item" data-highlight="category" data-key="' + esc(category) + '">'
        + '<span class="swatch-line" style="border-top-color:' + colors[category] + ';"></span>'
        + esc(category) + "</div>";
    }}).join("");
    return '<div class="ecdf-key"><div class="head">Algorithm class</div>' + items + "</div>";
  }}
  function renderStartLegend(traces) {{
    var colors = {{}};
    traces.forEach(function (trace) {{
      var start = trace.meta.start;
      if (!start || (start in colors)) return;
      colors[start] = (trace.meta.colors && trace.meta.colors.start) || "#444444";
    }});
    var items = ["cold", "warm"].filter(function (start) {{
      return start in colors;
    }}).map(function (start) {{
      var label = start === "warm" ? "Warm start" : "Cold start";
      return '<div class="ecdf-key-item" data-highlight="start" data-key="' + start + '">'
        + '<span class="swatch-line" style="border-top-color:' + colors[start] + ';"></span>'
        + label + "</div>";
    }}).join("");
    return '<div class="ecdf-key"><div class="head">Cold / warm start</div>' + items + "</div>";
  }}
  function renderLegend() {{
    var mode = document.getElementById("ecdf-comparison").value;
    var traces = drawnTraces();
    if (mode === "class") legend.innerHTML = renderCategoryLegend(traces);
    else if (mode === "start") legend.innerHTML = renderStartLegend(traces);
    else legend.innerHTML = renderAlgorithmLegend(traces);
  }}
  function hideTip() {{
    tip.style.display = "none";
  }}
  function applyBudget() {{
    if (!gd || !window.Plotly) return;
    var budget = selectedBudget();
    Plotly.relayout(gd, {{
      "xaxis.autorange": false,
      "xaxis.range": [0, budget],
      "yaxis.autorange": false,
      "yaxis.range": [0, 100],
      shapes: [{{
        type: "line",
        xref: "x",
        yref: "y",
        x0: budget,
        x1: budget,
        y0: 0,
        y1: 100,
        line: {{color: "#444444", width: 1, dash: "dot"}}
      }}]
    }});
  }}
  function writeTitle() {{
    var scope = document.getElementById("ecdf-scope");
    var pooled = scope.value === "all" || scope.value.slice(-4) === "/all";
    var chosen = selectedSet();
    var where = pooled
      ? (chosen && chosen.pooled ? chosen.pooled : "across optimisation problems")
      : "on " + esc(scope.options[scope.selectedIndex].text);
    var mode = document.getElementById("ecdf-comparison").value;
    var subject = "algorithms";
    if (mode === "class") subject = "algorithm classes";
    else if (mode === "start") subject = "cold and warm start algorithm variants";
    var margin = document.getElementById("ecdf-margin").value;
    var heading = "Cumulative success rate of " + subject + " " + where;
    document.getElementById("ecdf-title").innerHTML =
      heading
      + '<span class="ecdf-meta"><span class="ecdf-tolerance">relative target tolerance ε<sub>X</sub> = '
      + margin + '%</span><span class="ecdf-budget">'
      + formatInt(selectedBudget()) + " function evaluations</span></span>";
    document.title = heading;
  }}
  function apply() {{
    var pending = drawCurves();
    var finish = function () {{
      writeTitle();
      applyBudget();
      applyColors();
      renderLegend();
      clearHighlight();
    }};
    if (pending && pending.then) pending.then(finish);
    else finish();
  }}
  function clearHighlight() {{
    active = null;
    paint(null, null);
    hideTip();
    hideCursor();
  }}
  function showTip(event) {{
    var point = event.points && event.points[0];
    if (!point || !point.data || !point.data.meta || !point.data.meta.family) return;
    var meta = point.data.meta;
    var mode = document.getElementById("ecdf-comparison").value;
    var color = (meta.colors && meta.colors[mode]) || "#444444";
    var total = Number(meta.n_runs) || 0;
    var cumulativeCount = point.customdata == null
      ? Math.round(Number(point.y) / 100 * total)
      : Math.round(Number(point.customdata));
    var budget = selectedBudget();
    var at = valueAtBudget(point.data, budget);
    var lastHit = (at.count && at.hit != null) ? formatInt(at.hit) + " FE" : "—";
    tip.style.borderLeftColor = color;
    tip.innerHTML =
      '<div class="name">' + esc(meta.display_name) + "</div>" +
      '<div class="sub">' + esc(meta.subtitle) + "</div>" +
      '<div class="section">At this evaluation</div>' +
      "<div>Function evaluations: " + formatInt(point.x) + "</div>" +
      "<div>Cumulative success rate: " + formatPct(cumulativeCount, total) +
        " (" + formatInt(cumulativeCount) + "/" + formatInt(total) + " runs)</div>" +
      '<div class="section">At ' + formatInt(budget) + " FE</div>" +
      "<div>Success rate: " + formatPct(at.count, total) +
        " (" + formatInt(at.count) + "/" + formatInt(total) + " runs)</div>" +
      "<div>Last target hit: " + lastHit + "</div>";
    tip.style.display = "block";
    var mouse = event.event || {{}};
    var x = (mouse.clientX || 0) + 18;
    var y = (mouse.clientY || 0) + 18;
    if (x + tip.offsetWidth > window.innerWidth - 8) {{
      x = (mouse.clientX || 0) - tip.offsetWidth - 18;
    }}
    if (y + tip.offsetHeight > window.innerHeight - 8) {{
      y = (mouse.clientY || 0) - tip.offsetHeight - 18;
    }}
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
    showCursor(point);
  }}
  function resetView() {{
    var setMenu = document.getElementById("ecdf-set");
    if (setMenu) {{
      setMenu.selectedIndex = 0;
      fillEcdfProblems(setMenu.value);
    }}
    document.getElementById("ecdf-scope").value = initialScope;
    document.getElementById("ecdf-margin").value = initialMargin;
    document.getElementById("ecdf-algorithms").value = "all";
    document.getElementById("ecdf-budget").value = "15000";
    document.getElementById("ecdf-comparison").value = "algorithm";
    apply();
  }}
  var ecdfSet = document.getElementById("ecdf-set");
  if (ecdfSet) {{
    ecdfSet.addEventListener("change", function () {{
      fillEcdfProblems(ecdfSet.value);
      apply();
    }});
  }}
  document.getElementById("ecdf-scope").addEventListener("change", apply);
  document.getElementById("ecdf-algorithms").addEventListener("change", apply);
  document.getElementById("ecdf-budget").addEventListener("change", apply);
  document.getElementById("ecdf-margin").addEventListener("change", apply);
  document.getElementById("ecdf-reset").addEventListener("click", resetView);
  function csvCell(value) {{
    var text = String(value == null ? "" : value);
    if (text.indexOf(",") >= 0 || text.indexOf('"') >= 0) {{
      return '"' + text.replace(/"/g, '""') + '"';
    }}
    return text;
  }}
  function firstHitCsv() {{
    var margin = document.getElementById("ecdf-margin").value;
    var budget = selectedBudget();
    var lines = ["problem,run_id,tolerance_percent,algorithm,start,handling,n_runs,hit_number,evaluation"];
    drawnTraces().forEach(function (trace) {{
      var meta = trace.meta;
      var handling = meta.handling === "CON" ? "Constraint-aware" : "Penalty-based";
      if (meta.local_search) handling += ", " + meta.local_search;
      var algorithm = meta.display_name;
      if (meta.qualifier) algorithm += " (" + meta.qualifier + ")";
      var start = meta.start === "warm" ? "warm" : "cold";
      var runs = (meta.hits || []).map(function (run) {{
        return {{
          problem: run.problem || "",
          runId: run.run_id,
          evaluation: run.evaluation == null ? null : Number(run.evaluation)
        }};
      }});
      var successes = runs.filter(function (run) {{
        return run.evaluation != null && run.evaluation <= budget;
      }});
      var misses = runs.filter(function (run) {{
        return run.evaluation == null || run.evaluation > budget;
      }});
      successes.sort(function (left, right) {{
        return left.evaluation - right.evaluation || String(left.problem).localeCompare(right.problem) || left.runId - right.runId;
      }});
      misses.sort(function (left, right) {{
        return String(left.problem).localeCompare(right.problem) || left.runId - right.runId;
      }});
      function row(run, hitNumber, evaluation) {{
        return [
          csvCell(run.problem), run.runId, margin, csvCell(algorithm),
          start, csvCell(handling), meta.n_runs, hitNumber, evaluation
        ].join(",");
      }}
      if (!runs.length) {{
        lines.push(["", "", margin, csvCell(algorithm), start, csvCell(handling), meta.n_runs, "", ""].join(","));
        return;
      }}
      successes.forEach(function (run, index) {{
        lines.push(row(run, index + 1, Math.round(run.evaluation)));
      }});
      misses.forEach(function (run) {{
        lines.push(row(run, "", ""));
      }});
    }});
    return lines.join(String.fromCharCode(10));
  }}
  function downloadText(filename, text, type) {{
    var blob = new Blob([text], {{type: type}});
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }}
  function resultText(trace) {{
    if (!trace || !trace.meta) return "—";
    var total = Number(trace.meta.n_runs) || 0;
    var at = valueAtBudget(trace, selectedBudget());
    var pct = formatPct(at.count, total);
    if (!at.count || at.hit == null) return pct + "  —";
    return pct + "  " + formatInt(at.hit) + " FE";
  }}
  function exportRows() {{
    var mode = document.getElementById("ecdf-comparison").value;
    var traces = drawnTraces();
    var rows = [];
    if (mode === "class" || mode === "start") {{
      var colors = {{}};
      traces.forEach(function (trace) {{
        var key = mode === "class" ? trace.meta.category : trace.meta.start;
        if (!key || (key in colors)) return;
        colors[key] = (trace.meta.colors && trace.meta.colors[mode]) || "#444444";
      }});
      var keys = mode === "start"
        ? ["cold", "warm"]
        : (payload.classes || []);
      keys.forEach(function (key) {{
        if (!(key in colors)) return;
        var label = key === "warm" ? "Warm start" : key === "cold" ? "Cold start" : key;
        rows.push({{kind: "swatch", color: colors[key], label: label}});
      }});
      return rows;
    }}
    rows.push({{kind: "style"}});
    var groups = [];
    var index = {{}};
    traces.forEach(function (trace) {{
      var key = trace.meta.family;
      if (!(key in index)) {{
        index[key] = groups.length;
        var name = trace.meta.display_name;
        if (trace.meta.qualifier) name += " (" + trace.meta.qualifier + ")";
        groups.push({{
          name: name,
          handling: (trace.meta.handling === "CON" ? "Constraint-aware" : "Penalty-based") + (trace.meta.local_search ? ", " + trace.meta.local_search : ""),
          color: (trace.meta.colors && trace.meta.colors.algorithm) || "#444444",
          cold: null,
          warm: null
        }});
      }}
      if (trace.meta.start === "warm") groups[index[key]].warm = trace;
      else groups[index[key]].cold = trace;
    }});
    groups.forEach(function (group) {{
      rows.push({{
        kind: "algorithm",
        name: group.name,
        handling: group.handling,
        color: group.color,
        cold: resultText(group.cold),
        warm: resultText(group.warm)
      }});
    }});
    return rows;
  }}
  function exportLegendHeight(rows) {{
    var height = 16;
    rows.forEach(function (row) {{
      height += row.kind === "algorithm" ? 36 : 22;
    }});
    return height;
  }}
  function paintExportLegend(ctx, left, top, rows) {{
    ctx.font = "12px 'DejaVu Sans', sans-serif";
    ctx.textBaseline = "middle";
    var y = top;
    rows.forEach(function (row) {{
      if (row.kind === "style") {{
        ctx.strokeStyle = "#111111";
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(left, y + 8);
        ctx.lineTo(left + 28, y + 8);
        ctx.stroke();
        ctx.fillStyle = "#222222";
        ctx.fillText("Cold start", left + 36, y + 8);
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(left + 130, y + 8);
        ctx.lineTo(left + 158, y + 8);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillText("Warm start", left + 166, y + 8);
        y += 22;
        return;
      }}
      if (row.kind === "swatch") {{
        ctx.strokeStyle = row.color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(left, y + 8);
        ctx.lineTo(left + 22, y + 8);
        ctx.stroke();
        ctx.fillStyle = "#222222";
        ctx.fillText(row.label, left + 30, y + 8);
        y += 22;
        return;
      }}
      ctx.fillStyle = row.color;
      ctx.fillRect(left, y + 4, 8, 8);
      ctx.fillStyle = "#222222";
      ctx.font = "12px 'DejaVu Sans', sans-serif";
      ctx.fillText(row.name, left + 16, y + 8);
      ctx.fillStyle = "#666666";
      ctx.font = "11px 'DejaVu Sans', sans-serif";
      ctx.fillText(row.handling, left + 16, y + 22);
      ctx.fillStyle = "#222222";
      ctx.font = "12px 'DejaVu Sans', sans-serif";
      ctx.fillText(row.cold, left + 150, y + 22);
      ctx.fillText(row.warm, left + 270, y + 22);
      y += 36;
    }});
  }}
  function svgEscape(value) {{
    return String(value).replace(/[&<>"]/g, function (ch) {{
      return {{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}}[ch];
    }});
  }}
  function legendSvg(rows, left) {{
    var parts = [];
    var y = 16;
    rows.forEach(function (row) {{
      if (row.kind === "style") {{
        parts.push('<line x1="' + left + '" y1="' + (y + 8) + '" x2="' + (left + 28) + '" y2="' + (y + 8) + '" stroke="#111111" stroke-width="2"/>');
        parts.push('<text x="' + (left + 36) + '" y="' + (y + 12) + '" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#222222">Cold start</text>');
        parts.push('<line x1="' + (left + 130) + '" y1="' + (y + 8) + '" x2="' + (left + 158) + '" y2="' + (y + 8) + '" stroke="#111111" stroke-width="2" stroke-dasharray="3 2"/>');
        parts.push('<text x="' + (left + 166) + '" y="' + (y + 12) + '" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#222222">Warm start</text>');
        y += 22;
        return;
      }}
      if (row.kind === "swatch") {{
        parts.push('<line x1="' + left + '" y1="' + (y + 8) + '" x2="' + (left + 22) + '" y2="' + (y + 8) + '" stroke="' + row.color + '" stroke-width="3"/>');
        parts.push('<text x="' + (left + 30) + '" y="' + (y + 12) + '" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#222222">' + svgEscape(row.label) + "</text>");
        y += 22;
        return;
      }}
      parts.push('<rect x="' + left + '" y="' + (y + 4) + '" width="8" height="8" fill="' + row.color + '"/>');
      parts.push('<text x="' + (left + 16) + '" y="' + (y + 12) + '" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#222222">' + svgEscape(row.name) + "</text>");
      parts.push('<text x="' + (left + 16) + '" y="' + (y + 26) + '" font-family="DejaVu Sans, sans-serif" font-size="11" fill="#666666">' + svgEscape(row.handling) + "</text>");
      parts.push('<text x="' + (left + 150) + '" y="' + (y + 26) + '" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#222222">' + svgEscape(row.cold) + "</text>");
      parts.push('<text x="' + (left + 270) + '" y="' + (y + 26) + '" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#222222">' + svgEscape(row.warm) + "</text>");
      y += 36;
    }});
    return parts.join("");
  }}
  function svgFromDataUrl(url) {{
    var comma = url.indexOf(",");
    var meta = url.slice(0, comma);
    var body = url.slice(comma + 1);
    return meta.indexOf("base64") === -1 ? decodeURIComponent(body) : atob(body);
  }}
  function downloadFigure(format) {{
    var plotW = 1100;
    var plotH = 560;
    var legendW = 560;
    var rows = exportRows();
    var height = Math.max(plotH, exportLegendHeight(rows) + 16);
    if (format === "svg") {{
      Plotly.toImage(gd, {{format: "svg", width: plotW, height: plotH}}).then(function (url) {{
        var doc = new DOMParser().parseFromString(svgFromDataUrl(url), "image/svg+xml");
        var ns = "http://www.w3.org/2000/svg";
        var plot = doc.documentElement;
        var totalW = plotW + legendW;
        plot.setAttribute("x", "0");
        plot.setAttribute("y", "0");
        plot.setAttribute("width", String(plotW));
        plot.setAttribute("height", String(plotH));
        var outer = doc.createElementNS(ns, "svg");
        outer.setAttribute("xmlns", ns);
        outer.setAttribute("width", String(totalW));
        outer.setAttribute("height", String(height));
        outer.setAttribute("viewBox", "0 0 " + totalW + " " + height);
        var backdrop = doc.createElementNS(ns, "rect");
        backdrop.setAttribute("width", String(totalW));
        backdrop.setAttribute("height", String(height));
        backdrop.setAttribute("fill", "#ffffff");
        var key = doc.createElementNS(ns, "g");
        key.innerHTML = legendSvg(rows, plotW + 16);
        outer.appendChild(backdrop);
        outer.appendChild(plot);
        outer.appendChild(key);
        downloadText("ecdf_success.svg", new XMLSerializer().serializeToString(outer), "image/svg+xml");
      }});
      return;
    }}
    Plotly.toImage(gd, {{format: "png", width: plotW, height: plotH}}).then(function (url) {{
      var image = new Image();
      image.onload = function () {{
        var canvas = document.createElement("canvas");
        canvas.width = plotW + legendW;
        canvas.height = height;
        var ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0);
        paintExportLegend(ctx, plotW + 16, 16, rows);
        var link = document.createElement("a");
        link.href = canvas.toDataURL("image/png");
        link.download = "ecdf_success.png";
        link.click();
      }};
      image.src = url;
    }});
  }}
{export_script}
  document.getElementById("ecdf-comparison").addEventListener("change", function () {{
    applyColors();
    renderLegend();
    writeTitle();
    clearHighlight();
  }});
  legend.addEventListener("mouseover", function (event) {{
    var row = event.target.closest("[data-highlight]");
    if (!row) return;
    if (timer) {{ clearTimeout(timer); timer = null; }}
    var kind = row.getAttribute("data-highlight");
    var key = row.getAttribute("data-key");
    if (key && !isActive(kind, key)) {{
      active = {{kind: kind, key: key}};
      paint(kind, key);
    }}
  }});
  legend.addEventListener("mouseout", function (event) {{
    var row = event.target.closest("[data-highlight]");
    if (!row || row.contains(event.relatedTarget)) return;
    if (timer) clearTimeout(timer);
    timer = setTimeout(function () {{
      timer = null;
      active = null;
      paint(null, null);
    }}, 60);
  }});
  if (gd && gd.on) {{
    gd.on("plotly_hover", function (event) {{
      if (timer) {{ clearTimeout(timer); timer = null; }}
      var point = event.points && event.points[0];
      var family = point && point.data && point.data.meta && point.data.meta.family;
      if (family && !isActive("family", family)) {{
        active = {{kind: "family", key: family}};
        paint("family", family);
      }}
      showTip(event);
    }});
    gd.on("plotly_unhover", function () {{
      hideTip();
      hideCursor();
      if (timer) clearTimeout(timer);
      timer = setTimeout(function () {{
        timer = null;
        if (active === null) return;
        active = null;
        paint(null, null);
      }}, 60);
    }});
  }}
  apply();
  if (gd && window.Plotly) Plotly.Plots.resize(gd);
  if (document.fonts && document.fonts.ready && gd && window.Plotly) {{
    document.fonts.ready.then(function () {{
      Plotly.relayout(gd, {{"font.family": "DejaVu Sans"}});
    }});
  }}
}})();
</script>
"""
    marker = '<div id="ecdf-pooled"'
    if marker not in html:
        raise ValueError("interactive ECDF html is missing the plot div")
    html = html.replace(marker, controls + marker, 1)
    plain_title = heading.split('<span class="ecdf-meta">', 1)[0]
    html = html.replace(
        '<head><meta charset="utf-8" /></head>',
        '<head><meta charset="utf-8" /><title>'
        + html_escape(plain_title)
        + "</title></head>",
        1,
    )
    if "</body>" in html:
        return html.replace("</body>", script + "</body>", 1)
    return html + script


def html_escape(value: str) -> str:
    return html.escape(str(value), quote=True)


def _ecdf_curve_record(
    name: str,
    runs: tuple[RunTrajectory, ...],
    *,
    scope: str,
    percent: float,
    colors: dict[str, str],
    show_qualifier: bool,
) -> dict[str, object]:
    """One step curve, stored so the page can draw the selected view."""
    curve = success_probability_curve(
        runs,
        percent,
        global_optimum=None,
        target_reference=None,
    )
    item = presentation_for(name)
    n_runs = int(curve.n_runs)
    n_success = int(curve.n_success)
    plateau = int(curve.hit_times[-1]) if curve.hit_times.size else None
    if curve.hit_times.size == 0:
        x_values = np.asarray([0.0, curve.x_max], dtype=float)
        y_values = np.zeros(2, dtype=float)
        counts = np.zeros(2, dtype=int)
    else:
        steps = 100.0 * np.arange(1, n_success + 1) / float(n_runs)
        counts = np.concatenate(
            [
                np.asarray([0], dtype=int),
                np.arange(1, n_success + 1, dtype=int),
                np.asarray([n_success], dtype=int),
            ]
        )
        x_values = np.concatenate(
            [np.asarray([0.0]), curve.hit_times, np.asarray([curve.x_max])]
        )
        y_values = np.concatenate(
            [np.asarray([0.0]), steps, np.asarray([steps[-1]])]
        )
    return {
        "scope": scope,
        "margin": float(percent),
        "x": [float(value) for value in x_values],
        "y": [float(value) for value in y_values],
        "counts": [int(value) for value in counts],
        "meta": {
            "family": item.family,
            "display_name": item.display_name,
            "subtitle": tooltip_subtitle(item, show_qualifier=show_qualifier),
            "constraint": constraint_label(item),
            "qualifier": item.qualifier
            if show_qualifier and item.qualifier not in {"SI-DESC", "EQU-DESC"}
            else "",
            "local_search": local_search_label(item),
            "category": item.algorithm_class,
            "handling": "CON" if item.constraint_aware else "PEN",
            "start": item.start,
            "alpha": list(_alphabetical_rank(item.family)),
            "n_runs": n_runs,
            "n_success": n_success,
            "plateau": plateau,
            "colors": colors,
            "hits": [
                {
                    "problem": run.problem_id,
                    "run_id": int(run.task_index),
                    "evaluation": hit_time_at_percent(run, percent),
                }
                for run in runs
            ],
        },
    }


def _add_ecdf_trace(go, figure, record: dict[str, object]) -> None:
    """Draw one stored curve. The figure holds only the selected view."""
    meta = record["meta"]
    assert isinstance(meta, dict)
    start = "Warm start" if meta["start"] == "warm" else "Cold start"
    colors = meta["colors"]
    assert isinstance(colors, dict)
    figure.add_trace(
        go.Scatter(
            x=record["x"],
            y=record["y"],
            mode="lines",
            name=f"{meta['display_name']} · {start}",
            meta=meta,
            line={
                "color": colors["algorithm"],
                "width": 2,
                "dash": "3px,2px" if meta["start"] == "warm" else "solid",
                "shape": "hv",
            },
            customdata=record["counts"],
            hovertemplate=" ",
            hoverlabel={
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,0,0,0)",
                "font": {"color": "rgba(0,0,0,0)", "size": 1},
            },
            showlegend=False,
        )
    )


def _write_ecdf(
    go,
    grouped,
    path: Path,
    *,
    percent: float,
    color_index: dict[str, int] | None = None,
    title: str = "",
) -> Path:
    """Write the menu ECDF for one algorithm grouping."""
    del color_index, title
    return _write_ecdf_margins(
        go,
        (("all", "All problems", grouped),),
        path,
        percents=(float(percent),),
    )


def _ert_heading(
    scope: str,
    label: str,
    budget: int,
    mode: str = "algorithm",
) -> str:
    """Title above the menus: what is drawn, where, and the selected budget."""
    if _is_pooled_scope(scope):
        where = "across optimisation problems"
    else:
        where = f"on {html_escape(label)}"
    subject = {
        "algorithm": "algorithms",
        "class": "algorithm classes",
        "start": "cold and warm start algorithm variants",
    }.get(mode, "algorithms")
    return f"Expected Running Time of {subject} {where}"


def _ert_ratio_markup() -> str:
    """The shared sum in ERT_cold and ERT_warm."""
    return (
        '<span class="frac"><span class="num">'
        '<span class="sum"><span class="sum-lim">R</span>'
        '<span class="sum-op">∑</span>'
        '<span class="sum-lim sum-lower">r = 1</span></span>'
        " N<sub>r,ε<sub>X</sub></sub></span>"
        '<span class="den">N<sub>ε<sub>X</sub></sub></span></span>'
    )


def _reported_ert(value: float | None, start: str) -> float | None:
    """ERT_cold, or ERT_warm with the one-off predesign cost."""
    if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
        return None
    reported = float(value)
    if start == "warm":
        reported += _PREDESIGN_EVALUATIONS
    return reported


def _ert_description(
    n_problems: int,
    anchor: float,
    local_search_labels: object = (),
    *,
    with_sets: bool = False,
) -> str:
    """Definitions and the Expected Running Time reference under the plot."""
    coverage = _problem_coverage(n_problems, with_sets=with_sets)
    return (
        '<p style="margin:0 0 6px; font-weight:700; color:#222;">Definitions</p>'
        "<p style=\"margin:0 0 8px;\">The interactive Expected Running Time "
        "(ERT) plot compares how many function evaluations optimisation "
        "algorithms are expected to need to reach different relative target "
        "tolerances ε<sub>X</sub>.</p>"
        '<ul style="margin:0; padding-left:18px;">'
        "<li><b>Expected Running Time (ERT).</b> Expected Running Time estimates "
        "the number of function evaluations required to reach a target "
        "ε<sub>X</sub> under hypothetical independent restarts [1]. Each run "
        "contributes the evaluations required to reach the target, or those "
        "performed before termination if unsuccessful. The sum is divided by "
        "the number of successful runs. The quantity is a count of function "
        "evaluations."
        f'<p class="ert-formula">ERT<sub>cold</sub>(ε<sub>X</sub>) = '
        f"{_ert_ratio_markup()}.</p>"
        f'<p class="ert-formula">ERT<sub>warm</sub>(ε<sub>X</sub>) = '
        f"{_ert_ratio_markup()} + 2,700.</p>"
        "<p class=\"formula-note\">ERT<sub>cold</sub> is the Expected Running "
        "Time of the cold-start runs. ERT<sub>warm</sub> is the Expected "
        "Running Time of the warm-start runs plus a one-off cost of 2,700 "
        "predesign evaluations. Expected Running Time is omitted when no run "
        "succeeds.</p></li>"
        f"<li><b>Problems.</b> {coverage} Pooled Expected Running Time is "
        "calculated across all included runs, not as the mean of the "
        "individual problem values.</li>"
        "<li><b>Algorithms.</b> Each algorithm has a cold-start (solid line) and "
        "warm-start (dashed line) variant, shown in the same colour.</li>"
        f"{_local_search_item(local_search_labels)}"
        "<li><b>Function evaluation.</b> The algorithm submits a candidate "
        "design to the analysis model and receives objective and constraint "
        "values. Depending on the algorithm, optimisation uses the penalised "
        f"or unpenalised objective and constraint information. {_PREDESIGN_SENTENCE}</li>"
        "<li><b>Evaluation budget.</b> The maximum number of function "
        "evaluations available to a run to reach the target.</li>"
        "<li><b>Relative target tolerance.</b> A run reaches the target when "
        "its best penalised objective satisfies"
        '<p class="ert-formula">y<sub>p,best</sub> ≤ (1 + ε<sub>X</sub>) '
        "y<sub>p</sub>*,</p>"
        f'<p class="formula-note">{_target_note()}</p></li>'
        "<li><b>Successful run.</b> A completed run that first reaches the "
        "relative target tolerance at or before the selected evaluation "
        "budget.</li>"
        "<li><b>Legend ordering.</b> Algorithm families are ranked by the higher "
        f"overall success rate of their cold-start and warm-start variants at "
        f"ε<sub>X</sub> = {anchor:g}%, then by the lower Expected Running Time of "
        "that better variant, and finally alphabetically. The order follows "
        "the selected set of problem instances, problem instance, and evaluation budget. Cold start stays before "
        "warm start.</li>"
        "</ul>"
        f"{_HANSEN_REFERENCE}{_PUBLICATION}"
    )


def _log_tick_spec(values: list[float]) -> dict[str, object]:
    """Decade labels at 1, 2, 3 and 5, with unlabeled ticks at 4, 6, 7, 8 and 9.

    On a base-10 axis the usual minor ticks sit at every integer from 2 through
    9. Labeling 1, 2, 3 and 5 keeps those subdivisions readable.
    """
    finite = [float(value) for value in values if value is not None and float(value) > 0.0]
    if not finite:
        return {"type": "log", "autorange": False, "range": [0, 5]}
    lo = math.log10(min(finite))
    hi = math.log10(max(finite))
    pad = max(0.08, (hi - lo) * 0.08)
    start, stop = lo - pad, hi + pad
    labeled: list[float] = []
    minor: list[float] = []
    for exp in range(math.floor(start), math.ceil(stop) + 1):
        base = 10.0 ** exp
        for factor in (1, 2, 3, 5):
            value = factor * base
            if start - 0.02 <= math.log10(value) <= stop + 0.02:
                labeled.append(value)
        for factor in (4, 6, 7, 8, 9):
            value = factor * base
            if start <= math.log10(value) <= stop:
                minor.append(value)
    return {
        "type": "log",
        "autorange": False,
        "range": [start, stop],
        "tickmode": "array",
        "tickvals": labeled,
        "ticktext": [f"{int(round(value)):,}" for value in labeled],
        "minor": {
            "tickmode": "array",
            "tickvals": minor,
            "ticks": "outside",
            "ticklen": 3,
            "tickcolor": "#444444",
            "showgrid": False,
        },
    }


def _hitting_time_cache(
    scopes: tuple[tuple[str, str, dict[str, tuple[RunTrajectory, ...]]], ...],
    percents: tuple[float, ...],
) -> dict[int, dict[float, int | None]]:
    """First hitting time of each run at each tolerance, computed once."""
    cache: dict[int, dict[float, int | None]] = {}
    for _scope, _label, grouped in scopes:
        for runs in grouped.values():
            for run in runs:
                key = id(run)
                if key in cache:
                    continue
                cache[key] = {
                    float(percent): hit_time_at_percent(run, float(percent))
                    for percent in percents
                }
    return cache


def _last_success_within_budget(
    hits: list[int | None],
    evaluations_performed: list[int],
    budget: int,
) -> int | None:
    """Latest first-hit that still counts as a success at ``budget``."""
    last: int | None = None
    selected = int(budget)
    for hit, performed in zip(hits, evaluations_performed, strict=True):
        cap = min(int(performed), selected)
        if hit is not None and int(hit) <= cap:
            value = int(hit)
            last = value if last is None else max(last, value)
    return last


def _ert_curve_record(
    name: str,
    runs: tuple[RunTrajectory, ...],
    *,
    scope: str,
    budget: int,
    percents: tuple[float, ...],
    hits: dict[int, dict[float, int | None]],
    colors: dict[str, str],
    show_qualifier: bool,
) -> dict[str, object]:
    """One ERT curve across target tolerances, at one budget and one scope."""
    item = presentation_for(name)
    xs: list[float] = []
    ys: list[float | None] = []
    custom: list[list[float | int | None]] = []
    for percent in percents:
        hit_times = [hits[id(run)][float(percent)] for run in runs]
        performed = [int(run.n_evaluations) for run in runs]
        result = hansen_ert_from_outcomes(
            hit_times,
            performed,
            percent=float(percent),
            budget=int(budget),
        )
        xs.append(float(percent))
        ys.append(_reported_ert(result.ert, item.start))
        custom.append(
            [
                int(result.n_success),
                float(result.sum_hits),
                float(result.failure_cost),
                None if result.failure_cap is None else int(result.failure_cap),
                _last_success_within_budget(hit_times, performed, int(budget)),
            ]
        )
    return {
        "scope": scope,
        "budget": int(budget),
        "x": xs,
        "y": ys,
        "custom": custom,
        "meta": {
            "family": item.family,
            "display_name": item.display_name,
            "subtitle": tooltip_subtitle(item, show_qualifier=show_qualifier),
            "constraint": constraint_label(item),
            "qualifier": item.qualifier
            if show_qualifier and item.qualifier not in {"SI-DESC", "EQU-DESC"}
            else "",
            "local_search": local_search_label(item),
            "category": item.algorithm_class,
            "handling": "CON" if item.constraint_aware else "PEN",
            "start": item.start,
            "alpha": list(_alphabetical_rank(item.family)),
            "n_runs": len(runs),
            "colors": colors,
        },
    }


def _add_ert_trace(go, figure, record: dict[str, object]) -> None:
    """Draw one stored ERT curve. The figure holds only the selected view."""
    meta = record["meta"]
    assert isinstance(meta, dict)
    colors = meta["colors"]
    assert isinstance(colors, dict)
    start = "Warm start" if meta["start"] == "warm" else "Cold start"
    color = str(colors["algorithm"])
    figure.add_trace(
        go.Scatter(
            x=record["x"],
            y=record["y"],
            customdata=record["custom"],
            mode="lines+markers",
            name=f"{meta['display_name']} · {start}",
            meta=meta,
            connectgaps=False,
            line={
                "color": color,
                "width": 2,
                "dash": "3px,2px" if meta["start"] == "warm" else "solid",
            },
            marker={"size": 7, "color": color},
            hovertemplate=" ",
            hoverlabel={
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,0,0,0)",
                "font": {"color": "rgba(0,0,0,0)", "size": 1},
            },
            showlegend=False,
        )
    )


def _write_ert(
    go,
    scopes: tuple[tuple[str, str, dict[str, tuple[RunTrajectory, ...]]], ...],
    path: Path,
    *,
    percents: tuple[float, ...] = _POOLED_ECDF_PERCENTS,
    budgets: tuple[int, ...] = _BUDGETS,
    on_step: Callable[[str], None] | None = None,
    downloads: bool = True,
) -> Path:
    """The pooled interactive ERT: comparison, problem, budget, and algorithms."""
    figure = go.Figure()
    curves: list[dict[str, object]] = []
    targets = tuple(float(percent) for percent in percents)
    caps = tuple(int(budget) for budget in budgets)
    if not targets or not caps:
        raise ValueError("ERT plot requires at least one tolerance and one budget")
    anchor = 1.0 if any(item == 1.0 for item in targets) else targets[0]
    shown_budget = _FULL_BUDGET if _FULL_BUDGET in caps else caps[-1]
    first_scope, _first_label, first_group = scopes[0]
    palette_order = ecdf_algorithm_order(first_group, anchor)
    known = set(palette_order)
    for _scope, _label, grouped in scopes:
        for name in grouped:
            if name not in known:
                palette_order.append(name)
                known.add(name)
    palette = _comparison_palette(palette_order)
    present_classes = _present_classes(palette_order)
    if on_step is not None:
        on_step("hitting times")
    hits = _hitting_time_cache(scopes, targets)
    shown_ys: list[float] = []
    for scope, label, grouped in scopes:
        crowded = qualifiers_needed(tuple(grouped))
        for budget in caps:
            if on_step is not None:
                on_step(f"{label}, {int(budget):,} FE")
            for name in palette_order:
                if name not in grouped:
                    continue
                record = _ert_curve_record(
                name,
                grouped[name],
                    scope=scope,
                    budget=int(budget),
                    percents=targets,
                    hits=hits,
                    colors=palette[name],
                    show_qualifier=name in crowded,
                )
                curves.append(record)
                if scope == first_scope and int(budget) == int(shown_budget):
                    _add_ert_trace(go, figure, record)
                    for value in record["y"]:
                        if isinstance(value, (int, float)) and float(value) > 0.0:
                            shown_ys.append(float(value))
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "symbol": "x",
                "size": 12,
                "color": "#111111",
                "line": {"width": 0.5, "color": "#111111"},
            },
            hoverinfo="skip",
            showlegend=False,
            visible=False,
            meta={"cursor": True},
            name="",
        )
    )
    xaxis = _axis_style("Relative target tolerance ε<sub>X</sub> [%]")
    xaxis.update(
        {
            "tickmode": "array",
            "tickvals": list(targets),
            "ticktext": [f"{percent:g}" for percent in targets],
            "range": [targets[0] - 0.6, targets[-1] + 0.6],
            "autorange": False,
        }
    )
    yaxis = _axis_style("Expected Running Time [FE]")
    yaxis.update(_log_tick_spec(shown_ys))
    figure.update_layout(
        font={"family": _FONT, "size": 13, "color": "#222222"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=xaxis,
        yaxis=yaxis,
        hovermode="closest",
        **_ecdf_legend_layout([]),
    )
    figure.update_layout(margin={"l": 88, "r": 28, "t": 24, "b": 72})
    if on_step is not None:
        on_step("HTML file")
    groups = _problem_set_groups(scopes)
    scope_options = _shown_scope_options(scopes, groups)
    html = figure.to_html(
        full_html=True,
        include_plotlyjs=True,
        div_id="ert-pooled",
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    path.write_text(
        _inject_ert_menus(
            html,
            scope_options=scope_options,
            budgets=caps,
            shown_budget=shown_budget,
            anchor=anchor,
            curves=curves,
            algorithm_classes=present_classes,
            downloads=downloads,
            problem_sets=groups,
        ),
        encoding="utf-8",
    )
    return path


def _inject_ert_menus(
    html: str,
    *,
    scope_options: list[tuple[str, str]],
    budgets: tuple[int, ...],
    curves: list[dict[str, object]] | tuple[dict[str, object], ...] = (),
    algorithm_classes: tuple[str, ...] = (),
    shown_budget: int | None = None,
    anchor: float = 1.0,
    downloads: bool = True,
    problem_sets: list[dict[str, object]] | None = None,
) -> str:
    """Add menus, algorithm cards, the caption, and the hover card."""
    scope_html = "\n".join(
        f'<option value="{html_escape(scope)}">{html_escape(label)}</option>'
        for scope, label in scope_options
    )
    shown = int(budgets[-1] if shown_budget is None else shown_budget)
    budget_html = "\n".join(
        (
            f'<option value="{budget}"'
            f'{" selected" if int(budget) == shown else ""}>'
            f"{int(budget):,} FE</option>"
        )
        for budget in budgets
    )
    algorithm_html = "\n".join(
        [
            '<option value="all">All algorithms</option>',
            '<option value="top:3">Top 3</option>',
            '<option value="top:5">Top 5</option>',
            '<option value="top:10">Top 10</option>',
        ]
        + [
            f'<option value="class:{html_escape(label)}">{html_escape(label)}</option>'
            for label in algorithm_classes
        ]
    )
    payload = json.dumps(
        {
            "curves": list(curves),
            "classes": list(CLASS_ORDER),
            "anchor": float(anchor),
        }
    )
    n_problems = sum(
        1 for scope, _label in scope_options if not _is_pooled_scope(scope)
    )
    set_menu, set_json = _problem_set_control(problem_sets, "ert-set")
    comparison_style = ' style="margin-left: 16px;"' if set_menu else ""
    description = _ert_description(
        n_problems,
        anchor,
        _curve_local_searches(curves),
        with_sets=bool(problem_sets),
    )
    face = _dejavu_face()
    heading = _ert_heading(
        scope_options[0][0],
        scope_options[0][1],
        shown,
    )
    export_buttons = ""
    export_script = ""
    if downloads:
        export_buttons = (
            '<button type="button" id="ert-png">PNG</button>\n'
            '<button type="button" id="ert-svg">SVG</button>\n'
            '<button type="button" id="ert-table">CSV</button>'
        )
        export_script = """
  document.getElementById("ert-png").addEventListener("click", function () {
    downloadFigure("png");
  });
  document.getElementById("ert-svg").addEventListener("click", function () {
    downloadFigure("svg");
  });
  document.getElementById("ert-table").addEventListener("click", function () {
    downloadText("ert_table.csv", ertCsv(), "text/csv");
  });"""
    controls = f"""
<style>
{face}
html, body {{background:#fff; color:#222;}}
#ert-title, #ert-menus, #ert-menus label, #ert-menus select, #ert-reset,
#ert-caption, #ert-shell, #ert-tip, #ert-pooled text {{
  font-family:'DejaVu Sans', Arial, Helvetica, sans-serif;
}}
#ert-title {{font-size:18px; font-weight:700; line-height:1.35; margin:8px 4px 10px; color:#222;}}
#ert-shell {{display:flex; align-items:flex-start; gap:20px; color:#222;}}
#ert-main {{flex:1 1 auto; min-width:640px; overflow:hidden;}}
#ert-menus {{font-size:13px; margin:8px 0 4px 4px; color:#222;}}
#ert-menus select, #ert-reset, #ert-png, #ert-svg, #ert-table {{font-size:13px; color:#222; background:#fff;}}
#ert-reset, #ert-png, #ert-svg, #ert-table {{margin-left:16px; border:1px solid #bbb; border-radius:2px; padding:1px 8px; cursor:pointer;}}
#ert-caption {{font-size:13px; line-height:1.45; margin:12px 0 16px; width:100%; color:#222;}}
#ert-caption .ert-formula {{margin:8px 0 6px;}}
#ert-caption .formula-note {{margin:0 0 2px;}}
#ert-caption .frac {{display:inline-block; vertical-align:middle; text-align:center; margin:0 4px;}}
#ert-caption .frac .num {{display:block; border-bottom:1px solid #222; padding:0 8px 1px;}}
#ert-caption .frac .den {{display:block; padding-top:1px;}}
#ert-caption .sum {{display:inline-grid; grid-template-rows:auto auto auto; justify-items:center; vertical-align:middle; margin:0 0.35em; line-height:1; row-gap:0.22em;}}
#ert-caption .sum-op {{font-size:1.7em; line-height:1; padding:0 0.12em;}}
#ert-caption .sum-lim {{display:block; font-size:0.68em; line-height:1; white-space:nowrap;}}
#ert-caption .sum-lower {{padding-top:6px; padding-bottom:3px;}}
#ert-caption .sum .sub {{font-size:0.72em; line-height:0; position:relative; top:0.18em;}}
#ert-caption a {{color:#222;}}
#ert-legend {{width:560px; max-height:640px; overflow-y:auto; flex:0 0 560px; background:#fff; position:relative; z-index:2;}}
@media (max-width: 1100px) {{
  #ert-shell {{flex-direction: column;}}
  #ert-main {{min-width:0; width:100%;}}
  #ert-legend {{width:100%; flex:1 1 auto; max-height:420px;}}
}}
#ert-pooled {{width:100% !important; height:560px !important;}}
#ert-pooled .hoverlayer {{display:none !important;}}
.ert-anchor {{color:#666; font-size:11px; margin:0 0 6px;}}
.ert-compact .head, .ert-row {{display:grid; grid-template-columns:minmax(0,1fr) 108px 128px; column-gap:8px; align-items:start;}}
.ert-compact .head {{color:#666; font-size:11px; margin:0 0 6px;}}
.ert-row {{padding:4px 0; font-size:12px;}}
.ert-row .name {{min-width:0;}}
.ert-row .label {{display:inline-block; vertical-align:top; max-width:calc(100% - 16px);}}
.ert-row .handling {{display:block; color:#666; font-size:11px;}}
.dot {{width:8px; height:8px; border-radius:50%; display:inline-block; margin:0 6px 1px 0; vertical-align:middle;}}
.ert-row .pct {{font-weight:700;}}
.fe {{color:#555; font-size:11px;}}
.miss {{color:#888;}}
.ert-key .head {{color:#666; font-size:11px; margin:0 0 6px;}}
.ert-key-item {{display:flex; align-items:center; gap:8px; font-size:12px; padding:3px 0;}}
.ert-style {{display:flex; gap:18px; margin:0 0 10px; padding-bottom:8px; border-bottom:1px solid #ececec;}}
.ert-style-item {{display:flex; align-items:center; gap:8px; font-size:12px;}}
.style-line {{width:28px; border-top:2px solid #111; display:inline-block; flex:0 0 28px;}}
.style-line.dashed {{border:0; height:2px; background:repeating-linear-gradient(to right, #111 0 3px, transparent 3px 5px);}}
.swatch-line {{width:22px; border-top:2px solid; display:inline-block; flex:0 0 22px;}}
#ert-tip .name {{font-weight:700; font-size:13px; margin-bottom:2px;}}
#ert-tip .sub {{color:#555; font-size:11px; margin-bottom:2px;}}
#ert-tip .section {{margin-top:8px; font-size:11px; font-weight:700; color:#555;}}
</style>
<h1 id="ert-title">{heading}</h1>
<div id="ert-menus">
{set_menu}<label{comparison_style}>Comparison <select id="ert-comparison">
<option value="algorithm">Algorithm</option>
<option value="class">Algorithm class</option>
<option value="start">Cold and warm start</option>
</select></label>
<label style="margin-left: 16px;">Algorithms <select id="ert-algorithms">
{algorithm_html}
</select></label>
<label style="margin-left: 16px;">Problems <select id="ert-scope">
{scope_html}
</select></label>
<label style="margin-left: 16px;">Evaluation budget <select id="ert-budget">
{budget_html}
</select></label>
<button type="button" id="ert-reset">Reset view</button>
{export_buttons}
</div>
<div id="ert-shell">
<div id="ert-main"></div>
<div id="ert-legend"></div>
</div>
<div id="ert-caption">{description}</div>
<div id="ert-tip" style="display:none; position:fixed; z-index:20; pointer-events:none; max-width:420px; background:#f7f7f7; color:#222; border:1px solid #d5d5d5; border-left-width:4px; border-radius:2px; padding:8px 10px 8px 12px; font-size:12px; line-height:1.35; box-shadow:0 1px 4px rgba(0,0,0,0.12);"></div>
"""
    script = f"""
<script>
(function () {{
  var payload = {payload};
  var problemSets = {set_json};
  var gd = document.getElementById("ert-pooled");
  var tip = document.getElementById("ert-tip");
  var legend = document.getElementById("ert-legend");
  var main = document.getElementById("ert-main");
  var active = null;
  var timer = null;
  var initialScope = document.getElementById("ert-scope").value;
  var initialBudget = document.getElementById("ert-budget").value;
  var initialXRange = gd && gd.layout && gd.layout.xaxis && gd.layout.xaxis.range
    ? gd.layout.xaxis.range.slice()
    : null;
  if (gd && main) main.appendChild(gd);
  function formatInt(value) {{
    return Math.round(Number(value)).toLocaleString("en-US");
  }}
  function formatErt(value) {{
    if (value == null || !isFinite(Number(value)) || Number(value) <= 0) return "—";
    return formatInt(value) + " FE";
  }}
  function formatPercent(value) {{
    var number = Number(value);
    var digits = Math.abs(number - Math.round(number)) < 1e-6 ? 0 : 1;
    return number.toLocaleString("en-US", {{
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    }}) + "%";
  }}
  function esc(value) {{
    return String(value).replace(/[&<>"]/g, function (ch) {{
      return {{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}}[ch];
    }});
  }}
  function selectedErtSet() {{
    var found = null;
    var menu = document.getElementById("ert-set");
    if (!menu) return null;
    problemSets.forEach(function (item) {{
      if (item.id === menu.value) found = item;
    }});
    return found;
  }}
  function fillErtProblems(setId) {{
    var select = document.getElementById("ert-scope");
    var found = null;
    problemSets.forEach(function (item) {{
      if (item.id === setId) found = item;
    }});
    if (!found) return;
    select.innerHTML = found.options.map(function (item) {{
      return '<option value="' + esc(item.value) + '">' + esc(item.label) + "</option>";
    }}).join("");
  }}
  function selectedBudget() {{
    return Number(document.getElementById("ert-budget").value);
  }}
  function compareText(left, right) {{
    var n = Math.max(left.length, right.length);
    var i;
    for (i = 0; i < n; i++) {{
      var a = left[i] || "";
      var b = right[i] || "";
      if (a < b) return -1;
      if (a > b) return 1;
    }}
    return 0;
  }}
  function anchorRank(curve) {{
    var anchor = Number(payload.anchor);
    var found = -1;
    var i;
    for (i = 0; i < curve.x.length; i++) {{
      if (Number(curve.x[i]) === anchor) {{ found = i; break; }}
    }}
    var row = found < 0 || !curve.custom ? [] : (curve.custom[found] || []);
    var nSuccess = Number(row[0]) || 0;
    var total = Number(curve.meta.n_runs) || 0;
    var rate = total ? nSuccess / total : 0;
    var ert = found < 0 ? null : curve.y[found];
    if (ert == null || !isFinite(Number(ert)) || Number(ert) <= 0 || !nSuccess) ert = Infinity;
    else ert = Number(ert);
    return {{rate: rate, ert: ert}};
  }}
  function ertFamilyRank(curves) {{
    var bestRate = -1;
    var bestErt = Infinity;
    curves.forEach(function (curve) {{
      var rank = anchorRank(curve);
      if (rank.rate > bestRate || (rank.rate === bestRate && rank.ert < bestErt)) {{
        bestRate = rank.rate;
        bestErt = rank.ert;
      }}
    }});
    return {{rate: bestRate < 0 ? 0 : bestRate, ert: bestErt}};
  }}
  function byErtFamily(a, b) {{
    var left = ertFamilyRank(a.curves);
    var right = ertFamilyRank(b.curves);
    if (left.rate !== right.rate) return right.rate - left.rate;
    if (left.ert !== right.ert) return left.ert - right.ert;
    return compareText(
      (a.curves[0].meta && a.curves[0].meta.alpha) || [],
      (b.curves[0].meta && b.curves[0].meta.alpha) || []
    );
  }}
  function startRank(curve) {{
    return curve.meta && curve.meta.start === "warm" ? 1 : 0;
  }}
  function selectedCurves() {{
    var scope = document.getElementById("ert-scope").value;
    var budget = String(selectedBudget());
    var filter = document.getElementById("ert-algorithms").value;
    var groups = [];
    var index = {{}};
    (payload.curves || []).forEach(function (curve) {{
      if (curve.scope !== scope || String(curve.budget) !== budget) return;
      if (!curve.meta || !curve.meta.family) return;
      var family = curve.meta.family;
      if (!(family in index)) {{
        index[family] = groups.length;
        groups.push({{family: family, curves: []}});
      }}
      groups[index[family]].curves.push(curve);
    }});
    groups.sort(byErtFamily);
    if (filter.indexOf("class:") === 0) {{
      var className = filter.slice(6);
      groups = groups.filter(function (group) {{
        return group.curves[0].meta.category === className;
      }});
    }} else if (filter.indexOf("top:") === 0) {{
      groups = groups.slice(0, Number(filter.slice(4)));
    }}
    var selected = [];
    groups.forEach(function (group) {{
      var members = group.curves.slice().sort(function (a, b) {{
        return startRank(a) - startRank(b);
      }});
      selected = selected.concat(members);
    }});
    return selected;
  }}
  function scatterOf(curve) {{
    var mode = document.getElementById("ert-comparison").value;
    var meta = curve.meta;
    var palette = meta.colors || {{}};
    var color = palette[mode] || "#444444";
    var start = meta.start === "warm" ? "Warm start" : "Cold start";
    return {{
      x: curve.x,
      y: curve.y,
      customdata: curve.custom,
      mode: "lines+markers",
      connectgaps: false,
      name: meta.display_name + " · " + start,
      meta: meta,
      line: {{
        color: color,
        width: 2,
        dash: mode === "algorithm" && meta.start === "warm" ? "3px,2px" : "solid"
      }},
      marker: {{size: 7, color: color}},
      hovertemplate: " ",
      hoverlabel: {{
        bgcolor: "rgba(0,0,0,0)",
        bordercolor: "rgba(0,0,0,0)",
        font: {{color: "rgba(0,0,0,0)", size: 1}}
      }},
      showlegend: false
    }};
  }}
  function cursorTrace() {{
    return {{
      x: [null],
      y: [null],
      mode: "markers",
      marker: {{
        symbol: "x",
        size: 12,
        color: "#111111",
        line: {{width: 0.5, color: "#111111"}}
      }},
      hoverinfo: "skip",
      showlegend: false,
      visible: false,
      meta: {{cursor: true}},
      name: ""
    }};
  }}
  function drawCurves() {{
    var data = selectedCurves().map(scatterOf);
    data.push(cursorTrace());
    return Plotly.react(gd, data, gd.layout);
  }}
  function cursorIndex() {{
    return gd.data.length - 1;
  }}
  function hideCursor() {{
    if (!gd || !gd.data || !gd.data.length) return;
    Plotly.restyle(gd, {{visible: false}}, [cursorIndex()]);
  }}
  function showCursor(point) {{
    Plotly.restyle(gd, {{
      x: [[point.x]],
      y: [[point.y]],
      visible: true
    }}, [cursorIndex()]);
  }}
  function isActive(kind, key) {{
    return active && active.kind === kind && active.key === key;
  }}
  function paint(kind, key) {{
    var opacity = [];
    var width = [];
    var indices = [];
    gd.data.forEach(function (trace, index) {{
      if (!trace.meta || !trace.meta.family) return;
      var value = trace.meta.family;
      if (kind === "category") value = trace.meta.category;
      if (kind === "start") value = trace.meta.start;
      var on = !key || value === key;
      opacity.push(on ? 1 : 0.15);
      width.push(key && on ? 3.5 : 2);
      indices.push(index);
    }});
    if (indices.length) {{
      Plotly.restyle(gd, {{opacity: opacity, "line.width": width}}, indices);
    }}
    legend.querySelectorAll("[data-highlight]").forEach(function (row) {{
      var on = !key || (
        row.getAttribute("data-highlight") === kind
        && row.getAttribute("data-key") === key
      );
      row.style.opacity = on ? "1" : "0.45";
    }});
  }}
  function applyColors() {{
    var mode = document.getElementById("ert-comparison").value;
    var colors = [];
    var dashes = [];
    var indices = [];
    gd.data.forEach(function (trace, index) {{
      if (!trace.meta || !trace.meta.family) return;
      var palette = trace.meta.colors;
      var color = palette && palette[mode] ? palette[mode] : "#444444";
      colors.push(color);
      dashes.push(mode === "algorithm" && trace.meta.start === "warm" ? "3px,2px" : "solid");
      indices.push(index);
    }});
    if (indices.length) {{
      Plotly.restyle(gd, {{
        "line.color": colors,
        "line.dash": dashes,
        "marker.color": colors
      }}, indices);
    }}
    if (tip.style.display === "block" && active) {{
      var match = null;
      gd.data.forEach(function (trace) {{
        if (!match && trace.meta && trace.meta.family === active.key && active.kind === "family") {{
          match = trace;
        }}
      }});
      if (match && match.meta.colors && match.meta.colors[mode]) {{
        tip.style.borderLeftColor = match.meta.colors[mode];
      }}
    }}
  }}
  function sourceCurve(trace) {{
    if (!trace || !trace.meta) return null;
    var scope = document.getElementById("ert-scope").value;
    var budget = String(selectedBudget());
    var found = null;
    (payload.curves || []).forEach(function (curve) {{
      if (found || !curve.meta) return;
      if (curve.scope !== scope || String(curve.budget) !== budget) return;
      if (curve.meta.family !== trace.meta.family) return;
      if (curve.meta.start !== trace.meta.start) return;
      found = curve;
    }});
    return found;
  }}
  function readPoint(curve, index) {{
    var row = (curve.custom && curve.custom[index]) || [];
    var ert = curve.y[index];
    if (ert == null || !isFinite(Number(ert)) || Number(ert) <= 0) ert = null;
    else ert = Number(ert);
    return {{
      percent: Number(curve.x[index]),
      ert: ert,
      nSuccess: Number(row[0]) || 0,
      sumHits: Number(row[1]) || 0,
      failureCost: Number(row[2]) || 0,
      failureCap: row.length < 4 || row[3] == null ? null : Number(row[3])
    }};
  }}
  function anchorPoint(trace) {{
    var curve = sourceCurve(trace);
    if (!curve) return null;
    var anchor = Number(payload.anchor);
    for (var i = 0; i < curve.x.length; i++) {{
      if (Number(curve.x[i]) === anchor) return readPoint(curve, i);
    }}
    return null;
  }}
  function resultCell(trace) {{
    if (!trace || !trace.meta) return '<div class="miss">—</div>';
    var total = Number(trace.meta.n_runs) || 0;
    var at = anchorPoint(trace);
    var nSuccess = at ? at.nSuccess : 0;
    var count = '<div class="fe">' + formatInt(nSuccess) + "/" + formatInt(total) + " runs</div>";
    if (!at || at.ert == null) return '<div class="pct miss">—</div>' + count;
    return '<div class="pct">' + formatErt(at.ert) + "</div>" + count;
  }}
  function drawnTraces() {{
    var traces = [];
    gd.data.forEach(function (trace) {{
      if (!trace.meta || !trace.meta.family) return;
      traces.push(trace);
    }});
    return traces;
  }}
  function renderAlgorithmLegend(traces) {{
    var groups = [];
    var index = {{}};
    traces.forEach(function (trace) {{
      var key = trace.meta.family;
      if (!(key in index)) {{
        index[key] = groups.length;
        groups.push({{
          family: key,
          name: trace.meta.display_name,
          qualifier: trace.meta.qualifier || "",
          localSearch: trace.meta.local_search || "",
          handling: trace.meta.handling || "PEN",
          color: (trace.meta.colors && trace.meta.colors.algorithm) || "#444444",
          cold: null,
          warm: null
        }});
      }}
      if (trace.meta.start === "warm") groups[index[key]].warm = trace;
      else groups[index[key]].cold = trace;
    }});
    var anchor = formatPercent(payload.anchor);
    var rows = groups.map(function (group) {{
      var name = group.name;
      if (group.qualifier) name += " (" + group.qualifier + ")";
      var handling = group.handling === "CON" ? "Constraint-aware" : "Penalty-based";
      if (group.localSearch) handling += ", " + group.localSearch;
      return '<div class="ert-row" data-highlight="family" data-key="' + esc(group.family) + '">'
        + '<div class="name"><span class="dot" style="background:' + group.color + ';"></span>'
        + '<span class="label">' + esc(name)
        + '<span class="handling">' + esc(handling) + "</span></span></div>"
        + "<div>" + resultCell(group.cold) + "</div>"
        + "<div>" + resultCell(group.warm) + "</div></div>";
    }}).join("");
    return '<div class="ert-style">'
      + '<span class="ert-style-item"><span class="style-line"></span>Cold start</span>'
      + '<span class="ert-style-item"><span class="style-line dashed"></span>Warm start</span>'
      + '</div><div class="ert-anchor">Expected Running Time at ε<sub>X</sub> = ' + anchor + "</div>"
      + '<div class="ert-compact"><div class="head"><span>Algorithms</span><span>Cold</span><span>Warm</span></div>'
      + rows + "</div>";
  }}
  function renderCategoryLegend(traces) {{
    var colors = {{}};
    traces.forEach(function (trace) {{
      var category = trace.meta.category;
      if (!category || (category in colors)) return;
      colors[category] = (trace.meta.colors && trace.meta.colors.class) || "#444444";
    }});
    var items = (payload.classes || []).filter(function (category) {{
      return category in colors;
    }}).map(function (category) {{
      return '<div class="ert-key-item" data-highlight="category" data-key="' + esc(category) + '">'
        + '<span class="swatch-line" style="border-top-color:' + colors[category] + ';"></span>'
        + esc(category) + "</div>";
    }}).join("");
    return '<div class="ert-key"><div class="head">Algorithm class</div>' + items + "</div>";
  }}
  function renderStartLegend(traces) {{
    var colors = {{}};
    traces.forEach(function (trace) {{
      var start = trace.meta.start;
      if (!start || (start in colors)) return;
      colors[start] = (trace.meta.colors && trace.meta.colors.start) || "#444444";
    }});
    var items = ["cold", "warm"].filter(function (start) {{
      return start in colors;
    }}).map(function (start) {{
      var label = start === "warm" ? "Warm start" : "Cold start";
      return '<div class="ert-key-item" data-highlight="start" data-key="' + start + '">'
        + '<span class="swatch-line" style="border-top-color:' + colors[start] + ';"></span>'
        + label + "</div>";
    }}).join("");
    return '<div class="ert-key"><div class="head">Cold / warm start</div>' + items + "</div>";
  }}
  function renderLegend() {{
    var mode = document.getElementById("ert-comparison").value;
    var traces = drawnTraces();
    if (mode === "class") legend.innerHTML = renderCategoryLegend(traces);
    else if (mode === "start") legend.innerHTML = renderStartLegend(traces);
    else legend.innerHTML = renderAlgorithmLegend(traces);
  }}
  function hideTip() {{
    tip.style.display = "none";
  }}
  function logTickUpdate(values) {{
    if (!values.length) {{
      return {{
        "yaxis.type": "log",
        "yaxis.autorange": false,
        "yaxis.range": [0, 5]
      }};
    }}
    var minValue = Math.min.apply(null, values);
    var maxValue = Math.max.apply(null, values);
    var lo = Math.log10(minValue);
    var hi = Math.log10(maxValue);
    var pad = Math.max(0.08, (hi - lo) * 0.08);
    var range = [lo - pad, hi + pad];
    var labeled = [];
    var text = [];
    var minor = [];
    var exp, factor, value, logValue;
    for (exp = Math.floor(range[0]); exp <= Math.ceil(range[1]); exp++) {{
      for (factor = 0; factor < 4; factor++) {{
        value = [1, 2, 3, 5][factor] * Math.pow(10, exp);
        logValue = Math.log10(value);
        if (logValue < range[0] - 0.02 || logValue > range[1] + 0.02) continue;
        labeled.push(value);
        text.push(formatInt(value));
      }}
      for (factor = 0; factor < 5; factor++) {{
        value = [4, 6, 7, 8, 9][factor] * Math.pow(10, exp);
        logValue = Math.log10(value);
        if (logValue < range[0] || logValue > range[1]) continue;
        minor.push(value);
      }}
    }}
    return {{
      "yaxis.type": "log",
      "yaxis.autorange": false,
      "yaxis.range": range,
      "yaxis.tickmode": "array",
      "yaxis.tickvals": labeled,
      "yaxis.ticktext": text,
      "yaxis.minor.tickmode": "array",
      "yaxis.minor.tickvals": minor,
      "yaxis.minor.ticks": "outside",
      "yaxis.minor.ticklen": 3,
      "yaxis.minor.tickcolor": "#444444",
      "yaxis.minor.showgrid": false
    }};
  }}
  function fitAxes() {{
    var values = [];
    selectedCurves().forEach(function (curve) {{
      (curve.y || []).forEach(function (value) {{
        if (value != null && Number(value) > 0) values.push(Number(value));
      }});
    }});
    return Plotly.relayout(gd, logTickUpdate(values));
  }}
  function writeTitle() {{
    var scope = document.getElementById("ert-scope");
    var pooled = scope.value === "all" || scope.value.slice(-4) === "/all";
    var chosen = selectedErtSet();
    var where = pooled
      ? (chosen && chosen.pooled ? chosen.pooled : "across optimisation problems")
      : "on " + esc(scope.options[scope.selectedIndex].text);
    var mode = document.getElementById("ert-comparison").value;
    var subject = "algorithms";
    if (mode === "class") subject = "algorithm classes";
    else if (mode === "start") subject = "cold and warm start algorithm variants";
    var heading = "Expected Running Time of " + subject + " " + where;
    document.getElementById("ert-title").innerHTML = heading;
    document.title = heading;
  }}
  function apply() {{
    var pending = drawCurves();
    var finish = function () {{
      writeTitle();
      applyColors();
      renderLegend();
      clearHighlight();
      fitAxes();
    }};
    if (pending && pending.then) pending.then(finish);
    else finish();
  }}
  function clearHighlight() {{
    active = null;
    paint(null, null);
    hideTip();
    hideCursor();
  }}
  function showTip(event) {{
    var point = event.points && event.points[0];
    if (!point || !point.data || !point.data.meta || !point.data.meta.family) return;
    var meta = point.data.meta;
    var curve = sourceCurve(point.data);
    if (!curve) return;
    var index = point.pointNumber;
    if (index == null) index = point.pointIndex;
    var info = readPoint(curve, index);
    if (!info || info.ert == null) return;
    var mode = document.getElementById("ert-comparison").value;
    var color = (meta.colors && meta.colors[mode]) || "#444444";
    var total = Number(meta.n_runs) || 0;
    var nFail = total - info.nSuccess;
    var added = info.failureCap == null || !nFail
      ? formatInt(info.failureCost)
      : formatInt(nFail) + " × " + formatInt(info.failureCap);
    var label = meta.start === "warm" ? "ERT<sub>warm</sub>" : "ERT<sub>cold</sub>";
    var formula = label + " = (" + formatInt(info.sumHits) + " + " + added + ") / "
      + formatInt(info.nSuccess);
    if (meta.start === "warm") formula += " + 2,700";
    formula += " = " + formatErt(info.ert);
    tip.style.borderLeftColor = color;
    tip.innerHTML =
      '<div class="name">' + esc(meta.display_name) + "</div>" +
      '<div class="sub">' + esc(meta.subtitle) + "</div>" +
      '<div class="section">Input values</div>' +
      "<div>Selected evaluation budget: " + formatInt(selectedBudget()) + " FE</div>" +
      "<div>Unsuccessful runs contribute the evaluations they performed.</div>" +
      "<div>Relative target tolerance ε<sub>X</sub>: " + formatPercent(info.percent) + "</div>" +
      '<div class="section">Expected Running Time</div>' +
      "<div>" + formula + "</div>" +
      "<div>Successful runs: " + formatInt(info.nSuccess) + "/" + formatInt(total) + "</div>";
    tip.style.display = "block";
    var mouse = event.event || {{}};
    var x = (mouse.clientX || 0) + 18;
    var y = (mouse.clientY || 0) + 18;
    if (x + tip.offsetWidth > window.innerWidth - 8) {{
      x = (mouse.clientX || 0) - tip.offsetWidth - 18;
    }}
    if (y + tip.offsetHeight > window.innerHeight - 8) {{
      y = (mouse.clientY || 0) - tip.offsetHeight - 18;
    }}
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
    showCursor(point);
  }}
  function resetView() {{
    var setMenu = document.getElementById("ert-set");
    if (setMenu) {{
      setMenu.selectedIndex = 0;
      fillErtProblems(setMenu.value);
    }}
    document.getElementById("ert-scope").value = initialScope;
    document.getElementById("ert-algorithms").value = "all";
    document.getElementById("ert-budget").value = initialBudget;
    document.getElementById("ert-comparison").value = "algorithm";
    if (initialXRange && gd.layout && gd.layout.xaxis) {{
      gd.layout.xaxis.autorange = false;
      gd.layout.xaxis.range = initialXRange.slice();
    }}
    apply();
  }}
  var ertSet = document.getElementById("ert-set");
  if (ertSet) {{
    ertSet.addEventListener("change", function () {{
      fillErtProblems(ertSet.value);
      apply();
    }});
  }}
  document.getElementById("ert-scope").addEventListener("change", apply);
  document.getElementById("ert-algorithms").addEventListener("change", apply);
  document.getElementById("ert-budget").addEventListener("change", apply);
  document.getElementById("ert-reset").addEventListener("click", resetView);
  function csvCell(value) {{
    var text = String(value == null ? "" : value);
    if (text.indexOf(",") >= 0 || text.indexOf('"') >= 0) {{
      return '"' + text.replace(/"/g, '""') + '"';
    }}
    return text;
  }}
  function ertCsv() {{
    var scope = document.getElementById("ert-scope");
    var problem = scope.options[scope.selectedIndex].text;
    var budget = selectedBudget();
    var lines = ["problem,budget,tolerance_percent,algorithm,start,handling,n_runs,n_success,sum_hits,failure_cost,ert"];
    drawnTraces().forEach(function (trace) {{
      var curve = sourceCurve(trace);
      var meta = trace.meta;
      if (!curve || !meta) return;
      var handling = meta.handling === "CON" ? "Constraint-aware" : "Penalty-based";
      if (meta.local_search) handling += ", " + meta.local_search;
      var algorithm = meta.display_name;
      if (meta.qualifier) algorithm += " (" + meta.qualifier + ")";
      var start = meta.start === "warm" ? "warm" : "cold";
      curve.x.forEach(function (_percent, index) {{
        var info = readPoint(curve, index);
        lines.push([
          csvCell(problem), budget, info.percent, csvCell(algorithm), start,
          csvCell(handling), meta.n_runs, info.nSuccess, info.sumHits,
          info.failureCost, info.ert == null ? "" : Math.round(info.ert)
        ].join(","));
      }});
    }});
    return lines.join(String.fromCharCode(10));
  }}
  function downloadText(filename, text, type) {{
    var blob = new Blob([text], {{type: type}});
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }}
  function resultText(trace) {{
    if (!trace || !trace.meta) return "—";
    var total = Number(trace.meta.n_runs) || 0;
    var at = anchorPoint(trace);
    var nSuccess = at ? at.nSuccess : 0;
    var count = formatInt(nSuccess) + "/" + formatInt(total) + " runs";
    if (!at || at.ert == null) return "—  " + count;
    return formatErt(at.ert) + "  " + count;
  }}
  function exportRows() {{
    var mode = document.getElementById("ert-comparison").value;
    var traces = drawnTraces();
    var rows = [];
    if (mode === "class" || mode === "start") {{
      var colors = {{}};
      traces.forEach(function (trace) {{
        var key = mode === "class" ? trace.meta.category : trace.meta.start;
        if (!key || (key in colors)) return;
        colors[key] = (trace.meta.colors && trace.meta.colors[mode]) || "#444444";
      }});
      var keys = mode === "start" ? ["cold", "warm"] : (payload.classes || []);
      keys.forEach(function (key) {{
        if (!(key in colors)) return;
        var label = key === "warm" ? "Warm start" : key === "cold" ? "Cold start" : key;
        rows.push({{kind: "swatch", color: colors[key], label: label}});
      }});
      return rows;
    }}
    rows.push({{kind: "style"}});
    var groups = [];
    var index = {{}};
    traces.forEach(function (trace) {{
      var key = trace.meta.family;
      if (!(key in index)) {{
        index[key] = groups.length;
        var name = trace.meta.display_name;
        if (trace.meta.qualifier) name += " (" + trace.meta.qualifier + ")";
        groups.push({{
          name: name,
          handling: (trace.meta.handling === "CON" ? "Constraint-aware" : "Penalty-based") + (trace.meta.local_search ? ", " + trace.meta.local_search : ""),
          color: (trace.meta.colors && trace.meta.colors.algorithm) || "#444444",
          cold: null,
          warm: null
        }});
      }}
      if (trace.meta.start === "warm") groups[index[key]].warm = trace;
      else groups[index[key]].cold = trace;
    }});
    groups.forEach(function (group) {{
      rows.push({{
        kind: "algorithm",
        name: group.name,
        handling: group.handling,
        color: group.color,
        cold: resultText(group.cold),
        warm: resultText(group.warm)
      }});
    }});
    return rows;
  }}
  function exportLegendHeight(rows) {{
    var height = 28;
    rows.forEach(function (row) {{
      height += row.kind === "algorithm" ? 36 : 22;
    }});
    return height;
  }}
  function paintExportLegend(ctx, left, top, rows) {{
    ctx.font = "12px 'DejaVu Sans', sans-serif";
    ctx.textBaseline = "middle";
    var y = top;
    rows.forEach(function (row) {{
      if (row.kind === "style") {{
        ctx.strokeStyle = "#111111";
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(left, y + 8);
        ctx.lineTo(left + 28, y + 8);
        ctx.stroke();
        ctx.fillStyle = "#222222";
        ctx.fillText("Cold start", left + 36, y + 8);
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(left + 130, y + 8);
        ctx.lineTo(left + 158, y + 8);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillText("Warm start", left + 166, y + 8);
        y += 22;
        return;
      }}
      if (row.kind === "swatch") {{
        ctx.strokeStyle = row.color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(left, y + 8);
        ctx.lineTo(left + 22, y + 8);
        ctx.stroke();
        ctx.fillStyle = "#222222";
        ctx.fillText(row.label, left + 30, y + 8);
        y += 22;
        return;
      }}
      ctx.fillStyle = row.color;
      ctx.fillRect(left, y + 4, 8, 8);
      ctx.fillStyle = "#222222";
      ctx.font = "12px 'DejaVu Sans', sans-serif";
      ctx.fillText(row.name, left + 16, y + 8);
      ctx.fillStyle = "#666666";
      ctx.font = "11px 'DejaVu Sans', sans-serif";
      ctx.fillText(row.handling, left + 16, y + 22);
      ctx.fillStyle = "#222222";
      ctx.font = "12px 'DejaVu Sans', sans-serif";
      ctx.fillText(row.cold, left + 230, y + 22);
      ctx.fillText(row.warm, left + 400, y + 22);
      y += 36;
    }});
  }}
  function svgEscape(value) {{
    return String(value).replace(/[&<>"]/g, function (ch) {{
      return {{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}}[ch];
    }});
  }}
  function legendSvg(rows, left) {{
    var parts = [];
    var y = 16;
    rows.forEach(function (row) {{
      if (row.kind === "style") {{
        parts.push('<line x1="' + left + '" y1="' + (y + 8) + '" x2="' + (left + 28) + '" y2="' + (y + 8) + '" stroke="#111111" stroke-width="2"/>');
        parts.push('<text x="' + (left + 36) + '" y="' + (y + 12) + '" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#222222">Cold start</text>');
        parts.push('<line x1="' + (left + 130) + '" y1="' + (y + 8) + '" x2="' + (left + 158) + '" y2="' + (y + 8) + '" stroke="#111111" stroke-width="2" stroke-dasharray="3 2"/>');
        parts.push('<text x="' + (left + 166) + '" y="' + (y + 12) + '" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#222222">Warm start</text>');
        y += 22;
        return;
      }}
      if (row.kind === "swatch") {{
        parts.push('<line x1="' + left + '" y1="' + (y + 8) + '" x2="' + (left + 22) + '" y2="' + (y + 8) + '" stroke="' + row.color + '" stroke-width="3"/>');
        parts.push('<text x="' + (left + 30) + '" y="' + (y + 12) + '" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#222222">' + svgEscape(row.label) + "</text>");
        y += 22;
        return;
      }}
      parts.push('<rect x="' + left + '" y="' + (y + 4) + '" width="8" height="8" fill="' + row.color + '"/>');
      parts.push('<text x="' + (left + 16) + '" y="' + (y + 12) + '" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#222222">' + svgEscape(row.name) + "</text>");
      parts.push('<text x="' + (left + 16) + '" y="' + (y + 26) + '" font-family="DejaVu Sans, sans-serif" font-size="11" fill="#666666">' + svgEscape(row.handling) + "</text>");
      parts.push('<text x="' + (left + 230) + '" y="' + (y + 26) + '" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#222222">' + svgEscape(row.cold) + "</text>");
      parts.push('<text x="' + (left + 400) + '" y="' + (y + 26) + '" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#222222">' + svgEscape(row.warm) + "</text>");
      y += 36;
    }});
    return parts.join("");
  }}
  function svgFromDataUrl(url) {{
    var comma = url.indexOf(",");
    var meta = url.slice(0, comma);
    var body = url.slice(comma + 1);
    return meta.indexOf("base64") === -1 ? decodeURIComponent(body) : atob(body);
  }}
  function downloadFigure(format) {{
    var plotW = 1100;
    var plotH = 560;
    var legendW = 620;
    var rows = exportRows();
    var height = Math.max(plotH, exportLegendHeight(rows) + 16);
    if (format === "svg") {{
      Plotly.toImage(gd, {{format: "svg", width: plotW, height: plotH}}).then(function (url) {{
        var doc = new DOMParser().parseFromString(svgFromDataUrl(url), "image/svg+xml");
        var ns = "http://www.w3.org/2000/svg";
        var plot = doc.documentElement;
        var totalW = plotW + legendW;
        plot.setAttribute("x", "0");
        plot.setAttribute("y", "0");
        plot.setAttribute("width", String(plotW));
        plot.setAttribute("height", String(plotH));
        var outer = doc.createElementNS(ns, "svg");
        outer.setAttribute("xmlns", ns);
        outer.setAttribute("width", String(totalW));
        outer.setAttribute("height", String(height));
        outer.setAttribute("viewBox", "0 0 " + totalW + " " + height);
        var backdrop = doc.createElementNS(ns, "rect");
        backdrop.setAttribute("width", String(totalW));
        backdrop.setAttribute("height", String(height));
        backdrop.setAttribute("fill", "#ffffff");
        var key = doc.createElementNS(ns, "g");
        key.innerHTML = legendSvg(rows, plotW + 16);
        outer.appendChild(backdrop);
        outer.appendChild(plot);
        outer.appendChild(key);
        downloadText("ert_pooled.svg", new XMLSerializer().serializeToString(outer), "image/svg+xml");
      }});
      return;
    }}
    Plotly.toImage(gd, {{format: "png", width: plotW, height: plotH}}).then(function (url) {{
      var image = new Image();
      image.onload = function () {{
        var canvas = document.createElement("canvas");
        canvas.width = plotW + legendW;
        canvas.height = height;
        var ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0);
        paintExportLegend(ctx, plotW + 16, 16, rows);
        var link = document.createElement("a");
        link.href = canvas.toDataURL("image/png");
        link.download = "ert_pooled.png";
        link.click();
      }};
      image.src = url;
    }});
  }}
{export_script}
  document.getElementById("ert-comparison").addEventListener("change", function () {{
    applyColors();
    renderLegend();
    writeTitle();
    clearHighlight();
  }});
  legend.addEventListener("mouseover", function (event) {{
    var row = event.target.closest("[data-highlight]");
    if (!row) return;
    if (timer) {{ clearTimeout(timer); timer = null; }}
    var kind = row.getAttribute("data-highlight");
    var key = row.getAttribute("data-key");
    if (key && !isActive(kind, key)) {{
      active = {{kind: kind, key: key}};
      paint(kind, key);
    }}
  }});
  legend.addEventListener("mouseout", function (event) {{
    var row = event.target.closest("[data-highlight]");
    if (!row || row.contains(event.relatedTarget)) return;
    if (timer) clearTimeout(timer);
    timer = setTimeout(function () {{
      timer = null;
      active = null;
      paint(null, null);
    }}, 60);
  }});
  if (gd && gd.on) {{
    gd.on("plotly_hover", function (event) {{
      if (timer) {{ clearTimeout(timer); timer = null; }}
      var point = event.points && event.points[0];
      var family = point && point.data && point.data.meta && point.data.meta.family;
      if (family && !isActive("family", family)) {{
        active = {{kind: "family", key: family}};
        paint("family", family);
      }}
      showTip(event);
    }});
    gd.on("plotly_unhover", function () {{
      hideTip();
      hideCursor();
      if (timer) clearTimeout(timer);
      timer = setTimeout(function () {{
        timer = null;
        if (active === null) return;
        active = null;
        paint(null, null);
      }}, 60);
    }});
  }}
  apply();
  if (document.fonts && document.fonts.ready && gd && window.Plotly) {{
    document.fonts.ready.then(function () {{
      Plotly.relayout(gd, {{"font.family": "DejaVu Sans"}});
    }});
  }}
}})();
</script>
"""
    marker = '<div id="ert-pooled"'
    if marker not in html:
        raise ValueError("interactive ERT html is missing the plot div")
    html = html.replace(marker, controls + marker, 1)
    html = html.replace(
        '<head><meta charset="utf-8" /></head>',
        '<head><meta charset="utf-8" /><title>'
        + html_escape(heading)
        + "</title></head>",
        1,
    )
    if "</body>" in html:
        return html.replace("</body>", script + "</body>", 1)
    return html + script
