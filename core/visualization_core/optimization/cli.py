"""
CLI for algorithm-benchmarking optimisation result plots.

Console script (``pyproject.toml``):

``hdb-plot = core.visualization_core.optimization.cli:main``

Equivalent module invocation:

``python -m core.visualization_core.optimization.cli ...``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, TextIO

from core.visualization_core.optimization.api import (
    DEFAULT_PLOTS,
    PlotApiError,
    render_suite,
    render_suite_compare,
)
from core.visualization_core.optimization.io import ExperimentLoadError, load_experiments
from core.visualization_core.optimization.models import PlotConfig
from core.visualization_core.optimization.plots.interactive_pool import (
    INTERACTIVE_PLOTS,
    render_interactive_pool,
    selected_interactive_plots,
)
from core.visualization_core.optimization.trajectories import TrajectoryError


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that exits with code 1 on usage errors."""

    def error(self, message: str) -> None:  # noqa: D102
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def _add_shared_suite_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--exp",
        action="append",
        dest="experiments",
        required=True,
        metavar="DIR",
        help=(
            "Experiment directory (repeat for overlays / compare panels). "
            "Required at least once."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output directory for PNG and PDF figures",
    )
    parser.add_argument(
        "--problem-id",
        default=None,
        metavar="ID",
        help=(
            "Select one problem from a multi-problem experiment. Required "
            "for suite plots when loaded results contain multiple problems."
        ),
    )
    parser.add_argument(
        "--display",
        choices=("median", "mean", "best", "all"),
        default="median",
        help="Convergence aggregation mode (default: median)",
    )
    parser.add_argument(
        "--success-filter",
        choices=("successes", "all_runs"),
        default="successes",
        dest="success_filter",
        help="Runtime sample filter for box/hist/ECDF/violin (default: successes)",
    )
    parser.add_argument(
        "--plots",
        default=",".join(DEFAULT_PLOTS),
        help=(
            "Comma-separated plot names from "
            f"{','.join(DEFAULT_PLOTS)} (default: all)"
        ),
    )
    parser.add_argument(
        "--global-optimum",
        type=float,
        default=None,
        dest="global_optimum",
        help=(
            "Override / supply known optimum for fixed-target plots "
            "(suite: one value for the whole overlay; compare: use "
            "--global-optimum once per --exp, or omit to use sealed values)"
        ),
    )
    parser.add_argument(
        "--target-reference",
        choices=("all",),
        default=None,
        dest="target_reference",
        help="Hitting times use every finite y_p (only value: all)",
    )
    parser.add_argument(
        "--extend-short-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="extend_short_runs",
        help=(
            "Carry forward short/early-stopped runs to the longest trajectory "
            "(default: on). Use --no-extend-short-runs to truncate at B_common."
        ),
    )
    parser.add_argument(
        "--show-iqr",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="show_iqr",
        help="Draw median IQR ribbon on convergence (default: on)",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        dest="y_max",
        help=(
            "Optional convergence y-axis upper limit; percent of the global "
            "optimum when known (100%% = optimum, default top 300), else "
            "absolute y_p"
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        dest="exclude_algorithms",
        default=None,
        metavar="ALGORITHM",
        help=(
            "Algorithm to omit (repeat, or pass a comma-separated list). "
            "Applied before the completed-repetition check."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build the ``hdb-plot`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Root parser with ``suite``, ``compare``, and ``interactive``
        subcommands. The interactive command writes the success heatmap.
    """
    parser = _ArgumentParser(
        prog="hdb-plot",
        description=(
            "Plot algorithm-benchmarking results from one or more finished "
            "optimization_runs experiment directories."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    suite = sub.add_parser(
        "suite",
        help="Render the convergence / ERT / runtime plot suite (overlays)",
    )
    _add_shared_suite_args(suite)

    compare = sub.add_parser(
        "compare",
        help=(
            "Faceted multi-study / multi-problem comparison "
            "(one panel per --exp; does not overlay into a single axes)"
        ),
    )
    _add_shared_suite_args(compare)
    compare.add_argument(
        "--panel-title",
        action="append",
        dest="panel_titles",
        default=None,
        metavar="TEXT",
        help="Panel title (repeat once per --exp; default: directory name)",
    )
    compare.add_argument(
        "--title",
        default=None,
        help="Optional figure-level title for compare plots",
    )
    interactive = sub.add_parser(
        "interactive",
        help=(
            "Hover-only HTML plots from one load: convergence, ECDF, "
            "expected runtime, and the success heatmap"
        ),
    )
    interactive.add_argument(
        "--exp",
        action="append",
        dest="experiments",
        required=True,
        metavar="DIR",
        help="Experiment directory (repeat to pool problems)",
    )
    interactive.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help=(
            "Directory for the HTML files. Omit it to use "
            "{data root}/results/optimization/<experiment.id>, "
            "or <id1>__<id2> when several experiments are pooled"
        ),
    )
    interactive.add_argument(
        "--plots",
        default=",".join(INTERACTIVE_PLOTS),
        help=(
            "Comma-separated names from "
            f"{', '.join(INTERACTIVE_PLOTS)} (default: all). "
            "convergence includes the pooled figure and both per-problem figures."
        ),
    )
    interactive.add_argument(
        "--percent",
        type=float,
        default=1.0,
        help="Success-ECDF target percent above each run's sealed optimum (default: 1)",
    )
    interactive.add_argument(
        "--downloads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show PNG, SVG, and data buttons on the ECDF, expected-runtime, "
            "and heatmap figures. Pass --no-downloads to leave them out "
            "(default: show them)."
        ),
    )
    interactive.add_argument(
        "--exclude",
        action="append",
        dest="exclude_algorithms",
        default=None,
        metavar="ALGORITHM",
        help=(
            "Algorithm to omit from every interactive figure "
            "(repeat, or pass a comma-separated list)"
        ),
    )

    compare.add_argument(
        "--global-optimum-per-exp",
        action="append",
        dest="global_optima",
        default=None,
        type=float,
        metavar="YSTAR",
        help=(
            "Per-panel global optimum (repeat once per --exp). "
            "Overrides sealed values when given."
        ),
    )
    return parser


def _parse_plot_names(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        raise ValueError("--plots must name at least one plot")
    return names


def _exclude_names(raw: list[str] | None) -> tuple[str, ...]:
    """Split repeated and comma-separated ``--exclude`` values."""
    if not raw:
        return ()
    names: list[str] = []
    for item in raw:
        names.extend(part.strip() for part in str(item).split(",") if part.strip())
    return tuple(dict.fromkeys(names))


class _StatusReporter:
    """One updating status line on a terminal, or occasional lines otherwise."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._interactive = bool(stream.isatty())
        self._last_length = 0
        self._last_bucket = -1
        self._last_message = ""
        self._open = False

    def __call__(self, done: int, total: int, message: str) -> None:
        text = f"hdb-plot: {message} ({done}/{total})"
        if self._interactive:
            padded = text.ljust(self._last_length)
            print("\r" + padded, end="", file=self._stream, flush=True)
            self._last_length = len(text)
            self._open = True
            return
        if total <= 0:
            return
        head = message.split(" ", 1)[0]
        bucket = (done * 20) // total
        if head == "loading":
            emit = done in (0, total) or bucket != self._last_bucket
        else:
            emit = message != self._last_message
        if not emit:
            return
        print(text, file=self._stream, flush=True)
        self._last_bucket = bucket
        self._last_message = message

    def finish(self) -> None:
        """End an in-place status line so the next message starts clean."""
        if self._interactive and self._open:
            print(file=self._stream, flush=True)
            self._open = False
            self._last_length = 0


def _base_config(args: argparse.Namespace, *, global_optimum: float | None) -> PlotConfig:
    return PlotConfig(
        display=args.display,
        success_filter=args.success_filter,
        global_optimum=global_optimum,
        target_reference=args.target_reference,
        extend_short_runs=bool(args.extend_short_runs),
        show_iqr=bool(args.show_iqr),
        y_max=args.y_max,
    )


def run_suite(args: argparse.Namespace) -> int:
    """
    Load experiments and render the selected plot suite.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``suite`` arguments.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on user/data errors).
    """
    try:
        which = _parse_plot_names(str(args.plots))
        bundle = load_experiments([Path(p) for p in args.experiments])
        config = _base_config(args, global_optimum=args.global_optimum)
        written = render_suite(
            bundle,
            config,
            Path(args.out),
            which=which,
            exclude_algorithms=_exclude_names(args.exclude_algorithms),
            problem_id=None if args.problem_id is None else str(args.problem_id),
        )
    except (ExperimentLoadError, TrajectoryError, PlotApiError, ValueError) as exc:
        print(f"hdb-plot: error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


def run_compare(args: argparse.Namespace) -> int:
    """
    Load experiments and render faceted multi-problem comparison plots.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``compare`` arguments.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on user/data errors).
    """
    try:
        which = _parse_plot_names(str(args.plots))
        exp_dirs = [Path(p) for p in args.experiments]
        panel_titles = list(args.panel_titles or [])
        if panel_titles and len(panel_titles) != len(exp_dirs):
            raise ValueError(
                "--panel-title must be omitted or repeated once per --exp"
            )
        optima = list(args.global_optima or [])
        if optima and len(optima) != len(exp_dirs):
            raise ValueError(
                "--global-optimum-per-exp must be omitted or repeated once per --exp"
            )
        if args.global_optimum is not None and optima:
            raise ValueError(
                "use either --global-optimum or --global-optimum-per-exp, not both"
            )

        panels = []
        for index, exp_dir in enumerate(exp_dirs):
            bundle = load_experiments([exp_dir])
            optimum_runs = bundle.runs
            if args.problem_id is not None:
                selected = tuple(
                    run
                    for run in bundle.runs
                    if run.problem_id == str(args.problem_id)
                )
                if selected:
                    optimum_runs = selected
            if optima:
                y_star = float(optima[index])
            elif args.global_optimum is not None:
                y_star = float(args.global_optimum)
            elif optimum_runs and optimum_runs[0].global_optimum is not None:
                y_star = float(optimum_runs[0].global_optimum)
            else:
                y_star = None
            config = _base_config(args, global_optimum=y_star)
            title = (
                panel_titles[index]
                if panel_titles
                else exp_dir.name
            )
            panels.append((title, bundle, config))

        written = render_suite_compare(
            panels,
            Path(args.out),
            which=which,
            title=args.title,
            exclude_algorithms=_exclude_names(args.exclude_algorithms),
            problem_id=None if args.problem_id is None else str(args.problem_id),
        )
    except (ExperimentLoadError, TrajectoryError, PlotApiError, ValueError) as exc:
        print(f"hdb-plot: error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


def run_interactive(args: argparse.Namespace) -> int:
    """
    Write pooled and per-problem hover-only HTML plots.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``interactive`` arguments.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on user/data errors).
    """
    status = _StatusReporter(sys.stderr)
    try:
        which = selected_interactive_plots(_parse_plot_names(str(args.plots)))
        exp_dirs = [Path(p) for p in args.experiments]
        if args.out is None:
            from core.execution_core.output_layout import (
                LayoutError,
                default_optimization_plot_dir,
            )

            try:
                out_dir = default_optimization_plot_dir(exp_dirs)
            except LayoutError as exc:
                raise ValueError(str(exc)) from exc
        else:
            out_dir = Path(args.out)
        bundle = load_experiments(exp_dirs, progress=status)
        written = render_interactive_pool(
            bundle,
            out_dir,
            percent=float(args.percent),
            extend_short_runs=True,
            exclude_algorithms=_exclude_names(args.exclude_algorithms),
            progress=status,
            downloads=bool(args.downloads),
            which=which,
        )
    except (ExperimentLoadError, TrajectoryError, ValueError) as exc:
        status.finish()
        print(f"hdb-plot: error: {exc}", file=sys.stderr)
        return 2
    status.finish()
    for path in written:
        print(path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entry point.

    Parameters
    ----------
    argv : sequence of str or None
        Argument vector; ``None`` uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "suite":
        return run_suite(args)
    if args.command == "compare":
        return run_compare(args)
    if args.command == "interactive":
        return run_interactive(args)
    parser.error(f"unknown command {args.command!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
