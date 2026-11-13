"""Pooled interactive plots: one series per algorithm, per-run optima."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from core.visualization_core.optimization.aggregate import success_probability_curve
from core.visualization_core.optimization.models import ExperimentBundle, RunTrajectory
from core.visualization_core.optimization.plots.interactive_pool import (
    _POOLED_ECDF_PERCENTS,
    _ecdf_legend_layout,
    _inject_convergence_menus,
    _inject_ecdf_menus,
    downsample_curve,
    ecdf_algorithm_order,
    problem_label,
    relative_median_percent,
    relative_percent_curve,
    runs_by_algorithm,
    runs_by_experiment,
    selected_trace_mask,
    selected_view_mask,
    without_excluded_algorithms,
)


def _family(name: str) -> str:
    suffix = "_WARM_START"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _run(
    name: str,
    problem: str,
    y_star: float,
    curve: list[float],
    *,
    task_index: int,
) -> RunTrajectory:
    values = np.asarray(curve, dtype=float)
    n = int(values.shape[0])
    return RunTrajectory(
        experiment_dir=Path(problem),
        overlay_label=f"{name} [{problem}]",
        algorithm_name=name,
        problem_id=problem,
        task_index=task_index,
        repetition=0,
        evaluations=np.arange(1, n + 1),
        y_p=values,
        y_p_best_so_far_all=np.minimum.accumulate(values),
        constraint_utilizations={},
        feasible_mask=np.ones(n, dtype=bool),
        max_evaluations=n,
        global_optimum=y_star,
        target_reference="feasible",
        target_hit=None,
        evaluations_to_target=None,
        wall_time_s=None,
        dat_path=Path(problem) / f"{task_index}.dat",
    )


class TestInteractivePool(unittest.TestCase):
    def test_groups_by_algorithm_across_problems(self) -> None:
        bundle = ExperimentBundle(
            runs=(
                _run("A", "p1", 100.0, [200.0, 110.0], task_index=1),
                _run("A", "p2", 400.0, [800.0, 440.0], task_index=2),
                _run("B", "p1", 100.0, [150.0, 100.0], task_index=3),
            ),
            experiment_dirs=(Path("p1"), Path("p2")),
        )
        grouped = runs_by_algorithm(bundle)
        self.assertEqual(set(grouped), {"A", "B"})
        self.assertEqual(len(grouped["A"]), 2)

    def test_splits_one_grouping_per_experiment(self) -> None:
        bundle = ExperimentBundle(
            runs=(
                _run("A", "p1", 100.0, [100.0], task_index=1),
                _run("B", "p1", 100.0, [110.0], task_index=2),
                _run("A", "p2", 400.0, [400.0], task_index=3),
            ),
            experiment_dirs=(Path("p1"), Path("p2")),
        )
        slices = runs_by_experiment(bundle)
        self.assertEqual([path.name for path, _grouped in slices], ["p1", "p2"])
        self.assertEqual(set(slices[0][1]), {"A", "B"})
        self.assertEqual(len(slices[0][1]["A"]), 1)
        self.assertEqual(set(slices[1][1]), {"A"})

    def test_relative_median_uses_each_runs_own_optimum(self) -> None:
        runs = (
            _run("A", "p1", 100.0, [200.0, 200.0], task_index=1),
            _run("A", "p2", 400.0, [800.0, 800.0], task_index=2),
        )
        _evaluations, median, q25, q75 = relative_median_percent(runs)
        self.assertTrue(np.allclose(median, 200.0))
        self.assertTrue(np.allclose(q25, 200.0))
        self.assertTrue(np.allclose(q75, 200.0))

    def test_one_run_is_scaled_by_its_own_optimum(self) -> None:
        evaluations, percent = relative_percent_curve(
            _run("A", "p1", 100.0, [200.0, 150.0], task_index=1)
        )
        self.assertTrue(np.array_equal(evaluations, np.asarray([1, 2])))
        self.assertTrue(np.allclose(percent, [200.0, 150.0]))

    def test_downsample_keeps_endpoints(self) -> None:
        evaluations = np.arange(1, 1001)
        values = np.linspace(300.0, 100.0, 1000)
        kept_x, kept_y = downsample_curve(evaluations, values, max_points=50)
        self.assertLessEqual(kept_x.shape[0], 50)
        self.assertEqual(int(kept_x[0]), 1)
        self.assertEqual(int(kept_x[-1]), 1000)
        self.assertAlmostEqual(float(kept_y[0]), 300.0)
        self.assertAlmostEqual(float(kept_y[-1]), 100.0)

    def test_ecdf_menus_are_inserted_above_the_plot(self) -> None:
        html = _inject_ecdf_menus(
            '<html><body><div id="ecdf-pooled"></div></body></html>',
            scope_options=[("all", "all problems"), ("L6750_B2", "L6750_B2")],
            percents=(1.0, 5.0, 10.0),
            algorithm_classes=("Metaheuristic", "Direct search"),
        )
        self.assertLess(html.index('id="ecdf-comparison"'), html.index('id="ecdf-scope"'))
        self.assertLess(html.index('id="ecdf-scope"'), html.index('id="ecdf-margin"'))
        self.assertLess(html.index('id="ecdf-margin"'), html.index('id="ecdf-budget"'))
        self.assertLess(html.index('id="ecdf-budget"'), html.index('id="ecdf-algorithms"'))
        self.assertIn('value="2000"', html)
        self.assertIn('value="15000" selected', html)
        self.assertLess(html.index('id="ecdf-title"'), html.index('id="ecdf-menus"'))
        self.assertLess(html.index('id="ecdf-menus"'), html.index('id="ecdf-pooled"'))
        self.assertIn("Cumulative success rate of algorithms across optimisation problems", html)
        self.assertIn("style-line dashed", html)
        self.assertIn(">Cold start</span>", html)
        self.assertIn(">Warm start</span>", html)
        self.assertIn(">All algorithms</option>", html)
        self.assertIn('value="top:3"', html)
        self.assertIn('value="top:5"', html)
        self.assertIn('value="top:10"', html)
        self.assertIn(">Metaheuristic</option>", html)
        self.assertIn(">Direct search</option>", html)
        self.assertNotIn(">Sampling</option>", html)
        self.assertGreater(
            html.index('getElementById("ecdf-pooled")'),
            html.index('id="ecdf-pooled"'),
        )
        self.assertIn("all problems", html)
        self.assertIn(">1%</option>", html)
        self.assertIn(">10%</option>", html)
        self.assertEqual(_POOLED_ECDF_PERCENTS, (1.0, 2.0, 3.0, 4.0, 5.0, 10.0))
        self.assertIn("Plotly.react(gd, data, gd.layout)", html)
        self.assertIn('id="ecdf-png"', html)
        self.assertIn('id="ecdf-svg"', html)
        self.assertIn('id="ecdf-hits"', html)
        self.assertIn("Plotly.toImage", html)
        self.assertIn("paintExportLegend", html)
        self.assertIn("problem,run_id,tolerance_percent", html)
        self.assertIn("meta.hits", html)
        self.assertIn(">Problems <select", html)
        self.assertIn(">Evaluation budget <select", html)
        self.assertIn("Definitions", html)
        self.assertIn("earliest evaluation at which that rate is reached", html)
        self.assertIn("shows this proportion as a percentage", html)
        self.assertNotIn("ECDF<sub>s</sub>(n)", html)
        self.assertNotIn("indicator function", html)
        self.assertIn("minimum stored penalised objective", html)
        target_item = html.index("<b>Relative target tolerance.</b>")
        self.assertLess(
            html.index('class="formula-note"', target_item),
            html.index("</li>", target_item),
        )
        self.assertIn('class="ecdf-meta"', html)
        self.assertIn('class="ecdf-tolerance"', html)
        self.assertIn(
            '<span class="ecdf-budget">15,000 function evaluations</span>',
            html,
        )
        self.assertIn("#ecdf-title .ecdf-meta", html)
        self.assertLess(
            html.index("RR-6828"),
            html.index("part of the following publications"),
        )
        self.assertIn("text/csv", html)
        self.assertIn("meta.qualifier", html)
        self.assertIn('text.indexOf(",")', html)
        self.assertIn("String.fromCharCode(10)", html)
        self.assertIn("max-width: 1100px", html)
        self.assertIn("flex-direction: column", html)
        self.assertNotIn("Plotly.restyle(gd, {visible: visible}, traces)", html)
        budget_at = html.index('getElementById("ecdf-budget").addEventListener')
        comparison_at = html.index('getElementById("ecdf-comparison").addEventListener')
        legend_at = html.index("legend.addEventListener")
        budget_line = html[budget_at:html.index("\n", budget_at)]
        self.assertIn("apply", budget_line)
        self.assertIn("clearHighlight()", html[comparison_at:legend_at])
        self.assertIn("writeTitle()", html[comparison_at:legend_at])
        self.assertIn("function series(", html)
        self.assertIn("_inputArray", html)
        self.assertIn("value.bdata", html)

    def test_convergence_menu_shows_one_scope_and_one_view(self) -> None:
        keys = (
            ("all", "median"),
            ("all", "median"),
            ("all", "ALPHA"),
            ("L6750_B2", "median"),
            ("L6750_B2", "ALPHA"),
        )
        self.assertEqual(
            selected_view_mask(keys, "all", "median"),
            [True, True, False, False, False],
        )
        self.assertEqual(
            selected_view_mask(keys, "L6750_B2", "ALPHA"),
            [False, False, False, False, True],
        )

    def test_convergence_menus_are_inserted_above_the_plot(self) -> None:
        html = _inject_convergence_menus(
            '<html><body><div id="convergence-pooled"></div></body></html>',
            scope_options=[("all", "all problems"), ("L6750_B2", "L6750_B2")],
            view_options=[("median", "all algorithms (median)"), ("ALPHA", "ALPHA")],
            keys=[("all", "median"), ("L6750_B2", "ALPHA")],
            titles={"all|median": "pooled median", "L6750_B2|ALPHA": "one algorithm"},
        )
        self.assertLess(html.index('id="conv-scope"'), html.index('id="convergence-pooled"'))
        self.assertGreater(
            html.index('getElementById("convergence-pooled")'),
            html.index('id="convergence-pooled"'),
        )
        self.assertIn("all algorithms (median)", html)
        self.assertIn("Plotly.restyle(gd, {visible: visible}, traces)", html)

    def test_problem_label_uses_the_span_token(self) -> None:
        self.assertEqual(
            problem_label(Path("study_ws_cmp_p1_L6750_B2_r20")),
            "L6750_B2",
        )
        self.assertEqual(
            problem_label(Path("study_opt_algorithms_p1_L6750_ft1pct")),
            "L6750",
        )

    def test_ecdf_menu_shows_one_scope_and_one_margin(self) -> None:
        keys = (
            ("all", 1.0),
            ("all", 5.0),
            ("L6750_B2", 1.0),
            ("L6750_B2", 5.0),
        )
        self.assertEqual(
            selected_trace_mask(keys, "all", 1.0),
            [True, False, False, False],
        )
        self.assertEqual(
            selected_trace_mask(keys, "L6750_B2", 5.0),
            [False, False, False, True],
        )

    def test_ecdf_denominator_is_every_pooled_run(self) -> None:
        # 1% of 100 is 101; 1% of 400 is 404. One hit, one miss.
        runs = (
            _run("A", "p1", 100.0, [100.0], task_index=1),
            _run("A", "p2", 400.0, [500.0], task_index=2),
            _run("A", "p3", 200.0, [200.0], task_index=3),
            _run("A", "p4", 800.0, [900.0], task_index=4),
        )
        curve = success_probability_curve(runs, 1.0)
        self.assertEqual(curve.n_runs, 4)
        self.assertEqual(curve.n_success, 2)
        self.assertAlmostEqual(curve.success_rate, 0.5)

    def test_exclude_drops_named_algorithms(self) -> None:
        bundle = ExperimentBundle(
            runs=(
                _run("KEEP", "p1", 100.0, [100.0], task_index=1),
                _run("DROP", "p1", 100.0, [100.0], task_index=2),
                _run("DROP", "p2", 100.0, [100.0], task_index=3),
            ),
            experiment_dirs=(Path("p1"), Path("p2")),
        )
        kept = without_excluded_algorithms(bundle, [" DROP ", "DROP", ""])
        self.assertEqual({run.algorithm_name for run in kept.runs}, {"KEEP"})
        self.assertEqual(kept.experiment_dirs, bundle.experiment_dirs)

    def test_exclude_rejects_an_unknown_algorithm(self) -> None:
        bundle = ExperimentBundle(
            runs=(_run("KEEP", "p1", 100.0, [100.0], task_index=1),),
            experiment_dirs=(Path("p1"),),
        )
        with self.assertRaises(ValueError) as ctx:
            without_excluded_algorithms(bundle, ["MISSING"])
        self.assertIn("MISSING", str(ctx.exception))
        self.assertIn("KEEP", str(ctx.exception))

    def test_ecdf_order_is_success_rate_then_mean_evaluations(self) -> None:
        # 1% of 100 is 101. Z and A both hit once; Z hits sooner, so Z precedes A.
        # BOTH hits on every evaluation and therefore ranks first.
        grouped = {
            "A": (
                _run("A", "p1", 100.0, [150.0, 100.0], task_index=1),
                _run("A", "p1", 100.0, [150.0, 150.0], task_index=2),
            ),
            "Z": (
                _run("Z", "p1", 100.0, [100.0, 150.0], task_index=3),
                _run("Z", "p1", 100.0, [150.0, 150.0], task_index=4),
            ),
            "BOTH": (
                _run("BOTH", "p1", 100.0, [100.0], task_index=5),
                _run("BOTH", "p1", 100.0, [100.0], task_index=6),
            ),
        }
        self.assertEqual(
            ecdf_algorithm_order(grouped, 1.0, family_of=_family),
            ["BOTH", "Z", "A"],
        )

    def test_ecdf_order_keeps_a_cold_start_with_its_warm_start(self) -> None:
        # FAST's warm start reaches a full share at evaluation 1, so that pair
        # leads. SLOW's cold start is the better of its pair, and the cold
        # start is listed before the warm start.
        grouped = {
            "SLOW_WARM_START": (
                _run("SLOW_WARM_START", "p1", 100.0, [100.0, 150.0], task_index=1),
                _run("SLOW_WARM_START", "p1", 100.0, [150.0, 150.0], task_index=2),
            ),
            "SLOW": (
                _run("SLOW", "p1", 100.0, [150.0, 100.0], task_index=3),
                _run("SLOW", "p1", 100.0, [150.0, 100.0], task_index=4),
            ),
            "FAST_WARM_START": (
                _run("FAST_WARM_START", "p1", 100.0, [100.0], task_index=5),
                _run("FAST_WARM_START", "p1", 100.0, [100.0], task_index=6),
            ),
            "FAST": (
                _run("FAST", "p1", 100.0, [150.0, 100.0], task_index=7),
                _run("FAST", "p1", 100.0, [150.0, 150.0], task_index=8),
            ),
        }
        self.assertEqual(
            ecdf_algorithm_order(grouped, 1.0, family_of=_family),
            ["FAST", "FAST_WARM_START", "SLOW", "SLOW_WARM_START"],
        )

    def test_ecdf_order_uses_the_better_variants_evaluations(self) -> None:
        # Both families reach a full share. Basin hopping's warm start reaches
        # it at evaluation 1, so that family leads even though its cold start
        # never hits. The mean of the two basin-hopping rates is lower.
        grouped = {
            "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START": (
                _run("POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START", "p1", 100.0, [150.0, 150.0], task_index=1),
                _run("POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START", "p1", 100.0, [150.0, 150.0], task_index=2),
            ),
            "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START": (
                _run("POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START", "p1", 100.0, [100.0], task_index=3),
                _run("POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START", "p1", 100.0, [100.0], task_index=4),
            ),
            "K_ELITE_AMS_PEN_SI_DESC_COLD_START": (
                _run("K_ELITE_AMS_PEN_SI_DESC_COLD_START", "p1", 100.0, [150.0, 100.0], task_index=5),
                _run("K_ELITE_AMS_PEN_SI_DESC_COLD_START", "p1", 100.0, [150.0, 100.0], task_index=6),
            ),
            "K_ELITE_AMS_PEN_SI_DESC_WARM_START": (
                _run("K_ELITE_AMS_PEN_SI_DESC_WARM_START", "p1", 100.0, [150.0, 100.0], task_index=7),
                _run("K_ELITE_AMS_PEN_SI_DESC_WARM_START", "p1", 100.0, [150.0, 100.0], task_index=8),
            ),
        }
        self.assertEqual(
            ecdf_algorithm_order(grouped, 1.0),
            [
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
                "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
                "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
            ],
        )

    def test_shown_algorithm_uses_the_paper_name(self) -> None:
        from core.visualization_core.optimization.plots.interactive_pool import (
            _shown_algorithm,
        )

        self.assertEqual(
            _shown_algorithm("POPULATION_BASIN_HOPPING"),
            "Population Basin Hopping, SI · Cold start",
        )
        self.assertEqual(
            _shown_algorithm("POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START"),
            "Population Basin Hopping, SI · Warm start",
        )
        self.assertEqual(
            _shown_algorithm("POPULATION_BASIN_HOPPING_PEN_EQU_DESC_COLD_START"),
            "Population Basin Hopping, EQU · Cold start",
        )
        self.assertEqual(
            _shown_algorithm("OPTAX_ADAM"),
            "Adam · Cold start",
        )

    def test_local_search_caption_follows_the_algorithms_on_the_figure(self) -> None:
        from core.visualization_core.optimization.plots.interactive_pool import (
            local_search_caption,
        )

        self.assertEqual(local_search_caption(()), "")
        strict = local_search_caption(("Strict improvement",))
        self.assertIn("Strict improvement moves only", strict)
        self.assertIn("Equal descent also searches", strict)
        both = local_search_caption(("Equal descent", "Strict improvement"))
        self.assertIn("Strict improvement moves only", both)
        self.assertIn("Equal descent also searches", both)
        self.assertIn("L1 neighbourhood", both)

    def test_ecdf_order_breaks_equal_results_alphabetically(self) -> None:
        grouped = {
            name: (
                _run(name, "p1", 100.0, [100.0], task_index=index),
                _run(name, "p1", 100.0, [100.0], task_index=index + 10),
            )
            for index, name in enumerate(
                (
                    "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                    "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
                    "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
                    "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                    "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
                    "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
                ),
                start=1,
            )
        }
        self.assertEqual(
            ecdf_algorithm_order(grouped, 1.0),
            [
                "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
                "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
                "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
                "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
            ],
        )

    def test_ecdf_legend_sits_to_the_right_of_the_axes(self) -> None:
        layout = _ecdf_legend_layout(["PYMOO_GA (20/20)", "AMS (0/20)"])
        self.assertFalse(layout["showlegend"])
        self.assertTrue(layout["autosize"])
        self.assertGreater(layout["margin"]["l"], layout["margin"]["r"])
        self.assertGreaterEqual(layout["height"], 520)

    def test_written_ecdf_legend_lists_algorithms_in_sort_order(self) -> None:
        try:
            import plotly.graph_objects as go
        except ImportError:
            self.skipTest("plotly is not installed")
        from tempfile import TemporaryDirectory

        from core.visualization_core.optimization.plots.interactive_pool import (
            _write_ecdf,
        )

        grouped = {
            "K_ELITE_AMS_PEN_SI_DESC_COLD_START": (
                _run("K_ELITE_AMS_PEN_SI_DESC_COLD_START", "p1", 100.0, [150.0, 100.0], task_index=1),
                _run("K_ELITE_AMS_PEN_SI_DESC_COLD_START", "p1", 100.0, [150.0, 150.0], task_index=2),
            ),
            "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START": (
                _run(
                    "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                    "p1",
                    100.0,
                    [100.0, 150.0],
                    task_index=3,
                ),
                _run(
                    "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                    "p1",
                    100.0,
                    [150.0, 150.0],
                    task_index=4,
                ),
            ),
        }
        with TemporaryDirectory() as tmp:
            path = _write_ecdf(
                go,
                grouped,
                Path(tmp) / "ecdf_success.html",
                percent=1.0,
                title="success",
            )
            html = path.read_text(encoding="utf-8")
        data = html[html.rfind("Plotly.newPlot"):]
        z_at = data.index("Population Basin Hopping")
        a_at = data.index("Adaptive Multi-Start")
        self.assertLess(z_at, a_at)
        self.assertIn("Algorithms", html)
        self.assertIn("Metaheuristic", html)
        self.assertIn("Last target hit", html)
        self.assertNotIn(
            '\'<div class="fe">\' + formatInt(at.count) + "/" + formatInt(total)',
            html,
        )
        self.assertNotIn("At the full budget", html)
        self.assertIn("shows this proportion as a percentage", html)
        self.assertIn("Cumulative success rate [%]", html)
        self.assertIn("Empirical Cumulative Distribution Function", html)
        self.assertNotIn("Fixed-target performance", html)
        self.assertIn("relative target tolerance", html)
        self.assertIn("Predesign evaluations", html)
        self.assertIn("minimum stored penalised objective", html)
        self.assertIn(
            "follows the selected set of problem instances, problem instance, relative target tolerance",
            html,
        )
        self.assertNotIn("remains fixed", html)
        self.assertLess(
            html.index("RR-6828"),
            html.index("Dombrowski M, Melcer E, Merz P, Schlaich M, Wortmann T"),
        )
        self.assertIn("Penalty-based", html)
        self.assertIn("Constraint-aware", html)
        self.assertIn("<b>Local search.</b>", html)
        self.assertIn("Strict improvement moves only", html)
        self.assertIn("Equal descent also searches", html)
        self.assertIn("0.15", html)
        self.assertIn('id="ecdf-reset"', html)
        self.assertIn("font-family:'DejaVu Sans'", html)
        self.assertIn('"dtick":2000', data)
        self.assertIn('"shape":"hv"', data)
        self.assertNotIn('"shape":"vh"', data)
        self.assertIn("hoverlayer", html)

        from core.visualization_core.optimization.plots.interactive_pool import (
            _write_ecdf_margins,
        )

        scopes = (
            ("all", "all problems", grouped),
            ("p1", "p1", grouped),
        )
        with TemporaryDirectory() as tmp:
            pooled = _write_ecdf_margins(
                go,
                scopes,
                Path(tmp) / "ecdf_success_pooled.html",
                percents=(1.0, 5.0),
            )
            pooled_html = pooled.read_text(encoding="utf-8")
        pooled_data = pooled_html[pooled_html.rfind("Plotly.newPlot"):]
        self.assertLess(
            pooled_data.index("Population Basin Hopping"),
            pooled_data.index("Adaptive Multi-Start"),
        )
        self.assertIn('id="ecdf-scope"', pooled_html)
        self.assertIn('id="ecdf-comparison"', pooled_html)
        self.assertIn("Cumulative success rate", pooled_html)
        self.assertIn("plotly_hover", pooled_html)
        self.assertIn("0.5", pooled_html)
        figure_html = pooled_html[
            pooled_html.rfind("Plotly.newPlot") : pooled_html.find(
                "var payload =", pooled_html.rfind("Plotly.newPlot")
            )
        ]
        # One trace for the first scope and the first tolerance: the name and
        # the display name. The other scope and tolerance stay in the payload.
        # Count from the figure call so the embedded Plotly bundle is excluded.
        written = pooled_html[pooled_html.rfind("Plotly.newPlot") :]
        self.assertEqual(figure_html.count("Population Basin Hopping"), 2)
        self.assertEqual(written.count("Population Basin Hopping"), 6)
        self.assertIn('"margin": 5.0', pooled_html)
        self.assertNotIn('"margin": 5.0', figure_html)

    def test_comparison_colors_follow_dark24(self) -> None:
        from core.visualization_core.optimization.plots.interactive_pool import (
            _DARK24,
            _comparison_palette,
            _dark24_colors,
        )

        wide = _dark24_colors(16)
        pair = _dark24_colors(2)
        single = _dark24_colors(1)
        self.assertEqual(wide, list(_DARK24[:16]))
        self.assertEqual(len(set(wide)), 16)
        self.assertEqual(wide[0], "#2E91E5")
        self.assertEqual(pair, ["#2E91E5", "#E15F99"])
        self.assertEqual(single, [wide[0]])
        self.assertEqual(wide[1], pair[1])
        with self.assertRaises(ValueError):
            _dark24_colors(25)
        palette = _comparison_palette(
            [
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
                "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
                "RBFOPT",
            ]
        )
        cold = palette["POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START"]
        warm = palette["POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START"]
        other = palette["K_ELITE_AMS_PEN_SI_DESC_COLD_START"]
        model = palette["RBFOPT"]
        families = _dark24_colors(3)
        classes = _dark24_colors(2)
        self.assertEqual(cold["algorithm"], warm["algorithm"])
        self.assertEqual(cold["algorithm"], families[0])
        self.assertEqual(other["algorithm"], families[1])
        self.assertEqual(model["algorithm"], families[2])
        self.assertEqual(cold["class"], other["class"])
        self.assertEqual(cold["class"], classes[0])
        self.assertEqual(model["class"], classes[1])
        self.assertEqual(cold["start"], pair[0])
        self.assertEqual(warm["start"], pair[1])
        self.assertEqual(model["start"], pair[0])

    def test_downloads_can_be_left_out(self) -> None:
        html = _inject_ecdf_menus(
            '<html><body><div id="ecdf-pooled"></div></body></html>',
            scope_options=[("all", "all problems")],
            percents=(1.0,),
            downloads=False,
        )
        self.assertNotIn('id="ecdf-png"', html)
        self.assertNotIn('id="ecdf-svg"', html)
        self.assertNotIn('id="ecdf-hits"', html)
        self.assertNotIn('getElementById("ecdf-png")', html)
        self.assertIn('id="ecdf-title"', html)

    def test_written_ecdf_draws_a_warm_start_dashed(self) -> None:
        try:
            import plotly.graph_objects as go
        except ImportError:
            self.skipTest("plotly is not installed")
        from tempfile import TemporaryDirectory

        from core.visualization_core.optimization.plots.interactive_pool import (
            _write_ecdf,
        )

        grouped = {
            "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START": (
                _run(
                    "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                    "p1",
                    100.0,
                    [150.0, 150.0],
                    task_index=1,
                ),
            ),
            "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START": (
                _run(
                    "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
                    "p1",
                    100.0,
                    [100.0],
                    task_index=2,
                ),
            ),
            "K_ELITE_AMS_PEN_SI_DESC_COLD_START": (
                _run("K_ELITE_AMS_PEN_SI_DESC_COLD_START", "p1", 100.0, [150.0, 150.0], task_index=3),
            ),
        }
        with TemporaryDirectory() as tmp:
            path = _write_ecdf(
                go,
                grouped,
                Path(tmp) / "ecdf_success.html",
                percent=1.0,
                title="success",
            )
            html = path.read_text(encoding="utf-8")
        data = html[html.rfind("Plotly.newPlot"):]
        cold = data.index("Population Basin Hopping")
        warm = data.index("Warm start")
        other = data.index("Adaptive Multi-Start")
        self.assertLess(cold, warm)
        self.assertLess(warm, other)
        self.assertIn('"dash":"dash"', data[max(0, warm - 700) : warm])
        self.assertIn('"dash":"solid"', data[:warm])
        self.assertIn("plotly_hover", html)

    def test_problem_menu_lists_each_problem_id(self) -> None:
        from core.visualization_core.optimization.plots.interactive_pool import (
            ecdf_problem_scopes,
        )

        scopes = ecdf_problem_scopes(
            (
                _run("A", "L8100_B2", 100.0, [100.0], task_index=1),
                _run("A", "L6750_A1", 100.0, [100.0], task_index=2),
            )
        )
        self.assertEqual(
            [label for _scope, label, _grouped in scopes],
            ["All problems", "L6750_A1", "L8100_B2"],
        )

    def test_display_metadata_covers_the_registry(self) -> None:
        from core.ioh_core.algorithms.registry import list_algorithm_specs
        from core.visualization_core.optimization.algorithm_presentation import (
            PRESENTATIONS,
        )

        names = {spec.name for spec in list_algorithm_specs()}
        self.assertEqual(set(PRESENTATIONS), names)

    def test_study_classes_follow_the_overview_table(self) -> None:
        from core.visualization_core.optimization.algorithm_presentation import (
            legend_handling,
            local_search_label,
            presentation_for,
        )

        historical = presentation_for("POPULATION_BASIN_HOPPING")
        current = presentation_for("POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START")
        self.assertEqual(historical.display_name, current.display_name)
        self.assertEqual(historical.family, current.family)
        self.assertEqual(historical.qualifier, "SI-DESC")
        self.assertEqual(historical.start, "cold")
        self.assertEqual(local_search_label(historical), local_search_label(current))
        self.assertEqual(legend_handling(historical), legend_handling(current))
        warm = presentation_for("K_ELITE_AMS_WARM_START")
        self.assertEqual(
            warm.display_name,
            presentation_for("K_ELITE_AMS_PEN_SI_DESC_WARM_START").display_name,
        )
        self.assertEqual(warm.start, "warm")
        self.assertEqual(
            presentation_for("DISCRETE_MLSL").algorithm_class,
            "Metaheuristic",
        )
        self.assertEqual(
            presentation_for("POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START").algorithm_class,
            "Metaheuristic",
        )
        self.assertEqual(
            presentation_for("POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START").algorithm_class,
            "Metaheuristic",
        )
        self.assertTrue(presentation_for("POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START").constraint_aware)
        self.assertFalse(presentation_for("POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START").constraint_aware)
        self.assertEqual(presentation_for("RBFOPT").algorithm_class, "Model-based")
        self.assertEqual(presentation_for("HYPEROPT_TPE").algorithm_class, "Model-based")
        self.assertEqual(presentation_for("NLOPT_SBPLX").algorithm_class, "Direct search")
        self.assertEqual(presentation_for("SCIPY_COBYLA").algorithm_class, "Direct search")
        self.assertTrue(presentation_for("SCIPY_COBYLA").constraint_aware)
        self.assertEqual(presentation_for("OPTAX_ADAM").algorithm_class, "Derivative-based")
        self.assertEqual(
            presentation_for("PYMOO_GA_CONSTRAINED").algorithm_class,
            "Metaheuristic",
        )

    def test_ert_menus_follow_the_ecdf_type_and_can_hide_exports(self) -> None:
        from core.visualization_core.optimization.plots.interactive_pool import (
            _inject_ert_menus,
        )

        html = _inject_ert_menus(
            '<html><body><div id="ert-pooled"></div></body></html>',
            scope_options=[("all", "All problems"), ("L6750_B2", "L6750_B2")],
            budgets=(2000, 5000, 10000, 15000),
            shown_budget=15000,
            algorithm_classes=("Metaheuristic", "Direct search"),
        )
        self.assertLess(html.index('id="ert-title"'), html.index('id="ert-menus"'))
        self.assertLess(html.index('id="ert-menus"'), html.index('id="ert-pooled"'))
        self.assertLess(html.index('id="ert-comparison"'), html.index('id="ert-scope"'))
        self.assertLess(html.index('id="ert-scope"'), html.index('id="ert-budget"'))
        self.assertLess(html.index('id="ert-budget"'), html.index('id="ert-algorithms"'))
        self.assertLess(html.index('id="ert-algorithms"'), html.index('id="ert-reset"'))
        self.assertNotIn('id="ert-margin"', html)
        self.assertIn("Expected Running Time of algorithms across optimisation problems", html)
        self.assertNotIn("evaluation budget 15,000 function evaluations", html)
        self.assertIn('value="15000" selected', html)
        self.assertIn("font-size:18px", html)
        self.assertIn("font-size:13px", html)
        self.assertIn("font-size:12px", html)
        self.assertIn("font-size:11px", html)
        self.assertIn("DejaVu Sans", html)
        self.assertIn(">Cold start</span>", html)
        self.assertIn(">Warm start</span>", html)
        self.assertIn("Expected Running Time at", html)
        self.assertIn("ert-anchor", html)
        self.assertIn("ε<sub>X</sub> = 1%", html)
        self.assertIn("Definitions", html)
        self.assertLess(
            html.index("RR-6828"),
            html.index("part of the following publications"),
        )
        self.assertIn("not as the mean of the individual problem values", html)
        self.assertNotIn("descriptive aggregate", html)
        self.assertNotIn("N<sub>ε<sub>X</sub></sub> &gt; 0.", html)
        self.assertIn("y<sub>p</sub>*,", html)
        self.assertIn('" runs</div>"', html)
        self.assertIn("minimum stored penalised objective", html)
        self.assertIn(
            "follows the selected set of problem instances, problem instance, and evaluation budget",
            html,
        )
        self.assertIn("lower Expected Running Time of that better variant", html)
        self.assertNotIn("remains fixed", html)
        ert_item = html.index("Expected Running Time (ERT)")
        ert_end = html.index("</li>", ert_item)
        self.assertIn("ERT<sub>cold</sub>", html[ert_item:ert_end])
        self.assertIn("ERT<sub>warm</sub>", html[ert_item:ert_end])
        self.assertNotIn("Cold-Start ERT and Warm-Start adjusted ERT", html)
        self.assertLess(
            html.index('class="sum-lim">R</span>', ert_item),
            ert_end,
        )
        self.assertLess(html.index('class="formula-note"', ert_item), html.index("</li>", ert_item))
        target_item = html.index("Relative target tolerance")
        self.assertLess(
            html.index('class="formula-note"', target_item),
            html.index("</li>", target_item),
        )
        self.assertIn(">Input values</div>", html)
        self.assertIn(">Expected Running Time</div>", html)
        self.assertIn("Successful runs:", html)
        self.assertIn("Selected evaluation budget:", html)
        self.assertIn("Unsuccessful runs contribute the evaluations they performed.", html)
        self.assertNotIn(">Conditions</div>", html)
        self.assertNotIn("ERT at ε<sub>X</sub> =", html.split('id="ert-title"')[1].split("</h1>")[0])
        self.assertIn("plotly_hover", html)
        self.assertIn('id="ert-png"', html)
        self.assertIn('id="ert-svg"', html)
        self.assertIn('id="ert-table"', html)
        hidden = _inject_ert_menus(
            '<html><body><div id="ert-pooled"></div></body></html>',
            scope_options=[("all", "All problems")],
            budgets=(15000,),
            downloads=False,
        )
        self.assertNotIn('id="ert-png"', hidden)
        self.assertNotIn('id="ert-svg"', hidden)
        self.assertNotIn('id="ert-table"', hidden)
        self.assertNotIn('getElementById("ert-png")', hidden)
        self.assertIn('id="ert-reset"', hidden)

    def test_written_ert_uses_the_hansen_formula_and_ecdf_colours(self) -> None:
        try:
            import plotly.graph_objects as go
        except ImportError:
            self.skipTest("plotly is not installed")
        from dataclasses import replace
        from tempfile import TemporaryDirectory

        from core.visualization_core.optimization.plots.interactive_pool import (
            _DARK24,
            _write_ert,
        )

        def _budget(name: str, problem: str, curve: list[float], task_index: int):
            return replace(
                _run(name, problem, 100.0, curve, task_index=task_index),
                max_evaluations=15000,
            )

        grouped = {
            "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START": (
                _budget("POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START", "L6750_B2", [110.0, 100.0], 1),
                _budget(
                    "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                    "L6750_B2",
                    [200.0] * 2999 + [180.0],
                    2,
                ),
            ),
            "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START": (
                _budget(
                    "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
                    "L6750_B2",
                    [100.0],
                    3,
                ),
            ),
        }
        with TemporaryDirectory() as tmp:
            path = _write_ert(
                go,
                (("all", "All problems", grouped),),
                Path(tmp) / "ert_pooled.html",
                percents=(1.0, 2.0),
                budgets=(2000, 15000),
            )
            html = path.read_text(encoding="utf-8")
        self.assertIn('"y": [3002.0, 3002.0]', html)
        self.assertIn('"y": [2002.0, 2002.0]', html)
        self.assertIn('"y": [2701.0, 2701.0]', html)
        self.assertNotIn('"y": [1.0, 1.0]', html)
        self.assertIn(_DARK24[0], html)
        self.assertIn('"dash": "dash"', html.replace('"dash":"dash"', '"dash": "dash"'))
        self.assertTrue('"type": "log"' in html or '"type":"log"' in html)
        self.assertIn("Expected Running Time [FE]", html)
        self.assertIn("Relative target tolerance", html)


if __name__ == "__main__":
    unittest.main()
