"""Success heatmap and the shared completed-repetition gate."""
from __future__ import annotations

import csv
import io
import json
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from core.visualization_core.optimization.api import (
    render_suite,
    render_suite_compare,
)
from core.visualization_core.optimization.cli import main
from core.visualization_core.optimization.io import load_experiments
from core.visualization_core.optimization.models import PlotConfig
from core.visualization_core.optimization.plots.interactive_pool import (
    render_interactive_pool,
)
from core.visualization_core.optimization.plots.success_heatmap import (
    _MIN_SPAN_LOADS,
    _OVERALL_INSET,
    _OVERALL_TAIL,
    _OVERALL_W,
    _SPAN_GAP,
    _label_px,
    _overall_column_width,
    _span_axis,
    _span_field_width,
    _x_annotations,
    render_success_heatmap,
)


def _write_dat(path: Path, y_p: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["evaluations raw_y t_eval_s y y_p misses hits"]
    for index, value in enumerate(y_p, start=1):
        lines.append(
            f"{float(index)} {value} 0.01 {value} {value} 0 {float(index)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_experiment(root: Path, tasks: list[dict]) -> None:
    """Write one experiment directory from explicit task rows."""
    (root / "aggregate").mkdir(parents=True, exist_ok=True)
    fields = [
        "task_index",
        "task_id",
        "problem_id",
        "algorithm_name",
        "repetition",
        "attempt",
        "state",
        "artifact_path",
        "global_optimum",
        "target_hit",
        "evaluations_to_target",
        "wall_time_s",
    ]
    rows: list[dict[str, object]] = []
    for task_index, task in enumerate(tasks):
        state = str(task.get("state", "completed"))
        algorithm_name = str(task["algorithm_name"])
        problem_id = str(task["problem_id"])
        optimum = task.get("global_optimum", 100.0)
        attempt_dir = root / "tasks" / f"task_{task_index:06d}" / "attempts" / "001"
        summary_path = attempt_dir / "summary.json"
        if state == "completed":
            dat = (
                attempt_dir
                / "ioh"
                / "run"
                / "data_f0_1"
                / "IOHprofiler_f0_DIM2.dat"
            )
            _write_dat(dat, list(task.get("y_p", [100.0])))
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "algorithm_name": algorithm_name,
                        "problem_id": problem_id,
                        "task_index": task_index,
                        "repetition": int(task.get("repetition", 0)),
                        "evaluations": len(task.get("y_p", [100.0])),
                        "algorithm_parameters": {
                            "max_evaluations": int(task.get("max_evaluations", 10))
                        },
                        "global_optimum": optimum,
                        "target_reference": "all",
                        "target_hit": False,
                        "wall_time_s": 1.0,
                    }
                ),
                encoding="utf-8",
            )
        rows.append(
            {
                "task_index": task_index,
                "task_id": f"task_{task_index:06d}",
                "problem_id": problem_id,
                "algorithm_name": algorithm_name,
                "repetition": int(task.get("repetition", 0)),
                "attempt": 1,
                "state": state,
                "artifact_path": str(summary_path) if state == "completed" else "",
                "global_optimum": "" if optimum is None else optimum,
                "target_hit": False,
                "evaluations_to_target": "",
                "wall_time_s": 1.0,
            }
        )
    with (root / "aggregate" / "tasks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "aggregate" / "summary.json").write_text("{}", encoding="utf-8")


class SpanFieldTests(unittest.TestCase):
    def test_one_load_span_uses_the_three_load_width(self) -> None:
        problems = ["L6750_B2", "L8100_B2", "L9450_B2", "L10800_B2"]
        axis = _span_axis(problems)
        self.assertEqual(
            axis["categories"],
            [
                "__xpad_0",
                "L6750_B2",
                "__xpad_1",
                "__xpad_2",
                "L8100_B2",
                "__xpad_3",
                "__xpad_4",
                "L9450_B2",
                "__xpad_5",
                "__xpad_6",
                "L10800_B2",
                "__xpad_7",
            ],
        )
        self.assertEqual(axis["problem_x"], [1, 4, 7, 10])
        cell_w = 36
        minimum = _span_field_width(_MIN_SPAN_LOADS, cell_w)
        notes = [
            note
            for note in _x_annotations(problems, 400, cell_w, axis=axis)
            if note["labelkind"] == "span"
        ]
        self.assertEqual(len(notes), 8)
        for note in notes:
            self.assertEqual(note["spancount"], _MIN_SPAN_LOADS)
            self.assertEqual(note["width"], minimum)
            self.assertGreaterEqual(
                note["width"],
                _MIN_SPAN_LOADS * cell_w - _SPAN_GAP - 2,
            )

    def test_three_load_span_is_not_padded(self) -> None:
        problems = ["L6750_A1", "L6750_B1", "L6750_B2", "L7425_A1", "L7425_B1", "L7425_B2"]
        axis = _span_axis(problems)
        self.assertEqual(axis["categories"], problems)
        self.assertEqual(axis["problem_x"], [0, 1, 2, 3, 4, 5])
        wider = _span_field_width(4, 36)
        self.assertGreater(wider, _span_field_width(3, 36))


class CompletionGateTests(unittest.TestCase):
    def test_suite_refuses_a_failed_repetition_and_writes_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            out = root / "figures"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [100.0],
                    },
                    {
                        "algorithm_name": "CMAES_CATCMAWM",
                        "problem_id": "L8100_B2",
                        "state": "failed",
                    },
                ],
            )
            bundle = load_experiments([experiment])
            with self.assertRaises(ValueError) as raised:
                render_suite(
                    bundle,
                    PlotConfig(),
                    out,
                    which=("summary",),
                )
            message = str(raised.exception)
            self.assertIn("CMAES_CATCMAWM", message)
            self.assertIn("L8100_B2", message)
            self.assertIn("failed", message)
            self.assertFalse(out.exists())

    def test_excluding_the_failed_algorithm_lets_the_suite_render(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            out = root / "figures"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [100.0],
                    },
                    {
                        "algorithm_name": "CMAES_CATCMAWM",
                        "problem_id": "L8100_B2",
                        "state": "failed",
                    },
                ],
            )
            bundle = load_experiments([experiment])
            written = render_suite(
                bundle,
                PlotConfig(),
                out,
                which=("summary",),
                exclude_algorithms=("CMAES_CATCMAWM",),
            )
            csv_paths = [path for path in written if path.suffix == ".csv"]
            self.assertEqual(len(csv_paths), 1)
            text = csv_paths[0].read_text(encoding="utf-8")
            self.assertIn("OPTUNA_TPE", text)
            self.assertNotIn("CMAES_CATCMAWM", text)

    def test_selecting_one_problem_still_sees_an_unfinished_task_elsewhere(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_A1",
                        "y_p": [100.0],
                    },
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L8100_B2",
                        "state": "timed_out",
                        "repetition": 1,
                    },
                ],
            )
            bundle = load_experiments([experiment]).for_problem("L6750_A1")
            with self.assertRaises(ValueError) as raised:
                render_suite(
                    bundle,
                    PlotConfig(),
                    root / "figures",
                    which=("summary",),
                )
            self.assertIn("timed_out", str(raised.exception))
            self.assertIn("L8100_B2", str(raised.exception))

    def test_unknown_exclusion_names_the_algorithms_that_are_present(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [100.0],
                    }
                ],
            )
            bundle = load_experiments([experiment])
            with self.assertRaises(ValueError) as raised:
                render_suite(
                    bundle,
                    PlotConfig(),
                    root / "figures",
                    which=("summary",),
                    exclude_algorithms=("MISSING",),
                )
            message = str(raised.exception)
            self.assertIn("MISSING", message)
            self.assertIn("OPTUNA_TPE", message)

    def test_a_non_positive_optimum_refuses_the_suite(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [100.0],
                        "global_optimum": 0.0,
                    }
                ],
            )
            bundle = load_experiments([experiment])
            with self.assertRaises(ValueError) as raised:
                render_suite(
                    bundle,
                    PlotConfig(),
                    root / "figures",
                    which=("summary",),
                )
            self.assertIn("no positive sealed optimum", str(raised.exception))
            self.assertIn("OPTUNA_TPE", str(raised.exception))
            self.assertIn("L6750_B2", str(raised.exception))

    def test_compare_and_interactive_refuse_the_same_unfinished_task(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [100.0],
                    },
                    {
                        "algorithm_name": "CMAES_CATCMAWM",
                        "problem_id": "L6750_B2",
                        "state": "pending",
                        "repetition": 1,
                    },
                ],
            )
            bundle = load_experiments([experiment])
            compare_out = root / "compare"
            with self.assertRaises(ValueError) as compare_raised:
                render_suite_compare(
                    [("campaign", bundle, PlotConfig())],
                    compare_out,
                    which=("summary",),
                )
            self.assertIn("pending", str(compare_raised.exception))
            self.assertFalse(compare_out.exists())
            interactive_out = root / "interactive"
            with self.assertRaises(ValueError) as interactive_raised:
                render_interactive_pool(bundle, interactive_out)
            self.assertIn("CMAES_CATCMAWM", str(interactive_raised.exception))
            self.assertFalse(interactive_out.exists())

    def test_compare_accepts_an_exclusion_known_from_another_panel(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left"
            right = root / "right"
            write_experiment(
                left,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [100.0],
                    },
                    {
                        "algorithm_name": "CMAES_CATCMAWM",
                        "problem_id": "L6750_B2",
                        "state": "failed",
                        "repetition": 1,
                    },
                ],
            )
            write_experiment(
                right,
                [
                    {
                        "algorithm_name": "PYMOO_GA",
                        "problem_id": "L8100_B2",
                        "y_p": [100.0],
                    }
                ],
            )
            written = render_suite_compare(
                [
                    ("left", load_experiments([left]), PlotConfig()),
                    ("right", load_experiments([right]), PlotConfig()),
                ],
                root / "figures",
                which=("summary",),
                exclude_algorithms=("CMAES_CATCMAWM",),
            )
            self.assertTrue(written)

    def test_a_directory_with_no_completed_run_names_the_unfinished_task(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = root / "failed_only"
            out = root / "figures"
            write_experiment(
                failed,
                [
                    {
                        "algorithm_name": "CMAES_CATCMAWM",
                        "problem_id": "L8100_B2",
                        "state": "failed",
                    }
                ],
            )
            bundle = load_experiments([failed])
            self.assertEqual(bundle.runs, ())
            with self.assertRaises(ValueError) as raised:
                render_interactive_pool(bundle, out, which=("heatmap",))
            message = str(raised.exception)
            self.assertIn("CMAES_CATCMAWM", message)
            self.assertIn("L8100_B2", message)
            self.assertIn("failed", message)
            self.assertNotIn("no completed tasks", message)
            self.assertFalse(out.exists())

    def test_excluding_an_all_unfinished_directory_does_not_use_the_loader_error(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good"
            failed = root / "failed_only"
            out = root / "figures"
            write_experiment(
                good,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [100.0],
                    }
                ],
            )
            write_experiment(
                failed,
                [
                    {
                        "algorithm_name": "CMAES_CATCMAWM",
                        "problem_id": "L8100_B2",
                        "state": "timed_out",
                    }
                ],
            )
            bundle = load_experiments([good, failed])
            with self.assertRaises(ValueError) as refused:
                render_suite(
                    bundle,
                    PlotConfig(),
                    out,
                    which=("summary",),
                )
            self.assertIn("timed_out", str(refused.exception))
            self.assertIn("CMAES_CATCMAWM", str(refused.exception))
            self.assertFalse(out.exists())
            written = render_suite(
                bundle,
                PlotConfig(),
                out,
                which=("summary",),
                exclude_algorithms=("CMAES_CATCMAWM",),
            )
            self.assertTrue(any(path.suffix == ".csv" for path in written))

    def test_selecting_a_problem_with_no_completed_run_still_names_that_task(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            out = root / "figures"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_A1",
                        "y_p": [100.0],
                    },
                    {
                        "algorithm_name": "CMAES_CATCMAWM",
                        "problem_id": "L8100_B2",
                        "state": "failed",
                        "repetition": 1,
                    },
                ],
            )
            bundle = load_experiments([experiment])
            with self.assertRaises(ValueError) as raised:
                render_suite(
                    bundle,
                    PlotConfig(),
                    out,
                    which=("summary",),
                    problem_id="L8100_B2",
                )
            message = str(raised.exception)
            self.assertIn("failed", message)
            self.assertIn("CMAES_CATCMAWM", message)
            self.assertIn("L8100_B2", message)
            self.assertNotIn("unknown problem_id", message)
            self.assertFalse(out.exists())

    def test_a_bad_optimum_on_a_problem_that_is_not_plotted_does_not_abort(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_A1",
                        "y_p": [100.0],
                        "global_optimum": 100.0,
                    },
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L8100_B2",
                        "y_p": [100.0],
                        "global_optimum": 0.0,
                        "repetition": 1,
                    },
                ],
            )
            written = render_suite(
                load_experiments([experiment]),
                PlotConfig(),
                root / "figures",
                which=("summary",),
                problem_id="L6750_A1",
            )
            self.assertTrue(any(path.suffix == ".csv" for path in written))


def _payload(html: str) -> dict:
    match = re.search(
        r'<script id="success-heatmap-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("heatmap html has no data payload")
    return json.loads(match.group(1))


class SuccessHeatmapTests(unittest.TestCase):
    def test_overall_column_widens_only_for_long_pooled_counts(self) -> None:
        short = "100.0 % (4/5)"
        long = "94.0 % (1973/2100)"
        self.assertEqual(_overall_column_width([short, "Overall"]), _OVERALL_W)
        wide = _overall_column_width([short, long])
        self.assertGreater(wide, _OVERALL_W)
        self.assertGreaterEqual(
            wide,
            _label_px(long) + _OVERALL_INSET + _OVERALL_TAIL,
        )
        self.assertLess(wide, _OVERALL_W + 40)

    def test_rendered_overall_column_grows_with_the_pooled_count(self) -> None:
        import numpy as np

        from core.visualization_core.optimization.models import (
            ExperimentBundle,
            RunTrajectory,
        )
        from core.visualization_core.optimization.plots.success_heatmap import (
            _payload as build_payload,
        )

        y = np.array([100.0])
        runs = tuple(
            RunTrajectory(
                experiment_dir=Path("campaign"),
                overlay_label="LOW",
                algorithm_name="LOW",
                task_index=index,
                repetition=index,
                evaluations=np.array([1.0]),
                y_p=y,
                y_p_best_so_far_all=y,
                constraint_utilizations={},
                feasible_mask=None,
                max_evaluations=10,
                global_optimum=100.0,
                target_reference="all",
                target_hit=True,
                evaluations_to_target=1,
                wall_time_s=1.0,
                dat_path=Path("run.dat"),
                problem_id="L6750_A1",
            )
            for index in range(200)
        )
        payload = build_payload(
            ExperimentBundle(runs=runs, experiment_dirs=(Path("campaign"),))
        )
        label = payload["display"]["success"]["overall"]["text"]["1"][1]
        width = payload["display"]["geometry"]["overall_w"]
        self.assertEqual(label, "100.0 % (200/200)")
        self.assertGreater(width, _OVERALL_W)
        self.assertGreaterEqual(
            width,
            _label_px(label) + _OVERALL_INSET + _OVERALL_TAIL,
        )
        note = payload["display"]["success"]["overall_labels"]["1"][0]
        self.assertAlmostEqual(note["x"], -0.5 + _OVERALL_INSET / width)

    def test_matrix_order_rates_margins_and_colours(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "LOW",
                        "problem_id": "L8100_B2",
                        "y_p": [105.0],
                    },
                    {
                        "algorithm_name": "HIGH",
                        "problem_id": "L8100_B2",
                        "y_p": [100.0],
                        "repetition": 1,
                    },
                    {
                        "algorithm_name": "LOW",
                        "problem_id": "L6750_A1",
                        "y_p": [100.0],
                        "repetition": 2,
                    },
                    {
                        "algorithm_name": "LOW",
                        "problem_id": "L6750_B1",
                        "y_p": [100.0],
                        "repetition": 3,
                    },
                    {
                        "algorithm_name": "LOW",
                        "problem_id": "L10125_B1",
                        "y_p": [100.0],
                        "repetition": 4,
                    },
                    {
                        "algorithm_name": "LOW",
                        "problem_id": "custom",
                        "y_p": [100.0],
                        "repetition": 5,
                    },
                ],
            )
            written = render_success_heatmap(
                load_experiments([experiment]),
                root / "figures",
            )
            html = written[0].read_text(encoding="utf-8")
            payload = _payload(html)
            self.assertEqual(
                payload["problems"],
                ["L6750_A1", "L6750_B1", "L8100_B2", "L10125_B1", "custom"],
            )
            self.assertEqual(payload["algorithms"], ["HIGH", "LOW"])
            self.assertEqual(payload["margins"], ["1", "2", "3", "4", "5", "10"])
            self.assertEqual(payload["initial_margin"], "1")
            self.assertEqual(payload["zmin"], 0)
            self.assertEqual(payload["zmax"], 1)
            self.assertEqual(payload["colorscale"], "RdYlGn")
            self.assertEqual(
                payload["text"]["1"][0],
                ["—", "—", "1/1", "—", "—"],
            )
            self.assertEqual(
                payload["text"]["1"][1],
                ["1/1", "1/1", "0/1", "1/1", "1/1"],
            )
            self.assertIsNone(payload["rates"]["1"][0][0])
            self.assertEqual(payload["rates"]["1"][1][2], 0.0)
            self.assertEqual(payload["text"]["10"][1][2], "1/1")
            self.assertEqual(payload["algorithms"], ["HIGH", "LOW"])
            self.assertEqual(payload["axis_labels"], ["HIGH", "LOW"])
            self.assertEqual(
                payload["ticktext"],
                ["0%", "25%", "50%", "75%", "100%"],
            )
            self.assertEqual(payload["colorbar_titles"]["1"], "Success rate (%)")
            self.assertEqual(payload["summary_label"], "All algorithms")
            self.assertEqual(payload["summary"]["rates"]["1"][0][2], 0.5)
            self.assertEqual(payload["summary"]["rates"]["10"][0][2], 1.0)
            self.assertEqual(payload["summary"]["rates"]["1"][0][0], 1.0)
            self.assertEqual(
                payload["summary"]["hover"]["1"][0][2]["subtitle"],
                "Mean of 2 algorithm success rates",
            )
            self.assertEqual(
                payload["problem_ticks"],
                [
                    "<br>A1",
                    "<br>B1",
                    "<br>B2",
                    "<br>B1",
                    "<br>custom",
                ],
            )
            self.assertEqual(payload["column_gaps_after"], [1, 2, 3])
            self.assertEqual(payload["family_gaps_after"], [0])
            self.assertEqual(payload["improvement"]["labels"], [])
            self.assertEqual(payload["display"]["success"]["y"][0], "All algorithms")
            self.assertEqual(payload["display"]["success"]["y"][1], "__ygap_summary")
            self.assertIn("__ygap_0", payload["display"]["success"]["y"])
            self.assertNotIn("__xgap_0", payload["display"]["x"])
            self.assertFalse(payload["style"]["improvement"]["reversescale"])
            self.assertEqual(
                payload["style"]["success"]["colorscale"][0][1],
                "#a50026",
            )
            self.assertEqual(
                payload["style"]["success"]["colorscale"][-1][1],
                "#006837",
            )
            self.assertEqual(
                payload["style"]["improvement"]["colorscale"][0][1],
                "#67001f",
            )
            self.assertEqual(
                payload["style"]["improvement"]["colorscale"][-1][1],
                "#053061",
            )
            overall = payload["display"]["success"]["overall"]["text"]
            self.assertEqual(payload["display"]["geometry"]["overall_w"], _OVERALL_W)
            self.assertEqual(
                payload["display"]["geometry"]["overall_inset"],
                _OVERALL_INSET,
            )
            self.assertEqual(
                payload["display"]["geometry"]["overall_tail"],
                _OVERALL_TAIL,
            )
            self.assertEqual(overall["1"][0], "90.0 %")
            self.assertEqual(overall["1"][1], "100.0 % (1/1)")
            self.assertEqual(overall["1"][2], "80.0 % (4/5)")
            self.assertEqual(overall["10"][2], "100.0 % (5/5)")
            self.assertEqual(
                len(payload["display"]["success"]["overall"]["z"]["1"]),
                len(payload["display"]["success"]["y_tickvals"]),
            )
            self.assertEqual(
                len(payload["display"]["success"]["z"]["1"]),
                len(payload["display"]["success"]["y"]),
            )
            self.assertEqual(
                len(payload["display"]["success"]["z"]["1"][0]),
                len(payload["display"]["x"]),
            )
            self.assertNotIn("annotations", payload)
            miss = payload["hover"]["1"][1][2]
            hit = payload["hover"]["1"][1][0]
            self.assertEqual(miss["fraction"], "0/1")
            self.assertEqual(miss["rate"], "0.0%")
            self.assertEqual(hit["problem"], "L6750_A1")
            self.assertEqual(hit["fraction"], "1/1")
            self.assertNotIn('text": "1/1"', html)
            self.assertIn('<option value="1" selected>1%</option>', html)
            for margin in ("2", "3", "4", "5", "10"):
                self.assertIn(f'<option value="{margin}">{margin}%</option>', html)
            self.assertIn('id="heatmap-caption"', html)
            self.assertLess(
                html.rfind("Plotly.newPlot"),
                html.find('id="heatmap-caption"'),
            )
            self.assertNotIn("</p>Plotly.newPlot", html)
            self.assertIn("y<sub>p</sub>", html)
            self.assertIn("kN/m", html)
            self.assertIn("ε<sub>X</sub>", html)
            self.assertIn("percentage points", html)
            self.assertIn("DejaVu Sans", html)
            self.assertIn("hp shell slabs", html)
            self.assertIn("Dombrowski M, Melcer E, Merz P, Schlaich M, Wortmann T", html)
            self.assertNotIn("<sup>", html[html.find("hp shell slabs"):])
            self.assertIn("earliest evaluation at which that rate is reached", html)
            self.assertIn("follows the selected relative target tolerance", html)
            self.assertNotIn("remains fixed", html)
            self.assertIn("minimum stored penalised objective", html)
            self.assertNotIn("0.01 in this formula", html)
            self.assertNotIn(
                "Algorithm success rates across optimisation problems, relative target tolerance",
                html,
            )
            self.assertIn(
                "Algorithm success rates across optimisation problems",
                html,
            )
            self.assertIn('id="heatmap-view"', html)
            self.assertIn(">Measure <select id=\"heatmap-view\">", html)
            self.assertIn("Warm-start improvement", html)
            self.assertIn(
                'Relative target tolerance ε<sub>X</sub> <select id="heatmap-margin">',
                html,
            )
            self.assertLess(
                html.find('id="heatmap-title"'),
                html.find('id="heatmap-view"'),
            )
            self.assertLess(
                html.find('id="heatmap-view"'),
                html.find('id="heatmap-margin"'),
            )
            self.assertLess(
                html.find('id="heatmap-margin"'),
                html.find('id="heatmap-algorithms"'),
            )
            self.assertIn("#F2F2F2", html)
            self.assertIn('<option value="top:3">Top 3</option>', html)
            self.assertIn('<option value="top:10">Top 10</option>', html)
            self.assertNotIn("Success rate at 1% above the sealed optimum", html)
            self.assertIn(
                "p<sub>s,warm</sub>",
                payload["captions"]["improvement"]["1"],
            )
            self.assertIn(
                "each algorithm family has a cold-start row above its warm-start row",
                html,
            )
            self.assertIn(
                "The All algorithms row shows the mean success rate across all algorithm variants",
                html,
            )
            self.assertIn(
                "The All families row shows the mean warm-start improvement",
                html,
            )
            self.assertIn('id="heatmap-png"', html)
            self.assertIn('id="heatmap-svg"', html)
            self.assertIn('id="heatmap-table"', html)
            self.assertIn('"xanchor": "right"', html)
            hidden = render_success_heatmap(
                load_experiments([experiment]),
                root / "figures-plain",
                downloads=False,
            )[0].read_text(encoding="utf-8")
            self.assertNotIn('id="heatmap-png"', hidden)

    def test_pooling_warning_is_printed_and_absent_from_the_page(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left_campaign"
            right = root / "right_campaign"
            write_experiment(
                left,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [100.0],
                    }
                ],
            )
            write_experiment(
                right,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [120.0],
                    }
                ],
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                written = render_success_heatmap(
                    load_experiments([left, right]),
                    root / "figures",
                )
            html = written[0].read_text(encoding="utf-8")
            warning = stderr.getvalue()
            self.assertIn("OPTUNA_TPE", warning)
            self.assertIn("L6750_B2", warning)
            self.assertIn("pools runs from", warning)
            self.assertIn("left_campaign", warning)
            self.assertIn("right_campaign", warning)
            self.assertNotIn("pools runs from", html)
            self.assertEqual(_payload(html)["text"]["1"][0], ["1/2"])
            self.assertEqual(
                _payload(html)["axis_labels"],
                ["Tree-Structured Parzen Estimator (cold start)"],
            )
            self.assertEqual(
                _payload(html)["hover"]["1"][0][0]["subtitle"],
                "Cold start · Penalised objective",
            )

    def test_axis_label_names_the_cold_and_warm_starts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
                        "problem_id": "L6750_A1",
                        "y_p": [100.0],
                    },
                    {
                        "algorithm_name": "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                        "problem_id": "L6750_A1",
                        "y_p": [100.0],
                        "repetition": 1,
                    },
                ],
            )
            html = render_success_heatmap(
                load_experiments([experiment]),
                root / "figures",
            )[0].read_text(encoding="utf-8")
            self.assertIn("<b>Local search.</b>", html)
            self.assertIn("Strict improvement moves only", html)
            self.assertIn("Equal descent also searches", html)
            payload = _payload(html)
            self.assertEqual(
                payload["axis_labels"],
                [
                    "Population Basin Hopping, SI-DESC (cold start, Strict improvement)",
                    "Population Basin Hopping, SI-DESC (warm start, Strict improvement)",
                ],
            )
            self.assertEqual(
                payload["hover"]["1"][0][0]["subtitle"],
                "Cold start, Strict improvement · Constraint-aware",
            )

    def test_heatmap_exclude_and_unknown_name(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [100.0],
                    },
                    {
                        "algorithm_name": "PYMOO_GA",
                        "problem_id": "L6750_B2",
                        "y_p": [100.0],
                        "repetition": 1,
                    },
                ],
            )
            bundle = load_experiments([experiment])
            written = render_interactive_pool(
                bundle,
                root / "kept",
                which=("heatmap",),
                exclude_algorithms=("PYMOO_GA",),
            )
            self.assertEqual(
                _payload(written[0].read_text(encoding="utf-8"))["algorithms"],
                ["OPTUNA_TPE"],
            )
            out = root / "rejected"
            with self.assertRaises(ValueError) as raised:
                render_interactive_pool(
                    bundle,
                    out,
                    which=("heatmap",),
                    exclude_algorithms=("MISSING",),
                )
            self.assertIn("MISSING", str(raised.exception))
            self.assertIn("OPTUNA_TPE", str(raised.exception))
            self.assertFalse(out.exists())

    def test_each_rate_has_one_vote(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks: list[dict] = []
            repetition = 0
            for value in [100.0] * 10 + [105.0] * 10:
                tasks.append(
                    {
                        "algorithm_name": "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
                        "problem_id": "L6750_A1",
                        "y_p": [value],
                        "repetition": repetition,
                    }
                )
                repetition += 1
            tasks.append(
                {
                    "algorithm_name": "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                    "problem_id": "L6750_A1",
                    "y_p": [100.0],
                    "repetition": repetition,
                }
            )
            repetition += 1
            tasks.append(
                {
                    "algorithm_name": "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                    "problem_id": "L8100_A1",
                    "y_p": [105.0],
                    "repetition": repetition,
                }
            )
            payload = _payload(
                render_success_heatmap(
                    load_experiments([_written(root, tasks)]),
                    root / "figures",
                )[0].read_text(encoding="utf-8")
            )
            self.assertEqual(payload["summary"]["rates"]["1"][0][0], 0.75)
            self.assertEqual(payload["summary"]["rates"]["1"][0][1], 0.0)
            self.assertEqual(payload["improvement"]["deltas"]["1"], [[50.0, None]])
            self.assertEqual(
                payload["improvement"]["summary"]["deltas"]["1"],
                [50.0, None],
            )
            self.assertEqual(
                payload["improvement"]["hover"]["1"][0][0]["cold_fraction"],
                "10/20",
            )
            self.assertEqual(
                payload["improvement"]["hover"]["1"][0][0]["warm_fraction"],
                "1/1",
            )
            self.assertEqual(
                payload["improvement"]["hover"]["1"][0][0]["difference"],
                "+50.0 pp",
            )

    def test_family_order_uses_better_rate_then_evaluations_then_name(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [
                {
                    "algorithm_name": "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
                    "problem_id": "L6750_A1",
                    "y_p": [100.0],
                    "repetition": 0,
                },
                {
                    "algorithm_name": "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                    "problem_id": "L6750_A1",
                    "y_p": [100.0],
                    "repetition": 1,
                },
                {
                    "algorithm_name": "K_ELITE_AMS_DEB_SI_DESC_COLD_START",
                    "problem_id": "L6750_A1",
                    "y_p": [100.0],
                    "repetition": 2,
                },
                {
                    "algorithm_name": "K_ELITE_AMS_DEB_SI_DESC_WARM_START",
                    "problem_id": "L6750_A1",
                    "y_p": [105.0],
                    "repetition": 3,
                },
            ]
            payload = _payload(
                render_success_heatmap(
                    load_experiments([_written(root, tasks)]),
                    root / "figures",
                )[0].read_text(encoding="utf-8")
            )
            self.assertEqual(
                [label.split(" (")[0] for label in payload["axis_labels"]],
                [
                    "Adaptive Multi-Start, SI-DESC",
                    "Adaptive Multi-Start, SI-DESC",
                    "Population Basin Hopping, SI-DESC",
                    "Population Basin Hopping, SI-DESC",
                ],
            )
            self.assertIn("cold start", payload["axis_labels"][0])
            self.assertIn("warm start", payload["axis_labels"][1])
            self.assertEqual(
                payload["improvement"]["labels"],
                ["Adaptive Multi-Start, SI-DESC", "Population Basin Hopping, SI-DESC"],
            )
            self.assertEqual(payload["improvement"]["deltas"]["1"], [[-100.0], [0.0]])
            self.assertEqual(payload["improvement"]["summary"]["deltas"]["1"], [-50.0])
            self.assertEqual(payload["family_gaps_after"], [1])

    def test_pooled_runs_outrank_an_unweighted_problem_mean(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks: list[dict] = [
                {
                    "algorithm_name": "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                    "problem_id": "L6750_A1",
                    "y_p": [100.0],
                    "repetition": 0,
                }
            ]
            for repetition in range(1, 21):
                tasks.append(
                    {
                        "algorithm_name": "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                        "problem_id": "L6750_B1",
                        "y_p": [105.0],
                        "repetition": repetition,
                    }
                )
            for repetition in range(21, 41):
                tasks.append(
                    {
                        "algorithm_name": "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                        "problem_id": "L6750_B2",
                        "y_p": [105.0],
                        "repetition": repetition,
                    }
                )
            for repetition, value in enumerate(
                [100.0, 105.0, 105.0, 105.0, 105.0],
                start=41,
            ):
                tasks.append(
                    {
                        "algorithm_name": "K_ELITE_AMS_DEB_SI_DESC_COLD_START",
                        "problem_id": "L8100_A1",
                        "y_p": [value],
                        "repetition": repetition,
                    }
                )
            payload = _payload(
                render_success_heatmap(
                    load_experiments([_written(root, tasks)]),
                    root / "figures",
                )[0].read_text(encoding="utf-8")
            )
            self.assertIn("Adaptive Multi-Start", payload["axis_labels"][0])
            self.assertIn("Population Basin Hopping", payload["axis_labels"][1])

    def test_row_order_follows_the_selected_tolerance(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [
                {
                    "algorithm_name": "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
                    "problem_id": "L6750_A1",
                    "y_p": [100.0],
                    "repetition": 0,
                },
                {
                    "algorithm_name": "K_ELITE_AMS_DEB_SI_DESC_COLD_START",
                    "problem_id": "L6750_A1",
                    "y_p": [105.0],
                    "repetition": 1,
                },
            ]
            payload = _payload(
                render_success_heatmap(
                    load_experiments([_written(root, tasks)]),
                    root / "figures",
                )[0].read_text(encoding="utf-8")
            )
            self.assertTrue(
                payload["axis_labels"][0].startswith("Population Basin Hopping")
            )
            self.assertEqual(payload["rates"]["10"][1][0], 1.0)
            at_one = payload["display"]["layouts"]["all"]["1"]["success"]["y_tickvals"]
            at_ten = payload["display"]["layouts"]["all"]["10"]["success"]["y_tickvals"]
            self.assertTrue(at_one[1].startswith("Population Basin Hopping"))
            self.assertTrue(at_ten[1].startswith("Adaptive Multi-Start"))

    def test_cold_stays_above_warm_when_warm_is_better(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [
                {
                    "algorithm_name": "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
                    "problem_id": "L6750_A1",
                    "y_p": [105.0],
                    "repetition": 0,
                },
                {
                    "algorithm_name": "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                    "problem_id": "L6750_A1",
                    "y_p": [100.0],
                    "repetition": 1,
                },
            ]
            payload = _payload(
                render_success_heatmap(
                    load_experiments([_written(root, tasks)]),
                    root / "figures",
                )[0].read_text(encoding="utf-8")
            )
            self.assertIn("cold start", payload["axis_labels"][0])
            self.assertIn("warm start", payload["axis_labels"][1])
            self.assertEqual(payload["improvement"]["deltas"]["1"], [[100.0]])

    def test_colliding_family_names_keep_the_constraint_label(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = []
            repetition = 0
            for algorithm_name in (
                "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
                "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
            ):
                tasks.append(
                    {
                        "algorithm_name": algorithm_name,
                        "problem_id": "L6750_A1",
                        "y_p": [100.0],
                        "repetition": repetition,
                    }
                )
                repetition += 1
            payload = _payload(
                render_success_heatmap(
                    load_experiments([_written(root, tasks)]),
                    root / "figures",
                )[0].read_text(encoding="utf-8")
            )
            self.assertIn("Constraint-aware", payload["axis_labels"][0])
            self.assertIn("cold start, Strict improvement", payload["axis_labels"][0])
            self.assertIn("Penalised objective", payload["axis_labels"][2])
            self.assertIn("cold start, Strict improvement", payload["axis_labels"][2])
            self.assertEqual(
                payload["improvement"]["labels"],
                [
                    "Population Basin Hopping, SI-DESC · Constraint-aware",
                    "Population Basin Hopping, SI-DESC · Penalised objective",
                ],
            )

    def test_top_families_drop_the_rest_and_the_mean_uses_only_them(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [
                {
                    "algorithm_name": "SCIPY_COBYLA",
                    "problem_id": "L6750_A1",
                    "y_p": [100.0],
                    "repetition": 0,
                },
                {
                    "algorithm_name": "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
                    "problem_id": "L6750_A1",
                    "y_p": [100.0],
                    "repetition": 1,
                },
                {
                    "algorithm_name": "RANDOM_SEARCH",
                    "problem_id": "L6750_A1",
                    "y_p": [150.0],
                    "repetition": 2,
                },
                {
                    "algorithm_name": "OPTUNA_TPE",
                    "problem_id": "L6750_A1",
                    "y_p": [150.0],
                    "repetition": 3,
                },
            ]
            payload = _payload(
                render_success_heatmap(
                    load_experiments([_written(root, tasks)]),
                    root / "figures",
                )[0].read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["label"] for item in payload["algorithm_menu"]],
                [
                    "All algorithms",
                    "Top 3",
                    "Top 5",
                    "Top 10",
                    "Metaheuristic",
                    "Model-based",
                    "Direct search",
                    "Sampling",
                ],
            )
            shown = payload["display"]["filters"]["top:3"]["success"]["y_tickvals"]
            self.assertEqual(
                shown,
                [
                    "All algorithms",
                    "COBYLA (cold start)",
                    "Population Basin Hopping, SI-DESC (cold start, Strict improvement)",
                    "Random search (cold start)",
                ],
            )
            column = payload["display"]["x"].index("L6750_A1")
            self.assertAlmostEqual(
                payload["display"]["filters"]["top:3"]["success"]["z"]["1"][0][column],
                2.0 / 3.0,
            )
            direct = payload["display"]["filters"]["class:Direct search"]["success"]
            self.assertEqual(
                direct["y_tickvals"],
                ["All algorithms", "COBYLA (cold start)"],
            )
            self.assertEqual(direct["z"]["1"][0][column], 1.0)


def _written(root: Path, tasks: list[dict]) -> Path:
    experiment = root / "campaign"
    write_experiment(experiment, tasks)
    return experiment


class PlotCommandTests(unittest.TestCase):
    def test_suite_cli_exclude_allows_a_figure_and_otherwise_refuses(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_A1",
                        "y_p": [100.0],
                    },
                    {
                        "algorithm_name": "CMAES_CATCMAWM",
                        "problem_id": "L8100_B2",
                        "state": "failed",
                        "repetition": 1,
                    },
                ],
            )
            refused = root / "refused"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(
                    [
                        "suite",
                        "--exp",
                        str(experiment),
                        "--out",
                        str(refused),
                        "--plots",
                        "summary",
                        "--problem-id",
                        "L6750_A1",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("failed", stderr.getvalue())
            self.assertIn("L8100_B2", stderr.getvalue())
            self.assertFalse(refused.exists())
            allowed = root / "allowed"
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "suite",
                        "--exp",
                        str(experiment),
                        "--out",
                        str(allowed),
                        "--plots",
                        "summary",
                        "--problem-id",
                        "L6750_A1",
                        "--exclude",
                        "CMAES_CATCMAWM",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(any(allowed.glob("summary_by_target.csv")))

    def test_problem_id_of_an_unfinished_problem_names_that_task(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_A1",
                        "y_p": [100.0],
                    },
                    {
                        "algorithm_name": "CMAES_CATCMAWM",
                        "problem_id": "L8100_B2",
                        "state": "running",
                        "repetition": 1,
                    },
                ],
            )
            out = root / "figures"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(
                    [
                        "suite",
                        "--exp",
                        str(experiment),
                        "--out",
                        str(out),
                        "--plots",
                        "summary",
                        "--problem-id",
                        "L8100_B2",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("running", stderr.getvalue())
            self.assertIn("CMAES_CATCMAWM", stderr.getvalue())
            self.assertNotIn("unknown problem_id", stderr.getvalue())
            self.assertFalse(out.exists())

    def test_failed_only_directory_is_excluded_before_the_loader_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good"
            failed = root / "failed_only"
            write_experiment(
                good,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [100.0],
                    }
                ],
            )
            write_experiment(
                failed,
                [
                    {
                        "algorithm_name": "CMAES_CATCMAWM",
                        "problem_id": "L8100_B2",
                        "state": "cancelled",
                    }
                ],
            )
            refused = root / "refused"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(
                    [
                        "interactive",
                        "--plots",
                        "heatmap",
                        "--exp",
                        str(good),
                        "--exp",
                        str(failed),
                        "--out",
                        str(refused),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("cancelled", stderr.getvalue())
            self.assertIn("CMAES_CATCMAWM", stderr.getvalue())
            self.assertNotIn("no completed tasks", stderr.getvalue())
            self.assertFalse(refused.exists())
            allowed = root / "allowed"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "interactive",
                        "--plots",
                        "heatmap",
                        "--exp",
                        str(good),
                        "--exp",
                        str(failed),
                        "--out",
                        str(allowed),
                        "--exclude",
                        "CMAES_CATCMAWM",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue((allowed / "success_heatmap.html").is_file())

    def test_commands_refuse_unfinished_repetitions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [100.0],
                    },
                    {
                        "algorithm_name": "CMAES_CATCMAWM",
                        "problem_id": "L6750_B2",
                        "state": "cancelled",
                        "repetition": 1,
                    },
                ],
            )
            for command in ("suite", "compare", "interactive"):
                out = root / command
                stderr = io.StringIO()
                argv = [
                    command,
                    "--exp",
                    str(experiment),
                    "--out",
                    str(out),
                ]
                if command in ("suite", "compare"):
                    argv.extend(["--plots", "summary"])
                with redirect_stderr(stderr):
                    code = main(argv)
                self.assertEqual(code, 2, command)
                self.assertIn("cancelled", stderr.getvalue())
                self.assertIn("CMAES_CATCMAWM", stderr.getvalue())
                self.assertFalse(out.exists(), command)

    def test_heatmap_command_writes_one_html_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            out = root / "figures"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [100.0],
                    }
                ],
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "interactive",
                        "--plots",
                        "heatmap",
                        "--exp",
                        str(experiment),
                        "--out",
                        str(out),
                    ]
                )
            self.assertEqual(code, 0)
            html = out / "success_heatmap.html"
            self.assertTrue(html.is_file())
            self.assertEqual(
                [path.name for path in out.glob("*.html")],
                ["success_heatmap.html"],
            )
            self.assertIn(str(html), stdout.getvalue())
            self.assertEqual(
                _payload(html.read_text(encoding="utf-8"))["text"]["1"],
                [["1/1"]],
            )

    def test_interactive_can_write_the_heatmap_with_the_other_selected_plots(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "campaign"
            out = root / "figures"
            write_experiment(
                experiment,
                [
                    {
                        "algorithm_name": "OPTUNA_TPE",
                        "problem_id": "L6750_B2",
                        "y_p": [120.0, 100.0],
                    }
                ],
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "interactive",
                        "--exp",
                        str(experiment),
                        "--out",
                        str(out),
                        "--plots",
                        "heatmap,ecdf,ert",
                    ]
                )
            self.assertEqual(code, 0)
            names = sorted(path.name for path in out.glob("*.html"))
            self.assertEqual(
                names,
                [
                    "ecdf_success_pooled.html",
                    "ert_pooled.html",
                    "success_heatmap.html",
                ],
            )
            self.assertFalse((out / "convergence_pooled.html").exists())
