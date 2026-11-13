"""
Phase V1.3 gate: hdb-plot CLI help and end-to-end suite on multi-dir fixtures.
"""
from __future__ import annotations

import csv
import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

from core.visualization_core.optimization.cli import build_parser, main


def _write_dat(path: Path, *, y_p: list[float], c_a: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "evaluations",
        "raw_y",
        "t_eval_s",
        "y",
        "y_p",
        "misses",
        "hits",
        "c__A",
    ]
    lines = [" ".join(header)]
    for i, (yp, ca) in enumerate(zip(y_p, c_a, strict=True), start=1):
        lines.append(
            " ".join(
                str(v)
                for v in [float(i), yp, 0.01 * i, yp, yp, 0.0, float(i), ca]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_experiment(
    root: Path,
    *,
    algorithm_name: str,
    runs: list[dict],
    problem_id: str = "L6750_B2",
) -> None:
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
    ]
    rows: list[dict[str, object]] = []
    for item in runs:
        task_index = int(item["task_index"])
        attempt_dir = (
            root / "tasks" / f"task_{task_index:06d}" / "attempts" / "001"
        )
        dat = (
            attempt_dir
            / "ioh"
            / "run"
            / "data_f0_1"
            / "IOHprofiler_f0_DIM2.dat"
        )
        _write_dat(dat, y_p=list(item["y_p"]), c_a=list(item["c_a"]))
        summary_path = attempt_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "algorithm_name": algorithm_name,
                    "task_index": task_index,
                    "repetition": int(item["repetition"]),
                    "evaluations": len(item["y_p"]),
                    "algorithm_parameters": {
                        "max_evaluations": int(item["max_evaluations"])
                    },
                    "termination_mode": "fixed_target",
                    "global_optimum": 100.0,
                    "allowed_deviation_percent": 1.0,
                    "target_threshold": 101.0,
                    "target_reference": "all",
                    "target_hit": bool(item.get("target_hit", False)),
                    "evaluations_to_target": item.get("evaluations_to_target"),
                    "budget_exhausted": not bool(item.get("target_hit", False)),
                    "stop_reason": "target_hit"
                    if item.get("target_hit")
                    else "budget_exhausted",
                    "wall_time_s": float(item.get("wall_time_s", 1.5)),
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
                "repetition": int(item["repetition"]),
                "attempt": 1,
                "state": "completed",
                "artifact_path": str(summary_path),
                "global_optimum": 100.0,
                "target_hit": bool(item.get("target_hit", False)),
                "evaluations_to_target": item.get("evaluations_to_target", ""),
                "wall_time_s": float(item.get("wall_time_s", 1.5)),
            }
        )
    with (root / "aggregate" / "tasks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields + ["wall_time_s"],
        )
        writer.writeheader()
        writer.writerows(rows)
    (root / "aggregate" / "summary.json").write_text("{}", encoding="utf-8")


class HdbPlotCliTests(unittest.TestCase):
    def test_help_exits_zero(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_suite_help_exits_zero(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["suite", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_interactive_downloads_default_on_and_can_be_turned_off(self) -> None:
        parser = build_parser()
        shown = parser.parse_args(["interactive", "--exp", "a", "--out", "out"])
        hidden = parser.parse_args(
            ["interactive", "--exp", "a", "--out", "out", "--no-downloads"]
        )
        forced = parser.parse_args(
            ["interactive", "--exp", "a", "--out", "out", "--downloads"]
        )
        self.assertTrue(shown.downloads)
        self.assertFalse(hidden.downloads)
        self.assertTrue(forced.downloads)

    def test_interactive_plots_default_to_every_figure_and_reject_unknown_names(self) -> None:
        from core.visualization_core.optimization.plots.interactive_pool import (
            INTERACTIVE_PLOTS,
            selected_interactive_plots,
        )

        parser = build_parser()
        shown = parser.parse_args(["interactive", "--exp", "a", "--out", "out"])
        chosen = parser.parse_args(
            [
                "interactive",
                "--exp",
                "a",
                "--out",
                "out",
                "--plots",
                "heatmap, ecdf, ert",
            ]
        )
        self.assertEqual(
            selected_interactive_plots(shown.plots.split(",")),
            INTERACTIVE_PLOTS,
        )
        self.assertEqual(
            selected_interactive_plots(
                [part.strip() for part in chosen.plots.split(",")]
            ),
            ("heatmap", "ecdf", "ert"),
        )
        with self.assertRaises(ValueError) as raised:
            selected_interactive_plots(("heatmap", "convergence_runs"))
        self.assertIn("convergence_runs", str(raised.exception))
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(
                ["interactive", "--exp", "missing", "--out", "out", "--plots", "nope"]
            )
        self.assertEqual(code, 2)
        self.assertIn("nope", stderr.getvalue())

    def test_interactive_exclude_accepts_a_repeated_or_comma_separated_list(self) -> None:
        from core.visualization_core.optimization.cli import _exclude_names

        parser = build_parser()
        repeated = parser.parse_args(
            [
                "interactive",
                "--exp",
                "a",
                "--out",
                "out",
                "--exclude",
                "AMS",
                "--exclude",
                "GA",
            ]
        )
        comma = parser.parse_args(
            ["interactive", "--exp", "a", "--out", "out", "--exclude", "AMS, GA"]
        )
        self.assertEqual(_exclude_names(repeated.exclude_algorithms), ("AMS", "GA"))
        self.assertEqual(_exclude_names(comma.exclude_algorithms), ("AMS", "GA"))
        self.assertEqual(_exclude_names(None), ())

    def test_load_reports_progress_for_each_completed_run(self) -> None:
        from core.visualization_core.optimization.io import load_experiments

        events: list[tuple[int, int, str]] = []
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "study_a"
            _write_experiment(
                root,
                algorithm_name="OPTUNA_TPE",
                runs=[
                    {
                        "task_index": 0,
                        "repetition": 0,
                        "y_p": [120.0, 100.0],
                        "c_a": [0.5, 0.5],
                        "max_evaluations": 10,
                        "target_hit": True,
                        "evaluations_to_target": 2,
                    },
                    {
                        "task_index": 1,
                        "repetition": 1,
                        "y_p": [130.0, 120.0],
                        "c_a": [0.5, 0.5],
                        "max_evaluations": 10,
                        "target_hit": False,
                    },
                ],
            )
            load_experiments(
                [root],
                progress=lambda done, total, message: events.append(
                    (done, total, message)
                ),
            )
        self.assertEqual(events[0], (0, 2, "loading study_a"))
        self.assertEqual(events[-1][0], 2)
        self.assertEqual(events[-1][1], 2)
        self.assertIn("OPTUNA_TPE", events[-1][2])

    def test_status_line_rewrites_itself_on_a_terminal(self) -> None:
        import io

        from core.visualization_core.optimization.cli import _StatusReporter

        class _TTY(io.StringIO):
            def isatty(self) -> bool:
                return True

        stream = _TTY()
        reporter = _StatusReporter(stream)
        reporter(1, 4, "loading study A")
        reporter(2, 4, "loading study B")
        reporter.finish()
        text = stream.getvalue()
        self.assertIn("\r", text)
        self.assertIn("loading study B (2/4)", text)
        self.assertTrue(text.endswith("\n"))

    def test_status_line_throttles_loading_off_a_terminal(self) -> None:
        import io

        from core.visualization_core.optimization.cli import _StatusReporter

        stream = io.StringIO()
        reporter = _StatusReporter(stream)
        for done in range(1001):
            reporter(done, 1000, f"loading study step {done}")
        reporter(1, 5, "writing ecdf_success_pooled.html")
        reporter(1, 5, "writing ecdf_success_pooled.html: all problems, 1%")
        lines = stream.getvalue().splitlines()
        loading = [line for line in lines if "loading study" in line]
        self.assertLess(len(loading), 30)
        self.assertGreater(len(loading), 5)
        self.assertIn("(0/1000)", loading[0])
        self.assertIn("(1000/1000)", loading[-1])
        self.assertEqual(
            [line for line in lines if line.startswith("hdb-plot: writing")],
            [
                "hdb-plot: writing ecdf_success_pooled.html (1/5)",
                "hdb-plot: writing ecdf_success_pooled.html: all problems, 1% (1/5)",
            ],
        )

    def test_suite_end_to_end_multi_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_a = root / "study_a"
            exp_b = root / "study_b"
            out = root / "figures"
            _write_experiment(
                exp_a,
                algorithm_name="OPTUNA_TPE",
                runs=[
                    {
                        "task_index": 0,
                        "repetition": 0,
                        "y_p": [120.0, 110.0, 100.5],
                        "c_a": [0.5, 0.5, 0.5],
                        "max_evaluations": 10,
                        "target_hit": True,
                        "evaluations_to_target": 3,
                    },
                    {
                        "task_index": 1,
                        "repetition": 1,
                        "y_p": [130.0, 125.0, 120.0],
                        "c_a": [0.5, 0.5, 0.5],
                        "max_evaluations": 10,
                        "target_hit": False,
                    },
                ],
            )
            _write_experiment(
                exp_b,
                algorithm_name="PYMOO_GA",
                runs=[
                    {
                        "task_index": 0,
                        "repetition": 0,
                        "y_p": [115.0, 100.5],
                        "c_a": [0.4, 0.4],
                        "max_evaluations": 8,
                        "target_hit": True,
                        "evaluations_to_target": 2,
                    }
                ],
            )
            code = main(
                [
                    "suite",
                    "--exp",
                    str(exp_a),
                    "--exp",
                    str(exp_b),
                    "--out",
                    str(out),
                    "--display",
                    "median",
                    "--success-filter",
                    "successes",
                ]
            )
            self.assertEqual(code, 0)
            for stem in (
                "convergence",
                "ert",
                "box_runtime",
                "hist_runtime",
                "ecdf_success",
                "violin_runtime",
                "bar_first_feasible",
                "bar_wall_time",
            ):
                self.assertTrue((out / f"{stem}.png").is_file())
                self.assertTrue((out / f"{stem}.pdf").is_file())
            self.assertTrue((out / "summary_by_target.csv").is_file())
            self.assertTrue((out / "summary_by_target.md").is_file())
            self.assertFalse((out / "ecdf_runtime.png").is_file())

    def test_compare_cli_end_to_end(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_a = root / "study_p1"
            exp_b = root / "study_p2"
            out = root / "figures_compare"
            _write_experiment(
                exp_a,
                algorithm_name="OPTUNA_TPE",
                runs=[
                    {
                        "task_index": 0,
                        "repetition": 0,
                        "y_p": [120.0, 110.0, 101.0],
                        "c_a": [0.5, 0.5, 0.5],
                        "max_evaluations": 10,
                        "target_hit": True,
                        "evaluations_to_target": 3,
                    },
                ],
            )
            _write_experiment(
                exp_b,
                algorithm_name="OPTUNA_TPE",
                runs=[
                    {
                        "task_index": 0,
                        "repetition": 0,
                        "y_p": [130.0, 105.0],
                        "c_a": [0.4, 0.4],
                        "max_evaluations": 8,
                        "target_hit": True,
                        "evaluations_to_target": 2,
                    }
                ],
            )
            code = main(
                [
                    "compare",
                    "--exp",
                    str(exp_a),
                    "--exp",
                    str(exp_b),
                    "--panel-title",
                    "P1",
                    "--panel-title",
                    "P2",
                    "--out",
                    str(out),
                    "--title",
                    "Two problems",
                    "--global-optimum-per-exp",
                    "100",
                    "--global-optimum-per-exp",
                    "100",
                ]
            )
            self.assertEqual(code, 0)
            for stem in (
                "convergence_compare",
                "ert_compare",
                "box_runtime_compare",
                "hist_runtime_compare",
                "ecdf_success_compare",
                "violin_runtime_compare",
                "bar_first_feasible_compare",
                "bar_wall_time_compare",
            ):
                self.assertTrue((out / f"{stem}.png").is_file())
                self.assertTrue((out / f"{stem}.pdf").is_file())
            self.assertTrue((out / "summary_by_target_compare.csv").is_file())
            self.assertTrue((out / "summary_by_target_compare.md").is_file())
            # Pooled across panels: same algorithm → n_runs = 1+1 = 2
            import csv as _csv

            with (out / "summary_by_target_compare.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(_csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 1)
            self.assertTrue(all(r["algorithm"] == "OPTUNA_TPE" for r in rows))
            self.assertTrue(all(r["n_runs"] == "2" for r in rows))
            self.assertNotIn("panel", rows[0])
            self.assertIn("mean_wall_time_s", rows[0])
            self.assertFalse((out / "ecdf_runtime_compare.png").is_file())

    def test_missing_experiment_exits_2(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            missing = Path(tmp) / "missing"
            code = main(
                [
                    "suite",
                    "--exp",
                    str(missing),
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(code, 2)

    def test_suite_requires_problem_selection_for_mixed_problems(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiments = []
            for index, problem_id in enumerate(("P1", "P2")):
                exp = root / problem_id
                _write_experiment(
                    exp,
                    algorithm_name="ALG",
                    problem_id=problem_id,
                    runs=[
                        {
                            "task_index": index,
                            "repetition": 0,
                            "y_p": [101.0],
                            "c_a": [0.5],
                            "max_evaluations": 2,
                            "target_hit": True,
                            "evaluations_to_target": 1,
                        }
                    ],
                )
                experiments.extend(["--exp", str(exp)])

            args = ["suite", *experiments, "--out", str(root / "rejected")]
            self.assertEqual(main(args), 2)
            self.assertEqual(
                main(
                    [
                        *args[:-1],
                        str(root / "selected"),
                        "--problem-id",
                        "P2",
                        "--plots",
                        "summary",
                    ]
                ),
                0,
            )
            self.assertTrue(
                (root / "selected" / "summary_by_target.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
