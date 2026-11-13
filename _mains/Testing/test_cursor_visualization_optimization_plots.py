"""
Phase V1.2 gate: optimisation plot renderers and render_suite smoke tests.
"""
from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.visualization_core.optimization.api import PlotApiError, render_suite
from core.visualization_core.optimization.io import load_experiments
from core.visualization_core.optimization.models import PlotConfig


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
                for v in [
                    float(i),
                    yp,
                    0.01 * i,
                    yp,
                    yp,
                    0.0,
                    float(i),
                    ca,
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_experiment(
    root: Path,
    *,
    algorithm_name: str,
    runs: list[dict],
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
        "wall_time_s",
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
                "problem_id": "L6750_B2",
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "aggregate" / "summary.json").write_text("{}", encoding="utf-8")


class RenderSuiteSmokeTests(unittest.TestCase):
    def test_render_suite_writes_png_and_pdf_for_multi_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_a = root / "study_a"
            exp_b = root / "study_b"
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
            bundle = load_experiments([exp_a, exp_b])
            out = root / "figures"
            written = render_suite(
                bundle,
                PlotConfig(display="median", success_filter="successes"),
                out,
            )
            stems = {
                "convergence",
                "ert",
                "box_runtime",
                "hist_runtime",
                "ecdf_success",
                "violin_runtime",
                "bar_first_feasible",
                "bar_wall_time",
            }
            for stem in stems:
                png = out / f"{stem}.png"
                pdf = out / f"{stem}.pdf"
                self.assertTrue(png.is_file(), f"missing {png}")
                self.assertTrue(pdf.is_file(), f"missing {pdf}")
                self.assertGreater(png.stat().st_size, 0)
                self.assertGreater(pdf.stat().st_size, 0)
            self.assertTrue((out / "summary_by_target.csv").is_file())
            self.assertTrue((out / "summary_by_target.md").is_file())
            self.assertFalse((out / "ecdf_runtime.png").is_file())
            self.assertEqual(len(written), 18)

    def test_convergence_relative_axis_100_to_300(self) -> None:
        from unittest.mock import patch

        from core.visualization_core.optimization.plots import convergence as conv

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp = root / "study"
            _write_experiment(
                exp,
                algorithm_name="ALG",
                runs=[
                    {
                        "task_index": 0,
                        "repetition": 0,
                        "y_p": [250.0, 150.0, 105.0],
                        "c_a": [0.5, 0.5, 0.5],
                        "max_evaluations": 10,
                        "target_hit": True,
                        "evaluations_to_target": 3,
                    },
                    {
                        "task_index": 1,
                        "repetition": 1,
                        "y_p": [280.0, 200.0, 120.0],
                        "c_a": [0.5, 0.5, 0.5],
                        "max_evaluations": 10,
                        "target_hit": False,
                    },
                ],
            )
            bundle = load_experiments([exp])
            out = root / "figures"
            captured: dict[str, object] = {}
            real_save = conv.save_figure

            def _capture(fig, out_dir, stem):
                ax = fig.axes[0]
                captured["ylim"] = ax.get_ylim()
                captured["ylabel"] = ax.get_ylabel()
                return real_save(fig, out_dir, stem)

            with patch.object(conv, "save_figure", side_effect=_capture):
                conv.render_convergence(
                    bundle,
                    PlotConfig(display="median", show_iqr=True),
                    out,
                )
            y_lo, y_hi = captured["ylim"]  # type: ignore[misc]
            self.assertAlmostEqual(float(y_lo), 95.0)
            self.assertAlmostEqual(float(y_hi), 300.0)
            self.assertIn("[%]", str(captured["ylabel"]))
            self.assertAlmostEqual(
                float(conv._to_relative_percent([250.0], 100.0)[0]),
                250.0,
            )
            self.assertTrue((out / "convergence.png").is_file())

    def test_render_suite_rejects_unknown_plot(self) -> None:
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "one"
            _write_experiment(
                exp,
                algorithm_name="ALG",
                runs=[
                    {
                        "task_index": 0,
                        "repetition": 0,
                        "y_p": [100.5],
                        "c_a": [0.1],
                        "max_evaluations": 5,
                        "target_hit": True,
                        "evaluations_to_target": 1,
                    }
                ],
            )
            bundle = load_experiments([exp])
            with self.assertRaises(PlotApiError):
                render_suite(
                    bundle,
                    PlotConfig(),
                    Path(tmp) / "out",
                    which=("nope",),
                )


if __name__ == "__main__":
    unittest.main()
