"""
Phase V1.0 gate: optimisation experiment loaders and trajectory helpers.
"""
from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from core.visualization_core.optimization.io import (
    ExperimentLoadError,
    load_experiments,
    parse_ioh_dat,
)
from core.visualization_core.optimization.trajectories import (
    TrajectoryError,
    best_so_far,
    feasible_mask_from_constraints,
    first_hitting_time,
    hit_time_at_percent,
    hit_times_at_percents,
    target_threshold,
)


def _write_dat(
    path: Path,
    *,
    y_p: list[float],
    c_a: list[float] | None = None,
    c_b: list[float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["evaluations", "raw_y", "t_eval_s", "y", "y_p", "misses", "hits"]
    rows: list[list[float]] = []
    for i, yp in enumerate(y_p, start=1):
        row = [float(i), yp, float(i) * 0.01, yp, yp, 0.0, float(i)]
        rows.append(row)
    if c_a is not None:
        header.append("c__A")
        for i, value in enumerate(c_a):
            rows[i].append(float(value))
    if c_b is not None:
        header.append("c__B")
        for i, value in enumerate(c_b):
            rows[i].append(float(value))
    lines = [" ".join(header)]
    for row in rows:
        lines.append(" ".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_experiment(
    root: Path,
    *,
    algorithm_name: str,
    runs: list[dict],
    global_optimum: float = 100.0,
    target_reference: str = "all",
) -> None:
    """
    Create a minimal succeeded-experiment layout for loader tests.

    Parameters
    ----------
    root : Path
        Experiment directory.
    algorithm_name : str
        Algorithm label for all runs.
    runs : list of dict
        Each item needs ``task_index``, ``repetition``, ``y_p``, optional
        ``c_a`` / ``c_b``, ``max_evaluations``, ``target_hit``,
        ``evaluations_to_target``.
    global_optimum : float
        Sealed optimum written into summaries / CSV.
    target_reference : str
        Sealed target reference.
    """
    (root / "aggregate").mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "task_index",
        "task_id",
        "problem_id",
        "algorithm_name",
        "repetition",
        "random_seed",
        "attempt",
        "state",
        "best_objective",
        "best_x_json",
        "hits",
        "misses",
        "evaluations",
        "wall_time_s",
        "artifact_path",
        "failure_kind",
        "failure_message",
        "termination_mode",
        "global_optimum",
        "allowed_deviation_percent",
        "target_threshold",
        "target_hit",
        "evaluations_to_target",
        "budget_exhausted",
        "stop_reason",
    ]
    csv_rows: list[dict[str, object]] = []
    for item in runs:
        task_index = int(item["task_index"])
        attempt = 1
        attempt_dir = (
            root
            / "tasks"
            / f"task_{task_index:06d}"
            / "attempts"
            / f"{attempt:03d}"
        )
        dat_dir = attempt_dir / "ioh" / "run" / "data_f0_1"
        dat_path = dat_dir / "IOHprofiler_f0_DIM2.dat"
        _write_dat(
            dat_path,
            y_p=list(item["y_p"]),
            c_a=item.get("c_a"),
            c_b=item.get("c_b"),
        )
        max_evaluations = int(item.get("max_evaluations", len(item["y_p"])))
        target_hit = bool(item.get("target_hit", False))
        e2t = item.get("evaluations_to_target")
        summary = {
            "algorithm_name": algorithm_name,
            "problem_id": str(item.get("problem_id", "L6750_B2")),
            "task_index": task_index,
            "repetition": int(item["repetition"]),
            "evaluations": len(item["y_p"]),
            "algorithm_parameters": {"max_evaluations": max_evaluations},
            "termination_mode": "fixed_target",
            "global_optimum": global_optimum,
            "allowed_deviation_percent": 1.0,
            "target_threshold": global_optimum * 1.01,
            "target_reference": target_reference,
            "target_hit": target_hit,
            "evaluations_to_target": e2t,
            "budget_exhausted": not target_hit,
            "stop_reason": "target_hit" if target_hit else "budget_exhausted",
            "best_objective": min(item["y_p"]),
        }
        summary_path = attempt_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        csv_rows.append(
            {
                "task_index": task_index,
                "task_id": f"task_{task_index:06d}",
                "problem_id": str(item.get("problem_id", "L6750_B2")),
                "algorithm_name": algorithm_name,
                "repetition": int(item["repetition"]),
                "random_seed": 1,
                "attempt": attempt,
                "state": "completed",
                "best_objective": min(item["y_p"]),
                "best_x_json": "[]",
                "hits": 0,
                "misses": 0,
                "evaluations": len(item["y_p"]),
                "wall_time_s": 0.1,
                "artifact_path": str(summary_path),
                "failure_kind": "",
                "failure_message": "",
                "termination_mode": "fixed_target",
                "global_optimum": global_optimum,
                "allowed_deviation_percent": 1.0,
                "target_threshold": global_optimum * 1.01,
                "target_hit": target_hit,
                "evaluations_to_target": "" if e2t is None else e2t,
                "budget_exhausted": not target_hit,
                "stop_reason": summary["stop_reason"],
            }
        )
    with (root / "aggregate" / "tasks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    (root / "aggregate" / "summary.json").write_text(
        json.dumps({"task_count": len(csv_rows)}),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text("{}", encoding="utf-8")


class TrajectoryHelperTests(unittest.TestCase):
    def test_best_so_far_is_cumulative_minimum(self) -> None:
        curve = best_so_far(np.asarray([5.0, 4.0, 6.0, 3.5, 3.5]))
        np.testing.assert_allclose(curve, [5.0, 4.0, 4.0, 3.5, 3.5])

    def test_feasible_mask_requires_c_columns(self) -> None:
        with self.assertRaises(TrajectoryError):
            feasible_mask_from_constraints({})

    def test_feasible_mask_and_best_so_far_use_every_value(self) -> None:
        mask = feasible_mask_from_constraints(
            {
                "c__A": np.asarray([0.5, 1.5, 0.8, 0.2]),
                "c__B": np.asarray([0.9, 0.9, 0.9, 0.9]),
            }
        )
        np.testing.assert_array_equal(mask, [True, False, True, True])
        y_p = np.asarray([10.0, 1.0, 8.0, 7.0])
        curve = best_so_far(y_p)
        np.testing.assert_allclose(curve, [10.0, 1.0, 1.0, 1.0])

    def test_target_threshold_and_hitting_time(self) -> None:
        self.assertAlmostEqual(target_threshold(100.0, 1.0), 101.0)
        evals = np.asarray([1, 2, 3, 4], dtype=int)
        curve = np.asarray([110.0, 105.0, 101.0, 99.0])
        self.assertEqual(first_hitting_time(curve, evals, 101.0), 3)
        self.assertIsNone(first_hitting_time(curve, evals, 90.0))


class ExperimentLoaderTests(unittest.TestCase):
    def test_parse_ioh_dat_requires_y_p(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.dat"
            path.write_text("evaluations raw_y\n1 2\n", encoding="utf-8")
            with self.assertRaises(ExperimentLoadError):
                parse_ioh_dat(path)

    def test_multi_directory_overlay_grouping(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_a = root / "study_optuna_tpe_p1"
            exp_b = root / "study_pymoo_ga_p1"
            _write_experiment(
                exp_a,
                algorithm_name="OPTUNA_TPE",
                runs=[
                    {
                        "task_index": 0,
                        "repetition": 0,
                        "y_p": [120.0, 110.0, 100.5],
                        "c_a": [0.5, 0.5, 0.5],
                        "c_b": [0.5, 0.5, 0.5],
                        "max_evaluations": 10,
                        "target_hit": True,
                        "evaluations_to_target": 3,
                    },
                    {
                        "task_index": 1,
                        "repetition": 1,
                        "y_p": [130.0, 125.0, 120.0],
                        "c_a": [0.5, 0.5, 0.5],
                        "c_b": [0.5, 0.5, 0.5],
                        "max_evaluations": 10,
                        "target_hit": False,
                        "evaluations_to_target": None,
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
                        "y_p": [115.0, 101.0],
                        "c_a": [0.4, 0.4],
                        "c_b": [0.4, 0.4],
                        "max_evaluations": 8,
                        "target_hit": True,
                        "evaluations_to_target": 2,
                    }
                ],
            )
            bundle = load_experiments([exp_a, exp_b])
            self.assertEqual(bundle.overlay_labels(), ("OPTUNA_TPE", "PYMOO_GA"))
            self.assertEqual(len(bundle.runs), 3)
            self.assertEqual(len(bundle.runs_for("OPTUNA_TPE")), 2)
            self.assertEqual(bundle.runs[0].max_evaluations, 10)
            np.testing.assert_allclose(
                bundle.runs[0].y_p_best_so_far_all,
                [120.0, 110.0, 100.5],
            )

    def test_multi_problem_bundle_retains_and_filters_problem_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "study"
            _write_experiment(
                exp,
                algorithm_name="RANDOM_SEARCH",
                runs=[
                    {
                        "task_index": 0,
                        "repetition": 0,
                        "problem_id": "L6750_A1",
                        "y_p": [3.0, 2.0],
                    },
                    {
                        "task_index": 1,
                        "repetition": 0,
                        "problem_id": "L6750_B2",
                        "y_p": [5.0, 4.0],
                    },
                ],
            )
            bundle = load_experiments([exp])

        self.assertEqual(
            bundle.problem_ids(),
            ("L6750_A1", "L6750_B2"),
        )
        filtered = bundle.for_problem("L6750_B2")
        self.assertEqual(len(filtered.runs), 1)
        self.assertEqual(filtered.runs[0].problem_id, "L6750_B2")
        with self.assertRaisesRegex(ValueError, "available"):
            bundle.for_problem("missing")

    def test_duplicate_algorithm_names_disambiguated_by_stem(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_a = root / "seed_a"
            exp_b = root / "seed_b"
            for path in (exp_a, exp_b):
                _write_experiment(
                    path,
                    algorithm_name="RANDOM_SEARCH",
                    runs=[
                        {
                            "task_index": 0,
                            "repetition": 0,
                            "y_p": [2.0, 1.5],
                            "c_a": [0.1, 0.1],
                            "max_evaluations": 5,
                        }
                    ],
                )
            bundle = load_experiments([exp_a, exp_b])
            self.assertEqual(
                set(bundle.overlay_labels()),
                {"RANDOM_SEARCH [seed_a]", "RANDOM_SEARCH [seed_b]"},
            )

    def test_hit_times_at_1_2_5_10_percent(self) -> None:
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "study"
            # y* = 100 → thresholds 101, 102, 105, 110
            _write_experiment(
                exp,
                algorithm_name="OPTUNA_TPE",
                global_optimum=100.0,
                target_reference="all",
                runs=[
                    {
                        "task_index": 0,
                        "repetition": 0,
                        "y_p": [120.0, 112.0, 108.0, 103.0, 100.5],
                        "c_a": [0.5, 0.5, 0.5, 0.5, 0.5],
                        "max_evaluations": 20,
                        "target_hit": True,
                        "evaluations_to_target": 5,
                    }
                ],
            )
            bundle = load_experiments([exp])
            run = bundle.runs[0]
            hits = hit_times_at_percents(
                run,
                (1.0, 2.0, 5.0, 10.0),
                target_reference="all",
            )
            # y: 120, 112, 108, 103, 100.5 vs tau 110 / 105 / 102 / 101
            self.assertEqual(hits[10.0], 3)  # 108 <= 110
            self.assertEqual(hits[5.0], 4)  # 103 <= 105
            self.assertEqual(hits[2.0], 5)  # 100.5 <= 102
            self.assertEqual(hits[1.0], 5)  # 100.5 <= 101

    def test_hit_counts_every_finite_objective(self) -> None:
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "study"
            _write_experiment(
                exp,
                algorithm_name="PYMOO_GA",
                global_optimum=100.0,
                target_reference="feasible",
                runs=[
                    {
                        "task_index": 0,
                        "repetition": 0,
                        "y_p": [120.0, 90.0, 101.0],
                        "c_a": [0.5, 1.5, 0.5],
                        "max_evaluations": 10,
                        "target_hit": True,
                        "evaluations_to_target": 2,
                    }
                ],
            )
            bundle = load_experiments([exp])
            run = bundle.runs[0]
            self.assertEqual(run.target_reference, "all")
            self.assertEqual(hit_time_at_percent(run, 1.0), 2)

    def test_hit_without_constraint_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "study"
            _write_experiment(
                exp,
                algorithm_name="SCIPY_COBYLA",
                global_optimum=100.0,
                target_reference="all",
                runs=[
                    {
                        "task_index": 0,
                        "repetition": 0,
                        "y_p": [110.0, 100.0],
                        "max_evaluations": 5,
                    }
                ],
            )
            bundle = load_experiments([exp])
            run = bundle.runs[0]
            self.assertIsNone(run.feasible_mask)
            self.assertEqual(hit_time_at_percent(run, 1.0), 2)

    def test_missing_tasks_csv_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaises(ExperimentLoadError):
                load_experiments([empty])


if __name__ == "__main__":
    unittest.main()
