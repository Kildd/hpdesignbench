"""
Phase C3.5 gate: best-design extraction and near-optimum bands.
"""
from __future__ import annotations

import dataclasses
import math
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.lifecycle import (
    collect_experiment,
    dispatch_experiment,
    prepare_experiment,
)
from core.ioh_core.workflows.brute_force.analysis import (
    open_brute_force_dataset,
    open_query_engine,
    run_best_designs_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.engine import quote_ident
from core.ioh_core.workflows.brute_force.analysis.report import (
    NEAR_OPTIMUM_TOLERANCES_PERCENT,
)


def _write_bf_config(path: Path, *, experiment_id: str) -> Path:
    path.write_text(
        textwrap.dedent(
            f"""\
            schema_version = 1

            [experiment]
            id = "{experiment_id}"
            workflow = "brute_force"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [brute_force]
            rows_per_shard = 24
            max_rows = 48
            start_index = 0
            parquet_batch_rows = 24
            compression = "zstd"
            reuse_structural_artifacts = true
            acknowledge_large_run = true
            maximum_expected_rows = 1000000
            maximum_expected_bytes = 10000000000

            [execution]
            backend = "local"
            max_parallel_tasks = 2
            timeout_seconds = 3600
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        ),
        encoding="utf-8",
    )
    return path


def _run_tiny_bf(root: Path, *, experiment_id: str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = _write_bf_config(root / f"{experiment_id}.toml", experiment_id=experiment_id)
    experiment_dir = root / "experiments" / experiment_id
    prepare_experiment(config_path=config_path, output=experiment_dir)
    dispatch_experiment(experiment_dir)
    collect_experiment(experiment_dir, require_complete=True)
    return experiment_dir


def _install_view(engine, rows: list[tuple]) -> None:
    """
    Replace the parts view with a minimal synthetic design table.

    Row tuple layout:
    (global_index, y, y_p, analysis_ok, constraint_feasible, feasible)
    """
    conn = engine.connection
    conn.execute(f"DROP VIEW IF EXISTS {quote_ident(engine.view_name)}")
    conn.execute("DROP TABLE IF EXISTS toy_best")
    conn.execute(
        "CREATE TEMP TABLE toy_best ("
        "global_index BIGINT, "
        "shard_index INTEGER, "
        "problem_id VARCHAR, "
        "evaluation_time_s DOUBLE, "
        "analysis_ok BOOLEAN, "
        "constraint_feasible BOOLEAN, "
        "feasible BOOLEAN, "
        "y DOUBLE, "
        "y_p DOUBLE, "
        "penalty_sum DOUBLE, "
        "penalty_factor DOUBLE, "
        "analysis_error_stage VARCHAR, "
        "analysis_error_message VARCHAR, "
        "analysis_error_exception_type VARCHAR, "
        "analysis_error_material_id VARCHAR, "
        "analysis_error_json VARCHAR, "
        "remapped_utilizations VARCHAR[], "
        "cost_components_missing VARCHAR[], "
        '"idx__x" INTEGER, '
        '"param__x" DOUBLE'
        ")"
    )
    payload = []
    for global_index, y, y_p, analysis_ok, constraint_feasible, feasible in rows:
        penalty_factor = 1.0 if feasible else 2.0
        payload.append(
            (
                int(global_index),
                0,
                "1",
                0.01,
                bool(analysis_ok),
                bool(constraint_feasible),
                bool(feasible),
                float(y),
                float(y_p),
                float(penalty_factor - 1.0),
                float(penalty_factor),
                None,
                None,
                None,
                None,
                None,
                [],
                [],
                0,
                1.0,
            )
        )
    conn.executemany(
        "INSERT INTO toy_best VALUES ("
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
        ")",
        payload,
    )
    conn.execute(
        f"CREATE VIEW {quote_ident(engine.view_name)} AS SELECT * FROM toy_best"
    )


def _toy_context(base):
    return dataclasses.replace(
        base,
        var_names=("x",),
        param_names=("x",),
        constraint_names=(),
        collect_summary=None,
        problem_info={
            "model": {
                "x": {"role": "var", "values": [1.0], "lb": 0, "ub": 0},
            }
        },
    )


class TestBruteForceBestDesignsAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        cls.golden = _run_tiny_bf(cls.root / "golden", experiment_id="bf_best_gold")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_real_experiment_matches_b8_summary(self) -> None:
        context = open_brute_force_dataset(self.golden)
        with open_query_engine(context) as engine:
            report = run_best_designs_analysis(context, engine)

        self.assertEqual(report["best_finite"]["status"], "found")
        self.assertEqual(report["b8_cross_check"]["status"], "ok")
        summary = context.collect_summary
        assert summary is not None
        self.assertEqual(
            report["best_finite"]["global_index"],
            summary["best_global_index"],
        )
        self.assertTrue(
            math.isclose(
                float(report["best_finite"]["y_p"]),
                float(summary["best_finite_y_p"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )
        self.assertEqual(
            report["finite_objective_count"],
            summary["finite_objective_count"],
        )
        self.assertEqual(
            set(report["best_finite"]["indices"]),
            set(context.var_names),
        )
        self.assertEqual(
            set(report["best_finite"]["params"]),
            set(context.param_names),
        )
        self.assertEqual(
            set(report["best_finite"]["constraints"]),
            set(context.constraint_names),
        )
        near = report["near_optimum_bands"]
        self.assertEqual(near["status"], "computed")
        self.assertEqual(
            [band["tolerance_percent"] for band in near["bands"]],
            list(NEAR_OPTIMUM_TOLERANCES_PERCENT),
        )
        self.assertTrue(near["monotone_ok"])
        self.assertTrue(near["winner_in_all_bands"])
        self.assertTrue(report["ok"])
        if report["best_feasible"]["status"] == "found":
            self.assertEqual(report["worst_feasible"]["status"], "found")
            self.assertGreaterEqual(
                float(report["worst_feasible"]["y_p"]),
                float(report["best_feasible"]["y_p"]),
            )
        else:
            self.assertEqual(report["worst_feasible"]["status"], "unavailable")

    def test_equal_objective_prefers_lowest_global_index(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        with open_query_engine(context) as engine:
            _install_view(
                engine,
                [
                    (5, 10.0, 3.0, True, True, True),
                    (2, 10.0, 3.0, True, True, True),
                    (9, 10.0, 4.0, True, True, True),
                ],
            )
            report = run_best_designs_analysis(toy, engine)
        self.assertEqual(report["best_finite"]["global_index"], 2)
        self.assertEqual(report["best_feasible"]["global_index"], 2)
        self.assertEqual(report["worst_feasible"]["global_index"], 9)

    def test_worst_feasible_distinct_and_tied_max(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        with open_query_engine(context) as engine:
            _install_view(
                engine,
                [
                    (10, 100.0, 100.0, True, True, True),
                    (20, 150.0, 150.0, True, True, True),
                    (30, 150.0, 150.0, True, True, True),
                    (40, 200.0, 200.0, True, False, False),
                ],
            )
            report = run_best_designs_analysis(toy, engine)

        self.assertEqual(report["best_feasible"]["global_index"], 10)
        self.assertEqual(report["best_feasible"]["y_p"], 100.0)
        # Tied max feasible y_p at indices 20 and 30 -> lowest index wins.
        self.assertEqual(report["worst_feasible"]["status"], "found")
        self.assertEqual(report["worst_feasible"]["global_index"], 20)
        self.assertEqual(report["worst_feasible"]["y_p"], 150.0)
        self.assertTrue(report["worst_feasible"]["penalty_factor_is_one"])
        self.assertTrue(report["worst_feasible"]["y_equals_y_p"])

    def test_single_feasible_best_equals_worst(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        with open_query_engine(context) as engine:
            _install_view(
                engine,
                [
                    (7, 12.0, 12.0, True, True, True),
                    (8, 5.0, 5.0, True, False, False),
                    (9, 99.0, 99.0, True, False, False),
                ],
            )
            report = run_best_designs_analysis(toy, engine)

        self.assertEqual(report["best_feasible"]["global_index"], 7)
        self.assertEqual(report["worst_feasible"]["global_index"], 7)
        self.assertEqual(report["worst_feasible"]["y_p"], 12.0)

    def test_all_nonfinite_and_all_infeasible_reasons(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        with open_query_engine(context) as engine:
            _install_view(
                engine,
                [
                    (0, float("inf"), float("inf"), False, False, False),
                    (1, float("nan"), float("nan"), False, False, False),
                ],
            )
            nonfinite = run_best_designs_analysis(toy, engine)
            _install_view(
                engine,
                [
                    (0, 1.0, 1.0, True, False, False),
                    (1, 2.0, 2.0, True, False, False),
                ],
            )
            infeasible = run_best_designs_analysis(toy, engine)

        self.assertEqual(nonfinite["best_finite"]["status"], "unavailable")
        self.assertEqual(nonfinite["best_finite"]["reason"], "no_finite_y_p")
        self.assertEqual(
            nonfinite["near_optimum_bands"]["status"], "unavailable"
        )
        self.assertEqual(
            nonfinite["near_optimum_bands"]["reason"],
            "optimum_missing_or_nonfinite",
        )
        self.assertEqual(nonfinite["worst_feasible"]["status"], "unavailable")
        self.assertEqual(
            nonfinite["worst_feasible"]["reason"], "no_feasible_finite_y_p"
        )

        self.assertEqual(infeasible["best_finite"]["status"], "found")
        self.assertEqual(infeasible["best_feasible"]["status"], "unavailable")
        self.assertEqual(
            infeasible["best_feasible"]["reason"], "no_feasible_finite_y_p"
        )
        self.assertEqual(infeasible["worst_feasible"]["status"], "unavailable")
        self.assertEqual(
            infeasible["worst_feasible"]["reason"], "no_feasible_finite_y_p"
        )

    def test_near_optimum_band_counts_and_feasible_filter(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        # y_p* = 100
        # within 1% (<=101): indices 0,1
        # within 2% (<=102): +2
        # within 3% (<=103): same
        # within 4% (<=104): +3
        # within 5% (<=105): same
        # within 10% (<=110): +4
        # index 5 is infeasible but within 1% threshold -> counted in all, not feasible
        rows = [
            (0, 100.0, 100.0, True, True, True),
            (1, 100.5, 100.5, True, True, True),
            (2, 101.5, 101.5, True, True, True),
            (3, 104.0, 104.0, True, True, True),
            (4, 109.0, 109.0, True, True, True),
            (5, 100.2, 100.2, True, False, False),
            (6, 200.0, 200.0, True, True, True),
        ]
        with open_query_engine(context) as engine:
            _install_view(engine, rows)
            report = run_best_designs_analysis(toy, engine)

        near = report["near_optimum_bands"]
        self.assertEqual(near["status"], "computed")
        self.assertEqual(near["y_p_star"], 100.0)
        # y* = 100: within thresholds include infeasible index 5 at 100.2
        expected = {
            1.0: (3, 2),
            2.0: (4, 3),
            3.0: (4, 3),
            4.0: (5, 4),
            5.0: (5, 4),
            **{float(t): (6, 5) for t in range(10, 100, 5)},
            100.0: (7, 6),
        }
        self.assertEqual(
            [band["tolerance_percent"] for band in near["bands"]],
            list(NEAR_OPTIMUM_TOLERANCES_PERCENT),
        )
        for band in near["bands"]:
            tol = float(band["tolerance_percent"])
            within, feasible = expected[tol]
            self.assertEqual(band["count_within"], within, msg=f"tol={tol}")
            self.assertEqual(
                band["count_feasible_within"], feasible, msg=f"tol={tol}"
            )
        self.assertTrue(near["monotone_ok"])
        self.assertTrue(near["winner_in_all_bands"])

    def test_negative_optimum_marks_bands_unavailable(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        with open_query_engine(context) as engine:
            _install_view(
                engine,
                [
                    (0, -2.0, -2.0, True, True, True),
                    (1, -1.0, -1.0, True, True, True),
                ],
            )
            report = run_best_designs_analysis(toy, engine)
        self.assertEqual(report["best_finite"]["y_p"], -2.0)
        self.assertEqual(report["near_optimum_bands"]["status"], "unavailable")
        self.assertEqual(
            report["near_optimum_bands"]["reason"], "negative_optimum"
        )
        self.assertEqual(report["near_optimum_bands"]["bands"], [])


if __name__ == "__main__":
    unittest.main()
