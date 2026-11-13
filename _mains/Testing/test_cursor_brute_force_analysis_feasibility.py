"""
Phase C3.6 gate: feasibility totals and variable-level summaries.
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
    run_feasibility_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.engine import quote_ident
from core.ioh_core.workflows.brute_force.models import IndexSpace


def _write_bf_config(
    path: Path,
    *,
    experiment_id: str,
    max_rows: int | None = 48,
    start_index: int = 0,
) -> Path:
    max_rows_line = (
        f"max_rows = {int(max_rows)}" if max_rows is not None else ""
    )
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
            {max_rows_line}
            start_index = {int(start_index)}
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


def _run_tiny_bf(
    root: Path,
    *,
    experiment_id: str,
    max_rows: int | None = 48,
    start_index: int = 0,
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = _write_bf_config(
        root / f"{experiment_id}.toml",
        experiment_id=experiment_id,
        max_rows=max_rows,
        start_index=start_index,
    )
    experiment_dir = root / "experiments" / experiment_id
    prepare_experiment(config_path=config_path, output=experiment_dir)
    dispatch_experiment(experiment_dir)
    collect_experiment(experiment_dir, require_complete=True)
    return experiment_dir


def _install_view(engine, rows: list[tuple]) -> None:
    """
    Row tuple: (global_index, idx_a, idx_b, feasible, analysis_ok, y, y_p)
    """
    conn = engine.connection
    conn.execute(f"DROP VIEW IF EXISTS {quote_ident(engine.view_name)}")
    conn.execute("DROP TABLE IF EXISTS toy_feas")
    conn.execute(
        "CREATE TEMP TABLE toy_feas ("
        "global_index BIGINT, "
        "analysis_ok BOOLEAN, "
        "constraint_feasible BOOLEAN, "
        "feasible BOOLEAN, "
        "y DOUBLE, "
        "y_p DOUBLE, "
        '"idx__a" INTEGER, '
        '"idx__b" INTEGER'
        ")"
    )
    payload = []
    for global_index, idx_a, idx_b, feasible, analysis_ok, y, y_p in rows:
        payload.append(
            (
                int(global_index),
                bool(analysis_ok),
                bool(feasible),
                bool(feasible),
                float(y),
                float(y_p),
                int(idx_a),
                int(idx_b),
            )
        )
    conn.executemany(
        "INSERT INTO toy_feas VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        payload,
    )
    conn.execute(
        f"CREATE VIEW {quote_ident(engine.view_name)} AS SELECT * FROM toy_feas"
    )


def _toy_context(base, *, full_catalogue_rows: int = 6):
    space = IndexSpace.from_bounds(("a", "b"), (0, 0), (1, 2))
    return dataclasses.replace(
        base,
        index_space=space,
        var_names=("a", "b"),
        param_names=("a", "b"),
        constraint_names=(),
        collect_summary=None,
        full_catalogue_rows=full_catalogue_rows,
        problem_info={
            "model": {
                "a": {"role": "var", "values": [10, 20], "lb": 0, "ub": 1},
                "b": {"role": "var", "values": ["x", "y", "z"], "lb": 0, "ub": 2},
            }
        },
    )


class TestBruteForceFeasibilityAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        cls.golden = _run_tiny_bf(cls.root / "golden", experiment_id="bf_feas_gold")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_real_dataset_denominators_and_b8(self) -> None:
        context = open_brute_force_dataset(self.golden)
        with open_query_engine(context) as engine:
            report = run_feasibility_analysis(context, engine)

        totals = report["totals"]
        self.assertEqual(totals["observed_rows"], 48)
        self.assertEqual(totals["full_catalogue_rows"], 53_374_464)
        summary = context.collect_summary
        assert summary is not None
        self.assertEqual(
            totals["dataset_shares"]["analysis_ok"]["count"],
            summary["analysis_ok_count"],
        )
        self.assertEqual(
            totals["dataset_shares"]["feasible"]["count"],
            summary["feasible_count"],
        )
        self.assertEqual(report["b8_cross_check"]["status"], "ok")
        self.assertTrue(report["ok"])
        self.assertEqual(list(v["name"] for v in report["variables"]), list(context.var_names))
        for metric in totals["dataset_shares"].values():
            self.assertTrue(0.0 <= metric["share_of_observed"] <= 1.0)
        for metric in totals["catalogue_coverage"].values():
            self.assertTrue(0.0 <= metric["share_of_catalogue"] <= 1.0)
            # Truncated runs: catalogue coverage is strictly smaller than a
            # full-dataset share unless the count is zero.
            self.assertLessEqual(
                metric["share_of_catalogue"],
                metric["count"] / max(totals["observed_rows"], 1),
            )

    def test_zero_feasible_synthetic(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        with open_query_engine(context) as engine:
            _install_view(
                engine,
                [
                    (0, 0, 0, False, True, 1.0, 1.0),
                    (1, 1, 1, False, True, 2.0, 2.0),
                    (2, 0, 2, False, False, 3.0, 3.0),
                ],
            )
            report = run_feasibility_analysis(toy, engine)

        self.assertEqual(report["overall_status"], "no_feasible_rows")
        self.assertEqual(report["totals"]["dataset_shares"]["feasible"]["count"], 0)
        for variable in report["variables"]:
            for level in variable["levels"]:
                if level["observed_count"] == 0:
                    self.assertEqual(level["status"], "unobserved_level")
                    self.assertIsNone(level["feasible_level_enrichment"])
                    self.assertIsNone(level["conditional_feasibility_rate"])
                else:
                    self.assertEqual(level["status"], "no_feasible_rows")
                    self.assertIsNone(level["share_within_all_feasible_designs"])
                    self.assertIsNone(level["feasible_level_enrichment"])
                    self.assertEqual(level["conditional_feasibility_rate"], 0.0)

    def test_unobserved_level_enrichment_null(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        # Full catalogue has b in {0,1,2}; omit b=1 entirely.
        # 2 feasible rows both at b=0; one infeasible at b=2.
        with open_query_engine(context) as engine:
            _install_view(
                engine,
                [
                    (0, 0, 0, True, True, 1.0, 1.0),
                    (1, 1, 0, True, True, 1.0, 1.0),
                    (2, 0, 2, False, True, 1.0, 1.0),
                ],
            )
            report = run_feasibility_analysis(toy, engine)

        b = next(v for v in report["variables"] if v["name"] == "b")
        self.assertEqual([level["index"] for level in b["levels"]], [0, 1, 2])
        unobserved = b["levels"][1]
        self.assertEqual(unobserved["status"], "unobserved_level")
        self.assertEqual(unobserved["observed_count"], 0)
        self.assertIsNone(unobserved["conditional_feasibility_rate"])
        self.assertIsNone(unobserved["feasible_level_enrichment"])

        level0 = b["levels"][0]
        self.assertEqual(level0["status"], "ok")
        self.assertEqual(level0["feasible_count_at_level"], 2)
        self.assertEqual(level0["share_within_all_feasible_designs"], 1.0)
        # observed_share = 2/3, enrichment = 1.0 / (2/3) = 1.5
        self.assertTrue(
            math.isclose(level0["feasible_level_enrichment"], 1.5, rel_tol=0.0, abs_tol=1e-12)
        )

    def test_truncated_window_preserves_catalogue_order(self) -> None:
        experiment_dir = _run_tiny_bf(
            self.root / "trunc",
            experiment_id="bf_feas_trunc",
            max_rows=48,
            start_index=24,
        )
        context = open_brute_force_dataset(experiment_dir)
        with open_query_engine(context) as engine:
            report = run_feasibility_analysis(context, engine)

        self.assertFalse(report["is_full_enumeration"])
        self.assertEqual(report["totals"]["observed_rows"], 24)
        for variable, name in zip(report["variables"], context.var_names):
            self.assertEqual(variable["name"], name)
            space = context.index_space
            pos = space.var_names.index(name)
            expected = list(range(space.lower_bounds[pos], space.upper_bounds[pos] + 1))
            self.assertEqual([level["index"] for level in variable["levels"]], expected)
            if any(level["status"] == "unobserved_level" for level in variable["levels"]):
                break
        else:
            # Truncated windows often leave some levels unobserved; if not, still ok.
            pass

    def test_b8_disagreement(self) -> None:
        context = open_brute_force_dataset(self.golden)
        bad_summary = dict(context.collect_summary or {})
        bad_summary["feasible_count"] = -1
        mutated = dataclasses.replace(context, collect_summary=bad_summary)
        with open_query_engine(context) as engine:
            report = run_feasibility_analysis(mutated, engine)
        self.assertEqual(report["b8_cross_check"]["status"], "disagree")
        self.assertIn("feasible_count", report["b8_cross_check"]["disagreements"])
        self.assertFalse(report["ok"])


if __name__ == "__main__":
    unittest.main()
