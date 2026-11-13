"""
Phase C3.7 gate: constraint activity and penalty diagnostics.
"""
from __future__ import annotations

import dataclasses
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
    DatasetReportOptions,
    open_brute_force_dataset,
    open_query_engine,
    run_constraints_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.engine import quote_ident


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
    config_path = _write_bf_config(
        root / f"{experiment_id}.toml", experiment_id=experiment_id
    )
    experiment_dir = root / "experiments" / experiment_id
    prepare_experiment(config_path=config_path, output=experiment_dir)
    dispatch_experiment(experiment_dir)
    collect_experiment(experiment_dir, require_complete=True)
    return experiment_dir


def _toy_context(base, *, sentinel: float = 99.0, alpha: float = 1.0, beta: float = 1.0):
    return dataclasses.replace(
        base,
        constraint_names=("c1", "c2"),
        var_names=("x",),
        param_names=("x",),
        opt_penalty_alpha=alpha,
        opt_penalty_beta=beta,
        run_config=dataclasses.replace(
            base.run_config, invalid_utilization=sentinel
        ),
        problem_info={"model": {"x": {"role": "var", "values": [1.0], "lb": 0, "ub": 0}}},
    )


def _install_view(engine, rows: list[dict]) -> None:
    conn = engine.connection
    conn.execute(f"DROP VIEW IF EXISTS {quote_ident(engine.view_name)}")
    conn.execute("DROP TABLE IF EXISTS toy_cons")
    conn.execute(
        "CREATE TEMP TABLE toy_cons ("
        "global_index BIGINT, "
        "analysis_ok BOOLEAN, "
        "penalty_sum DOUBLE, "
        '"util__c1" DOUBLE, "violation__c1" DOUBLE, "penalty__c1" DOUBLE, '
        '"util__c2" DOUBLE, "violation__c2" DOUBLE, "penalty__c2" DOUBLE, '
        "remapped_utilizations VARCHAR[]"
        ")"
    )
    payload = []
    for row in rows:
        payload.append(
            (
                int(row["global_index"]),
                bool(row.get("analysis_ok", True)),
                float(row["penalty_sum"]),
                float(row["u1"]),
                float(row["v1"]),
                float(row["p1"]),
                float(row["u2"]),
                float(row["v2"]),
                float(row["p2"]),
                list(row.get("remapped", [])),
            )
        )
    conn.executemany(
        "INSERT INTO toy_cons VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        payload,
    )
    conn.execute(
        f"CREATE VIEW {quote_ident(engine.view_name)} AS SELECT * FROM toy_cons"
    )


class TestBruteForceConstraintsAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        cls.golden = _run_tiny_bf(cls.root / "golden", experiment_id="bf_cons_gold")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_activity_disagreement_and_zero_not_maximum(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        with open_query_engine(context) as engine:
            _install_view(
                engine,
                [
                    # Feasible-ish: all util <= 1, zero penalties, penalty_sum 0
                    # must NOT be classified as maxima.
                    {
                        "global_index": 0,
                        "penalty_sum": 0.0,
                        "u1": 0.8,
                        "v1": 0.0,
                        "p1": 0.0,
                        "u2": 1.0,
                        "v2": 0.0,
                        "p2": 0.0,
                    },
                    # Active c1 with consistent signals
                    {
                        "global_index": 1,
                        "penalty_sum": 1.0,
                        "u1": 2.0,
                        "v1": 1.0,
                        "p1": 1.0,
                        "u2": 0.5,
                        "v2": 0.0,
                        "p2": 0.0,
                    },
                    # Activity disagreement: util>1 but violation/penalty zero
                    {
                        "global_index": 2,
                        "penalty_sum": 0.0,
                        "u1": 1.5,
                        "v1": 0.0,
                        "p1": 0.0,
                        "u2": 0.2,
                        "v2": 0.0,
                        "p2": 0.0,
                    },
                ],
            )
            report = run_constraints_analysis(toy, engine)

        by_name = {c["name"]: c for c in report["activity"]["per_constraint"]}
        self.assertEqual(by_name["c1"]["active_count"], 2)
        self.assertEqual(by_name["c1"]["activity_signal_disagreements"], 1)
        self.assertEqual(report["maxima"]["rows_with_any_positive_penalty"], 1)
        self.assertEqual(
            report["maxima"]["per_constraint"][0]["max_penalty_count"]
            + report["maxima"]["per_constraint"][1]["max_penalty_count"],
            1,
        )
        self.assertFalse(report["ok"])

    def test_unique_and_tied_maxima(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        with open_query_engine(context) as engine:
            _install_view(
                engine,
                [
                    {
                        "global_index": 0,
                        "penalty_sum": 3.0,
                        "u1": 4.0,
                        "v1": 3.0,
                        "p1": 3.0,
                        "u2": 2.0,
                        "v2": 1.0,
                        "p2": 1.0,
                    },
                    {
                        "global_index": 1,
                        "penalty_sum": 4.0,
                        "u1": 3.0,
                        "v1": 2.0,
                        "p1": 2.0,
                        "u2": 3.0,
                        "v2": 2.0,
                        "p2": 2.0,
                    },
                ],
            )
            report = run_constraints_analysis(toy, engine)

        self.assertEqual(report["maxima"]["rows_with_a_unique_maximum"], 1)
        self.assertEqual(report["maxima"]["rows_with_tied_maxima"], 1)
        self.assertEqual(
            report["maxima"]["max_penalty_vs_max_violation_disagreements"], 0
        )
        self.assertTrue(report["ok"])

    def test_sentinel_direct_vs_remapped_and_formula(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context, sentinel=99.0, alpha=1.0, beta=1.0)
        expected_v = 98.0
        expected_p = expected_v**1.0  # alpha=1, beta=1 in this toy
        with open_query_engine(context) as engine:
            _install_view(
                engine,
                [
                    {
                        "global_index": 0,
                        "penalty_sum": expected_p,
                        "u1": 99.0,
                        "v1": expected_v,
                        "p1": expected_p,
                        "u2": 0.5,
                        "v2": 0.0,
                        "p2": 0.0,
                        "remapped": [],
                    },
                    {
                        "global_index": 1,
                        "penalty_sum": expected_p,
                        "u1": 99.0,
                        "v1": expected_v,
                        "p1": expected_p,
                        "u2": 0.5,
                        "v2": 0.0,
                        "p2": 0.0,
                        "remapped": ["c1"],
                    },
                    {
                        "global_index": 2,
                        "penalty_sum": 1.0,
                        "u1": 99.0,
                        "v1": 1.0,
                        "p1": 1.0,
                        "u2": 0.5,
                        "v2": 0.0,
                        "p2": 0.0,
                        "remapped": [],
                    },
                ],
            )
            report = run_constraints_analysis(toy, engine)

        c1 = next(c for c in report["activity"]["per_constraint"] if c["name"] == "c1")
        self.assertEqual(c1["invalid_sentinel_count"], 3)
        self.assertEqual(c1["direct_sentinel_count"], 2)
        self.assertEqual(c1["nonfinite_remapped_to_sentinel_count"], 1)
        self.assertEqual(c1["sentinel_formula_mismatches"], 1)
        self.assertEqual(report["metadata"]["quantile_method"], "quantile_cont")
        self.assertFalse(report["ok"])

    def test_near_boundary_histogram_pairwise_combinations(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        with open_query_engine(context) as engine:
            _install_view(
                engine,
                [
                    {
                        "global_index": 0,
                        "penalty_sum": 0.0,
                        "u1": 0.95,
                        "v1": 0.0,
                        "p1": 0.0,
                        "u2": 1.05,
                        "v2": 0.05,
                        "p2": 0.05**0.5,
                    },
                    {
                        "global_index": 1,
                        "penalty_sum": 2.0,
                        "u1": 2.0,
                        "v1": 1.0,
                        "p1": 1.0,
                        "u2": 3.0,
                        "v2": 2.0,
                        "p2": 2.0**0.5,
                    },
                    {
                        "global_index": 2,
                        "penalty_sum": 0.0,
                        "u1": 0.5,
                        "v1": 0.0,
                        "p1": 0.0,
                        "u2": 0.5,
                        "v2": 0.0,
                        "p2": 0.0,
                    },
                ],
            )
            report = run_constraints_analysis(
                toy,
                engine,
                options=DatasetReportOptions(
                    include_coviolation_matrix=True,
                    max_violation_combinations=10,
                    near_boundary_low=0.95,
                    near_boundary_high=1.05,
                ),
            )

        nb = {item["name"]: item for item in report["near_boundary"]}
        self.assertEqual(nb["c1"]["feasible_near_boundary"]["count"], 1)
        self.assertEqual(nb["c2"]["violated_near_boundary"]["count"], 1)

        hist = report["simultaneous_violation_histogram"][
            "counts_by_active_constraint_count"
        ]
        self.assertEqual(hist["0"], 1)
        self.assertEqual(hist["1"], 1)
        self.assertEqual(hist["2"], 1)

        self.assertIsNotNone(report["pairwise_coviolation"])
        pair = report["pairwise_coviolation"][0]
        self.assertEqual(pair["co_violation_count"], 1)

        combos = report["complete_violation_combinations"]
        self.assertGreaterEqual(len(combos), 1)
        counts = [item["count"] for item in combos]
        self.assertEqual(counts, sorted(counts, reverse=True))
        # Empty combination (no active constraints) sorts after equal counts by name.
        self.assertIn([], [item["constraints"] for item in combos])

    def test_utilization_shares_active_max_and_invalid(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context, sentinel=99.0)
        with open_query_engine(context) as engine:
            _install_view(
                engine,
                [
                    # Feasible: neither active nor max-violation
                    {
                        "global_index": 0,
                        "penalty_sum": 0.0,
                        "u1": 0.8,
                        "v1": 0.0,
                        "p1": 0.0,
                        "u2": 0.9,
                        "v2": 0.0,
                        "p2": 0.0,
                    },
                    # c1 uniquely maximum among active
                    {
                        "global_index": 1,
                        "penalty_sum": 1.5,
                        "u1": 2.5,
                        "v1": 1.5,
                        "p1": 1.5,
                        "u2": 1.2,
                        "v2": 0.2,
                        "p2": 0.2,
                    },
                    # Tied maxima (both active at 2.0)
                    {
                        "global_index": 2,
                        "penalty_sum": 2.0,
                        "u1": 2.0,
                        "v1": 1.0,
                        "p1": 1.0,
                        "u2": 2.0,
                        "v2": 1.0,
                        "p2": 1.0,
                    },
                    # Invalid sentinel on c1 (also active + max)
                    {
                        "global_index": 3,
                        "penalty_sum": 98.0,
                        "u1": 99.0,
                        "v1": 98.0,
                        "p1": 98.0,
                        "u2": 0.5,
                        "v2": 0.0,
                        "p2": 0.0,
                    },
                ],
            )
            report = run_constraints_analysis(toy, engine)

        shares = report["utilization_shares"]
        self.assertEqual(shares["observed_rows"], 4)
        by_name = {c["name"]: c for c in shares["per_constraint"]}
        # c1 active in rows 1,2,3
        self.assertEqual(by_name["c1"]["active_count"], 3)
        self.assertAlmostEqual(
            by_name["c1"]["active_share_of_all_observed_rows"], 0.75
        )
        # max: unique in row1, tied in row2, sentinel in row3 → 3
        self.assertEqual(by_name["c1"]["maximum_violation_count"], 3)
        self.assertEqual(by_name["c1"]["invalid_utilization_count"], 1)
        # c2 active in rows 1,2; max only in tied row2
        self.assertEqual(by_name["c2"]["active_count"], 2)
        self.assertEqual(by_name["c2"]["maximum_violation_count"], 1)
        self.assertEqual(by_name["c2"]["invalid_utilization_count"], 0)
        self.assertAlmostEqual(
            by_name["c2"]["maximum_violation_share_of_all_observed_rows"], 0.25
        )


if __name__ == "__main__":
    unittest.main()
