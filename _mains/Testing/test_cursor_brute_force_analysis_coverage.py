"""
Phase C3.4 gate: enumeration coverage and marginal distributions.
"""
from __future__ import annotations

import dataclasses
import shutil
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from core.execution_core.lifecycle import (
    collect_experiment,
    dispatch_experiment,
    prepare_experiment,
)
from core.ioh_core.workflows.brute_force.analysis import (
    DatasetReportOptions,
    open_brute_force_dataset,
    open_query_engine,
    run_coverage_analysis,
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


def _clone_experiment(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _toy_context(base_context):
    space = IndexSpace.from_bounds(("a", "b"), (0, 0), (1, 2))
    problem_info = {
        "model": {
            "a": {
                "role": "var",
                "values": [10, 20],
                "lb": 0,
                "ub": 1,
            },
            "b": {
                "role": "var",
                "values": ["x", "y", "z"],
                "lb": 0,
                "ub": 2,
            },
            "fixed_k": {
                "role": "fixed",
                "values": [42],
                "fixed_idx": 0,
                "lb": 0,
                "ub": 0,
            },
        }
    }
    return dataclasses.replace(
        base_context,
        index_space=space,
        var_names=("a", "b"),
        param_names=("a", "b", "fixed_k"),
        problem_info=problem_info,
        fixed_params={"fixed_k": 42},
        full_catalogue_rows=6,
        planned_start_index=0,
        planned_stop_index=6,
        planned_rows=6,
        is_full_enumeration=True,
        coverage_fraction=1.0,
    )


def _install_toy_view(engine, *, omit_b_level: int | None = None) -> None:
    rows = []
    global_index = 0
    for a in (0, 1):
        for b in (0, 1, 2):
            if omit_b_level is not None and b == omit_b_level:
                continue
            rows.append((global_index, a, b, 42))
            global_index += 1
    conn = engine.connection
    conn.execute(f"DROP VIEW IF EXISTS {quote_ident(engine.view_name)}")
    conn.execute("DROP TABLE IF EXISTS toy_parts")
    conn.execute(
        "CREATE TEMP TABLE toy_parts ("
        "global_index BIGINT, "
        '"idx__a" INTEGER, '
        '"idx__b" INTEGER, '
        '"param__fixed_k" INTEGER'
        ")"
    )
    conn.executemany(
        'INSERT INTO toy_parts VALUES (?, ?, ?, ?)',
        rows,
    )
    conn.execute(
        f"CREATE VIEW {quote_ident(engine.view_name)} AS SELECT * FROM toy_parts"
    )


class TestBruteForceCoverageAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        cls.golden = _run_tiny_bf(cls.root / "golden", experiment_id="bf_cov_gold")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_full_synthetic_cartesian_exact_marginals(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        with open_query_engine(context) as engine:
            _install_toy_view(engine)
            report = run_coverage_analysis(toy, engine)

        self.assertTrue(report["ok"])
        self.assertTrue(report["is_full_enumeration"])
        self.assertEqual(report["observed_rows"], 6)
        self.assertEqual(report["section"], "enumeration_coverage")

        by_name = {item["name"]: item for item in report["variables"]}
        self.assertEqual(list(by_name), ["a", "b"])

        a_levels = by_name["a"]["levels"]
        self.assertEqual([level["index"] for level in a_levels], [0, 1])
        self.assertEqual([level["value"] for level in a_levels], [10, 20])
        self.assertEqual([level["row_count"] for level in a_levels], [3, 3])
        self.assertTrue(by_name["a"]["ok"])
        self.assertEqual(by_name["a"]["failures"], [])

        b_levels = by_name["b"]["levels"]
        self.assertEqual([level["index"] for level in b_levels], [0, 1, 2])
        self.assertEqual([level["value"] for level in b_levels], ["x", "y", "z"])
        self.assertEqual([level["row_count"] for level in b_levels], [2, 2, 2])
        self.assertTrue(by_name["b"]["ok"])

        fixed = {item["name"]: item for item in report["fixed_parameters"]}
        self.assertTrue(fixed["fixed_k"]["ok"])
        self.assertEqual(fixed["fixed_k"]["expected"], 42)
        self.assertIsNone(report["pairwise_level_coverage"])

    def test_missing_level_fails_full_enumeration(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        with open_query_engine(context) as engine:
            _install_toy_view(engine, omit_b_level=1)
            report = run_coverage_analysis(toy, engine)

        self.assertFalse(report["ok"])
        by_name = {item["name"]: item for item in report["variables"]}
        self.assertFalse(by_name["b"]["ok"])
        self.assertIn("missing_declared_level", by_name["b"]["failures"])
        self.assertIn("exact_marginal_mismatch", by_name["b"]["failures"])
        self.assertEqual(by_name["b"]["levels"][1]["row_count"], 0)

    def test_truncated_window_unbalanced_marginals_still_ok(self) -> None:
        experiment_dir = _run_tiny_bf(
            self.root / "trunc",
            experiment_id="bf_cov_trunc",
            max_rows=48,
            start_index=24,
        )
        context = open_brute_force_dataset(experiment_dir)
        with open_query_engine(context) as engine:
            report = run_coverage_analysis(context, engine)

        self.assertFalse(report["is_full_enumeration"])
        self.assertEqual(report["observed_rows"], 24)
        self.assertTrue(report["ok"])
        for variable in report["variables"]:
            self.assertTrue(variable["ok"])
            self.assertFalse(variable["marginals_are_pass_fail"])
            self.assertEqual(
                len(variable["levels"]), variable["declared_level_count"]
            )

    def test_fixed_parameter_mismatch_fails(self) -> None:
        experiment_dir = _clone_experiment(
            self.golden, self.root / "mut" / "fixed"
        )
        context = open_brute_force_dataset(experiment_dir)
        fixed_name = next(
            name
            for name, meta in context.problem_info["model"].items()
            if meta.get("role") == "fixed"
        )
        part = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
        table = pq.read_table(part)
        col = f"param__{fixed_name}"
        arrays = []
        for field in table.schema:
            if field.name != col:
                arrays.append(table.column(field.name))
                continue
            values = table.column(field.name).to_pylist()
            if isinstance(values[0], str):
                values[0] = f"mutated-{values[0]}"
            elif isinstance(values[0], float):
                values[0] = float(values[0]) + 1.0
            else:
                values[0] = int(values[0]) + 1
            arrays.append(pa.array(values, type=field.type))
        pq.write_table(pa.Table.from_arrays(arrays, schema=table.schema), part)

        context = open_brute_force_dataset(experiment_dir)
        with open_query_engine(context) as engine:
            report = run_coverage_analysis(context, engine)

        self.assertFalse(report["ok"])
        fixed = {item["name"]: item for item in report["fixed_parameters"]}
        self.assertFalse(fixed[fixed_name]["ok"])
        self.assertGreaterEqual(fixed[fixed_name]["mismatch_count"], 1)

    def test_catalogue_order_and_pairwise_optional(self) -> None:
        context = open_brute_force_dataset(self.golden)
        toy = _toy_context(context)
        with open_query_engine(context) as engine:
            _install_toy_view(engine)
            report = run_coverage_analysis(
                toy,
                engine,
                options=DatasetReportOptions(include_pairwise_level_coverage=True),
            )

        b_levels = next(v for v in report["variables"] if v["name"] == "b")["levels"]
        self.assertEqual([level["index"] for level in b_levels], [0, 1, 2])
        self.assertIsNotNone(report["pairwise_level_coverage"])
        pair = report["pairwise_level_coverage"][0]
        self.assertEqual(pair["left"], "a")
        self.assertEqual(pair["right"], "b")
        self.assertEqual(pair["combination_count"], 6)
        self.assertEqual(
            sum(item["row_count"] for item in pair["combinations"]),
            6,
        )


if __name__ == "__main__":
    unittest.main()
