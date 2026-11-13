"""
Phase C3.2 gate: DuckDB query engine and column-family discovery.
"""
from __future__ import annotations

import hashlib
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.lifecycle import (
    collect_experiment,
    dispatch_experiment,
    prepare_experiment,
)
from core.ioh_core.workflows.brute_force.analysis.columns import (
    BruteForceColumnError,
    discover_column_families,
)
from core.ioh_core.workflows.brute_force.analysis.engine import (
    BruteForceEngineError,
    open_query_engine,
    quote_ident,
    quote_path,
    quote_string,
)
from core.ioh_core.workflows.brute_force.analysis.models import DatasetReportOptions
from core.ioh_core.workflows.brute_force.analysis import open_brute_force_dataset


def _write_bf_config(
    path: Path,
    *,
    experiment_id: str,
    rows_per_shard: int = 24,
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
            rows_per_shard = {int(rows_per_shard)}
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
    experiment_id: str = "bf_analysis_engine",
    max_rows: int | None = 48,
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = _write_bf_config(
        root / f"{experiment_id}.toml",
        experiment_id=experiment_id,
        max_rows=max_rows,
    )
    experiment_dir = root / "experiments" / experiment_id
    prepare_experiment(config_path=config_path, output=experiment_dir)
    dispatch_experiment(experiment_dir)
    collect_experiment(experiment_dir, require_complete=True)
    return experiment_dir


def _part_fingerprints(dataset_dir: Path) -> dict[str, tuple[str, int]]:
    parts_dir = dataset_dir / "parts"
    out: dict[str, tuple[str, int]] = {}
    for path in sorted(parts_dir.glob("shard-*.parquet")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out[path.name] = (digest, path.stat().st_mtime_ns)
    return out


class TestSqlQuoting(unittest.TestCase):
    def test_quote_ident_escapes_double_quotes(self) -> None:
        self.assertEqual(quote_ident('util__a"b'), '"util__a""b"')

    def test_quote_string_and_path_handle_special_characters(self) -> None:
        weird = Path(r"C:\OD\OneDrive - Technische Universität Berlin\O'Reilly")
        quoted = quote_path(weird)
        self.assertTrue(quoted.startswith("'"))
        self.assertTrue(quoted.endswith("'"))
        self.assertIn("Technische Universität Berlin", quoted)
        self.assertIn("O''Reilly", quoted)
        self.assertEqual(quote_string("a'b"), "'a''b'")

    def test_quote_ident_rejects_empty(self) -> None:
        with self.assertRaises(BruteForceEngineError):
            quote_ident("")


class TestColumnFamilies(unittest.TestCase):
    def test_complete_triplets_ok(self) -> None:
        schema_doc = {
            "constraint_names": ["c1", "c2"],
            "fields": [
                {"name": "global_index"},
                {"name": "idx__x"},
                {"name": "param__x"},
                {"name": "util__c1"},
                {"name": "violation__c1"},
                {"name": "penalty__c1"},
                {"name": "util__c2"},
                {"name": "violation__c2"},
                {"name": "penalty__c2"},
            ],
        }
        families = discover_column_families(
            schema_doc, expected_constraint_names=["c1", "c2"]
        )
        self.assertEqual(families.constraint_names, ("c1", "c2"))
        self.assertEqual(families.idx_columns, ("idx__x",))
        self.assertEqual(families.param_columns, ("param__x",))

    def test_incomplete_triplet_rejected(self) -> None:
        schema_doc = {
            "fields": [
                {"name": "util__c1"},
                {"name": "violation__c1"},
            ]
        }
        with self.assertRaises(BruteForceColumnError):
            discover_column_families(schema_doc)

    def test_unknown_family_prefix_rejected(self) -> None:
        schema_doc = {
            "fields": [
                {"name": "utils__c1"},
                {"name": "util__c1"},
                {"name": "violation__c1"},
                {"name": "penalty__c1"},
            ]
        }
        with self.assertRaises(BruteForceColumnError):
            discover_column_families(schema_doc)


class TestQueryEngine(unittest.TestCase):
    def test_aggregate_count_and_read_only_parts(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = _run_tiny_bf(Path(tmp), max_rows=48)
            context = open_brute_force_dataset(experiment_dir)
            before = _part_fingerprints(context.dataset_dir)

            with open_query_engine(context, options=DatasetReportOptions()) as engine:
                self.assertEqual(engine.count_rows(), 48)
                row = engine.fetch_one(
                    'SELECT MIN("global_index"), MAX("global_index"), COUNT(*) '
                    f'FROM "{engine.view_name}"'
                )
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(int(row[0]), 0)
                self.assertEqual(int(row[1]), 47)
                self.assertEqual(int(row[2]), 48)

                # Bounded example path.
                examples = engine.fetch_dicts(
                    f'SELECT "global_index" FROM "{engine.view_name}" '
                    'ORDER BY "global_index" LIMIT 3'
                )
                self.assertEqual(
                    [int(item["global_index"]) for item in examples],
                    [0, 1, 2],
                )

                self.assertEqual(
                    engine.tolerances_metadata(),
                    {"scientific_rtol": 1e-12, "sentinel_atol": 1e-9},
                )

                with self.assertRaises(BruteForceEngineError):
                    engine.fetch_rows(
                        f'SELECT * FROM "{engine.view_name}"',
                        max_rows=10,
                    )

            after = _part_fingerprints(context.dataset_dir)
            self.assertEqual(before, after)

    def test_engine_does_not_expose_fetchall_helper(self) -> None:
        from core.ioh_core.workflows.brute_force.analysis import engine as eng

        self.assertFalse(hasattr(eng.BruteForceQueryEngine, "fetchall"))
        self.assertFalse(hasattr(eng.BruteForceQueryEngine, "to_pylist"))


if __name__ == "__main__":
    unittest.main()
