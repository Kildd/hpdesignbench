"""
Phase C3.2 gate: dataset report serialization and empty report shell.
"""
from __future__ import annotations

import json
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
    empty_dataset_report,
    open_brute_force_dataset,
    write_dataset_reports,
)
from core.ioh_core.workflows.brute_force.analysis.report import (
    REPORT_SCHEMA_VERSION,
    REPORT_SECTION_KEYS,
    NEAR_OPTIMUM_TOLERANCES_PERCENT,
    render_dataset_report_markdown,
)
from core.ioh_core.workflows.brute_force.analysis.serialization import (
    NONFINITE_TAG,
    dumps_report_json,
    normalize_report_value,
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
            max_rows = 24
            start_index = 0
            parquet_batch_rows = 24
            compression = "zstd"
            reuse_structural_artifacts = true
            acknowledge_large_run = true
            maximum_expected_rows = 1000000
            maximum_expected_bytes = 10000000000

            [execution]
            backend = "local"
            max_parallel_tasks = 1
            timeout_seconds = 3600
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        ),
        encoding="utf-8",
    )
    return path


class TestSerialization(unittest.TestCase):
    def test_nonfinite_tags_are_strict_json(self) -> None:
        payload = {
            "nan": float("nan"),
            "pos": float("inf"),
            "neg": float("-inf"),
            "ok": 1.5,
        }
        text = dumps_report_json(payload)
        parsed = json.loads(text)
        self.assertEqual(parsed["nan"], {NONFINITE_TAG: "NaN"})
        self.assertEqual(parsed["pos"], {NONFINITE_TAG: "+Infinity"})
        self.assertEqual(parsed["neg"], {NONFINITE_TAG: "-Infinity"})
        self.assertEqual(parsed["ok"], 1.5)
        # Round-trip must remain allow_nan=False compatible.
        json.dumps(parsed, allow_nan=False)

    def test_serialization_is_byte_identical(self) -> None:
        report = {
            "b": {"z": 1, "a": float("nan")},
            "a": [3, 1, 2],
            "schema_version": 1,
        }
        first = dumps_report_json(report)
        second = dumps_report_json(report)
        self.assertEqual(first, second)
        self.assertTrue(first.endswith("\n"))
        # Mapping keys sorted; list order preserved.
        self.assertLess(first.index('"a"'), first.index('"b"'))
        nested = normalize_report_value(report)["b"]
        self.assertEqual(list(nested.keys()), ["a", "z"])

    def test_numpy_scalar_unwrapped_when_available(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError:  # pragma: no cover
            self.skipTest("numpy not installed")
        value = normalize_report_value({"x": np.float64(2.5), "y": np.int64(7)})
        self.assertEqual(value, {"x": 2.5, "y": 7})
        self.assertIsInstance(value["x"], float)
        self.assertIsInstance(value["y"], int)


class TestEmptyReportShell(unittest.TestCase):
    def test_stable_sections_and_near_optimum_reserved(self) -> None:
        report = empty_dataset_report()
        self.assertEqual(report["schema_version"], REPORT_SCHEMA_VERSION)
        for key in REPORT_SECTION_KEYS:
            self.assertIn(key, report)
        bands = report["best_designs"]["near_optimum_bands"]
        self.assertEqual(bands["status"], "not_computed")
        self.assertEqual(
            bands["tolerances_percent"], list(NEAR_OPTIMUM_TOLERANCES_PERCENT)
        )
        self.assertEqual(bands["bands"], [])
        tool = report["tool_metadata"]
        self.assertEqual(tool["scientific_rtol"], 1e-12)
        self.assertEqual(tool["sentinel_atol"], 1e-9)

    def test_write_reports_atomic_and_markdown_from_json(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_bf_config(root / "bf.toml", experiment_id="bf_rep")
            experiment_dir = root / "experiments" / "bf_rep"
            prepare_experiment(config_path=config_path, output=experiment_dir)
            dispatch_experiment(experiment_dir)
            collect_experiment(experiment_dir, require_complete=True)

            context = open_brute_force_dataset(experiment_dir)
            report = empty_dataset_report(context=context)
            written = write_dataset_reports(experiment_dir, report)
            json_path = written["json"]
            md_path = written["md"]

            self.assertTrue(json_path.is_file())
            self.assertTrue(md_path.is_file())
            text1 = json_path.read_text(encoding="utf-8")
            text2 = dumps_report_json(report)
            self.assertEqual(text1, text2)
            parsed = json.loads(text1)
            self.assertEqual(parsed["dataset"]["experiment_id"], "bf_rep")
            self.assertEqual(parsed["dataset"]["planned_rows"], 24)
            self.assertNotIn(":", str(parsed["dataset"].get("experiment_id")))

            md = md_path.read_text(encoding="utf-8")
            self.assertEqual(md, render_dataset_report_markdown(parsed))
            self.assertIn("## Status", md)
            self.assertIn("## Best designs", md)

            # Second write remains byte-identical for the same report object.
            write_dataset_reports(experiment_dir, report)
            self.assertEqual(json_path.read_text(encoding="utf-8"), text1)
            self.assertEqual(md_path.read_text(encoding="utf-8"), md)

    def test_failed_write_leaves_previous_report(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "aggregate" / "analysis"
            out.mkdir(parents=True)
            good = {"schema_version": 1, "dataset": {"experiment_id": "kept"}}
            # Minimal fake experiment_dir layout for write helper.
            experiment_dir = Path(tmp)
            write_dataset_reports(experiment_dir, good)
            json_path = out / "report.json"
            before = json_path.read_text(encoding="utf-8")

            from core.ioh_core.workflows.brute_force.analysis import report as report_mod

            original = report_mod.render_dataset_report_markdown

            def _boom(_report):  # noqa: ANN001
                raise RuntimeError("render boom")

            report_mod.render_dataset_report_markdown = _boom  # type: ignore[assignment]
            try:
                with self.assertRaises(RuntimeError):
                    write_dataset_reports(experiment_dir, good)
            finally:
                report_mod.render_dataset_report_markdown = original  # type: ignore[assignment]

            self.assertEqual(json_path.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
