"""
Phase C3.8 gate: dataset report assembly and atomic report publication.
"""
from __future__ import annotations

import hashlib
import shutil
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
    run_and_write_dataset_report,
    run_dataset_report,
    write_dataset_reports,
)
from core.ioh_core.workflows.brute_force.analysis.report import (
    REPORT_JSON_NAME,
    REPORT_MARKDOWN_NAME,
    analysis_output_dir,
    render_dataset_report_markdown,
)
from core.ioh_core.workflows.brute_force.analysis.serialization import (
    dumps_report_json,
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
    config_path = _write_bf_config(
        root / f"{experiment_id}.toml", experiment_id=experiment_id
    )
    experiment_dir = root / "experiments" / experiment_id
    prepare_experiment(config_path=config_path, output=experiment_dir)
    dispatch_experiment(experiment_dir)
    collect_experiment(experiment_dir, require_complete=True)
    return experiment_dir


def _part_fingerprints(experiment_dir: Path) -> dict[str, tuple[str, int]]:
    parts = experiment_dir / "dataset" / "parts"
    out: dict[str, tuple[str, int]] = {}
    for path in sorted(parts.glob("shard-*.parquet")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out[path.name] = (digest, path.stat().st_mtime_ns)
    return out


class TestDatasetReportAssembly(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        cls.golden = _run_tiny_bf(cls.root / "golden", experiment_id="bf_l1_gold")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_full_dataset_report_idempotent_and_read_only(self) -> None:
        before = _part_fingerprints(self.golden)
        report1 = run_dataset_report(self.golden)
        self.assertIn(report1["overall_status"], {"success", "scientific_integrity_failure"})
        self.assertIn("integrity", report1)
        self.assertIn("enumeration_coverage", report1)
        self.assertIn("best_designs", report1)
        self.assertIn("feasibility", report1)
        self.assertIn("constraints", report1)
        self.assertNotEqual(
            report1["enumeration_coverage"].get("status"), "skipped"
        )

        written1 = write_dataset_reports(self.golden, report1)
        json1 = written1["json"]
        md1 = written1["md"]
        text_json_1 = json1.read_text(encoding="utf-8")
        text_md_1 = md1.read_text(encoding="utf-8")
        self.assertEqual(text_json_1, dumps_report_json(report1))
        self.assertEqual(text_md_1, render_dataset_report_markdown(report1))
        self.assertIn("## Best designs", text_md_1)
        self.assertIn("near ", text_md_1)

        report2 = run_dataset_report(self.golden)
        self.assertEqual(dumps_report_json(report1), dumps_report_json(report2))
        write_dataset_reports(self.golden, report2)
        self.assertEqual(json1.read_text(encoding="utf-8"), text_json_1)
        self.assertEqual(md1.read_text(encoding="utf-8"), text_md_1)

        after = _part_fingerprints(self.golden)
        self.assertEqual(before, after)
        summary = self.golden / "aggregate" / "summary.json"
        self.assertTrue(summary.is_file())

    def test_structural_failure_writes_integrity_only(self) -> None:
        experiment_dir = self.root / "mut" / "struct"
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        shutil.copytree(self.golden, experiment_dir)
        part = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
        part.write_bytes(b"not parquet")
        (experiment_dir / "dataset" / "_SUCCESS").write_text("ok\n", encoding="utf-8")

        before = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in [
                ("corrupt", part),
                (
                    "ok_part",
                    experiment_dir / "dataset" / "parts" / "shard-000001.parquet",
                ),
            ]
            if path.exists()
        }

        report = run_and_write_dataset_report(experiment_dir, write=True)
        self.assertEqual(report["overall_status"], "structural_failure")
        self.assertFalse(report["ok"])
        self.assertEqual(
            report["enumeration_coverage"].get("status"), "skipped"
        )
        self.assertEqual(report["feasibility"].get("status"), "skipped")
        self.assertEqual(report["constraints"].get("status"), "skipped")

        out = analysis_output_dir(experiment_dir)
        self.assertTrue((out / REPORT_JSON_NAME).is_file())
        self.assertTrue((out / REPORT_MARKDOWN_NAME).is_file())
        md = (out / REPORT_MARKDOWN_NAME).read_text(encoding="utf-8")
        self.assertIn("structural_failure", md)
        self.assertIn("skipped", md)

        after = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in [
                ("corrupt", part),
                (
                    "ok_part",
                    experiment_dir / "dataset" / "parts" / "shard-000001.parquet",
                ),
            ]
            if path.exists()
        }
        self.assertEqual(before, after)

    def test_reject_output_inside_dataset(self) -> None:
        report = run_dataset_report(self.golden)
        with self.assertRaises(ValueError):
            write_dataset_reports(
                self.golden,
                report,
                output_dir=self.golden / "dataset" / "analysis_out",
            )


if __name__ == "__main__":
    unittest.main()
