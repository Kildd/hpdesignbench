"""
Phase C3.9 gate: hdb-bf-analyze CLI exit codes, formats, and read-only publish.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.lifecycle import (
    collect_experiment,
    dispatch_experiment,
    prepare_experiment,
)
from core.ioh_core.workflows.brute_force.analysis.cli import main as cli_main
from core.ioh_core.workflows.brute_force.analysis.report import (
    INTEGRITY_JSON_NAME,
    INTEGRITY_MARKDOWN_NAME,
    NEAR_OPTIMUM_TOLERANCES_PERCENT,
    REPORT_JSON_NAME,
    REPORT_MARKDOWN_NAME,
    analysis_output_dir,
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


def _run_cli(argv: list[str]) -> tuple[int, dict, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cli_main(argv)
    out_text = stdout.getvalue()
    err_text = stderr.getvalue()
    payload: dict = {}
    text = out_text.strip() or err_text.strip()
    if text:
        # Prefer the last JSON line.
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                payload = json.loads(line)
                break
    return code, payload, out_text, err_text


class TestBruteForceAnalysisCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        # Path with spaces (Windows-friendly).
        cls.root = Path(cls._tmpdir.name) / "bf analysis root"
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.golden = _run_tiny_bf(cls.root / "golden", experiment_id="bf_cli_gold")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_usage_error_exit_1(self) -> None:
        code, payload, _out, err = _run_cli(["report"])
        self.assertEqual(code, 1)
        self.assertIn("error", err.lower())

    def test_report_formats_and_idempotence(self) -> None:
        before = _part_fingerprints(self.golden)
        out = analysis_output_dir(self.golden)
        if out.exists():
            shutil.rmtree(out)

        code1, payload1, _o1, e1 = _run_cli(
            [
                "report",
                "--experiment-dir",
                str(self.golden),
                "--format",
                "both",
            ]
        )
        self.assertEqual(code1, 0)
        self.assertEqual(payload1["command"], "report")
        self.assertTrue(payload1["ok"])
        self.assertEqual(payload1["overall_status"], "success")
        self.assertIn("starting Layer 1 dataset analysis", e1)
        json_path = Path(payload1["paths"]["json"])
        md_path = Path(payload1["paths"]["md"])
        self.assertEqual(json_path.name, REPORT_JSON_NAME)
        self.assertEqual(md_path.name, REPORT_MARKDOWN_NAME)
        self.assertTrue(json_path.is_file())
        self.assertTrue(md_path.is_file())
        text_json = json_path.read_text(encoding="utf-8")
        text_md = md_path.read_text(encoding="utf-8")

        code2, payload2, _o2, _e2 = _run_cli(
            [
                "report",
                "--experiment-dir",
                str(self.golden),
                "--format",
                "both",
            ]
        )
        self.assertEqual(code2, 0)
        self.assertEqual(payload2["ok"], True)
        self.assertEqual(json_path.read_text(encoding="utf-8"), text_json)
        self.assertEqual(md_path.read_text(encoding="utf-8"), text_md)
        self.assertEqual(before, _part_fingerprints(self.golden))

        # JSON-only must not remove or rewrite markdown unexpectedly beyond write.
        code_j, payload_j, _oj, _ej = _run_cli(
            [
                "report",
                "--experiment-dir",
                str(self.golden),
                "--format",
                "json",
            ]
        )
        self.assertEqual(code_j, 0)
        self.assertIn("json", payload_j["paths"])
        self.assertNotIn("md", payload_j["paths"])
        self.assertEqual(json_path.read_text(encoding="utf-8"), text_json)

        code_m, payload_m, _om, _em = _run_cli(
            [
                "report",
                "--experiment-dir",
                str(self.golden),
                "--format",
                "md",
            ]
        )
        self.assertEqual(code_m, 0)
        self.assertIn("md", payload_m["paths"])
        self.assertNotIn("json", payload_m["paths"])
        self.assertEqual(md_path.read_text(encoding="utf-8"), text_md)
        self.assertEqual(before, _part_fingerprints(self.golden))

    def test_refresh_near_optimum_bands_keeps_constraints(self) -> None:
        experiment_dir = self.root / "mut" / "refresh_bands"
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        shutil.copytree(self.golden, experiment_dir)

        code_r, payload_r, _or, _er = _run_cli(
            ["report", "--experiment-dir", str(experiment_dir)]
        )
        self.assertEqual(code_r, 0)
        report_path = Path(payload_r["paths"]["json"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        constraints_before = json.dumps(
            report.get("constraints"), sort_keys=True
        )
        feasibility_before = json.dumps(
            report.get("feasibility"), sort_keys=True
        )

        # Simulate an older tolerance grid in the published report.
        report["best_designs"]["near_optimum_bands"] = {
            "status": "computed",
            "tolerances_percent": [1.0, 2.0, 5.0],
            "bands": [
                {"tolerance_percent": 1.0, "count_feasible_within": 0},
                {"tolerance_percent": 2.0, "count_feasible_within": 0},
                {"tolerance_percent": 5.0, "count_feasible_within": 0},
            ],
        }
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

        code, payload, _out, err = _run_cli(
            [
                "report",
                "--experiment-dir",
                str(experiment_dir),
                "--refresh-near-optimum-bands",
            ]
        )
        self.assertEqual(code, 0, msg=err)
        self.assertTrue(payload["refresh_near_optimum_bands"])
        self.assertEqual(
            payload["near_optimum_band_count"],
            len(NEAR_OPTIMUM_TOLERANCES_PERCENT),
        )
        self.assertIn("refreshing near-optimum bands", err)

        refreshed = json.loads(report_path.read_text(encoding="utf-8"))
        near = refreshed["best_designs"]["near_optimum_bands"]
        self.assertEqual(near["status"], "computed")
        self.assertEqual(
            [band["tolerance_percent"] for band in near["bands"]],
            list(NEAR_OPTIMUM_TOLERANCES_PERCENT),
        )
        self.assertEqual(
            json.dumps(refreshed.get("constraints"), sort_keys=True),
            constraints_before,
        )
        self.assertEqual(
            json.dumps(refreshed.get("feasibility"), sort_keys=True),
            feasibility_before,
        )

    def test_integrity_does_not_clobber_report(self) -> None:
        experiment_dir = self.root / "mut" / "integrity_keep"
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        shutil.copytree(self.golden, experiment_dir)

        code_r, payload_r, _or, _er = _run_cli(
            ["report", "--experiment-dir", str(experiment_dir)]
        )
        self.assertEqual(code_r, 0)
        report_json = Path(payload_r["paths"]["json"])
        report_before = report_json.read_bytes()
        summary_before = (
            experiment_dir / "aggregate" / "summary.json"
        ).read_bytes()

        code_i, payload_i, _oi, _ei = _run_cli(
            ["integrity", "--experiment-dir", str(experiment_dir)]
        )
        self.assertEqual(code_i, 0)
        self.assertEqual(payload_i["command"], "integrity")
        self.assertTrue(payload_i["ok"])
        self.assertEqual(payload_i["overall_status"], "success")
        integrity_json = Path(payload_i["paths"]["json"])
        integrity_md = Path(payload_i["paths"]["md"])
        self.assertEqual(integrity_json.name, INTEGRITY_JSON_NAME)
        self.assertEqual(integrity_md.name, INTEGRITY_MARKDOWN_NAME)
        self.assertTrue(integrity_json.is_file())
        self.assertTrue(integrity_md.is_file())
        self.assertEqual(report_json.read_bytes(), report_before)
        self.assertEqual(
            (experiment_dir / "aggregate" / "summary.json").read_bytes(),
            summary_before,
        )
        self.assertFalse(
            (analysis_output_dir(experiment_dir) / "layer1_report.json").exists()
        )

    def test_structural_failure_exit_2(self) -> None:
        experiment_dir = self.root / "mut" / "struct_fail"
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        shutil.copytree(self.golden, experiment_dir)
        part = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
        part.write_bytes(b"not parquet")
        (experiment_dir / "dataset" / "_SUCCESS").write_text("ok\n", encoding="utf-8")
        before = hashlib.sha256(part.read_bytes()).hexdigest()

        code, payload, _o, _e = _run_cli(
            ["report", "--experiment-dir", str(experiment_dir)]
        )
        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["overall_status"], "structural_failure")
        out = analysis_output_dir(experiment_dir)
        self.assertTrue((out / REPORT_JSON_NAME).is_file())
        self.assertTrue((out / REPORT_MARKDOWN_NAME).is_file())
        self.assertEqual(hashlib.sha256(part.read_bytes()).hexdigest(), before)

        code_i, payload_i, _oi, _ei = _run_cli(
            ["integrity", "--experiment-dir", str(experiment_dir)]
        )
        self.assertEqual(code_i, 2)
        self.assertFalse(payload_i["ok"])
        self.assertEqual(payload_i["overall_status"], "structural_failure")
        self.assertTrue(
            (out / INTEGRITY_JSON_NAME).is_file()
        )
        # Full report remains from the report command.
        self.assertTrue((out / REPORT_JSON_NAME).is_file())

    def test_reject_output_inside_dataset(self) -> None:
        bad = self.golden / "dataset" / "analysis_out"
        code, payload, _o, err = _run_cli(
            [
                "report",
                "--experiment-dir",
                str(self.golden),
                "--output-dir",
                str(bad),
            ]
        )
        self.assertEqual(code, 1)
        self.assertFalse(payload.get("ok", True))
        self.assertIn("error", payload)
        self.assertTrue(err.strip() or payload.get("error"))

    def test_module_entry_point(self) -> None:
        import runpy
        import sys

        argv_before = list(sys.argv)
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            sys.argv = [
                "analysis",
                "integrity",
                "--experiment-dir",
                str(self.golden),
                "--format",
                "json",
            ]
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    runpy.run_module(
                        "core.ioh_core.workflows.brute_force.analysis",
                        run_name="__main__",
                    )
            self.assertEqual(raised.exception.code, 0)
        finally:
            sys.argv = argv_before
        payload = json.loads(stdout.getvalue().strip().splitlines()[-1])
        self.assertEqual(payload["command"], "integrity")
        self.assertTrue(payload["ok"])


if __name__ == "__main__":
    unittest.main()
