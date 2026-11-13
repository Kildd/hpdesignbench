"""
C7.4 gate: factorial sensitivity CLI and artifact publication.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import shutil
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from core.execution_core.lifecycle import (
    collect_experiment,
    dispatch_experiment,
    prepare_experiment,
)
from core.ioh_core.workflows.brute_force.analysis import (
    open_brute_force_dataset,
    open_query_engine,
)
from core.ioh_core.workflows.brute_force.analysis.cli import main as cli_main
from core.ioh_core.workflows.brute_force.analysis.engine import quote_ident
from core.ioh_core.workflows.brute_force.analysis.factorial_sensitivity import (
    EXPRESSION_COLUMN,
    FACTORIAL_DIAGNOSTIC_JSON,
    FACTORIAL_INDICES_CSV,
    FACTORIAL_RESULTS_JSON,
    FACTORIAL_SUCCESS_MARKER,
    FACTORIAL_SUMMARY_MD,
    RESPONSE_FAMILY_OBJECTIVE,
    FactorialResponseSpec,
    FactorialSensitivityEligibility,
    FactorialSensitivityMethodMeta,
    FactorSobolIndices,
    FactorialResponseResult,
    FactorialSensitivityComputeResult,
    build_factorial_sensitivity_results,
    compute_factorial_sobol_indices,
    factorial_sensitivity_output_dir,
    write_factorial_sensitivity_artifacts,
)
from core.ioh_core.workflows.brute_force.models import IndexSpace


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
        root / f"{experiment_id}.toml",
        experiment_id=experiment_id,
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
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                payload = json.loads(line)
                break
    return code, payload, out_text, err_text


def _synthetic_results(experiment_dir: Path) -> dict:
    context = open_brute_force_dataset(experiment_dir)
    eligibility = FactorialSensitivityEligibility(
        ok=True,
        reasons=(),
        diagnostics={"ok": True},
        response_specs=(
            FactorialResponseSpec(
                family=RESPONSE_FAMILY_OBJECTIVE,
                name="y_p",
                source_column="y_p",
                expression=EXPRESSION_COLUMN,
            ),
        ),
        variable_names=("a", "b"),
        level_counts=(2, 3),
        cardinality=6,
        invalid_utilization=float(context.run_config.invalid_utilization),
    )
    compute = FactorialSensitivityComputeResult(
        method=FactorialSensitivityMethodMeta(),
        responses=(
            FactorialResponseResult(
                spec=eligibility.response_specs[0],
                status="ok",
                mean=1.5,
                total_variance=0.25,
                factors=(
                    FactorSobolIndices(
                        variable="a",
                        axis=0,
                        first_order=0.25,
                        total_effect=0.5,
                        total_effect_via_complement_mean=0.5,
                        interaction_gap=0.25,
                        identity_residual=0.0,
                    ),
                    FactorSobolIndices(
                        variable="b",
                        axis=1,
                        first_order=0.5,
                        total_effect=0.75,
                        total_effect_via_complement_mean=0.75,
                        interaction_gap=0.25,
                        identity_residual=0.0,
                    ),
                ),
            ),
            FactorialResponseResult(
                spec=FactorialResponseSpec(
                    family="utilization",
                    name="util__c1",
                    source_column="util__c1",
                    expression=EXPRESSION_COLUMN,
                ),
                status="constant",
                mean=0.5,
                total_variance=0.0,
                factors=(),
            ),
        ),
        variable_names=("a", "b"),
        query_count=5,
        largest_complement_group_count=3,
    )
    toy_context = dataclasses.replace(
        context,
        var_names=("a", "b"),
        constraint_names=("c1",),
        is_full_enumeration=True,
        full_catalogue_rows=6,
        planned_rows=6,
        planned_start_index=0,
        planned_stop_index=6,
    )
    return build_factorial_sensitivity_results(
        toy_context,
        eligibility,
        compute,
    )


class TestFactorialSensitivityPublication(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        cls.golden = _run_tiny_bf(
            cls.root / "golden",
            experiment_id="bf_fs_cli_gold",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_publish_success_and_deterministic_rerun(self) -> None:
        results = _synthetic_results(self.golden)
        out = factorial_sensitivity_output_dir(self.golden)
        if out.exists():
            shutil.rmtree(out)

        written1 = write_factorial_sensitivity_artifacts(
            self.golden,
            results,
            formats=("json", "csv", "md"),
            success=True,
        )
        self.assertTrue((out / FACTORIAL_SUCCESS_MARKER).is_file())
        self.assertEqual(written1["json"].name, FACTORIAL_RESULTS_JSON)
        self.assertEqual(written1["csv"].name, FACTORIAL_INDICES_CSV)
        self.assertEqual(written1["md"].name, FACTORIAL_SUMMARY_MD)

        json_bytes = written1["json"].read_bytes()
        csv_bytes = written1["csv"].read_bytes()
        md_bytes = written1["md"].read_bytes()

        payload = json.loads(json_bytes.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["method"]["method_id"], "sobol_hoeffding_functional_anova")

        with written1["csv"].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)  # two factors + one constant row
        names = {row["response_name"] for row in rows}
        self.assertEqual(names, {"y_p", "util__c1"})

        written2 = write_factorial_sensitivity_artifacts(
            self.golden,
            results,
            formats=("json", "csv", "md"),
            success=True,
        )
        self.assertEqual(written2["json"].read_bytes(), json_bytes)
        self.assertEqual(written2["csv"].read_bytes(), csv_bytes)
        self.assertEqual(written2["md"].read_bytes(), md_bytes)

    def test_failure_withholds_success_and_writes_diagnostic(self) -> None:
        out = factorial_sensitivity_output_dir(self.golden)
        out.mkdir(parents=True, exist_ok=True)
        (out / FACTORIAL_SUCCESS_MARKER).write_text("stale\n", encoding="utf-8")

        diagnostic = {
            "schema_version": 1,
            "kind": "factorial_sensitivity_diagnostic",
            "ok": False,
            "overall_status": "eligibility_failure",
            "eligibility": {"ok": False, "reasons": ["not_full_enumeration"]},
        }
        written = write_factorial_sensitivity_artifacts(
            self.golden,
            diagnostic,
            formats=("json", "md"),
            success=False,
        )
        self.assertFalse((out / FACTORIAL_SUCCESS_MARKER).exists())
        self.assertEqual(written["diagnostic"].name, FACTORIAL_DIAGNOSTIC_JSON)
        self.assertTrue(written["diagnostic"].is_file())
        self.assertTrue(written["md"].is_file())

    def test_alternate_output_dir_rejected_under_dataset(self) -> None:
        results = _synthetic_results(self.golden)
        with self.assertRaises(ValueError):
            write_factorial_sensitivity_artifacts(
                self.golden,
                results,
                output_dir=self.golden / "dataset" / "nope",
                success=True,
            )

    def test_cli_truncated_exits_2_and_leaves_dataset_untouched(self) -> None:
        before = _part_fingerprints(self.golden)
        out = factorial_sensitivity_output_dir(self.golden)
        if out.exists():
            shutil.rmtree(out)

        code, payload, _out, err = _run_cli(
            [
                "sensitivity",
                "--experiment-dir",
                str(self.golden),
                "--format",
                "all",
            ]
        )
        self.assertEqual(code, 2)
        self.assertEqual(payload["command"], "sensitivity")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["overall_status"], "eligibility_failure")
        self.assertIn("starting exact factorial analysis", err)
        self.assertFalse((out / FACTORIAL_SUCCESS_MARKER).exists())
        self.assertTrue((out / FACTORIAL_DIAGNOSTIC_JSON).is_file())
        self.assertEqual(_part_fingerprints(self.golden), before)

    def test_cli_usage_error_exit_1(self) -> None:
        code, _payload, _out, err = _run_cli(["sensitivity"])
        self.assertEqual(code, 1)
        self.assertIn("error", err.lower())

    def test_toy_compute_and_publish_roundtrip(self) -> None:
        base = open_brute_force_dataset(self.golden)
        space = IndexSpace.from_bounds(("a", "b"), (0, 0), (1, 2))
        context = dataclasses.replace(
            base,
            index_space=space,
            var_names=("a", "b"),
            constraint_names=("c1",),
            problem_info={
                "model": {
                    "a": {"role": "var", "values": [0, 1], "lb": 0, "ub": 1},
                    "b": {"role": "var", "values": [0, 1, 2], "lb": 0, "ub": 2},
                }
            },
            full_catalogue_rows=6,
            planned_start_index=0,
            planned_stop_index=6,
            planned_rows=6,
            is_full_enumeration=True,
            coverage_fraction=1.0,
        )
        grid = np.fromfunction(
            lambda i, j: 2.0 * i + 5.0 * j,
            (2, 3),
            dtype=float,
        )
        rows = []
        gi = 0
        for multi in np.ndindex(2, 3):
            rows.append((gi, *multi, float(grid[multi]), 0.5))
            gi += 1

        alt = self.root / "alt_out"
        if alt.exists():
            shutil.rmtree(alt)

        with open_query_engine(base) as engine:
            conn = engine.connection
            conn.execute(f"DROP VIEW IF EXISTS {quote_ident(engine.view_name)}")
            conn.execute("DROP TABLE IF EXISTS toy_cli")
            conn.execute(
                "CREATE TEMP TABLE toy_cli ("
                "global_index BIGINT, "
                '"idx__a" INTEGER, "idx__b" INTEGER, '
                "y_p DOUBLE, \"util__c1\" DOUBLE)"
            )
            conn.executemany(
                "INSERT INTO toy_cli VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.execute(
                f"CREATE VIEW {quote_ident(engine.view_name)} AS "
                'SELECT global_index, "idx__a", "idx__b", y_p, "util__c1" '
                "FROM toy_cli"
            )
            compute = compute_factorial_sobol_indices(
                context,
                engine,
                response_specs=(
                    FactorialResponseSpec(
                        family=RESPONSE_FAMILY_OBJECTIVE,
                        name="y_p",
                        source_column="y_p",
                        expression=EXPRESSION_COLUMN,
                    ),
                ),
            )

        eligibility = FactorialSensitivityEligibility(
            ok=True,
            reasons=(),
            diagnostics={"ok": True},
            response_specs=(compute.responses[0].spec,),
            variable_names=("a", "b"),
            level_counts=(2, 3),
            cardinality=6,
            invalid_utilization=1e6,
        )
        results = build_factorial_sensitivity_results(
            context,
            eligibility,
            compute,
        )
        written = write_factorial_sensitivity_artifacts(
            self.golden,
            results,
            output_dir=alt,
            success=True,
        )
        self.assertTrue((alt / FACTORIAL_SUCCESS_MARKER).is_file())
        payload = json.loads(written["json"].read_text(encoding="utf-8"))
        s1 = [
            factor["first_order"]
            for factor in payload["responses"][0]["factors"]
        ]
        self.assertAlmostEqual(s1[0], 3.0 / 53.0, places=12)
        self.assertAlmostEqual(s1[1], 50.0 / 53.0, places=12)


if __name__ == "__main__":
    unittest.main()
