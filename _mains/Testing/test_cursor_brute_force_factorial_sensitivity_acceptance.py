"""
C7.5 gate: factorial sensitivity smoke, shaped benchmark metrics, docs lock.
"""
from __future__ import annotations

import dataclasses
import time
import unittest
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
from core.ioh_core.workflows.brute_force.analysis.engine import quote_ident
from core.ioh_core.workflows.brute_force.analysis.factorial_sensitivity import (
    EXPRESSION_BINARY_FEASIBLE,
    EXPRESSION_COLUMN,
    FACTORIAL_SUCCESS_MARKER,
    RESPONSE_FAMILY_FEASIBILITY,
    RESPONSE_FAMILY_OBJECTIVE,
    RESPONSE_FAMILY_UTILIZATION,
    FactorialResponseSpec,
    compute_factorial_sobol_indices,
    exact_sobol_from_tensor,
    write_factorial_sensitivity_artifacts,
)
from core.ioh_core.workflows.brute_force.models import IndexSpace

REPO_ROOT = Path(__file__).resolve().parents[2]
BRUTE_FORCE_DOC = REPO_ROOT / "_docs" / "brute_force.md"


def _write_bf_config(path: Path, *, experiment_id: str) -> None:
    path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "",
                "[experiment]",
                f'id = "{experiment_id}"',
                'workflow = "brute_force"',
                'slab_type = "hp_slab"',
                'problem_ids = ["L6750_B2"]',
                'output_root = "experiments"',
                "",
                "[brute_force]",
                "rows_per_shard = 24",
                "max_rows = 48",
                "start_index = 0",
                "parquet_batch_rows = 24",
                'compression = "zstd"',
                "reuse_structural_artifacts = true",
                "acknowledge_large_run = true",
                "maximum_expected_rows = 1000000",
                "maximum_expected_bytes = 10000000000",
                "",
                "[execution]",
                'backend = "local"',
                "max_parallel_tasks = 2",
                "timeout_seconds = 3600",
                "cpus_per_task = 1",
                "",
                "[execution.local]",
                'python_executable = ""',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_tiny_bf(root: Path, *, experiment_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / f"{experiment_id}.toml"
    _write_bf_config(config_path, experiment_id=experiment_id)
    experiment_dir = root / "experiments" / experiment_id
    prepare_experiment(config_path=config_path, output=experiment_dir)
    dispatch_experiment(experiment_dir)
    collect_experiment(experiment_dir, require_complete=True)
    return experiment_dir


def _context_for_shapes(base, var_names: tuple[str, ...], shapes: tuple[int, ...]):
    space = IndexSpace.from_bounds(
        var_names,
        tuple(0 for _ in shapes),
        tuple(size - 1 for size in shapes),
    )
    return dataclasses.replace(
        base,
        index_space=space,
        var_names=var_names,
        param_names=var_names,
        constraint_names=("c1",),
        problem_info={
            "model": {
                name: {
                    "role": "var",
                    "values": list(range(size)),
                    "lb": 0,
                    "ub": size - 1,
                }
                for name, size in zip(var_names, shapes)
            }
        },
        full_catalogue_rows=int(space.cardinality),
        planned_start_index=0,
        planned_stop_index=int(space.cardinality),
        planned_rows=int(space.cardinality),
        is_full_enumeration=True,
        coverage_fraction=1.0,
    )


def _install_grid(engine, *, var_names, shapes, y_tensor, util_tensor):
    rows = []
    for flat, multi in enumerate(np.ndindex(*shapes)):
        rows.append(
            (
                flat,
                *multi,
                float(y_tensor[multi]),
                float(util_tensor[multi]),
            )
        )
    idx_defs = ", ".join(f'"idx__{name}" INTEGER' for name in var_names)
    idx_select = ", ".join(f'"idx__{name}"' for name in var_names)
    placeholders = ", ".join("?" for _ in range(1 + len(var_names) + 2))
    conn = engine.connection
    conn.execute(f"DROP VIEW IF EXISTS {quote_ident(engine.view_name)}")
    conn.execute("DROP TABLE IF EXISTS toy_fs_accept")
    conn.execute(
        "CREATE TEMP TABLE toy_fs_accept ("
        f"global_index BIGINT, {idx_defs}, y_p DOUBLE, \"util__c1\" DOUBLE)"
    )
    conn.executemany(
        f"INSERT INTO toy_fs_accept VALUES ({placeholders})",
        rows,
    )
    conn.execute(
        f"CREATE VIEW {quote_ident(engine.view_name)} AS "
        f"SELECT global_index, {idx_select}, y_p, \"util__c1\" "
        "FROM toy_fs_accept"
    )


class TestFactorialSensitivityAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        cls.golden = _run_tiny_bf(cls.root / "golden", experiment_id="bf_fs_acc")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_docs_lock_exact_factorial_section(self) -> None:
        text = BRUTE_FORCE_DOC.read_text(encoding="utf-8")
        self.assertIn("Exact full-factorial sensitivity", text)
        self.assertIn("hdb-bf-analyze sensitivity", text)
        self.assertIn("10.1137/120876782", text)
        self.assertIn("53,374,464", text)
        self.assertIn("Hoeffding", text)
        self.assertIn("Saltelli", text)
        self.assertIn("1 + 2d", text)

    def test_small_cartesian_smoke_matches_oracle_and_publishes(self) -> None:
        base = open_brute_force_dataset(self.golden)
        shapes = (2, 3)
        var_names = ("a", "b")
        context = _context_for_shapes(base, var_names, shapes)
        y = np.fromfunction(lambda i, j: 2.0 * i + 5.0 * j, shapes, dtype=float)
        util = np.fromfunction(lambda i, j: 0.5 + 0.8 * i + 0.4 * j, shapes, dtype=float)
        specs = (
            FactorialResponseSpec(
                family=RESPONSE_FAMILY_OBJECTIVE,
                name="y_p",
                source_column="y_p",
                expression=EXPRESSION_COLUMN,
            ),
            FactorialResponseSpec(
                family=RESPONSE_FAMILY_UTILIZATION,
                name="util__c1",
                source_column="util__c1",
                expression=EXPRESSION_COLUMN,
            ),
            FactorialResponseSpec(
                family=RESPONSE_FAMILY_FEASIBILITY,
                name="feasible__c1",
                source_column="util__c1",
                expression=EXPRESSION_BINARY_FEASIBLE,
            ),
        )
        with open_query_engine(base) as engine:
            _install_grid(
                engine,
                var_names=var_names,
                shapes=shapes,
                y_tensor=y,
                util_tensor=util,
            )
            started = time.perf_counter()
            result = compute_factorial_sobol_indices(
                context,
                engine,
                response_specs=specs,
            )
            elapsed = time.perf_counter() - started

        self.assertEqual(result.query_count, 1 + 2 * len(var_names))
        self.assertGreater(result.largest_complement_group_count, 0)
        self.assertLess(elapsed, 5.0)

        oracle = exact_sobol_from_tensor(y)
        yp = result.responses[0]
        np.testing.assert_allclose(
            [factor.first_order for factor in yp.factors],
            oracle.first_order,
            atol=1e-12,
        )

        # Deterministic republish of a minimal success document.
        out = self.root / "smoke_out"
        doc = {
            "schema_version": 1,
            "kind": "factorial_sensitivity",
            "ok": True,
            "overall_status": "success",
            "method": {
                "method_id": result.method.method_id,
                "doi": result.method.doi,
            },
            "computation": {
                "query_count": result.query_count,
                "largest_complement_group_count": (
                    result.largest_complement_group_count
                ),
                "elapsed_seconds": elapsed,
            },
            "responses": [
                {
                    "family": item.spec.family,
                    "name": item.spec.name,
                    "status": item.status,
                    "mean": item.mean,
                    "total_variance": item.total_variance,
                    "factors": [
                        {
                            "variable": factor.variable,
                            "axis": factor.axis,
                            "first_order": factor.first_order,
                            "total_effect": factor.total_effect,
                            "interaction_gap": factor.interaction_gap,
                            "identity_residual": factor.identity_residual,
                        }
                        for factor in item.factors
                    ],
                }
                for item in result.responses
            ],
        }
        written1 = write_factorial_sensitivity_artifacts(
            self.golden,
            doc,
            output_dir=out,
            success=True,
        )
        written2 = write_factorial_sensitivity_artifacts(
            self.golden,
            doc,
            output_dir=out,
            success=True,
        )
        self.assertTrue((out / FACTORIAL_SUCCESS_MARKER).is_file())
        self.assertEqual(
            written1["json"].read_bytes(),
            written2["json"].read_bytes(),
        )
        self.assertEqual(
            written1["csv"].read_bytes(),
            written2["csv"].read_bytes(),
        )

    def test_production_shaped_benchmark_metrics(self) -> None:
        # Shaped like a small multi-factor catalogue (not Problem 1): 4*3*2*2=48.
        base = open_brute_force_dataset(self.golden)
        shapes = (4, 3, 2, 2)
        var_names = ("a", "b", "c", "d")
        context = _context_for_shapes(base, var_names, shapes)
        rng = np.random.default_rng(2026)
        y = rng.normal(size=shapes)
        util = 0.4 + 0.5 * rng.random(size=shapes)
        with open_query_engine(base) as engine:
            _install_grid(
                engine,
                var_names=var_names,
                shapes=shapes,
                y_tensor=y,
                util_tensor=util,
            )
            started = time.perf_counter()
            result = compute_factorial_sobol_indices(
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
            elapsed = time.perf_counter() - started

        self.assertEqual(result.query_count, 1 + 2 * 4)
        # Peak complement cell size is max_i n_i = 4.
        self.assertEqual(result.largest_complement_group_count, 4)
        self.assertLess(elapsed, 10.0)
        oracle = exact_sobol_from_tensor(y)
        np.testing.assert_allclose(
            [factor.total_effect for factor in result.responses[0].factors],
            oracle.total_effect,
            atol=1e-11,
        )


if __name__ == "__main__":
    unittest.main()
