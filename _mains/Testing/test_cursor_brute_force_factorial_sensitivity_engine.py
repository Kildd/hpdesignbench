"""
C7.3 gate: fused DuckDB exact factorial Sobol engine.
"""
from __future__ import annotations

import dataclasses
import textwrap
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
    RESPONSE_FAMILY_FEASIBILITY,
    RESPONSE_FAMILY_OBJECTIVE,
    RESPONSE_FAMILY_UTILIZATION,
    STATUS_CONSTANT,
    STATUS_OK,
    FactorialResponseSpec,
    compute_factorial_sobol_indices,
    exact_sobol_from_tensor,
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


def _context_for_vars(base_context, var_names: tuple[str, ...], sizes: tuple[int, ...]):
    lowers = tuple(0 for _ in sizes)
    uppers = tuple(size - 1 for size in sizes)
    space = IndexSpace.from_bounds(var_names, lowers, uppers)
    model = {
        name: {
            "role": "var",
            "values": list(range(size)),
            "lb": 0,
            "ub": size - 1,
        }
        for name, size in zip(var_names, sizes)
    }
    return dataclasses.replace(
        base_context,
        index_space=space,
        var_names=var_names,
        param_names=var_names,
        constraint_names=("c1",),
        problem_info={"model": model},
        fixed_params={},
        full_catalogue_rows=int(space.cardinality),
        planned_start_index=0,
        planned_stop_index=int(space.cardinality),
        planned_rows=int(space.cardinality),
        is_full_enumeration=True,
        coverage_fraction=1.0,
    )


def _install_grid_view(
    engine,
    *,
    var_names: tuple[str, ...],
    shapes: tuple[int, ...],
    y_tensor: np.ndarray,
    util_tensor: np.ndarray | None = None,
    shuffle_seed: int | None = None,
) -> None:
    y_tensor = np.asarray(y_tensor, dtype=float)
    if y_tensor.shape != shapes:
        raise AssertionError(f"y shape {y_tensor.shape} != {shapes}")
    if util_tensor is None:
        util_tensor = np.full(shapes, 0.5, dtype=float)
    else:
        util_tensor = np.asarray(util_tensor, dtype=float)

    rows: list[tuple] = []
    for flat_index, multi in enumerate(np.ndindex(*shapes)):
        y_p = float(y_tensor[multi])
        util = float(util_tensor[multi])
        rows.append((flat_index, *multi, y_p, util))

    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        order = rng.permutation(len(rows))
        rows = [rows[i] for i in order]

    idx_cols = ", ".join(f'"idx__{name}" INTEGER' for name in var_names)
    n_values = 1 + len(var_names) + 2  # global_index + idxs + y_p + util
    insert_placeholders = ", ".join("?" for _ in range(n_values))
    select_idx = ", ".join(f'"idx__{name}"' for name in var_names)

    conn = engine.connection
    conn.execute(f"DROP VIEW IF EXISTS {quote_ident(engine.view_name)}")
    conn.execute("DROP TABLE IF EXISTS toy_factor_engine")
    conn.execute(
        "CREATE TEMP TABLE toy_factor_engine ("
        "global_index BIGINT, "
        f"{idx_cols}, "
        "y_p DOUBLE, "
        '"util__c1" DOUBLE'
        ")"
    )
    conn.executemany(
        f"INSERT INTO toy_factor_engine VALUES ({insert_placeholders})",
        rows,
    )
    conn.execute(
        f"CREATE VIEW {quote_ident(engine.view_name)} AS "
        f"SELECT global_index, {select_idx}, y_p, \"util__c1\" "
        "FROM toy_factor_engine"
    )


class TestFactorialSensitivityEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        cls.golden = _run_tiny_bf(
            cls.root / "golden",
            experiment_id="bf_fs_engine_gold",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def _compute(
        self,
        *,
        var_names: tuple[str, ...],
        shapes: tuple[int, ...],
        y_tensor: np.ndarray,
        util_tensor: np.ndarray | None = None,
        shuffle_seed: int | None = None,
        response_specs: tuple[FactorialResponseSpec, ...] | None = None,
    ):
        base = open_brute_force_dataset(self.golden)
        context = _context_for_vars(base, var_names, shapes)
        with open_query_engine(base) as engine:
            _install_grid_view(
                engine,
                var_names=var_names,
                shapes=shapes,
                y_tensor=y_tensor,
                util_tensor=util_tensor,
                shuffle_seed=shuffle_seed,
            )
            specs = response_specs or (
                FactorialResponseSpec(
                    family=RESPONSE_FAMILY_OBJECTIVE,
                    name="y_p",
                    source_column="y_p",
                    expression=EXPRESSION_COLUMN,
                ),
            )
            return compute_factorial_sobol_indices(
                context,
                engine,
                response_specs=specs,
            )

    def test_unequal_levels_matches_oracle(self) -> None:
        grid = np.fromfunction(
            lambda i, j: 2.0 * i + 5.0 * j,
            (2, 3),
            dtype=float,
        )
        result = self._compute(
            var_names=("a", "b"),
            shapes=(2, 3),
            y_tensor=grid,
            shuffle_seed=7,
        )
        oracle = exact_sobol_from_tensor(grid)
        self.assertEqual(result.query_count, 1 + 2 * 2)
        response = result.responses[0]
        self.assertEqual(response.status, STATUS_OK)
        np.testing.assert_allclose(
            [factor.first_order for factor in response.factors],
            oracle.first_order,
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            [factor.total_effect for factor in response.factors],
            oracle.total_effect,
            rtol=0.0,
            atol=1e-14,
        )
        for factor in response.factors:
            self.assertLessEqual(
                abs(factor.identity_residual),
                result.identity_abs_tol,
            )

    def test_mixed_oracle_shuffled_2x2(self) -> None:
        xs = np.array([-1.0, 1.0])
        ys = np.array([-1.0, 1.0])
        grid = np.empty((2, 2), dtype=float)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                grid[i, j] = x + 2.0 * y + 3.0 * x * y
        result = self._compute(
            var_names=("x", "y"),
            shapes=(2, 2),
            y_tensor=grid,
            shuffle_seed=11,
        )
        response = result.responses[0]
        expected_s1 = np.array([1.0 / 14.0, 4.0 / 14.0])
        expected_st = np.array([10.0 / 14.0, 13.0 / 14.0])
        np.testing.assert_allclose(
            [factor.first_order for factor in response.factors],
            expected_s1,
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            [factor.total_effect for factor in response.factors],
            expected_st,
            rtol=0.0,
            atol=1e-14,
        )

    def test_shuffled_2x3x2_matches_oracle(self) -> None:
        rng = np.random.default_rng(123)
        grid = rng.normal(size=(2, 3, 2))
        result = self._compute(
            var_names=("a", "b", "c"),
            shapes=(2, 3, 2),
            y_tensor=grid,
            shuffle_seed=99,
        )
        oracle = exact_sobol_from_tensor(grid)
        self.assertEqual(result.query_count, 1 + 2 * 3)
        # Largest complement cell size is max_i n_i = 3 (when conditioning on b).
        self.assertEqual(result.largest_complement_group_count, 3)
        response = result.responses[0]
        np.testing.assert_allclose(
            [factor.first_order for factor in response.factors],
            oracle.first_order,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [factor.total_effect for factor in response.factors],
            oracle.total_effect,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [
                factor.total_effect_via_complement_mean
                for factor in response.factors
            ],
            oracle.total_effect_via_complement_mean,
            rtol=0.0,
            atol=1e-12,
        )

    def test_query_count_scales_with_variables_not_responses(self) -> None:
        grid = np.fromfunction(
            lambda i, j: i + 0.5 * j,
            (2, 3),
            dtype=float,
        )
        util = np.fromfunction(
            lambda i, j: 0.5 + 0.8 * i + 0.4 * j,
            (2, 3),
            dtype=float,
        )
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
        result = self._compute(
            var_names=("a", "b"),
            shapes=(2, 3),
            y_tensor=grid,
            util_tensor=util,
            response_specs=specs,
        )
        self.assertEqual(result.query_count, 5)  # 1 + 2*2
        self.assertEqual(len(result.responses), 3)
        self.assertTrue(all(item.status == STATUS_OK for item in result.responses))

    def test_constant_response_skipped_others_continue(self) -> None:
        grid = np.fromfunction(
            lambda i, j: i + j,
            (2, 2),
            dtype=float,
        )
        util = np.ones((2, 2), dtype=float)  # constant util => constant binary too
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
        )
        result = self._compute(
            var_names=("a", "b"),
            shapes=(2, 2),
            y_tensor=grid,
            util_tensor=util,
            response_specs=specs,
        )
        by_name = {item.spec.name: item for item in result.responses}
        self.assertEqual(by_name["y_p"].status, STATUS_OK)
        self.assertEqual(by_name["util__c1"].status, STATUS_CONSTANT)
        self.assertEqual(by_name["util__c1"].factors, ())
        self.assertEqual(len(by_name["y_p"].factors), 2)

    def test_all_constant_raises(self) -> None:
        grid = np.ones((2, 2), dtype=float)
        with self.assertRaisesRegex(ValueError, "no non-constant"):
            self._compute(
                var_names=("a", "b"),
                shapes=(2, 2),
                y_tensor=grid,
            )

    def test_pure_interaction_indices(self) -> None:
        xs = np.array([-1.0, 1.0])
        ys = np.array([-1.0, 1.0])
        grid = xs[:, None] * ys[None, :]
        result = self._compute(
            var_names=("a", "b"),
            shapes=(2, 2),
            y_tensor=grid,
        )
        response = result.responses[0]
        np.testing.assert_allclose(
            [factor.first_order for factor in response.factors],
            [0.0, 0.0],
            atol=1e-15,
        )
        np.testing.assert_allclose(
            [factor.total_effect for factor in response.factors],
            [1.0, 1.0],
            atol=1e-15,
        )


if __name__ == "__main__":
    unittest.main()
