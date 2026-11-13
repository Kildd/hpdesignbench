"""
Spot-check a few sealed brute-force Parquet rows against live analysis
and locked Problem-1 midpoint goldens.
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from core.ioh_core.problem_builder import build_problem_from_snapshot
from core.ioh_core.workflows.brute_force.indexing import (
    indices_to_linear,
    linear_to_indices,
)
from core.ioh_core.workflows.brute_force.planner import finish_group_size
from core.ioh_core.workflows.brute_force.schema import build_brute_force_schema
from core.ioh_core.workflows.brute_force.worker import run_worker
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
)

from _mains.Testing.test_cursor_brute_force_worker import (
    _dispatch_attempt,
    _prepare_bf_experiment,
    _problem_fixtures,
)

# Locked 2026-08-03 after penalty reformulation (same numbers as
# ``test_cursor_analysis_goldens.TestProblem1MidpointAnalysis``).
_MIDPOINT_Y = 455.0512514305275
_MIDPOINT_Y_P = 879.6436712620355
_MIDPOINT_UTIL_A = 1.0776111778632105
_MIDPOINT_UTIL_B1A = 0.26509358188026905
_MIDPOINT_UTIL_B2A = 0.750274355046204

_Y_PLACES = 3
_Y_P_PLACES = 1
_UTIL_PLACES = 4


def _midpoint_indices(lb_idx: Sequence[int], ub_idx: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(round(0.5 * (lo + hi))) for lo, hi in zip(lb_idx, ub_idx))


def _rows_by_global_index(table: Any) -> dict[int, dict[str, Any]]:
    names = list(table.column_names)
    columns = {name: table.column(name).to_pylist() for name in names}
    rows: dict[int, dict[str, Any]] = {}
    for offset in range(table.num_rows):
        global_index = int(columns["global_index"][offset])
        rows[global_index] = {name: columns[name][offset] for name in names}
    return rows


def _assert_finite_close(
    test: unittest.TestCase,
    observed: Any,
    expected: Any,
    *,
    places: int,
    field: str,
) -> None:
    test.assertIsNotNone(observed, msg=f"{field} is missing on the Parquet row")
    test.assertIsNotNone(expected, msg=f"{field} is missing on the live result")
    left = float(observed)
    right = float(expected)
    test.assertTrue(math.isfinite(left), msg=f"{field} Parquet value is not finite: {left!r}")
    test.assertTrue(
        math.isfinite(right), msg=f"{field} expected value is not finite: {right!r}"
    )
    test.assertAlmostEqual(left, right, places=places, msg=field)


class TestBruteForceScientificSpotcheck(unittest.TestCase):
    """Parquet rows for a few catalogue vectors match analysis within tolerance."""

    @classmethod
    def setUpClass(cls) -> None:
        info, space, decode, materials, constraints = _problem_fixtures()
        cls.info = info
        cls.space = space
        cls.decode = staticmethod(decode)
        cls.materials = materials
        cls.constraints = constraints
        cls.var_names = tuple(space.var_names)
        cls.constraint_names = tuple(info["constraints"].keys())
        cls.group = finish_group_size(space)
        cls.mid_indices = _midpoint_indices(space.lower_bounds, space.upper_bounds)
        cls.mid_linear = indices_to_linear(cls.mid_indices, space)
        cls.start_index = (cls.mid_linear // cls.group) * cls.group
        cls.stop_index = cls.start_index + cls.group
        if not (cls.start_index <= cls.mid_linear < cls.stop_index):
            raise AssertionError(
                f"midpoint linear {cls.mid_linear} is outside the aligned "
                f"finish group [{cls.start_index}, {cls.stop_index})"
            )
        cls.spot_linears = (
            cls.start_index,
            cls.mid_linear,
            cls.stop_index - 1,
        )
        cls.schema_info = build_brute_force_schema(
            space=space,
            model=info["model"],
            constraint_names=cls.constraint_names,
        )
        cls._tmpdir = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmpdir.name)
        cls.experiment_dir = _prepare_bf_experiment(
            cls.root,
            start_index=cls.start_index,
            stop_index=cls.stop_index,
            schema_hash=cls.schema_info.schema_hash,
        )
        attempt = _dispatch_attempt(cls.experiment_dir)
        summary = run_worker(
            experiment_dir=cls.experiment_dir,
            task_index=0,
            attempt=attempt.attempt,
            reuse_structural=True,
        )
        cls.table = pq.read_table(Path(summary["part_path"]))
        cls.rows = _rows_by_global_index(cls.table)
        snapshot_dir = resolve_snapshot_slab_dir(
            cls.experiment_dir / "inputs", "hp_slab"
        )
        bundle = build_problem_from_snapshot(
            slab_type="hp_slab",
            problem_id="L6750_B2",
            snapshot_dir=snapshot_dir,
        )
        cls.live_decode = staticmethod(bundle["decode"])
        cls.live_analysis = staticmethod(bundle["analysis_fn"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def _live_result(self, linear: int) -> dict[str, Any]:
        indices = linear_to_indices(linear, self.space)
        return self.live_analysis(self.live_decode(list(indices)))

    def _assert_row_matches_live(
        self,
        row: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        linear: int,
    ) -> None:
        indices = linear_to_indices(linear, self.space)
        self.assertEqual(int(row["global_index"]), linear)
        for name, index in zip(self.var_names, indices):
            self.assertEqual(
                int(row[f"idx__{name}"]),
                int(index),
                msg=f"idx__{name} at global_index={linear}",
            )
        self.assertEqual(bool(row["analysis_ok"]), bool(result["analysis_ok"]))
        self.assertEqual(
            bool(row["constraint_feasible"]),
            bool(result["constraint_feasible"]),
        )
        self.assertEqual(bool(row["feasible"]), bool(result["feasible"]))
        _assert_finite_close(
            self, row["y"], result["y"], places=_Y_PLACES, field="y"
        )
        _assert_finite_close(
            self, row["y_p"], result["y_p"], places=_Y_P_PLACES, field="y_p"
        )
        utils = result["constraint_values"]
        self.assertIsInstance(utils, Mapping)
        for name in self.constraint_names:
            _assert_finite_close(
                self,
                row[f"util__{name}"],
                utils.get(name),
                places=_UTIL_PLACES,
                field=f"util__{name}",
            )

    def test_spotcheck_vectors_are_present_and_distinct(self) -> None:
        self.assertEqual(self.table.num_rows, self.group)
        self.assertEqual(len(set(self.spot_linears)), 3)
        for linear in self.spot_linears:
            self.assertIn(linear, self.rows)

    def test_parquet_matches_live_analysis_for_spotcheck_vectors(self) -> None:
        for linear in self.spot_linears:
            with self.subTest(global_index=linear):
                self._assert_row_matches_live(
                    self.rows[linear],
                    self._live_result(linear),
                    linear=linear,
                )

    def test_midpoint_row_matches_locked_goldens(self) -> None:
        row = self.rows[self.mid_linear]
        for name, index in zip(self.var_names, self.mid_indices):
            self.assertEqual(int(row[f"idx__{name}"]), int(index))
        self.assertTrue(bool(row["analysis_ok"]))
        self.assertFalse(bool(row["feasible"]))
        _assert_finite_close(self, row["y"], _MIDPOINT_Y, places=_Y_PLACES, field="y")
        _assert_finite_close(
            self, row["y_p"], _MIDPOINT_Y_P, places=_Y_P_PLACES, field="y_p"
        )
        _assert_finite_close(
            self,
            row["util__A_bending_capacity"],
            _MIDPOINT_UTIL_A,
            places=_UTIL_PLACES,
            field="util__A_bending_capacity",
        )
        _assert_finite_close(
            self,
            row["util__B1a_deflection_by_wmax_capacity"],
            _MIDPOINT_UTIL_B1A,
            places=_UTIL_PLACES,
            field="util__B1a_deflection_by_wmax_capacity",
        )
        _assert_finite_close(
            self,
            row["util__B2a_failure_announcement_by_wmin_capacity"],
            _MIDPOINT_UTIL_B2A,
            places=_UTIL_PLACES,
            field="util__B2a_failure_announcement_by_wmin_capacity",
        )


if __name__ == "__main__":
    unittest.main()
