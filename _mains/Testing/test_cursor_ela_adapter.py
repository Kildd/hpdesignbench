"""
ELA adapter: one-repetition load, keep list, finite y/y_p, no silent dedup.
"""
from __future__ import annotations

import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from core.ioh_core.constraint_selection import CONSTRAINT_SUBSET_WARNING
from core.ioh_core.workflows.la.adapter import load_landscape_sample
from core.ioh_core.workflows.la.deps import import_pyarrow
from core.ioh_core.workflows.la.models import ElaError


_SEALED = ("A_bending_capacity", "Z1_nt_dt_combination_capacity")
_VARS = ("geom_t_mm", "mat_conc_id")


def _write_schema(root: Path, names: tuple[str, ...] = _SEALED) -> None:
    dataset = root / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "parts").mkdir(parents=True, exist_ok=True)
    (dataset / "schema.json").write_text(
        json.dumps(
            {
                "var_names": list(_VARS),
                "constraint_names": list(names),
            }
        ),
        encoding="utf-8",
    )


def _write_parts(root: Path, rows: list[dict]) -> None:
    pa = import_pyarrow()
    import pyarrow.parquet as pq

    dataset = root / "dataset"
    (dataset / "parts").mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, dataset / "parts" / "shard-000.parquet")


def _row(
    *,
    sample_index: int,
    repetition: int,
    idx_t: int,
    idx_mat: int,
    y: float,
    y_p: float,
    util_a: float,
    util_z: float,
    analysis_ok: bool = True,
    feasible: bool = False,
    extra: float | None = None,
) -> dict:
    payload = {
        "sample_index": sample_index,
        "repetition": repetition,
        "idx__geom_t_mm": idx_t,
        "idx__mat_conc_id": idx_mat,
        "y": y,
        "y_p": y_p,
        "util__A_bending_capacity": util_a,
        "util__Z1_nt_dt_combination_capacity": util_z,
        "analysis_ok": analysis_ok,
        "feasible": feasible,
        "constraint_feasible": feasible,
    }
    if extra is not None:
        payload["extra_noise"] = extra
    return payload


class TestElaAdapter(unittest.TestCase):
    """One-repetition landscape load."""

    def test_default_selected_includes_z_without_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_schema(root)
            _write_parts(
                root,
                [
                    _row(
                        sample_index=0,
                        repetition=0,
                        idx_t=0,
                        idx_mat=3,
                        y=1.0,
                        y_p=2.0,
                        util_a=0.5,
                        util_z=1.2,
                    )
                ],
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sample = load_landscape_sample(
                    root,
                    repetition=0,
                    source_type="sampling",
                    lower_bounds=(0, 3),
                    upper_bounds=(2, 5),
                )
            runtime = [
                item for item in caught if issubclass(item.category, RuntimeWarning)
            ]
            self.assertEqual(runtime, [])
            self.assertEqual(sample.constraint_names_selected, _SEALED)
            self.assertFalse(sample.constraint_set_is_subset)
            self.assertEqual(sample.U.shape, (1, 2))
            np.testing.assert_allclose(sample.phi, [0.2])
            np.testing.assert_array_equal(sample.F, [False])

    def test_keep_list_drops_z_and_warns(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_schema(root)
            _write_parts(
                root,
                [
                    _row(
                        sample_index=0,
                        repetition=0,
                        idx_t=0,
                        idx_mat=3,
                        y=1.0,
                        y_p=2.0,
                        util_a=0.5,
                        util_z=1.2,
                    )
                ],
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sample = load_landscape_sample(
                    root,
                    repetition=0,
                    requested_constraint_names=("A_bending_capacity",),
                    source_type="sampling",
                    lower_bounds=(0, 3),
                    upper_bounds=(2, 5),
                )
            runtime = [
                item for item in caught if issubclass(item.category, RuntimeWarning)
            ]
            self.assertEqual(len(runtime), 1)
            self.assertEqual(str(runtime[0].message), CONSTRAINT_SUBSET_WARNING)
            self.assertEqual(
                sample.constraint_names_selected, ("A_bending_capacity",)
            )
            self.assertEqual(sample.U.shape, (1, 1))
            np.testing.assert_allclose(sample.phi, [0.0])
            np.testing.assert_array_equal(sample.F, [True])
            self.assertFalse(bool(sample.metadata["uses_stored_feasible_as_F"]))

    def test_extra_columns_ignored(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_schema(root)
            _write_parts(
                root,
                [
                    _row(
                        sample_index=0,
                        repetition=0,
                        idx_t=0,
                        idx_mat=3,
                        y=1.0,
                        y_p=2.0,
                        util_a=0.4,
                        util_z=0.4,
                        extra=99.0,
                    )
                ],
            )
            sample = load_landscape_sample(
                root,
                repetition=0,
                source_type="sampling",
                lower_bounds=(0, 3),
                upper_bounds=(2, 5),
            )
            self.assertEqual(sample.n_input, 1)

    def test_missing_y_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_schema(root)
            row = _row(
                sample_index=0,
                repetition=0,
                idx_t=0,
                idx_mat=3,
                y=1.0,
                y_p=2.0,
                util_a=0.4,
                util_z=0.4,
            )
            del row["y"]
            _write_parts(root, [row])
            with self.assertRaises(ElaError) as ctx:
                load_landscape_sample(
                    root,
                    repetition=0,
                    source_type="sampling",
                    lower_bounds=(0, 3),
                    upper_bounds=(2, 5),
                )
            self.assertIn("missing required column y", str(ctx.exception))

    def test_missing_selected_util_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_schema(root)
            row = _row(
                sample_index=0,
                repetition=0,
                idx_t=0,
                idx_mat=3,
                y=1.0,
                y_p=2.0,
                util_a=0.4,
                util_z=0.4,
            )
            del row["util__Z1_nt_dt_combination_capacity"]
            _write_parts(root, [row])
            with self.assertRaises(ElaError) as ctx:
                load_landscape_sample(
                    root,
                    repetition=0,
                    source_type="sampling",
                    lower_bounds=(0, 3),
                    upper_bounds=(2, 5),
                )
            self.assertIn("util columns", str(ctx.exception))

    def test_inf_y_rejects_source(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_schema(root)
            _write_parts(
                root,
                [
                    _row(
                        sample_index=0,
                        repetition=0,
                        idx_t=0,
                        idx_mat=3,
                        y=float("inf"),
                        y_p=2.0,
                        util_a=0.4,
                        util_z=0.4,
                    )
                ],
            )
            with self.assertRaises(ElaError) as ctx:
                load_landscape_sample(
                    root,
                    repetition=0,
                    source_type="sampling",
                    lower_bounds=(0, 3),
                    upper_bounds=(2, 5),
                )
            self.assertIn("non-finite", str(ctx.exception))

    def test_nonfinite_util_does_not_reject_source(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_schema(root)
            _write_parts(
                root,
                [
                    _row(
                        sample_index=0,
                        repetition=0,
                        idx_t=0,
                        idx_mat=3,
                        y=1.0,
                        y_p=1.0,
                        util_a=float("inf"),
                        util_z=0.4,
                    )
                ],
            )
            sample = load_landscape_sample(
                root,
                repetition=0,
                source_type="sampling",
                lower_bounds=(0, 3),
                upper_bounds=(2, 5),
            )
            self.assertEqual(sample.n_input, 1)
            np.testing.assert_allclose(sample.y_raw, [1.0])
            self.assertTrue(np.isinf(sample.U[0, 0]))

    def test_analysis_ok_false_is_kept(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_schema(root)
            _write_parts(
                root,
                [
                    _row(
                        sample_index=0,
                        repetition=0,
                        idx_t=0,
                        idx_mat=3,
                        y=1.5,
                        y_p=1.5,
                        util_a=99.0,
                        util_z=0.2,
                        analysis_ok=False,
                    )
                ],
            )
            sample = load_landscape_sample(
                root,
                repetition=0,
                source_type="sampling",
                lower_bounds=(0, 3),
                upper_bounds=(2, 5),
            )
            self.assertEqual(sample.n_input, 1)
            np.testing.assert_array_equal(sample.analysis_ok, [False])
            np.testing.assert_allclose(sample.y_raw, [1.5])

    def test_two_repetitions_are_not_concatenated(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_schema(root)
            _write_parts(
                root,
                [
                    _row(
                        sample_index=1,
                        repetition=0,
                        idx_t=1,
                        idx_mat=4,
                        y=20.0,
                        y_p=20.0,
                        util_a=0.2,
                        util_z=0.2,
                    ),
                    _row(
                        sample_index=0,
                        repetition=0,
                        idx_t=0,
                        idx_mat=3,
                        y=10.0,
                        y_p=10.0,
                        util_a=0.2,
                        util_z=0.2,
                    ),
                    _row(
                        sample_index=0,
                        repetition=1,
                        idx_t=2,
                        idx_mat=5,
                        y=99.0,
                        y_p=99.0,
                        util_a=0.2,
                        util_z=0.2,
                    ),
                ],
            )
            sample = load_landscape_sample(
                root,
                repetition=0,
                source_type="sampling",
                lower_bounds=(0, 3),
                upper_bounds=(2, 5),
            )
            self.assertEqual(sample.n_input, 2)
            np.testing.assert_array_equal(sample.sample_index, [0, 1])
            np.testing.assert_allclose(sample.y_raw, [10.0, 20.0])
            np.testing.assert_array_equal(
                sample.X_index, np.array([[0, 3], [1, 4]], dtype=np.int64)
            )

    def test_duplicate_rows_are_accepted(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_schema(root)
            duplicate = _row(
                sample_index=0,
                repetition=0,
                idx_t=0,
                idx_mat=3,
                y=1.0,
                y_p=1.0,
                util_a=0.3,
                util_z=0.3,
            )
            second = _row(
                sample_index=1,
                repetition=0,
                idx_t=0,
                idx_mat=3,
                y=4.0,
                y_p=4.0,
                util_a=0.3,
                util_z=0.3,
            )
            _write_parts(root, [duplicate, second])
            sample = load_landscape_sample(
                root,
                repetition=0,
                source_type="sampling",
                lower_bounds=(0, 3),
                upper_bounds=(2, 5),
            )
            self.assertEqual(sample.n_input, 2)
            self.assertEqual(sample.n_unique, 1)
            self.assertEqual(sample.n_duplicates, 1)
            np.testing.assert_allclose(sample.y_raw, [1.0, 4.0])

    def test_nonzero_mat_conc_id_lower_bound(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_schema(root)
            _write_parts(
                root,
                [
                    _row(
                        sample_index=0,
                        repetition=0,
                        idx_t=0,
                        idx_mat=2,
                        y=1.0,
                        y_p=1.0,
                        util_a=0.3,
                        util_z=0.3,
                    )
                ],
            )
            with self.assertRaises(ElaError) as ctx:
                load_landscape_sample(
                    root,
                    repetition=0,
                    source_type="sampling",
                    lower_bounds=(0, 3),
                    upper_bounds=(2, 5),
                )
            self.assertIn("mat_conc_id", str(ctx.exception))

    def test_phi_f_ignore_stored_feasible(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_schema(root)
            _write_parts(
                root,
                [
                    _row(
                        sample_index=0,
                        repetition=0,
                        idx_t=0,
                        idx_mat=3,
                        y=1.0,
                        y_p=9.0,
                        util_a=0.8,
                        util_z=0.9,
                        feasible=False,
                    )
                ],
            )
            sample = load_landscape_sample(
                root,
                repetition=0,
                source_type="sampling",
                lower_bounds=(0, 3),
                upper_bounds=(2, 5),
            )
            np.testing.assert_array_equal(sample.F, [True])
            np.testing.assert_allclose(sample.phi, [0.0])
            np.testing.assert_allclose(sample.y_p, [9.0])


if __name__ == "__main__":
    unittest.main()
