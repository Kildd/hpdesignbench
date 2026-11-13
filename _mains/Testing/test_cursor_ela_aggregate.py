"""
ELA signature grouping: identical pairs only; never silent mixes.
"""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from core.ioh_core.workflows.la.aggregate import (
    FEATURES_PARQUET,
    SUMMARY_DIR,
    SUMMARY_PARQUET,
    SUCCESS_MARKER,
    SignaturePair,
    group_ela_rows,
    load_feature_parquet,
    publish_family_summary,
    summarize_ela_rows,
    summarize_rfb_walks,
)
from core.ioh_core.workflows.la.cli import main
from core.ioh_core.workflows.la.deps import ElaDependencyError, import_pyarrow
from core.ioh_core.workflows.la.models import (
    ElaError,
    STATUS_CONSTANT_LANDSCAPE,
    STATUS_MISSING_DEPENDENCY,
    feature_result,
)
from core.ioh_core.workflows.la.orchestrate import (
    family_output_dir,
    write_feature_parquet,
)

_SAMPLING_A = "a" * 64
_SAMPLING_B = "b" * 64
_ANALYSIS_Y_RAW = "c" * 64
_ANALYSIS_PHI = "d" * 64


def _row(**overrides: object):
    payload: dict[str, object] = {
        "problem_id": "L6750_B2",
        "source_type": "sampling",
        "source_dir": Path("experiments/sampling_p1"),
        "landscape": "y_raw",
        "response_view": "minmax",
        "x_view": "norm",
        "family": "ela_meta",
        "feature_name": "ela_meta.lin_simple.adj_r2",
        "status": "ok",
        "role": "primary",
        "n_input": 8,
        "n_unique": 8,
        "runtime_s": 0.01,
        "library": "pflacco",
        "library_version": "1.2.2",
        "config_payload": {},
        "value": 0.5,
        "repetition": 0,
        "sampling_design_signature": _SAMPLING_A,
        "analysis_signature": _ANALYSIS_Y_RAW,
        "seed": 1000,
    }
    payload.update(overrides)
    return feature_result(**payload)  # type: ignore[arg-type]


class TestGroupElaRows(unittest.TestCase):
    """Default fail on mixed pairs; optional separate groups."""

    def test_two_reps_same_pair_are_one_group(self) -> None:
        rows = [
            _row(repetition=0, seed=1000, value=0.4),
            _row(
                repetition=1,
                seed=1001,
                feature_name="ela_meta.quad_simple.adj_r2",
                status="constant_landscape",
                value=None,
            ),
        ]
        grouped = group_ela_rows(rows)
        self.assertEqual(len(grouped), 1)
        pair = SignaturePair(_SAMPLING_A, _ANALYSIS_Y_RAW)
        self.assertEqual(tuple(grouped[pair]), tuple(rows))

    def test_multiple_landscapes_fail_by_default(self) -> None:
        rows = [
            _row(landscape="y_raw", analysis_signature=_ANALYSIS_Y_RAW),
            _row(landscape="phi", analysis_signature=_ANALYSIS_PHI),
        ]
        with self.assertRaises(ElaError) as caught:
            group_ela_rows(rows)
        message = str(caught.exception)
        self.assertIn("single", message)
        self.assertIn("sampling_design_signature", message)
        self.assertIn("allow_multiple_groups", message)

    def test_allow_multiple_groups_keeps_pairs_separate(self) -> None:
        y_raw = _row(landscape="y_raw", analysis_signature=_ANALYSIS_Y_RAW)
        phi = _row(landscape="phi", analysis_signature=_ANALYSIS_PHI)
        grouped = group_ela_rows(
            [y_raw, phi],
            allow_multiple_groups=True,
        )
        self.assertEqual(len(grouped), 2)
        self.assertEqual(
            grouped[SignaturePair(_SAMPLING_A, _ANALYSIS_Y_RAW)],
            (y_raw,),
        )
        self.assertEqual(
            grouped[SignaturePair(_SAMPLING_A, _ANALYSIS_PHI)],
            (phi,),
        )

    def test_different_sampling_design_fails_or_splits(self) -> None:
        left = _row(sampling_design_signature=_SAMPLING_A)
        right = _row(sampling_design_signature=_SAMPLING_B)
        with self.assertRaises(ElaError):
            group_ela_rows([left, right])
        grouped = group_ela_rows(
            [left, right],
            allow_multiple_groups=True,
        )
        self.assertEqual(len(grouped), 2)
        self.assertEqual(
            grouped[SignaturePair(_SAMPLING_A, _ANALYSIS_Y_RAW)],
            (left,),
        )
        self.assertEqual(
            grouped[SignaturePair(_SAMPLING_B, _ANALYSIS_Y_RAW)],
            (right,),
        )

    def test_missing_signatures_fail_closed(self) -> None:
        with self.assertRaises(ElaError) as caught:
            group_ela_rows([_row(sampling_design_signature=None)])
        self.assertIn("sampling_design_signature", str(caught.exception))
        with self.assertRaises(ElaError) as caught:
            group_ela_rows([_row(analysis_signature=None)])
        self.assertIn("analysis_signature", str(caught.exception))

    def test_empty_or_non_bool_flag_fail_closed(self) -> None:
        with self.assertRaises(ElaError):
            group_ela_rows([])
        with self.assertRaises(ElaError):
            group_ela_rows([_row()], allow_multiple_groups=1)  # type: ignore[arg-type]


def _pyarrow_available() -> bool:
    try:
        import_pyarrow()
    except ElaDependencyError:
        return False
    return True


def _expected_quantiles(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    median = float(np.quantile(array, 0.5, method="linear"))
    q1, q3 = (float(item) for item in np.quantile(array, [0.25, 0.75], method="linear"))
    return median, q1, q3, q3 - q1


def _run_main(argv: list[str]) -> tuple[int, str]:
    import io
    from contextlib import redirect_stderr, redirect_stdout

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stderr.getvalue()


def _json_from_stderr(stderr: str) -> dict:
    lines = [line for line in stderr.splitlines() if line.lstrip().startswith("{")]
    if not lines:
        raise AssertionError(f"no JSON object in stderr: {stderr!r}")
    return json.loads(lines[-1])


class TestSummarizeElaRows(unittest.TestCase):
    """Median / Q1 / IQR over ok finite values; skip runtime; keep occupancy."""

    def test_two_reps_median_q1_iqr(self) -> None:
        values = [0.2, 0.4]
        rows = [
            _row(repetition=0, seed=1000, value=values[0]),
            _row(repetition=1, seed=1001, value=values[1]),
        ]
        summaries = summarize_ela_rows(rows)
        self.assertEqual(len(summaries), 1)
        item = summaries[0]
        median, q1, q3, iqr = _expected_quantiles(values)
        self.assertEqual(item.count, 2)
        self.assertEqual(item.median, median)
        self.assertEqual(item.q1, q1)
        self.assertEqual(item.q3, q3)
        self.assertEqual(item.iqr, iqr)
        self.assertEqual(item.min_value, 0.2)
        self.assertEqual(item.max_value, 0.4)
        self.assertEqual(item.sampling_design_signature, _SAMPLING_A)
        self.assertEqual(item.analysis_signature, _ANALYSIS_Y_RAW)

    def test_mixed_landscapes_fail_unless_allowed(self) -> None:
        rows = [
            _row(landscape="y_raw", analysis_signature=_ANALYSIS_Y_RAW, value=0.2),
            _row(landscape="phi", analysis_signature=_ANALYSIS_PHI, value=0.4),
        ]
        with self.assertRaises(ElaError):
            summarize_ela_rows(rows)
        summaries = summarize_ela_rows(rows, allow_multiple_groups=True)
        self.assertEqual(len(summaries), 2)
        self.assertEqual({item.landscape for item in summaries}, {"y_raw", "phi"})

    def test_constant_and_missing_dependency_are_excluded(self) -> None:
        rows = [
            _row(repetition=0, value=0.5),
            _row(
                repetition=1,
                status=STATUS_CONSTANT_LANDSCAPE,
                value=None,
            ),
            _row(
                repetition=2,
                status=STATUS_MISSING_DEPENDENCY,
                value=None,
            ),
        ]
        item = summarize_ela_rows(rows)[0]
        self.assertEqual(item.count, 1)
        self.assertEqual(item.median, 0.5)
        self.assertIsNone(item.q1)
        self.assertIsNone(item.q3)
        self.assertIsNone(item.iqr)

    def test_n1_keeps_exact_median_and_null_quartiles(self) -> None:
        item = summarize_ela_rows([_row(value=0.42)])[0]
        self.assertEqual(item.count, 1)
        self.assertEqual(item.median, 0.42)
        self.assertIsNone(item.q1)
        self.assertIsNone(item.q3)
        self.assertIsNone(item.iqr)
        self.assertEqual(item.min_value, 0.42)
        self.assertEqual(item.max_value, 0.42)

    def test_skips_costs_runtime_keeps_occupancy(self) -> None:
        rows = [
            _row(
                feature_name="ela_meta.lin_simple.costs_runtime",
                value=0.01,
            ),
            _row(
                feature_name="cm_occupancy.n_cells",
                value=12.0,
                role="diagnostic",
            ),
        ]
        summaries = summarize_ela_rows(rows)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].feature_name, "cm_occupancy.n_cells")
        self.assertEqual(summaries[0].count, 1)
        self.assertEqual(summaries[0].median, 12.0)

    def test_matrix_feature_names_stay_independent(self) -> None:
        rows = [
            _row(feature_name="constrained.M_ij.A.Z1", value=0.1),
            _row(feature_name="constrained.M_ij.Z1.A", value=0.9),
        ]
        summaries = summarize_ela_rows(rows)
        self.assertEqual(
            [item.feature_name for item in summaries],
            ["constrained.M_ij.A.Z1", "constrained.M_ij.Z1.A"],
        )
        self.assertEqual(summaries[0].median, 0.1)
        self.assertEqual(summaries[1].median, 0.9)

    def test_duplicate_ok_rows_fail_closed(self) -> None:
        rows = [
            _row(repetition=0, walk_id=None, value=0.2),
            _row(repetition=0, walk_id=None, value=0.3),
        ]
        with self.assertRaises(ElaError) as caught:
            summarize_ela_rows(rows)
        self.assertIn("duplicate", str(caught.exception))

    def test_empty_ok_pool_emits_null_stats(self) -> None:
        rows = [
            _row(status=STATUS_MISSING_DEPENDENCY, value=None),
        ]
        item = summarize_ela_rows(rows)[0]
        self.assertEqual(item.count, 0)
        self.assertIsNone(item.median)
        self.assertIsNone(item.q1)
        self.assertIsNone(item.min_value)


class TestSummarizeRfbWalks(unittest.TestCase):
    """Keep walk rows; min / median / max of rfb_x."""

    def test_min_median_max(self) -> None:
        walks = (
            SimpleNamespace(walk_id=0, rfb_x=0.1),
            SimpleNamespace(walk_id=1, rfb_x=0.5),
            SimpleNamespace(walk_id=2, rfb_x=0.9),
        )
        summary = summarize_rfb_walks(walks)
        self.assertEqual(summary.walks, walks)
        self.assertEqual(summary.count, 3)
        self.assertEqual(summary.min_value, 0.1)
        self.assertEqual(summary.median, 0.5)
        self.assertEqual(summary.max_value, 0.9)

    def test_does_not_import_rfb_module(self) -> None:
        source = (
            Path(__file__).resolve().parents[2]
            / "core"
            / "ioh_core"
            / "workflows"
            / "la"
            / "aggregate.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertFalse(any("rfb" in name.split(".")[-1] for name in imported))

    def test_empty_or_duplicate_walk_id_fails(self) -> None:
        with self.assertRaises(ElaError):
            summarize_rfb_walks([])
        with self.assertRaises(ElaError) as caught:
            summarize_rfb_walks(
                [
                    SimpleNamespace(walk_id=0, rfb_x=0.1),
                    SimpleNamespace(walk_id=0, rfb_x=0.2),
                ]
            )
        self.assertIn("duplicate", str(caught.exception))


@unittest.skipUnless(_pyarrow_available(), "ela extra (pyarrow) not installed")
class TestPublishAndCliAggregate(unittest.TestCase):
    """Tidy summary.parquet; do not overwrite features.parquet."""

    def test_publish_writes_summary_not_features(self) -> None:
        with TemporaryDirectory() as tmp:
            family_dir = Path(tmp) / "constraints"
            family_dir.mkdir()
            rows = [
                _row(repetition=0, value=0.2),
                _row(repetition=1, value=0.4),
            ]
            features = family_dir / FEATURES_PARQUET
            write_feature_parquet(features, rows)
            before = features.read_bytes()
            summaries = summarize_ela_rows(rows)
            published = publish_family_summary(family_dir, summaries)
            self.assertEqual(published, family_dir / SUMMARY_PARQUET)
            self.assertEqual(features.read_bytes(), before)
            self.assertTrue((family_dir / SUMMARY_DIR / SUCCESS_MARKER).is_file())
            loaded = load_feature_parquet(features)
            self.assertEqual(len(loaded), 2)
            pa_parquet = __import__("importlib").import_module("pyarrow.parquet")
            table = pa_parquet.read_table(published)
            self.assertEqual(
                list(table.column_names),
                [
                    "sampling_design_signature",
                    "analysis_signature",
                    "family",
                    "landscape",
                    "feature_name",
                    "count",
                    "median",
                    "q1",
                    "q3",
                    "iqr",
                    "min",
                    "max",
                ],
            )
            self.assertEqual(table.num_rows, 1)
            self.assertEqual(table.column("count").to_pylist(), [2])

    def test_cli_aggregate_writes_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = root / "exp"
            experiment_dir.mkdir()
            out = root / "la"
            family_dir = family_output_dir(out, "constraints")
            family_dir.mkdir(parents=True)
            rows = [
                _row(repetition=0, value=0.2),
                _row(repetition=1, value=0.4),
            ]
            features = family_dir / FEATURES_PARQUET
            write_feature_parquet(features, rows)
            before = features.read_bytes()
            code, stderr = _run_main(
                [
                    "aggregate",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                    "--output-dir",
                    str(out),
                    "--landscapes",
                    "y_raw",
                    "--families",
                    "constraints",
                ]
            )
            payload = _json_from_stderr(stderr)
            self.assertEqual(code, 0, msg=stderr)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["written"], ["constraints"])
            self.assertEqual(features.read_bytes(), before)
            self.assertTrue((family_dir / SUMMARY_PARQUET).is_file())
            self.assertTrue((family_dir / SUMMARY_DIR / SUCCESS_MARKER).is_file())

    def test_cli_mixed_landscapes_fail_without_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = root / "exp"
            experiment_dir.mkdir()
            out = root / "la"
            family_dir = family_output_dir(out, "constraints")
            family_dir.mkdir(parents=True)
            rows = [
                _row(landscape="y_raw", analysis_signature=_ANALYSIS_Y_RAW, value=0.2),
                _row(landscape="phi", analysis_signature=_ANALYSIS_PHI, value=0.4),
            ]
            write_feature_parquet(family_dir / FEATURES_PARQUET, rows)
            code, stderr = _run_main(
                [
                    "aggregate",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                    "--output-dir",
                    str(out),
                    "--families",
                    "constraints",
                ]
            )
            payload = _json_from_stderr(stderr)
            self.assertEqual(code, 1, msg=stderr)
            self.assertFalse(payload["ok"])
            self.assertIn("sampling_design_signature", payload["error"])
            allowed, allowed_stderr = _run_main(
                [
                    "aggregate",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                    "--output-dir",
                    str(out),
                    "--families",
                    "constraints",
                    "--allow-multiple-groups",
                ]
            )
            allowed_payload = _json_from_stderr(allowed_stderr)
            self.assertEqual(allowed, 0, msg=allowed_stderr)
            self.assertTrue(allowed_payload["ok"])


if __name__ == "__main__":
    unittest.main()
