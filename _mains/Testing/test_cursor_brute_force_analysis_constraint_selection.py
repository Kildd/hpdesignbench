"""
BF analysis keep-list: default full sealed set, subset warning, output guard.
"""
from __future__ import annotations

import io
import json
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from core.ioh_core.constraint_selection import CONSTRAINT_SUBSET_WARNING
from core.ioh_core.workflows.brute_force.analysis.cli import main
from core.ioh_core.workflows.brute_force.analysis.keep_list import (
    assert_subset_analysis_output_dir,
    resolve_analysis_constraint_names,
)
from core.ioh_core.workflows.brute_force.analysis.report import analysis_output_dir
from core.ioh_core.workflows.brute_force.analysis.context import (
    BruteForceAnalysisError,
)


_SEALED = (
    "A_bending_capacity",
    "Z1_nt_dt_combination_capacity",
)


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _json_from_text(text: str) -> dict:
    lines = [line for line in text.splitlines() if line.lstrip().startswith("{")]
    if not lines:
        raise AssertionError(f"no JSON object in: {text!r}")
    return json.loads(lines[-1])


def _write_schema(experiment_dir: Path, names: tuple[str, ...] = _SEALED) -> None:
    dataset = Path(experiment_dir) / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "schema.json").write_text(
        json.dumps({"constraint_names": list(names)}),
        encoding="utf-8",
    )


class TestBruteForceAnalysisConstraintSelection(unittest.TestCase):
    """Keep-list overlay for hdb-bf-analyze."""

    def test_default_keep_list_emits_no_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selected = resolve_analysis_constraint_names(_SEALED, None)
        self.assertEqual(selected, _SEALED)
        runtime = [
            item for item in caught if issubclass(item.category, RuntimeWarning)
        ]
        self.assertEqual(runtime, [])

    def test_keep_list_omitting_z_warns(self) -> None:
        keep = ("A_bending_capacity",)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selected = resolve_analysis_constraint_names(_SEALED, keep)
        self.assertEqual(selected, keep)
        runtime = [
            item for item in caught if issubclass(item.category, RuntimeWarning)
        ]
        self.assertEqual(len(runtime), 1)
        self.assertEqual(str(runtime[0].message), CONSTRAINT_SUBSET_WARNING)

    def test_subset_refuses_canonical_output_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            canonical = analysis_output_dir(experiment_dir)
            with self.assertRaises(BruteForceAnalysisError) as ctx:
                assert_subset_analysis_output_dir(
                    experiment_dir,
                    canonical,
                    output_dir_provided=True,
                )
            self.assertIn("canonical", str(ctx.exception))
            with self.assertRaises(BruteForceAnalysisError) as missing:
                assert_subset_analysis_output_dir(
                    experiment_dir,
                    canonical,
                    output_dir_provided=False,
                )
            self.assertIn("--output-dir", str(missing.exception))

    def test_cli_keep_list_warns_and_refuses_canonical_output_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_schema(experiment_dir)
            canonical = analysis_output_dir(experiment_dir)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                code, _stdout, stderr = _run_main(
                    [
                        "report",
                        "--experiment-dir",
                        str(experiment_dir),
                        "--constraint-names",
                        "A_bending_capacity",
                        "--output-dir",
                        str(canonical),
                    ]
                )
            self.assertEqual(code, 1)
            payload = _json_from_text(stderr)
            self.assertIn("canonical", payload["error"])
            runtime = [
                item for item in caught if issubclass(item.category, RuntimeWarning)
            ]
            self.assertEqual(len(runtime), 1)
            self.assertEqual(str(runtime[0].message), CONSTRAINT_SUBSET_WARNING)

    def test_integrity_rejects_constraint_names(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_schema(experiment_dir)
            code, stdout, stderr = _run_main(
                [
                    "integrity",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--constraint-names",
                    "A_bending_capacity",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_text(stdout + stderr)
            self.assertIn("--constraint-names", payload["error"])
            self.assertIn("integrity", payload["error"].lower())

    def test_decision_tree_without_flag_warns_on_implicit_z_drop(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_schema(experiment_dir)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                code, _stdout, stderr = _run_main(
                    [
                        "decision-tree",
                        "--experiment-dir",
                        str(experiment_dir),
                    ]
                )
            runtime = [
                item for item in caught if issubclass(item.category, RuntimeWarning)
            ]
            self.assertEqual(len(runtime), 1)
            self.assertEqual(str(runtime[0].message), CONSTRAINT_SUBSET_WARNING)
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
