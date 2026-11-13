"""Predesign run writer: ranked table and InitialDesign, no slab solve."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from core.ioh_core.workflows.predesign_paths import (
    predesign_initial_design_deb_path,
    predesign_initial_design_path,
    predesign_results_path,
)
from slab_construction.slabs.hp_slab.predesign import write_predesign_run

_ACTIVE_NAMES: tuple[str, ...] | None = None


def _active_names() -> tuple[str, ...]:
    global _ACTIVE_NAMES
    if _ACTIVE_NAMES is None:
        from slab_construction.slabs.hp_slab.predesign import (
            _resolve_hp_slab_bundle,
        )

        bundle, _materials = _resolve_hp_slab_bundle("L8100_B2")
        _ACTIVE_NAMES = tuple(bundle["active_constraint_names"])
    return _ACTIVE_NAMES


def _row(
    *,
    candidate_index: int,
    predesign_y_p: float,
    height_mm: float,
    concrete: str,
    y: float | None = None,
    feasible: bool = True,
    analysis_ok: bool = True,
    violations: dict[str, float] | None = None,
) -> dict:
    names = _active_names()
    objective = predesign_y_p if y is None else y
    recorded = {name: 0.0 for name in names}
    if violations:
        recorded.update(violations)
    return {
        "candidate_index": candidate_index,
        "predesign_y_p": predesign_y_p,
        "y": objective,
        "geom_h_ges_mm": height_mm,
        "geom_t_mm": 20.0,
        "geom_nt": 5.0,
        "reinf_a_tex_mm2": 1.81,
        "mat_conc_id": concrete,
        "mat_reinf_id": "solidian Roving 2R (Z-1.6-308)",
        "geom_span_mm": 8100.0,
        "reinf_kap_t_percent": 0.0,
        "analysis_ok": analysis_ok,
        "feasible": feasible,
        "violations_": recorded,
        "constraint_values": {name: 0.5 for name in names},
    }


class TestPredesignRunWriter(unittest.TestCase):
    """Injected rows stand in for the grid. No slab is built."""

    def test_ranked_initial_design_and_replace(self) -> None:
        first_rows = [
            _row(
                candidate_index=0,
                predesign_y_p=2.0,
                height_mm=100.0,
                concrete="C25/30",
            ),
            _row(
                candidate_index=1,
                predesign_y_p=1.0,
                height_mm=200.0,
                concrete="C50/60",
            ),
        ]
        second_rows = [
            _row(
                candidate_index=4,
                predesign_y_p=0.5,
                height_mm=300.0,
                concrete="C100/115",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_predesign_run(
                "L8100_B2",
                root,
                evaluated_rows=first_rows,
            )
            results = predesign_results_path(root, "L8100_B2")
            design = predesign_initial_design_path(root, "L8100_B2")
            self.assertTrue(results.is_file())
            self.assertTrue(design.is_file())
            self.assertEqual(design.parent.parent.name, "predesign")
            self.assertNotIn("runs", design.parts)
            self.assertFalse((root / "runs" / "predesign").exists())
            payload = json.loads(design.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload),
                {"schema_version", "problem_id", "n", "vectors"},
            )
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["problem_id"], "L8100_B2")
            self.assertEqual(payload["n"], len(payload["vectors"]))
            self.assertEqual(payload["n"], 2)
            text = results.read_text(encoding="utf-8")
            self.assertIn("rank", text.splitlines()[0])
            self.assertLess(
                text.index("200.0"),
                text.index("100.0"),
            )

            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                write_predesign_run(
                    "L8100_B2",
                    root,
                    evaluated_rows=second_rows,
                )
            warning = buf.getvalue()
            self.assertIn("replacing InitialDesign for problem", warning)
            self.assertIn("L8100_B2", warning)
            self.assertIn("results.csv", warning)
            self.assertIn("initial_design.json", warning)
            self.assertIn("initial_design_deb.json", warning)
            replaced = json.loads(design.read_text(encoding="utf-8"))
            self.assertEqual(replaced["n"], 1)
            self.assertEqual(len(replaced["vectors"]), 1)
            self.assertNotIn(replaced["vectors"][0], payload["vectors"])
            replaced_csv = results.read_text(encoding="utf-8")
            self.assertIn("4", replaced_csv)

    def test_replacing_one_problem_leaves_the_other(self) -> None:
        row_a = [
            _row(
                candidate_index=0,
                predesign_y_p=1.0,
                height_mm=100.0,
                concrete="C25/30",
            )
        ]
        row_b = [
            _row(
                candidate_index=1,
                predesign_y_p=1.0,
                height_mm=600.0,
                concrete="C100/115",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_predesign_run("L8100_B2", root, evaluated_rows=row_a)
            write_predesign_run("L6750_B2", root, evaluated_rows=row_b)
            kept = predesign_initial_design_path(root, "L6750_B2").read_text(
                encoding="utf-8"
            )
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                write_predesign_run(
                    "L8100_B2",
                    root,
                    evaluated_rows=row_b,
                )
            self.assertEqual(
                predesign_initial_design_path(root, "L6750_B2").read_text(
                    encoding="utf-8"
                ),
                kept,
            )


def _vector_counts(path: Path) -> Counter:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Counter(tuple(vector) for vector in payload["vectors"])


class TestDebPredesignShelf(unittest.TestCase):
    """Both rankings are published from one evaluation."""

    def test_penalty_and_deb_orders_disagree_on_a_tiny_violation(self) -> None:
        from _mains.Testing._paths import PARAM_DEFAULTS_CSV
        from core.ioh_core.import_specs import load_param_defaults
        from slab_construction.slabs.hp_slab.analysis import _apply_active_penalties

        defaults = load_param_defaults(PARAM_DEFAULTS_CSV)
        alpha = float(
            defaults["opt_penalty_alpha"]["values"][
                defaults["opt_penalty_alpha"]["fixed_idx"]
            ]
        )
        beta = float(
            defaults["opt_penalty_beta"]["values"][
                defaults["opt_penalty_beta"]["fixed_idx"]
            ]
        )

        names = _active_names()
        constraints = {name: {"active": True} for name in names}
        values = {name: 0.5 for name in names}
        values[names[0]] = 1.0001
        (
            _sanitized,
            violations,
            _penalties,
            _penalty_sum,
            _factor,
            y_p,
            _remapped,
            analysis_ok,
            _error,
        ) = _apply_active_penalties(
            y=100.0,
            constraint_values=values,
            constraints=constraints,
            invalid_utilization=99.0,
            opt_penalty_alpha=alpha,
            opt_penalty_beta=beta,
        )
        self.assertTrue(analysis_ok)
        self.assertLess(y_p, 110.0)
        infeasible = _row(
            candidate_index=0,
            predesign_y_p=y_p,
            height_mm=100.0,
            concrete="C25/30",
            y=100.0,
            feasible=False,
            violations=violations,
        )
        feasible = _row(
            candidate_index=1,
            predesign_y_p=110.0,
            height_mm=200.0,
            concrete="C50/60",
            y=110.0,
            feasible=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_predesign_run(
                "L8100_B2",
                root,
                evaluated_rows=[infeasible, feasible],
            )
            legacy = json.loads(
                predesign_initial_design_path(root, "L8100_B2").read_text(
                    encoding="utf-8"
                )
            )
            deb = json.loads(
                predesign_initial_design_deb_path(root, "L8100_B2").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                _vector_counts(predesign_initial_design_path(root, "L8100_B2")),
                _vector_counts(predesign_initial_design_deb_path(root, "L8100_B2")),
            )
            self.assertEqual(legacy["vectors"][0], deb["vectors"][1])
            self.assertNotEqual(legacy["vectors"][0], deb["vectors"][0])
            with predesign_results_path(root, "L8100_B2").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                table = list(csv.DictReader(handle))
            by_rank = {int(row["rank"]): int(row["candidate_index"]) for row in table}
            by_deb = {
                int(row["deb_rank"]): int(row["candidate_index"]) for row in table
            }
            self.assertEqual(by_rank[1], 0)
            self.assertEqual(by_deb[1], 1)
            from slab_construction.slabs.hp_slab.predesign import (
                predesign_prefix_report,
            )

            report = predesign_prefix_report(
                predesign_results_path(root, "L8100_B2"),
                (1,),
            )
            self.assertEqual(report[0]["legacy_feasible"], 0)
            self.assertEqual(report[0]["deb_feasible"], 1)
            self.assertEqual(report[0]["candidate_index_overlap"], 0)
            text = predesign_results_path(root, "L8100_B2").read_text(
                encoding="utf-8"
            )
            header = next(csv.reader(io.StringIO(text)))
            self.assertEqual(header[-2:], ["predesign_phi", "deb_rank"])
            self.assertIn("rank", header)
            ranks = [
                row["deb_rank"]
                for row in csv.DictReader(io.StringIO(text))
            ]
            self.assertEqual(sorted(ranks), ["1", "2"])

    def test_equal_phi_keeps_candidate_order_and_unique_deb_ranks(self) -> None:
        names = _active_names()
        shared = {names[0]: 0.2}
        later = _row(
            candidate_index=4,
            predesign_y_p=50.0,
            height_mm=300.0,
            concrete="C100/115",
            y=1.0,
            feasible=False,
            violations=shared,
        )
        earlier = _row(
            candidate_index=1,
            predesign_y_p=50.0,
            height_mm=200.0,
            concrete="C50/60",
            y=9.0,
            feasible=False,
            violations=shared,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_predesign_run(
                "L8100_B2",
                root,
                evaluated_rows=[later, earlier],
            )
            table = list(
                csv.DictReader(
                    predesign_results_path(root, "L8100_B2").open(
                        encoding="utf-8",
                        newline="",
                    )
                )
            )
            self.assertEqual(
                sorted(row["deb_rank"] for row in table),
                ["1", "2"],
            )
            deb = json.loads(
                predesign_initial_design_deb_path(root, "L8100_B2").read_text(
                    encoding="utf-8"
                )
            )
            legacy = json.loads(
                predesign_initial_design_path(root, "L8100_B2").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                deb["vectors"],
                [legacy["vectors"][1], legacy["vectors"][0]],
            )

    def test_failed_analysis_ranks_after_a_finite_violation(self) -> None:
        names = _active_names()
        infeasible = _row(
            candidate_index=2,
            predesign_y_p=80.0,
            height_mm=200.0,
            concrete="C50/60",
            y=40.0,
            feasible=False,
            violations={names[0]: 0.5},
        )
        failed = _row(
            candidate_index=0,
            predesign_y_p=1.0,
            height_mm=100.0,
            concrete="C25/30",
            y=math.inf,
            feasible=False,
            analysis_ok=False,
            violations={},
        )
        failed["violations_"] = {}
        failed["constraint_values"] = {"ignored": "garbage"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_predesign_run(
                "L8100_B2",
                root,
                evaluated_rows=[failed, infeasible],
            )
            deb = json.loads(
                predesign_initial_design_deb_path(root, "L8100_B2").read_text(
                    encoding="utf-8"
                )
            )
            legacy = json.loads(
                predesign_initial_design_path(root, "L8100_B2").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(deb["vectors"][0], legacy["vectors"][1])

    def test_one_sided_shelf_is_rebuilt_from_rows(self) -> None:
        rows = [
            _row(
                candidate_index=0,
                predesign_y_p=2.0,
                height_mm=100.0,
                concrete="C25/30",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fresh = Path(tmp) / "fresh"
            write_predesign_run("L8100_B2", root, evaluated_rows=rows)
            predesign_initial_design_deb_path(root, "L8100_B2").unlink()
            write_predesign_run("L8100_B2", fresh, evaluated_rows=rows)
            write_predesign_run("L8100_B2", root, evaluated_rows=rows)
            self.assertEqual(
                predesign_initial_design_path(root, "L8100_B2").read_text(
                    encoding="utf-8"
                ),
                predesign_initial_design_path(fresh, "L8100_B2").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertEqual(
                predesign_initial_design_deb_path(root, "L8100_B2").read_text(
                    encoding="utf-8"
                ),
                predesign_initial_design_deb_path(fresh, "L8100_B2").read_text(
                    encoding="utf-8"
                ),
            )
            predesign_initial_design_path(root, "L8100_B2").unlink()
            write_predesign_run("L8100_B2", root, evaluated_rows=rows)
            self.assertEqual(
                predesign_initial_design_path(root, "L8100_B2").read_text(
                    encoding="utf-8"
                ),
                predesign_initial_design_path(fresh, "L8100_B2").read_text(
                    encoding="utf-8"
                ),
            )

    def test_malformed_record_leaves_the_shelf_unchanged(self) -> None:
        good = [
            _row(
                candidate_index=0,
                predesign_y_p=1.0,
                height_mm=100.0,
                concrete="C25/30",
            )
        ]
        broken = [
            _row(
                candidate_index=1,
                predesign_y_p=1.0,
                height_mm=200.0,
                concrete="C50/60",
                y=math.nan,
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_predesign_run("L8100_B2", root, evaluated_rows=good)
            before = [
                path.read_bytes()
                for path in (
                    predesign_results_path(root, "L8100_B2"),
                    predesign_initial_design_path(root, "L8100_B2"),
                    predesign_initial_design_deb_path(root, "L8100_B2"),
                )
            ]
            with self.assertRaises(ValueError):
                write_predesign_run("L8100_B2", root, evaluated_rows=broken)
            after = [
                path.read_bytes()
                for path in (
                    predesign_results_path(root, "L8100_B2"),
                    predesign_initial_design_path(root, "L8100_B2"),
                    predesign_initial_design_deb_path(root, "L8100_B2"),
                )
            ]
            self.assertEqual(after, before)

    def test_second_file_failure_rolls_the_shelf_back(self) -> None:
        first = [
            _row(
                candidate_index=0,
                predesign_y_p=2.0,
                height_mm=100.0,
                concrete="C25/30",
            )
        ]
        second = [
            _row(
                candidate_index=3,
                predesign_y_p=0.5,
                height_mm=400.0,
                concrete="C100/115",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_predesign_run("L8100_B2", root, evaluated_rows=first)
            before = {
                path: path.read_bytes()
                for path in (
                    predesign_results_path(root, "L8100_B2"),
                    predesign_initial_design_path(root, "L8100_B2"),
                    predesign_initial_design_deb_path(root, "L8100_B2"),
                )
            }
            real_replace = os.replace
            calls = {"n": 0}

            def fail_second(src: str, dst: str) -> None:
                calls["n"] += 1
                if calls["n"] == 2:
                    raise OSError("disk full")
                real_replace(src, dst)

            with self.assertRaises(OSError):
                with patch(
                    "slab_construction.slabs.hp_slab.predesign.os.replace",
                    side_effect=fail_second,
                ):
                    write_predesign_run("L8100_B2", root, evaluated_rows=second)
            for path, payload in before.items():
                self.assertEqual(path.read_bytes(), payload)
            shelf = predesign_results_path(root, "L8100_B2").parent
            leftovers = [
                path.name
                for path in shelf.iterdir()
                if path.name.startswith(".")
            ]
            self.assertEqual(leftovers, [])

    def test_publish_failure_on_a_new_shelf_leaves_nothing(self) -> None:
        rows = [
            _row(
                candidate_index=0,
                predesign_y_p=1.0,
                height_mm=100.0,
                concrete="C25/30",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_replace = os.replace
            calls = {"n": 0}

            def fail_second(src: str, dst: str) -> None:
                calls["n"] += 1
                if calls["n"] == 2:
                    raise OSError("disk full")
                real_replace(src, dst)

            with self.assertRaises(OSError):
                with patch(
                    "slab_construction.slabs.hp_slab.predesign.os.replace",
                    side_effect=fail_second,
                ):
                    write_predesign_run("L8100_B2", root, evaluated_rows=rows)
            for path in (
                predesign_results_path(root, "L8100_B2"),
                predesign_initial_design_path(root, "L8100_B2"),
                predesign_initial_design_deb_path(root, "L8100_B2"),
            ):
                self.assertFalse(path.exists(), path.name)
            shelf = predesign_results_path(root, "L8100_B2").parent
            if shelf.exists():
                leftovers = [
                    path.name
                    for path in shelf.iterdir()
                    if path.name.startswith(".")
                ]
                self.assertEqual(leftovers, [])


class TestPredesignPrefixDiagnostics(unittest.TestCase):
    """Prefix overlap is by candidate index, not by a unique vector."""

    def test_reports_feasible_counts_and_index_overlap(self) -> None:
        from slab_construction.slabs.hp_slab.predesign import (
            predesign_prefix_diagnostics,
        )

        legacy = [
            {"candidate_index": 0, "feasible": False},
            {"candidate_index": 1, "feasible": True},
            {"candidate_index": 2, "feasible": True},
        ]
        deb = [
            {"candidate_index": 1, "feasible": True},
            {"candidate_index": 2, "feasible": True},
            {"candidate_index": 0, "feasible": False},
        ]
        report = predesign_prefix_diagnostics(legacy, deb, (1, 2))
        self.assertEqual(report[0]["legacy_feasible"], 0)
        self.assertEqual(report[0]["deb_feasible"], 1)
        self.assertEqual(report[0]["candidate_index_overlap"], 0)
        self.assertEqual(report[1]["candidate_index_overlap"], 1)
        self.assertEqual(report[1]["deb_feasible"], 2)
