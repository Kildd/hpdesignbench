"""
Phase A4 gate: CSV snapshots, hashes, and selective snapshot problem builds.
"""
from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.ioh_core.import_specs import (
    load_constraint_defaults,
    load_param_defaults,
    load_problems_combined,
)
from core.ioh_core.problem_builder import (
    build_problem_from_snapshot,
    build_problems_for_slab,
    build_problems_for_slab_type,
)
from core.ioh_core.workflows.optimization_runs.snapshot import (
    SNAPSHOT_CSV_NAMES,
    SnapshotError,
    resolve_snapshot_slab_dir,
    sha256_file,
    snapshot_slab_inputs,
    validate_snapshot_hashes,
)
from _mains.Testing._paths import (
    CONSTRAINT_DEFAULTS_CSV,
    HP_SLAB_DIR,
    PARAM_DEFAULTS_CSV,
    PROBLEM_LIST_CSV,
)


def _fixed_params(bundle: dict) -> dict:
    """
    Extract fixed parameter values from a problem bundle model.

    Parameters
    ----------
    bundle : dict
        Problem bundle from :func:`build_problems_for_slab`.

    Returns
    -------
    dict
        Mapping of fixed parameter name to catalog value.
    """
    out = {}
    for name, meta in bundle["model"].items():
        if meta.get("role") == "fixed":
            out[name] = meta["values"][meta["fixed_idx"]]
    return out


def _midpoint(bundle: dict) -> list[int]:
    """
    Return the integer midpoint index vector for a problem bundle.

    Parameters
    ----------
    bundle : dict
        Problem bundle with ``lb_idx`` and ``ub_idx``.

    Returns
    -------
    list of int
        Midpoint candidate indices.
    """
    return [
        lo + (hi - lo) // 2
        for lo, hi in zip(bundle["lb_idx"], bundle["ub_idx"])
    ]


class TestSlabInputSnapshot(unittest.TestCase):
    """Snapshot file set, hashes, and post-edit isolation."""

    def test_snapshot_has_exactly_four_files_and_matching_hashes(self) -> None:
        with TemporaryDirectory() as tmp:
            inputs_dir = Path(tmp) / "inputs"
            hashes = snapshot_slab_inputs(slab_type="hp_slab", inputs_dir=inputs_dir)
            slab_dir = resolve_snapshot_slab_dir(inputs_dir, "hp_slab")

            self.assertEqual(len(hashes), 4)
            self.assertEqual(
                set(hashes),
                {f"hp_slab/{name}" for name in SNAPSHOT_CSV_NAMES},
            )
            present = sorted(p.name for p in slab_dir.iterdir() if p.is_file())
            self.assertEqual(present, sorted(SNAPSHOT_CSV_NAMES))
            for relative, digest in hashes.items():
                self.assertEqual(sha256_file(inputs_dir / relative), digest)
            validate_snapshot_hashes(inputs_dir=inputs_dir, expected_hashes=hashes)

    def test_source_edits_after_snapshot_do_not_change_snapshot(self) -> None:
        with TemporaryDirectory() as tmp:
            inputs_dir = Path(tmp) / "inputs"
            hashes = snapshot_slab_inputs(slab_type="hp_slab", inputs_dir=inputs_dir)
            before = build_problem_from_snapshot(
                slab_type="hp_slab",
                problem_id="L6750_B2",
                snapshot_dir=resolve_snapshot_slab_dir(inputs_dir, "hp_slab"),
            )

            # Mutate a disposable copy of the packaged CSV, not the repo file.
            # The snapshot must remain sealed to its original digests/content.
            scratch = Path(tmp) / "live_edit"
            scratch.mkdir()
            for name in SNAPSHOT_CSV_NAMES:
                shutil.copy2(HP_SLAB_DIR / name, scratch / name)
            edited = scratch / "problem_list.csv"
            edited.write_text(
                edited.read_text(encoding="utf-8") + "\n# edited after snapshot\n",
                encoding="utf-8",
            )

            after = build_problem_from_snapshot(
                slab_type="hp_slab",
                problem_id="L6750_B2",
                snapshot_dir=resolve_snapshot_slab_dir(inputs_dir, "hp_slab"),
            )
            validate_snapshot_hashes(inputs_dir=inputs_dir, expected_hashes=hashes)
            self.assertEqual(before["var_names"], after["var_names"])
            self.assertEqual(before["lb_idx"], after["lb_idx"])
            self.assertEqual(before["ub_idx"], after["ub_idx"])
            self.assertEqual(before["label"], after["label"])
            self.assertNotEqual(
                sha256_file(edited),
                hashes["hp_slab/problem_list.csv"],
            )

    def test_hash_mismatch_is_detected(self) -> None:
        with TemporaryDirectory() as tmp:
            inputs_dir = Path(tmp) / "inputs"
            hashes = snapshot_slab_inputs(slab_type="hp_slab", inputs_dir=inputs_dir)
            target = inputs_dir / "hp_slab" / "materials.csv"
            target.write_bytes(target.read_bytes() + b"\n")
            with self.assertRaisesRegex(SnapshotError, "SHA-256 mismatch"):
                validate_snapshot_hashes(inputs_dir=inputs_dir, expected_hashes=hashes)


class TestLiveVersusSnapshotProblems(unittest.TestCase):
    """Live packaged CSVs and snapshotted CSVs must resolve identically."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = TemporaryDirectory()
        cls.inputs_dir = Path(cls.tmp.name) / "inputs"
        cls.hashes = snapshot_slab_inputs(slab_type="hp_slab", inputs_dir=cls.inputs_dir)
        cls.snapshot_dir = resolve_snapshot_slab_dir(cls.inputs_dir, "hp_slab")
        cls.live = build_problems_for_slab_type("hp_slab")
        cls.snap_one = build_problem_from_snapshot(
            slab_type="hp_slab",
            problem_id="L6750_B2",
            snapshot_dir=cls.snapshot_dir,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_a2b_resolution_matches_and_default_is_non_runnable(self) -> None:
        live_specs = load_problems_combined(
            load_param_defaults(PARAM_DEFAULTS_CSV),
            load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV),
            PROBLEM_LIST_CSV,
        )
        snap_specs = load_problems_combined(
            load_param_defaults(self.snapshot_dir / "parameter_defaults.csv"),
            load_constraint_defaults(self.snapshot_dir / "constraint_defaults.csv"),
            self.snapshot_dir / "problem_list.csv",
        )
        self.assertNotIn("default", live_specs)
        self.assertNotIn("default", snap_specs)
        self.assertEqual(set(live_specs), set(snap_specs))
        self.assertEqual(live_specs["L6750_B2"]["label"], snap_specs["L6750_B2"]["label"])
        self.assertEqual(
            live_specs["L6750_B2"]["constraints"].keys(),
            snap_specs["L6750_B2"]["constraints"].keys(),
        )

    def test_problem_1_space_and_label_match(self) -> None:
        live = self.live["L6750_B2"]
        snap = self.snap_one
        self.assertEqual(live["var_names"], snap["var_names"])
        self.assertEqual(live["lb_idx"], snap["lb_idx"])
        self.assertEqual(live["ub_idx"], snap["ub_idx"])
        self.assertEqual(live["label"], snap["label"])
        self.assertEqual(live["active_constraint_names"], snap["active_constraint_names"])

        mid_live = live["decode"](_midpoint(live))
        mid_snap = snap["decode"](_midpoint(snap))
        self.assertEqual(mid_live, mid_snap)

    def test_fixed_params_materials_and_analysis_match(self) -> None:
        live = self.live["L6750_B2"]
        snap = self.snap_one
        self.assertEqual(_fixed_params(live), _fixed_params(snap))
        self.assertEqual(live["active_constraint_names"], snap["active_constraint_names"])

        x = _midpoint(live)
        live_rec = live["ctx"].get(x)
        snap_rec = snap["ctx"].get(x)
        self.assertEqual(live_rec["analysis_ok"], snap_rec["analysis_ok"])
        self.assertAlmostEqual(float(live_rec["y_p"]), float(snap_rec["y_p"]))
        self.assertEqual(set(live_rec["violations_"]), set(snap_rec["violations_"]))
        for name in live_rec["violations_"]:
            self.assertAlmostEqual(
                float(live_rec["violations_"][name]),
                float(snap_rec["violations_"][name]),
            )

        # Material binding is exercised by a successful analysis path.
        self.assertTrue(live_rec["analysis_ok"])
        self.assertIn("mat_reinf_id", live["decode"](x))

    def test_unknown_problem_id_fails_clearly(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown problem_id"):
            build_problem_from_snapshot(
                slab_type="hp_slab",
                problem_id="999",
                snapshot_dir=self.snapshot_dir,
            )

    def test_problem_ids_filter_and_slab_type_api_unchanged(self) -> None:
        filtered = build_problems_for_slab(
            HP_SLAB_DIR,
            __import__(
                "slab_construction.slabs.hp_slab.analysis",
                fromlist=["analysis"],
            ),
            problem_ids=["L6750_B2"],
        )
        self.assertEqual(set(filtered), {"L6750_B2"})
        self.assertIn("analysis_fn", filtered["L6750_B2"])
        self.assertIn("model", filtered["L6750_B2"])

        # Existing public API still builds every ordinary problem.
        all_problems = build_problems_for_slab_type("hp_slab")
        self.assertTrue({"L6750_B2", "L8100_B2", "L9450_B2", "L10800_B2"} <= set(all_problems))
        self.assertEqual(len(all_problems), 21)
        self.assertNotIn("default", all_problems)


if __name__ == "__main__":
    unittest.main()
