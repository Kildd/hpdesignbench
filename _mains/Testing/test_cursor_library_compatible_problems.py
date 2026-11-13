"""
Tests for multi-problem library identity and structural-relevant keys.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.analysis_core.run_config import AnalysisRunConfig
from core.ioh_core.structural_artifacts.annotate import set_compatible_problem_ids
from core.ioh_core.structural_artifacts.artifact import (
    make_structural_key,
    structural_keys_match,
)
from core.ioh_core.structural_artifacts.models import (
    LibraryIdentity,
    LibraryManifest,
    PartInfo,
    assert_problem_id_compatible,
)
from core.ioh_core.structural_artifacts.sealing import (
    StructuralArtifactsSealError,
    assert_library_compatible_with_experiment,
)


def _identity(**overrides) -> LibraryIdentity:
    base = dict(
        slab_type="hp_slab",
        problem_id="1",
        snapshot_hash="abc",
        run_config_fingerprint="def",
        structural_cardinality=4,
        full_cardinality=16,
        finish_names=("fa", "fb"),
        finish_sizes=(2, 2),
        structural_names=("s0", "s1"),
        structural_sizes=(2, 2),
    )
    base.update(overrides)
    return LibraryIdentity(**base)


class TestCompatibleProblemIds(unittest.TestCase):
    def test_from_dict_defaults_to_build_id(self) -> None:
        ident = LibraryIdentity.from_dict(
            {
                "slab_type": "hp_slab",
                "problem_id": "1",
                "snapshot_hash": "abc",
                "run_config_fingerprint": "def",
                "structural_cardinality": 4,
                "full_cardinality": 16,
                "finish_names": ["fa", "fb"],
                "finish_sizes": [2, 2],
                "structural_names": ["s0", "s1"],
                "structural_sizes": [2, 2],
            }
        )
        self.assertEqual(ident.compatible_problem_ids, ("1",))

    def test_assert_accepts_listed_consumer(self) -> None:
        stored = _identity(compatible_problem_ids=("1", "2", "3", "4"))
        expected = _identity(problem_id="2")
        assert_library_compatible_with_experiment(stored, expected)
        assert_problem_id_compatible(stored, "4")

    def test_assert_rejects_unlisted_consumer(self) -> None:
        stored = _identity(compatible_problem_ids=("1", "2"))
        expected = _identity(problem_id="3")
        with self.assertRaises(StructuralArtifactsSealError):
            assert_library_compatible_with_experiment(stored, expected)


class TestStructuralRelevantKeys(unittest.TestCase):
    def test_span_and_load_category_do_not_break_match(self) -> None:
        cfg = AnalysisRunConfig(invalid_utilization=999.0)
        structural_names = ("geom_h_ges_mm", "geom_t_mm")
        params_a = {
            "geom_h_ges_mm": 200.0,
            "geom_t_mm": 50.0,
            "geom_span_mm": 6750.0,
            "loads_category": "A",
            "geom_t_infill_mm": 0.0,
        }
        params_b = {
            "geom_h_ges_mm": 200.0,
            "geom_t_mm": 50.0,
            "geom_span_mm": 8100.0,
            "loads_category": "B",
            "geom_t_infill_mm": 40.0,
        }
        key_a = make_structural_key(
            params_a,
            finish_names=("geom_t_infill_mm",),
            snapshot_hash="snap",
            n=-1000.0,
            run_config=cfg,
            structural_names=structural_names,
        )
        # Legacy-style stored key that still embeds fixed params.
        key_legacy = make_structural_key(
            params_b,
            finish_names=("geom_t_infill_mm",),
            snapshot_hash="snap",
            n=-1000.0,
            run_config=cfg,
        )
        self.assertTrue(
            structural_keys_match(
                key_legacy,
                key_a,
                structural_names=structural_names,
            )
        )


class TestAnnotateCompatibleProblems(unittest.TestCase):
    def test_set_compatible_problem_ids_rewrites_manifest(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            (root / "_SUCCESS").write_text("ok\n", encoding="utf-8")
            identity = _identity(
                structural_cardinality=1,
                full_cardinality=4,
            )
            manifest = LibraryManifest(
                identity=identity,
                rows_per_shard=1,
                parts=(
                    PartInfo(
                        shard_index=0,
                        id_start=0,
                        id_stop=1,
                        nrows=1,
                        bytes=10,
                        sha256="0" * 64,
                        path="parts/structural-000000.arrow",
                    ),
                ),
                nrows=1,
            )
            (root / "manifest.json").write_text(
                json.dumps(manifest.to_dict()),
                encoding="utf-8",
            )
            updated = set_compatible_problem_ids(root, ["2", "3", "1"])
            self.assertEqual(updated.compatible_problem_ids, ("1", "2", "3"))
            reloaded = LibraryManifest.from_dict(
                json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            )
            self.assertEqual(
                reloaded.identity.compatible_problem_ids, ("1", "2", "3")
            )


if __name__ == "__main__":
    unittest.main()
