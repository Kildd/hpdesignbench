"""
Phase L3 gate: sealed Arrow IPC structural artifact library storage.
"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from core.analysis_core.run_config import AnalysisRunConfig
from core.ioh_core.structural_artifacts import (
    LibraryIdentity,
    StructuralLibraryError,
    artifacts_equal,
    open_structural_library,
    run_config_fingerprint,
    seal_structural_library,
    structural_key_digest,
)
from core.ioh_core.structural_artifacts.fixtures import (
    make_build_failure_artifact,
    make_mk_cleared_artifact,
    make_success_artifact,
)
from core.ioh_core.structural_artifacts.library import SUCCESS_MARKER, part_filename
from core.ioh_core.structural_artifacts.provider import (
    LibraryStructuralProvider,
    build_structural_provider,
)


def _identity(*, cardinality: int = 4) -> LibraryIdentity:
    cfg = AnalysisRunConfig(invalid_utilization=999.0, slab_type="hp_slab")
    return LibraryIdentity(
        slab_type="hp_slab",
        problem_id="toy",
        snapshot_hash="snap-l3",
        run_config_fingerprint=run_config_fingerprint(cfg),
        structural_cardinality=cardinality,
        full_cardinality=cardinality * 4,
        finish_names=("finish_a", "finish_b"),
        finish_sizes=(2, 2),
        structural_names=("a", "b"),
        structural_sizes=(2, cardinality // 2 if cardinality >= 2 else 1),
        input_hashes=(("params", "abc"),),
    )


def _aligned_artifact(seed: int, *, snapshot_hash: str = "snap-l3"):
    if seed % 3 == 0:
        art = make_success_artifact(seed=seed)
    elif seed % 3 == 1:
        art = make_mk_cleared_artifact(seed=seed)
    else:
        art = make_build_failure_artifact(seed=seed)
    # Force snapshot to match sealed identity.
    return type(art)(
        key=art.key,
        snapshot_hash=snapshot_hash,
        n=art.n,
        m_u=art.m_u,
        m_cr=art.m_cr,
        mk_curves=art.mk_curves,
        mk_metadata=art.mk_metadata,
        mk_cleared=art.mk_cleared,
        build_failure_result=(
            dict(art.build_failure_result)
            if art.build_failure_result is not None
            else None
        ),
    )


def _rows(cardinality: int = 4):
    return [
        (i, (i % 2, i // 2), _aligned_artifact(10 + i))
        for i in range(cardinality)
    ]


class TestSealedLibraryRoundTrip(unittest.TestCase):
    def test_round_trip_first_middle_last(self) -> None:
        rows = _rows(4)
        identity = _identity(cardinality=4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "lib"
            manifest = seal_structural_library(
                root, rows, identity=identity, rows_per_shard=2
            )
            self.assertEqual(manifest.nrows, 4)
            self.assertTrue((root / SUCCESS_MARKER).is_file())
            self.assertTrue((root / "schema.json").is_file())
            self.assertTrue((root / "aggregate" / "summary.json").is_file())
            with open_structural_library(
                root, expected_identity=identity, lru_max_shards=1
            ) as lib:
                for sid in (0, 1, 3):
                    got = lib.get(sid)
                    self.assertTrue(artifacts_equal(rows[sid][2], got))
                # LRU kept only one shard cached after several lookups.
                self.assertLessEqual(len(lib._tables), 1)


class TestSealedLibraryValidation(unittest.TestCase):
    def test_rejects_missing_success(self) -> None:
        rows = _rows(2)
        identity = _identity(cardinality=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "lib"
            seal_structural_library(root, rows, identity=identity, rows_per_shard=2)
            (root / SUCCESS_MARKER).unlink()
            with self.assertRaises(StructuralLibraryError):
                open_structural_library(root)

    def test_rejects_identity_mismatch(self) -> None:
        rows = _rows(2)
        identity = _identity(cardinality=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "lib"
            seal_structural_library(root, rows, identity=identity, rows_per_shard=2)
            other = LibraryIdentity(
                slab_type="hp_slab",
                problem_id="other",
                snapshot_hash="snap-l3",
                run_config_fingerprint=identity.run_config_fingerprint,
                structural_cardinality=2,
                full_cardinality=8,
                finish_names=("finish_a", "finish_b"),
                finish_sizes=(2, 2),
                structural_names=("a", "b"),
                structural_sizes=(1, 2),
            )
            with self.assertRaises(StructuralLibraryError):
                open_structural_library(root, expected_identity=other)

    def test_snapshot_hash_mismatch_requires_explicit_override(self) -> None:
        rows = _rows(2)
        identity = _identity(cardinality=2)
        expected = replace(identity, snapshot_hash="different-snapshot")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "lib"
            seal_structural_library(root, rows, identity=identity, rows_per_shard=2)
            with self.assertRaisesRegex(StructuralLibraryError, "snapshot_hash"):
                open_structural_library(root, expected_identity=expected)
            with open_structural_library(
                root,
                expected_identity=expected,
                allow_snapshot_hash_mismatch=True,
            ) as library:
                self.assertEqual(library.identity.snapshot_hash, "snap-l3")

    def test_rejects_wrong_key_and_snapshot(self) -> None:
        rows = _rows(2)
        identity = _identity(cardinality=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "lib"
            seal_structural_library(root, rows, identity=identity, rows_per_shard=2)
            with open_structural_library(root) as lib:
                with self.assertRaises(StructuralLibraryError):
                    lib.get(0, expected_snapshot_hash="wrong-snap")
                with self.assertRaises(StructuralLibraryError):
                    lib.get(0, expected_key_digest="0" * 64)
                # Matching digest succeeds.
                digest = structural_key_digest(rows[0][2].key)
                got = lib.get(0, expected_key_digest=digest)
                self.assertTrue(artifacts_equal(rows[0][2], got))

    def test_rejects_corrupt_truncated_part(self) -> None:
        rows = _rows(2)
        identity = _identity(cardinality=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "lib"
            seal_structural_library(root, rows, identity=identity, rows_per_shard=2)
            part = root / "parts" / part_filename(0)
            data = part.read_bytes()
            part.write_bytes(data[: max(16, len(data) // 3)])
            with open_structural_library(root) as lib:
                with self.assertRaises(StructuralLibraryError):
                    lib.get(0)

    def test_part_hash_verification(self) -> None:
        rows = _rows(2)
        identity = _identity(cardinality=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "lib"
            seal_structural_library(root, rows, identity=identity, rows_per_shard=2)
            # Flip one byte without breaking IPC enough to always fail open;
            # instead mutate manifest hash expectation by rewriting part after seal.
            part = root / "parts" / part_filename(0)
            # Append garbage then rely on hash check (verify_part_hashes=True).
            with open(part, "ab") as fh:
                fh.write(b"\x00\xff")
            with open_structural_library(root, verify_part_hashes=True) as lib:
                with self.assertRaises(StructuralLibraryError):
                    lib.get(0)

    def test_provider_uses_sealed_lookup(self) -> None:
        rows = _rows(2)
        identity = _identity(cardinality=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "lib"
            seal_structural_library(root, rows, identity=identity, rows_per_shard=2)
            with open_structural_library(root) as lib:
                provider = build_structural_provider("library", lookup=lib)
                self.assertIsInstance(provider, LibraryStructuralProvider)
                # Direct lookup path through protocol.
                art = provider.lookup.get(1)
                self.assertTrue(artifacts_equal(rows[1][2], art))


if __name__ == "__main__":
    unittest.main()
