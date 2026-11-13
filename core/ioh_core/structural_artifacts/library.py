"""
Sealed random-access Arrow IPC structural artifact libraries.

Layout::

    library_root/
      manifest.json
      schema.json
      aggregate/summary.json
      parts/structural-XXXXXX.arrow
      _SUCCESS

Runtime parts are uncompressed Arrow IPC for memory mapping. Distribute as an
external compressed archive and unpack before use.
"""
from __future__ import annotations

import os
import shutil
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.execution_core.atomic_io import read_json, write_json_atomic, write_text_atomic
from core.ioh_core.structural_artifacts.artifact import StructuralArtifact
from core.ioh_core.structural_artifacts.codec import (
    StructuralCodecError,
    decode_record,
    encode_record,
    structural_key_digest,
)
from core.ioh_core.structural_artifacts.models import (
    DEFAULT_ROWS_PER_SHARD,
    FORMAT_ARROW_IPC,
    KEY_ALGORITHM_VERSION,
    LibraryIdentity,
    LibraryManifest,
    PartInfo,
    StructuralLibraryError,
    assert_problem_id_compatible,
    sha256_file,
)
from core.ioh_core.structural_artifacts.schema import (
    schema_metadata,
    structural_artifact_schema,
)
from core.ioh_core.workflows.brute_force.deps import import_pyarrow

SUCCESS_MARKER = "_SUCCESS"


def part_filename(shard_index: int) -> str:
    """
    Return ``structural-XXXXXX.arrow`` for ``shard_index``.
    """
    if shard_index < 0:
        raise StructuralLibraryError(f"shard_index must be >= 0, got {shard_index}")
    return f"structural-{int(shard_index):06d}.arrow"


def schema_document() -> dict[str, Any]:
    """
    Return the JSON schema document published beside the manifest.
    """
    schema = structural_artifact_schema()
    return {
        **schema_metadata(),
        "format": FORMAT_ARROW_IPC,
        "key_algorithm_version": KEY_ALGORITHM_VERSION,
        "columns": [
            {"name": field.name, "type": str(field.type)} for field in schema
        ],
    }


def _records_to_table(records: Sequence[Mapping[str, Any]]) -> Any:
    pa = import_pyarrow()
    schema = structural_artifact_schema()
    normalized: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        if row.get("build_failure_json") is None:
            row["build_failure_json"] = None
        normalized.append(row)
    return pa.Table.from_pylist(normalized, schema=schema)


def _unique_partial(destination: Path) -> Path:
    token = f"{os.getpid()}_{uuid.uuid4().hex}"
    return destination.with_name(f"{destination.name}.{token}.partial")


def _same_filesystem(path_a: Path, path_b: Path) -> bool:
    return os.stat(path_a).st_dev == os.stat(path_b).st_dev


def publish_arrow_file(
    source: Path,
    destination: Path,
    *,
    force_cross_filesystem_copy: bool = False,
) -> Path:
    """
    Atomically publish a closed Arrow IPC file to ``destination``.
    """
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise StructuralLibraryError(f"missing source Arrow file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    use_copy = force_cross_filesystem_copy
    if not use_copy:
        try:
            use_copy = not _same_filesystem(source, destination.parent)
        except OSError:
            use_copy = True

    if not use_copy:
        os.replace(source, destination)
        return destination.resolve()

    partial = _unique_partial(destination)
    try:
        shutil.copy2(source, partial)
        with open(partial, "rb") as fh:
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(partial, destination)
        try:
            source.unlink()
        except OSError:
            pass
        return destination.resolve()
    finally:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass


def write_arrow_part_file(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """
    Write one Arrow IPC part (attempt-private or final) from logical rows.
    """
    pa = import_pyarrow()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _records_to_table(records)
    partial = _unique_partial(path)
    try:
        with pa.OSFile(str(partial), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
            try:
                # OSFile may not expose fileno; best-effort via reopen.
                with open(partial, "rb") as fh:
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
            except OSError:
                pass
        os.replace(partial, path)
    finally:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass


def quarantine_part(part_path: Path, *, reason: str) -> Path:
    """
    Move an invalid published part into ``library/quarantine/``.
    """
    part_path = Path(part_path)
    quarantine_dir = part_path.parent.parent / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    safe_reason = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in reason)[
        :64
    ]
    destination = (
        quarantine_dir / f"{part_path.name}.{safe_reason}.{os.getpid()}.bad"
    )
    os.replace(part_path, destination)
    return destination


def write_success_marker(library_root: Path) -> Path:
    """
    Atomically write ``_SUCCESS`` as the final publication action.
    """
    path = Path(library_root) / SUCCESS_MARKER
    write_text_atomic(path, "ok\n")
    return path


def finalize_sealed_library(
    root: Path,
    *,
    identity: LibraryIdentity,
    rows_per_shard: int,
    kind_counts: Mapping[str, int] | None = None,
) -> LibraryManifest:
    """
    Validate published Arrow parts and write manifest/schema/summary/``_SUCCESS``.

    Expects ``parts/structural-XXXXXX.arrow`` covering ``[0, structural_cardinality)``.
    """
    root = Path(root)
    if (root / SUCCESS_MARKER).exists():
        raise StructuralLibraryError(
            f"refusing to finalize over existing _SUCCESS: {root}"
        )
    cardinality = int(identity.structural_cardinality)
    if rows_per_shard <= 0:
        raise StructuralLibraryError(f"rows_per_shard must be > 0, got {rows_per_shard}")
    n_shards = (cardinality + rows_per_shard - 1) // rows_per_shard
    part_infos: list[PartInfo] = []
    for shard_index in range(n_shards):
        id_start = shard_index * rows_per_shard
        id_stop = min(cardinality, id_start + rows_per_shard)
        rel_path = f"parts/{part_filename(shard_index)}"
        path = root / rel_path
        if not path.is_file():
            raise StructuralLibraryError(f"missing published part: {path}")
        digest = sha256_file(path)
        # Lightweight structural_id coverage check.
        pa = import_pyarrow()
        source = None
        try:
            source = pa.memory_map(str(path), "r")
            reader = pa.ipc.open_file(source)
            table = reader.read_all()
        except Exception as exc:  # noqa: BLE001
            raise StructuralLibraryError(
                f"unreadable Arrow part {path}: {exc}"
            ) from exc
        finally:
            if source is not None:
                close = getattr(source, "close", None)
                if callable(close):
                    try:
                        close()
                    except OSError:
                        pass
        if table.num_rows != id_stop - id_start:
            raise StructuralLibraryError(
                f"part {rel_path} nrows {table.num_rows} != {id_stop - id_start}"
            )
        ids = table.column("structural_id").to_pylist()
        if ids != list(range(id_start, id_stop)):
            raise StructuralLibraryError(
                f"part {rel_path} structural_id sequence does not match "
                f"[{id_start}, {id_stop})"
            )
        part_infos.append(
            PartInfo(
                shard_index=shard_index,
                id_start=id_start,
                id_stop=id_stop,
                nrows=id_stop - id_start,
                bytes=int(path.stat().st_size),
                sha256=digest,
                path=rel_path,
            )
        )

    manifest = LibraryManifest(
        identity=identity,
        rows_per_shard=int(rows_per_shard),
        parts=tuple(part_infos),
        nrows=cardinality,
    )
    manifest.validate_structure()
    (root / "aggregate").mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / "schema.json", schema_document())
    write_json_atomic(root / "manifest.json", manifest.to_dict())
    write_json_atomic(
        root / "aggregate" / "summary.json",
        {
            "nrows": cardinality,
            "n_parts": len(part_infos),
            "rows_per_shard": int(rows_per_shard),
            "kind_counts": dict(kind_counts or {}),
            "identity": identity.to_dict(),
        },
    )
    write_success_marker(root)
    return manifest


def seal_structural_library(
    root: Path,
    artifacts: Sequence[tuple[int, Sequence[int], StructuralArtifact]],
    *,
    identity: LibraryIdentity,
    rows_per_shard: int = DEFAULT_ROWS_PER_SHARD,
) -> LibraryManifest:
    """
    Write a complete sealed library covering ``[0, structural_cardinality)``.

    Parameters
    ----------
    root : Path
        Library root directory.
    artifacts : sequence
        ``(structural_id, structural_indices, artifact)`` for every ID.
    identity : LibraryIdentity
        Sealed scientific identity.
    rows_per_shard : int, optional
        Contiguous IDs per Arrow part.

    Returns
    -------
    LibraryManifest
        Published manifest (``_SUCCESS`` is written last).
    """
    if rows_per_shard <= 0:
        raise StructuralLibraryError(f"rows_per_shard must be > 0, got {rows_per_shard}")
    root = Path(root)
    if (root / SUCCESS_MARKER).exists():
        raise StructuralLibraryError(
            f"refusing to overwrite sealed library with _SUCCESS: {root}"
        )

    by_id: dict[int, tuple[Sequence[int], StructuralArtifact]] = {}
    for structural_id, indices, artifact in artifacts:
        sid = int(structural_id)
        if sid in by_id:
            raise StructuralLibraryError(f"duplicate structural_id {sid}")
        by_id[sid] = (indices, artifact)

    cardinality = int(identity.structural_cardinality)
    expected_ids = list(range(cardinality))
    if sorted(by_id) != expected_ids:
        raise StructuralLibraryError(
            "sealed libraries require exact coverage of "
            f"[0, {cardinality}); got {len(by_id)} ids"
        )

    parts_dir = root / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    (root / "aggregate").mkdir(parents=True, exist_ok=True)

    kind_counts = {"success": 0, "mk_cleared": 0, "build_failure": 0}
    part_infos: list[PartInfo] = []
    n_shards = (cardinality + rows_per_shard - 1) // rows_per_shard
    for shard_index in range(n_shards):
        id_start = shard_index * rows_per_shard
        id_stop = min(cardinality, id_start + rows_per_shard)
        rows = []
        for sid in range(id_start, id_stop):
            indices, artifact = by_id[sid]
            if artifact.is_build_failure:
                kind_counts["build_failure"] += 1
            elif artifact.mk_cleared:
                kind_counts["mk_cleared"] += 1
            else:
                kind_counts["success"] += 1
            if str(artifact.snapshot_hash) != str(identity.snapshot_hash):
                raise StructuralLibraryError(
                    f"artifact snapshot_hash {artifact.snapshot_hash!r} != "
                    f"identity snapshot_hash {identity.snapshot_hash!r} "
                    f"(structural_id={sid})"
                )
            rows.append(
                encode_record(
                    artifact,
                    structural_id=sid,
                    structural_indices=indices,
                )
            )
        rel_path = f"parts/{part_filename(shard_index)}"
        dest = root / rel_path
        write_arrow_part_file(dest, rows)
        digest = sha256_file(dest)
        part_infos.append(
            PartInfo(
                shard_index=shard_index,
                id_start=id_start,
                id_stop=id_stop,
                nrows=id_stop - id_start,
                bytes=int(dest.stat().st_size),
                sha256=digest,
                path=rel_path,
            )
        )

    manifest = LibraryManifest(
        identity=identity,
        rows_per_shard=int(rows_per_shard),
        parts=tuple(part_infos),
        nrows=cardinality,
    )
    manifest.validate_structure()

    write_json_atomic(root / "schema.json", schema_document())
    write_json_atomic(root / "manifest.json", manifest.to_dict())
    write_json_atomic(
        root / "aggregate" / "summary.json",
        {
            "nrows": cardinality,
            "n_parts": len(part_infos),
            "rows_per_shard": int(rows_per_shard),
            "kind_counts": kind_counts,
            "identity": identity.to_dict(),
        },
    )
    write_success_marker(root)
    return manifest


class SealedStructuralLibrary:
    """
    Strict mmap reader for a sealed Arrow IPC structural library.
    """

    def __init__(
        self,
        root: Path,
        *,
        expected_identity: LibraryIdentity | None = None,
        allow_snapshot_hash_mismatch: bool = False,
        lru_max_shards: int = 8,
        verify_part_hashes: bool = False,
    ) -> None:
        self.root = Path(root)
        if lru_max_shards <= 0:
            raise StructuralLibraryError("lru_max_shards must be > 0")
        self.lru_max_shards = int(lru_max_shards)
        self.verify_part_hashes = bool(verify_part_hashes)

        success = self.root / SUCCESS_MARKER
        if not success.is_file():
            raise StructuralLibraryError(
                f"structural library is not sealed (missing {SUCCESS_MARKER}): "
                f"{self.root}"
            )
        manifest_path = self.root / "manifest.json"
        schema_path = self.root / "schema.json"
        if not manifest_path.is_file():
            raise StructuralLibraryError(f"missing manifest: {manifest_path}")
        if not schema_path.is_file():
            raise StructuralLibraryError(f"missing schema: {schema_path}")

        self.manifest = LibraryManifest.from_dict(read_json(manifest_path))
        schema_doc = read_json(schema_path)
        if int(schema_doc.get("schema_version", -1)) != self.manifest.schema_version:
            raise StructuralLibraryError("schema.json schema_version mismatch")
        if int(schema_doc.get("codec_version", -1)) != self.manifest.codec_version:
            raise StructuralLibraryError("schema.json codec_version mismatch")

        if expected_identity is not None:
            self._assert_identity_compatible(
                expected_identity,
                allow_snapshot_hash_mismatch=allow_snapshot_hash_mismatch,
            )

        self._pa = import_pyarrow()
        self._tables: OrderedDict[int, Any] = OrderedDict()
        self._mmaps: dict[int, Any] = {}
        self._part_by_shard = {part.shard_index: part for part in self.manifest.parts}

    @property
    def identity(self) -> LibraryIdentity:
        return self.manifest.identity

    @property
    def rows_per_shard(self) -> int:
        return int(self.manifest.rows_per_shard)

    def _assert_identity_compatible(
        self,
        expected: LibraryIdentity,
        *,
        allow_snapshot_hash_mismatch: bool = False,
    ) -> None:
        stored = self.manifest.identity
        assert_problem_id_compatible(stored, expected.problem_id)
        checks = (
            ("slab_type", stored.slab_type, expected.slab_type),
            (
                "run_config_fingerprint",
                stored.run_config_fingerprint,
                expected.run_config_fingerprint,
            ),
            (
                "structural_cardinality",
                stored.structural_cardinality,
                expected.structural_cardinality,
            ),
            ("full_cardinality", stored.full_cardinality, expected.full_cardinality),
            ("finish_names", stored.finish_names, expected.finish_names),
            ("finish_sizes", stored.finish_sizes, expected.finish_sizes),
            ("structural_names", stored.structural_names, expected.structural_names),
            ("structural_sizes", stored.structural_sizes, expected.structural_sizes),
        )
        if not allow_snapshot_hash_mismatch:
            checks = (
                ("snapshot_hash", stored.snapshot_hash, expected.snapshot_hash),
                *checks,
            )
        for name, left, right in checks:
            if left != right:
                raise StructuralLibraryError(
                    f"library identity mismatch on {name}: "
                    f"stored={left!r}, expected={right!r}"
                )

    def close(self) -> None:
        """
        Close cached memory maps deterministically.
        """
        self._tables.clear()
        for handle in list(self._mmaps.values()):
            close = getattr(handle, "close", None)
            if callable(close):
                try:
                    close()
                except OSError:
                    pass
        self._mmaps.clear()

    def __enter__(self) -> SealedStructuralLibrary:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _evict_if_needed(self) -> None:
        while len(self._tables) > self.lru_max_shards:
            shard_index, _table = self._tables.popitem(last=False)
            handle = self._mmaps.pop(shard_index, None)
            if handle is not None:
                close = getattr(handle, "close", None)
                if callable(close):
                    try:
                        close()
                    except OSError:
                        pass

    def _load_shard(self, shard_index: int) -> Any:
        if shard_index in self._tables:
            self._tables.move_to_end(shard_index)
            return self._tables[shard_index]
        part = self._part_by_shard.get(shard_index)
        if part is None:
            raise StructuralLibraryError(f"unknown shard_index {shard_index}")
        path = self.root / part.path
        if not path.is_file():
            raise StructuralLibraryError(f"missing Arrow part: {path}")
        if self.verify_part_hashes:
            digest = sha256_file(path)
            if digest != part.sha256:
                raise StructuralLibraryError(
                    f"part hash mismatch for {part.path}: "
                    f"stored={part.sha256}, computed={digest}"
                )
        try:
            source = self._pa.memory_map(str(path), "r")
            reader = self._pa.ipc.open_file(source)
            table = reader.read_all()
        except Exception as exc:  # noqa: BLE001 - fail closed on corrupt IPC
            raise StructuralLibraryError(
                f"failed to open Arrow part {path}: {exc}"
            ) from exc
        if table.num_rows != part.nrows:
            raise StructuralLibraryError(
                f"part {part.path} nrows {table.num_rows} != manifest {part.nrows}"
            )
        self._mmaps[shard_index] = source
        self._tables[shard_index] = table
        self._evict_if_needed()
        return table

    def get(
        self,
        structural_id: int,
        *,
        expected_key: tuple[Any, ...] | None = None,
        expected_key_digest: str | None = None,
        expected_snapshot_hash: str | None = None,
    ) -> StructuralArtifact:
        """
        Return the artifact for ``structural_id`` with strict validation.
        """
        if not isinstance(structural_id, int) or isinstance(structural_id, bool):
            raise StructuralLibraryError(
                f"structural_id must be an int, got {structural_id!r}"
            )
        sid = int(structural_id)
        if sid < self.manifest.id_start or sid >= self.manifest.id_stop:
            raise StructuralLibraryError(
                f"structural_id {sid} out of range "
                f"[{self.manifest.id_start}, {self.manifest.id_stop})"
            )
        shard_index = sid // self.rows_per_shard
        part = self._part_by_shard[shard_index]
        if not (part.id_start <= sid < part.id_stop):
            raise StructuralLibraryError(
                f"structural_id {sid} not in expected shard range "
                f"[{part.id_start}, {part.id_stop})"
            )
        local_index = sid - part.id_start
        table = self._load_shard(shard_index)
        if local_index < 0 or local_index >= table.num_rows:
            raise StructuralLibraryError(
                f"local row {local_index} out of range for shard {shard_index}"
            )
        try:
            row = table.slice(local_index, 1).to_pylist()[0]
            artifact = decode_record(row)
        except (StructuralCodecError, StructuralLibraryError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise StructuralLibraryError(
                f"failed to decode structural_id {sid}: {exc}"
            ) from exc

        stored_id = int(row["structural_id"])
        if stored_id != sid:
            raise StructuralLibraryError(
                f"stored structural_id {stored_id} != requested {sid}"
            )
        stored_digest = str(row["structural_key_digest"])
        if stored_digest != structural_key_digest(artifact.key):
            raise StructuralLibraryError(
                f"stored key digest disagrees with decoded key for id {sid}"
            )

        if expected_snapshot_hash is not None:
            if str(artifact.snapshot_hash) != str(expected_snapshot_hash):
                raise StructuralLibraryError(
                    "snapshot_hash mismatch: "
                    f"stored={artifact.snapshot_hash!r}, "
                    f"expected={expected_snapshot_hash!r}"
                )
        elif str(artifact.snapshot_hash) != str(self.identity.snapshot_hash):
            raise StructuralLibraryError(
                "artifact snapshot_hash disagrees with sealed identity: "
                f"stored={artifact.snapshot_hash!r}, "
                f"identity={self.identity.snapshot_hash!r}"
            )

        if expected_key is not None:
            expected_key_digest = structural_key_digest(expected_key)
        if expected_key_digest is not None:
            if stored_digest != str(expected_key_digest):
                raise StructuralLibraryError(
                    f"structural key digest mismatch for structural_id={sid}"
                )
        return artifact


def open_structural_library(
    root: Path,
    *,
    expected_identity: LibraryIdentity | None = None,
    allow_snapshot_hash_mismatch: bool = False,
    lru_max_shards: int = 8,
    verify_part_hashes: bool = False,
) -> SealedStructuralLibrary:
    """
    Open a sealed structural library for strict random-access lookups.
    """
    return SealedStructuralLibrary(
        root,
        expected_identity=expected_identity,
        allow_snapshot_hash_mismatch=allow_snapshot_hash_mismatch,
        lru_max_shards=lru_max_shards,
        verify_part_hashes=verify_part_hashes,
    )
