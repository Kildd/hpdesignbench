"""
Identity and manifest models for sealed structural-artifact libraries.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.analysis_core.run_config import AnalysisRunConfig
from core.ioh_core.structural_artifacts.codec import (
    CODEC_VERSION,
    SCHEMA_VERSION,
    dumps_tagged_json,
)

FORMAT_ARROW_IPC = "arrow_ipc"
KEY_ALGORITHM_VERSION = 1
DEFAULT_ROWS_PER_SHARD = 4096
MANIFEST_VERSION = 1


class StructuralLibraryError(ValueError):
    """
    Raised when a sealed structural library cannot be written, opened, or read.
    """


def run_config_fingerprint(run_config: AnalysisRunConfig) -> str:
    """
    Return a stable SHA-256 hex fingerprint of analysis run settings.
    """
    payload = dumps_tagged_json(
        (
            ("design_code", str(run_config.design_code)),
            ("invalid_utilization", float(run_config.invalid_utilization)),
            ("slab_type", str(run_config.slab_type)),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    """
    Return the SHA-256 hex digest of a file's bytes.
    """
    digest = hashlib.sha256()
    with open(Path(path), "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LibraryIdentity:
    """
    Sealed scientific identity for one structural artifact library.

    ``problem_id`` is the build provenance id. ``compatible_problem_ids`` lists
    consumer problem ids this library may serve (must include ``problem_id``).
    Older manifests without the field default to ``(problem_id,)``.
    """

    slab_type: str
    problem_id: str
    snapshot_hash: str
    run_config_fingerprint: str
    structural_cardinality: int
    full_cardinality: int
    finish_names: tuple[str, ...]
    finish_sizes: tuple[int, ...]
    structural_names: tuple[str, ...]
    structural_sizes: tuple[int, ...]
    input_hashes: tuple[tuple[str, str], ...] = ()
    compatible_problem_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        primary = str(self.problem_id)
        raw = self.compatible_problem_ids
        if not raw:
            object.__setattr__(self, "compatible_problem_ids", (primary,))
            return
        normalized = tuple(dict.fromkeys(str(item) for item in raw))
        if primary not in normalized:
            normalized = (primary, *normalized)
        object.__setattr__(self, "compatible_problem_ids", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slab_type": str(self.slab_type),
            "problem_id": str(self.problem_id),
            "compatible_problem_ids": [str(v) for v in self.compatible_problem_ids],
            "snapshot_hash": str(self.snapshot_hash),
            "run_config_fingerprint": str(self.run_config_fingerprint),
            "structural_cardinality": int(self.structural_cardinality),
            "full_cardinality": int(self.full_cardinality),
            "finish_names": list(self.finish_names),
            "finish_sizes": [int(v) for v in self.finish_sizes],
            "structural_names": list(self.structural_names),
            "structural_sizes": [int(v) for v in self.structural_sizes],
            "input_hashes": {
                str(key): str(value) for key, value in self.input_hashes
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LibraryIdentity:
        input_hashes_raw = payload.get("input_hashes") or {}
        if not isinstance(input_hashes_raw, Mapping):
            raise StructuralLibraryError("input_hashes must be a mapping")
        problem_id = str(payload["problem_id"])
        compatible_raw = payload.get("compatible_problem_ids")
        if compatible_raw is None:
            compatible: tuple[str, ...] = (problem_id,)
        else:
            if not isinstance(compatible_raw, Sequence) or isinstance(
                compatible_raw, (str, bytes)
            ):
                raise StructuralLibraryError(
                    "compatible_problem_ids must be a list of strings"
                )
            compatible = tuple(str(item) for item in compatible_raw)
        return cls(
            slab_type=str(payload["slab_type"]),
            problem_id=problem_id,
            snapshot_hash=str(payload["snapshot_hash"]),
            run_config_fingerprint=str(payload["run_config_fingerprint"]),
            structural_cardinality=int(payload["structural_cardinality"]),
            full_cardinality=int(payload["full_cardinality"]),
            finish_names=tuple(str(v) for v in payload["finish_names"]),
            finish_sizes=tuple(int(v) for v in payload["finish_sizes"]),
            structural_names=tuple(str(v) for v in payload["structural_names"]),
            structural_sizes=tuple(int(v) for v in payload["structural_sizes"]),
            input_hashes=tuple(
                (str(k), str(input_hashes_raw[k]))
                for k in sorted(input_hashes_raw.keys(), key=str)
            ),
            compatible_problem_ids=compatible,
        )


def assert_problem_id_compatible(
    stored: LibraryIdentity,
    expected_problem_id: str,
) -> None:
    """
    Require ``expected_problem_id`` to be listed on ``stored.compatible_problem_ids``.
    """
    token = str(expected_problem_id)
    allowed = tuple(str(item) for item in stored.compatible_problem_ids)
    if token not in allowed:
        raise StructuralLibraryError(
            "library identity mismatch on problem_id: "
            f"stored compatible_problem_ids={list(allowed)!r}, "
            f"expected={token!r} (build provenance problem_id={stored.problem_id!r})"
        )


@dataclass(frozen=True)
class PartInfo:
    """
    One sealed Arrow IPC part covering a contiguous structural-ID range.
    """

    shard_index: int
    id_start: int
    id_stop: int
    nrows: int
    bytes: int
    sha256: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_index": int(self.shard_index),
            "id_start": int(self.id_start),
            "id_stop": int(self.id_stop),
            "nrows": int(self.nrows),
            "bytes": int(self.bytes),
            "sha256": str(self.sha256),
            "path": str(self.path).replace("\\", "/"),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PartInfo:
        return cls(
            shard_index=int(payload["shard_index"]),
            id_start=int(payload["id_start"]),
            id_stop=int(payload["id_stop"]),
            nrows=int(payload["nrows"]),
            bytes=int(payload["bytes"]),
            sha256=str(payload["sha256"]),
            path=str(payload["path"]).replace("\\", "/"),
        )


@dataclass(frozen=True)
class LibraryManifest:
    """
    Top-level sealed-library manifest published before ``_SUCCESS``.
    """

    identity: LibraryIdentity
    rows_per_shard: int
    parts: tuple[PartInfo, ...]
    nrows: int
    format: str = FORMAT_ARROW_IPC
    manifest_version: int = MANIFEST_VERSION
    schema_version: int = SCHEMA_VERSION
    codec_version: int = CODEC_VERSION
    key_algorithm_version: int = KEY_ALGORITHM_VERSION

    @property
    def id_start(self) -> int:
        return 0

    @property
    def id_stop(self) -> int:
        return int(self.identity.structural_cardinality)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": int(self.manifest_version),
            "format": str(self.format),
            "schema_version": int(self.schema_version),
            "codec_version": int(self.codec_version),
            "key_algorithm_version": int(self.key_algorithm_version),
            "rows_per_shard": int(self.rows_per_shard),
            "id_start": int(self.id_start),
            "id_stop": int(self.id_stop),
            "nrows": int(self.nrows),
            "identity": self.identity.to_dict(),
            "parts": [part.to_dict() for part in self.parts],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LibraryManifest:
        parts_raw = payload.get("parts")
        if not isinstance(parts_raw, Sequence):
            raise StructuralLibraryError("manifest parts must be a sequence")
        identity = LibraryIdentity.from_dict(payload["identity"])
        manifest = cls(
            identity=identity,
            rows_per_shard=int(payload["rows_per_shard"]),
            parts=tuple(PartInfo.from_dict(item) for item in parts_raw),
            nrows=int(payload["nrows"]),
            format=str(payload.get("format", FORMAT_ARROW_IPC)),
            manifest_version=int(payload.get("manifest_version", MANIFEST_VERSION)),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            codec_version=int(payload.get("codec_version", CODEC_VERSION)),
            key_algorithm_version=int(
                payload.get("key_algorithm_version", KEY_ALGORITHM_VERSION)
            ),
        )
        manifest.validate_structure()
        return manifest

    def validate_structure(self) -> None:
        """
        Fail closed on inconsistent manifest geometry.
        """
        if self.format != FORMAT_ARROW_IPC:
            raise StructuralLibraryError(
                f"unsupported library format {self.format!r}; "
                f"expected {FORMAT_ARROW_IPC!r}"
            )
        if self.manifest_version != MANIFEST_VERSION:
            raise StructuralLibraryError(
                f"unsupported manifest_version {self.manifest_version}"
            )
        if self.schema_version != SCHEMA_VERSION or self.codec_version != CODEC_VERSION:
            raise StructuralLibraryError(
                "unsupported schema/codec version "
                f"{self.schema_version}/{self.codec_version}; "
                f"expected {SCHEMA_VERSION}/{CODEC_VERSION}"
            )
        if self.key_algorithm_version != KEY_ALGORITHM_VERSION:
            raise StructuralLibraryError(
                f"unsupported key_algorithm_version {self.key_algorithm_version}"
            )
        if self.rows_per_shard <= 0:
            raise StructuralLibraryError("rows_per_shard must be > 0")
        cardinality = int(self.identity.structural_cardinality)
        if self.nrows != cardinality:
            raise StructuralLibraryError(
                f"manifest nrows {self.nrows} != structural_cardinality {cardinality}"
            )
        if self.id_stop != cardinality:
            raise StructuralLibraryError(
                f"manifest id_stop {self.id_stop} != structural_cardinality {cardinality}"
            )
        if not self.parts:
            raise StructuralLibraryError("manifest has no parts")
        expected_shards = (cardinality + self.rows_per_shard - 1) // self.rows_per_shard
        if len(self.parts) != expected_shards:
            raise StructuralLibraryError(
                f"expected {expected_shards} parts, got {len(self.parts)}"
            )
        covered = 0
        for index, part in enumerate(self.parts):
            if part.shard_index != index:
                raise StructuralLibraryError(
                    f"part order broken at index {index}: shard_index={part.shard_index}"
                )
            expected_start = index * self.rows_per_shard
            expected_stop = min(cardinality, expected_start + self.rows_per_shard)
            if part.id_start != expected_start or part.id_stop != expected_stop:
                raise StructuralLibraryError(
                    f"part {index} range [{part.id_start}, {part.id_stop}) != "
                    f"expected [{expected_start}, {expected_stop})"
                )
            if part.nrows != expected_stop - expected_start:
                raise StructuralLibraryError(
                    f"part {index} nrows {part.nrows} != "
                    f"{expected_stop - expected_start}"
                )
            covered += part.nrows
        if covered != cardinality:
            raise StructuralLibraryError(
                f"parts cover {covered} rows, expected {cardinality}"
            )
