"""
Shard specifications for structural-library builds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class StructuralLibraryModelError(ValueError):
    """
    Raised when a structural-library shard payload is invalid.
    """


@dataclass(frozen=True)
class StructuralLibraryShardSpec:
    """
    One contiguous structural-ID shard.

    Attributes
    ----------
    problem_id : str
        Ordinary problem ID.
    slab_type : str
        Slab package name (``hp_slab`` only in L4).
    shard_index : int
        Zero-based part ordinal.
    id_start : int
        Inclusive structural ID.
    id_stop : int
        Exclusive structural ID.
    rows_per_shard : int
        Configured shard length (last shard may be shorter).
    snapshot_hash : str
        Combined input-snapshot digest.
    run_config_fingerprint : str
        Analysis run-config fingerprint.
    library_structural_cardinality : int
        Number of structural IDs sealed in this library build.
    full_structural_cardinality : int
        Full structural subspace cardinality for the problem.
    full_candidate_cardinality : int
        Full candidate catalogue cardinality.
    """

    problem_id: str
    slab_type: str
    shard_index: int
    id_start: int
    id_stop: int
    rows_per_shard: int
    snapshot_hash: str
    run_config_fingerprint: str
    library_structural_cardinality: int
    full_structural_cardinality: int
    full_candidate_cardinality: int

    def __post_init__(self) -> None:
        if self.shard_index < 0:
            raise StructuralLibraryModelError(
                f"shard_index must be >= 0, got {self.shard_index}"
            )
        if self.id_start < 0 or self.id_stop <= self.id_start:
            raise StructuralLibraryModelError(
                f"invalid shard range [{self.id_start}, {self.id_stop})"
            )
        if self.rows_per_shard < 1:
            raise StructuralLibraryModelError(
                f"rows_per_shard must be >= 1, got {self.rows_per_shard}"
            )

    @property
    def nrows(self) -> int:
        return int(self.id_stop - self.id_start)

    def to_payload(self) -> dict[str, Any]:
        return {
            "problem_id": str(self.problem_id),
            "slab_type": str(self.slab_type),
            "shard_index": int(self.shard_index),
            "id_start": int(self.id_start),
            "id_stop": int(self.id_stop),
            "rows_per_shard": int(self.rows_per_shard),
            "snapshot_hash": str(self.snapshot_hash),
            "run_config_fingerprint": str(self.run_config_fingerprint),
            "library_structural_cardinality": int(self.library_structural_cardinality),
            "full_structural_cardinality": int(self.full_structural_cardinality),
            "full_candidate_cardinality": int(self.full_candidate_cardinality),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralLibraryShardSpec:
        return cls(
            problem_id=str(payload["problem_id"]),
            slab_type=str(payload["slab_type"]),
            shard_index=int(payload["shard_index"]),
            id_start=int(payload["id_start"]),
            id_stop=int(payload["id_stop"]),
            rows_per_shard=int(payload["rows_per_shard"]),
            snapshot_hash=str(payload["snapshot_hash"]),
            run_config_fingerprint=str(payload["run_config_fingerprint"]),
            library_structural_cardinality=int(
                payload["library_structural_cardinality"]
            ),
            full_structural_cardinality=int(payload["full_structural_cardinality"]),
            full_candidate_cardinality=int(payload["full_candidate_cardinality"]),
        )
