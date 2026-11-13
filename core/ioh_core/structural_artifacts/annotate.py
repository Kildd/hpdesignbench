"""
Update ``compatible_problem_ids`` on an existing sealed library manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from core.execution_core.atomic_io import write_json_atomic
from core.ioh_core.structural_artifacts.library import SUCCESS_MARKER
from core.ioh_core.structural_artifacts.models import (
    LibraryIdentity,
    LibraryManifest,
    StructuralLibraryError,
)


def set_compatible_problem_ids(
    library_root: Path,
    problem_ids: Sequence[str],
) -> LibraryIdentity:
    """
    Rewrite ``identity.compatible_problem_ids`` on a sealed library.

    Leaves Arrow parts untouched. Requires ``_SUCCESS`` and a readable
    ``manifest.json``. The build provenance ``problem_id`` is always kept in
    the compatible set.
    """
    root = Path(library_root).expanduser().resolve()
    marker = root / SUCCESS_MARKER
    manifest_path = root / "manifest.json"
    if not marker.is_file():
        raise StructuralLibraryError(
            f"not a sealed library root (missing {SUCCESS_MARKER}): {root}"
        )
    if not manifest_path.is_file():
        raise StructuralLibraryError(f"manifest.json not found: {manifest_path}")

    tokens = [str(item).strip() for item in problem_ids if str(item).strip()]
    if not tokens:
        raise StructuralLibraryError("compatible_problem_ids must be non-empty")

    payload = LibraryManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    identity = payload.identity
    primary = str(identity.problem_id)
    compatible = tuple(dict.fromkeys([primary, *tokens]))
    updated = LibraryIdentity(
        slab_type=identity.slab_type,
        problem_id=identity.problem_id,
        snapshot_hash=identity.snapshot_hash,
        run_config_fingerprint=identity.run_config_fingerprint,
        structural_cardinality=identity.structural_cardinality,
        full_cardinality=identity.full_cardinality,
        finish_names=identity.finish_names,
        finish_sizes=identity.finish_sizes,
        structural_names=identity.structural_names,
        structural_sizes=identity.structural_sizes,
        input_hashes=identity.input_hashes,
        compatible_problem_ids=compatible,
    )
    new_manifest = LibraryManifest(
        identity=updated,
        rows_per_shard=payload.rows_per_shard,
        parts=payload.parts,
        nrows=payload.nrows,
        format=payload.format,
        manifest_version=payload.manifest_version,
        schema_version=payload.schema_version,
        codec_version=payload.codec_version,
        key_algorithm_version=payload.key_algorithm_version,
    )
    write_json_atomic(manifest_path, new_manifest.to_dict())
    return updated
