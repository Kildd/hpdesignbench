"""
Validate published structural-library parts against the sealed plan.
"""
from __future__ import annotations

import warnings
from pathlib import Path

from core.execution_core.manifest import load_manifest
from core.execution_core.models import ExperimentPaths
from core.ioh_core.structural_artifacts.library import part_filename
from core.ioh_core.structural_artifacts.models import sha256_file
from core.ioh_core.workflows.structural_library.models import (
    StructuralLibraryShardSpec,
)
from core.ioh_core.workflows.structural_library.planner import (
    STRUCTURAL_LIBRARY_TASK_KIND,
)


class StructuralLibraryValidationError(ValueError):
    """
    Raised when a structural library fails coverage / integrity checks.
    """


def library_directory(experiment_dir: Path) -> Path:
    """Return the sealed baseline library: ``{ROOT}/structural_library/baseline``."""
    from core.execution_core.output_layout import (
        default_structural_library_dir,
        layout_root_from_experiment_dir,
    )

    return default_structural_library_dir(
        layout_root_from_experiment_dir(experiment_dir)
    )


def consumer_library_payload_exists(library_dir: Path) -> bool:
    """Return True if ``library_dir`` already has a sealed or partial library."""
    root = Path(library_dir)
    if (root / "manifest.json").is_file() or (root / "_SUCCESS").is_file():
        return True
    parts = root / "parts"
    if not parts.is_dir():
        return False
    return any(path.is_file() and path.stat().st_size > 0 for path in parts.iterdir())


def warn_if_overwriting_consumer_library(library_dir: Path) -> None:
    """Warn when shards would replace an existing consumer library on this ROOT."""
    root = Path(library_dir)
    if not consumer_library_payload_exists(root):
        return
    warnings.warn(
        "existing structural library at "
        f"{root.resolve()} will be overwritten; "
        "this build publishes the baseline family at "
        "{ROOT}/structural_library/baseline.",
        RuntimeWarning,
        stacklevel=2,
    )


def validate_library_or_raise(experiment_dir: Path) -> tuple[StructuralLibraryShardSpec, ...]:
    """
    Require every planned shard part to exist with matching ID coverage.
    """
    root = Path(experiment_dir)
    paths = ExperimentPaths(
        root=root,
        inputs=root / "inputs",
        submissions=root / "submissions",
        tasks=root / "tasks",
        aggregate=root / "aggregate",
    )
    manifest = load_manifest(paths.manifest)
    if manifest.workflow_type != "structural_library":
        raise StructuralLibraryValidationError(
            f"expected workflow_type='structural_library', got "
            f"{manifest.workflow_type!r}"
        )
    lib_root = library_directory(root)
    parts_dir = lib_root / "parts"
    if not parts_dir.is_dir():
        raise StructuralLibraryValidationError(f"missing parts directory: {parts_dir}")

    specs: list[StructuralLibraryShardSpec] = []
    for task in manifest.tasks:
        if task.kind != STRUCTURAL_LIBRARY_TASK_KIND:
            raise StructuralLibraryValidationError(
                f"task {task.task_id} has unexpected kind {task.kind!r}"
            )
        spec = StructuralLibraryShardSpec.from_payload(task.payload)
        specs.append(spec)
        path = parts_dir / part_filename(spec.shard_index)
        if not path.is_file():
            raise StructuralLibraryValidationError(f"missing part: {path}")
        if path.stat().st_size < 1:
            raise StructuralLibraryValidationError(f"empty part: {path}")
        # Presence + non-empty is enough here; finalize_sealed_library checks IDs.
        _ = sha256_file(path)

    if not specs:
        raise StructuralLibraryValidationError("no structural_library shards planned")
    specs_sorted = tuple(sorted(specs, key=lambda item: item.shard_index))
    for index, spec in enumerate(specs_sorted):
        if spec.shard_index != index:
            raise StructuralLibraryValidationError(
                f"shard_index gap at {index}: got {spec.shard_index}"
            )
    if specs_sorted[0].id_start != 0:
        raise StructuralLibraryValidationError("first shard must start at structural_id 0")
    cursor = 0
    for spec in specs_sorted:
        if spec.id_start != cursor:
            raise StructuralLibraryValidationError(
                f"shard {spec.shard_index} starts at {spec.id_start}, expected {cursor}"
            )
        cursor = spec.id_stop
    expected = specs_sorted[0].library_structural_cardinality
    if cursor != expected:
        raise StructuralLibraryValidationError(
            f"parts cover [0, {cursor}), expected [0, {expected})"
        )
    return specs_sorted
