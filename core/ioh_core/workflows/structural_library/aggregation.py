"""
Collect and seal a structural-library experiment (``_SUCCESS`` last).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.analysis_core.run_config import analysis_run_config_from_params
from core.execution_core.lifecycle import load_prepared_config
from core.execution_core.manifest import load_manifest
from core.execution_core.models import ExperimentPaths
from core.ioh_core.import_specs import make_decode
from core.ioh_core.structural_artifacts.library import finalize_sealed_library
from core.ioh_core.workflows.brute_force.planner import (
    load_problem_space_and_schema_from_csv_dir,
)
from core.ioh_core.workflows.optimization_runs.snapshot import resolve_snapshot_slab_dir
from core.ioh_core.workflows.structural_library.identity import build_library_identity
from core.ioh_core.workflows.structural_library.validation import (
    StructuralLibraryValidationError,
    library_directory,
    validate_library_or_raise,
)


class StructuralLibraryAggregationError(RuntimeError):
    """
    Raised when structural-library collection cannot complete.
    """


def aggregate_structural_library_experiment(experiment_dir: Path) -> dict[str, Any]:
    """
    Validate published parts, write sealed metadata, then ``_SUCCESS``.
    """
    root = Path(experiment_dir)
    paths = ExperimentPaths(
        root=root,
        inputs=root / "inputs",
        submissions=root / "submissions",
        tasks=root / "tasks",
        aggregate=root / "aggregate",
    )
    try:
        specs = validate_library_or_raise(root)
    except StructuralLibraryValidationError as exc:
        raise StructuralLibraryAggregationError(str(exc)) from exc

    manifest = load_manifest(paths.manifest)
    first = specs[0]
    slab_dir = resolve_snapshot_slab_dir(paths.inputs, first.slab_type)
    space, _schema, info = load_problem_space_and_schema_from_csv_dir(
        slab_dir=slab_dir,
        problem_id=first.problem_id,
    )
    decode = make_decode(info["model"], space.var_names)
    params0 = decode(list([0] * len(space.var_names)))
    run_config = analysis_run_config_from_params(
        params0, slab_type=first.slab_type
    )
    from core.ioh_core.workflows.structural_library.worker import combined_snapshot_hash

    snapshot_hash = combined_snapshot_hash(manifest.input_files)
    identity = build_library_identity(
        slab_type=first.slab_type,
        problem_id=first.problem_id,
        space=space,
        snapshot_hash=snapshot_hash,
        run_config=run_config,
        library_structural_cardinality=first.library_structural_cardinality,
        input_hashes=dict(manifest.input_files),
        compatible_problem_ids=(
            load_prepared_config(root).structural_library.compatible_problem_ids or None
        ),
    )
    if identity.run_config_fingerprint != first.run_config_fingerprint:
        raise StructuralLibraryAggregationError(
            "run_config_fingerprint mismatch between plan and collect: "
            f"{first.run_config_fingerprint!r} != "
            f"{identity.run_config_fingerprint!r}"
        )

    lib_root = library_directory(root)
    # Kind counts are optional; workers may leave them in attempt summaries later.
    sealed = finalize_sealed_library(
        lib_root,
        identity=identity,
        rows_per_shard=first.rows_per_shard,
        kind_counts={},
    )
    return {
        "workflow": "structural_library",
        "nrows": sealed.nrows,
        "n_parts": len(sealed.parts),
        "library_dir": str(lib_root),
        "identity": identity.to_dict(),
    }
