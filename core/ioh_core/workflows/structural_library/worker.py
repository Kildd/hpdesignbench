"""
Single-shard structural-library worker for prepared experiments.

``python -m core.ioh_core.workflows.structural_library.worker
    --experiment-dir PATH --task-index N --attempt N``
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

from core.execution_core.artifacts import (
    read_task_status,
    touch_attempt_heartbeat,
    transition_task_status,
    write_attempt_owner,
)
from core.execution_core.atomic_io import write_json_atomic
from core.execution_core.lifecycle import load_prepared_config
from core.execution_core.manifest import load_manifest
from core.execution_core.models import ExperimentPaths, TaskState, format_task_id
from core.ioh_core.problem_builder import build_problem_from_snapshot
from core.ioh_core.structural_artifacts import (
    StructuralArtifact,
    make_structural_key,
)
from core.ioh_core.structural_artifacts.indexing import (
    structural_id_to_structural_indices,
    structural_variable_names,
)
from core.ioh_core.structural_artifacts.artifact import finish_variable_names
from core.ioh_core.structural_artifacts.codec import encode_record
from core.ioh_core.structural_artifacts.library import (
    part_filename,
    publish_arrow_file,
    write_arrow_part_file,
)
from core.ioh_core.io_util import _req_param
from core.unit_core import kN_to_N
from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
    validate_snapshot_hashes,
)
from core.ioh_core.workflows.structural_library.models import (
    StructuralLibraryShardSpec,
)
from core.ioh_core.workflows.structural_library.planner import (
    STRUCTURAL_LIBRARY_TASK_KIND,
)
from core.ioh_core.workflows.structural_library.validation import library_directory


class WorkerError(RuntimeError):
    """
    Raised for invalid worker inputs or ownership checks.
    """


def experiment_paths(experiment_dir: Path) -> ExperimentPaths:
    root = Path(experiment_dir).expanduser().resolve()
    return ExperimentPaths(
        root=root,
        inputs=root / "inputs",
        submissions=root / "submissions",
        tasks=root / "tasks",
        aggregate=root / "aggregate",
    )


def combined_snapshot_hash(input_files: Mapping[str, str]) -> str:
    payload = "|".join(
        f"{path}:{digest}" for path, digest in sorted(input_files.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def index_space_from_bundle(bundle: Mapping[str, Any]) -> IndexSpace:
    return IndexSpace.from_bounds(
        tuple(str(name) for name in bundle["var_names"]),
        tuple(int(v) for v in bundle["lb_idx"]),
        tuple(int(v) for v in bundle["ub_idx"]),
    )


def _capture_one(
    *,
    structural_id: int,
    space: IndexSpace,
    decode,
    analysis_fn,
    run_config,
    finish_names: tuple[str, ...],
    snapshot_hash: str,
    structural_names: tuple[str, ...],
) -> tuple[tuple[int, ...], StructuralArtifact]:
    struct_idx = structural_id_to_structural_indices(structural_id, space)
    full_idx = list(struct_idx) + [0, 0]
    params = decode(full_idx)
    n = kN_to_N(_req_param(params, "loads_N_kN"))
    key = make_structural_key(
        params,
        finish_names=finish_names,
        snapshot_hash=snapshot_hash,
        n=float(n),
        run_config=run_config,
        structural_names=structural_names,
    )
    holder: list[StructuralArtifact] = []
    analysis_fn(
        dict(params),
        run_config=run_config,
        structural_artifact_holder=holder,
        structural_capture_key=key,
        structural_snapshot_hash=snapshot_hash,
    )
    if not holder:
        raise WorkerError(
            f"analysis did not capture artifact for structural_id={structural_id}"
        )
    return struct_idx, holder[0]


def build_shard_rows(
    *,
    spec: StructuralLibraryShardSpec,
    space: IndexSpace,
    bundle: Mapping[str, Any],
    snapshot_hash: str,
) -> list[dict[str, Any]]:
    from core.analysis_core.run_config import analysis_run_config_from_params

    decode = bundle["decode"]
    analysis_fn = bundle["analysis_fn"]
    params0 = decode(list([0] * len(space.var_names)))
    run_config = analysis_run_config_from_params(
        params0, slab_type=spec.slab_type
    )
    finish_names = finish_variable_names(space.var_names)
    structural_names = structural_variable_names(space)
    rows: list[dict[str, Any]] = []
    for sid in range(spec.id_start, spec.id_stop):
        indices, artifact = _capture_one(
            structural_id=sid,
            space=space,
            decode=decode,
            analysis_fn=analysis_fn,
            run_config=run_config,
            finish_names=finish_names,
            snapshot_hash=snapshot_hash,
            structural_names=structural_names,
        )
        rows.append(
            encode_record(
                artifact,
                structural_id=sid,
                structural_indices=indices,
            )
        )
    return rows


def run_worker(
    *,
    experiment_dir: Path,
    task_index: int,
    attempt: int,
) -> dict[str, Any]:
    paths = experiment_paths(experiment_dir)
    manifest = load_manifest(paths.manifest)
    if manifest.workflow_type != "structural_library":
        raise WorkerError(
            f"worker requires workflow_type='structural_library', "
            f"got {manifest.workflow_type!r}"
        )
    if task_index < 0 or task_index >= len(manifest.tasks):
        raise WorkerError(
            f"task_index {task_index} out of range for {len(manifest.tasks)} tasks"
        )
    task = manifest.tasks[task_index]
    if task.kind != STRUCTURAL_LIBRARY_TASK_KIND:
        raise WorkerError(
            f"task {task_index} has kind {task.kind!r}, "
            f"expected {STRUCTURAL_LIBRARY_TASK_KIND!r}"
        )

    validate_snapshot_hashes(
        inputs_dir=paths.inputs,
        expected_hashes=manifest.input_files,
    )

    task_dir = paths.tasks / format_task_id(task_index)
    attempt_dir = task_dir / "attempts" / f"{attempt:03d}"
    if not attempt_dir.is_dir():
        raise WorkerError(f"attempt directory missing: {attempt_dir}")

    status = read_task_status(task_dir)
    current = TaskState(status["state"])
    if current is TaskState.PENDING:
        transition_task_status(
            task_dir,
            expected={TaskState.PENDING},
            new_state=TaskState.DISPATCHED,
            attempt=attempt,
        )
        current = TaskState.DISPATCHED
    if current is not TaskState.DISPATCHED:
        raise WorkerError(
            f"task {task_index} must be dispatched before running, got {current.value!r}"
        )
    transition_task_status(
        task_dir,
        expected={TaskState.DISPATCHED},
        new_state=TaskState.RUNNING,
        attempt=attempt,
    )
    write_attempt_owner(attempt_dir, backend="local", pid=os.getpid())
    touch_attempt_heartbeat(attempt_dir)

    config = load_prepared_config(paths.root)
    from core.ioh_core.structural_artifacts.sealing import (
        revalidate_structural_artifacts_seal,
    )

    revalidate_structural_artifacts_seal(paths.root)
    if config.structural_library is None:
        raise WorkerError("prepared config is missing [structural_library]")

    spec = StructuralLibraryShardSpec.from_payload(task.payload)
    snapshot_dir = resolve_snapshot_slab_dir(paths.inputs, spec.slab_type)
    bundle = build_problem_from_snapshot(
        slab_type=spec.slab_type,
        problem_id=spec.problem_id,
        snapshot_dir=snapshot_dir,
    )
    space = index_space_from_bundle(bundle)
    snapshot_hash = combined_snapshot_hash(manifest.input_files)

    lib_root = library_directory(paths.root)
    parts_dir = lib_root / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    destination = parts_dir / part_filename(spec.shard_index)

    if destination.is_file() and destination.stat().st_size > 0:
        # Adopt existing part (interrupted recovery).
        summary = {
            "status": "adopted",
            "shard_index": spec.shard_index,
            "id_start": spec.id_start,
            "id_stop": spec.id_stop,
            "part": str(destination),
            "nrows": spec.nrows,
        }
        write_json_atomic(attempt_dir / "summary.json", summary)
        transition_task_status(
            task_dir,
            expected={TaskState.RUNNING},
            new_state=TaskState.COMPLETED,
            attempt=attempt,
        )
        return summary

    staging = attempt_dir / part_filename(spec.shard_index)
    try:
        rows = build_shard_rows(
            spec=spec,
            space=space,
            bundle=bundle,
            snapshot_hash=snapshot_hash,
        )
        write_arrow_part_file(staging, rows)
        publish_arrow_file(staging, destination)
        summary = {
            "status": "built",
            "shard_index": spec.shard_index,
            "id_start": spec.id_start,
            "id_stop": spec.id_stop,
            "part": str(destination),
            "nrows": len(rows),
            "snapshot_hash": snapshot_hash,
        }
        write_json_atomic(attempt_dir / "summary.json", summary)
        transition_task_status(
            task_dir,
            expected={TaskState.RUNNING},
            new_state=TaskState.COMPLETED,
            attempt=attempt,
        )
        return summary
    except Exception as exc:
        write_json_atomic(
            attempt_dir / "failure.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        transition_task_status(
            task_dir,
            expected={TaskState.RUNNING},
            new_state=TaskState.FAILED,
            attempt=attempt,
        )
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run_worker(
            experiment_dir=args.experiment_dir,
            task_index=args.task_index,
            attempt=args.attempt,
        )
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
