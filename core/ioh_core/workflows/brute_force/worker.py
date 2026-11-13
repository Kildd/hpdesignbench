"""
Single-shard brute-force worker for prepared experiments.

Invoked as:

``python -m core.ioh_core.workflows.brute_force.worker
    --experiment-dir PATH --task-index N --attempt N``

The worker never runs an optimiser, never calls legacy
:func:`~core.ioh_core.experiment.run_experiment`, never attaches an IOH
logger, and never appends to another shard file. Scientific
``analysis_error`` rows remain successful task completions when ``y``
and ``y_p`` are finite. Non-finite ``y`` / ``y_p`` and infrastructure
exceptions write ``failure.json`` and exit nonzero.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import sys
import time
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
from core.ioh_core.structural_artifacts.artifact import (
    get_structural_build_count,
    reset_structural_build_count,
)
from core.ioh_core.structural_artifacts.indexing import global_index_to_structural_id
from core.ioh_core.structural_artifacts.provider import StructuralEvaluationProvider
from core.ioh_core.structural_artifacts.sealing import open_structural_provider_session
from core.ioh_core.workflows.brute_force.indexing import linear_to_indices
from core.ioh_core.workflows.brute_force.models import (
    BruteForceShardSpec,
    IndexSpace,
)
from core.ioh_core.workflows.brute_force.parquet_writer import (
    ParquetShardWriter,
    shard_part_path,
)
from core.ioh_core.workflows.brute_force.planner import (
    BRUTE_FORCE_TASK_KIND,
    finish_group_size,
)
from core.ioh_core.workflows.brute_force.schema import (
    BruteForceSchema,
    build_brute_force_schema,
    flatten_evaluation_row,
)
from core.ioh_core.workflows.brute_force.structural import (
    evaluate_finish_group,
    finish_variable_names,
)
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
    validate_snapshot_hashes,
)


class WorkerError(RuntimeError):
    """
    Raised for invalid worker inputs or ownership checks.
    """


def _reject_nonfinite_objective(row: Mapping[str, Any]) -> None:
    """
    Fail closed if flattened ``y`` or ``y_p`` is NaN or ±Inf.
    """
    y = row["y"]
    y_p = row["y_p"]
    if isinstance(y, (int, float)) and isinstance(y_p, (int, float)):
        if math.isfinite(float(y)) and math.isfinite(float(y_p)):
            return
    raise WorkerError(
        "non-finite y or y_p "
        f"(global_index={row.get('global_index')}, "
        f"analysis_error_stage={row.get('analysis_error_stage')!r}, "
        f"y={y!r}, y_p={y_p!r})"
    )


def experiment_paths(experiment_dir: Path) -> ExperimentPaths:
    """
    Build the standard path bundle for an experiment root.

    Parameters
    ----------
    experiment_dir : Path
        Absolute or relative experiment directory.

    Returns
    -------
    ExperimentPaths
        Resolved layout paths under the experiment root.
    """
    root = Path(experiment_dir).expanduser().resolve()
    return ExperimentPaths(
        root=root,
        inputs=root / "inputs",
        submissions=root / "submissions",
        tasks=root / "tasks",
        aggregate=root / "aggregate",
    )


def dataset_directory(paths: ExperimentPaths) -> Path:
    """
    Return ``experiment/dataset`` (parts published under ``parts/``).

    Parameters
    ----------
    paths : ExperimentPaths
        Experiment path bundle.

    Returns
    -------
    Path
        Dataset root.
    """
    return paths.root / "dataset"


def task_directory(paths: ExperimentPaths, task_index: int) -> Path:
    """
    Return the on-disk task directory for ``task_index``.
    """
    return paths.tasks / format_task_id(task_index)


def attempt_directory(task_dir: Path, attempt: int) -> Path:
    """
    Return the reserved attempt directory path.
    """
    if attempt < 1:
        raise WorkerError(f"attempt must be >= 1, got {attempt}")
    return Path(task_dir) / "attempts" / f"{attempt:03d}"


def validate_attempt_ownership(task_dir: Path, attempt: int) -> Path:
    """
    Require that the reserved attempt directory exists for this worker.
    """
    attempt_dir = attempt_directory(task_dir, attempt)
    if not attempt_dir.is_dir():
        raise WorkerError(
            f"attempt directory missing for ownership check: {attempt_dir}"
        )
    return attempt_dir


def _json_safe(value: Any) -> Any:
    """
    Convert summary values to JSON-serialisable forms.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def combined_snapshot_hash(input_files: Mapping[str, str]) -> str:
    """
    Deterministic digest over the sealed input-file hash map.

    Parameters
    ----------
    input_files : mapping
        Relative path → SHA-256 digests from the manifest.

    Returns
    -------
    str
        Lowercase SHA-256 hex digest.
    """
    payload = "|".join(
        f"{path}:{digest}" for path, digest in sorted(input_files.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def index_space_from_bundle(bundle: Mapping[str, Any]) -> IndexSpace:
    """
    Build an :class:`IndexSpace` from a problem bundle's bounds.
    """
    var_names = tuple(str(name) for name in bundle["var_names"])
    lower = tuple(int(v) for v in bundle["lb_idx"])
    upper = tuple(int(v) for v in bundle["ub_idx"])
    return IndexSpace.from_bounds(var_names, lower, upper)


def schema_from_bundle(bundle: Mapping[str, Any], space: IndexSpace) -> BruteForceSchema:
    """
    Build the Parquet schema for a snapshotted problem bundle.
    """
    model = bundle["model"]
    constraint_names = tuple(str(name) for name in bundle["constraints"].keys())
    return build_brute_force_schema(
        space=space,
        model=model,
        constraint_names=constraint_names,
    )


def _cleanup_attempt_partials(attempt_dir: Path, shard_index: int) -> None:
    """
    Remove attempt-owned ``*.partial`` files for this shard only.
    """
    prefix = f"shard-{int(shard_index):06d}.parquet"
    for path in Path(attempt_dir).glob(f"{prefix}*"):
        if path.is_file() and (
            path.name.endswith(".partial") or ".partial." in path.name
        ):
            try:
                path.unlink()
            except OSError:
                pass


def quarantine_part(part_path: Path, *, reason: str) -> Path:
    """
    Move an invalid published part into ``dataset/quarantine/``.

    Parameters
    ----------
    part_path : Path
        Existing final part path.
    reason : str
        Short reason token embedded in the quarantine filename.

    Returns
    -------
    Path
        Destination quarantine path.
    """
    part_path = Path(part_path)
    quarantine_dir = part_path.parent.parent / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    safe_reason = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in reason)[
        :48
    ]
    destination = quarantine_dir / f"{part_path.name}.{safe_reason}.{os.getpid()}.bad"
    try:
        os.replace(part_path, destination)
    except OSError:
        # Windows may still hold a reader handle briefly; copy then delete.
        shutil.copy2(part_path, destination)
        try:
            part_path.unlink()
        except OSError:
            # Leave the corrupt original if unlink races; destination is enough.
            pass
    return destination


def validate_existing_part(
    part_path: Path,
    *,
    schema_info: BruteForceSchema,
    expected_rows: int,
    start_index: int,
    stop_index: int,
) -> None:
    """
    Validate a published part for adoption.

    Raises
    ------
    WorkerError
        If the part is unreadable or does not match the shard contract.
    """
    import importlib

    from core.ioh_core.workflows.brute_force.deps import import_pyarrow

    import_pyarrow()
    pq = importlib.import_module("pyarrow.parquet")
    pf = None
    try:
        try:
            pf = pq.ParquetFile(str(part_path))
        except Exception as exc:
            raise WorkerError(f"unreadable parquet part {part_path}: {exc}") from exc
        if pf.metadata is None or pf.metadata.num_rows != expected_rows:
            raise WorkerError(
                f"part {part_path} has "
                f"{None if pf.metadata is None else pf.metadata.num_rows} rows, "
                f"expected {expected_rows}"
            )
        if not pf.schema_arrow.equals(schema_info.schema, check_metadata=False):
            raise WorkerError(f"part {part_path} schema does not match sealed schema")
        table = pf.read(columns=["global_index"])
        indices = table.column("global_index").to_pylist()
        if indices != list(range(start_index, stop_index)):
            raise WorkerError(
                f"part {part_path} global_index sequence does not match "
                f"[{start_index}, {stop_index})"
            )
    finally:
        if pf is not None and hasattr(pf, "close"):
            try:
                pf.close()
            except Exception:
                pass


def try_adopt_existing_part(
    part_path: Path,
    *,
    schema_info: BruteForceSchema,
    expected_rows: int,
    start_index: int,
    stop_index: int,
) -> bool:
    """
    Adopt a valid final part, or quarantine an invalid one.

    Returns
    -------
    bool
        ``True`` when the existing part was adopted (skip recomputation).
    """
    part_path = Path(part_path)
    if not part_path.is_file():
        return False
    try:
        validate_existing_part(
            part_path,
            schema_info=schema_info,
            expected_rows=expected_rows,
            start_index=start_index,
            stop_index=stop_index,
        )
        return True
    except WorkerError as exc:
        try:
            quarantine_part(part_path, reason=type(exc).__name__)
        except OSError:
            # If quarantine loses the Windows file-lock race, delete when possible
            # so recomputation can publish a fresh part.
            try:
                part_path.unlink()
            except OSError:
                pass
        return False


def evaluate_shard_rows(
    *,
    bundle: Mapping[str, Any],
    space: IndexSpace,
    schema_info: BruteForceSchema,
    spec: BruteForceShardSpec,
    snapshot_hash: str,
    reuse_structural: bool,
    provider: StructuralEvaluationProvider | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate one shard range into flattened Parquet rows (in-memory helper).

    Parameters
    ----------
    bundle : mapping
        Problem bundle from :func:`build_problem_from_snapshot`.
    space : IndexSpace
        Candidate index space.
    schema_info : BruteForceSchema
        Target schema.
    spec : BruteForceShardSpec
        Shard range.
    snapshot_hash : str
        Combined input snapshot digest.
    reuse_structural : bool
        Enable finish-group structural-artifact reuse.
    provider : StructuralEvaluationProvider or None, optional
        Structural backend. Library mode looks up one artifact per finish group.

    Returns
    -------
    list of dict
        Flattened rows in exact global-index order.
    """
    group = finish_group_size(space)
    if spec.start_index % group != 0 or spec.stop_index % group != 0:
        raise WorkerError(
            f"shard range [{spec.start_index}, {spec.stop_index}) is not aligned "
            f"to finish-group size {group}"
        )
    if spec.start_index < 0 or spec.stop_index > space.cardinality:
        raise WorkerError(
            f"shard range [{spec.start_index}, {spec.stop_index}) outside "
            f"cardinality {space.cardinality}"
        )
    if spec.schema_hash != schema_info.schema_hash:
        raise WorkerError(
            "shard schema_hash does not match rebuilt snapshot schema: "
            f"{spec.schema_hash!r} != {schema_info.schema_hash!r}"
        )

    decode = bundle["decode"]
    analysis_fn = bundle["analysis_fn"]
    run_config = bundle.get("run_config")
    finish_names = finish_variable_names(space.var_names)
    rows: list[dict[str, Any]] = []

    for group_start in range(spec.start_index, spec.stop_index, group):
        group_stop = group_start + group
        catalogue_vectors = [
            linear_to_indices(linear, space)
            for linear in range(group_start, group_stop)
        ]
        param_dicts = [decode(list(indices)) for indices in catalogue_vectors]
        started = time.perf_counter()
        structural_id = (
            global_index_to_structural_id(group_start, space)
            if provider is not None and provider.mode == "library"
            else None
        )
        results = evaluate_finish_group(
            param_dicts,
            constraints=bundle["constraints"],
            materials={},
            analysis_fn=analysis_fn,
            finish_names=finish_names,
            snapshot_hash=snapshot_hash,
            run_config=run_config,
            reuse=reuse_structural,
            provider=provider,
            structural_id=structural_id,
        )
        # Per-row wall time: share group wall evenly for streaming metadata.
        group_elapsed = time.perf_counter() - started
        per_row = group_elapsed / float(group)
        for offset, (indices, result) in enumerate(
            zip(catalogue_vectors, results)
        ):
            global_index = group_start + offset
            row = flatten_evaluation_row(
                schema_info=schema_info,
                global_index=global_index,
                shard_index=spec.shard_index,
                problem_id=spec.problem_id,
                evaluation_time_s=per_row,
                catalogue_indices=indices,
                result=result,
            )
            _reject_nonfinite_objective(row)
            rows.append(row)
    return rows


def run_worker(
    *,
    experiment_dir: Path,
    task_index: int,
    attempt: int,
    reuse_structural: bool | None = None,
) -> dict[str, Any]:
    """
    Execute one prepared brute-force shard attempt.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root containing ``manifest.json`` and ``inputs/``.
    task_index : int
        Zero-based task index from the manifest.
    attempt : int
        One-based reserved attempt index.
    reuse_structural : bool or None, optional
        When ``None`` (default), use
        ``brute_force.reuse_structural_artifacts`` from the sealed config.
        Explicit ``True`` / ``False`` overrides the sealed setting (tests /
        CLI ``--no-structural-reuse``).

    Returns
    -------
    dict
        Written summary document.
    """
    paths = experiment_paths(experiment_dir)
    manifest = load_manifest(paths.manifest)
    if manifest.workflow_type != "brute_force":
        raise WorkerError(
            f"worker requires workflow_type='brute_force', "
            f"got {manifest.workflow_type!r}"
        )
    if task_index < 0 or task_index >= len(manifest.tasks):
        raise WorkerError(
            f"task_index {task_index} out of range for "
            f"{len(manifest.tasks)} manifest tasks"
        )
    task = manifest.tasks[task_index]
    if task.kind != BRUTE_FORCE_TASK_KIND:
        raise WorkerError(
            f"task {task_index} has kind {task.kind!r}, "
            f"expected {BRUTE_FORCE_TASK_KIND!r}"
        )

    validate_snapshot_hashes(
        inputs_dir=paths.inputs,
        expected_hashes=manifest.input_files,
    )

    task_dir = task_directory(paths, task_index)
    attempt_dir = validate_attempt_ownership(task_dir, attempt)

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
            f"task {task_index} must be dispatched before running, "
            f"got {current.value!r}"
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
    if config.brute_force is None:
        raise WorkerError("prepared config is missing [brute_force]")
    if reuse_structural is None:
        reuse_structural = bool(config.brute_force.reuse_structural_artifacts)
    else:
        reuse_structural = bool(reuse_structural)

    spec = BruteForceShardSpec.from_payload(task.payload)
    snapshot_dir = resolve_snapshot_slab_dir(paths.inputs, spec.slab_type)
    bundle = build_problem_from_snapshot(
        slab_type=spec.slab_type,
        problem_id=spec.problem_id,
        snapshot_dir=snapshot_dir,
    )
    space = index_space_from_bundle(bundle)
    schema_info = schema_from_bundle(bundle, space)
    snapshot_hash = combined_snapshot_hash(manifest.input_files)

    if space.cardinality != IndexSpace.from_bounds(
        space.var_names, space.lower_bounds, space.upper_bounds
    ).cardinality:
        raise WorkerError("index-space cardinality inconsistency")
    group = finish_group_size(space)
    if space.var_names[-2:] != finish_variable_names(space.var_names):
        raise WorkerError("finish-group ordering mismatch")

    expected_rows = spec.stop_index - spec.start_index
    dataset_dir = dataset_directory(paths)
    parts_dir = dataset_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    destination = shard_part_path(dataset_dir, spec.shard_index)

    _cleanup_attempt_partials(attempt_dir, spec.shard_index)

    adopted = try_adopt_existing_part(
        destination,
        schema_info=schema_info,
        expected_rows=expected_rows,
        start_index=spec.start_index,
        stop_index=spec.stop_index,
    )

    analysis_ok_count = 0
    analysis_error_count = 0
    feasible_count = 0
    published_path = destination.resolve() if destination.is_file() else None
    structural_mode = "compute"
    library_hits = 0
    library_build_failure_fanouts = 0
    structural_builds = 0

    with open_structural_provider_session(paths.root) as (seal, provider):
        structural_mode = seal.mode
        if seal.mode == "library" and not reuse_structural:
            raise WorkerError(
                "structural_artifacts.mode='library' requires "
                "reuse_structural_artifacts=true"
            )
        reset_structural_build_count()

        if not adopted:
            staging = attempt_dir / f"shard-{spec.shard_index:06d}.parquet.partial"
            if staging.exists():
                try:
                    staging.unlink()
                except OSError:
                    pass
            writer: ParquetShardWriter | None = None
            try:
                writer = ParquetShardWriter(
                    schema_info=schema_info,
                    staging_path=staging,
                    destination_path=destination,
                    batch_rows=config.brute_force.parquet_batch_rows,
                    compression=config.brute_force.compression,
                )
                # Stream by finish group to keep buffering bounded.
                decode = bundle["decode"]
                analysis_fn = bundle["analysis_fn"]
                run_config = bundle.get("run_config")
                finish_names = finish_variable_names(space.var_names)
                if spec.schema_hash != schema_info.schema_hash:
                    raise WorkerError(
                        "shard schema_hash does not match rebuilt snapshot schema: "
                        f"{spec.schema_hash!r} != {schema_info.schema_hash!r}"
                    )
                if spec.start_index % group != 0 or spec.stop_index % group != 0:
                    raise WorkerError(
                        f"shard range [{spec.start_index}, {spec.stop_index}) is not "
                        f"aligned to finish-group size {group}"
                    )

                for group_start in range(spec.start_index, spec.stop_index, group):
                    touch_attempt_heartbeat(attempt_dir)
                    group_stop = group_start + group
                    catalogue_vectors = [
                        linear_to_indices(linear, space)
                        for linear in range(group_start, group_stop)
                    ]
                    param_dicts = [
                        decode(list(indices)) for indices in catalogue_vectors
                    ]
                    started = time.perf_counter()
                    structural_id = (
                        global_index_to_structural_id(group_start, space)
                        if provider.mode == "library"
                        else None
                    )
                    results = evaluate_finish_group(
                        param_dicts,
                        constraints=bundle["constraints"],
                        materials={},
                        analysis_fn=analysis_fn,
                        finish_names=finish_names,
                        snapshot_hash=snapshot_hash,
                        run_config=run_config,
                        reuse=reuse_structural,
                        provider=provider,
                        structural_id=structural_id,
                    )
                    group_elapsed = time.perf_counter() - started
                    per_row = group_elapsed / float(group)
                    for offset, (indices, result) in enumerate(
                        zip(catalogue_vectors, results)
                    ):
                        if result.get("analysis_ok"):
                            analysis_ok_count += 1
                        else:
                            analysis_error_count += 1
                        if result.get("feasible"):
                            feasible_count += 1
                        row = flatten_evaluation_row(
                            schema_info=schema_info,
                            global_index=group_start + offset,
                            shard_index=spec.shard_index,
                            problem_id=spec.problem_id,
                            evaluation_time_s=per_row,
                            catalogue_indices=indices,
                            result=result,
                        )
                        _reject_nonfinite_objective(row)
                        writer.write_row(row)
                writer.close()
                published_path = writer.publish()
            except Exception:
                if writer is not None:
                    writer.abandon()
                raise
        else:
            # Summarise adopted part without reloading full scientific columns.
            analysis_ok_count = -1
            analysis_error_count = -1
            feasible_count = -1

        library_hits = int(provider.metrics.library_hits)
        library_build_failure_fanouts = int(
            provider.metrics.library_build_failure_fanouts
        )
        structural_builds = int(get_structural_build_count())
        if seal.mode == "library" and structural_builds != 0:
            raise WorkerError(
                "library mode performed unexpected structural builds: "
                f"{structural_builds}"
            )

    summary = {
        "task_index": task_index,
        "task_id": task.task_id,
        "attempt": attempt,
        "problem_id": spec.problem_id,
        "slab_type": spec.slab_type,
        "shard_index": spec.shard_index,
        "start_index": spec.start_index,
        "stop_index": spec.stop_index,
        "rows": expected_rows,
        "schema_hash": schema_info.schema_hash,
        "part_path": str(published_path) if published_path is not None else None,
        "adopted_existing_part": bool(adopted),
        "reuse_structural": bool(reuse_structural),
        "finish_group_size": int(group),
        "analysis_ok_count": analysis_ok_count,
        "analysis_error_count": analysis_error_count,
        "feasible_count": feasible_count,
        "cardinality": int(space.cardinality),
        "structural_artifacts_mode": structural_mode,
        "structural_library_hits": library_hits,
        "structural_library_build_failure_fanouts": library_build_failure_fanouts,
        "structural_builds": structural_builds,
    }
    summary_path = attempt_dir / "summary.json"
    touch_attempt_heartbeat(attempt_dir)
    write_json_atomic(summary_path, _json_safe(summary))

    transition_task_status(
        task_dir,
        expected={TaskState.RUNNING},
        new_state=TaskState.COMPLETED,
        attempt=attempt,
        metadata={"summary": str(summary_path)},
    )
    return summary


def _write_failure(attempt_dir: Path, exc: BaseException) -> Path:
    """
    Atomically write ``failure.json`` for an infrastructure exception.
    """
    path = Path(attempt_dir) / "failure.json"
    write_json_atomic(
        path,
        {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        },
    )
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the worker CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m core.ioh_core.workflows.brute_force.worker",
        description="Execute one prepared brute-force shard attempt.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Prepared experiment root containing manifest.json",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        required=True,
        help="Zero-based task index from the manifest",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        required=True,
        help="One-based reserved attempt index",
    )
    parser.add_argument(
        "--no-structural-reuse",
        action="store_true",
        help="Disable finish-group structural-artifact reuse (reference mode)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint for the brute-force worker.
    """
    args = build_arg_parser().parse_args(argv)
    experiment_dir = Path(args.experiment_dir)
    task_index = int(args.task_index)
    attempt = int(args.attempt)

    attempt_dir: Path | None = None
    task_dir: Path | None = None
    try:
        paths = experiment_paths(experiment_dir)
        task_dir = task_directory(paths, task_index)
        attempt_dir = validate_attempt_ownership(task_dir, attempt)
        run_worker(
            experiment_dir=experiment_dir,
            task_index=task_index,
            attempt=attempt,
            reuse_structural=False if bool(args.no_structural_reuse) else None,
        )
        return 0
    except Exception as exc:
        if attempt_dir is not None:
            failure_path = _write_failure(attempt_dir, exc)
            if task_dir is not None:
                try:
                    status = read_task_status(task_dir)
                    state = TaskState(status["state"])
                    if state is TaskState.RUNNING:
                        transition_task_status(
                            task_dir,
                            expected={TaskState.RUNNING},
                            new_state=TaskState.FAILED,
                            attempt=attempt,
                            metadata={
                                "error_type": type(exc).__name__,
                                "failure": str(failure_path),
                            },
                        )
                except Exception:
                    pass
        print(f"[brute_force.worker] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
