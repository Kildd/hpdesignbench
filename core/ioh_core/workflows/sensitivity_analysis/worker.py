"""
Single-shard Sobol sensitivity worker for prepared experiments.

Invoked as:

``python -m core.ioh_core.workflows.sensitivity_analysis.worker
    --experiment-dir PATH --task-index N --attempt N``

The worker loads sealed A/B catalogue-index matrices, evaluates radial blocks
in SALib packing order ``A``, ``AB_0`` … ``AB_{k-1}``, ``B``, and publishes
one Parquet part. It never imports SALib, never runs an optimiser, and never
attaches an IOH logger. Scientific ``analysis_ok=False`` rows remain successful
task completions; only infrastructure exceptions write ``failure.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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
from core.ioh_core.cache_eval import EvalContext
from core.ioh_core.problem_builder import build_problem_from_snapshot
from core.ioh_core.structural_artifacts.artifact import (
    get_structural_build_count,
    reset_structural_build_count,
)
from core.ioh_core.structural_artifacts.eval_backend import (
    configure_eval_context_structural_backend,
)
from core.ioh_core.structural_artifacts.sealing import (
    open_structural_provider_session,
    slab_csv_input_hashes,
)
from core.ioh_core.workflows.brute_force.parquet_writer import (
    ParquetShardWriter,
    shard_part_path,
)
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
    validate_snapshot_hashes,
)
from core.ioh_core.workflows.sensitivity_analysis.design import (
    cross_sample_row,
    expand_full_catalogue_indices,
    load_npy,
    load_sensitivity_design,
    verify_sealed_arrays,
)
from core.ioh_core.workflows.sensitivity_analysis.models import (
    SensitivityDesignDocument,
    SensitivityShardSpec,
)
from core.ioh_core.workflows.sensitivity_analysis.planner import SENSITIVITY_TASK_KIND
from core.ioh_core.workflows.sensitivity_analysis.schema import (
    SensitivitySchema,
    build_sensitivity_schema,
    flatten_sensitivity_row,
    global_index_for,
    matrix_role_for_offset,
    rows_per_block,
)


class WorkerError(RuntimeError):
    """Raised for invalid worker inputs or ownership checks."""


def experiment_paths(experiment_dir: Path) -> ExperimentPaths:
    """Build the standard path bundle for an experiment root."""
    root = Path(experiment_dir).expanduser().resolve()
    return ExperimentPaths(
        root=root,
        inputs=root / "inputs",
        submissions=root / "submissions",
        tasks=root / "tasks",
        aggregate=root / "aggregate",
    )


def dataset_directory(paths: ExperimentPaths) -> Path:
    """Return ``experiment/dataset``."""
    return paths.root / "dataset"


def design_directory(paths: ExperimentPaths) -> Path:
    """Return ``experiment/inputs/sensitivity``."""
    return paths.inputs / "sensitivity"


def task_directory(paths: ExperimentPaths, task_index: int) -> Path:
    """Return the on-disk task directory for ``task_index``."""
    return paths.tasks / format_task_id(task_index)


def attempt_directory(task_dir: Path, attempt: int) -> Path:
    """Return the reserved attempt directory path."""
    if attempt < 1:
        raise WorkerError(f"attempt must be >= 1, got {attempt}")
    return Path(task_dir) / "attempts" / f"{attempt:03d}"


def validate_attempt_ownership(task_dir: Path, attempt: int) -> Path:
    """Require that the reserved attempt directory exists for this worker."""
    attempt_dir = attempt_directory(task_dir, attempt)
    if not attempt_dir.is_dir():
        raise WorkerError(
            f"attempt directory missing for ownership check: {attempt_dir}"
        )
    return attempt_dir


def _json_safe(value: Any) -> Any:
    """Convert summary values to JSON-serialisable forms."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def combined_snapshot_hash(input_files: Mapping[str, str]) -> str:
    """Deterministic digest over the sealed input-file hash map."""
    payload = "|".join(
        f"{path}:{digest}" for path, digest in sorted(input_files.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def schema_from_bundle_and_design(
    bundle: Mapping[str, Any],
    document: SensitivityDesignDocument,
) -> SensitivitySchema:
    """Build the Parquet schema for a snapshotted problem and sealed design."""
    var_names = tuple(str(name) for name in bundle["var_names"])
    constraint_names = tuple(str(name) for name in bundle["constraints"].keys())
    return build_sensitivity_schema(
        var_names=var_names,
        factor_names=document.factor_names,
        model=bundle["model"],
        constraint_names=constraint_names,
    )


def _cleanup_attempt_partials(attempt_dir: Path, shard_index: int) -> None:
    """Remove attempt-owned ``*.partial`` files for this shard only."""
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
    """Move an invalid published part into ``dataset/quarantine/``."""
    part_path = Path(part_path)
    quarantine_dir = part_path.parent.parent / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    safe_reason = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in reason
    )[:48]
    destination = quarantine_dir / f"{part_path.name}.{safe_reason}.{os.getpid()}.bad"
    try:
        os.replace(part_path, destination)
    except OSError:
        shutil.copy2(part_path, destination)
        try:
            part_path.unlink()
        except OSError:
            pass
    return destination


def validate_existing_part(
    part_path: Path,
    *,
    schema_info: SensitivitySchema,
    spec: SensitivityShardSpec,
) -> None:
    """
    Validate a published part for adoption.

    Checks schema, row count, global-index sequence, sample/block bounds,
    matrix offsets, and AB factor-name mapping.
    """
    from core.ioh_core.workflows.sensitivity_analysis.deps import import_pyarrow

    import_pyarrow()
    pq = importlib.import_module("pyarrow.parquet")
    expected_rows = spec.row_count
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

        table = pf.read(
            columns=[
                "global_index",
                "sample_index",
                "matrix_offset",
                "matrix_kind",
                "factor_index",
                "factor_name",
            ]
        )
        global_indices = table.column("global_index").to_pylist()
        sample_indices = table.column("sample_index").to_pylist()
        matrix_offsets = table.column("matrix_offset").to_pylist()
        matrix_kinds = table.column("matrix_kind").to_pylist()
        factor_indices = table.column("factor_index").to_pylist()
        factor_names = table.column("factor_name").to_pylist()

        expected_globals: list[int] = []
        expected_samples: list[int] = []
        expected_offsets: list[int] = []
        step = spec.rows_per_block
        for sample_index in range(spec.block_start, spec.block_stop):
            for offset in range(step):
                expected_globals.append(
                    global_index_for(
                        sample_index=sample_index,
                        matrix_offset=offset,
                        factor_count=spec.factor_count,
                    )
                )
                expected_samples.append(sample_index)
                expected_offsets.append(offset)
        if global_indices != expected_globals:
            raise WorkerError(
                f"part {part_path} global_index sequence does not match "
                f"blocks [{spec.block_start}, {spec.block_stop})"
            )
        if sample_indices != expected_samples or matrix_offsets != expected_offsets:
            raise WorkerError(
                f"part {part_path} sample_index/matrix_offset sequence is invalid"
            )

        for offset, kind, factor_index, factor_name in zip(
            matrix_offsets, matrix_kinds, factor_indices, factor_names
        ):
            expected_kind, expected_fi, expected_fn = matrix_role_for_offset(
                int(offset),
                factor_count=spec.factor_count,
                factor_names=schema_info.factor_names,
            )
            if kind != expected_kind:
                raise WorkerError(
                    f"part {part_path} matrix_kind={kind!r} at offset {offset}, "
                    f"expected {expected_kind!r}"
                )
            if factor_index != expected_fi or factor_name != expected_fn:
                raise WorkerError(
                    f"part {part_path} factor mapping mismatch at offset {offset}"
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
    schema_info: SensitivitySchema,
    spec: SensitivityShardSpec,
) -> bool:
    """Adopt a valid final part, or quarantine an invalid one."""
    part_path = Path(part_path)
    if not part_path.is_file():
        return False
    try:
        validate_existing_part(part_path, schema_info=schema_info, spec=spec)
        return True
    except WorkerError as exc:
        try:
            quarantine_part(part_path, reason=type(exc).__name__)
        except OSError:
            try:
                part_path.unlink()
            except OSError:
                pass
        return False


def build_block_candidates(
    *,
    sample_index: int,
    a_row: Sequence[int],
    b_row: Sequence[int],
    var_names: Sequence[str],
    document: SensitivityDesignDocument,
) -> list[tuple[int, list[int]]]:
    """
    Build ``(matrix_offset, full_catalogue_indices)`` rows for one radial block.
    """
    factor_count = document.factor_count
    step = rows_per_block(factor_count)
    rows: list[tuple[int, list[int]]] = []
    for offset in range(step):
        kind, factor_index, _factor_name = matrix_role_for_offset(
            offset,
            factor_count=factor_count,
            factor_names=document.factor_names,
        )
        if kind == "A":
            factor_row = np.asarray(a_row, dtype=int)
        elif kind == "B":
            factor_row = np.asarray(b_row, dtype=int)
        else:
            assert factor_index is not None
            factor_row = cross_sample_row(a_row, b_row, factor_index).astype(int)
        full = expand_full_catalogue_indices(
            var_names=var_names,
            factor_names=document.factor_names,
            factor_row=factor_row,
            held_constant_variables=document.held_constant_variables,
        )
        rows.append((offset, full))
    return rows


def run_worker(
    *,
    experiment_dir: Path,
    task_index: int,
    attempt: int,
) -> dict[str, Any]:
    """
    Execute one prepared sensitivity shard attempt.
    """
    paths = experiment_paths(experiment_dir)
    manifest = load_manifest(paths.manifest)
    if manifest.workflow_type != "sensitivity_analysis":
        raise WorkerError(
            "worker requires workflow_type='sensitivity_analysis', "
            f"got {manifest.workflow_type!r}"
        )
    if task_index < 0 or task_index >= len(manifest.tasks):
        raise WorkerError(
            f"task_index {task_index} out of range for "
            f"{len(manifest.tasks)} manifest tasks"
        )
    task = manifest.tasks[task_index]
    if task.kind != SENSITIVITY_TASK_KIND:
        raise WorkerError(
            f"task {task_index} has kind {task.kind!r}, "
            f"expected {SENSITIVITY_TASK_KIND!r}"
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
    if config.sensitivity_analysis is None:
        raise WorkerError("prepared config is missing [sensitivity_analysis]")
    section = config.sensitivity_analysis

    spec = SensitivityShardSpec.from_payload(task.payload)
    design_dir = design_directory(paths)
    document = load_sensitivity_design(design_dir)
    verify_sealed_arrays(design_dir, document)

    if document.factor_names != section.factor_names:
        raise WorkerError(
            "sealed design factor_names do not match prepared config: "
            f"{document.factor_names!r} != {section.factor_names!r}"
        )
    if document.sample_size != section.sample_size:
        raise WorkerError(
            "sealed design sample_size does not match prepared config: "
            f"{document.sample_size} != {section.sample_size}"
        )
    if spec.factor_count != document.factor_count:
        raise WorkerError(
            "shard factor_count does not match sealed design: "
            f"{spec.factor_count} != {document.factor_count}"
        )
    if spec.block_start < 0 or spec.block_stop > document.sample_size:
        raise WorkerError(
            f"shard block range [{spec.block_start}, {spec.block_stop}) outside "
            f"sample_size {document.sample_size}"
        )

    snapshot_dir = resolve_snapshot_slab_dir(paths.inputs, spec.slab_type)
    snapshot_hash = combined_snapshot_hash(
        slab_csv_input_hashes(
            manifest.input_files,
            slab_type=spec.slab_type,
        )
    )
    bundle = build_problem_from_snapshot(
        slab_type=spec.slab_type,
        problem_id=spec.problem_id,
        snapshot_dir=snapshot_dir,
    )
    schema_info = schema_from_bundle_and_design(bundle, document)
    if spec.schema_hash != schema_info.schema_hash:
        raise WorkerError(
            "shard schema_hash does not match rebuilt snapshot schema: "
            f"{spec.schema_hash!r} != {schema_info.schema_hash!r}"
        )

    a_index = np.asarray(load_npy(design_dir / "A_index.npy"))
    b_index = np.asarray(load_npy(design_dir / "B_index.npy"))
    if a_index.shape != (document.sample_size, document.factor_count):
        raise WorkerError(
            f"A_index shape {a_index.shape} != "
            f"({document.sample_size}, {document.factor_count})"
        )
    if b_index.shape != a_index.shape:
        raise WorkerError("B_index shape does not match A_index")

    expected_rows = spec.row_count
    dataset_dir = dataset_directory(paths)
    parts_dir = dataset_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    destination = shard_part_path(dataset_dir, spec.shard_index)

    _cleanup_attempt_partials(attempt_dir, spec.shard_index)

    adopted = try_adopt_existing_part(
        destination,
        schema_info=schema_info,
        spec=spec,
    )

    analysis_ok_count = 0
    analysis_error_count = 0
    feasible_count = 0
    evaluation_count = 0
    published_path = destination.resolve() if destination.is_file() else None
    structural_mode = "compute"
    library_hits = 0
    library_build_failure_fanouts = 0
    structural_builds = 0

    with open_structural_provider_session(paths.root) as (seal, provider):
        structural_mode = seal.mode
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
                    batch_rows=section.parquet_batch_rows,
                    compression=section.compression,
                )
                # Rebuild EvalContext so callers/tests can override analysis_fn
                # while still using the snapshot run_config from the bundle ctx.
                ctx = EvalContext(
                    bundle["decode"],
                    bundle["analysis_fn"],
                    bundle["ctx"].run_config,
                )
                configure_eval_context_structural_backend(
                    ctx,
                    bundle,
                    provider=provider,
                    snapshot_hash=snapshot_hash,
                )
                var_names = tuple(str(name) for name in bundle["var_names"])

                for sample_index in range(spec.block_start, spec.block_stop):
                    touch_attempt_heartbeat(attempt_dir)
                    candidates = build_block_candidates(
                        sample_index=sample_index,
                        a_row=a_index[sample_index],
                        b_row=b_index[sample_index],
                        var_names=var_names,
                        document=document,
                    )
                    for matrix_offset, catalogue_indices in candidates:
                        started = time.perf_counter()
                        result = ctx.get(catalogue_indices)
                        elapsed = time.perf_counter() - started
                        evaluation_count += 1
                        if result.get("analysis_ok"):
                            analysis_ok_count += 1
                        else:
                            analysis_error_count += 1
                        if result.get("feasible"):
                            feasible_count += 1
                        writer.write_row(
                            flatten_sensitivity_row(
                                schema_info=schema_info,
                                global_index=global_index_for(
                                    sample_index=sample_index,
                                    matrix_offset=matrix_offset,
                                    factor_count=document.factor_count,
                                ),
                                shard_index=spec.shard_index,
                                sample_index=sample_index,
                                matrix_offset=matrix_offset,
                                problem_id=spec.problem_id,
                                evaluation_time_s=elapsed,
                                catalogue_indices=catalogue_indices,
                                result=result,
                            )
                        )
                if evaluation_count != expected_rows:
                    raise WorkerError(
                        f"evaluation count {evaluation_count} != expected {expected_rows}"
                    )
                writer.close()
                published_path = writer.publish()
                ctx.clear_structural_backend()
            except Exception:
                if writer is not None:
                    writer.abandon()
                raise
        else:
            analysis_ok_count = -1
            analysis_error_count = -1
            feasible_count = -1
            evaluation_count = expected_rows

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
        "block_start": spec.block_start,
        "block_stop": spec.block_stop,
        "rows": expected_rows,
        "evaluations": evaluation_count,
        "schema_hash": schema_info.schema_hash,
        "part_path": str(published_path) if published_path is not None else None,
        "adopted_existing_part": bool(adopted),
        "analysis_ok_count": analysis_ok_count,
        "analysis_error_count": analysis_error_count,
        "feasible_count": feasible_count,
        "input_snapshot_hash": snapshot_hash,
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
    """Atomically write ``failure.json`` for an infrastructure exception."""
    path = Path(attempt_dir) / "failure.json"
    write_json_atomic(
        path,
        {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the sensitivity worker."""
    parser = argparse.ArgumentParser(
        description="Evaluate one Sobol sensitivity shard into a Parquet part."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Prepared experiment root",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        required=True,
        help="Zero-based manifest task index",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        required=True,
        help="One-based reserved attempt index",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    experiment_dir = Path(args.experiment_dir)
    task_index = int(args.task_index)
    attempt = int(args.attempt)

    paths = experiment_paths(experiment_dir)
    attempt_dir: Path | None = None
    try:
        attempt_dir = validate_attempt_ownership(
            task_directory(paths, task_index), attempt
        )
        run_worker(
            experiment_dir=experiment_dir,
            task_index=task_index,
            attempt=attempt,
        )
        return 0
    except Exception as exc:
        if attempt_dir is not None and attempt_dir.is_dir():
            try:
                _write_failure(attempt_dir, exc)
            except Exception:
                pass
            try:
                task_dir = task_directory(paths, task_index)
                status = read_task_status(task_dir)
                if TaskState(status["state"]) is TaskState.RUNNING:
                    transition_task_status(
                        task_dir,
                        expected={TaskState.RUNNING},
                        new_state=TaskState.FAILED,
                        attempt=attempt,
                        metadata={"error": str(exc)},
                    )
            except Exception:
                pass
        print(f"sensitivity worker failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
