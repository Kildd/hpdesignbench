"""
Validate sealed sensitivity Parquet datasets before publication.

Structural checks cover planned parts, SALib block identity, and AB catalogue
consistency against sealed A/B indices. Collect also regenerates the Saltelli
unit design from ``design.json`` (regenerate-and-match) before
``dataset/_SUCCESS``.
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.execution_core.atomic_io import read_json
from core.execution_core.lifecycle import load_prepared_config
from core.execution_core.manifest import load_manifest
from core.execution_core.models import ExperimentPaths
from core.ioh_core.problem_builder import build_problem_from_snapshot
from core.ioh_core.workflows.brute_force.parquet_writer import (
    shard_part_filename,
    shard_part_path,
)
from core.ioh_core.workflows.optimization_runs.snapshot import resolve_snapshot_slab_dir
from core.ioh_core.workflows.sensitivity_analysis.deps import import_pyarrow
from core.ioh_core.workflows.sensitivity_analysis.design import (
    cross_sample_row,
    expand_full_catalogue_indices,
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
    global_index_for,
    matrix_role_for_offset,
    schema_hash_from_document,
)

_PART_NAME_RE = re.compile(r"^shard-(\d{6})\.parquet$")


class SensitivityValidationError(ValueError):
    """
    Raised when a sensitivity dataset fails publication validation.

    Attributes
    ----------
    errors : tuple of str
        Human-readable failure reasons.
    """

    def __init__(self, errors: Sequence[str]) -> None:
        messages = tuple(str(item) for item in errors)
        if not messages:
            messages = ("sensitivity dataset validation failed",)
        super().__init__("; ".join(messages))
        self.errors = messages


@dataclass(frozen=True)
class PartValidationResult:
    """Outcome of validating one planned shard part."""

    shard_index: int
    path: Path
    ok: bool
    errors: tuple[str, ...] = ()
    row_count: int | None = None
    block_start: int | None = None
    block_stop: int | None = None
    bytes: int | None = None


@dataclass(frozen=True)
class DatasetValidationResult:
    """Full dataset validation report for the planned shard set."""

    ok: bool
    errors: tuple[str, ...]
    schema_hash: str | None
    planned_rows: int
    observed_rows: int
    design_verified: bool = False
    parts: tuple[PartValidationResult, ...] = ()
    ignored_files: tuple[str, ...] = ()


def _experiment_paths(experiment_dir: Path) -> ExperimentPaths:
    root = Path(experiment_dir).expanduser().resolve()
    return ExperimentPaths(
        root=root,
        inputs=root / "inputs",
        submissions=root / "submissions",
        tasks=root / "tasks",
        aggregate=root / "aggregate",
    )


def design_directory(paths: ExperimentPaths) -> Path:
    """Return ``experiment/inputs/sensitivity``."""
    return paths.inputs / "sensitivity"


def _load_pq():
    import_pyarrow()
    return importlib.import_module("pyarrow.parquet")


def list_ignored_part_files(parts_dir: Path) -> tuple[str, ...]:
    """Return basenames under ``parts/`` that validation intentionally ignores."""
    if not parts_dir.is_dir():
        return ()
    ignored: list[str] = []
    for path in sorted(parts_dir.iterdir()):
        name = path.name
        if not path.is_file():
            ignored.append(name)
            continue
        if name.endswith(".partial") or ".partial." in name:
            ignored.append(name)
            continue
        if _PART_NAME_RE.match(name) is None:
            ignored.append(name)
    return tuple(ignored)


def planned_shard_specs(experiment_dir: Path) -> tuple[SensitivityShardSpec, ...]:
    """Return planned shard specs from the sealed manifest."""
    paths = _experiment_paths(experiment_dir)
    manifest = load_manifest(paths.manifest)
    if manifest.workflow_type != "sensitivity_analysis":
        raise SensitivityValidationError(
            [
                "expected workflow_type='sensitivity_analysis', "
                f"got {manifest.workflow_type!r}"
            ]
        )
    specs: list[SensitivityShardSpec] = []
    for task in manifest.tasks:
        if task.kind != SENSITIVITY_TASK_KIND:
            raise SensitivityValidationError(
                [f"task {task.task_index} has unexpected kind {task.kind!r}"]
            )
        specs.append(SensitivityShardSpec.from_payload(task.payload))
    return tuple(specs)


def rebuild_schema_info(experiment_dir: Path) -> SensitivitySchema:
    """
    Rebuild the sensitivity Parquet schema from sealed design + snapshot.
    """
    paths = _experiment_paths(experiment_dir)
    config = load_prepared_config(paths.root)
    document = load_sensitivity_design(design_directory(paths))
    problem_id = config.experiment.problem_ids[0]
    if document.problem_id != problem_id:
        raise SensitivityValidationError(
            [
                "design.json problem_id does not match experiment: "
                f"{document.problem_id!r} != {problem_id!r}"
            ]
        )
    if document.slab_type != config.experiment.slab_type:
        raise SensitivityValidationError(
            [
                "design.json slab_type does not match experiment: "
                f"{document.slab_type!r} != {config.experiment.slab_type!r}"
            ]
        )
    snapshot_dir = resolve_snapshot_slab_dir(
        paths.inputs, config.experiment.slab_type
    )
    bundle = build_problem_from_snapshot(
        slab_type=config.experiment.slab_type,
        problem_id=problem_id,
        snapshot_dir=snapshot_dir,
    )
    return build_sensitivity_schema(
        var_names=bundle["var_names"],
        factor_names=document.factor_names,
        model=bundle["model"],
        constraint_names=tuple(bundle["constraints"].keys()),
    )


def _expected_identity_sequences(
    spec: SensitivityShardSpec,
    *,
    factor_names: Sequence[str],
) -> dict[str, list[Any]]:
    globals_: list[int] = []
    samples: list[int] = []
    offsets: list[int] = []
    kinds: list[str] = []
    factor_indices: list[int | None] = []
    factor_name_values: list[str | None] = []
    step = spec.rows_per_block
    for sample_index in range(spec.block_start, spec.block_stop):
        for offset in range(step):
            kind, factor_index, factor_name = matrix_role_for_offset(
                offset,
                factor_count=spec.factor_count,
                factor_names=factor_names,
            )
            globals_.append(
                global_index_for(
                    sample_index=sample_index,
                    matrix_offset=offset,
                    factor_count=spec.factor_count,
                )
            )
            samples.append(sample_index)
            offsets.append(offset)
            kinds.append(kind)
            factor_indices.append(factor_index)
            factor_name_values.append(factor_name)
    return {
        "global_index": globals_,
        "sample_index": samples,
        "matrix_offset": offsets,
        "matrix_kind": kinds,
        "factor_index": factor_indices,
        "factor_name": factor_name_values,
    }


def _validate_catalogue_rows(
    part_path: Path,
    *,
    spec: SensitivityShardSpec,
    schema_info: SensitivitySchema,
    document: SensitivityDesignDocument,
    a_index: np.ndarray,
    b_index: np.ndarray,
) -> list[str]:
    """
    Check every row's ``idx__*`` vector against sealed A/B expansion.
    """
    errors: list[str] = []
    pq = _load_pq()
    idx_columns = [f"idx__{name}" for name in schema_info.var_names]
    columns = [
        "sample_index",
        "matrix_offset",
        "matrix_kind",
        "factor_index",
        *idx_columns,
    ]
    try:
        table = pq.read_table(str(part_path), columns=columns)
    except Exception as exc:
        return [f"shard {spec.shard_index}: failed reading catalogue columns: {exc}"]

    sample_indices = table.column("sample_index").to_pylist()
    matrix_offsets = table.column("matrix_offset").to_pylist()
    matrix_kinds = table.column("matrix_kind").to_pylist()
    factor_indices = table.column("factor_index").to_pylist()
    idx_values = {
        name: table.column(f"idx__{name}").to_pylist()
        for name in schema_info.var_names
    }

    for row_pos, (sample_index, offset, kind, factor_index) in enumerate(
        zip(sample_indices, matrix_offsets, matrix_kinds, factor_indices)
    ):
        sample = int(sample_index)
        if sample < 0 or sample >= document.sample_size:
            errors.append(
                f"shard {spec.shard_index}: sample_index {sample} out of range"
            )
            continue
        a_row = a_index[sample]
        b_row = b_index[sample]
        if kind == "A":
            factor_row = np.asarray(a_row, dtype=int)
        elif kind == "B":
            factor_row = np.asarray(b_row, dtype=int)
        else:
            if factor_index is None:
                errors.append(
                    f"shard {spec.shard_index}: AB row missing factor_index "
                    f"at sample {sample}"
                )
                continue
            factor_row = cross_sample_row(a_row, b_row, int(factor_index)).astype(int)
        try:
            expected = expand_full_catalogue_indices(
                var_names=schema_info.var_names,
                factor_names=document.factor_names,
                factor_row=factor_row,
                held_constant_variables=document.held_constant_variables,
            )
        except Exception as exc:
            errors.append(
                f"shard {spec.shard_index}: cannot expand catalogue at "
                f"sample={sample} offset={offset}: {exc}"
            )
            continue
        actual = [int(idx_values[name][row_pos]) for name in schema_info.var_names]
        if actual != expected:
            errors.append(
                f"shard {spec.shard_index}: catalogue indices mismatch at "
                f"sample={sample} matrix_offset={offset} kind={kind!r}"
            )
            # One mismatch is enough signal; keep scanning for more samples.
            if len(errors) >= 8:
                errors.append(
                    f"shard {spec.shard_index}: further catalogue mismatches omitted"
                )
                break
    return errors


def validate_part(
    part_path: Path,
    *,
    spec: SensitivityShardSpec,
    schema_info: SensitivitySchema,
    document: SensitivityDesignDocument,
    a_index: np.ndarray,
    b_index: np.ndarray,
) -> PartValidationResult:
    """Validate one published shard part against its planned specification."""
    part_path = Path(part_path)
    errors: list[str] = []
    if not part_path.is_file():
        return PartValidationResult(
            shard_index=spec.shard_index,
            path=part_path,
            ok=False,
            errors=(f"missing part for shard {spec.shard_index}: {part_path}",),
            block_start=spec.block_start,
            block_stop=spec.block_stop,
        )

    pq = _load_pq()
    pf = None
    try:
        try:
            pf = pq.ParquetFile(str(part_path))
        except Exception as exc:
            return PartValidationResult(
                shard_index=spec.shard_index,
                path=part_path,
                ok=False,
                errors=(
                    f"unreadable parquet footer for shard {spec.shard_index}: {exc}",
                ),
                block_start=spec.block_start,
                block_stop=spec.block_stop,
                bytes=part_path.stat().st_size,
            )

        if pf.metadata is None:
            errors.append(f"shard {spec.shard_index}: missing parquet metadata/footer")
        if not pf.schema_arrow.equals(schema_info.schema, check_metadata=False):
            errors.append(
                f"shard {spec.shard_index}: schema does not match sealed schema"
            )

        expected_rows = spec.row_count
        num_rows = None if pf.metadata is None else int(pf.metadata.num_rows)
        if num_rows is not None and num_rows != expected_rows:
            errors.append(
                f"shard {spec.shard_index}: row count {num_rows} != expected "
                f"{expected_rows}"
            )

        try:
            identity_cols = [
                "global_index",
                "sample_index",
                "matrix_offset",
                "matrix_kind",
                "factor_index",
                "factor_name",
                "shard_index",
                "problem_id",
            ]
            table = pq.read_table(str(part_path), columns=identity_cols)
            observed = {
                name: table.column(name).to_pylist() for name in identity_cols
            }
            expected = _expected_identity_sequences(
                spec, factor_names=schema_info.factor_names
            )
            for key in (
                "global_index",
                "sample_index",
                "matrix_offset",
                "matrix_kind",
                "factor_index",
                "factor_name",
            ):
                if observed[key] != expected[key]:
                    errors.append(
                        f"shard {spec.shard_index}: {key} sequence does not match "
                        f"SALib blocks [{spec.block_start}, {spec.block_stop})"
                    )

            shard_values = {int(v) for v in observed["shard_index"]}
            if shard_values != {int(spec.shard_index)}:
                errors.append(
                    f"shard {spec.shard_index}: shard_index column values "
                    f"{sorted(shard_values)} != {{{spec.shard_index}}}"
                )
            problem_values = {str(v) for v in observed["problem_id"]}
            if problem_values != {str(spec.problem_id)}:
                errors.append(
                    f"shard {spec.shard_index}: problem_id column values "
                    f"{sorted(problem_values)} != {{{spec.problem_id!r}}}"
                )
        except Exception as exc:
            errors.append(
                f"shard {spec.shard_index}: failed reading identity columns: {exc}"
            )

        if not errors:
            errors.extend(
                _validate_catalogue_rows(
                    part_path,
                    spec=spec,
                    schema_info=schema_info,
                    document=document,
                    a_index=a_index,
                    b_index=b_index,
                )
            )

        return PartValidationResult(
            shard_index=spec.shard_index,
            path=part_path,
            ok=not errors,
            errors=tuple(errors),
            row_count=num_rows if num_rows is not None else expected_rows,
            block_start=spec.block_start,
            block_stop=spec.block_stop,
            bytes=part_path.stat().st_size,
        )
    finally:
        if pf is not None and hasattr(pf, "close"):
            try:
                pf.close()
            except Exception:
                pass


def validate_dataset(experiment_dir: Path) -> DatasetValidationResult:
    """
    Validate planned parts under ``dataset/parts/`` and regenerate the design.

    Existing parts are checked even when the planned set is incomplete.
    Missing planned parts, invalid parts, unexpected canonical shard files,
    and regenerate-and-match failures are errors.
    """
    paths = _experiment_paths(experiment_dir)
    dataset_dir = paths.root / "dataset"
    parts_dir = dataset_dir / "parts"
    schema_path = dataset_dir / "schema.json"
    errors: list[str] = []
    design_verified = False

    try:
        schema_info = rebuild_schema_info(experiment_dir)
    except SensitivityValidationError as exc:
        return DatasetValidationResult(
            ok=False,
            errors=exc.errors,
            schema_hash=None,
            planned_rows=0,
            observed_rows=0,
        )
    except Exception as exc:
        return DatasetValidationResult(
            ok=False,
            errors=(f"failed rebuilding sensitivity schema: {exc}",),
            schema_hash=None,
            planned_rows=0,
            observed_rows=0,
        )

    if schema_path.is_file():
        try:
            sealed_hash = schema_hash_from_document(read_json(schema_path))
        except Exception as exc:
            return DatasetValidationResult(
                ok=False,
                errors=(f"invalid sealed schema.json: {exc}",),
                schema_hash=None,
                planned_rows=0,
                observed_rows=0,
            )
        if sealed_hash != schema_info.schema_hash:
            errors.append(
                "schema.json hash does not match rebuilt snapshot schema: "
                f"{sealed_hash!r} != {schema_info.schema_hash!r}"
            )
    else:
        errors.append(f"missing sealed schema.json: {schema_path}")

    design_dir = design_directory(paths)
    try:
        document = load_sensitivity_design(design_dir)
        _a_unit, _b_unit, a_index, b_index = verify_sealed_arrays(
            design_dir, document=document
        )
        design_verified = True
    except Exception as exc:
        errors.append(f"regenerate-and-match design check failed: {exc}")
        document = None
        a_index = None
        b_index = None

    try:
        specs = planned_shard_specs(experiment_dir)
    except SensitivityValidationError as exc:
        return DatasetValidationResult(
            ok=False,
            errors=tuple([*errors, *exc.errors]),
            schema_hash=schema_info.schema_hash,
            planned_rows=0,
            observed_rows=0,
            design_verified=design_verified,
        )

    for spec in specs:
        if spec.schema_hash != schema_info.schema_hash:
            errors.append(
                f"shard {spec.shard_index} schema_hash does not match rebuilt "
                f"schema: {spec.schema_hash!r} != {schema_info.schema_hash!r}"
            )
        if document is not None and spec.factor_count != document.factor_count:
            errors.append(
                f"shard {spec.shard_index} factor_count {spec.factor_count} != "
                f"design factor_count {document.factor_count}"
            )
        if document is not None and spec.rows_per_block != document.factor_count + 2:
            errors.append(
                f"shard {spec.shard_index} rows_per_block "
                f"{spec.rows_per_block} != k+2={document.factor_count + 2}"
            )

    planned_rows = sum(spec.row_count for spec in specs)
    if document is not None and planned_rows != document.expected_evaluations:
        errors.append(
            f"planned rows {planned_rows} != design expected_evaluations "
            f"{document.expected_evaluations}"
        )

    ignored = list_ignored_part_files(parts_dir)
    planned_names = {shard_part_filename(spec.shard_index) for spec in specs}
    extra: list[str] = []
    if parts_dir.is_dir():
        for path in sorted(parts_dir.iterdir()):
            if not path.is_file():
                continue
            if _PART_NAME_RE.match(path.name) is None:
                continue
            if path.name not in planned_names:
                extra.append(path.name)
    if extra:
        errors.append(f"unexpected shard parts: {extra}")

    part_results: list[PartValidationResult] = []
    observed_rows = 0
    can_check_parts = (
        document is not None and a_index is not None and b_index is not None
    )
    for spec in specs:
        part_path = shard_part_path(dataset_dir, spec.shard_index)
        if not part_path.is_file():
            message = (
                f"missing part for planned shard {spec.shard_index} "
                f"blocks [{spec.block_start}, {spec.block_stop})"
            )
            part_results.append(
                PartValidationResult(
                    shard_index=spec.shard_index,
                    path=part_path,
                    ok=False,
                    errors=(message,),
                    block_start=spec.block_start,
                    block_stop=spec.block_stop,
                )
            )
            errors.append(message)
            continue
        if not can_check_parts:
            part_results.append(
                PartValidationResult(
                    shard_index=spec.shard_index,
                    path=part_path,
                    ok=False,
                    errors=("skipped part checks because design verification failed",),
                    block_start=spec.block_start,
                    block_stop=spec.block_stop,
                )
            )
            continue
        result = validate_part(
            part_path,
            spec=spec,
            schema_info=schema_info,
            document=document,
            a_index=a_index,
            b_index=b_index,
        )
        part_results.append(result)
        if result.ok:
            observed_rows += int(result.row_count or 0)
        else:
            errors.extend(result.errors)

    if specs and all(result.ok for result in part_results):
        if observed_rows != planned_rows:
            errors.append(
                f"total rows {observed_rows} != planned rows {planned_rows}"
            )
        previous_stop = specs[0].block_start
        for spec in specs:
            if spec.block_start != previous_stop:
                errors.append(
                    f"planned coverage gap or overlap before shard "
                    f"{spec.shard_index}: expected block_start {previous_stop}, "
                    f"got {spec.block_start}"
                )
            previous_stop = spec.block_stop
        if document is not None and previous_stop != document.sample_size:
            errors.append(
                f"planned blocks end at {previous_stop}, "
                f"expected sample_size {document.sample_size}"
            )

    unique_errors: list[str] = []
    seen: set[str] = set()
    for message in errors:
        if message not in seen:
            unique_errors.append(message)
            seen.add(message)

    return DatasetValidationResult(
        ok=not unique_errors,
        errors=tuple(unique_errors),
        schema_hash=schema_info.schema_hash,
        planned_rows=planned_rows,
        observed_rows=observed_rows,
        design_verified=design_verified and not unique_errors,
        parts=tuple(part_results),
        ignored_files=ignored,
    )


def validate_dataset_or_raise(experiment_dir: Path) -> DatasetValidationResult:
    """Validate the dataset and raise on failure."""
    result = validate_dataset(experiment_dir)
    if not result.ok:
        raise SensitivityValidationError(result.errors)
    return result
