"""
Validate sealed brute-force Parquet datasets without loading them whole.

Per-part and dataset checks consume only projected columns / metadata.
Temporary ``*.partial`` files and ``dataset/quarantine/`` entries are ignored.
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from core.execution_core.atomic_io import read_json
from core.execution_core.lifecycle import load_prepared_config
from core.execution_core.manifest import load_manifest
from core.execution_core.models import ExperimentPaths
from core.ioh_core.workflows.brute_force.deps import import_pyarrow
from core.ioh_core.workflows.brute_force.models import BruteForceShardSpec
from core.ioh_core.workflows.brute_force.parquet_writer import (
    shard_part_filename,
    shard_part_path,
)
from core.ioh_core.workflows.brute_force.planner import (
    BRUTE_FORCE_TASK_KIND,
    load_index_space_and_schema,
)
from core.ioh_core.workflows.brute_force.schema_io import schema_hash_from_document

_PART_NAME_RE = re.compile(r"^shard-(\d{6})\.parquet$")


class BruteForceValidationError(ValueError):
    """
    Raised when a brute-force dataset fails publication validation.

    Attributes
    ----------
    errors : tuple of str
        Human-readable failure reasons.
    """

    def __init__(self, errors: Sequence[str]) -> None:
        messages = tuple(str(item) for item in errors)
        if not messages:
            messages = ("brute-force dataset validation failed",)
        super().__init__("; ".join(messages))
        self.errors = messages


@dataclass(frozen=True)
class PartValidationResult:
    """
    Outcome of validating one planned shard part.
    """

    shard_index: int
    path: Path
    ok: bool
    errors: tuple[str, ...] = ()
    row_count: int | None = None
    index_min: int | None = None
    index_max: int | None = None
    bytes: int | None = None


@dataclass(frozen=True)
class DatasetValidationResult:
    """
    Full dataset validation report for the planned shard set.
    """

    ok: bool
    errors: tuple[str, ...]
    schema_hash: str | None
    planned_rows: int
    observed_rows: int
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


def _load_pq():
    import_pyarrow()
    return importlib.import_module("pyarrow.parquet")


def list_ignored_part_files(parts_dir: Path) -> tuple[str, ...]:
    """
    Return basenames under ``parts/`` that validation intentionally ignores.
    """
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


def _nonfinite_objective_errors(part_path: Path, shard_index: int) -> list[str]:
    """
    Return errors if any published ``y`` or ``y_p`` value is non-finite.
    """
    import_pyarrow()
    pc = importlib.import_module("pyarrow.compute")
    pq = _load_pq()
    table = pq.read_table(str(part_path), columns=["y", "y_p"])
    errors: list[str] = []
    for name in ("y", "y_p"):
        finite = pc.is_finite(table.column(name))
        if bool(pc.all(finite).as_py()):
            continue
        flags = finite.to_pylist()
        bad = next(i for i, ok in enumerate(flags) if not ok)
        value = table.column(name)[bad].as_py()
        errors.append(
            f"shard {shard_index}: non-finite {name} at row {bad} "
            f"(value={value!r})"
        )
    return errors


def _read_index_columns(part_path: Path) -> dict[str, list[Any]]:
    """
    Read only identity columns needed for range / monotonicity checks.
    """
    pq = _load_pq()
    table = pq.read_table(
        str(part_path),
        columns=["global_index", "shard_index", "problem_id"],
    )
    return {
        "global_index": table.column("global_index").to_pylist(),
        "shard_index": table.column("shard_index").to_pylist(),
        "problem_id": table.column("problem_id").to_pylist(),
    }


def validate_part(
    part_path: Path,
    *,
    spec: BruteForceShardSpec,
    expected_schema: Any,
) -> PartValidationResult:
    """
    Validate one published shard part against its planned specification.
    """
    part_path = Path(part_path)
    errors: list[str] = []
    if not part_path.is_file():
        return PartValidationResult(
            shard_index=spec.shard_index,
            path=part_path,
            ok=False,
            errors=(f"missing part for shard {spec.shard_index}: {part_path}",),
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
                bytes=part_path.stat().st_size,
            )

        if pf.metadata is None:
            errors.append(f"shard {spec.shard_index}: missing parquet metadata/footer")
        schema_ok = pf.schema_arrow.equals(expected_schema, check_metadata=False)
        if not schema_ok:
            errors.append(
                f"shard {spec.shard_index}: schema does not match sealed schema"
            )

        expected_rows = spec.stop_index - spec.start_index
        num_rows = None if pf.metadata is None else int(pf.metadata.num_rows)
        if num_rows is not None and num_rows != expected_rows:
            errors.append(
                f"shard {spec.shard_index}: row count {num_rows} != expected "
                f"{expected_rows}"
            )

        indices: list[int] = []
        try:
            columns = _read_index_columns(part_path)
            indices = [int(v) for v in columns["global_index"]]
            expected_indices = list(range(spec.start_index, spec.stop_index))
            if indices != expected_indices:
                if indices != sorted(indices) or len(set(indices)) != len(indices):
                    errors.append(
                        f"shard {spec.shard_index}: global_index is not "
                        "strictly monotonic"
                    )
                if not indices:
                    errors.append(
                        f"shard {spec.shard_index}: empty global_index column"
                    )
                elif (
                    indices[0] != spec.start_index
                    or indices[-1] != spec.stop_index - 1
                ):
                    errors.append(
                        f"shard {spec.shard_index}: index range "
                        f"[{indices[0]}, {indices[-1]}] != "
                        f"[{spec.start_index}, {spec.stop_index - 1}]"
                    )
                elif len(indices) == expected_rows:
                    errors.append(
                        f"shard {spec.shard_index}: global_index sequence mismatch"
                    )
                else:
                    errors.append(
                        f"shard {spec.shard_index}: global_index does not match "
                        f"planned range [{spec.start_index}, {spec.stop_index})"
                    )

            shard_values = {int(v) for v in columns["shard_index"]}
            if shard_values != {int(spec.shard_index)}:
                errors.append(
                    f"shard {spec.shard_index}: shard_index column values "
                    f"{sorted(shard_values)} != {{{spec.shard_index}}}"
                )
            problem_values = {str(v) for v in columns["problem_id"]}
            if problem_values != {str(spec.problem_id)}:
                errors.append(
                    f"shard {spec.shard_index}: problem_id column values "
                    f"{sorted(problem_values)} != {{{spec.problem_id!r}}}"
                )
        except Exception as exc:
            errors.append(
                f"shard {spec.shard_index}: failed reading identity columns: {exc}"
            )

        if schema_ok:
            try:
                errors.extend(
                    _nonfinite_objective_errors(part_path, spec.shard_index)
                )
            except Exception as exc:
                errors.append(
                    f"shard {spec.shard_index}: failed reading y/y_p: {exc}"
                )

        return PartValidationResult(
            shard_index=spec.shard_index,
            path=part_path,
            ok=not errors,
            errors=tuple(errors),
            row_count=len(indices) if indices else num_rows,
            index_min=indices[0] if indices else None,
            index_max=indices[-1] if indices else None,
            bytes=part_path.stat().st_size,
        )
    finally:
        if pf is not None and hasattr(pf, "close"):
            try:
                pf.close()
            except Exception:
                pass


def planned_shard_specs(experiment_dir: Path) -> tuple[BruteForceShardSpec, ...]:
    """
    Return planned shard specs from the sealed manifest (planned range only).
    """
    paths = _experiment_paths(experiment_dir)
    manifest = load_manifest(paths.manifest)
    if manifest.workflow_type != "brute_force":
        raise BruteForceValidationError(
            [f"expected workflow_type='brute_force', got {manifest.workflow_type!r}"]
        )
    specs: list[BruteForceShardSpec] = []
    for task in manifest.tasks:
        if task.kind != BRUTE_FORCE_TASK_KIND:
            raise BruteForceValidationError(
                [f"task {task.task_index} has unexpected kind {task.kind!r}"]
            )
        specs.append(BruteForceShardSpec.from_payload(task.payload))
    return tuple(specs)


def validate_dataset(experiment_dir: Path) -> DatasetValidationResult:
    """
    Validate planned parts under ``dataset/parts/``.

    Existing parts are checked even when the planned set is incomplete.
    Missing planned parts, invalid parts, and unexpected canonical shard
    files are errors. Temporary / quarantined names are ignored.
    """
    paths = _experiment_paths(experiment_dir)
    dataset_dir = paths.root / "dataset"
    parts_dir = dataset_dir / "parts"
    schema_path = dataset_dir / "schema.json"
    errors: list[str] = []

    if not schema_path.is_file():
        return DatasetValidationResult(
            ok=False,
            errors=(f"missing sealed schema.json: {schema_path}",),
            schema_hash=None,
            planned_rows=0,
            observed_rows=0,
        )

    schema_doc = read_json(schema_path)
    try:
        schema_hash = schema_hash_from_document(schema_doc)
    except ValueError as exc:
        return DatasetValidationResult(
            ok=False,
            errors=(str(exc),),
            schema_hash=None,
            planned_rows=0,
            observed_rows=0,
        )

    config = load_prepared_config(paths.root)
    problem_id = config.experiment.problem_ids[0]
    _space, schema_info = load_index_space_and_schema(
        slab_type=config.experiment.slab_type,
        problem_id=problem_id,
    )
    if schema_info.schema_hash != schema_hash:
        errors.append(
            "schema.json hash does not match rebuilt snapshot schema: "
            f"{schema_hash!r} != {schema_info.schema_hash!r}"
        )

    specs = planned_shard_specs(experiment_dir)
    planned_rows = sum(spec.stop_index - spec.start_index for spec in specs)
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
    for spec in specs:
        part_path = shard_part_path(dataset_dir, spec.shard_index)
        if not part_path.is_file():
            message = (
                f"missing part for planned shard {spec.shard_index} "
                f"range [{spec.start_index}, {spec.stop_index})"
            )
            part_results.append(
                PartValidationResult(
                    shard_index=spec.shard_index,
                    path=part_path,
                    ok=False,
                    errors=(message,),
                )
            )
            errors.append(message)
            continue
        result = validate_part(
            part_path,
            spec=spec,
            expected_schema=schema_info.schema,
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
        # Contiguous planned coverage with no gaps/overlaps.
        previous_stop = specs[0].start_index
        for spec in specs:
            if spec.start_index != previous_stop:
                errors.append(
                    f"planned coverage gap or overlap before shard "
                    f"{spec.shard_index}: expected start {previous_stop}, "
                    f"got {spec.start_index}"
                )
            previous_stop = spec.stop_index

    unique_errors: list[str] = []
    seen: set[str] = set()
    for message in errors:
        if message not in seen:
            unique_errors.append(message)
            seen.add(message)

    return DatasetValidationResult(
        ok=not unique_errors,
        errors=tuple(unique_errors),
        schema_hash=schema_hash,
        planned_rows=planned_rows,
        observed_rows=observed_rows,
        parts=tuple(part_results),
        ignored_files=ignored,
    )


def validate_dataset_or_raise(experiment_dir: Path) -> DatasetValidationResult:
    """
    Validate the dataset and raise :class:`BruteForceValidationError` on failure.
    """
    result = validate_dataset(experiment_dir)
    if not result.ok:
        raise BruteForceValidationError(result.errors)
    return result
