"""
Sensitivity dataset aggregation, SALib analyse, and ``_SUCCESS`` publication.

Publication order:

1. structural validate + regenerate-and-match
2. ``dataset/_SUCCESS``
3. extract full factorial responses → SALib analyse per response
4. ``aggregate/sensitivity_indices.json`` / ``.csv`` / ``summary.json``
5. ``aggregate/_SUCCESS`` last

The published response set matches brute-force factorial sensitivity:
objective (``response_key``), each ``util__*``, and each derived
``feasible__*`` binary. Constant responses are marked and skipped; aggregate
success requires at least one analyzable response.

If the dataset is structurally complete but no response is analyzable, the
dataset marker is preserved and aggregate success is withheld.
"""
from __future__ import annotations

import csv
import importlib
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.execution_core.artifacts import read_task_status
from core.execution_core.atomic_io import read_json, write_json_atomic, write_text_atomic
from core.execution_core.lifecycle import load_prepared_config
from core.execution_core.manifest import load_manifest
from core.execution_core.models import (
    ExperimentPaths,
    TaskState,
    TERMINAL_TASK_STATES,
    format_task_id,
)
from core.ioh_core.workflows.brute_force.analysis.factorial_sensitivity import (
    EXPRESSION_BINARY_FEASIBLE,
    EXPRESSION_COLUMN,
    STATUS_CONSTANT,
    STATUS_OK,
    FactorialResponseSpec,
    build_factorial_response_specs,
    materialize_response_values,
)
from core.ioh_core.workflows.brute_force.parquet_writer import shard_part_path
from core.ioh_core.workflows.sensitivity_analysis.deps import import_pyarrow
from core.ioh_core.workflows.sensitivity_analysis.design import load_sensitivity_design
from core.ioh_core.workflows.sensitivity_analysis.estimators import (
    ConstantResponseError,
    estimate_sobol_indices,
)
from core.ioh_core.workflows.sensitivity_analysis.models import SensitivityShardSpec
from core.ioh_core.workflows.sensitivity_analysis.planner import SENSITIVITY_TASK_KIND
from core.ioh_core.workflows.sensitivity_analysis.schema import (
    MATRIX_KIND_A,
    MATRIX_KIND_AB,
    MATRIX_KIND_B,
)
from core.ioh_core.workflows.sensitivity_analysis.validation import (
    SensitivityValidationError,
    design_directory,
    validate_dataset_or_raise,
)

_INDEX_CSV_FIELDS = (
    "response_family",
    "response_name",
    "status",
    "factor_index",
    "factor_name",
    "first_order",
    "first_order_conf_half_width",
    "total_effect",
    "total_effect_conf_half_width",
    "interaction_gap",
)


class SensitivityAggregationError(RuntimeError):
    """Raised when sensitivity aggregation cannot complete after validation."""


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


def write_dataset_success_marker(dataset_dir: Path) -> Path:
    """Atomically write ``dataset/_SUCCESS``."""
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / "_SUCCESS"
    write_text_atomic(path, "ok\n")
    return path


def write_aggregate_success_marker(aggregate_dir: Path) -> Path:
    """Atomically write ``aggregate/_SUCCESS``."""
    aggregate_dir = Path(aggregate_dir)
    path = aggregate_dir / "_SUCCESS"
    write_text_atomic(path, "ok\n")
    return path


def _json_float(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        raise SensitivityAggregationError(
            f"non-finite float in published indices: {number!r}"
        )
    return number


def _load_constraint_names(dataset_dir: Path) -> tuple[str, ...]:
    schema_path = Path(dataset_dir) / "schema.json"
    if not schema_path.is_file():
        raise SensitivityAggregationError(
            f"missing sealed dataset schema.json: {schema_path}"
        )
    document = read_json(schema_path)
    names = document.get("constraint_names")
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise SensitivityAggregationError(
            "dataset/schema.json constraint_names must be a list of non-empty strings"
        )
    return tuple(str(name) for name in names)


def build_saltelli_response_specs(
    *,
    response_key: str,
    constraint_names: Sequence[str],
) -> tuple[FactorialResponseSpec, ...]:
    """
    Build the full Saltelli response contract (BF factorial parity).

    Objective column is the sealed ``response_key`` (``y`` or ``y_p``).
    """
    if response_key not in {"y", "y_p"}:
        raise SensitivityAggregationError(
            f"unsupported response_key {response_key!r}; expected 'y' or 'y_p'"
        )
    return build_factorial_response_specs(
        constraint_names,
        objective_column=response_key,
    )


def extract_response_arrays(
    experiment_dir: Path,
    *,
    response_key: str,
    sample_size: int,
    factor_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Read one sealed column into ``y_A``, ``y_B``, ``y_AB`` (legacy helper).

    Prefer :func:`extract_factorial_response_arrays` for publication. This
    helper remains for unit tests that compare a single objective extract to
    SALib directly.
    """
    arrays, diagnostics = extract_factorial_response_arrays(
        experiment_dir,
        response_key=response_key,
        sample_size=sample_size,
        factor_count=factor_count,
        response_specs=build_saltelli_response_specs(
            response_key=response_key,
            constraint_names=(),
        ),
    )
    if response_key not in arrays:
        raise SensitivityAggregationError(
            f"objective response {response_key!r} missing after extract"
        )
    y_a, y_b, y_ab = arrays[response_key]
    return y_a, y_b, y_ab, diagnostics


def extract_factorial_response_arrays(
    experiment_dir: Path,
    *,
    response_key: str,
    sample_size: int,
    factor_count: int,
    response_specs: Sequence[FactorialResponseSpec] | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    """
    Read objective + util columns once and materialize all factorial responses.

    Returns
    -------
    arrays:
        Mapping response name → ``(y_A, y_B, y_AB)``.
    diagnostics:
        Row / analysis_ok counters shared across responses.

    Raises
    ------
    SensitivityAggregationError
        If any row has ``analysis_ok=False`` or a non-finite **source** value
        for a required column.
    """
    if response_key not in {"y", "y_p"}:
        raise SensitivityAggregationError(
            f"unsupported response_key {response_key!r}; expected 'y' or 'y_p'"
        )
    paths = _experiment_paths(experiment_dir)
    dataset_dir = paths.root / "dataset"
    manifest = load_manifest(paths.manifest)
    pq = _load_pq()

    constraint_names = _load_constraint_names(dataset_dir)
    specs = tuple(
        response_specs
        if response_specs is not None
        else build_saltelli_response_specs(
            response_key=response_key,
            constraint_names=constraint_names,
        )
    )
    source_columns = tuple(dict.fromkeys(spec.source_column for spec in specs))
    for column in source_columns:
        if column not in {response_key, *[f"util__{name}" for name in constraint_names]}:
            # Allow objective-only specs in unit tests (empty constraints).
            if column != response_key and not column.startswith("util__"):
                raise SensitivityAggregationError(
                    f"unsupported source column {column!r}"
                )

    source_a: dict[str, np.ndarray] = {
        column: np.full(sample_size, np.nan, dtype=float) for column in source_columns
    }
    source_b: dict[str, np.ndarray] = {
        column: np.full(sample_size, np.nan, dtype=float) for column in source_columns
    }
    source_ab: dict[str, np.ndarray] = {
        column: np.full((sample_size, factor_count), np.nan, dtype=float)
        for column in source_columns
    }
    seen_a = np.zeros(sample_size, dtype=bool)
    seen_b = np.zeros(sample_size, dtype=bool)
    seen_ab = np.zeros((sample_size, factor_count), dtype=bool)

    analysis_ok_count = 0
    analysis_error_count = 0
    total_rows = 0

    read_columns = [
        "sample_index",
        "matrix_kind",
        "factor_index",
        "analysis_ok",
        *source_columns,
    ]

    for task in manifest.tasks:
        if task.kind != SENSITIVITY_TASK_KIND:
            raise SensitivityAggregationError(
                f"task {task.task_index} has unexpected kind {task.kind!r}"
            )
        spec = SensitivityShardSpec.from_payload(task.payload)
        part = shard_part_path(dataset_dir, spec.shard_index)
        table = pq.read_table(str(part), columns=list(read_columns))
        samples = table.column("sample_index").to_pylist()
        kinds = table.column("matrix_kind").to_pylist()
        factor_indices = table.column("factor_index").to_pylist()
        analysis_ok = table.column("analysis_ok").to_pylist()
        column_values = {
            column: table.column(column).to_pylist() for column in source_columns
        }
        total_rows += len(samples)

        for row_index, (sample_index, kind, factor_index, ok) in enumerate(
            zip(samples, kinds, factor_indices, analysis_ok)
        ):
            sample = int(sample_index)
            if ok:
                analysis_ok_count += 1
            else:
                analysis_error_count += 1
                raise SensitivityAggregationError(
                    "every Saltelli row must have analysis_ok=True for "
                    f"publication; failed at sample={sample} kind={kind!r}"
                )

            parsed: dict[str, float] = {}
            for column in source_columns:
                raw = column_values[column][row_index]
                try:
                    value = float(raw)
                except (TypeError, ValueError) as exc:
                    raise SensitivityAggregationError(
                        f"non-numeric {column} at sample={sample} "
                        f"kind={kind!r}: {raw!r}"
                    ) from exc
                if not np.isfinite(value):
                    raise SensitivityAggregationError(
                        f"non-finite {column} at sample={sample} kind={kind!r}"
                    )
                parsed[column] = value

            if kind == MATRIX_KIND_A:
                if seen_a[sample]:
                    raise SensitivityAggregationError(
                        f"duplicate A response for sample_index={sample}"
                    )
                for column, value in parsed.items():
                    source_a[column][sample] = value
                seen_a[sample] = True
            elif kind == MATRIX_KIND_B:
                if seen_b[sample]:
                    raise SensitivityAggregationError(
                        f"duplicate B response for sample_index={sample}"
                    )
                for column, value in parsed.items():
                    source_b[column][sample] = value
                seen_b[sample] = True
            elif kind == MATRIX_KIND_AB:
                if factor_index is None:
                    raise SensitivityAggregationError(
                        f"AB row missing factor_index at sample={sample}"
                    )
                index = int(factor_index)
                if index < 0 or index >= factor_count:
                    raise SensitivityAggregationError(
                        f"AB factor_index {index} out of range at sample={sample}"
                    )
                if seen_ab[sample, index]:
                    raise SensitivityAggregationError(
                        f"duplicate AB response for sample={sample} "
                        f"factor_index={index}"
                    )
                for column, value in parsed.items():
                    source_ab[column][sample, index] = value
                seen_ab[sample, index] = True
            else:
                raise SensitivityAggregationError(
                    f"unknown matrix_kind {kind!r} at sample={sample}"
                )

    if not bool(np.all(seen_a)) or not bool(np.all(seen_b)) or not bool(
        np.all(seen_ab)
    ):
        raise SensitivityAggregationError(
            "incomplete response coverage for A/B/AB arrays"
        )

    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for spec in specs:
        source = spec.source_column
        y_a = materialize_response_values(
            source_a[source], expression=spec.expression
        )
        y_b = materialize_response_values(
            source_b[source], expression=spec.expression
        )
        # materialize AB column-wise without flattening factor axis incorrectly
        ab_source = source_ab[source]
        if spec.expression == EXPRESSION_COLUMN:
            y_ab = np.asarray(ab_source, dtype=float)
        elif spec.expression == EXPRESSION_BINARY_FEASIBLE:
            y_ab = np.where(ab_source <= 1.0, 1.0, 0.0).astype(float, copy=False)
        else:
            raise SensitivityAggregationError(
                f"unknown response expression: {spec.expression!r}"
            )
        arrays[spec.name] = (y_a, y_b, y_ab)

    diagnostics = {
        "total_rows": total_rows,
        "analysis_ok_count": analysis_ok_count,
        "analysis_error_count": analysis_error_count,
        "response_key": response_key,
        "response_names": [spec.name for spec in specs],
        "constraint_names": list(constraint_names),
    }
    return arrays, diagnostics


def _factors_from_estimate(
    *,
    factor_names: Sequence[str],
    estimate: Any,
) -> list[dict[str, Any]]:
    factors: list[dict[str, Any]] = []
    for index, name in enumerate(factor_names):
        first = float(estimate.first_order[index])
        total = float(estimate.total_effect[index])
        first_conf = None
        total_conf = None
        if estimate.first_order_conf_half_width is not None:
            first_conf = float(estimate.first_order_conf_half_width[index])
        if estimate.total_effect_conf_half_width is not None:
            total_conf = float(estimate.total_effect_conf_half_width[index])
        factors.append(
            {
                "factor_index": index,
                "factor_name": name,
                "first_order": _json_float(first),
                "first_order_conf_half_width": _json_float(first_conf),
                "total_effect": _json_float(total),
                "total_effect_conf_half_width": _json_float(total_conf),
                "interaction_gap": _json_float(total - first),
            }
        )
    return factors


def _constant_response_payload(spec: FactorialResponseSpec) -> dict[str, Any]:
    return {
        "family": spec.family,
        "name": spec.name,
        "status": STATUS_CONSTANT,
        "mean": None,
        "total_variance": 0.0,
        "factors": None,
        "sum_first_order": None,
        "sum_total_effect": None,
    }


def _ok_response_payload(
    *,
    spec: FactorialResponseSpec,
    estimate: Any,
    factor_names: Sequence[str],
    y_a: np.ndarray,
) -> dict[str, Any]:
    factors = _factors_from_estimate(factor_names=factor_names, estimate=estimate)
    return {
        "family": spec.family,
        "name": spec.name,
        "status": STATUS_OK,
        "mean": _json_float(float(np.mean(y_a))),
        "total_variance": _json_float(float(np.var(y_a, ddof=0))),
        "factors": factors,
        "sum_first_order": _json_float(float(np.sum(estimate.first_order))),
        "sum_total_effect": _json_float(float(np.sum(estimate.total_effect))),
        "salib_version": estimate.salib_version,
    }


def _build_indices_document(
    *,
    document: Any,
    section: Any,
    responses: Sequence[Mapping[str, Any]],
    diagnostics: dict[str, Any],
    design_hash: str,
    salib_version: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "method": "SALib.analyze.sobol",
        "sampling_engine": document.sampling_engine,
        "analysis_engine": document.analysis_engine,
        "packing_rule": document.packing_rule,
        "calc_second_order": False,
        "salib_version": salib_version,
        "numpy_version": document.numpy_version,
        "design_metadata_hash": design_hash,
        "response_key": document.response_key,
        "sample_size": document.sample_size,
        "factor_count": document.factor_count,
        "expected_evaluations": document.expected_evaluations,
        "evaluation_count": diagnostics["total_rows"],
        "compute_bootstrap_ci": bool(section.compute_bootstrap_ci),
        "bootstrap_resamples": int(section.bootstrap_resamples),
        "bootstrap_confidence_level": float(section.bootstrap_confidence_level),
        "bootstrap_seed": int(section.bootstrap_seed),
        "responses": list(responses),
        "ok_response_count": sum(
            1 for item in responses if item.get("status") == STATUS_OK
        ),
        "constant_response_count": sum(
            1 for item in responses if item.get("status") == STATUS_CONSTANT
        ),
    }
    if section.compute_bootstrap_ci:
        payload["confidence_quantity"] = "bootstrap_confidence_half_width"
    else:
        payload["confidence_quantity"] = None
    return payload


def _write_indices_csv(
    path: Path,
    responses: Sequence[Mapping[str, Any]],
) -> None:
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(_INDEX_CSV_FIELDS), lineterminator="\n"
    )
    writer.writeheader()
    for response in responses:
        status = str(response.get("status"))
        base = {
            "response_family": response.get("family"),
            "response_name": response.get("name"),
            "status": status,
        }
        factors = response.get("factors")
        if status == STATUS_CONSTANT or not factors:
            writer.writerow(
                {
                    **base,
                    "factor_index": "",
                    "factor_name": "",
                    "first_order": "",
                    "first_order_conf_half_width": "",
                    "total_effect": "",
                    "total_effect_conf_half_width": "",
                    "interaction_gap": "",
                }
            )
            continue
        for factor in factors:
            writer.writerow(
                {
                    **base,
                    "factor_index": factor.get("factor_index"),
                    "factor_name": factor.get("factor_name"),
                    "first_order": factor.get("first_order"),
                    "first_order_conf_half_width": factor.get(
                        "first_order_conf_half_width"
                    ),
                    "total_effect": factor.get("total_effect"),
                    "total_effect_conf_half_width": factor.get(
                        "total_effect_conf_half_width"
                    ),
                    "interaction_gap": factor.get("interaction_gap"),
                }
            )
    write_text_atomic(path, buffer.getvalue())


def _design_metadata_hash(design_dir: Path) -> str:
    from core.ioh_core.workflows.optimization_runs.snapshot import sha256_file

    return sha256_file(Path(design_dir) / "design.json")


def aggregate_sensitivity_experiment(experiment_dir: Path) -> dict[str, Any]:
    """
    Validate the dataset, publish markers, and write SALib Sobol indices.

    Always analyses the full factorial response set (objective + util +
    feasible binaries). Constant responses are recorded without failing
    siblings; ``aggregate/_SUCCESS`` requires at least one ``ok`` response.

    Returns
    -------
    dict
        Aggregate summary document (also written to ``aggregate/summary.json``).
    """
    paths = _experiment_paths(experiment_dir)
    manifest = load_manifest(paths.manifest)
    if manifest.workflow_type != "sensitivity_analysis":
        raise SensitivityAggregationError(
            "expected workflow_type='sensitivity_analysis', "
            f"got {manifest.workflow_type!r}"
        )

    dataset_dir = paths.root / "dataset"
    aggregate_dir = paths.aggregate
    dataset_success = dataset_dir / "_SUCCESS"
    aggregate_success = aggregate_dir / "_SUCCESS"

    for marker in (dataset_success, aggregate_success):
        if marker.exists():
            try:
                marker.unlink()
            except OSError:
                pass

    validation = validate_dataset_or_raise(experiment_dir)
    write_dataset_success_marker(dataset_dir)

    config = load_prepared_config(paths.root)
    section = config.sensitivity_analysis
    if section is None:
        raise SensitivityAggregationError(
            "prepared config is missing [sensitivity_analysis]"
        )
    design_dir = design_directory(paths)
    document = load_sensitivity_design(design_dir)
    design_hash = _design_metadata_hash(design_dir)

    task_rows: list[dict[str, Any]] = []
    for task in manifest.tasks:
        spec = SensitivityShardSpec.from_payload(task.payload)
        task_dir = paths.tasks / format_task_id(task.task_index)
        status = read_task_status(task_dir)
        state = TaskState(status["state"])
        attempt = int(status.get("attempt") or 0)
        part = shard_part_path(dataset_dir, spec.shard_index)
        row: dict[str, Any] = {
            "task_index": task.task_index,
            "task_id": task.task_id,
            "state": state.value,
            "attempt": attempt,
            "shard_index": spec.shard_index,
            "block_start": spec.block_start,
            "block_stop": spec.block_stop,
            "expected_rows": spec.row_count,
            "part_path": str(part),
            "part_present": part.is_file(),
        }
        if attempt >= 1:
            summary_path = task_dir / "attempts" / f"{attempt:03d}" / "summary.json"
            if summary_path.is_file():
                row["attempt_summary"] = read_json(summary_path)
        task_rows.append(row)

    terminal = sum(
        1 for row in task_rows if TaskState(row["state"]) in TERMINAL_TASK_STATES
    )
    completed = sum(
        1 for row in task_rows if row["state"] == TaskState.COMPLETED.value
    )
    failed = sum(1 for row in task_rows if row["state"] == TaskState.FAILED.value)

    summary: dict[str, Any] = {
        "workflow_type": "sensitivity_analysis",
        "validation_ok": True,
        "design_verified": bool(validation.design_verified),
        "dataset_success_marker_written": True,
        "aggregate_success_marker_written": False,
        "scientific_ok": False,
        "schema_hash": validation.schema_hash,
        "planned_rows": validation.planned_rows,
        "observed_rows": validation.observed_rows,
        "response_key": document.response_key,
        "sample_size": document.sample_size,
        "factor_count": document.factor_count,
        "compute_bootstrap_ci": bool(section.compute_bootstrap_ci),
        "design_metadata_hash": design_hash,
        "task_count": len(task_rows),
        "terminal_task_count": terminal,
        "completed_task_count": completed,
        "failed_task_count": failed,
        "ignored_part_files": list(validation.ignored_files),
        "tasks": task_rows,
        "scientific_error": None,
        "indices_path": None,
        "indices_csv_path": None,
    }

    aggregate_dir.mkdir(parents=True, exist_ok=True)

    try:
        constraint_names = _load_constraint_names(dataset_dir)
        response_specs = build_saltelli_response_specs(
            response_key=document.response_key,
            constraint_names=constraint_names,
        )
        arrays, diagnostics = extract_factorial_response_arrays(
            paths.root,
            response_key=document.response_key,
            sample_size=document.sample_size,
            factor_count=document.factor_count,
            response_specs=response_specs,
        )

        responses: list[dict[str, Any]] = []
        salib_version: str | None = None
        for spec in response_specs:
            y_a, y_b, y_ab = arrays[spec.name]
            try:
                estimate = estimate_sobol_indices(
                    y_a,
                    y_b,
                    y_ab,
                    compute_bootstrap_ci=bool(section.compute_bootstrap_ci),
                    num_resamples=int(section.bootstrap_resamples),
                    conf_level=float(section.bootstrap_confidence_level),
                    seed=int(section.bootstrap_seed),
                )
            except ConstantResponseError:
                responses.append(_constant_response_payload(spec))
                continue
            salib_version = estimate.salib_version
            responses.append(
                _ok_response_payload(
                    spec=spec,
                    estimate=estimate,
                    factor_names=document.factor_names,
                    y_a=y_a,
                )
            )

        ok_count = sum(1 for item in responses if item["status"] == STATUS_OK)
        if ok_count < 1:
            raise SensitivityAggregationError(
                "no analyzable Saltelli responses "
                "(all constant or otherwise unpublished)"
            )

        indices = _build_indices_document(
            document=document,
            section=section,
            responses=responses,
            diagnostics=diagnostics,
            design_hash=design_hash,
            salib_version=salib_version,
        )
        from core.execution_core.output_layout import (
            require_results_publication,
            resolve_results_root,
            saltelli_publication_dir,
        )

        indices_dir = require_results_publication(
            saltelli_publication_dir(
                resolve_results_root(paths.root),
                paths.root.name,
            )
        )
        indices_dir.mkdir(parents=True, exist_ok=True)
        indices_path = indices_dir / "sensitivity_indices.json"
        indices_csv_path = indices_dir / "sensitivity_indices.csv"
        write_json_atomic(indices_path, indices)
        _write_indices_csv(indices_csv_path, responses)

        summary["scientific_ok"] = True
        summary["analysis_ok_count"] = diagnostics["analysis_ok_count"]
        summary["analysis_error_count"] = diagnostics["analysis_error_count"]
        summary["evaluation_count"] = diagnostics["total_rows"]
        summary["indices_path"] = str(indices_path)
        summary["indices_csv_path"] = str(indices_csv_path)
        summary["ok_response_count"] = indices["ok_response_count"]
        summary["constant_response_count"] = indices["constant_response_count"]
        summary["response_names"] = [item["name"] for item in responses]
        summary["salib_version"] = salib_version
        write_json_atomic(aggregate_dir / "summary.json", summary)
        write_aggregate_success_marker(aggregate_dir)
        summary["aggregate_success_marker_written"] = True
        write_json_atomic(aggregate_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        summary["scientific_ok"] = False
        summary["scientific_error"] = str(exc)
        summary["aggregate_success_marker_written"] = False
        write_json_atomic(aggregate_dir / "summary.json", summary)
        if aggregate_success.exists():
            try:
                aggregate_success.unlink()
            except OSError:
                pass
        if isinstance(exc, SensitivityAggregationError):
            return summary
        if isinstance(exc, ValueError) and (
            "finite" in str(exc).lower()
            or "zero range" in str(exc).lower()
            or "constant" in str(exc).lower()
        ):
            return summary
        raise SensitivityAggregationError(
            f"sensitivity aggregation failed after dataset validation: {exc}"
        ) from exc


__all__ = [
    "SensitivityAggregationError",
    "SensitivityValidationError",
    "aggregate_sensitivity_experiment",
    "build_saltelli_response_specs",
    "extract_factorial_response_arrays",
    "extract_response_arrays",
    "write_aggregate_success_marker",
    "write_dataset_success_marker",
]
