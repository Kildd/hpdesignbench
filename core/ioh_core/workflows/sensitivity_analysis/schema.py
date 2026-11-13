"""
Stable Parquet schema for Sobol sensitivity evaluation rows.

Identity columns encode the locked SALib packing order
``A``, ``AB_0`` … ``AB_{k-1}``, ``B``. Scientific result columns match the
brute-force analysis contract.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.ioh_core.workflows.sensitivity_analysis.deps import import_pyarrow

_SCIENTIFIC_RESULT_KEYS = frozenset(
    {
        "y",
        "y_p",
        "constraint_values",
        "violations_",
        "penalties_",
        "penalty_sum",
        "penalty_factor",
        "params",
        "analysis_ok",
        "constraint_feasible",
        "feasible",
        "analysis_error",
        "remapped_utilizations",
        "cost_components_missing",
    }
)

MATRIX_KIND_A = "A"
MATRIX_KIND_B = "B"
MATRIX_KIND_AB = "AB"


class SensitivitySchemaError(ValueError):
    """Raised when a sensitivity Parquet schema or row is invalid."""


@dataclass(frozen=True)
class SensitivitySchema:
    """
    Explicit Arrow schema plus derived metadata for one sealed design.

    Attributes
    ----------
    schema : pyarrow.Schema
        Column layout used by the Parquet writer.
    schema_hash : str
        Deterministic SHA-256 hex digest of the canonical schema description.
    var_names : tuple of str
        Full problem variable names (``idx__*`` order).
    factor_names : tuple of str
        Selected sensitivity factors (AB factor_index order).
    param_names : tuple of str
        Full model parameter names (``param__*`` order).
    constraint_names : tuple of str
        Constraint-default names.
    """

    schema: Any
    schema_hash: str
    var_names: tuple[str, ...]
    factor_names: tuple[str, ...]
    param_names: tuple[str, ...]
    constraint_names: tuple[str, ...]


def rows_per_block(factor_count: int) -> int:
    """Return ``k + 2`` for a Saltelli first-order design."""
    if factor_count < 1:
        raise SensitivitySchemaError(
            f"factor_count must be >= 1, got {factor_count}"
        )
    return int(factor_count) + 2


def matrix_role_for_offset(
    matrix_offset: int,
    *,
    factor_count: int,
    factor_names: Sequence[str],
) -> tuple[str, int | None, str | None]:
    """
    Map a matrix offset to ``(matrix_kind, factor_index, factor_name)``.

    Packing is SALib order: offset ``0`` → A, ``1..k`` → AB_i, ``k+1`` → B.
    """
    count = int(factor_count)
    if len(factor_names) != count:
        raise SensitivitySchemaError(
            "factor_names length must equal factor_count "
            f"({len(factor_names)} != {count})"
        )
    step = rows_per_block(count)
    offset = int(matrix_offset)
    if offset < 0 or offset >= step:
        raise SensitivitySchemaError(
            f"matrix_offset {offset} out of range for k+2={step}"
        )
    if offset == 0:
        return MATRIX_KIND_A, None, None
    if offset == step - 1:
        return MATRIX_KIND_B, None, None
    factor_index = offset - 1
    return MATRIX_KIND_AB, factor_index, str(factor_names[factor_index])


def global_index_for(
    *,
    sample_index: int,
    matrix_offset: int,
    factor_count: int,
) -> int:
    """Return ``sample_index * (k + 2) + matrix_offset``."""
    if sample_index < 0:
        raise SensitivitySchemaError(
            f"sample_index must be >= 0, got {sample_index}"
        )
    step = rows_per_block(factor_count)
    offset = int(matrix_offset)
    if offset < 0 or offset >= step:
        raise SensitivitySchemaError(
            f"matrix_offset {offset} out of range for k+2={step}"
        )
    return int(sample_index) * step + offset


def validate_matrix_fields(
    *,
    matrix_kind: str,
    factor_index: int | None,
    factor_name: str | None,
    factor_names: Sequence[str],
) -> None:
    """Enforce conditional nullability rules for matrix identity fields."""
    kind = str(matrix_kind)
    names = tuple(str(name) for name in factor_names)
    if kind in {MATRIX_KIND_A, MATRIX_KIND_B}:
        if factor_index is not None or factor_name is not None:
            raise SensitivitySchemaError(
                f"matrix_kind={kind!r} requires factor_index/name to be null"
            )
        return
    if kind != MATRIX_KIND_AB:
        raise SensitivitySchemaError(f"unknown matrix_kind {kind!r}")
    if factor_index is None or factor_name is None:
        raise SensitivitySchemaError(
            "matrix_kind='AB' requires factor_index and factor_name"
        )
    index = int(factor_index)
    if index < 0 or index >= len(names):
        raise SensitivitySchemaError(
            f"factor_index {index} out of range for {len(names)} factors"
        )
    if str(factor_name) != names[index]:
        raise SensitivitySchemaError(
            f"factor_name {factor_name!r} does not match factor_names[{index}]="
            f"{names[index]!r}"
        )


def _param_arrow_type(sample: Any, *, pa: Any) -> Any:
    if isinstance(sample, bool):
        raise SensitivitySchemaError(
            f"boolean parameter values are not supported: {sample!r}"
        )
    if isinstance(sample, int) and not isinstance(sample, bool):
        return pa.int64()
    if isinstance(sample, float):
        return pa.float64()
    if isinstance(sample, str):
        return pa.string()
    raise SensitivitySchemaError(
        f"unsupported parameter sample type {type(sample).__name__}: {sample!r}"
    )


def _canonical_schema_description(schema: Any) -> str:
    fields = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]
    return json.dumps(
        {"fields": fields},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def compute_schema_hash(schema: Any) -> str:
    """Return the SHA-256 hex digest of a canonical schema description."""
    payload = _canonical_schema_description(schema).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_sensitivity_schema(
    *,
    var_names: Sequence[str],
    factor_names: Sequence[str],
    model: Mapping[str, Mapping[str, Any]],
    constraint_names: Sequence[str],
) -> SensitivitySchema:
    """
    Build the stable Parquet schema for one sensitivity problem snapshot.
    """
    pa = import_pyarrow()

    variables = tuple(str(name) for name in var_names)
    factors = tuple(str(name) for name in factor_names)
    if not variables:
        raise SensitivitySchemaError("var_names must be non-empty")
    if not factors:
        raise SensitivitySchemaError("factor_names must be non-empty")
    if len(set(factors)) != len(factors):
        raise SensitivitySchemaError(f"factor_names must be unique, got {factors!r}")
    missing_factors = [name for name in factors if name not in variables]
    if missing_factors:
        raise SensitivitySchemaError(
            f"factor_names not in var_names: {missing_factors}"
        )

    model_vars = {
        name for name, meta in model.items() if meta.get("role") == "var"
    }
    if set(variables) != model_vars:
        raise SensitivitySchemaError(
            "var_names do not match model variables: "
            f"{list(variables)!r} vs {sorted(model_vars)!r}"
        )

    param_names = tuple(str(name) for name in model.keys())
    constraints = tuple(str(name) for name in constraint_names)
    if len(set(constraints)) != len(constraints):
        raise SensitivitySchemaError(
            f"constraint_names must be unique, got {constraints!r}"
        )

    fields: list[Any] = [
        pa.field("global_index", pa.int64(), nullable=False),
        pa.field("shard_index", pa.int32(), nullable=False),
        pa.field("sample_index", pa.int64(), nullable=False),
        pa.field("matrix_offset", pa.int32(), nullable=False),
        pa.field("matrix_kind", pa.string(), nullable=False),
        pa.field("factor_index", pa.int32(), nullable=True),
        pa.field("factor_name", pa.string(), nullable=True),
        pa.field("problem_id", pa.string(), nullable=False),
        pa.field("evaluation_time_s", pa.float64(), nullable=False),
        pa.field("analysis_ok", pa.bool_(), nullable=False),
        pa.field("constraint_feasible", pa.bool_(), nullable=False),
        pa.field("feasible", pa.bool_(), nullable=False),
        pa.field("y", pa.float64(), nullable=False),
        pa.field("y_p", pa.float64(), nullable=False),
        pa.field("penalty_sum", pa.float64(), nullable=False),
        pa.field("penalty_factor", pa.float64(), nullable=False),
    ]

    for name in variables:
        fields.append(pa.field(f"idx__{name}", pa.int32(), nullable=False))

    for name in param_names:
        values = model[name].get("values")
        if not values:
            raise SensitivitySchemaError(
                f"parameter {name!r} has an empty catalogue; cannot derive type"
            )
        fields.append(
            pa.field(
                f"param__{name}",
                _param_arrow_type(values[0], pa=pa),
                nullable=True,
            )
        )

    for name in constraints:
        fields.append(pa.field(f"util__{name}", pa.float64(), nullable=True))
        fields.append(pa.field(f"violation__{name}", pa.float64(), nullable=True))
        fields.append(pa.field(f"penalty__{name}", pa.float64(), nullable=True))

    fields.extend(
        [
            pa.field("analysis_error_stage", pa.string(), nullable=True),
            pa.field("analysis_error_message", pa.string(), nullable=True),
            pa.field("analysis_error_exception_type", pa.string(), nullable=True),
            pa.field("analysis_error_material_id", pa.string(), nullable=True),
            pa.field("analysis_error_json", pa.string(), nullable=True),
            pa.field(
                "remapped_utilizations",
                pa.list_(pa.string()),
                nullable=False,
            ),
            pa.field(
                "cost_components_missing",
                pa.list_(pa.string()),
                nullable=False,
            ),
        ]
    )

    schema = pa.schema(fields)
    return SensitivitySchema(
        schema=schema,
        schema_hash=compute_schema_hash(schema),
        var_names=variables,
        factor_names=factors,
        param_names=param_names,
        constraint_names=constraints,
    )


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise SensitivitySchemaError(f"{field} must be bool, got {value!r}")
    return value


def _require_float(value: Any, *, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SensitivitySchemaError(
            f"{field} must be float-compatible, got {value!r}"
        ) from exc


def flatten_sensitivity_row(
    *,
    schema_info: SensitivitySchema,
    global_index: int,
    shard_index: int,
    sample_index: int,
    matrix_offset: int,
    problem_id: str,
    evaluation_time_s: float,
    catalogue_indices: Sequence[int],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Flatten one analysis result into a sensitivity schema-aligned row.
    """
    unknown = set(result.keys()) - _SCIENTIFIC_RESULT_KEYS
    if unknown:
        raise SensitivitySchemaError(
            f"unexpected scientific keys in analysis result: {sorted(unknown)}"
        )
    missing = _SCIENTIFIC_RESULT_KEYS - set(result.keys())
    if missing:
        raise SensitivitySchemaError(
            f"analysis result missing required keys: {sorted(missing)}"
        )

    if len(catalogue_indices) != len(schema_info.var_names):
        raise SensitivitySchemaError(
            f"expected {len(schema_info.var_names)} catalogue indices, "
            f"got {len(catalogue_indices)}"
        )

    factor_count = len(schema_info.factor_names)
    expected_global = global_index_for(
        sample_index=sample_index,
        matrix_offset=matrix_offset,
        factor_count=factor_count,
    )
    if int(global_index) != expected_global:
        raise SensitivitySchemaError(
            f"global_index {global_index} != sample_index*(k+2)+offset "
            f"({expected_global})"
        )

    matrix_kind, factor_index, factor_name = matrix_role_for_offset(
        matrix_offset,
        factor_count=factor_count,
        factor_names=schema_info.factor_names,
    )
    validate_matrix_fields(
        matrix_kind=matrix_kind,
        factor_index=factor_index,
        factor_name=factor_name,
        factor_names=schema_info.factor_names,
    )

    analysis_ok = _require_bool(result["analysis_ok"], field="analysis_ok")
    constraint_feasible = _require_bool(
        result["constraint_feasible"], field="constraint_feasible"
    )
    feasible = _require_bool(result["feasible"], field="feasible")
    expected_feasible = bool(analysis_ok and constraint_feasible)
    if feasible is not expected_feasible:
        raise SensitivitySchemaError(
            "feasible must equal analysis_ok and constraint_feasible "
            f"(got feasible={feasible}, analysis_ok={analysis_ok}, "
            f"constraint_feasible={constraint_feasible})"
        )

    row: dict[str, Any] = {
        "global_index": int(global_index),
        "shard_index": int(shard_index),
        "sample_index": int(sample_index),
        "matrix_offset": int(matrix_offset),
        "matrix_kind": matrix_kind,
        "factor_index": None if factor_index is None else int(factor_index),
        "factor_name": None if factor_name is None else str(factor_name),
        "problem_id": str(problem_id),
        "evaluation_time_s": _require_float(
            evaluation_time_s, field="evaluation_time_s"
        ),
        "analysis_ok": analysis_ok,
        "constraint_feasible": constraint_feasible,
        "feasible": feasible,
        "y": _require_float(result["y"], field="y"),
        "y_p": _require_float(result["y_p"], field="y_p"),
        "penalty_sum": _require_float(result["penalty_sum"], field="penalty_sum"),
        "penalty_factor": _require_float(
            result["penalty_factor"], field="penalty_factor"
        ),
    }

    for name, index in zip(schema_info.var_names, catalogue_indices):
        row[f"idx__{name}"] = int(index)

    params = result["params"]
    if not isinstance(params, Mapping):
        raise SensitivitySchemaError("params must be a mapping")
    for name in schema_info.param_names:
        row[f"param__{name}"] = params.get(name, None)

    utils = result["constraint_values"]
    violations = result["violations_"]
    penalties = result["penalties_"]
    if not isinstance(utils, Mapping):
        raise SensitivitySchemaError("constraint_values must be a mapping")
    if not isinstance(violations, Mapping):
        raise SensitivitySchemaError("violations_ must be a mapping")
    if not isinstance(penalties, Mapping):
        raise SensitivitySchemaError("penalties_ must be a mapping")

    for name in schema_info.constraint_names:
        util = utils.get(name, None)
        viol = violations.get(name, None)
        pen = penalties.get(name, None)
        row[f"util__{name}"] = (
            None if util is None else _require_float(util, field=f"util__{name}")
        )
        row[f"violation__{name}"] = (
            None if viol is None else _require_float(viol, field=f"violation__{name}")
        )
        row[f"penalty__{name}"] = (
            None if pen is None else _require_float(pen, field=f"penalty__{name}")
        )

    error = result["analysis_error"]
    if error is None:
        row["analysis_error_stage"] = None
        row["analysis_error_message"] = None
        row["analysis_error_exception_type"] = None
        row["analysis_error_material_id"] = None
        row["analysis_error_json"] = None
    elif isinstance(error, Mapping):
        row["analysis_error_stage"] = (
            None if error.get("stage") is None else str(error.get("stage"))
        )
        row["analysis_error_message"] = (
            None if error.get("message") is None else str(error.get("message"))
        )
        row["analysis_error_exception_type"] = (
            None
            if error.get("exception_type") is None
            else str(error.get("exception_type"))
        )
        row["analysis_error_material_id"] = (
            None
            if error.get("material_id") is None
            else str(error.get("material_id"))
        )
        row["analysis_error_json"] = json.dumps(
            error,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=True,
        )
    else:
        raise SensitivitySchemaError(
            f"analysis_error must be a mapping or None, got {type(error).__name__}"
        )

    remapped = result["remapped_utilizations"]
    missing_costs = result["cost_components_missing"]
    if not isinstance(remapped, list) or not all(isinstance(x, str) for x in remapped):
        raise SensitivitySchemaError("remapped_utilizations must be a list of strings")
    if not isinstance(missing_costs, list) or not all(
        isinstance(x, str) for x in missing_costs
    ):
        raise SensitivitySchemaError(
            "cost_components_missing must be a list of strings"
        )
    row["remapped_utilizations"] = list(remapped)
    row["cost_components_missing"] = list(missing_costs)

    for key in ("y", "y_p", "penalty_sum", "penalty_factor", "evaluation_time_s"):
        value = row[key]
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            row[key] = float(value)

    expected_names = {field.name for field in schema_info.schema}
    if set(row.keys()) != expected_names:
        raise SensitivitySchemaError(
            "flattened row fields do not match schema: "
            f"extra={sorted(set(row) - expected_names)} "
            f"missing={sorted(expected_names - set(row))}"
        )
    return row


def schema_document(schema_info: SensitivitySchema) -> dict[str, Any]:
    """Return a JSON-serialisable description of ``schema_info``."""
    fields = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema_info.schema
    ]
    return {
        "schema_hash": schema_info.schema_hash,
        "var_names": list(schema_info.var_names),
        "factor_names": list(schema_info.factor_names),
        "param_names": list(schema_info.param_names),
        "constraint_names": list(schema_info.constraint_names),
        "fields": fields,
    }


def write_dataset_schema_json(
    dataset_dir: Path,
    schema_info: SensitivitySchema,
) -> Path:
    """
    Atomically write ``dataset_dir/schema.json`` and ensure ``parts/`` exists.
    """
    from core.execution_core.atomic_io import write_json_atomic

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "parts").mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "schema.json"
    write_json_atomic(path, schema_document(schema_info))
    return path


def schema_hash_from_document(document: Mapping[str, Any]) -> str:
    """Extract ``schema_hash`` from a sealed schema document."""
    value = document.get("schema_hash")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("schema.json missing non-empty schema_hash")
    return value.strip()
