"""
Stable Parquet schema for brute-force evaluation rows.

Builds an explicit ``pyarrow.Schema`` from a snapshotted parameter model and
constraint-default superset, flattens analysis result dictionaries into that
schema, and computes a deterministic schema SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.ioh_core.workflows.brute_force.deps import import_pyarrow
from core.ioh_core.workflows.brute_force.models import IndexSpace

# First-class analysis keys that must map into the schema (or be rejected).
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


class BruteForceSchemaError(ValueError):
    """
    Raised when a Parquet schema cannot be built or a row violates it.
    """


@dataclass(frozen=True)
class BruteForceSchema:
    """
    Explicit Arrow schema plus derived metadata for one problem snapshot.

    Attributes
    ----------
    schema : pyarrow.Schema
        Column layout used by the Parquet writer.
    schema_hash : str
        Deterministic SHA-256 hex digest of the canonical schema description.
    var_names : tuple of str
        Optimisation variable names (``idx__*`` order).
    param_names : tuple of str
        Full model parameter names (``param__*`` order).
    constraint_names : tuple of str
        Constraint-default names (``util__*`` / ``violation__*`` / ``penalty__*``).
    """

    schema: Any
    schema_hash: str
    var_names: tuple[str, ...]
    param_names: tuple[str, ...]
    constraint_names: tuple[str, ...]


def _param_arrow_type(sample: Any, *, pa: Any) -> Any:
    """
    Map a model catalogue sample to an Arrow field type.

    Parameters
    ----------
    sample : Any
        One catalogue value from the parameter model.
    pa : module
        Imported ``pyarrow`` module.

    Returns
    -------
    pyarrow.DataType
        Arrow type for ``param__*`` columns.
    """
    if isinstance(sample, bool):
        raise BruteForceSchemaError(
            f"boolean parameter values are not supported in Parquet schema: {sample!r}"
        )
    if isinstance(sample, int) and not isinstance(sample, bool):
        return pa.int64()
    if isinstance(sample, float):
        return pa.float64()
    if isinstance(sample, str):
        return pa.string()
    raise BruteForceSchemaError(
        f"unsupported parameter sample type {type(sample).__name__}: {sample!r}"
    )


def _canonical_schema_description(schema: Any) -> str:
    """
    Return a stable JSON description of an Arrow schema for hashing.

    Parameters
    ----------
    schema : pyarrow.Schema
        Schema to serialise.

    Returns
    -------
    str
        Compact JSON with sorted keys.
    """
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
    """
    Return the SHA-256 hex digest of a canonical schema description.

    Parameters
    ----------
    schema : pyarrow.Schema
        Schema to fingerprint.

    Returns
    -------
    str
        Lowercase hex digest.
    """
    payload = _canonical_schema_description(schema).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_brute_force_schema(
    *,
    space: IndexSpace,
    model: Mapping[str, Mapping[str, Any]],
    constraint_names: Sequence[str],
) -> BruteForceSchema:
    """
    Build the stable Parquet schema for one snapshotted problem.

    Parameters
    ----------
    space : IndexSpace
        Catalogue index space (defines ``idx__*`` columns).
    model : mapping
        Full parameter model (all roles) from the snapshotted CSVs.
    constraint_names : sequence of str
        Stable constraint-default names (superset used for ``util__*`` etc.).

    Returns
    -------
    BruteForceSchema
        Explicit schema, hash, and name metadata.

    Raises
    ------
    BruteForceSchemaError
        If the model/space/constraints are inconsistent or unsupported.
    BruteForceDependencyError
        If PyArrow is not installed.
    """
    pa = import_pyarrow()

    model_vars = {
        name for name, meta in model.items() if meta.get("role") == "var"
    }
    if set(space.var_names) != model_vars:
        raise BruteForceSchemaError(
            "IndexSpace var_names do not match model variables: "
            f"{list(space.var_names)!r} vs {sorted(model_vars)!r}"
        )

    param_names = tuple(str(name) for name in model.keys())
    constraints = tuple(str(name) for name in constraint_names)
    if len(set(constraints)) != len(constraints):
        raise BruteForceSchemaError(
            f"constraint_names must be unique, got {constraints!r}"
        )

    fields: list[Any] = [
        pa.field("global_index", pa.int64(), nullable=False),
        pa.field("shard_index", pa.int32(), nullable=False),
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

    for name in space.var_names:
        fields.append(pa.field(f"idx__{name}", pa.int32(), nullable=False))

    for name in param_names:
        values = model[name].get("values")
        if not values:
            raise BruteForceSchemaError(
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
    return BruteForceSchema(
        schema=schema,
        schema_hash=compute_schema_hash(schema),
        var_names=tuple(space.var_names),
        param_names=param_names,
        constraint_names=constraints,
    )


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise BruteForceSchemaError(f"{field} must be bool, got {value!r}")
    return value


def _require_float(value: Any, *, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BruteForceSchemaError(f"{field} must be float-compatible, got {value!r}") from exc


def flatten_evaluation_row(
    *,
    schema_info: BruteForceSchema,
    global_index: int,
    shard_index: int,
    problem_id: str,
    evaluation_time_s: float,
    catalogue_indices: Sequence[int],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Flatten one analysis result into a schema-aligned row mapping.

    Parameters
    ----------
    schema_info : BruteForceSchema
        Target schema metadata.
    global_index : int
        Linear catalogue address.
    shard_index : int
        Shard ordinal.
    problem_id : str
        Ordinary problem ID.
    evaluation_time_s : float
        Wall seconds for this evaluation.
    catalogue_indices : sequence of int
        Catalogue index vector aligned with ``schema_info.var_names``.
    result : mapping
        Analysis result dictionary.

    Returns
    -------
    dict
        Row values keyed by schema field names.

    Raises
    ------
    BruteForceSchemaError
        If required keys are missing, unknown scientific keys appear, lengths
        disagree, or ``feasible`` is inconsistent with its components.
    """
    unknown = set(result.keys()) - _SCIENTIFIC_RESULT_KEYS
    if unknown:
        raise BruteForceSchemaError(
            f"unexpected scientific keys in analysis result: {sorted(unknown)}"
        )
    missing = _SCIENTIFIC_RESULT_KEYS - set(result.keys())
    if missing:
        raise BruteForceSchemaError(
            f"analysis result missing required keys: {sorted(missing)}"
        )

    if len(catalogue_indices) != len(schema_info.var_names):
        raise BruteForceSchemaError(
            f"expected {len(schema_info.var_names)} catalogue indices, "
            f"got {len(catalogue_indices)}"
        )

    analysis_ok = _require_bool(result["analysis_ok"], field="analysis_ok")
    constraint_feasible = _require_bool(
        result["constraint_feasible"], field="constraint_feasible"
    )
    feasible = _require_bool(result["feasible"], field="feasible")
    expected_feasible = bool(analysis_ok and constraint_feasible)
    if feasible is not expected_feasible:
        raise BruteForceSchemaError(
            "feasible must equal analysis_ok and constraint_feasible "
            f"(got feasible={feasible}, analysis_ok={analysis_ok}, "
            f"constraint_feasible={constraint_feasible})"
        )

    row: dict[str, Any] = {
        "global_index": int(global_index),
        "shard_index": int(shard_index),
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
        raise BruteForceSchemaError("params must be a mapping")
    for name in schema_info.param_names:
        row[f"param__{name}"] = params.get(name, None)

    utils = result["constraint_values"]
    violations = result["violations_"]
    penalties = result["penalties_"]
    if not isinstance(utils, Mapping):
        raise BruteForceSchemaError("constraint_values must be a mapping")
    if not isinstance(violations, Mapping):
        raise BruteForceSchemaError("violations_ must be a mapping")
    if not isinstance(penalties, Mapping):
        raise BruteForceSchemaError("penalties_ must be a mapping")

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
        raise BruteForceSchemaError(
            f"analysis_error must be a mapping or None, got {type(error).__name__}"
        )

    remapped = result["remapped_utilizations"]
    missing_costs = result["cost_components_missing"]
    if not isinstance(remapped, list) or not all(isinstance(x, str) for x in remapped):
        raise BruteForceSchemaError("remapped_utilizations must be a list of strings")
    if not isinstance(missing_costs, list) or not all(
        isinstance(x, str) for x in missing_costs
    ):
        raise BruteForceSchemaError(
            "cost_components_missing must be a list of strings"
        )
    row["remapped_utilizations"] = list(remapped)
    row["cost_components_missing"] = list(missing_costs)

    # Preserve IEEE non-finites already stored as floats; no conversion to null.
    for key in ("y", "y_p", "penalty_sum", "penalty_factor", "evaluation_time_s"):
        value = row[key]
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            row[key] = float(value)

    expected_names = {field.name for field in schema_info.schema}
    if set(row.keys()) != expected_names:
        raise BruteForceSchemaError(
            "flattened row fields do not match schema: "
            f"extra={sorted(set(row) - expected_names)} "
            f"missing={sorted(expected_names - set(row))}"
        )
    return row
