"""
Write sealed ``dataset/schema.json`` for a brute-force experiment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.execution_core.atomic_io import write_json_atomic
from core.ioh_core.workflows.brute_force.schema import BruteForceSchema


def schema_document(schema_info: BruteForceSchema) -> dict[str, Any]:
    """
    Return a JSON-serialisable description of ``schema_info``.

    Parameters
    ----------
    schema_info : BruteForceSchema
        Explicit Arrow schema plus hash metadata.

    Returns
    -------
    dict
        Document written to ``dataset/schema.json``.
    """
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
        "param_names": list(schema_info.param_names),
        "constraint_names": list(schema_info.constraint_names),
        "fields": fields,
    }


def write_dataset_schema_json(
    dataset_dir: Path,
    schema_info: BruteForceSchema,
) -> Path:
    """
    Atomically write ``dataset_dir/schema.json`` and ensure ``parts/`` exists.

    Parameters
    ----------
    dataset_dir : Path
        Dataset root under the experiment.
    schema_info : BruteForceSchema
        Schema to seal.

    Returns
    -------
    Path
        Path of the written ``schema.json``.
    """
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "parts").mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "schema.json"
    write_json_atomic(path, schema_document(schema_info))
    return path


def schema_hash_from_document(document: Mapping[str, Any]) -> str:
    """
    Extract ``schema_hash`` from a sealed schema document.
    """
    value = document.get("schema_hash")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("schema.json missing non-empty schema_hash")
    return value.strip()
