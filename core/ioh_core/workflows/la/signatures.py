"""
Canonical sampling-design and analysis signature hashes for ELA.

Compact fields on :class:`FeatureResult` are SHA-256 of canonical JSON,
**parallel to** ``config_hash`` / BF ``schema_hash``. Do not import
``optimization_runs.snapshot`` or ``workflows.sampling``.

``repetition`` and ``seed`` are replicate identity: they belong on the
row, not in the sampling-design payload. There is no ``n_used``.

Importing this module must not load DuckDB, pflacco, or SALib.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.ioh_core.workflows.la.models import (
    ElaError,
    canonical_config_json,
    compute_config_hash,
)

_SAMPLING_KEYS = (
    "problem_id",
    "schema_hash",
    "method",
    "sample_size",
    "scramble",
    "projection",
    "delete_duplicates",
    "var_names",
    "lower_bounds",
    "upper_bounds",
)
_FORBIDDEN_SAMPLING = frozenset(
    {
        "repetition",
        "seed",
        "n_used",
        "n_evaluate",
        "n_unique",
        "n_requested",
        "n_projected",
        "n_duplicates",
        "array_sha256",
        "numpy_version",
        "scipy_version",
    }
)
_ANALYSIS_KEYS = (
    "family",
    "landscape",
    "constraint_names",
    "x_view",
    "response_view",
    "config",
)


def hash_signature_payload(payload: Mapping[str, Any]) -> str:
    """Return SHA-256 hex of canonical JSON (same helper as ``config_hash``)."""
    return compute_config_hash(canonical_config_json(payload))


def sampling_design_payload(
    *,
    problem_id: str,
    schema_hash: str,
    method: str,
    sample_size: int,
    scramble: bool,
    projection: str,
    delete_duplicates: bool,
    var_names: Sequence[str],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> dict[str, Any]:
    """
    Canonical sampling-design grouping payload.

    Omits replicate identity (``repetition``, ``seed``) and outcome counts.
    """
    names = _as_str_list(var_names, field="var_names")
    lo = _as_int_list(lower_bounds, field="lower_bounds")
    hi = _as_int_list(upper_bounds, field="upper_bounds")
    if len(lo) != len(names) or len(hi) != len(names):
        raise ElaError(
            "sampling signature bounds must match var_names length"
        )
    payload = {
        "problem_id": _require_nonempty(problem_id, field="problem_id"),
        "schema_hash": _require_nonempty(schema_hash, field="schema_hash"),
        "method": _require_nonempty(method, field="method"),
        "sample_size": _require_positive_int(sample_size, field="sample_size"),
        "scramble": _require_bool(scramble, field="scramble"),
        "projection": _require_nonempty(projection, field="projection"),
        "delete_duplicates": _require_bool(
            delete_duplicates, field="delete_duplicates"
        ),
        "var_names": names,
        "lower_bounds": lo,
        "upper_bounds": hi,
    }
    extra = set(payload) - set(_SAMPLING_KEYS)
    if extra:
        raise ElaError(f"sampling signature has unexpected keys {sorted(extra)}")
    forbidden = set(payload) & _FORBIDDEN_SAMPLING
    if forbidden:
        raise ElaError(
            "sampling_design_signature must not include "
            f"{sorted(forbidden)}"
        )
    return payload


def analysis_signature_payload(
    *,
    family: str,
    landscape: str,
    constraint_names: Sequence[str],
    x_view: str,
    response_view: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Canonical analysis grouping payload for one family × landscape.

    ``response_view`` is the analysis intent (sampled pflacco: ``minmax``),
    not the per-row ``n/a`` used for constant or occupancy status.
    ``config`` holds only that family's knobs.
    """
    if not isinstance(config, Mapping) or isinstance(config, (str, bytes)):
        raise ElaError("analysis signature config must be a mapping")
    payload = {
        "family": _require_nonempty(family, field="family"),
        "landscape": _require_nonempty(landscape, field="landscape"),
        "constraint_names": _as_str_list(
            constraint_names, field="constraint_names"
        ),
        "x_view": _require_nonempty(x_view, field="x_view"),
        "response_view": _require_nonempty(response_view, field="response_view"),
        "config": _json_ready(dict(config)),
    }
    extra = set(payload) - set(_ANALYSIS_KEYS)
    if extra:
        raise ElaError(f"analysis signature has unexpected keys {sorted(extra)}")
    if "n_used" in payload["config"]:
        raise ElaError("analysis signature config must not include n_used")
    return payload


def sampling_design_signature(**kwargs: Any) -> str:
    """SHA-256 hex of :func:`sampling_design_payload`."""
    return hash_signature_payload(sampling_design_payload(**kwargs))


def analysis_signature(**kwargs: Any) -> str:
    """SHA-256 hex of :func:`analysis_signature_payload`."""
    return hash_signature_payload(analysis_signature_payload(**kwargs))


def load_schema_hash(experiment_dir: Path) -> str:
    """Read ``schema_hash`` from ``dataset/schema.json``."""
    document = _read_json_object(
        Path(experiment_dir) / "dataset" / "schema.json",
        what="dataset/schema.json",
    )
    return _require_nonempty(document.get("schema_hash"), field="schema_hash")


def load_sampling_design_fields(
    experiment_dir: Path,
    *,
    repetition: int,
) -> dict[str, Any]:
    """
    Read sealed ``design.json`` fields needed for the sampling signature.

    Also returns ``seed`` (row provenance only) and optional ``problem_id``.
    """
    if (
        not isinstance(repetition, int)
        or isinstance(repetition, bool)
        or repetition < 0
    ):
        raise ElaError(f"repetition must be >= 0, got {repetition!r}")
    path = (
        Path(experiment_dir)
        / "inputs"
        / "sampling"
        / f"rep_{int(repetition):03d}"
        / "design.json"
    )
    document = _read_json_object(path, what="design.json")
    return {
        "method": document.get("method"),
        "sample_size": document.get("sample_size"),
        "scramble": document.get("scramble"),
        "projection": document.get("index_mapping"),
        "delete_duplicates": document.get("delete_duplicates"),
        "var_names": document.get("var_names"),
        "lower_bounds": document.get("lower_bounds"),
        "upper_bounds": document.get("upper_bounds"),
        "seed": document.get("seed"),
        "problem_id": document.get("problem_id"),
        "repetition": document.get("repetition"),
    }


def _read_json_object(path: Path, *, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise ElaError(f"missing {what} at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ElaError(f"invalid {what}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ElaError(f"{what} must be a JSON object")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _require_nonempty(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ElaError(f"{field} must be a non-empty string")
    return value.strip()


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ElaError(f"{field} must be a bool, got {value!r}")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ElaError(f"{field} must be a positive int, got {value!r}")
    return value


def _as_str_list(values: Sequence[str], *, field: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ElaError(f"{field} must be a sequence of strings")
    out = [str(item) for item in values]
    if not out or any(not item.strip() for item in out):
        raise ElaError(f"{field} must be a non-empty list of strings")
    return out


def _as_int_list(values: Sequence[int], *, field: str) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ElaError(f"{field} must be a sequence of ints")
    out: list[int] = []
    for item in values:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ElaError(f"{field} must be a sequence of ints")
        out.append(int(item))
    if not out:
        raise ElaError(f"{field} must be a non-empty list of ints")
    return out
