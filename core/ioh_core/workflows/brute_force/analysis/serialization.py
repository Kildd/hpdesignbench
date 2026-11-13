"""
Deterministic JSON normalization for brute-force dataset analysis reports.

Non-finite floats are encoded as explicit tagged objects so serialization can
use ``allow_nan=False``. Mapping keys are sorted where order has no scientific
meaning.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from core.execution_core.atomic_io import write_text_atomic

NONFINITE_TAG = "__hdb_nonfinite__"
_NONFINITE_NAN = "NaN"
_NONFINITE_POS_INF = "+Infinity"
_NONFINITE_NEG_INF = "-Infinity"


def tag_nonfinite(value: float) -> dict[str, str]:
    """
    Encode a non-finite float as a JSON-compatible tagged object.
    """
    if math.isnan(value):
        return {NONFINITE_TAG: _NONFINITE_NAN}
    if value > 0.0:
        return {NONFINITE_TAG: _NONFINITE_POS_INF}
    return {NONFINITE_TAG: _NONFINITE_NEG_INF}


def is_nonfinite_tag(value: Any) -> bool:
    """
    Return True when ``value`` is a non-finite tag object.
    """
    return (
        isinstance(value, Mapping)
        and set(value.keys()) == {NONFINITE_TAG}
        and value[NONFINITE_TAG]
        in {_NONFINITE_NAN, _NONFINITE_POS_INF, _NONFINITE_NEG_INF}
    )


def _as_python_scalar(value: Any) -> Any:
    """
    Convert NumPy / DuckDB / Path scalars to plain Python values.
    """
    if isinstance(value, Path):
        return str(value)
    # NumPy / pyarrow scalars expose ``item()``.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return value


def normalize_report_value(value: Any) -> Any:
    """
    Recursively normalize a report fragment to JSON-compatible Python values.

    - ``NaN`` / ``±Infinity`` become tagged objects.
    - Mapping keys become strings and are sorted.
    - Sequences become lists.
    - NumPy / DuckDB scalars are unwrapped.
    """
    value = _as_python_scalar(value)

    if value is None or isinstance(value, (str, bool)):
        return value

    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return tag_nonfinite(value)
        return float(value)

    if isinstance(value, Mapping):
        if is_nonfinite_tag(value):
            return {NONFINITE_TAG: str(value[NONFINITE_TAG])}
        return {
            str(key): normalize_report_value(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }

    if isinstance(value, (list, tuple)):
        return [normalize_report_value(item) for item in value]

    if isinstance(value, set):
        # Deterministic ordering for unordered collections.
        normalized = [normalize_report_value(item) for item in value]
        return sorted(normalized, key=lambda item: repr(item))

    # Fallback: stringify unknown objects rather than fail later in dumps.
    return str(value)


def dumps_report_json(report: Mapping[str, Any], *, indent: int = 2) -> str:
    """
    Serialize a finalized dataset report to deterministic UTF-8 JSON text.

    Parameters
    ----------
    report : Mapping
        Already-normalized or raw report object.
    indent : int, optional
        Pretty-print indent, default ``2``.

    Returns
    -------
    str
        JSON text with sorted keys, ``allow_nan=False``, and a trailing newline.
    """
    import json

    normalized = normalize_report_value(report)
    text = json.dumps(
        normalized,
        indent=indent,
        sort_keys=True,
        allow_nan=False,
        ensure_ascii=False,
    )
    return text + "\n"


def write_report_json_atomic(path: Path, report: Mapping[str, Any]) -> None:
    """
    Atomically publish a dataset JSON report.
    """
    write_text_atomic(Path(path), dumps_report_json(report))
