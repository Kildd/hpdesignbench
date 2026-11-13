"""
Group tidy ELA feature rows and summarize compatible repetitions.

Median / Q1 / Q3 / IQR may only run inside one
``(sampling_design_signature, analysis_signature)`` group. This is not
an ``hdb-experiment`` aggregator.

Importing this module must not load DuckDB, pflacco, PyArrow, or SALib.
"""
from __future__ import annotations

import io
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.execution_core.atomic_io import write_bytes_atomic, write_text_atomic
from core.ioh_core.workflows.la.deps import import_pyarrow
from core.ioh_core.workflows.la.models import (
    ElaError,
    FeatureResult,
    STATUS_OK,
    canonical_config_json,
    compute_config_hash,
)

SUMMARY_PARQUET = "summary.parquet"
SUMMARY_DIR = "summary"
SUCCESS_MARKER = "_SUCCESS"
FEATURES_PARQUET = "features.parquet"
_QUANTILE_METHOD = "linear"
_RUNTIME_SUFFIX = ".costs_runtime"


@dataclass(frozen=True)
class SignaturePair:
    """One scientifically poolable grouping key."""

    sampling_design_signature: str
    analysis_signature: str


@dataclass(frozen=True)
class FeatureSummary:
    """One feature's summary inside a signature pair."""

    sampling_design_signature: str
    analysis_signature: str
    family: str
    landscape: str
    feature_name: str
    count: int
    median: float | None
    q1: float | None
    q3: float | None
    iqr: float | None
    min_value: float | None
    max_value: float | None


@dataclass(frozen=True)
class RfbWalkSummary:
    """Walk-level RFB× rows plus min / median / max of ``rfb_x``."""

    walks: tuple[Any, ...]
    count: int
    min_value: float
    median: float
    max_value: float


def group_ela_rows(
    rows: Sequence[FeatureResult],
    *,
    allow_multiple_groups: bool = False,
) -> Mapping[SignaturePair, tuple[FeatureResult, ...]]:
    """
    Partition rows by identical sampling-design and analysis signatures.

    Parameters
    ----------
    rows :
        Tidy feature rows. Order within each group follows ``rows``.
    allow_multiple_groups :
        If false (default), more than one signature pair is an error.
        If true, return one group per pair and never mix them.

    Returns
    -------
    mapping
        ``SignaturePair`` → rows, in first-seen pair order.

    Raises
    ------
    ElaError
        Empty input, missing signatures, non-bool flag, or (when
        ``allow_multiple_groups`` is false) more than one pair.
    """
    if not isinstance(allow_multiple_groups, bool):
        raise ElaError("allow_multiple_groups must be a bool")
    if not rows:
        raise ElaError("ELA grouping requires a non-empty row list")

    grouped: OrderedDict[SignaturePair, list[FeatureResult]] = OrderedDict()
    for index, row in enumerate(rows):
        if not isinstance(row, FeatureResult):
            raise ElaError(
                f"ELA grouping requires FeatureResult rows, got {type(row)!r} "
                f"at index {index}"
            )
        pair = _require_signature_pair(row, index=index)
        grouped.setdefault(pair, []).append(row)

    if len(grouped) > 1 and not allow_multiple_groups:
        pairs = [
            (
                pair.sampling_design_signature,
                pair.analysis_signature,
            )
            for pair in grouped
        ]
        raise ElaError(
            "ELA aggregation requires a single "
            "(sampling_design_signature, analysis_signature) pair; "
            f"got {len(grouped)} pairs {pairs}. "
            "Pass allow_multiple_groups=True for separate groups; "
            "do not mix landscapes, knobs, keep-lists, or designs."
        )
    return {pair: tuple(items) for pair, items in grouped.items()}


def summarize_ela_rows(
    rows: Sequence[FeatureResult],
    *,
    allow_multiple_groups: bool = False,
) -> tuple[FeatureSummary, ...]:
    """
    Median / Q1 / Q3 / IQR per ``feature_name`` inside each signature pair.

    Pools ``status=ok`` finite values only. Skips ``*.costs_runtime``.
    ``count < 2`` leaves ``q1`` / ``q3`` / ``iqr`` null (no fake CIs).
    """
    grouped = group_ela_rows(
        rows, allow_multiple_groups=allow_multiple_groups
    )
    summaries: list[FeatureSummary] = []
    for pair, group_rows in grouped.items():
        summaries.extend(_summarize_group(pair, group_rows))
    return tuple(summaries)


def summarize_rfb_walks(walks: Sequence[Any]) -> RfbWalkSummary:
    """
    Keep every walk and summarize ``rfb_x`` as min / median / max.

    Does not invent Q1/Q3 intervals. ``count < 1`` fails closed.
    """
    if not walks:
        raise ElaError("RFB× summary requires a non-empty walk list")
    values: list[float] = []
    seen: set[int] = set()
    for walk in walks:
        walk_id = getattr(walk, "walk_id", None)
        raw = getattr(walk, "rfb_x", None)
        if not isinstance(walk_id, int) or isinstance(walk_id, bool):
            raise ElaError("RFB× walk_id must be an int")
        if walk_id in seen:
            raise ElaError(f"duplicate RFB× walk_id {walk_id}")
        seen.add(walk_id)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ElaError("RFB× rfb_x must be a finite float")
        number = float(raw)
        if not math.isfinite(number):
            raise ElaError("RFB× rfb_x must be finite")
        values.append(number)
    array = np.asarray(values, dtype=float)
    return RfbWalkSummary(
        walks=tuple(walks),
        count=int(array.size),
        min_value=float(np.min(array)),
        median=float(np.quantile(array, 0.5, method=_QUANTILE_METHOD)),
        max_value=float(np.max(array)),
    )


def load_feature_parquet(path: Path) -> list[FeatureResult]:
    """Load tidy feature rows written by sampled orchestration."""
    parquet = _import_pyarrow_parquet()
    table = parquet.read_table(Path(path))
    records = table.to_pylist()
    if not records:
        raise ElaError(f"ELA feature parquet is empty: {path}")
    return [_feature_from_record(record, index=index) for index, record in enumerate(records)]


def write_summary_parquet(
    path: Path, summaries: Sequence[FeatureSummary]
) -> Path:
    """Atomically write summary rows as Parquet."""
    if not summaries:
        raise ElaError(f"refusing to publish empty ELA summary at {path}")
    pa = import_pyarrow()
    parquet = _import_pyarrow_parquet()
    table = pa.Table.from_pylist([_summary_record(item) for item in summaries])
    buffer = io.BytesIO()
    parquet.write_table(table, buffer)
    write_bytes_atomic(Path(path), buffer.getvalue())
    return Path(path)


def publish_family_summary(
    family_dir: Path,
    summaries: Sequence[FeatureSummary],
) -> Path:
    """Write ``summary.parquet`` then ``summary/_SUCCESS``."""
    family_dir = Path(family_dir)
    parquet_path = family_dir / SUMMARY_PARQUET
    write_summary_parquet(parquet_path, summaries)
    marker_dir = family_dir / SUMMARY_DIR
    marker_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(marker_dir / SUCCESS_MARKER, "ok\n")
    return parquet_path


def _summarize_group(
    pair: SignaturePair,
    rows: Sequence[FeatureResult],
) -> list[FeatureSummary]:
    by_name: OrderedDict[str, list[FeatureResult]] = OrderedDict()
    for row in rows:
        if _is_runtime_feature(row.feature_name):
            continue
        by_name.setdefault(row.feature_name, []).append(row)
    summaries: list[FeatureSummary] = []
    for feature_name, feature_rows in by_name.items():
        values = _pool_ok_values(feature_rows, feature_name=feature_name)
        stats = _quantile_stats(values)
        sample = feature_rows[0]
        summaries.append(
            FeatureSummary(
                sampling_design_signature=pair.sampling_design_signature,
                analysis_signature=pair.analysis_signature,
                family=sample.family,
                landscape=sample.landscape,
                feature_name=feature_name,
                count=stats["count"],
                median=stats["median"],
                q1=stats["q1"],
                q3=stats["q3"],
                iqr=stats["iqr"],
                min_value=stats["min_value"],
                max_value=stats["max_value"],
            )
        )
    return summaries


def _pool_ok_values(
    rows: Sequence[FeatureResult], *, feature_name: str
) -> list[float]:
    values: list[float] = []
    seen: set[tuple[int | None, int | None]] = set()
    for row in rows:
        if row.status != STATUS_OK:
            continue
        if row.value is None or not math.isfinite(float(row.value)):
            continue
        key = (row.repetition, row.walk_id)
        if key in seen:
            raise ElaError(
                "duplicate ok value for "
                f"feature_name={feature_name!r} repetition={row.repetition!r} "
                f"walk_id={row.walk_id!r}"
            )
        seen.add(key)
        values.append(float(row.value))
    return values


def _quantile_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "min_value": None,
            "max_value": None,
        }
    array = np.asarray(values, dtype=float)
    count = int(array.size)
    median = float(np.quantile(array, 0.5, method=_QUANTILE_METHOD))
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if count < 2:
        return {
            "count": count,
            "median": median,
            "q1": None,
            "q3": None,
            "iqr": None,
            "min_value": min_value,
            "max_value": max_value,
        }
    q1, q3 = (
        float(item)
        for item in np.quantile(array, [0.25, 0.75], method=_QUANTILE_METHOD)
    )
    return {
        "count": count,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min_value": min_value,
        "max_value": max_value,
    }


def _is_runtime_feature(feature_name: str) -> bool:
    return str(feature_name).endswith(_RUNTIME_SUFFIX)


def _require_signature_pair(row: FeatureResult, *, index: int) -> SignaturePair:
    sampling = row.sampling_design_signature
    analysis = row.analysis_signature
    if not isinstance(sampling, str) or not sampling.strip():
        raise ElaError(
            "sampling_design_signature is required for grouping "
            f"(row index {index})"
        )
    if not isinstance(analysis, str) or not analysis.strip():
        raise ElaError(
            "analysis_signature is required for grouping "
            f"(row index {index})"
        )
    return SignaturePair(
        sampling_design_signature=sampling.strip(),
        analysis_signature=analysis.strip(),
    )


def _import_pyarrow_parquet() -> Any:
    import_pyarrow()
    import importlib

    return importlib.import_module("pyarrow.parquet")


def _feature_from_record(record: Mapping[str, Any], *, index: int) -> FeatureResult:
    config_json = record.get("config_json")
    config_hash = record.get("config_hash")
    if not isinstance(config_json, str) or not config_json:
        raise ElaError(f"feature parquet row {index} missing config_json")
    if not isinstance(config_hash, str) or not config_hash:
        raise ElaError(f"feature parquet row {index} missing config_hash")
    try:
        payload = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ElaError(f"feature parquet row {index} has invalid config_json") from exc
    if not isinstance(payload, dict):
        raise ElaError(f"feature parquet row {index} config_json must be an object")
    expected = compute_config_hash(canonical_config_json(payload))
    if expected != config_hash:
        raise ElaError(
            f"feature parquet row {index} config_hash does not match config_json"
        )
    return FeatureResult(
        problem_id=str(record["problem_id"]),
        source_type=str(record["source_type"]),
        source_dir=Path(str(record["source_dir"])),
        repetition=_optional_int(record.get("repetition")),
        sample_id=_optional_str(record.get("sample_id")),
        walk_id=_optional_int(record.get("walk_id")),
        landscape=str(record["landscape"]),
        response_view=str(record["response_view"]),
        z_min=_optional_float(record.get("z_min")),
        z_max=_optional_float(record.get("z_max")),
        x_view=str(record["x_view"]),
        family=str(record["family"]),
        feature_name=str(record["feature_name"]),
        value=_optional_float(record.get("value")),
        status=str(record["status"]),
        role=str(record["role"]),
        n_input=int(record["n_input"]),
        n_unique=int(record["n_unique"]),
        runtime_s=float(record["runtime_s"]),
        library=str(record["library"]),
        library_version=str(record["library_version"]),
        config_payload=payload,
        config_json=config_json,
        config_hash=config_hash,
        sampling_design_signature=_optional_str(
            record.get("sampling_design_signature")
        ),
        analysis_signature=_optional_str(record.get("analysis_signature")),
        seed=_optional_int(record.get("seed")),
    )


def _summary_record(item: FeatureSummary) -> dict[str, Any]:
    return {
        "sampling_design_signature": item.sampling_design_signature,
        "analysis_signature": item.analysis_signature,
        "family": item.family,
        "landscape": item.landscape,
        "feature_name": item.feature_name,
        "count": item.count,
        "median": item.median,
        "q1": item.q1,
        "q3": item.q3,
        "iqr": item.iqr,
        "min": item.min_value,
        "max": item.max_value,
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ElaError(f"expected int or null, got {value!r}")
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ElaError(f"expected float or null, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        return None
    return number
