"""
Versioned logical codec for StructuralArtifact library records.

Typed Arrow columns carry IDs, keys, float64 M–κ arrays, and flags.
Heterogeneous frozen scalar metadata and ``build_failure_result`` use
deterministic tagged JSON only (no pickle / Python object serialization).
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from core.ioh_core.structural_artifacts.artifact import (
    StructuralArtifact,
    scientific_values_equal,
)
from slab_construction.slab_construction import MKCurveKey, SectionResultKey

SCHEMA_VERSION = 1
CODEC_VERSION = 1

TYPE_TAG = "__hdb_type__"
NONFINITE_TAG = "__hdb_nonfinite__"
_NONFINITE_NAN = "NaN"
_NONFINITE_POS_INF = "+Infinity"
_NONFINITE_NEG_INF = "-Infinity"


class StructuralCodecError(ValueError):
    """
    Raised when a structural artifact record cannot be encoded or decoded.
    """


def dumps_tagged_json(value: Any) -> str:
    """
    Serialize ``value`` to deterministic tagged JSON (``allow_nan=False``).
    """
    return json.dumps(
        encode_tagged(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def loads_tagged_json(text: str) -> Any:
    """
    Deserialize tagged JSON produced by :func:`dumps_tagged_json`.
    """
    return decode_tagged(json.loads(text))


def encode_tagged(value: Any) -> Any:
    """
    Convert nested Python values into JSON-compatible tagged forms.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return {NONFINITE_TAG: _NONFINITE_NAN}
        if math.isinf(value):
            return {
                NONFINITE_TAG: (
                    _NONFINITE_POS_INF if value > 0.0 else _NONFINITE_NEG_INF
                )
            }
        return float(value)
    if isinstance(value, Mapping):
        return {
            str(key): encode_tagged(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if isinstance(value, tuple):
        return {TYPE_TAG: "tuple", "items": [encode_tagged(item) for item in value]}
    if isinstance(value, list):
        return {TYPE_TAG: "list", "items": [encode_tagged(item) for item in value]}
    raise StructuralCodecError(
        f"unsupported value type for tagged JSON: {type(value)!r}"
    )


def decode_tagged(value: Any) -> Any:
    """
    Invert :func:`encode_tagged`.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, Mapping):
        if set(value.keys()) == {NONFINITE_TAG}:
            token = value[NONFINITE_TAG]
            if token == _NONFINITE_NAN:
                return float("nan")
            if token == _NONFINITE_POS_INF:
                return float("inf")
            if token == _NONFINITE_NEG_INF:
                return float("-inf")
            raise StructuralCodecError(f"unknown non-finite tag {token!r}")
        if set(value.keys()) == {TYPE_TAG, "items"}:
            items = value["items"]
            if not isinstance(items, list):
                raise StructuralCodecError("tagged sequence items must be a list")
            decoded = [decode_tagged(item) for item in items]
            kind = value[TYPE_TAG]
            if kind == "tuple":
                return tuple(decoded)
            if kind == "list":
                return decoded
            raise StructuralCodecError(f"unknown type tag {kind!r}")
        return {
            str(key): decode_tagged(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if isinstance(value, list):
        # Untagged lists (should be rare); preserve as list.
        return [decode_tagged(item) for item in value]
    raise StructuralCodecError(
        f"unsupported JSON value type for tagged decode: {type(value)!r}"
    )


def structural_key_digest(key: Any) -> str:
    """
    Return the SHA-256 hex digest of the canonical tagged JSON of ``key``.
    """
    payload = dumps_tagged_json(key).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _encode_section_entries(
    entries: Sequence[tuple[SectionResultKey, tuple[tuple[str, Any], ...]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for section_key, payload in entries:
        out.append(
            {
                "result_type": str(section_key.result_type),
                "x": float(section_key.x),
                "limit_state": str(section_key.limit_state),
                "n": float(section_key.n),
                "flipped": bool(section_key.flipped),
                "payload_json": dumps_tagged_json(payload),
            }
        )
    return out


def _decode_section_entries(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[SectionResultKey, tuple[tuple[str, Any], ...]], ...]:
    out: list[tuple[SectionResultKey, tuple[tuple[str, Any], ...]]] = []
    for row in rows:
        payload = loads_tagged_json(str(row["payload_json"]))
        if not isinstance(payload, tuple):
            raise StructuralCodecError("section payload must decode to a tuple")
        out.append(
            (
                SectionResultKey(
                    result_type=str(row["result_type"]),
                    x=float(row["x"]),
                    limit_state=str(row["limit_state"]),
                    n=float(row["n"]),
                    flipped=bool(row["flipped"]),
                ),
                payload,
            )
        )
    return tuple(out)


def _encode_mk_key_fields(mk_key: MKCurveKey) -> dict[str, Any]:
    return {
        "limit_state": str(mk_key.limit_state),
        "constitutive_law": str(mk_key.constitutive_law),
        "method": str(mk_key.method),
        "simplification_json": dumps_tagged_json(mk_key.simplification),
        "softening_json": dumps_tagged_json(mk_key.softening),
        "n": float(mk_key.n),
        "m_k_num_points": int(mk_key.m_k_num_points),
    }


def _decode_mk_key(row: Mapping[str, Any]) -> MKCurveKey:
    return MKCurveKey(
        limit_state=str(row["limit_state"]),
        constitutive_law=str(row["constitutive_law"]),
        method=str(row["method"]),
        simplification=loads_tagged_json(str(row["simplification_json"])),
        softening=loads_tagged_json(str(row["softening_json"])),
        n=float(row["n"]),
        m_k_num_points=int(row["m_k_num_points"]),
    )


def _encode_mk_curves(
    entries: Sequence[
        tuple[
            MKCurveKey,
            tuple[tuple[float, tuple[tuple[float, ...], tuple[float, ...]]], ...],
        ]
    ],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mk_key, curves in entries:
        curve_rows: list[dict[str, Any]] = []
        for x, (m_y, chi_y) in curves:
            curve_rows.append(
                {
                    "x": float(x),
                    "m_y": [float(v) for v in m_y],
                    "chi_y": [float(v) for v in chi_y],
                }
            )
        row = _encode_mk_key_fields(mk_key)
        row["curves"] = curve_rows
        out.append(row)
    return out


def _decode_mk_curves(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[
        MKCurveKey,
        tuple[tuple[float, tuple[tuple[float, ...], tuple[float, ...]]], ...],
    ],
    ...,
]:
    out: list[
        tuple[
            MKCurveKey,
            tuple[tuple[float, tuple[tuple[float, ...], tuple[float, ...]]], ...],
        ]
    ] = []
    for row in rows:
        mk_key = _decode_mk_key(row)
        curves = tuple(
            (
                float(curve["x"]),
                (
                    tuple(float(v) for v in curve["m_y"]),
                    tuple(float(v) for v in curve["chi_y"]),
                ),
            )
            for curve in row["curves"]
        )
        out.append((mk_key, curves))
    return tuple(out)


def _encode_mk_metadata(
    entries: Sequence[tuple[MKCurveKey, tuple[tuple[str, Any], ...]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mk_key, payload in entries:
        row = _encode_mk_key_fields(mk_key)
        row["payload_json"] = dumps_tagged_json(payload)
        out.append(row)
    return out


def _decode_mk_metadata(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[MKCurveKey, tuple[tuple[str, Any], ...]], ...]:
    out: list[tuple[MKCurveKey, tuple[tuple[str, Any], ...]]] = []
    for row in rows:
        payload = loads_tagged_json(str(row["payload_json"]))
        if not isinstance(payload, tuple):
            raise StructuralCodecError("mk_metadata payload must decode to a tuple")
        out.append((_decode_mk_key(row), payload))
    return tuple(out)


def encode_record(
    artifact: StructuralArtifact,
    *,
    structural_id: int,
    structural_indices: Sequence[int],
) -> dict[str, Any]:
    """
    Encode a :class:`StructuralArtifact` into a logical column dict.
    """
    if not isinstance(structural_id, int) or isinstance(structural_id, bool):
        raise StructuralCodecError(
            f"structural_id must be an int, got {structural_id!r}"
        )
    if structural_id < 0:
        raise StructuralCodecError(f"structural_id must be >= 0, got {structural_id}")
    indices = tuple(int(v) for v in structural_indices)
    is_build_failure = bool(artifact.is_build_failure)
    failure_json = (
        dumps_tagged_json(dict(artifact.build_failure_result))
        if artifact.build_failure_result is not None
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "codec_version": CODEC_VERSION,
        "structural_id": int(structural_id),
        "structural_indices": list(indices),
        "structural_key_digest": structural_key_digest(artifact.key),
        "snapshot_hash": str(artifact.snapshot_hash),
        "n": float(artifact.n),
        "mk_cleared": bool(artifact.mk_cleared),
        "is_build_failure": is_build_failure,
        "key_json": dumps_tagged_json(artifact.key),
        "build_failure_json": failure_json,
        "m_u": _encode_section_entries(artifact.m_u),
        "m_cr": _encode_section_entries(artifact.m_cr),
        "mk_curves": _encode_mk_curves(artifact.mk_curves),
        "mk_metadata": _encode_mk_metadata(artifact.mk_metadata),
    }


def decode_record(record: Mapping[str, Any]) -> StructuralArtifact:
    """
    Decode a logical column dict into a :class:`StructuralArtifact`.
    """
    schema_version = int(record.get("schema_version", SCHEMA_VERSION))
    codec_version = int(record.get("codec_version", CODEC_VERSION))
    if schema_version != SCHEMA_VERSION or codec_version != CODEC_VERSION:
        raise StructuralCodecError(
            f"unsupported schema/codec version "
            f"{schema_version}/{codec_version}; "
            f"expected {SCHEMA_VERSION}/{CODEC_VERSION}"
        )
    key = loads_tagged_json(str(record["key_json"]))
    if not isinstance(key, tuple):
        raise StructuralCodecError("structural key must decode to a tuple")
    digest = structural_key_digest(key)
    stored_digest = str(record["structural_key_digest"])
    if digest != stored_digest:
        raise StructuralCodecError(
            "structural_key_digest mismatch after key decode "
            f"(stored={stored_digest}, computed={digest})"
        )
    failure_raw = record.get("build_failure_json")
    build_failure: dict[str, Any] | None
    if failure_raw is None or failure_raw == "":
        build_failure = None
    else:
        decoded_failure = loads_tagged_json(str(failure_raw))
        if not isinstance(decoded_failure, Mapping):
            raise StructuralCodecError("build_failure_json must decode to a mapping")
        build_failure = dict(decoded_failure)
    is_build_failure = bool(record["is_build_failure"])
    if is_build_failure != (build_failure is not None):
        raise StructuralCodecError(
            "is_build_failure flag disagrees with build_failure_json presence"
        )
    return StructuralArtifact(
        key=key,
        snapshot_hash=str(record["snapshot_hash"]),
        n=float(record["n"]),
        m_u=_decode_section_entries(record["m_u"]),
        m_cr=_decode_section_entries(record["m_cr"]),
        mk_curves=_decode_mk_curves(record["mk_curves"]),
        mk_metadata=_decode_mk_metadata(record["mk_metadata"]),
        mk_cleared=bool(record["mk_cleared"]),
        build_failure_result=build_failure,
    )


def artifacts_equal(
    left: StructuralArtifact,
    right: StructuralArtifact,
    *,
    rtol: float = 1e-12,
) -> bool:
    """
    Return whether two artifacts match scientifically for codec round-trips.
    """
    if left.snapshot_hash != right.snapshot_hash:
        return False
    if left.mk_cleared != right.mk_cleared:
        return False
    if left.is_build_failure != right.is_build_failure:
        return False
    if not scientific_values_equal(left.key, right.key, rtol=rtol):
        return False
    if not scientific_values_equal(left.n, right.n, rtol=rtol):
        return False
    if not scientific_values_equal(left.m_u, right.m_u, rtol=rtol):
        return False
    if not scientific_values_equal(left.m_cr, right.m_cr, rtol=rtol):
        return False
    if not scientific_values_equal(left.mk_curves, right.mk_curves, rtol=rtol):
        return False
    if not scientific_values_equal(left.mk_metadata, right.mk_metadata, rtol=rtol):
        return False
    return scientific_values_equal(
        left.build_failure_result,
        right.build_failure_result,
        rtol=rtol,
    )
