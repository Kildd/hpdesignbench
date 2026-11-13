"""
Arrow schema for sealed structural-artifact library records.
"""
from __future__ import annotations

from core.ioh_core.structural_artifacts.codec import CODEC_VERSION, SCHEMA_VERSION
from core.ioh_core.workflows.brute_force.deps import import_pyarrow


def section_entry_type(pa: object) -> object:
    """
    Return the Arrow struct type for one ``m_u`` / ``m_cr`` entry.
    """
    return pa.struct(
        [
            ("result_type", pa.string()),
            ("x", pa.float64()),
            ("limit_state", pa.string()),
            ("n", pa.float64()),
            ("flipped", pa.bool_()),
            ("payload_json", pa.string()),
        ]
    )


def mk_curve_point_type(pa: object) -> object:
    """
    Return the Arrow struct type for one M–κ curve at a position.
    """
    return pa.struct(
        [
            ("x", pa.float64()),
            ("m_y", pa.list_(pa.float64())),
            ("chi_y", pa.list_(pa.float64())),
        ]
    )


def mk_curve_entry_type(pa: object) -> object:
    """
    Return the Arrow struct type for one ``mk_curves`` entry.
    """
    return pa.struct(
        [
            ("limit_state", pa.string()),
            ("constitutive_law", pa.string()),
            ("method", pa.string()),
            ("simplification_json", pa.string()),
            ("softening_json", pa.string()),
            ("n", pa.float64()),
            ("m_k_num_points", pa.int32()),
            ("curves", pa.list_(mk_curve_point_type(pa))),
        ]
    )


def mk_metadata_entry_type(pa: object) -> object:
    """
    Return the Arrow struct type for one ``mk_metadata`` entry.
    """
    return pa.struct(
        [
            ("limit_state", pa.string()),
            ("constitutive_law", pa.string()),
            ("method", pa.string()),
            ("simplification_json", pa.string()),
            ("softening_json", pa.string()),
            ("n", pa.float64()),
            ("m_k_num_points", pa.int32()),
            ("payload_json", pa.string()),
        ]
    )


def structural_artifact_schema() -> object:
    """
    Return the shared Arrow schema for logical structural-artifact records.
    """
    pa = import_pyarrow()
    return pa.schema(
        [
            ("schema_version", pa.int32()),
            ("codec_version", pa.int32()),
            ("structural_id", pa.int64()),
            ("structural_indices", pa.list_(pa.int32())),
            ("structural_key_digest", pa.string()),
            ("snapshot_hash", pa.string()),
            ("n", pa.float64()),
            ("mk_cleared", pa.bool_()),
            ("is_build_failure", pa.bool_()),
            ("key_json", pa.string()),
            ("build_failure_json", pa.string()),
            ("m_u", pa.list_(section_entry_type(pa))),
            ("m_cr", pa.list_(section_entry_type(pa))),
            ("mk_curves", pa.list_(mk_curve_entry_type(pa))),
            ("mk_metadata", pa.list_(mk_metadata_entry_type(pa))),
        ]
    )


def schema_metadata() -> dict[str, int]:
    """
    Return schema/codec version metadata for manifests.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "codec_version": CODEC_VERSION,
    }
