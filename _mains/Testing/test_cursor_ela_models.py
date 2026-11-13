"""
ELA tidy FeatureResult identity and config hashing.
"""
from __future__ import annotations

import unittest
from dataclasses import fields
from pathlib import Path

from core.ioh_core.workflows.la.models import (
    ElaError,
    FeatureResult,
    ROLE_PRIMARY,
    STATUS_CONSTANT_LANDSCAPE,
    STATUS_OK,
    canonical_config_json,
    feature_result,
)


def _row(**overrides: object) -> FeatureResult:
    payload: dict[str, object] = {
        "problem_id": "L6750_B2",
        "source_type": "sampling",
        "source_dir": Path("experiments/sampling_p1"),
        "landscape": "y_raw",
        "response_view": "minmax",
        "x_view": "norm",
        "family": "ela_meta",
        "feature_name": "ela_meta.lin_simple.adj_r2",
        "status": "ok",
        "role": "primary",
        "n_input": 8,
        "n_unique": 8,
        "runtime_s": 0.01,
        "library": "pflacco",
        "library_version": "1.2.2",
        "config_payload": {"minimize": True},
        "value": 0.5,
        "repetition": 0,
    }
    payload.update(overrides)
    return feature_result(**payload)  # type: ignore[arg-type]


class TestFeatureResult(unittest.TestCase):
    """Separate nullable ids; derived config hash; no n_used."""

    def test_schema_has_identity_fields_and_no_n_used(self) -> None:
        names = {item.name for item in fields(FeatureResult)}
        self.assertIn("repetition", names)
        self.assertIn("sample_id", names)
        self.assertIn("walk_id", names)
        self.assertIn("n_input", names)
        self.assertIn("n_unique", names)
        self.assertIn("config_payload", names)
        self.assertIn("config_json", names)
        self.assertIn("config_hash", names)
        self.assertIn("sampling_design_signature", names)
        self.assertIn("analysis_signature", names)
        self.assertNotIn("n_used", names)
        self.assertNotIn("repetition / sample_id / walk_id", names)

    def test_nullable_ids_are_independent(self) -> None:
        sampled = _row(repetition=0, sample_id=None, walk_id=None)
        named = _row(repetition=None, sample_id="pilot", walk_id=None)
        walk = _row(repetition=1, sample_id=None, walk_id=3)
        self.assertEqual(sampled.repetition, 0)
        self.assertIsNone(sampled.sample_id)
        self.assertIsNone(named.repetition)
        self.assertEqual(named.sample_id, "pilot")
        self.assertEqual(walk.walk_id, 3)

    def test_signatures_default_to_none(self) -> None:
        row = _row()
        self.assertIsNone(row.sampling_design_signature)
        self.assertIsNone(row.analysis_signature)
        self.assertIsNone(row.seed)

    def test_config_hash_is_stable_under_key_order(self) -> None:
        left = _row(config_payload={"b": 1, "a": 2})
        right = _row(config_payload={"a": 2, "b": 1})
        self.assertEqual(left.config_json, right.config_json)
        self.assertEqual(left.config_hash, right.config_hash)
        self.assertEqual(left.config_json, canonical_config_json({"a": 2, "b": 1}))
        self.assertEqual(len(left.config_hash), 64)

    def test_n_unique_is_stored_not_a_gate(self) -> None:
        row = _row(n_input=8, n_unique=3)
        self.assertEqual(row.n_input, 8)
        self.assertEqual(row.n_unique, 3)

    def test_mismatched_config_hash_fails_closed(self) -> None:
        row = _row()
        with self.assertRaises(ElaError) as ctx:
            FeatureResult(
                problem_id=row.problem_id,
                source_type=row.source_type,
                source_dir=row.source_dir,
                repetition=row.repetition,
                sample_id=row.sample_id,
                walk_id=row.walk_id,
                landscape=row.landscape,
                response_view=row.response_view,
                z_min=row.z_min,
                z_max=row.z_max,
                x_view=row.x_view,
                family=row.family,
                feature_name=row.feature_name,
                value=row.value,
                status=row.status,
                role=row.role,
                n_input=row.n_input,
                n_unique=row.n_unique,
                runtime_s=row.runtime_s,
                library=row.library,
                library_version=row.library_version,
                config_payload=row.config_payload,
                config_json=row.config_json,
                config_hash="0" * 64,
                sampling_design_signature=None,
                analysis_signature=None,
                seed=None,
            )
        self.assertIn("config_hash", str(ctx.exception))

    def test_unknown_status_and_role_fail_closed(self) -> None:
        with self.assertRaises(ElaError):
            _row(status="constant")
        with self.assertRaises(ElaError):
            _row(role="main")

    def test_landscape_and_view_vocabularies(self) -> None:
        self.assertEqual(_row(landscape="y_p").landscape, "y_p")
        self.assertEqual(
            _row(landscape="constraint_A_bending_capacity").landscape,
            "constraint_A_bending_capacity",
        )
        self.assertEqual(_row(response_view="n/a").response_view, "n/a")
        self.assertEqual(_row(x_view="index").x_view, "index")
        with self.assertRaises(ElaError):
            _row(landscape="util__A_bending_capacity")
        with self.assertRaises(ElaError):
            _row(landscape="constraint_")
        with self.assertRaises(ElaError):
            _row(response_view="unset")
        with self.assertRaises(ElaError):
            _row(x_view="param")

    def test_constant_landscape_status_token(self) -> None:
        row = _row(
            status=STATUS_CONSTANT_LANDSCAPE,
            role=ROLE_PRIMARY,
            value=None,
            response_view="n/a",
        )
        self.assertEqual(row.status, STATUS_CONSTANT_LANDSCAPE)
        self.assertEqual(_row(status=STATUS_OK).status, STATUS_OK)


if __name__ == "__main__":
    unittest.main()
