"""
ELA sampling-design and analysis signature hashes.
"""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.ioh_core.workflows.la.models import (
    ElaError,
    RESPONSE_VIEW_MINMAX,
    X_VIEW_NORM,
    canonical_config_json,
    compute_config_hash,
)
from core.ioh_core.workflows.la.signatures import (
    analysis_signature,
    analysis_signature_payload,
    hash_signature_payload,
    load_sampling_design_fields,
    load_schema_hash,
    sampling_design_payload,
    sampling_design_signature,
)

_PROJECTION = (
    "lower + floor(u * (upper - lower + 1)), clipped to [lower, upper]"
)
_VARS = ["geom_t_mm", "mat_conc_id"]
_LO = [0, 3]
_HI = [2, 5]
_HASH = "b" * 64


def _sampling_kwargs(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "problem_id": "L6750_B2",
        "schema_hash": _HASH,
        "method": "scipy_qmc_sobol",
        "sample_size": 8,
        "scramble": True,
        "projection": _PROJECTION,
        "delete_duplicates": False,
        "var_names": list(_VARS),
        "lower_bounds": list(_LO),
        "upper_bounds": list(_HI),
    }
    payload.update(overrides)
    return payload


def _analysis_kwargs(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "family": "ela_meta",
        "landscape": "y_raw",
        "constraint_names": [
            "A_bending_capacity",
            "Z1_nt_dt_combination_capacity",
        ],
        "x_view": X_VIEW_NORM,
        "response_view": RESPONSE_VIEW_MINMAX,
        "config": {},
    }
    payload.update(overrides)
    return payload


class TestSamplingDesignSignature(unittest.TestCase):
    """Replicate identity must not split otherwise-equivalent designs."""

    def test_seed_and_repetition_are_not_in_payload(self) -> None:
        payload = sampling_design_payload(**_sampling_kwargs())
        for key in ("repetition", "seed", "n_used", "n_evaluate", "n_unique"):
            self.assertNotIn(key, payload)
        digest = hash_signature_payload(payload)
        self.assertEqual(digest, compute_config_hash(canonical_config_json(payload)))
        self.assertEqual(len(digest), 64)

    def test_same_design_different_seed_same_hash(self) -> None:
        left = sampling_design_signature(**_sampling_kwargs())
        right = sampling_design_signature(**_sampling_kwargs())
        self.assertEqual(left, right)

    def test_sample_size_or_delete_duplicates_changes_hash(self) -> None:
        base = sampling_design_signature(**_sampling_kwargs())
        bigger = sampling_design_signature(**_sampling_kwargs(sample_size=16))
        dedup = sampling_design_signature(
            **_sampling_kwargs(delete_duplicates=True)
        )
        self.assertNotEqual(base, bigger)
        self.assertNotEqual(base, dedup)

    def test_problem_id_and_schema_hash_both_matter(self) -> None:
        base = sampling_design_signature(**_sampling_kwargs())
        other_problem = sampling_design_signature(
            **_sampling_kwargs(problem_id="L8100_B2")
        )
        other_schema = sampling_design_signature(
            **_sampling_kwargs(schema_hash="c" * 64)
        )
        self.assertNotEqual(base, other_problem)
        self.assertNotEqual(base, other_schema)

    def test_missing_projection_fails_closed(self) -> None:
        with self.assertRaises(ElaError):
            sampling_design_payload(**_sampling_kwargs(projection=""))


class TestAnalysisSignature(unittest.TestCase):
    """Family knobs, landscape, and keep-list split analysis groups."""

    def test_intent_response_view_is_minmax(self) -> None:
        payload = analysis_signature_payload(**_analysis_kwargs())
        self.assertEqual(payload["response_view"], "minmax")
        self.assertNotIn("n_used", payload)
        self.assertNotIn("n_used", payload["config"])

    def test_landscape_or_keep_list_or_blocks_change_hash(self) -> None:
        base = analysis_signature(**_analysis_kwargs())
        landscape = analysis_signature(**_analysis_kwargs(landscape="phi"))
        keep = analysis_signature(
            **_analysis_kwargs(constraint_names=["A_bending_capacity"])
        )
        blocks = analysis_signature(
            **_analysis_kwargs(
                family="cell_mapping",
                config={"blocks": [3, 3], "force": False, "minimize": True},
            )
        )
        self.assertNotEqual(base, landscape)
        self.assertNotEqual(base, keep)
        self.assertNotEqual(base, blocks)

    def test_config_tuples_are_hashed_as_lists(self) -> None:
        listed = analysis_signature(
            **_analysis_kwargs(config={"blocks": [2, 4], "force": True})
        )
        tupled = analysis_signature(
            **_analysis_kwargs(config={"blocks": (2, 4), "force": True})
        )
        self.assertEqual(listed, tupled)


class TestSignatureFileLoaders(unittest.TestCase):
    """Read schema_hash and design.json without importing sampling."""

    def test_loaders_read_sealed_json(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir(parents=True)
            (dataset / "schema.json").write_text(
                json.dumps({"schema_hash": _HASH, "var_names": _VARS}),
                encoding="utf-8",
            )
            folder = root / "inputs" / "sampling" / "rep_000"
            folder.mkdir(parents=True)
            (folder / "design.json").write_text(
                json.dumps(
                    {
                        "method": "scipy_qmc_sobol",
                        "sample_size": 8,
                        "scramble": True,
                        "seed": 29,
                        "index_mapping": _PROJECTION,
                        "delete_duplicates": False,
                        "var_names": _VARS,
                        "lower_bounds": _LO,
                        "upper_bounds": _HI,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_schema_hash(root), _HASH)
            fields = load_sampling_design_fields(root, repetition=0)
            self.assertEqual(fields["seed"], 29)
            self.assertEqual(fields["projection"], _PROJECTION)
            digest = sampling_design_signature(
                problem_id="L6750_B2",
                schema_hash=load_schema_hash(root),
                method=fields["method"],
                sample_size=fields["sample_size"],
                scramble=fields["scramble"],
                projection=fields["projection"],
                delete_duplicates=fields["delete_duplicates"],
                var_names=fields["var_names"],
                lower_bounds=fields["lower_bounds"],
                upper_bounds=fields["upper_bounds"],
            )
            self.assertEqual(len(digest), 64)

    def test_missing_schema_hash_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir(parents=True)
            (dataset / "schema.json").write_text(
                json.dumps({"var_names": _VARS}),
                encoding="utf-8",
            )
            with self.assertRaises(ElaError) as caught:
                load_schema_hash(root)
            self.assertIn("schema_hash", str(caught.exception))

    def test_module_does_not_import_sampling_or_snapshot(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "core"
            / "ioh_core"
            / "workflows"
            / "la"
            / "signatures.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        for forbidden in (
            "duckdb",
            "pflacco",
            "core.ioh_core.workflows.sampling",
            "core.ioh_core.workflows.optimization_runs.snapshot",
        ):
            for name in names:
                self.assertFalse(
                    name == forbidden or name.startswith(forbidden + "."),
                    msg=f"signatures.py imports {name}",
                )


if __name__ == "__main__":
    unittest.main()
