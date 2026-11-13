"""Synthetic fixed-target feasibility audit and reporting-gate tests."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.ioh_core.workflows.optimization_runs.fixed_target_feasibility_audit import (
    AUDIT_FILENAME,
    FEASIBILITY_INVARIANT_SQL,
    HASH_VERSION,
    HP_SLAB_PROBLEM_IDS,
    audit_catalogue_thresholds,
    catalogue_sha256_v1,
    feasible_design_target_claim_allowed,
    feasibility_guarantee_for_rows,
    feasibility_invariant_holds,
    main,
    read_catalogue_table,
    write_audit_artifacts,
)
from core.ioh_core.workflows.optimization_runs.models import _THRESHOLD_RTOL

_CAMPAIGN = {
    "campaign_problem_ids": ("P",),
    "campaign_constraint_names": ("g",),
    "campaign_penalty_alpha": 1.0,
    "campaign_penalty_beta": 1.0 / 3.0,
    "campaign_objective_identity": "y_p",
    "code_revision": "test",
    "audited_at": "2026-01-01T00:00:00Z",
}


def _row(
    index: object,
    analysis_ok: bool,
    constraint_feasible: bool,
    feasible: bool,
    y_p: float,
) -> dict[str, object]:
    return {
        "global_index": index,
        "analysis_ok": analysis_ok,
        "constraint_feasible": constraint_feasible,
        "feasible": feasible,
        "y_p": y_p,
    }


def _audit(
    rows: list[dict[str, object]],
    thresholds: list[dict[str, float]],
    *,
    cardinality: int | None = None,
    catalogue_updates: dict[str, object] | None = None,
    campaign_updates: dict[str, object] | None = None,
    catalogues: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    catalogue: dict[str, object] = {
        "problem_id": "P",
        "cardinality": len(rows) if cardinality is None else cardinality,
        "rows": rows,
        "constraint_names": ["g"],
        "penalty_alpha": 1.0,
        "penalty_beta": 1.0 / 3.0,
        "objective_identity": "y_p",
        "source_path": "synthetic.duckdb",
        "table_name": "catalogue",
        "thresholds": thresholds,
    }
    if catalogue_updates:
        catalogue.update(catalogue_updates)
    kwargs = dict(_CAMPAIGN)
    kwargs["campaign_problem_ids"] = (str(catalogue["problem_id"]),)
    if campaign_updates:
        kwargs.update(campaign_updates)
    return audit_catalogue_thresholds(
        catalogues if catalogues is not None else [catalogue],
        **kwargs,
    )


class TestFixedTargetFeasibilityAudit(unittest.TestCase):
    def test_parenthesized_invariant_admits_four_combinations(self) -> None:
        valid = (
            (False, False, False),
            (False, True, False),
            (True, False, False),
            (True, True, True),
        )
        invalid = (
            (False, False, True),
            (False, True, True),
            (True, False, True),
            (True, True, False),
        )
        for flags in valid:
            self.assertTrue(feasibility_invariant_holds(*flags))
        for flags in invalid:
            self.assertFalse(feasibility_invariant_holds(*flags))
        source = inspect.getsource(feasibility_invariant_holds)
        self.assertIn("feasible == (analysis_ok and constraint_feasible)", source)
        self.assertIn("(analysis_ok AND constraint_feasible)", FEASIBILITY_INVARIANT_SQL)
        self.assertIn("FEASIBILITY_INVARIANT_SQL", inspect.getsource(read_catalogue_table))
        self.assertEqual(_THRESHOLD_RTOL, 1e-12)

    def test_hash_orders_rows_and_includes_constraint_feasible(self) -> None:
        rows = [
            _row(1, False, True, False, 2.0),
            _row(0, True, True, True, 1.0),
        ]
        text = "\n".join(
            (
                f"0 true true true {float(1.0).hex()}",
                f"1 false true false {float(2.0).hex()}",
            )
        )
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual(catalogue_sha256_v1(rows), digest)
        changed = [
            _row(1, False, False, False, 2.0),
            _row(0, True, True, True, 1.0),
        ]
        self.assertNotEqual(catalogue_sha256_v1(rows), catalogue_sha256_v1(changed))
        self.assertNotEqual(digest, hash(text))

    def test_pass_when_every_finite_row_at_the_threshold_is_feasible(self) -> None:
        audit = _audit(
            [
                _row(0, True, True, True, 10.0),
                _row(1, False, False, False, 50.0),
                _row(2, False, True, False, 40.0),
                _row(3, True, False, False, 30.0),
            ],
            [{"target_threshold": 10.0, "allowed_deviation_percent": 0.0}],
        )
        problem = audit["problems"][0]
        threshold = problem["thresholds"][0]
        self.assertIsNone(problem["preflight_reason"])
        self.assertEqual(threshold["state"], "pass")
        self.assertIsNone(threshold["reason"])
        self.assertEqual(problem["y_f_star"], 10.0)
        self.assertEqual(threshold["infeasible_or_failed_minimum"], 30.0)
        self.assertEqual(threshold["offending_row_count"], 0)
        self.assertEqual(
            problem["catalogue_hash"],
            catalogue_sha256_v1(
                [
                    _row(0, True, True, True, 10.0),
                    _row(1, False, False, False, 50.0),
                    _row(2, False, True, False, 40.0),
                    _row(3, True, False, False, 30.0),
                ]
            ),
        )

    def test_false_true_false_stays_in_the_infeasible_minimum(self) -> None:
        audit = _audit(
            [
                _row(0, True, True, True, 10.0),
                _row(1, False, True, False, 10.0),
            ],
            [{"target_threshold": 10.0, "allowed_deviation_percent": 0.0}],
        )
        threshold = audit["problems"][0]["thresholds"][0]
        self.assertIsNone(audit["problems"][0]["preflight_reason"])
        self.assertEqual(threshold["state"], "fail")
        self.assertEqual(threshold["reason"], "infeasible_can_meet_target")
        self.assertEqual(threshold["infeasible_or_failed_minimum"], 10.0)
        self.assertEqual(threshold["examples"][0]["constraint_feasible"], True)
        self.assertEqual(threshold["examples"][0]["feasible"], False)

    def test_fail_for_analysis_valid_infeasible_and_encoded_failure(self) -> None:
        for failed in (
            _row(1, True, False, False, 10.0),
            _row(1, False, False, False, 10.0),
        ):
            audit = _audit(
                [_row(0, True, True, True, 10.0), failed],
                [{"target_threshold": 10.0, "allowed_deviation_percent": 0.0}],
            )
            threshold = audit["problems"][0]["thresholds"][0]
            self.assertEqual(threshold["state"], "fail")
            self.assertEqual(threshold["offending_row_count"], 1)

    def test_admissible_infinity_is_excluded_and_null_minimum_passes(self) -> None:
        audit = _audit(
            [
                _row(0, True, True, True, 10.0),
                _row(1, False, False, False, math.inf),
            ],
            [{"target_threshold": 10.0, "allowed_deviation_percent": 0.0}],
        )
        problem = audit["problems"][0]
        threshold = problem["thresholds"][0]
        self.assertEqual(threshold["state"], "pass")
        self.assertIsNone(threshold["infeasible_or_failed_minimum"])
        self.assertEqual(problem["admissible_posinf_failed_count"], 1)
        with TemporaryDirectory() as tmp:
            document = write_audit_artifacts(audit, Path(tmp))
            text = (Path(tmp) / AUDIT_FILENAME).read_text(encoding="utf-8")
            self.assertNotIn("Infinity", text)
            self.assertNotIn("NaN", text)
            loaded = json.loads(text)
            self.assertIsNone(
                loaded["problems"][0]["thresholds"][0]["infeasible_or_failed_minimum"]
            )
            self.assertNotIn("offending_rows", document)
            json.dumps(document, allow_nan=False)
            csv_text = (Path(tmp) / "fixed_target_feasibility_audit.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("pass", csv_text)
            self.assertNotIn("Infinity", csv_text)

    def test_no_finite_feasible_optimum_skips_threshold_comparison(self) -> None:
        audit = _audit(
            [_row(0, False, False, False, 1.0)],
            [{"target_threshold": 100.0, "allowed_deviation_percent": 0.0}],
        )
        problem = audit["problems"][0]
        threshold = problem["thresholds"][0]
        self.assertEqual(problem["preflight_reason"], "no_finite_feasible_optimum")
        self.assertEqual(threshold["state"], "unverified")
        self.assertEqual(threshold["reason"], "no_finite_feasible_optimum")
        self.assertIsNone(problem["y_f_star"])
        self.assertEqual(threshold["offending_row_count"], 0)
        self.assertNotEqual(threshold["reason"], "infeasible_can_meet_target")

    def test_nonfinite_y_p_rejects_the_catalogue(self) -> None:
        cases = (
            _row(1, True, True, True, math.nan),
            _row(1, False, False, False, -math.inf),
            _row(1, True, False, False, math.inf),
        )
        for bad in cases:
            audit = _audit(
                [_row(0, True, True, True, 10.0), bad],
                [{"target_threshold": 10.0, "allowed_deviation_percent": 0.0}],
            )
            problem = audit["problems"][0]
            self.assertEqual(problem["preflight_reason"], "invalid_y_p")
            self.assertEqual(problem["thresholds"][0]["state"], "unverified")
            self.assertEqual(problem["thresholds"][0]["reason"], "invalid_y_p")

    def test_index_coverage_rejections(self) -> None:
        short = _audit(
            [_row(0, True, True, True, 1.0), _row(1, True, True, True, 2.0)],
            [{"target_threshold": 1.0, "allowed_deviation_percent": 0.0}],
            cardinality=3,
        )
        self.assertEqual(short["problems"][0]["preflight_reason"], "incomplete_index_coverage")

        duplicate = _audit(
            [_row(0, True, True, True, 1.0), _row(0, True, True, True, 2.0)],
            [{"target_threshold": 1.0, "allowed_deviation_percent": 0.0}],
        )
        self.assertEqual(
            duplicate["problems"][0]["preflight_reason"],
            "incomplete_index_coverage",
        )

        null_index = _audit(
            [_row(None, True, True, True, 1.0)],
            [{"target_threshold": 1.0, "allowed_deviation_percent": 0.0}],
        )
        self.assertEqual(null_index["problems"][0]["preflight_reason"], "null_global_index")

        fractional = _audit(
            [_row(0.0, True, True, True, 1.0)],
            [{"target_threshold": 1.0, "allowed_deviation_percent": 0.0}],
        )
        self.assertEqual(
            fractional["problems"][0]["preflight_reason"],
            "non_integer_global_index",
        )

        replaced = _audit(
            [
                _row(0, True, True, True, 1.0),
                _row(1, True, False, False, 5.0),
                _row(3, False, False, False, 4.0),
            ],
            [{"target_threshold": 1.0, "allowed_deviation_percent": 0.0}],
        )
        problem = replaced["problems"][0]
        self.assertEqual(problem["preflight_reason"], "incomplete_index_coverage")
        self.assertEqual(problem["row_count"], 3)
        self.assertEqual(problem["distinct_global_indices"], 3)
        self.assertEqual(problem["min_global_index"], 0)
        self.assertEqual(problem["max_global_index"], 3)
        self.assertEqual(problem["thresholds"][0]["state"], "unverified")

    def test_invalid_boolean_combinations_stop_before_comparison(self) -> None:
        invalid = (
            (False, False, True),
            (False, True, True),
            (True, False, True),
            (True, True, False),
        )
        for flags in invalid:
            audit = _audit(
                [
                    _row(0, True, True, True, 10.0),
                    _row(1, *flags, 1.0),
                ],
                [{"target_threshold": 10.0, "allowed_deviation_percent": 0.0}],
            )
            threshold = audit["problems"][0]["thresholds"][0]
            self.assertEqual(audit["problems"][0]["preflight_reason"], "feasibility_invariant")
            self.assertEqual(threshold["state"], "unverified")
            self.assertEqual(threshold["offending_row_count"], 0)

    def test_threshold_mismatch_uses_manifest_value_unchanged(self) -> None:
        mismatched = _audit(
            [
                _row(0, True, True, True, 100.0),
                _row(1, True, False, False, 50.0),
            ],
            [{"target_threshold": 100.0, "allowed_deviation_percent": 1.0}],
        )
        threshold = mismatched["problems"][0]["thresholds"][0]
        self.assertEqual(threshold["state"], "unverified")
        self.assertEqual(threshold["reason"], "threshold_mismatch")
        self.assertEqual(threshold["offending_row_count"], 0)

        manifest = math.nextafter(100.0, 0.0)
        within = _audit(
            [
                _row(0, True, True, True, 100.0),
                _row(1, True, False, False, 100.0),
            ],
            [{"target_threshold": manifest, "allowed_deviation_percent": 0.0}],
        )
        kept = within["problems"][0]["thresholds"][0]
        self.assertEqual(kept["state"], "pass")
        self.assertEqual(kept["target_threshold"], manifest)
        self.assertEqual(kept["offending_row_count"], 0)

    def test_examples_are_capped_and_parquet_keeps_every_offending_row(self) -> None:
        rows = [_row(0, True, True, True, 10.0)]
        rows.extend(_row(index, True, False, False, 1.0) for index in range(1, 22))
        audit = _audit(
            rows,
            [{"target_threshold": 10.0, "allowed_deviation_percent": 0.0}],
        )
        threshold = audit["problems"][0]["thresholds"][0]
        self.assertEqual(threshold["state"], "fail")
        self.assertEqual(threshold["offending_row_count"], 21)
        self.assertEqual(len(threshold["examples"]), 20)
        with TemporaryDirectory() as tmp:
            document = write_audit_artifacts(audit, Path(tmp))
            published = document["problems"][0]["thresholds"][0]
            parquet_path = Path(tmp) / published["parquet_path"]
            pyarrow_parquet = __import__("pyarrow.parquet", fromlist=["read_table"])
            table = pyarrow_parquet.read_table(parquet_path)
            self.assertEqual(table.num_rows, 21)
            self.assertEqual(
                table.column_names,
                [
                    "global_index",
                    "y_p",
                    "analysis_ok",
                    "constraint_feasible",
                    "feasible",
                ],
            )
            text = json.dumps(document, allow_nan=False)
            self.assertLessEqual(text.count('"global_index"'), 20)

    def test_missing_campaign_catalogues_stay_unverified(self) -> None:
        extra = {
            "problem_id": "NOT_A_PROBLEM",
            "cardinality": 1,
            "rows": [_row(0, True, True, True, 1.0)],
            "constraint_names": ["g"],
            "penalty_alpha": 1.0,
            "penalty_beta": 1.0 / 3.0,
            "objective_identity": "y_p",
            "source_path": "synthetic.duckdb",
            "table_name": "catalogue",
            "thresholds": [
                {"target_threshold": 1.0, "allowed_deviation_percent": 0.0}
            ],
        }
        audit = audit_catalogue_thresholds(
            [extra],
            campaign_constraint_names=("g",),
            campaign_penalty_alpha=1.0,
            campaign_penalty_beta=1.0 / 3.0,
            campaign_objective_identity="y_p",
            code_revision="test",
            audited_at="2026-01-01T00:00:00Z",
        )
        self.assertEqual(len(HP_SLAB_PROBLEM_IDS), 21)
        self.assertEqual(
            [problem["problem_id"] for problem in audit["problems"][:21]],
            list(HP_SLAB_PROBLEM_IDS),
        )
        self.assertTrue(
            all(
                problem["preflight_reason"] == "missing_catalogue"
                for problem in audit["problems"][:21]
            )
        )
        self.assertEqual(audit["problems"][21]["problem_id"], "NOT_A_PROBLEM")
        self.assertEqual(audit["hash_version"], HASH_VERSION)

    def test_definition_mismatch_is_unverified(self) -> None:
        audit = _audit(
            [_row(0, True, True, True, 1.0)],
            [{"target_threshold": 1.0, "allowed_deviation_percent": 0.0}],
            catalogue_updates={"constraint_names": ["other"]},
        )
        self.assertEqual(audit["problems"][0]["preflight_reason"], "definition_mismatch")
        self.assertEqual(audit["problems"][0]["thresholds"][0]["state"], "unverified")

    def test_duckdb_reader_matches_the_in_memory_audit(self) -> None:
        duckdb = __import__("duckdb")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalogue.duckdb"
            connection = duckdb.connect(str(path))
            connection.execute(
                """
                CREATE TABLE catalogue (
                    global_index BIGINT,
                    analysis_ok BOOLEAN,
                    constraint_feasible BOOLEAN,
                    feasible BOOLEAN,
                    y_p DOUBLE
                )
                """
            )
            connection.execute(
                "INSERT INTO catalogue VALUES (0, true, true, true, 10.0), "
                "(1, false, true, false, 40.0)"
            )
            loaded = read_catalogue_table(connection, "catalogue")
            connection.close()
            catalogue = {
                "problem_id": "P",
                "cardinality": 2,
                "duckdb_path": str(path),
                "table_name": "catalogue",
                "constraint_names": ["g"],
                "penalty_alpha": 1.0,
                "penalty_beta": 1.0 / 3.0,
                "objective_identity": "y_p",
                "source_path": str(path),
                "thresholds": [
                    {"target_threshold": 10.0, "allowed_deviation_percent": 0.0}
                ],
            }
            from_db = audit_catalogue_thresholds([catalogue], **_CAMPAIGN)
            from_rows = _audit(
                loaded,
                [{"target_threshold": 10.0, "allowed_deviation_percent": 0.0}],
            )
            self.assertEqual(
                from_db["problems"][0]["thresholds"][0]["state"],
                from_rows["problems"][0]["thresholds"][0]["state"],
            )
            self.assertEqual(from_db["problems"][0]["catalogue_hash"], from_rows["problems"][0]["catalogue_hash"])
            self.assertEqual(from_db["problems"][0]["thresholds"][0]["state"], "pass")
            missing = dict(catalogue)
            missing["table_name"] = "absent"
            unread = audit_catalogue_thresholds([missing], **_CAMPAIGN)
            self.assertEqual(unread["problems"][0]["preflight_reason"], "catalogue_unreadable")
            self.assertEqual(unread["problems"][0]["thresholds"][0]["state"], "unverified")

    def test_cli_writes_a_summary_outside_the_source_tree(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = {
                "constraint_names": ["g"],
                "penalty_alpha": 1.0,
                "penalty_beta": 1.0 / 3.0,
                "objective_identity": "y_p",
                "problem_ids": ["P"],
                "catalogues": [
                    {
                        "problem_id": "P",
                        "cardinality": 1,
                        "rows": [_row(0, True, True, True, 1.0)],
                        "constraint_names": ["g"],
                        "penalty_alpha": 1.0,
                        "penalty_beta": 1.0 / 3.0,
                        "objective_identity": "y_p",
                        "source_path": "synthetic.duckdb",
                        "table_name": "catalogue",
                        "thresholds": [
                            {
                                "target_threshold": 1.0,
                                "allowed_deviation_percent": 0.0,
                            }
                        ],
                    }
                ],
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            output = root / "audit_out"
            status = main([str(request_path), "--output-dir", str(output), "--code-revision", "test"])
            self.assertEqual(status, 0)
            document = json.loads((output / AUDIT_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(document["problems"][0]["thresholds"][0]["state"], "pass")

    def test_audit_module_does_not_import_algorithm_modules(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "core"
            / "ioh_core"
            / "workflows"
            / "optimization_runs"
            / "fixed_target_feasibility_audit.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn("algorithms", alias.name)
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn("algorithms", node.module or "")

    def test_reporting_gate_matches_hash_penalty_and_exact_threshold(self) -> None:
        audit = _audit(
            [
                _row(0, True, True, True, 10.0),
                _row(1, True, False, False, 40.0),
            ],
            [{"target_threshold": 10.0, "allowed_deviation_percent": 0.0}],
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_audit_artifacts(audit, root)
            rows = [
                {
                    "termination_mode": "fixed_target",
                    "problem_id": "P",
                    "target_threshold": 10.0,
                    "target_hit": True,
                }
            ]
            state, pairs = feasibility_guarantee_for_rows(rows, root)
            self.assertEqual(state, "pass")
            self.assertTrue(feasible_design_target_claim_allowed(state))
            self.assertEqual(pairs[0]["state"], "pass")
            self.assertNotIn("target_hit", pairs[0])

            missing_state, _missing_pairs = feasibility_guarantee_for_rows(
                rows,
                root / "absent",
            )
            self.assertEqual(missing_state, "unverified")
            self.assertFalse(feasible_design_target_claim_allowed(missing_state))
            self.assertFalse(feasible_design_target_claim_allowed("fail"))

            document = json.loads((root / AUDIT_FILENAME).read_text(encoding="utf-8"))
            document["campaign"]["catalogue_hashes"]["P"] = "0" * 64
            (root / AUDIT_FILENAME).write_text(json.dumps(document), encoding="utf-8")
            hashed, _pairs = feasibility_guarantee_for_rows(rows, root)
            self.assertEqual(hashed, "unverified")

            document = json.loads((root / "fixed_target_feasibility_audit.json").read_text(encoding="utf-8"))
            # Restore a consistent hash, then change the campaign penalty.
            fresh = write_audit_artifacts(audit, root)
            self.assertEqual(fresh["problems"][0]["thresholds"][0]["state"], "pass")
            document = json.loads((root / AUDIT_FILENAME).read_text(encoding="utf-8"))
            document["campaign"]["penalty_beta"] = 2.0
            (root / AUDIT_FILENAME).write_text(json.dumps(document), encoding="utf-8")
            penalised, _pairs = feasibility_guarantee_for_rows(rows, root)
            self.assertEqual(penalised, "unverified")

            write_audit_artifacts(audit, root)
            shifted = [
                {
                    "termination_mode": "fixed_target",
                    "problem_id": "P",
                    "target_threshold": 10.0 + 1e-9,
                }
            ]
            shifted_state, _pairs = feasibility_guarantee_for_rows(shifted, root)
            self.assertEqual(shifted_state, "unverified")

            budget_state, budget_pairs = feasibility_guarantee_for_rows(
                [{"termination_mode": "fixed_budget", "problem_id": "P"}],
                root,
            )
            self.assertIsNone(budget_state)
            self.assertEqual(budget_pairs, [])


if __name__ == "__main__":
    unittest.main()
