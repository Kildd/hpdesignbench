"""
Constraint decision-tree (infeasible repair) unit tests.
"""
from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.ioh_core.workflows.brute_force.analysis.constraint_decision_tree import (
    DecisionTreeOptions,
    _active_constraint_names,
    _extract_paths,
    _fit_cart,
    _sample_infeasible_seeds,
    _score_actions,
    _pattern_move_table,
    _load_labeled_rows,
    write_constraint_decision_tree_artifacts,
)
from core.ioh_core.workflows.brute_force.analysis.engine import (
    BruteForceQueryEngine,
    quote_ident,
)
from core.ioh_core.workflows.brute_force.deps import import_duckdb


def _toy_context():
    @dataclasses.dataclass(frozen=True)
    class _Ctx:
        constraint_names: tuple[str, ...]
        var_names: tuple[str, ...]
        problem_info: dict

    return _Ctx(
        constraint_names=(
            "A_bending_capacity",
            "B1a_deflection_by_wmax_capacity",
            "Z1_nt_dt_combination_capacity",
        ),
        var_names=("geom_t_mm", "geom_nt"),
        problem_info={
            "model": {
                "geom_t_mm": {
                    "role": "var",
                    "values": [20.0, 30.0, 40.0],
                    "lb": 0,
                    "ub": 2,
                },
                "geom_nt": {
                    "role": "var",
                    "values": [5.0, 10.0],
                    "lb": 0,
                    "ub": 1,
                },
            }
        },
    )


class TestActiveConstraintsExcludeZ(unittest.TestCase):
    def test_z_dropped(self) -> None:
        ctx = _toy_context()
        active = _active_constraint_names(ctx)  # type: ignore[arg-type]
        self.assertEqual(
            active,
            ("A_bending_capacity", "B1a_deflection_by_wmax_capacity"),
        )


class TestCartRules(unittest.TestCase):
    def test_recovers_obvious_over_a_rule(self) -> None:
        rows = []
        for i in range(80):
            rows.append(
                {
                    "global_index": i,
                    "pattern": "A_bending_capacity",
                    "leading": "A_bending_capacity",
                    "best_action": "inc__geom_t_mm",
                    "over__A_bending_capacity": 1,
                    "over__B1a_deflection_by_wmax_capacity": 0,
                }
            )
        for i in range(80, 120):
            rows.append(
                {
                    "global_index": i,
                    "pattern": "B1a_deflection_by_wmax_capacity",
                    "leading": "B1a_deflection_by_wmax_capacity",
                    "best_action": "inc__geom_nt",
                    "over__A_bending_capacity": 0,
                    "over__B1a_deflection_by_wmax_capacity": 1,
                }
            )
        tree = _fit_cart(
            rows,
            feature_names=[
                "over__A_bending_capacity",
                "over__B1a_deflection_by_wmax_capacity",
            ],
            leading_names=[
                "A_bending_capacity",
                "B1a_deflection_by_wmax_capacity",
            ],
            max_depth=2,
            min_samples_leaf=10,
        )
        prior = {
            "inc__geom_t_mm": 80 / 120,
            "inc__geom_nt": 40 / 120,
        }
        rules = _extract_paths(
            tree, prior=prior, min_support=20, min_lift=1.05
        )
        actions = {r["action"] for r in rules}
        self.assertIn("inc__geom_t_mm", actions)
        self.assertTrue(
            any(
                "A_bending_capacity" in r["path_str"]
                or r["path_str"] == "(root)"
                for r in rules
                if r["action"] == "inc__geom_t_mm"
            )
        )

class TestNeighbourScoringDuckDB(unittest.TestCase):
    def test_inc_t_reduces_max_util(self) -> None:
        duckdb = import_duckdb()
        conn = duckdb.connect(database=":memory:")
        conn.execute(
            "CREATE TABLE toy ("
            "global_index BIGINT, "
            "analysis_ok BOOLEAN, "
            '"idx__geom_t_mm" INTEGER, '
            '"idx__geom_nt" INTEGER, '
            '"util__A_bending_capacity" DOUBLE, '
            '"util__B1a_deflection_by_wmax_capacity" DOUBLE, '
            '"util__Z1_nt_dt_combination_capacity" DOUBLE'
            ")"
        )
        # Grid: t=0,1,2 × nt=0,1
        # Only A varies with t: util_A = 1.2 - 0.15*t  → inc t repairs
        rows = []
        gid = 0
        for t in (0, 1, 2):
            for nt in (0, 1):
                rows.append(
                    (
                        gid,
                        True,
                        t,
                        nt,
                        1.2 - 0.15 * t,
                        0.5,
                        0.4,
                    )
                )
                gid += 1
        conn.executemany("INSERT INTO toy VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        conn.execute('CREATE VIEW "bf_parts" AS SELECT * FROM toy')

        ctx = _toy_context()
        engine = BruteForceQueryEngine(
            context=ctx,  # type: ignore[arg-type]
            columns=None,  # type: ignore[arg-type]
            connection=conn,
            view_name="bf_parts",
        )
        active = _active_constraint_names(ctx)  # type: ignore[arg-type]
        _n_base, n_seeds = _sample_infeasible_seeds(
            engine, ctx, active, sample_rows=100  # type: ignore[arg-type]
        )
        self.assertGreater(n_seeds, 0)
        n_labeled = _score_actions(engine, ctx, active)  # type: ignore[arg-type]
        self.assertGreater(n_labeled, 0)
        labeled = _load_labeled_rows(engine, active)
        self.assertTrue(
            any(r["best_action"] == "inc__geom_t_mm" for r in labeled)
        )
        moves = _pattern_move_table(engine, min_support=1, min_lift=0.0)
        self.assertTrue(
            any(m["action"] == "inc__geom_t_mm" for m in moves)
        )
        conn.close()


class TestArtifactWriter(unittest.TestCase):
    def test_writes_files(self) -> None:
        doc = {
            "ok": True,
            "overall_status": "success",
            "metadata": {
                "max_depth": 3,
                "min_support": 10,
                "min_lift": 1.2,
                "sample_rows": 100,
                "n_seeds": 10,
                "n_labeled": 8,
            },
            "tree_rules": [
                {
                    "path_str": "over__A_bending_capacity=1",
                    "action": "inc__geom_t_mm",
                    "support": 20,
                    "confidence": 0.8,
                    "lift": 1.5,
                }
            ],
            "pattern_moves": [],
        }
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            exp = Path(tmp)
            written = write_constraint_decision_tree_artifacts(exp, doc)
            self.assertTrue(written["results_json"].is_file())
            self.assertTrue(written["rules_md"].is_file())
            self.assertTrue(written["rules_csv"].is_file())
            self.assertTrue(written["success"].is_file())


class TestOptions(unittest.TestCase):
    def test_rejects_bad_depth(self) -> None:
        with self.assertRaises(ValueError):
            DecisionTreeOptions(max_depth=0)


if __name__ == "__main__":
    unittest.main()
