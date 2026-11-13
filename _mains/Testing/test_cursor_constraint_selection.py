"""
Shared constraint keep-list selector and subset RuntimeWarning.
"""
from __future__ import annotations

import ast
import unittest
import warnings
from pathlib import Path

from core.ioh_core.constraint_selection import (
    CONSTRAINT_SUBSET_WARNING,
    ConstraintSelectionError,
    select_constraint_names,
    warn_if_constraint_subset,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "core" / "ioh_core" / "constraint_selection.py"
_SEALED = (
    "A_bending_capacity",
    "B1a_deflection_by_wmax_capacity",
    "Z1_nt_dt_combination_capacity",
    "Z2_beam_theory_H_L_capacity",
    "Z3_beam_theory_B_L_capacity",
)
_FORBIDDEN = (
    "duckdb",
    "pflacco",
    "pyarrow",
    "SALib",
    "core.ioh_core.workflows.la",
    "core.ioh_core.workflows.brute_force.analysis.constraint_decision_tree",
)


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class TestConstraintSelection(unittest.TestCase):
    def test_source_does_not_import_ela_or_decision_tree(self) -> None:
        tree = ast.parse(
            MODULE_PATH.read_text(encoding="utf-8"),
            filename=str(MODULE_PATH),
        )
        imported = _imported_names(tree)
        for name in imported:
            for root in _FORBIDDEN:
                self.assertFalse(
                    name == root or name.startswith(root + "."),
                    msg=f"constraint_selection.py imports {name}",
                )

    def test_default_keeps_full_sealed_list_including_z(self) -> None:
        selected = select_constraint_names(_SEALED)
        self.assertEqual(selected, _SEALED)
        self.assertTrue(any(name.startswith("Z") for name in selected))

    def test_keep_list_omits_z_in_sealed_order(self) -> None:
        keep = (
            "A_bending_capacity",
            "B1a_deflection_by_wmax_capacity",
        )
        self.assertEqual(select_constraint_names(_SEALED, keep), keep)

    def test_wrong_relative_order_fails(self) -> None:
        with self.assertRaises(ConstraintSelectionError) as ctx:
            select_constraint_names(
                _SEALED,
                (
                    "Z1_nt_dt_combination_capacity",
                    "A_bending_capacity",
                ),
            )
        self.assertIn("relative order", str(ctx.exception))
        self.assertIsInstance(ctx.exception, ValueError)

    def test_unknown_name_fails(self) -> None:
        with self.assertRaises(ConstraintSelectionError) as ctx:
            select_constraint_names(_SEALED, ("not_a_constraint",))
        self.assertIn("not_a_constraint", str(ctx.exception))

    def test_empty_requested_fails(self) -> None:
        with self.assertRaises(ConstraintSelectionError):
            select_constraint_names(_SEALED, ())

    def test_duplicate_requested_fails(self) -> None:
        with self.assertRaises(ConstraintSelectionError):
            select_constraint_names(
                _SEALED,
                ("A_bending_capacity", "A_bending_capacity"),
            )

    def test_blank_sealed_fails(self) -> None:
        with self.assertRaises(ConstraintSelectionError):
            select_constraint_names(("A_bending_capacity", "  "))

    def test_empty_sealed_fails(self) -> None:
        with self.assertRaises(ConstraintSelectionError):
            select_constraint_names(())

    def test_duplicate_sealed_fails(self) -> None:
        with self.assertRaises(ConstraintSelectionError):
            select_constraint_names(("A", "A"))

    def test_full_list_does_not_warn(self) -> None:
        selected = select_constraint_names(_SEALED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_constraint_subset(_SEALED, selected)
        runtime = [item for item in caught if issubclass(item.category, RuntimeWarning)]
        self.assertEqual(runtime, [])

    def test_subset_emits_locked_runtime_warning(self) -> None:
        keep = ("A_bending_capacity",)
        selected = select_constraint_names(_SEALED, keep)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_constraint_subset(_SEALED, selected)
        runtime = [item for item in caught if issubclass(item.category, RuntimeWarning)]
        self.assertEqual(len(runtime), 1)
        self.assertEqual(str(runtime[0].message), CONSTRAINT_SUBSET_WARNING)

    def test_does_not_strip_z_by_prefix(self) -> None:
        keep = (
            "A_bending_capacity",
            "Z1_nt_dt_combination_capacity",
        )
        self.assertEqual(select_constraint_names(_SEALED, keep), keep)


if __name__ == "__main__":
    unittest.main()
