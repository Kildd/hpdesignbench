"""
ELA re-export of the shared constraint keep-list helper.
"""
from __future__ import annotations

import unittest

from core.ioh_core.constraint_selection import (
    select_constraint_names as shared_select,
    warn_if_constraint_subset as shared_warn,
)
from core.ioh_core.workflows.la.constraints import (
    ConstraintSelectionError,
    select_constraint_names,
    warn_if_constraint_subset,
)


class TestElaConstraintReexport(unittest.TestCase):
    def test_reexports_shared_helpers(self) -> None:
        self.assertIs(select_constraint_names, shared_select)
        self.assertIs(warn_if_constraint_subset, shared_warn)
        self.assertTrue(issubclass(ConstraintSelectionError, ValueError))

    def test_default_includes_z_names(self) -> None:
        sealed = (
            "A_bending_capacity",
            "Z1_nt_dt_combination_capacity",
        )
        self.assertEqual(select_constraint_names(sealed), sealed)


if __name__ == "__main__":
    unittest.main()
