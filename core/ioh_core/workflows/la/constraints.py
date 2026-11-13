"""
Select the ELA constraint set from sealed ``constraint_names``.

Default is the full sealed list (includes Z on hp_slab Problems 1–4).
An optional keep list must be non-empty and keep sealed relative order.
Do not reuse brute-force ``_active_constraint_names`` (that helper strips Z).
Do not hard-code an A–D-only default.

This module must not import DuckDB, pflacco, SALib, or the BF
decision-tree helper.
"""
from __future__ import annotations

from core.ioh_core.constraint_selection import (
    CONSTRAINT_SUBSET_WARNING,
    ConstraintSelectionError,
    select_constraint_names,
    warn_if_constraint_subset,
)

__all__ = [
    "CONSTRAINT_SUBSET_WARNING",
    "ConstraintSelectionError",
    "select_constraint_names",
    "warn_if_constraint_subset",
]
