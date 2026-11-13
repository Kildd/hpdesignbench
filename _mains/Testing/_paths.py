"""Shared filesystem helpers for the unittest suite."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HP_SLAB_DIR = REPO_ROOT / "slab_construction" / "slabs" / "hp_slab"

PARAM_DEFAULTS_CSV = HP_SLAB_DIR / "parameter_defaults.csv"
CONSTRAINT_DEFAULTS_CSV = HP_SLAB_DIR / "constraint_defaults.csv"
PROBLEM_LIST_CSV = HP_SLAB_DIR / "problem_list.csv"
MATERIALS_CSV = HP_SLAB_DIR / "materials.csv"
