"""
Low-level CSV reading and type-parsing utilities for the IOH core.

Author: Max Dombrowski

Modifications by Elliot Melcer:

* main modified to work with file structure
* docstrings
"""

import csv
from typing import List
from pathlib import Path
from ioh.iohcpp import ConstraintEnforcement as CE


def _req_param(params: dict, name: str) -> float:
    """
    Retrieve a required numeric parameter from a params dictionary.

    Parameters
    ----------
    params : dict
        Parameter dictionary to look up.
    name : str
        Key of the required parameter.

    Returns
    -------
    float
        Numeric value of the parameter.

    Raises
    ------
    KeyError
        If ``name`` is not present in ``params``.
    ValueError
        If the value cannot be converted to ``float``.
    """
    try:
        val = params[name]
    except KeyError as e:
        raise KeyError(f"Required parameter '{name}' missing in params. "
                       f"Got keys: {list(params.keys())}") from e
    try:
        return float(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Parameter '{name}' must be numeric, got {val!r}") from e


def _read_rows(path: Path, expected_keys: List[str] | None = None):
    """
    Read a semicolon-delimited CSV file and yield cleaned row dicts.

    Header matching is case- and whitespace-insensitive; UTF-8 BOM is
    handled transparently. Extra columns beyond ``expected_keys`` are
    allowed. Fully blank lines are skipped silently.

    Parameters
    ----------
    path : Path
        Path to the ``.csv`` file.
    expected_keys : list[str] or None, optional
        Column names that must be present in the header (case-insensitive).
        If ``None``, all normalized header names are returned. Default is
        ``None``.

    Yields
    ------
    dict
        One dict per data row. Keys are the original ``expected_keys``
        strings (if provided) or the normalized header names (lowercase,
        stripped). Values are stripped strings (empty string for missing
        cells).

    Raises
    ------
    ValueError
        If the file is empty or has no header, if any ``expected_keys``
        are missing from the header, or if no data rows are found.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        # error check (1): empty file / missing header
        if reader.fieldnames is None:
            raise ValueError(f"{path}: file is empty or missing a header row.")

        raw_headers = reader.fieldnames
        # Normalize headers for matching (strip + lower)
        norm_headers = [(h or "").strip().lower() for h in raw_headers]
        # Map normalized header -> original header text
        norm_to_raw = {hn: raw_headers[i] for i, hn in enumerate(norm_headers)}

        # header requirement: all expected_keys must be present (allow extras)
        if expected_keys is not None:
            need = [(k or "").strip().lower() for k in expected_keys]
            missing = [k for k in need if k not in norm_to_raw]
            if missing:
                raise ValueError(
                    f"{path}: missing header columns: {missing}. "
                    f"Found headers: {raw_headers}"
                )

        yielded = 0
        for row in reader:
            # Skip fully blank lines
            if not any((v or "").strip() for v in row.values()):        # skips empty lines (included for visual clarity)
                continue

            if expected_keys is not None:
                # Build row using the canonical expected_keys order/names
                clean = {}
                for k in expected_keys:
                    kn = (k or "").strip().lower()
                    rk = norm_to_raw[kn]  # original header name
                    v = row.get(rk, "")
                    clean[k] = v.strip() if v is not None else ""
            else:
                # No schema provided: return normalized-key dict
                clean = {}
                for hn, rk in norm_to_raw.items():
                    v = row.get(rk, "")
                    clean[hn] = v.strip() if v is not None else ""

            yielded += 1
            yield clean

        # (3) no data rows
        if yielded == 0:
            raise ValueError(f"{path}: no data rows found (only header and/or blank lines).")


def _to_bool(s: str, default=False) -> bool:
    """
    Parse a string to a boolean value.

    Parameters
    ----------
    s : str
        Input string. Recognized truthy values: ``"1"``, ``"true"``,
        ``"yes"``, ``"y"`` (case-insensitive).
    default : bool, optional
        Value returned for empty strings. Default is ``False``.

    Returns
    -------
    bool
        Parsed boolean value.
    """
    return default if s == "" else s.lower() in ("1","true","yes","y")   # if string in s is one of the given, return true


def _to_float(s: str, default=None):
    """
    Parse a string to a float value.

    Parameters
    ----------
    s : str
        Input string.
    default : float or None, optional
        Value returned for empty strings. Default is ``None``.

    Returns
    -------
    float or None
        Parsed float, or ``default`` if ``s`` is empty.
    """
    return default if s == "" else float(s)


def _enf(s: str, default="HIDDEN"):
    """
    Parse a string to an IOH :class:`~ioh.iohcpp.ConstraintEnforcement` enum value.

    Parameters
    ----------
    s : str
        Enforcement keyword. Accepted values (case-insensitive):
        ``"NOT"``, ``"HIDDEN"``, ``"SOFT"``, ``"HARD"``, ``"OVERRIDE"``.
    default : str, optional
        Keyword used when ``s`` is empty or ``None``. Default is
        ``"HIDDEN"``.

    Returns
    -------
    ioh.iohcpp.ConstraintEnforcement
        Corresponding enum member.

    Raises
    ------
    KeyError
        If the resolved keyword does not match any known enforcement level.
    """
    k = (s or default).upper()
    return {"NOT": CE.NOT, "HIDDEN": CE.HIDDEN, "SOFT": CE.SOFT, "HARD": CE.HARD, "OVERRIDE": CE.OVERRIDE}[k]


# ----------------------------------------------------------------------------------------------------------#
# TEST RUN TO TEST THIS MODULE ONLY - REMOVE BEFORE WRAPPING WHOLE TOOL
# ----------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    # Construct Paths relative to this working directory
    HERE = Path(__file__).resolve().parent  # .../core/ioh_core
    CORE = HERE.parent  # .../core
    PKG_ROOT = CORE.parent  # project root
    SLABS_DIR = PKG_ROOT / "slab_construction" / "slabs"
    SLAB_DIR = SLABS_DIR / "hp_slab"  # your slab folder

    PARAM_CSV = SLAB_DIR / "parameter_defaults.csv"
    CONSTR_CSV = SLAB_DIR / "constraint_defaults.csv"
    PROBLEM_LIST_CSV = SLAB_DIR / "problem_list.csv"

    print("HERE      :", HERE)
    print("PKG_ROOT  :", PKG_ROOT)
    print("SLABS_DIR :", SLABS_DIR)
    print("SLAB_DIR  :", SLAB_DIR)
    print()
    print("PARAM_CSV  :", PARAM_CSV)
    print("CONSTR_CSV :", CONSTR_CSV)
    print("PROBLEM_LIST_CSV:", PROBLEM_LIST_CSV)
    print()

    # Check if files exist
    for label, p in [("PARAM", PARAM_CSV), ("CONSTR", CONSTR_CSV), ("PROBLEM_LIST", PROBLEM_LIST_CSV)]:
        exists = "✓" if p.exists() else "✗"
        print(f"[{exists}] {label}: {p.exists()}")

    print("\n" + "=" * 60)

    # Load and peek at a few rows from each file
    for label, p in [("PARAM", PARAM_CSV), ("CONSTR", CONSTR_CSV), ("PROBLEM_LIST", PROBLEM_LIST_CSV)]:
        if p.exists():
            rows = list(_read_rows(p))
            print(f"\n[{label}] rows={len(rows)}")
            for r in rows[:3]:  # show the first 3 rows
                print(r)
        else:
            print(f"\n[{label}] FILE NOT FOUND: {p}")