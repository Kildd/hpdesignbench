"""
CSV-driven specification loaders for optimization parameters, constraints,
problems, and the materials registry.

Author: Max Dombrowski

Modifications by Elliot Melcer:

* main modified to work with file structure
* load_param_defaults: added lookup role
* loads constraint defaults when they are omitted from problem_list.csv
* docstrings
"""

# src/slab_benchmark/core/import_specs.py
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Any

from core.ioh_core.io_util import _read_rows, _enf, _to_float, _to_bool


def load_param_defaults(path: str | Path) -> Dict[str, dict]:
    """
    Read ``parameter_defaults.csv`` and return a parameter specification dict.

    Parameters
    ----------
    path : str or Path
        Path to ``parameter_defaults.csv``.

    Returns
    -------
    dict
        Mapping ``{name: spec}`` where each ``spec`` contains:

        For roles ``"var"`` and ``"fixed"``:

        - ``"domain"`` — ``"catalog"`` or ``"grid"``.
        - ``"values"`` — list of admissible values.
        - ``"lb"`` — lower bound index into ``"values"`` (0-based) [-].
        - ``"ub"`` — upper bound index into ``"values"`` (0-based) [-].
        - ``"fixed_idx"`` — index used when ``role="fixed"`` (0-based) [-].
        - ``"role"`` — ``"var"``, ``"fixed"``, or ``"lookup"``.
        - ``"value_type"`` — ``"float"``, ``"int"``, or ``"str"``.

        For role ``"lookup"``, additionally:

        - ``"lookup_key"`` — name of the parameter this one mirrors.

    Raises
    ------
    ValueError
        If a parameter has an invalid role, unknown domain, empty value
        list, non-positive grid count, or if a lookup parameter has no
        ``lookup_key``, references an unknown key, references another
        lookup (chained lookups are not supported), or has a value list
        of different length than its key.

    Notes
    -----
    Required CSV columns:
    ``name``, ``domain``, ``values``, ``grid_start``, ``grid_step``,
    ``grid_count``, ``default_lb``, ``default_ub``, ``default_fixed``,
    ``role``, ``value_type``, ``lookup_key``.

    Domain rules:

    - ``"catalog"`` — uses the ``values`` column as a ``|``-separated list.
    - ``"grid"`` — generates values from ``grid_start``, ``grid_step``,
      ``grid_count``; not allowed for ``role="lookup"``.

    Role rules:

    - ``"var"`` — optimization variable; ``lb`` / ``ub`` are active.
    - ``"fixed"`` — constant per problem; ``fixed_idx`` is active.
    - ``"lookup"`` — derived from another parameter via ``lookup_key``;
      never appears in ``problem_list.csv``; resolved in ``decode()``.

    ``lb``, ``ub``, and ``fixed_idx`` are 0-based indices into ``values``.
    ``value_type`` defaults to ``"float"`` if the column is empty.
    """
    path = Path(path)
    spec: Dict[str, dict] = {}

    expected = [
        "name", "domain", "values",
        "grid_start", "grid_step", "grid_count",
        "default_lb", "default_ub", "default_fixed",
        "role", "value_type", "lookup_key"
        # lookup_key is optional at the header level — _read_rows allows extra cols,
        # and we access it with r.get() below.
    ]

    for r in _read_rows(path, expected):
        name = r["name"]
        dom = (r["domain"] or "").lower()
        role = (r["role"] or "var").lower()

        if role not in ("var", "fixed", "lookup"):
            raise ValueError(
                f"{path}: parameter '{name}' has invalid role '{role}' "
                f"(use 'var', 'fixed', or 'lookup')."
            )

        value_type = (r.get("value_type") or "float").strip().lower()

        # ------------------------------------------------------------------
        # Parse the value list (same logic for all roles)
        # ------------------------------------------------------------------
        if dom == "catalog":
            vals_str = (r["values"] or "").strip('"').strip()
            if not vals_str:
                raise ValueError(
                    f"{path}: parameter '{name}' uses domain 'catalog' but has empty 'values'."
                )
            if value_type == "str":
                vals = [v.strip() for v in vals_str.split("|")]
            elif value_type == "int":
                try:
                    vals = [int(v) for v in vals_str.split("|")]
                except ValueError as e:
                    raise ValueError(
                        f"{path}: parameter '{name}' has non-integer catalog value."
                    ) from e
            else:  # float (default)
                try:
                    vals = [float(v) for v in vals_str.split("|")]
                except ValueError as e:
                    raise ValueError(
                        f"{path}: parameter '{name}' has non-numeric catalog value."
                    ) from e

        elif dom == "grid":
            if role == "lookup":
                raise ValueError(
                    f"{path}: parameter '{name}' has role 'lookup' but domain 'grid'. "
                    f"Lookup parameters must use domain 'catalog' so values are positionally aligned."
                )
            try:
                start = float(r["grid_start"])
                step  = float(r["grid_step"])
                count = int(float(r["grid_count"]))
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"{path}: parameter '{name}' has invalid grid spec."
                ) from e
            if count <= 0:
                raise ValueError(
                    f"{path}: parameter '{name}' has non-positive grid_count={count}."
                )
            vals = [start + i * step for i in range(count)]

        else:
            raise ValueError(
                f"{path}: parameter '{name}' has unknown domain '{dom}' "
                f"(use 'catalog' or 'grid')."
            )

        L = len(vals)
        if L == 0:
            raise ValueError(
                f"{path}: parameter '{name}' produced an empty value list."
            )

        # ------------------------------------------------------------------
        # Role: lookup — store values + lookup_key, skip lb/ub/fixed
        # ------------------------------------------------------------------
        if role == "lookup":
            lookup_key = (r.get("lookup_key") or "").strip()
            if not lookup_key:
                raise ValueError(
                    f"{path}: parameter '{name}' has role 'lookup' but no lookup_key specified."
                )
            spec[name] = {
                "domain":     dom,
                "values":     vals,
                "role":       "lookup",
                "value_type": value_type,
                "lookup_key": lookup_key,
                # Harmless sentinels so any code iterating the model doesn't KeyError:
                "lb":         0,
                "ub":         L - 1,
                "fixed_idx":  0,
            }
            continue  # skip lb/ub/fixed_idx parsing below

        # ------------------------------------------------------------------
        # Role: var / fixed — parse lb/ub/fixed_idx as before
        # ------------------------------------------------------------------
        def clamp_idx(s: str, dflt: int) -> int:
            # empty cell -> default; otherwise clamp to [0, L-1]
            return dflt if s == "" else max(0, min(int(float(s)), L - 1))

        lb = clamp_idx(r["default_lb"], 0)
        ub = clamp_idx(r["default_ub"], L - 1)
        if lb > ub:
            lb, ub = ub, lb
        fixed = clamp_idx(r["default_fixed"], lb)

        spec[name] = {
            "domain":     dom,
            "values":     vals,
            "lb":         lb,
            "ub":         ub,
            "fixed_idx":  fixed,
            "role":       role,
            "value_type": value_type,
        }

    # ----------------------------------------------------------------------
    # Post-load validation: every lookup_key must reference a known parameter
    # ----------------------------------------------------------------------
    for name, m in spec.items():
        if m["role"] == "lookup":
            key = m["lookup_key"]
            if key not in spec:
                raise ValueError(
                    f"{path}: lookup parameter '{name}' references unknown lookup_key '{key}'."
                )
            if spec[key]["role"] == "lookup":
                raise ValueError(
                    f"{path}: lookup parameter '{name}' references another lookup '{key}'. "
                    f"Chained lookups are not supported."
                )
            key_len = len(spec[key]["values"])
            if len(m["values"]) != key_len:
                raise ValueError(
                    f"{path}: lookup parameter '{name}' has {len(m['values'])} values "
                    f"but its key '{key}' has {key_len}. Lists must be the same length."
                )

    return spec


def build_space(model: Dict[str, dict]) -> tuple[list[str], list[int], list[int]]:
    """
    Extract the optimization variable space from a full parameter model.

    Parameters
    ----------
    model : dict
        Parameter model dict as returned by :func:`load_param_defaults`
        (possibly with per-problem overlays applied by
        :func:`load_problems_combined`).

    Returns
    -------
    tuple[list[str], list[int], list[int]]
        ``(var_names, lb_idx_list, ub_idx_list)`` — ordered lists of
        variable names and their lower/upper bound indices.
    """
    var_names, lb, ub = [], [], []
    for n, m in model.items():
        if m["role"] == "var":
            var_names.append(n)
            lb.append(m["lb"])
            ub.append(m["ub"])
    return var_names, lb, ub


def make_decode(model: Dict[str, dict], var_names: List[str]):
    """
    Build a ``decode(x_idx) -> dict`` function for a given parameter model.

    The returned decoder merges fixed parameters, variable parameters
    (index-clamped to their ``lb``/``ub``), and lookup parameters
    (resolved positionally from their key parameter).

    Parameters
    ----------
    model : dict
        Full parameter model dict (all roles included) as returned by
        :func:`load_param_defaults` with per-problem overlays.
    var_names : list[str]
        Ordered list of optimization variable names (must match the
        ``"var"``-role entries in ``model``).

    Returns
    -------
    callable
        ``decode(x_idx: list[int]) -> dict`` — maps an integer index
        vector to a named parameter dictionary. The length of ``x_idx``
        must equal ``len(var_names)``.

    Raises
    ------
    ValueError
        If ``x_idx`` has the wrong length; or if a lookup key value is not
        found in that key's catalog (no silent fallback to index 0).
    KeyError
        If a lookup parameter references a key not yet resolved in the
        output dict.

    Notes
    -----
    Parameter names must match the CSV column names exactly; no aliases
    or synonyms are supported.
    """
    n_vars = len(var_names)

    def decode(x_idx: List[int]) -> Dict[str, Any]:
        if len(x_idx) != n_vars:
            raise ValueError(
                f"decode: got {len(x_idx)} indices, expected {n_vars} ({var_names})."
            )

        full: Dict[str, Any] = {}

        # 1) fixed parameters
        for name, m in model.items():
            if m["role"] == "fixed":
                full[name] = m["values"][m["fixed_idx"]]

        # 2) variable parameters
        for name, k in zip(var_names, x_idx):
            m = model[name]
            kk = max(m["lb"], min(int(k), m["ub"]))
            full[name] = m["values"][kk]

        # 3) lookup params — resolved from same index as their key
        for name, m in model.items():
            if m["role"] == "lookup":
                key = m["lookup_key"]
                if key not in full:
                    raise KeyError(f"Lookup param '{name}' references unknown key '{key}'.")
                key_vals = model[key]["values"]
                # Find the index of the resolved key value
                resolved_key_val = full[key]
                try:
                    idx = key_vals.index(resolved_key_val)
                except ValueError as e:
                    raise ValueError(
                        f"Lookup param '{name}': key '{key}' value "
                        f"{resolved_key_val!r} is not in catalog {list(key_vals)}. "
                        f"Silent fallback to index 0 is not allowed."
                    ) from e
                if idx >= len(m["values"]):
                    raise ValueError(
                        f"Lookup param '{name}': index {idx} from key '{key}' "
                        f"exceeds value catalog length {len(m['values'])}."
                    )
                full[name] = m["values"][idx]

        return full

    return decode


def load_constraint_defaults(path: str | Path) -> Dict[str, dict]:
    """
    Read ``constraint_defaults.csv`` and return a constraint specification dict.

    Parameters
    ----------
    path : str or Path
        Path to ``constraint_defaults.csv``.

    Returns
    -------
    dict
        Mapping ``{name: spec}`` where each ``spec`` contains:

        - ``"kind"`` — constraint type string (e.g. ``"real"`` or
          ``"integer"``).
        - ``"enforced"`` — :class:`~ioh.iohcpp.ConstraintEnforcement`
          enum value.
        - ``"weight"`` — penalty weight [-].
        - ``"exponent"`` — penalty exponent [-].
        - ``"active"`` — whether the constraint is active by default [-].
    """
    path = Path(path)
    spec: Dict[str, dict] = {}
    expected = ["name", "kind", "enforced", "weight", "exponent", "active_default"]

    for r in _read_rows(path, expected):
        spec[r["name"]] = {
            "kind": (r["kind"] or "").lower(),
            "enforced": _enf(r["enforced"] or "SOFT"),
            "weight": _to_float(r["weight"], 1.0),
            "exponent": _to_float(r["exponent"], 1.0),
            "active": _to_bool(r["active_default"], True), # modified by: Elliot Melcer
        }
    return spec


def _parse_problem_bool(value: str, *, default: bool, context: str) -> bool:
    """
    Parse an optional problem-list boolean without accepting misspellings.

    Parameters
    ----------
    value : str
        CSV cell value. An empty cell inherits ``default``.
    default : bool
        Value returned for an empty cell.
    context : str
        Row description included in validation errors.

    Returns
    -------
    bool
        Parsed or inherited value.

    Raises
    ------
    ValueError
        If a non-empty value is not a recognized boolean token.
    """
    if value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(
        f"{context}: invalid boolean {value!r}; use true/false, yes/no, or 1/0."
    )


def load_problems_combined(
    param_defaults: Dict[str, dict],
    constr_defaults: Dict[str, dict],
    problem_list_csv: str | Path,
) -> Dict[str, dict]:
    """
    Read ``problem_list.csv`` and resolve its default and sparse overlays.

    The file must begin with a contiguous reserved ``problem_id=default``
    group. That non-runnable group overlays the parameter and constraint
    defaults once. Every ordinary problem is then a deep copy of the resolved
    default with only its own rows applied. A row with only ``problem_id`` and
    optionally ``label`` declares a runnable problem without changing it.

    Parameters
    ----------
    param_defaults : dict
        Base parameter model as returned by :func:`load_param_defaults`.
    constr_defaults : dict
        Base constraint model as returned by :func:`load_constraint_defaults`.
    problem_list_csv : str or Path
        Path to ``problem_list.csv``.

    Returns
    -------
    dict
        Mapping ``{problem_id: info}`` where each ``info`` dict contains:

        The reserved ``"default"`` ID is excluded. Each ordinary ``info``
        contains:

        - ``"model"`` — resolved default parameter model with this problem's
          roles, ``lb``, ``ub``, and ``fixed_idx`` overlays.
        - ``"constraints"`` — resolved default constraint config with this
          problem's active/enforced/weight/exponent overlays.
        - ``"label"`` — human-readable problem label string.

    Raises
    ------
    KeyError
        If a row references an unknown parameter or constraint name.
    ValueError
        If the default group is missing/misplaced, a declaration is malformed,
        a row/boolean/role is invalid, labels conflict, an override is
        duplicated, or a ``"lookup"`` parameter is referenced.
    """
    problem_list_csv = Path(problem_list_csv)
    expected = [
        "problem_id", "type", "name", "role",
        "lb_idx", "ub_idx", "fixed_idx",
        "active", "enforced", "weight", "exponent",
        "label",
    ]
    rows = list(_read_rows(problem_list_csv, expected))

    if rows[0]["problem_id"] != "default":
        raise ValueError(
            f"{problem_list_csv}: first data row must begin the reserved "
            "'default' problem group."
        )

    default_end = 0
    while default_end < len(rows) and rows[default_end]["problem_id"] == "default":
        default_end += 1
    if any(r["problem_id"] == "default" for r in rows[default_end:]):
        raise ValueError(
            f"{problem_list_csv}: the reserved 'default' group must be contiguous "
            "and appear before every ordinary problem."
        )

    default_info = {
        "model": deepcopy(param_defaults),
        "constraints": deepcopy(constr_defaults),
        "label": "default",
    }
    by_problem: Dict[str, dict] = {}
    labels: Dict[str, str] = {}
    seen_overrides: Dict[str, set[tuple[str, str]]] = {}
    default_has_overlay = False

    def apply_row(problem_ID: str, info: dict, r: dict) -> None:
        """
        Apply one validated overlay row to an already copied problem.

        Parameters
        ----------
        problem_ID : str
            Reserved default ID or ordinary problem ID owning the row.
        info : dict
            Mutable copied model/constraint bundle receiving the overlay.
        r : dict
            Clean CSV row returned by :func:`_read_rows`.
        """
        nonlocal default_has_overlay
        typ = (r["type"] or "").lower()
        name = r["name"]
        context = f"{problem_list_csv}: problem {problem_ID}"

        label = r["label"]
        if label:
            previous = labels.get(problem_ID)
            if previous is not None and previous != label:
                raise ValueError(
                    f"{context} has conflicting labels {previous!r} and {label!r}."
                )
            labels[problem_ID] = label

        if typ == "" and name == "":
            non_label_fields = (
                "role", "lb_idx", "ub_idx", "fixed_idx",
                "active", "enforced", "weight", "exponent",
            )
            populated = [field for field in non_label_fields if r[field] != ""]
            if populated:
                raise ValueError(
                    f"{context} declaration-only row has values in {populated}; "
                    "type and name are required for an override."
                )
            return
        if typ == "" or name == "":
            raise ValueError(
                f"{context} must provide both type and name, or leave both blank "
                "for a declaration-only row."
            )
        if typ not in {"param", "constraint"}:
            raise ValueError(
                f"{context} has unknown row type {typ!r}; use 'param' or "
                "'constraint'."
            )

        override_key = (typ, name)
        problem_seen = seen_overrides.setdefault(problem_ID, set())
        if override_key in problem_seen:
            raise ValueError(
                f"{context} duplicates the {typ} override for {name!r}."
            )
        problem_seen.add(override_key)
        if problem_ID == "default":
            default_has_overlay = True

        if typ == "param":
            if name not in info["model"]:
                raise KeyError(
                    f"{context} refers to unknown parameter {name!r}."
                )
            m = info["model"][name]

            # lookup params are resolved automatically; users must not add them to problem_list.csv
            if m["role"] == "lookup":
                raise ValueError(
                    f"{problem_list_csv}: problem {problem_ID} refers to '{name}' which has "
                    f"role 'lookup'. Lookup parameters are resolved automatically from "
                    f"'{m['lookup_key']}' and cannot be set in problem_list.csv."
                )

            role = (r["role"] or m["role"]).lower()
            if role not in ("var", "fixed"):
                raise ValueError(
                    f"{context}, param {name!r} has invalid role {role!r}."
                )
            m["role"] = role

            L = len(m["values"])
            clamp = lambda s, d: d if s == "" else max(0, min(int(float(s)), L - 1))
            if role == "var":
                m["lb"] = clamp(r["lb_idx"], m["lb"])
                m["ub"] = clamp(r["ub_idx"], m["ub"])
                if m["lb"] > m["ub"]:
                    m["lb"], m["ub"] = m["ub"], m["lb"]
            else:
                m["fixed_idx"] = clamp(r["fixed_idx"], m["fixed_idx"])

        elif typ == "constraint":
            if name not in info["constraints"]:
                raise KeyError(
                    f"{context} refers to unknown constraint {name!r}."
                )
            base = info["constraints"][name].copy()
            base["active"] = _parse_problem_bool(
                r["active"],
                default=base["active"],
                context=f"{context}, constraint {name!r}, active",
            ) # modified by: Elliot Melcer
            if r["enforced"]:
                prev = base["enforced"].name if hasattr(base["enforced"], "name") else "SOFT"
                base["enforced"] = _enf(r["enforced"], prev)
            if r["weight"]:
                base["weight"] = float(r["weight"])
            if r["exponent"]:
                base["exponent"] = float(r["exponent"])
            info["constraints"][name] = base

    for row in rows[:default_end]:
        apply_row("default", default_info, row)
    if not default_has_overlay:
        raise ValueError(
            f"{problem_list_csv}: reserved 'default' group must contain at least "
            "one parameter or constraint overlay."
        )

    for r in rows[default_end:]:
        problem_ID = r["problem_id"]
        if not problem_ID:
            raise ValueError(f"{problem_list_csv}: ordinary problem_id cannot be blank.")
        if problem_ID not in by_problem:
            by_problem[problem_ID] = deepcopy(default_info)
            by_problem[problem_ID]["label"] = problem_ID
        apply_row(problem_ID, by_problem[problem_ID], r)

    if not by_problem:
        raise ValueError(
            f"{problem_list_csv}: at least one ordinary problem must be declared."
        )
    for problem_ID, info in by_problem.items():
        info["label"] = labels.get(problem_ID, problem_ID)

    return by_problem


def load_materials_registry(path: str | Path) -> Dict[str, dict]:
    """
    Read ``materials.csv`` and return a materials registry dictionary.

    If the file does not exist, an empty registry is returned (the
    materials file is optional).

    Parameters
    ----------
    path : str or Path
        Path to ``materials.csv``.

    Returns
    -------
    dict
        Mapping ``{name: properties}`` where ``name`` is the unique
        material identifier (must match catalog values in
        ``parameter_defaults.csv``) and ``properties`` is a dict with
        the following keys (``None`` if the CSV cell is empty):

        - ``"type"`` — ``"concrete"``, ``"reinforcement"``, ``"infill"``,
          ``"insulation"``, or ``"screed"``.
        - ``"density"`` — density [kg/m³].
        - ``"gwp"`` — GWP [kg CO₂e/kg] (reinforcement / finishes).
        - ``"gwp_m3"`` — GWP [kg CO₂e/m³] (concrete).
        - ``"cost"`` — cost [€/kg] (reinforcement / finishes).
        - ``"cost_m3"`` — cost [€/m³] (concrete).
        - ``"comment"`` — free-text Kommentar from the CSV.
        - ``"f_ck"``, ``"alpha_cc"``, ``"gamma_c"`` — concrete only.
        - ``"f_yk"``, ``"f_tk"``, ``"E_tex"`` — reinforcement.
        - ``"eps_y"``, ``"eps_u"`` — tensile strains [‰]
          (converted to absolute strain with ``/1000`` when building CFRP).
        - ``"eps_y_c"``, ``"eps_u_c"`` — compressive strains [‰]
          (stored, not used in analysis yet).
        - ``"alpha_tex"`` — textile conversion factor α_tex [-]
          (ULS: ``f_td = α_tex · f_tk / γ_tex``).
        - ``"gamma_tex"`` — partial safety factor γ_tex [-].
        - ``"E_dyn"`` — dynamic stiffness (insulation) [MN/m³].

        Header names may include unit annotations in brackets
        (e.g. ``density [kg/m3]``); these are stripped for matching.
        Values are expected with decimal points; commas are still
        accepted as a fallback.
    """
    path = Path(path)
    if not path.exists():
        # Return empty registry if file doesn't exist (allows optional materials.csv)
        return {}

    # CSV header stem (after unit strip) -> registry key
    header_map = {
        "name": "name",
        "type": "type",
        "density": "density",
        "gwp": "gwp",
        "gwp_m3": "gwp_m3",
        "cost": "cost",
        "cost_m3": "cost_m3",
        "kommentar": "comment",
        "comment": "comment",
        "f_ck": "f_ck",
        "alpha_cc": "alpha_cc",
        "gamma_c": "gamma_c",
        "f_yk": "f_yk",
        "f_tk": "f_tk",
        "e_tex": "E_tex",
        "eps_y": "eps_y",
        "eps_u": "eps_u",
        "eps_y_c": "eps_y_c",
        "eps_u_c": "eps_u_c",
        "alpha_tex": "alpha_tex",
        "gamma_tex": "gamma_tex",
        "e_dyn": "E_dyn",
    }

    def header_stem(raw: str) -> str:
        h = (raw or "").strip().lower()
        if "[" in h:
            h = h.split("[", 1)[0].strip()
        return h

    def parse_numeric(s: str) -> float | None:
        """Parse a numeric cell (points preferred; commas still accepted)."""
        s = (s or "").strip()
        if not s:
            return None
        s = s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return None

    registry: Dict[str, dict] = {}
    for r in _read_rows(path, expected_keys=None):
        mapped: Dict[str, str] = {}
        for raw_key, value in r.items():
            logical = header_map.get(header_stem(raw_key))
            if logical is not None:
                mapped[logical] = value

        if "name" not in mapped or "type" not in mapped:
            raise ValueError(
                f"{path}: materials row missing required columns "
                f"['name', 'type']. Found headers: {list(r.keys())}"
            )

        name = (mapped.get("name") or "").strip()
        if not name:
            continue  # Skip rows without name

        mat_type = (mapped.get("type") or "").lower().strip()
        comment = (mapped.get("comment") or "").strip() or None

        registry[name] = {
            "type": mat_type,
            "density": parse_numeric(mapped.get("density", "")),
            "gwp": parse_numeric(mapped.get("gwp", "")),
            "gwp_m3": parse_numeric(mapped.get("gwp_m3", "")),
            "cost": parse_numeric(mapped.get("cost", "")),
            "cost_m3": parse_numeric(mapped.get("cost_m3", "")),
            "comment": comment,
            "f_ck": parse_numeric(mapped.get("f_ck", "")),
            "alpha_cc": parse_numeric(mapped.get("alpha_cc", "")),
            "gamma_c": parse_numeric(mapped.get("gamma_c", "")),
            "f_yk": parse_numeric(mapped.get("f_yk", "")),
            "f_tk": parse_numeric(mapped.get("f_tk", "")),
            "E_tex": parse_numeric(mapped.get("E_tex", "")),
            "eps_y": parse_numeric(mapped.get("eps_y", "")),
            "eps_u": parse_numeric(mapped.get("eps_u", "")),
            "eps_y_c": parse_numeric(mapped.get("eps_y_c", "")),
            "eps_u_c": parse_numeric(mapped.get("eps_u_c", "")),
            "alpha_tex": parse_numeric(mapped.get("alpha_tex", "")),
            "gamma_tex": parse_numeric(mapped.get("gamma_tex", "")),
            "E_dyn": parse_numeric(mapped.get("E_dyn", "")),
        }

    if not registry:
        raise ValueError(f"{path}: no material data rows found.")

    return registry


#----------------------------------------------------------------------------------------------------------#
# Example to test if import_specs.py is working correctly
#----------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent  # .../core/ioh_core
    CORE = HERE.parent  # .../core
    PKG_ROOT = CORE.parent  # project root
    SLABS_DIR = PKG_ROOT / "slab_construction" / "slabs"

    slab_type = "hp_slab"
    slab_dir = SLABS_DIR / slab_type

    print("HERE      :", HERE)
    print("CORE      :", CORE)
    print("PKG_ROOT  :", PKG_ROOT)
    print("SLABS_DIR :", SLABS_DIR)
    print("slab_dir  :", slab_dir)
    print("exists?   :", slab_dir.exists())
    print()

    params_csv = slab_dir / "parameter_defaults.csv"
    constr_csv = slab_dir / "constraint_defaults.csv"
    problem_list_csv = slab_dir / "problem_list.csv"

    print("== File Paths ==")
    for label, p in [("params", params_csv), ("constr", constr_csv), ("problem_list", problem_list_csv)]:
        exists = "✓" if p.exists() else "✗"
        print(f"[{exists}] {label}: {p}")
    print()

    # Only proceed if files exist
    if not all(p.exists() for p in [params_csv, constr_csv, problem_list_csv]):
        print("ERROR: Not all required CSV files exist. Exiting.")
        exit(1)

    print("== Loading defaults ==")
    PDEF = load_param_defaults(params_csv)
    CDEF = load_constraint_defaults(constr_csv)
    print(f"Parameters loaded: {len(PDEF)}  |  Constraints loaded: {len(CDEF)}")

    some_params = list(PDEF.items())[:2]
    print("\nSample parameter defaults (first 2):")
    for name, spec in some_params:
        vals_preview = spec['values'][:5] if len(spec['values']) > 5 else spec['values']
        vals_suffix = '...' if len(spec['values']) > 5 else ''
        print(
            f"- {name}: role={spec['role']}, "
            f"values={vals_preview}{vals_suffix}, "
            f"lb={spec['lb']}, ub={spec['ub']}, fixed_idx={spec['fixed_idx']}, "
            f"value_type={spec.get('value_type')}"
        )

    some_constr = list(CDEF.items())[:2]
    print("\nSample constraint defaults (first 2):")
    for name, spec in some_constr:
        print(
            f"- {name}: kind={spec['kind']}, "
            f"enforced={getattr(spec['enforced'], 'name', spec['enforced'])}, "
            f"weight={spec['weight']}, exponent={spec['exponent']}, "
            f"active={spec['active']}" # modified by: Elliot Melcer
        )

    print("\n== Loading problems & overlays ==")
    PROBLEMS = load_problems_combined(PDEF, CDEF, problem_list_csv)
    print(f"Problems found: {len(PROBLEMS)}  -> {sorted(PROBLEMS.keys())}")

    if not PROBLEMS:
        print("WARNING: No problems defined.")
        exit(0)

    problem_ID = sorted(PROBLEMS.keys())[0]
    info = PROBLEMS[problem_ID]
    print(f"\n== Inspect problem {problem_ID} ({info['label']}) ==")

    var_names, lb_idx, ub_idx = build_space(info["model"])
    print("Variables:", var_names)
    print("lb_idx   :", lb_idx)
    print("ub_idx   :", ub_idx)

    fixed = {n: m for n, m in info["model"].items() if m["role"] == "fixed"}
    if fixed:
        print("\nFixed params in this problem:")
        for n, m in list(fixed.items())[:5]:  # Show first 5
            print(f"- {n}: fixed_idx={m['fixed_idx']} -> value={m['values'][m['fixed_idx']]}")
        if len(fixed) > 5:
            print(f"  ... and {len(fixed) - 5} more")

    if info["constraints"]:
        print("\nConstraints in this problem:")
        for cname, c in info["constraints"].items():
            enforced = getattr(c["enforced"], "name", c["enforced"])
            print(
                f"- {cname}: active={c['active']}, kind={c['kind']}, "
                f"enforced={enforced}, weight={c['weight']}, exponent={c['exponent']}"
            )
    else:
        print("\nNo constraints configured for this problem.")

    print("\n== Decode test ==")
    decode = make_decode(info["model"], var_names)
    try:
        mid = [(lo + (hi - lo) // 2) for lo, hi in zip(lb_idx, ub_idx)]
        print("x_idx (mid) =", mid)
        params = decode(mid)
        print("decoded params:")
        pprint(params)
    except ValueError as e:
        print("Decode error:", e)

    print("\n== Negative length check ==")
    try:
        bad = [0] * (len(var_names) + 1)
        decode(bad)
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print("OK (caught):", e)

    print("\n== Test Complete ==")