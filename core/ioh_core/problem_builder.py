"""
IOH problem builder for HpDesignBench optimization studies.

Constructs IOHexperimenter problem instances from CSV-driven parameter and
constraint specifications, wires up the slab-specific analysis function
through an :class:`EvalContext` cache, and exposes the problems as a dict
ready for :func:`run_experiment`.

Author: Max Dombrowski

Modifications by Elliot Melcer:

* Pass materials to analysis.
* Let the slab module drop constraints that don't apply, e.g.
   ``hp_slab: defl_case_sls a/b`` cuts out unwanted checks from logging.
* Add docstrings.
"""

from pathlib import Path
from importlib import import_module
from importlib.resources import files
from typing import List, Sequence
from functools import partial           # used to bind constraints into cache logger per problem

import math
import ioh
from ioh import ProblemClass
from ioh.iohcpp import IntegerConstraint, RealConstraint, ConstraintEnforcement as CE

from .import_specs import (
    load_param_defaults, load_constraint_defaults,
    load_problems_combined, build_space, make_decode, load_materials_registry
)
from .cache_eval import EvalContext
from .io_util import _enf
from core.analysis_core.run_config import (
    analysis_run_config_from_params,
    apply_analysis_run_config,
)
from slab_construction.slabs.hp_slab.analysis import failed_evaluation_violation


# Fallback list in case dynamic discovery fails
_VALID_FALLBACK: tuple[str, ...] = (
    "flat_slab",
    "hp_slab",
    "ribbed_slab",
    "solid_slab_one_way",
    "solid_slab_two_way",
)


def _discover_slab_types() -> List[str]:
    """
    Discover available slab types by scanning ``slab_construction.slabs``.

    A slab type is recognized if its subfolder contains an ``analysis.py``
    module. Falls back to :data:`_VALID_FALLBACK` if dynamic discovery
    fails for any reason.

    Returns
    -------
    list[str]
        Sorted list of valid slab type names.
    """
    try:
        root = files("slab_construction.slabs")
        types: List[str] = []
        for entry in root.iterdir():
            if entry.is_dir() and (entry / "analysis.py").is_file():
                types.append(entry.name)
        return sorted(types) if types else list(_VALID_FALLBACK)
    except Exception:
        return list(_VALID_FALLBACK)


def _make_violation_reader(ctx, cname: str):
    """
    Return a single-argument violation reader for an IOH constraint callback.

    The returned callable fetches the raw (non-negative) violation
    ``max(0, η−1)`` for ``cname`` from the :class:`EvalContext` cache.
    Logger channels ``c__*`` separately publish utilizations η; do not
    confuse the two.

    Fail-closed contract
    --------------------
    Never return ``0.0`` because evaluation raised or produced a
    non-finite violation. Use
    ``failed_evaluation_violation(run_config.invalid_utilization)``
    (``max(0, sentinel − 1)`` from ``opt_invalid_utilization``) so the
    candidate is treated as
    strongly violated while keeping IOH callbacks alive. Missing violation
    entries on ``analysis_ok is False`` also return that derived value
    (e.g. material-lookup failure with an empty map).

    Parameters
    ----------
    ctx : EvalContext
        Cache context for the current problem.
    cname : str
        Constraint name key into ``rec["violations_"]``.

    Returns
    -------
    callable
        ``g(x) -> float`` — non-negative raw violation for ``cname``.
    """
    def g(x):
        # Never raise from here: IOH calls this from C++.
        try:
            r = ctx.get(x)
            violations = r.get("violations_") or {}
            sentinel = failed_evaluation_violation(ctx.run_config.invalid_utilization)
            if cname in violations:
                try:
                    v = float(violations[cname])
                except (TypeError, ValueError):
                    return sentinel
                if not math.isfinite(v) or v < 0.0:
                    return sentinel
                return v
            # Absent key: inactive/unknown → 0 on success; fail-closed on error.
            if r.get("analysis_ok") is False:
                return sentinel
            return 0.0
        except Exception as e:
            print(f"[violation_reader:{cname}] fail-closed after error: {e}")
            return failed_evaluation_violation(ctx.run_config.invalid_utilization)
    return g


def build_problems_for_slab(
    slab_dir: Path,
    slab_module,
    *,
    problem_ids: Sequence[str] | None = None,
    slab_type: str | None = None,
) -> dict[str, dict]:
    """
    Build all IOH problems for a slab type from its CSV specifications and
    analysis module.

    Reads ``parameter_defaults.csv``, ``constraint_defaults.csv``,
    ``problem_list.csv``, and ``materials.csv`` from ``slab_dir``.
    For each problem ID, creates an :class:`EvalContext` cache that binds
    the slab-specific ``analysis()`` function with the decoded parameters
    and materials, then wraps the whole thing in an IOH integer problem.

    Parameters
    ----------
    slab_dir : Path
        Path to the slab-type folder containing the four CSV files.
    slab_module : module
        Imported slab analysis module. Must expose an ``analysis(params,
        constraints, materials)`` callable and may optionally define
        ``resolve_active_constraints`` to filter constraints at
        problem-build time.
    problem_ids : sequence of str or None, optional
        When provided, build only these ordinary problem IDs (filter applied
        after CSV resolution and before IOH wrappers). ``None`` keeps the
        historical behaviour of building every ordinary problem returned by
        :func:`load_problems_combined`.
    slab_type : str or None, optional
        Logical slab type used for run-config and IOH problem naming.
        Defaults to ``slab_dir.name`` (packaged folders and
        ``inputs/{slab_type}/`` snapshots).

    Returns
    -------
    dict[str, dict]
        Mapping ``{problem_id: bundle}`` where each bundle contains:

        - ``"problem"`` — configured IOH integer problem instance.
        - ``"var_names"`` — list of optimization variable names.
        - ``"constraints"`` — full constraint config dict.
        - ``"decode"`` — ``decode(x_idx) -> dict`` callable.
        - ``"label"`` — human-readable problem label string.
        - ``"ctx"`` — :class:`EvalContext` instance for this problem.
        - ``"active_constraint_names"`` — list of active constraint names.
        - ``"analysis_fn"`` — bound slab ``analysis`` callable.
        - ``"lb_idx"`` — lower bound index vector for the variables.
        - ``"ub_idx"`` — upper bound index vector for the variables.
        - ``"model"`` — resolved per-problem parameter model.

    Raises
    ------
    ValueError
        If ``var_names``, ``lb_idx``, or ``ub_idx`` have mismatched
        lengths, or if a problem has zero optimization variables.
    KeyError
        If ``problem_ids`` requests an unknown ordinary problem ID.
    """
    slab_dir = Path(slab_dir)                           # slab-specific folder that contains the three .csv files with all info for problem setup
    resolved_slab_type = slab_type or slab_dir.name
    params_csv = slab_dir / "parameter_defaults.csv"
    constr_csv = slab_dir / "constraint_defaults.csv"
    problem_list_csv   = slab_dir / "problem_list.csv"
    materials_csv = slab_dir / "materials.csv"          # Addition by: Elliot Melcer

    PDEF = load_param_defaults(params_csv)              # Dict of all parameters with value lists, bounds, fixed values
    CDEF = load_constraint_defaults(constr_csv)         # Dict of constraint defaults
    MATERIALS = load_materials_registry(materials_csv)  # Dict of materials, Addition by: Elliot Melcer
    PROBLEMS = load_problems_combined(PDEF, CDEF, problem_list_csv)     # Dict[problem_ID] with problem-specific problem definitions from problem_list.csv

    if problem_ids is not None:
        # Filter before any IOH wrapping so unused problems stay cheap.
        selected: dict[str, dict] = {}
        missing: list[str] = []
        for problem_ID in problem_ids:
            if problem_ID == "default":
                raise KeyError(
                    "problem_ids cannot request the reserved non-runnable id 'default'"
                )
            if problem_ID not in PROBLEMS:
                missing.append(problem_ID)
                continue
            selected[problem_ID] = PROBLEMS[problem_ID]
        if missing:
            available = ", ".join(sorted(PROBLEMS)) or "(none)"
            raise KeyError(
                f"unknown problem_id(s) {missing}; available: {available}"
            )
        PROBLEMS = selected

    problems: dict[str, dict] = {}                      # container that saves problem definitions

    for problem_ID, info in PROBLEMS.items():
    # problem_ID = the problem ID
    # info = the per-problem config dict for that problem_ID
    # typical shape of PROBLEMS:
    # PROBLEMS = {
    #   "1": {
    #     "model": {                 # per-problem parameter model (roles/bounds/fixed already overlaid)
    #       "t_mm":   {"values": [...], "lb": 0, "ub": 5, "fixed_idx": 2, "role": "var",   "domain": "catalog"},
    #       "span_m": {"values": [...], "lb": 0, "ub": 8, "fixed_idx": 4, "role": "fixed", "domain": "grid"},
    #       # ...
    #     },...
        var_names, lb_idx, ub_idx = build_space(info["model"])      # extract var names and var bounds for this problem
        if not (len(var_names) == len(lb_idx) == len(ub_idx)):
            raise ValueError(
                f"problem {problem_ID}: mismatch in space: "
                f"len(var_names)={len(var_names)}, len(lb_idx)={len(lb_idx)}, len(ub_idx)={len(ub_idx)}" # catch malformed .CSV
            )

        n_vars = len(var_names)
        if n_vars == 0:
            raise ValueError(f"problem {problem_ID}: no variables (all fixed) — cannot build a 0-dim IOH problem.") # catch problems with 0 vars


        # log attributes of constraints to determine in analysis functions which constraints are activated/returned
        constraints_log = {
            name: {
                "active": c.get("active", True),
                "enforced": c.get("enforced", CE.HIDDEN),  # may already be enum; normalised below
                "weight": float(c.get("weight", 1)),
                "exponent": float(c.get("exponent", 1)),
                "kind": c.get("kind", "real"),
            }
            for name, c in info["constraints"].items()
        }

        # Resolve fixed parameter values for this problem (Addition by: Elliot Melcer)
        fixed_params = {}
        for pname, pmeta in info["model"].items():
            if pmeta.get("role") == "fixed":
                idx = pmeta.get("fixed_idx", 0)
                vals = pmeta.get("values", [])
                if 0 <= idx < len(vals):
                    fixed_params[pname] = vals[idx]

        # Let the slab module drop constraints that don't apply given those fixed params(Addition by: Elliot Melcer)
        if hasattr(slab_module, "resolve_active_constraints"):
            constraints_log = slab_module.resolve_active_constraints(fixed_params, constraints_log)

        # create decode, constraint list -> cache logger to store results from slab-specific analysis function
        decode = make_decode(info["model"], var_names)  # create a decide(x_idx) for this problem
        run_config = analysis_run_config_from_params(
            fixed_params,
            slab_type=resolved_slab_type,
        )
        apply_analysis_run_config(run_config)
        # bind constraints for this problem:
        analysis_fn = partial(
            slab_module.analysis,
            constraints=constraints_log,
            materials=MATERIALS,
            run_config=run_config,
        )          # pass materials + frozen run-config to analysis
        # slab-specific analysis(params) must return {"y","y_p","violations":{...}, ...}
        ctx = EvalContext(decode, analysis_fn, run_config=run_config)  # builds a cache for that specific problem - no cross-instance reuse!
        ctx.problem_id = str(problem_ID)

        # IOH constraints: log *raw violations* from cache; default to HIDDEN/1/1
        constraints = []
        for name, conf in constraints_log.items(): #Modified by: Elliot Melcer
            if not conf.get("active", True):
                continue
            # Build the 1-arg reader that pulls violations[name] from cache
            g = _make_violation_reader(ctx, name)

            # Effective logging settings (users can override via problem_list.csv if they insist)
            enforced = conf.get("enforced", CE.HIDDEN)
            try:
                enforced = _enf(enforced)  # accept enum/str/bool
            except Exception:
                enforced = CE.HIDDEN

            weight = float(conf.get("weight", 1.0) or 1.0)
            exponent = float(conf.get("exponent", 1.0) or 1.0)

            ctor = IntegerConstraint if conf.get("kind", "real") == "integer" else RealConstraint
            constraints.append(
                ctor(
                    g,
                    enforced=enforced,  # HIDDEN by default → logging only
                    weight=weight,      # 1.0 default → neutral
                    exponent=exponent,  # 1.0 default → neutral
                )
            )

        def objective_idx(x, _ctx=ctx, _n=n_vars, _problem_ID=problem_ID):  # n_vars = len(var_names)
            # Never raise from inside IOH callback; return a huge value if something is off.
            try:
                if len(x) != _n:
                    # mismatch can happen if an algorithm passes stale dimension;
                    # keep IOH alive and signal "awful" fitness
                    print(f"[objective:{_problem_ID}] len(x)={len(x)} != expected {_n} → returning +inf")
                    return float("inf")
                rec = _ctx.get(x)
                return float(rec.get("y_p", float("inf")))
            except Exception as e:
                print(f"[objective:{_problem_ID}] swallowed error: {e} → returning +inf")
                return float("inf")

        # --- Wrap IOH problem and set integer index bounds ---
        p = ioh.wrap_problem(
            function=objective_idx,
            name=f"{resolved_slab_type}_problem_{problem_ID}",
            problem_class=ProblemClass.INTEGER,
            dimension=n_vars,
            lb=0, ub=1,                 # temporary bounds for problem wrapping; overwrite below
            constraints=constraints,
        )
        # assign actual bounds arrays to IOH problem (no bound arrays possible in problem wrapper)
        p.bounds.lb = lb_idx
        p.bounds.ub = ub_idx

        active_constraint_names = [name for name, conf in constraints_log.items() if conf.get("active", True)] # Modified by: Elliot Melcer
        ctx.ensure_constraints(active_constraint_names)
        ctx.ensure_params(var_names)

        problems[problem_ID] = {           # a dict that collects all problem-specific relevant information
            "problem": p,
            "var_names": var_names,
            "constraints": constraints_log,
            "decode": decode,
            "label": info["label"],
            "ctx": ctx,
            "active_constraint_names": active_constraint_names,  # list of active constraints for debugging/information
            "analysis_fn": analysis_fn,
            "run_config": run_config,
            "lb_idx": list(lb_idx),
            "ub_idx": list(ub_idx),
            "model": info["model"],
        }

    return problems


def build_problems_for_slab_type(slab_type: str) -> dict[str, dict]:
    """
    Build all IOH problems for a slab type identified by name.

    Valid types are discovered by scanning ``slab_construction.slabs``
    for subfolders that contain an ``analysis.py`` module. Falls back to
    a hardcoded list if dynamic discovery fails.

    Parameters
    ----------
    slab_type : str
        Name of the slab type (e.g. ``"hp_slab"``). Must match a
        subfolder name in ``slab_construction.slabs``.

    Returns
    -------
    dict[str, dict]
        Problem bundle dict as returned by :func:`build_problems_for_slab`.

    Raises
    ------
    ValueError
        If ``slab_type`` is not among the discovered valid types.
    """
    valid = _discover_slab_types()
    if slab_type not in valid:
        raise ValueError(f"Unknown slab_type '{slab_type}'. Valid options: {', '.join(valid)}")

    # Import slab_benchmark.slabs.<slab_type>.analysis (must expose analysis)
    mod_path = f"slab_construction.slabs.{slab_type}.analysis"
    slab_module = import_module(mod_path)

    # Locate packaged CSV folder for this slab type
    slab_dir = files("slab_construction.slabs") / slab_type

    # Delegate to the original builder
    return build_problems_for_slab(slab_dir, slab_module)


def build_problem_from_snapshot(
    *,
    slab_type: str,
    problem_id: str,
    snapshot_dir: Path,
) -> dict:
    """
    Build one IOH problem bundle from a snapshotted slab CSV directory.

    Parameters
    ----------
    slab_type : str
        Slab package name such as ``\"hp_slab\"``. Used to import the
        analysis module; ``snapshot_dir`` must contain that slab's four
        CSV files.
    problem_id : str
        Ordinary problem ID to build (never ``\"default\"``).
    snapshot_dir : Path
        Directory containing the snapshotted
        ``parameter_defaults.csv``, ``constraint_defaults.csv``,
        ``problem_list.csv``, and ``materials.csv``.

    Returns
    -------
    dict
        Single problem bundle as returned by :func:`build_problems_for_slab`.

    Raises
    ------
    ValueError
        If ``slab_type`` is unknown.
    KeyError
        If ``problem_id`` is missing from the snapshotted problem list.
    """
    valid = _discover_slab_types()
    if slab_type not in valid:
        raise ValueError(
            f"Unknown slab_type '{slab_type}'. Valid options: {', '.join(valid)}"
        )
    if problem_id == "default":
        raise KeyError(
            "cannot build the reserved non-runnable problem id 'default'"
        )

    mod_path = f"slab_construction.slabs.{slab_type}.analysis"
    slab_module = import_module(mod_path)
    # Ensure slab_dir.name used by run-config matches the logical slab type
    # even if the snapshot folder was copied under a temporary name.
    snapshot_dir = Path(snapshot_dir)
    bundles = build_problems_for_slab(
        snapshot_dir,
        slab_module,
        problem_ids=[problem_id],
        slab_type=slab_type,
    )
    return bundles[problem_id]