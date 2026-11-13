"""
HP-slab analysis function for HpDesignBench.

Builds the full slab construction object from a decoded parameter
dictionary, evaluates all structural, construction, acoustic, and
modeling checks, computes the weighted objective function, and returns
a result dictionary compatible with the IOH benchmarking framework.

Framework adapted from: Max Dombrowski

Scope of checks and penalized penalty function formulation adapted
from Jamila Loutfi :cite:`loutfi_2023`

Author: Elliot Melcer
"""
from __future__ import annotations

import math

from structuralcodes.materials.concrete import create_concrete

from core.analysis_core.checks.acoustic_checks import AirborneSoundInsulationCheck, ImpactSoundInsulationCheck
from core.analysis_core.checks.construction_checks import MidlineConcreteCoverCheck, ReinforcementSpacingCheck, \
    MinimumHPShellThicknessCheck
from core.analysis_core.checks.modeling_checks import NtDyCombinationCheck, BeamTheoryHgesLRatioCheck, \
    BeamTheoryBLRatioCheck
from core.analysis_core.checks.structural_checks import UltimateMomentCheckEC2004DE, \
    DeflectionLimitByDeflectionCheckEC2004DE, \
    FailureAnnouncementByDeflectionCheckEC2004DE
from core.analysis_core.run_config import (
    AnalysisRunConfig,
    analysis_run_config_from_params,
    apply_analysis_run_config,
)
from core.analysis_core.section_methods import InvalidSectionForMKError, prepare_deflection_mk_cache, \
    prepare_reference_analysis_cache_EC
from core.analysis_core.statics.constants import SystemType, MomentType
from core.analysis_core.statics.loads import LoadsEC
from core.analysis_core.material_methods import (
    get_cfrp_reinforcement_from_registry,
    get_floor_material_from_registry,
    get_material_properties,
)
from core.ioh_core.io_util import _req_param
from core.unit_core import mm2_to_m2, mm3_to_m3, kN_to_N
from slab_construction.slab_construction import SlabConstruction
from slab_construction.floor import FloorLayer, Floor, InsulationMaterial, ScreedMaterial
from slab_construction.slabs.hp_slab.hp_model.hp_geometry import HPGeometry
from slab_construction.slabs.hp_slab.hp_model.hp_shell import HPShell
from slab_construction.slabs.hp_slab.hp_model.hp_slab import HPSlab


def failed_evaluation_violation(invalid_utilization: float) -> float:
    """
    Raw violation used for fail-closed evaluation failures.

    Derived from the shared invalid-utilization sentinel
    ``opt_invalid_utilization`` (via ``AnalysisRunConfig``):

    ``max(0, invalid_utilization − 1)``.

    This is not a second configuration value; it is always computed from
    the utilization sentinel.
    """
    return max(0.0, float(invalid_utilization) - 1.0)


def _analysis_error_result(
    params: dict,
    *,
    stage: str,
    material_id: str,
    exc: BaseException,
) -> dict:
    """Uniform infeasible result when material creation / lookup fails."""
    return {
        "y": float("inf"),
        "y_p": float("inf"),
        "constraint_values": {},
        "violations_": {},
        "penalties_": {},
        "penalty_sum": 0.0,
        "penalty_factor": 1.0,
        "params": dict(params),
        "analysis_ok": False,
        "constraint_feasible": False,
        "feasible": False,
        "analysis_error": {
            "stage": stage,
            "material_id": material_id,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        },
        "remapped_utilizations": [],
        "cost_components_missing": [],
    }


def unexpected_evaluation_failure_result(
    params: dict | None,
    *,
    exc: BaseException,
    active_constraint_names: list[str] | None = None,
    invalid_utilization: float,
    traceback_text: str | None = None,
) -> dict:
    """
    Structured fail-closed result for an unexpected evaluation exception.

    Used by :class:`~core.ioh_core.cache_eval.EvalContext` so objective and
    constraint callbacks share one cached failure record. ``y_p`` is
    ``inf`` because no trustworthy base objective exists; active
    constraints receive the configured invalid utilization / violation.
    """
    import traceback as _traceback

    names = list(active_constraint_names or [])
    invalid = float(invalid_utilization)
    violation = failed_evaluation_violation(invalid)
    constraint_values = {name: invalid for name in names}
    violations_ = {name: violation for name in names}
    return {
        "y": float("inf"),
        "y_p": float("inf"),
        "constraint_values": constraint_values,
        "violations_": violations_,
        "penalties_": {},
        "penalty_sum": 0.0,
        "penalty_factor": 1.0,
        "params": dict(params) if params is not None else {},
        "analysis_ok": False,
        "constraint_feasible": False,
        "feasible": False,
        "analysis_error": {
            "stage": "unexpected_exception",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback_text
            if traceback_text is not None
            else "".join(
                _traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        },
        "remapped_utilizations": [],
        "cost_components_missing": [],
    }


def _apply_active_penalties(
    *,
    y: float,
    constraint_values: dict,
    constraints: dict,
    invalid_utilization: float,
    opt_penalty_alpha: float,
    opt_penalty_beta: float,
) -> tuple[dict, dict, dict, float, float, float, list[str], bool, dict | None]:
    """
    Sanitize active utilizations, compute penalties, and derive status flags.

    Non-finite active utilizations are remapped to ``invalid_utilization``
    so the penalized objective stays finite and rankable when ``y`` is
    finite. Remapped names are recorded and reported via ``analysis_error``.

    Returns
    -------
    constraint_values, violations_, penalties_, penalty_sum, penalty_factor,
    y_p, remapped_utilizations, analysis_ok, analysis_error
    """
    remapped: list[str] = []
    sanitized = dict(constraint_values)
    violations_: dict[str, float] = {}
    penalties_: dict[str, float] = {}
    penalty_sum = 0.0
    invalid = float(invalid_utilization)

    for name, eta_raw in list(sanitized.items()):
        meta = constraints.get(name)
        if meta is None or not meta.get("active", True):
            continue
        try:
            eta = float(eta_raw)
        except (TypeError, ValueError):
            eta = float("nan")
        if not math.isfinite(eta):
            sanitized[name] = invalid
            eta = invalid
            remapped.append(name)
        violation = max(0.0, eta - 1.0)
        # Guard against any residual non-finite arithmetic.
        if not math.isfinite(violation):
            sanitized[name] = invalid
            eta = invalid
            violation = max(0.0, invalid - 1.0)
            if name not in remapped:
                remapped.append(name)
        penalty = opt_penalty_alpha * (violation ** opt_penalty_beta)
        if not math.isfinite(penalty):
            violation = max(0.0, invalid - 1.0)
            penalty = opt_penalty_alpha * (violation ** opt_penalty_beta)
            sanitized[name] = invalid
            if name not in remapped:
                remapped.append(name)
        violations_[name] = violation
        penalties_[name] = float(penalty)
        penalty_sum += float(penalty)

    penalty_factor = 1.0 + penalty_sum
    y_p = float(y) * penalty_factor if math.isfinite(float(y)) else float(y)

    analysis_error = None
    analysis_ok = True
    if remapped:
        analysis_ok = False
        analysis_error = {
            "stage": "nonfinite_utilization",
            "constraints": list(remapped),
            "message": (
                "non-finite utilization remapped to opt_invalid_utilization"
            ),
            "invalid_utilization": invalid,
        }

    return (
        sanitized,
        violations_,
        penalties_,
        penalty_sum,
        penalty_factor,
        y_p,
        remapped,
        analysis_ok,
        analysis_error,
    )


def _constraint_feasible(violations_: dict[str, float]) -> bool:
    """True iff every recorded active violation is finite and exactly 0."""
    if not violations_:
        return True
    for v in violations_.values():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(fv) or fv > 0.0:
            return False
    return True


def _capture_build_failure_artifact(
    holder: list | None,
    *,
    key: tuple,
    snapshot_hash: str,
    n: float,
    failure_result: dict,
) -> None:
    """
    Record a structural build-failure artifact when a holder is provided.
    """
    if holder is None:
        return
    from core.ioh_core.structural_artifacts.artifact import (
        capture_structural_artifact,
    )
    from slab_construction.slab_construction import AnalysisCache

    holder.clear()
    holder.append(
        capture_structural_artifact(
            AnalysisCache(),
            key=key,
            snapshot_hash=str(snapshot_hash),
            n=float(n),
            mk_cleared=True,
            build_failure_result=failure_result,
            count_build=True,
        )
    )


def analysis(
    params: dict,
    constraints: dict,
    materials: dict,
    debug: bool = False,
    *,
    run_config: AnalysisRunConfig | None = None,
    structural_artifact: object | None = None,
    structural_artifact_holder: list | None = None,
    structural_capture_key: tuple | None = None,
    structural_snapshot_hash: str = "",
) -> dict:
    """
    Run the full HP-slab analysis for a single candidate design.

    Builds the HP-slab construction object from decoded parameters,
    evaluates all structural (A), serviceability (B), construction (C),
    acoustic (D), and modeling (Z) checks, computes the weighted
    objective function, and applies the additive violation-power
    square-root penalty. Finally, it returns all results in a format
    compatible with the :class:`EvalContext` cache.

    All parameters in params : dict must be defined inside the
    .csv-files within the hp_slab package

    Optional brute-force reuse hooks (keyword-only):

    * ``structural_artifact`` — when provided, skip section / M–κ
      preparation and hydrate the analysis cache from an immutable
      numeric artifact built earlier for the same structural key.
    * ``structural_artifact_holder`` — when provided as a list, append
      a captured :class:`~core.ioh_core.structural_artifacts.artifact.StructuralArtifact`
      after preparation (or a build-failure artifact on expected
      material / construction failures).
    * ``structural_capture_key`` / ``structural_snapshot_hash`` — identity
      used when capturing into ``structural_artifact_holder``.

    Parameters
    ----------
    params : dict
        Decoded parameter dictionary as returned by ``decode(x_idx)``.
        Required keys (extracted via :func:`_req_param`):

        *Geometry:* ``geom_span_mm``, ``geom_b_mm``, ``geom_h_ges_mm``,
        ``geom_t_mm``, ``geom_nt``, ``geom_dy_mm``,
        ``geom_hx_hges_ratio``,
        ``geom_t_infill_mm``, ``geom_t_insulation_mm``,
        ``geom_t_screed_mm``.

        *Materials:* ``mat_conc_id``, ``mat_reinf_id``,
        ``reinf_a_tex_mm2``, ``reinf_kap_t_percent``,
        ``mat_infill_id``, ``mat_insu_id``, ``mat_screed_id``,
        ``mat_reinf_cost_crfp/conc``, ``mat_reinf_gwp_crfp/conc``.
        Concrete structural and environmental intensities are read from
        the materials registry (``C30/37`` for CFRP ratio denominators).

        *Loads:* ``loads_N_kN`` (axial force input [kN]; converted once
        to N for section/cache APIs; positive = tension, negative =
        compression), ``loads_category``.

        *Deflection:* ``defl_max_defl_limit`` (B1a L/limit),
        ``defl_max_announce_failure`` (B2a L/min research proxy, typically 100).

        *Acoustics:* ``insu_mod_damp``, ``insu_rw_req_db``,
        ``insu_lnw_max_db``.

        *Beam theory:* ``f_beam_theory``.

        *Optimization:* ``opt_penalty_alpha``, ``opt_penalty_beta``,
        ``opt_invalid_utilization``.

        *Objective weights:* ``weight_omega_1_gwp``, ``weight_omega_2_cost``.

    constraints : dict
        Active constraint configuration dict for this problem. Each entry
        maps a constraint name to its config dict (must contain
        ``"active"``).
    materials : dict
        Materials registry dict as returned by
        :func:`load_materials_registry`.
    debug : bool, optional
        If ``True``, prints detailed constraint values, violations, and
        penalty contributions to console. Default is ``False``.
    run_config : AnalysisRunConfig or None, optional
        Frozen per-run/problem settings (design code, invalid-utilization
        sentinel, slab type). When omitted, a config is derived from
        ``params`` and defaults. Design code is applied at most once per
        distinct value in the process. The invalid-utilization sentinel is
        taken from ``run_config`` / ``opt_invalid_utilization`` (CSV), not
        from a module constant.

    Returns
    -------
    dict
        Result dictionary with the following keys:

        - ``"y"`` — unpenalized objective value (weighted cost + GWP).
        - ``"y_p"`` — penalized objective ``y · (1 + Σ penalty_i)``.
        - ``"constraint_values"`` — dict of all computed utilization ratios [-];
          non-finite active values are remapped to ``opt_invalid_utilization``.
        - ``"violations_"`` — active ``max(0, η−1)`` values [-].
        - ``"penalties_"`` — active ``α · violation^β`` contributions [-].
        - ``"penalty_sum"`` / ``"penalty_factor"`` — Σ penalties and ``1 + Σ``.
        - ``"params"`` — copy of the input ``params`` dict.
        - ``"analysis_ok"`` — ``True`` when analysis completed without material
          failure or non-finite-utilization remapping.
        - ``"constraint_feasible"`` — ``True`` iff every active utilization is
          finite and at most ``1.0``.
        - ``"feasible"`` — ``analysis_ok and constraint_feasible``.
        - ``"analysis_error"`` — ``None`` on clean success; structured error
          info for material failure, remapped non-finite utilizations, or
          unexpected evaluation exceptions.
        - ``"remapped_utilizations"`` — active constraint names whose
          utilizations were non-finite and remapped to the invalid sentinel.
        - ``"cost_components_missing"`` — finish material keys omitted
          from ``y_cost`` because registry ``cost`` is empty.

    Notes
    -----
    **Objective function:**

    .. math::

        y = \\Omega_2 \\cdot y_{\\text{cost}} + \\Omega_1 \\cdot y_{\\text{gwp}}

    where :math:`y_{\\text{cost}}` and :math:`y_{\\text{gwp}}` are the
    material cost and global warming potential of concrete, CFRP, and
    floor finishes (infill, insulation, screed). Concrete intensities
    come from the materials registry (per m³). CFRP intensities are
    ``C30/37_registry · mat_reinf_*_crfp/conc`` (per kg), multiplied by
    reinforcement mass ``V·ρ``. Finish GWP uses registry per-kg values
    times layer mass; finish cost is included only when registry ``cost``
    is set.

    **Penalty formulation:**

    For each active utilization :math:`\\eta_i`:

    .. math::

        y_p = y \\cdot \\Bigl(1 + \\sum_i \\alpha(\\max(0,\\eta_i-1))^{\\beta}\\Bigr)

    with ``α = opt_penalty_alpha``, ``β = opt_penalty_beta``. Inactive
    constraints are excluded. Default catalogue values use ``α = 1`` and
    ``β = 1/3`` so small violations are penalized more strongly than the
    former nested square-root form with ``β = 1``.

    **SLS deflection checks:**

    B.1a limits maximum quasi-permanent deflection (typically L/250).
    B.2a is a research proxy requiring a minimum fundamental-load
    deflection magnitude of L / ``defl_max_announce_failure`` (typically
    L/100). Long-term deflection is out of scope.

    **Acoustic buffers:**

    A safety buffer of 2 dB is applied to the airborne sound check (D.1)
    and 3 dB to the impact sound check (D.2).
    """

    # ======================================================
    # EXTRACT PARAMETERS
    # ======================================================

    # Geometry
    span_mm     = _req_param(params, "geom_span_mm")
    B_mm        = _req_param(params, "geom_b_mm")
    h_ges_mm    = _req_param(params, "geom_h_ges_mm")
    t_mm        = _req_param(params, "geom_t_mm")
    nt          = int(_req_param(params, "geom_nt"))
    dy_mm       = _req_param(params, "geom_dy_mm")
    hx_hges_ratio = _req_param(params, "geom_hx_hges_ratio")

    # Concrete
    conc_id = params.get("mat_conc_id", "")

    # Reinforcement
    reinf_id = params.get("mat_reinf_id", "")

    reinf_area_mm2 = _req_param(params, "reinf_a_tex_mm2")
    prestress_pct = _req_param(params, "reinf_kap_t_percent")

    # Floor Materials
    infill_id = params.get("mat_infill_id", "")
    insu_id = params.get("mat_insu_id", "")
    screed_id = params.get("mat_screed_id", "")

    infill_t_mm = _req_param(params, "geom_t_infill_mm")
    insu_t_mm = _req_param(params, "geom_t_insulation_mm")
    screed_t_mm = _req_param(params, "geom_t_screed_mm")

    # Loads — CSV value is [kN]; section/cache APIs expect [N]
    # (positive = tension, negative = compression)
    n = kN_to_N(_req_param(params, "loads_N_kN"))
    load_category = params.get("loads_category", "")

    # Deformations — B1a uses L/defl_max_defl_limit; B2a uses L/defl_max_announce_failure
    defl_limit_factor_w_max = _req_param(params, "defl_max_defl_limit")
    defl_min_factor_announce_failure = _req_param(params, "defl_max_announce_failure")

    # Acoustic Parameters
    modular_attenuation = _req_param(params, "insu_mod_damp")
    R_w_req_db = _req_param(params, "insu_rw_req_db")
    L_nw_max_db = _req_param(params, "insu_lnw_max_db")

    # Beam Theory
    f_beam_theory = _req_param(params, "f_beam_theory")

    # Optimization controls
    opt_penalty_alpha = float(_req_param(params, "opt_penalty_alpha"))
    opt_penalty_beta = float(_req_param(params, "opt_penalty_beta"))
    if run_config is None:
        run_config = analysis_run_config_from_params(params, slab_type="hp_slab")
    apply_analysis_run_config(run_config)
    invalid_util = float(run_config.invalid_utilization)

    # ======================================================================================================================
    # CREATE MATERIALS
    # ======================================================================================================================

    # Concrete (structural props from registry; EC2 derives the rest)
    try:
        conc_props = get_material_properties(conc_id, materials)
        if conc_props.get("type") != "concrete":
            raise ValueError(
                f"Material '{conc_id}' has type '{conc_props.get('type')}', "
                f"expected 'concrete'."
            )
        conc_fck_MPa = conc_props.get("f_ck")
        alpha_cc = conc_props.get("alpha_cc")
        gamma_c = conc_props.get("gamma_c")
        if conc_fck_MPa is None or alpha_cc is None or gamma_c is None:
            raise ValueError(
                f"Concrete '{conc_id}' missing f_ck/alpha_cc/gamma_c in materials registry."
            )
        concrete_uls = create_concrete(
            fck=conc_fck_MPa,
            constitutive_law="parabolarectangle",
            alpha_cc=alpha_cc,
            gamma_c=gamma_c,
            name=f"{conc_id} ULS",
        )
    except (KeyError, ValueError) as e:
        print(f"Error creating concrete: {e}")
        failure = _analysis_error_result(
            params, stage="concrete", material_id=str(conc_id), exc=e
        )
        _capture_build_failure_artifact(
            structural_artifact_holder,
            key=structural_capture_key or (),
            snapshot_hash=structural_snapshot_hash,
            n=kN_to_N(_req_param(params, "loads_N_kN")),
            failure_result=failure,
        )
        return failure

    # Reinforcement
    try:
        reinforcement = get_cfrp_reinforcement_from_registry(
            mat_id=reinf_id,
            materials=materials,
            prestress_percent=prestress_pct,
            limit_state="ULS",
        )
    except (KeyError, ValueError) as e:
        print(f"Error creating reinforcement: {e}")
        failure = _analysis_error_result(
            params, stage="reinforcement", material_id=str(reinf_id), exc=e
        )
        _capture_build_failure_artifact(
            structural_artifact_holder,
            key=structural_capture_key or (),
            snapshot_hash=structural_snapshot_hash,
            n=kN_to_N(_req_param(params, "loads_N_kN")),
            failure_result=failure,
        )
        return failure

    # Floor materials
    try:
        infill_material = get_floor_material_from_registry(infill_id, materials)
        insulation_material = get_floor_material_from_registry(insu_id, materials)
        screed_material = get_floor_material_from_registry(screed_id, materials)
    except (KeyError, ValueError) as e:
        print(f"Error creating floor materials: {e}")
        failure = _analysis_error_result(
            params,
            stage="floor_materials",
            material_id=f"{infill_id}/{insu_id}/{screed_id}",
            exc=e,
        )
        _capture_build_failure_artifact(
            structural_artifact_holder,
            key=structural_capture_key or (),
            snapshot_hash=structural_snapshot_hash,
            n=kN_to_N(_req_param(params, "loads_N_kN")),
            failure_result=failure,
        )
        return failure

    # ======================================================================================================================
    # CREATE FLOOR
    # ======================================================================================================================

    infill_layer            = FloorLayer(infill_material, infill_t_mm)
    sound_insulation_layer  = FloorLayer(insulation_material, insu_t_mm)
    screed_layer            = FloorLayer(screed_material, screed_t_mm)

    floor = Floor([infill_layer, sound_insulation_layer, screed_layer])

    # ======================================================================================================================
    # CREATE HP SLAB
    # ======================================================================================================================

    # Calculate Hx and Hy
    Hx_mm = hx_hges_ratio       * h_ges_mm
    Hy_mm = (1-hx_hges_ratio)   * h_ges_mm

    # HP Geometry
    hp_geom = HPGeometry(
        B=B_mm,
        L=span_mm,
        Hx=Hx_mm,
        Hy=Hy_mm,
        t=t_mm,
        dy=dy_mm,
        nt=nt
    )

    # HP-Shell
    hp_shell = HPShell(hp_geom, concrete_uls, reinforcement, reinf_area = reinf_area_mm2)

    # HP-Slab
    hp_slab = HPSlab(hp_shell, infill_material)

    # HP-Slab Construction
    slab_construction = SlabConstruction(hp_slab, floor)

    # ======================================================================================================================
    # ANALYSIS CACHE — reference sections/ULS M_u + coordinated simplified M-κ
    # (M-κ also stores SLS M_cr for deflection ζ)
    # ======================================================================================================================
    cache = slab_construction.analysis_cache
    mk_cleared = False
    if structural_artifact is not None:
        from core.ioh_core.structural_artifacts.artifact import (
            apply_structural_artifact,
        )

        apply_structural_artifact(cache, structural_artifact)
    else:
        from core.ioh_core.structural_artifacts.artifact import (
            _bump_structural_build_count,
        )

        _bump_structural_build_count()
        prepare_reference_analysis_cache_EC(slab_construction, n=n)

        try:
            prepare_deflection_mk_cache(slab_construction, n=n)
        except InvalidSectionForMKError:
            # Infeasible sections are handled by the individual checks.
            cache.mk_curves.clear()
            cache.mk_metadata.clear()
            mk_cleared = True

        if structural_artifact_holder is not None:
            from core.ioh_core.structural_artifacts.artifact import (
                capture_structural_artifact,
                make_structural_key,
            )

            if structural_capture_key is None:
                if run_config is None:
                    run_config = analysis_run_config_from_params(
                        params, slab_type="hp_slab"
                    )
                capture_key = make_structural_key(
                    params,
                    finish_names=(),
                    snapshot_hash=structural_snapshot_hash,
                    n=float(n),
                    run_config=run_config,
                )
            else:
                capture_key = structural_capture_key
            structural_artifact_holder.clear()
            structural_artifact_holder.append(
                capture_structural_artifact(
                    cache,
                    key=capture_key,
                    snapshot_hash=structural_snapshot_hash,
                    n=float(n),
                    mk_cleared=mk_cleared,
                    count_build=False,
                )
            )

    # ======================================================================================================================
    # LIVE LOADS
    # ======================================================================================================================

    live_loads = LoadsEC.from_categories_EC0_NA_DE(load_category)

    # ======================================================================================================================
    # COMPUTE CONSTRAINTS (A) - ULS
    # ======================================================================================================================

    m_u_A_util = UltimateMomentCheckEC2004DE.calculate_utilization(
        slab_construction = slab_construction,
        loads = live_loads,
        system = SystemType.SIMPLE_BEAM,
        moment = MomentType.MAX_POS_MOMENT,
        n = n,
        invalid_utilization=invalid_util,
    )

    # ======================================================================================================================
    # COMPUTE CONSTRAINTS (B) - SLS
    # ======================================================================================================================

    # B.1a Limiting Deflection by Checking the Maximum Deflection against Limit Factor

    w_max_B1a_util = DeflectionLimitByDeflectionCheckEC2004DE.calculate_utilization(
        slab_construction = slab_construction,
        loads = live_loads,
        system = SystemType.SIMPLE_BEAM,
        limit_factor=defl_limit_factor_w_max,
        n=n,
        invalid_utilization=invalid_util,
    )

    # B.2a Minimum Deflection under Fundamental Combination (research proxy L/min_factor)

    fa_B2a_util = FailureAnnouncementByDeflectionCheckEC2004DE.calculate_utilization(
        slab_construction = slab_construction,
        loads = live_loads,
        system=SystemType.SIMPLE_BEAM,
        min_factor = defl_min_factor_announce_failure,
        n=n,
        invalid_utilization=invalid_util,
    )

    # ======================================================================================================================
    # COMPUTE CONSTRAINTS (C) - CONSTRUCTION
    # ======================================================================================================================

    # C.1. Sufficient Concrete Cover from the outermost Reinforcement to the Edge along the HP-Shell Midline

    cc_C1_util = MidlineConcreteCoverCheck.calculate_utilization(
        slab_construction=slab_construction,
        invalid_utilization=invalid_util,
    )

    # C.2. Check for Sufficient Clear Spacing between Reinforcement along the HP-Shell Midline

    s_C2_util = ReinforcementSpacingCheck.calculate_utilization(
        slab_construction = slab_construction,
        invalid_utilization=invalid_util,
    )

    # C.3 Check for Minimum Shell Thickness

    tmin_C3_util = MinimumHPShellThicknessCheck.calculate_utilization(
        slab_construction = slab_construction,
    )

    # ======================================================================================================================
    # COMPUTE CONSTRAINTS (D) - INSULATION
    # ======================================================================================================================

    # D.1 Airborne Sound Insulation Check

    asi_D1_util = AirborneSoundInsulationCheck.calculate_utilization(
        slab_construction = slab_construction,
        limit_dB = R_w_req_db,
        mod_att = modular_attenuation,
        buffer_dB = 2.0
    )

    isi_D2_util = ImpactSoundInsulationCheck.calculate_utilization(
        slab_construction = slab_construction,
        limit_dB = L_nw_max_db,
        mod_att = modular_attenuation,
        buffer_dB = 3.0
    )

    # ======================================================================================================================
    # COMPUTE CONSTRAINTS (Z) - MODELING
    # ======================================================================================================================

    # Z.1. Combination of nt and dy

    ntdy_Z1_util = NtDyCombinationCheck.calculate_utilization(
        slab_construction = slab_construction,
        invalid_utilization=invalid_util,
    )

    # Z.2. Beam Theory H_ges / L - Ratio

    beam_hL_Z2_util = BeamTheoryHgesLRatioCheck.calculate_utilization(
        slab_construction = slab_construction,
        f_b=f_beam_theory,
        invalid_utilization=invalid_util,
    )

    # Z.3. Beam Theory B / L - Ratio

    beam_BL_Z3_util = BeamTheoryBLRatioCheck.calculate_utilization(
        slab_construction = slab_construction,
        f_b=f_beam_theory,
        invalid_utilization=invalid_util,
    )

    # ======================================================================================================================
    # COMPUTE WEIGHTED OBJECTIVE VALUE
    # ======================================================================================================================

    # Reference C30/37 from registry (denominators for CFRP ratio factors)
    try:
        c30_props = get_material_properties("C30/37", materials)
        cost_conc_c30_ref = c30_props.get("cost_m3")  # €/m³
        gwp_conc_c30_ref = c30_props.get("gwp_m3")  # kgCO2e/m³
        if cost_conc_c30_ref is None or gwp_conc_c30_ref is None:
            raise ValueError("C30/37 missing cost_m3/gwp_m3 in materials registry.")
    except (KeyError, ValueError) as e:
        print(f"Error reading C30/37 reference intensities: {e}")
        return _analysis_error_result(
            params, stage="c30_reference", material_id="C30/37", exc=e
        )

    # CFRP ratios vs C30/37: (intensity_per_kg_CFRP) / (intensity_per_m3_C30)
    cost_reinf_factor = _req_param(params, "mat_reinf_cost_crfp/conc")
    gwp_reinf_factor = _req_param(params, "mat_reinf_gwp_crfp/conc")

    # Material quantities
    concrete_m3 = mm3_to_m3(hp_shell.net_concrete_volume())  # m³
    reinforcement_m3 = mm3_to_m3(hp_shell.total_reinforcement_volume())  # m³
    reinforcement_kg = reinforcement_m3 * float(reinforcement.density)  # kg

    # Concrete intensities from registry (per m³)
    cost_concrete_eur_m3 = conc_props.get("cost_m3")
    gwp_concrete_CO2eq_m3 = conc_props.get("gwp_m3")
    if cost_concrete_eur_m3 is None or gwp_concrete_CO2eq_m3 is None:
        exc = ValueError(
            f"Concrete '{conc_id}' missing cost_m3/gwp_m3 in materials registry."
        )
        print(f"Error: {exc}")
        return _analysis_error_result(
            params, stage="concrete_intensities", material_id=conc_id, exc=exc
        )

    # CFRP intensities from ratios are per kg; convert quantity via density
    cost_reinforcement_eur_kg = cost_conc_c30_ref * cost_reinf_factor  # €/kg
    gwp_reinforcement_CO2eq_kg = gwp_conc_c30_ref * gwp_reinf_factor  # kgCO2e/kg

    # Finish masses [kg] = area density [kg/m²] × plan area [m²]
    area_m2 = mm2_to_m2(hp_slab.L * hp_slab.B)
    insulation_layer = floor.get_layer_by_type(InsulationMaterial)
    screed_layer = floor.get_layer_by_type(ScreedMaterial)
    infill_kg = float(slab_construction.infill_area_density_kg_m2()) * area_m2
    insulation_kg = float(insulation_layer.area_density_kg_m2()) * area_m2
    screed_kg = float(screed_layer.area_density_kg_m2()) * area_m2

    try:
        infill_props = get_material_properties(infill_id, materials)
        insulation_props = get_material_properties(insu_id, materials)
        screed_props = get_material_properties(screed_id, materials)
        gwp_infill = infill_props.get("gwp")
        gwp_insulation = insulation_props.get("gwp")
        gwp_screed = screed_props.get("gwp")
        if gwp_infill is None or gwp_insulation is None or gwp_screed is None:
            raise ValueError(
                "Finish materials require gwp [kgCO2e/kg] in the materials registry "
                f"(infill={infill_id!r}, insulation={insu_id!r}, screed={screed_id!r})."
            )
    except (KeyError, ValueError) as e:
        print(f"Error reading finish intensities: {e}")
        return _analysis_error_result(
            params,
            stage="finish_intensities",
            material_id=f"{infill_id}|{insu_id}|{screed_id}",
            exc=e,
        )

    # Unpenalized objective function
    y_cost = (
        cost_concrete_eur_m3 * concrete_m3
        + cost_reinforcement_eur_kg * reinforcement_kg
    )
    y_gwp = (
        gwp_concrete_CO2eq_m3 * concrete_m3
        + gwp_reinforcement_CO2eq_kg * reinforcement_kg
        + gwp_infill * infill_kg
        + gwp_insulation * insulation_kg
        + gwp_screed * screed_kg
    )

    # Finish cost only when registry cost [€/kg] is available
    cost_components_missing = []
    for name, props, mass_kg in (
        ("infill", infill_props, infill_kg),
        ("insulation", insulation_props, insulation_kg),
        ("screed", screed_props, screed_kg),
    ):
        cost_per_kg = props.get("cost")
        if cost_per_kg is None:
            cost_components_missing.append(name)
        else:
            y_cost += cost_per_kg * mass_kg

    weight_cost = _req_param(params, "weight_omega_2_cost")
    weight_gwp = _req_param(params, "weight_omega_1_gwp")

    y = y_cost * weight_cost + y_gwp * weight_gwp

    # ======================================================================================================================
    # COMPUTE OBJECTIVE FUNCTION
    # Author: Max Dombrowski
    # ======================================================================================================================
    # Collect constraint values

    # A
    constraint_values = {"A_bending_capacity": m_u_A_util}

    # B — simply supported SLS pair (B1a max deflection, B2a min L/100 proxy)
    constraint_values["B1a_deflection_by_wmax_capacity"] = w_max_B1a_util
    constraint_values["B2a_failure_announcement_by_wmin_capacity"] = fa_B2a_util

    # C
    constraint_values["C1_concrete_cover_capacity"] = cc_C1_util
    constraint_values["C2_clear_spacing_capacity"] = s_C2_util
    constraint_values["C3_shell_thickness_capacity"] = tmin_C3_util

    # D
    constraint_values["D1_airborne_sound_insulation_capacity"] = asi_D1_util
    constraint_values["D2_impact_sound_insulation_capacity"] = isi_D2_util

    # Z
    constraint_values["Z1_nt_dt_combination_capacity"] = ntdy_Z1_util
    constraint_values["Z2_beam_theory_H_L_capacity"] = beam_hL_Z2_util
    constraint_values["Z3_beam_theory_B_L_capacity"] = beam_BL_Z3_util

    # Additive violation^β penalty over active constraints (fail-closed for non-finite η)
    invalid_utilization = invalid_util
    (
        constraint_values,
        violations_,
        penalties_,
        penalty_sum,
        penalty_factor,
        y_p,
        remapped_utilizations,
        analysis_ok,
        analysis_error,
    ) = _apply_active_penalties(
        y=y,
        constraint_values=constraint_values,
        constraints=constraints,
        invalid_utilization=invalid_utilization,
        opt_penalty_alpha=opt_penalty_alpha,
        opt_penalty_beta=opt_penalty_beta,
    )
    constraint_feasible = _constraint_feasible(violations_)
    feasible = bool(analysis_ok and constraint_feasible)

    if debug:
        print(
            "constraint values: {"
            + ", ".join(f"'{k}': {v:.3f}" for k, v in constraint_values.items())
            + "}"
        )
        print(
            "violations_: {"
            + ", ".join(f"'{k}': {v:.3f}" for k, v in violations_.items())
            + "}"
        )
        print(
            "penalties_: {"
            + ", ".join(f"'{k}': {v:.3f}" for k, v in penalties_.items())
            + "}"
        )
        print(f"penalty sum: {penalty_sum:.3f}")
        print(f"penalty factor: {penalty_factor:.3f}")
        print(f"penalized objective function: {y_p:.3f} \n")
        print(
            f"analysis_ok={analysis_ok} constraint_feasible={constraint_feasible} "
            f"feasible={feasible} remapped={remapped_utilizations}"
        )

    # ======================================================================================================================
    # RETURN RESULTS
    # ======================================================================================================================

    return {
        "y": y,
        "y_p": y_p,
        "constraint_values": constraint_values,
        "violations_": violations_,
        "penalties_": penalties_,
        "penalty_sum": penalty_sum,
        "penalty_factor": penalty_factor,
        "params": dict(params),
        "analysis_ok": analysis_ok,
        "constraint_feasible": constraint_feasible,
        "feasible": feasible,
        "analysis_error": analysis_error,
        "remapped_utilizations": remapped_utilizations,
        "cost_components_missing": cost_components_missing,
    }