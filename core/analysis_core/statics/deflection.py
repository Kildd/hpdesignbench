"""
Deflection calculation for SlabConstructions using Simpson's rule and the
virtual work method.

Curvature distributions are derived from position-dependent M-κ curves
computed at support and midspan. Raw equal-length reference curves are
parabolically interpolated in space; force-control (terminal softening cut
and dip-fill) is applied on the local curve before κ(M) lookup.

Author: Elliot Melcer
"""

import numpy as np
from typing import Union

from core import normalize_input
from slab_construction.slab_construction import SectionResultKey, SlabConstruction
from core.analysis_core.statics.loads import Loads
from core.analysis_core.section_methods import (
    _force_controlled_mk_arrays,
    lookup_kappa_force_controlled,
    make_mk_curve_key_sls_EC,
)
from core.analysis_core.statics.constants import SystemType
from core.analysis_core.statics.internal_forces import InternalForces
from core.unit_core import mm_to_m


class DeflectionCalculator:
    """
    Deflection calculator using Simpson's rule and the virtual work method.

    Curvature κ(x) is obtained by parabolically interpolating the raw M-κ
    diagrams computed at the support (x = 0) and midspan (x = 0.5), applying
    force-control on the local curve, then looking up κ at the applied moment
    at each Simpson integration point.
    By symmetry, only the half-span [0, 0.5] is integrated and the result
    is doubled.

    Position convention: x is normalized (0 at first support,
    1 at second support, etc.).
    """

    @staticmethod
    def calculate_deflection_mm_EC(
            slab_construction: SlabConstruction,
            loads: Loads,
            system: SystemType = SystemType.SIMPLE_BEAM,
            combination: str = "QUASI_PERMANENT",
            n_intervals: int = 20,
            N_axial_N: float = 0.0,
            constitutive_law: str = "TENSTIFF_PARABOLIC",
            load_history_method: str = "NONE",
            m_k_simplification = False,
            m_k_simplification_softening=None,
            m_k_num_points: int = 40,
            debug: bool = False,
            extended_debug: bool = False
    ) -> float:
        """
        Calculate the maximum midspan deflection per EN 1992-1-1 :cite:`ec2`.

        Dispatches to :meth:`_direct_deflection_method` (``load_history_method="NONE"``)
        or :meth:`_factor_deflection_method` (``load_history_method="FACTOR_EC"``).
        Currently only :attr:`SystemType.SIMPLE_BEAM` is supported.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        loads : Loads
            Applied loads object.
        system : SystemType, optional
            Structural system type. Currently only
            :attr:`SystemType.SIMPLE_BEAM` is implemented.
            Default is :attr:`SystemType.SIMPLE_BEAM`.
        combination : str, optional
            EN 1990 load combination used for the direct method. Ignored
            when ``load_history_method="FACTOR_EC"``. One of
            ``"FUNDAMENTAL"``, ``"QUASI_PERMANENT"``, ``"FREQUENT"``,
            ``"RARE"``. Default is ``"QUASI_PERMANENT"``.
        n_intervals : int, optional
            Number of Simpson integration intervals (must be even).
            Default is ``40``.
        N_axial_N : float, optional
            Applied normal force [N] (positive = tension). Default is ``0.0``.
        constitutive_law : str, optional
            Concrete constitutive law keyword identifying the cached M-κ
            curves prepared by :func:`prepare_deflection_mk_cache` /
            :func:`prepare_mk_cache_sls_EC`. Used only for the direct method;
            see ``create_sls_concrete_EC`` for valid keywords.
            Default is ``"TENSTIFF_PARABOLIC"``.
        load_history_method : str, optional
            Load history strategy:

            - ``"NONE"`` — no load history; direct M-κ integration under
              ``combination``.
            - ``"FACTOR_EC"`` — EN 1992-1-1 cl. 7.4.3, Eq. (7.18): weighted
              interpolation between fully cracked and fully uncracked κ using
              the distribution coefficient ζ.

            Default is ``"NONE"``.
        m_k_simplification : bool or int or float or sequence of floats, optional
            Identifies the cached simplified M-κ curves via
            :func:`make_mk_curve_key_sls_EC` / ``analysis_cache.require_mk_curve``.
            Fraction values may lie in ``(0, 1)`` (between ``kappa_cr`` and
            ``kappa_eoc``) or ``(1, 2)`` (between ``kappa_eoc`` and
            ``kappa_u``). If base cr–EOC extras are too coarse, softening may
            be missed and post-crack curvatures after force control can be
            overestimated. Default is ``False``.
        m_k_simplification_softening : int or float or sequence of floats or None, optional
            Extra M-κ fractions for softening densification. Same fraction
            encoding as ``m_k_simplification``. The corresponding curves must
            be prepared in :attr:`SlabConstruction.analysis_cache` before
            B1a/B2a.
            ``None`` means no softening extras. Default is ``None``.
        m_k_num_points : int, optional
            Number of curvature points for the full M-κ diagram (used when
            ``m_k_simplification=False``). Default is ``40``.
        debug : bool, optional
            If ``True``, prints intermediate values (cracking moments, ζ
            arrays) to console. Default is ``False``.
        extended_debug : bool, optional
            If ``True``, prints full Simpson and κ-interpolation tables in
            addition to the standard debug output. Default is ``False``.

        Returns
        -------
        float
            Maximum midspan deflection [mm].
            Positive = sagging (downward), negative = hogging (upward).

        Raises
        ------
        NotImplementedError
            If ``system`` is not :attr:`SystemType.SIMPLE_BEAM`.
        ValueError
            If ``n_intervals`` is odd, if ``m_k_num_points`` is less than 2,
            or if ``combination`` or ``load_history_method`` is not a recognized
            keyword.
        """
        # --------------------------
        # Normalize Input
        # --------------------------

        combination = normalize_input(combination)
        load_history_method = normalize_input(load_history_method)
        constitutive_law = normalize_input(constitutive_law)

        # --------------------------
        # Validate inputs
        # --------------------------
        if system != SystemType.SIMPLE_BEAM:
            raise NotImplementedError(
                "Deflection calculation currently only implemented for SIMPLE_BEAM"
            )
        if n_intervals % 2 != 0:
            raise ValueError("n_intervals must be even for Simpson's rule")
        if m_k_num_points < 2:
            raise ValueError("m_k_num_points must be >= 2")
        if combination not in ("FUNDAMENTAL", "QUASI_PERMANENT", "FREQUENT", "RARE"):
            raise ValueError("combination must be 'FUNDAMENTAL', 'QUASI_PERMANENT', 'FREQUENT' or 'RARE'")

        # --------------------------
        # Shared Input kwargs
        # --------------------------
        deflection_kwargs = dict(
            slab_construction=slab_construction,
            loads=loads,
            system=system,
            n_intervals=n_intervals,
            N_axial_N=N_axial_N,
            m_k_simplification=m_k_simplification,
            m_k_simplification_softening=m_k_simplification_softening,
            m_k_num_points=m_k_num_points,
            debug=debug,
            extended_debug=extended_debug,
        )

        # --------------------------
        # Consider Load History
        # --------------------------
        if load_history_method == "NONE":
            # Use Direct Calculation Method
            deflection_mm = DeflectionCalculator._direct_deflection_method(
                combination=combination,
                constitutive_law = constitutive_law,
                **deflection_kwargs
            )

        elif load_history_method == "FACTOR_EC":
            deflection_mm = DeflectionCalculator._factor_deflection_method(
                **deflection_kwargs
            )

        else:
            raise ValueError("load_history_method must be 'NONE' or 'FACTOR_EC'")

        return deflection_mm

    @staticmethod
    def _direct_deflection_method(
            slab_construction: SlabConstruction,
            loads: Loads,
            system: SystemType,
            combination: str,
            n_intervals: int,
            N_axial_N: float,
            constitutive_law: str,
            m_k_simplification,
            m_k_simplification_softening,
            m_k_num_points: int,
            debug: bool,
            extended_debug: bool
    ) -> float:
        """
        Compute deflection by direct κ(x) integration for a single load combination.

        Curvatures are looked up from the interpolated M-κ diagram at each
        Simpson point; no load history weighting is applied.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        loads : Loads
            Applied loads object.
        system : SystemType
            Structural system type.
        combination : str
            Normalized EN 1990 load combination name.
        n_intervals : int
            Number of Simpson integration intervals (must be even).
        N_axial_N : float
            Applied normal force [N] (positive = tension).
        constitutive_law : str
            Concrete constitutive law keyword.
        m_k_simplification : bool or int or float or sequence of floats
            Identifies the cached M-κ curves via
            :func:`make_mk_curve_key_sls_EC` /
            ``analysis_cache.require_mk_curve``.
        m_k_simplification_softening : int or float or sequence of floats or None
            Softening extras that identify the required cached M-κ curves.
        m_k_num_points : int
            Number of curvature points for the full M-κ diagram.
        debug : bool
            Unused in this method; passed through for interface consistency.
        extended_debug : bool
            If ``True``, prints κ-interpolation and Simpson tables to console.

        Returns
        -------
        float
            Maximum midspan deflection [mm].
        """
        # --------------------------
        # Setup
        # --------------------------
        span_m = mm_to_m(slab_construction.slab.L)

        # Calculate Kappas along the Beam
        kappa_array, kappa_debug = DeflectionCalculator._get_interpolated_kappa_array(
            slab_construction=slab_construction,
            loads=loads,
            system=system,
            combination=combination,
            n_intervals=n_intervals,
            N_axial_N=N_axial_N,
            constitutive_law = constitutive_law,
            m_k_simplification = m_k_simplification,
            m_k_simplification_softening=m_k_simplification_softening,
            m_k_num_points=m_k_num_points,
        )

        # Real Moments at integration points
        M_applied_array_kNm = kappa_debug["M_applied_kNm"]

        # Integrate using Simpson's Method
        deflection_mm, simpson_debug  = DeflectionCalculator._simpson_integration(
            span_m=span_m,
            n_intervals=n_intervals,
            M_applied_array_kNm = M_applied_array_kNm,
            kappa_array=kappa_array,
        )

        if extended_debug:
            print(f"\n\n[EXTENDED DEBUG] Constitutive Law: {constitutive_law}")
            DeflectionCalculator._print_kappa_interp_debug(kappa_debug)
            DeflectionCalculator._print_simpson_debug(simpson_debug)

        return deflection_mm

    @staticmethod
    def _factor_deflection_method(
            slab_construction: SlabConstruction,
            loads: Loads,
            system: SystemType,
            n_intervals: int,
            N_axial_N: float,
            m_k_simplification,
            m_k_simplification_softening,
            m_k_num_points: int,
            debug: bool,
            extended_debug: bool
    ) -> float:
        """
        Compute deflection according to DIN EN 1992-1-1 cl. 7.4.3 :cite:`ec2`

        Curvatures for the fully cracked (``NONE_PARABOLIC``) and fully
        uncracked (``ELASTIC_ELASTIC``) states are computed separately and
        combined via the distribution coefficient ζ per EC2 Eq. (7.18) :cite:`ec2`.
        The quasi-permanent combination is used for integration; the rare
        combination determines the cracking state for ζ.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        loads : Loads
            Applied loads object.
        system : SystemType
            Structural system type.
        n_intervals : int
            Number of Simpson integration intervals (must be even).
        N_axial_N : float
            Applied normal force [N] (positive = tension).
        m_k_simplification : bool or int or float or sequence of floats
            Identifies the cached M-κ curves via
            :func:`make_mk_curve_key_sls_EC` /
            ``analysis_cache.require_mk_curve``.
        m_k_simplification_softening : int or float or sequence of floats or None
            Softening extras that identify the required cached M-κ curves.
        m_k_num_points : int
            Number of curvature points for the full M-κ diagram.
        debug : bool
            If ``True``, prints M_cr and ζ arrays to console.
        extended_debug : bool
            If ``True``, prints full κ-interpolation, ζ, and Simpson tables.

        Returns
        -------
        float
            Maximum midspan deflection [mm].
        """
        # --------------------------
        # Setup
        # --------------------------
        span_m = mm_to_m(slab_construction.slab.L)
        # Interpolation points (half span due to symmetry)
        x_positions = np.linspace(0, 0.5, n_intervals + 1)

        # Shared args for both κ-array calls
        kappa_kwargs = dict(
            slab_construction=slab_construction,
            loads=loads,
            system=system,
            combination="QUASI_PERMANENT",
            n_intervals=n_intervals,
            N_axial_N=N_axial_N,
            m_k_simplification=m_k_simplification,
            m_k_simplification_softening=m_k_simplification_softening,
            m_k_num_points=m_k_num_points,
        )

        # --------------------------
        # Curvatures Fully Cracked
        # --------------------------
        kappa_array_fully_cracked, kappa_cracked_debug = DeflectionCalculator._get_interpolated_kappa_array(
            constitutive_law = "NONE_PARABOLIC",
            **kappa_kwargs
        )

        # --------------------------
        # Curvatures Fully Uncracked
        # --------------------------
        kappa_array_fully_uncracked, kappa_uncracked_debug = DeflectionCalculator._get_interpolated_kappa_array(
            constitutive_law="ELASTIC_ELASTIC",
            **kappa_kwargs
        )

        # --------------------------
        # Zeta Calculation
        # --------------------------
        zeta_array, zeta_debug = DeflectionCalculator._calculate_zeta_array(
            slab_construction=slab_construction,
            loads=loads,
            system=system,
            N_axial_N=N_axial_N,
            x_positions=x_positions,
        )
        m_real_array_kNm = zeta_debug["m_qp_kNm"] # Use quasi-permanent combination for integration
        m_cr_interp_list_kNm = zeta_debug["m_cr_interp_kNm"]

        # -----------------------------
        # Weighted Curvatures
        # -----------------------------
        kappa_weighted_array = DeflectionCalculator._calculate_weighted_kappa(
            kappa_array_fully_cracked=kappa_array_fully_cracked,
            kappa_array_fully_uncracked=kappa_array_fully_uncracked,
            zeta_array=zeta_array
        )

        # -----------------------------
        # Simpson Integration
        # -----------------------------
        deflection_weighted_mm, simpson_debug = DeflectionCalculator._simpson_integration(
            span_m=span_m,
            n_intervals=n_intervals,
            M_applied_array_kNm=m_real_array_kNm,
            kappa_array=kappa_weighted_array
        )

        if debug:
            print(f"M_cr_array: {m_cr_interp_list_kNm}")
            print(f"zeta_array: {zeta_array}")
        if extended_debug:
            DeflectionCalculator._print_kappa_interp_debug(kappa_cracked_debug, label="cracked")
            DeflectionCalculator._print_kappa_interp_debug(kappa_uncracked_debug, label="uncracked")
            DeflectionCalculator._print_zeta_debug(zeta_array, zeta_debug)
            DeflectionCalculator._print_simpson_debug(simpson_debug)

        return deflection_weighted_mm

    @staticmethod
    def _simpson_integration(
            span_m: float,
            n_intervals: int,
            M_applied_array_kNm: list[float],
            kappa_array: list[float],
    ) -> tuple[float, dict[str, list]]:
        """
        Integrate midspan deflection over the half-span using Simpson's rule
        and the virtual work principle, then double the result by symmetry.

        where M_v(x) is the virtual bending moment for a unit point load at
        midspan of a simply supported beam.

        Parameters
        ----------
        span_m : float
            Total beam span L [m].
        n_intervals : int
            Number of Simpson intervals (must be even).
        M_applied_array_kNm : list[float]
            Applied bending moment at each Simpson point [kNm].
            Length must be ``n_intervals + 1``.
        kappa_array : list[float]
            Curvature at each Simpson point [1/m].
            Length must be ``n_intervals + 1``.

        Returns
        -------
        tuple[float, dict[str, list]]
            - ``deflection_mm`` — maximum midspan deflection [mm].
            - ``debug_info`` — dict with per-point arrays:
              ``x_positions``, ``weights``, ``M_real_kNm``, ``M_virtual``,
              ``kappa``, ``increment``, ``cum_sum``.
        """
        # Setup integration points (half span due to symmetry)
        delta_x_norm = 0.5 / n_intervals  # normalized
        x_positions = np.linspace(0, 0.5, n_intervals + 1)
        weights = DeflectionCalculator._simpson_weights(n_intervals)

        # Calculate curvatures and integrate
        integral_sum = 0.0

        # Store detailed Simpson Point Information
        # Per-point debug arrays
        weight_list: list[float] = []
        m_virtual_list: list[float] = []
        increment_list: list[float] = []
        cum_sum_list: list[float] = []

        # Iterate over all integration points
        for i, (x_norm, weight_x) in enumerate(zip(x_positions, weights)):
            # ----------------------------------------------------------
            # Use Simpson Integration Method to estimate ∫ Mv(x) * κ(x)
            #
            # 1. Find Curvature at x
            # 2. Find Virtual Moment at x
            # 3. Virtual work integration for half span
            # ----------------------------------------------------------

            # 1. Find Curvature at x
            kappa_x = kappa_array[i]

            # 2. Find Virtual Moment at x
            M_virtual_x_kNm = DeflectionCalculator._virtual_moment(x_norm, span_m)

            # 3. Virtual work integration
            increment = kappa_x * M_virtual_x_kNm * weight_x
            integral_sum += increment

            # Store detailed Simpson Point Information
            weight_list.append(float(weight_x))
            m_virtual_list.append(M_virtual_x_kNm)
            increment_list.append(increment)
            cum_sum_list.append(integral_sum)

        # Complete Simpson's rule integration for full span
        delta_x_real = delta_x_norm * span_m
        deflection_m = (integral_sum * delta_x_real / 3) * 2  # ×2 for full span
        deflection_mm = deflection_m * 1000 # Convert m to mm

        debug_info = {
            "x_positions": list(x_positions),
            "weights": weight_list,
            "M_real_kNm": list(M_applied_array_kNm),
            "M_virtual": m_virtual_list,
            "kappa": list(kappa_array),
            "increment": increment_list,
            "cum_sum": cum_sum_list,
        }

        return deflection_mm, debug_info

    @staticmethod
    def _print_simpson_debug(simpson_debug: dict[str, list]) -> None:
        header = (
            f"{'i':>4} {'x_norm':>8} {'weight':>8} {'M_real[kNm]':>12} "
            f"{'M_virt':>12} {'kappa[1/m]':>14} {'increment':>14} {'cum_sum':>14}"
        )
        print("\n[DEBUG] Simpson integration points (0 → 0.5):")
        print(header)
        print("-" * len(header))
        for i, (x, w, m_r, m_v, k, incr, cs) in enumerate(zip(
                simpson_debug["x_positions"],
                simpson_debug["weights"],
                simpson_debug["M_real_kNm"],
                simpson_debug["M_virtual"],
                simpson_debug["kappa"],
                simpson_debug["increment"],
                simpson_debug["cum_sum"],
        )):
            print(
                f"{i:>4} {x:>8.4f} {w:>8.1f} {m_r:>12.4f} {m_v:>12.6f} "
                f"{k:>14.6e} {incr:>14.6e} {cs:>14.6e}"
            )

    @staticmethod
    def _calculate_zeta_array(
            slab_construction: SlabConstruction,
            loads: Loads,
            system: SystemType,
            N_axial_N: float,
            x_positions: np.ndarray,
            beta: float = 0.5,
    ) -> tuple[list[float], dict[str, list[float]]]:
        """
        Compute the EN 1992-1-1 distribution coefficient ζ along the half-span :cite:`ec2`.

        ζ weights the contributions of the fully cracked and fully uncracked
        curvatures per EC2 Eq. (7.18) :cite:`ec2`:


        The cracking state is evaluated under the rare combination per
        EC2 Eq. (7.19) :cite:`ec2`:

        M_cr is parabolically interpolated between the cracking moments at
        the support and midspan sections.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        loads : Loads
            Applied loads object.
        system : SystemType
            Structural system type.
        N_axial_N : float
            Applied normal force [N] (positive = tension).
        x_positions : np.ndarray
            Normalized Simpson points along the half-span [0, 0.5] [-].
        beta : float, optional
            Load duration factor (0.5 for short-term, 1.0 for sustained or
            cyclic loading) [-]. Default is ``0.5``.

        Returns
        -------
        tuple[list[float], dict[str, list[float]]]
            - ``zeta_array`` — ζ values at each point [-].
            - ``debug_info`` — dict with per-point lists aligned with
              ``x_positions``:

              - ``"x_positions"`` — normalized positions [-].
              - ``"m_cr_interp_kNm"`` — interpolated cracking moment [kNm].
              - ``"m_rare_kNm"`` — rare combination moment [kNm].
              - ``"m_qp_kNm"`` — quasi-permanent combination moment [kNm].
        """
        cache = slab_construction.analysis_cache
        m_cr_support = cache.require_m_cr(
            SectionResultKey("M_cr", 0.0, "SLS", N_axial_N)
        )
        m_cr_mid = cache.require_m_cr(
            SectionResultKey("M_cr", 0.5, "SLS", N_axial_N)
        )

        # Cracking moments at support and midspan: Nmm → kNm, flipped to positive
        m_cr_support_kNm = -m_cr_support["m_cr"] / 1e6
        m_cr_mid_kNm = -m_cr_mid["m_cr"] / 1e6

        zeta_array: list[float] = []
        m_cr_interp_list_kNm: list[float] = []
        m_rare_list_kNm: list[float] = []
        m_qp_list_kNm: list[float] = []

        for x_norm in x_positions:
            # Cracking moment at this point
            m_cr_kNm = DeflectionCalculator._parabolic_interpolate(
                m_cr_support_kNm,
                m_cr_mid_kNm,
                x_norm
            )

            # Quasi-Permanent moment for debug info
            m_qp_kNm = InternalForces.calculate_moment_kNm(
                slab_construction,
                loads,
                system,
                "QUASI_PERMANENT",
                x_norm
            )

            # Cracking state under rare combination, used for zeta calculation
            m_rare_kNm = InternalForces.calculate_moment_kNm(
                slab_construction,
                loads,
                system,
                "RARE",
                x_norm
            )

            # EC2 (7.19): Use rare combination to determine cracking state of structure
            if m_rare_kNm > m_cr_kNm:
                zeta = 1.0 - beta * (m_cr_kNm / m_rare_kNm) ** 2
                zeta = max(0.0, min(1.0, zeta))
            else:
                zeta = 0.0

            zeta_array.append(zeta)
            m_cr_interp_list_kNm.append(m_cr_kNm)
            m_rare_list_kNm.append(m_rare_kNm)
            m_qp_list_kNm.append(m_qp_kNm)

        debug_info = {
            "x_positions": list(x_positions),
            "m_cr_interp_kNm": m_cr_interp_list_kNm,
            "m_rare_kNm": m_rare_list_kNm,
            "m_qp_kNm": m_qp_list_kNm,
        }

        return zeta_array, debug_info

    @staticmethod
    def _print_zeta_debug(zeta_array: list[float], zeta_debug: dict[str, list]) -> None:
        header = (
            f"{'i':>4} {'x_norm':>8} {'M_rare':>10} {'M_qp':>10} "
            f"{'M_cr':>10} {'zeta':>8}"
        )
        print("\n[DEBUG] Zeta calculation (0 → 0.5):")
        print(header)
        print("-" * len(header))
        for i, (x, m_rare, m_qp, m_cr, z) in enumerate(zip(
                zeta_debug["x_positions"],
                zeta_debug["m_rare_kNm"],
                zeta_debug["m_qp_kNm"],
                zeta_debug["m_cr_interp_kNm"],
                zeta_array,
        )):
            print(f"{i:>4} {x:>8.4f} {m_rare:>10.4f} {m_qp:>10.4f} {m_cr:>10.4f} {z:>8.4f}")

    @staticmethod
    def _calculate_weighted_kappa(
            kappa_array_fully_cracked: list[float],
            kappa_array_fully_uncracked: list[float],
            zeta_array: list[float]
    ) -> list[float]:
        """
        Compute ζ-weighted curvatures per EN 1992-1-1 Eq. (7.18) :cite:`ec2`.

        Parameters
        ----------
        kappa_array_fully_cracked : list[float]
            Curvatures for the fully cracked state at each point [1/m].
        kappa_array_fully_uncracked : list[float]
            Curvatures for the fully uncracked state at each point [1/m].
        zeta_array : list[float]
            Distribution coefficients ζ at each point [-].

        Returns
        -------
        list[float]
            Weighted curvatures κ at each point [1/m].
        """
        kappa_weighted_array = []

        for k_cr, k_uncr, zeta in zip(kappa_array_fully_cracked, kappa_array_fully_uncracked, zeta_array):
            # Weighted Kappa according to EC2 Eq. (7.18)
            kappa_weighted = zeta * k_cr + (1 - zeta) * k_uncr
            kappa_weighted_array.append(kappa_weighted)

        return kappa_weighted_array

    @staticmethod
    def _get_interpolated_kappa_array(
            slab_construction: SlabConstruction,
            loads: Loads,
            system: SystemType,
            combination: str,
            n_intervals: int,
            N_axial_N: float,
            constitutive_law: str,
            m_k_simplification,
            m_k_simplification_softening,
            m_k_num_points: int,
    ) -> tuple[list[float], dict[str, list]]:
        """
        Compute κ(x) at each Simpson point by parabolically interpolating
        the raw M-κ curve between support and midspan at x, applying
        force-control (terminal cut + dip-fill) on that local curve, then
        looking up κ at the applied moment at x.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        loads : Loads
            Applied loads object.
        system : SystemType
            Structural system type.
        combination : str
            Normalized EN 1990 load combination name.
        n_intervals : int
            Number of Simpson intervals (must be even).
        N_axial_N : float
            Applied normal force [N] (positive = tension).
        constitutive_law : str
            Concrete constitutive law keyword.
        m_k_simplification : bool or int or float or sequence of floats
            Simplification control that identifies the required cached M-κ curves.
        m_k_simplification_softening : int or float or sequence of floats or None
            Softening extras that identify the required cached M-κ curves.
        m_k_num_points : int
            Number of curvature points for the full M-κ diagram.

        Returns
        -------
        tuple[list[float], dict[str, list]]
            - ``kappas`` — curvature κ at each Simpson point [1/m].
            - ``debug_info`` — dict with M-κ diagram arrays at support and
              midspan, applied moments, interpolation bounds, and κ values,
              all aligned with the Simpson points.
        """
        # Setup integration points (half span due to symmetry)
        x_positions = np.linspace(0, 0.5, n_intervals + 1)

        # Coordinated M-κ curves must be prepared before B1a/B2a.
        cache = slab_construction.analysis_cache
        mk_key = make_mk_curve_key_sls_EC(
            constitutive_law=constitutive_law,
            simplification=m_k_simplification,
            simplification_softening=m_k_simplification_softening,
            n=N_axial_N,
            m_k_num_points=m_k_num_points,
        )
        M_k_result_support = cache.require_mk_curve(mk_key, 0.0)
        M_k_result_mid = cache.require_mk_curve(mk_key, 0.5)

        # Calculate real Moment at integration points
        M_applied_array_kNm = [
            InternalForces.calculate_moment_kNm(
                slab_construction,
                loads,
                system,
                combination,
                x
            )
            for x in x_positions
        ]

        kappas = []

        # Debug arrays
        m_support_list      = list(M_k_result_support.m_y)
        kappa_support_list  = list(M_k_result_support.chi_y)
        m_midspan_list      = list(M_k_result_mid.m_y)
        kappa_midspan_list  = list(M_k_result_mid.chi_y)
        m_applied_list      = []
        m_low_list          = []
        m_high_list         = []
        kappa_low_list      = []
        kappa_high_list     = []

        for x_norm, M_applied_kNm in zip(x_positions, M_applied_array_kNm):
            # Compute interpolated M-k-curve at position x_norm from raw
            # equal-length cache curves (still include post-peak softening).
            M_array_interp_kNm = DeflectionCalculator._parabolic_interpolate(
                -M_k_result_support.m_y / 1e6,
                -M_k_result_mid.m_y / 1e6,
                x_norm
            )
            kappa_array_interp = DeflectionCalculator._parabolic_interpolate(
                -M_k_result_support.chi_y * 1000,
                -M_k_result_mid.chi_y * 1000,
                x_norm
            )

            # Force-control on the local curve only: terminal cut + dip-fill.
            # Support/mid may imply different (M_cap, κ_cap) without breaking
            # the equal-length cache topology.
            M_array_interp_kNm, kappa_array_interp, _fc_meta = (
                _force_controlled_mk_arrays(
                    M_array_interp_kNm,
                    kappa_array_interp,
                    cut_terminal_softening=True,
                )
            )

            # Linear κ(M) on the local force-controlled curve; reject above
            # capacity (do not clamp via numpy.interp endpoints).
            kappa_x = lookup_kappa_force_controlled(
                M_applied_kNm, M_array_interp_kNm, kappa_array_interp
            )
            kappas.append(kappa_x)

            # Record the bracketing points used for the lookup — same local
            # force-controlled curve as the κ(M) lookup above.
            M_low, M_high, kappa_low, kappa_high = DeflectionCalculator._interpolation_bounds(
                M_applied_kNm, M_array_interp_kNm, kappa_array_interp
            )
            m_applied_list.append(M_applied_kNm)
            m_low_list.append(M_low)
            m_high_list.append(M_high)
            kappa_low_list.append(kappa_low)
            kappa_high_list.append(kappa_high)

        debug_info = {
            "m_support_list": m_support_list,
            "kappa_support_list": kappa_support_list,
            "m_midspan_list": m_midspan_list,
            "kappa_midspan_list": kappa_midspan_list,
            "x_positions": list(x_positions),
            "M_applied_kNm": m_applied_list,
            "M_low": m_low_list,
            "M_high": m_high_list,
            "kappa_low": kappa_low_list,
            "kappa_high": kappa_high_list,
            "kappa": list(kappas)
        }

        return kappas, debug_info

    @staticmethod
    def _print_kappa_interp_debug(kappa_debug: dict[str, list], label: str = "") -> None:
        print("[DEBUG] Moment-Curvature-Diagram-Lists at Support and Midspan")
        print("Moments")
        print(" - Support")
        print(kappa_debug["m_support_list"])
        print(" - Midspan")
        print(kappa_debug["m_midspan_list"])
        print("\nCurvatures")
        print(" - Support")
        print(kappa_debug["kappa_support_list"])
        print(" - Midspan")
        print(kappa_debug["kappa_midspan_list"])

        header = (
            f"{'x_norm':>8} {'M_applied_kNm':>14} {'M_low':>12}"
            f"{'M_high':>12} {'kappa_low':>14} {'kappa_high':>14} {'kappa [1/m]':>14}"
        )
        title = "[DEBUG] κ-interpolation bounds"
        if label:
            title = f"{title} ({label})"
        print(f"\n{title}:")
        print(header)
        print("-" * len(header))
        for x, m_a, m_lo, m_hi, k_lo, k_hi, k in zip(
                kappa_debug["x_positions"],
                kappa_debug["M_applied_kNm"],
                kappa_debug["M_low"],
                kappa_debug["M_high"],
                kappa_debug["kappa_low"],
                kappa_debug["kappa_high"],
                kappa_debug["kappa"],
        ):
            print(
                f"{x:>8.4f} {m_a:>14.4f} "
                f"{m_lo:>12.4f} {m_hi:>12.4f} {k_lo:>14.6e} {k_hi:>14.6e} {k:>14.6e} "
            )

    @staticmethod
    def _interpolation_bounds(x, xs, ys):
        """
        Return the bracketing index pair ``(x_lo, x_hi, y_lo, y_hi)`` for a
        sorted array such that ``x_lo ≤ x ≤ x_hi``.
        Used for kappa interpolation within M-κ curve

        Parameters
        ----------
        x : float
            Query value.
        xs : array-like
            Sorted x-values (strictly monotone increasing).
        ys : array-like
            y-values aligned with ``xs``.

        Returns
        -------
        tuple[float, float, float, float]
            ``(x_lo, x_hi, y_lo, y_hi)`` — the bracketing x- and y-values.
        """
        xs = np.asarray(xs)
        ys = np.asarray(ys)

        i = np.searchsorted(xs, x)

        i_lo = max(i - 1, 0)
        i_hi = min(i, len(xs) - 1)

        return xs[i_lo], xs[i_hi], ys[i_lo], ys[i_hi]

    @staticmethod
    def _parabolic_interpolate(
            y_0: Union[float, np.ndarray],
            y_1: Union[float, np.ndarray],
            x_norm: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Parabolically interpolate between two reference values y_0 and y_1
        according to Loutfi :cite:`loutfi_2023`

        Parameters
        ----------
        y_0 : float or np.ndarray
            Function value at the reference point x = 0 (support).
        y_1 : float or np.ndarray
            Function value at the reference point x = 0.5 (midspan / vertex).
        x_norm : float or np.ndarray
            Normalized position(s) in [0, 0.5] [-].

        Returns
        -------
        float or np.ndarray
            Interpolated value(s) at ``x_norm``.
        """
        y_0 = np.asarray(y_0)
        y_1 = np.asarray(y_1)
        if y_0.shape != y_1.shape:
            raise ValueError(
                "parabolic interpolation requires equal-shaped endpoints, "
                f"got {y_0.shape} and {y_1.shape}"
            )
        factor = (y_1 - y_0) / 0.5 ** 2
        y_interp = -factor * (x_norm - 0.5) ** 2 + y_1

        return y_interp

    @staticmethod
    def _simpson_weights(n: int) -> np.ndarray:
        """
        Generate the Simpson's rule weight vector ``[1, 4, 2, 4, 2, …, 4, 1]``.

        Parameters
        ----------
        n : int
            Number of integration intervals (must be even).

        Returns
        -------
        np.ndarray
            Weight array of length ``n + 1``.
        """
        weights = np.ones(n + 1)
        weights[1:-1:2] = 4  # Odd indices
        weights[2:-1:2] = 2  # Even indices (except first and last)
        return weights

    @staticmethod
    def _virtual_moment(x_norm: float, span_m: float) -> float:
        """
        Compute the virtual bending moment for a unit point load at midspan
        of a simply supported beam. Used for virtual work method to find deflection
        at midspan for a simply supported beam.

        Parameters
        ----------
        x_norm : float
            Normalized position along the half-span (0 at support, 0.5 at
            midspan) [-].
        span_m : float
            Total span length L [m].

        Returns
        -------
        float
            Virtual bending moment M_v [m] (moment arm for a unit load [kN]).
        """
        return x_norm * span_m / 2