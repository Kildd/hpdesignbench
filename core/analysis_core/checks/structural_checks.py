"""
Structural utilization checks for HP-shell slab constructions.
Numbering according to Loutfi :cite:`loutfi_2023`

Covers ultimate limit state (ULS) bending strength and serviceability
limit state (SLS) deflection and cracking checks per EN 1992-1-1:2004/NA(DE) :cite:`ec2`

Adapted from: Jamila Loutfi :cite:`loutfi_2023`
"""
from abc import ABC, abstractmethod

from core.analysis_core.statics.constants import SystemType, MomentType
from core.analysis_core.statics.deflection import DeflectionCalculator
from core.analysis_core.statics.internal_forces import InternalForces
from core.analysis_core.statics.loads import Loads
from core.analysis_core.section_methods import (
    DEFLECTION_MK_SIMPLIFICATION,
    DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
    AboveMkCapacityError,
    InvalidSectionForMKError,
)
from core.unit_core import Nmm_to_kNm
from slab_construction.slab_construction import SectionResultKey, SlabConstruction
from core.analysis_core.run_config import require_invalid_utilization


class StructuralCheck(ABC):
    """
    Abstract base class for structural utilization checks.
    Author: Elliot Melcer

    All StructuralCheck subclasses must implement :meth:`calculate_utilization`,
    which returns a dimensionless utilization ratio where values ≤ 1.0
    indicate a passing design.
    """

    @staticmethod
    @abstractmethod
    def calculate_utilization(
            slab_construction: SlabConstruction,
            loads: Loads,
            system: str,
            moment: str) -> float:
        """
        Compute the structural utilization ratio for a slab construction.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        loads : Loads
            Applied loads object (uniformly distributed over all spans).
        system : str
            Structural system identifier (see :class:`SystemType`).
        moment : str
            Moment type identifier (see :class:`MomentType`).

        Returns
        -------
        float
            Utilization ratio [-]. Values ≤ 1.0 indicate a passing design.
        """
        raise NotImplementedError


"""ULTIMATE LIMIT STATE"""


"""A.1 Ultimate Moment Check"""


class UltimateMomentCheckEC2004DE(StructuralCheck):
    """
    ULS bending strength check per EN 1992-1-1:2004/NA(DE) :cite:`ec2`.

    Computes the utilization ratio |M_Ed| / M_Rd at the cross-section
    position where the governing design moment occurs. For positive
    moments the section is used as-is; for negative moments the section
    is flipped before calling the strength calculation.

    Infeasible sections (no valid equilibrium) are penalized with a
    sentinel ``opt_invalid_utilization``.
    """

    @staticmethod
    def calculate_utilization(
            slab_construction: SlabConstruction,
            loads: Loads,
            system: SystemType = SystemType.SIMPLE_BEAM,
            moment: MomentType = MomentType.MAX_POS_MOMENT,
            n: float = 0.0,
            debug_print: bool = False,
            *,
            invalid_utilization: float | None = None,
    ) -> float:
        """
        Compute the ULS bending utilization ratio |M_Ed| / M_Rd.
        Adapted from Loutfi :cite:`loutfi_2023`

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        loads : Loads
            Applied loads object (uniformly distributed over all spans).
        system : SystemType, optional
            Structural system type. Default is
            :attr:`SystemType.SIMPLE_BEAM`.
        moment : MomentType, optional
            Governing moment type. Default is
            :attr:`MomentType.MAX_POS_MOMENT`.
        n : float, optional
            Applied normal force [N] (positive = tension). Default is
            ``0.0``.
        debug_print : bool, optional
            If ``True``, prints intermediate values (system, moment type,
            x-position, M_Rd, M_Ed, utilization) to console.
            Default is ``False``.

        Returns
        -------
        float
            Utilization ratio |M_Ed| / M_Rd [-]. Returns ``opt_invalid_utilization`` for
            infeasible sections.
        """
        # Get moment data (coefficient and x-position) from lookup table
        moment_data = InternalForces.get_moment_data(system, moment)
        x_position = moment_data["x_position"]

        # Validate x-position is within bounds for this system
        InternalForces.validate_x_position(system, x_position)

        # Read resistance from the per-FE analysis cache.
        m_u_key = SectionResultKey(
            result_type="M_u",
            x=x_position,
            limit_state="ULS",
            n=n,
            flipped=moment != MomentType.MAX_POS_MOMENT,
        )
        res = slab_construction.analysis_cache.require_m_u(m_u_key)
        if not res.get("valid", True):
            return require_invalid_utilization(invalid_utilization)

        if moment == MomentType.MAX_POS_MOMENT:
            MRd_kNm = -Nmm_to_kNm(res["m_u"])
        else:
            MRd_kNm = Nmm_to_kNm(res["m_u"])

        # Calculate design moment based on system and moment type
        MEd_kNm = InternalForces.calculate_moment_kNm(
            slab_construction,
            loads,
            system,
            combination="FUNDAMENTAL",
            moment=moment
        )

        # Calculate utilization
        utilization = abs(MEd_kNm / MRd_kNm)

        if debug_print:
            print(f"System: {system}, Moment Type: {moment}")
            print(f"x-position: {x_position:.3f}")
            print(f"M_Rd = {MRd_kNm:.3f} kNm")
            print(f"M_Ed = {MEd_kNm:.3f} kNm")
            print(f"Utilization = {utilization:.3f} ({utilization * 100:.1f}%)\n")

        return utilization


"""SERVICEABILITY LIMIT STATE"""


"""B.1a Deflection Limit Check"""


class DeflectionLimitByDeflectionCheckEC2004DE(StructuralCheck):
    """
    SLS deflection limit check by computed deflection (B.1a).

    Verifies that the magnitude of the quasi-permanent deflection |w|
    does not exceed the span-based limit L / limit_factor. Sagging and
    hogging are penalized equally (GH-aligned).

    Infeasible sections that raise :exc:`InvalidSectionForMKError` are
    penalized with a sentinel ``opt_invalid_utilization``.
    """

    @staticmethod
    def calculate_utilization(
            slab_construction: SlabConstruction,
            loads: Loads,
            system: SystemType = SystemType.SIMPLE_BEAM,
            limit_factor: float = 250.0,
            n: float = 0.0,
            debug: bool = False,
            *,
            invalid_utilization: float | None = None,
    ) -> float:
        """
        Compute the SLS deflection utilization ratio |w| / (L / limit_factor).
        Adapted from Loutfi :cite:`loutfi_2023`

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        loads : Loads
            Applied loads object.
        system : SystemType, optional
            Structural system type. Default is
            :attr:`SystemType.SIMPLE_BEAM`.
        limit_factor : float, optional
            Span fraction used to derive the deflection limit
            w_limit = L / limit_factor [-]. Default is ``250.0``.
        n : float, optional
            External axial force [N] (positive = tension). Must match the
            ``n`` used when preparing the deflection M–κ / M_cr cache.
            Default is ``0.0``.
        debug : bool, optional
            If ``True``, passes debug flag to the deflection calculator
            and prints raw utilization, w_max, and L to console.
            Default is ``False``.

        Returns
        -------
        float
            Utilization ratio |w_max| / w_limit [-]. Returns
            ``opt_invalid_utilization`` for infeasible sections.
        """
        try:
            w_max_sls = DeflectionCalculator.calculate_deflection_mm_EC(
                slab_construction,
                loads,
                system,
                load_history_method="NONE",
                m_k_simplification=DEFLECTION_MK_SIMPLIFICATION,
                m_k_simplification_softening=DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
                combination="QUASI_PERMANENT",
                N_axial_N=n,
                debug=debug,
            )
        except (InvalidSectionForMKError, AboveMkCapacityError, RuntimeError) as e:
            if debug:
                print(f"[B1a] section infeasible for deflection calc: {e}")
            return require_invalid_utilization(invalid_utilization)

        L = slab_construction.slab.L

        w_limit = L / limit_factor

        utilization = abs(w_max_sls) / w_limit

        if debug:
            print("util = ", utilization)
            print("w_max = ", w_max_sls)
            print("L = ", L)

        return utilization


class FailureAnnouncementByDeflectionCheckEC2004DE(StructuralCheck):
    """
    Failure announcement check by deflection under fundamental loads (B.2a).

    Verifies that the sagging (downward, positive) midspan deflection under
    the fundamental load combination ``w_uls`` exceeds the minimum ductility
    threshold ``w_min = L / min_factor``, so that a sufficient warning is
    ensured. A utilization ≤ 1.0 means the deflection at failure is large
    enough.

    Negative (hogging / upward) and zero deflections are not meaningful for
    this simply-supported gravity proxy and return
    ``opt_invalid_utilization``. Infeasible sections likewise return the
    sentinel.
    """

    @staticmethod
    def calculate_utilization(
            slab_construction: SlabConstruction,
            loads: Loads,
            system: SystemType = SystemType.SIMPLE_BEAM,
            *,
            min_factor: float,
            n: float = 0.0,
            debug: bool = False,
            invalid_utilization: float | None = None,
    ) -> float:
        """
        Compute the failure announcement utilization ratio w_min / w_uls.
        Adapted from Loutfi :cite:`loutfi_2023`

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        loads : Loads
            Applied loads object.
        system : SystemType, optional
            Structural system type. Default is
            :attr:`SystemType.SIMPLE_BEAM`.
        min_factor : float
            Span fraction used to derive the minimum required deflection
            w_min = L / min_factor [-]. Required; production supplies
            ``defl_max_announce_failure`` from CSV (typically 100 → L/100).
        n : float, optional
            External axial force [N] (positive = tension). Must match the
            ``n`` used when preparing the deflection M–κ / M_cr cache.
            Default is ``0.0``.
        debug : bool, optional
            If ``True``, passes debug flag to the deflection calculator
            and prints w_max, w_min, and utilization to console.
            Default is ``False``.

        Returns
        -------
        float
            Utilization ratio w_min / w_uls [-] for sagging ``w_uls > 0``.
            Returns ``opt_invalid_utilization`` for infeasible sections,
            non-positive ``w_uls``, or near-zero deflection.
        """
        try:
            w_max_uls = DeflectionCalculator.calculate_deflection_mm_EC(
                slab_construction,
                loads,
                system,
                load_history_method="NONE",
                m_k_simplification=DEFLECTION_MK_SIMPLIFICATION,
                m_k_simplification_softening=DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
                combination="FUNDAMENTAL",
                N_axial_N=n,
                debug=debug,
            )
        except (InvalidSectionForMKError, AboveMkCapacityError, RuntimeError) as e:
            if debug:
                print(f"[B2a] section infeasible for deflection calc: {e}")
            return require_invalid_utilization(invalid_utilization)

        L = slab_construction.slab.L

        w_min = L / min_factor
        # Positive = sagging (downward). Hogging/upward or zero → invalid.
        if w_max_uls <= 0.0:
            if debug:
                print(f"[B2a] non-positive deflection w={w_max_uls}; invalid")
            return require_invalid_utilization(invalid_utilization)

        utilization = w_min / w_max_uls

        if debug:
            print(f"w_max = {w_max_uls:.2f}")
            print(f"w_min = {w_min:.2f}")
            print(f"util = {w_min:.2f} / {w_max_uls:.2f} = {utilization:.2f}")

        return utilization
