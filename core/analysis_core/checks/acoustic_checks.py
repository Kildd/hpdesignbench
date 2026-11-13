"""
Acoustic utilization checks for HP-shell slab constructions.

Wraps the regression-based acoustic calculation methods in a unified
check interface (utilization ≤ 1.0 = passing).

Adapted from: Jamila Loutfi :cite:`loutfi_2023`, Ahmad Eiz Eddin :cite:`eddin_2023`
"""
from abc import ABC, abstractmethod

from core.analysis_core.acoustic_methods import calculate_sound_reduction_index, \
    calculate_standard_impact_sound_pressure_level
from slab_construction.slab_construction import SlabConstruction


class AcousticCheck(ABC):
    """
    Abstract base class for acoustic utilization checks.
    Author: Elliot Melcer

    All AcousticCheck subclasses must implement :meth:`calculate_utilization`,
    which returns a dimensionless utilization ratio where values ≤ 1.0
    indicate a passing design.
    """

    @staticmethod
    @abstractmethod
    def calculate_utilization(
            slab_construction: SlabConstruction,
            limit_dB: float,
            mod_att: float,
            buffer_dB: float = 0.0
    ) -> float:
        """
        Compute the acoustic utilization ratio for a slab construction.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        limit_dB : float
            Regulatory limit value [dB].
        mod_att : float
            Modal attenuation parameter of the HP-shell [-].
        buffer_dB : float, optional
            Additional safety buffer applied to the calculated value [dB].
            Default is ``0.0``.

        Returns
        -------
        float
            Utilization ratio [-]. Values ≤ 1.0 indicate a passing design.
        """
        raise NotImplementedError


"""D.1 Airborne Sound Insulation Check"""


class AirborneSoundInsulationCheck(AcousticCheck):
    """
    Airborne sound insulation check (D.1).

    Computes the utilization ratio R_w,min / (R_w − buffer), where R_w is
    the weighted sound reduction index calculated by
    :func:`calculate_sound_reduction_index`. A utilization ≤ 1.0 means the
    slab provides sufficient airborne sound insulation.
    """

    @staticmethod
    def calculate_utilization(
        slab_construction: SlabConstruction,
        limit_dB: float,
        mod_att: float,
        buffer_dB: float = 0.0,
        debug_print: bool = False
    ) -> float:
        """
        Compute the airborne sound insulation utilization ratio.
        Adapted from: Jamila Loutfi :cite:`loutfi_2023`, Ahmad Eiz Eddin :cite:`eddin_2023`

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        limit_dB : float
            Minimum required weighted sound reduction index R_w,min [dB].
        mod_att : float
            Modal attenuation parameter of the HP-shell [-].
        buffer_dB : float, optional
            Safety buffer subtracted from the calculated R_w before forming
            the ratio [dB]. Default is ``0.0``.
        debug_print : bool, optional
            Reserved for future debug output. Currently unused.
            Default is ``False``.

        Returns
        -------
        float
            Utilization ratio R_w,min / (R_w − buffer) [-].
            Values ≤ 1.0 indicate a passing design.
        """
        R_w_min = limit_dB # dB

        R_w = calculate_sound_reduction_index(slab_construction, mod_att = mod_att)
        utilization = R_w_min / (R_w - buffer_dB)

        return utilization


"""D.2 Impact Sound Insulation Check"""


class ImpactSoundInsulationCheck(AcousticCheck):
    """
    Impact sound insulation check (D.2).

    Computes the utilization ratio (L_nw + buffer) / L_nw,max, where L_nw
    is the weighted standard impact sound pressure level calculated by
    :func:`calculate_standard_impact_sound_pressure_level`. A utilization
    ≤ 1.0 means the slab meets the impact sound requirement.
    """

    @staticmethod
    def calculate_utilization(
            slab_construction: SlabConstruction,
            limit_dB: float,
            mod_att: float,
            buffer_dB: float = 0.0,
            debug_print: bool = False
    ) -> float:
        """
        Compute the impact sound insulation utilization ratio.
        Adapted from: Jamila Loutfi :cite:`loutfi_2023`, Ahmad Eiz Eddin :cite:`eddin_2023`

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.
        limit_dB : float
            Maximum allowable weighted standard impact sound pressure level
            L_nw,max [dB].
        mod_att : float
            Modal attenuation parameter of the HP-shell [-].
        buffer_dB : float, optional
            Safety buffer added to the calculated L_nw before forming the
            ratio [dB]. Default is ``0.0``.
        debug_print : bool, optional
            Reserved for future debug output. Currently unused.
            Default is ``False``.

        Returns
        -------
        float
            Utilization ratio (L_nw + buffer) / L_nw,max [-].
            Values ≤ 1.0 indicate a passing design.
        """
        L_nw_max = limit_dB  # dB

        L_nw = calculate_standard_impact_sound_pressure_level(slab_construction, mod_att=mod_att)

        utilization = (L_nw + buffer_dB) / L_nw_max

        return utilization