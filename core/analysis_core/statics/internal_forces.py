"""
Internal forces utility class for uniform distributed load on a simply
supported one-way beam.

Author: Elliot Melcer
"""
from typing import Dict, Optional, Callable

from slab_construction.slab_construction import SlabConstruction
from core.analysis_core.statics.constants import (
    MOMENT_DATA,
    MAX_X_POSITIONS,
    SystemType,
    MomentType,
)
from core.analysis_core.statics.loads import LoadsEC, Loads
from ...unit_core import *


class InternalForces:
    """
    Utility class (not instantiable) for computing internal forces on a
    simply supported beam under uniformly distributed load.

    Position convention: x is normalized, with x = 0 at the first support
    and x = 1 at the second support.
    """

    @staticmethod
    def validate_x_position(system: SystemType, x_position: float) -> None:
        """
        Validate that an x-position is within the valid range for a given system.

        Parameters
        ----------
        system : SystemType
            Structural system type.
        x_position : float
            Normalized position to validate [-].

        Raises
        ------
        ValueError
            If ``system`` is not in :data:`MAX_X_POSITIONS`, or if
            ``x_position`` is outside ``[0, max_x]`` for that system.
        """
        if system not in MAX_X_POSITIONS:
            raise ValueError(
                f"Invalid system '{system}'. Must be one of: {list(MAX_X_POSITIONS.keys())}"
            )

        max_x = MAX_X_POSITIONS[system]

        if x_position < 0 or x_position > max_x:
            raise ValueError(
                f"Invalid x-position {x_position} for system '{system}'. "
                f"Must be between 0 and {max_x}."
            )

    @staticmethod
    def get_moment_data(system: SystemType, moment_type: MomentType) -> Dict[str, float]:
        """
        Return the moment coefficient and governing x-position from the lookup table.

        Parameters
        ----------
        system : SystemType
            Structural system type.
        moment_type : MomentType
            Governing moment type (``MAX_POS_MOMENT``).

        Returns
        -------
        dict
            Dictionary with keys:

            - ``"coefficient"`` — moment coefficient α such that M = α · w · L² [-].
            - ``"x_position"`` — normalized position where the governing moment
              occurs [-].

        Raises
        ------
        ValueError
            If ``system`` or ``moment_type`` is not found in :data:`MOMENT_DATA`.
        """
        if system not in MOMENT_DATA:
            raise ValueError(
                f"Invalid system '{system}'. Must be one of: {list(MOMENT_DATA.keys())}"
            )

        if moment_type not in MOMENT_DATA[system]:
            raise ValueError(
                f"Invalid moment_type '{moment_type}'. "
                f"Must be one of: {list(MOMENT_DATA[system].keys())}"
            )

        return MOMENT_DATA[system][moment_type]

    @staticmethod
    def calculate_moment_kNm(
            slab_construction: SlabConstruction,
            loads: Loads,
            system: SystemType = SystemType.SIMPLE_BEAM,
            combination: str = "FUNDAMENTAL",
            x_norm: Optional[float] = None,
            moment: Optional[MomentType] = None
    ) -> float:
        """
        Calculate the bending moment at a specific position or the governing
        maximum moment for a simply supported beam under uniformly distributed load.

        Exactly one of ``x_norm`` or ``moment`` must be provided.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object (provides span and dead loads).
        loads : Loads
            Loads object (provides live loads and combination factors).
        system : SystemType, optional
            Structural system type. Default is :attr:`SystemType.SIMPLE_BEAM`.
        combination : str, optional
            Load combination name. Default is ``"FUNDAMENTAL"``.
        x_norm : float, optional
            Normalized position along the beam (0 at first support, 1 at
            second support) [-]. Use this to compute M(x) at an arbitrary
            position via the analytical moment function.
        moment : MomentType, optional
            Governing moment type. Use this to look up the maximum moment
            via the coefficient table. Mutually exclusive with ``x_norm``.

        Returns
        -------
        float
            Bending moment [kNm]. Sign follows the moment function convention
            (positive = sagging).

        Raises
        ------
        ValueError
            If both or neither of ``x_norm`` and ``moment`` are provided, or
            if the moment coefficient is 0.0 (moment type not applicable for
            this system).
        NotImplementedError
            If ``x_norm`` is provided but no moment function is implemented
            for ``system``.

        Examples
        --------
        Moment at a specific position:

        >>> M = InternalForces.calculate_moment_kNm(
        ...     slab, loads, system=SystemType.SIMPLE_BEAM, x_norm=0.3
        ... )

        Maximum positive midspan moment (qL²/8):

        >>> M_max = InternalForces.calculate_moment_kNm(
        ...     slab, loads, system=SystemType.SIMPLE_BEAM,
        ...     moment=MomentType.MAX_POS_MOMENT
        ... )
        """
        if (x_norm is None) == (moment is None):
            raise ValueError(
                "Must provide EITHER 'x' OR 'moment_type', but not both and not neither. "
                "Use 'x' for M(x) at specific position, or 'moment_type' for maximum moment."
            )

        span_m = mm_to_m(slab_construction.slab.L)
        w_line_kN_m = loads.combined_line_load_kN_m(slab_construction, combination)

        if x_norm is not None:
            if system not in MOMENT_FUNCTIONS:
                raise NotImplementedError(
                    f"M(x) function not implemented for {system}. "
                    f"Available systems: {list(MOMENT_FUNCTIONS.keys())}. "
                    f"For this system, use moment_type='MAX_POS_MOMENT' instead."
                )

            InternalForces.validate_x_position(system, x_norm)
            x_m = x_norm * span_m
            moment_func = MOMENT_FUNCTIONS[system]
            return moment_func(x_m, w_line_kN_m, span_m)

        moment_data = InternalForces.get_moment_data(system, moment)
        coefficient = moment_data["coefficient"]

        if coefficient == 0.0:
            raise ValueError(
                f"{moment} does not exist for {system}. "
                f"(moment coefficient is 0.0 in lookup table)"
            )

        return coefficient * w_line_kN_m * span_m ** 2

    @staticmethod
    def moment_simple_beam(x: float, w: float, L: float) -> float:
        """
        Compute the bending moment at position x for a simply supported beam.

        Parameters
        ----------
        x : float
            Position along the beam [m].
        w : float
            Uniformly distributed load [kN/m].
        L : float
            Span length [m].

        Returns
        -------
        float
            Bending moment M(x) [kNm] (positive = sagging).
        """
        return w * x * (L - x) / 2


"""
Moment functions M(x) for systems where the full distribution is implemented.

Signature: ``(x_m: float, w: float, L: float) -> float``
where ``x_m`` is position in meters, ``w`` is line load in kN/m, and
``L`` is the span length in meters.
"""
MOMENT_FUNCTIONS: Dict[SystemType, Callable[[float, float, float], float]] = {
    SystemType.SIMPLE_BEAM: InternalForces.moment_simple_beam,
}
