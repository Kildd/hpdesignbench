"""
Enumerations and lookup tables for the simply supported one-way beam.

Only positive midspan demand is supported: M = (1/8) · w · L² at x = 0.5.
"""
from typing import Dict
from enum import Enum


class SystemType(str, Enum):
    """Enumeration of supported structural system types."""

    SIMPLE_BEAM = "SIMPLE_BEAM"


class MomentType(str, Enum):
    """Enumeration of governing moment types."""

    MAX_POS_MOMENT = "MAX_POS_MOMENT"


"""
Maximum valid x-position for each system type.
x = 0 at first support, x = 1 at second support.

  |-> x
  0      0.5      1

  ===================
  ^               ^
"""

MAX_X_POSITIONS: Dict[str, float] = {
    "SIMPLE_BEAM": 1.0,  # From support 0 to support 1
}


"""
Lookup table for moment coefficients and governing x-positions
according to :cite:`schneider_2016`.

M = coefficient · w · L²

Keys: system name → moment type → {coefficient, x_position}.
"""

MOMENT_DATA: Dict[str, Dict[str, Dict[str, float]]] = {
    "SIMPLE_BEAM": {
        "MAX_POS_MOMENT": {
            "coefficient": 0.125,  # q L² / 8
            "x_position": 0.5,
        },
    },
}
