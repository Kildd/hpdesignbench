"""
Unit conversion utilities for structuralcodes output (N, mm) to common
engineering units (kN, m).
"""

def mm_to_m(x: float) -> float:
    """
    Convert a length from millimetres to metres.

    Parameters
    ----------
    x : float
        Length [mm].

    Returns
    -------
    float
        Length [m].
    """
    return x * 1e-3


def mm2_to_m2(x: float) -> float:
    """
    Convert an area from square millimetres to square metres.

    Parameters
    ----------
    x : float
        Area [mm²].

    Returns
    -------
    float
        Area [m²].
    """
    return x * 1e-6


def mm3_to_m3(x: float) -> float:
    """
    Convert a volume from cubic millimetres to cubic metres.

    Parameters
    ----------
    x : float
        Volume [mm³].

    Returns
    -------
    float
        Volume [m³].
    """
    return x * 1e-9


def Nmm_to_kNm(x: float) -> float:
    """
    Convert a bending moment from Newton-millimetres to kilonewton-metres.

    Parameters
    ----------
    x : float
        Bending moment [Nmm].

    Returns
    -------
    float
        Bending moment [kNm].
    """
    return x * 1e-6


def N_to_kN(x: float) -> float:
    """
    Convert a force from Newtons to kilonewtons.

    Parameters
    ----------
    x : float
        Force [N].

    Returns
    -------
    float
        Force [kN].
    """
    return x * 1e-3


def kN_to_N(x: float) -> float:
    """
    Convert a force from kilonewtons to Newtons.

    Parameters
    ----------
    x : float
        Force [kN].

    Returns
    -------
    float
        Force [N].
    """
    return x * 1e3
