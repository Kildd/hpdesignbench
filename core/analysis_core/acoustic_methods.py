"""
Acoustic calculation methods for HP-shell slab constructions.

Implements regression-based models for airborne sound reduction and
impact sound pressure level, calibrated on HP-shell geometry and floor
build-up parameters by Eddin :cite:`eddin_2023`. Adapted from Loutfi's
Grasshopper script :cite:`loutfi_2023`

Adapted from: Jamila Loutfi, Ahmad Eiz Eddin
"""
import math

from slab_construction.floor import InfillMaterial, InsulationMaterial, ScreedMaterial
from slab_construction.slab_construction import SlabConstruction


def calculate_sound_reduction_index(
        slab_construction: SlabConstruction,
        mod_att: float,
        debug: bool = False
) -> float:
    """
    Calculate the weighted sound reduction index R_w for a slab construction.

    Uses a regression model for the bare HP-shell (R_w,BG) combined with the
    floating-screed improvement ΔR per DIN 4109-34. Only direct sound
    transmission is considered; flanking paths are excluded.

    Parameters
    ----------
    slab_construction : SlabConstruction
        Full slab construction object containing HP-shell geometry, concrete
        properties, and floor layer stack.
    mod_att : float
        Modal attenuation parameter of the HP-shell [-].
    debug : bool, optional
        If ``True``, intermediate values are printed to console.
        Default is ``False``.

    Returns
    -------
    float
        Weighted sound reduction index R_w [dB].

    Notes
    -----
    Regression model for R_w,BG:

    .. math::

        R_{w,BG} = 41.61 + 727.32 \\cdot \\text{mod\\_{att}}
                   + 53.55 \\cdot t_{\\text{infill}}
                   + 1.51 \\cdot B - 11.52 \\cdot B \\cdot t

    Improvement ΔR is computed from the resonant frequency f_0 of the
    floating-screed system per DIN 4109-34, Eq. (1):

    .. math::

        f_0 = 160 \\sqrt{s \\cdot \\left(\\frac{1}{m'_1} + \\frac{1}{m'_2}\\right)} \\quad [Hz]

    where s is the dynamic stiffness of the insulation layer [MN/m³],
    m_{1} is the combined area density of concrete slab and infill [kg/m²],
    and m_{2} is the screed area density [kg/m²].
    """
    hp_shell = slab_construction.slab.hp_shell
    hp_geometry = hp_shell.hp_geometry
    floor = slab_construction.floor

    "Geometry"
    _B = hp_geometry.B / 1000 # [m]
    _t = hp_geometry.t / 1000 # [m]
    _t_infill = floor.get_layer_by_type(InfillMaterial).thickness / 1000 # [m]

    "Regression Model: Rw"
    regression_Rw = 41.61 + 727.32 * mod_att + 53.55 * _t_infill + 1.51 * _B - 11.52 * _B * _t

    insulation_layer = floor.get_layer_by_type(InsulationMaterial)
    s = insulation_layer.material.E_dyn / insulation_layer.thickness * 1000 # MN/m³

    m_concrete_slab = hp_shell.concrete.density * _t  # kg/m²
    m_infill = slab_construction.infill_area_density_kg_m2()                   #kg/m²
    m_screed = floor.get_layer_by_type(ScreedMaterial).area_density_kg_m2()    #kg/m²

    m1 = m_infill + m_concrete_slab
    m2 = m_screed

    "Resonant Frequency f0 according to DIN 4109-34 Eq. (1)"
    resonant_frequency_f0 = 160 * (s * (1/m1 + 1/m2) ) ** 0.5

    "Improvement of the Sound Reduction Index according to nach DIN 4109-34 Table 1"
    delta_R = (74.4 - 20 * math.log10(resonant_frequency_f0)) - (regression_Rw / 2)

    "Sound Reduction Index R_w"
    sound_reduction_index_R_w = regression_Rw + delta_R

    if debug:
        print(f"R_w,BG = {regression_Rw:.2f} \n")

        print(f"_E_dyn = {s:.2f}")
        print(f"m_concrete_slab = {m_concrete_slab:.2f}")
        print(f"m_infill = {m_infill:.2f}")
        print(f"m_screed = {m_screed:.2f}")
        print(f"Resonant Frequency F0 = {resonant_frequency_f0:.2f}")

        print(f"delta_R = {delta_R:.2f}")

    return sound_reduction_index_R_w

def calculate_standard_impact_sound_pressure_level(
        slab_construction: SlabConstruction,
        mod_att: float,
        debug: bool = False
) -> float:
    """
    Calculate the weighted standard impact sound pressure level L_nw for a
    slab construction.

    Uses a regression model for the equivalent impact sound pressure level
    L_n,eq,0 of the bare HP-shell concrete layer, then applies the floating-
    screed reduction ΔL_w per DIN 4109-34, Eq. (3). Flanking sound
    transmission is not considered (K = 0).

    Parameters
    ----------
    slab_construction : SlabConstruction
        Full slab construction object containing HP-shell geometry, concrete
        properties, and floor layer stack.
    mod_att : float
        Modal attenuation parameter of the HP-shell [-].
    debug : bool, optional
        If ``True``, intermediate values are printed to console.
        Default is ``False``.

    Returns
    -------
    float
        Weighted standard impact sound pressure level L_nw [dB].

    Notes
    -----
    Regression model for L_n,eq,0 (bare concrete layer):

    .. math::

        L_{n,eq,0} = 259.59 - 107.55 \\cdot t - 6606.69 \\cdot \\text{mod\\_att}
                     - 1.068 \\cdot B - 41.16 \\cdot t_{\\text{infill}}
                     - 6.35 \\cdot H_y - 0.0043 \\cdot E_{cm}
                     + 263.50 \\cdot t \\cdot t_{\\text{infill}}
                     + 0.167 \\cdot \\text{mod\\_att} \\cdot E_{cm}

    Floating-screed reduction per DIN 4109-34, Eq. (3):

    .. math::

        \\Delta L_w = 13 \\log_{10}(m_2) - 14.2 \\log_{10}(s') + 20.8

    where s' is the dynamic stiffness of the insulation layer [MN/m³] and
    m_2 is the screed area density [kg/m²].

    The correctional value for flanking sound transmission is set to K = 0
    (not considered in the current model).
    """
    hp_shell = slab_construction.slab.hp_shell
    hp_geometry = hp_shell.hp_geometry
    floor = slab_construction.floor

    "Geometry"
    _B = hp_geometry.B / 1000  # [m]
    _L = hp_geometry.L / 1000  # [m]
    _t = hp_geometry.t / 1000  # [m]
    _Hx = hp_geometry.Hx / 1000  # [m]
    _Hy = hp_geometry.Hy / 1000  # [m]
    _t_infill = floor.get_layer_by_type(InfillMaterial).thickness / 1000

    "Material"
    _Ecm = hp_shell.concrete.Ecm

    "Equivalent Impact Sound Pressure Level L_n_eq_0 of the Concrete Layer"
    regression_L_n_eq_0 = (259.5925
                           - 107.5515 * _t
                           - 6606.6942 * mod_att
                           - 1.068 * _B
                           - 41.1633 * _t_infill
                           - 6.3487 * _Hy
                           - 0.0043 * _Ecm
                           + 263.4989 * _t * _t_infill
                           + 0.1665 * mod_att * _Ecm)


    m_screed = floor.get_layer_by_type(ScreedMaterial).area_density_kg_m2()  # kg/m²
    m2 = m_screed

    insulation_layer = floor.get_layer_by_type(InsulationMaterial)
    s = insulation_layer.material.E_dyn / insulation_layer.thickness * 1000 # MN/m³

    "Weighted impact sound reduction according to DIN 4109-34 Eq.(3)"
    delta_L_w = 13 * math.log10(m2) - 14.2 * math.log10(s) + 20.8

    "Correctional Value K for Flanking Sound Transmission"
    K = 0.0    # not considered

    "Weighted Normalized Impact Sound Pressure Level"
    L_nw = regression_L_n_eq_0 - delta_L_w + K

    if debug:
        print(f"B = {_B}")
        print(f"L = {_L}")
        print(f"t = {_t}")
        print(f"Hx = {_Hx}")
        print(f"Hy = {_Hy}")
        print(f"t_infill = {_t_infill}")
        print(f"Ecm = {_Ecm}")

        print(f"L_n_eq = {regression_L_n_eq_0:.2f}")

        print(f"delta_L_w = {delta_L_w:.2f}")

        print(f"L_nw = {L_nw:.2f}")

    return L_nw