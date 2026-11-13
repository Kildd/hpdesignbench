"""
Material functions and custom constitutive law classes for SLS/ULS
concrete and CFRP reinforcement, used in the structural benchmarking suite.

Author: Elliot Melcer
"""

from typing import Union

import numpy as np
from structuralcodes.materials.concrete import Concrete, create_concrete
from structuralcodes.materials.constitutive_laws import Sargin, UserDefined, Elastic
from structuralcodes.materials.reinforcement import Reinforcement, create_reinforcement

from slab_construction.floor import InsulationMaterial, InfillMaterial, ScreedMaterial

class CrackingConcreteLawEC(UserDefined):
    """
    Concrete constitutive law that handles tensile cracking in moment-curvature
    analysis.

    Overrides :meth:`get_ultimate_strain` of :class:`UserDefined` so that the
    tensile strain limit is set to ``1e6`` (effectively infinite). This prevents
    the concrete cracking inside ``get_balanced_failure_strain()`` from
    producing the governing curvature; ``chi_ultimate`` is therefore
    controlled by the two failure modes reinforcement fracture vs.
    concrete crushing.

    Attributes
    ----------
    _eps_cu1 : float
        Ultimate compressive strain (negative) [-].
    _eps_c1 : float
        Compressive strain at peak stress (negative), used when
        ``yielding=True`` [-].
    """

    def __init__(
            self,
            strains: np.ndarray,
            stresses: np.ndarray,
            eps_cu1: float,
            eps_c1: float,
            **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        strains : np.ndarray
            Strain values passed to :class:`UserDefined` [-].
        stresses : np.ndarray
            Stress values passed to :class:`UserDefined` [N/mm²].
        eps_cu1 : float
            Ultimate compressive strain (negative, e.g. ``-0.0035``) [-].
        eps_c1 : float
            Compressive strain at peak stress (negative, e.g. ``-0.0020``),
            used when ``yielding=True`` [-].
        **kwargs
            Additional keyword arguments forwarded to :class:`UserDefined`
            (e.g. ``name``, ``flag``).
        """
        super().__init__(strains, stresses, **kwargs)
        self._eps_cu1 = eps_cu1  # negative, e.g. -0.0035
        self._eps_c1 = eps_c1  # negative, e.g. -0.0020

    # ------------------------------------------------------------------
    def get_ultimate_strain(self, yielding: bool = False):
        """
        Return the compressive and tensile strain limits.

        The concrete tensile limit is fixed at ``1e6`` to ensure it is never the
        governing strain in ``get_balanced_failure_strain()``. The
        compressive limit depends on whether the yielding state is requested.

        Parameters
        ----------
        yielding : bool, optional
            If ``True``, return the strain at peak stress (``eps_c1``) as
            the compressive limit; otherwise return the ultimate compressive
            strain (``eps_cu1``). Default is ``False``.

        Returns
        -------
        tuple[float, float]
            ``(compressive_limit, 1e6)`` where ``compressive_limit`` is
            either ``eps_c1`` or ``eps_cu1`` depending on ``yielding`` [-].
        """
        if yielding:
            return self._eps_c1, 1e6
        return self._eps_cu1, 1e6

class TensionStiffeningConcreteLawEC(CrackingConcreteLawEC):
    """
    Concrete constitutive law that incorporates tension stiffening according
    to Nayal and Rasheed :cite:`nayal_2006`.

    Extends :class:`CrackingConcreteLawEC` by storing the two characteristic
    tension-stiffening strain values (``eps_F_t`` and ``eps_S_t``) directly
    on the instance for downstream access.

    Attributes
    ----------
    eps_F_t : float
        Final tension-stiffening strain (last entry of the strain array) :cite:`nayal_2006`.
    eps_S_t : float
        Secondary tension-stiffening strain (second-to-last entry of the
        strain array) :cite:`nayal_2006`.
    """

    def __init__(
            self,
            strains: np.ndarray,
            stresses: np.ndarray,
            eps_cu1: float,
            eps_c1: float,
            **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        strains : np.ndarray
            Strain values defining the constitutive curve [-].
        stresses : np.ndarray
            Stress values corresponding to ``strains`` [N/mm²].
        eps_cu1 : float
            Ultimate compressive strain (negative) [-].
        eps_c1 : float
            Compressive strain at peak stress (negative), used when
            ``yielding=True`` [-].
        **kwargs
            Additional keyword arguments forwarded to
            :class:`CrackingConcreteLawEC` (e.g. ``name``, ``flag``).
        """
        super().__init__(strains, stresses, eps_cu1, eps_c1, **kwargs)
        self.eps_F_t = strains[-1]
        self.eps_S_t = strains[-2]

def create_sls_concrete_EC(conc: Union[Concrete, float, int], constitutive_law: str) -> Concrete:
    """
    Create an SLS :class:`Concrete` object with constitutive law
    according to :cite:`ec2`.

    Parameters
    ----------
    conc : Concrete or float or int
        Either a :class:`Concrete` instance or a characteristic cylinder
        strength ``fck`` [N/mm²].
    constitutive_law : str
        Keyword selecting the tension/compression law pair. Case- and
        separator-insensitive. Available options:

        - ``"NONE_PARABOLIC"`` — no tensile capacity; Sargin compression.
        - ``"FCTM_PARABOLIC"`` — linear elastic tension up to ``fctm``;
          Sargin compression.
        - ``"TENSTIFF_PARABOLIC"`` — linear elastic tension up to ``fctm``
          followed by tension stiffening per Nayal and Rasheed :cite:`nayal_2006`; Sargin compression.
        - ``"ELASTIC_ELASTIC"`` — fully elastic in both tension and
          compression (no cracking or crushing).

    Returns
    -------
    Concrete
        A :class:`Concrete` instance configured for SLS analysis.

    Raises
    ------
    ValueError
        If ``constitutive_law`` does not match any of the available keywords.
    """
    f_ck = conc.fck if isinstance(conc, Concrete) else float(conc)
    f_cube = get_cube_EC(f_ck)

    # Normalize Input
    constitutive_law_normalized = constitutive_law.strip().upper().replace("-", "_").replace(" ", "_")

    # If Concrete should be able to take tension forces, use custom constitutive law (linear in tension and non-linear in compression)
    if constitutive_law_normalized == "NONE_PARABOLIC":
        sls_conc = create_concrete(fck=f_ck,
                                       constitutive_law='sargin',
                                       name=f"C{f_ck}/{f_cube} SLS")
    elif constitutive_law_normalized == "FCTM_PARABOLIC":
        sls_conc = create_concrete(fck=f_ck,
                                   constitutive_law=fctm_parabolic_law_EC(conc),
                                   name=f"C{f_ck}/{f_cube} SLS")
    elif constitutive_law_normalized == "TENSTIFF_PARABOLIC":
        sls_conc = create_concrete(fck=f_ck,
                                   constitutive_law=tenstiff_parabolic_law_EC(conc),
                                   name=f"C{f_ck}/{f_cube} SLS")
    elif constitutive_law_normalized == "ELASTIC_ELASTIC":
        sls_conc = create_concrete(fck=f_ck,
                                       constitutive_law='elastic',
                                       name=f"C{f_ck}/{f_cube} SLS")
    else:
        raise ValueError(
            f"{constitutive_law_normalized} is not a valid constitutive law. Available types (Tension / Compression):\n"
            f"     NONE_PARABOLIC:     Can't take tension, compression parabolic acc. to Sargin\n"
            f"     FCTM_PARABOLIC:     Linear elastic until fctm in tension, compression parabolic acc. to Sargin\n"
            f"     TENSTIFF_PARABOLIC: Linear elastic until fctm, then tension stiffening*, compression parabolic acc. to Sargin\n"
            f"     ELASTIC_ELASTIC:    Elastic in tension and compression without cracking or breaking")

    return sls_conc

def fctm_parabolic_law_EC(concrete: Union[Concrete, float, int], n_c: int = 80, n_t: int = 20) -> CrackingConcreteLawEC:
    """
    Build a constitutive law with linear tension and Sargin
    compression in accordance with Eurocode 2 :cite:`ec2`.

    The tension branch is linear elastic from zero to ``eps_ctm`` (the mean
    cracking strain); above ``eps_ctm`` the stress drops to zero. The
    compression branch follows the Sargin parabola from ``eps_cu1`` to
    zero.

    Parameters
    ----------
    concrete : Concrete or float or int
        Either a :class:`Concrete` instance or a characteristic cylinder
        strength ``fck`` [N/mm²].
    n_c : int, optional
        Number of discretization points on the compression branch.
        Default is ``80``.
    n_t : int, optional
        Number of discretization points on the linear tension branch.
        Default is ``20``.

    Returns
    -------
    CrackingConcreteLawEC
        A :class:`CrackingConcreteLawEC` instance with the assembled
        strain/stress arrays.
    """

    if not isinstance(concrete, Concrete):
        f_ck = float(concrete)
        _concrete = create_concrete(fck=f_ck, constitutive_law='sargin') #temporary Concrete object, constitutive_law is irrelevant
    else:
        _concrete = concrete

    # -------------------------------
    # Read parameters
    # -------------------------------
    fcm = _concrete.fcm
    Ecm = _concrete.Ecm
    fctm = _concrete.fctm

    eps_c1 = -abs(_concrete.eps_c1)
    eps_cu1 = -abs(_concrete.eps_cu1)
    k = _concrete.k_sargin

    eps_ctm = fctm / Ecm

    # -------------------------------
    # Sargin compression
    # -------------------------------
    sargin = Sargin(
        fc=fcm,
        eps_c1=eps_c1,
        eps_cu1=eps_cu1,
        k=k,
    )

    # ⚠ exclude zero explicitly
    eps_c = np.linspace(eps_cu1, 0.00, n_c, endpoint=True)
    eps_c = eps_c[eps_c < 0.0]
    sig_c = sargin.get_stress(eps_c)

    # -------------------------------
    # Elastic tension
    # -------------------------------
    eps_t = np.linspace(0.0, eps_ctm, n_t, endpoint=True)
    sig_t = Ecm * eps_t

    # -------------------------------
    # Merge safely
    # -------------------------------
    eps = np.concatenate((eps_c, eps_t))
    sig = np.concatenate((sig_c, sig_t))

    # -------------------------------
    # Final safety check (important)
    # -------------------------------
    eps, unique_idx = np.unique(eps, return_index=True)
    sig = sig[unique_idx]

    return CrackingConcreteLawEC(
        strains=eps,
        stresses=sig,
        eps_cu1=eps_cu1,
        eps_c1 = eps_c1,
        name="CrackingConcreteLawEC",
        flag=0,
    )

def tenstiff_parabolic_law_EC(concrete: Union[Concrete, float, int], n_c: int = 80, n_t: int = 20) -> TensionStiffeningConcreteLawEC:
    """
    Build a tension-stiffening constitutive law with Sargin compression
    according to Eurocode 2 :cite:`ec2` + Nayal and Rasheed :cite:`nayal_2006`.

    The compression branch follows the Sargin parabola. The tension branch
    is linear elastic up to ``fctm``, then descends through the
    tension-stiffening model defined by the fixed parameters
    ``P_t = 0.8``, ``R_t = 0.45``, ``S_t = 4``, ``F_t = 10`` from :cite:`nayal_2006`. This is an
    assumption that is yet to be verified experimentally for CRFP concrete sections

    Parameters
    ----------
    concrete : Concrete or float or int
        Either a :class:`Concrete` instance or a characteristic cylinder
        strength ``fck`` [N/mm²].
    n_c : int, optional
        Number of discretization points on the compression branch.
        Default is ``80``.
    n_t : int, optional
        Number of discretization points on the linear tension branch.
        Default is ``20``.

    Returns
    -------
    TensionStiffeningConcreteLawEC
        A :class:`TensionStiffeningConcreteLawEC` instance with the assembled
        strain/stress arrays.
    """

    if not isinstance(concrete, Concrete):
        f_ck = float(concrete)
        _concrete = create_concrete(fck=f_ck, constitutive_law='sargin') #temporary Concrete object, constitutive_law is irrelevant
    else:
        _concrete = concrete

    # -------------------------------
    # Read parameters
    # -------------------------------
    fcm = _concrete.fcm
    Ecm = _concrete.Ecm
    fctm = _concrete.fctm

    eps_c1 = -abs(_concrete.eps_c1)
    eps_cu1 = -abs(_concrete.eps_cu1)
    k = _concrete.k_sargin

    eps_ctm = fctm / Ecm

    # -------------------------------
    # Sargin compression
    # -------------------------------
    sargin = Sargin(
        fc=fcm,
        eps_c1=eps_c1,
        eps_cu1=eps_cu1,
        k=k,
    )

    # ⚠ exclude zero explicitly
    eps_c = np.linspace(eps_cu1, 0.00, n_c, endpoint=True)
    eps_c = eps_c[eps_c < 0.0]
    sig_c = sargin.get_stress(eps_c)

    # -------------------------------
    # Elastic tension
    # -------------------------------
    eps_t = np.linspace(0.0, eps_ctm, 2, endpoint=True) # dim = 1
    sig_t = Ecm * eps_t # dim = 1

    # -------------------------------
    # Tension Stiffening Factors according to Nayal and Rasheed :cite:`nayal_2006`
    # -------------------------------

    # Parameters
    P_t = 0.8
    R_t = 0.45
    S_t = 4
    F_t = 10

    eps_P_t = eps_ctm + 0.000_001 # dim = 0
    eps_S_t = S_t * eps_ctm # dim = 0
    eps_F_t = F_t * eps_ctm

    sig_P_t = P_t * sig_t[-1] # dim = 0
    sig_R_t = R_t * sig_t[-1] # dim = 0


    # -------------------------------
    # Merge strains and stresses
    # -------------------------------
    eps = np.concatenate((eps_c, eps_t, [eps_P_t], [eps_S_t], [eps_F_t]))
    sig = np.concatenate((sig_c, sig_t, [sig_P_t], [sig_R_t], [0.0]))

    # -------------------------------
    # Final safety check (important)
    # -------------------------------
    eps, unique_idx = np.unique(eps, return_index=True)
    sig = sig[unique_idx]

    return TensionStiffeningConcreteLawEC(
        strains=eps,
        stresses=sig,
        eps_cu1=eps_cu1,
        eps_c1=eps_c1,
        name="TensionStiffeningConcreteLawEC",
        flag=0,
    )

def get_cube_EC(cylinder_strength) -> int:
    """
    Return the cube compressive strength for a given cylinder strength per EN 206.

    Parameters
    ----------
    cylinder_strength : float or int
        Characteristic cylinder compressive strength ``fck`` [N/mm²].
        Must be a key in the EN 206 concrete class table.

    Returns
    -------
    int
        Corresponding cube compressive strength ``fck,cube`` [N/mm²].

    Raises
    ------
    ValueError
        If ``cylinder_strength`` is not listed in the EN 206 table.
    """
    table = {
        12.: 15,
        16.: 20,
        20.: 25,
        25.: 30,
        30.: 37,
        35.: 45,
        40.: 50,
        45.: 55,
        50.: 60,
        55.: 67,
        60.: 75,
        70: 85,
        80: 95,
        90: 105,
        100: 115,
    }

    if cylinder_strength not in table:
        raise ValueError("Cylinder strength not in EN 206 table")

    return table[cylinder_strength]


def get_cfrp_reinforcement_from_registry(
        mat_id: str,
        materials: dict,
        prestress_percent: float = 0.0,
        limit_state: str = "ULS",
):
    """
    Create a CFRP reinforcement object from a materials registry.

    CFRP is treated as a brittle-elastic material (no yielding plateau);
    the constitutive law is therefore a linear :class:`Elastic` law truncated
    at the relevant ultimate strain.

    Limit states
    ------------
    - ``"SLS"`` — characteristic values: ``Elastic(E, eps_u=eps_uk)``.
    - ``"ULS"`` — design values
      ``f_td = α_tex · f_tk / γ_tex``,
      ``eps_ud = α_tex · eps_uk / γ_tex``,
      with ``α_tex`` and ``γ_tex`` taken from ``materials.csv``.
      Characteristic ``fyk`` / ``ftk`` / ``epsuk`` are still stored on the
      reinforcement object; only the constitutive law uses design limits.

    Prestress ``initial_strain`` is always ``prestress_percent/100 · eps_uk``
    (characteristic), independent of limit state.

    Parameters
    ----------
    mat_id : str
        Material identifier key in the registry
        (e.g. ``"solidian GRID Q142/142-CCE-25"``).
    materials : dict
        Materials registry dictionary, typically loaded from a CSV file.
        Each reinforcement entry must contain ``type``, ``f_yk``, ``f_tk``,
        ``E_tex``, ``eps_u``, ``density``, ``alpha_tex``, and ``gamma_tex``.
    prestress_percent : float, optional
        Prestress level as a percentage of the ultimate strain (0–100).
        ``0.0`` means no prestress; ``50.0`` means prestressed to 50 % of
        ``epsuk``. Default is ``0.0`` [%].
    limit_state : {"SLS", "ULS"}, optional
        Constitutive-law limit state. Default is ``"ULS"``.

    Returns
    -------
    Reinforcement
        A :class:`Reinforcement` instance configured with the CFRP elastic
        constitutive law and the computed initial strain.

    Raises
    ------
    KeyError
        If ``mat_id`` is not found in ``materials``.
    ValueError
        If the material type is not ``"reinforcement"``, if any required
        property is ``None``, or if ``limit_state`` is unknown.
    """
    if mat_id not in materials:
        raise KeyError(
            f"Material '{mat_id}' not found in materials registry. "
            f"Available reinforcement: {[k for k, v in materials.items() if v['type'] == 'reinforcement']}"
        )

    mat = materials[mat_id]

    if mat["type"] != "reinforcement":
        raise ValueError(
            f"Material '{mat_id}' has type '{mat['type']}', expected 'reinforcement'"
        )

    limit_state_norm = limit_state.strip().upper()
    if limit_state_norm not in {"SLS", "ULS"}:
        raise ValueError(
            f"limit_state must be 'SLS' or 'ULS', got {limit_state!r}"
        )

    # Extract properties
    fyk = mat["f_yk"]  # MPa
    ftk = mat["f_tk"]  # MPa
    Es = mat["E_tex"]  # MPa
    epsuk = mat["eps_u"] / 1000  # Convert ‰ to absolute strain
    density = mat["density"]  # kg/m³
    alpha_tex = mat.get("alpha_tex")
    gamma_tex = mat.get("gamma_tex")

    # Validate required properties exist
    required = {
        "f_yk": fyk,
        "f_tk": ftk,
        "E_tex": Es,
        "eps_u": mat["eps_u"],
        "density": density,
        "alpha_tex": alpha_tex,
        "gamma_tex": gamma_tex,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            f"Material '{mat_id}' has missing required properties: {missing}"
        )
    if gamma_tex <= 0:
        raise ValueError(
            f"Material '{mat_id}': gamma_tex must be > 0, got {gamma_tex}"
        )
    if alpha_tex <= 0:
        raise ValueError(
            f"Material '{mat_id}': alpha_tex must be > 0, got {alpha_tex}"
        )

    # Prestress always relative to characteristic ultimate strain
    initial_strain = (prestress_percent / 100.0) * epsuk

    if limit_state_norm == "ULS":
        eps_u_law = alpha_tex * epsuk / gamma_tex
        gamma_s = gamma_tex
        name = f"{mat_id}, prestressed {prestress_percent}% (ULS)"
    else:
        eps_u_law = epsuk
        gamma_s = 1.0
        name = f"{mat_id}, prestressed {prestress_percent}% (SLS)"

    britte_elastic_const_law = Elastic(Es, eps_u=eps_u_law)

    reinforcement = create_reinforcement(
        fyk=fyk,
        Es=Es,
        ftk=ftk,
        epsuk=epsuk,
        density=density,
        constitutive_law=britte_elastic_const_law,
        initial_strain=initial_strain,
        gamma_s=gamma_s,
        name=name,
    )

    return reinforcement


def create_sls_cfrp_from_reinforcement(reinf: Reinforcement):
    """Rebuild CFRP with a characteristic (SLS) elastic constitutive law.

    Keeps ``fyk``, ``ftk``, ``epsuk``, ``Es``, density, and prestress
    ``initial_strain`` from ``reinf``. Sets ``gamma_s=1.0`` and truncates
    the elastic law at characteristic ``epsuk``.

    Parameters
    ----------
    reinf : Reinforcement
        Existing reinforcement (typically ULS design CFRP).

    Returns
    -------
    Reinforcement
        Characteristic SLS CFRP instance.
    """
    initial_strain = reinf.initial_strain
    return create_reinforcement(
        fyk=reinf.fyk,
        Es=reinf.Es,
        ftk=reinf.ftk,
        epsuk=reinf.epsuk,
        density=reinf.density,
        constitutive_law=Elastic(reinf.Es, eps_u=reinf.epsuk),
        initial_strain=initial_strain,
        gamma_s=1.0,
        name=f"{reinf.name} (SLS)" if reinf.name else "CFRP (SLS)",
    )


def get_floor_material_from_registry(mat_id: str, materials: dict):
    """
    Create a floor material object from a materials registry.

    Dispatches to :class:`InfillMaterial`, :class:`ScreedMaterial`, or
    :class:`InsulationMaterial` depending on the ``type`` field in the
    registry entry.

    Parameters
    ----------
    mat_id : str
        Material identifier key in the registry.
    materials : dict
        Materials registry dictionary. Each entry must contain ``type``
        and ``density``; insulation entries additionally require ``E_dyn``.

    Returns
    -------
    InfillMaterial or ScreedMaterial or InsulationMaterial
        The corresponding floor material instance.

    Raises
    ------
    KeyError
        If ``mat_id`` is not found in ``materials``.
    ValueError
        If the material type is not one of ``"infill"``, ``"insulation"``,
        or ``"screed"``; or if a required property is missing.
    """
    if mat_id not in materials:
        raise KeyError(
            f"Material '{mat_id}' not found in materials registry. "
            f"Available floor materials: {[k for k, v in materials.items() if v['type'] in ['infill', 'insulation', 'screed']]}"
        )

    mat = materials[mat_id]
    expected_types = ["infill", "insulation", "screed"]

    if mat["type"] not in expected_types:
        raise ValueError(
            f"Material '{mat_id}' has type '{mat['type']}', "
            f"expected one of {expected_types}"
        )

    density = mat["density"]
    if density is None:
        raise ValueError(
            f"Material '{mat_id}' missing density"
        )
    density = float(density)

    if mat["type"] == "infill":
        return InfillMaterial(density=density, name=mat_id)
    elif mat["type"] == "screed":
        return ScreedMaterial(density=density, name=mat_id)
    else:
        E_dyn = mat["E_dyn"]
        if E_dyn is None:
            raise ValueError(
                f"Material '{mat_id}' missing E_dyn"
            )

        return InsulationMaterial(density=density, E_dyn=float(E_dyn), name=mat_id)

def get_material_properties(mat_id: str, materials: dict) -> dict:
    """
    Return raw material properties from the registry.

    Useful for accessing non-structural properties (e.g. global warming
    potential, cost) that are needed by the objective function but not by
    the structural analysis routines.

    Parameters
    ----------
    mat_id : str
        Material identifier key in the registry.
    materials : dict
        Materials registry dictionary.

    Returns
    -------
    dict
        Raw properties dictionary for ``mat_id`` as stored in the registry.
        Typical keys include:

        - ``"type"`` — material category string.
        - ``"gwp"`` / ``"cost"`` — per-kg intensities (reinforcement/finishes).
        - ``"gwp_m3"`` / ``"cost_m3"`` — per-m³ intensities (concrete).
        - ``"density"`` — bulk density [kg/m³].
        - ``"alpha_tex"``, ``"gamma_tex"``, ``"f_ck"``, … — type-specific
          structural / environmental fields as present in ``materials.csv``.

    Raises
    ------
    KeyError
        If ``mat_id`` is not found in ``materials``.
    """
    if mat_id not in materials:
        raise KeyError(
            f"Material '{mat_id}' not found in materials registry. "
            f"Available: {list(materials.keys())}"
        )

    return materials[mat_id]