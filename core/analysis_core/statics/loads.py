"""
Load classes for Eurocode-based structural analysis of HP-shell slabs.

Provides an abstract :class:`Loads` base and the instantiable :class:`LoadsEC`
implementation for EN 1990 / EN 1991 load combinations with the German
National Annex (NA/DE). :cite:`ec1`

Author: Elliot Melcer
"""
from abc import abstractmethod, ABC
from enum import Enum
from typing import Callable

import numpy as np

from core.unit_core import mm_to_m
from slab_construction.slab_construction import SlabConstruction

# ── Abstract Base ────────────────────────────────────────────────────────────

class Loads(ABC):
    """
    Abstract base class for load objects.

    Defines the interface that all load implementations must satisfy so
    that structural analysis routines can remain independent of the
    underlying load model.
    """

    @property
    @abstractmethod
    def combinations_enum(self) -> type[Enum]:
        """
        Return the :class:`~enum.Enum` class that lists valid load combination names.

        Returns
        -------
        type[Enum]
            Enum class whose members name the supported combinations.
        """
        ...

    @property
    @abstractmethod
    def live_loads(self) -> np.ndarray:
        """
        Return the array of characteristic live load intensities.

        Returns
        -------
        np.ndarray
            Characteristic live load values Q_k [kN/m²].
        """
        ...

    @abstractmethod
    def combined_line_load_kN_m(self, slab_construction: SlabConstruction, combination: str) -> float:
        """
        Return the combined line load for a given load combination.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object providing geometry and dead loads.
        combination : str
            Load combination name (validated against :attr:`combinations_enum`).

        Returns
        -------
        float
            Combined line load [kN/m].
        """
        ...

    def check_valid_combination(self, combination: str) -> str:
        """
        Validate and normalize a load combination string.

        Parameters
        ----------
        combination : str
            Raw combination name (case- and separator-insensitive).

        Returns
        -------
        str
            Normalized combination name matching a member of
            :attr:`combinations_enum`.

        Raises
        ------
        ValueError
            If the normalized name is not a member of :attr:`combinations_enum`.
        """
        normalised = combination.strip().upper().replace("-", "_").replace(" ", "_")
        valid = {member.name for member in self.combinations_enum}
        if normalised not in valid:
            raise ValueError(
                f"Invalid combination '{combination}'. "
                f"Must be one of: {sorted(valid)}"
            )
        return normalised


class LoadsEC(Loads):
    """
    Load object for EN 1990 / EN 1991 combinations with German National Annex :cite:`ec1`

    Supports uniformly distributed loads applied over all spans. Multiple
    live load components (e.g. imposed + partitions) can be passed as arrays;
    the leading variable action always occupies index 0.

    Attributes
    ----------
    Qk : np.ndarray
        Characteristic live load intensities [kN/m²].
    psi_0_values : np.ndarray
        Combination factors ψ₀ for each live load component [-].
    psi_1_values : np.ndarray
        Combination factors ψ₁ for each live load component [-].
    psi_2_values : np.ndarray
        Combination factors ψ₂ for each live load component [-].
    gamma_g : float
        Partial safety factor for permanent actions [-].
    gamma_q : float
        Partial safety factor for variable actions [-].

    PSI_TABLE_EC0_2004_DE : dict
        Class-level lookup table mapping usage category codes (e.g. ``"A1"``)
        to their ψ-factors and Q_k values per EN 1990:2002/NA(DE) :cite:`ec0`.
    """

    class Combinations(Enum):
        """Supported EN 1990 load combination types."""

        FUNDAMENTAL = "FUNDAMENTAL"
        RARE = "RARE"
        FREQUENT = "FREQUENT"
        QUASI_PERMANENT = "QUASI_PERMANENT"

    @property
    def combinations_enum(self) -> type[Enum]:
        """
        Return the :class:`Combinations` enum for :class:`LoadsEC`.

        Returns
        -------
        type[Enum]
            :class:`LoadsEC.Combinations`.
        """
        return LoadsEC.Combinations

    def __init__(
        self,
        live_loads,
        psi_0_values,
        psi_1_values,
        psi_2_values,
        gamma_g = 1.35,
        gamma_q = 1.5,
    ):
        """
        Parameters
        ----------
        live_loads : array-like
            Characteristic live load intensities Q_k [kN/m²], one entry
            per load component.
        psi_0_values : array-like
            Combination factors ψ₀ aligned with ``live_loads`` [-].
        psi_1_values : array-like
            Combination factors ψ₁ aligned with ``live_loads`` [-].
        psi_2_values : array-like
            Combination factors ψ₂ aligned with ``live_loads`` [-].
        gamma_g : float, optional
            Partial safety factor for permanent actions. Default is ``1.35`` [-].
        gamma_q : float, optional
            Partial safety factor for variable actions. Default is ``1.5`` [-].
        """
        self.Qk = np.array(live_loads, dtype=float)
        self.psi_0_values = np.array(psi_0_values, dtype=float)
        self.psi_1_values = np.array(psi_1_values, dtype=float)
        self.psi_2_values = np.array(psi_2_values, dtype=float)
        self.gamma_g = gamma_g
        self.gamma_q = gamma_q
        self._check_dimensions()

    # Format: "psi": (psi_0, psi_1, psi_2)
    # Qk in kN/m²
    PSI_TABLE_EC0_2004_DE = {
        "A": {"psi": (0.7, 0.5, 0.3), "Qk": {1: 1.0, 2: 1.5, 3: 2.0}},
        "B": {"psi": (0.7, 0.5, 0.3), "Qk": {1: 2.0, 2: 3.0, 3: 5.0}},
        "C": {"psi": (0.7, 0.7, 0.6), "Qk": {1: 3.0, 2: 4.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 7.5}},
        "D": {"psi": (0.7, 0.7, 0.6), "Qk": {1: 2.0, 2: 5.0, 3: 5.0}},
        "E": {"psi": (1.0, 0.9, 0.8), "Qk": {1: 5.0, 2: 6.0, 3: 7.5}},
    }

    @property
    def live_loads(self) -> np.ndarray:
        """
        Return the characteristic live load array.

        Returns
        -------
        np.ndarray
            Q_k values [kN/m²].
        """
        return self.Qk

    @classmethod
    def _parse_category(cls, _category: str):
        """
        Parse a usage category code and return its Q_k and ψ-factors.

        Parameters
        ----------
        _category : str
            Category code in the form ``"<letter><number>"``
            (e.g. ``"A1"``, ``"C3"``), case-insensitive.

        Returns
        -------
        tuple[float, float, float, float]
            ``(Q_k, psi_0, psi_1, psi_2)`` for the given category [kN/m², -, -, -].

        Raises
        ------
        ValueError
            If the letter or subcategory number is not in
            :attr:`PSI_TABLE_EC0_2004_DE`.
        """
        key = _category.upper().strip()
        letter, number = key[0], int(key[1:])
        if letter not in cls.PSI_TABLE_EC0_2004_DE:
            raise ValueError(f"Unknown category '{letter}'")
        if number not in cls.PSI_TABLE_EC0_2004_DE[letter]["Qk"]:
            raise ValueError(f"Unknown subcategory '{number}' for category '{letter}'")
        psi = cls.PSI_TABLE_EC0_2004_DE[letter]["psi"]
        Qk = cls.PSI_TABLE_EC0_2004_DE[letter]["Qk"][number]
        return Qk, psi[0], psi[1], psi[2]

    @classmethod
    def from_categories_EC0_NA_DE(cls, categories: str | list[str], gamma_g=1.35, gamma_q=1.5):
        """
        Construct a :class:`LoadsEC` instance from EN 1990 usage category codes
        per the German National Annex :cite:`ec0`.

        Parameters
        ----------
        categories : str or list[str]
            One or more category codes (e.g. ``"A1"``, ``["B2", "C3"]``).
            The first entry is treated as the leading variable action.
        gamma_g : float, optional
            Partial safety factor for permanent actions. Default is ``1.35`` [-].
        gamma_q : float, optional
            Partial safety factor for variable actions. Default is ``1.5`` [-].

        Returns
        -------
        LoadsEC
            A new :class:`LoadsEC` instance with Q_k and ψ-factor arrays
            populated from the lookup table.
        """
        # Normalize Input
        if isinstance(categories, str):
            categories = [categories]

        Qk_values, psi_0s, psi_1s, psi_2s = [], [], [], []

        for cat in categories:
            Qk, psi_0, psi_1, psi_2 = cls._parse_category(cat)
            Qk_values.append(Qk)
            psi_0s.append(psi_0)
            psi_1s.append(psi_1)
            psi_2s.append(psi_2)

        return cls(Qk_values, psi_0s, psi_1s, psi_2s, gamma_g, gamma_q)

    def _check_dimensions(self) -> None:
        """
        Verify that all load and ψ-factor arrays have the same length.

        Raises
        ------
        ValueError
            If any ψ-factor array length differs from that of :attr:`Qk`.
        """
        n = len(self.Qk)
        for arr in [self.psi_0_values, self.psi_1_values, self.psi_2_values]:
            if len(arr) != n:
                raise ValueError("All live load (Qk) and psi arrays must have the same length")

    def combined_line_load_kN_m(self, slab_construction: SlabConstruction, combination: str = "FUNDAMENTAL") -> float:
        """
        Compute the combined line load for a given EN 1990 :cite:`ec0` load combination.

        Converts the area load of the governing combination to a line load
        by multiplying by the slab width.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object (provides slab width and dead loads).
        combination : str, optional
            Load combination name. One of ``"FUNDAMENTAL"``, ``"RARE"``,
            ``"FREQUENT"``, ``"QUASI_PERMANENT"``. Default is
            ``"FUNDAMENTAL"``.

        Returns
        -------
        float
            Combined line load w [kN/m].
        """
        width_m = mm_to_m(slab_construction.slab.B)
        combination = combination.strip().upper()

        dispatch: dict[str, Callable[[SlabConstruction], float]] = {
            "FUNDAMENTAL": self.fundamental_combination_kN_m2_EC0,
            "RARE": self.rare_combination_kN_m2_EC0,
            "FREQUENT": self.frequent_combination_kN_m2_EC0,
            "QUASI_PERMANENT": self.quasi_permanent_combination_kN_m2_EC0,
        }

        area_load_kN_m2 = dispatch[combination](slab_construction)

        return area_load_kN_m2 * width_m

    def fundamental_combination_kN_m2_EC0(self, slab_construction: SlabConstruction):
        """
        Compute the ULS fundamental combination area load per EN 1990 Eq. (6.10) :cite:`ec0`.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.

        Returns
        -------
        float
            Design area load q_d [kN/m²].
        """
        Gd = self.gamma_g * (slab_construction.structural_dead_load_kN_m2()
                             + slab_construction.non_structural_dead_load_kN_m2())

        # Set up psi values for frequent combination
        # Only the accompanying actions are multiplied by psi_0
        psi_0_mask = self.psi_0_values.copy()
        psi_0_mask[0] = 1.0

        Qd = self.gamma_q * float(np.sum(self.Qk * psi_0_mask))

        return Gd + Qd

    def rare_combination_kN_m2_EC0(self, slab_construction: SlabConstruction):
        """
        Compute the SLS rare (characteristic) combination area load per EN 1990 Eq. (6.14b) :cite:`ec0`.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.

        Returns
        -------
        float
            Characteristic area load q [kN/m²].
        """
        # Set up psi values for frequent combination
        # Only the accompanying actions are multiplied by psi_0
        psi_mask = self.psi_0_values.copy()
        psi_mask[0] = 1.0

        return (slab_construction.structural_dead_load_kN_m2()
                + slab_construction.non_structural_dead_load_kN_m2()
                + float(np.sum(self.Qk * psi_mask)))

    def frequent_combination_kN_m2_EC0(self, slab_construction: SlabConstruction):
        """
        Compute the SLS frequent combination area load per EN 1990 Eq. (6.15b) :cite:`ec0`.

        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.

        Returns
        -------
        float
            Frequent area load q [kN/m²].
        """
        # Set up psi values for frequent combination
        # Leading variable action multiplied by psi_1, all accompanying actions multiplied by psi_2
        psi_mask = self.psi_2_values.copy()
        psi_mask[0] = self.psi_1_values[0]

        return (slab_construction.structural_dead_load_kN_m2()
                + slab_construction.non_structural_dead_load_kN_m2()
                + float(np.sum(self.Qk * psi_mask)))

    def quasi_permanent_combination_kN_m2_EC0(self, slab_construction: SlabConstruction):
        """
        Compute the SLS quasi-permanent combination area load per EN 1990 Eq. (6.16b) :cite:`ec0`.


        Parameters
        ----------
        slab_construction : SlabConstruction
            Full slab construction object.

        Returns
        -------
        float
            Quasi-permanent area load q [kN/m²].
        """
        return (slab_construction.structural_dead_load_kN_m2()
                + slab_construction.non_structural_dead_load_kN_m2()
                + float(np.sum(self.Qk * self.psi_2_values)))