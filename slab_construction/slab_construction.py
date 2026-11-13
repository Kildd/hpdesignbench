"""
SlabConstruction class combining a load-bearing slab with floor finishing layers.

Author: Elliot Melcer
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.unit_core import mm2_to_m2, mm3_to_m3
from slab_construction.floor import Floor, InfillMaterial, InsulationMaterial
from slab_construction.slabs.hp_slab.hp_model.hp_slab import HPSlab
from slab_construction.slabs.slab import Slab


@dataclass(frozen=True)
class MKCurveKey:
    """Hashable key describing one M-κ curve generation configuration."""

    limit_state: str
    constitutive_law: str
    method: str
    simplification: bool | tuple[float, ...]
    softening: tuple[float, ...] | None
    n: float
    m_k_num_points: int


@dataclass(frozen=True)
class SectionResultKey:
    """Hashable key for cached scalar/dict section-analysis results."""

    result_type: str
    x: float
    limit_state: str
    n: float
    flipped: bool = False


@dataclass
class AnalysisCache:
    """Per-FE reusable analysis results attached to a :class:`SlabConstruction`.

    Cleared/rebuilt with every function evaluation. Holds reference sections,
    M-κ curves, and repeated section-analysis results such as ``M_cr`` and
    ``M_u``.
    """

    sections: dict[float, Any] = field(default_factory=dict)

    # General cache stores used by the next refactor steps.
    mk_curves: dict[MKCurveKey, dict[float, Any]] = field(default_factory=dict)
    mk_metadata: dict[MKCurveKey, dict[str, Any]] = field(default_factory=dict)
    m_cr: dict[SectionResultKey, dict[str, Any]] = field(default_factory=dict)
    m_u: dict[SectionResultKey, dict[str, Any]] = field(default_factory=dict)

    def clear(self) -> None:
        """Reset all cached results for a new function evaluation."""
        self.sections.clear()
        self.mk_curves.clear()
        self.mk_metadata.clear()
        self.m_cr.clear()
        self.m_u.clear()

    def require_section(self, x: float) -> Any:
        """Return a cached section or raise a clear cache-preparation error."""
        try:
            return self.sections[x]
        except KeyError as exc:
            raise RuntimeError(
                f"Missing cached section at x={x}. Prepare analysis_cache before checks."
            ) from exc

    def require_mk_curve(self, key: MKCurveKey, x: float) -> Any:
        """Return a cached M-κ curve or raise a clear cache-preparation error."""
        try:
            return self.mk_curves[key][x]
        except KeyError as exc:
            raise RuntimeError(
                "Missing cached M-κ curve. Prepare analysis_cache before "
                f"deflection checks (key={key!r}, x={x})."
            ) from exc

    def require_m_cr(self, key: SectionResultKey) -> dict[str, Any]:
        """Return a cached cracking moment result or raise a clear error."""
        try:
            return self.m_cr[key]
        except KeyError as exc:
            raise RuntimeError(
                "Missing cached M_cr result. Prepare analysis_cache before "
                f"cracking-moment checks (key={key!r})."
            ) from exc

    def require_m_u(self, key: SectionResultKey) -> dict[str, Any]:
        """Return a cached bending strength result or raise a clear error."""
        try:
            return self.m_u[key]
        except KeyError as exc:
            raise RuntimeError(
                "Missing cached M_u result. Prepare analysis_cache before "
                f"ultimate-moment checks (key={key!r})."
            ) from exc


class SlabConstruction:
    """
    Complete slab construction consisting of a load-bearing slab and floor
    finishing layers.

    Validates that the infill material used by the slab and the infill layer
    in the floor reference the same object on instantiation.

    Attributes
    ----------
    slab : Slab
        Load-bearing slab (e.g. :class:`HPSlab`).
    floor : Floor
        Floor finishing layer stack.
    analysis_cache : AnalysisCache
        Per-FE cache for sections, M-κ curves, and repeated structural
        section results. Empty on construction; populated once before checks.
    """

    def __init__(self, slab: Slab, floor: Floor):
        """
        Parameters
        ----------
        slab : Slab
            Load-bearing slab object.
        floor : Floor
            Floor finishing layer stack.

        Raises
        ------
        ValueError
            If the floor contains an infill layer whose material is not the
            same object as the slab's infill material (identity check).
        """
        self.slab = slab
        self.floor = floor
        self.analysis_cache = AnalysisCache()
        self.assert_infill_compatibility()

    def structural_dead_load_kN_m2(self) -> float:
        """
        Compute the structural dead load of the slab.

        Returns
        -------
        float
            Structural dead load [kN/m²].
        """
        return (
            self.slab.self_load()
        )

    def non_structural_dead_load_kN_m2(self) -> float:
        """
        Compute the non-structural dead load (infill + floor finishing).

        Returns
        -------
        float
            Non-structural dead load [kN/m²].
        """
        return (
            self.slab.infill_load()
            + self.floor.dead_load()
        )

    def assert_infill_compatibility(self) -> None:
        """
        Assert that the slab and floor share the same infill material object.

        Only relevant when :class:`SlabConstruction` is instantiated manually.
        The same :class:`InfillMaterial` instance must be passed to both the
        slab and the floor layer to avoid inconsistent density or name values.

        Raises
        ------
        ValueError
            If an infill layer exists in the floor but its material is not the
            same object (``is`` check) as the slab's infill material.
        """
        floor_infill_layers = [
            layer for layer in self.floor.layers
            if isinstance(layer.material, InfillMaterial)
        ]
        if not floor_infill_layers:
            return

        if self.slab.infill_load() == 0:
            return

        floor_infill = floor_infill_layers[0].material
        slab_infill = self.slab.infill_material

        if floor_infill is not slab_infill:
            raise ValueError(
                f"Infill material mismatch: the floor's infill layer and the slab "
                f"must reference the same object.\n"
                f"  Slab infill:  '{slab_infill.name}'  (id={id(slab_infill)})\n"
                f"  Floor infill: '{floor_infill.name}'  (id={id(floor_infill)})"
            )

    def infill_area_density_kg_m2(self) -> float:
        """
        Compute the total infill area density.

        For :class:`HPSlab`, the area density includes the minimum infill
        volume required to flatten the curved shell top surface plus any
        additional floor infill layer thickness. For other slab types only
        the floor layer contribution is used.

        Returns
        -------
        float
            Total infill area density [kg/m²].
        """
        infill_layer = self.floor.get_layer_by_type(InfillMaterial)

        # Reference Area
        if isinstance(self.slab, HPSlab):
            area_m2 = mm2_to_m2(self.slab.L * self.slab.B)
        else:
            area_m2 = 1.0

        # Infill Volume
        base_infill_volume_m3 = 0.0
            # base_infill_volume if hp_slab
        if isinstance(self.slab, HPSlab):
            base_infill_volume_m3 = mm3_to_m3(self.slab.minimum_infill_volume())

            # floor layer infill volume
        infill_layer_t_m = infill_layer.thickness / 1000
        floor_infill_volume_m3 = infill_layer_t_m * area_m2

        total_volume_m3 = base_infill_volume_m3 + floor_infill_volume_m3

        # Infill Material
        infill_mat = infill_layer.material
        infill_density_kg_m3 = infill_mat.density

        # infill area density
        infill_area_density_kg_m2 = total_volume_m3 * infill_density_kg_m3 / area_m2

        return infill_area_density_kg_m2

    def get_parameters(self) -> dict:
        """
        Return a dictionary of all slab construction parameters organized by
        category.

        Currently only implemented for :class:`HPSlab` with an
        :class:`HPShell`. For other slab types the geometry, concrete, and
        reinforcement sub-dicts will be empty.

        Returns
        -------
        dict
            Nested dictionary with the following structure:

            - ``"geometry"`` — dict with keys:

              - ``"span_L"`` — span length [mm].
              - ``"width_B"`` — width [mm].
              - ``"height"`` — total rise H_ges = H_x + H_y [mm].
              - ``"Hx_Hges"`` — rise ratio H_x / H_ges [-].
              - ``"thickness"`` — shell thickness t [mm].
              - ``"nt"`` — number of tendons per group [-].
              - ``"dy"`` — edge-tendon offset [mm].
              - ``"thickness_infill"`` — infill layer thickness [mm].
              - ``"thickness_screed"`` — screed layer thickness [mm].
              - ``"thickness_insulation"`` — insulation layer thickness [mm].

            - ``"concrete"`` — dict with keys:

              - ``"fck"`` — characteristic cylinder strength [N/mm²].

            - ``"reinforcement"`` — dict with keys:

              - ``"name"`` — reinforcement material name.
              - ``"initial_strain_percentage"`` — prestress level [%].
              - ``"cross_sectional_area"`` — bar area [mm²].
        """
        from slab_construction.slabs.hp_slab.hp_model.hp_slab import HPSlab

        # Initialize result dictionary
        params = {
            "geometry": {},
            "concrete": {},
            "reinforcement": {}
        }

        # --- Geometry Parameters ---
        if isinstance(self.slab, HPSlab):
            hp_shell = self.slab.hp_shell
            hp_geom = hp_shell.hp_geometry

            # Slab geometry
            H_ges = hp_geom.Hx + hp_geom.Hy

            params["geometry"] = {
                "span_L": hp_geom.L,  # mm
                "width_B": hp_geom.B,  # mm
                "height": H_ges,  # mm (Hx + Hy)
                "Hx_Hges": hp_geom.Hx / H_ges if H_ges > 0 else 0.0,  # ratio [-]
                "thickness": hp_geom.t,  # mm
                "nt": hp_geom.nt,  # number of tendons per group
                "dy": hp_geom.dy,  # mm
            }

            # --- Concrete Parameters ---
            concrete = hp_shell.concrete
            params["concrete"] = {
                "fck": concrete.fck  # MPa
            }

            # --- Reinforcement Parameters ---
            reinforcement = hp_shell.reinforcement

            # Get initial strain (prestress percentage)
            initial_strain = getattr(reinforcement, 'initial_strain', 0.0)
            if initial_strain is None:
                initial_strain = 0.0

            # Convert to percentage (strain is typically in absolute form)
            initial_strain_percentage = initial_strain * 100

            params["reinforcement"] = {
                "name": reinforcement.name if hasattr(reinforcement, 'name') else str(reinforcement),
                "initial_strain_percentage": initial_strain_percentage,  # %
                "cross_sectional_area": hp_shell.reinf_area  # mm²
            }

        # --- Floor Layer Thicknesses ---
        # Extract thicknesses from floor layers by material type
        thickness_infill = 0.0
        thickness_screed = 0.0
        thickness_insulation = 0.0

        for layer in self.floor.layers:
            material = layer.material
            material_name = getattr(material, 'name', '').lower() if hasattr(material, 'name') else ''

            # Check if it's an insulation material by type
            if isinstance(material, InsulationMaterial):
                thickness_insulation += layer.thickness
            # Otherwise check by name
            elif 'insulation' in material_name or 'insu' in material_name:
                thickness_insulation += layer.thickness
            elif 'screed' in material_name:
                thickness_screed += layer.thickness
            elif 'infill' in material_name:
                thickness_infill += layer.thickness

        params["geometry"]["thickness_infill"] = thickness_infill  # mm
        params["geometry"]["thickness_screed"] = thickness_screed  # mm
        params["geometry"]["thickness_insulation"] = thickness_insulation  # mm

        return params

    def print_parameters(self) -> None:
        """
        Print all slab construction parameters to console.

        Calls :meth:`get_parameters` and formats the result by category
        (geometry, concrete, reinforcement).
        """
        parameters = self.get_parameters()

        # Print results
        print("\nReturned parameters:")
        print("-" * 40)

        print("[GEOMETRY]")
        for key, value in parameters["geometry"].items():
            print(f"  {key}: {value}")

        print("\n[CONCRETE]")
        for key, value in parameters["concrete"].items():
            print(f"  {key}: {value}")

        print("\n[REINFORCEMENT]")
        for key, value in parameters["reinforcement"].items():
            print(f"  {key}: {value}")

        print("")