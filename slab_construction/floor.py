"""
Floor layer and floor construction classes for slab construction modeling.

All thicknesses are in [mm] to remain consistent with the structuralcodes
unit system (N, mm).

Author: Elliot Melcer
"""
from dataclasses import dataclass

from structuralcodes.core.base import Material


@dataclass(slots=True)
class FloorLayer:
    """
    A single floor layer defined by its material and thickness.

    Attributes
    ----------
    material : Material
        Material object for this layer.
    thickness : float
        Layer thickness [mm].
    """

    material: Material
    thickness: float

    def area_density_kg_m2(self) -> float:
        """
        Compute the area density of the layer.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Area density [kg/m²].
        """
        density_kg_m3 = self.material.density
        return density_kg_m3 * self.thickness / 1000


class FloorMaterial(Material):
    """
    Generic floor layer material.

    Parameters
    ----------
    density : float
        Material density [kg/m³].
    name : str or None, optional
        Material name. Default is ``"FloorMaterial"``.
    """

    def __init__(self, density: float, name: str | None = "FloorMaterial") -> None:
        super().__init__(density=density, name=name)


class ScreedMaterial(Material):
    """
    Screed layer material.

    Parameters
    ----------
    density : float
        Material density [kg/m³].
    name : str or None, optional
        Material name. Default is ``"ScreedMaterial"``.
    """

    def __init__(self, density: float, name: str | None = "ScreedMaterial") -> None:
        super().__init__(density=density, name=name)


class InfillMaterial(Material):
    """
    Infill layer material.

    Parameters
    ----------
    density : float
        Material density [kg/m³].
    name : str or None, optional
        Material name. Default is ``"InfillMaterial"``.
    """

    def __init__(self, density: float, name: str | None = "InfillMaterial") -> None:
        super().__init__(density=density, name=name)


class InsulationMaterial(Material):
    """
    Sound insulation layer material.

    Attributes
    ----------
    E_dyn : float
        Dynamic modulus of elasticity used for resonant frequency
        calculations [MN/m³].

    Parameters
    ----------
    density : float
        Material density [kg/m³].
    E_dyn : float
        Dynamic stiffness [MN/m³].
    name : str or None, optional
        Material name. Default is ``"InsulationMaterial"``.
    """

    def __init__(self, density: float, E_dyn: float, name: str | None = "InsulationMaterial") -> None:
        super().__init__(density=density, name=name)
        self.E_dyn = E_dyn # MN/m³


class Floor:
    """
    Floor construction composed of ordered :class:`FloorLayer` objects.

    Layers are stored in the order they are added. At most one layer per
    material type is permitted.

    Attributes
    ----------
    layers : list[FloorLayer]
        Ordered list of floor layers.
    """

    def __init__(self, layers: list[FloorLayer] | None = None) -> None:
        """
        Parameters
        ----------
        layers : list[FloorLayer] or None, optional
            Initial list of floor layers. Default is ``None`` (empty floor).
        """
        self.layers: list[FloorLayer] = list(layers) if layers is not None else []

    def add_layer(self, material: Material, thickness: float) -> None:
        """
        Add a layer to the floor construction.

        Parameters
        ----------
        material : Material
            Material for the new layer.
        thickness : float
            Layer thickness [mm]. Must be positive.

        Raises
        ------
        ValueError
            If ``thickness`` is not positive, or if a layer with the same
            material type already exists in the floor.
        """
        if thickness <= 0:
            raise ValueError("Thickness must be positive")

        material_type = type(material)
        if any(isinstance(layer.material, material_type) for layer in self.layers):
            raise ValueError(
                f"A layer with material type '{material_type.__name__}' already exists in this floor. "
                f"Only one layer per material type is permitted."
            )

        self.layers.append(FloorLayer(material, thickness))

    def dead_load(self) -> float:
        """
        Compute the total dead load of all floor layers.

        Returns
        -------
        float
            Total floor dead load [kN/m²].
        """
        floor_dead_load = sum(
            layer.material.density * layer.thickness * 1e-5
            for layer in self.layers
        )

        return floor_dead_load

    def get_layer_by_type(self, material_type: type) -> FloorLayer:
        """
        Return the floor layer whose material is an instance of
        ``material_type``.

        Parameters
        ----------
        material_type : type
            A :class:`Material` subclass, e.g. :class:`InsulationMaterial`,
            :class:`ScreedMaterial`, :class:`InfillMaterial`,
            :class:`FloorMaterial`.

        Returns
        -------
        FloorLayer
            The matching floor layer.

        Raises
        ------
        KeyError
            If no layer with the requested material type exists.
        """
        for layer in self.layers:
            if isinstance(layer.material, material_type):
                return layer
        raise KeyError(
            f"No floor layer with material type '{material_type.__name__}' found."
        )