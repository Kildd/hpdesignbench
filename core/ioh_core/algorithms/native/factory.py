"""
Import-safe constructors for native (library-free) algorithms.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.native.discrete_search import (
    DiscreteMLSL,
    DiscreteMLSLDeb,
    DiscreteMLSLDebEquDesc,
    DiscreteMLSLDebEquDescWarmStart,
    DiscreteMLSLDebWarmStart,
    DiscreteMLSLEquDesc,
    DiscreteMLSLEquDescWarmStart,
    DiscreteMLSLWarmStart,
    PopulationBasinHopping,
    SobolMultistart,
    SobolMultistartEquDesc,
    SobolSample,
    PopulationBasinHoppingDeb,
    PopulationBasinHoppingDebEquDesc,
    PopulationBasinHoppingDebEquDescWarmStart,
    PopulationBasinHoppingDebWarmStart,
    PopulationBasinHoppingEquDesc,
    PopulationBasinHoppingEquDescWarmStart,
    PopulationBasinHoppingWarmStart,
)
from core.ioh_core.algorithms.native.k_elite_ams import (
    KEliteAMS,
    KEliteAMSDeb,
    KEliteAMSDebEquDesc,
    KEliteAMSDebEquDescWarmStart,
    KEliteAMSDebWarmStart,
    KEliteAMSEquDesc,
    KEliteAMSEquDescWarmStart,
    KEliteAMSWarmStart,
)
from core.ioh_core.algorithms.native.latin_hypercube import LatinHypercubeSampling
from core.ioh_core.algorithms.native.random_search import RandomSearch
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue


def build_random_search(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`RandomSearch` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters; must include ``max_evaluations``.
    random_seed : int or None
        Task seed forwarded to the RNG.

    Returns
    -------
    IOHAlgorithm
        New independent RandomSearch instance.
    """
    return RandomSearch(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=random_seed,
    )


def build_latin_hypercube(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`LatinHypercubeSampling` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters; must include ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent Latin-hypercube instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None``.
    """
    if random_seed is None:
        raise ValueError(
            "LATIN_HYPERCUBE requires a random_seed (unsigned 32-bit int)"
        )
    return LatinHypercubeSampling(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_discrete_mlsl(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`DiscreteMLSL` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters, including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed. A new generator is built per call.

    Returns
    -------
    IOHAlgorithm
        New independent discrete MLSL instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None``.
    """
    if random_seed is None:
        raise ValueError("DISCRETE_MLSL_PEN_SI_DESC_COLD_START requires a random_seed (unsigned 32-bit int)")
    return DiscreteMLSL(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        batch_size=int(parameters["batch_size"]),
        reduction_fraction=float(parameters["reduction_fraction"]),
        sigma=float(parameters["sigma"]),
        sampler=str(parameters["sampler"]),
    )


def build_sobol_sample(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct scrambled Sobol sampling with no local search."""
    if random_seed is None:
        raise ValueError(
            "SOBOL_PEN_COLD_START requires a random_seed (unsigned 32-bit int)"
        )
    return SobolSample(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_sobol_multistart(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct scrambled-Sobol multistart with strict-improvement descent."""
    if random_seed is None:
        raise ValueError(
            "SOBOL_MULTISTART_PEN_SI_DESC_COLD_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    return SobolMultistart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_sobol_multistart_equ_desc(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct scrambled-Sobol multistart with equal descent."""
    if random_seed is None:
        raise ValueError(
            "SOBOL_MULTISTART_PEN_EQU_DESC_COLD_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    return SobolMultistartEquDesc(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_discrete_mlsl_equ_desc(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct equal-descent discrete MLSL."""
    if random_seed is None:
        raise ValueError(
            "DISCRETE_MLSL_PEN_EQU_DESC_COLD_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    return DiscreteMLSLEquDesc(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        batch_size=int(parameters["batch_size"]),
        reduction_fraction=float(parameters["reduction_fraction"]),
        sigma=float(parameters["sigma"]),
        sampler=str(parameters["sampler"]),
    )


def build_discrete_mlsl_equ_desc_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct equal-descent discrete MLSL from a ranked InitialDesign prefix."""
    if random_seed is None:
        raise ValueError(
            "DISCRETE_MLSL_PEN_EQU_DESC_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "DISCRETE_MLSL_PEN_EQU_DESC_WARM_START requires initial_design_path"
        )
    return DiscreteMLSLEquDescWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        batch_size=int(parameters["batch_size"]),
        reduction_fraction=float(parameters["reduction_fraction"]),
        sigma=float(parameters["sigma"]),
        sampler=str(parameters["sampler"]),
        initial_design_path=path,
    )


def build_discrete_mlsl_deb(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct constraint-aware dMLSL-L1-Deb."""
    if random_seed is None:
        raise ValueError(
            "DISCRETE_MLSL_DEB_SI_DESC_COLD_START requires a random_seed (unsigned 32-bit int)"
        )
    return DiscreteMLSLDeb(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        batch_size=int(parameters["batch_size"]),
        reduction_fraction=float(parameters["reduction_fraction"]),
        sigma=float(parameters["sigma"]),
        sampler=str(parameters["sampler"]),
    )


def build_discrete_mlsl_deb_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct dMLSL-L1-Deb from a ranked InitialDesign prefix."""
    if random_seed is None:
        raise ValueError(
            "DISCRETE_MLSL_DEB_SI_DESC_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "DISCRETE_MLSL_DEB_SI_DESC_WARM_START requires initial_design_path"
        )
    return DiscreteMLSLDebWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        batch_size=int(parameters["batch_size"]),
        reduction_fraction=float(parameters["reduction_fraction"]),
        sigma=float(parameters["sigma"]),
        sampler=str(parameters["sampler"]),
        initial_design_path=path,
    )


def build_discrete_mlsl_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct discrete MLSL from a ranked InitialDesign first batch.
    """
    if random_seed is None:
        raise ValueError(
            "DISCRETE_MLSL_PEN_SI_DESC_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError("DISCRETE_MLSL_PEN_SI_DESC_WARM_START requires initial_design_path")
    return DiscreteMLSLWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        batch_size=int(parameters["batch_size"]),
        reduction_fraction=float(parameters["reduction_fraction"]),
        sigma=float(parameters["sigma"]),
        sampler=str(parameters["sampler"]),
        initial_design_path=path,
    )


def build_discrete_mlsl_deb_equ_desc(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct constraint-aware equal-descent dMLSL."""
    if random_seed is None:
        raise ValueError(
            "DISCRETE_MLSL_DEB_EQU_DESC_COLD_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    return DiscreteMLSLDebEquDesc(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        batch_size=int(parameters["batch_size"]),
        reduction_fraction=float(parameters["reduction_fraction"]),
        sigma=float(parameters["sigma"]),
        sampler=str(parameters["sampler"]),
    )


def build_discrete_mlsl_deb_equ_desc_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct equal-descent dMLSL-Deb from a Deb-ranked InitialDesign prefix."""
    if random_seed is None:
        raise ValueError(
            "DISCRETE_MLSL_DEB_EQU_DESC_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "DISCRETE_MLSL_DEB_EQU_DESC_WARM_START requires initial_design_path"
        )
    return DiscreteMLSLDebEquDescWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        batch_size=int(parameters["batch_size"]),
        reduction_fraction=float(parameters["reduction_fraction"]),
        sigma=float(parameters["sigma"]),
        sampler=str(parameters["sampler"]),
        initial_design_path=path,
    )


def build_k_elite_ams(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`KEliteAMS` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters, including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed. A new generator is built per call.

    Returns
    -------
    IOHAlgorithm
        New independent k-elite AMS instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None``.
    """
    if random_seed is None:
        raise ValueError("K_ELITE_AMS_PEN_SI_DESC_COLD_START requires a random_seed (unsigned 32-bit int)")
    return KEliteAMS(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        archive_size=int(parameters["archive_size"]),
        n_parents=int(parameters["n_parents"]),
        mutation_probability=float(parameters["mutation_probability"]),
        random_restart_probability=float(parameters["random_restart_probability"]),
    )


def build_k_elite_ams_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct k-elite AMS from a ranked InitialDesign prefix.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters, including ``initial_design_path``.
    random_seed : int or None
        Required deterministic seed. Startup does not draw from it.

    Returns
    -------
    IOHAlgorithm
        New independent warm-start k-elite AMS instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is missing, or ``initial_design_path`` is missing
        or empty.
    """
    if random_seed is None:
        raise ValueError(
            "K_ELITE_AMS_PEN_SI_DESC_WARM_START requires a random_seed (unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError("K_ELITE_AMS_PEN_SI_DESC_WARM_START requires initial_design_path")
    return KEliteAMSWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        archive_size=int(parameters["archive_size"]),
        n_parents=int(parameters["n_parents"]),
        mutation_probability=float(parameters["mutation_probability"]),
        random_restart_probability=float(parameters["random_restart_probability"]),
        initial_design_path=path,
    )


def build_k_elite_ams_equ_desc(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct equal-descent k-elite AMS."""
    if random_seed is None:
        raise ValueError(
            "K_ELITE_AMS_PEN_EQU_DESC_COLD_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    return KEliteAMSEquDesc(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        archive_size=int(parameters["archive_size"]),
        n_parents=int(parameters["n_parents"]),
        mutation_probability=float(parameters["mutation_probability"]),
        random_restart_probability=float(parameters["random_restart_probability"]),
    )


def build_k_elite_ams_equ_desc_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct equal-descent k-elite AMS from a ranked InitialDesign prefix."""
    if random_seed is None:
        raise ValueError(
            "K_ELITE_AMS_PEN_EQU_DESC_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "K_ELITE_AMS_PEN_EQU_DESC_WARM_START requires initial_design_path"
        )
    return KEliteAMSEquDescWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        archive_size=int(parameters["archive_size"]),
        n_parents=int(parameters["n_parents"]),
        mutation_probability=float(parameters["mutation_probability"]),
        random_restart_probability=float(parameters["random_restart_probability"]),
        initial_design_path=path,
    )


def build_k_elite_ams_deb(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`KEliteAMSDeb` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters, including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed. A new generator is built per call.

    Returns
    -------
    IOHAlgorithm
        New independent Deb k-elite AMS instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None``.
    """
    if random_seed is None:
        raise ValueError(
            "K_ELITE_AMS_DEB_SI_DESC_COLD_START requires a random_seed (unsigned 32-bit int)"
        )
    return KEliteAMSDeb(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        archive_size=int(parameters["archive_size"]),
        n_parents=int(parameters["n_parents"]),
        mutation_probability=float(parameters["mutation_probability"]),
        random_restart_probability=float(parameters["random_restart_probability"]),
    )


def build_k_elite_ams_deb_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct Deb k-elite AMS from a ``y_p``-ranked InitialDesign prefix.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters, including ``initial_design_path``.
    random_seed : int or None
        Required deterministic seed. The prefix block does not draw from it.

    Returns
    -------
    IOHAlgorithm
        New independent warm-start Deb k-elite AMS instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is missing, or ``initial_design_path`` is missing
        or empty.
    """
    if random_seed is None:
        raise ValueError(
            "K_ELITE_AMS_DEB_SI_DESC_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError("K_ELITE_AMS_DEB_SI_DESC_WARM_START requires initial_design_path")
    return KEliteAMSDebWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        archive_size=int(parameters["archive_size"]),
        n_parents=int(parameters["n_parents"]),
        mutation_probability=float(parameters["mutation_probability"]),
        random_restart_probability=float(parameters["random_restart_probability"]),
        initial_design_path=path,
    )


def build_k_elite_ams_deb_equ_desc(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct constraint-aware equal-descent k-elite AMS."""
    if random_seed is None:
        raise ValueError(
            "K_ELITE_AMS_DEB_EQU_DESC_COLD_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    return KEliteAMSDebEquDesc(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        archive_size=int(parameters["archive_size"]),
        n_parents=int(parameters["n_parents"]),
        mutation_probability=float(parameters["mutation_probability"]),
        random_restart_probability=float(parameters["random_restart_probability"]),
    )


def build_k_elite_ams_deb_equ_desc_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct equal-descent Deb AMS from a Deb-ranked InitialDesign prefix."""
    if random_seed is None:
        raise ValueError(
            "K_ELITE_AMS_DEB_EQU_DESC_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "K_ELITE_AMS_DEB_EQU_DESC_WARM_START requires initial_design_path"
        )
    return KEliteAMSDebEquDescWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        archive_size=int(parameters["archive_size"]),
        n_parents=int(parameters["n_parents"]),
        mutation_probability=float(parameters["mutation_probability"]),
        random_restart_probability=float(parameters["random_restart_probability"]),
        initial_design_path=path,
    )


def build_population_basin_hopping(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`PopulationBasinHopping` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters, including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed. A new generator is built per call.

    Returns
    -------
    IOHAlgorithm
        New independent population basin hopping instance.

    Raises
    ------
    ValueError
        If ``random_seed`` is ``None``.
    """
    if random_seed is None:
        raise ValueError(
            "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    return PopulationBasinHopping(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        population_size=int(parameters["population_size"]),
        perturbation_fraction=float(parameters["perturbation_fraction"]),
        dcut_factor=float(parameters["dcut_factor"]),
        initial_sampler=str(parameters["initial_sampler"]),
    )


def build_population_basin_hopping_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct basin hopping from a ranked InitialDesign prefix.
    """
    if random_seed is None:
        raise ValueError(
            "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START requires initial_design_path"
        )
    return PopulationBasinHoppingWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        population_size=int(parameters["population_size"]),
        perturbation_fraction=float(parameters["perturbation_fraction"]),
        dcut_factor=float(parameters["dcut_factor"]),
        initial_sampler=str(parameters["initial_sampler"]),
        initial_design_path=path,
    )


def build_population_basin_hopping_equ_desc(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct equal-descent population basin hopping."""
    if random_seed is None:
        raise ValueError(
            "POPULATION_BASIN_HOPPING_PEN_EQU_DESC_COLD_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    return PopulationBasinHoppingEquDesc(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        population_size=int(parameters["population_size"]),
        perturbation_fraction=float(parameters["perturbation_fraction"]),
        dcut_factor=float(parameters["dcut_factor"]),
        initial_sampler=str(parameters["initial_sampler"]),
    )


def build_population_basin_hopping_equ_desc_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct equal-descent basin hopping from a ranked InitialDesign prefix."""
    if random_seed is None:
        raise ValueError(
            "POPULATION_BASIN_HOPPING_PEN_EQU_DESC_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "POPULATION_BASIN_HOPPING_PEN_EQU_DESC_WARM_START requires "
            "initial_design_path"
        )
    return PopulationBasinHoppingEquDescWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        population_size=int(parameters["population_size"]),
        perturbation_fraction=float(parameters["perturbation_fraction"]),
        dcut_factor=float(parameters["dcut_factor"]),
        initial_sampler=str(parameters["initial_sampler"]),
        initial_design_path=path,
    )


def build_population_basin_hopping_deb(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct constraint-aware PBH-L1-Deb."""
    if random_seed is None:
        raise ValueError(
            "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    return PopulationBasinHoppingDeb(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        population_size=int(parameters["population_size"]),
        perturbation_fraction=float(parameters["perturbation_fraction"]),
        dcut_factor=float(parameters["dcut_factor"]),
        initial_sampler=str(parameters["initial_sampler"]),
    )


def build_population_basin_hopping_deb_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct PBH-L1-Deb from a ranked InitialDesign prefix."""
    if random_seed is None:
        raise ValueError(
            "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START requires "
            "initial_design_path"
        )
    return PopulationBasinHoppingDebWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        population_size=int(parameters["population_size"]),
        perturbation_fraction=float(parameters["perturbation_fraction"]),
        dcut_factor=float(parameters["dcut_factor"]),
        initial_sampler=str(parameters["initial_sampler"]),
        initial_design_path=path,
    )


def build_population_basin_hopping_deb_equ_desc(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct constraint-aware equal-descent PBH."""
    if random_seed is None:
        raise ValueError(
            "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_COLD_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    return PopulationBasinHoppingDebEquDesc(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        population_size=int(parameters["population_size"]),
        perturbation_fraction=float(parameters["perturbation_fraction"]),
        dcut_factor=float(parameters["dcut_factor"]),
        initial_sampler=str(parameters["initial_sampler"]),
    )


def build_population_basin_hopping_deb_equ_desc_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """Construct equal-descent Deb PBH from a Deb-ranked InitialDesign prefix."""
    if random_seed is None:
        raise ValueError(
            "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START requires a "
            "random_seed (unsigned 32-bit int)"
        )
    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START requires "
            "initial_design_path"
        )
    return PopulationBasinHoppingDebEquDescWarmStart(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        population_size=int(parameters["population_size"]),
        perturbation_fraction=float(parameters["perturbation_fraction"]),
        dcut_factor=float(parameters["dcut_factor"]),
        initial_sampler=str(parameters["initial_sampler"]),
        initial_design_path=path,
    )
