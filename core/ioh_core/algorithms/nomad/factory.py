"""
Import-safe constructors for PyNomad / NOMAD algorithm adapters.
"""
from __future__ import annotations

from typing import Mapping

from core.ioh_core.algorithms.errors import import_optional_dependency
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from core.ioh_core.algorithms.registry import JSONValue, RegistryError


def build_pynomad_mads_pb(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`PyNomadMadsPb` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent PyNomad MADS-PB instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional PyNomadBBO extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "PyNomad",
        algorithm_name="PYNOMAD_MADS_PB",
        extra="algorithms-nomad",
    )
    if random_seed is None:
        raise ValueError(
            "PYNOMAD_MADS_PB requires a random_seed (unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.nomad.algorithms import PyNomadMadsPb

    return PyNomadMadsPb(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
    )


def build_pynomad_mads_pb_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct PyNomad MADS-PB starting at rank 1 of the InitialDesign.
    """
    import_optional_dependency(
        "PyNomad",
        algorithm_name="PYNOMAD_MADS_PB_WARM_START",
        extra="algorithms-nomad",
    )
    if random_seed is None:
        raise ValueError(
            "PYNOMAD_MADS_PB_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.nomad.algorithms import PyNomadMadsPb

    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or path == "":
        raise ValueError(
            "PYNOMAD_MADS_PB_WARM_START requires initial_design_path"
        )
    return PyNomadMadsPb(
        max_evaluations=int(parameters["max_evaluations"]),
        seed=int(random_seed),
        initial_design_path=path,
        name="PYNOMAD_MADS_PB_WARM_START",
    )


def validate_pynomad_vns_mads_pb_parameters(
    parameters: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    """
    Coerce ``vns_mads_search_trigger`` to float and reject non-positive values.

    ``ParameterSpec.minimum`` is inclusive, so ``0`` survives range checks.
    This token's identity is that VNS search may spend evaluations, so the
    trigger must be strictly greater than 0.

    Parameters
    ----------
    parameters : mapping
        Partially normalized hyperparameters (static defaults already applied).

    Returns
    -------
    dict
        Mapping with ``vns_mads_search_trigger`` as ``float``.

    Raises
    ------
    RegistryError
        If the trigger is not strictly positive.
    """
    out = dict(parameters)
    raw = out["vns_mads_search_trigger"]
    number = float(raw)
    if number <= 0.0:
        raise RegistryError(
            "PYNOMAD_VNS_MADS_PB: parameter 'vns_mads_search_trigger' "
            f"must be > 0, got {raw!r}"
        )
    out["vns_mads_search_trigger"] = number
    return out


def build_pynomad_vns_mads_pb(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct a fresh :class:`PyNomadVnsMadsPb` instance.

    Parameters
    ----------
    parameters : mapping
        Normalized hyperparameters including ``max_evaluations`` and
        ``vns_mads_search_trigger``.
    random_seed : int or None
        Required deterministic seed.

    Returns
    -------
    IOHAlgorithm
        New independent PyNomad VNS-MADS-PB instance.

    Raises
    ------
    AlgorithmDependencyError
        If the optional PyNomadBBO extra is not installed.
    ValueError
        If ``random_seed`` is ``None``.
    """
    import_optional_dependency(
        "PyNomad",
        algorithm_name="PYNOMAD_VNS_MADS_PB",
        extra="algorithms-nomad",
    )
    if random_seed is None:
        raise ValueError(
            "PYNOMAD_VNS_MADS_PB requires a random_seed "
            "(unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.nomad.algorithms import PyNomadVnsMadsPb

    return PyNomadVnsMadsPb(
        max_evaluations=int(parameters["max_evaluations"]),
        vns_mads_search_trigger=float(parameters["vns_mads_search_trigger"]),
        seed=int(random_seed),
    )


def build_pynomad_vns_mads_pb_warm_start(
    parameters: Mapping[str, JSONValue],
    random_seed: int | None,
) -> IOHAlgorithm:
    """
    Construct PyNomad VNS-MADS-PB starting at rank 1 of the InitialDesign.
    """
    import_optional_dependency(
        "PyNomad",
        algorithm_name="PYNOMAD_VNS_MADS_PB_WARM_START",
        extra="algorithms-nomad",
    )
    if random_seed is None:
        raise ValueError(
            "PYNOMAD_VNS_MADS_PB_WARM_START requires a random_seed "
            "(unsigned 32-bit int)"
        )
    from core.ioh_core.algorithms.nomad.algorithms import PyNomadVnsMadsPb

    path = parameters.get("initial_design_path")
    if not isinstance(path, str) or path == "":
        raise ValueError(
            "PYNOMAD_VNS_MADS_PB_WARM_START requires initial_design_path"
        )
    return PyNomadVnsMadsPb(
        max_evaluations=int(parameters["max_evaluations"]),
        vns_mads_search_trigger=float(parameters["vns_mads_search_trigger"]),
        seed=int(random_seed),
        initial_design_path=path,
        name="PYNOMAD_VNS_MADS_PB_WARM_START",
    )
