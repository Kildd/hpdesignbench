"""
Load a ranked warm-start InitialDesign.

One-point tokens use rank 1. A population token uses a prefix of the
length the cold algorithm already uses for its initial set. The file
is ``initial_design.json``, or ``initial_design_deb.json`` for every
warm start that searches on an unpenalized objective and separate
constraints. The planner chooses the path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

RANK1_SEEDLESS_TOKENS: frozenset[str] = frozenset(
    {
        "DFLINT_BBOA_WARM_START",
        "NLOPT_SBPLX_WARM_START",
        "OPTAX_ADAM_WARM_START",
        "OPTAX_RPROP_WARM_START",
        "OPTAX_SGD_NESTEROV_WARM_START",
        "SCIPY_COBYLA_WARM_START",
    }
)
INITIAL_DESIGN_PATH_TOKENS: frozenset[str] = frozenset(
    {
        "ALTBACKEN_SA_WARM_START",
        "CMAES_CATCMAWM_WARM_START",
        "DFLINT_BBOA_WARM_START",
        "DISCRETE_MLSL_DEB_EQU_DESC_WARM_START",
        "DISCRETE_MLSL_DEB_SI_DESC_WARM_START",
        "DISCRETE_MLSL_PEN_EQU_DESC_WARM_START",
        "DISCRETE_MLSL_PEN_SI_DESC_WARM_START",
        "HYPEROPT_TPE_WARM_START",
        "K_ELITE_AMS_DEB_EQU_DESC_WARM_START",
        "K_ELITE_AMS_DEB_SI_DESC_WARM_START",
        "K_ELITE_AMS_PEN_EQU_DESC_WARM_START",
        "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
        "LEAP_GA_WARM_START",
        "NEVERGRAD_METAMODEL_ONEPLUSONE_WARM_START",
        "NEVERGRAD_METAMODEL_WARM_START",
        "NLOPT_SBPLX_WARM_START",
        "OPTAX_ADAM_WARM_START",
        "OPTAX_RPROP_WARM_START",
        "OPTAX_SGD_NESTEROV_WARM_START",
        "OPTUNA_TPE_WARM_START",
        "PFN_CEI_WARM_START",
        "PFN_PEN_WARM_START",
        "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START",
        "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
        "POPULATION_BASIN_HOPPING_PEN_EQU_DESC_WARM_START",
        "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
        "PYMOO_GA_CONSTRAINED_WARM_START",
        "PYMOO_GA_WARM_START",
        "PYNOMAD_MADS_PB_WARM_START",
        "PYNOMAD_VNS_MADS_PB_WARM_START",
        "PYSTARPSO_INTEGER_PSO_WARM_START",
        "RBFOPT_WARM_START",
        "SCIPY_COBYLA_WARM_START",
        "SCIPY_DIFFERENTIAL_EVOLUTION_WARM_START",
        "SCIPY_DUAL_ANNEALING_WARM_START",
        "SMT_EGO_WARM_START",
    }
)
DEB_WARM_START_TOKENS: frozenset[str] = frozenset(
    {
        "DFLINT_BBOA_WARM_START",
        "DISCRETE_MLSL_DEB_EQU_DESC_WARM_START",
        "DISCRETE_MLSL_DEB_SI_DESC_WARM_START",
        "K_ELITE_AMS_DEB_EQU_DESC_WARM_START",
        "K_ELITE_AMS_DEB_SI_DESC_WARM_START",
        "OPTUNA_TPE_WARM_START",
        "PFN_CEI_WARM_START",
        "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START",
        "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
        "PYMOO_GA_CONSTRAINED_WARM_START",
        "PYNOMAD_MADS_PB_WARM_START",
        "PYNOMAD_VNS_MADS_PB_WARM_START",
        "SCIPY_COBYLA_WARM_START",
    }
)
WARM_START_TOKENS: frozenset[str] = frozenset(
    {
        "ALTBACKEN_SA_WARM_START",
        "CMAES_CATCMAWM_WARM_START",
        "DFLINT_BBOA_WARM_START",
        "DISCRETE_MLSL_DEB_EQU_DESC_WARM_START",
        "DISCRETE_MLSL_DEB_SI_DESC_WARM_START",
        "DISCRETE_MLSL_PEN_EQU_DESC_WARM_START",
        "DISCRETE_MLSL_PEN_SI_DESC_WARM_START",
        "HYPEROPT_TPE_WARM_START",
        "K_ELITE_AMS_DEB_EQU_DESC_WARM_START",
        "K_ELITE_AMS_DEB_SI_DESC_WARM_START",
        "K_ELITE_AMS_PEN_EQU_DESC_WARM_START",
        "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
        "LEAP_GA_WARM_START",
        "NEVERGRAD_METAMODEL_ONEPLUSONE_WARM_START",
        "NEVERGRAD_METAMODEL_WARM_START",
        "NLOPT_SBPLX_WARM_START",
        "OPTAX_ADAM_WARM_START",
        "OPTAX_RPROP_WARM_START",
        "OPTAX_SGD_NESTEROV_WARM_START",
        "OPTUNA_TPE_WARM_START",
        "PFN_CEI_WARM_START",
        "PFN_PEN_WARM_START",
        "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START",
        "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START",
        "POPULATION_BASIN_HOPPING_PEN_EQU_DESC_WARM_START",
        "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START",
        "PYMOO_GA_CONSTRAINED_WARM_START",
        "PYMOO_GA_WARM_START",
        "PYNOMAD_MADS_PB_WARM_START",
        "PYNOMAD_VNS_MADS_PB_WARM_START",
        "PYSTARPSO_INTEGER_PSO_WARM_START",
        "RBFOPT_WARM_START",
        "SCIPY_COBYLA_WARM_START",
        "SCIPY_DIFFERENTIAL_EVOLUTION_WARM_START",
        "SCIPY_DUAL_ANNEALING_WARM_START",
        "SMT_EGO_WARM_START",
    }
)


def warm_start_problem_id(problem: Any) -> str:
    """
    Return the problem id attached to ``problem``.

    A unit-test problem with no eval context and no ``decode`` uses
    ``L8100_B2``. A context that can decode but has no ``problem_id``
    raises.

    Parameters
    ----------
    problem : Any
        IOH-like problem.

    Returns
    -------
    str
        Problem id stored in the InitialDesign.

    Raises
    ------
    ValueError
        If the eval context can decode and has no ``problem_id``.
    """
    from core.ioh_core.algorithms.common import get_attached_eval_context

    ctx = get_attached_eval_context(problem)
    attached_id = getattr(ctx, "problem_id", None) if ctx is not None else None
    if attached_id:
        return str(attached_id)
    decode = getattr(ctx, "decode", None) if ctx is not None else None
    if not callable(decode):
        return "L8100_B2"
    raise ValueError("warm start eval context has no problem_id")


def validate_warm_start_candidate(
    _problem: Any,
    values: Sequence[int],
    *,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    algorithm_name: str,
) -> list[int]:
    """
    Validate one start vector against inclusive catalogue bounds.

    Parameters
    ----------
    _problem : Any
        Live problem. The check uses the bounds only.
    values : sequence of int
        Catalogue-index vector.
    lower_bounds, upper_bounds : sequence of int
        Inclusive catalogue-index bounds.
    algorithm_name : str
        Warm-start token name used in error text.

    Returns
    -------
    list of int
        The validated vector.

    Raises
    ------
    CandidateError
        If the vector is outside the bounds.
    """
    from core.ioh_core.algorithms.common import (
        CandidateError,
        validate_integer_candidate,
    )

    try:
        return validate_integer_candidate(
            values,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
    except CandidateError as exc:
        raise CandidateError(
            f"{algorithm_name} start is not a feasible catalogue "
            f"index for this problem: {exc}"
        ) from exc


_INITIAL_DESIGN_KEYS = frozenset(
    {"schema_version", "problem_id", "n", "vectors"}
)


def _missing_initial_design_message(
    path: Path,
    *,
    algorithm_name: str,
    problem_id: str,
) -> str:
    return (
        f"warm start for {algorithm_name} on problem {problem_id} has no "
        f"InitialDesign at {path}. Compute it by running the predesign grid "
        f"for this problem. That run writes {path}."
    )


def _is_json_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_initial_design(
    path: str | Path,
    *,
    algorithm_name: str,
    problem_id: str,
) -> tuple[tuple[int, ...], ...]:
    """
    Read a ranked InitialDesign JSON file.

    Parameters
    ----------
    path : path
        ``initial_design.json`` written by the predesign run.
    algorithm_name : str
        Warm-start token name used in error text.
    problem_id : str
        Task problem id. Must equal the id stored in the file.

    Returns
    -------
    tuple of tuple of int
        Catalogue-index vectors in rank order.

    Raises
    ------
    ValueError
        If the file is missing, not the expected object, or does not
        match ``problem_id``.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(
            _missing_initial_design_message(
                file_path,
                algorithm_name=algorithm_name,
                problem_id=problem_id,
            )
        )
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"InitialDesign at {file_path} is not JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"InitialDesign at {file_path} is not a JSON object")
    if set(payload) != _INITIAL_DESIGN_KEYS:
        raise ValueError(
            f"InitialDesign at {file_path} keys {sorted(payload)} "
            f"are not {sorted(_INITIAL_DESIGN_KEYS)}"
        )
    version = payload["schema_version"]
    if not _is_json_int(version) or version != 1:
        raise ValueError(
            f"InitialDesign at {file_path} has schema_version {version!r}; "
            "expected 1"
        )
    file_problem_id = payload["problem_id"]
    if file_problem_id != problem_id:
        raise ValueError(
            f"InitialDesign at {file_path} is for problem {file_problem_id}; "
            f"expected {problem_id}"
        )
    count = payload["n"]
    vectors = payload["vectors"]
    if not isinstance(vectors, list):
        raise ValueError(
            f"InitialDesign at {file_path} vectors is not a list"
        )
    if not _is_json_int(count):
        raise ValueError(
            f"InitialDesign at {file_path} has n={count!r}"
        )
    if count < 1:
        raise ValueError(f"InitialDesign at {file_path} has n={count}")
    if count != len(vectors):
        raise ValueError(
            f"InitialDesign at {file_path} has n={count} but {len(vectors)} vectors"
        )
    parsed: list[tuple[int, ...]] = []
    for vector in vectors:
        if not isinstance(vector, list):
            raise ValueError(
                f"InitialDesign at {file_path} has a vector that is not a list"
            )
        if len(vector) != 9:
            raise ValueError(
                f"InitialDesign at {file_path} has a vector of length "
                f"{len(vector)}; expected 9"
            )
        coords: list[int] = []
        for entry in vector:
            if not _is_json_int(entry):
                raise ValueError(
                    f"InitialDesign at {file_path} has non-int entry {entry!r}"
                )
            coords.append(int(entry))
        parsed.append(tuple(coords))
    return tuple(parsed)


def initial_design_prefix(
    path: str | Path,
    n: int,
    *,
    algorithm_name: str,
    problem_id: str,
) -> tuple[tuple[int, ...], ...]:
    """
    Return the first ``n`` ranked InitialDesign vectors.

    Parameters
    ----------
    path : path
        ``initial_design.json``.
    n : int
        Prefix length, at least 1. A bool is rejected.
    algorithm_name : str
        Warm-start token name used in error text.
    problem_id : str
        Task problem id.

    Returns
    -------
    tuple of tuple of int
        First ``n`` catalogue-index vectors.

    Raises
    ------
    ValueError
        If the file is missing or shorter than ``n``.
    """
    if not _is_json_int(n) or n < 1:
        raise ValueError(
            f"warm start prefix length must be an int >= 1, got {n!r}"
        )
    design = load_initial_design(
        path,
        algorithm_name=algorithm_name,
        problem_id=problem_id,
    )
    if len(design) < n:
        raise ValueError(
            f"warm start for {algorithm_name} needs {n} initial designs "
            f"but {path} contains {len(design)}."
        )
    return design[:n]


def rank1_start_vector(
    problem: Any,
    *,
    algorithm_name: str,
    initial_design_path: str,
    max_evaluations: int,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> tuple[int, ...]:
    """
    Load rank 1, then reject a short budget, then check bounds.

    Does not call ``problem``. A missing or short file is reported before
    the budget check.

    Parameters
    ----------
    problem : Any
        IOH-like problem used only to resolve ``problem_id``.
    algorithm_name : str
        Warm-start token name used in error text.
    initial_design_path : str
        Path of ``initial_design.json``.
    max_evaluations : int
        Run budget. Must be at least 1.
    lower_bounds, upper_bounds : sequence of int
        Inclusive catalogue-index bounds.

    Returns
    -------
    tuple of int
        Rank-1 catalogue-index vector.

    Raises
    ------
    ValueError
        If the file is missing or the budget is below 1.
    CandidateError
        If the vector is outside the bounds.
    """
    problem_id = warm_start_problem_id(problem)
    prefix = initial_design_prefix(
        initial_design_path,
        1,
        algorithm_name=algorithm_name,
        problem_id=problem_id,
    )
    require_warm_start_budget(algorithm_name, 1, max_evaluations)
    validate_prefix_bounds(prefix, lower_bounds, upper_bounds)
    return prefix[0]


def require_warm_start_budget(
    algorithm_name: str,
    x: int,
    max_evaluations: int,
) -> None:
    """
    Reject a budget that cannot pay for ``x`` initial analyses.

    Parameters
    ----------
    algorithm_name : str
        Warm-start token name.
    x : int
        Number of initial analyses the token will perform.
    max_evaluations : int
        Run budget.

    Raises
    ------
    ValueError
        If ``max_evaluations < x``.
    """
    if max_evaluations < x:
        raise ValueError(
            f"warm start for {algorithm_name} needs {x} initial evaluations "
            f"but max_evaluations is {max_evaluations}."
        )


def validate_prefix_bounds(
    vectors: Sequence[Sequence[int]],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> None:
    """
    Validate every prefix vector against inclusive catalogue bounds.

    The whole prefix is checked before this function returns. It does not
    call a problem.

    Parameters
    ----------
    vectors : sequence of sequence of int
        Catalogue-index vectors.
    lower_bounds : sequence of int
        Inclusive lower index per dimension.
    upper_bounds : sequence of int
        Inclusive upper index per dimension.

    Raises
    ------
    CandidateError
        If any vector is outside the bounds.
    """
    for vector in vectors:
        validate_warm_start_candidate(
            None,
            vector,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            algorithm_name="warm start",
        )


def load_warm_start_prefix(
    problem: Any,
    *,
    algorithm_name: str,
    initial_design_path: str,
    count: int,
    max_evaluations: int,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """
    Load a ranked prefix, reject a short budget, then check bounds.

    Does not call ``problem``. A short file is reported before the budget
    check.

    Parameters
    ----------
    problem : Any
        IOH-like problem used only to resolve ``problem_id``.
    algorithm_name : str
        Warm-start token name.
    initial_design_path : str
        Path of ``initial_design.json``.
    count : int
        Prefix length X.
    max_evaluations : int
        Run budget. Must be at least ``count``.
    lower_bounds, upper_bounds : sequence of int
        Inclusive catalogue-index bounds.

    Returns
    -------
    tuple of tuple of int
        The ranked prefix.
    """
    problem_id = warm_start_problem_id(problem)
    prefix = initial_design_prefix(
        initial_design_path,
        count,
        algorithm_name=algorithm_name,
        problem_id=problem_id,
    )
    require_warm_start_budget(algorithm_name, count, max_evaluations)
    validate_prefix_bounds(prefix, lower_bounds, upper_bounds)
    return prefix

