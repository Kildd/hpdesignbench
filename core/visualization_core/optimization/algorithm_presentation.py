"""
Display metadata for optimisation algorithms.

Plot labels, cold/warm-start class, and constraint handling are recorded
here. Callers must not recover them by parsing algorithm tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AlgorithmPresentation:
    """
    How one registered algorithm is named and classified on a figure.

    Attributes
    ----------
    token : str
        Registry name.
    display_name : str
        Name shown in the legend and tooltip.
    algorithm_class : str
        Study class: Metaheuristic, Model-based, Direct search,
        Derivative-based, or Sampling. Variants of one method share it.
    family : str
        Cold-start token of the cold/warm pair. Penalty and constraint-aware
        variants of the same method are different families.
    start : str
        ``\"cold\"`` or ``\"warm\"``.
    constraint_aware : bool
        ``True`` when the search is not on the scalar penalised objective.
    qualifier : str
        Extra label used only when two algorithms would otherwise share a
        legend entry. Empty when the display name is already unique.
    """

    token: str
    display_name: str
    algorithm_class: str
    family: str
    start: str
    constraint_aware: bool
    qualifier: str = ""


def _family_of(token: str) -> str:
    """Return the cold-start token of a cold/warm pair.

    Equal-descent and strictly-improving-descent tokens name the cold
    start explicitly. Every other warm start still drops ``_WARM_START``.
    """
    if token.endswith("_WARM_START"):
        stem = token[: -len("_WARM_START")]
        if stem.endswith(("_SI_DESC", "_EQU_DESC")):
            return f"{stem}_COLD_START"
        return stem
    return token


def _row(
    token: str,
    display_name: str,
    algorithm_class: str,
    *,
    constraint_aware: bool = False,
    qualifier: str = "",
    warm: bool = False,
) -> AlgorithmPresentation:
    family = _family_of(token)
    return AlgorithmPresentation(
        token=token,
        display_name=display_name,
        algorithm_class=algorithm_class,
        family=family,
        start="warm" if warm else "cold",
        constraint_aware=constraint_aware,
        qualifier=qualifier,
    )


def _pair(
    token: str,
    display_name: str,
    algorithm_class: str,
    *,
    constraint_aware: bool = False,
    qualifier: str = "",
) -> tuple[AlgorithmPresentation, AlgorithmPresentation]:
    return (
        _row(
            token,
            display_name,
            algorithm_class,
            constraint_aware=constraint_aware,
            qualifier=qualifier,
        ),
        _row(
            f"{token}_WARM_START",
            display_name,
            algorithm_class,
            constraint_aware=constraint_aware,
            qualifier=qualifier,
            warm=True,
        ),
    )


def _descent_pair(
    stem: str,
    display_name: str,
    algorithm_class: str,
    *,
    constraint_aware: bool = False,
    qualifier: str = "",
) -> tuple[AlgorithmPresentation, AlgorithmPresentation]:
    return (
        _row(
            f"{stem}_COLD_START",
            display_name,
            algorithm_class,
            constraint_aware=constraint_aware,
            qualifier=qualifier,
        ),
        _row(
            f"{stem}_WARM_START",
            display_name,
            algorithm_class,
            constraint_aware=constraint_aware,
            qualifier=qualifier,
            warm=True,
        ),
    )


CLASS_ORDER: tuple[str, ...] = (
    "Metaheuristic",
    "Model-based",
    "Direct search",
    "Derivative-based",
    "Sampling",
)


# Legend text is the manuscript algorithm name, not the registry token
# and not the table abbreviation. Penalty and Deb copies of one method
# share that name; the constraint label separates them when both are drawn.
_ENTRIES: tuple[AlgorithmPresentation, ...] = (
    _row("RANDOM_SEARCH", "Random search", "Sampling"),
    _row("LATIN_HYPERCUBE", "Latin hypercube", "Sampling"),
    *_descent_pair(
        "DISCRETE_MLSL_PEN_EQU_DESC",
        "Discrete Multi-Level Single Linkage, EQU",
        "Metaheuristic",
        qualifier="EQU-DESC",
    ),
    *_descent_pair(
        "DISCRETE_MLSL_PEN_SI_DESC",
        "Discrete Multi-Level Single Linkage, SI",
        "Metaheuristic",
        qualifier="SI-DESC",
    ),
    *_descent_pair(
        "DISCRETE_MLSL_DEB_EQU_DESC",
        "Discrete Multi-Level Single Linkage, EQU",
        "Metaheuristic",
        constraint_aware=True,
        qualifier="EQU-DESC",
    ),
    *_descent_pair(
        "DISCRETE_MLSL_DEB_SI_DESC",
        "Discrete Multi-Level Single Linkage, SI",
        "Metaheuristic",
        constraint_aware=True,
        qualifier="SI-DESC",
    ),
    *_descent_pair(
        "K_ELITE_AMS_PEN_EQU_DESC",
        "Adaptive Multi-Start, EQU",
        "Metaheuristic",
        qualifier="EQU-DESC",
    ),
    *_descent_pair(
        "K_ELITE_AMS_PEN_SI_DESC",
        "Adaptive Multi-Start, SI",
        "Metaheuristic",
        qualifier="SI-DESC",
    ),
    *_descent_pair(
        "K_ELITE_AMS_DEB_EQU_DESC",
        "Adaptive Multi-Start, EQU",
        "Metaheuristic",
        constraint_aware=True,
        qualifier="EQU-DESC",
    ),
    *_descent_pair(
        "K_ELITE_AMS_DEB_SI_DESC",
        "Adaptive Multi-Start, SI",
        "Metaheuristic",
        constraint_aware=True,
        qualifier="SI-DESC",
    ),
    *_descent_pair(
        "POPULATION_BASIN_HOPPING_PEN_EQU_DESC",
        "Population Basin Hopping, EQU",
        "Metaheuristic",
        qualifier="EQU-DESC",
    ),
    *_descent_pair(
        "POPULATION_BASIN_HOPPING_PEN_SI_DESC",
        "Population Basin Hopping, SI",
        "Metaheuristic",
        qualifier="SI-DESC",
    ),
    *_descent_pair(
        "POPULATION_BASIN_HOPPING_DEB_EQU_DESC",
        "Population Basin Hopping, EQU",
        "Metaheuristic",
        constraint_aware=True,
        qualifier="EQU-DESC",
    ),
    *_descent_pair(
        "POPULATION_BASIN_HOPPING_DEB_SI_DESC",
        "Population Basin Hopping, SI",
        "Metaheuristic",
        constraint_aware=True,
        qualifier="SI-DESC",
    ),
    *_pair("SCIPY_DIFFERENTIAL_EVOLUTION", "Differential Evolution", "Metaheuristic"),
    *_pair("SCIPY_COBYLA", "COBYLA", "Direct search", constraint_aware=True),
    _row("SCIPY_DIRECT_L", "DIRECT", "Direct search", qualifier="SciPy"),
    *_pair("SCIPY_DUAL_ANNEALING", "Dual Annealing", "Metaheuristic"),
    _row("NLOPT_DIRECT_L", "DIRECT", "Direct search", qualifier="NLopt"),
    *_pair("NLOPT_SBPLX", "SBPLX", "Direct search"),
    *_pair("LEAP_GA", "Genetic Algorithm (LEAP)", "Metaheuristic"),
    *_pair("CMAES_CATCMAWM", "CatCMAwM", "Metaheuristic"),
    *_pair("PYMOO_GA", "Genetic Algorithm (Pymoo)", "Metaheuristic"),
    *_pair(
        "PYMOO_GA_CONSTRAINED",
        "Genetic Algorithm (Pymoo)",
        "Metaheuristic",
        constraint_aware=True,
    ),
    *_pair("PYSTARPSO_INTEGER_PSO", "Particle Swarm Optimisation", "Metaheuristic"),
    *_pair("NEVERGRAD_METAMODEL", "Metamodel", "Model-based"),
    *_pair("NEVERGRAD_METAMODEL_ONEPLUSONE", "MM (1+1)", "Model-based"),
    *_pair(
        "HYPEROPT_TPE",
        "Tree-Structured Parzen Estimator",
        "Model-based",
        qualifier="Hyperopt",
    ),
    *_pair(
        "OPTUNA_TPE",
        "Tree-Structured Parzen Estimator",
        "Model-based",
        qualifier="Optuna",
    ),
    *_pair("RBFOPT", "RBFOpt", "Model-based"),
    *_pair("SMT_EGO", "Efficient global optimisation", "Model-based"),
    _row("BLACKBOXOPT_DYCORS", "DYCORS", "Model-based"),
    _row("BLACKBOXOPT_GOSAC", "GOSAC", "Model-based"),
    _row("BLACKBOXOPT_MSRS", "MSRS", "Model-based"),
    *_pair("ALTBACKEN_SA", "Simulated annealing", "Metaheuristic"),
    *_pair("DFLINT_BBOA", "DFLINT", "Direct search", constraint_aware=True),
    *_pair("PYNOMAD_MADS_PB", "MADS-PB", "Direct search", constraint_aware=True),
    *_pair(
        "PYNOMAD_VNS_MADS_PB",
        "VNS-MADS-PB",
        "Direct search",
        constraint_aware=True,
    ),
    *_pair("PFN_PEN", "Prior-data fitted network", "Model-based"),
    *_pair(
        "PFN_CEI",
        "Prior-data fitted network",
        "Model-based",
        constraint_aware=True,
    ),
    *_pair("OPTAX_ADAM", "Adam", "Derivative-based"),
    *_pair("OPTAX_RPROP", "Rprop", "Derivative-based"),
    *_pair(
        "OPTAX_SGD_NESTEROV",
        "Nesterov SGD",
        "Derivative-based",
    ),
)

PRESENTATIONS: dict[str, AlgorithmPresentation] = {
    item.token: item for item in _ENTRIES
}

# Dataset A stored the penalty strict-improvement local searches under these
# names. They are the same methods as the renamed tokens.
_HISTORICAL_SI_DESC: dict[str, str] = {
    "POPULATION_BASIN_HOPPING": "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
    "POPULATION_BASIN_HOPPING_WARM_START": (
        "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START"
    ),
    "K_ELITE_AMS": "K_ELITE_AMS_PEN_SI_DESC_COLD_START",
    "K_ELITE_AMS_WARM_START": "K_ELITE_AMS_PEN_SI_DESC_WARM_START",
    "DISCRETE_MLSL": "DISCRETE_MLSL_PEN_SI_DESC_COLD_START",
    "DISCRETE_MLSL_WARM_START": "DISCRETE_MLSL_PEN_SI_DESC_WARM_START",
}


class AlgorithmPresentationError(KeyError):
    """Raised when a plotted algorithm has no display metadata."""


def presentation_for(token: str) -> AlgorithmPresentation:
    """
    Return the display metadata for one algorithm token.

    Parameters
    ----------
    token : str
        Registry algorithm name, or a historical strict-improvement token
        from the reference archive.

    Returns
    -------
    AlgorithmPresentation
        Recorded name, class, start, and constraint handling.

    Raises
    ------
    AlgorithmPresentationError
        If ``token`` is not in the metadata table.
    """
    canonical = _HISTORICAL_SI_DESC.get(token, token)
    try:
        item = PRESENTATIONS[canonical]
    except KeyError as exc:
        raise AlgorithmPresentationError(
            f"no display metadata for algorithm {token!r}"
        ) from exc
    if canonical == token:
        return item
    return replace(item, token=token)


def start_label(item: AlgorithmPresentation) -> str:
    """Return ``\"Warm start\"`` or ``\"Cold start\"``."""
    return "Warm start" if item.start == "warm" else "Cold start"


def constraint_label(item: AlgorithmPresentation) -> str:
    """Return the constraint-handling label."""
    if item.constraint_aware:
        return "Constraint-aware"
    return "Penalised objective"


def local_search_label(item: AlgorithmPresentation) -> str:
    """Return the local-search caption for PBH, AMS, and MLSL.

    Other algorithms return an empty string. The caption is part of every
    legend and heatmap label for these three methods.
    """
    if item.qualifier == "EQU-DESC" or "_EQU_DESC_" in item.token:
        return "Equal descent"
    if item.qualifier == "SI-DESC" or "_SI_DESC_" in item.token:
        return "Strict improvement"
    return ""


def legend_handling(item: AlgorithmPresentation) -> str:
    """Return the small legend line under the algorithm name."""
    base = "Constraint-aware" if item.constraint_aware else "Penalty-based"
    search = local_search_label(item)
    if search:
        return f"{base}, {search}"
    return base


def _shown_qualifier(item: AlgorithmPresentation, *, show_qualifier: bool) -> str:
    """Return a qualifier that is not already stated as the local search."""
    if not show_qualifier or item.qualifier in {"", "SI-DESC", "EQU-DESC"}:
        return ""
    return item.qualifier


def tooltip_subtitle(item: AlgorithmPresentation, *, show_qualifier: bool) -> str:
    """Return the second tooltip line."""
    start = start_label(item)
    search = local_search_label(item)
    if search:
        start = f"{start}, {search}"
    text = f"{start} · {constraint_label(item)}"
    qualifier = _shown_qualifier(item, show_qualifier=show_qualifier)
    if qualifier:
        return f"{text} · {qualifier}"
    return text


def legend_label(
    item: AlgorithmPresentation,
    n_success: int,
    n_runs: int,
    *,
    show_qualifier: bool,
) -> str:
    """Return one legend entry."""
    head = item.display_name
    qualifier = _shown_qualifier(item, show_qualifier=show_qualifier)
    if qualifier:
        head = f"{head} ({qualifier})"
    return f"{head} · {start_label(item)} ({n_success}/{n_runs})"


def qualifiers_needed(tokens: tuple[str, ...] | list[str]) -> set[str]:
    """
    Return tokens whose legend entry would collide without the qualifier.

    Parameters
    ----------
    tokens : sequence of str
        Algorithm tokens on one figure.

    Returns
    -------
    set of str
        Tokens that share a display name, start, and constraint class with
        another token on the figure.
    """
    groups: dict[tuple[str, str, bool], list[str]] = {}
    for token in tokens:
        item = presentation_for(token)
        key = (item.display_name, item.start, item.constraint_aware)
        groups.setdefault(key, []).append(token)
    crowded: set[str] = set()
    for names in groups.values():
        if len(names) > 1:
            crowded.update(names)
    return crowded
