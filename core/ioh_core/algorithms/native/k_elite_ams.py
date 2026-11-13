"""
k-elite adaptive multi-start on certified L1 local minima.

The search is inspired by the adaptive multi-start principle of Boese et
al. and the multi-parent extension of Bonachea et al. Coordinate-wise
recombination is an adaptation to integer index vectors. It is not a
reproduction of their TSP edge operator.

The run minimises ``problem(x)`` through :class:`EvaluationManager` and
:func:`best_improvement_local_search`. It does not read a catalogue LON,
an ILS-LON, a CoLON, a known funnel, or a known optimum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from core.ioh_core.algorithms.common import (
    read_problem_bounds,
    resolve_run_controller,
)
from core.ioh_core.algorithms.native.discrete_search import (
    BudgetExhausted,
    DebEvaluatedPoint,
    DebEvaluationManager,
    DebLocalSearchResult,
    DiscreteSearchSpace,
    EvaluatedPoint,
    EvaluationManager,
    LocalSearchResult,
    best_improvement_local_search,
    deb_best_improvement_local_search,
    deb_equal_descent_local_search,
    equal_descent_local_search,
    deb_point_key,
)
from core.ioh_core.algorithms.registry import JSONValue, RegistryError

_CONSECUTIVE_DUPLICATE_START_CAP = 1000


@dataclass(frozen=True)
class EliteMember:
    """
    One certified local minimum remembered by the archive.

    Attributes
    ----------
    minimum_id : int
        Identity assigned the first time this index vector was certified.
        Eviction does not change it.
    point : EvaluatedPoint
        Certified index vector and its objective.
    """

    minimum_id: int
    point: EvaluatedPoint


@dataclass
class ArchiveBook:
    """
    Active elite archive plus bookkeeping for every certified minimum.

    ``ids_by_x`` does not affect admission. It only preserves identity
    and the distinct-minimum count.

    Attributes
    ----------
    members : list of EliteMember
        Current archive, kept in ``(f, x)`` order after every update.
    ids_by_x : dict
        First-seen id of every certified index vector.
    next_id : int
        Next id to assign.
    n_admissions : int
        Certified endpoints that were absent and remain after the update.
    n_duplicates : int
        Certified endpoints whose index is already a current member.
    n_rejections : int
        Certified endpoints that were absent and do not remain after the
        update.
    """

    members: list[EliteMember] = field(default_factory=list)
    ids_by_x: dict[tuple[int, ...], int] = field(default_factory=dict)
    next_id: int = 0
    n_admissions: int = 0
    n_duplicates: int = 0
    n_rejections: int = 0


def validate_k_elite_ams_parameters(
    parameters: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    """
    Reject a parent count above the archive capacity, and probabilities
    outside ``[0, 1]``.

    Parameters
    ----------
    parameters : dict
        Normalised registry parameters.

    Returns
    -------
    dict
        A copy of ``parameters``.

    Raises
    ------
    RegistryError
        If ``n_parents`` exceeds ``archive_size``, or a probability is
        outside ``[0, 1]``.
    """
    return _validate_k_elite_ams_parameters(parameters, token="K_ELITE_AMS_PEN_SI_DESC_COLD_START")


def validate_k_elite_ams_deb_parameters(
    parameters: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    """Validate Deb AMS parameters with the ``K_ELITE_AMS_DEB_SI_DESC_COLD_START`` error token."""
    return _validate_k_elite_ams_parameters(parameters, token="K_ELITE_AMS_DEB_SI_DESC_COLD_START")


def _validate_k_elite_ams_parameters(
    parameters: Mapping[str, JSONValue],
    *,
    token: str,
) -> dict[str, JSONValue]:
    """Reject an oversized parent count or a probability outside ``[0, 1]``."""
    archive_size = parameters["archive_size"]
    n_parents = parameters["n_parents"]
    if int(n_parents) > int(archive_size):
        raise RegistryError(
            f"{token}: n_parents must be <= archive_size, "
            f"got n_parents={n_parents!r} and archive_size={archive_size!r}"
        )
    for name in ("mutation_probability", "random_restart_probability"):
        _require_unit_interval(name, parameters[name], error=RegistryError)
    return dict(parameters)


def select_rank_weighted(
    members: list[EliteMember],
    n_parents: int,
    rng: np.random.Generator,
) -> list[EliteMember]:
    """
    Sample ``n_parents`` distinct elites without replacement.

    Members are ranked by ``(f, x)``, best first. Rank ``r`` (1-based)
    has weight ``1/r``. Those weights stay fixed while survivors are
    renormalised. The draw does not fall back to a shorter parent set.

    Parameters
    ----------
    members : list of EliteMember
        Current archive. Not modified.
    n_parents : int
        Exact number of parents. Must be at least 1 and at most
        ``len(members)``.
    rng : numpy.random.Generator
        Source of the weighted draws.

    Returns
    -------
    list of EliteMember
        Selected parents, in draw order.

    Raises
    ------
    ValueError
        If ``n_parents`` is not a positive integer, or the archive is
        shorter than ``n_parents``.
    """
    count = _require_positive_int("n_parents", n_parents)
    if len(members) < count:
        raise ValueError(
            f"k-elite selection needs {count} archive members, got {len(members)}"
        )
    order = sorted(
        range(len(members)),
        key=lambda index: (members[index].point.f, members[index].point.x),
    )
    weights = np.array(
        [1.0 / float(rank + 1) for rank in range(len(order))],
        dtype=np.float64,
    )
    available = np.ones(len(order), dtype=bool)
    chosen: list[EliteMember] = []
    for _ in range(count):
        masked = np.where(available, weights, 0.0)
        masked /= float(masked.sum())
        pick = int(rng.choice(len(order), p=masked))
        available[pick] = False
        chosen.append(members[order[pick]])
    return chosen


class _IndexVectorPoint(Protocol):
    """Evaluated record that stores an integer index vector."""

    x: tuple[int, ...]


class _IndexedParent(Protocol):
    """Parent record that exposes an integer index vector at ``point.x``."""

    point: _IndexVectorPoint


def inherit_parent_coordinates(
    parents: Sequence[_IndexedParent],
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """
    Copy each coordinate from one parent drawn uniformly.

    Rank weighting is not applied again. Every selected parent has equal
    influence on every coordinate.

    Parameters
    ----------
    parents : list of EliteMember
        Non-empty parent set from :func:`select_rank_weighted`.
    rng : numpy.random.Generator
        Source of one parent index per coordinate.

    Returns
    -------
    tuple of int
        Child index vector.

    Raises
    ------
    ValueError
        If ``parents`` is empty.
    """
    if not parents:
        raise ValueError("inheritance needs at least one parent")
    width = len(parents)
    dimension = len(parents[0].point.x)
    child: list[int] = []
    for axis in range(dimension):
        source = int(rng.integers(0, width))
        child.append(int(parents[source].point.x[axis]))
    return tuple(child)


def mutate_ordinal(
    x: tuple[int, ...],
    space: DiscreteSearchSpace,
    rng: np.random.Generator,
    mutation_probability: float,
) -> tuple[int, ...]:
    """
    Apply an independent valid ``±1`` step on each non-fixed coordinate.

    A fixed axis consumes no draw. Every other axis consumes one
    Bernoulli trial. A triggered mutation always changes that coordinate:
    a bound uses its only legal step and does not draw a sign; an
    interior coordinate draws the sign uniformly.

    Parameters
    ----------
    x : tuple of int
        Index vector to copy and mutate.
    space : DiscreteSearchSpace
        Inclusive bounds.
    rng : numpy.random.Generator
        Bernoulli trials and interior signs.
    mutation_probability : float
        Probability in ``[0, 1]`` for each non-fixed coordinate.

    Returns
    -------
    tuple of int
        Mutated index vector.
    """
    probability = _require_unit_interval("mutation_probability", mutation_probability)
    coords = list(space.validate(x))
    for axis in range(space.dimension):
        low = int(space.lower[axis])
        high = int(space.upper[axis])
        if low == high:
            continue
        if float(rng.random()) >= probability:
            continue
        current = coords[axis]
        if current == low:
            coords[axis] = current + 1
        elif current == high:
            coords[axis] = current - 1
        else:
            sign = int(rng.choice(np.array([-1, 1], dtype=np.int64)))
            coords[axis] = current + sign
    return tuple(coords)


def update_elite_archive(
    book: ArchiveBook,
    result: LocalSearchResult,
    archive_size: int,
) -> None:
    """
    Admit a certified endpoint into the top-``archive_size`` archive.

    An uncertified endpoint changes nothing. A current member's index is
    a duplicate. A previously evicted index is not a duplicate: it is
    scored again under ``(f, x)`` and may be readmitted. Its original id
    is kept. A new certified index receives an id even when it is
    rejected from the archive.

    Parameters
    ----------
    book : ArchiveBook
        Archive and seen-minimum book. Updated in place.
    result : LocalSearchResult
        Endpoint of one local search.
    archive_size : int
        Maximum number of members to retain. Must be at least 1.

    Raises
    ------
    ValueError
        If ``archive_size`` is not a positive integer.
    """
    capacity = _require_positive_int("archive_size", archive_size)
    if not result.certified_local_optimum:
        return
    index_vector = result.point.x
    if any(member.point.x == index_vector for member in book.members):
        book.n_duplicates += 1
        return
    if index_vector not in book.ids_by_x:
        book.ids_by_x[index_vector] = book.next_id
        book.next_id += 1
    member = EliteMember(book.ids_by_x[index_vector], result.point)
    book.members.append(member)
    book.members.sort(key=lambda item: (item.point.f, item.point.x))
    if len(book.members) > capacity:
        book.members = book.members[:capacity]
    if any(item.point.x == index_vector for item in book.members):
        book.n_admissions += 1
    else:
        book.n_rejections += 1


def take_unseen_start(
    space: DiscreteSearchSpace,
    rng: np.random.Generator,
    seen_starts: set[tuple[int, ...]],
    consecutive_duplicates: int,
) -> tuple[tuple[int, ...] | None, int]:
    """
    Draw a start index that this run has not already used.

    Duplicate draws are free. The consecutive-duplicate counter resets
    to zero when an unseen start is returned. It is not a count of all
    duplicates in the run. ``None`` means 1000 consecutive duplicates
    were drawn.

    Parameters
    ----------
    space : DiscreteSearchSpace
        Inclusive integer box.
    rng : numpy.random.Generator
        Uniform draw source.
    seen_starts : set of tuple of int
        Start indices already accepted. An unseen draw is inserted.
    consecutive_duplicates : int
        Duplicates already drawn since the last unseen start.

    Returns
    -------
    tuple of int or None, int
        The new start and a reset counter, or ``None`` and the cap count.
    """
    consecutive = int(consecutive_duplicates)
    while consecutive < _CONSECUTIVE_DUPLICATE_START_CAP:
        draw = _uniform_index(space, rng)
        if draw in seen_starts:
            consecutive += 1
            continue
        seen_starts.add(draw)
        return draw, 0
    return None, consecutive


def initial_local_searches(
    book: ArchiveBook,
    evaluator: EvaluationManager,
    space: DiscreteSearchSpace,
    rng: np.random.Generator,
    n_parents: int,
    archive_size: int,
    stats: dict[str, Any],
    *,
    local_search: Any = None,
) -> None:
    """
    Search uniform starts until the archive can supply ``n_parents``.

    Stops when the archive reaches ``n_parents``, the controller stops,
    or 1000 consecutive duplicate starts are drawn. Does not keep
    sampling in order to fill ``archive_size``.

    Parameters
    ----------
    book : ArchiveBook
        Archive updated with certified endpoints.
    evaluator : EvaluationManager
        Budgeted evaluation path.
    space : DiscreteSearchSpace
        Integer box.
    rng : numpy.random.Generator
        Uniform start source.
    n_parents : int
        Startup requirement.
    archive_size : int
        Archive capacity passed to each update.
    stats : dict
        Diagnostic counters updated in place.
    """
    search = best_improvement_local_search if local_search is None else local_search
    required = _require_positive_int("n_parents", n_parents)
    seen_starts: set[tuple[int, ...]] = set()
    consecutive = 0
    while len(book.members) < required and evaluator.has_budget():
        draw, consecutive = take_unseen_start(
            space,
            rng,
            seen_starts,
            consecutive,
        )
        if draw is None:
            break
        if not evaluator.has_budget():
            break
        start = evaluator.evaluate(draw)
        result = search(start, evaluator, space)
        _note_search(stats, result, initial=True)
        update_elite_archive(book, result, archive_size)


def run_k_elite_loop(
    book: ArchiveBook,
    evaluator: EvaluationManager,
    space: DiscreteSearchSpace,
    rng: np.random.Generator,
    n_parents: int,
    archive_size: int,
    mutation_probability: float,
    random_restart_probability: float,
    stats: dict[str, Any],
    *,
    local_search: Any = None,
) -> None:
    """
    Propose starts from exactly ``n_parents`` elites and search them.

    Parameters
    ----------
    book : ArchiveBook
        Archive with at least ``n_parents`` members.
    evaluator : EvaluationManager
        Budgeted evaluation path.
    space : DiscreteSearchSpace
        Integer box.
    rng : numpy.random.Generator
        Restart tests, parent draws, inheritance, and mutation.
    n_parents : int
        Exact parent count. The archive must already be at least this long.
    archive_size : int
        Archive capacity.
    mutation_probability : float
        Per-coordinate mutation probability.
    random_restart_probability : float
        Probability of a uniform start instead of a k-elite child.
    stats : dict
        Diagnostic counters updated in place.

    Raises
    ------
    ValueError
        If the archive holds fewer than ``n_parents`` members.
    """
    required = _require_positive_int("n_parents", n_parents)
    if len(book.members) < required:
        raise ValueError(
            f"k-elite loop needs {required} archive members, got {len(book.members)}"
        )
    restart_probability = _require_unit_interval(
        "random_restart_probability",
        random_restart_probability,
    )
    search = best_improvement_local_search if local_search is None else local_search
    while evaluator.has_budget():
        if float(rng.random()) < restart_probability:
            stats["n_random_restarts"] += 1
            child = _uniform_index(space, rng)
        else:
            stats["n_k_elite_proposals"] += 1
            parents = select_rank_weighted(book.members, required, rng)
            child = inherit_parent_coordinates(parents, rng)
            child = mutate_ordinal(child, space, rng, mutation_probability)
        if not evaluator.has_budget():
            break
        start = evaluator.evaluate(child)
        result = search(start, evaluator, space)
        _note_search(stats, result, initial=False)
        update_elite_archive(book, result, archive_size)


class KEliteAMS:
    """
    Cold-start k-elite adaptive multi-start.

    Uniform local searches run until the archive holds ``n_parents``
    certified minima. Later proposals use exactly that many rank-weighted
    parents. ``archive_size`` is only the memory capacity. The benchmark
    defaults are archive size 10, 5 parents, mutation probability 0.05,
    and random-restart probability 0.10.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int or None, optional
        Task seed. A new generator is built on every ``__call__``.
    archive_size : int, optional
        Maximum elite archive size. Default 10.
    n_parents : int, optional
        Parents per k-elite proposal, and the startup threshold. Default 5.
    mutation_probability : float, optional
        Per non-fixed coordinate. Default 0.05.
    random_restart_probability : float, optional
        Probability of a uniform main-loop start. Default 0.10.
    """

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        archive_size: int = 10,
        n_parents: int = 5,
        mutation_probability: float = 0.05,
        random_restart_probability: float = 0.10,
    ) -> None:
        self.max_evaluations = _require_positive_int(
            "max_evaluations",
            max_evaluations,
        )
        self.seed = seed
        self.archive_size = _require_positive_int("archive_size", archive_size)
        self.n_parents = _require_positive_int("n_parents", n_parents)
        if self.n_parents > self.archive_size:
            raise ValueError(
                "n_parents must be <= archive_size, "
                f"got n_parents={self.n_parents} and archive_size={self.archive_size}"
            )
        self.mutation_probability = _require_unit_interval(
            "mutation_probability",
            mutation_probability,
        )
        self.random_restart_probability = _require_unit_interval(
            "random_restart_probability",
            random_restart_probability,
        )
        self.local_search = best_improvement_local_search
        self.name = "K_ELITE_AMS_PEN_SI_DESC_COLD_START"
        self.info = (
            "k-elite AMS "
            f"M={self.archive_size} "
            f"k={self.n_parents} "
            f"p_mut={self.mutation_probability} "
            f"p_restart={self.random_restart_probability}"
        )
        self.last_run_stats: dict[str, Any] = {}

    def __call__(self, problem: Any, *, run_controller: Any = None) -> KEliteAMS:
        """
        Run one seeded call.

        Parameters
        ----------
        problem : Any
            IOH-like problem.
        run_controller : optional
            External controller. ``None`` builds a fixed-budget controller.

        Returns
        -------
        KEliteAMS
            ``self``. The benchmark reads ``problem.state``.
        """
        stats = _empty_stats(self.seed)
        self.last_run_stats = stats
        evaluator_slot: list[EvaluationManager | None] = [None]
        book_slot: list[ArchiveBook | None] = [None]
        init_recorded = [False]
        try:
            self._execute(
                problem,
                run_controller,
                stats,
                evaluator_slot,
                book_slot,
                init_recorded,
            )
        except BudgetExhausted:
            pass
        finally:
            evaluator = evaluator_slot[0]
            if evaluator is not None and not init_recorded[0]:
                stats["initialisation_evaluations"] = evaluator.n_evaluations
            if book_slot[0] is not None:
                _write_archive(stats, book_slot[0])
            if evaluator is not None:
                _write_best(stats, evaluator)
        return self

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
        evaluator_slot: list[EvaluationManager | None],
        book_slot: list[ArchiveBook | None],
        init_recorded: list[bool],
    ) -> None:
        """
        Initialise, then run the k-elite loop when the archive is ready.

        ``init_recorded[0]`` becomes true once ``initialisation_evaluations``
        has been stored, including when a later ``BudgetExhausted`` aborts
        the main loop.
        """
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        _dimension, lower, upper = read_problem_bounds(problem)
        space = DiscreteSearchSpace(lower, upper)
        evaluator = EvaluationManager(problem, controller, space)
        evaluator_slot[0] = evaluator
        rng = np.random.default_rng(self.seed)
        book = ArchiveBook()
        book_slot[0] = book
        initial_local_searches(
            book,
            evaluator,
            space,
            rng,
            self.n_parents,
            self.archive_size,
            stats,
            local_search=self.local_search,
        )
        continue_after_initialisation(
            book,
            evaluator,
            space,
            rng,
            self.n_parents,
            self.archive_size,
            self.mutation_probability,
            self.random_restart_probability,
            stats,
            init_recorded,
            local_search=self.local_search,
        )


def search_ranked_prefix(
    prefix: Sequence[tuple[int, ...]],
    book: ArchiveBook,
    evaluator: EvaluationManager,
    space: DiscreteSearchSpace,
    archive_size: int,
    stats: dict[str, Any],
    *,
    local_search: Any = None,
) -> None:
    """
    Evaluate a ranked prefix, then search those starts in file order.

    The prefix block finishes before any local search. A stop during the
    block leaves the remaining rows unevaluated. A stop during a search
    leaves later rows evaluated and unsearched.

    Parameters
    ----------
    prefix : sequence of tuple of int
        Ranked index vectors.
    book : ArchiveBook
        Archive updated with certified endpoints.
    evaluator : EvaluationManager
        Budgeted evaluation path.
    space : DiscreteSearchSpace
        Integer box.
    archive_size : int
        Archive capacity passed to each update.
    stats : dict
        Diagnostic counters updated in place.
    """
    search = best_improvement_local_search if local_search is None else local_search
    evaluated: list[EvaluatedPoint] = []
    for vector in prefix:
        if not evaluator.has_budget():
            break
        evaluated.append(evaluator.evaluate(tuple(int(value) for value in vector)))
    for start in evaluated:
        if not evaluator.has_budget():
            break
        result = search(start, evaluator, space)
        _note_search(stats, result, initial=True)
        update_elite_archive(book, result, archive_size)


def continue_after_initialisation(
    book: ArchiveBook,
    evaluator: EvaluationManager,
    space: DiscreteSearchSpace,
    rng: np.random.Generator,
    n_parents: int,
    archive_size: int,
    mutation_probability: float,
    random_restart_probability: float,
    stats: dict[str, Any],
    init_recorded: list[bool],
    *,
    local_search: Any = None,
) -> None:
    """
    Record the startup evaluation count, then start the loop if it can.

    ``init_recorded[0]`` is set before ``run_k_elite_loop``, so a later
    ``BudgetExhausted`` leaves ``initialisation_evaluations`` at the
    loop-entry count.

    Parameters
    ----------
    book : ArchiveBook
        Archive after startup.
    evaluator : EvaluationManager
        Budgeted evaluation path.
    space : DiscreteSearchSpace
        Integer box.
    rng : numpy.random.Generator
        Generator for the k-elite loop. Startup must not have drawn from it.
    n_parents : int
        Startup requirement and exact parent count.
    archive_size : int
        Archive capacity.
    mutation_probability : float
        Per-coordinate mutation probability.
    random_restart_probability : float
        Probability of a uniform main-loop start.
    stats : dict
        Diagnostic counters updated in place.
    init_recorded : list of bool
        One-element flag shared with ``KEliteAMS.__call__``.
    """
    stats["initialisation_evaluations"] = evaluator.n_evaluations
    init_recorded[0] = True
    if (not evaluator.has_budget()) or len(book.members) < n_parents:
        return
    stats["initial_archive_size"] = len(book.members)
    run_k_elite_loop(
        book,
        evaluator,
        space,
        rng,
        n_parents,
        archive_size,
        mutation_probability,
        random_restart_probability,
        stats,
        local_search=local_search,
    )


class KEliteAMSWarmStart(KEliteAMS):
    """
    k-elite adaptive multi-start from a ranked InitialDesign prefix.

        The prefix length is ``n_parents``. Those rows are evaluated before
        any local search. If they yield fewer than ``n_parents`` distinct
        certified minima, uniform local searches fill the archive up to
        that count. The k-elite loop then runs when the archive holds
        ``n_parents`` members.

    Parameters
    ----------
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        archive_size: int = 10,
        n_parents: int = 5,
        mutation_probability: float = 0.05,
        random_restart_probability: float = 0.10,
        *,
        initial_design_path: str,
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            archive_size=archive_size,
            n_parents=n_parents,
            mutation_probability=mutation_probability,
            random_restart_probability=random_restart_probability,
        )
        if not isinstance(initial_design_path, str) or not initial_design_path:
            raise ValueError("K_ELITE_AMS_PEN_SI_DESC_WARM_START requires initial_design_path")
        self.initial_design_path = initial_design_path
        self.name = "K_ELITE_AMS_PEN_SI_DESC_WARM_START"

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
        evaluator_slot: list[EvaluationManager | None],
        book_slot: list[ArchiveBook | None],
        init_recorded: list[bool],
    ) -> None:
        """
        Load the prefix, search it, then run the cold loop when ready.
        """
        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        _dimension, lower, upper = read_problem_bounds(problem)
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=self.n_parents,
            max_evaluations=self.max_evaluations,
            lower_bounds=lower,
            upper_bounds=upper,
        )
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        space = DiscreteSearchSpace(lower, upper)
        evaluator = EvaluationManager(problem, controller, space)
        evaluator_slot[0] = evaluator
        rng = np.random.default_rng(self.seed)
        book = ArchiveBook()
        book_slot[0] = book
        search_ranked_prefix(
            prefix,
            book,
            evaluator,
            space,
            self.archive_size,
            stats,
            local_search=self.local_search,
        )
        if evaluator.has_budget() and len(book.members) < self.n_parents:
            initial_local_searches(
                book,
                evaluator,
                space,
                rng,
                self.n_parents,
                self.archive_size,
                stats,
                local_search=self.local_search,
            )
        continue_after_initialisation(
            book,
            evaluator,
            space,
            rng,
            self.n_parents,
            self.archive_size,
            self.mutation_probability,
            self.random_restart_probability,
            stats,
            init_recorded,
            local_search=self.local_search,
        )


class KEliteAMSEquDesc(KEliteAMS):
    """k-elite adaptive multi-start whose inner local search is equal descent."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        archive_size: int = 10,
        n_parents: int = 5,
        mutation_probability: float = 0.05,
        random_restart_probability: float = 0.10,
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            archive_size=archive_size,
            n_parents=n_parents,
            mutation_probability=mutation_probability,
            random_restart_probability=random_restart_probability,
        )
        self.name = "K_ELITE_AMS_PEN_EQU_DESC_COLD_START"
        self.local_search = equal_descent_local_search
        self.info = (
            "k-elite AMS-EquDesc "
            f"M={self.archive_size} "
            f"k={self.n_parents} "
            f"p_mut={self.mutation_probability} "
            f"p_restart={self.random_restart_probability}"
        )


class KEliteAMSEquDescWarmStart(KEliteAMSWarmStart):
    """Equal-descent k-elite AMS from a ranked InitialDesign prefix."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        archive_size: int = 10,
        n_parents: int = 5,
        mutation_probability: float = 0.05,
        random_restart_probability: float = 0.10,
        *,
        initial_design_path: str,
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            archive_size=archive_size,
            n_parents=n_parents,
            mutation_probability=mutation_probability,
            random_restart_probability=random_restart_probability,
            initial_design_path=initial_design_path,
        )
        self.name = "K_ELITE_AMS_PEN_EQU_DESC_WARM_START"
        self.local_search = equal_descent_local_search
        self.info = (
            "k-elite AMS-EquDesc "
            f"M={self.archive_size} "
            f"k={self.n_parents} "
            f"p_mut={self.mutation_probability} "
            f"p_restart={self.random_restart_probability}"
        )


_DEB_AMS_LABEL = "k-elite AMS-Deb"


@dataclass(frozen=True)
class DebEliteMember:
    """One certified Deb local minimum remembered by the archive."""

    minimum_id: int
    point: DebEvaluatedPoint


@dataclass
class DebArchiveBook:
    """
    Active Deb elite archive.

    Members stay ordered by ``(deb_point_key, x)``. An evicted index keeps
    the id in ``ids_by_x``. Boundary-tie counters record lexicographic
    representative bias at capacity; they do not change that truncation.
    """

    members: list[DebEliteMember] = field(default_factory=list)
    ids_by_x: dict[tuple[int, ...], int] = field(default_factory=dict)
    next_id: int = 0
    n_admissions: int = 0
    n_duplicates: int = 0
    n_rejections: int = 0
    n_boundary_tie_events: int = 0
    n_boundary_tie_candidate_rejections: int = 0
    n_boundary_tie_existing_evictions: int = 0


def _deb_member_sort_key(
    member: DebEliteMember,
) -> tuple[tuple[int, float], tuple[int, ...]]:
    """Return the deterministic archive order key."""
    return (deb_point_key(member.point), member.point.x)


def _dense_rank_weights(
    members: Sequence[DebEliteMember],
) -> tuple[list[int], np.ndarray]:
    """
    Return sorted positions and equal dense-rank weights.

    Tied Deb keys share one rank and therefore one ``1/r`` weight. The
    next distinct key receives the next integer rank. ``x`` orders
    representatives inside a tie but does not change those weights.
    """
    order = sorted(
        range(len(members)),
        key=lambda index: _deb_member_sort_key(members[index]),
    )
    weights = np.empty(len(order), dtype=np.float64)
    rank = 0
    previous: tuple[int, float] | None = None
    for position, index in enumerate(order):
        key = deb_point_key(members[index].point)
        if previous is None or key != previous:
            rank += 1
            previous = key
        weights[position] = 1.0 / float(rank)
    return order, weights


def select_rank_weighted_deb(
    members: list[DebEliteMember],
    n_parents: int,
    rng: np.random.Generator,
) -> list[DebEliteMember]:
    """
    Sample ``n_parents`` distinct Deb elites without replacement.

    Dense rank ``r`` of the Deb key has weight ``1/r``. Tied keys share
    that weight, so a repeated quality is more likely to be drawn than
    one elite of the same quality would be. That multiplicity bias is
    intentional. When every key is unique, the weights and, for the same
    quality order and RNG, the selected ids match historical AMS.

    Parameters
    ----------
    members : list of DebEliteMember
        Current archive. Not modified.
    n_parents : int
        Exact number of parents. Must be at least 1 and at most
        ``len(members)``.
    rng : numpy.random.Generator
        Source of the weighted draws.

    Returns
    -------
    list of DebEliteMember
        Selected parents, in draw order.

    Raises
    ------
    ValueError
        If ``n_parents`` is not a positive integer, or the archive is
        shorter than ``n_parents``.
    """
    count = _require_positive_int("n_parents", n_parents)
    if len(members) < count:
        raise ValueError(
            f"k-elite selection needs {count} archive members, got {len(members)}"
        )
    order, weights = _dense_rank_weights(members)
    available = np.ones(len(order), dtype=bool)
    chosen: list[DebEliteMember] = []
    for _ in range(count):
        masked = np.where(available, weights, 0.0)
        masked /= float(masked.sum())
        pick = int(rng.choice(len(order), p=masked))
        available[pick] = False
        chosen.append(members[order[pick]])
    return chosen


def update_elite_archive_deb(
    book: DebArchiveBook,
    result: DebLocalSearchResult,
    archive_size: int,
) -> None:
    """
    Admit a certified Deb endpoint into the top-``archive_size`` archive.

    Uncertified endpoints, including failed analyses, change nothing.
    A current member's index is a duplicate. Capacity truncation keeps
    the smallest ``x`` among equal Deb keys and never grows past
    ``archive_size``. A boundary tie is the cut between index ``M-1``
    and index ``M`` after that sort.

    Parameters
    ----------
    book : DebArchiveBook
        Archive and seen-minimum book. Updated in place.
    result : DebLocalSearchResult
        Endpoint of one Deb local search.
    archive_size : int
        Maximum number of members to retain. Must be at least 1.

    Raises
    ------
    ValueError
        If ``archive_size`` is not a positive integer.
    """
    capacity = _require_positive_int("archive_size", archive_size)
    if not result.certified_local_optimum:
        return
    index_vector = result.point.x
    if any(member.point.x == index_vector for member in book.members):
        book.n_duplicates += 1
        return
    if index_vector not in book.ids_by_x:
        book.ids_by_x[index_vector] = book.next_id
        book.next_id += 1
    incoming = DebEliteMember(book.ids_by_x[index_vector], result.point)
    book.members.append(incoming)
    book.members.sort(key=_deb_member_sort_key)
    if len(book.members) > capacity:
        retained = book.members[capacity - 1]
        excluded = book.members[capacity]
        if deb_point_key(retained.point) == deb_point_key(excluded.point):
            book.n_boundary_tie_events += 1
            if excluded.point.x == index_vector:
                book.n_boundary_tie_candidate_rejections += 1
            else:
                book.n_boundary_tie_existing_evictions += 1
        book.members = book.members[:capacity]
    if any(item.point.x == index_vector for item in book.members):
        book.n_admissions += 1
    else:
        book.n_rejections += 1


def initial_local_searches_deb(
    book: DebArchiveBook,
    evaluator: DebEvaluationManager,
    space: DiscreteSearchSpace,
    rng: np.random.Generator,
    n_parents: int,
    archive_size: int,
    stats: dict[str, Any],
    local_search: Any = None,
) -> None:
    """
    Search uniform starts until the Deb archive can supply ``n_parents``.

    A target-hitting start returns immediately and is not searched.
    Otherwise a failed start is still a search location. The duplicate
    cap counts repeated start vectors, not convergence to one minimum.

    Parameters
    ----------
    book : DebArchiveBook
        Archive updated with certified endpoints.
    evaluator : DebEvaluationManager
        Budgeted Deb evaluation path.
    space : DiscreteSearchSpace
        Integer box.
    rng : numpy.random.Generator
        Uniform start source.
    n_parents : int
        Startup requirement.
    archive_size : int
        Archive capacity passed to each update.
    stats : dict
        Diagnostic counters updated in place.
    """
    search = (
        deb_best_improvement_local_search
        if local_search is None
        else local_search
    )
    required = _require_positive_int("n_parents", n_parents)
    seen_starts: set[tuple[int, ...]] = set()
    consecutive = 0
    while len(book.members) < required and evaluator.has_budget():
        draw, consecutive = take_unseen_start(
            space,
            rng,
            seen_starts,
            consecutive,
        )
        if draw is None:
            break
        if not evaluator.has_budget():
            break
        start = evaluator.evaluate(draw)
        _note_start_deb(stats, initial=True)
        if evaluator.controller.target_hit:
            return
        result = search(start, evaluator, space)
        _note_search_deb(stats, result, initial=True)
        update_elite_archive_deb(book, result, archive_size)


def run_k_elite_loop_deb(
    book: DebArchiveBook,
    evaluator: DebEvaluationManager,
    space: DiscreteSearchSpace,
    rng: np.random.Generator,
    n_parents: int,
    archive_size: int,
    mutation_probability: float,
    random_restart_probability: float,
    stats: dict[str, Any],
    local_search: Any = None,
) -> None:
    """
    Propose Deb-ranked starts and search every proposal, including duplicates.

    Parameters
    ----------
    book : DebArchiveBook
        Archive with at least ``n_parents`` members.
    evaluator : DebEvaluationManager
        Budgeted Deb evaluation path.
    space : DiscreteSearchSpace
        Integer box.
    rng : numpy.random.Generator
        Restart tests, parent draws, inheritance, and mutation.
    n_parents : int
        Exact parent count.
    archive_size : int
        Archive capacity.
    mutation_probability : float
        Per-coordinate mutation probability.
    random_restart_probability : float
        Probability of a uniform start instead of a k-elite child.
    stats : dict
        Diagnostic counters updated in place.

    Raises
    ------
    ValueError
        If the archive holds fewer than ``n_parents`` members.
    """
    required = _require_positive_int("n_parents", n_parents)
    if len(book.members) < required:
        raise ValueError(
            f"k-elite loop needs {required} archive members, got {len(book.members)}"
        )
    restart_probability = _require_unit_interval(
        "random_restart_probability",
        random_restart_probability,
    )
    while evaluator.has_budget():
        search = (
            deb_best_improvement_local_search
            if local_search is None
            else local_search
        )
        if float(rng.random()) < restart_probability:
            stats["n_random_restarts"] += 1
            child = _uniform_index(space, rng)
        else:
            stats["n_k_elite_proposals"] += 1
            parents = select_rank_weighted_deb(book.members, required, rng)
            child = inherit_parent_coordinates(parents, rng)
            child = mutate_ordinal(child, space, rng, mutation_probability)
        if not evaluator.has_budget():
            break
        start = evaluator.evaluate(child)
        _note_start_deb(stats, initial=False)
        if evaluator.controller.target_hit:
            return
        result = search(start, evaluator, space)
        _note_search_deb(stats, result, initial=False)
        update_elite_archive_deb(book, result, archive_size)


def search_ranked_prefix_deb(
    prefix: Sequence[tuple[int, ...]],
    book: DebArchiveBook,
    evaluator: DebEvaluationManager,
    space: DiscreteSearchSpace,
    archive_size: int,
    stats: dict[str, Any],
    local_search: Any = None,
) -> None:
    """
    Evaluate a ``y_p``-ranked prefix, then Deb-search those starts in file order.

    A target hit during the block stops before later rows, local search,
    and archive admission.

    Parameters
    ----------
    prefix : sequence of tuple of int
        ``y_p``-ranked index vectors, in file order.
    book : DebArchiveBook
        Archive updated with certified endpoints.
    evaluator : DebEvaluationManager
        Budgeted Deb evaluation path.
    space : DiscreteSearchSpace
        Integer box.
    archive_size : int
        Archive capacity passed to each update.
    stats : dict
        Diagnostic counters updated in place.
    """
    search = (
        deb_best_improvement_local_search
        if local_search is None
        else local_search
    )
    evaluated: list[DebEvaluatedPoint] = []
    for vector in prefix:
        if not evaluator.has_budget():
            break
        point = evaluator.evaluate(tuple(int(value) for value in vector))
        _note_start_deb(stats, initial=True)
        if evaluator.controller.target_hit:
            return
        evaluated.append(point)
    for start in evaluated:
        if not evaluator.has_budget():
            break
        result = search(start, evaluator, space)
        _note_search_deb(stats, result, initial=True)
        update_elite_archive_deb(book, result, archive_size)


def continue_after_initialisation_deb(
    book: DebArchiveBook,
    evaluator: DebEvaluationManager,
    space: DiscreteSearchSpace,
    rng: np.random.Generator,
    n_parents: int,
    archive_size: int,
    mutation_probability: float,
    random_restart_probability: float,
    stats: dict[str, Any],
    init_recorded: list[bool],
    local_search: Any = None,
) -> None:
    """
    Record startup archive size, then enter the Deb loop when it can start.

    ``startup_*`` is stored even when the archive is short of ``n_parents``.
    ``initial_*`` stays zero unless the main loop starts.

    Parameters
    ----------
    book : DebArchiveBook
        Archive after startup.
    evaluator : DebEvaluationManager
        Budgeted Deb evaluation path.
    space : DiscreteSearchSpace
        Integer box.
    rng : numpy.random.Generator
        Generator for the k-elite loop.
    n_parents : int
        Startup requirement and exact parent count.
    archive_size : int
        Archive capacity.
    mutation_probability : float
        Per-coordinate mutation probability.
    random_restart_probability : float
        Probability of a uniform main-loop start.
    stats : dict
        Diagnostic counters updated in place.
    init_recorded : list of bool
        One-element flag shared with ``KEliteAMSDeb.__call__``.
    """
    _record_startup_archive(stats, book)
    stats["initialisation_evaluations"] = evaluator.n_evaluations
    init_recorded[0] = True
    if (not evaluator.has_budget()) or len(book.members) < n_parents:
        return
    feasible, infeasible = _feasibility_split(book.members)
    stats["initial_archive_size"] = len(book.members)
    stats["initial_feasible_archive_size"] = feasible
    stats["initial_infeasible_archive_size"] = infeasible
    run_k_elite_loop_deb(
        book,
        evaluator,
        space,
        rng,
        n_parents,
        archive_size,
        mutation_probability,
        random_restart_probability,
        stats,
        local_search=local_search,
    )


def _deb_info(algorithm: Any, label: str) -> str:
    """Return the public info string for a Deb AMS variant."""
    return (
        f"{label} "
        f"M={algorithm.archive_size} "
        f"k={algorithm.n_parents} "
        f"p_mut={algorithm.mutation_probability} "
        f"p_restart={algorithm.random_restart_probability}"
    )


class KEliteAMSDeb:
    """
    Cold-start k-elite AMS ordered by Deb feasibility.

    The benchmark still observes penalised ``y_p``. Deb ordering is only
    the search and archive policy. Archive members are certified endpoint
    vectors, not catalogue CoLON nodes.
    """

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        archive_size: int = 10,
        n_parents: int = 5,
        mutation_probability: float = 0.05,
        random_restart_probability: float = 0.10,
    ) -> None:
        self.max_evaluations = _require_positive_int(
            "max_evaluations",
            max_evaluations,
        )
        self.seed = seed
        self.archive_size = _require_positive_int("archive_size", archive_size)
        self.n_parents = _require_positive_int("n_parents", n_parents)
        if self.n_parents > self.archive_size:
            raise ValueError(
                "n_parents must be <= archive_size, "
                f"got n_parents={self.n_parents} and archive_size={self.archive_size}"
            )
        self.mutation_probability = _require_unit_interval(
            "mutation_probability",
            mutation_probability,
        )
        self.random_restart_probability = _require_unit_interval(
            "random_restart_probability",
            random_restart_probability,
        )
        self.name = "K_ELITE_AMS_DEB_SI_DESC_COLD_START"
        self.local_search = deb_best_improvement_local_search
        self.info = _deb_info(self, _DEB_AMS_LABEL)
        self.last_run_stats: dict[str, Any] = {}
        self._summary: dict[str, JSONValue] = {}

    def __call__(
        self,
        problem: Any,
        *,
        run_controller: Any = None,
    ) -> KEliteAMSDeb:
        """Run one seeded call. ``BudgetExhausted`` does not escape."""
        stats = _empty_deb_stats(self.seed)
        self.last_run_stats = stats
        evaluator_slot: list[DebEvaluationManager | None] = [None]
        book_slot: list[DebArchiveBook | None] = [None]
        init_recorded = [False]
        try:
            self._execute(
                problem,
                run_controller,
                stats,
                evaluator_slot,
                book_slot,
                init_recorded,
            )
        except BudgetExhausted:
            pass
        finally:
            evaluator = evaluator_slot[0]
            if evaluator is not None and not init_recorded[0]:
                stats["initialisation_evaluations"] = evaluator.n_evaluations
            if book_slot[0] is not None:
                _write_deb_archive(stats, book_slot[0])
            if evaluator is not None:
                _write_deb_outcome(stats, evaluator)
            self._summary = _build_deb_summary(stats)
        return self

    def summary_diagnostics(self) -> Mapping[str, JSONValue]:
        """Return compact finite diagnostics, without the archive listing."""
        return dict(self._summary)

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
        evaluator_slot: list[DebEvaluationManager | None],
        book_slot: list[DebArchiveBook | None],
        init_recorded: list[bool],
    ) -> None:
        """Initialise from uniform starts, then run the Deb k-elite loop."""
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        _dimension, lower, upper = read_problem_bounds(problem)
        space = DiscreteSearchSpace(lower, upper)
        evaluator = DebEvaluationManager(
            problem,
            controller,
            space,
            algorithm_label=_DEB_AMS_LABEL,
        )
        evaluator_slot[0] = evaluator
        rng = np.random.default_rng(self.seed)
        book = DebArchiveBook()
        book_slot[0] = book
        initial_local_searches_deb(
            book,
            evaluator,
            space,
            rng,
            self.n_parents,
            self.archive_size,
            stats,
            local_search=self.local_search,
        )
        continue_after_initialisation_deb(
            book,
            evaluator,
            space,
            rng,
            self.n_parents,
            self.archive_size,
            self.mutation_probability,
            self.random_restart_probability,
            stats,
            init_recorded,
            local_search=self.local_search,
        )


class KEliteAMSDebWarmStart(KEliteAMSDeb):
    """Deb k-elite AMS from a production-``y_p``-ranked InitialDesign prefix."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        archive_size: int = 10,
        n_parents: int = 5,
        mutation_probability: float = 0.05,
        random_restart_probability: float = 0.10,
        *,
        initial_design_path: str,
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            archive_size=archive_size,
            n_parents=n_parents,
            mutation_probability=mutation_probability,
            random_restart_probability=random_restart_probability,
        )
        if not isinstance(initial_design_path, str) or not initial_design_path:
            raise ValueError(
                "K_ELITE_AMS_DEB_SI_DESC_WARM_START requires initial_design_path"
            )
        self.initial_design_path = initial_design_path
        self.name = "K_ELITE_AMS_DEB_SI_DESC_WARM_START"
        self.info = _deb_info(self, "k-elite AMS-Deb-WarmStart")

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
        evaluator_slot: list[DebEvaluationManager | None],
        book_slot: list[DebArchiveBook | None],
        init_recorded: list[bool],
    ) -> None:
        """Load the ``y_p`` prefix before any evaluation, then search it."""
        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        _dimension, lower, upper = read_problem_bounds(problem)
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=self.n_parents,
            max_evaluations=self.max_evaluations,
            lower_bounds=lower,
            upper_bounds=upper,
        )
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        space = DiscreteSearchSpace(lower, upper)
        evaluator = DebEvaluationManager(
            problem,
            controller,
            space,
            algorithm_label=_DEB_AMS_LABEL,
        )
        evaluator_slot[0] = evaluator
        rng = np.random.default_rng(self.seed)
        book = DebArchiveBook()
        book_slot[0] = book
        search_ranked_prefix_deb(
            prefix,
            book,
            evaluator,
            space,
            self.archive_size,
            stats,
            local_search=self.local_search,
        )
        if evaluator.has_budget() and len(book.members) < self.n_parents:
            initial_local_searches_deb(
                book,
                evaluator,
                space,
                rng,
                self.n_parents,
                self.archive_size,
                stats,
                local_search=self.local_search,
            )
        continue_after_initialisation_deb(
            book,
            evaluator,
            space,
            rng,
            self.n_parents,
            self.archive_size,
            self.mutation_probability,
            self.random_restart_probability,
            stats,
            init_recorded,
            local_search=self.local_search,
        )


class KEliteAMSDebEquDesc(KEliteAMSDeb):
    """k-elite AMS whose Deb local search is equal descent."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        archive_size: int = 10,
        n_parents: int = 5,
        mutation_probability: float = 0.05,
        random_restart_probability: float = 0.10,
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            archive_size=archive_size,
            n_parents=n_parents,
            mutation_probability=mutation_probability,
            random_restart_probability=random_restart_probability,
        )
        self.name = "K_ELITE_AMS_DEB_EQU_DESC_COLD_START"
        self.local_search = deb_equal_descent_local_search
        self.info = _deb_info(self, "k-elite AMS-Deb-EquDesc")


class KEliteAMSDebEquDescWarmStart(KEliteAMSDebWarmStart):
    """Equal-descent Deb AMS from a Deb-ranked InitialDesign prefix."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        archive_size: int = 10,
        n_parents: int = 5,
        mutation_probability: float = 0.05,
        random_restart_probability: float = 0.10,
        *,
        initial_design_path: str,
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            archive_size=archive_size,
            n_parents=n_parents,
            mutation_probability=mutation_probability,
            random_restart_probability=random_restart_probability,
            initial_design_path=initial_design_path,
        )
        self.name = "K_ELITE_AMS_DEB_EQU_DESC_WARM_START"
        self.local_search = deb_equal_descent_local_search
        self.info = _deb_info(self, "k-elite AMS-Deb-EquDesc-WarmStart")


def _uniform_index(
    space: DiscreteSearchSpace,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Draw one inclusive integer index vector."""
    drawn = rng.integers(space.lower, space.upper + 1)
    return tuple(int(value) for value in np.asarray(drawn, dtype=np.int64).reshape(-1))


def _note_search(
    stats: dict[str, Any],
    result: LocalSearchResult,
    *,
    initial: bool,
) -> None:
    """Count one local search, and initial searches separately."""
    stats["n_local_searches"] += 1
    if initial:
        stats["n_initial_local_searches"] += 1
    if result.certified_local_optimum:
        stats["n_certified_local_searches"] += 1
    if getattr(result, "early_certificate", False):
        stats["n_equal_descent_early_certificates"] = (
            int(stats.get("n_equal_descent_early_certificates", 0)) + 1
        )


def _empty_stats(seed: int | None) -> dict[str, Any]:
    """Return the version-1 diagnostic dict."""
    return {
        "seed": seed,
        "n_evaluations": 0,
        "best_f": None,
        "best_x": None,
        "n_local_searches": 0,
        "n_certified_local_searches": 0,
        "n_initial_local_searches": 0,
        "initialisation_evaluations": 0,
        "initial_archive_size": 0,
        "n_k_elite_proposals": 0,
        "n_random_restarts": 0,
        "n_archive_admissions": 0,
        "n_archive_duplicates": 0,
        "n_archive_rejections": 0,
        "n_distinct_certified_minima_seen": 0,
        "final_archive": [],
    }


def _write_archive(stats: dict[str, Any], book: ArchiveBook) -> None:
    """Copy archive counters and the ordered elite list into diagnostics."""
    stats["n_archive_admissions"] = int(book.n_admissions)
    stats["n_archive_duplicates"] = int(book.n_duplicates)
    stats["n_archive_rejections"] = int(book.n_rejections)
    stats["n_distinct_certified_minima_seen"] = len(book.ids_by_x)
    stats["final_archive"] = [
        {
            "id": int(member.minimum_id),
            "x": [int(value) for value in member.point.x],
            "f": float(member.point.f),
        }
        for member in book.members
    ]


def _write_best(stats: dict[str, Any], evaluator: EvaluationManager) -> None:
    """Copy the run-local evaluation count and the finite incumbent."""
    stats["n_evaluations"] = evaluator.n_evaluations
    if evaluator.best_point is None:
        stats["best_f"] = None
        stats["best_x"] = None
        return
    stats["best_f"] = float(evaluator.best_point.f)
    stats["best_x"] = [int(value) for value in evaluator.best_point.x]


def _require_positive_int(name: str, value: Any) -> int:
    """Reject bool and non-positive integers."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return int(value)


def _require_unit_interval(
    name: str,
    value: Any,
    error: type[Exception] = ValueError,
) -> float:
    """Reject values outside the closed unit interval."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{name} must be in [0, 1], got {value!r}")
    number = float(value)
    if (not math.isfinite(number)) or number < 0.0 or number > 1.0:
        raise error(f"{name} must be in [0, 1], got {value!r}")
    return number


def _note_start_deb(stats: dict[str, Any], *, initial: bool) -> None:
    """Count one explicit start sent to the problem."""
    stats["n_start_evaluations"] += 1
    key = (
        "n_initial_start_evaluations"
        if initial
        else "n_main_start_evaluations"
    )
    stats[key] += 1


def _note_search_deb(
    stats: dict[str, Any],
    result: DebLocalSearchResult,
    *,
    initial: bool,
) -> None:
    """Count one Deb local search and its new neighbour evaluations."""
    if getattr(result, "early_certificate", False):
        stats["n_equal_descent_early_certificates"] = (
            int(stats.get("n_equal_descent_early_certificates", 0)) + 1
        )
    stats["n_local_searches"] += 1
    stats["n_local_search_evaluations"] += int(result.n_evaluations)
    stats["n_certification_evaluations"] += int(result.n_certification_evaluations)
    if initial:
        stats["n_initial_local_searches"] += 1
        stats["n_initial_local_search_evaluations"] += int(result.n_evaluations)
    else:
        stats["n_main_local_search_evaluations"] += int(result.n_evaluations)
    if result.certified_local_optimum:
        stats["n_certified_local_searches"] += 1
    else:
        stats["n_uncertified_local_searches"] += 1


def _feasibility_split(members: Sequence[DebEliteMember]) -> tuple[int, int]:
    """Return feasible and infeasible member counts."""
    feasible = sum(1 for member in members if member.point.feasible)
    return feasible, len(members) - feasible


def _tie_group_stats(members: Sequence[DebEliteMember]) -> tuple[int, int]:
    """Return the number of distinct Deb keys and the largest tie."""
    counts: dict[tuple[int, float], int] = {}
    for member in members:
        key = deb_point_key(member.point)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return 0, 0
    return len(counts), max(counts.values())


def _record_startup_archive(stats: dict[str, Any], book: DebArchiveBook) -> None:
    """Store the archive size after prefix search and cold fallback."""
    feasible, infeasible = _feasibility_split(book.members)
    stats["startup_archive_size"] = len(book.members)
    stats["startup_feasible_archive_size"] = feasible
    stats["startup_infeasible_archive_size"] = infeasible


def _empty_deb_stats(seed: int | None) -> dict[str, Any]:
    """Return zeroed diagnostics for one Deb AMS call."""
    stats = _empty_stats(seed)
    stats.update(
        {
            "best_deb_x": None,
            "best_deb_y": None,
            "best_deb_phi": None,
            "best_deb_feasible": None,
            "n_valid_evaluations": 0,
            "n_feasible_evaluations": 0,
            "n_infeasible_evaluations": 0,
            "n_failed_analysis_evaluations": 0,
            "n_sentinel_evaluations": 0,
            "sentinel_evaluations_by_constraint": {},
            "n_infeasible_plateau_terminations": 0,
            "n_sentinel_infeasible_plateau_terminations": 0,
            "n_start_evaluations": 0,
            "n_initial_start_evaluations": 0,
            "n_main_start_evaluations": 0,
            "n_local_search_evaluations": 0,
            "n_initial_local_search_evaluations": 0,
            "n_main_local_search_evaluations": 0,
            "n_certification_evaluations": 0,
            "n_uncertified_local_searches": 0,
            "startup_archive_size": 0,
            "startup_feasible_archive_size": 0,
            "startup_infeasible_archive_size": 0,
            "initial_feasible_archive_size": 0,
            "initial_infeasible_archive_size": 0,
            "final_archive_size": 0,
            "final_feasible_archive_size": 0,
            "final_infeasible_archive_size": 0,
            "final_dense_rank_group_count": 0,
            "final_max_tie_group_size": 0,
            "n_boundary_tie_events": 0,
            "n_boundary_tie_candidate_rejections": 0,
            "n_boundary_tie_existing_evictions": 0,
        }
    )
    return stats


def _write_deb_archive(stats: dict[str, Any], book: DebArchiveBook) -> None:
    """Copy Deb archive counters and ordered rows into diagnostics."""
    feasible, infeasible = _feasibility_split(book.members)
    groups, max_tie = _tie_group_stats(book.members)
    stats["n_archive_admissions"] = int(book.n_admissions)
    stats["n_archive_duplicates"] = int(book.n_duplicates)
    stats["n_archive_rejections"] = int(book.n_rejections)
    stats["n_distinct_certified_minima_seen"] = len(book.ids_by_x)
    stats["n_boundary_tie_events"] = int(book.n_boundary_tie_events)
    stats["n_boundary_tie_candidate_rejections"] = int(
        book.n_boundary_tie_candidate_rejections
    )
    stats["n_boundary_tie_existing_evictions"] = int(
        book.n_boundary_tie_existing_evictions
    )
    stats["final_archive_size"] = len(book.members)
    stats["final_feasible_archive_size"] = feasible
    stats["final_infeasible_archive_size"] = infeasible
    stats["final_dense_rank_group_count"] = groups
    stats["final_max_tie_group_size"] = max_tie
    stats["final_archive"] = [
        {
            "id": int(member.minimum_id),
            "x": [int(value) for value in member.point.x],
            "y_p": float(member.point.y_p),
            "y": float(member.point.y),
            "phi": float(member.point.phi),
            "feasible": bool(member.point.feasible),
            "analysis_ok": bool(member.point.analysis_ok),
        }
        for member in book.members
    ]


def _write_deb_outcome(stats: dict[str, Any], evaluator: DebEvaluationManager) -> None:
    """Copy the benchmark incumbent, the Deb incumbent, and the partition."""
    benchmark = evaluator.best_benchmark_point
    if benchmark is None:
        stats["best_f"] = None
        stats["best_x"] = None
    else:
        stats["best_f"] = float(benchmark.y_p)
        stats["best_x"] = [int(value) for value in benchmark.x]
    best = evaluator.best_deb_point
    if best is None:
        stats["best_deb_x"] = None
        stats["best_deb_y"] = None
        stats["best_deb_phi"] = None
        stats["best_deb_feasible"] = None
    else:
        stats["best_deb_x"] = [int(value) for value in best.x]
        stats["best_deb_y"] = float(best.y)
        stats["best_deb_phi"] = float(best.phi)
        stats["best_deb_feasible"] = bool(best.feasible)
    diagnostics = dict(evaluator.diagnostics)
    sentinel = diagnostics.pop("sentinel_evaluations_by_constraint")
    stats["sentinel_evaluations_by_constraint"] = {
        str(name): int(count) for name, count in sentinel.items()
    }
    for name, value in diagnostics.items():
        stats[name] = int(value)


_DEB_SUMMARY_INT_KEYS = (
    "n_evaluations",
    "n_valid_evaluations",
    "n_feasible_evaluations",
    "n_infeasible_evaluations",
    "n_failed_analysis_evaluations",
    "n_sentinel_evaluations",
    "n_infeasible_plateau_terminations",
    "n_sentinel_infeasible_plateau_terminations",
    "n_local_searches",
    "n_certified_local_searches",
    "n_uncertified_local_searches",
    "n_initial_local_searches",
    "initialisation_evaluations",
    "n_start_evaluations",
    "n_initial_start_evaluations",
    "n_main_start_evaluations",
    "n_local_search_evaluations",
    "n_initial_local_search_evaluations",
    "n_main_local_search_evaluations",
    "n_certification_evaluations",
    "startup_archive_size",
    "startup_feasible_archive_size",
    "startup_infeasible_archive_size",
    "initial_archive_size",
    "initial_feasible_archive_size",
    "initial_infeasible_archive_size",
    "final_archive_size",
    "final_feasible_archive_size",
    "final_infeasible_archive_size",
    "final_dense_rank_group_count",
    "final_max_tie_group_size",
    "n_k_elite_proposals",
    "n_random_restarts",
    "n_archive_admissions",
    "n_archive_duplicates",
    "n_archive_rejections",
    "n_distinct_certified_minima_seen",
    "n_boundary_tie_events",
    "n_boundary_tie_candidate_rejections",
    "n_boundary_tie_existing_evictions",
)


def _build_deb_summary(stats: Mapping[str, Any]) -> dict[str, JSONValue]:
    """Return the persisted diagnostic object for both Deb AMS variants."""
    summary: dict[str, JSONValue] = {
        "label": _DEB_AMS_LABEL,
        "best_f": stats.get("best_f"),
        "best_x": stats.get("best_x"),
        "best_deb_x": stats.get("best_deb_x"),
        "best_deb_y": stats.get("best_deb_y"),
        "best_deb_phi": stats.get("best_deb_phi"),
        "best_deb_feasible": stats.get("best_deb_feasible"),
    }
    for name in _DEB_SUMMARY_INT_KEYS:
        summary[name] = int(stats.get(name, 0))
    sentinel = stats.get("sentinel_evaluations_by_constraint") or {}
    summary["sentinel_evaluations_by_constraint"] = {
        str(name): int(count) for name, count in sentinel.items()
    }
    return summary
