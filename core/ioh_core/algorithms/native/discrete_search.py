"""
Discrete multi-level single-linkage, Sobol sampling, Sobol multistart, and population basin hopping.

Both algorithms minimise ``problem(x)`` on inclusive integer index bounds.
They share one L1 neighbourhood, one normalised Euclidean distance, and one
evaluation path through :class:`RunController`. Strictly improving descent is
the default local search. Equal descent is the inner search of the penalty
``EQU_DESC`` classes. Constraint-aware equal descent is the inner search of
the Deb ``EQU_DESC`` classes.

Neither algorithm reads a catalogue LON, an ILS-LON, a CoLON, a known funnel,
a known local optimum, enumeration data, or a catalogue optimum. A run uses
only points evaluated during that run.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import gamma
from scipy.stats import qmc

from core.ioh_core.algorithms.common import (
    get_attached_eval_context,
    read_last_evaluation_record,
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)
from core.ioh_core.algorithms.native.deb_record import (
    deb_quality_key,
    validate_deb_auxiliary_record,
)
from core.ioh_core.algorithms.registry import JSONValue, RegistryError


class BudgetExhausted(Exception):
    """
    Internal stop when the run controller refuses another evaluation.

    Caught at the algorithm ``__call__`` boundary. It is not the
    ``BudgetExhausted`` type used by other adapters.
    """


@dataclass(frozen=True)
class EvaluatedPoint:
    """
    One evaluated integer design.

    Attributes
    ----------
    x : tuple of int
        Inclusive catalogue-index vector. A tuple so it can be a set member.
    f : float
        Objective value returned by ``problem(x)``. May be ``+inf``.
    """

    x: tuple[int, ...]
    f: float


@dataclass
class LocalSearchResult:
    """
    Outcome of one strict-improvement L1 local search.

    Attributes
    ----------
    point : EvaluatedPoint
        Best point this search is willing to return.
    certified_local_optimum : bool
        True when every legal neighbour was resolved and none was strictly
        better.
    n_steps : int
        Number of accepted strict improvements.
    n_evaluations : int
        ``problem`` calls made inside this search. Cache hits are not counted.
    """

    point: EvaluatedPoint
    certified_local_optimum: bool
    n_steps: int
    n_evaluations: int


class DiscreteSearchSpace:
    """
    Inclusive integer index box with normalised Euclidean distance.

    Fixed axes (``upper == lower``) normalise to 0 and contribute no L1
    neighbour. The constructor does not clip or round.

    Parameters
    ----------
    lower, upper : sequence of int
        Inclusive integer bounds of equal length.
    """

    def __init__(self, lower: Any, upper: Any) -> None:
        self.lower = _integer_bound_vector(lower, "lower")
        self.upper = _integer_bound_vector(upper, "upper")
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                "lower and upper bounds must have the same shape, "
                f"got {self.lower.shape} and {self.upper.shape}"
            )
        if self.lower.size < 1:
            raise ValueError("dimension must be > 0")
        if np.any(self.lower > self.upper):
            raise ValueError("each lower bound must be <= its upper bound")
        self.dimension = int(self.lower.size)
        self.cardinality = 1
        for low, high in zip(self.lower.tolist(), self.upper.tolist()):
            self.cardinality *= int(high) - int(low) + 1
        self.n_free_dimensions = int(np.count_nonzero(self.upper > self.lower))

    def validate(self, x: Any) -> tuple[int, ...]:
        """
        Return ``x`` as an in-bounds integer tuple.

        Parameters
        ----------
        x : sequence
            Candidate index vector.

        Returns
        -------
        tuple of int
            Validated indices.

        Raises
        ------
        ValueError
            If the length, types, or bounds are wrong. Never clips.
        """
        try:
            values = list(x)
        except TypeError as exc:
            raise ValueError(f"candidate must be a sequence, got {x!r}") from exc
        if len(values) != self.dimension:
            raise ValueError(
                f"candidate length {len(values)} != dimension {self.dimension}"
            )
        out: list[int] = []
        for index, raw in enumerate(values):
            if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
                raise ValueError(
                    f"candidate[{index}] must be an integer, got {raw!r}"
                )
            value = int(raw)
            low = int(self.lower[index])
            high = int(self.upper[index])
            if value < low or value > high:
                raise ValueError(
                    f"candidate[{index}]={value} outside inclusive bounds "
                    f"[{low}, {high}]"
                )
            out.append(value)
        return tuple(out)

    def normalise(self, x: Any) -> np.ndarray:
        """
        Map ``x`` onto the unit cube. A fixed axis becomes 0.

        Parameters
        ----------
        x : sequence of int
            In-bounds index vector.

        Returns
        -------
        numpy.ndarray
            Shape ``(dimension,)`` float vector.
        """
        point = np.asarray(self.validate(x), dtype=np.float64)
        span = (self.upper - self.lower).astype(np.float64)
        scaled = np.zeros(self.dimension, dtype=np.float64)
        free = span > 0.0
        scaled[free] = (point[free] - self.lower[free]) / span[free]
        return scaled

    def distance(self, x: Any, y: Any) -> float:
        """
        Normalised Euclidean distance between two index vectors.

        Parameters
        ----------
        x, y : sequence of int
            In-bounds index vectors.

        Returns
        -------
        float
            Euclidean norm of the normalised difference.
        """
        return float(np.linalg.norm(self.normalise(x) - self.normalise(y)))

    def neighbours_l1(self, x: Any) -> list[tuple[int, ...]]:
        """
        Return in-bounds L1=1 neighbours, dimension 0 −1 then +1, and so on.

        Parameters
        ----------
        x : sequence of int
            Centre index vector.

        Returns
        -------
        list of tuple of int
            Neighbours in that fixed order. Not shuffled.
        """
        point = self.validate(x)
        coords = list(point)
        neighbours: list[tuple[int, ...]] = []
        for dim in range(self.dimension):
            current = coords[dim]
            for step in (-1, 1):
                candidate = current + step
                if candidate < int(self.lower[dim]) or candidate > int(self.upper[dim]):
                    continue
                coords[dim] = candidate
                neighbours.append(tuple(coords))
            coords[dim] = current
        return neighbours


class EvaluationManager:
    """
    The only ``problem(...)`` path used by discrete MLSL and PBH.

    Parameters
    ----------
    problem : Any
        IOH-like callable.
    controller : RunController
        Existing budget and fixed-target controller.
    space : DiscreteSearchSpace
        Integer bounds used to validate candidates.
    """

    def __init__(self, problem: Any, controller: Any, space: DiscreteSearchSpace) -> None:
        self.problem = problem
        self.controller = controller
        self.space = space
        self.best_point: EvaluatedPoint | None = None

    @property
    def n_evaluations(self) -> int:
        """Evaluations consumed, read from ``problem.state.evaluations``."""
        return int(self.controller.used)

    def has_budget(self) -> bool:
        """Return whether the controller allows another evaluation."""
        return bool(self.controller.can_evaluate())

    def evaluate(self, x: Any) -> EvaluatedPoint:
        """
        Validate, evaluate, and record one candidate.

        Parameters
        ----------
        x : sequence of int
            In-bounds index vector.

        Returns
        -------
        EvaluatedPoint
            The recorded point. ``+inf`` is kept and does not become
            ``best_point``.

        Raises
        ------
        BudgetExhausted
            If the controller refuses the call.
        RuntimeError
            If the objective is NaN or ``-inf``.
        """
        if not self.controller.can_evaluate():
            raise BudgetExhausted("evaluation budget exhausted")
        point = self.space.validate(x)
        validated = validate_integer_candidate(
            point,
            lower_bounds=[int(v) for v in self.space.lower],
            upper_bounds=[int(v) for v in self.space.upper],
        )
        self.controller.require_evaluation()
        raw = self.problem(validated)
        value = float(raw)
        self.controller.observe_evaluation(
            value,
        )
        if math.isnan(value) or (math.isinf(value) and value < 0.0):
            raise RuntimeError(
                "Objective returned NaN or -inf. "
                "The optimiser must not silently modify the benchmark objective."
            )
        record = EvaluatedPoint(tuple(validated), value)
        if math.isfinite(value) and (
            self.best_point is None or value < self.best_point.f
        ):
            self.best_point = record
        return record


class DiscreteSampler:
    """
    One Sobol or uniform stream, created for a single ``__call__``.

    Parameters
    ----------
    space : DiscreteSearchSpace
        Target integer box.
    mode : str
        ``sobol`` or ``uniform``.
    seed : int or None
        Seed forwarded to Sobol when ``mode`` is ``sobol``.
    rng : numpy.random.Generator
        Stream used for uniform draws and PBH perturbation.
    """

    def __init__(
        self,
        space: DiscreteSearchSpace,
        mode: str,
        seed: int | None,
        rng: np.random.Generator,
    ) -> None:
        if mode not in {"sobol", "uniform"}:
            raise ValueError(f"sampler must be 'sobol' or 'uniform', got {mode!r}")
        self.space = space
        self.mode = mode
        self.rng = rng
        self._sobol = None
        if mode == "sobol":
            self._sobol = qmc.Sobol(
                d=space.dimension,
                scramble=True,
                seed=None if seed is None else int(seed),
            )

    def draw(self) -> tuple[int, ...]:
        """
        Project one new sample. Does not consult the seen-set.

        Returns
        -------
        tuple of int
            Inclusive index vector.
        """
        lower = self.space.lower
        upper = self.space.upper
        if self._sobol is not None:
            unit = np.asarray(self._sobol.random(1), dtype=np.float64)[0]
            projected = lower + np.floor(unit * (upper - lower + 1)).astype(np.int64)
            projected = np.clip(projected, lower, upper)
            return tuple(int(value) for value in projected)
        drawn = self.rng.integers(lower, upper + 1)
        return tuple(int(value) for value in np.asarray(drawn, dtype=np.int64).reshape(-1))

    def next_unique(self, seen: set[tuple[int, ...]]) -> tuple[int, ...] | None:
        """
        Return an index not in ``seen``, or ``None`` when the box is full.

        Parameters
        ----------
        seen : set of tuple of int
            Indices already accepted into the MLSL global sample.

        Returns
        -------
        tuple of int or None
            A new index, or ``None`` when every legal index is already seen.
            Rejected projections are not evaluated.
        """
        while len(seen) < self.space.cardinality:
            point = self.draw()
            if point not in seen:
                return point
        return None


def best_improvement_local_search(
    start: EvaluatedPoint,
    evaluator: EvaluationManager,
    space: DiscreteSearchSpace,
) -> LocalSearchResult:
    """
    Strict-improvement best-improvement L1 hill-climb.

    The start is not evaluated again. Points already resolved in this call
    are reused and are not sent to ``problem``. The map dies with the call.

    Parameters
    ----------
    start : EvaluatedPoint
        Already evaluated incumbent.
    evaluator : EvaluationManager
        Budgeted evaluation path.
    space : DiscreteSearchSpace
        Neighbourhood and bounds.

    Returns
    -------
    LocalSearchResult
        Certified when the full neighbourhood was resolved with no strict
        improvement. A budget stop inside a neighbourhood is not certified.
    """
    known: dict[tuple[int, ...], EvaluatedPoint] = {start.x: start}
    current = start
    n_steps = 0
    n_evaluations = 0
    while True:
        resolved: list[EvaluatedPoint] = []
        incomplete = False
        for neighbour in space.neighbours_l1(current.x):
            cached = known.get(neighbour)
            if cached is not None:
                resolved.append(cached)
                continue
            if not evaluator.has_budget():
                incomplete = True
                break
            try:
                record = evaluator.evaluate(neighbour)
            except BudgetExhausted:
                incomplete = True
                break
            n_evaluations += 1
            known[neighbour] = record
            resolved.append(record)
        if incomplete:
            point = current
            if resolved:
                winner = min(resolved, key=lambda item: (item.f, item.x))
                if winner.f < current.f:
                    point = winner
                    n_steps += 1
            return LocalSearchResult(
                point=point,
                certified_local_optimum=False,
                n_steps=n_steps,
                n_evaluations=n_evaluations,
            )
        if not resolved:
            return LocalSearchResult(
                point=current,
                certified_local_optimum=True,
                n_steps=n_steps,
                n_evaluations=n_evaluations,
            )
        winner = min(resolved, key=lambda item: (item.f, item.x))
        if winner.f < current.f:
            current = winner
            n_steps += 1
            continue
        return LocalSearchResult(
            point=current,
            certified_local_optimum=True,
            n_steps=n_steps,
            n_evaluations=n_evaluations,
        )


def _finite_equal(left: float, right: float) -> bool:
    """Return whether two penalised values are finite and exactly equal."""
    return math.isfinite(left) and math.isfinite(right) and left == right


def equal_descent_local_search(
    start: EvaluatedPoint,
    evaluator: EvaluationManager,
    space: DiscreteSearchSpace,
) -> LocalSearchResult:
    """
    Level-complete equal descent on the L1 neighbourhood.

    Parallel designs stay active only while their penalised values compare
    equal. A completed level keeps every finite improving neighbour at the
    best lower value and drops every worse one. The stored design changes
    only for that strictly better value. With no strict exit, the returned
    design is the one strictly improving descent would have returned.

    The within-call cache dies with the call. ``+inf`` is never a component
    member and never a tied exit value. A finite value below ``+inf`` is a
    strict exit. The start is not evaluated again.

    An unfinished incumbent neighbourhood matches strictly improving
    descent, including when an equal neighbour was already seen. After that
    neighbourhood is complete, an unfinished first-level component with no
    strict improvement returns the incumbent certified and sets
    ``early_certificate``. A strict improvement found before that first
    level is complete does not become the returned design. After an exit
    has been accepted, a later budget stop may move only to a strictly
    better neighbour already resolved in the interrupted neighbourhood.
    """
    known: dict[tuple[int, ...], EvaluatedPoint] = {start.x: start}
    n_steps = 0
    n_evaluations = 0
    n_equal_evaluations = 0
    component_sizes: list[int] = []
    retained_branches: list[int] = []
    active: list[EvaluatedPoint] = [start]

    def finish(
        point: EvaluatedPoint,
        certified: bool,
        *,
        early: bool = False,
    ) -> LocalSearchResult:
        result = LocalSearchResult(
            point=point,
            certified_local_optimum=certified,
            n_steps=n_steps,
            n_evaluations=n_evaluations,
        )
        if early:
            result.early_certificate = True
        # Logging only. These attributes are not read by the search.
        result.n_equal_component_evaluations = n_equal_evaluations
        result.equal_component_sizes = tuple(component_sizes)
        result.retained_equal_best_branches = tuple(retained_branches)
        return result

    def interrupted(incumbent: EvaluatedPoint, partial: list[EvaluatedPoint]) -> LocalSearchResult:
        nonlocal n_steps
        better = [item for item in partial if item.f < incumbent.f]
        if not better:
            return finish(incumbent, False)
        chosen = min(better, key=lambda item: (item.f, item.x))
        n_steps += 1
        return finish(chosen, False)

    while True:
        incumbent = min(active, key=lambda item: (item.f, item.x))
        level = incumbent.f
        seeds = sorted(active, key=lambda item: (item.f, item.x))
        queue: deque[EvaluatedPoint] = deque(seeds)
        seen = {item.x for item in seeds}
        exits: list[EvaluatedPoint] = []
        exit_seen: set[tuple[int, ...]] = set()
        incumbent_done = False
        partial: list[EvaluatedPoint] = []
        incomplete = False
        while queue:
            node = queue.popleft()
            partial = []
            finished = True
            for neighbour in space.neighbours_l1(node.x):
                cached = known.get(neighbour)
                fresh = cached is None
                if cached is not None:
                    record = cached
                else:
                    if not evaluator.has_budget():
                        finished = False
                        break
                    try:
                        record = evaluator.evaluate(neighbour)
                    except BudgetExhausted:
                        finished = False
                        break
                    known[neighbour] = record
                    n_evaluations += 1
                partial.append(record)
                if _finite_equal(record.f, level):
                    if fresh:
                        n_equal_evaluations += 1
                    if record.x not in seen:
                        seen.add(record.x)
                        queue.append(record)
                elif (
                    math.isfinite(record.f)
                    and record.f < level
                    and record.x not in exit_seen
                ):
                    exit_seen.add(record.x)
                    exits.append(record)
            if node.x == incumbent.x:
                incumbent_done = finished
            if not finished:
                incomplete = True
                break
        if incomplete:
            if not incumbent_done:
                return interrupted(incumbent, partial)
            if n_steps == 0 and not exits:
                return finish(incumbent, True, early=True)
            if n_steps == 0:
                return finish(incumbent, False)
            return interrupted(incumbent, partial)
        component_sizes.append(len(seen))
        if not exits:
            return finish(incumbent, True)
        best_f = min(item.f for item in exits)
        active = sorted(
            (item for item in exits if item.f == best_f),
            key=lambda item: (item.f, item.x),
        )
        retained_branches.append(len(active))
        n_steps += 1


def critical_radius(sample_count: int, n_free_dimensions: int, sigma: float) -> float:
    """
    MLSL critical radius on the free unit cube.

    ``sigma`` is the literature factor ζ. Fixed axes are not part of
    ``n_free_dimensions``. ``(ln M) / M`` rises from M=2 to M=3 and falls
    for every integer M >= 3.

    Parameters
    ----------
    sample_count : int
        Number of accumulated global samples.
    n_free_dimensions : int
        Count of axes with ``upper > lower``.
    sigma : float
        Positive MLSL factor ζ.

    Returns
    -------
    float
        Critical radius, or ``+inf`` when M <= 1 or no axis is free.
    """
    if n_free_dimensions <= 0 or sample_count <= 1:
        return math.inf
    term = (
        float(gamma(1.0 + n_free_dimensions / 2.0))
        * float(sigma)
        * float(np.log(sample_count))
        / float(sample_count)
    )
    return float(term ** (1.0 / n_free_dimensions) / math.sqrt(math.pi))


def better_sample_within_radius(
    candidate: EvaluatedPoint,
    samples: list[EvaluatedPoint],
    radius: float,
    space: DiscreteSearchSpace,
) -> bool:
    """
    Return whether a strictly better global sample lies inside ``radius``.

    Parameters
    ----------
    candidate : EvaluatedPoint
        Reduced-sample point under test.
    samples : list of EvaluatedPoint
        Full accumulated global sample.
    radius : float
        Critical radius. Distance is compared with strict ``<``.
    space : DiscreteSearchSpace
        Normalised Euclidean distance.

    Returns
    -------
    bool
        True when suppression applies.
    """
    better = [sample for sample in samples if sample.f < candidate.f]
    if not better:
        return False
    if math.isinf(radius):
        return True
    origin = space.normalise(candidate.x)
    matrix = np.vstack([space.normalise(sample.x) for sample in better])
    distances = np.linalg.norm(matrix - origin, axis=1)
    return bool(np.any(distances < radius))


def reduce_samples_deb(
    samples: Sequence[DebEvaluatedPoint],
    reduction_fraction: float,
) -> list[DebEvaluatedPoint]:
    """
    Return the exact-size Deb-ranked reduced sample.

    Equal Deb keys use ``x`` only to select deterministic representatives
    when the reduction boundary cuts a tie.
    """
    fraction = _require_open_unit("reduction_fraction", reduction_fraction)
    count = max(1, math.ceil(fraction * len(samples)))
    return sorted(
        samples,
        key=lambda point: (deb_point_key(point), point.x),
    )[:count]


def better_deb_sample_within_radius(
    candidate: DebEvaluatedPoint,
    samples: Sequence[DebEvaluatedPoint],
    radius: float,
    space: DiscreteSearchSpace,
) -> bool:
    """Return whether a strictly Deb-better global sample is inside ``radius``."""
    better = [
        sample
        for sample in samples
        if deb_strictly_better(sample, candidate)
    ]
    if not better:
        return False
    if math.isinf(radius):
        return True
    origin = space.normalise(candidate.x)
    matrix = np.vstack([space.normalise(sample.x) for sample in better])
    distances = np.linalg.norm(matrix - origin, axis=1)
    return bool(np.any(distances < radius))


def box_perturbation(
    x: Any,
    space: DiscreteSearchSpace,
    rng: np.random.Generator,
    perturbation_fraction: float,
) -> tuple[int, ...] | None:
    """
    Draw a discrete box perturbation that is not identical to ``x``.

    Coordinates are drawn inside the truncated integer interval. One hundred
    identical draws are followed by one feasible ±1 step. ``None`` means no
    legal move exists.

    Parameters
    ----------
    x : sequence of int
        Centre index vector.
    space : DiscreteSearchSpace
        Inclusive bounds.
    rng : numpy.random.Generator
        Call-local generator.
    perturbation_fraction : float
        Relative half-width ρ. ``round`` is NumPy half-to-even.

    Returns
    -------
    tuple of int or None
        A different in-bounds vector, or ``None`` when the point cannot move.
    """
    point = space.validate(x)
    widths: list[int] = []
    for low, high in zip(space.lower.tolist(), space.upper.tolist()):
        span = int(high) - int(low)
        if span <= 0:
            widths.append(0)
        else:
            widths.append(max(1, int(np.round(perturbation_fraction * float(span)))))
    for _ in range(100):
        draw: list[int] = []
        for index, center in enumerate(point):
            low = max(int(space.lower[index]), int(center) - widths[index])
            high = min(int(space.upper[index]), int(center) + widths[index])
            draw.append(int(rng.integers(low, high + 1)))
        proposed = tuple(draw)
        if proposed != point:
            return proposed
    free: list[tuple[int, list[int]]] = []
    for index, center in enumerate(point):
        signs = []
        if int(center) - 1 >= int(space.lower[index]):
            signs.append(-1)
        if int(center) + 1 <= int(space.upper[index]):
            signs.append(1)
        if signs:
            free.append((index, signs))
    if not free:
        return None
    choice = int(rng.integers(0, len(free)))
    dim, signs = free[choice]
    sign = signs[int(rng.integers(0, len(signs)))]
    moved = list(point)
    moved[dim] += sign
    return tuple(moved)


def apply_population_replacement(
    child: EvaluatedPoint,
    population: list[EvaluatedPoint],
    space: DiscreteSearchSpace,
    dcut: float,
) -> str:
    """
    Apply one PBH replacement step in place.

    A child whose index is already in the population is rejected before
    either branch, so ``dcut == 0`` cannot duplicate that index.

    Parameters
    ----------
    child : EvaluatedPoint
        Certified child.
    population : list of EvaluatedPoint
        Current population. Updated in place on acceptance.
    space : DiscreteSearchSpace
        Normalised Euclidean distance.
    dcut : float
        Constant diversity threshold. Distance ``== dcut`` is the diverse
        branch.

    Returns
    -------
    str
        ``near``, ``diverse``, or ``rejected``.
    """
    if any(member.x == child.x for member in population):
        return "rejected"
    nearest = min(
        population,
        key=lambda member: (space.distance(child.x, member.x), member.x),
    )
    distance = space.distance(child.x, nearest.x)
    if distance < dcut:
        if child.f < nearest.f:
            _replace_member(population, nearest, child)
            return "near"
        return "rejected"
    worst = max(population, key=lambda member: (member.f, member.x))
    if child.f < worst.f:
        _replace_member(population, worst, child)
        return "diverse"
    return "rejected"


def mean_pairwise_distance(
    population: list[EvaluatedPoint],
    space: DiscreteSearchSpace,
) -> float | None:
    """
    Return the mean pairwise normalised distance, or ``None`` when K < 2.

    Parameters
    ----------
    population : list of EvaluatedPoint
        Points to compare.
    space : DiscreteSearchSpace
        Distance.

    Returns
    -------
    float or None
        ``2 / (K (K - 1))`` times the sum of pairs.
    """
    count = len(population)
    if count < 2:
        return None
    total = 0.0
    for left in range(count):
        for right in range(left + 1, count):
            total += space.distance(population[left].x, population[right].x)
    return (2.0 / (count * (count - 1))) * total


def validate_discrete_mlsl_parameters(
    parameters: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    """
    Reject non-positive ``reduction_fraction`` and ``sigma``.

    Parameters
    ----------
    parameters : mapping
        Normalised registry parameters.

    Returns
    -------
    dict
        A copy of ``parameters``.

    Raises
    ------
    RegistryError
        If an open bound is violated.
    """
    return _validate_discrete_mlsl_parameters(
        parameters,
        token="DISCRETE_MLSL_PEN_SI_DESC_COLD_START",
    )


def validate_discrete_mlsl_deb_parameters(
    parameters: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    """Validate MLSL-Deb open-bound parameters with token-specific messages."""
    return _validate_discrete_mlsl_parameters(
        parameters,
        token="DISCRETE_MLSL_DEB_SI_DESC_COLD_START",
    )


def _validate_discrete_mlsl_parameters(
    parameters: Mapping[str, JSONValue],
    *,
    token: str,
) -> dict[str, JSONValue]:
    """Validate the shared MLSL open-bound parameters for one registry token."""
    reduction = parameters["reduction_fraction"]
    sigma = parameters["sigma"]
    if _not_positive_number(reduction) or float(reduction) > 1.0:
        raise RegistryError(
            f"{token}: reduction_fraction must be in (0, 1], "
            f"got {reduction!r}"
        )
    if _not_positive_number(sigma):
        raise RegistryError(
            f"{token}: sigma must be > 0, got {sigma!r}"
        )
    return dict(parameters)


def validate_population_basin_hopping_parameters(
    parameters: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    """
    Reject non-positive perturbation fraction and ``dcut_factor``.

    Parameters
    ----------
    parameters : mapping
        Normalised registry parameters.

    Returns
    -------
    dict
        A copy of ``parameters``.

    Raises
    ------
    RegistryError
        If an open bound is violated.
    """
    return _validate_population_basin_hopping_parameters(
        parameters,
        token="POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START",
    )


def _validate_population_basin_hopping_parameters(
    parameters: Mapping[str, JSONValue],
    *,
    token: str,
) -> dict[str, JSONValue]:
    """Validate the shared PBH open-bound parameters for one registry token."""
    fraction = parameters["perturbation_fraction"]
    factor = parameters["dcut_factor"]
    if _not_positive_number(fraction) or float(fraction) > 1.0:
        raise RegistryError(
            f"{token}: perturbation_fraction must be in (0, 1], "
            f"got {fraction!r}"
        )
    if _not_positive_number(factor):
        raise RegistryError(
            f"{token}: dcut_factor must be > 0, got {factor!r}"
        )
    return dict(parameters)


class DiscreteMLSL:
    """
    Discrete multi-level single-linkage with strict L1 local search.

    Global samples accumulate. A reduced-sample point is launched at most
    once, and only when no strictly better global sample lies inside the
    critical radius. Local-search evaluations are not global samples.

    ``sigma`` is the MLSL literature factor ζ. Defaults are development
    settings, not a tuned benchmark configuration.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int or None, optional
        Task seed. A new generator is built on every ``__call__``.
    batch_size : int, optional
        Global samples requested per iteration. Default 128.
    reduction_fraction : float, optional
        Fraction of the accumulated sample eligible for local search.
    sigma : float, optional
        Positive critical-radius factor ζ. Default 2.
    sampler : str, optional
        ``sobol`` or ``uniform``.
    """

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        batch_size: int = 128,
        reduction_fraction: float = 0.10,
        sigma: float = 2.0,
        sampler: str = "sobol",
    ) -> None:
        self.max_evaluations = _require_budget(max_evaluations)
        self.seed = seed
        self.batch_size = _require_positive_int("batch_size", batch_size)
        self.reduction_fraction = _require_open_unit(
            "reduction_fraction",
            reduction_fraction,
        )
        self.sigma = _require_positive_float("sigma", sigma)
        if sampler not in {"sobol", "uniform"}:
            raise ValueError(f"sampler must be 'sobol' or 'uniform', got {sampler!r}")
        self.sampler_name = sampler
        self.local_search = best_improvement_local_search
        self.name = "DISCRETE_MLSL_PEN_SI_DESC_COLD_START"
        label = "dMLSL-L1-Sobol" if sampler == "sobol" else "dMLSL-L1-uniform"
        self.info = (
            f"{label} batch={self.batch_size} "
            f"reduction_fraction={self.reduction_fraction} "
            f"sigma={self.sigma} sampler={self.sampler_name}"
        )
        self.last_run_stats: dict[str, Any] = {}

    def __call__(self, problem: Any, *, run_controller: Any = None) -> DiscreteMLSL:
        """
        Run one fixed call from ``seed``.

        Parameters
        ----------
        problem : Any
            IOH-like problem.
        run_controller : optional
            External controller. ``None`` builds a fixed-budget controller.

        Returns
        -------
        DiscreteMLSL
            ``self``. The benchmark reads ``problem.state``.
        """
        stats: dict[str, Any] = {
            "n_evaluations": 0,
            "n_local_searches": 0,
            "n_local_search_steps": 0,
            "n_certified_local_searches": 0,
            "n_unique_certified_local_optima": 0,
            "evaluations_per_local_search": [],
            "best_f": None,
            "best_x": None,
            "seed": self.seed,
            "n_global_samples": 0,
            "n_mlsl_iterations": 0,
            "n_reduced_candidates": 0,
            "n_candidates_radius_rejected": 0,
            "n_candidates_already_started": 0,
            "critical_radius_history": [],
            "local_search_start_points": [],
        }
        self.last_run_stats = stats
        slot: list[EvaluationManager | None] = [None]
        try:
            self._execute(problem, run_controller, stats, slot)
        except BudgetExhausted:
            pass
        finally:
            stats.pop("_unique_optima", None)
            if slot[0] is not None:
                _write_best(stats, slot[0])
        return self

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
        slot: list[EvaluationManager | None],
    ) -> None:
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        _dimension, lower, upper = read_problem_bounds(problem)
        space = DiscreteSearchSpace(lower, upper)
        evaluator = EvaluationManager(problem, controller, space)
        slot[0] = evaluator
        rng = np.random.default_rng(self.seed)
        sampler = DiscreteSampler(space, self.sampler_name, self.seed, rng)
        samples: list[EvaluatedPoint] = []
        sample_keys: set[tuple[int, ...]] = set()
        started: set[tuple[int, ...]] = set()
        unique: set[tuple[int, ...]] = set()

        while evaluator.has_budget():
            drew_any = False
            space_exhausted = False
            for _ in range(self.batch_size):
                if not evaluator.has_budget():
                    break
                point = sampler.next_unique(sample_keys)
                if point is None:
                    space_exhausted = True
                    break
                record = evaluator.evaluate(point)
                samples.append(record)
                sample_keys.add(record.x)
                drew_any = True
            stats["n_global_samples"] = len(samples)
            if not evaluator.has_budget():
                break
            if not drew_any:
                break

            sample_count = len(samples)
            radius = critical_radius(
                sample_count,
                space.n_free_dimensions,
                self.sigma,
            )
            n_reduced = max(1, math.ceil(self.reduction_fraction * sample_count))
            reduced = sorted(samples, key=lambda item: (item.f, item.x))[:n_reduced]
            stats["n_mlsl_iterations"] += 1
            stats["critical_radius_history"].append(radius)
            stats["n_reduced_candidates"] += len(reduced)
            for candidate in reduced:
                if not evaluator.has_budget():
                    break
                if candidate.x in started:
                    stats["n_candidates_already_started"] += 1
                    continue
                if better_sample_within_radius(candidate, samples, radius, space):
                    stats["n_candidates_radius_rejected"] += 1
                    continue
                started.add(candidate.x)
                stats["local_search_start_points"].append(list(candidate.x))
                result = self.local_search(candidate, evaluator, space)
                _note_search(stats, result, unique)
            if space_exhausted:
                break


class DiscreteMLSLWarmStart(DiscreteMLSL):
    """
    Discrete MLSL whose first batch is a ranked InitialDesign prefix.

    Later batches use the cold sampler. The sampler is not advanced during
    the first batch.

    Parameters
    ----------
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        batch_size: int = 128,
        reduction_fraction: float = 0.10,
        sigma: float = 2.0,
        sampler: str = "sobol",
        *,
        initial_design_path: str,
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            batch_size=batch_size,
            reduction_fraction=reduction_fraction,
            sigma=sigma,
            sampler=sampler,
        )
        self.initial_design_path = initial_design_path
        self.name = "DISCRETE_MLSL_PEN_SI_DESC_WARM_START"

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
        slot: list[EvaluationManager | None],
    ) -> None:
        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        _dimension, lower, upper = read_problem_bounds(problem)
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=self.batch_size,
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
        slot[0] = evaluator
        rng = np.random.default_rng(self.seed)
        sampler = DiscreteSampler(space, self.sampler_name, self.seed, rng)
        samples: list[EvaluatedPoint] = []
        sample_keys: set[tuple[int, ...]] = set()
        started: set[tuple[int, ...]] = set()
        unique: set[tuple[int, ...]] = set()
        first_batch = True

        while evaluator.has_budget():
            drew_any = False
            space_exhausted = False
            if first_batch:
                for vector in prefix:
                    if not evaluator.has_budget():
                        break
                    record = evaluator.evaluate(vector)
                    samples.append(record)
                    sample_keys.add(record.x)
                    drew_any = True
                first_batch = False
            else:
                for _ in range(self.batch_size):
                    if not evaluator.has_budget():
                        break
                    point = sampler.next_unique(sample_keys)
                    if point is None:
                        space_exhausted = True
                        break
                    record = evaluator.evaluate(point)
                    samples.append(record)
                    sample_keys.add(record.x)
                    drew_any = True
            stats["n_global_samples"] = len(samples)
            if not evaluator.has_budget():
                break
            if not drew_any:
                break

            sample_count = len(samples)
            radius = critical_radius(
                sample_count,
                space.n_free_dimensions,
                self.sigma,
            )
            n_reduced = max(1, math.ceil(self.reduction_fraction * sample_count))
            reduced = sorted(samples, key=lambda item: (item.f, item.x))[:n_reduced]
            stats["n_mlsl_iterations"] += 1
            stats["critical_radius_history"].append(radius)
            stats["n_reduced_candidates"] += len(reduced)
            for candidate in reduced:
                if not evaluator.has_budget():
                    break
                if candidate.x in started:
                    stats["n_candidates_already_started"] += 1
                    continue
                if better_sample_within_radius(candidate, samples, radius, space):
                    stats["n_candidates_radius_rejected"] += 1
                    continue
                started.add(candidate.x)
                stats["local_search_start_points"].append(list(candidate.x))
                result = self.local_search(candidate, evaluator, space)
                _note_search(stats, result, unique)
            if space_exhausted:
                break


class DiscreteMLSLEquDesc(DiscreteMLSL):
    """Discrete MLSL whose inner local search is equal descent."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        batch_size: int = 128,
        reduction_fraction: float = 0.10,
        sigma: float = 2.0,
        sampler: str = "sobol",
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            batch_size=batch_size,
            reduction_fraction=reduction_fraction,
            sigma=sigma,
            sampler=sampler,
        )
        self.name = "DISCRETE_MLSL_PEN_EQU_DESC_COLD_START"
        self.local_search = equal_descent_local_search
        label = (
            "dMLSL-L1-EquDesc-Sobol"
            if self.sampler_name == "sobol"
            else "dMLSL-L1-EquDesc-uniform"
        )
        self.info = (
            f"{label} batch={self.batch_size} "
            f"reduction_fraction={self.reduction_fraction} "
            f"sigma={self.sigma} sampler={self.sampler_name}"
        )


class DiscreteMLSLEquDescWarmStart(DiscreteMLSLWarmStart):
    """Equal-descent MLSL from a ranked InitialDesign prefix."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        batch_size: int = 128,
        reduction_fraction: float = 0.10,
        sigma: float = 2.0,
        sampler: str = "sobol",
        *,
        initial_design_path: str,
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            batch_size=batch_size,
            reduction_fraction=reduction_fraction,
            sigma=sigma,
            sampler=sampler,
            initial_design_path=initial_design_path,
        )
        self.name = "DISCRETE_MLSL_PEN_EQU_DESC_WARM_START"
        self.local_search = equal_descent_local_search
        label = (
            "dMLSL-L1-EquDesc-Sobol"
            if self.sampler_name == "sobol"
            else "dMLSL-L1-EquDesc-uniform"
        )
        self.info = (
            f"{label} batch={self.batch_size} "
            f"reduction_fraction={self.reduction_fraction} "
            f"sigma={self.sigma} sampler={self.sampler_name}"
        )


class SobolMultistart:
    """
    Scrambled Sobol sampling with one local descent from every new sample.

    The sampler is the same scrambled Sobol stream MLSL builds for the same
    seed and integer box. A projection that repeats an earlier accepted
    sample is discarded without an objective evaluation and without a local
    search. The Sobol stream has already advanced, which is the same rule as
    :meth:`DiscreteSampler.next_unique`. A projection that only coincides
    with a point visited inside a previous local search is still evaluated
    and searched: the seen set is the set of accepted samples, not the set
    of all evaluated designs.

    There is no batch, no top-fraction reduction, and no critical-radius
    test. The next sample does not depend on a certified local optimum.
    The local search is :func:`best_improvement_local_search`. A sample
    that meets the fixed target does not open a neighbourhood, because the
    run controller then refuses further evaluations.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int or None, optional
        Task seed. A new generator and Sobol stream are built on every call.
    """

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
    ) -> None:
        self.max_evaluations = _require_budget(max_evaluations)
        self.seed = seed
        self.local_search = best_improvement_local_search
        self.name = "SOBOL_MULTISTART_PEN_SI_DESC_COLD_START"
        self.info = "Sobol-multistart-L1 sampler=sobol selection=none"
        self.last_run_stats: dict[str, Any] = {}

    def __call__(self, problem: Any, *, run_controller: Any = None) -> SobolMultistart:
        """
        Run one fixed call from ``seed``.

        Parameters
        ----------
        problem : Any
            IOH-like problem.
        run_controller : optional
            External controller. ``None`` builds a fixed-budget controller.

        Returns
        -------
        SobolMultistart
            ``self``. The benchmark reads ``problem.state``.
        """
        stats = _sobol_multistart_stats(self.seed)
        self.last_run_stats = stats
        slot: list[EvaluationManager | None] = [None]
        try:
            self._execute(problem, run_controller, stats, slot)
        except BudgetExhausted:
            pass
        finally:
            stats.pop("_unique_optima", None)
            if slot[0] is not None:
                _write_best(stats, slot[0])
        return self

    def summary_diagnostics(self) -> Mapping[str, JSONValue]:
        """Return the JSON-safe counters from the last call."""
        stats = self.last_run_stats
        starts = [
            [int(value) for value in point]
            for point in stats.get("local_search_start_points", [])
        ]
        return {
            "n_evaluations": int(stats.get("n_evaluations", 0)),
            "n_local_searches": int(stats.get("n_local_searches", 0)),
            "n_local_search_steps": int(stats.get("n_local_search_steps", 0)),
            "n_certified_local_searches": int(stats.get("n_certified_local_searches", 0)),
            "n_unique_certified_local_optima": int(
                stats.get("n_unique_certified_local_optima", 0)
            ),
            "evaluations_per_local_search": [
                int(value) for value in stats.get("evaluations_per_local_search", [])
            ],
            "n_global_samples": int(stats.get("n_global_samples", 0)),
            "n_duplicate_projections_skipped": int(
                stats.get("n_duplicate_projections_skipped", 0)
            ),
            "n_candidates_radius_rejected": 0,
            "n_reduced_candidates": 0,
            "selection": "none",
            "local_search_start_points": starts,
        }

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
        slot: list[EvaluationManager | None],
    ) -> None:
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        _dimension, lower, upper = read_problem_bounds(problem)
        space = DiscreteSearchSpace(lower, upper)
        evaluator = EvaluationManager(problem, controller, space)
        slot[0] = evaluator
        rng = np.random.default_rng(self.seed)
        sampler = DiscreteSampler(space, "sobol", self.seed, rng)
        sample_keys: set[tuple[int, ...]] = set()
        unique: set[tuple[int, ...]] = set()

        while evaluator.has_budget():
            point = _next_unseen_sobol(sampler, sample_keys, stats, space)
            if point is None:
                break
            record = evaluator.evaluate(point)
            sample_keys.add(record.x)
            stats["n_global_samples"] = len(sample_keys)
            if not evaluator.has_budget():
                break
            stats["local_search_start_points"].append(list(record.x))
            result = self.local_search(record, evaluator, space)
            _note_search(stats, result, unique)


class SobolSample:
    """
    Scrambled Sobol sampling with no local search.

    The sampler and the duplicate-projection rule are the same objects
    :class:`SobolMultistart` uses. A projection that repeats an accepted
    sample is discarded without an objective evaluation. The stream has
    already advanced. A projection that only coincides with an earlier
    evaluated design is still evaluated, because every accepted sample is
    evaluated exactly once and there is no neighbourhood.

    Nothing in the loop reads a previous objective value except the
    controller's target and budget checks. There is no elite archive and
    no critical-radius test.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int or None, optional
        Task seed. A new generator and Sobol stream are built on every call.
    """

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
    ) -> None:
        self.max_evaluations = _require_budget(max_evaluations)
        self.seed = seed
        self.name = "SOBOL_PEN_COLD_START"
        self.info = "Sobol sampler=sobol local_search=none selection=none"
        self.last_run_stats: dict[str, Any] = {}

    def __call__(self, problem: Any, *, run_controller: Any = None) -> SobolSample:
        """
        Run one fixed call from ``seed``.

        Parameters
        ----------
        problem : Any
            IOH-like problem.
        run_controller : optional
            External controller. ``None`` builds a fixed-budget controller.

        Returns
        -------
        SobolSample
            ``self``. The benchmark reads ``problem.state``.
        """
        stats = _sobol_sample_stats(self.seed)
        self.last_run_stats = stats
        slot: list[EvaluationManager | None] = [None]
        try:
            self._execute(problem, run_controller, stats, slot)
        except BudgetExhausted:
            pass
        finally:
            if slot[0] is not None:
                _write_best(stats, slot[0])
        return self

    def summary_diagnostics(self) -> Mapping[str, JSONValue]:
        """Return the JSON-safe counters from the last call."""
        stats = self.last_run_stats
        return {
            "n_evaluations": int(stats.get("n_evaluations", 0)),
            "n_local_searches": 0,
            "n_local_search_steps": 0,
            "n_certified_local_searches": 0,
            "n_unique_certified_local_optima": 0,
            "evaluations_per_local_search": [],
            "n_global_samples": int(stats.get("n_global_samples", 0)),
            "n_duplicate_projections_skipped": int(
                stats.get("n_duplicate_projections_skipped", 0)
            ),
            "n_candidates_radius_rejected": 0,
            "n_reduced_candidates": 0,
            "selection": "none",
            "local_search": "none",
            "local_search_start_points": [],
        }

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
        slot: list[EvaluationManager | None],
    ) -> None:
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        _dimension, lower, upper = read_problem_bounds(problem)
        space = DiscreteSearchSpace(lower, upper)
        evaluator = EvaluationManager(problem, controller, space)
        slot[0] = evaluator
        rng = np.random.default_rng(self.seed)
        sampler = DiscreteSampler(space, "sobol", self.seed, rng)
        sample_keys: set[tuple[int, ...]] = set()

        while evaluator.has_budget():
            point = _next_unseen_sobol(sampler, sample_keys, stats, space)
            if point is None:
                break
            record = evaluator.evaluate(point)
            sample_keys.add(record.x)
            stats["n_global_samples"] = len(sample_keys)


class SobolMultistartEquDesc(SobolMultistart):
    """Sobol multistart whose inner local search is equal descent."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
    ) -> None:
        super().__init__(max_evaluations, seed=seed)
        self.local_search = equal_descent_local_search
        self.name = "SOBOL_MULTISTART_PEN_EQU_DESC_COLD_START"
        self.info = "Sobol-multistart-L1-EquDesc sampler=sobol selection=none"


def _sobol_sample_stats(seed: int | None) -> dict[str, Any]:
    """Counters for one Sobol sample call. No local search is opened."""
    return {
        "n_evaluations": 0,
        "n_global_samples": 0,
        "n_duplicate_projections_skipped": 0,
        "best_f": None,
        "best_x": None,
        "seed": seed,
        "selection": "none",
        "local_search": "none",
    }


def _sobol_multistart_stats(seed: int | None) -> dict[str, Any]:
    """Counters for one Sobol-multistart call. No reduction and no radius."""
    return {
        "n_evaluations": 0,
        "n_local_searches": 0,
        "n_local_search_steps": 0,
        "n_certified_local_searches": 0,
        "n_unique_certified_local_optima": 0,
        "evaluations_per_local_search": [],
        "best_f": None,
        "best_x": None,
        "seed": seed,
        "n_global_samples": 0,
        "n_duplicate_projections_skipped": 0,
        "n_candidates_radius_rejected": 0,
        "n_reduced_candidates": 0,
        "selection": "none",
        "local_search_start_points": [],
    }


def _next_unseen_sobol(
    sampler: DiscreteSampler,
    seen: set[tuple[int, ...]],
    stats: dict[str, Any],
    space: DiscreteSearchSpace,
) -> tuple[int, ...] | None:
    """
    Draw the next scrambled Sobol index that is not an accepted sample.

    A repeated projection increments ``n_duplicate_projections_skipped`` and
    is not returned. ``None`` means every legal index has already been
    accepted. The draw itself advances the Sobol stream.
    """
    while len(seen) < space.cardinality:
        point = sampler.draw()
        if point in seen:
            stats["n_duplicate_projections_skipped"] += 1
            continue
        return point
    return None


class PopulationBasinHopping:
    """
    Discrete population basin hopping on certified L1 local optima.

    Children of one generation are built from a snapshot of the parent
    population. ``dcut_factor`` is a development setting, not a canonical
    PBH constant. ``population_size == 1`` is monotonic basin hopping
    because ``dcut`` is infinite.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int or None, optional
        Task seed. A new generator is built on every ``__call__``.
    population_size : int, optional
        Requested population size K. Default 8.
    perturbation_fraction : float, optional
        Relative box half-width ρ. Default 0.15.
    dcut_factor : float, optional
        Multiplier on the initial mean distance. Must be > 0. Default 0.5.
    initial_sampler : str, optional
        ``sobol`` or ``uniform`` for the initial starts.
    """

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        population_size: int = 8,
        perturbation_fraction: float = 0.15,
        dcut_factor: float = 0.5,
        initial_sampler: str = "sobol",
    ) -> None:
        self.max_evaluations = _require_budget(max_evaluations)
        self.seed = seed
        self.population_size = _require_positive_int(
            "population_size",
            population_size,
        )
        self.perturbation_fraction = _require_open_unit(
            "perturbation_fraction",
            perturbation_fraction,
        )
        self.dcut_factor = _require_positive_float("dcut_factor", dcut_factor)
        if initial_sampler not in {"sobol", "uniform"}:
            raise ValueError(
                "initial_sampler must be 'sobol' or 'uniform', "
                f"got {initial_sampler!r}"
            )
        self.initial_sampler = initial_sampler
        self.local_search = best_improvement_local_search
        self.name = "POPULATION_BASIN_HOPPING_PEN_SI_DESC_COLD_START"
        self.info = (
            "PBH-L1 "
            f"K={self.population_size} "
            f"rho={self.perturbation_fraction} "
            f"dcut_factor={self.dcut_factor} "
            f"initial={self.initial_sampler}"
        )
        self.last_run_stats: dict[str, Any] = {}

    def __call__(
        self,
        problem: Any,
        *,
        run_controller: Any = None,
    ) -> PopulationBasinHopping:
        """
        Run one fixed call from ``seed``.

        Parameters
        ----------
        problem : Any
            IOH-like problem.
        run_controller : optional
            External controller. ``None`` builds a fixed-budget controller.

        Returns
        -------
        PopulationBasinHopping
            ``self``. The benchmark reads ``problem.state``.
        """
        stats: dict[str, Any] = {
            "n_evaluations": 0,
            "n_local_searches": 0,
            "n_local_search_steps": 0,
            "n_certified_local_searches": 0,
            "n_unique_certified_local_optima": 0,
            "evaluations_per_local_search": [],
            "best_f": None,
            "best_x": None,
            "seed": self.seed,
            "initial_population_size": 0,
            "n_generations": 0,
            "n_children": 0,
            "n_near_replacements": 0,
            "n_diverse_replacements": 0,
            "n_rejected_children": 0,
            "dcut": None,
            "initial_mean_distance": None,
            "population_mean_distance_history": [],
        }
        self.last_run_stats = stats
        slot: list[EvaluationManager | None] = [None]
        try:
            self._execute(problem, run_controller, stats, slot)
        except BudgetExhausted:
            pass
        finally:
            stats.pop("_unique_optima", None)
            if slot[0] is not None:
                _write_best(stats, slot[0])
        return self

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
        slot: list[EvaluationManager | None],
    ) -> None:
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        _dimension, lower, upper = read_problem_bounds(problem)
        space = DiscreteSearchSpace(lower, upper)
        evaluator = EvaluationManager(problem, controller, space)
        slot[0] = evaluator
        rng = np.random.default_rng(self.seed)
        sampler = DiscreteSampler(space, self.initial_sampler, self.seed, rng)
        population = _initial_population(
            evaluator,
            space,
            sampler,
            self.population_size,
            stats,
            local_search=self.local_search,
        )
        continue_pbh_generations(
            population,
            evaluator,
            space,
            rng,
            self.perturbation_fraction,
            self.dcut_factor,
            stats,
            local_search=self.local_search,
        )


class PopulationBasinHoppingWarmStart(PopulationBasinHopping):
    """
    Basin hopping whose initial starts are a ranked InitialDesign prefix.

    The prefix is evaluated in order before any local-search neighbour.

    Parameters
    ----------
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        population_size: int = 8,
        perturbation_fraction: float = 0.15,
        dcut_factor: float = 0.5,
        initial_sampler: str = "sobol",
        *,
        initial_design_path: str,
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            population_size=population_size,
            perturbation_fraction=perturbation_fraction,
            dcut_factor=dcut_factor,
            initial_sampler=initial_sampler,
        )
        self.initial_design_path = initial_design_path
        self.name = "POPULATION_BASIN_HOPPING_PEN_SI_DESC_WARM_START"

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
        slot: list[EvaluationManager | None],
    ) -> None:
        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        _dimension, lower, upper = read_problem_bounds(problem)
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=self.population_size,
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
        slot[0] = evaluator
        rng = np.random.default_rng(self.seed)
        starts: list[EvaluatedPoint] = []
        for vector in prefix:
            if not evaluator.has_budget():
                break
            starts.append(evaluator.evaluate(vector))
        population = build_pbh_population_from_starts(
            starts,
            evaluator,
            space,
            stats,
            local_search=self.local_search,
        )
        continue_pbh_generations(
            population,
            evaluator,
            space,
            rng,
            self.perturbation_fraction,
            self.dcut_factor,
            stats,
            local_search=self.local_search,
        )


class PopulationBasinHoppingEquDesc(PopulationBasinHopping):
    """Population basin hopping whose inner local search is equal descent."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        population_size: int = 8,
        perturbation_fraction: float = 0.15,
        dcut_factor: float = 0.5,
        initial_sampler: str = "sobol",
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            population_size=population_size,
            perturbation_fraction=perturbation_fraction,
            dcut_factor=dcut_factor,
            initial_sampler=initial_sampler,
        )
        self.name = "POPULATION_BASIN_HOPPING_PEN_EQU_DESC_COLD_START"
        self.local_search = equal_descent_local_search
        self.info = (
            "PBH-L1-EquDesc "
            f"K={self.population_size} "
            f"rho={self.perturbation_fraction} "
            f"dcut_factor={self.dcut_factor} "
            f"initial={self.initial_sampler}"
        )


class PopulationBasinHoppingEquDescWarmStart(PopulationBasinHoppingWarmStart):
    """Equal-descent basin hopping from a ranked InitialDesign prefix."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        population_size: int = 8,
        perturbation_fraction: float = 0.15,
        dcut_factor: float = 0.5,
        initial_sampler: str = "sobol",
        *,
        initial_design_path: str,
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            population_size=population_size,
            perturbation_fraction=perturbation_fraction,
            dcut_factor=dcut_factor,
            initial_sampler=initial_sampler,
            initial_design_path=initial_design_path,
        )
        self.name = "POPULATION_BASIN_HOPPING_PEN_EQU_DESC_WARM_START"
        self.local_search = equal_descent_local_search
        self.info = (
            "PBH-L1-EquDesc "
            f"K={self.population_size} "
            f"rho={self.perturbation_fraction} "
            f"dcut_factor={self.dcut_factor} "
            f"initial={self.initial_sampler}"
        )


def run_pbh_generation(
    population: list[Any],
    evaluator: Any,
    space: DiscreteSearchSpace,
    rng: np.random.Generator,
    perturbation_fraction: float,
    dcut: float,
    stats: dict[str, Any],
    *,
    local_search: Any = None,
    note_search: Any = None,
    replacement: Any = None,
    stop_after_start: Any = None,
    child_observer: Any = None,
) -> bool:
    """
    Build children from a parent snapshot, then apply replacement.

    Parameters
    ----------
    population : list of EvaluatedPoint
        Updated in place by replacement.
    evaluator : EvaluationManager
        Budgeted evaluations.
    space : DiscreteSearchSpace
        Perturbation bounds and distance.
    rng : numpy.random.Generator
        Call-local generator.
    perturbation_fraction : float
        Relative box half-width.
    dcut : float
        Constant diversity threshold.
    stats : dict
        Diagnostic counters updated in place.

    Returns
    -------
    bool
        True when a parent had no legal perturbation.
    """
    if local_search is None:
        local_search = best_improvement_local_search
    if note_search is None:
        note_search = _note_search
    if replacement is None:
        replacement = apply_population_replacement
    parents = list(population)
    stats["n_generations"] += 1
    children: list[EvaluatedPoint] = []
    immovable = False
    unique = stats.setdefault("_unique_optima", set())
    for parent in parents:
        if not evaluator.has_budget():
            break
        perturbed = box_perturbation(
            parent.x,
            space,
            rng,
            perturbation_fraction,
        )
        if perturbed is None:
            immovable = True
            break
        start = evaluator.evaluate(perturbed)
        if stop_after_start is not None and stop_after_start(evaluator):
            break
        result = local_search(start, evaluator, space)
        note_search(stats, result, unique)
        if result.certified_local_optimum:
            children.append(result.point)
            if child_observer is not None:
                child_observer(result.point)
    stats["n_children"] += len(children)
    for child in children:
        outcome = replacement(child, population, space, dcut)
        if outcome == "near":
            stats["n_near_replacements"] += 1
        elif outcome == "diverse":
            stats["n_diverse_replacements"] += 1
        else:
            stats["n_rejected_children"] += 1
    return immovable


def _initial_population(
    evaluator: Any,
    space: DiscreteSearchSpace,
    sampler: DiscreteSampler,
    population_size: int,
    stats: dict[str, Any],
    *,
    local_search: Any = None,
    note_search: Any = None,
    stop_after_start: Any = None,
) -> list[Any]:
    """Build the deduplicated certified initial population."""
    if local_search is None:
        local_search = best_improvement_local_search
    if note_search is None:
        note_search = _note_search
    population: list[Any] = []
    seen_starts: set[tuple[int, ...]] = set()
    unique: set[tuple[int, ...]] = set()
    stats["_unique_optima"] = unique
    attempt_cap = 20 * population_size
    attempts = 0
    while (
        evaluator.has_budget()
        and len(population) < population_size
        and attempts < attempt_cap
    ):
        if len(seen_starts) >= space.cardinality:
            break
        point = sampler.draw()
        attempts += 1
        if point in seen_starts:
            continue
        seen_starts.add(point)
        start = evaluator.evaluate(point)
        if stop_after_start is not None and stop_after_start(evaluator):
            break
        result = local_search(start, evaluator, space)
        note_search(stats, result, unique)
        if result.certified_local_optimum and result.point.x not in {
            member.x for member in population
        }:
            population.append(result.point)
    stats["initial_population_size"] = len(population)
    return population


def build_pbh_population_from_starts(
    starts: list[Any],
    evaluator: Any,
    space: DiscreteSearchSpace,
    stats: dict[str, Any],
    *,
    local_search: Any = None,
    note_search: Any = None,
) -> list[Any]:
    """Build a deduplicated certified population from evaluated starts."""
    if local_search is None:
        local_search = best_improvement_local_search
    if note_search is None:
        note_search = _note_search
    population: list[Any] = []
    unique: set[tuple[int, ...]] = set()
    stats["_unique_optima"] = unique
    for start in starts:
        if not evaluator.has_budget():
            break
        result = local_search(start, evaluator, space)
        note_search(stats, result, unique)
        if (
            result.certified_local_optimum
            and result.point.x not in {member.x for member in population}
        ):
            population.append(result.point)
    stats["initial_population_size"] = len(population)
    return population


def continue_pbh_generations(
    population: list[Any],
    evaluator: Any,
    space: DiscreteSearchSpace,
    rng: np.random.Generator,
    perturbation_fraction: float,
    dcut_factor: float,
    stats: dict[str, Any],
    *,
    local_search: Any = None,
    note_search: Any = None,
    replacement: Any = None,
    stop_after_start: Any = None,
    child_observer: Any = None,
    population_observer: Any = None,
) -> None:
    """Run the shared PBH diversity-threshold and generation schedule."""
    if (not evaluator.has_budget()) or (not population):
        return
    mean_distance = mean_pairwise_distance(population, space)
    stats["initial_mean_distance"] = mean_distance
    if "population_mean_distance_history" in stats:
        stats["population_mean_distance_history"].append(mean_distance)
    if population_observer is not None:
        population_observer(population)
    dcut = (
        math.inf
        if len(population) == 1
        else float(dcut_factor) * float(mean_distance)
    )
    if "dcut" in stats:
        stats["dcut"] = dcut
    while evaluator.has_budget():
        immovable = run_pbh_generation(
            population,
            evaluator,
            space,
            rng,
            perturbation_fraction,
            dcut,
            stats,
            local_search=local_search,
            note_search=note_search,
            replacement=replacement,
            stop_after_start=stop_after_start,
            child_observer=child_observer,
        )
        if "population_mean_distance_history" in stats:
            stats["population_mean_distance_history"].append(
                mean_pairwise_distance(population, space)
            )
        if population_observer is not None:
            population_observer(population)
        if immovable or not evaluator.has_budget():
            break


def _write_best(stats: dict[str, Any], evaluator: EvaluationManager) -> None:
    """Copy the finite incumbent and the controller count into diagnostics."""
    stats.pop("_unique_optima", None)
    stats["n_evaluations"] = evaluator.n_evaluations
    if evaluator.best_point is None:
        stats["best_f"] = None
        stats["best_x"] = None
        return
    stats["best_f"] = float(evaluator.best_point.f)
    stats["best_x"] = [int(value) for value in evaluator.best_point.x]


def _replace_member(
    population: list[EvaluatedPoint],
    target: EvaluatedPoint,
    child: EvaluatedPoint,
) -> None:
    """Replace the population member with the same index as ``target``."""
    for index, member in enumerate(population):
        if member.x == target.x:
            population[index] = child
            return


def _integer_bound_vector(values: Any, field: str) -> np.ndarray:
    """Convert a bound sequence to ``int64``, rejecting bool and non-integrals."""
    try:
        sequence = list(values)
    except TypeError as exc:
        raise ValueError(f"{field} bounds must be a sequence, got {values!r}") from exc
    out: list[int] = []
    for index, raw in enumerate(sequence):
        if isinstance(raw, bool) or isinstance(raw, np.bool_):
            raise ValueError(f"{field}[{index}] must be an integer, got {raw!r}")
        if isinstance(raw, (int, np.integer)):
            out.append(int(raw))
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field}[{index}] must be an integer, got {raw!r}"
            ) from exc
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{field}[{index}] must be an integer, got {raw!r}")
        out.append(int(number))
    return np.asarray(out, dtype=np.int64)


def _require_budget(value: Any) -> int:
    """Return a positive evaluation budget."""
    return _require_positive_int("max_evaluations", value)


def _require_positive_int(name: str, value: Any) -> int:
    """Reject bool and non-positive integers."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return int(value)


def _require_positive_float(name: str, value: Any) -> float:
    """Reject bool, non-numbers, and non-positive values."""
    if _not_positive_number(value):
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return float(value)


def _require_open_unit(name: str, value: Any) -> float:
    """Reject values outside (0, 1]."""
    if _not_positive_number(value) or float(value) > 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")
    return float(value)


def _not_positive_number(value: Any) -> bool:
    """Return whether ``value`` is not a number strictly greater than 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    number = float(value)
    return (not math.isfinite(number)) or number <= 0.0


@dataclass(frozen=True)
class DebEvaluatedPoint:
    """One evaluation with benchmark and Deb-ordering channels."""

    x: tuple[int, ...]
    y_p: float
    y: float
    phi: float
    feasible: bool
    analysis_ok: bool
    sentinel_constraints: tuple[str, ...]


PBHDebEvaluatedPoint = DebEvaluatedPoint


@dataclass(frozen=True)
class DebLocalSearchResult:
    """Outcome of one Deb L1 local search."""

    point: DebEvaluatedPoint
    certified_local_optimum: bool
    n_steps: int
    n_evaluations: int
    n_certification_evaluations: int = 0


PBHDebLocalSearchResult = DebLocalSearchResult


def deb_point_key(point: DebEvaluatedPoint) -> tuple[int, float]:
    """Return the canonical Deb feasibility-rule key."""
    return deb_quality_key(feasible=point.feasible, y=point.y, phi=point.phi)


pbh_deb_key = deb_point_key


def deb_strictly_better(
    left: DebEvaluatedPoint,
    right: DebEvaluatedPoint,
) -> bool:
    """Return whether ``left`` is a strict Deb improvement over ``right``."""
    return deb_point_key(left) < deb_point_key(right)


pbh_strictly_better = deb_strictly_better


class DebEvaluationManager:
    """Validate the auxiliary-record contract and track both incumbents."""

    def __init__(
        self,
        problem: Any,
        controller: Any,
        space: DiscreteSearchSpace,
        *,
        algorithm_label: str = "PBH-L1-Deb",
    ) -> None:
        self.algorithm_label = str(algorithm_label)
        self.problem = problem
        self.controller = controller
        self.space = space
        context = get_attached_eval_context(problem)
        if context is None:
            raise ValueError(
                f"{self.algorithm_label} requires an EvalContext attached to the problem"
            )
        names = getattr(context, "_c_names", None)
        if not isinstance(names, list) or not names:
            raise ValueError(
                f"{self.algorithm_label} requires at least one active constraint name"
            )
        frozen: list[str] = []
        for raw in names:
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(
                    f"{self.algorithm_label} active constraint names must be non-empty strings"
                )
            frozen.append(raw)
        if len(set(frozen)) != len(frozen):
            raise ValueError(
                f"{self.algorithm_label} active constraint names must be unique"
            )
        self.active_constraint_names = tuple(frozen)
        run_config = getattr(context, "run_config", None)
        try:
            self.invalid_utilization = float(run_config.invalid_utilization)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.algorithm_label} requires EvalContext.run_config.invalid_utilization"
            ) from exc
        if not math.isfinite(self.invalid_utilization):
            raise ValueError(
                f"{self.algorithm_label} invalid-utilization sentinel must be finite"
            )
        self.best_benchmark_point: PBHDebEvaluatedPoint | None = None
        self.best_deb_point: PBHDebEvaluatedPoint | None = None
        self.diagnostics: dict[str, Any] = {
            "n_evaluations": 0,
            "n_valid_evaluations": 0,
            "n_feasible_evaluations": 0,
            "n_infeasible_evaluations": 0,
            "n_failed_analysis_evaluations": 0,
            "n_sentinel_evaluations": 0,
            "sentinel_evaluations_by_constraint": {
                name: 0 for name in self.active_constraint_names
            },
            "n_infeasible_plateau_terminations": 0,
            "n_sentinel_infeasible_plateau_terminations": 0,
        }

    @property
    def n_evaluations(self) -> int:
        """Return evaluations consumed according to the run controller."""
        return int(self.controller.used)

    def has_budget(self) -> bool:
        """Return whether another evaluation is permitted."""
        return bool(self.controller.can_evaluate())

    def evaluate(self, x: Any) -> PBHDebEvaluatedPoint:
        """Evaluate one point exactly once and validate its cached record."""
        if not self.controller.can_evaluate():
            raise BudgetExhausted("evaluation budget exhausted")
        point = self.space.validate(x)
        validated = validate_integer_candidate(
            point,
            lower_bounds=[int(value) for value in self.space.lower],
            upper_bounds=[int(value) for value in self.space.upper],
        )
        self.controller.require_evaluation()
        y_p = float(self.problem(validated))
        if math.isnan(y_p) or (math.isinf(y_p) and y_p < 0.0):
            raise RuntimeError(
                "Objective returned NaN or -inf. "
                "The optimiser must not silently modify the benchmark objective."
            )
        self.controller.observe_evaluation(y_p)
        record = read_last_evaluation_record(self.problem)
        channels = validate_deb_auxiliary_record(
            record,
            active_constraint_names=self.active_constraint_names,
            invalid_utilization=self.invalid_utilization,
        )
        if channels.analysis_ok:
            if math.isinf(y_p):
                raise ValueError(
                    "y_p=+inf is permitted only for an infeasible failed analysis"
                )
        elif math.isinf(y_p) and y_p < 0.0:
            raise ValueError("failed-analysis y_p must not be -inf")

        evaluated = PBHDebEvaluatedPoint(
            x=tuple(validated),
            y_p=y_p,
            y=channels.y,
            phi=channels.phi,
            feasible=channels.feasible,
            analysis_ok=channels.analysis_ok,
            sentinel_constraints=channels.sentinel_constraints,
        )
        self._record(evaluated)
        return evaluated

    def _record(self, point: PBHDebEvaluatedPoint) -> None:
        self.diagnostics["n_evaluations"] += 1
        if point.analysis_ok:
            self.diagnostics["n_valid_evaluations"] += 1
            category = (
                "n_feasible_evaluations"
                if point.feasible
                else "n_infeasible_evaluations"
            )
            self.diagnostics[category] += 1
            if point.sentinel_constraints:
                self.diagnostics["n_sentinel_evaluations"] += 1
                counts = self.diagnostics[
                    "sentinel_evaluations_by_constraint"
                ]
                for name in point.sentinel_constraints:
                    counts[name] += 1
            if self.best_deb_point is None or pbh_strictly_better(
                point,
                self.best_deb_point,
            ):
                self.best_deb_point = point
        else:
            self.diagnostics["n_failed_analysis_evaluations"] += 1
        if math.isfinite(point.y_p) and (
            self.best_benchmark_point is None
            or point.y_p < self.best_benchmark_point.y_p
        ):
            self.best_benchmark_point = point


PBHDebEvaluationManager = DebEvaluationManager


def deb_best_improvement_local_search(
    start: DebEvaluatedPoint,
    evaluator: DebEvaluationManager,
    space: DiscreteSearchSpace,
) -> DebLocalSearchResult:
    """Run strict best-improvement L1 search under Deb ordering."""
    known = {start.x: start}
    current = start
    n_steps = 0
    n_evaluations = 0
    while True:
        resolved: list[DebEvaluatedPoint] = []
        neighbours = space.neighbours_l1(current.x)
        incomplete = False
        target_hit = False
        n_new_this_round = 0
        for neighbour in neighbours:
            cached = known.get(neighbour)
            if cached is not None:
                resolved.append(cached)
                continue
            if not evaluator.has_budget():
                incomplete = True
                break
            try:
                record = evaluator.evaluate(neighbour)
            except BudgetExhausted:
                incomplete = True
                break
            n_evaluations += 1
            n_new_this_round += 1
            known[neighbour] = record
            resolved.append(record)
            if evaluator.controller.target_hit:
                target_hit = True
                incomplete = True
                break

        winner = _deb_winner(resolved)
        if incomplete:
            point = current
            if winner is not None and deb_strictly_better(winner, current):
                point = winner
                n_steps += 1
            return DebLocalSearchResult(
                point=point,
                certified_local_optimum=False,
                n_steps=n_steps,
                n_evaluations=n_evaluations,
            )
        if winner is not None and deb_strictly_better(winner, current):
            current = winner
            n_steps += 1
            if not evaluator.has_budget():
                return DebLocalSearchResult(
                    point=current,
                    certified_local_optimum=False,
                    n_steps=n_steps,
                    n_evaluations=n_evaluations,
                )
            continue

        certified = current.analysis_ok and not target_hit
        if certified and not current.feasible:
            equal_neighbours = [
                point
                for point in resolved
                if point.analysis_ok and deb_point_key(point) == deb_point_key(current)
            ]
            if equal_neighbours:
                evaluator.diagnostics["n_infeasible_plateau_terminations"] += 1
                if current.sentinel_constraints or any(
                    point.sentinel_constraints for point in equal_neighbours
                ):
                    evaluator.diagnostics[
                        "n_sentinel_infeasible_plateau_terminations"
                    ] += 1
        return DebLocalSearchResult(
            point=current,
            certified_local_optimum=certified,
            n_steps=n_steps,
            n_evaluations=n_evaluations,
            n_certification_evaluations=n_new_this_round if certified else 0,
        )


pbh_deb_best_improvement_local_search = deb_best_improvement_local_search


def deb_equal_descent_local_search(
    start: DebEvaluatedPoint,
    evaluator: DebEvaluationManager,
    space: DiscreteSearchSpace,
) -> DebLocalSearchResult:
    """
    Level-complete equal descent under the Deb key.

    Parallel designs stay active only while their finite Deb keys compare
    equal. A completed level keeps every finite improving neighbour at the
    best lower key and drops every worse one. The stored design changes
    only for that strictly better key. With no strict exit, the returned
    design is the one strict Deb search would have returned.

    The within-call cache dies with the call. A non-finite key is never a
    component member and never a tied exit. A finite key below a non-finite
    key is a strict exit. The start is not evaluated again. A target hit
    stops the neighbour list immediately and is not certified.
    """
    known = {start.x: start}
    n_steps = 0
    n_evaluations = 0
    active: list[DebEvaluatedPoint] = [start]

    def finish(
        point: DebEvaluatedPoint,
        certified: bool,
        *,
        early: bool = False,
        certification_evaluations: int = 0,
        resolved: list[DebEvaluatedPoint] | None = None,
    ) -> DebLocalSearchResult:
        actual = (
            certified
            and point.analysis_ok
            and not evaluator.controller.target_hit
        )
        if actual and not point.feasible:
            neighbours = resolved or []
            equal_neighbours = [
                item
                for item in neighbours
                if item.analysis_ok and deb_point_key(item) == deb_point_key(point)
            ]
            if equal_neighbours:
                evaluator.diagnostics["n_infeasible_plateau_terminations"] += 1
                if point.sentinel_constraints or any(
                    item.sentinel_constraints for item in equal_neighbours
                ):
                    evaluator.diagnostics[
                        "n_sentinel_infeasible_plateau_terminations"
                    ] += 1
        result = DebLocalSearchResult(
            point=point,
            certified_local_optimum=actual,
            n_steps=n_steps,
            n_evaluations=n_evaluations,
            n_certification_evaluations=(
                certification_evaluations if actual else 0
            ),
        )
        if early and actual:
            object.__setattr__(result, "early_certificate", True)
        return result

    def interrupted(
        incumbent: DebEvaluatedPoint,
        partial: list[DebEvaluatedPoint],
    ) -> DebLocalSearchResult:
        nonlocal n_steps
        better = [
            item for item in partial if deb_strictly_better(item, incumbent)
        ]
        if not better:
            return finish(incumbent, False)
        chosen = min(better, key=lambda item: (deb_point_key(item), item.x))
        n_steps += 1
        return finish(chosen, False)

    while True:
        incumbent = min(active, key=lambda item: (deb_point_key(item), item.x))
        level = deb_point_key(incumbent)
        seeds = sorted(active, key=lambda item: (deb_point_key(item), item.x))
        queue: deque[DebEvaluatedPoint] = deque(seeds)
        seen = {item.x for item in seeds}
        exits: list[DebEvaluatedPoint] = []
        exit_seen: set[tuple[int, ...]] = set()
        incumbent_done = False
        partial: list[DebEvaluatedPoint] = []
        incomplete = False
        incumbent_resolved: list[DebEvaluatedPoint] = []
        incumbent_new = 0
        while queue:
            node = queue.popleft()
            partial = []
            finished = True
            node_new = 0
            for neighbour in space.neighbours_l1(node.x):
                cached = known.get(neighbour)
                if cached is not None:
                    record = cached
                else:
                    if not evaluator.has_budget():
                        finished = False
                        break
                    try:
                        record = evaluator.evaluate(neighbour)
                    except BudgetExhausted:
                        finished = False
                        break
                    known[neighbour] = record
                    n_evaluations += 1
                    node_new += 1
                partial.append(record)
                record_key = deb_point_key(record)
                if math.isfinite(record_key[1]) and record_key == level:
                    if record.x not in seen:
                        seen.add(record.x)
                        queue.append(record)
                elif (
                    math.isfinite(record_key[1])
                    and record_key < level
                    and record.x not in exit_seen
                ):
                    exit_seen.add(record.x)
                    exits.append(record)
                if cached is None and evaluator.controller.target_hit:
                    finished = False
                    break
            if node.x == incumbent.x:
                incumbent_done = finished
                if finished:
                    incumbent_resolved = list(partial)
                    incumbent_new = node_new
            if not finished:
                incomplete = True
                break
        if incomplete:
            if not incumbent_done:
                return interrupted(incumbent, partial)
            if n_steps == 0 and not exits:
                return finish(
                    incumbent,
                    True,
                    early=True,
                    certification_evaluations=incumbent_new,
                    resolved=incumbent_resolved,
                )
            if n_steps == 0:
                return finish(incumbent, False)
            return interrupted(incumbent, partial)
        if not exits:
            return finish(
                incumbent,
                True,
                certification_evaluations=incumbent_new,
                resolved=incumbent_resolved,
            )
        best_key = min(deb_point_key(item) for item in exits)
        active = sorted(
            (item for item in exits if deb_point_key(item) == best_key),
            key=lambda item: (deb_point_key(item), item.x),
        )
        n_steps += 1


_MLSL_DEB_LABEL = "dMLSL-L1-Deb"


def run_discrete_mlsl_deb(
    *,
    evaluator: DebEvaluationManager,
    space: DiscreteSearchSpace,
    sampler: DiscreteSampler,
    batch_size: int,
    reduction_fraction: float,
    sigma: float,
    stats: dict[str, Any],
    first_batch: Sequence[tuple[int, ...]] | None,
    local_search: Any = None,
) -> None:
    """Run cold or warm MLSL with Deb reduction, linkage, and local search."""
    search = (
        deb_best_improvement_local_search
        if local_search is None
        else local_search
    )
    samples: list[DebEvaluatedPoint] = []
    sample_keys: set[tuple[int, ...]] = set()
    started: set[tuple[int, ...]] = set()
    unique_certified: set[tuple[int, ...]] = set()
    unique_feasible: set[tuple[int, ...]] = set()
    unique_infeasible: set[tuple[int, ...]] = set()
    first_batch_pending = first_batch is not None

    while evaluator.has_budget():
        drew_any = False
        space_exhausted = False
        if first_batch_pending:
            batch = first_batch if first_batch is not None else ()
            for vector in batch:
                if not evaluator.has_budget():
                    break
                record = evaluator.evaluate(vector)
                samples.append(record)
                sample_keys.add(record.x)
                _note_mlsl_deb_global_sample(stats, record)
                drew_any = True
            first_batch_pending = False
        else:
            for _ in range(batch_size):
                if not evaluator.has_budget():
                    break
                point = sampler.next_unique(sample_keys)
                if point is None:
                    space_exhausted = True
                    break
                record = evaluator.evaluate(point)
                samples.append(record)
                sample_keys.add(record.x)
                _note_mlsl_deb_global_sample(stats, record)
                drew_any = True

        stats["n_global_samples"] = len(samples)
        stats["n_unique_global_sample_points"] = len(sample_keys)
        if not evaluator.has_budget():
            break
        if not drew_any:
            break

        radius = critical_radius(
            len(samples),
            space.n_free_dimensions,
            sigma,
        )
        reduced = reduce_samples_deb(samples, reduction_fraction)
        stats["n_mlsl_iterations"] += 1
        stats["critical_radius_history"].append(radius)
        stats["n_reduced_candidates"] += len(reduced)
        for candidate in reduced:
            if not evaluator.has_budget():
                break
            if candidate.x in started:
                stats["n_candidates_already_started"] += 1
                continue
            if better_deb_sample_within_radius(
                candidate,
                samples,
                radius,
                space,
            ):
                stats["n_candidates_radius_rejected"] += 1
                continue
            started.add(candidate.x)
            stats["local_search_start_points"].append(
                [int(value) for value in candidate.x]
            )
            result = search(
                candidate,
                evaluator,
                space,
            )
            _note_mlsl_deb_search(
                stats,
                result,
                unique_certified,
                unique_feasible,
                unique_infeasible,
            )
        if space_exhausted:
            break


class DiscreteMLSLDeb:
    """Discrete MLSL with Deb-ordered reduction, linkage, and L1 search."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        batch_size: int = 128,
        reduction_fraction: float = 0.10,
        sigma: float = 2.0,
        sampler: str = "sobol",
    ) -> None:
        self.max_evaluations = _require_budget(max_evaluations)
        self.seed = seed
        self.batch_size = _require_positive_int("batch_size", batch_size)
        self.reduction_fraction = _require_open_unit(
            "reduction_fraction",
            reduction_fraction,
        )
        self.sigma = _require_positive_float("sigma", sigma)
        if sampler not in {"sobol", "uniform"}:
            raise ValueError(
                f"sampler must be 'sobol' or 'uniform', got {sampler!r}"
            )
        self.sampler_name = sampler
        self.name = "DISCRETE_MLSL_DEB_SI_DESC_COLD_START"
        self.local_search = deb_best_improvement_local_search
        self.info = self._info(warm_start=False)
        self.last_run_stats: dict[str, Any] = {}
        self._summary: dict[str, JSONValue] = {}

    def _info(self, *, warm_start: bool) -> str:
        sampler_label = "Sobol" if self.sampler_name == "sobol" else "uniform"
        warm = "-WarmStart" if warm_start else ""
        return (
            f"dMLSL-L1-Deb{warm}-{sampler_label} "
            f"batch={self.batch_size} "
            f"reduction_fraction={self.reduction_fraction} "
            f"sigma={self.sigma} sampler={self.sampler_name}"
        )

    def __call__(
        self,
        problem: Any,
        *,
        run_controller: Any = None,
    ) -> DiscreteMLSLDeb:
        """Run one seeded Deb MLSL call."""
        stats = _new_mlsl_deb_stats(self.seed)
        self.last_run_stats = stats
        self._summary = _build_mlsl_deb_summary(stats)
        slot: list[DebEvaluationManager | None] = [None]
        try:
            self._execute(problem, run_controller, stats, slot)
        except BudgetExhausted:
            pass
        finally:
            _finalize_mlsl_deb_stats(stats, slot[0])
            self._summary = _build_mlsl_deb_summary(stats)
        return self

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
        slot: list[DebEvaluationManager | None],
    ) -> None:
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
            algorithm_label=_MLSL_DEB_LABEL,
        )
        slot[0] = evaluator
        rng = np.random.default_rng(self.seed)
        sampler = DiscreteSampler(
            space,
            self.sampler_name,
            self.seed,
            rng,
        )
        run_discrete_mlsl_deb(
            evaluator=evaluator,
            space=space,
            sampler=sampler,
            batch_size=self.batch_size,
            reduction_fraction=self.reduction_fraction,
            sigma=self.sigma,
            stats=stats,
            first_batch=None,
            local_search=self.local_search,
        )

    def summary_diagnostics(self) -> Mapping[str, JSONValue]:
        """Return compact finite JSON-safe Deb MLSL diagnostics."""
        return dict(self._summary)


class DiscreteMLSLDebWarmStart(DiscreteMLSLDeb):
    """Deb MLSL whose first batch is a production-y_p-ranked prefix."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        batch_size: int = 128,
        reduction_fraction: float = 0.10,
        sigma: float = 2.0,
        sampler: str = "sobol",
        *,
        initial_design_path: str,
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            batch_size=batch_size,
            reduction_fraction=reduction_fraction,
            sigma=sigma,
            sampler=sampler,
        )
        if not isinstance(initial_design_path, str) or not initial_design_path:
            raise ValueError(
                "DISCRETE_MLSL_DEB_SI_DESC_WARM_START requires initial_design_path"
            )
        self.initial_design_path = initial_design_path
        self.name = "DISCRETE_MLSL_DEB_SI_DESC_WARM_START"
        self.info = self._info(warm_start=True)

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
        slot: list[DebEvaluationManager | None],
    ) -> None:
        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        _dimension, lower, upper = read_problem_bounds(problem)
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=self.batch_size,
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
            algorithm_label=_MLSL_DEB_LABEL,
        )
        slot[0] = evaluator
        rng = np.random.default_rng(self.seed)
        sampler = DiscreteSampler(
            space,
            self.sampler_name,
            self.seed,
            rng,
        )
        run_discrete_mlsl_deb(
            evaluator=evaluator,
            space=space,
            sampler=sampler,
            batch_size=self.batch_size,
            reduction_fraction=self.reduction_fraction,
            sigma=self.sigma,
            stats=stats,
            first_batch=prefix,
            local_search=self.local_search,
        )


def _note_mlsl_deb_global_sample(
    stats: dict[str, Any],
    point: DebEvaluatedPoint,
) -> None:
    if not point.analysis_ok:
        stats["n_global_failed_analysis_samples"] += 1
    elif point.feasible:
        stats["n_global_feasible_samples"] += 1
    else:
        stats["n_global_infeasible_samples"] += 1


def _mlsl_deb_equ_info(algorithm: DiscreteMLSLDeb, *, warm_start: bool) -> str:
    """Return the info string for constraint-aware equal-descent MLSL."""
    sampler_label = "Sobol" if algorithm.sampler_name == "sobol" else "uniform"
    warm = "-WarmStart" if warm_start else ""
    return (
        f"dMLSL-L1-Deb-EquDesc{warm}-{sampler_label} "
        f"batch={algorithm.batch_size} "
        f"reduction_fraction={algorithm.reduction_fraction} "
        f"sigma={algorithm.sigma} sampler={algorithm.sampler_name}"
    )


class DiscreteMLSLDebEquDesc(DiscreteMLSLDeb):
    """Discrete MLSL whose Deb local search is equal descent."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        batch_size: int = 128,
        reduction_fraction: float = 0.10,
        sigma: float = 2.0,
        sampler: str = "sobol",
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            batch_size=batch_size,
            reduction_fraction=reduction_fraction,
            sigma=sigma,
            sampler=sampler,
        )
        self.name = "DISCRETE_MLSL_DEB_EQU_DESC_COLD_START"
        self.local_search = deb_equal_descent_local_search
        self.info = _mlsl_deb_equ_info(self, warm_start=False)


class DiscreteMLSLDebEquDescWarmStart(DiscreteMLSLDebWarmStart):
    """Equal-descent Deb MLSL from a Deb-ranked InitialDesign prefix."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        batch_size: int = 128,
        reduction_fraction: float = 0.10,
        sigma: float = 2.0,
        sampler: str = "sobol",
        *,
        initial_design_path: str,
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            batch_size=batch_size,
            reduction_fraction=reduction_fraction,
            sigma=sigma,
            sampler=sampler,
            initial_design_path=initial_design_path,
        )
        self.name = "DISCRETE_MLSL_DEB_EQU_DESC_WARM_START"
        self.local_search = deb_equal_descent_local_search
        self.info = _mlsl_deb_equ_info(self, warm_start=True)


def _note_mlsl_deb_search(
    stats: dict[str, Any],
    result: DebLocalSearchResult,
    unique_certified: set[tuple[int, ...]],
    unique_feasible: set[tuple[int, ...]],
    unique_infeasible: set[tuple[int, ...]],
) -> None:
    if getattr(result, "early_certificate", False):
        stats["n_equal_descent_early_certificates"] = (
            int(stats.get("n_equal_descent_early_certificates", 0)) + 1
        )
    stats["n_local_searches"] += 1
    stats["n_local_search_steps"] += int(result.n_steps)
    stats["n_local_search_evaluations"] += int(result.n_evaluations)
    stats["n_certification_evaluations"] += int(
        result.n_certification_evaluations
    )
    stats["evaluations_per_local_search"].append(int(result.n_evaluations))
    if not result.certified_local_optimum:
        stats["n_uncertified_local_searches"] += 1
        return
    stats["n_certified_local_searches"] += 1
    unique_certified.add(result.point.x)
    if result.point.feasible:
        stats["n_certified_feasible_local_searches"] += 1
        unique_feasible.add(result.point.x)
    else:
        stats["n_certified_infeasible_local_searches"] += 1
        unique_infeasible.add(result.point.x)
    stats["n_unique_certified_local_optima"] = len(unique_certified)
    stats["n_unique_certified_feasible_local_optima"] = len(unique_feasible)
    stats["n_unique_certified_infeasible_local_optima"] = len(unique_infeasible)


def _new_mlsl_deb_stats(seed: int | None) -> dict[str, Any]:
    return {
        "seed": seed,
        "n_evaluations": 0,
        "best_f": None,
        "best_x": None,
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
        "n_global_samples": 0,
        "n_global_feasible_samples": 0,
        "n_global_infeasible_samples": 0,
        "n_global_failed_analysis_samples": 0,
        "n_unique_global_sample_points": 0,
        "n_mlsl_iterations": 0,
        "n_reduced_candidates": 0,
        "n_candidates_radius_rejected": 0,
        "n_candidates_already_started": 0,
        "n_local_searches": 0,
        "n_local_search_steps": 0,
        "n_local_search_evaluations": 0,
        "n_certification_evaluations": 0,
        "n_certified_local_searches": 0,
        "n_uncertified_local_searches": 0,
        "n_certified_feasible_local_searches": 0,
        "n_certified_infeasible_local_searches": 0,
        "n_unique_certified_local_optima": 0,
        "n_unique_certified_feasible_local_optima": 0,
        "n_unique_certified_infeasible_local_optima": 0,
        "evaluations_per_local_search": [],
        "critical_radius_history": [],
        "local_search_start_points": [],
    }


def _finalize_mlsl_deb_stats(
    stats: dict[str, Any],
    evaluator: DebEvaluationManager | None,
) -> None:
    if evaluator is None:
        return
    stats["n_evaluations"] = evaluator.n_evaluations
    benchmark = evaluator.best_benchmark_point
    if benchmark is not None:
        stats["best_f"] = float(benchmark.y_p)
        stats["best_x"] = [int(value) for value in benchmark.x]
    best = evaluator.best_deb_point
    if best is not None:
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


_MLSL_DEB_SUMMARY_INT_KEYS = (
    "n_evaluations",
    "n_valid_evaluations",
    "n_feasible_evaluations",
    "n_infeasible_evaluations",
    "n_failed_analysis_evaluations",
    "n_sentinel_evaluations",
    "n_infeasible_plateau_terminations",
    "n_sentinel_infeasible_plateau_terminations",
    "n_global_samples",
    "n_global_feasible_samples",
    "n_global_infeasible_samples",
    "n_global_failed_analysis_samples",
    "n_unique_global_sample_points",
    "n_mlsl_iterations",
    "n_reduced_candidates",
    "n_candidates_radius_rejected",
    "n_candidates_already_started",
    "n_local_searches",
    "n_local_search_steps",
    "n_local_search_evaluations",
    "n_certification_evaluations",
    "n_certified_local_searches",
    "n_uncertified_local_searches",
    "n_certified_feasible_local_searches",
    "n_certified_infeasible_local_searches",
    "n_unique_certified_local_optima",
    "n_unique_certified_feasible_local_optima",
    "n_unique_certified_infeasible_local_optima",
)


def _build_mlsl_deb_summary(
    stats: Mapping[str, Any],
) -> dict[str, JSONValue]:
    summary: dict[str, JSONValue] = {
        "label": _MLSL_DEB_LABEL,
        "best_f": stats.get("best_f"),
        "best_x": stats.get("best_x"),
        "best_deb_x": stats.get("best_deb_x"),
        "best_deb_y": stats.get("best_deb_y"),
        "best_deb_phi": stats.get("best_deb_phi"),
        "best_deb_feasible": stats.get("best_deb_feasible"),
    }
    for name in _MLSL_DEB_SUMMARY_INT_KEYS:
        summary[name] = int(stats.get(name, 0))
    sentinel = stats.get("sentinel_evaluations_by_constraint") or {}
    summary["sentinel_evaluations_by_constraint"] = {
        str(name): int(count) for name, count in sentinel.items()
    }
    return summary


def apply_population_replacement_deb(
    child: PBHDebEvaluatedPoint,
    population: list[PBHDebEvaluatedPoint],
    space: DiscreteSearchSpace,
    dcut: float,
) -> str:
    """Apply the historical PBH geometry with strict Deb replacement."""
    if any(member.x == child.x for member in population):
        return "rejected"
    nearest = min(
        population,
        key=lambda member: (space.distance(child.x, member.x), member.x),
    )
    distance = space.distance(child.x, nearest.x)
    if distance < dcut:
        if pbh_strictly_better(child, nearest):
            _replace_member(population, nearest, child)
            return "near"
        return "rejected"
    worst = max(population, key=lambda member: (pbh_deb_key(member), member.x))
    if pbh_strictly_better(child, worst):
        _replace_member(population, worst, child)
        return "diverse"
    return "rejected"


class PopulationBasinHoppingDeb:
    """Constraint-aware PBH with strict Deb-ordered L1 local search."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        population_size: int = 8,
        perturbation_fraction: float = 0.15,
        dcut_factor: float = 0.5,
        initial_sampler: str = "sobol",
    ) -> None:
        self.max_evaluations = _require_budget(max_evaluations)
        self.seed = seed
        self.population_size = _require_positive_int("population_size", population_size)
        self.perturbation_fraction = _require_open_unit(
            "perturbation_fraction",
            perturbation_fraction,
        )
        self.dcut_factor = _require_positive_float("dcut_factor", dcut_factor)
        if initial_sampler not in {"sobol", "uniform"}:
            raise ValueError(
                "initial_sampler must be 'sobol' or 'uniform', "
                f"got {initial_sampler!r}"
            )
        self.initial_sampler = initial_sampler
        self.name = "POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START"
        self.local_search = deb_best_improvement_local_search
        self.info = self._info("PBH-L1-Deb")
        self.last_run_stats: dict[str, Any] = {}
        self._summary: dict[str, JSONValue] = {}

    def _info(self, label: str) -> str:
        return (
            f"{label} K={self.population_size} "
            f"rho={self.perturbation_fraction} "
            f"dcut_factor={self.dcut_factor} "
            f"initial={self.initial_sampler}"
        )

    def __call__(
        self,
        problem: Any,
        *,
        run_controller: Any = None,
    ) -> PopulationBasinHoppingDeb:
        stats = _new_stats(self.seed)
        self.last_run_stats = stats
        evaluator: PBHDebEvaluationManager | None = None
        try:
            evaluator = self._execute(problem, run_controller, stats)
        except BudgetExhausted:
            pass
        finally:
            stats.pop("_unique_optima", None)
            _finalize_stats(stats, evaluator)
            self._summary = _build_summary("PBH-L1-Deb", stats, evaluator)
        return self

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
    ) -> PBHDebEvaluationManager:
        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        _dimension, lower, upper = read_problem_bounds(problem)
        space = DiscreteSearchSpace(lower, upper)
        evaluator = PBHDebEvaluationManager(problem, controller, space)
        rng = np.random.default_rng(self.seed)
        sampler = DiscreteSampler(space, self.initial_sampler, self.seed, rng)
        population = _initial_population(
            evaluator,
            space,
            sampler,
            self.population_size,
            stats,
            local_search=self.local_search,
            note_search=_note_search,
            stop_after_start=_target_hit,
        )
        stats["initial_feasible_population_size"] = _feasible_count(population)
        continue_pbh_generations(
            population,
            evaluator,
            space,
            rng,
            self.perturbation_fraction,
            self.dcut_factor,
            stats,
            local_search=self.local_search,
            note_search=_note_search,
            replacement=lambda child, members, search_space, dcut: (
                _replace_and_diagnose(
                    child,
                    members,
                    search_space,
                    dcut,
                    stats,
                )
            ),
            stop_after_start=_target_hit,
            child_observer=lambda child: _note_child(stats, child),
            population_observer=lambda members: _note_population(
                stats,
                members,
            ),
        )
        stats["final_feasible_population_size"] = _feasible_count(population)
        return evaluator

    def summary_diagnostics(self) -> Mapping[str, JSONValue]:
        """Return compact, finite, JSON-safe run diagnostics."""
        return dict(self._summary)


class PopulationBasinHoppingDebWarmStart(PopulationBasinHoppingDeb):
    """PBH-L1-Deb initialized from a ranked InitialDesign prefix."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        population_size: int = 8,
        perturbation_fraction: float = 0.15,
        dcut_factor: float = 0.5,
        initial_sampler: str = "sobol",
        *,
        initial_design_path: str,
    ) -> None:
        super().__init__(
            max_evaluations=max_evaluations,
            seed=seed,
            population_size=population_size,
            perturbation_fraction=perturbation_fraction,
            dcut_factor=dcut_factor,
            initial_sampler=initial_sampler,
        )
        if not isinstance(initial_design_path, str) or not initial_design_path:
            raise ValueError("initial_design_path must be a non-empty string")
        self.initial_design_path = initial_design_path
        self.name = "POPULATION_BASIN_HOPPING_DEB_SI_DESC_WARM_START"
        self.info = self._info("PBH-L1-Deb-WarmStart")

    def _execute(
        self,
        problem: Any,
        run_controller: Any,
        stats: dict[str, Any],
    ) -> PBHDebEvaluationManager:
        from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

        _dimension, lower, upper = read_problem_bounds(problem)
        prefix = load_warm_start_prefix(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            count=self.population_size,
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
        evaluator = PBHDebEvaluationManager(problem, controller, space)
        rng = np.random.default_rng(self.seed)
        starts: list[PBHDebEvaluatedPoint] = []
        for vector in prefix:
            if not evaluator.has_budget():
                break
            starts.append(evaluator.evaluate(vector))
            if controller.target_hit:
                break
        population = (
            []
            if controller.target_hit
            else build_pbh_population_from_starts(
                starts,
                evaluator,
                space,
                stats,
                local_search=self.local_search,
                note_search=_note_search,
            )
        )
        stats["initial_population_size"] = len(population)
        stats["initial_feasible_population_size"] = _feasible_count(population)
        continue_pbh_generations(
            population,
            evaluator,
            space,
            rng,
            self.perturbation_fraction,
            self.dcut_factor,
            stats,
            local_search=self.local_search,
            note_search=_note_search,
            replacement=lambda child, members, search_space, dcut: (
                _replace_and_diagnose(
                    child,
                    members,
                    search_space,
                    dcut,
                    stats,
                )
            ),
            stop_after_start=_target_hit,
            child_observer=lambda child: _note_child(stats, child),
            population_observer=lambda members: _note_population(
                stats,
                members,
            ),
        )
        stats["final_feasible_population_size"] = _feasible_count(population)
        return evaluator


class PopulationBasinHoppingDebEquDesc(PopulationBasinHoppingDeb):
    """Constraint-aware PBH whose inner local search is equal descent."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        population_size: int = 8,
        perturbation_fraction: float = 0.15,
        dcut_factor: float = 0.5,
        initial_sampler: str = "sobol",
    ) -> None:
        super().__init__(
            max_evaluations,
            seed=seed,
            population_size=population_size,
            perturbation_fraction=perturbation_fraction,
            dcut_factor=dcut_factor,
            initial_sampler=initial_sampler,
        )
        self.name = "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_COLD_START"
        self.local_search = deb_equal_descent_local_search
        self.info = self._info("PBH-L1-Deb-EquDesc")


class PopulationBasinHoppingDebEquDescWarmStart(PopulationBasinHoppingDebWarmStart):
    """Equal-descent Deb PBH from a Deb-ranked InitialDesign prefix."""

    def __init__(
        self,
        max_evaluations: int,
        seed: int | None = None,
        population_size: int = 8,
        perturbation_fraction: float = 0.15,
        dcut_factor: float = 0.5,
        initial_sampler: str = "sobol",
        *,
        initial_design_path: str,
    ) -> None:
        super().__init__(
            max_evaluations=max_evaluations,
            seed=seed,
            population_size=population_size,
            perturbation_fraction=perturbation_fraction,
            dcut_factor=dcut_factor,
            initial_sampler=initial_sampler,
            initial_design_path=initial_design_path,
        )
        self.name = "POPULATION_BASIN_HOPPING_DEB_EQU_DESC_WARM_START"
        self.local_search = deb_equal_descent_local_search
        self.info = self._info("PBH-L1-Deb-EquDesc-WarmStart")


def validate_population_basin_hopping_deb_parameters(
    parameters: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    """Validate PBH-Deb open-bound parameters with token-specific messages."""
    return _validate_population_basin_hopping_parameters(
        parameters,
        token="POPULATION_BASIN_HOPPING_DEB_SI_DESC_COLD_START",
    )


def _new_stats(seed: int | None) -> dict[str, Any]:
    return {
        "seed": seed,
        "n_evaluations": 0,
        "n_local_searches": 0,
        "n_local_search_steps": 0,
        "n_certified_local_searches": 0,
        "n_unique_certified_local_optima": 0,
        "evaluations_per_local_search": [],
        "best_f": None,
        "best_x": None,
        "initial_population_size": 0,
        "initial_feasible_population_size": 0,
        "final_feasible_population_size": 0,
        "population_feasible_count_history": [],
        "n_generations": 0,
        "n_children": 0,
        "n_feasible_children": 0,
        "n_infeasible_children": 0,
        "n_near_replacements": 0,
        "n_diverse_replacements": 0,
        "n_rejected_children": 0,
        "n_near_replacements_by_feasible_child": 0,
        "n_diverse_replacements_by_feasible_child": 0,
        "n_infeasible_children_with_all_feasible_population": 0,
        "n_rejected_infeasible_children_with_all_feasible_population": 0,
        "dcut": None,
        "initial_mean_distance": None,
        "population_mean_distance_history": [],
    }


def _note_search(
    stats: dict[str, Any],
    result: LocalSearchResult | PBHDebLocalSearchResult,
    unique: set[tuple[int, ...]],
) -> None:
    """Record one local-search result on the shared PBH and MLSL counters.

    Penalty results and Deb results both reach this function. A second
    definition of the same name would replace this one at import time.
    """
    stats["n_local_searches"] += 1
    stats["n_local_search_steps"] += result.n_steps
    stats["evaluations_per_local_search"].append(result.n_evaluations)
    if result.certified_local_optimum:
        stats["n_certified_local_searches"] += 1
        unique.add(result.point.x)
    stats["n_unique_certified_local_optima"] = len(unique)
    if getattr(result, "early_certificate", False):
        stats["n_equal_descent_early_certificates"] = (
            int(stats.get("n_equal_descent_early_certificates", 0)) + 1
        )


def _target_hit(evaluator: PBHDebEvaluationManager) -> bool:
    return bool(evaluator.controller.target_hit)


def _feasible_count(population: list[PBHDebEvaluatedPoint]) -> int:
    return sum(1 for member in population if member.feasible)


def _note_population(
    stats: dict[str, Any],
    population: list[PBHDebEvaluatedPoint],
) -> None:
    stats["population_feasible_count_history"].append(
        _feasible_count(population)
    )


def _note_child(
    stats: dict[str, Any],
    child: PBHDebEvaluatedPoint,
) -> None:
    key = "n_feasible_children" if child.feasible else "n_infeasible_children"
    stats[key] += 1


def _replace_and_diagnose(
    child: PBHDebEvaluatedPoint,
    population: list[PBHDebEvaluatedPoint],
    space: DiscreteSearchSpace,
    dcut: float,
    stats: dict[str, Any],
) -> str:
    all_feasible_before = all(member.feasible for member in population)
    outcome = apply_population_replacement_deb(child, population, space, dcut)
    if child.feasible and outcome == "near":
        stats["n_near_replacements_by_feasible_child"] += 1
    elif child.feasible and outcome == "diverse":
        stats["n_diverse_replacements_by_feasible_child"] += 1
    if all_feasible_before and not child.feasible:
        stats["n_infeasible_children_with_all_feasible_population"] += 1
        if outcome == "rejected":
            stats[
                "n_rejected_infeasible_children_with_all_feasible_population"
            ] += 1
    return outcome


def _finalize_stats(
    stats: dict[str, Any],
    evaluator: PBHDebEvaluationManager | None,
) -> None:
    if evaluator is None:
        return
    stats["n_evaluations"] = evaluator.n_evaluations
    benchmark = evaluator.best_benchmark_point
    if benchmark is not None:
        stats["best_f"] = float(benchmark.y_p)
        stats["best_x"] = [int(value) for value in benchmark.x]
    best = evaluator.best_deb_point
    stats["best_deb_x"] = (
        None if best is None else [int(value) for value in best.x]
    )
    stats["best_deb_y"] = None if best is None else float(best.y)
    stats["best_deb_phi"] = None if best is None else float(best.phi)
    stats["best_deb_feasible"] = None if best is None else bool(best.feasible)
    stats.update(evaluator.diagnostics)


def _build_summary(
    label: str,
    stats: Mapping[str, Any],
    evaluator: PBHDebEvaluationManager | None,
) -> dict[str, JSONValue]:
    summary: dict[str, JSONValue] = {
        "label": label,
        "best_deb_x": None,
        "best_deb_y": None,
        "best_deb_phi": None,
        "best_deb_feasible": None,
    }
    for name in (
        "n_local_searches",
        "n_local_search_steps",
        "n_certified_local_searches",
        "n_unique_certified_local_optima",
        "initial_population_size",
        "initial_feasible_population_size",
        "final_feasible_population_size",
        "n_generations",
        "n_children",
        "n_feasible_children",
        "n_infeasible_children",
        "n_near_replacements",
        "n_diverse_replacements",
        "n_rejected_children",
        "n_near_replacements_by_feasible_child",
        "n_diverse_replacements_by_feasible_child",
        "n_infeasible_children_with_all_feasible_population",
        "n_rejected_infeasible_children_with_all_feasible_population",
    ):
        summary[name] = int(stats.get(name, 0))
    if evaluator is None:
        return summary
    for name, value in evaluator.diagnostics.items():
        if name == "sentinel_evaluations_by_constraint":
            summary[name] = {
                str(key): int(count) for key, count in value.items()
            }
        else:
            summary[name] = int(value)
    best = evaluator.best_deb_point
    if best is not None:
        summary["best_deb_x"] = [int(value) for value in best.x]
        summary["best_deb_y"] = float(best.y)
        summary["best_deb_phi"] = float(best.phi)
        summary["best_deb_feasible"] = bool(best.feasible)
    return summary


def _deb_winner(
    points: list[PBHDebEvaluatedPoint],
) -> PBHDebEvaluatedPoint | None:
    if not points:
        return None
    return min(points, key=lambda point: (pbh_deb_key(point), point.x))


def _replace_member(
    population: list[PBHDebEvaluatedPoint],
    target: PBHDebEvaluatedPoint,
    child: PBHDebEvaluatedPoint,
) -> None:
    for index, member in enumerate(population):
        if member.x == target.x:
            population[index] = child
            return

