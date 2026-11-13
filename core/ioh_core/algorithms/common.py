"""
Shared candidate conversion, bounds validation, and evaluation-budget helpers.

Continuous-to-integer conversion uses round-half-away-from-zero, then clips to
inclusive IOH catalogue-index bounds. This avoids Python 3 / NumPy banker
rounding at exact half-integers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class CandidateError(ValueError):
    """
    Raised when a candidate vector is invalid for an IOH integer problem.

    Typical causes: wrong dimension, non-finite values, or indices outside the
    inclusive catalogue bounds.
    """


class BudgetError(ValueError):
    """
    Raised when an adapter would exceed the hard IOH evaluation budget.
    """


def round_half_away_from_zero(value: float) -> int:
    """
    Round ``value`` to the nearest integer; ties round away from zero.

    Parameters
    ----------
    value : float
        Continuous coordinate.

    Returns
    -------
    int
        Rounded integer.

    Notes
    -----
    Examples: ``0.5 → 1``, ``-0.5 → -1``, ``1.5 → 2``, ``2.5 → 3``.
    """
    if not math.isfinite(value):
        raise CandidateError(f"candidate coordinate must be finite, got {value!r}")
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def seeded_uniform_continuous_start(
    lower_bounds: Sequence[float | int],
    upper_bounds: Sequence[float | int],
    seed: int,
) -> list[float]:
    """
    Draw one continuous start in the inclusive box using ``seed``.

    Each coordinate is sampled independently and uniformly on
    ``[lower_i, upper_i]`` (both endpoints reachable). Degenerate axes
    (``lower_i == upper_i``) return that bound. Used by local DFO adapters
    (COBYLA, SBPLX) for seed-reproducible multistart.

    Parameters
    ----------
    lower_bounds : sequence of float or int
        Inclusive lower bounds (catalogue-index space as floats).
    upper_bounds : sequence of float or int
        Inclusive upper bounds; must match ``lower_bounds`` length.
    seed : int
        Deterministic RNG seed (task ``random_seed``).

    Returns
    -------
    list of float
        Continuous start vector of the same length as the bounds.

    Raises
    ------
    CandidateError
        If bounds are empty, mismatched, non-finite, or inverted.
    """
    if len(lower_bounds) != len(upper_bounds):
        raise CandidateError(
            "lower_bounds and upper_bounds must have the same length, "
            f"got {len(lower_bounds)} and {len(upper_bounds)}"
        )
    if len(lower_bounds) < 1:
        raise CandidateError("bounds must be non-empty")

    # Local import keeps NumPy optional for callers that only need rounding.
    import numpy as np

    rng = np.random.default_rng(int(seed))
    start: list[float] = []
    for lower_raw, upper_raw in zip(lower_bounds, upper_bounds):
        lower = float(lower_raw)
        upper = float(upper_raw)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise CandidateError(
                f"bounds must be finite, got lower={lower_raw!r}, "
                f"upper={upper_raw!r}"
            )
        if upper < lower:
            raise CandidateError(
                f"upper bound {upper} is below lower bound {lower}"
            )
        if upper == lower:
            start.append(lower)
        else:
            start.append(lower + float(rng.random()) * (upper - lower))
    return start


def read_problem_bounds(problem: Any) -> tuple[int, list[int], list[int]]:
    """
    Read dimension and inclusive integer bounds from an IOH-like problem.

    Parameters
    ----------
    problem : Any
        Object exposing ``meta_data.n_variables`` and ``bounds.lb`` / ``.ub``.

    Returns
    -------
    tuple
        ``(dimension, lower_bounds, upper_bounds)`` as Python ints.

    Raises
    ------
    CandidateError
        If dimension/bounds are missing or inconsistent.
    """
    try:
        dimension = int(problem.meta_data.n_variables)
    except Exception as exc:
        raise CandidateError("problem.meta_data.n_variables is required") from exc
    if dimension < 1:
        raise CandidateError(f"dimension must be >= 1, got {dimension}")

    try:
        lb_raw = problem.bounds.lb
        ub_raw = problem.bounds.ub
    except Exception as exc:
        raise CandidateError("problem.bounds.lb/ub are required") from exc

    if len(lb_raw) != dimension or len(ub_raw) != dimension:
        raise CandidateError(
            f"bounds length must equal dimension {dimension}, "
            f"got lb={len(lb_raw)} ub={len(ub_raw)}"
        )

    lower = [int(v) for v in lb_raw]
    upper = [int(v) for v in ub_raw]
    for index, (lo, hi) in enumerate(zip(lower, upper)):
        if lo > hi:
            raise CandidateError(
                f"bound[{index}] lower {lo} exceeds upper {hi}"
            )
    return dimension, lower, upper


def _as_float_sequence(values: Sequence[Any], *, dimension: int) -> list[float]:
    """
    Coerce a candidate sequence to floats of the expected dimension.

    Parameters
    ----------
    values : sequence
        Candidate coordinates (lists, tuples, or NumPy arrays).
    dimension : int
        Expected length.

    Returns
    -------
    list of float
        Coerced coordinates.

    Raises
    ------
    CandidateError
        If the length is wrong or values are non-finite / non-numeric.
    """
    try:
        sequence = list(values)
    except TypeError as exc:
        raise CandidateError(f"candidate must be a sequence, got {type(values)!r}") from exc
    if len(sequence) != dimension:
        raise CandidateError(
            f"candidate dimension {len(sequence)} != problem dimension {dimension}"
        )
    out: list[float] = []
    for index, raw in enumerate(sequence):
        try:
            # Reject bool explicitly (bool is an int subclass).
            if isinstance(raw, bool):
                raise TypeError("bool is not a numeric candidate coordinate")
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise CandidateError(
                f"candidate[{index}] must be numeric, got {raw!r}"
            ) from exc
        if not math.isfinite(number):
            raise CandidateError(
                f"candidate[{index}] must be finite, got {number!r}"
            )
        out.append(number)
    return out


def validate_integer_candidate(
    values: Sequence[Any],
    *,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> list[int]:
    """
    Validate an integer catalogue-index vector.

    Parameters
    ----------
    values : sequence
        Proposed integer candidate.
    lower_bounds : sequence of int
        Inclusive lower bounds.
    upper_bounds : sequence of int
        Inclusive upper bounds.

    Returns
    -------
    list of int
        Validated integer vector.

    Raises
    ------
    CandidateError
        If dimension, types, or bounds checks fail.
    """
    dimension = len(lower_bounds)
    if len(upper_bounds) != dimension:
        raise CandidateError("lower/upper bound lengths disagree")
    floats = _as_float_sequence(values, dimension=dimension)
    ints: list[int] = []
    for index, number in enumerate(floats):
        if not float(number).is_integer():
            raise CandidateError(
                f"candidate[{index}] must be an integer index, got {number!r}"
            )
        value = int(number)
        lo = int(lower_bounds[index])
        hi = int(upper_bounds[index])
        if value < lo or value > hi:
            raise CandidateError(
                f"candidate[{index}]={value} outside inclusive bounds [{lo}, {hi}]"
            )
        ints.append(value)
    return ints


def continuous_to_integer_candidate(
    values: Sequence[Any],
    *,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> list[int]:
    """
    Convert a continuous vector with round-half-away-from-zero, then clip.

    Parameters
    ----------
    values : sequence
        Continuous candidate (lists/tuples/NumPy arrays/scalars-in-sequence).
    lower_bounds : sequence of int
        Inclusive lower bounds used for clipping.
    upper_bounds : sequence of int
        Inclusive upper bounds used for clipping.

    Returns
    -------
    list of int
        Integer catalogue-index vector inside inclusive bounds.

    Raises
    ------
    CandidateError
        If dimension or numeric conversion fails.
    """
    dimension = len(lower_bounds)
    if len(upper_bounds) != dimension:
        raise CandidateError("lower/upper bound lengths disagree")
    floats = _as_float_sequence(values, dimension=dimension)
    ints: list[int] = []
    for index, number in enumerate(floats):
        rounded = round_half_away_from_zero(number)
        lo = int(lower_bounds[index])
        hi = int(upper_bounds[index])
        if rounded < lo:
            rounded = lo
        elif rounded > hi:
            rounded = hi
        ints.append(rounded)
    return ints


def clip_integer_candidate(
    values: Sequence[int],
    *,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> list[int]:
    """
    Clip an integer vector to inclusive bounds without rounding.

    Parameters
    ----------
    values : sequence of int
        Integer candidate.
    lower_bounds : sequence of int
        Inclusive lower bounds.
    upper_bounds : sequence of int
        Inclusive upper bounds.

    Returns
    -------
    list of int
        Clipped vector.
    """
    dimension = len(lower_bounds)
    if len(values) != dimension or len(upper_bounds) != dimension:
        raise CandidateError(
            f"clip dimension mismatch: values={len(values)} "
            f"bounds=({len(lower_bounds)}, {len(upper_bounds)})"
        )
    return [
        min(max(int(value), int(lo)), int(hi))
        for value, lo, hi in zip(values, lower_bounds, upper_bounds)
    ]


@dataclass
class EvaluationBudget:
    """
    Hard upper bound on IOH problem evaluations for one algorithm call.

    Tracks remaining evaluations from ``problem.state.evaluations`` relative to
    the count observed when the budget was created. Adapters must not own
    logger attach/detach or ``problem.reset()``.

    Parameters
    ----------
    max_evaluations : int
        Hard upper bound on IOH calls for this run (>= 1).
    problem : Any
        IOH-like problem with ``state.evaluations``.
    """

    max_evaluations: int
    problem: Any
    _baseline_evaluations: int

    def __init__(self, max_evaluations: int, problem: Any) -> None:
        if not isinstance(max_evaluations, int) or isinstance(max_evaluations, bool):
            raise BudgetError(
                f"max_evaluations must be an int >= 1, got {max_evaluations!r}"
            )
        if max_evaluations < 1:
            raise BudgetError(
                f"max_evaluations must be an int >= 1, got {max_evaluations!r}"
            )
        self.max_evaluations = int(max_evaluations)
        self.problem = problem
        state = getattr(problem, "state", None)
        self._baseline_evaluations = int(getattr(state, "evaluations", 0) or 0)

    @property
    def used(self) -> int:
        """
        Number of IOH evaluations consumed since the budget was created.

        Returns
        -------
        int
            Non-negative evaluation count delta.
        """
        state = getattr(self.problem, "state", None)
        current = int(getattr(state, "evaluations", 0) or 0)
        return max(0, current - self._baseline_evaluations)

    @property
    def remaining(self) -> int:
        """
        Remaining IOH evaluations before the hard budget is exhausted.

        Returns
        -------
        int
            ``max(0, max_evaluations - used)``.
        """
        return max(0, self.max_evaluations - self.used)

    def can_evaluate(self, count: int = 1) -> bool:
        """
        Return whether ``count`` further IOH evaluations are allowed.

        Parameters
        ----------
        count : int, optional
            Requested evaluation count (default ``1``).

        Returns
        -------
        bool
            ``True`` when the remaining budget is sufficient.
        """
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise BudgetError(f"count must be an int >= 1, got {count!r}")
        return self.remaining >= count

    def require(self, count: int = 1) -> None:
        """
        Raise :class:`BudgetError` when ``count`` evaluations are unavailable.

        Parameters
        ----------
        count : int, optional
            Requested evaluation count.

        Raises
        ------
        BudgetError
            If the remaining budget is insufficient.
        """
        if not self.can_evaluate(count):
            raise BudgetError(
                f"evaluation budget exhausted: requested {count}, "
                f"remaining {self.remaining}, max {self.max_evaluations}"
            )


_TERMINATION_MODES = frozenset({"fixed_budget", "fixed_target"})
_TARGET_REFERENCES = frozenset({"all"})


class RunController:
    """
    Target-aware hard evaluation controller for one algorithm call.

    Wraps :class:`EvaluationBudget`. Authoritative evaluation counts come from
    ``problem.state.evaluations``. Every finite observed ``y_p`` can update
    the tracked best and satisfy a fixed target.

    Parameters
    ----------
    problem : Any
        IOH-like problem with ``state.evaluations``.
    max_evaluations : int
        Hard upper bound on IOH calls (>= 1).
    mode : str, optional
        ``\"fixed_budget\"`` or ``\"fixed_target\"``.
    target_threshold : float or None, optional
        Minimisation threshold for fixed-target mode.
    target_reference : str or None, optional
        ``\"all\"`` in fixed-target mode; ``None`` for fixed-budget.
    """

    def __init__(
        self,
        problem: Any,
        max_evaluations: int,
        *,
        mode: str = "fixed_budget",
        target_threshold: float | None = None,
        target_reference: str | None = None,
    ) -> None:
        if mode not in _TERMINATION_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_TERMINATION_MODES)}, got {mode!r}"
            )
        if mode == "fixed_budget":
            if target_threshold is not None or target_reference is not None:
                raise ValueError(
                    "target_threshold and target_reference must be None when "
                    "mode='fixed_budget'"
                )
        else:
            if target_threshold is None:
                raise ValueError(
                    "target_threshold is required when mode='fixed_target'"
                )
            if not isinstance(target_threshold, (int, float)) or isinstance(
                target_threshold, bool
            ):
                raise ValueError(
                    f"target_threshold must be a finite number, got "
                    f"{target_threshold!r}"
                )
            threshold = float(target_threshold)
            if not math.isfinite(threshold) or threshold < 0.0:
                raise ValueError(
                    "target_threshold must be a finite number >= 0, "
                    f"got {target_threshold!r}"
                )
            if target_reference not in _TARGET_REFERENCES:
                raise ValueError(
                    "target_reference must be one of "
                    f"{sorted(_TARGET_REFERENCES)}, got {target_reference!r}"
                )
            target_threshold = threshold

        self.problem = problem
        self.max_evaluations = int(max_evaluations)
        self.mode = mode
        self.target_threshold = target_threshold
        self.target_reference = target_reference
        self.best_finite_y_p = math.inf
        self.target_hit = False
        self.evaluations_to_target: int | None = None
        self._budget = EvaluationBudget(max_evaluations, problem)

    @property
    def used(self) -> int:
        """IOH evaluations consumed since controller creation."""
        return self._budget.used

    @property
    def remaining(self) -> int:
        """Remaining IOH evaluations before the hard budget is exhausted."""
        return self._budget.remaining

    @property
    def budget_exhausted(self) -> bool:
        """Return whether the hard evaluation budget is exhausted."""
        return self._budget.remaining == 0

    def can_evaluate(self, count: int = 1) -> bool:
        """
        Return whether ``count`` further IOH evaluations are allowed.

        Returns ``False`` after a target hit or when the budget is insufficient.
        """
        if self.target_hit:
            return False
        return self._budget.can_evaluate(count)

    def require_evaluation(self, count: int = 1) -> None:
        """
        Raise :class:`BudgetError` when another evaluation is not allowed.

        This remains a programming/contract violation, not a normal target-hit
        signal.
        """
        if self.target_hit:
            raise BudgetError(
                "evaluation refused after target hit at evaluation "
                f"{self.evaluations_to_target}"
            )
        self._budget.require(count)

    def observe_evaluation(self, y_p: Any) -> None:
        """
        Observe one successful IOH return value and update target state.

        Every finite penalised objective updates the tracked best. A
        fixed-target run hits when that best is at or below the threshold.

        Parameters
        ----------
        y_p : Any
            Penalised objective returned by ``problem(candidate)``.
        """
        try:
            value = float(y_p)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"y_p must be numeric, got {y_p!r}") from exc

        considers = math.isfinite(value)
        if considers and value < self.best_finite_y_p:
            self.best_finite_y_p = value

        if (
            self.mode == "fixed_target"
            and not self.target_hit
            and considers
            and self.best_finite_y_p <= float(self.target_threshold)
        ):
            self.target_hit = True
            self.evaluations_to_target = int(self._budget.used)

    def should_stop(self) -> bool:
        """Return whether the run should stop (target hit or budget exhausted)."""
        return self.target_hit or self.budget_exhausted

    def stop_reason(self) -> str | None:
        """
        Return the authoritative stop reason, or ``None`` while still running.

        When the target is first reached on the final allowed evaluation, the
        reason is ``\"target_hit\"`` even though ``budget_exhausted`` is also
        true.
        """
        if self.target_hit:
            return "target_hit"
        if self.budget_exhausted:
            return "budget_exhausted"
        return None


# IOH C++ wrapped problems reject arbitrary Python attributes, so EvalContext
# attachment for feasibility reads uses this process-local registry keyed by
# ``id(problem)``. Fake/test problems may still set ``hdb_eval_context``.
_EVAL_CONTEXT_BY_PROBLEM_ID: dict[int, Any] = {}


def attach_eval_context(problem: Any, eval_context: Any) -> None:
    """
    Register ``eval_context`` for feasibility reads on ``problem``.

    Parameters
    ----------
    problem : Any
        IOH-like problem about to be evaluated.
    eval_context : Any
        Bundle ``EvalContext`` exposing ``last_data``.
    """
    _EVAL_CONTEXT_BY_PROBLEM_ID[id(problem)] = eval_context
    try:
        problem.hdb_eval_context = eval_context
    except (AttributeError, TypeError):
        pass


def detach_eval_context(problem: Any) -> None:
    """
    Remove any EvalContext registration for ``problem``.

    Parameters
    ----------
    problem : Any
        IOH-like problem whose registration should be cleared.
    """
    _EVAL_CONTEXT_BY_PROBLEM_ID.pop(id(problem), None)
    try:
        if hasattr(problem, "hdb_eval_context"):
            delattr(problem, "hdb_eval_context")
    except (AttributeError, TypeError):
        pass


def read_last_evaluation_feasibility(problem: Any) -> bool:
    """
    Return feasibility for the most recent evaluation when available.

    Resolution order:

    1. ``problem.hdb_eval_context.last_data["feasible"]`` when set;
    2. EvalContext registered via :func:`attach_eval_context`;
    3. ``problem.hdb_last_feasible`` when set to a bool (tests / wrappers);
    4. otherwise ``True`` on bare IOH-like fakes.

    Parameters
    ----------
    problem : Any
        IOH-like problem just evaluated.

    Returns
    -------
    bool
        Feasibility flag stored for the most recent evaluation.
    """
    ctx = getattr(problem, "hdb_eval_context", None)
    if ctx is None:
        ctx = _EVAL_CONTEXT_BY_PROBLEM_ID.get(id(problem))
    if ctx is not None:
        data = getattr(ctx, "last_data", None)
        if isinstance(data, dict) and "feasible" in data:
            return bool(data["feasible"])
    override = getattr(problem, "hdb_last_feasible", None)
    if isinstance(override, bool):
        return override
    return True


def _resolve_eval_context(problem: Any) -> Any | None:
    """
    Return the EvalContext attached to ``problem``, if any.

    Parameters
    ----------
    problem : Any
        IOH-like problem.

    Returns
    -------
    Any or None
        Attached context or ``None``.
    """
    ctx = getattr(problem, "hdb_eval_context", None)
    if ctx is None:
        ctx = _EVAL_CONTEXT_BY_PROBLEM_ID.get(id(problem))
    return ctx


def get_attached_eval_context(problem: Any) -> Any | None:
    """
    Return the EvalContext attached to ``problem``, if any.

    Parameters
    ----------
    problem : Any
        IOH-like problem.

    Returns
    -------
    Any or None
        Attached context or ``None``.
    """
    return _resolve_eval_context(problem)


def read_last_evaluation_record(problem: Any) -> Mapping[str, Any]:
    """
    Return the cached analysis dict for the most recent ``problem(x)`` call.

    Resolution order mirrors :func:`read_last_evaluation_feasibility`:
    ``problem.hdb_eval_context`` then the process-local attach registry.

    Parameters
    ----------
    problem : Any
        IOH-like problem just evaluated.

    Returns
    -------
    mapping
        Cached analysis record (``y``, ``y_p``, ``violations_``, …).

    Raises
    ------
    ValueError
        If no EvalContext is attached or ``last_data`` is missing/invalid.
    """
    ctx = _resolve_eval_context(problem)
    if ctx is None:
        raise ValueError(
            "no EvalContext attached to problem; call attach_eval_context "
            "before reading the last evaluation record"
        )
    data = getattr(ctx, "last_data", None)
    if not isinstance(data, dict):
        raise ValueError(
            "EvalContext.last_data is missing or invalid; evaluate "
            "problem(x) before reading the last evaluation record"
        )
    return data


def read_last_evaluation_objective_y(problem: Any) -> float:
    """
    Return unpenalized ``y`` from the last evaluation record.

    Parameters
    ----------
    problem : Any
        IOH-like problem just evaluated.

    Returns
    -------
    float
        Unpenalized objective (may be non-finite).

    Raises
    ------
    ValueError
        If the record cannot be read or ``y`` is missing/non-numeric.
    """
    record = read_last_evaluation_record(problem)
    if "y" not in record:
        raise ValueError("last evaluation record is missing key 'y'")
    try:
        return float(record["y"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"last evaluation record 'y' must be numeric, got {record['y']!r}"
        ) from exc


def read_last_evaluation_violations(problem: Any) -> list[float]:
    """
    Return ordered active clipped violation values ``max(0, η−1)``.

    Prefer :func:`read_last_evaluation_constraint_slacks` for library-native
    inequality channels that expect signed ``g ≤ 0`` (Pymoo, DFL, NOMAD, CEI).
    This helper remains for callers that need non-negative violation magnitudes.

    Order follows EvalContext ``_c_names`` (same order as
    ``ensure_constraints`` / ``active_constraint_names``).
    Missing keys on a successful analysis record become ``0.0``.
    Fail-closed: non-finite / negative stored values raise ``ValueError``.

    Parameters
    ----------
    problem : Any
        IOH-like problem just evaluated.

    Returns
    -------
    list of float
        Ordered ``max(0, η−1)`` violation values.

    Raises
    ------
    ValueError
        If the context/record is missing, constraint names are unavailable,
        or a stored violation is invalid.
    """
    ctx = _resolve_eval_context(problem)
    if ctx is None:
        raise ValueError(
            "no EvalContext attached to problem; call attach_eval_context "
            "before reading last evaluation violations"
        )
    names = getattr(ctx, "_c_names", None)
    if not isinstance(names, list) or not names:
        raise ValueError(
            "EvalContext has no active constraint names; call "
            "ensure_constraints before reading violations"
        )
    record = read_last_evaluation_record(problem)
    violations = record.get("violations_")
    if violations is None:
        violations = {}
    if not isinstance(violations, Mapping):
        raise ValueError(
            "last evaluation record 'violations_' must be a mapping, "
            f"got {type(violations).__name__}"
        )
    ordered: list[float] = []
    for name in names:
        if name not in violations:
            ordered.append(0.0)
            continue
        raw = violations[name]
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"violation for {name!r} must be numeric, got {raw!r}"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"violation for {name!r} must be finite and >= 0, got {value!r}"
            )
        ordered.append(value)
    return ordered


def read_last_evaluation_constraint_slacks(problem: Any) -> list[float]:
    """
    Return ordered signed constraint values ``η − 1`` for DFL-style ``g ≤ 0``.

    Order follows EvalContext ``_c_names`` (same order as
    ``ensure_constraints`` / ``active_constraint_names``). Values come from
    ``constraint_values`` (utilizations η); missing keys on a successful
    analysis record become ``-1.0`` (η = 0). Fail-closed: non-finite stored
    utilizations raise ``ValueError``.

    Parameters
    ----------
    problem : Any
        IOH-like problem just evaluated.

    Returns
    -------
    list of float
        Ordered ``η − 1`` slack/violation values (negative ⇒ feasible slack).

    Raises
    ------
    ValueError
        If the context/record is missing, constraint names are unavailable,
        or a stored utilization is invalid.
    """
    ctx = _resolve_eval_context(problem)
    if ctx is None:
        raise ValueError(
            "no EvalContext attached to problem; call attach_eval_context "
            "before reading last evaluation constraint slacks"
        )
    names = getattr(ctx, "_c_names", None)
    if not isinstance(names, list) or not names:
        raise ValueError(
            "EvalContext has no active constraint names; call "
            "ensure_constraints before reading constraint slacks"
        )
    record = read_last_evaluation_record(problem)
    utilizations = record.get("constraint_values")
    if utilizations is None:
        utilizations = {}
    if not isinstance(utilizations, Mapping):
        raise ValueError(
            "last evaluation record 'constraint_values' must be a mapping, "
            f"got {type(utilizations).__name__}"
        )
    ordered: list[float] = []
    for name in names:
        if name not in utilizations:
            ordered.append(-1.0)
            continue
        raw = utilizations[name]
        try:
            eta = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"constraint utilization for {name!r} must be numeric, "
                f"got {raw!r}"
            ) from exc
        if not math.isfinite(eta):
            raise ValueError(
                f"constraint utilization for {name!r} must be finite, "
                f"got {eta!r}"
            )
        ordered.append(eta - 1.0)
    return ordered


def resolve_run_controller(
    problem: Any,
    *,
    max_evaluations: int,
    run_controller: RunController | None = None,
) -> RunController:
    """
    Return a validated controller for one algorithm ``__call__``.

    When ``run_controller`` is ``None``, create a fixed-budget controller from
    ``max_evaluations``. When a controller is supplied, reject mismatched
    problem identity or budget.

    Parameters
    ----------
    problem : Any
        IOH-like problem instance for this run.
    max_evaluations : int
        Algorithm hard budget.
    run_controller : RunController or None, optional
        Optional externally constructed controller.

    Returns
    -------
    RunController
        Controller bound to ``problem``.

    Raises
    ------
    ValueError
        If the supplied controller does not match ``problem`` or
        ``max_evaluations``.
    """
    if run_controller is None:
        return RunController(
            problem,
            max_evaluations,
            mode="fixed_budget",
            target_threshold=None,
            target_reference=None,
        )
    if run_controller.problem is not problem:
        raise ValueError(
            "run_controller.problem must be the same object passed to the "
            "algorithm"
        )
    if int(run_controller.max_evaluations) != int(max_evaluations):
        raise ValueError(
            "run_controller.max_evaluations "
            f"{run_controller.max_evaluations!r} does not match algorithm "
            f"max_evaluations {max_evaluations!r}"
        )
    return run_controller

