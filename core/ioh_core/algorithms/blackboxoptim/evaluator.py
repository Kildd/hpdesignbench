"""
Shared IOH batch evaluator and helpers for blackboxoptim adapters.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from core.ioh_core.algorithms.common import (
    CandidateError,
    get_attached_eval_context,
    read_last_evaluation_constraint_slacks,
    read_last_evaluation_objective_y,
    validate_integer_candidate,
)

BLACKBOXOPT_POST_STOP_SENTINEL = 1.0e30


def ensure_numpy_inf_alias() -> None:
    """
    Restore ``np.Inf`` for blackboxoptim GOSAC on NumPy 2.

    ``blackboxoptim.optimize.gosac`` still uses ``np.Inf``, which NumPy 2
    removed. This alias is a local compatibility shim, not a library default.
    """
    if not hasattr(np, "Inf"):
        np.Inf = np.inf  # type: ignore[attr-defined]


class BlackboxoptimEvaluator:
    """
    Batch objective wrapper for blackboxoptim ``fun(x)`` calls.

    Each row is validated as an inclusive integer catalogue vector, evaluated
    through the IOH problem (penalized ``y_p``), and observed on the shared
    :class:`~core.ioh_core.algorithms.common.RunController`. When the controller
    forbids further IOH calls, remaining batch rows receive
    :data:`BLACKBOXOPT_POST_STOP_SENTINEL` without an IOH call.
    """

    def __init__(
        self,
        problem: Any,
        controller: Any,
        lower_bounds: Sequence[int],
        upper_bounds: Sequence[int],
    ) -> None:
        self._problem = problem
        self._controller = controller
        self._lower_bounds = list(lower_bounds)
        self._upper_bounds = list(upper_bounds)
        self.stop_requested = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(x, dtype=float))
        objectives = np.empty(points.shape[0], dtype=float)
        for row_index, row in enumerate(points):
            if not self._controller.can_evaluate():
                objectives[row_index:] = BLACKBOXOPT_POST_STOP_SENTINEL
                self.stop_requested = True
                break
            candidate = validate_integer_candidate(
                row,
                lower_bounds=self._lower_bounds,
                upper_bounds=self._upper_bounds,
            )
            self._controller.require_evaluation()
            y_p = float(self._problem(candidate))
            self._controller.observe_evaluation(
                y_p,
            )
            objectives[row_index] = y_p
            if self._controller.should_stop():
                self.stop_requested = True
                if row_index + 1 < points.shape[0]:
                    objectives[row_index + 1 :] = BLACKBOXOPT_POST_STOP_SENTINEL
                break
        return objectives


def count_ioh_constraints(problem: Any) -> int:
    """
    Return the number of active IOH inequality constraints for ``problem``.
    """
    ctx = get_attached_eval_context(problem)
    if ctx is not None:
        names = getattr(ctx, "_c_names", None)
        if isinstance(names, list):
            return len(names)

    constraints = getattr(problem, "constraints", None)
    if constraints is not None:
        n_method = getattr(constraints, "n", None)
        if callable(n_method):
            try:
                return int(n_method())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "problem.constraints.n() did not return an integer "
                    f"constraint count: {exc}"
                ) from exc
        if isinstance(n_method, int) and not isinstance(n_method, bool):
            return int(n_method)

    raise ValueError(
        "cannot determine IOH constraint count; attach an EvalContext with "
        "ensure_constraints(...) or expose problem.constraints.n()"
    )


class GosacEvaluator:
    """
    Shared ``fun`` / ``gfun`` wrappers for GOSAC.

    One IOH ``problem(x)`` fills a cache of ``(y_p, y, g=η−1)`` so GOSAC's
    extra feasible-objective calls do not consume extra budget.
    """

    def __init__(
        self,
        problem: Any,
        controller: Any,
        lower_bounds: Sequence[int],
        upper_bounds: Sequence[int],
        mm: int,
    ) -> None:
        self._problem = problem
        self._controller = controller
        self._lower_bounds = list(lower_bounds)
        self._upper_bounds = list(upper_bounds)
        self._mm = int(mm)
        self._cache: dict[tuple[int, ...], tuple[float, float, list[float]]] = {}
        self.stop_requested = False

    def _sentinel(self) -> tuple[float, float, list[float]]:
        return (
            BLACKBOXOPT_POST_STOP_SENTINEL,
            BLACKBOXOPT_POST_STOP_SENTINEL,
            [BLACKBOXOPT_POST_STOP_SENTINEL] * self._mm,
        )

    def evaluate_once(self, row: np.ndarray) -> tuple[float, float, list[float]]:
        try:
            candidate = validate_integer_candidate(
                row,
                lower_bounds=self._lower_bounds,
                upper_bounds=self._upper_bounds,
            )
        except CandidateError:
            if not self._controller.can_evaluate():
                self.stop_requested = True
                return self._sentinel()
            raise
        key = tuple(int(v) for v in candidate)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if not self._controller.can_evaluate():
            self.stop_requested = True
            return self._sentinel()
        self._controller.require_evaluation()
        y_p = float(self._problem(candidate))
        y = float(read_last_evaluation_objective_y(self._problem))
        g = read_last_evaluation_constraint_slacks(self._problem)
        if len(g) != self._mm:
            raise ValueError(
                f"expected {self._mm} constraint slacks, got {len(g)}"
            )
        self._controller.observe_evaluation(
            y_p,
        )
        record = (y_p, y, list(g))
        self._cache[key] = record
        if self._controller.should_stop():
            self.stop_requested = True
        return record

    def objective(self, x: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(x, dtype=float))
        values = np.empty(points.shape[0], dtype=float)
        for row_index, row in enumerate(points):
            _y_p, y, _g = self.evaluate_once(row)
            values[row_index] = y
        return values

    def constraints(self, x: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(x, dtype=float))
        values = np.empty((points.shape[0], self._mm), dtype=float)
        for row_index, row in enumerate(points):
            _y_p, _y, g = self.evaluate_once(row)
            values[row_index, :] = np.asarray(g, dtype=float)
        return values


def build_blackboxoptim_bounds(
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> list[list[int]]:
    """
    Convert IOH inclusive bounds to blackboxoptim ``[[lb, ub], ...]`` form.
    """
    return [
        [int(lower), int(upper)]
        for lower, upper in zip(lower_bounds, upper_bounds, strict=True)
    ]


def build_integer_rbf_model(dimension: int) -> Any:
    """
    Return an :class:`blackboxoptim.rbf.RbfModel` with all dimensions integer.
    """
    from blackboxoptim.rbf import MedianLpfFilter, RbfModel

    return RbfModel(
        iindex=tuple(range(int(dimension))),
        filter=MedianLpfFilter(),
    )


def make_budget_stop_callback(
    controller: Any,
    evaluator: Any,
) -> Callable[[Any], None]:
    """
    Force blackboxoptim to exit when the IOH controller stops the run.
    """

    def callback(out: Any) -> None:
        if controller.should_stop() or evaluator.stop_requested:
            if out.sample is not None:
                out.nfev = int(out.sample.shape[0])

    return callback
