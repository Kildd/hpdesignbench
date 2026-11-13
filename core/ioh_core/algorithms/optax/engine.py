"""
Shared discrete-gradient loop for Optax catalogue-index adapters.

``z`` is the continuous index-space parameter vector. ``opt_state`` is the
Optax optimizer state and is never stored in ``z``. JAX and Optax are
imported inside :func:`run_optax_loop`, so this module can be imported
before the optional extra is checked.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from core.ioh_core.algorithms.common import (
    continuous_to_integer_candidate,
    read_problem_bounds,
    seeded_uniform_continuous_start,
)


class RunStopped(Exception):
    """
    Internal signal that the budget or the fixed target ended the run.

    Caught only by :func:`run_optax_loop`. It must not leave ``__call__``.
    """


def _evaluate(problem: Any, controller: Any, x: Sequence[int]) -> float:
    """
    Count one IOH evaluation and return ``problem(x)``.

    Parameters
    ----------
    problem : Any
        IOH-like callable problem.
    controller : RunController
        Hard-budget and fixed-target controller.
    x : sequence of int
        Catalogue-index vector.

    Returns
    -------
    float
        Objective value from ``problem(x)``.

    Raises
    ------
    RunStopped
        If another evaluation is not allowed, or the controller stops
        after this evaluation.
    """
    if controller.should_stop() or not controller.can_evaluate():
        raise RunStopped()
    controller.require_evaluation()
    y_p = float(problem(list(x)))
    controller.observe_evaluation(
        y_p,
    )
    if controller.should_stop():
        raise RunStopped()
    return y_p


def _finite_difference(
    *,
    center: float,
    left: float | None,
    right: float | None,
    kind: str,
) -> float:
    """
    Return one index-space partial, or 0 when a required sample is non-finite.

    Parameters
    ----------
    center : float
        Objective at the current integer point.
    left : float or None
        Objective at ``x - e_i`` when that neighbour was evaluated.
    right : float or None
        Objective at ``x + e_i`` when that neighbour was evaluated.
    kind : str
        ``\"central\"``, ``\"lower\"``, or ``\"upper\"``.

    Returns
    -------
    float
        Partial derivative in index units.
    """
    if kind == "central":
        if (
            left is not None
            and right is not None
            and math.isfinite(left)
            and math.isfinite(right)
        ):
            return (right - left) / 2.0
        return 0.0
    if kind == "lower":
        if right is not None and math.isfinite(center) and math.isfinite(right):
            return right - center
        return 0.0
    if kind == "upper":
        if left is not None and math.isfinite(center) and math.isfinite(left):
            return center - left
        return 0.0
    raise ValueError(f"unknown finite-difference kind {kind!r}")


def compute_index_gradient(
    problem: Any,
    controller: Any,
    x: Sequence[int],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> list[float]:
    """
    Evaluate the L1 finite-difference gradient at integer ``x``.

    The centre is evaluated once. Each axis then probes −1 and then +1
    when that neighbour lies inside the inclusive bounds. A singleton axis
    contributes 0 and performs no neighbour call. A non-finite sample on
    one axis sets only that component to 0.

    Parameters
    ----------
    problem : Any
        IOH-like callable problem.
    controller : RunController
        Hard-budget and fixed-target controller.
    x : sequence of int
        Integer catalogue-index point.
    lower_bounds, upper_bounds : sequence of int
        Inclusive catalogue-index bounds.

    Returns
    -------
    list of float
        Raw finite-difference gradient, one component per axis.

    Raises
    ------
    RunStopped
        If the budget or the fixed target ends during the probes.
    """
    point = [int(value) for value in x]
    dimension = len(lower_bounds)
    if len(point) != dimension or len(upper_bounds) != dimension:
        raise ValueError(
            f"gradient dimension mismatch: x={len(point)} "
            f"bounds=({len(lower_bounds)}, {len(upper_bounds)})"
        )
    center = _evaluate(problem, controller, point)
    gradient: list[float] = []
    for index, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
        current = point[index]
        if int(lower) == int(upper):
            gradient.append(0.0)
            continue
        left: float | None = None
        right: float | None = None
        if current - 1 >= int(lower):
            neighbour = list(point)
            neighbour[index] = current - 1
            left = _evaluate(problem, controller, neighbour)
        if current + 1 <= int(upper):
            neighbour = list(point)
            neighbour[index] = current + 1
            right = _evaluate(problem, controller, neighbour)
        if int(lower) < current < int(upper):
            gradient.append(
                _finite_difference(
                    center=center,
                    left=left,
                    right=right,
                    kind="central",
                )
            )
        elif current == int(lower):
            gradient.append(
                _finite_difference(
                    center=center,
                    left=left,
                    right=right,
                    kind="lower",
                )
            )
        else:
            gradient.append(
                _finite_difference(
                    center=center,
                    left=left,
                    right=right,
                    kind="upper",
                )
            )
    return gradient


def sign_gradient(gradient: Sequence[float]) -> list[float]:
    """
    Map a raw finite-difference gradient to component signs.

    Parameters
    ----------
    gradient : sequence of float
        Raw partial derivatives.

    Returns
    -------
    list of float
        ``-1.0``, ``0.0``, or ``1.0`` per component.
    """
    signed: list[float] = []
    for value in gradient:
        number = float(value)
        if number > 0.0:
            signed.append(1.0)
        elif number < 0.0:
            signed.append(-1.0)
        else:
            signed.append(0.0)
    return signed


def run_optax_loop(algorithm: Any, problem: Any, controller: Any) -> None:
    """
    Run one continuous Optax search on the discrete index gradient.

    Enables JAX float64 for the body of the call and restores the previous
    flag afterwards. A budget or target stop returns normally. A non-finite
    Optax update raises ``RuntimeError`` without committing ``z`` or
    ``opt_state``.

    Parameters
    ----------
    algorithm : OptaxDiscreteGradientAlgorithm
        Adapter supplying the start, the Optax transform, and preprocessing.
    problem : Any
        IOH-like problem.
    controller : RunController
        Hard-budget and fixed-target controller.

    Raises
    ------
    RuntimeError
        If an Optax update produces a non-finite continuous state.
    """
    import jax
    import jax.numpy as jnp
    import optax

    previous = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    try:
        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        start = algorithm.initial_continuous_state(lower_bounds, upper_bounds)
        lower = jnp.asarray(lower_bounds, dtype=jnp.float64)
        upper = jnp.asarray(upper_bounds, dtype=jnp.float64)
        z = jnp.clip(
            jnp.asarray(start, dtype=jnp.float64),
            lower,
            upper,
        )
        optimizer = algorithm.build_optimizer(optax, jnp)
        opt_state = optimizer.init(z)
        while controller.can_evaluate() and not controller.should_stop():
            x_int = continuous_to_integer_candidate(
                [float(value) for value in z.tolist()],
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            try:
                raw_gradient = compute_index_gradient(
                    problem,
                    controller,
                    x_int,
                    lower_bounds,
                    upper_bounds,
                )
            except RunStopped:
                break
            gradient = algorithm.preprocess_gradient(raw_gradient)
            updates, next_opt_state = optimizer.update(
                jnp.asarray(gradient, dtype=jnp.float64),
                opt_state,
                z,
            )
            z_candidate = optax.apply_updates(z, updates)
            if not bool(jnp.all(jnp.isfinite(z_candidate))):
                raise RuntimeError(
                    "Optax update produced a non-finite continuous state"
                )
            z = jnp.clip(z_candidate, lower, upper)
            opt_state = next_opt_state
    finally:
        jax.config.update("jax_enable_x64", previous)


class OptaxDiscreteGradientAlgorithm:
    """
    Base adapter: one finite-difference loop, one Optax transform.

    Subclasses set :attr:`name` and :attr:`info` and implement
    :meth:`build_optimizer`. SGD overrides :meth:`preprocess_gradient`.
    """

    name: str
    info: str

    def __init__(self, max_evaluations: int, seed: int) -> None:
        """
        Parameters
        ----------
        max_evaluations : int
            Hard upper bound on IOH ``problem`` calls.
        seed : int
            Task seed for :func:`seeded_uniform_continuous_start`.
        """
        self.max_evaluations = int(max_evaluations)
        self.evaluations = self.max_evaluations
        self.seed = int(seed)

    def build_optimizer(self, optax_module: Any, jnp_module: Any) -> Any:
        """
        Return the Optax gradient transformation for this token.

        Parameters
        ----------
        optax_module : module
            Imported ``optax`` module.
        jnp_module : module
            Imported ``jax.numpy`` module.

        Returns
        -------
        GradientTransformation
            Optax optimizer.
        """
        raise NotImplementedError

    def preprocess_gradient(self, gradient: Sequence[float]) -> list[float]:
        """
        Return the vector passed to ``optimizer.update``.

        Parameters
        ----------
        gradient : sequence of float
            Raw finite-difference gradient.

        Returns
        -------
        list of float
            Unchanged gradient. SGD overrides this with component signs.
        """
        return [float(value) for value in gradient]

    def initial_continuous_state(
        self,
        lower_bounds: Sequence[int],
        upper_bounds: Sequence[int],
    ) -> list[float]:
        """
        Draw the continuous index-space start from the task seed.

        Parameters
        ----------
        lower_bounds, upper_bounds : sequence of int
            Inclusive catalogue-index bounds.

        Returns
        -------
        list of float
            Continuous start. Not rounded.
        """
        return seeded_uniform_continuous_start(
            lower_bounds,
            upper_bounds,
            self.seed,
        )

    def __call__(self, problem: Any, *, run_controller: Any = None) -> Any:
        """
        Minimise penalised ``y_p`` until the shared controller stops.

        Parameters
        ----------
        problem : Any
            IOH-like problem.
        run_controller : RunController or None, optional
            Worker-supplied controller. When omitted, a fixed-budget
            controller is built from :attr:`max_evaluations`.

        Returns
        -------
        OptaxDiscreteGradientAlgorithm
            ``self``. The reported best point is ``problem.state``.
        """
        from core.ioh_core.algorithms.common import resolve_run_controller

        controller = resolve_run_controller(
            problem,
            max_evaluations=self.max_evaluations,
            run_controller=run_controller,
        )
        run_optax_loop(self, problem, controller)
        return self


class _Rank1OptaxStart:
    """
    Load rank 1 before the Optax loop and return it as the continuous state.

    Subclasses keep the cold optimizer. They do not draw a seeded start.
    """

    name: str
    info: str

    def __init__(self, max_evaluations: int, initial_design_path: str) -> None:
        self.max_evaluations = int(max_evaluations)
        self.evaluations = self.max_evaluations
        self.initial_design_path = str(initial_design_path)
        self._rank1_start: list[float] | None = None

    def initial_continuous_state(
        self,
        lower_bounds: Sequence[int],
        upper_bounds: Sequence[int],
    ) -> list[float]:
        """
        Return the loaded rank-1 vector as floats.

        Parameters
        ----------
        lower_bounds, upper_bounds : sequence of int
            Unused. Bounds were checked when the vector was loaded.

        Returns
        -------
        list of float
            Rank-1 catalogue indices.
        """
        del lower_bounds, upper_bounds
        if self._rank1_start is None:
            raise RuntimeError(f"{self.name} rank-1 start was not loaded")
        return list(self._rank1_start)

    def __call__(self, problem: Any, *, run_controller: Any = None) -> Any:
        """
        Load rank 1, then run the cold Optax loop from that state.

        Parameters
        ----------
        problem : Any
            IOH-like problem.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        _Rank1OptaxStart
            ``self``.
        """
        from core.ioh_core.algorithms.common import read_problem_bounds
        from core.ioh_core.algorithms.warm_start import rank1_start_vector

        _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
        vector = rank1_start_vector(
            problem,
            algorithm_name=self.name,
            initial_design_path=self.initial_design_path,
            max_evaluations=self.max_evaluations,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        self._rank1_start = [float(value) for value in vector]
        return OptaxDiscreteGradientAlgorithm.__call__(
            self,
            problem,
            run_controller=run_controller,
        )


class OptaxAdamWarmStart(_Rank1OptaxStart, OptaxDiscreteGradientAlgorithm):
    """Adam from rank 1 of a ranked InitialDesign."""

    name = "OPTAX_ADAM_WARM_START"
    info = (
        "Adam on y_p from rank 1 of a ranked InitialDesign; discrete "
        "finite-difference gradient in catalogue-index space"
    )

    def build_optimizer(self, optax_module: Any, jnp_module: Any) -> Any:
        from core.ioh_core.algorithms.optax.adam import OptaxAdam

        return OptaxAdam.build_optimizer(self, optax_module, jnp_module)


class OptaxRPropWarmStart(_Rank1OptaxStart, OptaxDiscreteGradientAlgorithm):
    """RProp from rank 1 of a ranked InitialDesign."""

    name = "OPTAX_RPROP_WARM_START"
    info = (
        "RProp on y_p from rank 1 of a ranked InitialDesign; discrete "
        "finite-difference gradient in catalogue-index space"
    )

    def build_optimizer(self, optax_module: Any, jnp_module: Any) -> Any:
        from core.ioh_core.algorithms.optax.rprop import OptaxRProp

        return OptaxRProp.build_optimizer(self, optax_module, jnp_module)


class OptaxSGDNesterovWarmStart(_Rank1OptaxStart, OptaxDiscreteGradientAlgorithm):
    """Sign-normalised Nesterov SGD from rank 1 of a ranked InitialDesign."""

    name = "OPTAX_SGD_NESTEROV_WARM_START"
    info = (
        "Sign-normalised Nesterov SGD on y_p from rank 1 of a ranked "
        "InitialDesign; discrete finite-difference gradient in "
        "catalogue-index space"
    )

    def build_optimizer(self, optax_module: Any, jnp_module: Any) -> Any:
        from core.ioh_core.algorithms.optax.sgd_nesterov import OptaxSGDNesterov

        return OptaxSGDNesterov.build_optimizer(self, optax_module, jnp_module)

    def preprocess_gradient(self, gradient: Sequence[float]) -> list[float]:
        from core.ioh_core.algorithms.optax.sgd_nesterov import OptaxSGDNesterov

        return OptaxSGDNesterov.preprocess_gradient(self, gradient)
