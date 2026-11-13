"""
PyNomad MADS Progressive Barrier for constrained IOH catalogue-index problems.
"""
from __future__ import annotations

from random import Random
from typing import Any, Sequence

from core.ioh_core.algorithms.common import (
    get_attached_eval_context,
    read_last_evaluation_constraint_slacks,
    read_last_evaluation_objective_y,
    read_problem_bounds,
    resolve_run_controller,
    validate_integer_candidate,
)


def _count_ioh_constraints(problem: Any) -> int:
    """
    Return the number of active IOH inequality constraints for ``problem``.

    Prefers attached EvalContext ``_c_names``; falls back to
    ``problem.constraints.n()``.

    Parameters
    ----------
    problem : Any
        IOH-like problem with an attached EvalContext and/or constraint set.

    Returns
    -------
    int
        Non-negative constraint count.

    Raises
    ------
    ValueError
        If the constraint count cannot be determined.
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


def _sample_integer_vector(
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    rng: Random,
) -> list[int]:
    """
    Draw one inclusive integer vector from ``rng``.

    Parameters
    ----------
    lower_bounds : sequence of int
        Inclusive lower bounds.
    upper_bounds : sequence of int
        Inclusive upper bounds.
    rng : Random
        Seeded RNG.

    Returns
    -------
    list of int
        One candidate.
    """
    return [
        rng.randint(int(lo), int(hi))
        for lo, hi in zip(lower_bounds, upper_bounds)
    ]


def _format_bbo(objective: float, slacks: Sequence[float]) -> bytes:
    """
    Encode NOMAD blackbox output ``OBJ`` then ``PB`` values as UTF-8 bytes.

    Parameters
    ----------
    objective : float
        Unpenalized objective value.
    slacks : sequence of float
        Signed ``η − 1`` constraint values.

    Returns
    -------
    bytes
        Space-separated BBO string for :meth:`EvalPoint.setBBO`.
    """
    parts = [f"{float(objective):.16g}"]
    parts.extend(f"{float(value):.16g}" for value in slacks)
    return " ".join(parts).encode("UTF-8")


# NOMAD / PyNomad ``SEED`` and ``setSeed`` take a signed 32-bit C long on
# Windows. Planner seeds are unsigned 32-bit and may exceed 2**31 - 1.
_NOMAD_SIGNED_SEED_MASK = 0x7FFFFFFF


def nomad_signed_seed(seed: int) -> int:
    """
    Map an unsigned 32-bit task seed into NOMAD's signed 32-bit domain.

    Parameters
    ----------
    seed : int
        Task seed (typically in ``[0, 2**32 - 1]``).

    Returns
    -------
    int
        Value in ``[0, 2**31 - 1]`` safe for ``SEED`` / ``PyNomad.setSeed``.
    """
    return int(seed) & _NOMAD_SIGNED_SEED_MASK


def _vns_mads_extra_params(trigger: float) -> tuple[str, ...]:
    """
    Return NOMAD param lines that enable VNS Mads search.

    Parameters
    ----------
    trigger : float
        ``VNS_MADS_SEARCH_TRIGGER`` in ``(0, 1]``.

    Returns
    -------
    tuple of str
        ``VNS_MADS_SEARCH yes`` and the formatted trigger line.
    """
    return (
        "VNS_MADS_SEARCH yes",
        f"VNS_MADS_SEARCH_TRIGGER {float(trigger):g}",
    )


def _run_pynomad_mads(
    problem: Any,
    *,
    max_evaluations: int,
    seed: int,
    algorithm_name: str,
    extra_params: Sequence[str] = (),
    initial_x: Sequence[int] | None = None,
    initial_design_path: str | None = None,
    run_controller=None,
) -> None:
    """
    Run PyNomad MADS Progressive Barrier on one IOH problem.

    Parameters
    ----------
    problem : Any
        IOH problem exposing bounds, callable evaluation, and attached
        EvalContext with at least one active constraint.
    max_evaluations : int
        Hard unique blackbox / IOH evaluation budget (NOMAD ``MAX_BB_EVAL``
        and ``RunController``). ``MAX_EVAL`` is set to the same integer:
        for granular (integer) variables NOMAD otherwise injects
        ``MAX_EVAL 100000``, and after a target hit failed blackbox
        calls do not count toward ``MAX_BB_EVAL``, so ``optimize`` spins
        for that full cache/fail budget. ``PyNomad.cb`` ignores Python
        exceptions, so raising cannot abort the C++ loop.
    seed : int
        Full unsigned task seed for Python ``Random``; masked for NOMAD.
    algorithm_name : str
        Registry token used in the zero-constraint error message.
    extra_params : sequence of str, optional
        Extra NOMAD param lines appended after the shared MADS-PB list.
        ``PYNOMAD_MADS_PB`` passes none.
    initial_x : sequence of int or None, optional
        Catalogue-index start. When ``None`` and ``initial_design_path``
        is also ``None``, a seeded random integer vector is used.
    initial_design_path : str or None, optional
        Ranked InitialDesign JSON. Rank 1 is loaded before any evaluation.
    run_controller : optional
        Optional target-aware evaluation controller.

    Returns
    -------
    None
        Best values live on the IOH problem state.

    Raises
    ------
    ValueError
        If the problem has no active constraints.
    """
    import PyNomad

    mm = _count_ioh_constraints(problem)
    if mm < 1:
        raise ValueError(
            f"{algorithm_name} requires at least one active IOH "
            f"constraint; got mm={mm}"
        )

    dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
    # Python ``Random`` accepts the full unsigned seed; only the NOMAD C
    # API needs the signed 32-bit mapping.
    rng = Random(seed)
    resolved_x = initial_x
    if initial_design_path is not None:
        from core.ioh_core.algorithms.warm_start import rank1_start_vector

        resolved_x = rank1_start_vector(
            problem,
            algorithm_name=algorithm_name,
            initial_design_path=initial_design_path,
            max_evaluations=max_evaluations,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
    controller = resolve_run_controller(
        problem,
        max_evaluations=max_evaluations,
        run_controller=run_controller,
    )
    if resolved_x is None:
        x0 = [float(v) for v in _sample_integer_vector(
            lower_bounds, upper_bounds, rng
        )]
    else:
        from core.ioh_core.algorithms.warm_start import validate_warm_start_candidate

        x0 = [float(v) for v in validate_warm_start_candidate(
            problem,
            resolved_x,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            algorithm_name=algorithm_name,
        )]
    lb = [float(v) for v in lower_bounds]
    ub = [float(v) for v in upper_bounds]
    pb_types = " ".join(["PB"] * mm)
    budget = int(max_evaluations)
    nomad_seed = nomad_signed_seed(seed)
    params = [
        f"DIMENSION {int(dimension)}",
        "BB_INPUT_TYPE * I",
        f"BB_OUTPUT_TYPE OBJ {pb_types}",
        f"MAX_BB_EVAL {budget}",
        f"MAX_EVAL {budget}",
        f"SEED {nomad_seed}",
        "DISPLAY_DEGREE 0",
        "DISPLAY_ALL_EVAL false",
        "NB_THREADS_PARALLEL_EVAL 1",
        *list(extra_params),
    ]

    def bb(eval_point: Any) -> int:
        if not controller.can_evaluate():
            return 0
        try:
            coords = [
                eval_point.get_coord(i)
                for i in range(int(eval_point.size()))
            ]
            candidate = validate_integer_candidate(
                coords,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
        except Exception:
            return 0
        controller.require_evaluation()
        y_p = float(problem(candidate))
        y = read_last_evaluation_objective_y(problem)
        g = read_last_evaluation_constraint_slacks(problem)
        if len(g) != mm:
            raise ValueError(
                f"expected {mm} constraint slacks, got {len(g)}"
            )
        controller.observe_evaluation(y_p)
        try:
            eval_point.setBBO(_format_bbo(y, g))
        except Exception:
            return 0
        return 1

    # SEED in params alone does not reseed the process-global NOMAD RNG
    # across successive PyNomad.optimize calls; setSeed does.
    PyNomad.resetRandomNumberGenerator()
    PyNomad.setSeed(nomad_seed)
    PyNomad.optimize(bb, x0, lb, ub, params)


class PyNomadMadsPb:
    """
    PyNomad MADS with Progressive Barrier on integer catalogue indices.

    Authoritative evaluation counts and best solutions come from the IOH
    problem state. The ``PyNomad.optimize`` result dict is ignored for
    reporting.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Deterministic seed for the initial integer vector and NOMAD ``SEED``.
        Values above ``2**31 - 1`` are masked for the C API via
        :func:`nomad_signed_seed`; Python ``Random`` still uses the full seed.
    """

    name = "PYNOMAD_MADS_PB"
    info = (
        "PyNomad MADS Progressive Barrier; minimizes unpenalized y "
        "subject to signed g=η−1 constraints"
    )

    def __init__(
        self,
        max_evaluations: int,
        seed: int,
        *,
        initial_x: Sequence[int] | None = None,
        initial_design_path: str | None = None,
        name: str | None = None,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)
        self._initial_x = (
            None if initial_x is None else tuple(int(v) for v in initial_x)
        )
        self._initial_design_path = (
            None if initial_design_path is None else str(initial_design_path)
        )
        if name is not None:
            self.name = name
        if self._initial_x is not None and self._initial_design_path is not None:
            raise ValueError(
                f"{self.name} accepts either initial_x or "
                "initial_design_path, not both"
            )
        if self._initial_x is not None or self._initial_design_path is not None:
            self.info = (
                "PyNomad MADS Progressive Barrier from rank 1 of a ranked "
                "InitialDesign; minimizes unpenalized y subject to signed "
                "g=η−1 constraints"
            )

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run PyNomad MADS-PB on a constrained IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds, callable evaluation, and attached
            EvalContext with at least one active constraint.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        PyNomadMadsPb
            ``self`` after the run; best values live on the IOH problem state.

        Raises
        ------
        ValueError
            If the problem has no active constraints.
        """
        _run_pynomad_mads(
            problem,
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            algorithm_name=self.name,
            initial_x=self._initial_x,
            initial_design_path=self._initial_design_path,
            run_controller=run_controller,
        )
        return self


class PyNomadVnsMadsPb:
    """
    PyNomad MADS Progressive Barrier with VNS search.

    Same integer Progressive Barrier contract as :class:`PyNomadMadsPb`, plus
    NOMAD ``VNS_MADS_SEARCH yes`` and optional ``VNS_MADS_SEARCH_TRIGGER``
    (registry default 0.75). The ``PyNomad.optimize`` result dict is ignored
    for reporting; authoritative results live on the IOH problem state.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    vns_mads_search_trigger : float
        Maximum desired fraction of blackbox evaluations inside VNS searches
        (``(0, 1]``).
    seed : int
        Deterministic seed for the initial integer vector and NOMAD ``SEED``.
        Values above ``2**31 - 1`` are masked for the C API via
        :func:`nomad_signed_seed`; Python ``Random`` still uses the full seed.
    """

    name = "PYNOMAD_VNS_MADS_PB"
    info = (
        "PyNomad MADS Progressive Barrier with VNS search; minimizes "
        "unpenalized y subject to signed g=η−1 constraints"
    )

    def __init__(
        self,
        max_evaluations: int,
        vns_mads_search_trigger: float,
        seed: int,
        *,
        initial_x: Sequence[int] | None = None,
        initial_design_path: str | None = None,
        name: str | None = None,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.vns_mads_search_trigger = float(vns_mads_search_trigger)
        self.seed = int(seed)
        self._initial_x = (
            None if initial_x is None else tuple(int(v) for v in initial_x)
        )
        self._initial_design_path = (
            None if initial_design_path is None else str(initial_design_path)
        )
        if name is not None:
            self.name = name
        if self._initial_x is not None and self._initial_design_path is not None:
            raise ValueError(
                f"{self.name} accepts either initial_x or "
                "initial_design_path, not both"
            )
        if self._initial_x is not None or self._initial_design_path is not None:
            self.info = (
                "PyNomad MADS Progressive Barrier with VNS search from "
                "rank 1 of a ranked InitialDesign; minimizes unpenalized y "
                "subject to signed g=η−1 constraints"
            )

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run PyNomad MADS-PB with VNS search on a constrained IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds, callable evaluation, and attached
            EvalContext with at least one active constraint.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        PyNomadVnsMadsPb
            ``self`` after the run; best values live on the IOH problem state.

        Raises
        ------
        ValueError
            If the problem has no active constraints.
        """
        _run_pynomad_mads(
            problem,
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            algorithm_name=self.name,
            extra_params=_vns_mads_extra_params(self.vns_mads_search_trigger),
            initial_x=self._initial_x,
            initial_design_path=self._initial_design_path,
            run_controller=run_controller,
        )
        return self
