"""
PFNs4BO PFN-PEN and PFN-CEI wrappers for IOH catalogue-index problems.
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
from core.ioh_core.algorithms.pfns4bo.space import (
    default_n_doe,
    encode_integers_to_unit_cube,
    sample_integer_candidates,
)

_DEVICE = "cpu:0"
_N_PENDING = 1000


def _count_ioh_constraints(problem: Any) -> int:
    """
    Return the number of active IOH inequality constraints for ``problem``.

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


def _patch_pfns4bo_torch_compat() -> None:
    """
    Restore typing aliases removed from ``torch.nn.modules.transformer``.

    ``pfns4bo==0.1.5`` imports ``Optional`` / ``Module`` / ``Tensor`` from
    that module; recent PyTorch builds no longer re-export them.
    """
    import typing

    import torch
    import torch.nn as nn
    import torch.nn.modules.transformer as transformer_mod

    aliases = {
        "Optional": typing.Optional,
        "Module": nn.Module,
        "Tensor": torch.Tensor,
        "MultiheadAttention": nn.MultiheadAttention,
        "Linear": nn.Linear,
        "Dropout": nn.Dropout,
        "LayerNorm": nn.LayerNorm,
    }
    for name, value in aliases.items():
        if not hasattr(transformer_mod, name):
            setattr(transformer_mod, name, value)


def _ensure_hebo_plus_model_path() -> str:
    """
    Resolve HEBO+ weights, downloading and gunzipping with Python if needed.

    Upstream ``pfns4bo.prepare_models`` shells out to ``gzip``, which is often
    missing on Windows. This helper uses the stdlib ``gzip`` module instead.

    Returns
    -------
    str
        Absolute path to ``model_hebo_morebudget_9_unused_features_3.pt``.
    """
    import gzip
    import os
    import shutil

    import pfns4bo
    import requests

    name = "model_hebo_morebudget_9_unused_features_3.pt"
    models_dir = os.path.join(os.path.dirname(pfns4bo.__file__), "final_models")
    os.makedirs(models_dir, exist_ok=True)
    weights_path = os.path.join(models_dir, name)
    compressed_path = weights_path + ".gz"
    if not os.path.exists(weights_path):
        if not os.path.exists(compressed_path):
            url = (
                "https://github.com/automl/PFNs4BO/raw/main/pfns4bo/"
                f"final_models/{name}.gz"
            )
            response = requests.get(url, allow_redirects=True, timeout=600)
            response.raise_for_status()
            with open(compressed_path, "wb") as handle:
                handle.write(response.content)
        with gzip.open(compressed_path, "rb") as source:
            with open(weights_path, "wb") as target:
                shutil.copyfileobj(source, target)
    return weights_path


def _load_hebo_plus_model(torch_mod: Any) -> Any:
    """
    Load the HEBO+ PFN weights, downloading/unzipping on first use.

    Parameters
    ----------
    torch_mod : module
        Imported ``torch`` module.

    Returns
    -------
    object
        Deserialized transformer model on CPU.
    """
    model_path = _ensure_hebo_plus_model_path()
    try:
        return torch_mod.load(
            model_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        return torch_mod.load(model_path, map_location="cpu")


def _evaluate_integer(
    problem: Any,
    controller: Any,
    candidate: Sequence[int],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    *,
    read_unpenalized_y: bool,
) -> tuple[float, float]:
    """
    Gate one IOH evaluation and return ``(y_p, y_for_search)``.

    Parameters
    ----------
    problem : Any
        IOH-like problem.
    controller : Any
        Run controller.
    candidate : sequence of int
        Integer catalogue vector.
    lower_bounds : sequence of int
        Inclusive lower bounds.
    upper_bounds : sequence of int
        Inclusive upper bounds.
    read_unpenalized_y : bool
        If True (CEI), read unpenalized ``y`` from the last evaluation
        record. If False (PEN), use ``y_p`` only and never touch constraint
        or ``y`` helpers.

    Returns
    -------
    tuple of float
        ``(y_p, y_for_search)`` where ``y_for_search`` is unpenalized ``y``
        when ``read_unpenalized_y`` else ``y_p``.
    """
    validated = validate_integer_candidate(
        candidate,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    controller.require_evaluation()
    y_p = float(problem(validated))
    if read_unpenalized_y:
        y_for_search = read_last_evaluation_objective_y(problem)
    else:
        y_for_search = y_p
    controller.observe_evaluation(
        y_p,
    )
    return y_p, y_for_search


def _run_pfn_loop(
    *,
    problem: Any,
    max_evaluations: int,
    seed: int,
    run_controller: Any,
    constrained: bool,
    initial_doe: tuple[tuple[int, ...], ...] | None = None,
) -> None:
    """
    Shared integer DOE + ask–suggest loop for PEN and CEI.

    Parameters
    ----------
    problem : Any
        IOH-like problem.
    max_evaluations : int
        Hard IOH budget.
    seed : int
        Deterministic seed.
    run_controller : optional
        Optional target-aware controller.
    constrained : bool
        If True, run PFN-CEI; else constraint-blind PFN-PEN.
    """
    import torch

    _patch_pfns4bo_torch_compat()
    from pfns4bo.scripts.acquisition_functions import TransformerBOMethod

    controller = resolve_run_controller(
        problem,
        max_evaluations=max_evaluations,
        run_controller=run_controller,
    )
    _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)

    mm = 0
    if constrained:
        mm = _count_ioh_constraints(problem)
        if mm < 1:
            raise ValueError(
                "PFN_CEI requires at least one active IOH "
                f"constraint; got mm={mm}"
            )

    torch.manual_seed(int(seed))
    rng = Random(int(seed))
    model = _load_hebo_plus_model(torch)

    if constrained:
        from core.ioh_core.algorithms.pfns4bo._vendor.pfn_cei import (
            PFNCEI_TransformerBOMethod,
        )

        suggester = PFNCEI_TransformerBOMethod(model, device=_DEVICE)
    else:
        suggester = TransformerBOMethod(model, device=_DEVICE)

    x_int: list[list[int]] = []
    y_obs_values: list[float] = []
    gx_rows: list[list[float]] = []

    if initial_doe is None:
        doe = sample_integer_candidates(
            lower_bounds,
            upper_bounds,
            n_samples=default_n_doe(max_evaluations),
            rng=rng,
        )
    else:
        doe = [tuple(int(value) for value in vector) for vector in initial_doe]
    for candidate in doe:
        if not controller.can_evaluate() or controller.should_stop():
            return
        y_p, y_for_search = _evaluate_integer(
            problem,
            controller,
            candidate,
            lower_bounds,
            upper_bounds,
            read_unpenalized_y=constrained,
        )
        x_int.append(list(candidate))
        y_obs_values.append(-float(y_for_search))
        if constrained:
            gx_rows.append(read_last_evaluation_constraint_slacks(problem))

    while not controller.should_stop():
        if not controller.can_evaluate():
            break
        if not x_int:
            break

        pending_int = sample_integer_candidates(
            lower_bounds,
            upper_bounds,
            n_samples=_N_PENDING,
            rng=rng,
        )
        x_obs = torch.tensor(
            encode_integers_to_unit_cube(x_int, lower_bounds, upper_bounds),
            dtype=torch.float32,
        )
        y_obs = torch.tensor(y_obs_values, dtype=torch.float32)
        x_pen = torch.tensor(
            encode_integers_to_unit_cube(
                pending_int, lower_bounds, upper_bounds
            ),
            dtype=torch.float32,
        )

        if constrained:
            gx = torch.tensor(gx_rows, dtype=torch.float32)
            ei, p_feas = suggester.observe_and_suggest(
                X_obs=x_obs,
                y_obs=y_obs,
                X_pen=x_pen,
                GX=gx,
                return_actual_ei=True,
            )
            cei = ei
            for column in range(int(p_feas.shape[1])):
                cei = cei * p_feas[:, column]
            rec_idx = int(torch.argmax(cei).item())
        else:
            rec_idx = int(
                suggester.observe_and_suggest(
                    X_obs=x_obs,
                    y_obs=y_obs,
                    X_pen=x_pen,
                    return_actual_ei=False,
                )
            )

        next_candidate = pending_int[rec_idx]
        if not controller.can_evaluate() or controller.should_stop():
            break
        y_p, y_for_search = _evaluate_integer(
            problem,
            controller,
            next_candidate,
            lower_bounds,
            upper_bounds,
            read_unpenalized_y=constrained,
        )
        x_int.append(list(next_candidate))
        y_obs_values.append(-float(y_for_search))
        if constrained:
            gx_rows.append(read_last_evaluation_constraint_slacks(problem))


class PfnPen:
    """
    Constraint-blind PFNs4BO EI on integer catalogue indices.

    Searches on negated penalized objectives ``-y_p`` (upstream maximizes).
    Does not read constraint channels.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Deterministic seed for DOE, pending samples, and torch.
    """

    name = "PFN_PEN"
    info = (
        "PFNs4BO HEBO+ EI on integer catalogue indices; "
        "maximizes -y_p (constraint-blind)"
    )

    def __init__(self, max_evaluations: int, seed: int) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run PFN-PEN on an IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem exposing bounds and callable evaluation.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        PfnPen
            ``self`` after the run; best values live on the IOH problem state.
        """
        _run_pfn_loop(
            problem=problem,
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            run_controller=run_controller,
            constrained=False,
        )
        return self


class PfnCei:
    """
    Constrained PFNs4BO CEI on integer catalogue indices.

    Searches on negated unpenalized ``-y`` with signed ``g=η−1`` constraints.
    Observes/stops on ``y_p``.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Deterministic seed for DOE, pending samples, and torch.
    """

    name = "PFN_CEI"
    info = (
        "PFNs4BO HEBO+ CEI on integer catalogue indices; "
        "maximizes -y subject to signed g=η−1"
    )

    def __init__(self, max_evaluations: int, seed: int) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """
        Run PFN-CEI on a constrained IOH-like problem.

        Parameters
        ----------
        problem : Any
            IOH problem with at least one active constraint.
        run_controller : optional
            Optional target-aware evaluation controller.

        Returns
        -------
        PfnCei
            ``self`` after the run; best values live on the IOH problem state.

        Raises
        ------
        ValueError
            If the problem has no active constraints.
        """
        _run_pfn_loop(
            problem=problem,
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            run_controller=run_controller,
            constrained=True,
        )
        return self


def _load_pfn_prefix(problem: Any, *, name: str, path: str, max_evaluations: int):
    """Load the DOE prefix of length ``default_n_doe`` without evaluating."""
    from core.ioh_core.algorithms.warm_start import load_warm_start_prefix

    _dimension, lower_bounds, upper_bounds = read_problem_bounds(problem)
    return load_warm_start_prefix(
        problem,
        algorithm_name=name,
        initial_design_path=path,
        count=default_n_doe(max_evaluations),
        max_evaluations=max_evaluations,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


class PfnPenWarmStart:
    """
    PFN-PEN whose DOE is a ranked InitialDesign prefix.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Deterministic seed.
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    name = "PFN_PEN_WARM_START"
    info = "PFN-PEN from a ranked InitialDesign DOE"

    def __init__(
        self,
        max_evaluations: int,
        seed: int,
        *,
        initial_design_path: str,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)
        self.initial_design_path = initial_design_path

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """Run warm-start PFN-PEN."""
        prefix = _load_pfn_prefix(
            problem,
            name=self.name,
            path=self.initial_design_path,
            max_evaluations=self.max_evaluations,
        )
        _run_pfn_loop(
            problem=problem,
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            run_controller=run_controller,
            constrained=False,
            initial_doe=prefix,
        )
        return self


class PfnCeiWarmStart:
    """
    PFN-CEI whose DOE is a ranked InitialDesign prefix.

    Parameters
    ----------
    max_evaluations : int
        Hard IOH evaluation budget.
    seed : int
        Deterministic seed.
    initial_design_path : str
        Path of ``initial_design.json``.
    """

    name = "PFN_CEI_WARM_START"
    info = "PFN-CEI from a ranked InitialDesign DOE"

    def __init__(
        self,
        max_evaluations: int,
        seed: int,
        *,
        initial_design_path: str,
    ) -> None:
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)
        self.initial_design_path = initial_design_path

    def __call__(self, problem: Any, *, run_controller=None) -> object:
        """Run warm-start PFN-CEI."""
        prefix = _load_pfn_prefix(
            problem,
            name=self.name,
            path=self.initial_design_path,
            max_evaluations=self.max_evaluations,
        )
        _run_pfn_loop(
            problem=problem,
            max_evaluations=self.max_evaluations,
            seed=self.seed,
            run_controller=run_controller,
            constrained=True,
            initial_doe=prefix,
        )
        return self
        return self
