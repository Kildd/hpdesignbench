"""
Cache-based evaluation context for IOHexperimenter optimization problems.

Avoids redundant re-evaluation of the (expensive) structural analysis
function by memorizing the last call in a single-entry cache. Also
publishes decoded parameter values and constraint **utilizations** as
watchable attributes for the IOH Analyzer logger.

IOH constraint callables ``g(x)`` read raw **violations**
(``max(0, η−1)``) from ``violations_`` in the cached result (see
``problem_builder``), not from the ``c__*`` logger attributes.

Author: Max Dombrowski

Modifications by Elliot Melcer:

* changed search strings to match analysis.py in hp_slab
* docstrings

"""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Sequence


class EvalContext:
    """
    Single-entry memo cache that runs the structural analysis at most once
    per unique candidate vector and distributes results to the IOH objective
    and constraint callbacks.

    The cache key is the integer index tuple of the candidate. On a cache
    hit the stored result is returned immediately; on a miss the decode and
    analysis functions are called and the result is stored.

    Unexpected exceptions during decode/analysis are converted into one
    structured failure record for that candidate so objective and constraint
    callbacks do not re-execute the failing evaluation.

    ``lg.watch(ctx, f"c__{{name}}")`` records **constraint utilization** η
    from ``constraint_values`` (1.0 = at the limit). GH penalty terms can be
    reconstructed offline from logged η together with α and β.

    Adapted from: Max Dombrowski

    Attributes
    ----------
    decode : callable
        ``decode(x_idx) -> dict`` — maps an integer index vector to a
        parameter dictionary.
    results : callable
        ``analysis(params) -> dict`` — runs the structural analysis and
        returns a result dictionary with at least ``"y"``, ``"y_p"``,
        ``"constraint_values"``, and ``"violations_"`` keys.
    last_key : tuple[int, ...] or None
        Index tuple of the most recently evaluated candidate, or ``None``
        if no evaluation has been performed yet.
    last_data : dict or None
        Cached result dict from the last analysis call, or ``None``.
    hits : int
        Number of times the cached result was reused [-].
    misses : int
        Number of times a fresh analysis was triggered [-].
    t_eval_s : float
        Wall time in seconds since the last :meth:`reset` (or construction),
        updated at the end of each :meth:`evaluate` call. Logged for IOH so
        consecutive-row differences measure inter-evaluation wall time.
    y : float
        Most recent unpenalized objective value (published for IOH logger).
    y_p : float
        Most recent penalized objective value (published for IOH logger).
    run_config : AnalysisRunConfig
        Frozen per-run analysis settings used for fail-closed records.
    """

    def __init__(self, decode_func, analysis_func, run_config):
        """
        Parameters
        ----------
        decode_func : callable
            ``decode(x_idx) -> dict`` — decodes an integer index vector
            to a named parameter dictionary.
        analysis_func : callable
            ``analysis(params) -> dict`` — performs the structural analysis
            and returns a result dict containing at least ``"y"``,
            ``"y_p"``, ``"constraint_values"``, and ``"violations_"``.
        run_config : AnalysisRunConfig
            Frozen per-run analysis settings used for fail-closed
            unexpected-exception records. Required so the invalid-utilization
            sentinel comes from ``opt_invalid_utilization`` rather than a
            hardcoded module default.
        """
        if run_config is None:
            raise TypeError(
                "EvalContext requires run_config=AnalysisRunConfig(...); "
                "build it with analysis_run_config_from_params from "
                "opt_invalid_utilization."
            )

        # stored callables
        self.decode = decode_func       # decode(x_idx) -> params dict
        self.results = analysis_func    # analysis(params) -> result dict
        self.run_config = run_config

        # tiny single-entry memo cache
        self.last_key = None            # tuple of ints (candidate)
        self.last_data = None           # whatever heavy_analysis returns
        self.hits = 0                   # hits = how often data was reused from cache
        self.misses = 0                 # misses = how often data had to be executed
        self._t0 = time.perf_counter()
        self.t_eval_s = 0.0             # seconds since reset; updated each evaluate

        # containers for active constraints and var names to pass them to logger
        self._c_names: list[str] = []  # constraint names to publish on self as c__<name>
        self._var_names: list[str] = []  # variable names to publish on self as var__<name> and idx__<name>
        # public attributes for logger to watch
        self.y = 0.0  # unpenalised (optional)
        self.y_p = 0.0  # penalised

        # Optional structural library backend (configured by workers; L7).
        self._structural_provider: Any | None = None
        self._structural_id_fn: Callable[[tuple[int, ...]], int] | None = None
        self._finish_names: tuple[str, ...] = ()
        self._snapshot_hash: str = ""

    def configure_structural_backend(
        self,
        *,
        provider: Any,
        structural_id_fn: Callable[[tuple[int, ...]], int],
        finish_names: Sequence[str] = (),
        snapshot_hash: str = "",
    ) -> None:
        """
        Attach a structural evaluation provider for cache-miss analysis.

        Library-mode lookups are tracked only on the provider metrics and must
        not alter :attr:`hits` / :attr:`misses` semantics.

        Parameters
        ----------
        provider : StructuralEvaluationProvider
            Compute or library provider. Only ``mode=\"library\"`` changes the
            miss path; compute providers leave conventional analysis in place.
        structural_id_fn : callable
            ``(candidate_index_tuple) -> structural_id``.
        finish_names : sequence of str, optional
            Finish catalogue variable names for structural-key checks.
        snapshot_hash : str, optional
            Input snapshot fingerprint bound into structural keys.
        """
        self._structural_provider = provider
        self._structural_id_fn = structural_id_fn
        self._finish_names = tuple(str(name) for name in finish_names)
        self._snapshot_hash = str(snapshot_hash)

    def clear_structural_backend(self) -> None:
        """
        Detach any configured structural provider (tests / task teardown).
        """
        self._structural_provider = None
        self._structural_id_fn = None
        self._finish_names = ()
        self._snapshot_hash = ""

    # --- declare stable attributes so IOH can bind watchers once and reuse them ---
    def ensure_constraints(self, names: list[str]):
        """
        Declare stable ``c__<name>`` attributes for all active constraints.

        Must be called before attaching the problem to an IOH logger so
        that the logger can bind watchers to stable attribute names.
        Logged values are utilizations η (see :meth:`get`).

        Parameters
        ----------
        names : list[str]
            Ordered list of active constraint names.
        """
        self._c_names = list(names)
        for n in self._c_names:
            setattr(self, f"c__{n}", 0.0)

    def ensure_params(self, var_names: list[str]):
        """
        Declare stable ``var__<name>`` and ``idx__<name>`` attributes for
        all optimization variable parameters.

        Must be called before attaching the problem to an IOH logger.

        Parameters
        ----------
        var_names : list[str]
            Ordered list of optimization variable names (matching the
            order of the integer index vector).
        """
        self._var_names = list(var_names)
        for n in self._var_names:
            setattr(self, f"var__{n}", 0.0)
            setattr(self, f"idx__{n}", 0)

    def _failure_record(self, params: dict | None, exc: BaseException) -> dict:
        """Build a serializable fail-closed result for an unexpected exception."""
        from slab_construction.slabs.hp_slab.analysis import (
            unexpected_evaluation_failure_result,
        )

        return unexpected_evaluation_failure_result(
            params,
            exc=exc,
            active_constraint_names=list(self._c_names),
            invalid_utilization=self.run_config.invalid_utilization,
            traceback_text="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        )

    def _publish(self, x_idx, rec: dict[str, Any]) -> None:
        """
        Update IOH-watched public attributes from ``rec``.

        Publishes objective channels (``y``, ``y_p``), constraint
        utilization channels (``c__*`` from ``constraint_values``), and
        parameter channels (``var__*``, ``idx__*``).
        """
        try:
            self.y = float(rec.get("y", 0.0))
        except (TypeError, ValueError):
            self.y = float("inf")
        try:
            self.y_p = float(rec.get("y_p", float("inf")))
        except (TypeError, ValueError):
            self.y_p = float("inf")

        # publish constraint utilization channels for IOH logs (η, not violation)
        utils = rec.get("constraint_values") or {}
        for n in self._c_names:
            try:
                setattr(self, f"c__{n}", float(utils.get(n, 0.0)))
            except (TypeError, ValueError):
                setattr(self, f"c__{n}", 99.0)

        # publish parameter channels
        # prefer the analysis-returned decoded map to avoid re-decoding drift
        pmap = rec.get("params")
        if pmap is None:
            # fallback to a fresh decode if analysis didn't return "params"
            try:
                pmap = self.decode(x_idx)
            except Exception:
                pmap = {}

        for i, n in enumerate(self._var_names):
            # decoded value (attempt float cast; if not numeric, leave previous)
            v = pmap.get(n) if isinstance(pmap, dict) else None
            try:
                setattr(self, f"var__{n}", float(v))
            except (TypeError, ValueError):
                # keep previous numeric value to avoid breaking IOH numeric property
                pass
            # catalogue index from x_idx ordering
            try:
                setattr(self, f"idx__{n}", int(x_idx[i]))
            except Exception:
                # if dimension mismatch, keep previous
                pass

    def _analyze_on_miss(self, key: tuple[int, ...], params: dict) -> dict[str, Any]:
        """
        Run conventional or library-backed analysis for one cache miss.
        """
        provider = self._structural_provider
        if (
            provider is not None
            and getattr(provider, "mode", "compute") == "library"
        ):
            if self._structural_id_fn is None:
                raise RuntimeError(
                    "library structural backend requires structural_id_fn"
                )
            structural_id = int(self._structural_id_fn(key))
            return provider.evaluate(
                params,
                analysis_fn=self.results,
                structural_id=structural_id,
                finish_names=self._finish_names,
                snapshot_hash=self._snapshot_hash,
                run_config=self.run_config,
            )
        return self.results(params)

    def get(self, x_idx):
        """
        Evaluate (or recall) the analysis for candidate ``x_idx``.

        On a cache miss, decodes ``x_idx`` and calls the analysis function
        (or a configured structural library provider). Unexpected exceptions
        are converted into one structured failure record and cached for that
        candidate so objective and constraint callbacks do not re-execute the
        failing evaluation. Always updates the public logger attributes
        (``y``, ``y_p``, ``c__*``, ``var__*``, ``idx__*``, ``t_eval_s``) from
        the result before returning.

        Parameters
        ----------
        x_idx : sequence of int
            Integer catalogue-index vector identifying the candidate.

        Returns
        -------
        dict
            Analysis result dictionary (cached or freshly computed), or a
            structured fail-closed failure record after an unexpected
            exception.
        """
        key = tuple(int(k) for k in x_idx)
        if key == self.last_key:
            self.hits += 1
            rec = self.last_data
        else:
            self.misses += 1
            params = None
            try:
                params = self.decode(x_idx)
                rec = self._analyze_on_miss(key, params)
            except Exception as exc:
                rec = self._failure_record(params, exc)
            self.last_key = key
            self.last_data = rec

        self._publish(x_idx, rec)
        self.t_eval_s = time.perf_counter() - self._t0
        return rec

    def reset(self):
        """
        Clear the cache, reset hit/miss counters, and restart the eval clock.

        Call before each optimization run to ensure a clean state. Structural
        backend configuration is preserved across resets.
        """
        self.last_key = None
        self.last_data = None
        self.hits = 0
        self.misses = 0
        self._t0 = time.perf_counter()
        self.t_eval_s = 0.0
