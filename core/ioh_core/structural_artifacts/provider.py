"""
Structural artifact evaluation providers (compute vs library).

Library hits are tracked only on :class:`StructuralProviderMetrics` and must
never mutate :class:`~core.ioh_core.cache_eval.EvalContext` candidate
``hits`` / ``misses``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from core.analysis_core.run_config import AnalysisRunConfig
from core.ioh_core.structural_artifacts.artifact import (
    StructuralArtifact,
    fanout_build_failure,
    make_structural_key,
    structural_keys_match,
)
from core.ioh_core.io_util import _req_param
from core.unit_core import kN_to_N


class StructuralProviderError(ValueError):
    """
    Raised when a structural evaluation provider cannot proceed safely.
    """


@dataclass
class StructuralProviderMetrics:
    """
    Process-local structural backend counters (separate from EvalContext).

    Attributes
    ----------
    compute_evaluations : int
        Candidates evaluated via conventional analysis (no library lookup).
    library_hits : int
        Successful strict library lookups that hydrated an artifact.
    library_build_failure_fanouts : int
        Library hits that returned a stored build-failure artifact.
    """

    compute_evaluations: int = 0
    library_hits: int = 0
    library_build_failure_fanouts: int = 0


class StructuralArtifactLookup(Protocol):
    """
    Minimal reader protocol for library-mode providers (filled in L3).
    """

    def get(self, structural_id: int) -> StructuralArtifact:
        """
        Return the artifact for ``structural_id`` or raise.
        """


class StructuralEvaluationProvider(Protocol):
    """
    Evaluate one candidate via compute or strict library lookup.
    """

    @property
    def mode(self) -> str:
        """
        ``\"compute\"`` or ``\"library\"``.
        """

    @property
    def metrics(self) -> StructuralProviderMetrics:
        """
        Structural-backend metrics (not EvalContext hits/misses).
        """

    def evaluate(
        self,
        params: Mapping[str, Any],
        *,
        analysis_fn: Callable[..., dict[str, Any]],
        structural_id: int | None = None,
        finish_names: tuple[str, ...] = (),
        snapshot_hash: str = "",
        expected_key: tuple[Any, ...] | None = None,
        run_config: AnalysisRunConfig | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate ``params`` and return a full analysis result dict.
        """


@dataclass
class ComputeStructuralProvider:
    """
    Default provider: always run conventional analysis (optionally with
    in-process finish-group reuse handled by the caller).
    """

    metrics: StructuralProviderMetrics = field(
        default_factory=StructuralProviderMetrics
    )

    @property
    def mode(self) -> str:
        return "compute"

    def evaluate(
        self,
        params: Mapping[str, Any],
        *,
        analysis_fn: Callable[..., dict[str, Any]],
        structural_id: int | None = None,
        finish_names: tuple[str, ...] = (),
        snapshot_hash: str = "",
        expected_key: tuple[Any, ...] | None = None,
        run_config: AnalysisRunConfig | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:
        # structural_id / expected_key are ignored in compute mode; callers
        # that need finish-group reuse use evaluate_finish_group instead.
        _ = (structural_id, finish_names, snapshot_hash, expected_key)
        self.metrics.compute_evaluations += 1
        return analysis_fn(
            dict(params),
            debug=debug,
            run_config=run_config,
        )


def resolve_library_lookup_snapshot_hash(
    provider: Any,
    requested: str,
) -> str:
    """
    Return the snapshot hash used for library artifact / key validation.

    When ``provider.allow_snapshot_hash_mismatch`` is true and the lookup
    exposes a sealed identity hash, that library hash is authoritative for
    validation. Analysis still uses the caller's current experiment inputs.
    """
    if not bool(getattr(provider, "allow_snapshot_hash_mismatch", False)):
        return str(requested)
    identity = getattr(getattr(provider, "lookup", None), "identity", None)
    library_hash = (
        getattr(identity, "snapshot_hash", None) if identity is not None else None
    )
    if library_hash:
        return str(library_hash)
    return str(requested)


@dataclass
class LibraryStructuralProvider:
    """
    Strict library provider: lookup one artifact, validate, then analyze.

    Storage sealing arrives in L3/L5. This type is ready for callers but
    requires an injected :class:`StructuralArtifactLookup`.
    """

    lookup: StructuralArtifactLookup
    metrics: StructuralProviderMetrics = field(
        default_factory=StructuralProviderMetrics
    )
    structural_names: tuple[str, ...] = ()
    allow_snapshot_hash_mismatch: bool = False

    @property
    def mode(self) -> str:
        return "library"

    def _resolve_structural_names(self) -> tuple[str, ...]:
        if self.structural_names:
            return tuple(str(name) for name in self.structural_names)
        identity = getattr(self.lookup, "identity", None)
        names = getattr(identity, "structural_names", None) if identity is not None else None
        if names:
            return tuple(str(name) for name in names)
        return ()

    def evaluate(
        self,
        params: Mapping[str, Any],
        *,
        analysis_fn: Callable[..., dict[str, Any]],
        structural_id: int | None = None,
        finish_names: tuple[str, ...] = (),
        snapshot_hash: str = "",
        expected_key: tuple[Any, ...] | None = None,
        run_config: AnalysisRunConfig | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:
        if structural_id is None:
            raise StructuralProviderError(
                "library mode requires structural_id for each evaluation"
            )
        artifact = self.lookup.get(int(structural_id))

        cfg = run_config
        if cfg is None:
            from core.analysis_core.run_config import analysis_run_config_from_params

            cfg = analysis_run_config_from_params(dict(params))
        n = kN_to_N(_req_param(params, "loads_N_kN"))
        structural_names = self._resolve_structural_names()
        if not structural_names:
            raise StructuralProviderError(
                "library mode requires structural_names from the sealed "
                "library identity (or LibraryStructuralProvider.structural_names)"
            )
        lookup_hash = resolve_library_lookup_snapshot_hash(self, snapshot_hash)
        key = expected_key
        if key is None:
            key = make_structural_key(
                params,
                finish_names=finish_names,
                snapshot_hash=lookup_hash,
                n=float(n),
                run_config=cfg,
                structural_names=structural_names,
            )
        if artifact.snapshot_hash != str(lookup_hash) and lookup_hash:
            raise StructuralProviderError(
                "library artifact snapshot_hash mismatch: "
                f"stored={artifact.snapshot_hash!r}, expected={lookup_hash!r}"
            )
        if not structural_keys_match(
            artifact.key,
            key,
            structural_names=structural_names,
        ):
            raise StructuralProviderError(
                "library artifact structural key digest mismatch for "
                f"structural_id={structural_id}"
            )

        self.metrics.library_hits += 1
        if artifact.is_build_failure:
            self.metrics.library_build_failure_fanouts += 1
            return fanout_build_failure(artifact, params)
        return analysis_fn(
            dict(params),
            debug=debug,
            run_config=cfg,
            structural_artifact=artifact,
        )


def build_structural_provider(
    mode: str = "compute",
    *,
    lookup: StructuralArtifactLookup | None = None,
    allow_snapshot_hash_mismatch: bool = False,
) -> StructuralEvaluationProvider:
    """
    Construct a provider for ``mode``.

    Parameters
    ----------
    mode : str, optional
        ``\"compute\"`` (default) or ``\"library\"``.
    lookup : StructuralArtifactLookup or None
        Required when ``mode=\"library\"``.
    allow_snapshot_hash_mismatch : bool, optional
        When true in library mode, validate artifacts against the sealed
        library identity hash rather than the caller's experiment hash.

    Returns
    -------
    StructuralEvaluationProvider
        Compute or library provider instance.

    Raises
    ------
    StructuralProviderError
        If ``mode`` is unknown or library mode lacks a lookup backend.
    """
    token = str(mode).strip().lower()
    if token == "compute":
        return ComputeStructuralProvider()
    if token == "library":
        if lookup is None:
            raise StructuralProviderError(
                "library mode requires a StructuralArtifactLookup backend"
            )
        identity = getattr(lookup, "identity", None)
        names = tuple(str(v) for v in (getattr(identity, "structural_names", ()) or ()))
        return LibraryStructuralProvider(
            lookup=lookup,
            structural_names=names,
            allow_snapshot_hash_mismatch=bool(allow_snapshot_hash_mismatch),
        )
    raise StructuralProviderError(
        f"unknown structural_artifacts mode {mode!r}; expected compute or library"
    )
