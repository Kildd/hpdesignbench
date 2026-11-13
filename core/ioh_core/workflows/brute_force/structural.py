"""
Immutable structural-artifact reuse for finish-group brute-force evaluation.

Floor-finish catalogue variables (last two IndexSpace dimensions) change
within a 24-candidate group while structural section results stay fixed.
Workers build one :class:`StructuralArtifact` per group and re-evaluate
finish-dependent dead load, demand, deflection integration, acoustics,
cost/GWP, and penalties for each finish combination.

Shared capture/apply/key primitives live in
:mod:`core.ioh_core.structural_artifacts.artifact`. This module keeps
compatibility re-exports and the brute-force-only
:func:`evaluate_finish_group` helper (compute reuse or strict library lookup).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from core.analysis_core.run_config import AnalysisRunConfig
from core.ioh_core.io_util import _req_param
from core.ioh_core.structural_artifacts.artifact import (
    StructuralArtifact,
    StructuralReuseError,
    _bump_structural_build_count,
    apply_structural_artifact,
    capture_structural_artifact,
    fanout_build_failure,
    finish_variable_names,
    get_structural_build_count,
    make_structural_key,
    reset_structural_build_count,
    scientific_values_equal,
)
from core.ioh_core.structural_artifacts.provider import (
    StructuralEvaluationProvider,
    StructuralProviderError,
    resolve_library_lookup_snapshot_hash,
)
from core.unit_core import kN_to_N

# Compatibility alias used by older call sites / tests.
_fanout_build_failure = fanout_build_failure

__all__ = [
    "StructuralArtifact",
    "StructuralReuseError",
    "_bump_structural_build_count",
    "_fanout_build_failure",
    "apply_structural_artifact",
    "capture_structural_artifact",
    "evaluate_finish_group",
    "fanout_build_failure",
    "finish_variable_names",
    "get_structural_build_count",
    "make_structural_key",
    "reset_structural_build_count",
    "scientific_values_equal",
]


def _finish_group_structural_keys(
    param_dicts: Sequence[Mapping[str, Any]],
    *,
    finish_names: Sequence[str],
    snapshot_hash: str,
    run_config: AnalysisRunConfig | None,
    structural_names: Sequence[str] | None = None,
) -> list[tuple[Any, ...]]:
    keys: list[tuple[Any, ...]] = []
    for params in param_dicts:
        n = kN_to_N(_req_param(params, "loads_N_kN"))
        cfg = run_config
        if cfg is None:
            from core.analysis_core.run_config import analysis_run_config_from_params

            cfg = analysis_run_config_from_params(dict(params))
        keys.append(
            make_structural_key(
                params,
                finish_names=finish_names,
                snapshot_hash=snapshot_hash,
                n=float(n),
                run_config=cfg,
                structural_names=structural_names,
            )
        )
    return keys


def _evaluate_finish_group_library(
    param_dicts: Sequence[Mapping[str, Any]],
    *,
    analysis_fn: Callable[..., dict[str, Any]],
    finish_names: Sequence[str],
    snapshot_hash: str,
    run_config: AnalysisRunConfig | None,
    provider: StructuralEvaluationProvider,
    structural_id: int,
    debug: bool,
) -> list[dict[str, Any]]:
    lookup = getattr(provider, "lookup", None)
    if lookup is None:
        raise StructuralProviderError(
            "library finish-group evaluation requires a provider with lookup"
        )
    identity = getattr(lookup, "identity", None)
    structural_names = tuple(
        str(v) for v in (getattr(identity, "structural_names", ()) or ())
    )
    if not structural_names:
        structural_names = tuple(
            str(v) for v in (getattr(provider, "structural_names", ()) or ())
        )
    if not structural_names:
        raise StructuralProviderError(
            "library finish-group evaluation requires structural_names "
            "from the sealed library identity"
        )
    lookup_hash = resolve_library_lookup_snapshot_hash(provider, snapshot_hash)
    keys = _finish_group_structural_keys(
        param_dicts,
        finish_names=finish_names,
        snapshot_hash=lookup_hash,
        run_config=run_config,
        structural_names=structural_names,
    )
    if len(set(keys)) != 1:
        raise StructuralReuseError(
            "finish group contains more than one structural key; "
            "shard boundaries must not split a finish group"
        )
    artifact = lookup.get(int(structural_id))
    if lookup_hash and artifact.snapshot_hash != str(lookup_hash):
        raise StructuralProviderError(
            "library artifact snapshot_hash mismatch: "
            f"stored={artifact.snapshot_hash!r}, expected={lookup_hash!r}"
        )
    from core.ioh_core.structural_artifacts.artifact import structural_keys_match

    if not structural_keys_match(
        artifact.key,
        keys[0],
        structural_names=structural_names,
    ):
        raise StructuralProviderError(
            "library artifact structural key digest mismatch for "
            f"structural_id={structural_id}"
        )

    provider.metrics.library_hits += 1
    if artifact.is_build_failure:
        provider.metrics.library_build_failure_fanouts += 1
        return [fanout_build_failure(artifact, params) for params in param_dicts]

    results: list[dict[str, Any]] = []
    for params in param_dicts:
        results.append(
            analysis_fn(
                dict(params),
                debug=debug,
                run_config=run_config,
                structural_artifact=artifact,
            )
        )
    return results


def evaluate_finish_group(
    param_dicts: Sequence[Mapping[str, Any]],
    *,
    constraints: Mapping[str, Any],
    materials: Mapping[str, Any],
    analysis_fn: Callable[..., dict[str, Any]],
    finish_names: Sequence[str],
    snapshot_hash: str,
    run_config: AnalysisRunConfig | None = None,
    reuse: bool = True,
    debug: bool = False,
    provider: StructuralEvaluationProvider | None = None,
    structural_id: int | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate one complete finish group with optional structural reuse.

    Parameters
    ----------
    param_dicts : sequence of mapping
        Decoded parameter dictionaries for every finish in the group
        (typically 24). Structural parameters must share one key.
    constraints : mapping
        Constraint metadata.
    materials : mapping
        Materials registry.
    analysis_fn : callable
        Bound slab ``analysis`` accepting ``(params, **kwargs)`` with
        ``constraints`` / ``materials`` already applied (as from
        :func:`~core.ioh_core.problem_builder.build_problem_from_snapshot`).
        Must forward ``structural_artifact`` /
        ``structural_artifact_holder`` kwargs.
    finish_names : sequence of str
        Finish catalogue variable names.
    snapshot_hash : str
        Input snapshot fingerprint.
    run_config : AnalysisRunConfig or None, optional
        Frozen analysis settings.
    reuse : bool, optional
        When ``False``, run independent full analyses (reference mode).
        Forbidden when ``provider.mode == \"library\"``.
    debug : bool, optional
        Forwarded to analysis.
    provider : StructuralEvaluationProvider or None, optional
        When ``mode=\"library\"``, look up one artifact per group and fan it
        across all finishes. Compute mode (default / omitted) keeps in-process
        capture reuse.
    structural_id : int or None, optional
        Required when ``provider.mode == \"library\"``.

    Returns
    -------
    list of dict
        One complete analysis result per finish, in input order.

    Raises
    ------
    StructuralReuseError
        If structural keys disagree within the group when reuse/library is on.
    StructuralProviderError
        If library lookup or key/snapshot validation fails.
    """
    _ = (constraints, materials)
    if not param_dicts:
        return []

    if provider is not None and provider.mode == "library":
        if not reuse:
            raise StructuralReuseError(
                "structural_artifacts.mode='library' requires reuse=True"
            )
        if structural_id is None:
            raise StructuralProviderError(
                "library finish-group evaluation requires structural_id"
            )
        return _evaluate_finish_group_library(
            param_dicts,
            analysis_fn=analysis_fn,
            finish_names=finish_names,
            snapshot_hash=snapshot_hash,
            run_config=run_config,
            provider=provider,
            structural_id=int(structural_id),
            debug=debug,
        )

    if not reuse:
        return [
            analysis_fn(
                dict(params),
                debug=debug,
                run_config=run_config,
            )
            for params in param_dicts
        ]

    keys = _finish_group_structural_keys(
        param_dicts,
        finish_names=finish_names,
        snapshot_hash=snapshot_hash,
        run_config=run_config,
    )
    if len(set(keys)) != 1:
        raise StructuralReuseError(
            "finish group contains more than one structural key; "
            "shard boundaries must not split a finish group"
        )

    holder: list[StructuralArtifact] = []
    first = analysis_fn(
        dict(param_dicts[0]),
        debug=debug,
        run_config=run_config,
        structural_artifact_holder=holder,
        structural_capture_key=keys[0],
        structural_snapshot_hash=snapshot_hash,
    )
    if not holder:
        raise StructuralReuseError(
            "analysis did not capture a structural artifact; "
            "pass structural_artifact_holder through the analysis binding"
        )
    artifact = holder[0]
    results: list[dict[str, Any]] = [first]
    for params in param_dicts[1:]:
        if artifact.is_build_failure:
            results.append(fanout_build_failure(artifact, params))
            continue
        results.append(
            analysis_fn(
                dict(params),
                debug=debug,
                run_config=run_config,
                structural_artifact=artifact,
            )
        )
    return results
