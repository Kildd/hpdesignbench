"""
Shared helpers to attach a structural provider to evaluation contexts.
"""
from __future__ import annotations

from typing import Any, Mapping

from core.ioh_core.cache_eval import EvalContext
from core.ioh_core.structural_artifacts.artifact import finish_variable_names
from core.ioh_core.structural_artifacts.indexing import (
    candidate_indices_to_structural_id,
)
from core.ioh_core.workflows.brute_force.models import IndexSpace


def index_space_from_bundle(bundle: Mapping[str, Any]) -> IndexSpace:
    """
    Build an :class:`IndexSpace` from a problem bundle's catalogue bounds.
    """
    return IndexSpace.from_bounds(
        tuple(str(name) for name in bundle["var_names"]),
        tuple(int(v) for v in bundle["lb_idx"]),
        tuple(int(v) for v in bundle["ub_idx"]),
    )


def configure_eval_context_structural_backend(
    ctx: EvalContext,
    bundle: Mapping[str, Any],
    *,
    provider: Any,
    snapshot_hash: str,
) -> IndexSpace:
    """
    Attach a structural provider to an :class:`EvalContext` for library-mode evals.

    Returns
    -------
    IndexSpace
        Catalogue index space derived from the bundle (useful for tests).
    """
    space = index_space_from_bundle(bundle)
    finish_names = finish_variable_names(space.var_names)

    def _structural_id_fn(key: tuple[int, ...]) -> int:
        return candidate_indices_to_structural_id(key, space)

    ctx.configure_structural_backend(
        provider=provider,
        structural_id_fn=_structural_id_fn,
        finish_names=finish_names,
        snapshot_hash=snapshot_hash,
    )
    return space


def configure_bundle_structural_backend(
    bundle: dict[str, Any],
    *,
    provider: Any,
    snapshot_hash: str,
) -> None:
    """
    Attach a structural provider to ``bundle[\"ctx\"]`` and record metadata.
    """
    space = configure_eval_context_structural_backend(
        bundle["ctx"],
        bundle,
        provider=provider,
        snapshot_hash=snapshot_hash,
    )
    bundle["index_space"] = space
    bundle["finish_names"] = finish_variable_names(space.var_names)
    bundle["snapshot_hash"] = str(snapshot_hash)
    bundle["structural_provider"] = provider
