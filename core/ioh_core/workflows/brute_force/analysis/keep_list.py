"""
Keep-list overlay for brute-force analysis commands.

Default selected names are the sealed ``constraint_names`` (Z included).
This module must not import the ELA package.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.ioh_core.constraint_selection import (
    ConstraintSelectionError,
    select_constraint_names,
    warn_if_constraint_subset,
)
from core.ioh_core.workflows.brute_force.analysis.context import (
    BruteForceAnalysisError,
)
from core.ioh_core.workflows.brute_force.analysis.models import (
    BruteForceDatasetContext,
)


def analysis_constraint_names(context: BruteForceDatasetContext) -> tuple[str, ...]:
    """
    Return names used for constraint-column analysis.

    Parameters
    ----------
    context : BruteForceDatasetContext
        Sealed dataset context. ``selected_constraint_names`` overlays
        ``constraint_names`` when set.

    Returns
    -------
    tuple of str
        Selected names, or the sealed list when no overlay is set.
    """
    selected = context.selected_constraint_names
    if selected is None:
        return tuple(context.constraint_names)
    return tuple(selected)


def constraint_provenance(
    sealed: Sequence[str],
    selected: Sequence[str],
) -> dict[str, Any]:
    """Return JSON provenance for a sealed vs selected constraint list."""
    sealed_names = list(sealed)
    selected_names = list(selected)
    return {
        "constraint_names_sealed": sealed_names,
        "constraint_names_selected": selected_names,
        "constraint_set_is_subset": tuple(selected_names) != tuple(sealed_names),
    }


def merge_constraint_provenance(
    document: Mapping[str, Any],
    *,
    sealed: Sequence[str],
    selected: Sequence[str],
) -> dict[str, Any]:
    """Copy ``document`` and attach keep-list provenance keys."""
    payload = dict(document)
    payload.update(constraint_provenance(sealed, selected))
    return payload


def load_sealed_constraint_names(experiment_dir: Path) -> tuple[str, ...] | None:
    """
    Read ``constraint_names`` from ``dataset/schema.json`` when present.

    Uses the standard library only (no DuckDB / PyArrow).
    """
    path = Path(experiment_dir) / "dataset" / "schema.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BruteForceAnalysisError(
            f"invalid dataset/schema.json: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise BruteForceAnalysisError("dataset/schema.json must be a JSON object")
    names = payload.get("constraint_names")
    if not isinstance(names, list) or not names or not all(
        isinstance(name, str) and name.strip() and name == name.strip()
        for name in names
    ):
        raise BruteForceAnalysisError(
            "dataset/schema.json constraint_names must be a non-empty "
            "list of non-blank strings"
        )
    return tuple(str(name) for name in names)


def resolve_analysis_constraint_names(
    sealed: Sequence[str],
    requested: Sequence[str] | None,
    *,
    implicit_z_drop: bool = False,
) -> tuple[str, ...]:
    """
    Resolve a keep list or the decision-tree implicit Z-drop.

    Parameters
    ----------
    sealed : sequence of str
        Authoritative schema names.
    requested : sequence of str or None
        Explicit keep list. ``None`` keeps every sealed name unless
        ``implicit_z_drop`` is true.
    implicit_z_drop : bool, optional
        When true and ``requested`` is omitted, drop names starting with
        ``Z`` (decision-tree default).

    Returns
    -------
    tuple of str
        Selected names.

    Raises
    ------
    BruteForceAnalysisError
        If the keep list is invalid or the implicit drop leaves no names.
    """
    try:
        if requested is not None:
            selected = select_constraint_names(sealed, requested)
        elif implicit_z_drop:
            selected = tuple(
                name for name in sealed if not str(name).startswith("Z")
            )
            if not selected:
                raise BruteForceAnalysisError(
                    "no active non-Z constraints available for repair trees"
                )
        else:
            selected = select_constraint_names(sealed, None)
    except ConstraintSelectionError as exc:
        raise BruteForceAnalysisError(str(exc)) from exc
    warn_if_constraint_subset(sealed, selected)
    return selected


def attach_selected_constraint_names(
    context: BruteForceDatasetContext,
    selected: Sequence[str],
) -> BruteForceDatasetContext:
    """Return a copy of ``context`` with ``selected_constraint_names`` set."""
    return replace(context, selected_constraint_names=tuple(selected))


def assert_subset_analysis_output_dir(
    experiment_dir: Path,
    output_dir: Path,
    *,
    output_dir_provided: bool,
) -> None:
    """
    Refuse a subset keep-list writing ``aggregate/analysis``.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root.
    output_dir : Path
        Resolved output directory (may be the default).
    output_dir_provided : bool
        Whether the operator set ``--output-dir``.

    Raises
    ------
    BruteForceAnalysisError
        If output was omitted or resolves to the canonical analysis path.
    """
    if not output_dir_provided:
        raise BruteForceAnalysisError(
            "constraint keep-list is a proper subset of sealed "
            "constraint_names; pass --output-dir so the subset run cannot "
            "overwrite canonical analysis artifacts"
        )
    resolved = Path(output_dir).resolve()
    canonical = (Path(experiment_dir) / "aggregate" / "analysis").resolve()
    if resolved == canonical:
        raise BruteForceAnalysisError(
            "constraint keep-list subset must not write canonical "
            f"analysis directories: {output_dir}"
        )
