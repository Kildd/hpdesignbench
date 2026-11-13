"""
Shared keep-list selection for sealed ``constraint_names``.

Used by ELA and brute-force analysis. Default is the full sealed list
(including modeling/Z names on hp_slab). A proper subset emits
:class:`RuntimeWarning`; it does not fail.

This module must not import DuckDB, pflacco, SALib, PyArrow, the ELA
package, or ``constraint_decision_tree``.
"""
from __future__ import annotations

import warnings
from typing import Sequence

CONSTRAINT_SUBSET_WARNING = (
    "constraint keep-list is a proper subset of sealed constraint_names; "
    "phi/F and constraint-column metrics use the kept names only; "
    "stored y_p and Parquet feasible/constraint_feasible still use the "
    "full evaluation constraint set (including modeling/Z checks). "
    "This can change analysis results."
)


class ConstraintSelectionError(ValueError):
    """Raised when a sealed list or keep list is invalid."""


def select_constraint_names(
    sealed: Sequence[str],
    requested: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """
    Return sealed constraint names or a validated keep list.

    Parameters
    ----------
    sealed : sequence of str
        Authoritative ``schema.json`` ``constraint_names``. Must be
        non-empty, unique, and non-blank.
    requested : sequence of str or None, optional
        Keep list in sealed relative order. ``None`` keeps every sealed
        name. This is not an ignore list.

    Returns
    -------
    tuple of str
        Selected names in sealed order.

    Raises
    ------
    ConstraintSelectionError
        If ``sealed`` or ``requested`` is empty, blank, duplicated,
        unknown, or in the wrong relative order.
    """
    sealed_names = _require_name_tuple(sealed, field="sealed")
    if requested is None:
        return sealed_names
    keep = _require_name_tuple(requested, field="requested")
    sealed_index = {name: index for index, name in enumerate(sealed_names)}
    unknown = [name for name in keep if name not in sealed_index]
    if unknown:
        raise ConstraintSelectionError(
            f"requested constraint names are not in the sealed list: {unknown}"
        )
    previous = -1
    for name in keep:
        index = sealed_index[name]
        if index <= previous:
            raise ConstraintSelectionError(
                "requested constraint names must keep sealed relative order, "
                f"got {list(keep)!r} vs sealed {list(sealed_names)!r}"
            )
        previous = index
    return keep


def warn_if_constraint_subset(
    sealed: Sequence[str],
    selected: Sequence[str],
) -> None:
    """
    Warn when ``selected`` is a proper subset of ``sealed``.

    Parameters
    ----------
    sealed : sequence of str
        Authoritative sealed names.
    selected : sequence of str
        Names returned by :func:`select_constraint_names`.

    Notes
    -----
    Does not validate the keep list. Call :func:`select_constraint_names`
    first. No warning when the two sequences are equal.
    """
    if tuple(selected) == tuple(sealed):
        return
    warnings.warn(
        CONSTRAINT_SUBSET_WARNING,
        RuntimeWarning,
        stacklevel=2,
    )


def _require_name_tuple(names: Sequence[str], *, field: str) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise ConstraintSelectionError(
            f"{field} must be a sequence of strings, got {type(names).__name__}"
        )
    try:
        values = tuple(names)
    except TypeError as exc:
        raise ConstraintSelectionError(
            f"{field} must be a sequence of strings, got {type(names).__name__}"
        ) from exc
    if not values:
        raise ConstraintSelectionError(f"{field} must be a non-empty sequence")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str) or isinstance(item, bool):
            raise ConstraintSelectionError(
                f"{field} entries must be strings, got {item!r}"
            )
        name = item.strip()
        if not name or name != item:
            raise ConstraintSelectionError(
                f"{field} entries must be non-blank strings without "
                f"surrounding whitespace, got {item!r}"
            )
        if name in seen:
            raise ConstraintSelectionError(
                f"{field} must not contain duplicates, got {list(values)!r}"
            )
        seen.add(name)
        cleaned.append(name)
    return tuple(cleaned)
