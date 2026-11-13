"""
Sealed Parquet column-family discovery for brute-force analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class BruteForceColumnError(ValueError):
    """
    Raised when sealed schema column families are incomplete or inconsistent.
    """


@dataclass(frozen=True)
class BruteForceColumnFamilies:
    """
    Column families discovered from a sealed ``schema.json`` document.

    Attributes
    ----------
    field_names : tuple of str
        All sealed field names in schema order.
    idx_columns : tuple of str
        ``idx__*`` columns in sealed order.
    param_columns : tuple of str
        ``param__*`` columns in sealed order.
    util_columns : tuple of str
        ``util__*`` columns in sealed order.
    violation_columns : tuple of str
        ``violation__*`` columns in sealed order.
    penalty_columns : tuple of str
        ``penalty__*`` columns in sealed order.
    constraint_names : tuple of str
        Constraint stems implied by complete util/violation/penalty triplets.
    """

    field_names: tuple[str, ...]
    idx_columns: tuple[str, ...]
    param_columns: tuple[str, ...]
    util_columns: tuple[str, ...]
    violation_columns: tuple[str, ...]
    penalty_columns: tuple[str, ...]
    constraint_names: tuple[str, ...]


def _fields_from_schema_doc(schema_doc: Mapping[str, Any]) -> tuple[str, ...]:
    raw_fields = schema_doc.get("fields")
    if not isinstance(raw_fields, list) or not raw_fields:
        raise BruteForceColumnError("schema.json must contain a non-empty fields list")
    names: list[str] = []
    for index, field in enumerate(raw_fields):
        if not isinstance(field, Mapping):
            raise BruteForceColumnError(
                f"schema.json fields[{index}] must be an object"
            )
        name = field.get("name")
        if not isinstance(name, str) or not name:
            raise BruteForceColumnError(
                f"schema.json fields[{index}].name must be a non-empty string"
            )
        names.append(name)
    if len(set(names)) != len(names):
        raise BruteForceColumnError("schema.json field names must be unique")
    return tuple(names)


def _prefixed(field_names: Sequence[str], prefix: str) -> tuple[str, ...]:
    return tuple(name for name in field_names if name.startswith(prefix))


def _stem(column: str, prefix: str) -> str:
    return column[len(prefix) :]


def discover_column_families(
    schema_doc: Mapping[str, Any],
    *,
    expected_constraint_names: Sequence[str] | None = None,
) -> BruteForceColumnFamilies:
    """
    Discover and validate sealed column families.

    Parameters
    ----------
    schema_doc : Mapping
        Parsed ``dataset/schema.json`` document.
    expected_constraint_names : sequence of str or None, optional
        When provided, require exactly these constraint stems in the same order
        as the sealed schema constraint list.

    Returns
    -------
    BruteForceColumnFamilies
        Validated family lists.

    Raises
    ------
    BruteForceColumnError
        If families are incomplete, unknown, or inconsistent.
    """
    field_names = _fields_from_schema_doc(schema_doc)
    idx_columns = _prefixed(field_names, "idx__")
    param_columns = _prefixed(field_names, "param__")
    util_columns = _prefixed(field_names, "util__")
    violation_columns = _prefixed(field_names, "violation__")
    penalty_columns = _prefixed(field_names, "penalty__")

    util_stems = [_stem(name, "util__") for name in util_columns]
    violation_stems = [_stem(name, "violation__") for name in violation_columns]
    penalty_stems = [_stem(name, "penalty__") for name in penalty_columns]

    if util_stems != violation_stems or util_stems != penalty_stems:
        raise BruteForceColumnError(
            "util__/violation__/penalty__ constraint stems must match exactly; "
            f"got util={util_stems!r}, violation={violation_stems!r}, "
            f"penalty={penalty_stems!r}"
        )
    if any(not stem for stem in util_stems):
        raise BruteForceColumnError("constraint column stems must be non-empty")

    sealed_constraint_names = schema_doc.get("constraint_names")
    if sealed_constraint_names is not None:
        if not isinstance(sealed_constraint_names, list) or not all(
            isinstance(name, str) for name in sealed_constraint_names
        ):
            raise BruteForceColumnError(
                "schema.json constraint_names must be a list of strings when present"
            )
        if tuple(sealed_constraint_names) != tuple(util_stems):
            raise BruteForceColumnError(
                "schema.json constraint_names do not match util__/violation__/penalty__ "
                f"stems: {list(sealed_constraint_names)!r} != {util_stems!r}"
            )

    if expected_constraint_names is not None:
        expected = tuple(str(name) for name in expected_constraint_names)
        if expected != tuple(util_stems):
            raise BruteForceColumnError(
                "expected constraint names do not match sealed triplets: "
                f"{list(expected)!r} != {util_stems!r}"
            )

    known_prefixes = ("idx__", "param__", "util__", "violation__", "penalty__")
    for name in field_names:
        if "__" not in name:
            continue
        if not any(name.startswith(prefix) for prefix in known_prefixes):
            raise BruteForceColumnError(
                f"unknown column-family member {name!r}; "
                f"expected one of {known_prefixes}"
            )

    return BruteForceColumnFamilies(
        field_names=field_names,
        idx_columns=idx_columns,
        param_columns=param_columns,
        util_columns=util_columns,
        violation_columns=violation_columns,
        penalty_columns=penalty_columns,
        constraint_names=tuple(util_stems),
    )
