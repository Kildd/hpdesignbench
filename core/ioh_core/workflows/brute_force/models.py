"""
Frozen index-space model for deterministic brute-force enumeration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# Signed int64 upper bound used by Parquet / Arrow integer columns later.
_INT64_MAX = (1 << 63) - 1


class IndexSpaceError(ValueError):
    """
    Raised when an index space cannot be constructed or is inconsistent.
    """


def _checked_mul(left: int, right: int) -> int:
    """
    Multiply two non-negative ints or raise on signed-int64 overflow.

    Parameters
    ----------
    left, right : int
        Non-negative factors.

    Returns
    -------
    int
        ``left * right`` when the product fits in signed int64.

    Raises
    ------
    IndexSpaceError
        If the product would exceed ``2**63 - 1``.
    """
    if left < 0 or right < 0:
        raise IndexSpaceError(
            f"checked multiply requires non-negative factors, got {left} * {right}"
        )
    if left == 0 or right == 0:
        return 0
    if left > _INT64_MAX // right:
        raise IndexSpaceError(
            f"index-space product overflows signed int64: {left} * {right}"
        )
    return left * right


@dataclass(frozen=True)
class IndexSpace:
    """
    Inclusive catalogue-index domain with mixed-radix linear addressing.

    Ordering matches ``itertools.product``: the first variable is outermost and
    the last variable changes fastest. Catalogue indices are inclusive
    ``[lower_bounds[i], upper_bounds[i]]``; linear addresses use
    ``[0, cardinality)``.

    Attributes
    ----------
    var_names : tuple of str
        Variable names in product order.
    lower_bounds : tuple of int
        Inclusive lower catalogue index per variable.
    upper_bounds : tuple of int
        Inclusive upper catalogue index per variable.
    sizes : tuple of int
        ``upper_bounds[i] - lower_bounds[i] + 1`` per variable.
    strides : tuple of int
        ``stride[i] = product(sizes[i + 1:])`` (last stride is ``1``).
    cardinality : int
        Total number of catalogue vectors (``product(sizes)``).
    """

    var_names: tuple[str, ...]
    lower_bounds: tuple[int, ...]
    upper_bounds: tuple[int, ...]
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    cardinality: int

    @classmethod
    def from_bounds(
        cls,
        var_names: Sequence[str],
        lower_bounds: Sequence[int],
        upper_bounds: Sequence[int],
    ) -> IndexSpace:
        """
        Build an :class:`IndexSpace` from inclusive catalogue bounds.

        Parameters
        ----------
        var_names : sequence of str
            Non-empty unique variable names.
        lower_bounds : sequence of int
            Inclusive lower catalogue indices.
        upper_bounds : sequence of int
            Inclusive upper catalogue indices.

        Returns
        -------
        IndexSpace
            Frozen space with derived sizes, strides, and cardinality.

        Raises
        ------
        IndexSpaceError
            If dimensions are empty/mismatched, names duplicate, bounds are
            reversed, a size is non-positive, or products overflow int64.
        """
        names = tuple(str(name) for name in var_names)
        lowers = tuple(int(value) for value in lower_bounds)
        uppers = tuple(int(value) for value in upper_bounds)

        if not names:
            raise IndexSpaceError("index space requires at least one variable")
        if len(names) != len(lowers) or len(names) != len(uppers):
            raise IndexSpaceError(
                "var_names, lower_bounds, and upper_bounds must have equal "
                f"lengths, got {len(names)}, {len(lowers)}, {len(uppers)}"
            )
        if any(not name for name in names):
            raise IndexSpaceError("var_names must be non-empty strings")
        if len(set(names)) != len(names):
            raise IndexSpaceError(f"var_names must be unique, got {names!r}")

        sizes_list: list[int] = []
        for index, (lo, hi) in enumerate(zip(lowers, uppers)):
            if lo > hi:
                raise IndexSpaceError(
                    f"reversed bounds at dimension {index} ({names[index]!r}): "
                    f"lower={lo} upper={hi}"
                )
            size = hi - lo + 1
            if size < 1:
                raise IndexSpaceError(
                    f"non-positive size at dimension {index} ({names[index]!r}): "
                    f"{size}"
                )
            sizes_list.append(size)
        sizes = tuple(sizes_list)

        strides_list = [1] * len(sizes)
        for index in range(len(sizes) - 2, -1, -1):
            strides_list[index] = _checked_mul(sizes[index + 1], strides_list[index + 1])
        strides = tuple(strides_list)

        cardinality = 1
        for size in sizes:
            cardinality = _checked_mul(cardinality, size)

        return cls(
            var_names=names,
            lower_bounds=lowers,
            upper_bounds=uppers,
            sizes=sizes,
            strides=strides,
            cardinality=cardinality,
        )


@dataclass(frozen=True)
class BruteForceShardSpec:
    """
    One brute-force shard: a half-open linear index range for one problem.

    Attributes
    ----------
    problem_id : str
        Ordinary problem ID.
    slab_type : str
        Slab package name.
    shard_index : int
        Zero-based shard ordinal within the experiment.
    start_index : int
        Inclusive linear start (``0 <= start_index < stop_index``).
    stop_index : int
        Exclusive linear stop.
    schema_hash : str
        Deterministic Parquet schema fingerprint for the shard payload.
    """

    problem_id: str
    slab_type: str
    shard_index: int
    start_index: int
    stop_index: int
    schema_hash: str

    def __post_init__(self) -> None:
        if not self.problem_id or self.problem_id == "default":
            raise ValueError(
                f"problem_id must be a non-empty ordinary id, got {self.problem_id!r}"
            )
        if not self.slab_type.strip():
            raise ValueError("slab_type must be a non-empty string")
        if self.shard_index < 0:
            raise ValueError(f"shard_index must be >= 0, got {self.shard_index}")
        if self.start_index < 0 or self.stop_index <= self.start_index:
            raise ValueError(
                f"invalid shard range [{self.start_index}, {self.stop_index})"
            )
        if not self.schema_hash.strip():
            raise ValueError("schema_hash must be a non-empty string")

    def to_payload(self) -> dict[str, Any]:
        """
        Return a JSON-compatible payload for a generic ``TaskRecord``.

        Returns
        -------
        dict
            Mapping with the shard fields.
        """
        return {
            "problem_id": self.problem_id,
            "slab_type": self.slab_type,
            "shard_index": self.shard_index,
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "schema_hash": self.schema_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> BruteForceShardSpec:
        """
        Build a shard spec from a manifest task payload.

        Parameters
        ----------
        payload : Mapping[str, Any]
            Workflow payload stored on a ``TaskRecord``.

        Returns
        -------
        BruteForceShardSpec
            Validated frozen specification.
        """
        return cls(
            problem_id=str(payload["problem_id"]),
            slab_type=str(payload["slab_type"]),
            shard_index=int(payload["shard_index"]),
            start_index=int(payload["start_index"]),
            stop_index=int(payload["stop_index"]),
            schema_hash=str(payload["schema_hash"]),
        )
