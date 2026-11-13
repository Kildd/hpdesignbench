"""
Frozen metadata models for sealed Sobol sensitivity designs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

DESIGN_SCHEMA_VERSION = 1
PACKING_RULE = "saltelli_salib_A_then_AB_then_B"
SAMPLING_ENGINE = "SALib.sample.sobol"
ANALYSIS_ENGINE = "SALib.analyze.sobol"
INDEX_MAPPING = "lower + floor(u * (upper - lower + 1)), clipped to [lower, upper]"

_ARRAY_NAMES = (
    "A_unit.npy",
    "B_unit.npy",
    "A_index.npy",
    "B_index.npy",
)


class SensitivityDesignError(ValueError):
    """
    Raised when a sensitivity design cannot be resolved or sealed.
    """


def _json_safe(value: Any) -> Any:
    """Convert catalogue scalars to JSON-friendly Python types."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float):
            number = float(value)
            if number != number or number in (float("inf"), float("-inf")):
                raise SensitivityDesignError(
                    f"catalogue value is not JSON-finite: {value!r}"
                )
            return number
        return int(value)
    # NumPy scalars and similar.
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    raise SensitivityDesignError(
        f"unsupported catalogue value type {type(value).__name__}: {value!r}"
    )


@dataclass(frozen=True)
class SensitivityFactorRecord:
    """One selected sensitivity factor and its catalogue window."""

    name: str
    lower_index: int
    upper_index: int
    state_count: int
    catalogue_values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise SensitivityDesignError("factor name must be non-empty")
        if self.lower_index > self.upper_index:
            raise SensitivityDesignError(
                f"factor {self.name!r} has reversed bounds "
                f"[{self.lower_index}, {self.upper_index}]"
            )
        expected = self.upper_index - self.lower_index + 1
        if self.state_count != expected:
            raise SensitivityDesignError(
                f"factor {self.name!r} state_count={self.state_count} "
                f"does not match bounds window {expected}"
            )
        if self.state_count < 2:
            raise SensitivityDesignError(
                f"factor {self.name!r} needs at least two catalogue states"
            )
        if len(self.catalogue_values) != self.state_count:
            raise SensitivityDesignError(
                f"factor {self.name!r} catalogue_values length "
                f"{len(self.catalogue_values)} != state_count {self.state_count}"
            )
        object.__setattr__(
            self,
            "catalogue_values",
            tuple(_json_safe(item) for item in self.catalogue_values),
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-round-trippable mapping."""
        return {
            "name": self.name,
            "lower_index": self.lower_index,
            "upper_index": self.upper_index,
            "state_count": self.state_count,
            "catalogue_values": list(self.catalogue_values),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> SensitivityFactorRecord:
        """Build from a JSON object."""
        return cls(
            name=str(payload["name"]),
            lower_index=int(payload["lower_index"]),
            upper_index=int(payload["upper_index"]),
            state_count=int(payload["state_count"]),
            catalogue_values=tuple(payload["catalogue_values"]),
        )


@dataclass(frozen=True)
class HeldConstantVariable:
    """An unselected problem variable held at its resolved fixed index."""

    name: str
    fixed_index: int
    catalogue_value: Any

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise SensitivityDesignError("held-constant name must be non-empty")
        if self.fixed_index < 0:
            raise SensitivityDesignError(
                f"held-constant {self.name!r} fixed_index must be >= 0"
            )
        object.__setattr__(
            self,
            "catalogue_value",
            _json_safe(self.catalogue_value),
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-round-trippable mapping."""
        return {
            "name": self.name,
            "fixed_index": self.fixed_index,
            "catalogue_value": self.catalogue_value,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> HeldConstantVariable:
        """Build from a JSON object."""
        return cls(
            name=str(payload["name"]),
            fixed_index=int(payload["fixed_index"]),
            catalogue_value=payload["catalogue_value"],
        )


@dataclass(frozen=True)
class SensitivityDesignDocument:
    """
    Sealed ``design.json`` payload for one sensitivity experiment.

    Attributes
    ----------
    design_schema_version : int
        Document schema version.
    problem_id, slab_type : str
        Resolved problem identity.
    sample_size, factor_count : int
        ``N`` and ``k``.
    factor_names : tuple of str
        Ordered selected factors.
    factors : tuple of SensitivityFactorRecord
        Per-factor catalogue windows.
    held_constant_variables : tuple of HeldConstantVariable
        Unselected ``var`` roles held at ``fixed_idx``.
    response_key : str
        ``y`` or ``y_p``.
    scramble : bool
        SALib scramble flag.
    seed : int
        SALib sample seed.
    calc_second_order : bool
        Always ``False`` in C1.
    packing_rule : str
        Locked SALib packing identifier.
    index_mapping : str
        Discrete map description.
    expected_evaluations : int
        ``N * (k + 2)``.
    sampling_engine, analysis_engine : str
        Engine identifiers.
    numpy_version, salib_version : str
        Library versions recorded at seal time.
    array_sha256 : Mapping[str, str]
        Digests of the four sealed ``.npy`` files.
    """

    design_schema_version: int
    problem_id: str
    slab_type: str
    sample_size: int
    factor_count: int
    factor_names: tuple[str, ...]
    factors: tuple[SensitivityFactorRecord, ...]
    held_constant_variables: tuple[HeldConstantVariable, ...]
    response_key: str
    scramble: bool
    seed: int
    calc_second_order: bool
    packing_rule: str
    index_mapping: str
    expected_evaluations: int
    sampling_engine: str
    analysis_engine: str
    numpy_version: str
    salib_version: str
    array_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.design_schema_version != DESIGN_SCHEMA_VERSION:
            raise SensitivityDesignError(
                f"unsupported design_schema_version {self.design_schema_version}"
            )
        if self.factor_count != len(self.factor_names):
            raise SensitivityDesignError(
                "factor_count does not match factor_names length"
            )
        if len(self.factors) != self.factor_count:
            raise SensitivityDesignError(
                "factors length does not match factor_count"
            )
        names_from_factors = tuple(factor.name for factor in self.factors)
        if names_from_factors != self.factor_names:
            raise SensitivityDesignError(
                "factors order/names must match factor_names"
            )
        if self.calc_second_order:
            raise SensitivityDesignError(
                "C1 seals require calc_second_order=False"
            )
        if self.packing_rule != PACKING_RULE:
            raise SensitivityDesignError(
                f"unexpected packing_rule {self.packing_rule!r}"
            )
        expected = self.sample_size * (self.factor_count + 2)
        if self.expected_evaluations != expected:
            raise SensitivityDesignError(
                "expected_evaluations does not match N*(k+2)"
            )
        missing = [name for name in _ARRAY_NAMES if name not in self.array_sha256]
        if missing:
            raise SensitivityDesignError(
                f"array_sha256 missing entries: {missing}"
            )
        object.__setattr__(
            self,
            "array_sha256",
            dict(self.array_sha256),
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-round-trippable mapping."""
        return {
            "design_schema_version": self.design_schema_version,
            "problem_id": self.problem_id,
            "slab_type": self.slab_type,
            "sample_size": self.sample_size,
            "factor_count": self.factor_count,
            "factor_names": list(self.factor_names),
            "factors": [factor.to_json() for factor in self.factors],
            "held_constant_variables": [
                item.to_json() for item in self.held_constant_variables
            ],
            "response_key": self.response_key,
            "scramble": self.scramble,
            "seed": self.seed,
            "calc_second_order": self.calc_second_order,
            "packing_rule": self.packing_rule,
            "index_mapping": self.index_mapping,
            "expected_evaluations": self.expected_evaluations,
            "sampling_engine": self.sampling_engine,
            "analysis_engine": self.analysis_engine,
            "numpy_version": self.numpy_version,
            "salib_version": self.salib_version,
            "array_sha256": dict(self.array_sha256),
        }

    def to_json_text(self) -> str:
        """Canonical JSON text used for durable ``design.json`` writes."""
        return json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> SensitivityDesignDocument:
        """Build from a JSON object."""
        return cls(
            design_schema_version=int(payload["design_schema_version"]),
            problem_id=str(payload["problem_id"]),
            slab_type=str(payload["slab_type"]),
            sample_size=int(payload["sample_size"]),
            factor_count=int(payload["factor_count"]),
            factor_names=tuple(str(name) for name in payload["factor_names"]),
            factors=tuple(
                SensitivityFactorRecord.from_json(item)
                for item in payload["factors"]
            ),
            held_constant_variables=tuple(
                HeldConstantVariable.from_json(item)
                for item in payload["held_constant_variables"]
            ),
            response_key=str(payload["response_key"]),
            scramble=bool(payload["scramble"]),
            seed=int(payload["seed"]),
            calc_second_order=bool(payload["calc_second_order"]),
            packing_rule=str(payload["packing_rule"]),
            index_mapping=str(payload["index_mapping"]),
            expected_evaluations=int(payload["expected_evaluations"]),
            sampling_engine=str(payload["sampling_engine"]),
            analysis_engine=str(payload["analysis_engine"]),
            numpy_version=str(payload["numpy_version"]),
            salib_version=str(payload["salib_version"]),
            array_sha256={
                str(key): str(value)
                for key, value in dict(payload["array_sha256"]).items()
            },
        )


def expected_evaluation_count(sample_size: int, factor_count: int) -> int:
    """
    Return ``N * (k + 2)`` or raise if the product overflows signed int64.
    """
    if sample_size < 1 or factor_count < 1:
        raise SensitivityDesignError(
            "sample_size and factor_count must be positive for evaluation count"
        )
    rows_per_block = factor_count + 2
    int64_max = (1 << 63) - 1
    if sample_size > int64_max // rows_per_block:
        raise SensitivityDesignError(
            f"N*(k+2) overflows signed int64: N={sample_size}, k={factor_count}"
        )
    return sample_size * rows_per_block


@dataclass(frozen=True)
class SensitivityShardSpec:
    """
    One sensitivity shard: a half-open range of complete radial blocks.

    Attributes
    ----------
    problem_id : str
        Ordinary problem ID.
    slab_type : str
        Slab package name.
    shard_index : int
        Zero-based shard ordinal.
    block_start : int
        Inclusive radial block index.
    block_stop : int
        Exclusive radial block index.
    factor_count : int
        Number of factors ``k``.
    rows_per_block : int
        Always ``k + 2``.
    schema_hash : str
        Parquet schema fingerprint.
    """

    problem_id: str
    slab_type: str
    shard_index: int
    block_start: int
    block_stop: int
    factor_count: int
    rows_per_block: int
    schema_hash: str

    def __post_init__(self) -> None:
        if not self.problem_id or self.problem_id == "default":
            raise SensitivityDesignError(
                f"problem_id must be a non-empty ordinary id, got {self.problem_id!r}"
            )
        if not self.slab_type.strip():
            raise SensitivityDesignError("slab_type must be a non-empty string")
        if self.shard_index < 0:
            raise SensitivityDesignError(
                f"shard_index must be >= 0, got {self.shard_index}"
            )
        if self.block_start < 0 or self.block_stop <= self.block_start:
            raise SensitivityDesignError(
                f"invalid block range [{self.block_start}, {self.block_stop})"
            )
        if self.factor_count < 1:
            raise SensitivityDesignError(
                f"factor_count must be >= 1, got {self.factor_count}"
            )
        if self.rows_per_block != self.factor_count + 2:
            raise SensitivityDesignError(
                f"rows_per_block must equal k+2={self.factor_count + 2}, "
                f"got {self.rows_per_block}"
            )
        if not self.schema_hash.strip():
            raise SensitivityDesignError("schema_hash must be a non-empty string")

    @property
    def block_count(self) -> int:
        """Number of complete radial blocks in this shard."""
        return self.block_stop - self.block_start

    @property
    def row_count(self) -> int:
        """Number of evaluation rows written by this shard."""
        return self.block_count * self.rows_per_block

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible task payload."""
        return {
            "problem_id": self.problem_id,
            "slab_type": self.slab_type,
            "shard_index": self.shard_index,
            "block_start": self.block_start,
            "block_stop": self.block_stop,
            "factor_count": self.factor_count,
            "rows_per_block": self.rows_per_block,
            "schema_hash": self.schema_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SensitivityShardSpec:
        """Build from a manifest task payload."""
        return cls(
            problem_id=str(payload["problem_id"]),
            slab_type=str(payload["slab_type"]),
            shard_index=int(payload["shard_index"]),
            block_start=int(payload["block_start"]),
            block_stop=int(payload["block_stop"]),
            factor_count=int(payload["factor_count"]),
            rows_per_block=int(payload["rows_per_block"]),
            schema_hash=str(payload["schema_hash"]),
        )
