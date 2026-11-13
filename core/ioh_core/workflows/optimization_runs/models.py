"""
Typed optimisation-run task specifications.

These records are the workflow-specific payload stored inside generic
``TaskRecord`` entries of the experiment manifest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

_TERMINATION_PAYLOAD_KEYS = (
    "termination_mode",
    "global_optimum",
    "allowed_deviation_percent",
    "target_reference",
    "target_threshold",
)
_TERMINATION_MODES = frozenset({"fixed_budget", "fixed_target"})
_TARGET_REFERENCES = frozenset({"all"})
_THRESHOLD_RTOL = 1e-12


def _optional_finite_nonnegative(value: Any, *, field_name: str) -> float | None:
    """
    Validate an optional finite non-negative float payload field.

    Parameters
    ----------
    value : Any
        Candidate value (may be ``None``).
    field_name : str
        Field name for error messages.

    Returns
    -------
    float or None
        Coerced value, or ``None``.

    Raises
    ------
    ValueError
        If the value is not ``None`` and not a finite number >= 0.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number or null, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f"{field_name} must be a finite number >= 0, got {value!r}"
        )
    return number


@dataclass(frozen=True)
class OptimizationTaskSpec:
    """
    One optimisation task: problem × algorithm × repetition.

    Attributes
    ----------
    problem_id : str
        Ordinary problem ID from the snapshotted problem list.
    slab_type : str
        Slab package name such as ``\"hp_slab\"``.
    algorithm_name : str
        Factory name such as ``\"RANDOM_SEARCH\"`` or ``\"NLOPT_DIRECT_L\"``.
    algorithm_parameters : Mapping[str, Any]
        Strict algorithm hyperparameters (for example ``max_evaluations``).
    repetition : int
        Zero-based repetition index for this problem/algorithm pair.
    random_seed : int or None
        Deterministic unsigned 32-bit seed for stochastic algorithms, or
        ``None`` for seedless algorithms such as DIRECT-L.
    termination_mode : str
        ``\"fixed_budget\"`` or ``\"fixed_target\"``.
    global_optimum : float or None
        Known optimum for fixed-target mode; ``None`` for fixed-budget.
    allowed_deviation_percent : float or None
        Allowed percentage deviation for fixed-target mode; ``None`` for
        fixed-budget.
    target_reference : str or None
        ``\"all\"`` in fixed-target mode; ``None`` for fixed-budget.
    target_threshold : float or None
        Derived minimisation threshold for fixed-target mode; ``None`` for
        fixed-budget.
    """

    problem_id: str
    slab_type: str
    algorithm_name: str
    algorithm_parameters: Mapping[str, Any]
    repetition: int
    random_seed: int | None = None
    termination_mode: str = "fixed_budget"
    global_optimum: float | None = None
    allowed_deviation_percent: float | None = None
    target_reference: str | None = None
    target_threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.problem_id or self.problem_id == "default":
            raise ValueError(
                f"problem_id must be a non-empty ordinary id, got {self.problem_id!r}"
            )
        if not self.slab_type.strip():
            raise ValueError("slab_type must be a non-empty string")
        if not self.algorithm_name.strip():
            raise ValueError("algorithm_name must be a non-empty string")
        if self.repetition < 0:
            raise ValueError(f"repetition must be >= 0, got {self.repetition}")
        if self.random_seed is not None:
            if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
                raise ValueError(
                    f"random_seed must be an int or None, got {self.random_seed!r}"
                )
            if self.random_seed < 0 or self.random_seed > 0xFFFFFFFF:
                raise ValueError(
                    f"random_seed must be an unsigned 32-bit int, got {self.random_seed}"
                )
        object.__setattr__(self, "algorithm_parameters", dict(self.algorithm_parameters))
        self._validate_termination_fields()

    def _validate_termination_fields(self) -> None:
        """Validate termination mode and fixed-target field consistency."""
        if self.termination_mode not in _TERMINATION_MODES:
            raise ValueError(
                "termination_mode must be one of "
                f"{sorted(_TERMINATION_MODES)}, got {self.termination_mode!r}"
            )

        if self.termination_mode == "fixed_budget":
            if self.global_optimum is not None:
                raise ValueError(
                    "global_optimum must be null when termination_mode='fixed_budget'"
                )
            if self.allowed_deviation_percent is not None:
                raise ValueError(
                    "allowed_deviation_percent must be null when "
                    "termination_mode='fixed_budget'"
                )
            if self.target_reference is not None:
                raise ValueError(
                    "target_reference must be null when "
                    "termination_mode='fixed_budget'"
                )
            if self.target_threshold is not None:
                raise ValueError(
                    "target_threshold must be null when "
                    "termination_mode='fixed_budget'"
                )
            return

        if self.global_optimum is None:
            raise ValueError(
                "global_optimum is required when termination_mode='fixed_target'"
            )
        if self.allowed_deviation_percent is None:
            raise ValueError(
                "allowed_deviation_percent is required when "
                "termination_mode='fixed_target'"
            )
        if self.target_reference is None:
            raise ValueError(
                "target_reference is required when termination_mode='fixed_target'"
            )
        if self.target_reference not in _TARGET_REFERENCES:
            raise ValueError(
                "target_reference must be one of "
                f"{sorted(_TARGET_REFERENCES)}, got {self.target_reference!r}"
            )
        if self.target_threshold is None:
            raise ValueError(
                "target_threshold is required when termination_mode='fixed_target'"
            )

        optimum = _optional_finite_nonnegative(
            self.global_optimum,
            field_name="global_optimum",
        )
        deviation = _optional_finite_nonnegative(
            self.allowed_deviation_percent,
            field_name="allowed_deviation_percent",
        )
        threshold = _optional_finite_nonnegative(
            self.target_threshold,
            field_name="target_threshold",
        )
        assert optimum is not None and deviation is not None and threshold is not None
        expected = optimum * (1.0 + deviation / 100.0)
        if not math.isfinite(expected):
            raise ValueError(
                f"derived target_threshold must be finite, got {expected!r}"
            )
        if not math.isclose(threshold, expected, rel_tol=_THRESHOLD_RTOL, abs_tol=0.0):
            raise ValueError(
                "target_threshold must equal "
                "global_optimum * (1 + allowed_deviation_percent / 100) "
                f"within rel_tol={_THRESHOLD_RTOL}; got {threshold!r}, "
                f"expected {expected!r}"
            )
        object.__setattr__(self, "global_optimum", optimum)
        object.__setattr__(self, "allowed_deviation_percent", deviation)
        object.__setattr__(self, "target_threshold", threshold)

    def to_payload(self) -> dict[str, Any]:
        """
        Return a JSON-compatible payload for a generic ``TaskRecord``.

        Returns
        -------
        dict
            Mapping with the optimisation task fields.
        """
        return {
            "problem_id": self.problem_id,
            "slab_type": self.slab_type,
            "algorithm_name": self.algorithm_name,
            "algorithm_parameters": dict(self.algorithm_parameters),
            "repetition": self.repetition,
            "random_seed": self.random_seed,
            "termination_mode": self.termination_mode,
            "global_optimum": self.global_optimum,
            "allowed_deviation_percent": self.allowed_deviation_percent,
            "target_reference": self.target_reference,
            "target_threshold": self.target_threshold,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> OptimizationTaskSpec:
        """
        Build a spec from a manifest task payload.

        Completely absent termination fields are treated as fixed-budget
        defaults for older sealed manifests. Partially present termination
        fields are rejected.

        Parameters
        ----------
        payload : Mapping[str, Any]
            Workflow payload stored on a ``TaskRecord``.

        Returns
        -------
        OptimizationTaskSpec
            Validated frozen specification.

        Raises
        ------
        ValueError
            If termination fields are partial or inconsistent.
        """
        present = [key for key in _TERMINATION_PAYLOAD_KEYS if key in payload]
        if not present:
            termination_mode = "fixed_budget"
            global_optimum = None
            allowed_deviation_percent = None
            target_reference = None
            target_threshold = None
        elif len(present) != len(_TERMINATION_PAYLOAD_KEYS):
            raise ValueError(
                "optimization task payload has partial termination fields: "
                f"{present}; expected all of {list(_TERMINATION_PAYLOAD_KEYS)}"
            )
        else:
            termination_mode = str(payload["termination_mode"])
            global_optimum = payload["global_optimum"]
            allowed_deviation_percent = payload["allowed_deviation_percent"]
            target_reference = (
                None
                if payload["target_reference"] is None
                else str(payload["target_reference"])
            )
            target_threshold = payload["target_threshold"]

        return cls(
            problem_id=str(payload["problem_id"]),
            slab_type=str(payload["slab_type"]),
            algorithm_name=str(payload["algorithm_name"]),
            algorithm_parameters=dict(payload.get("algorithm_parameters") or {}),
            repetition=int(payload["repetition"]),
            random_seed=(
                None
                if payload.get("random_seed") is None
                else int(payload["random_seed"])
            ),
            termination_mode=termination_mode,
            global_optimum=(
                None if global_optimum is None else float(global_optimum)
            ),
            allowed_deviation_percent=(
                None
                if allowed_deviation_percent is None
                else float(allowed_deviation_percent)
            ),
            target_reference=target_reference,
            target_threshold=(
                None if target_threshold is None else float(target_threshold)
            ),
        )
