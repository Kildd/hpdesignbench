"""
ELA result, provenance, and landscape-sample records.

This module will own tidy feature rows and adapter output types. It must not
import DuckDB, pflacco, or SALib. Do not add a generic ``n_used`` field.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


class ElaNotImplementedError(RuntimeError):
    """Raised by ELA hooks that are not implemented yet."""


class ElaError(ValueError):
    """Raised when ELA configuration or source data is invalid."""


STATUS_OK = "ok"
STATUS_UNDEFINED = "undefined"
STATUS_MISSING_DEPENDENCY = "missing_dependency"
STATUS_INSUFFICIENT_N = "insufficient_n"
STATUS_CONSTANT_LANDSCAPE = "constant_landscape"
STATUS_ERROR = "error"
STATUS_VALUES = frozenset(
    {
        STATUS_OK,
        STATUS_UNDEFINED,
        STATUS_MISSING_DEPENDENCY,
        STATUS_INSUFFICIENT_N,
        STATUS_CONSTANT_LANDSCAPE,
        STATUS_ERROR,
    }
)

ROLE_PRIMARY = "primary"
ROLE_SECONDARY = "secondary"
ROLE_DIAGNOSTIC = "diagnostic"
ROLE_VALUES = frozenset({ROLE_PRIMARY, ROLE_SECONDARY, ROLE_DIAGNOSTIC})

LANDSCAPE_Y_RAW = "y_raw"
LANDSCAPE_Y_P = "y_p"
LANDSCAPE_PHI = "phi"
LANDSCAPE_FEASIBILITY = "feasibility"
LANDSCAPE_DEB = "deb"
LANDSCAPE_FIXED = frozenset(
    {
        LANDSCAPE_Y_RAW,
        LANDSCAPE_Y_P,
        LANDSCAPE_PHI,
        LANDSCAPE_FEASIBILITY,
        LANDSCAPE_DEB,
    }
)
CONSTRAINT_LANDSCAPE_PREFIX = "constraint_"

RESPONSE_VIEW_RAW = "raw"
RESPONSE_VIEW_MINMAX = "minmax"
RESPONSE_VIEW_NA = "n/a"
RESPONSE_VIEW_VALUES = frozenset(
    {RESPONSE_VIEW_RAW, RESPONSE_VIEW_MINMAX, RESPONSE_VIEW_NA}
)

X_VIEW_NORM = "norm"
X_VIEW_INDEX = "index"
X_VIEW_NA = "n/a"
X_VIEW_VALUES = frozenset({X_VIEW_NORM, X_VIEW_INDEX, X_VIEW_NA})

SOURCE_SAMPLING = "sampling"
SOURCE_BRUTE_FORCE = "brute_force"
SOURCE_TYPE_VALUES = frozenset({SOURCE_SAMPLING, SOURCE_BRUTE_FORCE})


def require_status(status: str) -> str:
    if status not in STATUS_VALUES:
        raise ElaError(f"unknown feature status {status!r}")
    return status


def require_role(role: str) -> str:
    if role not in ROLE_VALUES:
        raise ElaError(f"unknown feature role {role!r}")
    return role


def require_landscape(landscape: str) -> str:
    if landscape in LANDSCAPE_FIXED:
        return landscape
    if isinstance(landscape, str) and landscape.startswith(
        CONSTRAINT_LANDSCAPE_PREFIX
    ):
        stem = landscape[len(CONSTRAINT_LANDSCAPE_PREFIX) :]
        if stem and stem == stem.strip():
            return landscape
    raise ElaError(
        "landscape must be y_raw, y_p, phi, feasibility, deb, or constraint_<stem>, "
        f"got {landscape!r}"
    )


def require_response_view(response_view: str) -> str:
    if response_view not in RESPONSE_VIEW_VALUES:
        raise ElaError(f"unknown response_view {response_view!r}")
    return response_view


def require_x_view(x_view: str) -> str:
    if x_view not in X_VIEW_VALUES:
        raise ElaError(f"unknown x_view {x_view!r}")
    return x_view


@dataclass(frozen=True)
class LandscapeSample:
    """
    One ELA initial design: one sampling repetition or one brute-force scan.

    Attributes
    ----------
    source_type : str
        ``sampling`` or ``brute_force``.
    experiment_dir : Path
        Sealed experiment root.
    repetition : int or None
        Sampling repetition id. ``None`` for brute-force sources without
        a ``repetition`` column.
    var_names : tuple of str
        Sealed variable order (never alphabetised).
    constraint_names_sealed : tuple of str
        Full ``schema.json`` constraint list.
    constraint_names_selected : tuple of str
        Keep list used to build ``U`` / ``phi`` / ``F``.
    constraint_set_is_subset : bool
        True when selected names are a proper subset of sealed names.
    lower_bounds : tuple of int
        Inclusive catalogue lower index per variable.
    upper_bounds : tuple of int
        Inclusive catalogue upper index per variable.
    X_index : ndarray
        ``(n, d)`` catalogue indices in ``var_names`` order.
    y_raw : ndarray
        Stored unpenalized objective ``y``.
    y_p : ndarray
        Stored penalized objective (full evaluation constraint set).
    U : ndarray
        ``(n, k)`` utilizations for **selected** constraints.
    phi : ndarray
        Analysis-local ``sum max(u_i - 1, 0)`` over selected constraints.
    F : ndarray
        Analysis-local feasibility: all selected ``v_i == 0`` exactly.
    analysis_ok : ndarray
        Stored provenance flag. Not used as a row filter.
    sample_index : ndarray or None
        Evaluated-row order within the repetition (sampling).
    n_input : int
        Row count after the repetition filter (no silent drop).
    n_unique : int
        Unique complete ``X_index`` rows (diagnostic, not a gate).
    n_duplicates : int
        ``n_input - n_unique``.
    duplicate_fraction : float
        ``n_duplicates / n_input`` when ``n_input > 0``, else ``0.0``.
    delete_duplicates : bool or None
        Sealed sampling setting when ``design.json`` is present.
    metadata : dict
        Extra source provenance (workflow, duplicate stats, column notes).
    """

    source_type: str
    experiment_dir: Path
    repetition: int | None
    var_names: tuple[str, ...]
    constraint_names_sealed: tuple[str, ...]
    constraint_names_selected: tuple[str, ...]
    constraint_set_is_subset: bool
    lower_bounds: tuple[int, ...]
    upper_bounds: tuple[int, ...]
    X_index: NDArray[np.integer]
    y_raw: NDArray[np.floating]
    y_p: NDArray[np.floating]
    U: NDArray[np.floating]
    phi: NDArray[np.floating]
    F: NDArray[np.bool_]
    analysis_ok: NDArray[np.bool_]
    sample_index: NDArray[np.integer] | None
    n_input: int
    n_unique: int
    n_duplicates: int
    duplicate_fraction: float
    delete_duplicates: bool | None
    metadata: dict[str, Any]


def canonical_config_json(payload: Mapping[str, Any]) -> str:
    """
    Canonical JSON for ``config_payload`` (same pattern as schema hashes).

    ``json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True)``.
    Does not import ``optimization_runs.snapshot``.
    """
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Mapping):
        raise ElaError("config_payload must be a mapping")
    try:
        return json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except (TypeError, ValueError) as exc:
        raise ElaError("config_payload must be JSON-serialisable") from exc


def compute_config_hash(config_json: str) -> str:
    """Return the SHA-256 hex digest of canonical ``config_json`` bytes."""
    if not isinstance(config_json, str) or not config_json:
        raise ElaError("config_json must be a non-empty string")
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureResult:
    """
    One tidy ELA feature row.

    Identity columns ``repetition``, ``sample_id``, and ``walk_id`` are
    separate and nullable. There is no combined id field and no ``n_used``.
    ``n_unique`` is diagnostic only. Sampled orchestration fills
    ``sampling_design_signature`` and ``analysis_signature`` (Phase 19);
    direct wrapper calls may still leave them ``None``.
    """

    problem_id: str
    source_type: str
    source_dir: Path
    repetition: int | None
    sample_id: str | None
    walk_id: int | None
    landscape: str
    response_view: str
    z_min: float | None
    z_max: float | None
    x_view: str
    family: str
    feature_name: str
    value: float | None
    status: str
    role: str
    n_input: int
    n_unique: int
    runtime_s: float
    library: str
    library_version: str
    config_payload: dict[str, Any]
    config_json: str
    config_hash: str
    sampling_design_signature: str | None
    analysis_signature: str | None
    seed: int | None

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPE_VALUES:
            raise ElaError(
                "source_type must be 'sampling' or 'brute_force', "
                f"got {self.source_type!r}"
            )
        require_landscape(self.landscape)
        require_response_view(self.response_view)
        require_x_view(self.x_view)
        require_status(self.status)
        require_role(self.role)
        if not isinstance(self.n_input, int) or isinstance(self.n_input, bool):
            raise ElaError("n_input must be an int")
        if not isinstance(self.n_unique, int) or isinstance(self.n_unique, bool):
            raise ElaError("n_unique must be an int")
        if self.n_input < 0 or self.n_unique < 0:
            raise ElaError("n_input and n_unique must be >= 0")
        expected = compute_config_hash(self.config_json)
        if self.config_hash != expected:
            raise ElaError("config_hash does not match config_json")


def feature_result(
    *,
    problem_id: str,
    source_type: str,
    source_dir: Path,
    landscape: str,
    response_view: str,
    x_view: str,
    family: str,
    feature_name: str,
    status: str,
    role: str,
    n_input: int,
    n_unique: int,
    runtime_s: float,
    library: str,
    library_version: str,
    config_payload: Mapping[str, Any],
    value: float | None = None,
    repetition: int | None = None,
    sample_id: str | None = None,
    walk_id: int | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    sampling_design_signature: str | None = None,
    analysis_signature: str | None = None,
    seed: int | None = None,
) -> FeatureResult:
    """
    Build one feature row and derive ``config_json`` / ``config_hash``.

    Callers do not pass the hash. Signature fields stay ``None`` until
    Phase 19. ``n_unique`` is stored, not used as a validity gate.
    """
    payload = dict(config_payload)
    text = canonical_config_json(payload)
    return FeatureResult(
        problem_id=problem_id,
        source_type=source_type,
        source_dir=Path(source_dir),
        repetition=repetition,
        sample_id=sample_id,
        walk_id=walk_id,
        landscape=landscape,
        response_view=response_view,
        z_min=z_min,
        z_max=z_max,
        x_view=x_view,
        family=family,
        feature_name=feature_name,
        value=value,
        status=status,
        role=role,
        n_input=n_input,
        n_unique=n_unique,
        runtime_s=runtime_s,
        library=library,
        library_version=library_version,
        config_payload=payload,
        config_json=text,
        config_hash=compute_config_hash(text),
        sampling_design_signature=sampling_design_signature,
        analysis_signature=analysis_signature,
        seed=seed,
    )
