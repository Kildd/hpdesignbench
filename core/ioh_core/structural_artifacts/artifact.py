"""
Shared StructuralArtifact primitives (capture, apply, keys, equality).

These helpers are used by brute-force finish-group reuse, optional library
backends, and slab analysis hooks. They must not depend on workflow-specific
storage or EvalContext candidate-cache metrics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from structuralcodes.core._section_results import MomentCurvatureResults

from core.analysis_core.run_config import AnalysisRunConfig
from core.analysis_core.section_methods import DEFLECTION_MK_CACHE_POSITIONS
from slab_construction.slab_construction import (
    AnalysisCache,
    MKCurveKey,
    SectionResultKey,
)

# A BeamSection that can still be mutated is never stored in an artifact.
_STRIP_RESULT_KEYS = frozenset({"section"})

# Process-local counter for tests (prepare_* / capture path only).
_structural_build_count = 0


class StructuralReuseError(ValueError):
    """
    Raised when structural-artifact reuse cannot proceed safely.
    """


def reset_structural_build_count() -> None:
    """
    Reset the process-local structural-build counter (tests only).
    """
    global _structural_build_count
    _structural_build_count = 0


def get_structural_build_count() -> int:
    """
    Return how many structural artifacts were built in this process.

    Returns
    -------
    int
        Capture / prepare count since the last reset.
    """
    return int(_structural_build_count)


def _bump_structural_build_count() -> None:
    global _structural_build_count
    _structural_build_count += 1


def finish_variable_names(var_names: Sequence[str]) -> tuple[str, ...]:
    """
    Return the last two catalogue variable names (floor-finish group).

    Parameters
    ----------
    var_names : sequence of str
        Index-space variable names in product order.

    Returns
    -------
    tuple of str
        Finish variable names.

    Raises
    ------
    StructuralReuseError
        If fewer than two variables are present.
    """
    if len(var_names) < 2:
        raise StructuralReuseError(
            "finish-group reuse requires at least two variables, "
            f"got {len(var_names)}"
        )
    return (str(var_names[-2]), str(var_names[-1]))


def _canonical_param_value(value: Any) -> Any:
    """
    Convert a parameter value into a hashable structural-key component.
    """
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_param_value(v) for v in value)
    return str(value)


def make_structural_key(
    params: Mapping[str, Any],
    *,
    finish_names: Sequence[str] = (),
    snapshot_hash: str,
    n: float,
    run_config: AnalysisRunConfig,
    structural_names: Sequence[str] | None = None,
) -> tuple[Any, ...]:
    """
    Build the immutable key identifying one structural design state.

    When ``structural_names`` is provided (library / catalogue path), only those
    parameters are hashed — fixed non-structural params such as ``geom_span_mm``
    or ``loads_category`` are excluded. When omitted, all non-finish parameters
    are included (same inclusion set derived from ``finish_names``).

    Axial force, design-code / invalid-utilization settings, reference
    positions, and the input snapshot hash are always included.
    """
    if structural_names is not None:
        names = [str(name) for name in structural_names]
        missing = [name for name in names if name not in params]
        if missing:
            raise StructuralReuseError(
                "make_structural_key missing structural params: "
                + ", ".join(sorted(missing))
            )
        param_items = tuple(
            (name, _canonical_param_value(params[name]))
            for name in sorted(names)
        )
    else:
        finish = frozenset(str(name) for name in finish_names)
        param_items = tuple(
            (str(name), _canonical_param_value(params[name]))
            for name in sorted(params)
            if str(name) not in finish
        )
    return (
        ("snapshot_hash", str(snapshot_hash)),
        ("n", float(n)),
        ("design_code", str(run_config.design_code)),
        ("invalid_utilization", float(run_config.invalid_utilization)),
        ("slab_type", str(run_config.slab_type)),
        ("positions", tuple(float(x) for x in DEFLECTION_MK_CACHE_POSITIONS)),
        ("params", param_items),
    )


def normalize_structural_key(
    key: Sequence[Any],
    structural_names: Sequence[str],
) -> tuple[Any, ...]:
    """
    Filter a structural key's ``params`` entry down to ``structural_names``.

    Used so sealed artifacts that still embed fixed params (span, load
    category, …) compare equal to catalogue-only keys.
    """
    allowed = frozenset(str(name) for name in structural_names)
    items: list[Any] = []
    for entry in key:
        if (
            isinstance(entry, tuple)
            and len(entry) == 2
            and entry[0] == "params"
            and isinstance(entry[1], Sequence)
            and not isinstance(entry[1], (str, bytes))
        ):
            filtered = tuple(
                sorted(
                    (
                        (str(name), value)
                        for name, value in entry[1]
                        if str(name) in allowed
                    ),
                    key=lambda item: str(item[0]),
                )
            )
            items.append(("params", filtered))
        else:
            items.append(entry)
    return tuple(items)


def structural_keys_match(
    left: Sequence[Any],
    right: Sequence[Any],
    *,
    structural_names: Sequence[str],
) -> bool:
    """
    Return True when two keys match after structural-name normalization.
    """
    from core.ioh_core.structural_artifacts.codec import structural_key_digest

    return structural_key_digest(
        normalize_structural_key(left, structural_names)
    ) == structural_key_digest(normalize_structural_key(right, structural_names))


def _freeze_scalar(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (str(k), _freeze_scalar(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_scalar(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, bool)) or value is None:
        return value
    return str(value)


def _freeze_section_result(result: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """
    Snapshot numeric section-result fields (never BeamSection instances).
    """
    items: list[tuple[str, Any]] = []
    for key, value in sorted(result.items(), key=lambda item: str(item[0])):
        if key in _STRIP_RESULT_KEYS:
            continue
        items.append((str(key), _freeze_scalar(value)))
    return tuple(items)


def _thaw_section_result(payload: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload:
        if isinstance(value, tuple) and value and isinstance(value[0], tuple):
            # Nested mapping frozen as tuple of pairs.
            if all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                for item in value
            ):
                out[key] = {k: _thaw_nested(v) for k, v in value}
                continue
        out[key] = _thaw_nested(value)
    return out


def _thaw_nested(value: Any) -> Any:
    if isinstance(value, tuple):
        if value and all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {k: _thaw_nested(v) for k, v in value}
        return [_thaw_nested(v) for v in value]
    return value


def _freeze_mk_curve(curve: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    m_y = tuple(float(v) for v in np.asarray(curve.m_y, dtype=float).ravel())
    chi_y = tuple(float(v) for v in np.asarray(curve.chi_y, dtype=float).ravel())
    return m_y, chi_y


def _thaw_mk_curve(
    payload: tuple[tuple[float, ...], tuple[float, ...]],
) -> MomentCurvatureResults:
    m_y, chi_y = payload
    return MomentCurvatureResults(
        m_y=np.asarray(m_y, dtype=float),
        chi_y=np.asarray(chi_y, dtype=float),
    )


@dataclass(frozen=True)
class StructuralArtifact:
    """
    Immutable numeric structural snapshot for finish-group reuse.

    Attributes
    ----------
    key : tuple
        Structural identity from :func:`make_structural_key`.
    snapshot_hash : str
        Input-snapshot fingerprint embedded in ``key``.
    n : float
        Axial force [N] used when the caches were prepared.
    m_u : tuple
        Frozen ULS ``M_u`` entries ``(SectionResultKey, payload)``.
    m_cr : tuple
        Frozen SLS ``M_cr`` entries.
    mk_curves : tuple
        Frozen M–κ arrays keyed by :class:`MKCurveKey` and position.
    mk_metadata : tuple
        Frozen solver / coordination metadata per M–κ key.
    mk_cleared : bool
        ``True`` when M–κ preparation hit
        :class:`~core.analysis_core.section_methods.InvalidSectionForMKError`
        and curves were cleared (same contract as ordinary analysis).
    build_failure_result : dict or None
        Complete fail-closed analysis result when structural materials /
        construction failed before caches could be prepared. Fan-out uses
        this template for every finish in the group.
    """

    key: tuple[Any, ...]
    snapshot_hash: str
    n: float
    m_u: tuple[tuple[SectionResultKey, tuple[tuple[str, Any], ...]], ...]
    m_cr: tuple[tuple[SectionResultKey, tuple[tuple[str, Any], ...]], ...]
    mk_curves: tuple[
        tuple[
            MKCurveKey,
            tuple[tuple[float, tuple[tuple[float, ...], tuple[float, ...]]], ...],
        ],
        ...,
    ]
    mk_metadata: tuple[tuple[MKCurveKey, tuple[tuple[str, Any], ...]], ...]
    mk_cleared: bool
    build_failure_result: Mapping[str, Any] | None = None

    @property
    def is_build_failure(self) -> bool:
        """Whether this artifact records an expected structural build failure."""
        return self.build_failure_result is not None


def capture_structural_artifact(
    cache: AnalysisCache,
    *,
    key: tuple[Any, ...],
    snapshot_hash: str,
    n: float,
    mk_cleared: bool = False,
    build_failure_result: Mapping[str, Any] | None = None,
    count_build: bool = True,
) -> StructuralArtifact:
    """
    Snapshot an :class:`AnalysisCache` into an immutable artifact.

    Parameters
    ----------
    cache : AnalysisCache
        Prepared analysis cache (sections are ignored / never stored).
    key : tuple
        Structural key.
    snapshot_hash : str
        Input snapshot fingerprint.
    n : float
        Axial force [N].
    mk_cleared : bool, optional
        Whether M–κ curves were cleared after an expected invalid section.
    build_failure_result : mapping or None, optional
        Full fail-closed result when materials/construction failed.
    count_build : bool, optional
        When ``True``, increment the process-local build counter.

    Returns
    -------
    StructuralArtifact
        Detached numeric record.
    """
    if count_build:
        _bump_structural_build_count()
    if build_failure_result is not None:
        return StructuralArtifact(
            key=key,
            snapshot_hash=str(snapshot_hash),
            n=float(n),
            m_u=(),
            m_cr=(),
            mk_curves=(),
            mk_metadata=(),
            mk_cleared=True,
            build_failure_result=dict(build_failure_result),
        )

    m_u = tuple(
        (section_key, _freeze_section_result(result))
        for section_key, result in sorted(
            cache.m_u.items(),
            key=lambda item: (
                item[0].result_type,
                item[0].x,
                item[0].limit_state,
                item[0].n,
                item[0].flipped,
            ),
        )
    )
    m_cr = tuple(
        (section_key, _freeze_section_result(result))
        for section_key, result in sorted(
            cache.m_cr.items(),
            key=lambda item: (
                item[0].result_type,
                item[0].x,
                item[0].limit_state,
                item[0].n,
                item[0].flipped,
            ),
        )
    )
    mk_curves = tuple(
        (
            mk_key,
            tuple(
                (float(x), _freeze_mk_curve(curve))
                for x, curve in sorted(curves.items(), key=lambda item: item[0])
            ),
        )
        for mk_key, curves in sorted(
            cache.mk_curves.items(),
            key=lambda item: (
                item[0].limit_state,
                item[0].constitutive_law,
                item[0].method,
                item[0].n,
                item[0].m_k_num_points,
            ),
        )
    )
    mk_metadata = tuple(
        (mk_key, _freeze_section_result(meta))
        for mk_key, meta in sorted(
            cache.mk_metadata.items(),
            key=lambda item: (
                item[0].limit_state,
                item[0].constitutive_law,
                item[0].method,
                item[0].n,
                item[0].m_k_num_points,
            ),
        )
    )
    return StructuralArtifact(
        key=key,
        snapshot_hash=str(snapshot_hash),
        n=float(n),
        m_u=m_u,
        m_cr=m_cr,
        mk_curves=mk_curves,
        mk_metadata=mk_metadata,
        mk_cleared=bool(mk_cleared),
        build_failure_result=None,
    )


def apply_structural_artifact(
    cache: AnalysisCache,
    artifact: StructuralArtifact,
) -> None:
    """
    Hydrate ``cache`` from an immutable artifact (no BeamSection objects).

    Parameters
    ----------
    cache : AnalysisCache
        Destination cache (cleared then filled).
    artifact : StructuralArtifact
        Numeric snapshot.

    Raises
    ------
    StructuralReuseError
        If the artifact is a build-failure record (caller should fan-out
        the failure result instead of hydrating).
    """
    if artifact.is_build_failure:
        raise StructuralReuseError(
            "cannot hydrate analysis cache from a structural build-failure artifact"
        )
    cache.clear()
    for section_key, payload in artifact.m_u:
        cache.m_u[section_key] = _thaw_section_result(payload)
    for section_key, payload in artifact.m_cr:
        cache.m_cr[section_key] = _thaw_section_result(payload)
    for mk_key, curves in artifact.mk_curves:
        cache.mk_curves[mk_key] = {
            float(x): _thaw_mk_curve(payload) for x, payload in curves
        }
    for mk_key, payload in artifact.mk_metadata:
        cache.mk_metadata[mk_key] = _thaw_section_result(payload)


def fanout_build_failure(
    artifact: StructuralArtifact,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Return a complete fail-closed result for ``params`` from a cached failure.
    """
    if artifact.build_failure_result is None:
        raise StructuralReuseError("artifact has no build_failure_result")
    result = dict(artifact.build_failure_result)
    result["params"] = dict(params)
    # Keep nested analysis_error intact; only params change per finish.
    return result


def scientific_values_equal(left: Any, right: Any, *, rtol: float = 1e-12) -> bool:
    """
    Compare scientific result values with NaN/Inf-aware float equality.

    Parameters
    ----------
    left, right : Any
        Nested result fragments.
    rtol : float, optional
        Relative tolerance for finite floats.

    Returns
    -------
    bool
        ``True`` when values match under the contract used by reuse tests.
    """
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(
            scientific_values_equal(left[k], right[k], rtol=rtol) for k in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(
            scientific_values_equal(a, b, rtol=rtol) for a, b in zip(left, right)
        )
    if isinstance(left, (float, int, np.floating, np.integer)) and isinstance(
        right, (float, int, np.floating, np.integer)
    ):
        lf = float(left)
        rf = float(right)
        if math.isnan(lf) and math.isnan(rf):
            return True
        if math.isinf(lf) or math.isinf(rf):
            return lf == rf
        return math.isclose(lf, rf, rel_tol=rtol, abs_tol=0.0)
    return left == right
