"""
SALib Saltelli unit designs and discrete catalogue mapping for C1.

Sampling is owned by ``SALib.sample.sobol``. This module extracts A/B under
the locked packing order ``A``, ``A_B^(0)`` … ``A_B^(k-1)``, ``B``, maps
unit coordinates onto inclusive catalogue index ranges, and seals durable
``.npy`` / ``design.json`` artifacts from a snapshotted problem.
"""
from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.execution_core.atomic_io import read_json, write_bytes_atomic, write_text_atomic
from core.ioh_core.problem_builder import build_problem_from_snapshot
from core.ioh_core.workflows.optimization_runs.snapshot import sha256_hex
from core.ioh_core.workflows.sensitivity_analysis.deps import import_salib
from core.ioh_core.workflows.sensitivity_analysis.models import (
    ANALYSIS_ENGINE,
    DESIGN_SCHEMA_VERSION,
    INDEX_MAPPING,
    PACKING_RULE,
    SAMPLING_ENGINE,
    HeldConstantVariable,
    SensitivityDesignDocument,
    SensitivityDesignError,
    SensitivityFactorRecord,
    expected_evaluation_count,
)


def _require_power_of_two_sample_size(sample_size: int) -> int:
    if not isinstance(sample_size, (int, np.integer)) or isinstance(
        sample_size, bool
    ):
        raise ValueError("sample_size must be an integer.")
    size = int(sample_size)
    if size < 2 or size & (size - 1) != 0:
        raise ValueError(
            f"sample_size must be a power of two and at least 2; got {size}."
        )
    return size


def _require_factor_count(factor_count: int) -> int:
    if not isinstance(factor_count, (int, np.integer)) or isinstance(
        factor_count, bool
    ):
        raise ValueError("factor_count must be an integer.")
    count = int(factor_count)
    if count < 1:
        raise ValueError(f"factor_count must be at least 1; got {count}.")
    return count


def _require_seed(seed: int) -> int:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    value = int(seed)
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(
            f"seed must be an unsigned 32-bit integer; got {value}."
        )
    return value


def make_unit_problem(factor_count: int) -> dict:
    """
    Build a SALib problem dict on the unit hypercube.

    Parameters
    ----------
    factor_count:
        Number of independent factors ``k``.

    Returns
    -------
    dict
        SALib problem with placeholder names and bounds ``[0, 1]``.
    """
    count = _require_factor_count(factor_count)
    return {
        "num_vars": count,
        "names": [f"x_{index + 1}" for index in range(count)],
        "bounds": [[0.0, 1.0] for _ in range(count)],
    }


def sample_saltelli_unit_matrix(
    *,
    sample_size: int,
    factor_count: int,
    scramble: bool,
    seed: int,
) -> NDArray[np.floating]:
    """
    Draw the packed Saltelli unit matrix via SALib.

    Parameters
    ----------
    sample_size:
        Power-of-two radial block count ``N``.
    factor_count:
        Number of factors ``k``.
    scramble:
        Whether SALib should scramble the Sobol sequence.
    seed:
        Unsigned 32-bit seed passed through to SALib.

    Returns
    -------
    ndarray
        Shape ``(N * (k + 2), k)`` in packing order
        ``A``, ``AB_0`` … ``AB_{k-1}``, ``B`` per block.
    """
    if not isinstance(scramble, bool):
        raise ValueError("scramble must be a bool.")

    size = _require_power_of_two_sample_size(sample_size)
    count = _require_factor_count(factor_count)
    seed_value = _require_seed(seed)

    salib = import_salib()
    from SALib.sample import sobol as salib_sobol_sample

    _ = salib  # dependency gate exercised before submodule use
    packed = salib_sobol_sample.sample(
        make_unit_problem(count),
        size,
        calc_second_order=False,
        scramble=scramble,
        seed=seed_value,
    )
    matrix = np.asarray(packed, dtype=float)
    expected_rows = size * (count + 2)
    if matrix.shape != (expected_rows, count):
        raise ValueError(
            "SALib Saltelli sample has unexpected shape "
            f"{matrix.shape}; expected ({expected_rows}, {count})."
        )
    return matrix


def extract_base_unit_matrices(
    packed_unit: ArrayLike,
    factor_count: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Extract ``A`` and ``B`` from a packed SALib Saltelli unit matrix.

    Parameters
    ----------
    packed_unit:
        Array of shape ``(N * (k + 2), k)``.
    factor_count:
        Expected number of columns ``k``.

    Returns
    -------
    tuple of ndarray
        ``(A, B)`` each with shape ``(N, k)``.
    """
    count = _require_factor_count(factor_count)
    matrix = np.asarray(packed_unit, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != count:
        raise ValueError(
            f"packed unit matrix must have shape (N*(k+2), {count}); "
            f"got {matrix.shape}."
        )
    step = count + 2
    if matrix.shape[0] % step != 0:
        raise ValueError(
            f"packed row count {matrix.shape[0]} is not divisible by "
            f"k+2={step}."
        )
    a_unit = np.asarray(matrix[0::step], dtype=float)
    b_unit = np.asarray(matrix[(step - 1)::step], dtype=float)
    if a_unit.shape != b_unit.shape:
        raise ValueError("Extracted A and B shapes do not match.")
    return a_unit, b_unit


def generate_base_unit_matrices(
    *,
    sample_size: int,
    factor_count: int,
    scramble: bool,
    seed: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Sample with SALib and return base unit matrices ``A`` and ``B``.

    Parameters
    ----------
    sample_size:
        Power-of-two radial block count ``N``.
    factor_count:
        Number of factors ``k``.
    scramble:
        Whether SALib should scramble the Sobol sequence.
    seed:
        Unsigned 32-bit seed passed through to SALib.

    Returns
    -------
    tuple of ndarray
        ``(A_unit, B_unit)`` each with shape ``(N, k)``.
    """
    packed = sample_saltelli_unit_matrix(
        sample_size=sample_size,
        factor_count=factor_count,
        scramble=scramble,
        seed=seed,
    )
    return extract_base_unit_matrices(packed, factor_count)


def unit_to_catalogue_indices(
    unit_matrix: ArrayLike,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> NDArray[np.integer]:
    """
    Map unit-cube coordinates to inclusive catalogue indices.

    Uses ``index = clip(lower + floor(u * level_count), lower, upper)`` per
    column, with ``level_count = upper - lower + 1``.

    Parameters
    ----------
    unit_matrix:
        Array of shape ``(n, k)`` with values in ``[0, 1]``.
    lower_bounds:
        Inclusive lower catalogue index per factor.
    upper_bounds:
        Inclusive upper catalogue index per factor.

    Returns
    -------
    ndarray
        Integer index matrix with the same shape as ``unit_matrix``.
    """
    matrix = np.asarray(unit_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(
            f"unit_matrix must be 2-D; got shape {matrix.shape}."
        )
    lowers = np.asarray(list(lower_bounds), dtype=int)
    uppers = np.asarray(list(upper_bounds), dtype=int)
    if lowers.shape != (matrix.shape[1],) or uppers.shape != (matrix.shape[1],):
        raise ValueError(
            "lower_bounds and upper_bounds must each have length equal to "
            f"the number of columns ({matrix.shape[1]})."
        )
    if np.any(uppers < lowers):
        raise ValueError(
            "Each upper bound must be greater than or equal to its lower bound."
        )
    if np.any(~np.isfinite(matrix)):
        raise ValueError("unit_matrix contains non-finite values.")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("unit_matrix values must lie in [0, 1].")

    level_counts = uppers - lowers + 1
    ticks = np.floor(matrix * level_counts.astype(float)).astype(int)
    ticks = np.maximum(ticks, 0)
    ticks = np.minimum(ticks, level_counts - 1)
    return lowers + ticks


def cross_sample_row(
    a_row: ArrayLike,
    b_row: ArrayLike,
    factor_index: int,
) -> NDArray[np.floating]:
    """
    Build one radial cross-sample row ``A_B^(i)``.

    Parameters
    ----------
    a_row:
        One row from ``A``.
    b_row:
        Matching row from ``B``.
    factor_index:
        Zero-based column replaced by the value from ``B``.

    Returns
    -------
    ndarray
        Copy of ``a_row`` with column ``factor_index`` taken from ``b_row``.
    """
    a_values = np.asarray(a_row, dtype=float).reshape(-1)
    b_values = np.asarray(b_row, dtype=float).reshape(-1)
    if a_values.shape != b_values.shape:
        raise ValueError(
            f"a_row and b_row must have the same shape; got "
            f"{a_values.shape} and {b_values.shape}."
        )
    if a_values.size < 1:
        raise ValueError("a_row must be non-empty.")
    if not isinstance(factor_index, (int, np.integer)) or isinstance(
        factor_index, bool
    ):
        raise ValueError("factor_index must be an integer.")
    index = int(factor_index)
    if index < 0 or index >= a_values.size:
        raise ValueError(
            f"factor_index {index} is out of range for length {a_values.size}."
        )
    result = a_values.copy()
    result[index] = b_values[index]
    return result


def expand_full_catalogue_indices(
    *,
    var_names: Sequence[str],
    factor_names: Sequence[str],
    factor_row: ArrayLike,
    held_constant_variables: Sequence[HeldConstantVariable],
) -> list[int]:
    """
    Expand a length-``k`` factor index row into the full problem variable vector.

    Unselected variables take their sealed held-constant ``fixed_index``.
    """
    names = [str(name) for name in var_names]
    factors = [str(name) for name in factor_names]
    row = np.asarray(factor_row).reshape(-1)
    if row.size != len(factors):
        raise SensitivityDesignError(
            f"factor_row length {row.size} != factor_names length {len(factors)}"
        )
    held = {item.name: int(item.fixed_index) for item in held_constant_variables}
    selected = dict(zip(factors, (int(value) for value in row)))
    full: list[int] = []
    for name in names:
        if name in selected:
            full.append(selected[name])
        elif name in held:
            full.append(held[name])
        else:
            raise SensitivityDesignError(
                f"variable {name!r} is neither a selected factor nor held constant"
            )
    return full


def resolve_factor_layout(
    bundle: Mapping[str, Any],
    factor_names: Sequence[str],
) -> tuple[tuple[SensitivityFactorRecord, ...], tuple[HeldConstantVariable, ...]]:
    """
    Validate selected factors against a snapshotted problem bundle.

    Parameters
    ----------
    bundle:
        Problem bundle from :func:`build_problem_from_snapshot`.
    factor_names:
        Ordered selected variable names.

    Returns
    -------
    tuple
        ``(factors, held_constant_variables)``.
    """
    model = bundle["model"]
    var_names = list(bundle["var_names"])
    lb_by_name = dict(zip(var_names, bundle["lb_idx"]))
    ub_by_name = dict(zip(var_names, bundle["ub_idx"]))

    selected = [str(name).strip() for name in factor_names]
    if not selected:
        raise SensitivityDesignError("factor_names must be non-empty")
    if any(not name for name in selected):
        raise SensitivityDesignError("factor_names contains an empty name")
    if len(set(selected)) != len(selected):
        raise SensitivityDesignError("factor_names must be unique")

    factors: list[SensitivityFactorRecord] = []
    for name in selected:
        if name not in model:
            raise SensitivityDesignError(
                f"factor {name!r} is not a parameter in the resolved problem"
            )
        meta = model[name]
        role = str(meta.get("role", "")).lower()
        if role != "var":
            raise SensitivityDesignError(
                f"factor {name!r} has role {role!r}; only role='var' is allowed"
            )
        if name not in lb_by_name:
            raise SensitivityDesignError(
                f"factor {name!r} is not present in the problem variable space"
            )
        lower = int(lb_by_name[name])
        upper = int(ub_by_name[name])
        values = list(meta["values"][lower : upper + 1])
        factors.append(
            SensitivityFactorRecord(
                name=name,
                lower_index=lower,
                upper_index=upper,
                state_count=upper - lower + 1,
                catalogue_values=tuple(values),
            )
        )

    selected_set = set(selected)
    held: list[HeldConstantVariable] = []
    for name in var_names:
        if name in selected_set:
            continue
        meta = model[name]
        fixed_index = int(meta["fixed_idx"])
        held.append(
            HeldConstantVariable(
                name=name,
                fixed_index=fixed_index,
                catalogue_value=meta["values"][fixed_index],
            )
        )
    return tuple(factors), tuple(held)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def write_npy_atomic(path: Path, array: np.ndarray) -> str:
    """
    Atomically write a NumPy ``.npy`` array and return its SHA-256 digest.
    """
    payload = _npy_bytes(array)
    write_bytes_atomic(path, payload)
    return sha256_hex(payload)


def load_npy(path: Path) -> np.ndarray:
    """Load a sealed ``.npy`` file with ``allow_pickle=False``."""
    return np.load(Path(path), allow_pickle=False)


def seal_sensitivity_design(
    *,
    snapshot_dir: Path,
    slab_type: str,
    problem_id: str,
    factor_names: Sequence[str],
    sample_size: int,
    response_key: str,
    scramble: bool,
    seed: int,
    design_dir: Path,
    overwrite: bool = False,
) -> SensitivityDesignDocument:
    """
    Resolve factors from a snapshot, sample with SALib, and seal design artifacts.

    Writes ``A_unit.npy``, ``B_unit.npy``, ``A_index.npy``, ``B_index.npy``,
    and ``design.json`` under ``design_dir``.
    """
    snapshot_dir = Path(snapshot_dir)
    design_dir = Path(design_dir)
    if response_key not in {"y", "y_p"}:
        raise SensitivityDesignError(
            f"response_key must be 'y' or 'y_p', got {response_key!r}"
        )

    paths = {
        name: design_dir / name
        for name in (
            "A_unit.npy",
            "B_unit.npy",
            "A_index.npy",
            "B_index.npy",
            "design.json",
        )
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise SensitivityDesignError(
            "sensitivity design artifacts already exist; pass overwrite=True "
            f"to replace: {[str(path.name) for path in existing]}"
        )

    bundle = build_problem_from_snapshot(
        slab_type=slab_type,
        problem_id=problem_id,
        snapshot_dir=snapshot_dir,
    )
    factors, held = resolve_factor_layout(bundle, factor_names)
    factor_count = len(factors)
    expected = expected_evaluation_count(int(sample_size), factor_count)

    a_unit, b_unit = generate_base_unit_matrices(
        sample_size=int(sample_size),
        factor_count=factor_count,
        scramble=bool(scramble),
        seed=int(seed),
    )
    lowers = [factor.lower_index for factor in factors]
    uppers = [factor.upper_index for factor in factors]
    a_index = unit_to_catalogue_indices(a_unit, lowers, uppers)
    b_index = unit_to_catalogue_indices(b_unit, lowers, uppers)

    salib = import_salib()
    design_dir.mkdir(parents=True, exist_ok=True)
    digests = {
        "A_unit.npy": write_npy_atomic(paths["A_unit.npy"], a_unit),
        "B_unit.npy": write_npy_atomic(paths["B_unit.npy"], b_unit),
        "A_index.npy": write_npy_atomic(paths["A_index.npy"], a_index),
        "B_index.npy": write_npy_atomic(paths["B_index.npy"], b_index),
    }

    document = SensitivityDesignDocument(
        design_schema_version=DESIGN_SCHEMA_VERSION,
        problem_id=str(problem_id),
        slab_type=str(slab_type),
        sample_size=int(sample_size),
        factor_count=factor_count,
        factor_names=tuple(factor.name for factor in factors),
        factors=factors,
        held_constant_variables=held,
        response_key=str(response_key),
        scramble=bool(scramble),
        seed=int(seed),
        calc_second_order=False,
        packing_rule=PACKING_RULE,
        index_mapping=INDEX_MAPPING,
        expected_evaluations=expected,
        sampling_engine=SAMPLING_ENGINE,
        analysis_engine=ANALYSIS_ENGINE,
        numpy_version=str(np.__version__),
        salib_version=str(getattr(salib, "__version__", "unknown")),
        array_sha256=digests,
    )
    write_text_atomic(paths["design.json"], document.to_json_text())
    return document


def load_sensitivity_design(design_dir: Path) -> SensitivityDesignDocument:
    """Load and validate sealed ``design.json`` under ``design_dir``."""
    path = Path(design_dir) / "design.json"
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise SensitivityDesignError("design.json must contain a JSON object")
    return SensitivityDesignDocument.from_json(payload)


def verify_sealed_arrays(
    design_dir: Path,
    document: SensitivityDesignDocument | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reload sealed arrays, check digests, and regenerate A/B from metadata.
    """
    design_dir = Path(design_dir)
    document = document or load_sensitivity_design(design_dir)

    arrays = {}
    for name in ("A_unit.npy", "B_unit.npy", "A_index.npy", "B_index.npy"):
        path = design_dir / name
        data = path.read_bytes()
        digest = sha256_hex(data)
        expected = document.array_sha256[name]
        if digest != expected:
            raise SensitivityDesignError(
                f"{name} digest mismatch: got {digest}, expected {expected}"
            )
        arrays[name] = np.load(io.BytesIO(data), allow_pickle=False)

    a_unit = np.asarray(arrays["A_unit.npy"], dtype=float)
    b_unit = np.asarray(arrays["B_unit.npy"], dtype=float)
    a_index = np.asarray(arrays["A_index.npy"])
    b_index = np.asarray(arrays["B_index.npy"])

    regenerated_a, regenerated_b = generate_base_unit_matrices(
        sample_size=document.sample_size,
        factor_count=document.factor_count,
        scramble=document.scramble,
        seed=document.seed,
    )
    if not np.array_equal(a_unit, regenerated_a) or not np.array_equal(
        b_unit, regenerated_b
    ):
        raise SensitivityDesignError(
            "sealed unit matrices do not match SALib regeneration from metadata"
        )

    lowers = [factor.lower_index for factor in document.factors]
    uppers = [factor.upper_index for factor in document.factors]
    expected_a_index = unit_to_catalogue_indices(a_unit, lowers, uppers)
    expected_b_index = unit_to_catalogue_indices(b_unit, lowers, uppers)
    if not np.array_equal(a_index, expected_a_index) or not np.array_equal(
        b_index, expected_b_index
    ):
        raise SensitivityDesignError(
            "sealed index matrices do not match the discrete catalogue map"
        )
    return a_unit, b_unit, a_index, b_index
