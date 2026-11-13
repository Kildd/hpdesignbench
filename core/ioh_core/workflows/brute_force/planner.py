"""
Plan brute-force shards as aligned half-open linear index ranges.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

from core.execution_core.config import ExperimentConfig
from core.execution_core.models import TaskRecord, format_task_id
from core.ioh_core.import_specs import (
    build_space,
    load_constraint_defaults,
    load_param_defaults,
    load_problems_combined,
)
from core.ioh_core.workflows.brute_force.models import (
    BruteForceShardSpec,
    IndexSpace,
    IndexSpaceError,
)
from core.ioh_core.workflows.optimization_runs.snapshot import (
    SNAPSHOT_CSV_NAMES,
    read_packaged_slab_csv_bytes,
)

BRUTE_FORCE_TASK_KIND = "brute_force_shard"


class BruteForcePlanError(ValueError):
    """
    Raised when a brute-force experiment cannot be planned.
    """


def finish_group_size(space: IndexSpace) -> int:
    """
    Return the product of the last two catalogue sizes (floor-finish group).

    Parameters
    ----------
    space : IndexSpace
        Candidate index space with at least two variables.

    Returns
    -------
    int
        Finish-group cardinality (``24`` for Problem 1).

    Raises
    ------
    BruteForcePlanError
        If the space has fewer than two variables.
    """
    if len(space.sizes) < 2:
        raise BruteForcePlanError(
            "finish-group alignment requires at least two variables, "
            f"got {len(space.sizes)}"
        )
    return int(space.sizes[-2] * space.sizes[-1])


def aligned_rows_per_shard(requested: int, group_size: int) -> int:
    """
    Round ``requested`` down to a positive multiple of ``group_size``.

    Parameters
    ----------
    requested : int
        Configured ``rows_per_shard``.
    group_size : int
        Finish-group size (>= 1).

    Returns
    -------
    int
        Aligned shard length (at least ``group_size``).

    Raises
    ------
    BruteForcePlanError
        If ``group_size`` is not positive.
    """
    if group_size < 1:
        raise BruteForcePlanError(f"group_size must be >= 1, got {group_size}")
    if requested < group_size:
        return group_size
    return (requested // group_size) * group_size


def aligned_enumeration_start(
    start_index: int, group_size: int, cardinality: int
) -> int:
    """
    Validate and return a finish-group-aligned inclusive start index.

    Parameters
    ----------
    start_index : int
        Requested inclusive linear start (>= 0).
    group_size : int
        Finish-group size.
    cardinality : int
        Full catalogue cardinality.

    Returns
    -------
    int
        ``start_index`` unchanged when valid.

    Raises
    ------
    BruteForcePlanError
        If ``start_index`` is negative, not group-aligned, or not strictly
        less than ``cardinality``.
    """
    if start_index < 0:
        raise BruteForcePlanError(f"start_index must be >= 0, got {start_index}")
    if group_size < 1:
        raise BruteForcePlanError(f"group_size must be >= 1, got {group_size}")
    if start_index % group_size != 0:
        raise BruteForcePlanError(
            f"start_index={start_index} must be a multiple of finish-group "
            f"size {group_size}"
        )
    if start_index >= cardinality:
        raise BruteForcePlanError(
            f"start_index={start_index} must be < cardinality {cardinality}"
        )
    return int(start_index)


def aligned_enumeration_stop(max_rows: int, group_size: int, cardinality: int) -> int:
    """
    Return the exclusive finish-group-aligned stop for a truncated run.

    Parameters
    ----------
    max_rows : int
        Requested exclusive upper bound (>= 1).
    group_size : int
        Finish-group size.
    cardinality : int
        Full catalogue cardinality.

    Returns
    -------
    int
        ``stop`` in ``[group_size, cardinality]`` aligned to ``group_size``.
    """
    if max_rows < 1:
        raise BruteForcePlanError(f"max_rows must be >= 1, got {max_rows}")
    if group_size < 1:
        raise BruteForcePlanError(f"group_size must be >= 1, got {group_size}")
    limited = min(int(max_rows), int(cardinality))
    stop = (limited // group_size) * group_size
    if stop < group_size:
        raise BruteForcePlanError(
            f"max_rows={max_rows} is smaller than finish-group size {group_size}"
        )
    return stop


def resolve_enumeration_bounds(
    *,
    start_index: int,
    max_rows: int | None,
    group_size: int,
    cardinality: int,
) -> tuple[int, int]:
    """
    Resolve the half-open planned range ``[start, stop)``.

    Parameters
    ----------
    start_index : int
        Inclusive absolute linear start (default ``0`` in config).
    max_rows : int or None
        Exclusive absolute linear stop before alignment, or ``None`` for
        catalogue end.
    group_size : int
        Finish-group size.
    cardinality : int
        Full catalogue cardinality.

    Returns
    -------
    tuple of int
        ``(start, stop)`` with ``0 <= start < stop <= cardinality``, both
        multiples of ``group_size``.
    """
    start = aligned_enumeration_start(start_index, group_size, cardinality)
    if max_rows is None:
        stop = int(cardinality)
    else:
        stop = aligned_enumeration_stop(max_rows, group_size, cardinality)
    if start >= stop:
        raise BruteForcePlanError(
            f"empty enumeration range [{start}, {stop}); "
            "require start_index < stop (from max_rows or cardinality)"
        )
    return start, stop


def provisional_schema_hash(space: IndexSpace) -> str:
    """
    Return a stable SHA-256 hex digest for the index-space identity.

    Until the Parquet schema step lands, shards fingerprint the catalogue
    bounds so payloads remain deterministic across preparation runs.

    Parameters
    ----------
    space : IndexSpace
        Candidate index space.

    Returns
    -------
    str
        Lowercase hex digest.
    """
    payload = {
        "var_names": list(space.var_names),
        "lower_bounds": list(space.lower_bounds),
        "upper_bounds": list(space.upper_bounds),
        "sizes": list(space.sizes),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_index_space(*, slab_type: str, problem_id: str) -> IndexSpace:
    """
    Build an :class:`IndexSpace` from packaged slab CSVs for one problem.

    Parameters
    ----------
    slab_type : str
        Slab package name such as ``\"hp_slab\"``.
    problem_id : str
        Ordinary problem ID.

    Returns
    -------
    IndexSpace
        Derived catalogue index space.

    Raises
    ------
    BruteForcePlanError
        If the problem or packaged CSVs cannot be loaded.
    """
    space, _schema = load_index_space_and_schema(
        slab_type=slab_type, problem_id=problem_id
    )
    return space


def load_problem_space_and_schema_from_csv_dir(
    *,
    slab_dir: Path,
    problem_id: str,
) -> tuple[IndexSpace, Any, dict[str, Any]]:
    """
    Build index space, Parquet schema, and problem info from a CSV directory.

    Parameters
    ----------
    slab_dir : Path
        Directory containing the four slab CSVs
        (``parameter_defaults.csv``, ``constraint_defaults.csv``,
        ``problem_list.csv``, ``materials.csv``).
    problem_id : str
        Ordinary problem ID.

    Returns
    -------
    tuple
        ``(IndexSpace, BruteForceSchema, problem_info)`` where
        ``problem_info`` is the per-problem dict from
        :func:`load_problems_combined`.

    Raises
    ------
    BruteForcePlanError
        If the problem or CSVs cannot be loaded.
    """
    from core.ioh_core.workflows.brute_force.schema import build_brute_force_schema

    slab_dir = Path(slab_dir)
    try:
        pdef = load_param_defaults(slab_dir / "parameter_defaults.csv")
        cdef = load_constraint_defaults(slab_dir / "constraint_defaults.csv")
        problems = load_problems_combined(
            pdef, cdef, slab_dir / "problem_list.csv"
        )
        if problem_id not in problems:
            available = ", ".join(sorted(problems)) or "(none)"
            raise BruteForcePlanError(
                f"unknown problem_id {problem_id!r}; available: {available}"
            )
        info = problems[problem_id]
        var_names, lb_idx, ub_idx = build_space(info["model"])
        space = IndexSpace.from_bounds(var_names, lb_idx, ub_idx)
        schema_info = build_brute_force_schema(
            space=space,
            model=info["model"],
            constraint_names=tuple(info["constraints"].keys()),
        )
        return space, schema_info, info
    except (OSError, KeyError, ValueError, IndexSpaceError, ImportError) as exc:
        if isinstance(exc, BruteForcePlanError):
            raise
        raise BruteForcePlanError(
            f"could not build index space/schema from {slab_dir}: {exc}"
        ) from exc


def load_index_space_and_schema(
    *,
    slab_type: str,
    problem_id: str,
):
    """
    Build the index space and Parquet schema from packaged slab CSVs.

    Parameters
    ----------
    slab_type : str
        Slab package name such as ``\"hp_slab\"``.
    problem_id : str
        Ordinary problem ID.

    Returns
    -------
    tuple
        ``(IndexSpace, BruteForceSchema)``.

    Raises
    ------
    BruteForcePlanError
        If the problem or packaged CSVs cannot be loaded.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="hdb_bf_space_") as tmp:
            root = Path(tmp)
            for name in SNAPSHOT_CSV_NAMES:
                (root / name).write_bytes(read_packaged_slab_csv_bytes(slab_type, name))
            space, schema_info, _info = load_problem_space_and_schema_from_csv_dir(
                slab_dir=root,
                problem_id=problem_id,
            )
            return space, schema_info
    except BruteForcePlanError:
        raise
    except (OSError, KeyError, ValueError, IndexSpaceError, ImportError) as exc:
        raise BruteForcePlanError(
            f"could not build index space/schema for slab_type={slab_type!r} "
            f"problem_id={problem_id!r}: {exc}"
        ) from exc


def plan_shard_specs(
    *,
    space: IndexSpace,
    problem_id: str,
    slab_type: str,
    rows_per_shard: int,
    schema_hash: str | None = None,
    start_index: int = 0,
    stop_index: int | None = None,
) -> tuple[BruteForceShardSpec, ...]:
    """
    Cover ``[start, stop)`` with finish-group-aligned shard ranges.

    Parameters
    ----------
    space : IndexSpace
        Candidate index space.
    problem_id : str
        Ordinary problem ID.
    slab_type : str
        Slab package name.
    rows_per_shard : int
        Requested shard length before alignment.
    schema_hash : str or None, optional
        Schema fingerprint; defaults to :func:`provisional_schema_hash`.
    start_index : int, optional
        Inclusive start of the planned range (default ``0``). Must be a
        multiple of the finish-group size and ``< stop``.
    stop_index : int or None, optional
        Exclusive end of the planned range. Defaults to ``space.cardinality``.
        Must be a positive multiple of the finish-group size and
        ``<= cardinality``.

    Returns
    -------
    tuple of BruteForceShardSpec
        Contiguous shards with no gaps or overlaps.

    Raises
    ------
    BruteForcePlanError
        If the space cannot be aligned to finish groups.
    """
    group = finish_group_size(space)
    if space.cardinality % group != 0:
        raise BruteForcePlanError(
            f"cardinality {space.cardinality} is not a multiple of finish-group "
            f"size {group}"
        )
    start = aligned_enumeration_start(
        int(start_index), group, space.cardinality
    )
    stop = space.cardinality if stop_index is None else int(stop_index)
    if stop < group or stop > space.cardinality or stop % group != 0:
        raise BruteForcePlanError(
            f"stop_index={stop} must be a multiple of finish-group size {group} "
            f"in [{group}, {space.cardinality}]"
        )
    if start >= stop:
        raise BruteForcePlanError(f"empty enumeration range [{start}, {stop})")
    step = aligned_rows_per_shard(int(rows_per_shard), group)
    digest = schema_hash or provisional_schema_hash(space)

    specs: list[BruteForceShardSpec] = []
    cursor = start
    shard_index = 0
    while cursor < stop:
        end = min(cursor + step, stop)
        specs.append(
            BruteForceShardSpec(
                problem_id=problem_id,
                slab_type=slab_type,
                shard_index=shard_index,
                start_index=cursor,
                stop_index=end,
                schema_hash=digest,
            )
        )
        cursor = end
        shard_index += 1
    return tuple(specs)


def specs_to_task_records(
    specs: Iterable[BruteForceShardSpec],
) -> tuple[TaskRecord, ...]:
    """
    Convert shard specs into contiguous generic task records.

    Parameters
    ----------
    specs : iterable of BruteForceShardSpec
        Ordered shard specifications.

    Returns
    -------
    tuple of TaskRecord
        Contiguous ``task_000000`` … records with kind
        ``brute_force_shard``.
    """
    records: list[TaskRecord] = []
    for index, spec in enumerate(specs):
        if spec.shard_index != index:
            raise BruteForcePlanError(
                f"shard_index mismatch: expected {index}, got {spec.shard_index}"
            )
        records.append(
            TaskRecord(
                task_index=index,
                task_id=format_task_id(index),
                kind=BRUTE_FORCE_TASK_KIND,
                payload=spec.to_payload(),
            )
        )
    return tuple(records)


def plan_brute_force_tasks(config: ExperimentConfig) -> tuple[TaskRecord, ...]:
    """
    Plan contiguous brute-force ``TaskRecord`` entries for a config.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment TOML configuration.

    Returns
    -------
    tuple of TaskRecord
        Manifest-ready shard task list.

    Raises
    ------
    BruteForcePlanError
        If the workflow/settings are invalid or the space cannot be planned.
    """
    if config.experiment.workflow != "brute_force":
        raise BruteForcePlanError(
            "plan_brute_force_tasks requires workflow='brute_force', "
            f"got {config.experiment.workflow!r}"
        )
    if config.brute_force is None:
        raise BruteForcePlanError("missing [brute_force] configuration")
    if len(config.experiment.problem_ids) != 1:
        raise BruteForcePlanError(
            "brute_force currently supports exactly one problem_id, "
            f"got {list(config.experiment.problem_ids)!r}"
        )

    problem_id = config.experiment.problem_ids[0]
    slab_type = config.experiment.slab_type
    space, schema_info = load_index_space_and_schema(
        slab_type=slab_type, problem_id=problem_id
    )
    start_index, stop_index = resolve_enumeration_bounds(
        start_index=config.brute_force.start_index,
        max_rows=config.brute_force.max_rows,
        group_size=finish_group_size(space),
        cardinality=space.cardinality,
    )
    specs = plan_shard_specs(
        space=space,
        problem_id=problem_id,
        slab_type=slab_type,
        rows_per_shard=config.brute_force.rows_per_shard,
        schema_hash=schema_info.schema_hash,
        start_index=start_index,
        stop_index=stop_index,
    )
    return specs_to_task_records(specs)
