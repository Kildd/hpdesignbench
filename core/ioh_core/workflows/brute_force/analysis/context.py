"""
Open a sealed brute-force experiment for read-only dataset analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.analysis_core.run_config import (
    AnalysisRunConfig,
    analysis_run_config_from_params,
)
from core.execution_core.atomic_io import read_json
from core.execution_core.lifecycle import experiment_paths, load_prepared_config
from core.execution_core.manifest import load_manifest
from core.ioh_core.workflows.brute_force.analysis.models import (
    BruteForceDatasetContext,
)
from core.ioh_core.workflows.brute_force.models import BruteForceShardSpec
from core.ioh_core.workflows.brute_force.planner import (
    BRUTE_FORCE_TASK_KIND,
    finish_group_size,
    load_problem_space_and_schema_from_csv_dir,
    resolve_enumeration_bounds,
)
from core.ioh_core.workflows.brute_force.schema_io import schema_hash_from_document
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
    validate_snapshot_hashes,
)


class BruteForceAnalysisError(ValueError):
    """
    Raised when a sealed brute-force experiment cannot be opened for analysis.
    """


def _fixed_params_from_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """
    Decode fixed-role parameter values from a sealed problem model.
    """
    fixed: dict[str, Any] = {}
    for name, meta in model.items():
        if meta.get("role") != "fixed":
            continue
        values = meta.get("values", [])
        idx = int(meta.get("fixed_idx", 0))
        if not values:
            raise BruteForceAnalysisError(
                f"fixed parameter {name!r} has an empty catalogue"
            )
        if idx < 0 or idx >= len(values):
            raise BruteForceAnalysisError(
                f"fixed parameter {name!r} fixed_idx={idx} out of range "
                f"for catalogue length {len(values)}"
            )
        fixed[str(name)] = values[idx]
    return fixed


def _require_float_param(params: Mapping[str, Any], name: str) -> float:
    if name not in params:
        raise BruteForceAnalysisError(
            f"required fixed parameter {name!r} missing from sealed snapshot"
        )
    try:
        value = float(params[name])
    except (TypeError, ValueError) as exc:
        raise BruteForceAnalysisError(
            f"fixed parameter {name!r} must be float-compatible, "
            f"got {params[name]!r}"
        ) from exc
    return value


def _planned_shard_specs(manifest_tasks: tuple[Any, ...]) -> tuple[BruteForceShardSpec, ...]:
    specs: list[BruteForceShardSpec] = []
    for task in manifest_tasks:
        if task.kind != BRUTE_FORCE_TASK_KIND:
            raise BruteForceAnalysisError(
                f"task {task.task_index} has unexpected kind {task.kind!r}"
            )
        specs.append(BruteForceShardSpec.from_payload(task.payload))
    return tuple(specs)


def open_brute_force_dataset(experiment_dir: Path) -> BruteForceDatasetContext:
    """
    Open a published brute-force experiment as a sealed analysis context.

    The opener is read-only: it never creates ``aggregate/analysis`` or any
    report artifact.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root that must already contain
        ``dataset/_SUCCESS``.

    Returns
    -------
    BruteForceDatasetContext
        Immutable sealed-dataset context.

    Raises
    ------
    BruteForceAnalysisError
        If the experiment is missing, wrong workflow, unpublished, or
        inconsistent with its sealed snapshot/schema.
    """
    paths = experiment_paths(experiment_dir)
    if not paths.source_config.is_file():
        raise BruteForceAnalysisError(
            f"missing source-config.toml under {paths.root}"
        )
    if not paths.manifest.is_file():
        raise BruteForceAnalysisError(f"missing manifest.json under {paths.root}")

    try:
        config = load_prepared_config(paths.root)
        manifest = load_manifest(paths.manifest)
    except Exception as exc:  # noqa: BLE001 - seal surface for analysis callers
        raise BruteForceAnalysisError(
            f"could not load sealed config/manifest: {exc}"
        ) from exc

    if config.experiment.workflow != "brute_force":
        raise BruteForceAnalysisError(
            "open_brute_force_dataset requires workflow='brute_force', "
            f"got {config.experiment.workflow!r}"
        )
    if manifest.workflow_type != "brute_force":
        raise BruteForceAnalysisError(
            "manifest.workflow_type must be 'brute_force', "
            f"got {manifest.workflow_type!r}"
        )
    if config.brute_force is None:
        raise BruteForceAnalysisError("prepared config is missing [brute_force]")
    if len(config.experiment.problem_ids) != 1:
        raise BruteForceAnalysisError(
            "brute_force analysis currently supports exactly one problem_id, "
            f"got {list(config.experiment.problem_ids)!r}"
        )

    try:
        validate_snapshot_hashes(
            inputs_dir=paths.inputs,
            expected_hashes=manifest.input_files,
        )
    except Exception as exc:  # noqa: BLE001
        raise BruteForceAnalysisError(
            f"snapshot hash validation failed: {exc}"
        ) from exc

    problem_id = config.experiment.problem_ids[0]
    slab_type = config.experiment.slab_type
    try:
        snapshot_dir = resolve_snapshot_slab_dir(paths.inputs, slab_type)
    except Exception as exc:  # noqa: BLE001
        raise BruteForceAnalysisError(
            f"could not resolve sealed snapshot directory: {exc}"
        ) from exc

    dataset_dir = paths.root / "dataset"
    success_path = dataset_dir / "_SUCCESS"
    if not success_path.is_file():
        raise BruteForceAnalysisError(
            f"dataset is not published; missing {success_path}"
        )

    schema_path = dataset_dir / "schema.json"
    if not schema_path.is_file():
        raise BruteForceAnalysisError(f"missing sealed schema.json: {schema_path}")
    try:
        schema_doc = read_json(schema_path)
        sealed_hash = schema_hash_from_document(schema_doc)
    except Exception as exc:  # noqa: BLE001
        raise BruteForceAnalysisError(
            f"could not read sealed schema.json: {exc}"
        ) from exc

    try:
        space, schema_info, problem_info = load_problem_space_and_schema_from_csv_dir(
            slab_dir=snapshot_dir,
            problem_id=problem_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise BruteForceAnalysisError(
            f"could not rebuild snapshot schema/space: {exc}"
        ) from exc

    if schema_info.schema_hash != sealed_hash:
        raise BruteForceAnalysisError(
            "schema.json hash does not match rebuilt snapshot schema: "
            f"{sealed_hash!r} != {schema_info.schema_hash!r}"
        )

    fixed_params = _fixed_params_from_model(problem_info["model"])
    try:
        run_config: AnalysisRunConfig = analysis_run_config_from_params(
            fixed_params,
            slab_type=slab_type,
        )
    except Exception as exc:  # noqa: BLE001
        raise BruteForceAnalysisError(
            f"could not derive AnalysisRunConfig from sealed fixed params: {exc}"
        ) from exc
    opt_penalty_alpha = _require_float_param(fixed_params, "opt_penalty_alpha")
    opt_penalty_beta = _require_float_param(fixed_params, "opt_penalty_beta")

    shard_specs = _planned_shard_specs(manifest.tasks)
    if not shard_specs:
        raise BruteForceAnalysisError("manifest contains no brute-force shard tasks")

    group_size = finish_group_size(space)
    planned_start, planned_stop = resolve_enumeration_bounds(
        start_index=config.brute_force.start_index,
        max_rows=config.brute_force.max_rows,
        group_size=group_size,
        cardinality=space.cardinality,
    )
    planned_rows = sum(spec.stop_index - spec.start_index for spec in shard_specs)
    expected_planned = planned_stop - planned_start
    if planned_rows != expected_planned:
        raise BruteForceAnalysisError(
            f"manifest planned rows {planned_rows} != "
            f"config enumeration range [{planned_start}, {planned_stop}) "
            f"({expected_planned} rows)"
        )
    if shard_specs[0].start_index != planned_start:
        raise BruteForceAnalysisError(
            f"first shard start {shard_specs[0].start_index} != "
            f"planned start {planned_start}"
        )
    if shard_specs[-1].stop_index != planned_stop:
        raise BruteForceAnalysisError(
            f"last shard stop {shard_specs[-1].stop_index} != "
            f"planned stop {planned_stop}"
        )

    full_catalogue_rows = int(space.cardinality)
    is_full_enumeration = (
        planned_start == 0
        and planned_stop == full_catalogue_rows
        and config.brute_force.start_index == 0
        and config.brute_force.max_rows is None
    )
    coverage_fraction = (
        float(planned_rows) / float(full_catalogue_rows)
        if full_catalogue_rows > 0
        else 0.0
    )

    summary_path = paths.aggregate / "summary.json"
    collect_summary: dict[str, Any] | None = None
    if summary_path.is_file():
        try:
            loaded = read_json(summary_path)
        except Exception as exc:  # noqa: BLE001
            raise BruteForceAnalysisError(
                f"could not read aggregate/summary.json: {exc}"
            ) from exc
        if not isinstance(loaded, Mapping):
            raise BruteForceAnalysisError(
                "aggregate/summary.json must be a JSON object"
            )
        collect_summary = dict(loaded)

    parts_glob = str((dataset_dir / "parts" / "shard-*.parquet").as_posix())

    return BruteForceDatasetContext(
        experiment_dir=paths.root,
        paths=paths,
        config=config,
        manifest=manifest,
        dataset_dir=dataset_dir,
        parts_glob=parts_glob,
        schema_doc=schema_doc,
        schema_info=schema_info,
        index_space=space,
        problem_info=problem_info,
        run_config=run_config,
        fixed_params=fixed_params,
        opt_penalty_alpha=opt_penalty_alpha,
        opt_penalty_beta=opt_penalty_beta,
        problem_id=str(problem_id),
        slab_type=str(slab_type),
        var_names=tuple(schema_info.var_names),
        param_names=tuple(schema_info.param_names),
        constraint_names=tuple(schema_info.constraint_names),
        shard_specs=shard_specs,
        full_catalogue_rows=full_catalogue_rows,
        planned_start_index=planned_start,
        planned_stop_index=planned_stop,
        planned_rows=planned_rows,
        is_full_enumeration=is_full_enumeration,
        coverage_fraction=coverage_fraction,
        collect_summary=collect_summary,
    )
