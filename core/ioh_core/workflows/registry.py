"""
Workflow registry for experiment planning, workers, and aggregation.

``execution_core.lifecycle`` looks up workflow entries here so the generic
runtime never imports slab/IOH helpers directly beyond this boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from core.execution_core.config import ExperimentConfig
from core.execution_core.models import TaskRecord
from core.execution_core.manifest import worker_module_for


class RegistryError(KeyError):
    """
    Raised when a workflow name is unknown to the registry.
    """


@dataclass(frozen=True)
class WorkflowEntry:
    """
    One registered experiment workflow.

    Attributes
    ----------
    name : str
        Workflow selector matching TOML ``experiment.workflow``.
    planner : callable
        ``(config) -> tuple[TaskRecord, ...]``.
    worker_module : str
        Import path launched as ``python -m <worker_module>``.
    aggregator : callable
        ``(experiment_dir) -> dict``.
    snapshot : callable
        ``(config, experiment_dir) -> Mapping[str, str]`` of relative input
        hashes.
    validator : callable or None
        Optional ``(config, tasks) -> None`` hook.
    """

    name: str
    planner: Callable[[ExperimentConfig], tuple[TaskRecord, ...]]
    worker_module: str
    aggregator: Callable[[Path], dict[str, Any]]
    snapshot: Callable[[ExperimentConfig, Path], Mapping[str, str]]
    validator: Callable[[ExperimentConfig, tuple[TaskRecord, ...]], None] | None = None


def _snapshot_optimization(config: ExperimentConfig, experiment_dir: Path) -> Mapping[str, str]:
    """
    Snapshot slab CSV inputs for an optimisation experiment.

    Parameters
    ----------
    config : ExperimentConfig
        Validated configuration.
    experiment_dir : Path
        Experiment root (``inputs/`` is created underneath).

    Returns
    -------
    Mapping[str, str]
        Relative path → SHA-256 digest map.
    """
    from core.ioh_core.workflows.optimization_runs.snapshot import snapshot_slab_inputs

    inputs_dir = Path(experiment_dir) / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_slab_inputs(
        slab_type=config.experiment.slab_type,
        inputs_dir=inputs_dir,
    )


def _plan_optimization(config: ExperimentConfig) -> tuple[TaskRecord, ...]:
    """
    Plan optimisation tasks from a validated config.

    Parameters
    ----------
    config : ExperimentConfig
        Validated configuration.

    Returns
    -------
    tuple of TaskRecord
        Contiguous optimisation tasks.
    """
    from core.ioh_core.workflows.optimization_runs.planner import plan_optimization_tasks

    return plan_optimization_tasks(config)


def _aggregate_optimization(experiment_dir: Path) -> dict[str, Any]:
    """
    Aggregate optimisation outcomes.

    Parameters
    ----------
    experiment_dir : Path
        Prepared experiment root.

    Returns
    -------
    dict
        Aggregate summary document.
    """
    from core.ioh_core.workflows.optimization_runs.aggregation import (
        aggregate_optimization_experiment,
    )

    return aggregate_optimization_experiment(experiment_dir)


def _validate_optimization(
    config: ExperimentConfig,
    tasks: tuple[TaskRecord, ...],
) -> None:
    """
    Reject unknown algorithms/parameters before the manifest is published.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment configuration.
    tasks : tuple of TaskRecord
        Planned optimisation tasks.

    Raises
    ------
    AlgorithmFactoryError
        If an algorithm name/parameter/seed combination is invalid, or if
        ``parameters`` tries to override ``max_evaluations``.
    """
    from core.ioh_core.algorithm_factory import (
        AlgorithmFactoryError,
        validate_algorithm_parameters,
    )
    from core.ioh_core.algorithms.registry import (
        RegistryError,
        normalize_algorithm_parameters,
    )
    from core.ioh_core.workflows.optimization_runs.models import OptimizationTaskSpec

    for algorithm in config.experiment.algorithms:
        if "max_evaluations" in algorithm.parameters:
            raise AlgorithmFactoryError(
                f"algorithm {algorithm.name!r}: parameters must not include "
                "'max_evaluations' (set it on the algorithm entry)"
            )
        declared = {
            "max_evaluations": int(algorithm.max_evaluations),
            **dict(algorithm.parameters),
        }
        try:
            normalize_algorithm_parameters(algorithm.name, declared)
        except RegistryError as exc:
            raise AlgorithmFactoryError(str(exc)) from exc

    for task in tasks:
        spec = OptimizationTaskSpec.from_payload(task.payload)
        validate_algorithm_parameters(
            spec.algorithm_name,
            spec.algorithm_parameters,
            random_seed=spec.random_seed,
        )
        if "max_evaluations" not in spec.algorithm_parameters:
            raise AlgorithmFactoryError(
                f"task {task.task_id!r}: algorithm_parameters must include "
                "'max_evaluations'"
            )
        forbidden = (
            "termination_mode",
            "global_optimum",
            "allowed_deviation_percent",
            "target_reference",
            "target_threshold",
        )
        leaked = [key for key in forbidden if key in spec.algorithm_parameters]
        if leaked:
            raise AlgorithmFactoryError(
                f"task {task.task_id!r}: algorithm_parameters must not include "
                f"termination fields {leaked}"
            )

        term = config.experiment.termination
        if spec.termination_mode != term.mode:
            raise AlgorithmFactoryError(
                f"task {task.task_id!r}: termination_mode {spec.termination_mode!r} "
                f"does not match config mode {term.mode!r}"
            )
        expected_optimum = term.global_optimum_for(spec.problem_id)
        if spec.global_optimum != expected_optimum:
            raise AlgorithmFactoryError(
                f"task {task.task_id!r}: global_optimum {spec.global_optimum!r} "
                f"does not match config {expected_optimum!r} for "
                f"problem_id {spec.problem_id!r}"
            )
        if spec.allowed_deviation_percent != term.allowed_deviation_percent:
            raise AlgorithmFactoryError(
                f"task {task.task_id!r}: allowed_deviation_percent "
                f"{spec.allowed_deviation_percent!r} does not match config "
                f"{term.allowed_deviation_percent!r}"
            )
        if spec.target_reference != term.target_reference:
            raise AlgorithmFactoryError(
                f"task {task.task_id!r}: target_reference "
                f"{spec.target_reference!r} does not match config "
                f"{term.target_reference!r}"
            )
        expected_threshold = term.target_threshold_for(spec.problem_id)
        if spec.target_threshold != expected_threshold:
            raise AlgorithmFactoryError(
                f"task {task.task_id!r}: target_threshold "
                f"{spec.target_threshold!r} does not match config "
                f"{expected_threshold!r} for problem_id {spec.problem_id!r}"
            )


def _snapshot_brute_force(
    config: ExperimentConfig, experiment_dir: Path
) -> Mapping[str, str]:
    """
    Snapshot slab CSVs and seal ``dataset/schema.json`` for brute-force.
    """
    from core.ioh_core.workflows.brute_force.planner import load_index_space_and_schema
    from core.ioh_core.workflows.brute_force.schema_io import write_dataset_schema_json
    from core.ioh_core.workflows.optimization_runs.snapshot import snapshot_slab_inputs

    inputs_dir = Path(experiment_dir) / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    hashes = snapshot_slab_inputs(
        slab_type=config.experiment.slab_type,
        inputs_dir=inputs_dir,
    )
    problem_id = config.experiment.problem_ids[0]
    _space, schema_info = load_index_space_and_schema(
        slab_type=config.experiment.slab_type,
        problem_id=problem_id,
    )
    write_dataset_schema_json(Path(experiment_dir) / "dataset", schema_info)
    return hashes


def _plan_brute_force(config: ExperimentConfig) -> tuple[TaskRecord, ...]:
    """
    Plan brute-force shard tasks from a validated config.
    """
    from core.ioh_core.workflows.brute_force.planner import plan_brute_force_tasks

    return plan_brute_force_tasks(config)


def _aggregate_brute_force(experiment_dir: Path) -> dict[str, Any]:
    """
    Validate, aggregate, and publish a brute-force experiment dataset.
    """
    from core.ioh_core.workflows.brute_force.aggregation import (
        aggregate_brute_force_experiment,
    )

    return aggregate_brute_force_experiment(experiment_dir)


def _validate_brute_force(
    config: ExperimentConfig,
    tasks: tuple[TaskRecord, ...],
) -> None:
    """
    Run preflight and basic shard-payload checks before sealing the manifest.
    """
    from core.ioh_core.workflows.brute_force.models import BruteForceShardSpec
    from core.ioh_core.workflows.brute_force.planner import (
        BRUTE_FORCE_TASK_KIND,
        load_index_space_and_schema,
    )
    from core.ioh_core.workflows.brute_force.preflight import run_preflight

    if not tasks:
        raise ValueError("brute_force planning produced no tasks")
    problem_id = config.experiment.problem_ids[0]
    _space, schema_info = load_index_space_and_schema(
        slab_type=config.experiment.slab_type,
        problem_id=problem_id,
    )
    for task in tasks:
        if task.kind != BRUTE_FORCE_TASK_KIND:
            raise ValueError(
                f"brute_force task {task.task_index} has kind {task.kind!r}"
            )
        spec = BruteForceShardSpec.from_payload(task.payload)
        if spec.schema_hash != schema_info.schema_hash:
            raise ValueError(
                "planned shard schema_hash does not match sealed schema: "
                f"{spec.schema_hash!r} != {schema_info.schema_hash!r}"
            )
    # Timing probes are optional at prepare; keep raise_if_blocked semantics.
    run_preflight(config, raise_if_blocked=True)


def _snapshot_sensitivity(
    config: ExperimentConfig, experiment_dir: Path
) -> Mapping[str, str]:
    """
    Snapshot slab CSVs, seal the SALib A/B design, and write ``schema.json``.
    """
    from core.ioh_core.workflows.optimization_runs.snapshot import (
        resolve_snapshot_slab_dir,
        sha256_file,
        snapshot_slab_inputs,
    )
    from core.ioh_core.workflows.sensitivity_analysis.design import (
        seal_sensitivity_design,
    )
    from core.ioh_core.workflows.sensitivity_analysis.planner import (
        load_sensitivity_schema_from_packaged,
    )
    from core.ioh_core.workflows.sensitivity_analysis.schema import (
        write_dataset_schema_json,
    )

    if config.sensitivity_analysis is None:
        raise ValueError("missing [sensitivity_analysis] configuration")
    section = config.sensitivity_analysis
    problem_id = config.experiment.problem_ids[0]

    inputs_dir = Path(experiment_dir) / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    hashes = dict(
        snapshot_slab_inputs(
            slab_type=config.experiment.slab_type,
            inputs_dir=inputs_dir,
        )
    )
    snapshot_dir = resolve_snapshot_slab_dir(
        inputs_dir, config.experiment.slab_type
    )
    design_dir = inputs_dir / "sensitivity"
    seal_sensitivity_design(
        snapshot_dir=snapshot_dir,
        slab_type=config.experiment.slab_type,
        problem_id=problem_id,
        factor_names=section.factor_names,
        sample_size=section.sample_size,
        response_key=section.response_key,
        scramble=section.scramble,
        seed=section.seed,
        design_dir=design_dir,
        overwrite=True,
    )
    for name in (
        "A_unit.npy",
        "B_unit.npy",
        "A_index.npy",
        "B_index.npy",
        "design.json",
    ):
        hashes[f"sensitivity/{name}"] = sha256_file(design_dir / name)

    schema_info = load_sensitivity_schema_from_packaged(
        slab_type=config.experiment.slab_type,
        problem_id=problem_id,
        factor_names=section.factor_names,
    )
    write_dataset_schema_json(Path(experiment_dir) / "dataset", schema_info)
    return hashes


def _plan_sensitivity(config: ExperimentConfig) -> tuple[TaskRecord, ...]:
    """Plan sensitivity shard tasks from a validated config."""
    from core.ioh_core.workflows.sensitivity_analysis.planner import (
        plan_sensitivity_tasks,
    )

    return plan_sensitivity_tasks(config)


def _aggregate_sensitivity(experiment_dir: Path) -> dict[str, Any]:
    """Validate, analyse, and publish a sensitivity experiment."""
    from core.ioh_core.workflows.sensitivity_analysis.aggregation import (
        aggregate_sensitivity_experiment,
    )

    return aggregate_sensitivity_experiment(experiment_dir)


def _validate_sensitivity(
    config: ExperimentConfig,
    tasks: tuple[TaskRecord, ...],
) -> None:
    """
    Run preflight and shard-payload checks before sealing the manifest.
    """
    from core.ioh_core.workflows.sensitivity_analysis.models import (
        SensitivityShardSpec,
    )
    from core.ioh_core.workflows.sensitivity_analysis.planner import (
        SENSITIVITY_TASK_KIND,
        load_sensitivity_schema_from_packaged,
        run_sensitivity_preflight,
    )

    if config.sensitivity_analysis is None:
        raise ValueError("missing [sensitivity_analysis] configuration")
    if not tasks:
        raise ValueError("sensitivity_analysis planning produced no tasks")
    section = config.sensitivity_analysis
    problem_id = config.experiment.problem_ids[0]
    schema_info = load_sensitivity_schema_from_packaged(
        slab_type=config.experiment.slab_type,
        problem_id=problem_id,
        factor_names=section.factor_names,
    )
    factor_count = len(section.factor_names)
    for task in tasks:
        if task.kind != SENSITIVITY_TASK_KIND:
            raise ValueError(
                f"sensitivity task {task.task_index} has kind {task.kind!r}"
            )
        spec = SensitivityShardSpec.from_payload(task.payload)
        if spec.schema_hash != schema_info.schema_hash:
            raise ValueError(
                "planned shard schema_hash does not match sealed schema: "
                f"{spec.schema_hash!r} != {schema_info.schema_hash!r}"
            )
        if spec.factor_count != factor_count:
            raise ValueError(
                f"planned shard factor_count {spec.factor_count} != "
                f"{factor_count}"
            )
    run_sensitivity_preflight(
        section=section,
        factor_count=factor_count,
        var_count=len(schema_info.var_names),
    )


def _snapshot_structural_library(
    config: ExperimentConfig, experiment_dir: Path
) -> Mapping[str, str]:
    """
    Snapshot slab CSVs and create the empty library layout.
    """
    from core.ioh_core.workflows.optimization_runs.snapshot import snapshot_slab_inputs
    from core.ioh_core.workflows.structural_library.identity import require_hp_slab

    require_hp_slab(config.experiment.slab_type)
    inputs_dir = Path(experiment_dir) / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    hashes = snapshot_slab_inputs(
        slab_type=config.experiment.slab_type,
        inputs_dir=inputs_dir,
    )
    from core.ioh_core.workflows.structural_library.validation import (
        library_directory,
        warn_if_overwriting_consumer_library,
    )

    library_dir = library_directory(experiment_dir)
    warn_if_overwriting_consumer_library(library_dir)
    (library_dir / "parts").mkdir(parents=True, exist_ok=True)
    (library_dir / "aggregate").mkdir(parents=True, exist_ok=True)
    return hashes


def _plan_structural_library(config: ExperimentConfig) -> tuple[TaskRecord, ...]:
    from core.ioh_core.workflows.structural_library.planner import (
        plan_structural_library_tasks,
    )

    return plan_structural_library_tasks(config)


def _aggregate_structural_library(experiment_dir: Path) -> dict[str, Any]:
    from core.ioh_core.workflows.structural_library.aggregation import (
        aggregate_structural_library_experiment,
    )

    return aggregate_structural_library_experiment(experiment_dir)


def _validate_structural_library(
    config: ExperimentConfig,
    tasks: tuple[TaskRecord, ...],
) -> None:
    from core.ioh_core.workflows.structural_library.models import (
        StructuralLibraryShardSpec,
    )
    from core.ioh_core.workflows.structural_library.planner import (
        STRUCTURAL_LIBRARY_TASK_KIND,
    )
    from core.ioh_core.workflows.structural_library.preflight import run_preflight

    if not tasks:
        raise ValueError("structural_library planning produced no tasks")
    for task in tasks:
        if task.kind != STRUCTURAL_LIBRARY_TASK_KIND:
            raise ValueError(
                f"structural_library task {task.task_index} has kind {task.kind!r}"
            )
        StructuralLibraryShardSpec.from_payload(task.payload)
    run_preflight(config, raise_if_blocked=True)


_REGISTRY: dict[str, WorkflowEntry] = {
    "optimization_runs": WorkflowEntry(
        name="optimization_runs",
        planner=_plan_optimization,
        worker_module=worker_module_for("optimization_runs"),
        aggregator=_aggregate_optimization,
        snapshot=_snapshot_optimization,
        validator=_validate_optimization,
    ),
    "brute_force": WorkflowEntry(
        name="brute_force",
        planner=_plan_brute_force,
        worker_module=worker_module_for("brute_force"),
        aggregator=_aggregate_brute_force,
        snapshot=_snapshot_brute_force,
        validator=_validate_brute_force,
    ),
    "sensitivity_analysis": WorkflowEntry(
        name="sensitivity_analysis",
        planner=_plan_sensitivity,
        worker_module=worker_module_for("sensitivity_analysis"),
        aggregator=_aggregate_sensitivity,
        snapshot=_snapshot_sensitivity,
        validator=_validate_sensitivity,
    ),
    "structural_library": WorkflowEntry(
        name="structural_library",
        planner=_plan_structural_library,
        worker_module=worker_module_for("structural_library"),
        aggregator=_aggregate_structural_library,
        snapshot=_snapshot_structural_library,
        validator=_validate_structural_library,
    ),
}


def get_workflow(name: str) -> WorkflowEntry:
    """
    Return the registered workflow entry for ``name``.

    Parameters
    ----------
    name : str
        Workflow selector.

    Returns
    -------
    WorkflowEntry
        Registered entry.

    Raises
    ------
    RegistryError
        If ``name`` is unknown.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise RegistryError(f"unknown workflow {name!r}") from exc


def registered_workflows() -> tuple[str, ...]:
    """
    Return sorted registered workflow names.

    Returns
    -------
    tuple of str
        Workflow selectors.
    """
    return tuple(sorted(_REGISTRY))
