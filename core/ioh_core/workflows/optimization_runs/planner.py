"""
Plan optimisation tasks from a validated experiment TOML configuration.

Expands ``problem_ids × algorithms × repetitions`` into contiguous generic
``TaskRecord`` entries with deterministic stochastic seeds.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Mapping

from core.execution_core.config import (
    AlgorithmConfig,
    ExperimentConfig,
    TerminationConfig,
)
from core.execution_core.models import TaskRecord, format_task_id
from core.ioh_core.algorithm_factory import AlgorithmFactoryError
from core.ioh_core.algorithms.registry import (
    RegistryError,
    SeedPolicy,
    get_algorithm_spec,
    normalize_algorithm_parameters,
)
from core.ioh_core.algorithms import warm_start as warm_start_tokens
from core.ioh_core.workflows.predesign_paths import (
    predesign_initial_design_deb_path,
    predesign_initial_design_path,
)
from core.ioh_core.workflows.optimization_runs.models import OptimizationTaskSpec

OPTIMIZATION_TASK_KIND = "optimization_run"


def termination_fields_for(
    termination: TerminationConfig,
    problem_id: str,
) -> dict:
    """
    Return task-spec termination kwargs from a validated config section.

    Parameters
    ----------
    termination : TerminationConfig
        Experiment-level termination settings.
    problem_id : str
        Problem whose scalar optimum and threshold are sealed into the task.

    Returns
    -------
    dict
        Keyword arguments for :class:`OptimizationTaskSpec`.
    """
    return {
        "termination_mode": termination.mode,
        "global_optimum": termination.global_optimum_for(problem_id),
        "allowed_deviation_percent": termination.allowed_deviation_percent,
        "target_reference": termination.target_reference,
        "target_threshold": termination.target_threshold_for(problem_id),
    }


def natural_sort_key(value: str) -> list:
    """
    Return a natural-sort key for problem IDs such as ``\"2\"`` before ``\"10\"``.

    Parameters
    ----------
    value : str
        Problem ID or other label.

    Returns
    -------
    list
        Alternating text/int key suitable for ``sorted(..., key=...)``.
    """
    parts = re.split(r"(\d+)", value)
    key: list = []
    for part in parts:
        if part == "":
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def derive_random_seed(
    *,
    seed_base: int,
    slab_type: str,
    problem_id: str,
    algorithm_name: str,
    repetition: int,
) -> int:
    """
    Derive a stable unsigned 32-bit seed via SHA-256.

    The preimage is:

    ``seed_base | slab_type | problem_id | algorithm_name | repetition``

    Parameters
    ----------
    seed_base : int
        Experiment-level seed base from TOML.
    slab_type : str
        Slab package name.
    problem_id : str
        Ordinary problem ID.
    algorithm_name : str
        Algorithm factory name.
    repetition : int
        Zero-based repetition index.

    Returns
    -------
    int
        Unsigned 32-bit integer in ``[0, 2**32 - 1]``.
    """
    material = (
        f"{seed_base}|{slab_type}|{problem_id}|{algorithm_name}|{repetition}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def algorithm_parameters_for(algorithm: AlgorithmConfig) -> dict:
    """
    Return the normalized parameter dict stored on each optimisation task.

    Merges the common ``max_evaluations`` budget with a copy of the user
    ``parameters`` mapping, then validates against the production registry.

    Parameters
    ----------
    algorithm : AlgorithmConfig
        Validated algorithm entry from the experiment TOML.

    Returns
    -------
    dict
        Ordinary JSON-compatible hyperparameters.

    Raises
    ------
    AlgorithmFactoryError
        If ``parameters`` overrides ``max_evaluations`` or fails registry
        validation.
    """
    if "max_evaluations" in algorithm.parameters:
        raise AlgorithmFactoryError(
            f"algorithm {algorithm.name!r}: parameters must not include "
            "'max_evaluations' (set it on the algorithm entry)"
        )
    if algorithm.name in warm_start_tokens.INITIAL_DESIGN_PATH_TOKENS:
        for forbidden in ("initial_design_path", "start_index"):
            if forbidden in algorithm.parameters:
                raise AlgorithmFactoryError(
                    f"algorithm {algorithm.name!r}: {forbidden} is set by "
                    "the planner; do not set it in TOML"
                )
    merged = {
        "max_evaluations": int(algorithm.max_evaluations),
        **dict(algorithm.parameters),
    }
    try:
        normalized = normalize_algorithm_parameters(algorithm.name, merged)
    except RegistryError as exc:
        raise AlgorithmFactoryError(str(exc)) from exc
    return dict(normalized)


def apply_warm_start_parameters(
    *,
    algorithm_name: str,
    parameters: Mapping[str, object],
    output_root: str,
    problem_id: str,
    repetition: int,
    repetitions_count: int,
    initial_design_path_tokens: frozenset[str] | None = None,
    rank1_seedless_tokens: frozenset[str] | None = None,
) -> dict:
    """
    Attach the warm-start parameter for one planned repetition.

    Names in ``initial_design_path_tokens`` receive the predesign path and
    do not receive ``start_index``. Warm starts that search on an
    unpenalized objective and separate constraints receive
    ``initial_design_deb.json``. Every other path token receives
    ``initial_design.json``. The file is not required to exist.

    Parameters
    ----------
    algorithm_name : str
        Registry token.
    parameters : mapping
        Normalized parameters without the repetition start index.
    output_root : str
        Experiment output root.
    problem_id : str
        Task problem id.
    repetition : int
        Zero-based repetition index.
    repetitions_count : int
        Configured repetition count.
    initial_design_path_tokens : frozenset of str or None, optional
        Tokens that receive a predesign path. Defaults to the module set.
        Members of ``DEB_WARM_START_TOKENS`` receive the Deb file.
    rank1_seedless_tokens : frozenset of str or None, optional
        Tokens that reject ``repetitions.count > 1``. Defaults to the module set.

    Returns
    -------
    dict
        Parameters for this task.

    Raises
    ------
    AlgorithmFactoryError
        If a path token sets its own start parameter, or a seedless token
        asks for more than one repetition.
    """
    path_tokens = (
        warm_start_tokens.INITIAL_DESIGN_PATH_TOKENS
        if initial_design_path_tokens is None
        else initial_design_path_tokens
    )
    seedless_tokens = (
        warm_start_tokens.RANK1_SEEDLESS_TOKENS
        if rank1_seedless_tokens is None
        else rank1_seedless_tokens
    )
    task_params = dict(parameters)
    if algorithm_name in path_tokens:
        for forbidden in ("initial_design_path", "start_index"):
            if forbidden in parameters:
                raise AlgorithmFactoryError(
                    f"algorithm {algorithm_name!r}: {forbidden} is set by "
                    "the planner; do not set it in TOML"
                )
        if algorithm_name in seedless_tokens and repetitions_count > 1:
            raise AlgorithmFactoryError(
                f"algorithm {algorithm_name!r}: repetitions.count "
                f"{repetitions_count} exceeds 1"
            )
        design_path = (
            predesign_initial_design_deb_path(output_root, problem_id)
            if algorithm_name in warm_start_tokens.DEB_WARM_START_TOKENS
            else predesign_initial_design_path(output_root, problem_id)
        )
        task_params["initial_design_path"] = str(design_path)
        return task_params
    return task_params


def resolve_seed_manifest(config: ExperimentConfig) -> dict[tuple[str, str, int], int] | None:
    """
    Load an optional per-task seed manifest.

    The key is ``(algorithm_name, problem_id, repetition)``. The algorithm
    name is the full registry token, which already names the start mode.
    A relative path is resolved from the TOML file's directory.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment configuration.

    Returns
    -------
    dict or None
        Explicit seeds, or ``None`` when the config does not set a manifest.

    Raises
    ------
    AlgorithmFactoryError
        If the file is missing, a row is duplicated, or a seed is not an
        unsigned 32-bit integer.
    """
    repetitions = config.experiment.repetitions
    if repetitions is None or repetitions.seed_manifest_path is None:
        return None
    raw = Path(repetitions.seed_manifest_path)
    if not raw.is_absolute():
        base = config.source_path.parent if config.source_path is not None else Path.cwd()
        raw = base / raw
    if not raw.is_file():
        raise AlgorithmFactoryError(f"seed manifest not found: {raw}")
    seeds: dict[tuple[str, str, int], int] = {}
    for line_number, line in enumerate(raw.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AlgorithmFactoryError(
                f"seed manifest {raw} line {line_number} is not JSON"
            ) from exc
        try:
            key = (str(row["algorithm_name"]), str(row["problem_id"]), int(row["repetition"]))
            seed = int(row["random_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AlgorithmFactoryError(
                f"seed manifest {raw} line {line_number} is missing a task field"
            ) from exc
        if isinstance(row["repetition"], bool) or isinstance(row["random_seed"], bool):
            raise AlgorithmFactoryError(
                f"seed manifest {raw} line {line_number} has a boolean field"
            )
        if seed < 0 or seed > 0xFFFFFFFF:
            raise AlgorithmFactoryError(
                f"seed manifest {raw} line {line_number} seed {seed} is outside uint32"
            )
        token = key[0]
        if "start" in row:
            warm = "WARM_START" in token
            if (str(row["start"]) == "Warm") != warm:
                raise AlgorithmFactoryError(
                    f"seed manifest {raw} line {line_number} start does not match {token}"
                )
        if key in seeds:
            raise AlgorithmFactoryError(
                f"seed manifest duplicates {key[0]} {key[1]} repetition {key[2]}"
            )
        seeds[key] = seed
    return seeds


def plan_optimization_specs(config: ExperimentConfig) -> tuple[OptimizationTaskSpec, ...]:
    """
    Expand a validated config into ordered optimisation task specs.

    Order:

    1. naturally sorted ``problem_ids``;
    2. algorithm order from the TOML array;
    3. ascending repetition indices ``0 .. count-1``.

    Seed policy and parameter defaults come from the algorithm registry.
    When ``repetitions.seed_manifest_path`` is set, each required seed is
    taken from that file instead of :func:`derive_random_seed`. The manifest
    must contain each planned seeded task once and no other task.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment TOML configuration.

    Returns
    -------
    tuple of OptimizationTaskSpec
        Ordered task specifications.

    Raises
    ------
    ValueError
        If the workflow is not ``optimization_runs``.
    AlgorithmFactoryError
        If an algorithm name/parameter combination is invalid.
    """
    if config.experiment.workflow != "optimization_runs":
        raise ValueError(
            "plan_optimization_specs requires workflow='optimization_runs', "
            f"got {config.experiment.workflow!r}"
        )

    problem_ids = sorted(config.experiment.problem_ids, key=natural_sort_key)
    algorithms = config.experiment.algorithms
    repetitions = config.experiment.repetitions
    slab_type = config.experiment.slab_type
    manifest = resolve_seed_manifest(config)
    used_manifest: set[tuple[str, str, int]] = set()
    prepared: list[tuple[AlgorithmConfig, SeedPolicy, dict, int]] = []
    for algorithm in algorithms:
        try:
            spec = get_algorithm_spec(algorithm.name)
        except RegistryError as exc:
            raise AlgorithmFactoryError(str(exc)) from exc
        params = algorithm_parameters_for(algorithm)
        repetitions_count = (
            repetitions.count
            if algorithm.repetitions is None
            else algorithm.repetitions
        )
        prepared.append(
            (algorithm, spec.seed_policy, params, repetitions_count)
        )

    specs: list[OptimizationTaskSpec] = []
    for problem_id in problem_ids:
        termination_kwargs = termination_fields_for(
            config.experiment.termination,
            problem_id,
        )
        for algorithm, seed_policy, params, repetitions_count in prepared:
            for repetition in range(repetitions_count):
                if seed_policy is SeedPolicy.REQUIRED:
                    key = (algorithm.name, problem_id, repetition)
                    if manifest is None:
                        seed: int | None = derive_random_seed(
                            seed_base=repetitions.seed_base,
                            slab_type=slab_type,
                            problem_id=problem_id,
                            algorithm_name=algorithm.name,
                            repetition=repetition,
                        )
                    elif key not in manifest:
                        raise AlgorithmFactoryError(
                            "seed manifest is missing "
                            f"{algorithm.name} {problem_id} repetition {repetition}"
                        )
                    else:
                        seed = manifest[key]
                        used_manifest.add(key)
                else:
                    seed = None
                    if manifest is not None and (algorithm.name, problem_id, repetition) in manifest:
                        raise AlgorithmFactoryError(
                            "seed manifest names an algorithm that does not take a seed: "
                            f"{algorithm.name}"
                        )
                task_params = apply_warm_start_parameters(
                    algorithm_name=algorithm.name,
                    parameters=params,
                    output_root=config.experiment.output_root,
                    problem_id=problem_id,
                    repetition=repetition,
                    repetitions_count=repetitions_count,
                )
                specs.append(
                    OptimizationTaskSpec(
                        problem_id=problem_id,
                        slab_type=slab_type,
                        algorithm_name=algorithm.name,
                        algorithm_parameters=task_params,
                        repetition=repetition,
                        random_seed=seed,
                        **termination_kwargs,
                    )
                )
    if manifest is not None:
        extra = set(manifest) - used_manifest
        if extra:
            sample = sorted(extra)[:3]
            raise AlgorithmFactoryError(
                f"seed manifest has {len(extra)} tasks that are not in the plan, "
                f"including {sample}"
            )
    return tuple(specs)


def specs_to_task_records(
    specs: Iterable[OptimizationTaskSpec],
) -> tuple[TaskRecord, ...]:
    """
    Convert optimisation specs into contiguous generic task records.

    Parameters
    ----------
    specs : iterable of OptimizationTaskSpec
        Ordered optimisation task specifications.

    Returns
    -------
    tuple of TaskRecord
        Contiguous ``task_000000`` … records with kind
        ``optimization_run``.
    """
    records: list[TaskRecord] = []
    for index, spec in enumerate(specs):
        records.append(
            TaskRecord(
                task_index=index,
                task_id=format_task_id(index),
                kind=OPTIMIZATION_TASK_KIND,
                payload=spec.to_payload(),
            )
        )
    return tuple(records)


def plan_optimization_tasks(config: ExperimentConfig) -> tuple[TaskRecord, ...]:
    """
    Plan contiguous optimisation ``TaskRecord`` entries for a config.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment TOML configuration.

    Returns
    -------
    tuple of TaskRecord
        Manifest-ready task list.
    """
    return specs_to_task_records(plan_optimization_specs(config))
