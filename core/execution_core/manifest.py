"""
Immutable experiment manifest and environment artifact helpers.

Preparation writes ``manifest.json`` and ``environment.json`` once. Readers
reload and validate them; writers refuse to overwrite an existing experiment
unless an explicit overwrite flag is provided for controlled resume tooling.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

from core.execution_core.atomic_io import read_json, write_json_atomic, write_text_atomic
from core.execution_core.artifacts import create_experiment_layout, utc_now
from core.execution_core.config import ExperimentConfig
from core.execution_core.models import (
    ExperimentManifest,
    ExperimentPaths,
    TaskRecord,
    format_task_id,
)

MANIFEST_SCHEMA_VERSION = 1
ENVIRONMENT_SCHEMA_VERSION = 1
WORKFLOW_VERSION = 1

WORKER_MODULES: Mapping[str, str] = {
    "optimization_runs": "core.ioh_core.workflows.optimization_runs.worker",
    "brute_force": "core.ioh_core.workflows.brute_force.worker",
    "sensitivity_analysis": "core.ioh_core.workflows.sensitivity_analysis.worker",
    "sampling": "core.ioh_core.workflows.sampling.worker",
    "structural_library": "core.ioh_core.workflows.structural_library.worker",
}

TRACKED_PACKAGES: tuple[str, ...] = (
    "ioh",
    "nlopt",
    "numpy",
    "scipy",
    "triangle",
    "structuralcodes",
    "hpdesignbench",
)


class ManifestError(ValueError):
    """
    Raised when a manifest or environment artifact is invalid or protected.

    Typical causes: unknown schema versions, non-contiguous task indices,
    missing worker modules, JSON shape errors, or refused overwrites.
    """


@dataclass(frozen=True)
class EnvironmentRecord:
    """
    Runtime expectations recorded beside the experiment manifest.

    Attributes
    ----------
    schema_version : int
        Environment schema version (currently ``1`` only).
    python_executable : str
        Absolute interpreter path expected by workers.
    python_version : str
        ``platform.python_version()`` string.
    platform : Mapping[str, str]
        Coarse OS identity (``system``, ``release``, ``machine``).
    package_versions : Mapping[str, str | None]
        Selected dependency versions from packaging metadata; ``None`` when
        a distribution is not installed at preparation time.
    git_revision : str or None
        Git HEAD SHA when available.
    git_dirty : bool or None
        Whether the worktree had uncommitted changes; ``None`` when Git is
        unavailable.
    thread_environment : Mapping[str, str]
        Thread-limit environment variables workers must apply/validate.
    """

    schema_version: int
    python_executable: str
    python_version: str
    platform: Mapping[str, str]
    package_versions: Mapping[str, str | None]
    git_revision: str | None
    git_dirty: bool | None
    thread_environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", dict(self.platform))
        object.__setattr__(self, "package_versions", dict(self.package_versions))
        object.__setattr__(self, "thread_environment", dict(self.thread_environment))


def worker_module_for(workflow_type: str) -> str:
    """
    Return the worker import path for a supported workflow.

    Parameters
    ----------
    workflow_type : str
        Workflow selector such as ``\"optimization_runs\"``.

    Returns
    -------
    str
        Fully-qualified worker module path.

    Raises
    ------
    ManifestError
        If ``workflow_type`` is unknown.
    """
    try:
        return WORKER_MODULES[workflow_type]
    except KeyError as exc:
        raise ManifestError(f"unknown workflow_type {workflow_type!r}") from exc


def _iso_utc(moment: datetime) -> str:
    """
    Format a timezone-aware datetime as a UTC ISO-8601 string ending in ``Z``.

    Parameters
    ----------
    moment : datetime
        Timezone-aware timestamp.

    Returns
    -------
    str
        UTC timestamp such as ``\"2026-07-28T09:15:00.123456Z\"``.
    """
    if moment.tzinfo is None:
        raise ManifestError("timestamps must be timezone-aware UTC")
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_utc(value: str) -> datetime:
    """
    Parse a UTC ISO-8601 timestamp into a timezone-aware datetime.

    Parameters
    ----------
    value : str
        Timestamp string; trailing ``Z`` is accepted.

    Returns
    -------
    datetime
        Timezone-aware UTC datetime.
    """
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ManifestError(f"created_at must be timezone-aware, got {value!r}")
    return moment.astimezone(timezone.utc)


def _package_version(name: str) -> str | None:
    """
    Look up an installed distribution version without importing the package.

    Parameters
    ----------
    name : str
        Distribution name on ``importlib.metadata``.

    Returns
    -------
    str or None
        Version string, or ``None`` if the distribution is missing.
    """
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_state(repo_dir: Path | None = None) -> tuple[str | None, bool | None]:
    """
    Best-effort Git revision and dirty flag for reproducibility metadata.

    Parameters
    ----------
    repo_dir : Path or None, optional
        Directory used as the Git working tree. Defaults to the process CWD.

    Returns
    -------
    tuple
        ``(revision, dirty)`` where either field may be ``None`` when Git is
        unavailable or the directory is not a repository.
    """
    cwd = str(repo_dir) if repo_dir is not None else None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return revision, None
    return revision, bool(status.strip())


def build_environment_record(
    *,
    python_executable: str | None = None,
    thread_environment: Mapping[str, str] | None = None,
    repo_dir: Path | None = None,
) -> EnvironmentRecord:
    """
    Capture the current process runtime expectations for ``environment.json``.

    Parameters
    ----------
    python_executable : str or None, optional
        Absolute interpreter path. Defaults to ``sys.executable``.
    thread_environment : Mapping[str, str] or None, optional
        Thread-limit environment variables from the experiment config.
    repo_dir : Path or None, optional
        Optional Git working tree for revision / dirty detection.

    Returns
    -------
    EnvironmentRecord
        Frozen environment document ready for JSON publish.
    """
    executable = Path(python_executable or sys.executable).resolve()
    revision, dirty = _git_state(repo_dir)
    packages = {name: _package_version(name) for name in TRACKED_PACKAGES}
    return EnvironmentRecord(
        schema_version=ENVIRONMENT_SCHEMA_VERSION,
        python_executable=str(executable),
        python_version=platform.python_version(),
        platform={
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        package_versions=packages,
        git_revision=revision,
        git_dirty=dirty,
        thread_environment=dict(thread_environment or {}),
    )


def environment_to_dict(record: EnvironmentRecord) -> dict[str, Any]:
    """
    Convert an ``EnvironmentRecord`` to a JSON-compatible mapping.

    Parameters
    ----------
    record : EnvironmentRecord
        Environment document.

    Returns
    -------
    dict
        JSON-serialisable mapping.
    """
    return {
        "schema_version": record.schema_version,
        "python_executable": record.python_executable,
        "python_version": record.python_version,
        "platform": dict(record.platform),
        "package_versions": dict(record.package_versions),
        "git_revision": record.git_revision,
        "git_dirty": record.git_dirty,
        "thread_environment": dict(record.thread_environment),
    }


def environment_from_dict(data: Mapping[str, Any]) -> EnvironmentRecord:
    """
    Parse and validate an environment JSON mapping.

    Parameters
    ----------
    data : Mapping[str, Any]
        Parsed ``environment.json`` object.

    Returns
    -------
    EnvironmentRecord
        Validated frozen record.

    Raises
    ------
    ManifestError
        If the schema or required fields are invalid.
    """
    if not isinstance(data, Mapping):
        raise ManifestError("environment.json must be a JSON object")
    schema_version = data.get("schema_version")
    if schema_version != ENVIRONMENT_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported environment schema_version {schema_version!r}; "
            f"supported: {ENVIRONMENT_SCHEMA_VERSION}"
        )
    python_executable = data.get("python_executable")
    python_version = data.get("python_version")
    if not isinstance(python_executable, str) or not python_executable.strip():
        raise ManifestError("environment.python_executable must be a non-empty string")
    if not isinstance(python_version, str) or not python_version.strip():
        raise ManifestError("environment.python_version must be a non-empty string")

    platform_raw = data.get("platform")
    if not isinstance(platform_raw, Mapping):
        raise ManifestError("environment.platform must be an object")
    platform_data = {str(k): str(v) for k, v in platform_raw.items()}

    packages_raw = data.get("package_versions")
    if not isinstance(packages_raw, Mapping):
        raise ManifestError("environment.package_versions must be an object")
    packages: dict[str, str | None] = {}
    for key, value in packages_raw.items():
        if value is not None and not isinstance(value, str):
            raise ManifestError(
                f"environment.package_versions[{key!r}] must be a string or null"
            )
        packages[str(key)] = value

    thread_raw = data.get("thread_environment", {})
    if not isinstance(thread_raw, Mapping):
        raise ManifestError("environment.thread_environment must be an object")
    thread_environment = {str(k): str(v) for k, v in thread_raw.items()}

    git_revision = data.get("git_revision")
    git_dirty = data.get("git_dirty")
    if git_revision is not None and not isinstance(git_revision, str):
        raise ManifestError("environment.git_revision must be a string or null")
    if git_dirty is not None and not isinstance(git_dirty, bool):
        raise ManifestError("environment.git_dirty must be a bool or null")

    return EnvironmentRecord(
        schema_version=schema_version,
        python_executable=python_executable,
        python_version=python_version,
        platform=platform_data,
        package_versions=packages,
        git_revision=git_revision,
        git_dirty=git_dirty,
        thread_environment=thread_environment,
    )


def validate_tasks(tasks: tuple[TaskRecord, ...] | list[TaskRecord]) -> tuple[TaskRecord, ...]:
    """
    Validate contiguous task indices and stable task IDs.

    Parameters
    ----------
    tasks : sequence of TaskRecord
        Candidate task list.

    Returns
    -------
    tuple of TaskRecord
        Validated task tuple.

    Raises
    ------
    ManifestError
        If indices are non-contiguous or IDs are inconsistent.
    """
    records = tuple(tasks)
    for index, task in enumerate(records):
        if task.task_index != index:
            raise ManifestError(
                f"task_index must be contiguous starting at 0; "
                f"expected {index}, got {task.task_index}"
            )
        expected_id = format_task_id(index)
        if task.task_id != expected_id:
            raise ManifestError(
                f"task_id must be {expected_id!r}, got {task.task_id!r}"
            )
        if not task.kind:
            raise ManifestError(f"task {index} has an empty kind")
    return records


def build_manifest(
    *,
    experiment_id: str,
    workflow_type: str,
    tasks: tuple[TaskRecord, ...] | list[TaskRecord],
    input_files: Mapping[str, str] | None = None,
    created_at: datetime | None = None,
    workflow_version: int = WORKFLOW_VERSION,
    schema_version: int = MANIFEST_SCHEMA_VERSION,
) -> ExperimentManifest:
    """
    Build a validated immutable ``ExperimentManifest``.

    Parameters
    ----------
    experiment_id : str
        Experiment directory / human id.
    workflow_type : str
        Workflow selector used to resolve ``worker_module``.
    tasks : sequence of TaskRecord
        Contiguous task list (may be empty only before planner wiring; A5
        fills optimisation tasks).
    input_files : Mapping[str, str] or None, optional
        Relative input paths mapped to SHA-256 digests.
    created_at : datetime or None, optional
        Timezone-aware UTC timestamp; defaults to now.
    workflow_version : int, optional
        Workflow contract version, default ``1``.
    schema_version : int, optional
        Manifest schema version, default ``1``.

    Returns
    -------
    ExperimentManifest
        Frozen manifest ready for JSON publish.
    """
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema_version {schema_version!r}; "
            f"supported: {MANIFEST_SCHEMA_VERSION}"
        )
    if not experiment_id.strip():
        raise ManifestError("experiment_id must be a non-empty string")
    worker_module = worker_module_for(workflow_type)
    moment = created_at if created_at is not None else utc_now()
    return ExperimentManifest(
        schema_version=schema_version,
        experiment_id=experiment_id.strip(),
        workflow_type=workflow_type,
        workflow_version=workflow_version,
        worker_module=worker_module,
        created_at=moment,
        input_files=dict(input_files or {}),
        tasks=validate_tasks(tasks),
    )


def manifest_to_dict(manifest: ExperimentManifest) -> dict[str, Any]:
    """
    Convert an ``ExperimentManifest`` to a JSON-compatible mapping.

    Parameters
    ----------
    manifest : ExperimentManifest
        Manifest document.

    Returns
    -------
    dict
        JSON-serialisable mapping with relative task payloads preserved.
    """
    return {
        "schema_version": manifest.schema_version,
        "experiment_id": manifest.experiment_id,
        "workflow_type": manifest.workflow_type,
        "workflow_version": manifest.workflow_version,
        "worker_module": manifest.worker_module,
        "created_at": _iso_utc(manifest.created_at),
        "input_files": dict(manifest.input_files),
        "tasks": [
            {
                "task_index": task.task_index,
                "task_id": task.task_id,
                "kind": task.kind,
                "payload": dict(task.payload),
            }
            for task in manifest.tasks
        ],
    }


def manifest_from_dict(data: Mapping[str, Any]) -> ExperimentManifest:
    """
    Parse and validate a manifest JSON mapping.

    Parameters
    ----------
    data : Mapping[str, Any]
        Parsed ``manifest.json`` object.

    Returns
    -------
    ExperimentManifest
        Validated frozen manifest.

    Raises
    ------
    ManifestError
        If the schema, worker module, tasks, or timestamps are invalid.
    """
    if not isinstance(data, Mapping):
        raise ManifestError("manifest.json must be a JSON object")
    schema_version = data.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema_version {schema_version!r}; "
            f"supported: {MANIFEST_SCHEMA_VERSION}"
        )

    experiment_id = data.get("experiment_id")
    workflow_type = data.get("workflow_type")
    workflow_version = data.get("workflow_version")
    worker_module = data.get("worker_module")
    created_at_raw = data.get("created_at")
    input_files_raw = data.get("input_files", {})
    tasks_raw = data.get("tasks", [])

    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ManifestError("manifest.experiment_id must be a non-empty string")
    if not isinstance(workflow_type, str) or not workflow_type.strip():
        raise ManifestError("manifest.workflow_type must be a non-empty string")
    if not isinstance(workflow_version, int) or isinstance(workflow_version, bool):
        raise ManifestError("manifest.workflow_version must be an integer")
    if not isinstance(worker_module, str) or not worker_module.strip():
        raise ManifestError("manifest.worker_module must be a non-empty string")
    expected_worker = worker_module_for(workflow_type)
    if worker_module != expected_worker:
        raise ManifestError(
            f"manifest.worker_module must be {expected_worker!r} for "
            f"workflow {workflow_type!r}, got {worker_module!r}"
        )
    if not isinstance(created_at_raw, str):
        raise ManifestError("manifest.created_at must be an ISO-8601 string")
    if not isinstance(input_files_raw, Mapping):
        raise ManifestError("manifest.input_files must be an object")
    if not isinstance(tasks_raw, list):
        raise ManifestError("manifest.tasks must be an array")

    input_files = {str(k): str(v) for k, v in input_files_raw.items()}
    tasks: list[TaskRecord] = []
    for index, entry in enumerate(tasks_raw):
        if not isinstance(entry, Mapping):
            raise ManifestError(f"manifest.tasks[{index}] must be an object")
        payload = entry.get("payload", {})
        if not isinstance(payload, Mapping):
            raise ManifestError(f"manifest.tasks[{index}].payload must be an object")
        try:
            tasks.append(
                TaskRecord(
                    task_index=int(entry["task_index"]),
                    task_id=str(entry["task_id"]),
                    kind=str(entry["kind"]),
                    payload=dict(payload),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(
                f"manifest.tasks[{index}] is invalid: {exc}"
            ) from exc

    return ExperimentManifest(
        schema_version=schema_version,
        experiment_id=experiment_id,
        workflow_type=workflow_type,
        workflow_version=workflow_version,
        worker_module=worker_module,
        created_at=_parse_iso_utc(created_at_raw),
        input_files=input_files,
        tasks=validate_tasks(tasks),
    )


def write_manifest(
    path: Path,
    manifest: ExperimentManifest,
    *,
    overwrite: bool = False,
) -> None:
    """
    Atomically publish ``manifest.json``.

    Parameters
    ----------
    path : Path
        Destination path (normally ``experiments/{id}/manifest.json``).
    manifest : ExperimentManifest
        Validated manifest.
    overwrite : bool, optional
        When ``False`` (default), refuse if ``path`` already exists.

    Raises
    ------
    ManifestError
        If overwrite is refused or the manifest fails validation.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise ManifestError(
            f"refusing to overwrite existing manifest at {path}; "
            "pass overwrite=True only for an explicit resume/rebuild operation"
        )
    validate_tasks(manifest.tasks)
    write_json_atomic(path, manifest_to_dict(manifest))


def load_manifest(path: Path) -> ExperimentManifest:
    """
    Read and validate ``manifest.json``.

    Parameters
    ----------
    path : Path
        Manifest path.

    Returns
    -------
    ExperimentManifest
        Validated frozen manifest.
    """
    return manifest_from_dict(read_json(Path(path)))


def write_environment(
    path: Path,
    record: EnvironmentRecord,
    *,
    overwrite: bool = False,
) -> None:
    """
    Atomically publish ``environment.json``.

    Parameters
    ----------
    path : Path
        Destination path.
    record : EnvironmentRecord
        Environment document.
    overwrite : bool, optional
        When ``False`` (default), refuse if ``path`` already exists.

    Raises
    ------
    ManifestError
        If overwrite is refused.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise ManifestError(
            f"refusing to overwrite existing environment at {path}; "
            "pass overwrite=True only for an explicit resume/rebuild operation"
        )
    if record.schema_version != ENVIRONMENT_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported environment schema_version {record.schema_version!r}"
        )
    write_json_atomic(path, environment_to_dict(record))


def load_environment(path: Path) -> EnvironmentRecord:
    """
    Read and validate ``environment.json``.

    Parameters
    ----------
    path : Path
        Environment path.

    Returns
    -------
    EnvironmentRecord
        Validated frozen record.
    """
    return environment_from_dict(read_json(Path(path)))


def write_source_config(
    path: Path,
    source_toml: str | bytes | Path,
    *,
    overwrite: bool = False,
) -> None:
    """
    Copy the preparation TOML beside the experiment artifacts.

    Parameters
    ----------
    path : Path
        Destination ``source-config.toml`` path.
    source_toml : str or bytes or Path
        TOML text, bytes, or path to the original config file.
    overwrite : bool, optional
        When ``False`` (default), refuse if ``path`` already exists.

    Raises
    ------
    ManifestError
        If overwrite is refused.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise ManifestError(
            f"refusing to overwrite existing source config at {path}"
        )
    if isinstance(source_toml, Path):
        text = source_toml.read_text(encoding="utf-8")
    elif isinstance(source_toml, bytes):
        text = source_toml.decode("utf-8")
    else:
        text = source_toml
    if not text.endswith("\n"):
        text += "\n"
    write_text_atomic(path, text)


def prepare_experiment_artifacts(
    config: ExperimentConfig,
    *,
    root: Path,
    tasks: tuple[TaskRecord, ...] | list[TaskRecord] | None = None,
    input_files: Mapping[str, str] | None = None,
    python_executable: str | None = None,
    overwrite: bool = False,
    repo_dir: Path | None = None,
) -> tuple[ExperimentPaths, ExperimentManifest, EnvironmentRecord]:
    """
    Create the experiment layout and publish immutable preparation artifacts.

    Parameters
    ----------
    config : ExperimentConfig
        Validated TOML configuration.
    root : Path
        Absolute experiment root directory.
    tasks : sequence of TaskRecord or None, optional
        Manifest task list. Defaults to an empty list until the A5 planner
        supplies optimisation tasks.
    input_files : Mapping[str, str] or None, optional
        Relative input hashes (filled by A4 snapshots).
    python_executable : str or None, optional
        Absolute interpreter for ``environment.json``. Empty local config
        falls back to ``sys.executable``.
    overwrite : bool, optional
        When ``False``, refuse if ``manifest.json`` already exists.
    repo_dir : Path or None, optional
        Optional Git working tree for environment metadata.

    Returns
    -------
    tuple
        ``(paths, manifest, environment)`` after atomic publish.

    Raises
    ------
    ManifestError
        If overwrite protection triggers or artifacts fail validation.
    """
    root = Path(root).resolve()
    paths = create_experiment_layout(root)
    if paths.manifest.exists() and not overwrite:
        raise ManifestError(
            f"experiment already prepared at {root}; refusing to overwrite"
        )

    resolved_python = python_executable
    if not resolved_python:
        configured = config.execution.local.python_executable.strip()
        resolved_python = configured or sys.executable

    environment = build_environment_record(
        python_executable=resolved_python,
        thread_environment=config.execution.thread_environment,
        repo_dir=repo_dir,
    )
    manifest = build_manifest(
        experiment_id=config.experiment.id,
        workflow_type=config.experiment.workflow,
        tasks=tasks or (),
        input_files=input_files,
    )

    if config.source_path is not None and config.source_path.exists():
        write_source_config(
            paths.source_config,
            config.source_path,
            overwrite=overwrite,
        )

    write_environment(paths.environment, environment, overwrite=overwrite)
    write_manifest(paths.manifest, manifest, overwrite=overwrite)

    from core.execution_core.output_layout import (
        layout_root_from_output_root,
        write_layout_sidecar,
    )

    config_base = (
        config.source_path.parent
        if config.source_path is not None
        else Path.cwd()
    )
    write_layout_sidecar(
        paths.root,
        layout_root_from_output_root(
            config.experiment.output_root,
            cwd=config_base,
        ),
    )

    from core.ioh_core.structural_artifacts.sealing import (
        seal_structural_artifacts_for_experiment,
    )

    seal_structural_artifacts_for_experiment(
        config,
        experiment_dir=paths.root,
        input_hashes=manifest.input_files,
        overwrite=overwrite,
    )
    return paths, manifest, environment
