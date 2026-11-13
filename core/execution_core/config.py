"""
Strict TOML loader for experiment preparation and execution settings.

Uses the standard-library ``tomllib`` parser only. Unknown keys, unknown
workflows/backends, and incomplete Slurm settings are rejected loudly so
preparation cannot silently invent runtime behaviour.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_WORKFLOWS = frozenset(
    {
        "optimization_runs",
        "brute_force",
        "sensitivity_analysis",
        "structural_library",
    }
)
SUPPORTED_BACKENDS = frozenset({"local", "slurm"})

_TOP_KEYS = frozenset(
    {
        "schema_version",
        "experiment",
        "execution",
        "brute_force",
        "sensitivity_analysis",
        "sampling",
        "structural_library",
        "structural_artifacts",
    }
)
_EXPERIMENT_KEYS = frozenset(
    {
        "id",
        "workflow",
        "slab_type",
        "problem_ids",
        "output_root",
        "algorithms",
        "repetitions",
        "termination",
    }
)
_ALGORITHM_KEYS = frozenset(
    {"name", "max_evaluations", "parameters", "repetitions"}
)
_REPETITIONS_KEYS = frozenset({"count", "seed_base", "seed_manifest_path"})
_TERMINATION_KEYS = frozenset(
    {
        "mode",
        "global_optimum",
        "global_optima",
        "allowed_deviation_percent",
        "target_reference",
    }
)
_TERMINATION_MODES = frozenset({"fixed_budget", "fixed_target"})
_TARGET_REFERENCES = frozenset({"all"})
_BRUTE_FORCE_KEYS = frozenset(
    {
        "rows_per_shard",
        "parquet_batch_rows",
        "compression",
        "acknowledge_large_run",
        "maximum_expected_rows",
        "maximum_expected_bytes",
        "reuse_structural_artifacts",
        "start_index",
        "max_rows",
    }
)
_BRUTE_FORCE_COMPRESSIONS = frozenset({"zstd", "snappy", "gzip", "none"})
_SAMPLING_KEYS = frozenset(
    {
        "method",
        "sample_size",
        "scramble",
        "delete_duplicates",
        "rows_per_shard",
        "parquet_batch_rows",
        "compression",
        "acknowledge_large_run",
        "maximum_expected_evaluations",
        "maximum_expected_bytes",
        "explicit_seeds",
    }
)
_SAMPLING_COMPRESSIONS = frozenset({"zstd", "snappy", "gzip", "none"})
_SAMPLING_METHODS = frozenset({"scipy_qmc_sobol"})
_SENSITIVITY_KEYS = frozenset(
    {
        "sample_size",
        "factor_names",
        "response_key",
        "scramble",
        "seed",
        "blocks_per_shard",
        "parquet_batch_rows",
        "compression",
        "bootstrap_resamples",
        "bootstrap_confidence_level",
        "bootstrap_seed",
        "compute_bootstrap_ci",
        "acknowledge_large_run",
        "maximum_expected_evaluations",
        "maximum_expected_bytes",
    }
)
_SENSITIVITY_COMPRESSIONS = frozenset({"zstd", "snappy", "gzip", "none"})
_SENSITIVITY_RESPONSE_KEYS = frozenset({"y", "y_p"})
_STRUCTURAL_LIBRARY_KEYS = frozenset(
    {
        "rows_per_shard",
        "start_id",
        "max_ids",
        "acknowledge_large_run",
        "maximum_expected_ids",
        "maximum_expected_bytes",
        "compatible_problem_ids",
    }
)
_STRUCTURAL_ARTIFACTS_KEYS = frozenset(
    {"mode", "library_path", "allow_snapshot_hash_mismatch"}
)
_STRUCTURAL_ARTIFACTS_MODES = frozenset({"compute", "library"})
_EXECUTION_KEYS = frozenset(
    {
        "backend",
        "max_parallel_tasks",
        "timeout_seconds",
        "cpus_per_task",
        "thread_environment",
        "local",
        "slurm",
    }
)
_LOCAL_KEYS = frozenset({"python_executable"})
_SLURM_KEYS = frozenset(
    {
        "python_executable",
        "partition",
        "account",
        "memory",
        "time_limit",
        "array_concurrency",
        "maximum_array_size",
    }
)


class ConfigError(ValueError):
    """
    Raised when an experiment TOML file is invalid.

    Typical causes: unknown schema version, unknown keys, missing required
    fields, invalid numeric limits, duplicate problem/algorithm definitions,
    or incomplete Slurm settings when ``backend = \"slurm\"``.
    """


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    """
    Return ``value`` if it is a mapping, otherwise raise ``ConfigError``.

    Parameters
    ----------
    value : Any
        Candidate TOML table.
    context : str
        Human-readable location used in the error message.

    Returns
    -------
    dict
        Mapping with string keys.

    Raises
    ------
    ConfigError
        If ``value`` is not a mapping.
    """
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a table")
    return value


def _reject_unknown_keys(
    table: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    context: str,
) -> None:
    """
    Reject any key not listed in ``allowed``.

    Parameters
    ----------
    table : Mapping[str, Any]
        Parsed TOML table.
    allowed : frozenset of str
        Exact set of permitted keys.
    context : str
        Human-readable location used in the error message.

    Raises
    ------
    ConfigError
        If one or more unknown keys are present.
    """
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"{context} has unknown keys: {unknown}")


def _require_nonempty_str(value: Any, *, context: str) -> str:
    """
    Coerce and validate a non-empty string field.

    Parameters
    ----------
    value : Any
        Candidate string value.
    context : str
        Human-readable location used in the error message.

    Returns
    -------
    str
        Stripped non-empty string.

    Raises
    ------
    ConfigError
        If the value is missing, not a string, or blank after stripping.
    """
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a non-empty string")
    return value.strip()


def _validate_json_compatible(value: Any, *, context: str) -> None:
    """
    Reject values that are not JSON-compatible TOML payloads.

    Allowed leaves are ``None``, ``bool``, ``int``, ``float``, and ``str``.
    Containers may be lists or string-keyed mappings. TOML dates/times and
    non-finite floats are rejected.

    Parameters
    ----------
    value : Any
        Candidate value.
    context : str
        Human-readable location used in the error message.

    Raises
    ------
    ConfigError
        If ``value`` is not JSON-compatible.
    """
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{context} must be a finite number")
        return
    if isinstance(value, (datetime, date, time)):
        raise ConfigError(f"{context} must not be a TOML date/time")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_compatible(item, context=f"{context}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigError(f"{context} keys must be strings")
            _validate_json_compatible(item, context=f"{context}.{key}")
        return
    raise ConfigError(f"{context} must be a JSON scalar, list, or table")


@dataclass(frozen=True)
class AlgorithmConfig:
    """
    One algorithm entry from ``[[experiment.algorithms]]``.

    Attributes
    ----------
    name : str
        Algorithm factory name such as ``\"RANDOM_SEARCH\"``.
    max_evaluations : int
        Positive evaluation budget for this algorithm.
    parameters : Mapping[str, Any]
        Optional JSON-compatible hyperparameter mapping. Omitted TOML tables
        become an empty immutable mapping. Registry validation happens later.
    repetitions : int or None
        Optional per-algorithm repetition-count override. ``None`` inherits
        ``experiment.repetitions.count``.
    """

    name: str
    max_evaluations: int
    parameters: Mapping[str, Any] = field(default_factory=dict)
    repetitions: int | None = None

    def __post_init__(self) -> None:
        if self.max_evaluations < 1:
            raise ConfigError(
                f"algorithm {self.name!r}: max_evaluations must be >= 1, "
                f"got {self.max_evaluations}"
            )
        if self.repetitions is not None and (
            not isinstance(self.repetitions, int)
            or isinstance(self.repetitions, bool)
            or self.repetitions < 1
        ):
            raise ConfigError(
                f"algorithm {self.name!r}: repetitions must be an integer "
                f">= 1 or null, got {self.repetitions!r}"
            )
        if not isinstance(self.parameters, Mapping):
            raise ConfigError(
                f"algorithm {self.name!r}: parameters must be a table"
            )
        raw = dict(self.parameters)
        _validate_json_compatible(
            raw,
            context=f"algorithm {self.name!r}.parameters",
        )
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(copy.deepcopy(raw)),
        )


@dataclass(frozen=True)
class RepetitionsConfig:
    """
    Repetition and seed-base settings for optimisation runs.

    Attributes
    ----------
    count : int
        Number of independent repetitions per problem/algorithm pair (>= 1).
    seed_base : int
        Non-negative integer mixed into deterministic stochastic seeds.
    seed_manifest_path : str or None
        Optional JSONL file of explicit per-task seeds. Omitted configs keep
        deriving seeds from ``seed_base`` and the algorithm token.
    """

    count: int
    seed_base: int
    seed_manifest_path: str | None = None

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ConfigError(f"repetitions.count must be >= 1, got {self.count}")
        if self.seed_base < 0:
            raise ConfigError(
                f"repetitions.seed_base must be >= 0, got {self.seed_base}"
            )
        if self.seed_manifest_path is not None and (
            not isinstance(self.seed_manifest_path, str)
            or not self.seed_manifest_path.strip()
        ):
            raise ConfigError(
                "experiment.repetitions.seed_manifest_path must be a non-empty path"
            )


@dataclass(frozen=True)
class TerminationConfig:
    """
    Optimisation termination mode and fixed-target settings.

    Attributes
    ----------
    mode : str
        ``\"fixed_budget\"`` or ``\"fixed_target\"``.
    global_optimum : float or None
        Known non-negative finite optimum for a single-problem fixed-target
        experiment.
    global_optima : Mapping[str, float]
        Per-problem known optima for a multi-problem fixed-target experiment.
    allowed_deviation_percent : float or None
        Allowed percentage deviation (``5.0`` means five percent). Used only
        in fixed-target mode.
    target_reference : str or None
        ``\"all\"`` in fixed-target mode: every finite ``y_p`` updates the
        tracked best and can meet the threshold. ``None`` in fixed-budget
        mode.
    """

    mode: str = "fixed_budget"
    global_optimum: float | None = None
    global_optima: Mapping[str, float] = field(default_factory=dict)
    allowed_deviation_percent: float | None = None
    target_reference: str | None = None

    def __post_init__(self) -> None:
        optima = dict(self.global_optima)
        for problem_id, value in optima.items():
            if not isinstance(problem_id, str) or not problem_id.strip():
                raise ConfigError(
                    "experiment.termination.global_optima keys must be "
                    "non-empty strings"
                )
            if problem_id == "default":
                raise ConfigError(
                    "experiment.termination.global_optima forbids 'default'"
                )
            optima[problem_id] = _require_finite_nonnegative_number(
                value,
                context=(
                    "experiment.termination.global_optima"
                    f"[{problem_id!r}]"
                ),
            )
        object.__setattr__(self, "global_optima", MappingProxyType(optima))
        if self.target_reference == "feasible":
            # Historical TOMLs used this spelling. Preserve their ability to
            # run, but normalize immediately to the all-finite-y_p policy.
            object.__setattr__(self, "target_reference", "all")

        if self.mode not in _TERMINATION_MODES:
            raise ConfigError(
                f"experiment.termination.mode must be one of "
                f"{sorted(_TERMINATION_MODES)}, got {self.mode!r}"
            )
        if self.mode == "fixed_budget":
            if self.global_optimum is not None:
                raise ConfigError(
                    "experiment.termination.global_optimum is forbidden when "
                    "mode='fixed_budget'"
                )
            if self.global_optima:
                raise ConfigError(
                    "experiment.termination.global_optima is forbidden when "
                    "mode='fixed_budget'"
                )
            if self.allowed_deviation_percent is not None:
                raise ConfigError(
                    "experiment.termination.allowed_deviation_percent is "
                    "forbidden when mode='fixed_budget'"
                )
            if self.target_reference is not None:
                raise ConfigError(
                    "experiment.termination.target_reference is forbidden when "
                    "mode='fixed_budget'"
                )
            return

        has_scalar = self.global_optimum is not None
        has_mapping = bool(self.global_optima)
        if has_scalar == has_mapping:
            raise ConfigError(
                "fixed_target requires exactly one of "
                "experiment.termination.global_optimum or "
                "experiment.termination.global_optima"
            )
        if self.allowed_deviation_percent is None:
            raise ConfigError(
                "experiment.termination.allowed_deviation_percent is required "
                "when mode='fixed_target'"
            )
        if self.target_reference is None:
            raise ConfigError(
                "experiment.termination.target_reference is required when "
                "mode='fixed_target'"
            )
        if self.target_reference not in _TARGET_REFERENCES:
            raise ConfigError(
                "experiment.termination.target_reference must be one of "
                f"{sorted(_TARGET_REFERENCES)}, got {self.target_reference!r}"
            )
        if self.global_optimum is not None:
            object.__setattr__(
                self,
                "global_optimum",
                _require_finite_nonnegative_number(
                    self.global_optimum,
                    context="experiment.termination.global_optimum",
                ),
            )
        if (
            not math.isfinite(float(self.allowed_deviation_percent))
            or float(self.allowed_deviation_percent) < 0.0
        ):
            raise ConfigError(
                "experiment.termination.allowed_deviation_percent must be a "
                f"finite number >= 0, got {self.allowed_deviation_percent!r}"
            )
        problem_ids = self.global_optima if self.global_optima else ("<scalar>",)
        for problem_id in problem_ids:
            threshold = self.target_threshold_for(problem_id)
            if threshold is None or not math.isfinite(threshold):
                raise ConfigError(
                    "experiment.termination derived target_threshold must be "
                    f"finite for problem_id {problem_id!r}, got {threshold!r}"
                )

    @property
    def target_threshold(self) -> float | None:
        """
        Return the derived minimisation threshold, or ``None`` in fixed-budget
        mode.

        Returns
        -------
        float or None
            ``global_optimum * (1 + allowed_deviation_percent / 100)`` when
            fixed-target fields are present; otherwise ``None``.
        """
        if (
            self.mode != "fixed_target"
            or self.global_optimum is None
            or self.allowed_deviation_percent is None
        ):
            return None
        return float(self.global_optimum) * (
            1.0 + float(self.allowed_deviation_percent) / 100.0
        )

    def global_optimum_for(self, problem_id: str) -> float | None:
        """Return the fixed-target optimum resolved for ``problem_id``."""
        if self.mode != "fixed_target":
            return None
        if self.global_optimum is not None:
            return float(self.global_optimum)
        try:
            return float(self.global_optima[problem_id])
        except KeyError as exc:
            raise ConfigError(
                "experiment.termination.global_optima has no entry for "
                f"problem_id {problem_id!r}"
            ) from exc

    def target_threshold_for(self, problem_id: str) -> float | None:
        """Return the fixed-target threshold resolved for ``problem_id``."""
        optimum = self.global_optimum_for(problem_id)
        if optimum is None or self.allowed_deviation_percent is None:
            return None
        threshold = optimum * (
            1.0 + float(self.allowed_deviation_percent) / 100.0
        )
        if not math.isfinite(threshold):
            raise ConfigError(
                "experiment.termination derived target_threshold must be "
                f"finite for problem_id {problem_id!r}, got {threshold!r}"
            )
        return threshold


@dataclass(frozen=True)
class LocalBackendConfig:
    """
    Local-subprocess backend settings.

    Attributes
    ----------
    python_executable : str
        Absolute interpreter path, or empty string to resolve later to
        ``sys.executable`` at preparation time.
    """

    python_executable: str = ""


@dataclass(frozen=True)
class SlurmBackendConfig:
    """
    Direct Slurm array backend settings.

    Attributes
    ----------
    python_executable : str
        Absolute cluster interpreter path used inside generated job scripts.
    partition : str
        Slurm partition name.
    account : str
        Slurm account / project string (may be empty when the site allows it).
    memory : str
        Memory request such as ``\"8G\"``.
    time_limit : str
        Wall-time limit such as ``\"00:45:00\"``.
    array_concurrency : int
        Maximum concurrently running array tasks (`%N` throttle).
    maximum_array_size : int
        Upper bound on one generated array length.
    """

    python_executable: str = ""
    partition: str = ""
    account: str = ""
    memory: str = "8G"
    time_limit: str = "00:45:00"
    array_concurrency: int = 64
    maximum_array_size: int = 1000

    def require_complete(self) -> None:
        """
        Validate fields required when ``backend = \"slurm\"``.

        Raises
        ------
        ConfigError
            If any required Slurm field is blank or numeric limits are invalid.
        """
        if not self.python_executable.strip():
            raise ConfigError(
                "execution.slurm.python_executable is required when backend='slurm'"
            )
        if not self.partition.strip():
            raise ConfigError(
                "execution.slurm.partition is required when backend='slurm'"
            )
        if not self.memory.strip():
            raise ConfigError(
                "execution.slurm.memory is required when backend='slurm'"
            )
        if not self.time_limit.strip():
            raise ConfigError(
                "execution.slurm.time_limit is required when backend='slurm'"
            )
        if self.array_concurrency < 1:
            raise ConfigError(
                "execution.slurm.array_concurrency must be >= 1, "
                f"got {self.array_concurrency}"
            )
        if self.maximum_array_size < 1:
            raise ConfigError(
                "execution.slurm.maximum_array_size must be >= 1, "
                f"got {self.maximum_array_size}"
            )


@dataclass(frozen=True)
class ExperimentSection:
    """
    Workflow-facing experiment identity and task-producing definitions.

    Attributes
    ----------
    id : str
        Experiment directory name under ``output_root``.
    workflow : str
        Workflow selector (``optimization_runs``, ``brute_force``,
        ``sensitivity_analysis``, ``sampling``, or ``structural_library``).
    slab_type : str
        Slab package name such as ``\"hp_slab\"``.
    problem_ids : tuple of str
        Explicit ordinary problem IDs to include (never ``\"default\"``).
    output_root : str
        Root directory for experiment folders; may be relative.
    algorithms : tuple of AlgorithmConfig
        Ordered algorithm definitions for optimisation workflows. Empty for
        ``brute_force``, ``sensitivity_analysis``, and ``sampling``.
    repetitions : RepetitionsConfig or None
        Repetition count and seed base for optimisation and sampling
        workflows. ``None`` for ``brute_force`` and ``sensitivity_analysis``.
    termination : TerminationConfig
        Optimisation termination settings. Defaults to fixed-budget behaviour.
        Present for every workflow after parse, but a non-default termination
        table is rejected unless ``workflow = \"optimization_runs\"``.
    """

    id: str
    workflow: str
    slab_type: str
    problem_ids: tuple[str, ...]
    output_root: str
    algorithms: tuple[AlgorithmConfig, ...]
    repetitions: RepetitionsConfig | None
    termination: TerminationConfig = field(default_factory=TerminationConfig)


@dataclass(frozen=True)
class BruteForceSection:
    """
    Brute-force sharding and Parquet settings from ``[brute_force]``.

    Attributes
    ----------
    rows_per_shard : int
        Target rows per shard before finish-group alignment (>= 1).
    parquet_batch_rows : int
        Writer batch size (>= 1).
    compression : str
        Parquet compression codec (``zstd``, ``snappy``, ``gzip``, ``none``).
    acknowledge_large_run : bool
        When ``False``, preflight blocks runs over configured maxima.
    maximum_expected_rows : int
        Soft cardinality limit requiring acknowledgement (>= 1).
    maximum_expected_bytes : int
        Soft storage limit requiring acknowledgement (>= 1).
    reuse_structural_artifacts : bool
        When ``True`` (default), finish-group evaluations reuse one immutable
        structural artifact per 24-candidate group. When ``False``, every
        candidate runs a full independent analysis (reference / equivalence
        mode).
    start_index : int
        Inclusive absolute linear start of the planned range (default ``0``).
        Must be a non-negative multiple of the finish-group size at plan time.
    max_rows : int or None
        Optional exclusive absolute upper bound on planned linear indices.
        When set, planning covers ``[start_index, aligned_stop)`` where
        ``aligned_stop`` is the largest finish-group-aligned value
        ``<= max_rows``. ``None`` means the catalogue cardinality.
    """

    rows_per_shard: int = 10_000
    parquet_batch_rows: int = 4096
    compression: str = "zstd"
    acknowledge_large_run: bool = False
    maximum_expected_rows: int = 5_000_000
    maximum_expected_bytes: int = 50_000_000_000
    reuse_structural_artifacts: bool = True
    start_index: int = 0
    max_rows: int | None = None

    def __post_init__(self) -> None:
        if self.rows_per_shard < 1:
            raise ConfigError(
                f"brute_force.rows_per_shard must be >= 1, got {self.rows_per_shard}"
            )
        if self.parquet_batch_rows < 1:
            raise ConfigError(
                "brute_force.parquet_batch_rows must be >= 1, "
                f"got {self.parquet_batch_rows}"
            )
        if self.compression not in _BRUTE_FORCE_COMPRESSIONS:
            raise ConfigError(
                f"brute_force.compression must be one of "
                f"{sorted(_BRUTE_FORCE_COMPRESSIONS)}, got {self.compression!r}"
            )
        if self.maximum_expected_rows < 1:
            raise ConfigError(
                "brute_force.maximum_expected_rows must be >= 1, "
                f"got {self.maximum_expected_rows}"
            )
        if self.maximum_expected_bytes < 1:
            raise ConfigError(
                "brute_force.maximum_expected_bytes must be >= 1, "
                f"got {self.maximum_expected_bytes}"
            )
        if self.start_index < 0:
            raise ConfigError(
                f"brute_force.start_index must be >= 0, got {self.start_index}"
            )
        if self.max_rows is not None and self.max_rows < 1:
            raise ConfigError(
                f"brute_force.max_rows must be >= 1 when set, got {self.max_rows}"
            )


@dataclass(frozen=True)
class SamplingSection:
    """
    Landscape-sampling settings from ``[sampling]``.

    Attributes
    ----------
    method : str
        Sampler name. v1 allows only ``scipy_qmc_sobol``.
    sample_size : int
        Requested Sobol length; must be a power of two and at least 2.
        Non-power-of-two sizes are rejected (no optimiser-style truncation).
    scramble : bool
        When ``True`` (default), SciPy scrambles with the derived seed.
        When ``False``, the canonical unscrambled sequence is used.
    delete_duplicates : bool
        When ``True``, drop duplicate projected index rows before evaluation.
        Default ``False`` (locked): keep duplicates.
    rows_per_shard : int
        Target evaluated rows per shard (>= 1).
    parquet_batch_rows : int
        Writer batch size (>= 1).
    compression : str
        Parquet compression codec (``zstd``, ``snappy``, ``gzip``, ``none``).
    acknowledge_large_run : bool
        When ``False``, preflight blocks runs over configured maxima.
    maximum_expected_evaluations : int
        Soft evaluation-count limit requiring acknowledgement (>= 1).
    maximum_expected_bytes : int
        Soft storage limit requiring acknowledgement (>= 1).
    explicit_seeds : tuple of int or None
        Optional SciPy scramble seeds, one per repetition. When set, length
        must equal ``repetitions.count`` and ``seed_base`` is not used to
        derive Sobol seeds. ``None`` (default) keeps SHA-256 derivation.
    """

    sample_size: int
    method: str = "scipy_qmc_sobol"
    scramble: bool = True
    delete_duplicates: bool = False
    rows_per_shard: int = 512
    parquet_batch_rows: int = 512
    compression: str = "zstd"
    acknowledge_large_run: bool = False
    maximum_expected_evaluations: int = 100_000
    maximum_expected_bytes: int = 2_000_000_000
    explicit_seeds: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.method not in _SAMPLING_METHODS:
            raise ConfigError(
                "sampling.method must be one of "
                f"{sorted(_SAMPLING_METHODS)}, got {self.method!r}"
            )
        if (
            not isinstance(self.sample_size, int)
            or isinstance(self.sample_size, bool)
            or self.sample_size < 2
            or self.sample_size & (self.sample_size - 1) != 0
        ):
            raise ConfigError(
                "sampling.sample_size must be a power of two and at least 2 "
                "(optimiser-style truncation is not used), "
                f"got {self.sample_size!r}"
            )
        if not isinstance(self.scramble, bool):
            raise ConfigError("sampling.scramble must be a boolean")
        if not isinstance(self.delete_duplicates, bool):
            raise ConfigError("sampling.delete_duplicates must be a boolean")
        if self.rows_per_shard < 1:
            raise ConfigError(
                f"sampling.rows_per_shard must be >= 1, got {self.rows_per_shard}"
            )
        if self.parquet_batch_rows < 1:
            raise ConfigError(
                "sampling.parquet_batch_rows must be >= 1, "
                f"got {self.parquet_batch_rows}"
            )
        if self.compression not in _SAMPLING_COMPRESSIONS:
            raise ConfigError(
                f"sampling.compression must be one of "
                f"{sorted(_SAMPLING_COMPRESSIONS)}, got {self.compression!r}"
            )
        if not isinstance(self.acknowledge_large_run, bool):
            raise ConfigError("sampling.acknowledge_large_run must be a boolean")
        if self.maximum_expected_evaluations < 1:
            raise ConfigError(
                "sampling.maximum_expected_evaluations must be >= 1, "
                f"got {self.maximum_expected_evaluations}"
            )
        if self.maximum_expected_bytes < 1:
            raise ConfigError(
                "sampling.maximum_expected_bytes must be >= 1, "
                f"got {self.maximum_expected_bytes}"
            )
        if self.explicit_seeds is not None:
            if not self.scramble:
                raise ConfigError(
                    "sampling.explicit_seeds requires scramble=true"
                )
            _validate_sampling_explicit_seeds(self.explicit_seeds)


@dataclass(frozen=True)
class StructuralLibrarySection:
    """
    Structural-library sharding settings from ``[structural_library]``.

    Attributes
    ----------
    rows_per_shard : int
        Contiguous structural IDs per Arrow part (>= 1; production default 4096).
    start_id : int
        Inclusive structural-ID start (L4 requires ``0``).
    max_ids : int or None
        Optional exclusive upper bound on planned structural IDs.
    acknowledge_large_run : bool
        When ``False``, preflight blocks oversized builds.
    maximum_expected_ids : int
        Soft structural-ID limit requiring acknowledgement.
    maximum_expected_bytes : int
        Soft storage limit requiring acknowledgement.
    compatible_problem_ids : tuple of str
        Consumer problem ids this library may serve. Empty means
        ``(build problem_id,)`` at seal time.
    """

    rows_per_shard: int = 4096
    start_id: int = 0
    max_ids: int | None = None
    acknowledge_large_run: bool = False
    maximum_expected_ids: int = 100_000
    maximum_expected_bytes: int = 50_000_000_000
    compatible_problem_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.rows_per_shard < 1:
            raise ConfigError(
                "structural_library.rows_per_shard must be >= 1, "
                f"got {self.rows_per_shard}"
            )
        if self.start_id < 0:
            raise ConfigError(
                f"structural_library.start_id must be >= 0, got {self.start_id}"
            )
        if self.max_ids is not None and self.max_ids < 1:
            raise ConfigError(
                "structural_library.max_ids must be >= 1 when set, "
                f"got {self.max_ids}"
            )
        if self.maximum_expected_ids < 1:
            raise ConfigError(
                "structural_library.maximum_expected_ids must be >= 1, "
                f"got {self.maximum_expected_ids}"
            )
        if self.maximum_expected_bytes < 1:
            raise ConfigError(
                "structural_library.maximum_expected_bytes must be >= 1, "
                f"got {self.maximum_expected_bytes}"
            )


@dataclass(frozen=True)
class StructuralArtifactsSection:
    """
    Shared structural-artifact backend settings from ``[structural_artifacts]``.

    Attributes
    ----------
    mode : str
        ``\"compute\"`` (default scientific path) or ``\"library\"`` (strict
        sealed Arrow lookup).
    library_path : str or None
        Filesystem path to a sealed structural library root. Required when
        ``mode=\"library\"``; forbidden when ``mode=\"compute\"``.
    allow_snapshot_hash_mismatch : bool
        Deliberate opt-in to reuse an otherwise-compatible library whose
        sealed input snapshot hash differs from the current experiment.
    """

    mode: str = "compute"
    library_path: str | None = None
    allow_snapshot_hash_mismatch: bool = False

    def __post_init__(self) -> None:
        token = str(self.mode).strip().lower()
        if token not in _STRUCTURAL_ARTIFACTS_MODES:
            raise ConfigError(
                f"structural_artifacts.mode must be one of "
                f"{sorted(_STRUCTURAL_ARTIFACTS_MODES)}, got {self.mode!r}"
            )
        object.__setattr__(self, "mode", token)
        if self.mode == "compute":
            if self.library_path is not None and str(self.library_path).strip():
                raise ConfigError(
                    "structural_artifacts.library_path is only allowed when "
                    "mode='library'"
                )
            if self.allow_snapshot_hash_mismatch:
                raise ConfigError(
                    "structural_artifacts.allow_snapshot_hash_mismatch is only "
                    "allowed when mode='library'"
                )
            object.__setattr__(self, "library_path", None)
        else:
            if self.library_path is None or not str(self.library_path).strip():
                object.__setattr__(self, "library_path", None)
            else:
                object.__setattr__(self, "library_path", str(self.library_path).strip())


@dataclass(frozen=True)
class SensitivityAnalysisSection:
    """
    Sobol sensitivity settings from ``[sensitivity_analysis]``.

    Attributes
    ----------
    sample_size : int
        Power-of-two radial block count ``N`` (>= 2).
    factor_names : tuple of str
        Ordered selected problem variables (role ``var``).
    response_key : str
        Scalar response column (``y`` or ``y_p``).
    scramble : bool
        Whether SALib scrambles the Sobol sequence.
    seed : int
        Unsigned 32-bit SALib sample seed.
    blocks_per_shard : int
        Complete radial blocks per shard (>= 1).
    parquet_batch_rows : int
        Writer batch size (>= 1).
    compression : str
        Parquet compression codec.
    bootstrap_resamples : int
        SALib ``num_resamples`` (always present; ignored for publication when
        ``compute_bootstrap_ci`` is false).
    bootstrap_confidence_level : float
        SALib ``conf_level`` in ``(0, 1)``.
    bootstrap_seed : int
        SALib bootstrap seed.
    compute_bootstrap_ci : bool
        When true, publish SALib confidence half-widths.
    acknowledge_large_run : bool
        When false, preflight blocks runs over configured maxima.
    maximum_expected_evaluations : int
        Soft evaluation-count limit requiring acknowledgement (>= 1).
    maximum_expected_bytes : int
        Soft storage limit requiring acknowledgement (>= 1).
    """

    sample_size: int
    factor_names: tuple[str, ...]
    response_key: str = "y"
    scramble: bool = True
    seed: int = 0
    blocks_per_shard: int = 16
    parquet_batch_rows: int = 512
    compression: str = "zstd"
    bootstrap_resamples: int = 1000
    bootstrap_confidence_level: float = 0.95
    bootstrap_seed: int = 1
    compute_bootstrap_ci: bool = True
    acknowledge_large_run: bool = False
    maximum_expected_evaluations: int = 100_000
    maximum_expected_bytes: int = 2_000_000_000

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sample_size, int)
            or isinstance(self.sample_size, bool)
            or self.sample_size < 2
            or self.sample_size & (self.sample_size - 1) != 0
        ):
            raise ConfigError(
                "sensitivity_analysis.sample_size must be a power of two "
                f"and at least 2, got {self.sample_size!r}"
            )
        if not self.factor_names:
            raise ConfigError(
                "sensitivity_analysis.factor_names must be a non-empty array"
            )
        if any(not isinstance(name, str) or not name.strip() for name in self.factor_names):
            raise ConfigError(
                "sensitivity_analysis.factor_names entries must be non-empty strings"
            )
        stripped = tuple(name.strip() for name in self.factor_names)
        if len(set(stripped)) != len(stripped):
            raise ConfigError(
                "sensitivity_analysis.factor_names must be unique"
            )
        object.__setattr__(self, "factor_names", stripped)
        if self.response_key not in _SENSITIVITY_RESPONSE_KEYS:
            raise ConfigError(
                "sensitivity_analysis.response_key must be one of "
                f"{sorted(_SENSITIVITY_RESPONSE_KEYS)}, got {self.response_key!r}"
            )
        if not isinstance(self.scramble, bool):
            raise ConfigError("sensitivity_analysis.scramble must be a boolean")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
            or self.seed > 0xFFFFFFFF
        ):
            raise ConfigError(
                "sensitivity_analysis.seed must be an unsigned 32-bit integer, "
                f"got {self.seed!r}"
            )
        if self.blocks_per_shard < 1:
            raise ConfigError(
                "sensitivity_analysis.blocks_per_shard must be >= 1, "
                f"got {self.blocks_per_shard}"
            )
        if self.parquet_batch_rows < 1:
            raise ConfigError(
                "sensitivity_analysis.parquet_batch_rows must be >= 1, "
                f"got {self.parquet_batch_rows}"
            )
        if self.compression not in _SENSITIVITY_COMPRESSIONS:
            raise ConfigError(
                "sensitivity_analysis.compression must be one of "
                f"{sorted(_SENSITIVITY_COMPRESSIONS)}, got {self.compression!r}"
            )
        if (
            not isinstance(self.bootstrap_resamples, int)
            or isinstance(self.bootstrap_resamples, bool)
            or self.bootstrap_resamples < 2
        ):
            raise ConfigError(
                "sensitivity_analysis.bootstrap_resamples must be an integer "
                f">= 2, got {self.bootstrap_resamples!r}"
            )
        if (
            isinstance(self.bootstrap_confidence_level, bool)
            or not isinstance(self.bootstrap_confidence_level, (int, float))
            or not math.isfinite(float(self.bootstrap_confidence_level))
            or not (0.0 < float(self.bootstrap_confidence_level) < 1.0)
        ):
            raise ConfigError(
                "sensitivity_analysis.bootstrap_confidence_level must be a "
                f"finite number in (0, 1), got {self.bootstrap_confidence_level!r}"
            )
        object.__setattr__(
            self,
            "bootstrap_confidence_level",
            float(self.bootstrap_confidence_level),
        )
        if (
            not isinstance(self.bootstrap_seed, int)
            or isinstance(self.bootstrap_seed, bool)
        ):
            raise ConfigError(
                "sensitivity_analysis.bootstrap_seed must be an integer, "
                f"got {self.bootstrap_seed!r}"
            )
        if not isinstance(self.compute_bootstrap_ci, bool):
            raise ConfigError(
                "sensitivity_analysis.compute_bootstrap_ci must be a boolean"
            )
        if not isinstance(self.acknowledge_large_run, bool):
            raise ConfigError(
                "sensitivity_analysis.acknowledge_large_run must be a boolean"
            )
        if self.maximum_expected_evaluations < 1:
            raise ConfigError(
                "sensitivity_analysis.maximum_expected_evaluations must be >= 1, "
                f"got {self.maximum_expected_evaluations}"
            )
        if self.maximum_expected_bytes < 1:
            raise ConfigError(
                "sensitivity_analysis.maximum_expected_bytes must be >= 1, "
                f"got {self.maximum_expected_bytes}"
            )


@dataclass(frozen=True)
class ExecutionSection:
    """
    Backend and resource settings shared by local and Slurm runners.

    Attributes
    ----------
    backend : str
        ``\"local\"`` or ``\"slurm\"``.
    max_parallel_tasks : int
        Local concurrency / intended parallel slots (>= 1).
    timeout_seconds : float or None
        Optional per-task wall-time limit in seconds.
    cpus_per_task : int
        CPUs requested or expected per task (>= 1).
    thread_environment : Mapping[str, str]
        Environment variables capping BLAS/OpenMP threads in workers.
    local : LocalBackendConfig
        Local interpreter settings.
    slurm : SlurmBackendConfig
        Slurm array settings.
    """

    backend: str
    max_parallel_tasks: int
    timeout_seconds: float | None
    cpus_per_task: int
    thread_environment: Mapping[str, str]
    local: LocalBackendConfig
    slurm: SlurmBackendConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "thread_environment", dict(self.thread_environment))
        if self.max_parallel_tasks < 1:
            raise ConfigError(
                f"execution.max_parallel_tasks must be >= 1, "
                f"got {self.max_parallel_tasks}"
            )
        if self.cpus_per_task < 1:
            raise ConfigError(
                f"execution.cpus_per_task must be >= 1, got {self.cpus_per_task}"
            )
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ConfigError(
                "execution.timeout_seconds must be > 0 when set, "
                f"got {self.timeout_seconds}"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Fully validated experiment TOML document.

    Attributes
    ----------
    schema_version : int
        Configuration schema version (currently ``1`` only).
    experiment : ExperimentSection
        Experiment identity and task-producing definitions.
    execution : ExecutionSection
        Backend and resource settings.
    brute_force : BruteForceSection or None
        Brute-force settings when ``workflow = \"brute_force\"``; otherwise
        ``None``.
    sampling : SamplingSection or None
        Sampling settings when ``workflow = \"sampling\"``; otherwise ``None``.
    sensitivity_analysis : SensitivityAnalysisSection or None
        Sensitivity settings when ``workflow = \"sensitivity_analysis\"``;
        otherwise ``None``.
    structural_library : StructuralLibrarySection or None
        Structural-library settings when ``workflow = \"structural_library\"``;
        otherwise ``None``.
    structural_artifacts : StructuralArtifactsSection
        Shared compute/library backend settings (default ``mode=\"compute\"``).
    source_path : Path or None
        Absolute path of the loaded TOML file when loaded from disk.
    """

    schema_version: int
    experiment: ExperimentSection
    execution: ExecutionSection
    brute_force: BruteForceSection | None = None
    sampling: SamplingSection | None = None
    sensitivity_analysis: SensitivityAnalysisSection | None = None
    structural_library: StructuralLibrarySection | None = None
    structural_artifacts: StructuralArtifactsSection = field(
        default_factory=StructuralArtifactsSection
    )
    source_path: Path | None = None


def _parse_brute_force(raw: Any) -> BruteForceSection:
    """
    Parse ``[brute_force]`` into a frozen section.

    Parameters
    ----------
    raw : Any
        TOML table.

    Returns
    -------
    BruteForceSection
        Validated brute-force settings.
    """
    table = _require_mapping(raw, context="brute_force")
    _reject_unknown_keys(table, _BRUTE_FORCE_KEYS, context="brute_force")

    def _positive_int(key: str, default: int) -> int:
        value = table.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"brute_force.{key} must be an integer")
        return value

    compression = table.get("compression", "zstd")
    if not isinstance(compression, str) or not compression.strip():
        raise ConfigError("brute_force.compression must be a non-empty string")
    acknowledge = table.get("acknowledge_large_run", False)
    if not isinstance(acknowledge, bool):
        raise ConfigError("brute_force.acknowledge_large_run must be a boolean")
    reuse = table.get("reuse_structural_artifacts", True)
    if not isinstance(reuse, bool):
        raise ConfigError("brute_force.reuse_structural_artifacts must be a boolean")
    start_index_raw = table.get("start_index", 0)
    if not isinstance(start_index_raw, int) or isinstance(start_index_raw, bool):
        raise ConfigError("brute_force.start_index must be an integer")
    max_rows_raw = table.get("max_rows", None)
    if max_rows_raw is not None and (
        not isinstance(max_rows_raw, int) or isinstance(max_rows_raw, bool)
    ):
        raise ConfigError("brute_force.max_rows must be an integer when set")

    return BruteForceSection(
        rows_per_shard=_positive_int("rows_per_shard", 10_000),
        parquet_batch_rows=_positive_int("parquet_batch_rows", 4096),
        compression=compression.strip(),
        acknowledge_large_run=acknowledge,
        maximum_expected_rows=_positive_int("maximum_expected_rows", 5_000_000),
        maximum_expected_bytes=_positive_int(
            "maximum_expected_bytes", 50_000_000_000
        ),
        reuse_structural_artifacts=reuse,
        start_index=int(start_index_raw),
        max_rows=None if max_rows_raw is None else int(max_rows_raw),
    )


def _validate_sampling_explicit_seeds(seeds: tuple[int, ...]) -> None:
    """
    Fail closed on empty, non-uint32, or duplicate explicit scramble seeds.
    """
    if not seeds:
        raise ConfigError("sampling.explicit_seeds must be a non-empty list")
    seen: set[int] = set()
    for item in seeds:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ConfigError(
                "sampling.explicit_seeds entries must be integers, "
                f"got {item!r}"
            )
        if item < 0 or item > 0xFFFFFFFF:
            raise ConfigError(
                "sampling.explicit_seeds entries must be unsigned 32-bit "
                f"integers, got {item}"
            )
        if item in seen:
            raise ConfigError(
                f"sampling.explicit_seeds contains duplicate seed {item}"
            )
        seen.add(item)


def _parse_sampling_explicit_seeds(raw: Any) -> tuple[int, ...]:
    """
    Parse ``sampling.explicit_seeds`` as a unique uint32 tuple.
    """
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise ConfigError("sampling.explicit_seeds must be a list of integers")
    seeds = tuple(raw)
    _validate_sampling_explicit_seeds(seeds)
    return seeds


def _parse_sampling(raw: Any) -> SamplingSection:
    """
    Parse ``[sampling]`` into a frozen section.
    """
    table = _require_mapping(raw, context="sampling")
    _reject_unknown_keys(table, _SAMPLING_KEYS, context="sampling")

    if "sample_size" not in table:
        raise ConfigError("sampling is missing required key(s): ['sample_size']")
    sample_size = table["sample_size"]
    if not isinstance(sample_size, int) or isinstance(sample_size, bool):
        raise ConfigError("sampling.sample_size must be an integer")

    method = table.get("method", "scipy_qmc_sobol")
    if not isinstance(method, str) or not method.strip():
        raise ConfigError("sampling.method must be a non-empty string")

    scramble = table.get("scramble", True)
    if not isinstance(scramble, bool):
        raise ConfigError("sampling.scramble must be a boolean")
    delete_duplicates = table.get("delete_duplicates", False)
    if not isinstance(delete_duplicates, bool):
        raise ConfigError("sampling.delete_duplicates must be a boolean")

    def _positive_int(key: str, default: int) -> int:
        value = table.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"sampling.{key} must be an integer")
        return value

    compression = table.get("compression", "zstd")
    if not isinstance(compression, str) or not compression.strip():
        raise ConfigError("sampling.compression must be a non-empty string")
    acknowledge = table.get("acknowledge_large_run", False)
    if not isinstance(acknowledge, bool):
        raise ConfigError("sampling.acknowledge_large_run must be a boolean")
    explicit_raw = table.get("explicit_seeds")
    explicit_seeds = (
        None
        if explicit_raw is None
        else _parse_sampling_explicit_seeds(explicit_raw)
    )

    return SamplingSection(
        method=method.strip(),
        sample_size=int(sample_size),
        scramble=scramble,
        delete_duplicates=delete_duplicates,
        rows_per_shard=_positive_int("rows_per_shard", 512),
        parquet_batch_rows=_positive_int("parquet_batch_rows", 512),
        compression=compression.strip(),
        acknowledge_large_run=acknowledge,
        maximum_expected_evaluations=_positive_int(
            "maximum_expected_evaluations", 100_000
        ),
        maximum_expected_bytes=_positive_int(
            "maximum_expected_bytes", 2_000_000_000
        ),
        explicit_seeds=explicit_seeds,
    )


def _parse_structural_library(raw: Any) -> StructuralLibrarySection:
    """
    Parse ``[structural_library]`` into a frozen section.
    """
    table = _require_mapping(raw, context="structural_library")
    _reject_unknown_keys(table, _STRUCTURAL_LIBRARY_KEYS, context="structural_library")

    def _positive_int(key: str, default: int) -> int:
        value = table.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"structural_library.{key} must be an integer")
        return value

    acknowledge = table.get("acknowledge_large_run", False)
    if not isinstance(acknowledge, bool):
        raise ConfigError("structural_library.acknowledge_large_run must be a boolean")
    start_id_raw = table.get("start_id", 0)
    if not isinstance(start_id_raw, int) or isinstance(start_id_raw, bool):
        raise ConfigError("structural_library.start_id must be an integer")
    max_ids_raw = table.get("max_ids", None)
    if max_ids_raw is not None and (
        not isinstance(max_ids_raw, int) or isinstance(max_ids_raw, bool)
    ):
        raise ConfigError("structural_library.max_ids must be an integer when set")

    compatible_raw = table.get("compatible_problem_ids", ())
    if compatible_raw is None:
        compatible_ids: tuple[str, ...] = ()
    elif isinstance(compatible_raw, str):
        raise ConfigError(
            "structural_library.compatible_problem_ids must be a list of strings"
        )
    elif isinstance(compatible_raw, (list, tuple)):
        compatible_ids = tuple(str(item) for item in compatible_raw)
    else:
        raise ConfigError(
            "structural_library.compatible_problem_ids must be a list of strings"
        )

    return StructuralLibrarySection(
        rows_per_shard=_positive_int("rows_per_shard", 4096),
        start_id=int(start_id_raw),
        max_ids=None if max_ids_raw is None else int(max_ids_raw),
        acknowledge_large_run=acknowledge,
        maximum_expected_ids=_positive_int("maximum_expected_ids", 100_000),
        maximum_expected_bytes=_positive_int(
            "maximum_expected_bytes", 50_000_000_000
        ),
        compatible_problem_ids=compatible_ids,
    )


def _parse_structural_artifacts(raw: Any) -> StructuralArtifactsSection:
    """
    Parse optional ``[structural_artifacts]`` into a frozen section.
    """
    table = _require_mapping(raw, context="structural_artifacts")
    _reject_unknown_keys(
        table, _STRUCTURAL_ARTIFACTS_KEYS, context="structural_artifacts"
    )
    mode = table.get("mode", "compute")
    if not isinstance(mode, str) or not mode.strip():
        raise ConfigError("structural_artifacts.mode must be a non-empty string")
    library_path = table.get("library_path", None)
    if library_path is not None and not isinstance(library_path, str):
        raise ConfigError("structural_artifacts.library_path must be a string when set")
    allow_mismatch = table.get("allow_snapshot_hash_mismatch", False)
    if not isinstance(allow_mismatch, bool):
        raise ConfigError(
            "structural_artifacts.allow_snapshot_hash_mismatch must be a boolean"
        )
    return StructuralArtifactsSection(
        mode=mode,
        library_path=library_path,
        allow_snapshot_hash_mismatch=allow_mismatch,
    )


def _parse_sensitivity_analysis(raw: Any) -> SensitivityAnalysisSection:
    """
    Parse ``[sensitivity_analysis]`` into a frozen section.

    Parameters
    ----------
    raw : Any
        TOML table.

    Returns
    -------
    SensitivityAnalysisSection
        Validated sensitivity settings.
    """
    table = _require_mapping(raw, context="sensitivity_analysis")
    _reject_unknown_keys(table, _SENSITIVITY_KEYS, context="sensitivity_analysis")

    required = (
        "sample_size",
        "factor_names",
        "seed",
        "blocks_per_shard",
        "parquet_batch_rows",
        "bootstrap_resamples",
        "bootstrap_confidence_level",
        "bootstrap_seed",
        "maximum_expected_evaluations",
        "maximum_expected_bytes",
    )
    missing = [key for key in required if key not in table]
    if missing:
        raise ConfigError(
            "sensitivity_analysis is missing required key(s): "
            f"{missing}"
        )

    sample_size = table["sample_size"]
    if not isinstance(sample_size, int) or isinstance(sample_size, bool):
        raise ConfigError("sensitivity_analysis.sample_size must be an integer")

    factor_names_raw = table["factor_names"]
    if not isinstance(factor_names_raw, list):
        raise ConfigError(
            "sensitivity_analysis.factor_names must be an array of strings"
        )
    factor_names = tuple(factor_names_raw)

    response_key = table.get("response_key", "y")
    if not isinstance(response_key, str) or not response_key.strip():
        raise ConfigError(
            "sensitivity_analysis.response_key must be a non-empty string"
        )

    scramble = table.get("scramble", True)
    if not isinstance(scramble, bool):
        raise ConfigError("sensitivity_analysis.scramble must be a boolean")

    seed = table["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError("sensitivity_analysis.seed must be an integer")

    def require_int(key: str) -> int:
        value = table[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"sensitivity_analysis.{key} must be an integer")
        return int(value)

    compression = table.get("compression", "zstd")
    if not isinstance(compression, str) or not compression.strip():
        raise ConfigError(
            "sensitivity_analysis.compression must be a non-empty string"
        )

    bootstrap_confidence_level = table["bootstrap_confidence_level"]
    if isinstance(bootstrap_confidence_level, bool) or not isinstance(
        bootstrap_confidence_level, (int, float)
    ):
        raise ConfigError(
            "sensitivity_analysis.bootstrap_confidence_level must be a number"
        )

    compute_ci = table.get("compute_bootstrap_ci", True)
    if not isinstance(compute_ci, bool):
        raise ConfigError(
            "sensitivity_analysis.compute_bootstrap_ci must be a boolean"
        )

    acknowledge = table.get("acknowledge_large_run", False)
    if not isinstance(acknowledge, bool):
        raise ConfigError(
            "sensitivity_analysis.acknowledge_large_run must be a boolean"
        )

    return SensitivityAnalysisSection(
        sample_size=int(sample_size),
        factor_names=factor_names,
        response_key=response_key.strip(),
        scramble=scramble,
        seed=int(seed),
        blocks_per_shard=require_int("blocks_per_shard"),
        parquet_batch_rows=require_int("parquet_batch_rows"),
        compression=compression.strip(),
        bootstrap_resamples=require_int("bootstrap_resamples"),
        bootstrap_confidence_level=float(bootstrap_confidence_level),
        bootstrap_seed=require_int("bootstrap_seed"),
        compute_bootstrap_ci=compute_ci,
        acknowledge_large_run=acknowledge,
        maximum_expected_evaluations=require_int("maximum_expected_evaluations"),
        maximum_expected_bytes=require_int("maximum_expected_bytes"),
    )


def _parse_algorithms(raw: Any) -> tuple[AlgorithmConfig, ...]:
    """
    Parse ``[[experiment.algorithms]]`` into frozen algorithm configs.

    Parameters
    ----------
    raw : Any
        TOML array of algorithm tables.

    Returns
    -------
    tuple of AlgorithmConfig
        Ordered algorithm definitions.

    Raises
    ------
    ConfigError
        If the array is missing/empty, keys are unknown, names duplicate, or
        ``max_evaluations`` is invalid.
    """
    if not isinstance(raw, list) or not raw:
        raise ConfigError("experiment.algorithms must be a non-empty array of tables")
    algorithms: list[AlgorithmConfig] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(raw):
        table = _require_mapping(entry, context=f"experiment.algorithms[{index}]")
        _reject_unknown_keys(
            table,
            _ALGORITHM_KEYS,
            context=f"experiment.algorithms[{index}]",
        )
        name = _require_nonempty_str(
            table.get("name"),
            context=f"experiment.algorithms[{index}].name",
        )
        if name in seen_names:
            raise ConfigError(
                f"duplicate task-producing algorithm definition: {name!r}"
            )
        seen_names.add(name)
        max_evaluations = table.get("max_evaluations")
        if not isinstance(max_evaluations, int) or isinstance(max_evaluations, bool):
            raise ConfigError(
                f"experiment.algorithms[{index}].max_evaluations must be an integer"
            )
        if "parameters" in table:
            parameters = _require_mapping(
                table["parameters"],
                context=f"experiment.algorithms[{index}].parameters",
            )
        else:
            parameters = {}
        algorithms.append(
            AlgorithmConfig(
                name=name,
                max_evaluations=max_evaluations,
                parameters=parameters,
                repetitions=table.get("repetitions"),
            )
        )
    return tuple(algorithms)


def _parse_repetitions(raw: Any) -> RepetitionsConfig:
    """
    Parse ``[experiment.repetitions]``.

    Parameters
    ----------
    raw : Any
        TOML table.

    Returns
    -------
    RepetitionsConfig
        Validated repetition settings.
    """
    table = _require_mapping(raw, context="experiment.repetitions")
    _reject_unknown_keys(table, _REPETITIONS_KEYS, context="experiment.repetitions")
    count = table.get("count")
    seed_base = table.get("seed_base")
    if not isinstance(count, int) or isinstance(count, bool):
        raise ConfigError("experiment.repetitions.count must be an integer")
    if not isinstance(seed_base, int) or isinstance(seed_base, bool):
        raise ConfigError("experiment.repetitions.seed_base must be an integer")
    manifest = table.get("seed_manifest_path")
    if manifest is not None and (not isinstance(manifest, str) or not manifest.strip()):
        raise ConfigError(
            "experiment.repetitions.seed_manifest_path must be a non-empty path"
        )
    return RepetitionsConfig(
        count=count,
        seed_base=seed_base,
        seed_manifest_path=None if manifest is None else str(manifest),
    )


def _require_finite_nonnegative_number(value: Any, *, context: str) -> float:
    """
    Validate a finite numeric value that is not a bool and is >= 0.

    Parameters
    ----------
    value : Any
        Candidate numeric value.
    context : str
        Human-readable location used in the error message.

    Returns
    -------
    float
        Coerced finite non-negative float.

    Raises
    ------
    ConfigError
        If the value is a bool, non-numeric, non-finite, or negative.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ConfigError(f"{context} must be a finite number >= 0, got {value!r}")
    return number


def _parse_termination(
    raw: Any,
    *,
    workflow: str,
    problem_ids: tuple[str, ...],
) -> TerminationConfig:
    """
    Parse optional ``[experiment.termination]``.

    Parameters
    ----------
    raw : Any
        TOML table, or ``None`` when the table is omitted.
    workflow : str
        Already-validated workflow name.
    problem_ids : tuple of str
        Selected ordinary problem IDs.

    Returns
    -------
    TerminationConfig
        Validated termination settings (fixed-budget defaults when omitted).

    Raises
    ------
    ConfigError
        If the table is invalid or used outside ``optimization_runs``.
    """
    if raw is None:
        return TerminationConfig()

    if workflow != "optimization_runs":
        raise ConfigError(
            "experiment.termination is only allowed when "
            "workflow='optimization_runs'"
        )

    table = _require_mapping(raw, context="experiment.termination")
    _reject_unknown_keys(table, _TERMINATION_KEYS, context="experiment.termination")

    if "mode" not in table:
        raise ConfigError("experiment.termination.mode is required when the table is set")
    mode = _require_nonempty_str(table.get("mode"), context="experiment.termination.mode")
    if mode not in _TERMINATION_MODES:
        raise ConfigError(
            f"experiment.termination.mode must be one of "
            f"{sorted(_TERMINATION_MODES)}, got {mode!r}"
        )

    if mode == "fixed_budget":
        forbidden = (
            "global_optimum",
            "global_optima",
            "allowed_deviation_percent",
            "target_reference",
        )
        present = [key for key in forbidden if key in table]
        if present:
            raise ConfigError(
                "experiment.termination forbids "
                f"{present} when mode='fixed_budget'"
            )
        return TerminationConfig(mode="fixed_budget")

    if "allowed_deviation_percent" not in table:
        raise ConfigError(
            "experiment.termination is missing required key(s) for "
            "mode='fixed_target': ['allowed_deviation_percent']"
        )
    allowed_deviation_percent = _require_finite_nonnegative_number(
        table["allowed_deviation_percent"],
        context="experiment.termination.allowed_deviation_percent",
    )
    has_scalar = "global_optimum" in table
    has_mapping = "global_optima" in table
    if has_scalar == has_mapping:
        raise ConfigError(
            "fixed_target requires exactly one of "
            "experiment.termination.global_optimum or "
            "experiment.termination.global_optima"
        )

    global_optimum: float | None = None
    global_optima: dict[str, float] = {}
    if has_scalar:
        if len(problem_ids) != 1:
            raise ConfigError(
                "multi-problem fixed_target requires "
                "experiment.termination.global_optima"
            )
        global_optimum = _require_finite_nonnegative_number(
            table["global_optimum"],
            context="experiment.termination.global_optimum",
        )
    else:
        raw_optima = _require_mapping(
            table["global_optima"],
            context="experiment.termination.global_optima",
        )
        for problem_id, value in raw_optima.items():
            if not isinstance(problem_id, str) or not problem_id.strip():
                raise ConfigError(
                    "experiment.termination.global_optima keys must be "
                    "non-empty strings"
                )
            global_optima[problem_id] = _require_finite_nonnegative_number(
                value,
                context=(
                    "experiment.termination.global_optima"
                    f"[{problem_id!r}]"
                ),
            )
        selected = set(problem_ids)
        mapped = set(global_optima)
        missing = sorted(selected - mapped)
        extra = sorted(mapped - selected)
        if missing or extra:
            raise ConfigError(
                "experiment.termination.global_optima does not match "
                f"problem_ids: missing={missing!r}; extra={extra!r}"
            )

    if "target_reference" in table:
        target_reference = _require_nonempty_str(
            table.get("target_reference"),
            context="experiment.termination.target_reference",
        )
        if target_reference == "feasible":
            target_reference = "all"
        if target_reference not in _TARGET_REFERENCES:
            raise ConfigError(
                "experiment.termination.target_reference must be one of "
                f"{sorted(_TARGET_REFERENCES)}, got {target_reference!r}"
            )
    else:
        target_reference = "all"

    termination = TerminationConfig(
        mode="fixed_target",
        global_optimum=global_optimum,
        global_optima=global_optima,
        allowed_deviation_percent=allowed_deviation_percent,
        target_reference=target_reference,
    )
    return termination


def _parse_problem_ids(raw: Any) -> tuple[str, ...]:
    """
    Parse and validate explicit ordinary problem IDs.

    Parameters
    ----------
    raw : Any
        TOML array of problem ID strings.

    Returns
    -------
    tuple of str
        Deduplicated-order-preserving problem IDs.

    Raises
    ------
    ConfigError
        If the list is empty, contains blanks, includes ``default``, or has
        duplicates.
    """
    if not isinstance(raw, list) or not raw:
        raise ConfigError("experiment.problem_ids must be a non-empty array of strings")
    ids: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(
                f"experiment.problem_ids[{index}] must be a non-empty string"
            )
        problem_id = item.strip()
        if problem_id == "default":
            raise ConfigError(
                "experiment.problem_ids cannot include the reserved id 'default'"
            )
        if problem_id in seen:
            raise ConfigError(
                f"duplicate task-producing problem definition: {problem_id!r}"
            )
        seen.add(problem_id)
        ids.append(problem_id)
    return tuple(ids)


def _parse_thread_environment(raw: Any) -> dict[str, str]:
    """
    Parse ``[execution.thread_environment]`` as string-valued env vars.

    Parameters
    ----------
    raw : Any
        TOML table or ``None``.

    Returns
    -------
    dict[str, str]
        Environment variable mapping.
    """
    if raw is None:
        return {}
    table = _require_mapping(raw, context="execution.thread_environment")
    out: dict[str, str] = {}
    for key, value in table.items():
        if not isinstance(key, str) or not key:
            raise ConfigError(
                "execution.thread_environment keys must be non-empty strings"
            )
        if not isinstance(value, str):
            raise ConfigError(
                f"execution.thread_environment[{key!r}] must be a string"
            )
        out[key] = value
    return out


def _parse_local(raw: Any) -> LocalBackendConfig:
    """
    Parse ``[execution.local]``.

    Parameters
    ----------
    raw : Any
        TOML table or ``None``.

    Returns
    -------
    LocalBackendConfig
        Local backend settings.
    """
    if raw is None:
        return LocalBackendConfig()
    table = _require_mapping(raw, context="execution.local")
    _reject_unknown_keys(table, _LOCAL_KEYS, context="execution.local")
    executable = table.get("python_executable", "")
    if not isinstance(executable, str):
        raise ConfigError("execution.local.python_executable must be a string")
    return LocalBackendConfig(python_executable=executable)


def _parse_slurm(raw: Any) -> SlurmBackendConfig:
    """
    Parse ``[execution.slurm]``.

    Parameters
    ----------
    raw : Any
        TOML table or ``None``.

    Returns
    -------
    SlurmBackendConfig
        Slurm backend settings (may still be incomplete until backend check).
    """
    if raw is None:
        return SlurmBackendConfig()
    table = _require_mapping(raw, context="execution.slurm")
    _reject_unknown_keys(table, _SLURM_KEYS, context="execution.slurm")

    def optional_str(key: str, default: str) -> str:
        value = table.get(key, default)
        if not isinstance(value, str):
            raise ConfigError(f"execution.slurm.{key} must be a string")
        return value

    def require_int(key: str, default: int) -> int:
        value = table.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"execution.slurm.{key} must be an integer")
        return value

    return SlurmBackendConfig(
        python_executable=optional_str("python_executable", ""),
        partition=optional_str("partition", ""),
        account=optional_str("account", ""),
        memory=optional_str("memory", "8G"),
        time_limit=optional_str("time_limit", "00:45:00"),
        array_concurrency=require_int("array_concurrency", 64),
        maximum_array_size=require_int("maximum_array_size", 1000),
    )


def parse_experiment_config(
    data: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> ExperimentConfig:
    """
    Validate a parsed TOML mapping into an ``ExperimentConfig``.

    Parameters
    ----------
    data : Mapping[str, Any]
        Root TOML table.
    source_path : Path or None, optional
        Absolute path recorded on the returned config when loaded from disk.

    Returns
    -------
    ExperimentConfig
        Immutable validated configuration.

    Raises
    ------
    ConfigError
        If schema, keys, workflow/backend, or numeric fields are invalid.
    """
    table = dict(data)
    _reject_unknown_keys(table, _TOP_KEYS, context="root")

    schema_version = table.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ConfigError(
            f"unsupported schema_version {schema_version!r}; "
            f"supported: {SUPPORTED_SCHEMA_VERSION}"
        )

    experiment_raw = _require_mapping(
        table.get("experiment"),
        context="experiment",
    )
    _reject_unknown_keys(experiment_raw, _EXPERIMENT_KEYS, context="experiment")

    experiment_id = _require_nonempty_str(
        experiment_raw.get("id"),
        context="experiment.id",
    )
    workflow = _require_nonempty_str(
        experiment_raw.get("workflow"),
        context="experiment.workflow",
    )
    if workflow not in SUPPORTED_WORKFLOWS:
        raise ConfigError(
            f"unknown workflow {workflow!r}; "
            f"supported: {sorted(SUPPORTED_WORKFLOWS)}"
        )
    slab_type = _require_nonempty_str(
        experiment_raw.get("slab_type"),
        context="experiment.slab_type",
    )
    output_root = _require_nonempty_str(
        experiment_raw.get("output_root"),
        context="experiment.output_root",
    )
    problem_ids = _parse_problem_ids(experiment_raw.get("problem_ids"))
    sampling: SamplingSection | None = None

    if workflow == "brute_force":
        if "algorithms" in experiment_raw:
            raise ConfigError(
                "experiment.algorithms is not allowed when workflow='brute_force'"
            )
        if "repetitions" in experiment_raw:
            raise ConfigError(
                "experiment.repetitions is not allowed when workflow='brute_force'"
            )
        if "termination" in experiment_raw:
            raise ConfigError(
                "experiment.termination is not allowed when workflow='brute_force'"
            )
        if "sensitivity_analysis" in table:
            raise ConfigError(
                "[sensitivity_analysis] is only allowed when "
                "workflow='sensitivity_analysis'"
            )
        if "structural_library" in table:
            raise ConfigError(
                "[structural_library] is only allowed when "
                "workflow='structural_library'"
            )
        if "sampling" in table:
            raise ConfigError(
                "[sampling] is only allowed when workflow='sampling'"
            )
        if len(problem_ids) != 1:
            raise ConfigError(
                "brute_force currently supports exactly one problem_id, "
                f"got {list(problem_ids)!r}"
            )
        algorithms: tuple[AlgorithmConfig, ...] = ()
        repetitions: RepetitionsConfig | None = None
        termination = TerminationConfig()
        if "brute_force" not in table:
            raise ConfigError(
                "workflow='brute_force' requires a [brute_force] configuration table"
            )
        brute_force = _parse_brute_force(table.get("brute_force"))
        sensitivity_analysis: SensitivityAnalysisSection | None = None
        structural_library: StructuralLibrarySection | None = None
    elif workflow == "sensitivity_analysis":
        if "algorithms" in experiment_raw:
            raise ConfigError(
                "experiment.algorithms is not allowed when "
                "workflow='sensitivity_analysis'"
            )
        if "repetitions" in experiment_raw:
            raise ConfigError(
                "experiment.repetitions is not allowed when "
                "workflow='sensitivity_analysis'"
            )
        if "termination" in experiment_raw:
            raise ConfigError(
                "experiment.termination is not allowed when "
                "workflow='sensitivity_analysis'"
            )
        if "brute_force" in table:
            raise ConfigError(
                "[brute_force] is only allowed when workflow='brute_force'"
            )
        if "structural_library" in table:
            raise ConfigError(
                "[structural_library] is only allowed when "
                "workflow='structural_library'"
            )
        if "sampling" in table:
            raise ConfigError(
                "[sampling] is only allowed when workflow='sampling'"
            )
        if len(problem_ids) != 1:
            raise ConfigError(
                "sensitivity_analysis requires exactly one problem_id, "
                f"got {list(problem_ids)!r}"
            )
        algorithms = ()
        repetitions = None
        termination = TerminationConfig()
        brute_force = None
        if "sensitivity_analysis" not in table:
            raise ConfigError(
                "workflow='sensitivity_analysis' requires a "
                "[sensitivity_analysis] configuration table"
            )
        sensitivity_analysis = _parse_sensitivity_analysis(
            table.get("sensitivity_analysis")
        )
        structural_library = None
    elif workflow == "structural_library":
        if "algorithms" in experiment_raw:
            raise ConfigError(
                "experiment.algorithms is not allowed when "
                "workflow='structural_library'"
            )
        if "repetitions" in experiment_raw:
            raise ConfigError(
                "experiment.repetitions is not allowed when "
                "workflow='structural_library'"
            )
        if "termination" in experiment_raw:
            raise ConfigError(
                "experiment.termination is not allowed when "
                "workflow='structural_library'"
            )
        if "brute_force" in table:
            raise ConfigError(
                "[brute_force] is only allowed when workflow='brute_force'"
            )
        if "sensitivity_analysis" in table:
            raise ConfigError(
                "[sensitivity_analysis] is only allowed when "
                "workflow='sensitivity_analysis'"
            )
        if "sampling" in table:
            raise ConfigError(
                "[sampling] is only allowed when workflow='sampling'"
            )
        if len(problem_ids) != 1:
            raise ConfigError(
                "structural_library requires exactly one problem_id, "
                f"got {list(problem_ids)!r}"
            )
        if slab_type != "hp_slab":
            raise ConfigError(
                "structural_library currently supports slab_type='hp_slab' only, "
                f"got {slab_type!r}"
            )
        algorithms = ()
        repetitions = None
        termination = TerminationConfig()
        brute_force = None
        sensitivity_analysis = None
        if "structural_library" not in table:
            raise ConfigError(
                "workflow='structural_library' requires a "
                "[structural_library] configuration table"
            )
        structural_library = _parse_structural_library(table.get("structural_library"))
    elif workflow == "sampling":
        if "algorithms" in experiment_raw:
            raise ConfigError(
                "experiment.algorithms is not allowed when workflow='sampling'"
            )
        if "termination" in experiment_raw:
            raise ConfigError(
                "experiment.termination is not allowed when workflow='sampling'"
            )
        if "brute_force" in table:
            raise ConfigError(
                "[brute_force] is only allowed when workflow='brute_force'"
            )
        if "sensitivity_analysis" in table:
            raise ConfigError(
                "[sensitivity_analysis] is only allowed when "
                "workflow='sensitivity_analysis'"
            )
        if "structural_library" in table:
            raise ConfigError(
                "[structural_library] is only allowed when "
                "workflow='structural_library'"
            )
        if len(problem_ids) != 1:
            raise ConfigError(
                "sampling requires exactly one problem_id, "
                f"got {list(problem_ids)!r}"
            )
        if "repetitions" not in experiment_raw:
            raise ConfigError(
                "workflow='sampling' requires [experiment.repetitions]"
            )
        if "sampling" not in table:
            raise ConfigError(
                "workflow='sampling' requires a [sampling] configuration table"
            )
        algorithms = ()
        repetitions = _parse_repetitions(experiment_raw.get("repetitions"))
        termination = TerminationConfig()
        brute_force = None
        sensitivity_analysis = None
        structural_library = None
        sampling = _parse_sampling(table.get("sampling"))
        if (
            sampling.explicit_seeds is not None
            and len(sampling.explicit_seeds) != repetitions.count
        ):
            raise ConfigError(
                "sampling.explicit_seeds length must equal "
                "experiment.repetitions.count "
                f"({repetitions.count}), got {len(sampling.explicit_seeds)}"
            )
    else:
        if "brute_force" in table:
            raise ConfigError(
                "[brute_force] is only allowed when workflow='brute_force'"
            )
        if "sensitivity_analysis" in table:
            raise ConfigError(
                "[sensitivity_analysis] is only allowed when "
                "workflow='sensitivity_analysis'"
            )
        if "structural_library" in table:
            raise ConfigError(
                "[structural_library] is only allowed when "
                "workflow='structural_library'"
            )
        if "sampling" in table:
            raise ConfigError(
                "[sampling] is only allowed when workflow='sampling'"
            )
        algorithms = _parse_algorithms(experiment_raw.get("algorithms"))
        repetitions = _parse_repetitions(experiment_raw.get("repetitions"))
        termination = _parse_termination(
            experiment_raw.get("termination"),
            workflow=workflow,
            problem_ids=problem_ids,
        )
        brute_force = None
        sensitivity_analysis = None
        structural_library = None

    execution_raw = _require_mapping(
        table.get("execution"),
        context="execution",
    )
    _reject_unknown_keys(execution_raw, _EXECUTION_KEYS, context="execution")

    backend = _require_nonempty_str(
        execution_raw.get("backend"),
        context="execution.backend",
    )
    if backend not in SUPPORTED_BACKENDS:
        raise ConfigError(
            f"unknown backend {backend!r}; "
            f"supported: {sorted(SUPPORTED_BACKENDS)}"
        )

    max_parallel_tasks = execution_raw.get("max_parallel_tasks", 1)
    if not isinstance(max_parallel_tasks, int) or isinstance(max_parallel_tasks, bool):
        raise ConfigError("execution.max_parallel_tasks must be an integer")

    timeout_raw = execution_raw.get("timeout_seconds", None)
    timeout_seconds: float | None
    if timeout_raw is None:
        timeout_seconds = None
    elif isinstance(timeout_raw, bool) or not isinstance(timeout_raw, (int, float)):
        raise ConfigError("execution.timeout_seconds must be a number when set")
    else:
        timeout_seconds = float(timeout_raw)

    cpus_per_task = execution_raw.get("cpus_per_task", 1)
    if not isinstance(cpus_per_task, int) or isinstance(cpus_per_task, bool):
        raise ConfigError("execution.cpus_per_task must be an integer")

    execution = ExecutionSection(
        backend=backend,
        max_parallel_tasks=max_parallel_tasks,
        timeout_seconds=timeout_seconds,
        cpus_per_task=cpus_per_task,
        thread_environment=_parse_thread_environment(
            execution_raw.get("thread_environment")
        ),
        local=_parse_local(execution_raw.get("local")),
        slurm=_parse_slurm(execution_raw.get("slurm")),
    )
    if backend == "slurm":
        execution.slurm.require_complete()

    if "structural_artifacts" in table:
        structural_artifacts = _parse_structural_artifacts(
            table.get("structural_artifacts")
        )
    else:
        structural_artifacts = StructuralArtifactsSection()

    if structural_artifacts.mode == "library":
        if workflow == "structural_library":
            raise ConfigError(
                "structural_artifacts.mode='library' is not allowed when "
                "workflow='structural_library' (builder consumes compute mode only)"
            )
        if workflow not in {
            "brute_force",
            "optimization_runs",
            "sensitivity_analysis",
            "sampling",
        }:
            raise ConfigError(
                "structural_artifacts.mode='library' is only supported for "
                "brute_force, optimization_runs, sensitivity_analysis, "
                "and sampling"
            )
        if slab_type != "hp_slab":
            raise ConfigError(
                "structural_artifacts.mode='library' currently supports "
                f"slab_type='hp_slab' only, got {slab_type!r}"
            )
        if len(problem_ids) != 1 and workflow != "optimization_runs":
            raise ConfigError(
                "structural_artifacts.mode='library' requires exactly one "
                "problem_id outside workflow='optimization_runs', "
                f"got {list(problem_ids)!r}"
            )
        if brute_force is not None and not brute_force.reuse_structural_artifacts:
            raise ConfigError(
                "structural_artifacts.mode='library' requires "
                "brute_force.reuse_structural_artifacts=true "
                "(reuse_structural_artifacts=false is compute-mode reference only)"
            )

    return ExperimentConfig(
        schema_version=schema_version,
        experiment=ExperimentSection(
            id=experiment_id,
            workflow=workflow,
            slab_type=slab_type,
            problem_ids=problem_ids,
            output_root=output_root,
            algorithms=algorithms,
            repetitions=repetitions,
            termination=termination,
        ),
        execution=execution,
        brute_force=brute_force,
        sampling=sampling,
        sensitivity_analysis=sensitivity_analysis,
        structural_library=structural_library,
        structural_artifacts=structural_artifacts,
        source_path=Path(source_path).resolve() if source_path is not None else None,
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """
    Load and validate an experiment TOML file.

    Parameters
    ----------
    path : str or Path
        Path to the TOML configuration file. Relative paths are resolved
        against the current working directory only for locating the file;
        the returned ``source_path`` is always absolute.

    Returns
    -------
    ExperimentConfig
        Immutable validated configuration.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ConfigError
        If the TOML contents fail validation.
    tomllib.TOMLDecodeError
        If the file is not valid TOML.
    """
    config_path = Path(path).resolve()
    with open(config_path, "rb") as handle:
        data = tomllib.load(handle)
    return parse_experiment_config(data, source_path=config_path)


def experiment_root_for(config: ExperimentConfig, *, cwd: Path | None = None) -> Path:
    """
    Resolve the experiment directory path for a validated config.

    Parameters
    ----------
    config : ExperimentConfig
        Validated configuration.
    cwd : Path or None, optional
        Base directory used when ``output_root`` is relative. Defaults to the
        process current working directory.

    Returns
    -------
    Path
        Absolute path ``{output_root}/runs/<bucket>/{experiment.id}``.
    """
    from core.execution_core.output_layout import (
        experiment_dir_for,
        layout_root_from_output_root,
    )

    base = Path(cwd) if cwd is not None else Path.cwd()
    layout_root = layout_root_from_output_root(
        config.experiment.output_root,
        cwd=base,
    )
    return experiment_dir_for(
        layout_root=layout_root,
        workflow=config.experiment.workflow,
        experiment_id=config.experiment.id,
    )
