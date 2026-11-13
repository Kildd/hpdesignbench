"""
Prepare-time sealing and worker revalidation of structural-artifact backends.

Path locates an external sealed library; hashes and ``LibraryIdentity`` identify
it scientifically. Large library parts are never copied into the experiment.
"""
from __future__ import annotations

import hashlib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from core.analysis_core.run_config import analysis_run_config_from_params
from core.execution_core.atomic_io import read_json, write_json_atomic
from core.execution_core.config import ExperimentConfig
from core.ioh_core.import_specs import make_decode
from core.ioh_core.structural_artifacts.library import open_structural_library
from core.ioh_core.structural_artifacts.models import (
    LibraryIdentity,
    StructuralLibraryError,
    sha256_file,
)
from core.ioh_core.structural_artifacts.indexing import structural_cardinality
from core.ioh_core.structural_artifacts.provider import (
    StructuralEvaluationProvider,
    build_structural_provider,
)
from core.ioh_core.workflows.brute_force.planner import (
    load_problem_space_and_schema_from_csv_dir,
)
from core.ioh_core.workflows.optimization_runs.snapshot import resolve_snapshot_slab_dir

SEAL_FILENAME = "structural_artifacts.json"
SEAL_VERSION = 1


class StructuralArtifactsSealError(ValueError):
    """
    Raised when a structural-artifact backend seal cannot be written or verified.
    """


@dataclass(frozen=True)
class StructuralArtifactsSeal:
    """
    Immutable experiment binding to a structural-artifact backend.
    """

    mode: str
    seal_version: int = SEAL_VERSION
    library_path: str | None = None
    manifest_sha256: str | None = None
    schema_version: int | None = None
    codec_version: int | None = None
    key_algorithm_version: int | None = None
    identity: LibraryIdentity | None = None
    nrows: int | None = None
    rows_per_shard: int | None = None
    allow_snapshot_hash_mismatch: bool = False
    expected_snapshot_hash: str | None = None
    library_snapshot_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seal_version": int(self.seal_version),
            "mode": str(self.mode),
        }
        if self.mode == "library":
            payload.update(
                {
                    "library_path": str(self.library_path),
                    "manifest_sha256": str(self.manifest_sha256),
                    "schema_version": int(self.schema_version or 0),
                    "codec_version": int(self.codec_version or 0),
                    "key_algorithm_version": int(self.key_algorithm_version or 0),
                    "nrows": int(self.nrows or 0),
                    "rows_per_shard": int(self.rows_per_shard or 0),
                    "allow_snapshot_hash_mismatch": bool(
                        self.allow_snapshot_hash_mismatch
                    ),
                    "expected_snapshot_hash": self.expected_snapshot_hash,
                    "library_snapshot_hash": self.library_snapshot_hash,
                    "identity": (
                        self.identity.to_dict() if self.identity is not None else None
                    ),
                }
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StructuralArtifactsSeal:
        mode = str(payload.get("mode", "")).strip().lower()
        if mode not in {"compute", "library"}:
            raise StructuralArtifactsSealError(
                f"structural_artifacts seal mode must be compute|library, got {mode!r}"
            )
        seal_version = int(payload.get("seal_version", SEAL_VERSION))
        if seal_version != SEAL_VERSION:
            raise StructuralArtifactsSealError(
                f"unsupported structural_artifacts seal_version {seal_version}"
            )
        if mode == "compute":
            return cls(mode="compute", seal_version=seal_version)
        identity_raw = payload.get("identity")
        if not isinstance(identity_raw, Mapping):
            raise StructuralArtifactsSealError(
                "library-mode structural_artifacts seal requires identity"
            )
        library_path = payload.get("library_path")
        manifest_sha256 = payload.get("manifest_sha256")
        if not library_path or not manifest_sha256:
            raise StructuralArtifactsSealError(
                "library-mode seal requires library_path and manifest_sha256"
            )
        return cls(
            mode="library",
            seal_version=seal_version,
            library_path=str(library_path),
            manifest_sha256=str(manifest_sha256),
            schema_version=int(payload["schema_version"]),
            codec_version=int(payload["codec_version"]),
            key_algorithm_version=int(payload["key_algorithm_version"]),
            identity=LibraryIdentity.from_dict(identity_raw),
            nrows=int(payload["nrows"]),
            rows_per_shard=int(payload["rows_per_shard"]),
            allow_snapshot_hash_mismatch=bool(
                payload.get("allow_snapshot_hash_mismatch", False)
            ),
            expected_snapshot_hash=(
                str(payload["expected_snapshot_hash"])
                if payload.get("expected_snapshot_hash") is not None
                else None
            ),
            library_snapshot_hash=(
                str(payload["library_snapshot_hash"])
                if payload.get("library_snapshot_hash") is not None
                else None
            ),
        )


def structural_artifacts_seal_path(experiment_dir: Path | str) -> Path:
    """
    Return the seal path under an experiment root.
    """
    return Path(experiment_dir).expanduser().resolve() / SEAL_FILENAME


def combined_snapshot_hash(input_files: Mapping[str, str]) -> str:
    """
    Return the SHA-256 digest of sorted relative-path → content digests.
    """
    payload = "|".join(
        f"{path}:{digest}" for path, digest in sorted(input_files.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _family_directories(container: Path) -> list[Path]:
    """Return family directories that carry their own ``manifest.json``."""
    if not container.is_dir():
        return []
    families = [
        path
        for path in container.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    ]
    return sorted(families, key=lambda path: path.name)


def _family_problem_ids(family_dir: Path) -> tuple[str, ...]:
    """Read the access list from a family manifest."""
    payload = read_json(family_dir / "manifest.json")
    identity = payload.get("identity") if isinstance(payload, dict) else None
    if not isinstance(identity, dict):
        raise StructuralArtifactsSealError(
            f"structural library family {family_dir.name} has no identity"
        )
    raw = identity.get("compatible_problem_ids")
    if raw is None:
        problem_id = identity.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id:
            raise StructuralArtifactsSealError(
                f"structural library family {family_dir.name} has no "
                "compatible_problem_ids"
            )
        return (problem_id,)
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise StructuralArtifactsSealError(
            f"structural library family {family_dir.name} has an invalid "
            "compatible_problem_ids list"
        )
    return tuple(raw)


def resolve_structural_library_family(
    layout_root: Path,
    problem_ids: Sequence[str],
) -> Path:
    """
    Choose the sealed family directory for an omitted library path.

    A missing container, or a container whose only family is ``baseline``,
    resolves to ``{ROOT}/structural_library/baseline``. Several families are
    matched by ``compatible_problem_ids``. A manifest directly in
    ``structural_library/`` is ignored.
    """
    from core.execution_core.output_layout import (
        BASELINE_LIBRARY_NAME,
        STRUCTURAL_LIBRARY_DIRNAME,
        default_structural_library_dir,
    )

    container = Path(layout_root) / STRUCTURAL_LIBRARY_DIRNAME
    default = default_structural_library_dir(layout_root)
    families = _family_directories(container)
    if not families or (
        len(families) == 1 and families[0].name == BASELINE_LIBRARY_NAME
    ):
        return default
    if not problem_ids:
        raise StructuralArtifactsSealError(
            "omitted library_path requires problem ids when several "
            "structural library families are present"
        )
    chosen: Path | None = None
    for problem_id in problem_ids:
        token = str(problem_id)
        matches = [
            family
            for family in families
            if token in _family_problem_ids(family)
        ]
        if not matches:
            raise StructuralArtifactsSealError(
                f"no structural library family lists problem {token}"
            )
        if len(matches) > 1:
            names = ", ".join(family.name for family in matches)
            raise StructuralArtifactsSealError(
                f"problem {token} is listed on multiple structural library "
                f"families: {names}"
            )
        match = matches[0]
        if chosen is None:
            chosen = match
        elif chosen != match:
            raise StructuralArtifactsSealError(
                "experiment problem ids select different structural library "
                f"families: {chosen.name} and {match.name}"
            )
    assert chosen is not None
    return chosen


def resolve_library_path(
    library_path: str | None,
    *,
    config_dir: Path | None = None,
    layout_root: Path | None = None,
    problem_ids: Sequence[str] = (),
) -> Path:
    """
    Resolve a configured library path to an absolute directory.

    When ``library_path`` is omitted, select a family under
    ``{layout_root}/structural_library``. Explicit paths skip that scan.
    """
    token = "" if library_path is None else str(library_path).strip()
    if not token:
        if layout_root is None:
            raise StructuralArtifactsSealError(
                "structural_artifacts.library_path is omitted and no data ROOT "
                "is available for the default structural_library path"
            )
        return resolve_structural_library_family(layout_root, problem_ids)
    raw = Path(token).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    return (base / raw).resolve()


def load_structural_artifacts_seal(
    experiment_dir: Path | str,
) -> StructuralArtifactsSeal:
    """
    Load and validate ``structural_artifacts.json`` from an experiment root.
    """
    path = structural_artifacts_seal_path(experiment_dir)
    if not path.is_file():
        raise StructuralArtifactsSealError(
            f"missing structural artifacts seal at {path}"
        )
    return StructuralArtifactsSeal.from_dict(read_json(path))


def slab_csv_input_hashes(
    input_hashes: Mapping[str, str],
    *,
    slab_type: str,
) -> dict[str, str]:
    """
    Return the snapshotted slab CSV hashes used for library identity.

    Experiment manifests may also hash sampling designs or other inputs.
    The structural library is sealed on the four packaged CSVs only
    (``{slab_type}/{filename}``). Mixing those extra keys into
    ``snapshot_hash`` makes library mode fail on sampling (and any other
    workflow that snapshots more than the CSVs).
    """
    prefix = f"{str(slab_type).strip()}/"
    selected = {
        str(path): str(digest)
        for path, digest in input_hashes.items()
        if str(path).startswith(prefix)
    }
    if not selected:
        raise StructuralArtifactsSealError(
            "library identity requires snapshotted slab CSVs "
            f"under {prefix!r}; got keys {sorted(input_hashes)}"
        )
    return selected


def write_structural_artifacts_seal(
    experiment_dir: Path | str,
    seal: StructuralArtifactsSeal,
    *,
    overwrite: bool = True,
) -> Path:
    """
    Atomically publish a structural-artifacts seal beside the experiment manifest.
    """
    path = structural_artifacts_seal_path(experiment_dir)
    if path.exists() and not overwrite:
        raise StructuralArtifactsSealError(
            f"structural artifacts seal already exists at {path}"
        )
    write_json_atomic(path, seal.to_dict())
    return path


def build_expected_library_identity(
    *,
    config: ExperimentConfig,
    experiment_dir: Path,
    input_hashes: Mapping[str, str],
    problem_id: str,
) -> LibraryIdentity:
    """
    Build the scientific identity expected of a consumer library for this experiment.
    """
    inputs_dir = Path(experiment_dir).expanduser().resolve() / "inputs"
    slab_dir = resolve_snapshot_slab_dir(inputs_dir, config.experiment.slab_type)
    space, _schema, info = load_problem_space_and_schema_from_csv_dir(
        slab_dir=slab_dir,
        problem_id=problem_id,
    )
    decode = make_decode(info["model"], space.var_names)
    params0 = decode(list([0] * len(space.var_names)))
    run_config = analysis_run_config_from_params(
        params0, slab_type=config.experiment.slab_type
    )
    from core.ioh_core.workflows.structural_library.identity import (
        build_library_identity,
    )

    slab_hashes = slab_csv_input_hashes(
        input_hashes, slab_type=config.experiment.slab_type
    )
    return build_library_identity(
        slab_type=config.experiment.slab_type,
        problem_id=problem_id,
        space=space,
        snapshot_hash=combined_snapshot_hash(slab_hashes),
        run_config=run_config,
        library_structural_cardinality=structural_cardinality(space),
        input_hashes=dict(slab_hashes),
    )


def assert_library_compatible_with_experiment(
    stored: LibraryIdentity,
    expected: LibraryIdentity,
    *,
    allow_snapshot_hash_mismatch: bool = False,
) -> None:
    """
    Fail closed when a sealed library is scientifically incompatible.

    ``problem_id`` must appear in ``stored.compatible_problem_ids``.
    ``structural_cardinality`` may be smaller than the full catalogue when the
    library is intentionally truncated; other identity fields must match.
    """
    allowed = tuple(str(item) for item in stored.compatible_problem_ids)
    if str(expected.problem_id) not in allowed:
        raise StructuralArtifactsSealError(
            "library identity mismatch on problem_id: "
            f"stored={stored.problem_id!r}, expected={expected.problem_id!r}"
            if allowed == (str(stored.problem_id),)
            else (
                "library identity mismatch on problem_id: "
                f"stored compatible_problem_ids={list(allowed)!r}, "
                f"expected={expected.problem_id!r} "
                f"(build provenance problem_id={stored.problem_id!r})"
            )
        )
    checks = (
        ("slab_type", stored.slab_type, expected.slab_type),
        (
            "run_config_fingerprint",
            stored.run_config_fingerprint,
            expected.run_config_fingerprint,
        ),
        ("full_cardinality", stored.full_cardinality, expected.full_cardinality),
        ("finish_names", stored.finish_names, expected.finish_names),
        ("finish_sizes", stored.finish_sizes, expected.finish_sizes),
        ("structural_names", stored.structural_names, expected.structural_names),
        ("structural_sizes", stored.structural_sizes, expected.structural_sizes),
    )
    if not allow_snapshot_hash_mismatch:
        checks = (
            ("snapshot_hash", stored.snapshot_hash, expected.snapshot_hash),
            *checks,
        )
    for name, left, right in checks:
        if left != right:
            raise StructuralArtifactsSealError(
                f"library identity mismatch on {name}: "
                f"stored={left!r}, expected={right!r}"
            )
    if int(stored.structural_cardinality) < 1:
        raise StructuralArtifactsSealError(
            "library structural_cardinality must be >= 1"
        )
    if int(stored.structural_cardinality) > int(expected.structural_cardinality):
        raise StructuralArtifactsSealError(
            "library structural_cardinality "
            f"{stored.structural_cardinality} exceeds catalogue "
            f"{expected.structural_cardinality}"
        )


def seal_structural_artifacts_for_experiment(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    input_hashes: Mapping[str, str],
    overwrite: bool = True,
) -> StructuralArtifactsSeal:
    """
    Open/validate the configured backend and seal it into the experiment root.

    Library mode opens the external sealed library in place (no data copy).
    """
    section = config.structural_artifacts
    if section.mode == "compute":
        seal = StructuralArtifactsSeal(mode="compute")
        write_structural_artifacts_seal(
            experiment_dir, seal, overwrite=overwrite
        )
        return seal

    config_dir = (
        config.source_path.parent if config.source_path is not None else None
    )
    from core.execution_core.output_layout import layout_root_from_output_root

    layout_root = layout_root_from_output_root(
        config.experiment.output_root,
        cwd=config_dir if config_dir is not None else Path.cwd(),
    )
    library_root = resolve_library_path(
        section.library_path,
        config_dir=config_dir,
        layout_root=layout_root,
        problem_ids=config.experiment.problem_ids,
    )
    if not library_root.is_dir():
        raise StructuralArtifactsSealError(
            f"structural_artifacts.library_path is not a directory: {library_root}"
        )

    try:
        from core.ioh_core.workflows.brute_force.deps import import_pyarrow

        import_pyarrow()
    except Exception as exc:  # noqa: BLE001
        raise StructuralArtifactsSealError(
            "structural_artifacts.mode='library' requires pyarrow; "
            f"import failed: {exc}"
        ) from exc

    try:
        with open_structural_library(library_root) as library:
            allow_mismatch = bool(section.allow_snapshot_hash_mismatch)
            expected_identities: list[LibraryIdentity] = []
            mismatched_snapshot_ids: list[str] = []
            for problem_id in config.experiment.problem_ids:
                expected = build_expected_library_identity(
                    config=config,
                    experiment_dir=experiment_dir,
                    input_hashes=input_hashes,
                    problem_id=problem_id,
                )
                expected_identities.append(expected)
                try:
                    assert_library_compatible_with_experiment(
                        library.identity,
                        expected,
                        allow_snapshot_hash_mismatch=allow_mismatch,
                    )
                except StructuralArtifactsSealError as exc:
                    raise StructuralArtifactsSealError(
                        "structural library is incompatible with "
                        f"problem_id {problem_id!r}: {exc}"
                    ) from exc
                if (
                    allow_mismatch
                    and library.identity.snapshot_hash
                    != expected.snapshot_hash
                ):
                    mismatched_snapshot_ids.append(problem_id)
            if mismatched_snapshot_ids:
                warnings.warn(
                    "forcing structural library snapshot_hash mismatch: "
                    f"stored={library.identity.snapshot_hash!r}, "
                    "all other identity checks passed for problem_ids="
                    f"{mismatched_snapshot_ids!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            expected = expected_identities[0]
            manifest_path = library_root / "manifest.json"
            seal = StructuralArtifactsSeal(
                mode="library",
                library_path=str(library_root),
                manifest_sha256=sha256_file(manifest_path),
                schema_version=int(library.manifest.schema_version),
                codec_version=int(library.manifest.codec_version),
                key_algorithm_version=int(library.manifest.key_algorithm_version),
                identity=library.identity,
                nrows=int(library.manifest.nrows),
                rows_per_shard=int(library.manifest.rows_per_shard),
                allow_snapshot_hash_mismatch=allow_mismatch,
                expected_snapshot_hash=(
                    expected.snapshot_hash if allow_mismatch else None
                ),
                library_snapshot_hash=(
                    library.identity.snapshot_hash if allow_mismatch else None
                ),
            )
    except StructuralLibraryError as exc:
        raise StructuralArtifactsSealError(str(exc)) from exc

    write_structural_artifacts_seal(experiment_dir, seal, overwrite=overwrite)
    return seal


def revalidate_structural_artifacts_seal(
    experiment_dir: Path | str,
) -> StructuralArtifactsSeal:
    """
    Re-open the sealed backend and fail closed on identity or digest drift.

    Workers call this before evaluation. Compute mode is a no-op beyond loading
    the seal document.
    """
    seal = load_structural_artifacts_seal(experiment_dir)
    if seal.mode == "compute":
        return seal
    if seal.library_path is None or seal.identity is None or seal.manifest_sha256 is None:
        raise StructuralArtifactsSealError(
            "library-mode seal is missing path, identity, or manifest digest"
        )
    library_root = Path(seal.library_path)
    if not library_root.is_dir():
        raise StructuralArtifactsSealError(
            f"sealed library_path is missing or not a directory: {library_root}"
        )
    manifest_path = library_root / "manifest.json"
    if not manifest_path.is_file():
        raise StructuralArtifactsSealError(f"missing library manifest at {manifest_path}")
    digest = sha256_file(manifest_path)
    if digest != seal.manifest_sha256:
        raise StructuralArtifactsSealError(
            "library manifest digest changed since prepare: "
            f"sealed={seal.manifest_sha256!r}, current={digest!r}"
        )
    try:
        with open_structural_library(
            library_root, expected_identity=seal.identity
        ) as library:
            if library.manifest.schema_version != seal.schema_version:
                raise StructuralArtifactsSealError("schema_version drift since prepare")
            if library.manifest.codec_version != seal.codec_version:
                raise StructuralArtifactsSealError("codec_version drift since prepare")
            if library.manifest.key_algorithm_version != seal.key_algorithm_version:
                raise StructuralArtifactsSealError(
                    "key_algorithm_version drift since prepare"
                )
            if int(library.manifest.nrows) != int(seal.nrows or -1):
                raise StructuralArtifactsSealError("nrows drift since prepare")
    except StructuralLibraryError as exc:
        raise StructuralArtifactsSealError(str(exc)) from exc
    return seal


@contextmanager
def open_structural_provider_session(
    experiment_dir: Path | str,
) -> Iterator[tuple[StructuralArtifactsSeal, StructuralEvaluationProvider]]:
    """
    Revalidate the experiment seal and yield a ready structural provider.

    Library mode keeps the sealed Arrow library open for the duration of the
    context (mmap LRU). Compute mode yields a lightweight compute provider.
    """
    seal = revalidate_structural_artifacts_seal(experiment_dir)
    if seal.mode == "compute":
        yield seal, build_structural_provider("compute")
        return
    if seal.library_path is None or seal.identity is None:
        raise StructuralArtifactsSealError(
            "library-mode seal is missing library_path or identity"
        )
    library = open_structural_library(
        Path(seal.library_path),
        expected_identity=seal.identity,
    )
    try:
        yield seal, build_structural_provider(
            "library",
            lookup=library,
            allow_snapshot_hash_mismatch=bool(seal.allow_snapshot_hash_mismatch),
        )
    finally:
        library.close()
