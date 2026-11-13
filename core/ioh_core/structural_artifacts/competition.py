"""
Competition-facing structural library API: fixture, open/get, and audit.

The redistributable fixture is a tiny sealed Arrow library (four synthetic
artifacts: success, mk_cleared, build_failure, success). The full Problem 1
library (~2.22M IDs) is distributed externally and is never shipped in git
or the wheel.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from core.execution_core.atomic_io import write_json_atomic
from core.ioh_core.structural_artifacts.codec import (
    CODEC_VERSION,
    SCHEMA_VERSION,
    artifacts_equal,
)
from core.ioh_core.structural_artifacts.fixtures import (
    make_build_failure_artifact,
    make_mk_cleared_artifact,
    make_success_artifact,
)
from core.ioh_core.structural_artifacts.library import (
    SUCCESS_MARKER,
    open_structural_library,
    seal_structural_library,
)
from core.ioh_core.structural_artifacts.models import (
    KEY_ALGORITHM_VERSION,
    LibraryIdentity,
    StructuralLibraryError,
    run_config_fingerprint,
    sha256_file,
)
from core.ioh_core.structural_artifacts.provider import (
    LibraryStructuralProvider,
    StructuralEvaluationProvider,
    build_structural_provider,
)
from core.analysis_core.run_config import AnalysisRunConfig

COMPETITION_FIXTURE_NAME = "competition_fixture"
COMPETITION_SNAPSHOT_HASH = "competition-fixture-v1"
COMPETITION_PROBLEM_ID = "fixture"
COMPETITION_CARDINALITY = 4
COMPETITION_ROWS_PER_SHARD = 2


class CompetitionLibraryError(ValueError):
    """
    Raised when the competition fixture or audit cannot proceed safely.
    """


def competition_fixture_root() -> Path:
    """
    Return the packaged competition fixture directory.
    """
    return Path(__file__).resolve().parent / "competition_data" / COMPETITION_FIXTURE_NAME


def competition_identity() -> LibraryIdentity:
    """
    Return the sealed scientific identity for the redistributable fixture.
    """
    cfg = AnalysisRunConfig(invalid_utilization=999.0, slab_type="hp_slab")
    return LibraryIdentity(
        slab_type="hp_slab",
        problem_id=COMPETITION_PROBLEM_ID,
        snapshot_hash=COMPETITION_SNAPSHOT_HASH,
        run_config_fingerprint=run_config_fingerprint(cfg),
        structural_cardinality=COMPETITION_CARDINALITY,
        full_cardinality=COMPETITION_CARDINALITY * 4,
        finish_names=("finish_a", "finish_b"),
        finish_sizes=(2, 2),
        structural_names=("s0", "s1"),
        structural_sizes=(2, 2),
        input_hashes=(("fixture", COMPETITION_SNAPSHOT_HASH),),
    )


def _aligned_artifact(seed: int):
    if seed % 3 == 0:
        art = make_success_artifact(seed=seed)
    elif seed % 3 == 1:
        art = make_mk_cleared_artifact(seed=seed)
    else:
        art = make_build_failure_artifact(seed=seed)
    return type(art)(
        key=art.key,
        snapshot_hash=COMPETITION_SNAPSHOT_HASH,
        n=art.n,
        m_u=art.m_u,
        m_cr=art.m_cr,
        mk_curves=art.mk_curves,
        mk_metadata=art.mk_metadata,
        mk_cleared=art.mk_cleared,
        build_failure_result=(
            dict(art.build_failure_result)
            if art.build_failure_result is not None
            else None
        ),
    )


def competition_fixture_rows() -> list[tuple[int, tuple[int, ...], Any]]:
    """
    Return ``(structural_id, structural_indices, artifact)`` for the fixture.
    """
    rows = []
    for sid in range(COMPETITION_CARDINALITY):
        indices = (sid % 2, sid // 2)
        rows.append((sid, indices, _aligned_artifact(10 + sid)))
    return rows


def build_competition_fixture(
    destination: Path | str | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Build the redistributable sealed competition fixture.

    Parameters
    ----------
    destination : path or None
        Library root. Defaults to the packaged ``competition_data`` path.
    overwrite : bool, optional
        When ``True``, delete an existing ``_SUCCESS`` tree first.

    Returns
    -------
    Path
        Sealed library root containing ``_SUCCESS``.
    """
    root = Path(destination) if destination is not None else competition_fixture_root()
    root = root.resolve()
    if (root / SUCCESS_MARKER).exists():
        if not overwrite:
            return root
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    seal_structural_library(
        root,
        competition_fixture_rows(),
        identity=competition_identity(),
        rows_per_shard=COMPETITION_ROWS_PER_SHARD,
    )
    write_json_atomic(
        root / "provenance.json",
        competition_provenance(root),
    )
    return root


def open_competition_fixture(
    *,
    verify_part_hashes: bool = False,
    build_if_missing: bool = True,
):
    """
    Open the packaged competition fixture (building it if needed).
    """
    root = competition_fixture_root()
    if not (root / SUCCESS_MARKER).is_file():
        if not build_if_missing:
            raise CompetitionLibraryError(
                f"competition fixture missing at {root}; "
                "run hdb-structural-library build-fixture"
            )
        build_competition_fixture(root, overwrite=False)
    return open_structural_library(
        root,
        expected_identity=competition_identity(),
        verify_part_hashes=verify_part_hashes,
    )


def open_library(
    path: Path | str,
    *,
    expected_identity: LibraryIdentity | None = None,
    allow_snapshot_hash_mismatch: bool = False,
    verify_part_hashes: bool = False,
):
    """
    Open any sealed structural library for competition / operator use.
    """
    return open_structural_library(
        Path(path),
        expected_identity=expected_identity,
        allow_snapshot_hash_mismatch=allow_snapshot_hash_mismatch,
        verify_part_hashes=verify_part_hashes,
    )


def library_provider(library) -> StructuralEvaluationProvider:
    """
    Wrap an open sealed library as a strict evaluation provider.
    """
    return build_structural_provider("library", lookup=library)


@dataclass(frozen=True)
class AuditMismatch:
    """
    One structural ID that failed the recompute audit.
    """

    structural_id: int
    reason: str


@dataclass(frozen=True)
class AuditReport:
    """
    Result of a random or explicit recompute audit.
    """

    checked: int
    mismatches: tuple[AuditMismatch, ...]
    identity: dict[str, Any]

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": int(self.checked),
            "ok": bool(self.ok),
            "mismatches": [
                {"structural_id": m.structural_id, "reason": m.reason}
                for m in self.mismatches
            ],
            "identity": dict(self.identity),
        }


def audit_library(
    library,
    *,
    structural_ids: Sequence[int] | None = None,
    sample_size: int | None = None,
    seed: int = 0,
    recompute: Callable[[int], Any] | None = None,
    expected_artifacts: Mapping[int, Any] | None = None,
) -> AuditReport:
    """
    Audit library rows against recomputed or expected artifacts.

    Parameters
    ----------
    library : SealedStructuralLibrary
        Open sealed library.
    structural_ids : sequence of int or None
        Explicit IDs to check. When omitted, sample from the library range.
    sample_size : int or None
        When ``structural_ids`` is omitted, draw this many IDs (default: all
        IDs up to 64, else 64 random).
    seed : int
        RNG seed for sampling.
    recompute : callable or None
        ``(structural_id) -> StructuralArtifact`` reference builder.
    expected_artifacts : mapping or None
        Precomputed reference artifacts keyed by structural ID.

    Returns
    -------
    AuditReport
        Checked count and any mismatches.
    """
    cardinality = int(library.manifest.identity.structural_cardinality)
    if structural_ids is not None:
        ids = [int(v) for v in structural_ids]
    else:
        n = sample_size
        if n is None:
            n = min(64, cardinality)
        n = min(int(n), cardinality)
        rng = random.Random(int(seed))
        ids = sorted(rng.sample(range(cardinality), n)) if n < cardinality else list(
            range(cardinality)
        )

    mismatches: list[AuditMismatch] = []
    for sid in ids:
        try:
            got = library.get(sid)
        except StructuralLibraryError as exc:
            mismatches.append(AuditMismatch(sid, f"lookup failed: {exc}"))
            continue
        reference = None
        if expected_artifacts is not None and sid in expected_artifacts:
            reference = expected_artifacts[sid]
        elif recompute is not None:
            try:
                reference = recompute(sid)
            except Exception as exc:  # noqa: BLE001
                mismatches.append(AuditMismatch(sid, f"recompute failed: {exc}"))
                continue
        else:
            mismatches.append(
                AuditMismatch(sid, "no recompute or expected_artifacts provided")
            )
            continue
        if not artifacts_equal(got, reference):
            mismatches.append(AuditMismatch(sid, "artifact scientific mismatch"))
    return AuditReport(
        checked=len(ids),
        mismatches=tuple(mismatches),
        identity=library.identity.to_dict(),
    )


def audit_competition_fixture(*, seed: int = 0) -> AuditReport:
    """
    Audit the packaged fixture against its known synthetic artifacts.
    """
    expected = {sid: art for sid, _idx, art in competition_fixture_rows()}
    with open_competition_fixture() as library:
        return audit_library(
            library,
            structural_ids=list(range(COMPETITION_CARDINALITY)),
            expected_artifacts=expected,
            seed=seed,
        )


def _pkg_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def competition_provenance(library_root: Path | str) -> dict[str, Any]:
    """
    Build a provenance document for a sealed library root.
    """
    root = Path(library_root).resolve()
    manifest_path = root / "manifest.json"
    payload: dict[str, Any] = {
        "library_root": str(root),
        "schema_version": SCHEMA_VERSION,
        "codec_version": CODEC_VERSION,
        "key_algorithm_version": KEY_ALGORITHM_VERSION,
        "format": "arrow_ipc",
        "packages": {
            "hpdesignbench": _pkg_version("hpdesignbench"),
            "pyarrow": _pkg_version("pyarrow"),
            "numpy": _pkg_version("numpy"),
        },
    }
    if manifest_path.is_file():
        payload["manifest_sha256"] = sha256_file(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["identity"] = manifest.get("identity")
            payload["nrows"] = manifest.get("nrows")
            payload["rows_per_shard"] = manifest.get("rows_per_shard")
        except (OSError, json.JSONDecodeError):
            pass
    success = root / SUCCESS_MARKER
    payload["success_marker"] = success.is_file()
    return payload


def write_provenance(
    library_root: Path | str,
    destination: Path | str | None = None,
) -> Path:
    """
    Write ``provenance.json`` beside or at ``destination``.
    """
    root = Path(library_root).resolve()
    path = (
        Path(destination).resolve()
        if destination is not None
        else root / "provenance.json"
    )
    write_json_atomic(path, competition_provenance(root))
    return path


def validate_library_root(library_root: Path | str) -> dict[str, Any]:
    """
    Open a sealed library and return a short validation summary.
    """
    root = Path(library_root).resolve()
    if not (root / SUCCESS_MARKER).is_file():
        raise CompetitionLibraryError(f"missing {SUCCESS_MARKER} under {root}")
    with open_library(root, verify_part_hashes=True) as library:
        first = library.get(0)
        last_id = int(library.manifest.identity.structural_cardinality) - 1
        last = library.get(last_id)
        return {
            "ok": True,
            "library_root": str(root),
            "nrows": int(library.manifest.nrows),
            "identity": library.identity.to_dict(),
            "first_structural_id": 0,
            "last_structural_id": last_id,
            "first_mk_cleared": bool(first.mk_cleared),
            "last_is_build_failure": bool(last.is_build_failure),
            "manifest_sha256": sha256_file(root / "manifest.json"),
        }
