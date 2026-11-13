"""
Snapshot packaged slab CSV inputs for reproducible experiment preparation.

Copies the four study CSVs into an experiment ``inputs/`` tree, records
SHA-256 digests for the manifest, and re-validates those digests before
workers construct problems from the snapshot.
"""
from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import Path
from typing import Mapping

from core.execution_core.atomic_io import write_bytes_atomic

SNAPSHOT_CSV_NAMES: tuple[str, ...] = (
    "parameter_defaults.csv",
    "constraint_defaults.csv",
    "problem_list.csv",
    "materials.csv",
)


class SnapshotError(ValueError):
    """
    Raised when a slab input snapshot cannot be created or validated.

    Typical causes: unknown slab type, missing packaged CSV, unexpected
    file set under the snapshot directory, or SHA-256 mismatch against the
    sealed manifest digests.
    """


def sha256_hex(data: bytes) -> str:
    """
    Return the lowercase SHA-256 hex digest of ``data``.

    Parameters
    ----------
    data : bytes
        Raw file contents.

    Returns
    -------
    str
        64-character hexadecimal digest.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """
    Return the SHA-256 hex digest of a file on disk.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        64-character hexadecimal digest.
    """
    return sha256_hex(Path(path).read_bytes())


def packaged_slab_csv_path(slab_type: str, filename: str):
    """
    Return an ``importlib.resources`` path to one packaged slab CSV.

    Parameters
    ----------
    slab_type : str
        Slab package name such as ``\"hp_slab\"``.
    filename : str
        CSV basename listed in :data:`SNAPSHOT_CSV_NAMES`.

    Returns
    -------
    Traversable
        Resource path under ``slab_construction.slabs.<slab_type>``.

    Raises
    ------
    SnapshotError
        If ``filename`` is not a supported snapshot CSV.
    """
    if filename not in SNAPSHOT_CSV_NAMES:
        raise SnapshotError(
            f"unsupported snapshot CSV {filename!r}; "
            f"expected one of {list(SNAPSHOT_CSV_NAMES)}"
        )
    return files("slab_construction.slabs") / slab_type / filename


def read_packaged_slab_csv_bytes(slab_type: str, filename: str) -> bytes:
    """
    Read one packaged slab CSV as raw bytes via ``importlib.resources``.

    Parameters
    ----------
    slab_type : str
        Slab package name such as ``\"hp_slab\"``.
    filename : str
        CSV basename listed in :data:`SNAPSHOT_CSV_NAMES`.

    Returns
    -------
    bytes
        Exact packaged file contents.

    Raises
    ------
    SnapshotError
        If the packaged resource is missing or unreadable.
    """
    resource = packaged_slab_csv_path(slab_type, filename)
    try:
        return resource.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise SnapshotError(
            f"missing packaged CSV for slab_type={slab_type!r}: {filename}"
        ) from exc


def snapshot_relative_key(slab_type: str, filename: str) -> str:
    """
    Return the manifest-relative input key for one snapshotted CSV.

    Parameters
    ----------
    slab_type : str
        Slab package name.
    filename : str
        CSV basename.

    Returns
    -------
    str
        Relative path such as ``\"hp_slab/problem_list.csv\"``.
    """
    return f"{slab_type}/{filename}"


def snapshot_slab_inputs(
    *,
    slab_type: str,
    inputs_dir: Path,
) -> dict[str, str]:
    """
    Copy the four packaged slab CSVs into ``inputs_dir/{slab_type}/``.

    Parameters
    ----------
    slab_type : str
        Slab package name such as ``\"hp_slab\"``.
    inputs_dir : Path
        Experiment ``inputs/`` directory (parent of the slab folder).

    Returns
    -------
    dict[str, str]
        Mapping from relative keys (``{slab_type}/{filename}``) to SHA-256
        hex digests of the copied bytes. Suitable for
        ``ExperimentManifest.input_files``.

    Raises
    ------
    SnapshotError
        If a packaged CSV cannot be read.
    """
    inputs_dir = Path(inputs_dir)
    slab_dir = inputs_dir / slab_type
    slab_dir.mkdir(parents=True, exist_ok=True)

    hashes: dict[str, str] = {}
    for filename in SNAPSHOT_CSV_NAMES:
        data = read_packaged_slab_csv_bytes(slab_type, filename)
        destination = slab_dir / filename
        write_bytes_atomic(destination, data)
        hashes[snapshot_relative_key(slab_type, filename)] = sha256_hex(data)
    return hashes


def validate_snapshot_hashes(
    *,
    inputs_dir: Path,
    expected_hashes: Mapping[str, str],
) -> None:
    """
    Re-hash snapshotted files and require exact digest agreement.

    Parameters
    ----------
    inputs_dir : Path
        Experiment ``inputs/`` directory containing ``{slab_type}/...``.
    expected_hashes : Mapping[str, str]
        Manifest ``input_files`` mapping of relative paths to digests.

    Raises
    ------
    SnapshotError
        If a file is missing, an unexpected key is present, or a digest
        differs from the sealed manifest value.
    """
    inputs_dir = Path(inputs_dir)
    expected = dict(expected_hashes)
    if not expected:
        raise SnapshotError("expected_hashes must be a non-empty mapping")

    for relative, expected_digest in sorted(expected.items()):
        path = inputs_dir / relative
        if not path.is_file():
            raise SnapshotError(f"missing snapshotted input: {relative}")
        actual = sha256_file(path)
        if actual != expected_digest:
            raise SnapshotError(
                f"SHA-256 mismatch for {relative}: "
                f"expected {expected_digest}, got {actual}"
            )

    # Reject unexpected files under each top-level inputs folder. Allowed
    # basenames are exactly those listed for that folder in expected_hashes
    # (slab CSV snapshots list the four study CSVs; sensitivity seals list
    # the design artifacts).
    tops = {
        Path(relative).parts[0]
        for relative in expected
        if Path(relative).parts
    }
    for top in sorted(tops):
        top_dir = inputs_dir / top
        if not top_dir.is_dir():
            continue
        present = {path.name for path in top_dir.iterdir() if path.is_file()}
        allowed = {
            Path(key).name
            for key in expected
            if Path(key).parts and Path(key).parts[0] == top
        }
        unexpected = sorted(present - allowed)
        if unexpected:
            raise SnapshotError(
                f"unexpected files under inputs/{top}: {unexpected}"
            )


def resolve_snapshot_slab_dir(inputs_dir: Path, slab_type: str) -> Path:
    """
    Return the snapshotted slab CSV directory.

    Parameters
    ----------
    inputs_dir : Path
        Experiment ``inputs/`` directory.
    slab_type : str
        Slab package name.

    Returns
    -------
    Path
        ``inputs_dir / slab_type``.

    Raises
    ------
    SnapshotError
        If the directory or any required CSV is missing.
    """
    slab_dir = Path(inputs_dir) / slab_type
    if not slab_dir.is_dir():
        raise SnapshotError(f"missing snapshot directory: {slab_dir}")
    missing = [name for name in SNAPSHOT_CSV_NAMES if not (slab_dir / name).is_file()]
    if missing:
        raise SnapshotError(
            f"snapshot for {slab_type!r} is incomplete; missing {missing}"
        )
    return slab_dir
