"""
Atomic publish helpers for shared experiment artifacts.

Every durable JSON/text artifact is written to a unique temporary name in the
destination directory, flushed, optionally fsynced, then published with
``os.replace``. Readers must ignore ``*.tmp`` and ``*.partial`` names.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


_TEMP_SUFFIXES = (".tmp", ".partial")


def is_temporary_artifact_name(name: str) -> bool:
    """
    Return True if ``name`` is a temporary publish name to ignore.

    Parameters
    ----------
    name : str
        File or path basename (not necessarily absolute).

    Returns
    -------
    bool
        ``True`` when the name ends with ``.tmp`` or ``.partial``.
    """
    lower = name.lower()
    return any(lower.endswith(suffix) for suffix in _TEMP_SUFFIXES)


def _unique_temp_path(destination: Path, *, suffix: str = ".tmp") -> Path:
    """
    Build a unique temporary sibling path next to ``destination``.

    Parameters
    ----------
    destination : Path
        Final publish path.
    suffix : str, optional
        Temporary suffix, default ``\".tmp\"``.

    Returns
    -------
    Path
        Path including PID and UUID so concurrent writers do not collide.
    """
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    token = f"{os.getpid()}_{uuid.uuid4().hex}"
    return destination.with_name(f"{destination.name}.{token}{suffix}")


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """
    Atomically publish ``data`` at ``path``.

    Parameters
    ----------
    path : Path
        Final destination path.
    data : bytes
        Full file contents to publish.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_temp_path(path, suffix=".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                # Some platforms / filesystems do not support fsync.
                pass
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """
    Atomically publish text at ``path``.

    Parameters
    ----------
    path : Path
        Final destination path.
    text : str
        Full text contents to publish.
    encoding : str, optional
        Text encoding, default ``\"utf-8\"``.
    """
    write_bytes_atomic(path, text.encode(encoding))


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """
    Atomically publish a JSON document at ``path``.

    Parameters
    ----------
    path : Path
        Final destination path.
    payload : Any
        JSON-serialisable object (dict/list/scalars). Non-JSON types are
        stringified via ``default=str``.
    indent : int, optional
        Pretty-print indent, default ``2``.
    """
    text = json.dumps(payload, indent=indent, sort_keys=True, default=str)
    write_text_atomic(path, text + "\n")


def read_json(path: Path) -> Any:
    """
    Read a JSON document from ``path``.

    Temporary sibling names are never opened by this helper; callers that
    scan directories should skip names matching :func:`is_temporary_artifact_name`.

    Parameters
    ----------
    path : Path
        Published JSON file path (not a ``*.tmp`` / ``*.partial`` name).

    Returns
    -------
    Any
        Parsed JSON value.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
