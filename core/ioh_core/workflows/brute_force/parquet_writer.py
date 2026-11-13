"""
Streaming, attempt-private Parquet writing with atomic publication.

Workers buffer flattened evaluation rows, flush one Parquet row group per
batch into an attempt-local partial file, validate on close, then publish
into ``dataset/parts/`` via atomic rename (with copy when crossing
filesystems). Shared dataset metadata is never updated here.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from core.ioh_core.workflows.brute_force.deps import import_pyarrow


class ParquetWriterError(RuntimeError):
    """
    Raised when a shard Parquet file cannot be written, validated, or published.
    """


class SupportsParquetSchema(Protocol):
    """Minimal schema holder required by :class:`ParquetShardWriter`."""

    schema: Any


def shard_part_filename(shard_index: int) -> str:
    """
    Return the canonical part basename for ``shard_index``.

    Parameters
    ----------
    shard_index : int
        Zero-based shard ordinal.

    Returns
    -------
    str
        Name such as ``\"shard-000000.parquet\"``.
    """
    if shard_index < 0:
        raise ParquetWriterError(f"shard_index must be >= 0, got {shard_index}")
    return f"shard-{int(shard_index):06d}.parquet"


def shard_part_path(dataset_dir: Path, shard_index: int) -> Path:
    """
    Return ``dataset_dir/parts/shard-XXXXXX.parquet``.

    Parameters
    ----------
    dataset_dir : Path
        Dataset root containing ``parts/``.
    shard_index : int
        Zero-based shard ordinal.

    Returns
    -------
    Path
        Final published part path.
    """
    return Path(dataset_dir) / "parts" / shard_part_filename(shard_index)


def _normalize_compression(compression: str) -> str | None:
    """
    Map config compression tokens to PyArrow compression arguments.

    Parameters
    ----------
    compression : str
        Configured codec (``zstd``, ``snappy``, ``gzip``, ``none``).

    Returns
    -------
    str or None
        Codec name for ``ParquetWriter``, or ``None`` when uncompressed.
    """
    token = str(compression).strip().lower()
    if token in {"", "none", "uncompressed"}:
        return None
    if token in {"zstd", "snappy", "gzip"}:
        return token
    raise ParquetWriterError(
        f"unsupported parquet compression {compression!r}; "
        "expected zstd, snappy, gzip, or none"
    )


def _same_filesystem(path_a: Path, path_b: Path) -> bool:
    """
    Return whether two paths appear to share a filesystem device.

    Parameters
    ----------
    path_a, path_b : Path
        Existing paths (files or directories).

    Returns
    -------
    bool
        ``True`` when ``st_dev`` matches.
    """
    return os.stat(path_a).st_dev == os.stat(path_b).st_dev


def _unique_partial(destination: Path) -> Path:
    """
    Build a unique ``*.partial`` sibling of ``destination``.

    Parameters
    ----------
    destination : Path
        Final publish path.

    Returns
    -------
    Path
        Unique partial path in the same directory.
    """
    token = f"{os.getpid()}_{uuid.uuid4().hex}"
    return destination.with_name(f"{destination.name}.{token}.partial")


def publish_parquet_file(
    source: Path,
    destination: Path,
    *,
    force_cross_filesystem_copy: bool = False,
) -> Path:
    """
    Publish a closed Parquet file to ``destination`` atomically.

    Same-filesystem moves use ``os.replace``. Cross-filesystem (or forced)
    publication copies into a destination-side ``*.partial`` then renames.

    Parameters
    ----------
    source : Path
        Closed attempt-private Parquet file.
    destination : Path
        Final ``parts/shard-XXXXXX.parquet`` path.
    force_cross_filesystem_copy : bool, optional
        When ``True``, always use copy + rename (for tests).

    Returns
    -------
    Path
        Resolved destination path.

    Raises
    ------
    ParquetWriterError
        If ``source`` is missing or publication fails.
    """
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise ParquetWriterError(f"missing source parquet file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    use_copy = force_cross_filesystem_copy
    if not use_copy:
        try:
            use_copy = not _same_filesystem(source, destination.parent)
        except OSError:
            use_copy = True

    if not use_copy:
        os.replace(source, destination)
        return destination.resolve()

    partial = _unique_partial(destination)
    try:
        shutil.copy2(source, partial)
        with open(partial, "rb") as fh:
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(partial, destination)
        try:
            source.unlink()
        except OSError:
            pass
        return destination.resolve()
    finally:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass


class ParquetShardWriter:
    """
    Buffer flattened rows and stream them into one shard Parquet file.

    Parameters
    ----------
    schema_info : SupportsParquetSchema
        Explicit schema holder exposing a ``schema`` attribute (for example
        :class:`~core.ioh_core.workflows.brute_force.schema.BruteForceSchema`
        or the sensitivity schema type).
    staging_path : Path
        Attempt-private write path (typically ``*.partial``).
    destination_path : Path
        Final published part path under ``dataset/parts/``.
    batch_rows : int, optional
        Maximum buffered rows before flushing a row group (>= 1).
    compression : str, optional
        Parquet codec (``zstd``, ``snappy``, ``gzip``, ``none``).
    force_cross_filesystem_copy : bool, optional
        Force copyback publication path (testing).
    """

    def __init__(
        self,
        *,
        schema_info: SupportsParquetSchema,
        staging_path: Path,
        destination_path: Path,
        batch_rows: int = 4096,
        compression: str = "zstd",
        force_cross_filesystem_copy: bool = False,
    ) -> None:
        if batch_rows < 1:
            raise ParquetWriterError(f"batch_rows must be >= 1, got {batch_rows}")
        self.schema_info = schema_info
        self.staging_path = Path(staging_path)
        self.destination_path = Path(destination_path)
        self.batch_rows = int(batch_rows)
        self.compression = _normalize_compression(compression)
        self.force_cross_filesystem_copy = bool(force_cross_filesystem_copy)

        self._pa = import_pyarrow()
        import pyarrow.parquet as pq_imported

        self._pq = pq_imported
        self._buffer: list[dict[str, Any]] = []
        self._rows_written = 0
        self._row_groups_written = 0
        self._closed = False
        self._validated = False
        self._published = False
        self._writer: Any | None = None

        self.staging_path.parent.mkdir(parents=True, exist_ok=True)
        if self.staging_path.exists():
            raise ParquetWriterError(
                f"staging path already exists: {self.staging_path}"
            )
        self._writer = self._pq.ParquetWriter(
            where=str(self.staging_path),
            schema=self.schema_info.schema,
            compression=self.compression,
        )

    @property
    def rows_written(self) -> int:
        """Number of rows flushed to the staging file."""
        return self._rows_written

    @property
    def row_groups_written(self) -> int:
        """Number of Parquet row groups flushed so far."""
        return self._row_groups_written

    @property
    def is_closed(self) -> bool:
        """Whether :meth:`close` has completed."""
        return self._closed

    @property
    def is_published(self) -> bool:
        """Whether :meth:`publish` has completed."""
        return self._published

    def write_row(self, row: Mapping[str, Any]) -> None:
        """
        Buffer one schema-aligned row and flush when the batch is full.

        Parameters
        ----------
        row : mapping
            Flattened evaluation row.

        Raises
        ------
        ParquetWriterError
            If the writer is closed or the row does not match the schema.
        """
        if self._closed:
            raise ParquetWriterError("cannot write to a closed ParquetShardWriter")
        self._buffer.append(self._coerce_row(row))
        if len(self._buffer) >= self.batch_rows:
            self._flush_batch()

    def write_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """
        Buffer many rows, flushing full batches as needed.

        Parameters
        ----------
        rows : iterable of mapping
            Flattened evaluation rows.
        """
        for row in rows:
            self.write_row(row)

    def close(self) -> None:
        """
        Flush remaining rows, close the writer, and validate the staging file.

        Raises
        ------
        ParquetWriterError
            If validation fails after close.
        """
        if self._closed:
            return
        try:
            if self._buffer:
                self._flush_batch()
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            self._closed = True
            self.validate_staging()
        except Exception:
            # Leave staging in place for debugging; mark closed to prevent writes.
            self._closed = True
            if self._writer is not None:
                try:
                    self._writer.close()
                except Exception:
                    pass
                self._writer = None
            raise

    def abandon(self) -> None:
        """
        Close the underlying writer without validation (infrastructure failure).

        Staging files are left in place for debugging. Safe to call multiple
        times.
        """
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        self._buffer.clear()

    def validate_staging(self) -> None:
        """
        Reopen the staging Parquet file and check schema / row counts.

        Raises
        ------
        ParquetWriterError
            If the file is missing, unreadable, or metadata disagree.
        """
        if not self.staging_path.is_file():
            raise ParquetWriterError(
                f"staging parquet missing after close: {self.staging_path}"
            )
        try:
            parquet_file = self._pq.ParquetFile(str(self.staging_path))
        except Exception as exc:
            raise ParquetWriterError(
                f"could not reopen staging parquet {self.staging_path}: {exc}"
            ) from exc

        file_schema = parquet_file.schema_arrow
        if not file_schema.equals(self.schema_info.schema, check_metadata=False):
            raise ParquetWriterError(
                "staging parquet schema does not match expected schema"
            )
        metadata = parquet_file.metadata
        if metadata is None:
            raise ParquetWriterError("staging parquet has no metadata")
        if int(metadata.num_rows) != self._rows_written:
            raise ParquetWriterError(
                f"staging row count {metadata.num_rows} != written "
                f"{self._rows_written}"
            )
        if int(metadata.num_row_groups) != self._row_groups_written:
            raise ParquetWriterError(
                f"staging row groups {metadata.num_row_groups} != flushed "
                f"{self._row_groups_written}"
            )
        self._validated = True

    def publish(self) -> Path:
        """
        Atomically publish the validated staging file to ``destination_path``.

        Returns
        -------
        Path
            Final published part path.

        Raises
        ------
        ParquetWriterError
            If the writer was not closed/validated or publication fails.
        """
        if not self._closed:
            raise ParquetWriterError("close() before publish()")
        if not self._validated:
            self.validate_staging()
        if self._published:
            return self.destination_path.resolve()
        # os.replace overwrites an existing destination, which is required when
        # retrying after an invalid part could not be quarantined/removed.
        published = publish_parquet_file(
            self.staging_path,
            self.destination_path,
            force_cross_filesystem_copy=self.force_cross_filesystem_copy,
        )
        self._published = True
        return published

    def abort(self) -> None:
        """
        Close without publishing and remove the staging partial when present.
        """
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        self._closed = True
        self._buffer.clear()
        if self.staging_path.exists() and not self._published:
            try:
                self.staging_path.unlink()
            except OSError:
                pass

    def __enter__(self) -> ParquetShardWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
            return None
        self.close()
        return None

    def _coerce_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """
        Return a plain dict whose keys exactly match the schema field names.
        """
        expected = [field.name for field in self.schema_info.schema]
        if set(row.keys()) != set(expected):
            raise ParquetWriterError(
                "row fields do not match parquet schema: "
                f"extra={sorted(set(row) - set(expected))} "
                f"missing={sorted(set(expected) - set(row))}"
            )
        return {name: row[name] for name in expected}

    def _flush_batch(self) -> None:
        """
        Write the current buffer as one Parquet row group.
        """
        if not self._buffer:
            return
        if self._writer is None:
            raise ParquetWriterError("parquet writer is not open")
        table = self._pa.Table.from_pylist(
            self._buffer,
            schema=self.schema_info.schema,
        )
        self._writer.write_table(table)
        self._rows_written += len(self._buffer)
        self._row_groups_written += 1
        self._buffer.clear()
