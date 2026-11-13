"""
Phase B5 gate: streaming atomic Parquet shard writing.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from core.ioh_core.workflows.brute_force.parquet_writer import (
    ParquetShardWriter,
    ParquetWriterError,
    shard_part_path,
)
from core.ioh_core.workflows.brute_force.schema import (
    BruteForceSchema,
    compute_schema_hash,
)


def _tiny_schema_info() -> BruteForceSchema:
    """
    Return a minimal schema sufficient for writer mechanics tests.
    """
    schema = pa.schema(
        [
            pa.field("global_index", pa.int64(), nullable=False),
            pa.field("shard_index", pa.int32(), nullable=False),
            pa.field("problem_id", pa.string(), nullable=False),
            pa.field("y", pa.float64(), nullable=False),
            pa.field(
                "cost_components_missing",
                pa.list_(pa.string()),
                nullable=False,
            ),
        ]
    )
    return BruteForceSchema(
        schema=schema,
        schema_hash=compute_schema_hash(schema),
        var_names=(),
        param_names=(),
        constraint_names=(),
    )


def _row(index: int, *, shard_index: int = 0, y: float = 1.0) -> dict:
    """
    Build one tiny-schema row.
    """
    return {
        "global_index": int(index),
        "shard_index": int(shard_index),
        "problem_id": "L6750_B2",
        "y": float(y),
        "cost_components_missing": [],
    }


class TestParquetShardWriter(unittest.TestCase):
    """Batches, validation, publication, and failure modes."""

    def test_batches_create_one_row_group_each(self) -> None:
        schema_info = _tiny_schema_info()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "attempt" / "shard-000000.parquet.partial"
            destination = shard_part_path(root / "dataset", 0)
            writer = ParquetShardWriter(
                schema_info=schema_info,
                staging_path=staging,
                destination_path=destination,
                batch_rows=2,
                compression="zstd",
            )
            writer.write_rows([_row(0), _row(1), _row(2), _row(3), _row(4)])
            writer.close()
            self.assertEqual(writer.rows_written, 5)
            self.assertEqual(writer.row_groups_written, 3)  # 2+2+1
            self.assertTrue(staging.is_file())
            self.assertFalse(destination.exists())

            meta = pq.ParquetFile(str(staging)).metadata
            self.assertEqual(meta.num_rows, 5)
            self.assertEqual(meta.num_row_groups, 3)
            self.assertTrue(
                pq.ParquetFile(str(staging)).schema_arrow.equals(
                    schema_info.schema, check_metadata=False
                )
            )

            published = writer.publish()
            self.assertEqual(published, destination.resolve())
            self.assertTrue(destination.is_file())
            self.assertFalse(staging.exists())

    def test_delayed_publication_keeps_destination_absent(self) -> None:
        schema_info = _tiny_schema_info()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging.partial"
            destination = root / "dataset" / "parts" / "shard-000001.parquet"
            writer = ParquetShardWriter(
                schema_info=schema_info,
                staging_path=staging,
                destination_path=destination,
                batch_rows=10,
            )
            writer.write_row(_row(0, shard_index=1))
            writer.close()
            self.assertTrue(staging.is_file())
            self.assertFalse(destination.exists())
            writer.publish()
            self.assertTrue(destination.is_file())
            self.assertFalse(staging.exists())

    def test_interrupted_abort_removes_staging_and_skips_destination(self) -> None:
        schema_info = _tiny_schema_info()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "shard.partial"
            destination = shard_part_path(root / "dataset", 2)
            writer = ParquetShardWriter(
                schema_info=schema_info,
                staging_path=staging,
                destination_path=destination,
                batch_rows=100,
            )
            writer.write_row(_row(0, shard_index=2))
            writer.abort()
            self.assertFalse(staging.exists())
            self.assertFalse(destination.exists())

    def test_invalid_schema_row_rejected(self) -> None:
        schema_info = _tiny_schema_info()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ParquetShardWriter(
                schema_info=schema_info,
                staging_path=root / "a.partial",
                destination_path=root / "a.parquet",
                batch_rows=10,
            )
            with self.assertRaises(ParquetWriterError):
                writer.write_row({"global_index": 0})
            writer.abort()

    def test_publish_before_close_raises(self) -> None:
        schema_info = _tiny_schema_info()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ParquetShardWriter(
                schema_info=schema_info,
                staging_path=root / "b.partial",
                destination_path=root / "b.parquet",
            )
            writer.write_row(_row(0))
            with self.assertRaisesRegex(ParquetWriterError, "close\\(\\) before publish"):
                writer.publish()
            writer.abort()

    def test_force_cross_filesystem_copyback(self) -> None:
        schema_info = _tiny_schema_info()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "node_local" / "shard.partial"
            destination = shard_part_path(root / "dataset", 3)
            writer = ParquetShardWriter(
                schema_info=schema_info,
                staging_path=staging,
                destination_path=destination,
                batch_rows=2,
                force_cross_filesystem_copy=True,
            )
            writer.write_rows([_row(i, shard_index=3) for i in range(3)])
            writer.close()
            writer.publish()
            self.assertTrue(destination.is_file())
            self.assertFalse(staging.exists())
            table = pq.read_table(destination)
            self.assertEqual(table.num_rows, 3)
            self.assertTrue(
                table.schema.equals(schema_info.schema, check_metadata=False)
            )


if __name__ == "__main__":
    unittest.main()
