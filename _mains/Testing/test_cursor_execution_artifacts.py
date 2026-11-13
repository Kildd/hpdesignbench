"""
Phase A2 gate: atomic artifacts, attempt claims, and status transitions.
"""
from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from core.execution_core.atomic_io import (
    is_temporary_artifact_name,
    read_json,
    write_json_atomic,
)
from core.execution_core.artifacts import (
    TaskStatusError,
    create_experiment_layout,
    create_task_layout,
    read_task_status,
    reserve_attempt,
    transition_task_status,
)
from core.execution_core.models import TaskState


class TestAtomicWrites(unittest.TestCase):
    """Crash-safe publish behaviour for shared JSON artifacts."""

    def test_interrupted_atomic_write_preserves_old_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            write_json_atomic(path, {"state": "pending", "n": 1})

            def _boom(src, dst):
                raise OSError("simulated crash before publish")

            with mock.patch("core.execution_core.atomic_io.os.replace", side_effect=_boom):
                with self.assertRaises(OSError):
                    write_json_atomic(path, {"state": "running", "n": 2})

            self.assertEqual(read_json(path), {"state": "pending", "n": 1})

    def test_temporary_files_are_ignored_when_scanning(self) -> None:
        self.assertTrue(is_temporary_artifact_name("status.json.123_abcd.tmp"))
        self.assertTrue(is_temporary_artifact_name("summary.partial"))
        self.assertFalse(is_temporary_artifact_name("status.json"))
        self.assertFalse(is_temporary_artifact_name("manifest.json"))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json_atomic(root / "status.json", {"state": "pending"})
            (root / "status.json.999_deadbeef.tmp").write_text("partial", encoding="utf-8")
            (root / "summary.partial").write_text("partial", encoding="utf-8")

            visible = sorted(
                p.name
                for p in root.iterdir()
                if not is_temporary_artifact_name(p.name)
            )
            self.assertEqual(visible, ["status.json"])
            self.assertEqual(read_json(root / "status.json"), {"state": "pending"})


class TestExperimentLayoutAndAttempts(unittest.TestCase):
    """Directory layout and exclusive attempt reservation."""

    def test_layout_and_attempt_increment(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiments" / "smoke"
            paths = create_experiment_layout(root)
            self.assertEqual(paths.root, root)
            self.assertTrue(paths.inputs.is_dir())
            self.assertTrue(paths.tasks.is_dir())
            self.assertTrue(paths.aggregate.is_dir())
            self.assertTrue(paths.submissions.is_dir())

            task_dir = create_task_layout(paths, 0)
            self.assertEqual(task_dir.name, "task_000000")
            status = read_task_status(task_dir)
            self.assertEqual(status["state"], TaskState.PENDING.value)
            self.assertEqual(status["attempt"], 0)

            a1 = reserve_attempt(task_dir)
            a2 = reserve_attempt(task_dir)
            self.assertEqual(a1.attempt, 1)
            self.assertEqual(a2.attempt, 2)
            self.assertTrue(a1.attempt_dir.is_dir())
            self.assertTrue(a2.attempt_dir.is_dir())
            self.assertNotEqual(a1.attempt_dir, a2.attempt_dir)

    def test_duplicate_attempt_claim_fails_for_same_index(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = create_experiment_layout(Path(tmp) / "exp")
            task_dir = create_task_layout(paths, 3)
            attempts = task_dir / "attempts"
            # Pre-create 001 so the first reserve must take 002.
            (attempts / "001").mkdir()
            claimed = reserve_attempt(task_dir)
            self.assertEqual(claimed.attempt, 2)

            # Concurrent claim of an existing directory raises FileExistsError
            # inside reserve_attempt and advances; claiming the same path
            # directly still fails.
            with self.assertRaises(FileExistsError):
                claimed.attempt_dir.mkdir(exist_ok=False)


class TestTaskStatusTransitions(unittest.TestCase):
    """Legal status edges, immutability, and UTC timestamps."""

    def _pending_task(self, tmp: str) -> Path:
        paths = create_experiment_layout(Path(tmp) / "exp")
        return create_task_layout(paths, 1)

    def test_happy_path_and_utc_timestamps(self) -> None:
        with TemporaryDirectory() as tmp:
            task_dir = self._pending_task(tmp)
            transition_task_status(
                task_dir,
                expected={TaskState.PENDING},
                new_state=TaskState.DISPATCHED,
                attempt=1,
            )
            transition_task_status(
                task_dir,
                expected={TaskState.DISPATCHED},
                new_state=TaskState.RUNNING,
                attempt=1,
            )
            transition_task_status(
                task_dir,
                expected={TaskState.RUNNING},
                new_state=TaskState.COMPLETED,
                attempt=1,
                metadata={"exit_code": 0},
            )
            status = read_task_status(task_dir)
            self.assertEqual(status["state"], TaskState.COMPLETED.value)
            self.assertEqual(status["attempt"], 1)
            self.assertEqual(status["metadata"]["exit_code"], 0)
            updated = status["updated_at"]
            self.assertTrue(isinstance(updated, str) and updated.endswith("Z"))
            parsed = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            self.assertIsNotNone(parsed.tzinfo)
            self.assertEqual(parsed.utcoffset().total_seconds(), 0)

    def test_invalid_transition_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            task_dir = self._pending_task(tmp)
            with self.assertRaises(TaskStatusError):
                transition_task_status(
                    task_dir,
                    expected={TaskState.PENDING},
                    new_state=TaskState.COMPLETED,
                    attempt=1,
                )

    def test_expected_mismatch_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            task_dir = self._pending_task(tmp)
            with self.assertRaises(TaskStatusError):
                transition_task_status(
                    task_dir,
                    expected={TaskState.RUNNING},
                    new_state=TaskState.COMPLETED,
                    attempt=1,
                )

    def test_completed_status_is_immutable(self) -> None:
        with TemporaryDirectory() as tmp:
            task_dir = self._pending_task(tmp)
            transition_task_status(
                task_dir,
                expected={TaskState.PENDING},
                new_state=TaskState.DISPATCHED,
                attempt=1,
            )
            transition_task_status(
                task_dir,
                expected={TaskState.DISPATCHED},
                new_state=TaskState.RUNNING,
                attempt=1,
            )
            transition_task_status(
                task_dir,
                expected={TaskState.RUNNING},
                new_state=TaskState.COMPLETED,
                attempt=1,
            )
            with self.assertRaises(TaskStatusError):
                transition_task_status(
                    task_dir,
                    expected={TaskState.COMPLETED},
                    new_state=TaskState.FAILED,
                    attempt=2,
                )
            self.assertEqual(
                read_task_status(task_dir)["state"], TaskState.COMPLETED.value
            )

    def test_all_output_stays_in_temporary_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            paths = create_experiment_layout(root / "exp")
            task_dir = create_task_layout(paths, 0)
            ref = reserve_attempt(task_dir)
            write_json_atomic(ref.attempt_dir / "summary.json", {"ok": True})
            transition_task_status(
                task_dir,
                expected={TaskState.PENDING},
                new_state=TaskState.DISPATCHED,
                attempt=ref.attempt,
            )
            for path in root.rglob("*"):
                self.assertTrue(
                    str(path.resolve()).startswith(str(root)),
                    msg=f"artifact escaped TemporaryDirectory: {path}",
                )


class TestModelsBasics(unittest.TestCase):
    """Light checks for frozen model invariants used by artifacts."""

    def test_task_record_id_matches_index(self) -> None:
        from core.execution_core.models import TaskRecord, format_task_id

        rec = TaskRecord(
            task_index=7,
            task_id=format_task_id(7),
            kind="optimization_run",
            payload={"problem_id": "L6750_B2"},
        )
        self.assertEqual(rec.task_id, "task_000007")
        with self.assertRaises(ValueError):
            TaskRecord(
                task_index=7,
                task_id="task_7",
                kind="optimization_run",
            )


if __name__ == "__main__":
    unittest.main()
