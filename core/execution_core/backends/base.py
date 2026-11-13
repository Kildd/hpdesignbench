"""
Backend protocol for dispatching generic experiment tasks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from core.execution_core.models import SubmissionRecord


class ExecutionBackend(Protocol):
    """
    Schedule and run generic task indices for one prepared experiment.

    Implementations must not interpret workflow payloads; they only launch
    the configured worker module with experiment path, task index, and
    attempt index.
    """

    def dispatch(
        self,
        *,
        experiment_dir: Path,
        task_indices: list[int],
        worker_module: str,
    ) -> list[SubmissionRecord]:
        """
        Dispatch ``task_indices`` through this backend.

        Parameters
        ----------
        experiment_dir : Path
            Prepared experiment root containing ``manifest.json``.
        task_indices : list of int
            Zero-based task indices to run (FIFO order).
        worker_module : str
            Import path launched as ``python -m <worker_module>``.

        Returns
        -------
        list of SubmissionRecord
            One or more submission records describing the launched work.
        """
        ...
