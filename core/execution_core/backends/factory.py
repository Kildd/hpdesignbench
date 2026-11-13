"""
Factory for execution backends selected by experiment configuration.
"""
from __future__ import annotations

import sys
from pathlib import Path

from core.execution_core.backends.base import ExecutionBackend
from core.execution_core.backends.local import LocalSubprocessBackend
from core.execution_core.backends.slurm import SlurmArrayBackend
from core.execution_core.config import ExperimentConfig


class BackendFactoryError(ValueError):
    """
    Raised when a configured backend cannot be constructed.

    Typical causes: unknown backend name or incomplete local settings.
    """


def create_execution_backend(config: ExperimentConfig) -> ExecutionBackend:
    """
    Construct the backend named by ``config.execution.backend``.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment configuration.

    Returns
    -------
    ExecutionBackend
        Ready-to-dispatch backend instance.

    Raises
    ------
    BackendFactoryError
        If the backend name is unsupported or Slurm settings are incomplete.
    """
    name = config.execution.backend
    if name == "local":
        executable = config.execution.local.python_executable.strip()
        return LocalSubprocessBackend(
            max_parallel_tasks=config.execution.max_parallel_tasks,
            timeout_seconds=config.execution.timeout_seconds,
            python_executable=executable or sys.executable,
            thread_environment=config.execution.thread_environment,
        )
    if name == "slurm":
        try:
            config.execution.slurm.require_complete()
        except Exception as exc:
            raise BackendFactoryError(str(exc)) from exc
        return SlurmArrayBackend(
            python_executable=config.execution.slurm.python_executable,
            partition=config.execution.slurm.partition,
            account=config.execution.slurm.account,
            memory=config.execution.slurm.memory,
            time_limit=config.execution.slurm.time_limit,
            cpus_per_task=config.execution.cpus_per_task,
            array_concurrency=config.execution.slurm.array_concurrency,
            maximum_array_size=config.execution.slurm.maximum_array_size,
            thread_environment=config.execution.thread_environment,
        )
    raise BackendFactoryError(f"unknown backend {name!r}")
