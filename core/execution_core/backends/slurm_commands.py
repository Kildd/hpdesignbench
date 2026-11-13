"""
Injectable Slurm CLI runners used by the array backend.

Production code uses :class:`SubprocessSlurmCommands`. Tests inject fakes so
generation and submission logic can run on Windows without Slurm installed.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Protocol, Sequence


class SlurmCommandError(RuntimeError):
    """
    Raised when a Slurm CLI invocation fails or is unavailable.

    Typical causes: missing ``sbatch``/``squeue``/``sacct``/``scancel`` on
    ``PATH``, nonzero exit status, or empty ``sbatch --parsable`` output.
    """


class SlurmCommands(Protocol):
    """
    Minimal Slurm CLI surface used by the execution backend.

    Implementations must not mutate experiment task state; callers decide
    when to reserve attempts relative to command availability checks.
    """

    def available(self) -> bool:
        """
        Return whether the required Slurm binaries appear to be present.

        Returns
        -------
        bool
            ``True`` when ``sbatch`` (at minimum) can be resolved.
        """
        ...

    def sbatch(self, script_path: Path, *, extra_args: Sequence[str] | None = None) -> str:
        """
        Submit ``script_path`` with ``sbatch --parsable``.

        Parameters
        ----------
        script_path : Path
            Generated job script.
        extra_args : sequence of str or None, optional
            Additional ``sbatch`` arguments inserted before the script path.

        Returns
        -------
        str
            Job id string returned by ``sbatch --parsable``.
        """
        ...

    def squeue(self, args: Sequence[str]) -> str:
        """
        Run ``squeue`` with ``args`` and return combined stdout text.

        Parameters
        ----------
        args : sequence of str
            Arguments after the ``squeue`` executable.

        Returns
        -------
        str
            Command stdout.
        """
        ...

    def sacct(self, args: Sequence[str]) -> str:
        """
        Run ``sacct`` with ``args`` and return combined stdout text.

        Parameters
        ----------
        args : sequence of str
            Arguments after the ``sacct`` executable.

        Returns
        -------
        str
            Command stdout.
        """
        ...

    def scancel(self, args: Sequence[str]) -> str:
        """
        Run ``scancel`` with ``args`` and return combined stdout text.

        Parameters
        ----------
        args : sequence of str
            Arguments after the ``scancel`` executable.

        Returns
        -------
        str
            Command stdout.
        """
        ...


def _which_or_none(name: str) -> str | None:
    """
    Resolve an executable on ``PATH``.

    Parameters
    ----------
    name : str
        Command basename such as ``\"sbatch\"``.

    Returns
    -------
    str or None
        Absolute path, or ``None`` when missing.
    """
    return shutil.which(name)


def _run(argv: list[str]) -> str:
    """
    Run a Slurm CLI command and return stdout.

    Parameters
    ----------
    argv : list of str
        Full argument vector including the executable.

    Returns
    -------
    str
        Stripped stdout text.

    Raises
    ------
    SlurmCommandError
        If the process fails to start or exits nonzero.
    """
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
    except OSError as exc:
        raise SlurmCommandError(f"failed to execute {argv[0]!r}: {exc}") from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise SlurmCommandError(
            f"{argv[0]} exited {completed.returncode}: {stderr or completed.stdout}"
        )
    return (completed.stdout or "").strip()


class SubprocessSlurmCommands:
    """
    Real Slurm CLI runner backed by ``subprocess``.

    Parameters
    ----------
    sbatch_bin : str, optional
        Override path/name for ``sbatch``.
    squeue_bin : str, optional
        Override path/name for ``squeue``.
    sacct_bin : str, optional
        Override path/name for ``sacct``.
    scancel_bin : str, optional
        Override path/name for ``scancel``.
    """

    def __init__(
        self,
        *,
        sbatch_bin: str = "sbatch",
        squeue_bin: str = "squeue",
        sacct_bin: str = "sacct",
        scancel_bin: str = "scancel",
    ) -> None:
        self.sbatch_bin = sbatch_bin
        self.squeue_bin = squeue_bin
        self.sacct_bin = sacct_bin
        self.scancel_bin = scancel_bin

    def available(self) -> bool:
        """
        Return whether ``sbatch`` resolves on ``PATH``.

        Returns
        -------
        bool
            ``True`` when the sbatch binary is found.
        """
        return _which_or_none(self.sbatch_bin) is not None

    def sbatch(self, script_path: Path, *, extra_args: Sequence[str] | None = None) -> str:
        """
        Submit a job script and return the parsable job id.

        Parameters
        ----------
        script_path : Path
            Job script path.
        extra_args : sequence of str or None, optional
            Extra ``sbatch`` flags.

        Returns
        -------
        str
            Job id (first token of ``sbatch --parsable`` output).

        Raises
        ------
        SlurmCommandError
            If ``sbatch`` is missing, fails, or returns an empty id.
        """
        if not self.available():
            raise SlurmCommandError(
                f"sbatch is not available ({self.sbatch_bin!r} not found on PATH)"
            )
        argv = [self.sbatch_bin, "--parsable"]
        if extra_args:
            argv.extend(str(a) for a in extra_args)
        argv.append(str(Path(script_path)))
        raw = _run(argv)
        # sbatch --parsable may return "12345" or "12345;cluster".
        job_id = raw.split(";", 1)[0].strip()
        if not job_id:
            raise SlurmCommandError("sbatch --parsable returned an empty job id")
        return job_id

    def squeue(self, args: Sequence[str]) -> str:
        """
        Run ``squeue``.

        Parameters
        ----------
        args : sequence of str
            Arguments after the executable.

        Returns
        -------
        str
            Stdout text.
        """
        return _run([self.squeue_bin, *[str(a) for a in args]])

    def sacct(self, args: Sequence[str]) -> str:
        """
        Run ``sacct``.

        Parameters
        ----------
        args : sequence of str
            Arguments after the executable.

        Returns
        -------
        str
            Stdout text.
        """
        return _run([self.sacct_bin, *[str(a) for a in args]])

    def scancel(self, args: Sequence[str]) -> str:
        """
        Run ``scancel``.

        Parameters
        ----------
        args : sequence of str
            Arguments after the executable.

        Returns
        -------
        str
            Stdout text.
        """
        return _run([self.scancel_bin, *[str(a) for a in args]])


class RecordingSlurmCommands:
    """
    Test double that records Slurm CLI calls and returns scripted responses.

    Parameters
    ----------
    sbatch_job_ids : sequence of str, optional
        Job ids returned by successive ``sbatch`` calls.
    available_flag : bool, optional
        Value returned by :meth:`available`.
    squeue_output : str, optional
        Default stdout returned by ``squeue``.
    sacct_output : str, optional
        Default stdout returned by ``sacct``.
    squeue_by_job : mapping, optional
        Per-job ``squeue`` stdout overrides keyed by job id.
    sacct_by_job : mapping, optional
        Per-job ``sacct`` stdout overrides keyed by job id.
    scancel_error : str or None, optional
        When set, ``scancel`` raises :class:`SlurmCommandError` with this
        message (no state should be mutated by callers).
    """

    def __init__(
        self,
        *,
        sbatch_job_ids: Sequence[str] | None = None,
        available_flag: bool = True,
        squeue_output: str = "",
        sacct_output: str = "",
        squeue_by_job: Mapping[str, str] | None = None,
        sacct_by_job: Mapping[str, str] | None = None,
        scancel_error: str | None = None,
    ) -> None:
        self.available_flag = bool(available_flag)
        self._job_ids = list(sbatch_job_ids or [])
        self.squeue_output = str(squeue_output)
        self.sacct_output = str(sacct_output)
        self.squeue_by_job = dict(squeue_by_job or {})
        self.sacct_by_job = dict(sacct_by_job or {})
        self.scancel_error = scancel_error
        self.sbatch_calls: list[dict[str, object]] = []
        self.squeue_calls: list[list[str]] = []
        self.sacct_calls: list[list[str]] = []
        self.scancel_calls: list[list[str]] = []

    def available(self) -> bool:
        """
        Return the configured availability flag.

        Returns
        -------
        bool
            Availability override.
        """
        return self.available_flag

    def sbatch(self, script_path: Path, *, extra_args: Sequence[str] | None = None) -> str:
        """
        Record an ``sbatch`` call and return the next scripted job id.

        Parameters
        ----------
        script_path : Path
            Job script path.
        extra_args : sequence of str or None, optional
            Extra flags.

        Returns
        -------
        str
            Next configured job id (defaults to ``\"1000+n\"``).

        Raises
        ------
        SlurmCommandError
            If ``available_flag`` is ``False``.
        """
        if not self.available_flag:
            raise SlurmCommandError("sbatch is not available (recording fake)")
        call = {
            "script_path": Path(script_path),
            "extra_args": list(extra_args or []),
        }
        self.sbatch_calls.append(call)
        if self._job_ids:
            return str(self._job_ids.pop(0))
        return str(1000 + len(self.sbatch_calls) - 1)

    @staticmethod
    def _job_id_from_args(args: Sequence[str]) -> str | None:
        """
        Extract a ``-j JOBID`` value from CLI args when present.

        Parameters
        ----------
        args : sequence of str
            CLI arguments.

        Returns
        -------
        str or None
            Job id token, or ``None``.
        """
        values = [str(a) for a in args]
        for index, token in enumerate(values):
            if token == "-j" and index + 1 < len(values):
                return values[index + 1]
            if token.startswith("-j") and len(token) > 2:
                return token[2:]
        return None

    def squeue(self, args: Sequence[str]) -> str:
        """
        Record an ``squeue`` call and return scripted stdout.

        Parameters
        ----------
        args : sequence of str
            CLI arguments.

        Returns
        -------
        str
            Scripted stdout.
        """
        values = [str(a) for a in args]
        self.squeue_calls.append(values)
        job_id = self._job_id_from_args(values)
        if job_id is not None and job_id in self.squeue_by_job:
            return self.squeue_by_job[job_id]
        return self.squeue_output

    def sacct(self, args: Sequence[str]) -> str:
        """
        Record an ``sacct`` call and return scripted stdout.

        Parameters
        ----------
        args : sequence of str
            CLI arguments.

        Returns
        -------
        str
            Scripted stdout.
        """
        values = [str(a) for a in args]
        self.sacct_calls.append(values)
        job_id = self._job_id_from_args(values)
        if job_id is not None and job_id in self.sacct_by_job:
            return self.sacct_by_job[job_id]
        return self.sacct_output

    def scancel(self, args: Sequence[str]) -> str:
        """
        Record an ``scancel`` call, optionally failing.

        Parameters
        ----------
        args : sequence of str
            CLI arguments.

        Returns
        -------
        str
            Empty stdout placeholder.

        Raises
        ------
        SlurmCommandError
            When ``scancel_error`` is configured.
        """
        values = [str(a) for a in args]
        self.scancel_calls.append(values)
        if self.scancel_error is not None:
            raise SlurmCommandError(self.scancel_error)
        return ""
