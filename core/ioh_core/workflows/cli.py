"""
CLI for prepared experiment lifecycle commands.

Console script (``pyproject.toml``):

``hdb-experiment = core.ioh_core.workflows.cli:main``

Equivalent module invocation:

``python -m core.ioh_core.workflows.cli ...``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.execution_core.config import (
    ExperimentConfig,
    experiment_root_for,
    load_experiment_config,
)
from core.execution_core.lifecycle import (
    ACTIVE_STATES,
    LifecycleError,
    cancel_tasks,
    collect_experiment,
    dispatch_experiment,
    experiment_paths,
    list_task_statuses,
    load_prepared_config,
    prepare_experiment,
    print_experiment_startup,
    reconcile_experiment,
    retry_failed_tasks,
    run_experiment_lifecycle,
    wait_experiment,
)
from core.execution_core.models import TaskState


_SUCCESS_OUTCOMES = frozenset({"completed", "skipped_completed"})
_FAILED_TERMINAL_STATES = frozenset(
    {
        TaskState.FAILED,
        TaskState.TIMED_OUT,
        TaskState.CANCELLED,
    }
)


@dataclass(frozen=True)
class _QueuedExperiment:
    """
    One resolved queue entry before preparation/dispatch.

    Attributes
    ----------
    config_path : Path
        Absolute path to the operator-supplied experiment TOML.
    config : ExperimentConfig
        Validated configuration loaded from ``config_path``.
    experiment_dir : Path
        Resolved experiment root ``{output_root}/{experiment.id}``.
    """

    config_path: Path
    config: ExperimentConfig
    experiment_dir: Path


def build_parser() -> argparse.ArgumentParser:
    """
    Build the ``hdb-experiment`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Root parser with subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="hdb-experiment",
        description="Prepare, dispatch, and collect HpDesignBench experiments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Validate, snapshot, and write tasks")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Experiment directory. Omit it to use "
            "{output_root}/runs/<bucket>/<experiment.id> relative to the config file"
        ),
    )
    prepare.add_argument("--overwrite", action="store_true")

    plan = sub.add_parser(
        "plan",
        help="Validate and print the task plan without writing an experiment",
    )
    plan.add_argument("--config", type=Path, required=True)

    dispatch = sub.add_parser("dispatch", help="Dispatch pending tasks")
    dispatch.add_argument("--experiment-dir", type=Path, required=True)

    status = sub.add_parser("status", help="Print read-only task status counts")
    status.add_argument("--experiment-dir", type=Path, required=True)
    status.add_argument(
        "--reconcile",
        action="store_true",
        help="Optionally reconcile stale active attempts before reporting",
    )

    watch = sub.add_parser(
        "watch",
        help="Poll task counts as a plain table until the experiment is complete",
    )
    watch.add_argument("--experiment-dir", type=Path, required=True)
    watch.add_argument(
        "--interval-seconds",
        type=float,
        default=15.0,
        help="Sleep between polls (default 15)",
    )

    wait = sub.add_parser("wait", help="Block until all tasks are terminal")
    wait.add_argument("--experiment-dir", type=Path, required=True)
    wait.add_argument("--timeout-seconds", type=float, default=None)

    collect = sub.add_parser("collect", help="Aggregate results")
    collect.add_argument("--experiment-dir", type=Path, required=True)
    collect.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write aggregates even when tasks are still non-terminal",
    )

    retry = sub.add_parser("retry", help="Reset failed/timed-out/cancelled tasks")
    retry.add_argument("--experiment-dir", type=Path, required=True)

    cancel = sub.add_parser("cancel", help="Cancel pending/dispatched tasks")
    cancel.add_argument("--experiment-dir", type=Path, required=True)
    cancel.add_argument("--force-stale-seconds", type=float, default=None)

    run = sub.add_parser("run", help="Local prepare + dispatch + collect")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Experiment directory. Omit it to use "
            "{output_root}/runs/<bucket>/<experiment.id> relative to the config file"
        ),
    )
    run.add_argument("--overwrite", action="store_true")

    queue = sub.add_parser(
        "queue",
        help="Prepare and run multiple local experiment configs sequentially",
    )
    queue.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="Experiment TOML path (repeatable; order is execution order)",
    )

    return parser


def _status_payload(experiment_dir: Path, *, reconcile: bool = False) -> dict[str, Any]:
    """
    Build the JSON payload printed by ``status``.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root.
    reconcile : bool, optional
        When ``True``, reconcile before reading.

    Returns
    -------
    dict
        Counts and per-task snapshots.
    """
    if reconcile:
        reconcile_experiment(experiment_dir)
    status = list_task_statuses(experiment_dir)
    return {
        "experiment_dir": str(status.experiment_dir),
        "complete": status.complete,
        "counts": dict(status.counts),
        "tasks": [
            {
                "task_index": snap.task_index,
                "task_id": snap.task_id,
                "state": snap.state.value,
                "attempt": snap.attempt,
            }
            for snap in status.tasks
        ],
    }


def format_status_watch_text(payload: Mapping[str, Any]) -> str:
    """
    Format one watch poll as a short table (not JSON).

    Parameters
    ----------
    payload : mapping
        Output of :func:`_status_payload`.

    Returns
    -------
    str
        Multi-line report ending with a newline.
    """
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        counts = {}
    lines = [
        f"complete: {payload.get('complete')}",
        f"dir: {payload.get('experiment_dir')}",
    ]
    for name in (
        "pending",
        "dispatched",
        "running",
        "completed",
        "failed",
        "timed_out",
        "cancelled",
    ):
        lines.append(f"  {name:12} {int(counts.get(name, 0))}")
    live = [
        task
        for task in payload.get("tasks", ())
        if isinstance(task, Mapping)
        and task.get("state") in {"dispatched", "running", "failed", "timed_out"}
    ]
    lines.append(f"in flight / failed: {len(live)}")
    for task in live:
        lines.append(f"  {task.get('task_id')} {task.get('state')}")
    return "\n".join(lines) + "\n"


def _print_status(experiment_dir: Path, *, reconcile: bool = False) -> int:
    """
    Print status counts as JSON.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root.
    reconcile : bool, optional
        When ``True``, reconcile before reading.

    Returns
    -------
    int
        Exit code ``0``.
    """
    payload = _status_payload(experiment_dir, reconcile=reconcile)
    print(json.dumps(payload, indent=2))
    return 0


def _watch_status(experiment_dir: Path, *, interval_seconds: float) -> int:
    """
    Poll task counts until every task is terminal.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root (``manifest.json`` may appear after prepare).
    interval_seconds : float
        Delay between polls.

    Returns
    -------
    int
        Exit code ``0`` when complete.
    """
    delay = max(0.01, float(interval_seconds))
    root = Path(experiment_dir).expanduser()
    while True:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        manifest = root / "manifest.json"
        if not manifest.is_file():
            print(f"waiting for prepare (no manifest.json yet): {root}", flush=True)
        else:
            try:
                payload = _status_payload(root)
            except (LifecycleError, OSError, ValueError) as exc:
                print(f"[watch] {exc}", flush=True)
            else:
                print(format_status_watch_text(payload), end="", flush=True)
                if payload.get("complete"):
                    return 0
        time.sleep(delay)


def _path_key(path: Path) -> str:
    """
    Return a case-normalized absolute path key for duplicate detection.

    Parameters
    ----------
    path : Path
        Candidate filesystem path.

    Returns
    -------
    str
        Normalized absolute path string.
    """
    return os.path.normcase(str(path.expanduser().resolve()))


def _normalized_config_text(path: Path) -> str:
    """
    Return UTF-8 config text normalized the same way sealing does.

    ``write_source_config`` reads the operator TOML as text and rewrites it
    with a trailing newline. On Windows the live file may still use CRLF while
    the sealed copy uses LF, so reuse checks must compare normalized text
    rather than raw bytes.

    Parameters
    ----------
    path : Path
        Config file path.

    Returns
    -------
    str
        Newline-normalized text ending with ``\\n``.
    """
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    return text


def _resolve_queue_entries(config_paths: Sequence[Path]) -> list[_QueuedExperiment]:
    """
    Load and validate every queued config without writing artifacts.

    Parameters
    ----------
    config_paths : sequence of Path
        Operator-supplied TOML paths in execution order.

    Returns
    -------
    list of _QueuedExperiment
        Resolved entries in the same order.

    Raises
    ------
    LifecycleError
        If a config is missing, non-local, or collides on output path.
    """
    if not config_paths:
        raise LifecycleError("queue requires at least one --config")

    entries: list[_QueuedExperiment] = []
    seen_dirs: dict[str, Path] = {}
    for raw in config_paths:
        config_path = Path(raw).expanduser().resolve()
        if not config_path.is_file():
            raise LifecycleError(f"config not found: {config_path}")
        config = load_experiment_config(config_path)
        if config.execution.backend != "local":
            raise LifecycleError(
                "queue requires backend='local' for every config; "
                f"{config_path} has backend={config.execution.backend!r}"
            )
        base = (
            config.source_path.parent
            if config.source_path is not None
            else Path.cwd()
        )
        experiment_dir = experiment_root_for(config, cwd=base)
        key = _path_key(experiment_dir)
        if key in seen_dirs:
            raise LifecycleError(
                "queue has duplicate experiment directory "
                f"{experiment_dir} from {seen_dirs[key]} and {config_path}"
            )
        seen_dirs[key] = config_path
        entries.append(
            _QueuedExperiment(
                config_path=config_path,
                config=config,
                experiment_dir=experiment_dir,
            )
        )
    return entries


def _prepare_or_reuse_queue_entry(entry: _QueuedExperiment) -> Path:
    """
    Prepare a new experiment directory or safely reuse a matching one.

    Parameters
    ----------
    entry : _QueuedExperiment
        Resolved queue entry.

    Returns
    -------
    Path
        Prepared experiment root.

    Raises
    ------
    LifecycleError
        If the directory is ambiguous, mismatched, or cannot be prepared.
    """
    experiment_dir = entry.experiment_dir
    paths = experiment_paths(experiment_dir)

    if not experiment_dir.exists():
        prepared = prepare_experiment(
            config_path=entry.config_path,
            output=experiment_dir,
            overwrite=False,
        )
        return prepared.root

    if not paths.manifest.is_file():
        raise LifecycleError(
            f"experiment directory exists without manifest.json: {experiment_dir}; "
            "refusing to overwrite or delete an ambiguous partial directory"
        )
    if not paths.source_config.is_file():
        raise LifecycleError(
            f"prepared experiment is missing source-config.toml: {experiment_dir}"
        )

    requested = _normalized_config_text(entry.config_path)
    sealed = _normalized_config_text(paths.source_config)
    if requested != sealed:
        raise LifecycleError(
            "existing experiment source-config.toml does not match "
            f"{entry.config_path}; refusing to reuse {experiment_dir}"
        )

    # Validate sealed artifacts are loadable before treating as reusable.
    load_prepared_config(experiment_dir)
    list_task_statuses(experiment_dir)
    return experiment_dir.resolve()


def _queue_result(
    entry: _QueuedExperiment,
    *,
    outcome: str,
    counts: dict[str, int] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Build one per-config queue result dictionary.

    Parameters
    ----------
    entry : _QueuedExperiment
        Queue entry.
    outcome : str
        Outcome label.
    counts : dict or None, optional
        Task-state counts.
    error : str or None, optional
        Optional error message.

    Returns
    -------
    dict
        JSON-serializable result row.
    """
    return {
        "config": str(entry.config_path),
        "experiment_dir": str(entry.experiment_dir),
        "outcome": outcome,
        "counts": dict(counts or {}),
        "error": error,
    }


def _all_tasks_completed(counts: dict[str, int], *, n_tasks: int) -> bool:
    """
    Return whether every task is in the completed state.

    Parameters
    ----------
    counts : dict
        Task-state counts.
    n_tasks : int
        Total number of tasks.

    Returns
    -------
    bool
        ``True`` only when every task is completed.
    """
    return n_tasks > 0 and int(counts.get(TaskState.COMPLETED.value, 0)) == n_tasks


def _run_one_queue_entry(entry: _QueuedExperiment) -> dict[str, Any]:
    """
    Reconcile, dispatch, and collect one prepared queue entry.

    Parameters
    ----------
    entry : _QueuedExperiment
        Prepared or reusable experiment entry.

    Returns
    -------
    dict
        Per-config result row.

    Raises
    ------
    LifecycleError
        If tasks remain active after reconciliation (safety stop).
    """
    reconcile_experiment(entry.experiment_dir)
    status = list_task_statuses(entry.experiment_dir)
    counts = dict(status.counts)
    n_tasks = len(status.tasks)

    active = [snap for snap in status.tasks if snap.state in ACTIVE_STATES]
    if active:
        raise LifecycleError(
            f"experiment {entry.experiment_dir} still has active tasks "
            f"({', '.join(s.task_id + '=' + s.state.value for s in active)}); "
            "another process may still own it; refusing to start later queue entries"
        )

    if _all_tasks_completed(counts, n_tasks=n_tasks):
        collect_experiment(entry.experiment_dir, require_complete=True)
        return _queue_result(entry, outcome="skipped_completed", counts=counts)

    failed_terminal = [
        snap for snap in status.tasks if snap.state in _FAILED_TERMINAL_STATES
    ]
    if failed_terminal and not any(
        snap.state is TaskState.PENDING for snap in status.tasks
    ):
        # Terminal unsuccessful tasks and no pending work: do not auto-retry.
        try:
            collect_experiment(entry.experiment_dir, require_complete=False)
        except Exception as exc:  # noqa: BLE001 - inspection aggregate is best-effort
            return _queue_result(
                entry,
                outcome="failed",
                counts=counts,
                error=str(exc),
            )
        return _queue_result(
            entry,
            outcome="failed",
            counts=counts,
            error=(
                "terminal unsuccessful tasks present; queue does not auto-retry "
                f"({', '.join(s.task_id + '=' + s.state.value for s in failed_terminal)})"
            ),
        )

    pending = [snap for snap in status.tasks if snap.state is TaskState.PENDING]
    if pending:
        print_experiment_startup(entry.experiment_dir)
        dispatch_experiment(entry.experiment_dir)

    status = list_task_statuses(entry.experiment_dir)
    counts = dict(status.counts)
    n_tasks = len(status.tasks)
    require_complete = status.complete
    try:
        collect_experiment(
            entry.experiment_dir,
            require_complete=require_complete,
        )
    except LifecycleError:
        # Prefer a partial aggregate for inspection when tasks are incomplete.
        collect_experiment(entry.experiment_dir, require_complete=False)
        status = list_task_statuses(entry.experiment_dir)
        counts = dict(status.counts)
        return _queue_result(
            entry,
            outcome="failed",
            counts=counts,
            error="collection requires complete experiment",
        )

    if _all_tasks_completed(counts, n_tasks=n_tasks):
        return _queue_result(entry, outcome="completed", counts=counts)
    return _queue_result(
        entry,
        outcome="failed",
        counts=counts,
        error="not all tasks completed",
    )


def _run_config_queue(entries: Sequence[_QueuedExperiment]) -> tuple[int, dict[str, Any]]:
    """
    Run prepared queue entries sequentially.

    Parameters
    ----------
    entries : sequence of _QueuedExperiment
        Prepared or reusable entries in execution order.

    Returns
    -------
    tuple
        ``(exit_code, summary_payload)``.
    """
    results: list[dict[str, Any]] = []
    stopped = False
    stop_error: str | None = None

    for entry in entries:
        if stopped:
            results.append(
                _queue_result(
                    entry,
                    outcome="not_started",
                    error=stop_error,
                )
            )
            continue
        try:
            result = _run_one_queue_entry(entry)
            results.append(result)
            if result["outcome"] not in _SUCCESS_OUTCOMES:
                # Runtime failure: continue to later configs.
                continue
        except LifecycleError as exc:
            message = str(exc)
            results.append(
                _queue_result(entry, outcome="failed", error=message)
            )
            if "still has active tasks" in message:
                stopped = True
                stop_error = message
            # Other LifecycleErrors: continue to later configs.
        except Exception as exc:  # noqa: BLE001 - per-experiment isolation
            results.append(
                _queue_result(
                    entry,
                    outcome="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    summary = {
        "queue_size": len(entries),
        "results": results,
    }
    ok = all(row["outcome"] in _SUCCESS_OUTCOMES for row in results)
    return (0 if ok else 1, summary)


def _run_queue_command(config_paths: Sequence[Path]) -> int:
    """
    Resolve, prepare, and sequentially execute a local config queue.

    Parameters
    ----------
    config_paths : sequence of Path
        Operator-supplied configs.

    Returns
    -------
    int
        Process exit code.
    """
    entries = _resolve_queue_entries(config_paths)
    prepared: list[_QueuedExperiment] = []
    for entry in entries:
        experiment_dir = _prepare_or_reuse_queue_entry(entry)
        prepared.append(
            _QueuedExperiment(
                config_path=entry.config_path,
                config=entry.config,
                experiment_dir=experiment_dir,
            )
        )
    exit_code, summary = _run_config_queue(prepared)
    print(json.dumps(summary, indent=2))
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector excluding the program name.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            paths = prepare_experiment(
                config_path=args.config,
                output=args.output,
                overwrite=bool(args.overwrite),
            )
            print_experiment_startup(paths.root)
            print(json.dumps({"experiment_dir": str(paths.root)}, indent=2))
            return 0

        if args.command == "plan":
            from core.ioh_core.workflows.optimization_runs.planner import (
                plan_optimization_specs,
            )

            config = load_experiment_config(args.config)
            specs = plan_optimization_specs(config)
            repetitions = config.experiment.repetitions
            print(
                json.dumps(
                    {
                        "tasks": len(specs),
                        "seed_manifest_path": (
                            None
                            if repetitions is None
                            else repetitions.seed_manifest_path
                        ),
                        "seeds": [
                            {
                                "problem_id": spec.problem_id,
                                "algorithm_name": spec.algorithm_name,
                                "repetition": spec.repetition,
                                "random_seed": spec.random_seed,
                            }
                            for spec in specs
                        ],
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "dispatch":
            records = dispatch_experiment(args.experiment_dir)
            print(
                json.dumps(
                    {
                        "submissions": len(records),
                        "task_indices": [
                            idx
                            for record in records
                            for idx in record.task_indices
                        ],
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "status":
            return _print_status(
                args.experiment_dir,
                reconcile=bool(args.reconcile),
            )

        if args.command == "watch":
            return _watch_status(
                args.experiment_dir,
                interval_seconds=float(args.interval_seconds),
            )

        if args.command == "wait":
            status = wait_experiment(
                args.experiment_dir,
                timeout_seconds=args.timeout_seconds,
            )
            print(
                json.dumps(
                    {
                        "complete": status.complete,
                        "counts": dict(status.counts),
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "collect":
            try:
                summary = collect_experiment(
                    args.experiment_dir,
                    require_complete=not bool(args.allow_incomplete),
                )
            except LifecycleError as exc:
                # Incomplete collection is a nonzero exit as specified.
                print(f"[collect] {exc}", file=sys.stderr)
                # Still try to publish a partial aggregate for inspection.
                try:
                    collect_experiment(
                        args.experiment_dir,
                        require_complete=False,
                    )
                except Exception:
                    pass
                return 2
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "retry":
            reset = retry_failed_tasks(args.experiment_dir)
            print(json.dumps({"retried": reset}, indent=2))
            return 0

        if args.command == "cancel":
            cancelled = cancel_tasks(
                args.experiment_dir,
                force_stale_seconds=args.force_stale_seconds,
            )
            print(json.dumps({"cancelled": cancelled}, indent=2))
            return 0

        if args.command == "run":
            summary = run_experiment_lifecycle(
                config_path=args.config,
                output=args.output,
                overwrite=bool(args.overwrite),
            )
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "queue":
            return _run_queue_command(args.config)

        parser.error(f"unknown command {args.command!r}")
        return 2
    except LifecycleError as exc:
        print(f"[hdb-experiment] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[hdb-experiment] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
