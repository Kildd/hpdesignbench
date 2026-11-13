"""
Phase B6 gate: brute-force shard worker and structural-artifact reuse.
"""
from __future__ import annotations

import textwrap
import unittest
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow.parquet as pq

from core.analysis_core.run_config import analysis_run_config_from_params
from core.execution_core.artifacts import (
    create_experiment_layout,
    create_task_layout,
    read_task_status,
    reserve_attempt,
    transition_task_status,
)
from core.execution_core.atomic_io import read_json
from core.execution_core.config import load_experiment_config
from core.execution_core.manifest import prepare_experiment_artifacts
from core.execution_core.models import TaskRecord, TaskState, format_task_id
from core.ioh_core.import_specs import (
    build_space,
    load_constraint_defaults,
    load_materials_registry,
    load_param_defaults,
    load_problems_combined,
    make_decode,
)
from core.ioh_core.problem_builder import build_problem_from_snapshot
from core.ioh_core.workflows.brute_force.indexing import linear_to_indices
from core.ioh_core.workflows.brute_force.models import (
    BruteForceShardSpec,
    IndexSpace,
)
from core.ioh_core.workflows.brute_force.planner import (
    BRUTE_FORCE_TASK_KIND,
    aligned_rows_per_shard,
    finish_group_size,
)
from core.ioh_core.workflows.brute_force.schema import (
    build_brute_force_schema,
)
from core.ioh_core.workflows.brute_force.structural import (
    evaluate_finish_group,
    finish_variable_names,
    get_structural_build_count,
    make_structural_key,
    reset_structural_build_count,
    scientific_values_equal,
)
from core.ioh_core.workflows.brute_force.worker import (
    main as worker_main,
    run_worker,
)
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
    snapshot_slab_inputs,
)
from slab_construction.slabs.hp_slab.analysis import analysis

from _mains.Testing._paths import (
    CONSTRAINT_DEFAULTS_CSV,
    MATERIALS_CSV,
    PARAM_DEFAULTS_CSV,
    PROBLEM_LIST_CSV,
)


def _problem_fixtures():
    """
    Load Problem 1 fixtures shared by reuse and worker tests.
    """
    pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
    cdef = load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV)
    problems = load_problems_combined(pdef, cdef, PROBLEM_LIST_CSV)
    info = problems["L6750_B2"]
    var_names, lb_idx, ub_idx = build_space(info["model"])
    space = IndexSpace.from_bounds(var_names, lb_idx, ub_idx)
    decode = make_decode(info["model"], var_names)
    materials = load_materials_registry(MATERIALS_CSV)
    constraints = {
        name: {
            "active": c.get("active", True),
            "enforced": c.get("enforced"),
            "weight": float(c.get("weight", 1)),
            "exponent": float(c.get("exponent", 1)),
        }
        for name, c in info["constraints"].items()
    }
    return info, space, decode, materials, constraints


def _write_bf_config(path: Path, *, rows_per_shard: int = 24) -> Path:
    """
    Write a tiny local brute-force TOML.
    """
    path.write_text(
        textwrap.dedent(
            f"""\
            schema_version = 1

            [experiment]
            id = "bf_worker_smoke"
            workflow = "brute_force"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [brute_force]
            rows_per_shard = {int(rows_per_shard)}
            parquet_batch_rows = 8
            compression = "zstd"
            maximum_expected_rows = 100000
            maximum_expected_bytes = 1000000000
            acknowledge_large_run = true

            [execution]
            backend = "local"
            max_parallel_tasks = 1
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        ),
        encoding="utf-8",
    )
    return path


def _prepare_bf_experiment(
    root: Path,
    *,
    start_index: int,
    stop_index: int,
    schema_hash: str,
) -> Path:
    """
    Manually prepare a sealed brute-force experiment with one shard task.
    """
    config_path = _write_bf_config(root / "bf.toml", rows_per_shard=stop_index - start_index)
    config = load_experiment_config(config_path)
    experiment_dir = root / "experiments" / config.experiment.id
    create_experiment_layout(experiment_dir)
    hashes = snapshot_slab_inputs(
        slab_type=config.experiment.slab_type,
        inputs_dir=experiment_dir / "inputs",
    )
    spec = BruteForceShardSpec(
        problem_id="L6750_B2",
        slab_type="hp_slab",
        shard_index=0,
        start_index=start_index,
        stop_index=stop_index,
        schema_hash=schema_hash,
    )
    tasks = (
        TaskRecord(
            task_index=0,
            task_id=format_task_id(0),
            kind=BRUTE_FORCE_TASK_KIND,
            payload=spec.to_payload(),
        ),
    )
    paths, _manifest, _env = prepare_experiment_artifacts(
        config,
        root=experiment_dir,
        tasks=tasks,
        input_files=hashes,
    )
    create_task_layout(paths, 0)
    (experiment_dir / "dataset" / "parts").mkdir(parents=True, exist_ok=True)
    return experiment_dir


def _dispatch_attempt(experiment_dir: Path, task_index: int = 0):
    """
    Reserve an attempt and mark the task dispatched.
    """
    from core.ioh_core.workflows.brute_force.worker import (
        experiment_paths,
        task_directory,
    )

    paths = experiment_paths(experiment_dir)
    task_dir = task_directory(paths, task_index)
    attempt = reserve_attempt(task_dir)
    transition_task_status(
        task_dir,
        expected={TaskState.PENDING},
        new_state=TaskState.DISPATCHED,
        attempt=attempt.attempt,
    )
    return attempt


class TestStructuralArtifactReuse(unittest.TestCase):
    """Cached-versus-uncached equality and finish-group build counting."""

    @classmethod
    def setUpClass(cls) -> None:
        info, space, decode, materials, constraints = _problem_fixtures()
        cls.info = info
        cls.space = space
        cls.decode = staticmethod(decode)
        cls.materials = materials
        cls.constraints = constraints
        cls.finish_names = finish_variable_names(space.var_names)
        cls.group = finish_group_size(space)
        params0 = decode(list(linear_to_indices(0, space)))
        cls.run_config = analysis_run_config_from_params(params0)
        cls.analysis_fn = partial(
            analysis,
            constraints=constraints,
            materials=materials,
            run_config=cls.run_config,
        )

    def test_cached_equals_uncached_and_one_build_per_group(self) -> None:
        group_params = [
            self.decode(list(linear_to_indices(i, self.space)))
            for i in range(self.group)
        ]
        reset_structural_build_count()
        cached = evaluate_finish_group(
            group_params,
            constraints=self.constraints,
            materials=self.materials,
            analysis_fn=self.analysis_fn,
            finish_names=self.finish_names,
            snapshot_hash="snap-a",
            run_config=self.run_config,
            reuse=True,
        )
        self.assertEqual(get_structural_build_count(), 1)
        self.assertEqual(len(cached), self.group)

        reset_structural_build_count()
        uncached = evaluate_finish_group(
            group_params,
            constraints=self.constraints,
            materials=self.materials,
            analysis_fn=self.analysis_fn,
            finish_names=self.finish_names,
            snapshot_hash="snap-a",
            run_config=self.run_config,
            reuse=False,
        )
        self.assertEqual(get_structural_build_count(), self.group)

        for left, right in zip(cached, uncached):
            self.assertTrue(
                scientific_values_equal(left, right),
                msg="cached and uncached scientific results diverge",
            )

        # Finish thicknesses vary inside the group; objectives therefore move.
        finish_a = self.finish_names[0]
        self.assertTrue(
            any(
                cached[i]["params"][finish_a] != cached[0]["params"][finish_a]
                for i in range(1, self.group)
            )
        )
        self.assertNotEqual(cached[0]["y_p"], cached[-1]["y_p"])

    def test_structural_key_miss_on_structural_or_snapshot_change(self) -> None:
        from core.unit_core import kN_to_N

        params = self.decode(list(linear_to_indices(0, self.space)))
        key_a = make_structural_key(
            params,
            finish_names=self.finish_names,
            snapshot_hash="snap-a",
            n=kN_to_N(params["loads_N_kN"]),
            run_config=self.run_config,
        )
        key_b = make_structural_key(
            params,
            finish_names=self.finish_names,
            snapshot_hash="snap-b",
            n=kN_to_N(params["loads_N_kN"]),
            run_config=self.run_config,
        )
        self.assertNotEqual(key_a, key_b)

        other = self.decode(list(linear_to_indices(self.group, self.space)))
        key_c = make_structural_key(
            other,
            finish_names=self.finish_names,
            snapshot_hash="snap-a",
            n=kN_to_N(other["loads_N_kN"]),
            run_config=self.run_config,
        )
        self.assertNotEqual(key_a, key_c)

        # Finish-only change must hit.
        finish_only = dict(params)
        finish_only[self.finish_names[0]] = other[self.finish_names[0]]
        key_d = make_structural_key(
            finish_only,
            finish_names=self.finish_names,
            snapshot_hash="snap-a",
            n=kN_to_N(finish_only["loads_N_kN"]),
            run_config=self.run_config,
        )
        self.assertEqual(key_a, key_d)

    def test_expected_structural_failure_fans_out_24_complete_results(self) -> None:
        group_params = [
            self.decode(list(linear_to_indices(i, self.space)))
            for i in range(self.group)
        ]
        for params in group_params:
            params["mat_conc_id"] = "NOT_A_REAL_CONCRETE"

        reset_structural_build_count()
        results = evaluate_finish_group(
            group_params,
            constraints=self.constraints,
            materials=self.materials,
            analysis_fn=self.analysis_fn,
            finish_names=self.finish_names,
            snapshot_hash="snap-fail",
            run_config=self.run_config,
            reuse=True,
        )
        self.assertEqual(len(results), self.group)
        self.assertEqual(get_structural_build_count(), 1)
        for index, result in enumerate(results):
            self.assertFalse(result["analysis_ok"])
            self.assertFalse(result["feasible"])
            self.assertIsNotNone(result["analysis_error"])
            self.assertEqual(result["analysis_error"]["stage"], "concrete")
            self.assertEqual(
                result["params"][self.finish_names[0]],
                group_params[index][self.finish_names[0]],
            )
            self.assertEqual(
                result["params"][self.finish_names[1]],
                group_params[index][self.finish_names[1]],
            )


class TestBruteForceWorker(unittest.TestCase):
    """Worker streaming, adoption, and failure behaviour."""

    @classmethod
    def setUpClass(cls) -> None:
        info, space, decode, materials, constraints = _problem_fixtures()
        cls.space = space
        cls.group = finish_group_size(space)
        cls.schema_info = build_brute_force_schema(
            space=space,
            model=info["model"],
            constraint_names=tuple(info["constraints"].keys()),
        )

    def test_planner_aligns_non_multiple_shard_sizes(self) -> None:
        group = self.group
        self.assertEqual(aligned_rows_per_shard(group + 1, group), group)
        self.assertEqual(aligned_rows_per_shard(2 * group + 3, group), 2 * group)

    def test_worker_writes_monotonic_complete_part(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_bf_experiment(
                root,
                start_index=0,
                stop_index=self.group,
                schema_hash=self.schema_info.schema_hash,
            )
            attempt = _dispatch_attempt(experiment_dir)
            summary = run_worker(
                experiment_dir=experiment_dir,
                task_index=0,
                attempt=attempt.attempt,
                reuse_structural=True,
            )
            self.assertEqual(summary["rows"], self.group)
            self.assertFalse(summary["adopted_existing_part"])
            part = Path(summary["part_path"])
            self.assertTrue(part.is_file())
            table = pq.read_table(part)
            self.assertEqual(table.num_rows, self.group)
            indices = table.column("global_index").to_pylist()
            self.assertEqual(indices, list(range(self.group)))
            # Complete parameter columns present.
            self.assertIn("param__geom_h_ges_mm", table.column_names)
            self.assertIn("param__geom_t_infill_mm", table.column_names)
            status = read_task_status(experiment_dir / "tasks" / format_task_id(0))
            self.assertEqual(status["state"], TaskState.COMPLETED.value)

            # Independent decode/analysis spot-check on first and last row.
            snapshot_dir = resolve_snapshot_slab_dir(
                experiment_dir / "inputs", "hp_slab"
            )
            bundle = build_problem_from_snapshot(
                slab_type="hp_slab",
                problem_id="L6750_B2",
                snapshot_dir=snapshot_dir,
            )
            decode = bundle["decode"]
            analysis_fn = bundle["analysis_fn"]
            for linear in (0, self.group - 1):
                indices_vec = linear_to_indices(linear, self.space)
                result = analysis_fn(decode(list(indices_vec)))
                row_y = table.column("y_p").to_pylist()[
                    0 if linear == 0 else self.group - 1
                ]
                self.assertTrue(scientific_values_equal(row_y, result["y_p"]))

    def test_adopt_valid_part_and_quarantine_invalid(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_bf_experiment(
                root,
                start_index=0,
                stop_index=self.group,
                schema_hash=self.schema_info.schema_hash,
            )
            attempt = _dispatch_attempt(experiment_dir)
            summary = run_worker(
                experiment_dir=experiment_dir,
                task_index=0,
                attempt=attempt.attempt,
                reuse_structural=True,
            )
            part = Path(summary["part_path"])

            # Re-run after resetting task state: should adopt.
            task_dir = experiment_dir / "tasks" / format_task_id(0)
            # Force status back for a fresh attempt.
            status_path = task_dir / "status.json"
            status_path.write_text(
                status_path.read_text(encoding="utf-8").replace(
                    '"completed"', '"pending"'
                ),
                encoding="utf-8",
            )
            # status transition helper expects pending; rewrite cleanly.
            from core.execution_core.atomic_io import write_json_atomic

            write_json_atomic(
                status_path,
                {
                    "state": TaskState.PENDING.value,
                    "attempt": None,
                    "updated_at": "2020-01-01T00:00:00+00:00",
                    "metadata": {},
                },
            )
            attempt2 = _dispatch_attempt(experiment_dir)
            summary2 = run_worker(
                experiment_dir=experiment_dir,
                task_index=0,
                attempt=attempt2.attempt,
                reuse_structural=True,
            )
            self.assertTrue(summary2["adopted_existing_part"])
            self.assertTrue(part.is_file())

            # Corrupt the part and ensure quarantine + recompute.
            write_json_atomic(
                status_path,
                {
                    "state": TaskState.PENDING.value,
                    "attempt": None,
                    "updated_at": "2020-01-01T00:00:00+00:00",
                    "metadata": {},
                },
            )
            part.write_bytes(b"not-a-parquet-file")
            attempt3 = _dispatch_attempt(experiment_dir)
            summary3 = run_worker(
                experiment_dir=experiment_dir,
                task_index=0,
                attempt=attempt3.attempt,
                reuse_structural=True,
            )
            self.assertFalse(summary3["adopted_existing_part"])
            self.assertTrue(part.is_file())
            # Fresh part must be valid parquet again (quarantine or replace).
            table = pq.read_table(part)
            self.assertEqual(table.num_rows, self.group)
            quarantine = experiment_dir / "dataset" / "quarantine"
            if quarantine.is_dir():
                self.assertGreaterEqual(len(list(quarantine.glob("*.bad"))), 0)

    def test_unexpected_exception_fails_shard(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_bf_experiment(
                root,
                start_index=0,
                stop_index=self.group,
                schema_hash=self.schema_info.schema_hash,
            )
            attempt = _dispatch_attempt(experiment_dir)
            with patch(
                "core.ioh_core.workflows.brute_force.worker.evaluate_finish_group",
                side_effect=RuntimeError("boom"),
            ):
                code = worker_main(
                    [
                        "--experiment-dir",
                        str(experiment_dir),
                        "--task-index",
                        "0",
                        "--attempt",
                        str(attempt.attempt),
                    ]
                )
            self.assertEqual(code, 1)
            status = read_task_status(experiment_dir / "tasks" / format_task_id(0))
            self.assertEqual(status["state"], TaskState.FAILED.value)
            failure = read_json(attempt.attempt_dir / "failure.json")
            self.assertEqual(failure["error_type"], "RuntimeError")

    def test_nonfinite_y_fails_shard_without_publishing(self) -> None:
        real_eval = evaluate_finish_group

        def _poison(*args, **kwargs):
            results = list(real_eval(*args, **kwargs))
            first = dict(results[0])
            first["y"] = float("inf")
            results[0] = first
            return results

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_bf_experiment(
                root,
                start_index=0,
                stop_index=self.group,
                schema_hash=self.schema_info.schema_hash,
            )
            attempt = _dispatch_attempt(experiment_dir)
            with patch(
                "core.ioh_core.workflows.brute_force.worker.evaluate_finish_group",
                side_effect=_poison,
            ):
                code = worker_main(
                    [
                        "--experiment-dir",
                        str(experiment_dir),
                        "--task-index",
                        "0",
                        "--attempt",
                        str(attempt.attempt),
                    ]
                )
            self.assertEqual(code, 1)
            status = read_task_status(experiment_dir / "tasks" / format_task_id(0))
            self.assertEqual(status["state"], TaskState.FAILED.value)
            failure = read_json(attempt.attempt_dir / "failure.json")
            self.assertEqual(failure["error_type"], "WorkerError")
            self.assertEqual(
                set(failure),
                {"error_type", "message", "traceback"},
            )
            self.assertIn("global_index=", failure["message"])
            self.assertIn("analysis_error_stage=", failure["message"])
            part = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
            self.assertFalse(part.is_file())


if __name__ == "__main__":
    unittest.main()
