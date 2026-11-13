"""
C1.4 gate: sensitivity shard worker evaluation into Parquet.
"""
from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

from core.execution_core.artifacts import (
    create_experiment_layout,
    create_task_layout,
    read_task_status,
    reserve_attempt,
    transition_task_status,
)
from core.execution_core.atomic_io import write_json_atomic
from core.execution_core.config import load_experiment_config
from core.execution_core.manifest import prepare_experiment_artifacts
from core.execution_core.models import TaskRecord, TaskState, format_task_id
from core.ioh_core.problem_builder import build_problem_from_snapshot
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
    sha256_file,
    snapshot_slab_inputs,
)
from core.ioh_core.workflows.sensitivity_analysis.design import (
    seal_sensitivity_design,
)
from core.ioh_core.workflows.sensitivity_analysis.models import SensitivityShardSpec
from core.ioh_core.workflows.sensitivity_analysis.planner import (
    SENSITIVITY_TASK_KIND,
    plan_sensitivity_shard_specs,
)
from core.ioh_core.workflows.sensitivity_analysis.schema import (
    MATRIX_KIND_A,
    MATRIX_KIND_AB,
    MATRIX_KIND_B,
    build_sensitivity_schema,
)
from core.ioh_core.workflows.sensitivity_analysis.worker import (
    build_block_candidates,
    main as worker_main,
    run_worker,
)


def _cheap_analysis(params: dict) -> dict:
    """Deterministic fake analysis keyed by a few geometry values."""
    y = float(params["geom_h_ges_mm"]) + 0.01 * float(params["geom_t_mm"])
    constraint_names = [
        name
        for name in params
        if False
    ]
    # Empty constraint maps are fine; flatten fills schema constraint columns
    # from the sealed schema names via missing → None, but scientific result
    # must still provide mappings for the problem's constraint keys. Tests
    # rebuild schema from the real model, so supply a small fixed set via
    # closure when needed. Here we return empty and let the worker bundle
    # provide real constraint names through a wrapper.
    return {
        "y": y,
        "y_p": y,
        "constraint_values": {},
        "violations_": {},
        "penalties_": {},
        "penalty_sum": 0.0,
        "penalty_factor": 1.0,
        "params": dict(params),
        "analysis_ok": True,
        "constraint_feasible": True,
        "feasible": True,
        "analysis_error": None,
        "remapped_utilizations": [],
        "cost_components_missing": [],
    }


def _wrap_bundle_with_cheap_analysis(bundle: dict) -> dict:
    names = tuple(bundle["constraints"].keys())

    def analysis_fn(params: dict) -> dict:
        result = _cheap_analysis(params)
        result["constraint_values"] = {name: 0.5 for name in names}
        result["violations_"] = {name: 0.0 for name in names}
        result["penalties_"] = {name: 0.0 for name in names}
        return result

    wrapped = dict(bundle)
    wrapped["analysis_fn"] = analysis_fn
    return wrapped


def _write_sensitivity_config(
    path: Path,
    *,
    sample_size: int = 4,
    blocks_per_shard: int = 2,
    factor_names: list[str] | None = None,
) -> Path:
    factors = factor_names or ["geom_h_ges_mm", "geom_t_mm"]
    factor_toml = ", ".join(f'"{name}"' for name in factors)
    path.write_text(
        textwrap.dedent(
            f"""\
            schema_version = 1

            [experiment]
            id = "sensitivity_worker_smoke"
            workflow = "sensitivity_analysis"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [sensitivity_analysis]
            sample_size = {sample_size}
            factor_names = [{factor_toml}]
            response_key = "y"
            scramble = true
            seed = 11
            blocks_per_shard = {blocks_per_shard}
            parquet_batch_rows = 4
            compression = "zstd"
            bootstrap_resamples = 8
            bootstrap_confidence_level = 0.95
            bootstrap_seed = 1
            compute_bootstrap_ci = false
            acknowledge_large_run = true
            maximum_expected_evaluations = 100000
            maximum_expected_bytes = 2000000000

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


def _prepare_sensitivity_experiment(
    root: Path,
    *,
    sample_size: int = 4,
    blocks_per_shard: int = 2,
    factor_names: list[str] | None = None,
    shard_index: int = 0,
) -> tuple[Path, SensitivityShardSpec]:
    factors = factor_names or ["geom_h_ges_mm", "geom_t_mm"]
    config_path = _write_sensitivity_config(
        root / "sensitivity.toml",
        sample_size=sample_size,
        blocks_per_shard=blocks_per_shard,
        factor_names=factors,
    )
    config = load_experiment_config(config_path)
    experiment_dir = root / "experiments" / config.experiment.id
    create_experiment_layout(experiment_dir)
    hashes = snapshot_slab_inputs(
        slab_type=config.experiment.slab_type,
        inputs_dir=experiment_dir / "inputs",
    )
    snapshot_dir = resolve_snapshot_slab_dir(
        experiment_dir / "inputs", config.experiment.slab_type
    )
    design_dir = experiment_dir / "inputs" / "sensitivity"
    document = seal_sensitivity_design(
        snapshot_dir=snapshot_dir,
        slab_type=config.experiment.slab_type,
        problem_id=config.experiment.problem_ids[0],
        factor_names=factors,
        sample_size=sample_size,
        response_key="y",
        scramble=True,
        seed=11,
        design_dir=design_dir,
        overwrite=True,
    )
    for name in (
        "A_unit.npy",
        "B_unit.npy",
        "A_index.npy",
        "B_index.npy",
        "design.json",
    ):
        hashes[f"sensitivity/{name}"] = sha256_file(design_dir / name)

    bundle = build_problem_from_snapshot(
        slab_type=config.experiment.slab_type,
        problem_id=config.experiment.problem_ids[0],
        snapshot_dir=snapshot_dir,
    )
    schema_info = build_sensitivity_schema(
        var_names=bundle["var_names"],
        factor_names=document.factor_names,
        model=bundle["model"],
        constraint_names=tuple(bundle["constraints"].keys()),
    )
    specs = plan_sensitivity_shard_specs(
        sample_size=sample_size,
        factor_count=len(factors),
        blocks_per_shard=blocks_per_shard,
        problem_id=config.experiment.problem_ids[0],
        slab_type=config.experiment.slab_type,
        schema_hash=schema_info.schema_hash,
    )
    spec = specs[shard_index]
    tasks = tuple(
        TaskRecord(
            task_index=index,
            task_id=format_task_id(index),
            kind=SENSITIVITY_TASK_KIND,
            payload=item.to_payload(),
        )
        for index, item in enumerate(specs)
    )
    paths, _manifest, _env = prepare_experiment_artifacts(
        config,
        root=experiment_dir,
        tasks=tasks,
        input_files=hashes,
    )
    create_task_layout(paths, shard_index)
    (experiment_dir / "dataset" / "parts").mkdir(parents=True, exist_ok=True)
    return experiment_dir, spec


def _dispatch_attempt(experiment_dir: Path, task_index: int = 0):
    from core.ioh_core.workflows.sensitivity_analysis.worker import (
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


class TestSensitivityWorker(unittest.TestCase):
    """Worker streaming, packing order, adoption, and failure behaviour."""

    def test_block_candidates_ab_differs_in_one_coordinate(self) -> None:
        from core.ioh_core.workflows.sensitivity_analysis.models import (
            HeldConstantVariable,
            SensitivityDesignDocument,
            SensitivityFactorRecord,
            DESIGN_SCHEMA_VERSION,
            PACKING_RULE,
            INDEX_MAPPING,
            SAMPLING_ENGINE,
            ANALYSIS_ENGINE,
        )

        factors = (
            SensitivityFactorRecord(
                name="a",
                lower_index=0,
                upper_index=1,
                state_count=2,
                catalogue_values=(1.0, 2.0),
            ),
            SensitivityFactorRecord(
                name="b",
                lower_index=0,
                upper_index=2,
                state_count=3,
                catalogue_values=(10.0, 20.0, 30.0),
            ),
        )
        document = SensitivityDesignDocument(
            design_schema_version=DESIGN_SCHEMA_VERSION,
            problem_id="L6750_B2",
            slab_type="hp_slab",
            sample_size=2,
            factor_count=2,
            factor_names=("a", "b"),
            factors=factors,
            held_constant_variables=(
                HeldConstantVariable(name="c", fixed_index=3, catalogue_value=0.0),
            ),
            response_key="y",
            scramble=True,
            seed=1,
            calc_second_order=False,
            packing_rule=PACKING_RULE,
            index_mapping=INDEX_MAPPING,
            expected_evaluations=8,
            sampling_engine=SAMPLING_ENGINE,
            analysis_engine=ANALYSIS_ENGINE,
            numpy_version="0",
            salib_version="0",
            array_sha256={
                "A_unit.npy": "0" * 64,
                "B_unit.npy": "0" * 64,
                "A_index.npy": "0" * 64,
                "B_index.npy": "0" * 64,
            },
        )
        a_row = [0, 1]
        b_row = [1, 2]
        rows = build_block_candidates(
            sample_index=0,
            a_row=a_row,
            b_row=b_row,
            var_names=["a", "b", "c"],
            document=document,
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0][1], [0, 1, 3])  # A
        self.assertEqual(rows[1][1], [1, 1, 3])  # AB0
        self.assertEqual(rows[2][1], [0, 2, 3])  # AB1
        self.assertEqual(rows[3][1], [1, 2, 3])  # B
        # AB0 differs from A only in factor 0.
        self.assertEqual(
            sum(x != y for x, y in zip(rows[0][1], rows[1][1])),
            1,
        )

    def test_worker_writes_salib_order_and_exact_call_count(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir, spec = _prepare_sensitivity_experiment(root)
            attempt = _dispatch_attempt(experiment_dir)

            call_count = {"n": 0}
            real_build = build_problem_from_snapshot

            def fake_build(**kwargs):
                bundle = _wrap_bundle_with_cheap_analysis(real_build(**kwargs))
                original = bundle["analysis_fn"]

                def counted(params):
                    call_count["n"] += 1
                    return original(params)

                bundle["analysis_fn"] = counted
                return bundle

            with patch(
                "core.ioh_core.workflows.sensitivity_analysis.worker.build_problem_from_snapshot",
                side_effect=fake_build,
            ):
                summary = run_worker(
                    experiment_dir=experiment_dir,
                    task_index=0,
                    attempt=attempt.attempt,
                )

            self.assertEqual(summary["rows"], spec.row_count)
            self.assertEqual(call_count["n"], spec.row_count)
            self.assertFalse(summary["adopted_existing_part"])
            part = Path(summary["part_path"])
            table = pq.read_table(part)
            self.assertEqual(table.num_rows, spec.row_count)
            kinds = table.column("matrix_kind").to_pylist()
            step = spec.rows_per_block
            for block in range(spec.block_count):
                block_kinds = kinds[block * step : (block + 1) * step]
                self.assertEqual(block_kinds[0], MATRIX_KIND_A)
                self.assertEqual(block_kinds[-1], MATRIX_KIND_B)
                self.assertTrue(
                    all(kind == MATRIX_KIND_AB for kind in block_kinds[1:-1])
                )
            globals_ = table.column("global_index").to_pylist()
            self.assertEqual(
                globals_,
                list(
                    range(
                        spec.block_start * step,
                        spec.block_stop * step,
                    )
                ),
            )
            status = read_task_status(
                experiment_dir / "tasks" / format_task_id(0)
            )
            self.assertEqual(status["state"], TaskState.COMPLETED.value)

    def test_adopt_valid_part_and_quarantine_invalid(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            experiment_dir, spec = _prepare_sensitivity_experiment(root)
            attempt = _dispatch_attempt(experiment_dir)

            real_build = build_problem_from_snapshot

            def fake_build(**kwargs):
                return _wrap_bundle_with_cheap_analysis(real_build(**kwargs))

            with patch(
                "core.ioh_core.workflows.sensitivity_analysis.worker.build_problem_from_snapshot",
                side_effect=fake_build,
            ):
                summary = run_worker(
                    experiment_dir=experiment_dir,
                    task_index=0,
                    attempt=attempt.attempt,
                )
            part = Path(summary["part_path"])

            task_dir = experiment_dir / "tasks" / format_task_id(0)
            write_json_atomic(
                task_dir / "status.json",
                {
                    "state": TaskState.PENDING.value,
                    "attempt": None,
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "metadata": {},
                },
            )
            attempt2 = _dispatch_attempt(experiment_dir)
            with patch(
                "core.ioh_core.workflows.sensitivity_analysis.worker.build_problem_from_snapshot",
                side_effect=fake_build,
            ):
                summary2 = run_worker(
                    experiment_dir=experiment_dir,
                    task_index=0,
                    attempt=attempt2.attempt,
                )
            self.assertTrue(summary2["adopted_existing_part"])

            # Corrupt the part and ensure quarantine + recompute.
            write_json_atomic(
                task_dir / "status.json",
                {
                    "state": TaskState.PENDING.value,
                    "attempt": None,
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "metadata": {},
                },
            )
            part.write_bytes(b"not-a-parquet-file")
            attempt3 = _dispatch_attempt(experiment_dir)
            with patch(
                "core.ioh_core.workflows.sensitivity_analysis.worker.build_problem_from_snapshot",
                side_effect=fake_build,
            ):
                summary3 = run_worker(
                    experiment_dir=experiment_dir,
                    task_index=0,
                    attempt=attempt3.attempt,
                )
            self.assertFalse(summary3["adopted_existing_part"])
            quarantine = experiment_dir / "dataset" / "quarantine"
            self.assertTrue(quarantine.is_dir())
            self.assertTrue(any(quarantine.iterdir()))
            self.assertTrue(Path(summary3["part_path"]).is_file())
            self.assertEqual(
                pq.read_table(summary3["part_path"]).num_rows, spec.row_count
            )

    def test_infrastructure_failure_writes_failure_json(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir, _spec = _prepare_sensitivity_experiment(root)
            attempt = _dispatch_attempt(experiment_dir)

            with patch(
                "core.ioh_core.workflows.sensitivity_analysis.worker.build_problem_from_snapshot",
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
            failure = (
                experiment_dir
                / "tasks"
                / format_task_id(0)
                / "attempts"
                / f"{attempt.attempt:03d}"
                / "failure.json"
            )
            self.assertTrue(failure.is_file())
            status = read_task_status(
                experiment_dir / "tasks" / format_task_id(0)
            )
            self.assertEqual(status["state"], TaskState.FAILED.value)


if __name__ == "__main__":
    unittest.main()
