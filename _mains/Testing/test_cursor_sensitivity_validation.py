"""
C1.5 gate: sensitivity dataset validation, SALib analyse, and publication.
"""
from __future__ import annotations

import shutil
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
    reserve_attempt,
    transition_task_status,
)
from core.execution_core.atomic_io import read_json, write_json_atomic
from core.execution_core.config import load_experiment_config
from core.execution_core.manifest import prepare_experiment_artifacts
from core.execution_core.models import TaskRecord, TaskState, format_task_id
from core.execution_core.output_layout import (
    resolve_results_root,
    saltelli_publication_dir,
)
from core.ioh_core.problem_builder import build_problem_from_snapshot
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
    sha256_file,
    snapshot_slab_inputs,
)
from core.ioh_core.workflows.sensitivity_analysis.aggregation import (
    aggregate_sensitivity_experiment,
)
from core.ioh_core.workflows.sensitivity_analysis.design import seal_sensitivity_design
from core.ioh_core.workflows.sensitivity_analysis.deps import (
    SensitivityDependencyError,
    import_salib,
)
from core.ioh_core.workflows.sensitivity_analysis.estimators import (
    estimate_sobol_indices,
)
from core.ioh_core.workflows.sensitivity_analysis.planner import (
    SENSITIVITY_TASK_KIND,
    plan_sensitivity_shard_specs,
)
from core.ioh_core.workflows.sensitivity_analysis.schema import (
    build_sensitivity_schema,
    write_dataset_schema_json,
)
from core.ioh_core.workflows.sensitivity_analysis.validation import (
    SensitivityValidationError,
    validate_dataset,
    validate_dataset_or_raise,
)
from core.ioh_core.workflows.sensitivity_analysis.worker import run_worker


def _salib_available() -> bool:
    try:
        import_salib()
    except SensitivityDependencyError:
        return False
    return True


def _saltelli_indices_json(experiment_dir: Path) -> Path:
    return (
        saltelli_publication_dir(
            resolve_results_root(experiment_dir),
            experiment_dir.name,
        )
        / "sensitivity_indices.json"
    )


def _saltelli_indices_csv(experiment_dir: Path) -> Path:
    return (
        saltelli_publication_dir(
            resolve_results_root(experiment_dir),
            experiment_dir.name,
        )
        / "sensitivity_indices.csv"
    )


def _cheap_analysis(params: dict) -> dict:
    y = float(params["geom_h_ges_mm"]) + 0.01 * float(params["geom_t_mm"])
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
    compute_bootstrap_ci: bool = False,
    factor_names: list[str] | None = None,
) -> Path:
    factors = factor_names or ["geom_h_ges_mm", "geom_t_mm"]
    factor_toml = ", ".join(f'"{name}"' for name in factors)
    path.write_text(
        textwrap.dedent(
            f"""\
            schema_version = 1

            [experiment]
            id = "sensitivity_validate_smoke"
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
            compute_bootstrap_ci = {"true" if compute_bootstrap_ci else "false"}
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


def _prepare_full_experiment(
    root: Path,
    *,
    sample_size: int = 4,
    blocks_per_shard: int = 2,
    compute_bootstrap_ci: bool = False,
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    factors = ["geom_h_ges_mm", "geom_t_mm"]
    config_path = _write_sensitivity_config(
        root / "sensitivity.toml",
        sample_size=sample_size,
        blocks_per_shard=blocks_per_shard,
        compute_bootstrap_ci=compute_bootstrap_ci,
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
    for index in range(len(specs)):
        create_task_layout(paths, index)
    write_dataset_schema_json(experiment_dir / "dataset", schema_info)
    return experiment_dir


def _dispatch_and_run_all(experiment_dir: Path) -> None:
    from core.execution_core.manifest import load_manifest
    from core.ioh_core.workflows.sensitivity_analysis.worker import (
        experiment_paths,
        task_directory,
    )

    paths = experiment_paths(experiment_dir)
    manifest = load_manifest(paths.manifest)
    real_build = build_problem_from_snapshot

    def fake_build(**kwargs):
        return _wrap_bundle_with_cheap_analysis(real_build(**kwargs))

    with patch(
        "core.ioh_core.workflows.sensitivity_analysis.worker.build_problem_from_snapshot",
        side_effect=fake_build,
    ):
        for task in manifest.tasks:
            task_dir = task_directory(paths, task.task_index)
            attempt = reserve_attempt(task_dir)
            transition_task_status(
                task_dir,
                expected={TaskState.PENDING},
                new_state=TaskState.DISPATCHED,
                attempt=attempt.attempt,
            )
            run_worker(
                experiment_dir=experiment_dir,
                task_index=task.task_index,
                attempt=attempt.attempt,
            )


def _clone_experiment(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for marker in (
        dst / "dataset" / "_SUCCESS",
        dst / "aggregate" / "_SUCCESS",
    ):
        if marker.exists():
            marker.unlink()
    return dst


@unittest.skipUnless(_salib_available(), 'install with: pip install -e ".[sensitivity]"')
class TestSensitivityValidation(unittest.TestCase):
    """Structural validation, regenerate-and-match, and publication markers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory()
        root = Path(cls._tmpdir.name)
        cls.golden = _prepare_full_experiment(root / "golden")
        _dispatch_and_run_all(cls.golden)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def _mutated(self, name: str) -> Path:
        dst = Path(self._tmpdir.name) / name
        return _clone_experiment(self.golden, dst)

    def test_validate_and_aggregate_happy_path(self) -> None:
        experiment_dir = self._mutated("happy")
        result = validate_dataset_or_raise(experiment_dir)
        self.assertTrue(result.ok)
        self.assertTrue(result.design_verified)
        self.assertEqual(result.planned_rows, 16)

        summary = aggregate_sensitivity_experiment(experiment_dir)
        self.assertTrue(summary["dataset_success_marker_written"])
        self.assertTrue(summary["aggregate_success_marker_written"])
        self.assertTrue(summary["scientific_ok"])
        self.assertTrue((experiment_dir / "dataset" / "_SUCCESS").is_file())
        self.assertTrue((experiment_dir / "aggregate" / "_SUCCESS").is_file())

        indices = read_json(_saltelli_indices_json(experiment_dir))
        self.assertEqual(indices["method"], "SALib.analyze.sobol")
        self.assertEqual(indices["confidence_quantity"], None)
        self.assertIn("responses", indices)
        self.assertGreaterEqual(len(indices["responses"]), 1)
        objective = next(
            item for item in indices["responses"] if item["name"] == "y"
        )
        self.assertEqual(objective["status"], "ok")
        self.assertEqual(len(objective["factors"]), 2)
        for factor in objective["factors"]:
            self.assertIsNone(factor["first_order_conf_half_width"])
            self.assertIsNone(factor["total_effect_conf_half_width"])
            self.assertIn("interaction_gap", factor)
        # Full set includes util + feasible binaries (BF factorial parity).
        names = {item["name"] for item in indices["responses"]}
        self.assertTrue(any(name.startswith("util__") for name in names))
        self.assertTrue(any(name.startswith("feasible__") for name in names))

        csv_text = (
            _saltelli_indices_csv(experiment_dir)
        ).read_text(encoding="utf-8")
        self.assertIn("response_name", csv_text)
        self.assertIn("first_order", csv_text)
        self.assertIn("geom_h_ges_mm", csv_text)

    def test_aggregate_matches_salib_on_extracted_y(self) -> None:
        experiment_dir = self._mutated("salib_match")
        summary = aggregate_sensitivity_experiment(experiment_dir)
        self.assertTrue(summary["scientific_ok"])
        indices = read_json(_saltelli_indices_json(experiment_dir))

        from core.ioh_core.workflows.sensitivity_analysis.aggregation import (
            extract_response_arrays,
        )
        from core.ioh_core.workflows.sensitivity_analysis.design import (
            load_sensitivity_design,
        )

        document = load_sensitivity_design(
            experiment_dir / "inputs" / "sensitivity"
        )
        y_a, y_b, y_ab, _ = extract_response_arrays(
            experiment_dir,
            response_key=document.response_key,
            sample_size=document.sample_size,
            factor_count=document.factor_count,
        )
        estimate = estimate_sobol_indices(
            y_a,
            y_b,
            y_ab,
            compute_bootstrap_ci=False,
            num_resamples=8,
            conf_level=0.95,
            seed=1,
        )
        objective = next(
            item for item in indices["responses"] if item["name"] == document.response_key
        )
        for index, factor in enumerate(objective["factors"]):
            self.assertAlmostEqual(
                factor["first_order"], float(estimate.first_order[index]), places=12
            )
            self.assertAlmostEqual(
                factor["total_effect"], float(estimate.total_effect[index]), places=12
            )

    def test_rerun_reproduces_indices_json(self) -> None:
        experiment_dir = self._mutated("rerun")
        first = aggregate_sensitivity_experiment(experiment_dir)
        self.assertTrue(first["scientific_ok"])
        payload1 = (
            _saltelli_indices_json(experiment_dir)
        ).read_bytes()
        second = aggregate_sensitivity_experiment(experiment_dir)
        self.assertTrue(second["scientific_ok"])
        payload2 = (
            _saltelli_indices_json(experiment_dir)
        ).read_bytes()
        self.assertEqual(payload1, payload2)

    def test_missing_part_blocks_dataset_success(self) -> None:
        experiment_dir = self._mutated("missing_part")
        part = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
        part.unlink()
        result = validate_dataset(experiment_dir)
        self.assertFalse(result.ok)
        with self.assertRaises(SensitivityValidationError):
            aggregate_sensitivity_experiment(experiment_dir)
        self.assertFalse((experiment_dir / "dataset" / "_SUCCESS").exists())
        self.assertFalse((experiment_dir / "aggregate" / "_SUCCESS").exists())

    def test_wrong_matrix_kind_order_rejected(self) -> None:
        experiment_dir = self._mutated("bad_kind")
        part = experiment_dir / "dataset" / "parts" / "shard-000000.parquet"
        table = pq.read_table(part)
        import pyarrow as pa

        kinds = table.column("matrix_kind").to_pylist()
        if kinds[0] == "A" and kinds[1] == "AB":
            kinds[0], kinds[1] = "AB", "A"
        table = table.set_column(
            table.schema.get_field_index("matrix_kind"),
            "matrix_kind",
            pa.array(kinds, type=pa.string()),
        )
        pq.write_table(table, part)
        result = validate_dataset(experiment_dir)
        self.assertFalse(result.ok)
        self.assertTrue(any("matrix_kind" in err for err in result.errors))

    def test_regenerate_match_failure_blocks_success(self) -> None:
        experiment_dir = self._mutated("bad_design")
        design = read_json(experiment_dir / "inputs" / "sensitivity" / "design.json")
        design["seed"] = int(design["seed"]) + 1
        write_json_atomic(
            experiment_dir / "inputs" / "sensitivity" / "design.json", design
        )
        # Keep digest entries but change seed → regenerate mismatch.
        result = validate_dataset(experiment_dir)
        self.assertFalse(result.ok)
        self.assertTrue(
            any("regenerate-and-match" in err for err in result.errors)
        )
        self.assertFalse((experiment_dir / "dataset" / "_SUCCESS").exists())

    def test_constant_response_keeps_dataset_without_aggregate_success(self) -> None:
        experiment_dir = self._mutated("constant_y")
        import pyarrow as pa

        for part in sorted(
            (experiment_dir / "dataset" / "parts").glob("shard-*.parquet")
        ):
            table = pq.read_table(part)
            n = table.num_rows
            constants = pa.array([1.0] * n, type=pa.float64())
            # Zero-range every analysed source so no factorial response remains ok.
            for name in table.column_names:
                if name in {"y", "y_p"} or name.startswith("util__"):
                    index = table.schema.get_field_index(name)
                    field = table.schema.field(index)
                    table = table.set_column(index, field, constants)
            pq.write_table(table, part)

        summary = aggregate_sensitivity_experiment(experiment_dir)
        self.assertTrue((experiment_dir / "dataset" / "_SUCCESS").is_file())
        self.assertFalse((experiment_dir / "aggregate" / "_SUCCESS").exists())
        self.assertFalse(summary["scientific_ok"])
        self.assertIsNotNone(summary["scientific_error"])
        self.assertFalse(
            (_saltelli_indices_json(experiment_dir)).exists()
        )

    def test_constant_objective_still_publishes_when_util_varies(self) -> None:
        experiment_dir = self._mutated("constant_objective_only")
        import pyarrow as pa

        for part in sorted(
            (experiment_dir / "dataset" / "parts").glob("shard-*.parquet")
        ):
            table = pq.read_table(part)
            n = table.num_rows
            constants = pa.array([1.0] * n, type=pa.float64())
            for name in ("y", "y_p"):
                index = table.schema.get_field_index(name)
                field = table.schema.field(index)
                table = table.set_column(index, field, constants)
            # Cheap fixture emits constant util=0.5; inject variance so util/feasible
            # remain analyzable while the objective is constant.
            for name in table.column_names:
                if not str(name).startswith("util__"):
                    continue
                index = table.schema.get_field_index(name)
                field = table.schema.field(index)
                varying = pa.array(
                    [0.1 + 0.05 * (i % 7) for i in range(n)],
                    type=pa.float64(),
                )
                table = table.set_column(index, field, varying)
            pq.write_table(table, part)

        summary = aggregate_sensitivity_experiment(experiment_dir)
        self.assertTrue(summary["scientific_ok"])
        self.assertTrue((experiment_dir / "aggregate" / "_SUCCESS").is_file())
        indices = read_json(_saltelli_indices_json(experiment_dir))
        by_name = {item["name"]: item for item in indices["responses"]}
        self.assertEqual(by_name["y"]["status"], "constant")
        self.assertTrue(
            any(
                item["status"] == "ok" and str(item["name"]).startswith("util__")
                for item in indices["responses"]
            )
        )

    def test_bootstrap_ci_publishes_half_widths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = _prepare_full_experiment(
                root, compute_bootstrap_ci=True
            )
            _dispatch_and_run_all(experiment_dir)
            summary = aggregate_sensitivity_experiment(experiment_dir)
            self.assertTrue(summary["scientific_ok"])
            indices = read_json(
                _saltelli_indices_json(experiment_dir)
            )
            self.assertEqual(
                indices["confidence_quantity"], "bootstrap_confidence_half_width"
            )
            objective = next(
                item for item in indices["responses"] if item["name"] == "y"
            )
            for factor in objective["factors"]:
                self.assertIsInstance(factor["first_order_conf_half_width"], float)
                self.assertIsInstance(factor["total_effect_conf_half_width"], float)


if __name__ == "__main__":
    unittest.main()
