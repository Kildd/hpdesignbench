"""
Phase L8 gate: sensitivity per-row evaluation via structural libraries.
"""
from __future__ import annotations

import tempfile
import textwrap
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path

from core.analysis_core.run_config import analysis_run_config_from_params
from core.execution_core.artifacts import (
    reserve_attempt,
    transition_task_status,
)
from core.execution_core.config import load_experiment_config
from core.execution_core.lifecycle import prepare_experiment
from core.execution_core.models import TaskState, format_task_id
from core.ioh_core.cache_eval import EvalContext
from core.ioh_core.problem_builder import build_problem_from_snapshot
from core.ioh_core.structural_artifacts import (
    get_structural_build_count,
    reset_structural_build_count,
    seal_structural_library,
    structural_id_to_structural_indices,
)
from core.ioh_core.structural_artifacts.indexing import (
    candidate_indices_to_structural_id,
)
from core.ioh_core.structural_artifacts.sealing import (
    build_expected_library_identity,
    combined_snapshot_hash,
    slab_csv_input_hashes,
)
from core.ioh_core.workflows.brute_force.indexing import linear_to_indices
from core.ioh_core.workflows.brute_force.structural import (
    finish_variable_names,
    scientific_values_equal,
)
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
)
from core.ioh_core.workflows.sensitivity_analysis.design import (
    load_sensitivity_design,
)
from core.ioh_core.workflows.sensitivity_analysis.worker import (
    build_block_candidates,
    load_npy,
    run_worker,
)
from slab_construction.slabs.hp_slab.analysis import analysis

from _mains.Testing.test_cursor_brute_force_library_mode import (
    _capture_group_artifact,
    _in_memory_library_provider,
)
from _mains.Testing.test_cursor_brute_force_worker import _problem_fixtures

_FINISH_FACTORS = ["geom_t_infill_mm", "geom_t_screed_mm"]


class TestSensitivityLibraryEvalContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _info, space, decode, materials, constraints = _problem_fixtures()
        cls.space = space
        cls.decode = staticmethod(decode)
        cls.finish_names = finish_variable_names(space.var_names)
        params0 = decode(list(linear_to_indices(0, space)))
        cls.run_config = analysis_run_config_from_params(params0)
        cls.analysis_fn = partial(
            analysis,
            constraints=constraints,
            materials=materials,
            run_config=cls.run_config,
        )
        cls.snapshot_hash = "l8-sa-eval"

    def test_per_row_library_equals_compute_zero_builds(self) -> None:
        artifact, _ = _capture_group_artifact(
            decode=self.decode,
            analysis_fn=self.analysis_fn,
            space=self.space,
            finish_names=self.finish_names,
            snapshot_hash=self.snapshot_hash,
            run_config=self.run_config,
            structural_id=0,
        )
        # Finish-only catalogue vectors (structural id 0), as Saltelli would
        # emit when factors are the finish pair and held constants are zero.
        candidates = [
            list(linear_to_indices(i, self.space)) for i in (0, 5, 17)
        ]
        for x in candidates:
            self.assertEqual(
                candidate_indices_to_structural_id(tuple(x), self.space), 0
            )

        compute_ctx = EvalContext(
            self.decode, self.analysis_fn, run_config=self.run_config
        )
        reset_structural_build_count()
        compute_results = [compute_ctx.get(x) for x in candidates]
        self.assertGreaterEqual(get_structural_build_count(), 1)

        provider = _in_memory_library_provider(
            artifact, self.space, self.snapshot_hash
        )
        library_ctx = EvalContext(
            self.decode, self.analysis_fn, run_config=self.run_config
        )
        library_ctx.configure_structural_backend(
            provider=provider,
            structural_id_fn=lambda key: candidate_indices_to_structural_id(
                key, self.space
            ),
            finish_names=self.finish_names,
            snapshot_hash=self.snapshot_hash,
        )
        reset_structural_build_count()
        library_results = [library_ctx.get(x) for x in candidates]
        self.assertEqual(get_structural_build_count(), 0)
        self.assertEqual(provider.metrics.library_hits, 3)

        for left, right in zip(compute_results, library_results):
            self.assertTrue(
                scientific_values_equal(left, right),
                msg="compute/library sensitivity-style rows diverge",
            )


class TestSensitivityLibraryWorker(unittest.TestCase):
    def test_worker_finish_factors_library_zero_builds(self) -> None:
        """
        Saltelli over finish-only factors keeps structural_id=0 → one-ID library.

        Prepare once in compute mode so the sealed sensitivity design is part of
        the snapshot hash, then seal a matching library and re-prepare in library
        mode (same SA seed → identical design digests).
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factor_toml = ", ".join(f'"{n}"' for n in _FINISH_FACTORS)

            def _write_sa_toml(path: Path, *, experiment_id: str, library_path: str | None) -> None:
                structural = ""
                if library_path is not None:
                    structural = textwrap.dedent(
                        f"""\
                        [structural_artifacts]
                        mode = "library"
                        library_path = {library_path!r}
                        """
                    )
                path.write_text(
                    textwrap.dedent(
                        f"""\
                        schema_version = 1

                        [experiment]
                        id = "{experiment_id}"
                        workflow = "sensitivity_analysis"
                        slab_type = "hp_slab"
                        problem_ids = ["L6750_B2"]
                        output_root = "experiments"

                        {structural}
                        [sensitivity_analysis]
                        sample_size = 2
                        factor_names = [{factor_toml}]
                        response_key = "y"
                        scramble = true
                        seed = 11
                        blocks_per_shard = 2
                        parquet_batch_rows = 8
                        compression = "zstd"
                        bootstrap_resamples = 4
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

            compute_toml = root / "sa_compute.toml"
            _write_sa_toml(compute_toml, experiment_id="l8_sa_compute", library_path=None)
            compute_dir = root / "experiments" / "l8_sa_compute"
            prepare_experiment(config_path=compute_toml, output=compute_dir)

            from core.execution_core.manifest import load_manifest

            hashes = dict(load_manifest(compute_dir / "manifest.json").input_files)
            snap = combined_snapshot_hash(
                slab_csv_input_hashes(hashes, slab_type="hp_slab")
            )

            _info, space, decode, materials, constraints = _problem_fixtures()
            finish_names = finish_variable_names(space.var_names)
            self.assertEqual(list(finish_names), _FINISH_FACTORS)
            params0 = decode(list(linear_to_indices(0, space)))
            run_config = analysis_run_config_from_params(params0)
            analysis_fn = partial(
                analysis,
                constraints=constraints,
                materials=materials,
                run_config=run_config,
            )
            artifact, _ = _capture_group_artifact(
                decode=decode,
                analysis_fn=analysis_fn,
                space=space,
                finish_names=finish_names,
                snapshot_hash=snap,
                run_config=run_config,
                structural_id=0,
            )

            lib_dir = root / "ext_lib"
            library_toml = root / "sa_library.toml"
            _write_sa_toml(
                library_toml,
                experiment_id="l8_sa_library",
                library_path=str(lib_dir),
            )
            config = load_experiment_config(library_toml)
            # Identity expected at library prepare uses the same SA design bytes.
            # Build it against the compute experiment inputs (already sealed).
            expected = build_expected_library_identity(
                config=config,
                experiment_dir=compute_dir,
                input_hashes=hashes,
                problem_id=config.experiment.problem_ids[0],
            )
            identity = replace(expected, structural_cardinality=1)
            self.assertEqual(artifact.snapshot_hash, identity.snapshot_hash)
            seal_structural_library(
                lib_dir,
                [
                    (
                        0,
                        structural_id_to_structural_indices(0, space),
                        artifact,
                    )
                ],
                identity=identity,
                rows_per_shard=1,
            )

            experiment_dir = root / "experiments" / "l8_sa_library"
            prepare_experiment(config_path=library_toml, output=experiment_dir)

            design_dir = experiment_dir / "inputs" / "sensitivity"
            document = load_sensitivity_design(design_dir)
            a_index = load_npy(design_dir / "A_index.npy")
            b_index = load_npy(design_dir / "B_index.npy")
            bundle = build_problem_from_snapshot(
                slab_type="hp_slab",
                problem_id="L6750_B2",
                snapshot_dir=resolve_snapshot_slab_dir(
                    experiment_dir / "inputs", "hp_slab"
                ),
            )
            var_names = tuple(str(n) for n in bundle["var_names"])
            for sample_index in range(document.sample_size):
                for _offset, catalogue_indices in build_block_candidates(
                    sample_index=sample_index,
                    a_row=a_index[sample_index],
                    b_row=b_index[sample_index],
                    var_names=var_names,
                    document=document,
                ):
                    sid = candidate_indices_to_structural_id(
                        tuple(int(v) for v in catalogue_indices),
                        space,
                    )
                    self.assertEqual(sid, 0)

            task_dir = experiment_dir / "tasks" / format_task_id(0)
            attempt = reserve_attempt(task_dir)
            transition_task_status(
                task_dir,
                expected={TaskState.PENDING},
                new_state=TaskState.DISPATCHED,
                attempt=attempt.attempt,
            )
            summary = run_worker(
                experiment_dir=experiment_dir,
                task_index=0,
                attempt=attempt.attempt,
            )
            self.assertEqual(summary["structural_artifacts_mode"], "library")
            self.assertGreaterEqual(summary["structural_library_hits"], 1)
            self.assertEqual(summary["structural_builds"], 0)
            self.assertEqual(summary["evaluations"], 8)


if __name__ == "__main__":
    unittest.main()
