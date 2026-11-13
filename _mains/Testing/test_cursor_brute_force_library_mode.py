"""
Phase L6 gate: brute-force finish-group evaluation via structural libraries.
"""
from __future__ import annotations

import tempfile
import textwrap
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from core.analysis_core.run_config import analysis_run_config_from_params
from core.execution_core.config import load_experiment_config
from core.execution_core.lifecycle import (
    collect_experiment,
    dispatch_experiment,
    prepare_experiment,
)
from core.ioh_core.structural_artifacts import (
    LibraryStructuralProvider,
    get_structural_build_count,
    reset_structural_build_count,
    seal_structural_library,
    structural_id_to_structural_indices,
)
from core.ioh_core.structural_artifacts.indexing import structural_variable_names
from core.ioh_core.structural_artifacts.provider import StructuralProviderError
from core.ioh_core.structural_artifacts.sealing import (
    build_expected_library_identity,
    combined_snapshot_hash,
)
from core.ioh_core.workflows.brute_force.indexing import linear_to_indices
from core.ioh_core.workflows.brute_force.models import BruteForceShardSpec
from core.ioh_core.workflows.brute_force.schema import build_brute_force_schema
from core.ioh_core.workflows.brute_force.structural import (
    evaluate_finish_group,
    finish_variable_names,
    scientific_values_equal,
)
from core.ioh_core.workflows.brute_force.worker import evaluate_shard_rows
from core.ioh_core.workflows.optimization_runs.snapshot import snapshot_slab_inputs
from slab_construction.slabs.hp_slab.analysis import analysis

from _mains.Testing.test_cursor_brute_force_worker import _problem_fixtures
from _mains.Testing.test_cursor_structural_artifact_provider import _DictLookup


def _capture_group_artifact(
    *,
    decode,
    analysis_fn,
    space,
    finish_names,
    snapshot_hash: str,
    run_config,
    structural_id: int,
):
    group = 24
    group_start = structural_id * group
    group_params = [
        decode(list(linear_to_indices(group_start + i, space)))
        for i in range(group)
    ]
    holder: list = []
    from core.ioh_core.structural_artifacts import make_structural_key
    from core.ioh_core.io_util import _req_param
    from core.unit_core import kN_to_N

    n = kN_to_N(_req_param(group_params[0], "loads_N_kN"))
    key = make_structural_key(
        group_params[0],
        finish_names=finish_names,
        snapshot_hash=snapshot_hash,
        n=float(n),
        run_config=run_config,
    )
    analysis_fn(
        dict(group_params[0]),
        run_config=run_config,
        structural_artifact_holder=holder,
        structural_capture_key=key,
        structural_snapshot_hash=snapshot_hash,
    )
    if not holder:
        raise AssertionError("expected structural artifact capture")
    return holder[0], group_params


def _in_memory_library_provider(artifact, space, snapshot_hash: str):
    """Library provider with sealed ``structural_names`` (in-memory lookup)."""
    names = structural_variable_names(space)
    identity = SimpleNamespace(
        snapshot_hash=snapshot_hash,
        structural_names=names,
    )
    return LibraryStructuralProvider(
        lookup=_DictLookup({0: artifact}, identity=identity),
        structural_names=names,
    )


class TestLibraryFinishGroup(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _info, space, decode, materials, constraints = _problem_fixtures()
        cls.space = space
        cls.decode = staticmethod(decode)
        cls.finish_names = finish_variable_names(space.var_names)
        cls.group = 24
        params0 = decode(list(linear_to_indices(0, space)))
        cls.run_config = analysis_run_config_from_params(params0)
        cls.analysis_fn = partial(
            analysis,
            constraints=constraints,
            materials=materials,
            run_config=cls.run_config,
        )
        cls.snapshot_hash = "l6-finish-group"

    def test_library_equals_compute_reuse_zero_builds(self) -> None:
        artifact, group_params = _capture_group_artifact(
            decode=self.decode,
            analysis_fn=self.analysis_fn,
            space=self.space,
            finish_names=self.finish_names,
            snapshot_hash=self.snapshot_hash,
            run_config=self.run_config,
            structural_id=0,
        )
        reset_structural_build_count()
        compute = evaluate_finish_group(
            group_params,
            constraints={},
            materials={},
            analysis_fn=self.analysis_fn,
            finish_names=self.finish_names,
            snapshot_hash=self.snapshot_hash,
            run_config=self.run_config,
            reuse=True,
        )
        self.assertEqual(get_structural_build_count(), 1)

        provider = _in_memory_library_provider(
            artifact, self.space, self.snapshot_hash
        )
        reset_structural_build_count()
        library = evaluate_finish_group(
            group_params,
            constraints={},
            materials={},
            analysis_fn=self.analysis_fn,
            finish_names=self.finish_names,
            snapshot_hash=self.snapshot_hash,
            run_config=self.run_config,
            reuse=True,
            provider=provider,
            structural_id=0,
        )
        self.assertEqual(get_structural_build_count(), 0)
        self.assertEqual(provider.metrics.library_hits, 1)
        self.assertEqual(len(library), self.group)
        for left, right in zip(compute, library):
            self.assertTrue(
                scientific_values_equal(left, right),
                msg="library and compute-reuse results diverge",
            )

        reset_structural_build_count()
        uncached = evaluate_finish_group(
            group_params,
            constraints={},
            materials={},
            analysis_fn=self.analysis_fn,
            finish_names=self.finish_names,
            snapshot_hash=self.snapshot_hash,
            run_config=self.run_config,
            reuse=False,
        )
        self.assertEqual(get_structural_build_count(), self.group)
        for left, right in zip(library, uncached):
            self.assertTrue(
                scientific_values_equal(left, right),
                msg="library and uncached reference diverge",
            )

    def test_finish_group_snapshot_hash_mismatch_can_be_forced(self) -> None:
        library_hash = "l6-lib-hash"
        experiment_hash = "l6-experiment-hash"
        artifact, group_params = _capture_group_artifact(
            decode=self.decode,
            analysis_fn=self.analysis_fn,
            space=self.space,
            finish_names=self.finish_names,
            snapshot_hash=library_hash,
            run_config=self.run_config,
            structural_id=0,
        )
        names = structural_variable_names(self.space)
        identity = SimpleNamespace(
            snapshot_hash=library_hash,
            structural_names=names,
        )
        strict = LibraryStructuralProvider(
            lookup=_DictLookup({0: artifact}, identity=identity),
            structural_names=names,
        )
        with self.assertRaisesRegex(StructuralProviderError, "snapshot_hash"):
            evaluate_finish_group(
                group_params,
                constraints={},
                materials={},
                analysis_fn=self.analysis_fn,
                finish_names=self.finish_names,
                snapshot_hash=experiment_hash,
                run_config=self.run_config,
                reuse=True,
                provider=strict,
                structural_id=0,
            )

        forced = LibraryStructuralProvider(
            lookup=_DictLookup({0: artifact}, identity=identity),
            structural_names=names,
            allow_snapshot_hash_mismatch=True,
        )
        reset_structural_build_count()
        library = evaluate_finish_group(
            group_params,
            constraints={},
            materials={},
            analysis_fn=self.analysis_fn,
            finish_names=self.finish_names,
            snapshot_hash=experiment_hash,
            run_config=self.run_config,
            reuse=True,
            provider=forced,
            structural_id=0,
        )
        self.assertEqual(get_structural_build_count(), 0)
        self.assertEqual(forced.metrics.library_hits, 1)
        self.assertEqual(len(library), self.group)

    def test_library_rejects_reuse_false(self) -> None:
        provider = LibraryStructuralProvider(
            lookup=_DictLookup({0: _capture_group_artifact(
                decode=self.decode,
                analysis_fn=self.analysis_fn,
                space=self.space,
                finish_names=self.finish_names,
                snapshot_hash=self.snapshot_hash,
                run_config=self.run_config,
                structural_id=0,
            )[0]})
        )
        group_params = [
            self.decode(list(linear_to_indices(i, self.space)))
            for i in range(self.group)
        ]
        with self.assertRaisesRegex(Exception, "requires reuse=True"):
            evaluate_finish_group(
                group_params,
                constraints={},
                materials={},
                analysis_fn=self.analysis_fn,
                finish_names=self.finish_names,
                snapshot_hash=self.snapshot_hash,
                run_config=self.run_config,
                reuse=False,
                provider=provider,
                structural_id=0,
            )

    def test_library_missing_id_fails_closed(self) -> None:
        provider = LibraryStructuralProvider(lookup=_DictLookup({}))
        group_params = [
            self.decode(list(linear_to_indices(i, self.space)))
            for i in range(self.group)
        ]
        with self.assertRaises(StructuralProviderError):
            evaluate_finish_group(
                group_params,
                constraints={},
                materials={},
                analysis_fn=self.analysis_fn,
                finish_names=self.finish_names,
                snapshot_hash=self.snapshot_hash,
                run_config=self.run_config,
                reuse=True,
                provider=provider,
                structural_id=0,
            )


class TestSealedLibraryShardRows(unittest.TestCase):
    def test_evaluate_shard_rows_library_matches_compute(self) -> None:
        info, space, decode, materials, constraints = _problem_fixtures()
        finish_names = finish_variable_names(space.var_names)
        group = 24
        n_groups = 2
        params0 = decode(list(linear_to_indices(0, space)))
        run_config = analysis_run_config_from_params(params0)
        analysis_fn = partial(
            analysis,
            constraints=constraints,
            materials=materials,
            run_config=run_config,
        )
        snapshot_hash = "l6-shard-rows"

        rows = []
        for sid in range(n_groups):
            artifact, _params = _capture_group_artifact(
                decode=decode,
                analysis_fn=analysis_fn,
                space=space,
                finish_names=finish_names,
                snapshot_hash=snapshot_hash,
                run_config=run_config,
                structural_id=sid,
            )
            rows.append(
                (
                    sid,
                    structural_id_to_structural_indices(sid, space),
                    artifact,
                )
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from core.ioh_core.structural_artifacts.models import LibraryIdentity
            from core.ioh_core.structural_artifacts import (
                run_config_fingerprint,
                structural_variable_names,
            )
            from core.ioh_core.structural_artifacts.indexing import (
                structural_space,
            )

            struct = structural_space(space)
            identity = LibraryIdentity(
                slab_type="hp_slab",
                problem_id="L6750_B2",
                snapshot_hash=snapshot_hash,
                run_config_fingerprint=run_config_fingerprint(run_config),
                structural_cardinality=n_groups,
                full_cardinality=space.cardinality,
                finish_names=tuple(finish_names),
                finish_sizes=(int(space.sizes[-2]), int(space.sizes[-1])),
                structural_names=structural_variable_names(space),
                structural_sizes=tuple(int(v) for v in struct.sizes),
            )
            lib_dir = root / "lib"
            seal_structural_library(
                lib_dir, rows, identity=identity, rows_per_shard=2
            )

            from core.ioh_core.structural_artifacts import open_structural_library

            schema_info = build_brute_force_schema(
                space=space,
                model=info["model"],
                constraint_names=tuple(info["constraints"].keys()),
            )
            # Minimal bundle matching evaluate_shard_rows expectations.
            bundle = {
                "decode": decode,
                "analysis_fn": analysis_fn,
                "constraints": constraints,
                "run_config": run_config,
            }
            spec = BruteForceShardSpec(
                shard_index=0,
                start_index=0,
                stop_index=n_groups * group,
                problem_id="L6750_B2",
                slab_type="hp_slab",
                schema_hash=schema_info.schema_hash,
            )

            compute_rows = evaluate_shard_rows(
                bundle=bundle,
                space=space,
                schema_info=schema_info,
                spec=spec,
                snapshot_hash=snapshot_hash,
                reuse_structural=True,
            )
            with open_structural_library(
                lib_dir, expected_identity=identity
            ) as library:
                provider = LibraryStructuralProvider(lookup=library)
                reset_structural_build_count()
                library_rows = evaluate_shard_rows(
                    bundle=bundle,
                    space=space,
                    schema_info=schema_info,
                    spec=spec,
                    snapshot_hash=snapshot_hash,
                    reuse_structural=True,
                    provider=provider,
                )
                self.assertEqual(get_structural_build_count(), 0)
                self.assertEqual(provider.metrics.library_hits, n_groups)

            self.assertEqual(len(compute_rows), len(library_rows))
            skip = {"evaluation_time_s"}
            for left, right in zip(compute_rows, library_rows):
                for key in left:
                    if key in skip:
                        continue
                    self.assertTrue(
                        scientific_values_equal(left[key], right[key]),
                        msg=f"column {key!r} diverges at global_index={left.get('global_index')}",
                    )


class TestBruteForceLibraryExperiment(unittest.TestCase):
    def test_prepare_dispatch_library_shard(self) -> None:
        """
        End-to-end: seal one structural ID, run BF library mode for one group.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe"
            probe.mkdir()
            hashes = snapshot_slab_inputs(
                slab_type="hp_slab", inputs_dir=probe / "inputs"
            )
            snap = combined_snapshot_hash(hashes)

            info, space, decode, materials, constraints = _problem_fixtures()
            finish_names = finish_variable_names(space.var_names)
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
            config_path = root / "bf.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""\
                    schema_version = 1

                    [experiment]
                    id = "l6_bf_library"
                    workflow = "brute_force"
                    slab_type = "hp_slab"
                    problem_ids = ["L6750_B2"]
                    output_root = "experiments"

                    [structural_artifacts]
                    mode = "library"
                    library_path = {str(lib_dir)!r}

                    [brute_force]
                    rows_per_shard = 24
                    max_rows = 24
                    parquet_batch_rows = 24
                    compression = "zstd"
                    reuse_structural_artifacts = true
                    acknowledge_large_run = true
                    maximum_expected_rows = 1000
                    maximum_expected_bytes = 1000000000

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
            config = load_experiment_config(config_path)
            expected = build_expected_library_identity(
                config=config,
                experiment_dir=probe,
                input_hashes=hashes,
                problem_id=config.experiment.problem_ids[0],
            )
            identity = replace(expected, structural_cardinality=1)
            # Re-capture is unnecessary; artifact already uses snap == expected.snapshot_hash
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

            experiment_dir = root / "experiments" / "l6_bf_library"
            prepare_experiment(config_path=config_path, output=experiment_dir)
            dispatch_experiment(experiment_dir)
            collect_experiment(experiment_dir, require_complete=True)

            from core.execution_core.atomic_io import read_json
            import pyarrow.parquet as pq

            summary = read_json(
                experiment_dir / "tasks" / "task_000000" / "attempts" / "001" / "summary.json"
            )
            self.assertEqual(summary["structural_artifacts_mode"], "library")
            self.assertEqual(summary["structural_library_hits"], 1)
            self.assertEqual(summary["structural_builds"], 0)
            table = pq.read_table(experiment_dir / "dataset" / "parts" / "shard-000000.parquet")
            self.assertEqual(table.num_rows, 24)


if __name__ == "__main__":
    unittest.main()
