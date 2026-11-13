"""
Phase L5 gate: structural_artifacts config, prepare sealing, worker revalidation.
"""
from __future__ import annotations

import tempfile
import textwrap
import tomllib
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

from core.execution_core.config import ConfigError, parse_experiment_config
from core.execution_core.lifecycle import prepare_experiment
from core.ioh_core.structural_artifacts import (
    StructuralArtifactsSealError,
    load_structural_artifacts_seal,
    revalidate_structural_artifacts_seal,
    seal_structural_library,
)
from core.ioh_core.structural_artifacts.fixtures import (
    make_build_failure_artifact,
    make_mk_cleared_artifact,
    make_success_artifact,
)
from core.ioh_core.structural_artifacts.sealing import (
    build_expected_library_identity,
    seal_structural_artifacts_for_experiment,
)
from core.ioh_core.workflows.optimization_runs.snapshot import snapshot_slab_inputs


def _bf_compute_toml(*, reuse: bool = True) -> str:
    return textwrap.dedent(
        f"""\
        schema_version = 1

        [experiment]
        id = "l5_bf_compute"
        workflow = "brute_force"
        slab_type = "hp_slab"
        problem_ids = ["L6750_B2"]
        output_root = "experiments"

        [brute_force]
        rows_per_shard = 24
        max_rows = 24
        acknowledge_large_run = true
        maximum_expected_rows = 1000
        maximum_expected_bytes = 1000000000
        reuse_structural_artifacts = {"true" if reuse else "false"}

        [execution]
        backend = "local"
        max_parallel_tasks = 1
        cpus_per_task = 1

        [execution.local]
        python_executable = ""
        """
    )


def _bf_library_toml(
    library_path: str | None = None,
    *,
    allow_snapshot_hash_mismatch: bool = False,
) -> str:
    path_line = (
        f"library_path = {library_path!r}\n        "
        if library_path is not None
        else ""
    )
    return textwrap.dedent(
        f"""\
        schema_version = 1

        [experiment]
        id = "l5_bf_library"
        workflow = "brute_force"
        slab_type = "hp_slab"
        problem_ids = ["L6750_B2"]
        output_root = "experiments"

        [structural_artifacts]
        mode = "library"
        {path_line}allow_snapshot_hash_mismatch = {
            "true" if allow_snapshot_hash_mismatch else "false"
        }

        [brute_force]
        rows_per_shard = 24
        max_rows = 24
        acknowledge_large_run = true
        maximum_expected_rows = 1000
        maximum_expected_bytes = 1000000000
        reuse_structural_artifacts = true

        [execution]
        backend = "local"
        max_parallel_tasks = 1
        cpus_per_task = 1

        [execution.local]
        python_executable = ""
        """
    )


def _optimization_library_toml(library_path: str) -> str:
    return textwrap.dedent(
        f"""\
        schema_version = 1

        [experiment]
        id = "opt_multi"
        workflow = "optimization_runs"
        slab_type = "hp_slab"
        problem_ids = ["L6750_A1", "L6750_B2"]
        output_root = "experiments"

        [[experiment.algorithms]]
        name = "RANDOM_SEARCH"
        max_evaluations = 5

        [[experiment.algorithms]]
        name = "PYNOMAD_MADS_PB_WARM_START"
        max_evaluations = 5

        [[experiment.algorithms]]
        name = "DFLINT_BBOA_WARM_START"
        max_evaluations = 5
        repetitions = 1

        [experiment.repetitions]
        count = 2
        seed_base = 1

        [experiment.termination]
        mode = "fixed_target"
        allowed_deviation_percent = 5.0
        target_reference = "all"

        [experiment.termination.global_optima]
        L6750_A1 = 100.0
        L6750_B2 = 200.0

        [structural_artifacts]
        mode = "library"
        library_path = {library_path!r}

        [execution]
        backend = "local"
        max_parallel_tasks = 1
        cpus_per_task = 1

        [execution.local]
        python_executable = ""
        """
    )


def _aligned_artifact(seed: int, *, snapshot_hash: str):
    if seed % 3 == 0:
        art = make_success_artifact(seed=seed)
    elif seed % 3 == 1:
        art = make_mk_cleared_artifact(seed=seed)
    else:
        art = make_build_failure_artifact(seed=seed)
    return type(art)(
        key=art.key,
        snapshot_hash=snapshot_hash,
        n=art.n,
        m_u=art.m_u,
        m_cr=art.m_cr,
        mk_curves=art.mk_curves,
        mk_metadata=art.mk_metadata,
        mk_cleared=art.mk_cleared,
        build_failure_result=(
            dict(art.build_failure_result)
            if art.build_failure_result is not None
            else None
        ),
    )


class TestStructuralArtifactsConfig(unittest.TestCase):
    def test_default_compute_when_section_omitted(self) -> None:
        config = parse_experiment_config(tomllib.loads(_bf_compute_toml()))
        self.assertEqual(config.structural_artifacts.mode, "compute")
        self.assertIsNone(config.structural_artifacts.library_path)

    def test_explicit_compute(self) -> None:
        text = _bf_compute_toml() + "\n[structural_artifacts]\nmode = \"compute\"\n"
        config = parse_experiment_config(tomllib.loads(text))
        self.assertEqual(config.structural_artifacts.mode, "compute")

    def test_library_omits_path_for_root_default(self) -> None:
        text = _bf_compute_toml() + "\n[structural_artifacts]\nmode = \"library\"\n"
        config = parse_experiment_config(tomllib.loads(text))
        self.assertEqual(config.structural_artifacts.mode, "library")
        self.assertIsNone(config.structural_artifacts.library_path)

    def test_compute_rejects_path(self) -> None:
        text = (
            _bf_compute_toml()
            + '\n[structural_artifacts]\nmode = "compute"\nlibrary_path = "x"\n'
        )
        with self.assertRaisesRegex(ConfigError, "only allowed when"):
            parse_experiment_config(tomllib.loads(text))

    def test_compute_rejects_snapshot_hash_override(self) -> None:
        text = (
            _bf_compute_toml()
            + "\n[structural_artifacts]\n"
            + 'mode = "compute"\nallow_snapshot_hash_mismatch = true\n'
        )
        with self.assertRaisesRegex(ConfigError, "only allowed when"):
            parse_experiment_config(tomllib.loads(text))

    def test_snapshot_hash_override_requires_boolean(self) -> None:
        text = _bf_library_toml("lib").replace(
            "allow_snapshot_hash_mismatch = false",
            'allow_snapshot_hash_mismatch = "yes"',
        )
        with self.assertRaisesRegex(ConfigError, "must be a boolean"):
            parse_experiment_config(tomllib.loads(text))

    def test_library_rejects_reuse_off(self) -> None:
        text = _bf_library_toml("lib")
        text = text.replace(
            "reuse_structural_artifacts = true",
            "reuse_structural_artifacts = false",
        )
        with self.assertRaisesRegex(ConfigError, "reuse_structural_artifacts=true"):
            parse_experiment_config(tomllib.loads(text))

    def test_library_rejects_builder_workflow(self) -> None:
        text = textwrap.dedent(
            """\
            schema_version = 1

            [experiment]
            id = "sl_bad"
            workflow = "structural_library"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [structural_artifacts]
            mode = "library"
            library_path = "somewhere"

            [structural_library]
            rows_per_shard = 4
            max_ids = 4
            acknowledge_large_run = true
            maximum_expected_ids = 1000
            maximum_expected_bytes = 1000000000

            [execution]
            backend = "local"
            max_parallel_tasks = 1
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        )
        with self.assertRaisesRegex(ConfigError, "not allowed when"):
            parse_experiment_config(tomllib.loads(text))

    def test_optimization_library_accepts_multiple_problem_ids(self) -> None:
        text = textwrap.dedent(
            """\
            schema_version = 1

            [experiment]
            id = "opt_multi"
            workflow = "optimization_runs"
            slab_type = "hp_slab"
            problem_ids = ["L6750_A1", "L6750_B2"]
            output_root = "experiments"

            [[experiment.algorithms]]
            name = "RANDOM_SEARCH"
            max_evaluations = 5

            [experiment.repetitions]
            count = 1
            seed_base = 1

            [structural_artifacts]
            mode = "library"
            library_path = "somewhere"

            [execution]
            backend = "local"
            max_parallel_tasks = 1
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        )
        config = parse_experiment_config(tomllib.loads(text))
        self.assertEqual(
            config.experiment.problem_ids,
            ("L6750_A1", "L6750_B2"),
        )


class TestStructuralArtifactsSealing(unittest.TestCase):
    def test_multi_problem_library_validates_every_problem_before_sealing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe"
            probe.mkdir()
            hashes = snapshot_slab_inputs(
                slab_type="hp_slab", inputs_dir=probe / "inputs"
            )
            from core.execution_core.config import load_experiment_config

            lib_dir = root / "ext_lib"
            config_path = root / "optimization.toml"
            config_path.write_text(
                _optimization_library_toml(str(lib_dir)),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            first_problem = config.experiment.problem_ids[0]
            expected = build_expected_library_identity(
                config=config,
                experiment_dir=probe,
                input_hashes=hashes,
                problem_id=first_problem,
            )
            identity = replace(expected, structural_cardinality=1)
            seal_structural_library(
                lib_dir,
                [
                    (
                        0,
                        (0,) * len(identity.structural_sizes),
                        _aligned_artifact(
                            0, snapshot_hash=identity.snapshot_hash
                        ),
                    )
                ],
                identity=identity,
                rows_per_shard=1,
            )

            second_problem = config.experiment.problem_ids[1]
            with self.assertRaisesRegex(
                StructuralArtifactsSealError,
                rf"problem_id {second_problem!r}",
            ):
                seal_structural_artifacts_for_experiment(
                    config,
                    experiment_dir=probe,
                    input_hashes=hashes,
                )
            self.assertFalse((probe / "structural_artifacts.json").exists())

            compatible_lib = root / "compatible_lib"
            compatible_identity = replace(
                identity,
                compatible_problem_ids=config.experiment.problem_ids,
            )
            seal_structural_library(
                compatible_lib,
                [
                    (
                        0,
                        (0,) * len(compatible_identity.structural_sizes),
                        _aligned_artifact(
                            1,
                            snapshot_hash=compatible_identity.snapshot_hash,
                        ),
                    )
                ],
                identity=compatible_identity,
                rows_per_shard=1,
            )
            compatible_path = root / "optimization_compatible.toml"
            compatible_path.write_text(
                _optimization_library_toml(str(compatible_lib)),
                encoding="utf-8",
            )
            compatible_config = load_experiment_config(compatible_path)
            seal = seal_structural_artifacts_for_experiment(
                compatible_config,
                experiment_dir=probe,
                input_hashes=hashes,
            )
            self.assertEqual(
                seal.identity.compatible_problem_ids,
                config.experiment.problem_ids,
            )

            prepared = root / "prepared"
            prepare_experiment(
                config_path=compatible_path,
                output=prepared,
            )
            from core.execution_core.manifest import load_manifest

            manifest = load_manifest(prepared / "manifest.json")
            self.assertEqual(len(manifest.tasks), 10)
            self.assertEqual(
                [task.task_index for task in manifest.tasks],
                list(range(10)),
            )
            self.assertTrue(
                all(
                    "global_optima" not in task.payload
                    for task in manifest.tasks
                )
            )

    def test_prepare_compute_seal_and_revalidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bf.toml"
            config_path.write_text(_bf_compute_toml(), encoding="utf-8")
            experiment_dir = root / "experiments" / "l5_bf_compute"
            prepare_experiment(config_path=config_path, output=experiment_dir)
            seal = load_structural_artifacts_seal(experiment_dir)
            self.assertEqual(seal.mode, "compute")
            self.assertIsNone(seal.library_path)
            again = revalidate_structural_artifacts_seal(experiment_dir)
            self.assertEqual(again.mode, "compute")

    def test_prepare_library_seal_identity_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Build a truncated sealed library whose scientific fields match
            # Problem 1 inputs that prepare will snapshot.
            probe = root / "probe"
            probe.mkdir()
            hashes = snapshot_slab_inputs(
                slab_type="hp_slab", inputs_dir=probe / "inputs"
            )
            from core.execution_core.config import load_experiment_config

            config_path = root / "bf.toml"
            lib_dir = root / "ext_lib"
            # Placeholder path; rewritten after we know identity is buildable.
            config_path.write_text(_bf_library_toml(str(lib_dir)), encoding="utf-8")
            config = load_experiment_config(config_path)
            expected = build_expected_library_identity(
                config=config,
                experiment_dir=probe,
                input_hashes=hashes,
                problem_id=config.experiment.problem_ids[0],
            )
            cardinality = 4
            library_identity = replace(
                expected, structural_cardinality=cardinality
            )
            n_struct = len(library_identity.structural_sizes)
            rows = [
                (
                    i,
                    tuple(0 for _ in range(n_struct)),
                    _aligned_artifact(
                        20 + i, snapshot_hash=library_identity.snapshot_hash
                    ),
                )
                for i in range(cardinality)
            ]
            seal_structural_library(
                lib_dir,
                rows,
                identity=library_identity,
                rows_per_shard=2,
            )

            experiment_dir = root / "experiments" / "l5_bf_library"
            prepare_experiment(config_path=config_path, output=experiment_dir)
            seal = load_structural_artifacts_seal(experiment_dir)
            self.assertEqual(seal.mode, "library")
            self.assertEqual(Path(seal.library_path), lib_dir.resolve())
            self.assertEqual(seal.identity.structural_cardinality, cardinality)
            self.assertEqual(
                seal.identity.snapshot_hash, library_identity.snapshot_hash
            )
            revalidate_structural_artifacts_seal(experiment_dir)

            # Digest drift must fail closed.
            manifest_path = lib_dir / "manifest.json"
            original = manifest_path.read_text(encoding="utf-8")
            manifest_path.write_text(
                original.replace(
                    library_identity.snapshot_hash,
                    "0" * len(library_identity.snapshot_hash),
                ),
                encoding="utf-8",
            )
            with self.assertRaises(StructuralArtifactsSealError):
                revalidate_structural_artifacts_seal(experiment_dir)

    def test_prepare_omitted_library_path_uses_layout_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe"
            probe.mkdir()
            hashes = snapshot_slab_inputs(
                slab_type="hp_slab", inputs_dir=probe / "inputs"
            )
            from core.execution_core.config import load_experiment_config

            lib_dir = root / "experiments" / "structural_library" / "baseline"
            config_path = root / "bf.toml"
            config_path.write_text(_bf_library_toml(), encoding="utf-8")
            config = load_experiment_config(config_path)
            expected = build_expected_library_identity(
                config=config,
                experiment_dir=probe,
                input_hashes=hashes,
                problem_id=config.experiment.problem_ids[0],
            )
            cardinality = 4
            library_identity = replace(
                expected, structural_cardinality=cardinality
            )
            n_struct = len(library_identity.structural_sizes)
            rows = [
                (
                    i,
                    tuple(0 for _ in range(n_struct)),
                    _aligned_artifact(
                        20 + i, snapshot_hash=library_identity.snapshot_hash
                    ),
                )
                for i in range(cardinality)
            ]
            seal_structural_library(
                lib_dir,
                rows,
                identity=library_identity,
                rows_per_shard=2,
            )
            paths = prepare_experiment(config_path=config_path)
            expected_exp = (
                root / "experiments" / "runs" / "exhaustive" / "l5_bf_library"
            )
            self.assertEqual(paths.root, expected_exp.resolve())
            seal = load_structural_artifacts_seal(paths.root)
            self.assertEqual(seal.mode, "library")
            self.assertEqual(Path(seal.library_path), lib_dir.resolve())
            revalidate_structural_artifacts_seal(paths.root)

    def test_identity_mismatch_blocks_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe"
            probe.mkdir()
            hashes = snapshot_slab_inputs(
                slab_type="hp_slab", inputs_dir=probe / "inputs"
            )
            from core.execution_core.config import load_experiment_config

            lib_dir = root / "ext_lib"
            config_path = root / "bf.toml"
            config_path.write_text(_bf_library_toml(str(lib_dir)), encoding="utf-8")
            config = load_experiment_config(config_path)
            expected = build_expected_library_identity(
                config=config,
                experiment_dir=probe,
                input_hashes=hashes,
                problem_id=config.experiment.problem_ids[0],
            )
            bad = replace(
                expected,
                structural_cardinality=2,
                snapshot_hash="0" * len(expected.snapshot_hash),
            )
            rows = [
                (
                    i,
                    (0,) * len(bad.structural_sizes),
                    _aligned_artifact(i, snapshot_hash=bad.snapshot_hash),
                )
                for i in range(2)
            ]
            seal_structural_library(
                lib_dir, rows, identity=bad, rows_per_shard=2
            )
            with self.assertRaises(StructuralArtifactsSealError):
                seal_structural_artifacts_for_experiment(
                    config,
                    experiment_dir=probe,
                    input_hashes=hashes,
                )

    def test_snapshot_hash_mismatch_can_be_explicitly_forced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe"
            probe.mkdir()
            hashes = snapshot_slab_inputs(
                slab_type="hp_slab", inputs_dir=probe / "inputs"
            )
            from core.execution_core.config import load_experiment_config

            lib_dir = root / "ext_lib"
            config_path = root / "bf.toml"
            config_path.write_text(
                _bf_library_toml(
                    str(lib_dir),
                    allow_snapshot_hash_mismatch=True,
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
            stored_hash = "0" * len(expected.snapshot_hash)
            self.assertNotEqual(stored_hash, expected.snapshot_hash)
            library_identity = replace(
                expected,
                snapshot_hash=stored_hash,
                structural_cardinality=2,
            )
            rows = [
                (
                    i,
                    (0,) * len(library_identity.structural_sizes),
                    _aligned_artifact(i, snapshot_hash=stored_hash),
                )
                for i in range(2)
            ]
            seal_structural_library(
                lib_dir,
                rows,
                identity=library_identity,
                rows_per_shard=2,
            )

            strict_config_path = root / "bf_strict.toml"
            strict_config_path.write_text(
                _bf_library_toml(str(lib_dir)),
                encoding="utf-8",
            )
            strict_config = load_experiment_config(strict_config_path)
            with self.assertRaisesRegex(
                StructuralArtifactsSealError, "snapshot_hash"
            ):
                seal_structural_artifacts_for_experiment(
                    strict_config,
                    experiment_dir=probe,
                    input_hashes=hashes,
                )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                seal = seal_structural_artifacts_for_experiment(
                    config,
                    experiment_dir=probe,
                    input_hashes=hashes,
                )
            mismatch_warnings = [
                item
                for item in caught
                if issubclass(item.category, RuntimeWarning)
                and "forcing structural library snapshot_hash mismatch"
                in str(item.message)
            ]
            self.assertTrue(
                mismatch_warnings,
                msg="expected RuntimeWarning for forced snapshot_hash mismatch",
            )

            self.assertTrue(seal.allow_snapshot_hash_mismatch)
            self.assertEqual(seal.expected_snapshot_hash, expected.snapshot_hash)
            self.assertEqual(seal.library_snapshot_hash, stored_hash)
            loaded = load_structural_artifacts_seal(probe)
            self.assertTrue(loaded.allow_snapshot_hash_mismatch)
            self.assertEqual(loaded.expected_snapshot_hash, expected.snapshot_hash)
            self.assertEqual(loaded.library_snapshot_hash, stored_hash)
            revalidate_structural_artifacts_seal(probe)

    def test_sampling_input_hashes_do_not_change_library_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe"
            probe.mkdir()
            hashes = snapshot_slab_inputs(
                slab_type="hp_slab", inputs_dir=probe / "inputs"
            )
            from core.execution_core.config import load_experiment_config

            config_path = root / "bf.toml"
            config_path.write_text(_bf_library_toml(str(root / "lib")), encoding="utf-8")
            config = load_experiment_config(config_path)
            extra = dict(hashes)
            extra["sampling/index.json"] = "0" * 64
            extra["sampling/rep_000/design.json"] = "1" * 64
            slab_only = build_expected_library_identity(
                config=config,
                experiment_dir=probe,
                input_hashes=hashes,
                problem_id=config.experiment.problem_ids[0],
            )
            with_sampling = build_expected_library_identity(
                config=config,
                experiment_dir=probe,
                input_hashes=extra,
                problem_id=config.experiment.problem_ids[0],
            )
            self.assertEqual(slab_only.snapshot_hash, with_sampling.snapshot_hash)
            self.assertEqual(dict(slab_only.input_hashes), hashes)
            self.assertNotIn(
                "sampling/index.json",
                dict(with_sampling.input_hashes),
            )


if __name__ == "__main__":
    unittest.main()
