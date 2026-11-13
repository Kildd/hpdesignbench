"""
Phase L7 gate: optimization EvalContext library-backed cache misses.
"""
from __future__ import annotations

import tempfile
import textwrap
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path

from core.analysis_core.run_config import analysis_run_config_from_params
from core.execution_core.config import load_experiment_config
from core.execution_core.lifecycle import prepare_experiment
from core.ioh_core.cache_eval import EvalContext
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
)
from core.ioh_core.workflows.brute_force.indexing import linear_to_indices
from core.ioh_core.workflows.brute_force.structural import (
    finish_variable_names,
    scientific_values_equal,
)
from core.ioh_core.workflows.optimization_runs.snapshot import snapshot_slab_inputs
from slab_construction.slabs.hp_slab.analysis import analysis

from _mains.Testing.test_cursor_brute_force_library_mode import (
    _capture_group_artifact,
    _in_memory_library_provider,
)
from _mains.Testing.test_cursor_brute_force_worker import _problem_fixtures


class TestEvalContextLibraryMiss(unittest.TestCase):
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
        cls.snapshot_hash = "l7-evalcontext"

    def test_library_equals_compute_and_preserves_hit_miss(self) -> None:
        artifact, _ = _capture_group_artifact(
            decode=self.decode,
            analysis_fn=self.analysis_fn,
            space=self.space,
            finish_names=self.finish_names,
            snapshot_hash=self.snapshot_hash,
            run_config=self.run_config,
            structural_id=0,
        )
        # Candidates in structural group 0 with different finishes.
        candidates = [
            list(linear_to_indices(i, self.space)) for i in (0, 1, 23)
        ]

        compute_ctx = EvalContext(
            self.decode, self.analysis_fn, run_config=self.run_config
        )
        compute_ctx.ensure_params(list(self.space.var_names))
        reset_structural_build_count()
        compute_results = [compute_ctx.get(x) for x in candidates]
        compute_builds = get_structural_build_count()
        self.assertGreaterEqual(compute_builds, 1)
        self.assertEqual(compute_ctx.misses, 3)
        self.assertEqual(compute_ctx.hits, 0)
        # Repeat last candidate -> hit.
        compute_ctx.get(candidates[-1])
        self.assertEqual(compute_ctx.hits, 1)
        self.assertEqual(compute_ctx.misses, 3)

        provider = _in_memory_library_provider(
            artifact, self.space, self.snapshot_hash
        )
        library_ctx = EvalContext(
            self.decode, self.analysis_fn, run_config=self.run_config
        )
        library_ctx.ensure_params(list(self.space.var_names))
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
        self.assertEqual(library_ctx.misses, 3)
        self.assertEqual(library_ctx.hits, 0)
        library_ctx.get(candidates[-1])
        self.assertEqual(library_ctx.hits, 1)
        self.assertEqual(library_ctx.misses, 3)

        for left, right in zip(compute_results, library_results):
            self.assertTrue(
                scientific_values_equal(left, right),
                msg="compute/library EvalContext results diverge",
            )

    def test_reset_preserves_structural_backend(self) -> None:
        artifact, _ = _capture_group_artifact(
            decode=self.decode,
            analysis_fn=self.analysis_fn,
            space=self.space,
            finish_names=self.finish_names,
            snapshot_hash=self.snapshot_hash,
            run_config=self.run_config,
            structural_id=0,
        )
        provider = _in_memory_library_provider(
            artifact, self.space, self.snapshot_hash
        )
        ctx = EvalContext(self.decode, self.analysis_fn, run_config=self.run_config)
        ctx.configure_structural_backend(
            provider=provider,
            structural_id_fn=lambda key: candidate_indices_to_structural_id(
                key, self.space
            ),
            finish_names=self.finish_names,
            snapshot_hash=self.snapshot_hash,
        )
        x0 = list(linear_to_indices(0, self.space))
        reset_structural_build_count()
        ctx.get(x0)
        ctx.reset()
        self.assertEqual(ctx.hits, 0)
        self.assertEqual(ctx.misses, 0)
        ctx.get(x0)
        self.assertEqual(get_structural_build_count(), 0)
        self.assertEqual(provider.metrics.library_hits, 2)


class TestOptimizationLibraryWorker(unittest.TestCase):
    def test_worker_library_mode_zero_builds(self) -> None:
        """
        Tiny RANDOM_SEARCH run over finish indices of structural_id 0 only.

        Uses a custom algorithm that only evaluates candidates in group 0 so a
        one-ID sealed library is sufficient.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe"
            probe.mkdir()
            hashes = snapshot_slab_inputs(
                slab_type="hp_slab", inputs_dir=probe / "inputs"
            )
            snap = combined_snapshot_hash(hashes)

            _info, space, decode, materials, constraints = _problem_fixtures()
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
            config_path = root / "opt.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""\
                    schema_version = 1

                    [experiment]
                    id = "l7_opt_library"
                    workflow = "optimization_runs"
                    slab_type = "hp_slab"
                    problem_ids = ["L6750_B2"]
                    output_root = "experiments"

                    [[experiment.algorithms]]
                    name = "RANDOM_SEARCH"
                    max_evaluations = 3

                    [experiment.repetitions]
                    count = 1
                    seed_base = 0

                    [structural_artifacts]
                    mode = "library"
                    library_path = {str(lib_dir)!r}

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

            # Patch RANDOM_SEARCH to only sample finish indices within sid 0.
            import core.ioh_core.algorithms.native.random_search as rs_mod

            class _FinishOnlySearch(rs_mod.RandomSearch):
                def __call__(self, problem, *, run_controller=None):
                    from core.ioh_core.algorithms.common import (
                        resolve_run_controller,
                    )
                    import numpy as np

                    controller = resolve_run_controller(
                        problem,
                        max_evaluations=self.evaluations,
                        run_controller=run_controller,
                    )
                    lb = np.asarray(problem.bounds.lb, dtype=int)
                    ub = np.asarray(problem.bounds.ub, dtype=int)
                    best_y = None
                    best_x = None
                    for i in range(self.evaluations):
                        if not controller.can_evaluate():
                            break
                        controller.require_evaluation()
                        x = lb.copy()
                        # Vary only the last two finish dimensions within bounds.
                        x[-2] = int(i % (ub[-2] - lb[-2] + 1))
                        x[-1] = int((i // 2) % (ub[-1] - lb[-1] + 1))
                        y = problem(x)
                        controller.observe_evaluation(
                            y,
                        )
                        if best_y is None or y < best_y:
                            best_y, best_x = y, x.copy()
                        if controller.should_stop():
                            break
                    self.best_y_ = float("inf") if best_y is None else float(best_y)
                    self.best_x_ = (
                        [int(v) for v in lb]
                        if best_x is None
                        else [int(v) for v in best_x]
                    )
                    return self

            # Patch factory so create_algorithm uses finish-only search.
            # Call run_worker in-process (dispatch would spawn a clean interpreter).
            import core.ioh_core.algorithms.native.factory as native_factory
            from core.execution_core.artifacts import reserve_attempt
            from core.execution_core.models import TaskState, format_task_id
            from core.execution_core.artifacts import transition_task_status
            from core.ioh_core.workflows.optimization_runs.worker import run_worker

            original = native_factory.RandomSearch
            native_factory.RandomSearch = _FinishOnlySearch
            try:
                experiment_dir = root / "experiments" / "l7_opt_library"
                prepare_experiment(config_path=config_path, output=experiment_dir)
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
            finally:
                native_factory.RandomSearch = original

            self.assertEqual(summary["structural_artifacts_mode"], "library")
            self.assertGreaterEqual(summary["structural_library_hits"], 1)
            self.assertEqual(summary["structural_builds"], 0)
            self.assertIn("hits", summary)
            self.assertIn("misses", summary)


if __name__ == "__main__":
    unittest.main()
