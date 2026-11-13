"""
Phase L2 gate: shared structural artifact primitives and providers.
"""
from __future__ import annotations

import unittest
from functools import partial
from types import SimpleNamespace
from typing import Any

from core.analysis_core.run_config import analysis_run_config_from_params
from core.ioh_core.structural_artifacts import (
    StructuralArtifact,
    StructuralProviderError,
    StructuralReuseError,
    build_structural_provider,
    make_structural_key,
    scientific_values_equal,
)
from core.ioh_core.structural_artifacts.artifact import (
    apply_structural_artifact,
    capture_structural_artifact,
    get_structural_build_count,
    reset_structural_build_count,
)
from core.ioh_core.structural_artifacts.fixtures import (
    make_build_failure_artifact,
    make_success_artifact,
)
from core.ioh_core.structural_artifacts.provider import (
    LibraryStructuralProvider,
)
from core.ioh_core.workflows.brute_force.structural import (
    StructuralArtifact as CompatStructuralArtifact,
    evaluate_finish_group,
    finish_variable_names as bf_finish_variable_names,
)
from slab_construction.slab_construction import AnalysisCache

from _mains.Testing.test_cursor_brute_force_worker import _problem_fixtures
from core.ioh_core.workflows.brute_force.indexing import linear_to_indices
from slab_construction.slabs.hp_slab.analysis import analysis


class _DictLookup:
    def __init__(
        self,
        mapping: dict[int, StructuralArtifact],
        *,
        identity: Any | None = None,
    ) -> None:
        self._mapping = mapping
        self.identity = identity

    def get(self, structural_id: int) -> StructuralArtifact:
        try:
            return self._mapping[int(structural_id)]
        except KeyError as exc:
            raise StructuralProviderError(
                f"missing structural_id {structural_id}"
            ) from exc


class TestCompatibilityReexports(unittest.TestCase):
    def test_artifact_class_is_shared(self) -> None:
        self.assertIs(StructuralArtifact, CompatStructuralArtifact)

    def test_finish_names_sequence_helper(self) -> None:
        self.assertEqual(
            bf_finish_variable_names(("a", "b", "c", "d")),
            ("c", "d"),
        )
        with self.assertRaises(StructuralReuseError):
            bf_finish_variable_names(("a",))


class TestComputeProvider(unittest.TestCase):
    def test_compute_provider_calls_analysis_and_counts(self) -> None:
        calls: list[dict[str, Any]] = []

        def analysis_fn(params, **kwargs):
            calls.append({"params": dict(params), "kwargs": dict(kwargs)})
            return {"y_p": 1.0, "params": dict(params)}

        provider = build_structural_provider("compute")
        self.assertEqual(provider.mode, "compute")
        out = provider.evaluate(
            {"x": 1},
            analysis_fn=analysis_fn,
            structural_id=99,
        )
        self.assertEqual(out["y_p"], 1.0)
        self.assertEqual(provider.metrics.compute_evaluations, 1)
        self.assertEqual(provider.metrics.library_hits, 0)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("structural_artifact", calls[0]["kwargs"])


class TestLibraryProvider(unittest.TestCase):
    def test_library_requires_lookup_and_id(self) -> None:
        with self.assertRaises(StructuralProviderError):
            build_structural_provider("library")
        provider = LibraryStructuralProvider(lookup=_DictLookup({0: make_success_artifact()}))
        with self.assertRaises(StructuralProviderError):
            provider.evaluate({}, analysis_fn=lambda *a, **k: {})

    def test_library_hit_passes_artifact_without_evalcontext_semantics(self) -> None:
        artifact = make_success_artifact(seed=8)
        # Align key/snapshot with what the provider will derive.
        params = {"loads_N_kN": 0.0, "t": 8}
        run_config = analysis_run_config_from_params(
            {"opt_invalid_utilization": 999.0, **params}
        )
        # Rebuild artifact with a key matching make_structural_key for these params.
        key = make_structural_key(
            params,
            finish_names=(),
            snapshot_hash="snap-lib",
            n=0.0,
            run_config=run_config,
            structural_names=("t",),
        )
        artifact = StructuralArtifact(
            key=key,
            snapshot_hash="snap-lib",
            n=0.0,
            m_u=artifact.m_u,
            m_cr=artifact.m_cr,
            mk_curves=artifact.mk_curves,
            mk_metadata=artifact.mk_metadata,
            mk_cleared=False,
            build_failure_result=None,
        )
        seen: dict[str, Any] = {}

        def analysis_fn(p, **kwargs):
            seen["kwargs"] = dict(kwargs)
            return {"y_p": 2.0, "params": dict(p)}

        provider = LibraryStructuralProvider(
            lookup=_DictLookup({3: artifact}),
            structural_names=("t",),
        )
        out = provider.evaluate(
            params,
            analysis_fn=analysis_fn,
            structural_id=3,
            snapshot_hash="snap-lib",
            finish_names=(),
            run_config=run_config,
        )
        self.assertEqual(out["y_p"], 2.0)
        self.assertIs(seen["kwargs"]["structural_artifact"], artifact)
        self.assertEqual(provider.metrics.library_hits, 1)
        self.assertEqual(provider.metrics.compute_evaluations, 0)

    def test_library_build_failure_fanout(self) -> None:
        fail = make_build_failure_artifact(seed=9)
        params = {"loads_N_kN": 0.0, "t": 9}
        run_config = analysis_run_config_from_params(
            {"opt_invalid_utilization": 999.0, **params}
        )
        key = make_structural_key(
            params,
            finish_names=(),
            snapshot_hash=fail.snapshot_hash,
            n=0.0,
            run_config=run_config,
            structural_names=("t",),
        )
        fail = StructuralArtifact(
            key=key,
            snapshot_hash=fail.snapshot_hash,
            n=0.0,
            m_u=(),
            m_cr=(),
            mk_curves=(),
            mk_metadata=(),
            mk_cleared=True,
            build_failure_result=dict(fail.build_failure_result or {}),
        )
        provider = LibraryStructuralProvider(
            lookup=_DictLookup({1: fail}),
            structural_names=("t",),
        )
        out = provider.evaluate(
            params,
            analysis_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no analysis")),
            structural_id=1,
            snapshot_hash=fail.snapshot_hash,
            finish_names=(),
            run_config=run_config,
        )
        self.assertFalse(out["analysis_ok"])
        self.assertEqual(out["params"], params)
        self.assertEqual(provider.metrics.library_build_failure_fanouts, 1)

    def test_snapshot_hash_mismatch_requires_explicit_override(self) -> None:
        params = {"loads_N_kN": 0.0, "t": 8}
        run_config = analysis_run_config_from_params(
            {"opt_invalid_utilization": 999.0, **params}
        )
        library_hash = "snap-lib"
        experiment_hash = "snap-experiment"
        key = make_structural_key(
            params,
            finish_names=(),
            snapshot_hash=library_hash,
            n=0.0,
            run_config=run_config,
            structural_names=("t",),
        )
        base = make_success_artifact(seed=8)
        artifact = StructuralArtifact(
            key=key,
            snapshot_hash=library_hash,
            n=0.0,
            m_u=base.m_u,
            m_cr=base.m_cr,
            mk_curves=base.mk_curves,
            mk_metadata=base.mk_metadata,
            mk_cleared=False,
            build_failure_result=None,
        )
        identity = SimpleNamespace(snapshot_hash=library_hash, structural_names=("t",))
        lookup = _DictLookup({3: artifact}, identity=identity)

        strict = LibraryStructuralProvider(
            lookup=lookup,
            structural_names=("t",),
        )
        with self.assertRaisesRegex(StructuralProviderError, "snapshot_hash"):
            strict.evaluate(
                params,
                analysis_fn=lambda *a, **k: {"y_p": 1.0},
                structural_id=3,
                snapshot_hash=experiment_hash,
                finish_names=(),
                run_config=run_config,
            )

        forced = LibraryStructuralProvider(
            lookup=lookup,
            structural_names=("t",),
            allow_snapshot_hash_mismatch=True,
        )
        out = forced.evaluate(
            params,
            analysis_fn=lambda *a, **k: {"y_p": 2.0},
            structural_id=3,
            snapshot_hash=experiment_hash,
            finish_names=(),
            run_config=run_config,
        )
        self.assertEqual(out["y_p"], 2.0)
        self.assertEqual(forced.metrics.library_hits, 1)

        via_factory = build_structural_provider(
            "library",
            lookup=lookup,
            allow_snapshot_hash_mismatch=True,
        )
        out2 = via_factory.evaluate(
            params,
            analysis_fn=lambda *a, **k: {"y_p": 3.0},
            structural_id=3,
            snapshot_hash=experiment_hash,
            finish_names=(),
            run_config=run_config,
        )
        self.assertEqual(out2["y_p"], 3.0)
        self.assertEqual(via_factory.metrics.library_hits, 1)


class TestCaptureApplyRoundTrip(unittest.TestCase):
    def test_capture_apply_preserves_cache_payload(self) -> None:
        artifact = make_success_artifact(seed=12)
        cache = AnalysisCache()
        apply_structural_artifact(cache, artifact)
        self.assertEqual(len(cache.m_u), 1)
        self.assertEqual(len(cache.mk_curves), 1)
        restored = capture_structural_artifact(
            cache,
            key=artifact.key,
            snapshot_hash=artifact.snapshot_hash,
            n=artifact.n,
            mk_cleared=False,
            count_build=False,
        )
        # Nested mapping freeze/thaw is dict↔tuple-of-pairs; compare via apply.
        cache2 = AnalysisCache()
        apply_structural_artifact(cache2, restored)
        left = next(iter(cache.m_u.values()))
        right = next(iter(cache2.m_u.values()))
        self.assertTrue(scientific_values_equal(left, right))
        mk_key = next(iter(cache.mk_curves))
        self.assertTrue(
            scientific_values_equal(
                list(cache.mk_curves[mk_key][0.0].m_y),
                list(cache2.mk_curves[mk_key][0.0].m_y),
            )
        )


class TestFinishGroupStillUsesSharedPrimitives(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _info, space, decode, materials, constraints = _problem_fixtures()
        cls.space = space
        cls.decode = staticmethod(decode)
        cls.finish_names = bf_finish_variable_names(space.var_names)
        cls.group = 24
        params0 = decode(list(linear_to_indices(0, space)))
        cls.run_config = analysis_run_config_from_params(params0)
        cls.analysis_fn = partial(
            analysis,
            constraints=constraints,
            materials=materials,
            run_config=cls.run_config,
        )

    def test_one_build_per_group(self) -> None:
        group_params = [
            self.decode(list(linear_to_indices(i, self.space)))
            for i in range(self.group)
        ]
        reset_structural_build_count()
        results = evaluate_finish_group(
            group_params,
            constraints={},
            materials={},
            analysis_fn=self.analysis_fn,
            finish_names=self.finish_names,
            snapshot_hash="l2-compat",
            run_config=self.run_config,
            reuse=True,
        )
        self.assertEqual(len(results), self.group)
        self.assertEqual(get_structural_build_count(), 1)


if __name__ == "__main__":
    unittest.main()
