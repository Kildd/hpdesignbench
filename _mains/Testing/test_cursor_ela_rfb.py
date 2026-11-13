"""
ELA RFB× walk configuration and discrete L1 walks.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from core.ioh_core.workflows.brute_force.indexing import linear_to_indices
from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.la.models import (
    ElaError,
    FeatureResult,
    LANDSCAPE_FEASIBILITY,
    RESPONSE_VIEW_NA,
    ROLE_DIAGNOSTIC,
    ROLE_PRIMARY,
    SOURCE_BRUTE_FORCE,
    X_VIEW_INDEX,
)
from core.ioh_core.workflows.la.rfb import (
    FAMILY,
    FEATURE_N_CROSSINGS,
    FEATURE_RFB_X,
    START_POLICY_UNIFORM_CATALOGUE,
    RfbWalkConfig,
    derive_rfb_seed,
    ela_rfb_rows,
    ela_rfb_walks,
)
from core.ioh_core.workflows.la.topology import FeasibleRegion

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestRfbWalkConfig(unittest.TestCase):
    """Caller-supplied knobs; no library study defaults."""

    def test_valid_config_and_family_token(self) -> None:
        config = RfbWalkConfig(
            n_walks=3,
            n_transitions=10,
            seed_base=11,
            start_policy=START_POLICY_UNIFORM_CATALOGUE,
        )
        self.assertEqual(FAMILY, "rfbx")
        self.assertEqual(config.n_walks, 3)
        self.assertEqual(config.n_transitions, 10)
        self.assertEqual(config.seed_base, 11)
        self.assertEqual(config.start_policy, "uniform_catalogue")

    def test_rejects_non_positive_counts(self) -> None:
        kwargs = dict(
            n_walks=1,
            n_transitions=1,
            seed_base=0,
            start_policy=START_POLICY_UNIFORM_CATALOGUE,
        )
        with self.assertRaises(ElaError):
            RfbWalkConfig(**{**kwargs, "n_walks": 0})
        with self.assertRaises(ElaError):
            RfbWalkConfig(**{**kwargs, "n_transitions": 0})
        with self.assertRaises(ElaError):
            RfbWalkConfig(**{**kwargs, "n_walks": True})
        with self.assertRaises(ElaError):
            RfbWalkConfig(**{**kwargs, "n_transitions": -1})

    def test_rejects_bad_seed_base_and_policy(self) -> None:
        kwargs = dict(
            n_walks=1,
            n_transitions=1,
            seed_base=0,
            start_policy=START_POLICY_UNIFORM_CATALOGUE,
        )
        with self.assertRaises(ElaError):
            RfbWalkConfig(**{**kwargs, "seed_base": -1})
        with self.assertRaises(ElaError):
            RfbWalkConfig(**{**kwargs, "seed_base": True})
        with self.assertRaises(ElaError) as ctx:
            RfbWalkConfig(**{**kwargs, "start_policy": "uniform_feasible"})
        self.assertIn("uniform_catalogue", str(ctx.exception))


class TestDeriveRfbSeed(unittest.TestCase):
    """SHA-256 uint32 mix, local copy of the sampling pattern."""

    def test_matches_sha256_u32_pattern(self) -> None:
        seed = derive_rfb_seed(seed_base=11, problem_id="L6750_B2", walk_id=0)
        material = "11|ela|rfbx|L6750_B2|0"
        expected = int.from_bytes(
            hashlib.sha256(material.encode("utf-8")).digest()[:4],
            byteorder="big",
            signed=False,
        )
        self.assertEqual(seed, expected)
        self.assertGreaterEqual(seed, 0)
        self.assertLessEqual(seed, 0xFFFFFFFF)

    def test_walk_id_changes_seed(self) -> None:
        first = derive_rfb_seed(seed_base=11, problem_id="L6750_B2", walk_id=0)
        second = derive_rfb_seed(seed_base=11, problem_id="L6750_B2", walk_id=1)
        self.assertNotEqual(first, second)

    def test_config_seed_for_rejects_out_of_range_walk_id(self) -> None:
        config = RfbWalkConfig(
            n_walks=2,
            n_transitions=4,
            seed_base=11,
            start_policy=START_POLICY_UNIFORM_CATALOGUE,
        )
        self.assertEqual(
            config.seed_for(problem_id="L6750_B2", walk_id=0),
            derive_rfb_seed(seed_base=11, problem_id="L6750_B2", walk_id=0),
        )
        with self.assertRaises(ElaError):
            config.seed_for(problem_id="L6750_B2", walk_id=2)

    def test_importing_module_does_not_load_forbidden_libraries(self) -> None:
        script = """
import sys
import core.ioh_core.workflows.la
import core.ioh_core.workflows.la.rfb
loaded = [
    name
    for name in ("duckdb", "pflacco", "networkx", "SALib")
    if name in sys.modules
]
if loaded:
    raise SystemExit("loaded forbidden modules: " + ", ".join(loaded))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stderr or completed.stdout,
        )


def _config(**overrides: object) -> RfbWalkConfig:
    payload: dict[str, object] = {
        "n_walks": 1,
        "n_transitions": 8,
        "seed_base": 11,
        "start_policy": START_POLICY_UNIFORM_CATALOGUE,
    }
    payload.update(overrides)
    return RfbWalkConfig(**payload)  # type: ignore[arg-type]


def _region_from_mask(
    mask: np.ndarray,
    *,
    var_names: tuple[str, ...] = ("geom_t_mm", "mat_conc_id"),
    lower: tuple[int, ...] = (0, 3),
    upper: tuple[int, ...] = (1, 4),
) -> FeasibleRegion:
    space = IndexSpace.from_bounds(var_names, lower, upper)
    flags = np.asarray(mask, dtype=bool)
    flags.setflags(write=False)
    return FeasibleRegion(
        space=space,
        mask=flags,
        n_feasible=int(flags.sum()),
        n_input=space.cardinality,
        n_unique=space.cardinality,
        constraint_names=("A_bending_capacity",),
    )


class TestElaRfbWalks(unittest.TestCase):
    """Uniform catalogue starts, L1 steps, mask lookup, RFB× denominator."""

    def test_same_config_reproduces_paths(self) -> None:
        region = _region_from_mask(np.array([True, False, False, True]))
        config = _config(n_walks=3, n_transitions=12)
        first = ela_rfb_walks(region=region, config=config, problem_id="L6750_B2")
        second = ela_rfb_walks(region=region, config=config, problem_id="L6750_B2")
        self.assertEqual(len(first), 3)
        for left, right in zip(first, second, strict=True):
            self.assertEqual(left.walk_id, right.walk_id)
            self.assertEqual(left.seed, right.seed)
            self.assertEqual(left.start_linear, right.start_linear)
            self.assertEqual(left.start_index, right.start_index)
            self.assertEqual(left.n_crossings, right.n_crossings)
            self.assertEqual(left.rfb_x, right.rfb_x)
            np.testing.assert_array_equal(left.linear_path, right.linear_path)
            self.assertEqual(tuple(left.linear_path.shape), (13,))
            self.assertFalse(left.linear_path.flags.writeable)

    def test_two_cell_line_every_step_crosses(self) -> None:
        region = _region_from_mask(
            np.array([True, False]),
            var_names=("geom_t_mm",),
            lower=(0,),
            upper=(1,),
        )
        walks = ela_rfb_walks(
            region=region,
            config=_config(n_walks=2, n_transitions=5),
            problem_id="L6750_B2",
        )
        for walk in walks:
            self.assertEqual(walk.n_crossings, 5)
            self.assertEqual(walk.rfb_x, 1.0)
            path = [int(v) for v in walk.linear_path]
            self.assertEqual(path[0], walk.start_linear)
            for left, right in zip(path[:-1], path[1:], strict=True):
                self.assertEqual({left, right}, {0, 1})
                self.assertNotEqual(left, right)

    def test_full_space_f_has_zero_rfb(self) -> None:
        region = _region_from_mask(np.ones(4, dtype=bool))
        walk = ela_rfb_walks(
            region=region,
            config=_config(n_walks=1, n_transitions=7),
            problem_id="L6750_B2",
        )[0]
        self.assertEqual(walk.n_crossings, 0)
        self.assertEqual(walk.rfb_x, 0.0)
        self.assertEqual(walk.rfb_x, walk.n_crossings / 7)

    def test_empty_f_allows_infeasible_starts(self) -> None:
        region = _region_from_mask(np.zeros(4, dtype=bool))
        walk = ela_rfb_walks(
            region=region,
            config=_config(n_walks=1, n_transitions=4),
            problem_id="L6750_B2",
        )[0]
        self.assertGreaterEqual(walk.start_linear, 0)
        self.assertLess(walk.start_linear, 4)
        self.assertFalse(region.mask[walk.start_linear])
        self.assertEqual(walk.n_crossings, 0)
        self.assertEqual(walk.rfb_x, 0.0)

    def test_no_wrap_at_stop7_corner(self) -> None:
        region = _region_from_mask(np.array([True, False, False, True]))
        space = region.space
        walks = ela_rfb_walks(
            region=region,
            config=_config(n_walks=4, n_transitions=16),
            problem_id="L6750_B2",
        )
        for walk in walks:
            for linear in walk.linear_path:
                point = linear_to_indices(int(linear), space)
                for value, lower, upper in zip(
                    point, space.lower_bounds, space.upper_bounds, strict=True
                ):
                    self.assertGreaterEqual(value, lower)
                    self.assertLessEqual(value, upper)

    def test_one_cell_catalogue_cannot_step(self) -> None:
        region = _region_from_mask(
            np.array([True]),
            var_names=("geom_t_mm",),
            lower=(0,),
            upper=(0,),
        )
        with self.assertRaises(ElaError) as ctx:
            ela_rfb_walks(
                region=region,
                config=_config(n_walks=1, n_transitions=1),
                problem_id="L6750_B2",
            )
        self.assertIn("L1 neighbour", str(ctx.exception))


class TestElaRfbStop7Pipeline(unittest.TestCase):
    """16.3: L1=1 steps, no wrap, denominator, reproducibility on STOP 7."""

    def test_stop7_steps_are_l1_and_rfb_matches_path(self) -> None:
        region = _region_from_mask(np.array([True, False, False, True]))
        space = region.space
        config = _config(n_walks=4, n_transitions=16)
        first = ela_rfb_walks(region=region, config=config, problem_id="L6750_B2")
        second = ela_rfb_walks(region=region, config=config, problem_id="L6750_B2")
        self.assertEqual(len(first), 4)
        for walk, replay in zip(first, second, strict=True):
            np.testing.assert_array_equal(walk.linear_path, replay.linear_path)
            self.assertEqual(walk.rfb_x, walk.n_crossings / config.n_transitions)
            self.assertEqual(tuple(walk.linear_path.shape), (config.n_transitions + 1,))
            points = [
                linear_to_indices(int(linear), space) for linear in walk.linear_path
            ]
            n_crossings = 0
            prev_feasible = bool(region.mask[int(walk.linear_path[0])])
            for prev, curr, linear in zip(
                points[:-1], points[1:], walk.linear_path[1:], strict=True
            ):
                delta = tuple(abs(a - b) for a, b in zip(prev, curr, strict=True))
                self.assertEqual(sum(delta), 1)
                self.assertNotIn(-1, curr)
                self.assertNotIn(2, curr)
                self.assertNotIn(5, curr)
                for value, lower, upper in zip(
                    curr, space.lower_bounds, space.upper_bounds, strict=True
                ):
                    self.assertGreaterEqual(value, lower)
                    self.assertLessEqual(value, upper)
                feasible = bool(region.mask[int(linear)])
                if feasible != prev_feasible:
                    n_crossings += 1
                prev_feasible = feasible
            self.assertEqual(walk.n_crossings, n_crossings)
            self.assertEqual(walk.rfb_x, n_crossings / config.n_transitions)


class TestElaRfbFeatureRows(unittest.TestCase):
    """Walk-level tidy rows; no path dump, no extra quantile rows."""

    def test_emits_rfb_x_and_n_crossings_per_walk(self) -> None:
        region = _region_from_mask(np.array([True, False, False, True]))
        config = _config(n_walks=2, n_transitions=4)
        walks = ela_rfb_walks(region=region, config=config, problem_id="L6750_B2")
        rows = ela_rfb_rows(
            walks=walks,
            config=config,
            problem_id="L6750_B2",
            source_dir=Path("."),
            constraint_names=region.constraint_names,
            n_input=region.n_input,
            n_unique=region.n_unique,
            runtime_s=0.0,
        )
        self.assertEqual(len(rows), 4)
        names = {row.feature_name for row in rows}
        self.assertEqual(names, {FEATURE_RFB_X, FEATURE_N_CROSSINGS})
        self.assertFalse(any("linear" in row.feature_name for row in rows))
        self.assertFalse(any("start" in row.feature_name for row in rows))
        self.assertFalse(any("min" in row.feature_name for row in rows))
        by_walk: dict[int, dict[str, FeatureResult]] = {}
        for row in rows:
            self.assertEqual(row.family, FAMILY)
            self.assertEqual(row.source_type, SOURCE_BRUTE_FORCE)
            self.assertIsNone(row.repetition)
            self.assertEqual(row.landscape, LANDSCAPE_FEASIBILITY)
            self.assertEqual(row.x_view, X_VIEW_INDEX)
            self.assertEqual(row.response_view, RESPONSE_VIEW_NA)
            self.assertIsNone(row.sampling_design_signature)
            self.assertIsNotNone(row.walk_id)
            walk_id = int(row.walk_id)
            expected_seed = derive_rfb_seed(
                seed_base=config.seed_base, problem_id="L6750_B2", walk_id=walk_id
            )
            self.assertEqual(row.seed, expected_seed)
            self.assertNotEqual(row.seed, config.seed_base)
            by_walk.setdefault(walk_id, {})
            by_walk[walk_id][row.feature_name] = row
        self.assertEqual(set(by_walk), {0, 1})
        for walk in walks:
            primary = by_walk[walk.walk_id][FEATURE_RFB_X]
            diagnostic = by_walk[walk.walk_id][FEATURE_N_CROSSINGS]
            self.assertEqual(primary.role, ROLE_PRIMARY)
            self.assertEqual(diagnostic.role, ROLE_DIAGNOSTIC)
            self.assertEqual(primary.value, walk.rfb_x)
            self.assertEqual(diagnostic.value, float(walk.n_crossings))
            self.assertEqual(
                primary.config_payload,
                {
                    "n_transitions": 4,
                    "n_walks": 2,
                    "seed_base": 11,
                    "start_policy": START_POLICY_UNIFORM_CATALOGUE,
                },
            )


if __name__ == "__main__":
    unittest.main()
