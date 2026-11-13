"""
Phase B2 gate: deterministic mixed-radix indexing for brute-force spaces.
"""
from __future__ import annotations

import itertools
import unittest

from core.ioh_core.import_specs import (
    build_space,
    load_constraint_defaults,
    load_param_defaults,
    load_problems_combined,
)
from core.ioh_core.workflows.brute_force.indexing import (
    indices_to_linear,
    iter_range,
    linear_to_indices,
)
from core.ioh_core.workflows.brute_force.models import IndexSpace, IndexSpaceError

from _mains.Testing._paths import (
    CONSTRAINT_DEFAULTS_CSV,
    PARAM_DEFAULTS_CSV,
    PROBLEM_LIST_CSV,
)


def _toy_space() -> IndexSpace:
    """
    Return a small 2 x 3 x 4 space with zero lower bounds.
    """
    return IndexSpace.from_bounds(
        ("a", "b", "c"),
        (0, 0, 0),
        (1, 2, 3),
    )


class TestIndexSpaceConstruction(unittest.TestCase):
    """Sizes, strides, cardinality, and validation."""

    def test_toy_sizes_strides_cardinality(self) -> None:
        space = _toy_space()
        self.assertEqual(space.sizes, (2, 3, 4))
        self.assertEqual(space.strides, (12, 4, 1))
        self.assertEqual(space.cardinality, 24)

    def test_nonzero_lower_bounds_preserve_sizes(self) -> None:
        space = IndexSpace.from_bounds(
            ("x", "y"),
            (2, 5),
            (4, 7),
        )
        self.assertEqual(space.sizes, (3, 3))
        self.assertEqual(space.strides, (3, 1))
        self.assertEqual(space.cardinality, 9)

    def test_rejects_empty_dimensions(self) -> None:
        with self.assertRaises(IndexSpaceError):
            IndexSpace.from_bounds((), (), ())

    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(IndexSpaceError):
            IndexSpace.from_bounds(("a", "b"), (0,), (1, 1))

    def test_rejects_reversed_bounds(self) -> None:
        with self.assertRaises(IndexSpaceError):
            IndexSpace.from_bounds(("a",), (3,), (1,))

    def test_rejects_duplicate_names(self) -> None:
        with self.assertRaises(IndexSpaceError):
            IndexSpace.from_bounds(("a", "a"), (0, 0), (1, 1))

    def test_rejects_signed_int64_overflow(self) -> None:
        huge = (1 << 62) - 1
        with self.assertRaises(IndexSpaceError):
            IndexSpace.from_bounds(("a", "b"), (0, 0), (huge, huge))


class TestMixedRadixIndexing(unittest.TestCase):
    """Product equivalence, round trips, and range iteration."""

    def test_exhaustive_product_equivalence(self) -> None:
        space = _toy_space()
        expected = list(
            itertools.product(
                range(0, 2),
                range(0, 3),
                range(0, 4),
            )
        )
        self.assertEqual(len(expected), space.cardinality)
        for linear, vector in enumerate(expected):
            self.assertEqual(linear_to_indices(linear, space), vector)
            self.assertEqual(indices_to_linear(vector, space), linear)
        self.assertEqual(list(iter_range(0, space.cardinality, space)), expected)

    def test_first_and_last_vectors(self) -> None:
        space = _toy_space()
        self.assertEqual(linear_to_indices(0, space), (0, 0, 0))
        self.assertEqual(
            linear_to_indices(space.cardinality - 1, space),
            (1, 2, 3),
        )

    def test_nonzero_lower_round_trip(self) -> None:
        space = IndexSpace.from_bounds(("x", "y"), (2, 5), (4, 7))
        expected = list(itertools.product(range(2, 5), range(5, 8)))
        self.assertEqual(list(iter_range(0, space.cardinality, space)), expected)
        for linear, vector in enumerate(expected):
            self.assertEqual(linear_to_indices(linear, space), vector)
            self.assertEqual(indices_to_linear(vector, space), linear)

    def test_iter_range_partial(self) -> None:
        space = _toy_space()
        self.assertEqual(
            list(iter_range(3, 7, space)),
            [linear_to_indices(i, space) for i in range(3, 7)],
        )

    def test_rejects_out_of_range_linear(self) -> None:
        space = _toy_space()
        with self.assertRaises(IndexSpaceError):
            linear_to_indices(-1, space)
        with self.assertRaises(IndexSpaceError):
            linear_to_indices(space.cardinality, space)

    def test_rejects_out_of_range_indices(self) -> None:
        space = _toy_space()
        with self.assertRaises(IndexSpaceError):
            indices_to_linear((0, 0), space)
        with self.assertRaises(IndexSpaceError):
            indices_to_linear((0, 0, 4), space)

    def test_rejects_invalid_iter_range(self) -> None:
        space = _toy_space()
        with self.assertRaises(IndexSpaceError):
            list(iter_range(-1, 1, space))
        with self.assertRaises(IndexSpaceError):
            list(iter_range(5, 3, space))
        with self.assertRaises(IndexSpaceError):
            list(iter_range(0, space.cardinality + 1, space))


class TestRealProblemCardinality(unittest.TestCase):
    """Problem 1 cardinality without enumerating the full domain."""

    def test_problem_1_cardinality_matches_build_space(self) -> None:
        pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
        cdef = load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV)
        problems = load_problems_combined(pdef, cdef, PROBLEM_LIST_CSV)
        var_names, lb_idx, ub_idx = build_space(problems["L6750_B2"]["model"])
        space = IndexSpace.from_bounds(var_names, lb_idx, ub_idx)

        expected = 1
        for lo, hi in zip(lb_idx, ub_idx):
            expected *= hi - lo + 1
        self.assertEqual(space.cardinality, expected)
        self.assertEqual(space.cardinality, 53_374_464)
        # Spot-check ends only — never fully enumerate.
        self.assertEqual(
            linear_to_indices(0, space),
            tuple(lb_idx),
        )
        self.assertEqual(
            linear_to_indices(space.cardinality - 1, space),
            tuple(ub_idx),
        )
        self.assertEqual(indices_to_linear(tuple(lb_idx), space), 0)
        self.assertEqual(
            indices_to_linear(tuple(ub_idx), space),
            space.cardinality - 1,
        )


if __name__ == "__main__":
    unittest.main()
