"""
Phase L0 gate: structural mixed-radix IDs for artifact libraries.
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
from core.ioh_core.structural_artifacts import (
    StructuralIndexError,
    candidate_indices_to_structural_id,
    finish_group_size,
    finish_variable_names,
    global_index_to_structural_id,
    structural_cardinality,
    structural_id_to_structural_indices,
    structural_space,
    structural_variable_names,
)
from core.ioh_core.workflows.brute_force.indexing import (
    indices_to_linear,
    linear_to_indices,
)
from core.ioh_core.workflows.brute_force.models import IndexSpace

from _mains.Testing._paths import (
    CONSTRAINT_DEFAULTS_CSV,
    PARAM_DEFAULTS_CSV,
    PROBLEM_LIST_CSV,
)


def _toy_space() -> IndexSpace:
    """
    Structural dims a×b (2×3) and finish dims c×d (2×2) → group 4, struct 6.
    """
    return IndexSpace.from_bounds(
        ("a", "b", "c", "d"),
        (0, 0, 0, 0),
        (1, 2, 1, 1),
    )


class TestStructuralIndexingToy(unittest.TestCase):
    def test_names_sizes_and_cardinality(self) -> None:
        space = _toy_space()
        self.assertEqual(finish_variable_names(space), ("c", "d"))
        self.assertEqual(structural_variable_names(space), ("a", "b"))
        self.assertEqual(finish_group_size(space), 4)
        self.assertEqual(structural_cardinality(space), 6)
        struct = structural_space(space)
        self.assertEqual(struct.var_names, ("a", "b"))
        self.assertEqual(struct.cardinality, 6)

    def test_exhaustive_round_trips(self) -> None:
        space = _toy_space()
        group = finish_group_size(space)
        n_struct = structural_cardinality(space)
        seen: set[int] = set()
        for linear in range(space.cardinality):
            indices = linear_to_indices(linear, space)
            sid = candidate_indices_to_structural_id(indices, space)
            self.assertEqual(sid, global_index_to_structural_id(linear, space))
            self.assertEqual(sid, linear // group)
            self.assertEqual(
                structural_id_to_structural_indices(sid, space),
                indices[:2],
            )
            seen.add(sid)
        self.assertEqual(seen, set(range(n_struct)))

    def test_structural_id_round_trip(self) -> None:
        space = _toy_space()
        for sid in range(structural_cardinality(space)):
            struct_idx = structural_id_to_structural_indices(sid, space)
            # Attach a finish pair and recover the same structural id.
            for finish in itertools.product((0, 1), (0, 1)):
                full = struct_idx + finish
                self.assertEqual(
                    candidate_indices_to_structural_id(full, space),
                    sid,
                )
                self.assertEqual(
                    indices_to_linear(full, space) // finish_group_size(space),
                    sid,
                )

    def test_rejects_too_few_variables(self) -> None:
        space = IndexSpace.from_bounds(("a",), (0,), (2,))
        with self.assertRaises(StructuralIndexError):
            finish_variable_names(space)
        with self.assertRaises(StructuralIndexError):
            structural_variable_names(space)

    def test_rejects_finish_only_catalogue(self) -> None:
        space = IndexSpace.from_bounds(("c", "d"), (0, 0), (1, 1))
        with self.assertRaises(StructuralIndexError):
            structural_variable_names(space)

    def test_rejects_out_of_range_ids(self) -> None:
        space = _toy_space()
        with self.assertRaises(StructuralIndexError):
            structural_id_to_structural_indices(-1, space)
        with self.assertRaises(StructuralIndexError):
            structural_id_to_structural_indices(6, space)
        with self.assertRaises(StructuralIndexError):
            global_index_to_structural_id(space.cardinality, space)
        with self.assertRaises(StructuralIndexError):
            candidate_indices_to_structural_id((0, 0, 0), space)


class TestStructuralIndexingProblem1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
        cdef = load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV)
        problems = load_problems_combined(pdef, cdef, PROBLEM_LIST_CSV)
        info = problems["L6750_B2"]
        var_names, lb_idx, ub_idx = build_space(info["model"])
        cls.space = IndexSpace.from_bounds(var_names, lb_idx, ub_idx)

    def test_problem1_cardinalities(self) -> None:
        space = self.space
        self.assertEqual(len(space.var_names), 9)
        self.assertEqual(finish_group_size(space), 24)
        self.assertEqual(space.cardinality, 53_374_464)
        self.assertEqual(structural_cardinality(space), 2_223_936)
        self.assertEqual(
            space.cardinality,
            structural_cardinality(space) * finish_group_size(space),
        )
        self.assertEqual(
            finish_variable_names(space),
            (space.var_names[-2], space.var_names[-1]),
        )
        self.assertEqual(
            structural_variable_names(space),
            tuple(space.var_names[:-2]),
        )

    def test_problem1_boundary_ids(self) -> None:
        space = self.space
        group = finish_group_size(space)
        n_struct = structural_cardinality(space)
        cases = (
            0,
            group - 1,
            group,
            space.cardinality // 2,
            space.cardinality - 1,
        )
        for linear in cases:
            indices = linear_to_indices(linear, space)
            sid = candidate_indices_to_structural_id(indices, space)
            self.assertEqual(sid, linear // group)
            self.assertEqual(sid, global_index_to_structural_id(linear, space))
            self.assertEqual(
                structural_id_to_structural_indices(sid, space),
                indices[:-2],
            )
        self.assertEqual(
            structural_id_to_structural_indices(0, space),
            linear_to_indices(0, space)[:-2],
        )
        self.assertEqual(
            structural_id_to_structural_indices(n_struct - 1, space),
            linear_to_indices(space.cardinality - 1, space)[:-2],
        )


if __name__ == "__main__":
    unittest.main()
