"""
Catalogue local-search basins: nested best-exit, co-reach, SUCCESS scalars.
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.la.basins import (
    FAMILY,
    FEATURE_FITNESS_BASIN_SIZE_SPEARMAN,
    FEATURE_LARGEST_SHARE,
    FEATURE_N_BASINS,
    FEATURE_N_BASINS_AT_Y_P_STAR,
    FEATURE_N_CORES,
    FEATURE_N_SINGLETON_CORES,
    FEATURE_SHARE_AT_Y_P_STAR,
    FEATURE_Y_P_STAR,
    attractor_records,
    ela_basin_partition,
    ela_basins,
)
from core.ioh_core.workflows.la.catalogue_l1 import catalogue_l1_neighbours
from core.ioh_core.workflows.la.models import ElaError, SOURCE_SAMPLING

_KEEP = ("A_bending_capacity",)


def _by_name(rows) -> dict[str, object]:
    return {row.feature_name: row for row in rows}


class TestCatalogueL1Neighbours(unittest.TestCase):
    def test_no_wrap_no_diagonal(self) -> None:
        space = IndexSpace.from_bounds(("x", "y"), (0, 3), (1, 4))
        neighbours = catalogue_l1_neighbours((0, 3), space)
        self.assertEqual(set(neighbours), {(1, 3), (0, 4)})
        self.assertNotIn((1, 4), neighbours)
        self.assertNotIn((-1, 3), neighbours)

    def test_stride_neighbours_match_helper(self) -> None:
        space = IndexSpace.from_bounds(("r", "c"), (0, 0), (2, 2))
        from core.ioh_core.workflows.brute_force.indexing import (
            indices_to_linear,
            linear_to_indices,
        )

        for linear in range(int(space.cardinality)):
            indices = linear_to_indices(linear, space)
            from_helper = set(catalogue_l1_neighbours(indices, space))
            from_stride = set()
            for stride, size, lower in zip(
                space.strides, space.sizes, space.lower_bounds, strict=True
            ):
                coord = (linear // int(stride)) % int(size)
                if coord > 0:
                    from_stride.add(
                        linear_to_indices(linear - int(stride), space)
                    )
                if coord + 1 < int(size):
                    from_stride.add(
                        linear_to_indices(linear + int(stride), space)
                    )
            self.assertEqual(from_helper, from_stride)
            self.assertEqual(
                indices_to_linear(indices, space),
                linear,
            )


class TestNestedBestExitBasins(unittest.TestCase):
    def test_family_token(self) -> None:
        self.assertEqual(FAMILY, "basins")

    def test_greedy_first_height_misses_deeper_core(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (5,))
        y_p = np.array([10.0, 8.0, 0.0, 7.0, 6.0, 5.0])
        rows = ela_basins(
            space=space,
            y_p=y_p,
            problem_id="L6750_B2",
            source_dir=Path("."),
            constraint_names=_KEEP,
        )
        by_name = _by_name(rows)
        self.assertEqual(by_name[FEATURE_N_BASINS].value, 2.0)
        self.assertEqual(by_name[FEATURE_N_CORES].value, 2.0)
        self.assertEqual(by_name[FEATURE_N_SINGLETON_CORES].value, 2.0)
        self.assertEqual(by_name[FEATURE_Y_P_STAR].value, 0.0)
        self.assertEqual(by_name[FEATURE_N_BASINS_AT_Y_P_STAR].value, 1.0)
        self.assertAlmostEqual(by_name[FEATURE_SHARE_AT_Y_P_STAR].value, 4.0 / 6.0)
        self.assertAlmostEqual(by_name[FEATURE_LARGEST_SHARE].value, 4.0 / 6.0)
        self.assertAlmostEqual(by_name[FEATURE_FITNESS_BASIN_SIZE_SPEARMAN].value, -1.0)
        self.assertEqual(by_name[FEATURE_N_BASINS].landscape, "y_p")
        self.assertEqual(by_name[FEATURE_N_BASINS].x_view, "index")
        self.assertEqual(by_name[FEATURE_N_BASINS].response_view, "raw")
        partition = ela_basin_partition(space=space, y_p=y_p)
        self.assertEqual(int(partition.basin_of_linear[0]), int(partition.basin_of_linear[2]))
        self.assertNotEqual(int(partition.basin_of_linear[2]), int(partition.basin_of_linear[5]))
        self.assertEqual(int(partition.basin_of_linear[4]), int(partition.basin_of_linear[5]))

    def test_co_reach_transitivity_merges_four_corners(self) -> None:
        space = IndexSpace.from_bounds(("r", "c"), (0, 0), (2, 2))
        y_p = np.array(
            [
                5.0, 9.0, 5.0,
                9.0, 8.0, 9.0,
                5.0, 9.0, 5.0,
            ]
        )
        rows = ela_basins(
            space=space,
            y_p=y_p,
            problem_id="L6750_B2",
            source_dir=Path("."),
            constraint_names=_KEEP,
        )
        by_name = _by_name(rows)
        self.assertEqual(by_name[FEATURE_N_CORES].value, 5.0)
        self.assertEqual(by_name[FEATURE_N_BASINS].value, 2.0)
        self.assertEqual(by_name[FEATURE_N_SINGLETON_CORES].value, 5.0)
        self.assertEqual(by_name[FEATURE_Y_P_STAR].value, 5.0)
        self.assertAlmostEqual(by_name[FEATURE_SHARE_AT_Y_P_STAR].value, 8.0 / 9.0)
        partition = ela_basin_partition(space=space, y_p=y_p)
        corners = [0, 2, 6, 8]
        self.assertEqual(len({int(partition.basin_of_linear[i]) for i in corners}), 1)
        self.assertNotEqual(
            int(partition.basin_of_linear[4]),
            int(partition.basin_of_linear[0]),
        )

    def test_ieee_plateau_is_one_core(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (2,))
        y_p = np.array([5.0, 5.0, 10.0])
        rows = ela_basins(
            space=space,
            y_p=y_p,
            problem_id="L6750_B2",
            source_dir=Path("."),
            constraint_names=_KEEP,
        )
        by_name = _by_name(rows)
        self.assertEqual(by_name[FEATURE_N_CORES].value, 1.0)
        self.assertEqual(by_name[FEATURE_N_BASINS].value, 1.0)
        self.assertEqual(by_name[FEATURE_N_SINGLETON_CORES].value, 0.0)
        self.assertIsNone(by_name[FEATURE_FITNESS_BASIN_SIZE_SPEARMAN].value)
        self.assertEqual(by_name[FEATURE_FITNESS_BASIN_SIZE_SPEARMAN].status, "undefined")

    def test_nonfinite_and_sampling_fail_closed(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (1,))
        with self.assertRaises(ElaError):
            ela_basin_partition(
                space=space, y_p=np.array([1.0, math.nan])
            )
        with self.assertRaises(ElaError):
            ela_basin_partition(
                space=space,
                y_p=np.array([1.0, 2.0]),
                source_type=SOURCE_SAMPLING,
            )

    def test_attractor_rows_share_basin_id(self) -> None:
        space = IndexSpace.from_bounds(("r", "c"), (0, 0), (1, 1))
        y_p = np.array([11.0, 21.0, 21.0, 11.0])
        feasible = np.array([True, False, False, True])
        partition = ela_basin_partition(space=space, y_p=y_p, feasible=feasible)
        records = attractor_records(partition)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["basin_id"], records[1]["basin_id"])
        self.assertEqual(partition.n_basins, 1)
        self.assertTrue(all(row["is_y_p_star"] for row in records))
        self.assertTrue(all(row["feasible"] for row in records))

    def test_mixed_frontier_does_not_merge_sibling_core(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (3,))
        y_p = np.array([3.0, 5.0, 3.0, 0.0])
        rows = ela_basins(
            space=space,
            y_p=y_p,
            problem_id="L6750_B2",
            source_dir=Path("."),
            constraint_names=_KEEP,
        )
        by_name = _by_name(rows)
        self.assertEqual(by_name[FEATURE_N_CORES].value, 2.0)
        self.assertEqual(by_name[FEATURE_N_BASINS].value, 2.0)
        self.assertEqual(by_name[FEATURE_N_BASINS_AT_Y_P_STAR].value, 1.0)
        self.assertAlmostEqual(by_name[FEATURE_SHARE_AT_Y_P_STAR].value, 3.0 / 4.0)
        partition = ela_basin_partition(space=space, y_p=y_p)
        self.assertNotEqual(
            int(partition.basin_of_linear[0]),
            int(partition.basin_of_linear[3]),
        )
        self.assertEqual(
            int(partition.basin_of_linear[1]),
            int(partition.basin_of_linear[3]),
        )

    def test_joint_frontier_takes_unique_best_next_height(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (4,))
        y_p = np.array([0.0, 4.0, 6.0, 4.0, 2.0])
        rows = ela_basins(
            space=space,
            y_p=y_p,
            problem_id="L6750_B2",
            source_dir=Path("."),
            constraint_names=_KEEP,
        )
        by_name = _by_name(rows)
        self.assertEqual(by_name[FEATURE_N_CORES].value, 2.0)
        self.assertEqual(by_name[FEATURE_N_BASINS].value, 2.0)
        self.assertAlmostEqual(by_name[FEATURE_SHARE_AT_Y_P_STAR].value, 3.0 / 5.0)
        partition = ela_basin_partition(space=space, y_p=y_p)
        self.assertEqual(int(partition.basin_of_linear[2]), int(partition.basin_of_linear[0]))
        self.assertNotEqual(int(partition.basin_of_linear[2]), int(partition.basin_of_linear[4]))
        self.assertEqual(int(partition.basin_of_linear[3]), int(partition.basin_of_linear[4]))


if __name__ == "__main__":
    unittest.main()
