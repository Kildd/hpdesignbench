"""Exhaustive set-valued LON: nodes, support, escape edges, and funnels."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.ioh_core.workflows.brute_force.indexing import indices_to_linear
from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.la.catalogue_l1 import catalogue_l1_neighbours
from core.ioh_core.workflows.la.catalogue_table import load_catalogue_landscape
from core.ioh_core.workflows.la.lon import (
    FAMILY,
    FEATURE_AMBIGUOUS_DESIGN_SHARE,
    FEATURE_BASIN_F_SHARE_MAX,
    FEATURE_BASIN_F_SHARE_MEDIAN,
    FEATURE_BASIN_F_SHARE_MIN,
    FEATURE_GLOBAL_SUPPORT_SHARE,
    FEATURE_N_NODES,
    FEATURE_SUPPORT_SIZE_MAX,
    FEATURE_Y_BOX_STAR,
    ela_lon,
    escape_edge_records,
    lon_node_records,
    overlap_records,
    set_valued_lon,
)
from core.ioh_core.workflows.la.models import (
    ElaError,
    ROLE_DIAGNOSTIC,
    SOURCE_SAMPLING,
)

_KEEP = ("A_bending_capacity",)


def _by_name(rows):
    return {row.feature_name: row for row in rows}


class TestSetValuedLon(unittest.TestCase):
    def test_family_token(self) -> None:
        self.assertEqual(FAMILY, "lon")

    def test_fetchall_loader_rejects_null_feasibility(self) -> None:
        class Result:
            def fetchall(self):
                return [(0, 1.0, None)]

        class Connection:
            def execute(self, _query):
                return Result()

        space = IndexSpace.from_bounds(("x",), (0,), (0,))
        with self.assertRaisesRegex(ElaError, "NULL"):
            load_catalogue_landscape(
                Connection(),
                "catalogue",
                space,
                expected_n=1,
                landscape="y_raw",
            )

    def test_connected_equal_local_minimum_is_one_lonn(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (4,))
        result = set_valued_lon(
            space=space,
            y_p=np.array([4.0, 2.0, 2.0, 2.0, 5.0]),
            feasible=np.array([False, True, False, True, False]),
            radii=(2,),
        )

        self.assertEqual(result.n_nodes, 1)
        self.assertEqual(result.node_size.tolist(), [3])
        self.assertEqual(result.node_n_feasible.tolist(), [2])
        self.assertEqual(result.node_any_feasible.tolist(), [True])
        self.assertEqual(result.node_all_feasible.tolist(), [False])
        self.assertTrue(all(result.support_for_linear(i) == (0,) for i in range(5)))
        self.assertEqual(result.basin_n_feasible.tolist(), [2])
        self.assertEqual(result.basin_f_share.tolist(), [0.4])

    def test_equal_best_branches_create_overlapping_basins(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (4,))
        result = set_valued_lon(
            space=space,
            y_p=np.array([0.0, 4.0, 6.0, 4.0, 2.0]),
            radii=(2,),
        )

        self.assertEqual(result.n_nodes, 2)
        self.assertEqual(result.support_for_linear(0), (0,))
        self.assertEqual(result.support_for_linear(2), (0, 1))
        self.assertEqual(result.support_for_linear(4), (1,))
        self.assertEqual(result.support_size.tolist(), [3, 3])
        self.assertEqual(result.ambiguous_design_count, 1)
        self.assertEqual(
            overlap_records(result),
            [{"node_a": 0, "node_b": 1, "design_count": 1}],
        )

    def test_overlapping_basin_f_share_counts_ambiguous_design_in_each_basin(
        self,
    ) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (4,))
        result = set_valued_lon(
            space=space,
            y_p=np.array([0.0, 4.0, 6.0, 4.0, 2.0]),
            feasible=np.array([True, False, True, False, False]),
            radii=(),
        )

        self.assertEqual(result.support_for_linear(2), (0, 1))
        self.assertEqual(result.support_size.tolist(), [3, 3])
        self.assertEqual(result.basin_n_feasible.tolist(), [2, 1])
        self.assertEqual(result.basin_f_share.tolist(), [2.0 / 3.0, 1.0 / 3.0])

    def test_disconnected_equal_minima_are_distinct_nodes(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (4,))
        result = set_valued_lon(
            space=space,
            y_p=np.array([2.0, 3.0, 4.0, 3.0, 2.0]),
            radii=(),
        )

        self.assertEqual(result.n_nodes, 2)
        self.assertEqual(result.node_y_p.tolist(), [2.0, 2.0])
        self.assertEqual(result.support_for_linear(2), (0, 1))

    def test_escape_shells_and_multiscale_funnels(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (4,))
        result = set_valued_lon(
            space=space,
            y_p=np.array([0.0, 4.0, 6.0, 4.0, 2.0]),
            radii=(1, 2),
        )
        records = escape_edge_records(result)

        self.assertIn(
            {
                "source_node_id": 0,
                "target_node_id": 1,
                "shell": 2,
                "support_count": 1,
                "source_shell_candidate_count": 1,
                "source_y_p": 0.0,
                "target_y_p": 2.0,
                "relation": "worse",
            },
            records,
        )
        source_zero_shell_two = [
            row
            for row in records
            if row["source_node_id"] == 0 and row["shell"] == 2
        ]
        self.assertEqual(
            {row["target_node_id"] for row in source_zero_shell_two},
            {0, 1},
        )
        self.assertTrue(
            all(
                row["support_count"] == 1
                and row["source_shell_candidate_count"] == 1
                for row in source_zero_shell_two
            )
        )
        d1 = result.funnels[1]
        d2 = result.funnels[2]
        self.assertEqual(d1.n_funnels, 2)
        self.assertEqual(d1.global_node_share, 0.5)
        self.assertEqual(d1.global_design_share, 3.0 / 5.0)
        self.assertEqual(d2.n_funnels, 1)
        self.assertEqual(d2.global_node_share, 1.0)
        self.assertEqual(d2.global_design_share, 1.0)
        d1_pairs = {
            (source, target)
            for source, target, shell in result.edge_support_count
            if shell <= 1
        }
        d2_pairs = {
            (source, target)
            for source, target, shell in result.edge_support_count
            if shell <= 2
        }
        self.assertTrue(d1_pairs.issubset(d2_pairs))

    def test_exact_shells_are_boundary_clipped_and_duplicate_free(self) -> None:
        space = IndexSpace.from_bounds(("r", "c"), (0, 0), (1, 1))
        result = set_valued_lon(
            space=space,
            y_p=np.array([0.0, 5.0, 5.0, 6.0]),
            radii=(1, 2),
        )

        self.assertEqual(result.n_nodes, 1)
        self.assertEqual(result.shell_candidate_count[(0, 1)], 2)
        self.assertEqual(result.shell_candidate_count[(0, 2)], 1)

    def test_multidesign_node_deduplicates_shared_shell_candidate(self) -> None:
        space = IndexSpace.from_bounds(("r", "c"), (0, 0), (1, 1))
        result = set_valued_lon(
            space=space,
            y_p=np.array([0.0, 0.0, 0.0, 5.0]),
            radii=(1,),
        )

        self.assertEqual(result.node_size.tolist(), [3])
        self.assertEqual(result.shell_candidate_count[(0, 1)], 1)
        self.assertEqual(result.edge_support_count[(0, 0, 1)], 1)

    def test_equal_height_escape_cycle_is_one_sink_component(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (2,))
        result = set_valued_lon(
            space=space,
            y_p=np.array([0.0, 2.0, 0.0]),
            radii=(1,),
        )

        self.assertEqual(result.n_nodes, 2)
        self.assertEqual(result.funnels[1].n_funnels, 1)
        self.assertEqual(result.funnels[1].n_global_sinks, 1)

    def test_node_can_support_two_funnels(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (6,))
        result = set_valued_lon(
            space=space,
            y_p=np.array([0.0, 4.0, 5.0, 3.0, 5.0, 4.0, 1.0]),
            radii=(2,),
        )

        self.assertEqual(result.n_nodes, 3)
        funnel = result.funnels[2]
        self.assertEqual(funnel.n_funnels, 2)
        self.assertEqual(funnel.ambiguous_node_share, 1.0 / 3.0)
        middle_node = int(
            result.graph.plateau_to_node[result.graph.plateau_of_linear[3]]
        )
        self.assertEqual(len(funnel.node_sink_support[middle_node]), 2)

    def test_seeded_stochastic_outcomes_stay_inside_exact_support(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (4,))
        values = np.array([0.0, 4.0, 6.0, 4.0, 2.0])
        result = set_valued_lon(space=space, y_p=values, radii=())
        expected = set(result.support_for_linear(2))
        observed: set[int] = set()
        for seed in range(100):
            rng = np.random.default_rng(seed)
            current = 2
            while True:
                current_tuple = (current,)
                neighbours = catalogue_l1_neighbours(current_tuple, space)
                improving = [
                    indices_to_linear(candidate, space)
                    for candidate in neighbours
                    if values[indices_to_linear(candidate, space)] < values[current]
                ]
                if not improving:
                    break
                best = min(float(values[index]) for index in improving)
                tied = [index for index in improving if float(values[index]) == best]
                current = int(rng.choice(tied))
            node_id = int(
                result.graph.plateau_to_node[
                    result.graph.plateau_of_linear[current]
                ]
            )
            observed.add(node_id)
        self.assertEqual(observed, expected)

    def test_records_and_feature_rows_use_locked_names(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (4,))
        rows, result = ela_lon(
            space=space,
            y_p=np.array([0.0, 4.0, 6.0, 4.0, 2.0]),
            feasible=np.array([True, False, False, False, True]),
            radii=(2,),
            problem_id="L6750_B2",
            source_dir=Path("."),
            constraint_names=_KEEP,
        )
        by_name = _by_name(rows)

        self.assertEqual(by_name[FEATURE_N_NODES].value, 2.0)
        self.assertEqual(by_name[FEATURE_SUPPORT_SIZE_MAX].value, 3.0)
        self.assertEqual(by_name[FEATURE_AMBIGUOUS_DESIGN_SHARE].value, 0.2)
        self.assertEqual(by_name[FEATURE_GLOBAL_SUPPORT_SHARE].value, 0.6)
        self.assertEqual(by_name[FEATURE_BASIN_F_SHARE_MIN].value, 1.0 / 3.0)
        self.assertEqual(by_name[FEATURE_BASIN_F_SHARE_MEDIAN].value, 1.0 / 3.0)
        self.assertEqual(by_name[FEATURE_BASIN_F_SHARE_MAX].value, 1.0 / 3.0)
        self.assertEqual(by_name[FEATURE_BASIN_F_SHARE_MIN].role, "primary")
        self.assertEqual(by_name[FEATURE_N_NODES].family, "lon")
        self.assertEqual(by_name[FEATURE_N_NODES].landscape, "y_p")
        self.assertEqual(
            by_name["lon.funnel.d2.largest_nonglobal_node_share"].status,
            "undefined",
        )
        nodes = lon_node_records(result)
        self.assertEqual(nodes[0]["n_feasible"], 1)
        self.assertEqual(nodes[0]["all_feasible"], True)
        self.assertEqual(nodes[1]["n_feasible"], 1)
        self.assertEqual(nodes[0]["basin_n_feasible"], 1)
        self.assertEqual(nodes[0]["basin_f_share"], 1.0 / 3.0)
        self.assertEqual(nodes[1]["basin_n_feasible"], 1)
        self.assertEqual(nodes[1]["basin_f_share"], 1.0 / 3.0)

    def test_y_raw_uses_landscape_specific_identity_and_sidecars(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (4,))
        rows, result = ela_lon(
            space=space,
            values=np.array([5.0, 3.0, 4.0, 2.0, 6.0]),
            landscape="y_raw",
            feasible=np.array([True, True, False, False, False]),
            radii=(2,),
            problem_id="L6750_B2",
            source_dir=Path("."),
            constraint_names=_KEEP,
        )
        by_name = _by_name(rows)

        self.assertEqual(result.landscape, "y_raw")
        self.assertEqual(result.global_value, 2.0)
        self.assertEqual(by_name[FEATURE_N_NODES].landscape, "y_raw")
        self.assertEqual(by_name[FEATURE_Y_BOX_STAR].value, 2.0)
        self.assertNotIn("lon.y_p_star", by_name)
        nodes = lon_node_records(result)
        self.assertIn("y_raw", nodes[0])
        self.assertIn("is_y_box_star", nodes[0])
        self.assertNotIn("y_p", nodes[0])
        edges = escape_edge_records(result)
        self.assertTrue(edges)
        self.assertIn("source_y_raw", edges[0])
        self.assertIn("target_y_raw", edges[0])
        self.assertNotIn("source_y_p", edges[0])

    def test_landscapes_share_exactly_68_scientific_feature_definitions(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (4,))
        values = np.array([0.0, 4.0, 6.0, 4.0, 2.0])
        common = {
            "space": space,
            "feasible": np.array([True, False, False, False, True]),
            "radii": (2, 3, 4),
            "problem_id": "L6750_B2",
            "source_dir": Path("."),
            "constraint_names": _KEEP,
        }
        y_p_rows, _ = ela_lon(y_p=values, **common)
        y_raw_rows, _ = ela_lon(
            values=values,
            landscape="y_raw",
            **common,
        )
        y_p_scientific = {
            row.feature_name: (row.value, row.status, row.role)
            for row in y_p_rows
            if row.role != ROLE_DIAGNOSTIC
        }
        y_raw_scientific = {
            row.feature_name: (row.value, row.status, row.role)
            for row in y_raw_rows
            if row.role != ROLE_DIAGNOSTIC
        }
        basin_keys = {
            FEATURE_BASIN_F_SHARE_MIN,
            FEATURE_BASIN_F_SHARE_MEDIAN,
            FEATURE_BASIN_F_SHARE_MAX,
        }
        y_p_legacy = {
            name: payload
            for name, payload in y_p_scientific.items()
            if name not in basin_keys
        }

        self.assertEqual(len(y_p_legacy), 68)
        self.assertEqual(len(y_p_scientific), 71)
        self.assertEqual(y_raw_scientific, y_p_scientific)
        for name in basin_keys:
            self.assertEqual(y_p_scientific[name][2], "primary")
        frozen_payload = [
            (row.feature_name, row.value, row.status, row.role)
            for row in y_p_rows
            if row.role != ROLE_DIAGNOSTIC and row.feature_name not in basin_keys
        ]
        digest = hashlib.sha256(
            json.dumps(
                frozen_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.assertEqual(
            digest,
            "6930b73f61a617f7ce11ebb23f30a62b06fec7adadda1c2bb5121436d37416f2",
        )

    def test_nonfinite_sampling_and_invalid_radii_fail_closed(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (1,))
        with self.assertRaises(ElaError):
            set_valued_lon(space=space, y_p=np.array([1.0, np.nan]), radii=())
        with self.assertRaises(ElaError):
            set_valued_lon(
                space=space,
                y_p=np.array([1.0, 2.0]),
                source_type=SOURCE_SAMPLING,
                radii=(),
            )
        with self.assertRaises(ElaError):
            set_valued_lon(space=space, y_p=np.array([1.0, 2.0]), radii=(0,))
        with self.assertRaises(ElaError):
            set_valued_lon(space=space, y_p=np.array([1.0, 2.0]), radii=(2, 2))
        with self.assertRaisesRegex(ElaError, "landscape"):
            set_valued_lon(
                space=space,
                values=np.array([1.0, 2.0]),
                landscape="y",
                radii=(),
            )

    def test_inconsistent_escape_count_fails_closed(self) -> None:
        space = IndexSpace.from_bounds(("x",), (0,), (0,))
        with patch(
            "core.ioh_core.workflows.la.lon._escape_edges",
            return_value=(
                {(0, 0, 1): 2},
                {(0, 1): 1},
                {(0, 1): 0},
                {},
                {},
            ),
        ):
            with self.assertRaisesRegex(ElaError, "exceeds"):
                set_valued_lon(
                    space=space,
                    y_p=np.array([0.0]),
                    radii=(1,),
                )
        with patch(
            "core.ioh_core.workflows.la.lon._escape_edges",
            return_value=({}, {(0, 1): 0}, {}, {}, {}),
        ):
            with self.assertRaisesRegex(ElaError, "incomplete"):
                set_valued_lon(
                    space=space,
                    y_p=np.array([0.0]),
                    radii=(1,),
                )


if __name__ == "__main__":
    unittest.main()
