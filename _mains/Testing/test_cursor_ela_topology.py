"""
ELA feasible-region mask and L1=1 neighbours on raw X_index.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from core.ioh_core.workflows.brute_force.indexing import indices_to_linear
from core.ioh_core.workflows.brute_force.models import IndexSpace
from core.ioh_core.workflows.la.deps import import_duckdb
from core.ioh_core.workflows.la.models import ElaError
from core.ioh_core.workflows.la.preprocess import constraint_violations
from core.ioh_core.workflows.la.topology import (
    FAMILY,
    FEATURE_N_FEASIBLE,
    FEATURE_N_OPTIMA,
    FEATURE_Y_STAR,
    FeasibleRegion,
    ela_feasible_boundary,
    ela_feasible_components,
    ela_feasible_region,
    ela_global_optima,
    ela_near_optimal,
    ela_topology_features,
    n_g_feature_name,
    parse_ela_taus,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = Path("experiments/bf_p1")
_CONSTRAINTS = ("A_bending_capacity", "Z1_nt_dt_combination_capacity")
_VARS = ("geom_t_mm", "mat_conc_id")
_LOWER = (0, 3)
_UPPER = (1, 4)


def _complete_grid() -> tuple[np.ndarray, np.ndarray]:
    X_index = np.array([[0, 3], [0, 4], [1, 3], [1, 4]], dtype=np.int64)
    U = np.array(
        [
            [0.5, 0.5],
            [0.5, 1.2],
            [1.5, 0.5],
            [0.8, 0.8],
        ],
        dtype=float,
    )
    return X_index, U


def _load_table(
    X_index: np.ndarray,
    U: np.ndarray,
    *,
    extra_feasible: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> object:
    duckdb = import_duckdb()
    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE src ('
        '"idx__geom_t_mm" INTEGER, "idx__mat_conc_id" INTEGER, '
        '"util__A_bending_capacity" DOUBLE, '
        '"util__Z1_nt_dt_combination_capacity" DOUBLE, '
        '"y" DOUBLE, "feasible" BOOLEAN)'
    )
    stored = (
        extra_feasible
        if extra_feasible is not None
        else np.zeros(X_index.shape[0], dtype=bool)
    )
    objectives = (
        np.zeros(X_index.shape[0], dtype=float) if y is None else np.asarray(y, dtype=float)
    )
    connection.executemany(
        "INSERT INTO src VALUES (?, ?, ?, ?, ?, ?)",
        [
            (int(x[0]), int(x[1]), float(u[0]), float(u[1]), float(obj), bool(flag))
            for x, u, obj, flag in zip(X_index, U, objectives, stored, strict=True)
        ],
    )
    return connection


def _run(connection: object, cardinality: int = 4, **overrides: object):
    payload: dict[str, object] = {
        "connection": connection,
        "table": "src",
        "constraint_names": _CONSTRAINTS,
        "var_names": _VARS,
        "lower_bounds": _LOWER,
        "upper_bounds": _UPPER,
        "cardinality": cardinality,
        "problem_id": "L6750_B2",
        "source_dir": _SOURCE_DIR,
    }
    payload.update(overrides)
    return ela_feasible_region(**payload)  # type: ignore[arg-type]


def _region_from_mask(
    mask: np.ndarray,
    *,
    var_names: tuple[str, ...] = _VARS,
    lower: tuple[int, ...] = _LOWER,
    upper: tuple[int, ...] = _UPPER,
) -> FeasibleRegion:
    space = IndexSpace.from_bounds(var_names, lower, upper)
    flags = np.asarray(mask, dtype=bool)
    if flags.shape != (space.cardinality,):
        raise AssertionError(
            f"mask length {flags.size} != cardinality {space.cardinality}"
        )
    flags.setflags(write=False)
    return FeasibleRegion(
        space=space,
        mask=flags,
        n_feasible=int(flags.sum()),
        n_input=space.cardinality,
        n_unique=space.cardinality,
        constraint_names=_CONSTRAINTS,
    )


class TestElaFeasibleRegion(unittest.TestCase):
    """Cardinality-length feasible mask and L1=1 neighbourhood."""

    def test_two_by_two_mask_and_disconnected_f(self) -> None:
        X_index, U = _complete_grid()
        _V, _phi, feasible = constraint_violations(U)
        connection = _load_table(X_index, U)
        try:
            region = _run(connection)
        finally:
            connection.close()
        space = IndexSpace.from_bounds(_VARS, _LOWER, _UPPER)
        self.assertEqual(region.space.cardinality, 4)
        self.assertEqual(region.n_feasible, 2)
        self.assertEqual(region.n_input, 4)
        self.assertEqual(region.n_unique, 4)
        self.assertEqual(region.constraint_names, _CONSTRAINTS)
        self.assertEqual(FAMILY, "topology")
        self.assertEqual(tuple(region.mask.shape), (4,))
        self.assertFalse(region.mask.flags.writeable)
        expected = np.zeros(4, dtype=bool)
        for point, flag in zip(X_index, feasible, strict=True):
            expected[indices_to_linear(tuple(int(v) for v in point), space)] = bool(
                flag
            )
        np.testing.assert_array_equal(region.mask, expected)
        self.assertTrue(region.is_feasible((0, 3)))
        self.assertTrue(region.is_feasible((1, 4)))
        self.assertFalse(region.is_feasible((0, 4)))
        self.assertFalse(region.is_feasible((1, 3)))
        self.assertEqual(region.feasible_neighbours((0, 3)), ())
        self.assertEqual(region.feasible_neighbours((1, 4)), ())
        self.assertEqual(
            set(region.catalogue_neighbours((0, 3))),
            {(1, 3), (0, 4)},
        )
        self.assertNotIn((1, 4), region.catalogue_neighbours((0, 3)))

    def test_no_wrap_at_bounds(self) -> None:
        X_index, U = _complete_grid()
        connection = _load_table(X_index, U)
        try:
            region = _run(connection)
        finally:
            connection.close()
        neighbours = region.catalogue_neighbours((0, 3))
        self.assertNotIn((-1, 3), neighbours)
        self.assertNotIn((0, 2), neighbours)
        self.assertNotIn((2, 3), neighbours)
        corner = region.catalogue_neighbours((1, 4))
        self.assertNotIn((1, 5), corner)
        self.assertNotIn((2, 4), corner)

    def test_ignores_stored_feasible_flag(self) -> None:
        X_index, U = _complete_grid()
        connection = _load_table(
            X_index,
            U,
            extra_feasible=np.ones(X_index.shape[0], dtype=bool),
        )
        try:
            region = _run(connection)
        finally:
            connection.close()
        self.assertEqual(region.n_feasible, 2)
        self.assertFalse(region.is_feasible((0, 4)))

    def test_incomplete_catalogue_fails(self) -> None:
        X_index, U = _complete_grid()
        connection = _load_table(X_index[:3], U[:3])
        try:
            with self.assertRaises(ElaError) as ctx:
                _run(connection, cardinality=4)
        finally:
            connection.close()
        self.assertIn("cardinality", str(ctx.exception))

    def test_sampling_source_fails(self) -> None:
        X_index, U = _complete_grid()
        connection = _load_table(X_index, U)
        try:
            with self.assertRaises(ElaError) as ctx:
                _run(connection, source_type="sampling")
        finally:
            connection.close()
        self.assertIn("brute-force", str(ctx.exception))

    def test_importing_module_does_not_load_duckdb_pflacco_or_networkx(self) -> None:
        script = """
import sys
import core.ioh_core.workflows.la
import core.ioh_core.workflows.la.topology
loaded = [name for name in ("duckdb", "pflacco", "networkx") if name in sys.modules]
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


class TestElaFeasibleComponents(unittest.TestCase):
    """Union-find on analysis-local F under L1=1."""

    def test_two_by_two_disconnected_corners(self) -> None:
        X_index, U = _complete_grid()
        connection = _load_table(X_index, U)
        try:
            region = _run(connection)
        finally:
            connection.close()
        components = ela_feasible_components(region)
        self.assertEqual(components.n_components, 2)
        self.assertEqual(tuple(int(v) for v in components.sizes), (1, 1))
        self.assertEqual(components.n_isolated, 2)
        self.assertEqual(components.size_min, 1)
        self.assertEqual(components.size_max, 1)
        self.assertEqual(components.size_median, 1.0)
        self.assertEqual(components.largest_share, 0.5)
        self.assertEqual(components.isolated_fraction_of_f, 1.0)
        self.assertEqual(components.isolated_fraction_of_components, 1.0)
        self.assertFalse(components.labels.flags.writeable)
        self.assertFalse(components.sizes.flags.writeable)
        self.assertEqual(tuple(components.labels.shape), (4,))
        id_a = components.component_of((0, 3))
        id_b = components.component_of((1, 4))
        self.assertEqual(id_a, 0)
        self.assertEqual(id_b, 1)
        self.assertEqual(components.component_of((0, 4)), -1)
        self.assertEqual(components.component_of((1, 3)), -1)

    def test_connected_l_shape_is_one_component(self) -> None:
        mask = np.array([True, True, True, False])
        components = ela_feasible_components(_region_from_mask(mask))
        self.assertEqual(components.n_components, 1)
        self.assertEqual(tuple(int(v) for v in components.sizes), (3,))
        self.assertEqual(components.n_isolated, 0)
        self.assertEqual(components.size_min, 3)
        self.assertEqual(components.size_max, 3)
        self.assertEqual(components.size_median, 3.0)
        self.assertEqual(components.largest_share, 1.0)
        self.assertEqual(components.isolated_fraction_of_f, 0.0)
        self.assertEqual(components.isolated_fraction_of_components, 0.0)
        self.assertEqual(components.component_of((0, 3)), 0)
        self.assertEqual(components.component_of((0, 4)), 0)
        self.assertEqual(components.component_of((1, 3)), 0)
        self.assertEqual(components.component_of((1, 4)), -1)

    def test_both_isolated_fractions_on_mixed_sizes(self) -> None:
        mask = np.array([True, False, True, True])
        components = ela_feasible_components(
            _region_from_mask(
                mask,
                var_names=("geom_t_mm",),
                lower=(0,),
                upper=(3,),
            )
        )
        self.assertEqual(components.n_components, 2)
        self.assertEqual(tuple(int(v) for v in components.sizes), (1, 2))
        self.assertEqual(components.n_isolated, 1)
        self.assertEqual(components.size_min, 1)
        self.assertEqual(components.size_max, 2)
        self.assertEqual(components.size_median, 1.5)
        self.assertAlmostEqual(components.largest_share, 2.0 / 3.0)
        self.assertAlmostEqual(components.isolated_fraction_of_f, 1.0 / 3.0)
        self.assertEqual(components.isolated_fraction_of_components, 0.5)
        self.assertEqual(components.component_of((0,)), 0)
        self.assertEqual(components.component_of((1,)), -1)
        self.assertEqual(components.component_of((2,)), 1)
        self.assertEqual(components.component_of((3,)), 1)

    def test_empty_f_leaves_ratio_fields_undefined(self) -> None:
        mask = np.zeros(4, dtype=bool)
        components = ela_feasible_components(_region_from_mask(mask))
        self.assertEqual(components.n_components, 0)
        self.assertEqual(components.n_isolated, 0)
        self.assertEqual(tuple(components.sizes.tolist()), ())
        self.assertIsNone(components.size_min)
        self.assertIsNone(components.size_median)
        self.assertIsNone(components.size_max)
        self.assertIsNone(components.largest_share)
        self.assertIsNone(components.isolated_fraction_of_f)
        self.assertIsNone(components.isolated_fraction_of_components)
        np.testing.assert_array_equal(components.labels, np.full(4, -1, dtype=np.int32))


class TestElaGlobalOptima(unittest.TestCase):
    """Exact stored y == y* ties on analysis-local F."""

    def test_identical_y_keeps_both_disconnected_optima(self) -> None:
        X_index, U = _complete_grid()
        y = np.array([1.0, 9.0, 9.0, 1.0], dtype=float)
        connection = _load_table(X_index, U, y=y)
        try:
            components = ela_feasible_components(_run(connection))
            optima = ela_global_optima(
                connection=connection, table="src", components=components
            )
        finally:
            connection.close()
        self.assertEqual(optima.y_star, 1.0)
        self.assertEqual(optima.n_optima, 2)
        self.assertEqual(optima.n_optima_components, 2)
        self.assertEqual(tuple(int(v) for v in optima.linear_ids), (0, 3))
        self.assertEqual(tuple(int(v) for v in optima.component_ids), (0, 1))
        self.assertFalse(optima.linear_ids.flags.writeable)
        self.assertFalse(optima.component_ids.flags.writeable)

    def test_one_ulp_is_not_a_tie(self) -> None:
        X_index, U = _complete_grid()
        y = np.array([1.0, 9.0, 9.0, np.nextafter(1.0, 2.0)], dtype=float)
        connection = _load_table(X_index, U, y=y)
        try:
            components = ela_feasible_components(_run(connection))
            optima = ela_global_optima(
                connection=connection, table="src", components=components
            )
        finally:
            connection.close()
        self.assertEqual(optima.y_star, 1.0)
        self.assertEqual(optima.n_optima, 1)
        self.assertEqual(optima.n_optima_components, 1)
        self.assertEqual(tuple(int(v) for v in optima.linear_ids), (0,))
        self.assertEqual(tuple(int(v) for v in optima.component_ids), (0,))

    def test_infeasible_lower_y_is_ignored(self) -> None:
        X_index, U = _complete_grid()
        y = np.array([2.0, -100.0, 9.0, 2.0], dtype=float)
        connection = _load_table(X_index, U, y=y)
        try:
            components = ela_feasible_components(_run(connection))
            optima = ela_global_optima(
                connection=connection, table="src", components=components
            )
        finally:
            connection.close()
        self.assertEqual(optima.y_star, 2.0)
        self.assertEqual(optima.n_optima, 2)
        self.assertEqual(tuple(int(v) for v in optima.linear_ids), (0, 3))

    def test_empty_f_leaves_y_star_undefined(self) -> None:
        X_index, U = _complete_grid()
        U = np.full_like(U, 1.5)
        y = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
        connection = _load_table(X_index, U, y=y)
        try:
            components = ela_feasible_components(_run(connection))
            optima = ela_global_optima(
                connection=connection, table="src", components=components
            )
        finally:
            connection.close()
        self.assertEqual(components.n_components, 0)
        self.assertIsNone(optima.y_star)
        self.assertEqual(optima.n_optima, 0)
        self.assertEqual(optima.n_optima_components, 0)
        self.assertEqual(tuple(optima.linear_ids.tolist()), ())
        self.assertEqual(tuple(optima.component_ids.tolist()), ())

    def test_nonfinite_y_on_f_fails(self) -> None:
        X_index, U = _complete_grid()
        y = np.array([1.0, 9.0, 9.0, np.inf], dtype=float)
        connection = _load_table(X_index, U, y=y)
        try:
            components = ela_feasible_components(_run(connection))
            with self.assertRaises(ElaError) as ctx:
                ela_global_optima(
                    connection=connection, table="src", components=components
                )
        finally:
            connection.close()
        self.assertIn("finite", str(ctx.exception))


class TestElaFeasibleBoundary(unittest.TestCase):
    """∂F is an in-bounds infeasible L1 neighbour, not a variable bound."""

    def test_stop7_corners_are_on_boundary(self) -> None:
        X_index, U = _complete_grid()
        connection = _load_table(X_index, U)
        try:
            region = _run(connection)
        finally:
            connection.close()
        boundary = ela_feasible_boundary(region)
        self.assertEqual(boundary.n_boundary, 2)
        self.assertEqual(boundary.boundary_fraction_of_f, 1.0)
        self.assertTrue(boundary.is_boundary((0, 3)))
        self.assertTrue(boundary.is_boundary((1, 4)))
        self.assertFalse(boundary.is_boundary((0, 4)))
        self.assertFalse(boundary.is_boundary((1, 3)))
        self.assertFalse(boundary.mask.flags.writeable)

    def test_full_space_f_has_empty_boundary(self) -> None:
        mask = np.ones(4, dtype=bool)
        boundary = ela_feasible_boundary(_region_from_mask(mask))
        self.assertEqual(boundary.n_boundary, 0)
        self.assertEqual(boundary.boundary_fraction_of_f, 0.0)
        self.assertFalse(boundary.is_boundary((0, 3)))
        self.assertFalse(boundary.is_boundary((1, 4)))

    def test_one_cell_catalogue_is_not_boundary(self) -> None:
        mask = np.array([True])
        boundary = ela_feasible_boundary(
            _region_from_mask(
                mask,
                var_names=("geom_t_mm",),
                lower=(0,),
                upper=(0,),
            )
        )
        self.assertEqual(boundary.n_boundary, 0)
        self.assertEqual(boundary.boundary_fraction_of_f, 0.0)
        self.assertFalse(boundary.is_boundary((0,)))

    def test_three_by_three_hole_marks_edges_not_corners(self) -> None:
        mask = np.ones(9, dtype=bool)
        mask[4] = False
        boundary = ela_feasible_boundary(
            _region_from_mask(
                mask,
                lower=(0, 0),
                upper=(2, 2),
            )
        )
        self.assertEqual(boundary.n_boundary, 4)
        self.assertTrue(boundary.is_boundary((1, 0)))
        self.assertTrue(boundary.is_boundary((1, 2)))
        self.assertTrue(boundary.is_boundary((0, 1)))
        self.assertTrue(boundary.is_boundary((2, 1)))
        self.assertFalse(boundary.is_boundary((0, 0)))
        self.assertFalse(boundary.is_boundary((0, 2)))
        self.assertFalse(boundary.is_boundary((2, 0)))
        self.assertFalse(boundary.is_boundary((2, 2)))
        self.assertFalse(boundary.is_boundary((1, 1)))

    def test_empty_f_leaves_boundary_fraction_undefined(self) -> None:
        boundary = ela_feasible_boundary(_region_from_mask(np.zeros(4, dtype=bool)))
        self.assertEqual(boundary.n_boundary, 0)
        self.assertIsNone(boundary.boundary_fraction_of_f)


class TestElaNearOptimal(unittest.TestCase):
    """G_τ / B_τ on raw y with caller-supplied fractional τ."""

    def test_tau_zero_matches_y_star_set(self) -> None:
        X_index, U = _complete_grid()
        y = np.array([1.0, 9.0, 9.0, 2.0], dtype=float)
        connection = _load_table(X_index, U, y=y)
        try:
            region = _run(connection)
            components = ela_feasible_components(region)
            optima = ela_global_optima(
                connection=connection, table="src", components=components
            )
            boundary = ela_feasible_boundary(region)
            result = ela_near_optimal(
                connection=connection,
                table="src",
                optima=optima,
                boundary=boundary,
                taus=(0.0, 1.0),
            )
        finally:
            connection.close()
        self.assertEqual(optima.n_optima, 1)
        zero, one = result.bands
        self.assertEqual(zero.tau, 0.0)
        self.assertEqual(zero.n_g, 1)
        self.assertEqual(zero.g_fraction_of_f, 0.5)
        self.assertEqual(zero.n_components_represented, 1)
        self.assertEqual(zero.b_tau, 1.0)
        self.assertEqual(zero.n_g_components, 1)
        self.assertEqual(zero.g_largest_share, 1.0)
        self.assertEqual(one.tau, 1.0)
        self.assertEqual(one.n_g, 2)
        self.assertEqual(one.g_fraction_of_f, 1.0)
        self.assertEqual(one.n_components_represented, 2)
        self.assertEqual(one.b_tau, 1.0)
        self.assertEqual(one.n_g_components, 2)

    def test_empty_tau_list_returns_no_bands(self) -> None:
        X_index, U = _complete_grid()
        connection = _load_table(X_index, U, y=np.array([1.0, 9.0, 9.0, 1.0]))
        try:
            region = _run(connection)
            optima = ela_global_optima(
                connection=connection,
                table="src",
                components=ela_feasible_components(region),
            )
            result = ela_near_optimal(
                connection=connection,
                table="src",
                optima=optima,
                boundary=ela_feasible_boundary(region),
                taus=(),
            )
        finally:
            connection.close()
        self.assertEqual(result.bands, ())

    def test_negative_y_star_leaves_bands_undefined(self) -> None:
        X_index, U = _complete_grid()
        y = np.array([-2.0, 9.0, 9.0, -1.0], dtype=float)
        connection = _load_table(X_index, U, y=y)
        try:
            region = _run(connection)
            optima = ela_global_optima(
                connection=connection,
                table="src",
                components=ela_feasible_components(region),
            )
            result = ela_near_optimal(
                connection=connection,
                table="src",
                optima=optima,
                boundary=ela_feasible_boundary(region),
                taus=(0.0, 0.1),
            )
        finally:
            connection.close()
        self.assertEqual(optima.y_star, -2.0)
        for band in result.bands:
            self.assertIsNone(band.n_g)
            self.assertIsNone(band.g_fraction_of_f)
            self.assertIsNone(band.n_components_represented)
            self.assertIsNone(band.b_tau)
            self.assertIsNone(band.n_g_components)
            self.assertIsNone(band.g_largest_share)

    def test_empty_f_leaves_bands_undefined(self) -> None:
        X_index, U = _complete_grid()
        U = np.full_like(U, 1.5)
        connection = _load_table(X_index, U, y=np.array([0.1, 0.2, 0.3, 0.4]))
        try:
            region = _run(connection)
            optima = ela_global_optima(
                connection=connection,
                table="src",
                components=ela_feasible_components(region),
            )
            result = ela_near_optimal(
                connection=connection,
                table="src",
                optima=optima,
                boundary=ela_feasible_boundary(region),
                taus=(0.0,),
            )
        finally:
            connection.close()
        self.assertIsNone(optima.y_star)
        self.assertIsNone(result.bands[0].n_g)
        self.assertIsNone(result.bands[0].b_tau)

    def test_negative_tau_fails(self) -> None:
        X_index, U = _complete_grid()
        connection = _load_table(X_index, U, y=np.array([1.0, 9.0, 9.0, 1.0]))
        try:
            region = _run(connection)
            optima = ela_global_optima(
                connection=connection,
                table="src",
                components=ela_feasible_components(region),
            )
            with self.assertRaises(ElaError) as ctx:
                ela_near_optimal(
                    connection=connection,
                    table="src",
                    optima=optima,
                    boundary=ela_feasible_boundary(region),
                    taus=(-0.1,),
                )
        finally:
            connection.close()
        self.assertIn("tau", str(ctx.exception))

    def test_g_tau_induced_components_can_exceed_f_component_count(self) -> None:
        """One connected F, several disconnected near-optimal islands."""
        mask = np.ones(5, dtype=bool)
        region = _region_from_mask(
            mask,
            var_names=("geom_t_mm",),
            lower=(0,),
            upper=(4,),
        )
        components = ela_feasible_components(region)
        self.assertEqual(components.n_components, 1)
        y = np.array([1.0, 2.0, 1.0, 2.0, 1.0], dtype=float)
        duckdb = import_duckdb()
        connection = duckdb.connect()
        connection.execute(
            'CREATE TABLE src ("idx__geom_t_mm" INTEGER, "y" DOUBLE, '
            '"util__A_bending_capacity" DOUBLE, '
            '"util__Z1_nt_dt_combination_capacity" DOUBLE)'
        )
        connection.executemany(
            "INSERT INTO src VALUES (?, ?, 0.5, 0.5)",
            [(i, float(v)) for i, v in enumerate(y)],
        )
        try:
            optima = ela_global_optima(
                connection=connection, table="src", components=components
            )
            result = ela_near_optimal(
                connection=connection,
                table="src",
                optima=optima,
                boundary=ela_feasible_boundary(region),
                taus=(0.0,),
            )
        finally:
            connection.close()
        band = result.bands[0]
        self.assertEqual(band.n_g, 3)
        self.assertEqual(band.n_components_represented, 1)
        self.assertEqual(band.n_g_components, 3)
        self.assertEqual(band.n_g_isolated, 3)
        self.assertEqual(band.g_largest_share, 1.0 / 3.0)


class TestElaTopologyStop7Pipeline(unittest.TestCase):
    """15.5: disconnected F and both identical y* on one STOP 7 grid."""

    def test_disconnected_f_keeps_both_identical_y_star(self) -> None:
        X_index, U = _complete_grid()
        y = np.array([1.0, 9.0, 9.0, 1.0], dtype=float)
        connection = _load_table(X_index, U, y=y)
        try:
            region = _run(connection)
            components = ela_feasible_components(region)
            optima = ela_global_optima(
                connection=connection, table="src", components=components
            )
            boundary = ela_feasible_boundary(region)
            near = ela_near_optimal(
                connection=connection,
                table="src",
                optima=optima,
                boundary=boundary,
                taus=(0.0,),
            )
        finally:
            connection.close()
        self.assertEqual(region.n_feasible, 2)
        self.assertEqual(components.n_components, 2)
        self.assertEqual(tuple(int(v) for v in components.sizes), (1, 1))
        self.assertEqual(region.feasible_neighbours((0, 3)), ())
        self.assertEqual(region.feasible_neighbours((1, 4)), ())
        self.assertEqual(optima.y_star, 1.0)
        self.assertEqual(optima.n_optima, 2)
        self.assertEqual(tuple(int(v) for v in optima.linear_ids), (0, 3))
        self.assertEqual(tuple(int(v) for v in optima.component_ids), (0, 1))
        self.assertTrue(boundary.is_boundary((0, 3)))
        self.assertTrue(boundary.is_boundary((1, 4)))
        band = near.bands[0]
        self.assertEqual(band.tau, 0.0)
        self.assertEqual(band.n_g, 2)
        self.assertEqual(band.g_fraction_of_f, 1.0)
        self.assertEqual(band.n_components_represented, 2)
        self.assertEqual(band.b_tau, 1.0)


class TestElaTopologyFeatureRows(unittest.TestCase):
    """Tidy aggregates only; no per-component or linear-id rows."""

    def test_stop8_grid_matches_known_aggregates(self) -> None:
        X_index, U = _complete_grid()
        y = np.array([10.0, 20.0, 20.0, 10.0], dtype=float)
        connection = _load_table(X_index, U, y=y)
        try:
            rows = ela_topology_features(
                connection=connection,
                table="src",
                constraint_names=_CONSTRAINTS,
                var_names=_VARS,
                lower_bounds=_LOWER,
                upper_bounds=_UPPER,
                cardinality=4,
                problem_id="L6750_B2",
                source_dir=_SOURCE_DIR,
                taus=(0.0, 1.0),
            )
        finally:
            connection.close()
        by_name = {row.feature_name: row for row in rows}
        self.assertEqual(by_name[FEATURE_N_FEASIBLE].value, 2.0)
        self.assertEqual(by_name[FEATURE_Y_STAR].value, 10.0)
        self.assertEqual(by_name[FEATURE_Y_STAR].role, "diagnostic")
        self.assertEqual(by_name[FEATURE_N_OPTIMA].value, 2.0)
        self.assertEqual(by_name[n_g_feature_name(0.0)].value, 2.0)
        self.assertEqual(by_name[n_g_feature_name(1.0)].value, 2.0)
        self.assertEqual(len(rows), 13 + 1 + 2 * 10)
        self.assertEqual(rows[0].family, FAMILY)
        self.assertEqual(rows[0].x_view, "index")
        self.assertEqual(rows[0].source_type, "brute_force")
        names = [row.feature_name for row in rows]
        self.assertEqual(len(names), len(set(names)))
        self.assertFalse(any("linear" in name for name in names))
        self.assertFalse(any("component_size" in name for name in names))

    def test_parse_taus_rejects_duplicates_and_allows_empty(self) -> None:
        self.assertEqual(parse_ela_taus(()), ())
        self.assertEqual(parse_ela_taus((0.0, 1.0)), (0.0, 1.0))
        with self.assertRaises(ElaError):
            parse_ela_taus((0.0, 0.0))
        with self.assertRaises(ElaError):
            parse_ela_taus((-0.1,))


if __name__ == "__main__":
    unittest.main()
