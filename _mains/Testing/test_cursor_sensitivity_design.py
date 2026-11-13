"""
C1.1 gate: SALib Saltelli design helpers and sensitivity optional extra.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from core.ioh_core.workflows.sensitivity_analysis.deps import (
    SensitivityDependencyError,
    import_salib,
)
from core.ioh_core.workflows.sensitivity_analysis.design import (
    cross_sample_row,
    extract_base_unit_matrices,
    generate_base_unit_matrices,
    load_sensitivity_design,
    sample_saltelli_unit_matrix,
    seal_sensitivity_design,
    unit_to_catalogue_indices,
    verify_sealed_arrays,
)
from core.ioh_core.workflows.sensitivity_analysis.models import (
    SensitivityDesignDocument,
    SensitivityDesignError,
)
from core.ioh_core.workflows.optimization_runs.snapshot import (
    resolve_snapshot_slab_dir,
    snapshot_slab_inputs,
)
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

_LOCKED_SENSITIVITY = [
    "pyarrow>=25.0.0,<26",
    "SALib>=1.5.1,<1.6",
]


def _salib_available() -> bool:
    try:
        import_salib()
    except SensitivityDependencyError:
        return False
    return True


@unittest.skipUnless(_salib_available(), 'install with: pip install -e ".[sensitivity]"')
class TestSensitivityDesign(unittest.TestCase):
    """SALib packing, reproducibility, and catalogue mapping."""

    def test_sensitivity_extra_range_is_locked(self) -> None:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        self.assertEqual(extras["sensitivity"], _LOCKED_SENSITIVITY)

    def test_sensitivity_extra_is_not_in_algorithms_all(self) -> None:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        for req in extras["algorithms-all"]:
            self.assertNotIn("SALib", req)
            self.assertNotIn("pyarrow", req)

    def test_importing_package_does_not_load_salib(self) -> None:
        sys.modules.pop("SALib", None)
        sys.modules.pop("SALib.sample", None)
        sys.modules.pop("SALib.sample.sobol", None)

        import core.ioh_core.workflows.sensitivity_analysis  # noqa: F401
        import core.ioh_core.workflows.sensitivity_analysis.deps as deps_mod

        self.assertIs(deps_mod.import_salib, import_salib)
        self.assertNotIn("SALib", sys.modules)

    def test_missing_salib_error_message(self) -> None:
        from core.ioh_core.workflows.sensitivity_analysis import deps as deps_mod

        with patch(
            "core.ioh_core.workflows.sensitivity_analysis.deps.importlib.import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'SALib'", name="SALib"
            ),
        ):
            with self.assertRaises(deps_mod.SensitivityDependencyError) as ctx:
                deps_mod.import_salib()
        message = str(ctx.exception)
        self.assertEqual(
            message,
            "sensitivity_analysis requires optional dependency 'SALib'; "
            'install with:\npip install -e ".[sensitivity]"',
        )
        self.assertIsInstance(ctx.exception.__cause__, ModuleNotFoundError)
        self.assertIsInstance(ctx.exception, ImportError)

    def test_sample_size_must_be_power_of_two(self) -> None:
        with self.assertRaisesRegex(ValueError, "power of two"):
            generate_base_unit_matrices(
                sample_size=3,
                factor_count=2,
                scramble=True,
                seed=1,
            )

    def test_packed_shape_and_ab_extraction_order(self) -> None:
        sample_size = 8
        factor_count = 3
        packed = sample_saltelli_unit_matrix(
            sample_size=sample_size,
            factor_count=factor_count,
            scramble=True,
            seed=11,
        )
        self.assertEqual(packed.shape, (sample_size * (factor_count + 2), factor_count))

        a_unit, b_unit = extract_base_unit_matrices(packed, factor_count)
        self.assertEqual(a_unit.shape, (sample_size, factor_count))
        self.assertEqual(b_unit.shape, (sample_size, factor_count))

        step = factor_count + 2
        np.testing.assert_array_equal(a_unit, packed[0::step])
        np.testing.assert_array_equal(b_unit, packed[(step - 1)::step])

        for factor_index in range(factor_count):
            ab = packed[(factor_index + 1)::step]
            expected = np.array(
                [
                    cross_sample_row(a_unit[row], b_unit[row], factor_index)
                    for row in range(sample_size)
                ],
                dtype=float,
            )
            np.testing.assert_allclose(ab, expected, atol=0.0, rtol=0.0)

        # B is last in the block, not immediately after A.
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            b_unit,
            packed[1::step],
        )

    def test_generate_matches_fresh_salib_sample(self) -> None:
        kwargs = {
            "sample_size": 16,
            "factor_count": 4,
            "scramble": True,
            "seed": 20260802,
        }
        a_first, b_first = generate_base_unit_matrices(**kwargs)
        a_second, b_second = generate_base_unit_matrices(**kwargs)
        np.testing.assert_allclose(a_first, a_second, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(b_first, b_second, atol=0.0, rtol=0.0)

        packed = sample_saltelli_unit_matrix(**kwargs)
        a_from_packed, b_from_packed = extract_base_unit_matrices(
            packed, kwargs["factor_count"]
        )
        np.testing.assert_allclose(a_first, a_from_packed, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(b_first, b_from_packed, atol=0.0, rtol=0.0)

    def test_seed_change_changes_design(self) -> None:
        shared = {"sample_size": 8, "factor_count": 2, "scramble": True}
        a_a, b_a = generate_base_unit_matrices(seed=1, **shared)
        a_b, b_b = generate_base_unit_matrices(seed=2, **shared)
        self.assertFalse(np.allclose(a_a, a_b) and np.allclose(b_a, b_b))

    def test_unscrambled_seed_still_accepted(self) -> None:
        a_unit, b_unit = generate_base_unit_matrices(
            sample_size=4,
            factor_count=2,
            scramble=False,
            seed=7,
        )
        self.assertEqual(a_unit.shape, (4, 2))
        self.assertEqual(b_unit.shape, (4, 2))

    def test_unit_to_catalogue_indices_inclusive_floor_map(self) -> None:
        unit = np.array(
            [
                [0.0, 0.499999, 0.5, 0.999999],
                [0.25, 0.0, 1.0, 0.75],
            ],
            dtype=float,
        )
        # Force u=1 edge through clip path using a value that floors to last tick.
        unit[1, 2] = np.nextafter(1.0, 0.0)
        indices = unit_to_catalogue_indices(
            unit,
            lower_bounds=[10, 0, 2, 5],
            upper_bounds=[13, 3, 5, 8],
        )
        expected = np.array(
            [
                [10, 1, 4, 8],
                [11, 0, 5, 8],
            ],
            dtype=int,
        )
        np.testing.assert_array_equal(indices, expected)

    def test_unit_to_catalogue_rejects_out_of_unit_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            unit_to_catalogue_indices(
                np.array([[1.0000001, 0.0]]),
                lower_bounds=[0, 0],
                upper_bounds=[1, 1],
            )

    def test_cross_sample_row_replaces_exactly_one_column(self) -> None:
        a_row = np.array([1.0, 2.0, 3.0])
        b_row = np.array([9.0, 8.0, 7.0])
        crossed = cross_sample_row(a_row, b_row, 1)
        np.testing.assert_array_equal(crossed, np.array([1.0, 8.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "out of range"):
            cross_sample_row(a_row, b_row, 3)


@unittest.skipUnless(_salib_available(), 'install with: pip install -e ".[sensitivity]"')
class TestSensitivityDesignSealing(unittest.TestCase):
    """Seal / reload / regenerate design artifacts from snapshotted CSVs."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.inputs_dir = Path(self._tmp.name) / "inputs"
        snapshot_slab_inputs(slab_type="hp_slab", inputs_dir=self.inputs_dir)
        self.snapshot_dir = resolve_snapshot_slab_dir(self.inputs_dir, "hp_slab")
        self.design_dir = Path(self._tmp.name) / "sensitivity"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seal(self, **overrides):
        kwargs = {
            "snapshot_dir": self.snapshot_dir,
            "slab_type": "hp_slab",
            "problem_id": "L6750_B2",
            "factor_names": ["geom_h_ges_mm", "mat_conc_id", "geom_t_mm"],
            "sample_size": 8,
            "response_key": "y",
            "scramble": True,
            "seed": 20260802,
            "design_dir": self.design_dir,
            "overwrite": True,
        }
        kwargs.update(overrides)
        return seal_sensitivity_design(**kwargs)

    def test_seal_round_trip_and_regenerate(self) -> None:
        document = self._seal()
        self.assertEqual(document.factor_count, 3)
        self.assertEqual(
            document.factor_names,
            ("geom_h_ges_mm", "mat_conc_id", "geom_t_mm"),
        )
        self.assertTrue(
            any(item.name == "geom_nt" for item in document.held_constant_variables)
        )
        # Categorical concrete class appears as a string catalogue value.
        concrete = next(
            factor
            for factor in document.factors
            if factor.name == "mat_conc_id"
        )
        self.assertIsInstance(concrete.catalogue_values[0], str)

        loaded = load_sensitivity_design(self.design_dir)
        self.assertEqual(loaded.to_json(), document.to_json())
        a_unit, b_unit, a_index, b_index = verify_sealed_arrays(self.design_dir)
        self.assertEqual(a_unit.shape, (8, 3))
        self.assertEqual(b_unit.shape, (8, 3))
        self.assertEqual(a_index.dtype.kind, "i")
        self.assertEqual(b_index.dtype.kind, "i")

        for name in ("A_unit.npy", "B_unit.npy", "A_index.npy", "B_index.npy"):
            with open(self.design_dir / name, "rb") as handle:
                # NumPy format magic; ensure no pickle protocol payload.
                header = handle.read(8)
            self.assertTrue(header.startswith(b"\x93NUMPY"))

    def test_equal_science_same_array_hashes(self) -> None:
        first = self._seal()
        second_dir = Path(self._tmp.name) / "sensitivity_b"
        second = self._seal(design_dir=second_dir)
        self.assertEqual(first.array_sha256, second.array_sha256)
        self.assertEqual(first.salib_version, second.salib_version)

    def test_seed_change_changes_hashes(self) -> None:
        first = self._seal(seed=1)
        second = self._seal(seed=2, design_dir=Path(self._tmp.name) / "sens2")
        self.assertNotEqual(first.array_sha256, second.array_sha256)

    def test_factor_subset_and_order(self) -> None:
        document = self._seal(
            factor_names=["geom_t_mm", "geom_h_ges_mm"],
        )
        self.assertEqual(document.factor_names, ("geom_t_mm", "geom_h_ges_mm"))
        held_names = {item.name for item in document.held_constant_variables}
        self.assertIn("mat_conc_id", held_names)
        self.assertNotIn("geom_t_mm", held_names)

    def test_rejects_lookup_or_fixed_factor(self) -> None:
        with self.assertRaisesRegex(SensitivityDesignError, "role"):
            self._seal(factor_names=["mat_reinf_id"])
        with self.assertRaisesRegex(SensitivityDesignError, "role"):
            self._seal(factor_names=["geom_span_mm"])

    def test_overwrite_required_when_present(self) -> None:
        self._seal()
        with self.assertRaisesRegex(SensitivityDesignError, "already exist"):
            seal_sensitivity_design(
                snapshot_dir=self.snapshot_dir,
                slab_type="hp_slab",
                problem_id="L6750_B2",
                factor_names=["geom_h_ges_mm", "geom_t_mm"],
                sample_size=4,
                response_key="y",
                scramble=True,
                seed=1,
                design_dir=self.design_dir,
                overwrite=False,
            )

    def test_json_round_trip(self) -> None:
        document = self._seal()
        restored = SensitivityDesignDocument.from_json(document.to_json())
        self.assertEqual(restored.to_json(), document.to_json())


if __name__ == "__main__":
    unittest.main()
