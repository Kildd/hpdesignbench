"""
Phase ALG4.1 gate: optional algorithm extras and dependency errors.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from core.ioh_core.algorithms.errors import (
    AlgorithmDependencyError,
    import_optional_dependency,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

_LOCKED_EXTRAS = {
    "algorithms-pymoo": ["pymoo>=0.6.2,<0.7"],
    "algorithms-hyperopt": ["hyperopt>=0.3.0,<0.4"],
    "algorithms-rbfopt": ["rbfopt>=4.3.0,<4.4"],
    "algorithms-optuna": ["optuna>=4.0.0,<5"],
    "algorithms-leap": ["leap-ec>=0.8.2,<0.9"],
    "algorithms-cmaes": ["cmaes>=0.13.0,<0.14"],
    "algorithms-starpso": ["starpso>=0.1.3,<0.2"],
    "algorithms-nevergrad": ["nevergrad>=1.0.12,<1.1"],
    "algorithms-smt": ["smt>=2.14.1,<2.15"],
    "algorithms-blackboxoptim": ["blackboxoptim>=1.1.0,<1.2"],
    "algorithms-altbacken": ["altbacken>=0.15.1,<0.16"],
    "algorithms-dfl": [],
    "algorithms-nomad": ["PyNomadBBO>=4.5.1,<4.6"],
    "algorithms-pfns4bo": [
        "pfns4bo>=0.1.5,<0.2",
        "torch>=2.1.0,<3",
        "torchvision>=0.16.0,<1",
    ],
    "algorithms-optax": [
        "optax>=0.2.8,<0.3",
        "jax>=0.5.3,<0.10",
        "jaxlib>=0.5.3,<0.10",
    ],
}


class TestOptionalAlgorithmDependencies(unittest.TestCase):
    """Locked extras and actionable missing-dependency errors."""

    def test_optional_extra_ranges_are_locked(self) -> None:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        for name, expected in _LOCKED_EXTRAS.items():
            self.assertEqual(extras[name], expected)

    def test_algorithms_all_is_exact_union(self) -> None:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        expected = [
            req
            for name in (
                "algorithms-pymoo",
                "algorithms-hyperopt",
                "algorithms-rbfopt",
                "algorithms-optuna",
                "algorithms-leap",
                "algorithms-cmaes",
                "algorithms-starpso",
                "algorithms-nevergrad",
                "algorithms-smt",
                "algorithms-blackboxoptim",
                "algorithms-altbacken",
                "algorithms-dfl",
                "algorithms-nomad",
                "algorithms-pfns4bo",
                "algorithms-optax",
            )
            for req in extras[name]
        ]
        self.assertEqual(extras["algorithms-all"], expected)

    def test_missing_dependency_error_message(self) -> None:
        with patch(
            "core.ioh_core.algorithms.errors.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'pymoo'", name="pymoo"),
        ):
            with self.assertRaises(AlgorithmDependencyError) as ctx:
                import_optional_dependency(
                    "pymoo",
                    algorithm_name="PYMOO_GA",
                    extra="algorithms-pymoo",
                )
        message = str(ctx.exception)
        self.assertEqual(
            message,
            "PYMOO_GA requires optional dependency 'pymoo'; install with:\n"
            'pip install -e ".[algorithms-pymoo]"',
        )
        self.assertIsInstance(ctx.exception.__cause__, ModuleNotFoundError)
        self.assertIsInstance(ctx.exception, ImportError)

    def test_non_import_runtime_error_is_not_relabelled(self) -> None:
        with patch(
            "core.ioh_core.algorithms.errors.importlib.import_module",
            side_effect=RuntimeError("broken native extension"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                import_optional_dependency(
                    "pymoo",
                    algorithm_name="PYMOO_GA",
                    extra="algorithms-pymoo",
                )
        self.assertNotIsInstance(ctx.exception, AlgorithmDependencyError)
        self.assertIn("broken native extension", str(ctx.exception))

    def test_algorithms_package_import_does_not_load_optional_libs(self) -> None:
        import sys

        # Third-party import roots (leap-ec imports as leap_ec;
        # starpso imports as star_pso).
        third_party_roots = (
            "pymoo",
            "hyperopt",
            "rbfopt",
            "optuna",
            "leap_ec",
            "cmaes",
            "star_pso",
            "nevergrad",
            "smt",
            "blackboxoptim",
            "altbacken",
            "PyNomad",
            "pfns4bo",
            # Do not unload jax/jaxlib or torch: C-extension modules are unsafe
            # to delete from sys.modules and reimport in-process.
            "optax",
        )
        # Local adapter package directory names under algorithms/.
        adapter_dirs = (
            "pymoo",
            "hyperopt",
            "rbfopt",
            "optuna",
            "leap",
            "cmaes",
            "starpso",
            "nevergrad",
            "smt",
            "blackboxoptim",
            "altbacken",
            "dfl",
            "nomad",
            "pfns4bo",
            "optax",
        )

        for name in third_party_roots:
            sys.modules.pop(name, None)
            doomed = [key for key in sys.modules if key.startswith(f"{name}.")]
            for key in doomed:
                sys.modules.pop(key, None)

        for name in adapter_dirs:
            adapter_prefix = f"core.ioh_core.algorithms.{name}."
            for key in list(sys.modules):
                if key.startswith(adapter_prefix) or key == (
                    f"core.ioh_core.algorithms.{name}"
                ):
                    sys.modules.pop(key, None)

        import importlib

        importlib.import_module("core.ioh_core.algorithms")
        importlib.import_module("core.ioh_core.algorithms.registry")
        importlib.import_module("core.ioh_core.algorithms.errors")

        self.assertNotIn("pymoo", sys.modules)
        self.assertNotIn("hyperopt", sys.modules)
        self.assertNotIn("rbfopt", sys.modules)
        self.assertNotIn("optuna", sys.modules)
        self.assertNotIn("leap_ec", sys.modules)
        self.assertNotIn("cmaes", sys.modules)
        self.assertNotIn("star_pso", sys.modules)
        self.assertNotIn("nevergrad", sys.modules)
        self.assertNotIn("smt", sys.modules)
        self.assertNotIn("blackboxoptim", sys.modules)
        self.assertNotIn("altbacken", sys.modules)
        self.assertNotIn("PyNomad", sys.modules)
        self.assertNotIn("pfns4bo", sys.modules)
        self.assertNotIn("optax", sys.modules)
        import core.ioh_core.algorithms.registry as registry_mod

        self.assertFalse(hasattr(registry_mod, "torch"))
        self.assertFalse(hasattr(registry_mod, "pfns4bo"))


if __name__ == "__main__":
    unittest.main()
