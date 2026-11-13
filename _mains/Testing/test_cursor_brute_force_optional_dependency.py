"""
Phase B1 gate: optional bruteforce extra and PyArrow dependency errors.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from core.ioh_core.workflows.brute_force.deps import import_pyarrow

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

_LOCKED_BRUTEFORCE = ["pyarrow>=25.0.0,<26"]


class TestBruteForceOptionalDependency(unittest.TestCase):
    """Locked bruteforce extra and actionable missing-dependency errors."""

    def test_bruteforce_extra_range_is_locked(self) -> None:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        self.assertEqual(extras["bruteforce"], _LOCKED_BRUTEFORCE)

    def test_bruteforce_extra_is_not_in_algorithms_all(self) -> None:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        for req in extras["algorithms-all"]:
            self.assertNotIn("pyarrow", req)

    def test_importing_brute_force_package_does_not_load_pyarrow(self) -> None:
        sys.modules.pop("pyarrow", None)
        sys.modules.pop("pyarrow.parquet", None)

        import core.ioh_core.workflows.brute_force  # noqa: F401
        import core.ioh_core.workflows.brute_force.deps as deps_mod

        self.assertIs(deps_mod.import_pyarrow, import_pyarrow)
        self.assertNotIn("pyarrow", sys.modules)
        self.assertNotIn("pyarrow.parquet", sys.modules)

    def test_missing_pyarrow_error_message(self) -> None:
        from core.ioh_core.workflows.brute_force import deps as deps_mod

        with patch(
            "core.ioh_core.workflows.brute_force.deps.importlib.import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'pyarrow'", name="pyarrow"
            ),
        ):
            with self.assertRaises(deps_mod.BruteForceDependencyError) as ctx:
                deps_mod.import_pyarrow()
        message = str(ctx.exception)
        self.assertEqual(
            message,
            "brute_force requires optional dependency 'pyarrow'; install with:\n"
            'pip install -e ".[bruteforce]"',
        )
        self.assertIsInstance(ctx.exception.__cause__, ModuleNotFoundError)
        self.assertIsInstance(ctx.exception, ImportError)

    def test_non_import_runtime_error_is_not_relabelled(self) -> None:
        from core.ioh_core.workflows.brute_force import deps as deps_mod

        with patch(
            "core.ioh_core.workflows.brute_force.deps.importlib.import_module",
            side_effect=RuntimeError("broken native extension"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                deps_mod.import_pyarrow()
        self.assertEqual(str(ctx.exception), "broken native extension")


if __name__ == "__main__":
    unittest.main()
