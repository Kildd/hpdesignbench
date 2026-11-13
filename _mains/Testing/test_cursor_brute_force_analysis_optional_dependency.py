"""
Phase C3.1 gate: optional bruteforce-analysis extra and DuckDB import errors.
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

from core.ioh_core.workflows.brute_force.deps import import_duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

_LOCKED_BRUTEFORCE = ["pyarrow>=25.0.0,<26"]
_LOCKED_BRUTEFORCE_ANALYSIS = ["duckdb>=1.2.0,<2"]


class TestBruteForceAnalysisOptionalDependency(unittest.TestCase):
    """Locked analysis extra and actionable missing-dependency errors."""

    def test_bruteforce_analysis_extra_range_is_locked(self) -> None:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        self.assertEqual(extras["bruteforce"], _LOCKED_BRUTEFORCE)
        self.assertEqual(extras["bruteforce-analysis"], _LOCKED_BRUTEFORCE_ANALYSIS)

    def test_analysis_extra_is_not_in_algorithms_all(self) -> None:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        for req in extras["algorithms-all"]:
            self.assertNotIn("duckdb", req)
            self.assertNotIn("pyarrow", req)

    def test_importing_brute_force_package_does_not_load_duckdb(self) -> None:
        sys.modules.pop("duckdb", None)

        import core.ioh_core.workflows.brute_force  # noqa: F401
        import core.ioh_core.workflows.brute_force.analysis  # noqa: F401
        import core.ioh_core.workflows.brute_force.deps as deps_mod

        self.assertIs(deps_mod.import_duckdb, import_duckdb)
        self.assertNotIn("duckdb", sys.modules)

    def test_missing_duckdb_error_message(self) -> None:
        from core.ioh_core.workflows.brute_force import deps as deps_mod

        with patch(
            "core.ioh_core.workflows.brute_force.deps.importlib.import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'duckdb'", name="duckdb"
            ),
        ):
            with self.assertRaises(deps_mod.BruteForceDependencyError) as ctx:
                deps_mod.import_duckdb()
        message = str(ctx.exception)
        self.assertEqual(
            message,
            "brute_force analysis requires optional dependency 'duckdb'; "
            "install with:\n"
            'pip install -e ".[bruteforce-analysis]"',
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
                deps_mod.import_duckdb()
        self.assertEqual(str(ctx.exception), "broken native extension")


if __name__ == "__main__":
    unittest.main()
