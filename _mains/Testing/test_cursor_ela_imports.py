"""
ELA package skeleton: import isolation (no DuckDB, pflacco, or SALib).
"""
from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from core.execution_core.config import SUPPORTED_WORKFLOWS

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
_LOCKED_LA = [
    "pyarrow>=25.0.0,<26",
    "duckdb>=1.2.0,<2",
    "numdifftools>=0.9.40,<1",
]
LA_DIR = REPO_ROOT / "core" / "ioh_core" / "workflows" / "la"
_LA_MODULES = (
    "core.ioh_core.workflows.la",
    "core.ioh_core.workflows.la.adapter",
    "core.ioh_core.workflows.la.aggregate",
    "core.ioh_core.workflows.la.basins",
    "core.ioh_core.workflows.la.catalogue_l1",
    "core.ioh_core.workflows.la.catalogue_table",
    "core.ioh_core.workflows.la.cli",
    "core.ioh_core.workflows.la.colon",
    "core.ioh_core.workflows.la.constrained",
    "core.ioh_core.workflows.la.constraints",
    "core.ioh_core.workflows.la.deb",
    "core.ioh_core.workflows.la.deps",
    "core.ioh_core.workflows.la.lon",
    "core.ioh_core.workflows.la.lon_landscape",
    "core.ioh_core.workflows.la.models",
    "core.ioh_core.workflows.la.orchestrate",
    "core.ioh_core.workflows.la.plateau_graph",
    "core.ioh_core.workflows.la.preprocess",
    "core.ioh_core.workflows.la.rfb",
    "core.ioh_core.workflows.la.signatures",
    "core.ioh_core.workflows.la.topology",
)
_FORBIDDEN_ALWAYS = (
    "duckdb",
    "pflacco",
    "pyarrow",
    "numdifftools",
    "SALib",
    "core.ioh_core.algorithms.scipy.sobol",
    "core.ioh_core.workflows.sensitivity_analysis",
    "core.ioh_core.workflows.brute_force.analysis.constraint_decision_tree",
)


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class TestElaPackageSkeleton(unittest.TestCase):
    """Empty ELA package must not pull DuckDB, pflacco, or SALib."""

    def test_package_modules_exist(self) -> None:
        expected = {
            "__init__.py",
            "adapter.py",
            "aggregate.py",
            "basins.py",
            "catalogue_l1.py",
            "catalogue_table.py",
            "cli.py",
            "colon.py",
            "constrained.py",
            "constraints.py",
            "deb.py",
            "deps.py",
            "lon.py",
            "lon_landscape.py",
            "orchestrate.py",
            "preprocess.py",
            "plateau_graph.py",
            "rfb.py",
            "signatures.py",
            "topology.py",
        }
        found = {path.name for path in LA_DIR.glob("*.py")}
        self.assertTrue(expected.issubset(found), msg=sorted(expected - found))

    def test_source_does_not_import_forbidden_libraries(self) -> None:
        for path in sorted(LA_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = _imported_names(tree)
            for name in imported:
                for root in _FORBIDDEN_ALWAYS:
                    self.assertFalse(
                        name == root or name.startswith(root + "."),
                        msg=f"{path.name} imports {name}",
                    )

    def test_importing_package_does_not_load_forbidden_libraries(self) -> None:
        script = """
import sys
import core.ioh_core.workflows.la
loaded = [name for name in ("duckdb", "pflacco", "pyarrow", "numdifftools", "SALib")
          if name in sys.modules]
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

    def test_importing_all_modules_does_not_load_forbidden_libraries(self) -> None:
        imports = "\n".join(f"import {name}" for name in _LA_MODULES)
        forbidden = ", ".join(repr(name) for name in _FORBIDDEN_ALWAYS)
        script = f"""
import sys
{imports}
loaded = [name for name in ({forbidden},) if name in sys.modules]
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

    def test_importing_deps_does_not_load_pyarrow_or_duckdb(self) -> None:
        script = """
import sys
sys.modules.pop("pyarrow", None)
sys.modules.pop("duckdb", None)
sys.modules.pop("numdifftools", None)
sys.modules.pop("pflacco", None)
import core.ioh_core.workflows.la.deps as deps
assert deps.import_pyarrow is not None
assert deps.import_duckdb is not None
assert deps.import_numdifftools is not None
assert deps.import_pflacco is not None
assert "pyarrow" not in sys.modules
assert "duckdb" not in sys.modules
assert "numdifftools" not in sys.modules
assert "pflacco" not in sys.modules
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

    def test_missing_pyarrow_error_message(self) -> None:
        from core.ioh_core.workflows.la import deps as deps_mod

        with patch(
            "core.ioh_core.workflows.la.deps.importlib.import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'pyarrow'", name="pyarrow"
            ),
        ):
            with self.assertRaises(deps_mod.ElaDependencyError) as ctx:
                deps_mod.import_pyarrow()
        message = str(ctx.exception)
        self.assertEqual(
            message,
            "landscape analysis requires optional dependency 'pyarrow'; "
            "install with:\n"
            'pip install -e ".[la]"',
        )
        self.assertIsInstance(ctx.exception.__cause__, ModuleNotFoundError)
        self.assertIsInstance(ctx.exception, ImportError)

    def test_missing_duckdb_error_message(self) -> None:
        from core.ioh_core.workflows.la import deps as deps_mod

        with patch(
            "core.ioh_core.workflows.la.deps.importlib.import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'duckdb'", name="duckdb"
            ),
        ):
            with self.assertRaises(deps_mod.ElaDependencyError) as ctx:
                deps_mod.import_duckdb()
        message = str(ctx.exception)
        self.assertEqual(
            message,
            "landscape analysis requires optional dependency 'duckdb'; "
            "install with:\n"
            'pip install -e ".[la]"',
        )
        self.assertIsInstance(ctx.exception.__cause__, ModuleNotFoundError)
        self.assertIsInstance(ctx.exception, ImportError)

    def test_missing_numdifftools_error_message(self) -> None:
        from core.ioh_core.workflows.la import deps as deps_mod

        with patch(
            "core.ioh_core.workflows.la.deps.importlib.import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'numdifftools'", name="numdifftools"
            ),
        ):
            with self.assertRaises(deps_mod.ElaDependencyError) as ctx:
                deps_mod.import_numdifftools()
        message = str(ctx.exception)
        self.assertEqual(
            message,
            "landscape analysis requires optional dependency 'numdifftools'; "
            "install with:\n"
            'pip install -e ".[la]"',
        )
        self.assertIsInstance(ctx.exception.__cause__, ModuleNotFoundError)
        self.assertIsInstance(ctx.exception, ImportError)

    def test_missing_pflacco_error_message(self) -> None:
        from core.ioh_core.workflows.la import deps as deps_mod

        with patch(
            "core.ioh_core.workflows.la.deps.importlib.import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'pflacco'", name="pflacco"
            ),
        ):
            with self.assertRaises(deps_mod.ElaDependencyError) as ctx:
                deps_mod.import_pflacco()
        message = str(ctx.exception)
        self.assertIn("pflacco==1.2.2", message)
        self.assertIn("--no-deps", message)
        self.assertIn('pip install -e ".[la]"', message)
        self.assertIn("numpy~=1.24", message)
        self.assertIsInstance(ctx.exception.__cause__, ModuleNotFoundError)
        self.assertIsInstance(ctx.exception, ImportError)

    def test_init_does_not_import_siblings(self) -> None:
        init_path = LA_DIR / "__init__.py"
        tree = ast.parse(
            init_path.read_text(encoding="utf-8"),
            filename=str(init_path),
        )
        imported = _imported_names(tree)
        self.assertEqual(imported, set())

    def test_ela_extra_range_is_locked(self) -> None:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        self.assertEqual(extras["la"], _LOCKED_LA)
        self.assertEqual(extras["la"][0], extras["bruteforce"][0])
        self.assertEqual(extras["la"][1], extras["bruteforce-analysis"][0])
        self.assertTrue(extras["la"][2].startswith("numdifftools"))
        for req in extras["la"]:
            self.assertNotIn("pflacco", req)

    def test_ela_extra_is_not_in_algorithms_all(self) -> None:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        for req in extras["algorithms-all"]:
            self.assertNotIn("pyarrow", req)
            self.assertNotIn("duckdb", req)
            self.assertNotIn("pflacco", req)
            self.assertNotIn("numdifftools", req)

    def test_operator_docs_prefer_la_names(self) -> None:
        ela_md = (REPO_ROOT / "_docs" / "ela.md").read_text(encoding="utf-8")
        self.assertIn("hdb-la-analyze", ela_md)
        self.assertIn("core.ioh_core.workflows.la", ela_md)
        self.assertIn("results/la", ela_md)
        self.assertNotIn("hdb-ela-analyze", ela_md)

    def test_pyproject_declares_hdb_la_analyze_script(self) -> None:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        scripts = data["project"]["scripts"]
        self.assertNotIn("hdb-ela-analyze", scripts)
        self.assertEqual(
            scripts.get("hdb-la-analyze"),
            "core.ioh_core.workflows.la.cli:main",
        )

    def test_ela_is_not_a_supported_experiment_workflow(self) -> None:
        self.assertNotIn("ela", SUPPORTED_WORKFLOWS)
        self.assertNotIn("la", SUPPORTED_WORKFLOWS)

    def test_old_ela_package_directory_is_absent(self) -> None:
        leftover = LA_DIR.parent / "ela"
        self.assertFalse(
            leftover.exists(),
            f"leftover package {leftover} must be removed; library is workflows.la",
        )

    def test_la_package_init_does_not_import_siblings(self) -> None:
        init_path = (
            REPO_ROOT / "core" / "ioh_core" / "workflows" / "la" / "__init__.py"
        )
        tree = ast.parse(
            init_path.read_text(encoding="utf-8"),
            filename=str(init_path),
        )
        self.assertEqual(_imported_names(tree), set())


if __name__ == "__main__":
    unittest.main()
