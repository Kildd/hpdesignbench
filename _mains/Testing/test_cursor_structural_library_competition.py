"""
Phase L9 gate: competition fixture, audit API, and operator CLI.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from core.ioh_core.structural_artifacts.cli import main as cli_main
from core.ioh_core.structural_artifacts.competition import (
    COMPETITION_CARDINALITY,
    audit_competition_fixture,
    audit_library,
    build_competition_fixture,
    competition_fixture_root,
    competition_identity,
    library_provider,
    open_competition_fixture,
    open_library,
    validate_library_root,
    write_provenance,
)
from core.ioh_core.structural_artifacts.library import SUCCESS_MARKER


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(argv)
    return code, out.getvalue(), err.getvalue()


class TestCompetitionFixture(unittest.TestCase):
    def test_open_get_and_identity(self) -> None:
        with open_competition_fixture() as library:
            self.assertEqual(
                library.identity.to_dict(),
                competition_identity().to_dict(),
            )
            self.assertEqual(library.manifest.nrows, COMPETITION_CARDINALITY)
            first = library.get(0)
            failure = library.get(1)
            self.assertTrue(first.mk_cleared)
            self.assertTrue(failure.is_build_failure)
            provider = library_provider(library)
            self.assertEqual(provider.mode, "library")
            self.assertEqual(provider.lookup.get(0).n, first.n)

    def test_fixture_kinds_cover_success_cleared_failure(self) -> None:
        with open_competition_fixture() as library:
            kinds = []
            for sid in range(COMPETITION_CARDINALITY):
                art = library.get(sid)
                if art.is_build_failure:
                    kinds.append("failure")
                elif art.mk_cleared:
                    kinds.append("cleared")
                else:
                    kinds.append("success")
            self.assertIn("success", kinds)
            self.assertIn("cleared", kinds)
            self.assertIn("failure", kinds)

    def test_packaged_root_has_success(self) -> None:
        root = competition_fixture_root()
        self.assertTrue((root / SUCCESS_MARKER).is_file())
        self.assertTrue((root / "manifest.json").is_file())
        self.assertTrue((root / "provenance.json").is_file())


class TestCompetitionAudit(unittest.TestCase):
    def test_fixture_audit_ok(self) -> None:
        report = audit_competition_fixture(seed=7)
        self.assertTrue(report.ok)
        self.assertEqual(report.checked, COMPETITION_CARDINALITY)
        self.assertEqual(report.mismatches, ())

    def test_audit_detects_mismatch(self) -> None:
        with open_competition_fixture() as library:
            wrong = {0: library.get(1)}
            report = audit_library(
                library,
                structural_ids=[0],
                expected_artifacts=wrong,
            )
        self.assertFalse(report.ok)
        self.assertEqual(report.checked, 1)
        self.assertEqual(report.mismatches[0].structural_id, 0)

    def test_validate_library_root(self) -> None:
        summary = validate_library_root(competition_fixture_root())
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["nrows"], COMPETITION_CARDINALITY)


class TestCompetitionRebuildAndProvenance(unittest.TestCase):
    def test_build_to_temp_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fixture"
            built = build_competition_fixture(root, overwrite=False)
            self.assertEqual(built, root.resolve())
            self.assertTrue((root / SUCCESS_MARKER).is_file())
            with open_library(root, expected_identity=competition_identity()) as lib:
                self.assertEqual(lib.manifest.nrows, COMPETITION_CARDINALITY)
            prov_path = write_provenance(root)
            payload = json.loads(prov_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["success_marker"])
            self.assertEqual(payload["format"], "arrow_ipc")
            self.assertIn("identity", payload)
            # Second build without overwrite is a no-op.
            again = build_competition_fixture(root, overwrite=False)
            self.assertEqual(again, root.resolve())


class TestStructuralLibraryCli(unittest.TestCase):
    def test_info_and_audit_and_get(self) -> None:
        code, out, err = _run_cli(["info"])
        self.assertEqual(code, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(payload["nrows"], COMPETITION_CARDINALITY)
        self.assertEqual(payload["identity"]["problem_id"], "fixture")

        code, out, err = _run_cli(["audit"])
        self.assertEqual(code, 0, msg=err)
        report = json.loads(out)
        self.assertTrue(report["ok"])

        code, out, err = _run_cli(["get", "--id", "0"])
        self.assertEqual(code, 0, msg=err)
        got = json.loads(out)
        self.assertEqual(got["structural_id"], 0)
        self.assertFalse(got["is_build_failure"])

    def test_validate_path(self) -> None:
        path = str(competition_fixture_root())
        code, out, err = _run_cli(["validate", "--path", path])
        self.assertEqual(code, 0, msg=err)
        self.assertTrue(json.loads(out)["ok"])

    def test_audit_rejects_arbitrary_path(self) -> None:
        code, _out, err = _run_cli(
            ["audit", "--path", str(competition_fixture_root())]
        )
        self.assertEqual(code, 1)
        self.assertIn("fixture only", err)

    def test_build_fixture_to_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            code, out, err = _run_cli(
                ["build-fixture", "--output", str(dest), "--overwrite"]
            )
            self.assertEqual(code, 0, msg=err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertTrue((dest / SUCCESS_MARKER).is_file())


class TestCompetitionApiSurface(unittest.TestCase):
    def test_package_exports(self) -> None:
        from core.ioh_core import structural_artifacts as sa

        self.assertTrue(callable(sa.open_competition_fixture))
        self.assertTrue(callable(sa.audit_competition_fixture))
        self.assertTrue(callable(sa.validate_library_root))
        self.assertTrue(callable(sa.build_competition_fixture))


if __name__ == "__main__":
    unittest.main()
