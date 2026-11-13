"""
CLI for sealed structural artifact libraries.

Console script: ``hdb-structural-library``
Module: ``python -m core.ioh_core.structural_artifacts.cli``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from core.ioh_core.structural_artifacts.competition import (
    CompetitionLibraryError,
    audit_competition_fixture,
    build_competition_fixture,
    competition_fixture_root,
    competition_provenance,
    open_competition_fixture,
    open_library,
    validate_library_root,
    write_provenance,
)
from core.ioh_core.structural_artifacts.models import StructuralLibraryError
from core.ioh_core.workflows.brute_force.deps import BruteForceDependencyError


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # noqa: D102
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="hdb-structural-library",
        description=(
            "Open, validate, audit, and package sealed structural artifact "
            "libraries (Arrow IPC)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="Print sealed library identity summary")
    info.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Library root (default: packaged competition fixture)",
    )

    validate = sub.add_parser(
        "validate",
        help="Open the library and verify part hashes / first-last lookup",
    )
    validate.add_argument("--path", type=Path, required=True)

    audit = sub.add_parser(
        "audit",
        help="Audit competition fixture rows (or require --expected-dir later)",
    )
    audit.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Library root (default: competition fixture self-audit)",
    )
    audit.add_argument("--sample-size", type=int, default=None)
    audit.add_argument("--seed", type=int, default=0)
    audit.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated structural IDs (fixture self-audit when omitted)",
    )

    build = sub.add_parser(
        "build-fixture",
        help="Build/overwrite the redistributable competition fixture",
    )
    build.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination root (default: packaged competition_data path)",
    )
    build.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing sealed fixture",
    )

    prov = sub.add_parser(
        "provenance",
        help="Write provenance.json for a sealed library",
    )
    prov.add_argument("--path", type=Path, required=True)
    prov.add_argument("--output", type=Path, default=None)

    get_cmd = sub.add_parser("get", help="Fetch one structural ID as a short JSON summary")
    get_cmd.add_argument("--path", type=Path, default=None)
    get_cmd.add_argument("--id", type=int, required=True, dest="structural_id")

    compat = sub.add_parser(
        "set-compatible-problems",
        help=(
            "Update identity.compatible_problem_ids on an existing sealed "
            "library (no Arrow rewrite)"
        ),
    )
    compat.add_argument("--path", type=Path, required=True)
    compat.add_argument(
        "--problem-ids",
        required=True,
        help="Comma-separated consumer problem ids (build id is always kept)",
    )

    return parser


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "build-fixture":
            root = build_competition_fixture(
                args.output, overwrite=bool(args.overwrite)
            )
            _print_json({"ok": True, "library_root": str(root)})
            return 0
        if args.command == "info":
            path = args.path or competition_fixture_root()
            if args.path is None and not (path / "_SUCCESS").is_file():
                build_competition_fixture(path, overwrite=False)
            with open_library(path) as library:
                _print_json(
                    {
                        "library_root": str(Path(path).resolve()),
                        "identity": library.identity.to_dict(),
                        "nrows": library.manifest.nrows,
                        "rows_per_shard": library.manifest.rows_per_shard,
                        "schema_version": library.manifest.schema_version,
                        "codec_version": library.manifest.codec_version,
                    }
                )
            return 0
        if args.command == "validate":
            _print_json(validate_library_root(args.path))
            return 0
        if args.command == "provenance":
            path = write_provenance(args.path, args.output)
            _print_json({"ok": True, "path": str(path), **competition_provenance(args.path)})
            return 0
        if args.command == "get":
            path = args.path
            if path is None:
                with open_competition_fixture() as library:
                    art = library.get(int(args.structural_id))
            else:
                with open_library(path) as library:
                    art = library.get(int(args.structural_id))
            _print_json(
                {
                    "structural_id": int(args.structural_id),
                    "snapshot_hash": art.snapshot_hash,
                    "mk_cleared": bool(art.mk_cleared),
                    "is_build_failure": bool(art.is_build_failure),
                    "n": art.n,
                }
            )
            return 0
        if args.command == "set-compatible-problems":
            from core.ioh_core.structural_artifacts.annotate import (
                set_compatible_problem_ids,
            )

            ids = [
                part.strip()
                for part in str(args.problem_ids).split(",")
                if part.strip()
            ]
            identity = set_compatible_problem_ids(args.path, ids)
            _print_json(
                {
                    "ok": True,
                    "library_root": str(Path(args.path).resolve()),
                    "problem_id": identity.problem_id,
                    "compatible_problem_ids": list(identity.compatible_problem_ids),
                }
            )
            return 0
        if args.command == "audit":
            if args.path is not None:
                raise CompetitionLibraryError(
                    "CLI audit currently supports the packaged competition "
                    "fixture only (omit --path); for production libraries use "
                    "validate plus an offline recompute audit script"
                )
            report = audit_competition_fixture(seed=int(args.seed))
            _print_json(report.to_dict())
            return 0 if report.ok else 2
        parser.error(f"unknown command {args.command!r}")
        return 1
    except (
        CompetitionLibraryError,
        StructuralLibraryError,
        BruteForceDependencyError,
        OSError,
        ValueError,
    ) as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
