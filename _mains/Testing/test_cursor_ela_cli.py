"""
ELA CLI skeleton: argparse, TOML overlay, dataset/ output guard, keep list.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from core.ioh_core.constraint_selection import CONSTRAINT_SUBSET_WARNING
from core.ioh_core.workflows.la.cli import (
    ELA_COMMANDS,
    EXHAUSTIVE_FAMILIES,
    FAMILY_TOKENS,
    ILS_COLON_FAMILIES,
    ILS_LON_FAMILIES,
    WIRED_EXHAUSTIVE_FAMILIES,
    build_parser,
    default_bf_analysis_output_dir,
    default_ela_output_dir,
    default_la_output_dir,
    main,
    resolve_ela_families,
)
from core.ioh_core.workflows.la.constrained import FEATURE_FSR
from core.ioh_core.workflows.la.lon import FEATURE_BASIN_F_SHARE_MIN, FEATURE_N_NODES
from core.ioh_core.workflows.la.deps import ElaDependencyError, import_pyarrow
from core.ioh_core.workflows.la.models import ElaError


REPO_ROOT = Path(__file__).resolve().parents[2]
_SEALED = [
    "A_bending_capacity",
    "Z1_nt_dt_combination_capacity",
]
_VARS = ("geom_t_mm", "mat_conc_id")
_LOWER = (0, 3)
_UPPER = (1, 4)


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _json_from_stderr(stderr: str) -> dict:
    lines = [line for line in stderr.splitlines() if line.lstrip().startswith("{")]
    if not lines:
        raise AssertionError(f"no JSON object in stderr: {stderr!r}")
    return json.loads(lines[-1])


def _rfbx_cli_knobs() -> list[str]:
    return [
        "--n-walks",
        "2",
        "--n-transitions",
        "4",
        "--seed-base",
        "11",
        "--start-policy",
        "uniform_catalogue",
    ]


def _write_schema(experiment_dir: Path, names: list[str] | None = None) -> None:
    dataset = Path(experiment_dir) / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "schema.json").write_text(
        json.dumps({"constraint_names": list(names or _SEALED)}),
        encoding="utf-8",
    )


def _write_complete_bf_fixture(
    experiment_dir: Path,
    *,
    drop_last: bool = False,
    include_feasible: bool = True,
    null_feasible: bool = False,
) -> None:
    """Tiny 2×2 catalogue: sealed bounds product 4, optional incomplete row."""
    dataset = Path(experiment_dir) / "dataset"
    parts = dataset / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    (dataset / "schema.json").write_text(
        json.dumps(
            {
                "var_names": list(_VARS),
                "constraint_names": list(_SEALED),
                "lower_bounds": list(_LOWER),
                "upper_bounds": list(_UPPER),
                "problem_id": "L6750_B2",
            }
        ),
        encoding="utf-8",
    )
    (dataset / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    rows = [
        {
            "idx__geom_t_mm": 0,
            "idx__mat_conc_id": 3,
            "y": 10.0,
            "y_p": 11.0,
            "feasible": True,
            "util__A_bending_capacity": 0.5,
            "util__Z1_nt_dt_combination_capacity": 0.5,
        },
        {
            "idx__geom_t_mm": 0,
            "idx__mat_conc_id": 4,
            "y": 20.0,
            "y_p": 21.0,
            "feasible": False,
            "util__A_bending_capacity": 0.5,
            "util__Z1_nt_dt_combination_capacity": 1.2,
        },
        {
            "idx__geom_t_mm": 1,
            "idx__mat_conc_id": 3,
            "y": 20.0,
            "y_p": 21.0,
            "feasible": False,
            "util__A_bending_capacity": 1.5,
            "util__Z1_nt_dt_combination_capacity": 0.5,
        },
        {
            "idx__geom_t_mm": 1,
            "idx__mat_conc_id": 4,
            "y": 10.0,
            "y_p": 11.0,
            "feasible": True,
            "util__A_bending_capacity": 0.8,
            "util__Z1_nt_dt_combination_capacity": 0.8,
        },
    ]
    if not include_feasible:
        for row in rows:
            row.pop("feasible")
    elif null_feasible:
        rows[0]["feasible"] = None
    if drop_last:
        rows = rows[:-1]
    pa = import_pyarrow()
    parquet = __import__("importlib").import_module("pyarrow.parquet")
    parquet.write_table(pa.Table.from_pylist(rows), parts / "shard-000.parquet")


class TestElaCliSkeleton(unittest.TestCase):
    """Parser, overlay, and write-guard without running families."""

    def test_parser_prog_and_subcommands(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.prog, "hdb-la-analyze")
        completed = subprocess.run(
            [sys.executable, "-m", "core.ioh_core.workflows.la", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("constraints", completed.stdout)
        self.assertIn("run", completed.stdout)
        self.assertIn("aggregate", completed.stdout)
        for name in ELA_COMMANDS:
            self.assertIn(name, completed.stdout)
        sub_help = subprocess.run(
            [
                sys.executable,
                "-m",
                "core.ioh_core.workflows.la",
                "constraints",
                "--help",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(sub_help.returncode, 0)
        self.assertIn("--constraint-names", sub_help.stdout)
        self.assertIn("--source-type", sub_help.stdout)
        self.assertIn("--taus", sub_help.stdout)
        self.assertIn("--n-walks", sub_help.stdout)
        self.assertIn("--n-transitions", sub_help.stdout)
        self.assertIn("--seed-base", sub_help.stdout)
        self.assertIn("--start-policy", sub_help.stdout)
        self.assertIn("--write-attractors", sub_help.stdout)
        self.assertIn("--radii", sub_help.stdout)
        run_help = subprocess.run(
            [
                sys.executable,
                "-m",
                "core.ioh_core.workflows.la",
                "run",
                "--help",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(run_help.returncode, 0)
        self.assertIn("--families", run_help.stdout)
        aggregate_help = subprocess.run(
            [
                sys.executable,
                "-m",
                "core.ioh_core.workflows.la",
                "aggregate",
                "--help",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(aggregate_help.returncode, 0)
        self.assertIn("--families", aggregate_help.stdout)
        self.assertIn("--allow-multiple-groups", aggregate_help.stdout)
        self.assertIn("results/la", aggregate_help.stdout)
        missing = subprocess.run(
            [sys.executable, "-m", "core.ioh_core.workflows.ela", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(missing.returncode, 0)

    def test_ela_is_not_a_valid_analysis_root(self) -> None:
        with self.assertRaises(ElaError) as ctx:
            build_parser(analysis_root="ela")
        self.assertIn("analysis output root", str(ctx.exception))

    def test_la_parser_prog_and_module_help(self) -> None:
        parser = build_parser(prog="hdb-la-analyze", analysis_root="la")
        self.assertEqual(parser.prog, "hdb-la-analyze")
        completed = subprocess.run(
            [sys.executable, "-m", "core.ioh_core.workflows.la", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("hdb-la-analyze", completed.stdout)
        self.assertIn("constraints", completed.stdout)
        self.assertIn("lon", completed.stdout)
        sub_help = subprocess.run(
            [
                sys.executable,
                "-m",
                "core.ioh_core.workflows.la",
                "constraints",
                "--help",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(sub_help.returncode, 0, msg=sub_help.stderr)
        self.assertIn("results/la", sub_help.stdout)

    def test_importing_la_cli_does_not_load_duckdb_or_pflacco(self) -> None:
        script = """
import sys
import core.ioh_core.workflows.la.cli as cli
loaded = [name for name in ("duckdb", "pflacco", "pyarrow", "SALib")
          if name in sys.modules]
if loaded:
    raise SystemExit("loaded forbidden modules: " + ", ".join(loaded))
assert cli.main is not None
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

    def test_missing_subcommand_exits_1(self) -> None:
        code, _stdout, stderr = _run_main([])
        self.assertEqual(code, 1)
        self.assertIn("error", stderr.lower())

    def test_valid_parse_exits_2_without_writing(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            code, stdout, stderr = _run_main(
                [
                    "constraints",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                ]
            )
            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            payload = _json_from_stderr(stderr)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["command"], "constraints")
            self.assertEqual(payload["families"], ["constraints"])
            self.assertEqual(payload["source_type"], "sampling")
            self.assertIn("dataset/_SUCCESS", payload["error"])
            self.assertEqual(payload["experiment_dir"], str(experiment_dir))
            expected = default_la_output_dir(
                experiment_dir, families=["constraints"], source_type="sampling"
            )
            self.assertEqual(payload["output_dir"], str(expected))
            self.assertFalse(payload["constraint_set_is_subset"])
            self.assertIsNone(payload["constraint_names_sealed"])
            self.assertFalse(expected.exists())
            self.assertFalse((experiment_dir / "dataset").exists())

    def test_output_dir_under_dataset_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            bad = experiment_dir / "dataset" / "ela"
            code, _stdout, stderr = _run_main(
                [
                    "topology",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--output-dir",
                    str(bad),
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertFalse(payload["ok"])
            self.assertIn("dataset/", payload["error"])

    def test_cli_flags_override_toml(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            toml_experiment = root / "from_toml"
            toml_experiment.mkdir()
            cli_experiment = root / "from_cli"
            cli_experiment.mkdir()
            cli_output = cli_experiment / "out"
            config = root / "ela.toml"
            config.write_text(
                "\n".join(
                    [
                        "[la]",
                        'experiment_dir = "from_toml"',
                        'output_dir = "from_toml/ignored"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            code, _stdout, stderr = _run_main(
                [
                    "constraints",
                    "--config",
                    str(config),
                    "--experiment-dir",
                    str(cli_experiment),
                    "--output-dir",
                    str(cli_output),
                    "--source-type",
                    "sampling",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertEqual(payload["experiment_dir"], str(cli_experiment))
            self.assertEqual(payload["output_dir"], str(cli_output))
            self.assertIn("dataset/_SUCCESS", payload["error"])

    def test_toml_supplies_experiment_dir_when_cli_omits_it(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = root / "sealed"
            experiment_dir.mkdir()
            config = root / "ela.toml"
            config.write_text(
                '[la]\nexperiment_dir = "sealed"\nsource_type = "sampling"\n',
                encoding="utf-8",
            )
            code, _stdout, stderr = _run_main(
                ["constraints", "--config", str(config)]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertEqual(Path(payload["experiment_dir"]), experiment_dir)
            self.assertEqual(
                Path(payload["output_dir"]),
                default_la_output_dir(
                    experiment_dir,
                    families=["constraints"],
                    source_type="sampling",
                ),
            )

    def test_unknown_toml_key_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "ela.toml"
            config.write_text(
                '[la]\nexperiment_dir = "x"\nunknown_knob = 1\n',
                encoding="utf-8",
            )
            code, _stdout, stderr = _run_main(
                ["run", "--config", str(config)]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("unknown ELA config keys", payload["error"])

    def test_missing_experiment_dir_exits_1(self) -> None:
        code, _stdout, stderr = _run_main(["constraints"])
        self.assertEqual(code, 1)
        payload = _json_from_stderr(stderr)
        self.assertIn("experiment_dir is required", payload["error"])

    def test_missing_config_file_exits_1(self) -> None:
        code, _stdout, stderr = _run_main(
            ["rfbx", "--config", str(Path("missing-ela.toml"))]
        )
        self.assertEqual(code, 1)
        payload = _json_from_stderr(stderr)
        self.assertIn("not found", payload["error"])

    def test_importing_cli_does_not_load_duckdb_or_pflacco(self) -> None:
        script = """
import sys
import core.ioh_core.workflows.la.cli as cli
loaded = [name for name in ("duckdb", "pflacco", "pyarrow", "SALib")
          if name in sys.modules]
if loaded:
    raise SystemExit("loaded forbidden modules: " + ", ".join(loaded))
assert cli.build_parser().prog == "hdb-la-analyze"
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

    def test_schema_default_keeps_z_without_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_schema(experiment_dir)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                code, _stdout, stderr = _run_main(
                    ["constraints", "--experiment-dir", str(experiment_dir),
                     "--source-type", "sampling"]
                )
            self.assertEqual(code, 1)
            runtime = [
                item for item in caught if issubclass(item.category, RuntimeWarning)
            ]
            self.assertEqual(runtime, [])
            payload = _json_from_stderr(stderr)
            self.assertIn("dataset/_SUCCESS", payload["error"])
            self.assertEqual(payload["constraint_names_sealed"], _SEALED)
            self.assertEqual(payload["constraint_names_selected"], _SEALED)
            self.assertFalse(payload["constraint_set_is_subset"])

    def test_keep_list_subset_requires_distinct_output_dir_and_warns(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_schema(experiment_dir)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                code, _stdout, stderr = _run_main(
                    [
                        "constraints",
                        "--experiment-dir",
                        str(experiment_dir),
                        "--source-type",
                        "brute_force",
                        "--constraint-names",
                        "A_bending_capacity",
                    ]
                )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("--output-dir", payload["error"])
            runtime = [
                item for item in caught if issubclass(item.category, RuntimeWarning)
            ]
            self.assertEqual(len(runtime), 1)
            self.assertEqual(str(runtime[0].message), CONSTRAINT_SUBSET_WARNING)

    def test_keep_list_with_distinct_output_unpublished_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_schema(experiment_dir)
            out = Path(tmp) / "ela_subset"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                code, _stdout, stderr = _run_main(
                    [
                        "constraints",
                        "--experiment-dir",
                        str(experiment_dir),
                        "--source-type",
                        "brute_force",
                        "--constraint-names",
                        "A_bending_capacity",
                        "--output-dir",
                        str(out),
                    ]
                )
            self.assertEqual(code, 1)
            runtime = [
                item for item in caught if issubclass(item.category, RuntimeWarning)
            ]
            self.assertEqual(len(runtime), 1)
            payload = _json_from_stderr(stderr)
            self.assertEqual(
                payload["constraint_names_selected"],
                ["A_bending_capacity"],
            )
            self.assertTrue(payload["constraint_set_is_subset"])
            self.assertEqual(payload["output_dir"], str(out))
            self.assertEqual(payload["families"], ["constraints"])
            self.assertEqual(payload["source_type"], "brute_force")
            self.assertIn("dataset/_SUCCESS", payload["error"])
            self.assertFalse(out.exists())

    def test_keep_list_refuses_canonical_bf_analysis_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_schema(experiment_dir)
            canonical = default_bf_analysis_output_dir(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "constraints",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--constraint-names",
                    "A_bending_capacity",
                    "--output-dir",
                    str(canonical),
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("canonical", payload["error"])

    def test_keep_list_refuses_canonical_ela_output_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_schema(experiment_dir)
            canonical = default_ela_output_dir(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "constraints",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--constraint-names",
                    "A_bending_capacity",
                    "--output-dir",
                    str(canonical),
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("canonical", payload["error"])

    def test_keep_list_refuses_canonical_la_output_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_schema(experiment_dir)
            canonical = default_la_output_dir(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "constraints",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--constraint-names",
                    "A_bending_capacity",
                    "--output-dir",
                    str(canonical),
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("canonical", payload["error"])

    def test_keep_list_without_schema_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            code, _stdout, stderr = _run_main(
                [
                    "constraints",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--constraint-names",
                    "A_bending_capacity",
                    "--output-dir",
                    str(Path(tmp) / "out"),
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("schema.json", payload["error"])

    def test_toml_keep_list_subset_requires_output_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_schema(experiment_dir)
            config = Path(tmp) / "ela.toml"
            config.write_text(
                "\n".join(
                    [
                        "[la]",
                        'constraint_names = ["A_bending_capacity"]',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            code, _stdout, stderr = _run_main(
                [
                    "constraints",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                    "--config",
                    str(config),
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("--output-dir", payload["error"])

    def test_cli_constraint_names_override_toml(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_schema(experiment_dir)
            config = Path(tmp) / "ela.toml"
            config.write_text(
                "\n".join(
                    [
                        "[la]",
                        'constraint_names = ["A_bending_capacity"]',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "out"
            code, _stdout, stderr = _run_main(
                [
                    "constraints",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                    "--config",
                    str(config),
                    "--constraint-names",
                    *_SEALED,
                    "--output-dir",
                    str(out),
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertEqual(payload["constraint_names_selected"], _SEALED)
            self.assertFalse(payload["constraint_set_is_subset"])
            self.assertIn("dataset/_SUCCESS", payload["error"])


class TestElaFamilySelection(unittest.TestCase):
    """CLI family tokens, no-all, and source-type gates (Phase 18.1)."""

    def test_family_tokens_are_hyphen_cli_names(self) -> None:
        self.assertEqual(
            FAMILY_TOKENS,
            tuple(name for name in ELA_COMMANDS if name not in {"run", "aggregate"}),
        )
        self.assertNotIn("all", FAMILY_TOKENS)
        self.assertNotIn("run", FAMILY_TOKENS)
        self.assertNotIn("aggregate", FAMILY_TOKENS)
        self.assertIn("lon", FAMILY_TOKENS)
        self.assertIn("colon", FAMILY_TOKENS)
        self.assertNotIn("meta-model", FAMILY_TOKENS)
        self.assertNotIn("ils-lon", FAMILY_TOKENS)
        self.assertNotIn("ils-colon", FAMILY_TOKENS)
        self.assertNotIn("ils-lon", EXHAUSTIVE_FAMILIES)
        self.assertNotIn("ils-colon", EXHAUSTIVE_FAMILIES)
        self.assertNotIn("ils-lon", WIRED_EXHAUSTIVE_FAMILIES)
        self.assertNotIn("ils-colon", WIRED_EXHAUSTIVE_FAMILIES)
        self.assertEqual(ILS_LON_FAMILIES, frozenset({"ils-lon"}))
        self.assertEqual(ILS_COLON_FAMILIES, frozenset({"ils-colon"}))
        self.assertNotIn("mda", FAMILY_TOKENS)
        self.assertIn("aggregate", ELA_COMMANDS)

    def test_all_token_is_rejected(self) -> None:
        with self.assertRaises(ElaError) as caught:
            resolve_ela_families(
                command="run",
                cli_families=["all"],
                toml_families=None,
            )
        self.assertIn("all", str(caught.exception))
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            code, _stdout, stderr = _run_main(
                [
                    "run",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                    "--families",
                    "all",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("all", payload["error"])

    def test_module_id_and_duplicate_tokens_are_rejected(self) -> None:
        with self.assertRaises(ElaError):
            resolve_ela_families(
                command="run",
                cli_families=["ela_meta"],
                toml_families=None,
            )
        with self.assertRaises(ElaError) as caught:
            resolve_ela_families(
                command="run",
                cli_families=["constraints", "constraints"],
                toml_families=None,
            )
        self.assertIn("duplicate", str(caught.exception))

    def test_aggregate_may_omit_families(self) -> None:
        self.assertEqual(
            resolve_ela_families(
                command="aggregate",
                cli_families=None,
                toml_families=None,
            ),
            (),
        )
        self.assertEqual(
            resolve_ela_families(
                command="aggregate",
                cli_families=["constraints"],
                toml_families=None,
            ),
            ("constraints",),
        )

    def test_aggregate_without_parquet_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            code, _stdout, stderr = _run_main(
                [
                    "aggregate",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertFalse(payload["ok"])
            self.assertIn("features.parquet", payload["error"])

    def test_run_without_families_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            code, _stdout, stderr = _run_main(
                [
                    "run",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("no default all", payload["error"])

    def test_run_empty_toml_families_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = root / "exp"
            experiment_dir.mkdir()
            config = root / "ela.toml"
            config.write_text(
                "[la]\nfamilies = []\n",
                encoding="utf-8",
            )
            code, _stdout, stderr = _run_main(
                [
                    "run",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                    "--config",
                    str(config),
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("no default all", payload["error"])

    def test_subcommand_ignores_toml_families(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = root / "exp"
            experiment_dir.mkdir()
            config = root / "ela.toml"
            config.write_text(
                "\n".join(
                    [
                        "[la]",
                        'families = ["constraints", "topology"]',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            code, _stdout, stderr = _run_main(
                [
                    "constraints",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                    "--config",
                    str(config),
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertEqual(payload["families"], ["constraints"])
            self.assertEqual(payload["source_type"], "sampling")
            self.assertIn("dataset/_SUCCESS", payload["error"])

    def test_cli_families_override_toml_on_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = root / "exp"
            experiment_dir.mkdir()
            config = root / "ela.toml"
            config.write_text(
                '[la]\nfamilies = ["lon"]\n',
                encoding="utf-8",
            )
            code, _stdout, stderr = _run_main(
                [
                    "run",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                    "--config",
                    str(config),
                    "--families",
                    "constraints",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertEqual(payload["families"], ["constraints"])
            self.assertEqual(payload["source_type"], "sampling")
            self.assertIn("dataset/_SUCCESS", payload["error"])

    def test_constraints_on_sampling_passes_source_gate(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            code, _stdout, stderr = _run_main(
                [
                    "constraints",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertNotIn("exhaustive", payload["error"])
            self.assertIn("dataset/_SUCCESS", payload["error"])
            self.assertEqual(payload["families"], ["constraints"])
            self.assertEqual(payload["source_type"], "sampling")

    def test_run_exhaustive_on_unpublished_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            code, _stdout, stderr = _run_main(
                [
                    "run",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--families",
                    "constraints",
                    "topology",
                    "rfbx",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertEqual(
                payload["families"],
                ["constraints", "topology", "rfbx"],
            )
            self.assertEqual(payload["source_type"], "brute_force")
            self.assertIn("dataset/_SUCCESS", payload["error"])
            self.assertNotIn("not implemented yet", payload["error"])

    def test_exhaustive_stub_ignores_missing_pflacco(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            with patch(
                "core.ioh_core.workflows.la.deps.import_pflacco",
                side_effect=ElaDependencyError("missing pflacco"),
            ):
                code, _stdout, stderr = _run_main(
                    [
                        "constraints",
                        "--experiment-dir",
                        str(experiment_dir),
                        "--source-type",
                        "brute_force",
                    ]
                )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertEqual(payload["families"], ["constraints"])
            self.assertEqual(payload["source_type"], "brute_force")
            self.assertIn("dataset/_SUCCESS", payload["error"])
            self.assertNotIn("not implemented yet", payload["error"])

    def test_constraints_on_complete_fixture_writes_parquet(self) -> None:
        self.assertEqual(WIRED_EXHAUSTIVE_FAMILIES, EXHAUSTIVE_FAMILIES)
        self.assertEqual(
            WIRED_EXHAUSTIVE_FAMILIES,
            frozenset({"constraints", "topology", "rfbx", "basins", "lon", "colon"}),
        )
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "constraints",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            payload = _json_from_stderr(stderr)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["written"], ["constraints"])
            self.assertEqual(payload["skipped"], [])
            family_dir = default_la_output_dir(experiment_dir) / "constraints"
            parquet = family_dir / "features.parquet"
            self.assertTrue((family_dir / "_SUCCESS").is_file())
            self.assertTrue(parquet.is_file())
            import_pyarrow()
            parquet_mod = __import__("importlib").import_module("pyarrow.parquet")
            table = parquet_mod.read_table(parquet)
            names = table.column("feature_name").to_pylist()
            self.assertIn(FEATURE_FSR, names)
            families = set(table.column("family").to_pylist())
            self.assertEqual(families, {"constrained"})

    def test_la_constraints_writes_under_la_output_not_ela(self) -> None:
        from core.ioh_core.workflows.la.cli import main as la_main

        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = la_main(
                    [
                        "constraints",
                        "--experiment-dir",
                        str(experiment_dir),
                        "--source-type",
                        "brute_force",
                    ]
                )
            self.assertEqual(code, 0, msg=stderr.getvalue())
            la_dir = default_la_output_dir(experiment_dir) / "constraints"
            ela_dir = default_ela_output_dir(experiment_dir) / "constraints"
            self.assertTrue((la_dir / "_SUCCESS").is_file())
            self.assertTrue((la_dir / "features.parquet").is_file())
            self.assertFalse(ela_dir.exists())

    def test_constraints_incomplete_catalogue_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir, drop_last=True)
            code, _stdout, stderr = _run_main(
                [
                    "constraints",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("cardinality", payload["error"])
            family_dir = default_la_output_dir(experiment_dir) / "constraints"
            self.assertFalse((family_dir / "_SUCCESS").exists())

    def test_constraints_resume_skips_existing_success(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            argv = [
                "constraints",
                "--experiment-dir",
                str(experiment_dir),
                "--source-type",
                "brute_force",
            ]
            first, _stdout, stderr = _run_main(argv)
            self.assertEqual(first, 0, msg=stderr)
            family_dir = default_la_output_dir(experiment_dir) / "constraints"
            marker = family_dir / "_SUCCESS"
            parquet = family_dir / "features.parquet"
            stamp = marker.stat().st_mtime_ns
            parquet_stamp = parquet.stat().st_mtime_ns
            second, _stdout2, stderr2 = _run_main(argv)
            self.assertEqual(second, 0, msg=stderr2)
            payload = _json_from_stderr(stderr2)
            self.assertEqual(payload["skipped"], ["constraints"])
            self.assertEqual(payload["written"], [])
            self.assertEqual(marker.stat().st_mtime_ns, stamp)
            self.assertEqual(parquet.stat().st_mtime_ns, parquet_stamp)

    def test_topology_on_complete_fixture_writes_parquet(self) -> None:
        from core.ioh_core.workflows.la.topology import (
            FEATURE_N_FEASIBLE,
            FEATURE_N_OPTIMA,
            FEATURE_Y_STAR,
            n_g_feature_name,
        )

        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "topology",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--taus",
                    "0.0",
                    "1.0",
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            payload = _json_from_stderr(stderr)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["written"], ["topology"])
            family_dir = default_la_output_dir(experiment_dir) / "topology"
            parquet = family_dir / "features.parquet"
            self.assertTrue((family_dir / "_SUCCESS").is_file())
            import_pyarrow()
            parquet_mod = __import__("importlib").import_module("pyarrow.parquet")
            table = parquet_mod.read_table(parquet)
            by_name = dict(
                zip(
                    table.column("feature_name").to_pylist(),
                    table.column("value").to_pylist(),
                    strict=True,
                )
            )
            self.assertEqual(by_name[FEATURE_N_FEASIBLE], 2.0)
            self.assertEqual(by_name[FEATURE_Y_STAR], 10.0)
            self.assertEqual(by_name[FEATURE_N_OPTIMA], 2.0)
            self.assertEqual(by_name[n_g_feature_name(0.0)], 2.0)
            self.assertEqual(by_name[n_g_feature_name(1.0)], 2.0)
            families = set(table.column("family").to_pylist())
            self.assertEqual(families, {"topology"})
            x_views = set(table.column("x_view").to_pylist())
            self.assertEqual(x_views, {"index"})

    def test_topology_missing_taus_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "topology",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("taus", payload["error"])
            family_dir = default_la_output_dir(experiment_dir) / "topology"
            self.assertFalse((family_dir / "_SUCCESS").exists())

    def test_topology_toml_taus_and_cli_override(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            config = Path(tmp) / "ela.toml"
            config.write_text(
                "[la.topology]\ntaus = [0.0]\n",
                encoding="utf-8",
            )
            code, _stdout, stderr = _run_main(
                [
                    "topology",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--config",
                    str(config),
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            import_pyarrow()
            parquet_mod = __import__("importlib").import_module("pyarrow.parquet")
            table = parquet_mod.read_table(
                default_la_output_dir(experiment_dir)
                / "topology"
                / "features.parquet"
            )
            names = table.column("feature_name").to_pylist()
            self.assertTrue(any(name.endswith(".0.0") for name in names))
            self.assertFalse(any(name.endswith(".1.0") for name in names))

    def test_mixed_constraints_topology_runs_both(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "run",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--families",
                    "constraints",
                    "topology",
                    "--taus",
                    "0.0",
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            payload = _json_from_stderr(stderr)
            self.assertEqual(payload["written"], ["constraints", "topology"])
            ela_dir = default_la_output_dir(experiment_dir)
            self.assertTrue((ela_dir / "constraints" / "_SUCCESS").is_file())
            self.assertTrue((ela_dir / "topology" / "_SUCCESS").is_file())

    def test_mixed_constraints_topology_missing_taus_writes_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "run",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--families",
                    "constraints",
                    "topology",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("taus", payload["error"])
            ela_dir = default_la_output_dir(experiment_dir)
            self.assertFalse((ela_dir / "constraints" / "_SUCCESS").exists())
            self.assertFalse((ela_dir / "topology" / "_SUCCESS").exists())

    def test_rfbx_on_complete_fixture_writes_parquet(self) -> None:
        from core.ioh_core.workflows.la.rfb import (
            FEATURE_N_CROSSINGS,
            FEATURE_RFB_X,
            derive_rfb_seed,
        )

        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "rfbx",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    *_rfbx_cli_knobs(),
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            payload = _json_from_stderr(stderr)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["written"], ["rfbx"])
            family_dir = default_la_output_dir(experiment_dir) / "rfbx"
            parquet = family_dir / "features.parquet"
            self.assertTrue((family_dir / "_SUCCESS").is_file())
            import_pyarrow()
            parquet_mod = __import__("importlib").import_module("pyarrow.parquet")
            table = parquet_mod.read_table(parquet)
            names = table.column("feature_name").to_pylist()
            self.assertEqual(set(names), {FEATURE_RFB_X, FEATURE_N_CROSSINGS})
            self.assertFalse(any("linear" in name for name in names))
            self.assertEqual(len(names), 4)
            self.assertEqual(set(table.column("family").to_pylist()), {"rfbx"})
            self.assertEqual(set(table.column("x_view").to_pylist()), {"index"})
            self.assertEqual(
                set(table.column("landscape").to_pylist()), {"feasibility"}
            )
            self.assertEqual(set(table.column("walk_id").to_pylist()), {0, 1})
            self.assertTrue(all(value is None for value in table.column("repetition").to_pylist()))
            seeds = dict(
                zip(
                    table.column("walk_id").to_pylist(),
                    table.column("seed").to_pylist(),
                    strict=True,
                )
            )
            self.assertEqual(
                seeds[0],
                derive_rfb_seed(seed_base=11, problem_id="L6750_B2", walk_id=0),
            )
            self.assertNotEqual(seeds[0], 11)

    def test_rfbx_missing_knobs_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "rfbx",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("n-walks", payload["error"])
            family_dir = default_la_output_dir(experiment_dir) / "rfbx"
            self.assertFalse((family_dir / "_SUCCESS").exists())

    def test_rfbx_toml_knobs_and_cli_override(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            config = Path(tmp) / "ela.toml"
            config.write_text(
                "[la.rfbx]\n"
                "n_walks = 1\n"
                "n_transitions = 4\n"
                'start_policy = "uniform_catalogue"\n'
                "seed_base = 11\n",
                encoding="utf-8",
            )
            code, _stdout, stderr = _run_main(
                [
                    "rfbx",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--config",
                    str(config),
                    "--n-walks",
                    "2",
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            import_pyarrow()
            parquet_mod = __import__("importlib").import_module("pyarrow.parquet")
            table = parquet_mod.read_table(
                default_la_output_dir(experiment_dir) / "rfbx" / "features.parquet"
            )
            self.assertEqual(set(table.column("walk_id").to_pylist()), {0, 1})

    def test_rfbx_on_sampling_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            code, _stdout, stderr = _run_main(
                [
                    "rfbx",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                    *_rfbx_cli_knobs(),
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("brute_force", payload["error"])
            self.assertEqual(payload["families"], ["rfbx"])
            self.assertEqual(payload["source_type"], "sampling")

    def test_basins_on_complete_fixture_writes_parquet(self) -> None:
        from core.ioh_core.workflows.la.basins import (
            FEATURE_N_BASINS,
            FEATURE_N_CORES,
            FEATURE_Y_P_STAR,
        )

        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "basins",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            payload = _json_from_stderr(stderr)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["written"], ["basins"])
            family_dir = default_la_output_dir(experiment_dir) / "basins"
            parquet = family_dir / "features.parquet"
            self.assertTrue((family_dir / "_SUCCESS").is_file())
            self.assertFalse((family_dir / "attractors.parquet").exists())
            import_pyarrow()
            parquet_mod = __import__("importlib").import_module("pyarrow.parquet")
            table = parquet_mod.read_table(parquet)
            by_name = dict(
                zip(
                    table.column("feature_name").to_pylist(),
                    table.column("value").to_pylist(),
                    strict=True,
                )
            )
            self.assertEqual(by_name[FEATURE_N_BASINS], 1.0)
            self.assertEqual(by_name[FEATURE_N_CORES], 2.0)
            self.assertEqual(by_name[FEATURE_Y_P_STAR], 11.0)
            self.assertEqual(set(table.column("family").to_pylist()), {"basins"})
            self.assertEqual(set(table.column("landscape").to_pylist()), {"y_p"})
            self.assertEqual(set(table.column("x_view").to_pylist()), {"index"})
            self.assertEqual(set(table.column("response_view").to_pylist()), {"raw"})

    def test_lon_on_complete_fixture_writes_complete_sidecars(self) -> None:
        from core.ioh_core.workflows.la.lon import FEATURE_N_NODES

        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "lon",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--radii",
                    "2",
                    "3",
                    "4",
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            payload = _json_from_stderr(stderr)
            self.assertEqual(payload["written"], ["lon"])
            family_dir = default_la_output_dir(experiment_dir) / "lon" / "y_p"
            for name in (
                "features.parquet",
                "nodes.parquet",
                "terminal_supports.parquet",
                "overlaps.parquet",
                "escape_edges.parquet",
                "_SUCCESS",
            ):
                self.assertTrue((family_dir / name).is_file(), msg=name)
            parquet_mod = __import__("importlib").import_module("pyarrow.parquet")
            features = parquet_mod.read_table(family_dir / "features.parquet")
            names = set(features.column("feature_name").to_pylist())
            self.assertIn(FEATURE_N_NODES, names)
            self.assertIn(FEATURE_BASIN_F_SHARE_MIN, names)
            self.assertTrue(any(name.startswith("lon.funnel.d2.") for name in names))
            self.assertEqual(set(features.column("family").to_pylist()), {"lon"})
            self.assertGreater(
                parquet_mod.read_table(family_dir / "nodes.parquet").num_rows,
                0,
            )
            self.assertGreater(
                parquet_mod.read_table(
                    family_dir / "terminal_supports.parquet"
                ).num_rows,
                0,
            )
            node_columns = set(
                parquet_mod.read_table(family_dir / "nodes.parquet").column_names
            )
            self.assertIn("y_p", node_columns)
            self.assertIn("is_y_p_star", node_columns)
            self.assertIn("basin_n_feasible", node_columns)
            self.assertIn("basin_f_share", node_columns)

    def test_colon_on_complete_fixture_writes_census_under_la(self) -> None:
        from core.ioh_core.workflows.la.colon import (
            FEATURE_BASIN_F_SHARE_MIN,
            FEATURE_N_NODES,
            FEATURE_STAGE_ESCAPES,
            FEATURE_Y_STAR,
            GLOBAL_FEATURE_NAMES,
        )

        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "colon",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            payload = _json_from_stderr(stderr)
            self.assertEqual(payload["written"], ["colon"])
            family_dir = default_la_output_dir(experiment_dir) / "colon"
            ela_dir = default_ela_output_dir(experiment_dir) / "colon"
            for name in (
                "features.parquet",
                "nodes.parquet",
                "terminal_supports.parquet",
                "overlaps.parquet",
                "escape_edges.parquet",
                "census.json",
                "_SUCCESS",
            ):
                self.assertTrue((family_dir / name).is_file(), msg=name)
            self.assertFalse(ela_dir.exists())
            parquet_mod = __import__("importlib").import_module("pyarrow.parquet")
            features = parquet_mod.read_table(family_dir / "features.parquet")
            by_name = dict(
                zip(
                    features.column("feature_name").to_pylist(),
                    features.column("value").to_pylist(),
                    strict=True,
                )
            )
            status = dict(
                zip(
                    features.column("feature_name").to_pylist(),
                    features.column("status").to_pylist(),
                    strict=True,
                )
            )
            self.assertIn(FEATURE_N_NODES, by_name)
            self.assertIn(FEATURE_BASIN_F_SHARE_MIN, by_name)
            self.assertGreater(by_name[FEATURE_N_NODES], 0.0)
            self.assertEqual(status[FEATURE_STAGE_ESCAPES], "ok")
            self.assertEqual(by_name[FEATURE_STAGE_ESCAPES], 1.0)
            self.assertIn("colon.escape.d2.escape_candidate_feasible", by_name)
            self.assertIn("colon.funnel.d2.n_funnels", by_name)
            self.assertEqual(status[FEATURE_Y_STAR], "ok")
            self.assertIn("colon.stage.basin_census", by_name)
            self.assertEqual(set(features.column("family").to_pylist()), {"colon"})
            self.assertEqual(set(features.column("landscape").to_pylist()), {"deb"})
            node_columns = set(
                parquet_mod.read_table(family_dir / "nodes.parquet").column_names
            )
            self.assertIn("terminal_feasible", node_columns)
            self.assertIn("basin_n_feasible", node_columns)
            self.assertIn("basin_n_total", node_columns)
            self.assertIn("basin_f_share", node_columns)
            census = json.loads(
                (family_dir / "census.json").read_text(encoding="utf-8")
            )
            self.assertEqual(census["basin_census"], "complete")
            self.assertEqual(census["escapes"], "complete")
            self.assertEqual(census["funnels"], "complete")
            self.assertTrue(GLOBAL_FEATURE_NAMES)

    def test_lon_y_raw_writes_independent_landscape_partition(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "lon",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--landscape",
                    "y_raw",
                    "--radii",
                    "2",
                    "3",
                    "4",
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            family_dir = default_la_output_dir(experiment_dir) / "lon" / "y_raw"
            self.assertTrue((family_dir / "_SUCCESS").is_file())
            self.assertFalse(
                (default_la_output_dir(experiment_dir) / "lon" / "_SUCCESS").exists()
            )
            parquet_mod = __import__("importlib").import_module("pyarrow.parquet")
            features = parquet_mod.read_table(family_dir / "features.parquet")
            self.assertEqual(set(features.column("landscape").to_pylist()), {"y_raw"})
            names = set(features.column("feature_name").to_pylist())
            self.assertIn("lon.y_box_star", names)
            self.assertNotIn("lon.y_p_star", names)
            node_columns = set(
                parquet_mod.read_table(family_dir / "nodes.parquet").column_names
            )
            self.assertIn("y_raw", node_columns)
            self.assertIn("is_y_box_star", node_columns)
            self.assertNotIn("y_p", node_columns)
            edge_columns = set(
                parquet_mod.read_table(
                    family_dir / "escape_edges.parquet"
                ).column_names
            )
            self.assertIn("source_y_raw", edge_columns)
            self.assertIn("target_y_raw", edge_columns)

    def test_lon_landscape_partitions_skip_independently(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            outcomes: list[dict] = []
            for landscape in ("y_p", "y_raw", "y_p"):
                code, _stdout, stderr = _run_main(
                    [
                        "lon",
                        "--experiment-dir",
                        str(experiment_dir),
                        "--source-type",
                        "brute_force",
                        "--landscape",
                        landscape,
                        "--radii",
                        "2",
                        "3",
                        "4",
                    ]
                )
                self.assertEqual(code, 0, msg=stderr)
                outcomes.append(_json_from_stderr(stderr))
            self.assertEqual(outcomes[0]["written"], ["lon"])
            self.assertEqual(outcomes[1]["written"], ["lon"])
            self.assertEqual(outcomes[2]["written"], [])
            self.assertEqual(outcomes[2]["skipped"], ["lon"])
            root = default_la_output_dir(experiment_dir) / "lon"
            self.assertTrue((root / "y_p" / "_SUCCESS").is_file())
            self.assertTrue((root / "y_raw" / "_SUCCESS").is_file())

    def test_lon_requires_radii_and_toml_can_supply_them(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = root / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "lon",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("radii", _json_from_stderr(stderr)["error"])
            code, _stdout, stderr = _run_main(
                [
                    "lon",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--radii",
                    "1",
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("frozen", _json_from_stderr(stderr)["error"])
            config = root / "ela.toml"
            config.write_text(
                "[la.lon]\nradii = [2, 3, 4]\nlandscape = \"y_raw\"\n",
                encoding="utf-8",
            )
            code, _stdout, stderr = _run_main(
                [
                    "lon",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--config",
                    str(config),
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            self.assertTrue(
                (
                    default_la_output_dir(experiment_dir)
                    / "lon"
                    / "y_raw"
                    / "_SUCCESS"
                ).is_file()
            )
            code, _stdout, stderr = _run_main(
                [
                    "lon",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--landscape",
                    "y",
                    "--radii",
                    "2",
                    "3",
                    "4",
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("y_p or y_raw", _json_from_stderr(stderr)["error"])

    def test_lon_requires_stored_feasible_column(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(
                experiment_dir,
                include_feasible=False,
            )
            code, _stdout, stderr = _run_main(
                [
                    "lon",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--radii",
                    "2",
                    "3",
                    "4",
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("feasible", _json_from_stderr(stderr)["error"])
            self.assertFalse(
                (
                    default_la_output_dir(experiment_dir)
                    / "lon"
                    / "_SUCCESS"
                ).exists()
            )

    def test_lon_rejects_null_stored_feasible_value(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir, null_feasible=True)
            code, _stdout, stderr = _run_main(
                [
                    "lon",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--radii",
                    "2",
                    "3",
                    "4",
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("NULL", _json_from_stderr(stderr)["error"])

    def test_basins_write_attractors(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "basins",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--write-attractors",
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            family_dir = default_la_output_dir(experiment_dir) / "basins"
            attractors = family_dir / "attractors.parquet"
            self.assertTrue(attractors.is_file())
            import_pyarrow()
            parquet_mod = __import__("importlib").import_module("pyarrow.parquet")
            table = parquet_mod.read_table(attractors)
            self.assertEqual(table.num_rows, 2)
            self.assertEqual(set(table.column("basin_id").to_pylist()), {0})

    def test_basins_toml_write_attractors(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            config = Path(tmp) / "ela.toml"
            config.write_text("[la.basins]\nwrite_attractors = true\n", encoding="utf-8")
            code, _stdout, stderr = _run_main(
                [
                    "basins",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--config",
                    str(config),
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            family_dir = default_la_output_dir(experiment_dir) / "basins"
            self.assertTrue((family_dir / "attractors.parquet").is_file())
            self.assertTrue((family_dir / "features.parquet").is_file())
            self.assertTrue((family_dir / "_SUCCESS").is_file())

    def test_basins_on_sampling_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            code, _stdout, stderr = _run_main(
                [
                    "basins",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("brute_force", payload["error"])
            self.assertEqual(payload["families"], ["basins"])

    def test_mixed_constraints_topology_rfbx_runs_all(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "run",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--families",
                    "constraints",
                    "topology",
                    "rfbx",
                    "--taus",
                    "0.0",
                    *_rfbx_cli_knobs(),
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            payload = _json_from_stderr(stderr)
            self.assertTrue(payload["ok"])
            self.assertNotIn("not implemented yet", stderr)
            self.assertEqual(
                payload["written"], ["constraints", "topology", "rfbx"]
            )
            ela_dir = default_la_output_dir(experiment_dir)
            for token in ("constraints", "topology", "rfbx"):
                family_dir = ela_dir / token
                self.assertTrue((family_dir / "_SUCCESS").is_file(), msg=token)
                self.assertTrue((family_dir / "features.parquet").is_file(), msg=token)

    def test_mixed_constraints_rfbx_missing_knobs_writes_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            code, _stdout, stderr = _run_main(
                [
                    "run",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "brute_force",
                    "--families",
                    "constraints",
                    "rfbx",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("n-walks", payload["error"])
            ela_dir = default_la_output_dir(experiment_dir)
            self.assertFalse((ela_dir / "constraints" / "_SUCCESS").exists())
            self.assertFalse((ela_dir / "rfbx" / "_SUCCESS").exists())

    def test_topology_on_sampling_exits_1(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            code, _stdout, stderr = _run_main(
                [
                    "topology",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                    "--taus",
                    "0.0",
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("exhaustive", payload["error"])
            self.assertIn("sampling", payload["error"])
            self.assertEqual(payload["families"], ["topology"])
            self.assertEqual(payload["source_type"], "sampling")
            self.assertNotIn("not implemented yet", payload["error"])

    def test_subset_keep_list_complete_fixture_refuses_canonical(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            canonical = default_la_output_dir(experiment_dir)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                code, _stdout, stderr = _run_main(
                    [
                        "constraints",
                        "--experiment-dir",
                        str(experiment_dir),
                        "--source-type",
                        "brute_force",
                        "--constraint-names",
                        "A_bending_capacity",
                        "--output-dir",
                        str(canonical),
                    ]
                )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("canonical", payload["error"])
            self.assertTrue(payload["constraint_set_is_subset"])
            self.assertFalse((canonical / "constraints" / "_SUCCESS").exists())
            runtime = [
                item for item in caught if issubclass(item.category, RuntimeWarning)
            ]
            self.assertEqual(len(runtime), 1)

    def test_subset_keep_list_complete_fixture_writes_distinct_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            _write_complete_bf_fixture(experiment_dir)
            out = Path(tmp) / "ela_subset"
            canonical = default_la_output_dir(experiment_dir)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                code, _stdout, stderr = _run_main(
                    [
                        "constraints",
                        "--experiment-dir",
                        str(experiment_dir),
                        "--source-type",
                        "brute_force",
                        "--constraint-names",
                        "A_bending_capacity",
                        "--output-dir",
                        str(out),
                    ]
                )
            self.assertEqual(code, 0, msg=stderr)
            payload = _json_from_stderr(stderr)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["constraint_set_is_subset"])
            self.assertEqual(payload["written"], ["constraints"])
            self.assertTrue((out / "constraints" / "_SUCCESS").is_file())
            self.assertTrue((out / "constraints" / "features.parquet").is_file())
            self.assertFalse((canonical / "constraints" / "_SUCCESS").exists())
            runtime = [
                item for item in caught if issubclass(item.category, RuntimeWarning)
            ]
            self.assertEqual(len(runtime), 1)
            self.assertEqual(str(runtime[0].message), CONSTRAINT_SUBSET_WARNING)

    def test_mixed_families_fail_on_either_source(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            for source in ("sampling", "brute_force"):
                code, _stdout, stderr = _run_main(
                    [
                        "run",
                        "--experiment-dir",
                        str(experiment_dir),
                        "--source-type",
                        source,
                        "--families",
                        "meta-model",
                        "constraints",
                    ]
                )
                self.assertEqual(code, 1, msg=source)
                payload = _json_from_stderr(stderr)
                self.assertFalse(payload["ok"])

    def test_empty_dir_does_not_infer_brute_force(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            code, _stdout, stderr = _run_main(
                ["topology", "--experiment-dir", str(experiment_dir)]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertIn("cannot infer", payload["error"])

    def test_inputs_sampling_dir_infers_sampling(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            (experiment_dir / "inputs" / "sampling").mkdir(parents=True)
            code, _stdout, stderr = _run_main(
                ["constraints", "--experiment-dir", str(experiment_dir)]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertEqual(payload["source_type"], "sampling")
            self.assertEqual(payload["families"], ["constraints"])
            self.assertIn("dataset/_SUCCESS", payload["error"])

    def test_manifest_workflow_type_infers_brute_force(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            experiment_dir.mkdir()
            (experiment_dir / "manifest.json").write_text(
                json.dumps({"workflow_type": "brute_force"}),
                encoding="utf-8",
            )
            code, _stdout, stderr = _run_main(
                ["topology", "--experiment-dir", str(experiment_dir)]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertEqual(payload["source_type"], "brute_force")
            self.assertEqual(payload["families"], ["topology"])
            self.assertIn("dataset/_SUCCESS", payload["error"])

    def test_cli_source_type_overrides_toml(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_dir = root / "exp"
            experiment_dir.mkdir()
            config = root / "ela.toml"
            config.write_text(
                '[la]\nsource_type = "brute_force"\n',
                encoding="utf-8",
            )
            code, _stdout, stderr = _run_main(
                [
                    "constraints",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--source-type",
                    "sampling",
                    "--config",
                    str(config),
                ]
            )
            self.assertEqual(code, 1)
            payload = _json_from_stderr(stderr)
            self.assertEqual(payload["source_type"], "sampling")
            self.assertIn("dataset/_SUCCESS", payload["error"])


if __name__ == "__main__":
    unittest.main()
