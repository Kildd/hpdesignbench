"""
Phase A1 / ALG4.3 gate: execution/workflow imports stay free of optional extras.

Importing these packages must not pull optional PyArrow / optimiser libraries
and must not invoke Slurm commands.
"""
from __future__ import annotations

import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

_OPTIONAL_ALGORITHM_LIBS = ("pymoo", "hyperopt", "rbfopt")


def _drop_optional_algorithm_modules() -> None:
    """
    Remove optional optimiser modules from ``sys.modules`` when present.
    """
    doomed = [
        name
        for name in sys.modules
        if name in _OPTIONAL_ALGORITHM_LIBS
        or any(name.startswith(f"{lib}.") for lib in _OPTIONAL_ALGORITHM_LIBS)
    ]
    for name in doomed:
        sys.modules.pop(name, None)


class TestExecutionPackageImports(unittest.TestCase):
    """Skeleton packages import without optional/heavy side effects."""

    def test_execution_core_imports(self) -> None:
        import core.execution_core as execution_core
        import core.execution_core.backends as backends

        self.assertTrue(hasattr(execution_core, "__file__"))
        self.assertTrue(hasattr(backends, "__file__"))

    def test_ioh_workflows_imports(self) -> None:
        import core.ioh_core.workflows as workflows
        import core.ioh_core.workflows.optimization_runs as optimization_runs

        self.assertTrue(hasattr(workflows, "__file__"))
        self.assertTrue(hasattr(optimization_runs, "__file__"))

    def test_import_does_not_load_pyarrow(self) -> None:
        # Drop any prior accidental import so the check is meaningful.
        sys.modules.pop("pyarrow", None)
        sys.modules.pop("pyarrow.parquet", None)

        import importlib

        importlib.reload(importlib.import_module("core.execution_core"))
        importlib.reload(importlib.import_module("core.execution_core.backends"))
        importlib.reload(importlib.import_module("core.ioh_core.workflows"))
        importlib.reload(
            importlib.import_module("core.ioh_core.workflows.optimization_runs")
        )

        self.assertNotIn("pyarrow", sys.modules)
        self.assertNotIn("pyarrow.parquet", sys.modules)

    def test_import_does_not_invoke_slurm_commands(self) -> None:
        blocked = ("sbatch", "squeue", "scancel", "scontrol", "sacct")

        real_popen = __import__("subprocess").Popen

        def _guarded_popen(cmd, *args, **kwargs):
            exe = cmd[0] if isinstance(cmd, (list, tuple)) and cmd else cmd
            name = str(exe).lower().replace("\\", "/").rsplit("/", 1)[-1]
            if name in blocked or name.rstrip(".exe") in blocked:
                raise AssertionError(f"Slurm command invoked during import: {cmd!r}")
            return real_popen(cmd, *args, **kwargs)

        with mock.patch("subprocess.Popen", side_effect=_guarded_popen):
            with mock.patch("subprocess.run") as run_mock:
                with mock.patch("subprocess.call") as call_mock:
                    with mock.patch("subprocess.check_call") as check_call_mock:
                        with mock.patch("subprocess.check_output") as check_output_mock:
                            import importlib

                            importlib.import_module("core.execution_core")
                            importlib.import_module("core.execution_core.backends")
                            importlib.import_module("core.ioh_core.workflows")
                            importlib.import_module(
                                "core.ioh_core.workflows.optimization_runs"
                            )

                            run_mock.assert_not_called()
                            call_mock.assert_not_called()
                            check_call_mock.assert_not_called()
                            check_output_mock.assert_not_called()

    def test_status_collect_and_cli_help_do_not_import_optional_algorithms(
        self,
    ) -> None:
        from core.execution_core.lifecycle import prepare_experiment
        from core.ioh_core.algorithms import RandomSearch, list_algorithm_specs
        from core.ioh_core.workflows.cli import build_parser, main

        _drop_optional_algorithm_modules()

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "cfg.toml"
            config_path.write_text(
                textwrap.dedent(
                    """\
                    schema_version = 1

                    [experiment]
                    id = "optional_isolation"
                    workflow = "optimization_runs"
                    slab_type = "hp_slab"
                    problem_ids = ["L6750_B2"]
                    output_root = "experiments"

                    [[experiment.algorithms]]
                    name = "RANDOM_SEARCH"
                    max_evaluations = 2

                    [experiment.repetitions]
                    count = 1
                    seed_base = 1

                    [execution]
                    backend = "local"
                    max_parallel_tasks = 1
                    cpus_per_task = 1

                    [execution.local]
                    python_executable = ""
                    """
                ),
                encoding="utf-8",
            )
            experiment_dir = root / "exp"

            blocked = ("sbatch", "squeue", "scancel", "scontrol", "sacct")
            real_run = __import__("subprocess").run
            real_popen = __import__("subprocess").Popen

            def _blocked_name(cmd) -> str:
                exe = cmd[0] if isinstance(cmd, (list, tuple)) and cmd else cmd
                return str(exe).lower().replace("\\", "/").rsplit("/", 1)[-1]

            def _guarded_run(cmd, *args, **kwargs):
                name = _blocked_name(cmd)
                if name in blocked or name.rstrip(".exe") in blocked:
                    raise AssertionError(f"Slurm command invoked: {cmd!r}")
                return real_run(cmd, *args, **kwargs)

            def _guarded_popen(cmd, *args, **kwargs):
                name = _blocked_name(cmd)
                if name in blocked or name.rstrip(".exe") in blocked:
                    raise AssertionError(f"Slurm command invoked: {cmd!r}")
                return real_popen(cmd, *args, **kwargs)

            with mock.patch("subprocess.run", side_effect=_guarded_run):
                with mock.patch("subprocess.Popen", side_effect=_guarded_popen):
                    prepare_experiment(
                        config_path=config_path,
                        output=experiment_dir,
                    )

                    help_text = build_parser().format_help()
                    self.assertIn("status", help_text)
                    self.assertIn("collect", help_text)

                    self.assertEqual(
                        main(
                            [
                                "status",
                                "--experiment-dir",
                                str(experiment_dir),
                            ]
                        ),
                        0,
                    )
                    self.assertEqual(
                        main(
                            [
                                "collect",
                                "--experiment-dir",
                                str(experiment_dir),
                                "--allow-incomplete",
                            ]
                        ),
                        0,
                    )

            names = [spec.name for spec in list_algorithm_specs()]
            self.assertIn("RANDOM_SEARCH", names)
            algo = RandomSearch(max_evaluations=2, seed=0)
            self.assertEqual(algo.name, "RANDOM_SEARCH")

        for name in _OPTIONAL_ALGORITHM_LIBS:
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
