"""
Phase C4.1 gate: optimisation termination configuration parsing.
"""
from __future__ import annotations

import math
import textwrap
import unittest
import warnings
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from core.execution_core.config import (
    ConfigError,
    TerminationConfig,
    load_experiment_config,
    parse_experiment_config,
)


def _optimization_toml(*, termination: str = "", problem_ids: str = '["L6750_B2"]') -> str:
    """
    Return a minimal optimisation TOML with an optional termination table.

    Parameters
    ----------
    termination : str, optional
        Raw TOML fragment for ``[experiment.termination]`` (may be empty).
    problem_ids : str, optional
        Literal TOML array text for ``problem_ids``.

    Returns
    -------
    str
        Complete TOML document.
    """
    body = textwrap.dedent(
        f"""\
        schema_version = 1

        [experiment]
        id = "termination_smoke"
        workflow = "optimization_runs"
        slab_type = "hp_slab"
        problem_ids = {problem_ids}
        output_root = "experiments"

        [[experiment.algorithms]]
        name = "RANDOM_SEARCH"
        max_evaluations = 10

        [experiment.repetitions]
        count = 1
        seed_base = 42

        [execution]
        backend = "local"
        max_parallel_tasks = 1
        cpus_per_task = 1

        [execution.local]
        python_executable = ""
        """
    )
    if termination.strip():
        body = body + "\n" + termination.strip() + "\n"
    return body


def _parse(text: str):
    """Parse TOML text into an ExperimentConfig."""
    return parse_experiment_config(tomllib.loads(text))


class TestTerminationConfigModel(unittest.TestCase):
    """Direct TerminationConfig validation and threshold arithmetic."""

    def test_fixed_budget_defaults(self) -> None:
        config = TerminationConfig()
        self.assertEqual(config.mode, "fixed_budget")
        self.assertIsNone(config.global_optimum)
        self.assertIsNone(config.allowed_deviation_percent)
        self.assertIsNone(config.target_reference)
        self.assertIsNone(config.target_threshold)

    def test_percentage_arithmetic_zero_and_boundary(self) -> None:
        zero = TerminationConfig(
            mode="fixed_target",
            global_optimum=500.0,
            allowed_deviation_percent=0.0,
            target_reference="all",
        )
        self.assertEqual(zero.target_threshold, 500.0)

        five = TerminationConfig(
            mode="fixed_target",
            global_optimum=500.0,
            allowed_deviation_percent=5.0,
            target_reference="all",
        )
        self.assertEqual(five.target_threshold, 525.0)

    def test_overflow_threshold_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            TerminationConfig(
                mode="fixed_target",
                global_optimum=math.pow(10.0, 308),
                allowed_deviation_percent=100.0,
                target_reference="all",
            )

    def test_nan_and_negative_rejected_on_model(self) -> None:
        with self.assertRaises(ConfigError):
            TerminationConfig(
                mode="fixed_target",
                global_optimum=float("nan"),
                allowed_deviation_percent=1.0,
                target_reference="all",
            )
        with self.assertRaises(ConfigError):
            TerminationConfig(
                mode="fixed_target",
                global_optimum=1.0,
                allowed_deviation_percent=-1.0,
                target_reference="all",
            )


class TestTerminationConfigParsing(unittest.TestCase):
    """TOML parse matrix for [experiment.termination]."""

    def test_omitted_table_is_fixed_budget(self) -> None:
        config = _parse(_optimization_toml())
        termination = config.experiment.termination
        self.assertEqual(termination.mode, "fixed_budget")
        self.assertIsNone(termination.target_reference)
        self.assertIsNone(termination.target_threshold)

    def test_explicit_fixed_budget(self) -> None:
        config = _parse(
            _optimization_toml(
                termination=textwrap.dedent(
                    """\
                    [experiment.termination]
                    mode = "fixed_budget"
                    """
                )
            )
        )
        self.assertEqual(config.experiment.termination.mode, "fixed_budget")

    def test_fixed_budget_forbids_target_fields(self) -> None:
        for fragment in (
            'mode = "fixed_budget"\nglobal_optimum = 1.0\n',
            'mode = "fixed_budget"\nallowed_deviation_percent = 1.0\n',
            'mode = "fixed_budget"\ntarget_reference = "all"\n',
        ):
            with self.subTest(fragment=fragment):
                with self.assertRaises(ConfigError):
                    _parse(
                        _optimization_toml(
                            termination="[experiment.termination]\n" + fragment
                        )
                    )

    def test_fixed_target_requires_inputs_and_defaults_reference(self) -> None:
        config = _parse(
            _optimization_toml(
                termination=textwrap.dedent(
                    """\
                    [experiment.termination]
                    mode = "fixed_target"
                    global_optimum = 500.0
                    allowed_deviation_percent = 5.0
                    """
                )
            )
        )
        termination = config.experiment.termination
        self.assertEqual(termination.mode, "fixed_target")
        self.assertEqual(termination.global_optimum, 500.0)
        self.assertEqual(termination.allowed_deviation_percent, 5.0)
        self.assertEqual(termination.target_reference, "all")
        self.assertEqual(termination.target_threshold, 525.0)

    def test_fixed_target_accepts_all_reference(self) -> None:
        config = _parse(
            _optimization_toml(
                termination=textwrap.dedent(
                    """\
                    [experiment.termination]
                    mode = "fixed_target"
                    global_optimum = 100.0
                    allowed_deviation_percent = 1.0
                    target_reference = "all"
                    """
                )
            )
        )
        self.assertEqual(config.experiment.termination.target_reference, "all")
        self.assertEqual(config.experiment.termination.target_threshold, 101.0)

    def test_historical_feasible_spelling_normalizes_to_all(self) -> None:
        config = _parse(
            _optimization_toml(
                termination=textwrap.dedent(
                    """\
                    [experiment.termination]
                    mode = "fixed_target"
                    global_optimum = 100.0
                    allowed_deviation_percent = 1.0
                    target_reference = "feasible"
                    """
                )
            )
        )
        self.assertEqual(config.experiment.termination.target_reference, "all")

    def test_fixed_target_rejects_unknown_reference_and_mode(self) -> None:
        with self.assertRaises(ConfigError):
            _parse(
                _optimization_toml(
                    termination=textwrap.dedent(
                        """\
                        [experiment.termination]
                        mode = "fixed_target"
                        global_optimum = 100.0
                        allowed_deviation_percent = 1.0
                        target_reference = "infeasible"
                        """
                    )
                )
            )
        with self.assertRaises(ConfigError):
            _parse(
                _optimization_toml(
                    termination=textwrap.dedent(
                        """\
                        [experiment.termination]
                        mode = "budget"
                        """
                    )
                )
            )

    def test_rejects_bool_negative_and_unknown_keys(self) -> None:
        cases = (
            textwrap.dedent(
                """\
                [experiment.termination]
                mode = "fixed_target"
                global_optimum = true
                allowed_deviation_percent = 1.0
                """
            ),
            textwrap.dedent(
                """\
                [experiment.termination]
                mode = "fixed_target"
                global_optimum = 1.0
                allowed_deviation_percent = -1.0
                """
            ),
            textwrap.dedent(
                """\
                [experiment.termination]
                mode = "fixed_target"
                global_optimum = 1.0
                allowed_deviation_percent = 1.0
                extra = 1
                """
            ),
        )
        for fragment in cases:
            with self.subTest(fragment=fragment):
                with self.assertRaises(ConfigError):
                    _parse(_optimization_toml(termination=fragment))

    def test_multi_problem_fixed_target_requires_optimum_map(self) -> None:
        with self.assertRaisesRegex(ConfigError, "global_optima"):
            _parse(
                _optimization_toml(
                    problem_ids='["1", "2"]',
                    termination=textwrap.dedent(
                        """\
                        [experiment.termination]
                        mode = "fixed_target"
                        global_optimum = 500.0
                        allowed_deviation_percent = 5.0
                        target_reference = "all"
                        """
                    ),
                )
            )

    def test_multi_problem_fixed_target_accepts_exact_optimum_map(self) -> None:
        config = _parse(
            _optimization_toml(
                problem_ids='["1", "2"]',
                termination=textwrap.dedent(
                    """\
                    [experiment.termination]
                    mode = "fixed_target"
                    allowed_deviation_percent = 5.0
                    target_reference = "all"

                    [experiment.termination.global_optima]
                    1 = 100.0
                    2 = 200.0
                    """
                ),
            )
        )
        termination = config.experiment.termination
        self.assertIsNone(termination.global_optimum)
        self.assertEqual(dict(termination.global_optima), {"1": 100.0, "2": 200.0})
        self.assertEqual(termination.global_optimum_for("1"), 100.0)
        self.assertEqual(termination.target_threshold_for("1"), 105.0)
        self.assertEqual(termination.global_optimum_for("2"), 200.0)
        self.assertEqual(termination.target_threshold_for("2"), 210.0)
        self.assertIsNone(termination.target_threshold)

    def test_optimum_map_rejects_missing_extra_and_scalar_combination(self) -> None:
        cases = (
            textwrap.dedent(
                """\
                [experiment.termination]
                mode = "fixed_target"
                allowed_deviation_percent = 1.0
                [experiment.termination.global_optima]
                1 = 100.0
                """
            ),
            textwrap.dedent(
                """\
                [experiment.termination]
                mode = "fixed_target"
                allowed_deviation_percent = 1.0
                [experiment.termination.global_optima]
                1 = 100.0
                2 = 200.0
                3 = 300.0
                """
            ),
            textwrap.dedent(
                """\
                [experiment.termination]
                mode = "fixed_target"
                global_optimum = 100.0
                allowed_deviation_percent = 1.0
                [experiment.termination.global_optima]
                1 = 100.0
                2 = 200.0
                """
            ),
        )
        for termination in cases:
            with self.subTest(termination=termination):
                with self.assertRaises(ConfigError):
                    _parse(
                        _optimization_toml(
                            problem_ids='["1", "2"]',
                            termination=termination,
                        )
                    )

    def test_termination_forbidden_on_brute_force(self) -> None:
        toml = textwrap.dedent(
            """\
            schema_version = 1

            [experiment]
            id = "bf_term"
            workflow = "brute_force"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [experiment.termination]
            mode = "fixed_budget"

            [brute_force]
            rows_per_shard = 100
            acknowledge_large_run = true
            maximum_expected_rows = 1000
            maximum_expected_bytes = 1000000

            [execution]
            backend = "local"
            max_parallel_tasks = 1
            cpus_per_task = 1

            [execution.local]
            python_executable = ""
            """
        )
        with self.assertRaises(ConfigError):
            _parse(toml)

    def test_existing_example_configs_still_parse(self) -> None:
        examples = Path(__file__).resolve().parents[2] / (
            "_mains/studies/Execution/configs/examples"
        )
        fixed_target_reference = {
            "optimization_fixed_target.toml": "all",
            "optimization_smoke_optuna_pfn_library.toml": "all",
            "optimization_smoke_smt_nevergrad_library.toml": "all",
        }
        for path in sorted(examples.glob("*.toml")):
            with self.subTest(path=path.name):
                config = load_experiment_config(path)
                if config.experiment.workflow != "optimization_runs":
                    continue
                if path.name in fixed_target_reference:
                    self.assertEqual(
                        config.experiment.termination.mode,
                        "fixed_target",
                    )
                    self.assertEqual(
                        config.experiment.termination.target_reference,
                        fixed_target_reference[path.name],
                    )
                else:
                    self.assertEqual(
                        config.experiment.termination.mode,
                        "fixed_budget",
                    )


if __name__ == "__main__":
    unittest.main()
