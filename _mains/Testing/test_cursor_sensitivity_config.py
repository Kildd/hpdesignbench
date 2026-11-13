"""
C1.2 gate: strict sensitivity_analysis TOML configuration.
"""
from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from core.execution_core.config import (
    ConfigError,
    load_experiment_config,
    parse_experiment_config,
)


def _execution_block() -> str:
    return textwrap.dedent(
        """\
        [execution]
        backend = "local"
        max_parallel_tasks = 1
        cpus_per_task = 1

        [execution.thread_environment]
        OMP_NUM_THREADS = "1"

        [execution.local]
        python_executable = ""

        [execution.slurm]
        python_executable = ""
        partition = ""
        account = ""
        memory = "8G"
        time_limit = "01:00:00"
        array_concurrency = 64
        maximum_array_size = 1000
        """
    )


def _sensitivity_doc(
    *,
    sample_size: int = 16,
    factor_names: str = '["geom_h_ges_mm", "geom_t_mm"]',
    response_key: str = "y",
    scramble: str = "true",
    extra_tables: str = "",
    problem_ids: str = '["L6750_B2"]',
    include_bootstrap: bool = True,
    compute_bootstrap_ci: str = "true",
) -> str:
    bootstrap = ""
    if include_bootstrap:
        bootstrap = textwrap.dedent(
            f"""\
            bootstrap_resamples = 100
            bootstrap_confidence_level = 0.95
            bootstrap_seed = 11
            compute_bootstrap_ci = {compute_bootstrap_ci}
            """
        )
    return (
        textwrap.dedent(
            f"""\
            schema_version = 1

            [experiment]
            id = "sensitivity_smoke"
            workflow = "sensitivity_analysis"
            slab_type = "hp_slab"
            problem_ids = {problem_ids}
            output_root = "experiments"

            [sensitivity_analysis]
            sample_size = {sample_size}
            factor_names = {factor_names}
            response_key = "{response_key}"
            scramble = {scramble}
            seed = 7
            blocks_per_shard = 4
            parquet_batch_rows = 32
            compression = "zstd"
            acknowledge_large_run = false
            maximum_expected_evaluations = 100000
            maximum_expected_bytes = 2000000000
            """
        )
        + bootstrap
        + extra_tables
        + "\n"
        + _execution_block()
    )


class TestSensitivityConfig(unittest.TestCase):
    """Strict sensitivity workflow TOML parsing."""

    def test_valid_config_loads(self) -> None:
        config = parse_experiment_config(tomllib.loads(_sensitivity_doc()))
        self.assertEqual(config.experiment.workflow, "sensitivity_analysis")
        self.assertIsNone(config.brute_force)
        self.assertIsNotNone(config.sensitivity_analysis)
        section = config.sensitivity_analysis
        assert section is not None
        self.assertEqual(section.sample_size, 16)
        self.assertEqual(section.factor_names, ("geom_h_ges_mm", "geom_t_mm"))
        self.assertTrue(section.compute_bootstrap_ci)
        self.assertEqual(section.bootstrap_resamples, 100)

    def test_valid_config_from_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensitivity.toml"
            path.write_text(_sensitivity_doc(), encoding="utf-8")
            config = load_experiment_config(path)
            self.assertEqual(config.experiment.workflow, "sensitivity_analysis")
            self.assertEqual(config.experiment.algorithms, ())
            self.assertIsNone(config.experiment.repetitions)

    def test_rejects_algorithms(self) -> None:
        text = _sensitivity_doc(
            extra_tables=textwrap.dedent(
                """\
                [[experiment.algorithms]]
                name = "RANDOM_SEARCH"
                max_evaluations = 10
                """
            )
        )
        with self.assertRaisesRegex(ConfigError, "algorithms is not allowed"):
            parse_experiment_config(tomllib.loads(text))

    def test_requires_sensitivity_section(self) -> None:
        text = textwrap.dedent(
            """\
            schema_version = 1
            [experiment]
            id = "sensitivity_smoke"
            workflow = "sensitivity_analysis"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"
            """
        ) + _execution_block()
        with self.assertRaisesRegex(ConfigError, r"\[sensitivity_analysis\]"):
            parse_experiment_config(tomllib.loads(text))

    def test_rejects_brute_force_section(self) -> None:
        text = _sensitivity_doc(
            extra_tables=textwrap.dedent(
                """\
                [brute_force]
                rows_per_shard = 10
                parquet_batch_rows = 8
                compression = "zstd"
                acknowledge_large_run = false
                maximum_expected_rows = 1000
                maximum_expected_bytes = 1000
                """
            )
        )
        with self.assertRaisesRegex(ConfigError, r"\[brute_force\] is only allowed"):
            parse_experiment_config(tomllib.loads(text))

    def test_requires_exactly_one_problem_id(self) -> None:
        with self.assertRaisesRegex(ConfigError, "exactly one problem_id"):
            parse_experiment_config(
                tomllib.loads(_sensitivity_doc(problem_ids='["1", "2"]'))
            )

    def test_rejects_non_power_of_two_sample_size(self) -> None:
        with self.assertRaisesRegex(ConfigError, "power of two"):
            parse_experiment_config(tomllib.loads(_sensitivity_doc(sample_size=12)))

    def test_rejects_duplicate_factors(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unique"):
            parse_experiment_config(
                tomllib.loads(
                    _sensitivity_doc(
                        factor_names='["geom_h_ges_mm", "geom_h_ges_mm"]'
                    )
                )
            )

    def test_rejects_bad_response_key(self) -> None:
        with self.assertRaisesRegex(ConfigError, "response_key"):
            parse_experiment_config(
                tomllib.loads(_sensitivity_doc(response_key="y_raw"))
            )

    def test_rejects_integer_bool_for_scramble(self) -> None:
        with self.assertRaisesRegex(ConfigError, "scramble must be a boolean"):
            parse_experiment_config(tomllib.loads(_sensitivity_doc(scramble="1")))

    def test_requires_bootstrap_keys_even_when_ci_disabled(self) -> None:
        with self.assertRaisesRegex(ConfigError, "missing required key"):
            parse_experiment_config(
                tomllib.loads(
                    _sensitivity_doc(
                        include_bootstrap=False,
                        compute_bootstrap_ci="false",
                    )
                )
            )

    def test_ci_disabled_still_loads_when_bootstrap_keys_present(self) -> None:
        config = parse_experiment_config(
            tomllib.loads(_sensitivity_doc(compute_bootstrap_ci="false"))
        )
        assert config.sensitivity_analysis is not None
        self.assertFalse(config.sensitivity_analysis.compute_bootstrap_ci)

    def test_optimization_rejects_sensitivity_section(self) -> None:
        text = textwrap.dedent(
            """\
            schema_version = 1
            [experiment]
            id = "opt"
            workflow = "optimization_runs"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"

            [[experiment.algorithms]]
            name = "RANDOM_SEARCH"
            max_evaluations = 10

            [experiment.repetitions]
            count = 1
            seed_base = 1

            [sensitivity_analysis]
            sample_size = 8
            factor_names = ["geom_h_ges_mm"]
            seed = 1
            blocks_per_shard = 1
            parquet_batch_rows = 8
            bootstrap_resamples = 10
            bootstrap_confidence_level = 0.95
            bootstrap_seed = 1
            maximum_expected_evaluations = 1000
            maximum_expected_bytes = 1000
            """
        ) + _execution_block()
        with self.assertRaisesRegex(
            ConfigError, r"\[sensitivity_analysis\] is only allowed"
        ):
            parse_experiment_config(tomllib.loads(text))


if __name__ == "__main__":
    unittest.main()
