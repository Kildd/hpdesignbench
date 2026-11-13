"""
Phase A3 gate: strict experiment TOML loading and validation.
"""
from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.config import (
    ConfigError,
    experiment_root_for,
    load_experiment_config,
    parse_experiment_config,
)


def _valid_toml(**overrides: str) -> str:
    """
    Return a minimal valid experiment TOML document.

    Parameters
    ----------
    **overrides : str
        Optional replacement sections keyed by logical name:
        ``schema``, ``experiment``, ``algorithms``, ``repetitions``,
        ``execution``, ``thread``, ``local``, ``slurm``.

    Returns
    -------
    str
        Complete TOML text.
    """
    parts = {
        "schema": 'schema_version = 1',
        "experiment": textwrap.dedent(
            """\
            [experiment]
            id = "hp_slab_smoke"
            workflow = "optimization_runs"
            slab_type = "hp_slab"
            problem_ids = ["L6750_B2"]
            output_root = "experiments"
            """
        ),
        "algorithms": textwrap.dedent(
            """\
            [[experiment.algorithms]]
            name = "RANDOM_SEARCH"
            max_evaluations = 10
            """
        ),
        "repetitions": textwrap.dedent(
            """\
            [experiment.repetitions]
            count = 2
            seed_base = 42
            """
        ),
        "execution": textwrap.dedent(
            """\
            [execution]
            backend = "local"
            max_parallel_tasks = 2
            timeout_seconds = 2700
            cpus_per_task = 1
            """
        ),
        "thread": textwrap.dedent(
            """\
            [execution.thread_environment]
            OMP_NUM_THREADS = "1"
            MKL_NUM_THREADS = "1"
            OPENBLAS_NUM_THREADS = "1"
            BLIS_NUM_THREADS = "1"
            NUMEXPR_NUM_THREADS = "1"
            """
        ),
        "local": textwrap.dedent(
            """\
            [execution.local]
            python_executable = ""
            """
        ),
        "slurm": textwrap.dedent(
            """\
            [execution.slurm]
            python_executable = "/work/venv/bin/python"
            partition = ""
            account = ""
            memory = "8G"
            time_limit = "00:45:00"
            array_concurrency = 64
            maximum_array_size = 1000
            """
        ),
    }
    parts.update(overrides)
    return "\n".join(parts[key] for key in (
        "schema",
        "experiment",
        "algorithms",
        "repetitions",
        "execution",
        "thread",
        "local",
        "slurm",
    ))


class TestExperimentConfigLoading(unittest.TestCase):
    """Deterministic TOML loading and path independence."""

    def test_load_valid_config_is_deterministic(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "smoke.toml"
            path.write_text(_valid_toml(), encoding="utf-8")
            first = load_experiment_config(path)
            second = load_experiment_config(path)

            self.assertEqual(first.schema_version, 1)
            self.assertEqual(first.experiment.id, "hp_slab_smoke")
            self.assertEqual(first.experiment.problem_ids, ("L6750_B2",))
            self.assertEqual(first.experiment.algorithms[0].name, "RANDOM_SEARCH")
            self.assertEqual(first.execution.backend, "local")
            self.assertEqual(
                first.execution.thread_environment["OMP_NUM_THREADS"],
                "1",
            )
            self.assertEqual(first.source_path, path.resolve())
            self.assertEqual(first, second)

    def test_path_independence_of_logical_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            config_dir.mkdir()
            path = config_dir / "smoke.toml"
            path.write_text(_valid_toml(), encoding="utf-8")

            cfg_a = load_experiment_config(path)
            nested = root / "nested"
            nested.mkdir()
            cfg_b = load_experiment_config(path)

            self.assertEqual(cfg_a.experiment, cfg_b.experiment)
            self.assertEqual(cfg_a.execution, cfg_b.execution)
            self.assertEqual(
                experiment_root_for(cfg_a, cwd=root),
                (root / "experiments" / "runs" / "optimization" / "hp_slab_smoke").resolve(),
            )
            self.assertEqual(
                experiment_root_for(cfg_a, cwd=nested),
                (nested / "experiments" / "runs" / "optimization" / "hp_slab_smoke").resolve(),
            )


class TestExperimentConfigValidation(unittest.TestCase):
    """Invalid configurations must fail loudly."""

    def _parse(self, text: str):
        import tomllib

        return parse_experiment_config(tomllib.loads(text))

    def test_unknown_schema_version(self) -> None:
        with self.assertRaisesRegex(ConfigError, "schema_version"):
            self._parse(_valid_toml(schema="schema_version = 99"))

    def test_unknown_workflow_and_backend(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown workflow"):
            self._parse(
                _valid_toml(
                    experiment=textwrap.dedent(
                        """\
                        [experiment]
                        id = "x"
                        workflow = "not_a_workflow"
                        slab_type = "hp_slab"
                        problem_ids = ["L6750_B2"]
                        output_root = "experiments"
                        """
                    )
                )
            )
        with self.assertRaisesRegex(ConfigError, "unknown backend"):
            self._parse(
                _valid_toml(
                    execution=textwrap.dedent(
                        """\
                        [execution]
                        backend = "batch"
                        max_parallel_tasks = 1
                        cpus_per_task = 1
                        """
                    )
                )
            )

    def test_missing_experiment_id_and_repetitions(self) -> None:
        with self.assertRaisesRegex(ConfigError, "experiment.id"):
            self._parse(
                _valid_toml(
                    experiment=textwrap.dedent(
                        """\
                        [experiment]
                        id = ""
                        workflow = "optimization_runs"
                        slab_type = "hp_slab"
                        problem_ids = ["L6750_B2"]
                        output_root = "experiments"
                        """
                    )
                )
            )
        with self.assertRaisesRegex(ConfigError, "repetitions.count"):
            self._parse(
                _valid_toml(
                    repetitions=textwrap.dedent(
                        """\
                        [experiment.repetitions]
                        count = 0
                        seed_base = 42
                        """
                    )
                )
            )

    def test_invalid_parallelism_and_timeout(self) -> None:
        with self.assertRaisesRegex(ConfigError, "max_parallel_tasks"):
            self._parse(
                _valid_toml(
                    execution=textwrap.dedent(
                        """\
                        [execution]
                        backend = "local"
                        max_parallel_tasks = 0
                        cpus_per_task = 1
                        """
                    )
                )
            )
        with self.assertRaisesRegex(ConfigError, "timeout_seconds"):
            self._parse(
                _valid_toml(
                    execution=textwrap.dedent(
                        """\
                        [execution]
                        backend = "local"
                        max_parallel_tasks = 1
                        timeout_seconds = 0
                        cpus_per_task = 1
                        """
                    )
                )
            )

    def test_incomplete_slurm_configuration(self) -> None:
        with self.assertRaisesRegex(ConfigError, "partition"):
            self._parse(
                _valid_toml(
                    execution=textwrap.dedent(
                        """\
                        [execution]
                        backend = "slurm"
                        max_parallel_tasks = 1
                        cpus_per_task = 1
                        """
                    ),
                    slurm=textwrap.dedent(
                        """\
                        [execution.slurm]
                        python_executable = "/work/venv/bin/python"
                        partition = ""
                        memory = "8G"
                        time_limit = "00:45:00"
                        """
                    ),
                )
            )

    def test_unknown_keys_and_duplicate_definitions(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown keys"):
            self._parse(_valid_toml() + "\nextra = 1\n")
        with self.assertRaisesRegex(ConfigError, "duplicate task-producing problem"):
            self._parse(
                _valid_toml(
                    experiment=textwrap.dedent(
                        """\
                        [experiment]
                        id = "x"
                        workflow = "optimization_runs"
                        slab_type = "hp_slab"
                        problem_ids = ["1", "1"]
                        output_root = "experiments"
                        """
                    )
                )
            )
        with self.assertRaisesRegex(ConfigError, "duplicate task-producing algorithm"):
            self._parse(
                _valid_toml(
                    algorithms=textwrap.dedent(
                        """\
                        [[experiment.algorithms]]
                        name = "RANDOM_SEARCH"
                        max_evaluations = 10

                        [[experiment.algorithms]]
                        name = "RANDOM_SEARCH"
                        max_evaluations = 20
                        """
                    )
                )
            )

    def test_algorithm_parameters_default_to_empty_mapping(self) -> None:
        from types import MappingProxyType

        config = self._parse(_valid_toml())
        parameters = config.experiment.algorithms[0].parameters
        self.assertEqual(dict(parameters), {})
        self.assertIsInstance(parameters, MappingProxyType)
        self.assertIsNone(config.experiment.algorithms[0].repetitions)

    def test_algorithm_repetitions_override_parses(self) -> None:
        config = self._parse(
            _valid_toml(
                algorithms=textwrap.dedent(
                    """\
                    [[experiment.algorithms]]
                    name = "RANDOM_SEARCH"
                    max_evaluations = 10
                    repetitions = 3
                    """
                )
            )
        )
        self.assertEqual(config.experiment.algorithms[0].repetitions, 3)

    def test_algorithm_repetitions_override_rejects_invalid_values(self) -> None:
        for value in ("0", "-1", "true", "1.5", '"2"'):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError):
                    self._parse(
                        _valid_toml(
                            algorithms=textwrap.dedent(
                                f"""\
                                [[experiment.algorithms]]
                                name = "RANDOM_SEARCH"
                                max_evaluations = 10
                                repetitions = {value}
                                """
                            )
                        )
                    )

    def test_algorithm_repetition_singular_is_unknown(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown keys"):
            self._parse(
                _valid_toml(
                    algorithms=textwrap.dedent(
                        """\
                        [[experiment.algorithms]]
                        name = "RANDOM_SEARCH"
                        max_evaluations = 10
                        repetition = 1
                        """
                    )
                )
            )

    def test_algorithm_parameters_are_defensively_copied(self) -> None:
        from core.execution_core.config import AlgorithmConfig

        raw = {"nested": {"alpha": 1}, "values": [1, 2]}
        config = AlgorithmConfig(
            name="RANDOM_SEARCH",
            max_evaluations=10,
            parameters=raw,
        )
        raw["nested"]["alpha"] = 99
        raw["values"].append(3)
        raw["extra"] = True
        self.assertEqual(
            dict(config.parameters),
            {"nested": {"alpha": 1}, "values": [1, 2]},
        )
        with self.assertRaises(TypeError):
            config.parameters["extra"] = True  # type: ignore[index]

    def test_algorithm_parameters_accept_nested_json_values(self) -> None:
        config = self._parse(
            _valid_toml(
                algorithms=textwrap.dedent(
                    """\
                    [[experiment.algorithms]]
                    name = "RANDOM_SEARCH"
                    max_evaluations = 10
                    [experiment.algorithms.parameters]
                    flag = true
                    count = 3
                    rate = 0.5
                    label = "ga"
                    values = [1, 2, 3]
                    [experiment.algorithms.parameters.inner]
                    x = "y"
                    """
                )
            )
        )
        self.assertEqual(
            dict(config.experiment.algorithms[0].parameters),
            {
                "flag": True,
                "count": 3,
                "rate": 0.5,
                "label": "ga",
                "values": [1, 2, 3],
                "inner": {"x": "y"},
            },
        )

    def test_algorithm_parameters_reject_non_json_values(self) -> None:
        with self.assertRaisesRegex(ConfigError, "date/time"):
            self._parse(
                _valid_toml(
                    algorithms=textwrap.dedent(
                        """\
                        [[experiment.algorithms]]
                        name = "RANDOM_SEARCH"
                        max_evaluations = 10
                        [experiment.algorithms.parameters]
                        when = 2024-01-02T03:04:05Z
                        """
                    )
                )
            )

    def test_algorithm_parameters_reject_non_finite_floats(self) -> None:
        from core.execution_core.config import AlgorithmConfig

        with self.assertRaisesRegex(ConfigError, "finite"):
            AlgorithmConfig(
                name="RANDOM_SEARCH",
                max_evaluations=1,
                parameters={"rate": float("nan")},
            )
        with self.assertRaisesRegex(ConfigError, "finite"):
            AlgorithmConfig(
                name="RANDOM_SEARCH",
                max_evaluations=1,
                parameters={"rate": float("inf")},
            )


if __name__ == "__main__":
    unittest.main()
