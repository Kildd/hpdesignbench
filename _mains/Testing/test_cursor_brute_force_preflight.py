"""
Phase B3 gate: brute-force preflight estimates and threshold blocking.
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

from core.execution_core.config import parse_experiment_config
from core.ioh_core.workflows.brute_force.preflight import (
    BruteForcePreflightError,
    deterministic_probe_indices,
    run_preflight,
)


def _config_text(
    *,
    acknowledge: bool,
    max_rows: int,
    max_bytes: int = 50_000_000_000,
) -> str:
    """
    Return a minimal brute-force TOML document.
    """
    return textwrap.dedent(
        f"""\
        schema_version = 1

        [experiment]
        id = "bf_preflight"
        workflow = "brute_force"
        slab_type = "hp_slab"
        problem_ids = ["L6750_B2"]
        output_root = "experiments"

        [brute_force]
        rows_per_shard = 10000
        acknowledge_large_run = {"true" if acknowledge else "false"}
        maximum_expected_rows = {max_rows}
        maximum_expected_bytes = {max_bytes}

        [execution]
        backend = "local"
        max_parallel_tasks = 4
        cpus_per_task = 1

        [execution.local]
        python_executable = ""
        """
    )


def _parse(text: str):
    """
    Parse TOML text into an ExperimentConfig.
    """
    return parse_experiment_config(tomllib.loads(text))


class TestBruteForcePreflight(unittest.TestCase):
    """Probes, blocking, and no Parquet side effects."""

    def test_deterministic_probes(self) -> None:
        probes = deterministic_probe_indices(100)
        self.assertEqual(probes[0], 0)
        self.assertEqual(probes[-1], 99)
        self.assertEqual(probes, tuple(sorted(set(probes))))

    def test_blocks_without_acknowledgement(self) -> None:
        config = _parse(_config_text(acknowledge=False, max_rows=1_000))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(BruteForcePreflightError):
                run_preflight(
                    config,
                    probe_path=root,
                    evaluate=lambda _idx: None,
                )
            self.assertEqual(list(root.rglob("*.parquet")), [])

    def test_allows_with_acknowledgement(self) -> None:
        config = _parse(_config_text(acknowledge=True, max_rows=1_000))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = run_preflight(
                config,
                probe_path=root,
                evaluate=lambda _idx: None,
                raise_if_blocked=True,
            )
            self.assertFalse(report.blocked)
            self.assertEqual(report.cardinality, 53_374_464)
            self.assertEqual(report.finish_group_size, 24)
            self.assertEqual(report.rows_per_shard, 9984)
            self.assertGreater(report.shard_count, 1)
            self.assertEqual(report.probe_indices[0], 0)
            self.assertEqual(
                report.probe_indices[-1], report.cardinality - 1
            )
            self.assertGreater(report.mean_evaluation_time_s, 0.0)
            self.assertGreater(report.estimated_serial_time_s, 0.0)
            self.assertAlmostEqual(
                report.estimated_parallel_time_s,
                report.estimated_serial_time_s / 4.0,
                places=9,
            )
            self.assertTrue(
                any("acknowledge_large_run=true" in w for w in report.warnings)
            )
            self.assertEqual(list(root.rglob("*.parquet")), [])

    def test_report_without_raising_when_blocked(self) -> None:
        config = _parse(_config_text(acknowledge=False, max_rows=1_000))
        report = run_preflight(
            config,
            evaluate=lambda _idx: None,
            raise_if_blocked=False,
        )
        self.assertTrue(report.blocked)
        self.assertTrue(report.block_reasons)


if __name__ == "__main__":
    unittest.main()
