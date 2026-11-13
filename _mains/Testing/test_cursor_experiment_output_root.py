"""
R7 / ALG8.1: output_root resolution for experiment runners.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.ioh_core.experiment import run_configured_experiment, run_experiment


class TestExperimentOutputRoot(unittest.TestCase):
    def test_explicit_output_root_is_resolved_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nested" / "out"
            # Minimal stubs: empty problem_bundle skips the logger loop body
            # after logger construction — use a mock Analyzer.
            fake_logger = MagicMock()
            with patch("core.ioh_core.experiment.ioh.logger.Analyzer", return_value=fake_logger):
                summary = run_experiment(
                    problem_bundle={},
                    algorithm=MagicMock(name="alg", info=""),
                    n_runs=0,
                    name="r7_out_root",
                    zip_results=False,
                    return_summary=True,
                    output_root=root,
                )
            self.assertEqual(summary["output_root"], str(root.resolve()))
            self.assertTrue(Path(summary["output_root"]).is_absolute())
            self.assertTrue(root.exists())

    def test_configured_explicit_output_root_is_resolved_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nested" / "configured"
            summary = run_configured_experiment(
                {},
                algorithm_name="RANDOM_SEARCH",
                algorithm_parameters={"max_evaluations": 2},
                repetitions=1,
                seed_base=0,
                slab_type="hp_slab",
                name="alg81_out_root",
                zip_results=False,
                return_summary=True,
                output_root=root,
            )
            self.assertEqual(summary["output_root"], str(root.resolve()))
            self.assertTrue(Path(summary["output_root"]).is_absolute())
            self.assertTrue(root.exists())
            self.assertEqual(summary["runs"], [])


if __name__ == "__main__":
    unittest.main()
