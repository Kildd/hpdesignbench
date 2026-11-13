"""Operator ROOT layout: runs vs results (ADR 0008 / 0009)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.output_layout import (
    ILS_CATALOGUE_PREFIX_TOKEN,
    LayoutError,
    NESTED_PREFIX_DIAG_DIRNAME,
    assert_not_legacy_derived_publication,
    default_nested_prefix_dir,
    default_optimization_plot_dir,
    default_structural_library_dir,
    experiment_dir_for,
    factorial_publication_dir,
    ils_pair_token,
    la_family_parent_dir,
    layout_root_from_experiment_dir,
    layout_root_from_output_root,
    require_results_publication,
    results_root_for,
    saltelli_publication_dir,
    write_layout_sidecar,
)


class TestOutputLayout(unittest.TestCase):
    def test_relative_root_places_exhaustive_run_under_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            root = layout_root_from_output_root("experiments", cwd=cwd)
            experiment = experiment_dir_for(
                layout_root=root,
                workflow="brute_force",
                experiment_id="bf_p1",
            )
            self.assertEqual(
                experiment,
                (cwd / "experiments" / "runs" / "exhaustive" / "bf_p1").resolve(),
            )

    def test_absolute_root_and_other_run_buckets(self) -> None:
        with TemporaryDirectory() as tmp:
            root = (Path(tmp) / "data").resolve()
            self.assertEqual(
                layout_root_from_output_root(root, cwd=Path(tmp)),
                root,
            )
            self.assertEqual(
                experiment_dir_for(
                    layout_root=root,
                    workflow="sampling",
                    experiment_id="s1",
                ),
                root / "runs" / "sampling" / "s1",
            )
            self.assertEqual(
                experiment_dir_for(
                    layout_root=root,
                    workflow="sensitivity_analysis",
                    experiment_id="sa1",
                ),
                root / "runs" / "sensitivity" / "sa1",
            )
            self.assertEqual(
                experiment_dir_for(
                    layout_root=root,
                    workflow="optimization_runs",
                    experiment_id="opt1",
                ),
                root / "runs" / "optimization" / "opt1",
            )
            self.assertEqual(
                experiment_dir_for(
                    layout_root=root,
                    workflow="structural_library",
                    experiment_id="sl1",
                ),
                root / "runs" / "structural_library" / "sl1",
            )
            self.assertEqual(
                default_structural_library_dir(root),
                root / "structural_library" / "baseline",
            )

    def test_sidecar_wins_over_legacy_parent(self) -> None:
        with TemporaryDirectory() as tmp:
            layout_root = (Path(tmp) / "root").resolve()
            experiment = Path(tmp) / "elsewhere" / "exp"
            experiment.mkdir(parents=True)
            write_layout_sidecar(experiment, layout_root)
            self.assertEqual(layout_root_from_experiment_dir(experiment), layout_root)

    def test_ils_pair_token_and_shelves(self) -> None:
        with TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "runs" / "exhaustive" / "bf_p1"
            experiment.mkdir(parents=True)
            sampling = Path(tmp) / "runs" / "sampling" / "samp_p1"
            results = results_root_for(layout_root_from_experiment_dir(experiment))
            ils = la_family_parent_dir(
                experiment,
                shelf="ils",
                sampling_dir=sampling,
                start_policy="sobol_net",
            )
            self.assertEqual(
                ils,
                results / "la" / "sampling" / "bf_p1__samp_p1",
            )
            self.assertEqual(
                ils_pair_token(
                    brute_force_id="bf_p1",
                    sampling_id=None,
                    start_policy="catalogue_prefix",
                ),
                f"bf_p1__{ILS_CATALOGUE_PREFIX_TOKEN}",
            )
            with self.assertRaises(LayoutError):
                ils_pair_token(
                    brute_force_id="bf_p1",
                    sampling_id=None,
                    start_policy="sobol_net",
                )
            diag = default_nested_prefix_dir(sampling)
            self.assertEqual(diag.name, NESTED_PREFIX_DIAG_DIRNAME)
            self.assertEqual(diag.parent.name, "samp_p1")
            self.assertEqual(
                saltelli_publication_dir(results, "sa1"),
                results / "sensitivity-analysis" / "saltelli" / "sa1",
            )
            self.assertEqual(
                factorial_publication_dir(results, "bf_p1"),
                results / "sensitivity-analysis" / "factorial" / "bf_p1",
            )

    def test_refuses_legacy_aggregate_analysis_la(self) -> None:
        with TemporaryDirectory() as tmp:
            forbidden = Path(tmp) / "exp" / "aggregate" / "analysis" / "la"
            forbidden.mkdir(parents=True)
            with self.assertRaises(LayoutError):
                assert_not_legacy_derived_publication(forbidden)
            with self.assertRaises(ValueError):
                require_results_publication(forbidden, error_type=ValueError)


def _family(root: Path, name: str, problem_ids: list[str]) -> None:
    family = root / "structural_library" / name
    family.mkdir(parents=True)
    (family / "manifest.json").write_text(
        json.dumps(
            {
                "identity": {
                    "problem_id": problem_ids[0],
                    "compatible_problem_ids": problem_ids,
                }
            }
        ),
        encoding="utf-8",
    )


class TestOptimizationPlotDirectory(unittest.TestCase):
    def test_one_experiment_uses_its_id(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "runs" / "optimization" / "penalty_reference"
            experiment.mkdir(parents=True)
            self.assertEqual(
                default_optimization_plot_dir([experiment]),
                root / "results" / "optimization" / "penalty_reference",
            )

    def test_several_experiments_join_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "runs" / "optimization" / "penalty_reference"
            second = root / "runs" / "optimization" / "penalty_validation"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            self.assertEqual(
                default_optimization_plot_dir([first, second]),
                root / "results" / "optimization" / "penalty_reference__penalty_validation",
            )

    def test_different_roots_are_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            first = base / "a" / "runs" / "optimization" / "one"
            second = base / "b" / "runs" / "optimization" / "two"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            with self.assertRaises(LayoutError):
                default_optimization_plot_dir([first, second])


class TestStructuralLibraryFamily(unittest.TestCase):
    def test_single_baseline_is_the_default(self) -> None:
        from core.ioh_core.structural_artifacts.sealing import (
            resolve_structural_library_family,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _family(root, "baseline", ["L6750_B2"])
            (root / "structural_library" / "manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            chosen = resolve_structural_library_family(root, ["L8100_B2"])
            self.assertEqual(chosen, default_structural_library_dir(root))

    def test_problem_selects_exactly_one_family(self) -> None:
        from core.ioh_core.structural_artifacts.sealing import (
            StructuralArtifactsSealError,
            resolve_structural_library_family,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _family(root, "baseline", ["L6750_B2", "L8100_B2"])
            _family(root, "fire_r60", ["L12000_F60"])
            self.assertEqual(
                resolve_structural_library_family(root, ["L8100_B2"]),
                root / "structural_library" / "baseline",
            )
            self.assertEqual(
                resolve_structural_library_family(root, ["L12000_F60"]),
                root / "structural_library" / "fire_r60",
            )
            self.assertEqual(
                resolve_structural_library_family(
                    root, ["L6750_B2", "L8100_B2"]
                ),
                root / "structural_library" / "baseline",
            )
            with self.assertRaises(StructuralArtifactsSealError):
                resolve_structural_library_family(root, ["L9999_X"])
            with self.assertRaises(StructuralArtifactsSealError):
                resolve_structural_library_family(
                    root, ["L6750_B2", "L12000_F60"]
                )
            _family(root, "fire_r90", ["L12000_F60"])
            with self.assertRaises(StructuralArtifactsSealError):
                resolve_structural_library_family(root, ["L12000_F60"])


if __name__ == "__main__":
    unittest.main()
