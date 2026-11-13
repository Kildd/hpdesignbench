"""
Phase V1.1 gate: convergence aggregation and Hansen ERT helpers.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from core.visualization_core.optimization.aggregate import (
    AggregationError,
    aggregate_convergence,
    hansen_ert,
    hansen_ert_from_outcomes,
    hansen_ert_grid,
    hansen_ert_value,
    hansen_ert_within_budget,
    warm_instance_deployment_ert,
    median_iqr_band,
    runtime_sample,
    select_best_run,
    success_probability_curve,
    algorithm_feasibility_summaries,
    algorithm_target_summaries,
)
from core.visualization_core.optimization.models import ExperimentBundle, RunTrajectory
from core.visualization_core.optimization.trajectories import (
    best_so_far,
    feasible_share,
    first_feasible_evaluation,
)


def _run(
    *,
    task_index: int,
    y_p: list[float],
    max_evaluations: int,
    global_optimum: float = 100.0,
    target_reference: str = "all",
    overlay_label: str = "ALG",
    repetition: int = 0,
    feasible_mask: list[bool] | None = None,
) -> RunTrajectory:
    values = np.asarray(y_p, dtype=float)
    evaluations = np.arange(1, len(y_p) + 1, dtype=int)
    mask = (
        None
        if feasible_mask is None
        else np.asarray(feasible_mask, dtype=bool)
    )
    return RunTrajectory(
        experiment_dir=Path("synthetic"),
        overlay_label=overlay_label,
        algorithm_name=overlay_label,
        problem_id="L6750_B2",
        task_index=task_index,
        repetition=repetition,
        evaluations=evaluations,
        y_p=values,
        y_p_best_so_far_all=best_so_far(values),
        constraint_utilizations={},
        feasible_mask=mask,
        max_evaluations=max_evaluations,
        global_optimum=global_optimum,
        target_reference=target_reference,
        target_hit=None,
        evaluations_to_target=None,
        wall_time_s=None,
        dat_path=Path("synthetic.dat"),
    )


class ConvergenceAggregationTests(unittest.TestCase):
    def test_median_and_mean_use_b_common(self) -> None:
        runs = (
            _run(task_index=0, y_p=[9.0, 7.0, 5.0, 4.0], max_evaluations=10),
            _run(task_index=1, y_p=[8.0, 6.0, 6.0], max_evaluations=10),
            _run(task_index=2, y_p=[10.0, 9.0, 3.0, 2.0, 1.0], max_evaluations=10),
        )
        # B_common = 3 → columns at e=1,2,3:
        # [9,8,10], [7,6,9], [5,6,3]
        median = aggregate_convergence(
            runs, display="median", extend_short_runs=False
        )
        mean = aggregate_convergence(
            runs, display="mean", extend_short_runs=False
        )
        self.assertEqual(len(median), 1)
        self.assertEqual(len(mean), 1)
        np.testing.assert_array_equal(median[0].evaluations, [1, 2, 3])
        np.testing.assert_allclose(median[0].y_p_best_so_far, [9.0, 7.0, 5.0])
        np.testing.assert_allclose(
            mean[0].y_p_best_so_far,
            [9.0, 22.0 / 3.0, 14.0 / 3.0],
        )

    def test_extend_short_runs_carry_forward(self) -> None:
        runs = (
            _run(task_index=0, y_p=[5.0, 4.0], max_evaluations=5),
            _run(task_index=1, y_p=[6.0, 3.0, 2.0], max_evaluations=5),
        )
        series = aggregate_convergence(
            runs, display="mean", extend_short_runs=True
        )
        np.testing.assert_array_equal(series[0].evaluations, [1, 2, 3])
        # rows: [5,4,4] and [6,3,2] → means [5.5, 3.5, 3.0]
        np.testing.assert_allclose(series[0].y_p_best_so_far, [5.5, 3.5, 3.0])

    def test_best_selects_lowest_final_not_envelope(self) -> None:
        runs = (
            _run(task_index=0, y_p=[10.0, 1.0, 9.0], max_evaluations=5),
            _run(task_index=1, y_p=[8.0, 7.0, 2.0], max_evaluations=5),
        )
        # finals: run0 → 1.0, run1 → 2.0 → winner is run0 even though run1 is
        # better at e=1; envelope would be [8,1,1] which must NOT be returned.
        winner = select_best_run(runs)
        self.assertEqual(winner.task_index, 0)
        series = aggregate_convergence(runs, display="best")
        np.testing.assert_allclose(series[0].y_p_best_so_far, [10.0, 1.0, 1.0])

    def test_best_tie_breaks_on_task_index(self) -> None:
        runs = (
            _run(task_index=2, y_p=[5.0, 3.0], max_evaluations=4),
            _run(task_index=1, y_p=[4.0, 3.0], max_evaluations=4),
        )
        self.assertEqual(select_best_run(runs).task_index, 1)

    def test_all_returns_individual_curves(self) -> None:
        runs = (
            _run(task_index=0, y_p=[3.0, 2.0], max_evaluations=3),
            _run(task_index=1, y_p=[4.0], max_evaluations=3),
        )
        series = aggregate_convergence(runs, display="all")
        self.assertEqual(len(series), 2)
        self.assertEqual(series[0].kind, "individual")
        self.assertEqual(series[1].label, "ALG#1")

    def test_median_iqr_band(self) -> None:
        runs = (
            _run(task_index=0, y_p=[9.0, 7.0, 5.0], max_evaluations=10),
            _run(task_index=1, y_p=[8.0, 6.0, 6.0], max_evaluations=10),
            _run(task_index=2, y_p=[10.0, 9.0, 3.0], max_evaluations=10),
        )
        band = median_iqr_band(runs, extend_short_runs=False)
        np.testing.assert_array_equal(band.evaluations, [1, 2, 3])
        np.testing.assert_allclose(band.median, [9.0, 7.0, 5.0])
        np.testing.assert_allclose(band.q25, [8.5, 6.5, 4.0])
        np.testing.assert_allclose(band.q75, [9.5, 8.0, 5.5])
        with self.assertRaises(AggregationError):
            median_iqr_band(runs[:1])


class HansenErtTests(unittest.TestCase):
    def test_hansen_ert_with_failures(self) -> None:
        # y* = 100 → 1% threshold = 101
        # run0 hits at 2, run1 misses after 3 evaluations, run2 hits at 4.
        # The allocated ceiling is 10 and is not charged.
        # ERT = (2 + 3 + 4) / 2 = 4.5
        runs = (
            _run(task_index=0, y_p=[110.0, 100.5], max_evaluations=10),
            _run(task_index=1, y_p=[120.0, 115.0, 110.0], max_evaluations=10),
            _run(
                task_index=2,
                y_p=[130.0, 120.0, 110.0, 100.0],
                max_evaluations=10,
            ),
        )
        result = hansen_ert(runs, 1.0, target_reference="all")
        self.assertEqual(result.n_success, 2)
        self.assertEqual(result.n_runs, 3)
        self.assertAlmostEqual(result.sum_t, 9.0)
        self.assertAlmostEqual(result.ert, 4.5)

    def test_hansen_ert_undefined_when_zero_success(self) -> None:
        runs = (
            _run(task_index=0, y_p=[200.0, 150.0], max_evaluations=7),
            _run(task_index=1, y_p=[180.0, 160.0], max_evaluations=7),
        )
        result = hansen_ert(runs, 1.0, target_reference="all")
        self.assertIsNone(result.ert)
        self.assertFalse(result.defined)
        self.assertEqual(result.n_success, 0)
        self.assertAlmostEqual(result.sum_t, 4.0)

    def test_hansen_ert_not_mean_of_successes_only(self) -> None:
        runs = (
            _run(task_index=0, y_p=[100.5], max_evaluations=20),
            _run(task_index=1, y_p=[200.0, 190.0], max_evaluations=20),
        )
        result = hansen_ert(runs, 1.0, target_reference="all")
        # Successes-only mean would be 1.0. The miss logged 2 evaluations.
        self.assertAlmostEqual(result.ert, 3.0)

    def test_budget_ert_matches_hansen_when_the_budget_does_not_bind(self) -> None:
        runs = (
            _run(task_index=0, y_p=[110.0, 100.5], max_evaluations=10),
            _run(task_index=1, y_p=[120.0, 115.0, 110.0], max_evaluations=10),
            _run(
                task_index=2,
                y_p=[130.0, 120.0, 110.0, 100.0],
                max_evaluations=10,
            ),
        )
        full = hansen_ert(runs, 1.0, target_reference="all")
        capped = hansen_ert_within_budget(
            runs, 1.0, 10, target_reference="all"
        )
        self.assertEqual(capped.n_success, full.n_success)
        self.assertAlmostEqual(capped.ert, full.ert)
        self.assertAlmostEqual(capped.sum_hits, 6.0)
        self.assertAlmostEqual(capped.failure_cost, 3.0)
        self.assertEqual(capped.failure_cap, 3)

    def test_budget_ert_drops_hits_after_the_selected_budget(self) -> None:
        runs = (
            _run(task_index=0, y_p=[110.0, 100.5], max_evaluations=10),
            _run(task_index=1, y_p=[120.0, 115.0, 110.0], max_evaluations=10),
            _run(
                task_index=2,
                y_p=[130.0, 120.0, 110.0, 100.0],
                max_evaluations=10,
            ),
        )
        # Hit at 2 still counts. Hit at 4 is now a miss, and each miss costs 3.
        capped = hansen_ert_within_budget(
            runs, 1.0, 3, target_reference="all"
        )
        self.assertEqual(capped.n_success, 1)
        self.assertAlmostEqual(capped.sum_hits, 2.0)
        self.assertAlmostEqual(capped.failure_cost, 6.0)
        self.assertAlmostEqual(capped.ert, 8.0)
        self.assertEqual(capped.failure_cap, 3)
        none = hansen_ert_within_budget(
            runs, 1.0, 1, target_reference="all"
        )
        self.assertIsNone(none.ert)
        self.assertEqual(none.n_success, 0)
        self.assertAlmostEqual(none.sum_t, 3.0)

    def test_budget_ert_keeps_a_shorter_run_at_its_own_budget(self) -> None:
        result = hansen_ert_from_outcomes(
            [2, None],
            [4, 4],
            percent=1.0,
            budget=100,
        )
        self.assertAlmostEqual(result.ert, 6.0)
        self.assertAlmostEqual(result.failure_cost, 4.0)
        self.assertEqual(result.failure_cap, 4)
        with self.assertRaises(AggregationError):
            hansen_ert_from_outcomes([], [], percent=1.0, budget=10)
        with self.assertRaises(AggregationError):
            hansen_ert_from_outcomes([1], [10, 10], percent=1.0, budget=10)
        with self.assertRaises(AggregationError):
            hansen_ert_from_outcomes([1], [10], percent=1.0, budget=0)

    def test_ert_grid_and_runtime_filters(self) -> None:
        runs = (
            _run(task_index=0, y_p=[110.0, 100.5], max_evaluations=10),
            _run(task_index=1, y_p=[120.0, 115.0], max_evaluations=10),
        )
        grid = hansen_ert_grid(runs, (1.0, 10.0), target_reference="all")
        self.assertEqual(len(grid), 2)
        successes, n_s, n = runtime_sample(
            runs, 1.0, success_filter="successes", target_reference="all"
        )
        all_runs, n_s2, n2 = runtime_sample(
            runs, 1.0, success_filter="all_runs", target_reference="all"
        )
        self.assertEqual((n_s, n), (1, 2))
        self.assertEqual((n_s2, n2), (1, 2))
        np.testing.assert_allclose(successes, [2.0])
        np.testing.assert_allclose(all_runs, [2.0, 2.0])

    def test_full_budget_miss_and_legitimate_early_miss(self) -> None:
        full = hansen_ert_from_outcomes(
            [100, None],
            [100, 15000],
            percent=1.0,
            budget=15000,
        )
        self.assertAlmostEqual(full.ert, (100 + 15000) / 1)
        early = hansen_ert_from_outcomes(
            [None],
            [926],
            percent=1.0,
            budget=15000,
        )
        self.assertIsNone(early.ert)
        self.assertAlmostEqual(early.sum_t, 926.0)
        mixed = hansen_ert_from_outcomes(
            [4022, None],
            [4022, 926],
            percent=1.0,
            budget=15000,
        )
        self.assertAlmostEqual(mixed.ert, 4022 + 926)

    def test_reduced_view_budget_censors_the_conducted_count(self) -> None:
        early_view = hansen_ert_from_outcomes(
            [None, None],
            [926, 15000],
            percent=1.0,
            budget=2000,
        )
        self.assertIsNone(early_view.ert)
        self.assertAlmostEqual(early_view.failure_cost, 926 + 2000)
        late_hit = hansen_ert_from_outcomes(
            [5000],
            [15000],
            percent=1.0,
            budget=2000,
        )
        self.assertIsNone(late_hit.ert)
        self.assertAlmostEqual(late_hit.sum_t, 2000.0)

    def test_first_hit_is_not_the_final_evaluation_count(self) -> None:
        runs = (
            _run(
                task_index=0,
                y_p=[110.0, 100.5, 100.4, 100.2, 100.1],
                max_evaluations=15000,
            ),
        )
        result = hansen_ert(runs, 1.0, target_reference="all")
        self.assertAlmostEqual(result.ert, 2.0)
        self.assertEqual(runs[0].n_evaluations, 5)

    def test_zero_success_and_invalid_runs(self) -> None:
        undefined = hansen_ert_value([None, None], [15000, 926], budget=15000)
        self.assertIsNone(undefined)
        with self.assertRaises(AggregationError):
            hansen_ert_from_outcomes([5], [3], percent=1.0, budget=15000)
        with self.assertRaises(AggregationError):
            hansen_ert_from_outcomes([None], [0], percent=1.0, budget=15000)
        with self.assertRaises(AggregationError):
            hansen_ert_from_outcomes([], [], percent=1.0, budget=10)

    def test_pooled_ert_is_not_the_mean_of_instance_erts(self) -> None:
        first = hansen_ert_value([10, 30], [10, 30], budget=15000)
        second = hansen_ert_value([100, None], [100, 50], budget=15000)
        pooled = hansen_ert_value(
            [10, 30, 100, None],
            [10, 30, 100, 50],
            budget=15000,
        )
        self.assertAlmostEqual(first, 20.0)
        self.assertAlmostEqual(second, 150.0)
        self.assertAlmostEqual(pooled, (10 + 30 + 100 + 50) / 3)
        self.assertNotAlmostEqual(pooled, (first + second) / 2)

    def test_multiple_tolerances_share_the_conducted_miss_cost(self) -> None:
        # Group best is 100. The 10% threshold is 110 and the 1% threshold is 101.
        reached = _run(
            task_index=0,
            y_p=[130.0, 109.0, 108.0, 100.0],
            max_evaluations=20,
        )
        missed = _run(task_index=1, y_p=[140.0, 130.0], max_evaluations=20)
        grid = hansen_ert_grid(
            (reached, missed),
            (1.0, 10.0),
            target_reference="all",
        )
        self.assertEqual(grid[0].percent, 1.0)
        self.assertAlmostEqual(grid[0].ert, (4 + 2) / 1)
        self.assertEqual(grid[1].percent, 10.0)
        self.assertAlmostEqual(grid[1].ert, (2 + 2) / 1)

    def test_single_run_variants_and_warm_grid(self) -> None:
        success = hansen_ert_value([68], [68], budget=15000)
        failure = hansen_ert_value([None], [881], budget=15000)
        self.assertAlmostEqual(success, 68.0)
        self.assertIsNone(failure)
        self.assertAlmostEqual(warm_instance_deployment_ert(1261.9), 3961.9)
        self.assertIsNone(warm_instance_deployment_ert(None))
        self.assertAlmostEqual(
            warm_instance_deployment_ert(success),
            68.0 + 2700.0,
        )

    def test_success_curve_does_not_use_the_miss_cost(self) -> None:
        short_budget = (
            _run(task_index=0, y_p=[110.0, 100.5], max_evaluations=10),
            _run(task_index=1, y_p=[120.0, 115.0], max_evaluations=10),
        )
        long_budget = (
            _run(task_index=0, y_p=[110.0, 100.5], max_evaluations=15000),
            _run(task_index=1, y_p=[120.0, 115.0], max_evaluations=15000),
        )
        short_curve = success_probability_curve(short_budget, 1.0, target_reference="all")
        long_curve = success_probability_curve(long_budget, 1.0, target_reference="all")
        self.assertEqual(short_curve.n_success, long_curve.n_success)
        self.assertEqual(short_curve.n_runs, long_curve.n_runs)
        np.testing.assert_allclose(short_curve.hit_times, long_curve.hit_times)
        short_ert = hansen_ert(short_budget, 1.0, target_reference="all")
        long_ert = hansen_ert(long_budget, 1.0, target_reference="all")
        self.assertAlmostEqual(short_ert.ert, long_ert.ert)

    def test_success_probability_curve_asymptote_is_hit_rate(self) -> None:
        runs = (
            _run(task_index=0, y_p=[110.0, 100.5], max_evaluations=10),
            _run(task_index=1, y_p=[120.0, 115.0], max_evaluations=10),
        )
        curve = success_probability_curve(runs, 1.0, target_reference="all")
        self.assertEqual(curve.n_runs, 2)
        self.assertEqual(curve.n_success, 1)
        self.assertAlmostEqual(curve.success_rate, 0.5)
        np.testing.assert_allclose(curve.hit_times, [2.0])
        self.assertAlmostEqual(curve.x_max, 10.0)

    def test_algorithm_target_summaries_multi_percent(self) -> None:
        runs = (
            _run(task_index=0, y_p=[110.0, 100.5], max_evaluations=10),
            _run(task_index=1, y_p=[120.0, 115.0], max_evaluations=10),
        )
        bundle = ExperimentBundle(runs=runs, experiment_dirs=(Path("dummy"),))
        rows = algorithm_target_summaries(
            bundle, (1.0, 10.0), target_reference="all"
        )
        self.assertEqual(len(rows), 2)
        row1 = rows[0]
        self.assertEqual(row1.percent, 1.0)
        self.assertEqual(row1.n_runs, 2)
        self.assertEqual(row1.n_success, 1)
        self.assertAlmostEqual(row1.success_rate, 0.5)
        self.assertAlmostEqual(row1.e2t_min or 0.0, 2.0)
        self.assertIsNotNone(row1.ert)
        self.assertIsNone(row1.n_runs_with_feasible)
        self.assertIsNone(row1.mean_first_feasible_fe)
        self.assertIsNone(row1.mean_share_feasible)
        row10 = rows[1]
        self.assertEqual(row10.percent, 10.0)
        self.assertGreaterEqual(row10.n_success, row1.n_success)

    def test_first_feasible_and_share_means(self) -> None:
        runs = (
            _run(
                task_index=0,
                y_p=[110.0, 105.0, 100.0],
                max_evaluations=10,
                feasible_mask=[False, True, True],
            ),
            _run(
                task_index=1,
                y_p=[120.0, 115.0, 110.0, 100.0],
                max_evaluations=10,
                feasible_mask=[False, False, True, True],
            ),
            _run(
                task_index=2,
                y_p=[130.0, 125.0],
                max_evaluations=10,
                feasible_mask=[False, False],
            ),
        )
        self.assertEqual(first_feasible_evaluation(runs[0]), 2)
        self.assertEqual(first_feasible_evaluation(runs[1]), 3)
        self.assertIsNone(first_feasible_evaluation(runs[2]))
        self.assertAlmostEqual(feasible_share(runs[0]) or 0.0, 2.0 / 3.0)
        self.assertAlmostEqual(feasible_share(runs[1]) or 0.0, 0.5)
        self.assertEqual(feasible_share(runs[2]), 0.0)

        bundle = ExperimentBundle(runs=runs, experiment_dirs=(Path("dummy"),))
        feas = algorithm_feasibility_summaries(bundle)
        self.assertEqual(len(feas), 1)
        self.assertEqual(feas[0].n_runs_with_feasible, 2)
        self.assertAlmostEqual(feas[0].mean_first_feasible_fe or 0.0, 2.5)
        self.assertAlmostEqual(
            feas[0].mean_share_feasible or 0.0,
            (2.0 / 3.0 + 0.5 + 0.0) / 3.0,
        )
        row = algorithm_target_summaries(
            bundle, (1.0,), target_reference="all"
        )[0]
        self.assertEqual(row.n_runs_with_feasible, 2)
        self.assertAlmostEqual(row.mean_first_feasible_fe or 0.0, 2.5)
        self.assertAlmostEqual(
            row.mean_share_feasible or 0.0,
            (2.0 / 3.0 + 0.5 + 0.0) / 3.0,
        )

    def test_empty_runs_fail(self) -> None:
        with self.assertRaises(AggregationError):
            hansen_ert([], 1.0)


class Paper1HansenDiagnosticTests(unittest.TestCase):
    """Locked diagnostics on the stored penalty extract."""

    @staticmethod
    def _hits() -> list[dict[str, str]]:
        import csv

        path = (
            Path(__file__).resolve().parents[2]
            / "_artifacts"
            / "stage1_reference4"
            / "tolerance_2_5_10_2026-10-29"
            / "hits.csv"
        )
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _ert(rows: list[dict[str, str]]) -> float | None:
        hits = [int(row["hit_1"]) if row["hit_1"] else None for row in rows]
        performed = [int(row["n_logged"]) for row in rows]
        return hansen_ert_value(hits, performed, budget=15000, percent=1.0)

    def test_cold_de_and_sbplx_use_the_conducted_counts(self) -> None:
        rows = [
            row
            for row in self._hits()
            if row["sample"] == "reference"
            and row["portfolio"] == "primary_30"
            and row["start"] == "Cold"
            and row["source"] == "stage1_runs_csv_reference4_scan"
        ]
        de = [row for row in rows if row["algorithm"] == "SCIPY_DIFFERENTIAL_EVOLUTION"]
        sbplx = [row for row in rows if row["algorithm"] == "NLOPT_SBPLX"]
        self.assertAlmostEqual(self._ert(de) or 0.0, 20951.82, delta=1.0)
        self.assertAlmostEqual(self._ert(sbplx) or 0.0, 73915.0, delta=1.0)

    def test_local_search_archives_have_no_short_misses(self) -> None:
        local = {"POPULATION_BASIN_HOPPING", "K_ELITE_AMS", "DISCRETE_MLSL"}
        short = [
            row
            for row in self._hits()
            if row["algorithm"] in local and row["hit_1"] == "" and int(row["n_logged"]) < 15000
        ]
        self.assertEqual(short, [])

    def test_hits_extract_does_not_contain_rbfopt(self) -> None:
        present = {row["algorithm"] for row in self._hits() if "RBFOPT" in row["algorithm"]}
        self.assertEqual(present, set())


if __name__ == "__main__":
    unittest.main()
