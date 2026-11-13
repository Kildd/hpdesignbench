"""Deb auxiliary-record contract shared by optimizers and predesign."""
from __future__ import annotations

import math
import unittest

from core.ioh_core.algorithms.native.deb_record import (
    deb_quality_key,
    validate_deb_auxiliary_record,
)
from core.ioh_core.workflows.la.deb import deb_key


_NAMES = ("g1", "g2")


def _valid(**overrides: object) -> dict:
    record = {
        "y": 3.0,
        "violations_": {"g1": 0.0, "g2": 0.0},
        "constraint_values": {"g1": 0.2, "g2": 0.4},
        "analysis_ok": True,
        "feasible": True,
    }
    record.update(overrides)
    return record


class TestDebAuxiliaryRecord(unittest.TestCase):
    def test_failed_analysis_with_infinite_y_and_empty_diagnostics_is_accepted(self) -> None:
        channels = validate_deb_auxiliary_record(
            {
                "y": math.inf,
                "violations_": {},
                "constraint_values": {"extra": "not-a-number"},
                "analysis_ok": False,
                "feasible": False,
            },
            active_constraint_names=_NAMES,
            invalid_utilization=99.0,
        )
        self.assertFalse(channels.analysis_ok)
        self.assertFalse(channels.feasible)
        self.assertTrue(math.isinf(channels.phi))
        self.assertEqual(channels.y, math.inf)
        self.assertEqual(channels.sentinel_constraints, ())

    def test_nan_objective_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate_deb_auxiliary_record(
                _valid(y=math.nan),
                active_constraint_names=_NAMES,
                invalid_utilization=99.0,
            )

    def test_successful_analysis_rejects_incomplete_violations(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_deb_auxiliary_record(
                _valid(violations_={"g1": 0.0}),
                active_constraint_names=_NAMES,
                invalid_utilization=99.0,
            )

    def test_quality_key_matches_la_deb_key_on_finite_channels(self) -> None:
        self.assertEqual(
            deb_quality_key(feasible=True, y=1.25, phi=0.0),
            deb_key(feasible=True, y=1.25, phi=0.0),
        )
        self.assertEqual(
            deb_quality_key(feasible=False, y=9.0, phi=0.4),
            deb_key(feasible=False, y=9.0, phi=0.4),
        )


class TestSharedRecordConsumers(unittest.TestCase):
    """One record is accepted the same way by the optimizer and predesign."""

    def _channels(self, record: dict) -> object:
        return validate_deb_auxiliary_record(
            record,
            active_constraint_names=("g",),
            invalid_utilization=99.0,
        )

    def _manager_point(self, record: dict):
        from core.ioh_core.algorithms.common import RunController
        from core.ioh_core.algorithms.native.discrete_search import (
            DebEvaluationManager,
            DiscreteSearchSpace,
        )
        from _mains.Testing.test_cursor_population_basin_hopping_deb import (
            _DebProblem,
        )

        problem = _DebProblem({(0,): record}, lower=[0], upper=[0])
        controller = RunController(problem, 4, mode="fixed_budget")
        space = DiscreteSearchSpace(problem.bounds.lb, problem.bounds.ub)
        manager = DebEvaluationManager(problem, controller, space)
        return manager.evaluate((0,))

    def test_success_and_failure_agree(self) -> None:
        from slab_construction.slabs.hp_slab.predesign import (
            rank_predesign_records,
        )

        success = {
            "y_p": 2.0,
            "y": 2.0,
            "analysis_ok": True,
            "feasible": True,
            "violations_": {"g": 0.0},
            "constraint_values": {"g": 0.4},
        }
        failure = {
            "y_p": 5.0,
            "y": math.inf,
            "analysis_ok": False,
            "feasible": False,
            "violations_": {},
            "constraint_values": {"extra": "not-a-number"},
        }
        success_point = self._manager_point(success)
        failure_point = self._manager_point(failure)
        self.assertEqual(
            (success_point.y, success_point.phi, success_point.feasible),
            (
                self._channels(success).y,
                self._channels(success).phi,
                self._channels(success).feasible,
            ),
        )
        self.assertTrue(math.isinf(failure_point.phi))
        self.assertTrue(math.isinf(self._channels(failure).phi))
        ranked = rank_predesign_records(
            [
                {
                    "candidate_index": 0,
                    "predesign_y_p": success["y_p"],
                    **success,
                },
                {
                    "candidate_index": 1,
                    "predesign_y_p": failure["y_p"],
                    **failure,
                },
            ],
            active_constraint_names=("g",),
            invalid_utilization=99.0,
        )
        by_index = {row["candidate_index"]: row for row in ranked}
        self.assertEqual(by_index[0]["predesign_phi"], success_point.phi)
        self.assertTrue(math.isinf(by_index[1]["predesign_phi"]))
        self.assertLess(by_index[0]["deb_rank"], by_index[1]["deb_rank"])


if __name__ == "__main__":
    unittest.main()
