"""HP-shell predesign grid ranked by production y_p."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _mains.Testing._paths import (
    CONSTRAINT_DEFAULTS_CSV,
    MATERIALS_CSV,
    PARAM_DEFAULTS_CSV,
    PROBLEM_LIST_CSV,
)
from core.ioh_core.import_specs import (
    build_space,
    load_constraint_defaults,
    load_materials_registry,
    load_param_defaults,
    load_problems_combined,
    make_decode,
)
from slab_construction.slabs.hp_slab.analysis import analysis
from slab_construction.slabs.hp_slab.predesign import (
    PREDESIGN_CONCRETE_GRADES,
    PREDESIGN_FIXED_PARAMS,
    PREDESIGN_H_GES_MM,
    PREDESIGN_KAP_T_PERCENT,
    PREDESIGN_REINF_PAIRS,
    PREDESIGN_T_MM,
    PREDESIGN_CANDIDATE_COUNT,
    encode_catalogue_indices,
    evaluate_predesign_grid,
    index_vectors_from_rows,
    iter_predesign_candidates,
    predesign_base_params,
    rank_predesign_rows,
)


class TestPredesignGrid(unittest.TestCase):
    """Locked P2 candidate grid without evaluating analysis."""

    @classmethod
    def setUpClass(cls) -> None:
        pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
        cdef = load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV)
        problems = load_problems_combined(pdef, cdef, PROBLEM_LIST_CSV)
        info = problems["L8100_B2"]
        var_names, lb_idx, _ub = build_space(info["model"])
        decode = make_decode(info["model"], var_names)
        cls.base = predesign_base_params(decode, lb_idx)
        cls.model = info["model"]
        cls.var_names = var_names
        cls.decode_params = staticmethod(decode)

    def test_cardinality_is_2700(self) -> None:
        self.assertEqual(
            PREDESIGN_H_GES_MM,
            (100.0, 200.0, 300.0, 400.0, 500.0, 600.0),
        )
        self.assertEqual(PREDESIGN_T_MM, (20.0, 40.0, 60.0, 80.0, 100.0))
        self.assertEqual(
            PREDESIGN_REINF_PAIRS,
            (
                (5.0, 1.81),
                (15.0, 1.81),
                (30.0, 1.81),
                (30.0, 3.62),
                (30.0, 7.24),
                (30.0, 10.86),
            ),
        )
        self.assertEqual(PREDESIGN_KAP_T_PERCENT, (0.0, 15.0, 30.0, 45.0, 60.0))
        self.assertEqual(
            PREDESIGN_CONCRETE_GRADES,
            ("C25/30", "C50/60", "C100/115"),
        )
        self.assertEqual(PREDESIGN_CANDIDATE_COUNT, 2700)
        candidates = list(iter_predesign_candidates(self.base, self.model))
        self.assertEqual(len(candidates), 2700)
        _first_idx, first = candidates[0]
        self.assertAlmostEqual(first["geom_h_ges_mm"], 100.0)
        self.assertAlmostEqual(first["geom_t_mm"], 20.0)
        self.assertAlmostEqual(first["geom_nt"], 5.0)
        self.assertAlmostEqual(first["reinf_a_tex_mm2"], 1.81)
        self.assertAlmostEqual(first["reinf_kap_t_percent"], 0.0)
        self.assertEqual(first["mat_conc_id"], "C25/30")
        _last_idx, last = candidates[-1]
        self.assertEqual(_last_idx, 2699)
        self.assertAlmostEqual(last["geom_h_ges_mm"], 600.0)
        self.assertAlmostEqual(last["geom_t_mm"], 100.0)
        self.assertAlmostEqual(last["geom_nt"], 30.0)
        self.assertAlmostEqual(last["reinf_a_tex_mm2"], 10.86)
        self.assertAlmostEqual(last["reinf_kap_t_percent"], 60.0)
        self.assertEqual(last["mat_conc_id"], "C100/115")

    def test_kap_30_encodes_from_the_row(self) -> None:
        row = {
            "geom_h_ges_mm": 600.0,
            "geom_t_mm": 30.0,
            "geom_nt": 30.0,
            "reinf_a_tex_mm2": 10.86,
            "mat_conc_id": "C100/115",
            "mat_reinf_id": "solidian Roving 2R (Z-1.6-308)",
            "geom_span_mm": 8100.0,
            "reinf_kap_t_percent": 30.0,
            **PREDESIGN_FIXED_PARAMS,
        }
        encoded = index_vectors_from_rows(
            [row],
            model=self.model,
            var_names=self.var_names,
        )
        decoded = self.decode_params(list(encoded[0]))
        self.assertAlmostEqual(decoded["reinf_kap_t_percent"], 30.0)

    def test_problem_1_base_keeps_p1_span(self) -> None:
        pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
        cdef = load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV)
        problems = load_problems_combined(pdef, cdef, PROBLEM_LIST_CSV)
        info = problems["L6750_B2"]
        var_names, lb_idx, _ub = build_space(info["model"])
        decode = make_decode(info["model"], var_names)
        base = predesign_base_params(decode, lb_idx)
        self.assertAlmostEqual(base["geom_span_mm"], 6750.0)
        first_idx, first = next(
            iter_predesign_candidates(base, info["model"])
        )
        self.assertEqual(first_idx, 0)
        self.assertAlmostEqual(first["geom_span_mm"], 6750.0)

    def test_fixed_params_and_p2_span(self) -> None:
        self.assertAlmostEqual(self.base["geom_span_mm"], 8100.0)
        for name, value in PREDESIGN_FIXED_PARAMS.items():
            self.assertEqual(self.base[name], value)
        first_idx, first = next(iter_predesign_candidates(self.base, self.model))
        self.assertEqual(first_idx, 0)
        for name, value in PREDESIGN_FIXED_PARAMS.items():
            self.assertEqual(first[name], value)
        self.assertAlmostEqual(first["geom_span_mm"], 8100.0)


class TestPredesignRanking(unittest.TestCase):
    """Tie-break: predesign_y_p, then y, then candidate index."""

    def test_tie_break_yp_then_y_then_index(self) -> None:
        rows = [
            {"candidate_index": 0, "predesign_y_p": 2.0, "y": 10.0},
            {"candidate_index": 1, "predesign_y_p": 2.0, "y": 5.0},
            {"candidate_index": 2, "predesign_y_p": 1.0, "y": 20.0},
            {"candidate_index": 3, "predesign_y_p": 2.0, "y": 5.0},
        ]
        ranked = rank_predesign_rows(rows)
        self.assertEqual(
            [r["candidate_index"] for r in ranked],
            [2, 1, 3, 0],
        )
        self.assertEqual([r["rank"] for r in ranked], [1, 2, 3, 4])
        self.assertEqual(len(ranked), 4)

    def test_nonfinite_scores_sort_after_finite_rows(self) -> None:
        rows = [
            {"candidate_index": 4, "predesign_y_p": 1.0, "y": 1.0},
            {"candidate_index": 0, "predesign_y_p": float("nan"), "y": 0.0},
            {"candidate_index": 1, "predesign_y_p": float("inf"), "y": 0.0},
            {"candidate_index": 2, "predesign_y_p": float("-inf"), "y": 0.0},
            {"candidate_index": 3, "predesign_y_p": 2.0, "y": float("-inf")},
            {"candidate_index": 5, "predesign_y_p": 3.0, "y": 1.0},
            {"candidate_index": 6, "predesign_y_p": 3.0, "y": 1.0},
        ]
        ranked = rank_predesign_rows(rows)
        self.assertEqual(
            [row["candidate_index"] for row in ranked],
            [4, 3, 5, 6, 0, 1, 2],
        )


_ACTIVE = (
    "A_bending_capacity",
    "B1a_deflection_by_wmax_capacity",
    "B2a_failure_announcement_by_wmin_capacity",
    "C1_concrete_cover_capacity",
    "C2_clear_spacing_capacity",
    "C3_shell_thickness_capacity",
    "D1_airborne_sound_insulation_capacity",
    "D2_impact_sound_insulation_capacity",
    "Z1_nt_dt_combination_capacity",
    "Z2_beam_theory_H_L_capacity",
    "Z3_beam_theory_B_L_capacity",
)


class TestPredesignProductionScore(unittest.TestCase):
    """A grid row's predesign_y_p is production analysis y_p."""

    @classmethod
    def setUpClass(cls) -> None:
        pdef = load_param_defaults(PARAM_DEFAULTS_CSV)
        cdef = load_constraint_defaults(CONSTRAINT_DEFAULTS_CSV)
        problems = load_problems_combined(pdef, cdef, PROBLEM_LIST_CSV)
        info = problems["L6750_B2"]
        cls.model = info["model"]
        cls.constraints = info["constraints"]
        cls.materials = load_materials_registry(MATERIALS_CSV)
        var_names, lb_idx, _ub = build_space(info["model"])
        decode = make_decode(info["model"], var_names)
        cls.base = predesign_base_params(decode, lb_idx)

    def test_first_row_matches_production_y_p(self) -> None:
        _index, params = next(iter_predesign_candidates(self.base, self.model))
        production = analysis(params, self.constraints, self.materials)
        rows = evaluate_predesign_grid(
            materials=self.materials,
            base_params=self.base,
            model=self.model,
            constraints=self.constraints,
            limit=1,
        )
        self.assertEqual(set(self.constraints), set(_ACTIVE))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["predesign_y_p"], production["y_p"], places=12)
        self.assertAlmostEqual(row["y"], production["y"], places=12)
        self.assertEqual(row["feasible"], production["feasible"])
        for name in _ACTIVE:
            self.assertIn(name, row)
            self.assertAlmostEqual(row[name], production["constraint_values"][name], places=12)
        for dropped in ("u_c", "u_t", "v_2"):
            self.assertNotIn(dropped, row)
        self.assertFalse(any(str(key).startswith("predesign_strain") for key in row))


class _RecordingLibrary:
    """Lookup stand-in that records the structural id and then fails closed."""

    def __init__(self, *, mismatch: bool) -> None:
        self.calls: list[int] = []
        self.mismatch = mismatch
        self.identity = type(
            "Identity",
            (),
            {"structural_names": ("geom_h_ges_mm",), "snapshot_hash": "libhash"},
        )()

    def get(self, structural_id: int, **_kwargs: object):
        self.calls.append(int(structural_id))
        if not self.mismatch:
            from core.ioh_core.structural_artifacts.library import (
                StructuralLibraryError,
            )

            raise StructuralLibraryError(f"missing structural_id {structural_id}")
        from core.ioh_core.structural_artifacts.artifact import StructuralArtifact

        return StructuralArtifact(
            key=(("not-the-candidate", 0),),
            snapshot_hash="libhash",
            n=0.0,
            m_u=(),
            m_cr=(),
            mk_curves=(),
            mk_metadata=(),
            mk_cleared=False,
        )

    def close(self) -> None:
        return None

    def __enter__(self) -> "_RecordingLibrary":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class TestPredesignLibraryFailClosed(unittest.TestCase):
    """A library miss or key mismatch writes neither predesign file."""

    def _expected_first_structural_id(self) -> int:
        from core.ioh_core.structural_artifacts.indexing import (
            global_index_to_structural_id,
        )
        from core.ioh_core.workflows.brute_force.indexing import indices_to_linear
        from core.ioh_core.workflows.brute_force.models import IndexSpace
        from slab_construction.slabs.hp_slab.predesign import (
            _resolve_hp_slab_bundle,
        )

        bundle, _materials = _resolve_hp_slab_bundle("L6750_B2")
        base = predesign_base_params(bundle["decode"], bundle["lb_idx"])
        _index, params = next(iter_predesign_candidates(base, bundle["model"]))
        names = tuple(str(name) for name in bundle["var_names"])
        space = IndexSpace.from_bounds(
            names,
            tuple(int(v) for v in bundle["lb_idx"]),
            tuple(int(v) for v in bundle["ub_idx"]),
        )
        vector = encode_catalogue_indices(bundle["model"], names, params)
        return global_index_to_structural_id(indices_to_linear(vector, space), space)

    def _assert_nothing_written(self, root: Path) -> None:
        from core.ioh_core.workflows.predesign_paths import (
            predesign_initial_design_deb_path,
            predesign_initial_design_path,
            predesign_results_path,
        )

        self.assertFalse(predesign_results_path(root, "L6750_B2").exists())
        self.assertFalse(predesign_initial_design_path(root, "L6750_B2").exists())
        self.assertFalse(
            predesign_initial_design_deb_path(root, "L6750_B2").exists()
        )

    def test_missing_library_writes_nothing(self) -> None:
        from core.ioh_core.structural_artifacts.library import StructuralLibraryError
        from slab_construction.slabs.hp_slab.predesign import write_predesign_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(StructuralLibraryError):
                write_predesign_run(
                    "L6750_B2",
                    root,
                    library_path=root / "absent-library",
                )
            self._assert_nothing_written(root)

    def test_missing_structural_id_writes_nothing(self) -> None:
        from core.ioh_core.structural_artifacts.library import StructuralLibraryError
        from slab_construction.slabs.hp_slab.predesign import write_predesign_run

        library = _RecordingLibrary(mismatch=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "core.ioh_core.structural_artifacts.library.open_structural_library",
                return_value=library,
            ):
                with self.assertRaises(StructuralLibraryError):
                    write_predesign_run(
                        "L6750_B2",
                        root,
                        library_path=root / "library",
                    )
            self._assert_nothing_written(root)
        self.assertEqual(library.calls, [self._expected_first_structural_id()])

    def test_key_mismatch_writes_nothing(self) -> None:
        from core.ioh_core.structural_artifacts.provider import StructuralProviderError
        from slab_construction.slabs.hp_slab.predesign import write_predesign_run

        library = _RecordingLibrary(mismatch=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "core.ioh_core.structural_artifacts.library.open_structural_library",
                return_value=library,
            ):
                with self.assertRaises(StructuralProviderError):
                    write_predesign_run(
                        "L6750_B2",
                        root,
                        library_path=root / "library",
                    )
            self._assert_nothing_written(root)
        self.assertEqual(library.calls, [self._expected_first_structural_id()])


class TestDebRankingContract(unittest.TestCase):
    """Deb order of injected records, without a slab solve."""

    def _row(
        self,
        index: int,
        *,
        y: float,
        y_p: float,
        feasible: bool,
        analysis_ok: bool = True,
        violations: dict[str, float] | None = None,
        values: dict[str, float] | None = None,
    ) -> dict:
        recorded = {name: 0.0 for name in _ACTIVE}
        if violations:
            recorded.update(violations)
        constraint_values = {name: 0.2 for name in _ACTIVE}
        if values:
            constraint_values.update(values)
        return {
            "candidate_index": index,
            "predesign_y_p": y_p,
            "y": y,
            "analysis_ok": analysis_ok,
            "feasible": feasible,
            "violations_": recorded,
            "constraint_values": constraint_values,
        }

    def _rank(self, rows: list[dict]) -> list[dict]:
        from slab_construction.slabs.hp_slab.predesign import (
            rank_predesign_records,
        )

        return rank_predesign_records(
            rows,
            active_constraint_names=_ACTIVE,
            invalid_utilization=99.0,
        )

    def test_feasible_rows_lead_and_follow_raw_y_then_index(self) -> None:
        ranked = self._rank(
            [
                self._row(2, y=5.0, y_p=1.0, feasible=True),
                self._row(
                    0,
                    y=1.0,
                    y_p=50.0,
                    feasible=False,
                    violations={_ACTIVE[0]: 0.2},
                ),
                self._row(3, y=5.0, y_p=2.0, feasible=True),
                self._row(1, y=4.0, y_p=9.0, feasible=True),
            ]
        )
        deb_order = [
            row["candidate_index"]
            for row in sorted(ranked, key=lambda row: row["deb_rank"])
        ]
        self.assertEqual(deb_order, [1, 2, 3, 0])
        self.assertEqual(
            [row["candidate_index"] for row in ranked],
            [2, 3, 1, 0],
        )

    def test_infeasible_rows_follow_phi_then_index(self) -> None:
        ranked = self._rank(
            [
                self._row(
                    4,
                    y=1.0,
                    y_p=3.0,
                    feasible=False,
                    violations={_ACTIVE[0]: 0.2},
                ),
                self._row(
                    1,
                    y=9.0,
                    y_p=3.0,
                    feasible=False,
                    violations={_ACTIVE[0]: 0.2},
                ),
                self._row(
                    2,
                    y=0.0,
                    y_p=1.0,
                    feasible=False,
                    violations={_ACTIVE[0]: 0.5},
                ),
            ]
        )
        deb_order = [
            row["candidate_index"]
            for row in sorted(ranked, key=lambda row: row["deb_rank"])
        ]
        self.assertEqual(deb_order, [1, 4, 2])
        self.assertEqual(
            sorted(row["deb_rank"] for row in ranked),
            [1, 2, 3],
        )

    def test_every_grid_candidate_can_be_infeasible(self) -> None:
        rows = [
            self._row(
                index,
                y=float(index),
                y_p=float(index),
                feasible=False,
                violations={_ACTIVE[0]: (index + 1) / 1000.0},
            )
            for index in range(PREDESIGN_CANDIDATE_COUNT)
        ]
        ranked = self._rank(rows)
        self.assertEqual(len(ranked), PREDESIGN_CANDIDATE_COUNT)
        self.assertTrue(all(row["feasible"] is False for row in ranked))
        self.assertEqual(
            [row["deb_rank"] for row in ranked],
            list(range(1, PREDESIGN_CANDIDATE_COUNT + 1)),
        )

    def test_utilization_99_keeps_a_finite_phi(self) -> None:
        sentinel = self._row(
            0,
            y=10.0,
            y_p=20.0,
            feasible=False,
            violations={_ACTIVE[0]: 98.0},
            values={_ACTIVE[0]: 99.0},
        )
        failed = self._row(
            1,
            y=math.inf,
            y_p=1.0,
            feasible=False,
            analysis_ok=False,
        )
        failed["violations_"] = {}
        failed["constraint_values"] = {}
        ranked = self._rank([failed, sentinel])
        by_index = {row["candidate_index"]: row for row in ranked}
        self.assertAlmostEqual(by_index[0]["predesign_phi"], 98.0)
        self.assertEqual(by_index[0]["deb_rank"], 1)
        self.assertTrue(math.isinf(by_index[1]["predesign_phi"]))
        self.assertEqual(by_index[1]["deb_rank"], 2)

    def test_failed_nan_and_incomplete_success_are_rejected(self) -> None:
        failed = self._row(
            0,
            y=math.nan,
            y_p=1.0,
            feasible=False,
            analysis_ok=False,
        )
        with self.assertRaisesRegex(ValueError, "NaN"):
            self._rank([failed])
        incomplete = self._row(1, y=1.0, y_p=1.0, feasible=True)
        incomplete["violations_"] = {_ACTIVE[0]: 0.0}
        with self.assertRaisesRegex(ValueError, "exactly"):
            self._rank([incomplete])

    def test_duplicate_candidate_index_is_rejected(self) -> None:
        row = self._row(3, y=1.0, y_p=1.0, feasible=True)
        with self.assertRaisesRegex(ValueError, "duplicate candidate_index 3"):
            self._rank([row, dict(row)])

    def test_infinite_penalty_is_rejected_on_a_successful_analysis(self) -> None:
        row = self._row(0, y=1.0, y_p=math.inf, feasible=True)
        with self.assertRaisesRegex(ValueError, r"y_p=\+inf"):
            self._rank([row])
        negative = self._row(1, y=1.0, y_p=float("-inf"), feasible=True)
        with self.assertRaisesRegex(ValueError, "-inf"):
            self._rank([negative])
