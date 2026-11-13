"""
R6 goldens: reference-slab κ(x) and simplified M–κ vs Loutfi Grasshopper.

Loutfi κ(x) array and MKD keypoints from
``_mains/verification/comparison_loutfi_sdb_mkd.py``. SDB κ(x) is also locked
as a regression golden so silent interpolation/cache drift fails CI.
"""
from __future__ import annotations

import unittest

import numpy as np

from _mains.Testing.fixtures.floor import ref_floor
from _mains.Testing.fixtures.hp_sections import hp_shell_ref
from _mains.Testing.fixtures.hp_slabs import hp_slab_ref
from _mains.Testing.fixtures.loads import test_loads
from core.analysis_core.section_methods import (
    DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
    calculate_moment_curvature_sls_EC,
    prepare_mk_cache_sls_EC,
    prepare_reference_analysis_cache_EC,
)
from core.analysis_core.statics.constants import SystemType
from core.analysis_core.statics.deflection import DeflectionCalculator
from slab_construction.slab_construction import SlabConstruction

# Loutfi Grasshopper κ(x) [1/m], 41 points on [0, 0.5] (quasi-permanent).
_LOUTFI_KAPPA_X = np.array(
    [
        -0.000316,
        -0.000228762,
        -0.000144006,
        -6.16949e-05,
        1.82036e-05,
        9.57183e-05,
        0.000170875,
        0.000243697,
        0.000314204,
        0.000382417,
        0.000448351,
        0.000512021,
        0.000573443,
        0.000632628,
        0.000689588,
        0.000744333,
        0.000796873,
        0.000847216,
        0.000895371,
        0.000941345,
        0.000985143,
        0.001026773,
        0.00106624,
        0.001103548,
        0.001138703,
        0.001171709,
        0.001202569,
        0.001231287,
        0.001257866,
        0.001282309,
        0.001304618,
        0.001324796,
        0.001342845,
        0.001358766,
        0.001372562,
        0.001395748,
        0.001475312,
        0.001536854,
        0.001580632,
        0.001606827,
        0.001615546,
    ],
    dtype=float,
)

# Production SDB κ(x) [1/m] with simplification=0.15 (R6 lock, 2026-07-27).
_SDB_KAPPA_X_GOLDEN = np.array(
    [
        -2.837133650256e-04,
        -1.994403075535e-04,
        -1.176850913381e-04,
        -3.839706993201e-05,
        3.846889620825e-05,
        1.129531564428e-04,
        1.850918769443e-04,
        2.549175570403e-04,
        3.224594702733e-04,
        3.877440426395e-04,
        4.507951781589e-04,
        5.116345400968e-04,
        5.702817946816e-04,
        6.267548229802e-04,
        6.810699056335e-04,
        7.332418843709e-04,
        7.832843035885e-04,
        8.312095347481e-04,
        8.770288859237e-04,
        9.207526984644e-04,
        9.623904324452e-04,
        1.001950742328e-03,
        1.039441544045e-03,
        1.074870074553e-03,
        1.108242944736e-03,
        1.139566186438e-03,
        1.168845294277e-03,
        1.196085262816e-03,
        1.221290619573e-03,
        1.244465454310e-03,
        1.265613444956e-03,
        1.284737880489e-03,
        1.301841681042e-03,
        1.316927415481e-03,
        1.329997316646e-03,
        1.341053294427e-03,
        1.350096946833e-03,
        1.357129569157e-03,
        1.364371015243e-03,
        1.369080837364e-03,
        1.370650587788e-03,
    ],
    dtype=float,
)

_LOUTFI_MK = {
    0.0: {"M_cr_kNm": 37.445, "M_u_kNm": 92.85},
    0.5: {"M_cr_kNm": 53.85, "M_u_kNm": 144.235},
}


class TestLoutfiKappaX(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.slab = SlabConstruction(hp_slab_ref, ref_floor)
        prepare_reference_analysis_cache_EC(cls.slab, n=0.0)
        prepare_mk_cache_sls_EC(
            cls.slab,
            n=0.0,
            constitutive_law="TENSTIFF_PARABOLIC",
            simplification=0.15,
            simplification_softening=DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
            m_k_num_points=40,
        )
        kappa, _ = DeflectionCalculator._get_interpolated_kappa_array(
            cls.slab,
            loads=test_loads,
            system=SystemType.SIMPLE_BEAM,
            combination="QUASI_PERMANENT",
            n_intervals=40,
            N_axial_N=0.0,
            constitutive_law="TENSTIFF_PARABOLIC",
            m_k_simplification=0.15,
            m_k_simplification_softening=DEFLECTION_MK_SIMPLIFICATION_SOFTENING,
            m_k_num_points=40,
        )
        cls.kappa = np.asarray(kappa, dtype=float)

    def test_kappa_x_length_matches_loutfi(self) -> None:
        self.assertEqual(len(self.kappa), len(_LOUTFI_KAPPA_X))

    def test_kappa_x_regression_golden(self) -> None:
        np.testing.assert_allclose(
            self.kappa, _SDB_KAPPA_X_GOLDEN, rtol=1e-4, atol=1e-8
        )

    def test_kappa_x_within_loutfi_abs_band(self) -> None:
        # Documented comparison band (not a pointwise match).
        abs_err = np.abs(self.kappa - _LOUTFI_KAPPA_X)
        self.assertLessEqual(float(np.max(abs_err)), 3.0e-4)
        self.assertLessEqual(float(np.sqrt(np.mean(abs_err**2))), 1.0e-4)


class TestLoutfiSimplifiedMkKeypoints(unittest.TestCase):
    def test_cracking_and_ultimate_near_loutfi(self) -> None:
        for x, refs in _LOUTFI_MK.items():
            with self.subTest(x=x):
                mk = calculate_moment_curvature_sls_EC(
                    hp_shell_ref.section_at(x),
                    simplification=0.15,
                    m_k_num_points=40,
                )
                M = -np.asarray(mk.m_y, dtype=float) / 1e6
                self.assertAlmostEqual(float(M[0]), 0.0, delta=1e-6)
                self.assertAlmostEqual(float(M[1]), refs["M_cr_kNm"], delta=2.0)
                self.assertAlmostEqual(
                    float(M[-1]), refs["M_u_kNm"], delta=0.10 * refs["M_u_kNm"]
                )


if __name__ == "__main__":
    unittest.main()
