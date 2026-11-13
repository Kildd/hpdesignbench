"""
Phase ALG4.2 gate: reusable algorithm conformance harness.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.ioh_core.algorithm_factory import create_algorithm
from core.ioh_core.algorithms.protocol import IOHAlgorithm
from _mains.Testing.helpers.fake_ioh_problem import assert_algorithm_conformance


def _warm_start_parameters(
    directory: str,
    *,
    max_evaluations: int,
    n: int,
    extra: dict | None = None,
) -> dict:
    """Write a schema-1 InitialDesign and return factory parameters."""
    vector = [0, 0, 0, 0, 3, 0, 0, 0, 0]
    path = Path(directory) / "initial_design.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "problem_id": "L8100_B2",
                "n": n,
                "vectors": [vector for _ in range(n)],
            }
        ),
        encoding="utf-8",
    )
    parameters = {
        "max_evaluations": max_evaluations,
        "initial_design_path": str(path),
    }
    if extra:
        parameters.update(extra)
    return parameters


class TestAlgorithmConformance(unittest.TestCase):
    """Existing algorithms satisfy the shared adapter contract."""

    def test_random_search_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "RANDOM_SEARCH",
                {"max_evaluations": 8},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="RANDOM_SEARCH",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_direct_l_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "NLOPT_DIRECT_L",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="NLOPT_DIRECT_L",
            build_algorithm=build,
            seed=None,
            second_seed=None,
        )

    def test_pymoo_ga_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "PYMOO_GA",
                {"max_evaluations": 16, "population_size": 4},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="PYMOO_GA",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_pymoo_ga_constrained_conformance(self) -> None:
        from _mains.Testing.test_cursor_algorithm_pymoo import (
            _ConstrainedFakeIOHProblem,
        )

        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "PYMOO_GA_CONSTRAINED",
                {"max_evaluations": 16, "population_size": 4},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="PYMOO_GA_CONSTRAINED",
            build_algorithm=build,
            seed=11,
            second_seed=29,
            make_problem=_ConstrainedFakeIOHProblem,
        )

    def test_pymoo_ga_constrained_warm_start_conformance(self) -> None:
        from _mains.Testing.test_cursor_algorithm_pymoo import (
            _ConstrainedFakeIOHProblem,
        )

        with TemporaryDirectory() as tmp:
            parameters = _warm_start_parameters(
                tmp,
                max_evaluations=16,
                n=4,
                extra={"population_size": 4},
            )

            def build(seed: int | None) -> IOHAlgorithm:
                return create_algorithm(
                    "PYMOO_GA_CONSTRAINED_WARM_START",
                    parameters,
                    random_seed=seed,
                )

            assert_algorithm_conformance(
                self,
                algorithm_name="PYMOO_GA_CONSTRAINED_WARM_START",
                build_algorithm=build,
                seed=11,
                second_seed=29,
                make_problem=_ConstrainedFakeIOHProblem,
                lower=[0, 0, 0, 0, 3, 0, 0, 0, 0],
                upper=[10, 8, 5, 5, 14, 12, 3, 5, 3],
            )

    def test_leap_ga_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "LEAP_GA",
                {"max_evaluations": 12, "population_size": 4},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="LEAP_GA",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_cmaes_catcmawm_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "CMAES_CATCMAWM",
                {"max_evaluations": 12, "population_size": 4},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="CMAES_CATCMAWM",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_pystarpso_integer_pso_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "PYSTARPSO_INTEGER_PSO",
                {"max_evaluations": 12, "n_pop": 4},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="PYSTARPSO_INTEGER_PSO",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_nevergrad_metamodel_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "NEVERGRAD_METAMODEL",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="NEVERGRAD_METAMODEL",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_nevergrad_metamodel_oneplusone_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "NEVERGRAD_METAMODEL_ONEPLUSONE",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="NEVERGRAD_METAMODEL_ONEPLUSONE",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_smt_ego_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "SMT_EGO",
                {"max_evaluations": 12, "n_doe": 4},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="SMT_EGO",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_blackboxopt_msrs_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "BLACKBOXOPT_MSRS",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="BLACKBOXOPT_MSRS",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_blackboxopt_dycors_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "BLACKBOXOPT_DYCORS",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="BLACKBOXOPT_DYCORS",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_blackboxopt_gosac_conformance(self) -> None:
        from _mains.Testing.test_cursor_algorithm_scipy_cobyla import (
            _ConstrainedFakeIOHProblem,
        )

        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "BLACKBOXOPT_GOSAC",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="BLACKBOXOPT_GOSAC",
            build_algorithm=build,
            seed=3,
            second_seed=5,
            make_problem=_ConstrainedFakeIOHProblem,
        )

    def test_altbacken_sa_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "ALTBACKEN_SA",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="ALTBACKEN_SA",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_dflint_bboa_conformance(self) -> None:
        from _mains.Testing.test_cursor_algorithm_dflint_bboa import (
            _ConstrainedFakeIOHProblem,
        )

        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "DFLINT_BBOA",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="DFLINT_BBOA",
            build_algorithm=build,
            seed=11,
            second_seed=29,
            make_problem=_ConstrainedFakeIOHProblem,
        )

    def test_dflint_bboa_warm_start_conformance(self) -> None:
        from _mains.Testing.test_cursor_algorithm_dflint_bboa import (
            _ConstrainedFakeIOHProblem,
        )

        with TemporaryDirectory() as tmp:
            parameters = _warm_start_parameters(tmp, max_evaluations=12, n=1)

            def build(seed: int | None) -> IOHAlgorithm:
                return create_algorithm(
                    "DFLINT_BBOA_WARM_START",
                    parameters,
                    random_seed=seed,
                )

            assert_algorithm_conformance(
                self,
                algorithm_name="DFLINT_BBOA_WARM_START",
                build_algorithm=build,
                seed=None,
                second_seed=None,
                make_problem=_ConstrainedFakeIOHProblem,
                lower=[0, 0, 0, 0, 3, 0, 0, 0, 0],
                upper=[10, 8, 5, 5, 14, 12, 3, 5, 3],
            )

    def test_pynomad_mads_pb_conformance(self) -> None:
        from _mains.Testing.test_cursor_algorithm_pynomad_mads_pb import (
            _ConstrainedFakeIOHProblem,
        )

        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "PYNOMAD_MADS_PB",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="PYNOMAD_MADS_PB",
            build_algorithm=build,
            seed=11,
            second_seed=29,
            make_problem=_ConstrainedFakeIOHProblem,
        )

    def test_pynomad_mads_pb_warm_start_conformance(self) -> None:
        from _mains.Testing.test_cursor_algorithm_pynomad_mads_pb import (
            _ConstrainedFakeIOHProblem,
        )

        with TemporaryDirectory() as tmp:
            parameters = _warm_start_parameters(tmp, max_evaluations=12, n=1)

            def build(seed: int | None) -> IOHAlgorithm:
                return create_algorithm(
                    "PYNOMAD_MADS_PB_WARM_START",
                    parameters,
                    random_seed=seed,
                )

            assert_algorithm_conformance(
                self,
                algorithm_name="PYNOMAD_MADS_PB_WARM_START",
                build_algorithm=build,
                seed=11,
                second_seed=29,
                make_problem=_ConstrainedFakeIOHProblem,
                lower=[0, 0, 0, 0, 3, 0, 0, 0, 0],
                upper=[10, 8, 5, 5, 14, 12, 3, 5, 3],
                require_seed_divergence=False,
            )

    def test_pynomad_vns_mads_pb_conformance(self) -> None:
        from _mains.Testing.test_cursor_algorithm_pynomad_mads_pb import (
            _ConstrainedFakeIOHProblem,
        )

        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "PYNOMAD_VNS_MADS_PB",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="PYNOMAD_VNS_MADS_PB",
            build_algorithm=build,
            seed=11,
            second_seed=29,
            make_problem=_ConstrainedFakeIOHProblem,
        )

    def test_pynomad_vns_mads_pb_warm_start_conformance(self) -> None:
        from _mains.Testing.test_cursor_algorithm_pynomad_mads_pb import (
            _ConstrainedFakeIOHProblem,
        )

        with TemporaryDirectory() as tmp:
            parameters = _warm_start_parameters(tmp, max_evaluations=12, n=1)

            def build(seed: int | None) -> IOHAlgorithm:
                return create_algorithm(
                    "PYNOMAD_VNS_MADS_PB_WARM_START",
                    parameters,
                    random_seed=seed,
                )

            assert_algorithm_conformance(
                self,
                algorithm_name="PYNOMAD_VNS_MADS_PB_WARM_START",
                build_algorithm=build,
                seed=11,
                second_seed=29,
                make_problem=_ConstrainedFakeIOHProblem,
                lower=[0, 0, 0, 0, 3, 0, 0, 0, 0],
                upper=[10, 8, 5, 5, 14, 12, 3, 5, 3],
                require_seed_divergence=False,
            )

    def test_pfn_pen_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "PFN_PEN",
                {"max_evaluations": 8},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="PFN_PEN",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_pfn_cei_conformance(self) -> None:
        from _mains.Testing.test_cursor_algorithm_pfn import (
            _ConstrainedFakeIOHProblem,
        )

        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "PFN_CEI",
                {"max_evaluations": 8},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="PFN_CEI",
            build_algorithm=build,
            seed=11,
            second_seed=29,
            make_problem=_ConstrainedFakeIOHProblem,
        )

    def test_hyperopt_tpe_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "HYPEROPT_TPE",
                {"max_evaluations": 10},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="HYPEROPT_TPE",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_optuna_tpe_conformance(self) -> None:
        from _mains.Testing.test_cursor_algorithm_optuna_tpe import (
            _ConstrainedFakeIOHProblem,
        )

        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "OPTUNA_TPE",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="OPTUNA_TPE",
            build_algorithm=build,
            seed=17,
            second_seed=29,
            make_problem=_ConstrainedFakeIOHProblem,
        )

    def test_sbplx_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "NLOPT_SBPLX",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="NLOPT_SBPLX",
            build_algorithm=build,
            seed=17,
            second_seed=29,
        )

    def test_latin_hypercube_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "LATIN_HYPERCUBE",
                {"max_evaluations": 8},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="LATIN_HYPERCUBE",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_scipy_differential_evolution_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "SCIPY_DIFFERENTIAL_EVOLUTION",
                {"max_evaluations": 16},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="SCIPY_DIFFERENTIAL_EVOLUTION",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_scipy_cobyla_conformance(self) -> None:
        from _mains.Testing.test_cursor_algorithm_scipy_cobyla import (
            _ConstrainedFakeIOHProblem,
        )

        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "SCIPY_COBYLA",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="SCIPY_COBYLA",
            build_algorithm=build,
            seed=17,
            second_seed=29,
            make_problem=_ConstrainedFakeIOHProblem,
        )

    def test_scipy_dual_annealing_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "SCIPY_DUAL_ANNEALING",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="SCIPY_DUAL_ANNEALING",
            build_algorithm=build,
            seed=11,
            second_seed=29,
        )

    def test_scipy_direct_l_conformance(self) -> None:
        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "SCIPY_DIRECT_L",
                {"max_evaluations": 12},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="SCIPY_DIRECT_L",
            build_algorithm=build,
            seed=None,
            second_seed=None,
        )

    def test_conformance_does_not_require_optional_extras(self) -> None:
        import sys

        for name in ("pymoo", "hyperopt", "rbfopt", "optuna"):
            sys.modules.pop(name, None)
            for key in list(sys.modules):
                if key.startswith(f"{name}."):
                    sys.modules.pop(key, None)
                if key.startswith(f"core.ioh_core.algorithms.{name}."):
                    sys.modules.pop(key, None)

        def build(seed: int | None) -> IOHAlgorithm:
            return create_algorithm(
                "RANDOM_SEARCH",
                {"max_evaluations": 6},
                random_seed=seed,
            )

        assert_algorithm_conformance(
            self,
            algorithm_name="RANDOM_SEARCH",
            build_algorithm=build,
            seed=3,
            second_seed=7,
        )
        self.assertNotIn("pymoo", sys.modules)
        self.assertNotIn("hyperopt", sys.modules)
        self.assertNotIn("rbfopt", sys.modules)
        self.assertNotIn("optuna", sys.modules)


if __name__ == "__main__":
    unittest.main()
