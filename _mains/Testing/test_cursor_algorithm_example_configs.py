"""
ALG8.2: accepted-family example TOML parse/prepare contracts.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.execution_core.config import load_experiment_config
from core.execution_core.lifecycle import prepare_experiment
from core.ioh_core.algorithm_factory import create_algorithm
from core.ioh_core.algorithms.errors import AlgorithmDependencyError
from core.ioh_core.workflows.optimization_runs.planner import algorithm_parameters_for

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "_mains" / "studies" / "Execution" / "configs" / "examples"

EXAMPLE_PYMOO = _CONFIG_DIR / "optimization_pymoo.toml"
EXAMPLE_PYMOO_CONSTRAINED = _CONFIG_DIR / "optimization_pymoo_constrained.toml"
EXAMPLE_PYMOO_CONSTRAINED_WARM_START = (
    _CONFIG_DIR / "optimization_pymoo_constrained_warm_start.toml"
)
EXAMPLE_LEAP_GA = _CONFIG_DIR / "optimization_leap_ga.toml"
EXAMPLE_CMAES_CATCMAWM = _CONFIG_DIR / "optimization_cmaes_catcmawm.toml"
EXAMPLE_PYSTARPSO_INTEGER_PSO = (
    _CONFIG_DIR / "optimization_pystarpso_integer_pso.toml"
)
EXAMPLE_NEVERGRAD_METAMODEL = (
    _CONFIG_DIR / "optimization_nevergrad_metamodel.toml"
)
EXAMPLE_NEVERGRAD_METAMODEL_ONEPLUSONE = (
    _CONFIG_DIR / "optimization_nevergrad_metamodel_oneplusone.toml"
)
EXAMPLE_SMT_EGO = _CONFIG_DIR / "optimization_smt_ego.toml"
EXAMPLE_BLACKBOXOPT_MSRS = (
    _CONFIG_DIR / "optimization_blackboxopt_msrs.toml"
)
EXAMPLE_BLACKBOXOPT_DYCORS = (
    _CONFIG_DIR / "optimization_blackboxopt_dycors.toml"
)
EXAMPLE_BLACKBOXOPT_GOSAC = (
    _CONFIG_DIR / "optimization_blackboxopt_gosac.toml"
)
EXAMPLE_ALTBACKEN_SA = _CONFIG_DIR / "optimization_altbacken_sa.toml"
EXAMPLE_DFLINT_BBOA = _CONFIG_DIR / "optimization_dflint_bboa.toml"
EXAMPLE_DFLINT_BBOA_WARM_START = (
    _CONFIG_DIR / "optimization_dflint_bboa_warm_start.toml"
)
EXAMPLE_PYNOMAD_MADS_PB = _CONFIG_DIR / "optimization_pynomad_mads_pb.toml"
EXAMPLE_PYNOMAD_MADS_PB_WARM_START = (
    _CONFIG_DIR / "optimization_pynomad_mads_pb_warm_start.toml"
)
EXAMPLE_PYNOMAD_VNS_MADS_PB = (
    _CONFIG_DIR / "optimization_pynomad_vns_mads_pb.toml"
)
EXAMPLE_PYNOMAD_VNS_MADS_PB_WARM_START = (
    _CONFIG_DIR / "optimization_pynomad_vns_mads_pb_warm_start.toml"
)
EXAMPLE_PFN_PEN = _CONFIG_DIR / "optimization_pfn_pen.toml"
EXAMPLE_PFN_CEI = _CONFIG_DIR / "optimization_pfn_cei.toml"
EXAMPLE_HYPEROPT = _CONFIG_DIR / "optimization_hyperopt.toml"
EXAMPLE_OPTUNA_TPE = _CONFIG_DIR / "optimization_optuna_tpe.toml"
EXAMPLE_RBFOPT = _CONFIG_DIR / "optimization_rbfopt.toml"
EXAMPLE_SCIPY_DE = _CONFIG_DIR / "optimization_scipy_differential_evolution.toml"
EXAMPLE_SCIPY_COBYLA = _CONFIG_DIR / "optimization_scipy_cobyla.toml"
EXAMPLE_SCIPY_DIRECT_L = _CONFIG_DIR / "optimization_scipy_direct_l.toml"
EXAMPLE_SCIPY_DUAL_ANNEALING = (
    _CONFIG_DIR / "optimization_scipy_dual_annealing.toml"
)
EXAMPLE_OPTAX_ADAM = _CONFIG_DIR / "optimization_optax_adam.toml"
EXAMPLE_OPTAX_RPROP = _CONFIG_DIR / "optimization_optax_rprop.toml"
EXAMPLE_OPTAX_SGD_NESTEROV = (
    _CONFIG_DIR / "optimization_optax_sgd_nesterov.toml"
)
EXAMPLE_FIXED_TARGET = _CONFIG_DIR / "optimization_fixed_target.toml"
ALL_EXAMPLES = (
    EXAMPLE_PYMOO,
    EXAMPLE_PYMOO_CONSTRAINED,
    EXAMPLE_PYMOO_CONSTRAINED_WARM_START,
    EXAMPLE_LEAP_GA,
    EXAMPLE_CMAES_CATCMAWM,
    EXAMPLE_PYSTARPSO_INTEGER_PSO,
    EXAMPLE_NEVERGRAD_METAMODEL,
    EXAMPLE_NEVERGRAD_METAMODEL_ONEPLUSONE,
    EXAMPLE_SMT_EGO,
    EXAMPLE_BLACKBOXOPT_MSRS,
    EXAMPLE_BLACKBOXOPT_DYCORS,
    EXAMPLE_BLACKBOXOPT_GOSAC,
    EXAMPLE_ALTBACKEN_SA,
    EXAMPLE_DFLINT_BBOA,
    EXAMPLE_DFLINT_BBOA_WARM_START,
    EXAMPLE_PYNOMAD_MADS_PB,
    EXAMPLE_PYNOMAD_MADS_PB_WARM_START,
    EXAMPLE_PYNOMAD_VNS_MADS_PB,
    EXAMPLE_PYNOMAD_VNS_MADS_PB_WARM_START,
    EXAMPLE_PFN_PEN,
    EXAMPLE_PFN_CEI,
    EXAMPLE_HYPEROPT,
    EXAMPLE_OPTUNA_TPE,
    EXAMPLE_RBFOPT,
    EXAMPLE_SCIPY_DE,
    EXAMPLE_SCIPY_COBYLA,
    EXAMPLE_SCIPY_DIRECT_L,
    EXAMPLE_SCIPY_DUAL_ANNEALING,
    EXAMPLE_OPTAX_ADAM,
    EXAMPLE_OPTAX_RPROP,
    EXAMPLE_OPTAX_SGD_NESTEROV,
)


class TestAlgorithmExampleConfigs(unittest.TestCase):
    """Example TOMLs parse; prepare defers RBFOpt solver preflight."""

    def test_all_algorithm_examples_parse(self) -> None:
        expected = {
            EXAMPLE_PYMOO: ("PYMOO_GA", 10),
            EXAMPLE_PYMOO_CONSTRAINED: ("PYMOO_GA_CONSTRAINED", 10),
            EXAMPLE_PYMOO_CONSTRAINED_WARM_START: (
                "PYMOO_GA_CONSTRAINED_WARM_START",
                10,
            ),
            EXAMPLE_LEAP_GA: ("LEAP_GA", 10),
            EXAMPLE_CMAES_CATCMAWM: ("CMAES_CATCMAWM", 10),
            EXAMPLE_PYSTARPSO_INTEGER_PSO: ("PYSTARPSO_INTEGER_PSO", 10),
            EXAMPLE_NEVERGRAD_METAMODEL: ("NEVERGRAD_METAMODEL", 10),
            EXAMPLE_NEVERGRAD_METAMODEL_ONEPLUSONE: (
                "NEVERGRAD_METAMODEL_ONEPLUSONE",
                10,
            ),
            EXAMPLE_SMT_EGO: ("SMT_EGO", 10),
            EXAMPLE_BLACKBOXOPT_MSRS: ("BLACKBOXOPT_MSRS", 10),
            EXAMPLE_BLACKBOXOPT_DYCORS: ("BLACKBOXOPT_DYCORS", 10),
            EXAMPLE_BLACKBOXOPT_GOSAC: ("BLACKBOXOPT_GOSAC", 10),
            EXAMPLE_ALTBACKEN_SA: ("ALTBACKEN_SA", 10),
            EXAMPLE_DFLINT_BBOA: ("DFLINT_BBOA", 10),
            EXAMPLE_DFLINT_BBOA_WARM_START: ("DFLINT_BBOA_WARM_START", 10),
            EXAMPLE_PYNOMAD_MADS_PB: ("PYNOMAD_MADS_PB", 10),
            EXAMPLE_PYNOMAD_MADS_PB_WARM_START: (
                "PYNOMAD_MADS_PB_WARM_START",
                10,
            ),
            EXAMPLE_PYNOMAD_VNS_MADS_PB: ("PYNOMAD_VNS_MADS_PB", 10),
            EXAMPLE_PYNOMAD_VNS_MADS_PB_WARM_START: (
                "PYNOMAD_VNS_MADS_PB_WARM_START",
                10,
            ),
            EXAMPLE_PFN_PEN: ("PFN_PEN", 10),
            EXAMPLE_PFN_CEI: ("PFN_CEI", 10),
            EXAMPLE_HYPEROPT: ("HYPEROPT_TPE", 10),
            EXAMPLE_OPTUNA_TPE: ("OPTUNA_TPE", 10),
            EXAMPLE_RBFOPT: ("RBFOPT", 10),
            EXAMPLE_SCIPY_DE: ("SCIPY_DIFFERENTIAL_EVOLUTION", 40),
            EXAMPLE_SCIPY_COBYLA: ("SCIPY_COBYLA", 40),
            EXAMPLE_SCIPY_DIRECT_L: ("SCIPY_DIRECT_L", 40),
            EXAMPLE_SCIPY_DUAL_ANNEALING: ("SCIPY_DUAL_ANNEALING", 40),
            EXAMPLE_OPTAX_ADAM: ("OPTAX_ADAM", 64),
            EXAMPLE_OPTAX_RPROP: ("OPTAX_RPROP", 64),
            EXAMPLE_OPTAX_SGD_NESTEROV: ("OPTAX_SGD_NESTEROV", 64),
        }
        for path in ALL_EXAMPLES:
            self.assertTrue(path.is_file(), f"missing example config: {path}")
            config = load_experiment_config(path)
            name, budget = expected[path]
            self.assertEqual(len(config.experiment.algorithms), 1)
            algorithm = config.experiment.algorithms[0]
            self.assertEqual(algorithm.name, name)
            self.assertEqual(algorithm.max_evaluations, budget)
            # Registry merge must succeed at plan/prepare time.
            params = algorithm_parameters_for(algorithm)
            self.assertEqual(params["max_evaluations"], budget)

        pymoo = load_experiment_config(EXAMPLE_PYMOO).experiment.algorithms[0]
        self.assertEqual(pymoo.parameters.get("population_size"), 4)

        pymoo_c = load_experiment_config(
            EXAMPLE_PYMOO_CONSTRAINED
        ).experiment.algorithms[0]
        self.assertEqual(pymoo_c.parameters.get("population_size"), 4)

        leap = load_experiment_config(EXAMPLE_LEAP_GA).experiment.algorithms[0]
        self.assertEqual(leap.parameters.get("population_size"), 4)

        cmaes = load_experiment_config(
            EXAMPLE_CMAES_CATCMAWM
        ).experiment.algorithms[0]
        self.assertEqual(cmaes.parameters.get("population_size"), 4)

        starpso = load_experiment_config(
            EXAMPLE_PYSTARPSO_INTEGER_PSO
        ).experiment.algorithms[0]
        self.assertEqual(starpso.parameters.get("n_pop"), 4)

        smt_ego = load_experiment_config(EXAMPLE_SMT_EGO).experiment.algorithms[0]
        self.assertEqual(smt_ego.parameters.get("n_doe"), 4)
        smt_params = algorithm_parameters_for(smt_ego)
        self.assertEqual(
            smt_params,
            {"max_evaluations": 10, "n_doe": 4},
        )

        vns = load_experiment_config(
            EXAMPLE_PYNOMAD_VNS_MADS_PB
        ).experiment.algorithms[0]
        self.assertEqual(vns.parameters.get("vns_mads_search_trigger"), 0.75)
        vns_params = algorithm_parameters_for(vns)
        self.assertEqual(
            vns_params,
            {
                "max_evaluations": 10,
                "vns_mads_search_trigger": 0.75,
            },
        )

        rbfopt = load_experiment_config(EXAMPLE_RBFOPT).experiment.algorithms[0]
        self.assertEqual(rbfopt.parameters.get("ga_base_population_size"), 20)
        minlp = str(rbfopt.parameters["minlp_solver_path"])
        nlp = str(rbfopt.parameters["nlp_solver_path"])
        # Example ships POSIX-style absolute placeholders (edit before a real run).
        self.assertTrue(
            minlp.startswith("/") or Path(minlp).is_absolute(),
            msg=minlp,
        )
        self.assertTrue(
            nlp.startswith("/") or Path(nlp).is_absolute(),
            msg=nlp,
        )

    def test_fixed_target_example_parses_and_prepares(self) -> None:
        self.assertTrue(EXAMPLE_FIXED_TARGET.is_file())
        config = load_experiment_config(EXAMPLE_FIXED_TARGET)
        termination = config.experiment.termination
        self.assertEqual(termination.mode, "fixed_target")
        self.assertEqual(termination.global_optimum, 1.0e20)
        self.assertEqual(termination.allowed_deviation_percent, 0.0)
        self.assertEqual(termination.target_reference, "all")
        self.assertEqual(termination.target_threshold, 1.0e20)
        self.assertEqual(config.experiment.algorithms[0].name, "RANDOM_SEARCH")
        self.assertEqual(config.experiment.algorithms[0].max_evaluations, 8)
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "optimization_fixed_target_example"
            prepare_experiment(config_path=EXAMPLE_FIXED_TARGET, output=out)
            self.assertTrue((out / "manifest.json").is_file())
            from core.execution_core.atomic_io import read_json

            payload = read_json(out / "manifest.json")["tasks"][0]["payload"]
            self.assertEqual(payload["termination_mode"], "fixed_target")
            self.assertEqual(payload["target_threshold"], 1.0e20)

    def test_non_rbfopt_examples_prepare(self) -> None:
        for path, experiment_id in (
            (EXAMPLE_PYMOO, "optimization_pymoo_example"),
            (EXAMPLE_PYMOO_CONSTRAINED, "optimization_pymoo_constrained_example"),
            (EXAMPLE_LEAP_GA, "optimization_leap_ga_example"),
            (EXAMPLE_CMAES_CATCMAWM, "optimization_cmaes_catcmawm_example"),
            (
                EXAMPLE_PYSTARPSO_INTEGER_PSO,
                "optimization_pystarpso_integer_pso_example",
            ),
            (
                EXAMPLE_NEVERGRAD_METAMODEL,
                "optimization_nevergrad_metamodel_example",
            ),
            (
                EXAMPLE_NEVERGRAD_METAMODEL_ONEPLUSONE,
                "optimization_nevergrad_metamodel_oneplusone_example",
            ),
            (EXAMPLE_SMT_EGO, "optimization_smt_ego_example"),
            (
                EXAMPLE_BLACKBOXOPT_MSRS,
                "optimization_blackboxopt_msrs_example",
            ),
            (
                EXAMPLE_BLACKBOXOPT_DYCORS,
                "optimization_blackboxopt_dycors_example",
            ),
            (
                EXAMPLE_BLACKBOXOPT_GOSAC,
                "optimization_blackboxopt_gosac_example",
            ),
            (EXAMPLE_ALTBACKEN_SA, "optimization_altbacken_sa_example"),
            (EXAMPLE_HYPEROPT, "optimization_hyperopt_example"),
            (EXAMPLE_OPTUNA_TPE, "optimization_optuna_tpe_example"),
            (
                EXAMPLE_SCIPY_DE,
                "optimization_scipy_differential_evolution_example",
            ),
            (EXAMPLE_SCIPY_COBYLA, "optimization_scipy_cobyla_example"),
            (EXAMPLE_SCIPY_DIRECT_L, "optimization_scipy_direct_l_example"),
            (
                EXAMPLE_SCIPY_DUAL_ANNEALING,
                "optimization_scipy_dual_annealing_example",
            ),
            (EXAMPLE_OPTAX_ADAM, "optimization_optax_adam_example"),
            (EXAMPLE_OPTAX_RPROP, "optimization_optax_rprop_example"),
            (
                EXAMPLE_OPTAX_SGD_NESTEROV,
                "optimization_optax_sgd_nesterov_example",
            ),
        ):
            with TemporaryDirectory() as tmp:
                out = Path(tmp) / experiment_id
                prepare_experiment(config_path=path, output=out)
                self.assertTrue((out / "manifest.json").is_file())

    def test_rbfopt_example_reaches_only_runtime_solver_preflight(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "optimization_rbfopt_example"
            prepare_experiment(config_path=EXAMPLE_RBFOPT, output=out)
            self.assertTrue((out / "manifest.json").is_file())

        config = load_experiment_config(EXAMPLE_RBFOPT)
        params = algorithm_parameters_for(config.experiment.algorithms[0])
        with self.assertRaises(AlgorithmDependencyError) as ctx:
            create_algorithm("RBFOPT", params, random_seed=1)
        message = str(ctx.exception)
        self.assertTrue(
            "minlp_solver_path" in message
            or "nlp_solver_path" in message
            or "algorithms-rbfopt" in message,
            msg=message,
        )


if __name__ == "__main__":
    unittest.main()
