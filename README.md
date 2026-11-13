# hpdesignbench

Benchmarking framework for discrete structural design of prestressed CFRP hyperbolic-paraboloid shell slabs. Version 1.0.0 is the citable snapshot that can rerun the associated paper. Finished catalogues, structural libraries, and optimisation logs are not in this repository. The configs under `configs/release/` produce them.

Structural checks use a simply supported one-way strip idealization. They are not a full three-dimensional approval of an HP shell.

## Citation

Please cite both the software and the associated paper.

The software archive is the Zenodo record of the `v1.0.0` release. Its DOI is written into `CITATION.cff` before that tag is created. Until that DOI is present, cite the git tag `v1.0.0` of this repository together with the paper.

The associated paper, under its working title, is:

> Dombrowski, M., Melcer, E., Merz, P., Schlaich, M., & Wortmann, T. (working title). Optimisation methods for discrete structural design problems: A case study of HP shell slabs.

The title and the author list are replaced when the final reference exists. Correspondence: Max Dombrowski, m.dombrowski@tu-berlin.de.

Software authors are Max Dombrowski and Elliot Melcer. Elliot Melcer wrote the analysis model and its code, based on:

> Loutfi, J., Dombrowski, M., Merz, P., Eiz Eddin, A., & Schlaich, M. (2024). Interdisciplinary optimisation tool for doubly curved beam-like shell floors made of CFRP prestressed concrete. *Proceedings of the IASS Annual Symposium 2024*, 2024(11), 1–10.

Max Dombrowski wrote the optimisation model. The IASS 2024 paper is the basis of the analysis model. The preferred citation of this release is the paper above.

This snapshot is MIT licensed. The optional `algorithms-blackboxoptim` extra installs a GPL-3.0 package and is not the license of this snapshot. The paper configs do not use it.

## Installation

Python 3.12 or newer.

```bash
git clone https://github.com/Kildd/hpdesignbench.git
cd hpdesignbench
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[bruteforce,bruteforce-analysis,la,visualization-interactive]"
```

Linux and macOS: activate with `source .venv/bin/activate`, then use the same `pip` commands.

Algorithm libraries are optional extras named in `pyproject.toml` (`algorithms-pymoo`, `algorithms-optuna`, and the others). Install the extra for each solver you run.

## Where results go

Every release config sets `output_root = "../../data"`. That path is relative to the config file, so a checkout at `hpdesignbench/` writes to `hpdesignbench/data/`. `data/` is gitignored.

| Shelf | Contents |
| --- | --- |
| `data/structural_library/baseline/` | Shared structural library |
| `data/runs/structural_library/<id>/` | Library-build scaffolding |
| `data/runs/exhaustive/<id>/` | Brute-force catalogues |
| `data/runs/optimization/<id>/` | Optimisation logs |
| `data/predesign/<problem_id>/` | Predesign grid and warm-start lists |
| `data/results/la/exhaustive/<id>/` | Catalogue LON and CoLON |
| `data/results/optimization/<id>/` | The four HTML plots |

`hdb-experiment run --output` overrides the run directory. `hdb-plot interactive --out` overrides the HTML directory. Omit both to use the shelves above.

## Reproducing the paper

The 21 instances are spans 6750, 7425, 8100, 8775, 9450, 10125, and 10800 mm, each with loads A1, B1, and B2. Identifiers are `L{span}_{load}`. The reference set is `L6750_B2`, `L8100_B2`, `L9450_B2`, and `L10800_B2`. The other 17 instances are the validation set.

Run the configs from the repository root, in this order.

1. One structural library for all 21 instances:

```bash
hdb-experiment run --config configs/release/structural_library_21.toml
```

2. One full catalogue per instance. There are 21 files, `configs/release/brute_force_L6750_A1.toml` through `configs/release/brute_force_L10800_B2.toml`. Each catalogue has 53,374,464 rows and reads `data/structural_library/baseline`. Example:

```bash
hdb-experiment run --config configs/release/brute_force_L6750_B2.toml
```

3. Catalogue LON and CoLON on the four reference instances only. Production LON radii are 2, 3, and 4, on the penalised landscape `y_p`. Point each file at the finished catalogue, then:

```bash
hdb-la-analyze lon --config configs/release/lon_L6750_B2.toml
hdb-la-analyze colon --config configs/release/colon_L6750_B2.toml
```

Repeat for `L8100_B2`, `L9450_B2`, and `L10800_B2`.

4. Warm starts read `data/predesign/<problem_id>/initial_design.json` (penalty) or `initial_design_deb.json` (constraint-aware). The grid is 2,700 candidates and is outside the 15,000-evaluation solver budget. Build it after the structural library:

```bash
python -m core.ioh_core.workflows.predesign_cli --problem-id L6750_B2 --output-root data --structural-library data/structural_library/baseline
```

Repeat for every problem you run with a warm start.

5. Penalty optimisation, 20 repetitions, seed base 42, at most 15,000 evaluations, stopping at 1% above the sealed catalogue minimum `y_p*`. Seedless rank-1 warm starts are set to one repetition. The roster excludes SMT-EGO, PFN, and Hyperopt TPE.

```bash
hdb-experiment run --config configs/release/optimization_penalty_reference.toml
hdb-experiment run --config configs/release/optimization_penalty_validation.toml
```

6. Constraint-aware optimisation uses the same budget and the Deb comparator. `BLACKBOXOPT_GOSAC` is excluded. Optuna TPE, cold and warm, is on the reference config only.

```bash
hdb-experiment run --config configs/release/optimization_constrained_reference.toml
hdb-experiment run --config configs/release/optimization_constrained_validation.toml
```

## Figures

Four HTML plots, written by one command after an optimisation experiment has finished:

```bash
hdb-plot interactive --exp data/runs/optimization/penalty_reference
```

The files are `convergence`, `ecdf`, `ert`, and `heatmap`, under `data/results/optimization/penalty_reference/`. Install the `visualization-interactive` extra first.

Structural drawings are the six scripts in `drawings/`. From the repository root:

```bash
python drawings/plot_concrete_stress_strain.py
python drawings/plot_cfrp_stress_strain.py
python drawings/plot_uls_sls_laws_and_section.py
python drawings/plot_hp_section_integrator_meshes.py
python drawings/plot_mk_height_reinf_matrix.py
python drawings/plot_one_mk_panel.py
```

They write SVG and PNG files under `drawings/outputs/`.

## Small check

`configs/smoke/optimization_l6750_b2.toml` runs two seeded `RANDOM_SEARCH` repetitions of two evaluations on `L6750_B2`, with the sealed catalogue minimum as the 1% target. Two repetitions are the smallest set the HTML plots will draw. `scripts/fixed_design_l6750_b2.py` evaluates the lower-bound design and prints `y_p`.

```bash
python scripts/fixed_design_l6750_b2.py
hdb-experiment run --config configs/smoke/optimization_l6750_b2.toml
hdb-plot interactive --exp data/runs/optimization/identity_l6750_b2
```

## Tests

```bash
python -m unittest discover -s _mains/Testing -p "test_cursor_*.py"
```

`_mains` is not installed by `pip install -e .`.
