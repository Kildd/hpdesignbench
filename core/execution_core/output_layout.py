"""
Operator data ROOT layout: runs vs results under ``output_root``.

``output_root`` is the data ROOT (not the git checkout). Run trees live
under ``runs/<bucket>/<experiment.id>/``. Derived characterisation lives
under ``results/``. See ADR 0008 and ADR 0009.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from core.execution_core.atomic_io import write_json_atomic

RUNS_DIRNAME = "runs"
RESULTS_DIRNAME = "results"
LA_DIRNAME = "la"
SAMPLING_SHELF = "sampling"
EXHAUSTIVE_SHELF = "exhaustive"
SENSITIVITY_ANALYSIS_DIRNAME = "sensitivity-analysis"
SALTELLI_DIRNAME = "saltelli"
FACTORIAL_DIRNAME = "factorial"
CATALOGUE_ANALYSIS_DIRNAME = "catalogue-analysis"
OPTIMIZATION_RESULTS_DIRNAME = "optimization"
STRUCTURAL_LIBRARY_DIRNAME = "structural_library"
BASELINE_LIBRARY_NAME = "baseline"
NESTED_PREFIX_DIAG_DIRNAME = "nested-prefix-diag"
ILS_CATALOGUE_PREFIX_TOKEN = "catalogue_prefix"
LAYOUT_JSON_NAME = "layout.json"

WORKFLOW_RUN_BUCKET: Mapping[str, str] = {
    "brute_force": "exhaustive",
    "sampling": "sampling",
    "sensitivity_analysis": "sensitivity",
    "optimization_runs": "optimization",
    "structural_library": "structural_library",
}
RUN_BUCKETS = frozenset(WORKFLOW_RUN_BUCKET.values())

_LEGACY_DERIVED_MARKERS: tuple[tuple[str, ...], ...] = (
    ("aggregate", "analysis", "la"),
    ("aggregate", "analysis", "ela"),
    ("aggregate", "analysis", "factorial_sensitivity"),
    ("aggregate", "analysis", "variable_cooccurrence"),
    ("aggregate", "analysis", "constraint_decision_tree"),
    ("aggregate", "analysis", "la_nested_prefix_diag"),
)


class LayoutError(ValueError):
    """Invalid ROOT / results layout."""


def run_bucket_for_workflow(workflow: str) -> str:
    """Return the ``runs/`` bucket name for an ``hdb-experiment`` workflow."""
    bucket = WORKFLOW_RUN_BUCKET.get(str(workflow))
    if bucket is None:
        raise LayoutError(f"unknown workflow for run layout: {workflow!r}")
    return bucket


def layout_root_from_output_root(
    output_root: str | Path,
    *,
    cwd: Path | None = None,
) -> Path:
    """Resolve ``output_root`` to an absolute data ROOT."""
    base = Path(cwd) if cwd is not None else Path.cwd()
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        root = (base / root).resolve()
    else:
        root = root.resolve()
    return root


def experiment_dir_for(
    *,
    layout_root: Path,
    workflow: str,
    experiment_id: str,
) -> Path:
    """Return ``{ROOT}/runs/<bucket>/<id>``."""
    bucket = run_bucket_for_workflow(workflow)
    return Path(layout_root) / RUNS_DIRNAME / bucket / str(experiment_id)


def results_root_for(layout_root: Path) -> Path:
    """Return ``{ROOT}/results``."""
    return Path(layout_root) / RESULTS_DIRNAME


def default_structural_library_dir(layout_root: Path) -> Path:
    """Return ``{ROOT}/structural_library/baseline``."""
    return Path(layout_root) / STRUCTURAL_LIBRARY_DIRNAME / BASELINE_LIBRARY_NAME


def layout_sidecar_path(experiment_dir: Path) -> Path:
    """Return ``layout.json`` under an experiment root."""
    return Path(experiment_dir) / LAYOUT_JSON_NAME


def write_layout_sidecar(experiment_dir: Path, layout_root: Path) -> Path:
    """Persist the resolved ROOT next to the experiment manifest."""
    path = layout_sidecar_path(experiment_dir)
    write_json_atomic(
        path,
        {
            "layout_root": str(Path(layout_root).resolve()),
            "experiment_dir": str(Path(experiment_dir).resolve()),
        },
    )
    return path


def layout_root_from_sidecar(experiment_dir: Path) -> Path | None:
    """Load ROOT from ``layout.json`` when present."""
    path = layout_sidecar_path(experiment_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    raw = payload.get("layout_root")
    if not raw:
        return None
    return Path(str(raw)).expanduser().resolve()


def layout_root_from_experiment_dir(experiment_dir: Path) -> Path:
    """
    Infer ROOT from a prepared or legacy experiment directory.

    Prefers ``layout.json``, then ``…/runs/<bucket>/<id>``, then the
    parent (legacy ``{output_root}/{id}``).
    """
    experiment_dir = Path(experiment_dir).resolve()
    from_sidecar = layout_root_from_sidecar(experiment_dir)
    if from_sidecar is not None:
        return from_sidecar
    parent = experiment_dir.parent
    if parent.name in RUN_BUCKETS and parent.parent.name == RUNS_DIRNAME:
        return parent.parent.parent
    return parent


def resolve_results_root(
    experiment_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """
    Return the results directory.

    ``results_dir`` overrides ``{ROOT}/results`` when set.
    """
    if results_dir is not None:
        return Path(results_dir).expanduser().resolve()
    return results_root_for(layout_root_from_experiment_dir(experiment_dir))


def _has_part_sequence(path: Path, marker: tuple[str, ...]) -> bool:
    parts = Path(path).resolve().parts
    n = len(marker)
    for index in range(0, len(parts) - n + 1):
        if parts[index : index + n] == marker:
            return True
    return False


def is_legacy_derived_publication_path(path: Path) -> bool:
    """Return True if ``path`` sits on the historical ``aggregate/analysis`` derived roots."""
    return any(_has_part_sequence(path, marker) for marker in _LEGACY_DERIVED_MARKERS)


def assert_not_legacy_derived_publication(path: Path) -> None:
    """Refuse new LA / factorial / catalogue-analysis writes onto the archive root."""
    if is_legacy_derived_publication_path(path):
        raise LayoutError(
            "new landscape-analysis, factorial, nested-prefix, and "
            "catalogue-analysis writes must not target the canonical archive "
            f"{{experiment}}/aggregate/analysis/; got {path}"
        )


def require_results_publication(
    path: Path,
    *,
    error_type: type[BaseException] | None = None,
) -> Path:
    """Return ``path`` after refusing a legacy ``aggregate/analysis`` write.

    ``error_type`` wraps ``LayoutError`` for CLI surfaces that already raise
    ``ValueError`` or ``ElaError``.
    """
    path = Path(path)
    try:
        assert_not_legacy_derived_publication(path)
    except LayoutError as exc:
        if error_type is not None:
            raise error_type(str(exc)) from exc
        raise
    return path


def la_sampling_publication_dir(
    results_root: Path,
    experiment_id: str,
) -> Path:
    """Return ``results/la/sampling/{id}`` (family parent)."""
    return (
        Path(results_root) / LA_DIRNAME / SAMPLING_SHELF / str(experiment_id)
    )


def la_exhaustive_publication_dir(
    results_root: Path,
    experiment_id: str,
) -> Path:
    """Return ``results/la/exhaustive/{id}`` (family parent)."""
    return (
        Path(results_root) / LA_DIRNAME / EXHAUSTIVE_SHELF / str(experiment_id)
    )


def nested_prefix_diagnostic_dir(
    results_root: Path,
    sampling_id: str,
) -> Path:
    """Return ``results/la/sampling/{samp_id}/nested-prefix-diag``."""
    return (
        la_sampling_publication_dir(results_root, sampling_id)
        / NESTED_PREFIX_DIAG_DIRNAME
    )


def ils_pair_token(
    *,
    brute_force_id: str,
    sampling_id: str | None,
    start_policy: str | None,
) -> str:
    """Return ``{bf_id}__{samp_id}`` or ``{bf_id}__catalogue_prefix``."""
    policy = str(start_policy or "").strip().lower()
    if policy == "catalogue_prefix":
        return f"{brute_force_id}__{ILS_CATALOGUE_PREFIX_TOKEN}"
    if not sampling_id:
        raise LayoutError(
            "ILS Sobol-net starts require a sampling experiment id in the path"
        )
    return f"{brute_force_id}__{sampling_id}"


def la_ils_publication_dir(
    results_root: Path,
    *,
    brute_force_id: str,
    sampling_id: str | None,
    start_policy: str | None,
) -> Path:
    """Return the ILS family parent on the sampled-LA shelf."""
    token = ils_pair_token(
        brute_force_id=brute_force_id,
        sampling_id=sampling_id,
        start_policy=start_policy,
    )
    return Path(results_root) / LA_DIRNAME / SAMPLING_SHELF / token


def saltelli_publication_dir(results_root: Path, experiment_id: str) -> Path:
    """Return ``results/sensitivity-analysis/saltelli/{id}``."""
    return (
        Path(results_root)
        / SENSITIVITY_ANALYSIS_DIRNAME
        / SALTELLI_DIRNAME
        / str(experiment_id)
    )


def factorial_publication_dir(results_root: Path, experiment_id: str) -> Path:
    """Return ``results/sensitivity-analysis/factorial/{id}``."""
    return (
        Path(results_root)
        / SENSITIVITY_ANALYSIS_DIRNAME
        / FACTORIAL_DIRNAME
        / str(experiment_id)
    )


def catalogue_analysis_publication_dir(
    results_root: Path,
    experiment_id: str,
) -> Path:
    """Return ``results/catalogue-analysis/{id}``."""
    return Path(results_root) / CATALOGUE_ANALYSIS_DIRNAME / str(experiment_id)


def optimization_publication_dir(results_root: Path, folder_name: str) -> Path:
    """Return ``results/optimization/{folder_name}``."""
    return Path(results_root) / OPTIMIZATION_RESULTS_DIRNAME / str(folder_name)


def default_optimization_plot_dir(experiment_dirs: Sequence[Path]) -> Path:
    """
    Return the default HTML directory for one or more optimisation experiments.

    One experiment uses its id. Several experiments that share a data root
    use ``<id1>__<id2>``. Experiments with different data roots are refused.
    """
    if not experiment_dirs:
        raise LayoutError("optimisation plots require at least one experiment")
    roots: list[Path] = []
    names: list[str] = []
    for experiment_dir in experiment_dirs:
        path = Path(experiment_dir)
        roots.append(layout_root_from_experiment_dir(path))
        names.append(path.name)
    unique_roots = {root.resolve() for root in roots}
    if len(unique_roots) != 1:
        raise LayoutError(
            "optimisation experiments do not share one data root; pass --out"
        )
    folder = names[0] if len(names) == 1 else "__".join(names)
    return optimization_publication_dir(results_root_for(roots[0]), folder)


def la_family_parent_dir(
    experiment_dir: Path,
    *,
    shelf: str,
    sampling_dir: Path | None = None,
    results_dir: Path | None = None,
    start_policy: str | None = None,
) -> Path:
    """
    Return the LA family parent for ``shelf`` ``sampling``, ``exhaustive``,
    or ``ils``.
    """
    experiment_dir = Path(experiment_dir)
    results_root = resolve_results_root(experiment_dir, results_dir=results_dir)
    token = str(shelf).strip().lower()
    sampling_id = Path(sampling_dir).name if sampling_dir is not None else None
    if token == "ils":
        path = la_ils_publication_dir(
            results_root,
            brute_force_id=experiment_dir.name,
            sampling_id=sampling_id,
            start_policy=start_policy,
        )
    elif token == SAMPLING_SHELF:
        path = la_sampling_publication_dir(results_root, experiment_dir.name)
    elif token == EXHAUSTIVE_SHELF:
        path = la_exhaustive_publication_dir(results_root, experiment_dir.name)
    else:
        raise LayoutError(f"unknown LA results shelf {shelf!r}")
    assert_not_legacy_derived_publication(path)
    return path


def default_nested_prefix_dir(
    experiment_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Nested-prefix diagnostic on the sampled-LA shelf."""
    results_root = resolve_results_root(experiment_dir, results_dir=results_dir)
    path = nested_prefix_diagnostic_dir(results_root, Path(experiment_dir).name)
    assert_not_legacy_derived_publication(path)
    return path
