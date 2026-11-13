"""
ELA orchestration for sampled and exhaustive families.

Sampled path: one sealed repetition at a time, ``X_norm`` / ``z̃``, family
wrappers. Exhaustive path: DuckDB over complete BF Parquet, no ``z̃``.
Both publish tidy ``features.parquet`` plus ``_SUCCESS``. Importing this
module must not load DuckDB, PyArrow, pflacco, or SALib.
"""
from __future__ import annotations

import gc
import importlib
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.execution_core.atomic_io import write_bytes_atomic, write_text_atomic
from core.ioh_core.workflows.la.adapter import load_ela_schema, load_landscape_sample
from core.ioh_core.workflows.la.deps import import_duckdb, import_pyarrow
from core.ioh_core.workflows.la.lon_landscape import get_lon_landscape
from core.ioh_core.workflows.la.models import (
    ElaError,
    FeatureResult,
    LANDSCAPE_FEASIBILITY,
    LANDSCAPE_PHI,
    LANDSCAPE_Y_P,
    LANDSCAPE_Y_RAW,
    RESPONSE_VIEW_MINMAX,
    RESPONSE_VIEW_NA,
    SOURCE_BRUTE_FORCE,
    SOURCE_SAMPLING,
    X_VIEW_NA,
    X_VIEW_NORM,
    require_landscape,
)
from core.ioh_core.workflows.la.preprocess import (
    minmax_response,
    normalize_decision_coordinates,
)
from core.ioh_core.workflows.la.signatures import (
    analysis_signature,
    load_sampling_design_fields,
    load_schema_hash,
    sampling_design_signature,
)

SUCCESS_MARKER = "_SUCCESS"
FEATURES_PARQUET = "features.parquet"
DEFAULT_LANDSCAPES = (LANDSCAPE_Y_RAW, LANDSCAPE_Y_P, LANDSCAPE_PHI)
_SAMPLED_RESPONSE_LANDSCAPES = frozenset(DEFAULT_LANDSCAPES)
_SAMPLED_TOKENS = frozenset(
    {
        "meta-model",
        "distribution",
        "dispersion",
        "nbc",
        "level-set",
        "information-content",
        "cell-mapping",
        "sobol-length-scale",
        "constraints",
    }
)
_WIRED_EXHAUSTIVE_TOKENS = frozenset(
    {"constraints", "topology", "rfbx", "basins", "lon", "colon"}
)
_WIRED_CATALOGUE_TOKENS = _WIRED_EXHAUSTIVE_TOKENS | frozenset(
    {"ils-lon", "ils-colon"}
)
_EXHAUSTIVE_VIEW = "ela_parts"


def family_output_dir(output_dir: Path, token: str) -> Path:
    """Return ``<output_dir>/<cli-token>``."""
    return Path(output_dir) / str(token)


def lon_output_dir(output_dir: Path, landscape: str) -> Path:
    """Return the independent ``lon/<landscape>`` publication directory."""
    return family_output_dir(output_dir, "lon") / get_lon_landscape(landscape).token


def ils_lon_output_dir(output_dir: Path, landscape: str, kick: int) -> Path:
    """Return ``ils-lon/<landscape>/kick_<k>`` for one ILS-LON graph."""
    return (
        family_output_dir(output_dir, "ils-lon")
        / get_lon_landscape(landscape).token
        / f"kick_{int(kick)}"
    )


def ils_colon_output_dir(output_dir: Path, kick: int) -> Path:
    """Return ``ils_colon/kick_<k>`` for one ILS-CoLON graph."""
    return family_output_dir(output_dir, "ils_colon") / f"kick_{int(kick)}"


def run_ela_analysis(
    *,
    experiment_dir: Path,
    output_dir: Path,
    families: Sequence[str],
    source_type: str,
    landscapes: Sequence[str] | None = None,
    repetition: int | None = None,
    requested_constraint_names: Sequence[str] | None = None,
    cell_mapping: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run sampled ELA families, skipping any with a family ``_SUCCESS``.

    Parameters
    ----------
    experiment_dir : Path
        Published sampling experiment root (``dataset/_SUCCESS`` required).
    output_dir : Path
        Canonical ELA dir or an operator override. Family artifacts land in
        ``<output_dir>/<cli-token>/``.
    families : sequence of str
        Hyphen CLI tokens. Exhaustive-only tokens (topology, rfbx) are
        rejected; ``constraints`` is dual-source and allowed here.
    landscapes : sequence of str or None, optional
        Sampled response landscapes. Default ``y_raw``, ``y_p``, ``phi``.
    repetition : int or None, optional
        If set, that sampling repetition only; otherwise every distinct id.
    requested_constraint_names : sequence of str or None, optional
        Keep list forwarded to the adapter.
    cell_mapping : mapping or None, optional
        Required when ``cell-mapping`` will run: ``blocks``, ``force``,
        ``minimize``.

    Returns
    -------
    dict
        ``skipped`` / ``written`` family tokens and output paths.

    Raises
    ------
    ElaError
        Invalid selection, unpublished dataset, missing knobs, or corrupt
        resume marker.
    """
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir)
    tokens = tuple(str(name) for name in families)
    unknown = [name for name in tokens if name not in _SAMPLED_TOKENS]
    if unknown:
        raise ElaError(
            "orchestrate runs sampled ELA families only, "
            f"got {unknown}"
        )
    resolved_landscapes = resolve_ela_landscapes(landscapes)
    _require_published_dataset(experiment_dir)

    skipped: list[str] = []
    to_run: list[str] = []
    for token in tokens:
        family_dir = family_output_dir(output_dir, token)
        marker = family_dir / SUCCESS_MARKER
        parquet = family_dir / FEATURES_PARQUET
        if marker.is_file():
            if not parquet.is_file():
                raise ElaError(
                    f"landscape-analysis family {token!r} has {SUCCESS_MARKER} but no "
                    f"{FEATURES_PARQUET} under {family_dir}"
                )
            skipped.append(token)
            continue
        to_run.append(token)

    written: list[str] = []
    paths: dict[str, str] = {}
    if not to_run:
        return {
            "paths": paths,
            "skipped": skipped,
            "written": written,
        }

    if "cell-mapping" in to_run:
        _require_cell_mapping_table(cell_mapping)

    repetitions = list_sampling_repetitions(
        experiment_dir, repetition=repetition
    )
    problem_id = load_problem_id(experiment_dir)
    schema_hash = load_schema_hash(experiment_dir)
    sampling_hex = _sampling_signature_for_run(
        experiment_dir,
        repetitions=repetitions,
        problem_id=problem_id,
        schema_hash=schema_hash,
    )

    for token in to_run:
        family_dir = family_output_dir(output_dir, token)
        family_dir.mkdir(parents=True, exist_ok=True)
        _remove_success_marker(family_dir)
        try:
            rows: list[FeatureResult] = []
            if token == "constraints":
                from core.ioh_core.constraint_selection import (
                    select_constraint_names,
                )

                schema = load_ela_schema(experiment_dir)
                var_names = tuple(schema["var_names"])
                sealed = tuple(schema["constraint_names"])
                selected = select_constraint_names(
                    sealed, requested_constraint_names
                )
                if not selected:
                    raise ElaError(
                        "constrained ELA requires a non-empty constraint keep-list"
                    )
                for rep in repetitions:
                    design = load_sampling_design_fields(
                        experiment_dir, repetition=rep
                    )
                    rows.extend(
                        _run_constrained_family(
                            experiment_dir=experiment_dir,
                            var_names=var_names,
                            constraint_names=selected,
                            cardinality=int(design["sample_size"]),
                            problem_id=problem_id,
                            source_type=SOURCE_SAMPLING,
                            sampling_design_signature=sampling_hex,
                            seed=_optional_seed(design.get("seed")),
                            repetition=rep,
                            repetition_filter=rep,
                        )
                    )
            else:
                for rep in repetitions:
                    design = load_sampling_design_fields(
                        experiment_dir, repetition=rep
                    )
                    sample = load_landscape_sample(
                        experiment_dir,
                        repetition=rep,
                        requested_constraint_names=requested_constraint_names,
                        source_type=source_type,
                    )
                    _assert_design_matches_sample(design, sample)
                    _X_0, X_norm = normalize_decision_coordinates(
                        sample.X_index,
                        sample.lower_bounds,
                        sample.upper_bounds,
                    )
                    mapping_knobs = None
                    if token == "cell-mapping":
                        mapping_knobs = expand_cell_mapping_config(
                            cell_mapping,
                            n_dim=int(X_norm.shape[1]),
                        )
                    family_name, family_config = _analysis_family_fields(
                        token, cell_mapping=mapping_knobs
                    )
                    identity = {
                        "problem_id": problem_id,
                        "source_type": sample.source_type,
                        "source_dir": sample.experiment_dir,
                        "repetition": sample.repetition,
                        "sampling_design_signature": sampling_hex,
                        "seed": _optional_seed(design.get("seed")),
                    }
                    for landscape in resolved_landscapes:
                        analysis_hex = analysis_signature(
                            family=family_name,
                            landscape=landscape,
                            constraint_names=sample.constraint_names_selected,
                            x_view=X_VIEW_NORM,
                            response_view=RESPONSE_VIEW_MINMAX,
                            config=family_config,
                        )
                        view = minmax_response(_response_vector(sample, landscape))
                        rows.extend(
                            _call_sampled_family(
                                token,
                                X_norm=X_norm,
                                z_tilde=view.z_tilde,
                                z_min=view.z_min,
                                z_max=view.z_max,
                                landscape=landscape,
                                identity={
                                    **identity,
                                    "analysis_signature": analysis_hex,
                                },
                                cell_mapping=mapping_knobs,
                            )
                        )
            parquet_path = write_feature_parquet(family_dir / FEATURES_PARQUET, rows)
            marker_path = family_dir / SUCCESS_MARKER
            write_text_atomic(marker_path, "ok\n")
        except Exception:
            _remove_success_marker(family_dir)
            raise
        written.append(token)
        paths[token] = str(parquet_path)

    return {
        "paths": paths,
        "skipped": skipped,
        "written": written,
    }


def run_exhaustive_ela_analysis(
    *,
    experiment_dir: Path,
    output_dir: Path,
    families: Sequence[str],
    source_type: str,
    constraint_names: Sequence[str] | None = None,
    taus: Sequence[float] | None = None,
    rfbx: Mapping[str, Any] | None = None,
    write_attractors: bool = False,
    lon_radii: Sequence[int] | None = None,
    lon_landscape: str = LANDSCAPE_Y_P,
    ils_lon: Mapping[str, Any] | None = None,
    ils_colon: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run wired exhaustive landscape-analysis families on a complete brute-force catalogue.

    Parameters
    ----------
    experiment_dir : Path
        Published brute-force experiment root (``dataset/_SUCCESS``).
    output_dir : Path
        Canonical ELA dir or an operator override. Family artifacts land in
        ``<output_dir>/<cli-token>/``.
    families : sequence of str
        Hyphen CLI tokens. Only wired exhaustive tokens are accepted.
    source_type : str
        Must be ``brute_force``.
    constraint_names : sequence of str or None, optional
        Already-resolved keep list. ``None`` uses every sealed name.
    taus : sequence of float or None, optional
        Required when ``topology`` will run. Empty list is allowed (no
        ``G_τ`` rows). ``None`` means the operator omitted the knob.
    rfbx : mapping or None, optional
        Required when ``rfbx`` will run. Must contain ``n_walks``,
        ``n_transitions``, ``seed_base``, and ``start_policy``. ``None``
        means the operator omitted the knobs.
    write_attractors : bool, optional
        When true and ``basins`` runs, write ``attractors.parquet``
        (one row per core). Off by default. Does not change SUCCESS.
    lon_radii : sequence of int or None, optional
        Required when ``lon`` runs. Strictly increasing cumulative escape
        radii; there is no production default before the P1 STOP.
    lon_landscape : str, optional
        Scalar catalogue landscape for ``lon``: ``y_p`` (default) or ``y_raw``.
    ils_lon : mapping or None, optional
        Required when ``ils-lon`` runs. Must contain kicks, cap, patience,
        n_starts, seed_base, and start_policy.
    ils_colon : mapping or None, optional
        Required when ``ils-colon`` runs. Same knobs as ``ils-lon`` except
        landscape (always ``deb``).

    Returns
    -------
    dict
        ``skipped`` / ``written`` family tokens and output paths.

    Raises
    ------
    ElaError
        Invalid selection, unpublished dataset, incomplete catalogue, or
        corrupt resume marker.
    """
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir)
    if str(source_type) != SOURCE_BRUTE_FORCE:
        raise ElaError(
            "exhaustive landscape-analysis families require source_type='brute_force', "
            f"got {source_type!r}"
        )
    tokens = tuple(str(name) for name in families)
    unknown = [name for name in tokens if name not in _WIRED_CATALOGUE_TOKENS]
    if unknown:
        raise ElaError(
            "orchestrate catalogue ELA does not implement "
            f"{unknown}"
        )
    if not tokens:
        raise ElaError("exhaustive landscape analysis requires at least one family token")
    if "lon" in tokens:
        resolved_lon_radii = _require_lon_radii(lon_radii)
        resolved_lon_landscape = get_lon_landscape(lon_landscape).token
    else:
        resolved_lon_radii = None
        resolved_lon_landscape = LANDSCAPE_Y_P
    if "ils-lon" in tokens:
        resolved_ils = _require_ils_lon_config(ils_lon)
    else:
        resolved_ils = None
    if "ils-colon" in tokens:
        resolved_ils_colon = _require_ils_colon_config(ils_colon)
    else:
        resolved_ils_colon = None
    _require_published_dataset(experiment_dir)

    skipped: list[str] = []
    to_run: list[str] = []
    for token in tokens:
        if token == "ils-lon":
            assert resolved_ils is not None
            complete = True
            for kick in resolved_ils.kicks:
                family_dir = ils_lon_output_dir(
                    output_dir, resolved_ils.landscape, kick
                )
                marker = family_dir / SUCCESS_MARKER
                parquet = family_dir / FEATURES_PARQUET
                if marker.is_file():
                    if not parquet.is_file():
                        raise ElaError(
                            f"landscape-analysis family {token!r} has {SUCCESS_MARKER} but no "
                            f"{FEATURES_PARQUET} under {family_dir}"
                        )
                    missing = [
                        name
                        for name in ("nodes.parquet", "edges.parquet")
                        if not (family_dir / name).is_file()
                    ]
                    if missing:
                        raise ElaError(
                            f"landscape-analysis family 'ils-lon' has {SUCCESS_MARKER} but is "
                            f"missing required sidecars {missing}"
                        )
                else:
                    complete = False
            if complete:
                skipped.append(token)
            else:
                to_run.append(token)
            continue
        if token == "ils-colon":
            assert resolved_ils_colon is not None
            complete = True
            for kick in resolved_ils_colon.kicks:
                family_dir = ils_colon_output_dir(output_dir, kick)
                marker = family_dir / SUCCESS_MARKER
                parquet = family_dir / FEATURES_PARQUET
                if marker.is_file():
                    if not parquet.is_file():
                        raise ElaError(
                            f"landscape-analysis family {token!r} has {SUCCESS_MARKER} but no "
                            f"{FEATURES_PARQUET} under {family_dir}"
                        )
                    missing = [
                        name
                        for name in ("nodes.parquet", "edges.parquet")
                        if not (family_dir / name).is_file()
                    ]
                    if missing:
                        raise ElaError(
                            f"landscape-analysis family 'ils-colon' has {SUCCESS_MARKER} but is "
                            f"missing required sidecars {missing}"
                        )
                else:
                    complete = False
            if complete:
                skipped.append(token)
            else:
                to_run.append(token)
            continue
        family_dir = (
            lon_output_dir(output_dir, resolved_lon_landscape)
            if token == "lon"
            else family_output_dir(output_dir, token)
        )
        marker = family_dir / SUCCESS_MARKER
        parquet = family_dir / FEATURES_PARQUET
        if marker.is_file():
            if not parquet.is_file():
                raise ElaError(
                    f"landscape-analysis family {token!r} has {SUCCESS_MARKER} but no "
                    f"{FEATURES_PARQUET} under {family_dir}"
                )
            if token == "lon":
                missing = [
                    name
                    for name in (
                        "nodes.parquet",
                        "terminal_supports.parquet",
                        "overlaps.parquet",
                        "escape_edges.parquet",
                    )
                    if not (family_dir / name).is_file()
                ]
                if missing:
                    raise ElaError(
                        f"landscape-analysis family 'lon' has {SUCCESS_MARKER} but is "
                        f"missing required sidecars {missing}"
                    )
                _validate_lon_success_config(
                    parquet,
                    resolved_lon_radii,
                    resolved_lon_landscape,
                )
            if token == "colon":
                missing = [
                    name
                    for name in (
                        "nodes.parquet",
                        "terminal_supports.parquet",
                        "overlaps.parquet",
                        "escape_edges.parquet",
                        "census.json",
                    )
                    if not (family_dir / name).is_file()
                ]
                if missing:
                    raise ElaError(
                        f"landscape-analysis family 'colon' has {SUCCESS_MARKER} but is "
                        f"missing required sidecars {missing}"
                    )
            skipped.append(token)
            continue
        to_run.append(token)

    written: list[str] = []
    paths: dict[str, str] = {}
    if not to_run:
        return {
            "paths": paths,
            "skipped": skipped,
            "written": written,
        }

    if "topology" in to_run:
        if taus is None:
            raise ElaError(
                "topology requires --taus or [ela.topology] taus "
                "(no library default)"
            )
        from core.ioh_core.workflows.la.topology import parse_ela_taus

        resolved_taus = parse_ela_taus(taus)
    else:
        resolved_taus = None
    if "rfbx" in to_run:
        resolved_rfb = _require_rfb_config(rfbx)
    else:
        resolved_rfb = None
    schema = load_ela_schema(experiment_dir)
    var_names = tuple(schema["var_names"])
    sealed = tuple(schema["constraint_names"])
    selected = tuple(str(name) for name in constraint_names) if constraint_names else sealed
    if not selected:
        raise ElaError("exhaustive landscape analysis requires a non-empty constraint keep-list")
    space, problem_id = load_brute_force_index_space(experiment_dir)
    if tuple(space.var_names) != var_names:
        raise ElaError(
            "IndexSpace var_names do not match schema.json: "
            f"{tuple(space.var_names)!r} != {var_names!r}"
        )

    for token in to_run:
        if token == "ils-lon":
            assert resolved_ils is not None
            kick_paths = _publish_ils_lon_family(
                experiment_dir=experiment_dir,
                output_dir=output_dir,
                var_names=var_names,
                constraint_names=selected,
                lower_bounds=space.lower_bounds,
                upper_bounds=space.upper_bounds,
                cardinality=int(space.cardinality),
                problem_id=problem_id,
                config=resolved_ils,
            )
            written.append(token)
            paths.update(kick_paths)
            continue
        if token == "ils-colon":
            assert resolved_ils_colon is not None
            kick_paths = _publish_ils_colon_family(
                experiment_dir=experiment_dir,
                output_dir=output_dir,
                var_names=var_names,
                constraint_names=selected,
                lower_bounds=space.lower_bounds,
                upper_bounds=space.upper_bounds,
                cardinality=int(space.cardinality),
                problem_id=problem_id,
                config=resolved_ils_colon,
            )
            written.append(token)
            paths.update(kick_paths)
            continue
        family_dir = (
            lon_output_dir(output_dir, resolved_lon_landscape)
            if token == "lon"
            else family_output_dir(output_dir, token)
        )
        family_dir.mkdir(parents=True, exist_ok=True)
        try:
            if token == "constraints":
                rows = _run_constrained_family(
                    experiment_dir=experiment_dir,
                    var_names=var_names,
                    constraint_names=selected,
                    cardinality=int(space.cardinality),
                    problem_id=problem_id,
                    source_type=SOURCE_BRUTE_FORCE,
                    sampling_design_signature=None,
                    seed=None,
                    repetition=None,
                    repetition_filter=None,
                )
            elif token == "topology":
                rows = _run_topology_family(
                    experiment_dir=experiment_dir,
                    var_names=var_names,
                    constraint_names=selected,
                    lower_bounds=space.lower_bounds,
                    upper_bounds=space.upper_bounds,
                    cardinality=int(space.cardinality),
                    problem_id=problem_id,
                    taus=resolved_taus,
                )
            elif token == "rfbx":
                rows = _run_rfbx_family(
                    experiment_dir=experiment_dir,
                    var_names=var_names,
                    constraint_names=selected,
                    lower_bounds=space.lower_bounds,
                    upper_bounds=space.upper_bounds,
                    cardinality=int(space.cardinality),
                    problem_id=problem_id,
                    config=resolved_rfb,
                )
            elif token == "basins":
                rows, partition = _run_basins_family(
                    experiment_dir=experiment_dir,
                    var_names=var_names,
                    constraint_names=selected,
                    lower_bounds=space.lower_bounds,
                    upper_bounds=space.upper_bounds,
                    cardinality=int(space.cardinality),
                    problem_id=problem_id,
                )
                if write_attractors:
                    _write_attractors_parquet(
                        family_dir / "attractors.parquet", partition
                    )
            elif token == "lon":
                rows, lon_result = _run_lon_family(
                    experiment_dir=experiment_dir,
                    var_names=var_names,
                    constraint_names=selected,
                    lower_bounds=space.lower_bounds,
                    upper_bounds=space.upper_bounds,
                    cardinality=int(space.cardinality),
                    problem_id=problem_id,
                    radii=resolved_lon_radii,
                    landscape=resolved_lon_landscape,
                )
                _write_lon_sidecars(family_dir, lon_result)
            elif token == "colon":
                rows, colon_result = _run_colon_family(
                    experiment_dir=experiment_dir,
                    var_names=var_names,
                    constraint_names=selected,
                    lower_bounds=space.lower_bounds,
                    upper_bounds=space.upper_bounds,
                    cardinality=int(space.cardinality),
                    problem_id=problem_id,
                )
                _write_colon_sidecars(family_dir, colon_result)
            else:
                raise ElaError(f"exhaustive landscape-analysis token {token!r} is not wired")
            parquet_path = write_feature_parquet(family_dir / FEATURES_PARQUET, rows)
            write_text_atomic(family_dir / SUCCESS_MARKER, "ok\n")
        except Exception:
            _remove_success_marker(family_dir)
            raise
        written.append(token)
        paths[token] = str(parquet_path)

    return {
        "paths": paths,
        "skipped": skipped,
        "written": written,
    }


def load_brute_force_index_space(experiment_dir: Path) -> tuple[Any, str]:
    """
    Return sealed catalogue ``IndexSpace`` and ``problem_id``.

    Prepared experiments rebuild the space from the snapshot CSVs. Fixture
    datasets without ``source-config.toml`` may seal ``lower_bounds``,
    ``upper_bounds``, and ``problem_id`` on ``dataset/schema.json``.
    """
    from core.ioh_core.workflows.brute_force.models import IndexSpace, IndexSpaceError

    experiment_dir = Path(experiment_dir)
    schema = load_ela_schema(experiment_dir)
    var_names = tuple(schema["var_names"])
    if (experiment_dir / "source-config.toml").is_file():
        space, problem_id = _index_space_from_prepared_snapshot(experiment_dir)
        if tuple(space.var_names) != var_names:
            raise ElaError(
                "snapshot IndexSpace var_names do not match schema.json: "
                f"{tuple(space.var_names)!r} != {var_names!r}"
            )
        return space, problem_id
    lower = schema.get("lower_bounds")
    upper = schema.get("upper_bounds")
    problem_id = schema.get("problem_id")
    if lower is None or upper is None:
        raise ElaError(
            "catalogue bounds are required (prepared snapshot or "
            "schema.json lower_bounds/upper_bounds)"
        )
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ElaError(
            "schema.json problem_id is required when using schema bounds"
        )
    try:
        space = IndexSpace.from_bounds(var_names, lower, upper)
    except IndexSpaceError as exc:
        raise ElaError(str(exc)) from exc
    return space, problem_id.strip()


def _index_space_from_prepared_snapshot(experiment_dir: Path) -> tuple[Any, str]:
    from core.execution_core.lifecycle import experiment_paths, load_prepared_config
    from core.ioh_core.workflows.brute_force.planner import (
        BruteForcePlanError,
        load_problem_space_and_schema_from_csv_dir,
    )
    from core.ioh_core.workflows.optimization_runs.snapshot import (
        SnapshotError,
        resolve_snapshot_slab_dir,
    )

    try:
        config = load_prepared_config(experiment_dir)
    except Exception as exc:
        raise ElaError(
            f"could not load prepared brute-force config: {exc}"
        ) from exc
    if config.experiment.workflow != "brute_force":
        raise ElaError(
            "exhaustive landscape analysis requires workflow='brute_force', "
            f"got {config.experiment.workflow!r}"
        )
    problem_ids = tuple(str(item) for item in config.experiment.problem_ids)
    if len(problem_ids) != 1:
        raise ElaError(
            "exhaustive landscape analysis requires exactly one problem_id, "
            f"got {list(problem_ids)!r}"
        )
    problem_id = problem_ids[0]
    slab_type = str(config.experiment.slab_type)
    paths = experiment_paths(experiment_dir)
    try:
        slab_dir = resolve_snapshot_slab_dir(paths.inputs, slab_type)
        space, _schema_info, _info = load_problem_space_and_schema_from_csv_dir(
            slab_dir=slab_dir,
            problem_id=problem_id,
        )
    except (SnapshotError, BruteForcePlanError, OSError, ValueError) as exc:
        raise ElaError(
            f"could not rebuild sealed IndexSpace from snapshot: {exc}"
        ) from exc
    return space, problem_id


def _run_constrained_family(
    *,
    experiment_dir: Path,
    var_names: Sequence[str],
    constraint_names: Sequence[str],
    cardinality: int,
    problem_id: str,
    source_type: str = SOURCE_BRUTE_FORCE,
    sampling_design_signature: str | None = None,
    seed: int | None = None,
    repetition: int | None = None,
    repetition_filter: int | None = None,
) -> list[FeatureResult]:
    from core.ioh_core.workflows.la.constrained import (
        FAMILY,
        ela_constrained,
        ela_constrained_contribution,
        ela_constrained_cooccurrence,
        ela_constrained_distributions,
        ela_constrained_iz,
        ela_constrained_pairwise,
        ela_constrained_spearman,
    )

    analysis_hex = analysis_signature(
        family=FAMILY,
        landscape=LANDSCAPE_FEASIBILITY,
        constraint_names=constraint_names,
        x_view=X_VIEW_NA,
        response_view=RESPONSE_VIEW_NA,
        config={},
    )
    identity = {
        "problem_id": problem_id,
        "source_dir": Path(experiment_dir),
        "source_type": str(source_type),
        "var_names": tuple(var_names),
        "constraint_names": tuple(constraint_names),
        "cardinality": int(cardinality),
        "sampling_design_signature": sampling_design_signature,
        "analysis_signature": analysis_hex,
        "seed": seed,
        "repetition": repetition,
    }
    duckdb = import_duckdb()
    parts_glob = _parts_glob(experiment_dir)
    quoted = parts_glob.replace("'", "''")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 1")
        if repetition_filter is None:
            view_sql = (
                f"CREATE VIEW {_EXHAUSTIVE_VIEW} AS "
                f"SELECT * FROM read_parquet('{quoted}')"
            )
        else:
            rep = int(repetition_filter)
            view_sql = (
                f"CREATE VIEW {_EXHAUSTIVE_VIEW} AS "
                f"SELECT * FROM read_parquet('{quoted}') "
                f"WHERE repetition = {rep}"
            )
        try:
            connection.execute(view_sql)
        except Exception as exc:
            raise ElaError(
                f"could not open landscape Parquet parts at {parts_glob}: {exc}"
            ) from exc
        _assert_finite_objective_columns(connection, _EXHAUSTIVE_VIEW)
        rows: list[FeatureResult] = []
        for fn in (
            ela_constrained,
            ela_constrained_spearman,
            ela_constrained_iz,
            ela_constrained_distributions,
            ela_constrained_pairwise,
            ela_constrained_cooccurrence,
            ela_constrained_contribution,
        ):
            rows.extend(fn(connection=connection, table=_EXHAUSTIVE_VIEW, **identity))
            gc.collect()
    finally:
        connection.close()
    return rows


def _run_topology_family(
    *,
    experiment_dir: Path,
    var_names: Sequence[str],
    constraint_names: Sequence[str],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    cardinality: int,
    problem_id: str,
    taus: Sequence[float],
) -> list[FeatureResult]:
    from core.ioh_core.workflows.la.topology import ela_topology_features

    duckdb = import_duckdb()
    parts_glob = _parts_glob(experiment_dir)
    quoted = parts_glob.replace("'", "''")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 1")
        try:
            connection.execute(
                f"CREATE VIEW {_EXHAUSTIVE_VIEW} AS "
                f"SELECT * FROM read_parquet('{quoted}')"
            )
        except Exception as exc:
            raise ElaError(
                f"could not open brute-force Parquet parts at {parts_glob}: {exc}"
            ) from exc
        _assert_finite_objective_columns(connection, _EXHAUSTIVE_VIEW)
        return ela_topology_features(
            connection=connection,
            table=_EXHAUSTIVE_VIEW,
            constraint_names=constraint_names,
            var_names=var_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            cardinality=int(cardinality),
            problem_id=problem_id,
            source_dir=Path(experiment_dir),
            taus=taus,
            source_type=SOURCE_BRUTE_FORCE,
        )
    finally:
        connection.close()


def _run_rfbx_family(
    *,
    experiment_dir: Path,
    var_names: Sequence[str],
    constraint_names: Sequence[str],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    cardinality: int,
    problem_id: str,
    config: Any,
) -> list[FeatureResult]:
    from core.ioh_core.workflows.la.rfb import ela_rfb_features

    duckdb = import_duckdb()
    parts_glob = _parts_glob(experiment_dir)
    quoted = parts_glob.replace("'", "''")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 1")
        try:
            connection.execute(
                f"CREATE VIEW {_EXHAUSTIVE_VIEW} AS "
                f"SELECT * FROM read_parquet('{quoted}')"
            )
        except Exception as exc:
            raise ElaError(
                f"could not open brute-force Parquet parts at {parts_glob}: {exc}"
            ) from exc
        _assert_finite_objective_columns(connection, _EXHAUSTIVE_VIEW)
        return ela_rfb_features(
            connection=connection,
            table=_EXHAUSTIVE_VIEW,
            constraint_names=constraint_names,
            var_names=var_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            cardinality=int(cardinality),
            problem_id=problem_id,
            source_dir=Path(experiment_dir),
            config=config,
            source_type=SOURCE_BRUTE_FORCE,
        )
    finally:
        connection.close()


def _run_basins_family(
    *,
    experiment_dir: Path,
    var_names: Sequence[str],
    constraint_names: Sequence[str],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    cardinality: int,
    problem_id: str,
) -> tuple[list[Any], Any]:
    from core.ioh_core.workflows.la.basins import ela_basins_from_table

    duckdb = import_duckdb()
    parts_glob = _parts_glob(experiment_dir)
    quoted = parts_glob.replace("'", "''")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 1")
        try:
            connection.execute(
                f"CREATE VIEW {_EXHAUSTIVE_VIEW} AS "
                f"SELECT * FROM read_parquet('{quoted}')"
            )
        except Exception as exc:
            raise ElaError(
                f"could not open brute-force Parquet parts at {parts_glob}: {exc}"
            ) from exc
        _assert_finite_objective_columns(connection, _EXHAUSTIVE_VIEW)
        return ela_basins_from_table(
            connection=connection,
            table=_EXHAUSTIVE_VIEW,
            constraint_names=constraint_names,
            var_names=var_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            cardinality=int(cardinality),
            problem_id=problem_id,
            source_dir=Path(experiment_dir),
            source_type=SOURCE_BRUTE_FORCE,
        )
    finally:
        connection.close()


def _run_lon_family(
    *,
    experiment_dir: Path,
    var_names: Sequence[str],
    constraint_names: Sequence[str],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    cardinality: int,
    problem_id: str,
    radii: Sequence[int],
    landscape: str,
) -> tuple[list[Any], Any]:
    from core.ioh_core.workflows.la.lon import ela_lon_from_table

    duckdb = import_duckdb()
    parts_glob = _parts_glob(experiment_dir)
    quoted = parts_glob.replace("'", "''")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 1")
        try:
            connection.execute(
                f"CREATE VIEW {_EXHAUSTIVE_VIEW} AS "
                f"SELECT * FROM read_parquet('{quoted}')"
            )
        except Exception as exc:
            raise ElaError(
                f"could not open brute-force Parquet parts at {parts_glob}: {exc}"
            ) from exc
        _assert_finite_objective_columns(connection, _EXHAUSTIVE_VIEW)
        return ela_lon_from_table(
            connection=connection,
            table=_EXHAUSTIVE_VIEW,
            constraint_names=constraint_names,
            var_names=var_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            cardinality=int(cardinality),
            problem_id=problem_id,
            source_dir=Path(experiment_dir),
            radii=radii,
            landscape=landscape,
            source_type=SOURCE_BRUTE_FORCE,
        )
    finally:
        connection.close()


def _run_colon_family(
    *,
    experiment_dir: Path,
    var_names: Sequence[str],
    constraint_names: Sequence[str],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    cardinality: int,
    problem_id: str,
) -> tuple[list[Any], Any]:
    from core.ioh_core.workflows.la.colon import ela_colon_from_table

    duckdb = import_duckdb()
    parts_glob = _parts_glob(experiment_dir)
    quoted = parts_glob.replace("'", "''")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 1")
        try:
            connection.execute(
                f"CREATE VIEW {_EXHAUSTIVE_VIEW} AS "
                f"SELECT * FROM read_parquet('{quoted}')"
            )
        except Exception as exc:
            raise ElaError(
                f"could not open brute-force Parquet parts at {parts_glob}: {exc}"
            ) from exc
        _assert_finite_objective_columns(connection, _EXHAUSTIVE_VIEW)
        return ela_colon_from_table(
            connection=connection,
            table=_EXHAUSTIVE_VIEW,
            constraint_names=constraint_names,
            var_names=var_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            cardinality=int(cardinality),
            problem_id=problem_id,
            source_dir=Path(experiment_dir),
            source_type=SOURCE_BRUTE_FORCE,
        )
    finally:
        connection.close()


def _write_colon_sidecars(family_dir: Path, result: Any) -> None:
    from core.ioh_core.workflows.la.colon import (
        colon_escape_edge_records,
        colon_node_records,
        overlap_records,
    )

    pa = import_pyarrow()
    _write_records_parquet(
        family_dir / "nodes.parquet",
        colon_node_records(result),
        pa.schema(
            [
                ("node_id", pa.int32()),
                ("plateau_id", pa.int32()),
                ("min_linear", pa.int64()),
                ("y_raw", pa.float64()),
                ("node_size", pa.int64()),
                ("terminal_feasible", pa.bool_()),
                ("is_global", pa.bool_()),
                ("n_feasible", pa.int64()),
                ("basin_n_feasible", pa.int64()),
                ("basin_n_total", pa.int64()),
                ("basin_f_share", pa.float64()),
            ]
        ),
    )
    _write_records_parquet(
        family_dir / "overlaps.parquet",
        overlap_records(result),
        pa.schema(
            [
                ("node_a", pa.int32()),
                ("node_b", pa.int32()),
                ("design_count", pa.int64()),
            ]
        ),
    )
    _write_terminal_supports_parquet(
        family_dir / "terminal_supports.parquet",
        result,
    )
    _write_records_parquet(
        family_dir / "escape_edges.parquet",
        colon_escape_edge_records(result),
        pa.schema(
            [
                ("source_node_id", pa.int32()),
                ("target_node_id", pa.int32()),
                ("shell", pa.int32()),
                ("support_count", pa.int64()),
                ("source_shell_candidate_count", pa.int64()),
                ("source_shell_candidate_feasible_count", pa.int64()),
                ("source_shell_candidate_infeasible_count", pa.int64()),
                ("source_node_rank", pa.float64()),
                ("target_node_rank", pa.float64()),
                ("relation", pa.string()),
            ]
        ),
    )
    write_text_atomic(
        family_dir / "census.json",
        json.dumps(
            {
                "basin_census": "complete",
                "escapes": "complete",
                "funnels": "complete",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _write_lon_sidecars(family_dir: Path, result: Any) -> None:
    from core.ioh_core.workflows.la.lon import (
        escape_edge_records,
        lon_node_records,
        overlap_records,
    )

    pa = import_pyarrow()
    spec = get_lon_landscape(result.landscape)
    _write_records_parquet(
        family_dir / "nodes.parquet",
        lon_node_records(result),
        pa.schema(
            [
                ("node_id", pa.int32()),
                ("plateau_id", pa.int32()),
                ("min_linear", pa.int64()),
                (spec.token, pa.float64()),
                ("node_size", pa.int64()),
                (spec.global_marker_column, pa.bool_()),
                ("n_feasible", pa.int64()),
                ("feasible_share", pa.float64()),
                ("all_feasible", pa.bool_()),
                ("any_feasible", pa.bool_()),
                ("support_size", pa.int64()),
                ("basin_n_feasible", pa.int64()),
                ("basin_f_share", pa.float64()),
            ]
        ),
    )
    _write_records_parquet(
        family_dir / "overlaps.parquet",
        overlap_records(result),
        pa.schema(
            [
                ("node_a", pa.int32()),
                ("node_b", pa.int32()),
                ("design_count", pa.int64()),
            ]
        ),
    )
    _write_terminal_supports_parquet(
        family_dir / "terminal_supports.parquet",
        result,
    )
    _write_records_parquet(
        family_dir / "escape_edges.parquet",
        escape_edge_records(result),
        pa.schema(
            [
                ("source_node_id", pa.int32()),
                ("target_node_id", pa.int32()),
                ("shell", pa.int32()),
                ("support_count", pa.int64()),
                ("source_shell_candidate_count", pa.int64()),
                (spec.source_value_column, pa.float64()),
                (spec.target_value_column, pa.float64()),
                ("relation", pa.string()),
            ]
        ),
    )


def _write_terminal_supports_parquet(path: Path, result: Any) -> Path:
    """Stream the deterministic plateau-to-node relation, then publish atomically."""
    import numpy as np

    pa = import_pyarrow()
    parquet = import_pyarrow_parquet()
    schema = pa.schema(
        [
            ("plateau_id", pa.int32()),
            ("node_id", pa.int32()),
            ("plateau_size", pa.int64()),
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}_{uuid.uuid4().hex}"
    temporary = path.with_name(f"{path.name}.{token}.tmp")
    writer = parquet.ParquetWriter(temporary, schema)
    writer_closed = False
    try:
        chunk_size = 1_000_000
        single = result.support_single
        plateau_size = result.graph.plateau_size
        for start in range(0, int(single.size), chunk_size):
            stop = min(start + chunk_size, int(single.size))
            local = single[start:stop]
            keep = local >= 0
            if not bool(keep.any()):
                continue
            plateau_ids = (
                np.flatnonzero(keep).astype(np.int32)
                + np.int32(start)
            )
            writer.write_table(
                pa.Table.from_arrays(
                    [
                        pa.array(plateau_ids, type=pa.int32()),
                        pa.array(local[keep], type=pa.int32()),
                        pa.array(plateau_size[start:stop][keep], type=pa.int64()),
                    ],
                    schema=schema,
                )
            )
        buffered_plateaus: list[int] = []
        buffered_nodes: list[int] = []
        buffered_sizes: list[int] = []

        def flush_multiple() -> None:
            if not buffered_plateaus:
                return
            writer.write_table(
                pa.Table.from_arrays(
                    [
                        pa.array(buffered_plateaus, type=pa.int32()),
                        pa.array(buffered_nodes, type=pa.int32()),
                        pa.array(buffered_sizes, type=pa.int64()),
                    ],
                    schema=schema,
                )
            )
            buffered_plateaus.clear()
            buffered_nodes.clear()
            buffered_sizes.clear()

        for plateau_id, nodes in result.support_multiple.items():
            size = int(plateau_size[plateau_id])
            for node_id in nodes:
                buffered_plateaus.append(int(plateau_id))
                buffered_nodes.append(int(node_id))
                buffered_sizes.append(size)
            if len(buffered_plateaus) >= chunk_size:
                flush_multiple()
        flush_multiple()
        writer.close()
        writer_closed = True
        os.replace(temporary, path)
    finally:
        if not writer_closed:
            writer.close()
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
    return path


def _validate_lon_success_config(
    parquet_path: Path,
    expected_radii: Sequence[int],
    expected_landscape: str,
) -> None:
    from core.ioh_core.workflows.la.aggregate import load_feature_parquet

    rows = load_feature_parquet(parquet_path)
    if not rows or any(row.family != "lon" for row in rows):
        raise ElaError("published lon features contain an invalid family")
    if {row.landscape for row in rows} != {expected_landscape}:
        raise ElaError(
            "published lon SUCCESS landscape does not match requested "
            f"landscape {expected_landscape!r}"
        )
    observed = {
        tuple(int(value) for value in row.config_payload.get("radii", ()))
        for row in rows
    }
    expected = {tuple(int(value) for value in expected_radii)}
    if observed != expected:
        raise ElaError(
            "published lon SUCCESS radii do not match requested production "
            f"radii: observed={sorted(observed)!r}, expected={sorted(expected)!r}"
        )


def _write_records_parquet(path: Path, records: Sequence[Mapping[str, Any]], schema: Any) -> Path:
    pa = import_pyarrow()
    parquet = import_pyarrow_parquet()
    table = pa.Table.from_pylist(list(records), schema=schema)
    buffer = io.BytesIO()
    parquet.write_table(table, buffer)
    write_bytes_atomic(Path(path), buffer.getvalue())
    return Path(path)


def _write_attractors_parquet(path: Path, partition: Any) -> Path:
    from core.ioh_core.workflows.la.basins import attractor_records

    records = attractor_records(partition)
    if not records:
        raise ElaError(f"refusing to publish empty attractor table at {path}")
    pa = import_pyarrow()
    parquet = import_pyarrow_parquet()
    table = pa.Table.from_pylist(records)
    buffer = io.BytesIO()
    parquet.write_table(table, buffer)
    write_bytes_atomic(Path(path), buffer.getvalue())
    return Path(path)


def _assert_finite_objective_columns(connection: Any, table: str) -> None:
    try:
        row = connection.execute(
            f'SELECT COUNT(*) FROM {table} WHERE '
            f'y IS NULL OR "y_p" IS NULL OR '
            f"NOT isfinite(CAST(y AS DOUBLE)) OR "
            f'NOT isfinite(CAST("y_p" AS DOUBLE))'
        ).fetchone()
    except Exception as exc:
        raise ElaError(
            f"could not check finite y / y_p on exhaustive catalogue: {exc}"
        ) from exc
    if row is None:
        raise ElaError("finite y / y_p query returned no row")
    if int(row[0]) != 0:
        raise ElaError(
            "non-finite y or y_p in landscape source; reject whole source"
        )


def resolve_ela_landscapes(raw: Sequence[str] | None) -> tuple[str, ...]:
    """Default ``y_raw``, ``y_p``, ``phi``. Reject empties, duplicates, extras."""
    if raw is None:
        return DEFAULT_LANDSCAPES
    values = tuple(str(item) for item in raw)
    if not values:
        raise ElaError("landscapes must be a non-empty list")
    seen: set[str] = set()
    out: list[str] = []
    for name in values:
        token = require_landscape(name)
        if token not in _SAMPLED_RESPONSE_LANDSCAPES:
            raise ElaError(
                "sampled ELA landscapes must be y_raw, y_p, or phi, "
                f"got {token!r}"
            )
        if token in seen:
            raise ElaError(f"duplicate ELA landscape {token!r}")
        seen.add(token)
        out.append(token)
    return tuple(out)


def list_sampling_repetitions(
    experiment_dir: Path,
    *,
    repetition: int | None = None,
) -> tuple[int, ...]:
    """
    Distinct ``repetition`` ids from published parts, sorted.

    Does not concatenate rows. Optional ``repetition`` must exist.
    """
    duckdb = import_duckdb()
    parts_glob = _parts_glob(experiment_dir)
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 1")
        quoted = parts_glob.replace("'", "''")
        try:
            rows = connection.execute(
                f"SELECT DISTINCT repetition FROM read_parquet('{quoted}') "
                "ORDER BY repetition"
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            raise ElaError(
                f"could not list sampling repetitions at {parts_glob}: {exc}"
            ) from exc
    finally:
        connection.close()
    found = tuple(int(row[0]) for row in rows)
    if not found:
        raise ElaError(f"no sampling repetitions under {experiment_dir}")
    if repetition is None:
        return found
    chosen = int(repetition)
    if chosen not in found:
        raise ElaError(
            f"repetition {chosen} is not in the sampling dataset "
            f"{tuple(found)}"
        )
    return (chosen,)


def load_problem_id(experiment_dir: Path) -> str:
    """Require a single non-empty ``problem_id`` column value."""
    duckdb = import_duckdb()
    parts_glob = _parts_glob(experiment_dir)
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 1")
        quoted = parts_glob.replace("'", "''")
        try:
            rows = connection.execute(
                f"SELECT DISTINCT problem_id FROM read_parquet('{quoted}')"
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            raise ElaError(
                f"could not read problem_id at {parts_glob}: {exc}"
            ) from exc
    finally:
        connection.close()
    values = {str(row[0]) for row in rows if row[0] is not None and str(row[0])}
    if len(values) != 1:
        raise ElaError(
            "ELA requires exactly one problem_id in the sampling Parquet, "
            f"got {sorted(values)!r}"
        )
    return next(iter(values))


def write_feature_parquet(path: Path, rows: Sequence[FeatureResult]) -> Path:
    """Atomically write tidy feature rows as Parquet."""
    if not rows:
        raise ElaError(f"refusing to publish empty ELA features at {path}")
    pa = import_pyarrow()
    parquet = import_pyarrow_parquet()
    records = [_feature_record(row) for row in rows]
    table = pa.Table.from_pylist(records)
    buffer = io.BytesIO()
    parquet.write_table(table, buffer)
    write_bytes_atomic(Path(path), buffer.getvalue())
    return Path(path)


def import_pyarrow_parquet() -> Any:
    """Import ``pyarrow.parquet`` through the ela extra gate."""
    import_pyarrow()
    return importlib.import_module("pyarrow.parquet")


def expand_cell_mapping_config(
    raw: Mapping[str, Any] | None,
    *,
    n_dim: int,
) -> dict[str, Any]:
    """Expand ``blocks`` to length ``n_dim``; require ``force`` / ``minimize``."""
    table = _require_cell_mapping_table(raw)
    blocks_raw = table["blocks"]
    if isinstance(blocks_raw, int) and not isinstance(blocks_raw, bool):
        if blocks_raw < 1:
            raise ElaError("cell_mapping.blocks must be >= 1")
        blocks = tuple([int(blocks_raw)] * int(n_dim))
    elif isinstance(blocks_raw, list) and all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 1
        for item in blocks_raw
    ):
        blocks = tuple(int(item) for item in blocks_raw)
        if len(blocks) != int(n_dim):
            raise ElaError(
                "cell_mapping.blocks length must match dimension "
                f"({n_dim}), got {len(blocks)}"
            )
    else:
        raise ElaError(
            "cell_mapping.blocks must be a positive int or a list of "
            "positive ints"
        )
    return {
        "blocks": blocks,
        "force": bool(table["force"]),
        "minimize": bool(table["minimize"]),
    }


def coerce_cell_mapping_table(raw: Any) -> dict[str, Any]:
    """Parse a TOML ``[ela.cell_mapping]`` table (knobs may still be missing)."""
    if not isinstance(raw, dict):
        raise ElaError("cell_mapping must be a table")
    unknown = sorted(key for key in raw if key not in {"blocks", "force", "minimize"})
    if unknown:
        raise ElaError(f"unknown cell_mapping keys: {unknown}")
    out: dict[str, Any] = {}
    if "blocks" in raw:
        out["blocks"] = raw["blocks"]
    if "force" in raw:
        if not isinstance(raw["force"], bool):
            raise ElaError("cell_mapping.force must be a bool")
        out["force"] = raw["force"]
    if "minimize" in raw:
        if not isinstance(raw["minimize"], bool):
            raise ElaError("cell_mapping.minimize must be a bool")
        out["minimize"] = raw["minimize"]
    return out


def coerce_topology_table(raw: Any) -> dict[str, Any]:
    """Parse a TOML ``[ela.topology]`` table (``taus`` may still be missing)."""
    if not isinstance(raw, dict):
        raise ElaError("topology must be a table")
    unknown = sorted(key for key in raw if key not in {"taus"})
    if unknown:
        raise ElaError(f"unknown topology keys: {unknown}")
    out: dict[str, Any] = {}
    if "taus" in raw:
        from core.ioh_core.workflows.la.topology import parse_ela_taus

        out["taus"] = list(parse_ela_taus(raw["taus"]))
    return out


_RFBX_KEYS = frozenset(
    {"n_walks", "n_transitions", "seed_base", "start_policy"}
)


def coerce_rfbx_table(raw: Any) -> dict[str, Any]:
    """Parse a TOML ``[ela.rfbx]`` table (knobs may still be missing)."""
    if not isinstance(raw, dict):
        raise ElaError("rfbx must be a table")
    unknown = sorted(key for key in raw if key not in _RFBX_KEYS)
    if unknown:
        raise ElaError(f"unknown rfbx keys: {unknown}")
    out: dict[str, Any] = {}
    if "n_walks" in raw:
        out["n_walks"] = _require_rfbx_positive_int(raw["n_walks"], field="n_walks")
    if "n_transitions" in raw:
        out["n_transitions"] = _require_rfbx_positive_int(
            raw["n_transitions"], field="n_transitions"
        )
    if "seed_base" in raw:
        value = raw["seed_base"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ElaError(f"seed_base must be a non-negative int, got {value!r}")
        out["seed_base"] = value
    if "start_policy" in raw:
        value = raw["start_policy"]
        if not isinstance(value, str) or not value.strip():
            raise ElaError("rfbx start_policy must be a non-empty string")
        out["start_policy"] = value
    return out


def coerce_basins_table(raw: Any) -> dict[str, Any]:
    """Parse a TOML ``[ela.basins]`` table."""
    if not isinstance(raw, dict):
        raise ElaError("basins must be a table")
    unknown = sorted(key for key in raw if key not in {"write_attractors"})
    if unknown:
        raise ElaError(f"unknown basins keys: {unknown}")
    out: dict[str, Any] = {}
    if "write_attractors" in raw:
        value = raw["write_attractors"]
        if not isinstance(value, bool):
            raise ElaError("basins.write_attractors must be a boolean")
        out["write_attractors"] = value
    return out


def coerce_ils_lon_table(
    raw: Any, *, base: Path | None = None
) -> dict[str, Any]:
    """Parse a TOML ``[ela.ils_lon]`` table (knobs may still be missing)."""
    if not isinstance(raw, dict):
        raise ElaError("ils_lon must be a table")
    allowed = {
        "kicks",
        "cap",
        "patience",
        "n_starts",
        "seed_base",
        "start_policy",
        "landscape",
        "sampling_dir",
        "repetition",
    }
    unknown = sorted(key for key in raw if key not in allowed)
    if unknown:
        raise ElaError(f"unknown ils_lon keys: {unknown}")
    out: dict[str, Any] = {}
    if "kicks" in raw:
        if not isinstance(raw["kicks"], (list, tuple)):
            raise ElaError("ils_lon kicks must be a list of positive integers")
        out["kicks"] = [int(value) for value in raw["kicks"]]
    for key in ("cap", "patience", "n_starts", "seed_base", "repetition"):
        if key in raw:
            out[key] = raw[key]
    if "start_policy" in raw:
        out["start_policy"] = str(raw["start_policy"])
    if "landscape" in raw:
        out["landscape"] = get_lon_landscape(raw["landscape"]).token
    if "sampling_dir" in raw:
        path = Path(str(raw["sampling_dir"]))
        if not path.is_absolute() and base is not None:
            path = base / path
        out["sampling_dir"] = path
    return out


def _require_ils_lon_config(raw: Mapping[str, Any] | None) -> Any:
    from core.ioh_core.workflows.la.ils_lon import IlsLonConfig

    if raw is None:
        raise ElaError(
            "ils-lon requires --ils-* flags or [ela.ils_lon] "
            "(no library default)"
        )
    table = dict(raw)
    missing = [
        key
        for key in ("kicks", "cap", "patience", "n_starts", "seed_base", "start_policy")
        if key not in table
    ]
    if missing:
        raise ElaError(f"ils-lon requires fields {missing}")
    sampling_dir = table.get("sampling_dir")
    return IlsLonConfig(
        kicks=tuple(int(value) for value in table["kicks"]),
        cap=int(table["cap"]),
        patience=int(table["patience"]),
        n_starts=int(table["n_starts"]),
        seed_base=int(table["seed_base"]),
        start_policy=str(table["start_policy"]),
        landscape=str(table.get("landscape", LANDSCAPE_Y_P)),
        sampling_dir=Path(sampling_dir) if sampling_dir is not None else None,
        repetition=table.get("repetition"),
    )


def coerce_ils_colon_table(
    raw: Any, *, base: Path | None = None
) -> dict[str, Any]:
    """Parse a TOML ``[la.ils_colon]`` table (knobs may still be missing)."""
    if not isinstance(raw, dict):
        raise ElaError("ils_colon must be a table")
    allowed = {
        "kicks",
        "cap",
        "patience",
        "n_starts",
        "seed_base",
        "start_policy",
        "sampling_dir",
        "repetition",
    }
    unknown = sorted(key for key in raw if key not in allowed)
    if unknown:
        raise ElaError(f"unknown ils_colon keys: {unknown}")
    out: dict[str, Any] = {}
    if "kicks" in raw:
        if not isinstance(raw["kicks"], (list, tuple)):
            raise ElaError("ils_colon kicks must be a list of positive integers")
        out["kicks"] = [int(value) for value in raw["kicks"]]
    for key in ("cap", "patience", "n_starts", "seed_base", "repetition"):
        if key in raw:
            out[key] = raw[key]
    if "start_policy" in raw:
        out["start_policy"] = str(raw["start_policy"])
    if "sampling_dir" in raw:
        path = Path(str(raw["sampling_dir"]))
        if not path.is_absolute() and base is not None:
            path = base / path
        out["sampling_dir"] = path
    return out


def _require_ils_colon_config(raw: Mapping[str, Any] | None) -> Any:
    from core.ioh_core.workflows.la.ils_colon import IlsColonConfig

    if raw is None:
        raise ElaError(
            "ils-colon requires --ils-* flags or [la.ils_colon] "
            "(no library default)"
        )
    table = dict(raw)
    missing = [
        key
        for key in ("kicks", "cap", "patience", "n_starts", "seed_base", "start_policy")
        if key not in table
    ]
    if missing:
        raise ElaError(f"ils-colon requires fields {missing}")
    sampling_dir = table.get("sampling_dir")
    return IlsColonConfig(
        kicks=tuple(int(value) for value in table["kicks"]),
        cap=int(table["cap"]),
        patience=int(table["patience"]),
        n_starts=int(table["n_starts"]),
        seed_base=int(table["seed_base"]),
        start_policy=str(table["start_policy"]),
        sampling_dir=Path(sampling_dir) if sampling_dir is not None else None,
        repetition=table.get("repetition"),
    )


def _publish_ils_lon_family(
    *,
    experiment_dir: Path,
    output_dir: Path,
    var_names: Sequence[str],
    constraint_names: Sequence[str],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    cardinality: int,
    problem_id: str,
    config: Any,
) -> dict[str, str]:
    from core.ioh_core.workflows.la.ils_lon import ela_ils_lon_from_table
    from core.ioh_core.workflows.la.signatures import (
        load_sampling_design_fields,
        load_schema_hash,
        sampling_design_signature,
    )

    duckdb = import_duckdb()
    parts_glob = _parts_glob(experiment_dir)
    quoted = parts_glob.replace("'", "''")
    sampling_sig = None
    if config.start_policy == "sobol_net":
        sampling_dir = Path(config.sampling_dir)
        fields = load_sampling_design_fields(
            sampling_dir, repetition=int(config.repetition)
        )
        sampling_sig = sampling_design_signature(
            problem_id=str(fields.get("problem_id") or problem_id),
            schema_hash=load_schema_hash(sampling_dir),
            method=str(fields["method"]),
            sample_size=int(fields["sample_size"]),
            scramble=bool(fields["scramble"]),
            projection=str(fields["projection"]),
            delete_duplicates=bool(fields["delete_duplicates"]),
            var_names=tuple(fields["var_names"]),
            lower_bounds=tuple(int(v) for v in fields["lower_bounds"]),
            upper_bounds=tuple(int(v) for v in fields["upper_bounds"]),
        )
    from core.ioh_core.workflows.brute_force.models import IndexSpace
    from core.ioh_core.workflows.la.catalogue_table import load_catalogue_landscape
    from core.ioh_core.workflows.la.ils_lon import array_value_at

    connection = duckdb.connect()
    paths: dict[str, str] = {}
    getter = None
    try:
        connection.execute("SET threads = 1")
        try:
            connection.execute(
                f"CREATE VIEW {_EXHAUSTIVE_VIEW} AS "
                f"SELECT * FROM read_parquet('{quoted}')"
            )
        except Exception as exc:
            raise ElaError(
                f"could not open brute-force Parquet parts at {parts_glob}: {exc}"
            ) from exc
        _assert_finite_objective_columns(connection, _EXHAUSTIVE_VIEW)
        pending = []
        for kick in config.kicks:
            family_dir = ils_lon_output_dir(output_dir, config.landscape, kick)
            marker = family_dir / SUCCESS_MARKER
            parquet = family_dir / FEATURES_PARQUET
            if marker.is_file() and parquet.is_file():
                paths[f"ils-lon/{config.landscape}/kick_{int(kick)}"] = str(parquet)
                continue
            pending.append(int(kick))
        if pending:
            space = IndexSpace.from_bounds(var_names, lower_bounds, upper_bounds)
            values, _feasible = load_catalogue_landscape(
                connection,
                _EXHAUSTIVE_VIEW,
                space,
                expected_n=int(cardinality),
                landscape=config.landscape,
            )
            getter = array_value_at(values)
        connection.close()
        connection = None
        for kick in config.kicks:
            family_dir = ils_lon_output_dir(output_dir, config.landscape, kick)
            marker = family_dir / SUCCESS_MARKER
            parquet = family_dir / FEATURES_PARQUET
            if marker.is_file() and parquet.is_file():
                continue
            family_dir.mkdir(parents=True, exist_ok=True)
            try:
                rows, result = ela_ils_lon_from_table(
                    connection=None,
                    table=_EXHAUSTIVE_VIEW,
                    constraint_names=constraint_names,
                    var_names=var_names,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    cardinality=int(cardinality),
                    problem_id=problem_id,
                    source_dir=Path(experiment_dir),
                    config=config,
                    kick=int(kick),
                    value_at=getter,
                    source_type=SOURCE_BRUTE_FORCE,
                    sampling_design_signature=sampling_sig,
                )
                parquet_path = write_feature_parquet(parquet, rows)
                _write_ils_lon_sidecars(family_dir, result)
                write_text_atomic(marker, "ok\n")
            except Exception:
                _remove_success_marker(family_dir)
                raise
            paths[f"ils-lon/{config.landscape}/kick_{int(kick)}"] = str(parquet_path)
    finally:
        if connection is not None:
            connection.close()
    return paths


def _publish_ils_colon_family(
    *,
    experiment_dir: Path,
    output_dir: Path,
    var_names: Sequence[str],
    constraint_names: Sequence[str],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    cardinality: int,
    problem_id: str,
    config: Any,
) -> dict[str, str]:
    from core.ioh_core.workflows.la.catalogue_table import load_catalogue_y_phi_f
    from core.ioh_core.workflows.la.ils_colon import ela_ils_colon_from_table
    from core.ioh_core.workflows.la.signatures import (
        load_sampling_design_fields,
        load_schema_hash,
        sampling_design_signature,
    )

    duckdb = import_duckdb()
    parts_glob = _parts_glob(experiment_dir)
    quoted = parts_glob.replace("'", "''")
    sampling_sig = None
    if config.start_policy == "sobol_net":
        sampling_dir = Path(config.sampling_dir)
        fields = load_sampling_design_fields(
            sampling_dir, repetition=int(config.repetition)
        )
        sampling_sig = sampling_design_signature(
            problem_id=str(fields.get("problem_id") or problem_id),
            schema_hash=load_schema_hash(sampling_dir),
            method=str(fields["method"]),
            sample_size=int(fields["sample_size"]),
            scramble=bool(fields["scramble"]),
            projection=str(fields["projection"]),
            delete_duplicates=bool(fields["delete_duplicates"]),
            var_names=tuple(fields["var_names"]),
            lower_bounds=tuple(int(v) for v in fields["lower_bounds"]),
            upper_bounds=tuple(int(v) for v in fields["upper_bounds"]),
        )
    from core.ioh_core.workflows.brute_force.models import IndexSpace

    connection = duckdb.connect()
    paths: dict[str, str] = {}
    y = phi = feasible = None
    try:
        connection.execute("SET threads = 1")
        try:
            connection.execute(
                f"CREATE VIEW {_EXHAUSTIVE_VIEW} AS "
                f"SELECT * FROM read_parquet('{quoted}')"
            )
        except Exception as exc:
            raise ElaError(
                f"could not open brute-force Parquet parts at {parts_glob}: {exc}"
            ) from exc
        _assert_finite_objective_columns(connection, _EXHAUSTIVE_VIEW)
        pending = []
        for kick in config.kicks:
            family_dir = ils_colon_output_dir(output_dir, kick)
            marker = family_dir / SUCCESS_MARKER
            parquet = family_dir / FEATURES_PARQUET
            if marker.is_file() and parquet.is_file():
                paths[f"ils_colon/kick_{int(kick)}"] = str(parquet)
                continue
            pending.append(int(kick))
        if pending:
            space = IndexSpace.from_bounds(var_names, lower_bounds, upper_bounds)
            y, phi, feasible = load_catalogue_y_phi_f(
                connection,
                _EXHAUSTIVE_VIEW,
                space,
                constraint_names,
                expected_n=int(cardinality),
            )
        connection.close()
        connection = None
        for kick in config.kicks:
            family_dir = ils_colon_output_dir(output_dir, kick)
            marker = family_dir / SUCCESS_MARKER
            parquet = family_dir / FEATURES_PARQUET
            if marker.is_file() and parquet.is_file():
                continue
            family_dir.mkdir(parents=True, exist_ok=True)
            try:
                rows, result, _traps = ela_ils_colon_from_table(
                    connection=None,
                    table=_EXHAUSTIVE_VIEW,
                    constraint_names=constraint_names,
                    var_names=var_names,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    cardinality=int(cardinality),
                    problem_id=problem_id,
                    source_dir=Path(experiment_dir),
                    config=config,
                    kick=int(kick),
                    source_type=SOURCE_BRUTE_FORCE,
                    sampling_design_signature=sampling_sig,
                    y=y,
                    phi=phi,
                    feasible=feasible,
                )
                parquet_path = write_feature_parquet(parquet, rows)
                _write_ils_colon_sidecars(family_dir, result)
                write_text_atomic(marker, "ok\n")
            except Exception:
                _remove_success_marker(family_dir)
                raise
            paths[f"ils_colon/kick_{int(kick)}"] = str(parquet_path)
    finally:
        if connection is not None:
            connection.close()
    return paths


def _write_ils_lon_sidecars(family_dir: Path, result: Any) -> None:
    from core.ioh_core.workflows.la.ils_lon import ils_edge_records, ils_node_records

    pa = import_pyarrow()
    spec = get_lon_landscape(result.landscape)
    _write_records_parquet(
        family_dir / "nodes.parquet",
        ils_node_records(result),
        pa.schema(
            [
                ("node_id", pa.int32()),
                ("min_linear", pa.int64()),
                (spec.token, pa.float64()),
                ("node_size", pa.int64()),
                ("is_sample_best", pa.bool_()),
                ("is_funnel_floor", pa.bool_()),
            ]
        ),
    )
    _write_records_parquet(
        family_dir / "edges.parquet",
        ils_edge_records(result),
        pa.schema(
            [
                ("source_node_id", pa.int32()),
                ("target_node_id", pa.int32()),
                ("weight", pa.int64()),
                (spec.source_value_column, pa.float64()),
                (spec.target_value_column, pa.float64()),
                ("relation", pa.string()),
            ]
        ),
    )


def _write_ils_colon_sidecars(family_dir: Path, result: Any) -> None:
    from core.ioh_core.workflows.la.ils_colon import (
        ils_colon_edge_records,
        ils_colon_node_records,
    )

    pa = import_pyarrow()
    _write_records_parquet(
        family_dir / "nodes.parquet",
        ils_colon_node_records(result),
        pa.schema(
            [
                ("node_id", pa.int32()),
                ("min_linear", pa.int64()),
                ("h0", pa.int32()),
                ("h1", pa.float64()),
                ("node_size", pa.int64()),
                ("terminal_feasible", pa.bool_()),
                ("is_sample_best", pa.bool_()),
                ("is_funnel_floor", pa.bool_()),
            ]
        ),
    )
    _write_records_parquet(
        family_dir / "edges.parquet",
        ils_colon_edge_records(result),
        pa.schema(
            [
                ("source_node_id", pa.int32()),
                ("target_node_id", pa.int32()),
                ("weight", pa.int64()),
                ("relation", pa.string()),
            ]
        ),
    )


def coerce_lon_table(raw: Any) -> dict[str, Any]:
    """Parse a TOML ``[ela.lon]`` table."""
    if not isinstance(raw, dict):
        raise ElaError("lon must be a table")
    unknown = sorted(key for key in raw if key not in {"landscape", "radii"})
    if unknown:
        raise ElaError(f"unknown lon keys: {unknown}")
    out: dict[str, Any] = {}
    if "radii" in raw:
        out["radii"] = list(_require_lon_radii(raw["radii"]))
    if "landscape" in raw:
        out["landscape"] = get_lon_landscape(raw["landscape"]).token
    return out


def _require_lon_radii(raw: Sequence[int] | None) -> tuple[int, ...]:
    if raw is None:
        raise ElaError(
            "lon requires --radii or [ela.lon] radii (no production default)"
        )
    if not isinstance(raw, (list, tuple)):
        raise ElaError("lon radii must be a list of positive integers")
    values: list[int] = []
    for value in raw:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ElaError(f"lon radii must be positive integers, got {value!r}")
        values.append(int(value))
    if not values:
        raise ElaError("lon radii must not be empty")
    if values != sorted(set(values)):
        raise ElaError("lon radii must be strictly increasing without duplicates")
    from core.ioh_core.workflows.la.lon import PRODUCTION_RADII

    resolved = tuple(values)
    if resolved != PRODUCTION_RADII:
        raise ElaError(
            f"production lon radii are frozen at {list(PRODUCTION_RADII)}, "
            f"got {list(resolved)}"
        )
    return resolved


def _require_rfbx_positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ElaError(f"{field} must be a positive int, got {value!r}")
    return value


def _require_rfb_config(raw: Mapping[str, Any] | None) -> Any:
    from core.ioh_core.workflows.la.rfb import RfbWalkConfig

    if raw is None:
        raise ElaError(
            "rfbx requires --n-walks, --n-transitions, --seed-base, "
            "--start-policy or [ela.rfbx] (no library default)"
        )
    missing = [key for key in ("n_walks", "n_transitions", "seed_base", "start_policy") if key not in raw]
    if missing:
        raise ElaError(
            "rfbx requires --n-walks, --n-transitions, --seed-base, "
            f"--start-policy or [ela.rfbx] (missing {missing}; no library default)"
        )
    return RfbWalkConfig(
        n_walks=raw["n_walks"],
        n_transitions=raw["n_transitions"],
        seed_base=raw["seed_base"],
        start_policy=raw["start_policy"],
    )


def _require_cell_mapping_table(raw: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if raw is None:
        raise ElaError(
            "cell-mapping requires [ela.cell_mapping] with blocks, force, "
            "and minimize"
        )
    missing = [
        key for key in ("blocks", "force", "minimize") if key not in raw
    ]
    if missing:
        raise ElaError(
            "cell-mapping requires [ela.cell_mapping] fields "
            f"{missing}"
        )
    return raw


def _require_published_dataset(experiment_dir: Path) -> None:
    marker = Path(experiment_dir) / "dataset" / SUCCESS_MARKER
    if not marker.is_file():
        raise ElaError(
            "ELA requires published dataset/_SUCCESS under "
            f"{experiment_dir}"
        )


def _remove_success_marker(family_dir: Path) -> None:
    marker = Path(family_dir) / SUCCESS_MARKER
    if marker.is_file():
        marker.unlink()


def _parts_glob(experiment_dir: Path) -> str:
    return str((Path(experiment_dir) / "dataset" / "parts" / "*.parquet").as_posix())


def _response_vector(sample: Any, landscape: str) -> Any:
    if landscape == LANDSCAPE_Y_RAW:
        return sample.y_raw
    if landscape == LANDSCAPE_Y_P:
        return sample.y_p
    if landscape == LANDSCAPE_PHI:
        return sample.phi
    raise ElaError(f"unsupported sampled landscape {landscape!r}")


def _feature_record(row: FeatureResult) -> dict[str, Any]:
    return {
        "problem_id": row.problem_id,
        "source_type": row.source_type,
        "source_dir": str(row.source_dir),
        "repetition": row.repetition,
        "sample_id": row.sample_id,
        "walk_id": row.walk_id,
        "landscape": row.landscape,
        "response_view": row.response_view,
        "z_min": row.z_min,
        "z_max": row.z_max,
        "x_view": row.x_view,
        "family": row.family,
        "feature_name": row.feature_name,
        "value": row.value,
        "status": row.status,
        "role": row.role,
        "n_input": row.n_input,
        "n_unique": row.n_unique,
        "runtime_s": row.runtime_s,
        "library": row.library,
        "library_version": row.library_version,
        "config_json": row.config_json,
        "config_hash": row.config_hash,
        "sampling_design_signature": row.sampling_design_signature,
        "analysis_signature": row.analysis_signature,
        "seed": row.seed,
    }


def _sampling_signature_for_run(
    experiment_dir: Path,
    *,
    repetitions: Sequence[int],
    problem_id: str,
    schema_hash: str,
) -> str:
    found: str | None = None
    for rep in repetitions:
        design = load_sampling_design_fields(experiment_dir, repetition=int(rep))
        design_problem = design.get("problem_id")
        if design_problem is not None and str(design_problem) != problem_id:
            raise ElaError(
                "design.json problem_id does not match sampling Parquet: "
                f"{design_problem!r} != {problem_id!r}"
            )
        digest = sampling_design_signature(
            problem_id=problem_id,
            schema_hash=schema_hash,
            method=design["method"],
            sample_size=design["sample_size"],
            scramble=design["scramble"],
            projection=design["projection"],
            delete_duplicates=design["delete_duplicates"],
            var_names=design["var_names"],
            lower_bounds=design["lower_bounds"],
            upper_bounds=design["upper_bounds"],
        )
        if found is None:
            found = digest
        elif digest != found:
            raise ElaError(
                "sampling_design_signature differs across repetitions"
            )
    if found is None:
        raise ElaError("no sampling repetitions to fingerprint")
    return found


def _assert_design_matches_sample(
    design: Mapping[str, Any], sample: Any
) -> None:
    names = tuple(str(item) for item in design["var_names"])
    lo = tuple(int(item) for item in design["lower_bounds"])
    hi = tuple(int(item) for item in design["upper_bounds"])
    if names != sample.var_names:
        raise ElaError("design.json var_names do not match schema.json")
    if lo != sample.lower_bounds or hi != sample.upper_bounds:
        raise ElaError("design.json bounds do not match the loaded sample")
    if design.get("delete_duplicates") is not None and sample.delete_duplicates is not None:
        if bool(design["delete_duplicates"]) != bool(sample.delete_duplicates):
            raise ElaError("design.json delete_duplicates does not match the sample")


def _optional_seed(raw: Any) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        raise ElaError(f"design.json seed must be an int >= 0, got {raw!r}")
    return int(raw)


def _analysis_family_fields(
    token: str,
    *,
    cell_mapping: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if token == "meta-model":
        from core.ioh_core.workflows.la.ela_meta import FAMILY, NATIVE_CONFIG

        return FAMILY, dict(NATIVE_CONFIG)
    if token == "distribution":
        from core.ioh_core.workflows.la.ela_distribution import FAMILY, NATIVE_CONFIG

        return FAMILY, dict(NATIVE_CONFIG)
    if token == "dispersion":
        from core.ioh_core.workflows.la.ela_dispersion import FAMILY, NATIVE_CONFIG

        return FAMILY, dict(NATIVE_CONFIG)
    if token == "nbc":
        from core.ioh_core.workflows.la.ela_nbc import FAMILY, NATIVE_CONFIG

        return FAMILY, dict(NATIVE_CONFIG)
    if token == "level-set":
        from core.ioh_core.workflows.la.ela_level import FAMILY, NATIVE_CONFIG

        return FAMILY, dict(NATIVE_CONFIG)
    if token == "information-content":
        from core.ioh_core.workflows.la.ela_information_content import (
            FAMILY,
            NATIVE_CONFIG,
        )

        return FAMILY, dict(NATIVE_CONFIG)
    if token == "cell-mapping":
        from core.ioh_core.workflows.la.ela_cell_mapping import (
            FAMILY,
            cell_mapping_config_payload,
        )

        if cell_mapping is None:
            raise ElaError("cell-mapping knobs are required")
        return FAMILY, cell_mapping_config_payload(
            blocks=cell_mapping["blocks"],
            force=bool(cell_mapping["force"]),
            minimize=bool(cell_mapping["minimize"]),
        )
    if token == "sobol-length-scale":
        from core.ioh_core.workflows.la.ela_length_scale import FAMILY, NATIVE_CONFIG

        return FAMILY, dict(NATIVE_CONFIG)
    raise ElaError(f"unknown sampled family token {token!r}")


def _call_sampled_family(
    token: str,
    *,
    X_norm: Any,
    z_tilde: Any,
    z_min: float,
    z_max: float,
    landscape: str,
    identity: Mapping[str, Any],
    cell_mapping: Mapping[str, Any] | None,
) -> list[FeatureResult]:
    runner = _sampled_runner(token)
    payload: dict[str, Any] = {
        "X_norm": X_norm,
        "z_tilde": z_tilde,
        "z_min": z_min,
        "z_max": z_max,
        "landscape": landscape,
        "problem_id": identity["problem_id"],
        "source_type": identity["source_type"],
        "source_dir": identity["source_dir"],
        "repetition": identity["repetition"],
        "sampling_design_signature": identity["sampling_design_signature"],
        "analysis_signature": identity["analysis_signature"],
        "seed": identity.get("seed"),
    }
    if token == "cell-mapping":
        if cell_mapping is None:
            raise ElaError("cell-mapping knobs are required")
        payload.update(cell_mapping)
    return runner(**payload)


def _sampled_runner(token: str) -> Callable[..., list[FeatureResult]]:
    if token == "meta-model":
        from core.ioh_core.workflows.la.ela_meta import ela_meta

        return ela_meta
    if token == "distribution":
        from core.ioh_core.workflows.la.ela_distribution import ela_distribution

        return ela_distribution
    if token == "dispersion":
        from core.ioh_core.workflows.la.ela_dispersion import ela_dispersion

        return ela_dispersion
    if token == "nbc":
        from core.ioh_core.workflows.la.ela_nbc import ela_nbc

        return ela_nbc
    if token == "level-set":
        from core.ioh_core.workflows.la.ela_level import ela_level

        return ela_level
    if token == "information-content":
        from core.ioh_core.workflows.la.ela_information_content import (
            ela_information_content,
        )

        return ela_information_content
    if token == "cell-mapping":
        from core.ioh_core.workflows.la.ela_cell_mapping import ela_cell_mapping

        return ela_cell_mapping
    if token == "sobol-length-scale":
        from core.ioh_core.workflows.la.ela_length_scale import ela_length_scale

        return ela_length_scale
    raise ElaError(f"unknown sampled family token {token!r}")
