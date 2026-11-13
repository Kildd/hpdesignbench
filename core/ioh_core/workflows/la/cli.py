"""
CLI for read-only landscape analysis.

Preferred console script: ``hdb-la-analyze``
Preferred module: ``python -m core.ioh_core.workflows.la``

Argparse subcommands follow ``hdb-bf-analyze``. Optional ``--config`` TOML
holds research knobs; CLI flags override TOML. This is not an
``hdb-experiment`` workflow. Sampled, exhaustive, ILS-LON, and ILS-CoLON family
tokens run from this CLI; ``run`` / ``aggregate`` are not family tokens.

Importing this module must not load DuckDB, pflacco, or SALib.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from core.ioh_core.constraint_selection import (
    ConstraintSelectionError,
    select_constraint_names,
    warn_if_constraint_subset,
)
from core.ioh_core.workflows.la.deps import ElaDependencyError
from core.ioh_core.workflows.la.models import ElaError

ELA_COMMANDS = (
    "constraints",
    "topology",
    "rfbx",
    "basins",
    "lon",
    "colon",
    "run",
    "aggregate",
)
_NON_FAMILY_COMMANDS = frozenset({"run", "aggregate"})
FAMILY_TOKENS = tuple(
    name for name in ELA_COMMANDS if name not in _NON_FAMILY_COMMANDS
)
EXHAUSTIVE_FAMILIES = frozenset(
    {"constraints", "topology", "rfbx", "basins", "lon", "colon"}
)
WIRED_EXHAUSTIVE_FAMILIES = frozenset(
    {"constraints", "topology", "rfbx", "basins", "lon", "colon"}
)
ILS_LON_FAMILIES = frozenset({"ils-lon"})
ILS_COLON_FAMILIES = frozenset({"ils-colon"})
ILS_FAMILIES = ILS_LON_FAMILIES | ILS_COLON_FAMILIES
DUAL_SOURCE_FAMILIES = frozenset({"constraints"})
SAMPLED_FAMILIES = (
    frozenset(FAMILY_TOKENS) - EXHAUSTIVE_FAMILIES - ILS_FAMILIES
)
SOURCE_SAMPLING = "sampling"
SOURCE_BRUTE_FORCE = "brute_force"
SOURCE_TYPE_VALUES = frozenset({SOURCE_SAMPLING, SOURCE_BRUTE_FORCE})
ANALYSIS_ROOT_LA = "la"
ANALYSIS_ROOTS = frozenset({ANALYSIS_ROOT_LA})
LEGACY_ANALYSIS_ROOT_ELA = "ela"
DEFAULT_ANALYSIS_PARENT = Path("aggregate") / "analysis"
DEFAULT_OUTPUT_RELATIVE = DEFAULT_ANALYSIS_PARENT / ANALYSIS_ROOT_LA
DEFAULT_BF_ANALYSIS_RELATIVE = Path("aggregate") / "analysis"
_ELA_TABLE_KEY = "la"
_ALLOWED_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "experiment_dir",
        "output_dir",
        "source_type",
        "repetition",
        "constraint_names",
        "landscapes",
        "families",
        "cell_mapping",
        "topology",
        "rfbx",
        "basins",
        "lon",
        "ils_lon",
        "ils_colon",
        "results_dir",
    }
)
_STRING_LIST_KEYS = frozenset({"constraint_names", "landscapes", "families"})


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that exits with code 1 on usage errors."""

    def error(self, message: str) -> None:  # noqa: D102
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def default_analysis_output_dir(
    experiment_dir: Path,
    analysis_root: str = ANALYSIS_ROOT_LA,
    *,
    families: Sequence[str] | None = None,
    source_type: str | None = None,
    sampling_dir: Path | None = None,
    results_dir: Path | None = None,
    start_policy: str | None = None,
) -> Path:
    """Return the LA family parent under ``{ROOT}/results/la/…``."""
    from core.execution_core.output_layout import (
        EXHAUSTIVE_SHELF,
        LayoutError,
        SAMPLING_SHELF,
        la_family_parent_dir,
    )

    token = str(analysis_root)
    if token not in ANALYSIS_ROOTS:
        raise ElaError(f"unknown analysis output root {token!r}")
    names = tuple(str(item) for item in (families or ()))
    source = source_type
    if source is None:
        try:
            source = resolve_ela_source_type(
                experiment_dir,
                cli_source_type=None,
                toml_source_type=None,
            )
        except ElaError:
            source = None
    if any(name in ILS_FAMILIES for name in names):
        shelf = "ils"
    elif source == SOURCE_SAMPLING or (
        not names and (Path(experiment_dir) / "inputs" / "sampling").is_dir()
    ):
        shelf = SAMPLING_SHELF
    else:
        shelf = EXHAUSTIVE_SHELF
    try:
        return la_family_parent_dir(
            experiment_dir,
            shelf=shelf,
            sampling_dir=sampling_dir,
            results_dir=results_dir,
            start_policy=start_policy,
        )
    except LayoutError as exc:
        raise ElaError(str(exc)) from exc


def default_ela_output_dir(experiment_dir: Path) -> Path:
    """
    Historical ``<experiment-dir>/aggregate/analysis/ela`` path.

    Not a write target. Kept so keep-list and nested-prefix runs can
    refuse leftover ``ela/`` trees. New publications use
    :func:`default_la_output_dir`.
    """
    return Path(experiment_dir) / DEFAULT_ANALYSIS_PARENT / LEGACY_ANALYSIS_ROOT_ELA


def default_la_output_dir(
    experiment_dir: Path,
    **kwargs: Any,
) -> Path:
    """Return the default LA family parent for ``experiment_dir``."""
    return default_analysis_output_dir(
        experiment_dir, ANALYSIS_ROOT_LA, **kwargs
    )


def default_bf_analysis_output_dir(experiment_dir: Path) -> Path:
    """Return ``<experiment-dir>/aggregate/analysis``."""
    return Path(experiment_dir) / DEFAULT_BF_ANALYSIS_RELATIVE


def assert_output_not_under_dataset(
    experiment_dir: Path, output_dir: Path
) -> None:
    """
    Refuse an output path on or under ``dataset/``.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root that owns ``dataset/``.
    output_dir : Path
        Resolved or unresolved output directory.

    Raises
    ------
    ElaError
        If ``output_dir`` is ``dataset/`` or a descendant of it.
    """
    dataset_root = (Path(experiment_dir) / "dataset").resolve()
    resolved = Path(output_dir).resolve()
    if resolved == dataset_root or dataset_root in resolved.parents:
        raise ElaError(f"output_dir must not be inside dataset/: {output_dir}")


def assert_subset_output_dir(
    experiment_dir: Path,
    output_dir: Path,
    *,
    output_dir_provided: bool,
) -> None:
    """
    Refuse a subset keep-list writing canonical analysis directories.

    Parameters
    ----------
    experiment_dir : Path
        Experiment root.
    output_dir : Path
        Resolved output directory (may be the default).
    output_dir_provided : bool
        Whether the operator set ``--output-dir`` or TOML ``output_dir``.

    Raises
    ------
    ElaError
        If output was omitted or resolves to a canonical analysis path.
    """
    if not output_dir_provided:
        raise ElaError(
            "constraint keep-list is a proper subset of sealed "
            "constraint_names; pass --output-dir so the subset run cannot "
            "overwrite canonical analysis artifacts"
        )
    resolved = Path(output_dir).resolve()
    forbidden = (
        default_ela_output_dir(experiment_dir).resolve(),
        default_la_output_dir(experiment_dir).resolve(),
        default_bf_analysis_output_dir(experiment_dir).resolve(),
    )
    if resolved in forbidden:
        raise ElaError(
            "constraint keep-list subset must not write canonical "
            f"analysis directories: {output_dir}"
        )


def build_parser(
    *,
    prog: str = "hdb-la-analyze",
    analysis_root: str = ANALYSIS_ROOT_LA,
) -> argparse.ArgumentParser:
    """
    Build the landscape-analysis argument parser.

    Parameters
    ----------
    prog : str, optional
        Console-script name shown in help.
    analysis_root : str, optional
        Default output token under ``aggregate/analysis/`` (``la`` only).
        Historical ``ela/`` is a forbidden leftover path, not an
        analysis-root choice.

    Returns
    -------
    argparse.ArgumentParser
        Root parser with family subcommands plus ``run`` and ``aggregate``.
    """
    if analysis_root not in ANALYSIS_ROOTS:
        raise ElaError(f"unknown analysis output root {analysis_root!r}")
    description = (
        "Read-only landscape analysis of sealed brute-force or sampling "
        "datasets. Does not evaluate candidates."
    )
    parser = _ArgumentParser(
        prog=prog,
        description=description,
    )
    parser.set_defaults(analysis_root=analysis_root)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ELA_COMMANDS:
        command = sub.add_parser(
            name,
            help=(
                "Summarize published family features.parquet"
                if name == "aggregate"
                else "Run a selected set of landscape-analysis families"
                if name == "run"
                else "Run the deprecated legacy disjoint basin family"
                if name == "basins"
                else f"Run the {name} landscape-analysis family"
            ),
        )
        _add_common_flags(command, analysis_root=analysis_root)
        if name in {"lon", "ils-lon", "run", "aggregate"}:
            if name == "ils-lon":
                landscape_help = (
                    "Scalar landscape for ils-lon: y_p (default) or y_raw. "
                    "CLI overrides [ela.ils_lon] landscape."
                )
            elif name == "lon":
                landscape_help = (
                    "Scalar landscape for lon: y_p (default) or y_raw. "
                    "CLI overrides [ela.lon] landscape."
                )
            else:
                landscape_help = (
                    "Scalar landscape for lon or ils-lon: y_p (default) or "
                    "y_raw. CLI overrides [ela.lon] / [ela.ils_lon] landscape."
                )
            command.add_argument(
                "--landscape",
                default=None,
                dest="lon_landscape",
                metavar="NAME",
                help=landscape_help,
            )
        if name == "run":
            command.add_argument(
                "--families",
                nargs="+",
                default=None,
                help=(
                    "CLI family tokens to run (overrides --config families). "
                    "No default all. Hyphen names only."
                ),
            )
        if name == "aggregate":
            command.add_argument(
                "--families",
                nargs="+",
                default=None,
                help=(
                    "Family tokens to summarize under --output-dir. "
                    "Default: every subdirectory that has features.parquet."
                ),
            )
            command.add_argument(
                "--allow-multiple-groups",
                action="store_true",
                help=(
                    "Write one summary group per signature pair. "
                    "Default: fail if more than one pair is present."
                ),
            )
    return parser


def main(
    argv: list[str] | Sequence[str] | None = None,
    *,
    prog: str = "hdb-la-analyze",
    analysis_root: str = ANALYSIS_ROOT_LA,
) -> int:
    """
    Entry point for ``hdb-la-analyze``.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; ``None`` means ``sys.argv[1:]``.
    prog : str, optional
        Console-script name shown in usage errors.
    analysis_root : str, optional
        Default output token under ``aggregate/analysis/``.

    Returns
    -------
    int
        ``0`` on success, ``1`` on usage/config errors, ``2`` when a listed
        exhaustive family is not wired yet.
    """
    parser = build_parser(prog=prog, analysis_root=analysis_root)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1

    command = str(args.command)
    resolved: dict[str, Any] | None = None
    try:
        resolved = _resolve_invocation(args)
        assert_output_not_under_dataset(
            resolved["experiment_dir"],
            resolved["output_dir"],
        )
        if resolved["constraint_set_is_subset"]:
            assert_subset_output_dir(
                resolved["experiment_dir"],
                resolved["output_dir"],
                output_dir_provided=bool(resolved["output_dir_provided"]),
            )
        if command != "aggregate":
            assert_families_match_source(
                resolved["families"],
                resolved["source_type"],
            )
        if command == "aggregate":
            outcome = _run_ela_aggregate(
                resolved,
                allow_multiple_groups=bool(
                    getattr(args, "allow_multiple_groups", False)
                ),
            )
            _emit_json(
                {
                    "command": command,
                    "constraint_names_sealed": resolved[
                        "constraint_names_sealed"
                    ],
                    "constraint_names_selected": resolved[
                        "constraint_names_selected"
                    ],
                    "constraint_set_is_subset": resolved[
                        "constraint_set_is_subset"
                    ],
                    "error": None,
                    "experiment_dir": str(resolved["experiment_dir"]),
                    "families": list(outcome["written"]),
                    "ok": True,
                    "output_dir": str(resolved["output_dir"]),
                    "paths": outcome["paths"],
                    "skipped": [],
                    "source_type": resolved["source_type"],
                    "written": list(outcome["written"]),
                },
                stream=sys.stderr,
            )
            return 0
        exhaustive = [
            name for name in resolved["families"] if name in EXHAUSTIVE_FAMILIES
        ]
        ils_tokens = [
            name for name in resolved["families"] if name in ILS_FAMILIES
        ]
        if resolved["source_type"] == SOURCE_SAMPLING:
            # Dual-source families (constraints) run on the sampled path.
            exhaustive = [
                name for name in exhaustive if name not in DUAL_SOURCE_FAMILIES
            ]
        catalogue = tuple(dict.fromkeys((*exhaustive, *ils_tokens)))
        unimplemented = [
            name
            for name in exhaustive
            if name not in WIRED_EXHAUSTIVE_FAMILIES
        ]
        if unimplemented:
            _emit_json(
                {
                    "command": command,
                    "constraint_names_sealed": resolved[
                        "constraint_names_sealed"
                    ],
                    "constraint_names_selected": resolved[
                        "constraint_names_selected"
                    ],
                    "constraint_set_is_subset": resolved[
                        "constraint_set_is_subset"
                    ],
                    "error": "not implemented yet",
                    "experiment_dir": str(resolved["experiment_dir"]),
                    "families": list(resolved["families"]),
                    "ok": False,
                    "output_dir": str(resolved["output_dir"]),
                    "source_type": resolved["source_type"],
                },
                stream=sys.stderr,
            )
            return 2
        sampled = [
            name for name in resolved["families"] if name in SAMPLED_FAMILIES
        ]
        if resolved["source_type"] == SOURCE_SAMPLING:
            sampled = [
                name
                for name in resolved["families"]
                if name in SAMPLED_FAMILIES or name in DUAL_SOURCE_FAMILIES
            ]
        if sampled:
            from core.ioh_core.workflows.la.orchestrate import run_ela_analysis

            outcome = run_ela_analysis(
                experiment_dir=resolved["experiment_dir"],
                output_dir=resolved["output_dir"],
                families=sampled,
                source_type=resolved["source_type"],
                landscapes=resolved["landscapes"],
                repetition=resolved["repetition"],
                requested_constraint_names=resolved[
                    "constraint_names_requested"
                ],
                cell_mapping=resolved["cell_mapping"],
            )
            _emit_json(
                {
                    "command": command,
                    "constraint_names_sealed": resolved[
                        "constraint_names_sealed"
                    ],
                    "constraint_names_selected": resolved[
                        "constraint_names_selected"
                    ],
                    "constraint_set_is_subset": resolved[
                        "constraint_set_is_subset"
                    ],
                    "error": None,
                    "experiment_dir": str(resolved["experiment_dir"]),
                    "families": list(resolved["families"]),
                    "ok": True,
                    "output_dir": str(resolved["output_dir"]),
                    "paths": outcome["paths"],
                    "skipped": list(outcome["skipped"]),
                    "source_type": resolved["source_type"],
                    "written": list(outcome["written"]),
                },
                stream=sys.stderr,
            )
            return 0
        if catalogue:
            from core.ioh_core.workflows.la.orchestrate import (
                run_exhaustive_ela_analysis,
            )

            outcome = run_exhaustive_ela_analysis(
                experiment_dir=resolved["experiment_dir"],
                output_dir=resolved["output_dir"],
                families=catalogue,
                source_type=resolved["source_type"],
                constraint_names=resolved["constraint_names_selected"],
                taus=resolved["taus"],
                rfbx=resolved["rfbx"],
                write_attractors=bool(resolved["write_attractors"]),
                lon_radii=resolved["lon_radii"],
                lon_landscape=resolved["lon_landscape"],
                ils_lon=resolved["ils_lon"],
                ils_colon=resolved["ils_colon"],
            )
            _emit_json(
                {
                    "command": command,
                    "constraint_names_sealed": resolved[
                        "constraint_names_sealed"
                    ],
                    "constraint_names_selected": resolved[
                        "constraint_names_selected"
                    ],
                    "constraint_set_is_subset": resolved[
                        "constraint_set_is_subset"
                    ],
                    "error": None,
                    "experiment_dir": str(resolved["experiment_dir"]),
                    "families": list(resolved["families"]),
                    "ok": True,
                    "output_dir": str(resolved["output_dir"]),
                    "paths": outcome["paths"],
                    "skipped": list(outcome["skipped"]),
                    "source_type": resolved["source_type"],
                    "written": list(outcome["written"]),
                },
                stream=sys.stderr,
            )
            return 0
        raise ElaError("no runnable ELA families in this invocation")
    except (
        ElaError,
        ElaDependencyError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        payload: dict[str, Any] = {
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
            "ok": False,
        }
        if resolved is not None:
            payload.update(
                {
                    "constraint_names_sealed": resolved[
                        "constraint_names_sealed"
                    ],
                    "constraint_names_selected": resolved[
                        "constraint_names_selected"
                    ],
                    "constraint_set_is_subset": resolved[
                        "constraint_set_is_subset"
                    ],
                    "experiment_dir": str(resolved["experiment_dir"]),
                    "families": list(resolved["families"]),
                    "output_dir": str(resolved["output_dir"]),
                    "source_type": resolved["source_type"],
                }
            )
        _emit_json(payload, stream=sys.stderr)
        return 1


def _add_common_flags(
    parser: argparse.ArgumentParser,
    *,
    analysis_root: str = ANALYSIS_ROOT_LA,
) -> None:
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help=(
            "Sealed brute-force or sampling experiment root. "
            "Required unless set in --config. CLI overrides TOML."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Alternate LA family parent directory (default: "
            f"{{output_root}}/results/la/{{sampling|exhaustive}}/…). "
            "Must not be under dataset/ or {{experiment}}/aggregate/analysis/. "
            "CLI overrides TOML."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        dest="results_dir",
        help=(
            "Override the results directory ({output_root}/results). "
            "Ignored when --output-dir is set. CLI overrides TOML."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional ELA research-knob TOML. Not an hdb-experiment config. "
            "CLI flags override TOML."
        ),
    )
    parser.add_argument(
        "--constraint-names",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Keep list of sealed constraint names (same relative order). "
            "Omitting names ignores them. CLI overrides TOML. "
            "A proper subset warns and requires a non-canonical --output-dir."
        ),
    )
    parser.add_argument(
        "--source-type",
        choices=sorted(SOURCE_TYPE_VALUES),
        default=None,
        help=(
            "sampling or brute_force. Overrides --config source_type. "
            "If omitted, inferred from manifest.json or inputs/sampling/."
        ),
    )
    parser.add_argument(
        "--landscapes",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Sampled response landscapes (default: y_raw y_p phi). "
            "CLI overrides TOML."
        ),
    )
    parser.add_argument(
        "--repetition",
        type=int,
        default=None,
        help=(
            "Run one sampling repetition. Omitting uses every distinct "
            "repetition. CLI overrides TOML."
        ),
    )
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=None,
        metavar="TAU",
        help=(
            "Fractional near-optimal tau values for topology. "
            "Required when topology runs. Empty is only allowed via "
            "TOML taus = []. No library default. CLI overrides TOML."
        ),
    )
    parser.add_argument(
        "--n-walks",
        type=int,
        default=None,
        dest="n_walks",
        help=(
            "RFB× walk count. Required when rfbx runs. "
            "No library default. CLI overrides TOML."
        ),
    )
    parser.add_argument(
        "--n-transitions",
        type=int,
        default=None,
        dest="n_transitions",
        help=(
            "RFB× steps per walk (denominator). Required when rfbx runs. "
            "No library default. CLI overrides TOML."
        ),
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        dest="seed_base",
        help=(
            "RFB× seed_base mixed into each walk seed. Required when "
            "rfbx runs. No library default. CLI overrides TOML."
        ),
    )
    parser.add_argument(
        "--write-attractors",
        action="store_true",
        default=None,
        help=(
            "Write la/basins/attractors.parquet (one row per core). "
            "Off by default. CLI overrides TOML. Ignored unless basins runs."
        ),
    )
    parser.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=None,
        metavar="D",
        help=(
            "Strictly increasing cumulative L1 escape radii for lon. "
            "Required when lon runs. No production default. CLI overrides TOML."
        ),
    )
    parser.add_argument(
        "--start-policy",
        default=None,
        dest="start_policy",
        help=(
            "RFB× start policy. Must be uniform_catalogue. Required when "
            "rfbx runs. No library default. CLI overrides TOML."
        ),
    )
    parser.add_argument(
        "--ils-kicks",
        nargs="+",
        type=int,
        default=None,
        metavar="K",
        dest="ils_kicks",
        help=(
            "ILS kick radii for ils-lon and ils-colon. Required when either "
            "family runs. Each kick is its own graph. CLI overrides TOML."
        ),
    )
    parser.add_argument(
        "--ils-cap",
        type=int,
        default=None,
        dest="ils_cap",
        help="ILS iteration cap per chain. Required when ils-lon or ils-colon runs.",
    )
    parser.add_argument(
        "--ils-patience",
        type=int,
        default=None,
        dest="ils_patience",
        help="Non-improving kick patience. Required when ils-lon or ils-colon runs.",
    )
    parser.add_argument(
        "--ils-n-starts",
        type=int,
        default=None,
        dest="ils_n_starts",
        help="Sobol prefix / catalogue-prefix start count for ils-lon and ils-colon.",
    )
    parser.add_argument(
        "--ils-seed-base",
        type=int,
        default=None,
        dest="ils_seed_base",
        help="Seed mixed with kick for ils-lon and ils-colon RNG.",
    )
    parser.add_argument(
        "--ils-start-policy",
        default=None,
        dest="ils_start_policy",
        help="ils-lon / ils-colon start policy: sobol_net or catalogue_prefix.",
    )
    parser.add_argument(
        "--sampling-dir",
        type=Path,
        default=None,
        dest="sampling_dir",
        help="Sealed sampling experiment root for ils-lon / ils-colon sobol_net starts.",
    )


def _run_ela_aggregate(
    resolved: Mapping[str, Any],
    *,
    allow_multiple_groups: bool,
) -> dict[str, Any]:
    """Summarize published family ``features.parquet`` files."""
    from core.ioh_core.workflows.la.aggregate import (
        FEATURES_PARQUET,
        load_feature_parquet,
        publish_family_summary,
        summarize_ela_rows,
    )
    from core.ioh_core.workflows.la.orchestrate import (
        family_output_dir,
        lon_output_dir,
    )

    output_dir = Path(resolved["output_dir"])
    lon_landscape = str(resolved["lon_landscape"])
    def output_for(token: str) -> Path:
        return (
            lon_output_dir(output_dir, lon_landscape)
            if token == "lon"
            else family_output_dir(output_dir, token)
        )

    tokens = list(resolved["families"])
    if not tokens:
        tokens = [
            name
            for name in FAMILY_TOKENS
            if (output_for(name) / FEATURES_PARQUET).is_file()
        ]
    if not tokens:
        raise ElaError(
            f"no ELA {FEATURES_PARQUET} found under {output_dir}"
        )
    written: list[str] = []
    paths: dict[str, str] = {}
    landscapes = resolved.get("landscapes")
    repetition = resolved.get("repetition")
    for token in tokens:
        family_dir = output_for(token)
        parquet = family_dir / FEATURES_PARQUET
        if not parquet.is_file():
            raise ElaError(f"missing {FEATURES_PARQUET} under {family_dir}")
        rows = load_feature_parquet(parquet)
        if landscapes:
            allowed = set(landscapes)
            rows = [row for row in rows if row.landscape in allowed]
            if not rows:
                raise ElaError(
                    f"no ELA rows for landscapes {tuple(landscapes)} "
                    f"in {token}"
                )
        if repetition is not None:
            rows = [row for row in rows if row.repetition == int(repetition)]
            if not rows:
                raise ElaError(
                    f"no ELA rows for repetition {repetition} in {token}"
                )
        summaries = summarize_ela_rows(
            rows, allow_multiple_groups=allow_multiple_groups
        )
        path = publish_family_summary(family_dir, summaries)
        written.append(token)
        paths[token] = str(path)
    return {"paths": paths, "written": written}


def resolve_ela_families(
    *,
    command: str,
    cli_families: Sequence[str] | None,
    toml_families: Sequence[str] | None,
) -> tuple[str, ...]:
    """
    Resolve CLI family tokens. Subcommands ignore TOML ``families``.

    ``run`` requires a non-empty list from ``--families`` or TOML.
    ``aggregate`` may omit the list and discover family directories.
    There is no ``all`` expansion. Duplicates and unknown names fail.
    """
    if command == "aggregate":
        raw = cli_families if cli_families is not None else toml_families
        if raw is None or len(tuple(raw)) == 0:
            return ()
        return _require_family_tokens(raw)
    if command != "run":
        if command not in FAMILY_TOKENS:
            raise ElaError(f"unknown ELA command {command!r}")
        return (command,)
    raw = cli_families if cli_families is not None else toml_families
    if raw is None or len(tuple(raw)) == 0:
        raise ElaError(
            "run requires --families or config families; there is no default all"
        )
    return _require_family_tokens(raw)


def _require_family_tokens(raw: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for name in raw:
        token = str(name)
        if token == "all":
            raise ElaError("family list must not include 'all'")
        if token in _NON_FAMILY_COMMANDS or token not in FAMILY_TOKENS:
            raise ElaError(
                "unknown family token "
                f"{token!r}; use hyphen CLI names such as lon"
            )
        if token in seen:
            raise ElaError(f"duplicate family token {token!r}")
        seen.add(token)
        out.append(token)
    return tuple(out)


def resolve_ela_source_type(
    experiment_dir: Path,
    *,
    cli_source_type: str | None,
    toml_source_type: str | None,
) -> str:
    """
    Resolve source_type without reading Parquet.

    Order: CLI, TOML, ``manifest.json`` workflow, then ``inputs/sampling/``.
    Do not infer brute_force from an empty directory.
    """
    if cli_source_type is not None:
        return _require_source_type(cli_source_type)
    if toml_source_type is not None:
        return _require_source_type(toml_source_type)
    manifest_path = Path(experiment_dir) / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ElaError(f"invalid manifest.json: {exc}") from exc
        workflow = None
        if isinstance(manifest, dict):
            workflow = manifest.get("workflow_type") or manifest.get("workflow")
        if workflow in SOURCE_TYPE_VALUES:
            return str(workflow)
        if workflow is not None:
            raise ElaError(f"unknown source_type {workflow!r}")
    if (Path(experiment_dir) / "inputs" / "sampling").is_dir():
        return SOURCE_SAMPLING
    raise ElaError(
        "cannot infer ELA source_type; pass --source-type or set it in --config"
    )


def assert_families_match_source(
    families: Sequence[str],
    source_type: str,
) -> None:
    """Refuse catalogue-walk families on sampling and sampled families on BF."""
    source = _require_source_type(source_type)
    exhaustive_only = [
        name
        for name in families
        if name in (EXHAUSTIVE_FAMILIES - DUAL_SOURCE_FAMILIES)
    ]
    ils = [name for name in families if name in ILS_FAMILIES]
    sampled = [name for name in families if name in SAMPLED_FAMILIES]
    if source == SOURCE_SAMPLING and exhaustive_only:
        raise ElaError(
            "exhaustive landscape-analysis families require a brute_force experiment, "
            f"got sampling with {exhaustive_only}"
        )
    if source == SOURCE_SAMPLING and ils:
        raise ElaError(
            "ils-lon and ils-colon require a brute_force catalogue experiment, "
            f"got sampling with {ils}"
        )
    if source == SOURCE_BRUTE_FORCE and sampled:
        raise ElaError(
            "sampled ELA families require a sampling experiment, "
            f"got brute_force with {sampled}"
        )


def _require_source_type(value: str) -> str:
    name = str(value)
    if name not in SOURCE_TYPE_VALUES:
        raise ElaError(f"unknown source_type {value!r}")
    return name


def _resolve_invocation(args: argparse.Namespace) -> dict[str, Any]:
    """Merge TOML and CLI flags; CLI wins. Require experiment_dir."""
    document: dict[str, Any] = {}
    config_path = getattr(args, "config", None)
    base: Path | None = None
    if config_path is not None:
        path = Path(config_path)
        document = _load_ela_toml(path)
        base = path.parent

    experiment_dir = _overlay_path(
        cli_value=getattr(args, "experiment_dir", None),
        toml_value=document.get("experiment_dir"),
        base=base,
        field="experiment_dir",
    )
    if experiment_dir is None:
        raise ElaError(
            "experiment_dir is required (pass --experiment-dir or set it in --config)"
        )

    families = resolve_ela_families(
        command=str(args.command),
        cli_families=getattr(args, "families", None),
        toml_families=document.get("families"),
    )
    requested = _overlay_constraint_names(
        cli_value=getattr(args, "constraint_names", None),
        toml_value=document.get("constraint_names"),
    )
    sealed = _load_sealed_constraint_names(experiment_dir)
    if requested is not None and sealed is None:
        raise ElaError(
            "constraint keep-list requires dataset/schema.json "
            f"under {experiment_dir}"
        )
    selected: tuple[str, ...] | None = None
    is_subset = False
    if sealed is not None:
        try:
            selected = select_constraint_names(sealed, requested)
        except ConstraintSelectionError as exc:
            raise ElaError(str(exc)) from exc
        warn_if_constraint_subset(sealed, selected)
        is_subset = tuple(selected) != tuple(sealed)

    source_type = resolve_ela_source_type(
        experiment_dir,
        cli_source_type=getattr(args, "source_type", None),
        toml_source_type=document.get("source_type"),
    )

    results_dir = _overlay_path(
        cli_value=getattr(args, "results_dir", None),
        toml_value=document.get("results_dir"),
        base=base,
        field="results_dir",
    )
    output_dir = _overlay_path(
        cli_value=getattr(args, "output_dir", None),
        toml_value=document.get("output_dir"),
        base=base,
        field="output_dir",
    )
    output_dir_provided = output_dir is not None
    if output_dir is None:
        sampling_dir = getattr(args, "sampling_dir", None)
        if sampling_dir is None:
            for key in ("ils_lon", "ils_colon"):
                table = document.get(key)
                if isinstance(table, Mapping) and table.get("sampling_dir"):
                    sampling_dir = Path(str(table["sampling_dir"]))
                    if not sampling_dir.is_absolute() and base is not None:
                        sampling_dir = base / sampling_dir
                    break
        start_policy = getattr(args, "ils_start_policy", None)
        if start_policy is None:
            for key in ("ils_lon", "ils_colon"):
                table = document.get(key)
                if isinstance(table, Mapping) and table.get("start_policy"):
                    start_policy = str(table["start_policy"])
                    break
        from core.execution_core.output_layout import LayoutError

        try:
            output_dir = default_analysis_output_dir(
                experiment_dir,
                str(getattr(args, "analysis_root", ANALYSIS_ROOT_LA)),
                families=families,
                source_type=source_type,
                sampling_dir=sampling_dir,
                results_dir=results_dir,
                start_policy=start_policy,
            )
        except LayoutError as exc:
            raise ElaError(str(exc)) from exc
    from core.execution_core.output_layout import require_results_publication

    output_dir = require_results_publication(output_dir, error_type=ElaError)

    landscapes = _overlay_landscapes(
        cli_value=getattr(args, "landscapes", None),
        toml_value=document.get("landscapes"),
    )
    repetition = _overlay_repetition(
        cli_value=getattr(args, "repetition", None),
        toml_value=document.get("repetition"),
    )
    cell_mapping = document.get("cell_mapping")
    taus = _overlay_taus(
        cli_value=getattr(args, "taus", None),
        toml_topology=document.get("topology"),
    )
    rfbx = _overlay_rfbx(
        cli_n_walks=getattr(args, "n_walks", None),
        cli_n_transitions=getattr(args, "n_transitions", None),
        cli_seed_base=getattr(args, "seed_base", None),
        cli_start_policy=getattr(args, "start_policy", None),
        toml_rfbx=document.get("rfbx"),
    )
    write_attractors = _overlay_write_attractors(
        cli_value=getattr(args, "write_attractors", None),
        toml_basins=document.get("basins"),
    )
    lon_radii = _overlay_lon_radii(
        cli_value=getattr(args, "radii", None),
        toml_lon=document.get("lon"),
    )
    lon_landscape = _overlay_lon_landscape(
        cli_value=getattr(args, "lon_landscape", None),
        toml_lon=document.get("lon"),
    )
    ils_lon = _overlay_ils_lon(
        cli_kicks=getattr(args, "ils_kicks", None),
        cli_cap=getattr(args, "ils_cap", None),
        cli_patience=getattr(args, "ils_patience", None),
        cli_n_starts=getattr(args, "ils_n_starts", None),
        cli_seed_base=getattr(args, "ils_seed_base", None),
        cli_start_policy=getattr(args, "ils_start_policy", None),
        cli_sampling_dir=getattr(args, "sampling_dir", None),
        cli_repetition=repetition,
        cli_landscape=getattr(args, "lon_landscape", None),
        toml_ils=document.get("ils_lon"),
        base=base,
    )
    ils_colon = _overlay_ils_colon(
        cli_kicks=getattr(args, "ils_kicks", None),
        cli_cap=getattr(args, "ils_cap", None),
        cli_patience=getattr(args, "ils_patience", None),
        cli_n_starts=getattr(args, "ils_n_starts", None),
        cli_seed_base=getattr(args, "ils_seed_base", None),
        cli_start_policy=getattr(args, "ils_start_policy", None),
        cli_sampling_dir=getattr(args, "sampling_dir", None),
        cli_repetition=repetition,
        toml_ils=document.get("ils_colon"),
        base=base,
    )

    return {
        "cell_mapping": cell_mapping,
        "constraint_names_requested": (
            list(requested) if requested is not None else None
        ),
        "constraint_names_sealed": list(sealed) if sealed is not None else None,
        "constraint_names_selected": (
            list(selected) if selected is not None else None
        ),
        "constraint_set_is_subset": is_subset,
        "experiment_dir": experiment_dir,
        "families": families,
        "landscapes": landscapes,
        "lon_landscape": lon_landscape,
        "lon_radii": lon_radii,
        "ils_lon": ils_lon,
        "ils_colon": ils_colon,
        "output_dir": output_dir,
        "output_dir_provided": output_dir_provided,
        "repetition": repetition,
        "rfbx": rfbx,
        "source_type": source_type,
        "taus": taus,
        "write_attractors": write_attractors,
    }


def _overlay_path(
    *,
    cli_value: Path | None,
    toml_value: Any,
    base: Path | None,
    field: str,
) -> Path | None:
    if cli_value is not None:
        return Path(cli_value)
    if toml_value is None:
        return None
    if not isinstance(toml_value, str) or not toml_value.strip():
        raise ElaError(f"{field} must be a non-empty string")
    path = Path(toml_value)
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


def _overlay_landscapes(
    *,
    cli_value: list[str] | None,
    toml_value: Any,
) -> tuple[str, ...] | None:
    if cli_value is not None:
        return tuple(str(name) for name in cli_value)
    if toml_value is None:
        return None
    if not isinstance(toml_value, list):
        raise ElaError("landscapes must be a list of non-empty strings")
    return tuple(str(name) for name in toml_value)


def _overlay_repetition(
    *,
    cli_value: int | None,
    toml_value: Any,
) -> int | None:
    if cli_value is not None:
        if isinstance(cli_value, bool) or cli_value < 0:
            raise ElaError("repetition must be an integer >= 0")
        return int(cli_value)
    if toml_value is None:
        return None
    if not isinstance(toml_value, int) or isinstance(toml_value, bool) or toml_value < 0:
        raise ElaError("repetition must be an integer >= 0")
    return int(toml_value)


def _overlay_taus(
    *,
    cli_value: list[float] | None,
    toml_topology: Mapping[str, Any] | None,
) -> tuple[float, ...] | None:
    if cli_value is not None:
        from core.ioh_core.workflows.la.topology import parse_ela_taus

        return parse_ela_taus(cli_value)
    if isinstance(toml_topology, Mapping) and "taus" in toml_topology:
        return tuple(float(value) for value in toml_topology["taus"])
    return None


def _overlay_rfbx(
    *,
    cli_n_walks: int | None,
    cli_n_transitions: int | None,
    cli_seed_base: int | None,
    cli_start_policy: str | None,
    toml_rfbx: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    table: dict[str, Any] = dict(toml_rfbx) if isinstance(toml_rfbx, Mapping) else {}
    if cli_n_walks is not None:
        table["n_walks"] = cli_n_walks
    if cli_n_transitions is not None:
        table["n_transitions"] = cli_n_transitions
    if cli_seed_base is not None:
        table["seed_base"] = cli_seed_base
    if cli_start_policy is not None:
        table["start_policy"] = cli_start_policy
    if not table:
        return None
    return table


def _overlay_write_attractors(
    *,
    cli_value: bool | None,
    toml_basins: Mapping[str, Any] | None,
) -> bool:
    if cli_value:
        return True
    if isinstance(toml_basins, Mapping) and "write_attractors" in toml_basins:
        value = toml_basins["write_attractors"]
        if not isinstance(value, bool):
            raise ElaError("basins.write_attractors must be a boolean")
        return bool(value)
    return False


def _overlay_lon_radii(
    *,
    cli_value: list[int] | None,
    toml_lon: Mapping[str, Any] | None,
) -> tuple[int, ...] | None:
    from core.ioh_core.workflows.la.orchestrate import coerce_lon_table

    if cli_value is not None:
        return tuple(coerce_lon_table({"radii": cli_value})["radii"])
    if isinstance(toml_lon, Mapping) and "radii" in toml_lon:
        return tuple(int(value) for value in toml_lon["radii"])
    return None


def _overlay_lon_landscape(
    *,
    cli_value: str | None,
    toml_lon: Mapping[str, Any] | None,
) -> str:
    from core.ioh_core.workflows.la.models import LANDSCAPE_Y_P
    from core.ioh_core.workflows.la.orchestrate import coerce_lon_table

    if cli_value is not None:
        return str(coerce_lon_table({"landscape": cli_value})["landscape"])
    if isinstance(toml_lon, Mapping) and "landscape" in toml_lon:
        return str(toml_lon["landscape"])
    return LANDSCAPE_Y_P


def _overlay_ils_lon(
    *,
    cli_kicks: list[int] | None,
    cli_cap: int | None,
    cli_patience: int | None,
    cli_n_starts: int | None,
    cli_seed_base: int | None,
    cli_start_policy: str | None,
    cli_sampling_dir: Path | None,
    cli_repetition: int | None,
    cli_landscape: str | None,
    toml_ils: Mapping[str, Any] | None,
    base: Path | None,
) -> dict[str, Any] | None:
    """Merge CLI and ``[ela.ils_lon]``. ``None`` means the operator omitted knobs."""
    from core.ioh_core.workflows.la.orchestrate import coerce_ils_lon_table

    table: dict[str, Any] = {}
    if isinstance(toml_ils, Mapping):
        table.update(coerce_ils_lon_table(toml_ils, base=base))
    elif toml_ils is not None:
        raise ElaError("ils_lon must be a table")
    if cli_kicks is not None:
        table["kicks"] = [int(value) for value in cli_kicks]
    if cli_cap is not None:
        table["cap"] = int(cli_cap)
    if cli_patience is not None:
        table["patience"] = int(cli_patience)
    if cli_n_starts is not None:
        table["n_starts"] = int(cli_n_starts)
    if cli_seed_base is not None:
        table["seed_base"] = int(cli_seed_base)
    if cli_start_policy is not None:
        table["start_policy"] = str(cli_start_policy)
    if cli_sampling_dir is not None:
        table["sampling_dir"] = Path(cli_sampling_dir)
    if cli_repetition is not None:
        table["repetition"] = int(cli_repetition)
    if cli_landscape is not None:
        table["landscape"] = str(cli_landscape)
    return table or None


def _overlay_ils_colon(
    *,
    cli_kicks: list[int] | None,
    cli_cap: int | None,
    cli_patience: int | None,
    cli_n_starts: int | None,
    cli_seed_base: int | None,
    cli_start_policy: str | None,
    cli_sampling_dir: Path | None,
    cli_repetition: int | None,
    toml_ils: Mapping[str, Any] | None,
    base: Path | None,
) -> dict[str, Any] | None:
    """Merge CLI and ``[la.ils_colon]``. Landscape is always deb."""
    from core.ioh_core.workflows.la.orchestrate import coerce_ils_colon_table

    table: dict[str, Any] = {}
    if isinstance(toml_ils, Mapping):
        table.update(coerce_ils_colon_table(toml_ils, base=base))
    elif toml_ils is not None:
        raise ElaError("ils_colon must be a table")
    if cli_kicks is not None:
        table["kicks"] = [int(value) for value in cli_kicks]
    if cli_cap is not None:
        table["cap"] = int(cli_cap)
    if cli_patience is not None:
        table["patience"] = int(cli_patience)
    if cli_n_starts is not None:
        table["n_starts"] = int(cli_n_starts)
    if cli_seed_base is not None:
        table["seed_base"] = int(cli_seed_base)
    if cli_start_policy is not None:
        table["start_policy"] = str(cli_start_policy)
    if cli_sampling_dir is not None:
        table["sampling_dir"] = Path(cli_sampling_dir)
    if cli_repetition is not None:
        table["repetition"] = int(cli_repetition)
    return table or None


def _overlay_constraint_names(
    *,
    cli_value: list[str] | None,
    toml_value: Any,
) -> tuple[str, ...] | None:
    if cli_value is not None:
        return tuple(str(name) for name in cli_value)
    if toml_value is None:
        return None
    if not isinstance(toml_value, list):
        raise ElaError("constraint_names must be a list of non-empty strings")
    return tuple(str(name) for name in toml_value)


def _load_sealed_constraint_names(experiment_dir: Path) -> tuple[str, ...] | None:
    """
    Read ``constraint_names`` from ``dataset/schema.json`` when present.

    Uses the standard library only (no DuckDB / PyArrow).
    """
    path = Path(experiment_dir) / "dataset" / "schema.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ElaError(f"invalid dataset/schema.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ElaError("dataset/schema.json must be a JSON object")
    names = payload.get("constraint_names")
    if not isinstance(names, list) or not names or not all(
        isinstance(name, str) and name.strip() and name == name.strip()
        for name in names
    ):
        raise ElaError(
            "dataset/schema.json constraint_names must be a non-empty "
            "list of non-blank strings"
        )
    return tuple(str(name) for name in names)


def _load_ela_toml(path: Path) -> dict[str, Any]:
    """
    Load an LA research-knob TOML.

    Accepts a top-level ``[la]`` table, or a flat document of allowed keys.
    Unknown keys fail closed. Does not use ``parse_experiment_config``.
    """
    path = Path(path)
    if not path.is_file():
        raise ElaError(f"ELA config file not found: {path}")
    raw = path.read_bytes()
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ElaError(f"invalid ELA TOML {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ElaError(f"ELA config must be a table, got {type(data).__name__}")

    if _ELA_TABLE_KEY in data:
        extra = sorted(key for key in data if key != _ELA_TABLE_KEY)
        if extra:
            raise ElaError(
                f"unknown top-level ELA config keys: {extra}; "
                f"put research knobs under [{_ELA_TABLE_KEY}]"
            )
        table = data[_ELA_TABLE_KEY]
        if not isinstance(table, dict):
            raise ElaError("[la] must be a table")
    else:
        table = data

    unknown = sorted(key for key in table if key not in _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ElaError(f"unknown ELA config keys: {unknown}")
    return _coerce_ela_table(table)


def _coerce_ela_table(table: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "schema_version" in table:
        version = table["schema_version"]
        if not isinstance(version, int) or isinstance(version, bool) or version != 1:
            raise ElaError("la.schema_version must be 1")
        out["schema_version"] = version
    for key in ("experiment_dir", "output_dir", "source_type"):
        if key not in table:
            continue
        value = table[key]
        if not isinstance(value, str) or not value.strip():
            raise ElaError(f"{key} must be a non-empty string")
        out[key] = value
    if "repetition" in table:
        value = table["repetition"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ElaError("repetition must be an integer >= 0")
        out["repetition"] = value
    for key in _STRING_LIST_KEYS:
        if key not in table:
            continue
        value = table[key]
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ElaError(f"{key} must be a list of non-empty strings")
        out[key] = list(value)
    if "cell_mapping" in table:
        from core.ioh_core.workflows.la.orchestrate import coerce_cell_mapping_table

        out["cell_mapping"] = coerce_cell_mapping_table(table["cell_mapping"])
    if "topology" in table:
        from core.ioh_core.workflows.la.orchestrate import coerce_topology_table

        out["topology"] = coerce_topology_table(table["topology"])
    if "rfbx" in table:
        from core.ioh_core.workflows.la.orchestrate import coerce_rfbx_table

        out["rfbx"] = coerce_rfbx_table(table["rfbx"])
    if "basins" in table:
        from core.ioh_core.workflows.la.orchestrate import coerce_basins_table

        out["basins"] = coerce_basins_table(table["basins"])
    if "lon" in table:
        from core.ioh_core.workflows.la.orchestrate import coerce_lon_table

        out["lon"] = coerce_lon_table(table["lon"])
    if "ils_lon" in table:
        from core.ioh_core.workflows.la.orchestrate import coerce_ils_lon_table

        out["ils_lon"] = coerce_ils_lon_table(table["ils_lon"], base=None)
    if "ils_colon" in table:
        from core.ioh_core.workflows.la.orchestrate import coerce_ils_colon_table

        out["ils_colon"] = coerce_ils_colon_table(table["ils_colon"], base=None)
    return out


def _emit_json(payload: Mapping[str, Any], *, stream=None) -> None:
    target = sys.stderr if stream is None else stream
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), file=target)


if __name__ == "__main__":
    raise SystemExit(main())
