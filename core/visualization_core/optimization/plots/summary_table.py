"""Per-algorithm fixed-target summary table (CSV + Markdown)."""
from __future__ import annotations

import csv
from pathlib import Path

from core.visualization_core.optimization.aggregate import (
    AlgorithmTargetSummary,
    algorithm_target_summaries,
)
from core.visualization_core.optimization.models import ExperimentBundle, PlotConfig

_COLUMNS: tuple[str, ...] = (
    "algorithm",
    "percent",
    "n_runs",
    "n_success",
    "success_rate",
    "e2t_min",
    "e2t_max",
    "e2t_mean",
    "e2t_median",
    "e2t_std",
    "ert",
    "budget",
    "n_runs_with_feasible",
    "mean_first_feasible_fe",
    "mean_share_feasible",
    "mean_wall_time_s",
)


def _fmt(value: float | None, *, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (value != value):  # NaN
        return ""
    return f"{value:.{digits}g}"


def _row_dict(row: AlgorithmTargetSummary) -> dict[str, str]:
    return {
        "algorithm": row.algorithm,
        "percent": _fmt(row.percent, digits=4),
        "n_runs": str(int(row.n_runs)),
        "n_success": str(int(row.n_success)),
        "success_rate": _fmt(row.success_rate, digits=6),
        "e2t_min": _fmt(row.e2t_min, digits=8),
        "e2t_max": _fmt(row.e2t_max, digits=8),
        "e2t_mean": _fmt(row.e2t_mean, digits=8),
        "e2t_median": _fmt(row.e2t_median, digits=8),
        "e2t_std": _fmt(row.e2t_std, digits=8),
        "ert": _fmt(row.ert, digits=8),
        "budget": str(int(row.budget)),
        "n_runs_with_feasible": (
            ""
            if row.n_runs_with_feasible is None
            else str(int(row.n_runs_with_feasible))
        ),
        "mean_first_feasible_fe": _fmt(row.mean_first_feasible_fe, digits=8),
        "mean_share_feasible": _fmt(row.mean_share_feasible, digits=6),
        "mean_wall_time_s": _fmt(row.mean_wall_time_s, digits=8),
    }


def _write_csv(path: Path, rows: tuple[AlgorithmTargetSummary, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_dict(row))
    return path


def _write_markdown(path: Path, rows: tuple[AlgorithmTargetSummary, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fixed-target algorithm summary",
        "",
        "Hitting times (`e2t_*`) are function evaluations (FE) among "
        "**successful** runs only. `ert` is Hansen ERT over **all** runs "
        "(misses contribute budget `B`). `n_runs_with_feasible` counts runs "
        "with ≥1 feasible evaluation. `mean_first_feasible_fe` is the mean "
        "evaluation index of the first feasible sample (those runs only). "
        "`mean_share_feasible` is the mean over runs of the share of "
        "feasible evaluations. `mean_wall_time_s` is mean wall-clock "
        "seconds over runs with a recorded wall time. Empty numeric cells "
        "mean no successes / no feasible samples / missing masks or times.",
        "",
        "| " + " | ".join(_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _COLUMNS) + " |",
    ]
    for row in rows:
        data = _row_dict(row)
        lines.append("| " + " | ".join(data[c] for c in _COLUMNS) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def render_summary_table(
    bundle: ExperimentBundle,
    config: PlotConfig,
    out_dir: Path,
    *,
    stem: str = "summary_by_target",
) -> list[Path]:
    """
    Write one CSV (+ Markdown twin) covering all ``config.target_percents``.

    Parameters
    ----------
    bundle : ExperimentBundle
        Loaded experiments.
    config : PlotConfig
        Target percents / optimum / reference.
    out_dir : Path
        Output directory.
    stem : str
        Filename stem (``.csv`` / ``.md``).

    Returns
    -------
    list of Path
        Written ``.csv`` and ``.md`` paths.
    """
    rows = algorithm_target_summaries(
        bundle,
        config.target_percents,
        global_optimum=config.global_optimum,
        target_reference=config.target_reference,
    )
    out = Path(out_dir)
    csv_path = _write_csv(out / f"{stem}.csv", rows)
    md_path = _write_markdown(out / f"{stem}.md", rows)
    return [csv_path, md_path]
