"""Paths for one problem's predesign shelf and ranked InitialDesign."""
from __future__ import annotations

from pathlib import Path

PREDESIGN_DIRNAME = "predesign"


def predesign_results_path(output_root: str | Path, problem_id: str) -> Path:
    """Return ``{output_root}/predesign/{problem_id}/results.csv``."""
    return (
        Path(output_root)
        / PREDESIGN_DIRNAME
        / str(problem_id)
        / "results.csv"
    )


def predesign_initial_design_path(output_root: str | Path, problem_id: str) -> Path:
    """Return ``{output_root}/predesign/{problem_id}/initial_design.json``."""
    return (
        Path(output_root)
        / PREDESIGN_DIRNAME
        / str(problem_id)
        / "initial_design.json"
    )


def predesign_initial_design_deb_path(
    output_root: str | Path,
    problem_id: str,
) -> Path:
    """Return ``{output_root}/predesign/{problem_id}/initial_design_deb.json``."""
    return (
        Path(output_root)
        / PREDESIGN_DIRNAME
        / str(problem_id)
        / "initial_design_deb.json"
    )
