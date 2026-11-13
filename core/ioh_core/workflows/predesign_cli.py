"""Write one HP-slab predesign run and its ranked InitialDesign."""
from __future__ import annotations

import argparse

from slab_construction.slabs.hp_slab.predesign import write_predesign_run


def main(argv: list[str] | None = None) -> int:
    """
    Resolve ``PROBLEM_ID`` and write its predesign run under ``OUTPUT_ROOT``.

    Span and load come from the problem definition. This command does not
    take them as extra flags.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the HP-slab predesign grid for one problem and write "
            "results.csv together with initial_design.json."
        )
    )
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--structural-library",
        default=None,
        help=(
            "Sealed structural library. Omit to build each section during "
            "analysis. A missing artifact or key mismatch writes nothing."
        ),
    )
    args = parser.parse_args(argv)
    write_predesign_run(
        args.problem_id,
        args.output_root,
        library_path=args.structural_library,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
