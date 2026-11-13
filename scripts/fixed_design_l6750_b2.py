"""Evaluate one fixed design on L6750_B2 and print the penalised objective."""
from __future__ import annotations

from core.ioh_core.problem_builder import build_problems_for_slab_type


def main() -> None:
    bundle = build_problems_for_slab_type("hp_slab")["L6750_B2"]
    design = list(bundle["lb_idx"])
    record = bundle["ctx"].get(design)
    print(f"design={design}")
    print(f"y_p={record['y_p']}")
    print(f"y={record['y']}")


if __name__ == "__main__":
    main()
