"""Shape check for a fiber–Marin table that is already on disk.

Does not run the comparison. Skips when ``summary.json`` is absent.
"""
from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from _mains.verification.plot_mk_height_reinf_matrix import (
    MK_POSITIONS,
    _position_label,
    iter_cases,
)

_OUT = (
    Path(__file__).resolve().parents[1]
    / "verification"
    / "outputs"
    / "fiber_marin"
)


class TestFiberMarinSummary(unittest.TestCase):
    """72 section rows and 36 deflection rows match the imported matrix."""

    def test_saved_table_matches_matrix(self) -> None:
        summary_path = _OUT / "summary.json"
        if not summary_path.is_file():
            self.skipTest("fiber–Marin summary.json is not on disk")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["n_sections"], 72)
        self.assertEqual(summary["n_deflections"], 36)
        self.assertEqual(summary["mesh_size"], 0.01)
        self.assertEqual(summary["load_category"], "B2")
        timing = summary["timing"]
        self.assertEqual(timing["n_designs"], 36)
        self.assertEqual(timing["threads_per_library"], 1)
        for stage in ("uls_s", "mk_s", "defl_s", "total_s"):
            self.assertGreater(timing["sum_s"][stage]["fiber"], 0.0)
            self.assertGreater(timing["sum_s"][stage]["marin"], 0.0)
            self.assertGreater(timing["ratio_marin_over_fiber"][stage], 0.0)

        allowed = {
            (case.h_ges_mm, case.a_tex_mm2, case.nt, case.kap_percent)
            for case in iter_cases()
        }
        positions = {_position_label(x) for x in MK_POSITIONS}

        with (_OUT / "sections.csv").open(encoding="utf-8", newline="") as handle:
            sections = list(csv.DictReader(handle))
        with (_OUT / "deflections.csv").open(encoding="utf-8", newline="") as handle:
            deflections = list(csv.DictReader(handle))

        self.assertEqual(len(sections), 72)
        self.assertEqual(len(deflections), 36)
        for row in sections:
            key = (
                float(row["h_ges_mm"]),
                float(row["a_tex_mm2"]),
                int(row["nt"]),
                float(row["kap_percent"]),
            )
            self.assertIn(key, allowed)
            self.assertIn(row["position"], positions)
        for row in deflections:
            key = (
                float(row["h_ges_mm"]),
                float(row["a_tex_mm2"]),
                int(row["nt"]),
                float(row["kap_percent"]),
            )
            self.assertIn(key, allowed)
