"""
Success-rate heatmap across the problems present in loaded experiments.

The page has two views that share the problem axis and the ε_X menu.
Success rate keeps every variant, cold start above its warm start.
Warm-start improvement keeps one row per paired family and shows the
difference in percentage points.
"""
from __future__ import annotations

import html as html_lib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from core.visualization_core.optimization.aggregate import success_probability_curve
from core.visualization_core.optimization.algorithm_presentation import (
    CLASS_ORDER,
    AlgorithmPresentation,
    AlgorithmPresentationError,
    constraint_label,
    local_search_label,
    presentation_for,
    qualifiers_needed,
    tooltip_subtitle,
)
from core.visualization_core.optimization.models import ExperimentBundle, RunTrajectory

MARGINS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 10.0)
_GREY = "#b2b2b2"
_MISSING = "—"
_SUCCESS_TITLE = "Algorithm success rates across optimisation problems"
_IMPROVEMENT_TITLE = "Warm-start improvement across optimisation problems"
_FONT = 13
_SPAN_FONT = 11
_TITLE_SIZE = 18
_AXIS_TITLE_SIZE = 16
_FONT_FAMILY = "DejaVu Sans"
_FONT_STACK = "'DejaVu Sans', Arial, Helvetica, sans-serif"
_PAD_X = 8
_PAD_Y = 6
_MARGIN_TOP = 54
_MARGIN_BOTTOM = 54
_RIGHT_PAD = 108
_OVERALL_GAP = 4
_OVERALL_W = 128
_OVERALL_INSET = 4
_OVERALL_TAIL = 10
_ROW_GAP = 0.25
_LABEL_H = _FONT + 4
_SPAN_BG = "#F2F2F2"
_SPAN_GAP = 2
_MIN_SPAN_LOADS = 3
_Y_LABEL_GAP = 8
_Y_LABEL_PAD = 8
_RDYLGN_SCALE = [
    [0.0, "#a50026"],
    [0.1, "#d73027"],
    [0.2, "#f46d43"],
    [0.3, "#fdae61"],
    [0.4, "#fee08b"],
    [0.5, "#ffffbf"],
    [0.6, "#d9ef8b"],
    [0.7, "#a6d96a"],
    [0.8, "#66bd63"],
    [0.9, "#1a9850"],
    [1.0, "#006837"],
]
_RWB_SCALE = [
    [0.0, "#67001f"],
    [0.1, "#b2182b"],
    [0.2, "#d6604d"],
    [0.3, "#f4a582"],
    [0.4, "#fddbc7"],
    [0.5, "#ffffff"],
    [0.6, "#d1e5f0"],
    [0.7, "#92c5de"],
    [0.8, "#4393c3"],
    [0.9, "#2166ac"],
    [1.0, "#053061"],
]
_RDYLGN = tuple((stop, color[1:]) for stop, color in _RDYLGN_SCALE)
_RWB = tuple((stop, color[1:]) for stop, color in _RWB_SCALE)
_PROBLEM_ID = re.compile(r"^L(\d+)_(A1|B1|B2)$")
_LOAD_RANK = {"A1": 0, "B1": 1, "B2": 2}
_LOAD_TEXT = {"A1": "1 kN/m²", "B1": "2 kN/m²", "B2": "3 kN/m²"}
_PACK = (
    "kind",
    "display_name",
    "subtitle",
    "problem",
    "detail",
    "epsilon",
    "line1",
    "line2",
    "line3",
    "note",
    "color",
)


class _Family:
    def __init__(self, family_id: str) -> None:
        self.family_id = family_id
        self.cold: str | None = None
        self.warm: str | None = None

    def tokens(self) -> list[str]:
        ordered: list[str] = []
        if self.cold is not None:
            ordered.append(self.cold)
        if self.warm is not None:
            ordered.append(self.warm)
        return ordered


def _problem_key(problem_id: str) -> tuple:
    match = _PROBLEM_ID.fullmatch(problem_id)
    if match is None:
        return (1, problem_id)
    return (0, int(match.group(1)), _LOAD_RANK[match.group(2)])


def _problem_order(problem_ids: Sequence[str]) -> list[str]:
    return sorted(set(problem_ids), key=_problem_key)


def _group_key(problem_id: str) -> tuple:
    match = _PROBLEM_ID.fullmatch(problem_id)
    if match is None:
        return ("custom",)
    return ("span", int(match.group(1)))


def _column_gaps(problems: Sequence[str]) -> list[int]:
    gaps: list[int] = []
    previous = None
    for index, problem_id in enumerate(problems):
        key = _group_key(problem_id)
        if previous is not None and key != previous:
            gaps.append(index - 1)
        previous = key
    return gaps


def _span_label(span_mm: int) -> str:
    return f"{span_mm / 1000:.3f} m"


def _column_groups(problems: Sequence[str]) -> list[dict]:
    groups: list[dict] = []
    for index, problem_id in enumerate(problems):
        match = _PROBLEM_ID.fullmatch(problem_id)
        span = int(match.group(1)) if match is not None else None
        load = match.group(2) if match is not None else problem_id
        if not groups or groups[-1]["span"] != span or span is None:
            groups.append({"span": span, "start": index, "end": index, "load": load})
        else:
            groups[-1]["end"] = index
    return groups


def _span_slots(n_loads: int) -> int:
    """Column pitches occupied by one span header.

    A span with three imposed load levels keeps that width. A shorter span
    is padded out to the same width, and a longer span grows with its loads.
    """
    return max(int(n_loads), _MIN_SPAN_LOADS)


def _span_field_width(n_loads: int, cell_w: float) -> float:
    """Pixel width of the grey span label, including the three-load minimum."""
    return _span_slots(n_loads) * float(cell_w) - _SPAN_GAP - 2


def _span_axis(problems: Sequence[str]) -> dict:
    """Category axis with each span field at least three load columns wide.

    Short spans are centred in the extra columns. Those columns are named
    ``__xpad_`` and carry no heatmap cell. A span with more than three loads
    uses one column per load and no padding.
    """
    categories: list[str] = []
    problem_x: list[int] = []
    placed: list[dict] = []
    pad = 0
    for group in _column_groups(problems):
        member_at = range(group["start"], group["end"] + 1)
        n_loads = len(member_at)
        slots = n_loads if group["span"] is None else _span_slots(n_loads)
        extra = slots - n_loads
        left = extra // 2
        for _ in range(left):
            categories.append(f"__xpad_{pad}")
            pad += 1
        start = len(categories)
        for member in member_at:
            problem_x.append(len(categories))
            categories.append(problems[member])
        for _ in range(extra - left):
            categories.append(f"__xpad_{pad}")
            pad += 1
        placed.append(
            {
                "span": group["span"],
                "start": start,
                "end": len(categories) - 1,
                "slots": slots,
                "loads": n_loads,
            }
        )
    return {
        "categories": categories,
        "problem_x": problem_x,
        "groups": placed,
    }


def _problem_ticks(problems: Sequence[str]) -> list[str]:
    """Load case centred in every column. The span is drawn above the group."""
    ticks: list[str] = []
    for problem_id in problems:
        match = _PROBLEM_ID.fullmatch(problem_id)
        load = match.group(2) if match is not None else problem_id
        ticks.append(f"<br>{load}")
    return ticks


def _problem_detail(problem_id: str) -> str:
    match = _PROBLEM_ID.fullmatch(problem_id)
    if match is None:
        return ""
    metres = _span_label(int(match.group(1)))
    return f"{metres} · Live load {_LOAD_TEXT[match.group(2)]}"


def _hex_channel(value: str) -> tuple[int, int, int]:
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
    )


def _swatch(unit: float, scale: tuple = _RDYLGN) -> str:
    unit = min(1.0, max(0.0, float(unit)))
    for (left, left_hex), (right, right_hex) in zip(scale, scale[1:]):
        if unit <= right:
            span = right - left
            blend = 0.0 if span == 0 else (unit - left) / span
            start = _hex_channel(left_hex)
            end = _hex_channel(right_hex)
            mixed = [
                round(channel + (stop - channel) * blend)
                for channel, stop in zip(start, end)
            ]
            return "#" + "".join(f"{part:02x}" for part in mixed)
    return "#" + scale[-1][1]


def _ink(color: str) -> str:
    red, green, blue = _hex_channel(color[1:])
    luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0
    return "#222222" if luminance > 0.62 else "#ffffff"


def _rate_text(rate: float) -> str:
    return f"{100.0 * float(rate):.1f}%"


def _format_pp(value: float) -> str:
    rounded = round(float(value), 1)
    if rounded == 0:
        return "0.0 pp"
    if rounded > 0:
        return f"+{rounded:.1f} pp"
    return f"\u2212{abs(rounded):.1f} pp"


def _epsilon_label(percent: float) -> str:
    return f"relative target tolerance ε<sub>X</sub> = {percent:g}%"


def _mean_note(count: int, kind: str) -> str:
    if kind == "algorithm":
        noun = "algorithm success rate" if count == 1 else "algorithm success rates"
    else:
        noun = "family improvement" if count == 1 else "family improvements"
    return f"Mean of {count} {noun}"


def _budget_text(grouped: dict[tuple[str, str], list[RunTrajectory]]) -> str:
    """One budget when every loaded run shares it, otherwise the span of budgets."""
    values = sorted(
        {
            int(run.max_evaluations)
            for runs in grouped.values()
            for run in runs
        }
    )
    if len(values) == 1:
        return f"{values[0]:,} FE"
    if len(values) > 1:
        return f"{values[0]:,}–{values[-1]:,} FE"
    return "the budget recorded for each run"


def _description(budget_text: str, algorithm_names: object = ()) -> str:
    """Definitions under the heatmap. Shared by both measures."""
    from core.visualization_core.optimization.plots.interactive_pool import (
        _PREDESIGN_SENTENCE,
        _PUBLICATION,
        _labels_for_names,
        _local_search_item,
        _rank_sentence,
        _target_note,
    )

    return (
        '<p style="margin:0 0 6px; font-weight:700; color:#222;">Definitions</p>'
        '<p style="margin:0 0 8px;">The interactive heatmap compares optimisation '
        "algorithms across problem instances at the selected relative target tolerance "
        "ε<sub>X</sub>. It shows success rates and the improvement achieved by "
        "warm-start algorithm variants compared to cold-start algorithm variants.</p>"
        '<ul style="margin:0 0 8px; padding-left:18px;">'
        "<li><b>Success rate.</b> The proportion of completed runs that reach the "
        "selected relative target tolerance within the evaluation budget:</li>"
        "</ul>"
        '<p class="heatmap-formula">p<sub>s</sub> = '
        '<span class="frac"><span class="num">N<sub>ε<sub>X</sub></sub></span>'
        '<span class="den">R<sub>s</sub></span></span>.</p>'
        '<p class="formula-note">Here, N<sub>ε<sub>X</sub></sub> is the number of successful runs '
        "at that target and R<sub>s</sub> is the total number of completed runs. Success rates are "
        "displayed as percentages.</p>"
        '<ul style="margin:0 0 8px; padding-left:18px;">'
        "<li><b>Warm-start improvement.</b> The difference between the warm-start and "
        "cold-start success rates, expressed in percentage points:</li>"
        "</ul>"
        '<p class="heatmap-formula">Δp<sub>s</sub> = 100 (p<sub>s,warm</sub> − '
        "p<sub>s,cold</sub>).</p>"
        '<p class="formula-note">Positive values indicate an improvement from warm '
        "starting; negative values indicate a deterioration.</p>"
        '<ul style="margin:0 0 8px; padding-left:18px;">'
        "<li><b>Problems.</b> Columns are the problem instances in this figure, "
        "grouped by span, in metres. The instances combine different spans and live "
        "load classes A1, B1, and B2, representing 1, 2, and 3 kN/m².</li>"
        "<li><b>Algorithms.</b> In the success-rate view, each algorithm family has "
        "a cold-start row above its warm-start row. In the warm-start improvement "
        "view, each family has one row showing the difference between the two "
        f"variants.</li>{_local_search_item(_labels_for_names(algorithm_names))}"
        "<li><b>Summary rows.</b> The All algorithms row shows the mean success "
        "rate across all algorithm variants for each problem instance. The All "
        "families row shows the mean warm-start improvement across algorithm "
        "families, in percentage points.</li>"
        "<li><b>Overall.</b> The last column pools completed runs across all problem "
        "instances, reporting the success rate and the number of successful runs. "
        "The overall warm-start improvement is the difference between the pooled "
        "warm-start and cold-start success rates.</li>"
        "<li><b>Evaluation budget.</b> The maximum number of function evaluations "
        f"available to a run to reach the target. These results use {budget_text}. "
        f"{_PREDESIGN_SENTENCE} A cell is a success rate, not an ERT. "
        "ERT<sub>warm</sub> adds 2,700 predesign evaluations once to the "
        "warm-start result. That addition is not applied to these success "
        "rates.</li>"
        "<li><b>Relative target tolerance.</b> A run reaches the target when its "
        "best penalised objective satisfies</li>"
        "</ul>"
        '<p class="heatmap-formula">y<sub>p,best</sub> ≤ (1 + ε<sub>X</sub>) '
        "y<sub>p</sub>*,</p>"
        f'<p class="formula-note">{_target_note()}</p>'
        '<ul style="margin:0; padding-left:18px;">'
        "<li><b>Successful run.</b> A completed run that first reaches the relative "
        "target tolerance at or before the selected evaluation budget.</li>"
        f"<li><b>Ordering.</b> {_rank_sentence()} The order follows the "
        "selected relative target tolerance and stays the same when the "
        "displayed measure changes.</li>"
        "</ul>"
        f"{_PUBLICATION}"
    )


def _start_phrase(token: str) -> str:
    item = presentation_for(token)
    start = "warm start" if item.start == "warm" else "cold start"
    search = local_search_label(item)
    if search:
        return f"{start}, {search}"
    return start


def _visible_extra(
    item: AlgorithmPresentation,
    peers: Sequence[AlgorithmPresentation],
) -> str:
    """Constraint label, plus a local-search name when two descents collide."""
    others = [peer for peer in peers if peer.token != item.token]
    constraint_conflict = any(
        peer.display_name == item.display_name
        and peer.constraint_aware != item.constraint_aware
        for peer in others
    )
    qualifier_needed = item.token in qualifiers_needed(
        [peer.token for peer in peers]
    )
    search = local_search_label(item)
    if search:
        if qualifier_needed and constraint_conflict:
            return f"{constraint_label(item)}, {search}"
        if qualifier_needed:
            return search
        return constraint_label(item)
    if qualifier_needed and item.qualifier and constraint_conflict:
        return f"{constraint_label(item)} · {item.qualifier}"
    if qualifier_needed and item.qualifier:
        return item.qualifier
    if constraint_conflict:
        return constraint_label(item)
    if item.qualifier:
        return item.qualifier
    return constraint_label(item)


def _axis_extra(
    item: AlgorithmPresentation,
    peers: Sequence[AlgorithmPresentation],
) -> str:
    """Constraint or non-descent qualifier. The local search is in the start."""
    extra = _visible_extra(item, peers)
    search = local_search_label(item)
    if not search:
        return extra
    if extra == search:
        return ""
    suffix = f", {search}"
    if extra.endswith(suffix):
        return extra[: -len(suffix)]
    return extra


def _axis_label(token: str, peers: Sequence[str]) -> str:
    """Readable row name, with an extra qualifier only when the label collides."""
    try:
        item = presentation_for(token)
    except AlgorithmPresentationError:
        return token
    label = f"{item.display_name} ({_start_phrase(token)})"
    twins = [other for other in peers if _same_axis_label(token, other)]
    if len(twins) <= 1:
        return label
    presented = [presentation_for(other) for other in twins]
    extra = _axis_extra(item, presented)
    if not extra:
        return label
    return f"{label} · {extra}"


def _same_axis_label(token: str, other: str) -> bool:
    try:
        peer = presentation_for(other)
        item = presentation_for(token)
    except AlgorithmPresentationError:
        return token == other
    return peer.display_name == item.display_name and peer.start == item.start


def _has_presentation(token: str) -> bool:
    try:
        presentation_for(token)
    except AlgorithmPresentationError:
        return False
    return True


def _groups(
    bundle: ExperimentBundle,
) -> dict[tuple[str, str], list[RunTrajectory]]:
    grouped: dict[tuple[str, str], list[RunTrajectory]] = defaultdict(list)
    for run in bundle.runs:
        grouped[(run.algorithm_name, run.problem_id)].append(run)
    return grouped


def _success(runs: Sequence[RunTrajectory], percent: float) -> float:
    return float(success_probability_curve(runs, percent).success_rate)


def _fraction(runs: Sequence[RunTrajectory], percent: float) -> tuple[float, str]:
    curve = success_probability_curve(runs, percent)
    return float(curve.success_rate), f"{curve.n_success}/{curve.n_runs}"


def _counts(
    runs: Sequence[RunTrajectory] | None,
    percent: float,
) -> tuple[int, int] | None:
    if not runs:
        return None
    curve = success_probability_curve(runs, percent)
    return int(curve.n_success), int(curve.n_runs)


def _present_families(names: Sequence[str]) -> list[_Family]:
    found: dict[str, _Family] = {}
    order: list[str] = []
    for token in names:
        try:
            item = presentation_for(token)
            family_id = item.family
            start = item.start
        except AlgorithmPresentationError:
            family_id = token
            start = "cold"
        if family_id not in found:
            found[family_id] = _Family(family_id)
            order.append(family_id)
        family = found[family_id]
        if start == "warm":
            family.warm = token
        else:
            family.cold = token
    return [found[family_id] for family_id in order]


def _colliding_names(family_ids: Sequence[str]) -> set[str]:
    counts: dict[str, int] = defaultdict(int)
    for family_id in family_ids:
        try:
            name = presentation_for(family_id).display_name
        except AlgorithmPresentationError:
            name = family_id
        counts[name] += 1
    return {name for name, count in counts.items() if count > 1}


def _family_label(
    family_id: str,
    colliding: set[str],
    peers: Sequence[AlgorithmPresentation],
) -> tuple[str, str]:
    try:
        item = presentation_for(family_id)
    except AlgorithmPresentationError:
        return family_id, ""
    extra = _visible_extra(item, peers)
    if item.display_name in colliding:
        return f"{item.display_name} · {extra}", ""
    return item.display_name, extra


def _variant_rank(
    grouped: dict[tuple[str, str], list[RunTrajectory]],
    problems: Sequence[str],
    token: str | None,
    percent: float,
) -> tuple[float, float] | None:
    """Pooled success share at ``percent``, then the evaluation where it is reached."""
    if token is None:
        return None
    runs: list[RunTrajectory] = []
    for problem_id in problems:
        runs.extend(grouped.get((token, problem_id), ()))
    if not runs:
        return None
    from core.visualization_core.optimization.plots.interactive_pool import (
        _success_rank,
    )

    return _success_rank(tuple(runs), float(percent))


def _families_in_rank_order(
    families: Sequence[_Family],
    grouped: dict[tuple[str, str], list[RunTrajectory]],
    problems: Sequence[str],
    percent: float,
) -> list[_Family]:
    """Order families by the better variant at one relative target tolerance."""

    def sort_key(family: _Family) -> tuple:
        from core.visualization_core.optimization.plots.interactive_pool import (
            better_family_sort_key,
        )

        ranks = [
            rank
            for rank in (
                _variant_rank(grouped, problems, family.cold, percent),
                _variant_rank(grouped, problems, family.warm, percent),
            )
            if rank is not None
        ]
        return better_family_sort_key(ranks, family.family_id)

    return sorted(families, key=sort_key)


def _ordered_families(
    grouped: dict[tuple[str, str], list[RunTrajectory]],
    problems: Sequence[str],
) -> tuple[list[_Family], dict[str, tuple[str, str]]]:
    """Family labels, with the stored table ordered at ε_X = 1%."""
    names: list[str] = []
    for algorithm_name, _problem_id in grouped:
        if algorithm_name not in names:
            names.append(algorithm_name)
    families = _present_families(names)
    colliding = _colliding_names([family.family_id for family in families])
    peers: list[AlgorithmPresentation] = []
    for family in families:
        try:
            peers.append(presentation_for(family.family_id))
        except AlgorithmPresentationError:
            continue
    labels = {
        family.family_id: _family_label(family.family_id, colliding, peers)
        for family in families
    }
    return _families_in_rank_order(families, grouped, problems, 1.0), labels


def _family_gap_indexes(families: Sequence[_Family]) -> list[int]:
    if len(families) < 2:
        return []
    gaps: list[int] = []
    cursor = -1
    for family in families[:-1]:
        cursor += len(family.tokens())
        gaps.append(cursor)
    return gaps


def _has_pair(
    family: _Family,
    grouped: dict[tuple[str, str], list[RunTrajectory]],
    problems: Sequence[str],
) -> bool:
    if family.cold is None or family.warm is None:
        return False
    return any(
        (family.cold, problem_id) in grouped
        and (family.warm, problem_id) in grouped
        for problem_id in problems
    )


def _blank_cell(problem_id: str, percent: float, **overrides: object) -> dict:
    cell: dict = {
        "kind": "missing",
        "display_name": "",
        "subtitle": "",
        "problem": problem_id,
        "detail": _problem_detail(problem_id),
        "epsilon": _epsilon_label(percent),
        "fraction": _MISSING,
        "rate": None,
        "line1": "No completed run",
        "line2": "",
        "line3": "",
        "note": "",
        "color": _GREY,
        "cold_fraction": "",
        "warm_fraction": "",
        "difference": "",
    }
    cell.update(overrides)
    return cell


def _rate_cell(
    token: str,
    problem_id: str,
    runs: list[RunTrajectory] | None,
    percent: float,
    *,
    show_qualifier: bool,
) -> tuple[float | None, str, dict]:
    try:
        item = presentation_for(token)
        display_name = item.display_name
        subtitle = tooltip_subtitle(item, show_qualifier=show_qualifier)
    except AlgorithmPresentationError:
        display_name = token
        subtitle = ""
    if not runs:
        cell = _blank_cell(
            problem_id,
            percent,
            display_name=display_name,
            subtitle=subtitle,
        )
        return None, _MISSING, cell
    rate, fraction = _fraction(runs, percent)
    return (
        rate,
        fraction,
        _blank_cell(
            problem_id,
            percent,
            kind="rate",
            display_name=display_name,
            subtitle=subtitle,
            fraction=fraction,
            rate=_rate_text(rate),
            line1=f"{_rate_text(rate)} ({fraction} runs)",
            color=_swatch(rate),
        ),
    )


def _summary_cell(
    rates: Sequence[float],
    problem_id: str,
    percent: float,
    *,
    label: str,
    kind_name: str,
) -> tuple[float | None, dict]:
    if not rates:
        return None, _blank_cell(problem_id, percent, display_name=label)
    mean = sum(rates) / float(len(rates))
    note = _mean_note(len(rates), kind_name)
    if kind_name == "algorithm":
        line = _rate_text(mean)
        color = _swatch(mean)
        difference = ""
    else:
        line = _format_pp(mean)
        color = _swatch((mean + 100.0) / 200.0, _RWB)
        difference = line
    return mean, _blank_cell(
        problem_id,
        percent,
        kind="summary",
        display_name=label,
        subtitle=note,
        rate=_rate_text(mean) if kind_name == "algorithm" else None,
        line1=line,
        difference=difference,
        color=color,
    )


def _side_line(
    title: str,
    missing: str,
    runs: list[RunTrajectory] | None,
    percent: float,
) -> tuple[str, str, float | None]:
    if not runs:
        return missing, _MISSING, None
    rate, fraction = _fraction(runs, percent)
    return f"{title} {_rate_text(rate)} ({fraction} runs)", fraction, rate


def _delta_cell(
    family: _Family,
    label: str,
    subtitle: str,
    problem_id: str,
    grouped: dict[tuple[str, str], list[RunTrajectory]],
    percent: float,
) -> tuple[float | None, dict]:
    cold_runs = grouped.get((family.cold or "", problem_id))
    warm_runs = grouped.get((family.warm or "", problem_id))
    cold_line, cold_fraction, cold_rate = _side_line(
        "Cold start",
        "No completed cold-start run",
        cold_runs,
        percent,
    )
    warm_line, warm_fraction, warm_rate = _side_line(
        "Warm start",
        "No completed warm-start run",
        warm_runs,
        percent,
    )
    if cold_rate is None or warm_rate is None:
        return None, _blank_cell(
            problem_id,
            percent,
            display_name=label,
            subtitle=subtitle,
            line1=cold_line,
            line2=warm_line,
            cold_fraction=cold_fraction,
            warm_fraction=warm_fraction,
        )
    delta = 100.0 * (warm_rate - cold_rate)
    return delta, _blank_cell(
        problem_id,
        percent,
        kind="delta",
        display_name=label,
        subtitle=subtitle,
        line1=cold_line,
        line2=warm_line,
        line3=_format_pp(delta),
        difference=_format_pp(delta),
        cold_fraction=cold_fraction,
        warm_fraction=warm_fraction,
        color=_swatch((delta + 100.0) / 200.0, _RWB),
    )


def _pack(cell: dict | None) -> list | None:
    if cell is None:
        return None
    return [cell[key] for key in _PACK]


def _insert_after(items: list, gap_after: Sequence[int], blank) -> list:
    gaps = set(gap_after)
    out: list = []
    made = 0
    for index, item in enumerate(items):
        out.append(item)
        if index in gaps:
            out.append(blank(made) if callable(blank) else blank)
            made += 1
    return out


def _place_on_axis(row: Sequence, indexes: Sequence[int], n_cols: int) -> list:
    """Copy ``row`` onto the span axis. Padded span columns stay empty."""
    placed: list = [None] * n_cols
    for value, index in zip(row, indexes):
        placed[index] = value
    return placed


def _stack(
    summary: list,
    body: list[list],
    row_gaps: Sequence[int],
    column_index: Sequence[int],
    n_cols: int,
) -> list[list]:
    summary_row = _place_on_axis(summary, column_index, n_cols)
    width = n_cols
    expanded = [_place_on_axis(row, column_index, n_cols) for row in body]
    gapped = _insert_after(
        expanded,
        row_gaps,
        lambda _index: [None] * width,
    )
    return [summary_row, [None] * width, *gapped]


def _y_labels(summary: str, labels: Sequence[str], row_gaps: Sequence[int]) -> list[str]:
    body = _insert_after(list(labels), row_gaps, lambda index: f"__ygap_{index}")
    return [summary, "__ygap_summary", *body]


def _row_geometry(y_labels: Sequence[str]) -> tuple[list[float], list[int], float]:
    positions: list[float] = []
    indexes: list[int] = []
    cursor = 0.0
    for index, label in enumerate(y_labels):
        if str(label).startswith("__ygap"):
            cursor += _ROW_GAP
            continue
        positions.append(cursor)
        indexes.append(index)
        cursor += 1.0
    span = (positions[-1] + 1.0) if positions else 1.0
    return positions, indexes, span


def _gap_shapes(positions: Sequence[float], n_cols: int) -> list[dict]:
    shapes: list[dict] = []
    spans = ((previous, current) for previous, current in zip(positions, positions[1:]))
    for previous, current in spans:
        if current - previous <= 1.0 + 1e-9:
            continue
        band = {
            "type": "rect",
            "yref": "y",
            "y0": previous + 0.5,
            "y1": current - 0.5,
            "fillcolor": "#ffffff",
            "line": {"width": 0},
            "layer": "above",
        }
        shapes.append({**band, "xref": "x", "x0": -0.5, "x1": n_cols - 0.5})
        shapes.append({**band, "xref": "x2", "x0": -0.5, "x1": 0.5})
    return shapes


def _annotation_font() -> dict:
    return {"size": _FONT, "family": _FONT_FAMILY, "color": "#222"}


def _y_annotations(
    labels: Sequence[str],
    positions: Sequence[float],
    plot_w: int,
) -> list[dict]:
    """Right edge sits ``_Y_LABEL_GAP`` pixels before the heatmap."""
    x = -_Y_LABEL_GAP / plot_w if plot_w else 0
    return [
        {
            "x": x,
            "xref": "paper",
            "xanchor": "right",
            "y": y,
            "yref": "y",
            "yanchor": "middle",
            "text": label,
            "showarrow": False,
            "font": _annotation_font(),
            "labelkind": "y",
        }
        for label, y in zip(labels, positions)
    ]


def _axis_label_y(plot_h: float, *, outer: bool, top: bool) -> float:
    """Paper y. The load row sits against the cells; the span row is outside it."""
    inner = 6.0 + _LABEL_H / 2
    offset = (6.0 + _LABEL_H + 2.0 + _LABEL_H / 2) if outer else inner
    signed = offset / plot_h
    return 1.0 + signed if top else -signed


def _x_annotations(
    problems: Sequence[str],
    plot_h: float,
    cell_w: float,
    overall_w: int = _OVERALL_W,
    axis: dict | None = None,
) -> list[dict]:
    """Span outside the cells and the load id against them, on the top and the bottom.

    The span field is at least three load columns wide. ``axis`` is the padded
    category layout; without it the problems sit in one column each and the
    span field still uses that minimum.
    """
    layout = axis if axis is not None else _span_axis(problems)
    problem_x = layout["problem_x"]
    notes: list[dict] = []
    for top in (True, False):
        load_y = _axis_label_y(plot_h, outer=False, top=top)
        span_y = _axis_label_y(plot_h, outer=True, top=top)
        for index, problem_id in enumerate(problems):
            match = _PROBLEM_ID.fullmatch(problem_id)
            notes.append(
                {
                    "x": problem_x[index],
                    "xref": "x",
                    "xanchor": "center",
                    "y": load_y,
                    "yref": "paper",
                    "yanchor": "middle",
                    "text": match.group(2) if match is not None else problem_id,
                    "align": "center",
                    "width": cell_w - 2,
                    "height": _LABEL_H,
                    "bgcolor": "#ffffff",
                    "labelkind": "load",
                    "borderwidth": 0,
                    "borderpad": 1,
                    "showarrow": False,
                    "font": _annotation_font(),
                }
            )
        for group in layout["groups"]:
            if group["span"] is None:
                continue
            slots = _span_slots(group["loads"])
            notes.append(
                {
                    "x": (group["start"] + group["end"]) / 2,
                    "xref": "x",
                    "xanchor": "center",
                    "y": span_y,
                    "yref": "paper",
                    "yanchor": "middle",
                    "text": _span_label(group["span"]),
                    "align": "center",
                    "width": _span_field_width(group["loads"], cell_w),
                    "height": _LABEL_H,
                    "bgcolor": _SPAN_BG,
                    "borderwidth": 0,
                    "borderpad": 1,
                    "showarrow": False,
                    "font": {**_annotation_font(), "size": _SPAN_FONT},
                    "labelkind": "span",
                    "spancount": slots,
                }
            )
        notes.append(
            {
                "x": _overall_text_x(overall_w),
                "xref": "x2",
                "xanchor": "left",
                "y": load_y,
                "yref": "paper",
                "yanchor": "middle",
                "text": "Overall",
                "align": "left",
                "showarrow": False,
                "font": _annotation_font(),
                "labelkind": "overall-header",
            }
        )
    return notes


def _take_rows(matrix: list, indexes: Sequence[int]) -> list:
    return [matrix[index] for index in indexes]


def _grey_row(values: Sequence[float | None]) -> list[float | None]:
    return [0 if value is None else None for value in values]


def _text_width(text: str) -> int:
    plain = re.sub(r"<br\s*/?>", "", text)
    return math.ceil(len(plain) * _FONT * 0.5)


def _dejavu_font():
    """13px DejaVu Sans, the face embedded in the heatmap page."""
    font = getattr(_dejavu_font, "face", None)
    if font is None:
        import matplotlib
        from PIL import ImageFont

        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        font = ImageFont.truetype(str(path), _FONT)
        _dejavu_font.face = font
    return font


def _label_px(text: str) -> int:
    """Advance width of ``text`` in the heatmap face."""
    if not text:
        return 0
    return math.ceil(_dejavu_font().getlength(text))


def _overall_column_width(labels: Sequence[str]) -> int:
    """Width of the overall column.

    Short counts stay at ``_OVERALL_W``. A long pooled count, such as
    ``94.0 % (1973/2100)``, widens the column enough for that label.
    """
    pad = _OVERALL_INSET + _OVERALL_TAIL
    width = _OVERALL_W
    for text in labels:
        if not text:
            continue
        width = max(width, _label_px(text) + pad)
    return width


def _cell_size(ticktext: Sequence[str]) -> tuple[int, int]:
    loads = [tick.split("<br>")[-1] for tick in ticktext]
    longest = max((_text_width(load) for load in loads), default=_text_width("A1"))
    return longest + _PAD_X, _FONT + _PAD_Y


def _figure_annotations(notes: Sequence[dict]) -> list[dict]:
    hidden = {"labelkind", "spancount"}
    return [{key: value for key, value in note.items() if key not in hidden} for note in notes]


def _overall_text_x(overall_w: int = _OVERALL_W) -> float:
    """Left edge of the overall column, inset by a few pixels."""
    return -0.5 + _OVERALL_INSET / overall_w


def _axis_domains(
    problem_w: float,
    overall_w: int = _OVERALL_W,
) -> tuple[list[float], list[float], float]:
    inner = problem_w + _OVERALL_GAP + overall_w
    overall_start = (problem_w + _OVERALL_GAP) / inner
    return (
        [0, problem_w / inner],
        [overall_start, 1],
        1 + 8 / inner,
    )


def _figure_width(
    n_cols: int,
    y_labels: Sequence[str],
    cell_w: int,
    overall_w: int = _OVERALL_W,
) -> tuple[int, int]:
    labels = [label for label in y_labels if not str(label).startswith("__")]
    left = max((_text_width(label) for label in labels), default=48) + _Y_LABEL_GAP + _Y_LABEL_PAD
    inner = n_cols * cell_w + _OVERALL_GAP + overall_w
    return left + inner + _RIGHT_PAD, left


def _figure_height(n_rows: int, cell_h: int) -> int:
    return _MARGIN_TOP + n_rows * cell_h + _MARGIN_BOTTOM


def _view_block(
    *,
    summary_label: str,
    labels: Sequence[str],
    row_gaps: Sequence[int],
    column_index: Sequence[int],
    n_cols: int,
    x_labels: Sequence[str],
    cell_h: int,
    summary_rates: dict[str, list[float | None]],
    body_rates: dict[str, list[list[float | None]]],
    summary_hover: dict[str, list[dict]],
    body_hover: dict[str, list[list[dict]]],
) -> dict:
    y_labels = _y_labels(summary_label, labels, row_gaps)
    y_ticks = [summary_label, *labels]
    z: dict[str, list] = {}
    grey: dict[str, list] = {}
    customdata: dict[str, list] = {}
    for key in summary_rates:
        z[key] = _stack(
            summary_rates[key],
            body_rates[key],
            row_gaps,
            column_index,
            n_cols,
        )
        grey[key] = _stack(
            _grey_row(summary_rates[key]),
            [_grey_row(row) for row in body_rates[key]],
            row_gaps,
            column_index,
            n_cols,
        )
        customdata[key] = _stack(
            [_pack(cell) for cell in summary_hover[key]],
            [[_pack(cell) for cell in row] for row in body_hover[key]],
            row_gaps,
            column_index,
            n_cols,
        )
    return {
        "y": y_labels,
        "y_tickvals": y_ticks,
        "z": z,
        "grey": grey,
        "customdata": customdata,
        "height": _figure_height(len(y_labels), cell_h),
        "shapes": [],
    }


def _finish_view(
    block: dict,
    problems: Sequence[str],
    n_cols: int,
    _margin_left: int,
    problem_w: int,
    cell_h: int,
    overall_w: int = _OVERALL_W,
    axis: dict | None = None,
) -> None:
    positions, indexes, span = _row_geometry(block["y"])
    plot_h = max(span * cell_h, 1.0)
    inner = problem_w + _OVERALL_GAP + overall_w
    block["y_pos"] = positions
    block["row_index"] = indexes
    block["yrange"] = [positions[-1] + 0.5, -0.5] if positions else [0.5, -0.5]
    block["height"] = int(round(_MARGIN_TOP + plot_h + _MARGIN_BOTTOM))
    cell_w = problem_w / n_cols if n_cols else 0
    block["shapes"] = _gap_shapes(positions, n_cols)
    block["annotations"] = _x_annotations(
        problems,
        plot_h,
        cell_w,
        overall_w,
        axis,
    ) + _y_annotations(
        block["y_tickvals"],
        positions,
        inner,
    )
    block["overall_labels"] = _overall_labels(
        block.get("overall"),
        positions,
        overall_w,
    )


def _family_class(family: _Family) -> str | None:
    for token in family.tokens():
        try:
            return presentation_for(token).algorithm_class
        except AlgorithmPresentationError:
            continue
    return None


def _present_classes(families: Sequence[_Family]) -> tuple[str, ...]:
    present = {label for family in families if (label := _family_class(family))}
    rank = {label: index for index, label in enumerate(CLASS_ORDER)}
    return tuple(sorted(present, key=lambda label: (rank.get(label, len(CLASS_ORDER)), label)))


def _algorithm_menu(classes: Sequence[str]) -> list[dict[str, str]]:
    options = [
        {"value": "all", "label": "All algorithms"},
        {"value": "top:3", "label": "Top 3"},
        {"value": "top:5", "label": "Top 5"},
        {"value": "top:10", "label": "Top 10"},
    ]
    options.extend({"value": f"class:{label}", "label": label} for label in classes)
    return options


def _selected_families(families: Sequence[_Family], value: str) -> list[_Family]:
    if value == "all":
        return list(families)
    if value.startswith("top:"):
        return list(families[: int(value[4:])])
    label = value.split(":", 1)[1]
    return [family for family in families if _family_class(family) == label]


def _pool_counts(row: Sequence[tuple[int, int] | None]) -> tuple[int, int] | None:
    hits = 0
    runs = 0
    seen = False
    for item in row:
        if item is None:
            continue
        hits += int(item[0])
        runs += int(item[1])
        seen = True
    if not seen or runs == 0:
        return None
    return hits, runs


def _overall_rate_label(hits: int, runs: int) -> str:
    return f"{100.0 * hits / runs:.1f} % ({hits}/{runs})"


def _pooled_overall_labels(counts: dict[str, list]) -> list[str]:
    labels: list[str] = []
    for rows in counts.values():
        for row in rows:
            pooled = _pool_counts(row)
            if pooled is None:
                continue
            labels.append(_overall_rate_label(*pooled))
    return labels


def _overall_entry(
    *,
    text: str,
    value: float | None,
    color: str,
    cell: dict,
) -> dict:
    return {
        "text": text,
        "z": value,
        "color": color,
        "font": _ink(color),
        "customdata": _pack(cell),
    }


def _success_overall_column(
    count_rows: Sequence[Sequence[tuple[int, int] | None]],
    percent: float,
) -> list[dict]:
    entries: list[dict] = []
    rates: list[float] = []
    for row in count_rows:
        pooled = _pool_counts(row)
        if pooled is None:
            entries.append(
                _overall_entry(
                    text="",
                    value=None,
                    color=_GREY,
                    cell=_blank_cell("Overall", percent, display_name="Overall"),
                )
            )
            continue
        hits, runs = pooled
        rate = hits / runs
        rates.append(rate)
        entries.append(
            _overall_entry(
                text=_overall_rate_label(hits, runs),
                value=rate,
                color=_swatch(rate),
                cell=_blank_cell(
                    "Overall",
                    percent,
                    kind="rate",
                    display_name="Overall",
                    fraction=f"{hits}/{runs}",
                    rate=_rate_text(rate),
                    line1=f"{_rate_text(rate)} ({hits}/{runs} runs)",
                    color=_swatch(rate),
                ),
            )
        )
    if rates:
        mean = sum(rates) / float(len(rates))
        color = _swatch(mean)
        summary = _overall_entry(
            text=f"{100.0 * mean:.1f} %",
            value=mean,
            color=color,
            cell=_blank_cell(
                "Overall",
                percent,
                kind="summary",
                display_name="All algorithms",
                subtitle=(
                    _mean_note(len(rates), "algorithm")
                    + ". Not a pooled count of runs."
                ),
                rate=_rate_text(mean),
                line1=_rate_text(mean),
                color=color,
            ),
        )
    else:
        summary = _overall_entry(
            text="",
            value=None,
            color=_GREY,
            cell=_blank_cell("Overall", percent, display_name="All algorithms"),
        )
    return [summary, *entries]


def _improvement_overall_column(
    families: Sequence[_Family],
    token_index: dict[str, int],
    count_rows: Sequence[Sequence[tuple[int, int] | None]],
    percent: float,
) -> list[dict]:
    entries: list[dict] = []
    deltas: list[float] = []
    for family in families:
        cold = (
            _pool_counts(count_rows[token_index[family.cold]])
            if family.cold in token_index
            else None
        )
        warm = (
            _pool_counts(count_rows[token_index[family.warm]])
            if family.warm in token_index
            else None
        )
        if cold is None or warm is None:
            entries.append(
                _overall_entry(
                    text="",
                    value=None,
                    color=_GREY,
                    cell=_blank_cell("Overall", percent),
                )
            )
            continue
        cold_rate = cold[0] / cold[1]
        warm_rate = warm[0] / warm[1]
        delta = 100.0 * (warm_rate - cold_rate)
        deltas.append(delta)
        color = _swatch((delta + 100.0) / 200.0, _RWB)
        entries.append(
            _overall_entry(
                text=_format_pp(delta),
                value=delta,
                color=color,
                cell=_blank_cell(
                    "Overall",
                    percent,
                    kind="delta",
                    display_name="Overall",
                    line1=f"Cold start {_overall_rate_label(*cold)}",
                    line2=f"Warm start {_overall_rate_label(*warm)}",
                    line3=_format_pp(delta),
                    difference=_format_pp(delta),
                    color=color,
                ),
            )
        )
    if deltas:
        mean = sum(deltas) / float(len(deltas))
        color = _swatch((mean + 100.0) / 200.0, _RWB)
        summary = _overall_entry(
            text=_format_pp(mean),
            value=mean,
            color=color,
            cell=_blank_cell(
                "Overall",
                percent,
                kind="summary",
                display_name="All families",
                subtitle=_mean_note(len(deltas), "family"),
                line1=_format_pp(mean),
                difference=_format_pp(mean),
                color=color,
            ),
        )
    else:
        summary = _overall_entry(
            text="",
            value=None,
            color=_GREY,
            cell=_blank_cell("Overall", percent, display_name="All families"),
        )
    return [summary, *entries]


def _overall_payload(entries_by_margin: dict[str, list[dict]]) -> dict:
    payload: dict[str, dict] = {"z": {}, "text": {}, "grey": {}, "customdata": {}}
    for key, entries in entries_by_margin.items():
        payload["z"][key] = [[entry["z"]] for entry in entries]
        payload["text"][key] = [entry["text"] for entry in entries]
        payload["grey"][key] = [[0 if entry["z"] is None else None] for entry in entries]
        payload["customdata"][key] = [[entry["customdata"]] for entry in entries]
    payload["font"] = {
        key: [entry["font"] for entry in entries]
        for key, entries in entries_by_margin.items()
    }
    return payload


def _overall_labels(
    overall: dict | None,
    positions: Sequence[float],
    overall_w: int = _OVERALL_W,
) -> dict[str, list]:
    if not overall:
        return {}
    labels: dict[str, list] = {}
    for key, texts in overall["text"].items():
        notes = []
        for text, font, y in zip(texts, overall["font"][key], positions):
            if not text:
                continue
            notes.append(
                {
                    "x": _overall_text_x(overall_w),
                    "xref": "x2",
                    "xanchor": "left",
                    "y": y,
                    "yref": "y",
                    "yanchor": "middle",
                    "text": text,
                    "align": "left",
                    "showarrow": False,
                    "font": {
                        "size": _FONT,
                        "family": _FONT_FAMILY,
                        "color": font,
                    },
                    "labelkind": "overall",
                }
            )
        labels[key] = notes
    return labels


def _column_summary(
    rows: Sequence[Sequence[float | None]],
    problems: Sequence[str],
    percent: float,
    *,
    label: str,
    kind_name: str,
) -> tuple[list[float | None], list[dict]]:
    summary: list[float | None] = []
    hover: list[dict] = []
    for column, problem_id in enumerate(problems):
        present = [
            float(row[column])
            for row in rows
            if row[column] is not None
        ]
        mean, cell = _summary_cell(
            present,
            problem_id,
            percent,
            label=label,
            kind_name=kind_name,
        )
        summary.append(mean)
        hover.append(cell)
    return summary, hover


def _views_for_selection(
    *,
    selected: Sequence[_Family],
    included_at: dict[str, int],
    token_index: dict[str, int],
    labels: dict[str, tuple[str, str]],
    axis_labels: Sequence[str],
    problems: Sequence[str],
    margins: Sequence[str],
    rates: dict[str, list],
    hovers: dict[str, list],
    counts: dict[str, list],
    delta_rates: dict[str, list],
    delta_hovers: dict[str, list],
    cell_h: int,
    margin_left: int,
    plot_w: int,
    overall_w: int,
    axis: dict,
) -> dict[str, dict]:
    """One success view and one improvement view, rows in ``selected`` order."""
    paired = [family for family in selected if family.family_id in included_at]
    row_indexes = [
        token_index[token] for family in selected for token in family.tokens()
    ]
    pair_indexes = [included_at[family.family_id] for family in paired]
    success_rates: dict[str, list] = {}
    success_hover: dict[str, list] = {}
    success_summary: dict[str, list] = {}
    success_summary_hover: dict[str, list] = {}
    family_rates: dict[str, list] = {}
    family_hover: dict[str, list] = {}
    family_summary: dict[str, list] = {}
    family_summary_hover: dict[str, list] = {}
    success_overall: dict[str, list] = {}
    improvement_overall: dict[str, list] = {}
    for key in margins:
        percent = float(key)
        body = [rates[key][index] for index in row_indexes]
        body_hover = [hovers[key][index] for index in row_indexes]
        mean, cells = _column_summary(
            body,
            problems,
            percent,
            label="All algorithms",
            kind_name="algorithm",
        )
        success_rates[key] = body
        success_hover[key] = body_hover
        success_summary[key] = mean
        success_summary_hover[key] = cells
        deltas = [delta_rates[key][index] for index in pair_indexes]
        delta_cells = [delta_hovers[key][index] for index in pair_indexes]
        family_mean, family_cells = _column_summary(
            deltas,
            problems,
            percent,
            label="All families",
            kind_name="family",
        )
        family_rates[key] = deltas
        family_hover[key] = delta_cells
        family_summary[key] = family_mean
        family_summary_hover[key] = family_cells
        success_overall[key] = _success_overall_column(
            [counts[key][index] for index in row_indexes],
            percent,
        )
        improvement_overall[key] = _improvement_overall_column(
            paired,
            token_index,
            counts[key],
            percent,
        )
    n_cols = len(axis["categories"])
    success = _view_block(
        summary_label="All algorithms",
        labels=[axis_labels[index] for index in row_indexes],
        row_gaps=_family_gap_indexes(selected),
        column_index=axis["problem_x"],
        n_cols=n_cols,
        x_labels=axis["categories"],
        cell_h=cell_h,
        summary_rates=success_summary,
        body_rates=success_rates,
        summary_hover=success_summary_hover,
        body_hover=success_hover,
    )
    improvement = _view_block(
        summary_label="All families",
        labels=[labels[family.family_id][0] for family in paired],
        row_gaps=[],
        column_index=axis["problem_x"],
        n_cols=n_cols,
        x_labels=axis["categories"],
        cell_h=cell_h,
        summary_rates=family_summary,
        body_rates=family_rates,
        summary_hover=family_summary_hover,
        body_hover=family_hover,
    )
    success["overall"] = _overall_payload(success_overall)
    improvement["overall"] = _overall_payload(improvement_overall)
    _finish_view(
        success,
        problems,
        n_cols,
        margin_left,
        plot_w,
        cell_h,
        overall_w,
        axis,
    )
    _finish_view(
        improvement,
        problems,
        n_cols,
        margin_left,
        plot_w,
        cell_h,
        overall_w,
        axis,
    )
    return {"success": success, "improvement": improvement}


def _filtered_blocks(
    *,
    families: Sequence[_Family],
    included: Sequence[_Family],
    algorithms: Sequence[str],
    labels: dict[str, tuple[str, str]],
    axis_labels: Sequence[str],
    problems: Sequence[str],
    margins: Sequence[str],
    rates: dict[str, list],
    hovers: dict[str, list],
    counts: dict[str, list],
    delta_rates: dict[str, list],
    delta_hovers: dict[str, list],
    cell_h: int,
    margin_left: int,
    plot_w: int,
    overall_w: int,
    menu: Sequence[dict[str, str]],
    grouped: dict[tuple[str, str], list[RunTrajectory]],
    axis: dict,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Menu blocks in the stored order, plus one row order per tolerance."""
    included_at = {family.family_id: index for index, family in enumerate(included)}
    token_index = {token: index for index, token in enumerate(algorithms)}
    shared = dict(
        included_at=included_at,
        token_index=token_index,
        labels=labels,
        axis_labels=axis_labels,
        problems=problems,
        rates=rates,
        hovers=hovers,
        counts=counts,
        delta_rates=delta_rates,
        delta_hovers=delta_hovers,
        cell_h=cell_h,
        margin_left=margin_left,
        plot_w=plot_w,
        overall_w=overall_w,
        axis=axis,
    )
    ranked = {
        key: _families_in_rank_order(families, grouped, problems, float(key))
        for key in margins
    }
    filters: dict[str, dict] = {}
    layouts: dict[str, dict] = {}
    for option in menu:
        value = option["value"]
        filters[value] = _views_for_selection(
            selected=_selected_families(families, value),
            margins=margins,
            **shared,
        )
        layouts[value] = {
            key: _views_for_selection(
                selected=_selected_families(ranked[key], value),
                margins=[key],
                **shared,
            )
            for key in margins
        }
    return filters, layouts


def _warn_pooled_cells(
    grouped: dict[tuple[str, str], list[RunTrajectory]],
) -> None:
    for algorithm_name, problem_id in sorted(grouped):
        directories = sorted(
            {
                str(Path(run.experiment_dir))
                for run in grouped[(algorithm_name, problem_id)]
            }
        )
        if len(directories) < 2:
            continue
        print(
            "hdb-plot: warning: "
            f"{algorithm_name} on {problem_id} pools runs from "
            + ", ".join(directories),
            file=sys.stderr,
        )


def _payload(
    bundle: ExperimentBundle,
    margin_percents: tuple[float, ...] = MARGINS,
) -> dict:
    grouped = _groups(bundle)
    problems = _problem_order([problem_id for _name, problem_id in grouped])
    families, labels = _ordered_families(grouped, problems)
    algorithms = [token for family in families for token in family.tokens()]
    family_gaps = _family_gap_indexes(families)
    column_gaps = _column_gaps(problems)
    qualified = qualifiers_needed(
        [name for name in algorithms if _has_presentation(name)]
    )
    axis_labels = [_axis_label(name, algorithms) for name in algorithms]
    included = [
        family for family in families if _has_pair(family, grouped, problems)
    ]
    margin_values = tuple(float(item) for item in margin_percents)
    margins = [f"{percent:g}" for percent in margin_values]
    rates: dict[str, list] = {}
    texts: dict[str, list] = {}
    hovers: dict[str, list] = {}
    summary_rates: dict[str, list] = {}
    summary_hovers: dict[str, list] = {}
    delta_rates: dict[str, list] = {}
    delta_hovers: dict[str, list] = {}
    family_summary_rates: dict[str, list] = {}
    family_summary_hovers: dict[str, list] = {}
    counts: dict[str, list] = {}
    for percent in margin_values:
        key = f"{percent:g}"
        rate_rows: list[list[float | None]] = []
        text_rows: list[list[str]] = []
        hover_rows: list[list[dict]] = []
        count_rows: list[list[tuple[int, int] | None]] = []
        for token in algorithms:
            rate_row: list[float | None] = []
            text_row: list[str] = []
            hover_row: list[dict] = []
            count_row: list[tuple[int, int] | None] = []
            for problem_id in problems:
                runs = grouped.get((token, problem_id))
                rate, fraction, cell = _rate_cell(
                    token,
                    problem_id,
                    runs,
                    percent,
                    show_qualifier=token in qualified,
                )
                rate_row.append(rate)
                text_row.append(fraction)
                hover_row.append(cell)
                count_row.append(_counts(runs, percent))
            rate_rows.append(rate_row)
            text_rows.append(text_row)
            hover_rows.append(hover_row)
            count_rows.append(count_row)
        rates[key] = rate_rows
        texts[key] = text_rows
        hovers[key] = hover_rows
        counts[key] = count_rows
        summary_row: list[float | None] = []
        summary_hover: list[dict] = []
        for column, problem_id in enumerate(problems):
            present = [
                rate_rows[row][column]
                for row in range(len(algorithms))
                if rate_rows[row][column] is not None
            ]
            mean, cell = _summary_cell(
                [float(rate) for rate in present],
                problem_id,
                percent,
                label="All algorithms",
                kind_name="algorithm",
            )
            summary_row.append(mean)
            summary_hover.append(cell)
        summary_rates[key] = summary_row
        summary_hovers[key] = summary_hover
        delta_rows: list[list[float | None]] = []
        delta_hover_rows: list[list[dict]] = []
        for family in included:
            label, subtitle = labels[family.family_id]
            delta_row: list[float | None] = []
            delta_hover: list[dict] = []
            for problem_id in problems:
                delta, cell = _delta_cell(
                    family,
                    label,
                    subtitle,
                    problem_id,
                    grouped,
                    percent,
                )
                delta_row.append(delta)
                delta_hover.append(cell)
            delta_rows.append(delta_row)
            delta_hover_rows.append(delta_hover)
        delta_rates[key] = delta_rows
        delta_hovers[key] = delta_hover_rows
        family_summary: list[float | None] = []
        family_hover: list[dict] = []
        width = len(problems)
        for column in range(width):
            present = [
                delta_rows[row][column]
                for row in range(len(included))
                if delta_rows[row][column] is not None
            ]
            problem_id = problems[column]
            mean, cell = _summary_cell(
                [float(value) for value in present],
                problem_id,
                percent,
                label="All families",
                kind_name="family",
            )
            family_summary.append(mean)
            family_hover.append(cell)
        family_summary_rates[key] = family_summary
        family_summary_hovers[key] = family_hover
    axis = _span_axis(problems)
    x_labels = axis["categories"]
    ticktext = _problem_ticks(problems)
    cell_w, cell_h = _cell_size(ticktext)
    overall_w = _overall_column_width(_pooled_overall_labels(counts))
    width, margin_left = _figure_width(
        len(x_labels),
        ["All algorithms", *axis_labels],
        cell_w,
        overall_w,
    )
    plot_w = len(x_labels) * cell_w
    menu = _algorithm_menu(_present_classes(families))
    filters, layouts = _filtered_blocks(
        families=families,
        included=included,
        algorithms=algorithms,
        labels=labels,
        axis_labels=axis_labels,
        problems=problems,
        margins=margins,
        rates=rates,
        hovers=hovers,
        counts=counts,
        delta_rates=delta_rates,
        delta_hovers=delta_hovers,
        cell_h=cell_h,
        margin_left=margin_left,
        plot_w=plot_w,
        overall_w=overall_w,
        menu=menu,
        grouped=grouped,
        axis=axis,
    )
    success = filters["all"]["success"]
    improvement = filters["all"]["improvement"]
    return {
        "algorithms": algorithms,
        "axis_labels": axis_labels,
        "summary_label": "All algorithms",
        "problems": problems,
        "problem_ticks": ticktext,
        "column_gaps_after": column_gaps,
        "family_gaps_after": family_gaps,
        "margins": margins,
        "initial_margin": "1",
        "initial_view": "success",
        "initial_algorithms": "all",
        "algorithm_menu": menu,
        "rates": rates,
        "text": texts,
        "hover": hovers,
        "summary": {
            "rates": {key: [summary_rates[key]] for key in margins},
            "hover": {key: [summary_hovers[key]] for key in margins},
        },
        "improvement": {
            "labels": [labels[family.family_id][0] for family in included],
            "deltas": delta_rates,
            "hover": delta_hovers,
            "summary_label": "All families",
            "summary": {
                "deltas": family_summary_rates,
                "hover": family_summary_hovers,
            },
        },
        "captions": {
            "success": {
                key: _description(_budget_text(grouped), algorithms)
                for key in margins
            },
            "improvement": {
                key: _description(_budget_text(grouped), algorithms)
                for key in margins
            },
        },
        "titles": {
            "success": _SUCCESS_TITLE,
            "improvement": _IMPROVEMENT_TITLE,
        },
        "colorbar_titles": {key: "Success rate (%)" for key in margins},
        "ticktext": ["0%", "25%", "50%", "75%", "100%"],
        "colorscale": "RdYlGn",
        "zmin": 0,
        "zmax": 1,
        "missing_color": _GREY,
        "style": {
            "success": {
                "zmin": 0,
                "zmax": 1,
                "tickvals": [0, 0.25, 0.5, 0.75, 1],
                "ticktext": ["0%", "25%", "50%", "75%", "100%"],
                "colorbar": "Success rate (%)",
                "colorscale": _RDYLGN_SCALE,
                "reversescale": False,
            },
            "improvement": {
                "zmin": -100,
                "zmax": 100,
                "tickvals": [-100, -50, 0, 50, 100],
                "ticktext": ["−100", "−50", "0", "+50", "+100"],
                "colorbar": "Improvement (pp)",
                "colorscale": _RWB_SCALE,
                "reversescale": False,
            },
        },
        "display": {
            "x": x_labels,
            "x_tickvals": list(problems),
            "x_ticktext": ticktext,
            "width": width,
            "margin_left": margin_left,
            "cell_w": cell_w,
            "cell_h": cell_h,
            "geometry": {
                "margin_left": margin_left,
                "margin_top": _MARGIN_TOP,
                "margin_bottom": _MARGIN_BOTTOM,
                "right_pad": _RIGHT_PAD,
                "overall_gap": _OVERALL_GAP,
                "overall_w": overall_w,
                "overall_inset": _OVERALL_INSET,
                "overall_tail": _OVERALL_TAIL,
                "min_cell_w": cell_w,
                "n_cols": len(x_labels),
                "min_width": width,
                "font_size": _FONT,
                "axis_title_size": _AXIS_TITLE_SIZE,
                "font_family": _FONT_STACK,
                "label_pad": _Y_LABEL_PAD,
                "label_gap": _Y_LABEL_GAP,
                "span_gap": _SPAN_GAP,
                "min_span_loads": _MIN_SPAN_LOADS,
                "margin_labels": [
                    "All algorithms",
                    "All families",
                    *axis_labels,
                    *[labels[family.family_id][0] for family in included],
                ],
            },
            "success": success,
            "improvement": improvement,
            "filters": filters,
            "layouts": layouts,
        },
    }


def _figure(payload: dict):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ValueError(
            "success heatmap requires plotly; install with: "
            'pip install -e ".[visualization-interactive]"'
        ) from exc

    margin = payload["initial_margin"]
    block = payload["display"]["layouts"][payload["initial_algorithms"]][margin]["success"]
    style = payload["style"]["success"]
    indexes = block["row_index"]
    overall_w = int(payload["display"]["geometry"]["overall_w"])
    problem_domain, overall_domain, colorbar_x = _axis_domains(
        len(payload["display"]["x"]) * payload["display"]["cell_w"],
        overall_w,
    )
    shared = {
        "x": payload["display"]["x"],
        "y": block["y_pos"],
        "hoverongaps": False,
        "xgap": 1,
        "ygap": 1,
        "showscale": False,
    }
    colorbar = {
        "title": {
            "text": style["colorbar"],
            "side": "right",
            "font": {"family": _FONT_FAMILY, "size": _AXIS_TITLE_SIZE, "color": "#222222"},
        },
        "tickfont": {"family": _FONT_FAMILY, "size": _FONT, "color": "#222222"},
        "tickvals": style["tickvals"],
        "ticktext": style["ticktext"],
        "len": 1,
        "thickness": 12,
        "y": 0.5,
        "x": colorbar_x,
        "xref": "paper",
        "xanchor": "left",
        "xpad": 0,
    }
    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            z=_take_rows(block["grey"][margin], indexes),
            colorscale=[[0, _GREY], [1, _GREY]],
            zmin=0,
            zmax=1,
            hoverinfo="skip",
            **shared,
        )
    )
    figure.add_trace(
        go.Heatmap(
            z=_take_rows(block["z"][margin], indexes),
            customdata=_take_rows(block["customdata"][margin], indexes),
            colorscale=style["colorscale"],
            reversescale=style["reversescale"],
            zmin=style["zmin"],
            zmax=style["zmax"],
            showscale=True,
            hoverinfo="none",
            hoverongaps=True,
            xgap=1,
            ygap=1,
            colorbar=colorbar,
            x=payload["display"]["x"],
            y=block["y_pos"],
        )
    )
    figure.add_trace(
        go.Heatmap(
            z=block["overall"]["grey"][margin],
            x=["Overall"],
            y=block["y_pos"],
            xaxis="x2",
            colorscale=[[0, _GREY], [1, _GREY]],
            zmin=0,
            zmax=1,
            hoverinfo="skip",
            showscale=False,
        )
    )
    figure.add_trace(
        go.Heatmap(
            z=block["overall"]["z"][margin],
            customdata=block["overall"]["customdata"][margin],
            x=["Overall"],
            y=block["y_pos"],
            xaxis="x2",
            colorscale=style["colorscale"],
            reversescale=style["reversescale"],
            zmin=style["zmin"],
            zmax=style["zmax"],
            hoverinfo="none",
            showscale=False,
            xgap=0,
            ygap=1,
        )
    )
    figure.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font={"family": _FONT_FAMILY, "size": _FONT, "color": "#222222"},
        width=payload["display"]["width"],
        height=block["height"],
        margin={
            "t": _MARGIN_TOP,
            "b": _MARGIN_BOTTOM,
            "l": payload["display"]["margin_left"],
            "r": _RIGHT_PAD,
            "autoexpand": False,
        },
        shapes=block["shapes"],
        annotations=_figure_annotations(
            block["annotations"] + block["overall_labels"][margin]
        ),
        xaxis={
            "type": "category",
            "showticklabels": False,
            "ticks": "",
            "automargin": False,
            "categoryorder": "array",
            "categoryarray": list(payload["display"]["x"]),
            "range": [-0.5, len(payload["display"]["x"]) - 0.5],
            "domain": problem_domain,
        },
        xaxis2={
            "type": "category",
            "showticklabels": False,
            "ticks": "",
            "automargin": False,
            "anchor": "y",
            "domain": overall_domain,
            "range": [-0.5, 0.5],
            "categoryorder": "array",
            "categoryarray": ["Overall"],
        },
        yaxis={
            "type": "linear",
            "showticklabels": False,
            "ticks": "",
            "automargin": False,
            "range": block["yrange"],
        },
    )
    return figure


def _inject_margin_menu(html: str, payload: dict, *, downloads: bool = True) -> str:
    algorithm_options = "\n".join(
        (
            f'<option value="{html_lib.escape(option["value"], quote=True)}"'
            f'{" selected" if option["value"] == payload["initial_algorithms"] else ""}>'
            f'{html_lib.escape(option["label"])}</option>'
        )
        for option in payload["algorithm_menu"]
    )
    options = "\n".join(
        (
            f'<option value="{margin}" selected>{margin}%</option>'
            if margin == payload["initial_margin"]
            else f'<option value="{margin}">{margin}%</option>'
        )
        for margin in payload["margins"]
    )
    encoded = json.dumps(payload)
    initial_title = payload["titles"]["success"]
    from core.visualization_core.optimization.plots.interactive_pool import (
        _dejavu_face,
    )

    export_buttons = ""
    if downloads:
        export_buttons = (
            '<button type="button" id="heatmap-png">PNG</button>\n'
            '<button type="button" id="heatmap-svg">SVG</button>\n'
            '<button type="button" id="heatmap-table">CSV</button>'
        )
    controls = f"""
<style>
{_dejavu_face()}
#heatmap-shell, #heatmap-title, #heatmap-menus, #heatmap-menus label,
#heatmap-menus select, #heatmap-caption, #heatmap-tip {{
  font-family: {_FONT_STACK};
}}
#heatmap-title {{font-size:{_TITLE_SIZE}px; font-weight:700; line-height:1.35; margin:8px 0 10px; color:#222;}}
#heatmap-menus, #heatmap-menus select, #heatmap-png, #heatmap-svg, #heatmap-table {{font-size:{_FONT}px; color:#222; background:#fff;}}
#heatmap-png, #heatmap-svg, #heatmap-table {{margin-left:16px; border:1px solid #bbb; border-radius:2px; padding:1px 8px; cursor:pointer;}}
#heatmap-caption {{font-size:{_FONT}px; line-height:1.45; color:#222;}}
#heatmap-caption .heatmap-formula {{margin:2px 0 10px 18px;}}
#heatmap-caption .formula-note {{margin:0 0 8px 18px;}}
#heatmap-caption .frac {{display:inline-block; vertical-align:middle; text-align:center; margin:0 4px;}}
#heatmap-caption .frac .num {{display:block; border-bottom:1px solid #222; padding:0 8px 1px;}}
#heatmap-caption .frac .den {{display:block; padding-top:1px;}}
</style>
<h2 id="heatmap-title" style="text-align: left;">{initial_title}</h2>
<div id="heatmap-menus" style="margin: 0 0 8px 0; text-align: left;">
<label>Measure <select id="heatmap-view">
<option value="success" selected>Success rate</option>
<option value="improvement">Warm-start improvement</option>
</select></label>
<label style="margin-left: 16px;">Relative target tolerance ε<sub>X</sub> <select id="heatmap-margin">
{options}
</select></label>
<label style="margin-left: 16px;">Algorithms <select id="heatmap-algorithms">
{algorithm_options}
</select></label>
{export_buttons}
</div>
<div id="heatmap-tip" style="display:none; position:fixed; z-index:20; pointer-events:none; max-width:320px; background:#f7f7f7; color:#222; border:1px solid #d5d5d5; border-left-width:4px; border-radius:2px; padding:8px 10px 8px 12px; font-size:12px; line-height:1.35; box-shadow:0 1px 4px rgba(0,0,0,0.12);"></div>
<script id="success-heatmap-data" type="application/json">{encoded}</script>
"""
    script = """
<script>
(function () {
  var payload = JSON.parse(
    document.getElementById("success-heatmap-data").textContent
  );
  var gd = document.getElementById("success-heatmap");
  var tip = document.getElementById("heatmap-tip");
  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
  function hideTip() {
    tip.style.display = "none";
  }
  function showTip(event) {
    var point = event.points && event.points[0];
    var data = point && point.customdata;
    if (!data) return;
    var parts = [
      '<div style="font-weight:700; font-size:13px; margin-bottom:2px;">' +
        escapeHtml(data[1]) + "</div>"
    ];
    if (data[2]) {
      parts.push(
        '<div style="color:#555; font-size:11px; margin-bottom:6px;">' +
          escapeHtml(data[2]) + "</div>"
      );
    }
    if (data[3]) parts.push("<div>" + escapeHtml(data[3]) + "</div>");
    if (data[4]) {
      parts.push('<div style="color:#555;">' + escapeHtml(data[4]) + "</div>");
    }
    if (data[5]) {
      parts.push('<div style="margin-bottom:6px;">' + data[5] + "</div>");
    }
    [data[6], data[7], data[8]].forEach(function (line) {
      if (line) parts.push("<div>" + escapeHtml(line) + "</div>");
    });
    tip.style.borderLeftColor = data[10] || "#b2b2b2";
    tip.innerHTML = parts.join("");
    tip.style.display = "block";
    var mouse = event.event || {};
    var x = (mouse.clientX || 0) + 18;
    var y = (mouse.clientY || 0) + 18;
    if (x + tip.offsetWidth > window.innerWidth - 8) {
      x = (mouse.clientX || 0) - tip.offsetWidth - 18;
    }
    if (y + tip.offsetHeight > window.innerHeight - 8) {
      y = (mouse.clientY || 0) - tip.offsetHeight - 18;
    }
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
  }
  function rows(matrix, indexes) {
    return indexes.map(function (index) { return matrix[index]; });
  }
  function title(view) {
    return payload.titles[view];
  }
  function isYLabel(note) {
    return note.labelkind === "y";
  }
  function labelGap() {
    var geo = payload.display.geometry;
    return geo.label_gap || 8;
  }
  function labelPad() {
    var geo = payload.display.geometry;
    return geo.label_pad || 8;
  }
  function textWidth(text) {
    if (!textWidth.ctx) {
      textWidth.ctx = document.createElement("canvas").getContext("2d");
    }
    textWidth.ctx.font = payload.display.geometry.font_size + 'px "DejaVu Sans"';
    return Math.ceil(textWidth.ctx.measureText(String(text)).width);
  }
  function measuredMargin() {
    var geo = payload.display.geometry;
    try {
      var left = 48;
      (geo.margin_labels || []).forEach(function (text) {
        left = Math.max(left, textWidth(text) + labelGap() + labelPad());
      });
      return left;
    } catch (err) {
      return geo.margin_left;
    }
  }
  function renderedYWidth() {
    var wanted = {};
    (gd.layout.annotations || []).forEach(function (note) {
      if (isYLabel(note) && note.text) wanted[String(note.text)] = true;
    });
    var widest = 0;
    gd.querySelectorAll(".annotation text, text.annotation-text").forEach(function (node) {
      var text = String(node.textContent || "").trim();
      if (!wanted[text]) return;
      var box = node.getBBox();
      if (box && box.width > widest) widest = box.width;
    });
    return widest;
  }
  function domains(problemW) {
    var geo = payload.display.geometry;
    var inner = problemW + geo.overall_gap + geo.overall_w;
    var overallStart = (problemW + geo.overall_gap) / inner;
    return {
      problem: [0, problemW / inner],
      overall: [overallStart, 1],
      colorbar: 1 + 8 / inner,
      inner: inner,
      cell: problemW / geo.n_cols
    };
  }
  function fit(block) {
    var geo = payload.display.geometry;
    var inset = geo.overall_inset || 4;
    var tail = geo.overall_tail || 10;
    try {
      (gd.layout.annotations || []).forEach(function (note) {
        if (note.xref !== "x2" || !note.text) return;
        geo.overall_w = Math.max(
          geo.overall_w,
          textWidth(String(note.text)) + inset + tail
        );
      });
    } catch (err) {}
    var problemW = geo.n_cols * geo.min_cell_w;
    var placed = domains(problemW);
    var marginLeft = measuredMargin();
    var figureWidth = marginLeft + placed.inner + geo.right_pad;
    gd.layout.width = figureWidth;
    gd.layout.margin.l = marginLeft;
    gd.layout.margin.r = geo.right_pad;
    gd.layout.xaxis.domain = placed.problem;
    gd.layout.xaxis2.domain = placed.overall;
    gd.layout.annotations.forEach(function (note) {
      var load = note.labelkind === "load" ||
        (note.xref === "x" && note.yref === "paper" && note.bgcolor === "#ffffff");
      var span = note.labelkind === "span" && note.spancount;
      if (load) note.width = Math.max(placed.cell - 2, 8);
      if (span) {
        var slots = Math.max(note.spancount, geo.min_span_loads || 3);
        note.width = Math.max(slots * placed.cell - geo.span_gap - 2, 8);
      }
      if (isYLabel(note)) {
        note.xanchor = "right";
        note.x = -labelGap() / placed.inner;
      }
      if (note.xref === "x2" && note.xanchor === "left") {
        note.x = -0.5 + inset / geo.overall_w;
      }
    });
    if (gd.data[1] && gd.data[1].colorbar) {
      gd.data[1].colorbar.x = placed.colorbar;
      gd.data[1].colorbar.xanchor = "left";
    }
    gd.style.height = gd.layout.height + "px";
    var caption = document.getElementById("heatmap-caption");
    if (caption) caption.style.maxWidth = figureWidth + "px";
    return figureWidth;
  }
  var refinePasses = 0;
  function refineLabels() {
    var size = gd._fullLayout && gd._fullLayout._size;
    if (refinePasses > 2 || !size || !size.w) return;
    var rendered = renderedYWidth();
    var needed = rendered > 0
      ? Math.ceil(rendered + labelGap() + labelPad())
      : measuredMargin();
    var left = size.l || gd.layout.margin.l || needed;
    var grew = needed > left + 1;
    if (grew) {
      var extra = needed - left;
      gd.layout.margin.l = (gd.layout.margin.l || 0) + extra;
      gd.layout.width = (gd.layout.width || 0) + extra;
    }
    var moved = false;
    var x = -labelGap() / size.w;
    (gd.layout.annotations || []).forEach(function (note) {
      if (!isYLabel(note)) return;
      if (note.xanchor !== "right") moved = true;
      note.xanchor = "right";
      if (Math.abs((note.x || 0) - x) > 1e-4) moved = true;
      note.x = x;
    });
    var caption = document.getElementById("heatmap-caption");
    if (caption && gd.layout.width) caption.style.maxWidth = gd.layout.width + "px";
    if (!(grew || moved)) {
      refinePasses = 0;
      return;
    }
    refinePasses += 1;
    Plotly.react(gd, gd.data, gd.layout).then(function () {
      if (grew) refineLabels();
      else refinePasses = 0;
    });
  }
  function draw() {
    fit();
    return Plotly.react(gd, gd.data, gd.layout).then(refineLabels);
  }
  function apply() {
    var view = document.getElementById("heatmap-view").value;
    var margin = document.getElementById("heatmap-margin").value;
    var filter = document.getElementById("heatmap-algorithms").value;
    var block = payload.display.layouts[filter][margin][view];
    var style = payload.style[view];
    var indexes = block.row_index;
    hideTip();
    var data = gd.data;
    data[0].x = payload.display.x;
    data[0].y = block.y_pos;
    data[0].z = rows(block.grey[margin], indexes);
    data[0].showscale = false;
    data[1].x = payload.display.x;
    data[1].y = block.y_pos;
    data[1].z = rows(block.z[margin], indexes);
    data[1].customdata = rows(block.customdata[margin], indexes);
    data[1].zmin = style.zmin;
    data[1].zmax = style.zmax;
    data[1].colorscale = style.colorscale;
    data[1].reversescale = style.reversescale;
    data[1].showscale = true;
    var geo = payload.display.geometry;
    data[1].colorbar = {
      title: {
        text: style.colorbar,
        side: "right",
        font: {family: "DejaVu Sans", size: geo.axis_title_size, color: "#222222"}
      },
      tickfont: {family: "DejaVu Sans", size: geo.font_size, color: "#222222"},
      tickvals: style.tickvals,
      ticktext: style.ticktext,
      len: 1,
      thickness: 12,
      y: 0.5,
      xref: "paper",
      xpad: 0
    };
    data[2].y = block.y_pos;
    data[2].z = block.overall.grey[margin];
    data[2].showscale = false;
    data[3].y = block.y_pos;
    data[3].z = block.overall.z[margin];
    data[3].customdata = block.overall.customdata[margin];
    data[3].zmin = style.zmin;
    data[3].zmax = style.zmax;
    data[3].colorscale = style.colorscale;
    data[3].reversescale = style.reversescale;
    data[3].showscale = false;
    gd.layout.height = block.height;
    gd.layout.shapes = block.shapes;
    gd.layout.annotations = block.annotations.concat(block.overall_labels[margin]);
    gd.layout.yaxis.type = "linear";
    gd.layout.yaxis.range = block.yrange;
    gd.layout.yaxis.autorange = false;
    gd.layout.yaxis.showticklabels = false;
    fit(block);
    draw();
    document.getElementById("heatmap-title").innerHTML = title(view);
  }
  document.getElementById("heatmap-margin").addEventListener("change", apply);
  document.getElementById("heatmap-view").addEventListener("change", apply);
  document.getElementById("heatmap-algorithms").addEventListener("change", apply);
  var resizeTimer = null;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      draw();
    }, 80);
  });
  draw();
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(function () {
      draw();
    });
  }
  if (gd && gd.on) {
    gd.on("plotly_hover", showTip);
    gd.on("plotly_unhover", hideTip);
  }
  function csvCell(value) {
    var text = value == null ? "" : String(value);
    if (text.indexOf(",") >= 0 || text.indexOf('"') >= 0 || text.indexOf("\\n") >= 0) {
      return '"' + text.replace(/"/g, '""') + '"';
    }
    return text;
  }
  function heatmapCsv() {
    var view = document.getElementById("heatmap-view").value;
    var margin = document.getElementById("heatmap-margin").value;
    var filter = document.getElementById("heatmap-algorithms").value;
    var block = payload.display.layouts[filter][margin][view];
    var problems = payload.display.x || [];
    var overall = (block.overall && block.overall.text && block.overall.text[margin]) || [];
    var valueName = view === "improvement" ? "improvement_pp" : "success_rate";
    var lines = [[
      "measure", "tolerance_percent", "algorithm", "problem", valueName, "overall"
    ].join(",")];
    block.row_index.forEach(function (index, rowNumber) {
      var name = block.y[index];
      var values = (block.z[margin] && block.z[margin][index]) || [];
      var overallText = overall[rowNumber] == null ? "" : overall[rowNumber];
      problems.forEach(function (problem, column) {
        if (String(problem).indexOf("__xpad_") === 0) return;
        var value = values[column];
        lines.push([
          csvCell(view),
          csvCell(margin),
          csvCell(name),
          csvCell(problem),
          value == null ? "" : value,
          csvCell(overallText)
        ].join(","));
      });
    });
    return lines.join("\\n");
  }
  function downloadText(filename, text, type) {
    var blob = new Blob([text], {type: type});
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }
  function downloadFigure(format) {
    var width = gd.layout.width;
    var height = gd.layout.height;
    var name = format === "svg" ? "success_heatmap.svg" : "success_heatmap.png";
    Plotly.toImage(gd, {format: format, width: width, height: height}).then(function (url) {
      var link = document.createElement("a");
      link.href = url;
      link.download = name;
      link.click();
    });
  }
  var pngButton = document.getElementById("heatmap-png");
  var svgButton = document.getElementById("heatmap-svg");
  var tableButton = document.getElementById("heatmap-table");
  if (pngButton) pngButton.addEventListener("click", function () { downloadFigure("png"); });
  if (svgButton) svgButton.addEventListener("click", function () { downloadFigure("svg"); });
  if (tableButton) {
    tableButton.addEventListener("click", function () {
      downloadText("success_heatmap.csv", heatmapCsv(), "text/csv");
    });
  }
})();
</script>
"""
    marker = '<div id="success-heatmap"'
    if marker not in html:
        raise ValueError("success heatmap html is missing the plot div")
    caption = (
        '<div id="heatmap-caption" style="text-align: left; '
        f'max-width: {payload["display"]["width"]}px; margin: 12px 0 16px 0;">'
        f"{payload['captions']['success'][payload['initial_margin']]}</div>"
    )
    html = re.sub(
        r'<div style="height:\d+px; width:\d+px;">',
        '<div id="heatmap-shell" style="width:100%; overflow-x:auto; text-align:left;">',
        html,
        count=1,
    )
    html = html.replace(marker, controls + marker, 1)
    plotly_call = html.rfind("Plotly.newPlot")
    if plotly_call < 0:
        raise ValueError("success heatmap html is missing the plot script")
    script_end = html.find("</script>", plotly_call)
    if script_end < 0:
        raise ValueError("success heatmap html is missing the plot script end")
    script_end += len("</script>")
    html = html[:script_end] + caption + html[script_end:]
    if "</body>" in html:
        return html.replace("</body>", script + "</body>", 1)
    return html + script


def render_success_heatmap(
    bundle: ExperimentBundle,
    out_dir: Path,
    *,
    downloads: bool = True,
    margins: tuple[float, ...] = MARGINS,
) -> list[Path]:
    """
    Write one HTML heatmap for the problems in ``bundle``.

    Columns are the problem ids present on the remaining algorithms.
    Span groups are labelled in metres with A1, B1, and B2 underneath.
    Success-rate rows keep each cold start above its warm start and are
    ordered once, at ε_X = 1%. The top row is the unweighted mean of those
    rates. A second view shows the warm-start improvement in percentage
    points. Problem cells are unlabelled. The overall column shows each
    row's pooled rate. Grey means a completed run is absent.

    The caller checks completed repetitions and exclusions. ``hdb-plot
    interactive`` does that once before drawing this figure.

    Parameters
    ----------
    bundle : ExperimentBundle
        Experiments already limited to the runs that should appear.
    out_dir : Path
        Directory for ``success_heatmap.html``.
    downloads : bool
        Show PNG, SVG, and table buttons. ``False`` leaves them out.

    Returns
    -------
    list of Path
        The written HTML file.

    Raises
    ------
    ValueError
        plotly is not installed, or the page HTML cannot be assembled.
    """
    grouped = _groups(bundle)
    _warn_pooled_cells(grouped)
    payload = _payload(bundle, margin_percents=margins)
    figure = _figure(payload)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "success_heatmap.html"
    html = figure.to_html(
        full_html=True,
        include_plotlyjs=True,
        div_id="success-heatmap",
    )
    path.write_text(
        _inject_margin_menu(html, payload, downloads=downloads),
        encoding="utf-8",
    )
    return [path]
