"""Closed metadata contract for scalar landscapes supported by catalogue LON."""
from __future__ import annotations

from dataclasses import dataclass

from core.ioh_core.workflows.la.models import (
    ElaError,
    LANDSCAPE_Y_P,
    LANDSCAPE_Y_RAW,
)


@dataclass(frozen=True)
class LonLandscapeSpec:
    """Storage and publication names for one supported LON landscape."""

    token: str
    source_column: str
    global_feature: str
    global_marker_column: str

    @property
    def source_value_column(self) -> str:
        return f"source_{self.token}"

    @property
    def target_value_column(self) -> str:
        return f"target_{self.token}"


_SPECS = {
    LANDSCAPE_Y_P: LonLandscapeSpec(
        token=LANDSCAPE_Y_P,
        source_column="y_p",
        global_feature="lon.y_p_star",
        global_marker_column="is_y_p_star",
    ),
    LANDSCAPE_Y_RAW: LonLandscapeSpec(
        token=LANDSCAPE_Y_RAW,
        source_column="y",
        global_feature="lon.y_box_star",
        global_marker_column="is_y_box_star",
    ),
}
LON_LANDSCAPES = frozenset(_SPECS)


def get_lon_landscape(raw: object) -> LonLandscapeSpec:
    """Return supported landscape metadata or reject aliases and unknown tokens."""
    token = str(raw)
    try:
        return _SPECS[token]
    except KeyError as exc:
        raise ElaError(f"LON landscape must be y_p or y_raw, got {raw!r}") from exc
