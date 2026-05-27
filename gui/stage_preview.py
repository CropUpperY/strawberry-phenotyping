from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.grouping import PlantImageGroup

VIEW_SEQUENCE = ("TOP", "FRONT-1", "FRONT-2")


@dataclass(frozen=True, slots=True)
class PreviewViewState:
    sample_id: str | None
    mode: str
    view_name: str
    available_views: tuple[str, ...]
    fallback_notice: str = ""


@dataclass(frozen=True, slots=True)
class StagePreviewPayload:
    image_path: Path | None = None
    image_array: np.ndarray | None = None
    placeholder_text: str = ""
    status_text: str = ""


def available_views_for_group(group: PlantImageGroup | None) -> tuple[str, ...]:
    if group is None:
        return ("TOP",)

    views: list[str] = []
    if group.top_image is not None:
        views.append("TOP")
    if group.front_0_image is not None:
        views.append("FRONT-1")
    if group.front_180_image is not None:
        views.append("FRONT-2")
    return tuple(views) if views else ("TOP",)


def pick_active_view(group: PlantImageGroup | None, preferred_view: str | None) -> tuple[str, bool]:
    available_views = _ordered_available_views(available_views_for_group(group))
    if preferred_view in available_views:
        return preferred_view, False
    if preferred_view is None:
        return available_views[0], False
    return available_views[0], True


def step_view(current_view: str, available_views: tuple[str, ...], direction: int) -> str:
    available_views = _ordered_available_views(available_views)
    if not available_views:
        return "TOP"
    if current_view not in available_views:
        return available_views[0]

    current_index = available_views.index(current_view)
    next_index = (current_index + direction) % len(available_views)
    return available_views[next_index]


def _ordered_available_views(available_views: tuple[str, ...]) -> tuple[str, ...]:
    ordered_views = tuple(view for view in VIEW_SEQUENCE if view in available_views)
    return ordered_views or ("TOP",)
