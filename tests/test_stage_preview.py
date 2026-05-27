from __future__ import annotations

from pathlib import Path

import pytest

from core.grouping import PlantImageGroup
from gui.stage_preview import PreviewViewState
from gui.stage_preview import StagePreviewPayload
from gui.stage_preview import available_views_for_group
from gui.stage_preview import VIEW_SEQUENCE
from gui.stage_preview import pick_active_view
from gui.stage_preview import step_view


def test_available_views_for_complete_group_returns_all_views() -> None:
    group = PlantImageGroup(
        sample_id="sample-1",
        top_image=Path("top.jpg"),
        front_0_image=Path("front_0.jpg"),
        front_180_image=Path("front_180.jpg"),
    )

    assert available_views_for_group(group) == ("TOP", "FRONT-1", "FRONT-2")


def test_available_views_for_missing_front_views_returns_existing_views_only() -> None:
    group = PlantImageGroup(sample_id="sample-2", top_image=Path("top.jpg"))

    assert available_views_for_group(group) == ("TOP",)


def test_pick_active_view_keeps_existing_preferred_view() -> None:
    group = PlantImageGroup(
        sample_id="sample-3",
        top_image=Path("top.jpg"),
        front_0_image=Path("front_0.jpg"),
    )

    active_view, fallback_used = pick_active_view(group, "FRONT-1")

    assert active_view == "FRONT-1"
    assert fallback_used is False


def test_pick_active_view_falls_back_when_preferred_view_missing() -> None:
    group = PlantImageGroup(sample_id="sample-4", top_image=Path("top.jpg"))

    active_view, fallback_used = pick_active_view(group, "FRONT-1")

    assert active_view == "TOP"
    assert fallback_used is True


def test_step_view_cycles_with_partial_view_list() -> None:
    available_views = ("TOP", "FRONT-2")

    assert step_view("TOP", available_views, 1) == "FRONT-2"
    assert step_view("FRONT-2", available_views, 1) == "TOP"
    assert step_view("FRONT-2", available_views, -1) == "TOP"
    assert step_view("UNKNOWN", available_views, 1) == "TOP"


def test_step_view_uses_view_sequence_order_for_available_views() -> None:
    available_views = ("FRONT-2", "FRONT-1", "TOP")

    assert step_view("FRONT-1", available_views, 1) == "FRONT-2"
    assert step_view("FRONT-2", available_views, -1) == "FRONT-1"


def test_preview_view_state_exposes_expected_fields() -> None:
    state = PreviewViewState(
        sample_id="sample-5",
        mode="phenotype",
        view_name="TOP",
        available_views=VIEW_SEQUENCE,
        fallback_notice="切换到首个可用视角",
    )

    assert state.sample_id == "sample-5"
    assert state.mode == "phenotype"
    assert state.view_name == "TOP"
    assert state.available_views == VIEW_SEQUENCE
    assert state.fallback_notice == "切换到首个可用视角"


def test_stage_preview_payload_exposes_expected_fields() -> None:
    payload = StagePreviewPayload(
        placeholder_text="等待表型提取",
        status_text="TOP | 等待表型提取",
    )

    assert payload.image_path is None
    assert payload.image_array is None
    assert payload.placeholder_text == "等待表型提取"
    assert payload.status_text == "TOP | 等待表型提取"
