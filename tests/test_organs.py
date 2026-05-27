"""Tests for TOP-view flower and fruit detection helpers."""

import cv2
import numpy as np

from core.organs import _assemble_flower_instances
from core.organs import _FlowerSeed
from core.organs import detect_top_flowers, detect_top_fruits


def _build_top_scene() -> tuple[np.ndarray, np.ndarray]:
    """Create a simple synthetic canopy scene for organ-detection tests."""

    image = np.zeros((160, 160, 3), dtype=np.uint8)
    image[:, :] = (30, 120, 30)
    canopy_mask = np.zeros((160, 160), dtype=np.uint8)
    cv2.circle(canopy_mask, (80, 80), 58, 255, -1)
    return image, canopy_mask


def test_detect_top_flowers_counts_visible_blooms_with_yellow_centers() -> None:
    """Open blooms with a yellow center and white petals should be counted."""

    image, canopy_mask = _build_top_scene()
    cv2.circle(image, (60, 70), 8, (245, 245, 245), -1)
    cv2.circle(image, (96, 88), 9, (250, 250, 250), -1)
    cv2.circle(image, (60, 70), 4, (20, 215, 245), -1)
    cv2.circle(image, (96, 88), 4, (20, 215, 245), -1)

    result = detect_top_flowers(image, canopy_mask)

    assert result.status == "computed"
    assert result.count == 2
    assert len(result.instances) == 2
    assert result.mask.dtype == np.uint8


def test_detect_top_fruits_counts_visible_red_regions() -> None:
    """Red fruit-like blobs inside the canopy should be counted."""

    image, canopy_mask = _build_top_scene()
    cv2.circle(image, (68, 64), 9, (30, 30, 220), -1)
    cv2.circle(image, (92, 96), 10, (20, 40, 235), -1)

    result = detect_top_fruits(image, canopy_mask)

    assert result.status == "computed"
    assert result.count == 2
    assert len(result.instances) == 2


def test_detect_top_organs_ignores_noise_and_outside_canopy() -> None:
    """Tiny noise and objects outside the canopy should not be counted."""

    image, canopy_mask = _build_top_scene()
    cv2.circle(image, (20, 20), 9, (250, 250, 250), -1)
    cv2.circle(image, (75, 78), 1, (250, 250, 250), -1)
    cv2.circle(image, (88, 82), 8, (250, 250, 250), -1)
    cv2.circle(image, (88, 82), 4, (20, 215, 245), -1)

    flower_result = detect_top_flowers(image, canopy_mask)

    assert flower_result.count == 1


def test_detect_top_flowers_ignores_bright_leaf_highlights() -> None:
    """Bright desaturated canopy highlights should not be mistaken for flowers."""

    image, canopy_mask = _build_top_scene()
    cv2.circle(image, (60, 70), 9, (180, 212, 196), -1)
    cv2.circle(image, (95, 85), 9, (170, 205, 185), -1)
    cv2.circle(image, (85, 105), 9, (190, 220, 200), -1)

    result = detect_top_flowers(image, canopy_mask)

    assert result.count == 0
    assert len(result.instances) == 0


def test_detect_top_flowers_requires_yellow_center_for_bright_white_regions() -> None:
    """Bright white canopy regions without a yellow center should not be counted as flowers."""

    image, canopy_mask = _build_top_scene()
    cv2.ellipse(image, (62, 72), (10, 7), 15, 0, 360, (248, 248, 248), -1)
    cv2.ellipse(image, (96, 94), (11, 8), -20, 0, 360, (250, 250, 250), -1)

    result = detect_top_flowers(image, canopy_mask)

    assert result.count == 0
    assert len(result.instances) == 0


def test_detect_top_flowers_merges_split_petals_around_one_center() -> None:
    """One flower center with multiple separated petal blobs should still count as one flower."""

    image, canopy_mask = _build_top_scene()
    cv2.circle(image, (80, 80), 4, (20, 215, 245), -1)
    cv2.circle(image, (68, 80), 7, (248, 248, 248), -1)
    cv2.circle(image, (92, 80), 7, (248, 248, 248), -1)
    cv2.circle(image, (80, 68), 6, (248, 248, 248), -1)

    result = detect_top_flowers(image, canopy_mask)

    assert result.count == 1
    assert len(result.instances) == 1


def test_detect_top_flowers_splits_two_blooms_when_two_centers_share_petals() -> None:
    """Two flower centers inside one shared white bloom should count as two flowers."""

    image, canopy_mask = _build_top_scene()
    cv2.ellipse(image, (80, 80), (22, 11), 0, 0, 360, (248, 248, 248), -1)
    cv2.circle(image, (72, 80), 4, (20, 215, 245), -1)
    cv2.circle(image, (88, 80), 4, (20, 215, 245), -1)

    result = detect_top_flowers(image, canopy_mask)

    assert result.count == 2
    assert len(result.instances) == 2


def test_detect_top_flowers_handles_many_seed_markers_without_uint8_overflow() -> None:
    """Flower debug marker rendering should not overflow when many seeds are present."""

    petal_mask = np.zeros((120, 120), dtype=np.uint8)
    seeds: list[_FlowerSeed] = []
    positions = [
        (16, 16), (40, 16), (64, 16), (88, 16),
        (16, 48), (40, 48), (64, 48), (88, 48),
    ]
    for index, (center_x, center_y) in enumerate(positions, start=1):
        component_mask = np.zeros_like(petal_mask)
        cv2.circle(component_mask, (center_x, center_y), 2, 255, -1)
        cv2.circle(petal_mask, (center_x, center_y), 6, 255, -1)
        seeds.append(
            _FlowerSeed(
                label_id=index,
                area_pixels=16,
                centroid_xy=(float(center_x), float(center_y)),
                binary_mask=component_mask,
                search_radius=12,
            )
        )

    labeled_mask, kept_mask, instances, support_union, marker_mask = _assemble_flower_instances(petal_mask, seeds)

    assert kept_mask.dtype == np.uint8
    assert support_union.dtype == np.uint8
    assert marker_mask.dtype == np.uint8
    assert labeled_mask.shape == petal_mask.shape


def test_detect_top_fruits_splits_touching_regions() -> None:
    """Touching red blobs should be split into separate fruit instances."""

    image, canopy_mask = _build_top_scene()
    cv2.circle(image, (74, 80), 12, (20, 30, 230), -1)
    cv2.circle(image, (92, 80), 12, (20, 30, 230), -1)

    result = detect_top_fruits(image, canopy_mask)

    assert result.count == 2
    assert len(result.instances) == 2


def test_detect_top_fruits_ignores_thin_red_fragments() -> None:
    """Thin red streaks inside the canopy should not be counted as fruits."""

    image, canopy_mask = _build_top_scene()
    cv2.rectangle(image, (48, 70), (88, 74), (20, 30, 220), -1)
    cv2.rectangle(image, (92, 92), (97, 122), (20, 30, 220), -1)
    cv2.rectangle(image, (105, 52), (134, 57), (20, 30, 220), -1)

    result = detect_top_fruits(image, canopy_mask)

    assert result.count == 0
    assert len(result.instances) == 0


def test_detect_top_fruits_ignores_hollow_red_rings() -> None:
    """Hollow red rings should not be counted as solid fruit instances."""

    image, canopy_mask = _build_top_scene()
    cv2.circle(image, (82, 78), 10, (20, 30, 220), -1)
    cv2.circle(image, (82, 78), 6, (30, 120, 30), -1)

    result = detect_top_fruits(image, canopy_mask)

    assert result.count == 0
    assert len(result.instances) == 0
