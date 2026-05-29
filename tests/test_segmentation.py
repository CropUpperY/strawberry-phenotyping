"""Tests for strawberry canopy segmentation helpers."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from core.segmentation import segment_front_view_plant, segment_top_view_plant


def test_segment_top_view_plant_returns_mask_contour_and_hull() -> None:
    """A synthetic green canopy on dark background should be segmented."""

    image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(image, (140, 140), 45, (30, 170, 40), -1)
    cv2.circle(image, (220, 120), 38, (20, 180, 50), -1)
    cv2.circle(image, (200, 200), 52, (25, 160, 35), -1)
    cv2.rectangle(image, (350, 30), (356, 36), (30, 200, 40), -1)

    result = segment_top_view_plant(image)

    assert result.status == "segmented"
    assert result.has_foreground is True
    assert result.mask_area_pixels > 0
    assert result.contour_count >= 1
    assert result.convex_hull is not None
    assert result.hull_area_pixels >= result.mask_area_pixels
    assert result.mask.shape == image.shape[:2]
    assert result.contour_image.shape == image.shape
    assert result.hull_image.shape == image.shape


def test_segment_top_view_plant_removes_right_side_card_like_component() -> None:
    """A compact right-side green patch should be removed from TOP-view segmentation."""

    image = np.zeros((320, 420, 3), dtype=np.uint8)
    cv2.circle(image, (150, 150), 55, (30, 170, 40), -1)
    cv2.circle(image, (210, 205), 60, (25, 160, 35), -1)
    cv2.rectangle(image, (340, 140), (386, 188), (35, 195, 45), -1)

    result = segment_top_view_plant(image)

    assert result.status == "segmented"
    assert result.mask[164, 362] == 0
    assert cv2.countNonZero(result.debug_images["removed_right_card_mask"]) > 0


def test_segment_top_view_plant_removes_top_attached_pot_band() -> None:
    """A shallow top band connected to the canopy should be removed as pot rim."""

    image = np.zeros((360, 420, 3), dtype=np.uint8)
    cv2.ellipse(image, (210, 220), (125, 92), 0, 0, 360, (30, 170, 40), -1)
    cv2.rectangle(image, (95, 42), (325, 74), (35, 185, 45), -1)
    cv2.rectangle(image, (95, 74), (126, 148), (35, 185, 45), -1)
    cv2.rectangle(image, (294, 74), (325, 152), (35, 185, 45), -1)

    result = segment_top_view_plant(image)

    assert result.status == "segmented"
    assert result.mask[58, 210] == 0
    assert result.mask[220, 210] == 255
    assert cv2.countNonZero(result.debug_images["removed_top_pot_band"]) > 0
    assert "morphology_opened" in result.debug_images
    assert "morphology_closed" in result.debug_images
    assert "holes_filled_mask" in result.debug_images
    assert "top_band_candidate_mask" in result.debug_images


def test_segment_top_view_plant_preserves_white_flowers_inside_canopy() -> None:
    """TOP display mask should keep white petals and yellow centers near the green canopy."""

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(image, (150, 130), 70, (30, 170, 40), -1)
    cv2.circle(image, (150, 130), 18, (245, 245, 245), -1)
    cv2.circle(image, (150, 130), 6, (20, 215, 245), -1)
    cv2.rectangle(image, (270, 88), (302, 148), (245, 245, 245), -1)

    result = segment_top_view_plant(image)

    assert result.mask[130, 150] == 255
    assert result.leaf_mask is not None
    assert result.leaf_mask[130, 150] == 0
    assert result.mask[116, 150] == 255
    assert result.mask[118, 286] == 0
    assert cv2.countNonZero(result.debug_images["top_reproductive_mask"]) > 0


def test_segment_top_view_plant_cotton_profile_keeps_low_saturation_leaves() -> None:
    """Cotton TOP segmentation should not discard broad gray-green cotton leaves."""

    image = np.zeros((360, 460, 3), dtype=np.uint8)
    cv2.rectangle(image, (155, 230), (272, 270), (165, 190, 175), -1)
    cv2.rectangle(image, (388, 110), (442, 286), (42, 82, 58), -1)
    cv2.circle(image, (185, 145), 52, (118, 142, 116), -1)
    cv2.circle(image, (248, 138), 46, (112, 138, 108), -1)
    cv2.circle(image, (210, 205), 42, (34, 165, 48), -1)

    result = segment_top_view_plant(image, profile="cotton")

    assert result.status == "segmented"
    assert result.leaf_mask is not None
    assert result.leaf_mask[145, 185] == 255
    assert result.leaf_mask[138, 248] == 255
    assert result.leaf_mask[205, 210] == 255
    assert result.mask[145, 185] == 255
    assert result.mask[138, 248] == 255
    assert result.mask[205, 210] == 255
    assert result.leaf_mask[260, 210] == 0
    assert result.mask[198, 420] == 0
    assert cv2.countNonZero(result.debug_images["cotton_soil_removed_mask"]) > 0
    assert cv2.countNonZero(result.debug_images["cotton_far_component_removed_mask"]) > 0
    x, _y, width, _height = result.bounding_box
    assert x + width < 320


def test_segment_top_view_plant_cotton_profile_removes_soil_color_without_refill() -> None:
    """Cotton TOP segmentation should remove pale soil colors and not add them back."""

    image = np.zeros((280, 360, 3), dtype=np.uint8)
    cv2.circle(image, (140, 110), 50, (118, 142, 116), -1)
    cv2.circle(image, (220, 110), 48, (112, 138, 108), -1)
    cv2.circle(image, (180, 185), 46, (34, 165, 48), -1)
    cv2.rectangle(image, (150, 132), (220, 182), (165, 190, 175), -1)

    result = segment_top_view_plant(image, profile="cotton")

    assert result.status == "segmented"
    assert result.leaf_mask is not None
    assert result.leaf_mask[110, 140] == 255
    assert result.leaf_mask[110, 220] == 255
    assert result.leaf_mask[185, 180] == 255
    assert result.leaf_mask[156, 185] == 0
    assert result.mask[156, 185] == 0
    assert cv2.countNonZero(result.debug_images["cotton_soil_removed_mask"]) > 0
    assert cv2.countNonZero(result.debug_images["top_reproductive_mask"]) == 0


def test_segment_top_view_plant_handles_empty_foreground() -> None:
    """A dark background with no canopy should report no foreground."""

    image = np.zeros((200, 300, 3), dtype=np.uint8)
    result = segment_top_view_plant(image)

    assert result.status == "no_foreground"
    assert result.has_foreground is False
    assert result.mask_area_pixels == 0


def test_segment_front_view_plant_returns_mask_and_bounding_box() -> None:
    """A synthetic side-view canopy should yield a valid bounding box."""

    image = np.zeros((280, 360, 3), dtype=np.uint8)
    cv2.ellipse(image, (150, 150), (55, 80), 0, 0, 360, (30, 170, 40), -1)
    cv2.ellipse(image, (220, 145), (48, 72), 0, 0, 360, (20, 180, 50), -1)
    cv2.rectangle(image, (330, 60), (340, 100), (40, 190, 40), -1)

    result = segment_front_view_plant(image)

    assert result.status == "segmented"
    assert result.has_foreground is True
    assert result.mask_area_pixels > 0
    assert result.bounding_box is not None
    assert result.contour_count >= 1
    assert result.mask.shape == image.shape[:2]
    assert result.contour_image.shape == image.shape
    assert result.bounding_box_image.shape == image.shape
    assert "green_dominance_mask" in result.debug_images
    assert "combined_mask" in result.debug_images
    assert "morphology_opened" in result.debug_images
    assert "morphology_closed" in result.debug_images
    assert "holes_filled_mask" in result.debug_images
    assert "filtered_mask_before_front_augmentation" in result.debug_images
    assert "front_weak_green_mask" in result.debug_images
    assert "front_weak_green_seed_mask" in result.debug_images
    assert "front_recovered_weak_green_mask" in result.debug_images
    assert "filtered_mask_before_front_rules" in result.debug_images
    assert "front_component_kept_mask" in result.debug_images
    assert "front_component_removed_mask" in result.debug_images

    x, y, width, height = result.bounding_box
    assert width > 50
    assert height > 100
    assert x < 300


def test_segment_front_view_plant_recovers_adjacent_shadow_leaf_regions() -> None:
    """FRONT-view segmentation should recover nearby low-saturation leaf regions."""

    image = np.zeros((280, 360, 3), dtype=np.uint8)
    cv2.ellipse(image, (140, 150), (55, 80), 0, 0, 360, (30, 170, 40), -1)
    cv2.ellipse(image, (220, 150), (45, 72), 0, 0, 360, (80, 90, 90), -1)

    result = segment_front_view_plant(image)

    assert result.status == "segmented"
    assert result.mask[150, 220] == 255
    assert result.mask_area_pixels > 20000
    assert result.bounding_box is not None
    assert cv2.countNonZero(result.debug_images["front_recovered_weak_green_mask"]) > 0
    _, _, width, _ = result.bounding_box
    assert width > 150


def test_segment_front_view_plant_fills_reproductive_holes_near_canopy() -> None:
    """FRONT display mask should keep flower/fruit pixels that sit inside the plant region."""

    image = np.zeros((280, 360, 3), dtype=np.uint8)
    cv2.ellipse(image, (165, 155), (72, 84), 0, 0, 360, (30, 170, 40), -1)
    cv2.circle(image, (165, 155), 24, (235, 240, 210), -1)
    cv2.circle(image, (165, 155), 8, (20, 210, 240), -1)
    cv2.circle(image, (165, 155), 3, (12, 18, 16), -1)

    result = segment_front_view_plant(image)

    assert result.status == "segmented"
    assert result.mask[155, 165] == 255
    assert result.mask[148, 165] == 255
    assert cv2.countNonZero(result.debug_images["front_reproductive_mask"]) > 0


def test_segment_front_view_plant_removes_teal_pot_component() -> None:
    """A bottom teal pot component should be removed even when its hue overlaps leaf green."""

    image = np.zeros((320, 420, 3), dtype=np.uint8)
    cv2.ellipse(image, (205, 120), (92, 72), 0, 0, 360, (28, 175, 42), -1)
    cv2.rectangle(image, (60, 228), (360, 286), (92, 118, 86), -1)

    result = segment_front_view_plant(image)

    assert result.status == "segmented"
    assert result.mask[120, 205] == 255
    assert result.mask[258, 210] == 0
    assert cv2.countNonZero(result.debug_images["front_pot_like_removed_mask"]) > 0
