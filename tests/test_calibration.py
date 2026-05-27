"""Tests for color-card based calibration helpers."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from core.calibration import DEFAULT_COLOR_CARD_REFERENCE, calibrate_image_with_color_card, pixel_area_to_cm2, pixels_to_cm


def test_default_color_card_reference_matches_small_24_patch_card() -> None:
    """Default reference should match the 109 x 63.5 mm small 24-patch card."""

    assert DEFAULT_COLOR_CARD_REFERENCE.card_width_mm == 63.5
    assert DEFAULT_COLOR_CARD_REFERENCE.card_height_mm == 109.0
    assert DEFAULT_COLOR_CARD_REFERENCE.patch_width_mm == 15.0
    assert DEFAULT_COLOR_CARD_REFERENCE.patch_height_mm == 15.0


def test_calibrate_image_with_color_card_detects_synthetic_card() -> None:
    """A synthetic 24-patch card in the expected ROI should be detected and calibrated."""

    reference = DEFAULT_COLOR_CARD_REFERENCE
    card = np.zeros((reference.canonical_height, reference.canonical_width, 3), dtype=np.uint8)

    patch_height = reference.canonical_height // reference.rows
    patch_width = reference.canonical_width // reference.cols
    patch_rgb = (reference.patch_rgb * np.array([1.12, 0.88, 1.05], dtype=np.float32)).clip(0, 255).astype(np.uint8)

    index = 0
    for row_index in range(reference.rows):
        for col_index in range(reference.cols):
            y0 = row_index * patch_height + 5
            y1 = (row_index + 1) * patch_height - 5
            x0 = col_index * patch_width + 5
            x1 = (col_index + 1) * patch_width - 5
            rgb = patch_rgb[index]
            card[y0:y1, x0:x1] = rgb[::-1]
            index += 1

    canvas = np.zeros((900, 1400, 3), dtype=np.uint8)
    canvas[420:660, 920:1280] = card

    result = calibrate_image_with_color_card(canvas, view_name="TOP")

    assert result.is_calibrated is True
    assert result.card_corners is not None
    assert result.mm_per_pixel is not None
    assert 0.2 <= result.mm_per_pixel <= 0.4
    assert result.corrected_card is not None
    assert result.mean_patch_error is not None
    assert result.mean_patch_error < 1200.0


def test_scale_conversion_helpers_return_expected_units() -> None:
    """Pixel-to-physical conversion helpers should return expected values."""

    assert pixels_to_cm(100, 0.2) == 2.0
    assert pixel_area_to_cm2(2500, 0.2) == 1.0
