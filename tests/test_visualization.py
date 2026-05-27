"""Tests for visualization helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("matplotlib")

from core.visualization import build_analysis_debug_previews, create_image_montage


def test_create_image_montage_returns_bgr_canvas() -> None:
    """Montage generation should produce a stitched color image."""
    image_a = np.zeros((20, 10, 3), dtype=np.uint8)
    image_b = np.zeros((20, 10), dtype=np.uint8)

    montage = create_image_montage([image_a, image_b], columns=2)

    assert montage.shape == (20, 20, 3)
    assert montage.dtype == np.uint8


def test_create_image_montage_rejects_mismatched_titles() -> None:
    """Titles should align one-to-one with images."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        create_image_montage([image], titles=["Only", "Extra"])


def test_build_analysis_debug_previews_returns_expected_images() -> None:
    """GUI debug previews should include TOP tiles and FRONT montage tiles."""

    top_image = np.full((30, 40, 3), 120, dtype=np.uint8)
    top_mask = np.zeros((30, 40), dtype=np.uint8)
    top_mask[5:25, 10:30] = 255
    top_segmentation = SimpleNamespace(
        has_foreground=True,
        mask=top_mask,
        contour_image=np.full((30, 40, 3), 80, dtype=np.uint8),
        hull_image=np.full((30, 40, 3), 160, dtype=np.uint8),
        debug_images={"original": top_image},
    )

    front_mask = np.zeros((24, 32), dtype=np.uint8)
    front_mask[4:20, 8:24] = 255
    front_segmentation = SimpleNamespace(
        has_foreground=True,
        mask=front_mask,
        contour_image=np.full((24, 32, 3), 200, dtype=np.uint8),
        debug_images={"original": np.full((24, 32, 3), 90, dtype=np.uint8)},
    )

    result = SimpleNamespace(
        top_segmentation=top_segmentation,
        front_segmentations={"FRONT-1": front_segmentation},
        calibration_results={},
    )

    previews = build_analysis_debug_previews(result, tile_size=(60, 40))

    assert previews is not None
    assert previews.mask_image.shape == (30, 40, 3)
    assert previews.contour_image.shape == (30, 40, 3)
    assert previews.hull_image.shape == (30, 40, 3)
    assert previews.montage_image.shape == (160, 120, 3)
    assert previews.montage_titles[:4] == (
        "TOP Corrected",
        "TOP Mask",
        "TOP Contour",
        "TOP Hull",
    )


def test_build_analysis_debug_previews_returns_none_without_top_segmentation() -> None:
    """A missing TOP segmentation should suppress GUI debug previews."""

    result = SimpleNamespace(top_segmentation=None, front_segmentations={}, calibration_results={})

    assert build_analysis_debug_previews(result) is None
