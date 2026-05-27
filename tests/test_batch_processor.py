"""Tests for directory-level batch analysis helpers."""

from pathlib import Path

import numpy as np

from core.batch_processor import analyze_directory


class FakeTopSegmentationResult:
    """Simple TOP segmentation stub."""

    def __init__(self, height: int = 16, width: int = 16) -> None:
        self.has_foreground = True
        self.mask = np.ones((height, width), dtype=np.uint8) * 255
        self.mask_area_pixels = height * width
        self.contour_count = 1
        self.hull_area_pixels = float(height * width)
        self.contour_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.hull_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.convex_hull = np.array([[[0, 0]], [[width - 1, 0]], [[width - 1, height - 1]]], dtype=np.int32)
        self.debug_images = {
            "normalized": np.zeros((height, width, 3), dtype=np.uint8),
            "denoised": np.zeros((height, width, 3), dtype=np.uint8),
            "hsv_green_mask": np.zeros((height, width), dtype=np.uint8),
            "green_dominance_mask": np.zeros((height, width), dtype=np.uint8),
            "combined_mask": np.zeros((height, width), dtype=np.uint8),
            "cleaned_mask": np.zeros((height, width), dtype=np.uint8),
            "filtered_mask": np.ones((height, width), dtype=np.uint8) * 255,
        }


class FakeFrontSegmentationResult:
    """Simple FRONT segmentation stub."""

    def __init__(self, height: int, width: int, area: int) -> None:
        self.has_foreground = True
        self.mask = np.ones((height + 8, width + 8), dtype=np.uint8) * 255
        self.mask_area_pixels = area
        self.contour_count = 1
        self.bounding_box = (5, 6, width, height)
        self.contour_image = np.zeros((height + 8, width + 8, 3), dtype=np.uint8)
        self.bounding_box_image = np.zeros((height + 8, width + 8, 3), dtype=np.uint8)
        self.debug_images = {
            "filtered_mask": np.ones((height + 8, width + 8), dtype=np.uint8) * 255,
        }


def test_analyze_directory_handles_complete_and_incomplete_groups(tmp_path: Path) -> None:
    """Batch analysis should continue across complete and incomplete groups."""

    for file_name in ("1AB_TOP.png", "1AB-1.png", "1AB-2.png", "2AB_TOP.png", "2AB-1.png"):
        (tmp_path / file_name).write_bytes(b"")

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 120
    front_results = iter(
        [
            FakeFrontSegmentationResult(height=30, width=14, area=280),
            FakeFrontSegmentationResult(height=28, width=16, area=300),
        ]
    )

    report = analyze_directory(
        tmp_path,
        image_loader=lambda _: image.copy(),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=16, width=16),
        front_segmenter=lambda _: next(front_results),
        image_calibrator=lambda payload, *, view_name: _fake_calibration(payload, view_name=view_name),
        debug_output_dir=None,
    )

    assert report.total_groups == 2
    assert report.completed_groups == 1
    assert report.skipped_groups == 1
    assert report.failed_groups == 0
    assert report.sample_results[0].result.status == "analysis_complete"
    assert report.sample_results[1].result.status == "incomplete_input"


def _fake_calibration(image: np.ndarray, *, view_name: str) -> object:
    """Return a simple calibration-like object for batch tests."""

    from types import SimpleNamespace

    return SimpleNamespace(
        status="calibrated",
        message=f"{view_name} calibrated",
        view_name=view_name,
        corrected_image=image.copy(),
        mm_per_pixel=0.2,
        pixels_per_mm=5.0,
        debug_images={},
        is_calibrated=True,
    )
