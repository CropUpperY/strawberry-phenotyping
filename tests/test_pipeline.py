"""Tests for the grouped analysis pipeline."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.grouping import PlantImageGroup
from core.pipeline import _build_top_organ_filter_mask, analyze_plant_group, create_result_rows


class FakeTopSegmentationResult:
    """Simple stand-in for the TOP segmentation payload."""

    def __init__(self, height: int = 32, width: int = 48) -> None:
        self.has_foreground = True
        self.mask = np.ones((height, width), dtype=np.uint8) * 255
        self.mask_area_pixels = height * width
        self.contour_count = 2
        self.hull_area_pixels = 1820.5
        self.contour_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.hull_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.convex_hull = np.array([[[0, 0]], [[10, 0]], [[10, 10]]], dtype=np.int32)
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
    """Simple stand-in for one FRONT segmentation payload."""

    def __init__(self, height: int, width: int, mask_area: int) -> None:
        self.has_foreground = True
        self.mask = np.ones((height + 10, width + 10), dtype=np.uint8) * 255
        self.mask_area_pixels = mask_area
        self.contour_count = 1
        self.bounding_box = (10, 15, width, height)
        self.contour_image = np.zeros((height + 10, width + 10, 3), dtype=np.uint8)
        self.bounding_box_image = np.zeros((height + 10, width + 10, 3), dtype=np.uint8)
        self.debug_images = {
            "filtered_mask": np.ones((height + 10, width + 10), dtype=np.uint8) * 255,
        }


class FakeOrganDetectionResult:
    """Simple stand-in for flower/fruit detection payloads."""

    def __init__(self, count: int, height: int, width: int, *, mask: np.ndarray | None = None) -> None:
        self.status = "computed"
        self.message = "ok"
        self.count = count
        self.mask = mask.copy() if mask is not None else np.zeros((height, width), dtype=np.uint8)
        self.labeled_mask = np.zeros((height, width), dtype=np.int32)
        self.instances = []
        self.overlay_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.debug_images = {
            "canopy_source": np.zeros((height, width, 3), dtype=np.uint8),
            "raw_mask": np.zeros((height, width), dtype=np.uint8),
            "cleaned_mask": np.zeros((height, width), dtype=np.uint8),
            "distance_map": np.zeros((height, width), dtype=np.uint8),
            "labeled_mask": np.zeros((height, width), dtype=np.uint8),
            "overlay": np.zeros((height, width, 3), dtype=np.uint8),
        }


class FakeYoloOrganDetectionResult:
    """Simple stand-in for combined YOLO organ detection payloads."""

    def __init__(self, *, flower_count: int, flower_bud_count: int, fruit_count: int, height: int, width: int) -> None:
        self.status = "computed"
        self.message = "ok"
        self.flower_count = flower_count
        self.flower_bud_count = flower_bud_count
        self.fruit_count = fruit_count
        self.counts = {
            "flower": flower_count,
            "flower_bud": flower_bud_count,
            "fruit": fruit_count,
        }
        self.overlay_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.debug_images = {
            "overlay": self.overlay_image,
        }


def _build_fake_calibration(image: np.ndarray, *, view_name: str, mm_per_pixel: float | None) -> SimpleNamespace:
    """Build a calibration-like payload for pipeline tests."""

    debug_images = {
        "search_region_overlay": image.copy(),
        "detection_mask": np.zeros(image.shape[:2], dtype=np.uint8),
        "card_overlay": image.copy(),
        "warped_card": np.zeros((60, 40, 3), dtype=np.uint8),
        "corrected_card": np.zeros((60, 40, 3), dtype=np.uint8),
        "before_after": np.hstack([image, image]),
    }
    return SimpleNamespace(
        status="calibrated" if mm_per_pixel is not None else "not_detected",
        message=f"{view_name} calibration stub",
        view_name=view_name,
        corrected_image=image.copy(),
        corrected_card=debug_images["corrected_card"],
        warped_card=debug_images["warped_card"],
        card_corners=np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype=np.float32),
        observed_patch_rgb=np.zeros((24, 3), dtype=np.float32),
        reference_patch_rgb=np.zeros((24, 3), dtype=np.float32),
        correction_matrix=np.zeros((4, 3), dtype=np.float32),
        mean_patch_error=0.0,
        mm_per_pixel=mm_per_pixel,
        pixels_per_mm=(1.0 / mm_per_pixel) if mm_per_pixel is not None else None,
        card_width_pixels=10.0,
        card_height_pixels=20.0,
        search_region=(0, 0, image.shape[1], image.shape[0]),
        debug_images=debug_images,
        is_calibrated=mm_per_pixel is not None,
    )


def test_analyze_plant_group_computes_calibrated_traits() -> None:
    """A complete group should produce calibrated TOP and FRONT trait values."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )

    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, :, 1] = 120
    front_results = iter(
        [
            FakeFrontSegmentationResult(height=90, width=45, mask_area=2400),
            FakeFrontSegmentationResult(height=84, width=52, mask_area=2600),
        ]
    )

    def calibrator(payload: np.ndarray, *, view_name: str) -> SimpleNamespace:
        return _build_fake_calibration(payload, view_name=view_name, mm_per_pixel=0.2)

    result = analyze_plant_group(
        group,
        emit_log=lambda _: None,
        image_loader=lambda _: image.copy(),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=image.shape[0], width=image.shape[1]),
        front_segmenter=lambda _: next(front_results),
        image_calibrator=calibrator,
        debug_output_dir=None,
    )

    trait_map = result.trait_map()

    assert result.status == "analysis_complete"
    assert trait_map["leaf_area"].unit == "cm^2"
    assert trait_map["leaf_area"].value == 0.61
    assert trait_map["convex_hull_area"].unit == "cm^2"
    assert trait_map["canopy_height"].unit == "cm"
    assert trait_map["canopy_height"].value == 1.8
    assert trait_map["canopy_width"].value == 0.28
    assert trait_map["side_projection_area"].unit == "cm^2"
    assert trait_map["side_projection_area"].value == 1.0


def test_analyze_plant_group_falls_back_to_pixel_units_when_calibration_missing() -> None:
    """Failed calibration should not stop analysis, but units should stay in pixels."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )

    image = np.full((8, 8, 3), (10, 100, 10), dtype=np.uint8)
    front_results = iter(
        [
            FakeFrontSegmentationResult(height=40, width=20, mask_area=500),
            FakeFrontSegmentationResult(height=45, width=22, mask_area=550),
        ]
    )

    def calibrator(payload: np.ndarray, *, view_name: str) -> SimpleNamespace:
        return _build_fake_calibration(payload, view_name=view_name, mm_per_pixel=None)

    result = analyze_plant_group(
        group,
        image_loader=lambda _: image.copy(),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=image.shape[0], width=image.shape[1]),
        front_segmenter=lambda _: next(front_results),
        image_calibrator=calibrator,
        debug_output_dir=None,
    )

    trait_map = result.trait_map()

    assert result.status == "analysis_complete"
    assert trait_map["leaf_area"].unit == "px^2"
    assert trait_map["canopy_height"].unit == "px"
    assert trait_map["canopy_width"].unit == "px"
    assert trait_map["side_projection_area"].unit == "px^2"


def test_analyze_plant_group_detects_incomplete_input() -> None:
    """Missing views should stop the pipeline before image loading."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=None,
    )

    result = analyze_plant_group(
        group,
        image_loader=lambda _: np.zeros((1, 1, 3), dtype=np.uint8),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=1, width=1),
        front_segmenter=lambda _: FakeFrontSegmentationResult(height=1, width=1, mask_area=1),
        image_calibrator=lambda payload, *, view_name: _build_fake_calibration(payload, view_name=view_name, mm_per_pixel=0.2),
    )

    assert result.status == "incomplete_input"
    assert result.errors
    assert result.view_results["FRONT-2"].status == "missing"


def test_create_result_rows_reflects_trait_results() -> None:
    """UI rows should match the current pipeline result payload."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )

    image = np.full((8, 8, 3), (10, 100, 10), dtype=np.uint8)
    front_results = iter(
        [
            FakeFrontSegmentationResult(height=40, width=20, mask_area=500),
            FakeFrontSegmentationResult(height=45, width=22, mask_area=550),
        ]
    )

    result = analyze_plant_group(
        group,
        image_loader=lambda _: image.copy(),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=image.shape[0], width=image.shape[1]),
        front_segmenter=lambda _: next(front_results),
        image_calibrator=lambda payload, *, view_name: _build_fake_calibration(payload, view_name=view_name, mm_per_pixel=0.2),
        debug_output_dir=None,
    )
    rows = create_result_rows(result)

    assert len(rows) == len(result.traits)
    assert rows[0][0] == "叶面积"
    assert rows[0][3] == "cm^2"
    assert rows[0][4] == "已计算"
    labels = [row[0] for row in rows]
    assert "花朵数" in labels
    assert "花骨朵数" in labels
    assert "果实数" in labels


def test_analyze_plant_group_exports_debug_artifacts(tmp_path: Path) -> None:
    """Debug mode should export calibration and trait intermediate images."""

    pytest.importorskip("cv2")

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )

    image = np.full((16, 20, 3), (20, 150, 20), dtype=np.uint8)
    front_results = iter(
        [
            FakeFrontSegmentationResult(height=24, width=12, mask_area=180),
            FakeFrontSegmentationResult(height=26, width=14, mask_area=220),
        ]
    )

    result = analyze_plant_group(
        group,
        image_loader=lambda _: image.copy(),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=image.shape[0], width=image.shape[1]),
        front_segmenter=lambda _: next(front_results),
        image_calibrator=lambda payload, *, view_name: _build_fake_calibration(payload, view_name=view_name, mm_per_pixel=0.2),
        debug_output_dir=tmp_path,
    )

    assert result.status == "analysis_complete"
    assert set(result.debug_artifact_paths) == {
        "color_calibration",
        "top_segmentation",
        "front_segmentation",
        "leaf_area",
        "convex_hull_area",
        "greenness",
        "canopy_height",
        "canopy_width",
        "side_projection_area",
        "flower_count",
        "flower_bud_count",
        "fruit_count",
    }
    assert all(paths for key, paths in result.debug_artifact_paths.items() if key != "flower_bud_count")
    assert result.debug_artifact_paths["color_calibration"][0].exists()
    assert len(result.debug_artifact_paths["front_segmentation"]) >= 12


def test_analyze_plant_group_populates_flower_and_fruit_counts() -> None:
    """TOP organ detectors should populate flower_count and fruit_count traits."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )
    image = np.full((24, 24, 3), (20, 140, 20), dtype=np.uint8)
    front_results = iter(
        [
            FakeFrontSegmentationResult(height=40, width=20, mask_area=500),
            FakeFrontSegmentationResult(height=45, width=22, mask_area=550),
        ]
    )

    result = analyze_plant_group(
        group,
        image_loader=lambda _: image.copy(),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=image.shape[0], width=image.shape[1]),
        front_segmenter=lambda _: next(front_results),
        image_calibrator=lambda payload, *, view_name: _build_fake_calibration(payload, view_name=view_name, mm_per_pixel=0.2),
        top_flower_detector=lambda top_image, canopy_mask: FakeOrganDetectionResult(3, top_image.shape[0], top_image.shape[1]),
        top_fruit_detector=lambda top_image, canopy_mask: FakeOrganDetectionResult(2, top_image.shape[0], top_image.shape[1]),
        debug_output_dir=None,
    )

    trait_map = result.trait_map()

    assert trait_map["flower_count"].value == 3
    assert trait_map["flower_count"].unit == "count"
    assert trait_map["flower_count"].status == "computed"
    assert trait_map["fruit_count"].value == 2
    assert trait_map["fruit_count"].unit == "count"
    assert trait_map["fruit_count"].status == "computed"
    assert trait_map["flower_bud_count"].value is None
    assert trait_map["flower_bud_count"].status == "pending_algorithm"


def test_analyze_plant_group_computes_fruit_area_and_source_sink_ratio() -> None:
    """Fruit masks should produce calibrated fruit_area and source_sink_ratio values."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )
    image = np.full((20, 25, 3), (20, 140, 20), dtype=np.uint8)
    fruit_mask = np.zeros((20, 25), dtype=np.uint8)
    fruit_mask[10:15, 8:18] = 255
    front_results = iter(
        [
            FakeFrontSegmentationResult(height=40, width=20, mask_area=500),
            FakeFrontSegmentationResult(height=45, width=22, mask_area=550),
        ]
    )

    result = analyze_plant_group(
        group,
        image_loader=lambda _: image.copy(),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=image.shape[0], width=image.shape[1]),
        front_segmenter=lambda _: next(front_results),
        image_calibrator=lambda payload, *, view_name: _build_fake_calibration(payload, view_name=view_name, mm_per_pixel=0.2),
        top_flower_detector=lambda top_image, canopy_mask: FakeOrganDetectionResult(3, top_image.shape[0], top_image.shape[1]),
        top_fruit_detector=lambda top_image, canopy_mask: FakeOrganDetectionResult(
            2,
            top_image.shape[0],
            top_image.shape[1],
            mask=fruit_mask,
        ),
        debug_output_dir=None,
    )

    trait_map = result.trait_map()

    assert trait_map["fruit_area"].value == 0.02
    assert trait_map["fruit_area"].unit == "cm^2"
    assert trait_map["fruit_area"].status == "computed"
    assert trait_map["source_sink_ratio"].value == 10.0
    assert trait_map["source_sink_ratio"].unit == ""
    assert trait_map["source_sink_ratio"].status == "computed"


def test_analyze_plant_group_keeps_ratio_when_top_calibration_missing() -> None:
    """Missing TOP calibration should keep fruit_area in pixels and still compute source_sink_ratio."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )
    image = np.full((20, 25, 3), (20, 140, 20), dtype=np.uint8)
    fruit_mask = np.zeros((20, 25), dtype=np.uint8)
    fruit_mask[10:15, 8:18] = 255
    front_results = iter(
        [
            FakeFrontSegmentationResult(height=40, width=20, mask_area=500),
            FakeFrontSegmentationResult(height=45, width=22, mask_area=550),
        ]
    )

    result = analyze_plant_group(
        group,
        image_loader=lambda _: image.copy(),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=image.shape[0], width=image.shape[1]),
        front_segmenter=lambda _: next(front_results),
        image_calibrator=lambda payload, *, view_name: _build_fake_calibration(payload, view_name=view_name, mm_per_pixel=None),
        top_flower_detector=lambda top_image, canopy_mask: FakeOrganDetectionResult(3, top_image.shape[0], top_image.shape[1]),
        top_fruit_detector=lambda top_image, canopy_mask: FakeOrganDetectionResult(
            2,
            top_image.shape[0],
            top_image.shape[1],
            mask=fruit_mask,
        ),
        debug_output_dir=None,
    )

    trait_map = result.trait_map()

    assert trait_map["fruit_area"].value == 50
    assert trait_map["fruit_area"].unit == "px^2"
    assert trait_map["fruit_area"].status == "computed"
    assert trait_map["source_sink_ratio"].value == 10.0
    assert trait_map["source_sink_ratio"].unit == ""
    assert trait_map["source_sink_ratio"].status == "computed"


def test_analyze_plant_group_returns_empty_ratio_when_no_fruit_detected() -> None:
    """A zero-area fruit mask should keep fruit_area at zero and leave source_sink_ratio empty."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )
    image = np.full((20, 25, 3), (20, 140, 20), dtype=np.uint8)
    front_results = iter(
        [
            FakeFrontSegmentationResult(height=40, width=20, mask_area=500),
            FakeFrontSegmentationResult(height=45, width=22, mask_area=550),
        ]
    )

    result = analyze_plant_group(
        group,
        image_loader=lambda _: image.copy(),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=image.shape[0], width=image.shape[1]),
        front_segmenter=lambda _: next(front_results),
        image_calibrator=lambda payload, *, view_name: _build_fake_calibration(payload, view_name=view_name, mm_per_pixel=0.2),
        top_flower_detector=lambda top_image, canopy_mask: FakeOrganDetectionResult(3, top_image.shape[0], top_image.shape[1]),
        top_fruit_detector=lambda top_image, canopy_mask: FakeOrganDetectionResult(0, top_image.shape[0], top_image.shape[1]),
        debug_output_dir=None,
    )

    trait_map = result.trait_map()

    assert trait_map["fruit_area"].value == 0
    assert trait_map["fruit_area"].status == "computed"
    assert trait_map["source_sink_ratio"].value is None
    assert trait_map["source_sink_ratio"].status == "computed"


def test_analyze_plant_group_uses_classic_fruit_mask_for_area_when_yolo_counts_are_available() -> None:
    """YOLO can supply counts while classic fruit masks still supply area and ratio."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )
    image = np.full((20, 25, 3), (20, 140, 20), dtype=np.uint8)
    fruit_mask = np.zeros((20, 25), dtype=np.uint8)
    fruit_mask[10:15, 8:18] = 255
    front_results = iter(
        [
            FakeFrontSegmentationResult(height=40, width=20, mask_area=500),
            FakeFrontSegmentationResult(height=45, width=22, mask_area=550),
        ]
    )

    result = analyze_plant_group(
        group,
        image_loader=lambda _: image.copy(),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=image.shape[0], width=image.shape[1]),
        front_segmenter=lambda _: next(front_results),
        image_calibrator=lambda payload, *, view_name: _build_fake_calibration(payload, view_name=view_name, mm_per_pixel=0.2),
        top_organ_detector=lambda top_image, canopy_mask: FakeYoloOrganDetectionResult(
            flower_count=3,
            flower_bud_count=4,
            fruit_count=2,
            height=top_image.shape[0],
            width=top_image.shape[1],
        ),
        top_fruit_detector=lambda top_image, canopy_mask: FakeOrganDetectionResult(
            2,
            top_image.shape[0],
            top_image.shape[1],
            mask=fruit_mask,
        ),
        debug_output_dir=None,
    )

    trait_map = result.trait_map()

    assert trait_map["fruit_count"].value == 2
    assert trait_map["fruit_area"].value == 0.02
    assert trait_map["source_sink_ratio"].value == 10.0


def test_analyze_plant_group_populates_yolo_flower_bud_count() -> None:
    """Combined YOLO TOP detection should populate flower, flower bud, and fruit traits."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )
    image = np.full((24, 24, 3), (20, 140, 20), dtype=np.uint8)
    front_results = iter(
        [
            FakeFrontSegmentationResult(height=40, width=20, mask_area=500),
            FakeFrontSegmentationResult(height=45, width=22, mask_area=550),
        ]
    )

    result = analyze_plant_group(
        group,
        image_loader=lambda _: image.copy(),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=image.shape[0], width=image.shape[1]),
        front_segmenter=lambda _: next(front_results),
        image_calibrator=lambda payload, *, view_name: _build_fake_calibration(payload, view_name=view_name, mm_per_pixel=0.2),
        top_organ_detector=lambda top_image, canopy_mask: FakeYoloOrganDetectionResult(
            flower_count=3,
            flower_bud_count=4,
            fruit_count=2,
            height=top_image.shape[0],
            width=top_image.shape[1],
        ),
        debug_output_dir=None,
    )

    trait_map = result.trait_map()

    assert trait_map["flower_count"].value == 3
    assert trait_map["flower_bud_count"].value == 4
    assert trait_map["fruit_count"].value == 2


def test_top_organ_filter_mask_uses_convex_hull_not_green_mask_only() -> None:
    """YOLO organ filtering should keep flower centers inside the plant hull even if absent from the green mask."""

    segmentation = FakeTopSegmentationResult(height=100, width=120)
    segmentation.mask = np.zeros((100, 120), dtype=np.uint8)
    segmentation.mask[44:56, 54:66] = 255
    segmentation.convex_hull = np.array(
        [[[20, 20]], [[100, 25]], [[95, 80]], [[25, 75]]],
        dtype=np.int32,
    )

    organ_mask = _build_top_organ_filter_mask(segmentation)

    assert organ_mask[50, 60] == 255
    assert organ_mask[70, 80] == 255
    assert organ_mask[5, 5] == 0


def test_analyze_plant_group_exports_flower_and_fruit_debug_artifacts(tmp_path: Path) -> None:
    """Flower and fruit debug-artifact groups should be exported in debug mode."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )
    image = np.full((20, 20, 3), (20, 140, 20), dtype=np.uint8)
    front_results = iter(
        [
            FakeFrontSegmentationResult(height=24, width=12, mask_area=180),
            FakeFrontSegmentationResult(height=26, width=14, mask_area=220),
        ]
    )

    result = analyze_plant_group(
        group,
        image_loader=lambda _: image.copy(),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=image.shape[0], width=image.shape[1]),
        front_segmenter=lambda _: next(front_results),
        image_calibrator=lambda payload, *, view_name: _build_fake_calibration(payload, view_name=view_name, mm_per_pixel=0.2),
        top_flower_detector=lambda top_image, canopy_mask: FakeOrganDetectionResult(3, top_image.shape[0], top_image.shape[1]),
        top_fruit_detector=lambda top_image, canopy_mask: FakeOrganDetectionResult(2, top_image.shape[0], top_image.shape[1]),
        debug_output_dir=tmp_path,
    )

    assert "flower_count" in result.debug_artifact_paths
    assert "flower_bud_count" in result.debug_artifact_paths
    assert "fruit_count" in result.debug_artifact_paths
    assert result.debug_artifact_paths["flower_count"][0].exists()
    assert result.debug_artifact_paths["fruit_count"][0].exists()


def test_analyze_plant_group_keeps_main_traits_when_organ_detector_raises() -> None:
    """Flower/fruit detector failures should not discard already computed canopy traits."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )
    image = np.full((24, 24, 3), (20, 140, 20), dtype=np.uint8)
    front_results = iter(
        [
            FakeFrontSegmentationResult(height=40, width=20, mask_area=500),
            FakeFrontSegmentationResult(height=45, width=22, mask_area=550),
        ]
    )

    result = analyze_plant_group(
        group,
        image_loader=lambda _: image.copy(),
        top_segmenter=lambda _: FakeTopSegmentationResult(height=image.shape[0], width=image.shape[1]),
        front_segmenter=lambda _: next(front_results),
        image_calibrator=lambda payload, *, view_name: _build_fake_calibration(payload, view_name=view_name, mm_per_pixel=0.2),
        top_flower_detector=lambda top_image, canopy_mask: (_ for _ in ()).throw(RuntimeError("flower boom")),
        top_fruit_detector=lambda top_image, canopy_mask: (_ for _ in ()).throw(RuntimeError("fruit boom")),
        debug_output_dir=None,
    )

    trait_map = result.trait_map()

    assert result.status == "analysis_complete"
    assert trait_map["leaf_area"].status == "computed"
    assert trait_map["flower_count"].value is None
    assert trait_map["flower_bud_count"].value is None
    assert trait_map["fruit_count"].value is None
    assert trait_map["fruit_area"].value is None
    assert trait_map["source_sink_ratio"].value is None
    assert trait_map["flower_count"].status == "segmentation_failed"
    assert trait_map["flower_bud_count"].status == "pending_algorithm"
    assert trait_map["fruit_count"].status == "segmentation_failed"
    assert trait_map["fruit_area"].status == "segmentation_failed"
    assert trait_map["source_sink_ratio"].status == "segmentation_failed"
