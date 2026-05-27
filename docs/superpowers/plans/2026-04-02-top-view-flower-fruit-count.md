# TOP Flower Fruit Count Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为现有草莓 RGB 表型分析流程增加基于 TOP 俯视图的 `花朵数` 和 `果实数` 计数，并在 GUI 结果表、CSV、Excel、调试输出中完整贯通。

**Architecture:** 新增独立的 `core/organs.py` 负责俯视角花果识别，不改动现有 `core/segmentation.py` 的植株冠层分割职责。`core/pipeline.py` 负责在 TOP 分割成功后调用花果检测并写回 `TraitResult`，`utils/exporter.py` 负责保证新指标按稳定顺序导出。

**Tech Stack:** Python 3.11, NumPy, OpenCV, pytest, openpyxl

---

### Task 1: 新建 TOP 花果识别模块

**Files:**
- Create: `core/organs.py`
- Test: `tests/test_organs.py`

- [ ] **Step 1: 写失败测试，固定花果计数、噪声过滤、掩膜约束和粘连分裂行为**

```python
import cv2
import numpy as np

from core.organs import detect_top_flowers, detect_top_fruits


def _build_top_scene() -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    image[:, :] = (30, 120, 30)
    canopy_mask = np.zeros((160, 160), dtype=np.uint8)
    cv2.circle(canopy_mask, (80, 80), 58, 255, -1)
    return image, canopy_mask


def test_detect_top_flowers_counts_visible_white_regions() -> None:
    image, canopy_mask = _build_top_scene()
    cv2.circle(image, (60, 70), 8, (245, 245, 245), -1)
    cv2.circle(image, (96, 88), 9, (250, 250, 250), -1)

    result = detect_top_flowers(image, canopy_mask)

    assert result.status == "computed"
    assert result.count == 2
    assert len(result.instances) == 2
    assert result.mask.dtype == np.uint8


def test_detect_top_fruits_counts_visible_red_regions() -> None:
    image, canopy_mask = _build_top_scene()
    cv2.circle(image, (68, 64), 9, (30, 30, 220), -1)
    cv2.circle(image, (92, 96), 10, (20, 40, 235), -1)

    result = detect_top_fruits(image, canopy_mask)

    assert result.status == "computed"
    assert result.count == 2
    assert len(result.instances) == 2


def test_detect_top_organs_ignores_noise_and_outside_canopy() -> None:
    image, canopy_mask = _build_top_scene()
    cv2.circle(image, (20, 20), 9, (250, 250, 250), -1)
    cv2.circle(image, (75, 78), 1, (250, 250, 250), -1)
    cv2.circle(image, (88, 82), 8, (250, 250, 250), -1)

    flower_result = detect_top_flowers(image, canopy_mask)

    assert flower_result.count == 1


def test_detect_top_fruits_splits_touching_regions() -> None:
    image, canopy_mask = _build_top_scene()
    cv2.circle(image, (74, 80), 12, (20, 30, 230), -1)
    cv2.circle(image, (92, 80), 12, (20, 30, 230), -1)

    result = detect_top_fruits(image, canopy_mask)

    assert result.count == 2
    assert len(result.instances) == 2
```

- [ ] **Step 2: 运行测试，确认模块尚不存在而失败**

Run: `pytest tests/test_organs.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'core.organs'`

- [ ] **Step 3: 编写最小实现，新建 `core/organs.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class OrganInstance:
    label_id: int
    area_pixels: int
    centroid_xy: tuple[float, float]
    bounding_box: tuple[int, int, int, int] | None


@dataclass(slots=True)
class TopFlowerDetectionResult:
    status: str
    message: str
    count: int
    mask: np.ndarray
    labeled_mask: np.ndarray
    instances: list[OrganInstance]
    overlay_image: np.ndarray
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class TopFruitDetectionResult:
    status: str
    message: str
    count: int
    mask: np.ndarray
    labeled_mask: np.ndarray
    instances: list[OrganInstance]
    overlay_image: np.ndarray
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


def detect_top_flowers(image: np.ndarray, canopy_mask: np.ndarray) -> TopFlowerDetectionResult:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    raw_mask = (
        (hsv[:, :, 1] <= 90)
        & (hsv[:, :, 2] >= 185)
        & (lab[:, :, 0] >= 185)
        & (lab[:, :, 2] >= 110)
        & (lab[:, :, 2] <= 155)
    ).astype(np.uint8) * 255
    return _finalize_detection(
        image=image,
        canopy_mask=canopy_mask,
        raw_mask=raw_mask,
        min_area=20,
        label="flower",
        result_type=TopFlowerDetectionResult,
    )


def detect_top_fruits(image: np.ndarray, canopy_mask: np.ndarray) -> TopFruitDetectionResult:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask_1 = cv2.inRange(hsv, np.array([0, 80, 50], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8))
    red_mask_2 = cv2.inRange(hsv, np.array([165, 80, 50], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    raw_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    return _finalize_detection(
        image=image,
        canopy_mask=canopy_mask,
        raw_mask=raw_mask,
        min_area=24,
        label="fruit",
        result_type=TopFruitDetectionResult,
    )


def _finalize_detection(
    *,
    image: np.ndarray,
    canopy_mask: np.ndarray,
    raw_mask: np.ndarray,
    min_area: int,
    label: str,
    result_type,
):
    canopy_source = cv2.bitwise_and(image, image, mask=canopy_mask)
    canopy_only = cv2.bitwise_and(raw_mask, canopy_mask)
    denoised = cv2.GaussianBlur(canopy_only, (5, 5), 0)
    cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    cleaned = (cleaned > 0).astype(np.uint8) * 255
    split_labels, debug_images = _split_touching_instances(image, cleaned)
    instances = _collect_instances(split_labels, min_area=min_area)
    kept = np.zeros_like(cleaned)
    filtered_labels = np.zeros_like(split_labels, dtype=np.int32)
    for next_id, instance in enumerate(instances, start=1):
        filtered_labels[split_labels == instance.label_id] = next_id
        kept[split_labels == instance.label_id] = 255
    overlay = _build_overlay(image, filtered_labels)
    debug_images.update(
        {
            "canopy_source": canopy_source,
            "canopy_mask": canopy_mask.copy(),
            "raw_mask": raw_mask,
            "canopy_limited_mask": canopy_only,
            "cleaned_mask": kept,
            "labeled_mask": np.clip(filtered_labels * 40, 0, 255).astype(np.uint8),
            "overlay": overlay,
        }
    )
    return result_type(
        status="computed",
        message=f"{label} count computed from TOP view visible organs.",
        count=len(instances),
        mask=kept,
        labeled_mask=filtered_labels,
        instances=[
            OrganInstance(
                label_id=index,
                area_pixels=instance.area_pixels,
                centroid_xy=instance.centroid_xy,
                bounding_box=instance.bounding_box,
            )
            for index, instance in enumerate(instances, start=1)
        ],
        overlay_image=overlay,
        debug_images=debug_images,
    )


def _split_touching_instances(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    sure_background = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, peaks = cv2.threshold(distance, distance.max() * 0.45 if distance.max() > 0 else 0, 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    marker_count, markers = cv2.connectedComponents(peaks)
    markers = markers + 1
    markers[sure_background == 0] = 0
    watershed_input = image.copy()
    watershed_markers = cv2.watershed(watershed_input, markers.astype(np.int32))
    watershed_markers[watershed_markers < 1] = 0
    return watershed_markers.astype(np.int32), {
        "distance_map": cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        "peak_mask": peaks,
    }


def _collect_instances(labeled_mask: np.ndarray, *, min_area: int) -> list[OrganInstance]:
    instances: list[OrganInstance] = []
    for label_id in sorted(int(v) for v in np.unique(labeled_mask) if v > 1):
        binary = (labeled_mask == label_id).astype(np.uint8)
        area = int(binary.sum())
        if area < min_area:
            continue
        ys, xs = np.where(binary > 0)
        x, y, width, height = cv2.boundingRect(np.column_stack([xs, ys]).astype(np.int32))
        instances.append(
            OrganInstance(
                label_id=label_id,
                area_pixels=area,
                centroid_xy=(float(xs.mean()), float(ys.mean())),
                bounding_box=(int(x), int(y), int(width), int(height)),
            )
        )
    return instances


def _build_overlay(image: np.ndarray, labeled_mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    for label_id in sorted(int(v) for v in np.unique(labeled_mask) if v > 0):
        ys, xs = np.where(labeled_mask == label_id)
        if xs.size == 0:
            continue
        x, y, width, height = cv2.boundingRect(np.column_stack([xs, ys]).astype(np.int32))
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 255, 0), 2)
        cv2.putText(overlay, str(label_id), (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return overlay
```

- [ ] **Step 4: 运行器官识别测试，确认最小实现通过**

Run: `pytest tests/test_organs.py -v`

Expected: PASS for all new tests in `tests/test_organs.py`

- [ ] **Step 5: 做一次任务级检查点**

Run: `git status --short`

Expected:
- If this workspace is attached to a git root: see only `core/organs.py` and `tests/test_organs.py` changed, then commit with `git add core/organs.py tests/test_organs.py && git commit -m "feat: add top-view flower and fruit detectors"`
- If this workspace still returns `fatal: not a git repository`, skip the commit and continue to Task 2

### Task 2: 将花果识别接入 pipeline、结果表和调试输出

**Files:**
- Modify: `core/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: 写失败测试，固定新指标进入结果和调试输出**

```python
class FakeOrganDetectionResult:
    def __init__(self, count: int, height: int, width: int) -> None:
        self.status = "computed"
        self.message = "ok"
        self.count = count
        self.mask = np.zeros((height, width), dtype=np.uint8)
        self.labeled_mask = np.zeros((height, width), dtype=np.int32)
        self.instances = []
        self.overlay_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.debug_images = {
            "raw_mask": np.zeros((height, width), dtype=np.uint8),
            "cleaned_mask": np.zeros((height, width), dtype=np.uint8),
            "distance_map": np.zeros((height, width), dtype=np.uint8),
            "overlay": np.zeros((height, width, 3), dtype=np.uint8),
        }


def test_analyze_plant_group_populates_flower_and_fruit_counts() -> None:
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


def test_analyze_plant_group_exports_flower_and_fruit_debug_artifacts(tmp_path: Path) -> None:
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
    assert "fruit_count" in result.debug_artifact_paths
    assert result.debug_artifact_paths["flower_count"][0].exists()
    assert result.debug_artifact_paths["fruit_count"][0].exists()


def test_analyze_plant_group_keeps_main_traits_when_organ_detector_raises() -> None:
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
    assert trait_map["fruit_count"].value is None
    assert trait_map["flower_count"].status == "segmentation_failed"
    assert trait_map["fruit_count"].status == "segmentation_failed"
```

- [ ] **Step 2: 运行 pipeline 测试，确认新注入点尚不存在而失败**

Run: `pytest tests/test_pipeline.py -v`

Expected: FAIL with `TypeError` because `analyze_plant_group()` does not yet accept `top_flower_detector` or `top_fruit_detector`

- [ ] **Step 3: 修改 `core/pipeline.py`，贯通新指标和调试输出**

```python
from core.organs import detect_top_flowers, detect_top_fruits

TopFlowerDetector = Callable[[np.ndarray, np.ndarray], Any]
TopFruitDetector = Callable[[np.ndarray, np.ndarray], Any]


@dataclass(slots=True)
class PlantAnalysisResult:
    sample_id: str
    status: str
    message: str
    traits: list[TraitResult]
    view_results: dict[str, ViewLoadResult]
    top_segmentation: Any | None = None
    front_segmentations: dict[str, Any] = field(default_factory=dict)
    calibration_results: dict[str, Any] = field(default_factory=dict)
    debug_artifact_paths: dict[str, list[Path]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    top_flower_detection: Any | None = None
    top_fruit_detection: Any | None = None


TRAIT_SPECS: tuple[TraitSpec, ...] = (
    TraitSpec("leaf_area", "叶面积", (VIEW_TOP,), "cm^2"),
    TraitSpec("greenness", "绿色程度", (VIEW_TOP,), "ExG"),
    TraitSpec("convex_hull_area", "最小凸包面积", (VIEW_TOP,), "cm^2"),
    TraitSpec("canopy_height", "冠层高度", (VIEW_FRONT_0, VIEW_FRONT_180), "cm"),
    TraitSpec("canopy_width", "植株冠径", (VIEW_TOP,), "cm"),
    TraitSpec("side_projection_area", "侧视投影面积", (VIEW_FRONT_0, VIEW_FRONT_180), "cm^2"),
    TraitSpec("flower_count", "花朵数", (VIEW_TOP,), "count"),
    TraitSpec("fruit_count", "果实数", (VIEW_TOP,), "count"),
)


def analyze_plant_group(
    group: PlantImageGroup,
    *,
    emit_log: LogCallback | None = None,
    image_loader: ImageLoader | None = None,
    top_segmenter: TopSegmenter | None = None,
    front_segmenter: FrontSegmenter | None = None,
    image_calibrator: ImageCalibrator | None = None,
    top_flower_detector: TopFlowerDetector | None = None,
    top_fruit_detector: TopFruitDetector | None = None,
    calibration_reference: Any | None = None,
    debug_output_dir: str | Path | None = None,
    manual_color_card_regions: dict[str, tuple[int, int, int, int]] | None = None,
    precomputed_calibration: dict[str, Any] | None = None,
) -> PlantAnalysisResult:
    ...
    top_flower_detector = top_flower_detector or detect_top_flowers
    top_fruit_detector = top_fruit_detector or detect_top_fruits
    ...
    try:
        result.top_flower_detection = top_flower_detector(calibrated_images[VIEW_TOP], top_segmentation.mask)
    except Exception as error:  # noqa: BLE001
        result.errors.append(f"TOP flower detection failed: {error}")
        result.top_flower_detection = None
    try:
        result.top_fruit_detection = top_fruit_detector(calibrated_images[VIEW_TOP], top_segmentation.mask)
    except Exception as error:  # noqa: BLE001
        result.errors.append(f"TOP fruit detection failed: {error}")
        result.top_fruit_detection = None
    _apply_top_organ_counts(
        result,
        flower_detection=result.top_flower_detection,
        fruit_detection=result.top_fruit_detection,
        calibration=result.calibration_results.get(VIEW_TOP),
    )
    ...
    if debug_output_dir is not None:
        result.debug_artifact_paths.update(
            _export_top_organ_debug_artifacts(
                sample_id=group.sample_id,
                flower_detection=result.top_flower_detection,
                fruit_detection=result.top_fruit_detection,
                output_root=Path(debug_output_dir),
            )
        )
    ...


def _apply_top_organ_counts(
    result: PlantAnalysisResult,
    *,
    flower_detection: Any,
    fruit_detection: Any,
    calibration: Any | None,
) -> None:
    trait_map = result.trait_map()
    used_calibrated_top = bool(getattr(calibration, "is_calibrated", False))
    image_note = "基于色卡校正后的 TOP 图像" if used_calibrated_top else "基于原始 TOP 图像，可信度较低"

    if flower_detection is None:
        trait_map["flower_count"].value = None
        trait_map["flower_count"].unit = "count"
        trait_map["flower_count"].status = "segmentation_failed"
        trait_map["flower_count"].message = "花朵识别未完成，保留其它已计算表型。"
    else:
        trait_map["flower_count"].value = int(getattr(flower_detection, "count", 0))
        trait_map["flower_count"].unit = "count"
        trait_map["flower_count"].status = "computed"
        trait_map["flower_count"].message = f"{image_note}统计俯视图可见开放白花数量。"

    if fruit_detection is None:
        trait_map["fruit_count"].value = None
        trait_map["fruit_count"].unit = "count"
        trait_map["fruit_count"].status = "segmentation_failed"
        trait_map["fruit_count"].message = "果实识别未完成，保留其它已计算表型。"
    else:
        trait_map["fruit_count"].value = int(getattr(fruit_detection, "count", 0))
        trait_map["fruit_count"].unit = "count"
        trait_map["fruit_count"].status = "computed"
        trait_map["fruit_count"].message = f"{image_note}统计俯视图可见成熟红果数量。"


def _export_top_organ_debug_artifacts(
    *,
    sample_id: str,
    flower_detection: Any,
    fruit_detection: Any,
    output_root: Path,
) -> dict[str, list[Path]]:
    flower_steps = _collect_debug_steps(
        ("flower_canopy_source", getattr(flower_detection, "debug_images", {}).get("canopy_source")),
        ("flower_raw_mask", getattr(flower_detection, "debug_images", {}).get("raw_mask")),
        ("flower_cleaned_mask", getattr(flower_detection, "debug_images", {}).get("cleaned_mask")),
        ("flower_distance_map", getattr(flower_detection, "debug_images", {}).get("distance_map")),
        ("flower_labeled_mask", getattr(flower_detection, "debug_images", {}).get("labeled_mask")),
        ("flower_overlay", getattr(flower_detection, "overlay_image", None)),
    )
    fruit_steps = _collect_debug_steps(
        ("fruit_canopy_source", getattr(fruit_detection, "debug_images", {}).get("canopy_source")),
        ("fruit_raw_mask", getattr(fruit_detection, "debug_images", {}).get("raw_mask")),
        ("fruit_cleaned_mask", getattr(fruit_detection, "debug_images", {}).get("cleaned_mask")),
        ("fruit_distance_map", getattr(fruit_detection, "debug_images", {}).get("distance_map")),
        ("fruit_labeled_mask", getattr(fruit_detection, "debug_images", {}).get("labeled_mask")),
        ("fruit_overlay", getattr(fruit_detection, "overlay_image", None)),
    )
    return {
        "flower_count": save_debug_steps(sample_id, "flower_count", flower_steps, output_root=output_root),
        "fruit_count": save_debug_steps(sample_id, "fruit_count", fruit_steps, output_root=output_root),
    }


def _build_summary_message(result: PlantAnalysisResult) -> str:
    trait_map = result.trait_map()
    calibration_done = sum(
        1 for item in result.calibration_results.values() if bool(getattr(item, "is_calibrated", False))
    )
    return (
        f"分析完成。色卡校正成功{calibration_done}/3视角。"
        f" 叶面积={trait_map['leaf_area'].display_value} {trait_map['leaf_area'].unit},"
        f" 凸包面积={trait_map['convex_hull_area'].display_value} {trait_map['convex_hull_area'].unit},"
        f" 绿色程度={trait_map['greenness'].display_value} {trait_map['greenness'].unit},"
        f" 冠层高度={trait_map['canopy_height'].display_value} {trait_map['canopy_height'].unit},"
        f" 植株冠径={trait_map['canopy_width'].display_value} {trait_map['canopy_width'].unit},"
        f" 侧视投影面积={trait_map['side_projection_area'].display_value} {trait_map['side_projection_area'].unit},"
        f" 花朵数={trait_map['flower_count'].display_value} {trait_map['flower_count'].unit},"
        f" 果实数={trait_map['fruit_count'].display_value} {trait_map['fruit_count'].unit}。"
    )
```

- [ ] **Step 4: 运行 pipeline 测试，确认新指标链路打通**

Run: `pytest tests/test_pipeline.py -v`

Expected: PASS for the new flower/fruit assertions, and existing pipeline tests remain green

- [ ] **Step 5: 做一次任务级检查点**

Run: `git status --short`

Expected:
- If git is available: see `core/pipeline.py` and `tests/test_pipeline.py` changed, then commit with `git add core/pipeline.py tests/test_pipeline.py && git commit -m "feat: integrate top-view flower and fruit counts"`
- If the workspace is still not a git repo, skip the commit and continue to Task 3

### Task 3: 固定导出顺序和中英文结果输出

**Files:**
- Modify: `utils/exporter.py`
- Test: `tests/test_exporter.py`

- [ ] **Step 1: 写失败测试，锁定 CSV/Excel 必须包含新字段**

```python
def test_build_result_record_includes_flower_and_fruit_counts() -> None:
    group = PlantImageGroup(sample_id="1AB")
    result = _build_result()

    record = build_result_record(group, result)

    assert record["flower_count"] == 3
    assert record["fruit_count"] == 2


def test_export_batch_report_writes_excel_with_flower_and_fruit_headers(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    from openpyxl import load_workbook

    group = PlantImageGroup(sample_id="1AB")
    result = _build_result()
    report = BatchAnalysisReport(
        directory=tmp_path,
        sample_results=[BatchSampleResult(group=group, result=result)],
        started_at=__import__("datetime").datetime.now(),
        finished_at=__import__("datetime").datetime.now(),
    )

    export_path = export_batch_report(report, tmp_path / "batch.xlsx")

    workbook = load_workbook(export_path)
    headers = [cell.value for cell in workbook.active[1]]

    assert "花朵数(count)" in headers
    assert "果实数(count)" in headers
```

- [ ] **Step 2: 运行导出测试，确认当前导出契约尚未覆盖这两个新字段**

Run: `pytest tests/test_exporter.py -v`

Expected: FAIL because `_build_result()` 还未包含 `flower_count`、`fruit_count`，且导出列顺序未被显式固定

- [ ] **Step 3: 修改 `utils/exporter.py`，显式按 `TRAIT_SPECS` 顺序导出，并补齐测试构造结果**

```python
def _collect_fieldnames(records: Sequence[dict[str, Any]]) -> list[str]:
    include_debug_fields = any("result_status" in record for record in records)
    preferred_order = [
        "sample_id",
        "result_status",
        "result_message",
        "group_is_complete",
        "missing_views",
    ] if include_debug_fields else ["sample_id"]

    ordered_keys: list[str] = []
    seen: set[str] = set()

    for key in preferred_order:
        if any(key in record for record in records):
            ordered_keys.append(key)
            seen.add(key)

    trait_keys = [spec.key for spec in TRAIT_SPECS]
    debug_suffixes = ("_value", "_unit", "_status", "_message")
    for trait_key in trait_keys:
        if include_debug_fields:
            candidates = [f"{trait_key}{suffix}" for suffix in debug_suffixes]
        else:
            candidates = [trait_key]
        for candidate in candidates:
            if any(candidate in record for record in records) and candidate not in seen:
                ordered_keys.append(candidate)
                seen.add(candidate)

    for key in _iter_record_keys(records):
        if key not in seen:
            ordered_keys.append(key)
            seen.add(key)

    return ordered_keys
```

```python
def _build_result() -> PlantAnalysisResult:
    traits = [
        TraitResult("leaf_area", "叶面积", ("TOP",), "cm^2", 12.34, "computed", "ok"),
        TraitResult("greenness", "绿色程度", ("TOP",), "ExG", 45.67, "computed", "ok"),
        TraitResult("convex_hull_area", "最小凸包面积", ("TOP",), "cm^2", 14.56, "computed", "ok"),
        TraitResult("canopy_height", "冠层高度", ("FRONT-1", "FRONT-2"), "cm", 7.89, "computed", "ok"),
        TraitResult("canopy_width", "植株冠径", ("TOP",), "cm", 8.90, "computed", "ok"),
        TraitResult("side_projection_area", "侧视投影面积", ("FRONT-1", "FRONT-2"), "cm^2", 9.87, "computed", "ok"),
        TraitResult("flower_count", "花朵数", ("TOP",), "count", 3, "computed", "ok"),
        TraitResult("fruit_count", "果实数", ("TOP",), "count", 2, "computed", "ok"),
    ]
    ...
```

- [ ] **Step 4: 运行导出测试，确认 CSV 和 Excel 都带出新字段**

Run: `pytest tests/test_exporter.py -v`

Expected: PASS and exported headers include `花朵数(count)` and `果实数(count)`

- [ ] **Step 5: 做最终检查点**

Run: `git status --short`

Expected:
- If git is available: see `utils/exporter.py` and `tests/test_exporter.py` changed, then commit with `git add utils/exporter.py tests/test_exporter.py && git commit -m "test: cover flower and fruit export columns"`
- If the workspace is still not a git repo, skip the commit and move to final verification

### Task 4: 最终回归验证

**Files:**
- Modify: none
- Test: `tests/test_organs.py`, `tests/test_pipeline.py`, `tests/test_exporter.py`

- [ ] **Step 1: 运行新增的最小回归集**

Run: `pytest tests/test_organs.py tests/test_pipeline.py tests/test_exporter.py -v`

Expected: PASS for all targeted tests

- [ ] **Step 2: 运行项目全量测试**

Run: `pytest -q tests`

Expected:
- New flower/fruit tests PASS
- Existing tests stay green
- If unrelated historical failures remain, record them explicitly before claiming completion

- [ ] **Step 3: 记录最终状态**

Run: `python -c "print('flower/fruit plan verification complete')"`

Expected: prints `flower/fruit plan verification complete`
