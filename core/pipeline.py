"""Analysis pipeline for a three-view strawberry plant sample."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core.grouping import PlantImageGroup
from core.organs import detect_top_flowers, detect_top_fruits
from core.traits import compute_front_view_traits, compute_top_traits, fuse_front_traits
from core.yolo_counter import detect_top_organs_with_yolo, resolve_default_model_path
from utils.debug_artifacts import (
    create_debug_montage,
    create_gray_background_focus_image,
    create_heatmap,
    create_mask_overlay,
    create_masked_color_image,
    save_debug_steps,
)


VIEW_TOP = "TOP"
VIEW_FRONT_0 = "FRONT-1"
VIEW_FRONT_180 = "FRONT-2"

STATUS_LABELS = {
    "uninitialized": "未开始",
    "pending": "等待中",
    "pending_algorithm": "待实现",
    "ready_for_implementation": "待接入",
    "segmentation_ready": "分割已完成",
    "computed": "已计算",
    "analysis_complete": "分析完成",
    "incomplete_input": "输入不完整",
    "load_failed": "图像读取失败",
    "dependency_error": "依赖缺失",
    "segmentation_failed": "分割失败",
    "missing": "缺失",
    "loaded": "已加载",
    "calibrated": "已校正",
    "not_detected": "未检测到",
}

LogCallback = Callable[[str], None]
ImageLoader = Callable[[Path], Any]
TopSegmenter = Callable[[Any], Any]
FrontSegmenter = Callable[[Any], Any]
ImageCalibrator = Callable[..., Any]
TopFlowerDetector = Callable[[np.ndarray, np.ndarray], Any]
TopFruitDetector = Callable[[np.ndarray, np.ndarray], Any]
TopOrganDetector = Callable[[np.ndarray, np.ndarray], Any]


@dataclass(frozen=True, slots=True)
class TraitSpec:
    """Definition of one phenotype trait shown in the UI."""

    key: str
    label: str
    source_views: tuple[str, ...]
    unit: str


@dataclass(slots=True)
class TraitResult:
    """Analysis output for a single phenotype trait."""

    key: str
    label: str
    source_views: tuple[str, ...]
    unit: str
    value: float | int | str | None = None
    status: str = "pending_algorithm"
    message: str = "Trait extraction is not implemented yet."

    @property
    def display_value(self) -> str:
        """Return a UI-friendly value string."""

        if self.value is None:
            return "--"
        if isinstance(self.value, float):
            return f"{self.value:.2f}"
        return str(self.value)


@dataclass(slots=True)
class ViewLoadResult:
    """Loading status for one required input view."""

    view_name: str
    path: Path | None
    status: str
    shape: tuple[int, ...] | None = None
    message: str = ""


@dataclass(slots=True)
class PlantAnalysisResult:
    """Analysis result for one grouped strawberry plant sample."""

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
    top_flower_bud_detection: Any | None = None
    top_fruit_detection: Any | None = None
    top_organ_detection: Any | None = None

    def trait_map(self) -> dict[str, TraitResult]:
        """Return trait results keyed by trait key."""

        return {trait.key: trait for trait in self.traits}


TRAIT_SPECS: tuple[TraitSpec, ...] = (
    TraitSpec("leaf_area", "叶面积", (VIEW_TOP,), "cm^2"),
    TraitSpec("greenness", "绿色程度", (VIEW_TOP,), "ExG"),
    TraitSpec("convex_hull_area", "最小凸包面积", (VIEW_TOP,), "cm^2"),
    TraitSpec("canopy_height", "冠层高度", (VIEW_FRONT_0, VIEW_FRONT_180), "cm"),
    TraitSpec("canopy_width", "植株冠径", (VIEW_TOP,), "cm"),
    TraitSpec("side_projection_area", "侧视投影面积", (VIEW_FRONT_0, VIEW_FRONT_180), "cm^2"),
    TraitSpec("flower_count", "花朵数", (VIEW_TOP,), "count"),
    TraitSpec("flower_bud_count", "花骨朵数", (VIEW_TOP,), "count"),
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
    top_organ_detector: TopOrganDetector | None = None,
    calibration_reference: Any | None = None,
    debug_output_dir: str | Path | None = None,
    manual_color_card_regions: dict[str, tuple[int, int, int, int]] | None = None,
    precomputed_calibration: dict[str, Any] | None = None,
) -> PlantAnalysisResult:
    """Run the current analysis for one three-view plant group.
    
    Args:
        manual_color_card_regions: Optional dict mapping view names to manual regions.
                                  Keys: VIEW_TOP, VIEW_FRONT_0, VIEW_FRONT_180.
        precomputed_calibration: Optional dict with precomputed calibration results.
                                Keys: 'calibrated_images', 'calibration_results'.
                                If provided, skips the calibration step.
    """

    _emit(emit_log, f"开始分析样本组 {group.sample_id}")
    result = PlantAnalysisResult(
        sample_id=group.sample_id,
        status="pending",
        message="分析尚未开始。",
        traits=_build_placeholder_traits(),
        view_results={
            VIEW_TOP: ViewLoadResult(VIEW_TOP, group.top_image, "missing", message="View not loaded."),
            VIEW_FRONT_0: ViewLoadResult(VIEW_FRONT_0, group.front_0_image, "missing", message="View not loaded."),
            VIEW_FRONT_180: ViewLoadResult(
                VIEW_FRONT_180,
                group.front_180_image,
                "missing",
                message="View not loaded.",
            ),
        },
    )

    if not group.is_complete:
        missing_views = ", ".join(group.missing_views)
        result.status = "incomplete_input"
        result.message = f"样本组不完整，缺少视角: {missing_views}"
        result.errors.append(result.message)
        _emit(emit_log, result.message)
        return result

    has_injected_components = any(
        component is not None
        for component in (
            image_loader,
            top_segmenter,
            front_segmenter,
            image_calibrator,
            top_flower_detector,
            top_fruit_detector,
        )
    )

    image_loader = image_loader or _resolve_image_loader(result, emit_log)
    if image_loader is None:
        return result

    top_segmenter = top_segmenter or _resolve_top_segmenter(result, emit_log)
    if top_segmenter is None:
        return result

    front_segmenter = front_segmenter or _resolve_front_segmenter(result, emit_log)
    if front_segmenter is None:
        return result

    image_calibrator = image_calibrator or _resolve_image_calibrator(result, emit_log)
    if image_calibrator is None:
        return result

    top_organ_detector = top_organ_detector or (
        None if has_injected_components else _resolve_top_organ_detector(result, emit_log)
    )
    top_flower_detector = top_flower_detector or detect_top_flowers
    top_fruit_detector = top_fruit_detector or detect_top_fruits

    # 使用预处理结果（如果提供）
    if precomputed_calibration is not None:
        _emit(emit_log, "使用预处理结果，跳过色卡校准步骤...")
        calibrated_images = precomputed_calibration.get("calibrated_images", {})
        result.calibration_results = precomputed_calibration.get("calibration_results", {})
        
        # 更新视角加载状态
        for view_name in (VIEW_TOP, VIEW_FRONT_0, VIEW_FRONT_180):
            if view_name in calibrated_images:
                img = calibrated_images[view_name]
                shape = tuple(int(v) for v in img.shape) if hasattr(img, "shape") else None
                result.view_results[view_name] = ViewLoadResult(
                    view_name=view_name,
                    path=getattr(group, {VIEW_TOP: "top_image", VIEW_FRONT_0: "front_0_image", VIEW_FRONT_180: "front_180_image"}[view_name], None),
                    status="calibrated",
                    shape=shape,
                    message=f"{view_name} 已使用预处理结果。"
                )
                _emit(emit_log, f"{view_name} 使用预处理校准图像，尺寸: {shape}")
    else:
        loaded_images = _load_group_images(group, result, image_loader=image_loader, emit_log=emit_log)
        if loaded_images is None:
            return result

        calibrated_images = _calibrate_group_images(
            loaded_images,
            result=result,
            emit_log=emit_log,
            image_calibrator=image_calibrator,
            calibration_reference=calibration_reference,
            manual_regions=manual_color_card_regions,
        )

    _emit(emit_log, "开始执行 TOP 俯视图分割")
    top_segmentation = _run_segmentation_step(
        result=result,
        emit_log=emit_log,
        view_name=VIEW_TOP,
        segmenter=top_segmenter,
        image=calibrated_images[VIEW_TOP],
    )
    if top_segmentation is None:
        return result
    result.top_segmentation = top_segmentation

    _emit(emit_log, "开始执行 FRONT-1 正视图分割")
    front_0_segmentation = _run_segmentation_step(
        result=result,
        emit_log=emit_log,
        view_name=VIEW_FRONT_0,
        segmenter=front_segmenter,
        image=calibrated_images[VIEW_FRONT_0],
    )
    if front_0_segmentation is None:
        return result
    result.front_segmentations[VIEW_FRONT_0] = front_0_segmentation

    _emit(emit_log, "开始执行 FRONT-2 正视图分割")
    front_180_segmentation = _run_segmentation_step(
        result=result,
        emit_log=emit_log,
        view_name=VIEW_FRONT_180,
        segmenter=front_segmenter,
        image=calibrated_images[VIEW_FRONT_180],
    )
    if front_180_segmentation is None:
        return result
    result.front_segmentations[VIEW_FRONT_180] = front_180_segmentation

    if top_organ_detector is not None:
        try:
            result.top_organ_detection = top_organ_detector(calibrated_images[VIEW_TOP], top_segmentation.mask)
        except Exception as error:  # noqa: BLE001
            result.errors.append(f"TOP YOLO organ detection failed: {error}")
            _emit(emit_log, f"TOP YOLO organ detection failed: {error}")
            result.top_organ_detection = None

    if result.top_organ_detection is None:
        try:
            result.top_flower_detection = top_flower_detector(calibrated_images[VIEW_TOP], top_segmentation.mask)
        except Exception as error:  # noqa: BLE001
            result.errors.append(f"TOP flower detection failed: {error}")
            _emit(emit_log, f"TOP flower detection failed: {error}")
            result.top_flower_detection = None

        try:
            result.top_fruit_detection = top_fruit_detector(calibrated_images[VIEW_TOP], top_segmentation.mask)
        except Exception as error:  # noqa: BLE001
            result.errors.append(f"TOP fruit detection failed: {error}")
            _emit(emit_log, f"TOP fruit detection failed: {error}")
            result.top_fruit_detection = None

    if debug_output_dir is not None:
        output_root = Path(debug_output_dir)
        # 仅在非预处理模式下导出色卡调试图像（预处理阶段已单独保存）
        if precomputed_calibration is None:
            result.debug_artifact_paths.update(
                _export_calibration_debug_artifacts(
                    sample_id=group.sample_id,
                    calibration_results=result.calibration_results,
                    output_root=output_root,
                )
            )
        result.debug_artifact_paths.update(
            _export_top_trait_debug_artifacts(
                sample_id=group.sample_id,
                original_top=calibrated_images[VIEW_TOP],
                top_segmentation=top_segmentation,
                output_root=output_root,
            )
        )
        result.debug_artifact_paths.update(
            _export_front_trait_debug_artifacts(
                sample_id=group.sample_id,
                front_0_image=calibrated_images[VIEW_FRONT_0],
                front_180_image=calibrated_images[VIEW_FRONT_180],
                front_0_segmentation=front_0_segmentation,
                front_180_segmentation=front_180_segmentation,
                output_root=output_root,
            )
        )
        result.debug_artifact_paths.update(
            _export_top_organ_debug_artifacts(
                sample_id=group.sample_id,
                organ_detection=result.top_organ_detection,
                flower_detection=result.top_flower_detection,
                fruit_detection=result.top_fruit_detection,
                output_root=output_root,
            )
        )
        for trait_key, paths in result.debug_artifact_paths.items():
            if paths:
                _emit(emit_log, f"{trait_key} 中间过程已保存到: {paths[0].parent}")

    _emit(emit_log, "开始计算 TOP 表型参数")
    top_measurements = compute_top_traits(calibrated_images[VIEW_TOP], top_segmentation)
    _apply_top_trait_measurements(
        result,
        top_measurements,
        calibration=result.calibration_results.get(VIEW_TOP),
    )
    _apply_top_organ_counts(
        result,
        organ_detection=result.top_organ_detection,
        flower_detection=result.top_flower_detection,
        fruit_detection=result.top_fruit_detection,
        calibration=result.calibration_results.get(VIEW_TOP),
    )

    _emit(emit_log, "开始计算 FRONT 性状并进行双视角融合")
    front_measurements = fuse_front_traits(front_0_segmentation, front_180_segmentation)
    _apply_front_trait_measurements(
        result,
        front_measurements,
        front_0_calibration=result.calibration_results.get(VIEW_FRONT_0),
        front_180_calibration=result.calibration_results.get(VIEW_FRONT_180),
    )

    result.status = "analysis_complete"
    result.message = _build_summary_message(result)
    _emit(emit_log, result.message)
    return result


def create_result_rows(result: PlantAnalysisResult) -> list[tuple[str, str, str, str, str]]:
    """Convert result traits into table rows for UI rendering."""

    return [
        (
            trait.label,
            " / ".join(trait.source_views),
            trait.display_value,
            trait.unit,
            format_status_label(trait.status),
        )
        for trait in result.traits
    ]


def format_status_label(status: str) -> str:
    """Convert an internal status code into a UI-friendly label."""

    return STATUS_LABELS.get(status, status)


def _build_placeholder_traits() -> list[TraitResult]:
    """Create trait placeholders for the current analysis stage."""

    return [
        TraitResult(
            key=spec.key,
            label=spec.label,
            source_views=spec.source_views,
            unit=spec.unit,
        )
        for spec in TRAIT_SPECS
    ]


def _resolve_image_loader(result: PlantAnalysisResult, emit_log: LogCallback | None) -> ImageLoader | None:
    """Resolve the default image loader if dependencies are available."""

    try:
        from core.image_io import load_image
    except ModuleNotFoundError as error:
        result.status = "dependency_error"
        result.message = f"图像读取依赖不可用: {error}"
        result.errors.append(result.message)
        _mark_all_views_failed(result, "图像读取依赖不可用，请先安装 OpenCV。")
        _emit(emit_log, result.message)
        return None

    return load_image


def _resolve_top_segmenter(result: PlantAnalysisResult, emit_log: LogCallback | None) -> TopSegmenter | None:
    """Resolve the default TOP segmenter if dependencies are available."""

    try:
        from core.segmentation import segment_top_view_plant
    except ModuleNotFoundError as error:
        result.status = "dependency_error"
        result.message = f"TOP 分割依赖不可用: {error}"
        result.errors.append(result.message)
        _emit(emit_log, result.message)
        return None

    return segment_top_view_plant


def _resolve_front_segmenter(result: PlantAnalysisResult, emit_log: LogCallback | None) -> FrontSegmenter | None:
    """Resolve the default FRONT segmenter if dependencies are available."""

    try:
        from core.segmentation import segment_front_view_plant
    except ModuleNotFoundError as error:
        result.status = "dependency_error"
        result.message = f"FRONT 分割依赖不可用: {error}"
        result.errors.append(result.message)
        _emit(emit_log, result.message)
        return None

    return segment_front_view_plant


def _resolve_image_calibrator(result: PlantAnalysisResult, emit_log: LogCallback | None) -> ImageCalibrator | None:
    """Resolve the default color-card calibrator if dependencies are available."""

    try:
        from core.calibration import calibrate_image_with_color_card
    except ModuleNotFoundError as error:
        result.status = "dependency_error"
        result.message = f"色卡校正依赖不可用: {error}"
        result.errors.append(result.message)
        _emit(emit_log, result.message)
        return None

    return calibrate_image_with_color_card


def _resolve_top_organ_detector(result: PlantAnalysisResult, emit_log: LogCallback | None) -> TopOrganDetector | None:
    """Resolve the packaged YOLO TOP organ detector when models/best.onnx is available."""

    model_path = resolve_default_model_path()
    if not model_path.exists():
        _emit(emit_log, f"TOP YOLO organ model not found, using legacy color detectors: {model_path}")
        return None

    try:
        import onnxruntime  # noqa: F401, PLC0415
    except Exception as error:  # noqa: BLE001
        message = f"TOP YOLO organ detector unavailable, using legacy color detectors: {error}"
        result.errors.append(message)
        _emit(emit_log, message)
        return None

    return detect_top_organs_with_yolo


def _load_group_images(
    group: PlantImageGroup,
    result: PlantAnalysisResult,
    *,
    image_loader: ImageLoader,
    emit_log: LogCallback | None,
) -> dict[str, Any] | None:
    """Load all required views for a group."""

    loaded_images: dict[str, Any] = {}
    has_failure = False

    for view_name, image_path in (
        (VIEW_TOP, group.top_image),
        (VIEW_FRONT_0, group.front_0_image),
        (VIEW_FRONT_180, group.front_180_image),
    ):
        if image_path is None:
            has_failure = True
            result.view_results[view_name] = ViewLoadResult(
                view_name=view_name,
                path=None,
                status="missing",
                message="Required image is missing.",
            )
            continue

        _emit(emit_log, f"读取 {view_name} 图像: {image_path.name}")
        try:
            image = image_loader(image_path)
            shape = tuple(int(value) for value in getattr(image, "shape"))
            loaded_images[view_name] = image
        except Exception as error:  # noqa: BLE001
            has_failure = True
            error_message = f"{view_name} 图像读取失败: {error}"
            result.errors.append(error_message)
            result.view_results[view_name] = ViewLoadResult(
                view_name=view_name,
                path=image_path,
                status="load_failed",
                message=error_message,
            )
            _emit(emit_log, error_message)
            continue

        result.view_results[view_name] = ViewLoadResult(
            view_name=view_name,
            path=image_path,
            status="loaded",
            shape=shape,
            message=f"{view_name} 已读取，尺寸: {shape}",
        )
        _emit(emit_log, f"{view_name} 图像读取完成，尺寸: {shape}")

    if has_failure:
        result.status = "load_failed"
        result.message = "一个或多个输入图像读取失败。"
        _emit(emit_log, "样本组分析终止: 存在图像读取失败。")
        return None

    return loaded_images


def _calibrate_group_images(
    loaded_images: dict[str, Any],
    *,
    result: PlantAnalysisResult,
    emit_log: LogCallback | None,
    image_calibrator: ImageCalibrator,
    calibration_reference: Any | None,
    manual_regions: dict[str, tuple[int, int, int, int]] | None = None,
) -> dict[str, Any]:
    """Run color-card calibration for each input view.
    
    Args:
        manual_regions: Optional dict mapping view names to manual color card regions.
                       Keys should be VIEW_TOP, VIEW_FRONT_0, VIEW_FRONT_180.
    """

    calibrated_images: dict[str, Any] = {}

    for view_name in (VIEW_TOP, VIEW_FRONT_0, VIEW_FRONT_180):
        _emit(emit_log, f"开始执行 {view_name} 色卡检测与校正")
        
        # 获取该视角的手动区域（如果有）
        manual_region = manual_regions.get(view_name) if manual_regions else None
        
        try:
            # 尝试带手动区域参数调用
            if calibration_reference is None:
                try:
                    calibration = image_calibrator(
                        loaded_images[view_name], 
                        view_name=view_name,
                        manual_region=manual_region,
                    )
                except TypeError:
                    # 兼容不支持manual_region的旧实现
                    calibration = image_calibrator(
                        loaded_images[view_name], 
                        view_name=view_name,
                    )
            else:
                try:
                    calibration = image_calibrator(
                        loaded_images[view_name],
                        view_name=view_name,
                        reference=calibration_reference,
                        manual_region=manual_region,
                    )
                except TypeError:
                    # 兼容不同签名的calibrator
                    try:
                        calibration = image_calibrator(
                            loaded_images[view_name],
                            view_name=view_name,
                            reference=calibration_reference,
                        )
                    except TypeError:
                        calibration = image_calibrator(
                            loaded_images[view_name], 
                            view_name=view_name,
                        )
        except Exception as error:  # noqa: BLE001
            calibration = _fallback_calibration_result(
                loaded_images[view_name],
                view_name=view_name,
                message=f"{view_name} 色卡校正异常，已回退到原图: {error}",
            )

        result.calibration_results[view_name] = calibration
        calibrated_images[view_name] = getattr(calibration, "corrected_image", loaded_images[view_name])

        if getattr(calibration, "is_calibrated", False):
            _emit(
                emit_log,
                f"{view_name} 校正完成: mm_per_pixel={calibration.mm_per_pixel:.4f}, "
                f"pixels_per_mm={calibration.pixels_per_mm:.2f}",
            )
        else:
            _emit(emit_log, getattr(calibration, "message", f"{view_name} 未完成色卡校正，已使用原图继续分析。"))

    return calibrated_images


def _fallback_calibration_result(image: np.ndarray, *, view_name: str, message: str) -> Any:
    """Build a fallback calibration-like payload when calibration fails."""

    from types import SimpleNamespace

    return SimpleNamespace(
        status="not_detected",
        message=message,
        view_name=view_name,
        corrected_image=image.copy(),
        corrected_card=None,
        warped_card=None,
        card_corners=None,
        observed_patch_rgb=None,
        reference_patch_rgb=None,
        correction_matrix=None,
        mean_patch_error=None,
        mm_per_pixel=None,
        pixels_per_mm=None,
        card_width_pixels=None,
        card_height_pixels=None,
        search_region=(0, 0, image.shape[1], image.shape[0]),
        debug_images={},
        is_calibrated=False,
    )


def _run_segmentation_step(
    *,
    result: PlantAnalysisResult,
    emit_log: LogCallback | None,
    view_name: str,
    segmenter: Callable[[Any], Any],
    image: Any,
) -> Any | None:
    """Run one segmentation step and normalize failure handling."""

    try:
        segmentation = segmenter(image)
    except Exception as error:  # noqa: BLE001
        result.status = "segmentation_failed"
        result.message = f"{view_name} 分割失败: {error}"
        result.errors.append(result.message)
        _emit(emit_log, result.message)
        return None

    if not getattr(segmentation, "has_foreground", False):
        result.status = "segmentation_failed"
        result.message = f"{view_name} 分割未检测到有效植株区域。"
        result.errors.append(result.message)
        _emit(emit_log, result.message)
        return None

    return segmentation


def _mark_all_views_failed(result: PlantAnalysisResult, message: str) -> None:
    """Set all views to the same failure state."""

    for view_name, view_result in result.view_results.items():
        result.view_results[view_name] = ViewLoadResult(
            view_name=view_name,
            path=view_result.path,
            status="dependency_error",
            message=message,
        )


def _apply_top_trait_measurements(result: PlantAnalysisResult, measurements: Any, *, calibration: Any | None) -> None:
    """Update TOP-derived traits with computed values."""

    trait_map = result.trait_map()
    is_calibrated = bool(getattr(calibration, "is_calibrated", False))

    if is_calibrated:
        from core.calibration import pixel_area_to_cm2, pixels_to_cm

        mm_per_pixel = float(calibration.mm_per_pixel)
        trait_map["leaf_area"].value = round(pixel_area_to_cm2(measurements.leaf_area_pixels, mm_per_pixel), 2)
        trait_map["leaf_area"].unit = "cm^2"
        trait_map["leaf_area"].message = "基于 TOP 色卡尺度校正后的叶面积。"

        trait_map["convex_hull_area"].value = round(
            pixel_area_to_cm2(measurements.convex_hull_area_pixels, mm_per_pixel),
            2,
        )
        trait_map["convex_hull_area"].unit = "cm^2"
        trait_map["convex_hull_area"].message = "基于 TOP 色卡尺度校正后的最小凸包面积。"
        trait_map["canopy_width"].value = round(pixels_to_cm(measurements.canopy_diameter_pixels, mm_per_pixel), 2)
        trait_map["canopy_width"].unit = "cm"
        trait_map["canopy_width"].message = "基于 TOP 最小凸包最远两点距离计算的植株冠径。"
    else:
        trait_map["leaf_area"].value = measurements.leaf_area_pixels
        trait_map["leaf_area"].unit = "px^2"
        trait_map["leaf_area"].message = "未完成 TOP 尺度校正，当前为像素面积。"

        trait_map["convex_hull_area"].value = round(measurements.convex_hull_area_pixels, 2)
        trait_map["convex_hull_area"].unit = "px^2"
        trait_map["convex_hull_area"].message = "未完成 TOP 尺度校正，当前为像素面积。"
        trait_map["canopy_width"].value = round(measurements.canopy_diameter_pixels, 2)
        trait_map["canopy_width"].unit = "px"
        trait_map["canopy_width"].message = "未完成 TOP 尺度校正，当前为 TOP 最小凸包最远两点的像素距离。"

    trait_map["leaf_area"].status = "computed"
    trait_map["convex_hull_area"].status = "computed"
    trait_map["canopy_width"].status = "computed"

    trait_map["greenness"].value = round(measurements.greenness_exg_mean, 2)
    trait_map["greenness"].unit = "ExG"
    trait_map["greenness"].status = "computed"
    trait_map["greenness"].message = (
        "基于色卡校正后图像计算的平均 Excess Green 指数。"
        if is_calibrated
        else "基于原图计算的平均 Excess Green 指数。"
    )


def _apply_top_organ_counts(
    result: PlantAnalysisResult,
    *,
    organ_detection: Any | None,
    flower_detection: Any,
    fruit_detection: Any,
    calibration: Any | None,
) -> None:
    """Update TOP flower, flower-bud, and fruit count traits."""

    trait_map = result.trait_map()
    image_note = (
        "基于色卡校正后的 TOP 图像"
        if bool(getattr(calibration, "is_calibrated", False))
        else "基于原始 TOP 图像，可信度较低"
    )

    if organ_detection is not None:
        trait_map["flower_count"].value = int(getattr(organ_detection, "flower_count", 0))
        trait_map["flower_count"].unit = "count"
        trait_map["flower_count"].status = "computed"
        trait_map["flower_count"].message = f"{image_note}使用 YOLOv8 统计可见开放花朵数量。"

        trait_map["flower_bud_count"].value = int(getattr(organ_detection, "flower_bud_count", 0))
        trait_map["flower_bud_count"].unit = "count"
        trait_map["flower_bud_count"].status = "computed"
        trait_map["flower_bud_count"].message = f"{image_note}使用 YOLOv8 统计可见花骨朵数量。"

        trait_map["fruit_count"].value = int(getattr(organ_detection, "fruit_count", 0))
        trait_map["fruit_count"].unit = "count"
        trait_map["fruit_count"].status = "computed"
        trait_map["fruit_count"].message = f"{image_note}使用 YOLOv8 统计可见果实数量。"
        return

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

    trait_map["flower_bud_count"].value = None
    trait_map["flower_bud_count"].unit = "count"
    trait_map["flower_bud_count"].status = "pending_algorithm"
    trait_map["flower_bud_count"].message = "花骨朵识别需要 YOLOv8 模型，当前使用传统颜色检测回退。"

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


def _apply_front_trait_measurements(
    result: PlantAnalysisResult,
    measurements: Any,
    *,
    front_0_calibration: Any | None,
    front_180_calibration: Any | None,
) -> None:
    """Update FRONT-derived traits with fused values."""

    trait_map = result.trait_map()
    front_0_ready = bool(getattr(front_0_calibration, "is_calibrated", False))
    front_180_ready = bool(getattr(front_180_calibration, "is_calibrated", False))

    if front_0_ready and front_180_ready:
        from core.calibration import pixel_area_to_cm2, pixels_to_cm

        height_cm = max(
            pixels_to_cm(measurements.front_0.canopy_height_pixels, front_0_calibration.mm_per_pixel),
            pixels_to_cm(measurements.front_180.canopy_height_pixels, front_180_calibration.mm_per_pixel),
        )
        projection_area_cm2 = (
            pixel_area_to_cm2(measurements.front_0.projection_area_pixels, front_0_calibration.mm_per_pixel)
            + pixel_area_to_cm2(measurements.front_180.projection_area_pixels, front_180_calibration.mm_per_pixel)
        ) / 2.0

        trait_map["canopy_height"].value = round(height_cm, 2)
        trait_map["canopy_height"].unit = "cm"
        trait_map["canopy_height"].message = "基于 FRONT-1/FRONT-2 尺度校正后的冠层高度融合结果。"

        trait_map["side_projection_area"].value = round(projection_area_cm2, 2)
        trait_map["side_projection_area"].unit = "cm^2"
        trait_map["side_projection_area"].message = "基于 FRONT-1/FRONT-2 尺度校正后的侧视投影面积平均值。"
    else:
        front_0_metrics = compute_front_view_traits(result.front_segmentations[VIEW_FRONT_0])
        front_180_metrics = compute_front_view_traits(result.front_segmentations[VIEW_FRONT_180])

        trait_map["canopy_height"].value = measurements.fused_canopy_height_pixels
        trait_map["canopy_height"].unit = "px"
        trait_map["canopy_height"].message = (
            f"未完成双 FRONT 尺度校正，当前为像素高度。"
            f" FRONT-1={front_0_metrics.canopy_height_pixels}px, FRONT-2={front_180_metrics.canopy_height_pixels}px。"
        )

        trait_map["side_projection_area"].value = round(measurements.fused_projection_area_pixels, 2)
        trait_map["side_projection_area"].unit = "px^2"
        trait_map["side_projection_area"].message = "未完成双 FRONT 尺度校正，当前为像素面积。"

    trait_map["canopy_height"].status = "computed"
    trait_map["side_projection_area"].status = "computed"


def _export_calibration_debug_artifacts(
    *,
    sample_id: str,
    calibration_results: dict[str, Any],
    output_root: Path,
) -> dict[str, list[Path]]:
    """Save color-correction and scale-calibration debug images."""

    calibration_steps: list[tuple[str, np.ndarray]] = []

    for view_name in (VIEW_TOP, VIEW_FRONT_0, VIEW_FRONT_180):
        calibration = calibration_results.get(view_name)
        if calibration is None:
            continue

        debug_images = getattr(calibration, "debug_images", {})
        
        # 定义需要导出的调试图像的顺序
        # 首先是基本信息
        priority_keys = [
            "search_region_overlay",
            "detection_mask",
        ]
        
        # 掩膜检测中间步骤（自动包含所有以特定前缀开头的键）
        mask_keys = sorted([k for k in debug_images.keys() if any(
            k.startswith(p) for p in ["roi_", "hsv_", "color_mask", "bright_mask", "dark_mask", "combined_", "cleaned_"]
        )])
        
        # 轮廓检测中间步骤
        contour_keys = sorted([k for k in debug_images.keys() if k.startswith("contour_")])
        
        # 模板匹配中间步骤
        template_keys = sorted([k for k in debug_images.keys() if k.startswith("template_")])
        
        # 最终结果
        result_keys = [
            "card_overlay",
            "warped_card",
            "corrected_card",
            "before_after",
        ]
        
        # 按顺序导出
        all_keys = priority_keys + mask_keys + contour_keys + template_keys + result_keys
        for key in all_keys:
            image = debug_images.get(key)
            if image is not None:
                step_name = f"{view_name}_{key}"
                calibration_steps.append((step_name, image))

    return {
        "color_calibration": save_debug_steps(sample_id, "color_calibration", calibration_steps, output_root=output_root)
        if calibration_steps
        else [],
    }


def _export_top_trait_debug_artifacts(
    *,
    sample_id: str,
    original_top: np.ndarray,
    top_segmentation: Any,
    output_root: Path,
) -> dict[str, list[Path]]:
    """Save trait-specific TOP debug images for the current sample."""

    debug_images = getattr(top_segmentation, "debug_images", {})
    mask = top_segmentation.mask
    masked_top = create_masked_color_image(original_top, mask)
    leaf_focus_view = create_gray_background_focus_image(original_top, mask, gray_value=96)

    green_channel = masked_top[:, :, 1]
    red_channel = masked_top[:, :, 2].astype(np.int16)
    blue_channel = masked_top[:, :, 0].astype(np.int16)
    exg = np.clip(2 * green_channel.astype(np.int16) - red_channel - blue_channel, 0, 255).astype(np.uint8)
    exg[mask == 0] = 0
    exg_heatmap = create_heatmap(exg)
    diameter_overlay = original_top.copy()
    diameter_measurements = compute_top_traits(original_top, top_segmentation)
    if diameter_measurements.canopy_diameter_endpoints is not None:
        import cv2

        point_a, point_b = diameter_measurements.canopy_diameter_endpoints
        cv2.polylines(diameter_overlay, [top_segmentation.convex_hull], True, (255, 255, 0), 3)
        cv2.line(diameter_overlay, point_a, point_b, (0, 0, 255), 4)
        cv2.circle(diameter_overlay, point_a, 8, (255, 255, 255), -1)
        cv2.circle(diameter_overlay, point_b, 8, (255, 255, 255), -1)

    # TOP俯视图分割步骤（中文命名）
    top_segmentation_steps = _collect_debug_steps(
        ("原始图像", original_top),
        ("归一化处理", debug_images.get("normalized")),
        ("去噪处理", debug_images.get("denoised")),
        ("HSV绿色掩码", debug_images.get("hsv_green_mask")),
        ("绿色优势掩码", debug_images.get("green_dominance_mask")),
        ("合并掩码", debug_images.get("combined_mask")),
        ("形态学开运算", debug_images.get("morphology_opened")),
        ("形态学闭运算", debug_images.get("morphology_closed")),
        ("小孔洞填补", debug_images.get("holes_filled_mask")),
        ("形态学清理结果", debug_images.get("cleaned_mask")),
        ("右侧色卡去除结果", debug_images.get("after_right_card_removal")),
        ("顶部花盆边候选", debug_images.get("top_band_candidate_mask")),
        ("顶部花盆边去除区域", debug_images.get("removed_top_pot_band")),
        ("最终分割掩码", debug_images.get("filtered_mask")),
    )

    # 叶面积计算步骤
    leaf_area_steps = _collect_debug_steps(
        ("分割掩码", debug_images.get("filtered_mask")),
        ("叶面积覆盖图", leaf_focus_view),
    )

    # 凸包面积计算步骤
    convex_hull_steps = _collect_debug_steps(
        ("分割掩码", debug_images.get("filtered_mask")),
        ("轮廓提取", top_segmentation.contour_image),
        ("凸包覆盖图", top_segmentation.hull_image),
    )

    # 绿度计算步骤
    greenness_steps = _collect_debug_steps(
        ("分割掩码", debug_images.get("filtered_mask")),
        ("植株区域提取", masked_top),
        ("绿色通道", green_channel),
        ("ExG指数图", exg),
        ("ExG热力图", exg_heatmap),
    )

    canopy_width_steps = _collect_debug_steps(
        ("分割掩码", debug_images.get("filtered_mask")),
        ("凸包覆盖图", top_segmentation.hull_image),
        ("植株冠径", diameter_overlay),
    )

    return {
        "top_segmentation": save_debug_steps(sample_id, "top_segmentation", top_segmentation_steps, output_root=output_root),
        "leaf_area": save_debug_steps(sample_id, "leaf_area", leaf_area_steps, output_root=output_root),
        "convex_hull_area": save_debug_steps(sample_id, "convex_hull_area", convex_hull_steps, output_root=output_root),
        "greenness": save_debug_steps(sample_id, "greenness", greenness_steps, output_root=output_root),
        "canopy_width": save_debug_steps(sample_id, "canopy_width", canopy_width_steps, output_root=output_root),
    }


def _export_front_trait_debug_artifacts(
    *,
    sample_id: str,
    front_0_image: np.ndarray,
    front_180_image: np.ndarray,
    front_0_segmentation: Any,
    front_180_segmentation: Any,
    output_root: Path,
) -> dict[str, list[Path]]:
    """Save FRONT trait debug images for the current sample."""

    front_0_mask = front_0_segmentation.mask
    front_180_mask = front_180_segmentation.mask

    front_0_debug = getattr(front_0_segmentation, "debug_images", {})
    front_180_debug = getattr(front_180_segmentation, "debug_images", {})
    front_segmentation_steps = (
        _build_front_segmentation_debug_steps(
            front_tag="front_1",
            image=front_0_image,
            mask=front_0_mask,
            segmentation=front_0_segmentation,
            debug_images=front_0_debug,
        )
        + _build_front_segmentation_debug_steps(
            front_tag="front_2",
            image=front_180_image,
            mask=front_180_mask,
            segmentation=front_180_segmentation,
            debug_images=front_180_debug,
        )
    )

    canopy_height_steps = _collect_debug_steps(
        ("front_1_filtered_mask", front_0_debug.get("filtered_mask")),
        ("front_1_bounding_box", front_0_segmentation.bounding_box_image),
        ("front_2_filtered_mask", front_180_debug.get("filtered_mask")),
        ("front_2_bounding_box", front_180_segmentation.bounding_box_image),
    )

    side_projection_steps = _collect_debug_steps(
        ("front_1_mask", front_0_mask),
        ("front_1_masked_region", create_masked_color_image(front_0_image, front_0_mask)),
        (
            "front_1_projection_overlay",
            create_gray_background_focus_image(front_0_image, front_0_mask, gray_value=96),
        ),
        ("front_2_mask", front_180_mask),
        ("front_2_masked_region", create_masked_color_image(front_180_image, front_180_mask)),
        (
            "front_2_projection_overlay",
            create_gray_background_focus_image(front_180_image, front_180_mask, gray_value=96),
        ),
    )

    return {
        "front_segmentation": save_debug_steps(sample_id, "front_segmentation", front_segmentation_steps, output_root=output_root),
        "canopy_height": save_debug_steps(sample_id, "canopy_height", canopy_height_steps, output_root=output_root),
        "side_projection_area": save_debug_steps(
            sample_id,
            "side_projection_area",
            side_projection_steps,
            output_root=output_root,
        ),
    }


def _export_top_organ_debug_artifacts(
    *,
    sample_id: str,
    organ_detection: Any | None,
    flower_detection: Any,
    fruit_detection: Any,
    output_root: Path,
) -> dict[str, list[Path]]:
    """Save TOP organ detection debug images for the current sample."""

    if organ_detection is not None:
        organ_debug = getattr(organ_detection, "debug_images", {})
        organ_steps = _collect_debug_steps(
            ("yolo_overlay", getattr(organ_detection, "overlay_image", None)),
            ("yolo_debug_overlay", organ_debug.get("overlay")),
        )
        return {
            "flower_count": save_debug_steps(sample_id, "flower_count", organ_steps, output_root=output_root),
            "flower_bud_count": save_debug_steps(sample_id, "flower_bud_count", organ_steps, output_root=output_root),
            "fruit_count": save_debug_steps(sample_id, "fruit_count", organ_steps, output_root=output_root),
        }

    flower_debug = getattr(flower_detection, "debug_images", {})
    fruit_debug = getattr(fruit_detection, "debug_images", {})

    flower_steps = _collect_debug_steps(
        ("flower_canopy_source", flower_debug.get("canopy_source")),
        ("flower_raw_mask", flower_debug.get("raw_mask")),
        ("flower_cleaned_mask", flower_debug.get("cleaned_mask")),
        ("flower_distance_map", flower_debug.get("distance_map")),
        ("flower_labeled_mask", flower_debug.get("labeled_mask")),
        ("flower_overlay", getattr(flower_detection, "overlay_image", None)),
    )
    fruit_steps = _collect_debug_steps(
        ("fruit_canopy_source", fruit_debug.get("canopy_source")),
        ("fruit_raw_mask", fruit_debug.get("raw_mask")),
        ("fruit_cleaned_mask", fruit_debug.get("cleaned_mask")),
        ("fruit_distance_map", fruit_debug.get("distance_map")),
        ("fruit_labeled_mask", fruit_debug.get("labeled_mask")),
        ("fruit_overlay", getattr(fruit_detection, "overlay_image", None)),
    )
    return {
        "flower_count": save_debug_steps(sample_id, "flower_count", flower_steps, output_root=output_root),
        "flower_bud_count": [],
        "fruit_count": save_debug_steps(sample_id, "fruit_count", fruit_steps, output_root=output_root),
    }

    # FRONT正视图分割步骤（中文命名）
    front_segmentation_steps = _collect_debug_steps(
        ("FRONT-1_原始图像", front_0_image),
        ("FRONT-1_归一化处理", front_0_debug.get("normalized")),
        ("FRONT-1_去噪处理", front_0_debug.get("denoised")),
        ("FRONT-1_HSV绿色掩码", front_0_debug.get("hsv_green_mask")),
        ("FRONT-1_分割掩码", front_0_debug.get("filtered_mask")),
        ("FRONT-2_原始图像", front_180_image),
        ("FRONT-2_归一化处理", front_180_debug.get("normalized")),
        ("FRONT-2_去噪处理", front_180_debug.get("denoised")),
        ("FRONT-2_HSV绿色掩码", front_180_debug.get("hsv_green_mask")),
        ("FRONT-2_分割掩码", front_180_debug.get("filtered_mask")),
    )

    # 冠幅高度计算步骤
    canopy_height_steps = _collect_debug_steps(
        ("FRONT-1_分割掩码", front_0_debug.get("filtered_mask")),
        ("FRONT-1_边界框", front_0_segmentation.bounding_box_image),
        ("FRONT-2_分割掩码", front_180_debug.get("filtered_mask")),
        ("FRONT-2_边界框", front_180_segmentation.bounding_box_image),
    )

    # 侧面投影面积计算步骤
    side_projection_steps = _collect_debug_steps(
        ("FRONT-1_分割掩码", front_0_mask),
        ("FRONT-1_植株区域", create_masked_color_image(front_0_image, front_0_mask)),
        (
            "FRONT-1_投影覆盖图",
            create_gray_background_focus_image(front_0_image, front_0_mask, gray_value=96),
        ),
        ("FRONT-2_分割掩码", front_180_mask),
        ("FRONT-2_植株区域", create_masked_color_image(front_180_image, front_180_mask)),
        (
            "FRONT-2_投影覆盖图",
            create_gray_background_focus_image(front_180_image, front_180_mask, gray_value=96),
        ),
    )

    return {
        "front_segmentation": save_debug_steps(sample_id, "front_segmentation", front_segmentation_steps, output_root=output_root),
        "canopy_height": save_debug_steps(sample_id, "canopy_height", canopy_height_steps, output_root=output_root),
        "side_projection_area": save_debug_steps(
            sample_id,
            "side_projection_area",
            side_projection_steps,
            output_root=output_root,
        ),
    }


def _build_front_segmentation_debug_steps(
    *,
    front_tag: str,
    image: np.ndarray,
    mask: np.ndarray,
    segmentation: Any,
    debug_images: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    """Build a complete FRONT segmentation debug chain for one view."""

    ordered_steps = _collect_debug_steps(
        (f"{front_tag}_original", image),
        (f"{front_tag}_normalized", debug_images.get("normalized")),
        (f"{front_tag}_denoised", debug_images.get("denoised")),
        (f"{front_tag}_hsv_green_mask", debug_images.get("hsv_green_mask")),
        (f"{front_tag}_green_dominance_mask", debug_images.get("green_dominance_mask")),
        (f"{front_tag}_combined_mask", debug_images.get("combined_mask")),
        (f"{front_tag}_morphology_opened", debug_images.get("morphology_opened")),
        (f"{front_tag}_morphology_closed", debug_images.get("morphology_closed")),
        (f"{front_tag}_holes_filled_mask", debug_images.get("holes_filled_mask")),
        (f"{front_tag}_filtered_mask_before_front_rules", debug_images.get("filtered_mask_before_front_rules")),
        (f"{front_tag}_front_component_removed_mask", debug_images.get("front_component_removed_mask")),
        (f"{front_tag}_filtered_mask", debug_images.get("filtered_mask")),
        (f"{front_tag}_mask_overlay", create_mask_overlay(image, mask, alpha=0.55)),
        (f"{front_tag}_masked_region", create_masked_color_image(image, mask)),
        (f"{front_tag}_projection_overlay", create_gray_background_focus_image(image, mask, gray_value=96)),
        (f"{front_tag}_contour_overlay", getattr(segmentation, "contour_image", None)),
        (f"{front_tag}_bounding_box", getattr(segmentation, "bounding_box_image", None)),
    )
    ordered_steps.append(
        (
            f"{front_tag}_process_montage",
            create_debug_montage([debug_image for _, debug_image in ordered_steps], columns=3, tile_size=(300, 220)),
        )
    )
    return ordered_steps


def _build_summary_message(result: PlantAnalysisResult) -> str:
    """Build the final pipeline summary message."""

    trait_map = result.trait_map()
    calibration_done = sum(
        1 for item in result.calibration_results.values() if bool(getattr(item, "is_calibrated", False))
    )

    return (
        f"分析完成。色卡校正成功 {calibration_done}/3 视角。"
        f" 叶面积={trait_map['leaf_area'].display_value} {trait_map['leaf_area'].unit},"
        f" 凸包面积={trait_map['convex_hull_area'].display_value} {trait_map['convex_hull_area'].unit},"
        f" 绿色程度={trait_map['greenness'].display_value} {trait_map['greenness'].unit},"
        f" 冠层高度={trait_map['canopy_height'].display_value} {trait_map['canopy_height'].unit},"
        f" 植株冠径={trait_map['canopy_width'].display_value} {trait_map['canopy_width'].unit},"
        f" 侧视投影面积={trait_map['side_projection_area'].display_value} {trait_map['side_projection_area'].unit},"
        f" 花朵数={trait_map['flower_count'].display_value} {trait_map['flower_count'].unit},"
        f" 花骨朵数={trait_map['flower_bud_count'].display_value} {trait_map['flower_bud_count'].unit},"
        f" 果实数={trait_map['fruit_count'].display_value} {trait_map['fruit_count'].unit}。"
    )


def _collect_debug_steps(*steps: tuple[str, Any]) -> list[tuple[str, Any]]:
    """Filter debug step tuples, dropping any entry whose image value is None."""

    return [(name, image) for name, image in steps if image is not None]


def _emit(callback: LogCallback | None, message: str) -> None:
    """Emit a pipeline log message when a callback is provided."""

    if callback is not None:
        callback(message)
