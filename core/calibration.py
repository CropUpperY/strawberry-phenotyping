"""Color-card based image calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-missing environments
    cv2 = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    Image = ImageDraw = ImageFont = None


def _put_chinese_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int = 24,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Use PIL to render Chinese text on an OpenCV BGR image."""
    if Image is None:
        # PIL not available, fallback to cv2 (will show ? for Chinese)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return image

    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Try common Chinese fonts on Windows / Linux / macOS
    font = None
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",       # Windows: Microsoft YaHei
        "C:/Windows/Fonts/simhei.ttf",      # Windows: SimHei
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    # Convert BGR color to RGB for PIL
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


@dataclass(frozen=True, slots=True)
class ColorCardReference:
    """Reference definition for a 24-patch color card."""

    patch_rgb: np.ndarray
    card_width_mm: float
    card_height_mm: float
    patch_width_mm: float
    patch_height_mm: float
    rows: int = 4
    cols: int = 6
    canonical_width: int = 360
    canonical_height: int = 240


@dataclass(slots=True)
class ImageCalibrationResult:
    """Color and scale calibration result for one image."""

    status: str
    message: str
    view_name: str
    corrected_image: np.ndarray
    corrected_card: np.ndarray | None
    warped_card: np.ndarray | None
    card_corners: np.ndarray | None
    observed_patch_rgb: np.ndarray | None
    reference_patch_rgb: np.ndarray | None
    correction_matrix: np.ndarray | None
    mean_patch_error: float | None
    mm_per_pixel: float | None
    pixels_per_mm: float | None
    card_width_pixels: float | None
    card_height_pixels: float | None
    search_region: tuple[int, int, int, int]
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def is_calibrated(self) -> bool:
        """Return whether the image was calibrated successfully."""

        return self.status == "calibrated" and self.mm_per_pixel is not None


REFERENCE_PATCH_RGB = np.array(
    [
        [115, 82, 68],
        [194, 150, 130],
        [98, 122, 157],
        [87, 108, 67],
        [133, 128, 177],
        [103, 189, 170],
        [214, 126, 44],
        [80, 91, 166],
        [193, 90, 99],
        [94, 60, 108],
        [157, 188, 64],
        [224, 163, 46],
        [56, 61, 150],
        [70, 148, 73],
        [175, 54, 60],
        [231, 199, 31],
        [187, 86, 149],
        [8, 133, 161],
        [243, 243, 242],
        [200, 200, 200],
        [160, 160, 160],
        [122, 122, 121],
        [85, 85, 85],
        [52, 52, 52],
    ],
    dtype=np.float32,
)

DEFAULT_COLOR_CARD_REFERENCE = ColorCardReference(
    patch_rgb=REFERENCE_PATCH_RGB,
    card_width_mm=63.5,
    card_height_mm=109.0,
    patch_width_mm=15.0,
    patch_height_mm=15.0,
)


def create_color_card_reference(
    *,
    patch_width_cm: float,
    patch_height_cm: float,
    patch_rgb: np.ndarray | None = None,
    card_width_cm: float | None = None,
    card_height_cm: float | None = None,
) -> ColorCardReference:
    """Create a color-card reference from user-provided patch dimensions."""

    if patch_width_cm <= 0 or patch_height_cm <= 0:
        raise ValueError("patch_width_cm and patch_height_cm must be greater than 0")

    return ColorCardReference(
        patch_rgb=REFERENCE_PATCH_RGB.copy() if patch_rgb is None else patch_rgb.astype(np.float32),
        card_width_mm=DEFAULT_COLOR_CARD_REFERENCE.card_width_mm if card_width_cm is None else float(card_width_cm) * 10.0,
        card_height_mm=DEFAULT_COLOR_CARD_REFERENCE.card_height_mm if card_height_cm is None else float(card_height_cm) * 10.0,
        patch_width_mm=float(patch_width_cm) * 10.0,
        patch_height_mm=float(patch_height_cm) * 10.0,
    )


@dataclass(slots=True)
class _CardCandidate:
    """Internal candidate payload for card detection."""

    corners: np.ndarray
    warped_card: np.ndarray
    corrected_card: np.ndarray
    observed_patch_rgb: np.ndarray
    correction_matrix: np.ndarray
    mean_patch_error: float
    patch_polygons: list[np.ndarray] | None = None
    variant_index: int = -1
    rectification_inverse: np.ndarray | None = None
    patch_width_pixels: float | None = None
    patch_height_pixels: float | None = None


@dataclass(slots=True)
class _PatchBox:
    """Detected patch candidate in the warped card image."""

    x: int
    y: int
    w: int
    h: int
    area: float
    center_x: float
    center_y: float


@dataclass(slots=True)
class _PatchGridModel:
    """Affine grid model for the 4x6 color patch layout."""

    origin: np.ndarray
    col_vec: np.ndarray
    row_vec: np.ndarray
    half_col_vec: np.ndarray
    half_row_vec: np.ndarray


@dataclass(slots=True)
class _AdaptivePatchExtraction:
    """Adaptive patch extraction payload for scoring and scale estimation."""

    observed_rgb: np.ndarray
    patch_polygons: list[np.ndarray]
    candidate_count: int
    assignments: dict[tuple[int, int], _PatchBox]


def _select_best_candidate(first: _CardCandidate | None, second: _CardCandidate | None) -> _CardCandidate | None:
    """Return the better candidate according to patch reconstruction error."""

    if first is None:
        return second
    if second is None:
        return first
    return first if first.mean_patch_error <= second.mean_patch_error else second


def calibrate_image_with_color_card(
    image: np.ndarray,
    *,
    view_name: str,
    reference: ColorCardReference = DEFAULT_COLOR_CARD_REFERENCE,
    manual_region: tuple[int, int, int, int] | None = None,
) -> ImageCalibrationResult:
    """Detect a 24-patch color card, correct image colors, and estimate scale.
    
    Args:
        image: BGR input image.
        view_name: View identifier (TOP, FRONT-1, FRONT-2).
        reference: Color card reference with physical dimensions.
        manual_region: Optional manually-selected region (x, y, width, height).
                      When provided, skips auto-detection and uses this region.
    """

    _require_cv2()
    _validate_color_image(image)

    debug_images: dict[str, np.ndarray] = {}

    # 如果提供了手动区域，优先使用手动区域
    if manual_region is not None:
        return _calibrate_with_manual_region(
            image, view_name=view_name, reference=reference, manual_region=manual_region
        )

    # 自动检测流程
    search_region = _default_search_region(image, view_name=view_name)
    detection_mask, mask_debug_images = _build_detection_mask(image, search_region)
    contour_candidate, contour_debug = _detect_best_candidate(image, detection_mask, search_region, reference)
    template_candidate, template_debug = _detect_template_candidate(image, view_name=view_name, reference=reference)
    candidate = _select_best_candidate(contour_candidate, template_candidate)

    # 构建详细的调试图像
    debug_images = {
        "search_region_overlay": _draw_search_region_overlay(image, search_region),
        "detection_mask": detection_mask,
    }
    # 添加掩膜检测的中间步骤
    debug_images.update(mask_debug_images)
    # 添加轮廓检测的中间步骤
    debug_images.update(contour_debug)
    # 添加模板匹配的中间步骤
    debug_images.update(template_debug)

    if candidate is None:
        return ImageCalibrationResult(
            status="not_detected",
            message=f"{view_name} 未检测到有效色卡区域。",
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
            search_region=search_region,
            debug_images=debug_images,
        )

    corrected_image = apply_color_matrix_to_image(image, candidate.correction_matrix)
    corrected_card = apply_color_matrix_to_image(candidate.warped_card, candidate.correction_matrix)
    card_width_pixels, card_height_pixels = _card_dimensions_in_pixels(candidate.corners)
    if candidate.patch_width_pixels is not None and candidate.patch_height_pixels is not None:
        mm_per_pixel = (
            (reference.patch_width_mm / candidate.patch_width_pixels) +
            (reference.patch_height_mm / candidate.patch_height_pixels)
        ) / 2.0
    else:
        mm_per_pixel = (
            (reference.card_width_mm / card_width_pixels) + (reference.card_height_mm / card_height_pixels)
        ) / 2.0
    pixels_per_mm = 1.0 / mm_per_pixel

    # 添加成功检测时的额外调试图像
    debug_images.update({
        "card_overlay": _draw_card_overlay(image, candidate.corners),
        "warped_card": candidate.warped_card,
        "corrected_card": corrected_card,
        "before_after": np.hstack([image, corrected_image]),
    })

    return ImageCalibrationResult(
        status="calibrated",
        message=f"{view_name} 色卡检测与颜色/尺度校正完成（基于色块尺度）。",
        view_name=view_name,
        corrected_image=corrected_image,
        corrected_card=corrected_card,
        warped_card=candidate.warped_card,
        card_corners=candidate.corners,
        observed_patch_rgb=candidate.observed_patch_rgb,
        reference_patch_rgb=reference.patch_rgb.copy(),
        correction_matrix=candidate.correction_matrix,
        mean_patch_error=candidate.mean_patch_error,
        mm_per_pixel=mm_per_pixel,
        pixels_per_mm=pixels_per_mm,
        card_width_pixels=card_width_pixels,
        card_height_pixels=card_height_pixels,
        search_region=search_region,
        debug_images=debug_images,
    )


def _calibrate_with_manual_region(
    image: np.ndarray,
    *,
    view_name: str,
    reference: ColorCardReference,
    manual_region: tuple[int, int, int, int],
) -> ImageCalibrationResult:
    """在手动选择的区域内执行精确色卡定位和校准。
    
    用户手动框选只是给出大致范围，此函数在该区域内进行：
    1. 形态学处理构建检测掩膜
    2. 轮廓检测找到精确的色卡边界
    3. 如果轮廓检测失败，尝试模板匹配
    4. 使用找到的精确边界进行颜色校正和尺度标定
    """
    _require_cv2()
    
    x, y, w, h = manual_region
    debug_images: dict[str, np.ndarray] = {}
    
    # 绘制手动选择的区域
    region_overlay = image.copy()
    cv2.rectangle(region_overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
    region_overlay = _put_chinese_text(region_overlay, "手动选择区域", (x, max(0, y - 30)), color=(0, 255, 0))
    debug_images["manual_region_overlay"] = region_overlay
    
    # 提取手动选择的区域
    roi = image[y:y+h, x:x+w]
    debug_images["manual_region_roi"] = roi
    
    # 在ROI内构建检测掩膜（形态学处理）
    roi_search_region = (0, 0, w, h)
    roi_mask, mask_debug = _build_detection_mask(roi, roi_search_region)
    # 添加掩膜调试图像（带前缀避免覆盖）
    for key, img in mask_debug.items():
        debug_images[f"roi_{key}"] = img
    
    # 在ROI内进行轮廓检测，找到色卡的精确边界
    # 用户已手动框选，说明色卡确实在该区域内，放宽误差阈值
    roi_contour_candidate, contour_debug = _detect_best_candidate(
        roi, roi_mask[0:h, 0:w], roi_search_region, reference,
        error_threshold=5000.0,
    )
    for key, img in contour_debug.items():
        debug_images[f"roi_{key}"] = img
    
    candidate = None
    refined_corners = None
    
    if roi_contour_candidate is not None:
        # 轮廓检测成功，将ROI内坐标转换为全图坐标
        refined_corners = roi_contour_candidate.corners.copy()
        refined_corners[:, 0] += x
        refined_corners[:, 1] += y
        candidate = roi_contour_candidate
    else:
        # 轮廓检测失败，尝试在ROI内进行模板匹配
        roi_template_candidate, template_debug = _detect_template_candidate(
            roi, view_name=view_name, reference=reference
        )
        for key, img in template_debug.items():
            debug_images[f"roi_{key}"] = img
        
        if roi_template_candidate is not None:
            # 模板匹配成功，转换坐标
            refined_corners = roi_template_candidate.corners.copy()
            refined_corners[:, 0] += x
            refined_corners[:, 1] += y
            candidate = roi_template_candidate
    
    # 如果ROI内精确定位都失败了，回退到使用原始矩形边界
    if candidate is None or refined_corners is None:
        # 使用原始手动选择的矩形边界
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ], dtype=np.float32)
        
        # 透视变换到标准色卡尺寸
        warped_card = _warp_card(image, corners, reference)
        debug_images["manual_warped_card"] = warped_card
        
        # 评估色卡候选
        candidate, score_debug = _score_card_candidate(warped_card, corners, reference)
        debug_images.update(score_debug)
        refined_corners = corners
    
    # 绘制精确定位后的边界
    refined_overlay = image.copy()
    if refined_corners is not None:
        corners_int = refined_corners.astype(np.int32)
        cv2.polylines(refined_overlay, [corners_int], True, (255, 0, 255), 3)
        refined_overlay = _put_chinese_text(
            refined_overlay, "精确定位边界",
            (int(refined_corners[0, 0]), max(0, int(refined_corners[0, 1]) - 30)),
            color=(255, 0, 255),
        )
    debug_images["refined_boundary"] = refined_overlay
    
    if candidate is None:
        return ImageCalibrationResult(
            status="not_detected",
            message=f"{view_name} 手动选择区域内无法识别有效色卡。",
            view_name=view_name,
            corrected_image=image.copy(),
            corrected_card=None,
            warped_card=None,
            card_corners=refined_corners,
            observed_patch_rgb=None,
            reference_patch_rgb=None,
            correction_matrix=None,
            mean_patch_error=None,
            mm_per_pixel=None,
            pixels_per_mm=None,
            card_width_pixels=float(w),
            card_height_pixels=float(h),
            search_region=manual_region,
            debug_images=debug_images,
        )
    
    # 应用颜色校正
    corrected_image = apply_color_matrix_to_image(image, candidate.correction_matrix)
    corrected_card = apply_color_matrix_to_image(candidate.warped_card, candidate.correction_matrix)
    
    # 计算尺度（优先使用色块尺寸，回退到整卡边界）
    card_width_pixels, card_height_pixels = _card_dimensions_in_pixels(refined_corners)
    if candidate.patch_width_pixels is not None and candidate.patch_height_pixels is not None:
        mm_per_pixel = (
            (reference.patch_width_mm / candidate.patch_width_pixels) +
            (reference.patch_height_mm / candidate.patch_height_pixels)
        ) / 2.0
    else:
        mm_per_pixel = (
            (reference.card_width_mm / card_width_pixels) + (reference.card_height_mm / card_height_pixels)
        ) / 2.0
    pixels_per_mm = 1.0 / mm_per_pixel
    
    # 添加成功检测的调试图像
    debug_images.update({
        "manual_corrected_card": corrected_card,
        "manual_before_after": np.hstack([image, corrected_image]),
    })
    
    return ImageCalibrationResult(
        status="calibrated",
        message=f"{view_name} 在手动选择区域内完成精确定位和颜色/尺度校正（基于色块尺度）。",
        view_name=view_name,
        corrected_image=corrected_image,
        corrected_card=corrected_card,
        warped_card=candidate.warped_card,
        card_corners=refined_corners,
        observed_patch_rgb=candidate.observed_patch_rgb,
        reference_patch_rgb=reference.patch_rgb.copy(),
        correction_matrix=candidate.correction_matrix,
        mean_patch_error=candidate.mean_patch_error,
        mm_per_pixel=mm_per_pixel,
        pixels_per_mm=pixels_per_mm,
        card_width_pixels=card_width_pixels,
        card_height_pixels=card_height_pixels,
        search_region=manual_region,
        debug_images=debug_images,
    )


def apply_color_matrix_to_image(image: np.ndarray, correction_matrix: np.ndarray) -> np.ndarray:
    """Apply an affine RGB color correction matrix to a BGR image."""

    _validate_color_image(image)
    rgb = image[:, :, ::-1].astype(np.float32)
    corrected_rgb = _apply_color_matrix_to_rgb(rgb.reshape(-1, 3), correction_matrix).reshape(rgb.shape)
    # 关键：必须clip到0-255范围，否则会溢出导致颜色失真
    corrected_rgb = np.clip(corrected_rgb, 0, 255)
    return corrected_rgb[:, :, ::-1].astype(np.uint8)


def pixels_to_cm(length_pixels: float, mm_per_pixel: float) -> float:
    """Convert pixels to centimeters."""

    return (float(length_pixels) * float(mm_per_pixel)) / 10.0


def pixel_area_to_cm2(area_pixels: float, mm_per_pixel: float) -> float:
    """Convert pixel area to square centimeters."""

    return (float(area_pixels) * float(mm_per_pixel) * float(mm_per_pixel)) / 100.0


def _detect_best_candidate(
    image: np.ndarray,
    detection_mask: np.ndarray,
    search_region: tuple[int, int, int, int],
    reference: ColorCardReference,
    *,
    error_threshold: float = 1800.0,
) -> tuple[_CardCandidate | None, dict[str, np.ndarray]]:
    """Find the best color-card candidate inside the search region.
    
    Args:
        error_threshold: Maximum mean_patch_error to accept a candidate.
                        Use higher values (e.g. 5000) for manual ROI where
                        the user has confirmed the card is present.
    
    Returns:
        A tuple of (best_candidate, debug_images).
    """
    _require_cv2()
    x0, y0, width, height = search_region
    roi_mask = detection_mask[y0 : y0 + height, x0 : x0 + width]
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_images: dict[str, np.ndarray] = {}
    
    # 绘制所有检测到的轮廓
    contour_vis = image.copy()
    cv2.rectangle(contour_vis, (x0, y0), (x0 + width, y0 + height), (255, 255, 0), 2)
    
    candidates: list[tuple[_CardCandidate, np.ndarray, dict[str, np.ndarray]]] = []
    roi_area = width * height

    for i, contour in enumerate(contours):
        contour_area = float(cv2.contourArea(contour))
        if contour_area < roi_area * 0.008:
            continue

        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rect[1]
        if rect_width <= 1 or rect_height <= 1:
            continue

        aspect_ratio = min(rect_width, rect_height) / max(rect_width, rect_height)
        
        # 绘制所有符合面积条件的轮廓（不管宽高比）
        box = cv2.boxPoints(rect)
        box[:, 0] += x0
        box[:, 1] += y0
        box_int = box.astype(np.int32)
        
        # 用不同颜色标记是否通过宽高比筛选
        if aspect_ratio < 0.45 or aspect_ratio > 0.82:
            cv2.drawContours(contour_vis, [box_int], 0, (0, 0, 255), 2)  # 红色：宽高比不符合
            continue
        
        cv2.drawContours(contour_vis, [box_int], 0, (0, 255, 0), 2)  # 绿色：通过筛选
        
        corners = _order_points(box.astype(np.float32))
        warped_card = _warp_card(image, corners, reference)
        candidate, score_debug = _score_card_candidate(warped_card, corners, reference)
        if candidate is not None:
            candidates.append((candidate, box_int, score_debug))

    debug_images["contour_所有候选轮廓"] = contour_vis

    if not candidates:
        return None, debug_images

    candidates.sort(key=lambda item: item[0].mean_patch_error)
    best_candidate, best_box, best_score_debug = candidates[0]
    
    # 添加最佳候选的变体对比调试图
    debug_images.update(best_score_debug)
    
    # 绘制最佳候选
    best_vis = image.copy()
    cv2.drawContours(best_vis, [best_box], 0, (0, 255, 255), 3)
    debug_images["contour_最佳候选"] = best_vis
    
    if best_candidate.mean_patch_error > error_threshold:
        return None, debug_images
    return best_candidate, debug_images


def _detect_template_candidate(
    image: np.ndarray,
    *,
    view_name: str,
    reference: ColorCardReference,
) -> tuple[_CardCandidate | None, dict[str, np.ndarray]]:
    """Detect the color card via template matching inside a view-specific ROI.
    
    Returns:
        A tuple of (candidate, debug_images).
    """
    _require_cv2()
    match_result = _match_template_bbox(image, view_name=view_name, reference=reference)
    
    debug_images: dict[str, np.ndarray] = {}
    
    if match_result is None:
        debug_images["template_匹配失败"] = image.copy()
        return None, debug_images

    corners, match_score, template_debug = match_result
    debug_images.update(template_debug)
    
    # 绘制模板匹配结果
    match_vis = image.copy()
    corners_int = corners.astype(np.int32)
    cv2.polylines(match_vis, [corners_int], True, (255, 0, 255), 2)
    cv2.putText(match_vis, f"score={match_score:.3f}", (int(corners[0, 0]), int(corners[0, 1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    debug_images["template_匹配结果"] = match_vis
    
    warped_card = _warp_card(image, corners, reference)
    debug_images["template_透视校正"] = warped_card
    
    candidate, score_debug = _score_card_candidate(warped_card, corners, reference)
    debug_images.update(score_debug)
    if candidate is None:
        debug_images["template_评分失败_无有效候选"] = warped_card
        return None, debug_images
    if match_score < 0.33:
        debug_images["template_评分失败_分数过低"] = warped_card
        return None, debug_images
    if candidate.mean_patch_error > 3200.0:
        debug_images["template_评分失败_误差过大"] = warped_card
        return None, debug_images
    return candidate, debug_images


def _score_card_candidate(
    warped_card: np.ndarray,
    corners: np.ndarray,
    reference: ColorCardReference,
) -> tuple[_CardCandidate | None, dict[str, np.ndarray]]:
    """Score one warped card candidate against the reference patch colors.

    Returns:
        A tuple of (best_candidate, debug_images). debug_images contains
        a variant comparison grid showing all 8 orientations with errors.
    """
    debug_images: dict[str, np.ndarray] = {}

    # 保存原始透视校正图（变体旋转之前）
    debug_images["variant_原始透视校正"] = warped_card.copy()

    best_error: float | None = None
    best_patch_rgb: np.ndarray | None = None
    best_matrix: np.ndarray | None = None
    best_variant: np.ndarray | None = None
    best_patch_polygons: list[np.ndarray] | None = None
    best_rectification_inverse: np.ndarray | None = None
    best_idx: int = -1

    variants = _generate_card_variants(warped_card, reference)
    variant_results: list[tuple[int, float | None, np.ndarray, str]] = []

    for i, variant in enumerate(variants):
        adaptive = _extract_patch_rgb_adaptive(variant, reference)
        scoring_image = variant
        scoring_rgb: np.ndarray | None = None
        scoring_polygons: list[np.ndarray] | None = None
        scoring_rectification_inverse: np.ndarray | None = None
        variant_status = "原始"
        rectification_attempted = False
        rectification_succeeded = False

        candidate_choices: list[tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray | None]] = []
        if adaptive is not None:
            candidate_choices.append((variant, adaptive.observed_rgb, adaptive.patch_polygons, None))
            rectification_attempted = True
            rectified_variant, rectification_inverse = _rectify_variant_by_patch_grid(variant, adaptive.assignments, reference)
            if rectified_variant is not None and rectification_inverse is not None:
                rectified_adaptive = _extract_patch_rgb_adaptive(rectified_variant, reference)
                if rectified_adaptive is not None:
                    rectification_succeeded = True
                    candidate_choices.append(
                        (
                            rectified_variant,
                            rectified_adaptive.observed_rgb,
                            rectified_adaptive.patch_polygons,
                            rectification_inverse,
                        )
                    )

        if not candidate_choices:
            uniform_polygons = _build_uniform_patch_polygons(variant, reference)
            candidate_choices.append((variant, _extract_patch_rgb(variant, reference), uniform_polygons, None))

        variant_best_error: float | None = None
        for candidate_image, candidate_rgb, candidate_polygons, rectification_inverse in candidate_choices:
            if candidate_rgb.std() < 10:
                continue
            correction_matrix = _fit_color_matrix(candidate_rgb, reference.patch_rgb)
            corrected_patch_rgb = _apply_color_matrix_to_rgb(candidate_rgb, correction_matrix)
            error = float(np.mean((corrected_patch_rgb - reference.patch_rgb) ** 2))

            if variant_best_error is None or error < variant_best_error:
                variant_best_error = error
                scoring_image = candidate_image
                scoring_rgb = candidate_rgb
                scoring_polygons = candidate_polygons
                scoring_rectification_inverse = rectification_inverse

        if variant_best_error is None or scoring_rgb is None or scoring_polygons is None:
            if rectification_attempted and not rectification_succeeded:
                variant_status = "二次校正失败"
            elif adaptive is None:
                variant_status = "仅原始/回退"
            variant_results.append((i, None, variant, variant_status))
            continue

        correction_matrix = _fit_color_matrix(scoring_rgb, reference.patch_rgb)
        if scoring_rectification_inverse is not None:
            variant_status = "二次校正"
        elif rectification_succeeded:
            variant_status = "原始更优"
        elif adaptive is None:
            variant_status = "仅原始/回退"
        variant_results.append((i, variant_best_error, scoring_image, variant_status))

        if best_error is None or variant_best_error < best_error:
            best_error = variant_best_error
            best_patch_rgb = scoring_rgb
            best_matrix = correction_matrix
            best_variant = scoring_image
            best_patch_polygons = scoring_polygons
            best_rectification_inverse = scoring_rectification_inverse
            best_idx = i

    # 构建8变体对比网格图
    if variant_results:
        debug_images["variant_变体对比"] = _build_variant_grid(variant_results, best_idx)

    if best_error is None or best_patch_rgb is None or best_matrix is None or best_variant is None:
        return None, debug_images

    # 保存最佳变体（旋转/翻转后方向正确的色卡）
    debug_images["variant_最佳变体"] = best_variant.copy()
    if best_rectification_inverse is not None:
        debug_images["variant_二次网格校正"] = best_variant.copy()

    # 生成取色位置标注图：在最佳变体上标注24个取色区域和对应的参考色
    debug_images["variant_取色位置验证"] = _build_patch_extraction_debug(
        best_variant, best_patch_rgb, reference, patch_polygons=best_patch_polygons
    )

    patch_width_pixels, patch_height_pixels = _estimate_patch_scale_in_original_image(
        corners,
        warped_card.shape,
        best_patch_polygons,
        best_idx,
        reference,
        rectification_inverse=best_rectification_inverse,
    )

    return _CardCandidate(
        corners=corners,
        warped_card=best_variant,
        corrected_card=apply_color_matrix_to_image(best_variant, best_matrix),
        observed_patch_rgb=best_patch_rgb,
        correction_matrix=best_matrix,
        mean_patch_error=best_error,
        patch_polygons=best_patch_polygons,
        variant_index=best_idx,
        rectification_inverse=best_rectification_inverse,
        patch_width_pixels=patch_width_pixels,
        patch_height_pixels=patch_height_pixels,
    ), debug_images


def _build_variant_grid(
    variant_results: list[tuple[int, float | None, np.ndarray, str]],
    best_idx: int,
) -> np.ndarray:
    """Build a 2x4 grid showing all 8 card orientation variants with error scores."""
    _require_cv2()

    cell_w, cell_h = 360, 240
    label_h = 40
    cols, rows = 4, 2
    grid_w = cols * cell_w
    grid_h = rows * (cell_h + label_h)
    grid = np.full((grid_h, grid_w, 3), 40, dtype=np.uint8)

    labels = [
        "原始", "逆时针90°", "180°", "顺时针90°",
        "镜像", "镜像+逆90°", "镜像+180°", "镜像+顺90°",
    ]

    for idx, error, variant, status in variant_results:
        row = idx // cols
        col = idx % cols
        x0 = col * cell_w
        y0 = row * (cell_h + label_h)

        resized = cv2.resize(variant, (cell_w, cell_h))

        # 最佳变体用绿色边框高亮
        if idx == best_idx:
            cv2.rectangle(resized, (0, 0), (cell_w - 1, cell_h - 1), (0, 255, 0), 4)

        grid[y0:y0 + cell_h, x0:x0 + cell_w] = resized

        # 标签文字
        label_y = y0 + cell_h
        label = labels[idx] if idx < len(labels) else f"变体{idx}"
        if error is not None:
            label += f" err={error:.0f}"
        else:
            label += " (跳过)"
        if idx == best_idx:
            label += " ★最佳"
        label += f" [{status}]"

        text_color = (0, 255, 0) if idx == best_idx else (200, 200, 200)
        grid = _put_chinese_text(grid, label, (x0 + 5, label_y + 5), font_size=18, color=text_color)

    return grid


def _build_patch_extraction_debug(
    variant: np.ndarray,
    observed_rgb: np.ndarray,
    reference: ColorCardReference,
    patch_polygons: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Build a debug image showing the 24 patch extraction positions.
    
    Left side: the variant with numbered extraction rectangles drawn.
    Right side: a comparison panel showing observed vs reference RGB for each patch.
    """
    _require_cv2()

    h, w = variant.shape[:2]

    # --- 左图：在色卡上标注取色区域 ---
    # 放大3倍以便看清标注
    scale = 3
    left = cv2.resize(variant, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    if patch_polygons is None:
        patch_polygons = _build_uniform_patch_polygons(variant, reference)

    for patch_idx, polygon in enumerate(patch_polygons):
        scaled = np.round(polygon * scale).astype(np.int32)
        cv2.polylines(left, [scaled], True, (0, 255, 255), 2)
        center = np.mean(scaled, axis=0).astype(np.int32)
        cx = int(center[0]) - 8
        cy = int(center[1]) + 5
        cv2.putText(left, str(patch_idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(left, str(patch_idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # --- 右图：观测值 vs 参考值对比面板 ---
    swatch_size = 40
    row_height = swatch_size + 10
    panel_w = 400
    panel_h = h * scale
    right = np.full((panel_h, panel_w, 3), 50, dtype=np.uint8)

    right = _put_chinese_text(right, "序号  观测色      参考色      误差", (5, 5), font_size=16, color=(255, 255, 255))
    start_y = 30

    for i in range(min(len(observed_rgb), len(reference.patch_rgb))):
        y_pos = start_y + i * row_height
        if y_pos + swatch_size > panel_h:
            break

        obs = observed_rgb[i].astype(np.uint8)
        ref = reference.patch_rgb[i].astype(np.uint8)

        # 序号
        cv2.putText(right, f"{i:2d}", (5, y_pos + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 观测色块 (RGB→BGR)
        obs_bgr = (int(obs[2]), int(obs[1]), int(obs[0]))
        cv2.rectangle(right, (40, y_pos), (40 + swatch_size, y_pos + swatch_size), obs_bgr, -1)
        cv2.rectangle(right, (40, y_pos), (40 + swatch_size, y_pos + swatch_size), (200, 200, 200), 1)

        # 观测RGB文字
        cv2.putText(right, f"{int(obs[0]):3d},{int(obs[1]):3d},{int(obs[2]):3d}",
                    (90, y_pos + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # 参考色块 (RGB→BGR)
        ref_bgr = (int(ref[2]), int(ref[1]), int(ref[0]))
        cv2.rectangle(right, (210, y_pos), (210 + swatch_size, y_pos + swatch_size), ref_bgr, -1)
        cv2.rectangle(right, (210, y_pos), (210 + swatch_size, y_pos + swatch_size), (200, 200, 200), 1)

        # 参考RGB文字
        cv2.putText(right, f"{int(ref[0]):3d},{int(ref[1]):3d},{int(ref[2]):3d}",
                    (260, y_pos + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # 单色块误差
        err = float(np.mean((observed_rgb[i] - reference.patch_rgb[i]) ** 2))
        cv2.putText(right, f"{err:.0f}", (360, y_pos + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 255), 1)

    # 拼接左右
    if left.shape[0] != right.shape[0]:
        right = cv2.resize(right, (panel_w, left.shape[0]))

    return np.hstack([left, right])


def _build_uniform_patch_polygons(card_image: np.ndarray, reference: ColorCardReference) -> list[np.ndarray]:
    """Return the default axis-aligned patch polygons used as a fallback."""

    height, width = card_image.shape[:2]
    cell_width = width / reference.cols
    cell_height = height / reference.rows
    shrink = 0.22
    polygons: list[np.ndarray] = []

    for row_index in range(reference.rows):
        for col_index in range(reference.cols):
            x0 = float((col_index + shrink) * cell_width)
            x1 = float((col_index + 1 - shrink) * cell_width)
            y0 = float((row_index + shrink) * cell_height)
            y1 = float((row_index + 1 - shrink) * cell_height)
            polygons.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32))
    return polygons


def _component_to_patch_box(stats_row: np.ndarray, centroid_row: np.ndarray) -> _PatchBox | None:
    """Convert one connected component into a patch candidate."""

    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])
    area = float(stats_row[cv2.CC_STAT_AREA])
    if w <= 0 or h <= 0:
        return None
    return _PatchBox(
        x=x,
        y=y,
        w=w,
        h=h,
        area=area,
        center_x=float(centroid_row[0]),
        center_y=float(centroid_row[1]),
    )


def _box_iou(first: _PatchBox, second: _PatchBox) -> float:
    """Compute IoU between two patch candidate boxes."""

    x1 = max(first.x, second.x)
    y1 = max(first.y, second.y)
    x2 = min(first.x + first.w, second.x + second.w)
    y2 = min(first.y + first.h, second.y + second.h)
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = first.w * first.h + second.w * second.h - inter
    return inter / max(union, 1.0)


def _suppress_patch_overlaps(boxes: list[_PatchBox]) -> list[_PatchBox]:
    """Remove highly-overlapping patch candidates."""

    kept: list[_PatchBox] = []
    for patch in sorted(boxes, key=lambda item: item.area, reverse=True):
        if any(_box_iou(patch, existing) > 0.35 for existing in kept):
            continue
        kept.append(patch)
    return sorted(kept, key=lambda item: (item.center_y, item.center_x))


def _filter_patch_boxes(card_image: np.ndarray, stats: np.ndarray, centroids: np.ndarray) -> list[_PatchBox]:
    """Filter connected components down to likely individual color patches."""

    image_area = float(card_image.shape[0] * card_image.shape[1])
    boxes: list[_PatchBox] = []
    for index in range(1, stats.shape[0]):
        patch = _component_to_patch_box(stats[index], centroids[index])
        if patch is None:
            continue
        if patch.area < image_area * 0.008 or patch.area > image_area * 0.04:
            continue
        aspect_ratio = patch.w / max(1.0, float(patch.h))
        if not 0.55 <= aspect_ratio <= 1.65:
            continue
        if patch.w < card_image.shape[1] * 0.06 or patch.h < card_image.shape[0] * 0.08:
            continue
        if patch.w > card_image.shape[1] * 0.22 or patch.h > card_image.shape[0] * 0.28:
            continue
        fill_ratio = patch.area / max(float(patch.w * patch.h), 1.0)
        if fill_ratio < 0.72:
            continue
        boxes.append(patch)
    return _suppress_patch_overlaps(boxes)


def _cluster_axis(values: list[float], threshold: float) -> list[float]:
    """Cluster 1D coordinates into row or column groups."""

    if not values:
        return []
    sorted_values = sorted(values)
    groups: list[list[float]] = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if abs(value - float(np.mean(groups[-1]))) <= threshold:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.mean(group)) for group in groups]


def _axis_fit_cost(detected: list[float], candidate: list[float]) -> float:
    """Return the 1D fitting cost from detected positions to candidate positions."""

    return float(sum(min(abs(value - item) for item in candidate) for value in detected))


def _infer_axis_positions(centers: list[float], expected_count: int, nominal_step: float) -> list[float]:
    """Infer the full set of row or column centers from partial detections."""

    if not centers:
        return [nominal_step * (index + 0.5) for index in range(expected_count)]
    centers = sorted(float(value) for value in centers)
    if len(centers) == expected_count:
        return centers
    if len(centers) == 1:
        start = centers[0] - nominal_step * (expected_count - 1) / 2.0
        return [start + nominal_step * index for index in range(expected_count)]

    diffs = np.diff(centers)
    inferred_step = float(np.median(diffs)) if len(diffs) else nominal_step
    inferred_step = max(inferred_step, nominal_step * 0.6)
    best_positions = [centers[0] + inferred_step * index for index in range(expected_count)]
    best_cost = _axis_fit_cost(centers, best_positions)

    for anchor_index in range(expected_count):
        start = centers[0] - inferred_step * anchor_index
        candidate = [start + inferred_step * index for index in range(expected_count)]
        cost = _axis_fit_cost(centers, candidate)
        if cost < best_cost:
            best_cost = cost
            best_positions = candidate
    return best_positions


def _assign_boxes_to_grid(
    boxes: list[_PatchBox],
    row_centers: list[float],
    col_centers: list[float],
) -> dict[tuple[int, int], _PatchBox]:
    """Assign detected patch boxes to the nearest coarse row/column grid cell."""

    assignments: dict[tuple[int, int], _PatchBox] = {}
    for patch in boxes:
        row_index = int(np.argmin([abs(patch.center_y - center) for center in row_centers]))
        col_index = int(np.argmin([abs(patch.center_x - center) for center in col_centers]))
        key = (row_index, col_index)
        previous = assignments.get(key)
        if previous is None:
            assignments[key] = patch
            continue
        previous_dist = abs(previous.center_x - col_centers[col_index]) + abs(previous.center_y - row_centers[row_index])
        current_dist = abs(patch.center_x - col_centers[col_index]) + abs(patch.center_y - row_centers[row_index])
        if current_dist < previous_dist:
            assignments[key] = patch
    return assignments


def _estimate_patch_grid_model(
    assignments: dict[tuple[int, int], _PatchBox],
    nominal_w: float,
    nominal_h: float,
) -> _PatchGridModel:
    """Estimate a robust affine 4x6 patch grid from detected patch centers."""

    if len(assignments) < 3:
        return _PatchGridModel(
            origin=np.array([nominal_w / 2.0, nominal_h / 2.0], dtype=np.float32),
            col_vec=np.array([nominal_w * 1.15, 0.0], dtype=np.float32),
            row_vec=np.array([0.0, nominal_h * 1.15], dtype=np.float32),
            half_col_vec=np.array([nominal_w / 2.0, 0.0], dtype=np.float32),
            half_row_vec=np.array([0.0, nominal_h / 2.0], dtype=np.float32),
        )

    col_steps: list[np.ndarray] = []
    row_steps: list[np.ndarray] = []
    items = list(assignments.items())
    for (row_a, col_a), patch_a in items:
        center_a = np.array([patch_a.center_x, patch_a.center_y], dtype=np.float32)
        for (row_b, col_b), patch_b in items:
            if (row_a, col_a) == (row_b, col_b):
                continue
            center_b = np.array([patch_b.center_x, patch_b.center_y], dtype=np.float32)
            if row_a == row_b and col_b > col_a:
                col_steps.append((center_b - center_a) / float(col_b - col_a))
            if col_a == col_b and row_b > row_a:
                row_steps.append((center_b - center_a) / float(row_b - row_a))

    col_vec = np.median(np.asarray(col_steps, dtype=np.float32), axis=0) if col_steps else np.array([nominal_w * 1.15, 0.0], dtype=np.float32)
    row_vec = np.median(np.asarray(row_steps, dtype=np.float32), axis=0) if row_steps else np.array([0.0, nominal_h * 1.15], dtype=np.float32)

    origin_candidates: list[np.ndarray] = []
    for (row_index, col_index), patch in assignments.items():
        center = np.array([patch.center_x, patch.center_y], dtype=np.float32)
        origin_candidates.append(center - col_index * col_vec - row_index * row_vec)
    origin = np.median(np.asarray(origin_candidates, dtype=np.float32), axis=0)

    col_len = max(float(np.linalg.norm(col_vec)), 1.0)
    row_len = max(float(np.linalg.norm(row_vec)), 1.0)
    half_col_vec = col_vec * (nominal_w / col_len) * 0.5
    half_row_vec = row_vec * (nominal_h / row_len) * 0.5

    return _PatchGridModel(
        origin=origin,
        col_vec=col_vec,
        row_vec=row_vec,
        half_col_vec=half_col_vec,
        half_row_vec=half_row_vec,
    )


def _build_affine_patch_polygons(model: _PatchGridModel, reference: ColorCardReference) -> list[np.ndarray]:
    """Build 24 affine patch polygons from the estimated grid model."""

    polygons: list[np.ndarray] = []
    for row_index in range(reference.rows):
        for col_index in range(reference.cols):
            center = model.origin + col_index * model.col_vec + row_index * model.row_vec
            polygons.append(
                np.array(
                    [
                        center - model.half_col_vec - model.half_row_vec,
                        center + model.half_col_vec - model.half_row_vec,
                        center + model.half_col_vec + model.half_row_vec,
                        center - model.half_col_vec + model.half_row_vec,
                    ],
                    dtype=np.float32,
                )
            )
    return polygons


def _sample_patch_rgb_from_polygons(card_image: np.ndarray, patch_polygons: list[np.ndarray]) -> np.ndarray:
    """Sample mean RGB values from polygonal patch regions."""

    rgb_card = card_image[:, :, ::-1].astype(np.float32)
    patch_values: list[np.ndarray] = []
    height, width = card_image.shape[:2]

    for polygon in patch_polygons:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255)
        pixels = rgb_card[mask > 0]
        if pixels.size == 0:
            patch_values.append(np.zeros(3, dtype=np.float32))
            continue
        patch_values.append(pixels.reshape(-1, 3).mean(axis=0))

    return np.array(patch_values, dtype=np.float32)


def _extract_patch_rgb_adaptive(
    card_image: np.ndarray,
    reference: ColorCardReference,
) -> _AdaptivePatchExtraction | None:
    """Extract patch RGB with adaptive open-mask detection and affine grid fitting."""

    gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _threshold_value, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    component_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(opened_mask, 8)
    boxes = _filter_patch_boxes(card_image, stats, centroids)
    candidate_count = len(boxes)
    if candidate_count < 10:
        return None

    nominal_w = float(np.median([box.w for box in boxes])) if boxes else card_image.shape[1] / reference.cols
    nominal_h = float(np.median([box.h for box in boxes])) if boxes else card_image.shape[0] / reference.rows
    row_groups = _cluster_axis([box.center_y for box in boxes], threshold=max(8.0, nominal_h * 0.45))
    col_groups = _cluster_axis([box.center_x for box in boxes], threshold=max(8.0, nominal_w * 0.45))
    row_centers = _infer_axis_positions(row_groups, reference.rows, nominal_h * 1.25)
    col_centers = _infer_axis_positions(col_groups, reference.cols, nominal_w * 1.25)

    coarse_assignments = _assign_boxes_to_grid(boxes, row_centers, col_centers)
    model = _estimate_patch_grid_model(coarse_assignments, nominal_w, nominal_h)
    polygons = _build_affine_patch_polygons(model, reference)
    observed_rgb = _sample_patch_rgb_from_polygons(card_image, polygons)

    if observed_rgb.std() < 10:
        return None
    return _AdaptivePatchExtraction(
        observed_rgb=observed_rgb,
        patch_polygons=polygons,
        candidate_count=candidate_count,
        assignments=coarse_assignments,
    )


def _build_patch_target_centers(image_shape: tuple[int, int, int], reference: ColorCardReference) -> dict[tuple[int, int], np.ndarray]:
    """Build the ideal landscape patch-center grid used for second-stage rectification."""

    height, width = image_shape[:2]
    margin_x = max(16.0, width * 0.18)
    margin_y = max(20.0, height * 0.14)
    step_x = (width - 1 - 2.0 * margin_x) / max(reference.cols - 1, 1)
    step_y = (height - 1 - 2.0 * margin_y) / max(reference.rows - 1, 1)

    centers: dict[tuple[int, int], np.ndarray] = {}
    for row_index in range(reference.rows):
        for col_index in range(reference.cols):
            centers[(row_index, col_index)] = np.array(
                [margin_x + col_index * step_x, margin_y + row_index * step_y],
                dtype=np.float32,
            )
    return centers


def _rectify_variant_by_patch_grid(
    variant: np.ndarray,
    assignments: dict[tuple[int, int], _PatchBox],
    reference: ColorCardReference,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Rectify a scored variant by aligning detected patch centers to an ideal 4x6 grid."""

    if len(assignments) < 4:
        return None, None

    target_centers = _build_patch_target_centers(variant.shape, reference)
    source_points: list[np.ndarray] = []
    destination_points: list[np.ndarray] = []
    for key, patch in assignments.items():
        source_points.append(np.array([patch.center_x, patch.center_y], dtype=np.float32))
        destination_points.append(target_centers[key])

    source = np.asarray(source_points, dtype=np.float32).reshape(-1, 1, 2)
    destination = np.asarray(destination_points, dtype=np.float32).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(source, destination, cv2.RANSAC, 4.0)
    if homography is None or inlier_mask is None or int(inlier_mask.sum()) < 4:
        return None, None

    rectified = cv2.warpPerspective(variant, homography, (variant.shape[1], variant.shape[0]))
    rectification_inverse = np.linalg.inv(homography).astype(np.float32)
    return rectified, rectification_inverse


def _transform_polygons(polygons: list[np.ndarray], transform: np.ndarray) -> list[np.ndarray]:
    """Apply a perspective transform to patch polygons."""

    transformed: list[np.ndarray] = []
    for polygon in polygons:
        points = polygon.reshape(-1, 1, 2).astype(np.float32)
        transformed.append(cv2.perspectiveTransform(points, transform).reshape(-1, 2))
    return transformed


def _variant_inverse_rotation_transform(rotation: int, base_width: int, base_height: int) -> np.ndarray:
    """Return the transform from a rotated variant back to the pre-rotation image."""

    if rotation == 0:
        return np.eye(3, dtype=np.float32)
    if rotation == 1:
        return np.array([[0.0, -1.0, base_width - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    if rotation == 2:
        return np.array([[-1.0, 0.0, base_width - 1.0], [0.0, -1.0, base_height - 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, base_height - 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _variant_to_warped_transform(
    variant_index: int,
    warped_shape: tuple[int, int, int],
    reference: ColorCardReference,
) -> np.ndarray:
    """Return the transform from one canonical landscape variant back to the raw warped card."""

    canonical_width = float(reference.canonical_width)
    canonical_height = float(reference.canonical_height)
    warped_height, warped_width = warped_shape[:2]
    should_flip = variant_index >= 4
    rotation = variant_index % 4

    if rotation % 2 == 1:
        rotated_width = float(warped_height)
        rotated_height = float(warped_width)
    else:
        rotated_width = float(warped_width)
        rotated_height = float(warped_height)

    resize_inverse = np.array(
        [[rotated_width / canonical_width, 0.0, 0.0], [0.0, rotated_height / canonical_height, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rotation_inverse = _variant_inverse_rotation_transform(rotation, warped_width, warped_height)
    flip_inverse = (
        np.array([[-1.0, 0.0, warped_width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        if should_flip
        else np.eye(3, dtype=np.float32)
    )
    return (flip_inverse @ rotation_inverse @ resize_inverse).astype(np.float32)


def _build_warp_geometry(
    corners: np.ndarray,
    reference: ColorCardReference,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Return warp size, destination points, and perspective transform for the detected card."""

    detected_width = (
        float(np.linalg.norm(corners[1] - corners[0])) +
        float(np.linalg.norm(corners[2] - corners[3]))
    ) / 2.0
    detected_height = (
        float(np.linalg.norm(corners[3] - corners[0])) +
        float(np.linalg.norm(corners[2] - corners[1]))
    ) / 2.0

    if detected_width >= detected_height:
        dst_w = reference.canonical_width
        dst_h = reference.canonical_height
    else:
        dst_w = reference.canonical_height
        dst_h = reference.canonical_width

    destination = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    return dst_w, dst_h, destination, transform.astype(np.float32)


def _estimate_patch_scale_in_original_image(
    corners: np.ndarray,
    warped_shape: tuple[int, int, int],
    patch_polygons: list[np.ndarray] | None,
    variant_index: int,
    reference: ColorCardReference,
    *,
    rectification_inverse: np.ndarray | None,
) -> tuple[float | None, float | None]:
    """Project best patch polygons back to the original image and estimate patch pixel size."""

    if patch_polygons is None or variant_index < 0:
        return None, None

    variant_to_warped = _variant_to_warped_transform(variant_index, warped_shape, reference)
    if rectification_inverse is not None:
        variant_to_warped = (variant_to_warped @ rectification_inverse).astype(np.float32)

    _dst_w, _dst_h, _destination, warp_transform = _build_warp_geometry(corners, reference)
    warped_to_original = np.linalg.inv(warp_transform).astype(np.float32)
    polygons_in_warped = _transform_polygons(patch_polygons, variant_to_warped)
    polygons_in_original = _transform_polygons(polygons_in_warped, warped_to_original)

    widths: list[float] = []
    heights: list[float] = []
    for polygon in polygons_in_original:
        widths.append((float(np.linalg.norm(polygon[1] - polygon[0])) + float(np.linalg.norm(polygon[2] - polygon[3]))) / 2.0)
        heights.append((float(np.linalg.norm(polygon[3] - polygon[0])) + float(np.linalg.norm(polygon[2] - polygon[1]))) / 2.0)

    if not widths or not heights:
        return None, None
    return float(np.median(widths)), float(np.median(heights))


def _match_template_bbox(
    image: np.ndarray,
    *,
    view_name: str,
    reference: ColorCardReference,
) -> tuple[np.ndarray, float, dict[str, np.ndarray]] | None:
    """Locate the card by template matching and return the estimated outer box.
    
    Returns:
        A tuple of (corners, score, debug_images) or None if no match.
    """
    debug_images: dict[str, np.ndarray] = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if view_name == "TOP":
        roi_x0 = int(image.shape[1] * 0.55)
        roi_y0 = int(image.shape[0] * 0.30)
        roi_x1 = int(image.shape[1] * 0.90)
        roi_y1 = int(image.shape[0] * 0.95)
        roi = gray[roi_y0:roi_y1, roi_x0:roi_x1]
        template = _build_card_template(reference, mode="full")
        match = _run_template_match(roi, template, scale_range=np.linspace(1.0, 2.5, 16))
        
        # 添加调试图像
        debug_images["template_模板"] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        debug_images["template_搜索区域"] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        
        if match is None:
            return None

        score, (x, y), width, height = match
        corners = np.array(
            [
                [roi_x0 + x, roi_y0 + y],
                [roi_x0 + x + width, roi_y0 + y],
                [roi_x0 + x + width, roi_y0 + y + height],
                [roi_x0 + x, roi_y0 + y + height],
            ],
            dtype=np.float32,
        )
        return corners, score, debug_images

    # FRONT views: 扩大搜索区域
    roi_x0 = 0
    roi_y0 = 0
    roi_x1 = int(image.shape[1] * 0.45)
    roi_y1 = int(image.shape[0] * 0.55)  # 从40%扩大到55%
    roi = gray[roi_y0:roi_y1, roi_x0:roi_x1]
    anchor_template = _build_card_template(reference, mode="front_anchor")
    match = _run_template_match(roi, anchor_template, scale_range=np.linspace(1.0, 2.5, 16))
    
    # 添加调试图像
    debug_images["template_锚点模板"] = cv2.cvtColor(anchor_template, cv2.COLOR_GRAY2BGR)
    debug_images["template_搜索区域"] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    
    if match is None:
        return None

    score, (x, y), width, height = match
    full_height = int(round(height * (reference.canonical_height / anchor_template.shape[0]) * 1.03))
    full_height = min(full_height, roi.shape[0] - y)
    corners = np.array(
        [
            [roi_x0 + x, roi_y0 + y],
            [roi_x0 + x + width, roi_y0 + y],
            [roi_x0 + x + width, roi_y0 + y + full_height],
            [roi_x0 + x, roi_y0 + y + full_height],
        ],
        dtype=np.float32,
    )
    return corners, score, debug_images


def _run_template_match(
    roi_gray: np.ndarray,
    template_gray: np.ndarray,
    *,
    scale_range: np.ndarray,
) -> tuple[float, tuple[int, int], int, int] | None:
    """Run multi-scale template matching and return the best match."""

    best_match: tuple[float, tuple[int, int], int, int] | None = None

    for scale in scale_range:
        width = max(20, int(round(template_gray.shape[1] * float(scale))))
        height = max(20, int(round(template_gray.shape[0] * float(scale))))
        if width >= roi_gray.shape[1] or height >= roi_gray.shape[0]:
            continue

        resized_template = cv2.resize(template_gray, (width, height))
        result = cv2.matchTemplate(roi_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv2.minMaxLoc(result)

        if best_match is None or max_value > best_match[0]:
            best_match = (float(max_value), (int(max_location[0]), int(max_location[1])), width, height)

    return best_match


def _build_card_template(reference: ColorCardReference, *, mode: str) -> np.ndarray:
    """Build a grayscale card template for full-card or anchor matching."""

    card = np.full((reference.canonical_height, reference.canonical_width, 3), 55, dtype=np.uint8)
    margin_x = 22
    margin_y = 28
    cell_width = (reference.canonical_width - 2 * margin_x) // reference.cols
    cell_height = (reference.canonical_height - 2 * margin_y) // reference.rows

    patch_index = 0
    for row_index in range(reference.rows):
        for col_index in range(reference.cols):
            x0 = margin_x + col_index * cell_width + 6
            x1 = margin_x + (col_index + 1) * cell_width - 6
            y0 = margin_y + row_index * cell_height + 6
            y1 = margin_y + (row_index + 1) * cell_height - 6
            card[y0:y1, x0:x1] = reference.patch_rgb[patch_index][::-1].astype(np.uint8)
            patch_index += 1

    if mode == "full":
        return cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    if mode == "front_anchor":
        return cv2.cvtColor(card[:140], cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported template mode: {mode}")


def _build_detection_mask(image: np.ndarray, search_region: tuple[int, int, int, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build a global mask used to locate the color card.
    
    Returns:
        A tuple of (full_mask, debug_images) where debug_images contains intermediate steps.
    """
    _require_cv2()
    x0, y0, width, height = search_region
    roi = image[y0 : y0 + height, x0 : x0 + width]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 改进的掩膜检测策略：
    # 1. 检测高饱和度彩色区域（色卡的彩色色块）
    color_mask = cv2.inRange(hsv, np.array([0, 50, 50], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    
    # 2. 检测低饱和度但高亮度的区域（白色/浅灰色块）
    bright_mask = cv2.inRange(hsv, np.array([0, 0, 180], dtype=np.uint8), np.array([180, 50, 255], dtype=np.uint8))
    
    # 3. 检测低饱和度低亮度区域（黑色/深灰色块）
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 20], dtype=np.uint8), np.array([180, 50, 100], dtype=np.uint8))
    
    # 合并所有掩膜
    combined = cv2.bitwise_or(color_mask, bright_mask)
    combined = cv2.bitwise_or(combined, dark_mask)

    # 形态学处理
    kernel_size = max(5, int(round(min(width, height) * 0.02)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_make_odd(kernel_size), _make_odd(kernel_size)))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_mask[y0 : y0 + height, x0 : x0 + width] = cleaned

    # 构建调试图像
    debug_images = {
        "roi_原始区域": roi.copy(),
        "hsv_H通道": hsv[:, :, 0],
        "hsv_S通道": hsv[:, :, 1],
        "hsv_V通道": hsv[:, :, 2],
        "color_mask_彩色掩码": color_mask,
        "bright_mask_亮色掩码": bright_mask,
        "dark_mask_暗色掩码": dark_mask,
        "combined_合并掩码": combined,
        "cleaned_形态学处理后": cleaned,
    }
    
    return full_mask, debug_images


def _default_search_region(image: np.ndarray, *, view_name: str) -> tuple[int, int, int, int]:
    """Return the heuristic search region for the given view."""

    height, width = image.shape[:2]

    if view_name == "TOP":
        x0 = int(width * 0.55)
        y0 = int(height * 0.35)
        x1 = int(width * 0.98)
        y1 = int(height * 0.95)
    elif view_name in {"FRONT-1", "FRONT-2"}:
        # 扩大FRONT视图的搜索区域，特别是向下扩展
        x0 = int(width * 0.00)
        y0 = int(height * 0.00)
        x1 = int(width * 0.45)
        y1 = int(height * 0.65)  # 从52%扩大到65%
    else:
        x0, y0, x1, y1 = 0, 0, width, height

    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _extract_patch_rgb(card_image: np.ndarray, reference: ColorCardReference) -> np.ndarray:
    """Extract the mean RGB value of each patch from the warped card image."""
    return _sample_patch_rgb_from_polygons(card_image, _build_uniform_patch_polygons(card_image, reference))


def _fit_color_matrix(observed_patch_rgb: np.ndarray, reference_patch_rgb: np.ndarray) -> np.ndarray:
    """Fit an affine color mapping from observed RGB to reference RGB."""

    source = np.hstack([observed_patch_rgb, np.ones((observed_patch_rgb.shape[0], 1), dtype=np.float32)])
    matrix, _, _, _ = np.linalg.lstsq(source, reference_patch_rgb, rcond=None)
    return matrix.astype(np.float32)


def _apply_color_matrix_to_rgb(rgb_values: np.ndarray, correction_matrix: np.ndarray) -> np.ndarray:
    """Apply an affine color matrix to RGB rows."""

    source = np.hstack([rgb_values.astype(np.float32), np.ones((rgb_values.shape[0], 1), dtype=np.float32)])
    corrected = source @ correction_matrix
    return np.clip(corrected, 0, 255)


def _warp_card(image: np.ndarray, corners: np.ndarray, reference: ColorCardReference) -> np.ndarray:
    """Warp the detected card into a canonical rectangle, preserving aspect ratio direction.
    
    If the detected shape is portrait (height > width), warp to portrait canonical size.
    If landscape (width >= height), warp to landscape canonical size.
    This prevents distortion from forcing a portrait card into a landscape frame.
    """

    _require_cv2()
    dst_w, dst_h, _destination, transform = _build_warp_geometry(corners, reference)
    return cv2.warpPerspective(image, transform, (dst_w, dst_h))


def _generate_card_variants(card_image: np.ndarray, reference: ColorCardReference) -> list[np.ndarray]:
    """Generate plausible card orientations for patch matching.
    
    Each variant is resized to the canonical landscape size (360×240) for
    uniform patch extraction. Variants whose rotated shape is portrait
    are resized to landscape, ensuring the aspect ratio matches when
    the grid reading order is correct.
    """

    _require_cv2()
    canonical_landscape = (reference.canonical_width, reference.canonical_height)  # (360, 240)
    variants: list[np.ndarray] = []
    for should_flip in (False, True):
        base = cv2.flip(card_image, 1) if should_flip else card_image
        for rotation in range(4):
            rotated = np.rot90(base, rotation).copy()
            # 统一 resize 到标准横向尺寸，供 _extract_patch_rgb 按 4行6列提取
            resized = cv2.resize(rotated, canonical_landscape)
            variants.append(resized)
    return variants


def _order_points(points: np.ndarray) -> np.ndarray:
    """Order points as top-left, top-right, bottom-right, bottom-left."""

    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sums = points.sum(axis=1)
    point_diffs = np.diff(points, axis=1)

    ordered[0] = points[np.argmin(point_sums)]
    ordered[2] = points[np.argmax(point_sums)]
    ordered[1] = points[np.argmin(point_diffs)]
    ordered[3] = points[np.argmax(point_diffs)]
    return ordered


def _card_dimensions_in_pixels(corners: np.ndarray) -> tuple[float, float]:
    """Measure the detected card size in pixels."""

    top_width = float(np.linalg.norm(corners[1] - corners[0]))
    bottom_width = float(np.linalg.norm(corners[2] - corners[3]))
    left_height = float(np.linalg.norm(corners[3] - corners[0]))
    right_height = float(np.linalg.norm(corners[2] - corners[1]))
    return (top_width + bottom_width) / 2.0, (left_height + right_height) / 2.0


def _draw_search_region_overlay(image: np.ndarray, search_region: tuple[int, int, int, int]) -> np.ndarray:
    """Draw the search region on top of the input image."""

    _require_cv2()
    overlay = image.copy()
    x0, y0, width, height = search_region
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 255, 255), 4)
    return overlay


def _draw_card_overlay(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw the detected card quadrilateral on top of the image."""

    _require_cv2()
    overlay = image.copy()
    cv2.polylines(overlay, [corners.astype(np.int32)], True, (0, 255, 0), 4)
    return overlay


def _make_odd(value: int) -> int:
    """Return the next odd integer greater than or equal to value."""

    return value if value % 2 == 1 else value + 1


def _validate_color_image(image: np.ndarray) -> None:
    """Validate that the input is a non-empty 3-channel BGR image."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a 3-channel BGR image")


def _require_cv2() -> None:
    """Ensure OpenCV is available for calibration operations."""

    if cv2 is None:
        raise ModuleNotFoundError("No module named 'cv2'")
