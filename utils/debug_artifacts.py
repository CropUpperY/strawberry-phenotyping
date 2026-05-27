"""Helpers for exporting intermediate debug images during analysis."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


# ==============================================================================
# 调试目录结构定义（中文命名，按处理流程排序）
# ==============================================================================

class DebugCategory(NamedTuple):
    """调试输出分类定义"""
    order: int       # 流程顺序号
    folder_name: str # 中文文件夹名称
    key: str         # 内部标识键


# 定义所有调试输出分类，按处理流程排序
DEBUG_CATEGORIES = {
    "color_calibration": DebugCategory(1, "01_色卡检测与校正", "color_calibration"),
    "top_segmentation": DebugCategory(2, "02_TOP俯视图分割", "top_segmentation"),
    "front_segmentation": DebugCategory(3, "03_FRONT正视图分割", "front_segmentation"),
    "leaf_area": DebugCategory(4, "04_叶面积计算", "leaf_area"),
    "convex_hull_area": DebugCategory(5, "05_凸包面积计算", "convex_hull_area"),
    "greenness": DebugCategory(6, "06_绿度计算", "greenness"),
    "canopy_height": DebugCategory(7, "07_冠幅高度计算", "canopy_height"),
    "canopy_width": DebugCategory(8, "08_植株冠径计算", "canopy_width"),
    "side_projection_area": DebugCategory(9, "09_侧面投影面积", "side_projection_area"),
}

# 调试步骤的中文名称映射
STEP_NAME_TRANSLATIONS = {
    # 色卡检测相关（自动检测）
    "search_region_overlay": "搜索区域",
    "detection_mask": "检测掩码",
    "card_overlay": "色卡边界",
    "warped_card": "透视校正后色卡",
    "corrected_card": "色彩校正后色卡",
    "before_after": "校正前后对比",
    # 色卡检测相关（手动选择）
    "manual_region_overlay": "手动选择区域",
    "manual_region_roi": "色卡区域裁剪",
    "manual_warped_card": "透视校正后色卡",
    "manual_corrected_card": "色彩校正后色卡",
    "manual_before_after": "校正前后对比",
    # TOP分割相关
    "top_original": "原始图像",
    "normalized": "归一化处理",
    "denoised": "去噪处理",
    "hsv_green_mask": "HSV绿色掩码",
    "green_dominance_mask": "绿色优势掩码",
    "combined_mask": "合并掩码",
    "cleaned_mask": "形态学清理",
    "filtered_mask": "最终分割掩码",
    # FRONT分割相关
    "front_1_original": "FRONT-1原始图像",
    "front_1_normalized": "FRONT-1归一化",
    "front_1_denoised": "FRONT-1去噪",
    "front_1_hsv_green_mask": "FRONT-1绿色掩码",
    "front_1_filtered_mask": "FRONT-1分割掩码",
    "front_2_original": "FRONT-2原始图像",
    "front_2_normalized": "FRONT-2归一化",
    "front_2_denoised": "FRONT-2去噪",
    "front_2_hsv_green_mask": "FRONT-2绿色掩码",
    "front_2_filtered_mask": "FRONT-2分割掩码",
    # 叶面积/凸包相关
    "leaf_area_overlay": "叶面积覆盖图",
    "contour_overlay": "轮廓提取",
    "convex_hull_overlay": "凸包覆盖图",
    # 绿度相关
    "masked_plant_region": "植株区域提取",
    "green_channel": "绿色通道",
    "excess_green_map": "ExG指数图",
    "excess_green_heatmap": "ExG热力图",
    # FRONT测量相关
    "front_1_mask": "FRONT-1掩码",
    "front_1_masked_region": "FRONT-1植株区域",
    "front_1_projection_overlay": "FRONT-1投影覆盖",
    "front_1_bounding_box": "FRONT-1边界框",
    "front_1_contour_overlay": "FRONT-1轮廓",
    "front_2_mask": "FRONT-2掩码",
    "front_2_masked_region": "FRONT-2植株区域",
    "front_2_projection_overlay": "FRONT-2投影覆盖",
    "front_2_bounding_box": "FRONT-2边界框",
    "front_2_contour_overlay": "FRONT-2轮廓",
}


def get_debug_folder_name(category_key: str) -> str:
    """获取调试分类的中文文件夹名称"""
    category = DEBUG_CATEGORIES.get(category_key)
    if category is None:
        return category_key
    return category.folder_name


def translate_step_name(step_name: str) -> str:
    """将步骤名称翻译为中文"""
    return STEP_NAME_TRANSLATIONS.get(step_name, step_name)


def save_debug_steps(
    sample_id: str,
    category_key: str,
    steps: list[tuple[str, np.ndarray]],
    *,
    output_root: str | Path,
) -> list[Path]:
    """Save ordered debug images for one category under the sample debug directory.
    
    Args:
        sample_id: 样本编号
        category_key: 调试分类键（如 'color_calibration', 'leaf_area' 等）
        steps: 步骤列表，每个元素为 (步骤名, 图像数组)
        output_root: 调试输出根目录
    
    Returns:
        保存的文件路径列表
    """
    folder_name = get_debug_folder_name(category_key)
    target_dir = Path(output_root) / sample_id / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for index, (step_name, image) in enumerate(steps, start=1):
        # 翻译步骤名称为中文
        translated_name = translate_step_name(step_name)
        # 清理文件名中的特殊字符
        safe_name = translated_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        output_path = target_dir / f"{index:02d}_{safe_name}.png"
        _write_image(output_path, image)
        saved_paths.append(output_path)

    return saved_paths


# 兼容旧API的别名（保持向后兼容）
def save_debug_steps_legacy(
    sample_id: str,
    trait_key: str,
    steps: list[tuple[str, np.ndarray]],
    *,
    output_root: str | Path,
) -> list[Path]:
    """Legacy API for backward compatibility."""
    return save_debug_steps(sample_id, trait_key, steps, output_root=output_root)


def create_mask_overlay(image: np.ndarray, mask: np.ndarray, *, alpha: float = 0.35) -> np.ndarray:
    """Create a translucent overlay that highlights the foreground mask."""
    import cv2

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a 3-channel BGR image")
    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel image")

    mask = _align_mask_to_image(mask, image)

    overlay = image.copy()
    highlighted = image.copy()
    highlighted[mask > 0] = (0, 200, 0)
    return cv2.addWeighted(highlighted, alpha, overlay, 1 - alpha, 0)


def create_masked_color_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep only the masked foreground region in a color image."""

    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel image")

    mask = _align_mask_to_image(mask, image)

    output = np.zeros_like(image)
    output[mask > 0] = image[mask > 0]
    return output


def create_gray_background_focus_image(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    gray_value: int = 96,
) -> np.ndarray:
    """Highlight plant foreground on a neutral gray background."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a 3-channel BGR image")
    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel image")

    mask = _align_mask_to_image(mask, image)

    gray_value = int(max(0, min(255, gray_value)))
    background = np.full_like(image, gray_value)
    background[mask > 0] = image[mask > 0]
    return background


def _align_mask_to_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Resize mask to image height/width when dimensions do not match."""
    import cv2

    target_h, target_w = image.shape[:2]
    if mask.shape == (target_h, target_w):
        return mask
    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def create_heatmap(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale image into a color heatmap."""
    import cv2

    if image.ndim != 2:
        raise ValueError("image must be single-channel for heatmap conversion")
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def create_debug_montage(
    images: list[np.ndarray],
    *,
    columns: int = 3,
    tile_size: tuple[int, int] = (320, 240),
    background_color: tuple[int, int, int] = (24, 30, 26),
) -> np.ndarray:
    """Compose a simple BGR montage for debug review without matplotlib."""
    import cv2

    if not images:
        raise ValueError("images must contain at least one image")
    if columns <= 0:
        raise ValueError("columns must be greater than 0")

    tile_width, tile_height = tile_size
    rows = (len(images) + columns - 1) // columns
    canvas = np.full((rows * tile_height, columns * tile_width, 3), background_color, dtype=np.uint8)

    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        prepared = _ensure_color_image(image)
        resized = cv2.resize(prepared, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        y0 = row * tile_height
        x0 = column * tile_width
        canvas[y0:y0 + tile_height, x0:x0 + tile_width] = resized

    return canvas


def _ensure_color_image(image: np.ndarray) -> np.ndarray:
    """Normalize debug images to 3-channel BGR for montage composition."""

    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError("debug montage images must be grayscale or 3-channel BGR")


def _write_image(path: Path, image: np.ndarray) -> None:
    """Write an ndarray to disk as a PNG file."""
    import cv2

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"Failed to encode image for debug output: {path}")
    path.write_bytes(encoded.tobytes())
