from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.preprocessing import denoise_image, normalize_image


DEFAULT_INPUT = Path(
    "d:/code/pycharm/strawberry/output/调试输出/1AB/02_TOP俯视图分割/01_原始图像.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment: recover shadowed strawberry leaves in TOP segmentation.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="Input corrected TOP image path")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to save outputs. Defaults to "
            "experiments/top_segmentation/output/recover_shadow_leaf_mask/<image-stem>"
        ),
    )
    return parser.parse_args()


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def build_hsv_green_mask(image: np.ndarray) -> np.ndarray:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 25, 20], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv_image, lower, upper)


def build_green_dominance_mask(image: np.ndarray) -> np.ndarray:
    b_channel = image[:, :, 0].astype(np.float32)
    g_channel = image[:, :, 1].astype(np.float32)
    r_channel = image[:, :, 2].astype(np.float32)
    dominance = (g_channel > 30) & (g_channel > r_channel * 1.03) & (g_channel > b_channel * 1.05)
    return dominance.astype(np.uint8) * 255


def build_relaxed_green_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    relaxed_hsv_mask = cv2.inRange(hsv_image, np.array([20, 12, 15], dtype=np.uint8), np.array([110, 255, 190], dtype=np.uint8))

    b_channel = image[:, :, 0].astype(np.float32)
    g_channel = image[:, :, 1].astype(np.float32)
    r_channel = image[:, :, 2].astype(np.float32)
    relaxed_dominance_mask = ((g_channel > 18) & (g_channel >= r_channel * 0.95) & (g_channel >= b_channel * 0.98)).astype(np.uint8) * 255

    relaxed_green_mask = cv2.bitwise_and(relaxed_hsv_mask, relaxed_dominance_mask)
    return relaxed_hsv_mask, relaxed_dominance_mask, relaxed_green_mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    min_dim = min(mask.shape[:2])
    open_size = make_odd(max(3, int(round(min_dim * 0.004))))
    close_size = make_odd(max(open_size + 2, int(round(min_dim * 0.012))))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)


def filter_small_components(mask: np.ndarray, *, min_component_area_ratio: float) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return mask
    min_area_pixels = max(32, int(mask.shape[0] * mask.shape[1] * min_component_area_ratio))
    filtered = np.zeros_like(mask)
    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area >= min_area_pixels:
            filtered[labels == component_index] = 255
    return filtered


def find_external_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def apply_clahe_to_l_channel(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_l, enhanced_bgr


def build_shadow_candidate_mask(image: np.ndarray, green_seed_mask: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    enhanced_l, enhanced_bgr = apply_clahe_to_l_channel(image)
    relaxed_hsv_mask, relaxed_dominance_mask, relaxed_green_mask = build_relaxed_green_mask(enhanced_bgr)
    enhanced_lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
    l_channel = enhanced_lab[:, :, 0]
    a_channel = enhanced_lab[:, :, 1]
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

    local_std = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (0, 0), 5) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 5) ** 2
    local_std = np.sqrt(np.clip(local_std, 0.0, None))
    local_std_u8 = np.clip(local_std * 3.0, 0, 255).astype(np.uint8)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    gradient_u8 = np.clip(gradient * 1.5, 0, 255).astype(np.uint8)

    dark_mask = cv2.inRange(l_channel, 18, 120)
    weak_green_lab_mask = cv2.inRange(a_channel, 80, 132)
    texture_mask = cv2.inRange(local_std_u8, 10, 255)
    gradient_mask = cv2.inRange(gradient_u8, 12, 255)

    shadow_raw = cv2.bitwise_and(dark_mask, weak_green_lab_mask)
    shadow_raw = cv2.bitwise_and(shadow_raw, texture_mask)
    shadow_raw = cv2.bitwise_and(shadow_raw, gradient_mask)
    shadow_raw = cv2.bitwise_and(shadow_raw, relaxed_green_mask)

    dilate_size = make_odd(max(31, int(round(min(image.shape[:2]) * 0.08))))
    seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    seed_neighborhood = cv2.dilate(green_seed_mask, seed_kernel, iterations=1)
    constrained_shadow = cv2.bitwise_and(shadow_raw, seed_neighborhood)

    hull_mask = np.zeros_like(green_seed_mask)
    seed_contours = find_external_contours(green_seed_mask)
    if seed_contours:
        hull = cv2.convexHull(np.vstack(seed_contours))
        cv2.fillConvexPoly(hull_mask, hull, 255)
    hull_expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (make_odd(max(41, int(round(min(image.shape[:2]) * 0.1)))),) * 2)
    expanded_hull = cv2.dilate(hull_mask, hull_expand_kernel, iterations=1)
    constrained_shadow = cv2.bitwise_and(constrained_shadow, expanded_hull)

    return constrained_shadow, {
        "illumination_enhanced_l": enhanced_l,
        "illumination_enhanced_bgr": enhanced_bgr,
        "relaxed_hsv_mask": relaxed_hsv_mask,
        "relaxed_dominance_mask": relaxed_dominance_mask,
        "relaxed_green_mask": relaxed_green_mask,
        "shadow_dark_mask": dark_mask,
        "shadow_weak_green_lab_mask": weak_green_lab_mask,
        "shadow_texture_mask": texture_mask,
        "shadow_gradient_mask": gradient_mask,
        "shadow_raw": shadow_raw,
        "seed_neighborhood": seed_neighborhood,
        "seed_hull_mask": hull_mask,
        "seed_hull_expanded": expanded_hull,
        "shadow_constrained": constrained_shadow,
    }


def filter_added_components(candidate_mask: np.ndarray, baseline_mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    baseline_touch_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    baseline_touch_zone = cv2.dilate(baseline_mask, baseline_touch_kernel, iterations=1)

    filtered = np.zeros_like(candidate_mask)
    component_info: list[dict[str, float | int | bool]] = []
    image_area = candidate_mask.shape[0] * candidate_mask.shape[1]
    max_area = max(256, int(image_area * 0.015))

    for component_index in range(1, component_count):
        component_mask = np.zeros_like(candidate_mask)
        component_mask[labels == component_index] = 255
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        touches_baseline = bool(np.any((component_mask > 0) & (baseline_touch_zone > 0)))
        contours = find_external_contours(component_mask)
        solidity = 0.0
        if contours:
            hull = cv2.convexHull(np.vstack(contours))
            hull_area = float(cv2.contourArea(hull))
            if hull_area > 0:
                solidity = float(area) / hull_area
        keep = bool(touches_baseline and area <= max_area and solidity >= 0.18)
        component_info.append(
            {
                "component_index": component_index,
                "area": area,
                "touches_baseline": touches_baseline,
                "solidity": round(solidity, 3),
                "kept": keep,
            }
        )
        if keep:
            filtered[labels == component_index] = 255

    return filtered, component_info


def clear_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_file() and (child.suffix.lower() == ".png" or child.name == "summary.json"):
            child.unlink()


def component_metrics(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | bool]]]:
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_height, image_width = mask.shape
    metrics: list[dict[str, float | int | bool]] = []
    for component_index in range(1, component_count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        center_x = float(centroids[component_index, 0])
        center_y = float(centroids[component_index, 1])
        bbox_area = max(width * height, 1)
        fill_ratio = float(area) / float(bbox_area)
        contours = find_external_contours((labels == component_index).astype(np.uint8) * 255)
        hull_area = 0.0
        solidity = 0.0
        if contours:
            hull = cv2.convexHull(np.vstack(contours))
            hull_area = float(cv2.contourArea(hull))
            if hull_area > 0:
                solidity = float(area) / hull_area
        is_right_side = center_x > image_width * 0.70
        is_compact = width < image_width * 0.22 and height < image_height * 0.30
        is_card_like = bool(is_right_side and is_compact and fill_ratio > 0.12 and solidity > 0.35)
        metrics.append(
            {
                "component_index": component_index,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
                "fill_ratio": round(fill_ratio, 3),
                "solidity": round(solidity, 3),
                "is_card_like": is_card_like,
            }
        )
    return labels, stats, metrics


def remove_card_like_components(mask: np.ndarray, labels: np.ndarray, metrics: list[dict[str, float | int | bool]]) -> tuple[np.ndarray, np.ndarray]:
    filtered = mask.copy()
    removed = np.zeros_like(mask)
    for item in metrics:
        if bool(item["is_card_like"]):
            index = int(item["component_index"])
            filtered[labels == index] = 0
            removed[labels == index] = 255
    return filtered, removed


def draw_mask_overlay(image: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int], title: str) -> np.ndarray:
    overlay = image.copy()
    tint = np.zeros_like(image)
    tint[:, :] = color
    overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(overlay, 0.45, tint, 0.55, 0), overlay)
    contours = find_external_contours(mask)
    if contours:
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    cv2.putText(overlay, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return overlay


def draw_component_boxes(image: np.ndarray, stats: np.ndarray, metrics: list[dict[str, float | int | bool]]) -> np.ndarray:
    overlay = image.copy()
    metric_map = {int(item["component_index"]): item for item in metrics}
    for component_index in range(1, stats.shape[0]):
        item = metric_map.get(component_index)
        if item is None:
            continue
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        color = (0, 0, 255) if bool(item["is_card_like"]) else (0, 255, 255)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
        cv2.putText(overlay, f"c{component_index}", (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return overlay


def draw_hull_overlay(image: np.ndarray, mask: np.ndarray, *, title: str) -> np.ndarray:
    overlay = image.copy()
    contours = find_external_contours(mask)
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        hull = cv2.convexHull(np.vstack(contours))
        cv2.polylines(overlay, [hull], True, (255, 255, 0), 3)
        hull_area = float(cv2.contourArea(hull))
        cv2.putText(overlay, f"hull_area={hull_area:.1f}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(overlay, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return overlay


def build_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    separator = np.full((left.shape[0], 12, 3), 32, dtype=np.uint8)
    return np.hstack([left, separator, right])


def build_result_delta_overlay(image: np.ndarray, baseline_mask: np.ndarray, final_mask: np.ndarray) -> np.ndarray:
    """Overlay baseline/final mask differences on the original image.

    Colors:
    - Yellow: overlap between baseline and final
    - Green: newly added area in final
    - Red: area removed from baseline
    """

    overlay = image.copy()
    baseline_only = (baseline_mask > 0) & (final_mask == 0)
    final_only = (final_mask > 0) & (baseline_mask == 0)
    overlap = (baseline_mask > 0) & (final_mask > 0)

    tinted = overlay.copy()
    tinted[overlap] = cv2.addWeighted(overlay[overlap], 0.35, np.full_like(overlay[overlap], (0, 255, 255)), 0.65, 0)
    tinted[final_only] = cv2.addWeighted(overlay[final_only], 0.25, np.full_like(overlay[final_only], (0, 255, 0)), 0.75, 0)
    tinted[baseline_only] = cv2.addWeighted(overlay[baseline_only], 0.25, np.full_like(overlay[baseline_only], (0, 0, 255)), 0.75, 0)

    baseline_contours = find_external_contours(baseline_mask)
    final_contours = find_external_contours(final_mask)
    if baseline_contours:
        cv2.drawContours(tinted, baseline_contours, -1, (0, 0, 255), 2)
    if final_contours:
        cv2.drawContours(tinted, final_contours, -1, (0, 255, 0), 2)

    legend_y = 28
    cv2.putText(tinted, "Yellow=overlap  Green=added  Red=removed", (16, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(tinted, "Baseline contour=red  Final contour=green", (16, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return tinted


def write_summary(
    output_dir: Path,
    input_path: Path,
    component_info: list[dict[str, float | int | bool]],
    green_seed_mask: np.ndarray,
    baseline_mask: np.ndarray,
    recovered_shadow_mask: np.ndarray,
    merged_mask_before_card_removal: np.ndarray,
    final_mask: np.ndarray,
) -> None:
    def hull_area(mask: np.ndarray) -> float:
        contours = find_external_contours(mask)
        if not contours:
            return 0.0
        return float(cv2.contourArea(cv2.convexHull(np.vstack(contours))))

    payload = {
        "input": str(input_path),
        "green_seed_area": int(cv2.countNonZero(green_seed_mask)),
        "baseline_area_after_card_removal": int(cv2.countNonZero(baseline_mask)),
        "recovered_shadow_area": int(cv2.countNonZero(recovered_shadow_mask)),
        "merged_area_before_card_removal": int(cv2.countNonZero(merged_mask_before_card_removal)),
        "final_area": int(cv2.countNonZero(final_mask)),
        "baseline_hull_area_after_card_removal": round(hull_area(baseline_mask), 2),
        "hull_area_before_card_removal": round(hull_area(merged_mask_before_card_removal), 2),
        "final_hull_area_after_card_removal": round(hull_area(final_mask), 2),
        "area_gain_vs_baseline": int(cv2.countNonZero(final_mask)) - int(cv2.countNonZero(baseline_mask)),
        "removed_components": [int(item["component_index"]) for item in component_info if bool(item["is_card_like"])],
        "component_metrics": component_info,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("d:/code/pycharm/strawberry/experiments/top_segmentation/output/recover_shadow_leaf_mask") / input_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)

    image = read_image(input_path)
    normalized = normalize_image(image)
    denoised = denoise_image(normalized, method="gaussian", kernel_size=5)

    hsv_green_mask = build_hsv_green_mask(denoised)
    green_dominance_mask = build_green_dominance_mask(denoised)
    green_seed_raw = cv2.bitwise_and(hsv_green_mask, green_dominance_mask)
    green_seed_cleaned = clean_mask(green_seed_raw)
    green_seed_filtered = filter_small_components(green_seed_cleaned, min_component_area_ratio=0.0005)

    baseline_labels, _baseline_stats, baseline_component_info = component_metrics(green_seed_filtered)
    baseline_mask_after_card_removal, baseline_removed_card_mask = remove_card_like_components(
        green_seed_filtered,
        baseline_labels,
        baseline_component_info,
    )

    recovered_shadow_mask, shadow_debug = build_shadow_candidate_mask(denoised, green_seed_filtered)
    recovered_shadow_mask = clean_mask(recovered_shadow_mask)
    recovered_shadow_mask = filter_small_components(recovered_shadow_mask, min_component_area_ratio=0.00015)
    recovered_shadow_mask = cv2.bitwise_and(recovered_shadow_mask, cv2.bitwise_not(baseline_mask_after_card_removal))
    recovered_shadow_mask, added_component_info = filter_added_components(recovered_shadow_mask, baseline_mask_after_card_removal)
    recovered_shadow_mask = clean_mask(recovered_shadow_mask)

    merged_mask_before_card_removal = cv2.bitwise_or(green_seed_filtered, recovered_shadow_mask)
    merged_mask_before_card_removal = clean_mask(merged_mask_before_card_removal)

    labels, stats, component_info = component_metrics(merged_mask_before_card_removal)
    final_mask, removed_card_mask = remove_card_like_components(merged_mask_before_card_removal, labels, component_info)

    save_image(output_dir / "01_original.png", image)
    save_image(output_dir / "02_normalized.png", normalized)
    save_image(output_dir / "03_denoised.png", denoised)
    save_image(output_dir / "04_hsv_green_mask.png", hsv_green_mask)
    save_image(output_dir / "05_green_dominance_mask.png", green_dominance_mask)
    save_image(output_dir / "06_green_seed_raw.png", green_seed_raw)
    save_image(output_dir / "07_green_seed_cleaned.png", green_seed_cleaned)
    save_image(output_dir / "08_green_seed_filtered.png", green_seed_filtered)
    save_image(output_dir / "09_illumination_enhanced_l.png", shadow_debug["illumination_enhanced_l"])
    save_image(output_dir / "10_illumination_enhanced_bgr.png", shadow_debug["illumination_enhanced_bgr"])
    save_image(output_dir / "11_relaxed_hsv_mask.png", shadow_debug["relaxed_hsv_mask"])
    save_image(output_dir / "12_relaxed_dominance_mask.png", shadow_debug["relaxed_dominance_mask"])
    save_image(output_dir / "13_relaxed_green_mask.png", shadow_debug["relaxed_green_mask"])
    save_image(output_dir / "14_shadow_dark_mask.png", shadow_debug["shadow_dark_mask"])
    save_image(output_dir / "15_shadow_weak_green_lab_mask.png", shadow_debug["shadow_weak_green_lab_mask"])
    save_image(output_dir / "16_shadow_texture_mask.png", shadow_debug["shadow_texture_mask"])
    save_image(output_dir / "17_shadow_gradient_mask.png", shadow_debug["shadow_gradient_mask"])
    save_image(output_dir / "18_shadow_raw.png", shadow_debug["shadow_raw"])
    save_image(output_dir / "19_seed_neighborhood.png", shadow_debug["seed_neighborhood"])
    save_image(output_dir / "20_seed_hull_mask.png", shadow_debug["seed_hull_mask"])
    save_image(output_dir / "21_seed_hull_expanded.png", shadow_debug["seed_hull_expanded"])
    save_image(output_dir / "22_shadow_constrained.png", shadow_debug["shadow_constrained"])
    save_image(output_dir / "23_recovered_shadow_mask.png", recovered_shadow_mask)
    save_image(output_dir / "24_green_seed_overlay.png", draw_mask_overlay(image, green_seed_filtered, color=(0, 255, 0), title="green seed mask"))
    save_image(output_dir / "25_baseline_removed_card_mask.png", baseline_removed_card_mask)
    save_image(output_dir / "26_baseline_mask_after_card_removal.png", baseline_mask_after_card_removal)
    save_image(output_dir / "27_baseline_overlay_after_card_removal.png", draw_mask_overlay(image, baseline_mask_after_card_removal, color=(0, 255, 255), title="baseline after card removal"))
    save_image(output_dir / "28_recovered_shadow_overlay.png", draw_mask_overlay(image, recovered_shadow_mask, color=(0, 128, 255), title="recovered shadow mask"))
    save_image(output_dir / "29_merged_mask_before_card_removal.png", merged_mask_before_card_removal)
    save_image(output_dir / "30_merged_overlay_before_card_removal.png", draw_mask_overlay(image, merged_mask_before_card_removal, color=(255, 0, 255), title="merged mask before card removal"))
    save_image(output_dir / "31_component_boxes_before_card_removal.png", draw_component_boxes(image, stats, component_info))
    save_image(output_dir / "32_removed_card_mask.png", removed_card_mask)
    save_image(output_dir / "33_removed_card_overlay.png", draw_mask_overlay(image, removed_card_mask, color=(0, 0, 255), title="removed right-side card mask"))
    save_image(output_dir / "34_final_mask.png", final_mask)
    save_image(output_dir / "35_final_overlay.png", draw_mask_overlay(image, final_mask, color=(255, 255, 0), title="final recovered mask"))
    baseline_hull_overlay = draw_hull_overlay(image, baseline_mask_after_card_removal, title="baseline hull after card removal")
    final_hull_overlay = draw_hull_overlay(image, final_mask, title="final hull after shadow recovery")
    save_image(output_dir / "36_baseline_vs_final_hull.png", build_side_by_side(baseline_hull_overlay, final_hull_overlay))
    save_image(output_dir / "37_baseline_final_delta_overlay.png", build_result_delta_overlay(image, baseline_mask_after_card_removal, final_mask))
    write_summary(
        output_dir,
        input_path,
        component_info,
        green_seed_filtered,
        baseline_mask_after_card_removal,
        recovered_shadow_mask,
        merged_mask_before_card_removal,
        final_mask,
    )

    removed_components = [int(item["component_index"]) for item in component_info if bool(item["is_card_like"])]
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Green seed area: {int(cv2.countNonZero(green_seed_filtered))}")
    print(f"Baseline area after card removal: {int(cv2.countNonZero(baseline_mask_after_card_removal))}")
    print(f"Recovered shadow area: {int(cv2.countNonZero(recovered_shadow_mask))}")
    print(f"Merged area before card removal: {int(cv2.countNonZero(merged_mask_before_card_removal))}")
    print(f"Final area: {int(cv2.countNonZero(final_mask))}")
    print(f"Area gain vs baseline: {int(cv2.countNonZero(final_mask)) - int(cv2.countNonZero(baseline_mask_after_card_removal))}")
    print(f"Removed components: {removed_components}")
    print(f"Recovered components kept: {[int(item['component_index']) for item in added_component_info if bool(item['kept'])]}")


if __name__ == "__main__":
    main()