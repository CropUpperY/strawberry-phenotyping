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
    parser = argparse.ArgumentParser(description="Experiment: refine TOP strawberry mask with GrabCut.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="Input corrected TOP image path")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to save outputs. Defaults to "
            "experiments/top_segmentation/output/refine_top_mask_with_grabcut/<image-stem>"
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
        contour_mask = np.zeros_like(mask)
        contour_mask[labels == component_index] = 255
        contours = find_external_contours(contour_mask)
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
    cv2.putText(tinted, "Yellow=overlap  Green=added  Red=removed", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(tinted, "Baseline contour=red  Final contour=green", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return tinted


def apply_clahe_to_l_channel(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_l, enhanced_bgr


def build_shadow_candidate_mask(image: np.ndarray, baseline_mask: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    enhanced_l, enhanced_bgr = apply_clahe_to_l_channel(image)
    enhanced_lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
    l_channel = enhanced_lab[:, :, 0]
    a_channel = enhanced_lab[:, :, 1]
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

    mean_gray = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 5)
    mean_sq_gray = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (0, 0), 5)
    local_std = np.sqrt(np.clip(mean_sq_gray - mean_gray**2, 0.0, None))
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

    grow_size = make_odd(max(31, int(round(min(image.shape[:2]) * 0.08))))
    grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_size, grow_size))
    seed_neighborhood = cv2.dilate(baseline_mask, grow_kernel, iterations=1)
    constrained_shadow = cv2.bitwise_and(shadow_raw, seed_neighborhood)

    return constrained_shadow, {
        "illumination_enhanced_l": enhanced_l,
        "illumination_enhanced_bgr": enhanced_bgr,
        "shadow_dark_mask": dark_mask,
        "shadow_weak_green_lab_mask": weak_green_lab_mask,
        "shadow_texture_mask": texture_mask,
        "shadow_gradient_mask": gradient_mask,
        "shadow_raw": shadow_raw,
        "shadow_seed_neighborhood": seed_neighborhood,
        "shadow_constrained": constrained_shadow,
    }


def build_probable_foreground(baseline_mask: np.ndarray, shadow_candidate_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    contours = find_external_contours(baseline_mask)
    hull_mask = np.zeros_like(baseline_mask)
    if contours:
        hull = cv2.convexHull(np.vstack(contours))
        cv2.fillConvexPoly(hull_mask, hull, 255)
    expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (make_odd(max(61, int(round(min(baseline_mask.shape) * 0.10)))),) * 2)
    expanded_hull = cv2.dilate(hull_mask, expand_kernel, iterations=1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (make_odd(max(41, int(round(min(baseline_mask.shape) * 0.06)))),) * 2)
    baseline_dilated = cv2.dilate(baseline_mask, dilate_kernel, iterations=1)
    probable_foreground = cv2.bitwise_or(baseline_dilated, shadow_candidate_mask)
    probable_foreground = cv2.bitwise_and(probable_foreground, expanded_hull)
    return hull_mask, expanded_hull, baseline_dilated, probable_foreground


def build_sure_foreground(baseline_mask: np.ndarray) -> np.ndarray:
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (make_odd(max(31, int(round(min(baseline_mask.shape) * 0.05)))),) * 2)
    return cv2.erode(baseline_mask, erode_kernel, iterations=1)


def build_sure_background(image: np.ndarray, baseline_mask: np.ndarray, removed_card_mask: np.ndarray, probable_foreground: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = baseline_mask.shape
    border_mask = np.zeros_like(baseline_mask)
    border_thickness = max(20, int(round(min(height, width) * 0.05)))
    border_mask[:border_thickness, :] = 255
    border_mask[-border_thickness:, :] = 255
    border_mask[:, :border_thickness] = 255
    border_mask[:, -border_thickness:] = 255

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark_background = cv2.inRange(hsv[:, :, 2], 0, 70)
    low_sat_background = cv2.inRange(hsv[:, :, 1], 0, 70)
    dark_low_sat = cv2.bitwise_and(dark_background, low_sat_background)

    sure_background = cv2.bitwise_or(border_mask, removed_card_mask)
    sure_background = cv2.bitwise_or(sure_background, cv2.bitwise_and(dark_low_sat, cv2.bitwise_not(probable_foreground)))
    sure_background = cv2.bitwise_and(sure_background, cv2.bitwise_not(baseline_mask))
    return border_mask, dark_low_sat, sure_background


def build_grabcut_init_mask(
    baseline_mask: np.ndarray,
    sure_foreground: np.ndarray,
    probable_foreground: np.ndarray,
    sure_background: np.ndarray,
) -> np.ndarray:
    gc_mask = np.full(baseline_mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[probable_foreground > 0] = cv2.GC_PR_FGD
    gc_mask[sure_background > 0] = cv2.GC_BGD
    gc_mask[sure_foreground > 0] = cv2.GC_FGD
    return gc_mask


def visualize_grabcut_classes(gc_mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*gc_mask.shape, 3), dtype=np.uint8)
    canvas[gc_mask == cv2.GC_BGD] = (0, 0, 0)
    canvas[gc_mask == cv2.GC_PR_BGD] = (64, 64, 64)
    canvas[gc_mask == cv2.GC_PR_FGD] = (0, 180, 255)
    canvas[gc_mask == cv2.GC_FGD] = (0, 255, 0)
    cv2.putText(canvas, "BG=black  PR_BG=gray  PR_FG=orange  FG=green", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def run_grabcut(image: np.ndarray, gc_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, gc_mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return gc_mask, final_mask


def write_summary(output_dir: Path, input_path: Path, baseline_mask: np.ndarray, final_mask: np.ndarray) -> None:
    def hull_area(mask: np.ndarray) -> float:
        contours = find_external_contours(mask)
        if not contours:
            return 0.0
        return float(cv2.contourArea(cv2.convexHull(np.vstack(contours))))

    payload = {
        "input": str(input_path),
        "baseline_area_after_card_removal": int(cv2.countNonZero(baseline_mask)),
        "final_area": int(cv2.countNonZero(final_mask)),
        "baseline_hull_area_after_card_removal": round(hull_area(baseline_mask), 2),
        "final_hull_area_after_grabcut": round(hull_area(final_mask), 2),
        "area_gain_vs_baseline": int(cv2.countNonZero(final_mask)) - int(cv2.countNonZero(baseline_mask)),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_file() and (child.suffix.lower() == ".png" or child.name == "summary.json"):
            child.unlink()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("d:/code/pycharm/strawberry/experiments/top_segmentation/output/refine_top_mask_with_grabcut") / input_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)

    image = read_image(input_path)
    normalized = normalize_image(image)
    denoised = denoise_image(normalized, method="gaussian", kernel_size=5)

    hsv_green_mask = build_hsv_green_mask(denoised)
    green_dominance_mask = build_green_dominance_mask(denoised)
    baseline_raw = cv2.bitwise_and(hsv_green_mask, green_dominance_mask)
    baseline_cleaned = clean_mask(baseline_raw)
    baseline_filtered = filter_small_components(baseline_cleaned, min_component_area_ratio=0.0005)

    labels, _stats, metrics = component_metrics(baseline_filtered)
    baseline_mask, removed_card_mask = remove_card_like_components(baseline_filtered, labels, metrics)

    shadow_candidate_mask, shadow_debug = build_shadow_candidate_mask(denoised, baseline_mask)
    hull_mask, expanded_hull, baseline_dilated, probable_foreground = build_probable_foreground(baseline_mask, shadow_candidate_mask)
    sure_foreground = build_sure_foreground(baseline_mask)
    border_mask, dark_low_sat, sure_background = build_sure_background(denoised, baseline_mask, removed_card_mask, probable_foreground)
    grabcut_init = build_grabcut_init_mask(baseline_mask, sure_foreground, probable_foreground, sure_background)
    refined_gc_mask, final_mask = run_grabcut(denoised, grabcut_init.copy())
    final_mask = clean_mask(final_mask)
    final_mask = filter_small_components(final_mask, min_component_area_ratio=0.0003)

    save_image(output_dir / "01_original.png", image)
    save_image(output_dir / "02_normalized.png", normalized)
    save_image(output_dir / "03_denoised.png", denoised)
    save_image(output_dir / "04_hsv_green_mask.png", hsv_green_mask)
    save_image(output_dir / "05_green_dominance_mask.png", green_dominance_mask)
    save_image(output_dir / "06_baseline_raw.png", baseline_raw)
    save_image(output_dir / "07_baseline_cleaned.png", baseline_cleaned)
    save_image(output_dir / "08_baseline_filtered.png", baseline_filtered)
    save_image(output_dir / "09_removed_card_mask.png", removed_card_mask)
    save_image(output_dir / "10_baseline_after_card_removal.png", baseline_mask)
    save_image(output_dir / "11_baseline_overlay.png", draw_mask_overlay(image, baseline_mask, color=(0, 255, 255), title="baseline after card removal"))
    save_image(output_dir / "12_illumination_enhanced_l.png", shadow_debug["illumination_enhanced_l"])
    save_image(output_dir / "13_illumination_enhanced_bgr.png", shadow_debug["illumination_enhanced_bgr"])
    save_image(output_dir / "14_shadow_dark_mask.png", shadow_debug["shadow_dark_mask"])
    save_image(output_dir / "15_shadow_weak_green_lab_mask.png", shadow_debug["shadow_weak_green_lab_mask"])
    save_image(output_dir / "16_shadow_texture_mask.png", shadow_debug["shadow_texture_mask"])
    save_image(output_dir / "17_shadow_gradient_mask.png", shadow_debug["shadow_gradient_mask"])
    save_image(output_dir / "18_shadow_raw.png", shadow_debug["shadow_raw"])
    save_image(output_dir / "19_shadow_seed_neighborhood.png", shadow_debug["shadow_seed_neighborhood"])
    save_image(output_dir / "20_shadow_constrained.png", shadow_debug["shadow_constrained"])
    save_image(output_dir / "21_hull_mask.png", hull_mask)
    save_image(output_dir / "22_expanded_hull.png", expanded_hull)
    save_image(output_dir / "23_baseline_dilated.png", baseline_dilated)
    save_image(output_dir / "24_probable_foreground.png", probable_foreground)
    save_image(output_dir / "25_sure_foreground.png", sure_foreground)
    save_image(output_dir / "26_border_background.png", border_mask)
    save_image(output_dir / "27_dark_low_sat_background.png", dark_low_sat)
    save_image(output_dir / "28_sure_background.png", sure_background)
    save_image(output_dir / "29_grabcut_init_classes.png", visualize_grabcut_classes(grabcut_init))
    save_image(output_dir / "30_grabcut_result_classes.png", visualize_grabcut_classes(refined_gc_mask))
    save_image(output_dir / "31_final_mask.png", final_mask)
    save_image(output_dir / "32_final_overlay.png", draw_mask_overlay(image, final_mask, color=(0, 255, 0), title="final grabcut mask"))
    baseline_hull = draw_hull_overlay(image, baseline_mask, title="baseline hull after card removal")
    final_hull = draw_hull_overlay(image, final_mask, title="final hull after grabcut")
    save_image(output_dir / "33_baseline_vs_final_hull.png", build_side_by_side(baseline_hull, final_hull))
    save_image(output_dir / "34_baseline_final_delta_overlay.png", build_result_delta_overlay(image, baseline_mask, final_mask))
    write_summary(output_dir, input_path, baseline_mask, final_mask)

    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Baseline area after card removal: {int(cv2.countNonZero(baseline_mask))}")
    print(f"Final area: {int(cv2.countNonZero(final_mask))}")
    print(f"Area gain vs baseline: {int(cv2.countNonZero(final_mask)) - int(cv2.countNonZero(baseline_mask))}")


if __name__ == "__main__":
    main()