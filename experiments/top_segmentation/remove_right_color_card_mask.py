from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    parser = argparse.ArgumentParser(description="Experiment: remove right-side color-card mask from TOP segmentation.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="Input corrected TOP image path")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save outputs. Defaults to experiments/top_segmentation/output/remove_right_color_card_mask/<image-stem>",
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


def make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


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
        bbox_area = max(width * height, 1)
        fill_ratio = float(area) / float(bbox_area)
        center_x = float(centroids[component_index, 0])
        center_y = float(centroids[component_index, 1])
        aspect_ratio = float(width) / max(float(height), 1.0)
        contour_mask = np.zeros_like(mask)
        contour_mask[labels == component_index] = 255
        contours = find_external_contours(contour_mask)
        hull_area = 0.0
        solidity = 0.0
        if contours:
            merged = np.vstack(contours)
            hull = cv2.convexHull(merged)
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
                "aspect_ratio": round(aspect_ratio, 3),
                "fill_ratio": round(fill_ratio, 3),
                "solidity": round(solidity, 3),
                "is_right_side": is_right_side,
                "is_card_like": is_card_like,
            }
        )
    return labels, stats, metrics


def draw_component_overlays(image: np.ndarray, labels: np.ndarray, stats: np.ndarray, metrics: list[dict[str, float | int | bool]]) -> tuple[np.ndarray, np.ndarray]:
    overview = image.copy()
    card_candidates = image.copy()
    metric_map = {int(item["component_index"]): item for item in metrics}
    for component_index in range(1, stats.shape[0]):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        item = metric_map.get(component_index)
        if item is None:
            continue
        is_card_like = bool(item["is_card_like"])
        color = (0, 0, 255) if is_card_like else (0, 255, 255)
        cv2.rectangle(overview, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            overview,
            f"c{component_index}",
            (x, max(16, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        if is_card_like:
            cv2.rectangle(card_candidates, (x, y), (x + width, y + height), (0, 0, 255), 3)
            cv2.putText(
                card_candidates,
                f"card c{component_index}",
                (x, max(16, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
    return overview, card_candidates


def remove_card_like_components(mask: np.ndarray, labels: np.ndarray, metrics: list[dict[str, float | int | bool]]) -> np.ndarray:
    filtered = mask.copy()
    for item in metrics:
        if bool(item["is_card_like"]):
            filtered[labels == int(item["component_index"])] = 0
    return filtered


def build_removed_component_mask(mask: np.ndarray, labels: np.ndarray, metrics: list[dict[str, float | int | bool]]) -> np.ndarray:
    removed = np.zeros_like(mask)
    for item in metrics:
        if bool(item["is_card_like"]):
            removed[labels == int(item["component_index"])] = 255
    return removed


def draw_hull_overlay(image: np.ndarray, mask: np.ndarray, *, title: str) -> np.ndarray:
    canvas = image.copy()
    contours = find_external_contours(mask)
    if contours:
        cv2.drawContours(canvas, contours, -1, (0, 0, 255), 2)
        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)
        cv2.polylines(canvas, [hull], True, (255, 255, 0), 3)
        area = float(cv2.contourArea(hull))
        cv2.putText(canvas, f"hull_area={area:.1f}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(canvas, title, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def build_removed_overlay(image: np.ndarray, removed_mask: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    red = np.zeros_like(canvas)
    red[:, :, 2] = 255
    canvas = np.where(removed_mask[:, :, None] > 0, cv2.addWeighted(canvas, 0.35, red, 0.65, 0), canvas)
    cv2.putText(canvas, "removed right-side card mask", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def write_summary(output_dir: Path, input_path: Path, metrics: list[dict[str, float | int | bool]], before_mask: np.ndarray, after_mask: np.ndarray) -> None:
    before_contours = find_external_contours(before_mask)
    after_contours = find_external_contours(after_mask)
    before_hull_area = float(cv2.contourArea(cv2.convexHull(np.vstack(before_contours)))) if before_contours else 0.0
    after_hull_area = float(cv2.contourArea(cv2.convexHull(np.vstack(after_contours)))) if after_contours else 0.0
    payload = {
        "input": str(input_path),
        "component_metrics": metrics,
        "removed_components": [int(item["component_index"]) for item in metrics if bool(item["is_card_like"])],
        "before_mask_area": int(cv2.countNonZero(before_mask)),
        "after_mask_area": int(cv2.countNonZero(after_mask)),
        "before_hull_area": round(before_hull_area, 2),
        "after_hull_area": round(after_hull_area, 2),
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
        else Path("d:/code/pycharm/strawberry/experiments/top_segmentation/output/remove_right_color_card_mask") / input_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    image = read_image(input_path)
    normalized = normalize_image(image)
    denoised = denoise_image(normalized, method="gaussian", kernel_size=5)
    hsv_green_mask = build_hsv_green_mask(denoised)
    green_dominance_mask = build_green_dominance_mask(denoised)
    combined_mask = cv2.bitwise_and(hsv_green_mask, green_dominance_mask)
    cleaned_mask = clean_mask(combined_mask)
    filtered_mask = filter_small_components(cleaned_mask, min_component_area_ratio=0.0005)

    labels, stats, metrics = component_metrics(filtered_mask)
    components_overlay, card_candidates_overlay = draw_component_overlays(image, labels, stats, metrics)
    removed_mask = build_removed_component_mask(filtered_mask, labels, metrics)
    filtered_without_card = remove_card_like_components(filtered_mask, labels, metrics)

    save_image(output_dir / "01_original.png", image)
    save_image(output_dir / "02_normalized.png", normalized)
    save_image(output_dir / "03_denoised.png", denoised)
    save_image(output_dir / "04_hsv_green_mask.png", hsv_green_mask)
    save_image(output_dir / "05_green_dominance_mask.png", green_dominance_mask)
    save_image(output_dir / "06_combined_mask.png", combined_mask)
    save_image(output_dir / "07_cleaned_mask.png", cleaned_mask)
    save_image(output_dir / "08_filtered_mask_before_card_removal.png", filtered_mask)
    save_image(output_dir / "09_connected_components_overview.png", components_overlay)
    save_image(output_dir / "10_card_component_candidates.png", card_candidates_overlay)
    save_image(output_dir / "11_removed_card_mask.png", removed_mask)
    save_image(output_dir / "12_removed_card_overlay.png", build_removed_overlay(image, removed_mask))
    save_image(output_dir / "13_filtered_mask_after_card_removal.png", filtered_without_card)
    save_image(output_dir / "14_hull_before_card_removal.png", draw_hull_overlay(image, filtered_mask, title="before card removal"))
    save_image(output_dir / "15_hull_after_card_removal.png", draw_hull_overlay(image, filtered_without_card, title="after card removal"))
    write_summary(output_dir, input_path, metrics, filtered_mask, filtered_without_card)

    removed_components = [int(item["component_index"]) for item in metrics if bool(item["is_card_like"])]
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Total components: {len(metrics)}")
    print(f"Removed components: {removed_components}")
    print(f"Mask area before: {int(cv2.countNonZero(filtered_mask))}")
    print(f"Mask area after: {int(cv2.countNonZero(filtered_without_card))}")


if __name__ == "__main__":
    main()