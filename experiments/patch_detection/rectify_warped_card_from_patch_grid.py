from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from detect_patch_grid_from_best_variant import (
    EXPECTED_COLS,
    EXPECTED_ROWS,
    assign_boxes_to_affine_grid,
    assign_boxes_to_grid,
    build_affine_centers,
    build_affine_grid,
    build_patch_mask_variants,
    cluster_axis,
    draw_grid_assignments,
    estimate_lattice_model,
    filter_patch_boxes,
    infer_axis_positions,
    save_image,
)


DEFAULT_INPUT = Path(
    "d:/code/pycharm/strawberry/output/调试输出/1AB/01_色卡检测与校正/13_TOP_roi_variant_原始透视校正.png"
)

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

VARIANT_LABELS = [
    "原始",
    "逆时针90°",
    "180°",
    "顺时针90°",
    "镜像",
    "镜像+逆时针90°",
    "镜像+180°",
    "镜像+顺时针90°",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Current experiment: rectify a warped color card using the detected patch lattice.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="Input warped-card image path")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save outputs. Defaults to experiments/patch_detection/output/rectify_warped_card_from_patch_grid/<image-stem>",
    )
    parser.add_argument(
        "--mask-branch",
        default="open",
        choices=["direct", "open", "open_close"],
        help="Mask branch used for connected-component patch detection",
    )
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def generate_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = []
    for flip_index, should_flip in enumerate((False, True)):
        base = cv2.flip(image, 1) if should_flip else image
        for rotation in range(4):
            rotated = np.rot90(base, rotation).copy()
            resized = cv2.resize(rotated, (360, 240))
            label_index = flip_index * 4 + rotation
            variants.append((VARIANT_LABELS[label_index], resized))
    return variants


def detect_lattice(image: np.ndarray, mask_branch: str) -> dict[str, object]:
    threshold_value, mask_variants = build_patch_mask_variants(image)
    selected_mask = mask_variants[mask_branch]
    component_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(selected_mask, 8)
    boxes = filter_patch_boxes(image, stats, centroids)

    nominal_w = float(np.median([box.w for box in boxes])) if boxes else image.shape[1] / EXPECTED_COLS
    nominal_h = float(np.median([box.h for box in boxes])) if boxes else image.shape[0] / EXPECTED_ROWS

    row_groups = cluster_axis([box.center_y for box in boxes], threshold=max(8.0, nominal_h * 0.45))
    col_groups = cluster_axis([box.center_x for box in boxes], threshold=max(8.0, nominal_w * 0.45))
    row_centers = infer_axis_positions(row_groups, EXPECTED_ROWS, nominal_h * 1.25)
    col_centers = infer_axis_positions(col_groups, EXPECTED_COLS, nominal_w * 1.25)

    initial_assignments = assign_boxes_to_grid(boxes, row_centers, col_centers)
    lattice_model = estimate_lattice_model(initial_assignments, nominal_w, nominal_h)
    affine_centers = build_affine_centers(lattice_model)
    assignments = assign_boxes_to_affine_grid(boxes, affine_centers)
    lattice_model = estimate_lattice_model(assignments, nominal_w, nominal_h)
    affine_centers = build_affine_centers(lattice_model)
    inferred_grid = build_affine_grid(lattice_model)

    return {
        "threshold_value": float(threshold_value),
        "mask_variants": mask_variants,
        "selected_mask": selected_mask,
        "component_count": int(component_count),
        "boxes": boxes,
        "row_centers": row_centers,
        "col_centers": col_centers,
        "assignments": assignments,
        "model": lattice_model,
        "affine_centers": affine_centers,
        "inferred_grid": inferred_grid,
    }


def choose_best_variant(image: np.ndarray, mask_branch: str) -> tuple[str, np.ndarray, dict[str, object], list[tuple[str, int, int]]]:
    scored: list[tuple[str, np.ndarray, dict[str, object], tuple[int, int]]] = []
    summaries: list[tuple[str, int, int]] = []
    for label, variant in generate_variants(image):
        detected = detect_lattice(variant, mask_branch)
        score = (len(detected["assignments"]), len(detected["boxes"]))
        summaries.append((label, score[0], score[1]))
        scored.append((label, variant, detected, score))

    best_label, best_variant, best_detected, _score = max(scored, key=lambda item: item[3])
    return best_label, best_variant, best_detected, summaries


def build_target_centers(image_shape: tuple[int, int, int]) -> dict[tuple[int, int], np.ndarray]:
    height, width = image_shape[:2]
    margin_x = max(16.0, width * 0.18)
    margin_y = max(20.0, height * 0.14)
    step_x = (width - 1 - 2.0 * margin_x) / max(EXPECTED_COLS - 1, 1)
    step_y = (height - 1 - 2.0 * margin_y) / max(EXPECTED_ROWS - 1, 1)

    centers: dict[tuple[int, int], np.ndarray] = {}
    for row_index in range(EXPECTED_ROWS):
        for col_index in range(EXPECTED_COLS):
            centers[(row_index, col_index)] = np.array(
                [margin_x + col_index * step_x, margin_y + row_index * step_y],
                dtype=np.float32,
            )
    return centers


def build_target_grid(target_centers: dict[tuple[int, int], np.ndarray], image_shape: tuple[int, int, int]) -> dict[tuple[int, int], np.ndarray]:
    height, width = image_shape[:2]
    step_x = (width - 1 - 2.0 * target_centers[(0, 0)][0]) / max(EXPECTED_COLS - 1, 1)
    step_y = (height - 1 - 2.0 * target_centers[(0, 0)][1]) / max(EXPECTED_ROWS - 1, 1)
    half_w = step_x * 0.37
    half_h = step_y * 0.37

    grid: dict[tuple[int, int], np.ndarray] = {}
    for key, center in target_centers.items():
        cx, cy = float(center[0]), float(center[1])
        grid[key] = np.array(
            [
                [cx - half_w, cy - half_h],
                [cx + half_w, cy - half_h],
                [cx + half_w, cy + half_h],
                [cx - half_w, cy + half_h],
            ],
            dtype=np.float32,
        )
    return grid


def sample_patch_rgb(image: np.ndarray, grid: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    rgb_image = image[:, :, ::-1].astype(np.float32)
    patch_values: list[np.ndarray] = []
    height, width = image.shape[:2]
    for row in range(EXPECTED_ROWS):
        for col in range(EXPECTED_COLS):
            polygon = np.round(grid[(row, col)]).astype(np.int32)
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, polygon, 255)
            pixels = rgb_image[mask > 0]
            if pixels.size == 0:
                patch_values.append(np.zeros(3, dtype=np.float32))
            else:
                patch_values.append(pixels.reshape(-1, 3).mean(axis=0))
    return np.asarray(patch_values, dtype=np.float32)


def fit_color_matrix(observed_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    design = np.hstack([observed_rgb, np.ones((observed_rgb.shape[0], 1), dtype=np.float32)])
    matrix, _, _, _ = np.linalg.lstsq(design, reference_rgb.astype(np.float32), rcond=None)
    return matrix.T.astype(np.float32)


def apply_color_matrix_to_rgb(observed_rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    design = np.hstack([observed_rgb, np.ones((observed_rgb.shape[0], 1), dtype=np.float32)])
    corrected = design @ matrix.T
    return np.clip(corrected, 0, 255).astype(np.float32)


def apply_color_matrix_to_image(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    rgb = image[:, :, ::-1].astype(np.float32)
    design = np.concatenate([rgb, np.ones((*rgb.shape[:2], 1), dtype=np.float32)], axis=2)
    corrected = np.tensordot(design, matrix.T, axes=([2], [0]))
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected[:, :, ::-1]


def evaluate_patch_fit(image: np.ndarray, grid: dict[tuple[int, int], np.ndarray]) -> dict[str, object]:
    observed_rgb = sample_patch_rgb(image, grid)
    correction_matrix = fit_color_matrix(observed_rgb, REFERENCE_PATCH_RGB)
    corrected_rgb = apply_color_matrix_to_rgb(observed_rgb, correction_matrix)
    mse = float(np.mean((corrected_rgb - REFERENCE_PATCH_RGB) ** 2))
    mae = float(np.mean(np.abs(corrected_rgb - REFERENCE_PATCH_RGB)))
    return {
        "observed_rgb": observed_rgb,
        "corrected_rgb": corrected_rgb,
        "correction_matrix": correction_matrix,
        "mse": mse,
        "mae": mae,
    }


def estimate_rectification(
    assignments: dict[tuple[int, int], object],
    target_centers: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    source_points: list[np.ndarray] = []
    destination_points: list[np.ndarray] = []

    for key, patch in assignments.items():
        source_points.append(np.array([patch.center_x, patch.center_y], dtype=np.float32))
        destination_points.append(target_centers[key])

    if len(source_points) < 4:
        raise ValueError(f"Not enough assigned cells for homography: {len(source_points)}")

    src = np.asarray(source_points, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(destination_points, dtype=np.float32).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if homography is None or inlier_mask is None:
        raise ValueError("Failed to estimate homography from patch centers")

    projected = cv2.perspectiveTransform(src, homography).reshape(-1, 2)
    dst_points = dst.reshape(-1, 2)
    errors = np.linalg.norm(projected - dst_points, axis=1)
    mean_error = float(errors.mean()) if len(errors) else 0.0
    return homography, inlier_mask.reshape(-1), mean_error


def draw_target_grid(image_shape: tuple[int, int, int], target_grid: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    canvas = np.full(image_shape, 24, dtype=np.uint8)
    for row in range(EXPECTED_ROWS):
        for col in range(EXPECTED_COLS):
            polygon = np.round(target_grid[(row, col)]).astype(np.int32)
            cv2.polylines(canvas, [polygon], True, (0, 220, 255), 2)
            center = np.mean(polygon, axis=0)
            cv2.putText(
                canvas,
                f"{row},{col}",
                (int(round(center[0])) - 14, int(round(center[1])) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 255),
                1,
            )
    return canvas


def draw_patch_overlay(
    image: np.ndarray,
    grid: dict[tuple[int, int], np.ndarray],
    assigned_cells: set[tuple[int, int]],
    observed_rgb: np.ndarray,
) -> np.ndarray:
    canvas = image.copy()
    for row in range(EXPECTED_ROWS):
        for col in range(EXPECTED_COLS):
            index = row * EXPECTED_COLS + col
            polygon = np.round(grid[(row, col)]).astype(np.int32)
            assigned = (row, col) in assigned_cells
            color = (0, 255, 0) if assigned else (0, 128, 255)
            cv2.polylines(canvas, [polygon], True, color, 2)
            center = np.mean(polygon, axis=0)
            cx = int(round(center[0]))
            cy = int(round(center[1]))
            sample_rgb = observed_rgb[index]
            swatch_bgr = tuple(int(round(value)) for value in sample_rgb[::-1])
            cv2.rectangle(canvas, (cx - 10, cy - 10), (cx + 10, cy + 10), swatch_bgr, -1)
            cv2.rectangle(canvas, (cx - 10, cy - 10), (cx + 10, cy + 10), (255, 255, 255), 1)
            cv2.putText(canvas, str(index), (cx - 8, cy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return canvas


def build_patch_color_panel(title: str, observed_rgb: np.ndarray, corrected_rgb: np.ndarray) -> np.ndarray:
    cell_w = 56
    cell_h = 40
    cols = EXPECTED_COLS
    rows = EXPECTED_ROWS
    width = 28 + cols * cell_w * 3
    height = 50 + rows * cell_h
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 255), 2)
    headers = ["obs", "corr", "ref"]
    for block_index, header in enumerate(headers):
        cv2.putText(canvas, header, (35 + block_index * cols * cell_w, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            colors = [observed_rgb[index], corrected_rgb[index], REFERENCE_PATCH_RGB[index]]
            for block_index, rgb in enumerate(colors):
                x0 = 20 + block_index * cols * cell_w + col * cell_w
                y0 = 54 + row * cell_h
                bgr = tuple(int(round(value)) for value in rgb[::-1])
                cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 8, y0 + cell_h - 8), bgr, -1)
                cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 8, y0 + cell_h - 8), (255, 255, 255), 1)
                if block_index == 0:
                    cv2.putText(canvas, str(index), (x0 + 3, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return canvas


def build_metrics_panel(before_metrics: dict[str, object], after_metrics: dict[str, object]) -> np.ndarray:
    width = 720
    height = 150
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, "Rectification Metrics", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)

    before_mse = float(before_metrics["mse"])
    after_mse = float(after_metrics["mse"])
    before_mae = float(before_metrics["mae"])
    after_mae = float(after_metrics["mae"])
    delta_mse = before_mse - after_mse
    delta_mae = before_mae - after_mae

    lines = [
        f"MSE: before={before_mse:.2f} after={after_mse:.2f} delta={delta_mse:+.2f}",
        f"MAE: before={before_mae:.2f} after={after_mae:.2f} delta={delta_mae:+.2f}",
        "Delta > 0 means the second rectification improved color fit.",
    ]
    for index, text in enumerate(lines):
        color = (0, 255, 0) if index < 2 and ((index == 0 and delta_mse > 0) or (index == 1 and delta_mae > 0)) else (220, 220, 220)
        cv2.putText(canvas, text, (18, 62 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2 if index < 2 else 1)
    return canvas


def build_variant_summary_panel(summaries: list[tuple[str, int, int]]) -> np.ndarray:
    line_h = 28
    width = 520
    height = 20 + line_h * len(summaries)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    for index, (label, assigned, candidates) in enumerate(summaries):
        y = 24 + index * line_h
        text = f"{label}: assigned={assigned}/24 candidates={candidates}"
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2)
    return canvas


def build_comparison(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    separator = np.full((before.shape[0], 12, 3), 40, dtype=np.uint8)
    return np.hstack([before, separator, after])


def write_summary(
    output_dir: Path,
    input_path: Path,
    variant_label: str,
    variant_summaries: list[tuple[str, int, int]],
    before: dict[str, object],
    after: dict[str, object],
    before_metrics: dict[str, object],
    after_metrics: dict[str, object],
    homography: np.ndarray,
    inlier_mask: np.ndarray,
    reprojection_error: float,
    mask_branch: str,
) -> None:
    payload = {
        "input": str(input_path),
        "selected_variant": variant_label,
        "variant_scores": [
            {"label": label, "assigned": assigned, "candidates": candidates}
            for label, assigned, candidates in variant_summaries
        ],
        "mask_branch": mask_branch,
        "before_candidates": len(before["boxes"]),
        "before_assigned": len(before["assignments"]),
        "after_candidates": len(after["boxes"]),
        "after_assigned": len(after["assignments"]),
        "before_patch_mse": round(float(before_metrics["mse"]), 4),
        "before_patch_mae": round(float(before_metrics["mae"]), 4),
        "after_patch_mse": round(float(after_metrics["mse"]), 4),
        "after_patch_mae": round(float(after_metrics["mae"]), 4),
        "homography_inliers": int(inlier_mask.sum()),
        "homography_total": int(inlier_mask.shape[0]),
        "mean_reprojection_error": round(reprojection_error, 4),
        "homography": [[round(float(value), 6) for value in row] for row in homography],
        "before_assigned_cells": sorted([f"{row},{col}" for row, col in before["assignments"].keys()]),
        "after_assigned_cells": sorted([f"{row},{col}" for row, col in after["assignments"].keys()]),
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
        else Path("d:/code/pycharm/strawberry/experiments/patch_detection/output/rectify_warped_card_from_patch_grid") / input_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    image = read_image(input_path)
    variant_label, oriented_image, before, variant_summaries = choose_best_variant(image, args.mask_branch)
    before_overlay = draw_grid_assignments(oriented_image, before["inferred_grid"], before["assignments"])

    target_centers = build_target_centers(oriented_image.shape)
    target_grid = build_target_grid(target_centers, oriented_image.shape)
    homography, inlier_mask, reprojection_error = estimate_rectification(before["assignments"], target_centers)
    rectified = cv2.warpPerspective(oriented_image, homography, (oriented_image.shape[1], oriented_image.shape[0]))
    after = detect_lattice(rectified, args.mask_branch)
    after_overlay = draw_grid_assignments(rectified, after["inferred_grid"], after["assignments"])
    before_metrics = evaluate_patch_fit(oriented_image, before["inferred_grid"])
    after_metrics = evaluate_patch_fit(rectified, after["inferred_grid"])
    before_patch_overlay = draw_patch_overlay(
        oriented_image,
        before["inferred_grid"],
        set(before["assignments"].keys()),
        before_metrics["observed_rgb"],
    )
    after_patch_overlay = draw_patch_overlay(
        rectified,
        after["inferred_grid"],
        set(after["assignments"].keys()),
        after_metrics["observed_rgb"],
    )
    before_corrected = apply_color_matrix_to_image(oriented_image, before_metrics["correction_matrix"])
    after_corrected = apply_color_matrix_to_image(rectified, after_metrics["correction_matrix"])

    save_image(output_dir / "01_original.png", image)
    save_image(output_dir / "02_variant_summary.png", build_variant_summary_panel(variant_summaries))
    save_image(output_dir / "03_oriented_variant.png", oriented_image)
    save_image(output_dir / "04_selected_patch_mask.png", before["selected_mask"])
    save_image(output_dir / "05_grid_before_rectify.png", before_overlay)
    save_image(output_dir / "06_target_grid.png", draw_target_grid(oriented_image.shape, target_grid))
    save_image(output_dir / "07_rectified.png", rectified)
    save_image(output_dir / "08_grid_after_rectify.png", after_overlay)
    save_image(output_dir / "09_patch_overlay_before.png", before_patch_overlay)
    save_image(output_dir / "10_patch_overlay_after.png", after_patch_overlay)
    save_image(output_dir / "11_corrected_before.png", before_corrected)
    save_image(output_dir / "12_corrected_after.png", after_corrected)
    save_image(output_dir / "13_patch_colors_before.png", build_patch_color_panel("Before Rectify", before_metrics["observed_rgb"], before_metrics["corrected_rgb"]))
    save_image(output_dir / "14_patch_colors_after.png", build_patch_color_panel("After Rectify", after_metrics["observed_rgb"], after_metrics["corrected_rgb"]))
    save_image(output_dir / "15_metrics.png", build_metrics_panel(before_metrics, after_metrics))
    save_image(output_dir / "16_before_after.png", build_comparison(before_patch_overlay, after_patch_overlay))
    write_summary(output_dir, input_path, variant_label, variant_summaries, before, after, before_metrics, after_metrics, homography, inlier_mask, reprojection_error, args.mask_branch)

    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Selected variant: {variant_label}")
    print(f"Mask branch: {args.mask_branch}")
    print(f"Before assigned cells: {len(before['assignments'])}/{EXPECTED_ROWS * EXPECTED_COLS}")
    print(f"After assigned cells: {len(after['assignments'])}/{EXPECTED_ROWS * EXPECTED_COLS}")
    print(f"Before patch MSE: {float(before_metrics['mse']):.2f}")
    print(f"After patch MSE: {float(after_metrics['mse']):.2f}")
    print(f"Homography inliers: {int(inlier_mask.sum())}/{int(inlier_mask.shape[0])}")
    print(f"Mean reprojection error: {reprojection_error:.3f}")


if __name__ == "__main__":
    main()
