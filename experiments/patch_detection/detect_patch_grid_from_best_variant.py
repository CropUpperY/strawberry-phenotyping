from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT = Path(
    "d:/code/pycharm/strawberry/output/调试输出/1AB/01_色卡检测与校正/15_TOP_roi_variant_最佳变体.png"
)
EXPECTED_ROWS = 4
EXPECTED_COLS = 6


@dataclass(slots=True)
class PatchBox:
    x: int
    y: int
    w: int
    h: int
    area: float
    center_x: float
    center_y: float


@dataclass(slots=True)
class LatticeModel:
    origin: np.ndarray
    col_vec: np.ndarray
    row_vec: np.ndarray
    half_col_vec: np.ndarray
    half_row_vec: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy experiment: detect the 24-patch grid from an already-selected best-variant image.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="Input best-variant image path")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save intermediate outputs. Defaults to experiments/patch_detection/output/detect_patch_grid_from_best_variant/<image-stem>",
    )
    return parser.parse_args()


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def build_edges(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = normalize_gray(gray)
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return normalized, edges, closed


def build_patch_mask_variants(image: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold_value, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    open_close_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return float(threshold_value), {
        "direct": otsu_mask,
        "open": opened_mask,
        "open_close": open_close_mask,
    }


def component_to_patch_box(stats_row: np.ndarray, centroid_row: np.ndarray) -> PatchBox | None:
    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])
    area = float(stats_row[cv2.CC_STAT_AREA])
    if w <= 0 or h <= 0:
        return None
    return PatchBox(
        x=x,
        y=y,
        w=w,
        h=h,
        area=area,
        center_x=float(centroid_row[0]),
        center_y=float(centroid_row[1]),
    )


def filter_patch_boxes(image: np.ndarray, stats: np.ndarray, centroids: np.ndarray) -> list[PatchBox]:
    image_area = float(image.shape[0] * image.shape[1])
    boxes: list[PatchBox] = []
    for index in range(1, stats.shape[0]):
        patch = component_to_patch_box(stats[index], centroids[index])
        if patch is None:
            continue
        if patch.area < image_area * 0.008 or patch.area > image_area * 0.04:
            continue
        aspect_ratio = patch.w / max(1.0, float(patch.h))
        if not 0.55 <= aspect_ratio <= 1.65:
            continue
        if patch.w < image.shape[1] * 0.06 or patch.h < image.shape[0] * 0.08:
            continue
        if patch.w > image.shape[1] * 0.22 or patch.h > image.shape[0] * 0.28:
            continue
        fill_ratio = patch.area / max(float(patch.w * patch.h), 1.0)
        if fill_ratio < 0.72:
            continue
        boxes.append(patch)
    return suppress_overlaps(boxes)


def box_iou(first: PatchBox, second: PatchBox) -> float:
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


def suppress_overlaps(boxes: list[PatchBox]) -> list[PatchBox]:
    kept: list[PatchBox] = []
    for patch in sorted(boxes, key=lambda item: item.area, reverse=True):
        if any(box_iou(patch, existing) > 0.35 for existing in kept):
            continue
        kept.append(patch)
    return sorted(kept, key=lambda item: (item.center_y, item.center_x))


def cluster_axis(values: list[float], threshold: float) -> list[float]:
    if not values:
        return []
    sorted_values = sorted(values)
    groups: list[list[float]] = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if abs(value - np.mean(groups[-1])) <= threshold:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.mean(group)) for group in groups]


def infer_axis_positions(centers: list[float], expected_count: int, nominal_step: float) -> list[float]:
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
    start = centers[0]
    best_positions = [start + inferred_step * index for index in range(expected_count)]
    best_cost = axis_fit_cost(centers, best_positions)

    for anchor_index in range(expected_count):
        start = centers[0] - inferred_step * anchor_index
        candidate = [start + inferred_step * index for index in range(expected_count)]
        cost = axis_fit_cost(centers, candidate)
        if cost < best_cost:
            best_cost = cost
            best_positions = candidate
    return best_positions


def axis_fit_cost(detected: list[float], candidate: list[float]) -> float:
    total = 0.0
    for value in detected:
        total += min(abs(value - item) for item in candidate)
    return total


def assign_boxes_to_grid(
    boxes: list[PatchBox],
    row_centers: list[float],
    col_centers: list[float],
) -> dict[tuple[int, int], PatchBox]:
    assignments: dict[tuple[int, int], PatchBox] = {}
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


def estimate_lattice_model(
    assignments: dict[tuple[int, int], PatchBox],
    nominal_w: float,
    nominal_h: float,
) -> LatticeModel:
    if len(assignments) < 3:
        return LatticeModel(
            origin=np.array([nominal_w / 2.0, nominal_h / 2.0], dtype=np.float32),
            col_vec=np.array([nominal_w * 1.15, 0.0], dtype=np.float32),
            row_vec=np.array([0.0, nominal_h * 1.15], dtype=np.float32),
            half_col_vec=np.array([nominal_w / 2.0, 0.0], dtype=np.float32),
            half_row_vec=np.array([0.0, nominal_h / 2.0], dtype=np.float32),
        )

    design = []
    target_x = []
    target_y = []
    for (row_index, col_index), patch in assignments.items():
        design.append([1.0, float(col_index), float(row_index)])
        target_x.append(float(patch.center_x))
        target_y.append(float(patch.center_y))

    design_matrix = np.asarray(design, dtype=np.float32)
    coeff_x, _, _, _ = np.linalg.lstsq(design_matrix, np.asarray(target_x, dtype=np.float32), rcond=None)
    coeff_y, _, _, _ = np.linalg.lstsq(design_matrix, np.asarray(target_y, dtype=np.float32), rcond=None)

    origin = np.array([coeff_x[0], coeff_y[0]], dtype=np.float32)
    col_vec = np.array([coeff_x[1], coeff_y[1]], dtype=np.float32)
    row_vec = np.array([coeff_x[2], coeff_y[2]], dtype=np.float32)

    col_len = max(float(np.linalg.norm(col_vec)), 1.0)
    row_len = max(float(np.linalg.norm(row_vec)), 1.0)
    half_col_vec = col_vec * (nominal_w / col_len) * 0.5
    half_row_vec = row_vec * (nominal_h / row_len) * 0.5

    return LatticeModel(
        origin=origin,
        col_vec=col_vec,
        row_vec=row_vec,
        half_col_vec=half_col_vec,
        half_row_vec=half_row_vec,
    )


def build_affine_grid(model: LatticeModel) -> dict[tuple[int, int], np.ndarray]:
    grid: dict[tuple[int, int], np.ndarray] = {}
    for row_index in range(EXPECTED_ROWS):
        for col_index in range(EXPECTED_COLS):
            center = model.origin + col_index * model.col_vec + row_index * model.row_vec
            corners = np.array(
                [
                    center - model.half_col_vec - model.half_row_vec,
                    center + model.half_col_vec - model.half_row_vec,
                    center + model.half_col_vec + model.half_row_vec,
                    center - model.half_col_vec + model.half_row_vec,
                ],
                dtype=np.float32,
            )
            grid[(row_index, col_index)] = corners
    return grid


def build_affine_centers(model: LatticeModel) -> dict[tuple[int, int], np.ndarray]:
    centers: dict[tuple[int, int], np.ndarray] = {}
    for row_index in range(EXPECTED_ROWS):
        for col_index in range(EXPECTED_COLS):
            centers[(row_index, col_index)] = model.origin + col_index * model.col_vec + row_index * model.row_vec
    return centers


def assign_boxes_to_affine_grid(
    boxes: list[PatchBox],
    affine_centers: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], PatchBox]:
    assignments: dict[tuple[int, int], PatchBox] = {}
    for patch in boxes:
        patch_center = np.array([patch.center_x, patch.center_y], dtype=np.float32)
        key = min(
            affine_centers.keys(),
            key=lambda item: float(np.linalg.norm(patch_center - affine_centers[item])),
        )
        previous = assignments.get(key)
        if previous is None:
            assignments[key] = patch
            continue
        prev_center = np.array([previous.center_x, previous.center_y], dtype=np.float32)
        if np.linalg.norm(patch_center - affine_centers[key]) < np.linalg.norm(prev_center - affine_centers[key]):
            assignments[key] = patch
    return assignments


def draw_boxes(image: np.ndarray, boxes: list[PatchBox], color: tuple[int, int, int], label_prefix: str = "") -> np.ndarray:
    canvas = image.copy()
    for index, patch in enumerate(boxes):
        cv2.rectangle(canvas, (patch.x, patch.y), (patch.x + patch.w, patch.y + patch.h), color, 2)
        text = f"{label_prefix}{index}"
        cv2.putText(canvas, text, (patch.x, max(12, patch.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return canvas


def collect_component_boxes(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, list[PatchBox]]:
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    boxes = filter_patch_boxes(mask if mask.ndim == 3 else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), stats, centroids)
    return component_count, stats, centroids, boxes


def draw_component_rectangles(image: np.ndarray, stats: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    for component_index in range(1, stats.shape[0]):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        w = int(stats[component_index, cv2.CC_STAT_WIDTH])
        h = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 1)
    return canvas


def build_mask_comparison_panel(image: np.ndarray, mask_variants: dict[str, np.ndarray]) -> np.ndarray:
    panels: list[np.ndarray] = []
    labels = {
        "direct": "05直连通域",
        "open": "05+轻Open",
        "open_close": "05+OpenClose",
    }
    for key in ("direct", "open", "open_close"):
        mask = mask_variants[key]
        preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        preview = cv2.resize(preview, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.putText(preview, labels[key], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        panels.append(preview)
    return np.hstack(panels)


def build_candidate_comparison_panel(image: np.ndarray, mask_variants: dict[str, np.ndarray]) -> np.ndarray:
    panels: list[np.ndarray] = []
    labels = {
        "direct": "direct",
        "open": "open",
        "open_close": "open_close",
    }
    for key in ("direct", "open", "open_close"):
        component_count, stats, _centroids, boxes = collect_component_boxes(mask_variants[key])
        canvas = draw_boxes(image, boxes, (0, 255, 255), label_prefix="c")
        cv2.putText(
            canvas,
            f"{labels[key]} comps={component_count - 1} cand={len(boxes)}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        panels.append(canvas)
    return np.hstack(panels)


def draw_grid_assignments(
    image: np.ndarray,
    inferred_grid: dict[tuple[int, int], np.ndarray],
    assignments: dict[tuple[int, int], PatchBox],
) -> np.ndarray:
    canvas = image.copy()
    for (row_index, col_index), patch in inferred_grid.items():
        assigned = assignments.get((row_index, col_index))
        color = (0, 255, 0) if assigned is not None else (0, 128, 255)
        thickness = 2 if assigned is not None else 1
        cv2.polylines(canvas, [np.round(patch).astype(np.int32)], True, color, thickness)
        label = f"{row_index},{col_index}"
        anchor = np.round(patch[0]).astype(np.int32)
        cv2.putText(canvas, label, (int(anchor[0]) + 3, int(anchor[1]) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if assigned is not None:
            cx = int(round(assigned.center_x))
            cy = int(round(assigned.center_y))
            cv2.circle(canvas, (cx, cy), 3, (255, 255, 0), -1)
    return canvas


def write_summary(
    output_dir: Path,
    boxes: list[PatchBox],
    row_centers: list[float],
    col_centers: list[float],
    assignments: dict[tuple[int, int], PatchBox],
    model: LatticeModel,
    selected_mask: str,
) -> None:
    payload = {
        "selected_mask": selected_mask,
        "candidate_count": len(boxes),
        "row_centers": [round(value, 2) for value in row_centers],
        "col_centers": [round(value, 2) for value in col_centers],
        "lattice_origin": [round(float(value), 2) for value in model.origin],
        "col_vec": [round(float(value), 2) for value in model.col_vec],
        "row_vec": [round(float(value), 2) for value in model.row_vec],
        "assigned_cells": sorted([f"{row},{col}" for row, col in assignments.keys()]),
        "missing_cells": sorted(
            [
                f"{row},{col}"
                for row in range(EXPECTED_ROWS)
                for col in range(EXPECTED_COLS)
                if (row, col) not in assignments
            ]
        ),
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
        else Path("d:/code/pycharm/strawberry/experiments/patch_detection/output/detect_patch_grid_from_best_variant") / input_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imdecode(np.fromfile(str(input_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {input_path}")

    normalized, edges, closed_edges = build_edges(image)
    selected_mask_name = "open"
    threshold_value, mask_variants = build_patch_mask_variants(image)
    otsu_mask = mask_variants["direct"]
    cleaned_mask = mask_variants[selected_mask_name]
    component_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, 8)
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

    components_vis = draw_component_rectangles(image, stats)

    candidate_vis = draw_boxes(image, boxes, (0, 255, 255), label_prefix="c")
    assignment_vis = draw_grid_assignments(image, inferred_grid, assignments)
    final_vis = image.copy()
    for row in range(EXPECTED_ROWS):
        for col in range(EXPECTED_COLS):
            patch = inferred_grid[(row, col)]
            assigned = assignments.get((row, col))
            color = (0, 255, 0) if assigned is not None else (0, 128, 255)
            cv2.polylines(final_vis, [np.round(patch).astype(np.int32)], True, color, 2)
            index = row * EXPECTED_COLS + col
            center = np.mean(patch, axis=0)
            cv2.putText(
                final_vis,
                str(index),
                (int(round(center[0])) - 8, int(round(center[1])) + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    row_col_vis = image.copy()
    origin = lattice_model.origin
    for row in range(EXPECTED_ROWS):
        start = origin + row * lattice_model.row_vec
        end = start + (EXPECTED_COLS - 1) * lattice_model.col_vec
        cv2.line(
            row_col_vis,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            (255, 0, 255),
            1,
        )
    for col in range(EXPECTED_COLS):
        start = origin + col * lattice_model.col_vec
        end = start + (EXPECTED_ROWS - 1) * lattice_model.row_vec
        cv2.line(
            row_col_vis,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            (255, 255, 0),
            1,
        )

    save_image(output_dir / "01_original.png", image)
    save_image(output_dir / "02_gray_normalized.png", normalized)
    save_image(output_dir / "03_edges.png", edges)
    save_image(output_dir / "04_closed_edges.png", closed_edges)
    save_image(output_dir / "05_otsu_mask.png", otsu_mask)
    save_image(output_dir / "06_selected_patch_mask.png", cleaned_mask)
    save_image(output_dir / "05_mask_variants_comparison.png", build_mask_comparison_panel(image, mask_variants))
    save_image(output_dir / "06_candidate_variants_comparison.png", build_candidate_comparison_panel(image, mask_variants))
    save_image(output_dir / "07_connected_components.png", components_vis)
    save_image(output_dir / "08_candidate_boxes.png", candidate_vis)
    save_image(output_dir / "09_row_col_clusters.png", row_col_vis)
    save_image(output_dir / "10_grid_assignments.png", assignment_vis)
    save_image(output_dir / "11_final_inferred_grid.png", final_vis)
    write_summary(output_dir, boxes, row_centers, col_centers, assignments, lattice_model, selected_mask_name)

    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Selected mask branch: {selected_mask_name}")
    print(f"Otsu threshold: {threshold_value:.1f}")
    print(f"Connected components: {component_count - 1}")
    print(f"Candidates after filtering: {len(boxes)}")
    print(f"Row centers: {[round(value, 2) for value in row_centers]}")
    print(f"Col centers: {[round(value, 2) for value in col_centers]}")
    print(f"Lattice origin: {[round(float(value), 2) for value in lattice_model.origin]}")
    print(f"Column vector: {[round(float(value), 2) for value in lattice_model.col_vec]}")
    print(f"Row vector: {[round(float(value), 2) for value in lattice_model.row_vec]}")
    print(f"Assigned cells: {len(assignments)}/{EXPECTED_ROWS * EXPECTED_COLS}")


if __name__ == "__main__":
    main()
