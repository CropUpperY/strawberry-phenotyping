from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.segmentation import _filter_top_view_components, _prepare_segmentation, _remove_top_attached_pot_band
from utils.debug_artifacts import create_debug_montage, create_mask_overlay, create_masked_color_image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug TOP mask flower preserving experiment.")
    parser.add_argument("image", nargs="?", default=None)
    parser.add_argument("--image", dest="image_option", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--folder-mode", choices=("stem", "sample"), default="stem")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--max-distance", type=float, default=180.0)
    parser.add_argument("--min-white-area", type=int, default=20)
    parser.add_argument("--min-yellow-area", type=int, default=8)
    parser.add_argument("--min-rescue-area", type=int, default=18)
    parser.add_argument("--disable-hard-negative", action="store_true")
    parser.add_argument("--disable-pot-ring-hard-negative", action="store_true")
    parser.add_argument("--hard-negative-dilate", type=int, default=9)
    parser.add_argument("--pot-ring-width", type=int, default=14)
    parser.add_argument("--pot-ring-min-score", type=float, default=0.02)
    parser.add_argument("--min-flower-ratio", type=float, default=0.22)
    parser.add_argument("--min-green-ratio", type=float, default=0.30)
    parser.add_argument("--max-flower-distance", type=float, default=180.0)
    parser.add_argument("--max-leaf-distance", type=float, default=70.0)
    parser.add_argument("--min-context-green-ratio", type=float, default=0.06)
    parser.add_argument("--max-flower-component-area", type=int, default=6000)
    parser.add_argument("--max-strip-aspect", type=float, default=3.2)
    parser.add_argument("--crop-margin", type=int, default=20)
    parser.add_argument("--max-component-crops", type=int, default=40)
    args = parser.parse_args(argv)
    selected = args.image_option or args.image
    args.used_default_image = False
    if selected is None:
        default = resolve_default_top_image()
        if default is None:
            parser.error("No TOP image found.")
        selected = str(default)
        args.used_default_image = True
    args.image = selected
    return args


def resolve_default_top_image() -> Path | None:
    preferred = PROJECT_ROOT / "data" / "111AB_TOP.png"
    if preferred.exists():
        return preferred.resolve()
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        return None
    candidates = sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and "_TOP" in p.stem.upper()
    )
    return candidates[0].resolve() if candidates else None


def infer_sample_id(stem: str) -> str:
    up = stem.upper()
    for suffix in ("_TOP", "-TOP", "_FRONT-1", "_FRONT-2", "_FRONT1", "_FRONT2"):
        if up.endswith(suffix):
            return stem[: -len(suffix)] or stem
    return stem


def resolve_output_dir(args: argparse.Namespace, image_path: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    if args.folder_mode == "sample":
        name = (args.sample_id or infer_sample_id(image_path.stem)).strip() or image_path.stem
    else:
        name = image_path.stem
    root = Path(args.output_root).expanduser().resolve() if args.output_root else (SCRIPT_ROOT / "output" / "flower_preserving_top_mask").resolve()
    return root / name


def load_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return image


def filter_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask.copy()
    out = np.zeros_like(mask)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == i] = 255
    return out


def build_current_top_mask(image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    prepared = _prepare_segmentation(image, min_component_area_ratio=0.0005)
    after_card = _filter_top_view_components(prepared["filtered_mask"])
    after_pot, top_band, removed_top = _remove_top_attached_pot_band(after_card)
    return after_pot, {
        "removed_top_pot_band": removed_top,
        "top_band_candidate": top_band,
        "base_mask_after_card_removal": after_card,
    }


def build_flower_union(image: np.ndarray, min_white_area: int, min_yellow_area: int) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bgr16 = image.astype(np.int16)
    ch_min = np.min(bgr16, axis=2)
    ch_max = np.max(bgr16, axis=2)
    spread = ch_max - ch_min
    neutral_a = np.abs(lab[:, :, 1].astype(np.int16) - 128)
    neutral_b = np.abs(lab[:, :, 2].astype(np.int16) - 128)
    white = (
        (hsv[:, :, 1] <= 95) & (hsv[:, :, 2] >= 150) & (lab[:, :, 0] >= 165) &
        (ch_min >= 140) & (spread <= 85) & (neutral_a <= 24) & (neutral_b <= 30)
    ).astype(np.uint8) * 255
    blue, green, red = bgr16[:, :, 0], bgr16[:, :, 1], bgr16[:, :, 2]
    yellow = (
        (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 48) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 90) &
        (lab[:, :, 0] >= 105) & (lab[:, :, 2] >= 138) & (green >= blue + 3) & (red >= blue + 3)
    ).astype(np.uint8) * 255
    white_c = filter_small(cv2.morphologyEx(white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))), min_white_area)
    yellow_c = filter_small(cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))), min_yellow_area)
    union = cv2.bitwise_or(white_c, yellow_c)
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return {"white": white_c, "yellow": yellow_c, "union": union}, {
        "white_pixels": int(cv2.countNonZero(white_c)),
        "yellow_pixels": int(cv2.countNonZero(yellow_c)),
        "union_pixels": int(cv2.countNonZero(union)),
    }


def green_ref(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bgr = image.astype(np.float32)
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    exg = 2.0 * g - r - b
    m = (
        ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 20))
        | ((g > 30.0) & (g > r * 1.03) & (g > b * 1.05) & (exg >= 20.0))
    )
    return m.astype(np.uint8) * 255


def detect_pot_ring(image: np.ndarray, base_mask: np.ndarray, green_mask: np.ndarray, ring_width: int, min_score: float) -> tuple[np.ndarray, dict[str, float | int]]:
    h, w = base_mask.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 60, 140)
    min_dim = min(h, w)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.35, minDist=max(40, int(min_dim * 0.15)),
        param1=140, param2=36, minRadius=max(30, int(min_dim * 0.08)), maxRadius=max(31, int(min_dim * 0.48))
    )
    if circles is None:
        return np.zeros_like(base_mask), {"pot_ring_detected": 0, "pot_ring_best_score": 0.0}

    m = cv2.moments((base_mask > 0).astype(np.uint8))
    cx0 = float(m["m10"] / m["m00"]) if m["m00"] > 0 else w / 2.0
    cy0 = float(m["m01"] / m["m00"]) if m["m00"] > 0 else h / 2.0
    green = green_mask > 0
    canopy = base_mask > 0

    best_score = 0.0
    best = np.zeros_like(base_mask)
    best_circle = (0, 0, 0)
    for cx, cy, r in np.round(circles[0]).astype(np.int32)[:24]:
        outer, inner = max(1, int(r + ring_width)), max(1, int(r - ring_width))
        ring = np.zeros_like(base_mask)
        cv2.circle(ring, (int(cx), int(cy)), outer, 255, -1)
        hole = np.zeros_like(base_mask)
        cv2.circle(hole, (int(cx), int(cy)), inner, 255, -1)
        ring = cv2.bitwise_and(ring, cv2.bitwise_not(hole))
        rb = ring > 0
        if not np.any(rb):
            continue
        edge_cov = float(np.mean((edges > 0)[rb]))
        non_green = float(np.mean((~green)[rb]))
        disk = np.zeros_like(base_mask)
        cv2.circle(disk, (int(cx), int(cy)), max(1, int(r)), 255, -1)
        db = disk > 0
        overlap = float(np.mean(canopy[db])) if np.any(db) else 0.0
        center_dist = ((float(cx) - cx0) ** 2 + (float(cy) - cy0) ** 2) ** 0.5
        center_score = max(0.0, 1.0 - center_dist / (0.45 * float(min_dim)))
        score = edge_cov * (0.35 + 0.65 * non_green) * (0.25 + 0.75 * center_score) * (0.25 + 0.75 * min(1.0, overlap * 3.0))
        if score > best_score:
            best_score = score
            best = ring
            best_circle = (int(cx), int(cy), int(r))

    if best_score < float(min_score):
        return np.zeros_like(base_mask), {"pot_ring_detected": 0, "pot_ring_best_score": float(best_score)}
    return best, {
        "pot_ring_detected": 1,
        "pot_ring_best_score": float(best_score),
        "pot_ring_center_x": int(best_circle[0]),
        "pot_ring_center_y": int(best_circle[1]),
        "pot_ring_radius": int(best_circle[2]),
    }


def geom_metrics(local_mask: np.ndarray, area: int) -> tuple[float, float, float, float]:
    ys, xs = np.where(local_mask > 0)
    if xs.size == 0:
        return 1.0, 1.0, 1.0, 1.0
    pts = np.column_stack([xs, ys]).astype(np.int32)
    _, _, w, h = cv2.boundingRect(pts)
    aspect = float(max(w, h)) / float(max(1, min(w, h)))
    fill = float(area) / float(max(1, w * h))
    contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return aspect, fill, 1.0, 1.0
    c = max(contours, key=cv2.contourArea)
    peri = float(cv2.arcLength(c, True))
    circularity = (4.0 * np.pi * float(area) / (peri * peri)) if peri > 0 else 0.0
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area) / hull_area if hull_area > 0 else 0.0
    return aspect, fill, solidity, float(circularity)


def filter_components(
    image: np.ndarray,
    candidate: np.ndarray,
    base_mask: np.ndarray,
    flower_union: np.ndarray,
    dist_map: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, int | float], list[dict[str, int | float | str]]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    keep = np.zeros_like(candidate)
    removed = np.zeros_like(candidate)
    removed_geom = np.zeros_like(candidate)
    keep_flower = np.zeros_like(candidate)
    keep_green = np.zeros_like(candidate)
    if n <= 1:
        return {"keep": keep, "removed": removed, "removed_geom": removed_geom, "keep_flower": keep_flower, "keep_green": keep_green}, {"candidate_components": 0, "kept_components": 0, "removed_components": 0}, []

    green = green_ref(image) > 0
    flower = flower_union > 0
    base = base_mask > 0
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kept_count = 0
    removed_count = 0
    rows: list[dict[str, int | float | str]] = []

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        comp = labels == i
        min_dist = float(np.min(dist_map[comp])) if np.any(comp) else float("inf")
        green_ratio = float(np.mean(green[comp]))
        flower_ratio = float(np.mean(flower[comp]))
        overlaps_base = bool(np.any(comp & base))

        pad = 5
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(candidate.shape[1], x + w + pad), min(candidate.shape[0], y + h + pad)
        local = (labels[y0:y1, x0:x1] == i).astype(np.uint8) * 255
        ring = cv2.bitwise_and(cv2.dilate(local, ring_kernel, 1), cv2.bitwise_not(local))
        rb = ring > 0
        context_green = float(np.mean(green[y0:y1, x0:x1][rb])) if np.any(rb) else 0.0
        aspect, fill, solidity, circularity = geom_metrics(local, area)

        reject_strip = aspect >= float(args.max_strip_aspect) and circularity <= 0.33
        reject_arc = fill <= 0.30 and solidity <= 0.70 and circularity <= 0.20
        reject_huge = area > int(args.max_flower_component_area)
        geom_reject = bool(reject_strip or reject_arc or reject_huge)

        keep_by_flower = flower_ratio >= float(args.min_flower_ratio) and min_dist <= float(args.max_flower_distance) and context_green >= float(args.min_context_green_ratio) and not geom_reject
        keep_by_green = green_ratio >= float(args.min_green_ratio) and flower_ratio <= 0.10 and min_dist <= float(args.max_leaf_distance) and not reject_strip
        keep_comp = keep_by_flower or keep_by_green
        reason = "kept" if keep_comp else ("removed_geometry" if geom_reject else ("removed_overlap_only" if overlaps_base else "removed_non_green"))

        rows.append({"component_id": i, "area": area, "aspect": aspect, "fill": fill, "solidity": solidity, "circularity": circularity, "flower_ratio": flower_ratio, "green_ratio": green_ratio, "context_green": context_green, "min_dist": min_dist, "keep": int(keep_comp), "reason": reason})

        if keep_comp:
            keep[comp] = 255; kept_count += 1
            if keep_by_flower:
                keep_flower[comp] = 255
            if keep_by_green:
                keep_green[comp] = 255
        else:
            removed[comp] = 255; removed_count += 1
            if geom_reject:
                removed_geom[comp] = 255

    return {
        "keep": keep, "removed": removed, "removed_geom": removed_geom, "keep_flower": keep_flower, "keep_green": keep_green
    }, {
        "candidate_components": int(n - 1),
        "kept_components": int(kept_count),
        "removed_components": int(removed_count),
        "removed_geom_pixels": int(cv2.countNonZero(removed_geom)),
    }, rows


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def write_rows_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    if not rows:
        path.write_text("component_id,reason\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_component_crops(image: np.ndarray, mask: np.ndarray, out_dir: Path, margin: int, max_crops: int) -> int:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    ranked = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > 0:
            ranked.append((area, i))
    ranked.sort(reverse=True)
    h, w = mask.shape
    saved = 0
    for _a, i in ranked[:max(0, max_crops)]:
        x = max(0, int(stats[i, cv2.CC_STAT_LEFT]) - margin)
        y = max(0, int(stats[i, cv2.CC_STAT_TOP]) - margin)
        ww = int(stats[i, cv2.CC_STAT_WIDTH]) + margin * 2
        hh = int(stats[i, cv2.CC_STAT_HEIGHT]) + margin * 2
        x1 = min(w, x + ww); y1 = min(h, y + hh)
        cm = np.zeros_like(mask); cm[labels == i] = 255
        crop = image[y:y1, x:x1]; crop_m = cm[y:y1, x:x1]
        save_image(out_dir / f"component_{saved+1:02d}_original.png", crop)
        save_image(out_dir / f"component_{saved+1:02d}_overlay.png", create_mask_overlay(crop, crop_m, alpha=0.45))
        saved += 1
    return saved


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_dir = resolve_output_dir(args, image_path)
    image = load_image(image_path)

    base_mask, base_debug = build_current_top_mask(image)
    flower_masks, flower_stats = build_flower_union(image, args.min_white_area, args.min_yellow_area)
    green_mask = green_ref(image)

    pot_ring = np.zeros_like(base_mask)
    pot_stats: dict[str, int | float] = {"pot_ring_detected": 0, "pot_ring_best_score": 0.0}
    if not args.disable_pot_ring_hard_negative:
        pot_ring, pot_stats = detect_pot_ring(image, base_mask, green_mask, args.pot_ring_width, args.pot_ring_min_score)

    hard_negative = np.zeros_like(base_mask)
    if not args.disable_hard_negative:
        hard_negative = cv2.bitwise_or(hard_negative, base_debug["removed_top_pot_band"])
        hard_negative = cv2.bitwise_or(hard_negative, pot_ring)

    base_binary = (base_mask > 0).astype(np.uint8)
    dist_map = cv2.distanceTransform((1 - base_binary).astype(np.uint8), cv2.DIST_L2, 5)
    near_canopy = (dist_map <= float(args.max_distance)).astype(np.uint8) * 255
    raw = cv2.bitwise_and(flower_masks["union"], near_canopy)
    raw = filter_small(raw, args.min_rescue_area)

    dilate_k = int(max(1, args.hard_negative_dilate)); dilate_k += 1 - (dilate_k % 2)
    exp_hn = cv2.dilate(hard_negative, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)), 1) if cv2.countNonZero(hard_negative) > 0 else np.zeros_like(hard_negative)
    removed_hn = cv2.bitwise_and(raw, exp_hn)
    after_hn = cv2.bitwise_and(raw, cv2.bitwise_not(exp_hn))

    filtered, filter_stats, rows = filter_components(image, after_hn, base_mask, flower_masks["union"], dist_map, args)
    rescued_only = cv2.bitwise_and(filtered["keep"], cv2.bitwise_not(base_mask))
    augmented = cv2.bitwise_or(base_mask, filtered["keep"])

    save_image(output_dir / "01_original.png", image)
    save_image(output_dir / "02_current_top_mask.png", base_mask)
    save_image(output_dir / "03_current_top_overlay.png", create_mask_overlay(image, base_mask, alpha=0.45))
    save_image(output_dir / "04_white_mask.png", flower_masks["white"])
    save_image(output_dir / "05_yellow_mask.png", flower_masks["yellow"])
    save_image(output_dir / "06_flower_union.png", flower_masks["union"])
    save_image(output_dir / "07_near_canopy_zone.png", near_canopy)
    save_image(output_dir / "08_flower_near_canopy_raw.png", raw)
    save_image(output_dir / "09_removed_by_hard_negative.png", removed_hn)
    save_image(output_dir / "10_after_hard_negative.png", after_hn)
    save_image(output_dir / "11_removed_non_green.png", filtered["removed"])
    save_image(output_dir / "12_removed_by_geometry.png", filtered["removed_geom"])
    save_image(output_dir / "13_kept_rescue_mask.png", filtered["keep"])
    save_image(output_dir / "14_rescued_only.png", rescued_only)
    save_image(output_dir / "15_augmented_mask.png", augmented)
    save_image(output_dir / "16_augmented_overlay.png", create_mask_overlay(image, augmented, alpha=0.45))
    save_image(output_dir / "17_pot_ring_mask.png", pot_ring)
    save_image(output_dir / "18_hard_negative_input.png", hard_negative)
    save_image(output_dir / "19_hard_negative_expanded.png", exp_hn)
    save_image(output_dir / "20_kept_by_flower_rule.png", filtered["keep_flower"])
    save_image(output_dir / "21_kept_by_green_rule.png", filtered["keep_green"])

    montage = create_debug_montage(
        [image, create_mask_overlay(image, base_mask, alpha=0.45), flower_masks["union"], removed_hn, filtered["removed_geom"], filtered["removed"], rescued_only, create_mask_overlay(image, augmented, alpha=0.45)],
        columns=4,
        tile_size=(420, 300),
    )
    save_image(output_dir / "90_montage.png", montage)

    crop_count = save_component_crops(image, rescued_only, output_dir / "rescued_components", args.crop_margin, args.max_component_crops)
    write_rows_csv(output_dir / "component_features.csv", rows)
    (output_dir / "说明.txt").write_text("看 90_montage.png、17_pot_ring_mask.png、12_removed_by_geometry.png、component_features.csv。", encoding="utf-8")

    summary = {
        "image": str(image_path),
        **flower_stats,
        **filter_stats,
        **pot_stats,
        "raw_pixels": int(cv2.countNonZero(raw)),
        "removed_hard_negative_pixels": int(cv2.countNonZero(removed_hn)),
        "rescued_pixels": int(cv2.countNonZero(rescued_only)),
        "augmented_pixels": int(cv2.countNonZero(augmented)),
        "component_crops": int(crop_count),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved flower-preserving TOP-mask debug outputs to: {output_dir}")
    print(f"Rescued components: {crop_count}")


if __name__ == "__main__":
    main()
