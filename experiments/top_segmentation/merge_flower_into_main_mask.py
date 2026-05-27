from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.segmentation import (
    _filter_top_view_components,
    _prepare_segmentation,
    _remove_top_attached_pot_band,
)
from utils.debug_artifacts import create_debug_montage, create_mask_overlay, create_masked_color_image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment: merge white/yellow flower-organ candidates into TOP main mask stage "
            "before top-specific post-rules."
        ),
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to one TOP RGB image. Leave empty to auto-pick a default TOP image.",
    )
    parser.add_argument(
        "--image",
        dest="image_option",
        default=None,
        help="Path to one TOP RGB image. Overrides the positional image argument.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to "
            "experiments/top_segmentation/output/merge_flower_into_main_mask/<image-stem>."
        ),
    )
    parser.add_argument("--min-white-area", type=int, default=20, help="Minimum white-candidate component area.")
    parser.add_argument("--min-yellow-area", type=int, default=8, help="Minimum yellow-candidate component area.")
    parser.add_argument(
        "--min-main-area-ratio",
        type=float,
        default=0.0005,
        help="Minimum area ratio for the main mask components (same meaning as TOP segmentation).",
    )

    args = parser.parse_args(argv)
    selected_image = args.image_option or args.image
    args.used_default_image = False
    if selected_image is None:
        default_image = resolve_default_top_image()
        if default_image is None:
            parser.error("No image provided and no TOP image found in project data directory.")
        selected_image = str(default_image)
        args.used_default_image = True
    args.image = selected_image
    return args


def resolve_default_top_image() -> Path | None:
    preferred = PROJECT_ROOT / "data" / "111AB_TOP.png"
    if preferred.exists():
        return preferred.resolve()

    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        return None
    candidates = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and "_TOP" in path.stem.upper()
    )
    if not candidates:
        return None
    return candidates[0].resolve()


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return image


def filter_small_components_by_area(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return mask.copy()
    filtered = np.zeros_like(mask)
    for component_index in range(1, component_count):
        if int(stats[component_index, cv2.CC_STAT_AREA]) >= min_area:
            filtered[labels == component_index] = 255
    return filtered


def build_flower_color_union(
    image: np.ndarray,
    *,
    min_white_area: int,
    min_yellow_area: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bgr16 = image.astype(np.int16)
    channel_min = np.min(bgr16, axis=2)
    channel_max = np.max(bgr16, axis=2)
    channel_spread = channel_max - channel_min
    neutral_a = np.abs(lab[:, :, 1].astype(np.int16) - 128)
    neutral_b = np.abs(lab[:, :, 2].astype(np.int16) - 128)

    white_raw = (
        (hsv[:, :, 1] <= 95)
        & (hsv[:, :, 2] >= 150)
        & (lab[:, :, 0] >= 165)
        & (channel_min >= 140)
        & (channel_spread <= 85)
        & (neutral_a <= 24)
        & (neutral_b <= 30)
    ).astype(np.uint8) * 255

    blue = bgr16[:, :, 0]
    green = bgr16[:, :, 1]
    red = bgr16[:, :, 2]
    yellow_raw = (
        (hsv[:, :, 0] >= 10)
        & (hsv[:, :, 0] <= 48)
        & (hsv[:, :, 1] >= 35)
        & (hsv[:, :, 2] >= 90)
        & (lab[:, :, 0] >= 105)
        & (lab[:, :, 2] >= 138)
        & (green >= blue + 3)
        & (red >= blue + 3)
    ).astype(np.uint8) * 255

    white_cleaned = filter_small_components_by_area(
        cv2.morphologyEx(white_raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))),
        min_area=min_white_area,
    )
    yellow_cleaned = filter_small_components_by_area(
        cv2.morphologyEx(yellow_raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))),
        min_area=min_yellow_area,
    )
    flower_union = cv2.bitwise_or(white_cleaned, yellow_cleaned)
    flower_union = cv2.morphologyEx(
        flower_union,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    stats = {
        "white_pixels_raw": int(cv2.countNonZero(white_raw)),
        "white_pixels_kept": int(cv2.countNonZero(white_cleaned)),
        "yellow_pixels_raw": int(cv2.countNonZero(yellow_raw)),
        "yellow_pixels_kept": int(cv2.countNonZero(yellow_cleaned)),
        "flower_union_pixels": int(cv2.countNonZero(flower_union)),
    }
    debug = {
        "white_raw": white_raw,
        "white_cleaned": white_cleaned,
        "yellow_raw": yellow_raw,
        "yellow_cleaned": yellow_cleaned,
        "flower_union": flower_union,
    }
    return flower_union, debug, stats


def build_baseline_mask(image: np.ndarray, *, min_area_ratio: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    prepared = _prepare_segmentation(image, min_component_area_ratio=min_area_ratio)
    after_card = _filter_top_view_components(prepared["filtered_mask"])
    final_mask, top_band_candidate, removed_top_band = _remove_top_attached_pot_band(after_card)
    debug = {
        "base_filtered_mask": prepared["filtered_mask"],
        "after_card_removal": after_card,
        "top_band_candidate": top_band_candidate,
        "removed_top_pot_band": removed_top_band,
        "final_mask": final_mask,
    }
    return final_mask, debug


def build_main_stage_merged_mask(
    image: np.ndarray,
    *,
    min_area_ratio: float,
    flower_union: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    prepared = _prepare_segmentation(image, min_component_area_ratio=min_area_ratio)
    merged_before_top_rules = cv2.bitwise_or(prepared["filtered_mask"], flower_union)
    after_card = _filter_top_view_components(merged_before_top_rules)
    final_mask, top_band_candidate, removed_top_band = _remove_top_attached_pot_band(after_card)
    debug = {
        "merged_before_top_rules": merged_before_top_rules,
        "after_card_removal": after_card,
        "top_band_candidate": top_band_candidate,
        "removed_top_pot_band": removed_top_band,
        "final_mask": final_mask,
    }
    return final_mask, debug


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def write_readme(output_dir: Path, image_path: Path, args: argparse.Namespace) -> None:
    text = (
        "merge_flower_into_main_mask 输出说明\n\n"
        f"输入图像: {image_path}\n\n"
        "目标: 在主掩膜阶段（Top-specific规则前）直接并入白/黄花器官候选，"
        "并与基线流程做并排对比。\n\n"
        "参数:\n"
        f"- min_white_area={args.min_white_area}\n"
        f"- min_yellow_area={args.min_yellow_area}\n"
        f"- min_main_area_ratio={args.min_main_area_ratio}\n\n"
        "关键输出:\n"
        "- 02_baseline_final_mask.png: 现有基线最终掩膜\n"
        "- 12_merged_main_final_mask.png: 主掩膜阶段并入花候选后的最终掩膜\n"
        "- 14_delta_added_by_main_merge.png: 相比基线新增区域\n"
        "- 16_delta_removed_vs_baseline.png: 相比基线减少区域\n"
        "- 90_montage.png: 总览拼图\n"
        "- summary.json: 像素统计\n"
    )
    (output_dir / "说明.txt").write_text(text, encoding="utf-8")


def write_summary(output_dir: Path, image_path: Path, stats: dict[str, int | float]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps({"image": str(image_path), **stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (SCRIPT_ROOT / "output" / "merge_flower_into_main_mask" / image_path.stem).resolve()
    )
    if getattr(args, "used_default_image", False):
        print(f"[INFO] No image argument provided, using default: {image_path}")

    image = load_image(image_path)
    flower_union, flower_debug, flower_stats = build_flower_color_union(
        image,
        min_white_area=args.min_white_area,
        min_yellow_area=args.min_yellow_area,
    )
    baseline_mask, baseline_debug = build_baseline_mask(image, min_area_ratio=args.min_main_area_ratio)
    merged_mask, merged_debug = build_main_stage_merged_mask(
        image,
        min_area_ratio=args.min_main_area_ratio,
        flower_union=flower_union,
    )

    delta_added = cv2.bitwise_and(merged_mask, cv2.bitwise_not(baseline_mask))
    delta_removed = cv2.bitwise_and(baseline_mask, cv2.bitwise_not(merged_mask))

    save_image(output_dir / "01_original.png", image)
    save_image(output_dir / "02_baseline_final_mask.png", baseline_mask)
    save_image(output_dir / "03_baseline_overlay.png", create_mask_overlay(image, baseline_mask, alpha=0.45))
    save_image(output_dir / "04_baseline_masked_region.png", create_masked_color_image(image, baseline_mask))

    save_image(output_dir / "05_white_raw.png", flower_debug["white_raw"])
    save_image(output_dir / "06_white_cleaned.png", flower_debug["white_cleaned"])
    save_image(output_dir / "07_yellow_raw.png", flower_debug["yellow_raw"])
    save_image(output_dir / "08_yellow_cleaned.png", flower_debug["yellow_cleaned"])
    save_image(output_dir / "09_flower_union.png", flower_union)

    save_image(output_dir / "10_main_merge_before_top_rules.png", merged_debug["merged_before_top_rules"])
    save_image(output_dir / "11_main_merge_after_card_removal.png", merged_debug["after_card_removal"])
    save_image(output_dir / "12_merged_main_final_mask.png", merged_mask)
    save_image(output_dir / "13_merged_main_overlay.png", create_mask_overlay(image, merged_mask, alpha=0.45))
    save_image(output_dir / "14_delta_added_by_main_merge.png", delta_added)
    save_image(output_dir / "15_delta_added_overlay.png", create_mask_overlay(image, delta_added, alpha=0.55))
    save_image(output_dir / "16_delta_removed_vs_baseline.png", delta_removed)
    save_image(output_dir / "17_delta_removed_overlay.png", create_mask_overlay(image, delta_removed, alpha=0.55))

    # Keep the key top-rule debug maps for side-by-side analysis.
    save_image(output_dir / "20_baseline_top_band_candidate.png", baseline_debug["top_band_candidate"])
    save_image(output_dir / "21_baseline_removed_top_pot_band.png", baseline_debug["removed_top_pot_band"])
    save_image(output_dir / "22_merged_top_band_candidate.png", merged_debug["top_band_candidate"])
    save_image(output_dir / "23_merged_removed_top_pot_band.png", merged_debug["removed_top_pot_band"])

    montage = create_debug_montage(
        [
            image,
            create_mask_overlay(image, baseline_mask, alpha=0.45),
            create_mask_overlay(image, merged_mask, alpha=0.45),
            create_mask_overlay(image, delta_added, alpha=0.55),
            flower_union,
            baseline_debug["top_band_candidate"],
            merged_debug["top_band_candidate"],
            create_masked_color_image(image, merged_mask),
        ],
        columns=4,
        tile_size=(420, 300),
    )
    save_image(output_dir / "90_montage.png", montage)

    stats: dict[str, int | float] = {
        **flower_stats,
        "baseline_pixels": int(cv2.countNonZero(baseline_mask)),
        "merged_main_pixels": int(cv2.countNonZero(merged_mask)),
        "delta_added_pixels": int(cv2.countNonZero(delta_added)),
        "delta_removed_pixels": int(cv2.countNonZero(delta_removed)),
    }
    write_summary(output_dir, image_path, stats)
    write_readme(output_dir, image_path, args)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
