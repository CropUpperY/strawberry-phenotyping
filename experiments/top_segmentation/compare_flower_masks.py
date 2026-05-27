from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比花器官掩膜：纯轮廓法 vs 纯颜色法（单张 TOP 图像）。",
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="一张 TOP 图像路径。为空时自动选择默认 TOP 图像。",
    )
    parser.add_argument(
        "--image",
        dest="image_option",
        default=None,
        help="一张 TOP 图像路径。会覆盖位置参数 image。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="直接指定输出目录（优先级最高）。",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="输出根目录；默认写到 experiments/top_segmentation/output/flower_mask_compare/<name>。",
    )
    parser.add_argument(
        "--folder-mode",
        choices=("stem", "sample"),
        default="stem",
        help="目录命名模式：stem=按图片名，sample=按样本编号。",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="folder-mode=sample 时可显式指定样本编号。",
    )
    args = parser.parse_args(argv)

    selected_image = args.image_option or args.image
    args.used_default_image = False
    if selected_image is None:
        default_image = resolve_default_top_image()
        if default_image is None:
            parser.error("未提供图像路径，且未在项目 data 目录中找到可用 TOP 图像。")
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


def infer_sample_id_from_stem(stem: str) -> str:
    upper = stem.upper()
    for suffix in ("_TOP", "-TOP", "_FRONT-1", "_FRONT-2", "_FRONT1", "_FRONT2"):
        if upper.endswith(suffix):
            return stem[: -len(suffix)] or stem
    return stem


def resolve_output_dir(args: argparse.Namespace, image_path: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()

    if args.folder_mode == "sample":
        folder_name = (args.sample_id or infer_sample_id_from_stem(image_path.stem)).strip() or image_path.stem
    else:
        folder_name = image_path.stem

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (SCRIPT_ROOT / "output" / "flower_mask_compare").resolve()
    )
    return output_root / folder_name


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return image


def build_contour_only_mask(image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dilated = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 24 or area > gray.shape[0] * gray.shape[1] * 0.08:
            continue
        cv2.drawContours(mask, [contour], -1, 255, -1)

    debug = {
        "gray": gray,
        "clahe": clahe,
        "edges": edges,
        "closed_edges": closed,
        "dilated_edges": dilated,
    }
    return mask, debug


def build_color_only_mask(image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bgr = image.astype(np.int16)
    channel_min = np.min(bgr, axis=2)
    channel_max = np.max(bgr, axis=2)
    channel_spread = channel_max - channel_min
    white_mask = (
        (hsv[:, :, 1] <= 60)
        & (hsv[:, :, 2] >= 185)
        & (lab[:, :, 0] >= 188)
        & (channel_min >= 175)
        & (channel_spread <= 30)
        & (np.abs(lab[:, :, 1].astype(np.int16) - 128) <= 12)
        & (np.abs(lab[:, :, 2].astype(np.int16) - 128) <= 16)
    ).astype(np.uint8) * 255

    b_channel = image[:, :, 0].astype(np.int16)
    g_channel = image[:, :, 1].astype(np.int16)
    r_channel = image[:, :, 2].astype(np.int16)
    yellow_mask = (
        (hsv[:, :, 0] >= 12)
        & (hsv[:, :, 0] <= 42)
        & (hsv[:, :, 1] >= 40)
        & (hsv[:, :, 2] >= 110)
        & (lab[:, :, 0] >= 120)
        & (lab[:, :, 2] >= 145)
        & (g_channel >= b_channel + 5)
        & (r_channel >= b_channel + 5)
    ).astype(np.uint8) * 255

    combined = cv2.bitwise_or(white_mask, yellow_mask)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    debug = {
        "white_mask": white_mask,
        "yellow_mask": yellow_mask,
    }
    return combined, debug


def build_overlay(image: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int]) -> np.ndarray:
    overlay = image.copy()
    colored = image.copy()
    colored[mask > 0] = color
    return cv2.addWeighted(colored, 0.45, overlay, 0.55, 0)


def build_montage(images: list[np.ndarray]) -> np.ndarray:
    resized: list[np.ndarray] = []
    for image in images:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        resized.append(cv2.resize(image, (420, 320), interpolation=cv2.INTER_AREA))
    top_row = np.hstack(resized[:2])
    bottom_row = np.hstack(resized[2:4])
    return np.vstack([top_row, bottom_row])


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if path.suffix else path.with_suffix(".png")
    suffix = target.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Failed to encode image: {target}")
    target.write_bytes(encoded.tobytes())


def write_output_readme(output_dir: Path, image_path: Path) -> None:
    content = (
        "flower-mask-compare 输出说明\n\n"
        f"输入图像: {image_path}\n\n"
        "本次输出用于对比两种花器官候选提取方法:\n"
        "1. 纯轮廓法（不使用颜色阈值）\n"
        "2. 纯颜色法（不使用结构轮廓）\n\n"
        "文件说明:\n"
        "- 01_original.png: 原始图像\n"
        "- 02_contour_only_mask.png: 纯轮廓掩膜\n"
        "- 03_contour_only_overlay.png: 纯轮廓 overlay\n"
        "- 04_color_only_mask.png: 纯颜色掩膜\n"
        "- 05_color_only_overlay.png: 纯颜色 overlay\n"
        "- 06_union_mask.png: 两者并集\n"
        "- 07_montage.png: 总览拼图\n"
        "- 20_contour_*.png: 轮廓法中间步骤\n"
        "- 30_color_*.png: 颜色法中间步骤\n\n"
        "建议先看 07_montage.png，再回看中间图定位问题。"
    )
    (output_dir / "说明.txt").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_dir = resolve_output_dir(args, image_path)
    if getattr(args, "used_default_image", False):
        print(f"[INFO] 未提供图像参数，默认使用: {image_path}")

    image = load_image(image_path)
    contour_mask, contour_debug = build_contour_only_mask(image)
    color_mask, color_debug = build_color_only_mask(image)
    contour_overlay = build_overlay(image, contour_mask, color=(255, 200, 0))
    color_overlay = build_overlay(image, color_mask, color=(0, 255, 255))
    montage = build_montage([image, contour_overlay, color_overlay, cv2.bitwise_or(contour_mask, color_mask)])

    save_image(output_dir / "01_original.png", image)
    save_image(output_dir / "02_contour_only_mask.png", contour_mask)
    save_image(output_dir / "03_contour_only_overlay.png", contour_overlay)
    save_image(output_dir / "04_color_only_mask.png", color_mask)
    save_image(output_dir / "05_color_only_overlay.png", color_overlay)
    save_image(output_dir / "06_union_mask.png", cv2.bitwise_or(contour_mask, color_mask))
    save_image(output_dir / "07_montage.png", montage)

    for index, (name, debug_image) in enumerate(contour_debug.items(), start=20):
        save_image(output_dir / f"{index:02d}_contour_{name}.png", debug_image)
    for index, (name, debug_image) in enumerate(color_debug.items(), start=30):
        save_image(output_dir / f"{index:02d}_color_{name}.png", debug_image)

    write_output_readme(output_dir, image_path)
    print(f"Saved comparison outputs to: {output_dir}")


if __name__ == "__main__":
    main()
