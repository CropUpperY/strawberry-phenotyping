from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import experiments.top_segmentation.compare_flower_masks as compare_flower_masks
import experiments.top_segmentation.debug_flower_preserving_top_mask as debug_flower_preserving_top_mask


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def test_compare_flower_masks_uses_default_top_image_when_no_args(tmp_path: Path, monkeypatch) -> None:
    """PyCharm direct-run should work without forcing --image."""

    default_image = tmp_path / "data" / "111AB_TOP.png"
    _write_png(default_image, np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(compare_flower_masks, "PROJECT_ROOT", tmp_path)

    args = compare_flower_masks.parse_args([])

    assert Path(args.image) == default_image
    assert args.used_default_image is True


def test_debug_flower_preserving_top_mask_uses_default_top_image_when_no_args(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The flower-preserving debug script should also support direct IDE runs."""

    default_image = tmp_path / "data" / "111AB_TOP.png"
    _write_png(default_image, np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(debug_flower_preserving_top_mask, "PROJECT_ROOT", tmp_path)

    args = debug_flower_preserving_top_mask.parse_args([])

    assert Path(args.image) == default_image
    assert args.used_default_image is True
