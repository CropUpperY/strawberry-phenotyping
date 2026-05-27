"""Tests for the OpenCV image loading helpers."""

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from core.image_io import ImageLoadError, UnsupportedImageFormatError, load_image, load_images_from_directory


def test_load_image_reads_supported_file(tmp_path: Path) -> None:
    """A supported image should be loaded as an OpenCV ndarray."""
    image_path = tmp_path / "sample.png"
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :] = (0, 255, 0)
    cv2.imwrite(str(image_path), image)

    loaded = load_image(image_path)

    assert loaded.shape == (8, 8, 3)
    assert loaded.dtype == np.uint8


def test_load_image_raises_for_missing_file(tmp_path: Path) -> None:
    """Missing files should raise a clear error."""
    missing_path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        load_image(missing_path)


def test_load_image_raises_for_unsupported_format(tmp_path: Path) -> None:
    """Unsupported file suffixes should be rejected before decoding."""
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not an image", encoding="utf-8")

    with pytest.raises(UnsupportedImageFormatError):
        load_image(text_file)


def test_load_image_raises_for_corrupt_image(tmp_path: Path) -> None:
    """Corrupt image data with a supported suffix should raise ImageLoadError."""
    bad_image = tmp_path / "broken.jpg"
    bad_image.write_bytes(b"this is not a valid image")

    with pytest.raises(ImageLoadError):
        load_image(bad_image)


def test_load_images_from_directory_reads_supported_images_only(tmp_path: Path) -> None:
    """Directory loading should include all supported images and ignore other files."""
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.bmp"
    ignored = tmp_path / "readme.txt"

    cv2.imwrite(str(image_a), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(image_b), np.zeros((6, 6, 3), dtype=np.uint8))
    ignored.write_text("ignore me", encoding="utf-8")

    loaded_images = load_images_from_directory(tmp_path)

    assert [path.name for path, _ in loaded_images] == ["a.png", "b.bmp"]
    assert loaded_images[0][1].shape == (4, 4, 3)
    assert loaded_images[1][1].shape == (6, 6, 3)
