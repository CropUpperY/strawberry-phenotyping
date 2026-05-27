"""Tests for image preprocessing helpers."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from core.preprocessing import convert_color_space, denoise_image, normalize_image, resize_image


def test_resize_image_scales_dimensions() -> None:
    """Resize should preserve channel count and scale width and height."""
    image = np.zeros((20, 10, 3), dtype=np.uint8)

    resized = resize_image(image, scale=0.5)

    assert resized.shape == (10, 5, 3)


def test_convert_color_space_returns_same_shape_for_rgb() -> None:
    """Color conversion should keep a 3-channel output."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    converted = convert_color_space(image, target_space="RGB")

    assert converted.shape == image.shape


def test_denoise_image_rejects_even_kernel_size() -> None:
    """Blur kernels must be positive odd integers."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        denoise_image(image, method="gaussian", kernel_size=4)


def test_normalize_image_returns_same_shape() -> None:
    """Brightness normalization should preserve image shape."""
    image = np.full((12, 12, 3), 60, dtype=np.uint8)
    image[3:9, 3:9] = 180

    normalized = normalize_image(image)

    assert normalized.shape == image.shape
    assert normalized.dtype == np.uint8
