"""Image preprocessing helpers for phenotype analysis.

Example:
    >>> import cv2
    >>> from core.preprocessing import (
    ...     convert_color_space,
    ...     denoise_image,
    ...     normalize_image,
    ...     resize_image,
    ... )
    >>> image = cv2.imread("data/sample.jpg")
    >>> resized = resize_image(image, scale=0.5)
    >>> rgb_image = convert_color_space(resized, target_space="RGB")
    >>> denoised = denoise_image(rgb_image, method="gaussian", kernel_size=5)
    >>> normalized = normalize_image(denoised)
"""

from __future__ import annotations

import cv2
import numpy as np


SUPPORTED_COLOR_SPACES = {"RGB", "HSV", "LAB"}
SUPPORTED_DENOISE_METHODS = {"gaussian", "median"}


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize an image with a uniform scale factor.

    Args:
        image: Input image in OpenCV ndarray format.
        scale: Positive scale factor, such as ``0.5`` or ``2.0``.

    Returns:
        The resized image.
    """

    _validate_image_array(image)
    if scale <= 0:
        raise ValueError("scale must be greater than 0")

    height, width = image.shape[:2]
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def convert_color_space(image: np.ndarray, target_space: str) -> np.ndarray:
    """Convert a BGR image into RGB, HSV, or Lab color space."""

    _validate_image_array(image, require_color=True)
    normalized_target = target_space.upper()

    if normalized_target not in SUPPORTED_COLOR_SPACES:
        raise ValueError(
            f"Unsupported color space: {target_space}. "
            f"Supported values: {', '.join(sorted(SUPPORTED_COLOR_SPACES))}"
        )

    conversion_map = {
        "RGB": cv2.COLOR_BGR2RGB,
        "HSV": cv2.COLOR_BGR2HSV,
        "LAB": cv2.COLOR_BGR2LAB,
    }
    return cv2.cvtColor(image, conversion_map[normalized_target])


def denoise_image(image: np.ndarray, method: str = "gaussian", kernel_size: int = 5) -> np.ndarray:
    """Denoise an image using Gaussian blur or median blur."""

    _validate_image_array(image)
    normalized_method = method.lower()

    if normalized_method not in SUPPORTED_DENOISE_METHODS:
        raise ValueError(
            f"Unsupported denoise method: {method}. "
            f"Supported values: {', '.join(sorted(SUPPORTED_DENOISE_METHODS))}"
        )
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    if normalized_method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)

    return cv2.medianBlur(image, kernel_size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Apply basic brightness normalization.

    For color images, the function normalizes the V channel in HSV space and
    converts the result back to BGR. For grayscale images, it normalizes pixel
    intensities directly to the ``0-255`` range.
    """

    _validate_image_array(image)

    if image.ndim == 2:
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = cv2.normalize(
        hsv_image[:, :, 2],
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def _validate_image_array(image: np.ndarray, *, require_color: bool = False) -> None:
    """Validate that the input is a non-empty OpenCV image array."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if require_color and (image.ndim != 3 or image.shape[2] != 3):
        raise ValueError("image must be a 3-channel BGR image")
