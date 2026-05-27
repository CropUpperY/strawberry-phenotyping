"""Visualization helpers for image debugging and analysis review.

Example:
    >>> from core.visualization import create_image_montage, show_analysis_views
    >>> montage = create_image_montage(
    ...     [original_image, mask_image, contour_image, hull_image],
    ...     titles=["Original", "Mask", "Contour", "Hull"],
    ... )
    >>> show_analysis_views(
    ...     original_image,
    ...     mask_image,
    ...     contour_image,
    ...     hull_image,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-missing environments
    cv2 = None
import matplotlib.pyplot as plt
import numpy as np

from utils.debug_artifacts import create_mask_overlay


@dataclass(frozen=True, slots=True)
class AnalysisDebugPreviews:
    """In-memory preview images used by the GUI debug panel."""

    mask_image: np.ndarray
    contour_image: np.ndarray
    hull_image: np.ndarray
    montage_image: np.ndarray
    montage_titles: tuple[str, ...]


def show_single_image(
    image: np.ndarray,
    *,
    title: str = "Image",
    figure_size: tuple[int, int] = (6, 6),
    cmap: str | None = None,
) -> None:
    """Display a single image with matplotlib."""

    _require_cv2()
    display_image, inferred_cmap = _prepare_display_image(image)

    plt.figure(figsize=figure_size)
    plt.imshow(display_image, cmap=cmap or inferred_cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_analysis_views(
    original_image: np.ndarray,
    mask_image: np.ndarray,
    contour_image: np.ndarray,
    hull_image: np.ndarray,
    *,
    figure_size: tuple[int, int] = (12, 8),
) -> None:
    """Display original, mask, contour, and convex hull images in a 2x2 layout."""

    images = [original_image, mask_image, contour_image, hull_image]
    titles = ["Original", "Mask", "Contour", "Convex Hull"]
    _show_image_grid(images, titles=titles, columns=2, figure_size=figure_size)


def create_image_montage(
    images: list[np.ndarray],
    *,
    titles: list[str] | None = None,
    columns: int = 2,
    tile_size: tuple[int, int] | None = None,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Create a BGR montage image for multiple views.

    Args:
        images: A list of images in OpenCV format.
        titles: Optional title labels drawn on each tile.
        columns: Number of columns in the montage grid.
        tile_size: Optional fixed ``(width, height)`` for each tile.
        background_color: Background color for the montage canvas in BGR.

    Returns:
        A stitched BGR image suitable for saving or further display.
    """

    _require_cv2()
    if not images:
        raise ValueError("images must contain at least one image")
    if columns <= 0:
        raise ValueError("columns must be greater than 0")
    if titles is not None and len(titles) != len(images):
        raise ValueError("titles length must match images length")

    validated_images = [_ensure_color_image(image) for image in images]
    tile_width, tile_height = _resolve_tile_size(validated_images, tile_size)

    rows = ceil(len(validated_images) / columns)
    canvas = np.full(
        (rows * tile_height, columns * tile_width, 3),
        background_color,
        dtype=np.uint8,
    )

    for index, image in enumerate(validated_images):
        row = index // columns
        column = index % columns
        resized = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)

        y_start = row * tile_height
        y_end = y_start + tile_height
        x_start = column * tile_width
        x_end = x_start + tile_width
        canvas[y_start:y_end, x_start:x_end] = resized

        if titles is not None:
            _draw_title(canvas, titles[index], x_start, y_start)

    return canvas


def show_image_montage(
    images: list[np.ndarray],
    *,
    titles: list[str] | None = None,
    columns: int = 2,
    tile_size: tuple[int, int] | None = None,
    figure_size: tuple[int, int] = (12, 8),
) -> np.ndarray:
    """Create and display a multi-image montage.

    Returns:
        The stitched BGR montage image.
    """

    montage = create_image_montage(images, titles=titles, columns=columns, tile_size=tile_size)
    show_single_image(montage, title="Montage", figure_size=figure_size)
    return montage


def build_analysis_debug_previews(
    result: Any,
    *,
    tile_size: tuple[int, int] = (320, 240),
) -> AnalysisDebugPreviews | None:
    """Build GUI-friendly debug previews from one analysis result."""

    if cv2 is None:
        return None
    top_segmentation = getattr(result, "top_segmentation", None)
    if top_segmentation is None or not getattr(top_segmentation, "has_foreground", False):
        return None

    calibration_results = getattr(result, "calibration_results", {})
    front_segmentations = getattr(result, "front_segmentations", {})
    top_image = _resolve_preview_source_image(top_segmentation, calibration_results.get("TOP"))

    mask_image = create_mask_overlay(_ensure_color_image(top_image), top_segmentation.mask, alpha=0.55)
    contour_image = _ensure_color_image(top_segmentation.contour_image)
    hull_image = _ensure_color_image(top_segmentation.hull_image)

    montage_images = [top_image, mask_image, contour_image, hull_image]
    montage_titles = ["TOP Corrected", "TOP Mask", "TOP Contour", "TOP Hull"]

    for view_name in ("FRONT-1", "FRONT-2"):
        front_segmentation = front_segmentations.get(view_name)
        if front_segmentation is None or not getattr(front_segmentation, "has_foreground", False):
            continue

        front_image = _resolve_preview_source_image(front_segmentation, calibration_results.get(view_name))
        montage_images.extend(
            [
                front_image,
                create_mask_overlay(_ensure_color_image(front_image), front_segmentation.mask, alpha=0.55),
                _ensure_color_image(front_segmentation.contour_image),
            ]
        )
        montage_titles.extend([f"{view_name} Corrected", f"{view_name} Mask", f"{view_name} Contour"])

    montage_image = create_image_montage(
        montage_images,
        titles=montage_titles,
        columns=2,
        tile_size=tile_size,
        background_color=(28, 36, 32),
    )
    return AnalysisDebugPreviews(
        mask_image=mask_image,
        contour_image=contour_image,
        hull_image=hull_image,
        montage_image=montage_image,
        montage_titles=tuple(montage_titles),
    )


def _show_image_grid(
    images: list[np.ndarray],
    *,
    titles: list[str],
    columns: int,
    figure_size: tuple[int, int],
) -> None:
    """Render a collection of images in a matplotlib subplot grid."""

    rows = ceil(len(images) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=figure_size)
    flat_axes = np.atleast_1d(axes).ravel()

    for axis, image, title in zip(flat_axes, images, titles):
        display_image, cmap = _prepare_display_image(image)
        axis.imshow(display_image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")

    for axis in flat_axes[len(images) :]:
        axis.axis("off")

    figure.tight_layout()
    plt.show()


def _prepare_display_image(image: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Convert an OpenCV image into a matplotlib-friendly representation."""

    _require_cv2()
    _validate_image_array(image)

    if image.ndim == 2:
        return image, "gray"

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None

    raise ValueError("image must be either a grayscale or 3-channel BGR image")


def _resolve_preview_source_image(segmentation: Any, calibration_result: Any | None) -> np.ndarray:
    """Resolve the best available color image to pair with one segmentation result."""

    calibrated = getattr(calibration_result, "corrected_image", None)
    if calibrated is not None:
        return _ensure_color_image(calibrated)

    debug_images = getattr(segmentation, "debug_images", {})
    original = debug_images.get("original")
    if original is not None:
        return _ensure_color_image(original)

    raise ValueError("segmentation result does not contain a usable source image")


def _ensure_color_image(image: np.ndarray) -> np.ndarray:
    """Ensure the output image is a 3-channel BGR array."""

    _require_cv2()
    _validate_image_array(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError("image must be either a grayscale or 3-channel BGR image")


def _resolve_tile_size(images: list[np.ndarray], tile_size: tuple[int, int] | None) -> tuple[int, int]:
    """Resolve the montage tile size."""

    if tile_size is not None:
        tile_width, tile_height = tile_size
        if tile_width <= 0 or tile_height <= 0:
            raise ValueError("tile_size values must be greater than 0")
        return tile_width, tile_height

    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    return max_width, max_height


def _draw_title(canvas: np.ndarray, title: str, x_start: int, y_start: int) -> None:
    """Draw a title label onto a montage tile."""

    _require_cv2()
    cv2.putText(
        canvas,
        title,
        (x_start + 10, y_start + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _validate_image_array(image: np.ndarray) -> None:
    """Validate that the input is a non-empty image array."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.size == 0:
        raise ValueError("image must not be empty")


def _require_cv2() -> None:
    """Ensure OpenCV is available for visualization helpers."""

    if cv2 is None:
        raise ModuleNotFoundError("No module named 'cv2'")
