"""Image loading utilities based on OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class ImageIOError(Exception):
    """Base exception for image loading errors."""


class UnsupportedImageFormatError(ImageIOError):
    """Raised when an image format is not supported."""


class ImageLoadError(ImageIOError):
    """Raised when an image cannot be decoded by OpenCV."""


def load_image(image_path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Load a single image as an OpenCV ndarray.

    Args:
        image_path: Path to the target image.
        flags: OpenCV image loading flags. Defaults to color mode.

    Returns:
        The decoded image in OpenCV ndarray format.

    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path points to a directory.
        UnsupportedImageFormatError: If the file suffix is unsupported.
        ImageLoadError: If OpenCV fails to decode the image.
    """

    path = Path(image_path).expanduser().resolve()
    _validate_image_path(path)

    image_bytes = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, flags)

    if image is None:
        raise ImageLoadError(f"Failed to read image: {path}")

    return image


def load_images_from_directory(
    directory_path: str | Path,
    *,
    recursive: bool = False,
    flags: int = cv2.IMREAD_COLOR,
) -> list[tuple[Path, np.ndarray]]:
    """Load all supported images from a directory.

    Args:
        directory_path: Folder containing image files.
        recursive: Whether to scan subdirectories recursively.
        flags: OpenCV image loading flags.

    Returns:
        A list of ``(image_path, image)`` tuples sorted by path.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
        ImageLoadError: If any discovered image cannot be decoded.
    """

    directory = Path(directory_path).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    image_paths = list_image_files(directory, recursive=recursive)
    return [(path, load_image(path, flags=flags)) for path in image_paths]


def list_image_files(directory_path: str | Path, *, recursive: bool = False) -> list[Path]:
    """List supported image files in a directory."""

    directory = Path(directory_path).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def is_supported_image_file(path: str | Path) -> bool:
    """Return whether the file suffix is supported for loading."""
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _validate_image_path(path: Path) -> None:
    """Validate that the input path points to a supported image file."""

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected an image file but got a directory: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise UnsupportedImageFormatError(
            f"Unsupported image format: {path.suffix or '<no suffix>'}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_SUFFIXES))}"
        )
