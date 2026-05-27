"""Image input and output helper functions."""

from pathlib import Path


def normalize_image_path(path: str | Path) -> Path:
    """Return a normalized image path."""
    return Path(path).expanduser().resolve()
