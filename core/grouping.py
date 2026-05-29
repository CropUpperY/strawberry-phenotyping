"""Utilities for grouping multi-view plant images into a single sample.

Example:
    >>> from core.grouping import group_image_files
    >>> groups = group_image_files("data")
    >>> first_group = groups[0]
    >>> first_group.sample_id
    '1AB'
    >>> first_group.top_image.name
    '1AB_TOP.png'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

VIEW_TOP = "TOP"
VIEW_FRONT_0 = "1"
VIEW_FRONT_180 = "2"
SUPPORTED_VIEWS = (VIEW_TOP, VIEW_FRONT_0, VIEW_FRONT_180)
TWO_VIEW_REQUIRED_VIEWS = (VIEW_TOP, VIEW_FRONT_0)
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
STANDARD_SAMPLE_ID_PATTERN = re.compile(r"^\d+AB$", re.IGNORECASE)
COTTON_VIEW_FOLDER_MAP = {
    "\u4fef\u89c6\u56fe": VIEW_TOP,
    "\u5e73\u89c6\u56fe": VIEW_FRONT_0,
}


@dataclass(slots=True)
class GroupingSuggestion:
    """Suggestion payload for one potentially non-standard filename."""

    image_path: Path
    reason: str
    options: list[tuple[str, str, str]]  # (sample_id, view, label)


@dataclass(slots=True)
class PlantImageGroup:
    """A multi-view image group for a single plant."""

    sample_id: str
    top_image: Path | None = None
    front_0_image: Path | None = None
    front_180_image: Path | None = None
    required_views: tuple[str, ...] = SUPPORTED_VIEWS
    extra_files: list[Path] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Return whether the group contains all required views."""
        return all(self._path_for_view(view) is not None for view in self.required_views)

    @property
    def missing_views(self) -> list[str]:
        """Return the missing view labels for the current group."""
        missing: list[str] = []
        for view in self.required_views:
            if self._path_for_view(view) is None:
                missing.append(view)
        return missing

    def to_dict(self) -> dict[str, str | list[str] | bool | None]:
        """Serialize the group into a dictionary for downstream processing."""
        return {
            "sample_id": self.sample_id,
            "top": str(self.top_image) if self.top_image else None,
            "front_0": str(self.front_0_image) if self.front_0_image else None,
            "front_180": str(self.front_180_image) if self.front_180_image else None,
            "is_complete": self.is_complete,
            "missing_views": self.missing_views,
            "required_views": list(self.required_views),
            "extra_files": [str(path) for path in self.extra_files],
        }

    def _path_for_view(self, view: str) -> Path | None:
        """Return the image path for a normalized grouping view label."""

        if view == VIEW_TOP:
            return self.top_image
        if view == VIEW_FRONT_0:
            return self.front_0_image
        if view == VIEW_FRONT_180:
            return self.front_180_image
        raise ValueError(f"Unsupported view label: {view}")


def group_image_files(
    directory_path: str | Path,
    *,
    overrides: dict[str | Path, tuple[str, str]] | None = None,
) -> list[PlantImageGroup]:
    """Scan a directory and group images by sample ID and view.

    Supported naming examples:
        - ``1AB_TOP.png``
        - ``1AB-1.png``
        - ``1AB-2.png``

    Args:
        directory_path: Directory that contains strawberry image files.

    Returns:
        A sorted list of grouped plant samples.
    """

    directory = Path(directory_path).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    groups = _group_two_view_folder_images(directory)
    normalized_overrides = _normalize_override_keys(overrides)

    for image_path in sorted(path for path in directory.iterdir() if _is_supported_image(path)):
        parsed = normalized_overrides.get(str(image_path.resolve()))
        if parsed is None:
            parsed = parse_grouped_image_name(image_path)
        if parsed is None:
            continue

        sample_id, view = parsed
        if str(image_path.resolve()) not in normalized_overrides and not _is_standard_sample_id(sample_id):
            continue
        group = groups.setdefault(sample_id, PlantImageGroup(sample_id=sample_id))
        _assign_view(group, image_path, view)

    return sorted(groups.values(), key=lambda group: group.sample_id)


def _group_two_view_folder_images(directory: Path) -> dict[str, PlantImageGroup]:
    """Group images stored under cotton-style top/front view folders."""

    groups: dict[str, PlantImageGroup] = {}
    view_folders = {
        COTTON_VIEW_FOLDER_MAP[path.name.strip()]: path
        for path in directory.iterdir()
        if path.is_dir() and path.name.strip() in COTTON_VIEW_FOLDER_MAP
    }
    if not view_folders:
        return groups

    for view, folder_path in view_folders.items():
        for image_path in sorted(path for path in folder_path.iterdir() if _is_supported_image(path)):
            sample_id = _normalize_sample_id(image_path.stem)
            group = groups.setdefault(
                sample_id,
                PlantImageGroup(sample_id=sample_id, required_views=TWO_VIEW_REQUIRED_VIEWS),
            )
            _assign_view(group, image_path, view)

    return groups


def collect_grouping_suggestions(directory_path: str | Path) -> list[GroupingSuggestion]:
    """Collect grouping suggestions for files with non-standard or suspicious names."""

    directory = Path(directory_path).expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        return []

    image_files = sorted(path for path in directory.iterdir() if _is_supported_image(path))
    known_sample_ids: set[str] = set()
    for image_path in image_files:
        parsed = parse_grouped_image_name(image_path)
        if parsed is not None:
            known_sample_ids.add(parsed[0])

    suggestions: list[GroupingSuggestion] = []
    for image_path in image_files:
        parsed = parse_grouped_image_name(image_path)
        if parsed is None:
            relaxed = _parse_relaxed_grouped_name(image_path)
            if relaxed is None:
                suggestions.append(
                    GroupingSuggestion(
                        image_path=image_path,
                        reason="无法从文件名判断样本编号和视角，需要手动确认。",
                        options=[],
                    )
                )
                continue
            sample_guess, view_guess, reason = relaxed
            options = _build_grouping_options(sample_guess, view_guess, known_sample_ids)
            if options:
                suggestions.append(GroupingSuggestion(image_path=image_path, reason=reason, options=options))
            continue

        sample_id, view = parsed
        if _is_standard_sample_id(sample_id):
            continue
        options = _build_grouping_options(sample_id, view, known_sample_ids)
        if options and not (len(options) == 1 and options[0][0] == sample_id):
            suggestions.append(
                GroupingSuggestion(
                    image_path=image_path,
                    reason="样本编号不符合常见格式（推荐 数字+AB）。",
                    options=options,
                )
            )

    return suggestions


def find_incomplete_groups(groups: list[PlantImageGroup]) -> list[PlantImageGroup]:
    """Return only groups that are missing one or more required views."""
    return [group for group in groups if not group.is_complete]


def parse_grouped_image_name(image_path: str | Path) -> tuple[str, str] | None:
    """Parse a grouped image filename into ``(sample_id, view)``.

    The parser is case-insensitive and expects ``_TOP`` for the top view plus
    ``-1`` and ``-2`` for the two front views. Unrecognized filenames return
    ``None``.
    """

    stem = Path(image_path).stem
    uppercase_stem = stem.upper()

    top_suffix = "_TOP"
    if uppercase_stem.endswith(top_suffix):
        sample_id = stem[: -len(top_suffix)]
        return _normalize_sample_id(sample_id), VIEW_TOP

    for view in (VIEW_FRONT_0, VIEW_FRONT_180):
        suffix = f"-{view}"
        if uppercase_stem.endswith(suffix):
            sample_id = stem[: -len(suffix)]
            return _normalize_sample_id(sample_id), view

    return None


def _assign_view(group: PlantImageGroup, image_path: Path, view: str) -> None:
    """Assign a parsed image path to the correct group slot."""

    if view == VIEW_TOP:
        _set_or_collect_duplicate(group, "top_image", image_path)
        return
    if view == VIEW_FRONT_0:
        _set_or_collect_duplicate(group, "front_0_image", image_path)
        return
    if view == VIEW_FRONT_180:
        _set_or_collect_duplicate(group, "front_180_image", image_path)
        return

    raise ValueError(f"Unsupported view label: {view}")


def _set_or_collect_duplicate(group: PlantImageGroup, attribute_name: str, image_path: Path) -> None:
    """Store the first image for a slot and keep later duplicates as extras."""

    current_value = getattr(group, attribute_name)
    if current_value is None:
        setattr(group, attribute_name, image_path)
    else:
        group.extra_files.append(image_path)


def _normalize_sample_id(sample_id: str) -> str:
    """Normalize the sample ID parsed from filenames."""
    normalized = sample_id.rstrip("_- ").strip()
    if not normalized:
        raise ValueError("sample_id cannot be empty")
    return normalized


def _is_supported_image(path: Path) -> bool:
    """Return whether the path points to a supported image file."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _normalize_override_keys(overrides: dict[str | Path, tuple[str, str]] | None) -> dict[str, tuple[str, str]]:
    """Normalize override keys to absolute path strings."""

    if not overrides:
        return {}

    normalized: dict[str, tuple[str, str]] = {}
    for key, value in overrides.items():
        key_path = Path(key).expanduser().resolve()
        sample_id, view = value
        normalized[str(key_path)] = (_normalize_sample_id(sample_id), view)
    return normalized


def _is_standard_sample_id(sample_id: str) -> bool:
    """Return whether sample ID matches the common format (e.g. 12AB)."""

    return bool(STANDARD_SAMPLE_ID_PATTERN.match(sample_id))


def _parse_relaxed_grouped_name(image_path: str | Path) -> tuple[str, str, str] | None:
    """Try parsing loosely formatted grouped filenames and return a reason."""

    stem = Path(image_path).stem
    upper = stem.upper()

    top_match = re.match(r"^(?P<sample>.+)_TOP[_A-Z0-9]*$", upper)
    if top_match:
        sample_guess = stem[: top_match.end("sample")]
        return _normalize_sample_id(sample_guess), VIEW_TOP, "TOP后缀格式不标准（如 TOPP、TOP_）。"

    front_match = re.match(r"^(?P<sample>.+)-(?P<view>[12])[_-]*$", upper)
    if front_match:
        sample_guess = stem[: front_match.end("sample")]
        view = front_match.group("view")
        return _normalize_sample_id(sample_guess), view, "FRONT后缀格式不标准（如 -1_、-2-）。"

    return None


def _build_grouping_options(sample_guess: str, view: str, known_sample_ids: set[str]) -> list[tuple[str, str, str]]:
    """Build sample/view options for user selection."""

    candidates: list[str] = [sample_guess]
    candidates.extend(_guess_sample_id_variants(sample_guess, known_sample_ids))

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_sample_id(candidate)
        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(normalized)

    view_label = "TOP" if view == VIEW_TOP else f"FRONT-{view}"
    return [(sample_id, view, f"{sample_id} / {view_label}") for sample_id in unique_candidates]


def _guess_sample_id_variants(sample_id: str, known_sample_ids: set[str]) -> list[str]:
    """Generate likely sample-id variants for typo recovery."""

    variants: list[str] = []
    upper_id = sample_id.upper()

    if re.match(r"^\d+A$", upper_id):
        variants.append(f"{upper_id}B")
    if re.match(r"^\d+$", upper_id):
        variants.append(f"{upper_id}AB")

    digit_prefix = re.match(r"^(\d+)", upper_id)
    if digit_prefix:
        number = digit_prefix.group(1)
        exact = [item for item in known_sample_ids if item.upper().startswith(number)]
        variants.extend(exact)

    return variants
