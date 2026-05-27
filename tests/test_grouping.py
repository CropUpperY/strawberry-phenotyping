"""Tests for grouping multi-view plant images."""

from pathlib import Path

import pytest

from core.grouping import (
    VIEW_FRONT_0,
    VIEW_FRONT_180,
    VIEW_TOP,
    collect_grouping_suggestions,
    find_incomplete_groups,
    group_image_files,
    parse_grouped_image_name,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("1AB_TOP.png", ("1AB", VIEW_TOP)),
        ("1AB-1.png", ("1AB", VIEW_FRONT_0)),
        ("1AB-2.png", ("1AB", VIEW_FRONT_180)),
        ("sample_TOP.png", ("sample", VIEW_TOP)),
        ("1AB_2.png", None),
        ("ignore.png", None),
    ],
)
def test_parse_grouped_image_name(filename: str, expected: tuple[str, str] | None) -> None:
    """Filename parsing should recognize supported multi-view patterns."""
    assert parse_grouped_image_name(filename) == expected


def test_group_image_files_collects_complete_sample(tmp_path: Path) -> None:
    """Matching TOP, 1, and 2 images should be grouped into one sample."""
    for filename in ("1AB_TOP.png", "1AB-1.png", "1AB-2.png", "notes.txt"):
        (tmp_path / filename).write_bytes(b"test")

    groups = group_image_files(tmp_path)

    assert len(groups) == 1
    group = groups[0]
    assert group.sample_id == "1AB"
    assert group.is_complete is True
    assert group.missing_views == []
    assert group.top_image is not None and group.top_image.name == "1AB_TOP.png"
    assert group.front_0_image is not None and group.front_0_image.name == "1AB-1.png"
    assert group.front_180_image is not None and group.front_180_image.name == "1AB-2.png"


def test_group_image_files_reports_missing_views(tmp_path: Path) -> None:
    """Incomplete image sets should be preserved with missing view metadata."""
    for filename in ("2AB_TOP.png", "2AB-1.png"):
        (tmp_path / filename).write_bytes(b"test")

    groups = group_image_files(tmp_path)
    incomplete_groups = find_incomplete_groups(groups)

    assert len(groups) == 1
    assert len(incomplete_groups) == 1
    assert incomplete_groups[0].sample_id == "2AB"
    assert incomplete_groups[0].missing_views == [VIEW_FRONT_180]


def test_group_image_files_handles_duplicate_views_as_extras(tmp_path: Path) -> None:
    """Later duplicate view files should not overwrite the first matched image."""
    for filename in ("3AB_TOP.png", "3AB-1.jpg", "3AB-1.png", "3AB-2.png"):
        (tmp_path / filename).write_bytes(b"test")

    groups = group_image_files(tmp_path)

    assert len(groups) == 1
    assert groups[0].front_0_image is not None
    assert groups[0].front_0_image.name == "3AB-1.jpg"
    assert [path.name for path in groups[0].extra_files] == ["3AB-1.png"]


def test_collect_grouping_suggestions_detects_nonstandard_names(tmp_path: Path) -> None:
    """Non-standard but recoverable filenames should produce grouping suggestions."""

    for filename in ("4AB_TOP.png", "4AB-2.png", "4A-1.png", "40A_TOPP.png", "2AB-1_.png"):
        (tmp_path / filename).write_bytes(b"test")

    suggestions = collect_grouping_suggestions(tmp_path)
    suggested_names = {item.image_path.name for item in suggestions}

    assert "4A-1.png" in suggested_names
    assert "40A_TOPP.png" in suggested_names
    assert "2AB-1_.png" in suggested_names


def test_collect_grouping_suggestions_includes_unrecognized_names(tmp_path: Path) -> None:
    """Completely unrecognized image filenames should still require user confirmation."""

    (tmp_path / "1AB_TOP.png").write_bytes(b"test")
    (tmp_path / "mystery_view.png").write_bytes(b"test")

    suggestions = collect_grouping_suggestions(tmp_path)

    assert len(suggestions) == 1
    assert suggestions[0].image_path.name == "mystery_view.png"
    assert suggestions[0].options == []


def test_group_image_files_accepts_manual_overrides(tmp_path: Path) -> None:
    """Overrides should allow manual grouping of non-standard filenames."""

    weird_path = tmp_path / "2AB-1_.png"
    (tmp_path / "2AB_TOP.png").write_bytes(b"test")
    (tmp_path / "2AB-2.png").write_bytes(b"test")
    weird_path.write_bytes(b"test")

    groups = group_image_files(
        tmp_path,
        overrides={str(weird_path.resolve()): ("2AB", VIEW_FRONT_0)},
    )

    assert len(groups) == 1
    assert groups[0].is_complete is True
    assert groups[0].front_0_image is not None
    assert groups[0].front_0_image.name == "2AB-1_.png"


def test_group_image_files_skips_nonstandard_sample_ids(tmp_path: Path) -> None:
    """Files like 4A-1/4A-2 should be treated as unrecognized, not grouped."""

    for filename in ("4A-1.png", "4A-2.png", "4AB_TOP.png", "4AB-1.png", "4AB-2.png"):
        (tmp_path / filename).write_bytes(b"test")

    groups = group_image_files(tmp_path)

    assert len(groups) == 1
    assert groups[0].sample_id == "4AB"
