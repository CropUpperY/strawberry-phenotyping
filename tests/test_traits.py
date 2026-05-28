"""Tests for trait computation helpers."""

import numpy as np

from core.traits import compute_convex_hull_max_distance, compute_front_view_traits, compute_top_traits, fuse_front_traits


class DummyTopSegmentation:
    """Minimal TOP segmentation object for trait computation tests."""

    def __init__(self) -> None:
        self.mask = np.array(
            [
                [0, 255, 255],
                [0, 255, 255],
            ],
            dtype=np.uint8,
        )
        self.hull_area_pixels = 10.5
        self.convex_hull = np.array([[[0, 0]], [[2, 0]], [[2, 1]], [[0, 1]]], dtype=np.int32)


class DummyTopSegmentationWithLeafMask(DummyTopSegmentation):
    """TOP segmentation where display mask contains a flower but leaf_mask excludes it."""

    def __init__(self) -> None:
        super().__init__()
        self.mask = np.array(
            [
                [0, 255, 255],
                [0, 255, 255],
                [0, 0, 255],
            ],
            dtype=np.uint8,
        )
        self.leaf_mask = np.array(
            [
                [0, 255, 255],
                [0, 255, 255],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )


class DummyFrontSegmentation:
    """Minimal FRONT segmentation object for trait computation tests."""

    def __init__(self, width: int, height: int, area: int) -> None:
        self.bounding_box = (10, 20, width, height)
        self.mask_area_pixels = area


def test_compute_top_traits_returns_expected_values() -> None:
    """Leaf area, hull area, and greenness should be derived from the mask."""

    image = np.array(
        [
            [[0, 0, 0], [10, 100, 10], [10, 120, 10]],
            [[0, 0, 0], [10, 140, 10], [10, 160, 10]],
        ],
        dtype=np.uint8,
    )

    measurements = compute_top_traits(image, DummyTopSegmentation())

    assert measurements.leaf_area_pixels == 4
    assert measurements.convex_hull_area_pixels == 10.5
    assert measurements.canopy_diameter_pixels > 0
    assert measurements.canopy_diameter_endpoints is not None
    assert measurements.greenness_exg_mean > 0


def test_compute_top_traits_prefers_leaf_mask_over_display_mask() -> None:
    """Leaf traits should not count flowers that are only present in the display plant mask."""

    image = np.array(
        [
            [[0, 0, 0], [10, 100, 10], [10, 120, 10]],
            [[0, 0, 0], [10, 140, 10], [10, 160, 10]],
            [[0, 0, 0], [245, 245, 245], [245, 245, 245]],
        ],
        dtype=np.uint8,
    )

    measurements = compute_top_traits(image, DummyTopSegmentationWithLeafMask())

    assert measurements.leaf_area_pixels == 4
    assert measurements.greenness_exg_mean > 0


def test_compute_convex_hull_max_distance_returns_farthest_pair() -> None:
    """Convex-hull diameter should be the farthest pairwise point distance."""

    distance, endpoints = compute_convex_hull_max_distance(
        np.array([[[0, 0]], [[3, 0]], [[3, 4]], [[0, 4]]], dtype=np.int32)
    )

    assert round(distance, 2) == 5.0
    assert endpoints is not None


def test_compute_front_view_traits_uses_bounding_box_and_mask_area() -> None:
    """FRONT-view traits should come from the side-view bounding box."""

    measurements = compute_front_view_traits(DummyFrontSegmentation(width=45, height=90, area=2300))

    assert measurements.canopy_height_pixels == 90
    assert measurements.canopy_width_pixels == 45
    assert measurements.projection_area_pixels == 2300


def test_fuse_front_traits_combines_two_views() -> None:
    """Two FRONT views should be fused with max height/width and mean projection area."""

    measurements = fuse_front_traits(
        DummyFrontSegmentation(width=40, height=88, area=2200),
        DummyFrontSegmentation(width=52, height=81, area=2600),
    )

    assert measurements.fused_canopy_height_pixels == 88
    assert measurements.fused_canopy_width_pixels == 52
    assert measurements.fused_projection_area_pixels == 2400.0
