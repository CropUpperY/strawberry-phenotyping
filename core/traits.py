"""Trait extraction helpers for strawberry phenotype analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class TopTraitMeasurements:
    """TOP-view phenotype measurements derived from segmentation results."""

    leaf_area_pixels: int
    convex_hull_area_pixels: float
    canopy_diameter_pixels: float
    canopy_diameter_endpoints: tuple[tuple[int, int], tuple[int, int]] | None
    greenness_exg_mean: float


@dataclass(frozen=True, slots=True)
class FrontViewMeasurements:
    """Per-view side-projection measurements."""

    canopy_height_pixels: int
    canopy_width_pixels: int
    projection_area_pixels: int


@dataclass(frozen=True, slots=True)
class FrontTraitMeasurements:
    """Fused FRONT-view phenotype measurements."""

    front_0: FrontViewMeasurements
    front_180: FrontViewMeasurements
    fused_canopy_height_pixels: int
    fused_canopy_width_pixels: int
    fused_projection_area_pixels: float


def compute_top_traits(top_image: np.ndarray, top_segmentation: Any) -> TopTraitMeasurements:
    """Compute TOP-view traits from the segmented plant region."""

    if top_image.ndim != 3 or top_image.shape[2] != 3:
        raise ValueError("top_image must be a 3-channel BGR image")

    mask = getattr(top_segmentation, "mask", None)
    if mask is None or mask.ndim != 2:
        raise ValueError("top_segmentation must provide a single-channel mask")

    foreground = mask > 0
    leaf_area_pixels = int(np.count_nonzero(foreground))
    convex_hull_area_pixels = float(getattr(top_segmentation, "hull_area_pixels", 0.0))
    canopy_diameter_pixels, canopy_diameter_endpoints = compute_convex_hull_max_distance(
        getattr(top_segmentation, "convex_hull", None)
    )

    if leaf_area_pixels == 0:
        greenness_exg_mean = 0.0
    else:
        b_channel = top_image[:, :, 0].astype(np.float32)
        g_channel = top_image[:, :, 1].astype(np.float32)
        r_channel = top_image[:, :, 2].astype(np.float32)
        exg = 2.0 * g_channel - r_channel - b_channel
        greenness_exg_mean = float(exg[foreground].mean())

    return TopTraitMeasurements(
        leaf_area_pixels=leaf_area_pixels,
        convex_hull_area_pixels=convex_hull_area_pixels,
        canopy_diameter_pixels=canopy_diameter_pixels,
        canopy_diameter_endpoints=canopy_diameter_endpoints,
        greenness_exg_mean=greenness_exg_mean,
    )


def compute_convex_hull_max_distance(
    convex_hull: Any,
) -> tuple[float, tuple[tuple[int, int], tuple[int, int]] | None]:
    """Return the maximum pairwise distance on the convex hull."""

    if convex_hull is None:
        return 0.0, None

    points = np.asarray(convex_hull, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2:
        return 0.0, None

    deltas = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    max_index = int(np.argmax(distances))
    row_index, col_index = divmod(max_index, distances.shape[1])
    point_a = tuple(int(round(value)) for value in points[row_index])
    point_b = tuple(int(round(value)) for value in points[col_index])
    return float(distances[row_index, col_index]), (point_a, point_b)


def compute_front_view_traits(front_segmentation: Any) -> FrontViewMeasurements:
    """Compute side-view traits from one FRONT segmentation result."""

    bounding_box = getattr(front_segmentation, "bounding_box", None)
    mask_area_pixels = int(getattr(front_segmentation, "mask_area_pixels", 0))

    if bounding_box is None:
        return FrontViewMeasurements(
            canopy_height_pixels=0,
            canopy_width_pixels=0,
            projection_area_pixels=mask_area_pixels,
        )

    _, _, width, height = bounding_box
    return FrontViewMeasurements(
        canopy_height_pixels=int(height),
        canopy_width_pixels=int(width),
        projection_area_pixels=mask_area_pixels,
    )


def fuse_front_traits(front_0_segmentation: Any, front_180_segmentation: Any) -> FrontTraitMeasurements:
    """Fuse FRONT-1 and FRONT-2 measurements to reduce occlusion bias."""

    front_0 = compute_front_view_traits(front_0_segmentation)
    front_180 = compute_front_view_traits(front_180_segmentation)

    return FrontTraitMeasurements(
        front_0=front_0,
        front_180=front_180,
        fused_canopy_height_pixels=max(front_0.canopy_height_pixels, front_180.canopy_height_pixels),
        fused_canopy_width_pixels=max(front_0.canopy_width_pixels, front_180.canopy_width_pixels),
        fused_projection_area_pixels=(
            front_0.projection_area_pixels + front_180.projection_area_pixels
        )
        / 2.0,
    )
