"""Segmentation utilities for extracting strawberry canopies from RGB images."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.preprocessing import denoise_image, normalize_image


@dataclass(slots=True)
class TopSegmentationResult:
    """Result payload for TOP-view plant segmentation."""

    status: str
    message: str
    mask: np.ndarray
    contours: list[np.ndarray]
    largest_contour: np.ndarray | None
    convex_hull: np.ndarray | None
    contour_image: np.ndarray
    hull_image: np.ndarray
    mask_area_pixels: int
    hull_area_pixels: float
    contour_count: int
    bounding_box: tuple[int, int, int, int] | None
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    leaf_mask: np.ndarray | None = None

    @property
    def has_foreground(self) -> bool:
        """Return whether the segmentation contains a valid foreground object."""

        return self.mask_area_pixels > 0 and self.largest_contour is not None


@dataclass(slots=True)
class FrontSegmentationResult:
    """Result payload for one FRONT-view plant segmentation."""

    status: str
    message: str
    mask: np.ndarray
    contours: list[np.ndarray]
    largest_contour: np.ndarray | None
    contour_image: np.ndarray
    bounding_box_image: np.ndarray
    mask_area_pixels: int
    contour_count: int
    bounding_box: tuple[int, int, int, int] | None
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def has_foreground(self) -> bool:
        """Return whether the segmentation contains a valid foreground object."""

        return self.mask_area_pixels > 0 and self.bounding_box is not None


def segment_top_view_plant(
    image: np.ndarray,
    *,
    min_component_area_ratio: float = 0.0005,
) -> TopSegmentationResult:
    """Segment the strawberry plant from a TOP-view RGB image."""

    prepared = _prepare_segmentation(image, min_component_area_ratio=min_component_area_ratio)
    prepared["filtered_mask_before_card_removal"] = prepared["filtered_mask"].copy()
    filtered_mask = _filter_top_view_components(prepared["filtered_mask"])
    prepared["after_right_card_removal"] = filtered_mask.copy()
    prepared["removed_right_card_mask"] = cv2.subtract(prepared["filtered_mask_before_card_removal"], filtered_mask)
    filtered_mask, top_band_candidate, removed_top_band = _remove_top_attached_pot_band(filtered_mask)
    prepared["top_band_candidate_mask"] = top_band_candidate
    prepared["removed_top_pot_band"] = removed_top_band
    leaf_mask = filtered_mask
    plant_mask, reproductive_debug = _augment_top_mask_with_reproductive_organs(
        prepared["denoised"],
        leaf_mask,
    )
    prepared.update(reproductive_debug)
    prepared["leaf_mask"] = leaf_mask
    prepared["filtered_mask"] = plant_mask
    contours = _find_external_contours(plant_mask)

    if not contours:
        empty_overlay = image.copy()
        return TopSegmentationResult(
            status="no_foreground",
            message="No strawberry canopy was detected in the TOP image.",
            mask=filtered_mask,
            contours=[],
            largest_contour=None,
            convex_hull=None,
            contour_image=empty_overlay,
            hull_image=empty_overlay.copy(),
            mask_area_pixels=0,
            hull_area_pixels=0.0,
            contour_count=0,
            bounding_box=None,
            debug_images=prepared,
            leaf_mask=leaf_mask,
        )

    all_points = np.vstack(contours)
    largest_contour = max(contours, key=cv2.contourArea)
    convex_hull = cv2.convexHull(all_points)
    contour_image = image.copy()
    hull_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 3)
    cv2.polylines(hull_image, [convex_hull], True, (255, 255, 0), 4)

    prepared["contour_overlay"] = contour_image
    prepared["hull_overlay"] = hull_image

    return TopSegmentationResult(
        status="segmented",
        message="TOP-view strawberry canopy segmentation completed.",
        mask=plant_mask,
        contours=contours,
        largest_contour=largest_contour,
        convex_hull=convex_hull,
        contour_image=contour_image,
        hull_image=hull_image,
        mask_area_pixels=int(cv2.countNonZero(plant_mask)),
        hull_area_pixels=float(cv2.contourArea(convex_hull)),
        contour_count=len(contours),
        bounding_box=tuple(int(value) for value in cv2.boundingRect(all_points)),
        debug_images=prepared,
        leaf_mask=leaf_mask,
    )


def segment_front_view_plant(
    image: np.ndarray,
    *,
    min_component_area_ratio: float = 0.00035,
) -> FrontSegmentationResult:
    """Segment the strawberry plant from one FRONT-view RGB image."""

    prepared = _prepare_segmentation(image, min_component_area_ratio=min_component_area_ratio)
    prepared["filtered_mask_before_front_rules"] = prepared["filtered_mask"].copy()
    filtered_mask, front_filter_debug = _filter_front_view_components(prepared["filtered_mask"])
    prepared.update(front_filter_debug)
    prepared["filtered_mask"] = filtered_mask
    contours = _find_external_contours(filtered_mask)

    if not contours:
        empty_overlay = image.copy()
        return FrontSegmentationResult(
            status="no_foreground",
            message="No strawberry canopy was detected in the FRONT image.",
            mask=filtered_mask,
            contours=[],
            largest_contour=None,
            contour_image=empty_overlay,
            bounding_box_image=empty_overlay.copy(),
            mask_area_pixels=0,
            contour_count=0,
            bounding_box=None,
            debug_images=prepared,
        )

    merged_contour = _merge_contours(contours)
    bounding_box = tuple(int(value) for value in cv2.boundingRect(merged_contour))
    contour_image = image.copy()
    bounding_box_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 3)

    x, y, width, height = bounding_box
    cv2.rectangle(bounding_box_image, (x, y), (x + width, y + height), (255, 255, 0), 4)

    prepared["contour_overlay"] = contour_image
    prepared["bounding_box_overlay"] = bounding_box_image

    return FrontSegmentationResult(
        status="segmented",
        message="FRONT-view strawberry canopy segmentation completed.",
        mask=filtered_mask,
        contours=contours,
        largest_contour=max(contours, key=cv2.contourArea),
        contour_image=contour_image,
        bounding_box_image=bounding_box_image,
        mask_area_pixels=int(cv2.countNonZero(filtered_mask)),
        contour_count=len(contours),
        bounding_box=bounding_box,
        debug_images=prepared,
    )


def _prepare_segmentation(image: np.ndarray, *, min_component_area_ratio: float) -> dict[str, np.ndarray]:
    """Run the shared preprocessing and masking stages."""

    _validate_color_image(image)
    if min_component_area_ratio <= 0:
        raise ValueError("min_component_area_ratio must be greater than 0")

    normalized = normalize_image(image)
    denoised = denoise_image(normalized, method="gaussian", kernel_size=5)
    hsv_mask = _build_hsv_green_mask(denoised)
    green_dominance_mask = _build_green_dominance_mask(denoised)
    combined_mask = cv2.bitwise_and(hsv_mask, green_dominance_mask)
    clean_steps = _clean_mask_steps(combined_mask)
    cleaned_mask = clean_steps["cleaned_mask"]
    filtered_mask = _filter_small_components(cleaned_mask, min_component_area_ratio=min_component_area_ratio)

    return {
        "original": image.copy(),
        "normalized": normalized,
        "denoised": denoised,
        "hsv_green_mask": hsv_mask,
        "green_dominance_mask": green_dominance_mask,
        "combined_mask": combined_mask,
        "morphology_opened": clean_steps["opened_mask"],
        "morphology_closed": clean_steps["closed_mask"],
        "holes_filled_mask": clean_steps["holes_filled_mask"],
        "cleaned_mask": cleaned_mask,
        "filtered_mask": filtered_mask,
    }


def _build_hsv_green_mask(image: np.ndarray) -> np.ndarray:
    """Create a broad green mask in HSV space."""

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 25, 20], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv_image, lower, upper)


def _build_green_dominance_mask(image: np.ndarray) -> np.ndarray:
    """Create a green-dominance mask in BGR space."""

    b_channel = image[:, :, 0].astype(np.float32)
    g_channel = image[:, :, 1].astype(np.float32)
    r_channel = image[:, :, 2].astype(np.float32)

    dominance = (g_channel > 30) & (g_channel > r_channel * 1.03) & (g_channel > b_channel * 1.05)
    return dominance.astype(np.uint8) * 255


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Apply morphology to connect the canopy and suppress small noise."""

    return _clean_mask_steps(mask)["cleaned_mask"]


def _clean_mask_steps(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Run lighter morphology with explicit intermediate outputs."""

    min_dim = min(mask.shape[:2])
    open_size = _make_odd(max(3, int(round(min_dim * 0.003))))
    close_size = _make_odd(max(open_size + 2, int(round(min_dim * 0.0075))))

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    holes_filled = _fill_small_holes(closed)
    return {
        "opened_mask": opened,
        "closed_mask": closed,
        "holes_filled_mask": holes_filled,
        "cleaned_mask": holes_filled,
    }


def _filter_small_components(mask: np.ndarray, *, min_component_area_ratio: float) -> np.ndarray:
    """Remove connected components that are too small to be part of the canopy."""

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return mask

    min_area_pixels = max(32, int(mask.shape[0] * mask.shape[1] * min_component_area_ratio))
    filtered_mask = np.zeros_like(mask)

    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area >= min_area_pixels:
            filtered_mask[labels == component_index] = 255

    return filtered_mask


def _filter_front_view_components(mask: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Keep plausible canopy components and suppress side color-card patches."""

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        empty = np.zeros_like(mask)
        return mask, {
            "front_component_kept_mask": mask.copy(),
            "front_component_removed_mask": empty,
        }

    image_height, image_width = mask.shape
    image_area = image_height * image_width
    min_area_pixels = max(48, int(image_area * 0.00035))
    filtered_mask = np.zeros_like(mask)
    removed_mask = np.zeros_like(mask)

    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        center_x = x + width / 2.0

        component_mask = labels == component_index

        if area < min_area_pixels:
            removed_mask[component_mask] = 255
            continue
        if height < max(12, int(image_height * 0.05)):
            removed_mask[component_mask] = 255
            continue
        if center_x > image_width * 0.88 and width < image_width * 0.12:
            removed_mask[component_mask] = 255
            continue

        filtered_mask[component_mask] = 255

    return filtered_mask, {
        "front_component_kept_mask": filtered_mask.copy(),
        "front_component_removed_mask": removed_mask,
    }


def _filter_top_view_components(mask: np.ndarray) -> np.ndarray:
    """Suppress right-side compact components that match the TOP-view color card."""

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return mask

    image_height, image_width = mask.shape
    filtered_mask = mask.copy()

    for component_index in range(1, component_count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        center_x = float(centroids[component_index, 0])
        bbox_area = max(width * height, 1)
        fill_ratio = float(area) / float(bbox_area)

        contour_mask = np.zeros_like(mask)
        contour_mask[labels == component_index] = 255
        contours = _find_external_contours(contour_mask)
        solidity = 0.0
        if contours:
            hull = cv2.convexHull(np.vstack(contours))
            hull_area = float(cv2.contourArea(hull))
            if hull_area > 0:
                solidity = float(area) / hull_area

        is_right_side = center_x > image_width * 0.70
        is_compact = width < image_width * 0.22 and height < image_height * 0.30
        is_card_like = is_right_side and is_compact and fill_ratio > 0.12 and solidity > 0.35
        if is_card_like:
            filtered_mask[labels == component_index] = 0

    return filtered_mask


def _augment_top_mask_with_reproductive_organs(
    image: np.ndarray,
    leaf_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Add white petals, yellow centers, and fruit-like organs near the TOP canopy back into the display mask."""

    allowed_region = _build_top_reproductive_allowed_region(leaf_mask)
    candidate_mask = _build_top_reproductive_candidate_mask(image)
    reproductive_mask = cv2.bitwise_and(candidate_mask, allowed_region)

    min_dim = min(leaf_mask.shape)
    close_size = _make_odd(max(3, int(round(min_dim * 0.006))))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    reproductive_mask = cv2.morphologyEx(reproductive_mask, cv2.MORPH_CLOSE, close_kernel)
    reproductive_mask = _filter_small_components(reproductive_mask, min_component_area_ratio=0.00002)

    plant_mask = cv2.bitwise_or(leaf_mask, reproductive_mask)
    return plant_mask, {
        "top_reproductive_allowed_region": allowed_region,
        "top_reproductive_candidate_mask": candidate_mask,
        "top_reproductive_mask": reproductive_mask,
    }


def _build_top_reproductive_allowed_region(leaf_mask: np.ndarray) -> np.ndarray:
    """Limit flower/fruit recovery to the plant body so color cards and background highlights stay excluded."""

    allowed_region = np.zeros_like(leaf_mask)
    contours = _find_external_contours(leaf_mask)
    if contours:
        hull = cv2.convexHull(np.vstack(contours))
        cv2.fillConvexPoly(allowed_region, hull, 255)

    min_dim = min(leaf_mask.shape)
    dilate_size = _make_odd(max(9, int(round(min_dim * 0.045))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    dilated_leaf = cv2.dilate(leaf_mask, kernel, iterations=1)
    return cv2.bitwise_or(allowed_region, dilated_leaf)


def _build_top_reproductive_candidate_mask(image: np.ndarray) -> np.ndarray:
    """Find non-green visible organs that should remain in the TOP display mask."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bgr = image.astype(np.int16)
    channel_min = np.min(bgr, axis=2)
    channel_max = np.max(bgr, axis=2)
    channel_spread = channel_max - channel_min

    white_petals = (
        (hsv[:, :, 1] <= 80)
        & (hsv[:, :, 2] >= 145)
        & (lab[:, :, 0] >= 145)
        & (channel_min >= 120)
        & (channel_spread <= 85)
    )
    yellow_centers = (
        (hsv[:, :, 0] >= 10)
        & (hsv[:, :, 0] <= 45)
        & (hsv[:, :, 1] >= 45)
        & (hsv[:, :, 2] >= 95)
    )
    pale_buds = (
        (hsv[:, :, 0] >= 35)
        & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] >= 20)
        & (hsv[:, :, 1] <= 150)
        & (hsv[:, :, 2] >= 120)
    )
    red_fruits = (
        (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 165)))
        & (hsv[:, :, 1] >= 70)
        & (hsv[:, :, 2] >= 60)
    )
    return (white_petals | yellow_centers | pale_buds | red_fruits).astype(np.uint8) * 255


def _remove_top_attached_pot_band(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove shallow, wide foreground bands near the top that likely belong to the pot rim."""

    image_height, image_width = mask.shape
    if cv2.countNonZero(mask) == 0:
        empty = np.zeros_like(mask)
        return mask, empty, empty

    top_margin = max(24, int(round(image_height * 0.18)))
    shallow_depth = max(18, int(round(image_height * 0.12)))
    min_gap_to_body = max(6, int(round(image_height * 0.012)))
    min_run_width = max(32, int(round(image_width * 0.07)))

    candidate_mask = np.zeros_like(mask)
    for x in range(image_width):
        ys = np.flatnonzero(mask[:, x] > 0)
        if ys.size == 0:
            continue

        top_y = int(ys[0])
        if top_y > top_margin:
            continue

        end_y = top_y
        while end_y + 1 < image_height and mask[end_y + 1, x] > 0:
            end_y += 1

        top_run_depth = end_y - top_y + 1
        if top_run_depth > shallow_depth:
            continue

        remaining_pixels = ys[ys > end_y + min_gap_to_body]
        if remaining_pixels.size == 0:
            continue

        candidate_mask[top_y : end_y + 1, x] = 255

    if cv2.countNonZero(candidate_mask) == 0:
        empty = np.zeros_like(mask)
        return mask, empty, empty

    horizontal_close = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (_make_odd(max(5, int(round(image_width * 0.02)))), 1),
    )
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, horizontal_close)

    column_hits = np.any(candidate_mask > 0, axis=0)
    wide_candidate = np.zeros_like(candidate_mask)
    run_start: int | None = None
    for x in range(image_width + 1):
        is_hit = x < image_width and bool(column_hits[x])
        if is_hit and run_start is None:
            run_start = x
            continue
        if is_hit or run_start is None:
            continue

        run_width = x - run_start
        if run_width >= min_run_width:
            wide_candidate[:, run_start:x] = candidate_mask[:, run_start:x]
        run_start = None

    removed_mask = cv2.bitwise_and(mask, wide_candidate)
    filtered_mask = cv2.subtract(mask, removed_mask)
    return filtered_mask, wide_candidate, removed_mask


def _fill_small_holes(mask: np.ndarray) -> np.ndarray:
    """Fill only small interior holes to avoid over-smoothing the canopy boundary."""

    height, width = mask.shape
    image_area = height * width
    max_hole_area = max(64, int(round(image_area * 0.0002)))

    inverted = cv2.bitwise_not(mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    if component_count <= 1:
        return mask

    filled_mask = mask.copy()
    for component_index in range(1, component_count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width_px = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height_px = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])

        touches_border = (
            x == 0
            or y == 0
            or x + width_px >= width
            or y + height_px >= height
        )
        if touches_border or area > max_hole_area:
            continue

        filled_mask[labels == component_index] = 255

    return filled_mask


def _find_external_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Return external contours for the binary plant mask."""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def _merge_contours(contours: list[np.ndarray]) -> np.ndarray:
    """Merge contours into a single point set for bounding box extraction."""

    if not contours:
        raise ValueError("contours must not be empty")
    return np.vstack(contours)


def _make_odd(value: int) -> int:
    """Return the next odd integer greater than or equal to value."""

    return value if value % 2 == 1 else value + 1


def _validate_color_image(image: np.ndarray) -> None:
    """Validate that the input is a non-empty 3-channel BGR image."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a 3-channel BGR image")
