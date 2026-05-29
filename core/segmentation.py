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
    profile: str = "default",
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
    if profile == "cotton":
        filtered_mask, cotton_debug = _filter_cotton_top_view_mask(
            prepared["denoised"],
            prepared["original"],
            filtered_mask,
        )
        prepared.update(cotton_debug)
    leaf_mask = filtered_mask
    if profile == "cotton":
        plant_mask = leaf_mask.copy()
        reproductive_debug = _empty_top_reproductive_debug(leaf_mask)
    else:
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
    prepared["filtered_mask_before_front_augmentation"] = prepared["filtered_mask"].copy()
    augmented_mask, front_augment_debug = _augment_front_mask_with_weak_green_regions(
        prepared["filtered_mask"],
        hsv_green_mask=prepared["hsv_green_mask"],
        green_dominance_mask=prepared["green_dominance_mask"],
    )
    prepared.update(front_augment_debug)
    plant_mask, front_reproductive_debug = _augment_front_mask_with_reproductive_organs(
        prepared["denoised"],
        augmented_mask,
    )
    prepared.update(front_reproductive_debug)
    prepared["filtered_mask_before_front_rules"] = plant_mask.copy()
    filtered_mask, front_filter_debug = _filter_front_view_components(
        plant_mask,
        image=prepared["denoised"],
        leaf_reference_mask=prepared["filtered_mask_before_front_augmentation"],
    )
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


def _augment_front_mask_with_weak_green_regions(
    strong_mask: np.ndarray,
    *,
    hsv_green_mask: np.ndarray,
    green_dominance_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Recover shadowed FRONT leaf regions that satisfy HSV green cues near the main canopy."""

    weak_green_mask = cv2.bitwise_and(hsv_green_mask, cv2.bitwise_not(green_dominance_mask))
    weak_green_mask = cv2.bitwise_and(weak_green_mask, cv2.bitwise_not(strong_mask))
    weak_green_mask = _filter_small_components(weak_green_mask, min_component_area_ratio=0.00008)

    empty = np.zeros_like(strong_mask)
    if cv2.countNonZero(strong_mask) == 0 or cv2.countNonZero(weak_green_mask) == 0:
        return strong_mask, {
            "front_weak_green_mask": weak_green_mask,
            "front_weak_green_seed_mask": empty,
            "front_recovered_weak_green_mask": empty,
        }

    image_height, image_width = strong_mask.shape
    seed_size = _make_odd(max(5, int(round(min(image_height, image_width) * 0.018))))
    seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seed_size, seed_size))
    strong_seed_mask = cv2.dilate(strong_mask, seed_kernel, iterations=1)

    union_mask = cv2.bitwise_or(strong_seed_mask, weak_green_mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(union_mask, connectivity=8)
    min_recovered_area = max(80, int(image_height * image_width * 0.00008))
    min_recovered_height = max(10, int(image_height * 0.03))
    recovered_mask = np.zeros_like(strong_mask)

    for component_index in range(1, component_count):
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        if height < min_recovered_height:
            continue

        component_mask = (labels == component_index).astype(np.uint8) * 255
        weak_component = cv2.bitwise_and(component_mask, weak_green_mask)
        recovered_area = int(cv2.countNonZero(weak_component))
        if recovered_area < min_recovered_area:
            continue
        if cv2.countNonZero(cv2.bitwise_and(component_mask, strong_seed_mask)) == 0:
            continue

        recovered_mask = cv2.bitwise_or(recovered_mask, weak_component)

    augmented_mask = cv2.bitwise_or(strong_mask, recovered_mask)
    return augmented_mask, {
        "front_weak_green_mask": weak_green_mask,
        "front_weak_green_seed_mask": strong_seed_mask,
        "front_recovered_weak_green_mask": recovered_mask,
    }


def _augment_front_mask_with_reproductive_organs(
    image: np.ndarray,
    plant_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Recover non-green flower/fruit pixels enclosed by or adjacent to the FRONT canopy."""

    allowed_region = _build_front_reproductive_allowed_region(plant_mask)
    candidate_mask = _build_top_reproductive_candidate_mask(image)
    reproductive_mask = cv2.bitwise_and(candidate_mask, allowed_region)
    reproductive_mask = _filter_small_components(reproductive_mask, min_component_area_ratio=0.00002)

    augmented_mask = cv2.bitwise_or(plant_mask, reproductive_mask)
    augmented_mask = _fill_small_holes(augmented_mask)
    return augmented_mask, {
        "front_reproductive_allowed_region": allowed_region,
        "front_reproductive_candidate_mask": candidate_mask,
        "front_reproductive_mask": reproductive_mask,
    }


def _build_front_reproductive_allowed_region(plant_mask: np.ndarray) -> np.ndarray:
    """Allow flower/fruit recovery inside each plant component without opening the whole frame."""

    allowed_region = np.zeros_like(plant_mask)
    contours = _find_external_contours(plant_mask)
    min_area = max(64, int(plant_mask.shape[0] * plant_mask.shape[1] * 0.0002))
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        hull = cv2.convexHull(contour)
        cv2.fillConvexPoly(allowed_region, hull, 255)

    min_dim = min(plant_mask.shape)
    dilate_size = _make_odd(max(7, int(round(min_dim * 0.025))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    dilated_plant = cv2.dilate(plant_mask, kernel, iterations=1)
    return cv2.bitwise_or(allowed_region, dilated_plant)


def _filter_front_view_components(
    mask: np.ndarray,
    *,
    image: np.ndarray | None = None,
    leaf_reference_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Keep plausible canopy components and suppress side color-card patches."""

    pot_filtered_mask, pot_debug = _remove_front_pot_like_pixels(
        image,
        mask,
        leaf_reference_mask=leaf_reference_mask,
    )
    mask = pot_filtered_mask

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        empty = np.zeros_like(mask)
        debug_images = {
            "front_component_kept_mask": mask.copy(),
            "front_component_removed_mask": empty,
        }
        debug_images.update(pot_debug)
        return mask, debug_images

    image_height, image_width = mask.shape
    image_area = image_height * image_width
    min_area_pixels = max(48, int(image_area * 0.00035))
    removed_mask = np.zeros_like(mask)
    candidates: list[tuple[int, int, np.ndarray]] = []

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

        candidates.append((area, component_index, component_mask))

    filtered_mask = np.zeros_like(mask)
    if candidates:
        _largest_area, largest_component_index, largest_component_mask = max(candidates, key=lambda item: item[0])
        largest_x = int(stats[largest_component_index, cv2.CC_STAT_LEFT])
        largest_y = int(stats[largest_component_index, cv2.CC_STAT_TOP])
        largest_width = int(stats[largest_component_index, cv2.CC_STAT_WIDTH])
        largest_height = int(stats[largest_component_index, cv2.CC_STAT_HEIGHT])
        largest_box = (largest_x, largest_y, largest_width, largest_height)
        filtered_mask[largest_component_mask] = 255

        for _area, component_index, component_mask in candidates:
            if component_index == largest_component_index:
                continue

            if _is_front_satellite_component(stats[component_index], largest_box, image_width, image_height):
                filtered_mask[component_mask] = 255
            else:
                removed_mask[component_mask] = 255

    debug_images = {
        "front_component_kept_mask": filtered_mask.copy(),
        "front_component_removed_mask": removed_mask,
    }
    debug_images.update(pot_debug)
    return filtered_mask, debug_images


def _remove_front_pot_like_pixels(
    image: np.ndarray | None,
    mask: np.ndarray,
    *,
    leaf_reference_mask: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Remove bottom teal/gray-green pixels whose color is far from the leaf reference."""

    empty = np.zeros_like(mask)
    if image is None or leaf_reference_mask is None or cv2.countNonZero(mask) == 0:
        return mask, {"front_pot_like_removed_mask": empty}

    reference_pixels = _sample_front_leaf_reference_pixels(image, leaf_reference_mask)
    if reference_pixels.size == 0:
        return mask, {"front_pot_like_removed_mask": empty}

    reference_hsv = cv2.cvtColor(reference_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    reference_lab = cv2.cvtColor(reference_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    ref_hsv_median = np.median(reference_hsv, axis=0).astype(np.float32)
    ref_lab_median = np.median(reference_lab, axis=0).astype(np.float32)

    reference_bgr = reference_pixels.astype(np.float32)
    reference_green_excess = reference_bgr[:, 1] - np.maximum(reference_bgr[:, 0], reference_bgr[:, 2])
    ref_green_excess = float(np.median(reference_green_excess))

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bgr = image.astype(np.float32)
    green_excess = bgr[:, :, 1] - np.maximum(bgr[:, :, 0], bgr[:, :, 2])
    lab_delta = np.linalg.norm(lab_image - ref_lab_median.reshape(1, 1, 3), axis=2)

    y_indices = np.indices(mask.shape)[0]
    foreground_rows = np.flatnonzero(np.any(mask > 0, axis=1))
    if foreground_rows.size == 0:
        return mask, {"front_pot_like_removed_mask": empty}

    y_min = int(foreground_rows[0])
    y_max = int(foreground_rows[-1])
    lower_canopy = y_indices >= y_min + int(round((y_max - y_min + 1) * 0.48))
    greenish_hue = (hsv_image[:, :, 0] >= 32) & (hsv_image[:, :, 0] <= 105)
    low_leaf_saturation = hsv_image[:, :, 1] < max(95.0, float(ref_hsv_median[1]) * 0.72)
    weak_green_excess = green_excess < max(10.0, ref_green_excess * 0.55)
    color_far_from_leaf = lab_delta > 24.0

    pot_like = (
        (mask > 0)
        & lower_canopy
        & greenish_hue
        & color_far_from_leaf
        & (low_leaf_saturation | weak_green_excess)
    ).astype(np.uint8) * 255

    open_size = _make_odd(max(3, int(round(min(mask.shape) * 0.004))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    pot_like = cv2.morphologyEx(pot_like, cv2.MORPH_OPEN, kernel)
    pot_like = _keep_large_bottom_pot_components(pot_like, mask)
    filtered_mask = cv2.subtract(mask, pot_like)
    return filtered_mask, {"front_pot_like_removed_mask": pot_like}


def _keep_large_bottom_pot_components(pot_like_mask: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    """Keep only pot-color regions with the broad bottom geometry expected from a pot."""

    foreground_rows = np.flatnonzero(np.any(foreground_mask > 0, axis=1))
    if foreground_rows.size == 0:
        return np.zeros_like(pot_like_mask)

    image_height, image_width = pot_like_mask.shape
    foreground_top = int(foreground_rows[0])
    foreground_bottom = int(foreground_rows[-1])
    lower_start = foreground_top + int(round((foreground_bottom - foreground_top + 1) * 0.50))
    min_area = max(128, int(image_height * image_width * 0.001))
    min_width = max(24, int(image_width * 0.12))
    min_height = max(8, int(image_height * 0.025))

    kept_mask = np.zeros_like(pot_like_mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(pot_like_mask, connectivity=8)
    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        _ = x
        if area < min_area or width < min_width or height < min_height:
            continue
        if y < lower_start:
            continue

        kept_mask[labels == component_index] = 255

    return kept_mask


def _sample_front_leaf_reference_pixels(image: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    """Sample likely leaf pixels from the upper part of the largest green component."""

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(reference_mask, connectivity=8)
    if component_count <= 1:
        return np.empty((0, 3), dtype=image.dtype)

    largest_index = max(
        range(1, component_count),
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
    )
    x = int(stats[largest_index, cv2.CC_STAT_LEFT])
    y = int(stats[largest_index, cv2.CC_STAT_TOP])
    width = int(stats[largest_index, cv2.CC_STAT_WIDTH])
    height = int(stats[largest_index, cv2.CC_STAT_HEIGHT])
    upper_cutoff = y + max(1, int(round(height * 0.72)))

    component_mask = labels == largest_index
    y_indices = np.indices(reference_mask.shape)[0]
    sample_mask = component_mask & (y_indices < upper_cutoff)
    pixels = image[sample_mask]
    if pixels.size == 0:
        pixels = image[component_mask]

    _ = x, width
    return pixels


def _is_front_satellite_component(
    stats_row: np.ndarray,
    largest_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> bool:
    """Keep flower/fruit satellites near the main canopy while dropping distant pot/card regions."""

    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    width = int(stats_row[cv2.CC_STAT_WIDTH])
    height = int(stats_row[cv2.CC_STAT_HEIGHT])
    area = int(stats_row[cv2.CC_STAT_AREA])

    largest_x, largest_y, largest_width, largest_height = largest_box
    largest_right = largest_x + largest_width
    largest_bottom = largest_y + largest_height
    component_center_y = y + height / 2.0
    horizontal_overlap = min(x + width, largest_right) - max(x, largest_x)
    near_horizontally = horizontal_overlap > -image_width * 0.08
    near_vertically = y < largest_bottom + image_height * 0.08 and y + height > largest_y - image_height * 0.22
    small_satellite = area < max(1, largest_width * largest_height * 0.08)

    return small_satellite and near_horizontally and near_vertically and component_center_y < image_height * 0.72


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


def _filter_cotton_top_view_mask(
    image: np.ndarray,
    soil_color_image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Keep cotton TOP leaves conservatively and remove only distant isolated components."""

    empty = np.zeros_like(mask)
    if cv2.countNonZero(mask) == 0:
        return mask, {
            "cotton_base_leaf_mask": empty,
            "cotton_strong_leaf_seed_mask": empty,
            "cotton_soil_removed_mask": empty,
            "cotton_internal_hole_mask": empty,
            "cotton_far_component_removed_mask": empty,
            "cotton_component_kept_mask": empty,
            "cotton_soil_filtered_mask": empty,
            "cotton_gap_closed_mask": empty,
            "cotton_hole_filled_mask": empty,
            "cotton_weak_green_mask": empty,
            "cotton_recovered_weak_green_mask": empty,
            "cotton_weak_green_search_region": empty,
        }

    augmented_mask, weak_debug = _augment_cotton_top_mask_with_nearby_weak_green_regions(image, mask)
    component_mask, far_removed_mask = _filter_cotton_top_components(augmented_mask)
    filtered_mask, soil_removed_mask = _remove_cotton_top_soil_like_pixels(soil_color_image, component_mask)
    gap_closed_mask = _close_cotton_top_leaf_edge_gaps(filtered_mask)
    hole_filled_mask, internal_hole_mask = _fill_cotton_top_internal_holes(gap_closed_mask)
    debug_images = {
        "cotton_base_leaf_mask": mask.copy(),
        # Kept for backward-compatible debug displays; cotton no longer uses a strong-green seed.
        "cotton_strong_leaf_seed_mask": mask.copy(),
        "cotton_soil_removed_mask": soil_removed_mask,
        "cotton_internal_hole_mask": internal_hole_mask,
        "cotton_far_component_removed_mask": far_removed_mask,
        "cotton_component_kept_mask": component_mask.copy(),
        "cotton_soil_filtered_mask": filtered_mask.copy(),
        "cotton_gap_closed_mask": gap_closed_mask.copy(),
        "cotton_hole_filled_mask": hole_filled_mask.copy(),
    }
    debug_images.update(weak_debug)
    return hole_filled_mask, debug_images


def _remove_cotton_top_soil_like_pixels(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove yellow-brown cotton pot soil while protecting gray-green leaf pixels."""

    empty = np.zeros_like(mask)
    if cv2.countNonZero(mask) == 0:
        return mask, empty

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    bgr = image.astype(np.float32)
    b_channel = bgr[:, :, 0]
    g_channel = bgr[:, :, 1]
    r_channel = bgr[:, :, 2]
    green_excess = g_channel - np.maximum(b_channel, r_channel)
    green_minus_red = g_channel - r_channel
    green_minus_blue = g_channel - b_channel
    red_minus_blue = r_channel - b_channel

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_mean = cv2.blur(gray, (11, 11))
    gray_mean_sq = cv2.blur(gray * gray, (11, 11))
    local_std = np.sqrt(np.maximum(gray_mean_sq - gray_mean * gray_mean, 0.0))

    leaf_protected = (
        (green_excess >= 14.0)
        | ((hue >= 45.0) & (hue <= 105.0) & (saturation >= 18.0) & (green_excess >= 6.0))
        | ((green_minus_red >= 12.0) & (green_minus_blue >= 10.0))
        | ((hue >= 28.0) & (hue <= 70.0) & (value >= 70.0) & (local_std <= 28.0))
    )
    stem_protected = (
        (hue >= 18.0)
        & (hue <= 58.0)
        & (value >= 85.0)
        & (green_minus_blue >= 12.0)
        & (local_std <= 35.0)
    )
    granular_soil = (saturation >= 85.0) & (green_excess <= 10.0) & (local_std >= 18.0)
    pale_soil = (saturation <= 38.0) & (green_excess <= 8.0) & (green_minus_blue <= 18.0) & (local_std >= 18.0)
    brown_soil = (red_minus_blue >= 42.0) & (green_excess <= 6.0) & (local_std >= 18.0)
    soil_colored = (
        (mask > 0)
        & (hue >= 18.0)
        & (hue <= 55.0)
        & (saturation >= 18.0)
        & (saturation <= 175.0)
        & (value >= 45.0)
        & (granular_soil | pale_soil | brown_soil)
        & (~leaf_protected)
        & (~stem_protected)
    )

    soil_mask = soil_colored.astype(np.uint8) * 255
    open_size = _make_odd(max(3, int(round(min(mask.shape) * 0.004))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_OPEN, kernel)
    filtered_mask = cv2.subtract(mask, soil_mask)
    return filtered_mask, soil_mask


def _close_cotton_top_leaf_edge_gaps(mask: np.ndarray) -> np.ndarray:
    """Close small edge gaps in cotton TOP leaves without changing the outer background."""

    if cv2.countNonZero(mask) == 0:
        return mask

    image_height, image_width = mask.shape
    kernel_size = _make_odd(max(5, int(round(min(image_height, image_width) * 0.012))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return _filter_cotton_top_components(closed)[0]


def _fill_cotton_top_internal_holes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill enclosed black holes inside cotton TOP leaves/components."""

    empty = np.zeros_like(mask)
    if cv2.countNonZero(mask) == 0:
        return mask, empty

    binary_mask = ((mask > 0).astype(np.uint8)) * 255
    height, width = binary_mask.shape
    inverted = cv2.bitwise_not(binary_mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    filled_mask = binary_mask.copy()
    hole_mask = np.zeros_like(binary_mask)
    for component_index in range(1, component_count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width_px = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height_px = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        touches_border = (
            x == 0
            or y == 0
            or x + width_px >= width
            or y + height_px >= height
        )
        if touches_border:
            continue

        component_pixels = labels == component_index
        filled_mask[component_pixels] = 255
        hole_mask[component_pixels] = 255

    return filled_mask, hole_mask


def _augment_cotton_top_mask_with_nearby_weak_green_regions(
    image: np.ndarray,
    strong_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Recover only weak green cotton TOP pixels adjacent to the existing plant mask."""

    empty = np.zeros_like(strong_mask)
    hsv_green_mask = _build_hsv_green_mask(image)
    green_dominance_mask = _build_green_dominance_mask(image)
    weak_green_mask = cv2.bitwise_and(hsv_green_mask, cv2.bitwise_not(green_dominance_mask))
    weak_green_mask = cv2.bitwise_and(weak_green_mask, cv2.bitwise_not(strong_mask))

    if cv2.countNonZero(strong_mask) == 0 or cv2.countNonZero(weak_green_mask) == 0:
        return strong_mask, {
            "cotton_weak_green_mask": weak_green_mask,
            "cotton_recovered_weak_green_mask": empty,
            "cotton_weak_green_search_region": empty,
        }

    image_height, image_width = strong_mask.shape
    search_size = _make_odd(max(7, int(round(min(image_height, image_width) * 0.035))))
    search_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (search_size, search_size))
    search_region = cv2.dilate(strong_mask, search_kernel, iterations=1)
    nearby_weak_green = cv2.bitwise_and(weak_green_mask, search_region)
    nearby_weak_green = _filter_small_components(nearby_weak_green, min_component_area_ratio=0.00006)

    augmented_mask = cv2.bitwise_or(strong_mask, nearby_weak_green)
    return augmented_mask, {
        "cotton_weak_green_mask": weak_green_mask,
        "cotton_recovered_weak_green_mask": nearby_weak_green,
        "cotton_weak_green_search_region": search_region,
    }


def _filter_cotton_top_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the main cotton TOP plant body plus nearby satellite leaf components."""

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    empty = np.zeros_like(mask)
    if component_count <= 1:
        return mask, empty

    image_height, image_width = mask.shape
    image_area = image_height * image_width
    min_area_pixels = max(48, int(image_area * 0.00025))
    candidate_indices: list[int] = []
    far_removed_mask = np.zeros_like(mask)

    for component_index in range(1, component_count):
        component_mask = labels == component_index
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < min_area_pixels:
            far_removed_mask[component_mask] = 255
            continue
        candidate_indices.append(component_index)

    if not candidate_indices:
        return empty, mask.copy()

    main_component_index = _select_cotton_top_main_component(
        stats,
        centroids,
        candidate_indices,
        image_width,
        image_height,
    )
    main_box = (
        int(stats[main_component_index, cv2.CC_STAT_LEFT]),
        int(stats[main_component_index, cv2.CC_STAT_TOP]),
        int(stats[main_component_index, cv2.CC_STAT_WIDTH]),
        int(stats[main_component_index, cv2.CC_STAT_HEIGHT]),
    )
    main_area = int(stats[main_component_index, cv2.CC_STAT_AREA])

    kept_mask = np.zeros_like(mask)
    for component_index in candidate_indices:
        component_mask = labels == component_index
        if component_index == main_component_index or _is_cotton_top_related_component(
            stats[component_index],
            main_box,
            main_area,
            image_width,
            image_height,
        ):
            kept_mask[component_mask] = 255
        else:
            far_removed_mask[component_mask] = 255

    return kept_mask, far_removed_mask


def _select_cotton_top_main_component(
    stats: np.ndarray,
    centroids: np.ndarray,
    candidate_indices: list[int],
    image_width: int,
    image_height: int,
) -> int:
    """Pick the cotton TOP component most likely to be the plant body."""

    interior_indices = [
        component_index
        for component_index in candidate_indices
        if not _component_touches_image_border(stats[component_index], image_width, image_height)
    ]
    scoring_indices = interior_indices if interior_indices else candidate_indices
    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0
    best_index = scoring_indices[0]
    best_score = float("-inf")

    for component_index in scoring_indices:
        area = float(stats[component_index, cv2.CC_STAT_AREA])
        width = float(stats[component_index, cv2.CC_STAT_WIDTH])
        center_x = float(centroids[component_index, 0])
        center_y = float(centroids[component_index, 1])
        distance_penalty = (
            abs(center_x - image_center_x) / max(image_width, 1)
            + abs(center_y - image_center_y) / max(image_height, 1)
        )
        score = area * (1.0 - min(distance_penalty, 1.0) * 0.35)
        if center_x > image_width * 0.82 and width < image_width * 0.20:
            score *= 0.35

        if score > best_score:
            best_score = score
            best_index = component_index

    return best_index


def _component_touches_image_border(stats_row: np.ndarray, image_width: int, image_height: int) -> bool:
    """Return whether a component touches the frame edge, which is background-like in cotton TOP images."""

    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    width = int(stats_row[cv2.CC_STAT_WIDTH])
    height = int(stats_row[cv2.CC_STAT_HEIGHT])
    return x <= 0 or y <= 0 or x + width >= image_width or y + height >= image_height


def _is_cotton_top_related_component(
    stats_row: np.ndarray,
    main_box: tuple[int, int, int, int],
    main_area: int,
    image_width: int,
    image_height: int,
) -> bool:
    """Return whether a TOP component is spatially related to the cotton plant body."""

    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    width = int(stats_row[cv2.CC_STAT_WIDTH])
    height = int(stats_row[cv2.CC_STAT_HEIGHT])
    area = int(stats_row[cv2.CC_STAT_AREA])
    main_x, main_y, main_width, main_height = main_box
    main_right = main_x + main_width
    main_bottom = main_y + main_height
    horizontal_gap = max(main_x - (x + width), x - main_right, 0)
    vertical_gap = max(main_y - (y + height), y - main_bottom, 0)
    horizontally_near = horizontal_gap <= image_width * 0.12
    vertically_near = vertical_gap <= image_height * 0.12
    if horizontally_near and vertically_near:
        return True
    if _component_touches_image_border(stats_row, image_width, image_height):
        return False

    component_center_x = x + width / 2.0
    component_center_y = y + height / 2.0
    center_distance = (
        abs(component_center_x - (main_x + main_width / 2.0)) / max(image_width, 1)
        + abs(component_center_y - (main_y + main_height / 2.0)) / max(image_height, 1)
    )
    return area >= main_area * 0.20 and center_distance <= 0.34


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


def _empty_top_reproductive_debug(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Return empty TOP reproductive debug masks for profiles that should not recover them."""

    empty = np.zeros_like(mask)
    return {
        "top_reproductive_allowed_region": empty,
        "top_reproductive_candidate_mask": empty,
        "top_reproductive_mask": empty,
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
