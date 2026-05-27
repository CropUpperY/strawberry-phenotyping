"""TOP-view flower and fruit detection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class OrganInstance:
    """One counted organ instance extracted from a labeled mask."""

    label_id: int
    area_pixels: int
    centroid_xy: tuple[float, float]
    bounding_box: tuple[int, int, int, int] | None


@dataclass(slots=True)
class TopFlowerDetectionResult:
    """Flower-detection payload for one TOP image."""

    status: str
    message: str
    count: int
    mask: np.ndarray
    labeled_mask: np.ndarray
    instances: list[OrganInstance]
    overlay_image: np.ndarray
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class TopFruitDetectionResult:
    """Fruit-detection payload for one TOP image."""

    status: str
    message: str
    count: int
    mask: np.ndarray
    labeled_mask: np.ndarray
    instances: list[OrganInstance]
    overlay_image: np.ndarray
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


DetectionResultT = TypeVar("DetectionResultT", TopFlowerDetectionResult, TopFruitDetectionResult)


@dataclass(frozen=True, slots=True)
class _FlowerSeed:
    label_id: int
    area_pixels: int
    centroid_xy: tuple[float, float]
    binary_mask: np.ndarray
    search_radius: int


def detect_top_flowers(image: np.ndarray, canopy_mask: np.ndarray) -> TopFlowerDetectionResult:
    """Detect visible flowers using yellow-center anchors plus white-petal support."""

    validated_image, validated_mask = _validate_inputs(image, canopy_mask)
    canopy_source = cv2.bitwise_and(validated_image, validated_image, mask=validated_mask)
    yellow_center_mask = _extract_flower_center_mask(validated_image, validated_mask)
    white_petal_raw, white_petal_cleaned = _extract_flower_petal_masks(validated_image, validated_mask)
    flower_seeds = _collect_flower_seeds(yellow_center_mask)
    labeled_mask, kept_mask, instances, support_mask, marker_mask = _assemble_flower_instances(
        white_petal_cleaned,
        flower_seeds,
    )
    overlay = _build_overlay(validated_image, labeled_mask)
    debug_images = {
        "canopy_source": canopy_source,
        "yellow_center_mask": yellow_center_mask,
        "white_petal_mask_raw": white_petal_raw,
        "white_petal_mask_cleaned": white_petal_cleaned,
        "flower_seed_overlay": _draw_flower_seed_overlay(validated_image, flower_seeds),
        "flower_merge_result": support_mask,
        "flower_split_markers": marker_mask,
        "labeled_mask": np.clip(labeled_mask * 40, 0, 255).astype(np.uint8),
        "overlay": overlay,
    }
    return TopFlowerDetectionResult(
        status="computed",
        message="flower count computed from TOP view visible blooms anchored by yellow centers.",
        count=len(instances),
        mask=kept_mask,
        labeled_mask=labeled_mask,
        instances=instances,
        overlay_image=overlay,
        debug_images=debug_images,
    )


def detect_top_fruits(image: np.ndarray, canopy_mask: np.ndarray) -> TopFruitDetectionResult:
    """Detect visible red fruits inside the TOP canopy mask."""

    validated_image, validated_mask = _validate_inputs(image, canopy_mask)
    hsv = cv2.cvtColor(validated_image, cv2.COLOR_BGR2HSV)
    red_mask_1 = cv2.inRange(
        hsv,
        np.array([0, 80, 50], dtype=np.uint8),
        np.array([12, 255, 255], dtype=np.uint8),
    )
    red_mask_2 = cv2.inRange(
        hsv,
        np.array([165, 80, 50], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    raw_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    return _finalize_detection(
        image=validated_image,
        canopy_mask=validated_mask,
        raw_mask=raw_mask,
        min_area=45,
        label="fruit",
        apply_close=False,
        min_fill_ratio=0.45,
        min_short_side=7,
        max_aspect_ratio=2.5,
        min_circularity=0.45,
        result_type=TopFruitDetectionResult,
    )


def _extract_flower_center_mask(image: np.ndarray, canopy_mask: np.ndarray) -> np.ndarray:
    """Extract yellow flower-center candidates inside the canopy."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    yellow_mask = (
        (hsv[:, :, 0] >= 12)
        & (hsv[:, :, 0] <= 42)
        & (hsv[:, :, 1] >= 70)
        & (hsv[:, :, 2] >= 110)
        & (lab[:, :, 0] >= 120)
        & (lab[:, :, 2] >= 145)
    ).astype(np.uint8) * 255
    yellow_mask = cv2.bitwise_and(yellow_mask, canopy_mask)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    return _filter_small_components(yellow_mask, min_area=12)


def _extract_flower_petal_masks(image: np.ndarray, canopy_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract white petal candidates inside the canopy."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bgr = image.astype(np.int16)
    channel_min = np.min(bgr, axis=2)
    channel_max = np.max(bgr, axis=2)
    channel_spread = channel_max - channel_min
    lab_a_delta = np.abs(lab[:, :, 1].astype(np.int16) - 128)
    lab_b_delta = np.abs(lab[:, :, 2].astype(np.int16) - 128)
    raw_mask = (
        (hsv[:, :, 1] <= 60)
        & (hsv[:, :, 2] >= 185)
        & (lab[:, :, 0] >= 188)
        & (channel_min >= 175)
        & (channel_spread <= 30)
        & (lab_a_delta <= 12)
        & (lab_b_delta <= 16)
    ).astype(np.uint8) * 255
    raw_mask = cv2.bitwise_and(raw_mask, canopy_mask)
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    cleaned = _filter_small_components(cleaned, min_area=24)
    return raw_mask, cleaned


def _collect_flower_seeds(center_mask: np.ndarray) -> list[_FlowerSeed]:
    """Turn filtered yellow-center components into flower anchor seeds."""

    seed_count, labels, stats, centroids = cv2.connectedComponentsWithStats(center_mask)
    seeds: list[_FlowerSeed] = []
    for label_id in range(1, seed_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        component_mask = (labels == label_id).astype(np.uint8) * 255
        cx, cy = centroids[label_id]
        radius = int(np.clip(np.sqrt(area / float(np.pi)) * 5.5, 14, 34))
        seeds.append(
            _FlowerSeed(
                label_id=label_id,
                area_pixels=area,
                centroid_xy=(float(cx), float(cy)),
                binary_mask=component_mask,
                search_radius=radius,
            )
        )
    return seeds


def _assemble_flower_instances(
    petal_mask: np.ndarray,
    flower_seeds: list[_FlowerSeed],
) -> tuple[np.ndarray, np.ndarray, list[OrganInstance], np.ndarray, np.ndarray]:
    """Build flower instances from yellow centers and nearby white petals."""

    if not flower_seeds:
        empty_labels = np.zeros(petal_mask.shape, dtype=np.int32)
        empty_mask = np.zeros_like(petal_mask)
        return empty_labels, empty_mask, [], empty_mask.copy(), empty_mask.copy()

    petal_binary = (petal_mask > 0).astype(np.uint8) * 255
    component_count, component_labels = cv2.connectedComponents(petal_binary)
    candidate_regions: list[np.ndarray] = []
    marker_mask = np.zeros_like(petal_mask)
    valid_seeds: list[_FlowerSeed] = []

    for next_marker, seed in enumerate(flower_seeds, start=1):
        neighborhood = _build_disk_mask(petal_mask.shape, seed.centroid_xy, seed.search_radius)
        local_petals = cv2.bitwise_and(petal_binary, neighborhood)
        if cv2.countNonZero(local_petals) == 0:
            continue

        supporting_mask = np.zeros_like(petal_binary)
        supporting_components: list[int] = []
        for component_id in range(1, component_count):
            component_mask = (component_labels == component_id).astype(np.uint8) * 255
            if cv2.countNonZero(cv2.bitwise_and(component_mask, local_petals)) == 0:
                continue
            supporting_mask = cv2.bitwise_or(supporting_mask, component_mask)
            supporting_components.append(component_id)

        petal_area = int(cv2.countNonZero(supporting_mask))
        if petal_area < max(24, int(seed.area_pixels * 1.5)):
            continue
        if _occupied_angle_bins(supporting_mask, seed.centroid_xy, bins=8) < 2:
            continue

        marker_value = min(255, next_marker * 40)
        marker_mask[seed.binary_mask > 0] = np.uint8(marker_value)
        candidate_regions.append(cv2.bitwise_or(supporting_mask, seed.binary_mask))
        valid_seeds.append(seed)

    if not valid_seeds:
        empty_labels = np.zeros(petal_mask.shape, dtype=np.int32)
        empty_mask = np.zeros_like(petal_mask)
        return empty_labels, empty_mask, [], empty_mask.copy(), marker_mask

    labeled_mask, support_union = _assign_flower_pixels_to_seeds(candidate_regions, valid_seeds, petal_mask.shape)
    instances = _collect_instances(labeled_mask, min_area=36)
    kept_mask = (labeled_mask > 0).astype(np.uint8) * 255
    return labeled_mask, kept_mask, instances, support_union, marker_mask


def _build_disk_mask(shape: tuple[int, int], centroid_xy: tuple[float, float], radius: int) -> np.ndarray:
    """Build a filled disk centered on one candidate flower center."""

    disk = np.zeros(shape, dtype=np.uint8)
    center = (int(round(centroid_xy[0])), int(round(centroid_xy[1])))
    cv2.circle(disk, center, int(radius), 255, -1)
    return disk


def _occupied_angle_bins(mask: np.ndarray, centroid_xy: tuple[float, float], *, bins: int) -> int:
    """Count how many angular sectors around one center are occupied by petals."""

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0
    angles = np.arctan2(ys.astype(np.float32) - centroid_xy[1], xs.astype(np.float32) - centroid_xy[0])
    normalized = (angles + np.pi) / (2.0 * np.pi)
    occupied = np.unique(np.floor(normalized * bins).astype(np.int32))
    return int(occupied.size)


def _draw_flower_seed_overlay(image: np.ndarray, flower_seeds: list[_FlowerSeed]) -> np.ndarray:
    """Render yellow-center anchors on top of the source image."""

    overlay = image.copy()
    for seed in flower_seeds:
        center = (int(round(seed.centroid_xy[0])), int(round(seed.centroid_xy[1])))
        cv2.circle(overlay, center, 5, (0, 255, 255), 2)
        cv2.circle(overlay, center, seed.search_radius, (0, 180, 255), 1)
    return overlay


def _assign_flower_pixels_to_seeds(
    candidate_regions: list[np.ndarray],
    valid_seeds: list[_FlowerSeed],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Assign flower support pixels to the nearest eligible yellow center."""

    labeled_mask = np.zeros(shape, dtype=np.int32)
    support_union = np.zeros(shape, dtype=np.uint8)
    if not candidate_regions or not valid_seeds:
        return labeled_mask, support_union

    region_stack = np.stack([region > 0 for region in candidate_regions], axis=0)
    support_union[np.any(region_stack, axis=0)] = 255
    ys, xs = np.where(support_union > 0)

    for y, x in zip(ys, xs, strict=False):
        eligible = np.where(region_stack[:, y, x])[0]
        if eligible.size == 0:
            continue

        if eligible.size == 1:
            labeled_mask[y, x] = int(eligible[0]) + 1
            continue

        nearest_index = min(
            eligible.tolist(),
            key=lambda index: (
                (valid_seeds[index].centroid_xy[0] - x) ** 2
                + (valid_seeds[index].centroid_xy[1] - y) ** 2
            ),
        )
        labeled_mask[y, x] = int(nearest_index) + 1

    return labeled_mask, support_union


def _filter_small_components(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    """Remove connected components smaller than the given area."""

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    filtered = np.zeros_like(mask)
    for label_id in range(1, component_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        filtered[labels == label_id] = 255
    return filtered


def _validate_inputs(image: np.ndarray, canopy_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate shared organ-detection inputs."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a 3-channel BGR image")
    if not isinstance(canopy_mask, np.ndarray):
        raise TypeError("canopy_mask must be a numpy.ndarray")
    if canopy_mask.ndim != 2:
        raise ValueError("canopy_mask must be a single-channel mask")
    if image.shape[:2] != canopy_mask.shape:
        raise ValueError("image and canopy_mask must have matching height and width")

    normalized_mask = (canopy_mask > 0).astype(np.uint8) * 255
    return image, normalized_mask


def _finalize_detection(
    *,
    image: np.ndarray,
    canopy_mask: np.ndarray,
    raw_mask: np.ndarray,
    min_area: int,
    label: str,
    apply_close: bool,
    min_fill_ratio: float = 0.0,
    min_short_side: int = 0,
    max_aspect_ratio: float | None = None,
    min_circularity: float = 0.0,
    result_type: type[DetectionResultT],
) -> DetectionResultT:
    """Apply shared cleanup, splitting, instance extraction, and debug rendering."""

    canopy_source = cv2.bitwise_and(image, image, mask=canopy_mask)
    canopy_only = cv2.bitwise_and(raw_mask, canopy_mask)
    cleaned = canopy_only.copy()
    if apply_close:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    cleaned = (cleaned > 0).astype(np.uint8) * 255

    split_labels, split_debug = _split_touching_instances(image, cleaned)
    instances = _collect_instances(
        split_labels,
        min_area=min_area,
        min_fill_ratio=min_fill_ratio,
        min_short_side=min_short_side,
        max_aspect_ratio=max_aspect_ratio,
        min_circularity=min_circularity,
    )

    kept_mask = np.zeros_like(cleaned)
    filtered_labels = np.zeros_like(split_labels, dtype=np.int32)
    public_instances: list[OrganInstance] = []
    for next_id, instance in enumerate(instances, start=1):
        filtered_labels[split_labels == instance.label_id] = next_id
        kept_mask[split_labels == instance.label_id] = 255
        public_instances.append(
            OrganInstance(
                label_id=next_id,
                area_pixels=instance.area_pixels,
                centroid_xy=instance.centroid_xy,
                bounding_box=instance.bounding_box,
            )
        )

    overlay = _build_overlay(image, filtered_labels)
    debug_images = {
        "canopy_source": canopy_source,
        "raw_mask": raw_mask.copy(),
        "canopy_limited_mask": canopy_only,
        "cleaned_mask": kept_mask,
        "distance_map": split_debug["distance_map"],
        "peak_mask": split_debug["peak_mask"],
        "labeled_mask": np.clip(filtered_labels * 40, 0, 255).astype(np.uint8),
        "overlay": overlay,
    }
    return result_type(
        status="computed",
        message=f"{label} count computed from TOP view visible organs.",
        count=len(public_instances),
        mask=kept_mask,
        labeled_mask=filtered_labels,
        instances=public_instances,
        overlay_image=overlay,
        debug_images=debug_images,
    )


def _split_touching_instances(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Split touching binary regions with a distance-transform watershed pass."""

    if cv2.countNonZero(mask) == 0:
        empty_labels = np.zeros(mask.shape, dtype=np.int32)
        empty_image = np.zeros(mask.shape, dtype=np.uint8)
        return empty_labels, {
            "distance_map": empty_image,
            "peak_mask": empty_image,
        }

    component_count, component_labels = cv2.connectedComponents(mask)
    if component_count > 2:
        empty_image = np.zeros(mask.shape, dtype=np.uint8)
        return _normalize_labels(component_labels.astype(np.int32), keep_labels_gt=0), {
            "distance_map": empty_image,
            "peak_mask": empty_image,
        }

    sure_background = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    threshold_value = distance.max() * 0.70 if distance.max() > 0 else 0.0
    _, peaks = cv2.threshold(distance, threshold_value, 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)

    marker_count, markers = cv2.connectedComponents(peaks)
    if marker_count <= 1:
        distance_map = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return _normalize_labels(component_labels.astype(np.int32), keep_labels_gt=0), {
            "distance_map": distance_map,
            "peak_mask": peaks,
        }

    markers = markers + 1
    markers[sure_background == 0] = 0
    watershed_markers = cv2.watershed(image.copy(), markers.astype(np.int32))
    watershed_markers[watershed_markers < 0] = 0
    watershed_markers[mask == 0] = 0
    distance_map = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return _normalize_labels(watershed_markers.astype(np.int32), keep_labels_gt=1), {
        "distance_map": distance_map,
        "peak_mask": peaks,
    }


def _collect_instances(
    labeled_mask: np.ndarray,
    *,
    min_area: int,
    min_fill_ratio: float = 0.0,
    min_short_side: int = 0,
    max_aspect_ratio: float | None = None,
    min_circularity: float = 0.0,
) -> list[OrganInstance]:
    """Collect surviving instances from a label image."""

    instances: list[OrganInstance] = []
    for label_id in sorted(int(value) for value in np.unique(labeled_mask) if value > 0):
        binary = (labeled_mask == label_id).astype(np.uint8)
        area = int(binary.sum())
        if area < min_area:
            continue
        ys, xs = np.where(binary > 0)
        contour_points = np.column_stack([xs, ys]).astype(np.int32)
        x, y, width, height = cv2.boundingRect(contour_points)
        short_side = min(width, height)
        if short_side < min_short_side:
            continue
        bbox_area = width * height
        fill_ratio = area / bbox_area if bbox_area > 0 else 0.0
        if fill_ratio < min_fill_ratio:
            continue
        long_side = max(width, height)
        aspect_ratio = long_side / short_side if short_side > 0 else float("inf")
        if max_aspect_ratio is not None and aspect_ratio > max_aspect_ratio:
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        circularity = (4.0 * float(np.pi) * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        if circularity < min_circularity:
            continue
        instances.append(
            OrganInstance(
                label_id=label_id,
                area_pixels=area,
                centroid_xy=(float(xs.mean()), float(ys.mean())),
                bounding_box=(int(x), int(y), int(width), int(height)),
            )
        )
    return instances


def _build_overlay(image: np.ndarray, labeled_mask: np.ndarray) -> np.ndarray:
    """Draw simple bounding boxes and labels for each detected instance."""

    overlay = image.copy()
    for label_id in sorted(int(value) for value in np.unique(labeled_mask) if value > 0):
        ys, xs = np.where(labeled_mask == label_id)
        if xs.size == 0:
            continue
        contour_points = np.column_stack([xs, ys]).astype(np.int32)
        x, y, width, height = cv2.boundingRect(contour_points)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 255, 0), 2)
        cv2.putText(
            overlay,
            str(label_id),
            (x, max(14, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )
    return overlay


def _normalize_labels(labeled_mask: np.ndarray, *, keep_labels_gt: int) -> np.ndarray:
    """Re-map foreground labels to 1..N while keeping background at 0."""

    normalized = np.zeros_like(labeled_mask, dtype=np.int32)
    next_id = 1
    for label_id in sorted(int(value) for value in np.unique(labeled_mask) if value > keep_labels_gt):
        normalized[labeled_mask == label_id] = next_id
        next_id += 1
    return normalized
