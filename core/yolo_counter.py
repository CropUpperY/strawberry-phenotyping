"""ONNX YOLOv8 TOP-view organ counting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np


CLASS_NAMES: tuple[str, ...] = ("flower", "flower_bud", "fruit")
DEFAULT_MODEL_PATH = Path("models") / "best.onnx"


class _OnnxSession(Protocol):
    def get_inputs(self) -> list[object]:
        ...

    def run(self, output_names: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        ...


SessionFactory = Callable[[Path], _OnnxSession]


@dataclass(frozen=True, slots=True)
class YoloOrganInstance:
    """One YOLO detection kept after confidence filtering and NMS."""

    class_id: int
    class_name: str
    confidence: float
    bounding_box: tuple[int, int, int, int]


@dataclass(slots=True)
class YoloOrganDetectionResult:
    """Combined flower, flower-bud, and fruit detection payload."""

    status: str
    message: str
    counts: dict[str, int]
    instances: list[YoloOrganInstance]
    overlay_image: np.ndarray
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def flower_count(self) -> int:
        return int(self.counts.get("flower", 0))

    @property
    def flower_bud_count(self) -> int:
        return int(self.counts.get("flower_bud", 0))

    @property
    def fruit_count(self) -> int:
        return int(self.counts.get("fruit", 0))


class YoloOrganCounter:
    """Run the packaged YOLOv8 ONNX model and count TOP-view organs."""

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self.model_path = _resolve_model_path(model_path)
        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self._session = self._create_session(session_factory)
        self._input_name = str(getattr(self._session.get_inputs()[0], "name"))

    def detect(self, image: np.ndarray, canopy_mask: np.ndarray | None = None) -> YoloOrganDetectionResult:
        """Detect flower, flower bud, and fruit instances in one BGR TOP image."""

        validated_image = _validate_image(image)
        validated_mask = _validate_mask(canopy_mask, validated_image.shape[:2]) if canopy_mask is not None else None
        input_tensor, scale, pad_xy = _preprocess(validated_image, imgsz=self.imgsz)
        output = self._session.run(None, {self._input_name: input_tensor})[0]
        instances = _decode_yolov8_output(
            output,
            image_shape=validated_image.shape[:2],
            scale=scale,
            pad_xy=pad_xy,
            canopy_mask=validated_mask,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
        counts = {name: 0 for name in CLASS_NAMES}
        for instance in instances:
            counts[instance.class_name] += 1
        overlay = _draw_overlay(validated_image, instances)
        return YoloOrganDetectionResult(
            status="computed",
            message="YOLOv8 ONNX TOP-view organ counts computed.",
            counts=counts,
            instances=instances,
            overlay_image=overlay,
            debug_images={
                "overlay": overlay,
            },
        )

    def _create_session(self, session_factory: SessionFactory | None) -> _OnnxSession:
        if session_factory is not None:
            return session_factory(self.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO ONNX model not found: {self.model_path}")
        import onnxruntime as ort  # noqa: PLC0415

        return ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])


def resolve_default_model_path() -> Path:
    """Return the absolute path to the packaged best.onnx model."""

    return (Path(__file__).resolve().parents[1] / DEFAULT_MODEL_PATH).resolve()


@lru_cache(maxsize=1)
def _default_counter() -> YoloOrganCounter:
    return YoloOrganCounter(model_path=resolve_default_model_path())


def detect_top_organs_with_yolo(image: np.ndarray, canopy_mask: np.ndarray) -> YoloOrganDetectionResult:
    """Use the packaged best.onnx model to count TOP-view strawberry organs."""

    return _default_counter().detect(image, canopy_mask)


def _resolve_model_path(model_path: Path | str | None) -> Path:
    if model_path is None:
        return resolve_default_model_path()
    return Path(model_path)


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a 3-channel BGR image")
    return image


def _validate_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError("canopy_mask must be a numpy.ndarray")
    if mask.ndim != 2:
        raise ValueError("canopy_mask must be a single-channel mask")
    if mask.shape != image_shape:
        raise ValueError("image and canopy_mask must have matching height and width")
    return (mask > 0).astype(np.uint8) * 255


def _preprocess(image: np.ndarray, *, imgsz: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    scale = min(imgsz / float(width), imgsz / float(height))
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (imgsz - resized_width) / 2.0
    pad_y = (imgsz - resized_height) / 2.0
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor), scale, (float(left), float(top))


def _decode_yolov8_output(
    output: np.ndarray,
    *,
    image_shape: tuple[int, int],
    scale: float,
    pad_xy: tuple[float, float],
    canopy_mask: np.ndarray | None,
    conf_threshold: float,
    iou_threshold: float,
) -> list[YoloOrganInstance]:
    predictions = _normalize_output_shape(output)
    candidates: list[YoloOrganInstance] = []
    image_height, image_width = image_shape
    pad_x, pad_y = pad_xy
    for row in predictions:
        scores = row[4 : 4 + len(CLASS_NAMES)]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if confidence < conf_threshold:
            continue
        center_x, center_y, box_width, box_height = [float(value) for value in row[:4]]
        x1 = (center_x - box_width / 2.0 - pad_x) / scale
        y1 = (center_y - box_height / 2.0 - pad_y) / scale
        x2 = (center_x + box_width / 2.0 - pad_x) / scale
        y2 = (center_y + box_height / 2.0 - pad_y) / scale
        x1 = float(np.clip(x1, 0, image_width - 1))
        y1 = float(np.clip(y1, 0, image_height - 1))
        x2 = float(np.clip(x2, 0, image_width - 1))
        y2 = float(np.clip(y2, 0, image_height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        center_x_original = int(round((x1 + x2) / 2.0))
        center_y_original = int(round((y1 + y2) / 2.0))
        if canopy_mask is not None and canopy_mask[center_y_original, center_x_original] == 0:
            continue
        candidates.append(
            YoloOrganInstance(
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                confidence=confidence,
                bounding_box=(
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                ),
            )
        )
    return _nms_by_class(candidates, iou_threshold=iou_threshold)


def _normalize_output_shape(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim == 3:
        array = array[0]
    if array.shape[0] == 4 + len(CLASS_NAMES):
        array = array.T
    if array.ndim != 2 or array.shape[1] < 4 + len(CLASS_NAMES):
        raise ValueError(f"Unexpected YOLO output shape: {tuple(np.asarray(output).shape)}")
    return array


def _nms_by_class(instances: list[YoloOrganInstance], *, iou_threshold: float) -> list[YoloOrganInstance]:
    kept: list[YoloOrganInstance] = []
    for class_id in range(len(CLASS_NAMES)):
        class_instances = [item for item in instances if item.class_id == class_id]
        class_instances.sort(key=lambda item: item.confidence, reverse=True)
        while class_instances:
            current = class_instances.pop(0)
            kept.append(current)
            class_instances = [
                item for item in class_instances if _bbox_iou(current.bounding_box, item.bounding_box) <= iou_threshold
            ]
    kept.sort(key=lambda item: item.confidence, reverse=True)
    return kept


def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def _draw_overlay(image: np.ndarray, instances: list[YoloOrganInstance]) -> np.ndarray:
    overlay = image.copy()
    colors = {
        "flower": (255, 255, 255),
        "flower_bud": (0, 215, 255),
        "fruit": (0, 0, 255),
    }
    for instance in instances:
        x1, y1, x2, y2 = instance.bounding_box
        color = colors[instance.class_name]
        label = f"{instance.class_name} {instance.confidence:.2f}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            label,
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay
