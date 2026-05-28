"""Tests for ONNX YOLO organ counting."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.yolo_counter import DEFAULT_MODEL_PATH, YoloOrganCounter


class FakeSession:
    """Minimal ONNX Runtime session stand-in for post-processing tests."""

    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="images")]

    def run(self, _output_names: object, _inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [self.output]


def _empty_yolov8_output() -> np.ndarray:
    return np.zeros((1, 7, 8400), dtype=np.float32)


def test_default_model_path_uses_existing_best_onnx_name() -> None:
    """The packaged detector should use models/best.onnx without renaming it."""

    assert DEFAULT_MODEL_PATH == Path("models") / "best.onnx"


def test_yolo_counter_decodes_counts_from_raw_yolov8_output() -> None:
    """Raw YOLOv8 ONNX output should become class counts after confidence filtering and NMS."""

    output = _empty_yolov8_output()
    output[0, :, 0] = [320, 320, 80, 80, 0.90, 0.05, 0.01]
    output[0, :, 1] = [322, 322, 82, 82, 0.60, 0.05, 0.01]
    output[0, :, 2] = [200, 300, 50, 60, 0.02, 0.82, 0.01]
    output[0, :, 3] = [480, 220, 70, 55, 0.02, 0.03, 0.76]
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    canopy_mask = np.ones((640, 640), dtype=np.uint8) * 255

    counter = YoloOrganCounter(
        model_path=Path("unused.onnx"),
        session_factory=lambda _path: FakeSession(output),
    )
    result = counter.detect(image, canopy_mask)

    assert result.counts == {"flower": 1, "flower_bud": 1, "fruit": 1}
    assert result.flower_count == 1
    assert result.flower_bud_count == 1
    assert result.fruit_count == 1
    assert len(result.instances) == 3
    assert result.overlay_image.shape == image.shape


def test_yolo_counter_filters_detections_outside_canopy_mask() -> None:
    """Detections centered outside the TOP canopy should not count color cards or background objects."""

    output = _empty_yolov8_output()
    output[0, :, 0] = [100, 100, 60, 60, 0.90, 0.01, 0.01]
    output[0, :, 1] = [540, 540, 60, 60, 0.88, 0.01, 0.01]
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    canopy_mask = np.zeros((640, 640), dtype=np.uint8)
    canopy_mask[:240, :240] = 255

    counter = YoloOrganCounter(
        model_path=Path("unused.onnx"),
        session_factory=lambda _path: FakeSession(output),
    )
    result = counter.detect(image, canopy_mask)

    assert result.flower_count == 1
    assert result.flower_bud_count == 0
    assert result.fruit_count == 0
