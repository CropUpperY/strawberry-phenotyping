"""Smoke test for the GUI prototype."""

import os
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QGroupBox
from PyQt5.QtGui import QColor, QPixmap

from core.batch_processor import BatchAnalysisReport, BatchSampleResult
from core.grouping import GroupingSuggestion
from core.grouping import PlantImageGroup
from core.grouping import VIEW_FRONT_0, VIEW_TOP
from core.pipeline import PlantAnalysisResult, TraitResult, ViewLoadResult
from core.pipeline import TRAIT_SPECS
from config.settings import DEFAULT_CONFIG
import gui.main_window as main_window_module
from gui.main_window import ImageViewerDialog, PreprocessResult, PreprocessThread, StrawberryMainWindow
from gui.color_card_selector import ColorCardRegion, ColorCardRegions


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", image)[1].tofile(str(path))


def test_main_window_initializes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The main window should build successfully in an offscreen Qt session."""
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()
    window.show()
    app.processEvents()

    assert window.windowTitle() == DEFAULT_CONFIG.app_name
    assert window.group_list is not None
    assert window.results_table.rowCount() == len([spec for spec in TRAIT_SPECS if "TOP" in spec.source_views])
    assert window.results_table.columnCount() == 2
    assert window.card_width_spin.value() == 1.50
    assert window.card_height_spin.value() == 1.50
    assert window.preview_view_prev_button.text() == "<"
    assert window.preview_view_next_button.text() == ">"
    assert window.preview_view_name_label.text() == "TOP"
    assert window.preview_view_index_label.text() == "1 / 1"
    assert window.preview_notice_label.text() == ""
    assert window.orig_preview.title.text() == "1. 原始图像"
    assert window.calib_preview.title.text() == "2. 颜色校正"
    assert window.mask_preview.title.text() == "3. 背景消除"
    assert window.final_preview.title.text() == "4. 表型提取"
    assert all(group.title() != "当前样本信息" for group in window.findChildren(QGroupBox))
    assert not window.trait_gallery_group.isVisible()
    for card in (
        window.orig_preview,
        window.calib_preview,
        window.mask_preview,
        window.final_preview,
    ):
        assert card.meta_label.isVisible()
        assert card.prev_nav_button is not None
        assert card.next_nav_button is not None
    assert hasattr(window, "top_preview")
    assert hasattr(window, "front0_preview")
    assert hasattr(window, "front180_preview")
    # 检查色卡预览控件
    assert hasattr(window, "cc_top_preview")
    assert hasattr(window, "cc_front0_preview")
    assert hasattr(window, "cc_front180_preview")
    assert hasattr(window, "color_card_group")
    # 检查色卡选择和预处理按钮
    assert hasattr(window, "select_color_card_btn")
    assert hasattr(window, "preprocess_button")
    assert window.preprocess_button.text() == "预处理"
    assert window.single_button.text() == "表型提取"
    assert window.batch_button.text() == "批量处理"
    window.close()


def test_main_window_builds_shared_preview_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """The center panel should expose the shared view switcher and fixed four-stage cards."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()
    window.show()
    app.processEvents()

    assert window.preview_view_prev_button is not None
    assert window.preview_view_next_button is not None
    assert window.preview_view_name_label.text() == "TOP"
    assert window.preview_view_index_label.text() == "1 / 1"
    assert window.preview_notice_label.text() == ""
    assert window.orig_preview.title.text() == "1. 原始图像"
    assert window.calib_preview.title.text() == "2. 颜色校正"
    assert window.mask_preview.title.text() == "3. 背景消除"
    assert window.final_preview.title.text() == "4. 表型提取"
    assert not window.trait_gallery_group.isVisible()
    for card in (
        window.orig_preview,
        window.calib_preview,
        window.mask_preview,
        window.final_preview,
    ):
        assert card.meta_label.isVisible()
        assert card.prev_nav_button is not None
        assert card.next_nav_button is not None
    assert hasattr(window, "top_preview")
    assert hasattr(window, "front0_preview")
    assert hasattr(window, "front180_preview")
    window.close()


def test_grouping_issue_message_lists_problem_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grouping issues should list only unrecognized filenames and rename guidance."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()
    message = window._build_grouping_issue_message(
        [
            GroupingSuggestion(
                image_path=Path("mystery_view.png"),
                reason="无法从文件名判断样本编号和视角，需要手动确认。",
                options=[],
            )
        ],
    )

    assert "mystery_view.png" in message
    assert "正确格式" in message
    assert "4A、4B 都是错误命名" in message
    assert "重新扫描" in message
    window.close()


def test_select_color_card_regions_skips_missing_optional_front_view(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Two-view cotton samples should pass None for the absent FRONT-2 image."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()
    group = PlantImageGroup(
        sample_id="cotton-1",
        top_image=tmp_path / "top.jpg",
        front_0_image=tmp_path / "front.jpg",
        front_180_image=None,
        required_views=(VIEW_TOP, VIEW_FRONT_0),
    )
    window.groups = [group]
    monkeypatch.setattr(window, "_selected_group", lambda: group)
    monkeypatch.setattr(window, "_update_color_card_status", lambda: None)
    monkeypatch.setattr(window, "_update_color_card_preview", lambda: None)
    monkeypatch.setattr(main_window_module, "save_color_card_regions", lambda regions, path: None)
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args, **kwargs: None)

    image = np.zeros((12, 12, 3), dtype=np.uint8)

    def fake_imread(path: str) -> np.ndarray | None:
        return None if path == "None" else image

    captured: dict[str, np.ndarray | None] = {}

    def fake_selector(**kwargs) -> ColorCardRegions:
        captured.update(
            {
                "top_image": kwargs["top_image"],
                "front_1_image": kwargs["front_1_image"],
                "front_2_image": kwargs["front_2_image"],
            }
        )
        region = ColorCardRegion(1, 1, 4, 4)
        return ColorCardRegions(top=region, front_1=region, front_2=None)

    monkeypatch.setattr(cv2, "imread", fake_imread)
    monkeypatch.setattr(main_window_module, "select_color_card_regions_interactive", fake_selector)

    window._handle_select_color_card_regions()

    assert captured["top_image"] is image
    assert captured["front_1_image"] is image
    assert captured["front_2_image"] is None
    window.close()


def test_cotton_preprocess_uses_original_images_without_color_card(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cotton samples have no color card, so preprocessing should be a no-op."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()
    group = PlantImageGroup(
        sample_id="cotton-1",
        top_image=tmp_path / "top.jpg",
        front_0_image=tmp_path / "front.jpg",
        front_180_image=None,
        required_views=(VIEW_TOP, VIEW_FRONT_0),
    )

    image_by_name = {
        "top.jpg": np.full((8, 8, 3), 40, dtype=np.uint8),
        "front.jpg": np.full((8, 8, 3), 90, dtype=np.uint8),
    }

    def fake_load_image(path: Path) -> np.ndarray:
        return image_by_name[path.name].copy()

    monkeypatch.setattr(main_window_module, "load_image", fake_load_image)

    result = window._build_noop_preprocess_result(group)

    assert result.is_valid
    assert set(result.loaded_images) == {"TOP", "FRONT-1"}
    assert set(result.calibrated_images) == {"TOP", "FRONT-1"}
    assert np.array_equal(result.calibrated_images["TOP"], image_by_name["top.jpg"])
    assert np.array_equal(result.calibrated_images["FRONT-1"], image_by_name["front.jpg"])
    assert all(not item.is_calibrated for item in result.calibration_results.values())
    assert "无色卡" in result.message
    window.close()


def test_cotton_preprocess_button_does_not_require_color_card(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Clicking preprocess for cotton should not show the color-card warning."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()
    group = PlantImageGroup(
        sample_id="cotton-1",
        top_image=tmp_path / "top.jpg",
        front_0_image=tmp_path / "front.jpg",
        front_180_image=None,
        required_views=(VIEW_TOP, VIEW_FRONT_0),
    )
    window.groups = [group]
    image = np.full((8, 8, 3), 80, dtype=np.uint8)

    warnings: list[tuple[str, str]] = []
    finished_results: list[PreprocessResult] = []

    monkeypatch.setattr(window, "_selected_group", lambda: group)
    monkeypatch.setattr(main_window_module, "load_image", lambda path: image.copy())
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((title, text)),
    )
    monkeypatch.setattr(window, "_on_preprocess_finished", lambda result: finished_results.append(result))

    window.color_card_regions = None
    window._handle_preprocess()

    assert warnings == []
    assert len(finished_results) == 1
    assert finished_results[0].is_valid
    window.close()


def test_preprocess_runtime_visualizations_are_downsampled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Runtime preprocessing previews should not write full-resolution PNGs."""

    captured_steps: list[tuple[str, np.ndarray]] = []

    def fake_save_debug_steps(
        sample_id: str,
        category_key: str,
        steps: list[tuple[str, np.ndarray]],
        *,
        output_root: str | Path,
    ) -> list[Path]:
        captured_steps.extend(steps)
        return [Path(output_root) / sample_id / category_key / "preview.png"]

    monkeypatch.setattr("utils.debug_artifacts.save_debug_steps", fake_save_debug_steps)

    large = np.zeros((2160, 3840, 3), dtype=np.uint8)
    result = PreprocessResult(sample_id="1AB")
    result.loaded_images["TOP"] = large
    result.calibrated_images["TOP"] = large.copy()

    region = ColorCardRegion(1, 1, 4, 4)
    worker = PreprocessThread(
        group=PlantImageGroup(sample_id="1AB"),
        color_card_regions=ColorCardRegions(top=region, front_1=region, front_2=region),
        calibration_reference=None,
        debug_output_dir=tmp_path,
        save_full_debug=False,
    )

    worker._save_preprocess_visualizations(result)

    assert captured_steps
    assert all(max(image.shape[:2]) <= 1200 for _, image in captured_steps)


def test_trait_table_selection_updates_center_trait_focus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting a trait row should update the linked phenotype summary in the center panel."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()
    result = PlantAnalysisResult(
        sample_id="1AB",
        status="analysis_complete",
        message="done",
        traits=[
            TraitResult(
                key=spec.key,
                label=spec.label,
                source_views=spec.source_views,
                unit=spec.unit,
                value=12.34 if spec.key == "leaf_area" else None,
                status="computed" if spec.key == "leaf_area" else "pending",
                message="leaf area ok" if spec.key == "leaf_area" else "pending",
            )
            for spec in TRAIT_SPECS
        ],
        view_results={
            "TOP": ViewLoadResult("TOP", None, "loaded"),
            "FRONT-1": ViewLoadResult("FRONT-1", None, "loaded"),
            "FRONT-2": ViewLoadResult("FRONT-2", None, "loaded"),
        },
    )

    window.current_result = result
    window._set_displayed_result(result)

    assert "叶面积" in window.trait_focus_title.text()
    assert "12.34 cm^2" in window.trait_focus_value.text()
    assert "已计算" in window.trait_focus_status.text()
    assert "leaf area ok" in window.trait_focus_message.text()
    window.close()


def test_result_table_uses_single_result_column_and_chinese_count_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measurement results should combine value and unit in one localized result column."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()
    result = PlantAnalysisResult(
        sample_id="1AB",
        status="analysis_complete",
        message="done",
        traits=[
            TraitResult(
                key=spec.key,
                label=spec.label,
                source_views=spec.source_views,
                unit=spec.unit,
                value=2 if spec.key == "flower_count" else None,
                status="computed" if spec.key == "flower_count" else "pending",
            )
            for spec in TRAIT_SPECS
        ],
        view_results={
            "TOP": ViewLoadResult("TOP", None, "loaded"),
            "FRONT-1": ViewLoadResult("FRONT-1", None, "loaded"),
            "FRONT-2": ViewLoadResult("FRONT-2", None, "loaded"),
        },
    )

    window._set_result_rows(result)
    flower_label = next(spec.label for spec in TRAIT_SPECS if spec.key == "flower_count")
    flower_row = next(
        row for row in range(window.results_table.rowCount())
        if window.results_table.item(row, 0).text() == flower_label
    )

    assert window.results_table.horizontalHeaderItem(0).text() == "性状"
    assert window.results_table.horizontalHeaderItem(1).text() == "结果"
    assert window.results_table.item(flower_row, 1).text() == "2 个"
    window.close()


def test_result_table_tracks_active_preview_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switching the preview view should show measurements for that view only."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)
    monkeypatch.setattr(StrawberryMainWindow, "_find_latest_sample_visualization_root", lambda self, sample_id: None)

    window = StrawberryMainWindow()
    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )
    result = PlantAnalysisResult(
        sample_id="1AB",
        status="analysis_complete",
        message="done",
        traits=[
            TraitResult(
                key=spec.key,
                label=spec.label,
                source_views=spec.source_views,
                unit=spec.unit,
                value={
                    "leaf_area": 12.34,
                    "greenness": 45.67,
                    "convex_hull_area": 14.56,
                    "canopy_width": 8.9,
                    "flower_count": 2,
                    "flower_bud_count": 4,
                    "fruit_count": 0,
                    "canopy_height": 7.89,
                    "side_projection_area": 9.87,
                }.get(spec.key),
                status="computed",
            )
            for spec in TRAIT_SPECS
        ],
        view_results={
            "TOP": ViewLoadResult("TOP", None, "loaded"),
            "FRONT-1": ViewLoadResult("FRONT-1", None, "loaded"),
            "FRONT-2": ViewLoadResult("FRONT-2", None, "loaded"),
        },
        calibration_results={
            "FRONT-1": SimpleNamespace(status="calibrated", mm_per_pixel=0.2, is_calibrated=True),
            "FRONT-2": SimpleNamespace(status="calibrated", mm_per_pixel=0.2, is_calibrated=True),
        },
        front_segmentations={
            "FRONT-1": SimpleNamespace(bounding_box=(0, 0, 10, 200), mask_area_pixels=5000),
            "FRONT-2": SimpleNamespace(bounding_box=(0, 0, 20, 300), mask_area_pixels=10000),
        },
    )

    window.groups = [group]
    window.current_result = result
    window._show_trait_preview_for_sample("1AB")

    top_labels = [window.results_table.item(row, 0).text() for row in range(window.results_table.rowCount())]
    top_values = [window.results_table.item(row, 1).text() for row in range(window.results_table.rowCount())]
    assert result.trait_map()["canopy_height"].label not in top_labels
    assert "12.34 cm^2" in top_values
    assert any(value.startswith("2 ") for value in top_values)

    window._show_next_preview_view()
    front_1_labels = [window.results_table.item(row, 0).text() for row in range(window.results_table.rowCount())]
    front_1_values = [window.results_table.item(row, 1).text() for row in range(window.results_table.rowCount())]
    assert front_1_labels == [
        result.trait_map()["canopy_height"].label,
        result.trait_map()["side_projection_area"].label,
    ]
    assert front_1_values == ["4.00 cm", "2.00 cm^2"]

    window._show_next_preview_view()
    front_2_values = [window.results_table.item(row, 1).text() for row in range(window.results_table.rowCount())]
    assert front_2_values == ["6.00 cm", "4.00 cm^2"]
    window.close()


def test_batch_preview_switch_updates_measurement_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switching displayed batch samples should sync the measurement table."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)
    monkeypatch.setattr(StrawberryMainWindow, "_find_latest_sample_visualization_root", lambda self, sample_id: None)

    window = StrawberryMainWindow()

    def make_result(sample_id: str, leaf_value: float) -> PlantAnalysisResult:
        return PlantAnalysisResult(
            sample_id=sample_id,
            status="analysis_complete",
            message=sample_id,
            traits=[
                TraitResult(
                    key=spec.key,
                    label=spec.label,
                    source_views=spec.source_views,
                    unit=spec.unit,
                    value=leaf_value if spec.key == "leaf_area" else None,
                    status="computed" if spec.key == "leaf_area" else "pending",
                    message=sample_id,
                )
                for spec in TRAIT_SPECS
            ],
            view_results={
                "TOP": ViewLoadResult("TOP", None, "loaded"),
                "FRONT-1": ViewLoadResult("FRONT-1", None, "loaded"),
                "FRONT-2": ViewLoadResult("FRONT-2", None, "loaded"),
            },
        )

    report = BatchAnalysisReport(
        directory=Path("."),
        sample_results=[
            BatchSampleResult(PlantImageGroup(sample_id="1AB"), make_result("1AB", 11.11)),
            BatchSampleResult(PlantImageGroup(sample_id="2AB"), make_result("2AB", 22.22)),
        ],
        started_at=datetime.now(),
        finished_at=datetime.now(),
    )

    window.current_batch_report = report
    window._show_trait_preview_for_sample("1AB")
    assert window.results_table.item(0, 1).text() == "11.11 cm^2"

    window._show_trait_preview_for_sample("2AB")
    assert window.results_table.item(0, 1).text() == "22.22 cm^2"
    assert window.displayed_result is not None
    assert window.displayed_result.sample_id == "2AB"
    window.close()


def test_image_viewer_dialog_size_remains_stable_after_show() -> None:
    """Opening the zoom viewer should not keep enlarging the dialog."""

    app = QApplication.instance() or QApplication([])

    pixmap = QPixmap(1600, 900)
    pixmap.fill(QColor("#6b8f71"))

    dialog = ImageViewerDialog(pixmap, title="test")
    dialog.show()

    observed_sizes = []
    for _ in range(6):
        app.processEvents()
        observed_sizes.append((dialog.width(), dialog.height()))

    assert len(set(observed_sizes[-3:])) == 1
    dialog.close()


def test_top_background_preview_prefers_background_removed_plant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """TOP background stage should prefer the masked plant image over the raw mask."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    sample_root = tmp_path / "111AB"
    green_folder = sample_root / "06_绿度计算"
    top_folder = sample_root / "02_TOP俯视图分割"
    green_folder.mkdir(parents=True)
    top_folder.mkdir(parents=True)

    mask_path = top_folder / "14_最终分割掩码.png"
    plant_path = green_folder / "02_植株区域提取.png"
    mask_path.write_bytes(b"mask")
    plant_path.write_bytes(b"plant")

    window = StrawberryMainWindow()

    assert window._pick_top_background_image(sample_root) == plant_path
    window.close()


def test_front_background_preview_prefers_background_removed_plant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FRONT background stage should prefer the plant-only image over montage/process summaries."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    sample_root = tmp_path / "111AB"
    front_folder = sample_root / "03_FRONT正视图分割"
    front_folder.mkdir(parents=True)

    montage_path = front_folder / "18_front_1_process_montage.png"
    overlay_path = front_folder / "15_FRONT-1投影覆盖.png"
    plant_path = front_folder / "14_FRONT-1植株区域.png"
    montage_path.write_bytes(b"montage")
    overlay_path.write_bytes(b"overlay")
    plant_path.write_bytes(b"plant")

    window = StrawberryMainWindow()

    assert window._pick_front_process_image(sample_root, front_tag="front_1") == plant_path
    window.close()


def test_calibration_preview_prefers_full_view_corrected_image_over_roi_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Calibration preview should prefer the full corrected plant image, not ROI debug crops."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    sample_root = tmp_path / "2AB"
    calibration_folder = sample_root / "01_色卡检测与校正"
    calibration_folder.mkdir(parents=True)

    roi_path = calibration_folder / "16_TOP_roi_variant_最佳变体.png"
    corrected_path = calibration_folder / "02_TOP_色彩增强后.png"
    roi_path.write_bytes(b"roi")
    corrected_path.write_bytes(b"corrected")

    window = StrawberryMainWindow()

    assert window._pick_calibration_image(sample_root, "TOP") == corrected_path
    window.close()


def test_calibration_preview_ignores_roi_only_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Calibration preview should not display ROI/card crops as the corrected full-view image."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    sample_root = tmp_path / "1AB"
    calibration_folder = sample_root / "01_色卡检测与校正"
    calibration_folder.mkdir(parents=True)

    (calibration_folder / "15_TOP_roi_variant_最佳变体.png").write_bytes(b"roi")
    (calibration_folder / "19_TOP_manual_corrected_card.png").write_bytes(b"card")

    window = StrawberryMainWindow()

    assert window._pick_calibration_image(sample_root, "TOP") is None
    window.close()


def test_calibration_preview_extracts_corrected_half_from_comparison_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Before/after comparison images should render only the corrected half in the calibration card."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    sample_root = tmp_path / "111AB"
    calibration_folder = sample_root / "01_色卡检测与校正"
    calibration_folder.mkdir(parents=True)

    left_half = np.full((20, 18, 3), (10, 20, 30), dtype=np.uint8)
    right_half = np.full((20, 18, 3), (30, 200, 180), dtype=np.uint8)
    comparison = np.hstack([left_half, right_half])
    comparison_path = calibration_folder / "20_TOP_manual_before_after.png"
    _write_png(comparison_path, comparison)

    window = StrawberryMainWindow()
    extracted = window._extract_corrected_half_from_comparison(comparison_path)

    assert extracted is not None
    assert extracted.shape == right_half.shape
    assert np.array_equal(extracted, right_half)
    window.close()


def test_single_analysis_skips_runtime_visualization_export_when_debug_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single analysis should avoid heavy runtime PNG export in non-debug mode."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    class _Signal:
        def connect(self, _slot) -> None:
            return None

    captured: dict[str, object] = {}

    class DummyAnalysisThread:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.log_message = _Signal()
            self.status_message = _Signal()
            self.single_finished = _Signal()
            self.batch_finished = _Signal()
            self.failed = _Signal()
            self.finished = _Signal()

        def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr(main_window_module, "AnalysisThread", DummyAnalysisThread)

    window = StrawberryMainWindow()
    group = PlantImageGroup(sample_id="1AB")
    window.preprocess_result = PreprocessResult(
        sample_id="1AB",
        calibrated_images={},
        calibration_results={},
        is_valid=True,
    )
    monkeypatch.setattr(window, "_selected_group", lambda: group)
    window.debug_mode_checkbox.setChecked(False)

    window._handle_single_analysis()

    assert captured["mode"] == "single"
    assert captured["group"] == group
    assert captured["debug_output_dir"] is None
    assert captured["started"] is True
    window.close()


def test_phenotype_stage_payloads_use_in_memory_results_without_saved_visualizations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phenotype cards should render from the current result even when no visualization folder exists."""

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)
    monkeypatch.setattr(StrawberryMainWindow, "_find_latest_sample_visualization_root", lambda self, sample_id: None)

    window = StrawberryMainWindow()
    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("top.png"),
        front_0_image=Path("front-1.png"),
        front_180_image=Path("front-2.png"),
    )
    top_image = np.full((24, 24, 3), (20, 150, 40), dtype=np.uint8)
    front_1_image = np.full((24, 24, 3), (30, 120, 60), dtype=np.uint8)
    front_2_image = np.full((24, 24, 3), (40, 110, 80), dtype=np.uint8)
    top_mask = np.zeros((24, 24), dtype=np.uint8)
    top_mask[4:20, 5:19] = 255
    front_1_mask = np.zeros((24, 24), dtype=np.uint8)
    front_1_mask[3:21, 7:17] = 255
    front_2_mask = np.zeros((24, 24), dtype=np.uint8)
    front_2_mask[2:22, 6:18] = 255

    window.preprocess_result = PreprocessResult(
        sample_id="1AB",
        calibrated_images={
            "TOP": top_image,
            "FRONT-1": front_1_image,
            "FRONT-2": front_2_image,
        },
        calibration_results={},
        is_valid=True,
    )
    window.current_result = PlantAnalysisResult(
        sample_id="1AB",
        status="analysis_complete",
        message="done",
        traits=[
            TraitResult(
                key=spec.key,
                label=spec.label,
                source_views=spec.source_views,
                unit=spec.unit,
                status="computed",
            )
            for spec in TRAIT_SPECS
        ],
        view_results={
            "TOP": ViewLoadResult("TOP", None, "calibrated"),
            "FRONT-1": ViewLoadResult("FRONT-1", None, "calibrated"),
            "FRONT-2": ViewLoadResult("FRONT-2", None, "calibrated"),
        },
        top_segmentation=SimpleNamespace(
            mask=top_mask,
            contours=[
                np.array([[[5, 4]], [[18, 4]], [[18, 19]], [[5, 19]]], dtype=np.int32),
            ],
            convex_hull=np.array([[[5, 4]], [[18, 4]], [[18, 19]], [[5, 19]]], dtype=np.int32),
            hull_image=np.full((24, 24, 3), (0, 255, 255), dtype=np.uint8),
        ),
        top_organ_detection=SimpleNamespace(
            instances=[
                SimpleNamespace(
                    class_name="fruit",
                    confidence=0.91,
                    bounding_box=(9, 9, 13, 13),
                ),
            ],
        ),
        front_segmentations={
            "FRONT-1": SimpleNamespace(
                mask=front_1_mask,
                bounding_box_image=np.full((24, 24, 3), (255, 200, 0), dtype=np.uint8),
                contour_image=np.full((24, 24, 3), (200, 0, 0), dtype=np.uint8),
            ),
            "FRONT-2": SimpleNamespace(
                mask=front_2_mask,
                bounding_box_image=np.full((24, 24, 3), (0, 200, 255), dtype=np.uint8),
                contour_image=np.full((24, 24, 3), (0, 0, 200), dtype=np.uint8),
            ),
        },
    )

    top_payloads = window._build_phenotype_stage_payloads(group, "1AB", "TOP")
    front_payloads = window._build_phenotype_stage_payloads(group, "1AB", "FRONT-1")

    assert top_payloads["masked"].image_array is not None
    assert top_payloads["masked"].image_path is None
    assert top_payloads["final"].image_array is not None
    assert np.array_equal(top_payloads["final"].image_array[0, 0], np.array([0, 0, 0], dtype=np.uint8))
    assert np.any(np.all(top_payloads["final"].image_array == np.array([255, 255, 0], dtype=np.uint8), axis=2))
    assert np.array_equal(top_payloads["final"].image_array[9, 9], np.array([0, 0, 255], dtype=np.uint8))
    assert front_payloads["masked"].image_array is not None
    assert front_payloads["final"].image_array is not None
    window.close()
