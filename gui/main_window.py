"""Desktop GUI for the strawberry phenotype analysis workflow."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PyQt5.QtCore import QEvent, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from config.settings import DEFAULT_CONFIG
from core.image_io import load_image
from core.batch_processor import BatchAnalysisReport, analyze_groups
from core.calibration import create_color_card_reference
from core.grouping import (
    GroupingSuggestion,
    PlantImageGroup,
    collect_grouping_suggestions,
    find_incomplete_groups,
    group_image_files,
)
from core.pipeline import PlantAnalysisResult, TRAIT_SPECS, analyze_plant_group, format_status_label
from core.visualization import AnalysisDebugPreviews, build_analysis_debug_previews
from gui.color_card_selector import (
    ColorCardRegions,
    select_color_card_regions_interactive,
    save_color_card_regions,
    load_color_card_regions,
    draw_region_on_image,
)
from gui.stage_preview import PreviewViewState, StagePreviewPayload, available_views_for_group, pick_active_view, step_view
from utils.debug_artifacts import create_masked_color_image
from utils.exporter import export_batch_report, export_single_result
from utils.logger import setup_logger

try:
    import cv2
except ImportError:
    cv2 = None


DEFAULT_PATCH_WIDTH_CM = 1.50
DEFAULT_PATCH_HEIGHT_CM = 1.50
APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "logo.png"
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.calibration import ImageCalibrationResult


@dataclass
class PreprocessResult:
    """预处理结果缓存，包含校准后的图像和校准信息。"""
    sample_id: str
    loaded_images: dict[str, np.ndarray] = field(default_factory=dict)
    calibrated_images: dict[str, np.ndarray] = field(default_factory=dict)
    calibration_results: dict[str, Any] = field(default_factory=dict)
    is_valid: bool = False
    message: str = ""
    debug_images: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


class ImagePreviewCard(QFrame):
    """Reusable card widget for file-based image preview."""

    def __init__(self, title: str, *, preview_height: int = 260, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.preview_height = preview_height
        self._source_pixmap: QPixmap | None = None
        self._selected = False
        self._prev_handler: Callable[[], None] | None = None
        self._next_handler: Callable[[], None] | None = None

        self.title = QLabel(title)
        self.image_label = QLabel("等待加载图像")
        self.meta_label = QLabel("等待加载图像")
        self.prev_nav_button: QToolButton | None = None
        self.next_nav_button: QToolButton | None = None
        self._viewer_title = title
        self._build_ui()

    def _build_ui(self) -> None:
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.title.setObjectName("CardTitle")
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        header_layout.addWidget(self.title, 1)

        self.prev_nav_button = QToolButton()
        self.prev_nav_button.setObjectName("CardNavButton")
        self.prev_nav_button.setText("‹")
        self.prev_nav_button.setAutoRaise(True)
        self.prev_nav_button.setCursor(Qt.PointingHandCursor)
        self.prev_nav_button.setFixedSize(24, 24)
        self.prev_nav_button.clicked.connect(self._handle_prev_navigation)

        self.next_nav_button = QToolButton()
        self.next_nav_button.setObjectName("CardNavButton")
        self.next_nav_button.setText("›")
        self.next_nav_button.setAutoRaise(True)
        self.next_nav_button.setCursor(Qt.PointingHandCursor)
        self.next_nav_button.setFixedSize(24, 24)
        self.next_nav_button.clicked.connect(self._handle_next_navigation)

        header_layout.addWidget(self.prev_nav_button)
        header_layout.addWidget(self.next_nav_button)

        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(self.preview_height)
        self.image_label.setMaximumHeight(self.preview_height)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.image_label.setStyleSheet("border: 1px dashed #6b8f71; border-radius: 10px;")
        self.image_label.installEventFilter(self)
        self.image_label.setCursor(Qt.ArrowCursor)
        self.meta_label.setWordWrap(True)
        self.meta_label.setObjectName("MetaLabel")
        self.meta_label.setVisible(True)

        layout.addLayout(header_layout)
        layout.addWidget(self.image_label)
        layout.addWidget(self.meta_label)
        self.set_navigation_enabled(False)
        self._apply_selected_style()

    def set_image_path(self, image_path: Path | None, *, meta_text: str = "已加载") -> None:
        """Display an image from disk."""

        if image_path is None:
            self.clear("缺少该视角图像", "缺少该视角图像")
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.clear("图像预览加载失败", "图像预览加载失败")
            return

        self._source_pixmap = pixmap
        self._refresh_pixmap()
        self.image_label.setText("")
        self.meta_label.setText(meta_text)
        self.meta_label.setVisible(True)
        self.image_label.setCursor(Qt.PointingHandCursor)
        self.image_label.setToolTip("点击查看大图")

    def set_image_array(self, image: np.ndarray | None, *, meta_text: str = "已生成") -> None:
        """Display an image that is already available in memory."""

        if image is None:
            self.clear()
            return

        self._source_pixmap = _array_to_pixmap(image)
        self._refresh_pixmap()
        self.image_label.setText("")
        self.meta_label.setText(meta_text)
        self.meta_label.setVisible(True)
        self.image_label.setCursor(Qt.PointingHandCursor)
        self.image_label.setToolTip("点击查看大图")

    def clear(self, placeholder_text: str = "等待加载图像", meta_text: str = "等待加载图像") -> None:
        """Reset the card to an empty state."""

        self._source_pixmap = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(placeholder_text)
        self.meta_label.setText(meta_text)
        self.meta_label.setVisible(True)
        self.image_label.setCursor(Qt.ArrowCursor)
        self.image_label.setToolTip("")

    def set_viewer_title(self, title: str) -> None:
        self._viewer_title = title

    def set_navigation_handlers(
        self,
        prev_handler: Callable[[], None] | None,
        next_handler: Callable[[], None] | None,
    ) -> None:
        self._prev_handler = prev_handler
        self._next_handler = next_handler

    def set_navigation_enabled(self, enabled: bool) -> None:
        for button in (self.prev_nav_button, self.next_nav_button):
            if button is None:
                continue
            button.setVisible(enabled)
            button.setEnabled(enabled)

    def set_selected(self, selected: bool) -> None:
        """Apply a stronger border to the active trait preview."""

        self._selected = selected
        self._apply_selected_style()

    def _handle_prev_navigation(self) -> None:
        if self._prev_handler is not None:
            self._prev_handler()

    def _handle_next_navigation(self) -> None:
        if self._next_handler is not None:
            self._next_handler()

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        """Open a zoom viewer when clicking the preview image."""

        if watched is self.image_label and event.type() == QEvent.MouseButtonPress:
            if self._source_pixmap is not None:
                self._open_zoom_viewer()
                return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        """Rescale the preview pixmap when the card is resized."""

        self._refresh_pixmap()
        super().resizeEvent(event)

    def _refresh_pixmap(self) -> None:
        if self._source_pixmap is None:
            return

        scaled = self._source_pixmap.scaled(
            self.image_label.width(),
            self.preview_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _open_zoom_viewer(self) -> None:
        if self._source_pixmap is None:
            return
        dialog = ImageViewerDialog(self._source_pixmap, title=self._viewer_title, parent=self)
        dialog.exec_()

    def _apply_selected_style(self) -> None:
        border_color = "#2f6f4f" if self._selected else "#d4c4a8"
        border_width = 2 if self._selected else 1
        self.setStyleSheet(
            f"QFrame {{ border: {border_width}px solid {border_color}; border-radius: 12px; background: #fbf8f1; }}"
        )


class ImageViewerDialog(QDialog):
    """Image viewer dialog that opens in fit-to-window mode."""

    def __init__(self, pixmap: QPixmap, *, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._original_pixmap = pixmap

        self.setWindowTitle(f"大图查看 - {title}")

        initial_width, initial_height = self._compute_initial_size(pixmap)
        self.resize(initial_width, initial_height)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        # Ignore the pixmap size hint so the dialog size is not driven by the
        # currently displayed scaled image, which would otherwise cause the
        # window to grow when resize handling updates the pixmap.
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setMinimumSize(1, 1)

        layout.addWidget(self.image_label, 1)
        self._fit_to_window()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._fit_to_window()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._fit_to_window()

    def _compute_initial_size(self, pixmap: QPixmap) -> tuple[int, int]:
        screen = QApplication.primaryScreen()
        if screen is None or pixmap.width() <= 0 or pixmap.height() <= 0:
            return 1200, 860

        available = screen.availableGeometry()
        max_width = int(available.width() * 0.85)
        max_height = int(available.height() * 0.85)
        ratio = pixmap.width() / pixmap.height()

        width = max_width
        height = int(round(width / ratio))
        if height > max_height:
            height = max_height
            width = int(round(height * ratio))

        width = max(640, width)
        height = max(480, height)
        return width, height

    def _fit_to_window(self) -> None:
        if self._original_pixmap.width() <= 0 or self._original_pixmap.height() <= 0:
            return

        target_width = max(1, self.image_label.width())
        target_height = max(1, self.image_label.height())
        scaled = self._original_pixmap.scaled(
            target_width,
            target_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)


class AnalysisThread(QThread):
    """Background worker for single-sample or batch analysis."""

    log_message = pyqtSignal(str)
    single_finished = pyqtSignal(object)
    batch_finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(
        self,
        *,
        mode: str,
        debug_output_dir: Path | None,
        calibration_reference: Any,
        color_card_regions: ColorCardRegions | None = None,
        precomputed_calibration: dict[str, Any] | None = None,
        group: PlantImageGroup | None = None,
        groups: list[PlantImageGroup] | None = None,
        directory: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.mode = mode
        self.group = group
        self.groups = groups or []
        self.directory = directory
        self.debug_output_dir = debug_output_dir
        self.calibration_reference = calibration_reference
        self.color_card_regions = color_card_regions
        self.precomputed_calibration = precomputed_calibration

    def _build_manual_regions_dict(self) -> dict[str, tuple[int, int, int, int]] | None:
        """Convert ColorCardRegions to the dict format expected by pipeline."""
        if self.color_card_regions is None:
            return None
        
        regions = {}
        if self.color_card_regions.top is not None:
            regions["TOP"] = self.color_card_regions.top.to_tuple()
        if self.color_card_regions.front_1 is not None:
            regions["FRONT-1"] = self.color_card_regions.front_1.to_tuple()
        if self.color_card_regions.front_2 is not None:
            regions["FRONT-2"] = self.color_card_regions.front_2.to_tuple()
        
        return regions if regions else None

    def run(self) -> None:
        """Execute the requested analysis task in the background."""

        try:
            manual_regions = self._build_manual_regions_dict()
            
            if self.mode == "single":
                if self.group is None:
                    raise ValueError("Single analysis requires a group.")

                result = analyze_plant_group(
                    self.group,
                    emit_log=self.log_message.emit,
                    calibration_reference=self.calibration_reference,
                    debug_output_dir=self.debug_output_dir,
                    manual_color_card_regions=manual_regions,
                    precomputed_calibration=self.precomputed_calibration,
                )
                self.single_finished.emit(result)
                return

            if self.mode == "batch":
                if self.directory is None:
                    raise ValueError("Batch analysis requires a directory.")

                def on_progress(index: int, total: int, group: PlantImageGroup, result: PlantAnalysisResult) -> None:
                    self.status_message.emit(f"批量分析进度: {index}/{total} - {group.sample_id} - {result.status}")

                report = analyze_groups(
                    self.groups,
                    directory=self.directory,
                    emit_log=self.log_message.emit,
                    emit_progress=on_progress,
                    calibration_reference=self.calibration_reference,
                    debug_output_dir=self.debug_output_dir,
                    manual_color_card_regions=manual_regions,
                )
                self.batch_finished.emit(report)
                return

            raise ValueError(f"Unsupported analysis mode: {self.mode}")
        except Exception as error:  # noqa: BLE001
            self.failed.emit(str(error))


class PreprocessThread(QThread):
    """后台执行预处理任务：色卡定位、颜色校正和尺度标定。"""

    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal(object)  # PreprocessResult
    failed = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(
        self,
        *,
        group: PlantImageGroup,
        color_card_regions: ColorCardRegions,
        calibration_reference: Any,
        debug_output_dir: Path | None = None,
        save_full_debug: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.group = group
        self.color_card_regions = color_card_regions
        self.calibration_reference = calibration_reference
        self.debug_output_dir = debug_output_dir
        self.save_full_debug = save_full_debug

    def _build_manual_regions_dict(self) -> dict[str, tuple[int, int, int, int]]:
        """Convert ColorCardRegions to the dict format expected by calibration."""
        regions = {}
        if self.color_card_regions.top is not None:
            regions["TOP"] = self.color_card_regions.top.to_tuple()
        if self.color_card_regions.front_1 is not None:
            regions["FRONT-1"] = self.color_card_regions.front_1.to_tuple()
        if self.color_card_regions.front_2 is not None:
            regions["FRONT-2"] = self.color_card_regions.front_2.to_tuple()
        return regions

    def run(self) -> None:
        """执行预处理：读取图像、色卡校准。"""
        try:
            from core.image_io import load_image
            from core.calibration import calibrate_image_with_color_card

            result = PreprocessResult(sample_id=self.group.sample_id)
            manual_regions = self._build_manual_regions_dict()

            # 1. 读取三视角图像
            self.log_message.emit(f"开始读取样本 {self.group.sample_id} 的图像...")
            view_paths = {
                "TOP": self.group.top_image,
                "FRONT-1": self.group.front_0_image,
                "FRONT-2": self.group.front_180_image,
            }

            for view_name, path in view_paths.items():
                if path is None:
                    result.message = f"{view_name} 图像缺失"
                    result.is_valid = False
                    self.finished_signal.emit(result)
                    return

                self.log_message.emit(f"读取 {view_name}: {path.name}")
                try:
                    image = load_image(path)
                    result.loaded_images[view_name] = image
                except Exception as e:
                    result.message = f"读取 {view_name} 失败: {e}"
                    result.is_valid = False
                    self.finished_signal.emit(result)
                    return

            # 2. 对每个视角执行色卡校准
            self.log_message.emit("开始色卡检测与颜色校正...")
            for view_name, image in result.loaded_images.items():
                self.status_message.emit(f"正在处理 {view_name} 色卡校准...")
                self.log_message.emit(f"执行 {view_name} 色卡检测...")

                manual_region = manual_regions.get(view_name)
                try:
                    if self.calibration_reference is None:
                        calibration = calibrate_image_with_color_card(
                            image,
                            view_name=view_name,
                            manual_region=manual_region,
                        )
                    else:
                        calibration = calibrate_image_with_color_card(
                            image,
                            view_name=view_name,
                            reference=self.calibration_reference,
                            manual_region=manual_region,
                        )
                except Exception as e:
                    self.log_message.emit(f"{view_name} 色卡校准异常: {e}")
                    # 创建回退结果
                    from types import SimpleNamespace
                    calibration = SimpleNamespace(
                        status="not_detected",
                        message=f"{view_name} 色卡校准失败: {e}",
                        view_name=view_name,
                        corrected_image=image.copy(),
                        is_calibrated=False,
                        mm_per_pixel=None,
                        pixels_per_mm=None,
                        debug_images={},
                    )

                result.calibration_results[view_name] = calibration
                result.calibrated_images[view_name] = getattr(
                    calibration, "corrected_image", image
                )
                result.debug_images[view_name] = getattr(calibration, "debug_images", {})

                if getattr(calibration, "is_calibrated", False):
                    self.log_message.emit(
                        f"{view_name} 校正完成: mm_per_pixel={calibration.mm_per_pixel:.4f}"
                    )
                else:
                    self.log_message.emit(
                        f"{view_name} 校正未成功: {getattr(calibration, 'message', '未知原因')}"
                    )

            # 3. 保存预处理可视化结果
            if self.debug_output_dir is not None:
                if self.save_full_debug:
                    self._save_debug_artifacts(result)
                self._save_preprocess_visualizations(result)

            # 4. 检查是否至少有一个视角校准成功
            calibrated_count = sum(
                1 for cal in result.calibration_results.values()
                if getattr(cal, "is_calibrated", False)
            )
            if calibrated_count > 0:
                result.is_valid = True
                result.message = f"预处理完成，{calibrated_count}/3 个视角色卡校准成功"
            else:
                result.is_valid = True  # 仍允许继续，但有警告
                result.message = "预处理完成，但所有视角均未成功检测色卡（将使用原图和默认尺度）"

            self.log_message.emit(result.message)
            self.finished_signal.emit(result)

        except Exception as error:  # noqa: BLE001
            self.failed.emit(str(error))

    def _save_debug_artifacts(self, result: PreprocessResult) -> None:
        """保存预处理调试图像。"""
        from utils.debug_artifacts import save_debug_steps

        output_root = Path(self.debug_output_dir)
        
        # 收集所有视角的调试图像，统一放在一个目录
        all_steps = []
        for view_name, calibration in result.calibration_results.items():
            debug_imgs = getattr(calibration, "debug_images", {})
            if not debug_imgs:
                continue
            # 为每个视角的图像添加视角前缀
            view_prefix = view_name.replace("-", "_")
            for name, img in debug_imgs.items():
                all_steps.append((f"{view_prefix}_{name}", img))
        
        if all_steps:
            saved_paths = save_debug_steps(
                sample_id=result.sample_id,
                category_key="color_calibration",
                steps=all_steps,
                output_root=output_root,
            )
            if saved_paths:
                self.log_message.emit(f"色卡检测调试图像已保存到: {saved_paths[0].parent}")

    def _save_preprocess_visualizations(self, result: PreprocessResult) -> None:
        """保存轻量预处理可视化：各视角色彩增强结果。"""
        from utils.debug_artifacts import save_debug_steps

        output_root = Path(self.debug_output_dir)
        steps: list[tuple[str, np.ndarray]] = []

        for view_name in ("TOP", "FRONT-1", "FRONT-2"):
            loaded = result.loaded_images.get(view_name)
            corrected = result.calibrated_images.get(view_name)
            if loaded is None or corrected is None:
                continue

            view_key = view_name.replace("-", "_")
            steps.append((f"{view_key}_原始图像", loaded))
            steps.append((f"{view_key}_色彩增强后", corrected))

            # 生成原图/增强图对比图，便于快速查看校正效果
            if loaded.shape == corrected.shape:
                before_after = np.hstack([loaded, corrected])
                steps.append((f"{view_key}_增强前后对比", before_after))

        if steps:
            saved_paths = save_debug_steps(
                sample_id=result.sample_id,
                category_key="color_calibration",
                steps=steps,
                output_root=output_root,
            )
            if saved_paths:
                self.log_message.emit(f"预处理可视化结果已保存到: {saved_paths[0].parent}")


class StrawberryMainWindow(QMainWindow):
    """Main GUI window for grouped sample analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = setup_logger(name="strawberry.gui", log_file="logs/gui.log")
        self.current_directory = DEFAULT_CONFIG.data_dir.resolve()
        self.groups: list[PlantImageGroup] = []
        self.current_result: PlantAnalysisResult | None = None
        self.displayed_result: PlantAnalysisResult | None = None
        self.current_batch_report: BatchAnalysisReport | None = None
        self.analysis_thread: AnalysisThread | None = None
        
        # 色卡区域选择状态
        self.color_card_regions: ColorCardRegions | None = None
        self.color_card_config_path = DEFAULT_CONFIG.output_dir / "color_card_regions.json"
        
        # 预处理状态
        self.preprocess_thread: PreprocessThread | None = None
        self.preprocess_result: PreprocessResult | None = None

        self.preview_mode = "original"
        self.preview_view_state = PreviewViewState(
            sample_id=None,
            mode="original",
            view_name="TOP",
            available_views=("TOP",),
        )
        self.batch_preview_sample_ids: list[str] = []
        self.batch_preview_index = 0
        self._active_trait_key: str | None = None

        self.setWindowIcon(self._load_app_icon())

        self._build_ui()
        self._apply_styles()
        self._load_groups(self.current_directory)
        self._try_load_color_card_regions()

    def _try_load_color_card_regions(self) -> None:
        """Try to load previously saved color card regions."""
        regions = load_color_card_regions(self.color_card_config_path)
        if regions and regions.is_complete():
            self.color_card_regions = regions
            self._update_color_card_status()
            self._append_log("已加载先前保存的色卡区域配置。")

    def _build_ui(self) -> None:
        self.setWindowTitle(DEFAULT_CONFIG.app_name)
        self.resize(1700, 980)

        central_widget = QWidget()
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([260, 900, 440])

        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central_widget)
        self.statusBar().showMessage("GUI 已初始化。")

    def _build_strawberry_logo(self) -> QIcon:
        """Build a strawberry-themed application logo for the window/taskbar icon."""

        size = 256
        canvas = QPixmap(size, size)
        canvas.fill(Qt.transparent)

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing, True)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#f8f3e9"))
        painter.drawRoundedRect(8, 8, size - 16, size - 16, 48, 48)

        leaf_path = QPainterPath()
        leaf_path.moveTo(128, 56)
        leaf_path.cubicTo(102, 54, 92, 72, 94, 96)
        leaf_path.cubicTo(110, 88, 124, 82, 128, 66)
        leaf_path.cubicTo(132, 82, 146, 88, 162, 96)
        leaf_path.cubicTo(164, 72, 154, 54, 128, 56)
        painter.setBrush(QColor("#2f8f4f"))
        painter.drawPath(leaf_path)

        berry_path = QPainterPath()
        berry_path.moveTo(128, 74)
        berry_path.cubicTo(78, 76, 56, 114, 64, 154)
        berry_path.cubicTo(74, 204, 104, 232, 128, 232)
        berry_path.cubicTo(152, 232, 182, 204, 192, 154)
        berry_path.cubicTo(200, 114, 178, 76, 128, 74)
        painter.setBrush(QColor("#d83a45"))
        painter.setPen(QPen(QColor("#b92f39"), 4))
        painter.drawPath(berry_path)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#f6dd75"))
        for x, y in (
            (104, 120), (128, 114), (151, 121),
            (93, 142), (117, 138), (140, 144), (163, 140),
            (101, 164), (126, 160), (150, 167),
            (111, 186), (136, 190),
        ):
            painter.drawEllipse(x, y, 10, 7)

        painter.end()
        return QIcon(canvas)

    def _load_app_icon(self) -> QIcon:
        """Load static app icon first, fallback to generated logo."""

        if APP_ICON_PATH.exists():
            icon = QIcon(str(APP_ICON_PATH))
            if not icon.isNull():
                return icon
        return self._build_strawberry_logo()

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        source_group = QGroupBox("数据目录")
        source_layout = QVBoxLayout(source_group)
        self.data_dir_label = QLabel("--")
        self.data_dir_label.setWordWrap(True)
        self.total_groups_label = QLabel("样本组数: 0")
        self.complete_groups_label = QLabel("完整组数: 0")
        self.incomplete_groups_label = QLabel("缺图组数: 0")

        select_button = QPushButton("选择数据目录")
        refresh_button = QPushButton("重新扫描")
        select_button.clicked.connect(self._select_data_directory)
        refresh_button.clicked.connect(self._refresh_groups)

        source_layout.addWidget(self.data_dir_label)
        source_layout.addWidget(self.total_groups_label)
        source_layout.addWidget(self.complete_groups_label)
        source_layout.addWidget(self.incomplete_groups_label)
        source_layout.addWidget(select_button)
        source_layout.addWidget(refresh_button)

        sample_group = QGroupBox("样本组列表")
        sample_layout = QVBoxLayout(sample_group)
        self.group_list = QListWidget()
        self.group_list.currentRowChanged.connect(self._on_group_selected)
        sample_layout.addWidget(self.group_list)

        layout.addWidget(source_group)
        layout.addWidget(sample_group, 1)
        return panel

    def _build_center_panel(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.preview_group = QGroupBox("处理流程预览")
        preview_shell_layout = QVBoxLayout(self.preview_group)
        preview_shell_layout.setContentsMargins(10, 12, 10, 10)
        preview_shell_layout.setSpacing(8)

        preview_toolbar = QFrame()
        preview_toolbar.setObjectName("PreviewToolbar")
        toolbar_layout = QVBoxLayout(preview_toolbar)
        toolbar_layout.setContentsMargins(10, 10, 10, 10)
        toolbar_layout.setSpacing(6)

        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)
        self.preview_mode_label = QLabel("当前样本: --    当前阶段: 原始图像")
        self.preview_mode_label.setObjectName("PreviewStatusLabel")
        self.preview_notice_label = QLabel("")
        self.preview_notice_label.setObjectName("PreviewNoticeLabel")
        self.preview_notice_label.setVisible(False)
        status_layout.addWidget(self.preview_mode_label, 1)
        status_layout.addWidget(self.preview_notice_label, 0, Qt.AlignRight)

        switcher_layout = QHBoxLayout()
        switcher_layout.setContentsMargins(0, 0, 0, 0)
        switcher_layout.setSpacing(8)
        self.preview_view_prev_button = QToolButton()
        self.preview_view_prev_button.setObjectName("PreviewNavButton")
        self.preview_view_prev_button.setText("‹")
        self.preview_view_prev_button.clicked.connect(self._show_previous_preview_view)
        self.preview_view_name_label = QLabel("TOP")
        self.preview_view_name_label.setObjectName("PreviewViewChip")
        self.preview_view_name_label.setAlignment(Qt.AlignCenter)
        self.preview_view_next_button = QToolButton()
        self.preview_view_next_button.setObjectName("PreviewNavButton")
        self.preview_view_next_button.setText("›")
        self.preview_view_next_button.clicked.connect(self._show_next_preview_view)
        self.preview_view_index_label = QLabel("1 / 1")
        self.preview_view_index_label.setObjectName("PreviewPageLabel")
        switcher_layout.addWidget(self.preview_view_prev_button)
        switcher_layout.addWidget(self.preview_view_name_label)
        switcher_layout.addWidget(self.preview_view_next_button)
        switcher_layout.addSpacing(6)
        switcher_layout.addWidget(self.preview_view_index_label)
        switcher_layout.addStretch(1)

        toolbar_layout.addLayout(status_layout)
        toolbar_layout.addLayout(switcher_layout)

        preview_layout = QGridLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setHorizontalSpacing(10)
        preview_layout.setVerticalSpacing(10)
        preview_layout.setColumnStretch(0, 1)
        preview_layout.setColumnStretch(1, 1)

        self.orig_preview = ImagePreviewCard("1. 原始图像", preview_height=320)
        self.calib_preview = ImagePreviewCard("2. 颜色校正", preview_height=320)
        self.mask_preview = ImagePreviewCard("3. 背景消除", preview_height=320)
        self.final_preview = ImagePreviewCard("4. 表型提取", preview_height=320)
        # Backward-compatible aliases used by the phenotype preview flow and smoke tests.
        self.top_preview = self.orig_preview
        self.front0_preview = self.calib_preview
        self.front180_preview = self.mask_preview

        for card in (self.orig_preview, self.calib_preview, self.mask_preview, self.final_preview):
            card.set_navigation_handlers(self._show_previous_preview_view, self._show_next_preview_view)

        preview_layout.addWidget(self.orig_preview, 0, 0)
        preview_layout.addWidget(self.calib_preview, 0, 1)
        preview_layout.addWidget(self.mask_preview, 1, 0)
        preview_layout.addWidget(self.final_preview, 1, 1)

        preview_shell_layout.addWidget(preview_toolbar)
        preview_shell_layout.addLayout(preview_layout)

        self.trait_gallery_group = QGroupBox("表型特征总览")
        trait_gallery_layout = QGridLayout(self.trait_gallery_group)
        trait_gallery_layout.setSpacing(12)
        self.trait_leaf_area_preview = ImagePreviewCard("叶面积", preview_height=180)
        self.trait_hull_area_preview = ImagePreviewCard("凸包面积", preview_height=180)
        self.trait_greenness_preview = ImagePreviewCard("绿度", preview_height=180)
        self.trait_canopy_height_preview = ImagePreviewCard("冠层高度", preview_height=180)
        self.trait_canopy_width_preview = ImagePreviewCard("植株冠径", preview_height=180)
        self.trait_side_projection_preview = ImagePreviewCard("侧面投影面积", preview_height=180)
        trait_gallery_layout.addWidget(self.trait_leaf_area_preview, 0, 0)
        trait_gallery_layout.addWidget(self.trait_hull_area_preview, 0, 1)
        trait_gallery_layout.addWidget(self.trait_greenness_preview, 0, 2)
        trait_gallery_layout.addWidget(self.trait_canopy_height_preview, 1, 0)
        trait_gallery_layout.addWidget(self.trait_canopy_width_preview, 1, 1)
        trait_gallery_layout.addWidget(self.trait_side_projection_preview, 1, 2)

        self.trait_focus_title = QLabel("当前性状: --")
        self.trait_focus_title.setStyleSheet("color: #2f6f4f; font-size: 15px; font-weight: 700;")
        self.trait_focus_value = QLabel("数值: --")
        self.trait_focus_views = QLabel("来源视角: --")
        self.trait_focus_status = QLabel("状态: --")
        self.trait_focus_message = QLabel("说明: --")
        self.trait_focus_message.setWordWrap(True)
        self.trait_focus_message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        trait_gallery_layout.addWidget(self.trait_focus_title, 2, 0, 1, 3)
        trait_gallery_layout.addWidget(self.trait_focus_value, 3, 0, 1, 3)
        trait_gallery_layout.addWidget(self.trait_focus_views, 4, 0, 1, 3)
        trait_gallery_layout.addWidget(self.trait_focus_status, 5, 0, 1, 3)
        trait_gallery_layout.addWidget(self.trait_focus_message, 6, 0, 1, 3)
        self.trait_gallery_group.setVisible(False)

        self.batch_pager_widget = QWidget()
        pager_layout = QHBoxLayout(self.batch_pager_widget)
        pager_layout.setContentsMargins(0, 0, 0, 0)
        self.batch_prev_button = QPushButton("上一组")
        self.batch_next_button = QPushButton("下一组")
        self.batch_page_label = QLabel("批量浏览: --")
        self.batch_prev_button.clicked.connect(self._show_previous_batch_preview)
        self.batch_next_button.clicked.connect(self._show_next_batch_preview)
        self.batch_prev_button.setVisible(False)
        self.batch_next_button.setVisible(False)
        self.batch_page_label.setVisible(False)
        pager_layout.addWidget(self.batch_prev_button)
        pager_layout.addWidget(self.batch_next_button)
        pager_layout.addWidget(self.batch_page_label)
        pager_layout.addStretch(1)

        # 色卡区域预览面板已移除（保留属性兼容旧代码/测试）
        self.color_card_group = None
        self.cc_top_preview = None
        self.cc_front0_preview = None
        self.cc_front180_preview = None

        meta_group = QGroupBox("当前样本信息")
        meta_layout = QFormLayout(meta_group)
        self.sample_id_value = QLabel("--")
        self.completeness_value = QLabel("--")
        self.missing_views_value = QLabel("--")
        self.analysis_status_value = QLabel("--")
        self.analysis_message_value = QLabel("--")
        self.segmentation_summary_value = QLabel("--")
        self.debug_output_value = QLabel("--")
        self.top_path_value = QLabel("--")
        self.front0_path_value = QLabel("--")
        self.front180_path_value = QLabel("--")

        for label in (
            self.analysis_message_value,
            self.segmentation_summary_value,
            self.debug_output_value,
            self.top_path_value,
            self.front0_path_value,
            self.front180_path_value,
        ):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        meta_layout.addRow("样本编号", self.sample_id_value)
        meta_layout.addRow("视角完整性", self.completeness_value)
        meta_layout.addRow("缺失视角", self.missing_views_value)
        meta_layout.addRow("分析状态", self.analysis_status_value)
        meta_layout.addRow("分析说明", self.analysis_message_value)
        meta_layout.addRow("TOP 分割摘要", self.segmentation_summary_value)
        meta_layout.addRow("调试输出目录", self.debug_output_value)
        meta_layout.addRow("TOP", self.top_path_value)
        meta_layout.addRow("FRONT-1", self.front0_path_value)
        meta_layout.addRow("FRONT-2", self.front180_path_value)

        layout.addWidget(self.preview_group)
        layout.addWidget(self.batch_pager_widget)
        layout.addWidget(meta_group)
        scroll_area.setWidget(content)
        self._refresh_preview_header()
        return scroll_area

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # === 色卡区域选择面板 ===
        color_card_group = QGroupBox("选择色卡区域")
        color_card_layout = QVBoxLayout(color_card_group)
        color_card_layout.setSpacing(8)
        
        # 色块尺寸设置
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("色块尺寸:"))
        self.card_width_spin = QDoubleSpinBox()
        self.card_width_spin.setRange(0.01, 1000.0)
        self.card_width_spin.setDecimals(2)
        self.card_width_spin.setSuffix(" cm")
        self.card_width_spin.setValue(DEFAULT_PATCH_WIDTH_CM)
        self.card_width_spin.setFixedWidth(90)
        size_layout.addWidget(self.card_width_spin)
        size_layout.addWidget(QLabel("×"))
        self.card_height_spin = QDoubleSpinBox()
        self.card_height_spin.setRange(0.01, 1000.0)
        self.card_height_spin.setDecimals(2)
        self.card_height_spin.setSuffix(" cm")
        self.card_height_spin.setValue(DEFAULT_PATCH_HEIGHT_CM)
        self.card_height_spin.setFixedWidth(90)
        size_layout.addWidget(self.card_height_spin)
        size_layout.addStretch()
        color_card_layout.addLayout(size_layout)

        # 色卡状态
        self.color_card_status_label = QLabel("状态: 未选择色卡区域")
        self.color_card_status_label.setStyleSheet("color: #a05a2c; font-size: 12px;")
        color_card_layout.addWidget(self.color_card_status_label)
        
        # 色卡选择按钮
        self.select_color_card_btn = QPushButton("选择色卡区域")
        self.select_color_card_btn.setToolTip(
            "手动框选三个视角图像中的色卡区域。\n"
            "用于后续色卡定位、颜色校正和基于色块的尺度标定。"
        )
        self.select_color_card_btn.clicked.connect(self._handle_select_color_card_regions)
        color_card_layout.addWidget(self.select_color_card_btn)

        # === 单株作物处理面板 ===
        single_group = QGroupBox("单株作物处理")
        single_layout = QVBoxLayout(single_group)
        single_layout.setSpacing(6)
        
        # 预处理按钮
        self.preprocess_button = QPushButton("预处理")
        self.preprocess_button.setToolTip(
            "在选择的色卡区域内进行精确定位，\n"
            "然后执行颜色校正和基于色块的尺度标定。"
        )
        self.preprocess_button.clicked.connect(self._handle_preprocess)
        single_layout.addWidget(self.preprocess_button)
        
        # 表型提取按钮
        self.single_button = QPushButton("表型提取")
        self.single_button.setToolTip(
            "对当前选中的样本执行表型分析。\n"
            "需要先完成预处理步骤。\n"
            "分析完成后自动保存结果。"
        )
        self.single_button.clicked.connect(self._handle_single_analysis)
        single_layout.addWidget(self.single_button)

        # === 调试模式面板 ===
        debug_group = QGroupBox("调试选项")
        debug_layout = QVBoxLayout(debug_group)
        debug_layout.setSpacing(6)
        
        self.debug_mode_checkbox = QCheckBox("启用调试模式")
        self.debug_mode_checkbox.setToolTip(
            "启用后，预处理和分析过程中会在 output/调试输出 目录下\n"
            "保存每个处理步骤的可视化图像，包括色卡检测、图像分割、\n"
            "各表型计算的中间结果，方便排查问题。"
        )
        self.debug_mode_checkbox.setChecked(False)
        debug_layout.addWidget(self.debug_mode_checkbox)

        # === 批量处理面板 ===
        batch_group = QGroupBox("批量处理")
        batch_layout = QVBoxLayout(batch_group)
        batch_layout.setSpacing(6)
        
        self.batch_button = QPushButton("批量处理")
        self.batch_button.setToolTip(
            "对当前目录下所有样本执行批量处理。\n"
            "需要先完成预处理步骤。\n"
            "分析完成后自动保存结果。"
        )
        self.batch_button.clicked.connect(self._handle_batch_analysis)
        batch_layout.addWidget(self.batch_button)

        # === 表型结果预览 ===
        results_group = QGroupBox("表型测量结果")
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(6, 12, 6, 6)
        self.results_table = QTableWidget(len(TRAIT_SPECS), 3)
        self.results_table.setHorizontalHeaderLabels(["性状", "数值", "单位"])
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMaximumHeight(320)
        self.results_table.currentCellChanged.connect(self._on_trait_row_changed)
        self._set_result_rows()
        results_layout.addWidget(self.results_table)

        # === 调试日志 ===
        log_group = QGroupBox("调试日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(6, 12, 6, 6)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("这里会显示目录扫描、分析进度和导出日志。")
        self.log_view.setMaximumHeight(160)
        log_layout.addWidget(self.log_view)

        # 隐藏的重新选择按钮（保持兼容性但不显示）
        self.reselect_color_card_btn = QPushButton()
        self.reselect_color_card_btn.setVisible(False)
        
        # 隐藏的导出按钮（保持兼容性但不显示，导出改为自动）
        self.export_button = QPushButton()
        self.export_button.setVisible(False)

        layout.addWidget(color_card_group)
        layout.addWidget(debug_group)
        layout.addWidget(single_group)
        layout.addWidget(batch_group)
        layout.addWidget(results_group)
        layout.addWidget(log_group)
        layout.addStretch()
        return panel

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #f4f0e6;
                color: #21312a;
                font-size: 13px;
            }
            QMainWindow, QGroupBox, QFrame, QScrollArea {
                background: #fbf8f1;
            }
            QGroupBox {
                border: 1px solid #d4c4a8;
                border-radius: 12px;
                margin-top: 12px;
                font-weight: 600;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px 0 4px;
                color: #5f3b23;
            }
            QPushButton {
                background: #2f6f4f;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 14px;
                font-weight: 600;
            }
            QPushButton:hover:!disabled {
                background: #24573d;
            }
            QPushButton:disabled {
                background: #8ca394;
                color: #eff4f0;
            }
            QListWidget, QPlainTextEdit, QTableWidget, QDoubleSpinBox {
                background: white;
                border: 1px solid #d9d2c2;
                border-radius: 10px;
            }
            QLabel#CardTitle {
                font-size: 16px;
                font-weight: 700;
                color: #2f6f4f;
            }
            QLabel#MetaLabel {
                color: #6a6f67;
            }
            QFrame#PreviewToolbar {
                border: 1px solid #d9cab0;
                border-radius: 12px;
                background: #fbf8f1;
            }
            QLabel#PreviewStatusLabel {
                color: #5f3b23;
                font-weight: 700;
            }
            QLabel#PreviewNoticeLabel {
                color: #a05a2c;
                font-size: 12px;
            }
            QLabel#PreviewViewChip {
                min-width: 84px;
                padding: 4px 10px;
                border-radius: 999px;
                background: #e7efe8;
                border: 1px solid #8aaa91;
                color: #2f6f4f;
                font-weight: 700;
            }
            QLabel#PreviewPageLabel {
                color: #6a6f67;
                font-weight: 600;
            }
            QToolButton#PreviewNavButton, QToolButton#CardNavButton {
                background: white;
                border: 1px solid #8aaa91;
                border-radius: 12px;
                color: #2f6f4f;
                font-weight: 700;
                min-width: 24px;
                min-height: 24px;
            }
            QToolButton#PreviewNavButton:hover:!disabled, QToolButton#CardNavButton:hover:!disabled {
                background: #eef5ef;
            }
            QToolButton#PreviewNavButton:disabled, QToolButton#CardNavButton:disabled {
                color: #96a599;
                border-color: #cfd8cf;
                background: #f8faf8;
            }
            QHeaderView::section {
                background: #efe4cf;
                color: #4b3728;
                padding: 6px;
                border: none;
                border-bottom: 1px solid #d4c4a8;
            }
            """
        )

    def _set_result_rows(self, result: PlantAnalysisResult | None = None) -> None:
        if result is None:
            rows = [(spec.label, "--", spec.unit) for spec in TRAIT_SPECS]
        else:
            rows = [(trait.label, trait.display_value, trait.unit) for trait in result.traits]

        for row_index, row_values in enumerate(rows):
            for column_index, value in enumerate(row_values):
                self.results_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))

    def _on_trait_row_changed(self, current_row: int, _current_column: int, _previous_row: int, _previous_column: int) -> None:
        """Link the selected trait row to the center phenotype preview panel."""

        if current_row < 0 or current_row >= len(TRAIT_SPECS):
            return
        self._apply_trait_focus_by_row(current_row)

    def _apply_trait_focus_by_row(self, row_index: int) -> None:
        """Update trait summary and highlighted preview according to the selected row."""

        if row_index < 0 or row_index >= len(TRAIT_SPECS):
            return

        spec = TRAIT_SPECS[row_index]
        self._active_trait_key = spec.key

        if self.displayed_result is None:
            self._update_trait_focus_panel(
                title=spec.label,
                value_text=f"数值: -- {spec.unit}",
                source_views=spec.source_views,
                status_text="状态: --",
                message="尚未得到该性状的分析结果。",
            )
            self._set_active_trait_preview(spec.key)
            return

        trait_map = self.displayed_result.trait_map()
        trait = trait_map.get(spec.key)
        if trait is None:
            return

        self._update_trait_focus_panel(
            title=trait.label,
            value_text=f"数值: {trait.display_value} {trait.unit}",
            source_views=trait.source_views,
            status_text=f"状态: {format_status_label(trait.status)}",
            message=trait.message,
        )
        self._set_active_trait_preview(trait.key)

    def _update_trait_focus_panel(
        self,
        *,
        title: str,
        value_text: str,
        source_views: tuple[str, ...],
        status_text: str,
        message: str,
    ) -> None:
        """Render the currently focused trait information in the center panel."""

        self.trait_focus_title.setText(f"当前性状: {title}")
        self.trait_focus_value.setText(value_text)
        self.trait_focus_views.setText(f"来源视角: {' / '.join(source_views)}")
        self.trait_focus_status.setText(status_text)
        self.trait_focus_message.setText(f"说明: {message}")

    def _set_active_trait_preview(self, trait_key: str | None) -> None:
        """Highlight the preview card that corresponds to the selected trait."""

        for key, card in self._trait_preview_cards().items():
            card.set_selected(key == trait_key)

    def _trait_preview_cards(self) -> dict[str, ImagePreviewCard]:
        """Return the mapping between trait keys and preview cards."""

        return {
            "leaf_area": self.trait_leaf_area_preview,
            "convex_hull_area": self.trait_hull_area_preview,
            "greenness": self.trait_greenness_preview,
            "canopy_height": self.trait_canopy_height_preview,
            "canopy_width": self.trait_canopy_width_preview,
            "side_projection_area": self.trait_side_projection_preview,
        }

    def _clear_debug_previews(self) -> None:
        """已弃用 - 调试预览功能已移除，由调试输出目录保存。"""
        pass

    def _apply_debug_previews(self, previews: AnalysisDebugPreviews | None) -> None:
        """已弃用 - 调试预览功能已移除，由调试输出目录保存。"""
        pass

    def _handle_select_color_card_regions(self) -> None:
        """打开色卡区域选择对话框，让用户手动框选三个视角的色卡位置。"""
        group = self._selected_group()
        if group is None:
            QMessageBox.information(self, "提示", "请先在左侧列表中选择一个样本组，再选择色卡区域。")
            return

        # 检查视角完整性
        if not group.is_complete:
            QMessageBox.warning(
                self, "视角不完整",
                f"当前样本缺少部分视角图像（{', '.join(group.missing_views)}），\n"
                "请选择一个图像完整的样本组后再进行色卡区域选择。"
            )
            return

        import cv2 as cv
        
        # 加载三个视角的图像
        try:
            top_image = cv.imread(str(group.top_image))
            front_1_image = cv.imread(str(group.front_0_image))
            front_2_image = cv.imread(str(group.front_180_image))
            
            if top_image is None or front_1_image is None or front_2_image is None:
                QMessageBox.warning(self, "图像加载失败", "无法加载一个或多个视角图像，请检查文件是否存在。")
                return
        except Exception as e:
            QMessageBox.warning(self, "图像加载失败", f"加载图像时发生错误: {e}")
            return

        self._append_log("开始色卡区域选择...")
        regions = select_color_card_regions_interactive(
            top_image=top_image,
            front_1_image=front_1_image,
            front_2_image=front_2_image,
            existing_regions=self.color_card_regions,
            parent=self,
        )

        if regions is None:
            self._append_log("色卡区域选择被取消。", level="WARNING")
            return

        # 保存选择结果
        self.color_card_regions = regions
        save_color_card_regions(regions, self.color_card_config_path)
        self._append_log(f"色卡区域配置已保存到: {self.color_card_config_path}")

        # 更新UI状态
        self._update_color_card_status()
        self._update_color_card_preview()

    def _update_color_card_status(self) -> None:
        """更新色卡选择状态标签。"""
        if self.color_card_regions is not None:
            self.color_card_status_label.setText("状态: 已选择色卡区域 ✓")
            self.color_card_status_label.setStyleSheet("color: #2f6f4f; font-weight: 600; font-size: 12px;")
            self.select_color_card_btn.setText("重新选择色卡区域")
        else:
            self.color_card_status_label.setText("状态: 未选择色卡区域")
            self.color_card_status_label.setStyleSheet("color: #a05a2c; font-size: 12px;")
            self.select_color_card_btn.setText("选择色卡区域")

    def _update_color_card_preview(self) -> None:
        """色卡区域预览面板已移除，此方法保留为空以兼容调用链。"""
        return

    def _current_color_card_reference(self) -> Any:
        """Build the current color-card reference from patch-size input."""

        return create_color_card_reference(
            patch_width_cm=self.card_width_spin.value(),
            patch_height_cm=self.card_height_spin.value(),
        )

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        """Enable or disable interactive controls during background work."""

        self.single_button.setDisabled(busy)
        self.batch_button.setDisabled(busy)
        self.export_button.setDisabled(busy)
        self.preprocess_button.setDisabled(busy)
        self.group_list.setDisabled(busy)
        self.card_width_spin.setDisabled(busy)
        self.card_height_spin.setDisabled(busy)
        self.select_color_card_btn.setDisabled(busy)
        if message is not None:
            self.statusBar().showMessage(message)

    def _select_data_directory(self) -> None:
        selected_dir = QFileDialog.getExistingDirectory(self, "选择草莓图像目录", str(self.current_directory))
        if selected_dir:
            self._load_groups(Path(selected_dir))

    def _refresh_groups(self) -> None:
        self._load_groups(self.current_directory)

    def _load_groups(self, directory: Path) -> None:
        try:
            self.groups = group_image_files(directory)
        except (FileNotFoundError, NotADirectoryError, ValueError) as error:
            QMessageBox.warning(self, "目录扫描失败", str(error))
            self._append_log(f"扫描目录失败: {error}", level="WARNING")
            return

        self.current_directory = directory
        self.current_batch_report = None
        self.current_result = None
        self.displayed_result = None
        self.data_dir_label.setText(str(directory))

        incomplete_count = len(find_incomplete_groups(self.groups))
        self.total_groups_label.setText(f"样本组数: {len(self.groups)}")
        self.complete_groups_label.setText(f"完整组数: {len(self.groups) - incomplete_count}")
        self.incomplete_groups_label.setText(f"缺图组数: {incomplete_count}")

        self.group_list.clear()
        for group in self.groups:
            suffix = "完整" if group.is_complete else f"缺失 {', '.join(group.missing_views)}"
            self.group_list.addItem(QListWidgetItem(f"{group.sample_id}  [{suffix}]"))

        if self.groups:
            self.group_list.setCurrentRow(0)
        else:
            self._clear_current_group()

        self._append_log(f"已加载目录 {directory}，识别到 {len(self.groups)} 个样本组。")
        self._report_grouping_issues(directory)
        self.statusBar().showMessage(f"当前目录: {directory}")

    def _report_grouping_issues(self, directory: Path) -> None:
        """Show only unrecognized filenames for manual renaming."""

        suggestions = collect_grouping_suggestions(directory)
        if not suggestions:
            return

        message = self._build_grouping_issue_message(suggestions)
        for line in message.splitlines():
            if line.strip():
                self._append_log(line, level="WARNING")

        QMessageBox.information(self, "发现需要重命名的文件", message)

    def _build_grouping_issue_message(
        self,
        suggestions: list[GroupingSuggestion],
    ) -> str:
        """Build a minimal rename guidance message after directory scan."""

        lines = [
            "以下图片名称无法识别，请修改后重新扫描：",
            "",
        ]

        for item in suggestions:
            lines.append(f"- {item.image_path.name}")

        lines.extend(
            [
                "",
                "正确格式: 12AB_TOP.png / 12AB-1.png / 12AB-2.png",
                "样本编号只能是 数字+AB，例如 4AB；4A、4B 都是错误命名。",
            ]
        )
        return "\n".join(lines)

    def _on_group_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.groups):
            self._clear_current_group()
            return

        group = self.groups[row]
        self.current_result = None
        self.displayed_result = None
        self.current_batch_report = None
        self._update_group_view(group)
        self._append_log(f"切换到样本组: {group.sample_id}")

    def _update_group_view(self, group: PlantImageGroup) -> None:
        self.sample_id_value.setText(group.sample_id)
        self.completeness_value.setText("完整" if group.is_complete else "不完整")
        self.missing_views_value.setText(", ".join(group.missing_views) if group.missing_views else "无")
        self.analysis_status_value.setText("未分析")
        self.analysis_message_value.setText("尚未执行分析。")
        self.segmentation_summary_value.setText("--")
        self.debug_output_value.setText("--")
        self.top_path_value.setText(str(group.top_image) if group.top_image else "--")
        self.front0_path_value.setText(str(group.front_0_image) if group.front_0_image else "--")
        self.front180_path_value.setText(str(group.front_180_image) if group.front_180_image else "--")

        if self.preview_mode == "phenotype" and self._has_previewable_phenotype_result(group.sample_id):
            self._show_trait_preview_for_sample(group.sample_id)
        elif self.preview_mode == "preprocess" and self.preprocess_result is not None and self.preprocess_result.sample_id == group.sample_id:
            self._show_preprocess_preview(self.preprocess_result)
        else:
            self._show_original_preview_for_group(group)
        self._set_result_rows()
        self._clear_debug_previews()
        
        # 如果有缓存的色卡区域，显示预览
        self._update_color_card_preview()

    def _clear_current_group(self) -> None:
        self.current_result = None
        self.displayed_result = None
        self.current_batch_report = None
        self.sample_id_value.setText("--")
        self.completeness_value.setText("--")
        self.missing_views_value.setText("--")
        self.analysis_status_value.setText("--")
        self.analysis_message_value.setText("--")
        self.segmentation_summary_value.setText("--")
        self.debug_output_value.setText("--")
        self.top_path_value.setText("--")
        self.front0_path_value.setText("--")
        self.front180_path_value.setText("--")
        self._set_preview_state(sample_id=None, mode="original")
        self._render_current_preview()
        self._set_batch_pager_visible(False)
        self._set_trait_gallery_visible(False)
        self._set_result_rows()
        self._clear_debug_previews()

    def _selected_group(self) -> PlantImageGroup | None:
        current_row = self.group_list.currentRow()
        if current_row < 0 or current_row >= len(self.groups):
            return None
        return self.groups[current_row]

    def _handle_preprocess(self) -> None:
        """执行色卡预处理：色卡定位、颜色校正和尺度标定。"""
        if self.preprocess_thread is not None and self.preprocess_thread.isRunning():
            return

        group = self._selected_group()
        if group is None:
            QMessageBox.information(self, "提示", "请先选择一个样本组。")
            return

        if self.color_card_regions is None:
            QMessageBox.warning(
                self, "未选择色卡区域",
                '请先点击"选择色卡区域"按钮框选三个视角的色卡位置。'
            )
            return

        self._append_log(f"开始预处理样本 {group.sample_id}...")
        self._append_log(
            f"色卡尺寸: {self.card_width_spin.value():.2f} x {self.card_height_spin.value():.2f} cm"
        )
        self._set_busy(True, f"正在预处理样本 {group.sample_id}...")

        # 清除上次预处理结果（如果样本ID不同）
        if self.preprocess_result is not None and self.preprocess_result.sample_id != group.sample_id:
            self.preprocess_result = None

        debug_enabled = self.debug_mode_checkbox.isChecked()
        debug_output_dir = self._current_visualization_output_dir(debug_enabled)

        self.preprocess_thread = PreprocessThread(
            group=group,
            color_card_regions=self.color_card_regions,
            calibration_reference=self._current_color_card_reference(),
            debug_output_dir=debug_output_dir,
            save_full_debug=debug_enabled,
            parent=self,
        )
        self.preprocess_thread.log_message.connect(self._append_log)
        self.preprocess_thread.status_message.connect(self.statusBar().showMessage)
        self.preprocess_thread.finished_signal.connect(self._on_preprocess_finished)
        self.preprocess_thread.failed.connect(self._on_preprocess_failed)
        self.preprocess_thread.finished.connect(self._on_preprocess_thread_finished)
        self.preprocess_thread.start()

    def _on_preprocess_finished(self, result: PreprocessResult) -> None:
        """处理预处理完成。"""
        self.preprocess_result = result
        if result.is_valid:
            self._append_log(f"预处理完成: {result.message}")
            self.statusBar().showMessage(f"样本 {result.sample_id} 预处理完成")
            # 更新色卡状态显示
            calibrated_count = sum(
                1 for cal in result.calibration_results.values()
                if getattr(cal, "is_calibrated", False)
            )
            self.color_card_status_label.setText(
                f"状态: 预处理完成 ({calibrated_count}/3 视角校准成功)"
            )
            self.color_card_status_label.setStyleSheet("color: #2d8a4e; font-size: 12px;")
            self._show_preprocess_preview(result)
        else:
            self._append_log(f"预处理失败: {result.message}", level="ERROR")
            QMessageBox.warning(self, "预处理失败", result.message)

    def _on_preprocess_failed(self, error_message: str) -> None:
        """处理预处理异常。"""
        self._append_log(f"预处理异常: {error_message}", level="ERROR")
        self.statusBar().showMessage("预处理异常终止。")
        QMessageBox.warning(self, "预处理失败", error_message)

    def _on_preprocess_thread_finished(self) -> None:
        """预处理线程完成后恢复UI。"""
        self._set_busy(False, "就绪")
        self.preprocess_thread = None

    def _handle_single_analysis(self) -> None:
        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            return

        group = self._selected_group()
        if group is None:
            QMessageBox.information(self, "提示", "请先选择一个样本组。")
            return

        # 检查是否已完成预处理
        if self.preprocess_result is None or self.preprocess_result.sample_id != group.sample_id:
            QMessageBox.warning(
                self, "未完成预处理",
                '请先完成预处理步骤：\n\n'
                '1. 点击"选择色卡区域"框选三个视角的色卡\n'
                '2. 点击"预处理"进行色卡定位和颜色校正'
            )
            return

        self.current_batch_report = None
        debug_enabled = self.debug_mode_checkbox.isChecked()
        debug_status = "已启用" if debug_enabled else "未启用"
        self._append_log(
            f"准备执行表型提取: {group.sample_id}，"
            f"使用预处理结果，调试模式: {debug_status}"
        )
        self._set_busy(True, f"正在分析样本 {group.sample_id}...")

        # 无论是否调试模式，都保存运行可视化；调试模式保存完整过程
        debug_output_dir = self._current_visualization_output_dir(True) if debug_enabled else None

        # 构建预处理数据供 pipeline 使用
        precomputed_calibration = {
            "calibrated_images": self.preprocess_result.calibrated_images,
            "calibration_results": self.preprocess_result.calibration_results,
        }

        self.analysis_thread = AnalysisThread(
            mode="single",
            group=group,
            calibration_reference=self._current_color_card_reference(),
            debug_output_dir=debug_output_dir,
            color_card_regions=self.color_card_regions,
            precomputed_calibration=precomputed_calibration,
            parent=self,
        )
        self.analysis_thread.log_message.connect(self._append_log)
        self.analysis_thread.status_message.connect(self.statusBar().showMessage)
        self.analysis_thread.single_finished.connect(self._on_single_analysis_finished)
        self.analysis_thread.failed.connect(self._on_analysis_failed)
        self.analysis_thread.finished.connect(self._on_analysis_thread_finished)
        self.analysis_thread.start()

    def _on_single_analysis_finished(self, result: PlantAnalysisResult) -> None:
        """Apply the single-sample analysis result on the UI thread."""

        self.current_result = result
        self._apply_analysis_result(result)

        if result.status in {"load_failed", "incomplete_input", "dependency_error", "segmentation_failed"}:
            QMessageBox.warning(self, "分析未完成", result.message)
            self.statusBar().showMessage(f"样本 {result.sample_id} 分析未完成。")
            return

        # 自动保存结果
        self._auto_save_single_result(result)
        self.statusBar().showMessage(f"样本 {result.sample_id} 分析完成，结果已自动保存。")

    def _auto_save_single_result(self, result: PlantAnalysisResult) -> None:
        """自动保存单样本分析结果。"""
        try:
            export_dir = self._current_export_dir()
            export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = export_dir / f"sample_{result.sample_id}_{timestamp}.xlsx"
            
            selected_group = self._selected_group()
            if selected_group is not None:
                exported_path = export_single_result(
                    selected_group,
                    result,
                    str(export_path),
                    include_debug_fields=False,
                )
                self._append_log(f"结果已自动保存到: {exported_path}")
        except Exception as error:  # noqa: BLE001
            self._append_log(f"自动保存失败: {error}", level="ERROR")

    def _apply_analysis_result(self, result: PlantAnalysisResult) -> None:
        self.analysis_status_value.setText(format_status_label(result.status))
        self.analysis_message_value.setText(result.message)
        self._set_displayed_result(result)
        self._apply_debug_previews(build_analysis_debug_previews(result))

        artifact_paths = next((paths for paths in result.debug_artifact_paths.values() if paths), None)
        self.debug_output_value.setText(str(artifact_paths[0].parent.parent) if artifact_paths else "--")

        top_segmentation = result.top_segmentation
        if top_segmentation is None:
            self.segmentation_summary_value.setText("--")
        else:
            self.segmentation_summary_value.setText(
                f"mask_area={top_segmentation.mask_area_pixels}, "
                f"contours={top_segmentation.contour_count}, "
                f"hull_area={top_segmentation.hull_area_pixels:.1f}"
            )

        for view_name, view_result in result.view_results.items():
            if view_result.shape is not None:
                self._append_log(f"{view_name} 输入摘要: status={view_result.status}, shape={view_result.shape}")

        self._show_trait_preview_for_sample(result.sample_id)

    def _handle_batch_analysis(self) -> None:
        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            return

        if not self.groups:
            QMessageBox.information(self, "提示", "当前目录没有可分析的样本组。")
            return

        # 检查是否已完成预处理
        if self.color_card_regions is None:
            QMessageBox.warning(
                self, "未完成预处理",
                '请先完成色卡预处理步骤：\n\n'
                '1. 点击"选择色卡区域"框选三个视角的色卡\n'
                '2. 点击"预处理"进行颜色校正和尺度标定'
            )
            return

        self.current_result = None
        debug_enabled = self.debug_mode_checkbox.isChecked()
        debug_status = "已启用" if debug_enabled else "未启用"
        self._append_log(
            f"开始批量处理，色卡尺寸={self.card_width_spin.value():.2f} x {self.card_height_spin.value():.2f} cm，"
            f"调试模式: {debug_status}"
        )
        self._set_busy(True, "正在批量处理...")

        # 批量处理视为预处理+表型提取结合，始终保存可视化结果
        debug_output_dir = self._current_visualization_output_dir(debug_enabled)

        self.analysis_thread = AnalysisThread(
            mode="batch",
            groups=self.groups.copy(),
            directory=self.current_directory,
            calibration_reference=self._current_color_card_reference(),
            debug_output_dir=debug_output_dir,
            color_card_regions=self.color_card_regions,
            parent=self,
        )
        self.analysis_thread.log_message.connect(self._append_log)
        self.analysis_thread.status_message.connect(self.statusBar().showMessage)
        self.analysis_thread.batch_finished.connect(self._on_batch_analysis_finished)
        self.analysis_thread.failed.connect(self._on_analysis_failed)
        self.analysis_thread.finished.connect(self._on_analysis_thread_finished)
        self.analysis_thread.start()

    def _on_batch_analysis_finished(self, report: BatchAnalysisReport) -> None:
        """Apply the batch analysis summary on the UI thread."""

        self.current_batch_report = report
        summary_message = (
            f"批量处理完成。总样本 {report.total_groups} 组，"
            f"成功 {report.completed_groups} 组，"
            f"跳过 {report.skipped_groups} 组，"
            f"失败 {report.failed_groups} 组。"
        )
        self._append_log(summary_message)

        self.batch_preview_sample_ids = [
            item.group.sample_id
            for item in report.sample_results
            if item.result.status == "analysis_complete"
        ]
        self.batch_preview_index = 0
        if self.batch_preview_sample_ids:
            self._show_trait_preview_for_sample(self.batch_preview_sample_ids[0])
            self._set_batch_pager_visible(True)
            self._update_batch_page_label()
        else:
            self._set_batch_pager_visible(False)
        
        # 自动保存批量结果
        self._auto_save_batch_result(report)
        self.statusBar().showMessage(f"{summary_message} 结果已自动保存。")
        QMessageBox.information(self, "批量处理", f"{summary_message}\n结果已自动保存。")

    def _auto_save_batch_result(self, report: BatchAnalysisReport) -> None:
        """自动保存批量分析结果。"""
        try:
            export_dir = self._current_export_dir()
            export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = export_dir / f"batch_results_{timestamp}.xlsx"
            
            exported_path = export_batch_report(
                report,
                str(export_path),
                include_debug_fields=False,
            )
            self._append_log(f"批量结果已自动保存到: {exported_path}")
        except Exception as error:  # noqa: BLE001
            self._append_log(f"自动保存失败: {error}", level="ERROR")

    def _on_analysis_failed(self, error_message: str) -> None:
        """Handle unexpected background-task failure."""

        self._append_log(f"后台分析异常: {error_message}", level="ERROR")
        self.statusBar().showMessage("分析异常终止。")
        QMessageBox.warning(self, "分析失败", error_message)

    def _on_analysis_thread_finished(self) -> None:
        """Restore the UI after a background analysis task finishes."""

        self._set_busy(False, "就绪")
        self.analysis_thread = None

    def _handle_export(self) -> None:
        if self.current_batch_report is None and self.current_result is None:
            QMessageBox.information(self, "提示", "请先执行单样本或批量分析，再导出结果。")
            return

        export_dir = self._current_export_dir()
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = (
            f"batch_results_{timestamp}.xlsx"
            if self.current_batch_report is not None
            else f"sample_result_{timestamp}.xlsx"
        )
        selected_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出表型结果",
            str(export_dir / default_name),
            "Excel Workbook (*.xlsx);;CSV File (*.csv)",
        )
        if not selected_path:
            return

        try:
            if self.current_batch_report is not None:
                exported_path = export_batch_report(
                    self.current_batch_report,
                    selected_path,
                    include_debug_fields=False,
                )
            else:
                selected_group = self._selected_group()
                if selected_group is None or self.current_result is None:
                    raise ValueError("No current sample result available for export.")
                exported_path = export_single_result(
                    selected_group,
                    self.current_result,
                    selected_path,
                    include_debug_fields=False,
                )
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(error))
            self._append_log(f"导出失败: {error}", level="ERROR")
            return

        self._append_log(f"结果已导出到: {exported_path}")
        self.statusBar().showMessage(f"已导出结果: {exported_path}")
        QMessageBox.information(self, "导出完成", f"结果已导出到:\n{exported_path}")

    def _current_export_dir(self) -> Path:
        """Return export directory by mode so formal/debug files do not mix."""

        if self.debug_mode_checkbox.isChecked():
            return (DEFAULT_CONFIG.output_dir / "exports" / "调试模式").resolve()
        return (DEFAULT_CONFIG.output_dir / "exports").resolve()

    def _append_log(self, message: str, *, level: str = "INFO") -> None:
        getattr(self.logger, level.lower())(message)
        self.log_view.appendPlainText(f"[{level}] {message}")

    def _current_visualization_output_dir(self, debug_enabled: bool) -> Path:
        """Return output root for runtime visualizations by mode."""

        if debug_enabled:
            return DEFAULT_CONFIG.output_dir / "调试输出"
        return DEFAULT_CONFIG.output_dir / "运行可视化"

    def _find_group_by_sample_id(self, sample_id: str | None) -> PlantImageGroup | None:
        if sample_id is None:
            return None
        for group in self.groups:
            if group.sample_id == sample_id:
                return group
        return None

    def _image_path_for_view(self, group: PlantImageGroup | None, view_name: str) -> Path | None:
        if group is None:
            return None
        if view_name == "TOP":
            return group.top_image
        if view_name == "FRONT-1":
            return group.front_0_image
        if view_name == "FRONT-2":
            return group.front_180_image
        return None

    def _set_preview_state(self, *, sample_id: str | None, mode: str) -> PlantImageGroup | None:
        group = self._find_group_by_sample_id(sample_id) if sample_id is not None else self._selected_group()
        resolved_sample_id = sample_id if sample_id is not None else (group.sample_id if group is not None else None)
        active_view, fallback_used = pick_active_view(group, self.preview_view_state.view_name)
        fallback_notice = ""
        if fallback_used and resolved_sample_id is not None:
            fallback_notice = "当前视角不可用，已自动切换到可用视角。"

        self.preview_view_state = PreviewViewState(
            sample_id=resolved_sample_id,
            mode=mode,
            view_name=active_view,
            available_views=available_views_for_group(group),
            fallback_notice=fallback_notice,
        )
        self.preview_mode = mode
        self._refresh_preview_header()
        return group

    def _refresh_preview_header(self) -> None:
        mode_labels = {
            "original": "原始图像",
            "preprocess": "预处理结果",
            "phenotype": "表型提取结果",
        }
        state = self.preview_view_state
        sample_text = state.sample_id or "--"
        mode_text = mode_labels.get(state.mode, "原始图像")
        available_views = state.available_views or ("TOP",)
        if state.view_name in available_views:
            current_index = available_views.index(state.view_name) + 1
        else:
            current_index = 1

        self.preview_mode_label.setText(f"当前样本: {sample_text}    当前阶段: {mode_text}")
        self.preview_view_name_label.setText(state.view_name)
        self.preview_view_index_label.setText(f"{current_index} / {len(available_views)}")
        self.preview_notice_label.setText(state.fallback_notice)
        self.preview_notice_label.setVisible(bool(state.fallback_notice))

        can_step = len(available_views) > 1
        self.preview_view_prev_button.setEnabled(can_step)
        self.preview_view_next_button.setEnabled(can_step)
        for card in (self.orig_preview, self.calib_preview, self.mask_preview, self.final_preview):
            card.set_navigation_enabled(can_step)

    def _show_previous_preview_view(self) -> None:
        self._step_preview_view(-1)

    def _show_next_preview_view(self) -> None:
        self._step_preview_view(1)

    def _step_preview_view(self, direction: int) -> None:
        state = self.preview_view_state
        if len(state.available_views) <= 1:
            return

        next_view = step_view(state.view_name, state.available_views, direction)
        self.preview_view_state = PreviewViewState(
            sample_id=state.sample_id,
            mode=state.mode,
            view_name=next_view,
            available_views=state.available_views,
            fallback_notice="",
        )
        self._refresh_preview_header()
        self._render_current_preview()

    def _render_current_preview(self) -> None:
        state = self.preview_view_state
        group = self._find_group_by_sample_id(state.sample_id) if state.sample_id is not None else self._selected_group()
        if group is None:
            self._render_stage_payloads(
                original=StagePreviewPayload(placeholder_text="等待选择样本", status_text="等待选择样本"),
                calibrated=StagePreviewPayload(placeholder_text="等待颜色校正", status_text="等待颜色校正"),
                masked=StagePreviewPayload(placeholder_text="等待背景消除", status_text="等待背景消除"),
                final=StagePreviewPayload(placeholder_text="等待表型提取", status_text="等待表型提取"),
            )
            return

        if state.mode == "preprocess":
            payloads = self._build_preprocess_stage_payloads(group, group.sample_id, state.view_name)
        elif state.mode == "phenotype":
            payloads = self._build_phenotype_stage_payloads(group, state.sample_id or group.sample_id, state.view_name)
        else:
            payloads = self._build_original_stage_payloads(group, state.view_name)

        self._render_stage_payloads(**payloads)

    def _render_stage_payloads(
        self,
        *,
        original: StagePreviewPayload,
        calibrated: StagePreviewPayload,
        masked: StagePreviewPayload,
        final: StagePreviewPayload,
    ) -> None:
        self._apply_stage_payload(self.orig_preview, original)
        self._apply_stage_payload(self.calib_preview, calibrated)
        self._apply_stage_payload(self.mask_preview, masked)
        self._apply_stage_payload(self.final_preview, final)

    def _apply_stage_payload(self, card: ImagePreviewCard, payload: StagePreviewPayload) -> None:
        card.set_viewer_title(f"{card.title.text()} - {self.preview_view_state.view_name}")
        if payload.image_array is not None:
            card.set_image_array(payload.image_array, meta_text=payload.status_text or "已生成")
            return
        if payload.image_path is not None:
            card.set_image_path(payload.image_path, meta_text=payload.status_text or "已加载")
            return
        placeholder = payload.placeholder_text or "等待加载图像"
        card.clear(placeholder, payload.status_text or placeholder)

    def _raw_stage_payload(self, group: PlantImageGroup, view_name: str) -> StagePreviewPayload:
        raw_path = self._image_path_for_view(group, view_name)
        if raw_path is not None:
            return StagePreviewPayload(image_path=raw_path, status_text="已加载")
        return StagePreviewPayload(placeholder_text="缺少该视角图像", status_text="缺少该视角图像")

    def _resolve_calibrated_payload(
        self,
        *,
        group: PlantImageGroup,
        sample_id: str,
        view_name: str,
        sample_root: Path | None = None,
    ) -> StagePreviewPayload:
        if self.preprocess_result is not None and self.preprocess_result.sample_id == sample_id:
            calibrated_image = self.preprocess_result.calibrated_images.get(view_name)
            if calibrated_image is not None:
                return StagePreviewPayload(image_array=calibrated_image, status_text="已校正")

        if sample_root is not None:
            calibrated_path = self._pick_calibration_image(sample_root, view_name)
            if calibrated_path is not None:
                return StagePreviewPayload(image_path=calibrated_path, status_text="已校正")

            comparison_path = self._pick_calibration_comparison_image(sample_root, view_name)
            if comparison_path is not None:
                corrected_half = self._extract_corrected_half_from_comparison(comparison_path)
                if corrected_half is not None:
                    return StagePreviewPayload(image_array=corrected_half, status_text="已校正")

        raw_path = self._image_path_for_view(group, view_name)
        if raw_path is not None:
            return StagePreviewPayload(image_path=raw_path, status_text="未找到整图校正结果")
        return StagePreviewPayload(placeholder_text="缺少该视角图像", status_text="缺少该视角图像")

    def _build_original_stage_payloads(
        self,
        group: PlantImageGroup,
        view_name: str,
    ) -> dict[str, StagePreviewPayload]:
        return {
            "original": self._raw_stage_payload(group, view_name),
            "calibrated": StagePreviewPayload(placeholder_text="等待颜色校正", status_text="等待颜色校正"),
            "masked": StagePreviewPayload(placeholder_text="等待背景消除", status_text="等待背景消除"),
            "final": StagePreviewPayload(placeholder_text="等待表型提取", status_text="等待表型提取"),
        }

    def _build_preprocess_stage_payloads(
        self,
        group: PlantImageGroup,
        sample_id: str,
        view_name: str,
    ) -> dict[str, StagePreviewPayload]:
        sample_root = self._find_latest_sample_visualization_root(sample_id)
        return {
            "original": self._raw_stage_payload(group, view_name),
            "calibrated": self._resolve_calibrated_payload(
                group=group,
                sample_id=sample_id,
                view_name=view_name,
                sample_root=sample_root,
            ),
            "masked": StagePreviewPayload(placeholder_text="等待表型提取", status_text="等待表型提取"),
            "final": StagePreviewPayload(placeholder_text="等待表型提取", status_text="等待表型提取"),
        }

    def _build_phenotype_stage_payloads(
        self,
        group: PlantImageGroup,
        sample_id: str,
        view_name: str,
    ) -> dict[str, StagePreviewPayload]:
        sample_root = self._find_latest_sample_visualization_root(sample_id)
        original = self._raw_stage_payload(group, view_name)
        calibrated = self._resolve_calibrated_payload(
            group=group,
            sample_id=sample_id,
            view_name=view_name,
            sample_root=sample_root,
        )
        in_memory_payloads = self._build_in_memory_phenotype_stage_payloads(
            sample_id=sample_id,
            view_name=view_name,
        )
        if in_memory_payloads is not None:
            return {
                "original": original,
                "calibrated": calibrated,
                **in_memory_payloads,
            }

        if sample_root is None:
            return {
                "original": original,
                "calibrated": calibrated,
                "masked": StagePreviewPayload(placeholder_text="暂无背景消除图", status_text="暂无背景消除图"),
                "final": StagePreviewPayload(placeholder_text="暂无表型可视化", status_text="暂无表型可视化"),
            }

        if view_name == "TOP":
            masked_path = self._pick_top_background_image(sample_root)
            final_path = self._pick_top_final_image(sample_root)
        else:
            front_process_tag = "front_1" if view_name == "FRONT-1" else "front_2"
            masked_path = self._pick_front_process_image(sample_root, front_tag=front_process_tag)
            final_path = self._pick_front_trait_image(sample_root, front_tag=view_name)

        masked_payload = (
            StagePreviewPayload(image_path=masked_path, status_text="已分割")
            if masked_path is not None
            else StagePreviewPayload(placeholder_text="缺少背景消除图", status_text="缺少背景消除图")
        )
        if final_path is not None:
            final_payload = StagePreviewPayload(image_path=final_path, status_text="已生成")
        elif masked_path is not None:
            final_payload = StagePreviewPayload(image_path=masked_path, status_text="使用近似结果图")
        else:
            final_payload = StagePreviewPayload(placeholder_text="缺少表型提取图", status_text="缺少表型提取图")

        return {
            "original": original,
            "calibrated": calibrated,
            "masked": masked_payload,
            "final": final_payload,
        }

    def _has_previewable_phenotype_result(self, sample_id: str) -> bool:
        return self._analysis_result_for_sample(sample_id) is not None or (
            self._find_latest_sample_visualization_root(sample_id) is not None
        )

    def _analysis_result_for_sample(self, sample_id: str) -> PlantAnalysisResult | None:
        if self.displayed_result is not None and self.displayed_result.sample_id == sample_id:
            return self.displayed_result
        if self.current_result is not None and self.current_result.sample_id == sample_id:
            return self.current_result
        if self.current_batch_report is not None:
            for item in self.current_batch_report.sample_results:
                if item.group.sample_id == sample_id:
                    return item.result
        return None

    def _build_in_memory_phenotype_stage_payloads(
        self,
        *,
        sample_id: str,
        view_name: str,
    ) -> dict[str, StagePreviewPayload] | None:
        result = self._analysis_result_for_sample(sample_id)
        if result is None or result.status != "analysis_complete":
            return None

        source_image = self._resolve_result_source_image(result, view_name)
        if source_image is None:
            return None

        if view_name == "TOP":
            segmentation = result.top_segmentation
            if segmentation is None or getattr(segmentation, "mask", None) is None:
                return None
            masked_image = create_masked_color_image(source_image, segmentation.mask)
            final_image = self._build_top_phenotype_overlay(masked_image, segmentation)
        else:
            segmentation = result.front_segmentations.get(view_name)
            if segmentation is None or getattr(segmentation, "mask", None) is None:
                return None
            masked_image = create_masked_color_image(source_image, segmentation.mask)
            final_image = self._build_front_phenotype_overlay(masked_image, segmentation)

        final_payload = (
            StagePreviewPayload(image_array=final_image, status_text="已生成")
            if final_image is not None
            else StagePreviewPayload(image_array=masked_image, status_text="使用近似结果图")
        )
        return {
            "masked": StagePreviewPayload(image_array=masked_image, status_text="已分割"),
            "final": final_payload,
        }

    def _build_top_phenotype_overlay(self, masked_image: np.ndarray, segmentation: Any) -> np.ndarray:
        """Draw TOP phenotype measurements on the background-removed plant image."""

        overlay = masked_image.copy()
        if cv2 is None:
            return overlay

        line_width = self._preview_overlay_line_width(overlay)
        contours = getattr(segmentation, "contours", None) or []
        if contours:
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), line_width)

        convex_hull = getattr(segmentation, "convex_hull", None)
        if convex_hull is not None:
            cv2.polylines(overlay, [convex_hull], True, (255, 255, 0), line_width)
        return overlay

    def _build_front_phenotype_overlay(self, masked_image: np.ndarray, segmentation: Any) -> np.ndarray:
        """Draw FRONT phenotype measurements on the background-removed plant image."""

        overlay = masked_image.copy()
        if cv2 is None:
            return overlay

        line_width = self._preview_overlay_line_width(overlay)
        contours = getattr(segmentation, "contours", None) or []
        if contours:
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), line_width)

        bounding_box = getattr(segmentation, "bounding_box", None)
        if bounding_box is not None:
            x, y, width, height = (int(value) for value in bounding_box)
            cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 255, 0), line_width)
        return overlay

    def _preview_overlay_line_width(self, image: np.ndarray) -> int:
        min_dimension = min(image.shape[:2])
        return max(2, int(round(min_dimension * 0.002)))

    def _resolve_result_source_image(
        self,
        result: PlantAnalysisResult,
        view_name: str,
    ) -> np.ndarray | None:
        calibration = result.calibration_results.get(view_name)
        corrected = getattr(calibration, "corrected_image", None)
        if corrected is not None:
            return corrected

        if self.preprocess_result is not None and self.preprocess_result.sample_id == result.sample_id:
            precomputed = self.preprocess_result.calibrated_images.get(view_name)
            if precomputed is not None:
                return precomputed

        segmentation = result.top_segmentation if view_name == "TOP" else result.front_segmentations.get(view_name)
        if segmentation is None:
            return None
        return getattr(segmentation, "debug_images", {}).get("original")

    def _show_original_preview_for_group(self, group: PlantImageGroup) -> None:
        """Show the shared four-stage preview in original-image mode."""

        self._set_batch_pager_visible(False)
        self._set_trait_gallery_visible(False)
        self._set_preview_state(sample_id=group.sample_id, mode="original")
        self._render_current_preview()

    def _show_preprocess_preview(self, result: PreprocessResult) -> None:
        """Show the shared four-stage preview in preprocess mode."""

        self._set_batch_pager_visible(False)
        self._set_trait_gallery_visible(False)
        self._set_preview_state(sample_id=result.sample_id, mode="preprocess")
        self._render_current_preview()

    def _show_trait_preview_for_sample(self, sample_id: str) -> None:
        """Show the shared four-stage preview in phenotype mode."""

        self._sync_displayed_result_for_sample(sample_id)
        self._set_trait_gallery_visible(False)
        self._set_preview_state(sample_id=sample_id, mode="phenotype")
        self._render_current_preview()

    def _pick_calibration_image(self, sample_root: Path, view_name: str) -> Path | None:
        folder = self._find_numbered_debug_folder(sample_root, prefix="01_")
        if folder is None:
            return None

        markers = {
            "TOP": ("TOP_色彩增强后",),
            "FRONT-1": ("FRONT_1_色彩增强后", "FRONT-1_色彩增强后"),
            "FRONT-2": ("FRONT_2_色彩增强后", "FRONT-2_色彩增强后"),
        }.get(view_name, ())
        return self._pick_calibration_artifact_by_markers(folder, markers)

    def _pick_calibration_comparison_image(self, sample_root: Path, view_name: str) -> Path | None:
        folder = self._find_numbered_debug_folder(sample_root, prefix="01_")
        if folder is None:
            return None

        markers = {
            "TOP": ("TOP_增强前后对比", "TOP_manual_before_after"),
            "FRONT-1": ("FRONT_1_增强前后对比", "FRONT_1_manual_before_after", "FRONT-1_增强前后对比"),
            "FRONT-2": ("FRONT_2_增强前后对比", "FRONT_2_manual_before_after", "FRONT-2_增强前后对比"),
        }.get(view_name, ())
        return self._pick_calibration_artifact_by_markers(folder, markers)

    def _pick_calibration_artifact_by_markers(self, folder: Path, markers: tuple[str, ...]) -> Path | None:
        if not markers:
            return None

        candidates = [
            path for path in folder.glob("*.png")
            if any(marker in path.stem for marker in markers)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _extract_corrected_half_from_comparison(self, comparison_path: Path) -> np.ndarray | None:
        try:
            comparison = load_image(comparison_path)
        except Exception:  # noqa: BLE001
            return None

        height, width = comparison.shape[:2]
        if width < 2:
            return None

        midpoint = width // 2
        corrected_half = comparison[:, midpoint:, :]
        if corrected_half.size == 0:
            return None
        return corrected_half

    def _pick_top_background_image(self, sample_root: Path) -> Path | None:
        masked_region = self._pick_from_folder(sample_root / "06_绿度计算", ("植株区域提取", "masked"))
        if masked_region is not None:
            return masked_region

        folder = self._find_numbered_debug_folder(sample_root, prefix="02_")
        if folder is None:
            return None
        return self._pick_from_folder(folder, ("mask_overlay", "overlay", "最终分割掩码", "filtered_mask", "mask"))

    def _pick_top_final_image(self, sample_root: Path) -> Path | None:
        image = self._pick_from_folder(sample_root / "05_凸包面积计算", ("凸包覆盖图", "覆盖图", "overlay"))
        if image is not None:
            return image
        return self._pick_from_folder(sample_root / "04_叶面积计算", ("叶面积覆盖图", "覆盖图", "overlay"))

    def _pick_front_process_image(self, sample_root: Path, *, front_tag: str) -> Path | None:
        """Choose the background-removed FRONT plant image for one side view."""

        folder = self._find_numbered_debug_folder(sample_root, prefix="03_")
        if folder is None:
            return None

        tag_tokens = {
            front_tag,
            front_tag.replace("front_", "FRONT-"),
            front_tag.replace("front_", "FRONT_"),
            front_tag.replace("FRONT-", "front_"),
            front_tag.replace("-", "_"),
            front_tag.replace("_", "-"),
        }

        preferred = [
            path for path in folder.glob("*.png")
            if any(token in path.name for token in tag_tokens) and any(keyword in path.name for keyword in ("植株区域", "masked_region"))
        ]
        if preferred:
            return max(preferred, key=lambda path: path.stat().st_mtime)

        preferred = [
            path for path in folder.glob("*.png")
            if any(token in path.name for token in tag_tokens) and any(keyword in path.name for keyword in ("projection_overlay", "mask_overlay", "overlay"))
        ]
        if preferred:
            return max(preferred, key=lambda path: path.stat().st_mtime)

        fallback = [path for path in folder.glob("*.png") if any(token in path.name for token in tag_tokens)]
        if fallback:
            return max(fallback, key=lambda path: path.stat().st_mtime)
        return None

    def _find_numbered_debug_folder(self, sample_root: Path, *, prefix: str) -> Path | None:
        """Locate one debug folder by its ordered numeric prefix."""

        matches = [path for path in sample_root.iterdir() if path.is_dir() and path.name.startswith(prefix)]
        if not matches:
            return None
        return max(matches, key=lambda path: path.stat().st_mtime)

    def _pick_front_trait_image(self, sample_root: Path, *, front_tag: str) -> Path | None:
        """Choose one representative FRONT final-stage image for FRONT-1 or FRONT-2."""

        ordered_dirs = [
            sample_root / "07_冠幅高度计算",
            sample_root / "08_冠幅宽度计算",
            sample_root / "09_侧面投影面积",
            sample_root / "03_FRONT正视图分割",
        ]
        preferred_keywords = (front_tag, "边界框", "轮廓", "覆盖图", "overlay")

        for folder in ordered_dirs:
            if not folder.exists():
                continue

            preferred = [
                path for path in folder.glob("*.png")
                if front_tag in path.name and any(keyword in path.name for keyword in preferred_keywords)
            ]
            if preferred:
                return max(preferred, key=lambda path: path.stat().st_mtime)

            fallback = [path for path in folder.glob("*.png") if front_tag in path.name]
            if fallback:
                return max(fallback, key=lambda path: path.stat().st_mtime)

        return None

    def _pick_from_folder(self, folder: Path, keywords: tuple[str, ...]) -> Path | None:
        """Pick latest image from a folder with preferred keywords."""

        if not folder.exists():
            return None
        preferred = [
            path for path in folder.glob("*.png")
            if any(keyword in path.name for keyword in keywords)
        ]
        if preferred:
            return max(preferred, key=lambda path: path.stat().st_mtime)
        all_png = list(folder.glob("*.png"))
        if not all_png:
            return None
        return max(all_png, key=lambda path: path.stat().st_mtime)

    def _set_batch_pager_visible(self, visible: bool) -> None:
        self.batch_pager_widget.setVisible(visible)
        self.batch_prev_button.setVisible(visible)
        self.batch_next_button.setVisible(visible)
        self.batch_page_label.setVisible(visible)

    def _set_trait_gallery_visible(self, visible: bool) -> None:
        self.trait_gallery_group.setVisible(visible)
        if visible:
            return
        self.trait_leaf_area_preview.clear("等待表型提取")
        self.trait_hull_area_preview.clear("等待表型提取")
        self.trait_greenness_preview.clear("等待表型提取")
        self.trait_canopy_height_preview.clear("等待表型提取")
        self.trait_canopy_width_preview.clear("等待表型提取")
        self.trait_side_projection_preview.clear("等待表型提取")
        self._active_trait_key = None
        self._set_active_trait_preview(None)
        self._update_trait_focus_panel(
            title="--",
            value_text="数值: --",
            source_views=(),
            status_text="状态: --",
            message="等待表型提取。",
        )

    def _set_displayed_result(self, result: PlantAnalysisResult | None) -> None:
        """Render one result in the measurement table and linked trait focus panel."""

        self.displayed_result = result
        self._set_result_rows(result)

        target_row = self.results_table.currentRow()
        if target_row < 0 or target_row >= len(TRAIT_SPECS):
            target_row = 0
        self.results_table.setCurrentCell(target_row, 0)

    def _sync_displayed_result_for_sample(self, sample_id: str) -> None:
        """Update the measurement table to match the sample shown in the center panel."""

        matched_result: PlantAnalysisResult | None = None
        if self.current_batch_report is not None:
            for item in self.current_batch_report.sample_results:
                if item.group.sample_id == sample_id:
                    matched_result = item.result
                    break
        elif self.current_result is not None and self.current_result.sample_id == sample_id:
            matched_result = self.current_result

        self._set_displayed_result(matched_result)

    def _update_batch_page_label(self) -> None:
        if not self.batch_preview_sample_ids:
            self.batch_page_label.setText("批量浏览: --")
            return
        current = self.batch_preview_index + 1
        total = len(self.batch_preview_sample_ids)
        sample_id = self.batch_preview_sample_ids[self.batch_preview_index]
        self.batch_page_label.setText(f"批量浏览: {current}/{total}  样本 {sample_id}")

    def _show_previous_batch_preview(self) -> None:
        if not self.batch_preview_sample_ids:
            return
        self.batch_preview_index = (self.batch_preview_index - 1) % len(self.batch_preview_sample_ids)
        sample_id = self.batch_preview_sample_ids[self.batch_preview_index]
        self._show_trait_preview_for_sample(sample_id)
        self._update_batch_page_label()

    def _show_next_batch_preview(self) -> None:
        if not self.batch_preview_sample_ids:
            return
        self.batch_preview_index = (self.batch_preview_index + 1) % len(self.batch_preview_sample_ids)
        sample_id = self.batch_preview_sample_ids[self.batch_preview_index]
        self._show_trait_preview_for_sample(sample_id)
        self._update_batch_page_label()

    def _find_latest_sample_visualization_root(self, sample_id: str) -> Path | None:
        """Pick the newest existing visualization root for the sample."""

        candidates = [
            (DEFAULT_CONFIG.output_dir / "运行可视化" / sample_id),
            (DEFAULT_CONFIG.output_dir / "调试输出" / sample_id),
        ]
        existing = [path for path in candidates if path.exists() and path.is_dir()]
        if not existing:
            return None
        return max(existing, key=lambda path: path.stat().st_mtime)


def launch_app() -> None:
    """Create the Qt application and open the main window."""

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(DEFAULT_CONFIG.app_name)
    window = StrawberryMainWindow()
    app.setWindowIcon(window.windowIcon())
    window.show()
    app.exec_()


def _array_to_pixmap(image: np.ndarray) -> QPixmap:
    """Convert a grayscale or BGR ndarray into a Qt pixmap."""

    if image.ndim == 2:
        contiguous = np.ascontiguousarray(image)
        qimage = QImage(
            contiguous.data,
            contiguous.shape[1],
            contiguous.shape[0],
            contiguous.strides[0],
            QImage.Format_Grayscale8,
        )
        return QPixmap.fromImage(qimage.copy())

    if image.ndim == 3 and image.shape[2] == 3:
        rgb_image = np.ascontiguousarray(image[:, :, ::-1])
        qimage = QImage(
            rgb_image.data,
            rgb_image.shape[1],
            rgb_image.shape[0],
            rgb_image.strides[0],
            QImage.Format_RGB888,
        )
        return QPixmap.fromImage(qimage.copy())

    raise ValueError("image must be a grayscale or 3-channel BGR ndarray")
