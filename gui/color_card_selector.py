"""Color card region selection dialog for manual ROI selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QWidget,
)


class ColorCardRegion(NamedTuple):
    """Represents a color card region as (x, y, width, height) in image coordinates."""
    x: int
    y: int
    width: int
    height: int
    
    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)
    
    @classmethod
    def from_tuple(cls, t: tuple[int, int, int, int]) -> "ColorCardRegion":
        return cls(x=t[0], y=t[1], width=t[2], height=t[3])


class ColorCardRegions(NamedTuple):
    """Color card regions for available plant views."""
    top: ColorCardRegion | None
    front_1: ColorCardRegion | None
    front_2: ColorCardRegion | None
    
    def is_complete(self) -> bool:
        return self.top is not None and self.front_1 is not None and self.front_2 is not None
    
    def to_dict(self) -> dict:
        return {
            "top": self.top.to_tuple() if self.top else None,
            "front_1": self.front_1.to_tuple() if self.front_1 else None,
            "front_2": self.front_2.to_tuple() if self.front_2 else None,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ColorCardRegions":
        return cls(
            top=ColorCardRegion.from_tuple(d["top"]) if d.get("top") else None,
            front_1=ColorCardRegion.from_tuple(d["front_1"]) if d.get("front_1") else None,
            front_2=ColorCardRegion.from_tuple(d["front_2"]) if d.get("front_2") else None,
        )


def save_color_card_regions(regions: ColorCardRegions, config_path: Path) -> None:
    """Save color card regions to a JSON config file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(regions.to_dict(), f, indent=2, ensure_ascii=False)


def load_color_card_regions(config_path: Path) -> ColorCardRegions | None:
    """Load color card regions from a JSON config file."""
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ColorCardRegions.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


class ImageRegionSelector(QLabel):
    """Widget for selecting a rectangular region on an image."""
    
    region_selected = pyqtSignal(QRect)
    
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._original_pixmap: QPixmap | None = None
        self._image_data: np.ndarray | None = None  # Keep reference to image data
        self._scale_factor: float = 1.0
        self._start_point: QPoint | None = None
        self._current_rect: QRect | None = None
        self._final_rect: QRect | None = None
        self._image_offset: QPoint = QPoint(0, 0)
        
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
    
    def set_image(self, image: np.ndarray) -> None:
        """Set the image to display."""
        if image is None:
            self.clear()
            return
        
        # Convert BGR to RGB and ensure contiguous memory
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Make a contiguous copy to ensure data is owned
        rgb_image = np.ascontiguousarray(rgb_image)
        self._image_data = rgb_image  # Keep reference to prevent garbage collection
        
        height, width = rgb_image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # Create a deep copy of the pixmap to own the data
        self._original_pixmap = QPixmap.fromImage(q_image.copy())
        self._final_rect = None
        self._current_rect = None
        self._update_display()
    
    def _update_display(self) -> None:
        """Update the displayed pixmap with any selection overlay."""
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        
        # Scale pixmap to fit the widget while maintaining aspect ratio
        available_size = self.size()
        if available_size.width() <= 0 or available_size.height() <= 0:
            return
            
        scaled_pixmap = self._original_pixmap.scaled(
            available_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        if scaled_pixmap.isNull() or scaled_pixmap.width() <= 0:
            return
        
        # Calculate scale factor and offset for coordinate mapping
        self._scale_factor = scaled_pixmap.width() / self._original_pixmap.width()
        self._image_offset = QPoint(
            (available_size.width() - scaled_pixmap.width()) // 2,
            (available_size.height() - scaled_pixmap.height()) // 2
        )
        
        # Draw selection rectangle if exists
        display_pixmap = scaled_pixmap.copy()
        painter = QPainter(display_pixmap)
        if painter.isActive():
            pen = QPen(QColor(255, 0, 0), 3)
            painter.setPen(pen)
            
            rect_to_draw = self._current_rect or self._final_rect
            if rect_to_draw:
                # Convert to display coordinates
                display_rect = QRect(
                    int(rect_to_draw.x() * self._scale_factor),
                    int(rect_to_draw.y() * self._scale_factor),
                    int(rect_to_draw.width() * self._scale_factor),
                    int(rect_to_draw.height() * self._scale_factor),
                )
                painter.drawRect(display_rect)
            
            painter.end()
        self.setPixmap(display_pixmap)
    
    def _widget_to_image_coords(self, pos: QPoint) -> QPoint:
        """Convert widget coordinates to original image coordinates."""
        if self._original_pixmap is None:
            return pos
        
        # Remove offset
        adjusted = pos - self._image_offset
        # Scale back to original image coordinates
        return QPoint(
            int(adjusted.x() / self._scale_factor),
            int(adjusted.y() / self._scale_factor)
        )
    
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._original_pixmap is not None:
            self._start_point = self._widget_to_image_coords(event.pos())
            self._current_rect = None
    
    def mouseMoveEvent(self, event) -> None:
        if self._start_point is not None and self._original_pixmap is not None:
            current_point = self._widget_to_image_coords(event.pos())
            self._current_rect = QRect(self._start_point, current_point).normalized()
            
            # Clamp to image bounds
            img_rect = QRect(0, 0, self._original_pixmap.width(), self._original_pixmap.height())
            self._current_rect = self._current_rect.intersected(img_rect)
            
            self._update_display()
    
    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._current_rect is not None:
            if self._current_rect.width() > 10 and self._current_rect.height() > 10:
                self._final_rect = self._current_rect
                self.region_selected.emit(self._final_rect)
            self._current_rect = None
            self._start_point = None
            self._update_display()
    
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_display()
    
    def get_selected_region(self) -> ColorCardRegion | None:
        """Get the selected region in original image coordinates."""
        if self._final_rect is None:
            return None
        return ColorCardRegion(
            x=self._final_rect.x(),
            y=self._final_rect.y(),
            width=self._final_rect.width(),
            height=self._final_rect.height(),
        )
    
    def set_region(self, region: ColorCardRegion | None) -> None:
        """Set the selection region."""
        if region is None:
            self._final_rect = None
        else:
            self._final_rect = QRect(region.x, region.y, region.width, region.height)
        self._update_display()


class SingleViewSelectorDialog(QDialog):
    """Dialog for selecting color card region in a single view."""
    
    def __init__(
        self,
        view_name: str,
        image: np.ndarray,
        step_info: str,
        existing_region: ColorCardRegion | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.view_name = view_name
        self.image = image
        self.selected_region: ColorCardRegion | None = existing_region
        
        self._build_ui(step_info, existing_region)
        self.setWindowTitle(f"选择色卡区域 - {view_name}")
        self.resize(900, 700)
    
    def _build_ui(self, step_info: str, existing_region: ColorCardRegion | None) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Instructions
        instruction_label = QLabel(
            f"<b>{step_info}</b><br><br>"
            f"请使用鼠标在图像中<b>拖拽选择</b>色卡所在区域。<br>"
            f'选择完成后点击"确认"按钮。'
        )
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet("font-size: 14px; padding: 10px; background: #fffbe6; border-radius: 8px;")
        layout.addWidget(instruction_label)
        
        # Image selector
        self.selector = ImageRegionSelector()
        self.selector.set_image(self.image)
        if existing_region:
            self.selector.set_region(existing_region)
        self.selector.region_selected.connect(self._on_region_selected)
        layout.addWidget(self.selector, 1)
        
        # Status label
        self.status_label = QLabel("请在图中拖拽选择色卡区域")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 13px; color: #666;")
        if existing_region:
            self.status_label.setText(
                f"已有选区: ({existing_region.x}, {existing_region.y}) - "
                f"{existing_region.width}×{existing_region.height}"
            )
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.reset_button = QPushButton("重新选择")
        self.reset_button.clicked.connect(self._on_reset)
        button_layout.addWidget(self.reset_button)
        
        self.confirm_button = QPushButton("确认")
        self.confirm_button.setEnabled(existing_region is not None)
        self.confirm_button.clicked.connect(self.accept)
        self.confirm_button.setStyleSheet(
            "QPushButton { background: #2f6f4f; color: white; padding: 10px 30px; font-weight: bold; }"
        )
        button_layout.addWidget(self.confirm_button)
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def _on_region_selected(self, rect: QRect) -> None:
        self.selected_region = ColorCardRegion(
            x=rect.x(),
            y=rect.y(),
            width=rect.width(),
            height=rect.height(),
        )
        self.status_label.setText(
            f"已选择区域: ({rect.x()}, {rect.y()}) - {rect.width()}×{rect.height()}"
        )
        self.confirm_button.setEnabled(True)
    
    def _on_reset(self) -> None:
        self.selector.set_region(None)
        self.selected_region = None
        self.status_label.setText("请在图中拖拽选择色卡区域")
        self.confirm_button.setEnabled(False)


def select_color_card_regions_interactive(
    top_image: np.ndarray | None,
    front_1_image: np.ndarray | None,
    front_2_image: np.ndarray | None,
    existing_regions: ColorCardRegions | None = None,
    parent: QWidget | None = None,
) -> ColorCardRegions | None:
    """Show sequential dialogs to select color-card regions for available views."""

    candidates = [
        ("top", "TOP", top_image, existing_regions.top if existing_regions else None),
        ("front_1", "FRONT-1", front_1_image, existing_regions.front_1 if existing_regions else None),
        ("front_2", "FRONT-2", front_2_image, existing_regions.front_2 if existing_regions else None),
    ]
    available_views = [
        (key, view_name, image, existing_region)
        for key, view_name, image, existing_region in candidates
        if image is not None
    ]
    if not available_views:
        QMessageBox.warning(parent, "缺少图像", "没有可用的视角图像，无法选择色卡区域。")
        return None

    results: dict[str, ColorCardRegion | None] = {"top": None, "front_1": None, "front_2": None}
    total_views = len(available_views)
    for index, (key, view_name, image, existing_region) in enumerate(available_views, start=1):
        dialog = SingleViewSelectorDialog(
            view_name=view_name,
            image=image,
            step_info=f"Step {index}/{total_views}: select the color card in {view_name}",
            existing_region=existing_region,
            parent=parent,
        )

        if dialog.exec_() != QDialog.Accepted:
            return None
        if dialog.selected_region is None:
            QMessageBox.warning(parent, "未选择区域", f"未选择 {view_name} 的色卡区域，请重新操作。")
            return None
        results[key] = dialog.selected_region

    return ColorCardRegions(
        top=results["top"],
        front_1=results["front_1"],
        front_2=results["front_2"],
    )


def draw_region_on_image(image: np.ndarray, region: ColorCardRegion, color: tuple = (0, 0, 255), thickness: int = 3) -> np.ndarray:
    """Draw a rectangle on the image to show the selected region."""
    result = image.copy()
    x, y, w, h = region.x, region.y, region.width, region.height
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    return result
