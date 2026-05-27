# GUI Preview Panel Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 GUI 中间预览区，使其固定为 4 张阶段图，并由统一视角状态驱动同步翻页、样本切换保留视角和稳定状态切换。

**Architecture:** 新增 `gui/stage_preview.py` 承载纯逻辑的视角顺序、回退和阶段 payload 数据结构；`gui/main_window.py` 负责统一视角状态、顶部翻页器、卡片级导航按钮和 `_refresh_stage_previews()` 单一刷新入口。测试分两层：`tests/test_stage_preview.py` 验证纯逻辑，`tests/test_gui_smoke.py` 验证 Qt 连线、固定标题、三种状态切换和同步翻页。

**Tech Stack:** Python 3、PyQt5、pytest、numpy、pathlib

---

## File Map

- Create: `gui/stage_preview.py`
  - 责任：定义 `VIEW_SEQUENCE`、`PreviewViewState`、`StagePreviewPayload`、可用视角判断、视角回退和循环切换逻辑。
- Create: `tests/test_stage_preview.py`
  - 责任：为 `gui.stage_preview` 编写不依赖 Qt 的单元测试。
- Modify: `gui/main_window.py`
  - 责任：新增顶部视角控制栏、卡片悬浮导航、统一刷新状态、4 阶段卡片内容解析与旧 trait gallery 下线。
- Modify: `tests/test_gui_smoke.py`
  - 责任：覆盖新中间区结构、三种模式下四张卡片显示规则、同步翻页和样本切换回退。

## Workspace Note

当前目录 `d:\code\pycharm\strawberry` 不是 git 仓库根目录。因此每个任务中的 “Commit” 步骤都写成“若 git 可用则执行”的命令；没有 git 也不影响实施和验证。

### Task 1: Add shared preview state helpers

**Files:**
- Create: `gui/stage_preview.py`
- Test: `tests/test_stage_preview.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from core.grouping import PlantImageGroup
from gui.stage_preview import (
    VIEW_SEQUENCE,
    PreviewViewState,
    StagePreviewPayload,
    available_views_for_group,
    pick_active_view,
    step_view,
)


def test_available_views_for_group_returns_only_present_views() -> None:
    group = PlantImageGroup(
        sample_id="2AB",
        top_image=Path("data/2AB_TOP.png"),
        front_0_image=Path("data/2AB-1.png"),
        front_180_image=None,
    )

    assert VIEW_SEQUENCE == ("TOP", "FRONT-1", "FRONT-2")
    assert available_views_for_group(group) == ("TOP", "FRONT-1")


def test_pick_active_view_keeps_requested_view_when_available() -> None:
    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )

    assert pick_active_view(group, preferred_view="FRONT-1") == ("FRONT-1", False)


def test_pick_active_view_falls_back_to_first_available_view() -> None:
    group = PlantImageGroup(
        sample_id="2AB",
        top_image=Path("data/2AB_TOP.png"),
        front_0_image=None,
        front_180_image=Path("data/2AB-2.png"),
    )

    assert pick_active_view(group, preferred_view="FRONT-1") == ("TOP", True)


def test_step_view_cycles_inside_available_views_only() -> None:
    available = ("TOP", "FRONT-2")

    assert step_view("TOP", available, 1) == "FRONT-2"
    assert step_view("FRONT-2", available, 1) == "TOP"
    assert step_view("TOP", available, -1) == "FRONT-2"


def test_preview_view_state_and_payload_are_lightweight() -> None:
    state = PreviewViewState(
        sample_id="1AB",
        mode="original",
        view_name="TOP",
        available_views=("TOP", "FRONT-1", "FRONT-2"),
    )
    payload = StagePreviewPayload(placeholder_text="等待颜色校正", status_text="TOP | 等待颜色校正")

    assert state.view_name == "TOP"
    assert state.mode == "original"
    assert payload.placeholder_text == "等待颜色校正"
    assert payload.status_text == "TOP | 等待颜色校正"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage_preview.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'gui.stage_preview'`

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.grouping import PlantImageGroup

VIEW_SEQUENCE = ("TOP", "FRONT-1", "FRONT-2")


@dataclass(frozen=True, slots=True)
class PreviewViewState:
    sample_id: str | None
    mode: str
    view_name: str
    available_views: tuple[str, ...]
    fallback_notice: str = ""


@dataclass(frozen=True, slots=True)
class StagePreviewPayload:
    image_path: Path | None = None
    image_array: np.ndarray | None = None
    placeholder_text: str = ""
    status_text: str = ""


def available_views_for_group(group: PlantImageGroup | None) -> tuple[str, ...]:
    if group is None:
        return ("TOP",)

    views: list[str] = []
    if group.top_image is not None:
        views.append("TOP")
    if group.front_0_image is not None:
        views.append("FRONT-1")
    if group.front_180_image is not None:
        views.append("FRONT-2")
    return tuple(views or ("TOP",))


def pick_active_view(group: PlantImageGroup | None, preferred_view: str | None) -> tuple[str, bool]:
    available = available_views_for_group(group)
    if preferred_view in available:
        return preferred_view, False
    return available[0], True


def step_view(current_view: str, available_views: tuple[str, ...], direction: int) -> str:
    if not available_views:
        return "TOP"
    if current_view not in available_views:
        return available_views[0]

    current_index = available_views.index(current_view)
    next_index = (current_index + direction) % len(available_views)
    return available_views[next_index]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage_preview.py -q`

Expected: PASS with `5 passed`

- [ ] **Step 5: Commit**

```bash
# 当前目录不是 git 仓库；若 git 可用则执行：
git add gui/stage_preview.py tests/test_stage_preview.py
git commit -m "feat: add shared preview state helpers"
```

### Task 2: Lock the new preview shell in GUI smoke tests

**Files:**
- Modify: `tests/test_gui_smoke.py`
- Test: `tests/test_gui_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
def test_main_window_builds_shared_preview_switcher(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()

    assert window.preview_view_prev_button.text() == "‹"
    assert window.preview_view_next_button.text() == "›"
    assert window.preview_view_name_label.text() == "TOP"
    assert window.preview_view_index_label.text() == "1 / 3"
    assert window.preview_notice_label.text() == ""
    window.close()


def test_main_window_keeps_fixed_stage_titles_and_hides_trait_gallery(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()

    assert window.orig_preview.title.text() == "1. 原始图像"
    assert window.calib_preview.title.text() == "2. 颜色校正"
    assert window.mask_preview.title.text() == "3. 背景消除"
    assert window.final_preview.title.text() == "4. 表型提取"
    assert not window.trait_gallery_group.isVisible()
    assert window.orig_preview.meta_label.isVisible()
    assert window.orig_preview.prev_nav_button is not None
    assert window.orig_preview.next_nav_button is not None
    window.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gui_smoke.py::test_main_window_builds_shared_preview_switcher tests/test_gui_smoke.py::test_main_window_keeps_fixed_stage_titles_and_hides_trait_gallery -q`

Expected: FAIL with `AttributeError` for missing preview switcher widgets and/or failed visibility assertions on the card status line.

- [ ] **Step 3: Write minimal implementation**

```python
from typing import Callable

from PyQt5.QtWidgets import QToolButton


class ImagePreviewCard(QFrame):
    def __init__(self, title: str, *, preview_height: int = 260, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.preview_height = preview_height
        self._source_pixmap: QPixmap | None = None
        self._selected = False
        self._nav_enabled = False
        self.prev_nav_button: QToolButton | None = None
        self.next_nav_button: QToolButton | None = None
        self._prev_handler: Callable[[], None] | None = None
        self._next_handler: Callable[[], None] | None = None
        self.title = QLabel(title)
        self.image_label = QLabel("等待加载图像")
        self.meta_label = QLabel("--")
        self._viewer_title = title
        self._build_ui()

    def _build_ui(self) -> None:
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(self.preview_height)
        self.image_label.setMaximumHeight(self.preview_height)
        self.image_label.setStyleSheet("border: 1px dashed #6b8f71; border-radius: 10px;")
        self.image_label.installEventFilter(self)

        self.meta_label.setWordWrap(True)
        self.meta_label.setVisible(True)

        self.prev_nav_button = QToolButton(self.image_label)
        self.prev_nav_button.setText("‹")
        self.prev_nav_button.setAutoRaise(True)
        self.prev_nav_button.hide()

        self.next_nav_button = QToolButton(self.image_label)
        self.next_nav_button.setText("›")
        self.next_nav_button.setAutoRaise(True)
        self.next_nav_button.hide()

        layout.addWidget(self.title)
        layout.addWidget(self.image_label)
        layout.addWidget(self.meta_label)

    def set_navigation_handlers(self, prev_handler: Callable[[], None], next_handler: Callable[[], None]) -> None:
        self._prev_handler = prev_handler
        self._next_handler = next_handler
        self.prev_nav_button.clicked.connect(prev_handler)
        self.next_nav_button.clicked.connect(next_handler)
        self._nav_enabled = True


def _build_center_panel(self) -> QWidget:
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QFrame.NoFrame)
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setSpacing(12)

    self.preview_group = QGroupBox("处理流程预览")
    preview_layout = QGridLayout(self.preview_group)
    preview_layout.setSpacing(12)

    self.preview_mode_label = QLabel("当前展示: 原始图像")
    preview_layout.addWidget(self.preview_mode_label, 0, 0, 1, 2)

    toolbar_widget = QWidget()
    toolbar_layout = QHBoxLayout(toolbar_widget)
    toolbar_layout.setContentsMargins(0, 0, 0, 0)
    toolbar_layout.setSpacing(8)

    self.preview_view_prev_button = QPushButton("‹")
    self.preview_view_name_label = QLabel("TOP")
    self.preview_view_next_button = QPushButton("›")
    self.preview_view_index_label = QLabel("1 / 3")
    self.preview_notice_label = QLabel("")

    toolbar_layout.addWidget(self.preview_view_prev_button)
    toolbar_layout.addWidget(self.preview_view_name_label)
    toolbar_layout.addWidget(self.preview_view_next_button)
    toolbar_layout.addStretch(1)
    toolbar_layout.addWidget(self.preview_view_index_label)

    preview_layout.addWidget(toolbar_widget, 1, 0, 1, 2)
    preview_layout.addWidget(self.preview_notice_label, 2, 0, 1, 2)

    self.orig_preview = ImagePreviewCard("1. 原始图像", preview_height=300)
    self.calib_preview = ImagePreviewCard("2. 颜色校正", preview_height=300)
    self.mask_preview = ImagePreviewCard("3. 背景消除", preview_height=300)
    self.final_preview = ImagePreviewCard("4. 表型提取", preview_height=300)

    preview_layout.addWidget(self.orig_preview, 3, 0)
    preview_layout.addWidget(self.calib_preview, 3, 1)
    preview_layout.addWidget(self.mask_preview, 4, 0)
    preview_layout.addWidget(self.final_preview, 4, 1)

    self.trait_gallery_group.setVisible(False)
    layout.addWidget(self.preview_group)
    layout.addWidget(self.batch_pager_widget)
    layout.addWidget(meta_group)
    scroll_area.setWidget(content)
    return scroll_area
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gui_smoke.py::test_main_window_builds_shared_preview_switcher tests/test_gui_smoke.py::test_main_window_keeps_fixed_stage_titles_and_hides_trait_gallery -q`

Expected: PASS with `2 passed`

- [ ] **Step 5: Commit**

```bash
# 当前目录不是 git 仓库；若 git 可用则执行：
git add gui/main_window.py tests/test_gui_smoke.py
git commit -m "feat: add preview toolbar shell"
```

### Task 3: Add mode-transition smoke tests for the 4-stage panel

**Files:**
- Modify: `tests/test_gui_smoke.py`
- Test: `tests/test_gui_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
def _save_color_preview(path: Path, color_name: str) -> Path:
    pixmap = QPixmap(80, 60)
    pixmap.fill(QColor(color_name))
    assert pixmap.save(str(path))
    return path


def test_show_original_preview_uses_current_view_and_placeholders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=_save_color_preview(tmp_path / "1AB_TOP.png", "#ff6666"),
        front_0_image=_save_color_preview(tmp_path / "1AB-1.png", "#66ff66"),
        front_180_image=_save_color_preview(tmp_path / "1AB-2.png", "#6666ff"),
    )

    window = StrawberryMainWindow()
    window.preview_view_state = PreviewViewState(
        sample_id="1AB",
        mode="original",
        view_name="FRONT-1",
        available_views=("TOP", "FRONT-1", "FRONT-2"),
    )

    window._show_original_preview_for_group(group)

    assert window.preview_view_name_label.text() == "FRONT-1"
    assert window.orig_preview.meta_label.text() == "FRONT-1 | 已加载"
    assert window.calib_preview.image_label.text() == "等待颜色校正"
    assert window.mask_preview.image_label.text() == "等待背景消除"
    assert window.final_preview.image_label.text() == "等待表型提取"
    window.close()


def test_show_preprocess_preview_updates_first_two_stages_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()
    group = PlantImageGroup(
        sample_id="1AB",
        top_image=_save_color_preview(tmp_path / "1AB_TOP.png", "#ff6666"),
        front_0_image=_save_color_preview(tmp_path / "1AB-1.png", "#66ff66"),
        front_180_image=_save_color_preview(tmp_path / "1AB-2.png", "#6666ff"),
    )
    window.groups = [group]
    window.group_list.addItem("1AB [完整]")
    window.group_list.setCurrentRow(0)
    window.preview_view_state = PreviewViewState(
        sample_id="1AB",
        mode="preprocess",
        view_name="FRONT-2",
        available_views=("TOP", "FRONT-1", "FRONT-2"),
    )

    preprocess = SimpleNamespace(
        sample_id="1AB",
        calibrated_images={"FRONT-2": None},
        calibration_results={},
    )

    window._show_preprocess_preview(preprocess)

    assert window.preview_view_name_label.text() == "FRONT-2"
    assert window.orig_preview.meta_label.text() == "FRONT-2 | 已加载"
    assert window.calib_preview.meta_label.text() == "FRONT-2 | 未校正"
    assert window.mask_preview.image_label.text() == "等待表型提取"
    assert window.final_preview.image_label.text() == "等待表型提取"
    window.close()


def test_switching_sample_preserves_view_or_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()
    group = PlantImageGroup(
        sample_id="2AB",
        top_image=_save_color_preview(tmp_path / "2AB_TOP.png", "#ffaa66"),
        front_0_image=None,
        front_180_image=_save_color_preview(tmp_path / "2AB-2.png", "#6666ff"),
    )
    window.preview_view_state = PreviewViewState(
        sample_id="1AB",
        mode="original",
        view_name="FRONT-1",
        available_views=("TOP", "FRONT-1", "FRONT-2"),
    )

    window._show_original_preview_for_group(group)

    assert window.preview_view_name_label.text() == "TOP"
    assert "已自动切换" in window.preview_notice_label.text()
    window.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gui_smoke.py::test_show_original_preview_uses_current_view_and_placeholders tests/test_gui_smoke.py::test_show_preprocess_preview_updates_first_two_stages_only tests/test_gui_smoke.py::test_switching_sample_preserves_view_or_falls_back -q`

Expected: FAIL because `_show_original_preview_for_group()`、`_show_preprocess_preview()` still hardcode TOP and do not use a shared `preview_view_state`.

- [ ] **Step 3: Write minimal implementation**

```python
from gui.stage_preview import PreviewViewState, StagePreviewPayload, available_views_for_group, pick_active_view, step_view


def _path_for_view(self, group: PlantImageGroup | None, view_name: str) -> Path | None:
    if group is None:
        return None
    if view_name == "TOP":
        return group.top_image
    if view_name == "FRONT-1":
        return group.front_0_image
    return group.front_180_image


def _set_preview_view_state(self, group: PlantImageGroup | None, *, mode: str) -> None:
    previous_view = getattr(getattr(self, "preview_view_state", None), "view_name", "TOP")
    active_view, fell_back = pick_active_view(group, previous_view)
    self.preview_view_state = PreviewViewState(
        sample_id=group.sample_id if group is not None else None,
        mode=mode,
        view_name=active_view,
        available_views=available_views_for_group(group),
        fallback_notice="当前视角不可用，已自动切换" if fell_back else "",
    )


def _update_preview_toolbar(self) -> None:
    state = self.preview_view_state
    current_index = state.available_views.index(state.view_name) + 1
    self.preview_view_name_label.setText(state.view_name)
    self.preview_view_index_label.setText(f"{current_index} / {len(state.available_views)}")
    self.preview_notice_label.setText(state.fallback_notice)


def _apply_stage_preview_payload(self, card: ImagePreviewCard, payload: StagePreviewPayload) -> None:
    if payload.image_array is not None:
        card.set_image_array(payload.image_array, meta_text=payload.status_text)
        card.meta_label.setText(payload.status_text)
        return
    if payload.image_path is not None:
        card.set_image_path(payload.image_path)
        card.meta_label.setText(payload.status_text)
        return
    card.clear(payload.placeholder_text, payload.status_text)


def _build_stage_preview_payloads(self, group: PlantImageGroup | None) -> dict[str, StagePreviewPayload]:
    view_name = self.preview_view_state.view_name
    raw_path = self._path_for_view(group, view_name)

    if self.preview_view_state.mode == "original":
        return {
            "original": StagePreviewPayload(image_path=raw_path, status_text=f"{view_name} | 已加载"),
            "calibrated": StagePreviewPayload(placeholder_text="等待颜色校正", status_text=f"{view_name} | 等待颜色校正"),
            "mask": StagePreviewPayload(placeholder_text="等待背景消除", status_text=f"{view_name} | 等待背景消除"),
            "final": StagePreviewPayload(placeholder_text="等待表型提取", status_text=f"{view_name} | 等待表型提取"),
        }

    calibrated = self.preprocess_result.calibrated_images.get(view_name) if self.preprocess_result is not None else None
    calibration_status = "已校正" if calibrated is not None else "未校正"
    return {
        "original": StagePreviewPayload(image_path=raw_path, status_text=f"{view_name} | 已加载"),
        "calibrated": StagePreviewPayload(
            image_array=calibrated,
            image_path=raw_path if calibrated is None else None,
            placeholder_text="校正失败",
            status_text=f"{view_name} | {calibration_status}",
        ),
        "mask": StagePreviewPayload(placeholder_text="等待表型提取", status_text=f"{view_name} | 等待表型提取"),
        "final": StagePreviewPayload(placeholder_text="等待表型提取", status_text=f"{view_name} | 等待表型提取"),
    }


def _refresh_stage_previews(self) -> None:
    payloads = self._build_stage_preview_payloads(self._selected_group())
    self._update_preview_toolbar()
    self._apply_stage_preview_payload(self.orig_preview, payloads["original"])
    self._apply_stage_preview_payload(self.calib_preview, payloads["calibrated"])
    self._apply_stage_preview_payload(self.mask_preview, payloads["mask"])
    self._apply_stage_preview_payload(self.final_preview, payloads["final"])


def _show_original_preview_for_group(self, group: PlantImageGroup) -> None:
    self.preview_mode = "original"
    self._set_preview_view_state(group, mode="original")
    self._set_trait_gallery_visible(False)
    self._refresh_stage_previews()


def _show_preprocess_preview(self, result: PreprocessResult) -> None:
    self.preview_mode = "preprocess"
    self.preprocess_result = result
    self._set_preview_view_state(self._selected_group(), mode="preprocess")
    self._set_trait_gallery_visible(False)
    self._refresh_stage_previews()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage_preview.py tests/test_gui_smoke.py::test_show_original_preview_uses_current_view_and_placeholders tests/test_gui_smoke.py::test_show_preprocess_preview_updates_first_two_stages_only tests/test_gui_smoke.py::test_switching_sample_preserves_view_or_falls_back -q`

Expected: PASS for the preview-state unit tests and the 3 new GUI smoke tests.

- [ ] **Step 5: Commit**

```bash
# 当前目录不是 git 仓库；若 git 可用则执行：
git add gui/stage_preview.py gui/main_window.py tests/test_stage_preview.py tests/test_gui_smoke.py
git commit -m "feat: add stage preview state transitions"
```

### Task 4: Add synchronized navigation tests and wire all controls to the same state

**Files:**
- Modify: `tests/test_gui_smoke.py`
- Modify: `gui/main_window.py`

- [ ] **Step 1: Write the failing test**

```python
def test_preview_view_navigation_updates_all_cards_in_sync(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()
    group = PlantImageGroup(
        sample_id="1AB",
        top_image=_save_color_preview(tmp_path / "1AB_TOP.png", "#ff6666"),
        front_0_image=_save_color_preview(tmp_path / "1AB-1.png", "#66ff66"),
        front_180_image=_save_color_preview(tmp_path / "1AB-2.png", "#6666ff"),
    )
    monkeypatch.setattr(window, "_selected_group", lambda: group)

    window.preview_view_state = PreviewViewState(
        sample_id="1AB",
        mode="phenotype",
        view_name="TOP",
        available_views=("TOP", "FRONT-1", "FRONT-2"),
    )

    def fake_payloads(current_group: PlantImageGroup) -> dict[str, StagePreviewPayload]:
        view_name = window.preview_view_state.view_name
        source_path = {
            "TOP": current_group.top_image,
            "FRONT-1": current_group.front_0_image,
            "FRONT-2": current_group.front_180_image,
        }[view_name]
        return {
            "original": StagePreviewPayload(image_path=source_path, status_text=f"{view_name} | 已加载"),
            "calibrated": StagePreviewPayload(image_path=source_path, status_text=f"{view_name} | 已校正"),
            "mask": StagePreviewPayload(image_path=source_path, status_text=f"{view_name} | 已分割"),
            "final": StagePreviewPayload(image_path=source_path, status_text=f"{view_name} | 已生成"),
        }

    monkeypatch.setattr(window, "_build_stage_preview_payloads", fake_payloads)

    window._refresh_stage_previews()
    window._step_preview_view(1)

    assert window.preview_view_name_label.text() == "FRONT-1"
    assert window.orig_preview.meta_label.text() == "FRONT-1 | 已加载"
    assert window.calib_preview.meta_label.text() == "FRONT-1 | 已校正"
    assert window.mask_preview.meta_label.text() == "FRONT-1 | 已分割"
    assert window.final_preview.meta_label.text() == "FRONT-1 | 已生成"
    window.close()


def test_preview_navigation_buttons_drive_the_same_global_view_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(StrawberryMainWindow, "_load_groups", lambda self, directory: None)

    window = StrawberryMainWindow()
    group = PlantImageGroup(
        sample_id="1AB",
        top_image=_save_color_preview(tmp_path / "1AB_TOP.png", "#ff6666"),
        front_0_image=_save_color_preview(tmp_path / "1AB-1.png", "#66ff66"),
        front_180_image=_save_color_preview(tmp_path / "1AB-2.png", "#6666ff"),
    )
    monkeypatch.setattr(window, "_selected_group", lambda: group)

    calls: list[int] = []
    monkeypatch.setattr(window, "_step_preview_view", lambda direction: calls.append(direction))

    window.preview_view_prev_button.click()
    window.preview_view_next_button.click()
    window.orig_preview.next_nav_button.click()
    window.final_preview.prev_nav_button.click()

    assert calls == [-1, 1, 1, -1]
    window.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gui_smoke.py::test_preview_view_navigation_updates_all_cards_in_sync tests/test_gui_smoke.py::test_preview_navigation_buttons_drive_the_same_global_view_state -q`

Expected: FAIL because `_step_preview_view()` does not yet drive a single global state and the card navigation buttons are not wired to the same entrypoint.

- [ ] **Step 3: Write minimal implementation**

```python
def _step_preview_view(self, direction: int) -> None:
    state = self.preview_view_state
    next_view = step_view(state.view_name, state.available_views, direction)
    self.preview_view_state = PreviewViewState(
        sample_id=state.sample_id,
        mode=state.mode,
        view_name=next_view,
        available_views=state.available_views,
        fallback_notice="",
    )
    self._refresh_stage_previews()


def _build_center_panel(self) -> QWidget:
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QFrame.NoFrame)
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setSpacing(12)
    layout.addWidget(self.preview_group)
    layout.addWidget(self.batch_pager_widget)
    layout.addWidget(meta_group)

    self.preview_view_prev_button.clicked.connect(lambda: self._step_preview_view(-1))
    self.preview_view_next_button.clicked.connect(lambda: self._step_preview_view(1))

    for card in (self.orig_preview, self.calib_preview, self.mask_preview, self.final_preview):
        card.set_navigation_handlers(
            lambda: self._step_preview_view(-1),
            lambda: self._step_preview_view(1),
        )

    scroll_area.setWidget(content)
    return scroll_area
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gui_smoke.py::test_preview_view_navigation_updates_all_cards_in_sync tests/test_gui_smoke.py::test_preview_navigation_buttons_drive_the_same_global_view_state -q`

Expected: PASS with `2 passed`

- [ ] **Step 5: Commit**

```bash
# 当前目录不是 git 仓库；若 git 可用则执行：
git add gui/main_window.py tests/test_gui_smoke.py
git commit -m "feat: wire synchronized preview navigation"
```

### Task 5: Finish real image-source resolution and run full verification

**Files:**
- Modify: `gui/main_window.py`
- Test: `tests/test_stage_preview.py`
- Test: `tests/test_gui_smoke.py`

- [ ] **Step 1: Complete the final image-source resolution**

```python
def _calibrated_array_for_view(self, view_name: str) -> np.ndarray | None:
    if self.preprocess_result is not None and self.preprocess_result.sample_id == self.preview_view_state.sample_id:
        calibrated = self.preprocess_result.calibrated_images.get(view_name)
        if calibrated is not None:
            return calibrated

    if self.current_result is not None and self.current_result.sample_id == self.preview_view_state.sample_id:
        calibration = self.current_result.calibration_results.get(view_name)
        corrected = getattr(calibration, "corrected_image", None)
        if corrected is not None:
            return corrected
    return None


def _mask_preview_path_for_view(self, sample_root: Path | None, view_name: str) -> Path | None:
    if sample_root is None:
        return None
    if view_name == "TOP":
        return self._pick_top_process_image(sample_root)
    front_tag = "front_1" if view_name == "FRONT-1" else "front_2"
    return self._pick_front_process_image(sample_root, front_tag=front_tag)


def _final_preview_path_for_view(self, sample_root: Path | None, view_name: str) -> Path | None:
    if sample_root is None:
        return None
    if view_name == "TOP":
        top_image = self._pick_from_folder(sample_root / "05_凸包面积计算", ("凸包", "覆盖图", "overlay"))
        if top_image is None:
            top_image = self._pick_from_folder(sample_root / "04_叶面积计算", ("覆盖图", "overlay"))
        return top_image
    front_tag = "FRONT-1" if view_name == "FRONT-1" else "FRONT-2"
    return self._pick_front_trait_image(sample_root, front_tag=front_tag)


def _build_stage_preview_payloads(self, group: PlantImageGroup | None) -> dict[str, StagePreviewPayload]:
    view_name = self.preview_view_state.view_name
    sample_root = self._find_latest_sample_visualization_root(group.sample_id) if group is not None else None
    raw_path = self._path_for_view(group, view_name)
    calibrated_array = self._calibrated_array_for_view(view_name)
    mask_path = self._mask_preview_path_for_view(sample_root, view_name)
    final_path = self._final_preview_path_for_view(sample_root, view_name)

    if self.preview_mode == "original":
        return {
            "original": StagePreviewPayload(image_path=raw_path, status_text=f"{view_name} | 已加载"),
            "calibrated": StagePreviewPayload(placeholder_text="等待颜色校正", status_text=f"{view_name} | 等待颜色校正"),
            "mask": StagePreviewPayload(placeholder_text="等待背景消除", status_text=f"{view_name} | 等待背景消除"),
            "final": StagePreviewPayload(placeholder_text="等待表型提取", status_text=f"{view_name} | 等待表型提取"),
        }

    if self.preview_mode == "preprocess":
        calibration_status = "已校正" if calibrated_array is not None else "未校正"
        return {
            "original": StagePreviewPayload(image_path=raw_path, status_text=f"{view_name} | 已加载"),
            "calibrated": StagePreviewPayload(
                image_array=calibrated_array,
                image_path=raw_path if calibrated_array is None else None,
                placeholder_text="校正失败",
                status_text=f"{view_name} | {calibration_status}",
            ),
            "mask": StagePreviewPayload(placeholder_text="等待表型提取", status_text=f"{view_name} | 等待表型提取"),
            "final": StagePreviewPayload(placeholder_text="等待表型提取", status_text=f"{view_name} | 等待表型提取"),
        }

    calibration_status = "已校正" if calibrated_array is not None else "未校正"
    final_status = "已生成" if final_path is not None else "使用近似结果图"
    return {
        "original": StagePreviewPayload(image_path=raw_path, status_text=f"{view_name} | 已加载"),
        "calibrated": StagePreviewPayload(
            image_array=calibrated_array,
            image_path=raw_path if calibrated_array is None else None,
            status_text=f"{view_name} | {calibration_status}",
        ),
        "mask": StagePreviewPayload(
            image_path=mask_path,
            placeholder_text="缺少背景消除结果",
            status_text=f"{view_name} | 已分割" if mask_path is not None else f"{view_name} | 缺少分割图",
        ),
        "final": StagePreviewPayload(
            image_path=final_path or mask_path,
            placeholder_text="缺少表型提取结果",
            status_text=f"{view_name} | {final_status}",
        ),
    }
```

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/test_stage_preview.py tests/test_gui_smoke.py -q`

Expected: PASS with all preview-state and GUI smoke tests green.

- [ ] **Step 3: Run full regression suite**

Run: `pytest -q tests`

Expected: PASS with the full suite green, including pipeline、exporter、organs 和现有 smoke tests。

- [ ] **Step 4: Commit**

```bash
# 当前目录不是 git 仓库；若 git 可用则执行：
git add gui/stage_preview.py gui/main_window.py tests/test_stage_preview.py tests/test_gui_smoke.py
git commit -m "feat: redesign gui preview panel"
```
