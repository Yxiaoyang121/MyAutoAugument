from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QSlider, QVBoxLayout, QWidget

from gui.widgets.section_card import SectionCard


PARAM_RANGES: dict[str, tuple[float, float, int, str, str, str]] = {
    "max_delta": (0.0, 2.0, 2, "0", "1.0", "2.0"),
    "max_angle": (0.0, 180.0, 1, "0", "90", "180"),
    "max_translate": (0.0, 1.0, 2, "0", "0.5", "1.0"),
    "max_scale_delta": (0.0, 1.0, 2, "0", "0.5", "1.0"),
    "max_shear": (0.0, 45.0, 1, "0", "15", "45"),
    "max_std": (0.0, 1.0, 2, "0", "0.5", "1.0"),
    "max_amount": (0.0, 1.0, 2, "0", "0.5", "1.0"),
    "max_crop_fraction": (0.0, 0.95, 2, "0", "0.5", "0.95"),
    "min_visibility": (0.0, 1.0, 2, "0", "0.5", "1.0"),
    "max_kernel": (1.0, 31.0, 0, "1", "15", "31"),
    "amount": (0.0, 5.0, 2, "0", "2.5", "5"),
    "sigma": (0.0, 10.0, 2, "0", "5", "10"),
}

HELP_TEXT = {
    "brightness": "随机调整图像亮度，max_delta 控制亮度扰动上限，bbox 不变。",
    "contrast": "随机调整图像对比度，max_delta 控制对比度扰动范围，bbox 不变。",
    "gamma": "进行非线性亮度调整，min_gamma/max_gamma 控制采样范围。",
    "rotate": "随机旋转图像，max_angle 控制旋转角度范围，bbox 会同步变换。",
    "horizontal_flip": "水平翻转图像，bbox 会按图像宽度重新映射。",
    "vertical_flip": "垂直翻转图像，bbox 会按图像高度重新映射。",
    "scale": "以图像中心缩放，max_delta 控制缩放扰动幅度，bbox 会同步变换。",
    "translate": "平移图像内容，max_translate 控制位移上限，bbox 会同步变换。",
    "gaussian_noise": "加入高斯噪声，max_std 控制噪声强度，bbox 不变。",
    "blur": "模糊图像以模拟离焦，通常不改变 bbox。",
    "motion_blur": "模拟运动模糊，通常不改变 bbox。",
}


class OperationParamEditor(SectionCard):
    operation_changed = Signal(dict)

    def __init__(self) -> None:
        super().__init__("操作参数编辑")
        self.operation: dict | None = None
        self._loading = False
        self._pending_operation: dict | None = None
        self.param_widgets: dict[str, tuple[str, QWidget, QWidget | None]] = {}
        self.emit_timer = QTimer(self)
        self.emit_timer.setSingleShot(True)
        self.emit_timer.setInterval(120)
        self.emit_timer.timeout.connect(self._flush_change)

        self.name_input = QLineEdit()
        self.name_input.setReadOnly(True)
        self.name_input.setObjectName("policyInput")
        self.type_badge = QLabel("-")
        self.type_badge.setObjectName("paramTypeBadge")
        self.type_badge.setMaximumHeight(30)
        self.bbox_badge = QLabel("-")
        self.bbox_badge.setObjectName("paramBboxBadge")
        self.bbox_badge.setMaximumHeight(30)

        self.prob_spin, self.prob_slider = self._numeric_control(0.0, 1.0, 0.5, decimals=2)
        self.strength_spin, self.strength_slider = self._numeric_control(0.0, 1.0, 0.5, decimals=2)

        self.common_form = QGridLayout()
        self.common_form.setContentsMargins(0, 0, 0, 0)
        self.common_form.setHorizontalSpacing(10)
        self.common_form.setVerticalSpacing(7)
        self.param_form = QGridLayout()
        self.param_form.setContentsMargins(0, 0, 0, 0)
        self.param_form.setHorizontalSpacing(10)
        self.param_form.setVerticalSpacing(7)

        self.params_json = QPlainTextEdit()
        self.params_json.setObjectName("paramJsonBox")
        self.params_json.setMaximumHeight(74)
        self.help_detail = QLabel("")
        self.help_detail.setObjectName("paramHelpText")
        self.help_detail.setWordWrap(True)

        self._build_form()
        self._connect_common()

    def set_operation(self, operation: dict | None) -> None:
        self.operation = dict(operation) if operation else None
        self.emit_timer.stop()
        self._pending_operation = None
        self._loading = True
        self._clear_param_form()
        if not self.operation:
            self.name_input.setText("")
            self.type_badge.setText("-")
            self.bbox_badge.setText("-")
            self.params_json.setPlainText("请选择一个增强操作以编辑参数。")
            self.help_detail.setText("请选择一个增强操作以编辑参数。")
            self._loading = False
            return

        params = dict(self.operation.get("params", {}) or {})
        self.name_input.setText(str(self.operation.get("name", "")))
        self.type_badge.setText(str(self.operation.get("category", "-")))
        effect = str(self.operation.get("bbox_effect", "不改变"))
        self.bbox_badge.setText(effect)
        self.bbox_badge.setProperty("effect", "change" if effect == "会改变" else "stable")
        self.bbox_badge.style().unpolish(self.bbox_badge)
        self.bbox_badge.style().polish(self.bbox_badge)
        self.prob_spin.setValue(float(self.operation.get("prob", 0.5)))
        self.prob_slider.setValue(int(round(float(self.operation.get("prob", 0.5)) * 100)))
        self.strength_spin.setValue(float(self.operation.get("strength", 0.5)))
        self.strength_slider.setValue(int(round(float(self.operation.get("strength", 0.5)) * 100)))

        self._build_param_rows(params)
        self.params_json.setPlainText(json.dumps(params, ensure_ascii=False, indent=2))
        name = str(self.operation.get("name", ""))
        help_name = {"gaussian_blur": "blur", "median_blur": "blur", "affine": "shear"}.get(name, name)
        self.help_detail.setText(HELP_TEXT.get(help_name, "根据后端注册的参数调整该增强操作。"))
        self._loading = False

    def _build_form(self) -> None:
        self._add_row(self.common_form, 0, "操作名称", self.name_input)
        self._add_badge_row(self.common_form, 1, "操作类型", self.type_badge)
        self._add_badge_row(self.common_form, 2, "bbox 影响", self.bbox_badge)
        self._add_slider_row(self.common_form, 3, "概率（prob）", self.prob_spin, self.prob_slider, "0", "0.5", "1.0")
        self._add_slider_row(self.common_form, 4, "强度（strength）", self.strength_spin, self.strength_slider, "0", "0.5", "1.0")
        self.body.addLayout(self.common_form)

        params_title = QLabel("增强参数")
        params_title.setObjectName("infoBlockTitle")
        self.body.addWidget(params_title)
        self.body.addLayout(self.param_form)

        info = QFrame()
        info.setObjectName("paramHelpBox")
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(8, 5, 8, 5)
        info_layout.addWidget(self.help_detail)
        self.body.addWidget(info)
        self.body.addStretch(1)

    def _build_param_rows(self, params: dict[str, Any]) -> None:
        if not params:
            empty = QLabel("该操作无额外参数。")
            empty.setObjectName("emptyDetail")
            self.param_form.addWidget(empty, 0, 1)
            return
        for row, (name, value) in enumerate(params.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                minimum, maximum, decimals, left, mid, right = self._range_for(name, float(value))
                spin, slider = self._numeric_control(minimum, maximum, float(value), decimals=decimals)
                self.param_widgets[name] = ("number", spin, slider)
                spin.valueChanged.connect(lambda _, key=name: self._on_param_changed(key))
                slider.valueChanged.connect(lambda _, key=name: self._on_param_slider_changed(key))
                self._add_slider_row(self.param_form, row, name, spin, slider, left, mid, right)
            else:
                line = QLineEdit(json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value))
                line.setObjectName("policyInput")
                line.editingFinished.connect(lambda key=name: self._on_param_changed(key))
                self.param_widgets[name] = ("text", line, None)
                self._add_row(self.param_form, row, name, line)

    def _add_row(self, form: QGridLayout, row: int, label: str, widget) -> None:
        label_widget = QLabel(label)
        label_widget.setObjectName("paramLabel")
        form.addWidget(label_widget, row, 0, Qt.AlignmentFlag.AlignTop)
        form.addWidget(widget, row, 1)

    def _add_badge_row(self, form: QGridLayout, row: int, label: str, widget: QLabel) -> None:
        box = QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(widget)
        box.addStretch(1)
        label_widget = QLabel(label)
        label_widget.setObjectName("paramLabel")
        form.addWidget(label_widget, row, 0, Qt.AlignmentFlag.AlignTop)
        form.addLayout(box, row, 1)

    def _add_slider_row(self, form: QGridLayout, row: int, label: str, spin: QDoubleSpinBox, slider: QSlider, left: str, mid: str, right: str) -> None:
        label_widget = QLabel(label)
        label_widget.setObjectName("paramLabel")
        container = QVBoxLayout()
        container.setSpacing(1)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        top.addWidget(spin)
        top.addWidget(slider, 1)
        ticks = QHBoxLayout()
        ticks.setContentsMargins(54, 0, 0, 0)
        for text in [left, mid, right]:
            tick = QLabel(text)
            tick.setObjectName("sliderTick")
            ticks.addWidget(tick)
        container.addLayout(top)
        container.addLayout(ticks)
        form.addWidget(label_widget, row, 0, Qt.AlignmentFlag.AlignTop)
        form.addLayout(container, row, 1)

    def _numeric_control(self, minimum: float, maximum: float, value: float, decimals: int = 2) -> tuple[QDoubleSpinBox, QSlider]:
        spin = QDoubleSpinBox()
        spin.setObjectName("paramSpin")
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.05 if maximum <= 2.0 else 1.0)
        spin.setValue(value)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setObjectName("paramSlider")
        slider.setRange(0, 100)
        slider.setValue(self._value_to_slider(value, minimum, maximum))
        slider.setTracking(False)
        return spin, slider

    def _connect_common(self) -> None:
        self.prob_spin.valueChanged.connect(lambda value: self._sync_spin_slider(self.prob_slider, value, 0.0, 1.0))
        self.prob_slider.valueChanged.connect(lambda value: self._sync_slider_spin(self.prob_spin, value, 0.0, 1.0))
        self.strength_spin.valueChanged.connect(lambda value: self._sync_spin_slider(self.strength_slider, value, 0.0, 1.0))
        self.strength_slider.valueChanged.connect(lambda value: self._sync_slider_spin(self.strength_spin, value, 0.0, 1.0))
        self.prob_spin.valueChanged.connect(lambda _: self._emit_change())
        self.strength_spin.valueChanged.connect(lambda _: self._emit_change())

    def _on_param_slider_changed(self, key: str) -> None:
        if self._loading:
            return
        kind, spin, slider = self.param_widgets[key]
        if kind != "number" or not isinstance(spin, QDoubleSpinBox) or not isinstance(slider, QSlider):
            return
        minimum, maximum = spin.minimum(), spin.maximum()
        spin.blockSignals(True)
        spin.setValue(self._slider_to_value(slider.value(), minimum, maximum))
        spin.blockSignals(False)
        self._on_param_changed(key)

    def _on_param_changed(self, key: str) -> None:
        if self._loading:
            return
        kind, widget, slider = self.param_widgets[key]
        if kind == "number" and isinstance(widget, QDoubleSpinBox) and isinstance(slider, QSlider):
            slider.blockSignals(True)
            slider.setValue(self._value_to_slider(widget.value(), widget.minimum(), widget.maximum()))
            slider.blockSignals(False)
        self._update_json()
        self._emit_change()

    def _sync_spin_slider(self, slider: QSlider, value: float, minimum: float, maximum: float) -> None:
        if self._loading:
            return
        slider.blockSignals(True)
        slider.setValue(self._value_to_slider(value, minimum, maximum))
        slider.blockSignals(False)

    def _sync_slider_spin(self, spin: QDoubleSpinBox, slider_value: int, minimum: float, maximum: float) -> None:
        if self._loading:
            return
        spin.blockSignals(True)
        spin.setValue(self._slider_to_value(slider_value, minimum, maximum))
        spin.blockSignals(False)
        self._emit_change()

    def _params_from_controls(self) -> dict:
        params: dict[str, Any] = {}
        for key, (kind, widget, _) in self.param_widgets.items():
            if kind == "number" and isinstance(widget, QDoubleSpinBox):
                params[key] = int(widget.value()) if widget.decimals() == 0 else float(widget.value())
            elif kind == "text" and isinstance(widget, QLineEdit):
                text = widget.text().strip()
                try:
                    params[key] = json.loads(text)
                except json.JSONDecodeError:
                    params[key] = text
        return params

    def _update_json(self) -> None:
        if self._loading or not self.operation:
            return
        self.params_json.setPlainText(json.dumps(self._params_from_controls(), ensure_ascii=False, indent=2))

    def _emit_change(self) -> None:
        if self._loading or not self.operation:
            return
        updated = dict(self.operation)
        updated["prob"] = float(self.prob_spin.value())
        updated["strength"] = float(self.strength_spin.value())
        updated["params"] = self._params_from_controls()
        self.operation = updated
        self.params_json.setPlainText(json.dumps(updated["params"], ensure_ascii=False, indent=2))
        self._pending_operation = updated
        self.emit_timer.start()

    def _flush_change(self) -> None:
        if self._loading or not self._pending_operation:
            return
        self.operation_changed.emit(dict(self._pending_operation))
        self._pending_operation = None

    def commit_pending_change(self) -> None:
        if self.emit_timer.isActive():
            self.emit_timer.stop()
        self._flush_change()

    def _range_for(self, key: str, value: float) -> tuple[float, float, int, str, str, str]:
        if key in PARAM_RANGES:
            return PARAM_RANGES[key]
        maximum = max(1.0, value * 2.0)
        return 0.0, maximum, 2, "0", f"{maximum / 2:g}", f"{maximum:g}"

    def _value_to_slider(self, value: float, minimum: float, maximum: float) -> int:
        if maximum <= minimum:
            return 0
        return int(round((value - minimum) / (maximum - minimum) * 100))

    def _slider_to_value(self, value: int, minimum: float, maximum: float) -> float:
        return minimum + (maximum - minimum) * value / 100.0

    def _clear_param_form(self) -> None:
        self.param_widgets.clear()
        while self.param_form.count():
            item = self.param_form.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
            elif item.layout():
                self._clear_layout(item.layout())

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
            elif item.layout():
                self._clear_layout(item.layout())
