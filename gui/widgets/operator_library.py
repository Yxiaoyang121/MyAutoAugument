from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout

from gui.widgets.section_card import SectionCard


OPERATOR_GROUPS = {
    "光照 / 颜色": ["brightness", "contrast", "gamma"],
    "几何变换": ["horizontal_flip", "vertical_flip", "rotate", "scale", "translate", "affine", "crop"],
    "噪声 / 模糊": ["gaussian_noise", "gaussian_blur", "motion_blur", "median_blur", "salt_pepper_noise"],
    "工业增强": ["clahe", "sharpen", "cutout", "random_erasing"],
}

for _operator_names in OPERATOR_GROUPS.values():
    if "clahe" not in _operator_names:
        continue
    for _operator_name in ("local_contrast", "copy_paste"):
        if _operator_name not in _operator_names:
            _operator_names.append(_operator_name)
    break


class OperatorGroupHeader(QFrame):
    toggled = Signal(str)

    def __init__(self, title: str, expanded: bool) -> None:
        super().__init__()
        self.title = title
        self.expanded = expanded
        self.setObjectName("operatorGroupHeader")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        label = QLabel(title)
        label.setObjectName("operatorGroupTitle")
        self.arrow = QLabel("v" if expanded else ">")
        self.arrow.setObjectName("operatorGroupArrow")

        layout.addWidget(label, 1)
        layout.addWidget(self.arrow)

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self.arrow.setText("v" if expanded else ">")

    def mousePressEvent(self, event) -> None:
        self.toggled.emit(self.title)
        super().mousePressEvent(event)


class OperatorRow(QFrame):
    activated = Signal(str)

    def __init__(self, name: str, selected: bool = False) -> None:
        super().__init__()
        self.name = name
        self.setObjectName("operatorRow")
        self.setProperty("selected", selected)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(8)

        icon = QLabel(self._icon_for(name))
        icon.setObjectName("operatorIcon")
        label = QLabel(name)
        label.setObjectName("operatorName")

        layout.addWidget(icon)
        layout.addWidget(label, 1)

    def mouseDoubleClickEvent(self, event) -> None:
        self.activated.emit(self.name)
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:
        self.activated.emit(self.name)
        super().mousePressEvent(event)

    def _icon_for(self, name: str) -> str:
        if name in {"brightness", "contrast", "gamma"}:
            return "C"
        if name in {"horizontal_flip", "vertical_flip", "rotate", "scale", "translate", "affine", "crop"}:
            return "G"
        if "blur" in name or "noise" in name:
            return "N"
        return "I"


class OperatorLibrary(SectionCard):
    operator_activated = Signal(str)

    def __init__(self) -> None:
        super().__init__("增强算子库")
        self.rows: list[OperatorRow] = []
        self.group_rows: dict[str, list[OperatorRow]] = {}
        self.group_headers: dict[str, OperatorGroupHeader] = {}

        self.search = QLineEdit()
        self.search.setObjectName("operatorSearch")
        self.search.setPlaceholderText("搜索算子...")
        self.search.textChanged.connect(self._filter_rows)
        self.body.addWidget(self.search)

        for index, (group, names) in enumerate(OPERATOR_GROUPS.items()):
            self._add_group(group, names, expanded=index == 0)

        hint = QFrame()
        hint.setObjectName("operatorHint")
        hint_layout = QVBoxLayout(hint)
        hint_layout.setContentsMargins(8, 4, 8, 4)
        hint_layout.setSpacing(0)

        title = QLabel("双击算子添加到策略流水线")
        title.setObjectName("operatorHintTitle")
        hint_layout.addWidget(title)

        self.body.addWidget(hint)
        self.body.addStretch(1)

    def _add_group(self, title: str, names: list[str], *, expanded: bool = True) -> None:
        header = OperatorGroupHeader(title, expanded)
        header.toggled.connect(self._toggle_group)
        self.group_headers[title] = header
        self.body.addWidget(header)

        self.group_rows[title] = []
        for name in names:
            row = OperatorRow(name, selected=name == "brightness")
            row.activated.connect(self.operator_activated.emit)
            self.rows.append(row)
            self.group_rows[title].append(row)
            row.setVisible(expanded)
            self.body.addWidget(row)

    def _filter_rows(self, text: str) -> None:
        needle = text.strip().lower()
        for group, rows in self.group_rows.items():
            header = self.group_headers[group]
            any_visible = False
            for row in rows:
                visible = bool(needle and needle in row.name.lower()) or (not needle and header.expanded)
                row.setVisible(visible)
                any_visible = any_visible or visible
            header.setVisible(not needle or any_visible)

    def _toggle_group(self, group: str) -> None:
        if group not in self.group_rows:
            return
        header = self.group_headers[group]
        header.set_expanded(not header.expanded)
        self._filter_rows(self.search.text())
