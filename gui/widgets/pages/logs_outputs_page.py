from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from gui.services import ResultLoader
from gui.widgets.common import Section


class LogsOutputsPage(QWidget):
    def __init__(self, project_root: Path, result_loader: ResultLoader) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.result_loader = result_loader
        self._build_ui()
        self.set_output_dir(str(self.project_root / "outputs"))

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(16)

        title = QLabel("日志 / 输出")
        title.setObjectName("pageTitle")
        subtitle = QLabel("浏览当前实验目录，并查看日志或 JSON 文件。")
        subtitle.setObjectName("pageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        source = Section("输出目录")
        row = QHBoxLayout()
        self.output_dir = QLineEdit()
        choose = QPushButton("浏览")
        reload_button = QPushButton("重新加载")
        choose.clicked.connect(self._choose_output)
        reload_button.clicked.connect(self.reload)
        row.addWidget(self.output_dir, 1)
        row.addWidget(choose)
        row.addWidget(reload_button)
        source.layout.addLayout(row)
        layout.addWidget(source)

        body = QHBoxLayout()
        tree_section = Section("目录结构")
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["文件", "大小"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemSelectionChanged.connect(self._show_selected_file)
        tree_section.layout.addWidget(self.tree)

        content_section = Section("文件内容 / 最佳试验")
        self.content = QPlainTextEdit()
        self.content.setReadOnly(True)
        content_section.layout.addWidget(self.content)
        body.addWidget(tree_section, 1)
        body.addWidget(content_section, 2)
        layout.addLayout(body, 1)

    def set_output_dir(self, path: str) -> None:
        self.output_dir.setText(path)
        self.reload()

    def reload(self) -> None:
        self.tree.clear()
        root = Path(self.output_dir.text().strip())
        if not root.is_absolute():
            root = self.project_root / root
        if not root.exists():
            self.content.setPlainText(f"输出目录不存在: {root}")
            return
        root_item = QTreeWidgetItem([root.name, ""])
        root_item.setData(0, 0x0100, str(root))
        self.tree.addTopLevelItem(root_item)
        self._add_children(root_item, root, depth=0)
        root_item.setExpanded(True)
        self._show_best_summary(root)

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir.text())
        if path:
            self.output_dir.setText(path)
            self.reload()

    def _add_children(self, parent: QTreeWidgetItem, path: Path, *, depth: int) -> None:
        if depth >= 4:
            return
        children = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        for child in children[:300]:
            size = f"{child.stat().st_size} B" if child.is_file() else ""
            item = QTreeWidgetItem([child.name, size])
            item.setData(0, 0x0100, str(child))
            parent.addChild(item)
            if child.is_dir():
                self._add_children(item, child, depth=depth + 1)

    def _show_selected_file(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        path_text = items[0].data(0, 0x0100)
        if not path_text:
            return
        path = Path(path_text)
        if path.is_dir():
            self._show_best_summary(path)
            return
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.content.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
                return
            except Exception as exc:
                self.content.setPlainText(f"JSON 解析失败: {exc}\n\n{path.read_text(encoding='utf-8', errors='replace')[:12000]}")
                return
        if path.suffix.lower() in {".log", ".txt", ".csv", ".jsonl", ".md", ".yaml", ".yml"}:
            self.content.setPlainText(path.read_text(encoding="utf-8", errors="replace")[:20000])
        else:
            self.content.setPlainText(f"已选择文件: {path}\n二进制文件或暂不支持预览的类型。")

    def _show_best_summary(self, root: Path) -> None:
        data = self.result_loader.load(root)
        best = data.get("best_trial", {})
        payload = {
            "output_dir": str(root),
            "best_trial": best,
            "best_policy": data.get("policy") or data.get("best_policy"),
            "summary": data.get("summary"),
            "read_errors": data.get("read_errors", []),
        }
        self.content.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
