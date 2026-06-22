from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.services import ResultLoader
from gui.services.app_state import AppState
from gui.widgets.pages.analysis_page import AnalysisPage
from gui.widgets.pages.dashboard_page import DashboardPage
from gui.widgets.pages.dataset_page import DatasetPage
from gui.widgets.pages.experiment_page import ExperimentPage
from gui.widgets.pages.policy_page import PolicyPage
from gui.widgets.pages.preview_page import PreviewPage
from gui.widgets.status_badge import StatusBadge


MOCK_OUTPUT_DIR = r"D:\AutoAugment\outputs"
MOCK_UPDATED_AT = "2026-06-16 14:30:22"
MOCK_DATASET_UPDATED_AT = "2026-06-16 14:28:10"
MOCK_CONFIG_FILE = "gui_experiment.json"

PAGE_HEADERS = {
    0: ("AutoAugment 实验控制台", "基于 AutoAugment 的自动增强实验平台", MOCK_UPDATED_AT),
    1: ("数据集", "数据集检查与统计", MOCK_DATASET_UPDATED_AT),
    2: ("策略", "管理和编辑数据增强策略", MOCK_UPDATED_AT),
    3: ("预览", "增强效果预览与 bbox 检查", MOCK_UPDATED_AT),
    4: ("实验", "配置并运行后端训练任务", MOCK_UPDATED_AT),
    5: ("分析", "分析实验结果、诊断问题并优化策略", MOCK_UPDATED_AT),
}


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path, *, auto_inspect_dataset: bool = True) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.auto_inspect_dataset = auto_inspect_dataset
        self.result_loader = ResultLoader(self.project_root)
        self.app_state = AppState()
        self.setWindowTitle("AutoAugment 桌面端")
        self.resize(1500, 900)
        self.setMinimumSize(1180, 680)
        self._build_ui()
        self._apply_styles()

    def _build_ui(self) -> None:
        shell = QWidget()
        outer = QHBoxLayout(shell)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sidebar = self._build_sidebar()

        content = QWidget()
        content.setObjectName("contentArea")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        content_layout.addWidget(self._build_topbar())
        self.stack = QStackedWidget()
        self.stack.setObjectName("contentStack")
        self._build_pages()
        content_layout.addWidget(self.stack, 1)
        content_layout.addWidget(self._build_footer())

        outer.addWidget(sidebar)
        outer.addWidget(content, 1)
        self.setCentralWidget(shell)

        self._connect_signals()

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebarFrame")
        sidebar.setFixedWidth(168)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 10)
        sidebar_layout.setSpacing(0)

        brand = QFrame()
        brand.setObjectName("brand")
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(14, 12, 12, 10)
        brand_layout.setSpacing(8)
        logo = QLabel("A")
        logo.setObjectName("brandLogo")
        brand_text = QLabel("AutoAugment")
        brand_text.setObjectName("brandText")
        brand_layout.addWidget(logo)
        brand_layout.addWidget(brand_text, 1)
        sidebar_layout.addWidget(brand)

        self.nav = QListWidget()
        self.nav.setObjectName("sidebar")
        self.nav.setFrameShape(QFrame.Shape.NoFrame)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for name in ["总览", "数据集", "策略", "预览", "实验", "分析"]:
            item = QListWidgetItem(name)
            item.setSizeHint(QSize(168, 38))
            self.nav.addItem(item)
        sidebar_layout.addWidget(self.nav, 1)

        for name in ["设置", "关于"]:
            button = QPushButton(name)
            button.setObjectName("sideButton")
            sidebar_layout.addWidget(button)

        return sidebar

    def _build_topbar(self) -> QFrame:
        header = QFrame()
        header.setObjectName("topbar")
        header.setFixedHeight(60)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 0, 18, 0)
        header_layout.setSpacing(14)

        title_block = QVBoxLayout()
        title_block.setContentsMargins(0, 0, 0, 0)
        title_block.setSpacing(4)
        self.title_label = QLabel("AutoAugment 实验控制台")
        self.title_label.setObjectName("appTitle")
        self.subtitle_label = QLabel("基于 AutoAugment 的自动增强实验平台")
        self.subtitle_label.setObjectName("appSubtitle")
        title_block.addWidget(self.title_label)
        title_block.addWidget(self.subtitle_label)
        header_layout.addLayout(title_block, 1)

        header_layout.addWidget(self._status_meta("后端状态:", "正常"))
        self.output_meta = QLabel(f"输出目录：{MOCK_OUTPUT_DIR}")
        self.output_meta.setObjectName("topbarMeta")
        self.update_meta = QLabel(f"最近更新：{MOCK_UPDATED_AT}")
        self.update_meta.setObjectName("topbarMeta")
        header_layout.addWidget(self.output_meta)
        header_layout.addWidget(self.update_meta)
        return header

    def _status_meta(self, label: str, value: str) -> QWidget:
        widget = QWidget()
        widget.setObjectName("topbarMetaGroup")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        prefix = QLabel(label)
        prefix.setObjectName("topbarMeta")
        dot = QLabel("●")
        dot.setObjectName("statusDot")
        status = QLabel(value)
        status.setObjectName("topbarMeta")
        layout.addWidget(prefix)
        layout.addWidget(dot)
        layout.addWidget(status)
        return widget

    def _build_pages(self) -> None:
        self.dashboard_page = DashboardPage(self.project_root, self.result_loader)
        self.dataset_page = DatasetPage(self.project_root, auto_inspect=self.auto_inspect_dataset)
        self.policy_page = PolicyPage(self.project_root)
        self.preview_page = PreviewPage(self.project_root)
        self.experiment_page = ExperimentPage(self.project_root, self.app_state)
        self.analysis_page = AnalysisPage(self.project_root, self.result_loader, self.app_state)

        for page in [
            self.dashboard_page,
            self.dataset_page,
            self.policy_page,
            self.preview_page,
            self.experiment_page,
            self.analysis_page,
        ]:
            self.stack.addWidget(page)

    def _build_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("bottomStatusBar")
        footer.setFixedHeight(34)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(18, 0, 14, 0)
        layout.setSpacing(14)

        self.footer_output = QLabel(f"输出目录：{MOCK_OUTPUT_DIR}")
        self.footer_output.setObjectName("footerMeta")
        layout.addWidget(self.footer_output)
        layout.addWidget(self._separator())
        layout.addWidget(self._footer_status())
        layout.addWidget(self._separator())
        layout.addWidget(self._footer_label(f"配置文件：{MOCK_CONFIG_FILE}"))
        layout.addWidget(self._separator())
        self.footer_update = self._footer_label(f"最近更新：{MOCK_UPDATED_AT}")
        layout.addWidget(self.footer_update)
        layout.addStretch(1)
        layout.addWidget(StatusBadge("就绪", "success"))
        return footer

    def _footer_status(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("footerStatusGroup")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._footer_label("后端状态："))
        dot = QLabel("●")
        dot.setObjectName("statusDot")
        layout.addWidget(dot)
        layout.addWidget(self._footer_label("正常"))
        return widget

    def _footer_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("footerMeta")
        return label

    def _separator(self) -> QLabel:
        label = QLabel("|")
        label.setObjectName("footerSeparator")
        return label

    def _connect_signals(self) -> None:
        self.nav.currentRowChanged.connect(self._switch_page)
        self.nav.setCurrentRow(5)
        self.experiment_page.output_dir_changed.connect(self.analysis_page.set_output_dir)
        self.experiment_page.output_dir_changed.connect(self._set_output_dir)
        self.experiment_page.output_dir_changed.connect(self.app_state.set_output_dir)
        self.policy_page.policy_changed.connect(self.preview_page.set_policy)
        self.policy_page.policy_changed.connect(self.experiment_page.set_policy)
        self.policy_page.policy_changed.connect(self.app_state.set_policy)
        self.policy_page.preview_requested.connect(lambda: self.nav.setCurrentRow(3))
        self.policy_page.experiment_requested.connect(lambda: self.nav.setCurrentRow(4))
        self.preview_page.policy_changed.connect(self.policy_page.set_policy)
        self.preview_page.policy_changed.connect(self.experiment_page.set_policy)
        self.preview_page.policy_changed.connect(self.app_state.set_policy)
        self.preview_page.policy_editor_requested.connect(lambda: self.nav.setCurrentRow(2))
        self.dashboard_page.navigate_requested.connect(self.nav.setCurrentRow)
        self.dataset_page.dataset_changed.connect(self.experiment_page.set_dataset_path)
        self.dataset_page.dataset_changed.connect(self.app_state.set_dataset)
        self.dataset_page.dataset_changed.connect(lambda path, info: self.preview_page.set_dataset_path(path))
        self.app_state.result_dir_changed.connect(self.analysis_page.set_output_dir)
        self.preview_page.set_policy(self.policy_page.current_policy_dict())
        self.experiment_page.set_policy(self.policy_page.current_policy_dict())
        self.experiment_page.set_dataset_path(self.dataset_page.current_dataset_path(), self.dataset_page.inspection)

    def _switch_page(self, row: int) -> None:
        self.stack.setCurrentIndex(row)
        title, subtitle, updated_at = PAGE_HEADERS.get(row, PAGE_HEADERS[0])
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)
        self.update_meta.setText(f"最近更新：{updated_at}")
        self.footer_update.setText(f"最近更新：{updated_at}")

    def _set_output_dir(self, path: str) -> None:
        self.output_meta.setText(f"输出目录：{path}")
        self.footer_output.setText(f"输出目录：{path}")

    def shutdown(self) -> None:
        if hasattr(self, "dataset_page"):
            self.dataset_page.shutdown()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.shutdown()
        super().closeEvent(event)

    def _apply_styles(self) -> None:
        qss_path = self.project_root / "gui" / "styles" / "app.qss"
        self.setStyleSheet(qss_path.read_text(encoding="utf-8"))
