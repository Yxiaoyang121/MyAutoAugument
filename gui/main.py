from __future__ import annotations

import argparse
import faulthandler
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QLibraryInfo, QLocale, QTimer, QTranslator
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.widgets.main_window import MainWindow  # noqa: E402

LOG_DIR = PROJECT_ROOT / "outputs" / "gui_logs"
CRASH_LOG = LOG_DIR / "crash.log"
CRASH_FILE_HANDLE = None


def install_crash_logging() -> None:
    global CRASH_FILE_HANDLE
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CRASH_FILE_HANDLE = CRASH_LOG.open("a", encoding="utf-8")
    faulthandler.enable(file=CRASH_FILE_HANDLE, all_threads=True)

    def excepthook(exc_type, exc, tb) -> None:
        with CRASH_LOG.open("a", encoding="utf-8") as handle:
            handle.write("\n=== Unhandled GUI exception ===\n")
            traceback.print_exception(exc_type, exc, tb, file=handle)
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = excepthook


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoAugment Qt 中文桌面界面")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="创建主窗口，短暂处理事件后退出。",
    )
    return parser


def install_chinese_ui(app: QApplication) -> None:
    QLocale.setDefault(QLocale(QLocale.Language.Chinese, QLocale.Country.China))
    app.setFont(QFont("Microsoft YaHei", 9))

    translations_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    translators: list[QTranslator] = []
    for catalog in ["qt_zh_CN", "qtbase_zh_CN"]:
        translator = QTranslator(app)
        if translator.load(catalog, translations_path):
            app.installTranslator(translator)
            translators.append(translator)
        else:
            qm_path = Path(translations_path) / f"{catalog}.qm"
            print(f"warning: Qt 中文翻译文件不存在或加载失败: {qm_path}", file=sys.stderr)
    app._autoaugment_translators = translators  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> int:
    install_crash_logging()
    args = build_parser().parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    install_chinese_ui(app)
    app.setApplicationName("AutoAugment 桌面端")
    app.setOrganizationName("AutoAugment")
    app.setStyle("Fusion")

    window = MainWindow(project_root=PROJECT_ROOT, auto_inspect_dataset=not args.smoke_test)
    app.aboutToQuit.connect(window.shutdown)
    window.showMaximized()

    if args.smoke_test:
        QTimer.singleShot(250, app.quit)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
