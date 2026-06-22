from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_gui_main_smoke_test_command() -> None:
    if importlib.util.find_spec("PySide6") is None:
        pytest.skip("PySide6 is not installed")
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    result = subprocess.run(
        [sys.executable, "gui/main.py", "--smoke-test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_main_window_resizes_across_pages() -> None:
    if importlib.util.find_spec("PySide6") is None:
        pytest.skip("PySide6 is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from gui.widgets.main_window import MainWindow

    app = QApplication.instance() or QApplication([sys.argv[0]])
    root = Path(__file__).resolve().parents[2]
    window = MainWindow(root)
    try:
        for width, height in [(860, 560), (1180, 760), (1500, 920)]:
            window.resize(width, height)
            app.processEvents()
            for row in range(window.nav.count()):
                window.nav.setCurrentRow(row)
                app.processEvents()
                assert window.stack.currentIndex() == row
    finally:
        window.close()


def test_experiment_page_exposes_only_three_training_modes() -> None:
    if importlib.util.find_spec("PySide6") is None:
        pytest.skip("PySide6 is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from gui.models.experiment_config import (
        TRAINING_MODE_CUSTOM,
        TRAINING_MODE_PRESERVE_WEAK,
        TRAINING_MODE_YOLO_DEFAULT,
    )
    from gui.widgets.experiment_config_panel import ExperimentConfigPanel

    app = QApplication.instance() or QApplication([sys.argv[0]])
    panel = ExperimentConfigPanel()
    try:
        modes = [panel.run_mode.itemData(index) for index in range(panel.run_mode.count())]
        labels = [panel.run_mode.itemText(index) for index in range(panel.run_mode.count())]
        assert modes == [TRAINING_MODE_CUSTOM, TRAINING_MODE_YOLO_DEFAULT, TRAINING_MODE_PRESERVE_WEAK]
        assert labels == ["自定义增强策略训练", "YOLO 默认训练", "Preserve-Weak Image-only CATF"]
        assert panel.run_mode.currentData() == TRAINING_MODE_PRESERVE_WEAK
        assert not hasattr(panel, "show_ablation_modes")
    finally:
        panel.close()
