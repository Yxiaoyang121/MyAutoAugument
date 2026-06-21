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


def test_experiment_page_hides_ablation_modes_by_default() -> None:
    if importlib.util.find_spec("PySide6") is None:
        pytest.skip("PySide6 is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from gui.models.experiment_config import TRAINING_MODE_FIXED_CATF, TRAINING_MODE_LEGACY_SEARCH
    from gui.widgets.experiment_config_panel import ExperimentConfigPanel

    app = QApplication.instance() or QApplication([sys.argv[0]])
    panel = ExperimentConfigPanel()
    try:
        default_modes = [panel.run_mode.itemData(index) for index in range(panel.run_mode.count())]
        assert len(default_modes) == 3
        assert TRAINING_MODE_FIXED_CATF not in default_modes
        assert TRAINING_MODE_LEGACY_SEARCH not in default_modes

        panel.show_ablation_modes.setChecked(True)
        app.processEvents()
        advanced_modes = [panel.run_mode.itemData(index) for index in range(panel.run_mode.count())]
        assert TRAINING_MODE_FIXED_CATF in advanced_modes
        assert TRAINING_MODE_LEGACY_SEARCH in advanced_modes
    finally:
        panel.close()
