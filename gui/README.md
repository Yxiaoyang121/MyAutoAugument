# AutoAugment Qt GUI

This directory contains the desktop interface for the AutoAugment project.
The GUI is built with Python, PySide6, and Qt Widgets. Backend experiment code
is accessed only through the service layer in `gui/services`.

当前界面为中文界面。PySide6 本身支持中文显示，不需要安装额外的中文插件。
`gui/main.py` 会把默认字体设置为 Microsoft YaHei / 微软雅黑，并尝试加载
Qt 自带的 `qt_zh_CN.qm` 翻译文件，让 `QFileDialog`、`QMessageBox`
等标准控件在翻译文件可用时显示中文。如果 Qt 翻译文件不存在，程序不会崩溃，
只会在启动时输出 warning。

## Install

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r gui/requirements.txt
```

The GUI requirements currently constrain PySide6 below 6.8 to avoid known
Qt DLL loading issues observed with newer wheels in this Anaconda environment.

## Launch

```bash
python gui/main.py
```

For a non-interactive startup check:

```bash
python gui/main.py --smoke-test
```

## Local Experiment Runner

`gui/services/local_experiment_runner.py` owns local process execution. The
Experiment page passes a configuration dictionary to the runner, and the runner
starts the backend with `QProcess`.

The backend script can be selected in the Experiment page or configured with:

```bash
set AUTOAUGMENT_EXPERIMENT_SCRIPT=E:\path\to\run_experiment.py
```

If no script is configured, the runner searches these conventional paths:

- `scripts/run_experiment.py`
- `scripts/run_closed_loop_experiment.py`
- `src/run_experiment.py`
- `src/experiment.py`

## Remote Boundary

`gui/services/remote_experiment_client.py` is a placeholder for future cloud
training deployment. It reserves these methods:

- `submit_experiment`
- `get_task_status`
- `get_logs`
- `download_results`

## Result Files

`gui/services/result_loader.py` reads these JSON files from an output directory:

- `summary.json`
- `trial_record.json`
- `diagnosis.json`
- `policy.json`
- `metrics.json`

The first UI version uses mock result files under:

```text
gui/mock_outputs/demo_experiment/
```
