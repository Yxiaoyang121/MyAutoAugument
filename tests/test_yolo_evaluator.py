from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

from AutoAugment.formats.yolo import save_yolo_labels
from AutoAugment.policies import Policy
from AutoAugment.search import YoloCommandEvaluator, parse_yolo_metrics, write_yolo_data_yaml
from AutoAugment.utils import default_run_output_dir, resolve_output_dir, smoke_output_dir
from examples.run_policy_search import build_evaluator


def fresh_dir(name: str) -> Path:
    path = Path("outputs/tests/pytest_tmp") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def create_trial_dataset(trial_dir: Path) -> None:
    images_dir = trial_dir / "images"
    labels_dir = trial_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    cv2.rectangle(image, (12, 14), (40, 42), (180, 180, 180), -1)
    cv2.imwrite(str(images_dir / "sample.jpg"), image)
    save_yolo_labels(
        labels_dir / "sample.txt",
        np.asarray([2], dtype=np.int64),
        np.asarray([[12, 14, 40, 42]], dtype=np.float32),
        64,
        64,
    )


def fake_yolo_command() -> str:
    return f'"{sys.executable}" -c "print(\'mAP50: 0.934\'); print(\'mAP50-95: 0.712\')"'


def fake_yolo_command_with_workers() -> str:
    return (
        f'"{sys.executable}" -c "import sys; '
        "print('mAP50: 0.934'); print('mAP50-95: 0.712'); print('workers={workers}')\""
    )


def test_parse_yolo_metrics_from_table() -> None:
    text = """
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
all          100        200       0.91       0.88       0.93       0.71
"""
    metrics = parse_yolo_metrics(text)
    assert metrics["map50"] == 0.93
    assert metrics["map50_95"] == 0.71


def test_parse_yolo_metrics_from_ultralytics_table_keeps_map50_and_map50_95_separate() -> None:
    text = """
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
all          19         50       0.90       0.88       0.971       0.572
"""
    metrics = parse_yolo_metrics(text)
    assert metrics["map50"] == 0.971
    assert metrics["map50_95"] == 0.572


def test_parse_yolo_metrics_from_ultralytics_progress_table() -> None:
    text = """
\x1b[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.0it/s 1.0s
                   all         19         19      0.768      0.924      0.971      0.572
"""
    metrics = parse_yolo_metrics(text)
    assert metrics["map50"] == 0.971
    assert metrics["map50_95"] == 0.572


def test_parse_yolo_metrics_from_plain_lines() -> None:
    metrics = parse_yolo_metrics("mAP50: 0.934\nmAP50-95: 0.712")
    assert metrics["map50"] == 0.934
    assert metrics["map50_95"] == 0.712


def test_parse_yolo_metrics_from_ultralytics_metric_lines() -> None:
    metrics = parse_yolo_metrics("metrics/mAP50(B): 0.934\nmetrics/mAP50-95(B): 0.712")
    assert metrics["map50"] == 0.934
    assert metrics["map50_95"] == 0.712


def test_yolo_command_evaluator_scores_map50_and_writes_logs() -> None:
    trial_dir = fresh_dir("yolo_eval_map50")
    create_trial_dataset(trial_dir)
    evaluator = YoloCommandEvaluator(fake_yolo_command(), metric="map50")
    result = evaluator.evaluate(trial_dir, Policy("p", []), context={"source_label_count": 1})
    assert result.score == 0.934
    assert (trial_dir / "data.yaml").exists()
    assert (trial_dir / "yolo_stdout.log").exists()
    assert (trial_dir / "yolo_stderr.log").exists()
    assert result.metrics["yolo_metric_name"] == "map50"


def test_yolo_command_evaluator_exposes_workers_to_command_template() -> None:
    trial_dir = fresh_dir("yolo_eval_workers")
    create_trial_dataset(trial_dir)
    evaluator = YoloCommandEvaluator(fake_yolo_command_with_workers(), metric="map50", workers=3)
    result = evaluator.evaluate(trial_dir, Policy("p", []), context={"source_label_count": 1})
    assert result.score == 0.934
    assert "workers=3" in result.metrics["command"]
    assert result.metrics["workers"] == 3


def test_yolo_command_evaluator_scores_map50_95() -> None:
    trial_dir = fresh_dir("yolo_eval_map50_95")
    create_trial_dataset(trial_dir)
    evaluator = YoloCommandEvaluator(fake_yolo_command(), metric="map50_95")
    result = evaluator.evaluate(trial_dir, Policy("p", []), context={"source_label_count": 1})
    assert result.score == 0.712
    assert result.metrics["yolo_metric_name"] == "map50_95"


def test_yolo_command_evaluator_can_use_hybrid_proxy_score() -> None:
    trial_dir = fresh_dir("yolo_eval_hybrid")
    create_trial_dataset(trial_dir)
    evaluator = YoloCommandEvaluator(fake_yolo_command(), metric="map50", hybrid_proxy_weight=0.2)
    result = evaluator.evaluate(trial_dir, Policy("p", []), context={"source_label_count": 1})
    assert result.metrics["proxy_score"] is not None
    assert result.metrics["final_score"] == result.score
    assert 0.0 <= result.score <= 1.0


def test_yolo_missing_command_has_clear_error() -> None:
    args = argparse.Namespace(
        evaluator="yolo",
        command_template=None,
        score_regex=r"score\s*[:=]\s*(\d+)",
        timeout=None,
        yolo_command=None,
        metric="map50",
        dataset_yaml=None,
        class_names=None,
        hybrid_proxy_weight=0.0,
    )
    try:
        build_evaluator(args)
    except ValueError as exc:
        assert "--yolo-command is required" in str(exc)
    else:
        raise AssertionError("expected missing yolo command error")


def test_auto_generated_data_yaml_uses_label_classes() -> None:
    trial_dir = fresh_dir("data_yaml")
    create_trial_dataset(trial_dir)
    yaml_path = write_yolo_data_yaml(trial_dir / "data.yaml", trial_dir)
    text = yaml_path.read_text(encoding="utf-8")
    assert "path: " in text
    assert "train: images" in text
    assert "val: images" in text
    assert "nc: 3" in text
    assert '2: "class2"' in text


def test_default_output_helpers_are_separated() -> None:
    assert default_run_output_dir("policy_search", run_name="r") == Path("outputs/runs/policy_search/r")
    assert default_run_output_dir("apply_policy", run_name="r") == Path("outputs/runs/apply_policy/r")
    assert smoke_output_dir() == Path("outputs/tests/smoke")
    assert resolve_output_dir(None, "policy_search", run_name="r") == Path("outputs/runs/policy_search/r")


def test_apply_policy_custom_output_has_clear_dataset_dirs() -> None:
    work_dir = fresh_dir("apply_cli")
    dataset_root = work_dir / "dataset"
    trial_dir = work_dir / "trial"
    create_trial_dataset(dataset_root)
    policy_path = work_dir / "policy.json"
    Policy("empty", []).save(policy_path)
    output_dir = work_dir / "custom_output"
    completed = subprocess.run(
        [
            sys.executable,
            "examples/apply_policy_to_dataset.py",
            "--dataset",
            str(dataset_root),
            "--policy",
            str(policy_path),
            "--output",
            str(output_dir),
            "--copies",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "images").exists()
    assert (output_dir / "labels").exists()
    assert (output_dir / "README.txt").exists()
    assert (output_dir / "run_config.json").exists()
    assert (output_dir / "source_dataset.txt").exists()
    assert not trial_dir.exists()
