from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

from AutoAugment.formats.yolo import save_yolo_labels
from AutoAugment.policies import Policy, default_detection_search_space
from AutoAugment.search import RandomSearch, YoloTrainValEvaluator, find_yolo_best_pt, write_train_val_data_yaml
from AutoAugment.utils import copy_yolo_records, resolve_yolo_train_val_records
from examples.run_policy_search import build_evaluator


def fresh_dir(name: str) -> Path:
    path = Path("outputs/tests/pytest_tmp") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def create_image_label_pair(images_dir: Path, labels_dir: Path, stem: str, class_id: int = 0) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((64, 64, 3), 96, dtype=np.uint8)
    cv2.rectangle(image, (12, 16), (42, 46), (180, 180, 180), -1)
    cv2.imwrite(str(images_dir / f"{stem}.jpg"), image)
    save_yolo_labels(
        labels_dir / f"{stem}.txt",
        np.asarray([class_id], dtype=np.int64),
        np.asarray([[12, 16, 42, 46]], dtype=np.float32),
        64,
        64,
    )


def create_trial_train_dataset(trial_dir: Path) -> tuple[Path, Path]:
    train_images = trial_dir / "dataset" / "images" / "train"
    train_labels = trial_dir / "dataset" / "labels" / "train"
    create_image_label_pair(train_images, train_labels, "train_sample", class_id=1)
    return train_images, train_labels


def create_fixed_val_dataset(root: Path) -> tuple[Path, Path]:
    val_images = root / "val_fixed" / "images" / "val"
    val_labels = root / "val_fixed" / "labels" / "val"
    create_image_label_pair(val_images, val_labels, "val_sample", class_id=1)
    return val_images, val_labels


def write_fake_yolo_scripts(root: Path) -> tuple[Path, Path]:
    train_script = root / "fake_train.py"
    val_script = root / "fake_val.py"
    train_script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import argparse",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--trial', required=True)",
                "args = parser.parse_args()",
                "weights = Path(args.trial) / 'train_runs' / 'train' / 'weights'",
                "weights.mkdir(parents=True, exist_ok=True)",
                "(weights / 'best.pt').write_text('fake weights', encoding='utf-8')",
                "print('train done')",
            ]
        ),
        encoding="utf-8",
    )
    val_script.write_text(
        "\n".join(
            [
                "import argparse",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--best', required=True)",
                "parser.add_argument('--data', required=True)",
                "args = parser.parse_args()",
                "print('mAP50: 0.934')",
                "print('mAP50-95: 0.712')",
            ]
        ),
        encoding="utf-8",
    )
    return train_script, val_script


def evaluator_context(train_images: Path, train_labels: Path, val_images: Path, val_labels: Path) -> dict:
    return {
        "images_dir": str(train_images.resolve()),
        "labels_dir": str(train_labels.resolve()),
        "val_images_dir": str(val_images.resolve()),
        "val_labels_dir": str(val_labels.resolve()),
        "source_label_count": 1,
    }


def test_train_yolo_mode_builds_without_existing_best_pt() -> None:
    args = argparse.Namespace(
        evaluator="train_yolo",
        command_template=None,
        score_regex=r"score\s*[:=]\s*(\d+)",
        timeout=None,
        yolo_command=None,
        metric="map50",
        dataset_yaml=None,
        class_names=None,
        hybrid_proxy_weight=0.0,
        train_command=None,
        val_command=None,
        epochs=5,
        imgsz=640,
        batch=8,
        model="yolov8n.pt",
        workers=0,
    )
    evaluator = build_evaluator(args)
    assert isinstance(evaluator, YoloTrainValEvaluator)
    assert evaluator.workers == 0


def test_train_yolo_default_commands_include_workers() -> None:
    evaluator = YoloTrainValEvaluator(workers=2)
    values = {
        "trial_dir": Path("trial_000"),
        "dataset_dir": Path("trial_000") / "dataset",
        "dataset_yaml": Path("trial_000") / "data.yaml",
        "train_images_dir": Path("trial_000") / "dataset" / "images" / "train",
        "train_labels_dir": Path("trial_000") / "dataset" / "labels" / "train",
        "val_images_dir": Path("val_fixed") / "images" / "val",
        "val_labels_dir": Path("val_fixed") / "labels" / "val",
        "best_pt": Path("trial_000") / "weights" / "best.pt",
        "diagnosis_conf": 0.25,
        "diagnosis_iou": 0.5,
    }
    train_command = evaluator._format_command(evaluator.DEFAULT_TRAIN_COMMAND, **values)
    val_command = evaluator._format_command(evaluator.DEFAULT_VAL_COMMAND, **values)
    predict_command = evaluator._format_command(evaluator.DEFAULT_PREDICT_COMMAND, **values)
    assert "workers=2" in train_command
    assert "workers=2" in val_command
    assert "workers=2" in predict_command


def test_train_val_data_yaml_points_to_augmented_train_and_fixed_val() -> None:
    root = fresh_dir("train_val_yaml")
    trial_dir = root / "trial_000"
    train_images, train_labels = create_trial_train_dataset(trial_dir)
    val_images, val_labels = create_fixed_val_dataset(root)
    yaml_path = write_train_val_data_yaml(
        trial_dir / "data.yaml",
        trial_dir / "dataset",
        train_images_dir=train_images,
        train_labels_dir=train_labels,
        val_images_dir=val_images,
        val_labels_dir=val_labels,
    )
    text = yaml_path.read_text(encoding="utf-8")
    assert "train: images/train" in text
    assert f"val: {val_images.resolve().as_posix()}" in text
    assert "nc: 2" in text


def test_find_best_pt_under_train_runs() -> None:
    trial_dir = fresh_dir("find_best")
    best = trial_dir / "train_runs" / "train" / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_text("weights", encoding="utf-8")
    assert find_yolo_best_pt(trial_dir) == best.resolve()


def test_train_yolo_evaluator_runs_fake_train_and_val_map50() -> None:
    root = fresh_dir("train_yolo_eval")
    trial_dir = root / "trial_000"
    train_images, train_labels = create_trial_train_dataset(trial_dir)
    val_images, val_labels = create_fixed_val_dataset(root)
    train_script, val_script = write_fake_yolo_scripts(root)
    evaluator = YoloTrainValEvaluator(
        train_command_template=f'"{sys.executable}" "{train_script}" --trial {{trial_dir}}',
        val_command_template=f'"{sys.executable}" "{val_script}" --best {{best_pt}} --data {{dataset_yaml}}',
        metric="map50",
    )
    result = evaluator.evaluate(
        trial_dir,
        Policy("p", []),
        context=evaluator_context(train_images, train_labels, val_images, val_labels),
    )
    assert result.score == 0.934
    assert (trial_dir / "weights" / "best.pt").exists()
    assert (trial_dir / "train_stdout.log").exists()
    assert (trial_dir / "val_stdout.log").exists()
    assert result.metrics["best_pt"] == str((trial_dir / "weights" / "best.pt").resolve())


def test_train_yolo_evaluator_scores_map50_95() -> None:
    root = fresh_dir("train_yolo_map50_95")
    trial_dir = root / "trial_000"
    train_images, train_labels = create_trial_train_dataset(trial_dir)
    val_images, val_labels = create_fixed_val_dataset(root)
    train_script, val_script = write_fake_yolo_scripts(root)
    evaluator = YoloTrainValEvaluator(
        train_command_template=f'"{sys.executable}" "{train_script}" --trial {{trial_dir}}',
        val_command_template=f'"{sys.executable}" "{val_script}" --best {{best_pt}} --data {{dataset_yaml}}',
        metric="map50_95",
    )
    result = evaluator.evaluate(
        trial_dir,
        Policy("p", []),
        context=evaluator_context(train_images, train_labels, val_images, val_labels),
    )
    assert result.score == 0.712
    assert result.metrics["yolo_metric_name"] == "map50_95"


def test_train_yolo_hybrid_score() -> None:
    root = fresh_dir("train_yolo_hybrid")
    trial_dir = root / "trial_000"
    train_images, train_labels = create_trial_train_dataset(trial_dir)
    val_images, val_labels = create_fixed_val_dataset(root)
    train_script, val_script = write_fake_yolo_scripts(root)
    evaluator = YoloTrainValEvaluator(
        train_command_template=f'"{sys.executable}" "{train_script}" --trial {{trial_dir}}',
        val_command_template=f'"{sys.executable}" "{val_script}" --best {{best_pt}} --data {{dataset_yaml}}',
        metric="map50",
        hybrid_proxy_weight=0.2,
    )
    result = evaluator.evaluate(
        trial_dir,
        Policy("p", []),
        context=evaluator_context(train_images, train_labels, val_images, val_labels),
    )
    proxy_score = result.metrics["proxy_score"]
    assert proxy_score is not None
    expected = 0.934 * 0.8 + proxy_score * 0.2
    assert abs(result.score - expected) < 1e-9
    assert result.metrics["final_score"] == result.score


def test_random_search_train_yolo_full_fake_loop() -> None:
    root = fresh_dir("train_yolo_search")
    dataset_root = root / "dataset"
    for index in range(4):
        create_image_label_pair(dataset_root / "images", dataset_root / "labels", f"sample_{index}", class_id=0)
    split = resolve_yolo_train_val_records(dataset_root, val_ratio=0.25, seed=3)
    output_dir = root / "search"
    fixed_val_images = output_dir / "val_fixed" / "images" / "val"
    fixed_val_labels = output_dir / "val_fixed" / "labels" / "val"
    copy_yolo_records(split.val_records, fixed_val_images, fixed_val_labels)
    train_script, val_script = write_fake_yolo_scripts(root)
    evaluator = YoloTrainValEvaluator(
        train_command_template=f'"{sys.executable}" "{train_script}" --trial {{trial_dir}}',
        val_command_template=f'"{sys.executable}" "{val_script}" --best {{best_pt}} --data {{dataset_yaml}}',
        metric="map50",
    )
    search = RandomSearch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        num_trials=1,
        num_samples=2,
        seed=5,
        search_space=default_detection_search_space(operation_count_range=(2, 2)),
        evaluator=evaluator,
        records=split.train_records,
        trial_layout="train_yolo",
        context={
            "val_images_dir": str(fixed_val_images.resolve()),
            "val_labels_dir": str(fixed_val_labels.resolve()),
        },
    )
    results = search.run()
    trial_dir = output_dir / "trials" / "trial_000"
    assert len(results) == 1
    assert results[0].score == 0.934
    assert (trial_dir / "dataset" / "images" / "train").exists()
    assert (trial_dir / "dataset" / "labels" / "train").exists()
    assert (trial_dir / "data.yaml").exists()
    assert (trial_dir / "weights" / "best.pt").exists()
    assert (output_dir / "best_policy.json").exists()
