from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

import cv2
import numpy as np

from AutoAugment.formats.yolo import save_yolo_labels
from tools.visualize_policy_search_results import visualize_policy_search_results
from tools.visualize_yolo_dataset import visualize_dataset


def test_visualize_yolo_dataset_writes_train_and_val_images() -> None:
    work_dir = fresh_dir("vis_dataset")
    dataset = work_dir / "dataset"
    output = work_dir / "vis"
    _write_yolo_pair(dataset, "train", "sample_train")
    _write_yolo_pair(dataset, "val", "sample_val")

    counts = visualize_dataset(dataset, output, max_images=20)

    assert counts == {"train": 1, "val": 1}
    assert len(list((output / "train").glob("*.jpg"))) == 1
    assert len(list((output / "val").glob("*.jpg"))) == 1


def test_visualize_policy_search_results_writes_report_files() -> None:
    work_dir = fresh_dir("vis_search")
    search = work_dir / "search"
    output = work_dir / "report"
    trial = search / "trials" / "trial_000"
    _write_yolo_pair(trial / "dataset", "train", "sample_train")
    trial.mkdir(parents=True, exist_ok=True)
    (trial / "val_stdout.log").write_text(
        "Class     Images  Instances      Box(P          R      mAP50  mAP50-95)\n"
        "all          1          1       0.5        0.5       0.6       0.4\n",
        encoding="utf-8",
    )
    metrics = {
        "yolo_map50": 0.6,
        "proxy_score": 0.7,
        "best_pt": str(trial / "weights" / "best.pt"),
        "dataset_yaml": str(trial / "data.yaml"),
    }
    policy = {
        "name": "policy_trial_000",
        "operations": [{"name": "brightness", "prob": 1.0, "strength": 0.2, "params": {}}],
    }
    search.mkdir(parents=True, exist_ok=True)
    (search / "best_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    with (search / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trial_index", "score", "trial_dir", "policy_name", "metrics_json", "policy_json"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trial_index": 0,
                "score": "0.65",
                "trial_dir": str(trial),
                "policy_name": "policy_trial_000",
                "metrics_json": json.dumps(metrics),
                "policy_json": json.dumps(policy),
            }
        )

    result = visualize_policy_search_results(search, output, max_images=20)

    assert result["best_trial_index"] == 0
    assert (output / "score_curve.png").exists()
    assert (output / "map50_curve.png").exists()
    assert (output / "proxy_score_curve.png").exists()
    assert (output / "top_10_policies.csv").exists()
    assert (output / "top_10_policies.txt").exists()
    assert (output / "best_policy_summary.txt").exists()
    assert (output / "best_trial_val_summary.txt").exists()
    assert len(list((output / "best_trial_samples").glob("*.jpg"))) == 1


def _write_yolo_pair(root: Path, split: str, stem: str) -> None:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((80, 120, 3), 80, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (70, 60), (180, 180, 180), -1)
    assert cv2.imwrite(str(images_dir / f"{stem}.bmp"), image)
    labels = np.asarray([0], dtype=np.int64)
    bboxes = np.asarray([[20, 20, 70, 60]], dtype=np.float32)
    save_yolo_labels(labels_dir / f"{stem}.txt", labels, bboxes, image_width=120, image_height=80)


def fresh_dir(name: str) -> Path:
    path = Path("outputs/tests/pytest_tmp") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path
