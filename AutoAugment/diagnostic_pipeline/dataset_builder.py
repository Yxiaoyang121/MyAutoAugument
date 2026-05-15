from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from AutoAugment.policies import Policy, apply_policy_to_sample
from AutoAugment.utils import YoloImageRecord, copy_yolo_records, flatten_relative_stem, load_yolo_sample, save_yolo_sample
from AutoAugment.diagnostic_pipeline.common import write_json, write_markdown


def build_final_augmented_dataset(
    *,
    selected_policy: dict[str, Any],
    train_records: list[YoloImageRecord],
    val_records: list[YoloImageRecord],
    output_dir: str | Path,
    class_names: dict[int, str] | None = None,
    seed: int = 42,
    augment_repeat: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build the final YOLO dataset with originals retained and policy augmentations added."""

    output = Path(output_dir)
    dataset_dir = output / "final_dataset"
    report_path = output / "dataset_build_report.json"
    output.mkdir(parents=True, exist_ok=True)
    if dry_run:
        payload = {
            "stage": "final_augmented_dataset_builder",
            "status": "planned",
            "dry_run": True,
            "dataset_dir": str(dataset_dir.resolve()),
            "data_yaml": str((dataset_dir / "data.yaml").resolve()),
            "selected_policy_id": selected_policy.get("policy_id", selected_policy.get("name")),
            "original_train_count": len(train_records),
            "augmented_train_count": len(train_records) * max(0, augment_repeat),
            "val_count": len(val_records),
            "augment_repeat": int(augment_repeat),
        }
        write_json(report_path, payload)
        return payload

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    train_images = dataset_dir / "images" / "train"
    train_labels = dataset_dir / "labels" / "train"
    val_images = dataset_dir / "images" / "val"
    val_labels = dataset_dir / "labels" / "val"
    copy_yolo_records(train_records, train_images, train_labels)
    copy_yolo_records(val_records, val_images, val_labels)

    policy = Policy.from_dict(selected_policy)
    rng = np.random.default_rng(seed)
    augmented_count = 0
    for record in train_records:
        sample = load_yolo_sample(record)
        for repeat_index in range(max(0, augment_repeat)):
            augmented = apply_policy_to_sample(sample, policy, rng=rng)
            stem = flatten_relative_stem(record.relative_path)
            suffix = record.image_path.suffix.lower()
            image_path = train_images / f"{stem}_diagaug_{repeat_index:03d}{suffix}"
            label_path = train_labels / f"{stem}_diagaug_{repeat_index:03d}.txt"
            save_yolo_sample(augmented, image_path, label_path)
            augmented_count += 1

    data_yaml = write_standard_data_yaml(
        dataset_dir / "data.yaml",
        dataset_dir=dataset_dir,
        class_names=class_names,
        train_labels_dir=train_labels,
        val_labels_dir=val_labels,
    )
    payload = {
        "stage": "final_augmented_dataset_builder",
        "status": "completed",
        "dry_run": False,
        "dataset_dir": str(dataset_dir.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "selected_policy_id": selected_policy.get("policy_id", selected_policy.get("name")),
        "selected_policy": selected_policy,
        "original_train_count": len(train_records),
        "augmented_train_count": augmented_count,
        "total_train_images": len(list(train_images.rglob("*.*"))),
        "val_count": len(val_records),
        "augment_repeat": int(augment_repeat),
        "images_train": str(train_images.resolve()),
        "labels_train": str(train_labels.resolve()),
        "images_val": str(val_images.resolve()),
        "labels_val": str(val_labels.resolve()),
    }
    write_json(report_path, payload)
    write_markdown(
        output / "dataset_build_report.md",
        [
            "# Final Augmented Dataset",
            "",
            f"- Dataset: {payload['dataset_dir']}",
            f"- Data YAML: {payload['data_yaml']}",
            f"- Original train images: {payload['original_train_count']}",
            f"- Augmented train images: {payload['augmented_train_count']}",
            f"- Validation images: {payload['val_count']}",
        ],
    )
    return payload


def write_standard_data_yaml(
    path: str | Path,
    *,
    dataset_dir: str | Path,
    class_names: dict[int, str] | None,
    train_labels_dir: str | Path,
    val_labels_dir: str | Path,
) -> Path:
    """Write a standard relative-path YOLO data.yaml file."""

    yaml_path = Path(path)
    dataset_root = Path(dataset_dir).resolve()
    names = resolve_class_names(class_names, train_labels_dir=train_labels_dir, val_labels_dir=val_labels_dir)
    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    for class_id, name in names.items():
        escaped = str(name).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  {class_id}: "{escaped}"')
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def resolve_class_names(
    class_names: dict[int, str] | None,
    *,
    train_labels_dir: str | Path,
    val_labels_dir: str | Path,
) -> dict[int, str]:
    """Resolve class names, filling missing ids with classN placeholders."""

    max_id = _detect_max_class_id([Path(train_labels_dir), Path(val_labels_dir)])
    if class_names:
        max_id = max(max_id, max(int(key) for key in class_names))
    count = max(1, max_id + 1)
    return {index: str((class_names or {}).get(index, f"class{index}")) for index in range(count)}


def _detect_max_class_id(label_dirs: list[Path]) -> int:
    max_id = -1
    for labels_dir in label_dirs:
        if not labels_dir.exists():
            continue
        for label_path in labels_dir.rglob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    max_id = max(max_id, int(float(parts[0])))
                except ValueError:
                    continue
    return max_id
