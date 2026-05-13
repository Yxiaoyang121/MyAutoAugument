from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class YoloRecord:
    image_path: Path
    label_path: Path
    relative_image: Path
    class_counts: Counter[int]
    total_boxes: int

    @property
    def class_ids(self) -> set[int]:
        return {class_id for class_id, count in self.class_counts.items() if count > 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the fixed 200-image DataSet2 subset used for fair comparison.")
    parser.add_argument("--source", default="E:/TJGY/DataSet2_fixed")
    parser.add_argument("--target", default="E:/TJGY/DataSet2_fixed_200sample")
    parser.add_argument("--output", default="outputs/dataset2_fixed_200sample_comparison")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nc", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    target = Path(args.target)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(f"source dataset does not exist: {source}")
    if target.exists():
        raise FileExistsError(f"target already exists; inspect it read-only instead of overwriting: {target}")

    train_records = load_records(source / "images" / "train", source / "labels" / "train")
    if len(train_records) < args.samples:
        raise ValueError(f"source train split has only {len(train_records)} images, need {args.samples}")
    selected = select_stratified_records(train_records, samples=args.samples, nc=args.nc, seed=args.seed)

    copy_records(selected, target / "images" / "train", target / "labels" / "train")
    val_records = load_records(source / "images" / "val", source / "labels" / "val")
    copy_records(val_records, target / "images" / "val", target / "labels" / "val")
    write_data_yaml(source / "data.yaml", target / "data.yaml", target=target, nc=args.nc)

    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "selected_train_images_200.txt"
    manifest.write_text(
        "\n".join(record.relative_image.as_posix() for record in sorted(selected, key=lambda item: item.relative_image.as_posix()))
        + "\n",
        encoding="utf-8",
    )
    stats = {
        "source": source.as_posix(),
        "target": target.as_posix(),
        "seed": args.seed,
        "samples": args.samples,
        "selection_method": "greedy_stratified_rare_class_priority",
        "source_train": summarize_records(train_records, nc=args.nc),
        "train_200": summarize_records(selected, nc=args.nc),
        "source_val": summarize_records(val_records, nc=args.nc),
        "rare_class_count_threshold": 5,
    }
    rare_classes = [
        class_id
        for class_id, count in enumerate(stats["source_train"]["per_class_bbox_counts"])
        if 0 < int(count) <= stats["rare_class_count_threshold"]
    ]
    stats["rare_source_classes"] = rare_classes
    stats["rare_classes_retained_in_train_200"] = {
        str(class_id): stats["train_200"]["per_class_bbox_counts"][class_id] > 0 for class_id in rare_classes
    }
    (output / "subset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"target": str(target), "selected_train_images": len(selected), "val_images": len(val_records)}, indent=2))


def load_records(images_dir: Path, labels_dir: Path) -> list[YoloRecord]:
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory does not exist: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels directory does not exist: {labels_dir}")
    records: list[YoloRecord] = []
    for image_path in sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
        relative_image = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_image.with_suffix(".txt")
        if not label_path.exists():
            raise FileNotFoundError(f"missing label for image: {image_path} -> {label_path}")
        class_counts = read_label_counts(label_path)
        records.append(
            YoloRecord(
                image_path=image_path,
                label_path=label_path,
                relative_image=relative_image,
                class_counts=class_counts,
                total_boxes=sum(class_counts.values()),
            )
        )
    return records


def read_label_counts(label_path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    text = label_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise ValueError(f"invalid YOLO label line in {label_path}: {line}")
        counts[int(float(parts[0]))] += 1
    return counts


def select_stratified_records(records: list[YoloRecord], *, samples: int, nc: int, seed: int) -> list[YoloRecord]:
    rng = random.Random(seed)
    global_counts = Counter()
    for record in records:
        global_counts.update(record.class_counts)
    target_counts = {
        class_id: max(1, math.ceil(global_counts[class_id] * samples / max(1, len(records))))
        for class_id in range(nc)
        if global_counts[class_id] > 0
    }
    selected: list[YoloRecord] = []
    selected_ids: set[int] = set()
    selected_counts: Counter[int] = Counter()
    covered: set[int] = set()

    def candidate_score(index: int, record: YoloRecord) -> tuple[float, float]:
        uncovered_bonus = 1000.0 * len(record.class_ids - covered)
        under_target_bonus = 0.0
        rare_bonus = 0.0
        for class_id, count in record.class_counts.items():
            target = target_counts.get(class_id, 1)
            if selected_counts[class_id] < target:
                under_target_bonus += (target - selected_counts[class_id]) / target * count
            rare_bonus += count / max(1.0, float(global_counts[class_id]))
        multi_class_bonus = 0.2 * len(record.class_ids)
        object_bonus = 0.01 * record.total_boxes
        tie_break = rng.random()
        score = uncovered_bonus + 25.0 * under_target_bonus + 12.0 * rare_bonus + multi_class_bonus + object_bonus
        return score, tie_break

    while len(selected) < samples:
        candidates = [(index, record) for index, record in enumerate(records) if index not in selected_ids]
        if not candidates:
            break
        best_index, best_record = max(candidates, key=lambda item: candidate_score(item[0], item[1]))
        selected.append(best_record)
        selected_ids.add(best_index)
        selected_counts.update(best_record.class_counts)
        covered.update(best_record.class_ids)

    if len(selected) != samples:
        raise RuntimeError(f"selected {len(selected)} records, expected {samples}")
    missing_classes = sorted(class_id for class_id in range(nc) if global_counts[class_id] > 0 and selected_counts[class_id] == 0)
    if missing_classes:
        raise RuntimeError(f"selection failed to cover classes: {missing_classes}")
    return selected


def copy_records(records: list[YoloRecord], images_dir: Path, labels_dir: Path) -> None:
    for record in records:
        image_dst = images_dir / record.relative_image
        label_dst = labels_dir / record.relative_image.with_suffix(".txt")
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.image_path, image_dst)
        shutil.copy2(record.label_path, label_dst)


def write_data_yaml(source_yaml: Path, target_yaml: Path, *, target: Path, nc: int) -> None:
    names_block = extract_names_block(source_yaml)
    target_yaml.write_text(
        "\n".join(
            [
                f"path: {target.as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {nc}",
                "names:",
                *names_block,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def extract_names_block(source_yaml: Path) -> list[str]:
    lines = source_yaml.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    in_names = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if stripped.startswith("-") or (line.startswith(" ") and stripped):
                out.append(stripped)
            elif stripped:
                break
    if not out:
        raise ValueError(f"could not extract names block from {source_yaml}")
    return out


def summarize_records(records: list[YoloRecord], *, nc: int) -> dict[str, object]:
    per_class = [0 for _ in range(nc)]
    images_per_class = [0 for _ in range(nc)]
    for record in records:
        for class_id, count in record.class_counts.items():
            if 0 <= class_id < nc:
                per_class[class_id] += int(count)
                images_per_class[class_id] += 1
    return {
        "image_count": len(records),
        "label_count": len(records),
        "bbox_count": sum(per_class),
        "per_class_bbox_counts": per_class,
        "per_class_image_counts": images_per_class,
        "covered_class_count": sum(1 for count in per_class if count > 0),
        "covers_all_classes": all(count > 0 for count in per_class),
    }


if __name__ == "__main__":
    main()
