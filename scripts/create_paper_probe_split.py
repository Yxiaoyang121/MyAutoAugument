from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    args = parse_args()
    source_yaml = Path(args.source_data).resolve()
    output_root = Path(args.output_root).resolve()
    report = create_split(
        source_yaml=source_yaml,
        output_root=output_root,
        split_seed=int(args.split_seed),
        probe_ratio=float(args.probe_ratio),
        min_train_core_bboxes_per_class=int(args.min_train_core_bboxes_per_class),
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a paper-mode CP-CATF train/probe split.")
    parser.add_argument("--source-data", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--probe-ratio", type=float, default=0.10)
    parser.add_argument("--min-train-core-bboxes-per-class", type=int, default=5)
    return parser.parse_args()


def create_split(
    *,
    source_yaml: Path,
    output_root: Path,
    split_seed: int,
    probe_ratio: float,
    min_train_core_bboxes_per_class: int,
) -> dict[str, Any]:
    data = read_yaml(source_yaml)
    source_root = resolve_root(data, source_yaml)
    train_images_dir = resolve_split_images(data, source_yaml, "train")
    train_labels_dir = image_dir_to_label_dir(train_images_dir, source_root)
    val_images_dir = resolve_split_images(data, source_yaml, "val")
    val_labels_dir = image_dir_to_label_dir(val_images_dir, source_root)
    names = normalize_names(data.get("names", {}))
    nc = int(data.get("nc", len(names)))

    train_images = list_images(train_images_dir)
    val_images = list_images(val_images_dir)
    records = []
    for image in train_images:
        label = image_to_label_path(image, train_images_dir, train_labels_dir)
        class_counts = read_label_class_counts(label)
        records.append(
            {
                "image": image,
                "label": label,
                "rel": image.relative_to(train_images_dir).as_posix(),
                "class_counts": class_counts,
            }
        )

    probe_rels = choose_probe_records(
        records,
        nc=nc,
        probe_ratio=probe_ratio,
        split_seed=split_seed,
        min_train_core_bboxes_per_class=min_train_core_bboxes_per_class,
    )
    probe_rel_set = set(probe_rels)
    train_core_records = [record for record in records if record["rel"] not in probe_rel_set]
    probe_records = [record for record in records if record["rel"] in probe_rel_set]

    reset_split_dirs(output_root)
    materialize_records(train_core_records, output_root / "images" / "train_core", output_root / "labels" / "train_core", train_images_dir, train_labels_dir)
    materialize_records(probe_records, output_root / "images" / "probe", output_root / "labels" / "probe", train_images_dir, train_labels_dir)
    materialize_split(val_images, val_images_dir, val_labels_dir, output_root / "images" / "val", output_root / "labels" / "val")

    data_yaml = output_root / "data.yaml"
    probe_yaml = output_root / "probe.yaml"
    write_yaml(
        data_yaml,
        {
            "path": str(output_root),
            "train": "images/train_core",
            "val": "images/val",
            "nc": nc,
            "names": names,
        },
    )
    write_yaml(
        probe_yaml,
        {
            "path": str(output_root),
            "train": "images/probe",
            "val": "images/probe",
            "probe": "images/probe",
            "nc": nc,
            "names": names,
        },
    )
    write_yaml(
        output_root / "data_with_probe.yaml",
        {
            "path": str(output_root),
            "train": "images/train_core",
            "probe": "images/probe",
            "val": "images/val",
            "nc": nc,
            "names": names,
        },
    )

    report = build_report(
        output_root=output_root,
        source_yaml=source_yaml,
        source_root=source_root,
        split_seed=split_seed,
        probe_ratio=probe_ratio,
        nc=nc,
        names=names,
        train_records=records,
        train_core_records=train_core_records,
        probe_records=probe_records,
        val_images=val_images,
        val_images_dir=val_images_dir,
        val_labels_dir=val_labels_dir,
        data_yaml=data_yaml,
        probe_yaml=probe_yaml,
        min_train_core_bboxes_per_class=min_train_core_bboxes_per_class,
    )
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "probe_split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reports_dir / "probe_split_report.md").write_text(build_markdown_report(report), encoding="utf-8")
    return report


def choose_probe_records(
    records: list[dict[str, Any]],
    *,
    nc: int,
    probe_ratio: float,
    split_seed: int,
    min_train_core_bboxes_per_class: int,
) -> list[str]:
    rng = random.Random(split_seed)
    target_images = max(1, round(len(records) * probe_ratio))
    total_bbox = Counter()
    for record in records:
        total_bbox.update(record["class_counts"])
    target_bbox = {
        class_id: max(1, round(total_bbox.get(class_id, 0) * probe_ratio))
        for class_id in range(nc)
        if total_bbox.get(class_id, 0) > 0
    }
    current_probe = Counter()
    current_train = Counter(total_bbox)
    selected: set[str] = set()
    shuffled = records[:]
    rng.shuffle(shuffled)

    def can_move(record: dict[str, Any]) -> bool:
        for class_id, count in record["class_counts"].items():
            if current_train[class_id] - count < min(min_train_core_bboxes_per_class, total_bbox[class_id]):
                return False
        return True

    while True:
        deficits = {
            class_id: target - current_probe.get(class_id, 0)
            for class_id, target in target_bbox.items()
            if current_probe.get(class_id, 0) < target
        }
        if not deficits:
            break
        best_record = None
        best_score = 0
        for record in shuffled:
            if record["rel"] in selected or not can_move(record):
                continue
            score = sum(min(deficits.get(class_id, 0), count) for class_id, count in record["class_counts"].items())
            if score > best_score:
                best_score = score
                best_record = record
        if best_record is None:
            break
        selected.add(best_record["rel"])
        current_probe.update(best_record["class_counts"])
        for class_id, count in best_record["class_counts"].items():
            current_train[class_id] -= count

    for record in shuffled:
        if len(selected) >= target_images:
            break
        if record["rel"] in selected or not can_move(record):
            continue
        selected.add(record["rel"])
        current_probe.update(record["class_counts"])
        for class_id, count in record["class_counts"].items():
            current_train[class_id] -= count

    return sorted(selected)


def build_report(
    *,
    output_root: Path,
    source_yaml: Path,
    source_root: Path,
    split_seed: int,
    probe_ratio: float,
    nc: int,
    names: dict[int, str],
    train_records: list[dict[str, Any]],
    train_core_records: list[dict[str, Any]],
    probe_records: list[dict[str, Any]],
    val_images: list[Path],
    val_images_dir: Path,
    val_labels_dir: Path,
    data_yaml: Path,
    probe_yaml: Path,
    min_train_core_bboxes_per_class: int,
) -> dict[str, Any]:
    train_core_counts = aggregate_counts(train_core_records)
    probe_counts = aggregate_counts(probe_records)
    val_counts = aggregate_val_counts(val_images, val_images_dir, val_labels_dir)
    train_core_rels = {record["rel"] for record in train_core_records}
    probe_rels = {record["rel"] for record in probe_records}
    val_rels = {image.relative_to(val_images_dir).as_posix() for image in val_images}
    train_probe_overlap = sorted(train_core_rels & probe_rels)
    train_val_overlap = sorted(train_core_rels & val_rels)
    probe_val_overlap = sorted(probe_rels & val_rels)
    missing_probe = [class_id for class_id in range(nc) if aggregate_counts(train_records).get(class_id, 0) > 0 and probe_counts.get(class_id, 0) == 0]
    low_train_core = [
        class_id
        for class_id in range(nc)
        if train_core_counts.get(class_id, 0) < min_train_core_bboxes_per_class
        and aggregate_counts(train_records).get(class_id, 0) >= min_train_core_bboxes_per_class
    ]
    per_class = {}
    total_train_counts = aggregate_counts(train_records)
    for class_id in range(nc):
        per_class[str(class_id)] = {
            "name": names.get(class_id, str(class_id)),
            "original_train_bboxes": int(total_train_counts.get(class_id, 0)),
            "train_core_bboxes": int(train_core_counts.get(class_id, 0)),
            "probe_bboxes": int(probe_counts.get(class_id, 0)),
            "val_bboxes": int(val_counts.get(class_id, 0)),
        }
    no_overlap = not train_probe_overlap and not train_val_overlap and not probe_val_overlap
    return {
        "summary": {
            "source_data_yaml": str(source_yaml),
            "source_root": str(source_root),
            "output_root": str(output_root),
            "data_yaml": str(data_yaml),
            "probe_yaml": str(probe_yaml),
            "split_seed": int(split_seed),
            "probe_ratio": float(probe_ratio),
            "original_train_images": len(train_records),
            "train_core_images": len(train_core_records),
            "probe_images": len(probe_records),
            "val_images": len(val_images),
            "probe_missing_classes": missing_probe,
            "train_core_low_classes": low_train_core,
            "train_core_probe_val_no_overlap": no_overlap,
        },
        "per_class": per_class,
        "overlap_checks": {
            "train_core_probe_overlap_count": len(train_probe_overlap),
            "train_core_val_overlap_count": len(train_val_overlap),
            "probe_val_overlap_count": len(probe_val_overlap),
            "train_core_probe_overlap": train_probe_overlap[:20],
            "train_core_val_overlap": train_val_overlap[:20],
            "probe_val_overlap": probe_val_overlap[:20],
        },
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Paper-Mode CP-CATF Probe Split Report",
        "",
        "## Summary",
        "",
        f"- Source data: `{summary['source_data_yaml']}`",
        f"- Data yaml: `{summary['data_yaml']}`",
        f"- Probe yaml: `{summary['probe_yaml']}`",
        f"- Split seed: `{summary['split_seed']}`",
        f"- Original train images: `{summary['original_train_images']}`",
        f"- Train core images: `{summary['train_core_images']}`",
        f"- Probe images: `{summary['probe_images']}`",
        f"- Final val images: `{summary['val_images']}`",
        f"- Classes missing in probe: `{summary['probe_missing_classes']}`",
        f"- Classes too small in train_core: `{summary['train_core_low_classes']}`",
        f"- Train/probe/val no overlap: `{str(summary['train_core_probe_val_no_overlap']).lower()}`",
        "",
        "## Per-Class BBox Counts",
        "",
        "| class_id | name | train_core | probe | val | original_train |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for class_id, row in sorted(report["per_class"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {class_id} | {row['name']} | {row['train_core_bboxes']} | {row['probe_bboxes']} | "
            f"{row['val_bboxes']} | {row['original_train_bboxes']} |"
        )
    return "\n".join(lines) + "\n"


def materialize_records(
    records: list[dict[str, Any]],
    dst_images_root: Path,
    dst_labels_root: Path,
    src_images_root: Path,
    src_labels_root: Path,
) -> None:
    for record in records:
        rel = Path(record["rel"])
        link_or_copy(record["image"], dst_images_root / rel)
        label = image_to_label_path(record["image"], src_images_root, src_labels_root)
        dst_label = dst_labels_root / rel.with_suffix(".txt")
        if label.exists():
            link_or_copy(label, dst_label)
        else:
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("", encoding="utf-8")


def materialize_split(images: list[Path], src_images_root: Path, src_labels_root: Path, dst_images_root: Path, dst_labels_root: Path) -> None:
    for image in images:
        rel = image.relative_to(src_images_root)
        link_or_copy(image, dst_images_root / rel)
        label = image_to_label_path(image, src_images_root, src_labels_root)
        dst_label = dst_labels_root / rel.with_suffix(".txt")
        if label.exists():
            link_or_copy(label, dst_label)
        else:
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("", encoding="utf-8")


def reset_split_dirs(output_root: Path) -> None:
    for rel in [
        "images/train_core",
        "labels/train_core",
        "images/probe",
        "labels/probe",
        "images/val",
        "labels/val",
    ]:
        target = output_root / rel
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def aggregate_counts(records: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for record in records:
        counts.update(record["class_counts"])
    return counts


def aggregate_val_counts(images: list[Path], images_root: Path, labels_root: Path) -> Counter:
    counts: Counter = Counter()
    for image in images:
        counts.update(read_label_class_counts(image_to_label_path(image, images_root, labels_root)))
    return counts


def read_label_class_counts(label_path: Path) -> Counter:
    counts: Counter = Counter()
    if not label_path.exists():
        return counts
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            counts[int(float(parts[0]))] += 1
        except ValueError:
            continue
    return counts


def list_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def image_to_label_path(image: Path, images_root: Path, labels_root: Path) -> Path:
    return labels_root / image.relative_to(images_root).with_suffix(".txt")


def image_dir_to_label_dir(images_dir: Path, dataset_root: Path) -> Path:
    try:
        rel = images_dir.relative_to(dataset_root)
        parts = list(rel.parts)
        if parts and parts[0] == "images":
            return dataset_root.joinpath("labels", *parts[1:])
    except ValueError:
        pass
    parts = list(images_dir.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def resolve_root(data: dict[str, Any], data_yaml: Path) -> Path:
    root = Path(str(data.get("path", data_yaml.parent)))
    return root if root.is_absolute() else (data_yaml.parent / root).resolve()


def resolve_split_images(data: dict[str, Any], data_yaml: Path, split: str) -> Path:
    root = resolve_root(data, data_yaml)
    value = Path(str(data[split]))
    return value if value.is_absolute() else (root / value).resolve()


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {idx: str(value) for idx, value in enumerate(names)}
    return {}


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
