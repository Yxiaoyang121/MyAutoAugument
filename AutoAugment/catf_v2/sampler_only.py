from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_WEIGHT = 1.0
TARGET_CLASS_HARD_SAMPLE_WEIGHT = 1.25
LOW_RECALL_CLASS_WEIGHT = 1.30
LOW_SUPPORT_CLASS_WEIGHT = 1.20
MAX_SAMPLE_WEIGHT = 1.80


def build_sampler_only_artifacts(
    *,
    data_yaml: str | Path,
    decision_payload: dict[str, Any],
    epoch_num: int,
    output_dir: str | Path | None = None,
    debug_dir: str | Path | None = None,
    default_weight: float = DEFAULT_WEIGHT,
    target_class_hard_sample_weight: float = TARGET_CLASS_HARD_SAMPLE_WEIGHT,
    low_recall_class_weight: float = LOW_RECALL_CLASS_WEIGHT,
    low_support_class_weight: float = LOW_SUPPORT_CLASS_WEIGHT,
    max_weight: float = MAX_SAMPLE_WEIGHT,
) -> dict[str, Any]:
    """Build auditable sampler-only weight artifacts from probe diagnostics and train_core labels."""

    data_yaml = Path(data_yaml).resolve()
    output_path = Path(output_dir).resolve() if output_dir is not None else None
    debug_path = Path(debug_dir).resolve() if debug_dir is not None else None
    data = read_data_yaml(data_yaml)
    class_names = normalize_class_names(data.get("names", {}))
    train_images_dir, train_labels_dir = resolve_split_image_label_dirs(data_yaml, "train")
    train_records = read_train_core_records(train_images_dir, train_labels_dir)
    focus_rows = sampler_focus_rows(decision_payload)
    context_rows = sampler_context_rows(decision_payload)
    target_classes, protected_classes = sampler_target_classes(
        focus_rows=focus_rows,
        context_rows=context_rows,
        class_names=class_names,
        target_class_hard_sample_weight=target_class_hard_sample_weight,
        low_recall_class_weight=low_recall_class_weight,
        low_support_class_weight=low_support_class_weight,
        max_weight=max_weight,
    )
    image_weight_records = build_image_weight_records(
        train_records=train_records,
        target_classes=target_classes,
        protected_classes=protected_classes,
        default_weight=default_weight,
        max_weight=max_weight,
    )
    weighted_indices = build_weighted_train_indices(image_weight_records)
    before_distribution = class_distribution(train_records)
    after_distribution = class_distribution(train_records, weighted_indices)
    distribution_changed = distributions_changed(before_distribution, after_distribution)
    weighted_image_count = sum(1 for item in image_weight_records if float(item["weight"]) > float(default_weight))
    extra_sample_count = max(0, len(weighted_indices) - len(train_records))
    sampler_effective = bool(weighted_image_count > 0 and extra_sample_count > 0 and distribution_changed)

    weighted_images = [item for item in image_weight_records if float(item["weight"]) > float(default_weight)]
    sample_weight_map = {
        "generated": True,
        "strategy": "weighted_index_list",
        "epoch": int(epoch_num),
        "data_yaml": str(data_yaml),
        "train_core_images_dir": str(train_images_dir),
        "train_core_labels_dir": str(train_labels_dir),
        "paper_mode": bool(decision_payload.get("paper_probe_mode", False)),
        "policy_selection_data": str(decision_payload.get("policy_selection_data") or decision_payload.get("policy_selection_data_yaml") or ""),
        "final_val_used_for_policy_selection": bool(decision_payload.get("final_val_used_for_policy_selection", False)),
        "default_weight": float(default_weight),
        "target_class_hard_sample_weight": float(target_class_hard_sample_weight),
        "low_recall_class_weight": float(low_recall_class_weight),
        "low_support_class_weight": float(low_support_class_weight),
        "max_weight": float(max_weight),
        "train_core_images_count": len(train_records),
        "weighted_train_core_images_count": weighted_image_count,
        "target_classes": target_classes,
        "protected_classes": protected_classes,
        "weighted_images": weighted_images,
        "image_weights": image_weight_records,
        "sampler_only_effective": sampler_effective,
    }
    weighted_indices_payload = {
        "generated": True,
        "strategy": "weighted_index_list",
        "weighted_sampler_enabled": False,
        "weighted_index_list_enabled": True,
        "epoch": int(epoch_num),
        "original_train_core_images_count": len(train_records),
        "weighted_train_indices_count": len(weighted_indices),
        "extra_sample_count": extra_sample_count,
        "weighted_train_core_images_count": weighted_image_count,
        "sampler_only_effective": sampler_effective,
        "weighted_train_indices": weighted_indices,
        "extra_index_records": extra_index_records(train_records, weighted_indices),
    }
    distribution_payload = {
        "generated": True,
        "epoch": int(epoch_num),
        "strategy": "weighted_index_list",
        "original_train_core_images_count": len(train_records),
        "weighted_train_indices_count": len(weighted_indices),
        "sampled_distribution_changed": distribution_changed,
        "before": before_distribution,
        "after": after_distribution,
        "delta_class_fraction": distribution_delta(before_distribution, after_distribution),
    }
    paths = write_sampler_only_artifacts(
        sample_weight_map=sample_weight_map,
        weighted_indices_payload=weighted_indices_payload,
        distribution_payload=distribution_payload,
        output_dir=output_path,
        debug_dir=debug_path,
        epoch_num=epoch_num,
    )
    return {
        "sample_weight_map": sample_weight_map,
        "weighted_train_indices": weighted_indices_payload,
        "sampled_distribution_before_after": distribution_payload,
        "paths": paths,
        "summary": {
            "sample_weight_map_generated": True,
            "weighted_train_core_images_count": weighted_image_count,
            "weighted_sampler_enabled": False,
            "weighted_index_list_enabled": True,
            "sampler_only_effective": sampler_effective,
            "sampled_distribution_changed": distribution_changed,
            "extra_sample_count": extra_sample_count,
        },
    }


def read_data_yaml(data_yaml: Path) -> dict[str, Any]:
    return yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig")) or {}


def normalize_class_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        out: dict[int, str] = {}
        for key, value in names.items():
            try:
                out[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def resolve_split_image_label_dirs(data_yaml: Path, split: str) -> tuple[Path, Path]:
    data = read_data_yaml(data_yaml)
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_value = data.get(split)
    if isinstance(split_value, list):
        if not split_value:
            raise ValueError(f"{data_yaml} has an empty {split} split")
        split_value = split_value[0]
    if not split_value:
        raise ValueError(f"{data_yaml} is missing {split} split")
    images_dir = Path(str(split_value))
    if not images_dir.is_absolute():
        images_dir = (root / images_dir).resolve()
    labels_dir = infer_labels_dir(images_dir)
    return images_dir, labels_dir


def infer_labels_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def read_train_core_records(images_dir: Path, labels_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    image_files = sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    for index, image_path in enumerate(image_files):
        label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
        class_ids, invalid_rows = read_yolo_label_classes(label_path)
        records.append(
            {
                "index": int(index),
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "classes": sorted(set(class_ids)),
                "class_instances": class_ids,
                "label_exists": label_path.exists(),
                "invalid_label_rows": invalid_rows,
            }
        )
    return records


def read_yolo_label_classes(label_path: Path) -> tuple[list[int], int]:
    if not label_path.exists():
        return [], 0
    class_ids: list[int] = []
    invalid_rows = 0
    for raw_line in label_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            class_ids.append(int(float(parts[0])))
        except (IndexError, TypeError, ValueError):
            invalid_rows += 1
    return class_ids, invalid_rows


def sampler_focus_rows(decision_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = decision_payload.get("sampler_only_focus_rows")
    if isinstance(rows, list) and rows:
        return [deepcopy(row) for row in rows if isinstance(row, dict)]
    selected = decision_payload.get("selected_candidate") or {}
    probe_set = selected.get("probe_set") or {}
    diagnosis_record = probe_set.get("diagnosis_record") or probe_set.get("issue_record")
    if isinstance(diagnosis_record, dict) and diagnosis_record:
        return [deepcopy(diagnosis_record)]
    return []


def sampler_context_rows(decision_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = decision_payload.get("sampler_only_context_rows")
    if isinstance(rows, list) and rows:
        return [deepcopy(row) for row in rows if isinstance(row, dict)]
    return sampler_focus_rows(decision_payload)


def sampler_target_classes(
    *,
    focus_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    class_names: dict[int, str],
    target_class_hard_sample_weight: float,
    low_recall_class_weight: float,
    low_support_class_weight: float,
    max_weight: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    target_rows = focus_rows or [
        row for row in context_rows
        if sampler_row_is_candidate(row)
    ]
    targets: dict[str, dict[str, Any]] = {}
    protected: dict[str, dict[str, Any]] = {}
    for row in context_rows:
        class_id = safe_int(row.get("class_id"), default=-1)
        if class_id < 0:
            continue
        if sampler_row_is_protected(row, class_names):
            protected[str(class_id)] = protected_class_payload(row, class_names)
    for row in target_rows:
        class_id = safe_int(row.get("class_id"), default=-1)
        if class_id < 0:
            continue
        if sampler_row_is_protected(row, class_names):
            protected[str(class_id)] = protected_class_payload(row, class_names)
            continue
        if not sampler_row_is_candidate(row):
            continue
        weight, reasons = sampler_row_weight(
            row,
            target_class_hard_sample_weight=target_class_hard_sample_weight,
            low_recall_class_weight=low_recall_class_weight,
            low_support_class_weight=low_support_class_weight,
            max_weight=max_weight,
        )
        targets[str(class_id)] = {
            "class_id": class_id,
            "class_name": str(row.get("class_name") or class_names.get(class_id, class_id)),
            "dominant_issue": str(row.get("dominant_issue", "none")),
            "weight": weight,
            "reasons": reasons,
            "strong_update_allowed": bool(row.get("strong_update_allowed", False)),
            "low_recall": bool(row.get("low_recall", False)),
            "low_support": bool(row.get("low_support", False) or row.get("low_support_only", False) or row.get("low_support_class", False)),
            "oversampling_candidate": bool(row.get("oversampling_candidate", False)),
        }
    return targets, protected


def sampler_row_is_candidate(row: dict[str, Any]) -> bool:
    issue = str(row.get("dominant_issue", ""))
    return bool(
        row.get("strong_update_allowed", False)
        or row.get("oversampling_candidate", False)
        or row.get("low_recall", False)
        or row.get("low_support", False)
        or row.get("low_support_only", False)
        or row.get("low_support_class", False)
        or issue in {"low_recall", "low_support", "low_contrast_fn", "texture_boundary_weak", "weak_localization"}
    )


def sampler_row_is_protected(row: dict[str, Any], class_names: dict[int, str]) -> bool:
    class_id = safe_int(row.get("class_id"), default=-1)
    class_name = str(row.get("class_name") or class_names.get(class_id, "")).strip()
    issue = str(row.get("dominant_issue", "")).strip()
    return bool(
        class_name in {"OK2", "OK3"}
        or class_id in {0, 1}
        or row.get("no_aug_class", False)
        or row.get("high_fp_guarded", False)
        or row.get("high_fp_prior", False)
        or row.get("domain_high_fp_prior", False)
        or row.get("high_fp", False)
        or issue == "high_fp"
    )


def protected_class_payload(row: dict[str, Any], class_names: dict[int, str]) -> dict[str, Any]:
    class_id = safe_int(row.get("class_id"), default=-1)
    class_name = str(row.get("class_name") or class_names.get(class_id, class_id))
    return {
        "class_id": class_id,
        "class_name": class_name,
        "dominant_issue": str(row.get("dominant_issue", "none")),
        "high_fp_protected": bool(
            row.get("high_fp_guarded", False)
            or row.get("high_fp_prior", False)
            or row.get("domain_high_fp_prior", False)
            or row.get("high_fp", False)
            or str(row.get("dominant_issue", "")) == "high_fp"
        ),
        "ok_protected": bool(class_name in {"OK2", "OK3"} or class_id in {0, 1}),
        "no_aug_protected": bool(row.get("no_aug_class", False)),
    }


def sampler_row_weight(
    row: dict[str, Any],
    *,
    target_class_hard_sample_weight: float,
    low_recall_class_weight: float,
    low_support_class_weight: float,
    max_weight: float,
) -> tuple[float, list[str]]:
    issue = str(row.get("dominant_issue", ""))
    low_recall = bool(row.get("low_recall", False) or issue == "low_recall")
    low_support = bool(row.get("low_support", False) or row.get("low_support_only", False) or row.get("low_support_class", False) or issue == "low_support")
    if low_recall:
        return min(float(max_weight), float(low_recall_class_weight)), ["low_recall"]
    if low_support:
        return min(float(max_weight), float(low_support_class_weight)), ["low_support"]
    reasons = [issue] if issue and issue != "none" else ["target_class_hard_sample"]
    return min(float(max_weight), float(target_class_hard_sample_weight)), reasons


def build_image_weight_records(
    *,
    train_records: list[dict[str, Any]],
    target_classes: dict[str, dict[str, Any]],
    protected_classes: dict[str, dict[str, Any]],
    default_weight: float,
    max_weight: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target_ids = {int(key) for key in target_classes}
    protected_ids = {int(key) for key in protected_classes}
    for record in train_records:
        classes = {int(value) for value in record.get("classes", [])}
        present_targets = sorted(classes & target_ids)
        present_protected = sorted(classes & protected_ids)
        weight = float(default_weight)
        reasons: list[str] = []
        related_classes: list[dict[str, Any]] = []
        protected_present = bool(present_protected)
        if present_targets and not protected_present:
            for class_id in present_targets:
                target = target_classes[str(class_id)]
                weight = max(weight, float(target.get("weight", default_weight)))
                reasons.extend([str(item) for item in target.get("reasons", [])])
                related_classes.append(
                    {
                        "class_id": class_id,
                        "class_name": target.get("class_name"),
                        "weight": target.get("weight"),
                        "reasons": target.get("reasons", []),
                    }
                )
            weight = min(float(max_weight), weight)
        elif present_targets and protected_present:
            reasons.append("protected_class_present")
        out.append(
            {
                "index": int(record["index"]),
                "image_path": record["image_path"],
                "label_path": record["label_path"],
                "weight": float(weight),
                "classes": sorted(classes),
                "related_classes": related_classes,
                "weight_reasons": sorted(set(reasons)),
                "protected_present": protected_present,
                "protected_classes_present": present_protected,
                "high_fp_protected": any(protected_classes[str(class_id)].get("high_fp_protected", False) for class_id in present_protected),
                "ok_protected": any(protected_classes[str(class_id)].get("ok_protected", False) for class_id in present_protected),
                "no_aug_protected": any(protected_classes[str(class_id)].get("no_aug_protected", False) for class_id in present_protected),
            }
        )
    return out


def build_weighted_train_indices(image_weight_records: list[dict[str, Any]]) -> list[int]:
    base_indices = [int(item["index"]) for item in image_weight_records]
    weighted_records = [item for item in image_weight_records if float(item.get("weight", DEFAULT_WEIGHT)) > DEFAULT_WEIGHT]
    if not weighted_records:
        return base_indices
    extra_budget = int(round(sum(max(0.0, float(item.get("weight", DEFAULT_WEIGHT)) - DEFAULT_WEIGHT) for item in weighted_records)))
    extra_budget = max(1, extra_budget)
    ordered = sorted(weighted_records, key=lambda item: (-float(item.get("weight", DEFAULT_WEIGHT)), int(item["index"])))
    extras: list[int] = []
    cursor = 0
    while len(extras) < extra_budget:
        extras.append(int(ordered[cursor % len(ordered)]["index"]))
        cursor += 1
    return base_indices + extras


def class_distribution(train_records: list[dict[str, Any]], indices: list[int] | None = None) -> dict[str, Any]:
    selected_indices = indices if indices is not None else [int(item["index"]) for item in train_records]
    by_index = {int(item["index"]): item for item in train_records}
    counts: Counter[int] = Counter()
    image_counts: Counter[int] = Counter()
    for index in selected_indices:
        record = by_index[int(index)]
        for class_id in record.get("class_instances", []):
            counts[int(class_id)] += 1
        for class_id in record.get("classes", []):
            image_counts[int(class_id)] += 1
    total_instances = sum(counts.values())
    return {
        "image_count": len(selected_indices),
        "total_instances": int(total_instances),
        "class_instance_counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "class_image_counts": {str(key): int(value) for key, value in sorted(image_counts.items())},
        "class_instance_fraction": {
            str(key): (float(value) / float(total_instances) if total_instances else 0.0)
            for key, value in sorted(counts.items())
        },
    }


def distributions_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    before_fraction = before.get("class_instance_fraction") or {}
    after_fraction = after.get("class_instance_fraction") or {}
    keys = set(before_fraction) | set(after_fraction)
    return any(abs(float(after_fraction.get(key, 0.0)) - float(before_fraction.get(key, 0.0))) > 1e-12 for key in keys)


def distribution_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    before_fraction = before.get("class_instance_fraction") or {}
    after_fraction = after.get("class_instance_fraction") or {}
    keys = set(before_fraction) | set(after_fraction)
    return {
        str(key): float(after_fraction.get(key, 0.0)) - float(before_fraction.get(key, 0.0))
        for key in sorted(keys, key=lambda value: int(value) if str(value).isdigit() else str(value))
    }


def extra_index_records(train_records: list[dict[str, Any]], weighted_indices: list[int]) -> list[dict[str, Any]]:
    original_count = len(train_records)
    by_index = {int(item["index"]): item for item in train_records}
    records = []
    for position, original_index in enumerate(weighted_indices[original_count:], start=original_count):
        source = by_index[int(original_index)]
        records.append(
            {
                "weighted_position": int(position),
                "original_index": int(original_index),
                "image_path": source["image_path"],
                "classes": source.get("classes", []),
                "is_extra_copy": True,
            }
        )
    return records


def write_sampler_only_artifacts(
    *,
    sample_weight_map: dict[str, Any],
    weighted_indices_payload: dict[str, Any],
    distribution_payload: dict[str, Any],
    output_dir: Path | None,
    debug_dir: Path | None,
    epoch_num: int,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    if debug_dir is not None:
        paths.update(
            write_artifact_triplet(
                base_dir=debug_dir,
                sample_weight_map=sample_weight_map,
                weighted_indices_payload=weighted_indices_payload,
                distribution_payload=distribution_payload,
                suffix="",
            )
        )
    if output_dir is not None:
        for base_dir in (output_dir / "reports", output_dir / "reports" / "catf_v2"):
            paths.update(
                write_artifact_triplet(
                    base_dir=base_dir,
                    sample_weight_map=sample_weight_map,
                    weighted_indices_payload=weighted_indices_payload,
                    distribution_payload=distribution_payload,
                    suffix=f"_epoch_{int(epoch_num)}",
                )
            )
    return paths


def write_artifact_triplet(
    *,
    base_dir: Path,
    sample_weight_map: dict[str, Any],
    weighted_indices_payload: dict[str, Any],
    distribution_payload: dict[str, Any],
    suffix: str,
) -> dict[str, str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    sample_path = base_dir / f"sample_weight_map{suffix}.json"
    legacy_sample_path = base_dir / f"cp_catf_sample_weight_map{suffix}.json"
    indices_path = base_dir / f"weighted_train_indices{suffix}.json"
    distribution_path = base_dir / f"sampled_distribution_before_after{suffix}.json"
    write_json(sample_path, sample_weight_map)
    write_json(legacy_sample_path, sample_weight_map)
    write_json(indices_path, weighted_indices_payload)
    write_json(distribution_path, distribution_payload)
    return {
        f"sample_weight_map{suffix or '_latest'}": str(sample_path),
        f"legacy_sample_weight_map{suffix or '_latest'}": str(legacy_sample_path),
        f"weighted_train_indices{suffix or '_latest'}": str(indices_path),
        f"sampled_distribution_before_after{suffix or '_latest'}": str(distribution_path),
    }


def write_json(path: Path, payload: Any) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
