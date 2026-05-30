from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


TEXTURE_NAME_TOKENS = ("scratch", "edge", "dent", "bruise", "crack", "碰", "划", "伤", "轮廓")
LOW_CONTRAST_NAME_TOKENS = ("low", "dark", "dirty", "oil", "锡", "污", "脏", "油")


def count_train_instances(data_yaml: str | Path) -> dict[int, int]:
    """Count YOLO label instances in the train split for support estimation."""

    yaml_path = Path(data_yaml)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
    root = Path(data.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    train_value = data.get("train")
    if train_value is None:
        return {}
    train_images = Path(str(train_value))
    if not train_images.is_absolute():
        train_images = root / train_images
    labels_dir = _image_dir_to_label_dir(train_images, root)
    counts: Counter[int] = Counter()
    if not labels_dir.exists():
        return {}
    for label_path in labels_dir.rglob("*.txt"):
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                counts[int(float(parts[0]))] += 1
            except ValueError:
                continue
    return {int(class_id): int(count) for class_id, count in sorted(counts.items())}


def build_per_class_diagnosis(
    diagnosis: dict[str, Any],
    *,
    class_names: dict[int, str],
    train_instances: dict[int, int] | None = None,
    epoch: int | None = None,
    reference_per_class: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build CATF-v2 per-class diagnosis rows from the existing diagnosis schema."""

    train_instances = train_instances or {}
    reference_per_class = reference_per_class or {}
    raw_per_class = diagnosis.get("per_class", {}) or {}
    quality = diagnosis.get("quality", {}) or {}
    fn_quality = quality.get("false_negatives", {}) or {}
    global_low_contrast_count = int(fn_quality.get("low_contrast_count", 0) or 0)
    global_dark_count = int(fn_quality.get("dark_count", 0) or 0)
    localization_weak_total = int((diagnosis.get("global", {}) or {}).get("localization_weak", 0) or 0)

    ids = set(class_names) | {int(k) for k in raw_per_class.keys()} | set(train_instances) | set(reference_per_class)
    classes: dict[str, dict[str, Any]] = {}
    for class_id in sorted(ids):
        item = _raw_class_item(raw_per_class, class_id)
        class_name = str(item.get("class_name") or class_names.get(class_id, f"class{class_id}"))
        train_count = int(train_instances.get(class_id, 0) or 0)
        val_count = int(item.get("gt", item.get("gt_count", item.get("val_instances", 0))) or 0)
        tp = int(item.get("tp", item.get("tp_count", 0)) or 0)
        fp = int(item.get("fp", item.get("fp_count", 0)) or 0)
        fn = int(item.get("fn", item.get("fn_count", 0)) or 0)
        precision = _metric(item, "precision", tp / max(1, tp + fp))
        recall = _metric(item, "recall", tp / max(1, tp + fn))
        fn_rate = _metric(item, "fn_rate", fn / max(1, val_count))
        fp_rate = _metric(item, "fp_rate", fp / max(1, tp + fp))
        ref = reference_per_class.get(class_id, {})
        ap50 = _metric(item, "AP50", _metric(ref, "AP50", _metric(ref, "ap50", precision)))
        ap95 = _metric(item, "AP50_95", _metric(ref, "AP50_95", _metric(ref, "ap50_95", min(ap50, recall))))
        weak_loc = _weak_localization_count(localization_weak_total, val_count, diagnosis)
        evidence_count = int(fn + fp + weak_loc)
        low_contrast_fn = _class_low_contrast_count(global_low_contrast_count, fn, class_name)
        dark_fn = _class_low_contrast_count(global_dark_count, fn, class_name)
        support_level = support_bucket(train_count, val_count)
        low_support = val_count < 10 or train_count < 20
        low_recall = recall < 0.60 or fn_rate > 0.35
        high_fp = fp_rate > 0.35 or (precision < 0.55 and fp > 0)
        weak_localization = (ap50 - ap95) > 0.18 or weak_loc > 0
        texture_boundary = weak_localization or _name_has(class_name, TEXTURE_NAME_TOKENS)
        low_contrast = (low_contrast_fn >= max(2, int(0.30 * max(1, fn)))) or _name_has(class_name, LOW_CONTRAST_NAME_TOKENS)
        stable = (
            not low_support
            and precision >= 0.75
            and recall >= 0.75
            and ap95 >= 0.50
            and fp <= max(1, int(0.10 * max(1, tp + fp)))
            and fn <= max(1, int(0.10 * max(1, val_count)))
        )
        classes[str(class_id)] = {
            "class_id": class_id,
            "class_name": class_name,
            "train_instances": train_count,
            "val_instances": val_count,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "AP50": ap50,
            "AP50_95": ap95,
            "FN_rate": fn_rate,
            "FP_rate": fp_rate,
            "support_level": support_level,
            "low_recall": bool(low_recall),
            "high_fp": bool(high_fp),
            "low_precision": bool(precision < 0.55 and val_count >= 10),
            "low_ap50_95": bool(ap95 < 0.45 and val_count >= 10),
            "weak_localization": bool(weak_localization),
            "low_contrast_fn": bool(low_contrast),
            "dark_fn": bool(dark_fn > 0),
            "small_object_fn": bool(item.get("small_object_fn", False)),
            "edge_object_fn": bool(item.get("edge_object_fn", False)),
            "texture_boundary_weak": bool(texture_boundary),
            "class_confusion": bool(item.get("confused_with_classes")),
            "low_support": bool(low_support),
            "low_support_class": bool(val_count < 10),
            "stable_class": bool(stable),
            "evidence_count": evidence_count,
            "low_contrast_fn_count": int(low_contrast_fn),
            "dark_fn_count": int(dark_fn),
            "small_fn_count": int(item.get("small_fn_count", 0) or 0),
            "weak_loc_count": int(weak_loc),
            "fp_count": fp,
            "confused_with_classes": item.get("confused_with_classes", []),
            "mean_fn_area": _nullable_float(item.get("mean_fn_area")),
            "mean_fn_brightness": _nullable_float(item.get("mean_fn_brightness")),
            "mean_fn_contrast": _nullable_float(item.get("mean_fn_contrast")),
            "mean_tp_conf": _nullable_float(item.get("mean_tp_conf")),
            "mean_fp_conf": _nullable_float(item.get("mean_fp_conf")),
            "diagnosis_confidence": confidence(val_count, evidence_count),
            "support_confidence": confidence(val_count, val_count),
            "evidence_confidence": confidence(evidence_count, evidence_count),
            "strong_update_allowed": bool(val_count >= 10 and evidence_count >= 5 and not low_support),
        }
    return {
        "epoch": epoch,
        "classes": classes,
        "summary": {
            "class_count": len(classes),
            "low_support_count": sum(1 for row in classes.values() if row["low_support"]),
            "stable_count": sum(1 for row in classes.values() if row["stable_class"]),
            "strong_update_allowed_count": sum(1 for row in classes.values() if row["strong_update_allowed"]),
        },
    }


def support_bucket(train_instances: int, val_instances: int) -> str:
    support = min(int(train_instances), int(val_instances))
    if support < 10:
        return "low"
    if support < 50:
        return "medium"
    return "high"


def confidence(val_instances: int, evidence_count: int) -> float:
    support = min(1.0, max(0.0, float(val_instances) / 50.0))
    evidence = min(1.0, max(0.0, float(evidence_count) / 20.0))
    return round(0.5 * support + 0.5 * evidence, 4)


def _image_dir_to_label_dir(images_dir: Path, root: Path) -> Path:
    try:
        parts = list(images_dir.relative_to(root).parts)
        if parts and parts[0] == "images":
            return root.joinpath("labels", *parts[1:])
    except ValueError:
        pass
    parts = list(images_dir.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def _raw_class_item(raw_per_class: dict[str, Any], class_id: int) -> dict[str, Any]:
    item = raw_per_class.get(str(class_id), raw_per_class.get(class_id, {}))
    return item if isinstance(item, dict) else {}


def _metric(item: dict[str, Any], key: str, default: float) -> float:
    value = item.get(key, item.get(key.lower(), default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _nullable_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _name_has(name: str, tokens: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(token.lower() in lowered for token in tokens)


def _class_low_contrast_count(global_count: int, fn: int, class_name: str) -> int:
    if fn <= 0 or global_count <= 0:
        return 0
    if _name_has(class_name, LOW_CONTRAST_NAME_TOKENS):
        return min(fn, max(1, int(round(0.5 * fn))))
    return min(fn, max(0, int(round(global_count * fn / max(1, global_count + fn)))))


def _weak_localization_count(total: int, val_count: int, diagnosis: dict[str, Any]) -> int:
    if total <= 0 or val_count <= 0:
        return 0
    overall_gt = int((diagnosis.get("global", {}) or {}).get("gt", 0) or 0)
    return min(val_count, int(round(total * val_count / max(1, overall_gt))))
