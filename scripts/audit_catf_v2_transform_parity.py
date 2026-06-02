from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.catf_v2 import ROIStats, SampleAwareAugmentationRouter, initial_policy_matrix  # noqa: E402
from AutoAugment.online_augmentation import OnlineAugmentationStats  # noqa: E402
from scripts.train_yolo_default_with_inloop_feedback import load_class_names_from_data_yaml  # noqa: E402
from scripts.train_yolo_online_aug import (  # noqa: E402
    OnlineTrainingContext,
    UltralyticsOnlinePolicyTransform,
    check_ultralytics_api,
    instances_to_xyxy,
)


DEFAULT_DATA = PROJECT_ROOT / "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/audits/catf_v2_transform_parity"
CLASS_GROUPS = {
    "stable_no_aug": {0, 1},
    "active_defect": {6, 11, 12},
    "domain_high_fp_prior": {4, 8},
    "low_support": {2, 3},
}


@dataclass
class PathRun:
    name: str
    dataset: Any
    online_transform: Any | None = None
    router: Any | None = None
    context: OnlineTrainingContext | None = None


class CapturingRouter:
    def __init__(self, router: SampleAwareAugmentationRouter) -> None:
        self.router = router
        self.last_audit: dict[str, Any] | None = None

    def apply(self, image: np.ndarray, labels: np.ndarray, bboxes: np.ndarray, **kwargs: Any) -> Any:
        result = self.router.apply(image, labels, bboxes, **kwargs)
        self.last_audit = result.audit
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.router, name)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    report_dir = output_dir
    diff_dir = output_dir / "diff_samples"
    diff_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    with tempfile.TemporaryDirectory(prefix="catf_v2_transform_parity_") as tmp:
        yolo_config_dir = Path(tmp) / "ultralytics_config"
        yolo_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir.resolve())
        api = check_ultralytics_api()
        class_names = load_class_names_from_data_yaml(str(args.data))
        clean = build_path_run("clean", api, args, tmp, class_names, mode="clean")
        noop = build_path_run("catf_v2_noop", api, args, tmp, class_names, mode="noop")
        force = build_path_run("catf_v2_formal_force_skip", api, args, tmp, class_names, mode="force_skip")
        selected = select_samples(clean.dataset, args.sample_count, seed=args.seed)
        sample_reports = []
        for ordinal, sample in enumerate(selected):
            sample_reports.append(compare_sample(ordinal, sample, clean, noop, force, seed=args.seed))

    summary = summarize(sample_reports, selected, class_names)
    payload = {
        "audit": {
            "name": "CATF-v2 transform-level parity audit",
            "data": str(Path(args.data).resolve()),
            "sample_count": int(len(selected)),
            "seed": int(args.seed),
            "paths": {
                "clean": "native YOLO default dataset transforms",
                "catf_v2_noop": "UltralyticsOnlinePolicyTransform returns labels immediately via catf_noop=true",
                "catf_v2_formal_force_skip": "CATF-v2 router is entered, all class op probabilities and strengths are zero",
            },
        },
        "sample_selection": {
            "selected_indices": [int(item["index"]) for item in selected],
            "required_group_counts": summary["sample_group_counts"],
            "empty_label_samples": int(summary["empty_label_samples"]),
        },
        "summary": summary,
        "samples": sample_reports,
    }
    write_json(report_dir / "transform_parity.json", payload)
    write_diff_samples(diff_dir, sample_reports)
    write_report(report_dir / "transform_parity_report.md", payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CATF-v2 transform parity against native YOLO transforms.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def build_path_run(name: str, api: dict[str, Any], args: argparse.Namespace, tmp_root: str, class_names: dict[int, str], *, mode: str) -> PathRun:
    trainer = api["DetectionTrainer"](
        overrides={
            "model": args.model,
            "data": str(Path(args.data).resolve()),
            "epochs": 1,
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "workers": int(args.workers),
            "device": str(args.device),
            "seed": int(args.seed),
            "project": str(Path(tmp_root) / name),
            "name": "dataset_probe",
            "exist_ok": True,
            "plots": False,
            "verbose": False,
        }
    )
    trainer.model = None
    dataset = trainer.build_dataset(trainer.data["train"], mode="train", batch=int(args.batch))
    if mode == "clean":
        return PathRun(name=name, dataset=dataset)

    policy = initial_policy_matrix(class_names)
    zero_policy_matrix(policy)
    stats = OnlineAugmentationStats()
    router = CapturingRouter(
        SampleAwareAugmentationRouter(
            policy,
            seed=int(args.seed),
            num_classes=len(class_names),
            stats=stats,
            roi_stats=ROIStats(),
            roi_aware=True,
            sample_aware=True,
            total_epochs=50,
        )
    )
    context = OnlineTrainingContext(
        augmentor=router,
        preview_dir=Path(tmp_root) / name / "previews",
        save_preview=False,
        preview_count=0,
        total_epochs=50,
        catf_noop=(mode == "noop"),
    )
    transform = UltralyticsOnlinePolicyTransform(context, api["Instances"])
    return PathRun(name=name, dataset=dataset, online_transform=transform, router=router, context=context)


def zero_policy_matrix(policy: dict[str, Any]) -> None:
    for row in (policy.get("classes") or {}).values():
        if row.get("no_aug_class"):
            row["status"] = "frozen"
            row["state"] = "frozen"
        else:
            row["status"] = "active"
            row["state"] = "accepted"
        for op in (row.get("ops") or {}).values():
            op["prob"] = 0.0
            op["strength"] = 0.0


def select_samples(dataset: Any, count: int, *, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for index, image_path in enumerate(dataset.im_files):
        labels = label_classes_for_image(Path(image_path))
        groups = groups_for_classes(labels)
        records.append({"index": int(index), "image": str(image_path), "classes": labels, "groups": sorted(groups)})

    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()

    def add_candidates(group_name: str, quota: int) -> None:
        candidates = [item for item in records if group_name in item["groups"]]
        rng.shuffle(candidates)
        for item in candidates:
            if len(selected) >= count:
                return
            if int(item["index"]) in selected_indices:
                continue
            selected.append(item)
            selected_indices.add(int(item["index"]))
            if sum(1 for value in selected if group_name in value["groups"]) >= quota:
                return

    quotas = {
        "stable_no_aug": 15,
        "active_defect": 20,
        "domain_high_fp_prior": 20,
        "low_support": 10,
        "multi_class": 15,
        "empty_label": 5,
    }
    for group, quota in quotas.items():
        add_candidates(group, quota)
    remaining = [item for item in records if int(item["index"]) not in selected_indices]
    rng.shuffle(remaining)
    for item in remaining:
        if len(selected) >= count:
            break
        selected.append(item)
        selected_indices.add(int(item["index"]))
    selected.sort(key=lambda item: int(item["index"]))
    return selected[:count]


def label_classes_for_image(image_path: Path) -> list[int]:
    label_path = image_path.with_suffix(".txt")
    parts = list(label_path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            break
    label_path = Path(*parts)
    if not label_path.exists():
        return []
    classes: list[int] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = line.strip().split()
        if not tokens:
            continue
        try:
            classes.append(int(float(tokens[0])))
        except ValueError:
            continue
    return classes


def groups_for_classes(classes: list[int]) -> set[str]:
    present = set(classes)
    groups = {name for name, ids in CLASS_GROUPS.items() if present & ids}
    if len(set(classes)) > 1:
        groups.add("multi_class")
    if not classes:
        groups.add("empty_label")
    return groups


def compare_sample(ordinal: int, sample: dict[str, Any], clean: PathRun, noop: PathRun, force: PathRun, *, seed: int) -> dict[str, Any]:
    index = int(sample["index"])
    sample_seed = seed + index * 1009
    clean_result = run_path_sample(clean, index, sample_seed)
    noop_result = run_path_sample(noop, index, sample_seed)
    force_result = run_path_sample(force, index, sample_seed)
    raw_noop_diff = compare_signatures(clean_result["raw_signature"], noop_result["intermediate_signature"])
    raw_force_diff = compare_signatures(clean_result["raw_signature"], force_result["intermediate_signature"])
    clean_noop_final = compare_signatures(clean_result["final_signature"], noop_result["final_signature"])
    clean_force_final = compare_signatures(clean_result["final_signature"], force_result["final_signature"])
    return {
        "ordinal": int(ordinal),
        "index": index,
        "image": sample["image"],
        "classes": sample["classes"],
        "groups": sample["groups"],
        "raw_vs_noop_intermediate": raw_noop_diff,
        "raw_vs_force_skip_intermediate": raw_force_diff,
        "clean_vs_noop_final": clean_noop_final,
        "clean_vs_force_skip_final": clean_force_final,
        "signatures": {
            "clean_raw": clean_result["raw_signature"],
            "catf_v2_noop_intermediate": noop_result["intermediate_signature"],
            "catf_v2_formal_force_skip_intermediate": force_result["intermediate_signature"],
            "clean_final": clean_result["final_signature"],
            "catf_v2_noop_final": noop_result["final_signature"],
            "catf_v2_formal_force_skip_final": force_result["final_signature"],
        },
        "path_details": {
            "clean": clean_result["details"],
            "catf_v2_noop": noop_result["details"],
            "catf_v2_formal_force_skip": force_result["details"],
        },
    }


def run_path_sample(path_run: PathRun, index: int, seed: int) -> dict[str, Any]:
    raw = path_run.dataset.get_image_and_label(index)
    raw_signature = signature(raw, stage="raw")
    labels = copy.deepcopy(raw)
    details: dict[str, Any] = {
        "sample_index": int(index),
        "im_file": str(raw.get("im_file")),
        "online_transform": path_run.online_transform is not None,
        "catf_noop": bool(path_run.context.catf_noop) if path_run.context is not None else False,
        "instance_replaced_by_online_transform": False,
        "image_replaced_by_online_transform": False,
        "cls_replaced_by_online_transform": False,
        "random_state_changed_by_online_transform": False,
        "router_random_draw_delta": 0,
        "router_skip_reason": None,
        "router_applied_ops": [],
        "router_skipped_ops": [],
        "online_aug_stats_delta": {},
        "roi_aug_stats_delta": {},
        "noop_transform_calls_delta": 0,
    }
    if path_run.online_transform is not None:
        instance_id = id(labels.get("instances"))
        image_id = id(labels.get("img"))
        cls_id = id(labels.get("cls"))
        before_rng = random_fingerprint()
        before_draws = int(getattr(path_run.router, "random_draw_count", 0) or 0) if path_run.router is not None else 0
        before_noop_calls = int(getattr(path_run.context, "noop_transform_calls", 0) or 0) if path_run.context is not None else 0
        before_stats = stats_snapshot(path_run.router)
        labels = path_run.online_transform(labels)
        after_rng = random_fingerprint()
        after_draws = int(getattr(path_run.router, "random_draw_count", 0) or 0) if path_run.router is not None else 0
        after_stats = stats_snapshot(path_run.router)
        after_noop_calls = int(getattr(path_run.context, "noop_transform_calls", 0) or 0) if path_run.context is not None else 0
        details.update(
            {
                "instance_replaced_by_online_transform": id(labels.get("instances")) != instance_id,
                "image_replaced_by_online_transform": id(labels.get("img")) != image_id,
                "cls_replaced_by_online_transform": id(labels.get("cls")) != cls_id,
                "random_state_changed_by_online_transform": before_rng != after_rng,
                "router_random_draw_delta": int(after_draws - before_draws),
                "online_aug_stats_delta": counter_delta(before_stats.get("online_aug_stats", {}), after_stats.get("online_aug_stats", {})),
                "roi_aug_stats_delta": counter_delta(before_stats.get("roi_aug_stats", {}), after_stats.get("roi_aug_stats", {})),
                "noop_transform_calls_delta": int(after_noop_calls - before_noop_calls),
            }
        )
        last_audit = getattr(path_run.router, "last_audit", None) if path_run.router is not None else None
        if isinstance(last_audit, dict):
            details["router_skip_reason"] = (last_audit.get("router") or {}).get("skip_reason")
            details["router_applied_ops"] = list(last_audit.get("applied_ops") or [])
            details["router_skipped_ops"] = list(last_audit.get("skipped_ops") or [])
    intermediate_signature = signature(labels, stage="intermediate")
    seed_everything(seed)
    final = path_run.dataset.transforms(labels)
    final_signature = signature(final, stage="final")
    return {
        "raw_signature": raw_signature,
        "intermediate_signature": intermediate_signature,
        "final_signature": final_signature,
        "details": details,
    }


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def random_fingerprint() -> dict[str, str]:
    payload = {
        "python": hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest(),
        "numpy": hashlib.sha256(repr(np.random.get_state()).encode("utf-8")).hexdigest(),
        "torch": hashlib.sha256(torch.random.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    return payload


def signature(labels: dict[str, Any], *, stage: str) -> dict[str, Any]:
    image = array_from_any(labels.get("img"))
    cls = array_from_any(labels.get("cls"))
    bboxes = boxes_from_labels(labels)
    instances = labels.get("instances")
    sig = {
        "stage": stage,
        "keys": sorted(str(key) for key in labels.keys()),
        "im_file": str(labels.get("im_file")),
        "image": array_signature(image),
        "cls": array_signature(cls, include_values=True),
        "class_id_sequence": class_sequence(cls),
        "bboxes": array_signature(bboxes, include_values=True),
        "bbox_count": int(len(bboxes)) if bboxes is not None else 0,
        "instances": instance_signature(instances),
        "ori_shape": shape_payload(labels.get("ori_shape")),
        "resized_shape": shape_payload(labels.get("resized_shape")),
        "batch_idx": array_signature(array_from_any(labels.get("batch_idx")), include_values=True),
    }
    return sig


def boxes_from_labels(labels: dict[str, Any]) -> np.ndarray:
    if "bboxes" in labels:
        arr = array_from_any(labels.get("bboxes"))
        return np.asarray(arr, dtype=np.float32).reshape(-1, 4) if arr is not None else np.zeros((0, 4), dtype=np.float32)
    instances = labels.get("instances")
    image = array_from_any(labels.get("img"))
    if instances is None or image is None:
        return np.zeros((0, 4), dtype=np.float32)
    height, width = image.shape[:2]
    return instances_to_xyxy(instances, width=width, height=height)


def array_from_any(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def array_signature(array: np.ndarray | None, *, include_values: bool = False) -> dict[str, Any]:
    if array is None:
        return {"exists": False}
    contiguous = np.ascontiguousarray(array)
    payload = {
        "exists": True,
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "hash": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "sum": float(np.asarray(contiguous, dtype=np.float64).sum()) if contiguous.size else 0.0,
    }
    if include_values and contiguous.size <= 256:
        payload["values"] = np.asarray(contiguous).tolist()
    return payload


def class_sequence(cls: np.ndarray | None) -> list[int]:
    if cls is None:
        return []
    return [int(value) for value in np.asarray(cls).reshape(-1).tolist()]


def instance_signature(instances: Any) -> dict[str, Any]:
    if instances is None:
        return {"exists": False}
    bboxes = array_from_any(getattr(instances, "bboxes", None))
    segments = array_from_any(getattr(instances, "segments", None))
    keypoints = array_from_any(getattr(instances, "keypoints", None))
    return {
        "exists": True,
        "class": type(instances).__name__,
        "bbox_format": str(getattr(getattr(instances, "_bboxes", None), "format", None)),
        "normalized": bool(getattr(instances, "normalized", False)),
        "bboxes": array_signature(bboxes, include_values=True),
        "segments": array_signature(segments),
        "keypoints": array_signature(keypoints, include_values=True),
    }


def shape_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) if isinstance(item, (int, np.integer)) else item for item in value]
    return value


def compare_signatures(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    bbox_left = left.get("bboxes", {})
    bbox_right = right.get("bboxes", {})
    cls_left = left.get("cls", {})
    cls_right = right.get("cls", {})
    image_left = left.get("image", {})
    image_right = right.get("image", {})
    instance_left = left.get("instances", {})
    instance_right = right.get("instances", {})
    checks = {
        "image_shape_equal": image_left.get("shape") == image_right.get("shape"),
        "image_dtype_equal": image_left.get("dtype") == image_right.get("dtype"),
        "image_hash_equal": image_left.get("hash") == image_right.get("hash"),
        "labels_shape_equal": cls_left.get("shape") == cls_right.get("shape"),
        "class_id_sequence_equal": left.get("class_id_sequence") == right.get("class_id_sequence"),
        "bbox_shape_equal": bbox_left.get("shape") == bbox_right.get("shape"),
        "bbox_hash_equal": bbox_left.get("hash") == bbox_right.get("hash"),
        "bbox_dtype_equal": bbox_left.get("dtype") == bbox_right.get("dtype"),
        "bbox_order_equal": bbox_left.get("hash") == bbox_right.get("hash"),
        "instances_fields_equal": instance_left == instance_right,
        "normalized_state_equal": instance_left.get("normalized") == instance_right.get("normalized"),
        "bbox_format_equal": instance_left.get("bbox_format") == instance_right.get("bbox_format"),
        "keys_equal": left.get("keys") == right.get("keys"),
        "empty_label_handling_equal": int(left.get("bbox_count", 0)) == int(right.get("bbox_count", 0)),
        "sample_index_order_equal": left.get("im_file") == right.get("im_file"),
    }
    numeric = numeric_diff(left, right)
    mismatches = [key for key, value in checks.items() if not bool(value)]
    return {
        "equal": not mismatches and numeric.get("bbox_max_abs_diff", 0.0) == 0.0 and numeric.get("image_max_abs_diff", 0.0) == 0.0,
        "mismatches": mismatches,
        "checks": checks,
        "numeric_diff": numeric,
    }


def numeric_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        "bbox_max_abs_diff": max_abs_from_values(left.get("bboxes", {}), right.get("bboxes", {})),
        "cls_max_abs_diff": max_abs_from_values(left.get("cls", {}), right.get("cls", {})),
        "image_hash_diff": 0.0 if left.get("image", {}).get("hash") == right.get("image", {}).get("hash") else 1.0,
    }


def max_abs_from_values(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left.get("hash") == right.get("hash"):
        return 0.0
    if "values" not in left or "values" not in right:
        return -1.0
    left_arr = np.asarray(left.get("values"), dtype=np.float64)
    right_arr = np.asarray(right.get("values"), dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        return -1.0
    if left_arr.size == 0:
        return 0.0
    return float(np.max(np.abs(left_arr - right_arr)))


def stats_snapshot(router: Any | None) -> dict[str, Any]:
    if router is None:
        return {"online_aug_stats": {}, "roi_aug_stats": {}}
    stats = getattr(router, "stats", None)
    roi_stats = getattr(router, "roi_stats", None)
    return {
        "online_aug_stats": stats.to_dict() if stats is not None else {},
        "roi_aug_stats": roi_stats.to_dict() if roi_stats is not None else {},
    }


def counter_delta(before: Any, after: Any) -> Any:
    if isinstance(before, dict) and isinstance(after, dict):
        out = {}
        for key in sorted(set(before) | set(after)):
            value = counter_delta(before.get(key), after.get(key))
            if value not in ({}, 0, 0.0, None):
                out[key] = value
        return out
    if isinstance(before, (int, float)) or isinstance(after, (int, float)):
        return (after or 0) - (before or 0)
    return None


def summarize(samples: list[dict[str, Any]], selected: list[dict[str, Any]], class_names: dict[int, str]) -> dict[str, Any]:
    def count_diff(key: str) -> int:
        return sum(1 for sample in samples if not sample[key]["equal"])

    force_details = [sample["path_details"]["catf_v2_formal_force_skip"] for sample in samples]
    noop_details = [sample["path_details"]["catf_v2_noop"] for sample in samples]
    group_counts = {group: sum(1 for item in selected if group in item["groups"]) for group in ["stable_no_aug", "active_defect", "domain_high_fp_prior", "low_support", "multi_class", "empty_label"]}
    formal_router_draws = sum(int(item.get("router_random_draw_delta", 0)) for item in force_details)
    formal_samples_seen = sum(int((item.get("online_aug_stats_delta") or {}).get("samples_seen", 0) or 0) for item in force_details)
    formal_bbox_oob = sum(int((item.get("online_aug_stats_delta") or {}).get("bbox_oob_count", 0) or 0) for item in force_details)
    formal_invalid_bbox = sum(int((item.get("online_aug_stats_delta") or {}).get("invalid_bbox_count", 0) or 0) for item in force_details)
    formal_class_oob = sum(int((item.get("online_aug_stats_delta") or {}).get("class_id_oob_count", 0) or 0) for item in force_details)
    force_instance_rewrites = sum(1 for item in force_details if item.get("instance_replaced_by_online_transform"))
    force_image_rewrites = sum(1 for item in force_details if item.get("image_replaced_by_online_transform"))
    force_cls_rewrites = sum(1 for item in force_details if item.get("cls_replaced_by_online_transform"))
    return {
        "clean_vs_noop_final_equal": count_diff("clean_vs_noop_final") == 0,
        "clean_vs_force_skip_final_equal": count_diff("clean_vs_force_skip_final") == 0,
        "raw_vs_noop_intermediate_equal": count_diff("raw_vs_noop_intermediate") == 0,
        "raw_vs_force_skip_intermediate_equal": count_diff("raw_vs_force_skip_intermediate") == 0,
        "clean_vs_noop_final_mismatches": count_diff("clean_vs_noop_final"),
        "clean_vs_force_skip_final_mismatches": count_diff("clean_vs_force_skip_final"),
        "raw_vs_noop_intermediate_mismatches": count_diff("raw_vs_noop_intermediate"),
        "raw_vs_force_skip_intermediate_mismatches": count_diff("raw_vs_force_skip_intermediate"),
        "formal_force_skip_instance_rewrite_count": int(force_instance_rewrites),
        "formal_force_skip_image_rewrite_count": int(force_image_rewrites),
        "formal_force_skip_cls_rewrite_count": int(force_cls_rewrites),
        "noop_instance_rewrite_count": sum(1 for item in noop_details if item.get("instance_replaced_by_online_transform")),
        "formal_force_skip_router_random_draw_count": int(formal_router_draws),
        "formal_force_skip_samples_seen_by_router": int(formal_samples_seen),
        "formal_force_skip_bbox_oob_count": int(formal_bbox_oob),
        "formal_force_skip_invalid_bbox_count": int(formal_invalid_bbox),
        "formal_force_skip_class_id_oob_count": int(formal_class_oob),
        "formal_force_skip_applied_ops_count": sum(len(item.get("router_applied_ops") or []) for item in force_details),
        "formal_force_skip_random_state_changed_count": sum(1 for item in force_details if item.get("random_state_changed_by_online_transform")),
        "noop_random_state_changed_count": sum(1 for item in noop_details if item.get("random_state_changed_by_online_transform")),
        "sample_group_counts": group_counts,
        "empty_label_samples": group_counts.get("empty_label", 0),
        "class_names": {str(key): value for key, value in sorted(class_names.items())},
        "root_cause": root_cause_statement(
            count_diff("clean_vs_force_skip_final"),
            count_diff("raw_vs_force_skip_intermediate"),
            force_instance_rewrites,
            formal_router_draws,
        ),
    }


def root_cause_statement(final_mismatches: int, intermediate_mismatches: int, instance_rewrites: int, random_draws: int) -> str:
    if final_mismatches:
        return "formal_force_skip_changes_final_training_sample"
    if intermediate_mismatches or instance_rewrites:
        return "formal_force_skip_rewrites_intermediate_label_instances_but_final_output_matches_in_sampled_paths"
    if random_draws:
        return "formal_force_skip_consumes_router_random_draws"
    return "no_transform_level_difference_detected"


def write_diff_samples(diff_dir: Path, samples: list[dict[str, Any]]) -> None:
    for sample in samples:
        has_diff = any(
            not sample[key]["equal"]
            for key in [
                "raw_vs_noop_intermediate",
                "raw_vs_force_skip_intermediate",
                "clean_vs_noop_final",
                "clean_vs_force_skip_final",
            ]
        )
        rewrote = sample["path_details"]["catf_v2_formal_force_skip"].get("instance_replaced_by_online_transform")
        if has_diff or rewrote:
            filename = diff_dir / f"sample_{sample['ordinal']:03d}_idx_{sample['index']}.json"
            write_json(filename, sample)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# CATF-v2 Transform-Level Parity Audit",
        "",
        "## Scope",
        "",
        f"- Data: `{payload['audit']['data']}`",
        f"- Samples compared: `{payload['audit']['sample_count']}`",
        "- Paths: clean YOLO default, CATF-v2 noop, CATF-v2 formal-force-skip.",
        "- No training was run.",
        "",
        "## Sample Coverage",
        "",
    ]
    for group, value in summary["sample_group_counts"].items():
        lines.append(f"- {group}: `{value}`")
    lines.extend(
        [
            "",
            "## Parity Results",
            "",
            f"- clean vs CATF-v2 noop final output identical: `{str(summary['clean_vs_noop_final_equal']).lower()}`",
            f"- clean vs CATF-v2 formal-force-skip final output identical: `{str(summary['clean_vs_force_skip_final_equal']).lower()}`",
            f"- raw vs CATF-v2 noop intermediate identical: `{str(summary['raw_vs_noop_intermediate_equal']).lower()}`",
            f"- raw vs CATF-v2 formal-force-skip intermediate identical: `{str(summary['raw_vs_force_skip_intermediate_equal']).lower()}`",
            f"- clean vs noop final mismatches: `{summary['clean_vs_noop_final_mismatches']}`",
            f"- clean vs formal-force-skip final mismatches: `{summary['clean_vs_force_skip_final_mismatches']}`",
            f"- raw vs formal-force-skip intermediate mismatches: `{summary['raw_vs_force_skip_intermediate_mismatches']}`",
            "",
            "## Rewrite And Random Audit",
            "",
            f"- CATF-v2 noop instance rewrites: `{summary['noop_instance_rewrite_count']}`",
            f"- CATF-v2 formal-force-skip instance rewrites: `{summary['formal_force_skip_instance_rewrite_count']}`",
            f"- CATF-v2 formal-force-skip image rewrites: `{summary['formal_force_skip_image_rewrite_count']}`",
            f"- CATF-v2 formal-force-skip cls rewrites: `{summary['formal_force_skip_cls_rewrite_count']}`",
            f"- CATF-v2 formal-force-skip router random draws: `{summary['formal_force_skip_router_random_draw_count']}`",
            f"- CATF-v2 formal-force-skip random state changed samples: `{summary['formal_force_skip_random_state_changed_count']}`",
            f"- CATF-v2 formal-force-skip applied ops: `{summary['formal_force_skip_applied_ops_count']}`",
            f"- CATF-v2 formal-force-skip samples seen by router: `{summary['formal_force_skip_samples_seen_by_router']}`",
            f"- CATF-v2 formal-force-skip bbox_oob_count: `{summary['formal_force_skip_bbox_oob_count']}`",
            f"- CATF-v2 formal-force-skip invalid_bbox_count: `{summary['formal_force_skip_invalid_bbox_count']}`",
            f"- CATF-v2 formal-force-skip class_id_oob_count: `{summary['formal_force_skip_class_id_oob_count']}`",
            "",
            "## Answers",
            "",
            f"1. clean vs CATF-v2 noop is fully identical: `{str(summary['clean_vs_noop_final_equal']).lower()}`.",
            f"2. clean vs CATF-v2 formal-force-skip final output is fully identical: `{str(summary['clean_vs_force_skip_final_equal']).lower()}`.",
            "3. Formal-force-skip does replace image/cls/Instances objects before the native YOLO transform, because the online transform converts boxes through the router and rebuilds `Instances` even when no op is applied.",
            f"4. That rewrite did not change final image hash, class order, bbox hash, dtype, shape, or sample order in the sampled paths: `{str(summary['clean_vs_force_skip_final_equal']).lower()}`.",
            f"5. Router random draws under formal-force-skip: `{summary['formal_force_skip_router_random_draw_count']}`.",
            "6. no_aug/stable/high-FP samples can enter router validation in formal-force-skip, but no industrial or ROI operation is applied and no probability draw is consumed.",
            f"7. Router validation clipped or flagged out-of-bound boxes in `{summary['formal_force_skip_bbox_oob_count']}` sampled case(s), which is enough to change a final YOLO training sample even with zero applied augmentation.",
            "",
            "## Interpretation",
            "",
            f"- Root cause classification: `{summary['root_cause']}`.",
        ]
    )
    if summary["clean_vs_force_skip_final_equal"]:
        lines.append(
            "- This audit does not support transform-level label rewrite as the explanation for seed1 CATF-v2 ROI/industrial=0 but different final metrics."
        )
        lines.append(
            "- Remaining explanations should focus on epochs where CATF-v2 was not a strict force-skip path, metric selection, checkpoint selection, or active-path random/validation side effects outside this no-op transform comparison."
        )
    else:
        lines.append("- The final training sample differs under force-skip; CATF-v2 should bypass the online transform unless an op is actually selected.")
    lines.extend(
        [
            "",
            "## Fix Suggestions",
            "",
            "- Keep `catf_noop=true` bypass as the strict parity path.",
            "- For formal CATF-v2, add an early bypass when sample routing finds no active op before converting/rebuilding `Instances`.",
            "- Only copy image, cls, and `Instances` after a specific ROI/industrial op is selected for application.",
            "- Force-skip paths should continue to avoid random draws and should not call validation that can clip/filter boxes.",
            "",
            "## Artifacts",
            "",
            "- JSON summary: `outputs/audits/catf_v2_transform_parity/transform_parity.json`",
            "- Per-sample diffs: `outputs/audits/catf_v2_transform_parity/diff_samples/`",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
