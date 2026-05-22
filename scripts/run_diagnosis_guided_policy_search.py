from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.augmentations import apply_augmentation, list_augmentations  # noqa: E402
from AutoAugment.bbox.iou import bbox_iou  # noqa: E402
from AutoAugment.diagnostic_pipeline.dataset_builder import write_standard_data_yaml  # noqa: E402
from AutoAugment.diagnostics.yolo_error_analysis import load_class_names_from_data_yaml  # noqa: E402
from AutoAugment.utils import (  # noqa: E402
    IMAGE_EXTENSIONS,
    YoloImageRecord,
    find_yolo_records_from_dirs,
    flatten_relative_stem,
    load_yolo_sample,
    sample_records,
    save_yolo_sample,
)


RUN_ID = "20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search"
BASELINE_ID = "20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep"
DIAGAUG_ID = "20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep"
RANDOM_ID = "20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep"
TOP3_SHORT_ID = "20260518_tiled1024_safe_no_ok_position_policy_top3_shorttrain"
TOP3_BASELINE_ID = "20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline"

DATASET_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full_safe_no_ok_position"
DATA_YAML = DATASET_ROOT / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_ID
BASELINE_BEST = (
    PROJECT_ROOT / "outputs" / "experiments" / BASELINE_ID / "train" / "weights" / "best.pt"
)
BASELINE_METRICS = (
    PROJECT_ROOT / "outputs" / "experiments" / BASELINE_ID / "reports" / "baseline_50ep_metrics.json"
)
DIAGAUG_METRICS = (
    PROJECT_ROOT / "outputs" / "experiments" / DIAGAUG_ID / "reports" / "diagaug_50ep_metrics.json"
)
RANDOM_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / RANDOM_ID
    / "reports"
    / "random_external_aug_50ep_metrics.json"
)
DIAG001_SHORT_METRICS = (
    PROJECT_ROOT / "outputs" / "experiments" / TOP3_SHORT_ID / "policies" / "diag_policy_001" / "metrics.json"
)
BASELINE_DIAGNOSIS = (
    PROJECT_ROOT / "outputs" / "experiments" / TOP3_BASELINE_ID / "diagnosis" / "diagnosis.json"
)

YOLO = Path("D:/Anaconda/envs/pytorch/Scripts/yolo.exe")
MODEL = "yolo11n.pt"
EPOCHS = 5
IMGSZ = 1024
DEFAULT_BATCH = 2
WORKERS = 0
DEVICE = "0"
SEED = 42
POLICY_COUNT = 30
SHORT_TRAIN_COUNT = 10
PROXY_SAMPLES = 96
DISABLED_YOLO_AUGS = {
    "mosaic": 0,
    "mixup": 0,
    "copy_paste": 0,
    "hsv_h": 0,
    "hsv_s": 0,
    "hsv_v": 0,
    "degrees": 0,
    "translate": 0,
    "scale": 0,
    "shear": 0,
    "perspective": 0,
    "fliplr": 0,
    "flipud": 0,
}

OP_POOL = [
    "clahe",
    "contrast",
    "gamma",
    "brightness",
    "sharpen",
    "local_contrast",
    "cutout",
    "horizontal_flip",
    "scale",
    "translate",
    "gaussian_noise",
    "copy_paste",
]

OP_PARAMS = {
    "clahe": {"max_clip_limit": 3.0, "tile_grid_size": [8, 8]},
    "contrast": {"max_delta": 0.35},
    "gamma": {"min_gamma": 0.75, "max_gamma": 1.35},
    "brightness": {"max_delta": 0.20},
    "sharpen": {"amount": 0.8, "sigma": 1.0},
    "local_contrast": {"max_clip_limit": 2.5, "tile_grid_size": [8, 8], "blend": 0.65},
    "cutout": {"max_holes": 2, "max_fraction": 0.18},
    "horizontal_flip": {},
    "scale": {"max_delta": 0.16},
    "translate": {"max_translate": 0.06},
    "gaussian_noise": {"max_std": 0.035},
    "copy_paste": {
        "max_paste_count": 3,
        "max_overlap": 0.18,
        "fallback_max_overlap": 0.45,
        "prefer_small": True,
        "class_balanced": True,
        "max_attempts": 60,
    },
}

OP_PROB_RANGE = {
    "clahe": (0.22, 0.58),
    "contrast": (0.25, 0.62),
    "gamma": (0.22, 0.56),
    "brightness": (0.25, 0.70),
    "sharpen": (0.18, 0.58),
    "local_contrast": (0.18, 0.55),
    "cutout": (0.08, 0.32),
    "horizontal_flip": (0.28, 0.62),
    "scale": (0.18, 0.55),
    "translate": (0.18, 0.55),
    "gaussian_noise": (0.08, 0.35),
    "copy_paste": (0.18, 0.62),
}

OP_STRENGTH_RANGE = {
    "clahe": (0.10, 0.45),
    "contrast": (0.10, 0.45),
    "gamma": (0.10, 0.45),
    "brightness": (0.10, 0.45),
    "sharpen": (0.08, 0.38),
    "local_contrast": (0.08, 0.42),
    "cutout": (0.04, 0.18),
    "horizontal_flip": (1.0, 1.0),
    "scale": (0.05, 0.28),
    "translate": (0.05, 0.28),
    "gaussian_noise": (0.03, 0.18),
    "copy_paste": (0.22, 0.60),
}

OP_RISK = {
    "clahe": 0.35,
    "contrast": 0.35,
    "gamma": 0.35,
    "brightness": 0.35,
    "sharpen": 0.35,
    "local_contrast": 0.35,
    "cutout": 0.85,
    "horizontal_flip": 0.15,
    "scale": 0.55,
    "translate": 0.55,
    "gaussian_noise": 0.75,
    "copy_paste": 0.70,
}


@dataclass
class TrialPaths:
    policy_dir: Path
    data_yaml: Path


def main() -> None:
    args = parse_args()
    configure_environment()
    prepare_output_dir(OUTPUT_DIR, force=args.force)

    class_names = load_class_names_from_data_yaml(DATA_YAML)
    train_records = find_yolo_records_from_dirs(
        DATASET_ROOT / "images" / "train",
        DATASET_ROOT / "labels" / "train",
        missing_label="empty",
    )
    val_records = find_yolo_records_from_dirs(
        DATASET_ROOT / "images" / "val",
        DATASET_ROOT / "labels" / "val",
        missing_label="empty",
    )
    baseline_metrics = read_json(BASELINE_METRICS)
    diagaug_metrics = read_json(DIAGAUG_METRICS)
    random_metrics = read_json(RANDOM_METRICS)
    diag001_short = read_json(DIAG001_SHORT_METRICS)
    diagnosis = load_diagnosis()

    source = build_search_source(
        diagnosis=diagnosis,
        baseline_metrics=baseline_metrics,
        class_names=class_names,
        train_records=train_records,
    )
    write_json(OUTPUT_DIR / "diagnosis_source.json", source)

    policies = generate_search_policies(source=source, class_names=class_names, seed=SEED)
    write_json(OUTPUT_DIR / "policies" / "candidate_policies.json", {"policy_count": len(policies), "policies": policies})
    write_candidate_policies_md(OUTPUT_DIR / "policies" / "candidate_policies.md", policies, source)

    print(f"[policy-search] generated {len(policies)} candidate policies")
    proxy = run_proxy_screening(
        policies=policies,
        train_records=train_records,
        class_names=class_names,
        seed=SEED,
        proxy_samples=PROXY_SAMPLES,
    )
    write_json(OUTPUT_DIR / "proxy" / "proxy_ranking.json", proxy)
    write_proxy_safety_report(OUTPUT_DIR / "proxy" / "proxy_safety_report.md", proxy)

    passed = [row for row in proxy["ranking"] if row["hard_filter_pass"]]
    selected = passed[:SHORT_TRAIN_COUNT]
    print(f"[policy-search] proxy pass={len(passed)}; short-training top {len(selected)}")

    trials: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        policy = row["policy"]
        policy_id = str(policy["policy_id"])
        print(f"[policy-search] short-training {index}/{len(selected)} {policy_id}")
        trial = run_short_training_trial(
            policy_row=row,
            train_records=train_records,
            val_records=val_records,
            class_names=class_names,
            resume=args.resume,
        )
        trials.append(trial)

    summary = build_results_payload(
        source=source,
        policies=policies,
        proxy=proxy,
        selected=selected,
        trials=trials,
        baseline_metrics=baseline_metrics,
        diagaug_metrics=diagaug_metrics,
        random_metrics=random_metrics,
        diag001_short=diag001_short,
    )
    write_json(OUTPUT_DIR / "reports" / "policy_search_results.json", summary)
    write_policy_search_report(OUTPUT_DIR / "reports" / "policy_search_report.md", summary)
    write_best_policy_summary(OUTPUT_DIR / "reports" / "best_policy_summary.md", summary)
    update_state_docs(summary)
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnosis-guided augmentation policy search with short training.")
    parser.add_argument("--force", action="store_true", help="Remove existing output directory before running.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed policy trial metrics if present.")
    return parser.parse_args()


def configure_environment() -> None:
    if YOLO.parent.exists():
        os.environ["PATH"] = str(YOLO.parent) + os.pathsep + os.environ.get("PATH", "")
    yolo_config = PROJECT_ROOT / "outputs" / "Ultralytics"
    yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config.resolve())
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"


def prepare_output_dir(path: Path, *, force: bool) -> None:
    resolved = path.resolve()
    experiments_root = (PROJECT_ROOT / "outputs" / "experiments").resolve()
    if resolved.exists() and force:
        if not str(resolved).startswith(str(experiments_root)):
            raise RuntimeError(f"refusing to remove output outside experiments: {resolved}")
        shutil.rmtree(resolved)
    for subdir in ["policies", "proxy", "short_training", "reports"]:
        (resolved / subdir).mkdir(parents=True, exist_ok=True)


def load_diagnosis() -> dict[str, Any]:
    if not BASELINE_DIAGNOSIS.exists():
        raise FileNotFoundError(
            "baseline diagnosis is required for policy search; expected "
            f"{BASELINE_DIAGNOSIS}"
        )
    diagnosis = read_json(BASELINE_DIAGNOSIS)
    diagnosis.setdefault("baseline_best_pt", str(BASELINE_BEST.resolve()))
    diagnosis.setdefault("dataset", rel(DATA_YAML))
    return diagnosis


def build_search_source(
    *,
    diagnosis: dict[str, Any],
    baseline_metrics: dict[str, Any],
    class_names: dict[int, str],
    train_records: list[YoloImageRecord],
) -> dict[str, Any]:
    vector = flatten_diagnosis_vector(diagnosis.get("diagnosis_vector", {}))
    baseline_overall = extract_overall_metrics(baseline_metrics)
    baseline_per_class = extract_per_class_metrics(baseline_metrics)
    train_counts = collect_train_class_counts(train_records)
    val_counts = {int(row["class_id"]): int(row.get("instances", 0) or 0) for row in baseline_per_class}
    low_support_classes = [
        {
            "class_id": class_id,
            "class_name": class_names.get(class_id, str(class_id)),
            "train_instances": int(train_counts.get(class_id, 0)),
            "val_instances": int(val_counts.get(class_id, 0)),
        }
        for class_id in sorted(class_names)
        if int(val_counts.get(class_id, 0)) < 10 or int(train_counts.get(class_id, 0)) < 40
    ]
    low_support_score = float(np.clip(len(low_support_classes) / max(1, len(class_names)), 0.0, 1.0))
    vector["low_support_score"] = max(vector.get("class_imbalance_score", 0.0), low_support_score)
    op_weights, op_reasons = build_operation_sampling_weights(vector)
    return {
        "run_id": RUN_ID,
        "mode": "diagnosis_guided_policy_search",
        "baseline_run_id": BASELINE_ID,
        "baseline_best_pt": str(BASELINE_BEST.resolve()),
        "data_yaml": rel(DATA_YAML),
        "diagnosis_vector": vector,
        "diagnosis_issues": diagnosis.get("issues", []),
        "baseline_metrics": baseline_overall,
        "low_support_classes": low_support_classes,
        "operation_sampling_weights": op_weights,
        "operation_sampling_reasons": op_reasons,
        "policy_generation_rule": (
            "Diagnosis weights adjust operation sampling probabilities; short-training, not proxy, "
            "selects the final policy."
        ),
    }


def build_operation_sampling_weights(vector: dict[str, float]) -> tuple[dict[str, float], dict[str, list[str]]]:
    low = vector.get("low_contrast_score", 0.0)
    imb = max(vector.get("class_imbalance_score", 0.0), vector.get("low_support_score", 0.0))
    loc = vector.get("localization_score", 0.0)
    fp = vector.get("false_positive_score", 0.0)
    small = vector.get("small_object_score", 0.0)
    weights = {name: 1.0 for name in OP_POOL}
    reasons: dict[str, list[str]] = {name: ["base exploration weight"] for name in OP_POOL}

    for name in ["clahe", "contrast", "gamma", "brightness"]:
        weights[name] += 2.0 * low
        reasons[name].append(f"low_contrast_missed_defect boost={2.0 * low:.3f}")
    for name in ["sharpen", "local_contrast"]:
        weights[name] += 2.4 * low + 0.5 * small
        reasons[name].append(f"texture/visibility boost from low_contrast={2.4 * low:.3f}, small={0.5 * small:.3f}")
    for name in ["copy_paste"]:
        weights[name] += 2.7 * imb + 0.9 * small
        reasons[name].append(f"class_imbalance/low_support boost={2.7 * imb:.3f}, small={0.9 * small:.3f}")
    for name in ["scale", "translate"]:
        weights[name] += 2.0 * loc + 0.7 * small
        reasons[name].append(f"localization/small-object boost={2.0 * loc:.3f}, small={0.7 * small:.3f}")
    for name in ["cutout", "gaussian_noise"]:
        weights[name] += 1.2 * fp
        reasons[name].append(f"false_positive robustness boost={1.2 * fp:.3f}")
    weights["horizontal_flip"] += 0.25
    reasons["horizontal_flip"].append("orientation-invariance exploration prior")
    weights["cutout"] += 0.20
    reasons["cutout"].append("mAP-oriented robustness exploration prior")
    weights["sharpen"] += 0.35
    reasons["sharpen"].append("counterfactual/random-control search prior for boundary detail")
    return weights, reasons


def generate_search_policies(
    *,
    source: dict[str, Any],
    class_names: dict[int, str],
    seed: int,
) -> list[dict[str, Any]]:
    registered = set(list_augmentations())
    missing = sorted(set(OP_POOL) - registered)
    if missing:
        raise RuntimeError(f"policy search operation pool contains unregistered ops: {missing}")
    rng = np.random.default_rng(seed)
    weights = {key: float(value) for key, value in source["operation_sampling_weights"].items()}
    probabilities = np.asarray([weights[name] for name in OP_POOL], dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    train_counts = {
        int(item["class_id"]): int(item["train_instances"])
        for item in source.get("low_support_classes", [])
    }
    signatures: set[str] = set()
    policies: list[dict[str, Any]] = []
    attempts = 0
    while len(policies) < POLICY_COUNT and attempts < 600:
        attempts += 1
        op_count = int(rng.integers(3, 7))
        sampled = list(rng.choice(OP_POOL, size=op_count, replace=False, p=probabilities))
        sampled = normalize_operation_set(sampled, rng, probabilities)
        operations = [sample_operation(name, rng, source, train_counts) for name in sampled]
        signature = "|".join(op["name"] for op in operations) + "|" + "|".join(
            f"{op['prob']:.2f}:{op['strength']:.2f}" for op in operations
        )
        if signature in signatures:
            continue
        policy_id = f"search_policy_{len(policies) + 1:03d}"
        policy = {
            "policy_id": policy_id,
            "name": policy_id,
            "type": "diagnosis_guided_sampled_policy",
            "source_diagnosis_weights": source["diagnosis_vector"],
            "operation_sampling_weights": {name: weights[name] for name in sampled},
            "operations": operations,
            "why_sampled": {
                op["name"]: source["operation_sampling_reasons"].get(op["name"], [])
                for op in operations
            },
            "risk_control": risk_control_for_policy(operations),
            "contains_copy_paste": any(op["name"] == "copy_paste" for op in operations),
            "generation": {
                "seed": seed,
                "attempt_index": attempts,
                "rule": "weighted random sampling; diagnosis changes probabilities, not final selection",
            },
        }
        policies.append(policy)
        signatures.add(signature)
    if len(policies) != POLICY_COUNT:
        raise RuntimeError(f"generated only {len(policies)} unique policies after {attempts} attempts")
    return policies


def normalize_operation_set(sampled: list[str], rng: np.random.Generator, probabilities: np.ndarray) -> list[str]:
    selected = list(dict.fromkeys(sampled))
    if "copy_paste" in selected and "cutout" in selected and rng.random() < 0.55:
        selected.remove("cutout")
    if "gaussian_noise" in selected and "cutout" in selected and rng.random() < 0.35:
        selected.remove("gaussian_noise")
    if len(selected) < 3:
        for name in rng.choice(OP_POOL, size=3, replace=False, p=probabilities):
            if name not in selected:
                selected.append(str(name))
            if len(selected) >= 3:
                break
    return selected


def sample_operation(
    name: str,
    rng: np.random.Generator,
    source: dict[str, Any],
    low_support_train_counts: dict[int, int],
) -> dict[str, Any]:
    vector = source["diagnosis_vector"]
    signal = operation_signal(name, vector)
    p_low, p_high = OP_PROB_RANGE[name]
    s_low, s_high = OP_STRENGTH_RANGE[name]
    prob_high = max(p_low, min(0.92, p_high + 0.10 * signal))
    strength_high = max(s_low, min(0.82, s_high + 0.12 * signal))
    prob = float(rng.uniform(p_low, prob_high)) if prob_high > p_low else float(p_low)
    strength = float(rng.uniform(s_low, strength_high)) if strength_high > s_low else float(s_low)
    params = dict(OP_PARAMS[name])
    if name == "copy_paste":
        params["dataset_class_counts"] = low_support_train_counts
        if source.get("low_support_classes"):
            params["target_classes"] = [int(item["class_id"]) for item in source["low_support_classes"]]
    if name == "cutout":
        params["max_fraction"] = float(np.clip(params["max_fraction"] * (0.75 + 0.5 * strength), 0.05, 0.18))
    return {
        "name": name,
        "prob": prob,
        "strength": strength if name != "horizontal_flip" else 1.0,
        "params": params,
    }


def operation_signal(name: str, vector: dict[str, float]) -> float:
    if name in {"clahe", "contrast", "gamma", "brightness", "sharpen", "local_contrast"}:
        return vector.get("low_contrast_score", 0.0)
    if name == "copy_paste":
        return max(vector.get("class_imbalance_score", 0.0), vector.get("low_support_score", 0.0))
    if name in {"scale", "translate"}:
        return max(vector.get("localization_score", 0.0), vector.get("small_object_score", 0.0))
    if name in {"cutout", "gaussian_noise"}:
        return vector.get("false_positive_score", 0.0)
    return 0.1


def risk_control_for_policy(operations: list[dict[str, Any]]) -> list[str]:
    names = {op["name"] for op in operations}
    controls = ["proxy hard filters reject class-id out-of-range, invalid boxes, image-save failures, and illegal copy-paste boxes"]
    if names & {"scale", "translate"}:
        controls.append("geometry ops are limited to mild scale/translate and audited for bbox retention")
    if "cutout" in names:
        controls.append("cutout is audited for bbox subject occlusion and downweighted when it masks object bodies")
    if "copy_paste" in names:
        controls.append("copy_paste uses low overlap, optional class-balanced target classes, and new bbox validity audit")
    if names & {"clahe", "contrast", "gamma", "brightness", "local_contrast", "sharpen"}:
        controls.append("photometric/detail operations are capped to avoid overexposure or excessive sharpening")
    return controls


def run_proxy_screening(
    *,
    policies: list[dict[str, Any]],
    train_records: list[YoloImageRecord],
    class_names: dict[int, str],
    seed: int,
    proxy_samples: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    selected_records = sample_records(train_records, proxy_samples, rng)
    rows: list[dict[str, Any]] = []
    for index, policy in enumerate(policies):
        policy_rng = np.random.default_rng(seed + 1009 + index)
        metrics = evaluate_policy_safety_proxy(
            policy=policy,
            records=selected_records,
            class_count=len(class_names),
            rng=policy_rng,
        )
        hard_pass, hard_reasons = severe_hard_filter(metrics)
        soft_reasons = soft_penalty_reasons(metrics)
        diagnosis_alignment = diagnosis_alignment_score(policy)
        diversity_score = min(1.0, len({op["name"] for op in policy["operations"]}) / 5.0)
        risk_penalty = policy_risk_penalty(policy)
        safety_score = compute_soft_safety_score(metrics)
        proxy_score = float(
            np.clip(
                0.45 * safety_score
                + 0.25 * diagnosis_alignment
                + 0.15 * diversity_score
                + 0.15 * (1.0 - risk_penalty),
                0.0,
                1.0,
            )
        )
        rows.append(
            {
                "policy_id": policy["policy_id"],
                "candidate_index": index,
                "policy": policy,
                "hard_filter_pass": hard_pass,
                "hard_filter_reasons": hard_reasons,
                "safety_soft_penalty_reasons": soft_reasons,
                "bbox_valid_rate": metrics["bbox_valid_rate"],
                "original_bbox_retention": metrics["original_bbox_retention"],
                "new_bbox_valid_rate": metrics["new_bbox_valid_rate"],
                "total_bbox_valid_rate": metrics["total_bbox_valid_rate"],
                "small_object_retention": metrics["small_object_retention"],
                "cutout_bbox_occlusion_rate": metrics["cutout_bbox_occlusion_rate"],
                "cutout_max_bbox_occlusion": metrics["cutout_max_bbox_occlusion"],
                "copy_paste_audit": metrics["copy_paste_audit"],
                "class_distribution_change": metrics["class_distribution_change"],
                "operation_application_counts": metrics["operation_application_counts"],
                "safety_score": safety_score,
                "diagnosis_alignment_score": diagnosis_alignment,
                "risk_penalty": risk_penalty,
                "proxy_score": proxy_score,
                "combined_proxy_safety_score": float(np.clip(0.35 * proxy_score + 0.65 * safety_score, 0.0, 1.0)),
                "proxy_role": "safety filtering and coarse screening only; not the final selection criterion",
            }
        )
    rows.sort(key=lambda item: (item["hard_filter_pass"], item["combined_proxy_safety_score"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "run_id": RUN_ID,
        "stage": "proxy_safety_screening",
        "proxy_role": "Safety filtering and coarse screening only. Final policy selection is from 5 epoch short-training balanced score.",
        "seed": seed,
        "proxy_samples": len(selected_records),
        "candidate_count": len(policies),
        "hard_filter_pass_count": sum(1 for row in rows if row["hard_filter_pass"]),
        "ranking": rows,
    }


def evaluate_policy_safety_proxy(
    *,
    policy: dict[str, Any],
    records: list[YoloImageRecord],
    class_count: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    before_class_counts: Counter[int] = Counter()
    after_class_counts: Counter[int] = Counter()
    op_counts: Counter[str] = Counter()
    before_bbox_count = 0
    after_bbox_count = 0
    valid_bbox_count = 0
    invalid_bbox_count = 0
    class_checked = 0
    class_valid = 0
    original_retained_total = 0
    original_total = 0
    small_before_total = 0
    small_retained_total = 0
    new_bbox_total = 0
    new_bbox_valid = 0
    cutout_ratios: list[float] = []
    cutout_occluded_count = 0
    image_save_failures = 0
    copy_paste_new_count = 0
    copy_paste_invalid_new = 0

    for record in records:
        sample = load_yolo_sample(record)
        before_labels = np.asarray(sample["labels"], dtype=np.int64)
        before_boxes = np.asarray(sample["bboxes"], dtype=np.float32).reshape(-1, 4)
        before_bbox_count += len(before_boxes)
        before_class_counts.update(int(label) for label in before_labels)
        before_area = box_area(before_boxes)
        small_mask = before_area / max(1.0, float(sample["image"].shape[0] * sample["image"].shape[1])) <= 0.005
        small_before_total += int(small_mask.sum())

        try:
            augmented, audit = apply_policy_with_audit(sample, policy, rng)
        except Exception:
            image_save_failures += 1
            continue
        op_counts.update(audit["ops_applied"])
        cutout_ratios.extend(audit["cutout_bbox_occlusion_ratios"])
        cutout_occluded_count += sum(1 for value in audit["cutout_bbox_occlusion_ratios"] if value >= 0.35)

        after_labels = np.asarray(augmented["labels"], dtype=np.int64)
        after_boxes = np.asarray(augmented["bboxes"], dtype=np.float32).reshape(-1, 4)
        after_bbox_count += len(after_boxes)
        after_class_counts.update(int(label) for label in after_labels)
        class_checked += len(after_labels)
        class_valid += int(((after_labels >= 0) & (after_labels < class_count)).sum()) if len(after_labels) else 0
        valid_mask = xyxy_valid_in_bounds(after_boxes, augmented["image"].shape[1], augmented["image"].shape[0])
        valid_bbox_count += int(valid_mask.sum())
        invalid_bbox_count += int((~valid_mask).sum()) if len(valid_mask) else 0

        original_count_after = min(len(before_boxes), len(after_boxes))
        original_total += len(before_boxes)
        original_retained_total += original_count_after
        if small_mask.any():
            small_retained_total += min(int(small_mask.sum()), original_count_after)

        if any(op["name"] == "copy_paste" for op in policy["operations"]):
            new_boxes = after_boxes[len(before_boxes) :] if len(after_boxes) > len(before_boxes) else np.zeros((0, 4), dtype=np.float32)
            new_bbox_total += len(new_boxes)
            copy_paste_new_count += len(new_boxes)
            if len(new_boxes):
                new_valid = xyxy_valid_in_bounds(new_boxes, augmented["image"].shape[1], augmented["image"].shape[0])
                new_bbox_valid += int(new_valid.sum())
                copy_paste_invalid_new += int((~new_valid).sum())

    new_rate = new_bbox_valid / max(1, new_bbox_total) if new_bbox_total else 1.0
    class_valid_rate = class_valid / max(1, class_checked) if class_checked else 1.0
    distribution_change = class_distribution_change(before_class_counts, after_class_counts)
    return {
        "sample_count": len(records),
        "before_bbox_count": before_bbox_count,
        "after_bbox_count": after_bbox_count,
        "bbox_valid_rate": valid_bbox_count / max(1, after_bbox_count) if after_bbox_count else 1.0,
        "total_bbox_valid_rate": valid_bbox_count / max(1, after_bbox_count) if after_bbox_count else 1.0,
        "invalid_bbox_count": invalid_bbox_count,
        "class_id_valid_rate": class_valid_rate,
        "class_out_of_range_count": class_checked - class_valid,
        "original_bbox_retention": original_retained_total / max(1, original_total),
        "small_object_retention": small_retained_total / max(1, small_before_total) if small_before_total else 1.0,
        "new_bbox_valid_rate": new_rate,
        "new_bbox_count": new_bbox_total,
        "image_save_failures": image_save_failures,
        "cutout_bbox_occlusion_rate": cutout_occluded_count / max(1, len(cutout_ratios)) if cutout_ratios else 0.0,
        "cutout_max_bbox_occlusion": max(cutout_ratios) if cutout_ratios else 0.0,
        "cutout_mean_bbox_occlusion": float(np.mean(cutout_ratios)) if cutout_ratios else 0.0,
        "operation_application_counts": dict(op_counts),
        "class_distribution_change": distribution_change,
        "copy_paste_audit": {
            "enabled": any(op["name"] == "copy_paste" for op in policy["operations"]),
            "new_bbox_count": copy_paste_new_count,
            "new_bbox_valid_rate": new_rate,
            "invalid_new_bbox_count": copy_paste_invalid_new,
        },
    }


def severe_hard_filter(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if int(metrics.get("class_out_of_range_count", 0)) > 0:
        reasons.append(f"class id out of range count={metrics['class_out_of_range_count']}")
    if int(metrics.get("invalid_bbox_count", 0)) > 0:
        reasons.append(f"invalid bbox count={metrics['invalid_bbox_count']}")
    if int(metrics.get("image_save_failures", 0)) > 0:
        reasons.append(f"image save/apply failures={metrics['image_save_failures']}")
    cp = metrics.get("copy_paste_audit", {})
    if cp.get("enabled") and int(cp.get("invalid_new_bbox_count", 0)) > 0:
        reasons.append(f"copy_paste invalid new bbox count={cp.get('invalid_new_bbox_count')}")
    if float(metrics.get("cutout_max_bbox_occlusion", 0.0)) >= 0.75:
        reasons.append(f"cutout masks bbox subject too strongly max={metrics['cutout_max_bbox_occlusion']:.3f}")
    return not reasons, reasons


def soft_penalty_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons = []
    thresholds = {
        "original_bbox_retention": 0.97,
        "small_object_retention": 0.92,
        "new_bbox_valid_rate": 0.98,
        "bbox_valid_rate": 0.995,
    }
    for key, threshold in thresholds.items():
        value = float(metrics.get(key, 1.0) or 0.0)
        if value < threshold:
            reasons.append(f"{key} {value:.4f} < soft target {threshold:.4f}")
    if float(metrics.get("cutout_bbox_occlusion_rate", 0.0)) > 0.10:
        reasons.append(
            f"cutout_bbox_occlusion_rate {metrics['cutout_bbox_occlusion_rate']:.4f} > soft target 0.1000"
        )
    if float(metrics.get("class_distribution_change", 0.0)) > 0.20:
        reasons.append(f"class_distribution_change {metrics['class_distribution_change']:.4f} > soft target 0.2000")
    return reasons


def compute_soft_safety_score(metrics: dict[str, Any]) -> float:
    cutout_factor = 1.0 - min(0.6, float(metrics.get("cutout_bbox_occlusion_rate", 0.0)) * 1.5)
    distribution_factor = 1.0 - min(0.5, float(metrics.get("class_distribution_change", 0.0)))
    score = (
        float(metrics.get("bbox_valid_rate", 1.0))
        * float(metrics.get("original_bbox_retention", 1.0))
        * float(metrics.get("new_bbox_valid_rate", 1.0))
        * float(metrics.get("small_object_retention", 1.0))
        * cutout_factor
        * distribution_factor
    )
    return float(np.clip(score, 0.0, 1.0))


def diagnosis_alignment_score(policy: dict[str, Any]) -> float:
    weights = policy.get("operation_sampling_weights", {})
    if not weights:
        return 0.0
    selected = [float(weights.get(op["name"], 1.0)) for op in policy.get("operations", [])]
    max_weight = max(float(v) for v in weights.values()) if weights else 1.0
    return float(np.clip(np.mean(selected) / max(1e-6, max_weight), 0.0, 1.0))


def policy_risk_penalty(policy: dict[str, Any]) -> float:
    values = []
    for op in policy.get("operations", []):
        values.append(float(op.get("prob", 1.0)) * float(op.get("strength", 1.0)) * OP_RISK.get(op["name"], 0.6))
    return float(np.clip(np.mean(values) if values else 0.0, 0.0, 1.0))


def apply_policy_with_audit(
    sample: dict[str, Any],
    policy: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_image = np.asarray(sample["image"]).copy()
    current_labels = np.asarray(sample["labels"], dtype=np.int64).copy()
    current_bboxes = np.asarray(sample["bboxes"], dtype=np.float32).reshape(-1, 4).copy()
    ops_applied: list[str] = []
    op_draws: list[dict[str, Any]] = []
    cutout_ratios: list[float] = []
    for op in policy.get("operations", []):
        draw = float(rng.random())
        apply = draw <= float(op.get("prob", 1.0))
        op_draws.append({"name": op["name"], "draw": draw, "applied": apply})
        if not apply:
            continue
        before_image = current_image.copy()
        before_bboxes = current_bboxes.copy()
        current_image, current_labels, current_bboxes = apply_augmentation(
            op["name"],
            current_image,
            current_labels,
            current_bboxes,
            params=op.get("params", {}) or {},
            strength=float(op.get("strength", 1.0)),
            rng=rng,
        )
        ops_applied.append(op["name"])
        if op["name"] == "cutout":
            cutout_ratios.extend(bbox_pixel_change_ratios(before_image, current_image, before_bboxes))
    out = dict(sample)
    out["image"] = current_image
    out["labels"] = current_labels.astype(np.int64, copy=False)
    out["bboxes"] = current_bboxes.astype(np.float32, copy=False).reshape(-1, 4)
    return out, {
        "ops_applied": ops_applied,
        "op_draws": op_draws,
        "cutout_bbox_occlusion_ratios": cutout_ratios,
    }


def bbox_pixel_change_ratios(before: np.ndarray, after: np.ndarray, boxes: np.ndarray) -> list[float]:
    if len(boxes) == 0:
        return []
    before_f = before.astype(np.float32)
    after_f = after.astype(np.float32)
    diff = np.abs(after_f - before_f)
    if diff.ndim == 3:
        changed = diff.mean(axis=2) > 8.0
    else:
        changed = diff > 8.0
    h, w = changed.shape[:2]
    ratios = []
    for box in boxes:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = changed[y1:y2, x1:x2]
        ratios.append(float(patch.mean()) if patch.size else 0.0)
    return ratios


def run_short_training_trial(
    *,
    policy_row: dict[str, Any],
    train_records: list[YoloImageRecord],
    val_records: list[YoloImageRecord],
    class_names: dict[int, str],
    resume: bool,
) -> dict[str, Any]:
    policy = policy_row["policy"]
    policy_dir = OUTPUT_DIR / "short_training" / "policies" / str(policy["policy_id"])
    metrics_path = policy_dir / "metrics.json"
    if resume and metrics_path.exists():
        return read_json(metrics_path)
    if policy_dir.exists():
        shutil.rmtree(policy_dir)
    policy_dir.mkdir(parents=True, exist_ok=True)
    write_json(policy_dir / "policy.json", policy)
    write_json(policy_dir / "proxy_row.json", slim_proxy_row(policy_row))
    paths, dataset_report = build_trial_dataset(policy, policy_dir, train_records, val_records, class_names)
    write_json(policy_dir / "augmented_dataset_summary.json", dataset_report)
    write_dataset_report_md(policy_dir / "augmented_dataset_summary.md", dataset_report)

    batch = DEFAULT_BATCH
    oom = False
    train_result = run_train(policy_dir, paths.data_yaml, batch=batch)
    if train_result["returncode"] != 0 and has_oom(policy_dir / "train_stdout.log", policy_dir / "train_stderr.log"):
        oom = True
        batch = 1
        remove_yolo_run(policy_dir / "train")
        train_result = run_train(policy_dir, paths.data_yaml, batch=batch)
        if train_result["returncode"] != 0 and has_oom(policy_dir / "train_stdout.log", policy_dir / "train_stderr.log"):
            raise RuntimeError(f"{policy['policy_id']} still OOM at batch=1; stopping")
    if train_result["returncode"] != 0:
        raise RuntimeError(f"short training failed for {policy['policy_id']}; see {policy_dir / 'train_stderr.log'}")
    copy_weights(policy_dir)

    val_result = run_val(policy_dir, paths.data_yaml, batch=batch)
    if val_result["returncode"] != 0 and has_oom(policy_dir / "val_stdout.log", policy_dir / "val_stderr.log"):
        oom = True
        batch = 1
        val_result = run_val(policy_dir, paths.data_yaml, batch=batch)
    if val_result["returncode"] != 0:
        raise RuntimeError(f"short validation failed for {policy['policy_id']}; see {policy_dir / 'val_stderr.log'}")

    val_metrics = parse_yolo_val_log(policy_dir / "val_stdout.log", class_names)
    scores = compute_scores(val_metrics["overall"])
    train_summary = read_training_summary(policy_dir / "train" / "results.csv")
    trial = {
        "policy_id": policy["policy_id"],
        "policy": policy,
        "proxy_rank": policy_row["rank"],
        "proxy_score": policy_row["proxy_score"],
        "safety_score": policy_row["safety_score"],
        "combined_proxy_safety_score": policy_row["combined_proxy_safety_score"],
        "hard_filter_pass": policy_row["hard_filter_pass"],
        "hard_filter_reasons": policy_row["hard_filter_reasons"],
        "safety_soft_penalty_reasons": policy_row["safety_soft_penalty_reasons"],
        "augmented_dataset": dataset_report,
        "train_settings": {
            "model": MODEL,
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": batch,
            "workers": WORKERS,
            "device": DEVICE,
            "seed": SEED,
            "disabled_yolo_augmentations": DISABLED_YOLO_AUGS,
        },
        "commands": {
            "train_command": train_result["command"],
            "val_command": val_result["command"],
        },
        "artifacts": {
            "policy_dir": str(policy_dir.resolve()),
            "data_yaml": str(paths.data_yaml.resolve()),
            "best_pt": str((policy_dir / "weights" / "best.pt").resolve()),
            "last_pt": str((policy_dir / "weights" / "last.pt").resolve()),
            "train_stdout": str((policy_dir / "train_stdout.log").resolve()),
            "train_stderr": str((policy_dir / "train_stderr.log").resolve()),
            "val_stdout": str((policy_dir / "val_stdout.log").resolve()),
            "val_stderr": str((policy_dir / "val_stderr.log").resolve()),
        },
        "metrics": {
            **val_metrics["overall"],
            "per_class": val_metrics["per_class"],
            **scores,
        },
        "last_epoch_metrics_from_training_loop": train_summary.get("last_epoch", {}),
        "best_epoch_by_map50_95_from_training_loop": train_summary.get("best_epoch_by_map50_95", {}),
        "oom": oom,
        "training_wall_seconds": train_result["wall_seconds"],
        "validation_wall_seconds": val_result["wall_seconds"],
    }
    write_json(metrics_path, trial)
    write_trial_report(policy_dir / "report.md", trial)
    return trial


def build_trial_dataset(
    policy: dict[str, Any],
    policy_dir: Path,
    train_records: list[YoloImageRecord],
    val_records: list[YoloImageRecord],
    class_names: dict[int, str],
) -> tuple[TrialPaths, dict[str, Any]]:
    dataset_dir = policy_dir / "dataset" / "final_dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    images_train = dataset_dir / "images" / "train"
    labels_train = dataset_dir / "labels" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_val = dataset_dir / "labels" / "val"
    for path in [images_train, labels_train, images_val, labels_val]:
        path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    op_counts: Counter[str] = Counter()
    raw_aug_bboxes = 0
    valid_aug_bboxes = 0
    invalid_aug_bboxes = 0
    image_failures = 0
    label_failures = 0
    cutout_ratios: list[float] = []

    for record in train_records:
        sample = load_yolo_sample(record)
        stem = flatten_relative_stem(record.relative_path)
        ext = record.image_path.suffix.lower() or ".jpg"
        try:
            save_yolo_sample(sample, images_train / f"orig_{stem}{ext}", labels_train / f"orig_{stem}.txt")
        except Exception:
            image_failures += 1
            continue
        try:
            augmented, audit = apply_policy_with_audit(sample, policy, rng)
            filtered, safety = filter_valid_sample(augmented, class_count=len(class_names))
        except Exception:
            image_failures += 1
            continue
        op_counts.update(audit["ops_applied"])
        cutout_ratios.extend(audit["cutout_bbox_occlusion_ratios"])
        raw_aug_bboxes += safety["raw_bbox_count"]
        valid_aug_bboxes += safety["valid_bbox_count"]
        invalid_aug_bboxes += safety["invalid_bbox_count"]
        label_failures += safety["class_out_of_range_count"]
        try:
            save_yolo_sample(filtered, images_train / f"aug_{stem}{ext}", labels_train / f"aug_{stem}.txt")
        except Exception:
            image_failures += 1

    for record in val_records:
        sample = load_yolo_sample(record)
        try:
            save_yolo_sample(sample, images_val / record.relative_path, labels_val / record.relative_path.with_suffix(".txt"))
        except Exception:
            image_failures += 1

    data_yaml = write_standard_data_yaml(
        dataset_dir / "data.yaml",
        dataset_dir=dataset_dir,
        class_names=class_names,
        train_labels_dir=labels_train,
        val_labels_dir=labels_val,
    )
    report = {
        "stage": "policy_search_trial_dataset_builder",
        "status": "completed",
        "policy_id": policy["policy_id"],
        "dataset_dir": str(dataset_dir.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "original_train_count": len(train_records),
        "augmented_train_count": len(train_records),
        "train_images": count_images(images_train),
        "train_bboxes": count_label_rows(labels_train),
        "val_images": count_images(images_val),
        "val_bboxes": count_label_rows(labels_val),
        "augmented_train_raw_bboxes": raw_aug_bboxes,
        "augmented_train_valid_bboxes": valid_aug_bboxes,
        "invalid_augmented_bbox_count": invalid_aug_bboxes,
        "bbox_valid_rate": valid_aug_bboxes / max(1, raw_aug_bboxes),
        "image_failures": image_failures,
        "label_failures": label_failures,
        "operation_application_counts": dict(op_counts),
        "cutout_bbox_occlusion_rate": sum(1 for value in cutout_ratios if value >= 0.35) / max(1, len(cutout_ratios))
        if cutout_ratios
        else 0.0,
        "cutout_max_bbox_occlusion": max(cutout_ratios) if cutout_ratios else 0.0,
        "images_train": str(images_train.resolve()),
        "labels_train": str(labels_train.resolve()),
        "images_val": str(images_val.resolve()),
        "labels_val": str(labels_val.resolve()),
    }
    if image_failures or label_failures:
        raise RuntimeError(f"dataset build failed safety checks for {policy['policy_id']}: {image_failures=} {label_failures=}")
    return TrialPaths(policy_dir=policy_dir, data_yaml=data_yaml), report


def run_train(policy_dir: Path, data_yaml: Path, *, batch: int) -> dict[str, Any]:
    command = [
        str(YOLO),
        "detect",
        "train",
        f"model={MODEL}",
        f"data={data_yaml}",
        f"epochs={EPOCHS}",
        f"imgsz={IMGSZ}",
        f"batch={batch}",
        f"workers={WORKERS}",
        f"device={DEVICE}",
        f"seed={SEED}",
        f"project={policy_dir}",
        "name=train",
        "exist_ok=True",
        *[f"{key}={value}" for key, value in DISABLED_YOLO_AUGS.items()],
    ]
    return run_command(command, policy_dir, "train")


def run_val(policy_dir: Path, data_yaml: Path, *, batch: int) -> dict[str, Any]:
    command = [
        str(YOLO),
        "detect",
        "val",
        f"model={policy_dir / 'weights' / 'best.pt'}",
        f"data={data_yaml}",
        f"imgsz={IMGSZ}",
        f"batch={batch}",
        f"workers={WORKERS}",
        f"device={DEVICE}",
        f"project={policy_dir}",
        "name=val",
        "exist_ok=True",
    ]
    return run_command(command, policy_dir, "val")


def run_command(command: list[str], output_dir: Path, prefix: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command_text = subprocess.list2cmdline(command)
    (output_dir / f"{prefix}_command.txt").write_text(command_text + "\n", encoding="utf-8")
    start = time.time()
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
    )
    wall = time.time() - start
    (output_dir / f"{prefix}_stdout.log").write_text(completed.stdout, encoding="utf-8", errors="replace")
    (output_dir / f"{prefix}_stderr.log").write_text(completed.stderr, encoding="utf-8", errors="replace")
    return {"command": command_text, "returncode": completed.returncode, "wall_seconds": wall}


def copy_weights(policy_dir: Path) -> None:
    src = policy_dir / "train" / "weights"
    dst = policy_dir / "weights"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["best.pt", "last.pt"]:
        source = src / name
        if not source.exists():
            raise FileNotFoundError(f"missing trained weight: {source}")
        shutil.copy2(source, dst / name)


def remove_yolo_run(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def parse_yolo_val_log(path: Path, class_names: dict[int, str]) -> dict[str, Any]:
    text = strip_ansi(path.read_text(encoding="utf-8", errors="replace"))
    rows: list[dict[str, Any]] = []
    overall: dict[str, Any] | None = None
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        values = numeric_tail(parts, 6)
        if values is None:
            continue
        images, instances, precision, recall, ap50, ap50_95 = values
        if parts[0] == "all":
            overall = {
                "images": int(images),
                "instances": int(instances),
                "precision": precision,
                "recall": recall,
                "map50": ap50,
                "map50_95": ap50_95,
            }
            continue
        if overall is None or len(rows) >= len(class_names):
            continue
        class_id = len(rows)
        rows.append(
            {
                "class_id": class_id,
                "name": class_names.get(class_id, str(class_id)),
                "log_name": parts[0],
                "images": int(images),
                "instances": int(instances),
                "precision": precision,
                "recall": recall,
                "ap50": ap50,
                "ap50_95": ap50_95,
            }
        )
    if overall is None:
        raise RuntimeError(f"could not parse YOLO val metrics from {path}")
    if len(rows) != len(class_names):
        raise RuntimeError(f"expected {len(class_names)} class rows from {path}, parsed {len(rows)}")
    return {"overall": overall, "per_class": rows}


def numeric_tail(parts: list[str], count: int) -> list[float] | None:
    values: list[float] = []
    for item in reversed(parts):
        try:
            values.append(float(item))
        except ValueError:
            continue
        if len(values) == count:
            return list(reversed(values))
    return None


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def compute_scores(overall: dict[str, Any]) -> dict[str, float | str]:
    precision = float(overall["precision"])
    recall = float(overall["recall"])
    map50 = float(overall["map50"])
    map50_95 = float(overall["map50_95"])
    balanced = 0.30 * map50_95 + 0.25 * map50 + 0.25 * recall + 0.20 * precision
    recall_priority = 0.40 * recall + 0.25 * map50 + 0.20 * map50_95 + 0.15 * precision
    map_priority = 0.40 * map50_95 + 0.30 * map50 + 0.20 * precision + 0.10 * recall
    return {
        "balanced_score": float(balanced),
        "balanced_score_formula": "0.30*mAP50-95 + 0.25*mAP50 + 0.25*Recall + 0.20*Precision",
        "recall_priority_score": float(recall_priority),
        "recall_priority_score_formula": "0.40*Recall + 0.25*mAP50 + 0.20*mAP50-95 + 0.15*Precision",
        "map_priority_score": float(map_priority),
        "map_priority_score_formula": "0.40*mAP50-95 + 0.30*mAP50 + 0.20*Precision + 0.10*Recall",
    }


def read_training_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    def metrics_from_row(row: dict[str, str]) -> dict[str, Any]:
        return {
            "epoch": int(float(row["epoch"])),
            "time_seconds": float(row["time"]),
            "precision": float(row["metrics/precision(B)"]),
            "recall": float(row["metrics/recall(B)"]),
            "map50": float(row["metrics/mAP50(B)"]),
            "map50_95": float(row["metrics/mAP50-95(B)"]),
        }

    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "last_epoch": metrics_from_row(rows[-1]),
        "best_epoch_by_map50_95": metrics_from_row(best),
        "epoch_count": len(rows),
    }


def build_results_payload(
    *,
    source: dict[str, Any],
    policies: list[dict[str, Any]],
    proxy: dict[str, Any],
    selected: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
    diagaug_metrics: dict[str, Any],
    random_metrics: dict[str, Any],
    diag001_short: dict[str, Any],
) -> dict[str, Any]:
    ranked = sorted(trials, key=lambda item: float(item["metrics"]["balanced_score"]), reverse=True)
    best = ranked[0] if ranked else None
    diag001_short_metrics = diag001_short["metrics"]
    best_beats_diag001_short = bool(best and float(best["metrics"]["balanced_score"]) > balanced_from_legacy(diag001_short_metrics))
    random_overall = extract_overall_metrics(random_metrics)
    random_trend = None
    if best:
        b = best["metrics"]
        random_trend = {
            "precision_delta_vs_random_50ep": float(b["precision"]) - float(random_overall["precision"]),
            "recall_delta_vs_random_50ep": float(b["recall"]) - float(random_overall["recall"]),
            "map50_delta_vs_random_50ep": float(b["map50"]) - float(random_overall["map50"]),
            "map50_95_delta_vs_random_50ep": float(b["map50_95"]) - float(random_overall["map50_95"]),
            "note": "5 epoch short-training is a trend check and is not directly equivalent to random external 50 epoch.",
        }
    recommendation = "No short-training trials completed."
    if best:
        recommendation = (
            "Run a formal 50 epoch experiment for the best balanced-score policy."
            if best_beats_diag001_short
            else "Do not run formal 50 epoch yet; best policy did not beat diag_policy_001 short-training balanced score."
        )
    return {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "diagnosis-guided policy search + 5 epoch short-training; no formal 50 epoch training",
        "source": source,
        "baseline_metrics": extract_overall_metrics(baseline_metrics),
        "diagaug_50ep_metrics": extract_overall_metrics(diagaug_metrics),
        "random_external_50ep_metrics": random_overall,
        "diag_policy_001_shorttrain_metrics": {
            **{key: diag001_short_metrics[key] for key in ["precision", "recall", "map50", "map50_95"]},
            "balanced_score": balanced_from_legacy(diag001_short_metrics),
            "legacy_short_train_score": diag001_short_metrics.get("short_train_score"),
        },
        "candidate_count": len(policies),
        "candidate_policies": policies,
        "proxy": proxy,
        "short_training_selected_policy_ids": [row["policy_id"] for row in selected],
        "short_training_count": len(trials),
        "trials": trials,
        "balanced_ranking": ranked,
        "best_policy": best,
        "best_policy_id": best["policy_id"] if best else None,
        "best_beats_diag_policy_001_shorttrain": best_beats_diag001_short,
        "random_external_trend_comparison": random_trend,
        "recommend_formal_50epoch": bool(best and best_beats_diag001_short),
        "recommendation": recommendation,
        "summary": {
            "candidate_count": len(policies),
            "proxy_pass_count": proxy["hard_filter_pass_count"],
            "short_training_count": len(trials),
            "best_policy_id": best["policy_id"] if best else None,
            "best_precision": best["metrics"]["precision"] if best else None,
            "best_recall": best["metrics"]["recall"] if best else None,
            "best_map50": best["metrics"]["map50"] if best else None,
            "best_map50_95": best["metrics"]["map50_95"] if best else None,
            "best_balanced_score": best["metrics"]["balanced_score"] if best else None,
            "beats_diag_policy_001_shorttrain": best_beats_diag001_short,
            "recommend_formal_50epoch": bool(best and best_beats_diag001_short),
        },
    }


def balanced_from_legacy(metrics: dict[str, Any]) -> float:
    return float(
        0.30 * float(metrics["map50_95"])
        + 0.25 * float(metrics["map50"])
        + 0.25 * float(metrics["recall"])
        + 0.20 * float(metrics["precision"])
    )


def write_candidate_policies_md(path: Path, policies: list[dict[str, Any]], source: dict[str, Any]) -> None:
    lines = [
        "# Diagnosis-Guided Candidate Policies",
        "",
        "- Generation mode: weighted random policy search.",
        "- Diagnosis vector changes operation sampling probabilities; it does not choose the final policy.",
        f"- Candidate count: `{len(policies)}`",
        "",
        "## Diagnosis Weights",
        "",
    ]
    for key, value in source["diagnosis_vector"].items():
        lines.append(f"- {key}: `{float(value):.4f}`")
    lines.extend(
        [
            "",
            "## Operation Sampling Weights",
            "",
            "| operation | weight | reasons |",
            "| --- | ---: | --- |",
        ]
    )
    for name, weight in source["operation_sampling_weights"].items():
        lines.append(
            f"| {name} | {float(weight):.4f} | {escape_pipe('; '.join(source['operation_sampling_reasons'].get(name, [])))} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Policies",
            "",
            "| policy_id | operations | contains_copy_paste | risk_control |",
            "| --- | --- | --- | --- |",
        ]
    )
    for policy in policies:
        lines.append(
            f"| {policy['policy_id']} | {escape_pipe(format_operations(policy['operations']))} | "
            f"{str(policy['contains_copy_paste']).lower()} | {escape_pipe('; '.join(policy['risk_control']))} |"
        )
    write_text(path, "\n".join(lines) + "\n")


def write_proxy_safety_report(path: Path, proxy: dict[str, Any]) -> None:
    lines = [
        "# Proxy Safety Report",
        "",
        "- Proxy role: safety filtering and coarse screening only.",
        "- Final policy selection is based on 5 epoch short-training balanced score.",
        f"- Candidate count: `{proxy['candidate_count']}`",
        f"- Proxy samples: `{proxy['proxy_samples']}`",
        f"- Hard filter pass count: `{proxy['hard_filter_pass_count']}`",
        "",
        "| rank | policy_id | pass | proxy_score | safety_score | combined | bbox_valid | original_retention | new_bbox_valid | small_retention | cutout_occ | reasons |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in proxy["ranking"]:
        reasons = row["hard_filter_reasons"] or row["safety_soft_penalty_reasons"][:2]
        lines.append(
            "| {rank} | {policy_id} | {passed} | {proxy_score:.4f} | {safety_score:.4f} | {combined:.4f} | {bbox:.4f} | {ret:.4f} | {new:.4f} | {small:.4f} | {cutout:.4f} | {reasons} |".format(
                rank=row["rank"],
                policy_id=row["policy_id"],
                passed=str(row["hard_filter_pass"]).lower(),
                proxy_score=row["proxy_score"],
                safety_score=row["safety_score"],
                combined=row["combined_proxy_safety_score"],
                bbox=row["bbox_valid_rate"],
                ret=row["original_bbox_retention"],
                new=row["new_bbox_valid_rate"],
                small=row["small_object_retention"],
                cutout=row["cutout_bbox_occlusion_rate"],
                reasons=escape_pipe("; ".join(reasons)),
            )
        )
    write_text(path, "\n".join(lines) + "\n")


def write_policy_search_report(path: Path, summary: dict[str, Any]) -> None:
    best = summary["best_policy"]
    lines = [
        "# Diagnosis-Guided Policy Search Report",
        "",
        "## Scope",
        "",
        "- This run performs diagnosis-guided policy search, not fixed-policy selection.",
        "- 30 candidate policies were generated by weighted random sampling.",
        "- Proxy was used only for safety filtering and coarse screening.",
        "- Top proxy-passing policies were selected by 5 epoch short-training balanced score.",
        "- No formal 50 epoch training was run.",
        "",
        "## Reference Metrics",
        "",
        "| method | Precision | Recall | mAP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
        metric_row("baseline 50ep", summary["baseline_metrics"]),
        metric_row("current DiagAug 50ep", summary["diagaug_50ep_metrics"]),
        metric_row("random external 50ep", summary["random_external_50ep_metrics"]),
        metric_row("diag_policy_001 shorttrain", summary["diag_policy_001_shorttrain_metrics"]),
        "",
        "## Candidate Policies",
        "",
        f"- Candidate count: `{summary['candidate_count']}`",
        f"- Candidate list: `outputs/experiments/{RUN_ID}/policies/candidate_policies.md`",
        "",
        "## Proxy Filter",
        "",
        f"- Proxy pass count: `{summary['proxy']['hard_filter_pass_count']}` / `{summary['candidate_count']}`",
        f"- Proxy ranking: `outputs/experiments/{RUN_ID}/proxy/proxy_ranking.json`",
        f"- Proxy safety report: `outputs/experiments/{RUN_ID}/proxy/proxy_safety_report.md`",
        "",
        "## Top10 Short-Training Results",
        "",
        "| rank | policy_id | operations | Precision | Recall | mAP50 | mAP50-95 | balanced | recall_priority | map_priority | time_s |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, trial in enumerate(summary["balanced_ranking"], start=1):
        metrics = trial["metrics"]
        lines.append(
            "| {rank} | {policy_id} | {ops} | {p:.3f} | {r:.3f} | {m50:.3f} | {m95:.3f} | {score:.6f} | {rs:.6f} | {ms:.6f} | {time:.1f} |".format(
                rank=rank,
                policy_id=trial["policy_id"],
                ops=escape_pipe(format_operations(trial["policy"]["operations"])),
                p=metrics["precision"],
                r=metrics["recall"],
                m50=metrics["map50"],
                m95=metrics["map50_95"],
                score=metrics["balanced_score"],
                rs=metrics["recall_priority_score"],
                ms=metrics["map_priority_score"],
                time=trial["training_wall_seconds"],
            )
        )
    if best:
        lines.extend(
            [
                "",
                "## Best Policy",
                "",
                f"- Best policy: `{summary['best_policy_id']}`",
                f"- Operations: `{format_operations(best['policy']['operations'])}`",
                f"- Balanced score: `{best['metrics']['balanced_score']:.6f}`",
                f"- Exceeds diag_policy_001 shorttrain balanced score: `{str(summary['best_beats_diag_policy_001_shorttrain']).lower()}`",
                f"- Recommend formal 50 epoch: `{str(summary['recommend_formal_50epoch']).lower()}`",
                "",
                "## Random External Trend",
                "",
                "- The random external result is a 50 epoch run, while this search uses 5 epoch short-training.",
                "- Treat this as directional evidence only, not a final head-to-head.",
                f"- Trend deltas vs random 50ep: `{json.dumps(summary['random_external_trend_comparison'], ensure_ascii=False)}`",
                "",
                "## Recommendation",
                "",
                summary["recommendation"],
            ]
        )
    write_text(path, "\n".join(lines) + "\n")


def write_best_policy_summary(path: Path, summary: dict[str, Any]) -> None:
    best = summary["best_policy"]
    if not best:
        write_text(path, "# Best Policy Summary\n\nNo policy completed short-training.\n")
        return
    metrics = best["metrics"]
    lines = [
        "# Best Policy Summary",
        "",
        f"- policy_id: `{best['policy_id']}`",
        f"- operations: `{format_operations(best['policy']['operations'])}`",
        f"- Precision: `{metrics['precision']:.3f}`",
        f"- Recall: `{metrics['recall']:.3f}`",
        f"- mAP50: `{metrics['map50']:.3f}`",
        f"- mAP50-95: `{metrics['map50_95']:.3f}`",
        f"- balanced_score: `{metrics['balanced_score']:.6f}`",
        f"- recall_priority_score: `{metrics['recall_priority_score']:.6f}`",
        f"- map_priority_score: `{metrics['map_priority_score']:.6f}`",
        f"- beats diag_policy_001 shorttrain: `{str(summary['best_beats_diag_policy_001_shorttrain']).lower()}`",
        f"- recommend formal 50 epoch: `{str(summary['recommend_formal_50epoch']).lower()}`",
        f"- best.pt: `{best['artifacts']['best_pt']}`",
    ]
    write_text(path, "\n".join(lines) + "\n")


def write_trial_report(path: Path, trial: dict[str, Any]) -> None:
    metrics = trial["metrics"]
    dataset = trial["augmented_dataset"]
    lines = [
        f"# Policy Search Short-Training: {trial['policy_id']}",
        "",
        f"- operations: `{format_operations(trial['policy']['operations'])}`",
        f"- train images / bboxes: `{dataset['train_images']}` / `{dataset['train_bboxes']}`",
        f"- val images / bboxes: `{dataset['val_images']}` / `{dataset['val_bboxes']}`",
        f"- Precision: `{metrics['precision']:.3f}`",
        f"- Recall: `{metrics['recall']:.3f}`",
        f"- mAP50: `{metrics['map50']:.3f}`",
        f"- mAP50-95: `{metrics['map50_95']:.3f}`",
        f"- balanced_score: `{metrics['balanced_score']:.6f}`",
        f"- recall_priority_score: `{metrics['recall_priority_score']:.6f}`",
        f"- map_priority_score: `{metrics['map_priority_score']:.6f}`",
        f"- OOM: `{str(trial['oom']).lower()}`",
        f"- training wall seconds: `{trial['training_wall_seconds']:.1f}`",
        "",
        "## Per-Class Recall/AP50",
        "",
        "| class id | class | Recall | AP50 |",
        "| ---: | --- | ---: | ---: |",
    ]
    for row in metrics["per_class"]:
        lines.append(f"| {row['class_id']} | {row['name']} | {row['recall']:.3f} | {row['ap50']:.3f} |")
    write_text(path, "\n".join(lines) + "\n")


def write_dataset_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Augmented Dataset Summary",
        "",
        f"- policy_id: `{report['policy_id']}`",
        f"- train images: `{report['train_images']}`",
        f"- train bboxes: `{report['train_bboxes']}`",
        f"- val images: `{report['val_images']}`",
        f"- val bboxes: `{report['val_bboxes']}`",
        f"- bbox_valid_rate: `{report['bbox_valid_rate']:.6f}`",
        f"- cutout_bbox_occlusion_rate: `{report['cutout_bbox_occlusion_rate']:.6f}`",
        f"- operation_application_counts: `{json.dumps(report['operation_application_counts'], ensure_ascii=False)}`",
    ]
    write_text(path, "\n".join(lines) + "\n")


def update_state_docs(summary: dict[str, Any]) -> None:
    section = build_state_section(summary)
    for path in [PROJECT_ROOT / "PROJECT_STATE.md", PROJECT_ROOT / "CODEX_HANDOFF.md", PROJECT_ROOT / "EXPERIMENT_LOG.md"]:
        upsert_section(path, "DIAGNOSIS_GUIDED_POLICY_SEARCH", section)


def build_state_section(summary: dict[str, Any]) -> str:
    best = summary["best_policy"]
    best_line = "`none`"
    if best:
        best_line = (
            f"`{best['policy_id']}` balanced={best['metrics']['balanced_score']:.6f} "
            f"P/R/mAP50/mAP50-95={best['metrics']['precision']:.3f}/"
            f"{best['metrics']['recall']:.3f}/{best['metrics']['map50']:.3f}/{best['metrics']['map50_95']:.3f}"
        )
    return "\n".join(
        [
            "## Diagnosis-Guided Policy Search",
            "",
            f"- Run ID: `{RUN_ID}`",
            "- Scope: diagnosis-guided sampled policy search plus 5 epoch short-training; no formal 50 epoch training.",
            "- Change in method: diagnosis adjusts operation sampling probabilities instead of directly selecting a fixed policy.",
            f"- Candidate policies: `{summary['candidate_count']}`",
            f"- Proxy pass count: `{summary['proxy']['hard_filter_pass_count']}`",
            f"- Short-training trials: `{summary['short_training_count']}`",
            f"- Best balanced-score policy: {best_line}",
            f"- Beats diag_policy_001 short-training balanced score: `{str(summary['best_beats_diag_policy_001_shorttrain']).lower()}`",
            f"- Recommend formal 50 epoch: `{str(summary['recommend_formal_50epoch']).lower()}`",
            f"- Report: `outputs/experiments/{RUN_ID}/reports/policy_search_report.md`",
            f"- Results JSON: `outputs/experiments/{RUN_ID}/reports/policy_search_results.json`",
            f"- Best summary: `outputs/experiments/{RUN_ID}/reports/best_policy_summary.md`",
        ]
    )


def upsert_section(path: Path, key: str, section: str) -> None:
    start = f"<!-- {key}_START -->"
    end = f"<!-- {key}_END -->"
    text = path.read_text(encoding="utf-8-sig", errors="replace") if path.exists() else ""
    block = f"{start}\n{section.rstrip()}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _match: block, text)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def metric_row(name: str, metrics: dict[str, Any]) -> str:
    return "| {name} | {precision:.3f} | {recall:.3f} | {map50:.3f} | {map50_95:.3f} |".format(
        name=name,
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        map50=float(metrics["map50"]),
        map50_95=float(metrics["map50_95"]),
    )


def extract_overall_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    if "validation" in payload:
        return payload["validation"]["overall"]
    if "final_metrics" in payload:
        return payload["final_metrics"]
    if "metrics" in payload:
        return payload["metrics"]
    return payload


def extract_per_class_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "validation" in payload:
        return list(payload["validation"].get("per_class", []))
    if "final_per_class" in payload:
        return list(payload.get("final_per_class", []))
    if "metrics" in payload:
        return list(payload["metrics"].get("per_class", []))
    return list(payload.get("per_class", []))


def collect_train_class_counts(records: list[YoloImageRecord]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for record in records:
        sample = load_yolo_sample(record)
        counts.update(int(label) for label in np.asarray(sample["labels"], dtype=np.int64))
    return dict(counts)


def class_distribution_change(before: Counter[int], after: Counter[int]) -> float:
    keys = sorted(set(before) | set(after))
    before_total = sum(before.values())
    after_total = sum(after.values())
    if before_total <= 0 or after_total <= 0:
        return 0.0 if before_total == after_total else 1.0
    distance = 0.0
    for key in keys:
        distance += abs(before.get(key, 0) / before_total - after.get(key, 0) / after_total)
    return float(distance / 2.0)


def flatten_diagnosis_vector(vector: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key in ["small_object_score", "low_contrast_score", "class_imbalance_score", "localization_score", "false_positive_score"]:
        value = vector.get(key, 0.0)
        if isinstance(value, dict):
            value = value.get("score", 0.0)
        out[key] = float(np.clip(float(value or 0.0), 0.0, 1.0))
    return out


def filter_valid_sample(sample: dict[str, Any], *, class_count: int) -> tuple[dict[str, Any], dict[str, int]]:
    image = sample["image"]
    h, w = image.shape[:2]
    labels = np.asarray(sample["labels"], dtype=np.int64)
    boxes = np.asarray(sample["bboxes"], dtype=np.float32).reshape(-1, 4)
    raw_count = len(boxes)
    class_valid = (labels >= 0) & (labels < class_count) if len(labels) else np.zeros((0,), dtype=bool)
    box_valid = xyxy_valid_in_bounds(boxes, w, h)
    valid = class_valid & box_valid
    out = dict(sample)
    out["labels"] = labels[valid]
    out["bboxes"] = boxes[valid]
    return out, {
        "raw_bbox_count": raw_count,
        "valid_bbox_count": int(valid.sum()),
        "invalid_bbox_count": int((~box_valid).sum()) if len(box_valid) else 0,
        "class_out_of_range_count": int((~class_valid).sum()) if len(class_valid) else 0,
    }


def xyxy_valid_in_bounds(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=bool)
    return (
        np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
        & (boxes[:, 0] >= 0)
        & (boxes[:, 1] >= 0)
        & (boxes[:, 2] <= width)
        & (boxes[:, 3] <= height)
    )


def box_area(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])


def count_images(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def count_label_rows(root: Path) -> int:
    total = 0
    for path in root.rglob("*.txt"):
        total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


def has_oom(*paths: Path) -> bool:
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace").lower() for path in paths if path.exists())
    return any(pattern in text for pattern in ["out of memory", "cuda oom", "cuda out of memory"])


def slim_proxy_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rank",
        "policy_id",
        "proxy_score",
        "safety_score",
        "combined_proxy_safety_score",
        "hard_filter_pass",
        "hard_filter_reasons",
        "safety_soft_penalty_reasons",
        "bbox_valid_rate",
        "original_bbox_retention",
        "new_bbox_valid_rate",
        "small_object_retention",
        "cutout_bbox_occlusion_rate",
        "copy_paste_audit",
    ]
    return {key: row.get(key) for key in keys}


def format_operations(operations: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{op.get('name')}(p={float(op.get('prob', 0.0)):.3f}, s={float(op.get('strength', 0.0)):.3f})"
        for op in operations
    )


def escape_pipe(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


if __name__ == "__main__":
    main()
