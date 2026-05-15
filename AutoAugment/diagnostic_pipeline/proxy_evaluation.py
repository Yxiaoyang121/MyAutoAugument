from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from AutoAugment.bbox.convert import xyxy_to_yolo
from AutoAugment.policies import Policy, apply_policy_to_sample
from AutoAugment.search import ProxyEvaluator
from AutoAugment.search.proxy_metrics import (
    DEFAULT_EDGE_MARGIN,
    DEFAULT_RARE_CLASS_COUNT_THRESHOLD,
    DEFAULT_SMALL_AREA_THRESHOLD,
    DEFAULT_TINY_AREA_THRESHOLD,
    apply_proxy_hard_filter,
    compute_proxy_score,
    yolo_bbox_edge_mask,
)
from AutoAugment.utils import YoloImageRecord, flatten_relative_stem, load_yolo_sample, sample_records, save_yolo_sample
from AutoAugment.diagnostic_pipeline.common import write_json


def run_proxy_evaluation(
    *,
    policies_payload: dict[str, Any],
    train_records: list[YoloImageRecord],
    output_dir: str | Path,
    seed: int = 42,
    proxy_samples: int = 32,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate candidate policies with non-training proxy metrics and rank them."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    policies = list(policies_payload.get("policies", []))
    if dry_run:
        rows = [_dry_run_proxy_row(policy, index) for index, policy in enumerate(policies)]
        payload = _proxy_payload(rows, dry_run=True, seed=seed, proxy_samples=proxy_samples)
        write_json(output / "proxy_metrics.json", payload)
        write_json(output / "proxy_ranking.json", payload["ranking"])
        return payload

    rng = np.random.default_rng(seed)
    selected_records = sample_records(train_records, proxy_samples, rng)
    source_context = collect_source_proxy_context(selected_records)
    rows: list[dict[str, Any]] = []
    evaluator = ProxyEvaluator(score_version="dataset2_v1")
    artifacts_root = output / "proxy_artifacts"
    if artifacts_root.exists():
        shutil.rmtree(artifacts_root)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    for index, policy_dict in enumerate(policies):
        policy = Policy.from_dict(policy_dict)
        policy_id = str(policy_dict.get("policy_id", policy.name))
        candidate_dir = artifacts_root / policy_id
        images_dir = candidate_dir / "images"
        labels_dir = candidate_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        write_json(candidate_dir / "policy.json", policy_dict)
        candidate_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.uint32).max)))
        write_policy_dataset(selected_records, policy, images_dir=images_dir, labels_dir=labels_dir, rng=candidate_rng)
        context = {
            **source_context,
            "images_dir": str(images_dir.resolve()),
            "labels_dir": str(labels_dir.resolve()),
            "proxy_score_version": "dataset2_v1",
        }
        evaluation = evaluator.evaluate(candidate_dir, policy, context=context)
        metrics = dict(evaluation.metrics)
        proxy_score, components, missing_components = compute_proxy_score(metrics, version="dataset2_v1")
        hard_filter_pass, hard_filter_reasons = apply_proxy_hard_filter(metrics, profile="dataset2_v1")
        after_counts = collect_class_counts(labels_dir)
        metrics.update(
            {
                "policy_id": policy_id,
                "candidate_index": index,
                "proxy_score": proxy_score,
                "proxy_score_components": components,
                "missing_components": missing_components,
                "hard_filter_pass": hard_filter_pass,
                "hard_filter_reasons": hard_filter_reasons,
                "small_object_retention": metrics.get("small_target_retention"),
                "invalid_sample_count": int(metrics.get("image_count", 0) - metrics.get("valid_image_count", 0)),
                "class_distribution_change": class_distribution_change(source_context["source_class_counts"], after_counts),
                "source_class_counts": source_context["source_class_counts"],
                "after_class_counts": after_counts,
                "policy": policy_dict,
            }
        )
        write_json(candidate_dir / "proxy_metrics.json", metrics)
        rows.append(metrics)

    payload = _proxy_payload(rows, dry_run=False, seed=seed, proxy_samples=len(selected_records))
    write_json(output / "proxy_metrics.json", payload)
    write_json(output / "proxy_ranking.json", payload["ranking"])
    return payload


def write_policy_dataset(
    records: list[YoloImageRecord],
    policy: Policy,
    *,
    images_dir: str | Path,
    labels_dir: str | Path,
    rng: np.random.Generator,
) -> int:
    """Apply a policy to records and write a temporary YOLO dataset."""

    image_root = Path(images_dir)
    label_root = Path(labels_dir)
    source_label_count = 0
    for record in records:
        sample = load_yolo_sample(record)
        source_label_count += len(sample["labels"])
        augmented = apply_policy_to_sample(sample, policy, rng=rng)
        stem = flatten_relative_stem(record.relative_path)
        image_name = f"{stem}{record.image_path.suffix.lower()}"
        label_name = f"{stem}.txt"
        save_yolo_sample(augmented, image_root / image_name, label_root / label_name)
    return source_label_count


def collect_source_proxy_context(records: list[YoloImageRecord]) -> dict[str, Any]:
    """Collect source label, size, edge, class, and image statistics for proxy scoring."""

    source_label_count = 0
    tiny_count = 0
    small_count = 0
    edge_count = 0
    class_counts: dict[int, int] = {}
    image_means_by_output_name: dict[str, float] = {}
    for record in records:
        sample = load_yolo_sample(record)
        image = sample["image"]
        height, width = image.shape[:2]
        image_name = f"{flatten_relative_stem(record.relative_path)}{record.image_path.suffix.lower()}"
        image_means_by_output_name[image_name] = float(image.mean())
        yolo_boxes = xyxy_to_yolo(sample["bboxes"], width, height)
        source_label_count += len(sample["labels"])
        if yolo_boxes.size:
            areas = yolo_boxes[:, 2] * yolo_boxes[:, 3]
            tiny_count += int((areas <= DEFAULT_TINY_AREA_THRESHOLD).sum())
            small_count += int((areas <= DEFAULT_SMALL_AREA_THRESHOLD).sum())
            edge_count += int(yolo_bbox_edge_mask(yolo_boxes, edge_margin=DEFAULT_EDGE_MARGIN).sum())
        for label in sample["labels"]:
            class_id = int(label)
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    return {
        "source_label_count": source_label_count,
        "source_tiny_box_count": tiny_count,
        "source_small_box_count": small_count,
        "source_edge_box_count": edge_count,
        "source_class_counts": class_counts,
        "source_image_means_by_output_name": image_means_by_output_name,
        "tiny_area_threshold": DEFAULT_TINY_AREA_THRESHOLD,
        "small_area_threshold": DEFAULT_SMALL_AREA_THRESHOLD,
        "edge_margin": DEFAULT_EDGE_MARGIN,
        "rare_class_count_threshold": DEFAULT_RARE_CLASS_COUNT_THRESHOLD,
    }


def collect_class_counts(labels_dir: str | Path) -> dict[int, int]:
    """Count YOLO labels by class id."""

    counts: dict[int, int] = {}
    for label_path in Path(labels_dir).rglob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def class_distribution_change(before: dict[int, int], after: dict[int, int]) -> float:
    """Return half L1 distance between normalized class distributions."""

    keys = sorted(set(before) | set(after))
    before_total = sum(before.values())
    after_total = sum(after.values())
    if before_total <= 0 or after_total <= 0:
        return 0.0 if before_total == after_total else 1.0
    distance = 0.0
    for key in keys:
        distance += abs(before.get(key, 0) / before_total - after.get(key, 0) / after_total)
    return float(distance / 2.0)


def _proxy_payload(rows: list[dict[str, Any]], *, dry_run: bool, seed: int, proxy_samples: int) -> dict[str, Any]:
    ranking = sorted(
        rows,
        key=lambda item: (
            bool(item.get("hard_filter_pass")),
            float(item.get("proxy_score", 0.0) or 0.0),
            -len(item.get("hard_filter_reasons", []) or []),
        ),
        reverse=True,
    )
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank
    return {
        "stage": "proxy_evaluation",
        "status": "planned" if dry_run else "completed",
        "dry_run": dry_run,
        "seed": int(seed),
        "proxy_samples": int(proxy_samples),
        "candidate_count": len(rows),
        "metrics": rows,
        "ranking": ranking,
        "filter_rules": {
            "bbox_retention_raw": "reject if below dataset2_v1 threshold",
            "bbox_valid_rate": "reject if below dataset2_v1 threshold",
            "exposure_score": "penalize over-dark or over-bright images",
            "small_object_retention": "reject if small/tiny target retention is too low",
        },
    }


def _dry_run_proxy_row(policy: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "policy_id": policy.get("policy_id", policy.get("name", f"diag_policy_{index + 1:03d}")),
        "candidate_index": index,
        "proxy_score": 0.0,
        "bbox_retention": None,
        "bbox_valid_rate": None,
        "diversity_score": None,
        "exposure_score": None,
        "mean_intensity": None,
        "mean_std": None,
        "class_distribution_change": None,
        "small_object_retention": None,
        "invalid_sample_count": None,
        "hard_filter_pass": False,
        "hard_filter_reasons": ["dry_run_not_evaluated"],
        "policy": policy,
    }
