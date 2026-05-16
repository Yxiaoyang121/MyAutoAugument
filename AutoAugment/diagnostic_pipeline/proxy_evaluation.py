from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from AutoAugment.bbox.convert import xyxy_to_yolo
from AutoAugment.bbox.iou import bbox_iou
from AutoAugment.policies import Policy, apply_policy_to_sample
from AutoAugment.search import ProxyEvaluator
from AutoAugment.search.proxy_metrics import (
    DEFAULT_EDGE_MARGIN,
    DEFAULT_RARE_CLASS_COUNT_THRESHOLD,
    DEFAULT_SMALL_AREA_THRESHOLD,
    DEFAULT_TINY_AREA_THRESHOLD,
    apply_proxy_hard_filter,
    compute_safety_score,
    compute_proxy_score,
    safety_soft_penalty_reasons,
    yolo_bbox_edge_mask,
)
from AutoAugment.utils import YoloImageRecord, flatten_relative_stem, load_yolo_sample, sample_records, save_yolo_sample
from AutoAugment.diagnostic_pipeline.common import write_json
from AutoAugment.diagnostic_pipeline.common import write_markdown


def run_proxy_evaluation(
    *,
    policies_payload: dict[str, Any],
    train_records: list[YoloImageRecord],
    output_dir: str | Path,
    seed: int = 42,
    proxy_samples: int = 32,
    dry_run: bool = False,
    class_names: dict[int, str] | None = None,
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
        write_copy_paste_filter_audit(output / "copy_paste_filter_audit.md", rows)
        return payload

    rng = np.random.default_rng(seed)
    selected_records = sample_records(train_records, proxy_samples, rng)
    source_context = collect_source_proxy_context(selected_records, class_names=class_names)
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
        copy_paste_audit = write_policy_dataset(
            selected_records,
            policy,
            images_dir=images_dir,
            labels_dir=labels_dir,
            rng=candidate_rng,
            audit_dir=candidate_dir / "copy_paste_audit",
            policy_dict=policy_dict,
            class_count=int(source_context.get("class_count", 0) or 0),
        )
        context = {
            **source_context,
            "images_dir": str(images_dir.resolve()),
            "labels_dir": str(labels_dir.resolve()),
            "proxy_score_version": "dataset2_v1",
        }
        evaluation = evaluator.evaluate(candidate_dir, policy, context=context)
        metrics = dict(evaluation.metrics)
        proxy_score, components, missing_components = compute_proxy_score(metrics, version="dataset2_v1")
        safety_score, safety_components, safety_missing = compute_safety_score(metrics, version="dataset2_v1")
        soft_penalty_reasons = safety_soft_penalty_reasons(metrics, profile="dataset2_v1")
        hard_filter_pass, hard_filter_reasons = apply_proxy_hard_filter(metrics, profile="dataset2_v1")
        combined_score = float(0.70 * proxy_score + 0.30 * safety_score)
        after_counts = collect_class_counts(labels_dir)
        metrics.update(
            {
                "policy_id": policy_id,
                "candidate_index": index,
                "proxy_score": proxy_score,
                "safety_score": safety_score,
                "safety_score_formula": "total_bbox_valid_rate * original_bbox_retention * small_object_retention * exposure_score",
                "safety_score_components": safety_components,
                "safety_missing_components": safety_missing,
                "safety_soft_penalty_reasons": soft_penalty_reasons,
                "combined_proxy_safety_score": combined_score,
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
                "copy_paste_audit": copy_paste_audit,
            }
        )
        write_json(candidate_dir / "proxy_metrics.json", metrics)
        rows.append(metrics)

    payload = _proxy_payload(rows, dry_run=False, seed=seed, proxy_samples=len(selected_records))
    write_json(output / "proxy_metrics.json", payload)
    write_json(output / "proxy_ranking.json", payload["ranking"])
    write_copy_paste_filter_audit(output / "copy_paste_filter_audit.md", rows)
    return payload


def write_policy_dataset(
    records: list[YoloImageRecord],
    policy: Policy,
    *,
    images_dir: str | Path,
    labels_dir: str | Path,
    rng: np.random.Generator,
    audit_dir: str | Path | None = None,
    policy_dict: dict[str, Any] | None = None,
    class_count: int = 0,
) -> dict[str, Any]:
    """Apply a policy to records and write a temporary YOLO dataset."""

    image_root = Path(images_dir)
    label_root = Path(labels_dir)
    source_label_count = 0
    audit = CopyPasteAudit(audit_dir, class_count=class_count) if audit_dir is not None else None
    for record in records:
        sample = load_yolo_sample(record)
        source_label_count += len(sample["labels"])
        augmented = apply_policy_to_sample(sample, policy, rng=rng)
        if audit is not None:
            audit.add_sample(record, sample, augmented, policy_dict or policy.to_dict())
        stem = flatten_relative_stem(record.relative_path)
        image_name = f"{stem}{record.image_path.suffix.lower()}"
        label_name = f"{stem}.txt"
        save_yolo_sample(augmented, image_root / image_name, label_root / label_name)
    if audit is None:
        return {"enabled": False, "source_label_count": source_label_count}
    return audit.finish(source_label_count=source_label_count)


def collect_source_proxy_context(records: list[YoloImageRecord], class_names: dict[int, str] | None = None) -> dict[str, Any]:
    """Collect source label, size, edge, class, and image statistics for proxy scoring."""

    source_label_count = 0
    tiny_count = 0
    small_count = 0
    edge_count = 0
    class_counts: dict[int, int] = {}
    image_means_by_output_name: dict[str, float] = {}
    box_counts_by_output_name: dict[str, int] = {}
    for record in records:
        sample = load_yolo_sample(record)
        image = sample["image"]
        height, width = image.shape[:2]
        image_name = f"{flatten_relative_stem(record.relative_path)}{record.image_path.suffix.lower()}"
        image_means_by_output_name[image_name] = float(image.mean())
        box_counts_by_output_name[image_name] = int(len(sample["labels"]))
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
        "source_box_counts_by_output_name": box_counts_by_output_name,
        "class_count": max([int(key) + 1 for key in (class_names or {})] + [max(class_counts) + 1 if class_counts else 0]),
        "tiny_area_threshold": DEFAULT_TINY_AREA_THRESHOLD,
        "small_area_threshold": DEFAULT_SMALL_AREA_THRESHOLD,
        "edge_margin": DEFAULT_EDGE_MARGIN,
        "rare_class_count_threshold": DEFAULT_RARE_CLASS_COUNT_THRESHOLD,
    }


class CopyPasteAudit:
    """Collect bbox-level audit data for policies that contain copy_paste."""

    def __init__(self, audit_dir: str | Path, *, class_count: int = 0, max_debug_images: int = 8) -> None:
        self.audit_dir = Path(audit_dir)
        self.debug_dir = self.audit_dir / "debug"
        self.class_count = int(class_count)
        self.max_debug_images = int(max_debug_images)
        self.records: list[dict[str, Any]] = []
        self.policy_has_copy_paste = False

    def add_sample(
        self,
        record: YoloImageRecord,
        before: dict[str, Any],
        after: dict[str, Any],
        policy_dict: dict[str, Any],
    ) -> None:
        operations = list(policy_dict.get("operations", []) or [])
        if not any(str(operation.get("name", "")).lower() == "copy_paste" for operation in operations):
            return
        self.policy_has_copy_paste = True
        image = after["image"]
        height, width = image.shape[:2]
        before_labels = np.asarray(before["labels"], dtype=np.int64)
        before_boxes = np.asarray(before["bboxes"], dtype=np.float32).reshape(-1, 4)
        after_labels = np.asarray(after["labels"], dtype=np.int64)
        after_boxes = np.asarray(after["bboxes"], dtype=np.float32).reshape(-1, 4)
        before_count = int(len(before_labels))
        after_count = int(len(after_labels))
        original_count_after = min(before_count, after_count)
        new_boxes = after_boxes[before_count:] if after_count > before_count else np.zeros((0, 4), dtype=np.float32)
        new_labels = after_labels[before_count:] if after_count > before_count else np.zeros((0,), dtype=np.int64)
        total_shape_valid = _xyxy_shape_valid_mask(after_boxes)
        total_legal = _xyxy_in_bounds_mask(after_boxes, width, height)
        new_legal = _xyxy_in_bounds_mask(new_boxes, width, height)
        iou_retained = _count_iou_retained_originals(before_labels, before_boxes, after_labels, after_boxes)
        class_out_of_range = 0
        if self.class_count > 0 and after_count:
            class_out_of_range = int(((after_labels < 0) | (after_labels >= self.class_count)).sum())
        sample_record = {
            "image": str(record.image_path),
            "before_bbox_count": before_count,
            "after_bbox_count": after_count,
            "new_bbox_count": int(max(0, after_count - before_count)),
            "original_bbox_retention": original_count_after / max(1, before_count),
            "iou_matched_original_bbox_retention": iou_retained / max(1, before_count),
            "new_bbox_valid_rate": float(new_legal.mean()) if len(new_legal) else 1.0,
            "total_bbox_valid_rate": float(total_legal.mean()) if len(total_legal) else 1.0,
            "invalid_bbox_count": int((~total_shape_valid).sum()),
            "out_of_bounds_bbox_count": int((~total_legal).sum()),
            "class_out_of_range_count": class_out_of_range,
            "new_class_out_of_range_count": int(((new_labels < 0) | (new_labels >= self.class_count)).sum())
            if self.class_count > 0 and len(new_labels)
            else 0,
        }
        self.records.append(sample_record)
        if len(self.records) <= self.max_debug_images:
            self._write_debug_image(record, before["image"], before_boxes, after_boxes)

    def finish(self, *, source_label_count: int) -> dict[str, Any]:
        if not self.policy_has_copy_paste:
            return {"enabled": False, "source_label_count": source_label_count, "reason": "policy_has_no_copy_paste"}
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        before_total = sum(int(item["before_bbox_count"]) for item in self.records)
        after_total = sum(int(item["after_bbox_count"]) for item in self.records)
        new_total = sum(int(item["new_bbox_count"]) for item in self.records)
        summary = {
            "enabled": True,
            "sample_count": len(self.records),
            "source_label_count": int(source_label_count),
            "before_bbox_count": before_total,
            "after_bbox_count": after_total,
            "new_bbox_count": new_total,
            "original_bbox_retention": _weighted_rate(self.records, "original_bbox_retention", "before_bbox_count"),
            "iou_matched_original_bbox_retention": _weighted_rate(
                self.records, "iou_matched_original_bbox_retention", "before_bbox_count"
            ),
            "new_bbox_valid_rate": _weighted_rate(self.records, "new_bbox_valid_rate", "new_bbox_count", default=1.0),
            "total_bbox_valid_rate": _weighted_rate(self.records, "total_bbox_valid_rate", "after_bbox_count", default=1.0),
            "invalid_bbox_count": sum(int(item["invalid_bbox_count"]) for item in self.records),
            "out_of_bounds_bbox_count": sum(int(item["out_of_bounds_bbox_count"]) for item in self.records),
            "class_out_of_range_count": sum(int(item["class_out_of_range_count"]) for item in self.records),
            "debug_dir": str(self.debug_dir.resolve()),
        }
        payload = {"summary": summary, "samples": self.records}
        write_json(self.audit_dir / "copy_paste_filter_audit.json", payload)
        write_markdown(
            self.audit_dir / "copy_paste_filter_audit.md",
            [
                "# Copy Paste Filter Audit",
                "",
                f"- Samples: {summary['sample_count']}",
                f"- Before bboxes: {summary['before_bbox_count']}",
                f"- After bboxes: {summary['after_bbox_count']}",
                f"- New bboxes: {summary['new_bbox_count']}",
                f"- Original bbox retention: {summary['original_bbox_retention']:.4f}",
                f"- New bbox valid rate: {summary['new_bbox_valid_rate']:.4f}",
                f"- Total bbox valid rate: {summary['total_bbox_valid_rate']:.4f}",
                f"- Invalid bbox count: {summary['invalid_bbox_count']}",
                f"- Out-of-bounds bbox count: {summary['out_of_bounds_bbox_count']}",
                f"- Class out-of-range count: {summary['class_out_of_range_count']}",
                f"- Debug dir: {summary['debug_dir']}",
            ],
        )
        return summary

    def _write_debug_image(
        self,
        record: YoloImageRecord,
        before_image: np.ndarray,
        before_boxes: np.ndarray,
        after_boxes: np.ndarray,
    ) -> None:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        canvas = before_image.copy()
        for box in before_boxes:
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 0), 1)
        for box in after_boxes:
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 128, 255), 1)
        out = self.debug_dir / f"{flatten_relative_stem(record.relative_path)}_copy_paste_audit.jpg"
        cv2.imwrite(str(out), canvas)


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


def _xyxy_shape_valid_mask(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )


def _xyxy_in_bounds_mask(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        _xyxy_shape_valid_mask(boxes)
        & (boxes[:, 0] >= 0)
        & (boxes[:, 1] >= 0)
        & (boxes[:, 2] <= width)
        & (boxes[:, 3] <= height)
    )


def _count_iou_retained_originals(
    before_labels: np.ndarray,
    before_boxes: np.ndarray,
    after_labels: np.ndarray,
    after_boxes: np.ndarray,
    *,
    iou_threshold: float = 0.95,
) -> int:
    if len(before_boxes) == 0 or len(after_boxes) == 0:
        return 0
    ious = bbox_iou(before_boxes, after_boxes)
    used_after: set[int] = set()
    retained = 0
    for before_index, before_label in enumerate(before_labels):
        candidates = [
            after_index
            for after_index, after_label in enumerate(after_labels)
            if after_index not in used_after and int(after_label) == int(before_label)
        ]
        if not candidates:
            continue
        best_after = max(candidates, key=lambda after_index: float(ious[before_index, after_index]))
        if float(ious[before_index, best_after]) >= iou_threshold:
            retained += 1
            used_after.add(best_after)
    return retained


def _weighted_rate(records: list[dict[str, Any]], value_key: str, weight_key: str, *, default: float = 0.0) -> float:
    total_weight = sum(float(item.get(weight_key, 0) or 0) for item in records)
    if total_weight <= 0:
        return float(default)
    weighted = sum(float(item.get(value_key, 0.0) or 0.0) * float(item.get(weight_key, 0) or 0) for item in records)
    return float(weighted / total_weight)


def write_copy_paste_filter_audit(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    audited = [row for row in rows if (row.get("copy_paste_audit") or {}).get("enabled")]
    aggregate = {
        "policy_count": len(rows),
        "copy_paste_policy_count": len(audited),
        "copy_paste_was_hard_filtered": any(
            any(str(reason).startswith("copy_paste") for reason in row.get("hard_filter_reasons", []) or [])
            or (
                any(str(op.get("name", "")).lower() == "copy_paste" for op in (row.get("policy", {}).get("operations", []) or []))
                and not bool(row.get("hard_filter_pass", False))
            )
            for row in rows
        ),
    }
    write_json(output.with_suffix(".json"), {"summary": aggregate, "policies": rows})
    lines = [
        "# Copy Paste Filter Audit",
        "",
        f"- Candidate policies: {aggregate['policy_count']}",
        f"- Policies containing copy_paste: {aggregate['copy_paste_policy_count']}",
        f"- Any copy_paste policy hard rejected: {aggregate['copy_paste_was_hard_filtered']}",
        "",
        "## Policy Results",
    ]
    if not audited:
        lines.append("- No evaluated policy contained copy_paste.")
    for row in audited:
        audit = row.get("copy_paste_audit") or {}
        lines.extend(
            [
                f"- {row.get('policy_id')}: hard_filter_pass={row.get('hard_filter_pass')} "
                f"proxy={float(row.get('proxy_score', 0.0) or 0.0):.4f} "
                f"safety={float(row.get('safety_score', 0.0) or 0.0):.4f} "
                f"combined={float(row.get('combined_proxy_safety_score', 0.0) or 0.0):.4f}",
                f"  - before_bbox_count={audit.get('before_bbox_count', 0)} after_bbox_count={audit.get('after_bbox_count', 0)} new_bbox_count={audit.get('new_bbox_count', 0)}",
                f"  - original_bbox_retention={float(audit.get('original_bbox_retention', 0.0) or 0.0):.4f} "
                f"new_bbox_valid_rate={float(audit.get('new_bbox_valid_rate', 0.0) or 0.0):.4f} "
                f"total_bbox_valid_rate={float(audit.get('total_bbox_valid_rate', 0.0) or 0.0):.4f}",
                f"  - invalid_bbox_count={audit.get('invalid_bbox_count', 0)} "
                f"out_of_bounds_bbox_count={audit.get('out_of_bounds_bbox_count', 0)} "
                f"class_out_of_range_count={audit.get('class_out_of_range_count', 0)}",
                f"  - hard_filter_reasons={row.get('hard_filter_reasons', [])}",
                f"  - soft_penalty_reasons={row.get('safety_soft_penalty_reasons', [])}",
                f"  - debug_dir={audit.get('debug_dir')}",
            ]
        )
    write_markdown(output, lines)


def _proxy_payload(rows: list[dict[str, Any]], *, dry_run: bool, seed: int, proxy_samples: int) -> dict[str, Any]:
    ranking = sorted(
        rows,
        key=lambda item: (
            bool(item.get("hard_filter_pass")),
            float(item.get("combined_proxy_safety_score", item.get("proxy_score", 0.0)) or 0.0),
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
            "hard_reject": "only severe bbox/class/exposure/strength failures reject a candidate",
            "safety_score": "SafetyScore = total_bbox_valid_rate * original_bbox_retention * small_object_retention * exposure_score",
            "soft_penalty": "ordinary risks below target thresholds are retained as safety_soft_penalty_reasons",
            "final_ranking": "sort by hard_filter_pass then 0.70 * proxy_score + 0.30 * SafetyScore",
        },
    }


def _dry_run_proxy_row(policy: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "policy_id": policy.get("policy_id", policy.get("name", f"diag_policy_{index + 1:03d}")),
        "candidate_index": index,
        "proxy_score": 0.0,
        "safety_score": 0.0,
        "combined_proxy_safety_score": 0.0,
        "bbox_retention": None,
        "bbox_valid_rate": None,
        "total_bbox_valid_rate": None,
        "original_bbox_retention": None,
        "new_bbox_valid_rate": None,
        "diversity_score": None,
        "exposure_score": None,
        "mean_intensity": None,
        "mean_std": None,
        "class_distribution_change": None,
        "small_object_retention": None,
        "invalid_sample_count": None,
        "hard_filter_pass": False,
        "hard_filter_reasons": ["dry_run_not_evaluated"],
        "safety_soft_penalty_reasons": ["dry_run_not_evaluated"],
        "copy_paste_audit": {"enabled": False, "reason": "dry_run_not_evaluated"},
        "policy": policy,
    }
