from __future__ import annotations

import json
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

from AutoAugment.formats.yolo import save_yolo_labels
from AutoAugment.policies import OperationSpace, SearchSpace
import AutoAugment.search.random_search as random_search_module
from AutoAugment.search import RandomSearch
from AutoAugment.search.proxy_metrics import (
    apply_proxy_hard_filter,
    bbox_retention_raw,
    compute_proxy_score,
    select_proxy_candidate,
    yolo_bbox_safe_mask,
)


def fresh_dir(name: str) -> Path:
    path = Path("outputs/tests/pytest_tmp") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def create_proxy_dataset(root: Path) -> None:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    specs = [
        ("sample_0", 0, np.asarray([[18, 18, 42, 42]], dtype=np.float32)),
        ("sample_1", 1, np.asarray([[2, 2, 8, 8]], dtype=np.float32)),
        ("sample_2", 2, np.asarray([[1, 24, 9, 32]], dtype=np.float32)),
        ("sample_3", 2, np.asarray([[28, 28, 44, 44]], dtype=np.float32)),
    ]
    for index, (stem, class_id, boxes) in enumerate(specs):
        image = np.full((64, 64, 3), 80 + index * 20, dtype=np.uint8)
        for box in boxes.astype(int):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (180, 180, 180), -1)
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), image)
        save_yolo_labels(
            labels_dir / f"{stem}.txt",
            np.asarray([class_id], dtype=np.int64),
            boxes,
            64,
            64,
        )


def base_metrics(**overrides):
    metrics = {
        "bbox_safe_rate": 1.0,
        "bbox_valid_rate": 1.0,
        "bbox_retention_raw": 1.0,
        "small_target_retention": 1.0,
        "tiny_target_retention": 1.0,
        "edge_target_retention": 1.0,
        "class_coverage_after": 1.0,
        "rare_class_present": True,
        "rare_class_retention": 1.0,
        "exposure_diversity_score": 1.0,
        "strength_penalty": 0.1,
    }
    metrics.update(overrides)
    return metrics


def test_bbox_retention_raw() -> None:
    assert bbox_retention_raw(100, 100) == 1.0
    assert bbox_retention_raw(100, 80) == 0.8
    assert bbox_retention_raw(0, 0) == 1.0


def test_bbox_safe_rate_mask_rejects_boxes_crossing_image_boundary() -> None:
    boxes = np.asarray(
        [
            [0.5, 0.5, 0.2, 0.2],
            [0.98, 0.5, 0.1, 0.2],
        ],
        dtype=np.float32,
    )
    mask = yolo_bbox_safe_mask(boxes)
    assert mask.tolist() == [True, False]
    assert float(mask.mean()) == 0.5


def test_hard_filter_rejects_low_retention_and_tiny_retention() -> None:
    ok, reasons = apply_proxy_hard_filter(base_metrics(bbox_retention_raw=0.91))
    assert ok is False
    assert any("bbox_retention_raw" in reason for reason in reasons)

    ok, reasons = apply_proxy_hard_filter(base_metrics(tiny_target_retention=0.80))
    assert ok is False
    assert any("tiny_target_retention" in reason for reason in reasons)


def test_hard_filter_ignores_rare_class_retention_when_no_rare_class_present() -> None:
    ok, reasons = apply_proxy_hard_filter(
        base_metrics(rare_class_present=False, rare_class_retention=0.0)
    )
    assert ok is True
    assert not any("rare_class_retention" in reason for reason in reasons)


def test_proxy_score_decreases_when_strength_penalty_increases() -> None:
    high_score, _, missing = compute_proxy_score(base_metrics(strength_penalty=0.0))
    low_score, _, missing_low = compute_proxy_score(base_metrics(strength_penalty=1.0))
    assert missing == []
    assert missing_low == []
    assert 0.0 <= low_score < high_score <= 1.0


def test_select_proxy_candidate_prefers_passing_highest_score() -> None:
    selection = select_proxy_candidate(
        [
            {"candidate_index": 0, "hard_filter_pass": False, "hard_filter_reasons": ["x"], "proxy_score": 0.99},
            {"candidate_index": 1, "hard_filter_pass": True, "hard_filter_reasons": [], "proxy_score": 0.70},
            {"candidate_index": 2, "hard_filter_pass": True, "hard_filter_reasons": [], "proxy_score": 0.80},
        ]
    )
    assert selection.selected_index == 2
    assert selection.fallback_used is False
    assert selection.fallback_reason is None


def test_select_proxy_candidate_fallback_uses_fewest_violations_then_score() -> None:
    selection = select_proxy_candidate(
        [
            {"candidate_index": 0, "hard_filter_pass": False, "hard_filter_reasons": ["a", "b"], "proxy_score": 0.99},
            {"candidate_index": 1, "hard_filter_pass": False, "hard_filter_reasons": ["a"], "proxy_score": 0.60},
            {"candidate_index": 2, "hard_filter_pass": False, "hard_filter_reasons": ["a"], "proxy_score": 0.80},
        ]
    )
    assert selection.selected_index == 2
    assert selection.fallback_used is True
    assert selection.fallback_reason == "no_candidate_passed_hard_filter"


def test_random_search_default_records_proxy_prefilter_disabled() -> None:
    work_dir = fresh_dir("proxy_prefilter_default")
    dataset_root = work_dir / "dataset"
    output_dir = work_dir / "search"
    create_proxy_dataset(dataset_root)
    search = RandomSearch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        num_trials=1,
        num_samples=2,
        seed=7,
        search_space=SearchSpace(
            [OperationSpace("brightness", (0.0, 0.0), (0.1, 0.1), {"max_delta": 0.1})],
            operation_count_range=(1, 1),
        ),
    )
    search.run()
    trial_record = json.loads((output_dir / "trials" / "trial_000" / "trial_record.json").read_text(encoding="utf-8"))
    assert trial_record["proxy_prefilter"] == {"enabled": False}
    assert not (output_dir / "trials" / "trial_000" / "proxy_candidates_summary.csv").exists()


def test_proxy_prefilter_writes_candidate_records_and_trial_record() -> None:
    work_dir = fresh_dir("proxy_prefilter_records")
    dataset_root = work_dir / "dataset"
    output_dir = work_dir / "search"
    create_proxy_dataset(dataset_root)
    search = RandomSearch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        num_trials=1,
        num_samples=4,
        seed=9,
        search_space=SearchSpace(
            [OperationSpace("brightness", (0.0, 0.0), (0.1, 0.1), {"max_delta": 0.1})],
            operation_count_range=(1, 1),
        ),
        proxy_prefilter={
            "enabled": True,
            "candidate_policies": 3,
            "proxy_eval_samples": 3,
            "proxy_top_k": 1,
            "proxy_score_version": "dataset2_v1",
            "hard_filter_profile": "dataset2_v1",
        },
    )
    results = search.run()
    trial_dir = output_dir / "trials" / "trial_000"
    assert len(results) == 1
    assert (trial_dir / "proxy_candidates_summary.csv").exists()
    assert (trial_dir / "selected_policy_by_proxy.json").exists()
    assert (trial_dir / "proxy_candidates" / "candidate_000" / "policy.json").exists()
    assert (trial_dir / "proxy_candidates" / "candidate_000" / "proxy_metrics.json").exists()
    assert not (trial_dir / "proxy_candidates" / "candidate_000" / "images").exists()
    assert not (trial_dir / "proxy_candidates" / "candidate_000" / "labels").exists()
    assert not (trial_dir / "proxy_tmp").exists()
    selected = json.loads((trial_dir / "selected_policy_by_proxy.json").read_text(encoding="utf-8"))
    assert selected["selected_candidate_index"] in {0, 1, 2}
    assert selected["artifact_mode"] == "metrics_only"
    assert selected["candidate_image_artifacts_kept"] is False
    trial_record = json.loads((trial_dir / "trial_record.json").read_text(encoding="utf-8"))
    assert trial_record["proxy_prefilter"]["enabled"] is True
    assert trial_record["proxy_prefilter"]["candidate_policies"] == 3
    assert trial_record["proxy_prefilter"]["artifact_mode"] == "metrics_only"
    assert trial_record["proxy_prefilter"]["keep_candidate_artifacts"] is False
    assert trial_record["proxy_prefilter"]["proxy_tmp_cleaned"] is True
    assert trial_record["proxy_prefilter"]["candidate_image_artifacts_kept"] is False
    assert trial_record["proxy_prefilter"]["candidate_artifact_bytes"] > 0
    assert "selected_candidate_index" in trial_record["proxy_prefilter"]


def test_proxy_prefilter_all_artifact_mode_requires_explicit_keep() -> None:
    work_dir = fresh_dir("proxy_prefilter_artifacts_all")
    dataset_root = work_dir / "dataset"
    output_dir = work_dir / "search"
    create_proxy_dataset(dataset_root)
    search = RandomSearch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        num_trials=1,
        num_samples=4,
        seed=11,
        search_space=SearchSpace(
            [OperationSpace("brightness", (0.0, 0.0), (0.1, 0.1), {"max_delta": 0.1})],
            operation_count_range=(1, 1),
        ),
        proxy_prefilter={
            "enabled": True,
            "candidate_policies": 1,
            "proxy_eval_samples": 2,
            "proxy_top_k": 1,
            "proxy_score_version": "dataset2_v1",
            "hard_filter_profile": "dataset2_v1",
            "keep_candidate_artifacts": True,
            "artifact_mode": "all",
            "max_proxy_artifact_gb": 1.0,
        },
    )
    search.run()
    trial_dir = output_dir / "trials" / "trial_000"
    candidate_dir = trial_dir / "proxy_candidates" / "candidate_000"
    assert (candidate_dir / "images").exists()
    assert (candidate_dir / "labels").exists()
    assert not (trial_dir / "proxy_tmp").exists()
    trial_record = json.loads((trial_dir / "trial_record.json").read_text(encoding="utf-8"))
    assert trial_record["proxy_prefilter"]["artifact_mode"] == "all"
    assert trial_record["proxy_prefilter"]["keep_candidate_artifacts"] is True
    assert trial_record["proxy_prefilter"]["candidate_image_artifacts_kept"] is True


def test_proxy_prefilter_cleans_proxy_tmp_when_candidate_evaluation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = fresh_dir("proxy_prefilter_failure_cleanup")
    dataset_root = work_dir / "dataset"
    output_dir = work_dir / "search"
    create_proxy_dataset(dataset_root)

    def fail_evaluate(*args, **kwargs):
        raise RuntimeError("forced proxy failure")

    monkeypatch.setattr(random_search_module.ProxyEvaluator, "evaluate", fail_evaluate)
    search = RandomSearch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        num_trials=1,
        num_samples=4,
        seed=13,
        search_space=SearchSpace(
            [OperationSpace("brightness", (0.0, 0.0), (0.1, 0.1), {"max_delta": 0.1})],
            operation_count_range=(1, 1),
        ),
        proxy_prefilter={
            "enabled": True,
            "candidate_policies": 1,
            "proxy_eval_samples": 2,
            "proxy_top_k": 1,
            "proxy_score_version": "dataset2_v1",
            "hard_filter_profile": "dataset2_v1",
        },
    )
    with pytest.raises(RuntimeError, match="forced proxy failure"):
        search.run()
    assert not (output_dir / "trials" / "trial_000" / "proxy_tmp" / "candidate_000").exists()
