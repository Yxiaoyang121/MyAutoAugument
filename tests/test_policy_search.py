from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import cv2
import numpy as np

from AutoAugment.diagnostics import generate_augmentation_advice
from AutoAugment.search.adaptive import DiagnosticPolicyUpdater, diagnose_trial_result
from AutoAugment.formats.yolo import save_yolo_labels
from AutoAugment.policies import default_detection_search_space
from AutoAugment.search import RandomSearch
from AutoAugment.utils import default_run_output_dir
from examples.run_policy_search import FINAL_FULL_DATASET_NOT_IMPLEMENTED, ensure_final_stage_supported


def fresh_dir(name: str) -> Path:
    path = Path("outputs/tests/pytest_tmp") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def create_tiny_yolo_dataset(root) -> None:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for index in range(3):
        image = np.full((64, 64, 3), 80 + index * 20, dtype=np.uint8)
        cv2.rectangle(image, (16, 18), (42, 44), (180, 180, 180), -1)
        image_path = images_dir / f"sample_{index}.jpg"
        cv2.imwrite(str(image_path), image)
        labels = np.asarray([0], dtype=np.int64)
        bboxes = np.asarray([[16, 18, 42, 44]], dtype=np.float32)
        save_yolo_labels(labels_dir / f"sample_{index}.txt", labels, bboxes, 64, 64)


def test_random_search_runs_two_trials() -> None:
    work_dir = fresh_dir("policy_search")
    dataset_root = work_dir / "dataset"
    output_dir = work_dir / "search"
    create_tiny_yolo_dataset(dataset_root)
    space = default_detection_search_space(operation_count_range=(2, 2))
    search = RandomSearch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        num_trials=2,
        num_samples=2,
        seed=10,
        search_space=space,
    )
    results = search.run()
    assert len(results) == 2
    assert (output_dir / "best_policy.json").exists()
    assert (output_dir / "trials.csv").exists()
    assert (output_dir / "trials.json").exists()
    assert (output_dir / "trials" / "trial_000" / "images").exists()
    assert (output_dir / "trials" / "trial_000" / "labels").exists()
    assert (output_dir / "trials" / "trial_000" / "metrics.json").exists()


def test_adaptive_search_records_before_after_diff_and_reason() -> None:
    work_dir = fresh_dir("adaptive_policy_search")
    dataset_root = work_dir / "dataset"
    output_dir = work_dir / "search"
    create_tiny_yolo_dataset(dataset_root)
    context = {
        "trial_error_summary": {
            "overall": {"gt_count": 10, "tp_count": 7, "fp_count": 2, "fn_count": 1, "precision": 0.78, "recall": 0.7},
            "by_size": {},
            "by_class": [],
            "by_position": {},
            "quality": {"false_negatives": {"count": 1}, "false_positives": {"count": 2, "dark_rate": 0.5}},
        }
    }
    search = RandomSearch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        num_trials=2,
        num_samples=2,
        seed=11,
        search_space=default_detection_search_space(operation_count_range=(2, 2)),
        adaptive_policy=True,
        context=context,
    )
    results = search.run()
    first = results[0]
    assert first.before_policy is not None
    assert first.after_policy is not None
    assert first.diagnosis
    assert first.adjust_reason
    assert first.policy_diff is not None
    assert first.accepted is True
    assert "search_space_weight_delta" in first.policy_diff
    assert (output_dir / "policy_history.jsonl").exists()
    assert (output_dir / "policy_history.csv").exists()
    assert (output_dir / "policy_history.md").exists()
    trial_record = (output_dir / "trials" / "trial_000" / "trial_record.json").read_text(encoding="utf-8")
    assert "before_policy" in trial_record
    assert "after_policy" in trial_record
    assert "policy_diff" in trial_record


def test_diagnosis_updater_changes_search_space_when_metrics_have_no_direct_issues() -> None:
    space = default_detection_search_space(operation_count_range=(2, 2))
    updater = DiagnosticPolicyUpdater(space)
    rng = np.random.default_rng(9)
    policy = updater.sample_policy(rng, name="policy_trial_000")
    metrics = {"yolo_map50": 0.995, "yolo_map50_95": 0.659, "precision": 0.904, "recall": 1.0}
    diagnosis = diagnose_trial_result(metrics)
    adjustment = updater.update(
        evaluated_policy=policy,
        metrics=metrics,
        diagnosis=diagnosis,
        accepted=True,
        rng=rng,
        next_policy_name="policy_trial_001",
    )
    assert diagnosis["main_issues"]
    before_weights = adjustment.before_search_space.operation_weights
    after_weights = adjustment.after_search_space.operation_weights
    assert any(abs(float(after_weights.get(name, 1.0)) - float(before_weights.get(name, 1.0))) > 1e-9 for name in after_weights)
    assert adjustment.policy_diff["search_space_weight_delta"]


def test_diagnostic_policy_diff_records_weights_prob_and_strength_reasons() -> None:
    space = default_detection_search_space(operation_count_range=(2, 2))
    updater = DiagnosticPolicyUpdater(space)
    rng = np.random.default_rng(12)
    policy = updater.sample_policy(rng, name="policy_trial_000")
    diagnosis = generate_augmentation_advice(
        {
            "overall": {"gt_count": 10, "tp_count": 2, "fp_count": 0, "fn_count": 8},
            "by_size": {},
            "by_class": [],
            "by_position": {"edge": {"gt_count": 4, "fn_count": 4, "fn_rate": 1.0}},
            "quality": {
                "false_negatives": {"count": 8, "dark_rate": 0.5, "bright_rate": 0.0, "low_contrast_rate": 0.0},
                "false_positives": {"count": 0},
            },
        }
    )
    adjustment = updater.update(
        evaluated_policy=policy,
        metrics={"final_score": 0.1},
        diagnosis=diagnosis,
        accepted=True,
        rng=rng,
        next_policy_name="policy_trial_001",
    )
    diff = adjustment.policy_diff
    assert "operation_weight_changes" in diff
    assert diff["operation_weight_changes"]["translate"]["before"] == 1.0
    assert diff["operation_weight_changes"]["translate"]["after"] > 1.0
    assert "edge_object_fn_high" in diff["operation_weight_changes"]["translate"]["reason"]
    assert "prob_range_changes" in diff
    assert diff["prob_range_changes"]["translate"]["before"] != diff["prob_range_changes"]["translate"]["after"]
    assert "edge_object_fn_high" in diff["prob_range_changes"]["translate"]["reason"]
    assert "strength_range_changes" in diff
    assert diff["strength_range_changes"]["brightness"]["before"] != diff["strength_range_changes"]["brightness"]["after"]
    assert "exposure_fn_high" in diff["strength_range_changes"]["brightness"]["reason"]
    assert "operation_weight_unchanged" in diff
    assert "prob_range_unchanged" in diff
    assert "strength_range_unchanged" in diff


def test_default_policy_search_output_dir_is_under_runs() -> None:
    output_dir = default_run_output_dir("policy_search", run_name="unit_test_run")
    assert output_dir == Path("outputs/runs/policy_search/unit_test_run")


def test_final_full_dataset_explicitly_fails_until_implemented() -> None:
    args = argparse.Namespace(final_full_dataset=True)
    try:
        ensure_final_stage_supported(args)
    except NotImplementedError as exc:
        assert str(exc) == FINAL_FULL_DATASET_NOT_IMPLEMENTED
    else:
        raise AssertionError("expected final_full_dataset not implemented error")
