from __future__ import annotations

import numpy as np

from AutoAugment.catf_v2.issue_attribution import attribute_class_issues
from AutoAugment.catf_v2.per_class_diagnosis import build_per_class_diagnosis
from AutoAugment.catf_v2.policy_matrix import ClassAwarePolicyMatrix, initial_policy_matrix
from AutoAugment.catf_v2.sample_router import SampleAwareAugmentationRouter
from AutoAugment.catf_v2.threshold_calibration import ThresholdCalibrationAnalyzer


def make_per_class(rows: dict[str, dict]) -> dict:
    return {"epoch": 5, "classes": rows}


def row(
    class_id: int,
    name: str,
    *,
    precision: float = 0.8,
    recall: float = 0.5,
    ap50: float = 0.7,
    ap95: float = 0.4,
    tp: int = 10,
    fp: int = 0,
    fn: int = 10,
    evidence: int = 10,
    low_support: bool = False,
    no_aug: bool = False,
    domain_prior: bool = False,
    high_fp_guarded: bool = False,
) -> dict:
    return {
        "class_id": class_id,
        "class_name": name,
        "train_instances": 100,
        "val_instances": 30 if not low_support else 5,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "AP50": ap50,
        "AP50_95": ap95,
        "FN_rate": fn / max(1, tp + fn),
        "FP_rate": fp / max(1, tp + fp),
        "support_level": "low" if low_support else "medium",
        "low_recall": recall < 0.6,
        "high_fp": fp >= 10,
        "low_precision": precision < 0.55,
        "low_ap50_95": ap95 < 0.45,
        "weak_localization": ap50 - ap95 > 0.18,
        "low_contrast_fn": True,
        "texture_boundary_weak": True,
        "low_support": low_support,
        "no_aug_class": no_aug,
        "no_aug_exception_allowed": False,
        "domain_high_fp_prior": domain_prior,
        "high_fp_guarded": high_fp_guarded or fp >= 10,
        "stable_class": precision >= 0.90 and recall >= 0.90 and ap50 >= 0.90 and fn <= 3 and fp < 10,
        "evidence_count": evidence,
        "low_contrast_fn_count": max(0, fn),
        "diagnosis_confidence": 0.9 if evidence >= 5 else 0.2,
        "strong_update_allowed": bool(not low_support and evidence >= 5 and not no_aug and not high_fp_guarded and fp < 10),
    }


def update_policy(rows: dict[str, dict], class_names: dict[int, str], *, top_k: int = 2) -> dict:
    per_class = make_per_class(rows)
    attribution = attribute_class_issues(per_class)
    matrix = ClassAwarePolicyMatrix(initial_policy_matrix(class_names), top_k=top_k, top_m=2)
    return matrix.update(epoch=5, per_class_diagnosis=per_class, issue_attribution=attribution, metrics={}, reference_metrics={})


def test_ok2_ok3_default_no_aug_in_initial_matrix() -> None:
    policy = initial_policy_matrix({0: "OK2", 1: "OK3", 2: "defect"})

    assert policy["classes"]["0"]["no_aug_class"] is True
    assert policy["classes"]["1"]["no_aug_class"] is True
    assert policy["classes"]["0"]["status"] == "frozen"
    assert policy["classes"]["2"]["no_aug_class"] is False


def test_ok_class_high_recall_low_fn_cannot_activate() -> None:
    diagnosis = {
        "global": {"gt": 200, "localization_weak": 0},
        "quality": {"false_negatives": {"low_contrast_count": 2}},
        "per_class": {
            "1": {"class_id": 1, "class_name": "OK3", "gt": 100, "tp": 98, "fp": 0, "fn": 2, "precision": 1.0, "recall": 0.98},
        },
    }
    per_class = build_per_class_diagnosis(diagnosis, class_names={1: "OK3"}, train_instances={1: 100})
    updated = update_policy(per_class["classes"], {1: "OK3"})

    assert per_class["classes"]["1"]["no_aug_class"] is True
    assert per_class["classes"]["1"]["strong_update_allowed"] is False
    assert updated["classes"]["1"]["status"] == "frozen"
    assert all(op["prob"] == 0 for op in updated["classes"]["1"]["ops"].values())


def test_ok_class_high_fp_becomes_guard_not_enhancement() -> None:
    diagnosis = {
        "global": {"gt": 300, "localization_weak": 0},
        "quality": {"false_negatives": {"low_contrast_count": 2}},
        "per_class": {
            "1": {"class_id": 1, "class_name": "OK3", "gt": 274, "tp": 271, "fp": 49, "fn": 2, "precision": 0.846875, "recall": 0.989},
        },
    }
    per_class = build_per_class_diagnosis(diagnosis, class_names={1: "OK3"}, train_instances={1: 946})
    updated = update_policy(per_class["classes"], {1: "OK3"})

    assert per_class["classes"]["1"]["high_fp_guarded"] is True
    assert updated["classes"]["1"]["status"] == "guarded"
    assert updated["classes"]["1"]["guards"]["high_fp_guarded"] is True
    assert all(op["prob"] == 0 for op in updated["classes"]["1"]["ops"].values())


def test_recovered_defect_clears_stale_high_fp_guard() -> None:
    matrix = ClassAwarePolicyMatrix(initial_policy_matrix({6: "漏背锡"}), top_k=1, top_m=2)
    high_fp_rows = {
        "6": row(6, "漏背锡", precision=0.18, recall=0.65, tp=30, fp=137, fn=7, evidence=145),
    }
    guarded = matrix.update(
        epoch=5,
        per_class_diagnosis=make_per_class(high_fp_rows),
        issue_attribution=attribute_class_issues(make_per_class(high_fp_rows)),
        metrics={},
        reference_metrics={},
    )
    recovered_rows = {
        "6": row(6, "漏背锡", precision=0.81, recall=0.28, tp=13, fp=3, fn=31, evidence=34),
    }
    updated = matrix.update(
        epoch=10,
        per_class_diagnosis=make_per_class(recovered_rows),
        issue_attribution=attribute_class_issues(make_per_class(recovered_rows)),
        metrics={},
        reference_metrics={},
    )

    assert guarded["classes"]["6"]["guards"]["high_fp_guarded"] is True
    assert updated["classes"]["6"]["status"] == "active"
    assert updated["classes"]["6"]["guards"]["high_fp_guarded"] is False
    assert updated["classes"]["6"]["ops"]["sharpen_mild"]["prob"] > 0


def test_stable_class_does_not_enter_active_top_k() -> None:
    rows = {
        "0": row(0, "stable", precision=0.95, recall=0.95, ap50=0.95, ap95=0.7, fn=1, evidence=8),
        "1": row(1, "target", recall=0.2, fn=20, evidence=20),
    }

    updated = update_policy(rows, {0: "stable", 1: "target"})

    assert updated["classes"]["0"]["status"] == "frozen"
    assert updated["classes"]["1"]["status"] == "active"


def test_domain_high_fp_prior_does_not_trigger_strong_photometric_or_threshold_drop() -> None:
    rows = {
        "8": row(8, "脏污", recall=0.0, fn=42, evidence=42, domain_prior=True),
    }
    updated = update_policy(rows, {8: "脏污"})
    threshold = ThresholdCalibrationAnalyzer(default_threshold=0.25).analyze(make_per_class(rows), attribute_class_issues(make_per_class(rows)))

    assert updated["classes"]["8"]["domain_high_fp_prior"] is True
    assert updated["classes"]["8"]["ops"]["gamma"]["prob"] == 0
    assert updated["classes"]["8"]["ops"]["clahe"]["prob"] == 0
    assert updated["classes"]["8"]["ops"]["sharpen_mild"]["prob"] <= 0.005
    assert threshold["classes"]["8"]["recommended_threshold"] == 0.25


def test_top_k_active_classes_is_enforced() -> None:
    rows = {str(i): row(i, f"target{i}", recall=0.1, fn=20, evidence=20) for i in range(4)}

    updated = update_policy(rows, {i: f"target{i}" for i in range(4)}, top_k=2)
    active = [cid for cid, item in updated["classes"].items() if item["status"] == "active"]

    assert len(active) == 2


def test_domain_high_fp_prior_does_not_crowd_out_non_prior_defect() -> None:
    rows = {
        "4": row(4, "油污", recall=0.0, fn=18, evidence=18, domain_prior=True),
        "6": row(6, "漏背锡", precision=0.65, recall=0.2826, ap50=0.65, ap95=0.2826, tp=13, fp=7, fn=32, evidence=39),
        "8": row(8, "脏污", recall=0.0, fn=42, evidence=42, domain_prior=True),
    }

    updated = update_policy(rows, {4: "油污", 6: "漏背锡", 8: "脏污"}, top_k=2)
    active = {cid for cid, item in updated["classes"].items() if item["status"] == "active"}

    assert "6" in active
    assert len(active) == 2


def test_roi_augmentation_skips_no_aug_class() -> None:
    policy = initial_policy_matrix({1: "OK3"})
    policy["classes"]["1"]["ops"]["local_contrast"]["prob"] = 1.0
    policy["classes"]["1"]["ops"]["local_contrast"]["strength"] = 0.3
    router = SampleAwareAugmentationRouter(policy, seed=1, num_classes=2)
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    labels = np.array([1], dtype=np.int64)
    boxes = np.array([[10, 10, 40, 40]], dtype=np.float32)

    result = router.apply(image, labels, boxes)

    assert result.audit["applied_ops"] == []
    assert router.roi_stats.roi_aug_applied == 0


def test_high_fp_guarded_class_does_not_execute_roi_photometric() -> None:
    policy = initial_policy_matrix({4: "油污"})
    policy["classes"]["4"]["status"] = "guarded"
    policy["classes"]["4"]["guards"]["high_fp_guarded"] = True
    policy["classes"]["4"]["ops"]["local_contrast"]["prob"] = 1.0
    policy["classes"]["4"]["ops"]["local_contrast"]["strength"] = 0.3
    router = SampleAwareAugmentationRouter(policy, seed=1, num_classes=5)
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    labels = np.array([4], dtype=np.int64)
    boxes = np.array([[10, 10, 40, 40]], dtype=np.float32)

    result = router.apply(image, labels, boxes)

    assert result.audit["applied_ops"] == []
    assert router.roi_stats.roi_aug_applied == 0


def test_low_support_class_does_not_trigger_strong_photometric() -> None:
    rows = {"3": row(3, "rare", recall=0.0, fn=5, evidence=5, low_support=True)}

    updated = update_policy(rows, {3: "rare"})

    assert updated["classes"]["3"]["status"] == "observe"
    assert all(op["prob"] == 0 for op in updated["classes"]["3"]["ops"].values())


def test_low_recall_low_contrast_defect_can_still_activate() -> None:
    rows = {
        "6": row(6, "漏背锡", precision=0.65, recall=0.2826, ap50=0.65, ap95=0.2826, tp=13, fp=7, fn=32, evidence=39),
    }

    updated = update_policy(rows, {6: "漏背锡"})

    assert updated["classes"]["6"]["status"] == "active"
    assert updated["classes"]["6"]["ops"]["sharpen_mild"]["prob"] > 0
    assert updated["classes"]["6"]["ops"]["local_contrast"]["prob"] > 0
