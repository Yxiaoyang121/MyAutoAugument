from __future__ import annotations

from AutoAugment.catf_v2.issue_attribution import attribute_class_issues
from AutoAugment.catf_v2.policy_matrix import ClassAwarePolicyMatrix, initial_policy_matrix


def diagnosis_rows() -> dict:
    return {
        "epoch": 5,
        "classes": {
            "0": {
                "class_id": 0,
                "class_name": "low contrast",
                "Precision": 0.80,
                "Recall": 0.45,
                "AP50": 0.70,
                "AP50_95": 0.45,
                "FN_rate": 0.55,
                "FP_rate": 0.05,
                "FN": 12,
                "FP": 1,
                "low_contrast_fn": True,
                "low_contrast_fn_count": 8,
                "low_recall": True,
                "texture_boundary_weak": False,
                "weak_localization": False,
                "high_fp": False,
                "low_support": False,
                "stable_class": False,
                "evidence_count": 13,
                "diagnosis_confidence": 0.9,
                "strong_update_allowed": True,
            },
            "1": {
                "class_id": 1,
                "class_name": "scratch",
                "Precision": 0.82,
                "Recall": 0.70,
                "AP50": 0.80,
                "AP50_95": 0.50,
                "FN_rate": 0.20,
                "FP_rate": 0.04,
                "FN": 4,
                "FP": 1,
                "low_contrast_fn": False,
                "texture_boundary_weak": True,
                "weak_localization": True,
                "high_fp": False,
                "low_support": False,
                "stable_class": False,
                "evidence_count": 8,
                "diagnosis_confidence": 0.8,
                "strong_update_allowed": True,
            },
            "2": {
                "class_id": 2,
                "class_name": "fp class",
                "Precision": 0.30,
                "Recall": 0.80,
                "AP50": 0.60,
                "AP50_95": 0.40,
                "FN_rate": 0.05,
                "FP_rate": 0.70,
                "FN": 1,
                "FP": 20,
                "low_contrast_fn": False,
                "texture_boundary_weak": False,
                "weak_localization": False,
                "high_fp": True,
                "low_support": False,
                "stable_class": False,
                "evidence_count": 21,
                "diagnosis_confidence": 0.9,
                "strong_update_allowed": True,
            },
            "3": {
                "class_id": 3,
                "class_name": "stable",
                "Precision": 0.90,
                "Recall": 0.90,
                "AP50": 0.90,
                "AP50_95": 0.70,
                "FN_rate": 0.02,
                "FP_rate": 0.02,
                "FN": 0,
                "FP": 0,
                "low_contrast_fn": False,
                "texture_boundary_weak": False,
                "weak_localization": False,
                "high_fp": False,
                "low_support": False,
                "stable_class": True,
                "evidence_count": 0,
                "diagnosis_confidence": 0.9,
                "strong_update_allowed": False,
            },
        },
    }


def test_policy_matrix_trust_region_and_top_k() -> None:
    per_class = diagnosis_rows()
    attribution = attribute_class_issues(per_class)
    matrix = ClassAwarePolicyMatrix(initial_policy_matrix({0: "low", 1: "scratch", 2: "fp", 3: "stable"}), top_k=2, top_m=2)

    updated = matrix.update(epoch=5, per_class_diagnosis=per_class, issue_attribution=attribution, metrics={}, reference_metrics={})

    active = [cid for cid, row in updated["classes"].items() if row["status"] == "active"]
    assert len(active) <= 2
    for record in matrix.history[-1]["adjustments"]:
        if record["field"] == "prob":
            assert abs(record["after"] - record["before"]) <= 0.0150001
        if record["field"] == "strength":
            assert abs(record["after"] - record["before"]) <= 0.0250001


def test_low_contrast_class_activates_only_target_class_ops() -> None:
    per_class = diagnosis_rows()
    per_class["classes"] = {"0": per_class["classes"]["0"]}
    attribution = attribute_class_issues(per_class)
    matrix = ClassAwarePolicyMatrix(initial_policy_matrix({0: "low", 1: "stable"}), top_k=3, top_m=2)

    updated = matrix.update(epoch=5, per_class_diagnosis=per_class, issue_attribution=attribution, metrics={}, reference_metrics={})

    assert updated["classes"]["0"]["ops"]["local_contrast"]["prob"] > 0
    assert updated["classes"]["1"]["ops"]["local_contrast"]["prob"] == 0


def test_texture_class_prefers_sharpen_and_local_contrast() -> None:
    per_class = {"classes": {"1": diagnosis_rows()["classes"]["1"]}}
    attribution = attribute_class_issues(per_class)
    matrix = ClassAwarePolicyMatrix(initial_policy_matrix({1: "scratch"}), top_k=3, top_m=2)

    updated = matrix.update(epoch=5, per_class_diagnosis=per_class, issue_attribution=attribution, metrics={}, reference_metrics={})

    assert updated["classes"]["1"]["ops"]["sharpen_mild"]["prob"] > 0
    assert updated["classes"]["1"]["ops"]["local_contrast"]["prob"] > 0


def test_high_fp_class_triggers_photometric_guard() -> None:
    per_class = {"classes": {"2": diagnosis_rows()["classes"]["2"]}}
    attribution = attribute_class_issues(per_class)
    policy = initial_policy_matrix({2: "fp"})
    policy["classes"]["2"]["ops"]["gamma"]["prob"] = 0.04
    matrix = ClassAwarePolicyMatrix(policy, top_k=3, top_m=2)

    updated = matrix.update(epoch=5, per_class_diagnosis=per_class, issue_attribution=attribution, metrics={}, reference_metrics={})

    assert updated["classes"]["2"]["guards"]["high_fp_guarded"] is True
    assert updated["classes"]["2"]["ops"]["gamma"]["prob"] < 0.04


def test_stable_class_freezes() -> None:
    per_class = {"classes": {"3": diagnosis_rows()["classes"]["3"]}}
    attribution = attribute_class_issues(per_class)
    matrix = ClassAwarePolicyMatrix(initial_policy_matrix({3: "stable"}), top_k=3, top_m=2)

    updated = matrix.update(epoch=5, per_class_diagnosis=per_class, issue_attribution=attribution, metrics={}, reference_metrics={})

    assert updated["classes"]["3"]["status"] == "frozen"
