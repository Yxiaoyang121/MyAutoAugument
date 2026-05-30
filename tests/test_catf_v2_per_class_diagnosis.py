from __future__ import annotations

from AutoAugment.catf_v2.per_class_diagnosis import build_per_class_diagnosis


def test_per_class_diagnosis_outputs_required_fields() -> None:
    diagnosis = {
        "global": {"gt": 30, "localization_weak": 3},
        "quality": {"false_negatives": {"low_contrast_count": 4, "dark_count": 2}},
        "per_class": {
            "0": {"class_id": 0, "class_name": "low contrast", "gt": 20, "tp": 10, "fp": 2, "fn": 10, "precision": 0.83, "recall": 0.50},
        },
    }

    payload = build_per_class_diagnosis(diagnosis, class_names={0: "low contrast"}, train_instances={0: 100}, epoch=5)
    row = payload["classes"]["0"]

    for key in [
        "class_id",
        "class_name",
        "train_instances",
        "val_instances",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "AP50",
        "AP50_95",
        "FN_rate",
        "FP_rate",
        "support_level",
        "low_contrast_fn",
        "stable_class",
        "diagnosis_confidence",
    ]:
        assert key in row
    assert row["low_recall"] is True
    assert row["strong_update_allowed"] is True


def test_low_support_class_disallows_strong_update() -> None:
    diagnosis = {"per_class": {"2": {"gt": 4, "tp": 1, "fp": 0, "fn": 3, "precision": 1.0, "recall": 0.25}}}

    payload = build_per_class_diagnosis(diagnosis, class_names={2: "rare"}, train_instances={2: 5}, epoch=5)
    row = payload["classes"]["2"]

    assert row["support_level"] == "low"
    assert row["low_support_class"] is True
    assert row["strong_update_allowed"] is False
