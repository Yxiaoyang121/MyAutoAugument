from __future__ import annotations

from AutoAugment.catf_v2.threshold_calibration import ThresholdCalibrationAnalyzer


def test_threshold_calibration_raises_high_fp_class_threshold() -> None:
    diagnosis = {
        "classes": {
            "0": {"class_id": 0, "class_name": "fp", "high_fp": True, "low_recall": False},
        }
    }

    payload = ThresholdCalibrationAnalyzer(default_threshold=0.25).analyze(diagnosis)

    assert payload["classes"]["0"]["recommended_threshold"] > 0.25


def test_threshold_calibration_lowers_low_recall_low_fp_threshold() -> None:
    diagnosis = {
        "classes": {
            "1": {"class_id": 1, "class_name": "missed", "high_fp": False, "low_recall": True},
        }
    }

    payload = ThresholdCalibrationAnalyzer(default_threshold=0.25).analyze(diagnosis)

    assert payload["classes"]["1"]["recommended_threshold"] < 0.25
