from __future__ import annotations

import json

from AutoAugment.feedback_policy_controller import FeedbackPolicyController, default_catf_policy


REFERENCE = {"precision": 0.70, "recall": 0.70, "map50": 0.72, "map50_95": 0.50}


def op(policy: dict, name: str) -> dict:
    return next(item for item in policy["operations"] if item["name"] == name)


def test_feedback_low_contrast_fn_high_increases_safe_texture_first(tmp_path) -> None:
    controller = FeedbackPolicyController(default_catf_policy(), history_dir=tmp_path, policy_state_path=tmp_path / "policy_state.json")

    updated = controller.update(
        {"low_contrast_fn_high": True},
        epoch=5,
        metrics={"precision": 0.705, "recall": 0.69, "map50": 0.72, "map50_95": 0.50},
        reference_metrics=REFERENCE,
    )

    assert op(updated, "sharpen_mild")["prob"] > 0.08
    assert op(updated, "local_contrast")["prob"] > 0.06
    assert op(updated, "brightness")["prob"] == 0.03
    assert (tmp_path / "policy_history.json").exists()
    assert (tmp_path / "policy_history.md").exists()
    assert (tmp_path / "policy_history.csv").exists()
    history = json.loads((tmp_path / "policy_history.json").read_text(encoding="utf-8"))
    assert history["history"][0]["action"] == "accept"
    assert "reference_metrics" in history["history"][0]


def test_feedback_fp_high_decreases_aggressive_photometric(tmp_path) -> None:
    controller = FeedbackPolicyController(default_catf_policy(), history_dir=tmp_path)

    updated = controller.update(
        {"fp_high": True},
        epoch=5,
        metrics={"precision": 0.689, "recall": 0.72, "map50": 0.72, "map50_95": 0.50},
        reference_metrics=REFERENCE,
    )

    assert op(updated, "brightness")["prob"] < 0.03
    assert op(updated, "contrast")["strength"] < 0.15
    assert op(updated, "cutout_safe")["prob"] < 0.03
    assert "precision_guard" in controller.history[-1]["guard_triggered"]
