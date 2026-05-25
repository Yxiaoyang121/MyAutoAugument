from __future__ import annotations

import json

from AutoAugment.feedback_policy_controller import FeedbackPolicyController


def make_feedback_policy() -> dict:
    return {
        "policy_id": "test_feedback_policy",
        "operations": [
            {"name": "clahe", "prob": 0.20, "min_prob": 0.0, "max_prob": 0.8, "strength": 0.20, "min_strength": 0.0, "max_strength": 0.8},
            {"name": "gamma", "prob": 0.20, "min_prob": 0.0, "max_prob": 0.8, "strength": 0.20, "min_strength": 0.0, "max_strength": 0.8},
            {"name": "brightness", "prob": 0.40, "min_prob": 0.0, "max_prob": 0.8, "strength": 0.40, "min_strength": 0.0, "max_strength": 0.8},
            {"name": "contrast", "prob": 0.40, "min_prob": 0.0, "max_prob": 0.8, "strength": 0.40, "min_strength": 0.0, "max_strength": 0.8},
            {"name": "cutout_safe", "prob": 0.10, "min_prob": 0.0, "max_prob": 0.6, "strength": 0.20, "min_strength": 0.0, "max_strength": 0.6},
            {"name": "copy_paste", "prob": 0.0, "min_prob": 0.0, "max_prob": 0.4, "strength": 0.0, "status": "pending_object_bank_design"},
        ],
    }


def op(policy: dict, name: str) -> dict:
    return next(item for item in policy["operations"] if item["name"] == name)


def test_feedback_low_contrast_fn_high_increases_photometric(tmp_path) -> None:
    controller = FeedbackPolicyController(make_feedback_policy(), history_dir=tmp_path, policy_state_path=tmp_path / "policy_state.json")

    updated = controller.update({"low_contrast_fn_high": True}, stage_index=0, metrics={"precision": 0.80, "recall": 0.65})

    assert op(updated, "clahe")["prob"] > 0.20
    assert op(updated, "gamma")["prob"] > 0.20
    assert op(updated, "brightness")["strength"] > 0.40
    assert (tmp_path / "policy_history.json").exists()
    assert (tmp_path / "policy_history.md").exists()
    assert (tmp_path / "policy_history.csv").exists()
    history = json.loads((tmp_path / "policy_history.json").read_text(encoding="utf-8"))
    assert len(history["history"]) == 1


def test_feedback_fp_high_decreases_aggressive_photometric(tmp_path) -> None:
    controller = FeedbackPolicyController(make_feedback_policy(), history_dir=tmp_path)

    updated = controller.update({"fp_high": True}, stage_index=1, metrics={"precision": 0.60, "recall": 0.80})

    assert op(updated, "brightness")["prob"] < 0.40
    assert op(updated, "contrast")["strength"] < 0.40
    assert op(updated, "cutout_safe")["prob"] > 0.10
    assert op(updated, "copy_paste")["prob"] == 0.0
