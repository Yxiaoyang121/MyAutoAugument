from __future__ import annotations

import csv
import json
from pathlib import Path

from AutoAugment.catf_v2.policy_matrix import active_class_ids, initial_policy_matrix
from AutoAugment.catf_v2.sample_router import SampleAwareAugmentationRouter
from scripts.build_preserve_weak_decision_schedule import build_schedule, load_fixed_policy_by_epoch
from scripts.train_yolo_default_with_inloop_feedback import apply_offline_probe_decision_to_policy


def stale_policy() -> dict:
    policy = initial_policy_matrix({4: "class4", 9: "class9", 11: "class11", 12: "class12"})
    for cid in (4, 9, 12):
        class_policy = policy["classes"][str(cid)]
        class_policy["status"] = "active"
        class_policy["state"] = "accepted"
        class_policy["ops"]["local_contrast"]["prob"] = 0.2
        class_policy["ops"]["local_contrast"]["strength"] = 0.3
    return policy


def exact_payload(active_class: str, op_list: str, prob_strength: str) -> dict:
    return {
        "selected_candidate_policy_id": "candidate_policy_preserve_original",
        "selected_candidate_action": "preserve_original",
        "preserve_original": True,
        "epoch_exact_fixed_active_class": active_class,
        "epoch_exact_fixed_op_list": op_list,
        "epoch_exact_fixed_prob_strength": prob_strength,
        "original_fixed_active_class": "4;11;12",
        "original_fixed_op_list": "c4:stale:local_contrast@p=0.2/s=0.3; c12:stale:local_contrast@p=0.2/s=0.3",
        "selected_candidate": {
            "candidate_policy_id": "candidate_policy_preserve_original",
            "probe_set": {
                "class_id": 4,
                "active_classes": [4, 11, 12],
            },
            "decision": {
                "decision": "preserve_original",
                "image_modification_allowed": True,
                "sample_weighting_allowed": False,
            },
        },
    }


def test_preserve_epoch_exact_empty_policy_clears_stale_union(tmp_path) -> None:
    preserved, event = apply_offline_probe_decision_to_policy(
        stale_policy(),
        exact_payload("", "", ""),
        epoch_num=20,
        output_dir=tmp_path,
        disable_sampler_only=True,
    )

    assert event["action"] == "preserve_original"
    assert event["preserve_epoch_exact"] is True
    assert event["preserve_policy_empty_for_epoch"] is True
    assert event["sample_router_allowed"] is False
    assert event["sample_weighting_allowed"] is False
    assert active_class_ids(preserved) == []


def test_preserve_epoch_exact_does_not_use_seed_level_union(tmp_path) -> None:
    payload = exact_payload(
        "12",
        "c12:weak_localization:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02",
        "c12:local_contrast:0.015/0.02; c12:sharpen_mild:0.015/0.02",
    )
    preserved, event = apply_offline_probe_decision_to_policy(
        stale_policy(),
        payload,
        epoch_num=15,
        output_dir=tmp_path,
        disable_sampler_only=True,
    )

    assert event["preserve_expected_active_classes"] == [12]
    assert active_class_ids(preserved) == [12]
    assert preserved["classes"]["12"]["ops"]["local_contrast"]["prob"] == 0.015
    assert preserved["classes"]["12"]["ops"]["local_contrast"]["strength"] == 0.02
    assert preserved["classes"]["4"]["ops"]["local_contrast"]["prob"] == 0.0
    assert preserved["classes"]["9"]["ops"]["local_contrast"]["prob"] == 0.0


def test_preserve_epoch_exact_router_sees_only_current_epoch_class(tmp_path) -> None:
    payload = exact_payload(
        "11",
        "c11:texture_boundary_weak:local_contrast@p=0.015/s=0.02",
        "c11:local_contrast:0.015/0.02",
    )
    preserved, _ = apply_offline_probe_decision_to_policy(
        stale_policy(),
        payload,
        epoch_num=5,
        output_dir=tmp_path,
        disable_sampler_only=True,
    )
    router = SampleAwareAugmentationRouter(preserved, seed=1, num_classes=13, roi_aware=True, sample_aware=True)

    assert active_class_ids(router.policy_matrix) == [11]
    assert 4 not in active_class_ids(router.policy_matrix)
    assert 9 not in active_class_ids(router.policy_matrix)
    assert 12 not in active_class_ids(router.policy_matrix)


def test_build_schedule_uses_epoch_exact_fixed_history(tmp_path) -> None:
    fixed_history = {
        "history": [
            {
                "epoch": 5,
                "accepted_policy": {
                    "classes": {
                        "4": {
                            "class_name": "class4",
                            "dominant_issue": "low_recall",
                            "status": "active",
                            "state": "accepted",
                            "ops": {"local_contrast": {"prob": 0.005, "strength": 0.01}},
                        }
                    }
                },
            },
            {"epoch": 10, "accepted_policy": {"classes": {}}},
        ]
    }
    fixed_path = tmp_path / "policy_history.json"
    fixed_path.write_text(json.dumps(fixed_history), encoding="utf-8")
    replay_path = tmp_path / "records.csv"
    with replay_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "epoch", "final_replay_action", "original_fixed_active_class"])
        writer.writeheader()
        writer.writerow({"seed": "0", "epoch": "5", "final_replay_action": "preserve_original", "original_fixed_active_class": "4;12"})
        writer.writerow({"seed": "0", "epoch": "10", "final_replay_action": "preserve_original", "original_fixed_active_class": "4;12"})

    records = [
        {"seed": "0", "epoch": "5", "final_replay_action": "preserve_original", "original_fixed_active_class": "4;12"},
        {"seed": "0", "epoch": "10", "final_replay_action": "preserve_original", "original_fixed_active_class": "4;12"},
    ]
    fixed_by_epoch = load_fixed_policy_by_epoch(fixed_path)
    schedule = build_schedule(records, seed=0, replay_csv=replay_path, fixed_policy_history=fixed_path, fixed_by_epoch=fixed_by_epoch)

    assert schedule["epoch_exact_preserve_original"] is True
    assert schedule["epoch_decisions"]["5"]["epoch_exact_fixed_active_class"] == "4"
    assert schedule["epoch_decisions"]["10"]["epoch_exact_fixed_active_class"] == ""
    assert schedule["epoch_decisions"]["10"]["epoch_exact_fixed_policy_empty"] is True
    assert schedule["sampler_only_involved"] is False
