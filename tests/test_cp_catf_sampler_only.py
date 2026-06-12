from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from AutoAugment.catf_v2.causal_probe import candidate_policy_catalog
from AutoAugment.catf_v2.policy_matrix import initial_policy_matrix
from AutoAugment.catf_v2.sample_router import SampleAwareAugmentationRouter
from AutoAugment.catf_v2.sampler_only import build_sampler_only_artifacts
from scripts.train_yolo_default_with_inloop_feedback import (
    activate_sampler_only_weighted_index_list,
    apply_offline_probe_decision_to_policy,
)


def _write_dataset(root: Path) -> Path:
    images = root / "images" / "train_core"
    labels = root / "labels" / "train_core"
    final_val_images = root / "images" / "val"
    final_val_labels = root / "labels" / "val"
    for directory in (images, labels, final_val_images, final_val_labels):
        directory.mkdir(parents=True, exist_ok=True)
    label_rows = {
        "img0": "4 0.5 0.5 0.2 0.2\n",
        "img1": "6 0.5 0.5 0.2 0.2\n",
        "img2": "0 0.5 0.5 0.2 0.2\n",
        "img3": "4 0.5 0.5 0.2 0.2\n0 0.4 0.4 0.1 0.1\n",
        "img4": "2 0.5 0.5 0.2 0.2\n",
    }
    for stem, label_text in label_rows.items():
        (images / f"{stem}.jpg").write_bytes(b"")
        (labels / f"{stem}.txt").write_text(label_text, encoding="utf-8")
    (final_val_images / "val0.jpg").write_bytes(b"")
    (final_val_labels / "val0.txt").write_text("12 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train_core",
                "val: images/val",
                "nc: 7",
                "names:",
                "  0: OK2",
                "  1: OK3",
                "  2: low_support_defect",
                "  3: unused",
                "  4: oil",
                "  5: unused2",
                "  6: high_fp_defect",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def _decision_payload() -> dict:
    return {
        "epoch": 5,
        "paper_probe_mode": True,
        "policy_selection_data": "probe.yaml",
        "final_val_used_for_policy_selection": False,
        "selected_candidate_policy_id": "candidate_policy_3_sampler_only",
        "selected_candidate_action": "sampler_only",
        "sampler_only_focus_rows": [
            {
                "class_id": 4,
                "class_name": "oil",
                "dominant_issue": "low_recall",
                "low_recall": True,
                "strong_update_allowed": True,
                "evidence_count": 8,
                "diagnosis_confidence": 0.7,
            },
            {
                "class_id": 2,
                "class_name": "low_support_defect",
                "dominant_issue": "low_support",
                "low_support": True,
                "strong_update_allowed": False,
                "evidence_count": 3,
                "diagnosis_confidence": 0.2,
            },
        ],
        "sampler_only_context_rows": [
            {"class_id": 0, "class_name": "OK2", "dominant_issue": "stable_class", "no_aug_class": True},
            {"class_id": 4, "class_name": "oil", "dominant_issue": "low_recall", "low_recall": True, "strong_update_allowed": True},
            {"class_id": 2, "class_name": "low_support_defect", "dominant_issue": "low_support", "low_support": True},
            {"class_id": 6, "class_name": "high_fp_defect", "dominant_issue": "high_fp", "high_fp_guarded": True},
        ],
    }


def _sampler_policy_payload() -> dict:
    candidate = candidate_policy_catalog()["candidate_policy_3_sampler_only"]
    return {
        "selected_candidate_policy_id": "candidate_policy_3_sampler_only",
        "selected_candidate_action": "sampler_only",
        "paper_probe_mode": True,
        "policy_selection_source": "probe_split",
        "final_val_used_for_policy_selection": False,
        "forbid_final_val_policy_selection": True,
        "selected_candidate": {
            "candidate_policy_id": "candidate_policy_3_sampler_only",
            "candidate_policy": candidate,
            "probe_set": {"class_id": 4},
            "decision": {
                "decision": "sampler_only",
                "image_modification_allowed": False,
                "sample_weighting_allowed": True,
                "sample_weighting_effective": False,
                "sample_weighting_status": "pending_dataloader_support",
            },
        },
    }


def test_sample_weight_map_uses_probe_rows_and_not_final_val(tmp_path: Path) -> None:
    data_yaml = _write_dataset(tmp_path / "dataset")

    artifacts = build_sampler_only_artifacts(
        data_yaml=data_yaml,
        decision_payload=_decision_payload(),
        epoch_num=5,
        output_dir=tmp_path / "run",
        debug_dir=tmp_path / "debug",
    )

    weight_map = artifacts["sample_weight_map"]
    assert weight_map["generated"] is True
    assert weight_map["final_val_used_for_policy_selection"] is False
    assert "12" not in weight_map["target_classes"]
    assert weight_map["weighted_train_core_images_count"] > 0
    assert Path(artifacts["paths"]["sample_weight_map_latest"]).exists()


def test_probe_issue_maps_to_train_core_weights_and_protected_images_are_skipped(tmp_path: Path) -> None:
    data_yaml = _write_dataset(tmp_path / "dataset")

    artifacts = build_sampler_only_artifacts(
        data_yaml=data_yaml,
        decision_payload=_decision_payload(),
        epoch_num=5,
        debug_dir=tmp_path / "debug",
    )
    by_name = {Path(item["image_path"]).stem: item for item in artifacts["sample_weight_map"]["image_weights"]}

    assert by_name["img0"]["weight"] == 1.3
    assert by_name["img4"]["weight"] == 1.2
    assert by_name["img1"]["weight"] == 1.0
    assert by_name["img2"]["weight"] == 1.0
    assert by_name["img3"]["weight"] == 1.0
    assert by_name["img3"]["protected_present"] is True
    assert by_name["img3"]["ok_protected"] is True


def test_weighted_index_list_is_effective_and_changes_distribution(tmp_path: Path) -> None:
    data_yaml = _write_dataset(tmp_path / "dataset")

    artifacts = build_sampler_only_artifacts(
        data_yaml=data_yaml,
        decision_payload=_decision_payload(),
        epoch_num=5,
        debug_dir=tmp_path / "debug",
    )

    weighted = artifacts["weighted_train_indices"]
    distribution = artifacts["sampled_distribution_before_after"]
    assert weighted["weighted_index_list_enabled"] is True
    assert weighted["weighted_sampler_enabled"] is False
    assert weighted["weighted_train_indices_count"] > weighted["original_train_core_images_count"]
    assert weighted["sampler_only_effective"] is True
    assert distribution["sampled_distribution_changed"] is True


def test_sampler_only_dataloader_activation_marks_effective_only_when_connected(tmp_path: Path) -> None:
    data_yaml = _write_dataset(tmp_path / "dataset")
    artifacts = build_sampler_only_artifacts(
        data_yaml=data_yaml,
        decision_payload=_decision_payload(),
        epoch_num=5,
        debug_dir=tmp_path / "debug",
    )

    class FakeDataset:
        def __init__(self) -> None:
            self.weighted_indices = None
            self.metadata = None

        def set_weighted_indices(self, weighted_indices, metadata=None) -> None:
            self.weighted_indices = list(weighted_indices)
            self.metadata = dict(metadata or {})

    class FakeLoader:
        def __init__(self) -> None:
            self.reset_called = False

        def reset(self) -> None:
            self.reset_called = True

    dataset = FakeDataset()
    loader = FakeLoader()
    context = SimpleNamespace(train_dataset=dataset, train_loader=loader)

    activation = activate_sampler_only_weighted_index_list(context, artifacts)

    assert activation["sampler_only_effective"] is True
    assert activation["weighted_index_list_enabled"] is True
    assert loader.reset_called is True
    assert dataset.weighted_indices == artifacts["weighted_train_indices"]["weighted_train_indices"]
    disconnected = activate_sampler_only_weighted_index_list(SimpleNamespace(train_dataset=None, train_loader=None), artifacts)
    assert disconnected["sampler_only_effective"] is False
    assert disconnected["sample_weighting_status"] == "pending_dataloader_support"
    assert disconnected["blocker"]


def test_sampler_only_does_not_trigger_image_or_roi_augmentation(tmp_path: Path) -> None:
    policy = initial_policy_matrix({4: "oil"})

    filtered, event = apply_offline_probe_decision_to_policy(policy, _sampler_policy_payload(), epoch_num=5, output_dir=tmp_path)
    router = SampleAwareAugmentationRouter(filtered, seed=3, num_classes=7, roi_aware=True)
    image = np.full((64, 64, 3), 96, dtype=np.uint8)
    labels = np.array([4], dtype=np.int64)
    bboxes = np.array([[14, 14, 48, 48]], dtype=np.float32)

    result = router.apply(image, labels, bboxes)

    assert event["action"] == "sampler_only_pending"
    assert event["probe_reject_image_aug"] is True
    assert router.random_draw_count == 0
    assert router.roi_stats.to_dict()["roi_aug_applied"] == 0
    assert router.stats.to_dict()["samples_augmented"] == 0
    assert result.audit["router"]["skip_reason"] in {"stable_classes_only", "no_active_target_class"}


def test_sampler_only_does_not_rewrite_labels(tmp_path: Path) -> None:
    data_yaml = _write_dataset(tmp_path / "dataset")
    label_path = tmp_path / "dataset" / "labels" / "train_core" / "img0.txt"
    before = label_path.read_text(encoding="utf-8")

    build_sampler_only_artifacts(
        data_yaml=data_yaml,
        decision_payload=_decision_payload(),
        epoch_num=5,
        debug_dir=tmp_path / "debug",
    )

    assert label_path.read_text(encoding="utf-8") == before
