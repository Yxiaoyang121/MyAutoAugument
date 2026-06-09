from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from AutoAugment.catf_v2.causal_probe import apply_candidate_view, candidate_policy_catalog
from scripts.train_yolo_default_with_inloop_feedback import (
    apply_offline_probe_decision_to_policy,
    build_probe_decision_from_rows,
    normalize_paper_probe_args,
    policy_selection_data_yaml,
    policy_selection_source,
    validate_paper_probe_configuration,
)


def _write_dataset(root: Path, *, overlap_probe_val: bool = False) -> tuple[Path, Path]:
    names = {0: "ok", 1: "defect"}
    for split in ("train_core", "probe", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split, count in {"train_core": 2, "probe": 1, "val": 1}.items():
        for idx in range(count):
            stem = f"{split}_{idx}"
            if overlap_probe_val and split == "probe":
                stem = "val_0"
            (root / "images" / split / f"{stem}.jpg").write_bytes(b"fake")
            (root / "labels" / split / f"{stem}.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    probe_yaml = root / "probe.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"path": str(root), "train": "images/train_core", "val": "images/val", "nc": 2, "names": names}),
        encoding="utf-8",
    )
    probe_yaml.write_text(
        yaml.safe_dump({"path": str(root), "train": "images/probe", "val": "images/probe", "probe": "images/probe", "nc": 2, "names": names}),
        encoding="utf-8",
    )
    return data_yaml, probe_yaml


def _args(data_yaml: Path, probe_yaml: Path) -> Namespace:
    return Namespace(
        data=str(data_yaml),
        probe_data=str(probe_yaml),
        train_core_data=str(data_yaml),
        paper_probe_mode=True,
        forbid_final_val_policy_selection=True,
        probe_source="train_probe_split",
        catf_causal_probe=False,
        catf_causal_probe_mode="development",
    )


def test_train_core_probe_val_no_overlap(tmp_path: Path) -> None:
    data_yaml, probe_yaml = _write_dataset(tmp_path)
    audit = validate_paper_probe_configuration(_args(data_yaml, probe_yaml))

    assert audit["policy_selection_source"] == "probe_split"
    assert audit["train_core_probe_overlap_count"] == 0
    assert audit["probe_final_val_overlap_count"] == 0
    assert audit["train_core_final_val_overlap_count"] == 0
    assert audit["final_val_used_for_policy_selection"] is False


def test_paper_mode_final_val_not_policy_selection_source(tmp_path: Path) -> None:
    data_yaml, probe_yaml = _write_dataset(tmp_path)
    args = _args(data_yaml, probe_yaml)

    assert policy_selection_source(args) == "probe_split"
    assert policy_selection_data_yaml(args) == probe_yaml.resolve()


def test_paper_and_development_modes_are_explicit(tmp_path: Path) -> None:
    data_yaml, probe_yaml = _write_dataset(tmp_path)
    args = _args(data_yaml, probe_yaml)
    normalize_paper_probe_args(args)

    assert args.catf_causal_probe is True
    assert args.catf_causal_probe_mode == "paper"

    dev_args = Namespace(data=str(data_yaml), paper_probe_mode=False, probe_data=None)
    assert policy_selection_source(dev_args) == "final_val_diagnostics"


def test_candidate_decision_records_probe_split_not_seed_or_fixed_class() -> None:
    rows = [
        {
            "class_id": 4,
            "evidence_count": 8,
            "diagnosis_confidence": 0.7,
            "fn_count": 6,
            "val_instances": 12,
            "dominant_issue": "texture_boundary_weak",
            "AP50": 0.62,
            "AP50_95": 0.48,
            "Precision": 0.74,
            "strong_update_allowed": True,
        }
    ]
    policy = {"classes": {"4": {"ops": {"sharpen_mild": {"prob": 0.1, "strength": 0.2}, "local_contrast": {"prob": 0.1, "strength": 0.2}}}}}

    decision = build_probe_decision_from_rows(
        active_rows=rows,
        policy=policy,
        epoch_num=5,
        mode="paper",
        policy_selection_data="probe.yaml",
        policy_selection_source="probe_split",
    )

    assert decision["policy_selection_source"] == "probe_split"
    assert decision["seed_specific_rule"] is False
    assert decision["fixed_class_id_specific_rule"] is False
    assert decision["dataset_specific_rule"] is False


def test_noop_path_does_not_rewrite_labels_or_instances(tmp_path: Path) -> None:
    policy = {"classes": {"1": {"ops": {"sharpen_mild": {"prob": 0.2, "strength": 0.2}}}}}
    decision_payload = {
        "selected_candidate_policy_id": "candidate_policy_0_noop",
        "selected_candidate": {
            "candidate_policy": candidate_policy_catalog()["candidate_policy_0_noop"],
            "decision": {
                "decision": "no_op",
                "image_modification_allowed": False,
                "sample_weighting_allowed": False,
                "strict_noop": True,
            },
        },
    }

    gated, event = apply_offline_probe_decision_to_policy(policy, decision_payload, epoch_num=5, output_dir=tmp_path)

    assert event["action"] == "strict_no_op"
    assert all(float(op.get("prob", 0.0) or 0.0) == 0.0 for row in gated["classes"].values() for op in row["ops"].values())


def test_probe_does_not_mutate_model_optimizer_ema_rng() -> None:
    rng = {"state": 7}
    model = object()
    optimizer = object()
    ema = object()

    view = apply_candidate_view(
        candidate_policy=candidate_policy_catalog()["candidate_policy_1_roi_texture"],
        model=model,
        optimizer=optimizer,
        ema=ema,
        rng_state=rng,
    )

    assert view["model_state_updated"] is False
    assert view["optimizer_state_updated"] is False
    assert view["ema_state_updated"] is False
    assert view["rng_state_before"] == rng
    assert view["rng_state_after"] == rng


def test_riskguard_blacklist_not_final_accept_reject_rule() -> None:
    rows = [
        {
            "class_id": 9,
            "evidence_count": 8,
            "diagnosis_confidence": 0.7,
            "fn_count": 6,
            "val_instances": 12,
            "dominant_issue": "texture_boundary_weak",
            "AP50": 0.62,
            "AP50_95": 0.48,
            "Precision": 0.74,
            "strong_update_allowed": True,
        }
    ]

    decision = build_probe_decision_from_rows(
        active_rows=rows,
        policy={},
        epoch_num=5,
        mode="paper",
        policy_selection_data="probe.yaml",
        policy_selection_source="probe_split",
    )

    image_eval = next(item for item in decision["candidate_evaluations"] if item["candidate_policy_id"] == "candidate_policy_1_roi_texture")
    assert image_eval["decision"]["audit_prior_only"] is True
    assert image_eval["decision"]["decision"] == "accept"
    assert decision["fixed_class_id_specific_rule"] is False


def test_final_val_as_probe_fails(tmp_path: Path) -> None:
    data_yaml, _probe_yaml = _write_dataset(tmp_path)
    args = _args(data_yaml, data_yaml)

    with pytest.raises(ValueError):
        validate_paper_probe_configuration(args)
