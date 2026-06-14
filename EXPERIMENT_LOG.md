# Experiment Log

## 2026-06-13

### Seed2 Image-Only Weak Augmentation 50ep

Scope:

- Ran only seed2.
- Did not run seed0, seed1, or multiseed.
- Did not use sampler_only or weighted index lists.
- Did not modify sampling or data splits.
- Attenuation ratio remained `0.25`.

Code / schedule:

- Added `candidate_policy_1b_weak_roi_texture`.
- Added image-only weak augmentation CLI support to `scripts/train_yolo_default_with_inloop_feedback.py`.
- Added weak interval cap support to `AutoAugment/catf_v2/sample_router.py`.
- Generated epoch-specific schedule:
  - `outputs/experiments/catf_v2_image_only_weak_aug_seed2_50ep/configs/weak_image_aug_offline_decisions_seed2.json`

Outputs:

- `outputs/experiments/catf_v2_image_only_weak_aug_seed2_50ep/reports/seed2_weak_image_aug_report.md`
- `outputs/experiments/catf_v2_image_only_weak_aug_seed2_50ep/reports/seed2_weak_image_aug_report.json`
- `outputs/experiments/catf_v2_image_only_weak_aug_seed2_50ep/reports/final_metrics.json`
- `outputs/experiments/catf_v2_image_only_weak_aug_seed2_50ep/reports/online_aug_stats.json`
- `outputs/experiments/catf_v2_image_only_weak_aug_seed2_50ep/reports/roi_aug_stats.json`

Execution:

| item | value |
|---|---:|
| 50ep completed | true |
| weak image augmentation executed | true |
| industrial images augmented | 80 |
| ROI applied | 95 |
| router random draws | 1922 |
| weak interval cap | 16 |
| sampler_only_enabled | false |
| weighted_index_list_enabled | false |
| sampled_distribution_changed | false |

Weak candidates:

| epoch | op | original prob | original strength | weak prob | weak strength |
|---:|---|---:|---:|---:|---:|
| 20 | local_contrast | 0.18 | 0.20 | 0.045 | 0.05 |
| 25 | local_contrast | 0.18 | 0.20 | 0.045 | 0.05 |
| 30 | local_contrast | 0.18 | 0.20 | 0.045 | 0.05 |
| 35 | local_contrast | 0.18 | 0.20 | 0.045 | 0.05 |
| 45 | local_contrast | 0.18 | 0.20 | 0.045 | 0.05 |

Metrics:

| run | P | R | mAP50 | mAP50-95 | constraint_failed |
|---|---:|---:|---:|---:|---|
| clean seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | false |
| fixed CATF-v2 seed2 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | true |
| weak image aug seed2 | 0.7533 | 0.6942 | 0.7727 | 0.5151 | false |

Delta:

- vs clean seed2: dP=+0.0570, dR=-0.0344, dM50=+0.0035, dM95=-0.0072.
- vs fixed CATF-v2 seed2: dP=-0.0104, dR=+0.0079, dM50=+0.0145, dM95=+0.0185.

Interpretation:

- Weak image-only augmentation repairs the seed2 fixed CATF-v2 constraint failure.
- Class9 recovers relative to fixed CATF-v2 but is not fully recovered relative to clean.
- Non-active regression is mitigated relative to fixed CATF-v2.
- Next validation should be seed0/seed1 sanity with the same image-only weak augmentation setup.

Verification:

- Requested py_compile passed.
- Requested targeted pytest set passed: `55 passed`.

### Image-Only Weak Augmentation Replay

Scope:

- Offline replay and method design only.
- No training was run.
- No seed0/seed1/seed2 run was started.
- No clean baseline was rerun.
- No gate, sampler, causal score, data split, or training augmentation logic was changed.

Script:

- `scripts/replay_weak_image_aug.py`

Outputs:

- `outputs/experiments/catf_v2_image_only_weak_aug_replay/reports/weak_image_aug_replay.md`
- `outputs/experiments/catf_v2_image_only_weak_aug_replay/reports/weak_image_aug_replay.json`
- `outputs/experiments/catf_v2_image_only_weak_aug_replay/weak_candidate_records.csv`
- `outputs/experiments/catf_v2_image_only_weak_aug_replay/reports/weak_image_aug_training_plan.md`

Replay design:

- `candidate_policy_3_sampler_only` is disabled for the main path.
- Added replay candidate `candidate_policy_1b_weak_roi_texture`.
- Weak ROI texture keeps one lower-risk op, attenuates probability and strength, and caps augmented samples per feedback interval.
- Tested attenuation ratios: `0.5` and `0.25`.
- Ratio `0.5` remains too risky.
- Ratio `0.25` passes replay gates for legacy image-evidence rows.
- No seed-id, class-id, or dataset-class-name hard rule is used.

Coverage:

| item | count |
|---|---:|
| total image candidates | 54 |
| original ROI texture candidates | 27 |
| legacy image-evidence candidates | 8 |
| weak ROI texture accepted by replay | 8 |
| strict no-op | 19 |
| ratio 0.5 accepts | 0 |
| ratio 0.25 accepts | 8 |

Seed-level replay:

| seed | weak epochs | weak count | strict no-op count |
|---:|---|---:|---:|
| 0 | [25] | 1 | 8 |
| 1 | [25, 40] | 2 | 7 |
| 2 | [20, 25, 30, 35, 45] | 5 | 4 |

Interpretation:

- sampler_only involved=false.
- final_val_leakage=false.
- seed0/seed1 legacy image candidates are preserved as weak image candidates.
- seed2 has offline gate-safe weak image candidates, but this is not a training result.
- Recommended next validation, if requested: run seed2 50ep image-only weak augmentation first, reuse clean seed2, compare against fixed CATF-v2 seed2, and only then run seed0/seed1 sanity if seed2 passes.

Verification:

- `python scripts/replay_weak_image_aug.py`
- `python -m py_compile scripts/replay_weak_image_aug.py`

### Restore Image-Only CATF Mainline and Demote Sampler-Only

Scope:

- No training was run.
- No seed0/seed1/seed2 run was started.
- No clean baseline was rerun.
- No gate, sampler, augmentation policy, causal score, or data split was changed.
- Added image-only mainline reports and demoted sampler-only from the paper main method.

Outputs:

- `outputs/experiments/catf_v2_image_only_mainline/reports/image_only_catf_v2_mainline_summary.md`
- `outputs/experiments/catf_v2_image_only_mainline/reports/image_only_catf_v2_mainline_summary.json`
- `outputs/experiments/catf_v2_image_only_mainline/reports/sampler_only_demoted_note.md`

Method positioning:

- The paper mainline is restored to image augmentation based CATF.
- `sampler_only` has been implemented and verified, but it is demoted to engineering exploration / ablation only.
- `sampler_only` is a training sampling intervention closer to hard example mining / weighted sampling; it changes the train distribution and is not equivalent to image data augmentation.
- Current image-augmentation mainline baseline: fixed CATF-v2.
- Main method candidates are fixed CATF-v2 and CP-CATF image-only. Future main results must come from image augmentation, not sampling reweighting.

Fixed CATF-v2 image-only baseline:

| seed | clean P | clean R | clean mAP50 | clean mAP50-95 | fixed P | fixed R | fixed mAP50 | fixed mAP50-95 | dP | dR | dM50 | dM95 | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | -0.0061 | -0.0068 | +0.0090 | +0.0136 | false |
| 1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | +0.0127 | +0.0528 | +0.0284 | +0.0390 | false |
| 2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | +0.0675 | -0.0423 | -0.0110 | -0.0257 | true |
| mean | 0.7511 | 0.6843 | 0.7527 | 0.4927 | 0.7758 | 0.6855 | 0.7615 | 0.5017 | +0.0247 | +0.0012 | +0.0088 | +0.0090 | n/a |

Interpretation:

- fixed CATF-v2 is the current image augmentation mainline.
- seed0/seed1 show CATF image augmentation has real potential.
- seed2 fails because high-risk image augmentation caused non-active regression and mAP drops.
- Seed2 must be repaired inside the image-augmentation mainline: causal probe, weak image augmentation / attenuation, strict image no-op safety, and non-active regression constraints.
- Do not use `sampler_only`, weighted index lists, or hard-example mining as the paper main result.

## 2026-06-12

### CP-CATF Paper-Mode Sampler-Only Multiseed

Scope:

- Continued from completed seed0 sampler-only.
- Ran only seed1 and seed2 under paper-mode sampler-only.
- Reused clean paper baselines and existing seed0 CP-CATF sampler-only result.
- Did not rerun clean, did not rerun seed0, and did not change the gate, sampler weights, causal score, candidate strategy, augmentation policy, or data split.

Outputs:

- Root: `outputs/experiments/multiseed_cp_catf_paper_mode_sampler_only/`
- Seed1: `outputs/experiments/multiseed_cp_catf_paper_mode_sampler_only/cp_catf_seed_1/`
- Seed2: `outputs/experiments/multiseed_cp_catf_paper_mode_sampler_only/cp_catf_seed_2/`
- Summary:
  - `outputs/experiments/multiseed_cp_catf_paper_mode_sampler_only/reports/multiseed_sampler_only_summary.md`
  - `outputs/experiments/multiseed_cp_catf_paper_mode_sampler_only/reports/multiseed_sampler_only_summary.json`

Sampler-only:

- Uses weighted index list, not `WeightedRandomSampler`.
- sampler_only_effective seed count: 3/3.
- Effective feedback event count: 25.
- Weighted train_core images: seed0=72, seed1=123, seed2=176, total=371.
- sampled_distribution_changed=true for all seeds.
- image augmented=0, ROI applied=0, router random draw count=0 for all seeds.
- final_val_used_for_policy_selection=false and final_val_leakage=false for all seeds.

Metrics:

| seed | clean P | clean R | clean mAP50 | clean mAP50-95 | CP P | CP R | CP mAP50 | CP mAP50-95 | dP | dR | dM50 | dM95 | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7513 | 0.6763 | 0.7566 | 0.5114 | 0.7588 | 0.6878 | 0.7779 | 0.5204 | +0.0075 | +0.0115 | +0.0213 | +0.0090 | false |
| 1 | 0.7220 | 0.7582 | 0.7777 | 0.5251 | 0.7085 | 0.7258 | 0.7412 | 0.5112 | -0.0135 | -0.0324 | -0.0365 | -0.0139 | true |
| 2 | 0.6290 | 0.6385 | 0.6590 | 0.4381 | 0.7062 | 0.6512 | 0.6843 | 0.4427 | +0.0772 | +0.0127 | +0.0253 | +0.0046 | false |
| mean | 0.7008 | 0.6910 | 0.7311 | 0.4915 | 0.7245 | 0.6883 | 0.7345 | 0.4914 | +0.0237 | -0.0027 | +0.0034 | -0.0001 | n/a |

Conclusion:

- 3/3 pass=false; pass_count=2/3.
- CP-CATF paper-mode sampler-only should not be used as the current paper main-method result.
- The run proves sampler_only is genuinely connected to training, but it is a sampling intervention rather than image data augmentation.
- As of 2026-06-13, sampler-only is demoted to engineering exploration / ablation only.

Verification:

- `python -m py_compile AutoAugment/catf_v2/causal_probe.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py scripts/train_yolo_default_with_inloop_feedback.py scripts/summarize_cp_catf_sampler_only_multiseed.py`
- `python -m pytest -q tests/test_cp_catf_sampler_only.py tests/test_cp_catf_accept_to_execution.py tests/test_cp_catf_paper_mode.py tests/test_catf_v2_causal_probe.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_sample_router.py tests/test_online_augmentation.py`
- Result: `60 passed`.

### CP-CATF Effective Sampler-Only Dataloader Intervention

Scope:

- Implemented sampler-only dataloader support.
- Ran sampler-only smoke.
- Ran seed0 50ep only after smoke passed.
- Did not run seed1/seed2 training.
- Did not run multiseed.
- Did not change causal score, precision gate thresholds, augmentation strength, or data splits.

Implementation:

- Added `AutoAugment/catf_v2/sampler_only.py`.
- Added `--sampler-only-enabled`.
- Updated `OnlineYOLODataset` with weighted index-list mapping.
- Updated `OnlineAugDetectionTrainer` to expose the active train dataset and dataloader.
- Activated sampler-only in the feedback callback by installing weighted indices and calling dataloader `reset()`.
- Chose weighted index list instead of `WeightedRandomSampler` because the active Ultralytics dataloader builder does not expose a sampler injection argument.

Dataloader audit:

- `outputs/debug/cp_catf_sampler_only_dataloader_impl/dataloader_entry_audit.md`
- `outputs/debug/cp_catf_sampler_only_dataloader_impl/sample_weight_map.json`
- `outputs/debug/cp_catf_sampler_only_dataloader_impl/weighted_train_indices.json`
- `outputs/debug/cp_catf_sampler_only_dataloader_impl/sampled_distribution_before_after.json`

Smoke:

- Output: `outputs/debug/cp_catf_sampler_only_execution_smoke/`
- `sample_weight_map_generated=true`.
- `weighted_train_core_images_count=178`.
- `weighted_index_list_enabled=true`.
- `sampler_only_effective=true`.
- `sampled_distribution_changed=true`.
- Industrial image augmented=0.
- ROI applied=0.
- Router random draw count=0.
- bbox/class legal.

Seed0 50ep:

- Output: `outputs/experiments/cp_catf_paper_mode_sampler_only_seed0/`
- Paper-mode=true.
- final val was not used for policy selection.
- Feedback epochs: 5/10/15/20/25/30/35/40/45.
- Epoch5 and epoch10 selected sampler-only but stayed pending with explicit blocker: `sample_weight_map contains no train_core images with weight > 1`.
- Epochs 15/20/25/30/35/40/45 were effective sampler-only.
- Final sampler-only status: `effective_weighted_index_list`.
- Final weighted train_core images count: 72.
- Final weighted index list enabled=true.
- Final sampled distribution changed=true.
- Industrial image augmented=0.
- ROI applied=0.
- Router random draw count=0.

Metrics vs requested paper clean seed0:

| run | P | R | mAP50 | mAP50-95 | constraint_failed |
|---|---:|---:|---:|---:|---|
| clean paper seed0 | 0.7513 | 0.6763 | 0.7566 | 0.5114 | false |
| sampler-only seed0 | 0.7588 | 0.6878 | 0.7779 | 0.5204 | false |
| delta | +0.0075 | +0.0115 | +0.0213 | +0.0090 |  |

Verification:

- `python -m py_compile AutoAugment/catf_v2/sampler_only.py scripts/train_yolo_online_aug.py scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/causal_probe.py AutoAugment/catf_v2/policy_matrix.py AutoAugment/catf_v2/sample_router.py`
- `python -m pytest tests/test_cp_catf_sampler_only.py tests/test_cp_catf_accept_to_execution.py tests/test_catf_v2_causal_probe.py -q`
- Result: `29 passed`.

### CP-CATF Paper-Mode Decision Coverage Audit

Scope:

- Analysis only.
- No 50ep training was run.
- No seed1/seed2 execution was started.
- No multiseed run was started.
- No clean rerun was started.
- No causal score, precision gate, augmentation strength, data split, or sampler_only implementation was changed.

Script:

- `scripts/analyze_cp_catf_decision_coverage.py`

Inputs:

- `outputs/experiments/multiseed_cp_catf_paper_mode/`
- `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/`
- `outputs/experiments/cp_catf_precision_gate_dry_run_seed0/`
- `outputs/experiments/cp_catf_paper_mode_precision_gate_seed0_rerun/`
- `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/`

Command:

- `python scripts/analyze_cp_catf_decision_coverage.py`

Paper-mode status:

- train_core=2071.
- probe=230.
- final val=677.
- final val leakage=false.
- final val was not used for policy selection.

Coverage result:

| metric | value |
|---|---:|
| total_candidates | 81 |
| total_image_candidates | 54 |
| logged original image causal accepts | 8 |
| current replay image causal accepts | 0 |
| precision-gate rejects after logged original image accept | 8 |
| final image accepts | 0 |
| sampler-only selected | 27 |
| effective sampler-only | 0 |
| strict no-op | 27 |

Seed-level coverage:

| seed | total candidates | image candidates | logged original image accepts | precision-gate rejects | final image accepts | sampler selected | strict no-op |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 27 | 18 | 1 | 1 | 0 | 9 | 9 |
| 1 | 27 | 18 | 2 | 2 | 0 | 9 | 9 |
| 2 | 27 | 18 | 5 | 5 | 0 | 9 | 9 |

Main rejection buckets:

- no_positive_benefit=54.
- high_fp_spillover_rate_too_high=40.
- non_active_regression_too_high=40.
- estimated_precision_drop_too_high=20.
- non_active_fp_delta_too_high=20.
- high_confidence_fp_delta_too_high=20.
- insufficient_evidence=14.
- bbox_instability_too_high=5.

Key interpretation:

- Paper-mode leakage control is complete.
- Accept-to-execution is already proven by seed0 execution-fixed ROI application.
- The precision-aware gate blocks the known seed0 Precision-risk candidate.
- Current replay rejects every image candidate, so paper-mode image augmentation becomes sampler-only/strict no-op.
- `high_confidence_fp_delta > 0.0` is strict but not the sole blocker; relaxing only that condition admits zero candidates.
- Relaxing only `non_active_fp_delta` or only `estimated_precision_drop` also admits zero candidates.
- There are 8 legacy `roi_texture` accepts that are reasonable graded-attenuation study candidates, but none is safe at original probability/strength under the current gate.
- Sampler-only was later implemented and verified, but is now demoted to engineering exploration / ablation only.
- Mainline work should pursue causal-probe image acceptance and graded image-augmentation attenuation before any seed0 image-rerun.
- Do not use current paper-mode CP-CATF as a paper method result; it is leakage-free safety/no-op evidence.

Outputs:

- `outputs/experiments/cp_catf_decision_coverage_audit/reports/decision_coverage_audit.md`
- `outputs/experiments/cp_catf_decision_coverage_audit/reports/decision_coverage_audit.json`
- `outputs/experiments/cp_catf_decision_coverage_audit/decision_records.csv`
- `outputs/experiments/cp_catf_decision_coverage_audit/rejection_reason_summary.csv`
- `outputs/experiments/cp_catf_decision_coverage_audit/seed_level_coverage.csv`

Verification:

- `python -m py_compile scripts/analyze_cp_catf_decision_coverage.py`
- `python -m py_compile AutoAugment/catf_v2/causal_probe.py`
- `python -m py_compile AutoAugment/catf_v2/policy_matrix.py`
- `python -m py_compile AutoAugment/catf_v2/sample_router.py`

## 2026-06-11

### CP-CATF Precision-Aware Accept Gate From Seed0 Audit

Scope:

- No training was run.
- No seed1/seed2 run was started.
- No multiseed run was started.
- This update converts the paper-mode seed0 precision-risk audit into a generic accept-gate change.

Seed0 context:

- Paper-mode accept-to-execution was already proven: `candidate_policy_1_roi_texture` accepted at epoch 25.
- Actual execution: ROI applied=312, industrial image augmented=226, router random draw count=1330.
- Final validation leakage remained false.
- Clean paper seed0 Precision=0.7513; CP-CATF seed0 Precision=0.7399; delta=-0.0114.
- The audit found that Precision loss was driven mainly by non-active false-positive spillover, not by class 9. Class 9 improved locally.

Implementation:

- Added precision-aware reject fields in `AutoAugment/catf_v2/causal_probe.py`:
  - `estimated_precision_drop`;
  - `non_active_fp_delta`;
  - `high_confidence_fp_delta`.
- Image candidates are rejected when these exceed the configured margins:
  - `estimated_precision_drop > 0.005`;
  - `non_active_fp_delta > 0.005`;
  - `high_confidence_fp_delta > 0.0`.
- These terms do not change the causal score formula.
- Paper-mode risk estimation now uses full probe-split per-class context for non-active risk; candidate selection remains active-row based.
- No seed-id, fixed class-id, or dataset class-name rule was added.

Verification:

- `python -m py_compile AutoAugment/catf_v2/causal_probe.py scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py`
- `pytest -q tests/test_cp_catf_accept_to_execution.py tests/test_cp_catf_paper_mode.py tests/test_catf_v2_causal_probe.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_policy_matrix.py tests/test_catf_v2_sample_router.py tests/test_catf_v2_roi_augmentation.py tests/test_inloop_feedback_training.py tests/test_online_augmentation.py`
- Result: `70 passed`.

Reports:

- `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_precision_aware_gate_update.md`
- `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_precision_aware_gate_update.json`

## 2026-06-08

### CATF-v2 RiskGuard Seed2 Validation

Implemented a minimum high-risk class-op guard for the seed2 class 9 ROI texture failure path.

Scope:

- Added `--catf-riskguard true`.
- Added `AutoAugment/catf_v2/high_risk_class_ops.py`.
- Added `tests/test_catf_v2_riskguard.py`.
- Did not run seed0/seed1 sanity because seed2 did not pass the constraint gate.

RiskGuard rule:

- High-risk combination is class 9 + `sharpen_mild` / `local_contrast`.
- The rule is class-op based, not seed-id based.
- Blocked ops require future causal probe clearance before training augmentation.
- Blocked ops do not execute ROI augmentation, do not rewrite labels/Instances, and do not consume CATF random draws.
- Sampler-only fallback is recorded, but sample weighting remains pending dataloader support.

Verification:

- `python -m py_compile scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py AutoAugment/catf_v2/high_risk_class_ops.py`
- `pytest -q tests/test_catf_v2_riskguard.py ... tests/test_copy_paste.py`
- Result: `121 passed`.

Seed2 50ep run:

- Output: `outputs/experiments/catf_v2_riskguard_seed2_50ep/`
- Reference/control: clean seed2 artifacts from `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/seed_2/clean_native_yolo_default/`.
- Epoch5 class 9 `texture_boundary_weak` reproduced.
- RiskGuard blocked class 9 `local_contrast` and `sharpen_mild` at epoch5.
- RiskGuard events: 2.
- Industrial samples augmented: 13 versus fixed seed2 86.
- ROI applied: 15 versus fixed seed2 90.
- ROI affected classes: class 12 only.
- OK3 active=false; OK3 ROI applied=0.

Metrics:

| group | P | R | mAP50 | mAP50-95 | constraint_failed |
|---|---:|---:|---:|---:|---|
| clean seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | false |
| fixed CATF-v2 seed2 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | true |
| RiskGuard seed2 | 0.6394 | 0.7231 | 0.7552 | 0.5073 | true |

Class 9 local effect:

- Recall clean/fixed/RiskGuard: 0.6310 / 0.4643 / 0.7960.
- AP50 clean/fixed/RiskGuard: 0.7440 / 0.6973 / 0.7944.
- AP50-95 clean/fixed/RiskGuard: 0.4139 / 0.3484 / 0.4198.

Conclusion:

- RiskGuard successfully prevents the audited class 9 texture ROI path and restores class 9 local metrics versus fixed CATF-v2.
- RiskGuard alone is not enough: overall seed2 still fails due to Precision/mAP drops and residual class12-only ROI activity.
- Next minimum technical step should combine RiskGuard with CP-CATF causal probe or a precision/mAP-aware gate for later class-op interventions.

Reports:

- `outputs/experiments/catf_v2_riskguard_seed2_50ep/reports/final_report.md`
- `outputs/experiments/catf_v2_riskguard_seed2_50ep/reports/final_metrics.json`
- `outputs/experiments/catf_v2_riskguard_seed2_50ep/reports/riskguard_events.json`
- `outputs/experiments/catf_v2_riskguard_seed2_50ep/reports/compare_with_clean_and_fixed_catf_v2.md`
- `outputs/experiments/catf_v2_riskguard_seed2_50ep/reports/riskguard_seed2_summary.json`

## 2026-06-07

### CATF-v2 Strategy Limitation Analysis From Seed2 Audit

Generated a strategy-level limitation analysis from the completed seed2 root-cause audit.

Scope:

- No training was run.
- No 10ep or 50ep ablation was run.
- No CATF-v2 rule or augmentation strategy was changed.
- The report reads the existing seed2 root-cause audit reports and fixed multiseed summary only.

Outputs:

- `outputs/experiments/seed2_failure_root_cause/reports/catf_v2_strategy_limitation_analysis.md`
- `outputs/experiments/seed2_failure_root_cause/reports/catf_v2_strategy_limitation_analysis.json`

Main conclusion:

- CATF-v2 should not be described as a failed method. Fixed CATF-v2 has valid gains on seed0/seed1 and average metrics improve.
- Seed2 exposes immature strategy selection and risk control: diagnosis does not automatically imply that a specific augmentation is beneficial.

Seed2 evidence used:

- clean seed2: P=0.6962, R=0.7286, mAP50=0.7692, mAP50-95=0.5224.
- fixed seed2: P=0.7637, R=0.6863, mAP50=0.7582, mAP50-95=0.4967.
- fixed seed2 is more conservative: Precision rises, Recall and AP metrics fall.
- Recall first lags clean at epoch 6; mAP50/mAP50-95 clearly lag by epoch 8.
- Most suspicious update: epoch5 class 9 `texture_boundary_weak` activation with `sharpen_mild` and `local_contrast`.
- Most suspicious class-op evidence: class 9 ROI applied=25, final Recall -0.1667, AP50 -0.0466, AP50-95 -0.0654.
- Non-active class regression exists, so active-class-only monitoring is insufficient.

Strategy limitations:

- Diagnosis is not equivalent to augmentation benefit.
- Active class improvement cannot be assumed after a local diagnosis.
- Candidate augmentation lacks pre-training causal validation.
- Non-active class regression is not sufficiently constrained.
- Co-enabled `sharpen_mild` and `local_contrast` prevent operator-level risk isolation.
- Fallback/gate can detect problems after weights have already been affected.
- Strong clean-baseline seeds should prefer strict image no-op unless image-augmentation causal evidence is positive.

Minimal improvement direction:

- CP-CATF causal probe before weight updates.
- Active/non-active dual constraints.
- High-risk class-op candidate gating for class 9 + ROI texture combinations.

Verification:

- `D:\Anaconda\envs\pytorch\python.exe -m py_compile scripts\summarize_catf_v2_strategy_limitations.py`

### Seed2 CATF-v2 Failure Root-Cause Audit

Completed a root-cause audit for the fixed CATF-v2 seed2 failure.

Scope:

- No training was run.
- Existing clean, fixed CATF-v2, Gated, Safe, and adaptive-RB artifacts were read.
- Predict-only validation was run on existing clean/fixed best weights to generate cached prediction JSON and debug images.
- CATF-v2 rules and training code were not changed.

Outputs:

- `outputs/experiments/seed2_failure_root_cause/reports/seed2_curve_degradation_analysis.md`
- `outputs/experiments/seed2_failure_root_cause/reports/seed2_curve_degradation_analysis.json`
- `outputs/experiments/seed2_failure_root_cause/reports/seed2_per_class_regression_analysis.md`
- `outputs/experiments/seed2_failure_root_cause/reports/seed2_per_class_regression_analysis.json`
- `outputs/experiments/seed2_failure_root_cause/reports/seed2_augmentation_operator_attribution.md`
- `outputs/experiments/seed2_failure_root_cause/reports/seed2_augmentation_operator_attribution.json`
- `outputs/experiments/seed2_failure_root_cause/reports/seed2_active_vs_regressed_class_analysis.md`
- `outputs/experiments/seed2_failure_root_cause/reports/seed2_active_vs_regressed_class_analysis.json`
- `outputs/experiments/seed2_failure_root_cause/reports/seed2_prediction_diff_analysis.md`
- `outputs/experiments/seed2_failure_root_cause/reports/seed2_prediction_diff_analysis.json`
- `outputs/experiments/seed2_failure_root_cause/reports/seed2_root_cause_summary.md`
- `outputs/experiments/seed2_failure_root_cause/reports/seed2_root_cause_summary.json`
- `outputs/experiments/seed2_failure_root_cause/debug_images/`

Curve localization:

| run | Recall first lag | mAP50 first clear lag | mAP50-95 first clear lag | key context |
|---|---:|---:|---:|---|
| fixed CATF-v2 seed2 | 6 | 8 | 8 | epoch5 class 9 active, texture ops enabled |
| Gated seed2 | 6 | 8 | 8 | epoch10 fallback after the damaged epoch6-10 window |

Fixed seed2 augmentation audit:

- Industrial samples augmented: 86.
- ROI applied: 90.
- Router random draw count: 5610.
- Operators:
  - `local_contrast`: applied 44.
  - `sharpen_mild`: applied 42.
- ROI affected classes:
  - class 9: 25.
  - class 11: 59.
  - class 8: 6.

Per-class finding:

- class 9 is the most suspicious active class: active at epoch5, ROI affected, Recall -0.1667, AP50 -0.0466, AP50-95 -0.0654, estimated FN +14.
- class 8 is active later and loses AP50/AP50-95 despite Recall improving.
- class 11 is active but improves slightly, so texture ROI ops are not uniformly harmful.
- Non-active regression exists: classes 10, 12, 6, 5, 3, and 2 show AP or Recall regression without ROI application.

Prediction diff:

- Clean prediction JSON: `outputs/experiments/seed2_failure_root_cause/predictions/clean_best/validation_predictions.json`.
- Fixed prediction JSON: `outputs/experiments/seed2_failure_root_cause/predictions/fixed_best/validation_predictions.json`.
- Clean detected but fixed missed: 24 GT objects.
- Clean high-IoU but fixed worse localization: 11 GT objects.
- Fixed confidence drop on matched objects: 20 GT objects.
- True pre-NMS ordering cannot be audited from saved post-NMS predictions.

Interpretation:

- The strongest root cause is not final Precision; it is an early candidate-branch side effect after epoch5.
- The most suspicious policy update is epoch5 class 9 `texture_boundary_weak`, with `sharpen_mild` and `local_contrast`.
- The most suspicious operator family is ROI texture enhancement, but current logs cannot separate `sharpen_mild` from `local_contrast` because they were co-applied.
- Seed2 should default to strict image no-op unless a CP-CATF causal probe validates class 9 before image-space intervention.
- Next minimum-cost validation should be CP-CATF causal probe and 10ep short ablations, not another full 50ep run.

Verification:

- `D:\Anaconda\envs\pytorch\python.exe -m py_compile scripts\audit_seed2_catf_v2_failure_root_cause.py scripts\train_yolo_default_with_inloop_feedback.py AutoAugment\catf_v2\sample_router.py AutoAugment\catf_v2\policy_matrix.py`
- Skipped missing tests: `tests/test_catf_v2_causal_probe.py`, `tests/test_catf_v2_adaptive_rb_v2.py`.
- `D:\Anaconda\envs\pytorch\python.exe -m pytest -q tests\test_catf_v2_adaptive_burnin.py tests\test_catf_v2_rollback_controller.py tests\test_catf_v2_gated_controller.py tests\test_catf_v2_safe_controller.py tests\test_catf_v2_transform_bypass.py tests\test_catf_v2_activation_rules.py tests\test_catf_v2_per_class_diagnosis.py tests\test_catf_v2_policy_matrix.py tests\test_catf_v2_sample_router.py tests\test_catf_v2_roi_augmentation.py tests\test_catf_v2_threshold_calibration.py tests\test_feedback_policy_guard.py tests\test_feedback_policy_controller.py tests\test_inloop_feedback_training.py tests\test_online_augmentation.py tests\test_proxy_prefilter.py tests\test_copy_paste.py`
- Result: `111 passed`.

## 2026-06-06

### CATF-v2 Adaptive-RB Full Multiseed Validation

Completed full multiseed adaptive burn-in + RB validation for seeds 0, 1, and 2.

Run group:

- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_adaptive_rb/`

Execution:

- Newly ran adaptive-RB seed0 50ep at `seed_0/catf_v2_adaptive_rb/`.
- Newly ran adaptive-RB seed1 50ep at `seed_1/catf_v2_adaptive_rb/`.
- Reused completed seed2 adaptive-RB 50ep from `outputs/experiments/catf_v2_adaptive_rb_seed2_50ep/` and wrote a reuse reference under `seed_2/catf_v2_adaptive_rb/reports/`.
- Training protocol stayed within the requested constraints: `yolo11n.pt`, safe tiled no-OK-position dataset, epochs=50, imgsz=1024, batch=2, workers=0, device=0, YOLO default augmentation enabled, no fixed augmented dataset, no copy-paste, train images=2301, val not augmented, and continuous 1..50 epoch results for completed runs.
- Fixed epoch5 is no longer treated as a hard augmentation start. It remains only `min_burnin_epoch=5`; adaptive burn-in decides whether candidate intervention is justified.

Metrics:

| Seed | Clean P | Clean R | Clean mAP50 | Clean mAP50-95 | Fixed P | Fixed R | Fixed mAP50 | Fixed mAP50-95 | Adaptive-RB P | Adaptive-RB R | Adaptive-RB mAP50 | Adaptive-RB mAP50-95 | Adaptive constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | false |
| 1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | 0.7550 | 0.7261 | 0.7653 | 0.4938 | true |
| 2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | false |

Gate and augmentation audit:

| Seed | adaptive start | candidate | rollback | accepted | no-op fallback | no-op epoch | industrial samples | ROI applied | router draws | active applied classes |
|---:|---:|---|---|---|---|---:|---:|---:|---:|---|
| 0 | None | false | false | false | true | 15 | 0 | 0 | 0 | [] |
| 1 | 15 | true | false | true | false | None | 17 | 20 | 1430 | [12] |
| 2 | None | false | false | false | true | 15 | 0 | 0 | 0 | [] |

Findings:

- Constraint result: adaptive-RB is `2/3` pass, not `3/3`.
- Seed0 passed constraints by reproducing clean native, but did not preserve fixed CATF-v2 mAP gains. The adaptive controller reached epoch15 with `metric_unstable` and `insufficient_diagnosis_evidence`, then entered strict no-op fallback.
- Seed1 started a low-risk candidate at epoch15, saved an RB checkpoint, and accepted the probe at epoch20. It improved Recall by +0.0784, mAP50 by +0.0111, and mAP50-95 by +0.0139 versus clean, but Precision dropped by -0.0175, so it fails the stated industrial constraint.
- Seed2 remained protected: no adaptive start, no candidate, no rollback required, no-op fallback at epoch15, and final metrics exactly match clean seed2.
- OK3 remained inactive and OK3 ROI applied total was 0 across all adaptive-RB runs.

Interpretation:

- Adaptive-RB successfully upgrades the method from fixed-time epoch5 triggering to adaptive candidate intervention and validates strong clean-baseline protection for seed2.
- Current adaptive-RB is not a final paper main method because seed0 loses useful fixed CATF-v2 gains and seed1 violates the precision constraint despite useful recall/mAP gains.
- Next parameter changes should focus on cumulative burn-in evidence for seed0 and a precision-aware RB accept gate for seed1. RB probe accept should enforce clean/reference industrial constraints, not only the immediate probe delta.

Reports:

- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_adaptive_rb/reports/multiseed_adaptive_rb_summary.md`
- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_adaptive_rb/reports/multiseed_adaptive_rb_summary.json`

Verification:

- `python -m py_compile scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py scripts/summarize_catf_v2_adaptive_rb_multiseed.py`
- `pytest -q tests/test_catf_v2_adaptive_burnin.py tests/test_catf_v2_rollback_controller.py tests/test_catf_v2_gated_controller.py tests/test_catf_v2_safe_controller.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_activation_rules.py tests/test_catf_v2_per_class_diagnosis.py tests/test_catf_v2_policy_matrix.py tests/test_catf_v2_sample_router.py tests/test_catf_v2_roi_augmentation.py tests/test_catf_v2_threshold_calibration.py tests/test_feedback_policy_guard.py tests/test_feedback_policy_controller.py tests/test_inloop_feedback_training.py tests/test_online_augmentation.py tests/test_proxy_prefilter.py tests/test_copy_paste.py`
- Result: `111 passed`.

### CATF-v2 Adaptive Burn-in Trigger and Seed2 Adaptive-RB Validation

Implemented adaptive burn-in for CATF-v2 and first-branch RB wiring. This keeps the project centered on diagnosis-driven augmentation and does not change the YOLO network.

Implementation:

- New mode: `--adaptive-burnin true`.
- New RB flag: `--catf-rollback-mode true`.
- New controller module: `AutoAugment/catf_v2/adaptive_burnin.py`.
- New rollback helper: `AutoAugment/catf_v2/rollback_controller.py`.
- New tests:
  - `tests/test_catf_v2_adaptive_burnin.py`
  - `tests/test_catf_v2_rollback_controller.py`
- Fixed `feedback_start_epoch=5` is no longer the conceptual augmentation start. It remains only the default earliest burn-in check point through `min_burnin_epoch=5`.
- Paper wording should not claim epoch5 is optimal. The method should be described as adaptive burn-in: start candidate augmentation only after the model is diagnosable, validation curves are sufficiently stable, and class evidence is credible.

Retrospective simulation:

- Script: `scripts/simulate_catf_v2_adaptive_burnin_retrospective.py`
- Outputs:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/adaptive_burnin_retrospective_simulation.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/adaptive_burnin_retrospective_simulation.json`
- Result:
  - seed0 adaptive start epoch: 15, low-risk RB candidate.
  - seed1 adaptive start epoch: 15, low-risk RB candidate.
  - seed2 candidate start: false.
  - seed2 no-op fallback epoch: 15.
  - seed2 strong clean-baseline protection: true.
  - Safe epoch5 early no-op is avoided because epoch5 is observe, not fallback.
  - Gated seed2 epoch10 late fallback is avoided because no candidate augmentation is started before seed2 protection.
- Overfit note: the controller uses metric stability, evidence/support guards, no-aug/high-FP/stable-class guards, and final clean-baseline protection rather than seed IDs or class IDs. Thresholds still need validation beyond seeds 0/1/2.

10ep smoke:

- Run: `outputs/experiments/catf_v2_adaptive_burnin_10ep_smoke/`
- Seed: 2
- Epochs: 10
- Configuration: `yolo11n.pt`, safe tiled no-OK-position dataset, imgsz=1024, batch=2, workers=0, device=0, YOLO default augmentation enabled, CATF-v2, adaptive burn-in, RB mode, class-aware feedback, ROI-aware augmentation, sample-aware routing.
- Epoch5 checked start_condition.
- Candidate branch started: false.
- Reasons: `metric_unstable`, `strong_clean_baseline_protection`.
- Strict no-op audit: industrial samples=0, ROI applied=0, CATF router random draw count=0.
- BBox/class legal: true.
- Report:
  - `outputs/experiments/catf_v2_adaptive_burnin_10ep_smoke/reports/adaptive_burnin_smoke_report.md`
  - `outputs/experiments/catf_v2_adaptive_burnin_10ep_smoke/reports/adaptive_burnin_events.json`

Seed2 adaptive-burnin + RB 50ep:

- Run: `outputs/experiments/catf_v2_adaptive_rb_seed2_50ep/`
- Seed: 2
- Epochs: 50
- `results.csv` epochs 1..50 continuous.
- Adaptive start epoch: `None`.
- Candidate branch started: false.
- No-op fallback epoch: 15.
- Rollback triggered: false, because no candidate branch was entered.
- Strict no-op audit: industrial samples=0, ROI applied=0, router random draw count=0.
- Train images=2301, no fixed augmented dataset, bbox/class legal=true.

Metrics:

| Run | Precision | Recall | mAP50 | mAP50-95 | constraint_failed |
|---|---:|---:|---:|---:|---|
| clean seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | false |
| fixed CATF-v2 seed2 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | true |
| Gated seed2 | 0.7850 | 0.6795 | 0.7521 | 0.5083 | true |
| Safe seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | false |
| Adaptive-RB seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | false |

Interpretation:

- Adaptive burn-in + RB is more methodologically defensible than fixed epoch5 triggering for seed2 because the system waits for diagnosability and blocks intervention under strong clean-baseline protection.
- The seed2 adaptive-RB run proves strict clean parity can be preserved without using Safe's hard epoch5 early abstention.
- Full multiseed adaptive-RB is recommended next. Seed0/seed1 need real validation to determine whether the low-risk epoch15 candidate path preserves fixed CATF-v2 gains.

Reports:

- `outputs/experiments/catf_v2_adaptive_rb_seed2_50ep/reports/adaptive_rb_seed2_50ep_report.md`
- `outputs/experiments/catf_v2_adaptive_rb_seed2_50ep/reports/adaptive_rb_seed2_50ep_summary.json`

Verification:

- `python -m py_compile scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py AutoAugment/catf_v2/adaptive_burnin.py AutoAugment/catf_v2/rollback_controller.py scripts/simulate_catf_v2_adaptive_burnin_retrospective.py scripts/summarize_catf_v2_adaptive_rb_seed2.py`
- `pytest -q tests/test_catf_v2_adaptive_burnin.py tests/test_catf_v2_rollback_controller.py tests/test_catf_v2_gated_controller.py tests/test_catf_v2_safe_controller.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_activation_rules.py tests/test_catf_v2_per_class_diagnosis.py tests/test_catf_v2_policy_matrix.py tests/test_catf_v2_sample_router.py tests/test_catf_v2_roi_augmentation.py tests/test_catf_v2_threshold_calibration.py tests/test_feedback_policy_guard.py tests/test_feedback_policy_controller.py tests/test_inloop_feedback_training.py tests/test_online_augmentation.py tests/test_proxy_prefilter.py tests/test_copy_paste.py`
- Result: `111 passed`.

### CATF-v2-Gated Controller and Full Multiseed Validation

Implemented `--catf-gated-mode true` as a separate CATF-v2 controller path. Safe mode was preserved and remains available through `--catf-safe-mode true`.

Controller behavior:

- epoch 5 is an observation point and does not fallback on no-gain alone.
- epoch 10 is the first formal gate.
- strict no-op fallback uses the existing CATF-v2 no-op policy path and keeps labels/Instances untouched after fallback.
- gate decisions are written into `policy_history`.

Retrospective simulation:

- Reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/gated_controller_retrospective_simulation.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/gated_controller_retrospective_simulation.json`
- Result: predicted seed0 no fallback, seed1 no fallback, seed2 fallback at epoch 10, expected `3/3` constraint pass.
- Overfit assessment in the report: no seed-id or class-id hardcoding, but the thresholds are still derived from the current three-seed failure analysis and need broader validation.

Full multiseed run:

- Run group: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/`
- Seeds: 0, 1, 2
- Training protocol: `yolo11n.pt`, `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`, epochs=50, imgsz=1024, batch=2, workers=0, device=0, seed-specific clean reference curves, YOLO default augmentation enabled, no fixed augmented dataset, no copy-paste, train images=2301.
- All Gated runs have `results.csv` epochs 1..50 continuous.

Metrics:

| Seed | Clean P | Clean R | Clean mAP50 | Clean mAP50-95 | Fixed P | Fixed R | Fixed mAP50 | Fixed mAP50-95 | Safe P | Safe R | Safe mAP50 | Safe mAP50-95 | Gated P | Gated R | Gated mAP50 | Gated mAP50-95 | Gated constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | false |
| 1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | false |
| 2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7850 | 0.6795 | 0.7521 | 0.5083 | true |

Findings:

- CATF-v2-Gated retained fixed CATF-v2 gains on seed0 and seed1.
- Seed0: industrial samples=41, ROI applied=45, no fallback.
- Seed1: industrial samples=55, ROI applied=56, no fallback.
- Seed2 triggered fallback at epoch 10 with reason `epoch10_bad_pattern_A`; from `active_policy_epoch_010.json` onward all op probabilities are 0.
- Seed2 still failed constraints: delta vs clean is Precision +0.0887, Recall -0.0491, mAP50 -0.0171, mAP50-95 -0.0141.
- Seed2 total pre-fallback augmentation was industrial samples=22, router draws=1540, ROI applied=25. The post-fallback no-op is strict, but it cannot undo the already-applied tentative augmentation.
- OK3 was never active and OK3 ROI applied was 0 across all Gated runs.
- Actual Gated constraint_failed count is `1/3`, so Gated did not achieve `3/3`.
- Critical interpretation: the retrospective simulation was overly optimistic because it treated epoch10 fallback as selecting clean fallback. In real single-run training, fallback at epoch10 cannot rewind model/optimizer/EMA state after epochs 6-10 have already updated under tentative augmentation.
- Recommendation: CATF-v2-Gated is not ready as the paper main method. It is useful evidence that seed0/seed1 gains can be preserved, but the seed2 result requires either pre-application gating or auditable in-run rollback before claiming a robust main method.

Reports:

- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/reports/multiseed_catf_v2_gated_summary.md`
- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/reports/multiseed_catf_v2_gated_summary.json`

Verification:

- `python -m py_compile AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py AutoAugment/catf_v2/gated_controller.py scripts/train_yolo_default_with_inloop_feedback.py scripts/simulate_catf_v2_gated_controller_retrospective.py scripts/summarize_catf_v2_gated_multiseed.py`
- `pytest -q tests/test_catf_v2_gated_controller.py tests/test_catf_v2_safe_controller.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_activation_rules.py tests/test_catf_v2_per_class_diagnosis.py tests/test_catf_v2_policy_matrix.py tests/test_catf_v2_sample_router.py tests/test_catf_v2_roi_augmentation.py tests/test_catf_v2_threshold_calibration.py tests/test_feedback_policy_guard.py tests/test_feedback_policy_controller.py tests/test_inloop_feedback_training.py tests/test_online_augmentation.py tests/test_proxy_prefilter.py tests/test_copy_paste.py`
- Result: `99 passed`.

## 2026-06-05

### CATF-v2-Safe Full Multiseed Validation

Completed full multiseed CATF-v2-Safe validation for seeds 0, 1, and 2.

Run group:

- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe/`

Execution:

- Newly ran CATF-v2-Safe seed0 and seed1 50ep under the required in-loop entrypoint.
- Reused the completed seed2 Safe 50ep result through `seed_2/catf_v2_safe/` link/copy artifacts.
- Training protocol: `yolo11n.pt`, `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`, epochs=50, imgsz=1024, batch=2, workers=0, device=0, seed-specific clean reference curves, YOLO default augmentation enabled, no fixed augmented dataset, no copy-paste, train images=2301.
- Each Safe run has `results.csv` epochs 1..50 continuous and no stage restart.

Metrics:

| Seed | Clean P | Clean R | Clean mAP50 | Clean mAP50-95 | Fixed P | Fixed R | Fixed mAP50 | Fixed mAP50-95 | Safe P | Safe R | Safe mAP50 | Safe mAP50-95 | Safe constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | false |
| 1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | false |
| 2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | false |

Findings:

- Safe vs clean deltas are `0.0000/0.0000/0.0000/0.0000` for Precision, Recall, mAP50, and mAP50-95 on all seeds.
- Safe constraint_failed count is `0/3`, improved from fixed CATF-v2 `1/3`.
- Safe reached `3/3` pass by early-abstention no-op fallback at epoch 5 on all seeds.
- Industrial samples augmented=0, ROI applied=0, and router random draws=0 for seed0/1/2.
- Active proposals before fallback: seed0 class 11/class 4, seed1 class 11/class 4, seed2 class 9.
- OK3 was never active and OK3 ROI applied was 0.
- Seed0 did not preserve fixed CATF-v2's mAP gains; seed1 did not preserve fixed CATF-v2's all-metric gains. Seed2 was protected exactly as intended.
- Recommendation: CATF-v2-Safe should be described as a conservative safety/protection variant or fallback layer. It is not strong enough as the sole paper main augmentation method because it removes valid seed0/seed1 fixed CATF-v2 gains.

Reports:

- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe/reports/multiseed_catf_v2_safe_summary.md`
- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe/reports/multiseed_catf_v2_safe_summary.json`

Verification:

- `python -m py_compile AutoAugment/catf_v2/sample_router.py`
- `python -m py_compile AutoAugment/catf_v2/policy_matrix.py`
- `python -m py_compile scripts/train_yolo_default_with_inloop_feedback.py`
- `pytest -q tests/test_catf_v2_safe_controller.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_activation_rules.py tests/test_catf_v2_per_class_diagnosis.py tests/test_catf_v2_policy_matrix.py tests/test_catf_v2_sample_router.py tests/test_catf_v2_roi_augmentation.py tests/test_catf_v2_threshold_calibration.py tests/test_feedback_policy_guard.py tests/test_feedback_policy_controller.py tests/test_inloop_feedback_training.py tests/test_online_augmentation.py tests/test_proxy_prefilter.py tests/test_copy_paste.py`
- Result: `89 passed`.

## 2026-06-04

### CATF-v2-Safe Seed2 Validation

Implemented CATF-v2-Safe and validated the known seed2 failure case. This does not change YOLO architecture and does not modify the existing CATF-v2 activation rules except when `--catf-safe-mode true` is explicitly enabled.

Safe controller mechanisms:

- high-recall baseline protection
- negative-effect attribution
- early abstention
- safe accept guard
- strict no-op fallback

Seed2 failure curve design report:

- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed2_safe_controller_design.md`
- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed2_safe_controller_design.json`

Key curve finding:

- fixed CATF-v2 seed2 first lags clean Recall at epoch 6.
- fixed CATF-v2 seed2 first lags clean mAP50-95 at epoch 8.
- feedback epoch 10 already shows mAP50-95 guard.
- epoch 5 proposed class 9 with no positive Recall/mAP gain, so Safe should abstain before ROI/industrial augmentation affects later training.

10ep smoke:

- Output: `outputs/experiments/catf_v2_safe_10ep_smoke/`
- Seed: 2
- Epochs: 10
- Train images: 2301
- Fixed augmented dataset generated: false
- BBox/class legal: true
- Safe event: `safe_accept_blocked`; no no-op freeze was expected as a formal conclusion because this smoke uses a 10ep schedule against a 50ep clean reference curve.

Seed2 50ep Safe run:

- Output: `outputs/experiments/catf_v2_safe_seed2_50ep/`
- Seed: 2
- Epochs: 50
- Single-run continuous: true
- YOLO default augmentation: enabled
- Train images: 2301
- Fixed augmented dataset generated: false
- Safe fallback: true
- Trigger: `early_abstention_no_recall_or_map_gain` at epoch 5
- Industrial samples augmented: 0
- ROI applied: 0
- Router random draw count: 0
- BBox/class legal: true

Metrics:

| Run | Precision | Recall | mAP50 | mAP50-95 | constraint_failed |
|---|---:|---:|---:|---:|---|
| clean seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | false |
| fixed CATF-v2 seed2 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | true |
| CATF-v2-Safe seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | false |

Interpretation:

- CATF-v2-Safe fixes the known seed2 failure by abstaining on a high-recall/high-mAP clean baseline.
- The result intentionally matches clean seed2 rather than forcing a risky precision-biased CATF-v2 update.
- The next validation should be full multiseed CATF-v2-Safe, not further unified RC threshold optimization.

Reports:

- `outputs/experiments/catf_v2_safe_seed2_50ep/reports/final_report.md`
- `outputs/experiments/catf_v2_safe_seed2_50ep/reports/final_metrics.json`
- `outputs/experiments/catf_v2_safe_seed2_50ep/reports/safe_controller_events.json`
- `outputs/experiments/catf_v2_safe_seed2_50ep/reports/compare_with_clean_and_fixed_catf_v2.md`

### Official-Path CATF-v2 Threshold Re-optimization

No training was run. Re-optimized per-class thresholds using existing official Ultralytics `YOLO.predict(conf=0.10)` outputs and the official-path post-processing evaluator, not the old cached post-hoc evaluator.

Inputs:

- Run group: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/`
- Existing official prediction outputs from `catf_v2_rc_official_path_validation/`
- Clean native official val metrics
- Fixed CATF-v2 official val metrics
- Old saved `catf_v2_rc_per_class_thresholds.json`

Script:

- `scripts/reoptimize_catf_v2_thresholds_official_path.py`

Pass counts:

| Scheme | Pass count | Note |
|---|---:|---|
| fixed CATF-v2 without RC | 2/3 | official Ultralytics val constraints |
| old unified RC | 1/3 | saved `catf_v2_rc_per_class_thresholds.json` |
| reoptimized unified RC | 2/3 | one threshold table for all seeds |
| per-seed RC | 3/3 | model/seed-specific deployment calibration |
| conservative default RC | 2/3 | limited threshold changes, high-FP-prior protected |

Key conclusions:

- There is no unified threshold table found in the official path that makes CATF-v2-RC pass 3/3 seeds.
- The best unified precision-guard table passes seed1 and seed2, but seed0 still fails the mAP50-95 guard by a small margin.
- Per-seed calibration can pass 3/3, so RC should be framed as deployment/model-specific calibration rather than the core training method.
- The old RC failed because it lowered many classes to `0.10`, including FP-sensitive classes, creating Precision failures on seed0 and seed2.
- Recommended paper mainline: fixed CATF-v2 is the training method; threshold calibration is a deployment-time companion and official-path ablation.

Outputs:

- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/official_threshold_reoptimization.md`
- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/official_threshold_reoptimization.json`
- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/unified_precision_guard_thresholds.json`
- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/per_seed_thresholds.json`
- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/conservative_default_thresholds.json`

### Fixed CATF-v2 Multiseed Validation

Ran fixed CATF-v2 seed0 and seed2 after the strict no-augmentation bypass repair, reusing clean native seed0/1/2 and the already completed fixed seed1 run.

Run group:

- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/`
- Code commit used for training: `dfcd177fa058046073e9b8e87dc8052b7b705f9a`

Shared configuration:

- Model: `yolo11n.pt`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Epochs: 50
- Image size: 1024
- Batch: 2
- Workers: 0
- Device: 0
- YOLO default augmentation: official Ultralytics default remained enabled
- CATF-v2: enabled
- Class-aware feedback, ROI-aware augmentation, sample-aware routing: enabled
- Fixed augmented dataset generated: false
- Train images: 2301
- Copy-paste: not enabled

Per-seed results:

| Seed | Clean P | Clean R | Clean mAP50 | Clean mAP50-95 | Fixed P | Fixed R | Fixed mAP50 | Fixed mAP50-95 | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | false |
| 1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | false |
| 2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | true |

Delta vs clean:

- seed0: ΔP=-0.0060, ΔR=-0.0068, ΔmAP50=+0.0090, ΔmAP50-95=+0.0136.
- seed1: ΔP=+0.0127, ΔR=+0.0528, ΔmAP50=+0.0284, ΔmAP50-95=+0.0390.
- seed2: ΔP=+0.0674, ΔR=-0.0423, ΔmAP50=-0.0110, ΔmAP50-95=-0.0257.

Aggregate and safety:

- Fixed CATF-v2 constraint_failed count: 1/3.
- Old CATF-v2 constraint_failed count: 2/3.
- OK3 active across fixed seeds: false.
- OK3 ROI applied total: 0.
- Active class counts: class 11 x3, class 4 x2, class 12 x1, class 9 x1, class 8 x1.
- ROI affected class counts: class 11=132, class 9=25, class 4=14, class 12=14, class 8=6.

Interpretation:

- Strict no-augmentation bypass repair improved multiseed constraint stability from 1/3 passing to 2/3 passing.
- The fixed CATF-v2 path is cleaner and more credible than the old CATF-v2 multiseed result.
- Seed2 still fails due mAP50 and mAP50-95 drops, so CATF-v2 should not yet be claimed as a fully stable sole main method.

Reports:

- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.md`
- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.json`

## 2026-06-03

### CATF-v2 Fixed Seed1 50 Epoch Rerun

Reran the critical seed1 CATF-v2 validation after fixing the formal no-augmentation transform bypass. This run is important because the old CATF-v2 seed1 result passed constraints while recording `ROI applied=0` and `industrial samples augmented=0`.

Run:

- Output: `outputs/experiments/catf_v2_fixed_seed1_50ep/`
- Entry: `scripts/train_yolo_default_with_inloop_feedback.py`
- Model: `yolo11n.pt`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Epochs: 50
- Seed: 1
- Batch: 2
- Workers: 0
- YOLO default augmentation: enabled
- CATF-v2: enabled
- Class-aware feedback, ROI-aware augmentation, sample-aware routing, and threshold calibration report: enabled
- Fixed augmented dataset generated: false
- Train images: 2301
- Epoch continuity: 1..50

Reference and result:

- Clean native seed1: Precision=0.7725, Recall=0.6477, mAP50=0.7542, mAP50-95=0.4799.
- Old CATF-v2 seed1: Precision=0.7691, Recall=0.6950, mAP50=0.7549, mAP50-95=0.4898.
- Fixed CATF-v2 seed1: Precision=0.7852, Recall=0.7005, mAP50=0.7826, mAP50-95=0.5189.
- Delta vs clean native seed1: Precision +0.0127, Recall +0.0528, mAP50 +0.0284, mAP50-95 +0.0390.
- Delta vs old CATF-v2 seed1: Precision +0.0161, Recall +0.0055, mAP50 +0.0277, mAP50-95 +0.0291.
- Constraint status: `constraint_failed=false`.

Augmentation and activation:

- Industrial samples augmented: 55.
- ROI applied: 56.
- ROI affected classes: class 11 (51), class 4 (5).
- Industrial ops applied: `local_contrast=28`, `sharpen_mild=27`.
- OK3 active: false.
- OK3 ROI applied: 0.
- Invalid bbox count: 0.
- Bbox out-of-bounds count: 0.
- Class id out-of-bounds count: 0.
- Policy actions: shrink=5, freeze=2, accept=1, observe=1.

Interpretation: the fixed seed1 improvement is not explained by the previously identified no-op label/Instances rewrite issue, because this rerun recorded actual CATF-v2 ROI/industrial augmentation.

Reports:

- `outputs/experiments/catf_v2_fixed_seed1_50ep/reports/final_report.md`
- `outputs/experiments/catf_v2_fixed_seed1_50ep/reports/final_metrics.json`
- `outputs/experiments/catf_v2_fixed_seed1_50ep/reports/compare_with_clean_native_seed1.md`

### CATF-v2 Strict No-Augmentation Bypass Fix

No training was run. Fixed CATF-v2 formal transform no-augmentation paths so they bypass label conversion, bbox validation/clip, and `Instances` rebuild unless an industrial/ROI augmentation is actually applied.

Implementation summary:

- `SampleAwareAugmentationRouter` now records `applied_any_aug`.
- No-active, empty-label, no_aug/stable-only, high-FP guarded-only, all-prob-zero, and ROI-unavailable paths return bypass results without bbox validation/clip.
- `UltralyticsOnlinePolicyTransform` returns the original YOLO label dict unchanged when `applied_any_aug=false`.
- ROI-unavailable precheck avoids probability draws when a ROI op cannot be applied.

Re-run transform parity audit:

- clean vs CATF-v2 noop final output identical: true
- clean vs CATF-v2 formal-force-skip final output identical: true
- bbox hash mismatches: 0 / 100
- raw vs formal-force-skip intermediate mismatches: 0 / 100
- formal-force-skip image/cls/Instances rewrites: 0 / 100
- router random draws: 0
- industrial/ROI applied ops: 0
- router bbox_oob/invalid/class-oob count: 0

Validation:

- `pytest -q tests/test_catf_v2_transform_bypass.py ... tests/test_copy_paste.py`: 81 passed.
- Syntax checks passed for CATF-v2 router/policy/threshold modules and in-loop training entrypoint.

Outputs:

- `outputs/audits/catf_v2_transform_parity/transform_parity_report.md`
- `outputs/audits/catf_v2_transform_parity/transform_parity.json`
- `outputs/audits/catf_v2_transform_parity/diff_samples/README.md`

### CATF-v2 Transform-Level Parity Audit

No training was run. Added and executed `scripts/audit_catf_v2_transform_parity.py` to compare clean native YOLO default transform output, CATF-v2 noop output, and CATF-v2 formal-force-skip output on 100 deterministic train samples from `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`.

Coverage:

- stable/no_aug samples: 42
- active defect samples: 23
- domain high-FP prior samples: 23
- low-support samples: 10
- multi-class samples: 38
- empty-label samples: 10

Result:

- clean vs CATF-v2 noop final output identical: true
- clean vs CATF-v2 formal-force-skip final output identical: false
- clean vs formal-force-skip final mismatches: 1 / 100
- raw vs formal-force-skip intermediate mismatches: 89 / 100
- formal-force-skip image/cls/Instances rewrites: 100 / 100
- router random draws: 0
- industrial/ROI applied ops: 0
- router `bbox_oob_count`: 1

The final mismatch was a class 4 `油污` sample where the raw bbox had `y2=1024.00048828125`; formal-force-skip route ran validation and clipped it to `1024.0`, causing a downstream final bbox hash difference even without applied augmentation. This can explain residual CATF-v2 path effects that are not reflected by applied-op statistics.

Outputs:

- `outputs/audits/catf_v2_transform_parity/transform_parity_report.md`
- `outputs/audits/catf_v2_transform_parity/transform_parity.json`
- `outputs/audits/catf_v2_transform_parity/diff_samples/`

Recommended next fix: bypass CATF-v2 transform before label conversion/validation unless an actual op is selected; force-skip/no-active/no_aug/stable/high-FP samples should return the native YOLO transform input unchanged.

## 2026-05-18

### Safe Tiled No OK/Position Baseline YOLO11n 50 Epoch

Ran the formal baseline on the audited safe tiled dataset with `OK` and `定位` removed. The first sandboxed launch failed during Ultralytics label-cache multiprocessing pipe creation with Windows permission error; the same train command was rerun outside the sandbox and completed. This was not CUDA OOM.

Dataset audit:

- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Safe tiled dataset: true
- Removed classes: `OK`, `定位`
- Retained classes: `OK2`, `OK3`, `加强筋打伤`, `开裂`, `油污`, `浅划伤`, `漏背锡`, `碰伤`, `脏污`, `轮廓划伤`, `锡丝残留`, `锡尖`, `锡膏`
- Train/val tiles: 2301 / 677
- Train/val bboxes: 3182 / 905
- Total bboxes: 4087
- Class ids out of range: false

Training command:

```powershell
yolo detect train model=yolo11n.pt data=outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml epochs=50 imgsz=1024 batch=2 workers=0 device=0 project=outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep name=train exist_ok=True mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0
```

Validation command:

```powershell
yolo detect val model=outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/best.pt data=outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml imgsz=1024 batch=2 workers=0 device=0 project=outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep name=val exist_ok=True
```

Result:

- Completed 50 epochs: true
- Final batch: 2
- OOM: false
- Training time: 2.826 hours
- Precision: 0.690
- Recall: 0.615
- mAP50: 0.669
- mAP50-95: 0.434
- Lowest Recall class: `开裂` (0.000)
- best.pt: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/best.pt`
- last.pt: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/last.pt`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_report.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_metrics.json`

### Filtered Baseline Dataset Without OK And 定位

No training was run. Built a filtered dataset from `outputs/datasets/tiled/tiled_1024_ov20_full_safe/` that removes only `OK` and `定位`, keeps `OK2` and `OK3`, and remaps class ids to `0..12`.

Build command:

```powershell
python scripts\filter_tiled_dataset.py --source-root outputs\datasets\tiled\tiled_1024_ov20_full_safe --output-root outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position --debug-limit 50 --overwrite
```

Outputs:

- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/`
- Data YAML: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Class filter report: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/class_filter_report.md`
- Class filter report JSON: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/class_filter_report.json`
- Dataset summary: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/dataset_summary.md`
- Debug samples: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/debug_samples/` (50 images)

Filter result:

- Deleted classes: `OK`, `定位`
- Kept classes: `OK2`, `OK3`, `加强筋打伤`, `开裂`, `油污`, `浅划伤`, `漏背锡`, `碰伤`, `脏污`, `轮廓划伤`, `锡丝残留`, `锡尖`, `锡膏`
- New class ids: `0..12`
- Source train/val images: 2452 / 677
- Filtered train/val images: 2301 / 677
- Source bbox count: 4269
- Filtered bbox count: 4087
- Train bbox count before/after: 3337 / 3182
- Val bbox count before/after: 932 / 905
- Train empty tiles retained: 210
- Val empty tiles retained: 83
- Class id out of range: false
- Chinese class names damaged: false
- Formal baseline ready: true

Verification:

- Syntax check passed for `scripts/filter_tiled_dataset.py`.
- `pytest -q tests\test_filter_tiled_dataset.py` passed: 1 test.

### Tiling Quality Audit And Safe Full Dataset

No training or YOLO validation was run. Audited the previously built `tiled_1024_ov20_full` dataset for partial-object bbox risk, then rebuilt a stricter safe tiled dataset.

Old full tiled quality audit command:

```powershell
python scripts\audit_tiling_quality.py
```

Audit outputs:

- `outputs/audits/tiling_quality/tiling_quality_audit.md`
- `outputs/audits/tiling_quality/tiling_quality_audit.json`
- `outputs/audits/tiling_quality/debug_truncated_bboxes/` (50 images)

Old full tiled audit result:

- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full/`
- Total bboxes: 7465
- Visibility < 0.5: 1091
- Visibility < 0.7: 1956
- Visibility < 0.8: 2405
- Visibility < 0.9: 2874
- Bboxes touching tile boundary: 3249 (43.52%)
- Border-truncated bboxes: 3197 (42.83%)
- Severe truncated bboxes with visibility < 0.7: 1956 (26.20%)
- Status: unsafe for formal baseline; retained only for audit.

Safe rebuild command:

```powershell
python scripts\build_yolo_tiled_dataset.py --dataset-root E:\TJGY\DataSet2_fixed --data-yaml E:\TJGY\DataSet2_fixed\data.yaml --output-dir outputs\datasets\tiled\tiled_1024_ov20_full_safe --tile-size 1024 --overlap 0.2 --min-visibility 0.7 --large-object-min-visibility 0.9 --drop-border-truncated True --border-margin 2 --require-box-center-inside True --keep-empty-ratio 0.1 --seed 42 --debug-limit 100 --overwrite
```

Safe dataset outputs:

- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/`
- Data YAML: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/data.yaml`
- Dataset summary: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/dataset_summary.md`
- Dataset report: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/tiled_dataset_report.md`
- Dataset report JSON: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/tiled_dataset_report.json`
- Debug tile bbox visualizations: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/debug_tiling/` (100 images)

Safe dataset result:

- Tiled images: 2452 train, 677 val
- Original bboxes: 3084
- Safe tiled bboxes: 4269
- Dropped bbox candidates after tile intersection: 13826
- Visibility-failed dropped candidates: 13346
- Border-truncated dropped candidates: 3961
- Center-outside dropped candidates: 9855
- Obvious half-target bbox remains: false
- Class id out of range: false
- Chinese class names damaged: false
- Caveat: class `定位` has 0 retained bboxes under the strict large-structure rule.
- Policy: subsequent formal baseline runs must use `outputs/datasets/tiled/tiled_1024_ov20_full_safe/data.yaml`.

Verification:

- Syntax check passed for `scripts/build_yolo_tiled_dataset.py` and `scripts/audit_tiling_quality.py`.
- `pytest -q tests\test_build_yolo_tiled_dataset.py` passed: 2 tests.

### Full Tiled Dataset Build And Mapping Audit

No training was run. Built the full tiled YOLO dataset from `E:\TJGY\DataSet2_fixed` without `--max-images-per-split`.

Build command:

```powershell
python scripts\build_yolo_tiled_dataset.py --dataset-root E:\TJGY\DataSet2_fixed --data-yaml E:\TJGY\DataSet2_fixed\data.yaml --output-dir outputs\datasets\tiled\tiled_1024_ov20_full --tile-size 1024 --overlap 0.2 --min-visibility 0.3 --keep-empty-ratio 0.1 --seed 42 --debug-limit 30 --overwrite
```

Outputs:

- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full/`
- Data YAML: `outputs/datasets/tiled/tiled_1024_ov20_full/data.yaml`
- Dataset summary: `outputs/datasets/tiled/tiled_1024_ov20_full/dataset_summary.md`
- Dataset report: `outputs/datasets/tiled/tiled_1024_ov20_full/tiled_dataset_report.md`
- Dataset report JSON: `outputs/datasets/tiled/tiled_1024_ov20_full/tiled_dataset_report.json`
- Debug tile bbox visualizations: `outputs/datasets/tiled/tiled_1024_ov20_full/debug_tiling/` (30 images)

Build result:

- Original images: 461 train, 116 val
- Tiled images: 4155 train, 1098 val
- Original bboxes: 3084
- Tiled bboxes: 7465
- Empty tiles retained: 478
- Dropped bboxes in retained tiles: 22202
- Drop reasons: `below_min_visibility=6212`, `outside_tile=15990`
- `data.yaml`: `nc=15`, names inherited from original with `yaml.safe_dump(..., allow_unicode=True)`
- Tiled class id range: 0..14
- Class id >= nc: none
- Chinese class names damaged: false

Audit command:

```powershell
python scripts\audit_dataset_mapping.py
```

Audit outputs:

- `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.md`
- `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.json`

Audit result:

- Full source coverage confirmed: 461 train and 116 val source images.
- Tiled names match the original `data.yaml` exactly.
- No class id >= nc and no negative class id found.
- Chinese class names are intact.
- Superseded by the tiling quality audit above: this dataset is now marked unsafe for formal baseline because partial-object bboxes were retained.

Verification:

- Syntax check passed for `scripts/build_yolo_tiled_dataset.py` and `scripts/audit_dataset_mapping.py`.
- `pytest -q tests\test_build_yolo_tiled_dataset.py` passed: 2 tests.

## 2026-05-17

### Artifact Cleanup And Output Normalization

Paused training and normalized generated artifacts.

- Added `scripts/audit_artifacts.py`.
- Generated inventory at `outputs/audits/artifact_inventory/artifact_inventory.md` and `.json`.
- Generated cleanup summary at `outputs/audits/artifact_inventory/cleanup_summary.md`.
- Added output convention at `docs/output_convention.md`.
- Archived legacy root-level outputs to `outputs/archive/old_outputs_20260517/`.
- Archived legacy `runs/detect/*` outputs to `outputs/archive/old_runs_20260517/runs_detect/`.
- `runs/detect` has no remaining old experiment entries.
- Active tiled smoke dataset is now `outputs/datasets/tiled/tiled_1024_ov20_smoke/`.
- Active 20 epoch baseline is now `outputs/experiments/20260517_tiled_baseline_20epoch/`.
- GPU preflight reports are now `outputs/audits/gpu_preflight/gpu_preflight_report.md` and `.json`.
- Project snapshots are now written to `outputs/snapshots/project_snapshot_latest.md` and `outputs/project_snapshot_latest.md`.

Policy recorded:

- Formal experiments must use `project=outputs/experiments/<run_id>`.
- Formal diagnostic runs must pass `--run-id` or explicitly set `--output-dir`.
- `runs/` is not a formal result location.
- Weights, generated images, generated dataset image/label files, and archive contents are local artifacts and are ignored by Git.

### Dataset Mapping And Class Distribution Audit

No training was run. Audited `E:\TJGY\DataSet2_fixed` and `outputs/datasets/tiled/tiled_1024_ov20_smoke/`.

Outputs:

- `outputs/audits/dataset_mapping/dataset_mapping_audit.md`
- `outputs/audits/dataset_mapping/dataset_mapping_audit.json`

Code updates:

- Added `scripts/audit_dataset_mapping.py`.
- Updated `scripts/build_yolo_tiled_dataset.py` so tiled `data.yaml` is written with `yaml.safe_dump(..., allow_unicode=True)`.
- Repaired `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml` to fully inherit names from `E:\TJGY\DataSet2_fixed\data.yaml`.

Findings:

- Original data.yaml: `nc=15`, class ids in labels are 0..14, no class id >= nc.
- Tiled smoke data.yaml: `nc=15`, class ids in labels are 0..14, no class id >= nc.
- Original data: 461 train images, 116 val images, 2433 train bboxes, 651 val bboxes.
- Tiled smoke data: 107 train tiles, 86 val tiles, 232 train bboxes, 143 val bboxes.
- Tiled smoke is not full: it uses 16 source images and the dataset summary records the per-split source cap.
- Existing 20 epoch `blank-or-unrendered` rows are from the prior corrupted/non-renderable tiled class names in saved val logs/metrics, not from invalid class ids.
- Low mAP is mainly from smoke subset imbalance and underrepresented classes: 开裂, 漏背锡, 碰伤, 轮廓划伤, 锡丝残留, and 锡膏 have recall 0 in the saved per-class metrics.

### Tiled Baseline 20 Epoch

Ran the tiled baseline on GPU with no external diagnostic augmentation and YOLO built-in augmentation knobs disabled.

Status after normalization:

- Run ID: `20260517_tiled_baseline_20epoch`
- Formal result: no
- Reason: this used the smoke tiled dataset, not the future full tiled dataset.
- Result path: `outputs/experiments/20260517_tiled_baseline_20epoch/`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml`
- Train command: `outputs/experiments/20260517_tiled_baseline_20epoch/configs/train_command.txt`
- Val command: `outputs/experiments/20260517_tiled_baseline_20epoch/configs/val_command.txt`
- Report: `outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_report.md`
- Metrics: `outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_metrics.json`
- best.pt: `outputs/experiments/20260517_tiled_baseline_20epoch/train/weights/best.pt`
- last.pt: `outputs/experiments/20260517_tiled_baseline_20epoch/train/weights/last.pt`

Normalized train command:

```powershell
D:\Anaconda\Scripts\conda.exe run -n pytorch yolo detect train model=yolo11n.pt data=E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_smoke\data.yaml epochs=20 imgsz=1024 batch=2 workers=0 device=0 project=outputs\experiments\20260517_tiled_baseline_20epoch name=train exist_ok=True mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0
```

Normalized validation command:

```powershell
D:\Anaconda\envs\pytorch\Scripts\yolo.exe detect val model=outputs\experiments\20260517_tiled_baseline_20epoch\train\weights\best.pt data=E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_smoke\data.yaml imgsz=1024 batch=2 workers=0 device=0 project=outputs\experiments\20260517_tiled_baseline_20epoch name=val exist_ok=True
```

Result:

- Completed 20 epochs: true
- Final train batch: 2
- Final val batch: 2
- OOM: false
- Precision: 0.828
- Recall: 0.213
- mAP50: 0.247
- mAP50-95: 0.181
- Train duration: 312.4 seconds
- Validation duration: 20.137 seconds

Note:

- Training artifacts and epoch 20 metrics were produced successfully.
- The conda wrapper emitted a UnicodeEncodeError while forwarding YOLO stdout after training; this was not a CUDA OOM and did not prevent `best.pt`, `last.pt`, or `results.csv` from being created.

### GPU Environment Confirmation

GPU environment has been confirmed in the dedicated conda env `pytorch`. Future formal training must use this environment rather than base.

- Conda env: `pytorch`
- Python executable: `D:\Anaconda\envs\pytorch\python.exe`
- Python version: 3.9.19
- PyTorch: 2.4.1
- `torch.cuda.is_available()`: True
- `torch.version.cuda`: 12.4
- CUDA device count: 1
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Ultralytics: 8.3.221

Training rule:

- CPU is only for smoke/debug runs.
- Formal YOLO training must use `device=0`.
- The base conda environment must not be used for formal training because it previously resolved to CPU-only PyTorch.

### GPU Preflight In Pytorch Env

Command:

```powershell
D:\Anaconda\Scripts\conda.exe run -n pytorch python scripts\run_gpu_preflight.py
```

Outputs:

- `outputs/audits/gpu_preflight/gpu_preflight_report.md`
- `outputs/audits/gpu_preflight/gpu_preflight_report.json`

Result:

- Conda env: `pytorch`
- `sys.executable`: `D:\Anaconda\envs\pytorch\python.exe`
- PyTorch: 2.4.1
- `torch.cuda.is_available()`: True
- `torch.version.cuda`: 12.4
- CUDA device count: 1
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Ultralytics: 8.3.221
- `yolo checks`: passed
- YOLO GPU smoke: passed

YOLO GPU smoke used the tiled smoke `data.yaml`, `epochs=1`, `imgsz=640`, `batch=1`, `workers=0`, `device=0`, and YOLO built-in augmentations disabled.

### Historical Base Env GPU Preflight

Historical base-env preflight confirmed that base was not a valid formal training environment.

- Python: 3.12.4
- PyTorch: 2.4.1+cpu
- `torch.cuda.is_available()`: False
- `torch.version.cuda`: None
- CUDA device count: 0
- PyTorch device name: NO CUDA
- `nvidia-smi`: NVIDIA GeForce RTX 3060 Laptop GPU, driver 560.81, 6144 MiB
- Ultralytics: 8.4.48
- `yolo checks`: passed, but reported CPU / GPU None / CUDA None
- YOLO GPU smoke: not run because CUDA was unavailable to PyTorch

Conclusion:

- `torch.cuda.is_available() = False` in base.
- Only CPU smoke/debug can run in base.
- Base results cannot be used as formal experiment results.

## 2026-05-16

### Code Changes

- Added tiled YOLO dataset builder at `scripts/build_yolo_tiled_dataset.py`.
- Added normalized `diagnosis_vector` to diagnosis outputs.
- Replaced fixed policy if-else mapping with severity-score dynamic formulas.
- Added proxy `SafetyScore` and converted ordinary bbox safety misses into soft penalties.
- Added copy-paste filter audit with before/after bbox counts, original retention, new bbox validity, total bbox validity, invalid/out-of-bounds/class-range counts, and debug images.
- Added strategy memory JSONL append and memory-guided reranking.
- Added metric consistency audit.

### Verification

- `pytest -q tests/test_build_yolo_tiled_dataset.py tests/test_copy_paste.py tests/test_proxy_prefilter.py tests/test_yolo_error_analysis.py`
  - Result: 22 passed.
- `pytest -q`
  - Result: 89 passed.

### Tiled Dataset Smoke

The smoke tiled dataset was originally built from `E:\TJGY\DataSet2_fixed` with `--max-images-per-split 8`, so it is explicitly a smoke dataset.

Current normalized outputs:

- `outputs/datasets/tiled/tiled_1024_ov20_smoke/`
- `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml`
- `outputs/datasets/tiled/tiled_1024_ov20_smoke/tiled_dataset_report.json`
- `outputs/datasets/tiled/tiled_1024_ov20_smoke/tiled_dataset_report.md`
- `outputs/datasets/tiled/tiled_1024_ov20_smoke/dataset_summary.md`

Summary:

- Source images: 16
- Output tiles: 193
- Retained bboxes: 375
- Empty tiles: 18

### Real Tiled Pipeline Smoke

The older CPU smoke pipeline output was archived to `outputs/archive/old_outputs_20260517/diagnostic_aug_tiled_smoke/`.

Key conclusions from that archived smoke:

- `copy_paste` was not hard filtered; soft penalties recorded bbox safety risks.
- Metric consistency audit reported zero precision/recall delta for this smoke and documented threshold/matching causes for expected metric differences.
- Strategy memory appended one record to the archived `strategy_memory.jsonl`.

## 2026-05-15

### Code Refactor

- Added the diagnosis-driven augmentation pipeline modules under `AutoAugment/diagnostic_pipeline/`.
- Added a unified pipeline entrypoint at `scripts/run_diagnostic_augmentation_pipeline.py`.
- Updated documentation and README to reflect the new research framing.
- Fixed the proxy-prefilter cleanup path in `AutoAugment/search/random_search.py`.
- Added a minimal executable same-image bbox-level `copy_paste` augmentation.
- Connected `copy_paste` to diagnosis-driven `small_object_low_recall` and `class_imbalance` policy generation.
- Added runtime policy-op validation so unavailable augmentations fail explicitly.

### Verification

- Syntax check on new pipeline modules passed.
- Regression tests passed:
  - `pytest -q tests/test_yolo_error_analysis.py tests/test_proxy_prefilter.py tests/test_yolo_train_evaluator.py`
  - `pytest -q tests/test_copy_paste.py`
  - `pytest -q tests/test_augmentations.py tests/test_copy_paste.py tests/test_yolo_error_analysis.py tests/test_proxy_prefilter.py tests/test_yolo_train_evaluator.py`
- Dry-run pipeline invocation passed and produced an archived smoke output under `outputs/archive/old_outputs_20260517/diagnostic_aug_pipeline_smoke/`.

<!-- DIAGAUG_50EP_START -->
## Diagnosis-Driven Augmentation 50 Epoch Result

- Run ID: `20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Baseline best.pt: `E:/TJGY/MinPaper/MyAutoAugument/outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/best.pt`
- Selected policy: `diag_policy_001` from `low_contrast_missed_defect`
- Selected policy contains copy_paste: `false`
- Copy-paste candidates retained in proxy ranking: `2`
- Copy-paste hard rejected: `false`
- Augmented train images / bboxes: `4602` / `6364`
- Precision: `0.686` (-0.004 vs baseline)
- Recall: `0.688` (+0.073 vs baseline)
- mAP50: `0.717` (+0.048 vs baseline)
- mAP50-95: `0.496` (+0.062 vs baseline)
- OOM: `false`
- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep\train\weights\best.pt`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/diagaug_50ep_report.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/diagaug_50ep_metrics.json`
- Baseline comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/baseline_vs_diagaug.md`
<!-- DIAGAUG_50EP_END -->

<!-- YOLO_DEFAULT_AUG_50EP_START -->
## YOLO Default Augmentation 50 Epoch Control

- Run ID: `20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Model/settings: `yolo11n.pt epochs=50 imgsz=1024 batch=2 workers=0 device=0`
- YOLO default augmentations enabled; actual args recorded from `train/args.yaml`.
- Precision: `0.785` (+0.095 vs baseline, +0.099 vs DiagAug)
- Recall: `0.676` (+0.061 vs baseline, -0.012 vs DiagAug)
- mAP50: `0.735` (+0.066 vs baseline, +0.018 vs DiagAug)
- mAP50-95: `0.476` (+0.042 vs baseline, -0.020 vs DiagAug)
- OOM: `false`
- Training wall time: `13154.0s (3.65h)`
- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep\train\weights\best.pt`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/reports/yolo_default_aug_50ep_report.md`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/reports/compare_baseline_yolo_default_diagaug.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/reports/yolo_default_aug_50ep_metrics.json`
<!-- YOLO_DEFAULT_AUG_50EP_END -->

<!-- BASELINE_POLICY_TOP3_START -->
## Baseline Diagnosis Top3 Candidate Policies

- Run ID: `20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline`
- Baseline best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep\train\weights\best.pt`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Triggered issues: `low_contrast_missed_defect, low_contrast_missed_defect, class_imbalance, localization_bias`
- Candidate policies generated: `5`
- Top3 proxy policies: `diag_policy_001, diag_policy_005, diag_policy_002`
- Top3 containing copy_paste: `diag_policy_005`
- Scope: single-round baseline diagnosis only; not multi-round optimization.
- Training status: no YOLO train, no final 50 epoch train, no top3 short-training.
- Next step for final strategy selection: run short-training for all top3 and select by short_train_score.
- Trace report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/reports/policy_selection_trace.md`
- Top3 report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/top3_policies/top3_policies.md`
- Proxy ranking: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/proxy/proxy_ranking.json`
<!-- BASELINE_POLICY_TOP3_END -->

<!-- TOP3_POLICY_SHORTTRAIN_START -->
## Top3 Policy Short-Training Validation

- Run ID: `20260518_tiled1024_safe_no_ok_position_policy_top3_shorttrain`
- Source top3 run: `20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline`
- Scope: top3 short-training strategy validation only; not a final model result.
- Each policy trained for `5` epochs with YOLO built-in augmentations disabled.
- Top3 came from one baseline diagnosis and proxy/safety ranking, not multi-round closed-loop search.
- Short-training scores: `diag_policy_001=0.609204, diag_policy_005=0.578862, diag_policy_002=0.595412`
- Best short-training policy: `diag_policy_001`
- Matches current formal DiagAug policy: `true`
- Result should decide whether a formal DiagAug 50 epoch rerun is needed.
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_shorttrain/reports/top3_policy_shorttrain_report.md`
- Results JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_shorttrain/reports/top3_policy_shorttrain_results.json`
<!-- TOP3_POLICY_SHORTTRAIN_END -->

<!-- COUNTERFACTUAL_DIAGNOSIS_START -->
## Counterfactual Diagnosis for Baseline Missed Defects

- Run ID: `20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis`
- Scope: prediction-only counterfactual diagnosis; no training, no 50 epoch run, no top3 short-training.
- Purpose: validate whether baseline FN cases respond to low-contrast/brightness-style transforms.
- Baseline FN count: `215`
- Tested FN count: `200`
- Highest recovery transform: `sharpen_mild` recovery_rate=`0.0700`
- diag_policy_001 unique photometric FN recovery rate: `0.1050`
- Supports low_contrast_missed_defect -> diag_policy_001: `True`
- copy_paste is not directly testable by prediction-only counterfactual inference.
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/reports/counterfactual_diagnosis_report.md`
- Summary JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/counterfactual_summary.json`
- Instance table: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/counterfactual_instances.csv`
<!-- COUNTERFACTUAL_DIAGNOSIS_END -->

<!-- CLASS_AWARE_DIAGNOSIS_START -->
## Class-Aware Per-Class Diagnosis and Policy Generation

- Run ID: `20260518_tiled1024_safe_no_ok_position_per_class_diagnosis`
- Scope: upgraded diagnosis and policy generation only; no training, no 50 epoch run, no short-training.
- Method upgrade: global policy selection -> class-aware error attribution policy generation.
- Inputs: baseline 50 epoch metrics, baseline diagnosis, and counterfactual diagnosis.
- Policy generated: `class_aware_policy_001` with branches `photometric_branch, copy_paste_branch, texture_branch, localization_branch`
- final_policy_score: `0.570452`
- Next step: run short-training for `class_aware_policy_001` before any formal 50 epoch rerun.
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_per_class_diagnosis/reports/per_class_diagnosis_report.md`
- Policy: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_per_class_diagnosis/policies/class_aware_mixed_policy.json`
- Score: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_per_class_diagnosis/policies/class_aware_policy_score.json`
<!-- CLASS_AWARE_DIAGNOSIS_END -->

<!-- CLASS_AWARE_POLICY_SHORTTRAIN_START -->
## Class-Aware Policy Short-Training Validation

- Run ID: `20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain`
- Scope: top1 class-aware mixed policy short-training only; no formal 50 epoch training.
- Policy: `class_aware_policy_001`
- Train images / bboxes: `4602` / `6412`
- Precision: `0.626`
- Recall: `0.680`
- mAP50: `0.685`
- mAP50-95: `0.456`
- short_train_score: `0.603477`
- Beats diag_policy_001 short-training score `0.609204`: `false`
- Recommend formal 50 epoch rerun: `false`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain/reports/class_aware_shorttrain_report.md`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain/reports/class_aware_vs_diag_policy_001_shorttrain.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain/metrics/class_aware_shorttrain_metrics.json`
<!-- CLASS_AWARE_POLICY_SHORTTRAIN_END -->

<!-- RANDOM_EXTERNAL_AUG_50EP_START -->
## Random External Augmentation 50 Epoch Control

- Run ID: `20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep`
- Scope: random external augmentation control group; not diagnosis-driven.
- Training set: original train images + 1x random augmented train images.
- YOLO built-in augmentations: disabled to match DiagAug final training.
- Random policy: `random_external_policy_seed42` with ops `sharpen(p=0.238,s=0.343), brightness(p=0.681,s=0.414), cutout(p=0.151,s=0.140), horizontal_flip(p=0.448,s=1.000)`
- Train images / bboxes: `4602` / `6364`
- Safety: hard_filter_pass=`true`, bbox_valid_rate=`1.000000`
- Precision: `0.750`
- Recall: `0.668`
- mAP50: `0.734`
- mAP50-95: `0.501`
- Delta vs DiagAug: P `+0.064`, R `-0.020`, mAP50 `+0.017`, mAP50-95 `+0.005`
- OOM: `false`
- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep\train\weights\best.pt`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/reports/random_external_aug_50ep_report.md`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/reports/compare_baseline_yolo_default_diagaug_random.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/reports/random_external_aug_50ep_metrics.json`
<!-- RANDOM_EXTERNAL_AUG_50EP_END -->

<!-- DIAGNOSIS_GUIDED_POLICY_SEARCH_START -->
## Diagnosis-Guided Policy Search

- Run ID: `20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search`
- Scope: diagnosis-guided sampled policy search plus 5 epoch short-training; no formal 50 epoch training.
- Change in method: diagnosis adjusts operation sampling probabilities instead of directly selecting a fixed policy.
- Candidate policies: `30`
- Proxy pass count: `20`
- Short-training trials: `10`
- Best balanced-score policy: `search_policy_017` balanced=0.632550 P/R/mAP50/mAP50-95=0.697/0.691/0.720/0.468
- Beats diag_policy_001 short-training balanced score: `true`
- Recommend formal 50 epoch: `true`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/reports/policy_search_report.md`
- Results JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/reports/policy_search_results.json`
- Best summary: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/reports/best_policy_summary.md`
<!-- DIAGNOSIS_GUIDED_POLICY_SEARCH_END -->

<!-- SEARCH_POLICY_017_50EP_START -->
## search_policy_017 Formal 50 Epoch Result

- Run ID: `20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep`
- Scope: diagnosis-guided policy search winner promoted to formal 50 epoch YOLO training.
- Training set: original train images + 1x `search_policy_017` augmented train images.
- YOLO built-in augmentations: disabled.
- Policy: `gaussian_noise(p=0.189, s=0.104), gamma(p=0.475, s=0.456), local_contrast(p=0.371, s=0.179), cutout(p=0.142, s=0.163), copy_paste(p=0.621, s=0.273)`
- Train images / bboxes: `4602` / `6408`
- Safety: bbox_valid_rate=`0.999380`, image_failures=`0`, label_failures=`0`
- Precision: `0.710`
- Recall: `0.616`
- mAP50: `0.681`
- mAP50-95: `0.474`
- Balanced score: `0.608450`
- Delta vs baseline: P `+0.020`, R `+0.001`, mAP50 `+0.012`, mAP50-95 `+0.040`
- Delta vs diag_policy_001: P `+0.024`, R `-0.072`, mAP50 `-0.036`, mAP50-95 `-0.022`
- Delta vs random external: P `-0.040`, R `-0.052`, mAP50 `-0.053`, mAP50-95 `-0.027`
- Exceeds random external by mAP50-95: `false`
- Exceeds diag_policy_001 by mAP50-95: `false`
- Current best formal by mAP50-95: `false`
- Current best formal by balanced score: `false`
- OOM: `false`
- Training wall seconds: `34573.7`
- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep\train\weights\best.pt`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/reports/search_policy_017_50ep_report.md`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/reports/compare_baseline_diagaug_random_yolo_search017.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/reports/search_policy_017_50ep_metrics.json`
<!-- SEARCH_POLICY_017_50EP_END -->

<!-- ONLINE_AUG_SMOKE_START -->
## Online Policy Augmentation Smoke

- Run ID: `online_yolo_like_base_smoke`
- New entrypoint: `scripts/train_yolo_online_aug.py`.
- Method change: custom policy is applied dynamically inside the YOLO training dataloader instead of building a fixed offline augmented dataset.
- Train image count: `2301`; no train image doubling.
- Fixed augmented dataset generated: `false`
- YOLO built-in augmentation mode for this smoke: disabled, so this is `only_custom_online_aug`.
- Online copy-paste: pending object-bank implementation; copy_paste ops are skipped safely for now.
- 1 epoch smoke train success: `true`
- 1 epoch smoke val success: `true`
- Val P/R/mAP50/mAP50-95: `0.4922/0.2364/0.1763/0.0986`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\online_yolo_like_base_smoke\previews`
- Report: `outputs/experiments/online_yolo_like_base_smoke/reports/online_aug_smoke_report.md`
- Stats JSON: `outputs/experiments/online_yolo_like_base_smoke/reports/online_aug_stats.json`
- Next step: inspect smoke safety/history and tune the feedback controller before any formal 50 epoch experiment.
<!-- ONLINE_AUG_SMOKE_END -->

<!-- ONLINE_DIAG_POLICY_001_50EP_START -->
## Online Diag Policy 001 50 Epoch

- Run ID: `20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep`
- Entrypoint: `scripts/train_yolo_online_aug.py`.
- Mechanism: custom policy is sampled online in the YOLO training dataloader; no fixed augmented dataset is built.
- Train image count: `2301`; no train image doubling.
- Fixed augmented dataset generated: `false`
- Validation custom augmentation: `false`; val uses original val tiles.
- YOLO built-in augmentation: disabled for `only_custom_online_aug`.
- Online copy-paste: pending object-bank implementation; copy_paste ops are skipped safely.
- Train success: `true`
- Val success: `true`
- Val P/R/mAP50/mAP50-95: `0.7297/0.6772/0.6814/0.4607`
- Online better than offline DiagAug by mAP50-95: `false`
- Online close to YOLO default by mAP50-95 within 0.03: `true`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/reports/online_diag_policy_001_50ep_report.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/reports/online_diag_policy_001_50ep_metrics.json`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/reports/compare_online_offline_yolo_default_random.md`
- Stats JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/reports/online_aug_stats.json`
<!-- ONLINE_DIAG_POLICY_001_50EP_END -->

<!-- ONLINE_RANDOM_LIKE_50EP_START -->
## Online Random-Like 50 Epoch

- Run ID: `20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep`
- Entrypoint: `scripts/train_yolo_online_aug.py`.
- Policy: online random-like mix of `sharpen_mild`, `brightness`, `cutout_safe`, and `horizontal_flip`.
- Mechanism: policy is sampled online in the YOLO training dataloader; no fixed augmented dataset is built.
- Train image count: `2301`; no train image doubling.
- Fixed augmented dataset generated: `false`
- Validation custom augmentation: `false`; val uses original val tiles.
- YOLO built-in augmentation: disabled for `only_custom_online_aug`.
- Online copy-paste: disabled.
- Train success: `true`
- Val success: `true`
- Val P/R/mAP50/mAP50-95: `0.7132/0.6641/0.6859/0.4661`
- Online random-like better than offline random by mAP50-95: `false`
- Online random-like better than online DiagAug by mAP50-95: `true`
- Random external advantage source: `operator_combo_helps_but_offline_doubling_or_training_variance_still_contributes`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/reports/online_random_like_50ep_report.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/reports/online_random_like_metrics.json`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/reports/compare_online_random_like_with_all.md`
- Stats JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/reports/online_aug_stats.json`
<!-- ONLINE_RANDOM_LIKE_50EP_END -->

<!-- FEEDBACK_ONLINE_AUG_SMOKE_START -->
## Feedback Online Augmentation Smoke

- Run ID: `feedback_online_policy_2stage_smoke`
- Entrypoint: `scripts/train_yolo_online_aug.py` with `--feedback-enabled`.
- Mechanism: custom YOLO-like/industrial online augmentation remains inside the training dataloader; no fixed augmented dataset is built.
- Stage count: `2`
- Policy history updates: `1`
- Train image count: `2301`; no train image doubling.
- Fixed augmented dataset generated: `false`
- Validation custom augmentation: `false`; val uses original val tiles.
- YOLO built-in augmentation: disabled for `only_custom_online_aug`.
- Online copy-paste: pending object-bank implementation; feedback may raise pending copy-paste probabilities but execution is skipped safely.
- Train success: `true`
- Val success: `true`
- Val P/R/mAP50/mAP50-95: `0.6540/0.3189/0.3338/0.2071`
- Report: `outputs/experiments/feedback_online_policy_2stage_smoke/reports/online_aug_smoke_report.md`
- Stats JSON: `outputs/experiments/feedback_online_policy_2stage_smoke/reports/online_aug_stats.json`
- Policy history JSON: `outputs/experiments/feedback_online_policy_2stage_smoke/reports/policy_history.json`
<!-- FEEDBACK_ONLINE_AUG_SMOKE_END -->

<!-- CUSTOM_YOLO_LIKE_BASE_50EP_START -->
## Custom YOLO-Like Base 50 Epoch

- Run ID: `20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep`
- Entrypoint: `scripts/train_yolo_online_aug.py`.
- Policy: `configs/online_policies/yolo_like_base_policy.json`.
- Mechanism: custom YOLO-like operators are sampled online in the YOLO training dataloader; no fixed augmented dataset is built.
- Train image count: `2301`; no train image doubling.
- Fixed augmented dataset generated: `false`
- Validation custom augmentation: `false`; val uses original val tiles.
- YOLO built-in augmentation: disabled for `only_custom_online_aug`.
- Feedback applied: `false`; policy history records no feedback applied.
- close_mosaic active: `true`
- Train success: `true`
- Val success: `true`
- Val P/R/mAP50/mAP50-95: `0.6661/0.7490/0.7129/0.4415`
- Close to YOLO default by mAP50-95 within 0.03: `false`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/custom_yolo_like_base_50ep_report.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/custom_yolo_like_base_50ep_metrics.json`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/compare_custom_yolo_like_with_yolo_default.md`
- Stats JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/online_aug_stats.json`
- Policy history JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/policy_history.json`
<!-- CUSTOM_YOLO_LIKE_BASE_50EP_END -->

<!-- BEGIN YOLO_DEFAULT_DIAGNOSIS_CONSTRAINED_50EP -->
## YOLO Default Diagnosis-Constrained 50 Epoch

- Run ID: `20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep`
- Objective: keep Ultralytics YOLO default augmentation enabled and add only constrained diagnostic online augmentation.
- Groups: YOLO default, diagnosis_light, diagnosis_precision_safe, diagnosis_recall_safe, plus old custom_yolo_like_base control.
- All A-D groups use `yolo11n.pt`, tiled safe no-OK/no-position data, `epochs=50`, `imgsz=1024`, `batch=2`, `workers=0`, `device=0`, `seed=42`.
- YOLO built-in augmentation: enabled for A-D; custom diagnosis policies are added online in the dataloader.
- Fixed augmented dataset generated: `false`.
- YOLO default reference P/R/mAP50/mAP50-95: `0.7132/0.7600/0.7759/0.5241`
- Best under industrial constraints: `YOLO default`
- Recall improved while constraints hold: `false`
- Failure driver if no improvement: `diagnosis_light: mAP50-95 dropped without localization_weak increase in conf=0.25 diagnosis; diagnosis_precision_safe: FP increased by 18; diagnosis_precision_safe: mAP50-95 dropped without localization_weak increase in conf=0.25 diagnosis; diagnosis_recall_safe: mAP50-95 dropped without localization_weak increase in conf=0.25 diagnosis`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/reports/diagnosis_constrained_experiment_report.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/reports/diagnosis_constrained_metrics.json`
- Constraint scoring: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/reports/constraint_scoring.json`
<!-- END YOLO_DEFAULT_DIAGNOSIS_CONSTRAINED_50EP -->

<!-- YOLO_DEFAULT_FEEDBACK_AUG_SMOKE_START -->
## YOLO Default Feedback Augmentation Smoke

- Entrypoint: `scripts/train_yolo_default_with_feedback.py`.
- Base: Ultralytics YOLO default augmentation remains enabled; custom YOLO-like `mosaic4` and `randaugment_like` are not used.
- Scope: 2-stage smoke when `epochs=2 feedback_interval=1`; no formal 50 epoch run in this step.
- Output: `outputs/experiments/yolo_default_feedback_aug_50ep/`
- Stage count: `2`
- Policy history updates: `1`
- Fixed augmented dataset generated: `false`
- Final P/R/mAP50/mAP50-95: `0.5259/0.3311/0.3109/0.1961`
- Constraint accepted: `false`
- Report: `outputs/experiments/yolo_default_feedback_aug_50ep/reports/yolo_default_feedback_smoke_report.md`
- Policy history: `outputs/experiments/yolo_default_feedback_aug_50ep/reports/policy_history.json`
<!-- YOLO_DEFAULT_FEEDBACK_AUG_SMOKE_END -->

<!-- YOLO_DEFAULT_FEEDBACK_AUG_50EP_FULL_START -->
## YOLO Default Feedback Augmentation

- Entrypoint: `scripts/train_yolo_default_with_feedback.py`.
- Base: Ultralytics YOLO default augmentation remains enabled; custom YOLO-like `mosaic4` and `randaugment_like` are not used.
- Scope: formal 50 epoch segmented feedback run.
- Output: `outputs/experiments/yolo_default_feedback_aug_50ep_full/`
- Stage count: `10`
- Policy history updates: `9`
- Fixed augmented dataset generated: `false`
- Final P/R/mAP50/mAP50-95: `0.7712/0.6689/0.7439/0.4993`
- Constraint accepted: `false`
- Report: `outputs/experiments/yolo_default_feedback_aug_50ep_full/reports/final_report.md`
- Policy history: `outputs/experiments/yolo_default_feedback_aug_50ep_full/reports/policy_history.json`
<!-- YOLO_DEFAULT_FEEDBACK_AUG_50EP_FULL_END -->
<!-- YOLO_DEFAULT_INLOOP_FEEDBACK_SMOKE_START -->
## YOLO Default In-Loop Feedback / Control

- Entrypoint: `scripts/train_yolo_default_with_inloop_feedback.py`.
- The previous `yolo_default_feedback_aug_50ep_full` run is a segmented fine-tune experiment, not strict continuous feedback.
- New direction: one `YOLO.train()` run with in-loop feedback callbacks; optimizer/scheduler/EMA/epoch/close_mosaic remain under one Ultralytics trainer.
- Feedback controller: `CATF` (Constraint-Aware Trust-region Feedback Controller).
- CATF uses the clean native YOLO default reference curve at matching feedback epochs, trust-region step limits, group budgets, delayed acceptance, rollback, cooldown, and epoch>=40 freeze.
- CATF-v1 is global feedback; CATF-v2 is class-aware, issue-aware, and sample-aware feedback with ROI-aware industrial augmentation.
- CATF-v2 current goal is to reduce CATF-v1 Precision instability by activating only diagnosed classes and freezing stable classes.
- Current CATF-v2 work is smoke-only; no formal 50 epoch CATF-v2 run should be inferred from it.
- No-feedback control disables both feedback and industrial augmentation, using Ultralytics YOLO default augmentation as the behavior check.
- The old YOLO default reference is not the final baseline after parity audit; feedback comparisons should use `clean_native_yolo_default_seed42_50ep`.
- Output: `outputs/experiments/seed1/`
- Epochs: `50`
- Feedback enabled: `true`
- Industrial augmentation enabled: `true`
- CATF version: `v2`
- Class-aware feedback: `true`
- ROI-aware augmentation: `true`
- Sample-aware routing: `true`
- Reference curve loaded: `true`
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Stage restart count: `0`
- Epoch continuous: `true`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- Constraint baseline: `clean_native_yolo_default`
- Constraint failed: `False`
- Report: `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed1/reports/final_report.md`
- Policy history: `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed1/reports/policy_history.json`
<!-- YOLO_DEFAULT_INLOOP_FEEDBACK_SMOKE_END -->
<!-- YOLO_DEFAULT_INLOOP_PARITY_AUDIT_START -->
## YOLO Default In-Loop Parity Audit

- Scope: parity audit only; no new 50 epoch feedback or augmentation experiment was run.
- Reference run: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/runs/yolo_default_seed42/`.
- In-loop no-feedback control: `outputs/experiments/yolo_default_inloop_no_feedback_control_50ep/`.
- Finding: the selected reference run used `OnlineAugDetectionTrainer` with an empty passthrough policy, so it is not a pure native YOLO CLI/Python baseline.
- Finding: the completed in-loop control used default trainer by command, but the old script still attached a no-op in-loop callback, as shown by `epoch_records.json`.
- Args diff: data path absolute vs relative, plus project/save_dir; augmentation args and validation args matched.
- close_mosaic: both runs used `close_mosaic=10` and both logs triggered `Closing dataloader mosaic`.
- Repair: `feedback=false` and `industrial_aug=false` now enters a native passthrough branch with no custom trainer, dataset, transform, or feedback callback.
- 1 epoch parity smoke: passed; native Python API and repaired in-loop no-feedback had identical args except output paths, identical loss/metric/lr values, and zero in-loop callback records.
- Report: `outputs/audits/yolo_default_inloop_parity/parity_audit_report.md`
- JSON: `outputs/audits/yolo_default_inloop_parity/parity_audit.json`
<!-- YOLO_DEFAULT_INLOOP_PARITY_AUDIT_END -->
<!-- CLEAN_NATIVE_YOLO_DEFAULT_REFERENCE_START -->
## Clean Native YOLO Default Reference

- The old YOLO default reference is no longer treated as the final baseline because parity audit found it used `OnlineAugDetectionTrainer` with an empty passthrough policy.
- New baseline: `outputs/experiments/clean_native_yolo_default_seed42_50ep/`.
- Training mode: pure native Ultralytics `YOLO.train(**same_args)`.
- Custom trainer / callback / dataset / transform / industrial augmentation: `false`.
- Train image count: `2301`
- Precision/Recall/mAP50/mAP50-95: `0.7262/0.6844/0.7616/0.5250`
- close_mosaic official schedule expected: `true`
- close_mosaic expected start epoch: `41`
- Future feedback experiments should compare only against this clean native reference.
- Report: `outputs/experiments/clean_native_yolo_default_seed42_50ep/reports/clean_native_yolo_default_report.md`
- Metrics JSON: `outputs/experiments/clean_native_yolo_default_seed42_50ep/reports/clean_native_yolo_default_metrics.json`
<!-- CLEAN_NATIVE_YOLO_DEFAULT_REFERENCE_END -->

<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_INLOOP_FEEDBACK_START -->
## Multiseed Clean YOLO Default vs In-Loop Feedback

- Scope: seeds `0, 1, 2`; seed 42 is not included in the multiseed mean.
- Output: `outputs/experiments/multiseed_clean_yolo_default_vs_inloop_feedback/`.
- Clean group uses pure native Ultralytics `YOLO.train`; feedback group uses single-run in-loop feedback with YOLO default augmentation still enabled.
- Train images: `2301`; fixed augmented dataset generated: `false`; copy_paste remains pending/not enabled.
- Feedback wins under industrial constraints: `1/3`.
- Constraint failed seeds: `2/3`.
- Mean delta P/R/mAP50/mAP50-95: `-0.0275/0.0254/0.0093/0.0211`.
- Verdict: not stable enough to claim as the paper main result yet; use as diagnostic/ablation unless a stricter controller passes multiseed constraints.
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_inloop_feedback/reports/multiseed_summary.md`.
- JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_inloop_feedback/reports/multiseed_summary.json`.
<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_INLOOP_FEEDBACK_END -->

<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_CATF_FEEDBACK_START -->
## Multiseed Clean YOLO Default vs CATF Feedback

- Scope: seeds `0, 1, 2`; seed 42 is retained as a positive single-seed case but is not included in this multiseed mean.
- Output: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_feedback/`.
- Clean native results were reused from `outputs/experiments/multiseed_clean_yolo_default_vs_inloop_feedback/seed_*/clean_native_yolo_default/`; no duplicate clean native training was run.
- CATF group uses single-run in-loop feedback with YOLO default augmentation still enabled, per-seed clean native reference curves, industrial online augmentation enabled, and copy_paste pending/not enabled.
- Train images: `2301`; fixed augmented dataset generated: `false`; val uses original val tiles.
- Per-seed deltas P/R/mAP50/mAP50-95:
  - seed 0: `-0.0272/-0.0014/+0.0179/+0.0445`, constraint_failed=`true`.
  - seed 1: `-0.0349/+0.0621/+0.0008/+0.0386`, constraint_failed=`true`.
  - seed 2: `-0.0116/-0.0120/-0.0084/-0.0424`, constraint_failed=`true`.
- Mean delta P/R/mAP50/mAP50-95: `-0.0246/+0.0163/+0.0034/+0.0136`.
- CATF wins under industrial constraints: `0/3`; constraint failed seeds: `3/3`.
- Control statistics across seeds: rollback `17`, cooldown `3`, freeze `3`; frozen policy records `15`.
- Compared with old in-loop feedback, CATF is more conservative in logs but not more stable by the industrial constraint criterion: old feedback failed `2/3`, CATF failed `3/3`.
- Verdict: CATF is not recommended as the paper main method based on seeds `0/1/2`; keep seed42 as a positive case and report this as an ablation/controller attempt unless a later controller passes multiseed constraints.
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_feedback/reports/multiseed_catf_summary.md`.
- JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_feedback/reports/multiseed_catf_summary.json`.
<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_CATF_FEEDBACK_END -->

<!-- CATF_V2_ACTIVATION_AUDIT_START -->
## CATF-v2 Activation Audit

- Scope: `outputs/experiments/catf_v2_class_aware_10ep_smoke/reports/`.
- No training was run; this is a report-only audit of epoch 5 CATF-v2 activation.
- Audit outputs: `outputs/experiments/catf_v2_class_aware_10ep_smoke/reports/catf_v2_activation_audit.md` and `.json`.
- Active classes from smoke: class `1` OK3, class `6` 漏背锡, class `8` 脏污.
- OK3 activation is judged not reasonable for a formal run: Recall is already high (`0.9891`), FN count is only `2`, FP count is `49`, and OK-like classes should default to stable/no_aug unless evidence is very strong.
- 漏背锡 activation is judged reasonable: low Recall (`0.2826`), many FN (`32`), low-contrast evidence, and conservative ROI `sharpen_mild`/`local_contrast` ops.
- 脏污 activation is partially reasonable as low Recall, but should be guarded by stain/dirty high-FP domain priors and should not lower threshold or escalate photometric before FP behavior is known.
- Low-support classes `2` and `3` did not trigger strong photometric augmentation; they remain oversampling/copy-paste pending candidates.
- ROI augmentation applied `150` times, including OK3 (`122`), which is the main activation-rule concern.
- Recommendation: do not enter CATF-v2 seed42 50ep until activation rules are tightened with OK2/OK3 no_aug, stronger activation threshold, domain high-FP guards for stain/oil/dirty classes, and likely top_k reduced from `3` to `2`.
<!-- CATF_V2_ACTIVATION_AUDIT_END -->

<!-- CATF_V2_ACTIVATION_FIXED_START -->
## CATF-v2 Activation Rule Fix

- Scope: no 50 epoch training; only rule changes, tests, and a 10 epoch smoke run.
- Output: `outputs/experiments/catf_v2_activation_fixed_10ep_smoke/`.
- Report: `outputs/experiments/catf_v2_activation_fixed_10ep_smoke/reports/catf_v2_activation_fixed_report.md`.
- Rule changes: OK2/OK3 default no_aug, stricter activation thresholds, domain high-FP prior for OK2/OK3/oil/dirty classes, top_k_active_classes=`2`, ROI blocks no_aug/high-FP conflict classes.
- Smoke result: train_success=`true`, val_success=`true`, train_images=`2301`, fixed_augmented_dataset_generated=`false`.
- Active classes after fix: class `6` 漏背锡 (texture_boundary_weak), class `8` 脏污 (low_recall).
- OK3 active=`false`; OK3 ROI applied=`0`.
- 漏背锡 active=`true` with conservative ROI sharpen/local_contrast.
- 脏污 domain_high_fp_prior=`true`; photometric probs `{'clahe': 0.0, 'gamma': 0.0, 'brightness': 0.0, 'contrast': 0.0}`; threshold recommendation `0.25` with reason `domain_high_fp_prior_keep_threshold`.
- ROI stats: `{'roi_aug_applied': 23, 'roi_aug_skipped_small_roi': 0, 'roi_aug_skipped_conflict': 2, 'affected_classes': {'6': 20, '8': 3}}`.
- BBox/class valid: `true`.
- Recommendation: proceed to CATF-v2 seed42 50 epoch validation only after this fixed activation rule set; do not use the earlier CATF-v2 smoke as formal evidence.
<!-- CATF_V2_ACTIVATION_FIXED_END -->

<!-- CATF_V2_SEED42_50EP_START -->
## CATF-v2 Seed42 50 Epoch

- Output: `outputs/experiments/catf_v2_seed42_50ep/`.
- Entry: `scripts/train_yolo_default_with_inloop_feedback.py` with `--catf-version v2`, class-aware feedback, ROI-aware augmentation, sample-aware routing, threshold calibration report, top_k=`2`, top_m=`2`.
- Training mode: single-run continuous YOLO default training with official YOLO augmentation kept enabled; no stage restart; no self-implemented mosaic/randaugment replacement.
- Train images: `2301`; fixed augmented dataset generated: `false`; copy_paste remains `pending_object_bank_design`.
- Epoch continuity: `1..50` continuous.
- Final metrics P/R/mAP50/mAP50-95: `0.7498/0.7257/0.7679/0.5212`.
- Delta vs clean native seed42 P/R/mAP50/mAP50-95: `+0.0236/+0.0413/+0.0062/-0.0039`.
- constraint_failed: `false`.
- OK2/OK3 active epochs: `[]`; OK3 ROI applied: `0`.
- Active class counts: `{'6:漏背锡': 1, '12:锡膏': 1}`.
- ROI stats: `{'roi_aug_applied': 37, 'roi_aug_skipped_small_roi': 0, 'roi_aug_skipped_conflict': 0, 'affected_classes': {'6': 18, '12': 19}}`.
- Feedback actions: `{'observe': 2, 'accept': 1, 'shrink': 4, 'freeze': 2}`; class actions: `{'propose': 2, 'observe': 3}`; rollback/cooldown/freeze: `0/0/4`.
- Report: `outputs/experiments/catf_v2_seed42_50ep/reports/final_report.md`.
- Metrics JSON: `outputs/experiments/catf_v2_seed42_50ep/reports/final_metrics.json`.
- Policy history: `outputs/experiments/catf_v2_seed42_50ep/reports/policy_history.json`.
- Best checkpoint: `outputs/experiments/catf_v2_seed42_50ep/train/weights/best.pt`.
- Verdict: seed42 passes industrial constraints and is suitable for CATF-v2 multiseed validation; do not claim final method before multiseed passes.
<!-- CATF_V2_SEED42_50EP_END -->

<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_CATF_V2_START -->
## Multiseed Clean YOLO Default vs CATF-v2

- Scope: seeds `0, 1, 2`; seed 42 remains a positive single-seed validation and is not included in the multiseed mean.
- Output: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/`.
- Clean native results were reused from existing per-seed clean YOLO default runs; CATF-v2 was newly trained for each seed.
- CATF-v2 settings: official YOLO default augmentation kept enabled, class-aware feedback, ROI-aware augmentation, sample-aware routing, threshold calibration report, top_k=`2`, top_m=`2`, feedback interval=`5`.
- Train images: `2301`; fixed augmented dataset generated: `false`; copy_paste remains `pending_object_bank_design`.
- Per-seed deltas P/R/mAP50/mAP50-95:
  - seed 0: `-0.0417/+0.0218/+0.0223/+0.0264`, constraint_failed=`true` due Precision drop.
  - seed 1: `-0.0034/+0.0474/+0.0007/+0.0099`, constraint_failed=`false`.
  - seed 2: `+0.1395/-0.1247/-0.0269/-0.0285`, constraint_failed=`true` due mAP50 and mAP50-95 drops.
- Mean delta P/R/mAP50/mAP50-95: `+0.0315/-0.0185/-0.0013/+0.0026`.
- CATF-v2 wins under industrial constraints: `1/3`; constraint failed seeds: `2/3`.
- OK2/OK3 were never active; OK3 ROI applied total: `0`.
- Active class counts: `{'8:脏污': 1, '11:锡尖': 1}`; ROI affected totals: `{'8:脏污': 3, '11:锡尖': 20}`.
- Control statistics across seeds: rollback `0`, cooldown `0`, freeze events `12`; policy actions `{'shrink': 14, 'accept': 2, 'observe': 5, 'freeze': 6}`.
- Compared with CATF-v1, CATF-v2 improves activation discipline and reduces constraint failures from `3/3` to `2/3`, but is still not stable enough for the paper main method.
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/multiseed_catf_v2_summary.md`.
- JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/multiseed_catf_v2_summary.json`.
<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_CATF_V2_END -->

<!-- CATF_V2_FAILURE_MODE_ANALYSIS_START -->
## CATF-v2 Multiseed Failure-Mode Analysis

- Scope: analysis only; no training was run and no CATF-v2 rules were changed.
- Source experiment: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/`.
- Generated reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/curve_diagnosis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/active_class_effect_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/precision_drop_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/recall_drop_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/threshold_calibration_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/controller_behavior_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/catf_v2_failure_mode_summary.md`
- Main failure modes: `intervention too weak`, `intervention wrong target`, `precision threshold issue`, `over-conservative freeze`, `per-class diagnosis trajectory sensitive`, and `in-loop diagnosis/callback side effect not isolated`.
- Seed 0 failure: Recall/mAP improved, but Precision failed due approximate FP increases across multiple classes, led by oil/dirty-like and defect classes; threshold calibration is the most direct repair candidate.
- Seed 1 pass: no ROI/industrial augmentation was applied, so the pass is not causal evidence for ROI augmentation; it may include in-loop diagnosis/callback RNG effects.
- Seed 2 failure: Precision increased while Recall/mAP dropped, indicating conservative confidence/detection behavior on a clean seed that already had high Recall.
- ROI-aware augmentation was too sparse to prove benefit: only `23` ROI applications across seeds `0/1/2`.
- Controller behavior: shrink-dominant (`14` shrink vs `2` accept), zero rollback/cooldown, and freeze at epoch 40/45; negative-effect attribution is missing.
- Recommendation: do not claim CATF-v2 as paper main method yet; next step should be diagnosis-only in-loop control plus per-class threshold calibration analysis before more 50 epoch training.
<!-- CATF_V2_FAILURE_MODE_ANALYSIS_END -->

<!-- THRESHOLD_CALIBRATION_DIAGNOSIS_ONLY_START -->
## CATF-v2 Threshold Calibration and Diagnosis-Only Control

- Scope: no new 50 epoch training and no CATF-v2 rule changes.
- Post-hoc threshold report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/threshold_calibration_posthoc.md`.
- Post-hoc JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/threshold_calibration_posthoc.json`.
- Prediction-only inputs: cached/generated validation predictions for clean native and CATF-v2 seeds `0/1/2`; no training was run.
- Threshold grid: per-class confidence threshold `0.10..0.70` step `0.05`; objectives `constrained_score`, `balanced_score`, `industrial_score`.
- Constrained post-hoc result: CATF-v2 passes industrial constraints for `2/3` seeds after calibration.
- Seed 0: Precision/constraint failure is repairable by threshold calibration in the post-hoc evaluator; precision-oriented objectives raise thresholds for OK/oil/dirty-like and stable classes while lowering difficult defect classes.
- Seed 2: Recall/mAP failure is not repaired by threshold lowering; this remains a training/trajectory degradation, not a pure confidence-threshold issue.
- Diagnosis-only smoke: `outputs/experiments/diagnosis_only_inloop_control_10ep_smoke/`.
- Diagnosis-only plan: `outputs/experiments/diagnosis_only_inloop_control_plan.md`.
- Smoke result: diagnosis callback executed at epoch `5`; industrial samples augmented=`0`, ROI applied=`0`, policy update applied=`0`, train images=`2301`, fixed augmented dataset generated=`false`, epoch sequence `1..10` continuous, bbox/class legal.
- Interpretation: threshold calibration can be a deployment/post-processing companion, but seed 2 shows it cannot substitute for robust training feedback.
<!-- THRESHOLD_CALIBRATION_DIAGNOSIS_ONLY_END -->

<!-- DIAGNOSIS_ONLY_CONTROL_50EP_START -->
## Diagnosis-Only In-Loop Control 50 Epoch

- Scope: seeds `0, 1, 2`; this is the control for in-loop diagnosis callback/RNG/training-path effects.
- Output: `outputs/experiments/diagnosis_only_inloop_control_50ep/`.
- Entry: `scripts/train_yolo_default_with_inloop_feedback.py`.
- Configuration: YOLO default augmentation enabled, feedback diagnosis enabled, `diagnosis_only=true`, `industrial_aug_enabled=false`, ROI-aware augmentation disabled, sample-aware routing disabled, threshold mutation disabled, copy_paste not enabled.
- Train images: `2301`; fixed augmented dataset generated: `false`; results.csv epoch `1..50` continuous for all seeds.
- Control counters for all seeds: industrial samples augmented=`0`, ROI applied=`0`, policy update applied=`0`, bbox/class legal=`true`.
- Final diagnosis-only metrics P/R/mAP50/mAP50-95:
  - seed 0: `0.7846/0.6765/0.7347/0.4759`.
  - seed 1: `0.7725/0.6477/0.7542/0.4799`.
  - seed 2: `0.6962/0.7286/0.7692/0.5224`.
- Delta vs clean native for all seeds and all four metrics: `0.0000`; diagnosis-only constraint pass count: `3/3`.
- Interpretation: the diagnosis callback alone did not change training results. Seed 1 CATF-v2 success is not explained by callback/RNG alone, though CATF-v2 industrial-enabled training-path differences still need caution because that seed reported zero actual industrial/ROI augmentation.
- CATF-v2 + post-hoc threshold calibration retains independent value as a deployment/post-processing companion: pass count improves from `1/3` to `2/3`, but seed 2 remains unrepaired.
- Summary report: `outputs/experiments/diagnosis_only_inloop_control_50ep/reports/diagnosis_only_multiseed_summary.md`.
- Summary JSON: `outputs/experiments/diagnosis_only_inloop_control_50ep/reports/diagnosis_only_multiseed_summary.json`.
<!-- DIAGNOSIS_ONLY_CONTROL_50EP_END -->

<!-- CATF_V2_NOOP_PARITY_AUDIT_START -->
## CATF-v2 No-op Parity Audit

- Scope: no new CATF-v2 strategy training; added explicit `--catf-noop` audit mode and ran parity controls.
- Output smoke: `outputs/experiments/catf_v2_noop_parity_smoke/`.
- Output 50ep control: `outputs/experiments/catf_v2_noop_control_50ep/seed_1/`.
- `--catf-noop` behavior: CATF-v2 custom trainer/dataset/router objects are built, but the online transform returns labels unchanged before bbox conversion, sample router is not called, router random draws=`0`, ROI applied=`0`, industrial samples augmented=`0`, and policy update applied=`0`.
- 1ep parity smoke: clean native vs CATF-v2 noop metrics/loss/lr/results.csv numeric fields are identical except wall-clock `time`; args.yaml differs only in `project` and `save_dir`.
- 1ep smoke report: `outputs/experiments/catf_v2_noop_parity_smoke/reports/noop_parity_report.md`.
- Random path audit: `outputs/experiments/catf_v2_noop_parity_smoke/reports/random_path_audit.md`.
- Seed1 50ep CATF-v2 noop metrics P/R/mAP50/mAP50-95: `0.7725/0.6477/0.7542/0.4799`.
- Delta seed1 noop vs clean native: `0.0000/0.0000/0.0000/0.0000`; constraint_failed=`false`.
- Delta seed1 noop vs CATF-v2: `+0.0034/-0.0474/-0.0007/-0.0099`.
- Interpretation: diagnosis-only already showed the callback itself is neutral; CATF-v2 noop now shows the custom CATF-v2 framework path is also neutral when augmentation/policy mutation are hard-disabled. Seed1 CATF-v2 gain is therefore not explained by no-op framework perturbation, but still cannot be attributed to ROI augmentation for that seed because actual ROI/industrial counters were zero.
- Seed1 noop report: `outputs/experiments/catf_v2_noop_control_50ep/reports/noop_control_seed1_report.md`.
<!-- CATF_V2_NOOP_PARITY_AUDIT_END -->

<!-- FIXED_CATF_V2_SEED2_THRESHOLD_ANALYSIS_START -->
## Fixed CATF-v2 Seed2 Failure and Threshold Calibration Analysis

- Date: `2026-06-04`.
- Scope: analysis-only; no train command was run.
- Source: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/`.
- Script: `D:\Anaconda\envs\pytorch\python.exe scripts\analyze_fixed_catf_v2_seed2_and_thresholds.py`.
- Reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed2_failure_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_threshold_calibration_posthoc.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_vs_old_catf_v2_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_catf_v2_next_step_summary.md`
- Seed2 fixed CATF-v2 result vs clean: P/R/mAP50/mAP50-95 delta `+0.0674/-0.0423/-0.0110/-0.0257`; constraint failed due mAP drops.
- Failure diagnosis: CATF-v2 made seed2 more precision-oriented on a high-recall clean baseline, with Recall loss in `浅划伤`, `轮廓划伤`, `加强筋打伤`, `漏背锡`, `锡膏` and AP/localization loss in `锡丝残留`, `脏污`, `锡膏`, `轮廓划伤`, `开裂`, `加强筋打伤`.
- Seed2 active classes were class `9` 轮廓划伤, class `11` 锡尖, and class `8` 脏污; ROI was not aligned with all final degraded classes.
- Post-hoc threshold calibration repairs seed2 in the analysis evaluator, but fixed CATF-v2 calibrated pass count remains `2/3` because seed0 still fails mAP50-95.
- Conclusion: fixed CATF-v2 is a strong candidate and threshold calibration is useful, but the method should not yet be described as stably superior to clean YOLO default.
<!-- FIXED_CATF_V2_SEED2_THRESHOLD_ANALYSIS_END -->

<!-- CATF_V2_RC_SEED0_CALIBRATION_ROLLBACK_START -->
## CATF-v2-RC Seed0 Calibration and Rollback Analysis

- Date: `2026-06-04`.
- Scope: analysis-only; no train command was run.
- Script: `D:\Anaconda\envs\pytorch\python.exe scripts\refine_fixed_catf_v2_seed0_calibration_and_rollback.py`.
- Added module: `AutoAugment/catf_v2/per_class_thresholds.py`.
- Reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed0_failure_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed0_threshold_calibration_posthoc.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_catf_v2_threshold_calibration_all_seeds.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_catf_v2_class_level_rollback_simulation.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/catf_v2_rc_final_candidate_plan.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_catf_v2_ready_for_paper_summary.md`
- Seed0 official fixed CATF-v2 already passes constraints; the remaining failure was the earlier post-hoc threshold evaluator mAP50-95 deficit.
- Seed0 RC threshold objective result in post-hoc evaluator: P/R/mAP50/mAP50-95 `0.6636/0.7915/0.6737/0.4303`, constraint_failed=`false`.
- RC threshold calibration pass count: `3/3`.
- Recommended threshold table saved to `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/catf_v2_rc_per_class_thresholds.json`.
- Rollback candidates: `加强筋打伤`, `锡丝残留`; high-risk classes: `轮廓划伤`, `锡膏`; threshold-only class: `脏污`.
- Conclusion: CATF-v2-RC is the current paper-candidate framing, but official validation/export confirmation of the threshold table is still required.
<!-- CATF_V2_RC_SEED0_CALIBRATION_ROLLBACK_END -->

<!-- CATF_V2_RC_OFFICIAL_PATH_VALIDATION_START -->
## CATF-v2-RC Official Val/Predict Path Validation

- Date: `2026-06-04`.
- Scope: validation/inference-only; no training was run.
- Command: `D:\Anaconda\envs\pytorch\python.exe scripts\validate_catf_v2_rc_official_path.py`.
- Script: `scripts/validate_catf_v2_rc_official_path.py`.
- Threshold config: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/catf_v2_rc_per_class_thresholds.json`.
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/catf_v2_rc_official_path_validation.md`.
- JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/catf_v2_rc_official_path_validation.json`.
- Fixed seed0/1/2 checkpoints were rerun with Ultralytics `YOLO.val`; predictions were generated with Ultralytics `YOLO.predict(conf=0.10)`.
- Official val fixed-vs-clean deltas:
  - seed0 `-0.0060/-0.0068/+0.0090/+0.0136`.
  - seed1 `+0.0127/+0.0528/+0.0284/+0.0390`.
  - seed2 `+0.0674/-0.0423/-0.0110/-0.0257`.
- Official predict + saved RC threshold table pass count: `1/3`.
- RC seed0 fails Precision: delta `-0.0666/+0.0321/+0.0054/-0.0095`.
- RC seed1 passes: delta `+0.0496/+0.0501/+0.0395/+0.0289`.
- RC seed2 fails Precision: delta `-0.0148/+0.0655/+0.0435/+0.0096`.
- Conclusion: the saved unified RC threshold table is too recall-aggressive. The earlier 3/3 post-hoc conclusion does not hold under the official predict post-processing validation; next step is threshold re-optimization with stronger Precision guard, not training.
<!-- CATF_V2_RC_OFFICIAL_PATH_VALIDATION_END -->

<!-- CP_CATF_OFFLINE_CAUSAL_PROBE_START -->
## CP-CATF Offline Causal Probe

- Date: `2026-06-08`.
- Scope: offline strategy-screening only; no train command was run.
- Code changes:
  - Added `AutoAugment/catf_v2/causal_probe.py`.
  - Added `scripts/run_catf_v2_offline_causal_probe.py`.
  - Added `tests/test_catf_v2_causal_probe.py`.
  - Updated `AutoAugment/catf_v2/high_risk_class_ops.py` so historical high-risk class-op entries are audit priors, not default direct blocks.
- Command: `python scripts\run_catf_v2_offline_causal_probe.py`.
- Inputs:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.json`
  - `outputs/experiments/seed2_failure_root_cause/reports/seed2_root_cause_summary.json`
  - `outputs/experiments/catf_v2_riskguard_seed2_50ep/reports/riskguard_seed2_summary.json`
- Outputs:
  - `outputs/experiments/catf_v2_causal_probe/seed_0/probe_decisions.json`
  - `outputs/experiments/catf_v2_causal_probe/seed_1/probe_decisions.json`
  - `outputs/experiments/catf_v2_causal_probe/seed_2/probe_decisions.json`
  - `outputs/experiments/catf_v2_causal_probe/reports/offline_causal_probe_summary.md`
  - `outputs/experiments/catf_v2_causal_probe/reports/offline_causal_probe_summary.json`
  - `outputs/experiments/catf_v2_causal_probe/reports/cp_catf_training_plan.md`
- Development-mode caveat: offline probe uses existing validation diagnostics/results to validate the mechanism. Paper-mode CP-CATF must use train/probe examples and keep final validation/test out of policy selection.
- Decisions:
  - seed0 accepted `candidate_policy_1_roi_texture`.
  - seed1 accepted `candidate_policy_1_roi_texture`.
  - seed2 rejected image-space candidates and selected `candidate_policy_3_sampler_only` with image modification disabled.
- Key interpretation: RiskGuard's fixed blacklist is not needed as final method logic and is insufficient as a final fix because seed2 RiskGuard still failed overall constraints. CP-CATF converts the audit lesson into run-specific causal screening.
- Verification:
  - `python -m py_compile AutoAugment/catf_v2/causal_probe.py scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py AutoAugment/catf_v2/high_risk_class_ops.py scripts/run_catf_v2_offline_causal_probe.py`
  - `pytest -q tests/test_catf_v2_causal_probe.py tests/test_catf_v2_riskguard.py tests/test_catf_v2_adaptive_burnin.py tests/test_catf_v2_rollback_controller.py tests/test_catf_v2_gated_controller.py tests/test_catf_v2_safe_controller.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_activation_rules.py tests/test_catf_v2_per_class_diagnosis.py tests/test_catf_v2_policy_matrix.py tests/test_catf_v2_sample_router.py tests/test_catf_v2_roi_augmentation.py tests/test_catf_v2_threshold_calibration.py tests/test_feedback_policy_guard.py tests/test_feedback_policy_controller.py tests/test_inloop_feedback_training.py tests/test_online_augmentation.py tests/test_proxy_prefilter.py tests/test_copy_paste.py`
  - Result: `132 passed`.
<!-- CP_CATF_OFFLINE_CAUSAL_PROBE_END -->

<!-- CP_CATF_MULTISEED_TRAINING_VALIDATION_START -->
## CP-CATF Multiseed Training Validation

- Date: `2026-06-09`.
- Scope: full 50ep multiseed CP-CATF training validation using offline causal-probe decisions.
- Run root: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_cp_catf/`.
- Code changes:
  - Added offline causal-probe decision integration to `scripts/train_yolo_default_with_inloop_feedback.py`.
  - Added `--causal-probe-mode true` alias and `--use-offline-probe-decisions true`.
  - Added `scripts/summarize_catf_v2_cp_catf_multiseed.py`.
  - Added regression tests for accepted offline candidate filtering and sampler-only image no-op behavior.
- Training commands used the requested YOLO default setup:
  - `model=yolo11n.pt`
  - `data=outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
  - `epochs=50`, `imgsz=1024`, `batch=2`, `workers=0`, `device=0`
  - `--catf-version v2 --causal-probe-mode true --use-offline-probe-decisions true`
  - class-aware feedback, ROI-aware augmentation, sample-aware routing, threshold calibration report, feedback interval/start epoch 5, and industrial online augmentation enabled.
- Offline probe decisions used in training:
  - seed0: `candidate_policy_1_roi_texture`, accepted; texture ROI image augmentation allowed.
  - seed1: `candidate_policy_1_roi_texture`, accepted; texture ROI image augmentation allowed.
  - seed2: `candidate_policy_3_sampler_only`; image augmentation rejected. Sample weighting remains pending dataloader support, so seed2 actual path is strict image no-op.
- Final CP-CATF metrics:
  - seed0: Precision=0.7785, Recall=0.6697, mAP50=0.7437, mAP50-95=0.4895, `constraint_failed=false`.
  - seed1: Precision=0.7852, Recall=0.7005, mAP50=0.7826, mAP50-95=0.5189, `constraint_failed=false`.
  - seed2: Precision=0.6962, Recall=0.7286, mAP50=0.7692, mAP50-95=0.5224, `constraint_failed=false`.
- Key checks:
  - Constraint pass count: `3/3`.
  - seed0 retained fixed CATF-v2 mAP50/mAP50-95 gains.
  - seed1 retained fixed CATF-v2's clear multi-metric improvement.
  - seed2 protected clean baseline and rejected ROI/image augmentation.
  - seed2 industrial image samples augmented=0, ROI applied=0, router random draw count=0.
  - OK3 stayed inactive and OK3 ROI applied=0 across all seeds.
  - RiskGuard was not used as the final accept/reject rule.
  - No seed-id, fixed-class-id, or dataset-specific rule determined the CP-CATF training decision.
- Mean CP-CATF delta vs clean: dP=+0.0022, dR=+0.0153, dM50=+0.0125, dM95=+0.0175.
- Reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_cp_catf/reports/multiseed_cp_catf_summary.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_cp_catf/reports/multiseed_cp_catf_summary.json`
  - Per-seed `compare_with_clean_and_fixed_catf_v2.md` reports under each seed run.
- Verification:
  - `python -m py_compile AutoAugment/catf_v2/causal_probe.py scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py scripts/summarize_catf_v2_cp_catf_multiseed.py`
  - Requested pytest suite result: `134 passed`.
- Interpretation:
  - CP-CATF development-mode validation remains feasibility evidence for image-space causal control because it keeps seed0/seed1 gains and protects seed2.
  - This is not yet a leakage-free paper result. The current paper mainline is image augmentation based CATF with fixed CATF-v2 as the image-only baseline and CP-CATF image-only as the repair direction.
<!-- CP_CATF_MULTISEED_TRAINING_VALIDATION_END -->

<!-- CP_CATF_PAPER_MODE_VALIDATION_START -->
## CP-CATF Paper-Mode Probe Split Validation

- Date: `2026-06-09` to `2026-06-10`.
- Scope: paper-mode CP-CATF implementation and validation with a train/probe split; final validation was kept out of policy selection.
- Split command: `python scripts/create_paper_probe_split.py --source-data outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml --output-root outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe --split-seed 2026 --probe-ratio 0.10`.
- Split result: original train `2301`, train_core `2071`, probe `230`, final val `677`, no train_core/probe/final-val overlap.
- Smoke run: `outputs/experiments/cp_catf_paper_mode_10ep_smoke/`; seed2 10ep passed with `policy_selection_source=probe_split`, `final_val_used_for_policy_selection=false`, selected `candidate_policy_3_sampler_only`, and kept industrial/ROI/router random at `0`.
- Full run root: `outputs/experiments/multiseed_cp_catf_paper_mode/`.
- Clean paper baseline final metrics:
  - seed0: P=0.7513, R=0.6763, mAP50=0.7566, mAP50-95=0.5114.
  - seed1: P=0.7220, R=0.7582, mAP50=0.7777, mAP50-95=0.5251.
  - seed2: P=0.6290, R=0.6385, mAP50=0.6590, mAP50-95=0.4381.
- CP-CATF paper-mode final metrics:
  - seed0: P=0.7513, R=0.6763, mAP50=0.7566, mAP50-95=0.5114, `constraint_failed=false`.
  - seed1: P=0.7220, R=0.7582, mAP50=0.7777, mAP50-95=0.5251, `constraint_failed=false`.
  - seed2: P=0.6290, R=0.6385, mAP50=0.6590, mAP50-95=0.4381, `constraint_failed=false`.
- Candidate decisions:
  - seed0 mostly `candidate_policy_3_sampler_only`, with one `candidate_policy_1_roi_texture` accept.
  - seed1 mostly `candidate_policy_3_sampler_only`, with two `candidate_policy_1_roi_texture` accepts.
  - seed2 selected sampler-only at epochs 5/10/15/40 and accepted roi_texture at epochs 20/25/30/35/45.
- Actual image-space augmentation did not execute in any paper-mode CP-CATF seed: industrial samples augmented `0`, ROI applied `0`, router random draw count `0`.
- Constraint result: `0/3` failed, `3/3` pass.
- Mean delta vs clean paper baseline: dP=+0.0000, dR=+0.0000, dmAP50=+0.0000, dmAP50-95=+0.0000.
- Final-val leakage detected: `false`.
- Interpretation: paper-mode CP-CATF validates the no-leakage safety path but does not retain the development-mode gains. It should not yet be used as the final paper main result; development-mode remains method feasibility evidence, and paper-mode needs stronger probe evidence.
- Reports:
  - `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/reports/probe_split_report.md`
  - `outputs/experiments/cp_catf_paper_mode_10ep_smoke/reports/paper_mode_smoke_report.md`
  - `outputs/experiments/multiseed_cp_catf_paper_mode/reports/multiseed_cp_catf_paper_mode_summary.md`
  - `outputs/experiments/multiseed_cp_catf_paper_mode/reports/multiseed_cp_catf_paper_mode_summary.json`
- Verification:
  - `python -m py_compile AutoAugment/catf_v2/causal_probe.py scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py`
  - Requested pytest suite result: `142 passed`.
<!-- CP_CATF_PAPER_MODE_VALIDATION_END -->

<!-- CP_CATF_ACCEPT_TO_EXECUTION_AUDIT_START -->
## CP-CATF Paper-Mode Accept-to-Execution Audit

- Date: `2026-06-11`.
- Scope: audit, minimal fix, targeted tests, and one 10ep smoke. No 50ep experiment was run.
- Audit command: `python scripts/audit_cp_catf_accept_to_execution.py`.
- Audit outputs:
  - `outputs/experiments/multiseed_cp_catf_paper_mode/reports/cp_catf_accept_to_execution_audit.md`
  - `outputs/experiments/multiseed_cp_catf_paper_mode/reports/cp_catf_accept_to_execution_audit.json`
- Audit result:
  - Existing paper-mode multiseed had `8` accept events.
  - Accept events with executable active class-op policy entries: `0`.
  - Total industrial samples augmented: `0`.
  - Total ROI applied: `0`.
  - Total router random draws: `0`.
- Root cause: accepted causal-probe image candidates were recorded but not materialized into active policy entries. The implementation filtered existing policy ops with the accepted op whitelist, but did not inject the candidate target class or nonzero op probabilities/strengths into the policy matrix.
- Code fix:
  - added accepted-candidate policy activation in `scripts/train_yolo_default_with_inloop_feedback.py`;
  - accepted image candidates now write executable class-op entries for sample routing;
  - rejected and sampler-only candidates remain strict image no-op;
  - RiskGuard remains audit/debug prior only.
- Fixed 10ep smoke command used the `D:\Anaconda\envs\pytorch\python.exe` environment to match the expected Ultralytics version.
- Fixed smoke output:
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_10ep_smoke/`
  - seed `1`, 10 epochs, paper-mode source `probe_split`.
  - Final-val leakage: `false`.
  - The only causal-probe event selected `candidate_policy_3_sampler_only`; no image candidate was accepted in this 10ep smoke.
  - Industrial samples augmented `0`, ROI applied `0`, router random draw count `0`.
  - BBox/class checks stayed legal.
- Fixed smoke reports:
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_10ep_smoke/reports/execution_fixed_smoke_report.md`
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_10ep_smoke/reports/execution_fixed_smoke_report.json`
- Verification:
  - `python -m py_compile AutoAugment/catf_v2/causal_probe.py scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py scripts/audit_cp_catf_accept_to_execution.py`
  - `pytest -q tests/test_cp_catf_accept_to_execution.py tests/test_cp_catf_paper_mode.py tests/test_catf_v2_causal_probe.py tests/test_catf_v2_riskguard.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_policy_matrix.py tests/test_catf_v2_sample_router.py tests/test_catf_v2_roi_augmentation.py tests/test_inloop_feedback_training.py tests/test_online_augmentation.py`
  - Result: `74 passed`.
- Interpretation: paper-mode split and leakage controls are valid, and the accept-to-execution code path is now fixed and tested. The latest 10ep smoke did not contain an accept case, so it does not constitute a performance or applied-augmentation validation. A new paper-mode multiseed rerun is required before making paper-mode CP-CATF effectiveness claims.
<!-- CP_CATF_ACCEPT_TO_EXECUTION_AUDIT_END -->

<!-- CP_CATF_SEED0_EXECUTION_VALIDATION_START -->
## CP-CATF Paper-Mode Seed0 Execution Validation

- Date: `2026-06-11`.
- Scope: seed0 only; no clean rerun, no seed1/seed2, no multiseed summary.
- Command used `D:\Anaconda\envs\pytorch\python.exe` with:
  - `--data outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/data.yaml`
  - `--probe-data outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/probe.yaml`
  - `--epochs 50 --imgsz 1024 --batch 2 --workers 0 --device 0 --seed 0`
  - `--catf-version v2 --paper-probe-mode true --causal-probe-mode true`
  - class-aware feedback, ROI-aware augmentation, sample-aware routing, threshold calibration, feedback interval/start epoch 5, industrial augmentation, and final-val policy-selection guard enabled.
- Run root: `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/`.
- Execution flow:
  - epoch 25 accepted `candidate_policy_1_roi_texture`.
  - active class: `9`.
  - executable ops injected: `sharpen_mild(prob=0.20,strength=0.22)` and `local_contrast(prob=0.18,strength=0.20)`.
  - online stats: industrial samples augmented `226`; router random draw count `1330`.
  - ROI stats: ROI applied `312`, affected class `9`.
  - bbox/class checks: invalid bbox `0`, bbox OOB `0`, class OOB `0`.
  - final-val leakage: `false`.
- Final seed0 metrics:
  - CP-CATF: P=0.739931, R=0.676311, mAP50=0.759345, mAP50-95=0.502791.
  - Reused clean paper seed0: P=0.751343, R=0.676301, mAP50=0.756646, mAP50-95=0.511423.
  - Delta: dP=-0.011412, dR=+0.000010, dmAP50=+0.002699, dmAP50-95=-0.008631.
  - Constraint: `constraint_failed=true`, reason `precision_drop_gt_0.01`.
- Reports:
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_execution_flow_report.md`
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_execution_flow_report.json`
- Verification before training:
  - py_compile checks passed for causal probe, training entry, sample router, and policy matrix.
  - Targeted pytest result: `65 passed`.
- Interpretation: paper-mode CP-CATF accept-to-execution is confirmed on seed0, but the seed0 precision drop means this should not yet be treated as a paper-mode performance success.
<!-- CP_CATF_SEED0_EXECUTION_VALIDATION_END -->

<!-- CP_CATF_SEED0_PRECISION_RISK_AUDIT_START -->
## CP-CATF Paper-Mode Seed0 Precision-Risk Audit

- Date: `2026-06-11`.
- Scope: analysis only; no training, no clean rerun, no seed1/seed2, no multiseed.
- Script: `scripts/audit_seed0_cp_catf_precision_risk.py`.
- Command: `D:\Anaconda\envs\pytorch\python.exe scripts/audit_seed0_cp_catf_precision_risk.py --device 0`.
- Inputs:
  - clean seed0: `outputs/experiments/multiseed_cp_catf_paper_mode/clean_seed_0/`.
  - CP-CATF seed0: `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/`.
  - final val/probe split: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/`.
- Outputs:
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_precision_risk_audit.md`.
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_precision_risk_audit.json`.
  - prediction caches under `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/precision_risk_predictions/`.
- Key metric context:
  - clean seed0 P/R/mAP50/mAP50-95: `0.7513/0.6763/0.7566/0.5114`.
  - CP-CATF seed0: `0.7399/0.6763/0.7593/0.5028`.
  - constraint failure reason: `precision_drop_gt_0.01`.
- Prediction-level finding at conf `0.25`:
  - FP `344 -> 354` (`+10`).
  - TP `716 -> 719` (`+3`).
  - FN `189 -> 186` (`-3`).
- Main Precision-drop drivers:
  - class5 FP `3 -> 17` (`+14`), metric Precision `-0.1481`.
  - class6 FP `33 -> 45` (`+12`), metric Precision `-0.1056`.
  - class8 FP `16 -> 22` (`+6`), metric Precision `-0.0721`.
  - class2 FP `1 -> 4` (`+3`), metric Precision `-0.1217`.
- Class9 result:
  - class9 was the active ROI texture class.
  - class9 metric Precision improved `0.7214 -> 0.8376`; AP50 improved `+0.0502`; AP50-95 improved `+0.0228`.
  - matched class9 FP decreased `50 -> 38`.
  - Therefore class9 is not the primary Precision-drop class; the issue is non-active FP spillover.
- Threshold calibration:
  - final-val diagnostic threshold can restore Precision but is leakage-only.
  - probe-based threshold selected on probe did not restore final-val Precision within clean-minus-0.01.
- Conclusion: accept-to-execution works, but CP-CATF needs a precision-aware accept gate before more paper-mode performance validation.
- Verification: `python -m py_compile scripts/audit_seed0_cp_catf_precision_risk.py`.
<!-- CP_CATF_SEED0_PRECISION_RISK_AUDIT_END -->

<!-- CP_CATF_PRECISION_GATE_SEED0_VALIDATION_START -->
## CP-CATF Precision-Aware Gate Dry Run and Seed0 Rerun

- Date: `2026-06-12`.
- Scope: seed0 only. No seed1, seed2, clean rerun, or multiseed run was started.
- Code additions:
  - `--precision-aware-accept-gate` CLI support in `scripts/train_yolo_default_with_inloop_feedback.py`.
  - `scripts/run_cp_catf_precision_gate_dry_run_seed0.py`.
  - `scripts/summarize_cp_catf_precision_gate_seed0_rerun.py`.
- Dry run:
  - Input: `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/`.
  - Output: `outputs/experiments/cp_catf_precision_gate_dry_run_seed0/`.
  - Replayed epoch 25 from the seed0 paper-mode execution-fixed run.
  - Original candidate: `candidate_policy_1_roi_texture`.
  - Original active class: `9`.
  - Original ops: `sharpen_mild` and `local_contrast`.
  - Original execution counts: ROI applied `312`, industrial image augmented `226`, router random draw count `1330`.
- Dry-run precision-aware result:
  - roi_texture rejected: `true`.
  - Rejection reasons include `estimated_precision_drop_too_high`, `non_active_fp_delta_too_high`, and `high_confidence_fp_delta_too_high`.
  - estimated_precision_drop=`0.0300`.
  - non_active_fp_delta=`0.0500`.
  - high_confidence_fp_delta=`0.0500`.
  - Selected fallback: `candidate_policy_3_sampler_only`.
  - Sampler-only remains pending dataloader support, so the image path is strict no-op.
  - final validation leakage: `false`.
  - Decision does not depend on seed id, fixed class id, or RiskGuard final blocking.
- Seed0 50ep rerun:
  - Output: `outputs/experiments/cp_catf_paper_mode_precision_gate_seed0_rerun/`.
  - Command used paper-mode train/probe split, `--causal-probe-mode true`, `--precision-aware-accept-gate true`, and `--forbid-final-val-policy-selection true`.
  - Feedback epochs 5/10/15/20/25/30/35/40/45 all selected `candidate_policy_3_sampler_only`.
  - No image candidate was accepted.
  - ROI applied `0`.
  - Industrial image augmented `0`.
  - Router random draw count `0`.
  - BBox/class legal: `true`.
  - Final val leakage: `false`.
- Metrics against the requested paper clean seed0 baseline:
  - paper clean seed0: P=0.7513, R=0.6763, mAP50=0.7566, mAP50-95=0.5114.
  - precision-gate rerun: P=0.7513, R=0.6763, mAP50=0.7566, mAP50-95=0.5114.
  - delta at report precision: all `0.0000`.
  - `constraint_failed=false`.
- Interpretation:
  - The precision-aware gate successfully blocks the known seed0 FP-spillover image candidate before training.
  - The rerun is safety/no-op evidence, not image-augmentation benefit evidence, because all image candidates were rejected.
  - Sampler-only was later connected to the dataloader and verified, but is now demoted because it is sampling reweighting, not image augmentation.
  - Do not proceed to formal seed1/seed2 image-augmentation validation until an image candidate can pass the precision-aware gate or be safely attenuated inside the image path.
- Reports:
  - `outputs/experiments/cp_catf_precision_gate_dry_run_seed0/reports/precision_gate_dry_run_report.md`
  - `outputs/experiments/cp_catf_precision_gate_dry_run_seed0/reports/precision_gate_dry_run_report.json`
  - `outputs/experiments/cp_catf_paper_mode_precision_gate_seed0_rerun/reports/seed0_precision_gate_rerun_report.md`
  - `outputs/experiments/cp_catf_paper_mode_precision_gate_seed0_rerun/reports/seed0_precision_gate_rerun_report.json`
- Verification:
  - `python -m py_compile AutoAugment/catf_v2/causal_probe.py scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py scripts/run_cp_catf_precision_gate_dry_run_seed0.py scripts/summarize_cp_catf_precision_gate_seed0_rerun.py`
  - `pytest -q tests/test_catf_v2_causal_probe.py tests/test_cp_catf_accept_to_execution.py tests/test_cp_catf_paper_mode.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_policy_matrix.py tests/test_catf_v2_sample_router.py tests/test_catf_v2_roi_augmentation.py tests/test_inloop_feedback_training.py tests/test_online_augmentation.py`
  - Result: `70 passed`.
<!-- CP_CATF_PRECISION_GATE_SEED0_VALIDATION_END -->

<!-- IMAGE_ONLY_WEAK_AUG_MULTISEED_SANITY_START -->
## Image-Only Weak Augmentation Seed0/Seed1 Sanity

- Date: `2026-06-14`.
- Request scope:
  - run seed0 and seed1 only;
  - reuse completed seed2 weak image augmentation result;
  - do not rerun clean;
  - do not use sampler_only or weighted index list;
  - do not change data split, gate threshold, attenuation ratio, causal score, or augmentation strategy.
- Commands used `D:\Anaconda\envs\pytorch\python.exe`.
- Pre-training validation:
  - `python -m py_compile scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/causal_probe.py AutoAugment/catf_v2/policy_matrix.py AutoAugment/catf_v2/sample_router.py`
  - `pytest -q tests/test_catf_v2_causal_probe.py tests/test_cp_catf_accept_to_execution.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_policy_matrix.py tests/test_catf_v2_sample_router.py tests/test_catf_v2_roi_augmentation.py tests/test_online_augmentation.py`
  - Result: `55 passed`.
- Offline weak schedules:
  - seed0: weak ROI texture at epoch `25`, strict no-op for the other feedback epochs.
  - seed1: weak ROI texture at epochs `25` and `40`, strict no-op for the other feedback epochs.
  - sampler_only involved: `false`; final-val leakage flag in decisions: `false`.
- Seed0 command used:
  - `--project outputs/experiments/catf_v2_image_only_weak_aug_multiseed --run-id seed0 --seed 0`
  - `--catf-version v2 --image-only-mainline true --weak-image-aug-enabled true --attenuation-ratio 0.25 --disable-sampler-only true --sampler-only-enabled false`
  - `--use-offline-probe-decisions true --offline-probe-decisions-file outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed0/configs/weak_image_aug_offline_decisions_seed0.json`
- Seed0 result:
  - 50ep completed.
  - Weak image augmentation executed: `true`.
  - Industrial images augmented `16`; ROI applied `18`; router random draw count `341`.
  - `sampler_only_enabled=false`; `weighted_index_list_enabled=false`; `sampled_distribution_changed=false`.
  - Metrics P/R/mAP50/mAP50-95: `0.679043/0.711821/0.722170/0.476005`.
  - Delta vs clean seed0: `-0.105511/+0.035364/-0.012525/+0.000111`.
  - Delta vs fixed CATF-v2 seed0: `-0.099463/+0.042167/-0.021487/-0.013536`.
  - Constraint: `constraint_failed=true`, reasons `precision_drop_gt_0.01`, `map50_drop_gt_0.01`.
- Seed1 command used:
  - `--project outputs/experiments/catf_v2_image_only_weak_aug_multiseed --run-id seed1 --seed 1`
  - same image-only weak augmentation flags as seed0.
  - `--use-offline-probe-decisions true --offline-probe-decisions-file outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed1/configs/weak_image_aug_offline_decisions_seed1.json`
- Seed1 result:
  - 50ep completed.
  - Weak image augmentation executed: `true`.
  - Industrial images augmented `32`; ROI applied `38`; router random draw count `775`.
  - `sampler_only_enabled=false`; `weighted_index_list_enabled=false`; `sampled_distribution_changed=false`.
  - Metrics P/R/mAP50/mAP50-95: `0.765605/0.712074/0.776844/0.506815`.
  - Delta vs clean seed1: `-0.006920/+0.064394/+0.022653/+0.026896`.
  - Delta vs fixed CATF-v2 seed1: `-0.019594/+0.011564/-0.005773/-0.012064`.
  - Constraint: `constraint_failed=false`; fixed CATF-v2 mAP50-95 gain not fully retained within 0.01.
- Reused seed2 weak image augmentation:
  - Run root: `outputs/experiments/catf_v2_image_only_weak_aug_seed2_50ep/`.
  - Metrics P/R/mAP50/mAP50-95: `0.753254/0.694235/0.772718/0.515138`.
  - Delta vs clean seed2: `+0.057015/-0.034395/+0.003515/-0.007233`.
  - Constraint: `constraint_failed=false`; Recall warning remains.
- Multiseed summary:
  - Constraint pass count: `2/3`; `3/3 pass=false`.
  - Mean P/R/mAP50/mAP50-95: `0.732634/0.706043/0.757244/0.499319`.
  - Mean delta vs clean: `-0.018472/+0.021788/+0.004548/+0.006591`.
  - Mean delta vs fixed CATF-v2: `-0.043154/+0.020541/-0.004263/-0.002382`.
  - Total industrial images augmented `128`; total ROI applied `151`.
  - sampler_only involved `false`; weighted index list involved `false`; sampled distribution changed `false`; final val used for policy selection `false`.
- Conclusion:
  - Weak image augmentation repaired seed2 and seed1 passes constraints.
  - Seed0 fails hard constraints, so the exact current image-only weak augmentation setting is not a 3-seed paper main-method candidate.
  - Next work should stay image-only and add recall/precision-aware safety; sampler_only should remain demoted to ablation/exploration.
- Reports:
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed0/reports/seed0_weak_image_aug_report.md`
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed1/reports/seed1_weak_image_aug_report.md`
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/reports/weak_image_aug_multiseed_summary.md`
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/reports/weak_image_aug_multiseed_summary.json`
- Summary script:
  - Added `scripts/summarize_weak_image_aug_multiseed.py`.
  - `python -m py_compile scripts/summarize_weak_image_aug_multiseed.py` passed.
<!-- IMAGE_ONLY_WEAK_AUG_MULTISEED_SANITY_END -->

<!-- SEED0_FIXED_VS_WEAK_FAILURE_AUDIT_START -->
## Seed0 Fixed-vs-Weak Image Augmentation Failure Audit

- Date: `2026-06-14`.
- Scope:
  - analysis only;
  - no training run;
  - no seed1/seed2 run;
  - no multiseed run;
  - no sampler_only or weighted index list;
  - no gate, attenuation-ratio, causal-score, augmentation-strategy, or data-split change.
- Script:
  - `scripts/analyze_seed0_fixed_vs_weak.py`
- Inputs:
  - fixed seed0: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/seed_0/catf_v2/`
  - weak seed0: `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed0/`
  - clean seed0 metrics reused from existing clean run.
- Outputs:
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/reports/seed0_fixed_vs_weak_failure_audit.md`
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/reports/seed0_fixed_vs_weak_failure_audit.json`
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed0_fixed_vs_weak_epoch_policy_diff.csv`
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed0_fixed_vs_weak_per_class_regression.csv`
- Aggregate metrics:
  - clean seed0 P/R/mAP50/mAP50-95: `0.784554/0.676457/0.734696/0.475894`;
  - fixed CATF-v2 seed0: `0.778506/0.669654/0.743657/0.489541`, pass;
  - weak image augmentation seed0: `0.679043/0.711821/0.722170/0.476005`, fail.
- Execution contrast:
  - fixed seed0 augmented classes `4`, `11`, and `12`;
  - fixed seed0 industrial images augmented `41`, ROI applied `45`;
  - weak seed0 executed only epoch `25` class `9` weak local-contrast;
  - weak seed0 industrial images augmented `16`, ROI applied `18`;
  - sampler_only involved `false`, weighted index list enabled `false`, sampled distribution changed `false`.
- Main finding:
  - weak seed0 failed because the weak replay path globally replaced fixed CATF-v2 behavior instead of preserving the seed0 safe original policies;
  - the weak path suppressed fixed classes `4/11/12` and introduced a class `9` weak candidate;
  - Precision collapse is FP-driven and broad, not isolated to the active class.
- Per-class regression:
  - largest weak precision drops vs clean include classes `5`, `4`, `3`, `11`, `9`, and `7`;
  - largest estimated FP increases vs fixed are led by classes `7`, `9`, `5`, `12`, and `6`;
  - this indicates non-active regression/spillover.
- Interpretation:
  - weak augmentation is not proven globally harmful, because it repaired seed2 and seed1 passes;
  - the current global weak replacement is harmful for seed0;
  - seed0 should preserve fixed original policy when original fixed CATF-v2 policy is low-risk;
  - next image-only strategy should be `preserve-safe-original + weak-only-for-moderate-risk + strict no-op for high/critical risk`;
  - do not continue training until that decision logic is replayed offline.
- Verification:
  - `D:\Anaconda\envs\pytorch\python.exe -m py_compile scripts\analyze_seed0_fixed_vs_weak.py`
<!-- SEED0_FIXED_VS_WEAK_FAILURE_AUDIT_END -->

<!-- PRESERVE_WEAK_IMAGE_CATF_REPLAY_START -->
## Preserve-Original + Weak-Only Image CATF Replay

- Date: `2026-06-14`.
- Scope:
  - offline replay only;
  - no training;
  - no seed0/seed1/seed2 rerun;
  - no multiseed training;
  - no sampler_only or weighted index list;
  - no gate, attenuation-ratio, causal-score, augmentation-strategy, or data-split change.
- Script:
  - `scripts/replay_preserve_weak_image_catf.py`
- Inputs:
  - fixed CATF-v2 multiseed summary: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.json`;
  - weak image replay table: `outputs/experiments/catf_v2_image_only_weak_aug_replay/weak_candidate_records.csv`;
  - seed0 fixed-vs-weak audit: `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/reports/seed0_fixed_vs_weak_failure_audit.json`;
  - weak multiseed summary: `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/reports/weak_image_aug_multiseed_summary.json`.
- Outputs:
  - `outputs/experiments/catf_v2_image_only_preserve_weak_replay/reports/preserve_weak_replay.md`;
  - `outputs/experiments/catf_v2_image_only_preserve_weak_replay/reports/preserve_weak_replay.json`;
  - `outputs/experiments/catf_v2_image_only_preserve_weak_replay/preserve_weak_decision_records.csv`;
  - `outputs/experiments/catf_v2_image_only_preserve_weak_replay/reports/preserve_weak_training_plan.md`.
- Decision logic:
  - `preserve_original` first if fixed CATF-v2 already passed constraints and has executable conservative image policy;
  - `weak_roi_texture` only if fixed original is not preservable and attenuation `0.25` passes the secondary replay gate;
  - `strict_noop` for high/critical image risk;
  - `sampler_only_used=false` for every replay record.
- Replay counts:
  - seed0: `preserve_original=9`, `weak_roi_texture=0`, `strict_noop=0`;
  - seed1: `preserve_original=9`, `weak_roi_texture=0`, `strict_noop=0`;
  - seed2: `preserve_original=0`, `weak_roi_texture=5`, `strict_noop=4`.
- Expected-behavior checks:
  - seed0 preserves fixed class `4/11/12` strategy: `true`;
  - seed0 avoids weak class `9` replacement: `true`;
  - seed1 preserves fixed gain policy: `true`;
  - seed2 converts failed fixed policy to weak/no-op: `true`;
  - seed2 contains previous weak-safe candidates: `true`;
  - sampler_only absent: `true`.
- Interpretation:
  - weak global replacement failed because it could override safe fixed behavior;
  - the new replay restores the image-only mainline as preserve-safe-original plus weak-only-for-moderate-risk;
  - seed0 should be the first training sanity target if this policy is implemented, because it is the regression case;
  - seed2 should be second to verify the failed fixed path remains repaired without sampler_only;
  - seed1 sanity should be last.
- Verification:
  - `D:\Anaconda\envs\pytorch\python.exe -m py_compile scripts\replay_preserve_weak_image_catf.py`
<!-- PRESERVE_WEAK_IMAGE_CATF_REPLAY_END -->
