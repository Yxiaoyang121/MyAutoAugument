# Project Snapshot

- Generated: 2026-06-09T01:45:00
- Branch: codex/sync-latest
- Commit: e7b894d93ef23f8f8bdffbf20e4b7ce8127d7aec
- Remote: https://github.com/Yxiaoyang121/MyAutoAugument.git

## Working Tree

```text
 M CODEX_HANDOFF.md
 M EXPERIMENT_LOG.md
 M PROJECT_STATE.md
 M outputs/project_snapshot_latest.md
 M outputs/snapshots/project_snapshot_latest.md
 M scripts/train_yolo_default_with_inloop_feedback.py
 M tests/test_catf_v2_causal_probe.py
?? outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_cp_catf/
?? scripts/summarize_catf_v2_cp_catf_multiseed.py
```

## Key Files

- README.md
- PROJECT_STATE.md
- CODEX_HANDOFF.md
- EXPERIMENT_LOG.md
- AGENTS.md
- docs/ARCHITECTURE_CURRENT.md
- docs/diagnostic_augmentation_framework.md
- docs/experiment_protocol.md
- docs/output_convention.md
- scripts/audit_dataset_mapping.py
- scripts/audit_tiling_quality.py
- scripts/filter_tiled_dataset.py
- scripts/audit_artifacts.py
- scripts/build_yolo_tiled_dataset.py
- scripts/run_gpu_preflight.py
- scripts/run_diagnostic_augmentation_pipeline.py
- scripts/generate_top3_policies_from_baseline.py
- scripts/run_top3_policy_short_training.py
- scripts/run_counterfactual_diagnosis.py
- scripts/run_per_class_diagnosis.py
- scripts/run_class_aware_policy_short_training.py
- scripts/run_random_external_aug_50ep.py
- scripts/run_diagnosis_guided_policy_search.py
- scripts/run_search_policy_017_50ep.py
- scripts/train_yolo_online_aug.py
- scripts/run_yolo_default_diagnosis_constraints.py
- scripts/train_yolo_default_with_feedback.py
- scripts/train_yolo_default_with_inloop_feedback.py
- scripts/audit_yolo_default_inloop_parity.py
- scripts/run_clean_native_yolo_default.py
- scripts/write_yolo_default_aug_50ep_reports.py
- configs/online_policies/yolo_like_base_policy.json
- configs/online_policies/industrial_diag_policy.json
- configs/online_policies/feedback_online_policy.json
- configs/online_policies/yolo_default_passthrough_policy.json
- configs/online_policies/diagnosis_light_policy.json
- configs/online_policies/diagnosis_precision_safe_policy.json
- configs/online_policies/diagnosis_recall_safe_policy.json
- outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.md
- outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.json
- outputs/audits/tiling_quality/tiling_quality_audit.md
- outputs/audits/tiling_quality/tiling_quality_audit.json
- outputs/audits/dataset_mapping/dataset_mapping_audit.md
- outputs/audits/dataset_mapping/dataset_mapping_audit.json
- outputs/audits/gpu_preflight/gpu_preflight_report.md
- outputs/audits/gpu_preflight/gpu_preflight_report.json
- outputs/audits/artifact_inventory/artifact_inventory.md
- outputs/experiments/20260517_tiled_baseline_20epoch/reports/summary.md
- outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_report.md
- outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_metrics.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_metrics.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/diagaug_50ep_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/diagaug_50ep_metrics.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/baseline_vs_diagaug.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/reports/yolo_default_aug_50ep_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/reports/yolo_default_aug_50ep_metrics.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/reports/compare_baseline_yolo_default_diagaug.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/diagnosis/diagnosis.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/diagnosis/diagnosis_summary.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/policies/candidate_policies.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/policies/candidate_policies.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/proxy/proxy_ranking.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/proxy/proxy_ranking.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/proxy/proxy_safety_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/top3_policies/top3_policies.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/top3_policies/top3_policies.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/reports/policy_selection_trace.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/reports/policy_selection_trace.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_shorttrain/reports/top3_policy_shorttrain_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_shorttrain/reports/top3_policy_shorttrain_results.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/reports/counterfactual_diagnosis_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/counterfactual_summary.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/counterfactual_instances.csv
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/counterfactual_policy_ranking.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_per_class_diagnosis/reports/per_class_diagnosis_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_per_class_diagnosis/reports/per_class_diagnosis.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_per_class_diagnosis/reports/classwise_recommendations.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_per_class_diagnosis/reports/classwise_recommendations.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_per_class_diagnosis/policies/class_aware_mixed_policy.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_per_class_diagnosis/policies/class_aware_policy_score.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain/reports/class_aware_shorttrain_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain/reports/class_aware_vs_diag_policy_001_shorttrain.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain/metrics/class_aware_shorttrain_metrics.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/random_policy.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/random_augmented_dataset_report.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/proxy_safety_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/reports/random_external_aug_50ep_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/reports/random_external_aug_50ep_metrics.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/reports/compare_baseline_yolo_default_diagaug_random.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/diagnosis_source.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/policies/candidate_policies.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/policies/candidate_policies.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/proxy/proxy_ranking.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/proxy/proxy_safety_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/reports/policy_search_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/reports/policy_search_results.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/reports/best_policy_summary.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/policies/search_policy_017.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/dataset/augmented_dataset_summary.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/dataset/augmented_dataset_summary.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/final_training/final_train_metrics.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/final_training/final_val_metrics.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/reports/search_policy_017_50ep_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/reports/search_policy_017_50ep_metrics.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/reports/compare_baseline_diagaug_random_yolo_search017.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_smoke/reports/online_aug_smoke_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_smoke/reports/online_aug_stats.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_smoke/configs/train_config.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/reports/online_diag_policy_001_50ep_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/reports/online_diag_policy_001_50ep_metrics.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/reports/compare_online_offline_yolo_default_random.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/reports/online_aug_stats.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/configs/train_config.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/configs/online_random_like_policy.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/reports/online_random_like_50ep_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/reports/online_random_like_metrics.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/reports/compare_online_random_like_with_all.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/reports/online_aug_stats.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/configs/train_config.json
- outputs/experiments/online_yolo_like_base_smoke/reports/online_aug_smoke_report.md
- outputs/experiments/online_yolo_like_base_smoke/reports/online_aug_stats.json
- outputs/experiments/online_yolo_like_base_smoke/configs/train_config.json
- outputs/experiments/feedback_online_policy_2stage_smoke/reports/online_aug_smoke_report.md
- outputs/experiments/feedback_online_policy_2stage_smoke/reports/online_aug_stats.json
- outputs/experiments/feedback_online_policy_2stage_smoke/reports/policy_history.json
- outputs/experiments/feedback_online_policy_2stage_smoke/reports/policy_history.md
- outputs/experiments/feedback_online_policy_2stage_smoke/reports/policy_history.csv
- outputs/experiments/feedback_online_policy_2stage_smoke/configs/train_config.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/custom_yolo_like_base_50ep_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/custom_yolo_like_base_50ep_metrics.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/compare_custom_yolo_like_with_yolo_default.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/online_aug_stats.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/policy_history.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/configs/train_config.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/reports/diagnosis_constrained_experiment_report.md
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/reports/diagnosis_constrained_metrics.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/reports/constraint_scoring.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/reports/operator_impact.json
- outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/reports/unified_comparison.csv
- outputs/experiments/yolo_default_feedback_aug_50ep/reports/yolo_default_feedback_smoke_report.md
- outputs/experiments/yolo_default_feedback_aug_50ep/reports/final_metrics.json
- outputs/experiments/yolo_default_feedback_aug_50ep/reports/policy_history.json
- outputs/experiments/yolo_default_feedback_aug_50ep/reports/constraint_scoring.json
- outputs/experiments/yolo_default_feedback_aug_50ep_full/reports/final_report.md
- outputs/experiments/yolo_default_feedback_aug_50ep_full/reports/final_metrics.json
- outputs/experiments/yolo_default_feedback_aug_50ep_full/reports/stage_metrics.json
- outputs/experiments/yolo_default_feedback_aug_50ep_full/reports/policy_history.json
- outputs/experiments/yolo_default_feedback_aug_50ep_full/reports/constraint_scoring.json
- outputs/experiments/yolo_default_feedback_aug_50ep_full/reports/compare_with_yolo_default_baseline_diagaug_random.md
- outputs/experiments/yolo_default_inloop_feedback_10ep_smoke/reports/inloop_feedback_smoke_report.md
- outputs/experiments/yolo_default_inloop_feedback_10ep_smoke/reports/policy_history.json
- outputs/experiments/yolo_default_inloop_feedback_10ep_smoke/reports/policy_history.md
- outputs/experiments/yolo_default_inloop_feedback_10ep_smoke/reports/policy_history.csv
- outputs/experiments/yolo_default_inloop_feedback_10ep_smoke/reports/online_aug_stats.json
- outputs/experiments/yolo_default_inloop_no_feedback_control_50ep/reports/inloop_no_feedback_control_report.md
- outputs/experiments/yolo_default_inloop_no_feedback_control_50ep/reports/inloop_no_feedback_control_metrics.json
- outputs/experiments/yolo_default_inloop_no_feedback_control_50ep/reports/compare_with_yolo_default_reference.md
- outputs/experiments/yolo_default_inloop_feedback_50ep_full/reports/final_report.md
- outputs/experiments/yolo_default_inloop_feedback_50ep_full/reports/final_metrics.json
- outputs/experiments/yolo_default_inloop_feedback_50ep_full/reports/policy_history.json
- outputs/experiments/yolo_default_inloop_feedback_50ep_full/reports/constraint_scoring.json
- outputs/experiments/yolo_default_inloop_feedback_50ep_full/reports/compare_with_yolo_default_no_feedback.md
- outputs/audits/yolo_default_inloop_parity/parity_audit_report.md
- outputs/audits/yolo_default_inloop_parity/parity_audit.json
- outputs/experiments/clean_native_yolo_default_seed42_50ep/reports/clean_native_yolo_default_report.md
- outputs/experiments/clean_native_yolo_default_seed42_50ep/reports/clean_native_yolo_default_metrics.json
- outputs/experiments/yolo_default_inloop_feedback_clean_reference_50ep/reports/final_report.md
- outputs/experiments/yolo_default_inloop_feedback_clean_reference_50ep/reports/final_metrics.json
- outputs/experiments/yolo_default_inloop_feedback_clean_reference_50ep/reports/policy_history.json
- outputs/experiments/yolo_default_inloop_feedback_clean_reference_50ep/reports/policy_history.md
- outputs/experiments/yolo_default_inloop_feedback_clean_reference_50ep/reports/constraint_scoring.json
- outputs/experiments/yolo_default_inloop_feedback_clean_reference_50ep/reports/compare_with_clean_native_yolo_default.md
- outputs/datasets/tiled/tiled_1024_ov20_smoke/dataset_summary.md
- outputs/datasets/tiled/tiled_1024_ov20_full/data.yaml
- outputs/datasets/tiled/tiled_1024_ov20_full/dataset_summary.md
- outputs/datasets/tiled/tiled_1024_ov20_full/tiled_dataset_report.md
- outputs/datasets/tiled/tiled_1024_ov20_full/tiled_dataset_report.json
- outputs/datasets/tiled/tiled_1024_ov20_full_safe/data.yaml
- outputs/datasets/tiled/tiled_1024_ov20_full_safe/dataset_summary.md
- outputs/datasets/tiled/tiled_1024_ov20_full_safe/tiled_dataset_report.md
- outputs/datasets/tiled/tiled_1024_ov20_full_safe/tiled_dataset_report.json
- outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml
- outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/class_filter_report.md
- outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/class_filter_report.json
- outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/dataset_summary.md
- outputs/snapshots/project_snapshot_latest.md
- outputs/project_snapshot_latest.md
- AutoAugment/online_augmentation.py
- AutoAugment/feedback_policy_controller.py
- AutoAugment/diagnostic_pipeline/__init__.py
- AutoAugment/diagnostic_pipeline/strategy_memory.py
- AutoAugment/diagnostic_pipeline/metric_audit.py
- AutoAugment/diagnostic_pipeline/proxy_evaluation.py
- AutoAugment/diagnostic_pipeline/policy_mapping.py
- tests/test_online_augmentation.py
- tests/test_feedback_policy_controller.py

## Tracked File Count

- 6073 tracked files

## Notes

- This snapshot reflects the current local repository state.
- It does not invent benchmark results.

## Latest CP-CATF Status

- CP-CATF multiseed training validation completed at `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_cp_catf/`.
- RiskGuard is audit/debug prior only, not the final training accept/reject rule.
- Development-mode offline probe decisions were used:
  - seed0 accepted `candidate_policy_1_roi_texture`.
  - seed1 accepted `candidate_policy_1_roi_texture`.
  - seed2 selected `candidate_policy_3_sampler_only`; sample weighting is pending dataloader support, so the actual path is strict image no-op.
- Final CP-CATF metrics:
  - seed0: P=0.7785, R=0.6697, mAP50=0.7437, mAP50-95=0.4895, `constraint_failed=false`.
  - seed1: P=0.7852, R=0.7005, mAP50=0.7826, mAP50-95=0.5189, `constraint_failed=false`.
  - seed2: P=0.6962, R=0.7286, mAP50=0.7692, mAP50-95=0.5224, `constraint_failed=false`.
- Outcome: `3/3` constraint pass; seed0/seed1 retained fixed CATF-v2 gains; seed2 rejected image augmentation with industrial samples=0, ROI=0, router random draws=0; OK3 remained inactive.
- Reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_cp_catf/reports/multiseed_cp_catf_summary.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_cp_catf/reports/multiseed_cp_catf_summary.json`
- Verification: requested py_compile checks passed and requested pytest suite result was `134 passed`.
- Caveat: current CP-CATF validation uses development-mode offline probe decisions from existing validation diagnostics. Paper-mode CP-CATF still needs a train/probe split or train hard-example probe set before leakage-free final claims.

## Latest CP-CATF Paper-Mode Status

- Paper-mode CP-CATF validation completed at `outputs/experiments/multiseed_cp_catf_paper_mode/`.
- Probe split root: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/`.
- Split seed `2026`; original train `2301`, train_core `2071`, probe `230`, final val `677`; train_core/probe/final-val overlap count `0`.
- Smoke run passed at `outputs/experiments/cp_catf_paper_mode_10ep_smoke/`, with policy selection from `probe_split` and final val excluded from strategy selection.
- Clean paper baseline:
  - seed0: P=0.7513, R=0.6763, mAP50=0.7566, mAP50-95=0.5114.
  - seed1: P=0.7220, R=0.7582, mAP50=0.7777, mAP50-95=0.5251.
  - seed2: P=0.6290, R=0.6385, mAP50=0.6590, mAP50-95=0.4381.
- CP-CATF paper-mode metrics match clean exactly for all three seeds; `constraint_failed=false` for seed0/1/2.
- Mean delta vs clean paper baseline: dP=+0.0000, dR=+0.0000, dmAP50=+0.0000, dmAP50-95=+0.0000.
- Actual CP-CATF paper-mode image augmentation did not execute: industrial samples augmented `0`, ROI applied `0`, router random draws `0` for all seeds.
- Final val leakage detected: `false`; policy history records `policy_selection_source=probe_split`.
- Outcome: `3/3` constraint pass, but no retained gain over the clean paper baseline. Paper-mode CP-CATF is not yet a paper main-result candidate; it is currently a leakage-free safety validation. Next work should strengthen paper-mode probe evidence with a larger probe split or train hard-example probe set.
- Reports:
  - `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/reports/probe_split_report.md`
  - `outputs/experiments/cp_catf_paper_mode_10ep_smoke/reports/paper_mode_smoke_report.md`
  - `outputs/experiments/multiseed_cp_catf_paper_mode/reports/multiseed_cp_catf_paper_mode_summary.md`
  - `outputs/experiments/multiseed_cp_catf_paper_mode/reports/multiseed_cp_catf_paper_mode_summary.json`
- Verification: requested py_compile checks passed and requested pytest suite result was `142 passed`.

## Latest CP-CATF Accept-to-Execution Audit

- Audit completed at `outputs/experiments/multiseed_cp_catf_paper_mode/reports/cp_catf_accept_to_execution_audit.md`.
- Existing paper-mode multiseed had `8` causal-probe image accept events but `0` executable active class-op policies after causal probe.
- Root cause: accepted candidates were reduced to an op whitelist and were not materialized into target-class policy entries with nonzero op probabilities/strengths. The sample router therefore had no eligible active policy and returned no-op before random draws.
- Fix implemented in `scripts/train_yolo_default_with_inloop_feedback.py`: accepted image candidates now inject executable class-op entries into the policy matrix; sampler-only and rejected candidates remain strict image no-op.
- Fixed smoke root: `outputs/experiments/cp_catf_paper_mode_execution_fixed_10ep_smoke/`.
- Fixed smoke selected `candidate_policy_3_sampler_only` at epoch 5 and no image accept occurred in 10 epochs, so industrial samples, ROI applications, and router random draws stayed `0`.
- Final-val leakage remained `false`; split overlaps remained `0`.
- Smoke reports:
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_10ep_smoke/reports/execution_fixed_smoke_report.md`
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_10ep_smoke/reports/execution_fixed_smoke_report.json`
- Verification: targeted pytest suite passed `74 passed`.

## Latest CP-CATF Seed0 Execution Validation

- Seed0-only paper-mode execution-fixed run completed at `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/`.
- This run did not rerun clean, seed1, seed2, or multiseed summaries.
- Epoch 25 accepted `candidate_policy_1_roi_texture` for class `9`.
- Executable policy was generated and executed: industrial samples augmented `226`, ROI applied `312`, router random draw count `1330`.
- Final-val leakage remained `false`.
- CP-CATF seed0 metrics: P=0.739931, R=0.676311, mAP50=0.759345, mAP50-95=0.502791.
- Reused clean paper seed0 metrics: P=0.751343, R=0.676301, mAP50=0.756646, mAP50-95=0.511423.
- Delta: dP=-0.011412, dR=+0.000010, dmAP50=+0.002699, dmAP50-95=-0.008631.
- Constraint result: `constraint_failed=true` due Precision drop greater than `0.01`.
- Reports:
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_execution_flow_report.md`
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_execution_flow_report.json`
