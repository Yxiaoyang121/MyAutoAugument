# Project Snapshot

- Generated: 2026-06-17T23:20:00
- Branch: codex/sync-latest
- Commit: pending seed0 volume-fixed sanity rerun
- Remote: https://github.com/Yxiaoyang121/MyAutoAugument.git

## Current Status

- Mainline remains image-only CATF; sampler_only is not part of the paper main method.
- Preserve volume/lifetime parity has been fixed and validated in a seed0 50ep rerun.
- This run used only seed0; seed1/seed2/multiseed were not run.
- Training used `D:\Anaconda\envs\pytorch\python.exe` because base Python has Ultralytics `8.4.48`; the guarded training path requires `8.3.221`.
- Execution: preserve_original/weak/noop=`9/0/0`.
- Epoch-exact executable classes: epoch5 `[4, 11]`, epoch15 `[12]`, epochs10/20/25/30/35/40/45 `[]`.
- stale ops cleared=true; seed-level union avoided=true; weak class9 replacement=false.
- sampler_only=false; weighted_index_list=false; sampled_distribution_changed=false.
- Volume: fixed expected `41 industrial / 45 ROI`; volume-fixed preserve got `41 industrial / 45 ROI`, ROI by class `{4:9, 11:22, 12:14}`.
- Metrics: P=0.778506, R=0.669654, mAP50=0.743657, mAP50-95=0.489541.
- Delta vs requested clean seed0: dP=-0.006094, dR=-0.006846, dM50=+0.008957, dM95=+0.013641.
- Constraint failed=false.
- Recommendation: seed0 now passes; next image-only step can be seed2 validation.
- Reports:
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/reports/seed0_preserve_weak_sanity_rerun_report.md`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/reports/seed0_preserve_weak_sanity_rerun_report.json`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/reports/preserve_volume_lifetime_audit.md`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/reports/preserve_volume_parity_dryrun.md`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/fixed_vs_preserve_volume_parity_epoch.csv`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/fixed_vs_preserve_volume_parity_by_class.csv`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_volume_fixed/reports/seed0_preserve_weak_volume_fixed_report.md`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_volume_fixed/reports/seed0_preserve_weak_volume_fixed_report.json`

## Working Tree

```text
 M CODEX_HANDOFF.md
 M EXPERIMENT_LOG.md
 M PROJECT_STATE.md
 M outputs/project_snapshot_latest.md
 M outputs/snapshots/project_snapshot_latest.md
 A outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_volume_fixed/
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

- 9442 tracked files

## Notes

- This snapshot reflects the current local repository state.
- It does not invent benchmark results.

## Latest Experiment State

- Seed0 preserve-weak image CATF sanity has passed after execution and volume parity fixes.
- Seed2 preserve-weak image CATF completed at `outputs/experiments/catf_v2_image_only_preserve_weak_seed2/`.
- Seed2 decisions: preserve_original=`0`, weak_roi_texture=`5`, strict_noop=`4`.
- Seed2 execution: industrial images augmented=`80`, ROI applied=`95`, sampler_only=`false`, weighted index list=`false`, sampled distribution changed=`false`.
- Seed2 metrics: P/R/mAP50/mAP50-95=`0.753254/0.694235/0.772718/0.515138`, `constraint_failed=false`.
- Delta vs clean seed2: `+0.057054/-0.034365/+0.003518/-0.007262`.
- Delta vs fixed CATF-v2 seed2: `-0.010446/+0.007935/+0.014518/+0.018438`.
- Seed1 preserve-weak sanity completed at `outputs/experiments/catf_v2_image_only_preserve_weak_seed1_sanity/`.
- Seed1 decisions: preserve_original=`9`, weak_roi_texture=`0`, strict_noop=`0`.
- Seed1 metrics: P/R/mAP50/mAP50-95=`0.799748/0.697375/0.778737/0.516923`, `constraint_failed=false`.
- 3-seed preserve-weak summary is available at `outputs/experiments/catf_v2_image_only_preserve_weak_multiseed_summary/reports/preserve_weak_3seed_summary.md`.
- Current image-only mainline status: hard-constraint pass count is `3/3`; image-only preserve-weak CATF is a viable current main-method candidate.
- Remaining limitation: seed2 Recall remains below clean and should be reported as `recall_warning=true`.
- Sampler_only remains demoted to engineering exploration/ablation and is not part of the main method.
- Paper-ready evidence package has been generated under `outputs/experiments/catf_v2_image_only_preserve_weak_multiseed_summary/`.
- Main result tables are in `tables/main_result_table.*`; ablation tables are in `tables/ablation_table.*`.
- Paper-facing reports include `method_logic_for_paper.md`, `limitations_and_next_step.md`, `reviewer_risk_check.md`, and `preserve_weak_3seed_summary_paper_ready.*`.
- Next work should not be blind training; only targeted image-only recall-aware extensions are recommended if needed.
