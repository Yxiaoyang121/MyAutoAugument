# Codex Handoff

## Repository

- Path: `E:\TJGY\MinPaper\MyAutoAugument`
- Branch: `codex/sync-latest`
- Remote: `https://github.com/Yxiaoyang121/MyAutoAugument.git`

## Current Scope

The project is a diagnosis-driven augmentation pipeline for industrial defect detection. Keep work centered on dataset construction, tiling, validation-error diagnosis, policy generation, proxy safety, short-training validation, and auditable artifacts. Do not reframe this as YOLO backbone, neck, or head redesign.

## Current CATF-v2 Finding

- CATF-v2-Gated has been implemented with `--catf-gated-mode true`; Safe remains available separately as `--catf-safe-mode true`.
- Retrospective gate simulation report:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/gated_controller_retrospective_simulation.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/gated_controller_retrospective_simulation.json`
- Retrospective replay predicted seed0 no fallback, seed1 no fallback, seed2 fallback at epoch 10, and expected `3/3` pass. This prediction did not hold in the real single-run training.
- Full CATF-v2-Gated multiseed run completed at `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/`.
- CATF-v2-Gated metrics:
  - seed0: Precision=0.7785, Recall=0.6697, mAP50=0.7437, mAP50-95=0.4895, `constraint_failed=false`.
  - seed1: Precision=0.7852, Recall=0.7005, mAP50=0.7826, mAP50-95=0.5189, `constraint_failed=false`.
  - seed2: Precision=0.7850, Recall=0.6795, mAP50=0.7521, mAP50-95=0.5083, `constraint_failed=true`.
- Gated retained fixed CATF-v2 gains on seed0 and seed1:
  - seed0 industrial samples=41, ROI applied=45.
  - seed1 industrial samples=55, ROI applied=56.
- Seed2 did trigger fallback at epoch 10 with reason `epoch10_bad_pattern_A`; saved active policies from epoch 10 onward have all op probabilities zero.
- Seed2 still failed constraints because tentative augmentation between epochs 6 and 10 already changed the training trajectory. Strict no-op after fallback is not equivalent to clean fallback unless weights are rewound or risky updates are prevented before model updates.
- OK3 remained inactive and OK3 ROI applied stayed 0 across Gated.
- Gated outcome is `2/3` pass, not `3/3`. Do not recommend CATF-v2-Gated as the paper main method in its current form.
- Gated reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/reports/multiseed_catf_v2_gated_summary.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/reports/multiseed_catf_v2_gated_summary.json`
- Next technical direction, if continuing: either gate before applying epoch5 augmentation, add an auditable in-run rollback checkpoint, or design a seed-agnostic high-baseline risk gate that protects seed2 before ROI/industrial augmentation affects weights.

- Full multiseed CATF-v2-Safe validation completed at `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe/`.
- Safe seed0/1 were newly run for 50 epochs; Safe seed2 reuses the completed `outputs/experiments/catf_v2_safe_seed2_50ep/` run through `seed_2/catf_v2_safe/` summary/link artifacts.
- CATF-v2-Safe metrics:
  - seed0: Precision=0.7846, Recall=0.6765, mAP50=0.7347, mAP50-95=0.4759, `constraint_failed=false`.
  - seed1: Precision=0.7725, Recall=0.6477, mAP50=0.7542, mAP50-95=0.4799, `constraint_failed=false`.
  - seed2: Precision=0.6962, Recall=0.7286, mAP50=0.7692, mAP50-95=0.5224, `constraint_failed=false`.
- Safe vs clean deltas are all `0.0000`, so Safe achieves `3/3` constraint pass by reproducing clean native YOLO default on every seed.
- Fixed CATF-v2 was `1/3` failed; Safe is now `0/3` failed.
- Safe triggered `early_abstention_no_recall_or_map_gain` no-op fallback at epoch 5 on seed0, seed1, and seed2.
- Industrial samples augmented=0, ROI applied=0, and router random draws=0 for all three Safe seeds.
- Active class proposals before fallback: seed0 class 11/class 4, seed1 class 11/class 4, seed2 class 9.
- OK3 remained inactive and OK3 ROI applied remained 0.
- Critical interpretation: Safe protects seed2 and passes all constraints, but it is over-conservative. It does not preserve fixed CATF-v2's seed0 mAP gains or seed1 all-metric gains. Treat CATF-v2-Safe as a conservative safety/protection variant or fallback layer, not as the sole paper main augmentation method.
- New reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe/reports/multiseed_catf_v2_safe_summary.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe/reports/multiseed_catf_v2_safe_summary.json`
- Verification completed for this run: requested `py_compile` checks passed and the requested CATF-v2/feedback/online augmentation pytest subset passed with `89 passed`.

- CATF-v2-Safe has been implemented and seed2 has been validated.
- Safe mode adds high-recall baseline protection, negative-effect attribution, early abstention, safe accept guards, and strict no-op fallback.
- The seed2 design report is:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed2_safe_controller_design.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed2_safe_controller_design.json`
- Existing fixed CATF-v2 seed2 failure was localized:
  - Recall lag starts at epoch 6.
  - mAP50-95 lag starts at epoch 8.
  - A mAP50-95 guard is visible by feedback epoch 10.
  - Epoch 5 proposed class 9 without Recall/mAP gain, so Safe should abstain before industrial augmentation affects training.
- CATF-v2-Safe 10ep smoke completed at `outputs/experiments/catf_v2_safe_10ep_smoke/`; bbox/class legal, train images=2301, no fixed augmented dataset. It validated the wiring but did not enter no-op freeze because the 10ep training schedule is not directly comparable to the 50ep clean reference curve.
- CATF-v2-Safe seed2 50ep completed at `outputs/experiments/catf_v2_safe_seed2_50ep/`.
- Seed2 Safe final metrics: Precision=0.6962, Recall=0.7286, mAP50=0.7692, mAP50-95=0.5224.
- Delta vs clean seed2: all metrics `0.0000`; `constraint_failed=false`.
- Delta vs fixed CATF-v2 seed2: Precision -0.0674, Recall +0.0423, mAP50 +0.0110, mAP50-95 +0.0257.
- Safe triggered no-op freeze at epoch 5 with reason `early_abstention_no_recall_or_map_gain`.
- Industrial samples augmented=0, ROI applied=0, router random draws=0 after Safe fallback.
- Required reports:
  - `outputs/experiments/catf_v2_safe_seed2_50ep/reports/final_report.md`
  - `outputs/experiments/catf_v2_safe_seed2_50ep/reports/final_metrics.json`
  - `outputs/experiments/catf_v2_safe_seed2_50ep/reports/safe_controller_events.json`
  - `outputs/experiments/catf_v2_safe_seed2_50ep/reports/compare_with_clean_and_fixed_catf_v2.md`
- Recommended next step: run full multiseed CATF-v2-Safe validation. Do not continue optimizing unified RC thresholds.
- Official-path threshold re-optimization is complete and must be treated as the current threshold-calibration result.
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/official_threshold_reoptimization.md`.
- JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/official_threshold_reoptimization.json`.
- Threshold artifacts:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/unified_precision_guard_thresholds.json`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/per_seed_thresholds.json`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/conservative_default_thresholds.json`
- Pass counts on the official predict/val path:
  - fixed CATF-v2 without RC: `2/3`.
  - old saved unified RC: `1/3`.
  - reoptimized unified precision-guard RC: `2/3`.
  - per-seed RC: `3/3`.
  - conservative default RC: `2/3`.
- A single unified CATF-v2-RC threshold table does not achieve 3/3 under official-path constraints. The best unified table still leaves seed0 slightly outside the mAP50-95 guard.
- Per-seed threshold calibration can achieve 3/3, but should be positioned as deployment/model-specific calibration rather than the core training method.
- Do not cite the old cached post-hoc RC 3/3 result as a paper claim. The official path supersedes it.
- CATF-v2 no-op trainer parity is clean, and diagnosis-only parity is clean.
- The formal CATF-v2 transform no-augmentation path has been fixed to be a strict no-op.
- `scripts/audit_catf_v2_transform_parity.py` compares clean YOLO default, CATF-v2 noop, and CATF-v2 formal-force-skip on 100 deterministic train samples.
- Audit output:
  - `outputs/audits/catf_v2_transform_parity/transform_parity_report.md`
  - `outputs/audits/catf_v2_transform_parity/transform_parity.json`
  - `outputs/audits/catf_v2_transform_parity/diff_samples/`
- Result:
  - clean vs CATF-v2 noop final output is exact.
  - clean vs CATF-v2 formal-force-skip final output is exact.
  - bbox hash mismatches: 0/100.
  - formal-force-skip image/cls/Instances rewrites: 0/100.
  - router random draw count: 0.
  - applied industrial/ROI ops: 0.
- Force-skip/no-active/no_aug/stable/high-FP paths now return the native YOLO label object unchanged before bbox validation, clipping, or `Instances` rebuild.
- The critical fixed seed1 50-epoch rerun completed at `outputs/experiments/catf_v2_fixed_seed1_50ep/`.
- Fixed seed1 metrics: Precision=0.7852, Recall=0.7005, mAP50=0.7826, mAP50-95=0.5189.
- Relative to clean native seed1, fixed CATF-v2 improved Precision by +0.0127, Recall by +0.0528, mAP50 by +0.0284, and mAP50-95 by +0.0390; `constraint_failed=false`.
- Unlike the old seed1 CATF-v2 run, this fixed rerun applied actual augmentation: industrial samples augmented=55 and ROI applied=56.
- ROI affected class 11 for 51 applications and class 4 for 5 applications; OK3 remained inactive with OK3 ROI applied=0.
- The fixed seed1 result is no longer attributable to no-op transform perturbation; it should be treated as a valid CATF-v2 seed1 rerun after strict bypass repair.
- Fixed multiseed validation is now available at `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/`.
- Fixed CATF-v2 seed0/1/2 results:
  - seed0: P=0.7785, R=0.6697, mAP50=0.7437, mAP50-95=0.4895, constraint_failed=false.
  - seed1: P=0.7852, R=0.7005, mAP50=0.7826, mAP50-95=0.5189, constraint_failed=false.
  - seed2: P=0.7637, R=0.6863, mAP50=0.7582, mAP50-95=0.4967, constraint_failed=true.
- Strict bypass repair improved multiseed constraint stability from old CATF-v2 `1/3` pass to fixed CATF-v2 `2/3` pass.
- OK3 remained clean across fixed multiseed: never active, ROI applied=0.
- Recommendation: CATF-v2 is now a stronger candidate but not yet a fully stable sole main method because seed2 still fails mAP constraints.

## Output Layout Status

- Legacy root-level outputs were archived to `outputs/archive/old_outputs_20260517/`.
- Legacy Ultralytics auto outputs from `runs/detect/*` were archived to `outputs/archive/old_runs_20260517/runs_detect/`.
- `runs/` is no longer a formal result location. Formal YOLO commands must set `project=outputs/experiments/<run_id>`.
- Active generated datasets live under `outputs/datasets/`.
- Active experiment records live under `outputs/experiments/`.
- Active audits live under `outputs/audits/`.
- Snapshots live under `outputs/snapshots/`, with `outputs/project_snapshot_latest.md` retained for compatibility.
- Large local artifacts such as weights, generated images, dataset image/label files, and archive contents are intentionally ignored by Git. They remain available on disk.

## Active Paths

- Output convention: `docs/output_convention.md`
- Artifact inventory: `outputs/audits/artifact_inventory/artifact_inventory.md`
- Cleanup summary: `outputs/audits/artifact_inventory/cleanup_summary.md`
- Full tiled dataset mapping audit: `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.md`
- Full tiled dataset mapping JSON: `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.json`
- Tiling quality audit: `outputs/audits/tiling_quality/tiling_quality_audit.md`
- Tiling quality audit JSON: `outputs/audits/tiling_quality/tiling_quality_audit.json`
- Tiling quality debug images: `outputs/audits/tiling_quality/debug_truncated_bboxes/`
- Legacy smoke dataset mapping audit: `outputs/audits/dataset_mapping/dataset_mapping_audit.md`
- GPU preflight report: `outputs/audits/gpu_preflight/gpu_preflight_report.md`
- GPU preflight JSON: `outputs/audits/gpu_preflight/gpu_preflight_report.json`
- Tiled smoke dataset: `outputs/datasets/tiled/tiled_1024_ov20_smoke/`
- Tiled smoke data YAML: `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml`
- Unsafe full tiled dataset: `outputs/datasets/tiled/tiled_1024_ov20_full/`
- Full tiled data YAML: `outputs/datasets/tiled/tiled_1024_ov20_full/data.yaml`
- Full tiled dataset report: `outputs/datasets/tiled/tiled_1024_ov20_full/tiled_dataset_report.md`
- Full tiled dataset summary: `outputs/datasets/tiled/tiled_1024_ov20_full/dataset_summary.md`
- Full tiled debug visualizations: `outputs/datasets/tiled/tiled_1024_ov20_full/debug_tiling/`
- Safe full tiled dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/`
- Safe full tiled data YAML: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/data.yaml`
- Safe full tiled dataset report: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/tiled_dataset_report.md`
- Safe full tiled dataset summary: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/dataset_summary.md`
- Safe full tiled debug visualizations: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/debug_tiling/`
- Filtered dataset without `OK` and `瀹氫綅`: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/`
- Filtered data YAML: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Filtered class filter report: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/class_filter_report.md`
- Filtered dataset summary: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/dataset_summary.md`
- Filtered debug samples: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/debug_samples/`
- Current tiled baseline run: `outputs/experiments/20260517_tiled_baseline_20epoch/`
- Tiled baseline summary: `outputs/experiments/20260517_tiled_baseline_20epoch/reports/summary.md`
- Tiled baseline report: `outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_report.md`
- Tiled baseline metrics: `outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_metrics.json`
- Tiled baseline best weights: `outputs/experiments/20260517_tiled_baseline_20epoch/train/weights/best.pt`
- Tiled baseline last weights: `outputs/experiments/20260517_tiled_baseline_20epoch/train/weights/last.pt`
- Project snapshot: `outputs/snapshots/project_snapshot_latest.md`
- Compatibility snapshot: `outputs/project_snapshot_latest.md`

## Important Capabilities

- `scripts/build_yolo_tiled_dataset.py` builds tiled YOLO datasets with explicit output directories. Use `outputs/datasets/tiled/<dataset_id>/`.
- `scripts/build_yolo_tiled_dataset.py` writes tiled `data.yaml` via `yaml.safe_dump(..., allow_unicode=True)` and must preserve original `names`.
- `scripts/build_yolo_tiled_dataset.py` now defaults to safe bbox filtering: `min_visibility=0.7`, `large_object_min_visibility=0.9`, `drop_border_truncated=True`, `border_margin=2`, and `require_box_center_inside=True`.
- `scripts/audit_tiling_quality.py` audits retained tiled bbox visibility and tile-boundary truncation for `tiled_1024_ov20_full`.
- `scripts/filter_tiled_dataset.py` removes only `OK` and `瀹氫綅`, keeps `OK2` and `OK3`, remaps ids to `0..12`, and writes the filtered baseline dataset.
- `scripts/audit_dataset_mapping.py` audits original versus the full tiled dataset class names, label class ids, bbox distribution, full-source coverage, and formal baseline readiness.
- `scripts/audit_artifacts.py` scans `outputs/` and `runs/` and writes inventory reports under `outputs/audits/artifact_inventory/`.
- `scripts/run_gpu_preflight.py` writes GPU preflight reports under `outputs/audits/gpu_preflight/`.
- `scripts/run_diagnostic_augmentation_pipeline.py` supports `--run-id`; when `--output-dir` is omitted, it writes to `outputs/experiments/<run_id>/`.
- `diagnosis.json` includes `diagnosis_vector` scores and per-score evidence.
- `policy_mapping.py` uses severity-score dynamic formulas for operator probability and strength.
- Proxy ranking combines `proxy_score` and `SafetyScore`; bbox rates below soft targets are penalties instead of automatic rejection.
- `copy_paste` policies produce filter audits and debug visualizations.
- `metric_consistency_audit.md` documents differences between YOLO val metrics and diagnosis TP/FP/FN.

## GPU Environment

Formal training must use conda env `pytorch`, not `base`.

- Conda env: `pytorch`
- Python executable: `D:\Anaconda\envs\pytorch\python.exe`
- Python version: 3.9.19
- PyTorch: 2.4.1
- `torch.cuda.is_available()`: True
- `torch.version.cuda`: 12.4
- CUDA device count: 1
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Ultralytics: 8.3.221
- `yolo checks`: passed
- YOLO GPU smoke: passed with `model=yolo11n.pt`, tiled smoke `data.yaml`, `epochs=1`, `imgsz=640`, `batch=1`, `workers=0`, `device=0`, and YOLO built-in augmentations disabled.

The earlier base-env preflight resolved to CPU-only PyTorch. CPU is only for smoke/debug and must not be treated as formal experiment output.

## Current Baseline Result

The formal safe tiled 50 epoch baseline has completed on GPU.

- Run ID: `20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep`
- Result path: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Safe tiled dataset: true
- Removed classes: `OK`, `瀹氫綅`
- Retained classes: `OK2`, `OK3`, `鍔犲己绛嬫墦浼, `寮€瑁俙, `娌规薄`, `娴呭垝浼, `婕忚儗閿, `纰颁激`, `鑴忔薄`, `杞粨鍒掍激`, `閿′笣娈嬬暀`, `閿″皷`, `閿¤啅`
- Train/val tiles: 2301 / 677
- BBoxes: 4087
- Model: `yolo11n.pt`
- Epochs: 50
- imgsz: 1024
- batch: 2
- workers: 0
- device: 0
- OOM: false
- Precision: 0.690
- Recall: 0.615
- mAP50: 0.669
- mAP50-95: 0.434
- Lowest per-class Recall: `寮€瑁俙 (0.000)
- YOLO built-in augmentation switches disabled: `mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0`
- best.pt: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/best.pt`
- last.pt: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/last.pt`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_report.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_metrics.json`

The earlier `20260517_tiled_baseline_20epoch` run used the smoke tiled dataset and should remain a smoke reference, not the formal baseline.

## Tiled Dataset Status

No training was run after building or auditing these datasets.

- Source dataset: `E:\TJGY\DataSet2_fixed`
- Unsafe old full dataset: `outputs/datasets/tiled/tiled_1024_ov20_full/`
  - Built with `min_visibility=0.3`.
  - Tiling quality audit found 3197 / 7465 border-truncated bboxes (42.83%).
  - It must not be used as the formal baseline dataset.
- Safe full dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/`
  - Build parameters: `tile_size=1024`, `overlap=0.2`, `min_visibility=0.7`, `large_object_min_visibility=0.9`, `drop_border_truncated=True`, `border_margin=2`, `require_box_center_inside=True`, `keep_empty_ratio=0.1`, `seed=42`
  - Source coverage: 461 train images and 116 val images; no source-image cap was used.
  - Tiled images: 2452 train, 677 val
  - Original bboxes: 3084
  - Safe tiled bboxes: 4269
  - Dropped bbox candidates after tile intersection: 13826
  - Visibility-failed dropped candidates: 13346
  - Border-truncated dropped candidates: 3961
  - Obvious half-target bbox remains: false
  - Debug tile bbox visualizations: 100
  - Class ids remain in range and Chinese names remain intact.
  - Caveat: class `瀹氫綅` has 0 retained bboxes under the strict large-structure rule.
- Filtered baseline dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/`
  - Deletes only `OK` and `瀹氫綅`.
  - Keeps `OK2` and `OK3`.
  - Remaps ids to `0..12`.
  - Source train/val images: 2452 / 677.
  - Filtered train/val images: 2301 / 677.
  - Source bbox count: 4269.
  - Filtered bbox count: 4087.
  - Train empty tiles retained: 210.
  - Val empty tiles retained: 83.
  - Debug samples: 50.
  - Class id out of range: false.
  - Chinese class names damaged: false.
  - This is the dataset to use for formal baseline training.

## Verification Commands

- `python scripts\audit_artifacts.py`
- `python scripts\audit_dataset_mapping.py`
- `pytest -q tests/test_build_yolo_tiled_dataset.py tests/test_copy_paste.py tests/test_proxy_prefilter.py tests/test_yolo_error_analysis.py`

## Next Steps

- Use the completed formal baseline report at `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_report.md` as the reference for the next diagnostic augmentation experiment.
- Use explicit `--run-id` and `project=outputs/experiments/<run_id>` for every formal experiment.
- Keep Windows YOLO commands at `workers=0`.

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
- Output: `outputs/experiments/catf_v2_gated/`
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
- Constraint failed: `True`
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/seed_2/catf_v2_gated/reports/final_report.md`
- Policy history: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/seed_2/catf_v2_gated/reports/policy_history.json`
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
- Active classes from smoke: class `1` OK3, class `6` 婕忚儗閿? class `8` 鑴忔薄.
- OK3 activation is judged not reasonable for a formal run: Recall is already high (`0.9891`), FN count is only `2`, FP count is `49`, and OK-like classes should default to stable/no_aug unless evidence is very strong.
- 婕忚儗閿?activation is judged reasonable: low Recall (`0.2826`), many FN (`32`), low-contrast evidence, and conservative ROI `sharpen_mild`/`local_contrast` ops.
- 鑴忔薄 activation is partially reasonable as low Recall, but should be guarded by stain/dirty high-FP domain priors and should not lower threshold or escalate photometric before FP behavior is known.
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
- Active classes after fix: class `6` 婕忚儗閿?(texture_boundary_weak), class `8` 鑴忔薄 (low_recall).
- OK3 active=`false`; OK3 ROI applied=`0`.
- 婕忚儗閿?active=`true` with conservative ROI sharpen/local_contrast.
- 鑴忔薄 domain_high_fp_prior=`true`; photometric probs `{'clahe': 0.0, 'gamma': 0.0, 'brightness': 0.0, 'contrast': 0.0}`; threshold recommendation `0.25` with reason `domain_high_fp_prior_keep_threshold`.
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
- Active class counts: `{'6:婕忚儗閿?: 1, '12:閿¤啅': 1}`.
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
- Active class counts: `{'8:鑴忔薄': 1, '11:閿″皷': 1}`; ROI affected totals: `{'8:鑴忔薄': 3, '11:閿″皷': 20}`.
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
- Next recommended experiment: diagnosis-only 50ep control, then compare clean native vs diagnosis-only vs CATF-v2 before further controller changes.
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

- Scope: analysis only; no training was run and CATF-v2 rules were not changed.
- Source experiment: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/`.
- Analysis script: `scripts/analyze_fixed_catf_v2_seed2_and_thresholds.py`.
- Reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed2_failure_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_threshold_calibration_posthoc.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_vs_old_catf_v2_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_catf_v2_next_step_summary.md`
- Fixed seed2 failure: clean seed2 was already strong; fixed CATF-v2 increased Precision but reduced Recall and localization quality (`+0.0674/-0.0423/-0.0110/-0.0257` for P/R/mAP50/mAP50-95).
- Recall loss is concentrated in `浅划伤`, `轮廓划伤`, `加强筋打伤`, `漏背锡`, and `锡膏`; AP drops also involve `锡丝残留`, `脏污`, `开裂`, and `加强筋打伤`.
- Seed2 active/ROI classes were `轮廓划伤`, `锡尖`, and `脏污`; this only partially overlaps the final degraded classes.
- Post-hoc per-class threshold calibration repairs seed2 in the analysis evaluator, but total calibrated pass count is still `2/3` because seed0 remains below the mAP50-95 constraint.
- Handoff recommendation: do not call fixed CATF-v2 a fully stable main method yet. Next code work should target class-level rollback, negative-effect attribution, high-recall baseline protection, and better calibration constraints.
<!-- FIXED_CATF_V2_SEED2_THRESHOLD_ANALYSIS_END -->

<!-- CATF_V2_RC_SEED0_CALIBRATION_ROLLBACK_START -->
## CATF-v2-RC Seed0 Calibration and Rollback Analysis

- Scope: analysis only; no training was run and CATF-v2 augmentation rules were not rewritten.
- New helper module: `AutoAugment/catf_v2/per_class_thresholds.py`.
- Analysis script: `scripts/refine_fixed_catf_v2_seed0_calibration_and_rollback.py`.
- Reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed0_failure_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed0_threshold_calibration_posthoc.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_catf_v2_threshold_calibration_all_seeds.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_catf_v2_class_level_rollback_simulation.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/catf_v2_rc_final_candidate_plan.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_catf_v2_ready_for_paper_summary.md`
- Important correction: seed0 is not an official fixed-training failure. Official fixed seed0 passes constraints; the remaining seed0 failure was in the post-hoc threshold evaluator.
- Seed0 was repaired by the RC threshold objective in the post-hoc evaluator; RC calibration reaches `3/3` seeds passing constraints.
- Recommended threshold table: `{OK2:0.25, OK3:0.10, 加强筋打伤:0.25, 开裂:0.25, 油污:0.10, 浅划伤:0.10, 漏背锡:0.10, 碰伤:0.10, 脏污:0.10, 轮廓划伤:0.10, 锡丝残留:0.25, 锡尖:0.25, 锡膏:0.10}`.
- Rollback simulation: keep fixed CATF-v2 for `油污`, `锡尖`; threshold-only for `脏污`; rollback candidates `加强筋打伤`, `锡丝残留`; high-risk negative-effect classes `轮廓划伤`, `锡膏`.
- Recommended method name: `CATF-v2-RC = fixed CATF-v2 + per-class constrained threshold calibration + class-level rollback/high-recall protection`.
- Next safest verification: confirm the RC threshold table in the official validation/export path before making final paper claims; do not run broad new multiseed training unless official threshold validation exposes a mismatch.
<!-- CATF_V2_RC_SEED0_CALIBRATION_ROLLBACK_END -->

<!-- CATF_V2_RC_OFFICIAL_PATH_VALIDATION_START -->
## CATF-v2-RC Official Val/Predict Path Validation

- Scope: validation/inference only; no training was run.
- Script: `scripts/validate_catf_v2_rc_official_path.py`.
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/catf_v2_rc_official_path_validation.md`.
- JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/catf_v2_rc_official_path_validation.json`.
- Validated config: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/catf_v2_rc_per_class_thresholds.json`.
- Method: reran fixed seed0/1/2 `best.pt` with Ultralytics `YOLO.val`; then ran `YOLO.predict(conf=0.10)` and applied the saved per-class threshold table in post-processing.
- Result: saved unified CATF-v2-RC threshold table passes only `1/3` seeds in the official predict post-processing path.
- Passing seed: seed1.
- Failing seeds: seed0 and seed2, both due Precision dropping beyond tolerance after aggressive threshold lowering.
- Key deltas under official predict + RC post-processing:
  - seed0 `-0.0666/+0.0321/+0.0054/-0.0095`, fails Precision.
  - seed1 `+0.0496/+0.0501/+0.0395/+0.0289`, passes.
  - seed2 `-0.0148/+0.0655/+0.0435/+0.0096`, fails Precision.
- Handoff instruction: treat the earlier `fixed_catf_v2_ready_for_paper_summary.md` as superseded by this official path validation for claims about the saved unified threshold table. Next work should re-optimize thresholds with a stronger Precision guard or use per-seed/model-specific threshold configs; do not run training unless explicitly requested.
<!-- CATF_V2_RC_OFFICIAL_PATH_VALIDATION_END -->
