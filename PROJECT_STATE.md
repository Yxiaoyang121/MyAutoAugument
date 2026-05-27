# Project State

Last updated: 2026-05-18

## Current Position

The repository is centered on diagnosis-driven augmentation for industrial defect detection. The active path still keeps YOLO network architecture unchanged and focuses on dataset construction, validation-error diagnosis, policy generation, proxy safety, short training, and auditable reporting.

## Formal Safe Tiled Baseline Result (2026-05-18)

- Run ID: `20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Safe tiled dataset: true; safe source config used `tile_size=1024`, `overlap=0.2`, `min_visibility=0.7`, `large_object_min_visibility=0.9`, `drop_border_truncated=True`, `border_margin=2.0`, `require_box_center_inside=True`, `keep_empty_ratio=0.1`.
- Removed classes: `OK`, `定位`; retained classes: `OK2`, `OK3`, `加强筋打伤`, `开裂`, `油污`, `浅划伤`, `漏背锡`, `碰伤`, `脏污`, `轮廓划伤`, `锡丝残留`, `锡尖`, `锡膏`.
- Train/val tiles: 2301 / 677; bboxes: 4087.
- Model: `yolo11n.pt`; epochs: 50; imgsz: 1024; batch: 2; device: 0; workers: 0.
- YOLO augmentation switches requested for shutdown were all set to `0`: `mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0`.
- Completed 50 epochs: true; OOM: false; training time: 2.826 hours.
- Validation metrics from `best.pt`: Precision=0.690, Recall=0.615, mAP50=0.669, mAP50-95=0.434.
- Lowest per-class Recall: `开裂` (0.000).
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_report.md`.
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_metrics.json`.
- best.pt: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/best.pt`.
- last.pt: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/last.pt`.

## Implemented In This Update

- Added `scripts/filter_tiled_dataset.py` and built the filtered dataset at `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/`.
- The filtered dataset removes only `OK` and `定位`, keeps `OK2` and `OK3`, and remaps class ids to `0..12` with Chinese names preserved.
- The filtered dataset reports:
  - source train/val images: 2452 / 677
  - filtered train/val images: 2301 / 677
  - source bbox count: 4269
  - filtered bbox count: 4087
  - class id out of range: false
  - Chinese class names damaged: false
  - formal baseline ready: true
- Added `scripts/audit_tiling_quality.py` and audited `outputs/datasets/tiled/tiled_1024_ov20_full/` for tile-boundary truncation risk.
- Marked `outputs/datasets/tiled/tiled_1024_ov20_full/` as unsafe for formal baseline use because it retains partial-object bboxes.
- Hardened `scripts/build_yolo_tiled_dataset.py` with safe tiling controls:
  - `--min-visibility` default changed to `0.7`
  - `--large-object-min-visibility` added with default `0.9`
  - `--drop-border-truncated` added with default `True`
  - `--border-margin` added with default `2`
  - `--require-box-center-inside` added with default `True`
  - `--classwise-visibility-config` added for optional per-class overrides
- Built the new safe full tiled dataset at `outputs/datasets/tiled/tiled_1024_ov20_full_safe/`.
- Safe dataset debug visualizations now include retained boxes and dropped border-truncated/visibility-risk boxes; 100 images were written under `outputs/datasets/tiled/tiled_1024_ov20_full_safe/debug_tiling/`.
- Built the full tiled dataset from `E:\TJGY\DataSet2_fixed` at `outputs/datasets/tiled/tiled_1024_ov20_full/` without `--max-images-per-split`.
- Expanded `scripts/build_yolo_tiled_dataset.py` reporting for full baseline readiness:
  - original and tiled train/val image counts
  - original and tiled bbox totals
  - per-class original/tiled train/val instance counts
  - bbox drop reasons
  - empty tile retention
  - `data.yaml` nc/names, class id range, class id out-of-range flags, and Chinese-name damage flags
- Changed full dataset debug output to 30 random tile-level bbox visualizations under `outputs/datasets/tiled/tiled_1024_ov20_full/debug_tiling/`.
- Updated `scripts/audit_dataset_mapping.py` to audit the full tiled dataset and write:
  - `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.md`
  - `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.json`
- Added `scripts/audit_dataset_mapping.py` and generated dataset mapping audit under `outputs/audits/dataset_mapping/`.
- Repaired `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml` so it fully inherits original class names from `E:\TJGY\DataSet2_fixed\data.yaml`.
- Updated `scripts/build_yolo_tiled_dataset.py` to write tiled `data.yaml` with `yaml.safe_dump(..., allow_unicode=True)` so non-ASCII class names are preserved.
- Added `scripts/audit_artifacts.py` and generated artifact inventory under `outputs/audits/artifact_inventory/`.
- Added `docs/output_convention.md` and normalized active artifacts into `outputs/datasets/`, `outputs/experiments/`, `outputs/audits/`, and `outputs/snapshots/`.
- Archived legacy root-level outputs to `outputs/archive/old_outputs_20260517/`.
- Archived legacy Ultralytics auto outputs from `runs/detect/*` to `outputs/archive/old_runs_20260517/runs_detect/`.
- Updated `scripts/run_gpu_preflight.py` to capture the active conda env, `sys.executable`, PyTorch/CUDA, Ultralytics, `yolo checks`, optional `nvidia-smi`, and conditional YOLO GPU smoke results without falling back to base PATH.
- Generated auditable GPU preflight reports:
  - `outputs/audits/gpu_preflight/gpu_preflight_report.md`
  - `outputs/audits/gpu_preflight/gpu_preflight_report.json`
- Added `scripts/build_yolo_tiled_dataset.py` for YOLO sliding-window tiled dataset construction.
- Added `diagnosis_vector` to `diagnosis.json` with normalized small-object, low-contrast, class-imbalance, localization, and false-positive scores plus calculation basis.
- Reworked `AutoAugment/diagnostic_pipeline/policy_mapping.py` from fixed issue rules to severity-score dynamic probability/strength formulas.
- Split proxy safety into explicit `original_bbox_retention`, `new_bbox_valid_rate`, `total_bbox_valid_rate`, `small_object_retention`, `SafetyScore`, and soft safety penalties.
- Added copy-paste filter audit artifacts and debug images under the proxy stage.
- Added `AutoAugment/diagnostic_pipeline/strategy_memory.py` with JSONL append and cosine-similarity reranking.
- Added metric consistency audit output under `metric_consistency/`.

## Verified Locally

- Formal safe tiled YOLO11n baseline training and validation completed for `20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep`.
- Filtered no-OK/no-position dataset build completed with no training:
  - Output dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/`
  - Deleted classes: `OK`, `定位`
  - Kept classes: `OK2`, `OK3`, `加强筋打伤`, `开裂`, `油污`, `浅划伤`, `漏背锡`, `碰伤`, `脏污`, `轮廓划伤`, `锡丝残留`, `锡尖`, `锡膏`
  - New class id mapping: `0..12` in the order above
  - Source train/val images: 2452 / 677
  - Filtered train/val images: 2301 / 677
  - Source bbox count: 4269
  - Filtered bbox count: 4087
  - Train bbox count before/after: 3337 / 3182
  - Val bbox count before/after: 932 / 905
  - Train empty tiles retained: 210
  - Val empty tiles retained: 83
  - Debug samples: 50
  - Class id out of range: false
  - Chinese class names damaged: false
  - Formal baseline readiness: true
- Tiling quality audit for the old full tiled dataset:
  - Report: `outputs/audits/tiling_quality/tiling_quality_audit.md`
  - JSON: `outputs/audits/tiling_quality/tiling_quality_audit.json`
  - Debug truncated bbox images: `outputs/audits/tiling_quality/debug_truncated_bboxes/` (50 images)
  - Total old full tiled bboxes: 7465
  - Border-truncated bboxes: 3197 (42.83%)
  - Visibility < 0.5: 1091
  - Visibility < 0.7: 1956
  - Visibility < 0.8: 2405
  - Visibility < 0.9: 2874
  - Bboxes touching tile boundary: 3249 (43.52%)
- Safe full tiled dataset build completed with no training:
  - Output dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/`
  - Parameters: `tile_size=1024`, `overlap=0.2`, `min_visibility=0.7`, `large_object_min_visibility=0.9`, `drop_border_truncated=True`, `border_margin=2`, `require_box_center_inside=True`, `keep_empty_ratio=0.1`, `seed=42`
  - Tiled images: 2452 train, 677 val
  - Original bboxes: 3084
  - Safe tiled bboxes: 4269
  - Dropped bbox candidates after tile intersection: 13826
  - Visibility-failed dropped candidates: 13346
  - Border-truncated dropped candidates: 3961
  - Center-outside dropped candidates: 9855
  - Obvious half-target bbox remains: false
  - Debug tile visualizations: 100
  - Class ids remain in range and Chinese names remain intact.
  - Note: class `定位` has 0 retained bboxes under the strict large-structure rule; use this as an explicit audit caveat if that class must be evaluated.
- Full tiled dataset build completed with no training:
  - Command used no `--max-images-per-split`.
  - Source dataset: `E:\TJGY\DataSet2_fixed`
  - Output dataset: `outputs/datasets/tiled/tiled_1024_ov20_full/`
  - Parameters: `tile_size=1024`, `overlap=0.2`, `min_visibility=0.3`, `keep_empty_ratio=0.1`, `seed=42`
  - Original images: 461 train, 116 val
  - Tiled images: 4155 train, 1098 val
  - Original bboxes: 3084
  - Tiled bboxes: 7465
  - Empty tiles retained: 478
  - Dropped bboxes in retained tiles: 22202 (`below_min_visibility=6212`, `outside_tile=15990`)
  - Debug tile bbox visualizations: 30
  - `data.yaml`: `nc=15`, names inherited from original with Chinese names intact
  - Class ids: tiled min 0, max 14, no class id >= nc
  - Superseded status: this dataset is now marked unsafe for formal baseline use after the tiling quality audit found retained partial-object bboxes.
- Full tiled dataset mapping audit:
  - Report: `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.md`
  - JSON: `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.json`
  - Full source coverage confirmed: 461 train and 116 val source images.
  - Tiled names match original names exactly.
  - No class id >= nc and no negative class id found.
  - Chinese class names are not damaged.
- `python -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['scripts/build_yolo_tiled_dataset.py','scripts/audit_dataset_mapping.py']]; print('syntax ok')"` passed.
- `pytest -q tests\test_build_yolo_tiled_dataset.py` passed: 2 tests.
- GPU environment confirmed in conda env `pytorch`:
  - Python executable: `D:\Anaconda\envs\pytorch\python.exe`
  - Python version: 3.9.19
  - PyTorch: 2.4.1
  - `torch.cuda.is_available()`: True
  - `torch.version.cuda`: 12.4
  - CUDA device count: 1
  - GPU: NVIDIA GeForce RTX 3060 Laptop GPU
  - Ultralytics: 8.3.221
  - `yolo checks`: passed
  - YOLO GPU smoke: passed with `model=yolo11n.pt`, `data=outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml`, `epochs=1`, `imgsz=640`, `batch=1`, `workers=0`, `device=0`, and YOLO built-in augmentations disabled
  - Formal training environment: conda env `pytorch`, YOLO `device=0`
- Latest GPU preflight reports are `outputs/audits/gpu_preflight/gpu_preflight_report.md` and `outputs/audits/gpu_preflight/gpu_preflight_report.json`.
- Earlier base-env preflight showed CPU-only PyTorch; base must not be used for formal training.
- Dataset mapping audit for the earlier smoke dataset:
  - Report: `outputs/audits/dataset_mapping/dataset_mapping_audit.md`
  - JSON: `outputs/audits/dataset_mapping/dataset_mapping_audit.json`
  - Original dataset: 461 train images, 116 val images, 2433 train bboxes, 651 val bboxes, class ids 0..14.
  - Tiled smoke dataset: 107 train tiles, 86 val tiles, 232 train bboxes, 143 val bboxes, class ids 0..14.
  - No class id >= nc found in original or tiled smoke labels.
  - Tiled smoke names now match original names exactly.
  - Tiled smoke is not a formal baseline dataset because it uses only 16 source images, matching the capped smoke build.
  - The 20 epoch `blank-or-unrendered` rows came from the earlier corrupted/non-renderable tiled class names in the saved val log/metrics, not from out-of-range class IDs.
  - Low mAP is mainly driven by the smoke split and class imbalance: OK and OK3 have high AP50, while 开裂, 漏背锡, 碰伤, 轮廓划伤, 锡丝残留, and 锡膏 have recall 0 in the smoke val metrics.
- Ran tiled baseline 20 epoch on GPU:
  - Command target: `outputs/experiments/20260517_tiled_baseline_20epoch`
  - Dataset: `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml`
  - Model: `yolo11n.pt`
  - Epochs: 20
  - imgsz: 1024
  - batch: 2
  - workers: 0
  - device: 0
  - YOLO built-in augmentations disabled: `mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0`
  - OOM: false
  - best.pt: `outputs/experiments/20260517_tiled_baseline_20epoch/train/weights/best.pt`
  - last.pt: `outputs/experiments/20260517_tiled_baseline_20epoch/train/weights/last.pt`
  - Precision: 0.828
  - Recall: 0.213
  - mAP50: 0.247
  - mAP50-95: 0.181
  - Report: `outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_report.md`
  - Metrics JSON: `outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_metrics.json`
- `pytest -q tests/test_build_yolo_tiled_dataset.py tests/test_copy_paste.py tests/test_proxy_prefilter.py tests/test_yolo_error_analysis.py` passed.
- `pytest -q` passed: 89 tests.
- Built tiled smoke dataset:
  - `outputs/datasets/tiled/tiled_1024_ov20_smoke`
  - `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml`
  - `outputs/datasets/tiled/tiled_1024_ov20_smoke/tiled_dataset_report.md`
  - `outputs/datasets/tiled/tiled_1024_ov20_smoke/debug_tiling/`
- Historical real tiled smoke pipeline was archived:
  - `outputs/archive/old_outputs_20260517/diagnostic_aug_tiled_smoke`
  - baseline epochs: 1
  - short epochs: 1
  - final epochs: 1
  - top-k: 1
  - workers: 0
  - device: cpu
  - final training skipped by request

## Smoke Conclusions

- `copy_paste` was not hard filtered. Three copy-paste policies passed the hard filter; ordinary bbox risks were recorded as soft safety penalties.
- Metric consistency audit found no precision/recall delta for this smoke because both YOLO val and diagnosis were zero-recall at the chosen thresholds, while also documenting why mAP and single-threshold TP/FP/FN are not expected to match in general.
- Strategy memory appended the smoke record to the archived `outputs/archive/old_outputs_20260517/strategy_memory.jsonl`; no prior similar cases existed for rerank boost on this run.

## Operational Notes

- Formal training must use conda env `pytorch` with `D:\Anaconda\envs\pytorch\python.exe`.
- Formal YOLO training must use GPU `device=0`.
- Formal runs must use explicit `--run-id` and `project=outputs/experiments/<run_id>`; do not rely on `runs/detect`.
- Unsafe full tiled dataset path: `outputs/datasets/tiled/tiled_1024_ov20_full/`; do not use it for formal baseline because partial-object bboxes were retained.
- Active safe full tiled dataset path: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/`.
- Active filtered formal baseline dataset path: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/`.
- Formal baseline training has completed on `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`; primary result is `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_report.md`.
- CPU is allowed only for smoke/debug runs and must not be treated as formal experiment output.
- Do not use the base conda environment for formal training; it previously resolved to CPU-only PyTorch.
- Windows YOLO commands should keep `workers=0`.
- This machine needed `KMP_DUPLICATE_LIB_OK=TRUE` for the CPU YOLO smoke because the active Anaconda environment initialized duplicate OpenMP runtimes.
- The smoke used a capped tiled dataset (`--max-images-per-split 8`) to keep runtime bounded; it is not a final benchmark.

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
- No-feedback control disables both feedback and industrial augmentation, using Ultralytics YOLO default augmentation as the behavior check.
- Output: `outputs/experiments/yolo_default_inloop_no_feedback_control_50ep/`
- Epochs: `50`
- Feedback enabled: `false`
- Industrial augmentation enabled: `false`
- Feedback epochs: `[]`
- Stage restart count: `0`
- Epoch continuous: `true`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- Control P/R/mAP50/mAP50-95: `0.7262/0.6844/0.7616/0.5250`
- Delta vs YOLO default reference: `+0.0130/-0.0756/-0.0142/+0.0010`
- Close to YOLO default reference: `false`
- Feedback 50ep status: `skipped`; reason: no-feedback control did not reproduce YOLO default closely enough.
- Report: `outputs/experiments/yolo_default_inloop_no_feedback_control_50ep/reports/inloop_no_feedback_control_report.md`
- Policy history: `outputs/experiments/yolo_default_inloop_no_feedback_control_50ep/reports/policy_history.json`
- Feedback skip report: `outputs/experiments/yolo_default_inloop_feedback_50ep_full/reports/final_report.md`
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
