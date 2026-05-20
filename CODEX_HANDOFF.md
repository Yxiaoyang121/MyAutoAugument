# Codex Handoff

## Repository

- Path: `E:\TJGY\MinPaper\MyAutoAugument`
- Branch: `codex/sync-latest`
- Remote: `https://github.com/Yxiaoyang121/MyAutoAugument.git`

## Current Scope

The project is a diagnosis-driven augmentation pipeline for industrial defect detection. Keep work centered on dataset construction, tiling, validation-error diagnosis, policy generation, proxy safety, short-training validation, and auditable artifacts. Do not reframe this as YOLO backbone, neck, or head redesign.

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
- Filtered dataset without `OK` and `定位`: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/`
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
- `scripts/filter_tiled_dataset.py` removes only `OK` and `定位`, keeps `OK2` and `OK3`, remaps ids to `0..12`, and writes the filtered baseline dataset.
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
- Removed classes: `OK`, `定位`
- Retained classes: `OK2`, `OK3`, `加强筋打伤`, `开裂`, `油污`, `浅划伤`, `漏背锡`, `碰伤`, `脏污`, `轮廓划伤`, `锡丝残留`, `锡尖`, `锡膏`
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
- Lowest per-class Recall: `开裂` (0.000)
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
  - Caveat: class `定位` has 0 retained bboxes under the strict large-structure rule.
- Filtered baseline dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/`
  - Deletes only `OK` and `定位`.
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
