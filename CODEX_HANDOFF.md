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

The 20 epoch tiled baseline completed on GPU, but it is not a formal final result because it used the smoke tiled dataset.

- Run ID: `20260517_tiled_baseline_20epoch`
- Result path: `outputs/experiments/20260517_tiled_baseline_20epoch/`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml`
- Epochs: 20
- imgsz: 1024
- batch: 2
- workers: 0
- device: 0
- OOM: false
- Precision: 0.828
- Recall: 0.213
- mAP50: 0.247
- mAP50-95: 0.181
- YOLO built-in augmentations disabled: `mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0`
- Existing `blank-or-unrendered` class rows in the saved val report are a class-name rendering/parsing artifact from the earlier corrupted tiled `data.yaml`; current tiled class names match the original data.yaml.
- No class id >= nc was found in the original or tiled smoke labels.
- Main low-mAP drag in the smoke run: 开裂, 漏背锡, 碰伤, 轮廓划伤, 锡丝残留, and 锡膏 have recall 0; several of these have very few train/val samples or no train samples in the smoke subset.

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

## Verification Commands

- `python scripts\audit_artifacts.py`
- `python scripts\audit_dataset_mapping.py`
- `pytest -q tests/test_build_yolo_tiled_dataset.py tests/test_copy_paste.py tests/test_proxy_prefilter.py tests/test_yolo_error_analysis.py`

## Next Steps

- Run the formal baseline experiment against `outputs/datasets/tiled/tiled_1024_ov20_full_safe/data.yaml`.
- Use explicit `--run-id` and `project=outputs/experiments/<run_id>` for every formal experiment.
- Keep Windows YOLO commands at `workers=0`.
