# Project State

Last updated: 2026-05-18

## Current Position

The repository is centered on diagnosis-driven augmentation for industrial defect detection. The active path still keeps YOLO network architecture unchanged and focuses on dataset construction, validation-error diagnosis, policy generation, proxy safety, short training, and auditable reporting.

## Implemented In This Update

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
  - Full tiled dataset readiness: can be used as the formal baseline dataset.
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
- Active full tiled dataset path: `outputs/datasets/tiled/tiled_1024_ov20_full/`.
- Formal baseline training may now target `outputs/datasets/tiled/tiled_1024_ov20_full/data.yaml`; no full-dataset training has been run in this update.
- CPU is allowed only for smoke/debug runs and must not be treated as formal experiment output.
- Do not use the base conda environment for formal training; it previously resolved to CPU-only PyTorch.
- Windows YOLO commands should keep `workers=0`.
- This machine needed `KMP_DUPLICATE_LIB_OK=TRUE` for the CPU YOLO smoke because the active Anaconda environment initialized duplicate OpenMP runtimes.
- The smoke used a capped tiled dataset (`--max-images-per-split 8`) to keep runtime bounded; it is not a final benchmark.
