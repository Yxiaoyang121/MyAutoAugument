# Project State

Last updated: 2026-05-17

## Current Position

The repository is centered on diagnosis-driven augmentation for industrial defect detection. The active path still keeps YOLO network architecture unchanged and focuses on dataset construction, validation-error diagnosis, policy generation, proxy safety, short training, and auditable reporting.

## Implemented In This Update

- Updated `scripts/run_gpu_preflight.py` to capture the active conda env, `sys.executable`, PyTorch/CUDA, Ultralytics, `yolo checks`, optional `nvidia-smi`, and conditional YOLO GPU smoke results without falling back to base PATH.
- Generated auditable GPU preflight reports:
  - `outputs/gpu_preflight_report.md`
  - `outputs/gpu_preflight_report.json`
- Added `scripts/build_yolo_tiled_dataset.py` for YOLO sliding-window tiled dataset construction.
- Added `diagnosis_vector` to `diagnosis.json` with normalized small-object, low-contrast, class-imbalance, localization, and false-positive scores plus calculation basis.
- Reworked `AutoAugment/diagnostic_pipeline/policy_mapping.py` from fixed issue rules to severity-score dynamic probability/strength formulas.
- Split proxy safety into explicit `original_bbox_retention`, `new_bbox_valid_rate`, `total_bbox_valid_rate`, `small_object_retention`, `SafetyScore`, and soft safety penalties.
- Added copy-paste filter audit artifacts and debug images under the proxy stage.
- Added `AutoAugment/diagnostic_pipeline/strategy_memory.py` with JSONL append and cosine-similarity reranking.
- Added metric consistency audit output under `metric_consistency/`.

## Verified Locally

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
  - YOLO GPU smoke: passed with `model=yolo11n.pt`, `data=outputs\tiled_dataset_smoke\data.yaml`, `epochs=1`, `imgsz=640`, `batch=1`, `workers=0`, `device=0`, and YOLO built-in augmentations disabled
  - Formal training environment: conda env `pytorch`, YOLO `device=0`
- Latest GPU preflight reports are `outputs/gpu_preflight_report.md` and `outputs/gpu_preflight_report.json`.
- Earlier base-env preflight showed CPU-only PyTorch; base must not be used for formal training.
- Ran tiled baseline 20 epoch on GPU:
  - Command target: `outputs/tiled_baseline_20epoch`
  - Dataset: `outputs/tiled_dataset_smoke/data.yaml`
  - Model: `yolo11n.pt`
  - Epochs: 20
  - imgsz: 1024
  - batch: 2
  - workers: 0
  - device: 0
  - YOLO built-in augmentations disabled: `mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0`
  - OOM: false
  - best.pt: `outputs/tiled_baseline_20epoch/train/weights/best.pt`
  - last.pt: `outputs/tiled_baseline_20epoch/train/weights/last.pt`
  - Precision: 0.828
  - Recall: 0.213
  - mAP50: 0.247
  - mAP50-95: 0.181
  - Report: `outputs/tiled_baseline_20epoch/baseline_20epoch_report.md`
  - Metrics JSON: `outputs/tiled_baseline_20epoch/baseline_20epoch_metrics.json`
- `pytest -q tests/test_build_yolo_tiled_dataset.py tests/test_copy_paste.py tests/test_proxy_prefilter.py tests/test_yolo_error_analysis.py` passed.
- `pytest -q` passed: 89 tests.
- Built tiled smoke dataset:
  - `outputs/tiled_dataset_smoke`
  - `outputs/tiled_dataset_smoke/data.yaml`
  - `outputs/tiled_dataset_smoke/tiled_dataset_report.md`
  - `outputs/tiled_dataset_smoke/debug_tiling/`
- Ran real tiled smoke pipeline:
  - `outputs/diagnostic_aug_tiled_smoke`
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
- Strategy memory appended the smoke record to `outputs/strategy_memory.jsonl`; no prior similar cases existed for rerank boost on this run.

## Operational Notes

- Formal training must use conda env `pytorch` with `D:\Anaconda\envs\pytorch\python.exe`.
- Formal YOLO training must use GPU `device=0`.
- CPU is allowed only for smoke/debug runs and must not be treated as formal experiment output.
- Do not use the base conda environment for formal training; it previously resolved to CPU-only PyTorch.
- Windows YOLO commands should keep `workers=0`.
- This machine needed `KMP_DUPLICATE_LIB_OK=TRUE` for the CPU YOLO smoke because the active Anaconda environment initialized duplicate OpenMP runtimes.
- The smoke used a capped tiled dataset (`--max-images-per-split 8`) to keep runtime bounded; it is not a final benchmark.
