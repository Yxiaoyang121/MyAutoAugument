# Project State

Last updated: 2026-05-17

## Current Position

The repository is centered on diagnosis-driven augmentation for industrial defect detection. The active path still keeps YOLO network architecture unchanged and focuses on dataset construction, validation-error diagnosis, policy generation, proxy safety, short training, and auditable reporting.

## Implemented In This Update

- Added `scripts/run_gpu_preflight.py` to capture Python, PyTorch/CUDA, Ultralytics, `yolo checks`, optional `nvidia-smi`, and conditional YOLO GPU smoke results.
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

- GPU preflight:
  - `python scripts\run_gpu_preflight.py`
  - `python --version`: Python 3.12.4
  - PyTorch: 2.4.1+cpu
  - `torch.cuda.is_available()`: False
  - `torch.version.cuda`: None
  - CUDA device count: 0
  - PyTorch device name: NO CUDA
  - `nvidia-smi`: NVIDIA GeForce RTX 3060 Laptop GPU, driver 560.81, 6144 MiB
  - Ultralytics: 8.4.48
  - `yolo checks`: passed, but reported CPU / GPU None / CUDA None
  - YOLO GPU smoke: not run because CUDA is unavailable to PyTorch
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

- Formal training is paused until the active Python environment has CUDA-enabled PyTorch and the YOLO GPU smoke passes.
- Current blocker: `torch.cuda.is_available() = False`; although `nvidia-smi` detects the RTX 3060 Laptop GPU, the active PyTorch package is `2.4.1+cpu`.
- CPU smoke tests can run in this environment, but they cannot be used as formal experiment results.
- Windows YOLO commands should keep `workers=0`.
- This machine needed `KMP_DUPLICATE_LIB_OK=TRUE` for the CPU YOLO smoke because the active Anaconda environment initialized duplicate OpenMP runtimes.
- The smoke used a capped tiled dataset (`--max-images-per-split 8`) to keep runtime bounded; it is not a final benchmark.
