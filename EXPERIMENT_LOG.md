# Experiment Log

## 2026-05-17

### GPU Preflight

Formal training remains paused. The current goal was to verify whether this environment can start GPU-backed YOLO training before any further experiments.

Command:

```powershell
python scripts\run_gpu_preflight.py
```

Required commands captured in the report:

```powershell
python --version
python -c "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('cuda_version:', torch.version.cuda); print('device_count:', torch.cuda.device_count()); print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -c "import ultralytics; print('ultralytics:', ultralytics.__version__)"
yolo checks
```

Outputs:

- `outputs/gpu_preflight_report.md`
- `outputs/gpu_preflight_report.json`

Result:

- Python: 3.12.4
- PyTorch: 2.4.1+cpu
- `torch.cuda.is_available()`: False
- `torch.version.cuda`: None
- CUDA device count: 0
- PyTorch device name: NO CUDA
- `nvidia-smi`: NVIDIA GeForce RTX 3060 Laptop GPU, driver 560.81, 6144 MiB
- Ultralytics: 8.4.48
- `yolo checks`: passed, but reported CPU / GPU None / CUDA None
- YOLO GPU smoke: not run because CUDA is unavailable to PyTorch

Conclusion:

- `torch.cuda.is_available() = False`
- The active Python environment is likely using CPU-only PyTorch even though the NVIDIA driver can see the GPU.
- Only CPU smoke tests can run in this state; CPU smoke results cannot be used as formal experiment results.
- Do not start formal training until CUDA-enabled PyTorch is installed/activated and the YOLO GPU smoke test passes.

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

Command:

```powershell
python scripts\build_yolo_tiled_dataset.py --dataset-root E:\TJGY\DataSet2_fixed --data-yaml E:\TJGY\DataSet2_fixed\data.yaml --output-dir outputs\tiled_dataset_smoke --tile-size 1024 --overlap 0.2 --min-visibility 0.3 --keep-empty-ratio 0.1 --max-images-per-split 8 --debug-limit 8 --overwrite
```

Outputs:

- `outputs/tiled_dataset_smoke`
- `outputs/tiled_dataset_smoke/data.yaml`
- `outputs/tiled_dataset_smoke/tiled_dataset_report.json`
- `outputs/tiled_dataset_smoke/tiled_dataset_report.md`
- `outputs/tiled_dataset_smoke/debug_tiling/`

Summary:

- Source images: 16
- Output tiles: 193
- Retained bboxes: 375
- Empty tiles: 18

### Real Tiled Pipeline Smoke

Command:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python scripts\run_diagnostic_augmentation_pipeline.py --dataset-root outputs\tiled_dataset_smoke --data-yaml outputs\tiled_dataset_smoke\data.yaml --output-dir outputs\diagnostic_aug_tiled_smoke --model yolo11n.pt --imgsz 640 --batch 4 --baseline-epochs 1 --short-epochs 1 --final-epochs 1 --top-k 1 --workers 0 --device cpu --skip-final-train --proxy-samples 32 --augment-repeat 1
```

Outputs:

- `outputs/diagnostic_aug_tiled_smoke/diagnosis/diagnosis.json`
- `outputs/diagnostic_aug_tiled_smoke/policies/policy_update_report.md`
- `outputs/diagnostic_aug_tiled_smoke/proxy/copy_paste_filter_audit.md`
- `outputs/diagnostic_aug_tiled_smoke/metric_consistency/metric_consistency_audit.md`
- `outputs/diagnostic_aug_tiled_smoke/strategy_memory/memory_guided_ranking.json`
- `outputs/diagnostic_aug_tiled_smoke/strategy_memory/strategy_memory_report.md`

Key conclusions:

- `copy_paste` was not hard filtered; soft penalties recorded bbox safety risks.
- Metric consistency audit reported zero precision/recall delta for this smoke and documented threshold/matching causes for expected metric differences.
- Strategy memory appended one record to `outputs/strategy_memory.jsonl`.

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
- Dry-run pipeline invocation passed and produced `outputs/diagnostic_aug_pipeline_smoke`.
