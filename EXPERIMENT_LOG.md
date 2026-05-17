# Experiment Log

## 2026-05-17

### Artifact Cleanup And Output Normalization

Paused training and normalized generated artifacts.

- Added `scripts/audit_artifacts.py`.
- Generated inventory at `outputs/audits/artifact_inventory/artifact_inventory.md` and `.json`.
- Generated cleanup summary at `outputs/audits/artifact_inventory/cleanup_summary.md`.
- Added output convention at `docs/output_convention.md`.
- Archived legacy root-level outputs to `outputs/archive/old_outputs_20260517/`.
- Archived legacy `runs/detect/*` outputs to `outputs/archive/old_runs_20260517/runs_detect/`.
- `runs/detect` has no remaining old experiment entries.
- Active tiled smoke dataset is now `outputs/datasets/tiled/tiled_1024_ov20_smoke/`.
- Active 20 epoch baseline is now `outputs/experiments/20260517_tiled_baseline_20epoch/`.
- GPU preflight reports are now `outputs/audits/gpu_preflight/gpu_preflight_report.md` and `.json`.
- Project snapshots are now written to `outputs/snapshots/project_snapshot_latest.md` and `outputs/project_snapshot_latest.md`.

Policy recorded:

- Formal experiments must use `project=outputs/experiments/<run_id>`.
- Formal diagnostic runs must pass `--run-id` or explicitly set `--output-dir`.
- `runs/` is not a formal result location.
- Weights, generated images, generated dataset image/label files, and archive contents are local artifacts and are ignored by Git.

### Tiled Baseline 20 Epoch

Ran the tiled baseline on GPU with no external diagnostic augmentation and YOLO built-in augmentation knobs disabled.

Status after normalization:

- Run ID: `20260517_tiled_baseline_20epoch`
- Formal result: no
- Reason: this used the smoke tiled dataset, not the future full tiled dataset.
- Result path: `outputs/experiments/20260517_tiled_baseline_20epoch/`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml`
- Train command: `outputs/experiments/20260517_tiled_baseline_20epoch/configs/train_command.txt`
- Val command: `outputs/experiments/20260517_tiled_baseline_20epoch/configs/val_command.txt`
- Report: `outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_report.md`
- Metrics: `outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_metrics.json`
- best.pt: `outputs/experiments/20260517_tiled_baseline_20epoch/train/weights/best.pt`
- last.pt: `outputs/experiments/20260517_tiled_baseline_20epoch/train/weights/last.pt`

Normalized train command:

```powershell
D:\Anaconda\Scripts\conda.exe run -n pytorch yolo detect train model=yolo11n.pt data=E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_smoke\data.yaml epochs=20 imgsz=1024 batch=2 workers=0 device=0 project=outputs\experiments\20260517_tiled_baseline_20epoch name=train exist_ok=True mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0
```

Normalized validation command:

```powershell
D:\Anaconda\envs\pytorch\Scripts\yolo.exe detect val model=outputs\experiments\20260517_tiled_baseline_20epoch\train\weights\best.pt data=E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_smoke\data.yaml imgsz=1024 batch=2 workers=0 device=0 project=outputs\experiments\20260517_tiled_baseline_20epoch name=val exist_ok=True
```

Result:

- Completed 20 epochs: true
- Final train batch: 2
- Final val batch: 2
- OOM: false
- Precision: 0.828
- Recall: 0.213
- mAP50: 0.247
- mAP50-95: 0.181
- Train duration: 312.4 seconds
- Validation duration: 20.137 seconds

Note:

- Training artifacts and epoch 20 metrics were produced successfully.
- The conda wrapper emitted a UnicodeEncodeError while forwarding YOLO stdout after training; this was not a CUDA OOM and did not prevent `best.pt`, `last.pt`, or `results.csv` from being created.

### GPU Environment Confirmation

GPU environment has been confirmed in the dedicated conda env `pytorch`. Future formal training must use this environment rather than base.

- Conda env: `pytorch`
- Python executable: `D:\Anaconda\envs\pytorch\python.exe`
- Python version: 3.9.19
- PyTorch: 2.4.1
- `torch.cuda.is_available()`: True
- `torch.version.cuda`: 12.4
- CUDA device count: 1
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Ultralytics: 8.3.221

Training rule:

- CPU is only for smoke/debug runs.
- Formal YOLO training must use `device=0`.
- The base conda environment must not be used for formal training because it previously resolved to CPU-only PyTorch.

### GPU Preflight In Pytorch Env

Command:

```powershell
D:\Anaconda\Scripts\conda.exe run -n pytorch python scripts\run_gpu_preflight.py
```

Outputs:

- `outputs/audits/gpu_preflight/gpu_preflight_report.md`
- `outputs/audits/gpu_preflight/gpu_preflight_report.json`

Result:

- Conda env: `pytorch`
- `sys.executable`: `D:\Anaconda\envs\pytorch\python.exe`
- PyTorch: 2.4.1
- `torch.cuda.is_available()`: True
- `torch.version.cuda`: 12.4
- CUDA device count: 1
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Ultralytics: 8.3.221
- `yolo checks`: passed
- YOLO GPU smoke: passed

YOLO GPU smoke used the tiled smoke `data.yaml`, `epochs=1`, `imgsz=640`, `batch=1`, `workers=0`, `device=0`, and YOLO built-in augmentations disabled.

### Historical Base Env GPU Preflight

Historical base-env preflight confirmed that base was not a valid formal training environment.

- Python: 3.12.4
- PyTorch: 2.4.1+cpu
- `torch.cuda.is_available()`: False
- `torch.version.cuda`: None
- CUDA device count: 0
- PyTorch device name: NO CUDA
- `nvidia-smi`: NVIDIA GeForce RTX 3060 Laptop GPU, driver 560.81, 6144 MiB
- Ultralytics: 8.4.48
- `yolo checks`: passed, but reported CPU / GPU None / CUDA None
- YOLO GPU smoke: not run because CUDA was unavailable to PyTorch

Conclusion:

- `torch.cuda.is_available() = False` in base.
- Only CPU smoke/debug can run in base.
- Base results cannot be used as formal experiment results.

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

The smoke tiled dataset was originally built from `E:\TJGY\DataSet2_fixed` with `--max-images-per-split 8`, so it is explicitly a smoke dataset.

Current normalized outputs:

- `outputs/datasets/tiled/tiled_1024_ov20_smoke/`
- `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml`
- `outputs/datasets/tiled/tiled_1024_ov20_smoke/tiled_dataset_report.json`
- `outputs/datasets/tiled/tiled_1024_ov20_smoke/tiled_dataset_report.md`
- `outputs/datasets/tiled/tiled_1024_ov20_smoke/dataset_summary.md`

Summary:

- Source images: 16
- Output tiles: 193
- Retained bboxes: 375
- Empty tiles: 18

### Real Tiled Pipeline Smoke

The older CPU smoke pipeline output was archived to `outputs/archive/old_outputs_20260517/diagnostic_aug_tiled_smoke/`.

Key conclusions from that archived smoke:

- `copy_paste` was not hard filtered; soft penalties recorded bbox safety risks.
- Metric consistency audit reported zero precision/recall delta for this smoke and documented threshold/matching causes for expected metric differences.
- Strategy memory appended one record to the archived `strategy_memory.jsonl`.

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
- Dry-run pipeline invocation passed and produced an archived smoke output under `outputs/archive/old_outputs_20260517/diagnostic_aug_pipeline_smoke/`.
