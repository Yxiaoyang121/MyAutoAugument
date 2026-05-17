# Experiment Log

## 2026-05-17

### Tiled Baseline 20 Epoch

Ran the tiled baseline on GPU with no external diagnostic augmentation and YOLO built-in augmentation knobs disabled.

Dataset checks before training:

- `outputs/tiled_dataset_smoke/data.yaml`: exists
- `outputs/tiled_dataset_smoke/images/train`: 107 files
- `outputs/tiled_dataset_smoke/images/val`: 86 files
- `outputs/tiled_dataset_smoke/labels/train`: 107 files
- `outputs/tiled_dataset_smoke/labels/val`: 86 files

Train command:

```powershell
D:\Anaconda\Scripts\conda.exe run -n pytorch yolo detect train model=yolo11n.pt data=E:\TJGY\MinPaper\MyAutoAugument\outputs\tiled_dataset_smoke\data.yaml epochs=20 imgsz=1024 batch=2 workers=0 device=0 project=outputs\tiled_baseline_20epoch name=train exist_ok=True mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0
```

Validation command:

```powershell
D:\Anaconda\envs\pytorch\Scripts\yolo.exe detect val model=outputs\tiled_baseline_20epoch\train\weights\best.pt data=E:\TJGY\MinPaper\MyAutoAugument\outputs\tiled_dataset_smoke\data.yaml imgsz=1024 batch=2 workers=0 device=0 project=outputs\tiled_baseline_20epoch name=val exist_ok=True
```

Outputs:

- `outputs/tiled_baseline_20epoch/baseline_20epoch_report.md`
- `outputs/tiled_baseline_20epoch/baseline_20epoch_metrics.json`
- `outputs/tiled_baseline_20epoch/train/weights/best.pt`
- `outputs/tiled_baseline_20epoch/train/weights/last.pt`
- `outputs/tiled_baseline_20epoch/train_command.txt`
- `outputs/tiled_baseline_20epoch/val_command.txt`
- `outputs/tiled_baseline_20epoch/train_stdout.log`
- `outputs/tiled_baseline_20epoch/train_stderr.log`
- `outputs/tiled_baseline_20epoch/val_stdout.log`
- `outputs/tiled_baseline_20epoch/val_stderr.log`

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

Environment:

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

- `outputs/gpu_preflight_report.md`
- `outputs/gpu_preflight_report.json`
- `outputs/gpu_preflight_smoke/train_command.txt`
- `outputs/gpu_preflight_smoke/train_stdout.log`
- `outputs/gpu_preflight_smoke/train_stderr.log`

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

YOLO GPU smoke command:

```powershell
D:\Anaconda\envs\pytorch\Scripts\yolo.exe detect train model=E:\TJGY\MinPaper\MyAutoAugument\yolo11n.pt data=E:\TJGY\MinPaper\MyAutoAugument\outputs\tiled_dataset_smoke\data.yaml imgsz=640 batch=1 epochs=1 workers=0 device=0 project=E:\TJGY\MinPaper\MyAutoAugument\outputs\gpu_preflight_smoke name=train exist_ok=True mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0
```

Conclusion:

- GPU preflight passed in conda env `pytorch`.
- Minimal YOLO GPU smoke passed with YOLO built-in augmentations disabled.
- Formal training can proceed only from conda env `pytorch` using `device=0`; CPU remains smoke/debug only.

### Historical Base Env GPU Preflight

Historical base-env preflight. This run confirmed that base was not a valid formal training environment.

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

- Superseded by the current pytorch-env reports in `outputs/gpu_preflight_report.md/json`.

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
- The base Python environment is using CPU-only PyTorch even though the NVIDIA driver can see the GPU.
- Only CPU smoke/debug can run in base; base results cannot be used as formal experiment results.
- Superseding rule: formal training uses conda env `pytorch` and YOLO `device=0`.

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
