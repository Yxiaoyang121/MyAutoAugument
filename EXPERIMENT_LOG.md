# Experiment Log

## 2026-05-18

### Safe Tiled No OK/Position Baseline YOLO11n 50 Epoch

Ran the formal baseline on the audited safe tiled dataset with `OK` and `定位` removed. The first sandboxed launch failed during Ultralytics label-cache multiprocessing pipe creation with Windows permission error; the same train command was rerun outside the sandbox and completed. This was not CUDA OOM.

Dataset audit:

- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Safe tiled dataset: true
- Removed classes: `OK`, `定位`
- Retained classes: `OK2`, `OK3`, `加强筋打伤`, `开裂`, `油污`, `浅划伤`, `漏背锡`, `碰伤`, `脏污`, `轮廓划伤`, `锡丝残留`, `锡尖`, `锡膏`
- Train/val tiles: 2301 / 677
- Train/val bboxes: 3182 / 905
- Total bboxes: 4087
- Class ids out of range: false

Training command:

```powershell
yolo detect train model=yolo11n.pt data=outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml epochs=50 imgsz=1024 batch=2 workers=0 device=0 project=outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep name=train exist_ok=True mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0
```

Validation command:

```powershell
yolo detect val model=outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/best.pt data=outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml imgsz=1024 batch=2 workers=0 device=0 project=outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep name=val exist_ok=True
```

Result:

- Completed 50 epochs: true
- Final batch: 2
- OOM: false
- Training time: 2.826 hours
- Precision: 0.690
- Recall: 0.615
- mAP50: 0.669
- mAP50-95: 0.434
- Lowest Recall class: `开裂` (0.000)
- best.pt: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/best.pt`
- last.pt: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/last.pt`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_report.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_metrics.json`

### Filtered Baseline Dataset Without OK And 定位

No training was run. Built a filtered dataset from `outputs/datasets/tiled/tiled_1024_ov20_full_safe/` that removes only `OK` and `定位`, keeps `OK2` and `OK3`, and remaps class ids to `0..12`.

Build command:

```powershell
python scripts\filter_tiled_dataset.py --source-root outputs\datasets\tiled\tiled_1024_ov20_full_safe --output-root outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position --debug-limit 50 --overwrite
```

Outputs:

- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/`
- Data YAML: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Class filter report: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/class_filter_report.md`
- Class filter report JSON: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/class_filter_report.json`
- Dataset summary: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/dataset_summary.md`
- Debug samples: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/debug_samples/` (50 images)

Filter result:

- Deleted classes: `OK`, `定位`
- Kept classes: `OK2`, `OK3`, `加强筋打伤`, `开裂`, `油污`, `浅划伤`, `漏背锡`, `碰伤`, `脏污`, `轮廓划伤`, `锡丝残留`, `锡尖`, `锡膏`
- New class ids: `0..12`
- Source train/val images: 2452 / 677
- Filtered train/val images: 2301 / 677
- Source bbox count: 4269
- Filtered bbox count: 4087
- Train bbox count before/after: 3337 / 3182
- Val bbox count before/after: 932 / 905
- Train empty tiles retained: 210
- Val empty tiles retained: 83
- Class id out of range: false
- Chinese class names damaged: false
- Formal baseline ready: true

Verification:

- Syntax check passed for `scripts/filter_tiled_dataset.py`.
- `pytest -q tests\test_filter_tiled_dataset.py` passed: 1 test.

### Tiling Quality Audit And Safe Full Dataset

No training or YOLO validation was run. Audited the previously built `tiled_1024_ov20_full` dataset for partial-object bbox risk, then rebuilt a stricter safe tiled dataset.

Old full tiled quality audit command:

```powershell
python scripts\audit_tiling_quality.py
```

Audit outputs:

- `outputs/audits/tiling_quality/tiling_quality_audit.md`
- `outputs/audits/tiling_quality/tiling_quality_audit.json`
- `outputs/audits/tiling_quality/debug_truncated_bboxes/` (50 images)

Old full tiled audit result:

- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full/`
- Total bboxes: 7465
- Visibility < 0.5: 1091
- Visibility < 0.7: 1956
- Visibility < 0.8: 2405
- Visibility < 0.9: 2874
- Bboxes touching tile boundary: 3249 (43.52%)
- Border-truncated bboxes: 3197 (42.83%)
- Severe truncated bboxes with visibility < 0.7: 1956 (26.20%)
- Status: unsafe for formal baseline; retained only for audit.

Safe rebuild command:

```powershell
python scripts\build_yolo_tiled_dataset.py --dataset-root E:\TJGY\DataSet2_fixed --data-yaml E:\TJGY\DataSet2_fixed\data.yaml --output-dir outputs\datasets\tiled\tiled_1024_ov20_full_safe --tile-size 1024 --overlap 0.2 --min-visibility 0.7 --large-object-min-visibility 0.9 --drop-border-truncated True --border-margin 2 --require-box-center-inside True --keep-empty-ratio 0.1 --seed 42 --debug-limit 100 --overwrite
```

Safe dataset outputs:

- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/`
- Data YAML: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/data.yaml`
- Dataset summary: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/dataset_summary.md`
- Dataset report: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/tiled_dataset_report.md`
- Dataset report JSON: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/tiled_dataset_report.json`
- Debug tile bbox visualizations: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/debug_tiling/` (100 images)

Safe dataset result:

- Tiled images: 2452 train, 677 val
- Original bboxes: 3084
- Safe tiled bboxes: 4269
- Dropped bbox candidates after tile intersection: 13826
- Visibility-failed dropped candidates: 13346
- Border-truncated dropped candidates: 3961
- Center-outside dropped candidates: 9855
- Obvious half-target bbox remains: false
- Class id out of range: false
- Chinese class names damaged: false
- Caveat: class `定位` has 0 retained bboxes under the strict large-structure rule.
- Policy: subsequent formal baseline runs must use `outputs/datasets/tiled/tiled_1024_ov20_full_safe/data.yaml`.

Verification:

- Syntax check passed for `scripts/build_yolo_tiled_dataset.py` and `scripts/audit_tiling_quality.py`.
- `pytest -q tests\test_build_yolo_tiled_dataset.py` passed: 2 tests.

### Full Tiled Dataset Build And Mapping Audit

No training was run. Built the full tiled YOLO dataset from `E:\TJGY\DataSet2_fixed` without `--max-images-per-split`.

Build command:

```powershell
python scripts\build_yolo_tiled_dataset.py --dataset-root E:\TJGY\DataSet2_fixed --data-yaml E:\TJGY\DataSet2_fixed\data.yaml --output-dir outputs\datasets\tiled\tiled_1024_ov20_full --tile-size 1024 --overlap 0.2 --min-visibility 0.3 --keep-empty-ratio 0.1 --seed 42 --debug-limit 30 --overwrite
```

Outputs:

- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full/`
- Data YAML: `outputs/datasets/tiled/tiled_1024_ov20_full/data.yaml`
- Dataset summary: `outputs/datasets/tiled/tiled_1024_ov20_full/dataset_summary.md`
- Dataset report: `outputs/datasets/tiled/tiled_1024_ov20_full/tiled_dataset_report.md`
- Dataset report JSON: `outputs/datasets/tiled/tiled_1024_ov20_full/tiled_dataset_report.json`
- Debug tile bbox visualizations: `outputs/datasets/tiled/tiled_1024_ov20_full/debug_tiling/` (30 images)

Build result:

- Original images: 461 train, 116 val
- Tiled images: 4155 train, 1098 val
- Original bboxes: 3084
- Tiled bboxes: 7465
- Empty tiles retained: 478
- Dropped bboxes in retained tiles: 22202
- Drop reasons: `below_min_visibility=6212`, `outside_tile=15990`
- `data.yaml`: `nc=15`, names inherited from original with `yaml.safe_dump(..., allow_unicode=True)`
- Tiled class id range: 0..14
- Class id >= nc: none
- Chinese class names damaged: false

Audit command:

```powershell
python scripts\audit_dataset_mapping.py
```

Audit outputs:

- `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.md`
- `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.json`

Audit result:

- Full source coverage confirmed: 461 train and 116 val source images.
- Tiled names match the original `data.yaml` exactly.
- No class id >= nc and no negative class id found.
- Chinese class names are intact.
- Superseded by the tiling quality audit above: this dataset is now marked unsafe for formal baseline because partial-object bboxes were retained.

Verification:

- Syntax check passed for `scripts/build_yolo_tiled_dataset.py` and `scripts/audit_dataset_mapping.py`.
- `pytest -q tests\test_build_yolo_tiled_dataset.py` passed: 2 tests.

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

### Dataset Mapping And Class Distribution Audit

No training was run. Audited `E:\TJGY\DataSet2_fixed` and `outputs/datasets/tiled/tiled_1024_ov20_smoke/`.

Outputs:

- `outputs/audits/dataset_mapping/dataset_mapping_audit.md`
- `outputs/audits/dataset_mapping/dataset_mapping_audit.json`

Code updates:

- Added `scripts/audit_dataset_mapping.py`.
- Updated `scripts/build_yolo_tiled_dataset.py` so tiled `data.yaml` is written with `yaml.safe_dump(..., allow_unicode=True)`.
- Repaired `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml` to fully inherit names from `E:\TJGY\DataSet2_fixed\data.yaml`.

Findings:

- Original data.yaml: `nc=15`, class ids in labels are 0..14, no class id >= nc.
- Tiled smoke data.yaml: `nc=15`, class ids in labels are 0..14, no class id >= nc.
- Original data: 461 train images, 116 val images, 2433 train bboxes, 651 val bboxes.
- Tiled smoke data: 107 train tiles, 86 val tiles, 232 train bboxes, 143 val bboxes.
- Tiled smoke is not full: it uses 16 source images and the dataset summary records the per-split source cap.
- Existing 20 epoch `blank-or-unrendered` rows are from the prior corrupted/non-renderable tiled class names in saved val logs/metrics, not from invalid class ids.
- Low mAP is mainly from smoke subset imbalance and underrepresented classes: 开裂, 漏背锡, 碰伤, 轮廓划伤, 锡丝残留, and 锡膏 have recall 0 in the saved per-class metrics.

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
