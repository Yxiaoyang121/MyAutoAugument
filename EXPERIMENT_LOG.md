# Experiment Log

## 2026-06-04

### Fixed CATF-v2 Multiseed Validation

Ran fixed CATF-v2 seed0 and seed2 after the strict no-augmentation bypass repair, reusing clean native seed0/1/2 and the already completed fixed seed1 run.

Run group:

- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/`
- Code commit used for training: `dfcd177fa058046073e9b8e87dc8052b7b705f9a`

Shared configuration:

- Model: `yolo11n.pt`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Epochs: 50
- Image size: 1024
- Batch: 2
- Workers: 0
- Device: 0
- YOLO default augmentation: official Ultralytics default remained enabled
- CATF-v2: enabled
- Class-aware feedback, ROI-aware augmentation, sample-aware routing: enabled
- Fixed augmented dataset generated: false
- Train images: 2301
- Copy-paste: not enabled

Per-seed results:

| Seed | Clean P | Clean R | Clean mAP50 | Clean mAP50-95 | Fixed P | Fixed R | Fixed mAP50 | Fixed mAP50-95 | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | false |
| 1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | false |
| 2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | true |

Delta vs clean:

- seed0: ΔP=-0.0060, ΔR=-0.0068, ΔmAP50=+0.0090, ΔmAP50-95=+0.0136.
- seed1: ΔP=+0.0127, ΔR=+0.0528, ΔmAP50=+0.0284, ΔmAP50-95=+0.0390.
- seed2: ΔP=+0.0674, ΔR=-0.0423, ΔmAP50=-0.0110, ΔmAP50-95=-0.0257.

Aggregate and safety:

- Fixed CATF-v2 constraint_failed count: 1/3.
- Old CATF-v2 constraint_failed count: 2/3.
- OK3 active across fixed seeds: false.
- OK3 ROI applied total: 0.
- Active class counts: class 11 x3, class 4 x2, class 12 x1, class 9 x1, class 8 x1.
- ROI affected class counts: class 11=132, class 9=25, class 4=14, class 12=14, class 8=6.

Interpretation:

- Strict no-augmentation bypass repair improved multiseed constraint stability from 1/3 passing to 2/3 passing.
- The fixed CATF-v2 path is cleaner and more credible than the old CATF-v2 multiseed result.
- Seed2 still fails due mAP50 and mAP50-95 drops, so CATF-v2 should not yet be claimed as a fully stable sole main method.

Reports:

- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.md`
- `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.json`

## 2026-06-03

### CATF-v2 Fixed Seed1 50 Epoch Rerun

Reran the critical seed1 CATF-v2 validation after fixing the formal no-augmentation transform bypass. This run is important because the old CATF-v2 seed1 result passed constraints while recording `ROI applied=0` and `industrial samples augmented=0`.

Run:

- Output: `outputs/experiments/catf_v2_fixed_seed1_50ep/`
- Entry: `scripts/train_yolo_default_with_inloop_feedback.py`
- Model: `yolo11n.pt`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Epochs: 50
- Seed: 1
- Batch: 2
- Workers: 0
- YOLO default augmentation: enabled
- CATF-v2: enabled
- Class-aware feedback, ROI-aware augmentation, sample-aware routing, and threshold calibration report: enabled
- Fixed augmented dataset generated: false
- Train images: 2301
- Epoch continuity: 1..50

Reference and result:

- Clean native seed1: Precision=0.7725, Recall=0.6477, mAP50=0.7542, mAP50-95=0.4799.
- Old CATF-v2 seed1: Precision=0.7691, Recall=0.6950, mAP50=0.7549, mAP50-95=0.4898.
- Fixed CATF-v2 seed1: Precision=0.7852, Recall=0.7005, mAP50=0.7826, mAP50-95=0.5189.
- Delta vs clean native seed1: Precision +0.0127, Recall +0.0528, mAP50 +0.0284, mAP50-95 +0.0390.
- Delta vs old CATF-v2 seed1: Precision +0.0161, Recall +0.0055, mAP50 +0.0277, mAP50-95 +0.0291.
- Constraint status: `constraint_failed=false`.

Augmentation and activation:

- Industrial samples augmented: 55.
- ROI applied: 56.
- ROI affected classes: class 11 (51), class 4 (5).
- Industrial ops applied: `local_contrast=28`, `sharpen_mild=27`.
- OK3 active: false.
- OK3 ROI applied: 0.
- Invalid bbox count: 0.
- Bbox out-of-bounds count: 0.
- Class id out-of-bounds count: 0.
- Policy actions: shrink=5, freeze=2, accept=1, observe=1.

Interpretation: the fixed seed1 improvement is not explained by the previously identified no-op label/Instances rewrite issue, because this rerun recorded actual CATF-v2 ROI/industrial augmentation.

Reports:

- `outputs/experiments/catf_v2_fixed_seed1_50ep/reports/final_report.md`
- `outputs/experiments/catf_v2_fixed_seed1_50ep/reports/final_metrics.json`
- `outputs/experiments/catf_v2_fixed_seed1_50ep/reports/compare_with_clean_native_seed1.md`

### CATF-v2 Strict No-Augmentation Bypass Fix

No training was run. Fixed CATF-v2 formal transform no-augmentation paths so they bypass label conversion, bbox validation/clip, and `Instances` rebuild unless an industrial/ROI augmentation is actually applied.

Implementation summary:

- `SampleAwareAugmentationRouter` now records `applied_any_aug`.
- No-active, empty-label, no_aug/stable-only, high-FP guarded-only, all-prob-zero, and ROI-unavailable paths return bypass results without bbox validation/clip.
- `UltralyticsOnlinePolicyTransform` returns the original YOLO label dict unchanged when `applied_any_aug=false`.
- ROI-unavailable precheck avoids probability draws when a ROI op cannot be applied.

Re-run transform parity audit:

- clean vs CATF-v2 noop final output identical: true
- clean vs CATF-v2 formal-force-skip final output identical: true
- bbox hash mismatches: 0 / 100
- raw vs formal-force-skip intermediate mismatches: 0 / 100
- formal-force-skip image/cls/Instances rewrites: 0 / 100
- router random draws: 0
- industrial/ROI applied ops: 0
- router bbox_oob/invalid/class-oob count: 0

Validation:

- `pytest -q tests/test_catf_v2_transform_bypass.py ... tests/test_copy_paste.py`: 81 passed.
- Syntax checks passed for CATF-v2 router/policy/threshold modules and in-loop training entrypoint.

Outputs:

- `outputs/audits/catf_v2_transform_parity/transform_parity_report.md`
- `outputs/audits/catf_v2_transform_parity/transform_parity.json`
- `outputs/audits/catf_v2_transform_parity/diff_samples/README.md`

### CATF-v2 Transform-Level Parity Audit

No training was run. Added and executed `scripts/audit_catf_v2_transform_parity.py` to compare clean native YOLO default transform output, CATF-v2 noop output, and CATF-v2 formal-force-skip output on 100 deterministic train samples from `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`.

Coverage:

- stable/no_aug samples: 42
- active defect samples: 23
- domain high-FP prior samples: 23
- low-support samples: 10
- multi-class samples: 38
- empty-label samples: 10

Result:

- clean vs CATF-v2 noop final output identical: true
- clean vs CATF-v2 formal-force-skip final output identical: false
- clean vs formal-force-skip final mismatches: 1 / 100
- raw vs formal-force-skip intermediate mismatches: 89 / 100
- formal-force-skip image/cls/Instances rewrites: 100 / 100
- router random draws: 0
- industrial/ROI applied ops: 0
- router `bbox_oob_count`: 1

The final mismatch was a class 4 `油污` sample where the raw bbox had `y2=1024.00048828125`; formal-force-skip route ran validation and clipped it to `1024.0`, causing a downstream final bbox hash difference even without applied augmentation. This can explain residual CATF-v2 path effects that are not reflected by applied-op statistics.

Outputs:

- `outputs/audits/catf_v2_transform_parity/transform_parity_report.md`
- `outputs/audits/catf_v2_transform_parity/transform_parity.json`
- `outputs/audits/catf_v2_transform_parity/diff_samples/`

Recommended next fix: bypass CATF-v2 transform before label conversion/validation unless an actual op is selected; force-skip/no-active/no_aug/stable/high-FP samples should return the native YOLO transform input unchanged.

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
- Feedback controller: `CATF` (Constraint-Aware Trust-region Feedback Controller).
- CATF uses the clean native YOLO default reference curve at matching feedback epochs, trust-region step limits, group budgets, delayed acceptance, rollback, cooldown, and epoch>=40 freeze.
- CATF-v1 is global feedback; CATF-v2 is class-aware, issue-aware, and sample-aware feedback with ROI-aware industrial augmentation.
- CATF-v2 current goal is to reduce CATF-v1 Precision instability by activating only diagnosed classes and freezing stable classes.
- Current CATF-v2 work is smoke-only; no formal 50 epoch CATF-v2 run should be inferred from it.
- No-feedback control disables both feedback and industrial augmentation, using Ultralytics YOLO default augmentation as the behavior check.
- The old YOLO default reference is not the final baseline after parity audit; feedback comparisons should use `clean_native_yolo_default_seed42_50ep`.
- Output: `outputs/experiments/catf_v2/`
- Epochs: `50`
- Feedback enabled: `true`
- Industrial augmentation enabled: `true`
- CATF version: `v2`
- Class-aware feedback: `true`
- ROI-aware augmentation: `true`
- Sample-aware routing: `true`
- Reference curve loaded: `true`
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Stage restart count: `0`
- Epoch continuous: `true`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- Constraint baseline: `clean_native_yolo_default`
- Constraint failed: `True`
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/seed_2/catf_v2/reports/final_report.md`
- Policy history: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/seed_2/catf_v2/reports/policy_history.json`
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
<!-- CLEAN_NATIVE_YOLO_DEFAULT_REFERENCE_START -->
## Clean Native YOLO Default Reference

- The old YOLO default reference is no longer treated as the final baseline because parity audit found it used `OnlineAugDetectionTrainer` with an empty passthrough policy.
- New baseline: `outputs/experiments/clean_native_yolo_default_seed42_50ep/`.
- Training mode: pure native Ultralytics `YOLO.train(**same_args)`.
- Custom trainer / callback / dataset / transform / industrial augmentation: `false`.
- Train image count: `2301`
- Precision/Recall/mAP50/mAP50-95: `0.7262/0.6844/0.7616/0.5250`
- close_mosaic official schedule expected: `true`
- close_mosaic expected start epoch: `41`
- Future feedback experiments should compare only against this clean native reference.
- Report: `outputs/experiments/clean_native_yolo_default_seed42_50ep/reports/clean_native_yolo_default_report.md`
- Metrics JSON: `outputs/experiments/clean_native_yolo_default_seed42_50ep/reports/clean_native_yolo_default_metrics.json`
<!-- CLEAN_NATIVE_YOLO_DEFAULT_REFERENCE_END -->

<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_INLOOP_FEEDBACK_START -->
## Multiseed Clean YOLO Default vs In-Loop Feedback

- Scope: seeds `0, 1, 2`; seed 42 is not included in the multiseed mean.
- Output: `outputs/experiments/multiseed_clean_yolo_default_vs_inloop_feedback/`.
- Clean group uses pure native Ultralytics `YOLO.train`; feedback group uses single-run in-loop feedback with YOLO default augmentation still enabled.
- Train images: `2301`; fixed augmented dataset generated: `false`; copy_paste remains pending/not enabled.
- Feedback wins under industrial constraints: `1/3`.
- Constraint failed seeds: `2/3`.
- Mean delta P/R/mAP50/mAP50-95: `-0.0275/0.0254/0.0093/0.0211`.
- Verdict: not stable enough to claim as the paper main result yet; use as diagnostic/ablation unless a stricter controller passes multiseed constraints.
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_inloop_feedback/reports/multiseed_summary.md`.
- JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_inloop_feedback/reports/multiseed_summary.json`.
<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_INLOOP_FEEDBACK_END -->

<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_CATF_FEEDBACK_START -->
## Multiseed Clean YOLO Default vs CATF Feedback

- Scope: seeds `0, 1, 2`; seed 42 is retained as a positive single-seed case but is not included in this multiseed mean.
- Output: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_feedback/`.
- Clean native results were reused from `outputs/experiments/multiseed_clean_yolo_default_vs_inloop_feedback/seed_*/clean_native_yolo_default/`; no duplicate clean native training was run.
- CATF group uses single-run in-loop feedback with YOLO default augmentation still enabled, per-seed clean native reference curves, industrial online augmentation enabled, and copy_paste pending/not enabled.
- Train images: `2301`; fixed augmented dataset generated: `false`; val uses original val tiles.
- Per-seed deltas P/R/mAP50/mAP50-95:
  - seed 0: `-0.0272/-0.0014/+0.0179/+0.0445`, constraint_failed=`true`.
  - seed 1: `-0.0349/+0.0621/+0.0008/+0.0386`, constraint_failed=`true`.
  - seed 2: `-0.0116/-0.0120/-0.0084/-0.0424`, constraint_failed=`true`.
- Mean delta P/R/mAP50/mAP50-95: `-0.0246/+0.0163/+0.0034/+0.0136`.
- CATF wins under industrial constraints: `0/3`; constraint failed seeds: `3/3`.
- Control statistics across seeds: rollback `17`, cooldown `3`, freeze `3`; frozen policy records `15`.
- Compared with old in-loop feedback, CATF is more conservative in logs but not more stable by the industrial constraint criterion: old feedback failed `2/3`, CATF failed `3/3`.
- Verdict: CATF is not recommended as the paper main method based on seeds `0/1/2`; keep seed42 as a positive case and report this as an ablation/controller attempt unless a later controller passes multiseed constraints.
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_feedback/reports/multiseed_catf_summary.md`.
- JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_feedback/reports/multiseed_catf_summary.json`.
<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_CATF_FEEDBACK_END -->

<!-- CATF_V2_ACTIVATION_AUDIT_START -->
## CATF-v2 Activation Audit

- Scope: `outputs/experiments/catf_v2_class_aware_10ep_smoke/reports/`.
- No training was run; this is a report-only audit of epoch 5 CATF-v2 activation.
- Audit outputs: `outputs/experiments/catf_v2_class_aware_10ep_smoke/reports/catf_v2_activation_audit.md` and `.json`.
- Active classes from smoke: class `1` OK3, class `6` 漏背锡, class `8` 脏污.
- OK3 activation is judged not reasonable for a formal run: Recall is already high (`0.9891`), FN count is only `2`, FP count is `49`, and OK-like classes should default to stable/no_aug unless evidence is very strong.
- 漏背锡 activation is judged reasonable: low Recall (`0.2826`), many FN (`32`), low-contrast evidence, and conservative ROI `sharpen_mild`/`local_contrast` ops.
- 脏污 activation is partially reasonable as low Recall, but should be guarded by stain/dirty high-FP domain priors and should not lower threshold or escalate photometric before FP behavior is known.
- Low-support classes `2` and `3` did not trigger strong photometric augmentation; they remain oversampling/copy-paste pending candidates.
- ROI augmentation applied `150` times, including OK3 (`122`), which is the main activation-rule concern.
- Recommendation: do not enter CATF-v2 seed42 50ep until activation rules are tightened with OK2/OK3 no_aug, stronger activation threshold, domain high-FP guards for stain/oil/dirty classes, and likely top_k reduced from `3` to `2`.
<!-- CATF_V2_ACTIVATION_AUDIT_END -->

<!-- CATF_V2_ACTIVATION_FIXED_START -->
## CATF-v2 Activation Rule Fix

- Scope: no 50 epoch training; only rule changes, tests, and a 10 epoch smoke run.
- Output: `outputs/experiments/catf_v2_activation_fixed_10ep_smoke/`.
- Report: `outputs/experiments/catf_v2_activation_fixed_10ep_smoke/reports/catf_v2_activation_fixed_report.md`.
- Rule changes: OK2/OK3 default no_aug, stricter activation thresholds, domain high-FP prior for OK2/OK3/oil/dirty classes, top_k_active_classes=`2`, ROI blocks no_aug/high-FP conflict classes.
- Smoke result: train_success=`true`, val_success=`true`, train_images=`2301`, fixed_augmented_dataset_generated=`false`.
- Active classes after fix: class `6` 漏背锡 (texture_boundary_weak), class `8` 脏污 (low_recall).
- OK3 active=`false`; OK3 ROI applied=`0`.
- 漏背锡 active=`true` with conservative ROI sharpen/local_contrast.
- 脏污 domain_high_fp_prior=`true`; photometric probs `{'clahe': 0.0, 'gamma': 0.0, 'brightness': 0.0, 'contrast': 0.0}`; threshold recommendation `0.25` with reason `domain_high_fp_prior_keep_threshold`.
- ROI stats: `{'roi_aug_applied': 23, 'roi_aug_skipped_small_roi': 0, 'roi_aug_skipped_conflict': 2, 'affected_classes': {'6': 20, '8': 3}}`.
- BBox/class valid: `true`.
- Recommendation: proceed to CATF-v2 seed42 50 epoch validation only after this fixed activation rule set; do not use the earlier CATF-v2 smoke as formal evidence.
<!-- CATF_V2_ACTIVATION_FIXED_END -->

<!-- CATF_V2_SEED42_50EP_START -->
## CATF-v2 Seed42 50 Epoch

- Output: `outputs/experiments/catf_v2_seed42_50ep/`.
- Entry: `scripts/train_yolo_default_with_inloop_feedback.py` with `--catf-version v2`, class-aware feedback, ROI-aware augmentation, sample-aware routing, threshold calibration report, top_k=`2`, top_m=`2`.
- Training mode: single-run continuous YOLO default training with official YOLO augmentation kept enabled; no stage restart; no self-implemented mosaic/randaugment replacement.
- Train images: `2301`; fixed augmented dataset generated: `false`; copy_paste remains `pending_object_bank_design`.
- Epoch continuity: `1..50` continuous.
- Final metrics P/R/mAP50/mAP50-95: `0.7498/0.7257/0.7679/0.5212`.
- Delta vs clean native seed42 P/R/mAP50/mAP50-95: `+0.0236/+0.0413/+0.0062/-0.0039`.
- constraint_failed: `false`.
- OK2/OK3 active epochs: `[]`; OK3 ROI applied: `0`.
- Active class counts: `{'6:漏背锡': 1, '12:锡膏': 1}`.
- ROI stats: `{'roi_aug_applied': 37, 'roi_aug_skipped_small_roi': 0, 'roi_aug_skipped_conflict': 0, 'affected_classes': {'6': 18, '12': 19}}`.
- Feedback actions: `{'observe': 2, 'accept': 1, 'shrink': 4, 'freeze': 2}`; class actions: `{'propose': 2, 'observe': 3}`; rollback/cooldown/freeze: `0/0/4`.
- Report: `outputs/experiments/catf_v2_seed42_50ep/reports/final_report.md`.
- Metrics JSON: `outputs/experiments/catf_v2_seed42_50ep/reports/final_metrics.json`.
- Policy history: `outputs/experiments/catf_v2_seed42_50ep/reports/policy_history.json`.
- Best checkpoint: `outputs/experiments/catf_v2_seed42_50ep/train/weights/best.pt`.
- Verdict: seed42 passes industrial constraints and is suitable for CATF-v2 multiseed validation; do not claim final method before multiseed passes.
<!-- CATF_V2_SEED42_50EP_END -->

<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_CATF_V2_START -->
## Multiseed Clean YOLO Default vs CATF-v2

- Scope: seeds `0, 1, 2`; seed 42 remains a positive single-seed validation and is not included in the multiseed mean.
- Output: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/`.
- Clean native results were reused from existing per-seed clean YOLO default runs; CATF-v2 was newly trained for each seed.
- CATF-v2 settings: official YOLO default augmentation kept enabled, class-aware feedback, ROI-aware augmentation, sample-aware routing, threshold calibration report, top_k=`2`, top_m=`2`, feedback interval=`5`.
- Train images: `2301`; fixed augmented dataset generated: `false`; copy_paste remains `pending_object_bank_design`.
- Per-seed deltas P/R/mAP50/mAP50-95:
  - seed 0: `-0.0417/+0.0218/+0.0223/+0.0264`, constraint_failed=`true` due Precision drop.
  - seed 1: `-0.0034/+0.0474/+0.0007/+0.0099`, constraint_failed=`false`.
  - seed 2: `+0.1395/-0.1247/-0.0269/-0.0285`, constraint_failed=`true` due mAP50 and mAP50-95 drops.
- Mean delta P/R/mAP50/mAP50-95: `+0.0315/-0.0185/-0.0013/+0.0026`.
- CATF-v2 wins under industrial constraints: `1/3`; constraint failed seeds: `2/3`.
- OK2/OK3 were never active; OK3 ROI applied total: `0`.
- Active class counts: `{'8:脏污': 1, '11:锡尖': 1}`; ROI affected totals: `{'8:脏污': 3, '11:锡尖': 20}`.
- Control statistics across seeds: rollback `0`, cooldown `0`, freeze events `12`; policy actions `{'shrink': 14, 'accept': 2, 'observe': 5, 'freeze': 6}`.
- Compared with CATF-v1, CATF-v2 improves activation discipline and reduces constraint failures from `3/3` to `2/3`, but is still not stable enough for the paper main method.
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/multiseed_catf_v2_summary.md`.
- JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/multiseed_catf_v2_summary.json`.
<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_CATF_V2_END -->

<!-- CATF_V2_FAILURE_MODE_ANALYSIS_START -->
## CATF-v2 Multiseed Failure-Mode Analysis

- Scope: analysis only; no training was run and no CATF-v2 rules were changed.
- Source experiment: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/`.
- Generated reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/curve_diagnosis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/active_class_effect_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/precision_drop_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/recall_drop_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/threshold_calibration_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/controller_behavior_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/catf_v2_failure_mode_summary.md`
- Main failure modes: `intervention too weak`, `intervention wrong target`, `precision threshold issue`, `over-conservative freeze`, `per-class diagnosis trajectory sensitive`, and `in-loop diagnosis/callback side effect not isolated`.
- Seed 0 failure: Recall/mAP improved, but Precision failed due approximate FP increases across multiple classes, led by oil/dirty-like and defect classes; threshold calibration is the most direct repair candidate.
- Seed 1 pass: no ROI/industrial augmentation was applied, so the pass is not causal evidence for ROI augmentation; it may include in-loop diagnosis/callback RNG effects.
- Seed 2 failure: Precision increased while Recall/mAP dropped, indicating conservative confidence/detection behavior on a clean seed that already had high Recall.
- ROI-aware augmentation was too sparse to prove benefit: only `23` ROI applications across seeds `0/1/2`.
- Controller behavior: shrink-dominant (`14` shrink vs `2` accept), zero rollback/cooldown, and freeze at epoch 40/45; negative-effect attribution is missing.
- Recommendation: do not claim CATF-v2 as paper main method yet; next step should be diagnosis-only in-loop control plus per-class threshold calibration analysis before more 50 epoch training.
<!-- CATF_V2_FAILURE_MODE_ANALYSIS_END -->

<!-- THRESHOLD_CALIBRATION_DIAGNOSIS_ONLY_START -->
## CATF-v2 Threshold Calibration and Diagnosis-Only Control

- Scope: no new 50 epoch training and no CATF-v2 rule changes.
- Post-hoc threshold report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/threshold_calibration_posthoc.md`.
- Post-hoc JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/threshold_calibration_posthoc.json`.
- Prediction-only inputs: cached/generated validation predictions for clean native and CATF-v2 seeds `0/1/2`; no training was run.
- Threshold grid: per-class confidence threshold `0.10..0.70` step `0.05`; objectives `constrained_score`, `balanced_score`, `industrial_score`.
- Constrained post-hoc result: CATF-v2 passes industrial constraints for `2/3` seeds after calibration.
- Seed 0: Precision/constraint failure is repairable by threshold calibration in the post-hoc evaluator; precision-oriented objectives raise thresholds for OK/oil/dirty-like and stable classes while lowering difficult defect classes.
- Seed 2: Recall/mAP failure is not repaired by threshold lowering; this remains a training/trajectory degradation, not a pure confidence-threshold issue.
- Diagnosis-only smoke: `outputs/experiments/diagnosis_only_inloop_control_10ep_smoke/`.
- Diagnosis-only plan: `outputs/experiments/diagnosis_only_inloop_control_plan.md`.
- Smoke result: diagnosis callback executed at epoch `5`; industrial samples augmented=`0`, ROI applied=`0`, policy update applied=`0`, train images=`2301`, fixed augmented dataset generated=`false`, epoch sequence `1..10` continuous, bbox/class legal.
- Interpretation: threshold calibration can be a deployment/post-processing companion, but seed 2 shows it cannot substitute for robust training feedback.
<!-- THRESHOLD_CALIBRATION_DIAGNOSIS_ONLY_END -->

<!-- DIAGNOSIS_ONLY_CONTROL_50EP_START -->
## Diagnosis-Only In-Loop Control 50 Epoch

- Scope: seeds `0, 1, 2`; this is the control for in-loop diagnosis callback/RNG/training-path effects.
- Output: `outputs/experiments/diagnosis_only_inloop_control_50ep/`.
- Entry: `scripts/train_yolo_default_with_inloop_feedback.py`.
- Configuration: YOLO default augmentation enabled, feedback diagnosis enabled, `diagnosis_only=true`, `industrial_aug_enabled=false`, ROI-aware augmentation disabled, sample-aware routing disabled, threshold mutation disabled, copy_paste not enabled.
- Train images: `2301`; fixed augmented dataset generated: `false`; results.csv epoch `1..50` continuous for all seeds.
- Control counters for all seeds: industrial samples augmented=`0`, ROI applied=`0`, policy update applied=`0`, bbox/class legal=`true`.
- Final diagnosis-only metrics P/R/mAP50/mAP50-95:
  - seed 0: `0.7846/0.6765/0.7347/0.4759`.
  - seed 1: `0.7725/0.6477/0.7542/0.4799`.
  - seed 2: `0.6962/0.7286/0.7692/0.5224`.
- Delta vs clean native for all seeds and all four metrics: `0.0000`; diagnosis-only constraint pass count: `3/3`.
- Interpretation: the diagnosis callback alone did not change training results. Seed 1 CATF-v2 success is not explained by callback/RNG alone, though CATF-v2 industrial-enabled training-path differences still need caution because that seed reported zero actual industrial/ROI augmentation.
- CATF-v2 + post-hoc threshold calibration retains independent value as a deployment/post-processing companion: pass count improves from `1/3` to `2/3`, but seed 2 remains unrepaired.
- Summary report: `outputs/experiments/diagnosis_only_inloop_control_50ep/reports/diagnosis_only_multiseed_summary.md`.
- Summary JSON: `outputs/experiments/diagnosis_only_inloop_control_50ep/reports/diagnosis_only_multiseed_summary.json`.
<!-- DIAGNOSIS_ONLY_CONTROL_50EP_END -->

<!-- CATF_V2_NOOP_PARITY_AUDIT_START -->
## CATF-v2 No-op Parity Audit

- Scope: no new CATF-v2 strategy training; added explicit `--catf-noop` audit mode and ran parity controls.
- Output smoke: `outputs/experiments/catf_v2_noop_parity_smoke/`.
- Output 50ep control: `outputs/experiments/catf_v2_noop_control_50ep/seed_1/`.
- `--catf-noop` behavior: CATF-v2 custom trainer/dataset/router objects are built, but the online transform returns labels unchanged before bbox conversion, sample router is not called, router random draws=`0`, ROI applied=`0`, industrial samples augmented=`0`, and policy update applied=`0`.
- 1ep parity smoke: clean native vs CATF-v2 noop metrics/loss/lr/results.csv numeric fields are identical except wall-clock `time`; args.yaml differs only in `project` and `save_dir`.
- 1ep smoke report: `outputs/experiments/catf_v2_noop_parity_smoke/reports/noop_parity_report.md`.
- Random path audit: `outputs/experiments/catf_v2_noop_parity_smoke/reports/random_path_audit.md`.
- Seed1 50ep CATF-v2 noop metrics P/R/mAP50/mAP50-95: `0.7725/0.6477/0.7542/0.4799`.
- Delta seed1 noop vs clean native: `0.0000/0.0000/0.0000/0.0000`; constraint_failed=`false`.
- Delta seed1 noop vs CATF-v2: `+0.0034/-0.0474/-0.0007/-0.0099`.
- Interpretation: diagnosis-only already showed the callback itself is neutral; CATF-v2 noop now shows the custom CATF-v2 framework path is also neutral when augmentation/policy mutation are hard-disabled. Seed1 CATF-v2 gain is therefore not explained by no-op framework perturbation, but still cannot be attributed to ROI augmentation for that seed because actual ROI/industrial counters were zero.
- Seed1 noop report: `outputs/experiments/catf_v2_noop_control_50ep/reports/noop_control_seed1_report.md`.
<!-- CATF_V2_NOOP_PARITY_AUDIT_END -->

<!-- FIXED_CATF_V2_SEED2_THRESHOLD_ANALYSIS_START -->
## Fixed CATF-v2 Seed2 Failure and Threshold Calibration Analysis

- Date: `2026-06-04`.
- Scope: analysis-only; no train command was run.
- Source: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/`.
- Script: `D:\Anaconda\envs\pytorch\python.exe scripts\analyze_fixed_catf_v2_seed2_and_thresholds.py`.
- Reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed2_failure_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_threshold_calibration_posthoc.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_vs_old_catf_v2_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_catf_v2_next_step_summary.md`
- Seed2 fixed CATF-v2 result vs clean: P/R/mAP50/mAP50-95 delta `+0.0674/-0.0423/-0.0110/-0.0257`; constraint failed due mAP drops.
- Failure diagnosis: CATF-v2 made seed2 more precision-oriented on a high-recall clean baseline, with Recall loss in `浅划伤`, `轮廓划伤`, `加强筋打伤`, `漏背锡`, `锡膏` and AP/localization loss in `锡丝残留`, `脏污`, `锡膏`, `轮廓划伤`, `开裂`, `加强筋打伤`.
- Seed2 active classes were class `9` 轮廓划伤, class `11` 锡尖, and class `8` 脏污; ROI was not aligned with all final degraded classes.
- Post-hoc threshold calibration repairs seed2 in the analysis evaluator, but fixed CATF-v2 calibrated pass count remains `2/3` because seed0 still fails mAP50-95.
- Conclusion: fixed CATF-v2 is a strong candidate and threshold calibration is useful, but the method should not yet be described as stably superior to clean YOLO default.
<!-- FIXED_CATF_V2_SEED2_THRESHOLD_ANALYSIS_END -->
