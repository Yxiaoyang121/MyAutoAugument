# Diagnosis-Driven Augmentation 50 Epoch Report

## Run

- run_id: `20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep`
- Baseline best.pt: `E:/TJGY/MinPaper/MyAutoAugument/outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/best.pt`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Model: `yolo11n.pt`
- Training settings: `epochs=50 imgsz=1024 batch=2 workers=0 device=0`
- YOLO built-in augmentations disabled: `mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0`
- OOM: `false`

## Selected Policy

- Policy: `diag_policy_001`
- Source issue: `low_contrast_missed_defect`
- Contains copy_paste: `false`
- Operations: clahe(p=0.411, s=0.382), contrast(p=0.407, s=0.372), gamma(p=0.324, s=0.311), brightness(p=0.229, s=0.241)

## Augmented Dataset

- Data YAML: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep\dataset_builder\final_dataset\data.yaml`
- Train images: `4602`
- Train bboxes: `6364`
- Val images: `677`
- Val bboxes: `905`

## Final Metrics

- Precision: `0.686`
- Recall: `0.688`
- mAP50: `0.717`
- mAP50-95: `0.496`

## Baseline Delta

| Metric | Delta |
| --- | ---: |
| Precision | -0.004 |
| Recall | +0.073 |
| mAP50 | +0.048 |
| mAP50-95 | +0.062 |

## Commands

Training:

```powershell
D:\Anaconda\envs\pytorch\Scripts\yolo.exe detect train model=yolo11n.pt data=outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/dataset_builder/final_dataset/data.yaml epochs=50 imgsz=1024 batch=2 workers=0 device=0 project=outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep name=train exist_ok=True mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0
```

Validation:

```powershell
D:\Anaconda\envs\pytorch\Scripts\yolo.exe detect val model=outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/train/weights/best.pt data=outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml imgsz=1024 batch=2 workers=0 device=0 project=outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep name=val exist_ok=True
```
