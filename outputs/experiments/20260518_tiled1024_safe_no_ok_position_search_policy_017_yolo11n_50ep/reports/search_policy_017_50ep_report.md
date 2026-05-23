# search_policy_017 50 Epoch Report

- Scope: formal 50 epoch YOLO training for diagnosis-guided `search_policy_017`.
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Augmented dataset: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep\dataset\final_dataset`
- Training set: original train images `2301` + augmented train images `2301`.
- Train images / bboxes: `4602` / `6408`
- Val images / bboxes: `677` / `905`
- Policy JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/short_training/policies/search_policy_017/policy.json`
- Operations: `gaussian_noise(p=0.189, s=0.104), gamma(p=0.475, s=0.456), local_contrast(p=0.371, s=0.179), cutout(p=0.142, s=0.163), copy_paste(p=0.621, s=0.273)`
- bbox_valid_rate: `0.999380`
- image_failures / label_failures: `0` / `0`
- Train settings: `model=yolo11n.pt epochs=50 imgsz=1024 batch=2 workers=0 device=0 seed=42`
- YOLO built-in augmentations disabled: `mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0`
- Precision: `0.710`
- Recall: `0.616`
- mAP50: `0.681`
- mAP50-95: `0.474`
- Balanced score: `0.608450`
- Exceeds random external by mAP50-95: `false`
- Exceeds diag_policy_001 by mAP50-95: `false`
- Current best formal by mAP50-95: `false`
- Current best formal by balanced score: `false`
- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep\train\weights\best.pt`
- OOM: `false`
- Training wall seconds: `34573.7`
- Training wall hours: `9.604`

## Operation Details

| operation | prob | strength | params |
|---|---:|---:|---|
| gaussian_noise | 0.1887 | 0.1043 | `{"max_std": 0.035}` |
| gamma | 0.4747 | 0.4563 | `{"min_gamma": 0.75, "max_gamma": 1.35}` |
| local_contrast | 0.3709 | 0.1793 | `{"max_clip_limit": 2.5, "tile_grid_size": [8, 8], "blend": 0.65}` |
| cutout | 0.1416 | 0.1628 | `{"max_holes": 2, "max_fraction": 0.1496495372250915}` |
| copy_paste | 0.6206 | 0.2726 | `{"max_paste_count": 3, "max_overlap": 0.18, "fallback_max_overlap": 0.45, "prefer_small": true, "class_balanced": true, "max_attempts": 60, "dataset_class_counts": {"2": 36, "3": 17, "10": 32}, "target_classes": [2, 3, 10]}` |

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
|---:|---|---:|---:|
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.993 | 0.974 |
| 2 | 加强筋打伤 | 0.375 | 0.492 |
| 3 | 开裂 | 1.000 | 0.995 |
| 4 | 油污 | 0.429 | 0.394 |
| 5 | 浅划伤 | 0.615 | 0.643 |
| 6 | 漏背锡 | 0.565 | 0.615 |
| 7 | 碰伤 | 0.480 | 0.596 |
| 8 | 脏污 | 0.381 | 0.488 |
| 9 | 轮廓划伤 | 0.400 | 0.669 |
| 10 | 锡丝残留 | 0.333 | 0.329 |
| 11 | 锡尖 | 0.852 | 0.910 |
| 12 | 锡膏 | 0.587 | 0.749 |

## Overall Deltas

| reference | delta Precision | delta Recall | delta mAP50 | delta mAP50-95 | delta balanced |
|---|---:|---:|---:|---:|---:|
| baseline | +0.020 | +0.001 | +0.012 | +0.040 | +0.019250 |
| diag_policy_001 DiagAug 50ep | +0.024 | -0.072 | -0.036 | -0.022 | -0.028800 |
| YOLO default | -0.075 | -0.060 | -0.054 | -0.002 | -0.044100 |
| random external | -0.040 | -0.052 | -0.053 | -0.027 | -0.042350 |

## Commands

Training:

```powershell
D:\Anaconda\envs\pytorch\Scripts\yolo.exe detect train model=yolo11n.pt data=E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep\dataset\final_dataset\data.yaml epochs=50 imgsz=1024 batch=2 workers=0 device=0 seed=42 project=E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep name=train exist_ok=True mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0
```

Validation:

```powershell
D:\Anaconda\envs\pytorch\Scripts\yolo.exe detect val model=E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep\train\weights\best.pt data=E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position\data.yaml imgsz=1024 batch=2 workers=0 device=0 project=E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep name=val exist_ok=True
```
