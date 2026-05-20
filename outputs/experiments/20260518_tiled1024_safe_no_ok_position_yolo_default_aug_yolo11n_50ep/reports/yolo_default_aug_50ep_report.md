# YOLO Default Augmentation 50 Epoch Report

## Run

- Run ID: `20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Model: `yolo11n.pt`
- Training settings: `epochs=50 imgsz=1024 batch=2 workers=0 device=0`
- YOLO default augmentations: enabled by leaving Ultralytics defaults untouched.
- OOM: `false`
- Training wall time: `13154.0s (3.65h)`
- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep\train\weights\best.pt`

## Actual Ultralytics Augmentation Args

| Parameter | Value |
| --- | ---: |
| `hsv_h` | `0.015` |
| `hsv_s` | `0.7` |
| `hsv_v` | `0.4` |
| `degrees` | `0.0` |
| `translate` | `0.1` |
| `scale` | `0.5` |
| `shear` | `0.0` |
| `perspective` | `0.0` |
| `flipud` | `0.0` |
| `fliplr` | `0.5` |
| `bgr` | `0.0` |
| `mosaic` | `1.0` |
| `mixup` | `0.0` |
| `cutmix` | `0.0` |
| `copy_paste` | `0.0` |
| `copy_paste_mode` | `flip` |
| `auto_augment` | `randaugment` |
| `erasing` | `0.4` |
| `close_mosaic` | `10` |

## Final Metrics

| Metric | YOLO default | Delta vs baseline | Delta vs DiagAug |
| --- | ---: | ---: | ---: |
| Precision | 0.785 | +0.095 | +0.099 |
| Recall | 0.676 | +0.061 | -0.012 |
| mAP50 | 0.735 | +0.066 | +0.018 |
| mAP50-95 | 0.476 | +0.042 | -0.020 |

## Commands

Training:

```powershell
D:\Anaconda\envs\pytorch\Scripts\yolo.exe detect train model=yolo11n.pt data=outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml epochs=50 imgsz=1024 batch=2 workers=0 device=0 project=outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep name=train exist_ok=True
```

Validation:

```powershell
D:/Anaconda/envs/pytorch/Scripts/yolo.exe detect val model=outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/train/weights/best.pt data=outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml imgsz=1024 batch=2 workers=0 device=0 project=outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep name=val exist_ok=True
```
