# Clean Native YOLO Default 50 Epoch Report

## Integrity

- Pure native YOLO.train: `true`
- Custom trainer used: `false`
- OnlineAugDetectionTrainer used: `false`
- Feedback callback registered: `false`
- Industrial augmentation registered: `false`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- close_mosaic arg: `10`
- close_mosaic official schedule expected: `true`
- close_mosaic expected start epoch: `41`
- close_mosaic log triggered: `false`

## Metrics

- Precision: `0.7262`
- Recall: `0.6844`
- mAP50: `0.7616`
- mAP50-95: `0.5250`

## YOLO Default Augmentation Args

| arg | value |
|---|---:|
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

## Commands

Training:

```text
YOLO.train model=yolo11n.pt data=E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position\data.yaml epochs=50 imgsz=1024 batch=2 workers=0 device=0 seed=42 project=E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\clean_native_yolo_default_seed42_50ep name=train exist_ok=True plots=False trainer=native_ultralytics_default
```

Validation:

```text
YOLO.val model=E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\clean_native_yolo_default_seed42_50ep\train\weights\best.pt data=E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position\data.yaml imgsz=1024 batch=2 workers=0 device=0 project=E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\clean_native_yolo_default_seed42_50ep name=val exist_ok=True plots=False
```
