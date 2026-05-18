# Safe Tiled Baseline 50 Epoch Report

## Run

- run_id: `20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep`
- Dataset: `E:/TJGY/MinPaper/MyAutoAugument/outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Safe tiled dataset: `true`
- Removed `OK` and `定位`: `true`
- New classes: `OK2`, `OK3`, `加强筋打伤`, `开裂`, `油污`, `浅划伤`, `漏背锡`, `碰伤`, `脏污`, `轮廓划伤`, `锡丝残留`, `锡尖`, `锡膏`
- Train tiles: 2301
- Val tiles: 677
- BBoxes: 4087 (`train=3182`, `val=905`)

## Training

- model: `yolo11n.pt`
- epochs: `50`
- imgsz: `1024`
- batch: `2`
- device: `0`
- workers: `0`
- YOLO built-in augmentation switches disabled: `true`
- Disabled switches: `mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0`
- Training time: `2.826` hours (`10173.6` seconds)
- OOM: `false`
- best.pt: `E:/TJGY/MinPaper/MyAutoAugument/outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/best.pt`
- last.pt: `E:/TJGY/MinPaper/MyAutoAugument/outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/last.pt`

## Validation Metrics

- Precision: `0.690`
- Recall: `0.615`
- mAP50: `0.669`
- mAP50-95: `0.434`
- Lowest per-class Recall: `开裂` (`0.000`)

| class id | class | images | instances | Precision | Recall | AP50 | AP50-95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | OK2 | 64 | 64 | 0.971 | 1.000 | 0.995 | 0.841 |
| 1 | OK3 | 274 | 274 | 0.932 | 0.978 | 0.970 | 0.777 |
| 2 | 加强筋打伤 | 8 | 8 | 0.996 | 0.875 | 0.955 | 0.660 |
| 3 | 开裂 | 2 | 2 | 0.000 | 0.000 | 0.111 | 0.092 |
| 4 | 油污 | 14 | 18 | 0.254 | 0.333 | 0.242 | 0.111 |
| 5 | 浅划伤 | 11 | 13 | 1.000 | 0.550 | 0.712 | 0.338 |
| 6 | 漏背锡 | 36 | 46 | 0.577 | 0.712 | 0.694 | 0.325 |
| 7 | 碰伤 | 229 | 269 | 0.681 | 0.617 | 0.617 | 0.368 |
| 8 | 脏污 | 33 | 42 | 0.490 | 0.381 | 0.440 | 0.235 |
| 9 | 轮廓划伤 | 69 | 84 | 0.782 | 0.511 | 0.717 | 0.386 |
| 10 | 锡丝残留 | 12 | 12 | 0.634 | 0.583 | 0.619 | 0.375 |
| 11 | 锡尖 | 25 | 27 | 0.825 | 0.852 | 0.906 | 0.685 |
| 12 | 锡膏 | 41 | 46 | 0.826 | 0.609 | 0.725 | 0.450 |

## Commands

Training:

```powershell
yolo detect train model=yolo11n.pt data=outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml epochs=50 imgsz=1024 batch=2 workers=0 device=0 project=outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep name=train exist_ok=True mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0
```

Validation:

```powershell
yolo detect val model=outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/best.pt data=outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml imgsz=1024 batch=2 workers=0 device=0 project=outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep name=val exist_ok=True
```

## Audit Notes

- Source safe tiling config matched: `tile_size=1024`, `overlap=0.2`, `min_visibility=0.7`, `large_object_min_visibility=0.9`, `drop_border_truncated=True`, `border_margin=2.0`, `require_box_center_inside=True`, `keep_empty_ratio=0.1`.
- Safe source retained bbox quality: `retained_visibility_below_required_count=0`, `retained_border_truncated_count=0`.
- Filtered dataset label audit found no class id outside `0..12`.
- The first in-sandbox train launch failed while Ultralytics created Windows multiprocessing pipes for label caching; it was rerun outside the sandbox with the same YOLO train arguments and completed successfully. This was not CUDA OOM.
