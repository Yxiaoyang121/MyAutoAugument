# Tiled Baseline 20 Epoch Report

## Status

- 20 epoch training completed: True
- Final train batch: 2
- Final val batch: 2
- CUDA OOM occurred: False
- Conda wrapper stdout encoding error after train: True

## Environment

- Conda env: pytorch
- sys.executable: D:\Anaconda\envs\pytorch\python.exe
- Torch: 2.4.1
- CUDA available: True
- CUDA version: 12.4
- Device count: 1
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Ultralytics: 8.3.221

## Dataset And Parameters

- Dataset root: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_smoke
- data.yaml: E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_smoke\data.yaml
- Model: yolo11n.pt
- Epochs: 20
- imgsz: 1024
- batch: 2
- workers: 0
- device: 0
- YOLO built-in augmentations disabled: mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0

## Outputs

- best.pt: E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260517_tiled_baseline_20epoch\train\weights\best.pt
- last.pt: E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260517_tiled_baseline_20epoch\train\weights\last.pt
- Train command: E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260517_tiled_baseline_20epoch\train_command.txt
- Val command: E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260517_tiled_baseline_20epoch\val_command.txt

## Metrics

- Precision: 0.828
- Recall: 0.213
- mAP50: 0.247
- mAP50-95: 0.181
- Train duration seconds: 312.4
- Validation duration seconds: 20.137

## Per-Class Metrics Parsed From Val Log

| order | class | images | instances | precision | recall | AP50 | AP50-95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | OK | 21 | 21 | 0.951 | 0.857 | 0.964 | 0.841 |
| 2 | OK3 | 40 | 40 | 0.939 | 0.774 | 0.919 | 0.626 |
| 3 | blank-or-unrendered | 2 | 2 | 1 | 0 | 0.00147 | 0.00105 |
| 4 | blank-or-unrendered | 11 | 21 | 0.563 | 0.286 | 0.289 | 0.145 |
| 5 | blank-or-unrendered | 6 | 6 | 0 | 0 | 0.0255 | 0.0106 |
| 6 | blank-or-unrendered | 19 | 29 | 1 | 0 | 0.00235 | 0.0013 |
| 7 | blank-or-unrendered | 5 | 9 | 1 | 0 | 0 | 0 |
| 8 | blank-or-unrendered | 3 | 3 | 1 | 0 | 0.0101 | 0.00404 |
| 9 | blank-or-unrendered | 10 | 12 | 1 | 0 | 0.00873 | 0.00367 |

## Notes

- Training used conda env pytorch; base was not used.
- CPU was not used for this formal run.
- The non-empty train stderr is a conda stdout forwarding UnicodeEncodeError after YOLO completed; training artifacts and epoch 20 metrics were produced successfully.
