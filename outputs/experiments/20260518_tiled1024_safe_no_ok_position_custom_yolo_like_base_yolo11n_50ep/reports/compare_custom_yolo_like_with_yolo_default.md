# Custom YOLO-Like Base vs YOLO Default Comparison

| run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline no aug | 0.6900 | 0.6150 | 0.6690 | 0.4340 |
| Offline DiagAug | 0.6860 | 0.6880 | 0.7170 | 0.4960 |
| YOLO default | 0.7850 | 0.6760 | 0.7350 | 0.4760 |
| Random external | 0.7500 | 0.6680 | 0.7340 | 0.5010 |
| Online DiagAug | 0.7297 | 0.6772 | 0.6814 | 0.4607 |
| Custom YOLO-like base | 0.6661 | 0.7490 | 0.7129 | 0.4415 |

## Deltas For Custom YOLO-Like Base

| reference | dP | dR | d_mAP50 | d_mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline no aug | -0.0239 | +0.1340 | +0.0439 | +0.0075 |
| Offline DiagAug | -0.0199 | +0.0610 | -0.0041 | -0.0545 |
| YOLO default | -0.1189 | +0.0730 | -0.0221 | -0.0345 |
| Random external | -0.0839 | +0.0810 | -0.0211 | -0.0595 |
| Online DiagAug | -0.0636 | +0.0719 | +0.0315 | -0.0192 |

## Interpretation

- Custom YOLO-like close to Ultralytics YOLO default by mAP50-95 within 0.03: `no`
- Custom YOLO-like better than YOLO default: `no`
- Custom YOLO-like better than baseline: `yes`
- Custom YOLO-like better than Online DiagAug: `no`
- Custom YOLO-like better than offline random external: `no`
- This run evaluates whether the custom YOLO-like operator pool can stand in for Ultralytics default augmentation while keeping augmentation auditable.

## Source

- Online metrics JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep\reports\custom_yolo_like_base_50ep_metrics.json`
- Online augmentation stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep\reports\online_aug_stats.json`
