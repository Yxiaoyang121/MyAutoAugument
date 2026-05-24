# Online vs Offline/Yolo Default/Random Comparison

| run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline no aug | 0.6900 | 0.6150 | 0.6690 | 0.4340 |
| Offline DiagAug | 0.6860 | 0.6880 | 0.7170 | 0.4960 |
| YOLO default | 0.7850 | 0.6760 | 0.7350 | 0.4760 |
| Random external | 0.7500 | 0.6680 | 0.7340 | 0.5010 |
| Online DiagAug | 0.7297 | 0.6772 | 0.6814 | 0.4607 |

## Deltas For Online DiagAug

| reference | dP | dR | d_mAP50 | d_mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline no aug | +0.0397 | +0.0622 | +0.0124 | +0.0267 |
| Offline DiagAug | +0.0437 | -0.0108 | -0.0356 | -0.0353 |
| YOLO default | -0.0553 | +0.0012 | -0.0536 | -0.0153 |
| Random external | -0.0203 | +0.0092 | -0.0526 | -0.0403 |

## Interpretation

- Online DiagAug better than offline DiagAug: `no`
- Online DiagAug close to YOLO default by mAP50-95 within 0.03: `yes`
- Online improves Precision and mAP50-95 vs offline DiagAug: `no`
- Online DiagAug removes the fixed doubled dataset confound and is the fairer mechanism to compare against YOLO default online augmentation.

## Source

- Online metrics JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep\reports\online_aug_stats.json`
