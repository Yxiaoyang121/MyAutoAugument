# Online Random-Like vs All Comparison

| run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline no aug | 0.6900 | 0.6150 | 0.6690 | 0.4340 |
| Offline DiagAug | 0.6860 | 0.6880 | 0.7170 | 0.4960 |
| YOLO default | 0.7850 | 0.6760 | 0.7350 | 0.4760 |
| Random external | 0.7500 | 0.6680 | 0.7340 | 0.5010 |
| Online DiagAug | 0.7297 | 0.6772 | 0.6814 | 0.4607 |
| Online random-like | 0.7132 | 0.6641 | 0.6859 | 0.4661 |

## Deltas For Online Random-Like

| reference | dP | dR | d_mAP50 | d_mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline no aug | +0.0232 | +0.0491 | +0.0169 | +0.0321 |
| Offline DiagAug | +0.0272 | -0.0239 | -0.0311 | -0.0299 |
| YOLO default | -0.0718 | -0.0119 | -0.0491 | -0.0099 |
| Random external | -0.0368 | -0.0039 | -0.0481 | -0.0349 |
| Online DiagAug | -0.0165 | -0.0131 | +0.0044 | +0.0054 |

## Interpretation

- Online random-like better than offline random external: `no`
- Online random-like better than online DiagAug: `yes`
- Online random-like is current best by mAP50-95: `no`
- Random external advantage source: `operator_combo_helps_but_offline_doubling_or_training_variance_still_contributes`
- This run isolates the random-like operator mix from offline train-set doubling.

## Source

- Online metrics JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep\reports\online_random_like_metrics.json`
- Online augmentation stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep\reports\online_aug_stats.json`
