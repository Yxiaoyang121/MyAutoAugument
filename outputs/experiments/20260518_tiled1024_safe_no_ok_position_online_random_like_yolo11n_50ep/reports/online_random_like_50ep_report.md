# Online Random-Like 50 Epoch Report

- Run ID: `20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep`
- Mode: `only_custom_online_aug`
- Train images: `2301`
- Train images remain 2301: `true`
- Fixed augmented dataset generated: `false`
- Validation custom augmentation: `false`
- YOLO built-in augmentation disabled: `true`
- Copy-paste online supported: `false`

## Final Metrics

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.7132 | 0.6641 | 0.6859 | 0.4661 |

## Online Operation Counts

| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |
|---|---:|---:|---:|---:|---:|---:|
| brightness | 115050 | 78457 | 36593 | 0 | 0 | 0 |
| cutout_safe | 115050 | 17470 | 97580 | 0 | 0 | 0 |
| horizontal_flip | 115050 | 51711 | 63339 | 0 | 0 | 0 |
| sharpen_mild | 115050 | 27585 | 87465 | 0 | 0 | 0 |

## Safety

- Invalid bbox count: `0`
- Bbox out-of-bounds count before clipping: `142`
- Class id out-of-range count: `0`
- Cutout holes applied: `34940`
- Cutout skipped by center safety: `0`
- Cutout skipped by overlap safety: `0`
- Cutout skipped by no safe region: `0`

## Per-Class Recall/AP50

| class_id | name | instances | Precision | Recall | AP50 | AP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | OK2 | 64 | 0.9639 | 1.0000 | 0.9950 | 0.8302 |
| 1 | OK3 | 274 | 0.9046 | 0.9891 | 0.9427 | 0.7717 |
| 2 | 加强筋打伤 | 8 | 0.9642 | 0.8750 | 0.8857 | 0.5758 |
| 3 | 开裂 | 2 | 0.6330 | 1.0000 | 0.6633 | 0.5970 |
| 4 | 油污 | 18 | 0.1923 | 0.2386 | 0.1864 | 0.0960 |
| 5 | 浅划伤 | 13 | 0.6797 | 0.4615 | 0.5073 | 0.2297 |
| 6 | 漏背锡 | 46 | 0.5749 | 0.6304 | 0.6811 | 0.3356 |
| 7 | 碰伤 | 269 | 0.7471 | 0.4941 | 0.5897 | 0.3634 |
| 8 | 脏污 | 42 | 0.5892 | 0.3758 | 0.4187 | 0.2338 |
| 9 | 轮廓划伤 | 84 | 0.7862 | 0.5000 | 0.6574 | 0.3541 |
| 10 | 锡丝残留 | 12 | 0.7561 | 0.5182 | 0.7037 | 0.4406 |
| 11 | 锡尖 | 27 | 0.8064 | 0.9630 | 0.9392 | 0.7301 |
| 12 | 锡膏 | 46 | 0.6745 | 0.5870 | 0.7461 | 0.5012 |

## Comparison Summary

| run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline no aug | 0.6900 | 0.6150 | 0.6690 | 0.4340 |
| Offline DiagAug | 0.6860 | 0.6880 | 0.7170 | 0.4960 |
| YOLO default | 0.7850 | 0.6760 | 0.7350 | 0.4760 |
| Random external | 0.7500 | 0.6680 | 0.7340 | 0.5010 |
| Online DiagAug | 0.7297 | 0.6772 | 0.6814 | 0.4607 |
| Online random-like | 0.7132 | 0.6641 | 0.6859 | 0.4661 |

## Key Answers

- Online random-like better than offline random external: `no`
- Online random-like better than online DiagAug: `yes`
- Online random-like close to YOLO default by mAP50-95 within 0.03: `yes`
- Online random-like is current best by mAP50-95: `no`
- Random external advantage source: `operator_combo_helps_but_offline_doubling_or_training_variance_still_contributes`
- Online random-like removes the fixed doubled dataset confound; if it approaches offline random, the operator mix is the likely driver.

## Artifacts

- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep\train\weights\best.pt`
- last.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep\train\weights\last.pt`
- Stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep\reports\online_aug_stats.json`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep\previews`
