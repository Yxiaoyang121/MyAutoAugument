# Custom YOLO-Like Base 50 Epoch Report

- Run ID: `20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep`
- Mode: `only_custom_online_aug`
- Train images: `2301`
- Train images remain 2301: `true`
- Fixed augmented dataset generated: `false`
- Validation custom augmentation: `false`
- YOLO built-in augmentation disabled: `true`
- Copy-paste online supported: `false`
- Feedback applied: `false`
- Policy history: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep\reports\policy_history.json`

## Final Metrics

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.6661 | 0.7490 | 0.7129 | 0.4415 |

## Online Operation Counts

| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |
|---|---:|---:|---:|---:|---:|---:|
| horizontal_flip | 115050 | 57654 | 57396 | 0 | 0 | 0 |
| hsv_jitter | 115050 | 115050 | 0 | 0 | 0 | 0 |
| mosaic4 | 115050 | 92040 | 0 | 0 | 0 | 0 |
| randaugment_like | 115050 | 51325 | 63725 | 0 | 0 | 0 |
| random_erasing | 115050 | 46039 | 69011 | 0 | 0 | 0 |
| random_scale_translate | 115050 | 115050 | 0 | 0 | 0 | 0 |

## Safety

- Invalid bbox count: `0`
- Bbox out-of-bounds count before clipping: `0`
- Class id out-of-range count: `0`
- Cutout holes applied: `111234`
- Cutout skipped by center safety: `0`
- Cutout skipped by overlap safety: `0`
- Cutout skipped by no safe region: `0`
- Mosaic applied: `92040`
- Mosaic skipped by close_mosaic: `23010`
- close_mosaic active: `true`

## Per-Class Recall/AP50

| class_id | name | instances | Precision | Recall | AP50 | AP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | OK2 | 64 | 0.9349 | 0.9844 | 0.9912 | 0.8286 |
| 1 | OK3 | 274 | 0.9612 | 0.9947 | 0.9851 | 0.8037 |
| 2 | 加强筋打伤 | 8 | 0.7159 | 0.8750 | 0.8895 | 0.4869 |
| 3 | 开裂 | 2 | 0.2778 | 1.0000 | 0.2843 | 0.1808 |
| 4 | 油污 | 18 | 0.1853 | 0.3889 | 0.1906 | 0.1163 |
| 5 | 浅划伤 | 13 | 0.5657 | 0.6154 | 0.6365 | 0.2759 |
| 6 | 漏背锡 | 46 | 0.6971 | 0.7004 | 0.7582 | 0.3828 |
| 7 | 碰伤 | 269 | 0.6730 | 0.6877 | 0.7062 | 0.4284 |
| 8 | 脏污 | 42 | 0.6740 | 0.5714 | 0.5753 | 0.2708 |
| 9 | 轮廓划伤 | 84 | 0.7292 | 0.5000 | 0.6653 | 0.3304 |
| 10 | 锡丝残留 | 12 | 0.8173 | 0.7457 | 0.8761 | 0.5697 |
| 11 | 锡尖 | 27 | 0.8128 | 1.0000 | 0.9867 | 0.6266 |
| 12 | 锡膏 | 46 | 0.6151 | 0.6739 | 0.7232 | 0.4383 |

## Comparison Summary

| run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline no aug | 0.6900 | 0.6150 | 0.6690 | 0.4340 |
| Offline DiagAug | 0.6860 | 0.6880 | 0.7170 | 0.4960 |
| YOLO default | 0.7850 | 0.6760 | 0.7350 | 0.4760 |
| Random external | 0.7500 | 0.6680 | 0.7340 | 0.5010 |
| Online DiagAug | 0.7297 | 0.6772 | 0.6814 | 0.4607 |
| Custom YOLO-like base | 0.6661 | 0.7490 | 0.7129 | 0.4415 |

## Key Answers

- Custom YOLO-like close to Ultralytics YOLO default by mAP50-95 within 0.03: `no`
- Custom YOLO-like better than YOLO default: `no`
- Custom YOLO-like better than baseline: `yes`
- Custom YOLO-like better than Online DiagAug: `no`
- Custom YOLO-like better than offline random external: `no`
- This run isolates the custom implementation of YOLO-like online operators from Ultralytics built-in augmentation.

## Artifacts

- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep\train\weights\best.pt`
- last.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep\train\weights\last.pt`
- Stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep\reports\online_aug_stats.json`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep\previews`
