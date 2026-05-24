# Online Diag Policy 001 50 Epoch Report

- Run ID: `20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep`
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
| 0.7297 | 0.6772 | 0.6814 | 0.4607 |

## Online Operation Counts

| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |
|---|---:|---:|---:|---:|---:|---:|
| brightness | 115050 | 26756 | 88294 | 0 | 0 | 0 |
| clahe | 115050 | 47089 | 67961 | 0 | 0 | 0 |
| contrast | 115050 | 46790 | 68260 | 0 | 0 | 0 |
| gamma | 115050 | 37253 | 77797 | 0 | 0 | 0 |

## Safety

- Invalid bbox count: `0`
- Bbox out-of-bounds count before clipping: `250`
- Class id out-of-range count: `0`
- Cutout holes applied: `0`
- Cutout skipped by center safety: `0`
- Cutout skipped by overlap safety: `0`

## Per-Class Recall/AP50

| class_id | name | instances | Precision | Recall | AP50 | AP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | OK2 | 64 | 0.9588 | 1.0000 | 0.9948 | 0.8419 |
| 1 | OK3 | 274 | 0.9351 | 0.9996 | 0.9680 | 0.7941 |
| 2 | 加强筋打伤 | 8 | 1.0000 | 0.8662 | 0.9007 | 0.5655 |
| 3 | 开裂 | 2 | 0.2802 | 1.0000 | 0.3904 | 0.3513 |
| 4 | 油污 | 18 | 0.3539 | 0.3333 | 0.2277 | 0.1249 |
| 5 | 浅划伤 | 13 | 0.8742 | 0.5350 | 0.6799 | 0.3414 |
| 6 | 漏背锡 | 46 | 0.7600 | 0.6957 | 0.7400 | 0.4574 |
| 7 | 碰伤 | 269 | 0.7161 | 0.5428 | 0.6172 | 0.3726 |
| 8 | 脏污 | 42 | 0.5468 | 0.3810 | 0.3972 | 0.2214 |
| 9 | 轮廓划伤 | 84 | 0.7887 | 0.5333 | 0.7194 | 0.4164 |
| 10 | 锡丝残留 | 12 | 0.6200 | 0.4089 | 0.5168 | 0.3200 |
| 11 | 锡尖 | 27 | 0.8910 | 0.8148 | 0.9298 | 0.6935 |
| 12 | 锡膏 | 46 | 0.7611 | 0.6928 | 0.7768 | 0.4880 |

## Comparison Summary

| run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline no aug | 0.6900 | 0.6150 | 0.6690 | 0.4340 |
| Offline DiagAug | 0.6860 | 0.6880 | 0.7170 | 0.4960 |
| YOLO default | 0.7850 | 0.6760 | 0.7350 | 0.4760 |
| Random external | 0.7500 | 0.6680 | 0.7340 | 0.5010 |
| Online DiagAug | 0.7297 | 0.6772 | 0.6814 | 0.4607 |

## Key Answers

- Online DiagAug better than offline DiagAug: `no`
- Online DiagAug close to YOLO default by mAP50-95 within 0.03: `yes`
- Online improves Precision and mAP50-95 vs offline DiagAug: `no`
- Online mechanism is more methodologically reasonable than fixed offline doubling because the train image count stays unchanged and policy randomness is sampled per epoch/sample in the dataloader.

## Artifacts

- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep\train\weights\best.pt`
- last.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep\train\weights\last.pt`
- Stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep\reports\online_aug_stats.json`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep\previews`
