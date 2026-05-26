# Online Augmentation Smoke Report

- Run ID: `diagnosis_light`
- Mode: `yolo_default_plus_custom_online_aug`
- Online augmentation implemented: `true`
- Train image count: `2301`
- Train image count remains original 2301: `true`
- Fixed augmented dataset generated: `false`
- Policy: `diagnosis_light_policy`
- Training success: `true`
- Validation success: `true`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\runs\diagnosis_light\previews`
- Stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\runs\diagnosis_light\reports\online_aug_stats.json`

## Validation Metrics

- Precision: `0.7343`
- Recall: `0.6862`
- mAP50: `0.7555`
- mAP50-95: `0.4993`

## Operation Counts

| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |
|---|---:|---:|---:|---:|---:|---:|
| hsv_jitter | 115050 | 17396 | 97654 | 0 | 0 | 0 |
| random_scale_translate | 115050 | 11278 | 103772 | 0 | 0 | 0 |

## Safety

- Bbox transform valid: `true`
- Invalid bbox count: `0`
- Bbox out-of-bounds count before clipping: `228`
- Class id out-of-range count: `0`
- Cutout holes applied: `0`
- Cutout skipped by center safety: `0`
- Cutout skipped by overlap safety: `0`
- Cutout skipped by no safe region: `0`

## Copy-Paste

- Online copy-paste pending: object-bank paste is not enabled in this first smoke implementation.
- Current online policy verifies YOLO-like and industrial online operators while leaving copy-paste execution pending.
