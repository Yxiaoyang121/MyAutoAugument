# Online Augmentation Smoke Report

- Run ID: `diagnosis_precision_safe`
- Mode: `yolo_default_plus_custom_online_aug`
- Online augmentation implemented: `true`
- Train image count: `2301`
- Train image count remains original 2301: `true`
- Fixed augmented dataset generated: `false`
- Policy: `diagnosis_precision_safe_policy`
- Training success: `true`
- Validation success: `true`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\runs\diagnosis_precision_safe\previews`
- Stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\runs\diagnosis_precision_safe\reports\online_aug_stats.json`

## Validation Metrics

- Precision: `0.6647`
- Recall: `0.7493`
- mAP50: `0.7557`
- mAP50-95: `0.5055`

## Operation Counts

| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |
|---|---:|---:|---:|---:|---:|---:|
| hsv_jitter | 115050 | 11504 | 103546 | 0 | 0 | 0 |
| mosaic4 | 115050 | 2707 | 89333 | 0 | 0 | 0 |
| random_erasing | 115050 | 3434 | 111616 | 0 | 0 | 0 |
| random_scale_translate | 115050 | 9160 | 105890 | 0 | 0 | 0 |

## Safety

- Bbox transform valid: `true`
- Invalid bbox count: `0`
- Bbox out-of-bounds count before clipping: `230`
- Class id out-of-range count: `0`
- Cutout holes applied: `3434`
- Cutout skipped by center safety: `0`
- Cutout skipped by overlap safety: `0`
- Cutout skipped by no safe region: `0`

## Copy-Paste

- Online copy-paste pending: object-bank paste is not enabled in this first smoke implementation.
- Current online policy verifies YOLO-like and industrial online operators while leaving copy-paste execution pending.
