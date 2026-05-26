# Online Augmentation Smoke Report

- Run ID: `diagnosis_recall_safe`
- Mode: `yolo_default_plus_custom_online_aug`
- Online augmentation implemented: `true`
- Train image count: `2301`
- Train image count remains original 2301: `true`
- Fixed augmented dataset generated: `false`
- Policy: `diagnosis_recall_safe_policy`
- Training success: `true`
- Validation success: `true`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\runs\diagnosis_recall_safe\previews`
- Stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\runs\diagnosis_recall_safe\reports\online_aug_stats.json`

## Validation Metrics

- Precision: `0.6712`
- Recall: `0.7659`
- mAP50: `0.7411`
- mAP50-95: `0.5044`

## Operation Counts

| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |
|---|---:|---:|---:|---:|---:|---:|
| brightness | 115050 | 9112 | 105938 | 0 | 0 | 0 |
| clahe | 115050 | 13804 | 101246 | 0 | 0 | 0 |
| contrast | 115050 | 13840 | 101210 | 0 | 0 | 0 |
| gamma | 115050 | 11483 | 103567 | 0 | 0 | 0 |
| random_scale_translate | 115050 | 13662 | 101388 | 0 | 0 | 0 |
| sharpen_mild | 115050 | 11533 | 103517 | 0 | 0 | 0 |

## Safety

- Bbox transform valid: `true`
- Invalid bbox count: `0`
- Bbox out-of-bounds count before clipping: `222`
- Class id out-of-range count: `0`
- Cutout holes applied: `0`
- Cutout skipped by center safety: `0`
- Cutout skipped by overlap safety: `0`
- Cutout skipped by no safe region: `0`

## Copy-Paste

- Online copy-paste pending: object-bank paste is not enabled in this first smoke implementation.
- Current online policy verifies YOLO-like and industrial online operators while leaving copy-paste execution pending.
