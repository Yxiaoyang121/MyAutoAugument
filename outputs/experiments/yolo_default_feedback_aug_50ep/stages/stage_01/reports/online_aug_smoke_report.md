# Online Augmentation Smoke Report

- Run ID: `stage_01`
- Mode: `yolo_default_plus_custom_online_aug`
- Online augmentation implemented: `true`
- Train image count: `2301`
- Train image count remains original 2301: `true`
- Fixed augmented dataset generated: `false`
- Policy: `yolo_default_feedback_extra_industrial`
- Training success: `true`
- Validation success: `true`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_feedback_aug_50ep\stages\stage_01\previews`
- Stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_feedback_aug_50ep\stages\stage_01\reports\online_aug_stats.json`

## Validation Metrics

- Precision: `0.5259`
- Recall: `0.3311`
- mAP50: `0.3109`
- mAP50-95: `0.1961`

## Operation Counts

| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |
|---|---:|---:|---:|---:|---:|---:|
| clahe | 2301 | 54 | 2247 | 0 | 0 | 0 |
| cutout_safe | 2301 | 89 | 2212 | 0 | 0 | 0 |
| gamma | 2301 | 56 | 2245 | 0 | 0 | 0 |
| local_contrast | 2301 | 57 | 2244 | 0 | 0 | 0 |
| sharpen_mild | 2301 | 111 | 2190 | 0 | 0 | 0 |

## Safety

- Bbox transform valid: `true`
- Invalid bbox count: `0`
- Bbox out-of-bounds count before clipping: `5`
- Class id out-of-range count: `0`
- Cutout holes applied: `178`
- Cutout skipped by center safety: `0`
- Cutout skipped by overlap safety: `0`
- Cutout skipped by no safe region: `0`

## Copy-Paste

- Online copy-paste pending: object-bank paste is not enabled in this first smoke implementation.
- Current online policy verifies YOLO-like and industrial online operators while leaving copy-paste execution pending.
