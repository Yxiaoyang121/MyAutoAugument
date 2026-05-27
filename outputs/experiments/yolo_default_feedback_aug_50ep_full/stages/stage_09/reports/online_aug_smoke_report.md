# Online Augmentation Smoke Report

- Run ID: `stage_09`
- Mode: `yolo_default_plus_custom_online_aug`
- Online augmentation implemented: `true`
- Train image count: `2301`
- Train image count remains original 2301: `true`
- Fixed augmented dataset generated: `false`
- Policy: `yolo_default_feedback_extra_industrial`
- Training success: `true`
- Validation success: `true`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_feedback_aug_50ep_full\stages\stage_09\previews`
- Stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_feedback_aug_50ep_full\stages\stage_09\reports\online_aug_stats.json`

## Validation Metrics

- Precision: `0.7712`
- Recall: `0.6689`
- mAP50: `0.7439`
- mAP50-95: `0.4993`

## Operation Counts

| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |
|---|---:|---:|---:|---:|---:|---:|
| clahe | 11505 | 2681 | 8824 | 0 | 0 | 0 |
| cutout_safe | 11505 | 2780 | 8725 | 0 | 0 | 0 |
| gamma | 11505 | 2804 | 8701 | 0 | 0 | 0 |
| local_contrast | 11505 | 3438 | 8067 | 0 | 0 | 0 |
| sharpen_mild | 11505 | 3417 | 8088 | 0 | 0 | 0 |

## Safety

- Bbox transform valid: `true`
- Invalid bbox count: `0`
- Bbox out-of-bounds count before clipping: `25`
- Class id out-of-range count: `0`
- Cutout holes applied: `5560`
- Cutout skipped by center safety: `0`
- Cutout skipped by overlap safety: `0`
- Cutout skipped by no safe region: `0`

## Copy-Paste

- Online copy-paste pending: object-bank paste is not enabled in this first smoke implementation.
- Current online policy verifies YOLO-like and industrial online operators while leaving copy-paste execution pending.
