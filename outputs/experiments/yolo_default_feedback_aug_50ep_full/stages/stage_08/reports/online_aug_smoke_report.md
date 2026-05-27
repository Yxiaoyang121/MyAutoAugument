# Online Augmentation Smoke Report

- Run ID: `stage_08`
- Mode: `yolo_default_plus_custom_online_aug`
- Online augmentation implemented: `true`
- Train image count: `2301`
- Train image count remains original 2301: `true`
- Fixed augmented dataset generated: `false`
- Policy: `yolo_default_feedback_extra_industrial`
- Training success: `true`
- Validation success: `true`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_feedback_aug_50ep_full\stages\stage_08\previews`
- Stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_feedback_aug_50ep_full\stages\stage_08\reports\online_aug_stats.json`

## Validation Metrics

- Precision: `0.6764`
- Recall: `0.6996`
- mAP50: `0.7255`
- mAP50-95: `0.4825`

## Operation Counts

| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |
|---|---:|---:|---:|---:|---:|---:|
| clahe | 11505 | 2559 | 8946 | 0 | 0 | 0 |
| cutout_safe | 11505 | 2824 | 8681 | 0 | 0 | 0 |
| gamma | 11505 | 2713 | 8792 | 0 | 0 | 0 |
| local_contrast | 11505 | 3435 | 8070 | 0 | 0 | 0 |
| sharpen_mild | 11505 | 3411 | 8094 | 0 | 0 | 0 |

## Safety

- Bbox transform valid: `true`
- Invalid bbox count: `0`
- Bbox out-of-bounds count before clipping: `25`
- Class id out-of-range count: `0`
- Cutout holes applied: `5648`
- Cutout skipped by center safety: `0`
- Cutout skipped by overlap safety: `0`
- Cutout skipped by no safe region: `0`

## Copy-Paste

- Online copy-paste pending: object-bank paste is not enabled in this first smoke implementation.
- Current online policy verifies YOLO-like and industrial online operators while leaving copy-paste execution pending.
