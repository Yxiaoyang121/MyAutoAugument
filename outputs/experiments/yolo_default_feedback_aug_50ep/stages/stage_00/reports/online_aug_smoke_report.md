# Online Augmentation Smoke Report

- Run ID: `stage_00`
- Mode: `yolo_default_plus_custom_online_aug`
- Online augmentation implemented: `true`
- Train image count: `2301`
- Train image count remains original 2301: `true`
- Fixed augmented dataset generated: `false`
- Policy: `yolo_default_feedback_extra_industrial`
- Training success: `true`
- Validation success: `true`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_feedback_aug_50ep\stages\stage_00\previews`
- Stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_feedback_aug_50ep\stages\stage_00\reports\online_aug_stats.json`

## Validation Metrics

- Precision: `0.5863`
- Recall: `0.1715`
- mAP50: `0.1771`
- mAP50-95: `0.1157`

## Operation Counts

| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |
|---|---:|---:|---:|---:|---:|---:|

## Safety

- Bbox transform valid: `true`
- Invalid bbox count: `0`
- Bbox out-of-bounds count before clipping: `5`
- Class id out-of-range count: `0`
- Cutout holes applied: `0`
- Cutout skipped by center safety: `0`
- Cutout skipped by overlap safety: `0`
- Cutout skipped by no safe region: `0`

## Copy-Paste

- Online copy-paste pending: object-bank paste is not enabled in this first smoke implementation.
- Current online policy verifies YOLO-like and industrial online operators while leaving copy-paste execution pending.
