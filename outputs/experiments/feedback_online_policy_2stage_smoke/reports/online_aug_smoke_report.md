# Feedback Online Augmentation Smoke Report

- Run ID: `feedback_online_policy_2stage_smoke`
- Mode: `only_custom_online_aug`
- Online augmentation implemented: `true`
- Feedback enabled: `true`
- Stage count: `2`
- Policy history updates: `1`
- Train image count: `2301`
- Train image count remains original 2301: `true`
- Fixed augmented dataset generated: `false`
- YOLO built-in augmentation disabled: `true`
- Validation custom augmentation: `false`
- Training success: `true`
- Validation success: `true`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\feedback_online_policy_2stage_smoke\previews`
- Stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\feedback_online_policy_2stage_smoke\reports\online_aug_stats.json`
- Policy history JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\feedback_online_policy_2stage_smoke\reports\policy_history.json`

## Validation Metrics

- Precision: `0.6540`
- Recall: `0.3189`
- mAP50: `0.3338`
- mAP50-95: `0.2071`

## Stage Summary

| stage | epochs | train | val | policy_updated |
|---:|---:|---|---|---|
| 0 | 1 | true | true | true |
| 1 | 1 | true | true | false |

## Operation Counts

| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported | skipped_close_mosaic |
|---|---:|---:|---:|---:|---:|---:|---:|
| blur_mild | 2301 | 0 | 2301 | 0 | 0 | 0 | 0 |
| brightness | 4602 | 1610 | 2992 | 0 | 0 | 0 | 0 |
| clahe | 4602 | 1664 | 2938 | 0 | 0 | 0 | 0 |
| class_balanced_copy_paste | 2301 | 0 | 2165 | 0 | 136 | 0 | 0 |
| contrast | 4602 | 1551 | 3051 | 0 | 0 | 0 | 0 |
| copy_paste | 4602 | 0 | 4602 | 0 | 0 | 0 | 0 |
| cutout_safe | 4602 | 1077 | 3525 | 0 | 0 | 0 | 0 |
| gamma | 4602 | 1571 | 3031 | 0 | 0 | 0 | 0 |
| gaussian_noise | 4602 | 304 | 4298 | 0 | 0 | 0 | 0 |
| horizontal_flip | 4602 | 2225 | 2377 | 0 | 0 | 0 | 0 |
| hsv_jitter | 4602 | 4602 | 0 | 0 | 0 | 0 | 0 |
| local_contrast | 4602 | 1163 | 3439 | 0 | 0 | 0 | 0 |
| mild_scale | 4602 | 965 | 3637 | 0 | 0 | 0 | 0 |
| mild_translate | 4602 | 845 | 3757 | 0 | 0 | 0 | 0 |
| mosaic4 | 4602 | 3560 | 1042 | 0 | 0 | 0 | 0 |
| randaugment_like | 4602 | 2135 | 2467 | 0 | 0 | 0 | 0 |
| random_erasing | 2301 | 147 | 2154 | 0 | 0 | 0 | 0 |
| random_scale_translate | 4602 | 3454 | 1148 | 0 | 0 | 0 | 0 |
| sharpen | 2301 | 166 | 2135 | 0 | 0 | 0 | 0 |
| sharpen_mild | 4602 | 1324 | 3278 | 0 | 0 | 0 | 0 |

## Safety

- Bbox transform valid: `true`
- Invalid bbox count: `0`
- Bbox out-of-bounds count before clipping: `0`
- Class id out-of-range count: `0`
- Cutout holes applied: `3250`
- Cutout skipped by center safety: `0`
- Cutout skipped by overlap safety: `0`
- Mosaic applied: `3560`

## Copy-Paste

- Online copy-paste pending: object-bank paste is not enabled; copy_paste and class_balanced_copy_paste updates are recorded as pending.
