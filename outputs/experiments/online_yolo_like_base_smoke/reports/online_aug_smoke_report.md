# Online Augmentation Smoke Report

- Run ID: `online_yolo_like_base_smoke`
- Mode: `only_custom_online_aug`
- Online augmentation implemented: `true`
- Train image count: `2301`
- Train image count remains original 2301: `true`
- Fixed augmented dataset generated: `false`
- Policy: `yolo_like_base_policy`
- Training success: `true`
- Validation success: `true`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\online_yolo_like_base_smoke\previews`
- Stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\online_yolo_like_base_smoke\reports\online_aug_stats.json`

## Validation Metrics

- Precision: `0.4922`
- Recall: `0.2364`
- mAP50: `0.1763`
- mAP50-95: `0.0986`

## Operation Counts

| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |
|---|---:|---:|---:|---:|---:|---:|
| horizontal_flip | 2301 | 1076 | 1225 | 0 | 0 | 0 |
| hsv_jitter | 2301 | 2301 | 0 | 0 | 0 | 0 |
| mosaic4 | 2301 | 2301 | 0 | 0 | 0 | 0 |
| random_erasing | 2301 | 931 | 1370 | 0 | 0 | 0 |
| random_scale_translate | 2301 | 2301 | 0 | 0 | 0 | 0 |

## Safety

- Bbox transform valid: `true`
- Invalid bbox count: `0`
- Bbox out-of-bounds count before clipping: `0`
- Class id out-of-range count: `0`
- Cutout holes applied: `1862`
- Cutout skipped by center safety: `0`
- Cutout skipped by overlap safety: `0`
- Cutout skipped by no safe region: `0`

## Copy-Paste

- Online copy-paste pending: object-bank paste is not enabled in this first smoke implementation.
- Current online policy verifies YOLO-like and industrial online operators while leaving copy-paste execution pending.
