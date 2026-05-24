# Online Augmentation Smoke Report

- Run ID: `20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_smoke`
- Mode: `only_custom_online_aug`
- Online augmentation implemented: `true`
- Train image count: `2301`
- Train image count remains original 2301: `true`
- Fixed augmented dataset generated: `false`
- Policy: `diag_policy_001`
- Training success: `true`
- Validation success: `true`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_smoke\previews`
- Stats JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_smoke\reports\online_aug_stats.json`

## Validation Metrics

- Precision: `0.8099`
- Recall: `0.2518`
- mAP50: `0.2721`
- mAP50-95: `0.1789`

## Operation Counts

| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |
|---|---:|---:|---:|---:|---:|---:|
| brightness | 2301 | 523 | 1778 | 0 | 0 | 0 |
| clahe | 2301 | 986 | 1315 | 0 | 0 | 0 |
| contrast | 2301 | 923 | 1378 | 0 | 0 | 0 |
| gamma | 2301 | 759 | 1542 | 0 | 0 | 0 |

## Safety

- Bbox transform valid: `true`
- Invalid bbox count: `0`
- Bbox out-of-bounds count before clipping: `5`
- Class id out-of-range count: `0`
- Cutout holes applied: `0`
- Cutout skipped by center safety: `0`
- Cutout skipped by overlap safety: `0`

## Copy-Paste

- Online copy-paste pending: object-bank paste is not enabled in this first smoke implementation.
- Current online policy verifies photometric / texture / cutout / flip / mild geometry operations.
