# CATF-v2 Class-Aware Smoke Report

## Answers

- CATF-v2 implemented: `true`
- Per-class diagnosis generated: `true`
- Issue attribution generated: `true`
- Policy matrix active: `true`
- Sample-aware routing enabled: `true`
- ROI-aware augmentation enabled: `true`
- Threshold calibration report enabled: `true`
- Reference curve loaded: `true`
- Epoch 5 update happened: `true`
- BBox/class legal: `true`
- Fixed augmented dataset generated: `false`

## Class-Aware State

- Active classes: `[6, 8]`
- Frozen classes: `[]`
- High-FP guarded classes: `[0, 1, 7, 9, 12]`
- Low-contrast classes: `[1, 4, 5, 6, 7, 8, 9, 10, 11, 12]`
- Texture classes: `[6, 7, 11, 6, 7, 11]`
- Low-support only classes: `[2, 3]`
- Stable classes avoided: `false`

## ROI Augmentation

- ROI-aware applied count: `23`
- ROI skipped small ROI: `0`
- ROI skipped conflict: `2`
- Affected classes: `{'6': 20, '8': 3}`

## Required Artifacts

- Per-class diagnosis: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_activation_fixed_10ep_smoke\reports\per_class_diagnosis_epoch_5.json`
- Issue attribution: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_activation_fixed_10ep_smoke\reports\issue_attribution_epoch_5.json`
- Sample weight map: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_activation_fixed_10ep_smoke\reports\sample_weight_map_epoch_5.json`
- Policy history: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_activation_fixed_10ep_smoke\reports\policy_history.json`
- Online aug stats: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_activation_fixed_10ep_smoke\reports\online_aug_stats.json`
- ROI aug stats: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_activation_fixed_10ep_smoke\reports\roi_aug_stats.json`

## Next Step

- This smoke only validates the CATF-v2 control path. It is not a 50 epoch result.
