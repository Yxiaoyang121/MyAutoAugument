# CATF-v2 Class-Aware Smoke Report

## Answers

- CATF-v2 implemented: `true`
- Per-class diagnosis generated: `false`
- Issue attribution generated: `false`
- Policy matrix active: `true`
- Sample-aware routing enabled: `true`
- ROI-aware augmentation enabled: `true`
- Threshold calibration report enabled: `true`
- Reference curve loaded: `true`
- Epoch 5 update happened: `false`
- BBox/class legal: `true`
- Fixed augmented dataset generated: `false`

## Class-Aware State

- Active classes: `[]`
- Frozen classes: `[0, 1]`
- High-FP guarded classes: `[]`
- Low-contrast classes: `[]`
- Texture classes: `[]`
- Low-support only classes: `[]`
- Stable classes avoided: `true`

## ROI Augmentation

- ROI-aware applied count: `0`
- ROI skipped small ROI: `0`
- ROI skipped conflict: `0`
- Affected classes: `{}`

## Required Artifacts

- Per-class diagnosis: `None`
- Issue attribution: `None`
- Sample weight map: `None`
- Policy history: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_noop_parity_smoke\catf_v2_noop_1ep\reports\policy_history.json`
- Online aug stats: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_noop_parity_smoke\catf_v2_noop_1ep\reports\online_aug_stats.json`
- ROI aug stats: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_noop_parity_smoke\catf_v2_noop_1ep\reports\roi_aug_stats.json`

## Next Step

- This smoke only validates the CATF-v2 control path. It is not a 50 epoch result.
