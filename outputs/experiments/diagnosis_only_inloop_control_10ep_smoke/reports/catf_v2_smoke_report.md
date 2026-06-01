# CATF-v2 Class-Aware Smoke Report

## Answers

- CATF-v2 implemented: `true`
- Per-class diagnosis generated: `true`
- Issue attribution generated: `true`
- Policy matrix active: `true`
- Sample-aware routing enabled: `false`
- ROI-aware augmentation enabled: `false`
- Threshold calibration report enabled: `true`
- Reference curve loaded: `true`
- Epoch 5 update happened: `true`
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

- Per-class diagnosis: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\diagnosis_only_inloop_control_10ep_smoke\reports\per_class_diagnosis_epoch_5.json`
- Issue attribution: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\diagnosis_only_inloop_control_10ep_smoke\reports\issue_attribution_epoch_5.json`
- Sample weight map: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\diagnosis_only_inloop_control_10ep_smoke\reports\sample_weight_map_epoch_5.json`
- Policy history: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\diagnosis_only_inloop_control_10ep_smoke\reports\policy_history.json`
- Online aug stats: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\diagnosis_only_inloop_control_10ep_smoke\reports\online_aug_stats.json`
- ROI aug stats: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\diagnosis_only_inloop_control_10ep_smoke\reports\roi_aug_stats.json`

## Next Step

- This smoke only validates the CATF-v2 control path. It is not a 50 epoch result.
