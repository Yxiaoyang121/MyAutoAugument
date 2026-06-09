# CATF-v2 Class-Aware Smoke Report

## Answers

- CATF-v2 implemented: `true`
- Per-class diagnosis generated: `true`
- Issue attribution generated: `true`
- Policy matrix active: `true`
- Sample-aware routing enabled: `true`
- ROI-aware augmentation enabled: `true`
- Threshold calibration report enabled: `true`
- CATF-v2 safe mode enabled: `false`
- CATF-v2 gated mode enabled: `false`
- CATF-v2 rollback mode enabled: `false`
- Adaptive burn-in enabled: `false`
- Adaptive start epoch: `None`
- Baseline protection triggered: `false`
- Early abstention triggered: `false`
- No-op freeze entered: `false`
- Gated no-op freeze entered: `false`
- Reference curve loaded: `true`
- Epoch 5 update happened: `true`
- BBox/class legal: `true`
- Fixed augmented dataset generated: `false`

## Class-Aware State

- Active classes: `[]`
- Frozen classes: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`
- High-FP guarded classes: `[1, 6, 7, 11]`
- Low-contrast classes: `[1, 4, 5, 6, 7, 8, 9, 12]`
- Texture classes: `[9, 9]`
- Low-support only classes: `[2, 3, 10]`
- Stable classes avoided: `true`

## ROI Augmentation

- ROI-aware applied count: `0`
- ROI skipped small ROI: `0`
- ROI skipped conflict: `0`
- Affected classes: `{}`

## Required Artifacts

- Per-class diagnosis: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_10ep_smoke\reports\per_class_diagnosis_epoch_5.json`
- Issue attribution: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_10ep_smoke\reports\issue_attribution_epoch_5.json`
- Sample weight map: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_10ep_smoke\reports\sample_weight_map_epoch_5.json`
- Policy history: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_10ep_smoke\reports\policy_history.json`
- Online aug stats: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_10ep_smoke\reports\online_aug_stats.json`
- ROI aug stats: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_10ep_smoke\reports\roi_aug_stats.json`
- Safe controller events: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_10ep_smoke\reports\safe_controller_events.json`
- Gated controller events: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_10ep_smoke\reports\gated_controller_events.json`
- Adaptive burn-in events: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_10ep_smoke\reports\adaptive_burnin_events.json`
- Rollback controller events: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_10ep_smoke\reports\rollback_controller_events.json`

## Next Step

- This smoke only validates the CATF-v2 control path. It is not a 50 epoch result.
