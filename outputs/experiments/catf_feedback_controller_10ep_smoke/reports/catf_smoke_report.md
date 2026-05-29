# CATF Feedback Controller Smoke Report

## Answers

- CATF enabled: `true`
- Reference curve loaded: `true`
- Reference curve path: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\clean_native_yolo_default_seed42_50ep\train\results.csv`
- Epoch 5 update happened: `true`
- Trust-region active: `true`
- Group budget active: `true`
- Proposed policy recorded: `true`
- Accepted policy recorded: `true`
- Guard triggered: `['precision_guard', 'low_contrast_fn']`
- Rollback triggered: `false`
- Freeze triggered: `false`
- BBox/class legal: `true`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`

## Policy History

| epoch | action | guards | adjustments | frozen |
|---:|---|---|---:|---|
| 5 | accept | precision_guard,low_contrast_fn | 10 | false |

## Next Step

- The smoke is sufficient for a multi-seed 50ep CATF recheck only if training succeeded, bbox/class remained legal, and the first update recorded proposed/accepted policies with trust-region and budget evidence.
