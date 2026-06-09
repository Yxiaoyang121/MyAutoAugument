# Multiseed CP-CATF Paper-Mode Summary

## Split

- Original train images: `2301`
- Train core images: `2071`
- Probe images: `230`
- Final val images: `677`
- No overlap: `true`

## Metrics

| seed | clean P | clean R | clean mAP50 | clean mAP50-95 | CP P | CP R | CP mAP50 | CP mAP50-95 | candidate | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 0.7513 | 0.6763 | 0.7566 | 0.5114 | 0.7513 | 0.6763 | 0.7566 | 0.5114 | `['candidate_policy_3_sampler_only', 'candidate_policy_3_sampler_only', 'candidate_policy_3_sampler_only', 'candidate_policy_3_sampler_only', 'candidate_policy_1_roi_texture', 'candidate_policy_3_sampler_only', 'candidate_policy_3_sampler_only', 'candidate_policy_3_sampler_only', 'candidate_policy_3_sampler_only']` | `false` |
| 1 | 0.7220 | 0.7582 | 0.7777 | 0.5251 | 0.7220 | 0.7582 | 0.7777 | 0.5251 | `['candidate_policy_3_sampler_only', 'candidate_policy_3_sampler_only', 'candidate_policy_3_sampler_only', 'candidate_policy_3_sampler_only', 'candidate_policy_1_roi_texture', 'candidate_policy_3_sampler_only', 'candidate_policy_3_sampler_only', 'candidate_policy_1_roi_texture', 'candidate_policy_3_sampler_only']` | `false` |
| 2 | 0.6290 | 0.6385 | 0.6590 | 0.4381 | 0.6290 | 0.6385 | 0.6590 | 0.4381 | `['candidate_policy_3_sampler_only', 'candidate_policy_3_sampler_only', 'candidate_policy_3_sampler_only', 'candidate_policy_1_roi_texture', 'candidate_policy_1_roi_texture', 'candidate_policy_1_roi_texture', 'candidate_policy_1_roi_texture', 'candidate_policy_3_sampler_only', 'candidate_policy_1_roi_texture']` | `false` |

## Average

- Clean average: `{'precision': 0.700781386555506, 'recall': 0.6910248208682596, 'map50': 0.7311068039293481, 'map50_95': 0.491533946563534}`
- CP-CATF average: `{'precision': 0.700781386555506, 'recall': 0.6910248208682596, 'map50': 0.7311068039293481, 'map50_95': 0.491533946563534}`
- Average delta vs clean: `{'precision': 0.0, 'recall': 0.0, 'map50': 0.0, 'map50_95': 0.0}`
- Constraint failed count: `0/3`
- 3/3 pass: `true`
- Final val leakage detected: `false`
- Recommend as paper main result: `false`

## Notes

- Paper-mode uses `train_core` for training and `probe` for policy selection.
- Final validation is kept for final metrics only.
- If gains drop relative to development-mode, the likely causes are smaller `train_core`, smaller probe evidence, and less optimistic probe diagnostics.
