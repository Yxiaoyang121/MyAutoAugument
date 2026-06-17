# Seed0 Preserve-Weak Sanity Rerun

## Run Status

- Completed 50ep: `true`
- preserve_original / weak / noop: `9/0/0`
- Expected class union: `[4, 11, 12]`
- Runtime policy_matrix union: `[4, 11, 12]`
- Sample router eligible union: `[4, 11, 12]`
- Final executable union: `[4, 11, 12]`
- Contains class4/11/12: `true`
- Weak class9 replacement: `false`

## Augmentation

- Image augmentation executed: `true`
- Industrial images augmented: `155`
- ROI applied: `187`
- ROI affected classes: `{'4': 66, '11': 22, '12': 99}`
- Fixed CATF-v2 seed0 reference: `{'industrial': 41, 'roi': 45, 'roi_by_class': {'4': 9, '11': 22, '12': 14}}`
- Fixed application volume close: `false`
- Router random draw count: `16910`

## Sampling

- sampler_only_enabled: `false`
- sampler_only_effective: `false`
- weighted_index_list_enabled: `false`
- sampled_distribution_changed: `false`

## Metrics

| run | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| clean seed0 | 0.784600 | 0.676500 | 0.734700 | 0.475900 |
| fixed CATF-v2 seed0 | 0.778500 | 0.669700 | 0.743700 | 0.489500 |
| pre-fix preserve seed0 | 0.666585 | 0.730019 | 0.699934 | 0.466744 |
| rerun preserve seed0 | 0.700514 | 0.623545 | 0.712481 | 0.456610 |

## Delta

| comparison | dP | dR | dmAP50 | dmAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| vs clean | -0.084086 | -0.052955 | -0.022219 | -0.019290 |
| vs fixed | -0.077986 | -0.046155 | -0.031219 | -0.032890 |
| vs pre-fix preserve | +0.033930 | -0.106474 | +0.012547 | -0.010134 |

## Verdict

- constraint_failed: `true`
- Failure reasons: `['precision_drop_gt_0.01', 'map50_drop_gt_0.01', 'map50_95_drop_gt_0.01']`
- Interpretation: `class_union_execution_parity_fixed_but_fixed_policy_lifetime_or_application_volume_not_reproduced`
- Recommendation: Do not run seed2 yet; audit preserve policy lifetime/application-volume parity against fixed CATF-v2.
