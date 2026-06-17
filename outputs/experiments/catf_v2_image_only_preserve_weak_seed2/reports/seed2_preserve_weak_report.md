# Seed2 Preserve-Weak Image CATF Report

## Scope

- Run root: `outputs/experiments/catf_v2_image_only_preserve_weak_seed2/`
- Seed: `2`
- Epochs completed: `50`
- Method: image-only preserve-safe-original + weak-only-for-moderate-risk + strict no-op.
- Sampler-only: `false`
- Weighted index list: `false`
- Sampled distribution changed: `false`
- Final val used for policy selection: `false`

## Decision Summary

| action | count |
|---|---:|
| preserve_original | 0 |
| weak_roi_texture | 5 |
| strict_noop | 4 |

The runtime followed the seed2 replay schedule:

| epoch | risk_level | final action | class | selected op | original prob/strength | final prob/strength |
|---:|---|---|---:|---|---|---|
| 5 | critical | strict_noop | 9 | none | n/a | n/a |
| 10 | high | strict_noop | -1 | none | n/a | n/a |
| 15 | high | strict_noop | -1 | none | n/a | n/a |
| 20 | moderate | weak_roi_texture | 7 | local_contrast | 0.18 / 0.20 | 0.045 / 0.05 |
| 25 | moderate | weak_roi_texture | 9 | local_contrast | 0.18 / 0.20 | 0.045 / 0.05 |
| 30 | moderate | weak_roi_texture | 6 | local_contrast | 0.18 / 0.20 | 0.045 / 0.05 |
| 35 | moderate | weak_roi_texture | 9 | local_contrast | 0.18 / 0.20 | 0.045 / 0.05 |
| 40 | high | strict_noop | -1 | none | n/a | n/a |
| 45 | moderate | weak_roi_texture | 9 | local_contrast | 0.18 / 0.20 | 0.045 / 0.05 |

- Attenuation ratio: `0.25`
- High/critical risk candidates were routed to strict no-op.
- Weak image augmentation executed: `true`
- Stale policy accumulation avoided: `true`
- Seed-level union avoided: `true`

## Augmentation Execution

- Industrial images augmented: `80`
- ROI applied: `95`
- ROI affected classes: `{"6": 16, "7": 17, "9": 62}`
- Router random draw count: `1922`
- Op executed: `local_contrast`
- Invalid bbox count: `0`
- BBox out-of-bounds count: `0`
- Class id out-of-bounds count: `0`

## Metrics

| run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean seed2 | 0.696200 | 0.728600 | 0.769200 | 0.522400 |
| fixed CATF-v2 seed2 | 0.763700 | 0.686300 | 0.758200 | 0.496700 |
| weak-only seed2 | 0.753300 | 0.694200 | 0.772700 | 0.515100 |
| preserve-weak seed2 | 0.753254 | 0.694235 | 0.772718 | 0.515138 |

Delta vs clean seed2:

- Precision: `+0.057054`
- Recall: `-0.034365`
- mAP50: `+0.003518`
- mAP50-95: `-0.007262`

Delta vs fixed CATF-v2 seed2:

- Precision: `-0.010446`
- Recall: `+0.007935`
- mAP50: `+0.014518`
- mAP50-95: `+0.018438`

Delta vs weak-only seed2:

- Precision: `-0.000046`
- Recall: `+0.000035`
- mAP50: `+0.000018`
- mAP50-95: `+0.000038`

Constraint result:

- `constraint_failed=false`
- Recall warning: `true` because Recall remains below clean seed2 by `0.034365`.

## Class-Level Checks

Class 9 recovered relative to fixed CATF-v2 seed2:

- Precision: `+0.002076`
- Recall: `+0.108610`
- AP50: `+0.034223`
- AP50-95: `+0.009864`

Non-active regression relative to fixed CATF-v2 seed2 was alleviated under the class set excluding weak-active classes `{6, 7, 9}`:

- Mean non-active AP50 delta: `+0.009761`
- Mean non-active AP50-95 delta: `+0.018671`

## Conclusion

Seed2 completed successfully. The three-stage image-only CATF policy repaired the fixed CATF-v2 seed2 failure without sampler-only, weighted index lists, or any training distribution reweighting. The result is effectively identical to the previous weak-only seed2 repair, while preserving the stricter decision structure needed for seed0. Because seed0 volume-fixed sanity also passed, the next recommended step is seed1 sanity before a final 3-seed summary.
