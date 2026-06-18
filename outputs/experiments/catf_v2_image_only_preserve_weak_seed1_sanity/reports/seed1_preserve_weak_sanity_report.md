# Seed1 Preserve-Weak Image CATF Sanity Report

## Scope

- Run root: `outputs/experiments/catf_v2_image_only_preserve_weak_seed1_sanity/`
- Seed: `1`
- Epochs completed: `50`
- Method: image-only preserve-safe-original + weak-only-for-moderate-risk + strict no-op.
- Sampler-only: `false`
- Weighted index list: `false`
- Sampled distribution changed: `false`
- Final val used for policy selection: `false`

## Decision Summary

| action | count |
|---|---:|
| preserve_original | 9 |
| weak_roi_texture | 0 |
| strict_noop | 0 |

All feedback epochs used the epoch-exact fixed CATF-v2 preserve path. No weak candidate replaced the fixed policy, and no sampler-only path was involved.

| epoch | final action | executable ops |
|---:|---|---|
| 5 | preserve_original | class4 local_contrast/sharpen_mild; class11 local_contrast/sharpen_mild |
| 10 | preserve_original | class11 local_contrast/sharpen_mild |
| 15 | preserve_original | class11 local_contrast/sharpen_mild |
| 20 | preserve_original | empty executable policy |
| 25 | preserve_original | class11 local_contrast/sharpen_mild |
| 30 | preserve_original | empty executable policy |
| 35 | preserve_original | class11 local_contrast/sharpen_mild |
| 40 | preserve_original | empty executable policy |
| 45 | preserve_original | empty executable policy |

Execution checks:

- epoch-exact fixed policy used: `true`
- stale ops cleared on empty executable epochs: `true`
- seed-level union avoided: `true`
- weak replacement of fixed policy avoided: `true`

## Augmentation Execution

- Image augmentation executed: `true`
- Industrial images augmented: `78`
- ROI applied: `80`
- ROI affected classes: `{"4": 5, "11": 75}`
- Router random draw count: `6420`
- Ops:
  - `local_contrast`: applied `42`
  - `sharpen_mild`: applied `37`
- Invalid bbox count: `0`
- BBox out-of-bounds count: `0`
- Class id out-of-bounds count: `0`

Fixed CATF-v2 seed1 reference volume:

- Industrial images augmented: `55`
- ROI applied: `56`
- ROI affected classes: `{"4": 5, "11": 51}`

Volume parity note:

- The preserve seed1 run used the same fixed classes and ops, but the realized augmentation volume was higher than fixed seed1: industrial `78` vs `55`, ROI `80` vs `56`.
- This is not an exact volume reproduction, but it did not produce a constraint failure.

## Metrics

| run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean seed1 | 0.772500 | 0.647700 | 0.754200 | 0.479900 |
| fixed CATF-v2 seed1 | 0.785200 | 0.700500 | 0.782600 | 0.518900 |
| preserve-weak seed1 | 0.799748 | 0.697375 | 0.778737 | 0.516923 |

Delta vs clean seed1:

- Precision: `+0.027248`
- Recall: `+0.049675`
- mAP50: `+0.024537`
- mAP50-95: `+0.037023`

Delta vs fixed CATF-v2 seed1:

- Precision: `+0.014548`
- Recall: `-0.003125`
- mAP50: `-0.003863`
- mAP50-95: `-0.001977`

Constraint result:

- `constraint_failed=false`
- `recall_warning=false`

## Conclusion

Seed1 completed successfully. The run stayed on preserve_original for all nine feedback epochs, did not use sampler-only or weighted sampling, and did not trigger the hard Precision/mAP constraints. Metrics remain close to fixed CATF-v2 seed1, with a small mAP decrease vs fixed but a Precision gain and no constraint failure. Because seed0 and seed2 also pass, a 3-seed image-only preserve-weak summary is now appropriate.
