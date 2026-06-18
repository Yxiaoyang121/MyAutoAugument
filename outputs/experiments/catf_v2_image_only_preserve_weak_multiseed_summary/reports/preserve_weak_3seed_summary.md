# Preserve-Weak Image CATF 3-Seed Summary

## Scope

This report combines completed seed0, seed1, and seed2 image-only preserve-weak CATF results. No seed0 or seed2 rerun was performed for this summary, and no multiseed training was launched.

Inputs:

- seed0: `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_volume_fixed/`
- seed1: `outputs/experiments/catf_v2_image_only_preserve_weak_seed1_sanity/`
- seed2: `outputs/experiments/catf_v2_image_only_preserve_weak_seed2/`

Method:

- low risk: preserve fixed CATF-v2 original image augmentation policy;
- moderate risk: weak ROI texture image augmentation with attenuation ratio `0.25`;
- high/critical risk: strict no-op;
- sampler_only: never used.

## Per-Seed Results

| seed | method | Precision | Recall | mAP50 | mAP50-95 | constraint_failed | recall_warning |
|---:|---|---:|---:|---:|---:|---|---|
| 0 | clean | 0.784600 | 0.676500 | 0.734700 | 0.475900 | n/a | n/a |
| 0 | fixed CATF-v2 | 0.778500 | 0.669700 | 0.743700 | 0.489500 | false | n/a |
| 0 | preserve-weak | 0.778506 | 0.669654 | 0.743657 | 0.489541 | false | false |
| 1 | clean | 0.772500 | 0.647700 | 0.754200 | 0.479900 | n/a | n/a |
| 1 | fixed CATF-v2 | 0.785200 | 0.700500 | 0.782600 | 0.518900 | false | n/a |
| 1 | preserve-weak | 0.799748 | 0.697375 | 0.778737 | 0.516923 | false | false |
| 2 | clean | 0.696200 | 0.728600 | 0.769200 | 0.522400 | n/a | n/a |
| 2 | fixed CATF-v2 | 0.763700 | 0.686300 | 0.758200 | 0.496700 | true | n/a |
| 2 | preserve-weak | 0.753254 | 0.694235 | 0.772718 | 0.515138 | false | true |

## Deltas

Delta vs clean:

| seed | dP | dR | dmAP50 | dmAP50-95 |
|---:|---:|---:|---:|---:|
| 0 | -0.006094 | -0.006846 | +0.008957 | +0.013641 |
| 1 | +0.027248 | +0.049675 | +0.024537 | +0.037023 |
| 2 | +0.057054 | -0.034365 | +0.003518 | -0.007262 |

Delta vs fixed CATF-v2:

| seed | dP | dR | dmAP50 | dmAP50-95 |
|---:|---:|---:|---:|---:|
| 0 | +0.000006 | -0.000046 | -0.000043 | +0.000041 |
| 1 | +0.014548 | -0.003125 | -0.003863 | -0.001977 |
| 2 | -0.010446 | +0.007935 | +0.014518 | +0.018438 |

## Aggregate

- 3-seed pass count: `3/3`
- Mean preserve-weak P/R/mAP50/mAP50-95: `0.777170 / 0.687088 / 0.765037 / 0.507201`
- Mean delta vs clean: `+0.026070 / +0.002821 / +0.012337 / +0.014467`
- Mean delta vs fixed CATF-v2: `+0.001370 / +0.001588 / +0.003537 / +0.005501`
- Total industrial images augmented: `199`
- Total ROI applied: `220`

## Method Checks

- seed0 reproduced fixed CATF-v2 original policy: `true`
- seed1 preserved fixed CATF-v2 original policy: `true`
- seed2 fixed failure repaired through weak/no-op image control: `true`
- sampler_only involved: `false`
- weighted index list enabled: `false`
- sampled distribution changed: `false`

## Interpretation

The image-only preserve-weak CATF policy reaches `3/3` hard-constraint pass on the current seeds. It preserves seed0's known-safe fixed policy, keeps seed1 near the fixed CATF-v2 pass result, and repairs seed2's fixed CATF-v2 failure by replacing high-risk image augmentation with weak image augmentation or strict no-op.

This is a viable image-only CATF main-method candidate. The remaining limitation is Recall: seed2 Recall is still below clean by `0.034365`, even though Precision, mAP50, and mAP50-95 satisfy the hard constraints. This should be described as a recall-side limitation, and a future recall-aware image augmentation constraint may be needed.
