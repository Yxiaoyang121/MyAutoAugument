# Preserve-Weak Image CATF Paper-Ready Summary

## Method Status

Preserve-weak CATF is an image-only automatic data augmentation method. It does not use sampler-only, weighted index lists, or sampling distribution changes.

- sampler_only: `false`
- weighted index list: `false`
- sampled distribution changed: `false`
- hard-constraint pass count: `3/3`

## Results

| seed | clean P/R/mAP50/mAP50-95 | fixed CATF-v2 P/R/mAP50/mAP50-95 | preserve-weak P/R/mAP50/mAP50-95 | constraint_failed | recall_warning |
|---:|---|---|---|---|---|
| 0 | 0.784600 / 0.676500 / 0.734700 / 0.475900 | 0.778500 / 0.669700 / 0.743700 / 0.489500 | 0.778506 / 0.669654 / 0.743657 / 0.489541 | false | false |
| 1 | 0.772500 / 0.647700 / 0.754200 / 0.479900 | 0.785200 / 0.700500 / 0.782600 / 0.518900 | 0.799748 / 0.697375 / 0.778737 / 0.516923 | false | false |
| 2 | 0.696200 / 0.728600 / 0.769200 / 0.522400 | 0.763700 / 0.686300 / 0.758200 / 0.496700 | 0.753254 / 0.694235 / 0.772718 / 0.515138 | false | true |

Mean preserve-weak metrics:

- Precision: `0.777170`
- Recall: `0.687088`
- mAP50: `0.765037`
- mAP50-95: `0.507201`

Mean delta vs clean:

- Precision: `+0.026070`
- Recall: `+0.002821`
- mAP50: `+0.012337`
- mAP50-95: `+0.014467`

Mean delta vs fixed CATF-v2:

- Precision: `+0.001370`
- Recall: `+0.001588`
- mAP50: `+0.003537`
- mAP50-95: `+0.005501`

## Evidence Chain

Fixed CATF-v2:

- seed0 and seed1 pass;
- seed2 fails due high-risk image augmentation and non-active regression;
- this shows image augmentation can help, but needs risk control.

Preserve-weak CATF:

- seed0 preserves the known-safe fixed policy and reproduces fixed behavior;
- seed1 preserves the fixed policy without hard-constraint failure;
- seed2 routes moderate-risk candidates to weak image augmentation and high/critical candidates to strict no-op.

## Main Claim Candidate

Preserve-weak image-only CATF is the current main-method candidate:

- it is still image data augmentation;
- it does not alter training sample distribution;
- it passes hard constraints on all three seeds;
- it improves mean mAP50-95 vs clean by `+0.014467`;
- it improves mean mAP50-95 vs fixed CATF-v2 by `+0.005501`.

## Limitation

Seed2 Recall remains below clean by `0.034365`. This is marked as `recall_warning=true`. It does not violate the current hard constraints, but should be discussed as a limitation and motivation for recall-aware image augmentation constraints.

## Recommended Next Step

Do not continue blind training. The next method work should be a targeted, image-only recall-aware extension if the paper needs to address the seed2 Recall warning.
