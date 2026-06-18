# Method Logic For Paper

## Problem With Fixed CATF-v2

Fixed CATF-v2 is an image augmentation method and already shows that diagnosis-driven augmentation can be useful:

- seed0 passes;
- seed1 passes;
- seed2 fails.

The seed2 failure is not evidence that image augmentation is ineffective. It shows that fixed CATF-v2 lacks sufficient risk control. In seed2, high-risk ROI texture augmentation caused mAP degradation and non-active class regression. The method needed a way to keep useful image augmentation while refusing or attenuating risky augmentation.

## Preserve-Weak CATF

Preserve-weak CATF keeps the method inside the image augmentation mainline. It makes a risk-stratified decision for each image augmentation candidate:

| risk level | action | behavior |
|---|---|---|
| low risk | preserve_original | keep the fixed CATF-v2 original image augmentation policy, including class, op, probability, strength, ROI scope, and policy lifetime |
| moderate risk | weak_roi_texture | keep image augmentation but attenuate it with `attenuation_ratio=0.25`; use one lower-risk ROI texture op |
| high or critical risk | strict_noop | do not execute image augmentation for that feedback interval |

This logic avoids the failure mode of weak-only augmentation, where the weak policy globally replaced fixed CATF-v2 and damaged seed0.

## Why This Is Not Sampler-Only

Sampler-only is not part of the main method:

- `sampler_only=false`;
- `weighted_index_list=false`;
- `sampled_distribution_changed=false`;
- no weighted sampling or hard-example mining is used in the main result.

Sampler-only was implemented and evaluated as engineering exploration, but it changes the training sampling distribution. That is not equivalent to image data augmentation and is therefore excluded from the preserve-weak CATF main result.

## Evidence From Seed0

Seed0 fixed CATF-v2 already passed, so the correct behavior is preservation rather than replacement.

The preserve-weak seed0 sanity run shows:

- preserve/weak/noop = `9/0/0`;
- executable classes:
  - epoch5: `[4, 11]`;
  - epoch15: `[12]`;
  - other feedback epochs: empty;
- ROI/industrial = `45/41`, matching fixed CATF-v2 seed0;
- final metrics are effectively identical to fixed CATF-v2 seed0.

This demonstrates class/op/lifetime/volume parity for the preserve_original path.

## Evidence From Seed2

Seed2 fixed CATF-v2 failed due risky image augmentation. Preserve-weak CATF switches the failed path to weak/no-op:

- preserve/weak/noop = `0/5/4`;
- weak augmentation executes only as attenuated image augmentation;
- strict no-op blocks high/critical risk intervals;
- sampler-only remains off.

The result repairs the fixed seed2 failure:

- delta vs fixed seed2: Precision `-0.010446`, Recall `+0.007935`, mAP50 `+0.014518`, mAP50-95 `+0.018438`;
- class9 recovers relative to fixed: Recall `+0.108610`, AP50 `+0.034223`, AP50-95 `+0.009864`;
- non-active mean AP50-95 improves by `+0.018671` relative to fixed when excluding weak-active classes `{6, 7, 9}`.

## Evidence From Seed1

Seed1 fixed CATF-v2 already passed. Preserve-weak CATF keeps it on the preserve_original path:

- preserve/weak/noop = `9/0/0`;
- no weak replacement;
- sampler-only remains off;
- constraint_failed=`false`.

Seed1 confirms that the three-stage logic does not destroy an originally effective fixed CATF-v2 policy.

## Paper Position

The current paper candidate is image-only preserve-weak CATF:

- it remains diagnosis-driven image augmentation;
- it does not alter training sampling;
- it reaches `3/3` hard-constraint pass;
- it improves mean mAP50-95 vs clean by `+0.014467` and vs fixed CATF-v2 by `+0.005501`.
