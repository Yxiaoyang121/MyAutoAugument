# Reviewer Risk Check

## Did The Method Change Sampling?

No.

For the preserve-weak image-only CATF main result:

- `sampler_only=false`;
- `weighted_index_list=false`;
- `sampled_distribution_changed=false`.

The training distribution was not reweighted for the main result.

## Is This Hard Example Mining Disguised As Data Augmentation?

No.

Sampler-only was explored separately and then demoted to engineering exploration or ablation. It is not part of the main method table. The preserve-weak main result uses image augmentation decisions only.

## Where Does The Improvement Come From?

The improvement comes from risk-aware image augmentation policy selection:

- preserve safe original augmentation when fixed CATF-v2 is already low risk;
- attenuate moderate-risk image augmentation rather than applying full-strength ROI texture augmentation;
- use strict no-op for high or critical risk augmentation candidates.

This keeps beneficial image augmentation while reducing the chance of non-active regression.

## Is The Method Hardcoded To Seed2 Or A Specific Class?

No.

The rules are based on risk level, not seed ID or class name:

- seed0 automatically goes to preserve_original because its fixed policy is low risk and already effective;
- seed1 automatically goes to preserve_original for the same reason;
- seed2 automatically goes to weak/no-op because its fixed policy is risky.

Class IDs appear in audit outputs because those are the observed active classes in this dataset, not because the method contains seed-specific or class-specific hardcoding.

## How Should Seed2 Recall Drop Be Explained?

Seed2 Recall remains below clean by `0.034365`. This is reported as `recall_warning=true`.

The current hard constraints prioritize avoiding Precision, mAP50, and mAP50-95 regression. The Recall warning is a limitation and motivates future recall-aware image augmentation constraints.

## What Should Not Be Claimed?

Do not claim that this is a final fully optimized paper result. The correct claim is that preserve-weak image-only CATF is the current paper main-method candidate with `3/3` hard-constraint pass and a documented seed2 Recall warning.
