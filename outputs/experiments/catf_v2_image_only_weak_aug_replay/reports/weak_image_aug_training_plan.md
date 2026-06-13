# Weak Image Augmentation Training Plan

This is a plan only. No training was run in the replay step.

## Run Order

1. Run seed2 image-only weak augmentation 50ep first.
2. Reuse clean seed2 baseline; do not rerun clean.
3. Use fixed CATF-v2 seed2 as the failed image-augmentation comparison.
4. Keep constraints unchanged: Precision, mAP50, and mAP50-95 must not drop by more than 0.01 versus clean.
5. If seed2 passes, run seed0 and seed1 sanity to check that useful image candidates are preserved.
6. Do not use sampler_only, weighted index lists, or training-sampling reweighting.

## Required Method Settings

- Main path: image augmentation only.
- Candidate policies: `candidate_policy_0_noop`, `candidate_policy_1_roi_texture`, `candidate_policy_1b_weak_roi_texture`.
- Disable `candidate_policy_3_sampler_only` for the main method.
- Weak ROI texture should keep one low-risk op, use `attenuation_ratio=0.25`, and cap augmented samples per feedback interval at 16.
- If weak image augmentation fails precision or non-active regression gates, use strict no-op.

## Decision Criteria

- If seed2 passes constraints, proceed to seed0/seed1 sanity.
- If seed2 fails, do not tune sampler_only; inspect image-space candidate risk, attenuation strength, and non-active regression gate behavior.
