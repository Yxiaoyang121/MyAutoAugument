# Preserve Original Volume / Lifetime Audit

## Root Cause

preserve_original inflated seed0 augmentation volume because the runtime schedule used cumulative replay active classes instead of epoch-exact fixed CATF-v2 router-executable policies. Fixed seed0 retained nonzero old ops in some guarded/frozen class policies, but those rows were not router-executable. The executable fixed policy existed at epoch 5 (classes 4/11) and epoch 15 (class 12). The preserve rerun cleared guards and kept class4 active through epoch45 and class12 active from epoch15 through epoch45.

This made industrial augmentation grow from 41 to 155 and ROI augmentation grow from 45 to 187. The op probabilities and strengths were not numerically amplified; they were applied for too many feedback intervals.

## Findings

- seed-level/cumulative union present before fix: True
- stale policy accumulation before fix: True
- policy lifetime too long before fix: True
- router eligibility amplified before fix: True
- op prob/strength amplified: False
- sampler_only involved: False

## Fix

- build_preserve_weak_decision_schedule now can load fixed CATF-v2 policy_history and emits epoch_exact_fixed_active_class/op/prob_strength.
- preserve_original execution now prefers epoch-exact fixed fields and treats an explicit empty epoch as no active policy, instead of falling back to probe_set.class_id or a replay class union.
- each feedback update clears stale active ops before installing only that epoch's fixed policy.
- the corrected schedule replaced the old runtime config; the stale config was backed up for audit.

## Dry-Run After Fix

- fixed industrial expected: 41
- preserve industrial expected after fix: 41
- fixed ROI expected: 45
- preserve ROI expected after fix: 45
- old industrial volume ratio: 3.780488
- old ROI volume ratio: 4.155556
- post-fix industrial volume ratio: 1.0
- post-fix ROI volume ratio: 1.0
- epoch-exact preserve after fix: True
- stale accumulation after fix: False

## Recommendation

Rerun seed0 sanity before seed2. The dry-run now matches fixed CATF-v2 policy lifetime and should avoid the previous augmentation-volume overrun.
