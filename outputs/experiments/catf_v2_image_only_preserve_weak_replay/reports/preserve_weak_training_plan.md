# Preserve-Original + Weak-Only Training Plan

Do not execute this plan in the replay task. This is the next-step plan only.

## Order

1. Run seed0 sanity first.
   - Reason: fixed seed0 passed and global weak seed0 failed.
   - Requirement: preserve_original must keep class4/11/12 fixed behavior and must not execute weak class9 replacement.
   - Constraint: Precision, mAP50, and mAP50-95 drops vs clean must each stay within 0.01.
2. Run seed2 second.
   - Reason: fixed seed2 failed, while weak image augmentation repaired seed2.
   - Requirement: high-risk fixed image policy must convert to weak/no-op and remain image-only.
3. Run seed1 sanity last.
   - Reason: fixed seed1 already passed strongly; preserve_original should avoid damaging the gain.
4. Run three-seed validation only if seed0 and seed2 pass their single-seed checks.

## Fixed Conditions

- image-only CATF mainline;
- no sampler_only;
- no weighted index list;
- no sampling distribution change;
- attenuation_ratio remains 0.25 for weak candidates;
- no data split change;
- no final-val policy selection;
- no claim of final method until 3/3 pass.
