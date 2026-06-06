# CATF-v2 Adaptive Burn-in + RB Seed2 50ep Report

Generated: `2026-06-06T18:00:12`
Code commit used: `671a3e8f164dac4eff76193ba291004926667a4a`

## Required Answers

1. Seed2 adaptive start epoch: `None`.
2. CATF candidate started: `false`.
3. Rollback triggered: `false`.
4. Strict no-op: `true`; no-op fallback epoch: `15`.
5. Final metrics: P `0.6962`, R `0.7286`, mAP50 `0.7692`, mAP50-95 `0.5224`.
6. constraint_failed: `false`; reasons: `[]`.
7. Adaptive burn-in is more defensible than fixed epoch5 here because it waits for model/metric diagnosability and blocks intervention under strong seed2 clean-baseline protection.
8. Full multiseed adaptive-RB recommended: `true`.

## Metrics

| group | P | R | mAP50 | mAP50-95 | dP vs clean | dR vs clean | dM50 vs clean | dM95 vs clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| fixed CATF-v2 seed2 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | +0.0674 | -0.0423 | -0.0110 | -0.0257 |
| Safe seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Gated seed2 | 0.7850 | 0.6795 | 0.7521 | 0.5083 | +0.0887 | -0.0491 | -0.0171 | -0.0141 |
| Adaptive-RB seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Augmentation Audit

- Industrial samples augmented: `0`
- ROI applied: `0`
- Router random draw count: `0`
- Train images: `2301`
- BBox/class legal: `true`
- Epoch continuous: `true`

## Burn-in Events

| epoch | action | start_condition | reasons | map50_range | recall_range | eligible_classes | strong_baseline |
|---:|---|---|---|---:|---:|---|---|
| 5 | burnin_observe | false | `['metric_unstable', 'strong_clean_baseline_protection']` | 0.0744 | 0.1673 | `[9]` | `true` |
| 10 | burnin_observe | false | `['metric_unstable', 'strong_clean_baseline_protection']` | 0.1045 | 0.0886 | `[12]` | `true` |
| 15 | no_op_fallback | false | `['adaptive_burnin_not_ready']` | 0.0397 | 0.0390 | `[12]` | `true` |
| 20 | no_op_fallback | false | `['adaptive_burnin_not_ready']` | 0.0267 | 0.0702 | `[]` | `true` |
| 25 | no_op_fallback | false | `['adaptive_burnin_not_ready']` | 0.0664 | 0.0272 | `[]` | `true` |
| 30 | no_op_fallback | false | `['adaptive_burnin_not_ready']` | 0.0154 | 0.0647 | `[]` | `true` |
| 35 | no_op_fallback | false | `['adaptive_burnin_not_ready']` | 0.0631 | 0.0812 | `[8, 11]` | `true` |
| 40 | no_op_fallback | false | `['adaptive_burnin_not_ready']` | 0.0121 | 0.0215 | `[]` | `true` |
| 45 | no_op_fallback | false | `['adaptive_burnin_not_ready']` | 0.0619 | 0.0192 | `[]` | `true` |
