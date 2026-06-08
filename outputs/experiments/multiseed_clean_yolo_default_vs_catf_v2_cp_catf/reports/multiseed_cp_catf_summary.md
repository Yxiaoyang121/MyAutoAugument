# Multiseed CP-CATF Training Validation Summary

- Generated at: `2026-06-09T01:42:37`
- Code commit at report generation: `e7b894d93ef23f8f8bdffbf20e4b7ce8127d7aec`
- Run root: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2_cp_catf`
- Constraint failed count: `0/3`
- 3/3 pass: `true`
- RiskGuard used as final rule: `false`
- Dataset/seed/class-specific rule used: `false`
- Development probe uses existing validation diagnostics: `true`

## Metrics

| Seed | Candidate | Action | CP P | CP R | CP mAP50 | CP mAP50-95 | dP vs clean | dR vs clean | dM50 vs clean | dM95 vs clean | dP vs fixed | dR vs fixed | dM50 vs fixed | dM95 vs fixed | constraint_failed |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | `candidate_policy_1_roi_texture` | `accept` | 0.7785 | 0.6697 | 0.7437 | 0.4895 | -0.0060 | -0.0068 | +0.0090 | +0.0136 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | false |
| 1 | `candidate_policy_1_roi_texture` | `accept` | 0.7852 | 0.7005 | 0.7826 | 0.5189 | +0.0127 | +0.0528 | +0.0284 | +0.0390 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | false |
| 2 | `candidate_policy_3_sampler_only` | `sampler_only` | 0.6962 | 0.7286 | 0.7692 | 0.5224 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0674 | +0.0423 | +0.0110 | +0.0257 | false |

## Augmentation and Probe Decisions

| Seed | Image aug allowed | Probe reject image aug | Industrial samples | ROI applied | Router random draws | Active classes | OK3 active | OK3 ROI | Sample weighting |
|---:|---|---|---:|---:|---:|---|---|---:|---|
| 0 | true | false | 41 | 45 | 3050 | `[4, 11, 12]` | false | 0 | `['not_requested']` |
| 1 | true | false | 55 | 56 | 4260 | `[4, 11]` | false | 0 | `['not_requested']` |
| 2 | false | true | 0 | 0 | 0 | `[]` | false | 0 | `['pending_dataloader_support']` |

## Required Answers

1. Seed0 CP-CATF metrics: P=0.7785, R=0.6697, mAP50=0.7437, mAP50-95=0.4895.
2. Seed1 CP-CATF metrics: P=0.7852, R=0.7005, mAP50=0.7826, mAP50-95=0.5189.
3. Seed2 CP-CATF metrics: P=0.6962, R=0.7286, mAP50=0.7692, mAP50-95=0.5224.
4. Constraint pass: `3/3`; 3/3 pass is `true`.
5. Seed0 retained fixed CATF-v2 mAP gain: `true`; mAP50 retention=100.0%, mAP50-95 retention=100.0%.
6. Seed1 retained fixed CATF-v2 gain: `true`; mAP50 retention=100.0%, mAP50-95 retention=100.0%.
7. Seed2 rejected image augmentation and protected clean baseline: `true`; ROI/industrial zero: `true`.
8. OK3 was never active: `true`; OK3 total ROI applied=0.
9. CP-CATF did not rely on dataset-specific rules: `true`; RiskGuard final rule used: `false`.
10. Recommendation: CP-CATF is a final main-method candidate for this development-mode validation because seed0/seed1 retain fixed CATF-v2 behavior and seed2 is protected, but it is not yet a leakage-free paper result.
11. Limitation: current offline probe uses existing validation diagnostics. Paper mode must replace this with a train/probe split or train hard-example probe set before final claims.

## Mean Metrics

| Group | P mean | R mean | mAP50 mean | mAP50-95 mean |
|---|---:|---:|---:|---:|
| clean native | 0.7511 | 0.6843 | 0.7527 | 0.4927 |
| fixed CATF-v2 | 0.7758 | 0.6855 | 0.7615 | 0.5017 |
| CP-CATF | 0.7533 | 0.6996 | 0.7652 | 0.5103 |

- Mean CP-CATF delta vs clean: dP=+0.0022, dR=+0.0153, dM50=+0.0125, dM95=+0.0175.
- Mean CP-CATF delta vs fixed CATF-v2: dP=-0.0225, dR=+0.0141, dM50=+0.0037, dM95=+0.0086.
