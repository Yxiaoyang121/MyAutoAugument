# Full Multiseed CATF-v2 Adaptive Burn-in + RB Summary

Generated: `2026-06-07T02:48:43`
Code commit used: `f1c4e5606a8fb2db7073a6b3128c875d3aea3631`

## Metrics

| Seed | Clean P | Clean R | Clean mAP50 | Clean mAP50-95 | Fixed P | Fixed R | Fixed mAP50 | Fixed mAP50-95 | Adaptive-RB P | Adaptive-RB R | Adaptive-RB mAP50 | Adaptive-RB mAP50-95 | dP vs clean | dR vs clean | dM50 vs clean | dM95 vs clean | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | false |
| 1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | 0.7550 | 0.7261 | 0.7653 | 0.4938 | -0.0175 | +0.0784 | +0.0111 | +0.0139 | true |
| 2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | false |

## Gate And Augmentation Audit

| Seed | adaptive start | candidate | rollback | accepted | no-op fallback | no-op epoch | industrial samples | ROI applied | router draws | active classes | OK3 active | OK3 ROI |
|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---|---:|
| 0 | None | false | false | false | true | 15 | 0 | 0 | 0 | `[]` | false | 0 |
| 1 | 15 | true | false | true | false | None | 17 | 20 | 1430 | `[12]` | false | 0 |
| 2 | None | false | false | false | true | 15 | 0 | 0 | 0 | `[]` | false | 0 |

## Required Answers

- Constraint pass count: `2/3`; achieved 3/3: `false`.
- Seed0 retained fixed mAP gains: `false`. It fell back to clean/no-op and lost fixed CATF-v2 mAP50/mAP50-95 gains.
- Seed1 retained fixed gains: `false`. It retained part of the recall/mAP lift, but failed the precision constraint.
- Seed2 clean no-op fallback: `true` at epoch `15`.
- OK3 ever active: `false`; OK3 ROI applied total: `0`.
- Recommended as paper main method: `false`.
- Position: Adaptive-RB is not yet a final paper main method: it protects seed2, but seed0 falls back to clean and seed1 violates the precision-drop constraint despite recall/mAP gains.

## Comparison With Fixed / Safe / Gated

- Versus fixed CATF-v2: adaptive-RB protects seed2 through strict no-op, but does not preserve seed0 gains and seed1 fails the precision constraint.
- Versus Safe: adaptive-RB is less conservative on seed1 and recovers recall/mAP gains, but Safe remains better on constraint pass count in this run.
- Versus Gated: adaptive-RB fixes the seed2 non-rollback problem by preventing candidate start, but its low-risk seed1 candidate still needs a stricter precision-aware accept gate.

## Gain Retention

| Seed | dM50 fixed vs clean | dM95 fixed vs clean | dM50 adaptive vs clean | dM95 adaptive vs clean | mAP50 retention | mAP95 retention |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.0090 | +0.0136 | +0.0000 | +0.0000 | 0.0% | 0.0% |
| 1 | +0.0284 | +0.0390 | +0.0111 | +0.0139 | 39.1% | 35.6% |
| 2 | -0.0110 | -0.0257 | +0.0000 | +0.0000 | n/a | n/a |

## Next Parameter Adjustments

- Use cumulative burn-in evidence across epoch 5/10/15 instead of only the current diagnosis so seed0 is not lost when epoch15 evidence thins out.
- Make the RB probe gate compare against clean/reference constraints, not only the immediate probe reference, so seed1 precision_drop_gt_0.01 triggers rollback or shrink.
- Require a precision floor or threshold-calibration step before accepting a low-risk candidate with recall/mAP gains.
- Keep strong clean baseline protection for seed2 unchanged unless later seeds show false abstention under clear high-confidence issues.
