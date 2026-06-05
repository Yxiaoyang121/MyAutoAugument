# CATF-v2-Gated Multiseed Summary

Generated: `2026-06-06T06:49:19`
Code commit used: `ff77080a1046f3bdb05da10b5c1904c803380491`.

## Per-Seed Metrics

| Seed | Clean P | Clean R | Clean mAP50 | Clean mAP50-95 | Fixed P | Fixed R | Fixed mAP50 | Fixed mAP50-95 | Safe P | Safe R | Safe mAP50 | Safe mAP50-95 | Gated P | Gated R | Gated mAP50 | Gated mAP50-95 | Gated dP vs clean | Gated dR vs clean | Gated dM50 vs clean | Gated dM95 vs clean | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | -0.0060 | -0.0068 | +0.0090 | +0.0136 | false |
| 1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | +0.0127 | +0.0528 | +0.0284 | +0.0390 | false |
| 2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7850 | 0.6795 | 0.7521 | 0.5083 | +0.0887 | -0.0491 | -0.0171 | -0.0141 | true |

## Constraint Summary

- CATF-v2-Gated constraint_failed count: `1/3`.
- CATF-v2-Gated achieved 3/3 pass: `false`.
- Retrospective expected 3/3 pass: `true`.
- Retrospective vs actual mismatch: `true`.
- All Gated runs have results.csv epoch 1..50 continuous: `true`.

## Gated Controller Behavior

| Seed | fallback | fallback epoch | strict no-op after fallback | industrial samples | router draws | ROI applied | active classes | OK3 active | OK3 ROI | reasons |
|---:|---|---:|---|---:|---:|---:|---|---|---:|---|
| 0 | false | None | true | 41 | 3050 | 45 | `{'11': 1, '4': 1, '12': 1}` | false | 0 | `{}` |
| 1 | false | None | true | 55 | 4260 | 56 | `{'11': 1, '4': 1}` | false | 0 | `{}` |
| 2 | true | 10 | true | 22 | 1540 | 25 | `{'9': 1}` | false | 0 | `{'epoch10_bad_pattern_A': 8}` |

## Required Answers

- Seed0 retained fixed CATF-v2 mAP gain: `true`.
- Seed1 retained fixed CATF-v2 clear gain: `true`.
- Seed2 triggered fallback: `true` at epoch `10`.
- Seed2 passed constraints after fallback: `false`.
- OK3 ever active: `false`.
- OK3 ROI applied total: `0`.
- CATF-v2-Gated recommended as paper main method: `false`.

## Interpretation

CATF-v2-Gated retains fixed CATF-v2 gains on seed0 and seed1, but it is not sufficient as the paper main method because seed2 still violates the industrial constraints after an epoch10 fallback. The result shows that strict no-op after fallback is not equivalent to clean fallback unless the damaging tentative updates are prevented or rewound.

## Reported Paths

- Seed 0 Gated: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2_gated\seed_0\catf_v2_gated`
- Seed 1 Gated: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2_gated\seed_1\catf_v2_gated`
- Seed 2 Gated: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2_gated\seed_2\catf_v2_gated`
