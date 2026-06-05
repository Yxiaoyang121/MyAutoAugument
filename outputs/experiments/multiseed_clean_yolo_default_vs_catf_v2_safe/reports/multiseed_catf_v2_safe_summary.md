# CATF-v2-Safe Multiseed Summary

Generated: `2026-06-05T16:19:19`
Code commit used: `6dfc9bcf45e4d9989321330b469bce6f025483f8`.

## Per-Seed Metrics

| Seed | Clean P | Clean R | Clean mAP50 | Clean mAP50-95 | Fixed P | Fixed R | Fixed mAP50 | Fixed mAP50-95 | Safe P | Safe R | Safe mAP50 | Safe mAP50-95 | Safe dP vs clean | Safe dR vs clean | Safe dM50 vs clean | Safe dM95 vs clean | Safe dP vs fixed | Safe dR vs fixed | Safe dM50 vs fixed | Safe dM95 vs fixed | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0060 | +0.0068 | -0.0090 | -0.0136 | false |
| 1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0127 | -0.0528 | -0.0284 | -0.0390 | false |
| 2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0674 | +0.0423 | +0.0110 | +0.0257 | false |

## Constraint Summary

- CATF-v2-Safe constraint_failed count: `0/3`.
- Fixed CATF-v2 constraint_failed count: `1/3`.
- CATF-v2-Safe achieved 3/3 pass: `true`.
- All Safe runs have results.csv epoch 1..50 continuous: `true`.

## Safe Controller Behavior

| Seed | baseline protection | early abstention | no-op fallback | first no-op epoch | industrial samples | router draws | ROI applied | active classes | OK3 active | OK3 ROI |
|---:|---|---|---|---:|---:|---:|---:|---|---|---:|
| 0 | false | true | true | 5 | 0 | 0 | 0 | `{'11': 1, '4': 1}` | false | 0 |
| 1 | false | true | true | 5 | 0 | 0 | 0 | `{'11': 1, '4': 1}` | false | 0 |
| 2 | false | true | true | 5 | 0 | 0 | 0 | `{'9': 1}` | false | 0 |

## OK3 / No-Aug Safety

- OK3 ever active: `false`.
- OK3 ROI applied total: `0`.
- OK2/OK3 remain no-augmentation classes in the CATF-v2 policy matrix.

## Required Answers

- Seed0 retained fixed CATF-v2 gain: `false`. Safe passed constraints but no-op abstention removed fixed seed0 mAP gains.
- Seed1 retained fixed CATF-v2 gain: `false`.
- Seed2 no-op fallback and clean parity: `true`.
- Safe recommended as paper main method: `false`.

## Paper Wording

CATF-v2-Safe passes all three seeds under the industrial constraints by using an early-abstention no-op fallback when the feedback signal shows no recall or mAP gain. However, in this validation it also abstains on seed0 and seed1, eliminating the gains of fixed CATF-v2. It should be positioned as a conservative safety/protection variant or fallback layer, not as the sole main augmentation method.

## Reported Paths

- Seed 0 Safe: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2_safe\seed_0\catf_v2_safe`
- Seed 1 Safe: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2_safe\seed_1\catf_v2_safe`
- Seed 2 Safe: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2_safe\seed_2\catf_v2_safe`
