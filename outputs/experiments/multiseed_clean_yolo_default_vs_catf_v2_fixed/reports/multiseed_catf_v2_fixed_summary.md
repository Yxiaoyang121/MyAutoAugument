# Fixed CATF-v2 Multiseed Summary

Code commit used: `dfcd177fa058046073e9b8e87dc8052b7b705f9a`.

This summary reuses clean native YOLO default seed0/1/2 and the fixed CATF-v2 seed1 rerun, and adds fixed CATF-v2 seed0/2 after the strict no-augmentation bypass fix.

## Per-Seed Metrics

| Seed | Clean P | Clean R | Clean mAP50 | Clean mAP50-95 | Fixed P | Fixed R | Fixed mAP50 | Fixed mAP50-95 | ?P | ?R | ?mAP50 | ?mAP50-95 | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | -0.0060 | -0.0068 | +0.0090 | +0.0136 | False |
| 1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | +0.0127 | +0.0528 | +0.0284 | +0.0390 | False |
| 2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | +0.0674 | -0.0423 | -0.0110 | -0.0257 | True |

## Mean ? Std

| Group | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean native | 0.7511 ? 0.0479 | 0.6843 ? 0.0410 | 0.7527 ? 0.0173 | 0.4927 ? 0.0258 |
| fixed CATF-v2 | 0.7758 ? 0.0110 | 0.6855 ? 0.0154 | 0.7615 ? 0.0197 | 0.5017 ? 0.0153 |
| old CATF-v2 | 0.7826 ? 0.0479 | 0.6657 ? 0.0536 | 0.7514 ? 0.0080 | 0.4953 ? 0.0063 |
| fixed delta vs clean | 0.0247 ? 0.0382 | 0.0012 ? 0.0481 | 0.0088 ? 0.0197 | 0.0090 ? 0.0326 |
| old delta vs clean | 0.0315 ? 0.0955 | -0.0185 ? 0.0929 | -0.0013 ? 0.0247 | 0.0026 ? 0.0282 |

## Constraint Summary

- Fixed CATF-v2 constraint_failed count: `1/3`.
- Old CATF-v2 constraint_failed count: `2/3`.
- Fixed CATF-v2 constraint pass count: `2/3`.
- Fixed CATF-v2 all-metrics-improved seed count: `1/3`.
- Seed0 now passes constraints but Recall is slightly lower than clean native; mAP50 and mAP50-95 improve.
- Seed1 improves all four metrics and passes constraints.
- Seed2 still fails constraints because mAP50 and mAP50-95 drop beyond the allowed threshold, although Precision rises.

## OK3 / Activation Safety

- OK3 ever active: `False`.
- OK3 ROI applied total: `0`.
- Active classes across fixed runs: `{'11': 3, '4': 2, '12': 1, '9': 1, '8': 1}`.
- ROI affected classes across fixed runs: `{'4': 14, '11': 132, '12': 14, '8': 6, '9': 25}`.

## Per-Seed Augmentation Activity

| Seed | industrial samples augmented | ROI applied | ROI affected classes | active classes | policy actions |
|---:|---:|---:|---|---|---|
| 0 | 41 | 45 | `{'4': 9, '11': 22, '12': 14}` | `{'11': 1, '4': 1, '12': 1}` | `{'accept': 2, 'rollback': 1, 'shrink': 2, 'observe': 2, 'freeze': 2}` |
| 1 | 55 | 56 | `{'4': 5, '11': 51}` | `{'11': 1, '4': 1}` | `{'accept': 1, 'observe': 1, 'shrink': 5, 'freeze': 2}` |
| 2 | 86 | 90 | `{'8': 6, '9': 25, '11': 59}` | `{'9': 1, '11': 1, '8': 1}` | `{'accept': 3, 'shrink': 4, 'freeze': 2}` |

## Compared With Old CATF-v2

- Old CATF-v2 multiseed had constraint_failed `2/3` with seed0 and seed2 failing.
- Fixed CATF-v2 multiseed has constraint_failed `1/3`; seed0 moved from failed to passed after the strict bypass repair and rerun.
- The fixed seed1 rerun includes real ROI/industrial augmentation (`samples_augmented=55`, `ROI applied=56`), so its improvement is no longer attributable to formal no-op label rewriting.
- Fixed CATF-v2 improves the multiseed mean mAP50-95 relative to clean native, but seed2 remains unstable.

## Recommendation

Do not claim CATF-v2 as a fully stable sole main method yet. It is a stronger candidate after strict bypass repair, with 2/3 seeds passing constraints and real ROI/industrial activity, but seed2 still fails the industrial constraint. Position it as a promising constrained feedback augmentation method and continue with seed2 failure analysis or threshold-calibrated variant before making it the final?????.

## Reported Paths

- Seed 0 fixed CATF-v2: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/seed_0/catf_v2`
- Seed 1 fixed CATF-v2: `outputs/experiments/catf_v2_fixed_seed1_50ep`
- Seed 2 fixed CATF-v2: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/seed_2/catf_v2`
