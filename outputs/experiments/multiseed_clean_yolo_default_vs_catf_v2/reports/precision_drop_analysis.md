# CATF-v2 Precision Drop Analysis

Generated: `2026-06-01T13:08:26`

## Seed 0

- Global delta: `{'precision': -0.0417, 'recall': 0.0218, 'map50': 0.0223, 'map50_95': 0.0264}`.
- Top approximate FP increases:

| class | ΔFP | ΔPrecision | ΔRecall | active? | threshold recommendation |
|---|---:|---:|---:|:---:|---|
| 4:油污 | +10.0605 | -0.1955 | +0.1667 | false | 0.25 -> 0.4 high_fp_raise_threshold |
| 11:锡尖 | +3.6723 | -0.1281 | +0.1526 | true | keep |
| 6:漏背锡 | +3.0889 | -0.0521 | +0.0264 | false | 0.25 -> 0.4 high_fp_raise_threshold |
| 9:轮廓划伤 | +2.5208 | -0.0360 | -0.0119 | false | 0.25 -> 0.4 high_fp_raise_threshold |
| 3:开裂 | +1.6259 | -0.1810 | +0.0000 | false | 0.25 -> 0.4 high_fp_raise_threshold |
| 7:碰伤 | +1.0003 | -0.0189 | -0.0459 | false | 0.25 -> 0.4 high_fp_raise_threshold |
| 1:OK3 | +0.4756 | -0.0017 | +0.0038 | false | 0.25 -> 0.4 high_fp_raise_threshold |
| 10:锡丝残留 | +0.3256 | -0.0012 | +0.1001 | false | 0.25 -> 0.4 high_fp_raise_threshold |

## Seed 1

- Global Precision delta is `-0.0034`, within constraint.
- Seed 1 had no active ROI augmentation, so the slight Precision drop is normal seed/training variance rather than evidence of harmful CATF-v2 augmentation.

## Answers

- Precision drop mainly comes from classes with increased approximate FP, not only the active class.
- high-FP prior covered OK/oil/dirty classes, but final FP risk also appears in non-prior defect classes, so prior coverage is incomplete.
- Existing threshold reports suggest raising many high-FP classes to `0.40`; seed0 is the strongest candidate for threshold repair.
- Do not add OK2/OK3 back into augmentation. Keep them no_aug/high-FP guarded.
